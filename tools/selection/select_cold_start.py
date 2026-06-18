#!/usr/bin/env python
"""Run detector-free cold-start selection from DINOv3 image features."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from selection.algorithms import (  # noqa: E402
    kcenter_greedy,
    random_indices,
    select_cluster_representatives,
    select_tail_by_seed_clusters,
)
from selection.coco import (  # noqa: E402
    class_counts,
    coco_image_keys,
    image_key,
    load_coco,
    subset_coco,
    target_counts_by_image_key,
    write_json,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--features", type=Path, required=True)
    parser.add_argument("--paths", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/selection"))
    parser.add_argument(
        "--method",
        choices=["cluster_head", "cluster_tail", "kcenter", "random"],
        default="cluster_tail",
    )
    parser.add_argument("--budget", type=int, default=608)
    parser.add_argument("--seed", type=int, default=43)
    parser.add_argument("--key-mode", choices=["basename", "relative"], default="basename")

    parser.add_argument("--candidate-coco", type=Path, help="Optional COCO file limiting the candidate pool.")
    parser.add_argument("--exclude-coco", type=Path, action="append", default=[], help="COCO files excluded from selection.")
    parser.add_argument("--output-coco-from", type=Path, help="COCO annotations used to write the selected labeled subset.")

    parser.add_argument("--kcenter-init", choices=["medoid", "random", "first"], default="medoid")
    parser.add_argument("--head-budget", type=int, default=500)
    parser.add_argument("--n-clusters", type=int, default=50)
    parser.add_argument("--per-cluster", type=int, default=10)
    parser.add_argument("--kmeans-n-init", type=int, default=10)

    parser.add_argument("--seed-coco", type=Path, help="Labels for the already selected coverage head.")
    parser.add_argument("--target-class-id", type=int, default=0)
    parser.add_argument("--target-box-count", type=int, default=90)
    parser.add_argument("--tail-budget", type=int, default=0, help="0 means fill the remaining budget.")
    parser.add_argument("--tail-cluster-count", type=int, default=3)
    parser.add_argument("--tail-safety-factor", type=float, default=1.4)
    return parser.parse_args()


def load_feature_table(features_path: Path, paths_path: Path, key_mode: str) -> tuple[np.ndarray, list[dict]]:
    features = np.load(features_path).astype(np.float32, copy=False)
    paths = np.load(paths_path, allow_pickle=True)
    if features.ndim != 2:
        raise ValueError(f"Expected 2-D features, got {features.shape}.")
    if len(paths) != len(features):
        raise ValueError(f"Feature/path length mismatch: {len(features)} vs {len(paths)}.")

    records: list[dict] = []
    seen = set()
    for global_idx, raw_path in enumerate(paths):
        text = str(raw_path.item() if hasattr(raw_path, "item") else raw_path)
        key = image_key(text, key_mode)
        if key in seen:
            raise ValueError(f"Duplicate image key in feature paths: {key}")
        seen.add(key)
        records.append({"global_idx": global_idx, "path": text, "key": key})
    return features, records


def filter_candidates(
    records: list[dict],
    candidate_coco: Path | None,
    exclude_coco: list[Path],
    key_mode: str,
) -> list[int]:
    feature_keys = {record["key"] for record in records}
    allowed = set(feature_keys)
    if candidate_coco:
        requested = coco_image_keys(load_coco(candidate_coco), key_mode=key_mode)
        missing = sorted(requested - feature_keys)
        if missing:
            raise ValueError(
                f"Candidate COCO has {len(missing)} images without extracted features: {missing[:5]}"
            )
        allowed &= requested
    for path in exclude_coco:
        allowed -= coco_image_keys(load_coco(path), key_mode=key_mode)
    selected = [idx for idx, record in enumerate(records) if record["key"] in allowed]
    if not selected:
        raise ValueError("No candidate images remain after filtering.")
    return selected


def write_selection_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "rank",
        "stage",
        "file_name",
        "path",
        "global_idx",
        "cluster_id",
        "selected_rank_in_cluster",
        "distance_to_cluster_center",
        "distance_to_seed_centroid",
        "score_before_selection",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    features_all, records_all = load_feature_table(args.features, args.paths, args.key_mode)
    candidate_rows = filter_candidates(records_all, args.candidate_coco, args.exclude_coco, args.key_mode)
    features = features_all[np.array(candidate_rows, dtype=int)]
    records = [records_all[idx] for idx in candidate_rows]

    method_rows: dict[int, dict] = {}
    candidate_ranking: list[dict] = []
    manifest: dict = {
        "method": args.method,
        "budget": args.budget,
        "seed": args.seed,
        "candidate_count": len(records),
        "feature_shape": list(features.shape),
        "features": str(args.features),
        "paths": str(args.paths),
        "key_mode": args.key_mode,
    }

    if args.method == "random":
        selected_local = random_indices(len(records), args.budget, args.seed)
        for idx in selected_local:
            method_rows[idx] = {"stage": "random"}

    elif args.method == "kcenter":
        selected_local, scores, nearest = kcenter_greedy(
            features,
            args.budget,
            init=args.kcenter_init,
            seed=args.seed,
        )
        for idx, score in zip(selected_local, scores):
            method_rows[idx] = {"stage": "kcenter", "score_before_selection": score}
        manifest["kcenter"] = {
            "init": args.kcenter_init,
            "final_radius": float(np.max(nearest)),
        }

    else:
        if args.head_budget > args.budget:
            raise ValueError("--head-budget must be <= --budget.")
        if args.head_budget != args.n_clusters * args.per_cluster:
            raise ValueError("--head-budget must equal --n-clusters * --per-cluster.")

        head_local, cluster_labels, head_rows = select_cluster_representatives(
            features,
            n_clusters=args.n_clusters,
            per_cluster=args.per_cluster,
            seed=args.seed,
            n_init=args.kmeans_n_init,
        )
        for idx, row in zip(head_local, head_rows):
            method_rows[idx] = row

        if args.method == "cluster_head":
            if args.budget != args.head_budget:
                raise ValueError("cluster_head expects --budget to equal --head-budget.")
            selected_local = head_local
            manifest["cluster_head"] = {
                "head_budget": args.head_budget,
                "n_clusters": args.n_clusters,
                "per_cluster": args.per_cluster,
                "kmeans_n_init": args.kmeans_n_init,
            }
        else:
            seed_coco_path = args.seed_coco or args.output_coco_from
            if seed_coco_path is None:
                raise ValueError(
                    "cluster_tail requires --seed-coco for the selected head labels. "
                    "For paper replay, --output-coco-from may be used as the label source."
                )
            seed_counts_by_key = target_counts_by_image_key(
                load_coco(seed_coco_path),
                args.target_class_id,
                key_mode=args.key_mode,
            )
            target_counts = {
                local_idx: int(seed_counts_by_key.get(records[local_idx]["key"], 0))
                for local_idx in head_local
            }
            tail = select_tail_by_seed_clusters(
                features,
                cluster_labels,
                head_local,
                target_counts,
                total_budget=args.budget,
                target_box_count=args.target_box_count,
                tail_budget=(args.tail_budget or None),
                seed_cluster_count=args.tail_cluster_count,
                safety_factor=args.tail_safety_factor,
            )
            for idx in tail.selected:
                method_rows[idx] = {
                    "stage": "target_tail",
                    "cluster_id": int(cluster_labels[idx]),
                }
            selected_local = head_local + tail.selected
            candidate_ranking = tail.candidates
            manifest["cluster_tail"] = {
                "head_budget": args.head_budget,
                "n_clusters": args.n_clusters,
                "per_cluster": args.per_cluster,
                "kmeans_n_init": args.kmeans_n_init,
                "seed_label_source": str(seed_coco_path),
                "target_class_id": args.target_class_id,
                **tail.summary,
            }

    selected_keys = [records[idx]["key"] for idx in selected_local]
    if len(selected_keys) != args.budget or len(set(selected_keys)) != args.budget:
        raise AssertionError("Selection did not produce the requested number of unique images.")

    prefix = f"selected_{args.method}_seed{args.seed}_n{args.budget}"
    selection_csv = args.output_dir / f"{prefix}.csv"
    manifest_path = args.output_dir / f"{prefix}_manifest.json"
    rows = []
    for rank, idx in enumerate(selected_local, start=1):
        record = records[idx]
        rows.append(
            {
                "rank": rank,
                "file_name": record["key"],
                "path": record["path"],
                "global_idx": int(record["global_idx"]),
                **method_rows.get(idx, {}),
            }
        )
    write_selection_csv(selection_csv, rows)

    manifest["selection_csv"] = str(selection_csv)
    manifest["selected_count"] = len(rows)
    manifest["selected_global_indices"] = [int(records[idx]["global_idx"]) for idx in selected_local]

    if candidate_ranking:
        ranking_path = args.output_dir / f"{prefix}_tail_candidates.csv"
        ranking_rows = []
        for row in candidate_ranking:
            local_idx = int(row["local_idx"])
            ranking_rows.append(
                {
                    "rank": len(ranking_rows) + 1,
                    "file_name": records[local_idx]["key"],
                    "path": records[local_idx]["path"],
                    "global_idx": int(records[local_idx]["global_idx"]),
                    **row,
                }
            )
        write_selection_csv(ranking_path, ranking_rows)
        manifest["tail_candidate_csv"] = str(ranking_path)

    if args.output_coco_from:
        source_coco = load_coco(args.output_coco_from)
        selected_coco = subset_coco(source_coco, selected_keys, key_mode=args.key_mode)
        output_coco = args.output_dir / f"{prefix}.json"
        write_json(output_coco, selected_coco)
        manifest["selected_coco"] = str(output_coco)
        manifest["selected_coco_class_counts"] = class_counts(selected_coco)

    write_json(manifest_path, manifest)
    print(json.dumps(manifest, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
