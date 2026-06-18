#!/usr/bin/env python
"""Create nested label-budget subsets from a full cold-start selection."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from selection.coco import class_counts, load_coco, subset_coco, write_json  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--selection-csv", type=Path, required=True)
    parser.add_argument("--budgets", type=int, nargs="+", default=[152, 304, 608])
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/selection/nested"))
    parser.add_argument("--output-coco-from", type=Path)
    parser.add_argument("--key-mode", choices=["basename", "relative"], default="basename")
    parser.add_argument(
        "--tail-fraction",
        type=float,
        default=0.0,
        help="0 means infer target-tail fraction from the full selection CSV.",
    )
    return parser.parse_args()


def read_rows(path: Path) -> list[dict]:
    with path.open(encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise ValueError(f"Empty selection CSV: {path}")
    for row in rows:
        row["rank"] = int(row["rank"])
        row["global_idx"] = int(row["global_idx"])
        row["cluster_id"] = int(row["cluster_id"]) if row.get("cluster_id") not in ("", None) else -1
        row["selected_rank_in_cluster"] = (
            int(row["selected_rank_in_cluster"]) if row.get("selected_rank_in_cluster") else 10**9
        )
    return rows


def round_robin_head(rows: list[dict], budget: int) -> list[dict]:
    ordered = sorted(
        rows,
        key=lambda row: (
            row["selected_rank_in_cluster"],
            row["cluster_id"],
            row["global_idx"],
        ),
    )
    if len(ordered) < budget:
        raise ValueError(f"Only {len(ordered)} coverage-head rows available, need {budget}.")
    return ordered[:budget]


def choose_budget_rows(rows: list[dict], budget: int, tail_fraction: float) -> list[dict]:
    head_rows = [row for row in rows if row.get("stage") == "coverage_head"]
    tail_rows = [row for row in rows if row.get("stage") == "target_tail"]
    other_rows = [row for row in rows if row.get("stage") not in {"coverage_head", "target_tail"}]
    if other_rows:
        raise ValueError("Nested subset generation expects cluster_tail rows only.")
    if budget > len(rows):
        raise ValueError(f"Budget {budget} exceeds full selection size {len(rows)}.")
    if budget == len(rows):
        return sorted(rows, key=lambda row: row["rank"])

    tail_budget = int(round(float(budget) * tail_fraction))
    tail_budget = min(tail_budget, len(tail_rows), budget)
    head_budget = budget - tail_budget
    if head_budget > len(head_rows):
        extra = head_budget - len(head_rows)
        head_budget = len(head_rows)
        tail_budget += extra
    selected_head = round_robin_head(head_rows, head_budget)
    selected_tail = sorted(tail_rows, key=lambda row: row["rank"])[:tail_budget]
    selected = selected_head + selected_tail
    if len(selected) != budget:
        raise AssertionError(f"Expected {budget} selected rows, got {len(selected)}.")
    return selected


def write_selection_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0].keys())
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for rank, row in enumerate(rows, start=1):
            out = dict(row)
            out["rank"] = rank
            writer.writerow(out)


def main() -> int:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    rows = read_rows(args.selection_csv)
    full_tail = [row for row in rows if row.get("stage") == "target_tail"]
    tail_fraction = args.tail_fraction or (len(full_tail) / len(rows))
    source_coco = load_coco(args.output_coco_from) if args.output_coco_from else None

    manifests = []
    for budget in args.budgets:
        selected = choose_budget_rows(rows, budget, tail_fraction)
        prefix = f"nested_n{budget}"
        csv_path = args.output_dir / f"{prefix}.csv"
        write_selection_csv(csv_path, selected)
        manifest = {
            "budget": int(budget),
            "source_selection_csv": str(args.selection_csv),
            "selection_csv": str(csv_path),
            "tail_fraction": float(tail_fraction),
            "coverage_head_count": sum(1 for row in selected if row.get("stage") == "coverage_head"),
            "target_tail_count": sum(1 for row in selected if row.get("stage") == "target_tail"),
        }
        if source_coco is not None:
            keys = [row["file_name"] for row in selected]
            selected_coco = subset_coco(source_coco, keys, key_mode=args.key_mode)
            coco_path = args.output_dir / f"{prefix}.json"
            write_json(coco_path, selected_coco)
            manifest["selected_coco"] = str(coco_path)
            manifest["selected_coco_class_counts"] = class_counts(selected_coco)
        write_json(args.output_dir / f"{prefix}_manifest.json", manifest)
        manifests.append(manifest)

    summary = {"budgets": manifests}
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
