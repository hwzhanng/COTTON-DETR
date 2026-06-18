"""Selection algorithms used by the public cold-start pipeline.

The implementations operate on feature rows, not on project-specific files.
They therefore expose only the method used in the paper: detector-free DINOv3
coverage selection followed by a seed-cluster tail expansion.
"""

from __future__ import annotations

import math
import random
from collections import Counter
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class TailSelectionResult:
    selected: list[int]
    candidates: list[dict]
    summary: dict


def l2_normalize(features: np.ndarray) -> np.ndarray:
    matrix = np.asarray(features, dtype=np.float32)
    if matrix.ndim != 2:
        raise ValueError(f"Expected a 2-D feature matrix, got shape {matrix.shape}.")
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    if np.any(norms <= 0):
        raise ValueError("Feature matrix contains zero-norm rows.")
    return matrix / norms


def random_indices(size: int, budget: int, seed: int) -> list[int]:
    if budget <= 0 or budget > size:
        raise ValueError(f"Invalid budget {budget} for {size} candidates.")
    return sorted(random.Random(seed).sample(range(size), budget))


def _choose_first(features: np.ndarray, init: str, seed: int) -> int:
    if init == "first":
        return 0
    if init == "random":
        return random.Random(seed).randrange(len(features))
    if init != "medoid":
        raise ValueError(f"Unsupported init strategy: {init}")
    center = features.mean(axis=0)
    center = center / np.linalg.norm(center)
    return int(np.argmax(features @ center))


def kcenter_greedy(
    features: np.ndarray,
    budget: int,
    *,
    init: str = "medoid",
    seed: int = 0,
) -> tuple[list[int], list[float], np.ndarray]:
    """Farthest-first k-center selection with cosine distance."""

    matrix = l2_normalize(features)
    if budget <= 0 or budget > len(matrix):
        raise ValueError(f"Invalid budget {budget} for {len(matrix)} candidates.")

    selected: list[int] = []
    score_before: list[float] = []
    selected_mask = np.zeros(len(matrix), dtype=bool)
    nearest = np.full(len(matrix), np.inf, dtype=np.float32)
    idx = _choose_first(matrix, init=init, seed=seed)

    for rank in range(budget):
        selected.append(idx)
        score_before.append(float("inf") if rank == 0 else float(nearest[idx]))
        selected_mask[idx] = True

        distances = 1.0 - (matrix @ matrix[idx])
        nearest = np.minimum(nearest, distances.astype(np.float32, copy=False))
        scores = nearest.copy()
        scores[selected_mask] = -np.inf
        if rank + 1 < budget:
            idx = int(np.argmax(scores))

    return selected, score_before, nearest


def select_cluster_representatives(
    features: np.ndarray,
    *,
    n_clusters: int,
    per_cluster: int,
    seed: int,
    n_init: int = 10,
) -> tuple[list[int], np.ndarray, list[dict]]:
    """Run k-means and take the closest samples to each cluster center."""

    if n_clusters <= 0 or per_cluster <= 0:
        raise ValueError("n_clusters and per_cluster must be positive.")

    try:
        from sklearn.cluster import KMeans
    except ImportError as exc:  # pragma: no cover - depends on optional install.
        raise SystemExit("Install scikit-learn to run cluster_tail selection.") from exc

    matrix = l2_normalize(features)
    if len(matrix) < n_clusters * per_cluster:
        raise ValueError(
            f"Need at least {n_clusters * per_cluster} samples, got {len(matrix)}."
        )

    kmeans = KMeans(n_clusters=n_clusters, random_state=seed, n_init=n_init)
    cluster_labels = kmeans.fit_predict(matrix)
    centers = l2_normalize(kmeans.cluster_centers_.astype(np.float32, copy=False))

    selected: list[int] = []
    rows: list[dict] = []
    for cluster_id in range(n_clusters):
        members = np.where(cluster_labels == cluster_id)[0]
        if len(members) < per_cluster:
            raise ValueError(
                f"Cluster {cluster_id} has {len(members)} samples; "
                f"{per_cluster} are required."
            )
        distances = 1.0 - (matrix[members] @ centers[cluster_id])
        order = np.argsort(distances, kind="mergesort")[:per_cluster]
        for rank, local_order in enumerate(order, start=1):
            idx = int(members[int(local_order)])
            selected.append(idx)
            rows.append(
                {
                    "stage": "coverage_head",
                    "cluster_id": int(cluster_id),
                    "selected_rank_in_cluster": int(rank),
                    "distance_to_cluster_center": float(distances[int(local_order)]),
                }
            )

    return selected, cluster_labels.astype(int), rows


def select_tail_by_seed_clusters(
    features: np.ndarray,
    cluster_labels: np.ndarray,
    head_indices: list[int],
    target_counts: dict[int, int],
    *,
    total_budget: int,
    target_box_count: int,
    tail_budget: int | None,
    seed_cluster_count: int,
    safety_factor: float,
) -> TailSelectionResult:
    """Expand the coverage head toward rare target-class seed clusters.

    Only target counts for already selected head images are required. Labels for
    unselected images are not consulted by this function.
    """

    matrix = l2_normalize(features)
    head_set = set(head_indices)
    if len(head_set) != len(head_indices):
        raise ValueError("Head selection contains duplicate indices.")
    if total_budget < len(head_indices):
        raise ValueError("total_budget must be >= the head selection size.")

    seed_indices = [idx for idx in head_indices if target_counts.get(idx, 0) > 0]
    if not seed_indices:
        raise ValueError("No positive target-class seed images found in the selected head.")

    cluster_rows = []
    for cluster_id in sorted(set(int(cluster_labels[idx]) for idx in head_indices)):
        members = [idx for idx in head_indices if int(cluster_labels[idx]) == cluster_id]
        target_images = sum(1 for idx in members if target_counts.get(idx, 0) > 0)
        target_boxes = sum(int(target_counts.get(idx, 0)) for idx in members)
        cluster_rows.append(
            {
                "cluster_id": int(cluster_id),
                "selected_images": int(len(members)),
                "target_images": int(target_images),
                "target_boxes": int(target_boxes),
                "target_box_density": float(target_boxes / max(len(members), 1)),
            }
        )

    cluster_rows = sorted(
        cluster_rows,
        key=lambda row: (
            row["target_boxes"],
            row["target_images"],
            row["target_box_density"],
            -row["cluster_id"],
        ),
        reverse=True,
    )
    seed_clusters = [int(row["cluster_id"]) for row in cluster_rows[:seed_cluster_count]]
    top_members = [idx for idx in head_indices if int(cluster_labels[idx]) in seed_clusters]
    top_target_boxes = sum(int(target_counts.get(idx, 0)) for idx in top_members)
    top_density = float(top_target_boxes / max(len(top_members), 1))
    if top_density <= 0:
        raise ValueError("Seed-cluster target density is zero.")

    baseline_target_boxes = sum(int(target_counts.get(idx, 0)) for idx in head_indices)
    remaining_budget = total_budget - len(head_indices)
    estimated_tail_budget = None
    if tail_budget is None:
        required_boxes = max(0, int(target_box_count) - int(baseline_target_boxes))
        estimated_tail_budget = int(math.ceil((required_boxes / top_density) * safety_factor - 1e-9))
        tail_budget = remaining_budget
    tail_budget = min(int(tail_budget), remaining_budget)
    if tail_budget <= 0:
        return TailSelectionResult(
            selected=[],
            candidates=[],
            summary={
                "tail_budget": 0,
                "baseline_target_boxes": int(baseline_target_boxes),
                "seed_clusters": cluster_rows[:seed_cluster_count],
            },
        )

    seed_centroid = matrix[np.array(seed_indices, dtype=int)].mean(axis=0, keepdims=True)
    seed_centroid = l2_normalize(seed_centroid)[0]
    cluster_order = {cluster_id: order for order, cluster_id in enumerate(seed_clusters)}

    candidates = []
    for idx in range(len(matrix)):
        if idx in head_set:
            continue
        cluster_id = int(cluster_labels[idx])
        if cluster_id not in cluster_order:
            continue
        distance = float(1.0 - (matrix[idx] @ seed_centroid))
        candidates.append(
            {
                "local_idx": int(idx),
                "cluster_id": int(cluster_id),
                "cluster_order": int(cluster_order[cluster_id]),
                "distance_to_seed_centroid": distance,
            }
        )

    candidates = sorted(
        candidates,
        key=lambda row: (row["distance_to_seed_centroid"], row["cluster_order"], row["local_idx"]),
    )
    if len(candidates) < tail_budget:
        raise ValueError(f"Only {len(candidates)} tail candidates available, need {tail_budget}.")

    selected = [int(row["local_idx"]) for row in candidates[:tail_budget]]
    summary = {
        "selection_rule": (
            "Use target-class boxes from the selected coverage head to identify seed "
            "clusters; rank unselected images in those clusters by DINOv3 distance "
            "to the target seed centroid."
        ),
        "baseline_target_boxes": int(baseline_target_boxes),
        "seed_target_images": int(len(seed_indices)),
        "seed_cluster_count": int(seed_cluster_count),
        "seed_clusters": cluster_rows[:seed_cluster_count],
        "seed_cluster_density_boxes_per_selected_image": float(top_density),
        "target_box_count": int(target_box_count),
        "estimated_tail_budget_from_density": (
            None if estimated_tail_budget is None else int(estimated_tail_budget)
        ),
        "safety_factor": float(safety_factor),
        "tail_budget": int(tail_budget),
        "candidate_count": int(len(candidates)),
        "cluster_histogram": dict(Counter(int(cluster_labels[idx]) for idx in selected)),
    }
    return TailSelectionResult(selected=selected, candidates=candidates, summary=summary)
