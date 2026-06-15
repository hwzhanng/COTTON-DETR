"""Cost-based pseudo label mining adapted from Semi-DETR for DEIMv2 outputs."""

import os
import warnings
from typing import Dict, List, Optional, Tuple

import torch
from scipy.optimize import linear_sum_assignment

from .box_ops import box_cxcywh_to_xyxy, generalized_box_iou

os.environ.setdefault("OMP_NUM_THREADS", "1")


def _empty_target(device: torch.device) -> Dict[str, torch.Tensor]:
    return {
        "boxes": torch.zeros((0, 4), device=device),
        "labels": torch.zeros((0,), dtype=torch.long, device=device),
        "scores": torch.zeros((0,), device=device),
        "weights": torch.zeros((0,), device=device),
    }


def _prediction_view(outputs: Dict[str, torch.Tensor], batch_idx: int):
    scores_all = outputs["pred_logits"][batch_idx].sigmoid()
    scores, labels = scores_all.max(dim=-1)
    boxes = outputs["pred_boxes"][batch_idx].detach().clamp(0, 1)
    return scores.detach(), labels.detach(), boxes


def _score_summary(scores: torch.Tensor) -> Dict[str, float]:
    if scores.numel() == 0:
        return {
            "teacher_score_mean": 0.0,
            "teacher_score_p50": 0.0,
            "teacher_score_p75": 0.0,
            "teacher_score_p90": 0.0,
            "teacher_score_p95": 0.0,
            "teacher_score_max": 0.0,
        }
    values = scores.detach().float()
    return {
        "teacher_score_mean": float(values.mean().item()),
        "teacher_score_p50": float(torch.quantile(values, 0.50).item()),
        "teacher_score_p75": float(torch.quantile(values, 0.75).item()),
        "teacher_score_p90": float(torch.quantile(values, 0.90).item()),
        "teacher_score_p95": float(torch.quantile(values, 0.95).item()),
        "teacher_score_max": float(values.max().item()),
    }


def _score_candidates(
    scores: torch.Tensor,
    score_threshold: float,
    threshold_mode: str,
    min_candidates: int = 0,
    max_candidates: int = 0,
    score_floor: float = 0.0,
    return_threshold: bool = False,
    labels: Optional[torch.Tensor] = None,
    class_thresholds: Optional[Dict[int, float]] = None,
):
    """Select candidate query indices whose teacher score crosses a threshold.

    When ``class_thresholds`` is provided, the per-query threshold is looked up
    by predicted class; this enables LabelMatch-style per-class adaptive
    thresholds. The returned ``threshold`` summary is the mean of the
    per-query thresholds actually applied (or the global threshold otherwise),
    kept as a single scalar for diagnostic logging.
    """
    if scores.numel() == 0:
        empty = torch.zeros((0,), dtype=torch.long, device=scores.device)
        return (empty, scores.new_tensor(0.0)) if return_threshold else empty
    use_per_class = (
        class_thresholds is not None
        and labels is not None
        and labels.numel() == scores.numel()
    )
    if use_per_class:
        per_query_thr = torch.full_like(scores, float(score_threshold))
        for cls_id, cls_thr in class_thresholds.items():
            mask = labels == int(cls_id)
            if mask.any():
                per_query_thr[mask] = float(cls_thr)
        if score_floor > 0:
            per_query_thr = per_query_thr.clamp(min=float(score_floor))
        selected = torch.nonzero(scores >= per_query_thr, as_tuple=False).flatten()
        threshold_scalar = per_query_thr.mean().detach()
    else:
        if threshold_mode == "mean_std":
            threshold = scores.mean() + scores.std(unbiased=False)
        elif threshold_mode == "mean_half_std":
            threshold = scores.mean() + 0.5 * scores.std(unbiased=False)
        elif threshold_mode == "adaptive":
            threshold = torch.minimum(
                scores.new_tensor(float(score_threshold)),
                scores.mean() + scores.std(unbiased=False),
            )
        elif threshold_mode == "fixed":
            threshold = scores.new_tensor(score_threshold)
        else:
            raise ValueError(f"Unknown pseudo threshold_mode={threshold_mode!r}.")
        if score_floor > 0:
            threshold = torch.maximum(threshold, scores.new_tensor(float(score_floor)))
        selected = torch.nonzero(scores >= threshold, as_tuple=False).flatten()
        threshold_scalar = threshold.detach()
    min_candidates = max(int(min_candidates), 0)
    max_candidates = max(int(max_candidates), 0)
    if min_candidates > 0 and selected.numel() < min_candidates:
        # Bug #5 fix: the min_candidates fallback used to pick top-K queries
        # unconditionally, BYPASSING the score_floor. After Bug #1 raised the
        # per-class thresholds into the real-detection band (0.10-0.40), an
        # uncertain teacher would emit zero accepted queries -- and the
        # fallback would then quietly force in the top-K *noise* queries
        # (e.g. score=0.03) which still receive non-trivial weight via
        # pseudo_weight_floor. Respect the absolute score_floor here so the
        # fallback can return *fewer* than min_candidates (or even zero)
        # when the teacher genuinely has no candidates above the floor.
        k = min(min_candidates, scores.numel())
        topk_indices = torch.topk(scores, k=k).indices
        if score_floor > 0:
            floor = float(score_floor)
            topk_scores = scores[topk_indices]
            keep_mask = topk_scores >= floor
            selected = topk_indices[keep_mask]
        else:
            selected = topk_indices
    if max_candidates > 0 and selected.numel() > max_candidates:
        top = torch.topk(scores[selected], k=max_candidates).indices
        selected = selected[top]
    return (selected, threshold_scalar) if return_threshold else selected


def _pseudo_weight_multiplier(
    scores: torch.Tensor,
    score_power: float = 1.0,
    min_effective_weight: float = 0.0,
    max_effective_weight: float = 1.0,
) -> torch.Tensor:
    """Return target ``weights`` so criterion effective weight is calibrated.

    DEIM/DETR confidence scores can be poorly calibrated in SSOD even when the
    ranking is useful. The criterion multiplies ``scores * weights``. Keeping
    raw scores for diagnostics while raising the effective target weight avoids
    late-stage pseudo labels becoming numerically invisible.
    """
    if scores.numel() == 0:
        return torch.zeros_like(scores)
    power = max(float(score_power), 1e-6)
    min_weight = max(float(min_effective_weight), 0.0)
    max_weight = max(float(max_effective_weight), min_weight)
    safe_scores = scores.detach().clamp_min(1e-6)
    effective = safe_scores.pow(power).clamp(min=min_weight, max=max_weight)
    return effective.to(scores.dtype) / safe_scores.to(scores.dtype)


def _effective_weight_mean(scores: torch.Tensor, weights: torch.Tensor) -> float:
    if scores.numel() == 0:
        return 0.0
    return float((scores.detach() * weights.detach()).mean().item())


def _prediction_to_target_cost(
    logits: torch.Tensor,
    boxes: torch.Tensor,
    target_labels: torch.Tensor,
    target_boxes: torch.Tensor,
    class_weight: float = 1.0,
    bbox_weight: float = 5.0,
    giou_weight: float = 2.0,
) -> torch.Tensor:
    if logits.numel() == 0 or target_boxes.numel() == 0:
        return logits.new_zeros((logits.shape[0], target_boxes.shape[0]))
    probs = logits.sigmoid()
    class_cost = -probs[:, target_labels.long()]
    bbox_cost = torch.cdist(boxes, target_boxes, p=1)
    giou_cost = -generalized_box_iou(
        box_cxcywh_to_xyxy(boxes),
        box_cxcywh_to_xyxy(target_boxes),
    )
    return class_weight * class_cost + bbox_weight * bbox_cost + giou_weight * giou_cost


def _fit_gmm_threshold(costs: torch.Tensor, fallback_quantile: float) -> torch.Tensor:
    costs = costs.detach().float()
    if costs.numel() == 0:
        return costs.new_tensor(float("inf"))
    if costs.numel() == 1:
        return costs[0]

    sorted_costs = costs.sort().values
    try:
        import numpy as np
        from sklearn.mixture import GaussianMixture

        points = sorted_costs.cpu().numpy().reshape(-1, 1)
        means_init = np.array([points.min(), points.max()]).reshape(2, 1)
        gmm = GaussianMixture(
            2,
            weights_init=np.array([0.5, 0.5]),
            means_init=means_init,
            covariance_type="diag",
            reg_covar=1e-5,
        )
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message="KMeans is known to have a memory leak.*", category=UserWarning)
            gmm.fit(points)
        assignments = torch.as_tensor(gmm.predict(points), device=costs.device)
        scores = torch.as_tensor(gmm.score_samples(points), device=costs.device)
        reliable_cluster = int(gmm.means_.reshape(-1).argmin())
        reliable_mask = assignments == reliable_cluster
        if reliable_mask.any():
            best = torch.topk(scores[reliable_mask], k=1).indices
            return sorted_costs[reliable_mask][best][0]
    except Exception:
        pass

    q = min(max(float(fallback_quantile), 0.0), 1.0)
    return torch.quantile(sorted_costs, q=q)


@torch.no_grad()
def cost_based_pseudo_mining(
    teacher_outputs: Dict[str, torch.Tensor],
    student_outputs: Dict[str, torch.Tensor] = None,
    score_threshold: float = 0.7,
    min_pseudo_labels: int = 1,
    max_pseudo_labels: int = 50,
    max_mined_labels: int = 20,
    mined_score_floor: float = 0.05,
    mined_weight: float = 1.0,
    cost_quantile: float = 0.25,
    threshold_mode: str = "mean_std",
    gmm_enabled: bool = True,
    min_candidate_labels: int = 0,
    max_candidate_labels: int = 80,
    min_query_labels: int = 0,
    max_query_labels: int = 100,
    min_teacher_score_for_mining: float = 0.0,
    high_threshold_mode: str = None,
    high_score_floor: float = 0.0,
    candidate_score_floor: float = 0.0,
    pseudo_score_power: float = 1.0,
    pseudo_weight_floor: float = 0.0,
    pseudo_weight_ceiling: float = 1.0,
    class_thresholds: Optional[Dict[int, float]] = None,
    candidate_class_thresholds: Optional[Dict[int, float]] = None,
    num_classes: Optional[int] = None,
    max_per_class_labels: int = 0,
    max_mined_per_class_labels: int = 0,
) -> Tuple[List[Dict[str, torch.Tensor]], List[Dict[str, torch.Tensor]], Dict[str, float]]:
    """Build high-confidence and mined pseudo-label targets.

    High-confidence labels come from the teacher. Mined labels come from the
    provided student outputs when available, otherwise from the remaining
    teacher queries. This keeps the first integration robust while preserving
    the Semi-DETR cost-based expansion behavior.
    """
    batch_size = teacher_outputs["pred_logits"].shape[0]
    high_targets: List[Dict[str, torch.Tensor]] = []
    mined_targets: List[Dict[str, torch.Tensor]] = []
    score_stat_keys = (
        "teacher_score_mean",
        "teacher_score_p50",
        "teacher_score_p75",
        "teacher_score_p90",
        "teacher_score_p95",
        "teacher_score_max",
    )
    stats = {
        "high_conf_labels": 0.0,
        "mined_labels": 0.0,
        "consistency_labels": 0.0,
        "high_conf_effective_threshold": 0.0,
        "high_conf_target_weight_mean": 0.0,
        "mined_target_weight_mean": 0.0,
        "candidate_initial_labels": 0.0,
        "candidate_query_labels": 0.0,
        "matched_pairs": 0.0,
        "accepted_mined_labels": 0.0,
        **{key: 0.0 for key in score_stat_keys},
    }
    # Per-class diagnostic counters so we can see whether any rare
    # class) actually receives pseudo-labels -- flat aggregate counts hide that.
    if num_classes is not None and int(num_classes) > 0:
        for cls_id in range(int(num_classes)):
            stats[f"high_conf_labels_class_{cls_id}"] = 0.0
            stats[f"mined_labels_class_{cls_id}"] = 0.0

    candidate_outputs = student_outputs if student_outputs is not None else teacher_outputs

    for batch_idx in range(batch_size):
        teacher_scores, teacher_labels, teacher_boxes = _prediction_view(teacher_outputs, batch_idx)
        for key, value in _score_summary(teacher_scores).items():
            stats[key] += value
        cap_high_per_class = max_per_class_labels > 0 and num_classes is not None
        high_idx, high_threshold = _score_candidates(
            teacher_scores,
            score_threshold,
            high_threshold_mode or "fixed",
            min_candidates=min_pseudo_labels,
            max_candidates=0 if cap_high_per_class else max_pseudo_labels,
            score_floor=high_score_floor,
            return_threshold=True,
            labels=teacher_labels,
            class_thresholds=class_thresholds,
        )
        stats["high_conf_effective_threshold"] += float(high_threshold.item())
        # Bug #3 fix: cap pseudo-labels per class so a single noisy class
        # cannot occupy the full per-image budget. Without this, when
        # adaptive_threshold goes wrong the dominant-noise class consumes
        # all ``max_pseudo_labels`` slots and the model sees an inverted
        # class distribution. We keep the top-K scoring entries per class.
        if cap_high_per_class and high_idx.numel() > 0:
            high_label_vec = teacher_labels[high_idx].long()
            kept_indices = []
            for cls_id in range(int(num_classes)):
                cls_mask = high_label_vec == cls_id
                cls_pos = torch.nonzero(cls_mask, as_tuple=False).flatten()
                if cls_pos.numel() == 0:
                    continue
                if cls_pos.numel() > max_per_class_labels:
                    cls_scores = teacher_scores[high_idx[cls_pos]]
                    top = torch.topk(cls_scores, k=int(max_per_class_labels)).indices
                    cls_pos = cls_pos[top]
                kept_indices.append(cls_pos)
            if kept_indices:
                kept = torch.cat(kept_indices)
                high_idx = high_idx[kept]
            if max_pseudo_labels > 0 and high_idx.numel() > max_pseudo_labels:
                top = torch.topk(teacher_scores[high_idx], k=int(max_pseudo_labels)).indices
                high_idx = high_idx[top]
        high_scores = teacher_scores[high_idx]
        high_weights = _pseudo_weight_multiplier(
            high_scores,
            score_power=pseudo_score_power,
            min_effective_weight=pseudo_weight_floor,
            max_effective_weight=pseudo_weight_ceiling,
        )

        high_target = {
            "boxes": teacher_boxes[high_idx],
            "labels": teacher_labels[high_idx].long(),
            "scores": high_scores,
            "weights": high_weights,
        }
        high_targets.append(high_target)
        stats["high_conf_labels"] += float(high_idx.numel())
        stats["high_conf_target_weight_mean"] += _effective_weight_mean(high_scores, high_weights)
        if num_classes is not None and high_idx.numel() > 0:
            high_label_vec = teacher_labels[high_idx].long()
            for cls_id in range(int(num_classes)):
                count = int((high_label_vec == cls_id).sum().item())
                if count > 0:
                    stats[f"high_conf_labels_class_{cls_id}"] += float(count)

        if max_mined_labels <= 0:
            mined_targets.append(_empty_target(teacher_boxes.device))
            stats["consistency_labels"] += float(high_idx.numel())
            continue
        if teacher_scores.numel() == 0 or float(teacher_scores.max().item()) < float(min_teacher_score_for_mining):
            mined_targets.append(_empty_target(teacher_boxes.device))
            stats["consistency_labels"] += float(high_idx.numel())
            continue

        cand_scores, cand_labels, cand_boxes = _prediction_view(candidate_outputs, batch_idx)
        # A. mining revival: route a LOOSER threshold set to the candidate
        # selection. With identical class_thresholds for both branches,
        # candidate_initial_idx == high_idx -> gmm_idx subset of high_idx ->
        # mined_idx (= consistency - high) was always empty. Acceptance rate
        # dropped to 0.02% in fix6. By giving candidate a scaled-down
        # threshold (e.g. class_thresholds * 0.7) the candidate pool becomes
        # a SUPERSET of high, so cost-based matching can find labels not
        # already in high. Falls back to class_thresholds when caller did
        # not provide a separate set, preserving prior behaviour.
        cand_thresholds = (
            candidate_class_thresholds
            if candidate_class_thresholds is not None
            else class_thresholds
        )
        candidate_initial_idx = _score_candidates(
            teacher_scores,
            score_threshold,
            threshold_mode,
            min_candidates=min_candidate_labels,
            max_candidates=max_candidate_labels,
            score_floor=candidate_score_floor,
            labels=teacher_labels,
            class_thresholds=cand_thresholds,
        )
        stats["candidate_initial_labels"] += float(candidate_initial_idx.numel())
        if candidate_initial_idx.numel() == 0:
            candidate_initial_idx = high_idx
        if candidate_initial_idx.numel() == 0:
            mined_targets.append(_empty_target(teacher_boxes.device))
            stats["consistency_labels"] += float(high_idx.numel())
            continue

        query_mask = cand_scores >= mined_score_floor
        query_idx = torch.nonzero(query_mask, as_tuple=False).flatten()
        min_query_labels = max(int(min_query_labels), 0)
        max_query_labels = max(int(max_query_labels), 0)
        if min_query_labels > 0 and query_idx.numel() < min_query_labels:
            floor_idx = torch.nonzero(cand_scores >= float(mined_score_floor), as_tuple=False).flatten()
            if floor_idx.numel() > 0:
                k = min(min_query_labels, floor_idx.numel())
                top = torch.topk(cand_scores[floor_idx], k=k).indices
                query_idx = floor_idx[top]
            else:
                query_idx = floor_idx
        if max_query_labels > 0 and query_idx.numel() > max_query_labels:
            top = torch.topk(cand_scores[query_idx], k=max_query_labels).indices
            query_idx = query_idx[top]
        stats["candidate_query_labels"] += float(query_idx.numel())
        if query_idx.numel() == 0:
            mined_targets.append(_empty_target(teacher_boxes.device))
            stats["consistency_labels"] += float(high_idx.numel())
            continue

        cost = _prediction_to_target_cost(
            candidate_outputs["pred_logits"][batch_idx, query_idx],
            cand_boxes[query_idx],
            teacher_labels[candidate_initial_idx],
            teacher_boxes[candidate_initial_idx],
        )
        rows, cols = linear_sum_assignment(torch.nan_to_num(cost, nan=1e4).detach().cpu())
        stats["matched_pairs"] += float(len(rows))
        if len(rows) == 0:
            mined_targets.append(_empty_target(teacher_boxes.device))
            stats["consistency_labels"] += float(high_idx.numel())
            continue

        rows = torch.as_tensor(rows, device=teacher_boxes.device, dtype=torch.long)
        cols = torch.as_tensor(cols, device=teacher_boxes.device, dtype=torch.long)
        matched_cost = cost[rows, cols]
        threshold = _fit_gmm_threshold(matched_cost, cost_quantile) if gmm_enabled else torch.quantile(
            matched_cost, q=min(max(float(cost_quantile), 0.0), 1.0)
        )
        gmm_idx = candidate_initial_idx[cols[matched_cost <= threshold]]
        consistency_idx = torch.cat([high_idx, gmm_idx]).unique()
        stats["consistency_labels"] += float(consistency_idx.numel())

        if high_idx.numel() > 0:
            keep_mask = ~torch.isin(consistency_idx, high_idx)
            mined_idx = consistency_idx[keep_mask]
        else:
            mined_idx = consistency_idx
        # Bug #6 fix: mirror the high-branch per-class cap on the mined branch.
        # Without this, mined could still funnel a single noisy class through
        # max_mined_labels slots even when the class is very rare in the
        # labeled distribution.
        # Apply the class cap before the global cap so lower-scored-but-valid
        # classes are not discarded before the diversity constraint can act.
        mined_per_class_cap = int(max_mined_per_class_labels or 0)
        if mined_per_class_cap <= 0:
            mined_per_class_cap = int(max_per_class_labels or 0)
        if mined_per_class_cap > 0 and mined_idx.numel() > 0 and num_classes is not None:
            mined_label_vec = teacher_labels[mined_idx].long()
            kept_mined: List[torch.Tensor] = []
            for cls_id in range(int(num_classes)):
                cls_pos = torch.nonzero(mined_label_vec == cls_id, as_tuple=False).flatten()
                if cls_pos.numel() == 0:
                    continue
                if cls_pos.numel() > mined_per_class_cap:
                    cls_scores = teacher_scores[mined_idx[cls_pos]]
                    top = torch.topk(cls_scores, k=int(mined_per_class_cap)).indices
                    cls_pos = cls_pos[top]
                kept_mined.append(cls_pos)
            if kept_mined:
                mined_idx = mined_idx[torch.cat(kept_mined)]
        if mined_idx.numel() > max_mined_labels:
            order = torch.argsort(teacher_scores[mined_idx], descending=True)[:max_mined_labels]
            mined_idx = mined_idx[order]
        stats["accepted_mined_labels"] += float(mined_idx.numel())
        mined_scores = teacher_scores[mined_idx]
        # Note: ``mined_weight`` is intentionally NOT applied here. The trainer
        # scales the mined loss term once via ``mined_loss_weight``; applying it
        # again here would compound the scaling (e.g. 0.3 * 0.3 = 0.09) and
        # silently shrink mined supervision by an order of magnitude.
        mined_weights = _pseudo_weight_multiplier(
            mined_scores,
            score_power=pseudo_score_power,
            min_effective_weight=pseudo_weight_floor,
            max_effective_weight=pseudo_weight_ceiling,
        )

        mined_target = {
            "boxes": teacher_boxes[mined_idx],
            "labels": teacher_labels[mined_idx].long(),
            "scores": mined_scores,
            "weights": mined_weights,
        }
        mined_targets.append(mined_target)
        stats["mined_labels"] += float(mined_idx.numel())
        stats["mined_target_weight_mean"] += _effective_weight_mean(mined_scores, mined_weights)
        if num_classes is not None and mined_idx.numel() > 0:
            mined_label_vec = teacher_labels[mined_idx].long()
            for cls_id in range(int(num_classes)):
                count = int((mined_label_vec == cls_id).sum().item())
                if count > 0:
                    stats[f"mined_labels_class_{cls_id}"] += float(count)

    if batch_size > 0:
        for key in (
            *score_stat_keys,
            "high_conf_effective_threshold",
            "high_conf_target_weight_mean",
            "mined_target_weight_mean",
        ):
            stats[key] /= float(batch_size)
    stats["mined_acceptance_rate"] = stats["accepted_mined_labels"] / max(stats["candidate_initial_labels"], 1.0)
    return high_targets, mined_targets, stats
