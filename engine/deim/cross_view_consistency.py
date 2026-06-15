"""Cross-view query consistency for Semi-DETR style SSOD."""

from typing import Dict, List, Optional, Tuple

import torch
import torch.nn.functional as F
from scipy.optimize import linear_sum_assignment

from .box_ops import box_cxcywh_to_xyxy, generalized_box_iou


def _class_scores(logits: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    scores = logits.sigmoid()
    return scores.max(dim=-1)


def compute_query_cost(
    student_logits: torch.Tensor,
    student_boxes: torch.Tensor,
    teacher_logits: torch.Tensor,
    teacher_boxes: torch.Tensor,
    class_weight: float = 1.0,
    bbox_weight: float = 5.0,
    giou_weight: float = 2.0,
) -> torch.Tensor:
    """Compute query-to-query matching cost for one image."""
    student_prob = student_logits.sigmoid()
    teacher_prob = teacher_logits.sigmoid()
    class_cost = torch.cdist(student_prob, teacher_prob, p=1) / max(student_prob.shape[-1], 1)
    bbox_cost = torch.cdist(student_boxes, teacher_boxes, p=1)
    giou_cost = -generalized_box_iou(
        box_cxcywh_to_xyxy(student_boxes),
        box_cxcywh_to_xyxy(teacher_boxes),
    )
    return class_weight * class_cost + bbox_weight * bbox_cost + giou_weight * giou_cost


def compute_target_cost(
    logits: torch.Tensor,
    boxes: torch.Tensor,
    target_labels: torch.Tensor,
    target_boxes: torch.Tensor,
    class_weight: float = 1.0,
    bbox_weight: float = 5.0,
    giou_weight: float = 2.0,
) -> torch.Tensor:
    probs = logits.sigmoid()
    class_cost = -probs[:, target_labels.long()]
    bbox_cost = torch.cdist(boxes, target_boxes, p=1)
    giou_cost = -generalized_box_iou(
        box_cxcywh_to_xyxy(boxes),
        box_cxcywh_to_xyxy(target_boxes),
    )
    return class_weight * class_cost + bbox_weight * bbox_cost + giou_weight * giou_cost


def compute_query_consistency(
    student_outputs: Dict[str, torch.Tensor],
    teacher_outputs: Dict[str, torch.Tensor],
    targets: Optional[List[Dict[str, torch.Tensor]]] = None,
    score_threshold: float = 0.05,
    max_pairs: int = 100,
    normalize_features: bool = True,
    loss_scale: float = 1.0,
) -> Tuple[torch.Tensor, Dict[str, float]]:
    """Match student and teacher queries and penalize feature drift.

    Teacher features are always detached. If no valid teacher query is found,
    a zero loss connected to the student feature graph is returned.
    """
    student_consistency = student_outputs.get("consistency_query_features_all")
    teacher_consistency = teacher_outputs.get("consistency_query_features_all")
    if student_consistency is not None and teacher_consistency is not None:
        weights = student_outputs.get("consistency_query_weights")
        losses = []
        pair_count = 0
        for s_feat, t_feat in zip(student_consistency, teacher_consistency):
            t_feat = t_feat.detach()
            if normalize_features:
                s_feat = F.normalize(s_feat, p=2, dim=-1)
                t_feat = F.normalize(t_feat, p=2, dim=-1)
            per_pair = F.mse_loss(s_feat, t_feat, reduction="none").mean(dim=-1)
            if weights is not None:
                pair_weights = weights.to(device=per_pair.device, dtype=per_pair.dtype)
                per_pair = per_pair * pair_weights
                valid = pair_weights > 0
                if valid.any():
                    losses.append(per_pair[valid].mean())
                    pair_count += int(valid.sum().item())
                continue
            losses.append(per_pair.mean())
            pair_count += int(per_pair.numel())
        if losses:
            return torch.stack(losses).mean() * loss_scale, {"consistency_pairs": float(pair_count)}
        zero = student_consistency.sum() * 0.0
        return zero, {"consistency_pairs": 0.0}

    student_features = student_outputs.get("query_features")
    teacher_features = teacher_outputs.get("query_features")
    if student_features is None or teacher_features is None:
        zero = student_outputs["pred_logits"].sum() * 0.0
        return zero, {"consistency_pairs": 0.0}

    batch_size = student_features.shape[0]
    losses = []
    pair_count = 0
    for batch_idx in range(batch_size):
        if targets is not None:
            target = targets[batch_idx]
            target_boxes = target["boxes"]
            target_labels = target["labels"]
            if target_boxes.numel() == 0:
                continue
            if target_boxes.shape[0] > max_pairs:
                scores = target.get("scores", torch.ones(target_boxes.shape[0], device=target_boxes.device))
                top = torch.topk(scores, k=max_pairs).indices
                target_boxes = target_boxes[top]
                target_labels = target_labels[top]

            with torch.no_grad():
                student_cost = compute_target_cost(
                    student_outputs["pred_logits"][batch_idx],
                    student_outputs["pred_boxes"][batch_idx],
                    target_labels,
                    target_boxes,
                )
                teacher_cost = compute_target_cost(
                    teacher_outputs["pred_logits"][batch_idx],
                    teacher_outputs["pred_boxes"][batch_idx],
                    target_labels,
                    target_boxes,
                )
                student_idx = student_cost.argmin(dim=0)
                teacher_idx = teacher_cost.argmin(dim=0)

            s_feat = student_features[batch_idx, student_idx]
            t_feat = teacher_features[batch_idx, teacher_idx].detach()
            if normalize_features:
                s_feat = F.normalize(s_feat, p=2, dim=-1)
                t_feat = F.normalize(t_feat, p=2, dim=-1)
            weights = target.get("weights")
            per_pair = F.mse_loss(s_feat, t_feat, reduction="none").mean(dim=-1)
            if weights is not None and weights.numel() >= per_pair.numel():
                per_pair = per_pair * weights[:per_pair.numel()].to(per_pair.dtype)
            losses.append(per_pair.mean())
            pair_count += int(per_pair.numel())
            continue

        teacher_scores, _ = _class_scores(teacher_outputs["pred_logits"][batch_idx])
        valid_teacher = torch.nonzero(teacher_scores >= score_threshold, as_tuple=False).flatten()
        if valid_teacher.numel() == 0:
            valid_teacher = torch.topk(teacher_scores, k=min(max_pairs, teacher_scores.numel())).indices
        if valid_teacher.numel() > max_pairs:
            _, top_order = torch.topk(teacher_scores[valid_teacher], k=max_pairs)
            valid_teacher = valid_teacher[top_order]

        student_scores, _ = _class_scores(student_outputs["pred_logits"][batch_idx])
        valid_student = torch.topk(student_scores, k=min(valid_teacher.numel(), student_scores.numel())).indices
        if valid_student.numel() == 0 or valid_teacher.numel() == 0:
            continue

        with torch.no_grad():
            cost = compute_query_cost(
                student_outputs["pred_logits"][batch_idx, valid_student],
                student_outputs["pred_boxes"][batch_idx, valid_student],
                teacher_outputs["pred_logits"][batch_idx, valid_teacher],
                teacher_outputs["pred_boxes"][batch_idx, valid_teacher],
            )
            rows, cols = linear_sum_assignment(torch.nan_to_num(cost, nan=1e4).detach().cpu())

        s_idx = valid_student[torch.as_tensor(rows, device=student_features.device)]
        t_idx = valid_teacher[torch.as_tensor(cols, device=student_features.device)]
        s_feat = student_features[batch_idx, s_idx]
        t_feat = teacher_features[batch_idx, t_idx].detach()
        if normalize_features:
            s_feat = F.normalize(s_feat, p=2, dim=-1)
            t_feat = F.normalize(t_feat, p=2, dim=-1)
        losses.append(F.mse_loss(s_feat, t_feat, reduction="mean"))
        pair_count += int(s_idx.numel())

    if not losses:
        zero = student_features.sum() * 0.0
        return zero, {"consistency_pairs": 0.0}

    return torch.stack(losses).mean() * loss_scale, {"consistency_pairs": float(pair_count)}
