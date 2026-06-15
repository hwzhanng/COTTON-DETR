"""Metrics for semantic sidecar predictions."""

from __future__ import annotations

import torch


def macro_f1(pred: torch.Tensor, target: torch.Tensor, num_classes: int) -> float:
    scores = []
    for cls in range(num_classes):
        p = pred == cls
        t = target == cls
        tp = int((p & t).sum().item())
        fp = int((p & ~t).sum().item())
        fn = int((~p & t).sum().item())
        denom = 2 * tp + fp + fn
        scores.append(0.0 if denom == 0 else (2 * tp) / denom)
    return float(sum(scores) / len(scores))


def per_class_f1(pred: torch.Tensor, target: torch.Tensor, labels: list[str]) -> dict:
    out = {}
    for idx, label in enumerate(labels):
        p = pred == idx
        t = target == idx
        tp = int((p & t).sum().item())
        fp = int((p & ~t).sum().item())
        fn = int((~p & t).sum().item())
        precision = 0.0 if tp + fp == 0 else tp / (tp + fp)
        recall = 0.0 if tp + fn == 0 else tp / (tp + fn)
        f1 = 0.0 if 2 * tp + fp + fn == 0 else (2 * tp) / (2 * tp + fp + fn)
        out[label] = {"precision": precision, "recall": recall, "f1": f1, "support": int(t.sum().item())}
    return out
