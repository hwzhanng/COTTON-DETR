"""LabelMatch-style class-adaptive pseudo-label threshold for SSOD.

Motivation
----------
TARS v3 originally used a single global ``pseudo_threshold`` that warmed up
from 0.10 to 0.20. Two problems with that schedule:

1. DETR sigmoid scores are not calibrated across classes -- rare classes
   minority classes routinely score lower than dominant classes, so a global
   threshold either drowns the head class with noise or silently filters the
   tail class out entirely.
2. Monotonically raising the threshold near the end of training coincided
   with the observed AP decline (best epoch 119 -> early stop at 141), a
   classic confirmation-bias failure mode described in Consistent-Teacher
   (CVPR 2023) and LabelMatch (CVPR 2022).

This module maintains a rolling buffer of teacher max-scores per class and
periodically refreshes a per-class threshold equal to a class-specific
quantile of those recent scores. Rare classes can be given a looser quantile
(keep a larger fraction of teacher candidates), letting them survive even
when their absolute scores are lower than dominant classes.

The threshold tracker is *stateful* (so it can warm up its history before
emitting non-default thresholds) but performs no parameter learning.
"""

from __future__ import annotations

from collections import deque
from typing import Any, Deque, Dict, Iterable, Optional, Sequence

import torch


def _as_int_keyed_float_dict(raw: Any) -> Dict[int, float]:
    if raw is None:
        return {}
    if not isinstance(raw, dict):
        raise TypeError(f"expected dict, got {type(raw).__name__}")
    out: Dict[int, float] = {}
    for key, value in raw.items():
        out[int(key)] = float(value)
    return out


class ClassAdaptivePseudoThreshold:
    """Per-class rolling-quantile pseudo-label threshold tracker.

    Parameters
    ----------
    num_classes:
        Number of detector classes.
    target_quantile_per_class:
        Per-class quantile. ``threshold_c`` is set to the ``quantile_c``-th
        percentile of the rolling score history for class ``c``. Larger
        quantile = stricter threshold. Default 0.95 for every class.
    init_thresholds:
        Per-class initial thresholds returned while history is still warming
        up. Falls back to ``default_threshold`` if a class is missing.
    default_threshold:
        Initial threshold used for classes without an entry in
        ``init_thresholds`` and as the global fallback consumed downstream.
    min_thresh / max_thresh:
        Per-class clipping range applied after the quantile computation. A
        non-zero ``min_thresh`` prevents pathological score-collapse early on;
        ``max_thresh`` caps very confident classes so the head class cannot
        starve itself of pseudo-labels.
    history_size:
        Maximum number of stored teacher scores per class. Older entries are
        evicted FIFO so the threshold tracks recent teacher behaviour.
    min_history_for_quantile:
        A per-class quantile is only computed once that class has accumulated
        at least this many observations; otherwise the init threshold is used.
    update_interval:
        Recompute thresholds at most every ``update_interval`` calls. Saves
        the quantile work on every training step while still adapting often.
    enabled:
        Bypass switch. When False, ``get_thresholds`` returns ``init_thresholds``
        unchanged. Useful for ablations.
    """

    def __init__(
        self,
        num_classes: int,
        target_quantile_per_class: Optional[Dict[Any, float]] = None,
        init_thresholds: Optional[Dict[Any, float]] = None,
        default_threshold: float = 0.1,
        default_quantile: float = 0.95,
        min_thresh: float = 0.01,
        max_thresh: float = 0.5,
        history_size: int = 2000,
        min_history_for_quantile: int = 100,
        update_interval: int = 50,
        enabled: bool = True,
        input_score_floor: float = 0.0,
    ) -> None:
        if int(num_classes) <= 0:
            raise ValueError(f"num_classes must be positive, got {num_classes}")
        self.num_classes = int(num_classes)
        self.default_threshold = float(default_threshold)
        self.default_quantile = float(default_quantile)
        self.min_thresh = float(min_thresh)
        self.max_thresh = float(max_thresh)
        self.history_size = int(history_size)
        self.min_history_for_quantile = int(min_history_for_quantile)
        self.update_interval = max(int(update_interval), 1)
        self.enabled = bool(enabled)
        # Bug #1 fix: pre-filter teacher scores before pushing into the
        # per-class buffer. DETR decoders emit 300 queries per image but
        # only a handful are real detections; the rest argmax into some
        # class with very low confidence. Including those "noisy argmax"
        # entries makes the 0.85-0.95 quantile land in the NOISE band
        # (~0.02-0.04) instead of the real-detection band (~0.10-0.30).
        # Result with floor=0: pseudo-label class distribution gets
        # inverted in long-tailed datasets, causing minority classes to be oversampled
        # vs the labeled class distribution.
        # Setting input_score_floor=0.05 keeps only "candidate" scores in
        # the buffer so the quantile reflects real teacher predictions.
        self.input_score_floor = float(input_score_floor)

        quantiles = _as_int_keyed_float_dict(target_quantile_per_class)
        self.target_quantile_per_class: Dict[int, float] = {
            c: float(quantiles.get(c, self.default_quantile))
            for c in range(self.num_classes)
        }
        init = _as_int_keyed_float_dict(init_thresholds)
        self.init_thresholds: Dict[int, float] = {
            c: float(init.get(c, self.default_threshold))
            for c in range(self.num_classes)
        }

        self._history: Dict[int, Deque[float]] = {
            c: deque(maxlen=self.history_size) for c in range(self.num_classes)
        }
        self._cached_thresholds: Dict[int, float] = dict(self.init_thresholds)
        self._calls_since_recompute = 0
        self._total_updates = 0

    # ------------------------------------------------------------------
    # state introspection (used by checkpoint save/load if needed later)
    # ------------------------------------------------------------------
    def state_dict(self) -> Dict[str, Any]:
        return {
            "history": {c: list(self._history[c]) for c in range(self.num_classes)},
            "cached_thresholds": dict(self._cached_thresholds),
            "total_updates": int(self._total_updates),
        }

    def load_state_dict(self, state: Dict[str, Any]) -> None:
        history = state.get("history") or {}
        for c in range(self.num_classes):
            self._history[c].clear()
            for value in history.get(c, []):
                self._history[c].append(float(value))
        cached = state.get("cached_thresholds") or {}
        for c in range(self.num_classes):
            if c in cached:
                self._cached_thresholds[c] = float(cached[c])
        self._total_updates = int(state.get("total_updates", 0))

    # ------------------------------------------------------------------
    # main API
    # ------------------------------------------------------------------
    @torch.no_grad()
    def update(self, teacher_logits: torch.Tensor) -> None:
        """Push the teacher's per-query top scores into the per-class history.

        ``teacher_logits`` is the raw decoder logits of shape
        ``(batch, queries, num_classes)``. We use the argmax label and its
        sigmoid score, so each query contributes exactly one observation to a
        single class bucket -- the same accounting the mining code performs.
        """
        if not self.enabled or teacher_logits.numel() == 0:
            return
        probs = teacher_logits.sigmoid()
        scores, labels = probs.max(dim=-1)
        flat_scores = scores.flatten().detach().cpu()
        flat_labels = labels.flatten().detach().cpu()
        # Bug #1 fix: filter out background-noise argmax queries BEFORE the
        # per-class quantile estimate. Without this filter, the buffer is
        # dominated by ~290 noise queries per image (out of 300), and the
        # 0.85-0.95 quantile reflects "noise upper tail" not "real detection
        # lower tail". See the class header docstring for details.
        if self.input_score_floor > 0.0:
            keep = flat_scores >= self.input_score_floor
            if not bool(keep.any()):
                # Still increment the update counter so the recompute timer
                # fires eventually; just nothing to add this step.
                self._total_updates += 1
                self._calls_since_recompute += 1
                if self._calls_since_recompute >= self.update_interval:
                    self._recompute()
                    self._calls_since_recompute = 0
                return
            flat_scores = flat_scores[keep]
            flat_labels = flat_labels[keep]
        for cls_id in range(self.num_classes):
            mask = flat_labels == cls_id
            if not mask.any():
                continue
            values = flat_scores[mask].tolist()
            buf = self._history[cls_id]
            for v in values:
                buf.append(float(v))
        self._total_updates += 1
        self._calls_since_recompute += 1
        if self._calls_since_recompute >= self.update_interval:
            self._recompute()
            self._calls_since_recompute = 0

    def _recompute(self) -> None:
        for cls_id in range(self.num_classes):
            history = self._history[cls_id]
            if len(history) < self.min_history_for_quantile:
                self._cached_thresholds[cls_id] = float(self.init_thresholds[cls_id])
                continue
            quantile = self.target_quantile_per_class[cls_id]
            quantile = min(max(quantile, 0.0), 1.0)
            tensor = torch.as_tensor(list(history), dtype=torch.float32)
            thr = float(torch.quantile(tensor, q=quantile).item())
            thr = max(self.min_thresh, min(self.max_thresh, thr))
            self._cached_thresholds[cls_id] = thr

    def get_thresholds(self) -> Dict[int, float]:
        if not self.enabled:
            return dict(self.init_thresholds)
        return dict(self._cached_thresholds)

    def diagnostics(self) -> Dict[str, float]:
        """Return scalar metrics suitable for emitting alongside training logs."""
        thresholds = self.get_thresholds()
        out: Dict[str, float] = {
            "adaptive_threshold_total_updates": float(self._total_updates),
        }
        for cls_id, thr in thresholds.items():
            out[f"adaptive_threshold_class_{cls_id}"] = float(thr)
            out[f"adaptive_history_size_class_{cls_id}"] = float(len(self._history[cls_id]))
        return out
