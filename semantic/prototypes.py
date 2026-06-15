"""Prototype matching for rare damage subtype and severity prediction."""

from __future__ import annotations

from collections import Counter

import torch
import torch.nn.functional as F

from .taxonomy import DAMAGE_TYPES, SEVERITY_LEVELS


def majority(values: list[str], default: str) -> str:
    return Counter(values).most_common(1)[0][0] if values else default


def defected_prior(rows: list[dict]) -> dict:
    positive = [row for row in rows if str(row.get("damage_present", "")).lower() in {"yes", "true", "1"}]
    return {
        "opening_state": majority([row.get("opening_state", "closed") for row in positive], "closed"),
        "damage_type": majority([row.get("damage_type", "malformed_or_other") for row in positive], "malformed_or_other"),
        "damage_severity": majority([row.get("damage_severity", "moderate") for row in positive], "moderate"),
    }


def build_positive_prototypes(payload: dict) -> tuple[dict[int, torch.Tensor], dict[int, torch.Tensor]]:
    features = F.normalize(payload["features"], dim=1)
    type_proto: dict[int, torch.Tensor] = {}
    severity_proto: dict[int, torch.Tensor] = {}
    for cls in range(1, len(DAMAGE_TYPES)):
        idx = (payload["damage_type"] == cls).nonzero(as_tuple=False).flatten()
        if len(idx) > 0:
            type_proto[cls] = F.normalize(features[idx].mean(0, keepdim=True), dim=1)[0]
    for cls in range(1, len(SEVERITY_LEVELS)):
        idx = (payload["severity"] == cls).nonzero(as_tuple=False).flatten()
        if len(idx) > 0:
            severity_proto[cls] = F.normalize(features[idx].mean(0, keepdim=True), dim=1)[0]
    return type_proto, severity_proto


def nearest_name(feature: torch.Tensor, prototypes: dict[int, torch.Tensor], names: list[str]) -> tuple[str, float]:
    if not prototypes:
        return names[0], 0.0
    best_cls, best_proto = max(prototypes.items(), key=lambda item: float((feature * item[1]).sum().item()))
    return names[int(best_cls)], float((feature * best_proto).sum().item())
