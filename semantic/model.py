"""Semantic sidecar heads trained on frozen ROI features."""

from __future__ import annotations

from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

from .taxonomy import DAMAGE_TYPES, OPENING_STATES, SEVERITY_LEVELS


class SemanticHead(nn.Module):
    def __init__(self, in_dim: int, hidden_dim: int = 384, embed_dim: int = 256, dropout: float = 0.1):
        super().__init__()
        self.trunk = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.opening = nn.Linear(hidden_dim, len(OPENING_STATES))
        self.opening_pct = nn.Linear(hidden_dim, 1)
        self.damage = nn.Linear(hidden_dim, 1)
        self.damage_type = nn.Linear(hidden_dim, len(DAMAGE_TYPES))
        self.severity = nn.Linear(hidden_dim, len(SEVERITY_LEVELS))
        self.semantic_embed = nn.Linear(hidden_dim, embed_dim)

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        h = self.trunk(x)
        return {
            "opening": self.opening(h),
            "opening_pct": torch.sigmoid(self.opening_pct(h)).squeeze(-1),
            "damage": self.damage(h).squeeze(-1),
            "damage_type": self.damage_type(h),
            "severity": self.severity(h),
            "semantic_embed": F.normalize(self.semantic_embed(h), dim=-1),
        }


class HierarchicalSemanticHead(SemanticHead):
    """Predict non-none damage type/severity only when damage is present."""

    def __init__(self, in_dim: int, hidden_dim: int = 384, embed_dim: int = 256, dropout: float = 0.1):
        super().__init__(in_dim, hidden_dim, embed_dim, dropout)
        self.damage_type = nn.Linear(hidden_dim, len(DAMAGE_TYPES) - 1)
        self.severity = nn.Linear(hidden_dim, len(SEVERITY_LEVELS) - 1)


def build_head(feature_dim: int, model_cfg: dict, head_type: str = "flat") -> nn.Module:
    cls = HierarchicalSemanticHead if head_type == "hierarchical" else SemanticHead
    return cls(
        in_dim=int(feature_dim),
        hidden_dim=int(model_cfg.get("hidden_dim", 384)),
        embed_dim=int(model_cfg.get("embed_dim", 256)),
        dropout=float(model_cfg.get("dropout", 0.1)),
    )


def load_checkpoint(path: str | Path, device: torch.device) -> tuple[nn.Module, dict]:
    ckpt = torch.load(path, map_location="cpu")
    model_cfg = ckpt["model"]
    model = build_head(int(ckpt["feature_dim"]), model_cfg, model_cfg.get("head_type", "flat"))
    model.load_state_dict(ckpt["state_dict"])
    model.to(device).eval()
    return model, ckpt
