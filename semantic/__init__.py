"""Post-detection semantic encoding utilities for CROP-DETR."""

from .taxonomy import DAMAGE_TYPES, OPENING_STATES, SEVERITY_LEVELS
from .model import HierarchicalSemanticHead, SemanticHead

__all__ = [
    "DAMAGE_TYPES",
    "OPENING_STATES",
    "SEVERITY_LEVELS",
    "HierarchicalSemanticHead",
    "SemanticHead",
]
