"""Post-detection semantic encoding utilities for COTTON-DETR."""

from .taxonomy import DAMAGE_TYPES, OPENING_STATES, SEVERITY_LEVELS

__all__ = [
    "DAMAGE_TYPES",
    "OPENING_STATES",
    "SEVERITY_LEVELS",
    "HierarchicalSemanticHead",
    "SemanticHead",
]


def __getattr__(name: str):
    if name in {"HierarchicalSemanticHead", "SemanticHead"}:
        from .model import HierarchicalSemanticHead, SemanticHead

        return {
            "HierarchicalSemanticHead": HierarchicalSemanticHead,
            "SemanticHead": SemanticHead,
        }[name]
    raise AttributeError(name)
