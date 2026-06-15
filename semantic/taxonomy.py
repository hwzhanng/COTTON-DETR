"""Semantic vocabularies and deterministic class-to-attribute priors."""

from __future__ import annotations

OPENING_STATES = ["flower", "closed", "partial", "full"]
DAMAGE_TYPES = [
    "none",
    "insect_or_feeding",
    "mold_or_rot",
    "dry_shriveled",
    "mechanical_crack",
    "stain_or_contamination",
    "malformed_or_other",
]
SEVERITY_LEVELS = ["none", "mild", "moderate", "severe"]

OPENING_TO_ID = {name: idx for idx, name in enumerate(OPENING_STATES)}
DAMAGE_TYPE_TO_ID = {name: idx for idx, name in enumerate(DAMAGE_TYPES)}
SEVERITY_TO_ID = {name: idx for idx, name in enumerate(SEVERITY_LEVELS)}

OPENING_PERCENT_RANGES = {
    "flower": (0.0, 0.0),
    "closed": (0.0, 15.0),
    "partial": (15.0, 65.0),
    "full": (65.0, 100.0),
}

DEFAULT_OPENING_PERCENT = {"flower": 0.0, "closed": 5.0, "partial": 40.0, "full": 85.0}


def clip_opening_percentage(value: float, opening_state: str) -> float:
    lo, hi = OPENING_PERCENT_RANGES.get(opening_state, (0.0, 100.0))
    return max(lo, min(hi, float(value)))


def normalize_damage_present(value: str | bool | int) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, int):
        return value != 0
    return str(value).lower() in {"yes", "true", "1", "damage", "damaged"}


def category_prior(category_id: int, category_semantics: dict, defected_prior: dict | None = None) -> dict:
    """Return deterministic semantic defaults from a detector category id.

    Non-defected categories usually have no damage attributes. Defected
    categories use a prior estimated from the training semantic rows unless a
    stronger user-provided prior is configured.
    """

    key = str(category_id)
    mapping = category_semantics.get(key) or category_semantics.get(int(category_id))
    if mapping is None:
        mapping = {"opening_state": "closed", "damage_present": True}
    opening = mapping.get("opening_state", "closed")
    damage_present = normalize_damage_present(mapping.get("damage_present", False))
    damage_type = mapping.get("damage_type", "none")
    severity = mapping.get("damage_severity", "none")
    if damage_present and defected_prior:
        opening = defected_prior.get("opening_state", opening)
        damage_type = defected_prior.get("damage_type", damage_type)
        severity = defected_prior.get("damage_severity", severity)
    opening_pct = mapping.get("opening_percentage", DEFAULT_OPENING_PERCENT.get(opening, 0.0))
    return {
        "opening_state": opening,
        "opening_percentage": clip_opening_percentage(opening_pct, opening),
        "damage_present": bool(damage_present),
        "damage_type": damage_type if damage_present else "none",
        "damage_severity": severity if damage_present else "none",
    }
