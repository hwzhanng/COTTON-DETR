"""Schema and consistency checks for semantic ROI labels."""

from __future__ import annotations

from typing import Any

OPENING_RANGES = {
    "flower": (0, 0),
    "closed": (0, 15),
    "partial": (15, 65),
    "full": (65, 100),
}
OPENING_STATES = set(OPENING_RANGES)
DAMAGE_TYPES = {
    "none",
    "insect_or_feeding",
    "mold_or_rot",
    "dry_shriveled",
    "mechanical_crack",
    "stain_or_contamination",
    "malformed_or_other",
}
DAMAGE_SEVERITIES = {"none", "mild", "moderate", "severe"}
REQUIRED_FIELDS = {
    "opening_state",
    "opening_percentage",
    "damage_present",
    "damage_type",
    "damage_severity",
    "confidence",
    "needs_human_review",
}


def as_bool(value: Any) -> bool | None:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        lowered = value.strip().lower()
        if lowered in {"yes", "true", "1"}:
            return True
        if lowered in {"no", "false", "0"}:
            return False
    return None


def validate_semantic_row(row: dict, *, strict_opening_range: bool = True) -> list[str]:
    errors: list[str] = []
    missing = sorted(field for field in REQUIRED_FIELDS if field not in row)
    if missing:
        errors.append(f"missing_fields={missing}")
        return errors

    opening_state = str(row.get("opening_state")).strip()
    if opening_state not in OPENING_STATES:
        errors.append(f"invalid_opening_state={opening_state!r}")

    try:
        opening_percentage = float(row.get("opening_percentage"))
    except (TypeError, ValueError):
        errors.append("opening_percentage_not_numeric")
        opening_percentage = None
    if opening_percentage is not None:
        if opening_percentage < 0 or opening_percentage > 100:
            errors.append(f"opening_percentage_out_of_range={opening_percentage}")
        if strict_opening_range and opening_state in OPENING_RANGES:
            lo, hi = OPENING_RANGES[opening_state]
            if not (lo <= opening_percentage <= hi):
                errors.append(
                    f"opening_percentage_inconsistent={opening_state}:{opening_percentage}"
                )

    damage_present = as_bool(row.get("damage_present"))
    if damage_present is None:
        errors.append(f"invalid_damage_present={row.get('damage_present')!r}")
    damage_type = str(row.get("damage_type")).strip()
    damage_severity = str(row.get("damage_severity")).strip()
    if damage_type not in DAMAGE_TYPES:
        errors.append(f"invalid_damage_type={damage_type!r}")
    if damage_severity not in DAMAGE_SEVERITIES:
        errors.append(f"invalid_damage_severity={damage_severity!r}")
    if damage_present is False and (damage_type != "none" or damage_severity != "none"):
        errors.append("damage_absent_but_type_or_severity_non_none")
    if damage_present is True and (damage_type == "none" or damage_severity == "none"):
        errors.append("damage_present_but_type_or_severity_none")

    try:
        confidence = int(row.get("confidence"))
    except (TypeError, ValueError):
        errors.append("confidence_not_integer")
        confidence = None
    if confidence is not None and not (1 <= confidence <= 5):
        errors.append(f"confidence_out_of_range={confidence}")
    if not isinstance(row.get("needs_human_review"), bool):
        errors.append("needs_human_review_not_bool")
    return errors


def review_priority(row: dict, errors: list[str]) -> bool:
    confidence = row.get("confidence", 5)
    try:
        low_confidence = int(confidence) <= 2
    except (TypeError, ValueError):
        low_confidence = True
    return bool(errors or row.get("needs_human_review") is True or low_confidence)
