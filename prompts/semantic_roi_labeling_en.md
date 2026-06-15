# Cotton Boll ROI Semantic Labeling Prompt

Purpose: generate ROI-level structured labels for the CROP-DETR semantic sidecar. The input should be a crop of one target cotton boll or flower, or a local image with a clearly marked target box. Judge only the target object inside the box.

You are an annotator for in-field cotton imagery. Use visible evidence to classify the target object's opening state, damage type, and damage severity.

Return strict JSON only. Do not return Markdown, prose explanations, or extra fields.

```json
{
  "opening_state": "flower | closed | partial | full",
  "damage_present": "yes | no",
  "damage_type": "none | insect_or_feeding | mold_or_rot | dry_shriveled | mechanical_crack | stain_or_contamination | malformed_or_other",
  "damage_severity": "none | mild | moderate | severe",
  "opening_percentage": 0,
  "visual_description": "30-80 English words describing visible evidence only.",
  "confidence": 1,
  "needs_human_review": false,
  "evidence": "Short reason based on visible evidence."
}
```

## Labeling Rules

- `flower` means the target is a cotton flower, not a boll.
- `closed` means the boll is mostly closed and white lint is absent or barely visible.
- `partial` means the boll is opened, but exposed white lint is clearly below a fully opened state.
- `full` means white cotton lint is broadly exposed and the boll is close to harvestable.
- Set `damage_present=yes` only when the target object shows a clear agronomic abnormality, such as insect feeding, mold or rot, shriveling, mechanical cracking, staining or contamination, or malformation.
- Do not mark minor shadows, dust, natural color variation, or background clutter as damage.
- If `damage_present=no`, set `damage_type=none` and `damage_severity=none`.
- If uncertain, lower `confidence` and set `needs_human_review=true`.

## `opening_percentage`

- `flower`: 0.
- `closed`: 0-15.
- `partial`: 15-65.
- `full`: 65-100.

## `visual_description`

- Write a natural English description of visible evidence.
- Do not mechanically restate the structured labels.
- Avoid the words `flower`, `full`, `partial`, `closed`, `damage`, and `damaged`.
- Mention visible morphology, lint exposure, color or texture abnormality, and local position when relevant.
