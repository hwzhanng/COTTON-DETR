# Semantic Sidecar Tools

This directory contains the open-source semantic encoding part of COTTON-DETR.
It trains a lightweight head on frozen DINOv3 ROI features and can attach
structured attributes to detector outputs.

The tools expect user-provided ROI semantic labels in JSONL format. They do not
include or call Gemini, do not include manual-audit UI code, and do not ship any
paper data, outputs, checkpoints, crops, or feature caches.

Open prompt templates for producing compatible ROI labels are available in:

- `prompts/semantic_roi_labeling_zh.md`
- `prompts/semantic_roi_labeling_en.md`

Minimum JSONL row schema:

```json
{
  "ann_id": 1,
  "image_id": 10,
  "file_name": "example.jpg",
  "category_id": 0,
  "bbox": [100.0, 80.0, 42.0, 50.0],
  "crop_path": "data/semantic/crops/ann_000001.jpg",
  "opening_state": "full",
  "opening_percentage": 85,
  "damage_present": true,
  "damage_type": "stain_or_contamination",
  "damage_severity": "mild",
  "visual_description": "A concise English description of visible evidence.",
  "confidence": 4,
  "needs_human_review": false,
  "evidence": "Short visible-evidence rationale."
}
```

The training script uses the structured semantic targets. The validation script
also checks quality-control fields reported in the paper, including confidence
and review priority.

Train:

```bash
python tools/semantic/train_semantic_head.py \
  --config configs/cotton_detr/semantic_sidecar_template.yaml
```

Validate labels before training:

```bash
python tools/semantic/validate_semantic_labels.py \
  data/semantic/semantic_labels_train.jsonl \
  --output outputs/semantic/label_schema_report.json \
  --fail-on-error
```

Attach semantics to detections:

```bash
python tools/semantic/export_semantics.py \
  --config configs/cotton_detr/semantic_sidecar_template.yaml \
  --detections outputs/cotton_detr_ssod/test_detections.json \
  --coco-images data/annotations/test.json \
  --image-dir data/test/images \
  --output outputs/cotton_detr_ssod/test_detections_with_semantics.json
```
