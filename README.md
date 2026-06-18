# CROP-DETR

**Detector-free cold-start selection and DINOv3-verified semi-supervised DETR
for low-label, long-tail cotton boll-state detection.**

[English](README.md) | [中文](README.zh-CN.md)

<p align="center">
  <img src="docs/assets/figure_1_pipeline.png" alt="CROP-DETR pipeline" width="900">
</p>

CROP-DETR is the code release for a label-efficient field cotton detection
pipeline. The central result of the paper is not a single module, but a reusable
design pattern: the same family of self-supervised DINOv3 representations can
support four stages of a long-tail agricultural detection workflow.

1. **Cold-start selection** chooses which images to label before any detector is
   trained.
2. **Pseudo-label verification** filters teacher predictions with a frozen
   DINOv3 verifier instead of trusting detector confidence alone.
3. **DINOv3-distilled detection** provides a compact DETR backbone with strong
   corruption robustness.
4. **Post-detection semantic encoding** turns detected bolls into structured
   agronomic attributes.

The public repository contains code, sanitized configs, reproducibility
instructions, and web-optimized paper figures. It intentionally does **not**
include datasets, pretrained weights, verifier heads, checkpoints, training
logs, private paper drafts, or raw experiment archives.

## Pipeline Walkthrough

The repository follows the paper pipeline in execution order.

**1. Select images before training a detector.** Extract frozen DINOv3 features
for the candidate training pool, select a 500-image coverage head for
annotation, then expand from the labeled rare-class seed clusters to 608 images.

```bash
python tools/selection/extract_dinov3_features.py \
  --image-dir data/candidate/images \
  --model-dir weights/pretrained/dinov3_vits16 \
  --output-dir outputs/selection/features

python tools/selection/select_cold_start.py \
  --method cluster_head \
  --features outputs/selection/features/dinov3_features.npy \
  --paths outputs/selection/features/dinov3_paths.npy \
  --candidate-coco data/annotations/candidate_train.json \
  --exclude-coco data/annotations/val.json \
  --exclude-coco data/annotations/test.json \
  --budget 500 \
  --head-budget 500 \
  --n-clusters 50 \
  --per-cluster 10

python tools/selection/select_cold_start.py \
  --method cluster_tail \
  --features outputs/selection/features/dinov3_features.npy \
  --paths outputs/selection/features/dinov3_paths.npy \
  --candidate-coco data/annotations/candidate_train.json \
  --exclude-coco data/annotations/val.json \
  --exclude-coco data/annotations/test.json \
  --output-coco-from data/annotations/train_pool_with_boxes.json \
  --seed-coco data/annotations/coverage_head_labeled.json \
  --budget 608 \
  --head-budget 500 \
  --n-clusters 50 \
  --per-cluster 10 \
  --target-class-id 0 \
  --target-box-count 90
```

`random` and `kcenter` baselines are available from the same selector. The
`cluster_head` command needs no labels; `cluster_tail` uses only labels from
the already selected coverage head. See [tools/selection](tools/selection/) for
the exact options and leakage checks.

**2. Train CROP-DETR with labeled and unlabeled images.** Put the selected
labels under `data/annotations/labeled_train.json`, keep the remaining pool as
`unlabeled_train.json`, then use the public SSOD template.

**3. Evaluate the teacher EMA checkpoint.** Use
`tools/evaluate_ssod_teacher.py` for the paper-style teacher branch metrics.

**4. Add semantic attributes after detection.** The optional semantic sidecar
uses ROI crops and open prompt-generated labels to attach crack and damage
attributes to detections.

## Paper Takeaways

| Question | Paper finding |
|---|---|
| Where does semi-supervision help most? | mAP gains concentrate at the lowest label budget: **+0.043 mAP at 152 labeled images** with paired-test significance; gains shrink to about +0.01 at 304/608 labeled images. |
| Is the robustness gain real? | Yes. mPC/rPC improve at every budget; at 608 labeled images, CROP-DETR reaches **0.517 mPC / 0.856 rPC** versus supervised **0.494 / 0.835**. |
| Do generic SSOD methods transfer? | Not reliably in this long-tail setting. At 608 labeled images, Soft Teacher and Semi-DETR reach **0.541/0.547 mAP**, below the supervised anchor **0.592**, while CROP-DETR reaches **0.604**. |
| What improves the rare defective-boll class? | The gain comes from both ends: DINOv3 cold-start selection stabilizes defective F1 (**0.618±0.005** vs random **0.591±0.055**), and the verifier adds **+0.092 defective F1** over detector-score filtering. |
| Can detections carry more agronomic meaning? | The semantic encoder raises damage-severity consistency F1 from **0.448** to **0.611** under the paper's VLM-label agreement protocol. |

## Main Results

<p align="center">
  <img src="docs/assets/figure_3_label_efficiency.png" alt="Label efficiency and robustness across budgets" width="850">
</p>

Under augmentation-aligned comparison, the only difference between the
supervised anchor and CROP-DETR is the use of 3,374 unlabeled images. All rows
below are three-seed mean ± standard deviation on the 523-image test set.

| Method | Labeled images | mAP ↑ | AP50 ↑ | mPC / rPC ↑ | Defective F1 ↑ | Macro F1 ↑ |
|---|---:|---:|---:|---:|---:|---:|
| Supervised | 152 | 0.534±0.008 | 0.748±0.009 | 0.422±0.006 / 0.790±0.004 | 0.574±0.027 | 0.738±0.010 |
| CROP-DETR | 152 | **0.578±0.005** | **0.789±0.006** | **0.476±0.008 / 0.824±0.008** | **0.600±0.035** | **0.752±0.009** |
| Δ | 152 | **+0.043** | +0.041 | +0.054 / +0.035 | +0.026 | +0.014 |
| Supervised | 304 | 0.575±0.004 | **0.795±0.003** | 0.456±0.005 / 0.792±0.005 | **0.626±0.015** | **0.775±0.010** |
| CROP-DETR | 304 | **0.585±0.007** | 0.791±0.007 | **0.495±0.005 / 0.846±0.004** | 0.611±0.013 | 0.759±0.006 |
| Δ | 304 | +0.010 | -0.004 | +0.039 / +0.053 | -0.015 | -0.016 |
| Supervised | 608 | 0.592±0.009 | 0.804±0.013 | 0.494±0.004 / 0.835±0.015 | **0.618±0.005** | **0.792±0.010** |
| CROP-DETR | 608 | **0.604±0.005** | **0.812±0.003** | **0.517±0.003 / 0.856±0.005** | 0.606±0.013 | 0.766±0.004 |
| Δ | 608 | **+0.012** | +0.009 | +0.023 / +0.021 | -0.012 | -0.026 |

The result is deliberately narrow: unlabeled images improve mAP and robustness,
while class-level long-tail improvements are mainly delivered by cold-start
selection and verifier gating.

## Robustness

<p align="center">
  <img src="docs/assets/figure_4_robustness.png" alt="Corruption robustness" width="850">
</p>

The large robustness gap is primarily a backbone property. In the 608-image
setting, the DINOv3-based detector keeps about 85% of clean performance under
ImageNet-C-style corruptions, while CNN baselines retain about 51%. Within the
same DINOv3 architecture, semi-supervision contributes a smaller but consistent
+0.02 to +0.05 rPC gain across budgets.

## Why The Verifier Works

<p align="center">
  <img src="docs/assets/figure_5_verifier_basis.png" alt="DINOv3 verifier feature geometry" width="850">
</p>

The verifier is lightweight because the frozen DINOv3 feature space already
separates the four boll states well enough for a linear head. In the ablation,
detector-score filtering alone gives 0.592 mAP, 0.530 defective F1, and 0.764
macro F1. Adding the DINOv3 verifier raises these to 0.605, 0.622, and 0.790.

## Qualitative Examples

<p align="center">
  <img src="docs/assets/figure_6_qualitative.jpg" alt="Qualitative detection comparison" width="850">
</p>

CROP-DETR recovers defective and partially opened bolls that the supervised
anchor often misses on difficult long-tail images. The qualitative figure uses
the same 608-image setting as the main high-budget comparison.

## Semantic Encoding

<p align="center">
  <img src="docs/assets/figure_9_semantic.jpg" alt="Post-detection semantic encoding examples" width="850">
</p>

Beyond four-class detection, CROP-DETR reuses frozen DINOv3 ROI features to
predict crack state, crack ratio, damage subtype, and damage severity. This
module is intended as structured agronomic evidence, not as a replacement for
expert pathology diagnosis.

Open prompt templates for producing compatible ROI-level semantic labels are
provided in [Chinese](prompts/semantic_roi_labeling_zh.md) and
[English](prompts/semantic_roi_labeling_en.md).

Semantic sidecar training and export entry points are provided under
`tools/semantic/`:

```bash
python tools/semantic/train_semantic_head.py \
  --config configs/crop_detr/semantic_sidecar_template.yaml

python tools/semantic/export_semantics.py \
  --config configs/crop_detr/semantic_sidecar_template.yaml \
  --detections outputs/crop_detr_ssod/test_detections.json \
  --coco-images data/annotations/test.json \
  --image-dir data/test/images \
  --output outputs/crop_detr_ssod/test_detections_with_semantics.json
```

## Repository Layout

```text
configs/crop_detr/                       sanitized public SSOD template
engine/                                  model, criterion, solver, verifier, SSOD code
selection/                               reusable cold-start selection algorithms
tools/selection/                         DINOv3 feature extraction and selection CLIs
semantic/                                semantic sidecar library
tools/semantic/                          semantic sidecar train/export tools
tools/evaluate_ssod_teacher.py           teacher-EMA evaluation entry point
scripts/check_repro_package.py           data/weight/config preflight checker
docs/assets/                             web-optimized paper figures for README
prompts/                                 open prompts for ROI semantic labeling
REPRODUCIBILITY.md                       rerun protocol and reporting checklist
third_party/licenses/                    upstream license copies
```

## What You Need To Provide

- COCO-format labeled annotations.
- COCO-format unlabeled image records. Unlabeled records may omit boxes.
- COCO-format validation annotations.
- A candidate training image pool for cold-start selection.
- Optional seed-head annotations for rare-class tail expansion.
- DEIM/DINOv3 backbone weights accepted under their upstream licenses.
- Frozen DINOv3 verifier weights:
  - `weights/pretrained/dinov3_vits16/`
  - `weights/pretrained/verifier_head.pt`
  - `weights/pretrained/verifier_head_4cls.pt`

The public template uses neutral verifier thresholds and class weights. Tune
`num_classes`, `accept_threshold`, `class_loss_scale`, and
`class_balanced_sampling` for your dataset.

## Quick Start

```bash
pip install -r requirements.txt
python scripts/check_repro_package.py \
  --config configs/crop_detr/crop_detr_dinov3_tiny_ssod.yml

bash train_guarded.sh \
  -c configs/crop_detr/crop_detr_dinov3_tiny_ssod.yml \
  -t weights/pretrained/deimv2_dinov3_s_coco_tuning.pth \
  -d cuda \
  --seed 0 \
  --use-amp
```

Evaluate a trained SSOD checkpoint:

```bash
python tools/evaluate_ssod_teacher.py \
  -c configs/crop_detr/crop_detr_dinov3_tiny_ssod.yml \
  --checkpoint outputs/crop_detr_ssod/best_teacher_ema.pth \
  --output-dir outputs/crop_detr_ssod/eval \
  -d cuda
```

For a strict environment, seed plan, expected data layout, and result archiving
checklist, see [REPRODUCIBILITY.md](REPRODUCIBILITY.md).

## License And Notices

Original CROP-DETR additions are released under the Apache License 2.0 in
[LICENSE](LICENSE). Third-party derived components remain under upstream terms:
DEIM, D-FINE, and RT-DETR are Apache-2.0; DINOv3 code and model materials are
under the Meta DINOv3 License. Copies are stored in
[third_party/licenses/](third_party/licenses/).

This repository does not redistribute datasets, pretrained weights, verifier
heads, checkpoints, or raw experiment outputs. Users are responsible for
obtaining and using external model weights and datasets under their applicable
licenses.

## Citation

The manuscript title and citation metadata are not finalized yet. Add the paper
citation here before public release.
