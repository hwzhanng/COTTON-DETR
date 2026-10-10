<div align="center">

# COTTON-DETR

### DINOv3-Driven Label-Efficient In-Field Cotton Boll State Detection: From Cold-Start Selection to Semantic Interpretation

**One self-supervised model family for image selection, pseudo-label verification, detection, and ROI semantic interpretation.**

[English](README.md) · [中文](README.zh-CN.md) · [Interactive demo](https://huggingface.co/spaces/DEFF819/COTTON-DETR) · [Reproducibility](REPRODUCIBILITY.md)

[![License: CC BY-NC 4.0](https://img.shields.io/badge/License-CC_BY--NC_4.0-lightgrey.svg)](LICENSE)

</div>

![DINOv3-driven cotton boll monitoring pipeline](docs/assets/cotton_detr_pipeline.png)

*Manuscript Figure 1: data selection, independently verified teacher–student learning, and detection with ROI semantic interpretation.*

## Overview

COTTON-DETR connects **data construction, training reliability, object detection, and output interpretation** for cotton boll monitoring under limited annotations, long-tailed class distributions, and changing field conditions. It reuses the DINOv3 model family across four roles, with different model sizes and prediction heads for each stage.

Two selection rounds annotate **608 of 3,982 training images**, leaving **3,374 unlabeled images** for semi-supervised training. The clearest performance benefit is **cross-domain generalization**: combined external-domain mAP increases from **0.462 to 0.548** over the same-architecture supervised baseline. Independent verification mitigates rare-class degradation during self-training. After detection, frozen ROI features support predictions of opening and damage attributes.

The [Hugging Face demo](https://huggingface.co/spaces/DEFF819/COTTON-DETR) provides **20 precomputed examples**, interactive boxes and semantic attributes, English/Chinese switching, and exports. It currently **does not run inference on new uploads**. Curated examples are qualitative illustrations, not estimates of full-test-set performance.

## Four roles of DINOv3

| Stage | Representation | Role |
|---|---|---|
| **Select** | Frozen ViT-B/16 global features | Select 500 coverage images without a task detector, then add 108 images through rare-class supplementation guided by first-round human labels. |
| **Verify** | Frozen ViT-S/16 ROI features | Independently classify teacher proposals and filter pseudo-labels during semi-supervised training. |
| **Detect** | DINOv3-distilled ViT-Tiny | DEIMv2-DINOv3-S predicts boxes and four detection classes; the detector has 9.69M parameters. |
| **Interpret** | Frozen ViT-S/16 ROI features | A balanced MLP predicts opening attributes; training-feature prototypes predict damage subtype and severity. |

Detection classes are **Defected, Flower, Fully opened, and Partially opened**. Flower denotes the floral organ. **Closed** is a semantic opening state, not a fifth detection class. The four stages do not share one identical network or checkpoint.

### Independent verification during training

![Detector architecture and frozen verification branch](docs/assets/cotton_detr_architecture.png)

*Manuscript Figure 7. Upper panel: detector. Lower panel: frozen training-time verifier.*

The EMA teacher proposes boxes from weakly augmented unlabeled images. Frozen DINOv3 ROI features and calibrated classification heads determine class identity and acceptance independently of teacher–student updates. The student learns from manual annotations and accepted pseudo-labels on strongly augmented images.

**Verifier posteriors determine the accepted class; teacher confidence and class weights determine its loss contribution.** The pseudo-label verifier is a training component. At inference, the detector produces boxes and classes, followed by ROI semantic prediction.

## Main evidence

Detection comparisons below report **mean ± standard deviation over three seeds**, unless stated otherwise. COCO mAP is AP@[0.50:0.05:0.95]. These are manuscript results, not scores calculated from the demo.

### Cross-domain generalization

![External-domain retention and gains across annotation budgets](docs/assets/cotton_detr_crossdomain.png)

*Figure 12. Asterisks identify single-run reference models; the matched-architecture comparison uses three seeds.*

| Model, 608 labeled images | OUT100 mAP | GH80 mAP | Combined mAP | Retention ratio |
|---|---:|---:|---:|---:|
| Same-architecture supervised baseline | 0.477 ± 0.026 | 0.397 ± 0.046 | 0.462 ± 0.034 | 0.780 ± 0.061 |
| **COTTON-DETR** | **0.570 ± 0.008** | **0.472 ± 0.027** | **0.548 ± 0.005** | **0.908 ± 0.016** |

*Table 4. Retention = external mAP / in-domain mAP.*

The external evaluation covers **180 images and 510 boxes**: OUT100 has 100 outdoor images, and GH80 has 80 greenhouse images with no flowers. The **+0.086 cross-domain gain** is about seven times the **+0.012 in-domain gain** at the same budget. This evaluates transfer across acquisition conditions within the cotton task; cross-crop generalization is not established.

### Rare-class supervision and protection

| Comparison at 608 labels | Defective-boll F1 | Interpretation |
|---|---:|---|
| Self-training without → with the verifier | 0.561 ± 0.043 → **0.606 ± 0.013** | Mitigates rare-class degradation and reduces training-seed variability. |
| Random selection → DINOv3 cold-start selection | 0.591 ± 0.055 → **0.618 ± 0.005** | Supervised sampling ablation: three random subsets versus three training seeds on one fixed DINOv3-selected subset. |

*Table 3. These are separate ablations, not successive gains from one experiment.*

The verifier improves F1 relative to unverified self-training, but **does not exceed the supervised F1 of 0.618** at this budget. The sampling rows have different sources of variability and should not be interpreted as a matched repeated-subset variance test.

### Annotation budgets and corruption robustness

![Three-seed label-efficiency and robustness comparison](docs/assets/cotton_detr_label_efficiency.png)

*Figure 8. In-domain gains concentrate at the lowest budget; corruption robustness improves at all three budgets.*

| Labels | Supervised mAP | COTTON-DETR mAP | Reported ΔmAP | Supervised rPC | COTTON-DETR rPC |
|---:|---:|---:|---:|---:|---:|
| 152 | 0.534 ± 0.008 | 0.578 ± 0.005 | +0.043 | 0.790 ± 0.004 | 0.824 ± 0.008 |
| 304 | 0.575 ± 0.004 | 0.585 ± 0.007 | +0.010 | 0.792 ± 0.005 | 0.846 ± 0.004 |
| 608 | 0.592 ± 0.009 | 0.604 ± 0.005 | +0.012 | 0.835 ± 0.015 | 0.856 ± 0.005 |

*Table 2. Deltas use unrounded results. Architecture, annotated subsets, and augmentation are matched; COTTON-DETR additionally uses 3,374 unlabeled images. The test set contains 523 images.*

`mPC` averages mAP over 15 corruption types × 5 severity levels; `rPC = mPC / clean mAP`. Higher mAP does not imply higher F1 for every class. The manuscript's paired tests use three seeds and are not adjusted for multiple comparisons.

<details>
<summary>Corruption comparison across architectures</summary>

![Representative single-run corruption comparison](docs/assets/cotton_detr_robustness.png)

*Figure 10 reports representative single runs, not the three-seed means in Table 2.* DINOv3-based detectors retain high performance under synthetic corruption; the matched-architecture comparison isolates the smaller additional contribution of semi-supervised learning. Synthetic corruption is a controlled degradation benchmark, not a substitute for real domain shifts.

</details>

## ROI semantic interpretation

The semantic component turns each ROI into **opening state, opening percentage, damage subtype, severity, and a structured description**:

1. Frozen DINOv3 ViT-S/16 encodes the crop.
2. A class-balanced MLP predicts opening state and percentage. State-dependent clipping uses 0% for flowers, 0–15% for closed, 15–65% for partially opened, and 65–100% for fully opened bolls.
3. Damage presence follows the detection class. For defective bolls, subtype and severity use nearest positive training-feature prototypes.
4. Text is assembled deterministically from attributes and templates. **Gemini 3 Flash provides offline supervision; inference does not call a VLM or generate free-form text.**

Only the **1,203 annotated ROIs from the 608 training images** train the semantic heads and construct the prototypes. Opening percentage estimates visible exposed lint per ROI, not the proportion of open bolls in an image or field.

### Attribute agreement and evaluation scope

Table 5 evaluates **all 1,026 ground-truth ROIs in 523 test images** against Gemini attributes. Ground-truth locations and classes supply the class-based decisions; this measures attributes given known objects, not end-to-end detection.

| Component | Opening-state macro-F1 ↑ | Opening-% MAE ↓ | Severity macro-F1, 4 classes ↑ |
|---|---:|---:|---:|
| Deterministic class map | 0.715 | 8.80 | 0.449 |
| Original multi-head MLP | 0.794 | 4.96 | 0.448 |
| Fixed opening + damage majority | 0.794 | 4.65 | 0.449 |
| **Full semantic component** | **0.794** | **4.65** | **0.610** |

MAE is in **percentage points**. The majority-class control holds opening predictions, clipping, and damage-presence decisions fixed. Comparison with the full component isolates the contribution of damage prototypes: severity macro-F1 increases from 0.449 to 0.610 (approximately +0.162 before rounding).

![Opening and damage agreement with Gemini reference labels](docs/assets/cotton_detr_semantic.png)

*Figure 13. Rows are references; columns are predictions. Cells show counts and row proportions.*

- **All test ROIs:** Table 5 severity macro-F1 is **0.610 across four classes**, including 935 undamaged objects.
- **Defective ROIs only:** Figure 13 uses 91 defective test ROIs for damage panels. Severity macro-F1 is **0.481 across mild, moderate, and severe**; this is a different metric from Table 5.
- **Correctly detected objects only:** at confidence 0.65, 891 correct-class matches with IoU ≥ 0.5 yield opening/severity macro-F1 of **0.848 / 0.699**. This excludes 450 false positives and 135 false negatives and is not an end-to-end semantic score.

These metrics measure agreement with VLM labels. Independent expert annotation and agronomic calibration remain future work. Damage presence follows a class constraint, and prototype similarity is not a probability or an expert diagnosis.

## Code and reproducibility

```text
configs/cotton_detr/     Public SSOD and semantic configurations
engine/                 Detector and semi-supervised runtime
selection/              Cold-start selection algorithms
semantic/               ROI semantic encoding library
tools/selection/        Feature extraction and selection CLIs
tools/verifier/         Verifier training and calibration
tools/semantic/         Semantic training, evaluation, and export
tools/evaluation/       Corruption evaluation utilities
prompts/                ROI attribute-labeling prompts
scripts/                Reproducibility preflight checks
third_party/licenses/   Upstream license notices
```

For training, use Linux with an NVIDIA GPU. Install PyTorch/torchvision for your CUDA environment before the remaining dependencies:

```bash
git clone https://github.com/hwzhanng/COTTON-DETR.git
cd COTTON-DETR
python -m venv .venv
source .venv/bin/activate
# Install a compatible torch/torchvision build first.
pip install -r requirements-repro.txt
python scripts/check_repro_package.py \
  --config configs/cotton_detr/cotton_detr_dinov3_tiny_ssod.yml
```

[REPRODUCIBILITY.md](REPRODUCIBILITY.md) provides selection, verifier calibration, training, EMA-teacher evaluation, data layout, and semantic commands. The manuscript's **global cold-start selection uses ViT-B/16**; **ViT-S/16 is used for verifier/semantic ROIs**. Configure the corresponding backbone for each stage.

This Git checkout contains code, configuration templates, prompts, and selected figures. Data and checkpoint files are not bundled here; consult [data notes](data/README.md) and [weight setup](weights/pretrained/README.md). Original DINOv3 weights must be obtained under their upstream access and license terms. The full manuscript PDF, raw VLM responses, and local experiment archives are not included.

## Manuscript and citation

**DINOv3-Driven Label-Efficient In-Field Cotton Boll State Detection: From Cold-Start Selection to Semantic Interpretation**

The public paper URL and complete citation metadata will be added when available. See [figure sources](docs/assets/README.md) for the manuscript figures used in this repository.

## License and acknowledgements

Original COTTON-DETR additions and release materials retain the repository's [CC BY-NC 4.0 license](LICENSE). Non-commercial use requires attribution; commercial use requires prior written permission from the authors.

DEIM/DEIMv2, D-FINE, RT-DETR, and DINOv3 retain their upstream terms. See [NOTICE.md](NOTICE.md), [OPEN_SOURCE_AUDIT.md](OPEN_SOURCE_AUDIT.md), and [third-party licenses](third_party/licenses/). We thank their authors for the underlying methods and implementations.
