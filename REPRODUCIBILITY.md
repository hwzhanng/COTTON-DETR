# Reproducibility

This repository is a code-only release. It provides the training recipe,
configuration, and checks needed to reproduce a CROP-DETR run, but it does not
include private datasets, pretrained weights, verifier heads, checkpoints, logs,
or paper result archives.

## Reference Code Version

The default public configuration is:

```text
configs/crop_detr/crop_detr_dinov3_tiny_ssod.yml
```

It corresponds to the v5 SSOD stack used for release preparation: DINOv3STAs
ViT-Tiny backbone, teacher-student SSOD, DINOv3 verifier gating, multi-class
verifier reassignment, and soft-background ignore. The v6 DINOv3-S experiment is
not the default public template.

## Environment

Use Linux with an NVIDIA GPU for training. The guarded launcher expects common
Linux tools such as `bash`, `ps`, `awk`, `find`, and `/dev/shm`.

Recommended baseline:

```bash
python -m venv .venv
source .venv/bin/activate

# Install the PyTorch wheel that matches your CUDA driver first.
# Example only:
pip install torch==2.5.1 torchvision==0.20.1 --index-url https://download.pytorch.org/whl/cu121

pip install -r requirements-repro.txt
```

For local development or dependency resolution, `requirements.txt` keeps looser
minimums. For a paper-style rerun, prefer `requirements-repro.txt` and record
the final `pip freeze`.

## Data Layout

The public template expects COCO-style data under:

```text
data/
  labeled/images/
  unlabeled/images/
  val/images/
  annotations/
    labeled_train.json
    unlabeled_train.json
    val.json
```

Unlabeled COCO records may omit bounding boxes. Keep category ids consistent
across labeled, unlabeled, and validation annotation files, then update
`num_classes`, verifier thresholds, and class weights in the config.

## Required Weights

Place externally obtained weights at:

```text
weights/pretrained/deimv2_dinov3_s_coco_tuning.pth
weights/pretrained/vitt_distill.pt
weights/pretrained/dinov3_vits16/config.json
weights/pretrained/dinov3_vits16/model.safetensors
weights/pretrained/verifier_head.pt
weights/pretrained/verifier_head_4cls.pt
```

The DINOv3 files must be obtained under the DINOv3 License. This repository does
not redistribute DINOv3 weights or verifier heads.

## Preflight Check

Run the lightweight checker before training. It does not import Torch.

```bash
python scripts/check_repro_package.py \
  --config configs/crop_detr/crop_detr_dinov3_tiny_ssod.yml
```

Use `--allow-missing` when preparing a package before data or weights have been
placed locally.

## Cold-Start Selection

The paper pipeline starts before detector training. The selector uses frozen
DINOv3 image features and does not require detector scores.

Minimum inputs:

- Candidate training images under `data/candidate/images`.
- A candidate COCO image list, for example `data/annotations/candidate_train.json`.
- Validation/test COCO files passed through `--exclude-coco` to prevent leakage.
- A local DINOv3 model directory under `weights/pretrained/dinov3_vits16/`.
- For `cluster_tail`, labels for the already selected coverage head. In a live
  annotation workflow this is `data/annotations/coverage_head_labeled.json`.

Feature extraction:

```bash
python tools/selection/extract_dinov3_features.py \
  --image-dir data/candidate/images \
  --model-dir weights/pretrained/dinov3_vits16 \
  --output-dir outputs/selection/features
```

Select the coverage head before annotation:

```bash
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
  --per-cluster 10 \
  --output-dir outputs/selection/coverage_head
```

After labeling those images, run the paper-facing 608-image selector:

```bash
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
  --target-box-count 90 \
  --output-dir outputs/selection/cold_start
```

The generated CSV, selected COCO file, and manifest are run artifacts. They are
ignored by Git and should not be committed.

Build nested 152/304/608 subsets from the full 608 selection:

```bash
python tools/selection/make_nested_subsets.py \
  --selection-csv outputs/selection/cold_start/selected_cluster_tail_seed43_n608.csv \
  --budgets 152 304 608 \
  --output-coco-from data/annotations/train_pool_with_boxes.json \
  --output-dir outputs/selection/nested
```

## Verifier Head Calibration

The SSOD verifier uses frozen DINOv3 ROI features and linear heads. Train the
heads from your labeled training set and calibrate four-class acceptance
thresholds on validation crops:

```bash
python tools/verifier/train_linear_verifier.py \
  --train-coco data/annotations/labeled_train.json \
  --train-image-dir data/labeled/images \
  --val-coco data/annotations/val.json \
  --val-image-dir data/val/images \
  --model-dir weights/pretrained/dinov3_vits16 \
  --class-weight 0=1.5 \
  --class-weight 3=1.3 \
  --category-id-base auto \
  --target-precision 0.85 \
  --output-dir outputs/verifier
```

Verifier class ids are the detector's internal labels `0..num_classes-1`. The
tool auto-detects common 0-based and 1-based COCO `category_id` values; use
`--category-id-map raw=internal` for custom category ids.

Copy the generated `verifier_head.pt` and `verifier_head_4cls.pt` into
`weights/pretrained/` before SSOD training. The default public config records
the paper's 608-image verifier thresholds:

```text
Defected 0.57, Flower 0.51, Fully opened 0.70, Partially opened 0.46
```

## Training

Single-seed run:

```bash
bash train_guarded.sh \
  -c configs/crop_detr/crop_detr_dinov3_tiny_ssod.yml \
  -t weights/pretrained/deimv2_dinov3_s_coco_tuning.pth \
  -d cuda \
  --seed 0 \
  --use-amp
```

Three-seed rerun:

```bash
for seed in 0 1 2; do
  bash train_guarded.sh \
    -c configs/crop_detr/crop_detr_dinov3_tiny_ssod.yml \
    -t weights/pretrained/deimv2_dinov3_s_coco_tuning.pth \
    -d cuda \
    --seed "${seed}" \
    --output-dir "outputs/crop_detr_ssod_seed${seed}" \
    --use-amp
done
```

The `--seed` argument seeds Python, NumPy, and Torch through the training
launcher. Exact bitwise reproducibility is not guaranteed across CUDA, cuDNN,
driver, distributed, and dataloader implementations, so report mean and standard
deviation for multi-seed claims.

## Evaluation

Evaluate the teacher EMA checkpoint:

```bash
python tools/evaluate_ssod_teacher.py \
  -c configs/crop_detr/crop_detr_dinov3_tiny_ssod.yml \
  --checkpoint outputs/crop_detr_ssod_seed0/best_teacher_ema.pth \
  --output-dir outputs/crop_detr_ssod_seed0/eval \
  -d cuda
```

For corruption robustness, evaluate each corruption/severity image folder with
the same checkpoint, then summarize mPC/rPC:

```bash
python tools/evaluation/summarize_corruption.py \
  --eval-root outputs/corruption_eval \
  --clean-metrics outputs/crop_detr_ssod_seed0/eval/test_metrics.json \
  --output outputs/corruption_eval/corruption_summary.json \
  --cells-output outputs/corruption_eval/corruption_cells.csv
```

## Semantic Sidecar

The semantic sidecar is optional and uses ROI-level semantic labels supplied by
the user. This repository includes the training/export code and open prompt
templates, but does not include ROI crops, semantic labels, LLM responses,
feature caches, or semantic checkpoints.

Minimum inputs:

- `data/semantic/semantic_labels_train.jsonl`, with one row per ROI crop.
- ROI crop files referenced by each row's `crop_path`.
- DINOv3 feature model files under `weights/pretrained/dinov3_vits16/`.

Train the sidecar:

```bash
python tools/semantic/train_semantic_head.py \
  --config configs/crop_detr/semantic_sidecar_template.yaml
```

Attach semantic attributes to detections:

```bash
python tools/semantic/export_semantics.py \
  --config configs/crop_detr/semantic_sidecar_template.yaml \
  --detections outputs/crop_detr_ssod/test_detections.json \
  --coco-images data/annotations/test.json \
  --image-dir data/test/images \
  --output outputs/crop_detr_ssod/test_detections_with_semantics.json
```

Validate the VLM-generated semantic labels before training:

```bash
python tools/semantic/validate_semantic_labels.py \
  data/semantic/semantic_labels_train.jsonl \
  --output outputs/semantic/label_schema_report.json \
  --fail-on-error
```

## Reporting Checklist

For each reported run, archive:

- Git commit hash or release tag.
- Full config copy after command-line overrides.
- Seed value.
- Hardware model, GPU count, CUDA driver, PyTorch, torchvision, transformers,
  and faster-coco-eval versions.
- Checkpoint filename and SHA256 hash.
- Evaluation JSON and threshold sweep CSV, if produced.
- Training log and TensorBoard directory.

Do not commit these artifacts to the public repository.
