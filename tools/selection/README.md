# Cold-Start Selection

This folder exposes the detector-free image selection stage used before SSOD
training. It does not require a trained detector.

The paper-facing method is `cluster_tail`:

1. Extract frozen DINOv3 CLS features for every candidate training image.
2. Run k-means in the normalized feature space.
3. Select the nearest representatives from each cluster to form a coverage head.
4. After the coverage head is annotated, use only those labels to find
   target-class seed clusters.
5. Add the nearest unselected images from those seed clusters until the label
   budget is reached. The manifest also reports the density-based tail-budget
   estimate used in the paper to justify the 500 + 108 split.

The same CLI also exposes `cluster_head`, `random`, and `kcenter` baselines.

## Extract DINOv3 Features

```bash
python tools/selection/extract_dinov3_features.py \
  --image-dir data/candidate/images \
  --model-dir weights/pretrained/dinov3_vits16 \
  --output-dir outputs/selection/features \
  --batch-size 16 \
  --device cuda
```

## Paper Method

For the 608-image setting, the default public recipe mirrors the paper design:
500 coverage images plus an estimated tail expansion for the rare target class.

First select the coverage head for annotation:

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

After those images are labeled, expand toward the rare target class:

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

If you are replaying a completed experiment, `--output-coco-from` can be the
full training-pool annotation file; the code only reads target-class counts for
images selected in the coverage head. In a live annotation workflow, omit
`--output-coco-from` during tail selection, annotate the returned tail images,
then generate your final labeled COCO file after annotation is complete.

## Baselines

```bash
python tools/selection/select_cold_start.py \
  --method random \
  --features outputs/selection/features/dinov3_features.npy \
  --paths outputs/selection/features/dinov3_paths.npy \
  --candidate-coco data/annotations/candidate_train.json \
  --budget 608 \
  --seed 0 \
  --output-dir outputs/selection/random

python tools/selection/select_cold_start.py \
  --method kcenter \
  --features outputs/selection/features/dinov3_features.npy \
  --paths outputs/selection/features/dinov3_paths.npy \
  --candidate-coco data/annotations/candidate_train.json \
  --budget 608 \
  --kcenter-init medoid \
  --output-dir outputs/selection/kcenter
```

Outputs are CSV/JSON manifests under `outputs/selection/`, which is ignored by
Git. Do not commit generated selections or annotation files.
