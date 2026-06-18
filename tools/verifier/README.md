# DINOv3 Verifier Tools

The paper uses a frozen DINOv3 ViT-S/16 ROI encoder plus a linear four-class
head to verify teacher pseudo labels. The detector supplies candidate boxes;
the verifier supplies an independent class decision and per-class acceptance
gate.

Train and calibrate the public verifier heads from user-provided COCO labels:

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

The detector and verifier use internal labels `0..num_classes-1`. The tool
auto-detects common 0-based and 1-based COCO category ids. For custom ids, pass
explicit mappings such as `--category-id-map 10=0 --category-id-map 20=1`.
`--class-weight` uses the mapped internal labels.

Outputs:

- `verifier_head_4cls.pt`: four-class linear head with calibrated
  `accept_threshold` metadata.
- `verifier_head.pt`: optional object/background head trained with sampled
  background crops.
- `verifier_training_summary.json`: validation confusion matrix, thresholds,
  and crop counts.

Copy the two `.pt` files into `weights/pretrained/` for training. Generated
heads and summaries are ignored by Git and should not be committed.
