# Pretrained Weights

Weights are intentionally not included.

Default paths expected by `configs/crop_detr/crop_detr_dinov3_tiny_ssod.yml`:

```text
weights/pretrained/deimv2_dinov3_s_coco_tuning.pth
weights/pretrained/vitt_distill.pt
weights/pretrained/dinov3_vits16/config.json
weights/pretrained/dinov3_vits16/model.safetensors
weights/pretrained/verifier_head.pt
weights/pretrained/verifier_head_4cls.pt
```

Train or replace the verifier heads for your own dataset with:

```bash
python tools/verifier/train_linear_verifier.py --help
```
