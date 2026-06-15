# CROP-DETR Configs

`crop_detr_dinov3_tiny_ssod.yml` is a public template for the final v5 SSOD stack:

- DEIM-style detector with DINOv3STAs ViT-Tiny backbone.
- Teacher-student SSOD training.
- Fixed release-recall pseudo-label mining.
- Frozen DINOv3 object/background verifier.
- Optional frozen DINOv3 multi-class verifier with label reassignment.
- Soft-background ignore regions for probable objects not promoted to positives.

Before training, replace:

- `num_classes`
- labeled, unlabeled, and validation dataset paths
- verifier head paths
- `accept_threshold`
- `class_loss_scale`
- class-balanced sampling class ids

No project-specific experiment results or private dataset paths are stored here.
