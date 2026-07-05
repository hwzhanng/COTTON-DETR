# COTTON-DETR Configs

`cotton_detr_dinov3_tiny_ssod.yml` is the public paper template for the final v5
SSOD stack:

- DEIM-style detector with DINOv3STAs ViT-Tiny backbone.
- Teacher-student SSOD training.
- Fixed release-recall pseudo-label mining.
- Frozen DINOv3 object/background verifier.
- Frozen DINOv3 multi-class verifier with label reassignment and the paper's
  608-image acceptance thresholds.
- Soft-background ignore regions for probable objects not promoted to positives.

Before training, replace:

- `num_classes`
- labeled, unlabeled, and validation dataset paths
- verifier head paths
- `accept_threshold` if recalibrating the verifier for a new dataset
- `class_loss_scale` if changing the long-tail class weighting
- class-balanced sampling class ids

No project-specific experiment results or private dataset paths are stored here.
