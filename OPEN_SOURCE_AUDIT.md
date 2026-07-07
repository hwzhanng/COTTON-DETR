# Open-Source Preparation Audit

Prepared from the final local SSOD code tree used for release preparation. The
source workspace path is intentionally not recorded in this public scaffold.

Default public template:

```text
configs/cotton_detr/cotton_detr_dinov3_tiny_ssod.yml
```

Version decision:

- Public default uses the v5 SSOD stack: DINOv3STAs ViT-Tiny backbone, teacher-student
  SSOD, DINOv3 verifier, multi-class verifier reassignment, and soft-background ignore.
- The v6 DINOv3-S backbone experiment is not included as the default public version.

Excluded from this repository:

- Datasets and annotations.
- Pretrained weights and verifier heads.
- Checkpoints, logs, metric files, result archives, and generated outputs.
- Internal analysis notes, paper drafts, server upload scripts, and local machine paths.
- Historical experiment configs with project-specific paths or tuned result details.

Reproducibility materials added:

- `REPRODUCIBILITY.md` records the public config, data layout, required external
  weights, seed protocol, training/evaluation commands, and reporting checklist.
- `requirements-repro.txt` gives a pinned baseline environment while
  `requirements.txt` remains a looser install file.
- `scripts/check_repro_package.py` validates config, data, and weight paths
  without importing Torch.
- `engine/data/` includes the dataset loaders, transforms, COCO evaluator, and
  SSOD augmentations required by the public training entry point, without
  shipping datasets or annotations.
- `selection/`, `tools/selection/`, and
  `configs/cotton_detr/cold_start_selection_template.yaml` expose the
  detector-free cold-start selection stage, including random, k-center, and
  DINOv3 cluster-tail selection plus nested budget derivation, without shipping
  selected image lists, annotations, feature caches, or local experiment
  outputs.
- `tools/verifier/` exposes DINOv3 linear verifier-head training and validation
  threshold calibration without redistributing verifier weights or ROI crops.
- `tools/evaluation/` exposes the paper's mPC/rPC corruption summary formula
  without shipping corrupted datasets or evaluation outputs.
- `semantic/schema.py` and `tools/semantic/validate_semantic_labels.py` expose
  the semantic-label schema and consistency checks without including VLM
  responses or API callers.
- `docs/assets/` stores web-optimized selected manuscript figures for the
  README files only; raw experiment archives and figure-generation workspaces
  remain excluded.
- `semantic/`, `tools/semantic/`, `configs/cotton_detr/semantic_sidecar_template.yaml`,
  and `prompts/` expose the semantic encoding code path and prompt templates
  without shipping ROI crops, labels, feature caches, checkpoints, API callers,
  manual-audit UI files, or manuscript asset scripts.

License materials added:

- Root `LICENSE` covers original COTTON-DETR additions and release materials
  under CC BY-NC 4.0, with non-commercial use only unless separate permission is
  granted by the authors.
- `NOTICE.md` records that DEIM, D-FINE, and RT-DETR are Apache-2.0 and DINOv3
  code/model materials are under the Meta DINOv3 License.
- `third_party/licenses/` stores copied upstream license texts.

Sanitization checks run:

- No large model/result artifacts were found.
- No source-workspace paths were found.
- No remote server credentials or host strings were found.
- `git add --dry-run .` only lists code, configs, docs, and placeholder README files.
