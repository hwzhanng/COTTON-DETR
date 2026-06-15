# Open-Source Preparation Audit

Prepared from the final local SSOD code tree used for release preparation. The
source workspace path is intentionally not recorded in this public scaffold.

Default public template:

```text
configs/crop_detr/crop_detr_dinov3_tiny_ssod.yml
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
- `docs/assets/` stores web-optimized selected manuscript figures for the
  README files only; raw experiment archives and figure-generation workspaces
  remain excluded.
- `semantic/`, `tools/semantic/`, `configs/crop_detr/semantic_sidecar_template.yaml`,
  and `prompts/` expose the semantic encoding code path and prompt templates
  without shipping ROI crops, labels, feature caches, checkpoints, API callers,
  manual-audit UI files, or manuscript asset scripts.

License materials added:

- Root `LICENSE` covers original CROP-DETR additions under Apache License 2.0.
- `NOTICE.md` records that DEIM, D-FINE, and RT-DETR are Apache-2.0 and DINOv3
  code/model materials are under the Meta DINOv3 License.
- `third_party/licenses/` stores copied upstream license texts.

Sanitization checks run:

- No large model/result artifacts were found.
- No source-workspace paths were found.
- No remote server credentials or host strings were found.
- `git add --dry-run .` only lists code, configs, docs, and placeholder README files.
