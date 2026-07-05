# Evaluation Utilities

`summarize_corruption.py` implements the paper's corruption robustness summary:

- mPC: mean AP over all corruption-type and severity cells.
- rPC: mPC divided by clean-test mAP.

Example from a directory of per-cell evaluation JSON files:

```bash
python tools/evaluation/summarize_corruption.py \
  --eval-root outputs/corruption_eval \
  --clean-metrics outputs/cotton_detr_ssod/eval/test_metrics.json \
  --output outputs/corruption_eval/corruption_summary.json \
  --cells-output outputs/corruption_eval/corruption_cells.csv
```

Expected directory layout:

```text
outputs/corruption_eval/
  gaussian_noise/s1/test_metrics.json
  gaussian_noise/s2/test_metrics.json
  ...
```

Alternatively pass a CSV with `corruption`, `severity`, and `mAP` columns via
`--cells-csv`.
