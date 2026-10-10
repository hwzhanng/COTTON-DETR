# README figure assets

These six README figures are rendered directly from the manuscript, **DINOv3-Driven Label-Efficient In-Field Cotton Boll State Detection: From Cold-Start Selection to Semantic Interpretation**. The manuscript PDF itself is not included.

| Asset | Manuscript figure | PDF page |
|---|---:|---:|
| `cotton_detr_pipeline.png` | 1 | 3 |
| `cotton_detr_architecture.png` | 7 | 9 |
| `cotton_detr_label_efficiency.png` | 8 | 13 |
| `cotton_detr_crossdomain.png` | 12 | 16 |
| `cotton_detr_robustness.png` | 10 | 15 |
| `cotton_detr_semantic.png` | 13 | 18 |

See [figure-provenance.json](figure-provenance.json) for the source SHA-256 and crop rectangles in PDF points. Images are rendered at 4 pixels per point. No plot data, labels, or values were redrawn.

Figure 13 uses all 1,026 ROIs for opening state and 91 defective test ROIs for damage. Its three-class severity macro-F1 (0.481) differs from the four-class Table 5 result (0.610). Figure 10 uses representative single runs, whereas Table 2 reports three-seed means.
