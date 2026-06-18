#!/usr/bin/env python
"""Summarize corruption-cell AP values into mPC and rPC."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


def read_metric(path: Path, key: str) -> float:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if key in payload:
        return float(payload[key])
    aliases = ["mAP", "map", "coco_bbox_mAP", "bbox_mAP"]
    for alias in aliases:
        if alias in payload:
            return float(payload[alias])
    raise KeyError(f"Cannot find mAP key in {path}; tried {key} and {aliases}.")


def read_cells_csv(path: Path, metric_key: str) -> list[dict]:
    rows = []
    with path.open(encoding="utf-8-sig", newline="") as handle:
        for row in csv.DictReader(handle):
            metric = row.get(metric_key) or row.get("mAP") or row.get("map") or row.get("coco_bbox_mAP")
            if metric is None:
                raise KeyError(f"CSV row lacks metric column: {row}")
            rows.append(
                {
                    "corruption": row["corruption"],
                    "severity": int(row["severity"]),
                    "mAP": float(metric),
                }
            )
    return rows


def scan_eval_root(root: Path, metric_key: str) -> list[dict]:
    rows = []
    for metrics_path in sorted(root.glob("*/*/test_metrics.json")):
        severity_text = metrics_path.parent.name
        corruption = metrics_path.parent.parent.name
        if not severity_text.startswith("s"):
            continue
        rows.append(
            {
                "corruption": corruption,
                "severity": int(severity_text[1:]),
                "mAP": read_metric(metrics_path, metric_key),
            }
        )
    if not rows:
        raise FileNotFoundError(f"No */s*/test_metrics.json files found under {root}")
    return rows


def write_cells_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["corruption", "severity", "mAP"])
        writer.writeheader()
        writer.writerows(rows)


def summarize(rows: list[dict], clean_map: float) -> dict:
    if clean_map <= 0:
        raise ValueError("clean mAP must be positive to compute rPC.")
    mpc = sum(row["mAP"] for row in rows) / len(rows)
    by_corruption = {}
    for corruption in sorted({row["corruption"] for row in rows}):
        vals = [row["mAP"] for row in rows if row["corruption"] == corruption]
        by_corruption[corruption] = {
            "mPC": sum(vals) / len(vals),
            "rPC": (sum(vals) / len(vals)) / clean_map,
            "cells": len(vals),
        }
    by_severity = {}
    for severity in sorted({row["severity"] for row in rows}):
        vals = [row["mAP"] for row in rows if row["severity"] == severity]
        by_severity[str(severity)] = {
            "mPC": sum(vals) / len(vals),
            "rPC": (sum(vals) / len(vals)) / clean_map,
            "cells": len(vals),
        }
    return {
        "clean_mAP": clean_map,
        "mPC": mpc,
        "rPC": mpc / clean_map,
        "num_cells": len(rows),
        "by_corruption": by_corruption,
        "by_severity": by_severity,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--cells-csv", type=Path)
    source.add_argument("--eval-root", type=Path)
    clean = parser.add_mutually_exclusive_group(required=True)
    clean.add_argument("--clean-metrics", type=Path)
    clean.add_argument("--clean-map", type=float)
    parser.add_argument("--metric-key", default="coco_bbox_mAP")
    parser.add_argument("--output", type=Path, default=Path("outputs/eval/corruption_summary.json"))
    parser.add_argument("--cells-output", type=Path)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    rows = (
        read_cells_csv(args.cells_csv, args.metric_key)
        if args.cells_csv
        else scan_eval_root(args.eval_root, args.metric_key)
    )
    clean_map = args.clean_map if args.clean_map is not None else read_metric(args.clean_metrics, args.metric_key)
    summary = summarize(rows, clean_map)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8", newline="\n")
    if args.cells_output:
        write_cells_csv(args.cells_output, rows)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
