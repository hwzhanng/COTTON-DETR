#!/usr/bin/env python
"""Validate semantic ROI label JSONL files against the public schema."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from semantic.schema import review_priority, validate_semantic_row  # noqa: E402


def iter_jsonl(path: Path):
    with path.open(encoding="utf-8-sig") as handle:
        for line_no, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                yield line_no, json.loads(line)
            except json.JSONDecodeError as exc:
                yield line_no, {"_json_error": str(exc)}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("jsonl", type=Path)
    parser.add_argument("--output", type=Path, default=Path("outputs/semantic/label_schema_report.json"))
    parser.add_argument("--csv-output", type=Path)
    parser.add_argument("--lenient-opening-range", action="store_true")
    parser.add_argument("--fail-on-error", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    rows = []
    error_count = 0
    review_count = 0
    confidence_sum = 0.0
    confidence_n = 0

    for line_no, row in iter_jsonl(args.jsonl):
        if "_json_error" in row:
            errors = [f"json_decode_error={row['_json_error']}"]
        else:
            errors = validate_semantic_row(
                row,
                strict_opening_range=not args.lenient_opening_range,
            )
        priority = review_priority(row, errors)
        error_count += int(bool(errors))
        review_count += int(priority)
        try:
            confidence_sum += float(row.get("confidence"))
            confidence_n += 1
        except (TypeError, ValueError):
            pass
        rows.append(
            {
                "line": line_no,
                "ann_id": row.get("ann_id", ""),
                "file_name": row.get("file_name", ""),
                "errors": ";".join(errors),
                "review_priority": priority,
                "confidence": row.get("confidence", ""),
            }
        )

    report = {
        "input": str(args.jsonl),
        "rows": len(rows),
        "schema_error_rows": error_count,
        "review_priority_rows": review_count,
        "mean_confidence": None if confidence_n == 0 else confidence_sum / confidence_n,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8", newline="\n")
    if args.csv_output:
        args.csv_output.parent.mkdir(parents=True, exist_ok=True)
        with args.csv_output.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(
                handle,
                fieldnames=["line", "ann_id", "file_name", "errors", "review_priority", "confidence"],
            )
            writer.writeheader()
            writer.writerows(rows)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    if args.fail_on_error and error_count:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
