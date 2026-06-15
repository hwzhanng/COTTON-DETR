"""Check whether the public CROP-DETR package is ready for a rerun."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

try:
    import yaml
except ImportError:  # pragma: no cover - dependency guidance path.
    yaml = None


def _resolve(root: Path, value: str | None) -> Path | None:
    if not value:
        return None
    path = Path(value)
    if not path.is_absolute():
        path = root / path
    return path.resolve()


def _dataset_paths(config: dict) -> list[tuple[str, Path | None]]:
    checks: list[tuple[str, Path | None]] = []
    for loader_name in ("labeled_dataloader", "unlabeled_dataloader", "val_dataloader"):
        dataset = config.get(loader_name, {}).get("dataset", {})
        checks.append((f"{loader_name}.dataset.img_folder", dataset.get("img_folder")))
        checks.append((f"{loader_name}.dataset.ann_file", dataset.get("ann_file")))
    return checks


def _weight_paths(config: dict) -> list[tuple[str, Path | None]]:
    verifier = config.get("ssod", {}).get("verifier", {})
    return [
        ("DINOv3STAs.weights_path", config.get("DINOv3STAs", {}).get("weights_path")),
        ("ssod.verifier.weights_dir", verifier.get("weights_dir")),
        ("ssod.verifier.head_path", verifier.get("head_path")),
        ("ssod.verifier.head4_path", verifier.get("head4_path")),
    ]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "-c",
        "--config",
        default="configs/crop_detr/crop_detr_dinov3_tiny_ssod.yml",
        help="Path to the CROP-DETR config.",
    )
    parser.add_argument(
        "--tuning-weights",
        default="weights/pretrained/deimv2_dinov3_s_coco_tuning.pth",
        help="Pretrained detector checkpoint passed to train.py with -t/--tuning.",
    )
    parser.add_argument(
        "--allow-missing",
        action="store_true",
        help="Print missing paths but exit successfully.",
    )
    args = parser.parse_args()

    if yaml is None:
        print("PyYAML is required: pip install PyYAML", file=sys.stderr)
        return 2

    repo_root = Path(__file__).resolve().parents[1]
    config_path = _resolve(repo_root, args.config)
    if config_path is None or not config_path.exists():
        print(f"Missing config: {config_path}", file=sys.stderr)
        return 1

    with config_path.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    if not isinstance(config, dict):
        print(f"Config did not parse as a mapping: {config_path}", file=sys.stderr)
        return 1

    checks: list[tuple[str, Path | None]] = [("config", config_path)]
    checks.extend((name, _resolve(repo_root, str(path)) if path is not None else None) for name, path in _dataset_paths(config))
    checks.extend((name, _resolve(repo_root, str(path)) if path is not None else None) for name, path in _weight_paths(config))
    checks.append(("train.py --tuning", _resolve(repo_root, args.tuning_weights)))

    verifier_dir = _resolve(repo_root, config.get("ssod", {}).get("verifier", {}).get("weights_dir"))
    if verifier_dir is not None:
        checks.append(("DINOv3 config.json", verifier_dir / "config.json"))
        checks.append(("DINOv3 model.safetensors", verifier_dir / "model.safetensors"))

    missing = [(name, path) for name, path in checks if path is None or not path.exists()]

    print("CROP-DETR reproducibility preflight")
    print(f"repo_root: {repo_root}")
    print(f"config: {config_path}")
    print()

    if missing:
        print("Missing required paths:")
        for name, path in missing:
            print(f"  - {name}: {path}")
    else:
        print("All required data and weight paths exist.")

    print()
    print("Remember to record the git commit, full config, seed, environment versions,")
    print("checkpoint SHA256, evaluation JSON, and logs for each reported run.")

    if missing and not args.allow_missing:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
