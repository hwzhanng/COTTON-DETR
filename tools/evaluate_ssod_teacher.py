"""Evaluate a TARS/SSOD teacher checkpoint on a configured split."""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from engine.core import YAMLConfig
from engine.misc import dist_utils
from engine.solver._solver import remove_module_prefix
from engine.solver.det_engine import evaluate
from engine.solver.ssod_solver import SSODDetSolver


def _select_state_dict(state: dict) -> tuple[str, dict]:
    for key in ("teacher", "model"):
        if key in state:
            return key, state[key]
    if all(isinstance(k, str) for k in state.keys()):
        return "raw", state
    raise KeyError("Checkpoint does not contain a teacher/model state_dict")


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate the teacher EMA from a TARS SSOD checkpoint.")
    parser.add_argument("-c", "--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("-d", "--device", default="cuda")
    parser.add_argument("--print-method", default="builtin")
    parser.add_argument("--print-rank", type=int, default=0)
    args = parser.parse_args()

    dist_utils.setup_distributed(args.print_rank, args.print_method, seed=None)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    cfg = YAMLConfig(args.config, device=args.device, output_dir=str(output_dir))
    solver = SSODDetSolver(cfg)
    solver._setup()
    solver.val_dataloader = dist_utils.warp_loader(
        cfg.val_dataloader, shuffle=cfg.val_dataloader.shuffle
    )
    solver.evaluator = cfg.evaluator

    try:
        state = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    except TypeError:
        state = torch.load(args.checkpoint, map_location="cpu")
    state_key, state_dict = _select_state_dict(state)
    module = dist_utils.de_parallel(solver.model)
    missing, unexpected = module.load_state_dict(remove_module_prefix(state_dict), strict=False)
    print(f"Loaded checkpoint state='{state_key}' missing={len(missing)} unexpected={len(unexpected)}")
    if missing:
        print("Missing keys sample:", missing[:20])
    if unexpected:
        print("Unexpected keys sample:", unexpected[:20])

    test_stats, _ = evaluate(
        solver.model,
        solver.criterion,
        solver.postprocessor,
        solver.val_dataloader,
        solver.evaluator,
        solver.device,
    )
    sweep = solver._threshold_sweep(solver.model, iou_threshold=0.5)

    stats = {
        "checkpoint": str(args.checkpoint),
        "state_key": state_key,
        "coco_eval": test_stats,
        "threshold_sweep": sweep,
    }
    if "coco_eval_bbox" in test_stats:
        stats["coco_bbox_mAP"] = float(test_stats["coco_eval_bbox"][0])
        stats["coco_bbox_mAP50"] = float(test_stats["coco_eval_bbox"][1])

    metrics_path = output_dir / "test_metrics.json"
    with metrics_path.open("w", encoding="utf-8") as handle:
        json.dump(stats, handle, ensure_ascii=False, indent=2)

    sweep_path = output_dir / "val_threshold_sweep.csv"
    if sweep_path.exists():
        shutil.copy2(sweep_path, output_dir / "test_threshold_sweep.csv")

    print(json.dumps({
        "test_metrics": str(metrics_path),
        "coco_bbox_mAP": stats.get("coco_bbox_mAP"),
        "coco_bbox_mAP50": stats.get("coco_bbox_mAP50"),
        "best_threshold": sweep.get("best_global", {}).get("threshold"),
        "macro_f1_iou50": sweep.get("best_global", {}).get("f1"),
    }, ensure_ascii=False, indent=2))
    dist_utils.cleanup()


if __name__ == "__main__":
    main()
