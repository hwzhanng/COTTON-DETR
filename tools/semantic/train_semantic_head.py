"""Train a semantic sidecar head from ROI crops and semantic JSONL labels."""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, WeightedRandomSampler
import yaml

from semantic.dataset import build_or_load_feature_cache, tensor_dataset
from semantic.features import choose_device
from semantic.io import resolve, write_json
from semantic.metrics import macro_f1
from semantic.model import build_head
from semantic.taxonomy import DAMAGE_TYPES, OPENING_STATES, SEVERITY_LEVELS


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def class_weights(labels: torch.Tensor, num_classes: int, max_weight: float = 20.0) -> torch.Tensor:
    counts = torch.bincount(labels.long().cpu(), minlength=num_classes).float()
    weights = torch.zeros(num_classes)
    present = counts > 0
    if present.any():
        weights[present] = counts[present].sum() / counts[present].clamp_min(1.0)
        weights[present] = (weights[present] / weights[present].mean()).clamp(max=max_weight)
    return weights


def make_sampler(payload: dict, indices: torch.Tensor) -> WeightedRandomSampler:
    labels = payload["damage_type"][indices].long()
    weights = class_weights(labels, len(DAMAGE_TYPES))[labels]
    weights = weights * torch.where(payload["damage"][indices] > 0, torch.tensor(2.0), torch.tensor(1.0))
    return WeightedRandomSampler(weights.double(), num_samples=len(weights), replacement=True)


def loss_fn(out: dict, batch: list[torch.Tensor], weights: dict, balanced: bool) -> torch.Tensor:
    _, opening, opening_pct, damage, damage_type, severity = batch
    if balanced:
        opening_w = class_weights(opening, len(OPENING_STATES)).to(opening.device)
        damage_type_w = class_weights(damage_type, len(DAMAGE_TYPES)).to(opening.device)
        severity_w = class_weights(severity, len(SEVERITY_LEVELS)).to(opening.device)
    else:
        opening_w = damage_type_w = severity_w = None
    return (
        float(weights.get("opening_state_ce", 1.0)) * F.cross_entropy(out["opening"], opening, weight=opening_w)
        + float(weights.get("opening_percentage_smooth_l1", 1.0)) * F.smooth_l1_loss(out["opening_pct"], opening_pct)
        + float(weights.get("damage_present_bce", 1.0)) * F.binary_cross_entropy_with_logits(out["damage"], damage)
        + float(weights.get("damage_type_ce", 1.0)) * F.cross_entropy(out["damage_type"], damage_type, weight=damage_type_w)
        + float(weights.get("damage_severity_ce", 0.7)) * F.cross_entropy(out["severity"], severity, weight=severity_w)
    )


@torch.no_grad()
def evaluate(model, loader, device) -> dict:
    model.eval()
    chunks = {key: [] for key in ["opening_p", "opening_t", "damage_p", "damage_t", "type_p", "type_t", "sev_p", "sev_t", "pct_abs"]}
    for batch in loader:
        batch = [item.to(device) for item in batch]
        x, opening, opening_pct, damage, damage_type, severity = batch
        out = model(x)
        damage_pred = (torch.sigmoid(out["damage"]) >= 0.5).long()
        chunks["opening_p"].append(out["opening"].argmax(1).cpu())
        chunks["opening_t"].append(opening.cpu())
        chunks["damage_p"].append(damage_pred.cpu())
        chunks["damage_t"].append(damage.long().cpu())
        chunks["type_p"].append(out["damage_type"].argmax(1).cpu())
        chunks["type_t"].append(damage_type.cpu())
        chunks["sev_p"].append(out["severity"].argmax(1).cpu())
        chunks["sev_t"].append(severity.cpu())
        chunks["pct_abs"].append((out["opening_pct"].cpu() - opening_pct.cpu()).abs())
    cat = {key: torch.cat(value) for key, value in chunks.items()}
    return {
        "opening_macro_f1": macro_f1(cat["opening_p"], cat["opening_t"], len(OPENING_STATES)),
        "damage_macro_f1": macro_f1(cat["damage_p"], cat["damage_t"], 2),
        "damage_type_macro_f1": macro_f1(cat["type_p"], cat["type_t"], len(DAMAGE_TYPES)),
        "severity_macro_f1": macro_f1(cat["sev_p"], cat["sev_t"], len(SEVERITY_LEVELS)),
        "opening_pct_mae": float(cat["pct_abs"].mean().item() * 100.0),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/cotton_detr/semantic_sidecar_template.yaml")
    parser.add_argument("--force-features", action="store_true")
    args = parser.parse_args()

    cfg = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    root = Path(cfg["project"]["root"]).resolve()
    seed = int(cfg["project"].get("seed", 0))
    set_seed(seed)
    device = choose_device(str(cfg["training"].get("device", "auto")))
    labels_path = resolve(root, cfg["dataset"]["semantic_labels"])
    cache_path = resolve(root, cfg["model"]["feature_cache"])
    payload, _ = build_or_load_feature_cache(
        labels_path=labels_path,
        cache_path=cache_path,
        model_path=resolve(root, cfg["model"]["feature_model_path"]),
        device=device,
        image_size=int(cfg["model"].get("image_size", 224)),
        batch_size=int(cfg["training"].get("batch_size", 64)),
        force=args.force_features,
    )

    split = cfg["dataset"].get("split")
    n = int(payload["features"].shape[0])
    if split:
        split_obj = json.loads(resolve(root, split).read_text(encoding="utf-8"))
        train_ann = set(int(v) for v in split_obj["train_ann_ids"])
        val_ann = set(int(v) for v in split_obj["val_ann_ids"])
        train_idx = torch.tensor([i for i, ann_id in enumerate(payload["ann_ids"].tolist()) if int(ann_id) in train_ann])
        val_idx = torch.tensor([i for i, ann_id in enumerate(payload["ann_ids"].tolist()) if int(ann_id) in val_ann])
    else:
        generator = torch.Generator().manual_seed(seed)
        perm = torch.randperm(n, generator=generator)
        cut = int(n * float(cfg["dataset"].get("train_ratio", 0.8)))
        train_idx, val_idx = perm[:cut], perm[cut:]

    balanced = bool(cfg["training"].get("balanced_sampling", True))
    sampler = make_sampler(payload, train_idx) if balanced else None
    batch_size = int(cfg["training"].get("batch_size", 64))
    train_loader = DataLoader(tensor_dataset(payload, train_idx), batch_size=batch_size, sampler=sampler, shuffle=sampler is None)
    val_loader = DataLoader(tensor_dataset(payload, val_idx), batch_size=batch_size, shuffle=False)
    model = build_head(int(payload["features"].shape[1]), cfg["model"], cfg["model"].get("head_type", "flat")).to(device)
    optim = torch.optim.AdamW(model.parameters(), lr=float(cfg["training"].get("lr", 1e-3)), weight_decay=float(cfg["training"].get("weight_decay", 1e-4)))
    best = {"score": -1.0}
    history = []
    for epoch in range(1, int(cfg["training"].get("epochs", 40)) + 1):
        model.train()
        total = 0.0
        seen = 0
        for batch in train_loader:
            batch = [item.to(device) for item in batch]
            out = model(batch[0])
            loss = loss_fn(out, batch, cfg["training"].get("loss_weights", {}), balanced)
            optim.zero_grad(set_to_none=True)
            loss.backward()
            optim.step()
            total += float(loss.item()) * batch[0].shape[0]
            seen += batch[0].shape[0]
        metrics = evaluate(model, val_loader, device)
        metrics["epoch"] = epoch
        metrics["train_loss"] = total / max(1, seen)
        metrics["score"] = metrics["opening_macro_f1"] + metrics["severity_macro_f1"] + metrics["damage_type_macro_f1"] - metrics["opening_pct_mae"] / 100.0
        history.append(metrics)
        print(json.dumps(metrics, ensure_ascii=False), flush=True)
        if metrics["score"] > best["score"]:
            best = {**metrics, "state_dict": {key: value.cpu() for key, value in model.state_dict().items()}}

    ckpt_dir = resolve(root, cfg["outputs"]["checkpoints"])
    metrics_dir = resolve(root, cfg["outputs"]["metrics"])
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    metrics_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = ckpt_dir / "semantic_head_best.pt"
    metrics_path = metrics_dir / "semantic_head_metrics.json"
    torch.save(
        {
            "state_dict": best["state_dict"],
            "feature_dim": int(payload["features"].shape[1]),
            "model": {
                "hidden_dim": int(cfg["model"].get("hidden_dim", 384)),
                "embed_dim": int(cfg["model"].get("embed_dim", 256)),
                "dropout": float(cfg["model"].get("dropout", 0.1)),
                "head_type": cfg["model"].get("head_type", "flat"),
            },
            "vocab": {"opening": OPENING_STATES, "damage_types": DAMAGE_TYPES, "severity": SEVERITY_LEVELS},
            "seed": seed,
            "best_metrics": {key: value for key, value in best.items() if key != "state_dict"},
        },
        ckpt_path,
    )
    write_json(metrics_path, {"history": history, "best": {key: value for key, value in best.items() if key != "state_dict"}})
    print(json.dumps({"checkpoint": str(ckpt_path), "metrics": str(metrics_path)}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
