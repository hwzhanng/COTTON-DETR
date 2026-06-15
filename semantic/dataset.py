"""Semantic JSONL datasets and feature caches."""

from __future__ import annotations

import hashlib
from pathlib import Path

import torch
from PIL import Image

from .features import forward_features, load_feature_model, pil_to_tensor
from .io import read_jsonl
from .taxonomy import DAMAGE_TYPE_TO_ID, OPENING_TO_ID, SEVERITY_TO_ID, normalize_damage_present


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def row_to_ids(row: dict) -> dict[str, int | float]:
    opening = row.get("opening_state", row.get("opening_name", "closed"))
    damage_present = normalize_damage_present(row.get("damage_present", False))
    damage_type = row.get("damage_type", "none")
    severity = row.get("damage_severity", row.get("severity", "none"))
    return {
        "opening_id": int(row.get("opening_id", OPENING_TO_ID[opening])),
        "opening_percentage": float(row.get("opening_percentage", 0.0)),
        "damage_id": 1 if damage_present else 0,
        "damage_type_id": int(row.get("damage_type_id", DAMAGE_TYPE_TO_ID[damage_type])),
        "severity_id": int(row.get("severity_id", SEVERITY_TO_ID[severity])),
    }


@torch.no_grad()
def build_or_load_feature_cache(
    labels_path: str | Path,
    cache_path: str | Path,
    model_path: str | Path,
    device: torch.device,
    image_size: int = 224,
    batch_size: int = 64,
    force: bool = False,
) -> tuple[dict, list[dict]]:
    labels_path = Path(labels_path)
    cache_path = Path(cache_path)
    rows = read_jsonl(labels_path)
    labels_hash = sha256_file(labels_path)
    if cache_path.exists() and not force:
        payload = torch.load(cache_path, map_location="cpu")
        if payload.get("labels_sha256") == labels_hash and int(payload.get("num_rows", -1)) == len(rows):
            return payload, rows

    model = load_feature_model(model_path, device)
    features: list[torch.Tensor] = []
    fields = {"ann_ids": [], "image_ids": [], "opening": [], "opening_pct": [], "damage": [], "damage_type": [], "severity": []}

    for start in range(0, len(rows), batch_size):
        batch = rows[start : start + batch_size]
        tensors = []
        for row in batch:
            ids = row_to_ids(row)
            tensors.append(pil_to_tensor(Image.open(row["crop_path"]), image_size))
            fields["ann_ids"].append(int(row.get("ann_id", start)))
            fields["image_ids"].append(int(row.get("image_id", -1)))
            fields["opening"].append(ids["opening_id"])
            fields["opening_pct"].append(float(ids["opening_percentage"]) / 100.0)
            fields["damage"].append(ids["damage_id"])
            fields["damage_type"].append(ids["damage_type_id"])
            fields["severity"].append(ids["severity_id"])
        images = torch.stack(tensors).to(device)
        features.append(forward_features(model, images).cpu())
        print(f"encoded semantic crops: {min(start + len(batch), len(rows))}/{len(rows)}", flush=True)

    payload = {
        "num_rows": len(rows),
        "labels_path": str(labels_path),
        "labels_sha256": labels_hash,
        "feature_model_path": str(model_path),
        "features": torch.cat(features, dim=0) if features else torch.empty(0, 1),
        "ann_ids": torch.tensor(fields["ann_ids"], dtype=torch.long),
        "image_ids": torch.tensor(fields["image_ids"], dtype=torch.long),
        "opening": torch.tensor(fields["opening"], dtype=torch.long),
        "opening_pct": torch.tensor(fields["opening_pct"], dtype=torch.float32),
        "damage": torch.tensor(fields["damage"], dtype=torch.float32),
        "damage_type": torch.tensor(fields["damage_type"], dtype=torch.long),
        "severity": torch.tensor(fields["severity"], dtype=torch.long),
    }
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, cache_path)
    return payload, rows


def tensor_dataset(payload: dict, indices: torch.Tensor):
    from torch.utils.data import TensorDataset

    idx = indices.long()
    return TensorDataset(
        payload["features"][idx],
        payload["opening"][idx],
        payload["opening_pct"][idx],
        payload["damage"][idx],
        payload["damage_type"][idx],
        payload["severity"][idx],
    )
