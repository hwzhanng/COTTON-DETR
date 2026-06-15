"""Attach CROP-DETR semantic sidecar predictions to detection JSON."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
import torch.nn.functional as F
import yaml
from PIL import Image

from semantic.dataset import build_or_load_feature_cache
from semantic.features import choose_device, crop_detection, forward_features, load_feature_model, pil_to_tensor
from semantic.io import read_json, resolve, write_json
from semantic.model import load_checkpoint
from semantic.prototypes import build_positive_prototypes, defected_prior, nearest_name
from semantic.taxonomy import DAMAGE_TYPES, SEVERITY_LEVELS, category_prior, clip_opening_percentage, OPENING_STATES


def load_detections(path: str | Path) -> list[dict]:
    obj = read_json(path)
    if isinstance(obj, list):
        return obj
    if "detections" in obj:
        return obj["detections"]
    if "annotations" in obj:
        return obj["annotations"]
    raise ValueError("Detection JSON must be a list or contain detections/annotations.")


@torch.no_grad()
def encode_detections(cfg: dict, detections: list[dict], image_dir: Path, coco_images: list[dict], device: torch.device) -> torch.Tensor:
    model = load_feature_model(resolve(cfg["project"]["root"], cfg["model"]["feature_model_path"]), device)
    by_id = {int(item["id"]): item for item in coco_images}
    image_cache: dict[Path, Image.Image] = {}
    image_size = int(cfg["model"].get("image_size", 224))
    expand = float(cfg["dataset"].get("roi_expand", 0.15))
    batch_size = int(cfg["training"].get("batch_size", 64))
    features = []
    for start in range(0, len(detections), batch_size):
        tensors = []
        for det in detections[start : start + batch_size]:
            info = by_id[int(det["image_id"])]
            path = image_dir / info["file_name"]
            if path not in image_cache:
                image_cache[path] = Image.open(path).convert("RGB")
            tensors.append(pil_to_tensor(crop_detection(image_cache[path], det["bbox"], expand), image_size))
        if tensors:
            features.append(forward_features(model, torch.stack(tensors).to(device)).cpu())
        print(f"encoded detections: {min(start + batch_size, len(detections))}/{len(detections)}", flush=True)
    for image in image_cache.values():
        image.close()
    return torch.cat(features, dim=0) if features else torch.empty(0, 1)


@torch.no_grad()
def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/crop_detr/semantic_sidecar_template.yaml")
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--detections", required=True)
    parser.add_argument("--coco-images", required=True, help="COCO JSON with an images list, or a raw images list.")
    parser.add_argument("--image-dir", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--force-features", action="store_true")
    args = parser.parse_args()

    cfg = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    root = Path(cfg["project"]["root"]).resolve()
    cfg["project"]["root"] = str(root)
    device = choose_device(str(cfg["training"].get("device", "auto")))
    checkpoint = Path(args.checkpoint) if args.checkpoint else resolve(root, cfg["outputs"].get("default_checkpoint", "outputs/checkpoints/semantic_head_best.pt"))
    checkpoint = checkpoint if checkpoint.is_absolute() else root / checkpoint
    head, _ = load_checkpoint(checkpoint, device)
    train_payload, train_rows = build_or_load_feature_cache(
        labels_path=resolve(root, cfg["dataset"]["semantic_labels"]),
        cache_path=resolve(root, cfg["model"]["feature_cache"]),
        model_path=resolve(root, cfg["model"]["feature_model_path"]),
        device=device,
        image_size=int(cfg["model"].get("image_size", 224)),
        batch_size=int(cfg["training"].get("batch_size", 64)),
        force=args.force_features,
    )
    type_proto, severity_proto = build_positive_prototypes(train_payload)
    prior = defected_prior(train_rows)
    detections = load_detections(args.detections)
    coco_obj = read_json(args.coco_images)
    coco_images = coco_obj["images"] if isinstance(coco_obj, dict) and "images" in coco_obj else coco_obj
    features = encode_detections(cfg, detections, Path(args.image_dir), coco_images, device)
    norm_features = F.normalize(features, dim=1)
    category_semantics = cfg["taxonomy"]["category_semantics"]
    enriched = []
    for idx, det in enumerate(detections):
        item = dict(det)
        out = head(features[idx : idx + 1].to(device))
        opening_id = int(out["opening"].argmax(1).item())
        opening_state = OPENING_STATES[opening_id]
        opening_pct_raw = float(out["opening_pct"].item() * 100.0)
        semantic = category_prior(int(item["category_id"]), category_semantics, prior)
        semantic.update(
            {
                "opening_state": opening_state,
                "opening_percentage": clip_opening_percentage(opening_pct_raw, opening_state),
                "opening_percentage_raw": opening_pct_raw,
                "opening_confidence": float(torch.softmax(out["opening"], dim=-1).max().item()),
                "decision_sources": {
                    "opening": "semantic_head",
                    "damage_present": "detector_class_prior",
                    "damage_type": "none_or_train_prototype",
                    "damage_severity": "none_or_train_prototype",
                },
            }
        )
        if semantic["damage_present"]:
            semantic["damage_type"], semantic["damage_type_similarity"] = nearest_name(norm_features[idx], type_proto, DAMAGE_TYPES)
            semantic["damage_severity"], semantic["damage_severity_similarity"] = nearest_name(norm_features[idx], severity_proto, SEVERITY_LEVELS)
        item["semantics"] = semantic
        enriched.append(item)
    write_json(args.output, {"method": "crop_detr_semantic_sidecar", "checkpoint": str(checkpoint), "detections": enriched})
    print(json.dumps({"detections": len(enriched), "output": str(args.output)}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
