#!/usr/bin/env python
"""Extract frozen DINOv3 image features for cold-start selection."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms import functional as TF
from tqdm import tqdm

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


class ImagePathDataset(Dataset):
    def __init__(self, image_paths: list[Path], image_size: int):
        self.image_paths = image_paths
        self.image_size = image_size

    def __len__(self) -> int:
        return len(self.image_paths)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, str]:
        path = self.image_paths[idx]
        with Image.open(path) as image:
            image = image.convert("RGB")
            image = TF.resize(image, [self.image_size, self.image_size], antialias=True)
            tensor = TF.normalize(TF.to_tensor(image), IMAGENET_MEAN, IMAGENET_STD)
        return tensor, str(path)


def list_images(image_dir: Path) -> list[Path]:
    if not image_dir.exists():
        raise FileNotFoundError(f"Image directory not found: {image_dir}")
    images = [
        path
        for path in image_dir.rglob("*")
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    ]
    if not images:
        raise FileNotFoundError(f"No images found under {image_dir}")
    return sorted(images, key=lambda path: str(path).replace("\\", "/"))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image-dir", type=Path, required=True)
    parser.add_argument("--model-dir", type=Path, required=True, help="Local DINOv3 Hugging Face directory.")
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/selection/features"))
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--image-size", type=int, default=640)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--limit", type=int, default=0, help="Debug only; 0 means all images.")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    feature_path = args.output_dir / "dinov3_features.npy"
    path_path = args.output_dir / "dinov3_paths.npy"
    summary_path = args.output_dir / "dinov3_feature_summary.json"

    if feature_path.exists() and path_path.exists() and not args.overwrite:
        print(f"Feature outputs already exist: {feature_path}")
        return 0
    if not args.model_dir.exists():
        raise FileNotFoundError(f"DINOv3 model directory not found: {args.model_dir}")

    try:
        from transformers import AutoModel
    except ImportError as exc:
        raise SystemExit("Install transformers to extract DINOv3 features.") from exc

    image_paths = list_images(args.image_dir)
    if args.limit:
        image_paths = image_paths[: args.limit]

    device = args.device
    if device == "cuda" and not torch.cuda.is_available():
        device = "cpu"

    start = time.time()
    model = AutoModel.from_pretrained(
        str(args.model_dir),
        local_files_only=True,
        trust_remote_code=True,
    ).to(device)
    model.eval()

    dataset = ImagePathDataset(image_paths, args.image_size)
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=(device == "cuda"),
    )

    features: list[np.ndarray] = []
    paths: list[str] = []
    with torch.inference_mode():
        for pixel_values, batch_paths in tqdm(loader, desc="extract DINOv3 CLS"):
            pixel_values = pixel_values.to(device, non_blocking=True)
            outputs = model(pixel_values=pixel_values)
            cls = outputs.last_hidden_state[:, 0, :].detach().cpu().to(torch.float32).numpy()
            features.append(cls)
            paths.extend(batch_paths)

    feature_array = np.concatenate(features, axis=0)
    np.save(feature_path, feature_array.astype(np.float32, copy=False))
    np.save(path_path, np.array(paths, dtype=object), allow_pickle=True)

    summary = {
        "features": str(feature_path),
        "paths": str(path_path),
        "shape": list(feature_array.shape),
        "image_dir": str(args.image_dir),
        "model_dir": str(args.model_dir),
        "device": device,
        "batch_size": args.batch_size,
        "image_size": args.image_size,
        "seconds": round(time.time() - start, 3),
    }
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8", newline="\n")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
