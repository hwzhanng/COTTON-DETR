"""DINOv3 ROI feature extraction for semantic encoding."""

from __future__ import annotations

from pathlib import Path

import torch
import torch.nn.functional as F
from PIL import Image


def choose_device(requested: str = "auto") -> torch.device:
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(requested)


def pil_to_tensor(image: Image.Image, size: int) -> torch.Tensor:
    image = image.convert("RGB").resize((size, size), Image.BICUBIC)
    data = torch.frombuffer(image.tobytes(), dtype=torch.uint8).clone()
    tensor = data.view(size, size, 3).permute(2, 0, 1).float().div(255.0)
    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
    return (tensor - mean) / std


def load_feature_model(model_path: str | Path, device: torch.device):
    try:
        from transformers import AutoModel
    except ImportError as exc:
        raise SystemExit("Install transformers to extract DINOv3 features.") from exc
    model = AutoModel.from_pretrained(str(model_path), trust_remote_code=True).to(device)
    model.eval()
    for param in model.parameters():
        param.requires_grad_(False)
    return model


@torch.no_grad()
def forward_features(model, images: torch.Tensor) -> torch.Tensor:
    out = model(pixel_values=images)
    if hasattr(out, "pooler_output") and out.pooler_output is not None:
        feat = out.pooler_output
    elif hasattr(out, "last_hidden_state"):
        feat = out.last_hidden_state[:, 0]
    elif isinstance(out, (tuple, list)):
        feat = out[0][:, 0]
    else:
        raise RuntimeError("Cannot find image feature tensor in model output.")
    return F.normalize(feat.float(), dim=-1)


def expand_bbox_xywh(bbox: list[float], width: int, height: int, expand: float) -> tuple[int, int, int, int]:
    x, y, w, h = [float(v) for v in bbox]
    pad_w, pad_h = w * expand, h * expand
    x1 = max(0, int(x - pad_w))
    y1 = max(0, int(y - pad_h))
    x2 = min(width, int(x + w + pad_w))
    y2 = min(height, int(y + h + pad_h))
    if x2 <= x1 or y2 <= y1:
        raise ValueError(f"Invalid ROI after expansion: {bbox}")
    return x1, y1, x2, y2


def crop_detection(image: Image.Image, bbox: list[float], expand: float) -> Image.Image:
    return image.crop(expand_bbox_xywh(bbox, image.width, image.height, expand))
