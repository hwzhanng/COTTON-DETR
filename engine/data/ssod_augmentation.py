"""Geometry-safe weak/strong augmentations for TARS SSOD.

Teacher predictions are produced on the weak view and supervised on the student
strong view. Any geometric transform applied to the strong view must therefore
be replayed on pseudo boxes before the criterion sees them.
"""

from __future__ import annotations

import random
import math
from dataclasses import dataclass
from typing import Dict, Iterable, List, Sequence, Tuple

import torch
import torch.nn.functional as F
import torchvision.transforms.functional as TF


Step = Tuple[str, Dict[str, float]]


def _cxcywh_to_xyxy(boxes: torch.Tensor) -> torch.Tensor:
    cx, cy, w, h = boxes.unbind(-1)
    return torch.stack((cx - 0.5 * w, cy - 0.5 * h, cx + 0.5 * w, cy + 0.5 * h), dim=-1)


def _xyxy_to_cxcywh(boxes: torch.Tensor) -> torch.Tensor:
    x0, y0, x1, y1 = boxes.unbind(-1)
    return torch.stack(((x0 + x1) * 0.5, (y0 + y1) * 0.5, x1 - x0, y1 - y0), dim=-1)


@dataclass
class _SampleTransform:
    steps: List[Step]

    @property
    def has_geometry(self) -> bool:
        return any(name in {"crop", "zoomout", "hflip"} for name, _ in self.steps)


@dataclass
class SSODPairAugmentation:
    weak_images: torch.Tensor
    strong_images: torch.Tensor
    transforms: List[_SampleTransform]
    aug_stats: Dict[str, float]
    min_box_size: float

    def _map_xyxy(self, boxes: torch.Tensor, batch_idx: int) -> torch.Tensor:
        mapped = boxes.clone()
        for name, params in self.transforms[batch_idx].steps:
            if name == "crop":
                x0, y0, w, h = params["x0"], params["y0"], params["w"], params["h"]
                mapped[..., [0, 2]] = (mapped[..., [0, 2]] - x0) / max(w, 1e-6)
                mapped[..., [1, 3]] = (mapped[..., [1, 3]] - y0) / max(h, 1e-6)
            elif name == "zoomout":
                sx, sy = params["sx"], params["sy"]
                ox, oy = params["ox"], params["oy"]
                mapped[..., [0, 2]] = mapped[..., [0, 2]] * sx + ox
                mapped[..., [1, 3]] = mapped[..., [1, 3]] * sy + oy
            elif name == "hflip":
                x0 = 1.0 - mapped[..., 2].clone()
                x1 = 1.0 - mapped[..., 0].clone()
                mapped[..., 0] = x0
                mapped[..., 2] = x1
        return mapped

    def map_boxes_to_strong(self, boxes: torch.Tensor, batch_idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        if boxes.numel() == 0:
            return boxes.clone(), torch.zeros((boxes.shape[0],), dtype=torch.bool, device=boxes.device)
        xyxy = _cxcywh_to_xyxy(boxes)
        xyxy = self._map_xyxy(xyxy, batch_idx).clamp(0.0, 1.0)
        wh = xyxy[..., 2:] - xyxy[..., :2]
        keep = (wh[..., 0] > self.min_box_size) & (wh[..., 1] > self.min_box_size)
        return _xyxy_to_cxcywh(xyxy), keep

    def map_targets_to_strong(self, targets: Iterable[Dict[str, torch.Tensor]]) -> Tuple[List[Dict[str, torch.Tensor]], Dict[str, float]]:
        mapped_targets: List[Dict[str, torch.Tensor]] = []
        before = 0
        after = 0
        for batch_idx, target in enumerate(targets):
            boxes = target["boxes"]
            before += int(boxes.shape[0])
            mapped_boxes, keep = self.map_boxes_to_strong(boxes, batch_idx)
            after += int(keep.sum().item())
            mapped: Dict[str, torch.Tensor] = {}
            for key, value in target.items():
                if key == "boxes":
                    mapped[key] = mapped_boxes[keep]
                elif isinstance(value, torch.Tensor) and value.ndim > 0 and value.shape[0] == keep.shape[0]:
                    mapped[key] = value[keep]
                else:
                    mapped[key] = value
            mapped_targets.append(mapped)
        return mapped_targets, {
            "pseudo_boxes_before_mapping": float(before),
            "pseudo_boxes_after_mapping": float(after),
            "pseudo_boxes_dropped_by_mapping": float(before - after),
        }

    def map_outputs_to_strong(self, outputs: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        if "pred_boxes" not in outputs:
            return outputs
        pred_boxes = outputs["pred_boxes"]
        mapped_boxes = pred_boxes.clone()
        valid = torch.ones(pred_boxes.shape[:2], dtype=torch.bool, device=pred_boxes.device)
        for batch_idx in range(pred_boxes.shape[0]):
            boxes, keep = self.map_boxes_to_strong(pred_boxes[batch_idx], batch_idx)
            mapped_boxes[batch_idx] = boxes
            valid[batch_idx] = keep

        mapped_outputs = dict(outputs)
        mapped_outputs["pred_boxes"] = mapped_boxes
        if "pred_logits" in outputs:
            logits = outputs["pred_logits"].clone()
            logits[~valid] = -100.0
            mapped_outputs["pred_logits"] = logits
        return mapped_outputs


class SSODPairAugmentor:
    """Build weak/strong SSOD views and a replayable pseudo-box mapper."""

    def __init__(
        self,
        normalized: bool = False,
        mean: Sequence[float] = (0.485, 0.456, 0.406),
        std: Sequence[float] = (0.229, 0.224, 0.225),
        strong_hflip_prob: float = 0.5,
        strong_crop_prob: float = 0.35,
        strong_crop_scale: Sequence[float] = (0.75, 1.0),
        strong_zoomout_prob: float = 0.20,
        strong_zoomout_scale: Sequence[float] = (0.70, 1.0),
        photometric_prob: float = 0.8,
        grayscale_prob: float = 0.2,
        blur_prob: float = 0.3,
        min_box_size: float = 1.0 / 640.0,
    ) -> None:
        self.normalized = bool(normalized)
        self.mean = tuple(float(v) for v in mean)
        self.std = tuple(float(v) for v in std)
        self.strong_hflip_prob = float(strong_hflip_prob)
        self.strong_crop_prob = float(strong_crop_prob)
        self.strong_crop_scale = tuple(float(v) for v in strong_crop_scale)
        self.strong_zoomout_prob = float(strong_zoomout_prob)
        self.strong_zoomout_scale = tuple(float(v) for v in strong_zoomout_scale)
        self.photometric_prob = float(photometric_prob)
        self.grayscale_prob = float(grayscale_prob)
        self.blur_prob = float(blur_prob)
        self.min_box_size = float(min_box_size)

    def _mean_std(self, images: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        mean = images.new_tensor(self.mean).view(1, -1, 1, 1)
        std = images.new_tensor(self.std).view(1, -1, 1, 1)
        return mean, std

    def _to_unit(self, images: torch.Tensor) -> torch.Tensor:
        if not self.normalized:
            return images
        mean, std = self._mean_std(images)
        return (images * std + mean).clamp(0.0, 1.0)

    def _from_unit(self, images: torch.Tensor) -> torch.Tensor:
        if not self.normalized:
            return images.clamp(0.0, 1.0)
        mean, std = self._mean_std(images)
        return (images.clamp(0.0, 1.0) - mean) / std

    def _maybe_photometric(self, image: torch.Tensor, stats: Dict[str, float]) -> torch.Tensor:
        if random.random() >= self.photometric_prob:
            return image
        unit = self._to_unit(image.unsqueeze(0)).squeeze(0)
        brightness = random.uniform(0.6, 1.4)
        contrast = random.uniform(0.6, 1.4)
        saturation = random.uniform(0.6, 1.4)
        unit = TF.adjust_brightness(unit, brightness)
        unit = TF.adjust_contrast(unit, contrast)
        unit = TF.adjust_saturation(unit, saturation)
        if random.random() < self.grayscale_prob:
            gray = unit.mean(dim=0, keepdim=True)
            unit = gray.repeat(unit.shape[0], 1, 1)
            stats["strong_grayscale_count"] += 1.0
        if random.random() < self.blur_prob:
            kernel = 5 if min(unit.shape[-2:]) >= 5 else 3
            unit = TF.gaussian_blur(unit, kernel_size=[kernel, kernel], sigma=[0.1, 2.0])
            stats["strong_blur_count"] += 1.0
        stats["strong_photometric_count"] += 1.0
        return self._from_unit(unit.unsqueeze(0)).squeeze(0)

    def _crop(self, image: torch.Tensor) -> Tuple[torch.Tensor, Step]:
        _, height, width = image.shape
        scale = random.uniform(*self.strong_crop_scale)
        crop_h = max(2, min(height, int(round(height * scale))))
        crop_w = max(2, min(width, int(round(width * scale))))
        top = random.randint(0, max(height - crop_h, 0))
        left = random.randint(0, max(width - crop_w, 0))
        crop = image[:, top:top + crop_h, left:left + crop_w].unsqueeze(0)
        resized = F.interpolate(crop, size=(height, width), mode="bilinear", align_corners=False).squeeze(0)
        return resized, ("crop", {
            "x0": left / max(width, 1),
            "y0": top / max(height, 1),
            "w": crop_w / max(width, 1),
            "h": crop_h / max(height, 1),
        })

    def _zoomout(self, image: torch.Tensor) -> Tuple[torch.Tensor, Step]:
        _, height, width = image.shape
        scale = random.uniform(*self.strong_zoomout_scale)
        new_h = max(2, min(height, int(round(height * scale))))
        new_w = max(2, min(width, int(round(width * scale))))
        resized = F.interpolate(image.unsqueeze(0), size=(new_h, new_w), mode="bilinear", align_corners=False).squeeze(0)
        top = random.randint(0, max(height - new_h, 0))
        left = random.randint(0, max(width - new_w, 0))
        canvas = image.new_zeros(image.shape)
        canvas[:, top:top + new_h, left:left + new_w] = resized
        return canvas, ("zoomout", {
            "sx": new_w / max(width, 1),
            "sy": new_h / max(height, 1),
            "ox": left / max(width, 1),
            "oy": top / max(height, 1),
        })

    def __call__(self, images: torch.Tensor) -> SSODPairAugmentation:
        weak_images = images.clone()
        strong_images = images.clone()
        transforms: List[_SampleTransform] = []
        stats = {
            "strong_hflip_count": 0.0,
            "strong_crop_count": 0.0,
            "strong_zoomout_count": 0.0,
            "strong_geometric_count": 0.0,
            "strong_photometric_count": 0.0,
            "strong_grayscale_count": 0.0,
            "strong_blur_count": 0.0,
        }

        for batch_idx in range(images.shape[0]):
            image = strong_images[batch_idx]
            steps: List[Step] = []
            crop_applied = False
            if random.random() < self.strong_crop_prob:
                image, step = self._crop(image)
                steps.append(step)
                crop_applied = True
                stats["strong_crop_count"] += 1.0
            elif random.random() < self.strong_zoomout_prob:
                image, step = self._zoomout(image)
                steps.append(step)
                stats["strong_zoomout_count"] += 1.0

            if random.random() < self.strong_hflip_prob:
                image = torch.flip(image, dims=[-1])
                steps.append(("hflip", {}))
                stats["strong_hflip_count"] += 1.0

            if crop_applied or steps:
                stats["strong_geometric_count"] += 1.0

            image = self._maybe_photometric(image, stats)
            strong_images[batch_idx] = image
            transforms.append(_SampleTransform(steps))

        return SSODPairAugmentation(
            weak_images=weak_images,
            strong_images=strong_images,
            transforms=transforms,
            aug_stats=stats,
            min_box_size=self.min_box_size,
        )


def weak_augment(images: torch.Tensor) -> torch.Tensor:
    return images.clone()


def strong_augment(images: torch.Tensor) -> torch.Tensor:
    return SSODPairAugmentor(
        normalized=False,
        strong_hflip_prob=0.0,
        strong_crop_prob=0.0,
        strong_zoomout_prob=0.0,
    )(images).strong_images


def apply_mic_mask(
    images: torch.Tensor,
    patch_size: int = 32,
    mask_ratio: float = 0.5,
    fill_value: Sequence[float] = (0.0, 0.0, 0.0),
) -> Tuple[torch.Tensor, float]:
    """MIC (Hoyer et al., CVPR 2023) Masked Image Consistency on a batch.

    For each sample, partition the image into ``patch_size x patch_size``
    tiles and randomly set a fraction ``mask_ratio`` of them to
    ``fill_value`` (in the same coord space as ``images`` -- e.g. ImageNet
    normalized mean if the batch is already normalized).

    Returns (masked_images, achieved_mask_fraction). The achieved fraction is
    the empirical mean across the batch and is exposed for logging.

    Notes:
    - Applied to STUDENT input only. Teacher must never see the mask.
    - ``fill_value`` length must match the channel count (typically 3).
    - For DETR-family the recommended ratio is 0.3-0.5; >0.7 has been
      reported to destabilize deformable-attention sampling offsets.
    """
    if patch_size <= 0 or mask_ratio <= 0.0:
        return images, 0.0
    if images.ndim != 4:
        raise ValueError(f"apply_mic_mask expects 4D (B,C,H,W) tensor, got {tuple(images.shape)}")
    batch_size, channels, height, width = images.shape
    if len(fill_value) != channels:
        raise ValueError(
            f"fill_value length {len(fill_value)} does not match channel count {channels}"
        )
    gh = max(1, math.ceil(height / patch_size))
    gw = max(1, math.ceil(width / patch_size))
    masked = images.clone()
    fill_tensor = images.new_tensor(list(fill_value)).view(1, channels, 1, 1)
    total_kept = 0.0
    total_cells = 0.0
    for batch_idx in range(batch_size):
        # Random binary patch grid: 1 = keep, 0 = mask.
        cell_mask = (torch.rand(gh, gw, device=images.device) >= float(mask_ratio)).to(images.dtype)
        # Upsample to full resolution via nearest-neighbour repeat.
        full_mask = cell_mask.repeat_interleave(patch_size, dim=0).repeat_interleave(patch_size, dim=1)
        full_mask = full_mask[:height, :width].unsqueeze(0)  # (1, H, W)
        masked[batch_idx] = images[batch_idx] * full_mask + fill_tensor.squeeze(0) * (1.0 - full_mask)
        total_kept += float(cell_mask.sum().item())
        total_cells += float(gh * gw)
    achieved = 1.0 - (total_kept / max(total_cells, 1.0))
    return masked, achieved
