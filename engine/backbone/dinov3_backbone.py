"""
Frozen DINOv3 ViT backbone adapter for DEIMv2.

The DEIMv2 training environment already contains transformers and safetensors,
so this adapter uses the local HuggingFace-format DINOv3 cache instead of timm.
"""

import os
from pathlib import Path
from shutil import copy2
from typing import Iterable, Optional, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..core import register


def _has_non_ascii(path: Path) -> bool:
    try:
        str(path).encode("ascii")
    except UnicodeEncodeError:
        return True
    return False


@register()
class DINOv3Backbone(nn.Module):
    """Frozen DINOv3 ViT-S/16 feature extractor.

    It returns three 2D feature maps shaped for the existing DEIMv2
    HybridEncoder. Since ViT patch tokens are naturally stride 16, the first
    and third outputs are resized to emulate stride 8 and stride 32 features.
    """

    def __init__(
        self,
        weights_dir: Optional[str] = None,
        stage_dir: str = "weights/pretrained/dinov3_vits16",
        out_indices: Sequence[int] = (4, 8, 12),
        out_strides: Sequence[int] = (8, 16, 32),
        freeze: bool = True,
        normalize: bool = True,
    ) -> None:
        super().__init__()
        self.out_indices = tuple(out_indices)
        self.out_strides = tuple(out_strides)
        self.freeze = freeze
        self.normalize = normalize
        self.patch_size = 16
        self.out_channels = [384, 384, 384]
        self.out_strides_list = list(self.out_strides)
        self.channels = self.out_channels
        self.strides = self.out_strides_list
        self.return_idx = list(range(len(self.out_strides)))

        model_dir = self._prepare_model_dir(weights_dir, stage_dir)
        from transformers import AutoModel

        self.model = AutoModel.from_pretrained(str(model_dir), local_files_only=True)
        self.hidden_size = int(getattr(self.model.config, "hidden_size", 384))
        self.num_register_tokens = int(getattr(self.model.config, "num_register_tokens", 4))
        self.out_channels = [self.hidden_size for _ in self.out_strides]
        self.channels = self.out_channels

        if freeze:
            self.model.eval()
            for param in self.model.parameters():
                param.requires_grad_(False)

        mean = torch.tensor([0.485, 0.456, 0.406], dtype=torch.float32).view(1, 3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225], dtype=torch.float32).view(1, 3, 1, 1)
        self.register_buffer("image_mean", mean, persistent=False)
        self.register_buffer("image_std", std, persistent=False)

    @staticmethod
    def _repo_root() -> Path:
        return Path(__file__).resolve().parents[2]

    def _candidate_dirs(self, weights_dir: Optional[str]) -> Iterable[Path]:
        if weights_dir:
            yield Path(weights_dir)

        env_dir = os.environ.get("DINOV3_VITS16_DIR")
        if env_dir:
            yield Path(env_dir)
        yield self._repo_root() / "weights/pretrained/dinov3_vits16"

    def _prepare_model_dir(self, weights_dir: Optional[str], stage_dir: str) -> Path:
        source_dir = None
        for candidate in self._candidate_dirs(weights_dir):
            if (candidate / "config.json").exists() and (candidate / "model.safetensors").exists():
                source_dir = candidate
                break
        if source_dir is None:
            raise FileNotFoundError(
                "DINOv3 ViT-S local weights were not found. Provide DINOv3Backbone.weights_dir "
                "with config.json and model.safetensors, set DINOV3_VITS16_DIR, or place the files under "
                "weights/pretrained/dinov3_vits16."
            )

        if not _has_non_ascii(source_dir):
            return source_dir

        staged = self._repo_root() / stage_dir
        staged.mkdir(parents=True, exist_ok=True)
        for name in ("config.json", "model.safetensors", "preprocessor_config.json"):
            src = source_dir / name
            if not src.exists():
                continue
            dst = staged / name
            if not dst.exists() or dst.stat().st_size != src.stat().st_size:
                copy2(src, dst)
        return staged

    def train(self, mode: bool = True):
        super().train(mode)
        if self.freeze:
            self.model.eval()
        return self

    def forward(self, x: torch.Tensor):
        if x.shape[-2] % self.patch_size != 0 or x.shape[-1] % self.patch_size != 0:
            raise ValueError(f"DINOv3Backbone expects image sizes divisible by {self.patch_size}, got {tuple(x.shape[-2:])}.")

        if self.normalize:
            x = (x - self.image_mean.to(dtype=x.dtype)) / self.image_std.to(dtype=x.dtype)

        grad_enabled = torch.is_grad_enabled() and not self.freeze
        with torch.set_grad_enabled(grad_enabled):
            outputs = self.model(x, output_hidden_states=True, return_dict=True)

        hidden_states = outputs.hidden_states
        batch_size, _, height, width = x.shape
        patch_h, patch_w = height // self.patch_size, width // self.patch_size
        token_offset = 1 + self.num_register_tokens
        features = []

        for layer_idx, stride in zip(self.out_indices, self.out_strides):
            tokens = hidden_states[layer_idx][:, token_offset:, :]
            fmap = tokens.reshape(batch_size, patch_h, patch_w, self.hidden_size)
            fmap = fmap.permute(0, 3, 1, 2).contiguous()
            if stride != self.patch_size:
                size = (height // stride, width // stride)
                fmap = F.interpolate(fmap, size=size, mode="bilinear", align_corners=False)
            features.append(fmap)

        return features
