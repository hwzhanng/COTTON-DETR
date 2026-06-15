"""
DEIMv2: Real-Time Object Detection Meets DINOv3
Copyright (c) 2025 The DEIMv2 Authors. All Rights Reserved.
---------------------------------------------------------------------------------
Modified from DINOv3 (https://github.com/facebookresearch/dinov3)

Copyright (c) Meta Platforms, Inc. and affiliates.

This software may be used and distributed in accordance with
the terms of the DINOv3 License Agreement.
"""

import os

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as cp

from functools import partial
from ..core import register
from .vit_tiny import VisionTransformer
from .dinov3 import DinoVisionTransformer
from .dinov3_backbone import DINOv3Backbone


class SpatialPriorModulev2(nn.Module):
    def __init__(self, inplanes=16):
        super().__init__()

        # 1/4
        self.stem = nn.Sequential(
            *[
                nn.Conv2d(3, inplanes, kernel_size=3, stride=2, padding=1, bias=False),
                nn.SyncBatchNorm(inplanes),
                nn.GELU(),
                nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
            ]
        )
        # 1/8
        self.conv2 = nn.Sequential(
            *[
                nn.Conv2d(inplanes, 2 * inplanes, kernel_size=3, stride=2, padding=1, bias=False),
                nn.SyncBatchNorm(2 * inplanes),
            ]
        )
        # 1/16
        self.conv3 = nn.Sequential(
            *[
                nn.GELU(),
                nn.Conv2d(2 * inplanes, 4 * inplanes, kernel_size=3, stride=2, padding=1, bias=False),
                nn.SyncBatchNorm(4 * inplanes),
            ]
        )
        # 1/32
        self.conv4 = nn.Sequential(
            *[
                nn.GELU(),
                nn.Conv2d(4 * inplanes, 4 * inplanes, kernel_size=3, stride=2, padding=1, bias=False),
                nn.SyncBatchNorm(4 * inplanes),
            ]
        )

    def forward(self, x):
        c1 = self.stem(x)
        c2 = self.conv2(c1)     # 1/8
        c3 = self.conv3(c2)     # 1/16
        c4 = self.conv4(c3)     # 1/32

        return c2, c3, c4


@register()
class DINOv3STAs(nn.Module):
    def __init__(
        self,
        name=None,
        weights_path=None,
        interaction_indexes=[],
        finetune=True,
        embed_dim=192,
        num_heads=3,
        patch_size=16,
        use_sta=True,
        conv_inplane=16,
        hidden_dim=None,
    ):
        super(DINOv3STAs, self).__init__()
        if 'dinov3' in name:
            self.dinov3 = DinoVisionTransformer(name=name)
            if weights_path is not None and os.path.exists(weights_path):
                print(f'Loading ckpt from {weights_path}...')
                self.dinov3.load_state_dict(torch.load(weights_path))
            else:
                print('Training DINOv3 from scratch...')
        else:
            self.dinov3 =  VisionTransformer(embed_dim=embed_dim, num_heads=num_heads, return_layers=interaction_indexes)
            if weights_path is not None and os.path.exists(weights_path):
                print(f'Loading ckpt from {weights_path}...')
                self.dinov3._model.load_state_dict(torch.load(weights_path))
            else:
                print('Training ViT-Tiny from scratch...')

        embed_dim = self.dinov3.embed_dim
        self.interaction_indexes = interaction_indexes
        self.patch_size = patch_size

        if not finetune:
            self.dinov3.eval()
            self.dinov3.requires_grad_(False)

        # init the feature pyramid
        self.use_sta = use_sta
        if use_sta:
            print(f"Using Lite Spatial Prior Module with inplanes={conv_inplane}")
            self.sta = SpatialPriorModulev2(inplanes=conv_inplane)
        else:
            conv_inplane = 0

        # linear projection
        hidden_dim = hidden_dim if hidden_dim is not None else embed_dim
        self.convs = nn.ModuleList([
            nn.Conv2d(embed_dim + conv_inplane*2, hidden_dim, kernel_size=1, stride=1, padding=0, bias=False),
            nn.Conv2d(embed_dim + conv_inplane*4, hidden_dim, kernel_size=1, stride=1, padding=0, bias=False),
            nn.Conv2d(embed_dim + conv_inplane*4, hidden_dim, kernel_size=1, stride=1, padding=0, bias=False)
        ])
        # norm
        self.norms = nn.ModuleList([
            nn.SyncBatchNorm(hidden_dim),
            nn.SyncBatchNorm(hidden_dim),
            nn.SyncBatchNorm(hidden_dim)
        ])

    def forward(self, x):
        # Code for matching with oss
        H_c, W_c = x.shape[2] // 16, x.shape[3] // 16
        H_toks, W_toks = x.shape[2] // self.patch_size, x.shape[3] // self.patch_size
        bs, C, h, w = x.shape

        if len(self.interaction_indexes) > 0 and not isinstance(self.dinov3, VisionTransformer):
            all_layers = self.dinov3.get_intermediate_layers(
                x, n=self.interaction_indexes, return_class_token=True
            )
        else:
            all_layers = self.dinov3(x)

        if len(all_layers) == 1:    # repeat the same layer for all the three scales
            all_layers = [all_layers[0], all_layers[0], all_layers[0]]
        
        sem_feats = []
        num_scales = len(all_layers) - 2
        for i, sem_feat in enumerate(all_layers):
            feat, _ = sem_feat
            sem_feat = feat.transpose(1, 2).view(bs, -1, H_c, W_c).contiguous()  # [B, D, H, W]
            resize_H, resize_W = int(H_c * 2**(num_scales-i)), int(W_c * 2**(num_scales-i))
            sem_feat = F.interpolate(sem_feat, size=[resize_H, resize_W], mode="bilinear", align_corners=False)
            sem_feats.append(sem_feat)

        # fusion
        fused_feats = []
        if self.use_sta:
            detail_feats = self.sta(x)
            for sem_feat, detail_feat in zip(sem_feats, detail_feats):
                fused_feats.append(torch.cat([sem_feat, detail_feat], dim=1))
        else:
            fused_feats = sem_feats

        c2 = self.norms[0](self.convs[0](fused_feats[0]))
        c3 = self.norms[1](self.convs[1](fused_feats[1]))
        c4 = self.norms[2](self.convs[2](fused_feats[2]))

        return c2, c3, c4


@register()
class DINOv3STAsHF(nn.Module):
    """Backbone upgrade (v6): FROZEN full DINOv3-S (384d, HF weights) + the same
    Spatial-Tuning-Adapter fusion as ``DINOv3STAs``, projected down to ``hidden_dim``
    so the detection head (encoder/decoder) is byte-identical to v5.

    Rationale: the distilled ViT-Tiny(192) backbone is the measured ceiling; the
    frozen full DINOv3-S has far stronger features (probe: Def 0.746 vs 0.63). We
    reuse the already-tested ``DINOv3Backbone`` HF extractor (no facebook-format
    weight conversion -> no silent weight-mapping bugs) and the tested
    ``SpatialPriorModulev2`` detail branch. With ``freeze=True`` the big ViT adds
    ZERO trainable params -- only the STA + 1x1 projections + the head train.
    """

    def __init__(
        self,
        weights_dir="weights/pretrained/dinov3_vits16",
        interaction_indexes=(4, 8, 12),
        out_strides=(8, 16, 32),
        hidden_dim=192,
        conv_inplane=16,
        use_sta=True,
        freeze=True,
        normalize=False,
    ):
        super().__init__()
        # Frozen HF DINOv3-S feature extractor: 3 maps at the requested strides,
        # each ``embed_dim`` (384) channels. normalize=False because the SSOD
        # dataloader already ImageNet-normalises the batch (avoid double-norm).
        self.extractor = DINOv3Backbone(
            weights_dir=weights_dir,
            out_indices=tuple(interaction_indexes),
            out_strides=tuple(out_strides),
            freeze=freeze,
            normalize=normalize,
        )
        embed_dim = int(self.extractor.hidden_size)
        self.use_sta = bool(use_sta)
        if self.use_sta:
            print(f"DINOv3STAsHF: frozen DINOv3-S({embed_dim}) + STA(inplanes={conv_inplane}) -> hidden_dim={hidden_dim}")
            self.sta = SpatialPriorModulev2(inplanes=conv_inplane)
        else:
            conv_inplane = 0
            print(f"DINOv3STAsHF: frozen DINOv3-S({embed_dim}) (no STA) -> hidden_dim={hidden_dim}")

        hidden_dim = hidden_dim if hidden_dim is not None else embed_dim
        self.convs = nn.ModuleList([
            nn.Conv2d(embed_dim + conv_inplane * 2, hidden_dim, kernel_size=1, stride=1, padding=0, bias=False),
            nn.Conv2d(embed_dim + conv_inplane * 4, hidden_dim, kernel_size=1, stride=1, padding=0, bias=False),
            nn.Conv2d(embed_dim + conv_inplane * 4, hidden_dim, kernel_size=1, stride=1, padding=0, bias=False),
        ])
        self.norms = nn.ModuleList([
            nn.SyncBatchNorm(hidden_dim),
            nn.SyncBatchNorm(hidden_dim),
            nn.SyncBatchNorm(hidden_dim),
        ])

    def forward(self, x):
        sem_feats = self.extractor(x)  # 3 x [B, 384, H/s, W/s] at strides 8/16/32
        if self.use_sta:
            detail_feats = self.sta(x)  # c2,c3,c4 at strides 8/16/32
            fused = [torch.cat([s, d], dim=1) for s, d in zip(sem_feats, detail_feats)]
        else:
            fused = sem_feats
        c2 = self.norms[0](self.convs[0](fused[0]))
        c3 = self.norms[1](self.convs[1](fused[1]))
        c4 = self.norms[2](self.convs[2](fused[2]))
        return c2, c3, c4