"""
Copyright (c) 2024 The DEIM Authors. All Rights Reserved.
"""

import inspect

import torch.nn as nn
from ..core import register


__all__ = ['DEIM', ]


@register()
class DEIM(nn.Module):
    __inject__ = ['backbone', 'encoder', 'decoder', ]

    def __init__(self, \
        backbone: nn.Module,
        encoder: nn.Module,
        decoder: nn.Module,
    ):
        super().__init__()
        self.backbone = backbone
        self.decoder = decoder
        self.encoder = encoder

    def _decoder_accepts_kwarg(self, name: str) -> bool:
        try:
            return name in inspect.signature(self.decoder.forward).parameters
        except (TypeError, ValueError):
            return False

    def _decoder_accepts_consistency_queries(self):
        # Kept for backward compatibility with any external callers.
        return self._decoder_accepts_kwarg("consistency_queries")

    def _decode(self, feats, targets=None, consistency_queries=None, dn_noise_override=None):
        kwargs = {}
        if consistency_queries is not None and self._decoder_accepts_kwarg("consistency_queries"):
            kwargs["consistency_queries"] = consistency_queries
        # T9: forward DN noise override only when decoder supports it. Older
        # decoders (D-FINE / DFINETransformer used in the smoke config) do not
        # accept this kwarg yet; gracefully fall back to their default noise.
        if dn_noise_override is not None and self._decoder_accepts_kwarg("dn_noise_override"):
            kwargs["dn_noise_override"] = dn_noise_override
        if kwargs:
            return self.decoder(feats, targets, **kwargs)
        return self.decoder(feats, targets)

    def forward(self, x, targets=None, consistency_queries=None, dn_noise_override=None):
        x = self.backbone(x)
        x = self.encoder(x)
        x = self._decode(x, targets, consistency_queries=consistency_queries, dn_noise_override=dn_noise_override)

        return x

    def extract_encoded_features(self, x):
        return self.encoder(self.backbone(x))

    def decode_features(self, feats, targets=None, consistency_queries=None, dn_noise_override=None):
        return self._decode(feats, targets, consistency_queries=consistency_queries, dn_noise_override=dn_noise_override)

    def deploy(self, ):
        self.eval()
        for m in self.modules():
            if hasattr(m, 'convert_to_deploy'):
                m.convert_to_deploy()
        return self
