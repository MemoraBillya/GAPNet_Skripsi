"""ConvNeXtV2 backbone wrappers used by GAPNet."""

from typing import List

import torch
import torch.nn as nn
import torch.nn.functional as F
import timm


class ConvNeXtV2AttoBackbone(nn.Module):
    """Feature extractor that mirrors the MobileNetV2 interface for GAPNet."""

    def __init__(self, pretrained: bool = True) -> None:
        super().__init__()
        # We keep the original ConvNeXtV2 network so we can reuse its stages directly.
        self.model = timm.create_model(
            "convnextv2_atto.fcmae_ft_in1k",
            pretrained=pretrained,
            features_only=False,
        )
        # The official implementation exposes downsample layers and stages, which we cache
        # for readability and to avoid attribute lookups in forward().
        self.downsample_layers = self.model.downsample_layers
        self.stages = self.model.stages
        self.norm = self.model.norm
        # GAPNet expects five spatial feature maps; we expose their channel dimensions so
        # the decoder can be configured accordingly.
        self.out_channels: List[int] = [
            self.model.dims[0],
            self.model.dims[0],
            self.model.dims[1],
            self.model.dims[2],
            self.model.dims[3],
        ]

    def forward(self, x: torch.Tensor) -> List[torch.Tensor]:
        features: List[torch.Tensor] = []

        # Stage 0 keeps 1/4 resolution features; we upsample once to emulate the missing
        # stride-2 stage that MobileNetV2 originally provided to the decoder.
        x = self.downsample_layers[0](x)
        x = self.stages[0](x)
        features.append(F.interpolate(x, scale_factor=2.0, mode="bilinear", align_corners=False))
        features.append(x)

        # Remaining ConvNeXtV2 stages naturally produce the 1/8, 1/16, and 1/32 scales.
        for stage_index in range(1, len(self.stages)):
            x = self.downsample_layers[stage_index](x)
            x = self.stages[stage_index](x)
            features.append(x)

        # ConvNeXtV2 applies a final LayerNorm before pooling; we keep it so GAPNet receives
        # the same representation that the classification head would normally consume.
        features[-1] = self.norm(features[-1])

        return features


def convnextv2_atto(pretrained: bool = True) -> ConvNeXtV2AttoBackbone:
    """Factory that mimics the MobileNetV2 helper signature used in GAPNet."""

    return ConvNeXtV2AttoBackbone(pretrained=pretrained)
