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
        # Build the standard ConvNeXtV2 model so we can reuse its stem + stages.
        self.model = timm.create_model(
            "convnextv2_atto.fcmae_ft_in1k",
            pretrained=pretrained,
        )

        self.stem = self.model.stem
        self.stages = self.model.stages
        self.norm_pre = getattr(self.model, "norm_pre", nn.Identity())
        head = getattr(self.model, "head", None)
        self.head_norm = getattr(head, "norm", nn.Identity())

        dims = getattr(self.model, "dims", (40, 80, 160, 320))
        # GAPNet expects five spatial feature maps; mirror MobileNet by creating
        # pseudo 1/2 resolution features via upsampling stage-0 activations.
        self.out_channels: List[int] = [
            dims[0],
            dims[0],
            dims[1],
            dims[2],
            dims[3],
        ]

    def forward(self, x: torch.Tensor) -> List[torch.Tensor]:
        features: List[torch.Tensor] = []

        x = self.stem(x)
        x = self.stages[0](x)

        features.append(F.interpolate(x, scale_factor=2.0, mode="bilinear", align_corners=False))
        features.append(x)

        for stage in self.stages[1:]:
            x = stage(x)
            features.append(x)

        # Match the representation used by the classification head prior to pooling.
        features[-1] = self.head_norm(self.norm_pre(features[-1]))

        return features


def convnextv2_atto(pretrained: bool = True) -> ConvNeXtV2AttoBackbone:
    """Factory that mimics the MobileNetV2 helper signature used in GAPNet."""

    return ConvNeXtV2AttoBackbone(pretrained=pretrained)
