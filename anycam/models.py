"""Models.

Baseline (Step 5): an ImageNet-pretrained CNN whose first convolution is widened to take
all hyperspectral bands (204 for HyperLeaf). The new first-layer weights start from the
pretrained RGB filters averaged over colour and spread over the bands, so the rest of the
network keeps its pretrained features. Two heads: cultivar (classification) and four
regression targets, trained together (multi-task).
"""
from __future__ import annotations

import torch
import torch.nn as nn


def _widen_first_conv(model: nn.Module, in_channels: int) -> None:
    """Replace the first conv of a timm model so it accepts `in_channels` inputs."""
    name = model.default_cfg.get("first_conv", "conv1") if hasattr(model, "default_cfg") else "conv1"
    parent = model
    parts = name.split(".")
    for p in parts[:-1]:
        parent = getattr(parent, p)
    old: nn.Conv2d = getattr(parent, parts[-1])
    new = nn.Conv2d(in_channels, old.out_channels, old.kernel_size, old.stride, old.padding,
                    old.dilation, old.groups, old.bias is not None)
    with torch.no_grad():
        w = old.weight.mean(dim=1, keepdim=True)                       # (out, 1, k, k)
        new.weight.copy_(w.repeat(1, in_channels, 1, 1) * (old.in_channels / in_channels))
        if old.bias is not None:
            new.bias.copy_(old.bias)
    setattr(parent, parts[-1], new)


class MultiTaskNet(nn.Module):
    def __init__(self, in_channels: int, n_classes: int = 4, n_reg: int = 4,
                 backbone: str = "resnet18", pretrained: bool = True, dropout: float = 0.2,
                 upscale: float = 1.0):
        super().__init__()
        self.upscale = upscale
        import timm
        self.backbone = timm.create_model(backbone, pretrained=pretrained, num_classes=0, global_pool="avg")
        _widen_first_conv(self.backbone, in_channels)
        d = self.backbone.num_features
        self.drop = nn.Dropout(dropout)
        self.cls = nn.Linear(d, n_classes)
        self.reg = nn.Linear(d, n_reg)

    def forward(self, x):
        if self.upscale != 1.0:  # small leaf images: enlarge so the CNN keeps more spatial detail
            x = nn.functional.interpolate(x, scale_factor=self.upscale, mode="bilinear", align_corners=False)
        z = self.drop(self.backbone(x))
        return self.cls(z), self.reg(z)


def n_params(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())
