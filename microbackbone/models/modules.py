"""Core building blocks for MicroSign-Net backbone."""
from __future__ import annotations

from typing import List, Optional

import torch
import torch.nn as nn


class ChannelShuffle(nn.Module):
    """Channel shuffle utility for grouped convolutions."""

    def __init__(self, groups: int) -> None:
        super().__init__()
        self.groups = groups

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # noqa: D401
        b, c, h, w = x.size()
        channels_per_group = c // self.groups
        x = x.view(b, self.groups, channels_per_group, h, w)
        x = x.transpose(1, 2).contiguous()
        return x.view(b, c, h, w)


class DynamicConv(nn.Module):
    """Dynamic convolution with multiple kernels and attention-based weighting."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        stride: int = 1,
        num_kernels: int = 4,
    ) -> None:
        super().__init__()
        padding = kernel_size // 2
        self.num_kernels = num_kernels

        self.kernels = nn.ModuleList(
            [
                nn.Conv2d(
                    in_channels,
                    out_channels,
                    kernel_size,
                    stride,
                    padding,
                    bias=False,
                    groups=1,
                )
                for _ in range(num_kernels)
            ]
        )

        self.attention = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels, num_kernels, 1),
            nn.Softmax(dim=1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # noqa: D401
        attn = self.attention(x)
        outputs = [kernel(x) for kernel in self.kernels]
        return sum(attn[:, i : i + 1] * outputs[i] for i in range(self.num_kernels))


class SEBlock(nn.Module):
    """Squeeze-and-Excitation block."""

    def __init__(self, channels: int, reduction: int = 4) -> None:
        super().__init__()
        squeezed = max(channels // reduction, 8)
        self.squeeze = nn.AdaptiveAvgPool2d(1)
        self.excitation = nn.Sequential(
            nn.Conv2d(channels, squeezed, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(squeezed, channels, 1),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # noqa: D401
        weights = self.excitation(self.squeeze(x))
        return x * weights


class EfficientBlock(nn.Module):
    """Inverted residual block with optional dynamic depthwise and SE."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        stride: int = 1,
        expansion: int = 4,
        use_dynamic: bool = False,
        use_se: bool = True,
    ) -> None:
        super().__init__()
        self.stride = stride
        self.use_residual = stride == 1 and in_channels == out_channels
        hidden_dim = in_channels * expansion

        self.expand: Optional[nn.Sequential]
        if expansion != 1:
            self.expand = nn.Sequential(
                nn.Conv2d(in_channels, hidden_dim, 1, bias=False),
                nn.BatchNorm2d(hidden_dim),
                nn.ReLU6(inplace=True),
            )
        else:
            self.expand = None

        dw_channels = hidden_dim if expansion != 1 else in_channels
        if use_dynamic and stride == 1:
            depthwise_layer: nn.Module = DynamicConv(
                dw_channels, dw_channels, kernel_size=3, stride=stride, num_kernels=3
            )
        else:
            depthwise_layer = nn.Conv2d(
                dw_channels, dw_channels, 3, stride, 1, groups=dw_channels, bias=False
            )

        self.depthwise = nn.Sequential(
            depthwise_layer,
            nn.BatchNorm2d(dw_channels),
            nn.ReLU6(inplace=True),
        )

        self.se = SEBlock(dw_channels) if use_se else None

        self.project = nn.Sequential(
            nn.Conv2d(dw_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
        )

        self.shuffle = ChannelShuffle(groups=4) if in_channels >= 16 and out_channels >= 16 else None

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # noqa: D401
        identity = x
        out = self.expand(x) if self.expand is not None else x
        out = self.depthwise(out)
        if self.se is not None:
            out = self.se(out)
        out = self.project(out)
        if self.use_residual:
            out = out + identity
        if self.shuffle is not None:
            out = self.shuffle(out)
        return out


class SpatialAttention(nn.Module):
    """Lightweight spatial attention for salient regions."""

    def __init__(self, kernel_size: int = 7) -> None:
        super().__init__()
        self.conv = nn.Conv2d(2, 1, kernel_size, padding=kernel_size // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # noqa: D401
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        attn = self.sigmoid(self.conv(torch.cat([avg_out, max_out], dim=1)))
        return x * attn


__all__ = [
    "ChannelShuffle",
    "DynamicConv",
    "SEBlock",
    "EfficientBlock",
    "SpatialAttention",
]
