"""Core building blocks for the MicroSign-Edge backbone family."""
from __future__ import annotations

import math
from typing import List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class ChannelShuffle(nn.Module):
    """Channel shuffle utility reused for shift-mix stages."""

    def __init__(self, groups: int) -> None:
        super().__init__()
        self.groups = groups

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # noqa: D401
        b, c, h, w = x.size()
        channels_per_group = c // self.groups
        x = x.view(b, self.groups, channels_per_group, h, w)
        x = x.transpose(1, 2).contiguous()
        return x.view(b, c, h, w)


def _fuse_conv_bn(conv: nn.Conv2d, bn: nn.BatchNorm2d) -> Tuple[torch.Tensor, torch.Tensor]:
    """Return fused conv weights and bias for a Conv2d + BatchNorm2d pair."""

    if conv.bias is None:
        conv_bias = torch.zeros(conv.weight.size(0), device=conv.weight.device)
    else:
        conv_bias = conv.bias

    w = conv.weight
    bn_var_rsqrt = torch.rsqrt(bn.running_var + bn.eps)
    scale = bn.weight * bn_var_rsqrt
    fused_weight = w * scale.reshape(-1, 1, 1, 1)
    fused_bias = bn.bias + (conv_bias - bn.running_mean) * scale
    return fused_weight, fused_bias


def _shift_groups(x: torch.Tensor) -> torch.Tensor:
    """Zero-FLOP spatial shifts for four channel groups (up/left/down/right)."""

    b, c, h, w = x.shape
    if c < 4:
        return x

    split = torch.chunk(x, 4, dim=1)
    up = F.pad(split[0], (0, 0, 1, 0))[:, :, :h, :]
    left = F.pad(split[1], (1, 0, 0, 0))[:, :, :, :w]
    down = F.pad(split[2], (0, 0, 0, 1))[:, :, 1:, :]
    right = F.pad(split[3], (0, 1, 0, 0))[:, :, :, 1:]
    return torch.cat([up, left, down, right], dim=1)


def _nearest_divisor(value: int, target: int) -> int:
    """Return the largest divisor of ``value`` that is <= target (>=1)."""

    target = max(1, target)
    for g in range(target, 0, -1):
        if value % g == 0:
            return g
    return 1


class ReparamShiftDepthwiseBlock(nn.Module):
    """RSDBlock: rich training graph, single-path inference for MCUs.

    Training-time structure:
    - optional 1x1 expansion + BN + SiLU
    - shift-mix (directional shifts + channel shuffle)
    - depthwise 3x3 branch + BN plus identity DW 1x1 branch -> summed then BN + activation
    - grouped SE-like gating
    - 1x1 projection + BN with residual when shapes match

    switch_to_deploy() fuses Conv+BN pairs and merges branches so inference reduces to
    shift -> depthwise3x3 -> BN -> activation -> pointwise -> BN -> (residual) -> activation.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        stride: int = 1,
        expansion: int = 2,
        activation: str = "silu",
    ) -> None:
        super().__init__()
        self.stride = stride
        self.use_residual = stride == 1 and in_channels == out_channels
        self.mid_channels = in_channels * expansion
        self.activation = nn.SiLU(inplace=True) if activation == "silu" else nn.ReLU(inplace=True)
        self.deploy = False

        self.expand_conv: nn.Conv2d | None = None
        self.expand_bn: nn.BatchNorm2d | None = None
        if expansion != 1:
            self.expand_conv = nn.Conv2d(in_channels, self.mid_channels, 1, bias=False)
            self.expand_bn = nn.BatchNorm2d(self.mid_channels)
        else:
            self.mid_channels = in_channels

        self.shift_shuffle = ChannelShuffle(groups=4)
        self.depthwise = nn.Conv2d(
            self.mid_channels,
            self.mid_channels,
            3,
            stride=stride,
            padding=1,
            groups=self.mid_channels,
            bias=False,
        )
        self.depthwise_bn = nn.BatchNorm2d(self.mid_channels)

        self.identity_conv: nn.Conv2d | None = None
        self.identity_bn: nn.BatchNorm2d | None = None
        if stride == 1:
            self.identity_conv = nn.Conv2d(
                self.mid_channels,
                self.mid_channels,
                1,
                groups=self.mid_channels,
                bias=False,
            )
            self.identity_bn = nn.BatchNorm2d(self.mid_channels)

        self.merge_bn = nn.BatchNorm2d(self.mid_channels)

        # Grouped SE-like gating
        groups = _nearest_divisor(self.mid_channels, max(1, self.mid_channels // 16))
        reduced = max(self.mid_channels // 8, 4)
        # ensure reduced channels are divisible by groups for grouped 1x1s
        reduced = int(math.ceil(reduced / groups) * groups)
        self.se_reduce = nn.Conv2d(self.mid_channels, reduced, 1, groups=groups)
        self.se_expand = nn.Conv2d(reduced, self.mid_channels, 1, groups=groups)

        self.project = nn.Conv2d(self.mid_channels, out_channels, 1, bias=False)
        self.project_bn = nn.BatchNorm2d(out_channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # noqa: D401
        if self.deploy:
            return self._forward_deploy(x)

        out = x
        if self.expand_conv is not None and self.expand_bn is not None:
            out = self.activation(self.expand_bn(self.expand_conv(out)))

        out = _shift_groups(out)
        out = self.shift_shuffle(out)

        dw_out = self.depthwise_bn(self.depthwise(out))
        if self.identity_conv is not None and self.identity_bn is not None:
            id_out = self.identity_bn(self.identity_conv(out))
            out = dw_out + id_out
        else:
            out = dw_out

        out = self.activation(self.merge_bn(out))

        # SE-like gating
        se = F.adaptive_avg_pool2d(out, 1)
        se = F.relu(self.se_reduce(se))
        se = torch.sigmoid(self.se_expand(se))
        out = out * se

        out = self.project_bn(self.project(out))
        if self.use_residual:
            out = out + x
        return self.activation(out)

    def _forward_deploy(self, x: torch.Tensor) -> torch.Tensor:
        out = x
        if self.expand_conv is not None:
            out = self.activation(self.expand_conv(out))

        out = _shift_groups(out)
        out = self.shift_shuffle(out)
        out = self.activation(self.reparam_depthwise(out))
        if self.se_reduce is not None and self.se_expand is not None:
            se = F.adaptive_avg_pool2d(out, 1)
            se = F.relu(self.se_reduce(se))
            se = torch.sigmoid(self.se_expand(se))
            out = out * se
        out = self.project(out)
        if self.use_residual:
            out = out + x
        return self.activation(out)

    def _get_equivalent_depthwise(self) -> Tuple[torch.Tensor, torch.Tensor]:
        dw_kernel, dw_bias = _fuse_conv_bn(self.depthwise, self.depthwise_bn)

        if self.identity_conv is not None and self.identity_bn is not None:
            id_kernel, id_bias = _fuse_conv_bn(self.identity_conv, self.identity_bn)
            # expand 1x1 depthwise kernel to 3x3 center
            id_kernel_expanded = torch.zeros_like(dw_kernel)
            center = id_kernel.shape[-1] // 2
            id_kernel_expanded[:, :, center, center] = id_kernel.squeeze(-1).squeeze(-1)
            merged_kernel = dw_kernel + id_kernel_expanded
            merged_bias = dw_bias + id_bias
        else:
            merged_kernel, merged_bias = dw_kernel, dw_bias

        # fuse merge_bn
        bn = self.merge_bn
        bn_var_rsqrt = torch.rsqrt(bn.running_var + bn.eps)
        scale = bn.weight * bn_var_rsqrt
        fused_kernel = merged_kernel * scale.reshape(-1, 1, 1, 1)
        fused_bias = bn.bias + (merged_bias - bn.running_mean) * scale
        return fused_kernel, fused_bias

    def switch_to_deploy(self) -> None:
        """Fuse branches and BNs to a single depthwise + pointwise path."""

        if self.deploy:
            return

        # Fuse depthwise path
        dw_kernel, dw_bias = self._get_equivalent_depthwise()
        self.reparam_depthwise = nn.Conv2d(
            self.mid_channels,
            self.mid_channels,
            3,
            stride=self.stride,
            padding=1,
            groups=self.mid_channels,
            bias=True,
        )
        self.reparam_depthwise.weight.data = dw_kernel
        self.reparam_depthwise.bias.data = dw_bias

        # Fuse expand and project BNs if they exist
        if self.expand_conv is not None and self.expand_bn is not None:
            fused_w, fused_b = _fuse_conv_bn(self.expand_conv, self.expand_bn)
            self.expand_conv.weight.data = fused_w
            self.expand_conv.bias = nn.Parameter(fused_b)
            self.expand_bn = None

        fused_proj_w, fused_proj_b = _fuse_conv_bn(self.project, self.project_bn)
        self.project.weight.data = fused_proj_w
        self.project.bias = nn.Parameter(fused_proj_b)
        self.project_bn = None

        # Remove training-only branches
        self.depthwise = None  # type: ignore[assignment]
        self.depthwise_bn = None  # type: ignore[assignment]
        self.identity_conv = None  # type: ignore[assignment]
        self.identity_bn = None  # type: ignore[assignment]
        self.merge_bn = None  # type: ignore[assignment]
        self.deploy = True


def reparameterize_microsign_edge(model: nn.Module) -> nn.Module:
    """Walk a model and fuse all RSDBlocks into their deploy form."""

    for module in model.modules():
        if isinstance(module, ReparamShiftDepthwiseBlock):
            module.switch_to_deploy()
    return model


__all__ = [
    "ChannelShuffle",
    "ReparamShiftDepthwiseBlock",
    "reparameterize_microsign_edge",
]
