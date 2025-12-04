"""Fast, deployable MicroSign-Edge variant with export helpers."""
from __future__ import annotations

import time
from dataclasses import dataclass
from typing import List, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


def _make_divisible(value: int, divisor: int = 8) -> int:
    return max(divisor, int(value + divisor / 2) // divisor * divisor)


def _fuse_conv_bn_pair(conv: nn.Conv2d, bn: nn.BatchNorm2d) -> nn.Conv2d:
    """Return a Conv2d with fused BatchNorm parameters (bias is always present)."""

    fused = nn.Conv2d(
        conv.in_channels,
        conv.out_channels,
        kernel_size=conv.kernel_size,
        stride=conv.stride,
        padding=conv.padding,
        dilation=conv.dilation,
        groups=conv.groups,
        bias=True,
    )

    with torch.no_grad():
        if conv.bias is None:
            conv_bias = torch.zeros(conv.weight.size(0), device=conv.weight.device)
        else:
            conv_bias = conv.bias

        bn_var_rsqrt = torch.rsqrt(bn.running_var + bn.eps)
        scale = bn.weight * bn_var_rsqrt
        fused.weight.copy_(conv.weight * scale.reshape(-1, 1, 1, 1))
        fused.bias.copy_(bn.bias + (conv_bias - bn.running_mean) * scale)
    return fused


class ConvBNAct(nn.Module):
    """Conv-BN-Activation with optional depthwise grouping."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int = 1,
        groups: int = 1,
        activation: str | None = "relu",
    ) -> None:
        super().__init__()
        padding = kernel_size // 2
        self.conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size,
            stride=stride,
            padding=padding,
            groups=groups,
            bias=False,
        )
        self.bn = nn.BatchNorm2d(out_channels)
        if activation == "relu":
            self.act: nn.Module | None = nn.ReLU(inplace=True)
        elif activation == "silu":
            self.act = nn.SiLU(inplace=True)
        else:
            self.act = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # noqa: D401
        x = self.bn(self.conv(x))
        if self.act is not None:
            x = self.act(x)
        return x

    def fuse(self) -> nn.Module:
        fused_conv = _fuse_conv_bn_pair(self.conv, self.bn)
        if self.act is None:
            return fused_conv
        if isinstance(self.act, nn.ReLU):
            activation = nn.ReLU(inplace=True)
        elif isinstance(self.act, nn.SiLU):
            activation = nn.SiLU(inplace=True)
        else:
            activation = self.act
        return nn.Sequential(fused_conv, activation)


class GhostModule(nn.Module):
    """Ghost-style feature generator for cheap width expansion."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        ratio: int = 2,
        kernel_size: int = 1,
        stride: int = 1,
        activation: str = "relu",
    ) -> None:
        super().__init__()
        init_channels = int(out_channels / ratio)
        new_channels = out_channels - init_channels
        self.primary: nn.Module = ConvBNAct(
            in_channels,
            init_channels,
            kernel_size,
            stride=stride,
            groups=1,
            activation=activation,
        )
        self.cheap: nn.Module = ConvBNAct(
            init_channels,
            new_channels,
            kernel_size=3,
            stride=1,
            groups=init_channels,
            activation=activation,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # noqa: D401
        x_primary = self.primary(x)
        return torch.cat([x_primary, self.cheap(x_primary)], dim=1)

    def fuse(self) -> None:
        self.primary = self.primary.fuse()  # type: ignore[assignment]
        self.cheap = self.cheap.fuse()  # type: ignore[assignment]


class SEBlock(nn.Module):
    """Lightweight squeeze-and-excitation."""

    def __init__(self, channels: int, reduction: int = 8) -> None:
        super().__init__()
        squeezed = max(channels // reduction, 4)
        self.reduce = nn.Conv2d(channels, squeezed, 1)
        self.expand = nn.Conv2d(squeezed, channels, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # noqa: D401
        se = F.adaptive_avg_pool2d(x, 1)
        se = F.relu(self.reduce(se))
        se = torch.sigmoid(self.expand(se))
        return x * se


class EdgeBlock(nn.Module):
    """Fusion-friendly inverted bottleneck with Ghost expansion and SE."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        stride: int,
        expansion: int,
        use_se: bool = True,
        activation: str = "relu",
    ) -> None:
        super().__init__()
        mid_channels = _make_divisible(in_channels * expansion)
        self.use_residual = stride == 1 and in_channels == out_channels

        self.expand = GhostModule(
            in_channels,
            mid_channels,
            ratio=2,
            kernel_size=1,
            stride=1,
            activation=activation,
        )
        self.depthwise = ConvBNAct(
            mid_channels,
            mid_channels,
            kernel_size=3,
            stride=stride,
            groups=mid_channels,
            activation=activation,
        )
        self.se = SEBlock(mid_channels) if use_se else None
        self.project = ConvBNAct(
            mid_channels,
            out_channels,
            kernel_size=1,
            stride=1,
            groups=1,
            activation=None,
        )
        self.act = nn.ReLU(inplace=True) if activation == "relu" else nn.SiLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # noqa: D401
        identity = x
        x = self.expand(x)
        x = self.depthwise(x)
        if self.se is not None:
            x = self.se(x)
        x = self.project(x)
        if self.use_residual:
            x = x + identity
        return self.act(x)

    def fuse(self) -> None:
        """Fuse Conv+BN pairs to reduce inference overhead."""

        self.expand.fuse()
        self.depthwise = self.depthwise.fuse()  # type: ignore[assignment]
        self.project = self.project.fuse()  # type: ignore[assignment]
        # BatchNorms removed; activations kept intact


@dataclass
class EdgeNetConfig:
    channels: Sequence[int]
    expansions: Sequence[int]
    strides: Sequence[int]
    se: Sequence[bool]
    num_classes: int = 10


DEFAULT_CONFIG = EdgeNetConfig(
    channels=(16, 24, 32, 48, 64),
    expansions=(2, 3, 3, 4, 4),
    strides=(1, 2, 2, 2, 1),
    se=(False, True, True, True, True),
    num_classes=10,
)


class MicroSignEdgeOptimized(nn.Module):
    """Optimized backbone/head with deployment-friendly blocks."""

    def __init__(
        self,
        config: EdgeNetConfig = DEFAULT_CONFIG,
        input_channels: int = 3,
        activation: str = "relu",
    ) -> None:
        super().__init__()
        assert len(config.channels) == len(config.expansions) == len(config.strides) == len(config.se)

        stem_channels = config.channels[0]
        stages: List[nn.Module] = [
            ConvBNAct(input_channels, stem_channels, kernel_size=3, stride=1, activation=activation)
        ]

        in_ch = stem_channels
        for out_ch, stride, expansion, use_se in zip(
            config.channels[1:], config.strides, config.expansions, config.se
        ):
            stages.append(
                EdgeBlock(
                    in_channels=in_ch,
                    out_channels=out_ch,
                    stride=stride,
                    expansion=expansion,
                    use_se=use_se,
                    activation=activation,
                )
            )
            in_ch = out_ch

        self.features = nn.Sequential(*stages)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Linear(in_ch, config.num_classes)

    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        return self.features(x)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # noqa: D401
        x = self.forward_features(x)
        x = self.pool(x).flatten(1)
        return self.classifier(x)

    def fuse_model(self) -> None:
        """Fuse all eligible Conv+BN pairs inside blocks for inference."""

        fused_layers: List[nn.Module] = []
        for module in self.features:
            if isinstance(module, EdgeBlock):
                module.fuse()
            if isinstance(module, ConvBNAct):
                module = module.fuse()
            fused_layers.append(module)
        self.features = nn.Sequential(*fused_layers)

    def to_torchscript(self, example_input: torch.Tensor) -> torch.jit.ScriptModule:
        self.eval()
        scripted = torch.jit.trace(self, example_input)
        return torch.jit.freeze(scripted)

    def export_onnx(
        self,
        export_path: str,
        input_shape: Tuple[int, int, int, int] = (1, 3, 32, 32),
        opset_version: int = 13,
    ) -> None:
        self.eval()
        dummy = torch.randn(*input_shape)
        torch.onnx.export(
            self,
            dummy,
            export_path,
            input_names=["images"],
            output_names=["logits"],
            opset_version=opset_version,
            dynamic_axes={"images": {0: "batch"}, "logits": {0: "batch"}},
        )

    def quantized(self) -> nn.Module:
        """Return a dynamically quantized copy (Linear layers) for CPUs."""

        self.eval()
        return torch.ao.quantization.quantize_dynamic(self, {nn.Linear}, dtype=torch.qint8)


def benchmark_latency(
    model: nn.Module,
    device: str = "cpu",
    input_shape: Tuple[int, int, int, int] = (1, 3, 32, 32),
    warmup: int = 10,
    iters: int = 50,
) -> Tuple[float, float]:
    """Return (avg latency ms, throughput imgs/sec)."""

    model = model.to(device).eval()
    data = torch.randn(*input_shape, device=device)

    with torch.inference_mode():
        for _ in range(warmup):
            _ = model(data)
            if device.startswith("cuda"):
                torch.cuda.synchronize()

        start = time.perf_counter()
        for _ in range(iters):
            _ = model(data)
            if device.startswith("cuda"):
                torch.cuda.synchronize()
        elapsed = time.perf_counter() - start

    avg_ms = (elapsed / iters) * 1000.0
    throughput = iters / elapsed
    return avg_ms, throughput


__all__ = [
    "MicroSignEdgeOptimized",
    "EdgeNetConfig",
    "DEFAULT_CONFIG",
    "benchmark_latency",
]
