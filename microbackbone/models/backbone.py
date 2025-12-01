"""MicroSign-Edge backbone family for MCUs."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional

import torch
import torch.nn as nn

from .modules import ReparamShiftDepthwiseBlock, reparameterize_microsign_edge


def _make_divisible(value: float, divisor: int = 8) -> int:
    return max(divisor, int(value + divisor / 2) // divisor * divisor)


@dataclass
class EdgeBackboneConfig:
    """Defines a MicroSign-Edge variant."""

    width_mult: float
    depths: List[int]
    expansions: List[int]
    base_channels: List[int]
    description: str


CONFIGS: Dict[str, EdgeBackboneConfig] = {
    "edge_nano": EdgeBackboneConfig(
        width_mult=0.5,
        depths=[1, 2, 2, 1],
        expansions=[1, 2, 2, 2],
        base_channels=[24, 32, 48, 72],
        description="Sub-50KB MCU-friendly variant",
    ),
    "edge_micro": EdgeBackboneConfig(
        width_mult=0.75,
        depths=[1, 2, 3, 2],
        expansions=[1, 2, 2, 3],
        base_channels=[24, 36, 56, 88],
        description="TinyML tuned configuration",
    ),
    "edge_small": EdgeBackboneConfig(
        width_mult=1.0,
        depths=[2, 3, 3, 2],
        expansions=[2, 2, 3, 3],
        base_channels=[24, 32, 48, 72],
        description="Reference 32x32-friendly edge model",
    ),
}


def _compute_edge_channels(
    base_channels: List[int], input_size: int, sram_kb: int = 128, safety_factor: float = 0.5
) -> List[int]:
    """Reduce channels so activations fit a rough SRAM budget (8-bit assumed)."""

    adjusted: List[int] = []
    spatial = input_size // 2  # stem stride=2
    for idx, c in enumerate(base_channels):
        budget = sram_kb * 1024 * safety_factor
        channel_cap = max(8, int(budget // max(1, spatial * spatial)))
        tuned = min(c, channel_cap)
        tuned = _make_divisible(tuned, 8)
        adjusted.append(max(8, tuned))
        if idx < len(base_channels) - 1:
            spatial = max(1, spatial // 2 if idx in [0, 1] else spatial)
    return adjusted


class MicroSignEdgeBackbone(nn.Module):
    """Quantization-friendly MicroSign-Edge backbone with RSDBlocks."""

    def __init__(
        self,
        variant: str = "edge_small",
        num_classes: Optional[int] = None,
        return_stages: Optional[Iterable[int]] = None,
        input_size: int = 32,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        if variant not in CONFIGS:
            raise ValueError(f"Variant must be one of {list(CONFIGS.keys())}")

        self.variant = variant
        self.return_stages = set(return_stages) if return_stages is not None else None
        self.num_classes = num_classes
        self.input_size = input_size

        cfg = CONFIGS[variant]
        scaled = [
            _make_divisible(ch * cfg.width_mult)
            for ch in _compute_edge_channels(cfg.base_channels, input_size)
        ]

        stem_channels = _make_divisible(24 * cfg.width_mult)
        self.stem = nn.Sequential(
            nn.Conv2d(3, stem_channels, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(stem_channels),
            nn.SiLU(inplace=True),
        )

        self.stages = nn.ModuleList()
        in_channels = stem_channels
        for stage_idx, depth in enumerate(cfg.depths):
            out_channels = scaled[stage_idx]
            expansion = cfg.expansions[stage_idx]
            blocks: List[nn.Module] = []
            for block_idx in range(depth):
                stride = 1
                if stage_idx in [1, 2] and block_idx == 0:
                    stride = 2
                blocks.append(
                    ReparamShiftDepthwiseBlock(
                        in_channels if block_idx == 0 else out_channels,
                        out_channels,
                        stride=stride,
                        expansion=expansion,
                    )
                )
            self.stages.append(nn.Sequential(*blocks))
            in_channels = out_channels

        self.global_pool = nn.AdaptiveAvgPool2d(1)
        if num_classes is not None:
            self.pre_classifier = nn.Sequential(
                nn.Linear(in_channels, 128),
                nn.ReLU(inplace=True),
                nn.Dropout(dropout),
            )
            self.classifier = nn.Linear(128, num_classes)
        else:
            self.pre_classifier = None
            self.classifier = None

        self.out_channels = [stem_channels] + scaled
        self._initialize_weights()

    def _initialize_weights(self) -> None:
        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.kaiming_normal_(module.weight, mode="fan_out", nonlinearity="relu")
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.BatchNorm2d):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, 0, 0.01)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def forward_features(self, x: torch.Tensor) -> torch.Tensor | Dict[str, torch.Tensor]:
        outputs: Dict[str, torch.Tensor] = {}
        x = self.stem(x)
        if self.return_stages is not None and 0 in self.return_stages:
            outputs["C1"] = x
        for idx, stage in enumerate(self.stages):
            x = stage(x)
            if self.return_stages is not None and idx + 1 in self.return_stages:
                outputs[f"C{idx + 2}"] = x
        if self.return_stages is None:
            return x
        if len(outputs) == 1:
            return list(outputs.values())[0]
        return outputs

    def forward(self, x: torch.Tensor) -> torch.Tensor | Dict[str, torch.Tensor]:  # noqa: D401
        if self.classifier is None:
            return self.forward_features(x)
        feats = self.forward_features(x)
        if isinstance(feats, dict):
            x = list(feats.values())[-1]
        else:
            x = feats
        x = self.global_pool(x).flatten(1)
        x = self.pre_classifier(x)  # type: ignore[arg-type]
        return self.classifier(x)

    def reparameterize(self) -> None:
        """Fuse RSDBlocks for deployment."""

        reparameterize_microsign_edge(self)


class MicroSignEdgeDetector(nn.Module):
    """Toy detector head using MicroSign-Edge features."""

    def __init__(self, num_classes: int = 80, variant: str = "edge_small", input_size: int = 224) -> None:
        super().__init__()
        self.backbone = MicroSignEdgeBackbone(
            variant=variant, num_classes=None, return_stages=[2, 3, 4], input_size=input_size
        )
        channels = self.backbone.out_channels
        self.det_heads = nn.ModuleDict(
            {
                "small": nn.Conv2d(channels[2], num_classes + 5, 1),
                "medium": nn.Conv2d(channels[3], num_classes + 5, 1),
                "large": nn.Conv2d(channels[4], num_classes + 5, 1),
            }
        )

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:  # noqa: D401
        feats = self.backbone(x)
        assert isinstance(feats, dict)
        return {
            "small": self.det_heads["small"](feats["C3"]),
            "medium": self.det_heads["medium"](feats["C4"]),
            "large": self.det_heads["large"](feats["C5"]),
        }


__all__ = [
    "CONFIGS",
    "EdgeBackboneConfig",
    "_compute_edge_channels",
    "MicroSignEdgeBackbone",
    "MicroSignEdgeDetector",
]
