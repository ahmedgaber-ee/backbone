"""MicroSign-Net backbone definitions."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional

import torch
import torch.nn as nn

from .modules import EfficientBlock, SpatialAttention


@dataclass
class BackboneConfig:
    """Lightweight container describing a backbone variant."""

    width_mult: float
    depths: List[int]
    expansions: List[int]
    description: str
    use_dynamic: List[bool]
    use_se: List[bool]


CONFIGS: Dict[str, BackboneConfig] = {
    "nano": BackboneConfig(
        width_mult=0.25,
        depths=[1, 2, 2, 1],
        expansions=[1, 4, 4, 4],
        description="Extreme edge (20-50KB)",
        use_dynamic=[False, False, False, False],
        use_se=[False, True, True, False],
    ),
    "micro": BackboneConfig(
        width_mult=0.5,
        depths=[2, 2, 3, 2],
        expansions=[1, 4, 6, 6],
        description="Ultra-lightweight (100-200KB)",
        use_dynamic=[False, True, True, False],
        use_se=[True, True, True, True],
    ),
    "tiny": BackboneConfig(
        width_mult=0.75,
        depths=[2, 3, 4, 2],
        expansions=[1, 4, 6, 6],
        description="Lightweight (200-400KB)",
        use_dynamic=[False, True, True, True],
        use_se=[True, True, True, True],
    ),
    "small": BackboneConfig(
        width_mult=1.0,
        depths=[2, 3, 5, 3],
        expansions=[1, 4, 6, 6],
        description="Balanced (400-800KB)",
        use_dynamic=[True, True, True, True],
        use_se=[True, True, True, True],
    ),
    "base": BackboneConfig(
        width_mult=1.25,
        depths=[3, 4, 6, 3],
        expansions=[1, 4, 6, 6],
        description="High accuracy (800KB-1.5MB)",
        use_dynamic=[True, True, True, True],
        use_se=[True, True, True, True],
    ),
}


def _make_divisible(value: float, divisor: int = 8) -> int:
    return max(divisor, int(value + divisor / 2) // divisor * divisor)


class MicroSignBackbone(nn.Module):
    """Efficient backbone optimized for small devices and images."""

    def __init__(
        self,
        variant: str = "micro",
        num_classes: Optional[int] = None,
        return_stages: Optional[Iterable[int]] = None,
        dropout: float = 0.2,
        frozen_stages: int = -1,
        input_size: int = 224,
    ) -> None:
        super().__init__()
        if variant not in CONFIGS:
            raise ValueError(f"Variant must be one of {list(CONFIGS.keys())}")

        self.config = CONFIGS[variant]
        width_mult = self.config.width_mult
        depths = self.config.depths
        expansions = self.config.expansions
        use_dynamic = self.config.use_dynamic
        use_se = self.config.use_se

        base_channels = [16, 24, 32, 64, 128] if input_size <= 64 else [24, 32, 64, 128, 256]
        self.out_channels: List[int] = [_make_divisible(c * width_mult) for c in base_channels]

        if input_size <= 64:
            self.stem = nn.Sequential(
                nn.Conv2d(3, self.out_channels[0], 3, stride=1, padding=1, bias=False),
                nn.BatchNorm2d(self.out_channels[0]),
                nn.ReLU6(inplace=True),
            )
        else:
            self.stem = nn.Sequential(
                nn.Conv2d(3, self.out_channels[0], 3, stride=2, padding=1, bias=False),
                nn.BatchNorm2d(self.out_channels[0]),
                nn.ReLU6(inplace=True),
            )

        self.stages = nn.ModuleList()
        in_channels = self.out_channels[0]
        for stage_idx, depth in enumerate(depths):
            out_ch = self.out_channels[min(stage_idx + 1, len(self.out_channels) - 1)]
            expansion = expansions[stage_idx]
            blocks: List[nn.Module] = []
            for block_idx in range(depth):
                if input_size <= 64:
                    stride = 2 if block_idx == 0 and stage_idx in [1, 2] else 1
                else:
                    stride = 2 if block_idx == 0 else 1
                blocks.append(
                    EfficientBlock(
                        in_channels if block_idx == 0 else out_ch,
                        out_ch,
                        stride=stride,
                        expansion=expansion,
                        use_dynamic=use_dynamic[stage_idx],
                        use_se=use_se[stage_idx],
                    )
                )
            if stage_idx in [1, 2]:
                blocks.append(SpatialAttention())
            self.stages.append(nn.Sequential(*blocks))
            in_channels = out_ch

        self.variant = variant
        self.return_stages = set(return_stages) if return_stages is not None else None
        self.frozen_stages = frozen_stages
        self.input_size = input_size
        self.num_classes = num_classes

        if num_classes is not None:
            final_channels = self.out_channels[-1]
            self.global_pool = nn.AdaptiveAvgPool2d(1)
            self.pre_classifier = nn.Sequential(
                nn.Linear(final_channels, final_channels),
                nn.ReLU(inplace=True),
                nn.Dropout(dropout * 0.5),
            )
            self.classifier = nn.Linear(final_channels, num_classes)
            self.dropout = nn.Dropout(dropout)
        else:
            self.global_pool = None
            self.pre_classifier = None
            self.classifier = None
            self.dropout = None

        self._initialize_weights()
        self._freeze_stages()

        self.total_params = sum(p.numel() for p in self.parameters())
        self.trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)

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

    def _freeze_stages(self) -> None:
        if self.frozen_stages >= 0:
            self.stem.eval()
            for param in self.stem.parameters():
                param.requires_grad = False
        for stage_index in range(self.frozen_stages):
            if stage_index < len(self.stages):
                self.stages[stage_index].eval()
                for param in self.stages[stage_index].parameters():
                    param.requires_grad = False

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
        features = self.forward_features(x)
        if isinstance(features, dict):
            x = list(features.values())[-1]
        else:
            x = features
        x = self.global_pool(x)
        x = x.flatten(1)
        x = self.pre_classifier(x)
        x = self.dropout(x) if self.dropout is not None else x
        return self.classifier(x)

    def get_model_info(self) -> Dict[str, object]:
        return {
            "variant": self.variant,
            "description": self.config.description,
            "depths": self.config.depths,
            "expansions": self.config.expansions,
            "output_channels": self.out_channels,
            "return_stages": self.return_stages,
            "num_classes": self.num_classes,
            "input_size": self.input_size,
            "total_parameters": self.total_params,
            "trainable_parameters": self.trainable_params,
            "model_size_mb": self.total_params * 4 / (1024 * 1024),
            "frozen_stages": self.frozen_stages,
        }

    def train(self, mode: bool = True) -> "MicroSignBackbone":  # type: ignore[override]
        super().train(mode)
        self._freeze_stages()
        return self


class DetectionHead(nn.Module):
    """Minimal detection head for toy demos."""

    def __init__(self, in_channels: int, num_anchors: int = 3, num_classes: int = 80) -> None:
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, 3, padding=1, groups=in_channels, bias=False),
            nn.BatchNorm2d(in_channels),
            nn.ReLU6(inplace=True),
            nn.Conv2d(in_channels, num_anchors * (5 + num_classes), 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # noqa: D401
        return self.conv(x)


class MicroSignDetector(nn.Module):
    """Simple multi-scale detector used for benchmarking the backbone."""

    def __init__(self, num_classes: int = 80, variant: str = "micro", input_size: int = 224) -> None:
        super().__init__()
        self.backbone = MicroSignBackbone(variant=variant, return_stages=[2, 3, 4], input_size=input_size)
        channels = self.backbone.out_channels
        self.det_heads = nn.ModuleDict(
            {
                "small": DetectionHead(channels[2], num_classes=num_classes),
                "medium": DetectionHead(channels[3], num_classes=num_classes),
                "large": DetectionHead(channels[4], num_classes=num_classes),
            }
        )

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:  # noqa: D401
        features = self.backbone(x)
        assert isinstance(features, dict)
        return {
            "small": self.det_heads["small"](features["C3"]),
            "medium": self.det_heads["medium"](features["C4"]),
            "large": self.det_heads["large"](features["C5"]),
        }


__all__ = [
    "CONFIGS",
    "BackboneConfig",
    "MicroSignBackbone",
    "MicroSignDetector",
]
