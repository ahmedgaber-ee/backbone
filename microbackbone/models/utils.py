"""Model utilities and creation helpers."""
from __future__ import annotations

from typing import Dict, Iterable, Tuple

import torch
from torch import nn
from torchvision import models

from .backbone import MicroSignEdgeBackbone, MicroSignEdgeDetector


def create_model(
    task: str = "classification",
    num_classes: int = 10,
    variant: str = "edge_small",
    input_size: int = 224,
    pretrained_path: str | None = None,
    return_stages: Iterable[int] | None = None,
) -> MicroSignEdgeBackbone | MicroSignEdgeDetector:
    """Unified factory for classification, detection, or pure backbones."""
    if task == "classification":
        model = MicroSignEdgeBackbone(
            variant=variant, num_classes=num_classes, input_size=input_size, return_stages=None
        )
    elif task == "detection":
        model = MicroSignEdgeDetector(num_classes=num_classes, variant=variant, input_size=input_size)
    elif task == "backbone":
        model = MicroSignEdgeBackbone(
            variant=variant,
            num_classes=None,
            return_stages=return_stages if return_stages is not None else [2, 3, 4],
            input_size=input_size,
        )
    else:
        raise ValueError(f"Unknown task: {task}")

    if pretrained_path:
        checkpoint = torch.load(pretrained_path, map_location="cpu")
        state_dict = checkpoint.get("model_state_dict", checkpoint)
        if task in ["detection", "backbone"]:
            state_dict = {k: v for k, v in state_dict.items() if "classifier" not in k}
        model.load_state_dict(state_dict, strict=False)
    return model


# TorchVision registry for convenient creation
TORCHVISION_MODELS = {
    "resnet18": models.resnet18,
    "resnet34": models.resnet34,
    "resnet50": models.resnet50,
    "mobilenet_v2": models.mobilenet_v2,
    "mobilenet_v3_small": models.mobilenet_v3_small,
    "mobilenet_v3_large": models.mobilenet_v3_large,
    "shufflenet_v2_x0_5": models.shufflenet_v2_x0_5,
    "shufflenet_v2_x1_0": models.shufflenet_v2_x1_0,
    "efficientnet_b0": models.efficientnet_b0,
    "efficientnet_b1": models.efficientnet_b1,
    "convnext_tiny": models.convnext_tiny,
}


def replace_classifier(model: nn.Module, name: str, num_classes: int) -> nn.Module:
    """Adjust the classifier head of a TorchVision model to match num_classes."""
    if name.startswith("resnet"):
        in_features = model.fc.in_features
        model.fc = nn.Linear(in_features, num_classes)
    elif name.startswith("mobilenet_v2"):
        in_features = model.classifier[-1].in_features
        model.classifier[-1] = nn.Linear(in_features, num_classes)
    elif name.startswith("mobilenet_v3"):
        in_features = model.classifier[-1].in_features
        model.classifier[-1] = nn.Linear(in_features, num_classes)
    elif name.startswith("shufflenet_v2"):
        in_features = model.fc.in_features
        model.fc = nn.Linear(in_features, num_classes)
    elif name.startswith("efficientnet_b"):
        in_features = model.classifier[1].in_features
        model.classifier[1] = nn.Linear(in_features, num_classes)
    elif name.startswith("convnext"):
        in_features = model.classifier[2].in_features
        model.classifier[2] = nn.Linear(in_features, num_classes)
    else:  # pragma: no cover
        raise ValueError(f"Unhandled classifier adaptation for {name}")
    return model


def _resolve_weights_argument(model_name: str, pretrained: bool):
    """Return a weights argument compatible with both old/new TorchVision APIs."""
    weights_enum = getattr(models, f"{model_name.upper()}_Weights", None)
    if weights_enum is None:
        return pretrained  # likely older API expecting `pretrained` bool
    return weights_enum.DEFAULT if pretrained else None


def create_torchvision_model(
    name: str,
    num_classes: int,
    pretrained: bool,
    device: torch.device,
) -> nn.Module:
    if name not in TORCHVISION_MODELS:
        raise ValueError(f"Unsupported TorchVision model: {name}")

    model_fn = TORCHVISION_MODELS[name]
    weight_arg = _resolve_weights_argument(name, pretrained)

    try:
        model = model_fn(weights=weight_arg)
    except TypeError:
        # Fallback for TorchVision < 0.13 that expects `pretrained` instead of `weights`
        model = model_fn(pretrained=bool(pretrained))

    model = replace_classifier(model, name, num_classes)
    model.to(device)
    return model


def accuracy(output: torch.Tensor, target: torch.Tensor, topk: Tuple[int, ...] = (1,)) -> list[torch.Tensor]:
    """Computes accuracy over the k top predictions."""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)
        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))
        res = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res


__all__ = [
    "create_model",
    "create_torchvision_model",
    "replace_classifier",
    "accuracy",
    "TORCHVISION_MODELS",
]
