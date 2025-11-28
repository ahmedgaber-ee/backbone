"""Model utilities and creation helpers."""
from __future__ import annotations

from typing import Dict, Iterable, Tuple

import torch

from .backbone import MicroSignBackbone, MicroSignDetector


def create_model(
    task: str = "classification",
    num_classes: int = 10,
    variant: str = "micro",
    input_size: int = 224,
    pretrained_path: str | None = None,
    return_stages: Iterable[int] | None = None,
) -> MicroSignBackbone | MicroSignDetector:
    """Unified factory for classification, detection, or pure backbones."""
    if task == "classification":
        model = MicroSignBackbone(
            variant=variant, num_classes=num_classes, input_size=input_size, return_stages=None
        )
    elif task == "detection":
        model = MicroSignDetector(num_classes=num_classes, variant=variant, input_size=input_size)
    elif task == "backbone":
        model = MicroSignBackbone(
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


__all__ = ["create_model", "accuracy"]
