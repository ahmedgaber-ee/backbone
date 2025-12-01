"""Pruning and quantization utilities with CLI-friendly APIs."""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Iterable, List, Optional, Tuple

import torch
import torch.nn.utils.prune as prune
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from microbackbone.models.utils import create_model, create_torchvision_model

LOGGER = logging.getLogger(__name__)


def load_model_for_tools(
    checkpoint_path: Path,
    device: torch.device,
    arch: Optional[str] = None,
    num_classes: Optional[int] = None,
    variant: str = "micro",
) -> Tuple[nn.Module, dict]:
    """Load a model from a checkpoint or raw state dict.

    Attempts to reconstruct the model architecture using metadata saved in the
    training checkpoint. Falls back to provided `arch`/`num_classes` hints.
    """

    checkpoint = torch.load(checkpoint_path, map_location=device)
    metadata = checkpoint if isinstance(checkpoint, dict) else {}

    ckpt_state = metadata.get("model_state_dict", metadata)
    arch_name = (metadata.get("arch") or arch or "microbackbone").lower()
    variant = metadata.get("variant", variant)
    cfg = metadata.get("config", {})
    classes = metadata.get("data_config", {}).get("num_classes") or cfg.get("num_classes")
    classes = classes or num_classes or 10

    if arch_name == "microbackbone":
        model = create_model(
            task="classification",
            num_classes=classes,
            variant=variant,
            input_size=cfg.get("input_size", 224),
        )
    else:
        model = create_torchvision_model(
            name=arch_name,
            num_classes=classes,
            pretrained=False,
            device=device,
        )

    model.load_state_dict(ckpt_state, strict=False)
    model.to(device)
    model.eval()
    return model, metadata


def _collect_prunable_parameters(model: nn.Module) -> List[Tuple[nn.Module, str]]:
    params: List[Tuple[nn.Module, str]] = []
    for module in model.modules():
        if isinstance(module, (nn.Conv2d, nn.Linear)):
            params.append((module, "weight"))
    return params


def _apply_unstructured_pruning(model: nn.Module, amount: float) -> None:
    parameters_to_prune = _collect_prunable_parameters(model)
    prune.global_unstructured(parameters_to_prune, pruning_method=prune.L1Unstructured, amount=amount)
    for module, _ in parameters_to_prune:
        prune.remove(module, "weight")


def _apply_structured_pruning(model: nn.Module, amount: float, n: int = 2, dim: int = 0) -> None:
    for module in model.modules():
        if isinstance(module, nn.Conv2d):
            prune.ln_structured(module, name="weight", amount=amount, n=n, dim=dim)
            prune.remove(module, "weight")


def apply_pruning(model: nn.Module, pruning_type: str, amount: float) -> nn.Module:
    """Apply structured or unstructured pruning to a model inplace and return it."""

    pruning_type = pruning_type.lower()
    if pruning_type not in {"structured", "unstructured"}:
        raise ValueError("pruning_type must be 'structured' or 'unstructured'")
    if not 0.0 < amount < 1.0:
        raise ValueError("pruning ratio must be between 0 and 1")

    LOGGER.info("Applying %s pruning with ratio %.2f", pruning_type, amount)
    if pruning_type == "unstructured":
        _apply_unstructured_pruning(model, amount)
    else:
        _apply_structured_pruning(model, amount)
    return model


def _get_calibration_loader(data_path: Path, batch_size: int, input_size: int) -> DataLoader:
    transform = transforms.Compose(
        [
            transforms.Resize((input_size, input_size)),
            transforms.ToTensor(),
        ]
    )
    dataset = datasets.ImageFolder(root=str(data_path), transform=transform)
    return DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=2)


def _calibrate_static(model: nn.Module, loader: DataLoader, device: torch.device) -> None:
    model.eval()
    with torch.inference_mode():
        for images, _ in loader:
            images = images.to(device)
            model(images)


def apply_quantization(
    model: nn.Module,
    quantization_type: str,
    device: torch.device,
    calibration_data: Optional[Path] = None,
    input_size: int = 224,
    batch_size: int = 32,
) -> nn.Module:
    """Apply dynamic or static quantization."""

    quantization_type = quantization_type.lower()
    if quantization_type not in {"dynamic", "static"}:
        raise ValueError("quantization_type must be 'dynamic' or 'static'")

    LOGGER.info("Applying %s quantization", quantization_type)
    if quantization_type == "dynamic":
        quantized = torch.quantization.quantize_dynamic(model, {nn.Linear}, dtype=torch.qint8)
        return quantized

    # Static quantization
    model.eval()
    model.fuse_model = getattr(model, "fuse_model", None)
    if callable(model.fuse_model):
        LOGGER.info("Fusing modules prior to static quantization")
        model.fuse_model()

    model.qconfig = torch.quantization.get_default_qconfig("fbgemm")
    torch.quantization.prepare(model, inplace=True)

    if calibration_data is None:
        raise ValueError("Static quantization requires --calibration-data")
    if not calibration_data.exists():
        raise ValueError(f"Calibration path does not exist: {calibration_data}")

    loader = _get_calibration_loader(calibration_data, batch_size=batch_size, input_size=input_size)
    _calibrate_static(model, loader, device=device)
    quantized = torch.quantization.convert(model, inplace=False)
    return quantized


def save_model_checkpoint(model: nn.Module, output_path: Path, metadata: Optional[dict] = None) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {"model_state_dict": model.state_dict()}
    if metadata:
        payload.update(metadata)
    torch.save(payload, output_path)
    LOGGER.info("Saved model to %s", output_path)


__all__ = [
    "apply_pruning",
    "apply_quantization",
    "load_model_for_tools",
    "save_model_checkpoint",
]
