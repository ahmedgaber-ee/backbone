"""Evaluation script for MicroSign-Net and TorchVision backbones."""
from __future__ import annotations

import argparse
from pathlib import Path

import torch

from microbackbone.data.datamodule import MicroBackboneDataModule
from microbackbone.models.utils import accuracy, create_model, create_torchvision_model

try:
    import yaml
except ImportError:  # pragma: no cover
    yaml = None


def load_yaml(path: Path):
    if yaml is None:
        raise ImportError("pyyaml is required for loading config files")
    with path.open("r") as f:
        return yaml.safe_load(f)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate MicroSign-Edge or TorchVision checkpoints")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to checkpoint (.pth)")
    parser.add_argument("--config", type=str, default="microbackbone/config/model.yaml")
    parser.add_argument("--dataset-config", type=str, default="microbackbone/config/datasets.yaml")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument(
        "--arch",
        type=str,
        default=None,
        help=(
            "Optional override for backbone architecture; when omitted, the script reads the arch "
            "stored inside the checkpoint. Use TorchVision model names such as resnet18, mobilenet_v3_small, etc."
        ),
    )
    parser.add_argument(
        "--pretrained",
        action="store_true",
        help="Use DEFAULT TorchVision weights before loading the checkpoint when evaluating TorchVision models",
    )
    return parser.parse_args()


def evaluate() -> None:
    args = parse_args()
    cfg = load_yaml(Path(args.config))
    data_cfg = load_yaml(Path(args.dataset_config))

    device = torch.device(args.device)
    data = MicroBackboneDataModule(
        dataset=data_cfg["dataset"],
        data_root=data_cfg["data_root"],
        batch_size=data_cfg["batch_size"],
        num_workers=data_cfg["num_workers"],
        input_size=data_cfg["input_size"],
        train_split=data_cfg.get("train_split", 0.9),
    )
    data.setup()

    checkpoint = torch.load(args.checkpoint, map_location="cpu")
    arch = (args.arch or checkpoint.get("arch", "microsign_edge")).lower()

    if arch in {"microsign_edge", "microbackbone"}:
        model = create_model(
            task=cfg["task"],
            num_classes=cfg["num_classes"],
            variant=cfg["variant"],
            input_size=data_cfg["input_size"],
        )
    else:
        model = create_torchvision_model(
            name=arch,
            num_classes=cfg["num_classes"],
            pretrained=args.pretrained,
            device=torch.device("cpu"),
        )

    state_dict = checkpoint.get("model_state_dict", checkpoint)
    model.load_state_dict(state_dict, strict=False)
    model.to(device)
    model.eval()

    criterion = torch.nn.CrossEntropyLoss()
    test_loss = 0.0
    correct1 = 0
    correct5 = 0
    total = 0

    with torch.no_grad():
        for imgs, labels in data.test_dataloader():
            imgs, labels = imgs.to(device, non_blocking=True), labels.to(device, non_blocking=True)
            outputs = model(imgs)
            loss = criterion(outputs, labels)
            acc1, acc5 = accuracy(outputs, labels, topk=(1, 5))
            test_loss += loss.item() * imgs.size(0)
            correct1 += acc1.item() * imgs.size(0) / 100
            correct5 += acc5.item() * imgs.size(0) / 100
            total += imgs.size(0)

    print(
        f"Test Loss: {test_loss / total:.4f} | "
        f"Acc@1: {100 * correct1 / total:.2f}% | Acc@5: {100 * correct5 / total:.2f}%"
    )


if __name__ == "__main__":
    evaluate()
