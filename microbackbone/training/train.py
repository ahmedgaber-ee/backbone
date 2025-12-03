"""Training script for MicroSign-Net."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict

import torch
import torch.optim as optim
from torch.cuda.amp import GradScaler, autocast
from tqdm.auto import tqdm

from microbackbone.data.datamodule import MicroBackboneDataModule
from microbackbone.models.utils import accuracy, create_model, create_torchvision_model
from microbackbone.training.loss_functions import classification_loss

try:
    import yaml
except ImportError:  # pragma: no cover
    yaml = None


def load_yaml(path: Path) -> Dict:
    if yaml is None:
        raise ImportError("pyyaml is required for loading config files")
    with path.open("r") as f:
        return yaml.safe_load(f)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train MicroSign-Edge or TorchVision models")
    parser.add_argument("--config", type=str, default="microbackbone/config/model.yaml")
    parser.add_argument("--dataset-config", type=str, default="microbackbone/config/datasets.yaml")
    parser.add_argument("--default-config", type=str, default="microbackbone/config/defaults.yaml")
    parser.add_argument("--output-dir", type=str, default="outputs")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument(
        "--arch",
        type=str,
        default="microsign_edge",
        help="Model architecture (microsign_edge or TorchVision model name, e.g., resnet18)",
    )
    parser.add_argument("--pretrained", action="store_true", help="Use pretrained TorchVision weights when available")
    return parser.parse_args()


def train() -> None:
    args = parse_args()
    cfg = load_yaml(Path(args.config))
    data_cfg = load_yaml(Path(args.dataset_config))
    default_cfg = load_yaml(Path(args.default_config))

    torch.manual_seed(default_cfg.get("seed", 42))
    device = torch.device(args.device)

    data = MicroBackboneDataModule(
        dataset=data_cfg["dataset"],
        data_root=data_cfg["data_root"],
        batch_size=data_cfg["batch_size"],
        num_workers=data_cfg["num_workers"],
        input_size=data_cfg["input_size"],
        train_split=data_cfg.get("train_split", 0.9),
        seed=default_cfg.get("seed", 42),
    )
    data.setup()

    arch = args.arch.lower()
    if arch in {"microsign_edge", "microbackbone"}:
        model = create_model(
            task=cfg["task"],
            num_classes=cfg["num_classes"],
            variant=cfg["variant"],
            input_size=data_cfg["input_size"],
        ).to(device)
    else:
        model = create_torchvision_model(
            name=arch,
            num_classes=cfg["num_classes"],
            pretrained=args.pretrained,
            device=device,
        )

    criterion = classification_loss(cfg["num_classes"], cfg.get("label_smoothing", 0.0))
    optimizer = optim.AdamW(
        model.parameters(), lr=cfg["learning_rate"], weight_decay=cfg["weight_decay"]
    )
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=cfg["epochs"], eta_min=1e-6
    )
    scaler = GradScaler(enabled=device.type == "cuda")

    output_dir = Path(args.output_dir)
    ckpt_dir = Path(default_cfg.get("checkpoint_dir", output_dir / "checkpoints"))
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    best_val = 0.0
    history = {"train_loss": [], "train_acc1": [], "val_loss": [], "val_acc1": []}

    for epoch in range(cfg["epochs"]):
        model.train()
        train_loss = 0.0
        correct1 = 0
        total = 0
        train_loader = data.train_dataloader()
        progress = tqdm(
            train_loader,
            desc=f"Epoch {epoch+1}/{cfg['epochs']} [train]",
            leave=False,
        )
        for imgs, labels in progress:
            imgs, labels = imgs.to(device, non_blocking=True), labels.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with autocast(enabled=device.type == "cuda"):
                outputs = model(imgs)
                loss = criterion(outputs, labels)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            acc1 = accuracy(outputs, labels, topk=(1,))[0]
            train_loss += loss.item() * imgs.size(0)
            correct1 += acc1.item() * imgs.size(0) / 100
            total += imgs.size(0)

            progress.set_postfix(loss=loss.item(), acc1=acc1.item())

        scheduler.step()
        history["train_loss"].append(train_loss / total)
        history["train_acc1"].append(100 * correct1 / total)

        model.eval()
        val_loss = 0.0
        val_correct1 = 0
        val_total = 0
        val_loader = data.val_dataloader()
        val_progress = tqdm(
            val_loader,
            desc=f"Epoch {epoch+1}/{cfg['epochs']} [val]",
            leave=False,
        )
        with torch.no_grad():
            for imgs, labels in val_progress:
                imgs, labels = imgs.to(device, non_blocking=True), labels.to(device, non_blocking=True)
                outputs = model(imgs)
                loss = criterion(outputs, labels)
                acc1 = accuracy(outputs, labels, topk=(1,))[0]
                val_loss += loss.item() * imgs.size(0)
                val_correct1 += acc1.item() * imgs.size(0) / 100
                val_total += imgs.size(0)

                val_progress.set_postfix(loss=loss.item(), acc1=acc1.item())

        val_acc = 100 * val_correct1 / val_total
        history["val_loss"].append(val_loss / val_total)
        history["val_acc1"].append(val_acc)

        print(
            f"Epoch {epoch+1}: train_loss={history['train_loss'][-1]:.4f} "
            f"val_loss={history['val_loss'][-1]:.4f} val_acc@1={val_acc:.2f}"
        )

        if val_acc > best_val:
            best_val = val_acc
            checkpoint = {
                "epoch": epoch + 1,
                "arch": arch,
                "variant": cfg.get("variant", "edge_small"),
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_acc": val_acc,
                "config": cfg,
                "data_config": data_cfg,
            }
            ckpt_name = f"{arch}_best.pth" if arch not in {"microsign_edge", "microbackbone"} else f"microsignedge_{cfg['variant']}_best.pth"
            torch.save(checkpoint, ckpt_dir / ckpt_name)

    (output_dir / "history.json").write_text(json.dumps(history, indent=2))


if __name__ == "__main__":
    train()
