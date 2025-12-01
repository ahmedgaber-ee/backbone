"""Train and compare MicroSign-Net against common TorchVision backbones."""
from __future__ import annotations

import argparse
import csv
import tempfile
import time
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import torch
from torch import nn
from torch.optim import AdamW
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from torchvision import models

from microbackbone.data.datamodule import MicroBackboneDataModule
from microbackbone.models.utils import accuracy, create_model

try:
    from thop import profile  # type: ignore
except Exception:  # pragma: no cover
    profile = None

try:
    import yaml
except Exception:  # pragma: no cover
    yaml = None

# Supported TorchVision model registry
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


class BenchmarkResult(Dict[str, str]):
    """Container for a single model benchmark row."""


def load_yaml(path: Path) -> dict:
    if yaml is None:
        raise ImportError("pyyaml is required for loading config files")
    with path.open("r") as handle:
        return yaml.safe_load(handle)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Train TorchVision backbones and compare them against a MicroSign-Net checkpoint "
            "with latency, params, FLOPs, throughput, size, and accuracy reports."
        )
    )
    parser.add_argument(
        "--custom-weights",
        type=str,
        default=None,
        help="Path to custom MicroSign-Net checkpoint (.pth). If omitted the custom model is skipped unless --train-custom is used.",
    )
    parser.add_argument(
        "--custom-config",
        type=str,
        default="microbackbone/config/model.yaml",
        help="Optional YAML config describing the custom model (variant, num_classes)",
    )
    parser.add_argument(
        "--custom-variant",
        type=str,
        default=None,
        help="Override the variant to construct the custom backbone (helps avoid state_dict shape mismatches).",
    )
    parser.add_argument("--custom-num-classes", type=int, default=None, help="Override num_classes for the custom backbone.")
    parser.add_argument(
        "--models",
        type=str,
        default="resnet18,mobilenet_v3_small,shufflenet_v2_x1_0",
        help="Comma-separated list of TorchVision models to benchmark",
    )
    parser.add_argument("--all", action="store_true", help="Benchmark all supported TorchVision models")
    parser.add_argument("--device", type=str, default="cpu", choices=["cpu", "cuda"], help="Device to benchmark on")
    parser.add_argument(
        "--input-size",
        type=int,
        nargs=3,
        default=[3, 224, 224],
        metavar=("C", "H", "W"),
        help="Model input size (channels height width) for benchmarking",
    )
    parser.add_argument("--warmup", type=int, default=10, help="Number of warmup iterations before timing")
    parser.add_argument("--runs", type=int, default=100, help="Number of timed iterations for latency")
    parser.add_argument(
        "--dataset-config",
        type=str,
        default="microbackbone/config/datasets.yaml",
        help="Dataset YAML config for training/eval",
    )
    parser.add_argument(
        "--default-config",
        type=str,
        default="microbackbone/config/defaults.yaml",
        help="Defaults YAML used for seeds and checkpoint root",
    )
    parser.add_argument("--train", action="store_true", help="Train/Fine-tune TorchVision models before benchmarking")
    parser.add_argument("--train-custom", action="store_true", help="Train the MicroSign-Net model instead of loading weights")
    parser.add_argument("--epochs", type=int, default=5, help="Number of epochs when training models")
    parser.add_argument("--batch-size", type=int, default=None, help="Override batch size from dataset config")
    parser.add_argument("--num-workers", type=int, default=None, help="Override worker count from dataset config")
    parser.add_argument("--learning-rate", type=float, default=1e-3, help="Learning rate for training")
    parser.add_argument("--weight-decay", type=float, default=1e-4, help="Weight decay for training")
    parser.add_argument(
        "--train-split",
        type=float,
        default=None,
        help="Override train/val split ratio (e.g., 0.9 keeps 90% for training)",
    )
    parser.add_argument("--pretrained", action="store_true", help="Start TorchVision models from DEFAULT pretrained weights")
    parser.add_argument(
        "--save-dir",
        type=str,
        default="outputs/benchmarks",
        help="Directory to write CSV and checkpoints",
    )
    return parser.parse_args()


def _model_file_size(model: nn.Module, weights_path: Path | None = None) -> float:
    """Return model size on disk in MB, using provided weights if available."""
    if weights_path is not None and weights_path.exists():
        return weights_path.stat().st_size / (1024 * 1024)

    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_path = Path(tmpdir) / "temp_model.pth"
        torch.save(model.state_dict(), tmp_path)
        return tmp_path.stat().st_size / (1024 * 1024)


def _compute_flops(model: nn.Module, dummy: torch.Tensor) -> float | None:
    if profile is None:
        return None
    try:
        flops, _ = profile(model, inputs=(dummy,), verbose=False)
        return float(flops)
    except Exception:
        return None


def _measure_latency(
    model: nn.Module, dummy: torch.Tensor, runs: int, warmup: int, device: torch.device
) -> Tuple[float, float]:
    model.eval()
    with torch.no_grad():
        for _ in range(warmup):
            _ = model(dummy)
        if device.type == "cuda":
            torch.cuda.synchronize()
        start = time.perf_counter()
        for _ in range(runs):
            _ = model(dummy)
        if device.type == "cuda":
            torch.cuda.synchronize()
    elapsed = time.perf_counter() - start
    avg_ms = (elapsed / runs) * 1000
    throughput = runs / elapsed
    return avg_ms, throughput


def benchmark_model(
    name: str,
    model: nn.Module,
    dummy: torch.Tensor,
    runs: int,
    warmup: int,
    weights_path: Path | None,
    device: torch.device,
    top1: Optional[float] = None,
) -> BenchmarkResult:
    params = sum(p.numel() for p in model.parameters())
    flops = _compute_flops(model, dummy)
    latency_ms, throughput = _measure_latency(model, dummy, runs=runs, warmup=warmup, device=device)
    size_mb = _model_file_size(model, weights_path=weights_path)

    params_m = params / 1e6
    flops_m = flops / 1e6 if flops is not None else None

    result: BenchmarkResult = {
        "Model Name": name,
        "Params": f"{params_m:.2f}M",
        "FLOPs": f"{flops_m:.2f}M" if flops_m is not None else "N/A",
        "Latency (ms)": f"{latency_ms:.2f}",
        "Throughput (img/s)": f"{throughput:.2f}",
        "File Size (MB)": f"{size_mb:.2f}",
        "Top-1 Acc (%)": f"{top1:.2f}" if top1 is not None else "N/A",
    }
    return result


def load_custom_model(
    weights_path: Path | None,
    config_path: Path,
    device: torch.device,
    input_size: Iterable[int],
    override_variant: Optional[str] = None,
    override_num_classes: Optional[int] = None,
) -> nn.Module:
    variant = override_variant or "micro"
    num_classes = override_num_classes or 10
    if config_path.exists() and yaml is not None:
        try:
            cfg = load_yaml(config_path)
            variant = override_variant or cfg.get("variant", variant)
            num_classes = override_num_classes or int(cfg.get("num_classes", num_classes))
        except Exception:
            pass

    checkpoint_cfg = None
    state_dict = None
    if weights_path and weights_path.exists():
        checkpoint = torch.load(weights_path, map_location="cpu")
        checkpoint_cfg = checkpoint.get("config")
        state_dict = checkpoint.get("model_state_dict", checkpoint)
        if checkpoint_cfg:
            variant = override_variant or checkpoint_cfg.get("variant", variant)
            num_classes = override_num_classes or int(checkpoint_cfg.get("num_classes", num_classes))

    model = create_model(task="classification", num_classes=num_classes, variant=variant, input_size=input_size[-1])
    if state_dict is not None:
        model.load_state_dict(state_dict, strict=False)
    model.to(device)
    model.eval()
    return model


def load_torchvision_model(name: str, device: torch.device, pretrained: bool = True) -> nn.Module:
    if name not in TORCHVISION_MODELS:
        raise ValueError(f"Unsupported model: {name}")
    model_fn = TORCHVISION_MODELS[name]
    weight_arg = "DEFAULT" if pretrained else None
    model = model_fn(weights=weight_arg)
    model.to(device)
    model.eval()
    return model


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


def _num_classes_from_data(dm: MicroBackboneDataModule) -> int:
    if dm.dataset == "cifar10":
        return 10
    if dm.train_set and hasattr(dm.train_set, "dataset"):
        base = getattr(dm.train_set, "dataset")
        if hasattr(base, "classes"):
            return len(base.classes)
    if dm.train_set and hasattr(dm.train_set, "classes"):
        return len(dm.train_set.classes)
    return 10


def evaluate_accuracy(model: nn.Module, loader: DataLoader, device: torch.device) -> float:
    model.eval()
    correct = 0.0
    total = 0
    with torch.no_grad():
        for imgs, labels in loader:
            imgs, labels = imgs.to(device, non_blocking=True), labels.to(device, non_blocking=True)
            outputs = model(imgs)
            acc1 = accuracy(outputs, labels, topk=(1,))[0]
            correct += acc1.item() * imgs.size(0) / 100
            total += imgs.size(0)
    return 100 * correct / total if total > 0 else 0.0


def train_model(
    model: nn.Module,
    name: str,
    dm: MicroBackboneDataModule,
    device: torch.device,
    epochs: int,
    lr: float,
    weight_decay: float,
    save_dir: Path,
) -> Tuple[nn.Module, float, Path]:
    """Simple train/val loop returning best model and accuracy."""
    train_loader = dm.train_dataloader()
    val_loader = dm.val_dataloader()

    criterion = nn.CrossEntropyLoss()
    optimizer = AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    best_acc = 0.0
    ckpt_path = save_dir / f"{name}_best.pth"
    save_dir.mkdir(parents=True, exist_ok=True)

    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        running_correct = 0.0
        total = 0
        progress = tqdm(train_loader, desc=f"{name} epoch {epoch+1}/{epochs} [train]", leave=False)
        for imgs, labels in progress:
            imgs, labels = imgs.to(device, non_blocking=True), labels.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            outputs = model(imgs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            acc1 = accuracy(outputs, labels, topk=(1,))[0]
            running_loss += loss.item() * imgs.size(0)
            running_correct += acc1.item() * imgs.size(0) / 100
            total += imgs.size(0)
            progress.set_postfix(loss=loss.item(), acc1=acc1.item())

        val_acc = evaluate_accuracy(model, val_loader, device)
        train_loss = running_loss / total if total > 0 else 0.0
        train_acc = 100 * running_correct / total if total > 0 else 0.0
        tqdm.write(
            f"{name} epoch {epoch+1}: train_loss={train_loss:.4f} train_acc={train_acc:.2f} val_acc={val_acc:.2f}"
        )

        if val_acc > best_acc:
            best_acc = val_acc
            torch.save({"model_state_dict": model.state_dict(), "val_acc": val_acc}, ckpt_path)

    # load best weights for benchmarking
    if ckpt_path.exists():
        state_dict = torch.load(ckpt_path, map_location="cpu")["model_state_dict"]
        model.load_state_dict(state_dict, strict=False)
    model.eval()
    return model, best_acc, ckpt_path


def format_markdown_table(results: List[BenchmarkResult]) -> str:
    headers = list(results[0].keys())
    header_row = "| " + " | ".join(headers) + " |"
    separator = "|" + "---|" * len(headers)
    data_rows = ["| " + " | ".join(row[h] for h in headers) + " |" for row in results]
    return "\n".join([header_row, separator] + data_rows)


def print_text_table(results: List[BenchmarkResult]) -> None:
    headers = list(results[0].keys())
    col_widths = [max(len(row[h]) for row in results + [dict(zip(headers, headers))]) for h in headers]
    header_line = " | ".join(h.ljust(w) for h, w in zip(headers, col_widths))
    separator = "-+-".join("-" * w for w in col_widths)
    print(header_line)
    print(separator)
    for row in results:
        print(" | ".join(row[h].ljust(w) for h, w in zip(headers, col_widths)))


def save_csv(results: List[BenchmarkResult], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    headers = list(results[0].keys())
    with output_path.open("w", newline="") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=headers)
        writer.writeheader()
        for row in results:
            writer.writerow(row)


def main() -> None:
    args = parse_args()
    device = torch.device(args.device)
    input_shape = tuple(args.input_size)
    dummy = torch.randn(1, *input_shape, device=device)
    save_root = Path(args.save_dir)

    # Dataset configuration
    data_cfg = load_yaml(Path(args.dataset_config)) if yaml is not None else {}
    batch_size = args.batch_size or data_cfg.get("batch_size", 128)
    num_workers = args.num_workers or data_cfg.get("num_workers", 4)
    train_split = args.train_split or data_cfg.get("train_split", 0.9)
    input_size = data_cfg.get("input_size", input_shape[-1])
    data_root = data_cfg.get("data_root", "data")

    dm = MicroBackboneDataModule(
        dataset=data_cfg.get("dataset", "cifar10"),
        data_root=data_root,
        batch_size=batch_size,
        num_workers=num_workers,
        input_size=input_size,
        train_split=train_split,
    )
    dm.setup()
    num_classes = args.custom_num_classes or _num_classes_from_data(dm)

    selected_models = list(TORCHVISION_MODELS.keys()) if args.all else [m.strip() for m in args.models.split(",") if m.strip()]

    results: List[BenchmarkResult] = []

    # Custom model (load or train)
    custom_weights = Path(args.custom_weights) if args.custom_weights else None
    custom_model: Optional[nn.Module] = None
    custom_acc: Optional[float] = None
    if args.train_custom or custom_weights is not None:
        custom_model = load_custom_model(
            custom_weights,
            Path(args.custom_config),
            device=device,
            input_size=input_shape,
            override_variant=args.custom_variant,
            override_num_classes=num_classes,
        )
        if args.train_custom:
            custom_model, custom_acc, custom_weights = train_model(
                custom_model,
                name="microbackbone",
                dm=dm,
                device=device,
                epochs=args.epochs,
                lr=args.learning_rate,
                weight_decay=args.weight_decay,
                save_dir=save_root / "checkpoints",
            )
        if custom_acc is None and custom_model is not None:
            custom_acc = evaluate_accuracy(custom_model, dm.val_dataloader(), device)
        if custom_model is not None:
            results.append(
                benchmark_model(
                    name="microbackbone",
                    model=custom_model,
                    dummy=dummy,
                    runs=args.runs,
                    warmup=args.warmup,
                    weights_path=custom_weights,
                    device=device,
                    top1=custom_acc,
                )
            )

    # TorchVision models
    for model_name in selected_models:
        tv_model = load_torchvision_model(
            model_name,
            device=device if args.pretrained else torch.device("cpu"),
            pretrained=args.pretrained,
        )
        tv_model = replace_classifier(tv_model, model_name, num_classes=num_classes)
        tv_model.to(device)

        top1 = None
        ckpt_path = None
        if args.train:
            tv_model, top1, ckpt_path = train_model(
                tv_model,
                name=model_name,
                dm=dm,
                device=device,
                epochs=args.epochs,
                lr=args.learning_rate,
                weight_decay=args.weight_decay,
                save_dir=save_root / "checkpoints",
            )
        if top1 is None:
            top1 = evaluate_accuracy(tv_model, dm.val_dataloader(), device)

        results.append(
            benchmark_model(
                name=model_name,
                model=tv_model,
                dummy=dummy,
                runs=args.runs,
                warmup=args.warmup,
                weights_path=ckpt_path,
                device=device,
                top1=top1,
            )
        )

    print("\nModel Comparison (text table):")
    print_text_table(results)

    print("\nModel Comparison (markdown):")
    md_table = format_markdown_table(results)
    print(md_table)

    csv_path = save_root / "compare_models.csv"
    save_csv(results, csv_path)
    print(f"\nSaved CSV to {csv_path.resolve()}")


if __name__ == "__main__":
    main()
