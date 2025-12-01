"""Compare trained checkpoints against common TorchVision backbones."""
from __future__ import annotations

import argparse
import csv
import tempfile
import time
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import torch
from torch import nn
from torch.utils.data import DataLoader

from microbackbone.data.datamodule import MicroBackboneDataModule
from microbackbone.models.utils import (
    TORCHVISION_MODELS,
    accuracy,
    create_model,
    create_torchvision_model,
)

try:
    from thop import profile  # type: ignore
except Exception:  # pragma: no cover
    profile = None

try:
    import yaml
except Exception:  # pragma: no cover
    yaml = None

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
            "Compare trained checkpoints against TorchVision backbones (pretrained or random init) "
            "with accuracy, params, FLOPs, latency, throughput, and file size metrics."
        )
    )
    parser.add_argument(
        "--checkpoints",
        type=str,
        nargs="*",
        default=[],
        help="Paths to trained checkpoints (.pth) produced by microbackbone.training.train",
    )
    parser.add_argument(
        "--models",
        type=str,
        default="resnet18,mobilenet_v3_small",
        help="Comma-separated list of TorchVision models to benchmark",
    )
    parser.add_argument("--all", action="store_true", help="Benchmark all supported TorchVision models")
    parser.add_argument("--pretrained", action="store_true", help="Use DEFAULT TorchVision weights when available")
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
        help="Dataset YAML config for evaluation",
    )
    parser.add_argument(
        "--model-config",
        type=str,
        default="microbackbone/config/model.yaml",
        help="Fallback model config for MicroSign-Net checkpoints",
    )
    parser.add_argument("--batch-size", type=int, default=None, help="Override batch size from dataset config")
    parser.add_argument("--num-workers", type=int, default=None, help="Override worker count from dataset config")
    parser.add_argument(
        "--train-split",
        type=float,
        default=None,
        help="Override train/val split ratio (e.g., 0.9 keeps 90% for training)",
    )
    parser.add_argument(
        "--save-dir",
        type=str,
        default="outputs/benchmarks",
        help="Directory to write CSV and reports",
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


def load_checkpoint_model(
    weights_path: Path,
    device: torch.device,
    default_num_classes: int,
    input_size: Iterable[int],
    fallback_cfg: dict,
) -> tuple[nn.Module, str, int, Optional[float]]:
    """Load a trained checkpoint and rebuild the appropriate architecture."""
    checkpoint = torch.load(weights_path, map_location="cpu")
    cfg = checkpoint.get("config", fallback_cfg)
    arch = checkpoint.get("arch", cfg.get("arch", "microsign_edge"))
    variant = checkpoint.get("variant", cfg.get("variant", "edge_small"))
    num_classes = int(cfg.get("num_classes", default_num_classes))
    state_dict = checkpoint.get("model_state_dict", checkpoint)

    if arch in TORCHVISION_MODELS:
        model = create_torchvision_model(
            name=arch, num_classes=num_classes, pretrained=False, device=device
        )
    else:
        task = cfg.get("task", "classification")
        model = create_model(
            task=task,
            num_classes=num_classes,
            variant=variant,
            input_size=input_size[-1],
        ).to(device)

    model.load_state_dict(state_dict, strict=False)
    model.eval()
    recorded_acc = checkpoint.get("val_acc")
    return model, arch, num_classes, recorded_acc


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
    save_root.mkdir(parents=True, exist_ok=True)

    data_cfg = load_yaml(Path(args.dataset_config)) if yaml is not None else {}
    model_cfg = load_yaml(Path(args.model_config)) if yaml is not None else {}

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
    num_classes = _num_classes_from_data(dm)

    results: List[BenchmarkResult] = []

    # Load and benchmark checkpoints
    for ckpt in args.checkpoints:
        ckpt_path = Path(ckpt)
        if not ckpt_path.exists():
            print(f"[WARN] checkpoint not found: {ckpt_path}")
            continue
        model, arch, _, recorded_acc = load_checkpoint_model(
            ckpt_path,
            device=device,
            default_num_classes=num_classes,
            input_size=input_shape,
            fallback_cfg=model_cfg,
        )
        acc = recorded_acc if recorded_acc is not None else evaluate_accuracy(model, dm.val_dataloader(), device)
        results.append(
            benchmark_model(
                name=f"{ckpt_path.stem} ({arch})",
                model=model,
                dummy=dummy,
                runs=args.runs,
                warmup=args.warmup,
                weights_path=ckpt_path,
                device=device,
                top1=acc,
            )
        )

    # Benchmark TorchVision backbones
    selected_models = list(TORCHVISION_MODELS.keys()) if args.all else [m.strip() for m in args.models.split(",") if m.strip()]
    for model_name in selected_models:
        tv_model = create_torchvision_model(
            name=model_name,
            num_classes=num_classes,
            pretrained=args.pretrained,
            device=device,
        )
        acc = evaluate_accuracy(tv_model, dm.val_dataloader(), device)
        readable_name = f"{model_name} ({'pretrained' if args.pretrained else 'scratch'})"
        results.append(
            benchmark_model(
                name=readable_name,
                model=tv_model,
                dummy=dummy,
                runs=args.runs,
                warmup=args.warmup,
                weights_path=None,
                device=device,
                top1=acc,
            )
        )

    if not results:
        print("No models were benchmarked. Provide checkpoints or models via CLI.")
        return

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
