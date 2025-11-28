"""Compare MicroSign-Net against common TorchVision backbones."""
from __future__ import annotations

import argparse
import csv
import tempfile
import time
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import torch
from torch import nn
from torchvision import models

from microbackbone.models.utils import create_model

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
    parser = argparse.ArgumentParser(description="Compare MicroSign-Net against TorchVision backbones")
    parser.add_argument(
        "--custom-weights",
        type=str,
        required=True,
        help="Path to custom MicroSign-Net checkpoint (.pth)",
    )
    parser.add_argument(
        "--custom-config",
        type=str,
        default="microbackbone/config/model.yaml",
        help="Optional YAML config describing the custom model (variant, num_classes)",
    )
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
        help="Model input size (channels height width)",
    )
    parser.add_argument("--warmup", type=int, default=10, help="Number of warmup iterations before timing")
    parser.add_argument("--runs", type=int, default=100, help="Number of timed iterations for latency")
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
    }
    return result


def load_custom_model(weights_path: Path, config_path: Path, device: torch.device, input_size: Iterable[int]) -> nn.Module:
    variant = "micro"
    num_classes = 10
    if config_path.exists() and yaml is not None:
        try:
            cfg = load_yaml(config_path)
            variant = cfg.get("variant", variant)
            num_classes = int(cfg.get("num_classes", num_classes))
        except Exception:
            pass

    model = create_model(task="classification", num_classes=num_classes, variant=variant, input_size=input_size[-1])
    checkpoint = torch.load(weights_path, map_location="cpu")
    state_dict = checkpoint.get("model_state_dict", checkpoint)
    model.load_state_dict(state_dict, strict=False)
    model.to(device)
    model.eval()
    return model


def load_torchvision_model(name: str, device: torch.device) -> nn.Module:
    if name not in TORCHVISION_MODELS:
        raise ValueError(f"Unsupported model: {name}")
    model_fn = TORCHVISION_MODELS[name]
    model = model_fn(weights="DEFAULT")
    model.to(device)
    model.eval()
    return model


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

    selected_models = list(TORCHVISION_MODELS.keys()) if args.all else [m.strip() for m in args.models.split(",") if m.strip()]

    results: List[BenchmarkResult] = []

    # Custom model
    custom_path = Path(args.custom_weights)
    custom_model = load_custom_model(
        custom_path, Path(args.custom_config), device=device, input_size=input_shape
    )
    results.append(
        benchmark_model(
            name="microbackbone",
            model=custom_model,
            dummy=dummy,
            runs=args.runs,
            warmup=args.warmup,
            weights_path=custom_path,
            device=device,
        )
    )

    # TorchVision models
    for model_name in selected_models:
        tv_model = load_torchvision_model(model_name, device=device)
        results.append(
            benchmark_model(
                name=model_name,
                model=tv_model,
                dummy=dummy,
                runs=args.runs,
                warmup=args.warmup,
                weights_path=None,
                device=device,
            )
        )

    print("\nModel Comparison (text table):")
    print_text_table(results)

    print("\nModel Comparison (markdown):")
    md_table = format_markdown_table(results)
    print(md_table)

    csv_path = Path("outputs/benchmarks/compare_models.csv")
    save_csv(results, csv_path)
    print(f"\nSaved CSV to {csv_path.resolve()}")


if __name__ == "__main__":
    main()
