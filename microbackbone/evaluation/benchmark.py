"""Benchmark FLOPs, params, and inference speed for MicroSign-Net or TorchVision backbones."""
from __future__ import annotations

import argparse
import time
from pathlib import Path

import torch

from microbackbone.models.utils import create_model, create_torchvision_model

try:
    import yaml
except ImportError:  # pragma: no cover
    yaml = None

try:
    from thop import profile  # type: ignore
except Exception:  # pragma: no cover
    profile = None


def load_yaml(path: Path):
    if yaml is None:
        raise ImportError("pyyaml is required for loading config files")
    with path.open("r") as f:
        return yaml.safe_load(f)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Benchmark MicroSign-Net or TorchVision models")
    parser.add_argument("--config", type=str, default="microbackbone/config/model.yaml")
    parser.add_argument("--arch", type=str, default="microbackbone", help="Backbone name (microbackbone or TorchVision model)")
    parser.add_argument("--input-size", type=int, default=224)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--runs", type=int, default=50)
    parser.add_argument("--pretrained", action="store_true", help="Use DEFAULT TorchVision weights when available")
    return parser.parse_args()


def benchmark() -> None:
    args = parse_args()
    cfg = load_yaml(Path(args.config))

    device = torch.device(args.device)
    arch = args.arch.lower()
    if arch == "microbackbone":
        model = create_model(
            task=cfg["task"], num_classes=cfg["num_classes"], variant=cfg["variant"], input_size=args.input_size
        ).to(device)
    else:
        model = create_torchvision_model(
            name=arch, num_classes=cfg["num_classes"], pretrained=args.pretrained, device=device
        )
    model.eval()

    dummy = torch.randn(1, 3, args.input_size, args.input_size, device=device)

    params = sum(p.numel() for p in model.parameters())
    flops = None
    if profile is not None:
        try:
            flops, _ = profile(model, inputs=(dummy,), verbose=False)
        except Exception:
            flops = None

    with torch.no_grad():
        for _ in range(args.warmup):
            _ = model(dummy)
        start = time.perf_counter()
        for _ in range(args.runs):
            _ = model(dummy)
        torch.cuda.synchronize() if device.type == "cuda" else None
        elapsed = time.perf_counter() - start
    avg_ms = (elapsed / args.runs) * 1000
    fps = args.runs / elapsed

    print("Benchmark results")
    print(f"Parameters: {params:,} ({params * 4 / (1024 * 1024):.2f} MB)")
    if flops is not None:
        print(f"FLOPs (approx): {flops/1e6:.2f} MFLOPs")
    else:
        print("FLOPs: thop not installed, skipped")
    print(f"Latency: {avg_ms:.2f} ms | Throughput: {fps:.2f} FPS")


if __name__ == "__main__":
    benchmark()
