"""Command-line interface for pruning and quantization."""
from __future__ import annotations

import argparse
import logging
from pathlib import Path

import torch

from .pruning_quantization import (
    apply_pruning,
    apply_quantization,
    load_model_for_tools,
    save_model_checkpoint,
)


def _setup_logging(level: str) -> None:
    numeric_level = getattr(logging, level.upper(), logging.INFO)
    logging.basicConfig(level=numeric_level, format="[%(levelname)s] %(message)s")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Model maintenance utilities")
    subparsers = parser.add_subparsers(dest="command", required=True, help="Available tools")

    prune_parser = subparsers.add_parser("prune", help="Apply structured or unstructured pruning")
    prune_parser.add_argument("--input-model", required=True, type=Path, help="Path to input checkpoint (.pth)")
    prune_parser.add_argument("--output-model", required=True, type=Path, help="Where to save the pruned checkpoint")
    prune_parser.add_argument(
        "--pruning-type", choices=["structured", "unstructured"], default="unstructured", help="Pruning strategy"
    )
    prune_parser.add_argument(
        "--pruning-ratio", type=float, default=0.3, help="Fraction of parameters/filters to prune"
    )
    prune_parser.add_argument("--arch", type=str, default=None, help="Optional architecture hint if missing in checkpoint")
    prune_parser.add_argument("--num-classes", type=int, default=None, help="Optional num_classes hint")
    prune_parser.add_argument("--log-level", type=str, default="INFO", help="Logging level")

    quant_parser = subparsers.add_parser("quantize", help="Apply dynamic or static quantization")
    quant_parser.add_argument("--input-model", required=True, type=Path, help="Path to input checkpoint (.pth)")
    quant_parser.add_argument("--output-model", required=True, type=Path, help="Where to save the quantized checkpoint")
    quant_parser.add_argument(
        "--quantization-type", choices=["dynamic", "static"], default="dynamic", help="Quantization approach"
    )
    quant_parser.add_argument("--arch", type=str, default=None, help="Optional architecture hint if missing in checkpoint")
    quant_parser.add_argument("--num-classes", type=int, default=None, help="Optional num_classes hint")
    quant_parser.add_argument("--calibration-data", type=Path, default=None, help="Path to calibration folder for static")
    quant_parser.add_argument("--input-size", type=int, default=224, help="Input resolution for calibration")
    quant_parser.add_argument("--batch-size", type=int, default=32, help="Calibration batch size")
    quant_parser.add_argument("--log-level", type=str, default="INFO", help="Logging level")

    return parser.parse_args()


def _load_model(args: argparse.Namespace, device: torch.device):
    model, metadata = load_model_for_tools(
        checkpoint_path=args.input_model,
        device=device,
        arch=getattr(args, "arch", None),
        num_classes=getattr(args, "num_classes", None),
    )
    return model, metadata


def main() -> None:
    args = _parse_args()
    _setup_logging(args.log_level)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model, metadata = _load_model(args, device)

    if args.command == "prune":
        pruned = apply_pruning(model, pruning_type=args.pruning_type, amount=args.pruning_ratio)
        save_model_checkpoint(pruned, args.output_model, metadata)
    elif args.command == "quantize":
        quantized = apply_quantization(
            model,
            quantization_type=args.quantization_type,
            device=device,
            calibration_data=args.calibration_data,
            input_size=args.input_size,
            batch_size=args.batch_size,
        )
        save_model_checkpoint(quantized, args.output_model, metadata)
    else:  # pragma: no cover
        raise ValueError(f"Unknown command {args.command}")


if __name__ == "__main__":
    main()
