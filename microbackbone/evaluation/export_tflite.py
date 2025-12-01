"""Export utilities for Raspberry Pi and ESP32 (TFLite Micro)."""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Callable, Iterable

import torch

from microbackbone.models.utils import create_model

try:
    import yaml
except ImportError:  # pragma: no cover
    yaml = None

try:  # pragma: no cover
    import onnx  # noqa: F401
except Exception:
    onnx = None

try:  # pragma: no cover
    import tensorflow as tf
except Exception:
    tf = None


def load_yaml(path: Path):
    if yaml is None:
        raise ImportError("pyyaml is required for loading config files")
    with path.open("r") as f:
        return yaml.safe_load(f)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export MicroSign models")
    parser.add_argument("--checkpoint", type=str, required=False, help="Path to .pth checkpoint")
    parser.add_argument("--config", type=str, default="microbackbone/config/model.yaml")
    parser.add_argument("--input-size", type=int, default=224)
    parser.add_argument("--output-dir", type=str, default="outputs/export")
    parser.add_argument("--quantize", action="store_true", help="Export int8 quantized TFLite")
    parser.add_argument("--device", type=str, default="cpu")
    return parser.parse_args()


def _representative_dataset(input_size: int, samples: int = 100) -> Iterable[dict[str, torch.Tensor]]:
    for _ in range(samples):
        data = torch.rand(1, 3, input_size, input_size)
        yield {"input": data}


def export(args: argparse.Namespace) -> None:
    cfg = load_yaml(Path(args.config))
    device = torch.device(args.device)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    model = create_model(
        task=cfg["task"],
        num_classes=cfg["num_classes"],
        variant=cfg["variant"],
        input_size=args.input_size,
    )
    if args.checkpoint:
        ckpt = torch.load(args.checkpoint, map_location="cpu")
        state_dict = ckpt.get("model_state_dict", ckpt)
        model.load_state_dict(state_dict, strict=False)
    model.to(device)
    model.eval()

    dummy = torch.randn(1, 3, args.input_size, args.input_size, device=device)

    # PyTorch checkpoint
    torch.save(model.state_dict(), output_dir / "model.pth")

    # TorchScript
    traced = torch.jit.trace(model, dummy)
    torch.jit.save(traced, output_dir / "model.torchscript.pt")

    # ONNX
    if onnx is not None:
        onnx_path = output_dir / "model.onnx"
        torch.onnx.export(
            model,
            dummy,
            onnx_path,
            input_names=["input"],
            output_names=["logits"],
            opset_version=13,
            dynamic_axes={"input": {0: "batch"}, "logits": {0: "batch"}},
        )
        print(f"Saved ONNX model to {onnx_path}")
    else:
        print("ONNX not available; skipping .onnx export")

    # TFLite export
    if tf is not None:
        if onnx is None:
            raise RuntimeError("TensorFlow export requires onnx to be installed for conversion pipeline")
        import onnx_tf.backend as backend  # type: ignore

        onnx_model = onnx.load(str(onnx_path))
        tf_rep = backend.prepare(onnx_model)
        tf_model_path = output_dir / "model_tf"
        tf_rep.export_graph(str(tf_model_path))
        converter = tf.lite.TFLiteConverter.from_saved_model(str(tf_model_path))
        if args.quantize:
            converter.optimizations = [tf.lite.Optimize.DEFAULT]
            converter.representative_dataset = lambda: (
                {"input": x} for x in _representative_dataset(args.input_size, samples=100)
            )
            converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
            converter.inference_input_type = tf.int8
            converter.inference_output_type = tf.int8
        tflite_model = converter.convert()
        tflite_path = output_dir / ("model_int8.tflite" if args.quantize else "model.tflite")
        with open(tflite_path, "wb") as f:
            f.write(tflite_model)
        print(f"Saved TFLite model to {tflite_path}")
    else:
        print("TensorFlow not available; skipping TFLite export")


if __name__ == "__main__":
    export(parse_args())
