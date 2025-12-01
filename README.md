# MicroBackbone (MicroSign-Edge)

Lightweight CNN backbone family tailored for microcontrollers and edge devices. Extracted from the original notebook into a modular, deployable Python package with export paths for Raspberry Pi and ESP32 (TFLite Micro).

## Key Features
- **MicroSign-Edge backbone** (`edge_nano`, `edge_micro`, `edge_small`) built from Reparam Shift Depthwise (RSD) blocks.
- Zero-FLOP shift-mix + grouped SE gating for TinyML efficiency and quantization-friendliness.
- Activation-budget-aware channel scaling for SRAM-constrained deployments.
- Reparameterization utility to fuse branches for MCU inference (`--reparam-edge`).
- Training/eval scripts with clean configs and TorchVision drop-in support.
- Export to TorchScript, ONNX, and TFLite (int8 for ESP32).
- Deployment recipes for Raspberry Pi and ESP32.

### MicroSign-Edge at a glance
- **RSD Block**: optional 1×1 expansion → shift-mix (up/left/down/right) → depthwise + identity fusion → grouped SE → 1×1 projection with residuals when safe.
- **Deploy fusion**: `reparameterize_microsign_edge` collapses branches/BatchNorms to a single depthwise + pointwise path for MCU timing.
- **Activation-aware scaling**: `_compute_edge_channels` trims channels per stage so activations remain under a simple SRAM budget (8-bit assumption).

## Repository Structure
```
microbackbone/
  models/        # backbone + modules
  data/          # datamodule + augmentations
  training/      # training loop, losses, metrics
  evaluation/    # eval, benchmark, export utilities
  config/        # yaml configs
deployment/
  raspberry_pi/  # inference scripts + instructions
  esp32/         # TFLite Micro sketch + instructions
```

## Installation
```bash
pip install -e .
# or
pip install -r requirements.txt
```

## Training
Train MicroSign-Edge (default) or swap in a TorchVision backbone via `--arch`:
```bash
# MicroSign-Edge
python -m microbackbone.training.train \
  --config microbackbone/config/model.yaml \
  --dataset-config microbackbone/config/datasets.yaml \
  --output-dir outputs

# TorchVision examples
python -m microbackbone.training.train --arch resnet18 --pretrained \
  --config microbackbone/config/model.yaml --dataset-config microbackbone/config/datasets.yaml --output-dir outputs

python -m microbackbone.training.train --arch mobilenet_v3_small --pretrained \
  --config microbackbone/config/model.yaml --dataset-config microbackbone/config/datasets.yaml --output-dir outputs

python -m microbackbone.training.train --arch efficientnet_b0 --pretrained \
  --config microbackbone/config/model.yaml --dataset-config microbackbone/config/datasets.yaml --output-dir outputs

python -m microbackbone.training.train --arch convnext_tiny --pretrained \
  --config microbackbone/config/model.yaml --dataset-config microbackbone/config/datasets.yaml --output-dir outputs
```
Checkpoints are written to `outputs/checkpoints/` with the architecture name embedded in the filename (e.g., `microsignedge_edge_small_best.pth`).

## Testing / Validation
```bash
python -m microbackbone.evaluation.test \
  --checkpoint outputs/checkpoints/microsignedge_edge_small_best.pth \
  --config microbackbone/config/model.yaml \
  --dataset-config microbackbone/config/datasets.yaml

# Evaluate a TorchVision checkpoint (arch auto-detected from checkpoint metadata)
python -m microbackbone.evaluation.test \
  --checkpoint outputs/checkpoints/resnet18_best.pth \
  --arch resnet18 --pretrained \
  --config microbackbone/config/model.yaml \
  --dataset-config microbackbone/config/datasets.yaml
```

## Benchmarking
```bash
# MicroSign-Edge (with optional reparameterization)
python -m microbackbone.evaluation.benchmark \
  --config microbackbone/config/model.yaml --input-size 32 --runs 200 --reparam-edge

# TorchVision backbone (pretrained head replaced for dataset classes)
python -m microbackbone.evaluation.benchmark \
  --arch resnet18 --pretrained \
  --config microbackbone/config/model.yaml --input-size 32 --runs 200
```

### Compare trained and pretrained models
Run side-by-side benchmarks for your checkpoints and TorchVision backbones (pretrained or scratch):
```bash
# Compare multiple trained checkpoints
python -m microbackbone.evaluation.compare_models \
  --checkpoints outputs/checkpoints/microsignedge_edge_small_best.pth outputs/checkpoints/resnet18_best.pth \
  --dataset-config microbackbone/config/datasets.yaml \
  --input-size 3 32 32 \
  --device cpu

# Compare trained checkpoint vs. pretrained TorchVision models
python -m microbackbone.evaluation.compare_models \
  --checkpoints outputs/checkpoints/microsignedge_edge_small_best.pth \
  --models resnet50,mobilenet_v3_large \
  --pretrained \
  --dataset-config microbackbone/config/datasets.yaml

# Benchmark all supported TorchVision backbones (pretrained) without checkpoints
python -m microbackbone.evaluation.compare_models --all --pretrained --dataset-config microbackbone/config/datasets.yaml
```
Outputs are printed as text + markdown tables and saved as `outputs/benchmarks/compare_models.csv` (create the directory if it does not exist). Metrics include params, FLOPs, latency, throughput, file size, and top-1 accuracy on the validation split defined by your dataset config.
TorchVision-only comparisons are supported (omit `--checkpoints` and use `--models`/`--all`), and the dataset config drives the evaluation dataloaders and `num_classes` when adapting classifier heads.

### TorchVision-only testing/benchmarking
- **Test a pretrained TorchVision model on your dataset** (classifier head auto-replaced by `num_classes` in the dataset config):
  ```bash
  python -m microbackbone.evaluation.test \
    --checkpoint /tmp/random_init.pth \
    --arch resnet50 --pretrained \
    --config microbackbone/config/model.yaml \
    --dataset-config microbackbone/config/datasets.yaml
  ```
- **Benchmark a TorchVision backbone** without training:
  ```bash
  python -m microbackbone.evaluation.benchmark --arch mobilenet_v3_small --pretrained --input-size 32
  ```
- **Compare multiple TorchVision architectures** (pretrained) with one command:
  ```bash
  python -m microbackbone.evaluation.compare_models --models resnet18,efficientnet_b0,convnext_tiny --pretrained --dataset-config microbackbone/config/datasets.yaml
  ```


## Exporting (TorchScript / ONNX / TFLite)
```bash
python -m microbackbone.evaluation.export_tflite \
  --checkpoint outputs/checkpoints/microsignedge_edge_small_best.pth \
  --config microbackbone/config/model.yaml \
  --input-size 32 \
  --output-dir outputs/export --quantize
```
Artifacts: `model.pth`, `model.torchscript.pt`, `model.onnx`, `model.tflite`/`model_int8.tflite`.

## Raspberry Pi Inference
See `deployment/raspberry_pi/instructions.md`. Quick start:
```bash
cd deployment/raspberry_pi
python3 infer.py --image sample.jpg --model ../outputs/export/model.torchscript.pt --input-size 32 --dataset cifar10
```

## ESP32 (TinyML)
See `deployment/esp32/instructions.md`. Export an int8 TFLite model, convert to `model_data.h`, and flash `esp32_main.cpp` with your board settings.

## Notes
- Adjust `microbackbone/config/model.yaml` to change variants or dataset classes.
- Use `microbackbone/config/datasets.yaml` to point to custom folders or tweak batch size.

## Pruning and Quantization
Use the standalone `model-tools` CLI to prune or quantize checkpoints without changing existing training workflows.

### Overview
- **Pruning**: structured (filter/channel) or unstructured (magnitude) sparsification using PyTorch prune utilities.
- **Quantization**: dynamic (no calibration) or static (with calibration data) for reduced size and latency.
- Outputs reuse the original checkpoint metadata and save to the path you provide.

### CLI usage
Prune a model:
```bash
python -m microbackbone.tools.model_tools prune \
  --input-model outputs/checkpoints/resnet18_best.pth \
  --output-model outputs/pruned/resnet18_pruned.pth \
  --pruning-type structured \
  --pruning-ratio 0.2
```

Quantize a model (dynamic):
```bash
python -m microbackbone.tools.model_tools quantize \
  --input-model outputs/checkpoints/microsignedge_edge_small_best.pth \
  --output-model outputs/quantized/microsignedge_int8.pth \
  --quantization-type dynamic
```

Quantize with static calibration:
```bash
python -m microbackbone.tools.model_tools quantize \
  --input-model outputs/checkpoints/resnet18_best.pth \
  --output-model outputs/quantized/resnet18_static_int8.pth \
  --quantization-type static \
  --calibration-data /path/to/calibration/folder \
  --input-size 32 \
  --batch-size 32
```

### Recommended settings
- **Unstructured pruning**: `--pruning-type=unstructured --pruning-ratio=0.3` (good sparsity/accuracy trade-off).
- **Structured pruning**: `--pruning-type=structured --pruning-ratio=0.2` (hardware-friendly for CNNs).
- **Workflow**: apply moderate pruning (20–30%), fine-tune 3–5 epochs, then export.
- **Dynamic quantization**: `--quantization-type=dynamic` (great general-purpose option, no calibration needed).
- **Static quantization**: `--quantization-type=static` with 200–500 representative samples via `--calibration-data`.
- Combine pruning then quantization for further size/speed gains; validate accuracy after each step.

### Outputs
- Saved checkpoints mirror the input format (state dict + metadata) at `--output-model`.
- Logs describe pruning/quantization steps according to `--log-level`.
- Calibrated static quantization consumes the provided sample folder but leaves it unchanged.

