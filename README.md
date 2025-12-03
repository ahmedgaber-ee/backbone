# MicroBackbone (MicroSign-Edge)

Lightweight CNN backbones for microcontrollers, Raspberry Pi, and desktop GPUs. The repository includes the MicroSign-Edge family plus a TorchVision registry so you can train, evaluate, and benchmark your models with a single interface.

## What you get
- **Backbone variants:** `edge_nano`, `edge_micro`, `edge_small` with reparameterization for MCU-friendly inference.
- **TorchVision support:** drop-in training/eval/benchmarking for common architectures.
- **Deployability:** export to TorchScript, ONNX, and int8 TFLite; reference guides for Raspberry Pi and ESP32 (TFLite Micro).
- **Benchmarks:** latency/FLOPs/throughput measurement with deterministic data splits.

## Supported models
### MicroSign-Edge (built-in)
- `microsign_edge` with variants: `edge_nano`, `edge_micro`, `edge_small` (classification, detection, or backbone-only outputs).

### TorchVision (registry)
`resnet18`, `resnet34`, `resnet50`, `mobilenet_v2`, `mobilenet_v3_small`, `mobilenet_v3_large`, `shufflenet_v2_x0_5`, `shufflenet_v2_x1_0`, `mnasnet0_5`, `mnasnet0_75`, `mnasnet1_3`, `efficientnet_b0`, `efficientnet_b1`, `convnext_tiny`, `squeezenet1_0`, `squeezenet1_1`, `googlenet`, `densenet121`, `regnet_y_400mf`, `regnet_y_800mf` (pretrained weights optional).

## Installation
```bash
pip install -e .               # editable install
# or
pip install -r requirements.txt
```

## Configuration
- `microbackbone/config/model.yaml`: task, variant, num_classes, and optimizer/loss knobs.
- `microbackbone/config/datasets.yaml`: dataset name, data root, batch size, workers, input size, train/val split, seed.

## Training (GPU or CPU)
Train MicroSign-Edge or any supported TorchVision model. Set `--device cuda` for GPU (or `cpu` for portability/Raspberry Pi without GPU).
```bash
# MicroSign-Edge on GPU
python -m microbackbone.training.train \
  --config microbackbone/config/model.yaml \
  --dataset-config microbackbone/config/datasets.yaml \
  --output-dir outputs \
  --device cuda

# TorchVision example (pretrained ResNet-18) on CPU
python -m microbackbone.training.train \
  --arch resnet18 --pretrained \
  --config microbackbone/config/model.yaml \
  --dataset-config microbackbone/config/datasets.yaml \
  --output-dir outputs \
  --device cpu
```
Checkpoints land in `outputs/checkpoints/<arch>_best.pth` with metadata for arch/variant/num_classes.

## Evaluation / Testing
```bash
# Evaluate a MicroSign-Edge checkpoint
python -m microbackbone.evaluation.test \
  --checkpoint outputs/checkpoints/microsignedge_edge_small_best.pth \
  --config microbackbone/config/model.yaml \
  --dataset-config microbackbone/config/datasets.yaml \
  --metrics precision,recall,f1 \
  --device cuda

# Evaluate a TorchVision checkpoint
python -m microbackbone.evaluation.test \
  --checkpoint outputs/checkpoints/resnet18_best.pth \
  --arch resnet18 --pretrained \
  --config microbackbone/config/model.yaml \
  --dataset-config microbackbone/config/datasets.yaml \
  --device cpu
```
Reports include top-1/top-5 accuracy and optional macro/micro precision, recall, F1. Save JSON with `--save-report <path>`.

## Benchmarking (latency/FLOPs/throughput)
```bash
# MicroSign-Edge (optionally reparameterize for inference)
python -m microbackbone.evaluation.benchmark \
  --config microbackbone/config/model.yaml \
  --input-size 32 \
  --runs 200 \
  --device cuda \
  --reparam-edge

# TorchVision backbone (pretrained head adapted to dataset classes)
python -m microbackbone.evaluation.benchmark \
  --arch mobilenet_v3_small --pretrained \
  --config microbackbone/config/model.yaml \
  --input-size 224 \
  --runs 200 \
  --device cpu
```
Benchmarks honor deterministic seeds from `datasets.yaml`, warm up before timing, and synchronize CUDA to avoid skew. FLOPs use THOP when available.

### Compare multiple models side-by-side
```bash
# Compare trained checkpoints
python -m microbackbone.evaluation.compare_models \
  --checkpoints outputs/checkpoints/microsignedge_edge_small_best.pth outputs/checkpoints/resnet18_best.pth \
  --dataset-config microbackbone/config/datasets.yaml \
  --input-size 3 32 32 \
  --device cuda

# Compare a wider set of checkpoints on CPU with CIFAR-10 input shape
python -m microbackbone.evaluation.compare_models \
  --checkpoints outputs/checkpoints/microsignedge_edge_small_best.pth outputs/checkpoints/resnet18_best.pth \
  outputs/checkpoints/efficientnet_b0_best.pth outputs/checkpoints/mobilenet_v3_small_best.pth \
  outputs/checkpoints/shufflenet_v2_x0_5_best.pth outputs/checkpoints/shufflenet_v2_x1_0_best.pth \
  --dataset-config microbackbone/config/datasets.yaml \
  --input-size 3 32 32 \
  --device cuda

# TorchVision-only comparison (pretrained)
python -m microbackbone.evaluation.compare_models \
  --models resnet18,efficientnet_b0,convnext_tiny \
  --pretrained \
  --dataset-config microbackbone/config/datasets.yaml \
  --device cpu
```
Outputs are printed as text + markdown tables and saved to `outputs/benchmarks/compare_models.csv` with params, FLOPs, latency, throughput, file size, top-1/top-5 accuracy, precision, recall, and F1 score.

## Exporting
Create TorchScript/ONNX/TFLite artifacts for deployment.
```bash
python -m microbackbone.evaluation.export_tflite \
  --checkpoint outputs/checkpoints/microsignedge_edge_small_best.pth \
  --config microbackbone/config/model.yaml \
  --input-size 32 \
  --output-dir outputs/export \
  --quantize
```
Artifacts: `model.pth`, `model.torchscript.pt`, `model.onnx`, `model.tflite` (plus `model_int8.tflite` when `--quantize`).

## Raspberry Pi (CPU/GPU) quick start
See `deployment/raspberry_pi/instructions.md` for full steps. Minimal example:
```bash
cd deployment/raspberry_pi
python3 infer.py --image sample.jpg \
  --model ../outputs/export/model.torchscript.pt \
  --variant edge_small --num-classes 10 --input-size 32 --dataset cifar10 --device cpu
```
Use `--device cuda` on Pi 5 if drivers are installed. Camera streaming is supported via `camera_infer.py`.

## ESP32 (TFLite Micro)
See `deployment/esp32/instructions.md`. Export int8 TFLite, convert to `model_data.h`, and flash `esp32_main.cpp` with your board config.

## Reproducibility tips
- Set the `seed` field in `microbackbone/config/datasets.yaml` for deterministic train/val splits and benchmarking seeds.
- Keep `input_size`, `batch_size`, and `train_split` consistent between training, testing, and benchmarking.
- For GPUs, consider `torch.backends.cudnn.benchmark = False` if you need determinism over speed.

## Helpful paths
- Training/eval configs: `microbackbone/config/`
- Models: `microbackbone/models/`
- Data module: `microbackbone/data/`
- Training loop: `microbackbone/training/train.py`
- Eval/benchmark tools: `microbackbone/evaluation/`
- Deployment: `deployment/`

## Licensing and citation
This repository is provided as-is for research and edge deployment experiments. Cite MicroSign-Edge if you build upon the backbone or benchmarks.
