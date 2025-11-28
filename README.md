# MicroBackbone (MicroSign-Net)

Lightweight CNN backbone tailored for microcontrollers and edge devices. Extracted from the original notebook into a modular, deployable Python package with export paths for Raspberry Pi and ESP32 (TFLite Micro).

## Key Features
- Variant-aware MicroSign-Net backbone (`nano` → `base`)
- CIFAR-10 friendly stem for 32×32 data
- Dynamic depthwise convolutions, SE blocks, spatial attention
- Training/eval scripts with clean configs
- Export to TorchScript, ONNX, and TFLite (int8 for ESP32)
- Deployment recipes for Raspberry Pi and ESP32

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
```bash
python -m microbackbone.training.train \
  --config microbackbone/config/model.yaml \
  --dataset-config microbackbone/config/datasets.yaml \
  --output-dir outputs
```
Checkpoints are written to `outputs/checkpoints/`.

## Testing
```bash
python -m microbackbone.evaluation.test \
  --checkpoint outputs/checkpoints/microsign_micro_best.pth \
  --config microbackbone/config/model.yaml \
  --dataset-config microbackbone/config/datasets.yaml
```

## Benchmarking
```bash
python -m microbackbone.evaluation.benchmark --config microbackbone/config/model.yaml --input-size 32 --runs 200
```

## Exporting (TorchScript / ONNX / TFLite)
```bash
python -m microbackbone.evaluation.export_tflite \
  --checkpoint outputs/checkpoints/microsign_micro_best.pth \
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
