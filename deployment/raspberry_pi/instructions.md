# Raspberry Pi Deployment Guide

## 1. Install Dependencies
```bash
sudo apt update && sudo apt install -y python3-pip libatlas-base-dev
pip3 install -r requirements_rpi.txt
```

## 2. Copy Model Artifacts
Place one of the exported artifacts into this folder:
- `model.pth` (PyTorch weights)
- `model.torchscript.pt` (TorchScript)
- `model.onnx` (optional)

## 3. Run Static Image Inference
```bash
python3 infer.py --image path/to/test.jpg --model model.torchscript.pt --variant micro --num-classes 10 --input-size 32 --dataset cifar10 --device cpu
```

## 4. Measure FPS
Use the benchmark helper from the repo root:
```bash
python3 -m microbackbone.evaluation.benchmark --config microbackbone/config/model.yaml --input-size 32 --device cpu --runs 200
```

## 5. Enable Camera Inference
```bash
python3 camera_infer.py --model model.torchscript.pt --variant micro --num-classes 10 --input-size 32 --dataset cifar10 --device cpu --camera 0
```
Press `q` to exit.

## 6. Profile CPU Usage
```bash
sudo apt install -y htop
htop
```
Observe the Python process while running inference.

## 7. Notes
- Prefer TorchScript on Pi for faster startup.
- Use `--device cuda` on Pi 5 with GPU drivers.
