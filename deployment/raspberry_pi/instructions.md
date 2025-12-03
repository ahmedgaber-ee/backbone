# Raspberry Pi deployment (CPU/GPU)

End-to-end steps to run MicroSign-Edge or TorchVision models on Raspberry Pi (Zero/4/5). Examples assume Python 3 is installed.

## 1. Install dependencies
```bash
sudo apt update && sudo apt install -y python3-pip libatlas-base-dev
pip3 install -r requirements_rpi.txt
```
Pi 5 with GPU drivers: ensure Mesa/Vulkan stack is installed and pass `--device cuda` if CUDA is available; otherwise default to `cpu`.

## 2. Bring your model artifacts
Copy one of the exported files into `deployment/raspberry_pi/`:
- `model.torchscript.pt` (fastest startup, recommended)
- `model.pth` (PyTorch state dict)
- `model.onnx` (optional for ONNX Runtime if installed)

## 3. Static image inference
```bash
python3 infer.py --image path/to/test.jpg \
  --model model.torchscript.pt \
  --variant edge_small \
  --num-classes 10 \
  --input-size 32 \
  --dataset cifar10 \
  --device cpu
```
Swap `--device cuda` if the Pi has a working GPU driver.

## 4. Benchmark latency / throughput
Use the repo benchmark helper for MicroSign-Edge or TorchVision architectures. From the repository root:
```bash
# MicroSign-Edge (optionally load a checkpoint)
python3 -m microbackbone.evaluation.benchmark \
  --config microbackbone/config/model.yaml \
  --arch microsign_edge \
  --input-size 32 \
  --device cpu \
  --runs 200 \
  --weights outputs/checkpoints/best.pth \
  --reparam-edge

# TorchVision pretrained (e.g., ResNet-18)
python3 -m microbackbone.evaluation.benchmark \
  --config microbackbone/config/model.yaml \
  --arch resnet18 \
  --pretrained \
  --input-size 224 \
  --device cpu \
  --runs 200

# TorchVision with your fine-tuned checkpoint
python3 -m microbackbone.evaluation.benchmark \
  --config microbackbone/config/model.yaml \
  --arch mobilenet_v3_small \
  --input-size 224 \
  --device cpu \
  --runs 200 \
  --weights outputs/checkpoints/mobilenet_v3_small_best.pth
```
Set `--device cuda` to test Pi GPU performance if available.

## 5. Camera streaming inference
```bash
python3 camera_infer.py \
  --model model.torchscript.pt \
  --variant edge_small \
  --num-classes 10 \
  --input-size 32 \
  --dataset cifar10 \
  --device cpu \
  --camera 0
```
Press `q` to exit the preview window. Increase `--input-size` or switch `--arch` for TorchVision models if needed.

## 6. Monitor performance
```bash
sudo apt install -y htop
htop
```
Observe CPU/GPU utilization while inference or benchmarking runs.

## 7. Tips
- Prefer TorchScript for faster Pi startup and lower memory than raw PyTorch checkpoints.
- Keep input sizes small (e.g., 32×32 or 64×64) for Pi Zero/3; Pi 4/5 can handle larger TorchVision models.
- When benchmarking on Pi GPUs, ensure you synchronize timings with the provided scripts and avoid additional background workload.
