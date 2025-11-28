"""Live camera inference script for Raspberry Pi."""
from __future__ import annotations

import argparse
import time
from pathlib import Path

import cv2
import torch
import torchvision.transforms as T

from microbackbone.data.augmentations import CIFAR_MEAN, CIFAR_STD, IMAGENET_MEAN, IMAGENET_STD
from microbackbone.models.utils import create_model


def preprocess_frame(frame, input_size: int, dataset: str) -> torch.Tensor:
    transform = (
        T.Compose([T.ToTensor(), T.Resize((input_size, input_size)), T.Normalize(CIFAR_MEAN, CIFAR_STD)])
        if dataset.lower() == "cifar10"
        else T.Compose(
            [
                T.ToTensor(),
                T.Resize(int(input_size * 1.14)),
                T.CenterCrop(input_size),
                T.Normalize(IMAGENET_MEAN, IMAGENET_STD),
            ]
        )
    )
    tensor = transform(frame).unsqueeze(0)
    return tensor


def load_model(model_path: Path, variant: str, num_classes: int, input_size: int, device: str) -> torch.nn.Module:
    if model_path.suffix == ".pt":
        model = torch.jit.load(model_path, map_location=device)
    else:
        model = create_model("classification", num_classes=num_classes, variant=variant, input_size=input_size)
        state = torch.load(model_path, map_location=device)
        state = state.get("model_state_dict", state)
        model.load_state_dict(state, strict=False)
    model.to(device)
    model.eval()
    return model


def main() -> None:
    parser = argparse.ArgumentParser(description="Camera inference on Raspberry Pi")
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--variant", type=str, default="micro")
    parser.add_argument("--num-classes", type=int, default=10)
    parser.add_argument("--input-size", type=int, default=32)
    parser.add_argument("--dataset", type=str, default="cifar10")
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--camera", type=int, default=0)
    args = parser.parse_args()

    device = torch.device(args.device)
    model = load_model(Path(args.model), args.variant, args.num_classes, args.input_size, device=device)

    cap = cv2.VideoCapture(args.camera)
    if not cap.isOpened():
        raise RuntimeError("Could not open camera")

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            tensor = preprocess_frame(rgb, args.input_size, args.dataset).to(device)
            start = time.time()
            with torch.inference_mode():
                logits = model(tensor)
                probs = torch.softmax(logits, dim=1)
                pred = probs.argmax(dim=1).item()
                confidence = probs.max().item()
            latency_ms = (time.time() - start) * 1000
            text = f"cls {pred} ({confidence:.2f}) {latency_ms:.1f}ms"
            cv2.putText(frame, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
            cv2.imshow("MicroBackbone", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    finally:
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
