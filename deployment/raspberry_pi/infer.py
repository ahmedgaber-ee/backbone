"""Offline image inference for Raspberry Pi."""
from __future__ import annotations

import argparse
from pathlib import Path

import torch
from PIL import Image
import torchvision.transforms as T

from microbackbone.models.utils import create_model
from microbackbone.data.augmentations import CIFAR_MEAN, CIFAR_STD, IMAGENET_MEAN, IMAGENET_STD


def build_preprocess(input_size: int, dataset: str) -> T.Compose:
    if dataset.lower() == "cifar10":
        return T.Compose([T.Resize((input_size, input_size)), T.ToTensor(), T.Normalize(CIFAR_MEAN, CIFAR_STD)])
    return T.Compose(
        [
            T.Resize(int(input_size * 1.14)),
            T.CenterCrop(input_size),
            T.ToTensor(),
            T.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ]
    )


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
    parser = argparse.ArgumentParser(description="Raspberry Pi inference")
    parser.add_argument("--image", type=str, required=True)
    parser.add_argument("--model", type=str, required=True, help="Path to .pth or .pt model")
    parser.add_argument("--variant", type=str, default="micro")
    parser.add_argument("--num-classes", type=int, default=10)
    parser.add_argument("--input-size", type=int, default=32)
    parser.add_argument("--dataset", type=str, default="cifar10")
    parser.add_argument("--device", type=str, default="cpu")
    args = parser.parse_args()

    device = torch.device(args.device)
    model = load_model(Path(args.model), args.variant, args.num_classes, args.input_size, device=device)
    preprocess = build_preprocess(args.input_size, args.dataset)

    img = Image.open(args.image).convert("RGB")
    tensor = preprocess(img).unsqueeze(0).to(device)

    with torch.inference_mode():
        logits = model(tensor)
        probs = torch.softmax(logits, dim=1)
        pred = probs.argmax(dim=1).item()
        confidence = probs.max().item()

    print(f"Prediction: class={pred} confidence={confidence:.4f}")


if __name__ == "__main__":
    main()
