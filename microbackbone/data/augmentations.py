"""Data augmentation utilities."""
from __future__ import annotations

from typing import Tuple

import torchvision.transforms as T


CIFAR_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR_STD = (0.2470, 0.2435, 0.2616)
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def build_transforms(dataset: str, input_size: int = 224) -> Tuple[T.Compose, T.Compose]:
    """Return train/test transforms for a dataset name."""
    if dataset.lower() == "cifar10":
        train_tfms = T.Compose(
            [
                T.RandomHorizontalFlip(p=0.5),
                T.RandomCrop(32, padding=4),
                T.ToTensor(),
                T.Normalize(CIFAR_MEAN, CIFAR_STD),
            ]
        )
        test_tfms = T.Compose([T.ToTensor(), T.Normalize(CIFAR_MEAN, CIFAR_STD)])
    else:
        train_tfms = T.Compose(
            [
                T.RandomResizedCrop(input_size, scale=(0.6, 1.0)),
                T.RandomHorizontalFlip(),
                T.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.05),
                T.ToTensor(),
                T.Normalize(IMAGENET_MEAN, IMAGENET_STD),
            ]
        )
        test_tfms = T.Compose(
            [T.Resize(int(input_size * 1.14)), T.CenterCrop(input_size), T.ToTensor(), T.Normalize(IMAGENET_MEAN, IMAGENET_STD)]
        )
    return train_tfms, test_tfms


__all__ = ["build_transforms", "CIFAR_MEAN", "CIFAR_STD", "IMAGENET_MEAN", "IMAGENET_STD"]
