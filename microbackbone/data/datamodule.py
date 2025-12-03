"""Data loading and preprocessing module."""
from __future__ import annotations

from pathlib import Path
from typing import Optional, Tuple

import torch
from torch.utils.data import DataLoader, random_split
from torchvision.datasets import CIFAR10, ImageFolder

from .augmentations import build_transforms


class MicroBackboneDataModule:
    """Thin data module wrapper to mirror notebook data handling."""

    def __init__(
        self,
        dataset: str,
        data_root: str | Path,
        batch_size: int,
        num_workers: int,
        input_size: int,
        train_split: float = 0.9,
        download: bool = True,
        seed: int | None = None,
    ) -> None:
        self.dataset = dataset.lower()
        self.data_root = Path(data_root)
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.input_size = input_size
        self.train_split = train_split
        self.download = download
        self.seed = seed

        self.train_set = None
        self.val_set = None
        self.test_set = None

    def prepare_data(self) -> None:
        train_tfms, test_tfms = build_transforms(self.dataset, self.input_size)
        if self.dataset == "cifar10":
            full_train = CIFAR10(root=self.data_root, train=True, download=self.download, transform=train_tfms)
            self.test_set = CIFAR10(root=self.data_root, train=False, download=self.download, transform=test_tfms)
            val_len = int(len(full_train) * (1 - self.train_split))
            train_len = len(full_train) - val_len
            generator = torch.Generator().manual_seed(self.seed) if self.seed is not None else None
            self.train_set, self.val_set = random_split(full_train, [train_len, val_len], generator=generator)
        else:
            train_dir = self.data_root / "train"
            val_dir = self.data_root / "val"
            test_dir = self.data_root / "test"
            self.train_set = ImageFolder(train_dir, transform=train_tfms)
            self.val_set = ImageFolder(val_dir, transform=test_tfms)
            self.test_set = ImageFolder(test_dir, transform=test_tfms)

    def setup(self) -> None:
        if self.train_set is None or self.val_set is None or self.test_set is None:
            self.prepare_data()

    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            self.train_set,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=True,
        )

    def val_dataloader(self) -> DataLoader:
        return DataLoader(
            self.val_set,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
        )

    def test_dataloader(self) -> DataLoader:
        return DataLoader(
            self.test_set,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
        )


__all__ = ["MicroBackboneDataModule"]
