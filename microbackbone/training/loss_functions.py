"""Loss utilities."""
from __future__ import annotations

import torch.nn as nn


def classification_loss(num_classes: int, label_smoothing: float = 0.0) -> nn.Module:
    return nn.CrossEntropyLoss(label_smoothing=label_smoothing)


__all__ = ["classification_loss"]
