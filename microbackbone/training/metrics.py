"""Training and evaluation metrics utilities."""
from __future__ import annotations

from typing import Tuple

import torch


def topk_accuracy(output: torch.Tensor, target: torch.Tensor, topk: Tuple[int, ...] = (1,)) -> list[torch.Tensor]:
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)
        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))
        res = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res


def update_confusion_matrix(confusion: torch.Tensor, preds: torch.Tensor, labels: torch.Tensor) -> None:
    """Accumulate a confusion matrix given predictions and labels.

    Args:
        confusion: Tensor of shape (num_classes, num_classes) where rows are ground truth and columns are predictions.
        preds: Model predictions (shape: [batch]).
        labels: Ground-truth labels (shape: [batch]).
    """

    for p, t in zip(preds.view(-1), labels.view(-1)):
        confusion[t.long(), p.long()] += 1


def classification_report_from_confusion(confusion: torch.Tensor, eps: float = 1e-8) -> dict[str, float]:
    """Compute macro/micro precision, recall, and F1 from a confusion matrix.

    Returns a dictionary with keys: precision_macro, recall_macro, f1_macro, precision_micro, recall_micro, f1_micro.
    """

    tp = torch.diag(confusion).float()
    per_class_total = confusion.sum(dim=1).float()
    predicted_total = confusion.sum(dim=0).float()

    precision_per_class = tp / (predicted_total + eps)
    recall_per_class = tp / (per_class_total + eps)
    f1_per_class = 2 * precision_per_class * recall_per_class / (precision_per_class + recall_per_class + eps)

    precision_macro = precision_per_class.mean().item()
    recall_macro = recall_per_class.mean().item()
    f1_macro = f1_per_class.mean().item()

    tp_micro = tp.sum()
    precision_micro = tp_micro / (predicted_total.sum() + eps)
    recall_micro = tp_micro / (per_class_total.sum() + eps)
    f1_micro = 2 * precision_micro * recall_micro / (precision_micro + recall_micro + eps)

    return {
        "precision_macro": precision_macro,
        "recall_macro": recall_macro,
        "f1_macro": f1_macro,
        "precision_micro": precision_micro.item(),
        "recall_micro": recall_micro.item(),
        "f1_micro": f1_micro.item(),
    }


__all__ = [
    "topk_accuracy",
    "update_confusion_matrix",
    "classification_report_from_confusion",
]
