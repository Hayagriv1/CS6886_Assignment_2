"""
Evaluation utilities, shared by training and compression.

Kept separate from both so that accuracy is measured by exactly the same
code path whether the model is FP32 or quantized.
"""

import numpy as np
import torch
import torchvision.models as models

from data import CLASSES


def build_model(checkpoint, device):
    """Rebuild the CIFAR-adapted MobileNet-v2 and load a checkpoint."""
    m = models.mobilenet_v2(weights=None, num_classes=10)
    m.features[0][0].stride = (1, 1)
    m.features[2].conv[1][0].stride = (1, 1)
    m.load_state_dict(torch.load(checkpoint, map_location="cpu", weights_only=True))
    return m.to(device).eval()


@torch.no_grad()
def top1_accuracy(model, loader, device):
    correct = total = 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        correct += model(x).argmax(1).eq(y).sum().item()
        total += y.size(0)
    return 100.0 * correct / total


@torch.no_grad()
def confusion_matrix(model, loader, device):
    """Row-normalisable confusion matrix, rows = true class."""
    cm = np.zeros((10, 10), dtype=int)
    for x, y in loader:
        pred = model(x.to(device)).argmax(1).cpu().numpy()
        for t, p in zip(np.asarray(y), pred):
            cm[t, p] += 1
    return cm


def per_class_accuracy(cm):
    return {c: 100.0 * cm[i, i] / cm[i].sum() for i, c in enumerate(CLASSES)}


def top_confusions(cm, k=6):
    off = cm.copy()
    np.fill_diagonal(off, 0)
    out = []
    for idx in np.argsort(off.ravel())[::-1][:k]:
        i, j = divmod(idx, 10)
        out.append((CLASSES[i], CLASSES[j], int(off[i, j])))
    return out
