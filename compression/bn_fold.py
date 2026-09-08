"""
BatchNorm folding.

A Conv2d followed by BatchNorm2d is mathematically equivalent to a single
Conv2d with modified weights and a bias:

    y = gamma * (conv(x) - mu) / sqrt(var + eps) + beta
      = conv_{W'}(x) + b'      with     W' = W * gamma / sqrt(var + eps)
                                        b' = (b - mu) * gamma / sqrt(var+eps) + beta

Why we do this before quantizing:

  1. It removes every BatchNorm parameter and running statistic from the
     stored model. In MobileNet-v2 that is ~35k float values that would
     otherwise have to be stored at full precision, since BN cannot be
     quantized aggressively without destabilising inference.
  2. It is what a real integer inference kernel does -- BN is never a
     separate runtime op in a deployed quantized model, so folding first
     means our accuracy numbers correspond to a deployable artefact.

Important interaction: folding multiplies each output channel of W by a
different gamma/sigma, which *widens* the spread of per-channel weight
magnitudes. This is precisely why per-channel weight quantization is not
optional for a folded MobileNet-v2.
"""

import torch
import torch.nn as nn


def fold_conv_bn(conv, bn):
    """Fold `bn` into `conv` in place and return the modified conv."""
    w = conv.weight.data
    gamma, beta = bn.weight.data, bn.bias.data
    mu, var, eps = bn.running_mean, bn.running_var, bn.eps

    factor = gamma / torch.sqrt(var + eps)                       # [C_out]
    conv.weight.data = w * factor.reshape(-1, *([1] * (w.dim() - 1)))

    b = conv.bias.data if conv.bias is not None else torch.zeros_like(mu)
    b_fold = (b - mu) * factor + beta
    if conv.bias is None:
        conv.bias = nn.Parameter(b_fold)
    else:
        conv.bias.data = b_fold
    return conv


def fold_bn_in_model(model):
    """Walk every Sequential container and fold adjacent (Conv2d, BatchNorm2d).

    In torchvision's MobileNet-v2 every convolution is immediately followed
    by a BatchNorm2d inside a Sequential (Conv2dNormActivation blocks, and
    the final linear-bottleneck projection of each InvertedResidual), so a
    single pass over Sequentials folds all 52 of them.

    Returns (model, n_folded).
    """
    n_folded = 0
    for module in model.modules():
        if not isinstance(module, nn.Sequential):
            continue
        children = list(module.named_children())
        for i in range(len(children) - 1):
            (name_a, a), (name_b, b) = children[i], children[i + 1]
            if isinstance(a, nn.Conv2d) and isinstance(b, nn.BatchNorm2d):
                fold_conv_bn(a, b)
                setattr(module, name_b, nn.Identity())
                n_folded += 1
    return model, n_folded


def count_bn_values(model):
    """Float values held by BatchNorm layers (weight, bias, running stats).

    Used to show what folding saves in the storage accounting.
    """
    total = 0
    for m in model.modules():
        if isinstance(m, nn.BatchNorm2d):
            total += m.weight.numel() + m.bias.numel()
            total += m.running_mean.numel() + m.running_var.numel()
    return total
