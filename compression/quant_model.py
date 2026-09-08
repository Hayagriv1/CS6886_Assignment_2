"""
Quantized module wrappers and model surgery for MobileNet-v2.

Design: every nn.Conv2d and nn.Linear is replaced by a QuantWrapper that
  (i)  fake-quantizes its INPUT activation (asymmetric, per-tensor), and
  (ii) fake-quantizes its WEIGHT (symmetric, per-channel by default).

Quantizing at layer *inputs* rather than outputs means each intermediate
tensor is quantized exactly once, so the activation accounting below is not
double counting. Elementwise ops that consume the results (the residual
additions in InvertedResidual, ReLU6) are left in float; their inputs are
already quantized values and requantizing them would add error without
reducing what has to be stored or moved.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from .quantizers import fake_quant_weight, fake_quant_act, tensor_range

# Layers exempted from low-bit weight quantization. The stem convolution
# sees the raw image (no preceding layer has smoothed its input
# distribution) and the classifier directly produces logits; both are
# disproportionately sensitive while together holding <2% of parameters.
SENSITIVE_LAYERS = ("features.0.0", "classifier.1")


class QuantWrapper(nn.Module):
    def __init__(self, module, name, w_bits, a_bits,
                 per_channel=True, quantize_act=True, percentile=0.999,
                 ste=False):
        super().__init__()
        self.module = module
        self.name = name
        self.w_bits = w_bits
        self.a_bits = a_bits
        self.per_channel = per_channel
        self.quantize_act = quantize_act
        self.percentile = percentile
        self.ste = ste

        self.calibrating = False
        self.enabled = True
        self.register_buffer("a_min", torch.tensor(float("inf")))
        self.register_buffer("a_max", torch.tensor(float("-inf")))
        self.act_numel = 0      # elements in this layer's input tensor (bs=1)

    # -- calibration -------------------------------------------------------
    def _observe(self, x):
        lo, hi = tensor_range(x, self.percentile)
        if torch.isinf(self.a_min):
            self.a_min.fill_(lo)
            self.a_max.fill_(hi)
        else:
            # EMA smooths batch-to-batch variation in the observed range
            self.a_min.mul_(0.9).add_(0.1 * lo)
            self.a_max.mul_(0.9).add_(0.1 * hi)
        self.act_numel = x[0].numel()

    # -- forward -----------------------------------------------------------
    def _op(self, x, w):
        m = self.module
        if isinstance(m, nn.Conv2d):
            return F.conv2d(x, w, m.bias, m.stride, m.padding,
                            m.dilation, m.groups)
        return F.linear(x, w, m.bias)

    def forward(self, x):
        if self.calibrating:
            self._observe(x)
            return self._op(x, self.module.weight)

        if not self.enabled:
            return self._op(x, self.module.weight)

        if self.quantize_act and not torch.isinf(self.a_min):
            x = fake_quant_act(x, self.a_bits,
                               self.a_min.item(), self.a_max.item(),
                               ste=self.ste)
        w = fake_quant_weight(self.module.weight, self.w_bits,
                              self.per_channel, ste=self.ste)
        return self._op(x, w)


# --------------------------------------------------------------------------
# Model surgery
# --------------------------------------------------------------------------

def _get_parent(model, dotted_name):
    parts = dotted_name.split(".")
    parent = model
    for p in parts[:-1]:
        parent = getattr(parent, p) if not p.isdigit() else parent[int(p)]
    return parent, parts[-1]


def quantize_model(model, w_bits, a_bits, per_channel=True,
                   quantize_act=True, protect_sensitive=True,
                   percentile=0.999, ste=False):
    """Replace all Conv2d / Linear layers with QuantWrappers.

    Returns (model, list_of_wrappers).
    """
    targets = [n for n, m in model.named_modules()
               if isinstance(m, (nn.Conv2d, nn.Linear))]

    wrappers = []
    for name in targets:
        parent, attr = _get_parent(model, name)
        orig = getattr(parent, attr) if not attr.isdigit() else parent[int(attr)]

        bits_w = 8 if (protect_sensitive and name in SENSITIVE_LAYERS) else w_bits
        wrapped = QuantWrapper(orig, name, bits_w, a_bits, per_channel,
                               quantize_act, percentile, ste)
        if attr.isdigit():
            parent[int(attr)] = wrapped
        else:
            setattr(parent, attr, wrapped)
        wrappers.append(wrapped)
    return model, wrappers


@torch.no_grad()
def calibrate(model, wrappers, loader, device, n_batches=8):
    """Estimate activation ranges on a small sample of TRAINING data.

    Calibration must not touch the test set. 8 batches of 128 = 1024 images
    is ample; the percentile estimates are stable well before that.
    """
    for w in wrappers:
        w.calibrating = True
    model.eval()
    for i, (images, _) in enumerate(loader):
        if i >= n_batches:
            break
        model(images.to(device))
    for w in wrappers:
        w.calibrating = False
    return model


def set_enabled(wrappers, flag):
    for w in wrappers:
        w.enabled = flag
