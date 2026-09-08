"""
Core quantization primitives.

Everything here is written from scratch (no torch.quantization / no
compression library calls). Two schemes are provided:

  * Symmetric per-channel / per-tensor quantization  -> used for WEIGHTS
  * Asymmetric (affine) per-tensor quantization      -> used for ACTIVATIONS

We use "fake quantization" (quantize -> dequantize in float) so that the
network can be evaluated with ordinary float kernels while reproducing
exactly the numerical error an integer kernel would produce. The real
storage cost is computed analytically in compression/size_accounting.py.
"""

import torch


# --------------------------------------------------------------------------
# Symmetric quantization (weights)
# --------------------------------------------------------------------------

def symmetric_scale(w, bits, per_channel=True, ch_axis=0):
    """Scale factor(s) for symmetric signed quantization to `bits` bits.

    Signed range is [-2^(b-1), 2^(b-1)-1]. The scale is derived from the
    positive limit so that zero maps exactly to integer 0 (no zero point),
    which keeps the integer kernel free of cross terms.

    per_channel=True gives one scale per output channel (axis 0 for both
    Conv2d weights [C_out, C_in/g, kH, kW] and Linear weights [out, in]).
    """
    qmax = 2 ** (bits - 1) - 1
    if per_channel:
        reduce_dims = [d for d in range(w.dim()) if d != ch_axis]
        amax = w.detach().abs().amax(dim=reduce_dims, keepdim=True)
    else:
        amax = w.detach().abs().amax()
    scale = (amax / qmax).clamp(min=1e-12)
    return scale, qmax


def quantize_weight(w, bits, per_channel=True, ch_axis=0):
    """Return (integer codes, scale). Codes are floats holding integer values."""
    scale, qmax = symmetric_scale(w, bits, per_channel, ch_axis)
    q = torch.clamp(torch.round(w / scale), -qmax - 1, qmax)
    return q, scale


def fake_quant_weight(w, bits, per_channel=True, ch_axis=0, ste=False):
    """Quantize then dequantize a weight tensor.

    ste=True routes gradients around the non-differentiable round() using a
    straight-through estimator, which is what makes quantization-aware
    fine-tuning possible.
    """
    q, scale = quantize_weight(w, bits, per_channel, ch_axis)
    w_hat = q * scale
    if ste:
        return w + (w_hat - w).detach()
    return w_hat


# --------------------------------------------------------------------------
# Affine / asymmetric quantization (activations)
# --------------------------------------------------------------------------

def affine_qparams(x_min, x_max, bits):
    """Scale and zero point for unsigned affine quantization to [0, 2^b - 1].

    The observed range is widened to include zero so that padding zeros and
    post-ReLU zeros are exactly representable.
    """
    qmin, qmax = 0, 2 ** bits - 1
    x_min = float(min(x_min, 0.0))
    x_max = float(max(x_max, 0.0))
    scale = max((x_max - x_min) / (qmax - qmin), 1e-12)
    zero_point = int(round(qmin - x_min / scale))
    zero_point = max(qmin, min(qmax, zero_point))
    return scale, zero_point


def fake_quant_act(x, bits, x_min, x_max, ste=False):
    """Quantize then dequantize an activation tensor."""
    qmin, qmax = 0, 2 ** bits - 1
    scale, zp = affine_qparams(x_min, x_max, bits)
    q = torch.clamp(torch.round(x / scale) + zp, qmin, qmax)
    x_hat = (q - zp) * scale
    if ste:
        return x + (x_hat - x).detach()
    return x_hat


# --------------------------------------------------------------------------
# Calibration helper
# --------------------------------------------------------------------------

def tensor_range(x, percentile=None, max_samples=1_000_000):
    """Observed range of a tensor.

    percentile=None  -> absolute min/max (sensitive to single outliers)
    percentile=0.999 -> clip the extreme 0.1% on each side. On MobileNet-v2
                        this matters a lot: the expand layers produce rare
                        large activations that otherwise consume most of the
                        quantization levels.

    Large tensors are randomly subsampled because torch.quantile has an
    input-size limit.
    """
    flat = x.detach().flatten().float()
    if percentile is None:
        return flat.min().item(), flat.max().item()
    if flat.numel() > max_samples:
        idx = torch.randint(0, flat.numel(), (max_samples,), device=flat.device)
        flat = flat[idx]
    lo = torch.quantile(flat, 1.0 - percentile).item()
    hi = torch.quantile(flat, percentile).item()
    return lo, hi
