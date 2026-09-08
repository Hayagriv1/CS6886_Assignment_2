"""
Storage accounting.

Every quantized model carries metadata that a naive "params x bits / 8"
estimate ignores. We account for all of it explicitly:

  WEIGHT PAYLOAD   n_weights x b_w bits
  WEIGHT SCALES    one fp16 per output channel (per-channel mode)
                   or one fp16 per tensor (per-tensor mode)
  FOLDED BIAS      one fp32 per output channel. Biases are kept in float:
                   they are ~0.8% of the parameter count but sit on the
                   accumulator path, where quantization error is not
                   attenuated by any subsequent normalisation.
  ACT QPARAMS      one fp32 scale + one int32 zero point per quantized
                   tensor (negligible, but counted)

The FP32 baseline counts parameters AND BatchNorm running statistics,
because both must be shipped to reproduce the float model.
"""

import torch
import torch.nn as nn


BITS_PER_MB = 8 * 1024 * 1024

SCALE_BITS = 16     # fp16 is ample for a scale factor
BIAS_BITS = 32
ZP_BITS = 32
ACT_SCALE_BITS = 32


def fp32_baseline_bits(model):
    """Bits needed for the uncompressed float model (params + BN buffers)."""
    total = sum(p.numel() for p in model.parameters()) * 32
    for m in model.modules():
        if isinstance(m, nn.BatchNorm2d):
            total += (m.running_mean.numel() + m.running_var.numel()) * 32
    return total


def compressed_bits(wrappers, per_channel=True):
    """Total stored bits for the quantized model, with a per-layer breakdown."""
    payload = scales = biases = act_meta = 0
    rows = []

    for w in wrappers:
        weight = w.module.weight
        n = weight.numel()
        c_out = weight.shape[0]

        p_bits = n * w.w_bits

        s_bits = (c_out if per_channel else 1) * SCALE_BITS
        b_bits = (w.module.bias.numel() * BIAS_BITS
                  if w.module.bias is not None else 0)
        a_bits_meta = ACT_SCALE_BITS + ZP_BITS if w.quantize_act else 0

        payload += p_bits
        scales += s_bits
        biases += b_bits
        act_meta += a_bits_meta

        rows.append({
            "layer": w.name, "n_weights": n, "w_bits": w.w_bits,
            "payload_KB": p_bits / 8192,
            "overhead_KB": (s_bits + b_bits + a_bits_meta) / 8192,
        })

    total = payload + scales + biases + act_meta
    breakdown = {
        "weight_payload_bits": payload,
        "scale_factor_bits": scales,
        "folded_bias_bits": biases,
        "activation_qparam_bits": act_meta,
        "total_bits": total,
    }
    return total, breakdown, rows


def activation_footprint(wrappers, a_bits):
    """Activation cost, measured as the tensors crossing layer boundaries.

    We report two figures because they answer different questions:

      total : sum over all quantized layer inputs for batch size 1. This is
              the total activation traffic of one forward pass.
      peak  : the largest single such tensor, i.e. the minimum activation
              buffer a layer-by-layer executor must hold.

    Activation metadata is one scale + one zero point per tensor, which is
    why the activation ratio is effectively 32 / a_bits.
    """
    sizes = [w.act_numel for w in wrappers if w.quantize_act and w.act_numel]
    if not sizes:
        return {}
    total, peak = sum(sizes), max(sizes)
    meta = len(sizes) * (ACT_SCALE_BITS + ZP_BITS)
    return {
        "n_tensors": len(sizes),
        "total_elements": total,
        "peak_elements": peak,
        "fp32_total_KB": total * 32 / 8192,
        "quant_total_KB": (total * a_bits + meta) / 8192,
        "fp32_peak_KB": peak * 32 / 8192,
        "quant_peak_KB": peak * a_bits / 8192,
        "total_ratio": (total * 32) / (total * a_bits + meta),
        "peak_ratio": 32.0 / a_bits,
    }


def summarize(model_fp32_bits, total_bits, breakdown, act_stats):
    """Headline numbers for the report."""
    return {
        "fp32_size_MB": model_fp32_bits / BITS_PER_MB,
        "compressed_size_MB": total_bits / BITS_PER_MB,
        "model_compression_ratio": model_fp32_bits / total_bits,
        "weight_only_ratio": (
            model_fp32_bits / breakdown["weight_payload_bits"]),
        "overhead_fraction": (
            (total_bits - breakdown["weight_payload_bits"]) / total_bits),
        "activation_compression_ratio": act_stats.get("total_ratio"),
    }
