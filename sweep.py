"""
Question 3: compression sweep.

    python sweep.py --data_dir <path> --checkpoint best_mobilenetv2_cifar10.pth --wandb

Runs the bit-width grid plus four ablations, logging each configuration as a
separate wandb run so the parallel-coordinates chart can be built from the
resulting table. Use --quick for a two-point smoke test.
"""

import argparse, itertools, json, random, time

import numpy as np
import torch

from data import get_loaders
from evaluate import build_model, top1_accuracy
from compression.bn_fold import fold_bn_in_model
from compression.quant_model import quantize_model, calibrate
from compression.size_accounting import (
    fp32_baseline_bits, compressed_bits, activation_footprint, summarize)


def set_seed(seed=42):
    random.seed(seed); np.random.seed(seed)
    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# Weight bits move over a wider range than activation bits because weights
# dominate stored size while activations dominate accuracy loss.
FULL_GRID = {"w_bits": [8, 6, 5, 4, 3, 2], "a_bits": [8, 6, 4],
             "per_channel": [True], "quantize_act": [True], "protect": [True]}

QUICK_GRID = {"w_bits": [8, 5], "a_bits": [8], "per_channel": [True],
              "quantize_act": [True], "protect": [True]}

# Each ablation changes exactly one design choice, so any accuracy
# difference is attributable to that choice rather than to bit-width.
ABLATIONS = [
    dict(w_bits=4, a_bits=8, per_channel=False, quantize_act=True,  protect=True),
    dict(w_bits=4, a_bits=8, per_channel=True,  quantize_act=True,  protect=False),
    dict(w_bits=4, a_bits=8, per_channel=True,  quantize_act=False, protect=True),
    dict(w_bits=8, a_bits=8, per_channel=False, quantize_act=True,  protect=False),
]


def tag_of(cfg):
    return (f"w{cfg['w_bits']}a{cfg['a_bits'] if cfg['quantize_act'] else 32}"
            f"{'_pc' if cfg['per_channel'] else '_pt'}"
            f"{'' if cfg['protect'] else '_noprot'}")


def run_one(cfg, args, calib_loader, test_loader, device, fp32_acc, fp32_bits):
    set_seed(args.seed)
    model = build_model(args.checkpoint, device)
    model, _ = fold_bn_in_model(model)
    model, wraps = quantize_model(
        model, cfg["w_bits"], cfg["a_bits"],
        per_channel=cfg["per_channel"], quantize_act=cfg["quantize_act"],
        protect_sensitive=cfg["protect"], percentile=args.percentile)

    if cfg["quantize_act"]:
        calibrate(model, wraps, calib_loader, device, args.calib_batches)
    else:
        # Still record input tensor sizes so the activation accounting works.
        for w in wraps: w.calibrating = True
        with torch.no_grad():
            model(next(iter(calib_loader))[0].to(device))
        for w in wraps: w.calibrating = False

    acc = top1_accuracy(model, test_loader, device)
    total_bits, brk, _ = compressed_bits(wraps, cfg["per_channel"])
    act = activation_footprint(wraps, cfg["a_bits"] if cfg["quantize_act"] else 32)
    s = summarize(fp32_bits, total_bits, brk, act)

    del model
    torch.cuda.empty_cache()

    return {
        "config": tag_of(cfg),
        "weight_quant_bits": cfg["w_bits"],
        "activation_quant_bits": cfg["a_bits"] if cfg["quantize_act"] else 32,
        "per_channel": cfg["per_channel"],
        "fp32_acc": fp32_acc,
        "quantized_acc": acc,
        "accuracy_drop": fp32_acc - acc,
        "compression_ratio": s["model_compression_ratio"],
        "weight_compression_ratio": s["weight_only_ratio"],
        "activation_compression_ratio": act.get("total_ratio", 1.0),
        "model_size_mb": s["compressed_size_MB"],
        "overhead_pct": 100 * s["overhead_fraction"],
        "breakdown_KB": {k: v / 8192 for k, v in brk.items()
                         if k != "total_bits"},
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data_dir", default="./cifar-10-batches-py")
    p.add_argument("--checkpoint", default="./checkpoints/best_mobilenetv2_cifar10.pth")
    p.add_argument("--percentile", type=float, default=0.999)
    p.add_argument("--calib_batches", type=int, default=8)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--quick", action="store_true")
    p.add_argument("--wandb", action="store_true")
    p.add_argument("--project", default="cs6886-mobilenetv2-compression")
    args = p.parse_args()

    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    _, test_loader = get_loaders(args.data_dir)
    # Calibration uses the deterministic transform: it estimates the
    # activation ranges seen at inference, so it must not observe
    # augmented inputs.
    calib_loader, _ = get_loaders(args.data_dir, augment=False)

    base = build_model(args.checkpoint, device)
    fp32_acc = top1_accuracy(base, test_loader, device)
    fp32_bits = fp32_baseline_bits(base)
    print(f"FP32 baseline: {fp32_acc:.2f}%  "
          f"({fp32_bits/(8*1024*1024):.3f} MB)\n")
    del base

    grid = QUICK_GRID if args.quick else FULL_GRID
    configs = [dict(zip(grid.keys(), v))
               for v in itertools.product(*grid.values())]
    if not args.quick:
        configs += ABLATIONS

    if args.wandb:
        import wandb

    results = []
    for i, cfg in enumerate(configs, 1):
        t0 = time.time()
        r = run_one(cfg, args, calib_loader, test_loader, device,
                    fp32_acc, fp32_bits)
        results.append(r)
        print(f"[{i:2d}/{len(configs)}] {r['config']:16s} "
              f"acc {r['quantized_acc']:6.2f}%  "
              f"ratio {r['compression_ratio']:5.2f}x  "
              f"{r['model_size_mb']:.3f} MB  ({time.time()-t0:.0f}s)")

        if args.wandb:
            run = wandb.init(project=args.project, name=r["config"],
                             config=cfg, reinit=True)
            wandb.log({k: v for k, v in r.items()
                       if isinstance(v, (int, float, bool))})
            run.finish()

    with open("sweep_results.json", "w") as f:
        json.dump(results, f, indent=2)

    print(f"\n{'config':<17}{'w':>3}{'a':>4}{'acc':>8}{'drop':>7}"
          f"{'ratio':>8}{'MB':>8}")
    print("-" * 55)
    for r in sorted(results, key=lambda x: -x["compression_ratio"]):
        print(f"{r['config']:<17}{r['weight_quant_bits']:>3}"
              f"{r['activation_quant_bits']:>4}{r['quantized_acc']:>8.2f}"
              f"{r['accuracy_drop']:>7.2f}{r['compression_ratio']:>8.2f}"
              f"{r['model_size_mb']:>8.3f}")


if __name__ == "__main__":
    main()
