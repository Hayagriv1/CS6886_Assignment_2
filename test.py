"""
Compression + evaluation entry point.

    python test.py --weight_quant_bits 8 --activation_quant_bits 8
    python test.py --weight_quant_bits 5 --activation_quant_bits 6 \
                   --per_channel --protect_sensitive --wandb

Pipeline: load FP32 checkpoint -> fold BN -> wrap Conv/Linear ->
calibrate activation ranges on training data -> evaluate -> account size.
"""

import argparse, json, os, random
import numpy as np
import torch
import torch.nn as nn

from data import get_loaders
from evaluate import build_model, top1_accuracy
from compression.bn_fold import fold_bn_in_model, count_bn_values
from compression.quant_model import quantize_model, calibrate, set_enabled
from compression.size_accounting import (
    fp32_baseline_bits, compressed_bits, activation_footprint, summarize)


def set_seed(seed):
    random.seed(seed); np.random.seed(seed)
    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--weight_quant_bits", type=int, default=8)
    p.add_argument("--activation_quant_bits", type=int, default=8)
    p.add_argument("--checkpoint", default="./checkpoints/best_mobilenetv2_cifar10.pth")
    p.add_argument("--data_dir", default="./cifar-10-batches-py")
    p.add_argument("--per_channel", action="store_true", default=True)
    p.add_argument("--per_tensor", dest="per_channel", action="store_false")
    p.add_argument("--no_fold_bn", dest="fold_bn", action="store_false", default=True)
    p.add_argument("--no_act_quant", dest="quantize_act", action="store_false", default=True)
    p.add_argument("--protect_sensitive", action="store_true", default=True)
    p.add_argument("--percentile", type=float, default=0.999)
    p.add_argument("--calib_batches", type=int, default=8)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--wandb", action="store_true")
    args = p.parse_args()

    if not os.path.isdir(args.data_dir):
        raise SystemExit(
            f"CIFAR-10 not found at {args.data_dir}\n"
            "Download it with:\n"
            "  curl -O https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz\n"
            "  tar -xzf cifar-10-python.tar.gz\n"
            "or pass --data_dir <path to cifar-10-batches-py>")
    if not os.path.isfile(args.checkpoint):
        raise SystemExit(
            f"Checkpoint not found at {args.checkpoint}\n"
            "Train one with:  python train.py --data_dir " + args.data_dir)

    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cpu":
        print("No CUDA device found; running on CPU (slower but correct).")
    train_loader, test_loader = get_loaders(args.data_dir)

    model = build_model(args.checkpoint, device)
    fp32_bits = fp32_baseline_bits(model)
    bn_values = count_bn_values(model)

    fp32_acc = top1_accuracy(model, test_loader, device)
    print(f"FP32 baseline accuracy : {fp32_acc:.2f}%")

    if args.fold_bn:
        model, n_folded = fold_bn_in_model(model)
        print(f"Folded {n_folded} Conv-BN pairs "
              f"({bn_values:,} float values eliminated)")
        print(f"Post-fold FP32 accuracy: "
              f"{top1_accuracy(model, test_loader, device):.2f}%  (sanity check)")

    model, wrappers = quantize_model(
        model, args.weight_quant_bits, args.activation_quant_bits,
        per_channel=args.per_channel, quantize_act=args.quantize_act,
        protect_sensitive=args.protect_sensitive, percentile=args.percentile)

    if args.quantize_act:
        calibrate(model, wrappers, train_loader, device, args.calib_batches)

    quant_acc = top1_accuracy(model, test_loader, device)

    total_bits, breakdown, rows = compressed_bits(
        wrappers, args.per_channel)
    act_stats = activation_footprint(wrappers, args.activation_quant_bits)
    summary = summarize(fp32_bits, total_bits, breakdown, act_stats)

    print(f"\n{'='*58}")
    print(f"  w{args.weight_quant_bits} / a{args.activation_quant_bits}")
    print(f"{'='*58}")
    print(f"  FP32 accuracy            : {fp32_acc:.2f}%")
    print(f"  Quantized accuracy       : {quant_acc:.2f}%")
    print(f"  Accuracy drop            : {fp32_acc - quant_acc:.2f} pts")
    print(f"  FP32 size                : {summary['fp32_size_MB']:.3f} MB")
    print(f"  Compressed size          : {summary['compressed_size_MB']:.3f} MB")
    print(f"  Model compression ratio  : {summary['model_compression_ratio']:.2f}x")
    print(f"  Weight-only ratio        : {summary['weight_only_ratio']:.2f}x")
    print(f"  Activation ratio (total) : {act_stats.get('total_ratio', 0):.2f}x")
    print(f"  Metadata overhead        : {100*summary['overhead_fraction']:.2f}%")
    print("\n  Storage breakdown (KB):")
    for k, v in breakdown.items():
        if k != "total_bits":
            print(f"    {k:26s} {v/8192:10.2f}")

    result = dict(vars(args), fp32_acc=fp32_acc, quantized_acc=quant_acc,
                  **summary, **{f"act_{k}": v for k, v in act_stats.items()})
    tag = f"w{args.weight_quant_bits}a{args.activation_quant_bits}"
    with open(f"results_{tag}.json", "w") as f:
        json.dump(result, f, indent=2, default=str)

    if args.wandb:
        import wandb
        wandb.init(project="cs6886-mobilenetv2-compression",
                   name=tag, config=vars(args))
        wandb.log({k: v for k, v in result.items()
                   if isinstance(v, (int, float))})
        wandb.finish()


if __name__ == "__main__":
    main()
