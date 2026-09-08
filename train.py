"""
Question 1: train the MobileNet-v2 baseline on CIFAR-10.

    python train.py --data_dir <path> --epochs 50 --seed 42

Writes best_mobilenetv2_cifar10.pth, history.json and training_curves.png
to --out_dir.
"""

import argparse, json, os, random, time

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.models as models
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from data import get_loaders


def set_seed(seed):
    """Full seed configuration. cudnn.deterministic trades some throughput
    for run-to-run reproducibility, which matters because the compression
    results in Questions 3-4 are stated against this exact checkpoint."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def build_model(device):
    """MobileNet-v2 adapted for 32x32 input.

    Stock MobileNet-v2 downsamples 32x, which on CIFAR-10 leaves a 1x1
    feature map. Setting the stem stride and the first expand-ratio-6
    depthwise stride to 1 reduces this to 8x, giving a 4x4 map. No layers or
    channel counts change, so the architecture stays comparable to the
    reference model.
    """
    m = models.mobilenet_v2(weights=None, num_classes=10,
                            width_mult=1.0, dropout=0.2)
    m.features[0][0].stride = (1, 1)
    m.features[2].conv[1][0].stride = (1, 1)
    return m.to(device)


def train_one_epoch(model, loader, optimizer, criterion, device):
    model.train()
    loss_sum = correct = total = 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()
        out = model(x)
        loss = criterion(out, y)
        loss.backward()
        optimizer.step()
        loss_sum += loss.item() * x.size(0)
        correct += out.argmax(1).eq(y).sum().item()
        total += y.size(0)
    return loss_sum / total, 100.0 * correct / total


@torch.no_grad()
def evaluate(model, loader, criterion, device):
    model.eval()
    loss_sum = correct = total = 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        out = model(x)
        loss_sum += criterion(out, y).item() * x.size(0)
        correct += out.argmax(1).eq(y).sum().item()
        total += y.size(0)
    return loss_sum / total, 100.0 * correct / total


def plot_curves(history, path):
    ep = range(1, len(history["train_loss"]) + 1)
    best = max(history["test_acc"])
    fig, ax = plt.subplots(1, 2, figsize=(12, 4.5))
    ax[0].plot(ep, history["train_loss"], label="Train")
    ax[0].plot(ep, history["test_loss"], label="Test")
    ax[0].axhline(0.50, ls="--", c="gray", lw=1, label="Label-smoothing floor")
    ax[0].set_xlabel("Epoch"); ax[0].set_ylabel("Cross-entropy loss")
    ax[0].set_title("Loss"); ax[0].legend(); ax[0].grid(alpha=0.3)
    ax[1].plot(ep, history["train_acc"], label="Train")
    ax[1].plot(ep, history["test_acc"], label="Test")
    ax[1].set_xlabel("Epoch"); ax[1].set_ylabel("Top-1 accuracy (%)")
    ax[1].set_title(f"Accuracy (best test: {best:.2f}%)")
    ax[1].legend(); ax[1].grid(alpha=0.3)
    plt.tight_layout(); plt.savefig(path, dpi=150, bbox_inches="tight")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data_dir", default="./cifar-10-batches-py")
    p.add_argument("--out_dir", default="./checkpoints")
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--lr", type=float, default=0.05)
    p.add_argument("--weight_decay", type=float, default=5e-4)
    p.add_argument("--momentum", type=float, default=0.9)
    p.add_argument("--warmup_epochs", type=int, default=5)
    p.add_argument("--batch_size", type=int, default=128)
    p.add_argument("--label_smoothing", type=float, default=0.1)
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    set_seed(args.seed)
    os.makedirs(args.out_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_loader, test_loader = get_loaders(args.data_dir,
                                            train_batch=args.batch_size)

    model = build_model(device)
    print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")

    train_criterion = nn.CrossEntropyLoss(label_smoothing=args.label_smoothing)
    # Plain CE for evaluation so the reported test loss is not floored by
    # the label-smoothing entropy and the curves stay interpretable.
    eval_criterion = nn.CrossEntropyLoss()

    optimizer = optim.SGD(model.parameters(), lr=args.lr,
                          momentum=args.momentum,
                          weight_decay=args.weight_decay, nesterov=True)
    warmup = optim.lr_scheduler.LinearLR(optimizer, start_factor=0.1,
                                         total_iters=args.warmup_epochs)
    cosine = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs - args.warmup_epochs)
    scheduler = optim.lr_scheduler.SequentialLR(
        optimizer, [warmup, cosine], milestones=[args.warmup_epochs])

    history = {k: [] for k in
               ("train_loss", "train_acc", "test_loss", "test_acc")}
    best_acc = 0.0
    ckpt = os.path.join(args.out_dir, "best_mobilenetv2_cifar10.pth")

    for epoch in range(args.epochs):
        t0 = time.time()
        lr_now = scheduler.get_last_lr()[0]
        tr_loss, tr_acc = train_one_epoch(model, train_loader, optimizer,
                                          train_criterion, device)
        te_loss, te_acc = evaluate(model, test_loader, eval_criterion, device)
        scheduler.step()

        for k, v in zip(history, (tr_loss, tr_acc, te_loss, te_acc)):
            history[k].append(v)
        if te_acc > best_acc:
            best_acc = te_acc
            torch.save(model.state_dict(), ckpt)

        print(f"Epoch [{epoch+1}/{args.epochs}] "
              f"Train {tr_loss:.4f}/{tr_acc:.2f}% | "
              f"Test {te_loss:.4f}/{te_acc:.2f}% | "
              f"LR {lr_now:.5f} | {time.time()-t0:.1f}s")

    with open(os.path.join(args.out_dir, "history.json"), "w") as f:
        json.dump(history, f, indent=2)
    plot_curves(history, os.path.join(args.out_dir, "training_curves.png"))

    print(f"\nFinal test top-1 : {history['test_acc'][-1]:.2f}%")
    print(f"Best test top-1  : {best_acc:.2f}%  (saved to {ckpt})")


if __name__ == "__main__":
    main()
