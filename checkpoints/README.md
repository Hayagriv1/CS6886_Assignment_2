`best_mobilenetv2_cifar10.pth` — MobileNet-v2 trained from scratch on
CIFAR-10 for 50 epochs (seed 42), 94.28% test top-1. This is the exact
checkpoint all compression results in the report are measured against.

Regenerate with:

    python train.py --data_dir ./data/cifar-10-batches-py --epochs 50 --seed 42
