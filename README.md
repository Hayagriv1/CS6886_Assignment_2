# CS6886 Assignment 2 — MobileNet-v2 on CIFAR-10, Quantization-Based Compression

Training a MobileNet-v2 baseline on CIFAR-10 from scratch, then compressing it
with a quantization pipeline written from scratch (no `torch.quantization` or
any other compression library).

**Headline results**

| Metric | Value |
|---|---|
| FP32 baseline accuracy | **94.28%** |
| Quantized accuracy (w5/a6) | **93.75%** (−0.53 pts) |
| Model compression ratio | **6.12×** |
| Weight compression ratio | **6.57×** |
| Activation compression ratio | **5.33×** |
| Model size | **8.662 MB → 1.416 MB** |
| Metadata overhead | 6.93% of compressed model |

---

## Environment

Python 3.10, single NVIDIA P100 (Kaggle), CUDA 12.1.

```bash
pip install -r requirements.txt
```

```
torch==2.4.0        numpy==1.26.4       matplotlib==3.7.5
torchvision==0.19.0 Pillow==10.4.0      wandb==0.17.7
```

## Data

CIFAR-10 in the original pickled-batch format. `--data_dir` must point at the
directory holding `data_batch_1..5` and `test_batch`, not at an individual file.
It defaults to `./cifar-10-batches-py`.

The CIFAR-10 batches are committed at `cifar-10-batches-py/`, so no download
is needed. To fetch them fresh:

```bash
curl -O https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz
tar -xzf cifar-10-python.tar.gz     # -> ./cifar-10-batches-py
```

The trained checkpoint is committed at
`checkpoints/best_mobilenetv2_cifar10.pth` (8.5 MB), so the compression
results can be reproduced without retraining.

## Seed configuration

Every entry point calls `set_seed(42)`, which sets `random`, `numpy`,
`torch`, and `torch.cuda` seeds, and additionally sets
`cudnn.deterministic = True` and `cudnn.benchmark = False`. This trades some
throughput for run-to-run reproducibility, which matters because the
compression results are stated against one specific checkpoint. Override with
`--seed`.

---

## Reproducing the results

### 1. Train the baseline (Question 1)

```bash
python train.py --epochs 50 --seed 42
```

Writes `checkpoints/best_mobilenetv2_cifar10.pth`, `history.json`,
`training_curves.png`. Not needed if you use the committed checkpoint.

### 2. Single compression run (Question 2) 

```bash
python test.py --weight_quant_bits 8 --activation_quant_bits 8
```

Prints accuracy, compression ratios, and the full storage breakdown; writes
`results_w8a8.json`. 

### 3. Full sweep + wandb chart (Question 3)

```bash
python sweep.py --wandb
```

Writes `sweep_results.json` and logs one wandb run per configuration. For the
parallel-coordinates chart: project → Table view → Add Panel → Parallel
Coordinates, with axes `activation_quant_bits`, `weight_quant_bits`,
`compression_ratio`, `model_size_mb`, `quantized_acc`, coloured by
`quantized_acc`.

Add `--quick` for a two-configuration smoke test first.

### 4. Reported operating point (Question 4)

```bash
python test.py --weight_quant_bits 5 --activation_quant_bits 6
```

---

## Repository layout

```
├── data.py                        # CIFAR-10 dataset + transforms (shared)
├── cifar-10-batches-py/           # CIFAR-10 batches (committed)
├── checkpoints/                   # trained baseline (committed)
├── train.py                       # TRAINING      — Question 1
├── evaluate.py                    # EVALUATION    — accuracy, confusion matrix
├── test.py                        # COMPRESSION   — single configuration
├── sweep.py                       # COMPRESSION   — Question 3 sweep
├── requirements.txt
└── compression/
    ├── quantizers.py              # quantization primitives
    ├── bn_fold.py                 # BatchNorm folding
    ├── quant_model.py             # quantized wrappers, surgery, calibration
    └── size_accounting.py         # storage accounting with all overheads
```

Training, evaluation, and compression are separate modules. `evaluate.py` is
imported by both `test.py` and `sweep.py`, so FP32 and quantized accuracy are
measured by an identical code path.

## Method summary

- **Weights** — symmetric, per-output-channel, no zero point. Per-channel
  scaling is what makes sub-8-bit quantization viable on MobileNet-v2's
  depthwise convolutions: at 4 bits, switching to per-tensor costs 14.4
  accuracy points for a saving of 33 KB.
- **Activations** — asymmetric per-tensor, quantized at layer inputs so each
  intermediate tensor is quantized exactly once. Ranges calibrated at the
  99.9th percentile over 1024 *training* images; the test set is never used
  for calibration.
- **BatchNorm folding** — folded into the preceding convolution before
  quantization. Removes 68,224 float values (266.5 KB at FP32), more than all
  metadata combined, and matches deployment where BatchNorm is not a separate
  runtime operator.
- **Mixed precision** — stem convolution and classifier held at 8 bits. Under
  2% of parameters, worth 0.16 accuracy points, costs 0.04× in ratio.
- **Size accounting** — weight payload, FP16 per-channel scales, FP32 folded
  biases, and activation qparams are all counted. The FP32 baseline counts
  parameters *and* BatchNorm running statistics.

Accuracy is measured by fake quantization (quantize→dequantize in float),
which reproduces the exact numerical error of an integer kernel; stored size
is computed analytically. All results are post-training quantization with no
fine-tuning.
