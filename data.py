"""
CIFAR-10 data loading and augmentation.

Extracted from the Question 1 training script so that training, evaluation
and compression share exactly one definition of the data pipeline.
"""

import os
import pickle

import numpy as np
from PIL import Image
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader

CIFAR10_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR10_STD = (0.2470, 0.2435, 0.2616)

CLASSES = ['airplane', 'automobile', 'bird', 'cat', 'deer',
           'dog', 'frog', 'horse', 'ship', 'truck']


def unpickle(path):
    with open(path, 'rb') as fo:
        return pickle.load(fo, encoding='bytes')


class CIFAR10Local(Dataset):
    """CIFAR-10 read directly from the official pickled batch files."""

    def __init__(self, root, train=True, transform=None):
        self.transform = transform
        self.data, self.labels = [], []
        files = ([f'data_batch_{i}' for i in range(1, 6)] if train
                 else ['test_batch'])
        for fname in files:
            batch = unpickle(os.path.join(root, fname))
            self.data.append(batch[b'data'])
            self.labels += batch[b'labels']
        self.data = (np.vstack(self.data)
                     .reshape(-1, 3, 32, 32)
                     .transpose(0, 2, 3, 1))          # NCHW -> NHWC uint8

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img = Image.fromarray(self.data[idx])
        if self.transform:
            img = self.transform(img)
        return img, self.labels[idx]


# RandomErasing is placed AFTER Normalize so the erased region is filled
# with zeros in normalised space, i.e. the per-channel dataset mean in pixel
# space rather than black.
train_transform = transforms.Compose([
    transforms.RandomCrop(32, padding=4, padding_mode='reflect'),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.ToTensor(),
    transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD),
    transforms.RandomErasing(p=0.5, scale=(0.02, 0.2), ratio=(0.3, 3.3)),
])

test_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD),
])


def get_loaders(data_dir, train_batch=128, test_batch=256,
                num_workers=2, augment=True):
    """Return (train_loader, test_loader).

    augment=False gives the deterministic test transform on the training
    split, which is what calibration should use -- calibration is estimating
    the activation ranges seen at inference, so it must not see augmented
    inputs whose statistics differ from deployment.
    """
    train_set = CIFAR10Local(
        data_dir, train=True,
        transform=train_transform if augment else test_transform)
    test_set = CIFAR10Local(data_dir, train=False, transform=test_transform)

    train_loader = DataLoader(train_set, batch_size=train_batch, shuffle=True,
                              num_workers=num_workers, pin_memory=True)
    test_loader = DataLoader(test_set, batch_size=test_batch, shuffle=False,
                             num_workers=num_workers, pin_memory=True)
    return train_loader, test_loader
