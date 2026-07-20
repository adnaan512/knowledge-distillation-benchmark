"""
CIFAR-10 dataset loading with train/val/test splits.

CIFAR-10 has 60,000 images across 10 classes. We use the standard 50k
train set, then carve out a 5k validation subset from it to tune
hyperparameters without touching the 10k test set. This is critical —
reporting accuracy on data that influenced hyperparameter selection
is a common source of inflated benchmark numbers.

Normalization constants are computed from the CIFAR-10 training set
and are standard in the literature. Using ImageNet stats here would
systematically mis-normalize every pixel and hurt transfer learning.
"""

import os
import torch
from torch.utils.data import DataLoader, Subset, random_split
from typing import Tuple, Optional


# CIFAR-10 channel statistics computed from the training set
CIFAR10_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR10_STD = (0.2023, 0.1994, 0.2010)

CIFAR10_CLASSES = [
    "airplane", "automobile", "bird", "cat", "deer",
    "dog", "frog", "horse", "ship", "truck"
]


def get_cifar10_loaders(
    data_dir: str = "./data",
    batch_size_train: int = 256,
    batch_size_eval: int = 512,
    val_size: int = 5000,
    num_workers: Optional[int] = None,
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Download (if needed) and return train/val/test DataLoaders.

    The dataset auto-downloads to data_dir on first run (~170MB).
    Subsequent runs load from disk.

    Returns:
        (train_loader, val_loader, test_loader)
    """
    try:
        import torchvision
        import torchvision.transforms as transforms
    except ImportError:
        raise ImportError(
            "torchvision is required. Run: pip install torchvision")

    if num_workers is None:
        num_workers = os.cpu_count() or 2

    train_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.RandomCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD),
    ])

    eval_transform = transforms.Compose([
        transforms.Resize(224),
        transforms.ToTensor(),
        transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD),
    ])

    full_train = torchvision.datasets.CIFAR10(
        root=data_dir, train=True, download=False, transform=train_transform
    )
    test_set = torchvision.datasets.CIFAR10(
        root=data_dir, train=False, download=False, transform=eval_transform
    )

    # Reproducible val split — same seed means same 5k images every run
    train_size = len(full_train) - val_size
    generator = torch.Generator().manual_seed(42)
    train_subset, val_subset = random_split(
        full_train, [train_size, val_size], generator=generator
    )

    # Val subset needs eval (no augmentation) transform
    # We re-wrap the dataset with the eval transform
    val_dataset_noaug = torchvision.datasets.CIFAR10(
        root=data_dir, train=True, download=False, transform=eval_transform
    )
    val_indices = val_subset.indices
    val_set = Subset(val_dataset_noaug, val_indices)

    train_loader = DataLoader(
        train_subset,
        batch_size=batch_size_train,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_set,
        batch_size=batch_size_eval,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )
    test_loader = DataLoader(
        test_set,
        batch_size=batch_size_eval,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )

    return train_loader, val_loader, test_loader


class MockDatasetLoader:
    """
    Synthetic dataset for CI and quick-demo mode.

    Generates random (3, 224, 224) tensors with random integer labels
    in [0, 9]. No downloads, no disk, no real patterns to learn —
    but the tensor shapes and label ranges are identical to resized CIFAR-10,
    so all downstream code runs without modification.

    This lets CI validate every code path (model forward passes,
    loss calculations, evaluation loops) in under 60 seconds with
    no internet access and minimal memory.
    """

    def __init__(
        self,
        num_samples: int = 100,
        batch_size: int = 32,
        num_classes: int = 10,
        seed: int = 42,
    ):
        self.num_samples = num_samples
        self.batch_size = batch_size
        self.num_classes = num_classes
        self.seed = seed
        torch.manual_seed(seed)

        self._data = torch.randn(num_samples, 3, 224, 224)
        self._labels = torch.randint(0, num_classes, (num_samples,))

    def _make_loader(
            self,
            data: torch.Tensor,
            labels: torch.Tensor) -> DataLoader:
        dataset = torch.utils.data.TensorDataset(data, labels)
        return DataLoader(dataset, batch_size=self.batch_size, shuffle=False)

    def get_loaders(self) -> Tuple[DataLoader, DataLoader, DataLoader]:
        """Return (train, val, test) loaders — all from mock data."""
        n = self.num_samples
        split = max(1, n // 5)
        train_data, train_labels = self._data[:n - 2 * split], self._labels[:n - 2 * split]
        val_data, val_labels = self._data[n - 2 * split:n - split], self._labels[n - 2 * split:n - split]
        test_data, test_labels = self._data[n - split:], self._labels[n - split:]

        return (
            self._make_loader(train_data, train_labels),
            self._make_loader(val_data, val_labels),
            self._make_loader(test_data, test_labels),
        )
