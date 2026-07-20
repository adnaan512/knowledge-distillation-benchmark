"""
Minimal mock models for fast CI testing.

These models are:
- Tiny (< 10k parameters)
- Random-initialized (no pretrained download)
- API-compatible with TeacherModel and StudentModel
- Fast enough that the full test suite runs in under 30 seconds

The principle: test the logic, not the weights. All distillation
loss computations, evaluation loops, and report generation are
exercised against these mocks.
"""

import torch
import torch.nn as nn
from typing import Dict


class TinyTeacher(nn.Module):
    """
    2-layer teacher for unit testing.

    Layer 1: Conv2d(3, 64, 3) → produces the "intermediate features"
    Layer 2: Linear(64*16, 10) → final classifier

    Mimics the TeacherModel API: forward(), get_features(), feature_channels.
    """

    FEAT_CHANNELS = 64

    def __init__(self, num_classes: int = 10, frozen: bool = True):
        super().__init__()
        self.conv = nn.Conv2d(3, self.FEAT_CHANNELS, kernel_size=3, padding=1)
        self.pool = nn.AdaptiveAvgPool2d((4, 4))
        self.fc = nn.Linear(self.FEAT_CHANNELS * 16, num_classes)

        if frozen:
            for p in self.parameters():
                p.requires_grad = False
            for p in self.fc.parameters():
                p.requires_grad = True

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = torch.relu(self.conv(x))
        x = self.pool(x)
        return self.fc(x.flatten(1))

    def get_features(self, x: torch.Tensor, layer: int = 3) -> torch.Tensor:
        """Returns conv feature maps — shape (B, 64, H, W)."""
        with torch.no_grad():
            return torch.relu(self.conv(x))

    @property
    def feature_channels(self) -> Dict[int, int]:
        return {1: self.FEAT_CHANNELS, 2: self.FEAT_CHANNELS,
                3: self.FEAT_CHANNELS, 4: self.FEAT_CHANNELS}

    def num_total_params(self) -> int:
        return sum(p.numel() for p in self.parameters())


class TinyStudent(nn.Module):
    """
    1-layer student for unit testing.

    Smaller than TinyTeacher to simulate the compression setting.
    Mimics the StudentModel API.
    """

    FEAT_CHANNELS = 32
    FEAT_DIM = 32  # penultimate embedding dim

    def __init__(
        self,
        variant: str = "full",
        num_classes: int = 10,
    ):
        super().__init__()
        self.variant = variant
        self.width_mult = {
            "full": 1.0,
            "half": 0.5,
            "quarter": 0.35}.get(
            variant,
            1.0)

        ch = max(1, int(self.FEAT_CHANNELS * self.width_mult))
        self.ch = ch

        self.conv = nn.Conv2d(3, ch, kernel_size=3, padding=1)
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Linear(ch, num_classes)

        # Freeze conv (simulates backbone freeze), keep classifier trainable
        for p in self.conv.parameters():
            p.requires_grad = False
        for p in self.classifier.parameters():
            p.requires_grad = True

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = torch.relu(self.conv(x))
        x = self.pool(x).flatten(1)
        return self.classifier(x)

    def get_features(self, x: torch.Tensor) -> torch.Tensor:
        """Penultimate embeddings — shape (B, ch)."""
        x = torch.relu(self.conv(x))
        return self.pool(x).flatten(1)

    def get_intermediate_features(
            self,
            x: torch.Tensor,
            block_idx: int = 0) -> torch.Tensor:
        """Intermediate feature maps — shape (B, ch, H, W)."""
        return torch.relu(self.conv(x))

    @property
    def intermediate_channels(self) -> int:
        return self.ch

    def num_total_params(self) -> int:
        return sum(p.numel() for p in self.parameters())


def make_mock_batch(
    batch_size: int = 8,
    num_classes: int = 10,
    seed: int = 0,
) -> tuple:
    """Return (images, labels) tensors with CIFAR-10-like shapes."""
    torch.manual_seed(seed)
    images = torch.randn(batch_size, 3, 32, 32)
    labels = torch.randint(0, num_classes, (batch_size,))
    return images, labels
