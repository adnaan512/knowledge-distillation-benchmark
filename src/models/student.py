"""
MobileNetV2 student models in three compression variants.

MobileNetV2's width multiplier is the knob we turn to trade accuracy
for speed. The multiplier uniformly scales the number of channels in
every layer — a multiplier of 0.5 roughly halves parameters and FLOPs,
while 0.35 compresses aggressively to under 1M parameters.

We freeze all layers except the final classifier for the same reason
as the teacher: we are not training from scratch. The ImageNet features
are already good — we just need the classifier to reroute them to
CIFAR-10's 10-class output space. Fine-tuning everything would take
hours on CPU and wouldn't meaningfully improve results on a 32x32 dataset.
"""

import torch
import torch.nn as nn
from typing import Dict, Tuple


# Named compression variants and their MobileNetV2 width multipliers
COMPRESSION_VARIANTS = {
    "full":    1.0,   # ~3.4M parameters, compression ratio ~7x vs ResNet-50
    "half":    0.5,   # ~1.4M parameters, compression ratio ~17x
    "quarter": 0.35,  # ~0.9M parameters, compression ratio ~26x
}


class StudentModel(nn.Module):
    """
    MobileNetV2 student with configurable width multiplier.

    Width multiplier uniformly scales channel counts across all layers.
    We use pretrained ImageNet weights even for the compressed variants —
    PyTorch's MobileNetV2 supports arbitrary width_mult at construction,
    and the closest available pretrained checkpoint is width_mult=1.0.
    For half and quarter, we initialize from scratch (the weights don't
    transfer cleanly across width changes), which is why distillation
    matters more at higher compression ratios.
    """

    def __init__(
        self,
        variant: str = "full",
        num_classes: int = 10,
        pretrained: bool = True,
    ):
        super().__init__()
        assert variant in COMPRESSION_VARIANTS, (
            f"variant must be one of {list(COMPRESSION_VARIANTS.keys())}"
        )
        self.variant = variant
        self.width_mult = COMPRESSION_VARIANTS[variant]

        try:
            import torchvision.models as tvm
        except ImportError:
            raise ImportError("torchvision required. Run: pip install torchvision")

        # Only width_mult=1.0 has an official pretrained checkpoint
        if pretrained and self.width_mult == 1.0:
            weights = tvm.MobileNet_V2_Weights.DEFAULT
            self.backbone = tvm.mobilenet_v2(weights=weights)
        else:
            # Build with custom width but no pretrained weights
            self.backbone = tvm.mobilenet_v2(
                weights=None,
                width_mult=self.width_mult,
            )

        # Determine the last channel dimension (varies with width_mult)
        last_channel = self.backbone.last_channel

        # Replace classifier head
        self.backbone.classifier = nn.Sequential(
            nn.Dropout(p=0.2),
            nn.Linear(last_channel, num_classes),
        )

        # Freeze backbone, keep only classifier trainable
        self._freeze_backbone()

    def _freeze_backbone(self):
        """
        Freeze all backbone layers, unfreeze classifier.

        This mirrors the teacher strategy: use pre-learned feature
        extractors as-is, adapt only the final decision layer.
        On CPU with a 32x32 dataset, this converges in ~10 epochs
        where full fine-tuning would need 50+ epochs.
        """
        for param in self.backbone.features.parameters():
            param.requires_grad = False
        for param in self.backbone.classifier.parameters():
            param.requires_grad = True

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.backbone(x)

    def get_features(self, x: torch.Tensor) -> torch.Tensor:
        """
        Extract penultimate features (before classifier).

        These are the embeddings used for relation-based distillation.
        Shape: (B, last_channel)
        """
        features = self.backbone.features(x)
        features = nn.functional.adaptive_avg_pool2d(features, (1, 1))
        return features.flatten(1)

    def get_intermediate_features(self, x: torch.Tensor, block_idx: int = 14) -> torch.Tensor:
        """
        Extract intermediate feature maps from a specific inverted residual block.

        MobileNetV2's features module has 19 sub-layers (0–18).
        Block 14 is roughly equivalent to ResNet's layer3 in semantic depth.

        Args:
            x: Input tensor (B, 3, H, W)
            block_idx: Index into backbone.features (0–18)

        Returns:
            Intermediate feature maps (B, C, H', W')
        """
        intermediate = {}

        def hook_fn(module, input, output):
            intermediate["feat"] = output

        handle = self.backbone.features[block_idx].register_forward_hook(hook_fn)
        _ = self.backbone.features(x)
        handle.remove()
        return intermediate["feat"]

    @property
    def intermediate_channels(self) -> int:
        """Channel count at block 14 — depends on width_mult."""
        base = 160  # for width_mult=1.0
        return max(1, int(base * self.width_mult))

    def num_trainable_params(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def num_total_params(self) -> int:
        return sum(p.numel() for p in self.parameters())


class MockStudentModel(nn.Module):
    """
    Minimal 1-layer student for CI and unit tests.

    Exposes the same API as StudentModel. No pretrained weights, no
    downloads — just random initialization and the correct tensor shapes.
    """

    FEATURE_DIM = 32

    def __init__(self, variant: str = "full", num_classes: int = 10):
        super().__init__()
        self.variant = variant
        self.width_mult = COMPRESSION_VARIANTS.get(variant, 1.0)

        self.conv = nn.Conv2d(3, self.FEATURE_DIM, 3, padding=1)
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Linear(self.FEATURE_DIM, num_classes)

        for p in self.conv.parameters():
            p.requires_grad = False
        for p in self.classifier.parameters():
            p.requires_grad = True

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = torch.relu(self.conv(x))
        x = self.pool(x).flatten(1)
        return self.classifier(x)

    def get_features(self, x: torch.Tensor) -> torch.Tensor:
        x = torch.relu(self.conv(x))
        return self.pool(x).flatten(1)

    def get_intermediate_features(self, x: torch.Tensor, block_idx: int = 14) -> torch.Tensor:
        return torch.relu(self.conv(x))

    @property
    def intermediate_channels(self) -> int:
        return self.FEATURE_DIM

    def num_total_params(self) -> int:
        return sum(p.numel() for p in self.parameters())
