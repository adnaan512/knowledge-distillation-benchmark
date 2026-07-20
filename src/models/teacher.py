"""
ResNet-50 teacher model for knowledge distillation.

Why freeze the teacher entirely? Because the teacher is a fixed oracle —
its job is to supply soft probability distributions and intermediate
feature maps to guide the student. If we fine-tune the teacher during
student training, we create a moving target: the student's gradient
signal keeps shifting, making convergence unstable. We treat the teacher
as a lookup table of knowledge, not a co-learner.

The only exception is the final FC layer (2048 → 10 for CIFAR-10),
which we fine-tune briefly to adapt from ImageNet's 1000 classes to
our 10-class task. Everything below that remains frozen — those lower
layers already encode edge detectors, texture filters, and shape
representations that transfer extremely well to CIFAR-10.
"""

import torch
import torch.nn as nn
from typing import Dict


class TeacherModel(nn.Module):
    """
    ResNet-50 with ImageNet weights, adapted for CIFAR-10.

    The model is frozen at construction except for the final classifier.
    Call get_features(x, layer) to extract intermediate representations
    for feature-based and relation-based distillation.
    """

    def __init__(self, num_classes: int = 10, pretrained: bool = True):
        super().__init__()
        self._feature_cache: Dict[str, torch.Tensor] = {}
        self._hooks = []

        try:
            import torchvision.models as tvm
            if pretrained:
                weights = tvm.ResNet50_Weights.IMAGENET1K_V1
                self.backbone = tvm.resnet50(weights=weights)
            else:
                self.backbone = tvm.resnet50(weights=None)
        except ImportError:
            raise ImportError(
                "torchvision required. Run: pip install torchvision")

        # Replace the ImageNet head (1000-class) with CIFAR-10 head
        in_features = self.backbone.fc.in_features  # 2048
        self.backbone.fc = nn.Linear(in_features, num_classes)

        # Freeze everything first, then selectively unfreeze
        self._freeze_all()
        self._unfreeze_head()

    def _freeze_all(self):
        """Freeze all parameters — teacher is an oracle, not a learner."""
        for param in self.backbone.parameters():
            param.requires_grad = False

    def _unfreeze_head(self):
        """Unfreeze only the final classifier for CIFAR-10 adaptation."""
        for param in self.backbone.fc.parameters():
            param.requires_grad = True

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.backbone(x)

    def fine_tune_head(
            self,
            train_loader,
            val_loader,
            device: torch.device,
            epochs: int = 3):
        import torch.optim as optim
        import torch.nn.functional as F

        # Only optimize the fully connected layer
        optimizer = optim.Adam(self.backbone.fc.parameters(), lr=1e-3)
        self.to(device)

        print(
            f"    Fine-tuning teacher classifier head for {epochs} epochs...")
        for epoch in range(epochs):
            self.train()
            total_loss = 0.0

            for images, labels in train_loader:
                images, labels = images.to(device), labels.to(device)
                optimizer.zero_grad()
                outputs = self(images)
                loss = F.cross_entropy(outputs, labels)
                loss.backward()
                optimizer.step()
                total_loss += loss.item()

            # Quick validation accuracy evaluation
            self.eval()
            correct = 0
            total = 0
            with torch.no_grad():
                for images, labels in val_loader:
                    images, labels = images.to(device), labels.to(device)
                    outputs = self(images)
                    _, preds = torch.max(outputs, 1)
                    correct += (preds == labels).sum().item()
                    total += labels.size(0)

            val_acc = correct / total * 100
            print(
                f"      Epoch {epoch+1}/{epochs} | Loss: {total_loss/len(train_loader):.4f} | Val Acc: {val_acc:.1f}%")

    def get_features(self, x: torch.Tensor, layer: int = 3) -> torch.Tensor:
        """
        Extract intermediate feature maps from a specified ResNet layer.

        ResNet-50 has 4 layer groups (layer1–layer4). Layer 3 is a
        good compromise: early enough to retain generalizable spatial
        patterns, late enough to have learned task-relevant semantics.
        Layer 4 is too task-specific; layer 1 is too generic.

        Args:
            x: Input tensor of shape (B, 3, H, W)
            layer: Which ResNet layer group to tap (1–4)

        Returns:
            Feature tensor of shape (B, C, H', W') where C and H' depend
            on the layer chosen.
        """
        assert 1 <= layer <= 4, f"Layer must be in [1,4], got {layer}"

        features = {}

        def make_hook(name):
            def hook(module, input, output):
                features[name] = output
            return hook

        layer_module = getattr(self.backbone, f"layer{layer}")
        handle = layer_module.register_forward_hook(make_hook(f"layer{layer}"))

        with torch.no_grad():
            _ = self.backbone(x)

        handle.remove()
        return features[f"layer{layer}"]

    @property
    def feature_channels(self) -> Dict[int, int]:
        """Channel dimensions for each ResNet-50 layer group."""
        return {1: 256, 2: 512, 3: 1024, 4: 2048}

    def num_trainable_params(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def num_total_params(self) -> int:
        return sum(p.numel() for p in self.parameters())


class MockTeacherModel(nn.Module):
    """
    Minimal 2-layer teacher for CI and unit tests.

    Uses random weights — no pretrained download needed. Exposes the
    same API as TeacherModel so all downstream code is exercised
    without touching the internet.
    """

    FEATURE_CHANNELS = 64

    def __init__(self, num_classes: int = 10, input_channels: int = 3):
        super().__init__()
        self.conv = nn.Conv2d(
            input_channels,
            self.FEATURE_CHANNELS,
            3,
            padding=1)
        self.pool = nn.AdaptiveAvgPool2d((4, 4))
        self.fc = nn.Linear(self.FEATURE_CHANNELS * 16, num_classes)

        # All frozen — mock teacher doesn't train
        for p in self.parameters():
            p.requires_grad = False
        for p in self.fc.parameters():
            p.requires_grad = True

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = torch.relu(self.conv(x))
        x = self.pool(x)
        x = x.flatten(1)
        return self.fc(x)

    def get_features(self, x: torch.Tensor, layer: int = 3) -> torch.Tensor:
        """Returns conv features — same API as TeacherModel."""
        with torch.no_grad():
            feats = torch.relu(self.conv(x))
        return feats

    @property
    def feature_channels(self) -> Dict[int, int]:
        return {1: self.FEATURE_CHANNELS, 2: self.FEATURE_CHANNELS,
                3: self.FEATURE_CHANNELS, 4: self.FEATURE_CHANNELS}

    def num_total_params(self) -> int:
        return sum(p.numel() for p in self.parameters())
