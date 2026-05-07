"""
Feature-based knowledge distillation — Romero et al. 2015 (FitNets).

WHY INTERMEDIATE FEATURES?
The teacher's final layer has already made its decision — it has
compressed everything into a 10-dimensional output. The intermediate
layers, especially the middle layers, contain richer information:
spatial structure, multi-scale texture, part-level representations.

By forcing the student to mimic the teacher's layer3 output,
we are not just teaching it *what* the answer is, but *how to
think about* the problem at an intermediate level of abstraction.
The student learns to build similar internal representations, even
if its final decision pathway is compressed.

Later layers are task-specific (ImageNet-tuned, CIFAR-10-adapted)
and don't transfer cleanly. Early layers are too generic and provide
weak signal. Layer3 is the sweet spot: semantically meaningful
but not yet committed to a specific output.

THE ADAPTER PROBLEM:
Teacher's layer3 outputs 1024 channels; MobileNetV2's equivalent
layer might output 80 or 40 channels (depending on width_mult).
We can't compare them directly. A 1×1 convolution adapter projects
the student's features up to the teacher's channel dimension. This
adapter is trained alongside the student head — it learns the
optimal linear recombination of student channels to match the teacher.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from typing import List, Tuple, Dict

from src.data_models import DistillationResult


class FeatureAdapter(nn.Module):
    """
    1×1 conv that projects student features to teacher's channel dim.

    Trained alongside the student's classifier so it learns to
    express student feature space in teacher-compatible coordinates.
    """

    def __init__(self, student_channels: int, teacher_channels: int):
        super().__init__()
        self.proj = nn.Conv2d(student_channels, teacher_channels, kernel_size=1, bias=False)
        self.bn = nn.BatchNorm2d(teacher_channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.bn(self.proj(x))


class FeatureBasedDistillation:
    """
    Distillation via intermediate feature map matching.

    The student is trained to minimize:
        L = CE(student_logits, labels) + λ * MSE(adapted_student_feats, teacher_feats)

    where teacher_feats are spatially pooled to match student's feature map size.
    """

    def __init__(
        self,
        teacher: nn.Module,
        student: nn.Module,
        device: torch.device,
        teacher_layer: int = 3,
        lam: float = 0.5,
    ):
        self.teacher = teacher
        self.student = student
        self.device = device
        self.teacher_layer = teacher_layer
        self.lam = lam

        self.teacher.to(device).eval()
        self.student.to(device)

        # Determine channel dimensions for the adapter
        teacher_channels = teacher.feature_channels.get(teacher_layer, 1024)
        student_channels = student.intermediate_channels

        self.adapter = FeatureAdapter(student_channels, teacher_channels).to(device)

    def _get_teacher_features(self, images: torch.Tensor) -> torch.Tensor:
        """Extract teacher's layer3 feature maps (no grad needed)."""
        with torch.no_grad():
            feats = self.teacher.get_features(images, layer=self.teacher_layer)
        return feats

    def _get_student_features(self, images: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Extract student intermediate features and logits in one forward pass.

        Returns (intermediate_features, logits)
        """
        intermediate = {}

        # Tap into student's backbone at the intermediate block
        if hasattr(self.student, "backbone") and hasattr(self.student.backbone, "features"):
            # MobileNetV2 path
            block_idx = 14
            def hook_fn(module, input, output):
                intermediate["feat"] = output
            handle = self.student.backbone.features[block_idx].register_forward_hook(hook_fn)
            logits = self.student(images)
            handle.remove()
            feats = intermediate.get("feat", torch.zeros(images.size(0), self.student.intermediate_channels, 4, 4, device=self.device))
        else:
            # MockStudent path
            feats = self.student.get_intermediate_features(images)
            logits = self.student(images)

        return feats, logits

    def feature_loss(
        self,
        student_feats: torch.Tensor,
        teacher_feats: torch.Tensor,
    ) -> torch.Tensor:
        """
        MSE between adapted student features and teacher features.

        Spatial dimensions may differ — we pool teacher features down
        to student's spatial resolution for a fair comparison.
        """
        # Adapt student channels to teacher's channel space
        student_adapted = self.adapter(student_feats)

        # Pool teacher features to student's spatial size if needed
        if student_adapted.shape[2:] != teacher_feats.shape[2:]:
            teacher_feats = F.adaptive_avg_pool2d(
                teacher_feats, student_adapted.shape[2:]
            )

        return F.mse_loss(student_adapted, teacher_feats.detach())

    def train_one_epoch(
        self,
        train_loader: DataLoader,
        optimizer: torch.optim.Optimizer,
    ) -> Tuple[float, float]:
        """Returns (mean_total_loss, mean_feature_loss)."""
        self.student.train()
        self.adapter.train()
        total_loss_sum = feat_loss_sum = 0.0
        num_batches = 0

        for images, labels in train_loader:
            images = images.to(self.device)
            labels = labels.to(self.device)

            teacher_feats = self._get_teacher_features(images)
            student_feats, student_logits = self._get_student_features(images)

            ce_loss = F.cross_entropy(student_logits, labels)
            f_loss = self.feature_loss(student_feats, teacher_feats)
            loss = ce_loss + self.lam * f_loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss_sum += loss.item()
            feat_loss_sum += f_loss.item()
            num_batches += 1

        n = max(num_batches, 1)
        return total_loss_sum / n, feat_loss_sum / n

    @torch.no_grad()
    def evaluate(self, loader: DataLoader) -> float:
        self.student.eval()
        correct = total = 0
        for images, labels in loader:
            images, labels = images.to(self.device), labels.to(self.device)
            preds = self.student(images).argmax(dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)
        return correct / max(total, 1)

    def run(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
        test_loader: DataLoader,
        num_epochs: int = 10,
        lr: float = 1e-3,
        variant: str = "full",
        verbose: bool = True,
    ) -> DistillationResult:
        """Full training run with feature-based distillation."""
        # Include adapter parameters in optimizer
        params = (
            [p for p in self.student.parameters() if p.requires_grad]
            + list(self.adapter.parameters())
        )
        optimizer = torch.optim.Adam(params, lr=lr)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=num_epochs
        )

        train_losses = []
        val_accs = []
        best_val_acc = 0.0
        best_epoch = 0

        for epoch in range(1, num_epochs + 1):
            total_loss, feat_loss = self.train_one_epoch(train_loader, optimizer)
            val_acc = self.evaluate(val_loader)
            scheduler.step()

            train_losses.append(total_loss)
            val_accs.append(val_acc)

            if val_acc > best_val_acc:
                best_val_acc = val_acc
                best_epoch = epoch

            if verbose:
                print(
                    f"  [Feature λ={self.lam}] "
                    f"Epoch {epoch:2d}/{num_epochs} | "
                    f"Loss: {total_loss:.4f} (feat: {feat_loss:.4f}) | "
                    f"Val Acc: {val_acc*100:.1f}%"
                )

        test_acc = self.evaluate(test_loader)

        return DistillationResult(
            method="feature",
            compression_variant=variant,
            final_accuracy=test_acc,
            train_loss_history=train_losses,
            val_accuracy_history=val_accs,
            best_epoch=best_epoch,
            hyperparams={"lambda": self.lam, "teacher_layer": self.teacher_layer},
        )
