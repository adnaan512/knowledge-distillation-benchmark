"""
Response-based (soft-label) knowledge distillation — Hinton et al. 2015.

THE CORE INSIGHT — "DARK KNOWLEDGE":
A neural network trained on CIFAR-10 doesn't just output "cat = 1.0".
It outputs something like "cat = 0.85, lynx-like = 0.08, dog = 0.04".
These non-zero probabilities for wrong classes are not noise — they
encode the model's learned structure of the world. The network has
discovered that cats and small dogs are more similar to each other
than either is to a truck.

A one-hot label of [0, 0, 1, 0, ...] throws all of that away.
Distillation preserves it by training the student to match the
teacher's full output distribution. This is called "dark knowledge"
because it was always in the predictions but invisible until Hinton
pointed at it.

TEMPERATURE SCALING:
At T=1 the teacher's distribution is peaked (confident). At high T,
the distribution becomes softer — more of the dark knowledge is
amplified into visible probability mass. But T too high makes the
signal useless (uniform noise). Empirically T ∈ [2, 8] is the
sweet spot for most classification tasks.

LOSS FORMULA:
    L = α * CE(student_logits, hard_labels)
      + (1 - α) * T² * KL(softmax(s/T) || softmax(t/T))

The T² scaling compensates for the gradient shrinkage that happens
when you divide logits by T before softmax — without it, the soft
label term contributes too little to the parameter updates.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from typing import List, Tuple, Dict

from src.data_models import DistillationResult


class ResponseBasedDistillation:
    """
    Trains a student model using soft-label distillation.

    Supports hyperparameter sweeps over temperature T and mixing
    coefficient alpha. The best configuration (by validation accuracy)
    is returned as the final result.
    """

    def __init__(
        self,
        teacher: nn.Module,
        student: nn.Module,
        device: torch.device,
        temperatures: List[float] = None,
        alphas: List[float] = None,
    ):
        self.teacher = teacher
        self.student = student
        self.device = device
        self.temperatures = temperatures or [1.0, 2.0, 4.0, 8.0, 16.0]
        self.alphas = alphas or [0.1, 0.5, 0.9]

        self.teacher.to(device).eval()
        self.student.to(device)

    def distillation_loss(
        self,
        student_logits: torch.Tensor,
        teacher_logits: torch.Tensor,
        labels: torch.Tensor,
        T: float,
        alpha: float,
    ) -> torch.Tensor:
        """
        Combined hard + soft label loss.

        Args:
            student_logits: Raw logits from student (B, num_classes)
            teacher_logits: Raw logits from teacher (B, num_classes)
            labels: Ground-truth class indices (B,)
            T: Temperature for softening distributions
            alpha: Weight for hard label loss (1-alpha for soft)

        Returns:
            Scalar loss tensor
        """
        # Hard label cross-entropy (standard supervised loss)
        hard_loss = F.cross_entropy(student_logits, labels)

        # Soft label KL divergence
        # log_softmax for numerical stability in KLDivLoss
        student_soft = F.log_softmax(student_logits / T, dim=1)
        teacher_soft = F.softmax(teacher_logits / T, dim=1)

        # KLDivLoss expects (log_probs, probs), reduction='batchmean'
        # divides by batch size — important for stable gradient scale
        soft_loss = F.kl_div(student_soft, teacher_soft, reduction="batchmean")

        # T^2 scaling restores gradient magnitude lost by dividing by T
        return alpha * hard_loss + (1 - alpha) * (T ** 2) * soft_loss

    def train_one_epoch(
        self,
        train_loader: DataLoader,
        optimizer: torch.optim.Optimizer,
        T: float,
        alpha: float,
    ) -> float:
        """Run one training epoch, return mean loss."""
        self.student.train()
        total_loss = 0.0
        num_batches = 0

        for images, labels in train_loader:
            images = images.to(self.device)
            labels = labels.to(self.device)

            with torch.no_grad():
                teacher_logits = self.teacher(images)

            student_logits = self.student(images)
            loss = self.distillation_loss(
                student_logits, teacher_logits, labels, T, alpha
            )

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            num_batches += 1

        return total_loss / max(num_batches, 1)

    @torch.no_grad()
    def evaluate(self, loader: DataLoader) -> float:
        """Return accuracy on the given loader."""
        self.student.eval()
        correct = total = 0
        for images, labels in loader:
            images, labels = images.to(self.device), labels.to(self.device)
            preds = self.student(images).argmax(dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)
        return correct / max(total, 1)

    def train(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
        num_epochs: int = 10,
        lr: float = 1e-3,
        T: float = 4.0,
        alpha: float = 0.5,
        verbose: bool = True,
    ) -> Tuple[float, List[float], List[float]]:
        """
        Train for num_epochs with fixed T and alpha.

        Returns:
            (best_val_accuracy, train_loss_history, val_accuracy_history)
        """
        # Only optimize trainable parameters (classifier head)
        trainable = [p for p in self.student.parameters() if p.requires_grad]
        optimizer = torch.optim.Adam(trainable, lr=lr)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=num_epochs
        )

        best_val_acc = 0.0
        train_losses = []
        val_accs = []

        for epoch in range(1, num_epochs + 1):
            loss = self.train_one_epoch(train_loader, optimizer, T, alpha)
            val_acc = self.evaluate(val_loader)
            scheduler.step()

            train_losses.append(loss)
            val_accs.append(val_acc)
            best_val_acc = max(best_val_acc, val_acc)

            if verbose:
                print(
                    f"  [Response T={T} α={alpha}] "
                    f"Epoch {epoch:2d}/{num_epochs} | "
                    f"Loss: {loss:.4f} | Val Acc: {val_acc*100:.1f}%"
                )

        return best_val_acc, train_losses, val_accs

    def sweep(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
        test_loader: DataLoader,
        num_epochs: int = 10,
        variant: str = "full",
        verbose: bool = True,
    ) -> Tuple[DistillationResult, Dict[Tuple[float, float], float]]:
        """
        Grid search over temperatures × alphas.

        Returns the best DistillationResult and a full sweep table
        (T, alpha) → val_accuracy for analysis.

        The sweep doubles as the temperature sensitivity analysis
        for RQ2: does optimal T generalize across compression ratios?
        """
        sweep_results: Dict[Tuple[float, float], float] = {}
        best_val_acc = 0.0
        best_config = {"T": 4.0, "alpha": 0.5}
        best_history = ([], [])

        for T in self.temperatures:
            for alpha in self.alphas:
                if verbose:
                    print(f"\n[Sweep] T={T}, alpha={alpha}")

                # Re-initialize student classifier weights for fair comparison
                if hasattr(self.student, "backbone"):
                    for layer in self.student.backbone.classifier:
                        if hasattr(layer, "reset_parameters"):
                            layer.reset_parameters()
                elif hasattr(self.student, "classifier"):
                    for layer in self.student.classifier:
                        if hasattr(layer, "reset_parameters"):
                            layer.reset_parameters()

                val_acc, losses, accs = self.train(
                    train_loader, val_loader, num_epochs, T=T, alpha=alpha,
                    verbose=verbose
                )
                sweep_results[(T, alpha)] = val_acc

                if val_acc > best_val_acc:
                    best_val_acc = val_acc
                    best_config = {"T": T, "alpha": alpha}
                    best_history = (losses, accs)

        # Final test-set evaluation with best config
        test_acc = self.evaluate(test_loader)

        result = DistillationResult(
            method="response",
            compression_variant=variant,
            final_accuracy=test_acc,
            train_loss_history=best_history[0],
            val_accuracy_history=best_history[1],
            hyperparams=best_config,
        )
        return result, sweep_results

    def run(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
        test_loader: DataLoader,
        num_epochs: int = 10,
        T: float = 4.0,
        alpha: float = 0.5,
        variant: str = "full",
        verbose: bool = True,
    ) -> DistillationResult:
        """Single training run with fixed hyperparameters."""
        _, train_losses, val_accs = self.train(
            train_loader, val_loader, num_epochs, T=T, alpha=alpha,
            verbose=verbose
        )
        test_acc = self.evaluate(test_loader)
        return DistillationResult(
            method="response",
            compression_variant=variant,
            final_accuracy=test_acc,
            train_loss_history=train_losses,
            val_accuracy_history=val_accs,
            hyperparams={"T": T, "alpha": alpha},
        )
