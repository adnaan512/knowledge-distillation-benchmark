"""
Relation-based knowledge distillation — Park et al. 2019 (RKD).

WHY RELATIONAL KNOWLEDGE?
Both response-based and feature-based distillation transfer *point-wise*
knowledge: "for this input, produce this output" or "for this input,
produce these features." Relational distillation transfers *structural*
knowledge: "these two inputs should be more similar to each other than
either is to this third input."

This is analogous to how humans teach abstract reasoning — not by
memorizing facts but by learning relationships. The teacher's embedding
space encodes a rich geometry of class similarities. A student that
learns to reproduce this geometry, even at a compressed scale, will
generalize better to edge cases and distribution shifts.

IMPLEMENTATION — PAIRWISE DISTANCE MATCHING:
For each batch:
1. Compute pairwise L2 distances between all embeddings → distance matrix
2. Normalize to [0, 1] for scale invariance
3. Student minimizes MSE to teacher's normalized distance matrix

This is lightweight: no adapter layer needed, no channel matching.
The student just needs to produce embeddings that have the same
relative distances as the teacher's embeddings.

WHEN DOES IT UNDERPERFORM?
Relational distillation captures global structure but can miss
fine-grained per-class signal. When compression is severe (quarter
variant) and the student has too few parameters to reproduce the
teacher's embedding geometry faithfully, response-based distillation's
direct probability supervision often wins. This is the counter-intuitive
result in our benchmark: a simpler method outperforms a more elegant one
because the student capacity is the bottleneck, not the knowledge signal.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from typing import List, Tuple

from src.data_models import DistillationResult


class RelationBasedDistillation:
    """
    Trains a student to reproduce the teacher's pairwise embedding geometry.

    The student is trained with:
        L = CE(student_logits, labels) + λ * MSE(student_dist_matrix, teacher_dist_matrix)

    where dist_matrix[i,j] = normalized L2 distance between embeddings i and j.
    """

    def __init__(
        self,
        teacher: nn.Module,
        student: nn.Module,
        device: torch.device,
        lam: float = 0.5,
    ):
        self.teacher = teacher
        self.student = student
        self.device = device
        self.lam = lam

        self.teacher.to(device).eval()
        self.student.to(device)

    def _pairwise_distances(self, embeddings: torch.Tensor) -> torch.Tensor:
        """
        Compute normalized pairwise L2 distance matrix.

        Args:
            embeddings: (B, D) tensor of feature vectors

        Returns:
            (B, B) matrix where entry [i,j] = normalized distance between i and j
        """
        # Squared pairwise distances via expansion: ||a-b||^2 = ||a||^2 + ||b||^2 - 2*a^T*b
        sq_norm = (embeddings ** 2).sum(dim=1, keepdim=True)
        dist_sq = sq_norm + sq_norm.T - 2 * embeddings @ embeddings.T
        dist_sq = dist_sq.clamp(min=0.0)
        distances = dist_sq.sqrt()

        # Normalize to [0, 1] for scale invariance across teacher/student
        d_max = distances.max()
        if d_max > 1e-8:
            distances = distances / d_max

        return distances

    def _get_teacher_embeddings(self, images: torch.Tensor) -> torch.Tensor:
        """Get teacher penultimate embeddings (no gradient)."""
        with torch.no_grad():
            if hasattr(self.teacher, "get_features"):
                return self.teacher.get_features(images)
            # Fallback: pool the feature map from layer 3
            feats = self.teacher.get_features(images, layer=3)
            return F.adaptive_avg_pool2d(feats, (1, 1)).flatten(1)

    def _get_student_embeddings_and_logits(
        self, images: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Get student embeddings and logits in a single forward pass."""
        if hasattr(self.student, "get_features"):
            # Run full forward for logits
            logits = self.student(images)
            embeddings = self.student.get_features(images)
        else:
            logits = self.student(images)
            embeddings = logits  # fallback for mock models
        return embeddings, logits

    def relation_loss(
        self,
        student_embeddings: torch.Tensor,
        teacher_embeddings: torch.Tensor,
    ) -> torch.Tensor:
        """MSE between student and teacher normalized distance matrices."""
        student_dists = self._pairwise_distances(student_embeddings)
        teacher_dists = self._pairwise_distances(teacher_embeddings)
        return F.mse_loss(student_dists, teacher_dists.detach())

    def train_one_epoch(
        self,
        train_loader: DataLoader,
        optimizer: torch.optim.Optimizer,
    ) -> Tuple[float, float]:
        """Returns (mean_total_loss, mean_relation_loss)."""
        self.student.train()
        total_loss_sum = rel_loss_sum = 0.0
        num_batches = 0

        for images, labels in train_loader:
            images = images.to(self.device)
            labels = labels.to(self.device)

            teacher_embs = self._get_teacher_embeddings(images)
            student_embs, student_logits = self._get_student_embeddings_and_logits(images)

            ce_loss = F.cross_entropy(student_logits, labels)
            r_loss = self.relation_loss(student_embs, teacher_embs)
            loss = ce_loss + self.lam * r_loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss_sum += loss.item()
            rel_loss_sum += r_loss.item()
            num_batches += 1

        n = max(num_batches, 1)
        return total_loss_sum / n, rel_loss_sum / n

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
        """Full training run with relation-based distillation."""
        trainable = [p for p in self.student.parameters() if p.requires_grad]
        optimizer = torch.optim.Adam(trainable, lr=lr)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=num_epochs
        )

        train_losses = []
        val_accs = []
        best_val_acc = 0.0
        best_epoch = 0

        for epoch in range(1, num_epochs + 1):
            total_loss, rel_loss = self.train_one_epoch(train_loader, optimizer)
            val_acc = self.evaluate(val_loader)
            scheduler.step()

            train_losses.append(total_loss)
            val_accs.append(val_acc)

            if val_acc > best_val_acc:
                best_val_acc = val_acc
                best_epoch = epoch

            if verbose:
                print(
                    f"  [Relation λ={self.lam}] "
                    f"Epoch {epoch:2d}/{num_epochs} | "
                    f"Loss: {total_loss:.4f} (rel: {rel_loss:.4f}) | "
                    f"Val Acc: {val_acc*100:.1f}%"
                )

        test_acc = self.evaluate(test_loader)

        return DistillationResult(
            method="relation",
            compression_variant=variant,
            final_accuracy=test_acc,
            train_loss_history=train_losses,
            val_accuracy_history=val_accs,
            best_epoch=best_epoch,
            hyperparams={"lambda": self.lam},
        )
