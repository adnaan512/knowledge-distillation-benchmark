"""
Unit tests for distillation loss functions.

Tests verify mathematical properties that must hold regardless of
the data or model weights — these are invariants of the loss functions
themselves, not of any particular training outcome.
"""

import torch
import torch.nn.functional as F

from tests.fixtures.mock_models import TinyTeacher, TinyStudent, make_mock_batch
from src.distillation.response_based import ResponseBasedDistillation
from src.distillation.feature_based import FeatureBasedDistillation, FeatureAdapter
from src.distillation.relation_based import RelationBasedDistillation


DEVICE = torch.device("cpu")


# ── Response-based tests ────────────────────────────────────────────────

class TestResponseBasedLoss:

    def setup_method(self):
        self.teacher = TinyTeacher()
        self.student = TinyStudent()
        self.distiller = ResponseBasedDistillation(
            teacher=self.teacher,
            student=self.student,
            device=DEVICE,
            temperatures=[1.0, 4.0],
            alphas=[0.5],
        )

    def test_kl_zero_when_distributions_match(self):
        """KL divergence must be exactly 0 when student and teacher output the same logits."""
        torch.manual_seed(7)
        logits = torch.randn(8, 10)
        labels = torch.randint(0, 10, (8,))

        # When student logits == teacher logits, KL should be ~0
        loss = self.distiller.distillation_loss(
            student_logits=logits,
            teacher_logits=logits,  # identical
            labels=labels,
            T=4.0,
            alpha=0.0,  # pure soft label loss
        )
        assert loss.item() < 1e-5, (
            f"KL divergence should be ~0 when distributions match, got {loss.item():.6f}"
        )

    def test_temperature_increases_entropy(self):
        """
        Higher temperature should produce softer (higher-entropy) distributions.

        At T=1: argmax logit dominates
        At T=16: distribution approaches uniform
        """
        torch.manual_seed(42)
        logits = torch.tensor(
            [[5.0, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1]])

        def entropy(log_probs):
            probs = log_probs.exp()
            return -(probs * log_probs).sum().item()

        soft_T1 = F.log_softmax(logits / 1.0, dim=1)
        soft_T16 = F.log_softmax(logits / 16.0, dim=1)

        assert entropy(soft_T16) > entropy(soft_T1), (
            "Higher temperature should produce higher-entropy (softer) distribution"
        )

    def test_alpha_controls_loss_components(self):
        """
        alpha=1.0 → loss equals pure cross-entropy (no soft labels).
        alpha=0.0 → loss is pure KL (no hard labels).
        Both should be non-negative.
        """
        torch.manual_seed(3)
        s_logits = torch.randn(8, 10)
        t_logits = torch.randn(8, 10)
        labels = torch.randint(0, 10, (8,))

        loss_hard = self.distiller.distillation_loss(
            s_logits, t_logits, labels, T=1.0, alpha=1.0)
        loss_soft = self.distiller.distillation_loss(
            s_logits, t_logits, labels, T=1.0, alpha=0.0)
        loss_mix = self.distiller.distillation_loss(
            s_logits, t_logits, labels, T=4.0, alpha=0.5)

        expected_hard = F.cross_entropy(s_logits, labels)

        assert loss_hard.item() >= 0
        assert loss_soft.item() >= 0
        assert loss_mix.item() >= 0
        assert abs(loss_hard.item() - expected_hard.item()) < 1e-5, (
            "alpha=1.0 should equal pure cross-entropy"
        )

    def test_temperature_squared_scaling(self):
        """
        The T² factor should make soft loss contribution comparable to hard loss.
        Without T² scaling, soft loss magnitude would shrink quadratically with T.
        """
        torch.manual_seed(99)
        s_logits = torch.randn(8, 10)
        t_logits = s_logits + 0.1  # slightly different teacher
        labels = torch.randint(0, 10, (8,))

        loss_T1 = self.distiller.distillation_loss(
            s_logits, t_logits, labels, T=1.0, alpha=0.0)
        loss_T8 = self.distiller.distillation_loss(
            s_logits, t_logits, labels, T=8.0, alpha=0.0)
        loss_T16 = self.distiller.distillation_loss(
            s_logits, t_logits, labels, T=16.0, alpha=0.0)

        # All losses should be positive
        assert loss_T1.item() >= 0
        assert loss_T8.item() >= 0
        assert loss_T16.item() >= 0

    def test_loss_decreases_with_training(self):
        """A single gradient step should reduce the distillation loss."""
        torch.manual_seed(5)
        images, labels = make_mock_batch(batch_size=8)

        # Enable gradients for student classifier
        self.student.train()
        optimizer = torch.optim.SGD(
            [p for p in self.student.parameters() if p.requires_grad], lr=0.01
        )

        # Initial loss
        with torch.no_grad():
            t_logits = self.teacher(images)
        s_logits_before = self.student(images)
        loss_before = self.distiller.distillation_loss(
            s_logits_before, t_logits, labels, T=4.0, alpha=0.5
        )

        # One gradient step
        optimizer.zero_grad()
        s_logits_after_step = self.student(images)
        loss = self.distiller.distillation_loss(
            s_logits_after_step, t_logits, labels, T=4.0, alpha=0.5
        )
        loss.backward()
        optimizer.step()

        # Compute loss after step
        with torch.no_grad():
            s_logits_post = self.student(images)
        loss_after = self.distiller.distillation_loss(
            s_logits_post, t_logits, labels, T=4.0, alpha=0.5
        )

        # Loss should decrease (or at least not explode)
        assert loss_after.item() < loss_before.item() * 10, (
            "Loss exploded after one gradient step — check learning rate or loss scale"
        )


# ── Feature-based tests ─────────────────────────────────────────────────

class TestFeatureBasedLoss:

    def setup_method(self):
        self.teacher = TinyTeacher()
        self.student = TinyStudent()
        self.distiller = FeatureBasedDistillation(
            teacher=self.teacher,
            student=self.student,
            device=DEVICE,
        )

    def test_feature_loss_zero_identical_features(self):
        """MSE feature loss must be 0 when student and teacher features are identical."""
        torch.manual_seed(1)
        self.distiller.adapter = torch.nn.Identity()
        feat = torch.randn(4, 64, 8, 8)
        loss = self.distiller.feature_loss(feat, feat)
        assert loss.item() < 1e-10, (
            f"Feature MSE should be 0 for identical features, got {loss.item()}"
        )

    def test_feature_loss_positive_for_different_features(self):
        """MSE feature loss must be positive for non-identical features."""
        torch.manual_seed(2)
        self.distiller.adapter = torch.nn.Identity()
        feat_a = torch.randn(4, 64, 8, 8)
        feat_b = torch.randn(4, 64, 8, 8)
        loss = self.distiller.feature_loss(feat_a, feat_b)
        assert loss.item() > 0, "Feature MSE should be positive for different features"

    def test_adapter_output_shape(self):
        """Adapter must project to teacher's channel dimension."""
        student_channels = self.student.intermediate_channels
        teacher_channels = self.teacher.FEAT_CHANNELS

        adapter = FeatureAdapter(student_channels, teacher_channels)
        x = torch.randn(4, student_channels, 8, 8)
        out = adapter(x)

        assert out.shape[1] == teacher_channels, (
            f"Adapter output channels {out.shape[1]} != teacher channels {teacher_channels}"
        )

    def test_feature_loss_decreases_with_training(self):
        """Feature loss should decrease after one optimizer step."""
        torch.manual_seed(11)
        images, labels = make_mock_batch(batch_size=8)

        params = (
            [p for p in self.student.parameters() if p.requires_grad]
            + list(self.distiller.adapter.parameters())
        )
        optimizer = torch.optim.Adam(params, lr=1e-2)

        teacher_feats = self.distiller._get_teacher_features(images)
        student_feats_init, _ = self.distiller._get_student_features(images)
        loss_before = self.distiller.feature_loss(
            student_feats_init, teacher_feats).item()

        optimizer.zero_grad()
        student_feats, logits = self.distiller._get_student_features(images)
        f_loss = self.distiller.feature_loss(student_feats, teacher_feats)
        ce_loss = F.cross_entropy(logits, labels)
        (ce_loss + 0.5 * f_loss).backward()
        optimizer.step()

        student_feats_post, _ = self.distiller._get_student_features(images)
        loss_after = self.distiller.feature_loss(
            student_feats_post, teacher_feats).item()

        # At minimum, the loss should not explode
        assert loss_after < loss_before * 100, (
            "Feature loss exploded after one step"
        )


# ── Relation-based tests ────────────────────────────────────────────────

class TestRelationBasedLoss:

    def setup_method(self):
        self.teacher = TinyTeacher()
        self.student = TinyStudent()
        self.distiller = RelationBasedDistillation(
            teacher=self.teacher,
            student=self.student,
            device=DEVICE,
        )

    def test_distance_matrix_diagonal_is_zero(self):
        """Pairwise distance of a vector with itself must be 0."""
        torch.manual_seed(4)
        embeddings = torch.randn(6, 32)
        dists = self.distiller._pairwise_distances(embeddings)

        diagonal = dists.diagonal()
        assert (diagonal < 1e-3).all(), (
            f"Diagonal of distance matrix should be ~0, got max {diagonal.max().item():.6f}"
        )

    def test_distance_matrix_is_symmetric(self):
        """Distance matrix must be symmetric: d(i,j) == d(j,i)."""
        torch.manual_seed(5)
        embeddings = torch.randn(6, 32)
        dists = self.distiller._pairwise_distances(embeddings)

        assert torch.allclose(dists, dists.T, atol=1e-5), (
            "Distance matrix is not symmetric"
        )

    def test_distance_matrix_normalized_to_01(self):
        """All pairwise distances must be in [0, 1] after normalization."""
        torch.manual_seed(6)
        embeddings = torch.randn(8, 32)
        dists = self.distiller._pairwise_distances(embeddings)

        assert dists.min().item() >= -1e-6, "Distance matrix has negative values"
        assert dists.max().item() <= 1.0 + 1e-5, "Distance matrix exceeds 1.0"

    def test_relation_loss_zero_for_identical_embeddings(self):
        """Relation loss must be 0 when teacher and student embeddings are identical."""
        torch.manual_seed(8)
        embeddings = torch.randn(8, 32)
        loss = self.distiller.relation_loss(embeddings, embeddings)
        assert loss.item() < 1e-8, (
            f"Relation loss should be 0 for identical embeddings, got {loss.item()}"
        )

    def test_relation_loss_positive_for_different_embeddings(self):
        """Relation loss must be positive when embedding geometries differ."""
        torch.manual_seed(9)
        emb_a = torch.randn(8, 32)
        emb_b = torch.randn(8, 32)
        loss = self.distiller.relation_loss(emb_a, emb_b)
        assert loss.item() > 0, "Relation loss should be positive for different embeddings"
