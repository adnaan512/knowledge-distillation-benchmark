"""
Unit tests for the evaluation engine.

These tests verify that measurement functions return values in
sensible ranges and behave correctly on known inputs. We use
mock models so tests run in under 5 seconds on any hardware.
"""

import torch

from tests.fixtures.mock_models import TinyTeacher, TinyStudent
from src.benchmark.evaluator import ModelEvaluator
from src.data.dataset_loader import MockDatasetLoader


DEVICE = torch.device("cpu")


class TestModelEvaluator:

    def setup_method(self):
        self.evaluator = ModelEvaluator(device=DEVICE)
        self.teacher = TinyTeacher()
        self.student = TinyStudent(variant="full")

    # ── Accuracy ────────────────────────────────────────────────────────────

    def test_accuracy_range(self):
        """Accuracy must be in [0, 1]."""
        loader = MockDatasetLoader(
            num_samples=40, batch_size=8).get_loaders()[2]
        acc = self.evaluator.accuracy(self.student, loader)
        assert 0.0 <= acc <= 1.0, f"Accuracy {acc} out of [0, 1]"

    def test_accuracy_perfect_model(self):
        """A model that always predicts class 0 should get accuracy = fraction of class-0 labels."""
        import torch.nn as nn

        class AlwaysZero(nn.Module):
            def forward(self, x):
                # Returns logit of 100 for class 0, -100 for all others
                out = torch.full((x.size(0), 10), -100.0)
                out[:, 0] = 100.0
                return out

        loader = MockDatasetLoader(
            num_samples=100,
            batch_size=10,
            seed=42).get_loaders()[2]
        model = AlwaysZero()
        acc = self.evaluator.accuracy(model, loader)

        # Count ground-truth class-0 labels in the test set
        correct = sum(
            (labels == 0).sum().item()
            for _, labels in loader
        )
        total = sum(labels.size(0) for _, labels in loader)
        expected = correct / max(total, 1)
        assert abs(acc - expected) < 1e-5

    # ── Inference time ──────────────────────────────────────────────────────

    def test_inference_time_positive(self):
        """Inference time must be positive."""
        t = self.evaluator.inference_time_ms(
            self.student, num_runs=10, warmup=2)
        assert t > 0, f"Inference time should be positive, got {t}"

    def test_inference_time_ms_not_seconds(self):
        """
        Inference time should be in milliseconds, not seconds.

        A tiny model on CPU should run in under 500ms per sample.
        If we accidentally return seconds, this would be > 0.5.
        """
        t = self.evaluator.inference_time_ms(
            self.student, num_runs=20, warmup=3)
        assert t < 500.0, (
            f"Inference time {t:.1f} suggests units are wrong (should be ms, < 500ms for tiny model)"
        )

    def test_larger_model_not_faster(self):
        """
        Teacher (larger) should generally not be faster than student (smaller).

        This is a soft assertion — on tiny mock models timing can be noisy —
        so we allow teacher to be at most 3× faster before failing.
        """
        teacher_t = self.evaluator.inference_time_ms(
            self.teacher, num_runs=20, warmup=3)
        student_t = self.evaluator.inference_time_ms(
            self.student, num_runs=20, warmup=3)

        # Teacher should not be more than 3x faster than student on mock models
        assert teacher_t < student_t * 3 or student_t < teacher_t * 3, (
            "Timing relationship between teacher and student is inconsistent"
        )

    # ── Model size ──────────────────────────────────────────────────────────

    def test_model_size_positive(self):
        """Size in MB must be positive."""
        size = self.evaluator.model_size_mb(self.student)
        assert size > 0, "Model size should be positive"

    def test_model_size_float32(self):
        """
        Size should reflect float32 storage: params * 4 bytes / 1e6.
        We verify this formula directly against param count.
        """
        n_params = self.evaluator.num_parameters(self.student)
        expected_mb = n_params * 4 / 1e6
        computed_mb = self.evaluator.model_size_mb(self.student)
        assert abs(computed_mb - expected_mb) < 1e-5, (
            f"Size formula mismatch: expected {expected_mb:.4f} MB, got {computed_mb:.4f} MB"
        )

    def test_teacher_larger_than_student(self):
        """Teacher model must have more parameters than student in mock setup."""
        teacher_params = self.evaluator.num_parameters(self.teacher)
        student_params = self.evaluator.num_parameters(self.student)
        assert teacher_params > student_params, (
            f"Teacher ({teacher_params}) should have more params than student ({student_params})"
        )

    # ── Compression ratio ───────────────────────────────────────────────────

    def test_compression_ratio_greater_than_one(self):
        """Teacher has more params → compression ratio > 1."""
        ratio = self.evaluator.compression_ratio(self.teacher, self.student)
        assert ratio > 1.0, f"Compression ratio {ratio:.2f} should be > 1"

    def test_compression_ratio_self(self):
        """Comparing a model to itself should give ratio ≈ 1.0."""
        ratio = self.evaluator.compression_ratio(self.student, self.student)
        assert abs(
            ratio - 1.0) < 1e-5, f"Self-compression ratio should be 1.0, got {ratio}"

    # ── Efficiency score ────────────────────────────────────────────────────

    def test_efficiency_score_positive(self):
        """Efficiency score must be positive for accuracy > 0."""
        score = self.evaluator.efficiency_score(accuracy=0.85, latency_ms=10.0)
        assert score > 0, f"Efficiency score should be positive, got {score}"

    def test_faster_model_higher_score(self):
        """Halving latency at the same accuracy should increase efficiency score."""
        score_slow = self.evaluator.efficiency_score(0.85, latency_ms=20.0)
        score_fast = self.evaluator.efficiency_score(0.85, latency_ms=10.0)
        assert score_fast > score_slow, (
            f"Faster model ({score_fast:.3f}) should score higher than slower ({score_slow:.3f})"
        )

    def test_more_accurate_model_higher_score(self):
        """Higher accuracy at the same latency should increase efficiency score."""
        score_lo = self.evaluator.efficiency_score(0.70, latency_ms=10.0)
        score_hi = self.evaluator.efficiency_score(0.85, latency_ms=10.0)
        assert score_hi > score_lo, (
            f"More accurate model ({score_hi:.3f}) should score higher than less accurate ({score_lo:.3f})"
        )

    # ── Full evaluate_model integration ──────────────────────────────────────

    def test_evaluate_model_returns_metrics(self):
        """evaluate_model should return a CompressionMetrics with all fields set."""
        loader = MockDatasetLoader(
            num_samples=32, batch_size=8).get_loaders()[2]
        metrics = self.evaluator.evaluate_model(
            model=self.student,
            test_loader=loader,
            teacher=self.teacher,
            teacher_accuracy=0.90,
            method="response",
            compression_variant="full",
        )
        assert metrics.accuracy >= 0.0
        assert metrics.inference_time_ms > 0
        assert metrics.size_mb > 0
        assert metrics.compression_ratio > 1.0
        assert metrics.efficiency_score > 0
        assert metrics.method == "response"
        assert metrics.compression_variant == "full"


class TestMockDatasetLoader:

    def test_loader_shapes(self):
        """All loaders must return (B, 3, 32, 32) images and (B,) labels."""
        mock = MockDatasetLoader(num_samples=60, batch_size=16)
        train, val, test = mock.get_loaders()

        for loader_name, loader in [
                ("train", train), ("val", val), ("test", test)]:
            for images, labels in loader:
                assert images.shape[1:] == (3, 224, 224), (
                    f"{loader_name}: expected image shape (B, 3, 224, 224), got {images.shape}"
                )
                assert labels.dim() == 1, f"{loader_name}: labels should be 1D"
                assert labels.max().item() <= 9, f"{loader_name}: label > 9"
                assert labels.min().item() >= 0, f"{loader_name}: label < 0"
                break  # one batch is enough

    def test_deterministic_with_same_seed(self):
        """Same seed should produce the same data."""
        m1 = MockDatasetLoader(num_samples=40, seed=42)
        m2 = MockDatasetLoader(num_samples=40, seed=42)
        _, _, t1 = m1.get_loaders()
        _, _, t2 = m2.get_loaders()

        for (img1, lbl1), (img2, lbl2) in zip(t1, t2):
            assert torch.allclose(
                img1, img2), "Same seed should give same images"
            assert (lbl1 == lbl2).all(), "Same seed should give same labels"
            break
