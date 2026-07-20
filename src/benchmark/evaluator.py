"""
Evaluation engine: accuracy, latency, size, and efficiency scoring.

WHY NOT JUST ACCURACY?
A student that is 0.5% less accurate than the teacher but 10x faster
is almost certainly the right model to deploy. Accuracy alone cannot
capture this trade-off. We use:

    efficiency_score = accuracy / log2(inference_time_ms)

This rewards models that are both accurate and fast. Doubling speed
(halving latency) adds log2(2)=1 to the denominator, which is a
meaningful improvement. Halving accuracy for any speed gain is never
worth it under this metric.

MODEL SIZE vs LATENCY:
Size (MB) and inference time (ms) are related but not identical.
A quantized model might be small on disk but slow at runtime.
A model with efficient depthwise convolutions might be large in
parameters but fast in practice. We report both separately so
users can choose their deployment constraint.

INFERENCE TIME MEASUREMENT:
We run 100 forward passes and average. The first pass is discarded
(JIT warmup, cache effects). CPU timing is noisier than GPU timing —
we use time.perf_counter() which has microsecond resolution on all
modern platforms.
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import time
import math
from typing import Optional

from src.data_models import CompressionMetrics


class ModelEvaluator:
    """
    Computes all metrics needed to compare distillation methods.

    Designed to be called after training is complete. All methods
    are stateless — pass in the model and data each time.
    """

    def __init__(self, device: torch.device):
        self.device = device

    def accuracy(self, model: nn.Module, loader: DataLoader) -> float:
        """
        Classification accuracy on the provided DataLoader.

        Returns fraction correct in [0, 1]. Never reports on training
        data — callers should always pass val_loader or test_loader.
        """
        model.eval()
        correct = total = 0
        with torch.no_grad():
            for images, labels in loader:
                images = images.to(self.device)
                labels = labels.to(self.device)
                preds = model(images).argmax(dim=1)
                correct += (preds == labels).sum().item()
                total += labels.size(0)
        return correct / max(total, 1)

    def inference_time_ms(
        self,
        model: nn.Module,
        input_shape: tuple = (1, 3, 32, 32),
        num_runs: int = 100,
        warmup: int = 5,
    ) -> float:
        """
        Average per-sample inference time in milliseconds.

        Args:
            model: Trained model in eval mode
            input_shape: Single-sample shape (no batch dim needed — we add 1)
            num_runs: Number of timed forward passes to average
            warmup: Untimed passes to warm up memory allocation

        Returns:
            Mean inference time in ms per sample
        """
        model.eval()
        dummy_input = torch.randn(input_shape).to(self.device)

        # Warmup — JIT compilation, memory allocation, cache priming
        with torch.no_grad():
            for _ in range(warmup):
                _ = model(dummy_input)

        # Timed runs
        times = []
        with torch.no_grad():
            for _ in range(num_runs):
                t0 = time.perf_counter()
                _ = model(dummy_input)
                t1 = time.perf_counter()
                times.append((t1 - t0) * 1000)  # convert to ms

        # Drop top/bottom 5% to reduce outlier influence
        times_sorted = sorted(times)
        trim = max(1, num_runs // 20)
        trimmed = times_sorted[trim:-trim] if len(times_sorted) > 2 * trim else times_sorted
        return sum(trimmed) / len(trimmed)

    def model_size_mb(self, model: nn.Module) -> float:
        """
        Model size in megabytes (float32 parameters only).

        Formula: sum(numel * 4 bytes) / 1e6
        This is the in-memory footprint, which approximates the
        serialized .pt file size (ignoring optimizer state).
        """
        total_bytes = sum(
            p.numel() * 4  # float32 = 4 bytes
            for p in model.parameters()
        )
        return total_bytes / 1e6

    def num_parameters(self, model: nn.Module) -> int:
        """Total parameter count."""
        return sum(p.numel() for p in model.parameters())

    def compression_ratio(
            self,
            teacher: nn.Module,
            student: nn.Module) -> float:
        """
        Ratio of teacher parameters to student parameters.

        compression_ratio = 2x means the student is half the size.
        We use parameter count rather than FLOPs because parameter
        count is model-intrinsic; FLOPs depend on input resolution.
        """
        teacher_params = self.num_parameters(teacher)
        student_params = self.num_parameters(student)
        return teacher_params / max(student_params, 1)

    def accuracy_retained_pct(
        self, student_acc: float, teacher_acc: float
    ) -> float:
        """Percentage of teacher accuracy preserved by the student."""
        return (student_acc / max(teacher_acc, 1e-8)) * 100

    def efficiency_score(self, accuracy: float, latency_ms: float) -> float:
        """
        Accuracy-efficiency trade-off score.

        score = accuracy / log2(latency_ms)

        Higher is better. A model that halves latency (same accuracy)
        gains log2(2)=1 in the denominator, so score increases ~40%
        for a 50% speedup at 80% accuracy baseline.

        Edge case: log2(latency) ≤ 0 for latency < 1ms — we clamp to
        a minimum of 0.1ms to avoid division by zero or negative scores.
        """
        safe_latency = max(latency_ms, 0.1)
        log_latency = math.log2(safe_latency)
        if log_latency <= 0:
            log_latency = 0.01  # sub-millisecond models get a bonus
        return accuracy / log_latency

    def evaluate_model(
        self,
        model: nn.Module,
        test_loader: DataLoader,
        teacher: Optional[nn.Module] = None,
        teacher_accuracy: float = 0.0,
        method: str = "unknown",
        compression_variant: str = "full",
    ) -> CompressionMetrics:
        """
        Run the full evaluation suite and return CompressionMetrics.

        This is the single entry point for the benchmark loop. It
        computes everything in one call so the caller doesn't have
        to manage partial state.
        """
        acc = self.accuracy(model, test_loader)
        latency = self.inference_time_ms(model)
        size = self.model_size_mb(model)
        n_params = self.num_parameters(model)
        comp_ratio = self.compression_ratio(teacher, model) if teacher else 1.0
        acc_retained = self.accuracy_retained_pct(acc, teacher_accuracy)
        score = self.efficiency_score(acc, latency)

        return CompressionMetrics(
            method=method,
            compression_variant=compression_variant,
            accuracy=acc,
            inference_time_ms=latency,
            size_mb=size,
            num_parameters=n_params,
            compression_ratio=comp_ratio,
            accuracy_retained_pct=acc_retained,
            efficiency_score=score,
            teacher_accuracy=teacher_accuracy,
        )
