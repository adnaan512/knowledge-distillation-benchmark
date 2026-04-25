"""
Core dataclasses for the knowledge distillation benchmark.

These structures serve as the contract between experiment components —
distillation methods produce DistillationResult, evaluators produce
CompressionMetrics, and the reporter consumes both.
"""

from dataclasses import dataclass, field
from typing import Optional, Dict, List


@dataclass
class ModelProfile:
    """Snapshot of a model's static properties before any training."""
    name: str
    num_parameters: int
    size_mb: float
    architecture: str
    width_mult: float = 1.0

    @property
    def param_millions(self) -> float:
        return self.num_parameters / 1_000_000


@dataclass
class DistillationResult:
    """
    Output of a single distillation run.

    Captures both the training outcome (loss curves, best epoch) and
    the final evaluation metrics so the reporter can reconstruct the
    full story of each experiment.
    """
    method: str                        # "response", "feature", "relation"
    compression_variant: str           # "full", "half", "quarter"
    final_accuracy: float              # test accuracy [0, 1]
    train_loss_history: List[float] = field(default_factory=list)
    val_accuracy_history: List[float] = field(default_factory=list)
    best_epoch: int = 0
    hyperparams: Dict = field(default_factory=dict)  # T, alpha, lambda, etc.


@dataclass
class CompressionMetrics:
    """
    All numbers needed to evaluate the accuracy-efficiency trade-off.

    The efficiency_score = accuracy / log2(latency_ms) implements
    a simple Pareto-style ranking: a model that is twice as fast
    as another at the same accuracy scores higher, but halving
    accuracy to save 1ms is never worth it.
    """
    method: str
    compression_variant: str
    accuracy: float
    inference_time_ms: float
    size_mb: float
    num_parameters: int
    compression_ratio: float          # teacher_params / student_params
    accuracy_retained_pct: float      # student_acc / teacher_acc * 100
    efficiency_score: float           # accuracy / log2(inference_time_ms)
    teacher_accuracy: float = 0.0


@dataclass
class BenchmarkReport:
    """Aggregated results across all methods and compression variants."""
    teacher_accuracy: float
    teacher_size_mb: float
    results: List[CompressionMetrics] = field(default_factory=list)
    temperature_sweep: Dict[float, float] = field(default_factory=dict)
    best_method: str = ""
    best_score: float = 0.0

    def best_result(self) -> Optional[CompressionMetrics]:
        if not self.results:
            return None
        return max(self.results, key=lambda r: r.efficiency_score)
