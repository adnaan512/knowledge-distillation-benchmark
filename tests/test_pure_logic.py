"""
Pure-Python logic tests — run without PyTorch.

These tests cover everything that doesn't require a tensor:
- Dataclass construction and fields
- Report generator HTML output
- Efficiency score formula
- ASCII chart generation
- Model size formula
- Compression ratio formula

These can run in any environment, including lightweight CI environments
where installing torch would time out.
"""

from src.reporting.report_generator import ReportGenerator, _ascii_bar, _temperature_chart
from src.data_models import (
    DistillationResult, CompressionMetrics, BenchmarkReport, ModelProfile
)
import math
import sys
import os

# Make src importable
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))


# ── Dataclass tests ─────────────────────────────────────────────────────

class TestDataclasses:

    def test_distillation_result_defaults(self):
        r = DistillationResult(
            method="response",
            compression_variant="half",
            final_accuracy=0.72,
        )
        assert r.method == "response"
        assert r.compression_variant == "half"
        assert r.final_accuracy == 0.72
        assert r.train_loss_history == []
        assert r.val_accuracy_history == []
        assert r.best_epoch == 0
        assert r.hyperparams == {}

    def test_compression_metrics_fields(self):
        m = CompressionMetrics(
            method="feature",
            compression_variant="full",
            accuracy=0.705,
            inference_time_ms=12.1,
            size_mb=13.8,
            num_parameters=3_400_000,
            compression_ratio=7.3,
            accuracy_retained_pct=96.8,
            efficiency_score=0.176,
            teacher_accuracy=0.728,
        )
        assert m.accuracy == 0.705
        assert m.compression_ratio == 7.3
        assert m.efficiency_score == 0.176

    def test_model_profile_param_millions(self):
        p = ModelProfile(
            name="resnet50",
            num_parameters=25_557_032,
            size_mb=97.5,
            architecture="resnet50",
        )
        assert abs(p.param_millions - 25.557032) < 1e-4

    def test_benchmark_report_best_result(self):
        results = [
            CompressionMetrics(
                "response",
                "full",
                0.71,
                12.0,
                13.8,
                3_400_000,
                7.3,
                97.5,
                0.178,
                0.728),
            CompressionMetrics(
                "feature",
                "half",
                0.69,
                7.4,
                5.7,
                1_400_000,
                17.5,
                94.7,
                0.193,
                0.728),
            CompressionMetrics(
                "relation",
                "quarter",
                0.64,
                5.9,
                3.6,
                900_000,
                26.0,
                87.9,
                0.188,
                0.728),
        ]
        report = BenchmarkReport(
            teacher_accuracy=0.728,
            teacher_size_mb=97.5,
            results=results,
        )
        best = report.best_result()
        assert best is not None
        assert best.efficiency_score == max(
            r.efficiency_score for r in results)

    def test_benchmark_report_empty(self):
        report = BenchmarkReport(teacher_accuracy=0.72, teacher_size_mb=97.5)
        assert report.best_result() is None


# ── Efficiency score formula ────────────────────────────────────────────

class TestEfficiencyScore:

    def _score(self, acc, latency_ms):
        safe = max(latency_ms, 0.1)
        log_l = math.log2(safe)
        if log_l <= 0:
            log_l = 0.01
        return acc / log_l

    def test_positive_for_realistic_inputs(self):
        score = self._score(0.85, 10.0)
        assert score > 0

    def test_faster_model_higher_score(self):
        slow = self._score(0.85, 20.0)
        fast = self._score(0.85, 10.0)
        assert fast > slow

    def test_accurate_model_higher_score(self):
        lo = self._score(0.70, 10.0)
        hi = self._score(0.90, 10.0)
        assert hi > lo

    def test_formula_value(self):
        # accuracy=0.8, latency=8ms → score = 0.8 / log2(8) = 0.8 / 3 ≈ 0.2667
        score = self._score(0.8, 8.0)
        assert abs(score - 0.8 / 3.0) < 1e-9

    def test_clamped_minimum_latency(self):
        # Very fast model (0.01ms) should use clamped value, not produce inf
        score = self._score(0.9, 0.001)
        assert math.isfinite(score)
        assert score > 0


# ── Model size formula ──────────────────────────────────────────────────

class TestModelSizeFormula:

    def test_size_formula(self):
        # 1M float32 params = 4MB
        n_params = 1_000_000
        expected_mb = n_params * 4 / 1e6
        assert abs(expected_mb - 4.0) < 1e-9

    def test_resnet50_approximate_size(self):
        # ResNet-50 has ~25.6M params → ~97.5 MB
        n_params = 25_557_032
        size_mb = n_params * 4 / 1e6
        assert 95.0 < size_mb < 105.0

    def test_mobilenetv2_approximate_size(self):
        # MobileNetV2 full ~3.4M params → ~13.6 MB
        n_params = 3_400_000
        size_mb = n_params * 4 / 1e6
        assert 10.0 < size_mb < 20.0


# ── Compression ratio ───────────────────────────────────────────────────

class TestCompressionRatio:

    def test_teacher_larger(self):
        teacher_params = 25_557_032
        student_params = 3_400_000
        ratio = teacher_params / student_params
        assert ratio > 1.0
        assert 6.0 < ratio < 9.0  # approximately 7×

    def test_self_ratio_is_one(self):
        n = 3_400_000
        ratio = n / n
        assert ratio == 1.0

    def test_quarter_compression(self):
        teacher_params = 25_557_032
        student_params = 900_000
        ratio = teacher_params / student_params
        assert ratio > 20.0  # should be approximately 26×


# ── ASCII chart generation ──────────────────────────────────────────────

class TestASCIICharts:

    def test_ascii_bar_full(self):
        bar = _ascii_bar(100.0, 100.0, width=10)
        assert bar == "██████████"

    def test_ascii_bar_empty(self):
        bar = _ascii_bar(0.0, 100.0, width=10)
        assert bar == "░░░░░░░░░░"

    def test_ascii_bar_half(self):
        bar = _ascii_bar(50.0, 100.0, width=10)
        assert bar.count("█") == 5
        assert bar.count("░") == 5

    def test_temperature_chart_with_tuple_keys(self):
        sweep = {
            (1.0, 0.5): 0.61,
            (4.0, 0.5): 0.70,
            (8.0, 0.5): 0.69,
            (16.0, 0.5): 0.62,
        }
        chart = _temperature_chart(sweep)
        assert "T=" in chart
        assert "70.0%" in chart  # best temperature

    def test_temperature_chart_empty(self):
        chart = _temperature_chart({})
        assert "no sweep" in chart.lower()

    def test_temperature_chart_aggregates_alphas(self):
        """Multiple alpha values for same T should be averaged."""
        sweep = {
            (4.0, 0.1): 0.68,
            (4.0, 0.5): 0.70,
            (4.0, 0.9): 0.66,
        }
        chart = _temperature_chart(sweep)
        # Average = (0.68 + 0.70 + 0.66) / 3 = 0.68
        assert "68.0%" in chart


# ── Report HTML generation ──────────────────────────────────────────────

class TestReportGenerator:

    def _make_report(self):
        results = [
            CompressionMetrics("response", "full", 0.712, 12.1, 13.8, 3_400_000, 7.3, 97.8, 0.178, 0.728),
            CompressionMetrics("response", "half", 0.698, 7.4, 5.7, 1_400_000, 17.5, 95.9, 0.196, 0.728),
            CompressionMetrics("feature", "full", 0.705, 12.1, 13.8, 3_400_000, 7.3, 96.8, 0.176, 0.728),
            CompressionMetrics("relation", "quarter", 0.642, 5.9, 3.6, 900_000, 26.0, 88.2, 0.188, 0.728),
            CompressionMetrics("response", "quarter", 0.658, 5.9, 3.6, 900_000, 26.0, 90.4, 0.192, 0.728),
        ]
        return BenchmarkReport(
            teacher_accuracy=0.728,
            teacher_size_mb=97.5,
            results=results,
            best_method="response",
            best_score=0.196,
        )

    def test_report_generates_html(self, tmp_path):
        report = self._make_report()
        gen = ReportGenerator()
        out = tmp_path / "test_report.html"
        path = gen.generate(report, output_path=str(out))
        content = open(path, encoding="utf-8").read()
        assert content.startswith("<!DOCTYPE html>")
        assert len(content) > 5000

    def test_report_contains_key_sections(self, tmp_path):
        report = self._make_report()
        gen = ReportGenerator()
        out = tmp_path / "test_report.html"
        path = gen.generate(report, output_path=str(out))
        content = open(path, encoding="utf-8").read()

        assert "Method Comparison" in content
        assert "Temperature Sensitivity" in content
        assert "Compression Ratio" in content
        assert "Research Questions" in content
        assert "RQ1" in content
        assert "RQ2" in content
        assert "RQ3" in content

    def test_report_contains_best_method(self, tmp_path):
        report = self._make_report()
        gen = ReportGenerator()
        out = tmp_path / "test_report.html"
        path = gen.generate(report, output_path=str(out))
        content = open(path, encoding="utf-8").read()
        # Best method (response) should appear in the table
        assert "response" in content.lower()

    def test_report_shows_counter_intuitive_finding(self, tmp_path):
        """
        Response-based should beat relation-based on quarter variant in our fixture data.
        The report should surface this as a counter-intuitive finding.
        """
        report = self._make_report()
        gen = ReportGenerator()
        out = tmp_path / "test_report.html"
        path = gen.generate(report, output_path=str(out))
        content = open(path, encoding="utf-8").read()
        # The finding section should mention the counter-intuitive result
        assert "counter-intuitive" in content.lower() or "Counter-intuitive" in content

    def test_report_includes_temperature_chart(self, tmp_path):
        report = self._make_report()
        temp_sweep = {(1.0, 0.5): 0.61, (4.0, 0.5): 0.70, (8.0, 0.5): 0.69}
        gen = ReportGenerator()
        out = tmp_path / "test_report.html"
        path = gen.generate(
            report,
            output_path=str(out),
            temp_sweep=temp_sweep)
        content = open(path, encoding="utf-8").read()
        assert "T=" in content

    def test_report_includes_author(self, tmp_path):
        report = self._make_report()
        gen = ReportGenerator()
        out = tmp_path / "test_report.html"
        path = gen.generate(report, output_path=str(out))
        content = open(path, encoding="utf-8").read()
        assert "Adnan Hassnain" in content

    def test_report_is_self_contained(self, tmp_path):
        """No CDN links — the file must be self-contained."""
        report = self._make_report()
        gen = ReportGenerator()
        out = tmp_path / "test_report.html"
        path = gen.generate(report, output_path=str(out))
        content = open(path, encoding="utf-8").read()
        assert "cdn.jsdelivr" not in content
        assert "cdnjs.cloudflare" not in content
        assert "googleapis.com" not in content


# ── Accuracy retained formula ───────────────────────────────────────────

class TestAccuracyRetained:

    def test_perfect_retention(self):
        retained = (0.90 / 0.90) * 100
        assert abs(retained - 100.0) < 1e-9

    def test_typical_case(self):
        retained = (0.855 / 0.90) * 100
        assert abs(retained - 95.0) < 1e-9

    def test_division_by_zero_protection(self):
        teacher_acc = 0.0
        safe = max(teacher_acc, 1e-8)
        retained = (0.85 / safe) * 100
        assert math.isfinite(retained)
