"""
Quick demo: all 3 distillation methods on 100 mock samples.

No downloads. No GPU. Completes in ~60 seconds on any modern laptop.
Produces a full dark HTML report at ./demo_report.html.

Usage:
    python examples/run_demo.py
    python examples/run_demo.py --epochs 2 --output report.html
    python examples/run_demo.py --quiet
"""

import sys
import os
import argparse

# Allow running from project root or from examples/
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import torch
from src.data.dataset_loader import MockDatasetLoader
from src.data_models import BenchmarkReport, CompressionMetrics
from src.benchmark.evaluator import ModelEvaluator
from src.reporting.report_generator import ReportGenerator
from src.distillation.response_based import ResponseBasedDistillation
from src.distillation.feature_based import FeatureBasedDistillation
from src.distillation.relation_based import RelationBasedDistillation
from tests.fixtures.mock_models import TinyTeacher, TinyStudent


def parse_args():
    p = argparse.ArgumentParser(description="Knowledge Distillation Demo (mock mode)")
    p.add_argument("--epochs",  type=int, default=3,                  help="Epochs per method (default: 3)")
    p.add_argument("--samples", type=int, default=100,                 help="Mock dataset size (default: 100)")
    p.add_argument("--output",  type=str, default="demo_report.html",  help="Report output path")
    p.add_argument("--quiet",   action="store_true",                   help="Suppress per-epoch output")
    return p.parse_args()


def run_demo(
    epochs: int = 3,
    samples: int = 100,
    output: str = "demo_report.html",
    verbose: bool = True,
) -> BenchmarkReport:
    """
    Run all three distillation methods on synthetic mock data and
    produce an HTML report. Returns the BenchmarkReport for programmatic use.

    Args:
        epochs:  Training epochs per method/variant combination
        samples: Number of synthetic (3, 32, 32) samples to generate
        output:  Path to write the HTML report
        verbose: Print per-epoch loss and accuracy

    Returns:
        BenchmarkReport with all metrics populated
    """
    device  = torch.device("cpu")
    methods  = ["response", "feature", "relation"]
    variants = ["full", "half", "quarter"]

    print("\n" + "═" * 62)
    print("  Knowledge Distillation Benchmark — DEMO MODE")
    print(f"  Mock data ({samples} samples) · No downloads · CPU only")
    print(f"  {len(methods)} methods × {len(variants)} variants × {epochs} epochs")
    print("═" * 62 + "\n")

    # ── Dataset ───────────────────────────────────────────────────────────────
    print(f"[1/5] Building mock CIFAR-10-shaped dataset ({samples} samples)...")
    mock = MockDatasetLoader(num_samples=samples, batch_size=16, seed=42)
    train_loader, val_loader, test_loader = mock.get_loaders()
    print("      ✓ Train / val / test loaders ready\n")

    # ── Teacher ───────────────────────────────────────────────────────────────
    print("[2/5] Initialising mock teacher (TinyTeacher, 2-layer, frozen)...")
    teacher   = TinyTeacher(frozen=True).to(device)
    evaluator = ModelEvaluator(device=device)
    teacher_acc  = evaluator.accuracy(teacher, test_loader)
    teacher_size = evaluator.model_size_mb(teacher)
    print(f"      Teacher accuracy (random init): {teacher_acc*100:.1f}%  "
          f"| Size: {teacher_size:.3f} MB\n")

    all_results:  list[CompressionMetrics] = []
    temp_sweep:   dict = {}
    step = 3

    # ── Response-based ────────────────────────────────────────────────────────
    print(f"[{step}/5] Response-based distillation (soft labels + temperature sweep)...")
    step += 1
    for variant in variants:
        print(f"  ▶ variant = {variant}")
        student   = TinyStudent(variant=variant).to(device)
        distiller = ResponseBasedDistillation(
            teacher=teacher, student=student, device=device,
            temperatures=[1.0, 4.0, 8.0],
            alphas=[0.1, 0.5],
        )
        _, sweep = distiller.sweep(
            train_loader, val_loader, test_loader,
            num_epochs=epochs, variant=variant, verbose=verbose,
        )
        if not temp_sweep:
            temp_sweep = sweep

        metrics = evaluator.evaluate_model(
            model=student, test_loader=test_loader,
            teacher=teacher, teacher_accuracy=teacher_acc,
            method="response", compression_variant=variant,
        )
        all_results.append(metrics)
        _print_metrics(metrics)

    # ── Feature-based ─────────────────────────────────────────────────────────
    print(f"\n[{step}/5] Feature-based distillation (intermediate layer MSE)...")
    step += 1
    for variant in variants:
        print(f"  ▶ variant = {variant}")
        student   = TinyStudent(variant=variant).to(device)
        distiller = FeatureBasedDistillation(
            teacher=teacher, student=student, device=device, lam=0.5,
        )
        distiller.run(
            train_loader, val_loader, test_loader,
            num_epochs=epochs, variant=variant, verbose=verbose,
        )
        metrics = evaluator.evaluate_model(
            model=student, test_loader=test_loader,
            teacher=teacher, teacher_accuracy=teacher_acc,
            method="feature", compression_variant=variant,
        )
        all_results.append(metrics)
        _print_metrics(metrics)

    # ── Relation-based ────────────────────────────────────────────────────────
    print(f"\n[{step}/5] Relation-based distillation (pairwise geometry matching)...")
    step += 1
    for variant in variants:
        print(f"  ▶ variant = {variant}")
        student   = TinyStudent(variant=variant).to(device)
        distiller = RelationBasedDistillation(
            teacher=teacher, student=student, device=device, lam=0.5,
        )
        distiller.run(
            train_loader, val_loader, test_loader,
            num_epochs=epochs, variant=variant, verbose=verbose,
        )
        metrics = evaluator.evaluate_model(
            model=student, test_loader=test_loader,
            teacher=teacher, teacher_accuracy=teacher_acc,
            method="relation", compression_variant=variant,
        )
        all_results.append(metrics)
        _print_metrics(metrics)

    # ── Report ────────────────────────────────────────────────────────────────
    print(f"\n[{step}/5] Generating HTML report → {output}")
    best = max(all_results, key=lambda r: r.efficiency_score) if all_results else None
    report = BenchmarkReport(
        teacher_accuracy=teacher_acc,
        teacher_size_mb=teacher_size,
        results=all_results,
        temperature_sweep=temp_sweep,
        best_method=best.method if best else "",
        best_score=best.efficiency_score if best else 0.0,
    )
    ReportGenerator().generate(report, output_path=output, temp_sweep=temp_sweep)
    print(f"      ✓ Report written to: {output}\n")

    # ── Summary ───────────────────────────────────────────────────────────────
    print("─" * 62)
    print(f"{'Method':<12} {'Variant':<10} {'Acc':>6} {'Latency':>10} {'Score':>8}")
    print("─" * 62)
    for r in sorted(all_results, key=lambda x: x.efficiency_score, reverse=True):
        marker = " ◀ best" if r is best else ""
        print(
            f"{r.method:<12} {r.compression_variant:<10} "
            f"{r.accuracy*100:>5.1f}%  {r.inference_time_ms:>7.2f} ms  "
            f"{r.efficiency_score:>7.3f}{marker}"
        )
    print("─" * 62)
    print(
        "\n⚠  Demo uses random mock data — accuracy values are meaningless.\n"
        "   Run `python main.py --mode full` for real CIFAR-10 results.\n"
    )
    return report


def _print_metrics(m: CompressionMetrics) -> None:
    print(
        f"      acc={m.accuracy*100:.1f}%  "
        f"latency={m.inference_time_ms:.2f}ms  "
        f"size={m.size_mb:.3f}MB  "
        f"score={m.efficiency_score:.3f}"
    )


if __name__ == "__main__":
    args = parse_args()
    run_demo(
        epochs=args.epochs,
        samples=args.samples,
        output=args.output,
        verbose=not args.quiet,
    )
