"""
Knowledge Distillation Benchmark — CLI entry point.

Usage:
    python main.py --mode demo                          # Mock data, ~60s, no downloads
    python main.py --mode full                          # Real CIFAR-10, all methods (~20 min)
    python main.py --method response --compression half # Single method + variant
    python main.py --dry-run                            # Validate imports only
"""

import argparse
import sys
import os

# Ensure project root is on path when invoked directly
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def parse_args():
    parser = argparse.ArgumentParser(
        description="Knowledge Distillation Benchmark",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py --mode demo
  python main.py --mode full --epochs 10
  python main.py --method response --compression half --mode full --epochs 5
  python main.py --dry-run
        """,
    )
    parser.add_argument(
        "--mode", choices=["demo", "full"], default="demo",
        help="'demo' = mock data (no download); 'full' = real CIFAR-10",
    )
    parser.add_argument(
        "--method",
        choices=[
            "response",
            "feature",
            "relation",
            "all"],
        default="all",
        help="Distillation method to run (default: all)",
    )
    parser.add_argument(
        "--compression",
        choices=[
            "full",
            "half",
            "quarter",
            "all"],
        default="all",
        help="Student compression variant (default: all)",
    )
    parser.add_argument(
        "--epochs", type=int, default=None,
        help="Training epochs per run (default: 3 for demo, 5 for full)",
    )
    parser.add_argument(
        "--output", type=str, default="benchmark_report.html",
        help="HTML report output path (default: benchmark_report.html)",
    )
    parser.add_argument(
        "--data-dir", type=str, default="./data",
        help="Directory containing CIFAR-10 data (default: ./data)",
    )
    parser.add_argument(
        "--teacher-cache",
        type=str,
        default=".",
        help="Writable directory to save/load teacher checkpoint (default: ./ — use /kaggle/working on Kaggle)",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Validate imports and model construction, then exit",
    )
    parser.add_argument(
        "--quiet", action="store_true",
        help="Suppress per-epoch training output",
    )
    parser.add_argument(
        "--max-runs",
        type=int,
        default=5,
        help="Max number of (method, variant) combinations to train (default: 5)",
    )
    return parser.parse_args()


# ── Dry run ─────────────────────────────────────────────────────────────

def dry_run():
    """Validate all imports and a minimal forward pass — no training."""
    import torch
    print("\nDry run — validating imports and model construction...\n")

    from tests.fixtures.mock_models import TinyTeacher, TinyStudent

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    teacher = TinyTeacher().to(device)
    student = TinyStudent().to(device)

    x = torch.randn(2, 3, 32, 32)
    t_out = teacher(x)
    s_out = student(x)

    assert t_out.shape == (2, 10), f"Teacher output shape wrong: {t_out.shape}"
    assert s_out.shape == (2, 10), f"Student output shape wrong: {s_out.shape}"

    print("  ✓ All imports resolved")
    print(f"  ✓ Teacher forward pass: {t_out.shape}")
    print(f"  ✓ Student forward pass: {s_out.shape}")
    print("  ✓ Dry run complete — ready to train\n")


# ── Demo mode ───────────────────────────────────────────────────────────

def run_demo_mode(args):
    """All 3 methods on mock data — no downloads, runs in ~60s."""
    from examples.run_demo import run_demo
    epochs = args.epochs or 3
    run_demo(
        epochs=epochs,
        samples=100,
        output=args.output,
        verbose=not args.quiet,
    )


# ── Full mode ───────────────────────────────────────────────────────────

def run_full_mode(args):
    """Complete benchmark on real CIFAR-10."""
    import torch
    from src.data.dataset_loader import get_cifar10_loaders
    from src.data_models import BenchmarkReport
    from src.benchmark.evaluator import ModelEvaluator
    from src.reporting.report_generator import ReportGenerator
    from src.distillation.response_based import ResponseBasedDistillation
    from src.distillation.feature_based import FeatureBasedDistillation
    from src.distillation.relation_based import RelationBasedDistillation
    from src.models.teacher import TeacherModel
    from src.models.student import StudentModel, COMPRESSION_VARIANTS

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    epochs = args.epochs or 2
    verbose = not args.quiet

    methods = [
        "response",
        "feature",
        "relation"] if args.method == "all" else [
        args.method]
    variants = list(
        COMPRESSION_VARIANTS.keys()) if args.compression == "all" else [
        args.compression]

    # Build flat list of (method, variant) pairs, capped at --max-runs
    all_combinations = [(m, v) for m in methods for v in variants]
    if args.max_runs and args.max_runs < len(all_combinations):
        all_combinations = all_combinations[:args.max_runs]

    print("\n" + "═" * 62)
    print("  Knowledge Distillation Benchmark — FULL MODE")
    print(f"  Methods   : {', '.join(methods)}")
    print(f"  Variants  : {', '.join(variants)}")
    print(
        f"  Runs      : {len(all_combinations)} of {len([m for m in methods]) * len(variants)} combinations")
    print(f"  Epochs    : {epochs} per run")
    print(f"  Device    : {device}")
    print("═" * 62 + "\n")

    # Enable cuDNN auto-tuner for a free speed boost on GPU
    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = True
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    # ── Dataset ─────────────────────────────────────────────────────────────
    print(
        f"[1] Loading CIFAR-10 from '{args.data_dir}' (downloads ~170 MB if needed)...")
    train_loader, val_loader, test_loader = get_cifar10_loaders(
        data_dir=args.data_dir)
    print("    ✓ Dataset ready (45k train / 5k val / 10k test)\n")

    # ── Teacher ─────────────────────────────────────────────────────────────
    print("[2] Building ResNet-50 teacher (ImageNet pretrained)...")
    teacher = TeacherModel(num_classes=10, pretrained=True).to(device)
    teacher_path = os.path.join(args.teacher_cache, "teacher_finetuned.pth")
    if os.path.exists(teacher_path):
        print(
            f"    [!] Found cached teacher at {teacher_path}. Loading weights...")
        teacher.load_state_dict(torch.load(teacher_path, map_location=device))
    else:
        print("    [!] No cached teacher found. Fine-tuning from scratch...")
        teacher.fine_tune_head(train_loader, val_loader, device, epochs=2)
        print(f"    [!] Saving fine-tuned teacher to {teacher_path}...")
        os.makedirs(args.teacher_cache, exist_ok=True)
        torch.save(teacher.state_dict(), teacher_path)
    evaluator = ModelEvaluator(device=device)
    teacher_acc = evaluator.accuracy(teacher, test_loader)
    teacher_size = evaluator.model_size_mb(teacher)
    teacher_params = evaluator.num_parameters(teacher)
    print(f"    Accuracy  : {teacher_acc*100:.1f}%")
    print(f"    Parameters: {teacher_params/1e6:.1f}M")
    print(f"    Size      : {teacher_size:.1f} MB\n")

    all_results = []
    temp_sweep = {}
    step = 3

    for method, variant in all_combinations:
        print(f"[{step}] {method.upper()} distillation — '{variant}' student")
        step += 1

        student = StudentModel(variant=variant, num_classes=10).to(device)
        n_params = evaluator.num_parameters(student)
        print(f"    Student params: {n_params/1e6:.2f}M  "
              f"(compression ≈ {teacher_params/n_params:.1f}×)")

        if method == "response":
            distiller = ResponseBasedDistillation(
                teacher=teacher, student=student, device=device,
                temperatures=[4.0],
                alphas=[0.7],
                # single value = 1 training run per combination
            )
            _, sweep = distiller.sweep(
                train_loader, val_loader, test_loader,
                num_epochs=epochs, variant=variant, verbose=verbose,
            )
            if not temp_sweep:
                temp_sweep = sweep

        elif method == "feature":
            distiller = FeatureBasedDistillation(
                teacher=teacher, student=student, device=device, lam=0.5,
            )
            distiller.run(
                train_loader, val_loader, test_loader,
                num_epochs=epochs, variant=variant, verbose=verbose,
            )

        elif method == "relation":
            distiller = RelationBasedDistillation(
                teacher=teacher, student=student, device=device, lam=0.5,
            )
            distiller.run(
                train_loader, val_loader, test_loader,
                num_epochs=epochs, variant=variant, verbose=verbose,
            )

        metrics = evaluator.evaluate_model(
            model=student,
            test_loader=test_loader,
            teacher=teacher,
            teacher_accuracy=teacher_acc,
            method=method,
            compression_variant=variant,
        )
        all_results.append(metrics)
        print(f"    → Accuracy: {metrics.accuracy*100:.1f}%  "
              f"Retained: {metrics.accuracy_retained_pct:.1f}%  "
              f"Score: {metrics.efficiency_score:.3f}\n")

    # ── Report ──────────────────────────────────────────────────────────────
    print(f"[{step}] Generating HTML report → {args.output}")
    best = max(
        all_results,
        key=lambda r: r.efficiency_score) if all_results else None
    report = BenchmarkReport(
        teacher_accuracy=teacher_acc,
        teacher_size_mb=teacher_size,
        results=all_results,
        temperature_sweep=temp_sweep,
        best_method=best.method if best else "",
        best_score=best.efficiency_score if best else 0.0,
    )
    ReportGenerator().generate(report, output_path=args.output, temp_sweep=temp_sweep)
    print(f"    ✓ Report saved to: {args.output}\n")

    # ── Summary table ───────────────────────────────────────────────────────
    print("─" * 68)
    print(f"{'Method':<12} {'Variant':<10} {'Acc':>6} {'Retained':>10} {'Latency':>10} {'Score':>8}")
    print("─" * 68)
    for r in sorted(
            all_results,
            key=lambda x: x.efficiency_score,
            reverse=True):
        marker = " ◀ best" if r is best else ""
        print(
            f"{r.method:<12} {r.compression_variant:<10} "
            f"{r.accuracy*100:>5.1f}%  {r.accuracy_retained_pct:>7.1f}%  "
            f"{r.inference_time_ms:>8.1f}ms  {r.efficiency_score:>7.3f}{marker}")
    print("─" * 68 + "\n")


# ── Entry point ─────────────────────────────────────────────────────────

def main():
    args = parse_args()

    if args.dry_run:
        dry_run()
        return

    if args.mode == "demo":
        run_demo_mode(args)
    elif args.mode == "full":
        run_full_mode(args)


if __name__ == "__main__":
    main()
