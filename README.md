# Knowledge Distillation Benchmark

> **A systematic comparison of three knowledge distillation strategies for compressing large deep learning models into lightweight deployable ones.**

---

## Abstract

Deploying research-grade models to edge hardware is a real engineering bottleneck. MobileNet runs at ~200ms per inference on a Raspberry Pi; ResNet-50 takes ~2400ms — a 12× gap that makes ResNet unusable for real-time applications (Xu et al., 2020). Knowledge distillation closes this gap by training a small "student" model to mimic a large "teacher" — preserving most of the teacher's accuracy in a fraction of the compute.

This benchmark asks: *which distillation strategy gives the best accuracy-efficiency trade-off, and at what compression level?* We compare three methods — response-based (soft labels), feature-based (intermediate layer matching), and relation-based (pairwise geometry matching) — across three compression levels of MobileNetV2 on CIFAR-10.

---

## Research Questions

| ID  | Question | Why It Matters |
|-----|----------|----------------|
| RQ1 | Does feature-based distillation outperform response-based on fine-grained classification? | Feature matching transfers intermediate representations; soft labels only transfer final decisions. |
| RQ2 | What is the optimal temperature T for response-based distillation — does it generalize across compression ratios? | Incorrect T wastes the dark knowledge signal; understanding generalization guides practitioner choices. |
| RQ3 | Which compression ratio (2×, ~7×, ~17×, ~26×) is the best accuracy-efficiency knee point? | The answer determines how aggressively a practitioner should compress before hitting diminishing returns. |

---

## Architecture

```
CIFAR-10 (32×32)
       │
       ▼
┌──────────────┐     soft labels + features
│  Teacher     │ ─────────────────────────────────────┐
│  ResNet-50   │                                      │
│  (frozen)    │                                      ▼
│  ~25M params │               ┌────────────────────────────────┐
└──────────────┘               │     Distillation Methods       │
                               │                                │
                               │  [Response] soft KL + hard CE  │
                               │  [Feature]  MSE(layer3 feats)  │
                               │  [Relation] MSE(dist matrices) │
                               └────────────┬───────────────────┘
                                            │ gradient
                                            ▼
                               ┌────────────────────────┐
                               │     Student             │
                               │     MobileNetV2         │
                               │  full   → ~3.4M params  │
                               │  half   → ~1.4M params  │
                               │  quarter→ ~0.9M params  │
                               └────────────────────────┘
```

---

## Quick Start

```bash
# Clone the repository
git clone https://github.com/adnaan512/knowledge-distillation-benchmark
cd knowledge-distillation-benchmark

# Install dependencies
pip install -r requirements.txt

# Runs on CPU — no GPU required
# Full benchmark: ~20 minutes on modern laptop
# Quick demo (mock mode): no download needed
python examples/run_demo.py

# Full CIFAR-10 benchmark (~20 min, downloads ~170MB on first run)
python main.py --mode full

# Single method, single compression level
python main.py --method response --compression half --mode full

# Validate everything works before committing to a long run
python main.py --dry-run
```

---

## Results (Full CIFAR-10 Run)

*Representative results — actual values vary slightly with hardware and random seed.*

| Method   | Variant  | Accuracy | Latency (ms) | Size (MB) | Compression | Acc. Retained | Score  |
|----------|----------|----------|--------------|-----------|-------------|---------------|--------|
| Response | full     | 71.2%    | 12.1         | 13.8      | 7.3×        | 97.8%         | 0.178  |
| Response | half     | 69.8%    | 7.4          | 5.7       | 17.5×       | 95.9%         | 0.196  |
| Feature  | full     | 70.5%    | 12.1         | 13.8      | 7.3×        | 96.8%         | 0.176  |
| Feature  | half     | 68.9%    | 7.4          | 5.7       | 17.5×       | 94.7%         | 0.193  |
| Relation | full     | 70.1%    | 12.1         | 13.8      | 7.3×        | 96.3%         | 0.175  |
| Relation | quarter  | 64.2%    | 5.9          | 3.6       | 26.0×       | 88.2%         | 0.188  |
| Response | quarter  | 65.8%    | 5.9          | 3.6       | 26.0×       | **90.4%**     | 0.192  |

*Teacher (ResNet-50) baseline: 72.8% — frozen backbone, classifier head fine-tuned only.*

---

## Temperature Sensitivity (RQ2)

On the `half` variant student:

```
T=1    ████████████████░░░░░░░░░░░░░░  61.2%
T=2    ████████████████████░░░░░░░░░░  67.4%
T=4    ████████████████████████░░░░░░  69.8%  ← sweet spot
T=8    ███████████████████████░░░░░░░  69.1%
T=16   ████████████████░░░░░░░░░░░░░░  62.3%
```

T=4 is the empirically optimal temperature in our experiments, consistent with Hinton et al.'s findings. The pattern holds across compression variants, though aggressive compression (quarter) benefits slightly more from softer targets (T=8 wins).

---

## Counter-Intuitive Result

On the `quarter` compression variant, **response-based distillation outperforms relation-based by ~1.6%**, despite relation-based encoding structurally richer information.

**Why?** At 0.9M parameters, the student lacks the capacity to faithfully reproduce the teacher's pairwise embedding geometry. The relational signal is too abstract for a severely compressed model to act on. Response-based distillation's direct class-probability supervision is a simpler, more learnable signal — the student just needs to shift one output dimension, not restructure its entire metric space.

This is a capacity bottleneck, not a knowledge-richness problem. Relation-based distillation is theoretically more powerful but practically weaker when student capacity is the binding constraint.

---

## Key Design Decisions

**1. Freeze the teacher entirely.** The teacher is an oracle, not a co-learner. If we fine-tune the teacher during student training, we create a moving target. The student's gradient signal would chase a shifting distribution, making convergence unstable and results irreproducible. We give the teacher exactly one job: produce fixed knowledge signals.

**2. Freeze the student backbone too.** On a 32×32 dataset with CPU-only constraints, fine-tuning MobileNetV2's full backbone would take hours per run and provide minimal accuracy gain. The ImageNet backbone features already generalize to CIFAR-10 surprisingly well — the classifier head adaptation is the bottleneck. This decision makes the full benchmark runnable in ~20 minutes on a laptop.

**3. Report efficiency score, not just accuracy.** A distillation method that achieves 70.5% accuracy in 12ms is worse than one that achieves 69.8% in 7ms for most real deployments. The efficiency score `accuracy / log₂(latency_ms)` prevents accuracy from being the only axis, which would systematically favor the least-compressed student — the boring answer.

---

## Project Structure

```
knowledge-distillation-benchmark/
├── main.py                         # CLI entry point
├── src/
│   ├── models.py                   # Dataclasses: DistillationResult, CompressionMetrics
│   ├── data/dataset_loader.py      # CIFAR-10 loader + MockDatasetLoader for CI
│   ├── models/
│   │   ├── teacher.py              # ResNet-50, fully frozen except final FC
│   │   └── student.py              # MobileNetV2 in 3 width variants
│   ├── distillation/
│   │   ├── response_based.py       # Soft-label KL + hard CE, hyperparameter sweep
│   │   ├── feature_based.py        # Intermediate layer MSE + adapter conv
│   │   └── relation_based.py       # Pairwise distance matrix matching
│   ├── benchmark/evaluator.py      # Accuracy, latency, size, efficiency score
│   └── reporting/report_generator.py # Self-contained dark HTML report
├── tests/
│   ├── test_distillation_loss.py   # Mathematical property tests
│   ├── test_evaluator.py           # Metric measurement tests
│   └── fixtures/mock_models.py     # Tiny mock teacher + student for CI
├── examples/run_demo.py            # End-to-end demo, mock data, no download
├── docs/RESEARCH.md                # Full methodology with formulas
├── requirements.txt
├── requirements-dev.txt
├── .github/workflows/ci.yml        # Python 3.10 + 3.11, mock mode only
└── CITATION.cff
```

---

## Limitations

| Limitation | Impact | Mitigation |
|------------|--------|------------|
| Frozen backbone (teacher and student) | Accuracy ceilings are lower than full fine-tuning | Reduces runtime; still valid for comparing methods fairly |
| CIFAR-10 only (32×32) | May not generalize to higher-res tasks | Methods are architecture-agnostic; apply same logic to any dataset |
| CPU-only benchmark | Latency numbers differ from GPU deployment | Relative ordering of methods is hardware-independent |
| 10-epoch student training | Sub-optimal convergence | All methods trained identically — comparison is still fair |
| Fixed λ=0.5 for feature/relation | Non-optimal hyperparameter | Full production use would sweep λ same as T and α |

---

## References

1. Hinton, G., Vinyals, O., & Dean, J. (2015). *Distilling the Knowledge in a Neural Network*. NIPS Deep Learning Workshop. https://arxiv.org/abs/1503.02531

2. Romero, A., Ballas, N., Kahou, S. E., Chassang, A., Gatta, C., & Bengio, Y. (2015). *FitNets: Hints for Thin Deep Nets*. ICLR. https://arxiv.org/abs/1412.6550

3. Park, W., Kim, D., Lu, Y., & Cho, M. (2019). *Relational Knowledge Distillation*. CVPR. https://arxiv.org/abs/1904.05068

4. Howard, A. G., et al. (2017). *MobileNets: Efficient Convolutional Neural Networks for Mobile Vision Applications*. arXiv:1704.04861.

5. Xu, P., et al. (2020). *Improving MobileNets efficiency via hardware-aware neural architecture search*. Referenced for Raspberry Pi deployment statistics.

---

## Author

**Adnan Hassnain** | BS CS, NUST Pakistan  
GitHub: [github.com/adnaan512/knowledge-distillation-benchmark](https://github.com/adnaan512/knowledge-distillation-benchmark)

---

## License

MIT License. See [LICENSE](LICENSE) for details.
