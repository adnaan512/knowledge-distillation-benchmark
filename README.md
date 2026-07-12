# Knowledge Distillation Benchmark

> Comparing three knowledge distillation methods to compress a large model (ResNet-50) into a lightweight one (MobileNetV2) on CIFAR-10.

📓 **[View Kaggle Notebook](https://www.kaggle.com/code/adnanhassnain/knowledge-distillation-benchmark)**

---

## What is Knowledge Distillation?

Knowledge distillation trains a small **student** model to mimic a large **teacher** model. The goal is to keep most of the teacher's accuracy while using far fewer parameters — making the model faster and lighter for deployment.

**Teacher:** ResNet-50 (23.5M params, 78.7% accuracy)  
**Student:** MobileNetV2 in three sizes — `full`, `half`, `quarter`

---

## Methods Compared

| Method | How it works |
|--------|-------------|
| **Response-based** | Student learns from the teacher's output probabilities (soft labels) |
| **Feature-based** | Student learns to match the teacher's intermediate layer features |
| **Relation-based** | Student learns to match pairwise relationships between samples |

---

## Results (from Kaggle run)

| Method | Variant | Accuracy | Compression | Acc. Retained | Score |
|--------|---------|----------|-------------|---------------|-------|
| **Response** | **full** | **67.7%** | **10.5×** | **86.0%** | **0.310** ✅ best |
| Feature | full | 66.8% | 10.5× | 84.9% | 0.303 |
| Response | half | 20.2% | 33.6× | 25.7% | 0.093 |
| Response | quarter | 17.8% | 57.5× | 22.7% | 0.083 |
| Feature | half | 17.4% | 33.6× | 22.1% | 0.079 |

**Teacher (ResNet-50) baseline: 78.7%** — only classifier head fine-tuned (2 epochs).

### Key Takeaway
- The `full` MobileNetV2 student retains **~86% of teacher accuracy** at only 2.24M parameters (10.5× compression) — a strong result.
- The `half` and `quarter` variants collapsed to near-random (~17–20%) with only 2 training epochs. More epochs would be needed for heavily compressed models to converge.
- Response-based and Feature-based distillation perform comparably on the `full` variant.

---

## How to Run

### On Kaggle (recommended)
Open the notebook directly: [kaggle.com/code/adnanhassnain/knowledge-distillation-benchmark](https://www.kaggle.com/code/adnanhassnain/knowledge-distillation-benchmark)

Run in a notebook cell:
```python
!python main.py --mode full \
  --data-dir /kaggle/input/datasets/adnanhassnain/cifar-10-python \
  --teacher-cache /kaggle/working
```

### Locally

```bash
git clone https://github.com/adnaan512/knowledge-distillation-benchmark
cd knowledge-distillation-benchmark
pip install -r requirements.txt

# Quick demo (no download, mock data)
python main.py --mode demo

# Full run on CIFAR-10
python main.py --mode full
```

**CLI options:**

| Flag | Default | Description |
|------|---------|-------------|
| `--mode` | `demo` | `demo` = mock data, `full` = real CIFAR-10 |
| `--epochs` | 2 | Training epochs per run |
| `--max-runs` | 5 | Max method × variant combinations to run |
| `--data-dir` | `./data` | Where to load CIFAR-10 from |
| `--teacher-cache` | `.` | Writable directory to save teacher checkpoint |
| `--method` | `all` | `response`, `feature`, `relation`, or `all` |
| `--compression` | `all` | `full`, `half`, `quarter`, or `all` |

---

## Project Structure

```
knowledge-distillation-benchmark/
├── main.py                          # CLI entry point
├── src/
│   ├── data/dataset_loader.py       # CIFAR-10 loader
│   ├── models/
│   │   ├── teacher.py               # ResNet-50 (frozen backbone)
│   │   └── student.py               # MobileNetV2 (3 compression variants)
│   ├── distillation/
│   │   ├── response_based.py        # Soft-label KL divergence
│   │   ├── feature_based.py         # Intermediate layer MSE
│   │   └── relation_based.py        # Pairwise distance matching
│   ├── benchmark/evaluator.py       # Accuracy, latency, efficiency score
│   └── reporting/report_generator.py# HTML report
├── examples/run_demo.py             # Quick demo (no download needed)
├── requirements.txt
└── CITATION.cff
```

---

## References

1. Hinton et al. (2015). *Distilling the Knowledge in a Neural Network*. https://arxiv.org/abs/1503.02531  
2. Romero et al. (2015). *FitNets: Hints for Thin Deep Nets*. https://arxiv.org/abs/1412.6550  
3. Park et al. (2019). *Relational Knowledge Distillation*. https://arxiv.org/abs/1904.05068  

---

## Author

**Adnan Hassnain** | BS CS, NUST Pakistan  
GitHub: [github.com/adnaan512](https://github.com/adnaan512)

---

## License

MIT — see [LICENSE](LICENSE).
