#  Knowledge Distillation Benchmark

<p align="center">
  <a href="https://www.kaggle.com/code/adnanhassnain/knowledge-distillation-benchmark">
    <img src="https://img.shields.io/badge/Kaggle-Notebook-20BEFF?style=for-the-badge&logo=kaggle&logoColor=white" />
  </a>
  <img src="https://img.shields.io/badge/Python-3.10%2B-blue?style=for-the-badge&logo=python&logoColor=white" />
  <img src="https://img.shields.io/badge/PyTorch-2.0%2B-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white" />
  <img src="https://img.shields.io/badge/License-MIT-green?style=for-the-badge" />
</p>

<p align="center">
  <b>Can a 2MB student model keep 86% of a 94MB teacher's intelligence?</b><br/>
  A systematic benchmark of three knowledge distillation strategies on CIFAR-10.
</p>

---

##  The Problem

ResNet-50 is accurate but heavy — 94 MB, 23.5M parameters. You can't ship that to a phone or an edge device.

**Knowledge Distillation** solves this by training a small *student* model to mimic the large *teacher*  transferring its "knowledge" without copying its size.

This project benchmarks **three distillation strategies** to answer: *which one is best, and how much can you compress before accuracy collapses?*

---

##  Methods

| Method | Core Idea |
|--------|-----------|
| **Response-based** | Student mimics teacher's *output probabilities* (soft labels carry richer signal than one-hot labels) |
| **Feature-based** | Student mimics teacher's *internal layer representations* (not just what it thinks, but how it thinks) |
| **Relation-based** | Student mimics *pairwise relationships* between samples in the teacher's embedding space |

---

## Results

> Full run on CIFAR-10 — Teacher (ResNet-50): **78.7% accuracy**, 23.5M params, 94.1 MB

| Rank | Method | Variant | Accuracy | Compression | Retained | Score |
|------|--------|---------|----------|-------------|----------|-------|
| 🥇 1 | Response | full | **67.7%** | 10.5× | **86.0%** | **0.310** |
| 🥈 2 | Feature | full | 66.8% | 10.5× | 84.9% | 0.303 |
| 3 | Response | half | 20.2% | 33.6× | 25.7% | 0.093 |
| 4 | Response | quarter | 17.8% | 57.5× | 22.7% | 0.083 |
| 5 | Feature | half | 17.4% | 33.6× | 22.1% | 0.079 |

**Accuracy retained at `full` compression (2 epochs):**
```
Response  ████████████████████████████████████████████  86.0%
Feature   ████████████████████████████████████████████  84.9%
```

---

##  Key Findings

**1. 10× compression with minimal accuracy loss is achievable.**
The `full` MobileNetV2 student (2.24M params, ~9MB) retains **86% of the teacher's accuracy** in just 2 training epochs. For most deployment scenarios, this trade-off is entirely acceptable.

**2. Response-based and Feature-based distillation are neck-and-neck.**
Both methods score within 1% accuracy of each other on the `full` student. This suggests that for standard compression ratios, the simpler response-based method (soft labels only) is sufficient — no need for the added complexity of feature adapters.

**3. Aggressive compression (33×, 57×) needs more training epochs.**
The `half` and `quarter` variants (~0.4–0.7M params) collapsed to near-random (~17–20%) with only 2 epochs. Heavily compressed models need significantly more training time to converge. This is a known limitation at extreme compression ratios.

---

## Run It Yourself

### On Kaggle (GPU, no setup needed)
 [**Open Notebook**](https://www.kaggle.com/code/adnanhassnain/knowledge-distillation-benchmark)

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

# Quick demo (no download, mock data, ~60s)
python main.py --mode demo

# Full CIFAR-10 benchmark
python main.py --mode full
```

**Key options:**

```
--epochs N        Training epochs per run        (default: 2)
--max-runs N      Cap total training combinations (default: 5)
--method X        response / feature / relation / all
--compression X   full / half / quarter / all
--teacher-cache   Where to save teacher weights (use /kaggle/working on Kaggle)
```

---

## Architecture

```
ResNet-50 Teacher (78.7% acc, 94 MB)
        │
        │  soft labels / features / relations
        ▼
  Distillation Layer
        │
        ▼
MobileNetV2 Student
  ├── full    → 2.24M params │ 10.5× smaller │ 67.7% acc
  ├── half    → 0.70M params │ 33.6× smaller │ needs more epochs
  └── quarter → 0.41M params │ 57.5× smaller │ needs more epochs
```

---

##  Structure

```
├── main.py                          # CLI start here
├── src/
│   ├── models/teacher.py            # ResNet-50 (frozen backbone + fine-tuned head)
│   ├── models/student.py            # MobileNetV2 (3 width variants)
│   ├── distillation/
│   │   ├── response_based.py        # Soft KL + hard CE loss
│   │   ├── feature_based.py         # Intermediate MSE + channel adapter
│   │   └── relation_based.py        # Pairwise distance matching
│   ├── data/dataset_loader.py       # CIFAR-10 with train/val/test split
│   ├── benchmark/evaluator.py       # Accuracy, latency, efficiency score
│   └── reporting/report_generator.py# Self-contained HTML report
├── examples/run_demo.py             # Demo mode — no download needed
└── tests/                           # Unit tests for loss math + metrics
```

---

##  References

1. Hinton, Vinyals & Dean (2015). *Distilling the Knowledge in a Neural Network.* [arXiv:1503.02531](https://arxiv.org/abs/1503.02531)
2. Romero et al. (2015). *FitNets: Hints for Thin Deep Nets.* [arXiv:1412.6550](https://arxiv.org/abs/1412.6550)
3. Park et al. (2019). *Relational Knowledge Distillation.* [arXiv:1904.05068](https://arxiv.org/abs/1904.05068)

---

##  Author

**Adnan Hassnain** · BS CS, NUST Pakistan  
[github.com/adnaan512](https://github.com/adnaan512) · [Kaggle Notebook](https://www.kaggle.com/code/adnanhassnain/knowledge-distillation-benchmark)

---

## License

MIT — see [LICENSE](LICENSE).
