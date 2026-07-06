# Research Methodology

## 1. Problem Statement

Knowledge distillation addresses the deployment gap in deep learning: research models
are large, accurate, and slow; production models must be small, fast, and almost as
accurate. This project systematically compares three distillation strategies across
three compression levels to find the best accuracy-efficiency operating point.

---

## 2. Distillation Methods

### 2.1 Response-Based Distillation (Hinton et al. 2015)

**Core idea:** Train the student to match the teacher's full output probability
distribution, not just the argmax label.

**Loss function:**

```
L = α · CE(z_s, y) + (1 - α) · T² · KL( σ(z_s/T) || σ(z_t/T) )
```

Where:
- `z_s`, `z_t` — student and teacher logits
- `y` — one-hot ground truth labels
- `T` — temperature parameter
- `α` — mixing coefficient
- `σ(·)` — softmax function
- `CE` — cross-entropy loss
- `KL` — KL divergence

**Temperature scaling:** Dividing logits by T before softmax "softens" the
distribution. At T=1, the distribution is standard (peaked). At T=4, the
non-argmax probabilities are amplified, revealing the "dark knowledge" about
inter-class similarities. The T² factor compensates for gradient shrinkage
caused by the division.

**Hyperparameter sweep:**
- T ∈ {1, 2, 4, 8, 16}
- α ∈ {0.1, 0.5, 0.9}
- Best configuration selected by validation accuracy.

---

### 2.2 Feature-Based Distillation (Romero et al. 2015 — FitNets)

**Core idea:** Force the student to produce intermediate representations similar
to the teacher's at an intermediate layer (layer 3 of ResNet-50).

**Loss function:**

```
L = CE(z_s, y) + λ · MSE( Adapter(f_s), f_t )
```

Where:
- `f_s` — student intermediate feature map (B, C_s, H, W)
- `f_t` — teacher intermediate feature map (B, C_t, H, W)
- `Adapter` — 1×1 Conv2d projecting C_s → C_t channels
- `λ` — feature loss weight (default: 0.5)

**Why layer 3?** ResNet-50 layer 3 outputs 1024-channel feature maps that encode
mid-level semantics: shapes, textures, object parts. Earlier layers (layer 1–2) are
too generic; layer 4 is already committing to class-specific representations.

**Adapter network:** The teacher's channel dimension (1024) differs from the
compressed student's. A 1×1 convolution with BatchNorm projects the student's
features into the teacher's space. The adapter is trained alongside the student
head — it learns the optimal linear remapping.

---

### 2.3 Relation-Based Distillation (Park et al. 2019 — RKD)

**Core idea:** Transfer the structural geometry of the teacher's embedding space,
not individual point predictions.

**Loss function:**

```
L = CE(z_s, y) + λ · MSE( D(E_s), D(E_t) )
```

Where:
- `E_s`, `E_t` — student and teacher penultimate embeddings, shape (B, D)
- `D(E)` — normalized pairwise distance matrix, shape (B, B)
- `D_{ij} = ||e_i - e_j||_2 / max_distance`

**Pairwise distance computation:**

```python
# Efficient implementation using expansion:
sq_norm = (E ** 2).sum(dim=1, keepdim=True)
dist_sq = sq_norm + sq_norm.T - 2 * E @ E.T
distances = dist_sq.clamp(min=0).sqrt()
distances = distances / distances.max()  # normalize to [0, 1]
```

**Why this works:** If samples A and B are more similar to each other than to C
in the teacher's embedding space, the student should reproduce that relationship
even if it can't reproduce the exact feature values.

---

## 3. Model Architectures

### Teacher: ResNet-50
- Pre-trained on ImageNet-1k (torchvision weights)
- Final FC replaced: Linear(2048, 10) for CIFAR-10
- **All layers frozen** except the final classifier
- Baseline accuracy target: ~70–75% on CIFAR-10 (frozen backbone limitation)

### Student: MobileNetV2 Variants

| Variant  | Width Mult | Parameters | Compression vs Teacher |
|----------|-----------|------------|------------------------|
| full     | 1.0       | ~3.4M      | ~7×                    |
| half     | 0.5       | ~1.4M      | ~17×                   |
| quarter  | 0.35      | ~0.9M      | ~26×                   |

Width multiplier uniformly scales all channel counts. Only the classifier head
is trained (backbone frozen). This reflects a realistic deployment scenario where
we have a pre-trained efficient backbone and just need to adapt the final layer.

---

## 4. Evaluation Metrics

### 4.1 Accuracy
Standard classification accuracy on the held-out CIFAR-10 test set (10,000 samples).

### 4.2 Inference Time (ms)
Average time for a single forward pass, measured over 100 runs:
```
inference_time_ms = mean(100 forward passes) × 1000
```
First 5 passes are discarded as warmup. Measured with `time.perf_counter()`.

### 4.3 Model Size (MB)
```
size_mb = Σ(param.numel() × 4 bytes) / 1,000,000
```
Counts float32 parameters only. Represents in-memory footprint.

### 4.4 Compression Ratio
```
compression_ratio = teacher_params / student_params
```
A ratio of 10× means the student has 10× fewer parameters.

### 4.5 Accuracy Retained (%)
```
accuracy_retained = (student_acc / teacher_acc) × 100
```
A retained accuracy of 95% means the student is 5% less accurate than the teacher.

### 4.6 Efficiency Score
```
efficiency_score = accuracy / log₂(inference_time_ms)
```
Rewards models that are both accurate and fast. Halving latency (same accuracy)
increases the score by `1/log₂(latency)`. The log₂ scale prevents raw speed
from completely dominating — a 0.01ms model at 50% accuracy should not win over
a 10ms model at 92% accuracy.

---

## 5. Dataset Splits

| Split | Samples | Purpose |
|-------|---------|---------|
| Train | 45,000  | Model training |
| Val   | 5,000   | Hyperparameter selection |
| Test  | 10,000  | Final accuracy reporting |

The validation split is carved from the official CIFAR-10 training set using
a fixed random seed (42) for reproducibility. The test set is **never** used
during hyperparameter search.

---

## 6. Reproducibility

- All random seeds fixed to 42 where applicable
- Pretrained weights from torchvision (pinned version)
- CPU-only: no CUDA non-determinism
- Temperature sweep selects best config by **validation** accuracy only
- Test accuracy reported once, at the end

---

## 7. References

1. Hinton, G., Vinyals, O., & Dean, J. (2015). *Distilling the Knowledge in a Neural Network*. NIPS Workshop.
2. Romero, A., Ballas, N., Kahou, S. E., Chassang, A., Gatta, C., & Bengio, Y. (2015). *FitNets: Hints for Thin Deep Nets*. ICLR.
3. Park, W., Kim, D., Lu, Y., & Cho, M. (2019). *Relational Knowledge Distillation*. CVPR.
4. Howard, A. G., et al. (2017). *MobileNets: Efficient Convolutional Neural Networks for Mobile Vision Applications*. arXiv:1704.04861.
