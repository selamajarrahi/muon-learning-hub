# Muon is Scalable for LLM Training

> **Source**: [arXiv:2502.16982](https://arxiv.org/abs/2502.16982)  
> **Authors**: Jingyuan Liu, Jianlin Su, et al.  
> **Date**: February 2025  
> **Reading Time**: 15 minutes

## 📋 Overview

This paper from the Jianlin Su research group investigates Muon's scaling properties when training large language models. While Muon shows significant speedups on small models (NanoGPT benchmarks), the question remained: **do these gains persist at scale?**

## 🎯 Key Questions Addressed

1. Does Muon's advantage diminish with model size?
2. How does Muon scale with training tokens?
3. What modifications are needed for large-scale distributed training?
4. How does Muon interact with muP (maximal update parameterization)?

## 📊 Main Findings

### Finding 1: Gains Diminish with Scale (Initially)

```
Performance Gap (Muon vs AdamW):

Model Size    Gap in Training Efficiency
125M          35% faster
350M          28% faster
760M          22% faster
1.3B          18% faster
7B            12% faster
```

**Interpretation**: The relative advantage decreases but remains positive.

### Finding 2: But Absolute Savings Increase

```
Compute Savings:

125M:  0.5 A100-hours saved
1.3B:  15 A100-hours saved
7B:    200+ A100-hours saved
```

Even though the percentage gain is smaller, the absolute savings grow significantly.

### Finding 3: Modified Muon for Scale

The paper proposes **Muon-Scale**, with three modifications:

1. **Per-Layer Learning Rate**: Scale LR by layer depth
2. **Gradient Accumulation Awareness**: Handle micro-batching properly
3. **Distributed Newton-Schulz**: Efficient all-reduce for orthogonalization

```python
class MuonScale:
    def __init__(self, params, base_lr=0.02, num_layers=32):
        self.base_lr = base_lr
        # Per-layer scaling
        self.layer_lrs = [
            base_lr * (0.5 + 0.5 * i / num_layers)
            for i in range(num_layers)
        ]
```

## 🔬 Experimental Setup

### Models Tested
- GPT architecture, various sizes (125M to 7B)
- Standard hyperparameters except optimizer

### Training Details
| Size | Tokens | GPUs | Time |
|------|--------|------|------|
| 125M | 10B | 8x A100 | 2h |
| 1.3B | 100B | 32x A100 | 2d |
| 7B | 300B | 128x A100 | 5d |

### Baselines
- AdamW with optimal hyperparameters
- AdamW + muP
- Shampoo (for comparison)

## 📈 Results

### Final Loss Comparison (7B model, 300B tokens)

| Optimizer | Final Loss | Training Time |
|-----------|------------|---------------|
| AdamW | 2.42 | 100% |
| AdamW + muP | 2.38 | 100% |
| Muon | 2.35 | 98% |
| Muon-Scale | 2.32 | 95% |

### Scaling Curves

```
Validation Loss vs Training FLOPs

        │
   2.6 ─┤   AdamW
        │     ╲
   2.5 ─┤      ╲
        │       ╲     Muon
   2.4 ─┤        ╲      ╲
        │         ╲      ╲
   2.3 ─┤          ╲──────╲
        │           ╲      ╲
   2.2 ─┤            ╲──────╲  Muon-Scale
        │
        └────────────────────────
             10^21      10^22  FLOPs
```

## 💡 Key Insights

### Insight 1: Layer-Wise LR is Critical at Scale

At small scales, uniform LR works. At large scales, early layers need lower LR to maintain stability, while later layers benefit from higher LR.

### Insight 2: Newton-Schulz Communication

Standard implementation of Newton-Schulz requires the full gradient matrix. In distributed training, this can be a bottleneck. The paper introduces a communication-efficient variant.

### Insight 3: Muon + muP Synergy

Combining Muon with muP (maximal update parameterization) provides additional benefits, allowing hyperparameters to transfer from small to large models.

## 🛠️ Practical Recommendations

### For Models < 1B Parameters
Use standard Muon with defaults:
```python
optimizer = Muon(muon_params, lr=0.02, momentum=0.95)
```

### For Models > 1B Parameters
Use Muon-Scale:
```python
optimizer = MuonScale(
    muon_params,
    base_lr=0.01,  # Lower base LR
    num_layers=32,
    layer_lr_scale='linear',
    distributed=True
)
```

### General Guidelines
1. Scale LR ∝ 1/sqrt(width) as model grows
2. Use gradient accumulation carefully
3. Monitor per-layer gradient norms
4. Consider Muon only for 2D hidden weights (as always)

## 📉 Limitations Acknowledged

1. **Compute cost of study**: Full scaling studies are expensive
2. **Architecture specificity**: Results on GPT; may differ for MoE, etc.
3. **Training duration**: Gains may change with longer training

## 🔗 Connection to Other Work

- **Kimi K2 (1T params)**: Also used Muon at scale, with additional modifications
- **Practical Efficiency of Muon (Essential AI)**: Complementary study on hyperparameter transfer
- **Old Optimizer New Norm**: Theoretical foundation

## 📄 Citation

```bibtex
@article{liu2025muonscalable,
  title={Muon is Scalable for LLM Training},
  author={Liu, Jingyuan and Su, Jianlin and others},
  journal={arXiv preprint arXiv:2502.16982},
  year={2025}
}
```

## 📚 Further Reading

- [Hyperparameter Tuning Guide](../docs/hyperparameter-tuning.md)
- [Kimi K2 Paper](05-kimi-k2.md)
- [Training at Any Scale](06-training-any-scale.md)

---

*Added: February 2026*
