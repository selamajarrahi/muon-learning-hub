# ⚔️ Muon vs Adam: Complete Comparison

> A detailed head-to-head comparison of the two optimizers

## Quick Comparison Table

| Aspect | Muon | Adam/AdamW |
|--------|------|------------|
| **Philosophy** | Matrix-aware optimization | Element-wise optimization |
| **Gradient View** | Sees entire matrix structure | Treats each param independently |
| **Update Direction** | Orthogonalized (equal directions) | Scaled by gradient history |
| **Theoretical Basis** | Steepest descent under spectral norm | Adaptive learning rates |
| **Best For** | Linear layers, attention, MLPs | Embeddings, LayerNorm, 1D params |
| **Compute Overhead** | ~1% extra FLOPs | Baseline |
| **Memory** | Similar (stores momentum) | Stores momentum + variance |
| **Hyperparameters** | lr=0.02, momentum=0.95 | lr=3e-4, β₁=0.9, β₂=0.999, ε=1e-8 |

---

## The Core Difference

### Adam's Approach: Element-wise

```
┌─────────────────────────────────────────────────────────────┐
│                    ADAM UPDATE                               │
│                                                              │
│  Weight Matrix W:    Gradient G:       Update ΔW:           │
│  ┌───┬───┬───┐      ┌───┬───┬───┐     ┌───┬───┬───┐        │
│  │w₁₁│w₁₂│w₁₃│      │g₁₁│g₁₂│g₁₃│     │Δ₁₁│Δ₁₂│Δ₁₃│        │
│  ├───┼───┼───┤  →   ├───┼───┼───┤  →  ├───┼───┼───┤        │
│  │w₂₁│w₂₂│w₂₃│      │g₂₁│g₂₂│g₂₃│     │Δ₂₁│Δ₂₂│Δ₂₃│        │
│  └───┴───┴───┘      └───┴───┴───┘     └───┴───┴───┘        │
│                                                              │
│  Each Δᵢⱼ computed INDEPENDENTLY based on gᵢⱼ history       │
│                                                              │
│  Δᵢⱼ = -lr × m̂ᵢⱼ / (√v̂ᵢⱼ + ε)                              │
│                                                              │
│  ❌ No awareness that these form a LINEAR TRANSFORMATION    │
└─────────────────────────────────────────────────────────────┘
```

### Muon's Approach: Matrix-aware

```
┌─────────────────────────────────────────────────────────────┐
│                    MUON UPDATE                               │
│                                                              │
│  Gradient G:           SVD Decomposition:                    │
│  ┌───────────┐         ┌───┐ ┌─────┐ ┌───┐                  │
│  │           │    =    │   │ │σ₁   │ │   │                  │
│  │     G     │         │ U │ │  σ₂ │ │V^T│                  │
│  │           │         │   │ │   σ₃│ │   │                  │
│  └───────────┘         └───┘ └─────┘ └───┘                  │
│                              ↓                               │
│                        Replace with 1s                       │
│                              ↓                               │
│  Update ΔW:            ┌───┐ ┌─────┐ ┌───┐                  │
│  ┌───────────┐         │   │ │1    │ │   │                  │
│  │           │    =    │ U │ │  1  │ │V^T│  = U × V^T       │
│  │   U V^T   │         │   │ │   1 │ │   │                  │
│  └───────────┘         └───┘ └─────┘ └───┘                  │
│                                                              │
│  ✅ Preserves the DIRECTIONAL structure of the gradient     │
│  ✅ Gives EQUAL weight to all singular directions           │
└─────────────────────────────────────────────────────────────┘
```

---

## Why This Matters

### The "Rare Direction" Problem

```
Scenario: Your gradient is dominated by one direction

Adam sees:                         Muon sees:
┌─────────────────────┐           ┌─────────────────────┐
│                     │           │                     │
│  Large gradient     │           │   Large gradient    │
│  ────────────────►  │           │   ────────────────► │
│                     │           │   Small gradient    │
│  Tiny gradient ·    │           │   ────────────────► │
│                     │           │   (amplified!)      │
└─────────────────────┘           └─────────────────────┘

Adam's update:                    Muon's update:
   Moves mostly in                   Moves EQUALLY in
   large gradient direction          both directions
```

**Result:** Muon learns "rare but important" features that Adam might miss or learn slowly.

---

## Performance Comparison

### Training Speed

| Benchmark | Muon Time | Adam Time | Speedup |
|-----------|-----------|-----------|---------|
| NanoGPT 124M (val=3.28) | 2.92 A100-hrs | 3.94 A100-hrs | **1.35x** |
| CIFAR-10 94% accuracy | 2.6 A100-sec | 3.3 A100-sec | **1.27x** |
| GPT-2 XL HellaSwag | 10 8xH100-hrs | 13.3 8xH100-hrs | **1.33x** |

### Hyperparameter Sensitivity

```
Adam:                               Muon:
┌────────────────────────┐         ┌────────────────────────┐
│ lr: 1e-4 to 1e-3       │         │ lr: 0.01 to 0.05       │
│ (sensitive!)           │         │ (more robust!)         │
│                        │         │                        │
│ β₁: 0.9 (usually fixed)│         │ momentum: 0.95         │
│ β₂: 0.95-0.999         │         │ (usually fixed)        │
│ ε: 1e-8 (important!)   │         │                        │
│                        │         │ NS iters: 5 (fixed)    │
│ 4 hyperparams to tune  │         │ 2 hyperparams to tune  │
└────────────────────────┘         └────────────────────────┘
```

---

## When to Use Each

### Use Muon For ✅

| Parameter Type | Example | Why |
|---------------|---------|-----|
| Linear layers | `nn.Linear(in, out)` | These ARE matrix operations |
| Attention QKV | `W_Q, W_K, W_V` | Matrix projections |
| MLP weights | FFN hidden layers | Dense transformations |
| Conv2d (reshaped) | Convolutional kernels | Can treat as 2D |

### Use Adam For ✅

| Parameter Type | Example | Why |
|---------------|---------|-----|
| Embeddings | Token/position embeds | Not matrix operations |
| LayerNorm | Scale/shift params | 1D vectors |
| Biases | All bias terms | 1D vectors |
| Small params | < 256 elements | Overhead not worth it |

---

## Memory Usage

```
                Adam                           Muon
┌─────────────────────────────┐   ┌─────────────────────────────┐
│                             │   │                             │
│  For each parameter θ:      │   │  For each parameter θ:      │
│                             │   │                             │
│  ┌─────┐  ← Momentum (m)    │   │  ┌─────┐  ← Momentum        │
│  │ θ   │                    │   │  │ θ   │                    │
│  └─────┘                    │   │  └─────┘                    │
│  ┌─────┐  ← Variance (v)    │   │                             │
│  │     │                    │   │  (no variance needed!)      │
│  └─────┘                    │   │                             │
│                             │   │                             │
│  Memory: 3x param size      │   │  Memory: 2x param size      │
│                             │   │                             │
└─────────────────────────────┘   └─────────────────────────────┘
```

---

## Hybrid Approach (Recommended)

```python
from muon import Muon

# Split parameters by type
muon_params = []
adam_params = []

for name, param in model.named_parameters():
    if param.ndim == 2 and param.shape[0] >= 256:
        muon_params.append(param)  # Use Muon
    else:
        adam_params.append(param)  # Use Adam

optimizer = Muon(
    muon_params=muon_params,
    lr=0.02,
    momentum=0.95,
    adamw_params=adam_params,
    adamw_lr=3e-4,
)
```

---

## Historical Context

```
Timeline:
─────────────────────────────────────────────────────────────────►
     │                    │                    │
     │                    │                    │
   2014                 2018                 2024
   Adam                 AdamW               Muon
   
   "Adaptive           "Weight decay        "Matrix-aware
    moment              done right"          optimization"
    estimation"
```

| Era | Insight |
|-----|---------|
| Pre-Adam | SGD + momentum works, but learning rate tuning is painful |
| Adam (2014) | Adapt learning rates per-parameter using gradient moments |
| AdamW (2017) | Fix weight decay (decouple from gradient scaling) |
| Muon (2024) | For matrices, work in matrix space, not element space |

---

## The Bottom Line

> **Adam:** "Treat every number independently, adapt based on history"
> 
> **Muon:** "This is a *matrix* — use matrix structure to optimize better"

For modern transformers with large linear layers, Muon consistently trains **25-35% faster** while requiring **fewer hyperparameters** to tune.

---

📚 **Further Reading:**
- [Understanding Muon](../articles/02-laker-newhouse-understanding.md)
- [Implementation Checklist](implementation-checklist.md)
- [Common Mistakes](common-mistakes.md)
