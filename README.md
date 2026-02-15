# 🌀 Muon Learning Hub

> The ultimate resource for understanding Muon — the neural network optimizer behind NanoGPT speed records and Kimi K2's 1T parameter training.

## 🎯 What is Muon?

**Muon** (MomentUm Orthogonalized by Newton-Schulz) is a next-generation optimizer specifically designed for 2D parameters in neural network hidden layers. Unlike Adam (which treats every parameter independently), Muon *sees the matrix* — exploiting the structure of linear transformations to achieve faster, more stable training.

### Why Muon Matters
- 🏆 **1.35x faster** than AdamW on NanoGPT speedruns
- 🚀 Scaled to **1 trillion parameters** (Kimi K2)
- 📐 Derived from **exact theoretical principles** (not heuristics like Adam)
- ⚡ **<1% FLOP overhead** vs standard optimizers

---

## 📚 Learning Path

### Level 1: The Core Idea (Start Here)
| Resource | Author | Key Takeaway |
|----------|--------|--------------|
| [Keller Jordan's Muon Post](articles/01-keller-jordan-muon.md) | Keller Jordan | Practical definition, results, PyTorch implementation |
| [Understanding Muon Ch.1](articles/02-laker-newhouse-understanding.md) | Laker Newhouse | "Enter the Matrix" — why matrix norms matter |

### Level 2: The Theory
| Resource | Author | Key Takeaway |
|----------|--------|--------------|
| [Deriving Muon](articles/03-bernstein-deriving-muon.md) | Jeremy Bernstein | 4-step derivation from first principles |
| [Old Optimizer New Norm](articles/04-anthology.md) | Bernstein & Newhouse | Adam/Shampoo/Prodigy as steepest descent under norms |

### Level 3: Scaling & Applications
| Resource | Author | Key Takeaway |
|----------|--------|--------------|
| [Kimi K2 Paper](articles/05-kimi-k2.md) | Moonshot AI | Muon at 1T scale, agentic training |
| [Training at Any Scale](articles/06-training-any-scale.md) | Pethick et al. | Modern optimization survey including Muon |

---

## 🧠 Core Concepts

### The Fundamental Insight
Adam looks at **individual entries** of the gradient. Muon looks at the **entire matrix**.

```
Adam:  W ← W - η · sign(G)           # elementwise
Muon:  W ← W - η · UV^T              # where G = UΣV^T (SVD)
```

### Why Orthogonalize?

Gradient updates from Adam/SGD are often **nearly low-rank** — dominated by a few directions. Orthogonalization (replacing singular values with 1s) gives equal weight to all directions, letting the model learn "rare directions" it would otherwise miss.

### The Newton-Schulz Trick

Computing SVD is expensive. Instead, Muon uses **odd polynomial iterations** that converge to orthogonalization:

```python
# The magic polynomial (5 iterations)
def newtonschulz5(G, steps=5):
    a, b, c = (3.4445, -4.7750, 2.0315)
    X = G / G.norm()
    for _ in range(steps):
        A = X @ X.T
        B = b * A + c * A @ A
        X = a * X + B @ X
    return X
```

---

## 💻 Quick Start

```python
# pip install muon-pytorch
from muon import Muon

# Use Muon for 2D hidden layer params, AdamW for everything else
optimizer = Muon(
    muon_params=model.hidden_layers.parameters(),
    lr=0.02,
    momentum=0.95,
    adamw_params=model.embeddings.parameters(),  # 1D/embedding params
    adamw_lr=3e-4
)
```

**Key Usage Rules:**
- ✅ Use Muon for: Linear layers, attention projections, MLP weights
- ❌ Use AdamW for: Embeddings, LayerNorm, biases, 1D params

---

## 📊 Benchmark Results

| Task | Muon | AdamW | Speedup |
|------|------|-------|---------|
| NanoGPT (124M) val=3.28 | 2.92 A100-hrs | 3.94 A100-hrs | **1.35x** |
| CIFAR-10 94% | 2.6 A100-sec | 3.3 A100-sec | **1.27x** |
| GPT-2 XL HellaSwag | 10 8xH100-hrs | 13.3 8xH100-hrs | **1.33x** |

---

## 🔬 Key Papers

1. **Bernstein & Newhouse (2024)** - "Old Optimizer, New Norm: An Anthology" - [arXiv:2409.20325](https://arxiv.org/abs/2409.20325)
2. **Kimi K2 (2025)** - "Open Agentic Intelligence" - [arXiv:2507.20534](https://arxiv.org/abs/2507.20534)
3. **Pethick et al. (2025)** - "Training Neural Networks at Any Scale" - [arXiv:2511.11163](https://arxiv.org/abs/2511.11163)

---

## 🗂️ Repository Structure

```
muon-learning-hub/
├── README.md                 # You are here
├── articles/                 # Annotated article summaries
│   ├── 01-keller-jordan-muon.md
│   ├── 02-laker-newhouse-understanding.md
│   ├── 03-bernstein-deriving-muon.md
│   ├── 04-anthology.md
│   ├── 05-kimi-k2.md
│   └── 06-training-any-scale.md
├── notes/                    # Your personal study notes
│   └── GLOSSARY.md
└── code-examples/            # Runnable implementations
    └── muon_minimal.py
```

---

## 🎓 Prerequisites

To fully understand Muon, you should be comfortable with:
- Linear algebra (SVD, matrix norms, orthogonal matrices)
- Neural network basics (forward/backward pass, gradient descent)
- PyTorch fundamentals

---

## 🔗 External Links

- [Official Muon Repo](https://github.com/KellerJordan/Muon)
- [NanoGPT Speedrun Records](https://github.com/KellerJordan/modded-nanogpt)
- [Modula Systems](https://modula.systems) - The broader research program
- [Jeremy Bernstein's Site](https://jeremybernste.in)
- [Laker Newhouse's Site](https://lakernewhouse.com)

---

*Created with 🐈‍⬛ by codmire_'s assistant*
