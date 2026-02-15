# 🌀 Muon Learning Hub

> The ultimate resource for understanding Muon — the neural network optimizer behind NanoGPT speed records and Kimi K2's 1T parameter training.

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![PRs Welcome](https://img.shields.io/badge/PRs-welcome-brightgreen.svg)](CONTRIBUTING.md)

## 🎯 What is Muon?

**Muon** (MomentUm Orthogonalized by Newton-Schulz) is a next-generation optimizer specifically designed for 2D parameters in neural network hidden layers. Unlike Adam (which treats every parameter independently), Muon *sees the matrix* — exploiting the structure of linear transformations to achieve faster, more stable training.

### Why Muon Matters
- 🏆 **1.35x faster** than AdamW on NanoGPT speedruns
- 🚀 Scaled to **1 trillion parameters** (Kimi K2)
- 📐 Derived from **exact theoretical principles** (not heuristics like Adam)
- ⚡ **<1% FLOP overhead** vs standard optimizers

---

## 🗺️ Learning Roadmap

```
┌─────────────────────────────────────────────────────────────────────────┐
│                         YOUR MUON JOURNEY                                │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                          │
│   🌱 BEGINNER                                                            │
│   │                                                                      │
│   ├── Read: What is Muon? (this page)                                   │
│   ├── Read: Keller Jordan's Muon Post                                   │
│   └── Try: Quick Start code below                                       │
│       │                                                                  │
│       ▼                                                                  │
│   📚 INTERMEDIATE                                                        │
│   │                                                                      │
│   ├── Read: Understanding Muon (Enter the Matrix)                       │
│   ├── Read: Newton-Schulz Diagram (docs/newton-schulz-diagram.md)       │
│   ├── Study: Muon vs Adam Comparison (docs/muon-vs-adam.md)             │
│   └── Take: Self-Assessment Quiz (docs/learning-quiz.md)                │
│       │                                                                  │
│       ▼                                                                  │
│   🔬 ADVANCED                                                            │
│   │                                                                      │
│   ├── Read: Deriving Muon (first principles)                            │
│   ├── Read: Old Optimizer New Norm (anthology paper)                    │
│   ├── Study: Papers to Read Next (docs/papers-to-read.md)               │
│   └── Review: Real-World Deployments (docs/real-world-deployments.md)   │
│       │                                                                  │
│       ▼                                                                  │
│   🚀 EXPERT                                                              │
│   │                                                                      │
│   ├── Follow: Implementation Checklist (docs/implementation-checklist.md)│
│   ├── Avoid: Common Mistakes (docs/common-mistakes.md)                   │
│   ├── Code: JAX Implementation (code-examples/muon_jax.py)              │
│   └── Contribute! (CONTRIBUTING.md)                                     │
│                                                                          │
└─────────────────────────────────────────────────────────────────────────┘
```

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
| [Muon is Scalable](articles/07-muon-scalable.md) | Liu, Su et al. | **NEW**: Scaling analysis to 7B+ |
| [Practical Efficiency](articles/08-practical-efficiency.md) | Essential AI | **NEW**: Production deployment guide |

---

## 📖 Additional Resources

| Topic | Link | Description |
|-------|------|-------------|
| 🔄 Newton-Schulz Explained | [docs/newton-schulz-diagram.md](docs/newton-schulz-diagram.md) | Visual guide to the orthogonalization algorithm |
| ⚔️ Muon vs Adam | [docs/muon-vs-adam.md](docs/muon-vs-adam.md) | Detailed comparison table and analysis |
| ❓ FAQ | [docs/faq.md](docs/faq.md) | Frequently asked questions |
| ⚠️ Common Mistakes | [docs/common-mistakes.md](docs/common-mistakes.md) | Learn from others' errors |
| ✅ Implementation Checklist | [docs/implementation-checklist.md](docs/implementation-checklist.md) | Step-by-step adoption guide |
| 📚 Papers to Read | [docs/papers-to-read.md](docs/papers-to-read.md) | Curated reading list |
| 🚀 Real-World Deployments | [docs/real-world-deployments.md](docs/real-world-deployments.md) | Kimi K2, NanoGPT, and more |
| 🧪 Self-Assessment Quiz | [docs/learning-quiz.md](docs/learning-quiz.md) | Test your understanding |
| 🎛️ Hyperparameter Tuning | [docs/hyperparameter-tuning.md](docs/hyperparameter-tuning.md) | **NEW**: Complete tuning guide |
| 🔧 Troubleshooting | [docs/troubleshooting.md](docs/troubleshooting.md) | **NEW**: Debug common issues |
| ❌ When NOT to Use Muon | [docs/when-not-to-use-muon.md](docs/when-not-to-use-muon.md) | **NEW**: Know the limitations |
| 📄 Citations | [docs/citations.md](docs/citations.md) | **NEW**: BibTeX for all papers |
| 👥 Community Resources | [docs/community-resources.md](docs/community-resources.md) | **NEW**: Discords, forums, tools |

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

📐 **[See the full visual diagram →](docs/newton-schulz-diagram.md)**

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

📋 **[Full implementation checklist →](docs/implementation-checklist.md)**

---

## 📊 Benchmark Results

| Task | Muon | AdamW | Speedup |
|------|------|-------|---------|
| NanoGPT (124M) val=3.28 | 2.92 A100-hrs | 3.94 A100-hrs | **1.35x** |
| CIFAR-10 94% | 2.6 A100-sec | 3.3 A100-sec | **1.27x** |
| GPT-2 XL HellaSwag | 10 8xH100-hrs | 13.3 8xH100-hrs | **1.33x** |

🏆 **[View all records on NanoGPT Speedrun Leaderboard →](https://github.com/KellerJordan/modded-nanogpt#leaderboard)**

---

## 💻 Code Examples

| Framework | File | Description |
|-----------|------|-------------|
| PyTorch | [muon_minimal.py](code-examples/muon_minimal.py) | Minimal reference implementation |
| JAX/Optax | [muon_jax.py](code-examples/muon_jax.py) | Full Optax-style implementation |

---

## 🔬 Key Papers

1. **Bernstein & Newhouse (2024)** - "Old Optimizer, New Norm: An Anthology" - [arXiv:2409.20325](https://arxiv.org/abs/2409.20325)
2. **Kimi K2 (2025)** - "Open Agentic Intelligence" - [arXiv:2507.20534](https://arxiv.org/abs/2507.20534)
3. **Pethick et al. (2025)** - "Training Neural Networks at Any Scale" - [arXiv:2511.11163](https://arxiv.org/abs/2511.11163)

📚 **[Full reading list →](docs/papers-to-read.md)**

---

## 🗂️ Repository Structure

```
muon-learning-hub/
├── README.md                   # You are here
├── CONTRIBUTING.md             # How to contribute
├── articles/                   # Annotated article summaries
│   ├── 01-keller-jordan-muon.md
│   ├── 02-laker-newhouse-understanding.md
│   ├── 03-bernstein-deriving-muon.md
│   ├── 04-anthology.md
│   ├── 05-kimi-k2.md
│   └── 06-training-any-scale.md
├── docs/                       # In-depth documentation
│   ├── newton-schulz-diagram.md   # Visual explanation
│   ├── muon-vs-adam.md            # Comparison guide
│   ├── faq.md                     # FAQ
│   ├── common-mistakes.md         # Pitfalls to avoid
│   ├── implementation-checklist.md # Adoption guide
│   ├── papers-to-read.md          # Reading list
│   ├── real-world-deployments.md  # Production uses
│   └── learning-quiz.md           # Self-assessment
├── code-examples/              # Runnable implementations
│   ├── muon_minimal.py            # PyTorch reference
│   └── muon_jax.py                # JAX/Optax version
└── notes/                      # Study notes
    └── GLOSSARY.md
```

---

## 🎓 Prerequisites

To fully understand Muon, you should be comfortable with:
- Linear algebra (SVD, matrix norms, orthogonal matrices)
- Neural network basics (forward/backward pass, gradient descent)
- PyTorch fundamentals

---

## 🔗 External Links

- [Official Muon Repo](https://github.com/KellerJordan/Muon) - Reference implementation
- [NanoGPT Speedrun Records](https://github.com/KellerJordan/modded-nanogpt) - Leaderboard & code
- [Modula Systems](https://modula.systems) - The broader research program
- [Jeremy Bernstein's Site](https://jeremybernste.in) - Theory & derivations
- [Laker Newhouse's Site](https://lakernewhouse.com) - Intuitive explanations

---

## 🤝 Contributing

We welcome contributions! See [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

Ideas for contributions:
- Additional framework implementations (TensorFlow, Flax)
- Benchmark results from your experiments
- Corrections and clarifications
- Translations

---

*Created with 🐈‍⬛ by codmire_'s assistant*
