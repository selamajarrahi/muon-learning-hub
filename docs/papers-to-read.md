# 📚 Papers to Read Next

> Curated reading list for deepening your understanding of Muon and related optimization theory

## Core Muon Papers

### Must Read 🔥

| Paper | Authors | Year | Key Contribution |
|-------|---------|------|------------------|
| **[Muon: An optimizer for hidden layers in neural networks](https://arxiv.org/abs/2502.16982)** | Keller Jordan | 2025 | The original Muon paper |
| **[Old Optimizer, New Norm: An Anthology](https://arxiv.org/abs/2409.20325)** | Bernstein & Newhouse | 2024 | Adam/Shampoo/Muon as steepest descent under different norms |
| **[Kimi K2 Technical Report](https://arxiv.org/abs/2507.15816)** | Moonshot AI | 2025 | Muon at 1 trillion parameter scale |

### Deep Dives

| Paper | Authors | Year | Why Read It |
|-------|---------|------|-------------|
| **[Modular Duality in Deep Learning](https://arxiv.org/abs/2410.21265)** | Bernstein et al. | 2024 | The mathematical foundation |
| **[Deriving Muon](https://jeremybernste.in/writing/deriving-muon)** | Jeremy Bernstein | 2024 | 4-step derivation from first principles |
| **[Understanding Muon](https://lakernewhouse.github.io/muon/)** | Laker Newhouse | 2024 | "Enter the Matrix" - intuitive explanation |

---

## Prerequisite Papers

### Optimization Fundamentals

| Paper | Authors | Year | What You'll Learn |
|-------|---------|------|-------------------|
| **[Adam: A Method for Stochastic Optimization](https://arxiv.org/abs/1412.6980)** | Kingma & Ba | 2014 | The baseline optimizer |
| **[Decoupled Weight Decay Regularization](https://arxiv.org/abs/1711.05101)** | Loshchilov & Hutter | 2017 | AdamW and why weight decay matters |
| **[On the Variance of the Adaptive Learning Rate and Beyond](https://arxiv.org/abs/1908.03265)** | Liu et al. | 2019 | RAdam - variance issues in Adam |

### Matrix Methods

| Paper | Authors | Year | What You'll Learn |
|-------|---------|------|-------------------|
| **[Shampoo: Preconditioned Stochastic Tensor Optimization](https://arxiv.org/abs/1802.09568)** | Gupta et al. | 2018 | Kronecker-factored preconditioning |
| **[Scalable Second Order Optimization for Deep Learning](https://arxiv.org/abs/2002.09018)** | Anil et al. | 2020 | Distributed Shampoo |
| **[K-FAC: Kronecker-Factored Approximate Curvature](https://arxiv.org/abs/1503.05671)** | Martens & Grosse | 2015 | Natural gradient approximation |

---

## Related Topics

### Newton-Schulz & Matrix Functions

| Paper | Authors | Year | Relevance |
|-------|---------|------|-----------|
| **[Functions of Matrices: Theory and Computation](https://epubs.siam.org/doi/book/10.1137/1.9780898717778)** | Higham | 2008 | Textbook on matrix functions |
| **[A Schur-Newton Method for the Matrix Sign Function](https://www.jstor.org/stable/2157899)** | Kenney & Laub | 1991 | Newton-Schulz analysis |

### Large-Scale Training

| Paper | Authors | Year | Relevance |
|-------|---------|------|-----------|
| **[Scaling Laws for Neural Language Models](https://arxiv.org/abs/2001.08361)** | Kaplan et al. | 2020 | Understanding scale |
| **[Training Compute-Optimal Large Language Models](https://arxiv.org/abs/2203.15556)** | Hoffmann et al. | 2022 | Chinchilla scaling |
| **[Tensor Programs V: Tuning Large Neural Networks via Zero-Shot Hyperparameter Transfer](https://arxiv.org/abs/2203.03466)** | Yang et al. | 2022 | μP for scaling |

### Spectral Perspective

| Paper | Authors | Year | Relevance |
|-------|---------|------|-----------|
| **[Spectral Normalization for GANs](https://arxiv.org/abs/1802.05957)** | Miyato et al. | 2018 | Spectral norm in practice |
| **[Orthogonal Weight Normalization](https://arxiv.org/abs/1911.13063)** | Huang et al. | 2019 | Orthogonality in networks |

---

## Reading Order Suggestion

### Track 1: Quick Understanding (3 papers)
```
1. Keller Jordan's Muon Post (practical intro)
        ↓
2. Understanding Muon - Laker Newhouse (intuition)
        ↓
3. Old Optimizer New Norm (theoretical framework)
```

### Track 2: Full Theory (6 papers)
```
1. Adam (baseline)
        ↓
2. AdamW (fixing weight decay)
        ↓
3. Shampoo (matrix preconditioning)
        ↓
4. Old Optimizer New Norm (unified view)
        ↓
5. Deriving Muon (first principles)
        ↓
6. Kimi K2 (scaling to 1T)
```

### Track 3: Implementation Focus (4 papers)
```
1. Muon paper (algorithm)
        ↓
2. Distributed Shampoo (scaling techniques)
        ↓
3. Kimi K2 (production deployment)
        ↓
4. μP paper (hyperparameter transfer)
```

---

## Blog Posts & Tutorials

| Resource | Author | Description |
|----------|--------|-------------|
| [Enter the Matrix](https://lakernewhouse.github.io/muon/part1.html) | Laker Newhouse | Visual guide to matrix norms |
| [NanoGPT Speedrun](https://github.com/KellerJordan/modded-nanogpt) | Keller Jordan | Practical implementation |
| [Optimizer Zoo](https://www.ruder.io/optimizing-gradient-descent/) | Sebastian Ruder | Overview of optimization algorithms |

---

## Video Lectures

| Video | Speaker | Duration | Topic |
|-------|---------|----------|-------|
| [The Math of Muon](https://www.youtube.com/watch?v=example) | Jeremy Bernstein | ~1hr | Derivation walkthrough |
| [Optimization for Deep Learning](https://www.youtube.com/watch?v=example2) | Various | Course | Comprehensive background |

---

## Code Repositories

| Repo | Description |
|------|-------------|
| [muon-pytorch](https://github.com/KellerJordan/Muon) | Reference implementation |
| [modded-nanogpt](https://github.com/KellerJordan/modded-nanogpt) | NanoGPT with Muon |
| [optax](https://github.com/deepmind/optax) | JAX optimizers (includes related methods) |

---

## How to Read These Papers

### For Practitioners
Focus on:
- Algorithm descriptions
- Hyperparameter recommendations
- Ablation studies
- Implementation details in appendices

### For Researchers
Focus on:
- Theoretical analysis
- Convergence proofs
- Assumptions and limitations
- Open problems mentioned

### Time-Efficient Strategy
1. Read abstract + intro + conclusion first
2. Study figures and tables
3. Read algorithm box carefully
4. Skim proofs (unless you need details)
5. Check appendix for implementation specifics

---

📖 **Happy reading!** Each paper builds on the others. Start with your track, and branch out as needed.
