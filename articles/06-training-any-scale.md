# Training Neural Networks at Any Scale

**Authors:** Thomas Pethick et al.  
**Source:** https://arxiv.org/abs/2511.11163  
**Type:** Survey/review paper

---

## Summary

A comprehensive **review of modern neural network optimization** with emphasis on efficiency and scale. Covers state-of-the-art algorithms under a unified framework, including Muon.

---

## Key Themes

### 1. Structure-Aware Optimization
Modern optimizers should adapt to problem structure, not treat all parameters equally.

> "Highlights the importance of adapting to the structures in the problem."

### 2. Scale Agnosticism
Good optimizers should work across model sizes without extensive retuning.

> "How to make these algorithms agnostic to the scale of the problem."

---

## Muon in Context

The paper positions Muon within the broader landscape:

| Family | Examples | Key Idea |
|--------|----------|----------|
| SGD variants | SGD, Momentum, Nesterov | Basic gradient descent |
| Adaptive | Adam, AdaGrad, RMSprop | Per-parameter scaling |
| Second-order approx | Shampoo, K-FAC | Preconditioner matrices |
| **Norm-based** | **Muon**, Spectral SGD | Steepest descent under operator norms |

Muon represents the **norm-based** approach — choosing the right norm for the layer type.

---

## Unified Algorithmic Template

The paper presents algorithms as:
```
Δw = -η × Direction(G) × StepSize(G)
```

Where Direction and StepSize are determined by norm choice:
- **Adam:** Direction = sign(G), StepSize ∝ |G|
- **Shampoo:** Direction = UV^T (orthogonalized), StepSize = 1
- **Muon:** Direction = UV^T, StepSize = √(d_out/d_in)

---

## Scaling Considerations

For models at scale:
1. **Communication overhead** dominates — need efficient distributed algorithms
2. **Memory constraints** — can't store full preconditioners
3. **Learning rate schedules** — should transfer across scales
4. **Numerical stability** — bf16/fp16 training requires care

Muon addresses several of these:
- ✅ Newton-Schulz is communication-efficient (local computation)
- ✅ Same memory as SGD-momentum
- ✅ Learning rate transfers across width
- ✅ Stable in bf16

---

## Future Directions (from the paper)

1. **Per-layer norm selection:** Different layer types (attention, MLP, embedding) may need different norms
2. **Automatic hyperparameter tuning:** Prodigy-style distance estimation for Muon
3. **Better theory:** Convergence guarantees for non-convex cases
4. **Hardware co-design:** Custom kernels for Newton-Schulz

---

## Who Should Read This

- Researchers wanting broad optimization context
- Practitioners choosing between optimizers
- Anyone interested in scaling laws for optimization

---

## Links

- [arXiv Paper](https://arxiv.org/abs/2511.11163)
- [PDF](https://arxiv.org/pdf/2511.11163)
