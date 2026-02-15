# Muon: An Optimizer for Hidden Layers

**Author:** Keller Jordan  
**Source:** https://kellerjordan.github.io/posts/muon/  
**Type:** Blog post (practical implementation focus)

---

## Summary

This is the **canonical practical introduction** to Muon. Keller Jordan (lead implementer) explains the algorithm, provides PyTorch code, and shows benchmark results.

---

## Key Points

### Definition
Muon optimizes 2D parameters by:
1. Computing SGD momentum on gradients
2. Orthogonalizing the momentum via Newton-Schulz iteration
3. Applying the orthogonalized update

```python
M_{t+1} = β * M_t + G_{t+1}           # momentum
W_{t+1} = W_t - η * NewtonSchulz(M_t)  # orthogonalized update
```

### Newton-Schulz Iteration (5 steps)
```python
def newtonschulz5(G, steps=5, eps=1e-7):
    a, b, c = (3.4445, -4.7750, 2.0315)
    X = G.bfloat16()
    X /= (X.norm() + eps)
    if G.size(0) > G.size(1):
        X = X.T
    for _ in range(steps):
        A = X @ X.T
        B = b * A + c * A @ A
        X = a * X + B @ X
    if G.size(0) > G.size(1):
        X = X.T
    return X
```

### Results
- **NanoGPT speedrun:** 1.35x faster than AdamW
- **CIFAR-10:** 94% in 2.6 A100-seconds (was 3.3)
- **Scales to 1.5B params** with consistent improvement

### Why Orthogonalize?
Empirically, gradient matrices have **high condition number** (nearly low-rank). Orthogonalization "rescales rare directions" that would otherwise be dominated.

### When to Use Muon vs AdamW
| Use Muon | Use AdamW |
|----------|-----------|
| 2D hidden layer params | Embeddings |
| Attention QKV projections | LayerNorm weights |
| MLP weights | Biases, 1D params |
| Conv layers (flatten last 3 dims) | Input/output layers |

### Runtime Overhead
- **FLOP overhead:** ~5 * m/B (model dim / batch size)
- **NanoGPT (m=768, B=524k):** 0.7%
- **Llama 405B (m=16k, B=16M):** 0.5%
- Memory: Same as SGD-momentum

---

## Relationship to Prior Work

### Shampoo Connection
With preconditioner accumulation removed:
```
Shampoo: W ← W - η(GG^T)^{-1/4} G (G^TG)^{-1/4} = W - η UV^T
```
Muon is "instantaneous Shampoo" but much faster (Newton-Schulz vs inverse-fourth roots).

### Why Not SVD?
SVD gives exact orthogonalization but is **too slow**. Newton-Schulz runs in **bfloat16** and converges in 5 iterations.

---

## Implementation Notes

- Coefficients `(3.4445, -4.7750, 2.0315)` are **tuned** for fast convergence
- Must normalize G by Frobenius norm first
- 5 iterations sufficient for LLM training
- Handles tall/wide matrices via transpose trick

---

## Quotable

> "One valid answer would be: It just is OK?" — on why orthogonalization works

---

## Links
- [Official PyTorch implementation](https://github.com/KellerJordan/Muon)
- [NanoGPT speedrun repo](https://github.com/KellerJordan/modded-nanogpt)
