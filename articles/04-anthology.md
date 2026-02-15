# Old Optimizer, New Norm: An Anthology

**Authors:** Jeremy Bernstein, Laker Newhouse  
**Source:** https://arxiv.org/abs/2409.20325  
**Type:** Academic paper

---

## Summary

This paper reframes three popular optimizers (Adam, Shampoo, Prodigy) as **steepest descent under different norms**. The key insight: these algorithms aren't "second-order" or "adaptive" — they're first-order methods with implicit norm choices.

---

## Core Thesis

> "Deep learning optimizers are often motivated through convex and approximate second-order theory. We argue each method can instead be understood as a squarely first-order method without convexity assumptions."

---

## Steepest Descent Framework

**Steepest descent:** Choose update ΔW to minimize local quadratic model:

```
argmin_{ΔW} [g^T ΔW + λ/2 ‖ΔW‖²]
```

**Key insight (Proposition 1):**
```
Solution = -(‖g‖†/λ) × argmax_{‖t‖=1} g^T t
```

Where ‖·‖† is the dual norm. This separates:
1. **Step size:** dual norm of gradient / sharpness
2. **Direction:** unit vector maximizing inner product with gradient

---

## Story I: Adam as Max-of-Max Norm

**Adam without EMA = Sign descent**

With momentum turned off:
```
W ← W - η × sign(G)
```

This is steepest descent under **ℓ∞ norm** (infinity norm).

**The "max-of-max" connection:**
```
‖w‖_∞ = max_l max_r ‖row_r(W_l)‖_∞ = max_l ‖W_l‖_{ℓ1→ℓ∞}
```

So Adam implicitly uses the **ℓ1→ℓ∞ operator norm** on each layer.

**Problem:** This norm doesn't match how linear layers actually work in neural nets.

---

## Story II: Shampoo as Spectral Norm

**Shampoo without preconditioner accumulation:**
```
W ← W - η × (GG^T)^{-1/4} G (G^TG)^{-1/4}
     = W - η × UV^T  (orthogonalized gradient)
```

This is steepest descent under the **spectral norm** (ℓ2→ℓ2 operator norm).

**Why it makes sense:** Spectral norm measures how much a linear layer can stretch its inputs — directly relevant to neural net forward/backward passes.

---

## Story III: Prodigy as Automatic Step Size

Prodigy estimates the distance to the optimum to set step size automatically.

Under the steepest descent framework, this corresponds to estimating:
```
D = argmax_η [η × decrease_in_loss]
```

---

## Key Definitions

### Induced Operator Norm
```
‖M‖_{α→β} = max_{x} ‖Mx‖_β / ‖x‖_α
```

Different choices of (α, β) give different families of optimizers.

### RMS-to-RMS Norm (Muon's choice)
```
‖M‖_{RMS→RMS} = √(d_in/d_out) × spectral_norm(M)
```

Appropriate because neural net activations should have entries ~±1.

---

## The Takeaway

| Optimizer | Effective Norm | Layer Treatment |
|-----------|----------------|-----------------|
| Adam/SignSGD | ℓ∞ (max-of-max) | Elementwise |
| Shampoo | Spectral (ℓ2→ℓ2) | Matrix-aware |
| Muon | RMS→RMS | Matrix-aware + dimension-normalized |

**Recommendation:** Choose norms intentionally based on layer structure. Different layer types (Linear vs Embedding) should get different norms.

---

## Newton-Schulz Details

The paper's Appendix A describes using Newton-Schulz iteration to efficiently compute orthogonalization.

Polynomial: `p(σ) = (3/2)σ - (1/2)σ³`

Converges when initial singular values < √3.

---

## Impact

This paper:
1. Unified understanding of Adam/Shampoo/Prodigy
2. Provided theoretical foundation for Muon
3. Opened design space for "norm-aware" optimizers
4. Led to NanoGPT speed records
