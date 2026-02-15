# Understanding Muon: Chapter 1 — Into the Matrix

**Author:** Laker Newhouse  
**Source:** https://www.lakernewhouse.com/writing/muon-1  
**Type:** Educational blog series

---

## Summary

Laker's series is the **best pedagogical introduction** to Muon's theory. Uses Matrix movie metaphors to build intuition from scratch. This chapter explains *why* we care about matrix norms and orthogonalization.

---

## Key Insights

### The Gradient Normalization Problem

All optimizers face a tension:
- **SGD:** Step size ∝ gradient magnitude → bad scaling
- **SignSGD:** Step size = 1 per element → ignores structure
- **Adam:** Smoothed sign descent → still elementwise

The core question: **How do we measure "distance 1" for a weight matrix?**

### Enter the Matrix

Adam sees gradient entries. Muon sees **the entire matrix as a linear transformation**.

> "Forget that a matrix is made of numbers. A linear transformation is all there is."

### The RMS-to-RMS Norm

For neural nets, we want activations with entries ~±1. So we measure vectors by **root-mean-square**:

```
‖v‖_RMS = √(1/d Σ v_i²)
```

For matrices, we ask: "How much can this transformation stretch activations?"

```
‖W‖_{RMS→RMS} = max_{‖x‖_RMS=1} ‖Wx‖_RMS = √(d_in/d_out) × spectral_norm(W)
```

### The Fundamental Tension

**Goal:** Minimize linearized loss while controlling activation change.

```
argmin_{ΔW} ⟨G, ΔW⟩   s.t.  ‖ΔW‖_{RMS→RMS} ≤ 1
```

**Solution:** If G = UΣV^T (SVD), then:
```
ΔW = -√(d_out/d_in) × UV^T
```

This is **orthogonalization** — keep the directions, set all scales to 1.

### The Newton-Schulz Trick

**Problem:** SVD is slow.

**Solution:** Odd polynomials commute with SVD!
```
p(UΣV^T) = U p(Σ) V^T
```

Find polynomial p where p∘p∘p∘... → sign function. Then:
- Apply p repeatedly to G
- Singular values converge to 1
- No explicit SVD computation!

---

## The Polynomial

Basic: `p(x) = (3/2)x - (1/2)x³`  
Tuned: `p(x) = 3.4445x - 4.7750x³ + 2.0315x⁵`

The tuned version has higher linear coefficient (3.4445 vs 1.5), so small singular values grow faster toward 1.

---

## Visualization (from the article)

Gradient singular values before orthogonalization: **mostly tiny**  
After 5 Newton-Schulz iterations: **all near 1**

This shows why orthogonalization matters — most gradient directions are "drowned out" by a few dominant ones. Muon equalizes them.

---

## Key Takeaways

1. **Think in matrices**, not elements
2. **RMS norm** is the right measure for neural net activations
3. **Orthogonalization** is the solution to steepest descent under spectral norm
4. **Newton-Schulz** makes it computationally tractable
5. Gradient singular values are **highly skewed** — orthogonalization fixes this

---

## Series Navigation
- Chapter 1: Into the Matrix (this article)
- [Chapter 2: Source Code](https://www.lakernewhouse.com/writing/muon-2) — line-by-line PyTorch
- [Chapter 3: Weight Regulation](https://www.lakernewhouse.com/writing/muon-3) — MuonClip and future directions
