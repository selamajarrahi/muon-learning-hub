# Muon Glossary

Quick reference for key terms in the Muon literature.

---

## Core Concepts

### Muon
**M**oment**U**m **O**rthogonalized by **N**ewton-Schulz. An optimizer that orthogonalizes momentum updates before applying them.

### Orthogonalization
The operation G → UV^T where G = UΣV^T is the SVD. Replaces singular values with 1s while keeping directions.

### Newton-Schulz Iteration
An iterative algorithm to approximate orthogonalization using only matrix multiplications. Avoids expensive SVD.

### Singular Value Decomposition (SVD)
Factorization G = UΣV^T where U, V are orthogonal and Σ is diagonal with non-negative entries (singular values).

---

## Norms

### RMS Norm (Root Mean Square)
```
‖v‖_RMS = √(1/d × Σ v_i²)
```
Measures "average" entry size. A vector of all ±1s has RMS norm = 1.

### Spectral Norm
Largest singular value of a matrix. Measures maximum "stretching" of input vectors.

### RMS-to-RMS Operator Norm
```
‖M‖_{RMS→RMS} = max_x ‖Mx‖_RMS / ‖x‖_RMS = √(d_in/d_out) × spectral_norm(M)
```
The norm Muon uses. Measures how much a linear layer can amplify activation RMS.

### Dual Norm
For norm ‖·‖, the dual is ‖g‖† = max_{‖x‖=1} g^T x. Used in steepest descent theory.

---

## Algorithms

### Steepest Descent
```
Δw = argmin_Δ [g^T Δ + λ/2 ‖Δ‖²]
```
Update direction that maximizes gradient alignment while controlling step size.

### SGD-Momentum
```
M_t = β M_{t-1} + G_t
W_t = W_{t-1} - η M_t
```
Exponential moving average of gradients.

### Adam
```
m_t = β1 m_{t-1} + (1-β1) g_t
v_t = β2 v_{t-1} + (1-β2) g_t²
W_t = W_{t-1} - η m_t / √v_t
```
Without EMA, reduces to sign descent.

### Shampoo
Uses preconditioner matrices (GG^T)^{-1/4} and (G^TG)^{-1/4}. Without accumulation, equals orthogonalization.

---

## Mathematical Notation

| Symbol | Meaning |
|--------|---------|
| W | Weight matrix |
| G | Gradient matrix |
| M | Momentum buffer |
| η | Learning rate |
| β | Momentum coefficient (typically 0.95) |
| U, V | Singular vectors (orthogonal matrices) |
| Σ | Diagonal matrix of singular values |
| d_in, d_out | Input/output dimensions |
| fan_in, fan_out | Same as d_in, d_out |

---

## Key Parameters

### Muon Defaults
- **lr (η):** 0.02
- **momentum (β):** 0.95
- **Newton-Schulz steps:** 5
- **NS coefficients:** (3.4445, -4.7750, 2.0315)

### When to Adjust
- **Higher lr:** For smaller models or more aggressive training
- **More NS steps:** If orthogonalization quality matters (rare)
- **Lower momentum:** For faster adaptation (noisier)

---

## Papers & People

| Name | Contribution |
|------|--------------|
| Jeremy Bernstein | Theory lead, derivation, Newton-Schulz idea |
| Keller Jordan | Implementation lead, NanoGPT speedruns |
| Laker Newhouse | Theory, Understanding Muon series |
| Moonshot AI (Kimi) | Scaled to 1T params |

---

## Common Confusions

**Q: Is Muon a second-order method?**  
A: No. It's first-order (uses only gradients). The connection to Shampoo is coincidental — both happen to orthogonalize, but Muon derives this from first principles.

**Q: Does Muon need more memory than Adam?**  
A: No. Same memory as SGD-momentum (one buffer per parameter).

**Q: Should I use Muon for embeddings?**  
A: No. Use AdamW for embeddings, LayerNorm, biases, and 1D params.

**Q: What about weight decay?**  
A: Apply weight decay separately (like AdamW). Muon handles the update direction/magnitude.
