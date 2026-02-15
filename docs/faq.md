# ❓ Muon FAQ

> Frequently asked questions about the Muon optimizer

## General Questions

### What does "Muon" stand for?

**M**oment**U**m **O**rthogonalized by **N**ewton-schulz

The name captures the core technique: applying momentum with orthogonalization via Newton-Schulz iterations.

---

### Why is Muon faster than Adam?

Three reasons:

1. **Better update directions:** Orthogonalization gives equal weight to all gradient directions, helping the model learn "rare features" faster
2. **Implicit preconditioning:** The spectral normalization acts as a natural preconditioner
3. **Less hyperparameter sensitivity:** Muon's updates are more stable, requiring less tuning

---

### How much faster is Muon?

Typical speedups on language model training:

| Model Size | Speedup vs AdamW |
|------------|------------------|
| 124M (NanoGPT) | 1.35x |
| 1B | 1.30x |
| 10B+ | 1.25-1.30x |

The exact speedup depends on architecture and hyperparameter tuning.

---

### Does Muon use more memory?

**Less!** Muon only stores momentum, while Adam stores both momentum AND variance:

- Adam: 3x parameter memory (params + momentum + variance)
- Muon: 2x parameter memory (params + momentum)

---

### Is Muon harder to implement?

The core algorithm is ~15 lines of code. The main complexity is:
1. Identifying which parameters to use Muon for
2. Using Adam for the remaining parameters (1D, embeddings)

Most implementations use a hybrid optimizer that handles both.

---

## Usage Questions

### Which parameters should use Muon?

✅ **Use Muon for:**
- All 2D weight matrices ≥256 elements
- Linear layer weights
- Attention projection matrices (Q, K, V, O)
- MLP/FFN weights

❌ **Use Adam for:**
- Embeddings (token, position)
- LayerNorm/RMSNorm parameters
- All biases
- Any 1D parameters
- Very small 2D parameters (<256 elements)

---

### What learning rate should I use?

**Recommended starting points:**

| Muon Parameters | Adam Parameters |
|-----------------|-----------------|
| lr = 0.02 | lr = 3e-4 |
| momentum = 0.95 | betas = (0.9, 0.95) |

Muon is less sensitive to learning rate than Adam. You can often use a single lr=0.02 without much tuning.

---

### How do I handle distributed training?

Muon gradients need to be **all-reduced before orthogonalization**, not after:

```python
# CORRECT
grad = all_reduce(param.grad)  # First: reduce across GPUs
ortho_grad = newtonschulz(grad)  # Then: orthogonalize

# WRONG
ortho_grad = newtonschulz(param.grad)  # Orthogonalizing local grad
ortho_grad = all_reduce(ortho_grad)    # Then reducing (loses benefits!)
```

---

### Can I use Muon with gradient checkpointing?

Yes! Muon is compatible with gradient checkpointing. The Newton-Schulz iterations happen during the optimizer step, not during the backward pass.

---

### Does Muon work with mixed precision (fp16/bf16)?

Yes, but with a caveat:
- Compute Newton-Schulz in **fp32** for stability
- You can keep weights and gradients in bf16/fp16

```python
def newtonschulz5(G):
    G = G.float()  # Upcast for stability
    # ... iterations ...
    return X.type_as(original_G)  # Downcast back
```

---

## Theory Questions

### What is Newton-Schulz iteration?

It's an iterative method to compute the **matrix sign function**, which we use to orthogonalize the gradient. Instead of computing the expensive SVD, we apply a polynomial iteration 5 times:

```python
X = G / G.norm()
for _ in range(5):
    A = X @ X.T
    B = b*A + c*A@A
    X = a*X + B@X
```

After 5 iterations, X ≈ UV^T where G = UΣV^T.

---

### Why orthogonalize the gradient?

Orthogonalization replaces singular values with 1s. This means:

1. **All directions get equal update magnitude** (no dominant direction)
2. **Rare but important features aren't ignored**
3. **The update is the "steepest descent" under spectral norm**

---

### What's the "spectral norm" intuition?

The spectral norm measures how much a matrix can "stretch" a vector. When we do steepest descent under spectral norm, we're saying:

> "Move in the direction that maximizes loss decrease, but don't stretch any input direction too much."

This is a natural constraint for linear layers!

---

### Why 5 Newton-Schulz iterations?

Empirically, 5 iterations provide:
- Sufficient convergence for training benefits
- Minimal compute overhead (~1% extra FLOPs)
- Good balance between accuracy and speed

More iterations (6-10) give slightly better orthogonalization but rarely improve training.

---

## Troubleshooting

### My training is unstable with Muon

Common causes:

1. **Learning rate too high:** Try 0.01 instead of 0.02
2. **Using Muon on embeddings:** Switch these to Adam
3. **Skipping normalization:** Make sure to normalize G before iterations
4. **fp16 Newton-Schulz:** Compute in fp32, then cast back

---

### Muon is slower than expected

Check:
1. Are you using Muon only for 2D params? (1D should use Adam)
2. Is your batch size large enough? (Small batches = more overhead)
3. Are Newton-Schulz iterations fused? (Check implementation efficiency)

---

### Loss is NaN after switching to Muon

Try:
1. Reduce learning rate (0.01 or 0.005)
2. Add gradient clipping (max_norm=1.0)
3. Check for very small weight matrices (exclude from Muon)
4. Ensure Newton-Schulz uses fp32

---

### Different results across runs?

Muon is deterministic given the same initialization and data order. Check:
1. Random seed settings
2. Distributed training synchronization
3. CUDA determinism flags

---

## Production Questions

### Is Muon used in production?

Yes! Notable deployments:
- **Kimi K2** (Moonshot AI): 1 trillion parameter MoE model
- **NanoGPT speedruns**: World record training times
- Various research labs and startups

---

### Any patent/licensing concerns?

Muon is:
- Published in academic papers (open research)
- MIT licensed reference implementation
- No known patent restrictions

Always verify with your legal team for commercial use.

---

### How do I debug Muon training?

Useful metrics to track:
1. **Gradient norms** (pre and post orthogonalization)
2. **Update-to-weight ratio** (should be 0.01-0.1)
3. **Singular value distribution** (of gradients)
4. **Loss curves** (compare Muon vs Adam baseline)

---

## Still Have Questions?

- 📖 Read the [full articles](../articles/)
- 💻 Check the [code examples](../code-examples/)
- 🔗 Open an issue on GitHub
- 🐦 Ask on Twitter/X with #MuonOptimizer
