# ⚠️ Common Mistakes When Using Muon

> Learn from others' mistakes to get Muon working correctly the first time

## Mistake #1: Using Muon for Everything

### ❌ Wrong
```python
# DON'T: Apply Muon to all parameters
optimizer = Muon(model.parameters(), lr=0.02)
```

### ✅ Correct
```python
# DO: Split parameters by type
muon_params = []
adam_params = []

for name, param in model.named_parameters():
    if param.ndim == 2 and param.numel() >= 256:
        muon_params.append(param)
    else:
        adam_params.append(param)

optimizer = Muon(
    muon_params=muon_params,
    lr=0.02,
    adamw_params=adam_params,
    adamw_lr=3e-4,
)
```

### Why?
Muon is designed for **matrix parameters** (2D). For 1D parameters (biases, LayerNorm), embeddings, or very small parameters, Adam works better.

---

## Mistake #2: Forgetting to Normalize Before Newton-Schulz

### ❌ Wrong
```python
def newtonschulz_broken(G, steps=5):
    a, b, c = (3.4445, -4.7750, 2.0315)
    X = G  # Missing normalization!
    for _ in range(steps):
        A = X @ X.T
        B = b * A + c * A @ A
        X = a * X + B @ X
    return X
```

### ✅ Correct
```python
def newtonschulz(G, steps=5):
    a, b, c = (3.4445, -4.7750, 2.0315)
    X = G / G.norm()  # Critical: normalize first!
    for _ in range(steps):
        A = X @ X.T
        B = b * A + c * A @ A
        X = a * X + B @ X
    return X
```

### Why?
Newton-Schulz iteration only converges if the input has spectral norm ≤ 1. Normalizing by Frobenius norm ensures this.

---

## Mistake #3: Computing Newton-Schulz in fp16

### ❌ Wrong
```python
# fp16 throughout - can cause instability
grad = param.grad  # fp16
ortho_grad = newtonschulz(grad)  # unstable in fp16!
```

### ✅ Correct
```python
# Upcast to fp32 for numerical stability
grad = param.grad
ortho_grad = newtonschulz(grad.float()).type_as(grad)
```

### Why?
Newton-Schulz involves repeated matrix multiplications. In fp16, numerical errors accumulate and can cause NaN or unstable training.

---

## Mistake #4: Wrong Gradient Synchronization in Distributed Training

### ❌ Wrong
```python
# Orthogonalize local gradients, then sync
for param in muon_params:
    ortho_grad = newtonschulz(param.grad)
    dist.all_reduce(ortho_grad)  # WRONG ORDER!
    param.grad = ortho_grad
```

### ✅ Correct
```python
# Sync gradients first, then orthogonalize
for param in muon_params:
    dist.all_reduce(param.grad)  # First: get global gradient
    ortho_grad = newtonschulz(param.grad)  # Then: orthogonalize
    param.grad = ortho_grad
```

### Why?
Orthogonalization must happen on the **global averaged gradient**. Otherwise, each GPU orthogonalizes its local gradient, and the average of orthogonal matrices is NOT orthogonal.

---

## Mistake #5: Using Adam's Learning Rate for Muon

### ❌ Wrong
```python
# Using Adam's typical lr for Muon
optimizer = Muon(muon_params=params, lr=3e-4)  # Too low!
```

### ✅ Correct
```python
# Use Muon's recommended lr
optimizer = Muon(muon_params=params, lr=0.02)  # ~100x higher
```

### Why?
Muon's orthogonalized updates have different scale than Adam's. The typical range is:
- Adam: 1e-4 to 1e-3
- Muon: 0.01 to 0.05

---

## Mistake #6: Applying Muon to Embedding Layers

### ❌ Wrong
```python
class Model(nn.Module):
    def __init__(self):
        self.embed = nn.Embedding(50000, 768)
        self.layers = nn.ModuleList([...])

# Includes embeddings in Muon params!
muon_params = [p for p in model.parameters() if p.ndim == 2]
```

### ✅ Correct
```python
# Explicitly separate embeddings
muon_params = []
adam_params = []

for name, param in model.named_parameters():
    if 'embed' in name or param.ndim != 2:
        adam_params.append(param)
    else:
        muon_params.append(param)
```

### Why?
Embeddings are lookup tables, not matrix transformations. The rows are accessed independently, so matrix structure doesn't help. In fact, Muon can hurt embedding quality.

---

## Mistake #7: Not Scaling Down Newton-Schulz Output

### ❌ Wrong (for some implementations)
```python
def muon_step(param, momentum):
    grad = param.grad
    ortho_grad = newtonschulz(grad)
    # Missing: scale by original gradient norm
    momentum.mul_(0.95).add_(ortho_grad)
    param.sub_(lr * momentum)
```

### ✅ Correct
```python
def muon_step(param, momentum):
    grad = param.grad
    grad_norm = grad.norm()
    ortho_grad = newtonschulz(grad)
    ortho_grad = ortho_grad * grad_norm  # Restore magnitude
    momentum.mul_(0.95).add_(ortho_grad)
    param.sub_(lr * momentum)
```

### Why?
Newton-Schulz normalizes the gradient. If you want to preserve the overall update magnitude (relative to gradient scale), multiply back by the original norm. Some implementations do this, some adjust lr instead.

---

## Mistake #8: Using Too Many Newton-Schulz Iterations

### ❌ Wasteful
```python
def newtonschulz(G, steps=20):  # Overkill!
    ...
```

### ✅ Efficient
```python
def newtonschulz(G, steps=5):  # Sweet spot
    ...
```

### Why?
5 iterations is enough for practical convergence. More iterations give marginally better orthogonalization but ~4x more compute with no training benefit.

---

## Mistake #9: Ignoring Very Small Matrices

### ❌ Problematic
```python
# Including tiny matrices
for param in model.parameters():
    if param.ndim == 2:  # Even 2x2 matrices!
        muon_params.append(param)
```

### ✅ Better
```python
# Minimum size threshold
for param in model.parameters():
    if param.ndim == 2 and param.numel() >= 256:
        muon_params.append(param)
    else:
        adam_params.append(param)
```

### Why?
For tiny matrices, the Newton-Schulz overhead isn't worth it. Also, very small matrices may have numerical issues.

---

## Mistake #10: Expecting Muon to Fix Bad Architectures

### ❌ Wishful thinking
```
"My model doesn't converge with Adam. Let me try Muon!"
```

### ✅ Reality check
Muon accelerates training of **well-designed models**. It won't fix:
- Vanishing/exploding gradients
- Poor initialization
- Architecture bugs
- Inadequate data

**Fix fundamentals first**, then switch to Muon for efficiency gains.

---

## Debugging Checklist

When Muon isn't working, check:

- [ ] Are embeddings and 1D params using Adam?
- [ ] Is Newton-Schulz computed in fp32?
- [ ] Is the learning rate in the right range (0.01-0.05)?
- [ ] In distributed training, is all-reduce before orthogonalization?
- [ ] Is the input normalized before Newton-Schulz iterations?
- [ ] Are you using 5 iterations (not more, not less)?
- [ ] Are very small matrices (<256 elements) excluded?

---

📚 **See Also:**
- [FAQ](faq.md) - Common questions answered
- [Implementation Checklist](implementation-checklist.md) - Step-by-step adoption guide
