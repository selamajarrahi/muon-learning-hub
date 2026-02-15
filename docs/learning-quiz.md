# 🧪 Muon Self-Assessment Quiz

> Test your understanding of the Muon optimizer!

---

## Level 1: Fundamentals

### Question 1
What does "Muon" stand for?

<details>
<summary>Click to reveal answer</summary>

**M**oment**U**m **O**rthogonalized by **N**ewton-schulz

The name captures the core technique: momentum with orthogonalization using Newton-Schulz iterations.
</details>

---

### Question 2
Why is Muon specifically designed for **2D** parameters?

A) 2D is easier to implement  
B) 2D parameters represent matrix operations where structure matters  
C) 2D uses less memory  
D) 2D is faster to compute  

<details>
<summary>Click to reveal answer</summary>

**B) 2D parameters represent matrix operations where structure matters**

Linear layers (W @ x) are matrix transformations. The gradient's matrix structure contains useful information about which directions need updates. Adam ignores this by treating each element independently; Muon exploits it.
</details>

---

### Question 3
What does Newton-Schulz iteration compute?

A) The matrix inverse  
B) The eigenvalues  
C) The orthogonalized version of a matrix (UV^T from SVD)  
D) The transpose  

<details>
<summary>Click to reveal answer</summary>

**C) The orthogonalized version of a matrix (UV^T from SVD)**

Given G = UΣV^T (SVD), Newton-Schulz iterations approximate UV^T, effectively replacing all singular values with 1.
</details>

---

### Question 4
Why do we normalize the gradient before Newton-Schulz iterations?

<details>
<summary>Click to reveal answer</summary>

**To ensure convergence.**

Newton-Schulz iteration only converges when the input matrix has spectral norm ≤ 1. Normalizing by the Frobenius norm guarantees this condition is met.

```python
X = G / G.norm()  # Critical!
```
</details>

---

## Level 2: Implementation

### Question 5
What's wrong with this code?

```python
optimizer = Muon(model.parameters(), lr=0.02)
```

<details>
<summary>Click to reveal answer</summary>

**It applies Muon to ALL parameters, including those that shouldn't use it.**

Muon is designed for 2D weight matrices. Embeddings, biases, and LayerNorm should use Adam:

```python
# Correct approach
muon_params = [p for n, p in model.named_parameters() 
               if p.ndim == 2 and 'embed' not in n]
adam_params = [p for n, p in model.named_parameters() 
               if p.ndim != 2 or 'embed' in n]

optimizer = Muon(
    muon_params=muon_params, lr=0.02,
    adamw_params=adam_params, adamw_lr=3e-4
)
```
</details>

---

### Question 6
In distributed training, what's the correct order of operations?

A) Orthogonalize → All-reduce → Update  
B) All-reduce → Orthogonalize → Update  
C) Update → All-reduce → Orthogonalize  
D) It doesn't matter  

<details>
<summary>Click to reveal answer</summary>

**B) All-reduce → Orthogonalize → Update**

The orthogonalization must happen on the **global gradient** (average across all GPUs). If you orthogonalize first, each GPU gets a different orthogonal matrix, and their average is NOT orthogonal.

```python
# Correct
grad = all_reduce(param.grad)  # Get global gradient
ortho_grad = newtonschulz(grad)  # Then orthogonalize
```
</details>

---

### Question 7
What learning rate should you typically start with for Muon?

A) 1e-4 (like Adam)  
B) 0.02 (about 100x higher)  
C) 1.0  
D) 1e-6  

<details>
<summary>Click to reveal answer</summary>

**B) 0.02 (about 100x higher)**

Muon's orthogonalized updates have different scaling than Adam's. Typical ranges:
- Adam: 1e-4 to 1e-3
- Muon: 0.01 to 0.05 (usually 0.02)
</details>

---

### Question 8
Why should Newton-Schulz be computed in fp32 even when using mixed precision training?

<details>
<summary>Click to reveal answer</summary>

**Numerical stability.**

Newton-Schulz involves repeated matrix multiplications (5 iterations × 3 matmuls each = 15 matrix multiplies). In fp16, numerical errors accumulate and can cause:
- NaN values
- Divergent iteration
- Unstable training

The fix is simple:
```python
def newtonschulz(G):
    G = G.float()  # Upcast
    # ... iterations ...
    return X.type_as(original_G)  # Downcast
```
</details>

---

## Level 3: Theory

### Question 9
What norm does Muon perform steepest descent under?

A) L1 norm  
B) L2 (Frobenius) norm  
C) Spectral norm  
D) Max norm  

<details>
<summary>Click to reveal answer</summary>

**C) Spectral norm**

This is the key insight from "Old Optimizer, New Norm":
- Adam ≈ steepest descent under L∞ (max norm)
- SGD ≈ steepest descent under L2 (Frobenius norm)  
- Muon ≈ steepest descent under spectral norm

The spectral norm measures how much a matrix can "stretch" vectors, which is natural for linear transformations.
</details>

---

### Question 10
Why does orthogonalization help with "rare directions" in the gradient?

<details>
<summary>Click to reveal answer</summary>

**Because orthogonalization gives equal weight to ALL singular directions.**

Consider a gradient with singular values [10, 0.1]:
- Adam updates ~100x more in the dominant direction
- Muon (after ortho) updates equally in both directions

This means:
1. Rare but important features get full updates
2. The model doesn't overfit to dominant patterns
3. Learning is more balanced across all representational dimensions
</details>

---

### Question 11
What's the computational complexity of Newton-Schulz vs full SVD?

<details>
<summary>Click to reveal answer</summary>

**Newton-Schulz: O(n²m) per iteration, ~15 matmuls total**  
**SVD: O(min(m,n)³)**

For a 4096 × 4096 matrix:
- SVD: ~68 billion operations
- NS (5 iters): ~15 × 4096² × 4096 ≈ 1 billion operations

Newton-Schulz is ~70x faster for large matrices!

And for neural network gradients, we don't need exact SVD—approximate orthogonalization is sufficient.
</details>

---

### Question 12
Fill in the Newton-Schulz polynomial coefficients:

```python
a, b, c = (???, ???, ???)
```

<details>
<summary>Click to reveal answer</summary>

```python
a, b, c = (3.4445, -4.7750, 2.0315)
```

These values come from optimizing a 5th-order polynomial p(x) = ax + bx³ + cx⁵ to converge singular values to ±1.
</details>

---

## Level 4: Application

### Question 13
You're training a GPT-style model. Which parameters should use Muon vs Adam?

| Parameter | Muon or Adam? |
|-----------|---------------|
| Token embedding | ? |
| Position embedding | ? |
| Q, K, V projections | ? |
| Attention output projection | ? |
| MLP up projection | ? |
| MLP down projection | ? |
| RMSNorm scale | ? |
| Final LM head | ? |

<details>
<summary>Click to reveal answer</summary>

| Parameter | Optimizer | Reason |
|-----------|-----------|--------|
| Token embedding | **Adam** | Lookup table, not matrix transform |
| Position embedding | **Adam** | Embedding, not matrix transform |
| Q, K, V projections | **Muon** | 2D matrix transformations |
| Attention output projection | **Muon** | 2D matrix transformation |
| MLP up projection | **Muon** | 2D matrix transformation |
| MLP down projection | **Muon** | 2D matrix transformation |
| RMSNorm scale | **Adam** | 1D parameter |
| Final LM head | **Adam** or **Muon** | Often tied to embedding, so Adam |
</details>

---

### Question 14
Your training loss is spiking after switching to Muon. What should you check? (List 3+ things)

<details>
<summary>Click to reveal answer</summary>

1. **Learning rate too high** — Try 0.01 instead of 0.02
2. **Embeddings using Muon** — Switch to Adam
3. **fp16 Newton-Schulz** — Ensure fp32 computation
4. **Missing normalization** — Check G/G.norm() is present
5. **Gradient clipping** — Add if not present
6. **Very small matrices in Muon** — Exclude params with <256 elements
7. **Distributed sync order** — Ensure all-reduce before ortho
</details>

---

### Question 15
A colleague claims Muon is just "SVD + SGD". How would you correct them?

<details>
<summary>Click to reveal answer</summary>

While related, there are key differences:

1. **Not exact SVD**: Uses Newton-Schulz approximation (~5 iterations vs expensive SVD)

2. **Uses momentum**: Applies momentum to orthogonalized gradients, not raw gradients

3. **Hybrid approach**: Combines with Adam for non-matrix params

4. **Scale-aware**: Preserves gradient magnitude (some implementations)

5. **Computationally efficient**: <1% overhead vs 50%+ for true SVD

A more accurate description: "Muon is steepest descent under spectral norm, approximated via Newton-Schulz, with momentum."
</details>

---

## Scoring

Count your correct answers:

| Score | Level |
|-------|-------|
| 13-15 | 🏆 Expert — Ready to contribute to Muon development! |
| 10-12 | ⭐ Advanced — Can implement Muon in production |
| 7-9 | 📚 Intermediate — Good foundation, keep learning |
| 4-6 | 🌱 Beginner — Review the fundamentals |
| 0-3 | 🚀 Just Starting — Read the core articles first |

---

## Next Steps

Based on what you got wrong:

- **Fundamentals (Q1-4)**: Read [Understanding Muon](../articles/02-laker-newhouse-understanding.md)
- **Implementation (Q5-8)**: Check [Common Mistakes](common-mistakes.md)
- **Theory (Q9-12)**: Study [Deriving Muon](../articles/03-bernstein-deriving-muon.md)
- **Application (Q13-15)**: Review [Implementation Checklist](implementation-checklist.md)

---

📚 **More Resources:**
- [FAQ](faq.md)
- [Code Examples](../code-examples/)
- [Papers to Read](papers-to-read.md)
