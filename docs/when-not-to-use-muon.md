# When NOT to Use Muon

> Understanding Muon's limitations helps you choose the right tool for the job.

---

## ❌ Don't Use Muon For

### 1. Embeddings and 1D Parameters

**Why not:**
Embeddings are lookup tables, not linear transformations. The matrix structure that Muon exploits doesn't apply.

```python
# These should use Adam, NOT Muon:
model.embed_tokens.weight        # Embedding lookup
model.layer_norm.weight          # 1D scale parameter
model.layer_norm.bias            # 1D bias
model.some_layer.bias            # Any bias term
```

**What happens if you use Muon anyway:**
- Embeddings may collapse (all vectors become similar)
- LayerNorm becomes unstable
- No theoretical justification

---

### 2. Very Small Models

**Why not:**
Muon's overhead (Newton-Schulz iterations) doesn't pay off for small models where Adam is already cheap.

**Rule of thumb:**
```
Model < 50M parameters → Adam probably faster overall
Model > 100M parameters → Muon overhead becomes negligible
```

**The math:**
```
Muon overhead ≈ 5-10% of forward+backward
Muon speedup ≈ 20-35% fewer training steps

Net benefit only if: speedup > overhead
For small models: training is already fast, overhead is noticeable
```

---

### 3. Fine-tuning Pretrained Models

**Why not:**
When fine-tuning, you're making small adjustments to a well-trained model. Muon's aggressive orthogonalization can disrupt learned representations.

**Recommendations:**
```python
# For fine-tuning, prefer:
# 1. LoRA/QLoRA (don't need Muon)
# 2. Adam with low LR
# 3. Only use Muon if training from scratch

# If you must use Muon for fine-tuning:
optimizer = Muon(
    muon_params,
    lr=0.001,  # 10-20x lower than pretraining
    momentum=0.9  # Lower for stability
)
```

**Exception:** Continued pretraining at scale can use Muon.

---

### 4. Non-Matrix Operations

**Why not:**
Muon is designed for matrices. It doesn't make sense for:

- **Scalar parameters** (single learning rate, temperature)
- **Attention patterns** (they're not weight matrices)
- **Dynamic computations** (attention scores, activations)

---

### 5. Memory-Constrained Environments

**Why not:**
Newton-Schulz iterations require storing intermediate matrices:

```
Memory overhead per 2D param:
- Standard Adam: 2 × param_size (m, v)
- Muon: 2 × param_size + temporary buffers for N-S
```

**If memory is tight:**
```python
# Reduce Newton-Schulz steps
optimizer = Muon(muon_params, ns_steps=3)  # Minimum practical

# Or use gradient checkpointing
# Or just use Adam
```

---

### 6. Tasks Where Adam Already Works Great

**Why not:**
If Adam gives you state-of-the-art results with acceptable training time, switching to Muon adds complexity for marginal benefit.

**Tasks where Adam is proven:**
- Supervised classification on standard datasets
- Simple sequence-to-sequence tasks
- Applications where you're not compute-limited

**Tasks where Muon shines:**
- Large-scale language model pretraining
- Speedrun scenarios (reaching loss threshold ASAP)
- Compute-limited scenarios (fixed GPU budget)

---

### 7. Production Inference (Obviously)

**Clarification:** Optimizers only affect training. Using Muon for inference doesn't make sense.

---

## 🟡 Use Muon with Caution

### Multi-Task Learning

**Challenge:** Different tasks may benefit from different optimization dynamics.

**Approach:**
```python
# Consider per-task heads with Adam, shared backbone with Muon
shared_muon_params = model.backbone.parameters()
head_adam_params = model.heads.parameters()
```

### Very Long Sequences

**Challenge:** Attention matrices are N×N; orthogonalization on huge matrices can be expensive.

**Mitigation:**
- Ring attention / chunked attention
- Consider if attention layers should use Muon at all

### Mixed-Precision Training (FP16/BF16)

**Challenge:** Newton-Schulz iterations accumulate numerical errors.

**Solution:**
```python
# Keep optimizer states in FP32
# Only cast gradients for forward/backward
optimizer = Muon(muon_params, lr=0.02)
# Internally, Muon should handle precision appropriately
```

### Non-Standard Architectures

**Challenge:** Muon assumes standard dense layers. Exotic architectures may not benefit.

**Examples requiring care:**
- Sparse attention (only operates on subsets)
- Mixture of Experts (routing adds complexity)
- State-space models (SSMs like Mamba have different dynamics)

---

## 📊 Decision Flowchart

```
Is this a 2D weight in a hidden layer?
    │
    ├── NO → Use Adam ❌
    │
    └── YES
          │
          Are you pretraining from scratch?
          │
          ├── NO (fine-tuning) → Use Adam (low LR) ⚠️
          │
          └── YES
                │
                Is model > 100M params?
                │
                ├── NO → Adam is probably fine ⚠️
                │
                └── YES
                      │
                      Is training speed critical?
                      │
                      ├── NO → Either works ⚠️
                      │
                      └── YES → Use Muon ✅
```

---

## 🔑 Summary Table

| Scenario | Use Muon? | Recommendation |
|----------|-----------|----------------|
| LLM pretraining (>1B) | ✅ Yes | Strong benefit |
| LLM pretraining (<100M) | ⚠️ Maybe | Adam may be simpler |
| Fine-tuning | ❌ No | Use Adam with low LR |
| LoRA/QLoRA | ❌ No | Adam for adapter params |
| Embeddings | ❌ No | Always Adam |
| LayerNorm/biases | ❌ No | Always Adam |
| CNNs (except first layer) | ✅ Yes | Conv filters are 2D |
| ViT pretraining | ✅ Yes | Similar to LLM |
| Memory-limited | ⚠️ Careful | May need to reduce ns_steps |

---

## 📚 Further Reading

- [Muon vs Adam](muon-vs-adam.md) - Detailed comparison
- [Implementation Checklist](implementation-checklist.md) - Getting Muon right
- [Hyperparameter Tuning](hyperparameter-tuning.md) - When you do use Muon

---

*Added: February 2026*
