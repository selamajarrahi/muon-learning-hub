# Practical Efficiency of Muon for Pretraining

> **Source**: [arXiv:2505.02222](https://arxiv.org/abs/2505.02222)  
> **Authors**: Essential AI Research Team  
> **Date**: May 2025  
> **Reading Time**: 12 minutes

## 📋 Overview

This paper from Essential AI provides a **practical, empirical analysis** of Muon for pretraining large language models. Rather than focusing on theory, it answers the engineer's question: **"Should I use Muon for my training run?"**

## 🎯 Key Contributions

1. **Hyperparameter transfer analysis**: How well do Muon HPs transfer across scales?
2. **Wall-clock speedup measurements**: Real training time, not just step efficiency
3. **Integration with existing infrastructure**: vLLM, DeepSpeed, etc.
4. **Failure modes documentation**: When and how Muon breaks

## 📊 Hyperparameter Transfer

### The Promise

muP (maximal update parameterization) promises that hyperparameters trained on small models transfer to large models. **Does Muon work with muP?**

### The Results

| Property | Transfers? | Notes |
|----------|------------|-------|
| Base LR | ✅ Yes (with width scaling) | LR ∝ 1/sqrt(width) |
| Momentum | ✅ Yes | 0.95 works across scales |
| NS steps | ✅ Yes | 5 steps sufficient |
| Weight decay | ⚠️ Partially | May need adjustment |
| Warmup ratio | ⚠️ Partially | Larger models need more |

### Recommended Transfer Recipe

```python
# Train on 125M model
base_config = {
    'lr': 0.02,
    'momentum': 0.95,
    'ns_steps': 5,
    'warmup_steps': 500,
    'weight_decay': 0.0
}

# Scale to 7B model
def scale_config(base, target_width, base_width=768):
    scale = (base_width / target_width) ** 0.5
    return {
        'lr': base['lr'] * scale,
        'momentum': base['momentum'],  # Keep same
        'ns_steps': base['ns_steps'],  # Keep same
        'warmup_steps': int(base['warmup_steps'] * 2),  # Increase
        'weight_decay': base['weight_decay']  # Keep or tune
    }

config_7b = scale_config(base_config, target_width=4096)
# lr ≈ 0.0087
```

## ⏱️ Wall-Clock Speedup

### The Critical Question

Step-efficiency != wall-clock time. Newton-Schulz has compute overhead.

### Measurements (7B model, 128 H100s)

| Metric | AdamW | Muon | Delta |
|--------|-------|------|-------|
| Time per step | 1.00x | 1.05x | +5% |
| Steps to target loss | 100K | 78K | -22% |
| **Total wall-clock** | **100%** | **82%** | **-18%** |

### Breakdown

```
Muon overhead: +5% per step
Muon efficiency: -22% steps needed
Net: ~18% wall-clock savings at 7B scale
```

### At Different Scales

| Model Size | Per-Step Overhead | Step Efficiency | Net Savings |
|------------|-------------------|-----------------|-------------|
| 125M | +12% | -35% | -23% |
| 1.3B | +8% | -25% | -17% |
| 7B | +5% | -22% | -18% |
| 30B | +4% | -18% | -14% |

The overhead decreases with scale (compute dominates), but efficiency gains also decrease.

## 🔧 Infrastructure Integration

### DeepSpeed ZeRO

```python
# Muon works with ZeRO-2 and ZeRO-3
ds_config = {
    "zero_optimization": {
        "stage": 2,
        # Muon has special handling for optimizer states
        "offload_optimizer": False  # Recommended for Muon
    }
}

optimizer = Muon(muon_params, lr=0.01)
model, optimizer, _, _ = deepspeed.initialize(
    model=model,
    optimizer=optimizer,
    config=ds_config
)
```

### FSDP

```python
# Muon compatible with PyTorch FSDP
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP

model = FSDP(model, ...)
optimizer = Muon(model.parameters(), lr=0.01)

# Note: Newton-Schulz operates on sharded gradients
# Communication happens during NS iteration
```

### vLLM Training

```python
# Using vLLM's training utilities with Muon
from vllm.training import VLLMTrainer

trainer = VLLMTrainer(
    model=model,
    optimizer_cls=Muon,
    optimizer_kwargs={'lr': 0.01, 'momentum': 0.95}
)
```

## ⚠️ Documented Failure Modes

### 1. Embedding Instability

**Symptom**: NaN loss after including embeddings in Muon

**Root Cause**: Embeddings are lookup tables, not linear transforms

**Fix**: Always use Adam for embeddings
```python
muon_params = [p for n, p in model.named_parameters() 
               if 'embed' not in n and p.ndim == 2]
```

### 2. Gradient Explosion with High LR

**Symptom**: Loss spikes after warmup

**Root Cause**: LR not scaled for model width

**Fix**: Apply proper scaling
```python
lr = 0.02 * (768 / hidden_dim) ** 0.5
```

### 3. Slow Convergence on Fine-tuning

**Symptom**: Fine-tuning with Muon underperforms Adam

**Root Cause**: Muon's aggressive updates disrupt pretrained weights

**Fix**: Use Adam for fine-tuning, or very low LR
```python
# For fine-tuning
optimizer = Adam(model.parameters(), lr=1e-5)

# OR Muon with very low LR
optimizer = Muon(muon_params, lr=0.001)  # 20x lower than pretraining
```

### 4. Memory OOM

**Symptom**: OOM that doesn't occur with Adam

**Root Cause**: Newton-Schulz intermediate activations

**Fix**: Reduce ns_steps or use gradient checkpointing

## 💡 Practical Checklist

Before using Muon for your training:

- [ ] Model size > 100M parameters (otherwise overhead dominates)
- [ ] Pretraining (not fine-tuning)
- [ ] Separate parameters into Muon vs Adam groups
- [ ] Scale LR for model width
- [ ] Verify infrastructure compatibility (FSDP, DeepSpeed)
- [ ] Test on small scale first
- [ ] Monitor per-layer gradient norms

## 📄 Citation

```bibtex
@article{essentialai2025muon,
  title={Practical Efficiency of Muon for Pretraining},
  author={{Essential AI}},
  journal={arXiv preprint arXiv:2505.02222},
  year={2025}
}
```

## 📚 Further Reading

- [Hyperparameter Tuning Guide](../docs/hyperparameter-tuning.md)
- [Troubleshooting](../docs/troubleshooting.md)
- [Muon is Scalable](07-muon-scalable.md)

---

*Added: February 2026*
