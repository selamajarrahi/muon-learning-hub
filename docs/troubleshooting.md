# Muon Troubleshooting Guide

> Solutions to common issues when implementing and using Muon.

---

## 🔴 Critical Issues

### Issue: Training Immediately Explodes (NaN/Inf)

**Symptoms:**
- Loss becomes NaN within first 100 steps
- Gradients explode immediately

**Diagnosis:**
```python
# Check for NaN gradients
for name, param in model.named_parameters():
    if param.grad is not None and torch.isnan(param.grad).any():
        print(f"NaN gradient in: {name}")
```

**Solutions:**

1. **Check learning rate** — Start 10x lower:
```python
optimizer = Muon(muon_params, lr=0.002)  # Was 0.02
```

2. **Check data for NaNs:**
```python
for batch in dataloader:
    if torch.isnan(batch['input_ids']).any():
        print("Found NaN in data!")
```

3. **Add gradient clipping:**
```python
torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
```

4. **Check Newton-Schulz normalization:**
```python
# In Muon, ensure proper normalization
X = G / (G.norm() + 1e-7)  # Add epsilon to prevent division by zero
```

---

### Issue: Using Muon on Wrong Parameters

**Symptoms:**
- Performance worse than Adam
- Embeddings/norms become degenerate

**The Rule:**
```
✅ Muon: 2D parameters in hidden layers (Linear weights, Conv2d filters)
❌ Adam: Everything else (embeddings, LayerNorm, biases, 1D params)
```

**Correct Assignment:**
```python
def partition_params(model):
    muon_params = []
    adam_params = []
    
    for name, param in model.named_parameters():
        # Skip non-trainable
        if not param.requires_grad:
            continue
            
        # 2D hidden layer weights → Muon
        if param.ndim == 2 and 'embed' not in name.lower():
            muon_params.append(param)
        # Everything else → Adam
        else:
            adam_params.append(param)
    
    return muon_params, adam_params

muon_params, adam_params = partition_params(model)
optimizer = Muon(
    muon_params=muon_params,
    lr=0.02,
    adamw_params=adam_params,
    adamw_lr=3e-4
)
```

**Verification:**
```python
print(f"Muon params: {sum(p.numel() for p in muon_params):,}")
print(f"Adam params: {sum(p.numel() for p in adam_params):,}")
# Typically Muon should handle 60-80% of parameters
```

---

## 🟡 Performance Issues

### Issue: Muon is Slower than Adam

**Symptoms:**
- Step time significantly higher with Muon
- GPU memory higher than expected

**Solutions:**

1. **Reduce Newton-Schulz steps:**
```python
optimizer = Muon(muon_params, lr=0.02, ns_steps=3)  # Was 5
```

2. **Use efficient implementation:**
```python
# Ensure you're using the fused version
from muon import Muon  # Official, optimized
# NOT a naive implementation
```

3. **Check batch sizes** — Muon overhead is amortized over larger batches

4. **Profile the bottleneck:**
```python
import torch.profiler

with torch.profiler.profile(
    activities=[torch.profiler.ProfilerActivity.CPU, 
                torch.profiler.ProfilerActivity.CUDA]
) as prof:
    for _ in range(10):
        train_step()

print(prof.key_averages().table(sort_by="cuda_time_total"))
```

---

### Issue: Convergence is Slower than Expected

**Symptoms:**
- Loss decreasing but slower than Adam
- Not matching NanoGPT speedup claims

**Potential Causes:**

1. **LR not scaled for model size:**
```python
# For larger models, you need smaller LR
hidden_dim = 4096
lr = 0.02 * (768 / hidden_dim) ** 0.5  # ≈ 0.0087
```

2. **Missing warmup:**
```python
from torch.optim.lr_scheduler import LinearLR

warmup_scheduler = LinearLR(
    optimizer, 
    start_factor=0.1, 
    total_iters=500  # 500-2000 warmup steps
)
```

3. **Task mismatch** — Muon excels at language modeling; gains may differ on other tasks

---

### Issue: Distributed Training Hangs/Crashes

**Symptoms:**
- Training hangs when using DistributedDataParallel
- NCCL errors

**Solutions:**

1. **Use gradient buckets properly:**
```python
# DDP with gradient checkpointing needs care
model = DDP(model, find_unused_parameters=False)
```

2. **Synchronize before Newton-Schulz:**
```python
# In custom Muon for distributed:
if dist.is_initialized():
    dist.all_reduce(grad, op=dist.ReduceOp.AVG)
```

3. **Check Muon's distributed support:**
```python
# Official Muon handles this, ensure you're using latest version
pip install --upgrade muon-pytorch
```

---

## 🟢 Unexpected Behavior

### Issue: Training is Stable but Final Performance is Worse

**Symptoms:**
- No explosions, smooth loss curves
- But validation metrics lag behind Adam

**Diagnosis Checklist:**

- [ ] Are ALL linear layers using Muon? (Some might be missed)
- [ ] Is the comparison fair? (Same LR schedule, same total steps)
- [ ] Did you give Muon enough steps? (It often catches up late in training)

**Debug:**
```python
# Track layer-wise gradient norms
def log_gradient_norms():
    norms = {}
    for name, param in model.named_parameters():
        if param.grad is not None:
            norms[name] = param.grad.norm().item()
    return norms

# Compare Muon vs Adam runs
```

---

### Issue: Different Results Across Runs

**Symptoms:**
- Non-deterministic training with Muon
- Results vary more than with Adam

**Solutions:**

1. **Fix all seeds:**
```python
import torch
import numpy as np
import random

def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

set_seed(42)
```

2. **Use deterministic algorithms:**
```python
torch.use_deterministic_algorithms(True)
```

3. **Note**: Some variance is expected; Newton-Schulz iterations may have slight numerical differences on different hardware

---

## 🔧 Implementation Issues

### Issue: Custom Model Architecture Doesn't Work

**When you have:**
- Custom attention patterns
- Non-standard linear layers
- Mixed precision complications

**Checklist:**

1. **Identify all 2D parameters:**
```python
for name, param in model.named_parameters():
    if param.ndim == 2:
        print(f"2D: {name} - shape: {param.shape}")
```

2. **Exclude special layers:**
```python
# Some "linear" layers shouldn't use Muon
exclude_patterns = ['lm_head', 'embed', 'norm', 'head']
muon_params = [
    p for n, p in model.named_parameters()
    if p.ndim == 2 and not any(ex in n for ex in exclude_patterns)
]
```

3. **Test incrementally:**
```python
# Start with Muon on just attention layers
# Then add MLP layers
# Then add other 2D weights
```

---

### Issue: Memory Pressure

**Symptoms:**
- OOM errors that don't occur with Adam
- Higher memory usage

**Solutions:**

1. **Newton-Schulz stores intermediate tensors:**
```python
# Reduce ns_steps if memory constrained
optimizer = Muon(muon_params, ns_steps=3)
```

2. **Use gradient checkpointing:**
```python
from torch.utils.checkpoint import checkpoint

# In your forward pass
x = checkpoint(self.layer, x)
```

3. **Reduce batch size** (Muon's overhead is per-step, not per-sample)

---

## 📝 Reporting Issues

When reporting Muon issues, include:

```markdown
**Environment:**
- PyTorch version: 
- CUDA version:
- GPU model:
- muon-pytorch version:

**Code to reproduce:**
```python
# Minimal example
```

**Expected behavior:**
What you expected

**Actual behavior:**
What happened

**Additional context:**
- Model architecture
- Training configuration
- Error messages (full traceback)
```

---

## 📚 Resources

- [GitHub Issues](https://github.com/KellerJordan/Muon/issues) - Search existing issues
- [Muon Discord](https://discord.gg/...) - Community help
- [Paper](https://arxiv.org/abs/2409.20325) - Theoretical background

---

*Added: February 2026*
