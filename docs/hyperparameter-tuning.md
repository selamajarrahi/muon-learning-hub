# Muon Hyperparameter Tuning Guide

> A practical guide to tuning Muon for your specific model and task.

## 🎯 Overview

Unlike Adam where you mainly tune learning rate and weight decay, Muon has unique parameters that interact with your model architecture. This guide will help you find optimal settings.

---

## 📋 Key Hyperparameters

| Parameter | Default | Typical Range | Impact |
|-----------|---------|---------------|--------|
| `lr` | 0.02 | 0.005 - 0.05 | Primary tuning knob |
| `momentum` | 0.95 | 0.9 - 0.99 | Stability vs speed |
| `weight_decay` | 0.0 | 0.0 - 0.1 | Regularization |
| `ns_steps` | 5 | 3 - 10 | Orthogonalization precision |
| `nesterov` | True | True/False | Usually keep True |

---

## 🔧 Step-by-Step Tuning Process

### Step 1: Start with Defaults

```python
optimizer = Muon(
    muon_params=model.hidden_layers.parameters(),
    lr=0.02,              # Start here
    momentum=0.95,        # Usually fine
    ns_steps=5,           # Usually fine
    nesterov=True,        # Keep this
    adamw_params=model.embeddings.parameters(),
    adamw_lr=3e-4,
)
```

### Step 2: Find Learning Rate

**Method: LR Range Test**

```python
from torch.optim.lr_scheduler import LinearLR
import matplotlib.pyplot as plt

# Start very low, increase exponentially
lrs = []
losses = []

optimizer = Muon(muon_params, lr=1e-5)
scheduler = LinearLR(optimizer, start_factor=1.0, end_factor=1000.0, total_iters=100)

for step in range(100):
    loss = train_step()
    lrs.append(optimizer.param_groups[0]['lr'])
    losses.append(loss.item())
    scheduler.step()

# Plot to find optimal LR (steepest descent before explosion)
plt.plot(lrs, losses)
plt.xscale('log')
plt.xlabel('Learning Rate')
plt.ylabel('Loss')
plt.savefig('lr_finder.png')
```

**Expected Result**:
- Loss decreases steeply around optimal LR
- Explodes shortly after
- Choose LR at ~10x below explosion point

### Step 3: Adjust for Model Scale

**muP-like scaling** (recommended):

```python
# For hidden dimension d:
# lr ∝ 1/d (approximately)

base_lr = 0.02
base_dim = 768  # e.g., GPT-2 small

your_dim = 4096  # Your model
your_lr = base_lr * (base_dim / your_dim) ** 0.5  # ~0.0087
```

**Quick reference**:

| Model Size | Hidden Dim | Recommended LR |
|------------|------------|----------------|
| 125M | 768 | 0.020 |
| 350M | 1024 | 0.017 |
| 760M | 1536 | 0.014 |
| 1.3B | 2048 | 0.012 |
| 6.7B | 4096 | 0.009 |
| 13B | 5120 | 0.008 |

### Step 4: Tune Momentum (If Needed)

**When to adjust**:
- Training unstable? → Lower momentum (0.90-0.93)
- Converging slowly? → Higher momentum (0.97-0.99)

```python
# For unstable training
momentum = 0.90

# For slow convergence on smooth loss landscapes
momentum = 0.98
```

### Step 5: Consider Weight Decay

**Rule of thumb**: Muon often needs less weight decay than Adam.

```python
# Start without
weight_decay = 0.0

# If overfitting, add small amount
weight_decay = 0.01  # Much less than Adam's typical 0.1
```

---

## 🎛️ Newton-Schulz Steps (ns_steps)

This controls the precision of orthogonalization.

### Guidelines

| ns_steps | Precision | Speed | When to Use |
|----------|-----------|-------|-------------|
| 3 | Low | Fast | Prototyping, very stable losses |
| 5 | Medium | Medium | **Default**, most cases |
| 7 | High | Slow | Very precise training needed |
| 10 | Very High | Very Slow | Research, final runs |

```python
# Verify convergence of Newton-Schulz
def check_ns_convergence(G, ns_steps):
    X = G / G.norm()
    for i in range(ns_steps):
        A = X @ X.T
        X_new = 3.4445*X - 4.7750*A@X + 2.0315*A@A@X
        diff = (X_new - X).norm() / X.norm()
        print(f"Step {i}: relative change = {diff:.2e}")
        X = X_new
```

---

## 📊 Debugging Common Issues

### Training Explodes

**Symptoms**: Loss goes to NaN or infinity

**Solutions**:
1. Lower learning rate by 2-5x
2. Lower momentum to 0.90
3. Add gradient clipping
4. Check for bad data (NaNs in input)

```python
# Add gradient clipping
torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

optimizer = Muon(
    muon_params,
    lr=0.01,      # Lowered
    momentum=0.90 # Lowered
)
```

### Training Plateaus

**Symptoms**: Loss stops decreasing but not at good value

**Solutions**:
1. Increase learning rate
2. Add learning rate warmup
3. Increase momentum

```python
# Add warmup
from torch.optim.lr_scheduler import LinearLR, CosineAnnealingLR, SequentialLR

warmup = LinearLR(optimizer, start_factor=0.1, total_iters=500)
cosine = CosineAnnealingLR(optimizer, T_max=total_steps - 500)
scheduler = SequentialLR(optimizer, [warmup, cosine], milestones=[500])
```

### Worse than Adam

**Symptoms**: Adam clearly outperforms Muon

**Possible causes**:
1. Muon used on wrong parameters (check architecture)
2. LR not properly scaled for model size
3. Not enough training steps (Muon often catches up later)

```python
# Verify parameter assignment
for name, param in model.named_parameters():
    if param.ndim >= 2 and 'embed' not in name and 'norm' not in name:
        print(f"Muon: {name}")
    else:
        print(f"Adam: {name}")
```

---

## 🔬 Advanced: Per-Layer Learning Rates

Some practitioners report benefits from per-layer LR scaling:

```python
def get_muon_param_groups(model, base_lr=0.02):
    """Scale LR by layer position (optional)."""
    groups = []
    num_layers = len(model.transformer.layers)
    
    for i, layer in enumerate(model.transformer.layers):
        # Earlier layers: slightly lower LR
        # Later layers: base LR
        layer_lr = base_lr * (0.8 + 0.2 * i / num_layers)
        groups.append({
            'params': list(layer.parameters()),
            'lr': layer_lr
        })
    
    return groups
```

---

## 📈 Tuning Workflow Summary

```
┌─────────────────────────────────────────────────────────────┐
│  1. START: Use defaults                                     │
│     lr=0.02, momentum=0.95, ns_steps=5                     │
│                    ↓                                        │
│  2. SCALE: Adjust lr for model size                         │
│     lr ∝ 1/sqrt(hidden_dim)                                │
│                    ↓                                        │
│  3. TEST: Run LR finder or short runs                       │
│     Find range where loss decreases                         │
│                    ↓                                        │
│  4. VALIDATE: Full training run                             │
│     Compare to Adam baseline                                │
│                    ↓                                        │
│  5. REFINE: If needed, adjust momentum/wd                   │
│     Based on stability and convergence                      │
└─────────────────────────────────────────────────────────────┘
```

---

## 📚 References

- [Muon Official Repo](https://github.com/KellerJordan/Muon) - Reference implementation
- [muP Paper](https://arxiv.org/abs/2203.03466) - Scaling principles
- [NanoGPT Speedrun](https://github.com/KellerJordan/modded-nanogpt) - Working examples

---

*Added: February 2026*
