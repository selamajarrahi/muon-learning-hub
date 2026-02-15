# ✅ Muon Implementation Checklist

> A step-by-step guide to adopting Muon in your training pipeline

## Pre-Implementation Assessment

### Phase 0: Should You Use Muon?

- [ ] **Architecture check:** Does your model have significant 2D weight matrices?
  - Transformers: ✅ Yes (attention, MLPs)
  - MLPs: ✅ Yes
  - CNNs: ⚠️ Maybe (needs kernel reshaping)
  - RNNs/LSTMs: ✅ Yes
  
- [ ] **Scale check:** Are your weight matrices large enough to benefit?
  - Hidden dim ≥ 256: ✅ Recommended
  - Hidden dim < 128: ⚠️ Overhead may not be worth it

- [ ] **Training time matters:** Are you bottlenecked by training speed?
  - Yes: ✅ Muon can help (25-35% faster)
  - No: 🤷 Optional optimization

---

## Implementation Steps

### Phase 1: Parameter Classification

- [ ] **Identify 2D weight matrices**
  ```python
  muon_candidates = []
  for name, param in model.named_parameters():
      if param.ndim == 2:
          print(f"{name}: {param.shape}")
          muon_candidates.append(name)
  ```

- [ ] **Exclude embeddings**
  - Token embeddings
  - Position embeddings  
  - Any lookup tables

- [ ] **Exclude small matrices**
  - Threshold: < 256 elements total
  - Or: either dimension < 16

- [ ] **List Adam parameters**
  - All biases
  - LayerNorm/RMSNorm weights
  - All 1D parameters
  - Excluded 2D matrices

### Phase 2: Implement Newton-Schulz

- [ ] **Core function**
  ```python
  @torch.no_grad()
  def newtonschulz5(G, steps=5):
      assert G.ndim == 2
      a, b, c = (3.4445, -4.7750, 2.0315)
      X = G.float() / (G.norm() + 1e-7)
      for _ in range(steps):
          A = X @ X.T
          B = b * A + c * A @ A
          X = a * X + B @ X
      return X.type_as(G)
  ```

- [ ] **Numerical stability checks**
  - Normalize before iteration ✓
  - Use fp32 for iterations ✓
  - Handle zero gradients (add eps) ✓

### Phase 3: Build Hybrid Optimizer

- [ ] **Create parameter groups**
  ```python
  def create_param_groups(model):
      muon_params = []
      adam_params = []
      
      for name, param in model.named_parameters():
          if not param.requires_grad:
              continue
          if should_use_muon(name, param):
              muon_params.append(param)
          else:
              adam_params.append(param)
      
      return muon_params, adam_params
  
  def should_use_muon(name, param):
      if param.ndim != 2:
          return False
      if param.numel() < 256:
          return False
      if any(x in name.lower() for x in ['embed', 'norm']):
          return False
      return True
  ```

- [ ] **Initialize optimizers**
  ```python
  muon_params, adam_params = create_param_groups(model)
  
  optimizer = Muon(
      muon_params=muon_params,
      lr=0.02,
      momentum=0.95,
      adamw_params=adam_params,
      adamw_lr=3e-4,
      adamw_betas=(0.9, 0.95),
  )
  ```

### Phase 4: Distributed Training (if applicable)

- [ ] **Gradient synchronization order**
  ```python
  # Option A: Use hook for all-reduce before ortho
  def gradient_hook(param):
      def hook(grad):
          dist.all_reduce(grad)
          return grad
      return hook
  
  for param in muon_params:
      param.register_hook(gradient_hook(param))
  ```

- [ ] **Verify sync happens BEFORE Newton-Schulz**
  - All-reduce on raw gradients ✓
  - Then orthogonalize locally ✓

### Phase 5: Learning Rate Configuration

- [ ] **Set initial learning rates**
  | Component | Starting LR |
  |-----------|-------------|
  | Muon params | 0.02 |
  | Adam params | 3e-4 |

- [ ] **Configure scheduler**
  ```python
  # Example: Cosine decay for both
  scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
      optimizer, 
      T_max=total_steps
  )
  ```

- [ ] **Warmup (optional but recommended)**
  - 1-5% of total steps
  - Linear warmup works well

---

## Testing & Validation

### Phase 6: Sanity Checks

- [ ] **Loss decreases on toy data**
  ```python
  # Quick test with small batch
  for _ in range(100):
      loss = model(toy_batch).mean()
      loss.backward()
      optimizer.step()
      optimizer.zero_grad()
      print(loss.item())  # Should decrease
  ```

- [ ] **No NaN/Inf in gradients**
  ```python
  for name, param in model.named_parameters():
      if param.grad is not None:
          assert torch.isfinite(param.grad).all(), f"Bad grad in {name}"
  ```

- [ ] **Compare with Adam baseline**
  - Same init, same data
  - Muon should match or beat after N steps

### Phase 7: Monitoring

- [ ] **Add logging for:**
  - Gradient norms (pre-ortho)
  - Orthogonalized gradient norms
  - Parameter update magnitudes
  - Loss curve

- [ ] **Watch for:**
  - Loss spikes (reduce lr)
  - Gradient explosions (add clipping)
  - Slow convergence (check param classification)

---

## Production Deployment

### Phase 8: Optimization

- [ ] **Fuse operations (if possible)**
  - Custom CUDA kernel for Newton-Schulz
  - Or use torch.compile()

- [ ] **Memory optimization**
  - Recompute ortho_grad if memory-tight
  - Gradient checkpointing compatible

- [ ] **Profile performance**
  ```python
  with torch.profiler.profile() as prof:
      # Training step
      ...
  print(prof.key_averages().table())
  ```

### Phase 9: Hyperparameter Tuning

- [ ] **Learning rate sweep**
  - Muon: try [0.01, 0.02, 0.03, 0.05]
  - Adam: try [1e-4, 3e-4, 1e-3]

- [ ] **Momentum (usually keep default)**
  - Default: 0.95
  - Range: [0.9, 0.99]

- [ ] **Batch size interaction**
  - Larger batches often allow higher lr
  - Linear scaling rule may apply

### Phase 10: Documentation

- [ ] **Document your configuration**
  ```yaml
  optimizer:
    type: muon
    muon:
      lr: 0.02
      momentum: 0.95
      params: "2D weights >= 256 elements, excluding embeddings"
    adam:
      lr: 3e-4
      betas: [0.9, 0.95]
      params: "embeddings, norms, biases, small matrices"
  ```

- [ ] **Record baseline comparison**
  - Training curves
  - Final metrics
  - Wall-clock time

---

## Quick Reference Card

```
╔═══════════════════════════════════════════════════════════════╗
║                    MUON QUICK REFERENCE                        ║
╠═══════════════════════════════════════════════════════════════╣
║                                                                 ║
║  Muon params:  2D matrices, hidden_dim ≥ 256, no embeddings    ║
║  Adam params:  Everything else                                  ║
║                                                                 ║
║  Muon LR:      0.02 (range: 0.01 - 0.05)                       ║
║  Adam LR:      3e-4 (range: 1e-4 - 1e-3)                       ║
║                                                                 ║
║  NS iterations: 5                                               ║
║  NS precision:  fp32 (cast back after)                         ║
║                                                                 ║
║  Distributed:   all-reduce BEFORE orthogonalize                ║
║                                                                 ║
╚═══════════════════════════════════════════════════════════════╝
```

---

## Troubleshooting Quick Links

- [Common Mistakes](common-mistakes.md)
- [FAQ](faq.md)
- [Muon vs Adam Comparison](muon-vs-adam.md)
