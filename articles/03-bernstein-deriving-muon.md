# Deriving Muon

**Author:** Jeremy Bernstein  
**Source:** https://jeremybernste.in/writing/deriving-muon  
**Type:** Technical blog post (theoretical derivation)

---

## Summary

Jeremy Bernstein (Muon co-creator) provides a **rigorous 4-step derivation** of Muon from first principles. This is the theoretical foundation — explains *why* the algorithm is what it is, not just *what* it does.

---

## The Muon Update Rule

```
W ← W - η × √(fan_out/fan_in) × NewtonSchulz(∇_W L)
```

Where:
- η = step size
- fan_out, fan_in = matrix dimensions
- NewtonSchulz = orthogonalization routine

---

## Derivation in 4 Steps

### Step 1: Metrize the Linear Layer

**Goal:** Equip inputs, weights, outputs with measures of "size"

For "dense" vectors (entries ~±1), use RMS norm:
```
‖v‖_RMS = √(1/d × Σ v_i²) = √(1/d) × ‖v‖_2
```

For matrices, use RMS-to-RMS operator norm:
```
‖W‖_{RMS→RMS} = max_{x≠0} ‖Wx‖_RMS / ‖x‖_RMS = √(fan_in/fan_out) × spectral_norm(W)
```

### Step 2: Perturb the Linear Layer

**Insight:** Operator norm bounds how much outputs change from weight updates.

If Δy = ΔW × x, then:
```
‖Δy‖_RMS ≤ ‖ΔW‖_{RMS→RMS} × ‖x‖_RMS    (★)
```

### Step 3: Dualize the Gradient

**Goal:** Find update that maximizes linear improvement while bounding output change.

```
min_{ΔW} ⟨∇_W L, ΔW⟩   s.t.  ‖Δy‖_RMS ≤ η
```

Using bound (★), this becomes:
```
min_{ΔW} ⟨∇_W L, ΔW⟩   s.t.  ‖ΔW‖_{RMS→RMS} ≤ η    (†)
```

**Solution:** If ∇_W L = UΣV^T (SVD), then:
```
ΔW = -η × √(fan_out/fan_in) × UV^T    (§)
```

**Intuition:** Keep singular vectors, set all singular values to η. This saturates the bound (★).

### Step 4: Newton-Schulz for Speed

**Problem:** SVD is expensive.

**Solution:** Odd polynomials commute with SVD:
```
p(X) = aX + bXX^TX + cXX^TXX^TX + ...
p(UΣV^T) = U p(Σ) V^T
```

The polynomial `p(σ) = (3/2)σ - (1/2)σ³` converges to sign(σ) when iterated.

Combined with (§), this gives Muon!

---

## Why This Matters

### Payoff #1: Learning Rate Transfer

Dualizing under RMS-to-RMS norm makes updates saturate the bound. This leads to **automatic learning rate transfer across width** — same η works for 768-dim and 16384-dim models.

### Payoff #2: Faster Training

Experiments show dualized training beats Adam by significant margins, especially at scale.

---

## Key Theoretical Papers

1. **Modular Norm (2024)** — Different layer types need different norms
2. **Modular Duality (2024)** — Framework for deriving per-layer optimizers
3. **Spectral Condition for Feature Learning (2023)** — RMS-to-RMS norm origin

---

## The Broader Vision

Muon is part of a **modular approach** to deep learning theory:
1. Break architecture into building blocks (Linear, Embedding, etc.)
2. Derive theory/algorithms for each piece
3. Glue pieces together

> "The idea is to normalize weight updates in a clever way so that, given the structure of the inputs, the weight updates automatically induce a desirable effect on the outputs."

---

## Collaborators

- Keller Jordan (implementation lead, NanoGPT speedruns)
- Laker Newhouse (theory, understanding series)
- Yuchen Jin, Vlado Boza, Jiacheng You, Franz Cesista
