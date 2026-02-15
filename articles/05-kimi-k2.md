# Kimi K2: Open Agentic Intelligence

**Authors:** Kimi Team (Moonshot AI)  
**Source:** https://arxiv.org/abs/2507.20534  
**Type:** Technical report

---

## Summary

Kimi K2 is a **1 trillion parameter** mixture-of-experts (MoE) model trained by Moonshot AI. It's the first known production model at this scale to use **Muon optimizer**.

---

## Why This Matters for Muon

This paper proves Muon scales:
- **1T active parameters** (32B active per token)
- **15.5T training tokens**
- Production-quality results

Before K2, largest Muon training was ~1.5B params. This is a **~700x scale increase**.

---

## K2 Architecture

| Spec | Value |
|------|-------|
| Total params | 1T |
| Active params | 32B |
| Experts | 384 per layer |
| Layers | 61 |
| Hidden dim | 7168 |
| Context | 128K tokens |

---

## Training Details

**Optimizer:** Muon (for Linear layers) + AdamW (for Embedding/LayerNorm)

**Key Muon settings (inferred):**
- LR: ~0.02 for Muon params
- Momentum: 0.95
- Newton-Schulz iterations: 5
- bf16 computation

**Infrastructure:**
- Trained on distributed GPU clusters
- Custom parallelization for MoE

---

## Performance

K2 achieves SOTA or near-SOTA on:
- **Coding:** SWE-Bench, HumanEval
- **Reasoning:** MATH, GSM8K
- **General:** MMLU, ARC
- **Agentic:** Tool use benchmarks

The "agentic" focus means K2 is optimized for tool use, multi-step reasoning, and autonomous task completion.

---

## Muon at Scale Observations

1. **Learning rate transfer works:** Same η effective across model sizes
2. **Stability:** No special stabilization needed beyond standard practice
3. **Efficiency:** <1% FLOP overhead maintained at scale
4. **Memory:** Same as AdamW (momentum buffer only)

---

## Quote from Paper

> "We adopt the Muon optimizer for training, which has demonstrated superior efficiency in language model pretraining compared to conventional Adam-based methods."

---

## Implications

1. **Muon is production-ready** at frontier scale
2. Opens door for other labs to adopt
3. Validates theoretical foundations (RMS-to-RMS norm, orthogonalization)
4. Suggests further scaling is viable

---

## Links

- [Kimi K2 Project Page](https://moonshotai.github.io/Kimi-K2/)
- [arXiv Paper](https://arxiv.org/abs/2507.20534)
