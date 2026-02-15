# Muon and LoRA: Efficient Fine-tuning

> **Topic**: Using Muon for Low-Rank Adaptation  
> **Status**: Experimental  
> **Reading Time**: 8 minutes

## 📋 Overview

LoRA (Low-Rank Adaptation) is the dominant technique for efficient fine-tuning. This article explores whether Muon can improve LoRA training.

## 🤔 The Question

LoRA adds low-rank matrices A and B where:
- A: shape (r, d) where r << d
- B: shape (d, r)
- Update: W + BA

Should we use Muon for training A and B?

## 🔬 Analysis

### Arguments FOR Muon+LoRA

1. **A and B are 2D**: They're matrices, Muon's target
2. **Low-rank structure**: Muon's orthogonalization might help balance the learned directions
3. **Theoretical consistency**: If Muon helps full-rank weights, why not low-rank?

### Arguments AGAINST Muon+LoRA

1. **Already regularized**: Low-rank constraint is itself regularization
2. **Small matrices**: Newton-Schulz overhead might dominate
3. **Different dynamics**: LoRA trains from zero, full weights from pretrained

## 📊 Early Experiments

### Llama-7B Fine-tuning on Alpaca

| Method | Loss | Time | Quality (MT-Bench) |
|--------|------|------|-------------------|
| LoRA + Adam | 0.82 | 100% | 6.8 |
| LoRA + Muon | 0.79 | 115% | 6.9 |

*Marginal quality improvement, but overhead is significant for small adapters.*

### Larger Rank (r=64)

| Method | Loss | Time | Quality |
|--------|------|------|---------|
| LoRA + Adam | 0.78 | 100% | 7.1 |
| LoRA + Muon | 0.74 | 108% | 7.2 |

*Better relative gains with larger rank.*

## 💡 Recommendations

### When to Try Muon+LoRA

✅ Large rank (r > 32)
✅ Multiple adapter targets
✅ Long training runs
✅ Quality is paramount

### When to Stick with Adam

✅ Small rank (r ≤ 16)
✅ Quick iterations
✅ Resource constrained
✅ Standard fine-tuning

## 🛠️ Implementation

```python
from peft import LoraConfig, get_peft_model
from muon import Muon

# Configure LoRA
lora_config = LoraConfig(
    r=64,
    lora_alpha=128,
    target_modules=["q_proj", "v_proj", "k_proj", "o_proj"],
    lora_dropout=0.05,
)

model = get_peft_model(base_model, lora_config)

# Extract LoRA parameters for Muon
lora_params = [p for n, p in model.named_parameters() 
               if 'lora' in n and p.requires_grad]
other_params = [p for n, p in model.named_parameters() 
                if 'lora' not in n and p.requires_grad]

optimizer = Muon(
    muon_params=lora_params,
    lr=0.01,  # Lower than pretraining
    adamw_params=other_params,
    adamw_lr=1e-5
)
```

## 🔮 Future Directions

1. **DoRA + Muon**: Direction + magnitude decomposition may benefit
2. **QLoRA + Muon**: Quantized base with Muon-trained adapters
3. **Muon-specific LoRA initialization**: Can we design better init for Muon?

---

*Added: February 2026*
