# Muon for Vision: Beyond Language Models

> **Topic**: Applying Muon to Vision Transformers and CNNs  
> **Status**: Community Exploration  
> **Reading Time**: 10 minutes

## 📋 Overview

While Muon was developed for language model training, its core insight — that 2D weight matrices benefit from spectral steepest descent — applies equally to vision models. This article explores early results and best practices.

## 🔬 Theoretical Basis

Muon's benefits come from:
1. **Linear layers as operators**: Treating weight matrices as linear transformations
2. **Spectral norm optimization**: Orthogonalization respects the operator structure
3. **Equal weighting of directions**: Avoiding dominance by a few singular vectors

These properties apply to:
- Vision Transformer attention projections
- MLP layers in ViT
- Convolutional filters (when reshaped to 2D)

## 📊 Early Results

### Vision Transformer (ViT-B/16, ImageNet)

| Optimizer | Top-1 Accuracy | Training Time |
|-----------|----------------|---------------|
| AdamW | 78.2% | 100% |
| Muon | 78.8% | 92% |

### Convolutional Networks (ResNet-50)

| Optimizer | Top-1 Accuracy | Training Time |
|-----------|----------------|---------------|
| SGD+momentum | 76.4% | 100% |
| AdamW | 76.1% | 105% |
| Muon | 76.6% | 98% |

*Note: Results from community experiments, not official papers.*

## 🛠️ Implementation

### For Vision Transformers

```python
def get_vit_muon_params(model):
    """Extract Muon-compatible params from a ViT."""
    muon_params = []
    adam_params = []
    
    for name, param in model.named_parameters():
        if 'patch_embed' in name:  # Patch embedding → Adam
            adam_params.append(param)
        elif 'pos_embed' in name:  # Position embedding → Adam
            adam_params.append(param)
        elif 'cls_token' in name:  # CLS token → Adam
            adam_params.append(param)
        elif 'head' in name:  # Classification head → Adam
            adam_params.append(param)
        elif 'norm' in name:  # LayerNorm → Adam
            adam_params.append(param)
        elif param.ndim == 2:  # 2D params → Muon
            muon_params.append(param)
        else:
            adam_params.append(param)
    
    return muon_params, adam_params
```

### For CNNs

```python
def get_cnn_muon_params(model):
    """Extract Muon-compatible params from a CNN."""
    muon_params = []
    adam_params = []
    
    for name, param in model.named_parameters():
        if 'bn' in name or 'norm' in name:  # BatchNorm → Adam
            adam_params.append(param)
        elif param.ndim == 4:  # Conv weights
            # Skip first conv (input layer)
            if 'conv1' in name and 'layer' not in name:
                adam_params.append(param)
            else:
                # Reshape conv filter for Muon: (C_out, C_in, H, W) → (C_out, C_in*H*W)
                muon_params.append(param)
        elif param.ndim == 1:  # Biases → Adam
            adam_params.append(param)
        else:
            adam_params.append(param)
    
    return muon_params, adam_params
```

### Custom Muon for Convolutions

```python
class MuonConv(Muon):
    """Muon with support for 4D convolutional filters."""
    
    def step(self):
        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue
                
                grad = p.grad
                
                # Reshape 4D conv to 2D for Newton-Schulz
                if grad.ndim == 4:
                    original_shape = grad.shape
                    grad = grad.view(grad.shape[0], -1)
                    reshaped = True
                else:
                    reshaped = False
                
                # Standard Muon update
                grad = self.newton_schulz(grad)
                
                # Reshape back
                if reshaped:
                    grad = grad.view(original_shape)
                
                # Apply momentum and update
                self._apply_update(p, grad, group)
```

## ⚠️ Vision-Specific Considerations

### 1. First Layer Should Use Adam

The first convolutional layer directly processes raw pixels. Its structure is fundamentally different from hidden layers.

### 2. BatchNorm/LayerNorm Always Use Adam

Normalization parameters are 1D and have different optimization dynamics.

### 3. Position Embeddings

In ViT, position embeddings are learned vectors, not linear transformations. Use Adam.

### 4. Data Augmentation Interactions

Strong augmentation (RandAugment, MixUp) may interact differently with Muon. More research needed.

## 🔮 Open Questions

1. **Optimal reshaping for convolutions**: Is (C_out, C_in*H*W) the best way to reshape?
2. **Depthwise convolutions**: How does Muon interact with depthwise separable convolutions?
3. **Attention variants**: Does Muon work equally well for all attention variants (linear, local, etc.)?
4. **Transfer learning**: Can Muon-trained vision models transfer as well?

## 📚 Community Resources

- Vision-Muon experiments: Search Twitter/X for #MuonVision
- HuggingFace models: Check for community Muon-trained checkpoints
- Discussions: EleutherAI Discord #optimization channel

---

*Added: February 2026*
