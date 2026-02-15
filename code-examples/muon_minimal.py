"""
Minimal Muon Implementation

A from-scratch implementation for learning purposes.
For production, use: https://github.com/KellerJordan/Muon
"""

import torch
import torch.nn as nn
from typing import Iterable, Optional


def newton_schulz(G: torch.Tensor, steps: int = 5, eps: float = 1e-7) -> torch.Tensor:
    """
    Orthogonalize a matrix using Newton-Schulz iteration.
    
    Given G with SVD G = UΣV^T, returns approximately UV^T.
    
    The magic: odd polynomials commute with SVD!
        p(UΣV^T) = U p(Σ) V^T
    
    So we apply a polynomial that pushes singular values toward 1.
    """
    # Tuned coefficients for fast convergence
    # These form polynomial: p(x) = 3.4445x - 4.7750x³ + 2.0315x⁵
    a, b, c = (3.4445, -4.7750, 2.0315)
    
    # Work in bfloat16 for efficiency (Newton-Schulz is stable in bf16!)
    X = G.bfloat16()
    
    # Normalize so singular values < 1 (required for convergence)
    X = X / (X.norm() + eps)
    
    # Handle non-square matrices by working with the smaller dimension
    transposed = G.size(0) > G.size(1)
    if transposed:
        X = X.T
    
    # Newton-Schulz iterations
    for _ in range(steps):
        A = X @ X.T                    # A has singular values σ²
        B = b * A + c * A @ A          # Polynomial in σ²
        X = a * X + B @ X              # Full polynomial applied
    
    if transposed:
        X = X.T
    
    return X.to(G.dtype)


class Muon(torch.optim.Optimizer):
    """
    Muon: Momentum Orthogonalized by Newton-Schulz
    
    For 2D parameters in hidden layers of neural networks.
    
    Usage:
        # Separate params by dimension
        muon_params = [p for p in model.parameters() if p.ndim == 2]
        other_params = [p for p in model.parameters() if p.ndim != 2]
        
        # Use Muon for 2D, AdamW for rest
        optimizer = torch.optim.AdamW(other_params, lr=3e-4)
        muon = Muon(muon_params, lr=0.02, momentum=0.95)
    """
    
    def __init__(
        self,
        params: Iterable[torch.Tensor],
        lr: float = 0.02,
        momentum: float = 0.95,
        ns_steps: int = 5,
    ):
        defaults = dict(lr=lr, momentum=momentum, ns_steps=ns_steps)
        super().__init__(params, defaults)
    
    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        
        for group in self.param_groups:
            lr = group['lr']
            momentum = group['momentum']
            ns_steps = group['ns_steps']
            
            for p in group['params']:
                if p.grad is None:
                    continue
                
                grad = p.grad
                
                # Get or initialize momentum buffer
                state = self.state[p]
                if 'momentum_buffer' not in state:
                    state['momentum_buffer'] = torch.zeros_like(grad)
                
                buf = state['momentum_buffer']
                
                # Update momentum: M = β*M + G
                buf.mul_(momentum).add_(grad)
                
                # Orthogonalize momentum
                update = newton_schulz(buf, steps=ns_steps)
                
                # Scale by sqrt(fan_out/fan_in) for dimension-independence
                fan_out, fan_in = p.shape[0], p.shape[1]
                scale = (fan_out / fan_in) ** 0.5
                
                # Apply update
                p.add_(update, alpha=-lr * scale)
        
        return loss


# =============================================================================
# Demo: Train a small MLP on synthetic data
# =============================================================================

if __name__ == "__main__":
    torch.manual_seed(42)
    
    # Simple MLP
    model = nn.Sequential(
        nn.Linear(64, 256),
        nn.ReLU(),
        nn.Linear(256, 256),
        nn.ReLU(),
        nn.Linear(256, 10),
    )
    
    # Synthetic data
    X = torch.randn(1000, 64)
    y = torch.randint(0, 10, (1000,))
    
    # Separate params: Muon for 2D, AdamW for 1D (biases)
    muon_params = [p for n, p in model.named_parameters() if 'weight' in n]
    adamw_params = [p for n, p in model.named_parameters() if 'bias' in n]
    
    muon_opt = Muon(muon_params, lr=0.02, momentum=0.95)
    adamw_opt = torch.optim.AdamW(adamw_params, lr=3e-4)
    
    criterion = nn.CrossEntropyLoss()
    
    # Training loop
    for epoch in range(10):
        model.train()
        
        # Forward
        logits = model(X)
        loss = criterion(logits, y)
        
        # Backward
        loss.backward()
        
        # Step both optimizers
        muon_opt.step()
        adamw_opt.step()
        
        # Zero grads
        muon_opt.zero_grad()
        adamw_opt.zero_grad()
        
        # Accuracy
        with torch.no_grad():
            pred = logits.argmax(dim=1)
            acc = (pred == y).float().mean()
        
        print(f"Epoch {epoch+1:2d} | Loss: {loss.item():.4f} | Acc: {acc.item():.2%}")
    
    print("\n✓ Muon training complete!")
