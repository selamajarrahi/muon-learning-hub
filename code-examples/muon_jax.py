"""
Muon Optimizer Implementation in JAX/Optax
==========================================

A JAX implementation of the Muon optimizer using Optax-style transforms.

Usage:
    import optax
    from muon_jax import muon

    optimizer = muon(learning_rate=0.02, momentum=0.95)
    opt_state = optimizer.init(params)
    
    # Training step
    grads = jax.grad(loss_fn)(params, batch)
    updates, opt_state = optimizer.update(grads, opt_state, params)
    params = optax.apply_updates(params, updates)

Requirements:
    pip install jax jaxlib optax

Author: Muon Learning Hub Contributors
License: MIT
"""

import jax
import jax.numpy as jnp
from typing import NamedTuple, Optional, Callable
import optax
from optax import GradientTransformation


# ============================================================================
# Newton-Schulz Orthogonalization
# ============================================================================

def newtonschulz5(G: jnp.ndarray, steps: int = 5) -> jnp.ndarray:
    """
    Approximate orthogonalization using Newton-Schulz iteration.
    
    Args:
        G: Gradient matrix of shape (m, n)
        steps: Number of iterations (default: 5)
    
    Returns:
        Approximately orthogonalized matrix X ≈ UV^T where G = UΣV^T
    """
    # Optimal 5th-order polynomial coefficients
    a, b, c = 3.4445, -4.7750, 2.0315
    
    # Normalize for convergence
    X = G / (jnp.linalg.norm(G) + 1e-7)
    
    def iteration(X, _):
        A = X @ X.T
        B = b * A + c * A @ A
        X = a * X + B @ X
        return X, None
    
    X, _ = jax.lax.scan(iteration, X, None, length=steps)
    return X


def orthogonalize_gradient(grad: jnp.ndarray) -> jnp.ndarray:
    """
    Apply Newton-Schulz orthogonalization if gradient is 2D and large enough.
    """
    if grad.ndim != 2 or grad.size < 256:
        return grad
    
    # Ensure m <= n by transposing if needed
    transposed = grad.shape[0] > grad.shape[1]
    if transposed:
        grad = grad.T
    
    ortho_grad = newtonschulz5(grad)
    
    if transposed:
        ortho_grad = ortho_grad.T
    
    return ortho_grad


# ============================================================================
# Optax-Style Muon Transform
# ============================================================================

class MuonState(NamedTuple):
    """State for Muon optimizer."""
    momentum: optax.Params  # Momentum buffers
    count: jnp.ndarray  # Step counter


def scale_by_muon(
    momentum: float = 0.95,
    nesterov: bool = False,
) -> GradientTransformation:
    """
    Scale gradients using Muon-style orthogonalization with momentum.
    
    Args:
        momentum: Momentum coefficient (default: 0.95)
        nesterov: Whether to use Nesterov momentum (default: False)
    
    Returns:
        An Optax GradientTransformation
    """
    
    def init_fn(params):
        momentum_buffer = jax.tree.map(jnp.zeros_like, params)
        return MuonState(momentum=momentum_buffer, count=jnp.zeros([], jnp.int32))
    
    def update_fn(updates, state, params=None):
        del params  # Unused
        
        # Apply Newton-Schulz orthogonalization to 2D parameters
        ortho_updates = jax.tree.map(orthogonalize_gradient, updates)
        
        # Apply momentum
        new_momentum = jax.tree.map(
            lambda m, g: momentum * m + g,
            state.momentum, ortho_updates
        )
        
        if nesterov:
            # Nesterov: use momentum + current gradient direction
            updates = jax.tree.map(
                lambda m, g: momentum * m + g,
                new_momentum, ortho_updates
            )
        else:
            updates = new_momentum
        
        # Negate for gradient descent
        updates = jax.tree.map(lambda u: -u, updates)
        
        return updates, MuonState(momentum=new_momentum, count=state.count + 1)
    
    return GradientTransformation(init_fn, update_fn)


def muon(
    learning_rate: float = 0.02,
    momentum: float = 0.95,
    weight_decay: float = 0.0,
    nesterov: bool = False,
) -> GradientTransformation:
    """
    The Muon optimizer.
    
    Muon (MomentUm Orthogonalized by Newton-schulz) applies Newton-Schulz
    orthogonalization to gradients before momentum, providing faster
    convergence for matrix parameters.
    
    Args:
        learning_rate: Step size (default: 0.02)
        momentum: Momentum coefficient (default: 0.95)
        weight_decay: Weight decay coefficient (default: 0.0)
        nesterov: Use Nesterov momentum (default: False)
    
    Returns:
        An Optax optimizer
    
    Example:
        >>> optimizer = muon(learning_rate=0.02)
        >>> opt_state = optimizer.init(params)
        >>> grads = jax.grad(loss_fn)(params, batch)
        >>> updates, opt_state = optimizer.update(grads, opt_state)
        >>> params = optax.apply_updates(params, updates)
    """
    return optax.chain(
        scale_by_muon(momentum=momentum, nesterov=nesterov),
        optax.add_decayed_weights(weight_decay) if weight_decay > 0 else optax.identity(),
        optax.scale(learning_rate),
    )


# ============================================================================
# Hybrid Optimizer (Muon + Adam)
# ============================================================================

def partition_params(
    params,
    muon_filter: Callable = None,
) -> tuple:
    """
    Partition parameters into Muon and Adam groups.
    
    Args:
        params: Parameter tree
        muon_filter: Function (path, param) -> bool for Muon params
                     Default: 2D params with >= 256 elements
    
    Returns:
        (muon_mask, adam_mask) as boolean trees
    """
    if muon_filter is None:
        def muon_filter(path, param):
            return param.ndim == 2 and param.size >= 256
    
    def create_mask(params, prefix=""):
        if isinstance(params, dict):
            return {k: create_mask(v, f"{prefix}.{k}") for k, v in params.items()}
        else:
            return muon_filter(prefix, params)
    
    muon_mask = create_mask(params)
    adam_mask = jax.tree.map(lambda x: not x, muon_mask)
    
    return muon_mask, adam_mask


def hybrid_muon_adam(
    muon_lr: float = 0.02,
    muon_momentum: float = 0.95,
    adam_lr: float = 3e-4,
    adam_b1: float = 0.9,
    adam_b2: float = 0.95,
    weight_decay: float = 0.0,
    muon_filter: Callable = None,
) -> tuple[GradientTransformation, Callable]:
    """
    Create a hybrid Muon + Adam optimizer.
    
    Returns the optimizer AND a partition function that must be called
    with params to create the proper masks.
    
    Args:
        muon_lr: Learning rate for Muon parameters
        muon_momentum: Momentum for Muon
        adam_lr: Learning rate for Adam parameters
        adam_b1, adam_b2: Adam beta parameters
        weight_decay: Weight decay for both optimizers
        muon_filter: Custom filter for Muon params
    
    Returns:
        (optimizer, partition_fn) tuple
    
    Example:
        >>> optimizer, partition = hybrid_muon_adam()
        >>> labels = partition(params)  # Call once with your params
        >>> opt_state = optimizer.init(params)
    """
    muon_opt = muon(
        learning_rate=muon_lr,
        momentum=muon_momentum,
        weight_decay=weight_decay,
    )
    
    adam_opt = optax.chain(
        optax.scale_by_adam(b1=adam_b1, b2=adam_b2),
        optax.add_decayed_weights(weight_decay) if weight_decay > 0 else optax.identity(),
        optax.scale(-adam_lr),
    )
    
    def partition_fn(params):
        muon_mask, adam_mask = partition_params(params, muon_filter)
        return {"muon": muon_mask, "adam": adam_mask}
    
    optimizer = optax.multi_transform(
        {"muon": muon_opt, "adam": adam_opt},
        partition_fn,
    )
    
    return optimizer, partition_fn


# ============================================================================
# Example Usage
# ============================================================================

if __name__ == "__main__":
    import jax.random as random
    
    # Create dummy model parameters
    key = random.PRNGKey(42)
    params = {
        "embed": random.normal(key, (1000, 256)),  # Embedding - use Adam
        "layer1": {
            "weight": random.normal(key, (256, 512)),  # Linear - use Muon
            "bias": random.normal(key, (512,)),  # Bias - use Adam
        },
        "layer2": {
            "weight": random.normal(key, (512, 256)),  # Linear - use Muon
            "bias": random.normal(key, (256,)),  # Bias - use Adam
        },
        "norm": {
            "scale": random.normal(key, (256,)),  # LayerNorm - use Adam
        },
    }
    
    # Simple Muon (all params)
    print("=== Simple Muon ===")
    optimizer = muon(learning_rate=0.02)
    opt_state = optimizer.init(params)
    
    # Dummy gradient
    grads = jax.tree.map(jnp.ones_like, params)
    updates, opt_state = optimizer.update(grads, opt_state)
    new_params = optax.apply_updates(params, updates)
    print("Update applied successfully!")
    
    # Hybrid Muon + Adam
    print("\n=== Hybrid Muon + Adam ===")
    optimizer, partition = hybrid_muon_adam(
        muon_lr=0.02,
        adam_lr=3e-4,
    )
    
    # Show which params use which optimizer
    labels = partition(params)
    print("Muon params:", jax.tree.map(lambda x: x, labels))
    
    opt_state = optimizer.init(params)
    updates, opt_state = optimizer.update(grads, opt_state, params)
    new_params = optax.apply_updates(params, updates)
    print("Hybrid update applied successfully!")
    
    # Verify Newton-Schulz works
    print("\n=== Newton-Schulz Test ===")
    test_matrix = random.normal(key, (64, 128))
    ortho_matrix = newtonschulz5(test_matrix)
    
    # Check orthogonality: X @ X.T should be close to identity
    should_be_identity = ortho_matrix @ ortho_matrix.T
    identity_error = jnp.mean((should_be_identity - jnp.eye(64)) ** 2)
    print(f"Orthogonality error (should be small): {identity_error:.6f}")
