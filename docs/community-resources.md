# Muon Community Resources

> Discords, forums, implementations, and community projects.

---

## 💬 Discussion & Community

### Discord Servers

| Server | Focus | Link |
|--------|-------|------|
| EleutherAI | General ML research | [discord.gg/eleutherai](https://discord.gg/eleutherai) |
| MLOps Community | Production ML | [discord.gg/mlops](https://discord.gg/mlops) |
| Weights & Biases | Experiment tracking | [discord.gg/wandb](https://discord.gg/wandb) |
| HuggingFace | Transformers ecosystem | [discord.gg/huggingface](https://discord.gg/huggingface) |

### Forums & Discussions

| Platform | Muon Discussions |
|----------|------------------|
| Twitter/X | [@kellerjordan](https://twitter.com/kellerjordan_) (creator) |
| Reddit | r/MachineLearning threads |
| HN | Kimi K2 & NanoGPT speedrun discussions |
| GitHub Issues | [github.com/KellerJordan/Muon/issues](https://github.com/KellerJordan/Muon/issues) |

---

## 📦 Official Resources

### Code Repositories

| Repo | Description | Link |
|------|-------------|------|
| Muon | Official PyTorch implementation | [github.com/KellerJordan/Muon](https://github.com/KellerJordan/Muon) |
| modded-nanogpt | NanoGPT speedrun with Muon | [github.com/KellerJordan/modded-nanogpt](https://github.com/KellerJordan/modded-nanogpt) |
| muon-pytorch | PyPI package | [pypi.org/project/muon-pytorch](https://pypi.org/project/muon-pytorch) |

### Documentation

| Resource | Link |
|----------|------|
| Keller Jordan's Blog | [kellerjordan.github.io](https://kellerjordan.github.io) |
| Jeremy Bernstein's Site | [jeremybernste.in](https://jeremybernste.in) |
| Laker Newhouse's Site | [lakernewhouse.com](https://lakernewhouse.com) |
| Modula Systems | [modula.systems](https://modula.systems) |

---

## 🛠️ Community Implementations

### Framework Ports

| Framework | Implementation | Status |
|-----------|----------------|--------|
| PyTorch | Official | ✅ Production |
| JAX/Optax | Community | ✅ Tested |
| TensorFlow | Community | 🔄 Experimental |
| MLX (Apple) | Community | 🔄 Experimental |

### JAX Implementation

```python
# Community JAX port
# github.com/[community]/muon-jax
import jax
import optax

def muon_jax(lr=0.02, momentum=0.95, ns_steps=5):
    """JAX/Optax implementation of Muon."""
    def init_fn(params):
        return {'momentum_buffer': jax.tree_map(jnp.zeros_like, params)}
    
    def update_fn(updates, state, params):
        # Newton-Schulz orthogonalization
        updates = newton_schulz_jax(updates, ns_steps)
        
        # Momentum
        new_momentum = jax.tree_map(
            lambda m, g: momentum * m + g,
            state['momentum_buffer'], updates
        )
        
        # Update
        updates = jax.tree_map(lambda m: -lr * m, new_momentum)
        
        return updates, {'momentum_buffer': new_momentum}
    
    return optax.GradientTransformation(init_fn, update_fn)
```

---

## 📊 Benchmarks & Leaderboards

### NanoGPT Speedrun Leaderboard

Track the latest records at:
[github.com/KellerJordan/modded-nanogpt#leaderboard](https://github.com/KellerJordan/modded-nanogpt#leaderboard)

### Current Records (as of Feb 2026)

| Target | Time | Hardware | Method |
|--------|------|----------|--------|
| Val loss 3.28 | 2.92 A100-hrs | 8x A100 | Muon + tricks |
| Val loss 3.20 | 4.12 A100-hrs | 8x A100 | Muon + scaling |
| CIFAR-10 94% | 2.6 A100-sec | 1x A100 | Muon |

---

## 📚 Educational Resources

### Video Content

| Title | Creator | Link |
|-------|---------|------|
| "Understanding Muon" | ML Street Talk | YouTube |
| "NanoGPT Speedrun Deep Dive" | Yannic Kilcher | YouTube |
| "Optimizer Theory" | 3Blue1Brown (related) | YouTube |

### Blog Posts & Articles

| Title | Author | Read Time |
|-------|--------|-----------|
| "Muon: An optimizer for hidden layers" | Keller Jordan | 15 min |
| "Enter the Matrix: Understanding Muon" | Laker Newhouse | 20 min |
| "Deriving Muon from First Principles" | Jeremy Bernstein | 25 min |
| "Old Optimizer, New Norm" | Bernstein & Newhouse | 30 min |

### Tutorials

| Tutorial | Level | Link |
|----------|-------|------|
| Quick Start with Muon | Beginner | This repo |
| Implementing Muon from Scratch | Intermediate | Community |
| Scaling Muon to Production | Advanced | Essential AI |

---

## 🔬 Research Extensions

### Active Research Directions

1. **Muon + LoRA**: Can Muon improve low-rank adaptation?
2. **Muon for Vision**: Extending beyond language models
3. **Distributed Muon**: Efficient distributed implementation
4. **Muon + muP**: Hyperparameter transfer at scale

### Papers Building on Muon

| Paper | Key Extension |
|-------|---------------|
| "Muon is Scalable" | Scaling analysis |
| "Practical Efficiency of Muon" | Production deployment |
| "Kimi K2" | Trillion-parameter training |

---

## 🤝 Contributing

### Ways to Contribute

1. **Report bugs**: Open GitHub issues
2. **Share benchmarks**: Post your results
3. **Port to frameworks**: TensorFlow, MLX, etc.
4. **Improve docs**: PRs welcome
5. **Write tutorials**: Help others learn

### Contribution Guidelines

```markdown
For code contributions:
1. Fork the repository
2. Create a feature branch
3. Add tests for new functionality
4. Ensure all tests pass
5. Submit a pull request

For documentation:
1. Check existing docs for style
2. Use clear, concise language
3. Include code examples
4. Test examples before submitting
```

---

## 📰 Stay Updated

### Newsletters & Feeds

| Source | Frequency | Coverage |
|--------|-----------|----------|
| Import AI | Weekly | Broad ML news |
| The Batch | Weekly | DeepLearning.AI |
| Papers With Code | Daily | New papers |
| arXiv RSS (cs.LG) | Daily | Latest preprints |

### Key People to Follow

| Name | Platform | Focus |
|------|----------|-------|
| Keller Jordan | Twitter/X | Muon creator |
| Jeremy Bernstein | Twitter/X | Theory |
| Laker Newhouse | Twitter/X | Intuition |
| Jianlin Su | Twitter/X | Scaling |

---

## 📞 Getting Help

### Before Asking

1. Check [FAQ](faq.md)
2. Search GitHub Issues
3. Read [Troubleshooting](troubleshooting.md)
4. Try minimal reproduction

### Where to Ask

| Question Type | Where |
|---------------|-------|
| Bug reports | GitHub Issues |
| Usage questions | GitHub Discussions |
| Theory questions | Twitter/X, Reddit |
| General ML | EleutherAI Discord |

### How to Ask Good Questions

```markdown
## Environment
- PyTorch version:
- CUDA version:
- muon-pytorch version:

## What I'm trying to do
[Clear description]

## What I expected
[Expected behavior]

## What happened
[Actual behavior]

## Minimal reproduction
[Code snippet]
```

---

*Updated: February 2026*
