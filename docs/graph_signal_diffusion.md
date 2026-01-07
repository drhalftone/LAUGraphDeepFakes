# Graph Signal Diffusion for Data Augmentation

## Goal

Generate realistic graph signals on a fixed mesh using diffusion models. Two use cases:

1. **Generation**: Create new graph signals from pure noise
2. **Augmentation**: Create variations of existing signals (similar but not identical)

## The Problem

We have graph signals (e.g., flag vertex positions at a single timestep) and want to:
- Generate new signals that look realistic
- Create variations of existing signals for data augmentation
- Train other GNNs with more diverse data

## Background

### What we tried

1. **Spectral VAE on FEA data** - Limited variation (training signals too similar)
2. **Noise filtering on flag data** - Failed to capture:
   - Boundary conditions (fixed pole edge)
   - Spatial coherence (nearby vertices move together)

### Why diffusion

Diffusion models (Stable Diffusion, DALL-E 3) are state-of-the-art for generation because they:
- Learn implicit constraints from data
- Generate diverse, high-quality outputs
- Support **controllable augmentation** via partial denoising

### Related work

[Graph-Aware Diffusion (GAD)](https://arxiv.org/abs/2510.05036) does exactly this for static graph signals. Our implementation is similar but uses GNN message passing instead of polynomial graph filters.

## Data

From DeepMind's MeshGraphNets flag simulation:

```
flag_data/
├── meta.json           # Dataset metadata
├── test.tfrecord       # Raw simulation data
└── flag_test.npz       # Converted numpy
```

### Data structure

| Field | Shape | Description |
|-------|-------|-------------|
| `world_pos` | (50, 401, 1579, 3) | 50 simulations × 401 timesteps × 1579 vertices × xyz |
| `cells` | (3028, 3) | Triangle mesh connectivity |
| `mesh_pos` | (1579, 2) | Rest state vertex positions |

### Training samples

We treat each frame independently:
- 50 simulations × 401 frames = **20,050 training samples**
- Each sample is a single graph signal: (V, 3) = (1579, 3)

## Diffusion Overview

### Two "times" (don't confuse them!)

| Term | What it is | Range |
|------|------------|-------|
| Simulation time | Frame index in the animation | 0 → 400 |
| **Noise step** | Diffusion denoising iteration | 1000 → 0 |

We only care about **noise step** for diffusion. Each frame is treated independently.

### Forward process (training)

Add noise to a clean signal:

```
x_noisy = √ᾱₙ · x_clean + √(1-ᾱₙ) · ε

where:
  n = noise step (0 to 1000)
  ᾱₙ = cumulative noise schedule
  ε = random Gaussian noise
```

At n=0: signal is clean
At n=1000: signal is pure noise

### Backward process (generation)

Iteratively denoise:

```
for n = 1000, 999, ..., 1, 0:
    predicted_noise = model(x_n, n)
    x_{n-1} = denoise_step(x_n, predicted_noise, n)
```

## Key Insight: Augmentation via Partial Denoising

Instead of starting from pure noise (n=1000), start from a **real signal with added noise**:

```python
def augment(x_real, start_step=500):
    # Add noise to intermediate level
    noise = torch.randn_like(x_real)
    x_noisy = sqrt(alpha_bar[start_step]) * x_real + sqrt(1 - alpha_bar[start_step]) * noise

    # Denoise from start_step down to 0
    x = x_noisy
    for n in range(start_step, -1, -1):
        predicted_noise = model(x, n)
        x = denoise_step(x, predicted_noise, n)

    return x  # Augmented signal (similar to x_real but different)
```

### Controlling similarity

| Start step | Noise level | Result |
|------------|-------------|--------|
| 1000 | Maximum | Completely new signal |
| 700 | High | Loosely similar |
| 500 | Medium | Same structure, different details |
| 300 | Low | Very close to original |
| 0 | None | Identical to input |

**This is the augmentation we want**: generate signals that are similar but not identical, with controllable similarity.

## Architecture

### Overview

```
Input: noisy frame x_n (V, 3) + noise step n
Output: predicted noise ε (V, 3)

┌─────────────────────────────────────────┐
│                                         │
│   x_n (V, 3)         n (scalar)         │
│        │                 │              │
│        ▼                 ▼              │
│   [GNN Encoder]    [Timestep MLP]       │
│        │                 │              │
│        ▼                 ▼              │
│     (V, D)            (D,) ────┐        │
│        │                       │        │
│        └───────┬───────────────┘        │
│                ▼                        │
│         [Add/Concat]                    │
│                │                        │
│                ▼                        │
│        [GNN Decoder]                    │
│                │                        │
│                ▼                        │
│         ε_pred (V, 3)                   │
│                                         │
└─────────────────────────────────────────┘
```

### Graph Construction

```python
def mesh_to_graph(cells, pos):
    """
    Convert triangle mesh to graph with edge features.

    Args:
        cells: (F, 3) triangle vertex indices
        pos: (V, 3) vertex positions (rest state)

    Returns:
        edge_index: (2, E) edges
        edge_attr: (E, 4) edge features [dx, dy, dz, length]
    """
    edges = set()
    for tri in cells:
        for i in range(3):
            for j in range(3):
                if i != j:
                    edges.add((tri[i], tri[j]))

    edge_index = torch.tensor(list(edges)).T  # (2, E)

    # Edge features: direction vector + length
    src, dst = edge_index
    edge_vec = pos[dst] - pos[src]  # (E, 3)
    edge_len = edge_vec.norm(dim=-1, keepdim=True)  # (E, 1)
    edge_attr = torch.cat([edge_vec, edge_len], dim=-1)  # (E, 4)

    return edge_index, edge_attr
```

### MeshConv Layer

```python
class MeshConv(MessagePassing):
    """Graph convolution with edge features for direction awareness."""

    def __init__(self, in_dim, out_dim, edge_dim=4):
        super().__init__(aggr='mean')
        self.mlp = nn.Sequential(
            nn.Linear(in_dim * 2 + edge_dim, out_dim),
            nn.LayerNorm(out_dim),
            nn.SiLU(),
            nn.Linear(out_dim, out_dim),
        )

    def forward(self, x, edge_index, edge_attr):
        return self.propagate(edge_index, x=x, edge_attr=edge_attr)

    def message(self, x_i, x_j, edge_attr):
        return self.mlp(torch.cat([x_i, x_j, edge_attr], dim=-1))
```

### Timestep Embedding

```python
class TimestepEmbedding(nn.Module):
    """Sinusoidal embedding for noise step."""

    def __init__(self, dim):
        super().__init__()
        self.dim = dim
        self.mlp = nn.Sequential(
            nn.Linear(dim, dim * 4),
            nn.SiLU(),
            nn.Linear(dim * 4, dim),
        )

    def forward(self, n):
        # n: (B,) noise step indices
        half = self.dim // 2
        freqs = torch.exp(-math.log(10000) * torch.arange(half, device=n.device) / half)
        args = n[:, None] * freqs[None, :]
        emb = torch.cat([torch.sin(args), torch.cos(args)], dim=-1)
        return self.mlp(emb)
```

### Full Model

```python
class GraphSignalDiffusion(nn.Module):
    """Diffusion model for single-frame graph signals."""

    def __init__(self, cells, mesh_pos, hidden_dim=128, num_layers=4):
        super().__init__()

        # Precompute graph structure
        edge_index, edge_attr = mesh_to_graph(cells, mesh_pos)
        self.register_buffer('edge_index', edge_index)
        self.register_buffer('edge_attr', edge_attr)

        self.num_vertices = mesh_pos.shape[0]

        # Encoder: (V, 3) -> (V, D)
        self.input_proj = nn.Linear(3, hidden_dim)
        self.encoder_layers = nn.ModuleList([
            MeshConv(hidden_dim, hidden_dim) for _ in range(num_layers)
        ])

        # Timestep embedding
        self.time_embed = TimestepEmbedding(hidden_dim)

        # Decoder: (V, D) -> (V, 3)
        self.decoder_layers = nn.ModuleList([
            MeshConv(hidden_dim, hidden_dim) for _ in range(num_layers)
        ])
        self.output_proj = nn.Linear(hidden_dim, 3)

    def forward(self, x, n):
        """
        x: (B, V, 3) noisy graph signal
        n: (B,) noise step

        returns: (B, V, 3) predicted noise
        """
        B, V, C = x.shape

        # Timestep embedding
        t_emb = self.time_embed(n)  # (B, D)

        # Process each sample in batch
        outputs = []
        for b in range(B):
            # Encode
            h = self.input_proj(x[b])  # (V, D)
            for layer in self.encoder_layers:
                h = h + layer(h, self.edge_index, self.edge_attr)

            # Add timestep embedding to all vertices
            h = h + t_emb[b].unsqueeze(0)  # (V, D)

            # Decode
            for layer in self.decoder_layers:
                h = h + layer(h, self.edge_index, self.edge_attr)

            out = self.output_proj(h)  # (V, 3)
            outputs.append(out)

        return torch.stack(outputs)  # (B, V, 3)
```

## Training

```python
# Training loop
for epoch in range(num_epochs):
    for batch in dataloader:  # batch: (B, V, 3) single frames
        # Sample random noise steps
        n = torch.randint(0, 1000, (B,))

        # Add noise
        noise = torch.randn_like(batch)
        x_n = sqrt(alpha_bar[n]) * batch + sqrt(1 - alpha_bar[n]) * noise

        # Predict noise
        predicted = model(x_n, n)

        # Loss
        loss = F.mse_loss(predicted, noise)
        loss.backward()
        optimizer.step()
```

## Generation

### From pure noise (new signal)

```python
x = torch.randn(1, V, 3)  # Pure noise

for n in reversed(range(1000)):
    predicted_noise = model(x, torch.tensor([n]))
    x = denoise_step(x, predicted_noise, n)

# x is now a realistic graph signal
```

### From seed signal (augmentation)

```python
def augment(model, x_seed, start_step=500):
    """Generate variation of x_seed."""

    # Add noise to intermediate level
    noise = torch.randn_like(x_seed)
    alpha = alpha_bar[start_step]
    x = sqrt(alpha) * x_seed + sqrt(1 - alpha) * noise

    # Denoise from start_step to 0
    for n in reversed(range(start_step + 1)):
        predicted_noise = model(x, torch.tensor([n]))
        x = denoise_step(x, predicted_noise, n)

    return x  # Similar to x_seed but different
```

## Usage

```bash
# Setup and train
python setup_flag_data.py      # Download data
python train_flag_diffusion.py  # Train model

# Or use batch script
setup_gpu.bat                   # Windows with GPU
./setup_gpu.sh                  # Linux with GPU
```

## Files

| File | Description |
|------|-------------|
| `setup_flag_data.py` | Download and convert flag data |
| `train_flag_diffusion.py` | Train diffusion model |
| `load_flag_data.py` | TFRecord → numpy conversion |

## References

- [DDPM](https://arxiv.org/abs/2006.11239) - Denoising Diffusion Probabilistic Models
- [GAD](https://arxiv.org/abs/2510.05036) - Graph-Aware Diffusion for Signal Generation
- [MeshGraphNets](https://arxiv.org/abs/2010.03409) - Learning Mesh-Based Simulation
