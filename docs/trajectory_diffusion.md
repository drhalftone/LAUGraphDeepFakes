# Trajectory Diffusion for Flag Generation

## Goal

Generate realistic flag-in-wind trajectories using a diffusion model. Given random noise, denoise it into a physically plausible flag animation.

## Background

### What we tried

1. **Spectral VAE on FEA data** - Worked but limited variation (all signals too similar)
2. **Noise filtering on flag data** - Failed to capture:
   - Boundary conditions (fixed pole edge)
   - Spatial coherence (nearby vertices move together)
   - Temporal smoothness (motion was too jittery)

### Why diffusion

Diffusion models are the current state-of-the-art for generative AI (Stable Diffusion, DALL-E 3, Sora). They:
- Learn implicit constraints from data
- Generate diverse, high-quality outputs
- Handle high-dimensional data well

## Data

Downloaded from DeepMind's MeshGraphNets:

```
flag_data/
├── meta.json           # Dataset metadata
├── train.tfrecord      # 9.5GB training data
├── test.tfrecord       # 970MB test data
└── flag_test.npz       # Converted numpy (50 trajectories)
```

### Data structure

| Field | Shape | Description |
|-------|-------|-------------|
| `world_pos` | (N, 401, 1579, 3) | N trajectories, 401 timesteps, 1579 vertices, xyz |
| `cells` | (3028, 3) | Triangle mesh connectivity |
| `mesh_pos` | (1579, 2) | Rest state vertex positions |

- **dt = 0.02s** → 401 timesteps = 8 seconds of simulation
- Vertex 0 is fixed at the pole, other vertices wave freely

## Proposed Architecture

### Trajectory Diffusion Model

```
Training:
1. Take real trajectory x₀: (T, V, 3)
2. Add noise: xₜ = √ᾱₜ·x₀ + √(1-ᾱₜ)·ε
3. Train denoiser to predict ε given xₜ and t

Generation:
1. Start with pure noise: x_T ~ N(0, I)
2. For t = T, T-1, ..., 1:
      xₜ₋₁ = denoise(xₜ, t)
3. Output clean trajectory x₀
```

### Denoiser architecture options

**Option A: 3D U-Net on (T, V, 3)**
- Treat trajectory as a "volume"
- Temporal convolutions + spatial graph convolutions

**Option B: Transformer**
- Flatten to sequence of frames
- Self-attention across time
- Graph attention for spatial

**Option C: Factorized approach**
- Encode each frame with GNN
- Temporal transformer on frame embeddings
- Decode back to vertices

### Recommended: Option C (Factorized)

```
trajectory (T, V, 3)
       ↓
[GNN Encoder] per frame → (T, D) frame embeddings
       ↓
[Temporal Transformer] + timestep embedding
       ↓
[GNN Decoder] → denoised trajectory (T, V, 3)
```

This is efficient because:
- GNN respects mesh structure
- Transformer handles long-range temporal dependencies
- Factorization reduces memory (don't need full T×V attention)

## Diffusion schedule

Standard cosine schedule:
```python
betas = cosine_beta_schedule(num_timesteps=1000)
alphas = 1 - betas
alpha_bar = torch.cumprod(alphas, dim=0)
```

## Training procedure

```python
for epoch in range(epochs):
    for trajectory in dataloader:
        # Sample random timestep
        t = torch.randint(0, T, (batch_size,))

        # Add noise
        noise = torch.randn_like(trajectory)
        x_t = sqrt(alpha_bar[t]) * trajectory + sqrt(1 - alpha_bar[t]) * noise

        # Predict noise
        predicted_noise = model(x_t, t)

        # Loss
        loss = F.mse_loss(predicted_noise, noise)
        loss.backward()
        optimizer.step()
```

## Generation procedure

```python
# Start from noise
x = torch.randn(1, T, V, 3)

# Denoise iteratively
for t in reversed(range(num_timesteps)):
    predicted_noise = model(x, t)
    x = denoise_step(x, predicted_noise, t)  # DDPM or DDIM

# x is now a realistic flag trajectory
```

## Implementation plan

1. **Data loading** (done): `load_flag_data.py`, `flag_test.npz`

2. **Model definition**:
   - `FlagDiffusion` class with GNN encoder/decoder + temporal transformer
   - Timestep embedding (sinusoidal)
   - Noise schedule

3. **Training loop**:
   - Load more training data from `train.tfrecord`
   - Train on GPU
   - Save checkpoints

4. **Generation & visualization**:
   - Sample from trained model
   - Compare to real trajectories
   - Animate results

## Files in this repo

| File | Description |
|------|-------------|
| `load_flag_data.py` | Load TFRecord → numpy |
| `generate_flag_noise.py` | Noise filtering approach (baseline) |
| `train_spectral_vae.py` | Spectral VAE (for FEA data) |
| `flag_data/flag_test.npz` | 50 test trajectories (numpy) |

## Next steps (on GPU machine)

```bash
cd LAUGraphDeepFakes
git pull

# 1. Download flag data (not in repo due to size)
mkdir -p flag_data
curl -O https://storage.googleapis.com/dm-meshgraphnets/flag_simple/meta.json -o flag_data/meta.json
curl -O https://storage.googleapis.com/dm-meshgraphnets/flag_simple/train.tfrecord -o flag_data/train.tfrecord
curl -O https://storage.googleapis.com/dm-meshgraphnets/flag_simple/test.tfrecord -o flag_data/test.tfrecord

# 2. Convert to numpy (generates flag_test.npz)
python load_flag_data.py

# 3. Implement and train diffusion model
python train_flag_diffusion.py  # to be written
```

## References

- [Denoising Diffusion Probabilistic Models](https://arxiv.org/abs/2006.11239) (Ho et al., 2020)
- [MeshGraphNets](https://arxiv.org/abs/2010.03409) (Pfaff et al., 2021)
- [Diffusion Models Beat GANs](https://arxiv.org/abs/2105.05233) (Dhariwal & Nichol, 2021)

## Detailed GNN Architecture

### Overview

```
Input: noisy trajectory x_t (T, V, 3) + timestep t
Output: predicted noise ε (T, V, 3)

┌─────────────────────────────────────────────────────────┐
│                                                         │
│   x_t (T, V, 3)          t (scalar)                     │
│        │                     │                          │
│        ▼                     ▼                          │
│   [Per-frame GNN]      [Timestep MLP]                   │
│        │                     │                          │
│        ▼                     ▼                          │
│   (T, V, D)              (D,) ──────┐                   │
│        │                            │                   │
│        ▼                            │                   │
│   [Temporal Transformer] ◄──────────┘                   │
│        │                                                │
│        ▼                                                │
│   (T, V, D)                                             │
│        │                                                │
│        ▼                                                │
│   [Per-frame GNN Decoder]                               │
│        │                                                │
│        ▼                                                │
│   ε_pred (T, V, 3)                                      │
│                                                         │
└─────────────────────────────────────────────────────────┘
```

### Graph Construction from Mesh

```python
def mesh_to_graph(cells):
    """
    Convert triangle mesh to graph edges.

    cells: (F, 3) triangle vertex indices
    returns: edge_index (2, E) for PyTorch Geometric
    """
    edges = set()
    for tri in cells:
        # Add edges for each triangle edge (bidirectional)
        edges.add((tri[0], tri[1]))
        edges.add((tri[1], tri[0]))
        edges.add((tri[1], tri[2]))
        edges.add((tri[2], tri[1]))
        edges.add((tri[2], tri[0]))
        edges.add((tri[0], tri[2]))

    edge_index = torch.tensor(list(edges)).T  # (2, E)
    return edge_index
```

### MeshConv Layer (Message Passing)

```python
import torch
import torch.nn as nn
from torch_geometric.nn import MessagePassing

class MeshConv(MessagePassing):
    """
    Graph convolution for mesh data.
    Aggregates features from neighboring vertices.
    """
    def __init__(self, in_dim, out_dim):
        super().__init__(aggr='mean')  # Mean aggregation

        self.mlp = nn.Sequential(
            nn.Linear(in_dim * 2, out_dim),
            nn.LayerNorm(out_dim),
            nn.SiLU(),
            nn.Linear(out_dim, out_dim),
        )

    def forward(self, x, edge_index):
        # x: (V, D) node features
        # edge_index: (2, E) edges
        return self.propagate(edge_index, x=x)

    def message(self, x_i, x_j):
        # x_i: target node features
        # x_j: source (neighbor) node features
        return self.mlp(torch.cat([x_i, x_j], dim=-1))
```

### GNN Encoder

```python
class MeshEncoder(nn.Module):
    """
    Encode per-vertex 3D positions to feature vectors.
    """
    def __init__(self, in_dim=3, hidden_dim=64, out_dim=128, num_layers=4):
        super().__init__()

        self.input_proj = nn.Linear(in_dim, hidden_dim)

        self.layers = nn.ModuleList([
            MeshConv(hidden_dim, hidden_dim)
            for _ in range(num_layers)
        ])

        self.output_proj = nn.Linear(hidden_dim, out_dim)

    def forward(self, x, edge_index):
        # x: (V, 3) vertex positions
        # edge_index: (2, E)

        h = self.input_proj(x)  # (V, hidden_dim)

        for layer in self.layers:
            h = h + layer(h, edge_index)  # Residual connection

        return self.output_proj(h)  # (V, out_dim)
```

### GNN Decoder

```python
class MeshDecoder(nn.Module):
    """
    Decode feature vectors back to 3D positions (noise prediction).
    """
    def __init__(self, in_dim=128, hidden_dim=64, out_dim=3, num_layers=4):
        super().__init__()

        self.input_proj = nn.Linear(in_dim, hidden_dim)

        self.layers = nn.ModuleList([
            MeshConv(hidden_dim, hidden_dim)
            for _ in range(num_layers)
        ])

        self.output_proj = nn.Linear(hidden_dim, out_dim)

    def forward(self, h, edge_index):
        # h: (V, in_dim) features

        h = self.input_proj(h)

        for layer in self.layers:
            h = h + layer(h, edge_index)

        return self.output_proj(h)  # (V, 3)
```

### Temporal Transformer

```python
class TemporalTransformer(nn.Module):
    """
    Process temporal sequence of frame embeddings.
    """
    def __init__(self, dim=128, num_heads=8, num_layers=4, max_len=512):
        super().__init__()

        self.pos_embed = nn.Parameter(torch.randn(1, max_len, dim) * 0.02)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=dim,
            nhead=num_heads,
            dim_feedforward=dim * 4,
            dropout=0.1,
            activation='gelu',
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers)

    def forward(self, x, t_embed):
        # x: (B, T, V, D) or (B, T, D) if already pooled per frame
        # t_embed: (B, D) timestep embedding

        B, T, D = x.shape

        # Add positional embedding
        x = x + self.pos_embed[:, :T, :]

        # Add timestep embedding to all frames
        x = x + t_embed.unsqueeze(1)

        # Transformer
        return self.transformer(x)  # (B, T, D)
```

### Timestep Embedding

```python
class TimestepEmbedding(nn.Module):
    """
    Sinusoidal timestep embedding (like in original DDPM).
    """
    def __init__(self, dim):
        super().__init__()
        self.dim = dim
        self.mlp = nn.Sequential(
            nn.Linear(dim, dim * 4),
            nn.SiLU(),
            nn.Linear(dim * 4, dim),
        )

    def forward(self, t):
        # t: (B,) timestep indices
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=t.device) * -emb)
        emb = t[:, None] * emb[None, :]
        emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=-1)
        return self.mlp(emb)  # (B, dim)
```

### Full Diffusion Model

```python
class TrajectoryDiffusion(nn.Module):
    """
    Full trajectory diffusion model with GNN + Transformer.
    """
    def __init__(self,
                 num_vertices=1579,
                 hidden_dim=128,
                 num_gnn_layers=4,
                 num_transformer_layers=4,
                 num_heads=8):
        super().__init__()

        self.encoder = MeshEncoder(
            in_dim=3,
            hidden_dim=hidden_dim//2,
            out_dim=hidden_dim,
            num_layers=num_gnn_layers
        )

        self.time_embed = TimestepEmbedding(hidden_dim)

        # Pool per-vertex features to per-frame (for transformer efficiency)
        self.frame_pool = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
        )

        self.temporal = TemporalTransformer(
            dim=hidden_dim,
            num_heads=num_heads,
            num_layers=num_transformer_layers,
        )

        # Unpool back to per-vertex
        self.frame_unpool = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
        )

        self.decoder = MeshDecoder(
            in_dim=hidden_dim,
            hidden_dim=hidden_dim//2,
            out_dim=3,
            num_layers=num_gnn_layers
        )

        self.num_vertices = num_vertices

    def forward(self, x, t, edge_index):
        """
        x: (B, T, V, 3) noisy trajectory
        t: (B,) diffusion timestep
        edge_index: (2, E) mesh edges

        returns: (B, T, V, 3) predicted noise
        """
        B, T, V, C = x.shape

        # Timestep embedding
        t_emb = self.time_embed(t)  # (B, D)

        # Encode each frame with GNN
        # Reshape: (B, T, V, 3) -> (B*T, V, 3)
        x_flat = x.reshape(B * T, V, C)

        # Process each frame
        h_list = []
        for i in range(B * T):
            h_frame = self.encoder(x_flat[i], edge_index)  # (V, D)
            h_list.append(h_frame)
        h = torch.stack(h_list)  # (B*T, V, D)

        # Pool to per-frame embedding
        h = h.mean(dim=1)  # (B*T, D) - mean over vertices
        h = self.frame_pool(h)
        h = h.reshape(B, T, -1)  # (B, T, D)

        # Temporal transformer
        h = self.temporal(h, t_emb)  # (B, T, D)

        # Unpool to per-vertex
        h = self.frame_unpool(h)  # (B, T, D)
        h = h.unsqueeze(2).expand(-1, -1, V, -1)  # (B, T, V, D)
        h = h.reshape(B * T, V, -1)  # (B*T, V, D)

        # Decode each frame with GNN
        out_list = []
        for i in range(B * T):
            out_frame = self.decoder(h[i], edge_index)  # (V, 3)
            out_list.append(out_frame)
        out = torch.stack(out_list)  # (B*T, V, 3)

        return out.reshape(B, T, V, C)  # (B, T, V, 3)
```

### Memory-Efficient Alternative

The above processes each frame separately. For efficiency, use batched GNN:

```python
from torch_geometric.data import Batch, Data

def batch_frames(x, edge_index):
    """Batch all frames for efficient GNN processing."""
    B, T, V, C = x.shape

    # Create a batch of graphs (one per frame)
    data_list = []
    for b in range(B):
        for t in range(T):
            data = Data(x=x[b, t], edge_index=edge_index)
            data_list.append(data)

    return Batch.from_data_list(data_list)
```

### Model Size Estimates

| Variant | Parameters | VRAM (batch=4) |
|---------|------------|----------------|
| Small (D=64, L=2) | ~500K | ~2GB |
| Medium (D=128, L=4) | ~2M | ~4GB |
| Large (D=256, L=6) | ~8M | ~8GB |

Start with **Small** for testing, scale up as needed.

## Hardware requirements

- **GPU**: Recommended (RTX 3080+ or similar)
- **VRAM**: 8GB+ for batch training
- **Disk**: ~15GB for full dataset
