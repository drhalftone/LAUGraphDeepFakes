# Image-Based Diffusion: Ground Truth Baseline

## Overview

This document describes the **ground truth baseline** for mesh signal generation: treating the mesh as a regular grid image and using standard image diffusion with a U-Net architecture.

## Why Image Diffusion Is Ground Truth

Image-based diffusion is a **solved problem**:

- **Stable Diffusion**, **DALL-E 3**, **Midjourney** - production systems generating millions of images daily
- Extensive literature validating the approach (DDPM, Improved DDPM, LDM, etc.)
- Off-the-shelf implementations readily available (diffusers library, etc.)
- The research community **accepts image diffusion as reliable and proven**

By showing our GNN approach matches image diffusion on grid-structured data, we inherit this credibility. If GNN performs comparably to a method the audience already trusts, the GNN approach is validated.

## Key Insight

The flag mesh is defined on a **regular square grid** of nodes. Each vertex has (x, y, z) coordinates normalized to [-1, 1]. This can be directly interpreted as an RGB image:

```
Mesh vertices: (H × W, 3)  →  Image: (H, W, 3)

Channel mapping:
  R = x coordinate  ∈ [-1, 1]
  G = y coordinate  ∈ [-1, 1]
  B = z coordinate  ∈ [-1, 1]
```

This allows us to use **off-the-shelf image diffusion architectures** which are:
- Well-tested and understood
- Highly optimized (Conv2D on GPU)
- Known to produce high-quality results

## Why This Is Ground Truth

| Aspect | Image/U-Net | GNN |
|--------|-------------|-----|
| Data structure | Native (regular grid) | Requires graph construction |
| Spatial locality | Built into convolutions | Must learn via message passing |
| Positional awareness | Implicit in kernel position | Requires edge features |
| Computational efficiency | Highly optimized | Slower message passing |
| Maturity | Stable Diffusion, DALL-E, etc. | Less explored |

For regular grid data, **U-Net is the natural architecture**. The GNN approach should match its performance to be considered successful.

## Architecture: Hugging Face Diffusers (Off-the-Shelf)

We use the **`diffusers` library** from Hugging Face - the same codebase behind Stable Diffusion. This ensures we're using a battle-tested, published architecture.

### Why Diffusers?

- **Proven**: Powers Stable Diffusion, used by millions
- **Cited**: Well-documented in academic literature
- **Complete**: Includes attention, ResBlocks, proper conditioning
- **Reproducible**: Reviewers can verify our baseline

### Installation

```bash
pip install diffusers accelerate
```

### Model Configuration

```python
from diffusers import UNet2DModel, DDPMScheduler

# Standard DDPM U-Net architecture
# Same architecture class used in Ho et al. "Denoising Diffusion Probabilistic Models"
model = UNet2DModel(
    sample_size=64,              # Assumes 64x64 grid (adjust to actual H, W)
    in_channels=3,               # XYZ coordinates as RGB
    out_channels=3,              # Predict noise in XYZ
    layers_per_block=2,          # ResNet blocks per level
    block_out_channels=(128, 256, 512, 512),  # Channel progression
    down_block_types=(
        "DownBlock2D",           # Regular downsampling
        "DownBlock2D",
        "AttnDownBlock2D",       # With self-attention
        "AttnDownBlock2D",
    ),
    up_block_types=(
        "AttnUpBlock2D",         # With self-attention
        "AttnUpBlock2D",
        "UpBlock2D",
        "UpBlock2D",
    ),
)

# Standard DDPM scheduler (1000 steps, cosine beta schedule available)
scheduler = DDPMScheduler(
    num_train_timesteps=1000,
    beta_schedule="squaredcos_cap_v2",  # Cosine schedule (Improved DDPM)
)
```

### What This Architecture Includes

| Component | Description |
|-----------|-------------|
| **ResNet blocks** | Residual connections with GroupNorm + SiLU |
| **Self-attention** | At lower resolutions (32x32, 16x16) |
| **Timestep embedding** | Sinusoidal encoding, injected into every block |
| **Skip connections** | U-Net style encoder→decoder connections |
| **GroupNorm** | Normalization (better than BatchNorm for small batches) |

### Training Loop (Using Diffusers)

```python
from diffusers import DDPMScheduler
from diffusers.optimization import get_cosine_schedule_with_warmup
import torch.nn.functional as F

def train_epoch(model, dataloader, optimizer, scheduler, device):
    model.train()
    total_loss = 0

    for batch in dataloader:
        # batch: (B, 3, H, W) mesh images (channels first for diffusers)
        batch = batch.to(device)
        B = batch.shape[0]

        # Sample random timesteps
        timesteps = torch.randint(0, scheduler.config.num_train_timesteps, (B,), device=device)

        # Add noise using scheduler
        noise = torch.randn_like(batch)
        noisy = scheduler.add_noise(batch, noise, timesteps)

        # Predict noise
        noise_pred = model(noisy, timesteps).sample

        # MSE loss
        loss = F.mse_loss(noise_pred, noise)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    return total_loss / len(dataloader)
```

### Generation (Using Diffusers Pipeline)

```python
@torch.no_grad()
def generate(model, scheduler, H, W, device, num_inference_steps=1000):
    """Generate a new mesh from noise."""

    # Start from pure noise
    sample = torch.randn(1, 3, H, W, device=device)

    # Set timesteps
    scheduler.set_timesteps(num_inference_steps)

    # Iterative denoising
    for t in scheduler.timesteps:
        # Predict noise
        noise_pred = model(sample, t).sample

        # Denoise one step
        sample = scheduler.step(noise_pred, t, sample).prev_sample

    return sample  # (1, 3, H, W) generated mesh-image
```

## Data Pipeline

### Conversion Functions

```python
def mesh_to_image(vertices, H, W):
    """
    Convert mesh vertices to image format.

    Args:
        vertices: (H*W, 3) vertex positions, assumed in row-major order
        H, W: grid dimensions

    Returns:
        image: (H, W, 3) mesh as RGB image
    """
    return vertices.reshape(H, W, 3)


def image_to_mesh(image):
    """
    Convert image back to mesh vertices.

    Args:
        image: (H, W, 3) mesh as RGB image

    Returns:
        vertices: (H*W, 3) vertex positions
    """
    H, W, C = image.shape
    return image.reshape(H * W, C)
```

### Dataset

```python
class MeshImageDataset(Dataset):
    """Dataset treating mesh frames as images."""

    def __init__(self, frames, H, W):
        """
        Args:
            frames: (N, V, 3) mesh frames where V = H * W
            H, W: grid dimensions
        """
        self.H = H
        self.W = W

        # Reshape to images and normalize
        images = frames.reshape(-1, H, W, 3)

        # Normalize to [-1, 1]
        self.min_val = images.min()
        self.max_val = images.max()
        self.images = 2 * (images - self.min_val) / (self.max_val - self.min_val) - 1

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        return torch.tensor(self.images[idx], dtype=torch.float32)

    def denormalize(self, x):
        return (x + 1) / 2 * (self.max_val - self.min_val) + self.min_val
```

## Training

Standard DDPM training, identical to GNN version but with image I/O:

```python
def train_epoch(model, dataloader, optimizer, schedule, device):
    model.train()
    total_loss = 0

    for batch in dataloader:
        # batch: (B, H, W, 3) mesh images
        batch = batch.to(device)
        B = batch.shape[0]

        # Random timesteps
        n = torch.randint(0, schedule.num_steps, (B,), device=device)

        # Add noise
        noise = torch.randn_like(batch)
        x_noisy = schedule.q_sample(batch, n, noise)

        # Predict noise
        predicted = model(x_noisy, n)

        # Loss
        loss = F.mse_loss(predicted, noise)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    return total_loss / len(dataloader)
```

## Generation

```python
@torch.no_grad()
def generate(model, schedule, H, W, device):
    """Generate a new mesh from noise."""

    # Start from pure noise
    x = torch.randn(1, H, W, 3, device=device)

    # Iterative denoising
    for n in reversed(range(schedule.num_steps)):
        x = schedule.p_sample(model, x, n)

    return x  # (1, H, W, 3) generated mesh-image
```

## Boundary Conditions

The flag has fixed vertices along one edge (the pole). Two approaches:

### Approach 1: Mask and Replace

After each denoising step, replace fixed vertices with their known values:

```python
def p_sample_with_boundary(model, x_n, n, fixed_mask, fixed_values):
    """Denoise one step, enforcing boundary conditions."""
    x_new = standard_p_sample(model, x_n, n)

    # Replace fixed pixels
    x_new = torch.where(fixed_mask, fixed_values, x_new)

    return x_new
```

### Approach 2: Inpainting Formulation

Treat generation as inpainting: fixed boundary is "known", interior is "generated".

This naturally extends to conditional generation.

## Comparison Protocol: U-Net vs GNN

To validate the GNN approach, we compare against this U-Net baseline:

### Metrics

1. **Reconstruction MSE**: How well does the model denoise?
2. **Generation quality**: Visual inspection + FID-like metric on generated samples
3. **Edge length preservation**: Do generated meshes have valid edge lengths?
4. **Training time**: Wall-clock time per epoch
5. **Inference time**: Time to generate one sample

### Expected Results

| Metric | U-Net (baseline) | GNN (proposed) |
|--------|------------------|----------------|
| Reconstruction MSE | Lower (optimal for grids) | Should match |
| Generation quality | High | Should match |
| Edge lengths | May violate | Can enforce constraints |
| Training time | Faster | Slower |
| Inference time | Faster | Slower |

### Success Criterion

The GNN approach is successful if:
1. **Generation quality matches U-Net baseline** - This is the key validation. If GNN matches a method the community already trusts, reviewers will accept the GNN results.
2. GNN generalizes to irregular meshes (U-Net cannot)
3. GNN can enforce physics constraints (edge lengths, boundaries)

The first criterion is most important for paper acceptance. Points 2 and 3 are the motivation for *why* we want a GNN approach, but point 1 establishes credibility.

## What the GNN Needs to Match U-Net

For the GNN to match the `diffusers` U-Net baseline, it needs comparable architectural components:

| U-Net Component | GNN Equivalent |
|----------------|----------------|
| Conv2D + ResBlocks | Message passing + residual connections |
| Self-attention | Graph Attention (GAT) layers |
| Encoder-decoder | GNN encoder-decoder with pooling/unpooling |
| Skip connections | Skip connections between encoder/decoder GNN layers |
| Timestep embedding | Same (sinusoidal + MLP) |
| GroupNorm | GraphNorm or LayerNorm per node |

### GNN Architecture Requirements

```python
# The GNN should have:
# 1. Graph attention for long-range dependencies
# 2. Residual connections throughout
# 3. Proper timestep conditioning at each layer
# 4. Comparable parameter count to U-Net

class GraphAttentionDiffusion(nn.Module):
    """GNN with attention to match U-Net performance."""

    def __init__(self, ...):
        # Multi-head graph attention layers
        self.attention_layers = nn.ModuleList([
            GraphAttentionLayer(hidden_dim, num_heads=8)
            for _ in range(num_layers)
        ])

        # Timestep conditioning at every layer (like U-Net)
        self.time_projections = nn.ModuleList([
            nn.Linear(time_dim, hidden_dim)
            for _ in range(num_layers)
        ])

        # Residual connections
        # ...
```

This is achievable - graph transformers and attention-based GNNs have shown strong results on various tasks.

## Advantages of Each Approach

### U-Net (Ground Truth)

- Optimal for regular grid data
- Fast and well-optimized
- Extensive literature and pretrained models
- Simple implementation

### GNN (Proposed)

- Generalizes to irregular meshes
- Can encode mesh topology explicitly
- Natural for physics constraints (edge features = rest lengths)
- Differential coordinates approach (see `differential_coordinates_plan.md`)

## Conclusion

The U-Net image diffusion serves as **ground truth** for regular grid meshes. The GNN approach aims to:

1. **Match** U-Net performance on regular grids (validation)
2. **Extend** to irregular meshes where U-Net cannot apply
3. **Incorporate** physics constraints naturally via edge features

If GNN matches U-Net on grids, we have confidence it will work on irregular meshes where U-Net is not applicable.
