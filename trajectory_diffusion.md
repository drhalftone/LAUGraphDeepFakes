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

## Hardware requirements

- **GPU**: Recommended (RTX 3080+ or similar)
- **VRAM**: 8GB+ for batch training
- **Disk**: ~15GB for full dataset
