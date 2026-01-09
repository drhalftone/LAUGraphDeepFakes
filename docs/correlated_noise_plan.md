# Correlated Noise Trajectories for Video Generation

## Problem Statement

Standard diffusion training uses independent random noise for each training sample. When training on video frames:

- Consecutive frames A and B are nearly identical (high correlation in signal space)
- But they get mapped to completely unrelated noise patterns εA and εB
- During generation from pure noise, the model struggles because:
  - Many different noise patterns must map to nearly identical outputs
  - The reverse mapping is ambiguous/poorly conditioned
  - Generation from step 999 fails to produce useful signals

## Current Solution: Temporal Subsampling

Generate training data with large temporal stride (e.g., 100 steps between frames) so each sample is statistically independent. This works but:

- Discards most of the simulation data
- Doesn't support coherent video generation
- Each generated frame is independent (no temporal consistency)

## Proposed Solution: Correlated Noise Trajectories

### Key Insight

If similar frames should map to similar noise patterns, then:
- A smooth path through noise space → smooth path through signal space
- Video generation becomes: generate a noise trajectory, then denoise each frame
- Temporal coherence emerges naturally

### Training Approach

For a video sequence with frames x₀, x₁, x₂, ... at timesteps t₀, t₁, t₂, ...

1. **Generate base noise** for first frame:
   ```
   ε₀ ~ N(0, I)
   ```

2. **Generate correlated noise** for subsequent frames:
   ```
   ε₁ = √(1-σ²) · ε₀ + σ · η₁,  where η₁ ~ N(0, I)
   ε₂ = √(1-σ²) · ε₁ + σ · η₂,  where η₂ ~ N(0, I)
   ...
   ```

   The parameter σ controls correlation:
   - σ = 0: All frames use identical noise (fully correlated)
   - σ = 1: Independent noise (standard diffusion)
   - σ ≈ 0.1-0.3: Nearby frames have similar noise

3. **Train normally** with these correlated noise patterns:
   ```
   x_t = √(ᾱ_t) · x₀ + √(1-ᾱ_t) · ε
   Loss = ||ε - ε_pred||²
   ```

### Generation Approach

To generate a coherent video sequence:

1. **Generate noise trajectory**:
   ```python
   ε_0 = torch.randn(shape)  # Random start
   trajectory = [ε_0]
   for i in range(num_frames - 1):
       η = torch.randn(shape)
       ε_next = sqrt(1 - σ²) * trajectory[-1] + σ * η
       trajectory.append(ε_next)
   ```

2. **Denoise each noise pattern** to get video frames:
   ```python
   frames = [denoise(ε) for ε in trajectory]
   ```

3. **Result**: Temporally coherent video where:
   - Frame-to-frame changes are smooth
   - Overall motion follows a random but continuous path
   - No explicit temporal modeling needed in the network

### Correlation Parameter σ

The noise correlation should match the signal correlation:

| Frame Distance | Signal Correlation | Suggested σ |
|----------------|-------------------|-------------|
| 1 step | ~0.99 | 0.05-0.1 |
| 10 steps | ~0.90 | 0.2-0.3 |
| 100 steps | ~0.50 | 0.5-0.7 |
| 1000 steps | ~0.00 | 1.0 |

Could estimate σ empirically by measuring actual frame-to-frame variance in training data.

### Implementation Plan

#### Phase 1: Baseline (Current)
- [x] Generate independent frames with stride=100
- [ ] Train diffusion model on independent frames
- [ ] Evaluate single-frame generation quality

#### Phase 2: Correlated Noise Training
- [ ] Modify `train_flag_diffusion.py` to use correlated noise
- [ ] Load consecutive frames as batches (not shuffled randomly)
- [ ] Generate noise trajectories per video sequence
- [ ] Train with same architecture, different noise sampling

#### Phase 3: Video Generation
- [ ] Implement noise trajectory generation
- [ ] Generate multi-frame sequences
- [ ] Evaluate temporal coherence
- [ ] Compare to independent frame generation

### Code Changes Required

**1. Dataset modification** (`train_flag_diffusion.py`):
```python
class VideoSequenceDataset(Dataset):
    """Load consecutive frames as sequences."""

    def __init__(self, world_pos, sequence_length=16):
        # world_pos: (N_trajectories, T_frames, V_vertices, 3)
        self.sequences = []
        for traj in world_pos:
            for start in range(0, len(traj) - sequence_length, sequence_length // 2):
                self.sequences.append(traj[start:start + sequence_length])

    def __getitem__(self, idx):
        return self.sequences[idx]  # (seq_len, V, 3)
```

**2. Correlated noise generation**:
```python
def generate_correlated_noise(shape, seq_len, sigma=0.1):
    """Generate temporally correlated noise sequence."""
    noise = [torch.randn(shape)]
    for _ in range(seq_len - 1):
        eta = torch.randn(shape)
        next_noise = math.sqrt(1 - sigma**2) * noise[-1] + sigma * eta
        noise.append(next_noise)
    return torch.stack(noise)  # (seq_len, *shape)
```

**3. Training loop modification**:
```python
for sequences in dataloader:  # (B, seq_len, V, 3)
    B, S, V, C = sequences.shape

    # Generate correlated noise for each sequence
    noise = generate_correlated_noise((V, C), S, sigma=0.1)  # (S, V, C)
    noise = noise.unsqueeze(0).expand(B, -1, -1, -1)  # (B, S, V, C)

    # Random diffusion timestep (same for whole sequence)
    t = torch.randint(0, T, (B,))

    # Add noise to all frames
    x_t = sqrt_alphas[t] * sequences + sqrt_one_minus_alphas[t] * noise

    # Predict noise for all frames
    noise_pred = model(x_t.view(B*S, V, C), t.repeat_interleave(S))

    # Loss
    loss = F.mse_loss(noise_pred, noise.view(B*S, V, C))
```

### Alternative: DDIM Inversion

Instead of training with correlated noise, use DDIM to find the noise that reconstructs each frame:

1. Train standard diffusion model
2. For each training frame, run DDIM inversion to find corresponding noise
3. Noise patterns for consecutive frames will naturally be similar
4. Use these "inverted" noise patterns to generate new videos

Advantage: No training changes required
Disadvantage: Requires additional inversion step, may not perfectly preserve correlations

### References

- [DDPM](https://arxiv.org/abs/2006.11239) - Denoising Diffusion Probabilistic Models
- [DDIM](https://arxiv.org/abs/2010.02502) - Denoising Diffusion Implicit Models (deterministic sampling)
- [Video Diffusion Models](https://arxiv.org/abs/2204.03458) - Temporal attention for video
- [Stable Video Diffusion](https://arxiv.org/abs/2311.15127) - Image-to-video generation

### Questions to Explore

1. What σ value best matches the simulation's temporal dynamics?
2. Should σ vary with diffusion timestep t?
3. Can we learn σ as part of training?
4. How does this compare to explicit temporal modeling (3D convolutions, temporal attention)?
