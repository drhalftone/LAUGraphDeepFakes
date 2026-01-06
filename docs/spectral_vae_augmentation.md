# Spectral VAE for Graph Signal Augmentation

## Quick Start

```bash
cd LAUGraphDeepFakes

# Windows
setup.bat
gdf_env\Scripts\activate
python train_spectral_vae.py

# Linux/Mac (GPU)
chmod +x setup_gpu.sh && ./setup_gpu.sh
source gdf_gpu/bin/activate
python train_spectral_vae.py

# Linux/Mac (CPU)
chmod +x setup.sh && ./setup.sh
source gdf_env/bin/activate
python train_spectral_vae.py
```

Output saved to `spectral_vae_output/`.

## Overview

A lightweight variational autoencoder (VAE) that operates in the spectral domain of a fixed mesh. Given an input graph signal, it can generate multiple similar-but-different variations by sampling different noise vectors.

## Architecture

```
                         ENCODE                           DECODE

signal x ──→ [GFT] ──→ x̂ ──→ [MLP] ──→ μ, σ ──→ z ──→ [MLP] ──→ x̂' ──→ [IGFT] ──→ x'
  (N,)                (K,)            (D,)   ↑        (K,)              (N,)
                                             │
                                        ε ~ N(0,1)
                                       (noise injection)
```

### Key Components

| Component | Description |
|-----------|-------------|
| **GFT** | Graph Fourier Transform using Laplacian eigenvectors |
| **Truncation** | Keep only K lowest frequency components (K << N) |
| **Encoder** | MLP mapping K spectral coefficients to latent μ, σ |
| **Reparameterization** | z = μ + σ * ε, where ε ~ N(0,1) |
| **Decoder** | MLP mapping D-dim latent back to K spectral coefficients |
| **IGFT** | Inverse GFT to recover vertex-domain signal |

### Why Spectral Domain?

1. **Dimensionality reduction**: Mesh has N ≈ 6500 vertices, but smooth signals concentrate energy in low frequencies
2. **Natural compression**: Truncate to K ≈ 64-128 coefficients with minimal information loss
3. **Smaller networks**: Encoder/decoder operate on K dims instead of N
4. **Graph-aware**: Spectral basis respects mesh geometry

## Training

### Loss Function

```
L = L_reconstruction + β * L_KL

L_reconstruction = ‖x̂ - x̂'‖²     (MSE in spectral domain)
L_KL = KL(q(z|x) ‖ p(z))         (regularize latent to unit Gaussian)
```

The β parameter controls the tradeoff:
- **β small**: Better reconstruction, less variation between samples
- **β large**: More variation, but may lose fidelity

### Training Data

The model learns from a set of example graph signals. It captures the distribution of these signals and can generate new samples from this distribution.

## Generation (Augmentation)

Given an original signal x:

```python
# Encode to latent distribution
x_spec = gft(x)[:K]
mu, logvar = encoder(x_spec)

# Generate multiple augmentations
for i in range(num_augmentations):
    eps = torch.randn(D)                    # new noise
    z = mu + torch.exp(0.5 * logvar) * eps  # sample latent
    x_aug_spec = decoder(z)                 # decode
    x_aug = igft(x_aug_spec)                # to vertex domain
```

Each different noise sample ε produces a different augmented signal.

## Controlling Variation

### Method 1: Scale the noise

```python
z = mu + scale * torch.exp(0.5 * logvar) * eps
```
- `scale = 0`: Output equals reconstruction of input
- `scale = 1`: Normal VAE sampling
- `scale > 1`: More extreme variations

### Method 2: Interpolate in latent space

```python
z = alpha * z1 + (1 - alpha) * z2
```

Interpolate between latent codes of two different signals.

### Method 3: Adjust β during training

Lower β → latent codes stay closer to encoded input → less variation at generation time.

## Hyperparameters

| Parameter | Typical Value | Description |
|-----------|---------------|-------------|
| K | 64-128 | Number of spectral coefficients |
| D | 16-32 | Latent dimension |
| β | 0.001-0.01 | KL weight (start small) |
| Hidden dim | 32-64 | MLP hidden layer size |
| Learning rate | 1e-3 | Adam optimizer |
| Epochs | 500-1000 | Training iterations |

## Comparison to Non-Learned Approach

| Aspect | Spectral Noise Filtering | Spectral VAE |
|--------|--------------------------|--------------|
| Training required | No | Yes |
| Learns from data | No | Yes |
| Control over similarity | Via blending parameter α | Via noise scale, β |
| Captures data distribution | No (just spectral envelope) | Yes |
| Computational cost | Very low | Low (small MLPs) |

## References

- Kingma & Welling, "Auto-Encoding Variational Bayes" (2014)
- Shuman et al., "The Emerging Field of Signal Processing on Graphs" (2013)
