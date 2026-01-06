# Graph Signal Augmentation via Windowed GFT

## Goal

Take existing graph signals and generate new signals that are similar but not identical, for the purpose of data augmentation in GNN training.

## Approach: Spectral Envelope Filtering with Vertex-Frequency Localization

Based on Shuman's **Generalized Windowed Graph Fourier Transform** (vertex-frequency analysis).

### Method

1. **Compute windowed GFT** of original signal:
   - For each vertex n, apply a localized kernel g_n centered at that vertex
   - Compute GFT of the windowed signal
   - Result: S(n, λ) - a 2D representation (vertex × frequency)

2. **Generate white noise** in the (vertex × frequency) domain

3. **Filter noise** with the magnitude envelope |S(n, λ)|

4. **Inverse windowed GFT** to get augmented vertex-domain signal

### What This Preserves vs. Loses

| Preserved | Lost |
|-----------|------|
| Which frequencies appear at which vertices | Phase relationships within each window |
| Overall spatial distribution of features | Fine coherent structure |
| Local smoothness characteristics | Exact feature shapes |

## Hybrid Approach (Controlled Similarity)

Instead of pure noise filtering, blend the original signal with shaped noise:

```
S_aug(n, λ) = α · S_original(n, λ) + (1 - α) · |S_original(n, λ)| · noise(n, λ)
```

Where:
- **α = 1**: Output equals original signal (no augmentation)
- **α = 0**: Full noise filtering (maximum variation)
- **0 < α < 1**: Interpolation between original and noise-shaped variant

This provides a tunable parameter to control how "far" augmented signals deviate from the original.

## Alternative: Direct Spectral Perturbation

A simpler alternative that doesn't require the windowed GFT:

```
x̂ = GFT(x)                      # spectral coefficients
x̂_aug = x̂ + ε · noise           # small perturbation
x_aug = IGFT(x̂_aug)             # back to vertex domain
```

- Preserves both magnitude and phase structure
- ε controls the degree of perturbation
- Results in signals that are "nearby" the original in spectral space

## References

- Shuman, D. I., et al. "Vertex-Frequency Analysis on Graphs." *Applied and Computational Harmonic Analysis* (2016)
- Shuman, D. I., et al. "The Emerging Field of Signal Processing on Graphs." *IEEE Signal Processing Magazine* (2013)
