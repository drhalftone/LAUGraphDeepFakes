#!/usr/bin/env python3
"""
Generate synthetic flag trajectories by filtering noise.

This approach:
1. Analyzes the spectral characteristics of real flag data
2. Generates white noise
3. Filters it to match the spectral envelope
4. Applies spatial correlations between vertices

No neural network required!
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path


def load_flag_data(path='flag_data/flag_test.npz'):
    """Load flag trajectory data."""
    data = np.load(path)
    return {
        'world_pos': data['world_pos'],  # (N, T, V, 3)
        'cells': data['cells'],           # (F, 3)
        'mesh_pos': data['mesh_pos'],     # (V, 2)
        'dt': float(data['dt']),
    }


def compute_spectral_envelope(trajectories):
    """
    Compute the average spectral envelope across all trajectories.

    Args:
        trajectories: (N, T, V, 3) array of flag positions

    Returns:
        magnitude_spectrum: (T//2+1, V, 3) average magnitude spectrum
        mean_position: (V, 3) mean position per vertex
    """
    N, T, V, C = trajectories.shape

    # Remove mean (work with displacements)
    mean_position = trajectories.mean(axis=(0, 1))  # (V, 3)
    centered = trajectories - mean_position

    # Compute FFT for each trajectory
    spectra = np.fft.rfft(centered, axis=1)  # (N, T//2+1, V, 3)

    # Average magnitude spectrum
    magnitude_spectrum = np.abs(spectra).mean(axis=0)  # (T//2+1, V, 3)

    return magnitude_spectrum, mean_position


def compute_spatial_covariance(trajectories):
    """
    Compute spatial covariance between vertices.

    Args:
        trajectories: (N, T, V, 3) array

    Returns:
        cov: (V*3, V*3) covariance matrix
    """
    N, T, V, C = trajectories.shape

    # Flatten to (N*T, V*3)
    flat = trajectories.reshape(N * T, V * C)

    # Center
    flat_centered = flat - flat.mean(axis=0)

    # Covariance
    cov = (flat_centered.T @ flat_centered) / (N * T - 1)

    return cov


def generate_filtered_noise(magnitude_spectrum, mean_position, num_samples=1, seed=None):
    """
    Generate synthetic trajectories by filtering white noise.

    Args:
        magnitude_spectrum: (F, V, 3) target magnitude spectrum
        mean_position: (V, 3) mean position to add back
        num_samples: number of trajectories to generate
        seed: random seed

    Returns:
        synthetic: (num_samples, T, V, 3) synthetic trajectories
    """
    if seed is not None:
        np.random.seed(seed)

    F, V, C = magnitude_spectrum.shape
    T = (F - 1) * 2  # Reconstruct time length from rfft output

    synthetic_list = []

    for _ in range(num_samples):
        # Generate white noise in frequency domain (complex)
        # Random phase, unit magnitude
        noise_phase = np.exp(2j * np.pi * np.random.rand(F, V, C))

        # Apply magnitude spectrum (spectral shaping)
        shaped_spectrum = magnitude_spectrum * noise_phase

        # Inverse FFT
        synthetic = np.fft.irfft(shaped_spectrum, n=T, axis=0)  # (T, V, 3)

        # Add mean position back
        synthetic = synthetic + mean_position

        synthetic_list.append(synthetic)

    return np.array(synthetic_list)  # (num_samples, T, V, 3)


def generate_with_spatial_correlation(magnitude_spectrum, mean_position,
                                       spatial_cov, num_samples=1, seed=None):
    """
    Generate synthetic trajectories with both spectral and spatial correlations.

    This method:
    1. Generates spectrally-shaped noise per vertex
    2. Applies spatial correlations via Cholesky decomposition
    """
    if seed is not None:
        np.random.seed(seed)

    F, V, C = magnitude_spectrum.shape
    T = (F - 1) * 2

    # Cholesky decomposition for spatial correlations
    # Add small regularization for numerical stability
    reg = 1e-6 * np.eye(spatial_cov.shape[0])
    try:
        L = np.linalg.cholesky(spatial_cov + reg)
    except np.linalg.LinAlgError:
        print("Warning: Covariance not positive definite, using eigendecomposition")
        eigvals, eigvecs = np.linalg.eigh(spatial_cov)
        eigvals = np.maximum(eigvals, 1e-6)
        L = eigvecs @ np.diag(np.sqrt(eigvals))

    synthetic_list = []

    for _ in range(num_samples):
        # Generate spectrally-shaped noise (independent per vertex first)
        noise_phase = np.exp(2j * np.pi * np.random.rand(F, V, C))
        shaped_spectrum = magnitude_spectrum * noise_phase
        independent_signal = np.fft.irfft(shaped_spectrum, n=T, axis=0)  # (T, V, 3)

        # Apply spatial correlations
        # Reshape to (T, V*3), apply correlation, reshape back
        flat = independent_signal.reshape(T, V * C)

        # Normalize variance before applying correlation
        std = flat.std(axis=0, keepdims=True) + 1e-8
        flat_normalized = flat / std

        # Apply spatial correlation
        correlated = flat_normalized @ L.T

        # Restore original variance scale
        correlated = correlated * std

        # Reshape back
        synthetic = correlated.reshape(T, V, C)

        # Add mean position
        synthetic = synthetic + mean_position

        synthetic_list.append(synthetic)

    return np.array(synthetic_list)


def visualize_comparison(real, synthetic, cells, output_path='flag_data/noise_generation_comparison.png'):
    """Compare real and synthetic flag trajectories."""
    fig = plt.figure(figsize=(15, 8))

    T_synth = synthetic.shape[0]
    T_real = real.shape[0]
    timesteps_synth = [0, T_synth//4, T_synth//2, 3*T_synth//4, T_synth-1]
    timesteps_real = [0, T_real//4, T_real//2, 3*T_real//4, T_real-1]

    # Real flag
    for i, t in enumerate(timesteps_real):
        ax = fig.add_subplot(2, len(timesteps_real), i + 1, projection='3d')
        pos = real[t]
        ax.plot_trisurf(pos[:, 0], pos[:, 1], pos[:, 2],
                        triangles=cells, cmap='coolwarm', alpha=0.8,
                        linewidth=0.1, edgecolor='gray')
        ax.set_xlim([0, 4])
        ax.set_ylim([-2, 2])
        ax.set_zlim([-2, 1])
        ax.set_title(f'Real t={t}')
        ax.view_init(elev=20, azim=45)

    # Synthetic flag
    for i, t in enumerate(timesteps_synth):
        ax = fig.add_subplot(2, len(timesteps_synth), len(timesteps_synth) + i + 1, projection='3d')
        pos = synthetic[t]
        ax.plot_trisurf(pos[:, 0], pos[:, 1], pos[:, 2],
                        triangles=cells, cmap='coolwarm', alpha=0.8,
                        linewidth=0.1, edgecolor='gray')
        ax.set_xlim([0, 4])
        ax.set_ylim([-2, 2])
        ax.set_zlim([-2, 1])
        ax.set_title(f'Synthetic t={t}')
        ax.view_init(elev=20, azim=45)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
    print(f"Saved {output_path}")


def visualize_timeseries_comparison(real, synthetic, output_path='flag_data/noise_timeseries_comparison.png'):
    """Compare time series of real vs synthetic."""
    fig, axes = plt.subplots(3, 2, figsize=(14, 10), sharex='col')

    vertex_indices = [0, 500, 1000, 1500]
    t_real = np.arange(real.shape[0])
    t_synth = np.arange(synthetic.shape[0])
    labels = ['X', 'Y', 'Z']

    for i, label in enumerate(labels):
        # Real
        ax = axes[i, 0]
        for v in vertex_indices:
            ax.plot(t_real, real[:, v, i], label=f'V{v}', alpha=0.7)
        ax.set_ylabel(f'{label} position')
        ax.set_title('Real' if i == 0 else '')
        ax.grid(True, alpha=0.3)
        if i == 0:
            ax.legend(loc='upper right')

        # Synthetic
        ax = axes[i, 1]
        for v in vertex_indices:
            ax.plot(t_synth, synthetic[:, v, i], label=f'V{v}', alpha=0.7)
        ax.set_title('Synthetic' if i == 0 else '')
        ax.grid(True, alpha=0.3)

    axes[-1, 0].set_xlabel('Timestep')
    axes[-1, 1].set_xlabel('Timestep')

    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
    print(f"Saved {output_path}")


def visualize_spectrum_comparison(real_trajectories, synthetic, output_path='flag_data/spectrum_comparison.png'):
    """Compare power spectra of real vs synthetic."""
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    # Compute spectra
    real_centered = real_trajectories - real_trajectories.mean(axis=(0, 1))
    synth_centered = synthetic - synthetic.mean(axis=0)

    real_spectrum = np.abs(np.fft.rfft(real_centered, axis=1)).mean(axis=(0, 2))  # (F, V)
    synth_spectrum = np.abs(np.fft.rfft(synth_centered[np.newaxis], axis=1)).mean(axis=(0, 2))  # (F, V)

    # Average over vertices for visualization
    real_avg = real_spectrum.mean(axis=1)
    synth_avg = synth_spectrum.mean(axis=1)

    freqs = np.fft.rfftfreq(real_trajectories.shape[1])

    labels = ['X', 'Y', 'Z']
    for i, (ax, label) in enumerate(zip(axes, labels)):
        real_spec_i = np.abs(np.fft.rfft(real_centered[:, :, :, i], axis=1)).mean(axis=(0, 2))
        synth_spec_i = np.abs(np.fft.rfft(synth_centered[:, :, i], axis=0)).mean(axis=1)

        ax.semilogy(freqs, real_spec_i, label='Real', alpha=0.7)
        ax.semilogy(freqs, synth_spec_i, label='Synthetic', alpha=0.7)
        ax.set_xlabel('Frequency')
        ax.set_ylabel('Magnitude')
        ax.set_title(f'{label}-axis Spectrum')
        ax.legend()
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
    print(f"Saved {output_path}")


def main():
    print("Loading flag data...")
    data = load_flag_data()
    world_pos = data['world_pos']  # (N, T, V, 3)
    cells = data['cells']

    N, T, V, C = world_pos.shape
    print(f"  Shape: {world_pos.shape}")
    print(f"  {N} trajectories, {T} timesteps, {V} vertices")

    # Compute spectral envelope
    print("\nComputing spectral envelope...")
    magnitude_spectrum, mean_position = compute_spectral_envelope(world_pos)
    print(f"  Spectrum shape: {magnitude_spectrum.shape}")

    # Method 1: Simple spectral filtering (no spatial correlation)
    print("\nGenerating with spectral filtering only...")
    synthetic_simple = generate_filtered_noise(
        magnitude_spectrum, mean_position, num_samples=1, seed=42
    )[0]

    # Visualize
    print("\nGenerating visualizations...")
    real_example = world_pos[0]  # Pick first real trajectory

    visualize_comparison(real_example, synthetic_simple, cells,
                        'flag_data/noise_generation_simple.png')
    visualize_timeseries_comparison(real_example, synthetic_simple,
                                    'flag_data/noise_timeseries_simple.png')
    visualize_spectrum_comparison(world_pos, synthetic_simple,
                                  'flag_data/spectrum_comparison.png')

    # Method 2: With spatial correlations
    print("\nComputing spatial covariance...")
    spatial_cov = compute_spatial_covariance(world_pos)
    print(f"  Covariance shape: {spatial_cov.shape}")

    print("\nGenerating with spatial correlations...")
    synthetic_correlated = generate_with_spatial_correlation(
        magnitude_spectrum, mean_position, spatial_cov, num_samples=1, seed=42
    )[0]

    visualize_comparison(real_example, synthetic_correlated, cells,
                        'flag_data/noise_generation_correlated.png')
    visualize_timeseries_comparison(real_example, synthetic_correlated,
                                    'flag_data/noise_timeseries_correlated.png')

    print("\n=== Statistics Comparison ===")
    print(f"Real:       mean={real_example.mean():.4f}, std={real_example.std():.4f}")
    print(f"Simple:     mean={synthetic_simple.mean():.4f}, std={synthetic_simple.std():.4f}")
    print(f"Correlated: mean={synthetic_correlated.mean():.4f}, std={synthetic_correlated.std():.4f}")

    # Velocity statistics
    real_vel = np.diff(real_example, axis=0)
    simple_vel = np.diff(synthetic_simple, axis=0)
    corr_vel = np.diff(synthetic_correlated, axis=0)

    print(f"\nVelocity std:")
    print(f"  Real:       {real_vel.std():.6f}")
    print(f"  Simple:     {simple_vel.std():.6f}")
    print(f"  Correlated: {corr_vel.std():.6f}")

    print("\nDone!")


if __name__ == '__main__':
    main()
