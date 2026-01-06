#!/usr/bin/env python3
"""
Spectral VAE for Graph Signal Augmentation

A lightweight VAE operating in the spectral domain of a fixed mesh.
Learns to generate variations of graph signals for data augmentation.

Usage:
    python train_spectral_vae.py
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from pathlib import Path
import matplotlib.pyplot as plt


# =============================================================================
# Configuration
# =============================================================================

CONFIG = {
    # Spectral settings
    'K': 50,                    # Number of spectral coefficients (max 50 available)
    'D': 16,                    # Latent dimension

    # Architecture
    'hidden_dim': 64,           # MLP hidden layer size

    # Training
    'epochs': 1000,
    'batch_size': 32,
    'learning_rate': 1e-3,
    'beta': 0.001,              # KL weight (start small for good reconstruction)
    'beta_warmup': 200,         # Epochs to warm up beta from 0

    # Output
    'output_dir': 'spectral_vae_output',
    'save_interval': 200,
}


# =============================================================================
# Model Definition
# =============================================================================

class SpectralVAE(nn.Module):
    """
    Variational Autoencoder operating in the spectral domain of a graph.

    Input: Spectral coefficients (K,)
    Output: Reconstructed spectral coefficients (K,)
    Latent: Low-dimensional representation (D,) with noise injection
    """

    def __init__(self, K, D, hidden_dim=64):
        super().__init__()
        self.K = K
        self.D = D

        # Encoder: spectral coeffs -> latent distribution
        self.encoder = nn.Sequential(
            nn.Linear(K, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )
        self.fc_mu = nn.Linear(hidden_dim, D)
        self.fc_logvar = nn.Linear(hidden_dim, D)

        # Decoder: latent -> spectral coeffs
        self.decoder = nn.Sequential(
            nn.Linear(D, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, K),
        )

    def encode(self, x):
        """Encode spectral coefficients to latent distribution parameters."""
        h = self.encoder(x)
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)
        return mu, logvar

    def reparameterize(self, mu, logvar, noise_scale=1.0):
        """Sample from latent distribution using reparameterization trick."""
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + noise_scale * std * eps

    def decode(self, z):
        """Decode latent vector to spectral coefficients."""
        return self.decoder(z)

    def forward(self, x, noise_scale=1.0):
        """Full forward pass: encode, sample, decode."""
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar, noise_scale)
        x_recon = self.decode(z)
        return x_recon, mu, logvar


def vae_loss(x, x_recon, mu, logvar, beta=1.0):
    """
    VAE loss = reconstruction loss + beta * KL divergence
    """
    # Reconstruction loss (MSE in spectral domain)
    recon_loss = nn.functional.mse_loss(x_recon, x, reduction='mean')

    # KL divergence: KL(q(z|x) || p(z)) where p(z) = N(0,1)
    kl_loss = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())

    return recon_loss + beta * kl_loss, recon_loss, kl_loss


# =============================================================================
# Graph Fourier Transform
# =============================================================================

class GraphFourierTransform:
    """
    Graph Fourier Transform using precomputed Laplacian eigenvectors.
    """

    def __init__(self, eigenvectors, eigenvalues):
        """
        Args:
            eigenvectors: (N, K) matrix of Laplacian eigenvectors
            eigenvalues: (K,) vector of eigenvalues
        """
        self.U = torch.tensor(eigenvectors, dtype=torch.float32)  # (N, K)
        self.eigenvalues = torch.tensor(eigenvalues, dtype=torch.float32)
        self.N, self.K = self.U.shape

    def to(self, device):
        """Move to device."""
        self.U = self.U.to(device)
        self.eigenvalues = self.eigenvalues.to(device)
        return self

    def gft(self, x):
        """
        Graph Fourier Transform: vertex domain -> spectral domain

        Args:
            x: (batch, N) or (N,) signal in vertex domain
        Returns:
            x_hat: (batch, K) or (K,) spectral coefficients
        """
        if x.dim() == 1:
            return self.U.T @ x
        return x @ self.U  # (batch, N) @ (N, K) -> (batch, K)

    def igft(self, x_hat):
        """
        Inverse Graph Fourier Transform: spectral domain -> vertex domain

        Args:
            x_hat: (batch, K) or (K,) spectral coefficients
        Returns:
            x: (batch, N) or (N,) signal in vertex domain
        """
        if x_hat.dim() == 1:
            return self.U @ x_hat
        return x_hat @ self.U.T  # (batch, K) @ (K, N) -> (batch, N)


# =============================================================================
# Data Loading
# =============================================================================

def load_dataset(dataset_dir='dataset'):
    """Load mesh and solutions from the dataset directory."""
    dataset_dir = Path(dataset_dir)

    # Load mesh with eigenvectors
    mesh = np.load(dataset_dir / 'mesh.npz')
    eigenvectors = mesh['eigenvectors']  # (N, K)
    eigenvalues = mesh['eigenvalues']    # (K,)
    points = mesh['points']              # (N, 2)
    triangles = mesh['triangles']        # (T, 3)

    # Load solutions
    solutions = np.load(dataset_dir / 'solutions.npz')
    signals = solutions['solutions']      # (num_samples, N)

    print(f"Loaded {signals.shape[0]} signals on mesh with {signals.shape[1]} vertices")
    print(f"Using {eigenvectors.shape[1]} spectral components")

    return {
        'signals': signals,
        'eigenvectors': eigenvectors,
        'eigenvalues': eigenvalues,
        'points': points,
        'triangles': triangles,
    }


# =============================================================================
# Training
# =============================================================================

def train(config=CONFIG):
    """Train the Spectral VAE."""

    # Setup
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    output_dir = Path(config['output_dir'])
    output_dir.mkdir(exist_ok=True)

    # Load data
    data = load_dataset()
    signals = torch.tensor(data['signals'], dtype=torch.float32)

    # Initialize GFT
    gft = GraphFourierTransform(data['eigenvectors'], data['eigenvalues']).to(device)

    # Transform all signals to spectral domain
    signals = signals.to(device)
    signals_spectral = gft.gft(signals)  # (num_samples, K)

    # Normalize spectral coefficients for better training
    spec_mean = signals_spectral.mean(dim=0, keepdim=True)
    spec_std = signals_spectral.std(dim=0, keepdim=True) + 1e-8
    signals_spectral_norm = (signals_spectral - spec_mean) / spec_std

    print(f"Spectral coefficients shape: {signals_spectral.shape}")
    print(f"Spectral energy per component: min={signals_spectral.std(0).min():.4f}, max={signals_spectral.std(0).max():.4f}")

    # Initialize model
    K = config['K']
    D = config['D']
    model = SpectralVAE(K=K, D=D, hidden_dim=config['hidden_dim']).to(device)
    optimizer = optim.Adam(model.parameters(), lr=config['learning_rate'])

    print(f"\nModel: SpectralVAE(K={K}, D={D}, hidden={config['hidden_dim']})")
    print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Training loop
    num_samples = signals_spectral_norm.shape[0]
    batch_size = min(config['batch_size'], num_samples)

    history = {'loss': [], 'recon': [], 'kl': []}

    print(f"\nTraining for {config['epochs']} epochs...")

    for epoch in range(config['epochs']):
        model.train()

        # Shuffle data
        perm = torch.randperm(num_samples)

        epoch_loss = 0
        epoch_recon = 0
        epoch_kl = 0
        num_batches = 0

        # Beta warmup
        if epoch < config['beta_warmup']:
            beta = config['beta'] * (epoch / config['beta_warmup'])
        else:
            beta = config['beta']

        for i in range(0, num_samples, batch_size):
            batch_idx = perm[i:i+batch_size]
            x = signals_spectral_norm[batch_idx]

            # Forward pass
            x_recon, mu, logvar = model(x)

            # Compute loss
            loss, recon_loss, kl_loss = vae_loss(x, x_recon, mu, logvar, beta)

            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            epoch_recon += recon_loss.item()
            epoch_kl += kl_loss.item()
            num_batches += 1

        # Record history
        history['loss'].append(epoch_loss / num_batches)
        history['recon'].append(epoch_recon / num_batches)
        history['kl'].append(epoch_kl / num_batches)

        # Print progress
        if (epoch + 1) % 100 == 0:
            print(f"Epoch {epoch+1:4d}: loss={history['loss'][-1]:.6f}, "
                  f"recon={history['recon'][-1]:.6f}, kl={history['kl'][-1]:.6f}, beta={beta:.6f}")

        # Save checkpoint
        if (epoch + 1) % config['save_interval'] == 0:
            save_checkpoint(model, optimizer, epoch, history, spec_mean, spec_std, output_dir)

    # Final save
    save_checkpoint(model, optimizer, config['epochs']-1, history, spec_mean, spec_std, output_dir, final=True)

    # Generate visualizations
    print("\nGenerating visualizations...")
    visualize_results(model, gft, signals, signals_spectral_norm, spec_mean, spec_std,
                      data['points'], data['triangles'], history, output_dir, device)

    print(f"\nTraining complete! Results saved to {output_dir}/")
    return model, gft, spec_mean, spec_std


def save_checkpoint(model, optimizer, epoch, history, spec_mean, spec_std, output_dir, final=False):
    """Save model checkpoint."""
    filename = 'model_final.pt' if final else f'model_epoch_{epoch+1}.pt'
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'history': history,
        'spec_mean': spec_mean.cpu(),
        'spec_std': spec_std.cpu(),
        'config': CONFIG,
    }, output_dir / filename)


# =============================================================================
# Augmentation
# =============================================================================

def augment_signal(model, gft, signal, spec_mean, spec_std, num_augmentations=10, noise_scale=1.0, device='cpu'):
    """
    Generate augmented versions of a signal.

    Args:
        model: Trained SpectralVAE
        gft: GraphFourierTransform instance
        signal: (N,) original signal in vertex domain
        spec_mean, spec_std: Normalization parameters
        num_augmentations: Number of augmented signals to generate
        noise_scale: Controls variation (0=deterministic, 1=normal, >1=more variation)
        device: torch device

    Returns:
        augmented: (num_augmentations, N) augmented signals in vertex domain
    """
    model.eval()

    with torch.no_grad():
        # To spectral domain and normalize
        if not isinstance(signal, torch.Tensor):
            signal = torch.tensor(signal, dtype=torch.float32)
        signal = signal.to(device)

        x_spec = gft.gft(signal)
        x_spec_norm = (x_spec - spec_mean.squeeze()) / spec_std.squeeze()

        # Encode to get latent distribution
        mu, logvar = model.encode(x_spec_norm.unsqueeze(0))

        # Sample multiple times with different noise
        augmented_spectral = []
        for _ in range(num_augmentations):
            z = model.reparameterize(mu, logvar, noise_scale)
            x_recon_norm = model.decode(z)
            augmented_spectral.append(x_recon_norm)

        augmented_spectral = torch.cat(augmented_spectral, dim=0)  # (num_aug, K)

        # Denormalize
        augmented_spectral = augmented_spectral * spec_std + spec_mean

        # Back to vertex domain
        augmented = gft.igft(augmented_spectral)  # (num_aug, N)

    return augmented.cpu().numpy()


# =============================================================================
# Visualization
# =============================================================================

def visualize_results(model, gft, signals, signals_spectral_norm, spec_mean, spec_std,
                      points, triangles, history, output_dir, device):
    """Generate visualization plots."""

    model.eval()

    # 1. Training curves
    fig, axes = plt.subplots(1, 3, figsize=(12, 3))
    axes[0].plot(history['loss'])
    axes[0].set_title('Total Loss')
    axes[0].set_xlabel('Epoch')

    axes[1].plot(history['recon'])
    axes[1].set_title('Reconstruction Loss')
    axes[1].set_xlabel('Epoch')

    axes[2].plot(history['kl'])
    axes[2].set_title('KL Divergence')
    axes[2].set_xlabel('Epoch')

    plt.tight_layout()
    plt.savefig(output_dir / 'training_curves.png', dpi=150)
    plt.close()

    # 2. Original vs Reconstructed vs Augmented
    with torch.no_grad():
        # Pick a random signal
        idx = np.random.randint(0, signals.shape[0])
        original = signals[idx].cpu().numpy()

        # Reconstruct (noise_scale=0 for deterministic)
        x_spec_norm = signals_spectral_norm[idx:idx+1]
        x_recon_norm, mu, logvar = model(x_spec_norm, noise_scale=0)
        x_recon_spec = x_recon_norm * spec_std + spec_mean
        reconstructed = gft.igft(x_recon_spec).squeeze().cpu().numpy()

        # Generate augmentations
        augmented = augment_signal(
            model, gft, signals[idx], spec_mean, spec_std,
            num_augmentations=4, noise_scale=1.0, device=device
        )

    # Plot
    fig, axes = plt.subplots(2, 3, figsize=(12, 8))

    def plot_signal(ax, signal, title):
        sc = ax.tripcolor(points[:, 0], points[:, 1], triangles, signal,
                          shading='gouraud', cmap='viridis')
        ax.set_aspect('equal')
        ax.set_title(title)
        plt.colorbar(sc, ax=ax)

    plot_signal(axes[0, 0], original, 'Original')
    plot_signal(axes[0, 1], reconstructed, 'Reconstructed')
    plot_signal(axes[0, 2], original - reconstructed, 'Difference')

    for i in range(3):
        plot_signal(axes[1, i], augmented[i], f'Augmented {i+1}')

    plt.tight_layout()
    plt.savefig(output_dir / 'augmentation_examples.png', dpi=150)
    plt.close()

    # 3. Latent space visualization
    with torch.no_grad():
        mu_all, _ = model.encode(signals_spectral_norm)
        mu_all = mu_all.cpu().numpy()

    # PCA for visualization if D > 2
    if mu_all.shape[1] > 2:
        from sklearn.decomposition import PCA
        pca = PCA(n_components=2)
        mu_2d = pca.fit_transform(mu_all)
    else:
        mu_2d = mu_all

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.scatter(mu_2d[:, 0], mu_2d[:, 1], alpha=0.6)
    ax.set_xlabel('Latent dim 1')
    ax.set_ylabel('Latent dim 2')
    ax.set_title('Latent Space (PCA)' if mu_all.shape[1] > 2 else 'Latent Space')
    plt.tight_layout()
    plt.savefig(output_dir / 'latent_space.png', dpi=150)
    plt.close()

    # 4. Noise scale comparison
    with torch.no_grad():
        idx = np.random.randint(0, signals.shape[0])
        original = signals[idx].cpu().numpy()

        noise_scales = [0, 0.5, 1.0, 2.0]
        fig, axes = plt.subplots(1, len(noise_scales) + 1, figsize=(3 * (len(noise_scales) + 1), 3))

        plot_signal(axes[0], original, 'Original')

        for i, scale in enumerate(noise_scales):
            aug = augment_signal(
                model, gft, signals[idx], spec_mean, spec_std,
                num_augmentations=1, noise_scale=scale, device=device
            )[0]
            plot_signal(axes[i + 1], aug, f'scale={scale}')

    plt.tight_layout()
    plt.savefig(output_dir / 'noise_scale_comparison.png', dpi=150)
    plt.close()

    print(f"  - training_curves.png")
    print(f"  - augmentation_examples.png")
    print(f"  - latent_space.png")
    print(f"  - noise_scale_comparison.png")


# =============================================================================
# Main
# =============================================================================

if __name__ == '__main__':
    model, gft, spec_mean, spec_std = train()
