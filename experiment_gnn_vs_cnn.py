#!/usr/bin/env python3
"""
Experiment: Can a GNN match CNN performance on grid-structured data?

This experiment demonstrates that:
1. A regular 2D grid can be viewed as both an image (for CNN) and a graph (for GNN)
2. GNN with edge features can replicate position-aware convolutions like CNN
3. Both should achieve similar denoising performance

We use a simple denoising task:
- Add Gaussian noise to data
- Train model to predict the noise
- Compare reconstruction quality
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, TensorDataset
import time

# =============================================================================
# Create Regular Grid Mesh
# =============================================================================

def create_grid_mesh(H, W):
    """
    Create a regular grid mesh that can be viewed as both image and graph.

    Returns:
        positions: (H*W, 2) vertex positions
        edge_index: (2, E) edges connecting neighbors
        edge_attr: (E, 3) edge features [dx, dy, distance]
    """
    # Create grid positions
    y, x = torch.meshgrid(torch.arange(H), torch.arange(W), indexing='ij')
    positions = torch.stack([x.flatten(), y.flatten()], dim=1).float()
    positions = positions / max(H, W)  # Normalize to [0, 1]

    # Create edges (4-connectivity: up, down, left, right)
    V = H * W
    edges = []

    for i in range(H):
        for j in range(W):
            idx = i * W + j
            # Right neighbor
            if j < W - 1:
                edges.append([idx, idx + 1])
                edges.append([idx + 1, idx])
            # Down neighbor
            if i < H - 1:
                edges.append([idx, idx + W])
                edges.append([idx + W, idx])

    edge_index = torch.tensor(edges, dtype=torch.long).T

    # Edge features: direction vector + distance
    src, dst = edge_index
    edge_vec = positions[dst] - positions[src]
    edge_dist = edge_vec.norm(dim=-1, keepdim=True)
    edge_attr = torch.cat([edge_vec, edge_dist], dim=-1)

    return positions, edge_index, edge_attr


# =============================================================================
# Generate Synthetic Data
# =============================================================================

def generate_data(H, W, num_samples, pattern='waves'):
    """
    Generate synthetic data on a grid.
    Can be viewed as (N, H, W, C) images or (N, V, C) graph signals.
    """
    x = torch.linspace(0, 2*np.pi, W)
    y = torch.linspace(0, 2*np.pi, H)
    xx, yy = torch.meshgrid(x, y, indexing='xy')

    data = []
    for i in range(num_samples):
        # Random frequency and phase
        fx = 1 + torch.rand(1).item() * 2
        fy = 1 + torch.rand(1).item() * 2
        px = torch.rand(1).item() * 2 * np.pi
        py = torch.rand(1).item() * 2 * np.pi

        if pattern == 'waves':
            signal = torch.sin(fx * xx + px) * torch.cos(fy * yy + py)
        elif pattern == 'gaussian':
            cx, cy = torch.rand(2) * 2 * np.pi
            signal = torch.exp(-((xx - cx)**2 + (yy - cy)**2) / 2)

        # Normalize to [-1, 1]
        signal = 2 * (signal - signal.min()) / (signal.max() - signal.min() + 1e-8) - 1
        data.append(signal)

    data = torch.stack(data)  # (N, H, W)
    return data.unsqueeze(-1)  # (N, H, W, 1)


# =============================================================================
# CNN Model
# =============================================================================

class CNNDenoiser(nn.Module):
    """Simple CNN for denoising images."""

    def __init__(self, channels=1, hidden_dim=64):
        super().__init__()

        self.net = nn.Sequential(
            nn.Conv2d(channels, hidden_dim, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(hidden_dim, hidden_dim, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(hidden_dim, hidden_dim, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(hidden_dim, channels, 3, padding=1),
        )

    def forward(self, x):
        # x: (B, H, W, C) -> (B, C, H, W) for conv
        x = x.permute(0, 3, 1, 2)
        out = self.net(x)
        return out.permute(0, 2, 3, 1)  # Back to (B, H, W, C)


# =============================================================================
# GNN Model
# =============================================================================

class GNNConv(nn.Module):
    """Graph convolution with edge features."""

    def __init__(self, in_dim, out_dim, edge_dim=3):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(in_dim * 2 + edge_dim, out_dim),
            nn.ReLU(),
            nn.Linear(out_dim, out_dim),
        )
        self.out_dim = out_dim

    def forward(self, x, edge_index, edge_attr):
        # x: (V, D)
        src, dst = edge_index

        # Gather features
        x_src = x[src]
        x_dst = x[dst]

        # Message: concatenate source, dest, edge features
        messages = self.mlp(torch.cat([x_dst, x_src, edge_attr], dim=-1))

        # Aggregate (mean)
        out = torch.zeros(x.size(0), self.out_dim, device=x.device)
        count = torch.zeros(x.size(0), 1, device=x.device)
        out.scatter_add_(0, dst.unsqueeze(-1).expand(-1, self.out_dim), messages)
        count.scatter_add_(0, dst.unsqueeze(-1), torch.ones_like(dst.float()).unsqueeze(-1))

        return out / count.clamp(min=1)


class GNNDenoiser(nn.Module):
    """GNN for denoising graph signals."""

    def __init__(self, edge_index, edge_attr, channels=1, hidden_dim=64):
        super().__init__()

        self.register_buffer('edge_index', edge_index)
        self.register_buffer('edge_attr', edge_attr)

        self.input_proj = nn.Linear(channels, hidden_dim)
        self.conv1 = GNNConv(hidden_dim, hidden_dim)
        self.conv2 = GNNConv(hidden_dim, hidden_dim)
        self.conv3 = GNNConv(hidden_dim, hidden_dim)
        self.output_proj = nn.Linear(hidden_dim, channels)

    def forward(self, x):
        # x: (B, V, C)
        B, V, C = x.shape

        outputs = []
        for b in range(B):
            h = self.input_proj(x[b])
            h = h + F.relu(self.conv1(h, self.edge_index, self.edge_attr))
            h = h + F.relu(self.conv2(h, self.edge_index, self.edge_attr))
            h = h + F.relu(self.conv3(h, self.edge_index, self.edge_attr))
            out = self.output_proj(h)
            outputs.append(out)

        return torch.stack(outputs)


# =============================================================================
# Training
# =============================================================================

def train_model(model, train_data, val_data, epochs=100, lr=1e-3, noise_std=0.5,
                is_cnn=True, H=None, W=None):
    """Train a denoising model."""

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    device = next(model.parameters()).device

    train_losses = []
    val_losses = []

    for epoch in range(epochs):
        model.train()

        # Add noise
        noise = torch.randn_like(train_data) * noise_std
        noisy = train_data + noise

        if is_cnn:
            noisy_input = noisy.to(device)
        else:
            # Reshape to graph: (N, H, W, C) -> (N, H*W, C)
            noisy_input = noisy.reshape(noisy.shape[0], -1, noisy.shape[-1]).to(device)

        # Forward
        predicted_noise = model(noisy_input)

        if not is_cnn:
            # Reshape back for loss
            predicted_noise = predicted_noise.reshape(train_data.shape)

        loss = F.mse_loss(predicted_noise, noise.to(device))

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        train_losses.append(loss.item())

        # Validation
        model.eval()
        with torch.no_grad():
            noise_val = torch.randn_like(val_data) * noise_std
            noisy_val = val_data + noise_val

            if is_cnn:
                noisy_input_val = noisy_val.to(device)
            else:
                noisy_input_val = noisy_val.reshape(val_data.shape[0], -1, val_data.shape[-1]).to(device)

            pred_val = model(noisy_input_val)

            if not is_cnn:
                pred_val = pred_val.reshape(val_data.shape)

            val_loss = F.mse_loss(pred_val, noise_val.to(device))
            val_losses.append(val_loss.item())

        if (epoch + 1) % 20 == 0:
            print(f"  Epoch {epoch+1:3d} | Train: {loss.item():.4f} | Val: {val_loss.item():.4f}")

    return train_losses, val_losses


def evaluate_denoising(model, test_data, noise_std=0.5, is_cnn=True):
    """Evaluate denoising quality."""
    model.eval()
    device = next(model.parameters()).device

    with torch.no_grad():
        noise = torch.randn_like(test_data) * noise_std
        noisy = test_data + noise

        if is_cnn:
            noisy_input = noisy.to(device)
        else:
            noisy_input = noisy.reshape(test_data.shape[0], -1, test_data.shape[-1]).to(device)

        predicted_noise = model(noisy_input)

        if not is_cnn:
            predicted_noise = predicted_noise.reshape(test_data.shape)

        # Denoise
        denoised = noisy.to(device) - predicted_noise

        # Metrics
        mse_noisy = F.mse_loss(noisy, test_data).item()
        mse_denoised = F.mse_loss(denoised.cpu(), test_data).item()

    return {
        'noisy': noisy[0, :, :, 0].numpy(),
        'denoised': denoised[0, :, :, 0].cpu().numpy(),
        'clean': test_data[0, :, :, 0].numpy(),
        'mse_noisy': mse_noisy,
        'mse_denoised': mse_denoised,
    }


# =============================================================================
# Main Experiment
# =============================================================================

def main():
    print("=" * 60)
    print("Experiment: GNN vs CNN on Grid-Structured Data")
    print("=" * 60)

    # Setup
    H, W = 32, 32  # Grid size
    num_train = 500
    num_val = 100
    num_test = 10
    epochs = 100
    noise_std = 0.3
    hidden_dim = 64

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\nDevice: {device}")
    print(f"Grid size: {H}x{W} = {H*W} vertices")
    print(f"Training samples: {num_train}")
    print(f"Noise std: {noise_std}")

    # Create mesh
    print("\nCreating grid mesh...")
    positions, edge_index, edge_attr = create_grid_mesh(H, W)
    print(f"  Vertices: {positions.shape[0]}")
    print(f"  Edges: {edge_index.shape[1]}")

    # Generate data
    print("\nGenerating synthetic data...")
    train_data = generate_data(H, W, num_train, pattern='waves')
    val_data = generate_data(H, W, num_val, pattern='waves')
    test_data = generate_data(H, W, num_test, pattern='waves')
    print(f"  Train: {train_data.shape}")
    print(f"  Val: {val_data.shape}")
    print(f"  Test: {test_data.shape}")

    # Count parameters
    cnn_model = CNNDenoiser(channels=1, hidden_dim=hidden_dim).to(device)
    gnn_model = GNNDenoiser(edge_index.to(device), edge_attr.to(device),
                            channels=1, hidden_dim=hidden_dim).to(device)

    cnn_params = sum(p.numel() for p in cnn_model.parameters())
    gnn_params = sum(p.numel() for p in gnn_model.parameters())

    print(f"\nModel parameters:")
    print(f"  CNN: {cnn_params:,}")
    print(f"  GNN: {gnn_params:,}")

    # Train CNN
    print("\n" + "-" * 60)
    print("Training CNN...")
    print("-" * 60)
    cnn_model = CNNDenoiser(channels=1, hidden_dim=hidden_dim).to(device)
    start = time.time()
    cnn_train_loss, cnn_val_loss = train_model(
        cnn_model, train_data, val_data, epochs=epochs,
        noise_std=noise_std, is_cnn=True
    )
    cnn_time = time.time() - start
    print(f"  Time: {cnn_time:.1f}s")

    # Train GNN
    print("\n" + "-" * 60)
    print("Training GNN...")
    print("-" * 60)
    gnn_model = GNNDenoiser(edge_index.to(device), edge_attr.to(device),
                            channels=1, hidden_dim=hidden_dim).to(device)
    start = time.time()
    gnn_train_loss, gnn_val_loss = train_model(
        gnn_model, train_data, val_data, epochs=epochs,
        noise_std=noise_std, is_cnn=False, H=H, W=W
    )
    gnn_time = time.time() - start
    print(f"  Time: {gnn_time:.1f}s")

    # Evaluate
    print("\n" + "-" * 60)
    print("Evaluating...")
    print("-" * 60)

    cnn_results = evaluate_denoising(cnn_model, test_data, noise_std, is_cnn=True)
    gnn_results = evaluate_denoising(gnn_model, test_data, noise_std, is_cnn=False)

    print(f"\nDenoising MSE (lower is better):")
    print(f"  Noisy input:  {cnn_results['mse_noisy']:.4f}")
    print(f"  CNN denoised: {cnn_results['mse_denoised']:.4f}")
    print(f"  GNN denoised: {gnn_results['mse_denoised']:.4f}")

    improvement_cnn = (cnn_results['mse_noisy'] - cnn_results['mse_denoised']) / cnn_results['mse_noisy'] * 100
    improvement_gnn = (gnn_results['mse_noisy'] - gnn_results['mse_denoised']) / gnn_results['mse_noisy'] * 100

    print(f"\nNoise reduction:")
    print(f"  CNN: {improvement_cnn:.1f}%")
    print(f"  GNN: {improvement_gnn:.1f}%")

    # Plot results
    fig, axes = plt.subplots(2, 4, figsize=(16, 8))

    # Row 1: CNN
    axes[0, 0].imshow(cnn_results['clean'], cmap='viridis', vmin=-1, vmax=1)
    axes[0, 0].set_title('Clean')
    axes[0, 0].axis('off')

    axes[0, 1].imshow(cnn_results['noisy'], cmap='viridis', vmin=-1, vmax=1)
    axes[0, 1].set_title(f'Noisy (MSE={cnn_results["mse_noisy"]:.4f})')
    axes[0, 1].axis('off')

    axes[0, 2].imshow(cnn_results['denoised'], cmap='viridis', vmin=-1, vmax=1)
    axes[0, 2].set_title(f'CNN Denoised (MSE={cnn_results["mse_denoised"]:.4f})')
    axes[0, 2].axis('off')

    axes[0, 3].plot(cnn_train_loss, label='Train')
    axes[0, 3].plot(cnn_val_loss, label='Val')
    axes[0, 3].set_title('CNN Loss')
    axes[0, 3].legend()
    axes[0, 3].set_xlabel('Epoch')

    # Row 2: GNN
    axes[1, 0].imshow(gnn_results['clean'], cmap='viridis', vmin=-1, vmax=1)
    axes[1, 0].set_title('Clean')
    axes[1, 0].axis('off')

    axes[1, 1].imshow(gnn_results['noisy'], cmap='viridis', vmin=-1, vmax=1)
    axes[1, 1].set_title(f'Noisy (MSE={gnn_results["mse_noisy"]:.4f})')
    axes[1, 1].axis('off')

    axes[1, 2].imshow(gnn_results['denoised'], cmap='viridis', vmin=-1, vmax=1)
    axes[1, 2].set_title(f'GNN Denoised (MSE={gnn_results["mse_denoised"]:.4f})')
    axes[1, 2].axis('off')

    axes[1, 3].plot(gnn_train_loss, label='Train')
    axes[1, 3].plot(gnn_val_loss, label='Val')
    axes[1, 3].set_title('GNN Loss')
    axes[1, 3].legend()
    axes[1, 3].set_xlabel('Epoch')

    plt.suptitle('GNN vs CNN Denoising on Grid Data', fontsize=14)
    plt.tight_layout()
    plt.savefig('experiment_gnn_vs_cnn.png', dpi=150)
    print(f"\nSaved: experiment_gnn_vs_cnn.png")

    # Summary
    print("\n" + "=" * 60)
    print("CONCLUSION")
    print("=" * 60)

    if abs(improvement_cnn - improvement_gnn) < 5:
        print("✓ GNN achieves SIMILAR performance to CNN on grid data!")
        print("  This demonstrates that GNN with edge features can replicate")
        print("  the position-aware convolutions of CNNs.")
    elif improvement_gnn > improvement_cnn:
        print("✓ GNN OUTPERFORMS CNN on this task!")
    else:
        print("✗ CNN outperforms GNN on grid data.")
        print("  This may be due to GNN's less efficient message passing")
        print("  compared to CNN's optimized convolutions.")

    print(f"\nTiming:")
    print(f"  CNN: {cnn_time:.1f}s ({cnn_time/epochs:.2f}s/epoch)")
    print(f"  GNN: {gnn_time:.1f}s ({gnn_time/epochs:.2f}s/epoch)")
    print(f"  GNN is {gnn_time/cnn_time:.1f}x slower (expected due to message passing)")


if __name__ == "__main__":
    main()
