#!/usr/bin/env python3
"""
Train a diffusion model on edge deltas instead of vertex positions.

Key insight: Model edge vectors (deltas) between connected vertices rather than
absolute positions. This makes:
- Translation invariance built-in
- Edge length constraints easy to enforce (just clamp delta magnitudes)
- Boundary conditions simpler (fix one anchor vertex)

See docs/differential_coordinates_plan.md for details.

Usage:
    python train_edge_delta_diffusion.py
"""

import os
import math
import time
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from pathlib import Path
from collections import defaultdict

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader


# =============================================================================
# Configuration
# =============================================================================

class Config:
    # Data
    data_path = "flag_data/flag_test.npz"
    train_split = 0.9

    # Model
    hidden_dim = 128
    num_layers = 4

    # Diffusion
    num_steps = 1000

    # Physics constraints
    max_strain = 0.3  # Maximum stretch/compress (30%)

    # Training
    batch_size = 32
    num_epochs = 100
    learning_rate = 1e-4
    grad_clip = 1.0

    # Output
    output_dir = "flag_edge_delta_output"
    save_every = 20
    sample_every = 20

    # Device
    device = "cuda" if torch.cuda.is_available() else "cpu"


# =============================================================================
# Graph Construction
# =============================================================================

def mesh_to_graph(cells, pos):
    """
    Convert triangle mesh to graph edges.

    Returns:
        edge_index: (2, E) directed edges (each undirected edge appears twice)
        rest_deltas: (E, 3) edge vectors in rest state
        rest_lengths: (E,) edge lengths in rest state
    """
    edges = set()
    for tri in cells:
        for i in range(3):
            for j in range(3):
                if i != j:
                    edges.add((int(tri[i]), int(tri[j])))

    edge_list = list(edges)
    edge_index = torch.tensor(edge_list, dtype=torch.long).T

    # Compute rest state deltas
    src, dst = edge_index
    rest_deltas = pos[dst] - pos[src]
    rest_lengths = rest_deltas.norm(dim=-1)

    return edge_index, rest_deltas, rest_lengths


def build_adjacency(edge_index, num_vertices):
    """Build adjacency list for BFS traversal."""
    adj = defaultdict(list)
    src, dst = edge_index
    for i, (s, d) in enumerate(zip(src.tolist(), dst.tolist())):
        adj[s].append((d, i, +1))  # Forward edge
    return adj


# =============================================================================
# Position <-> Delta Conversion
# =============================================================================

def positions_to_deltas(pos, edge_index):
    """
    Convert vertex positions to edge deltas.

    Args:
        pos: (B, V, 3) or (V, 3) vertex positions
        edge_index: (2, E) edge indices

    Returns:
        deltas: (B, E, 3) or (E, 3) edge vectors
    """
    src, dst = edge_index
    if pos.dim() == 3:
        return pos[:, dst, :] - pos[:, src, :]
    else:
        return pos[dst] - pos[src]


def deltas_to_positions(deltas, edge_index, num_vertices, anchor_idx=0, anchor_pos=None):
    """
    Reconstruct positions from edge deltas via BFS.

    Args:
        deltas: (E, 3) edge vectors
        edge_index: (2, E) edge indices
        num_vertices: total number of vertices
        anchor_idx: which vertex to fix
        anchor_pos: position of anchor vertex (default: origin)

    Returns:
        pos: (V, 3) reconstructed positions
    """
    if anchor_pos is None:
        anchor_pos = torch.zeros(3, device=deltas.device, dtype=deltas.dtype)

    pos = torch.zeros(num_vertices, 3, device=deltas.device, dtype=deltas.dtype)
    pos[anchor_idx] = anchor_pos
    visited = {anchor_idx}

    # Build adjacency for traversal
    adj = defaultdict(list)
    src, dst = edge_index
    for i, (s, d) in enumerate(zip(src.tolist(), dst.tolist())):
        adj[s].append((d, i, +1))  # s -> d: pos[d] = pos[s] + delta[i]
        adj[d].append((s, i, -1))  # d -> s: pos[s] = pos[d] - delta[i]

    # BFS reconstruction
    queue = [anchor_idx]
    while queue:
        i = queue.pop(0)
        for j, edge_idx, sign in adj[i]:
            if j not in visited:
                pos[j] = pos[i] + sign * deltas[edge_idx]
                visited.add(j)
                queue.append(j)

    return pos


def deltas_to_positions_batch(deltas, edge_index, num_vertices, anchor_idx=0):
    """Batch version of deltas_to_positions."""
    B = deltas.shape[0]
    positions = []
    for b in range(B):
        pos = deltas_to_positions(deltas[b], edge_index, num_vertices, anchor_idx)
        positions.append(pos)
    return torch.stack(positions)


# =============================================================================
# Dataset
# =============================================================================

class EdgeDeltaDataset(Dataset):
    """Dataset that provides edge deltas instead of positions."""

    def __init__(self, positions, edge_index):
        """
        Args:
            positions: (N, V, 3) vertex positions
            edge_index: (2, E) edge indices
        """
        self.positions = torch.from_numpy(positions).float()
        self.edge_index = edge_index
        # Pre-compute all deltas
        self.deltas = positions_to_deltas(self.positions, edge_index)

    def __len__(self):
        return len(self.positions)

    def __getitem__(self, idx):
        return self.deltas[idx]


# =============================================================================
# Model Components
# =============================================================================

class TimestepEmbedding(nn.Module):
    """Sinusoidal embedding for noise timestep."""

    def __init__(self, dim):
        super().__init__()
        self.dim = dim
        self.mlp = nn.Sequential(
            nn.Linear(dim, dim * 4),
            nn.SiLU(),
            nn.Linear(dim * 4, dim),
        )

    def forward(self, n):
        half = self.dim // 2
        freqs = torch.exp(-math.log(10000) * torch.arange(half, device=n.device) / half)
        args = n[:, None].float() * freqs[None, :]
        emb = torch.cat([torch.sin(args), torch.cos(args)], dim=-1)
        return self.mlp(emb)


class EdgeConvLayer(nn.Module):
    """
    Message passing between edges that share vertices.

    Edges communicate through shared vertices:
    edge_ij aggregates info from all edges incident to i and j.
    """

    def __init__(self, hidden_dim):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.hidden_dim = hidden_dim

    def forward(self, h_edges, edge_index, num_vertices):
        """
        Args:
            h_edges: (B, E, D) edge features
            edge_index: (2, E) vertex indices for each edge
            num_vertices: V

        Returns:
            h_new: (B, E, D) updated edge features
        """
        B, E, D = h_edges.shape
        src, dst = edge_index

        # Aggregate edge features to vertices
        vertex_features = torch.zeros(B, num_vertices, D, device=h_edges.device, dtype=h_edges.dtype)

        # Sum features of incident edges to each vertex
        src_expanded = src.unsqueeze(0).unsqueeze(-1).expand(B, -1, D)
        dst_expanded = dst.unsqueeze(0).unsqueeze(-1).expand(B, -1, D)
        vertex_features.scatter_add_(1, src_expanded, h_edges)
        vertex_features.scatter_add_(1, dst_expanded, h_edges)

        # Each edge reads from its two vertices
        h_src = vertex_features[:, src, :]  # (B, E, D)
        h_dst = vertex_features[:, dst, :]  # (B, E, D)

        # Combine
        h_new = self.mlp(torch.cat([h_src, h_dst], dim=-1))
        return h_new


class EdgeDeltaDiffusion(nn.Module):
    """
    Diffusion model operating on edge deltas instead of vertex positions.

    The model predicts noise on edge deltas, and we can enforce edge length
    constraints during denoising.
    """

    def __init__(self, edge_index, rest_deltas, rest_lengths, num_vertices,
                 hidden_dim=128, num_layers=4):
        super().__init__()

        self.register_buffer('edge_index', edge_index)
        self.register_buffer('rest_deltas', rest_deltas)
        self.register_buffer('rest_lengths', rest_lengths)

        self.num_edges = edge_index.shape[1]
        self.num_vertices = num_vertices
        self.hidden_dim = hidden_dim

        # Input projection: delta xyz -> hidden
        self.input_proj = nn.Linear(3, hidden_dim)

        # Edge convolution layers
        self.layers = nn.ModuleList([
            EdgeConvLayer(hidden_dim) for _ in range(num_layers)
        ])

        # Layer norms for residual connections
        self.norms = nn.ModuleList([
            nn.LayerNorm(hidden_dim) for _ in range(num_layers)
        ])

        # Timestep embedding
        self.time_embed = TimestepEmbedding(hidden_dim)

        # Output projection: hidden -> delta noise
        self.output_proj = nn.Linear(hidden_dim, 3)

    def forward(self, delta_noisy, n):
        """
        Args:
            delta_noisy: (B, E, 3) noisy edge deltas
            n: (B,) noise timestep

        Returns:
            noise_pred: (B, E, 3) predicted noise on deltas
        """
        B, E, _ = delta_noisy.shape

        # Embed deltas
        h = self.input_proj(delta_noisy)  # (B, E, D)

        # Add timestep embedding
        t_emb = self.time_embed(n)  # (B, D)
        h = h + t_emb.unsqueeze(1)  # (B, E, D)

        # Message passing between edges (via shared vertices)
        for layer, norm in zip(self.layers, self.norms):
            h_new = layer(h, self.edge_index, self.num_vertices)
            h = norm(h + h_new)  # Residual + LayerNorm

        # Predict noise
        noise_pred = self.output_proj(h)  # (B, E, 3)
        return noise_pred


# =============================================================================
# Diffusion Schedule with Constraints
# =============================================================================

class ConstrainedDiffusionSchedule:
    """
    DDPM diffusion schedule with physics constraints on edge deltas.

    During denoising, we clamp edge lengths to stay within valid range.
    """

    def __init__(self, num_steps, rest_lengths, max_strain=0.3, device='cuda'):
        self.num_steps = num_steps
        self.device = device

        # Store rest lengths and compute valid range (all on device)
        self.rest_lengths = rest_lengths.to(device)
        self.min_length = self.rest_lengths * (1 - max_strain)
        self.max_length = self.rest_lengths * (1 + max_strain)

        # DDPM schedule (linear beta)
        self.betas = torch.linspace(1e-4, 0.02, num_steps, device=device)
        self.alphas = 1 - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)
        self.alphas_cumprod_prev = F.pad(self.alphas_cumprod[:-1], (1, 0), value=1.0)

        self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1 - self.alphas_cumprod)
        self.sqrt_recip_alphas = torch.sqrt(1 / self.alphas)

        # Posterior variance
        self.posterior_variance = self.betas * (1 - self.alphas_cumprod_prev) / (1 - self.alphas_cumprod)

    def q_sample(self, x0, n, noise=None):
        """Forward process: add noise to x0."""
        if noise is None:
            noise = torch.randn_like(x0)

        sqrt_alpha = self.sqrt_alphas_cumprod[n].view(-1, 1, 1)
        sqrt_one_minus = self.sqrt_one_minus_alphas_cumprod[n].view(-1, 1, 1)

        return sqrt_alpha * x0 + sqrt_one_minus * noise

    def constrain_deltas(self, deltas):
        """
        Project deltas to satisfy edge length constraints.

        Clamps edge lengths to [rest * (1-strain), rest * (1+strain)].
        """
        # Get directions and lengths
        lengths = deltas.norm(dim=-1, keepdim=True).clamp(min=1e-8)  # (B, E, 1)
        directions = deltas / lengths

        # Clamp lengths to valid range
        min_len = self.min_length.view(1, -1, 1)
        max_len = self.max_length.view(1, -1, 1)
        lengths_clamped = lengths.clamp(min=min_len, max=max_len)

        # Reconstruct constrained deltas
        return directions * lengths_clamped

    @torch.no_grad()
    def p_sample(self, model, delta_n, n, constrain=True):
        """Denoise one step, optionally with constraint enforcement."""
        B = delta_n.shape[0]
        n_tensor = torch.full((B,), n, device=delta_n.device, dtype=torch.long)

        # Predict noise
        predicted_noise = model(delta_n, n_tensor)

        # Compute mean
        beta = self.betas[n]
        sqrt_recip = self.sqrt_recip_alphas[n]
        sqrt_one_minus = self.sqrt_one_minus_alphas_cumprod[n]
        mean = sqrt_recip * (delta_n - beta / sqrt_one_minus * predicted_noise)

        if n == 0:
            delta_new = mean
        else:
            noise = torch.randn_like(delta_n)
            std = torch.sqrt(self.posterior_variance[n])
            delta_new = mean + std * noise

        # Enforce constraints
        if constrain:
            delta_new = self.constrain_deltas(delta_new)

        return delta_new

    @torch.no_grad()
    def generate(self, model, num_samples, num_edges, constrain=True):
        """Generate samples from noise."""
        # Start with random noise
        deltas = torch.randn(num_samples, num_edges, 3, device=self.device)

        # Denoise
        for n in reversed(range(self.num_steps)):
            deltas = self.p_sample(model, deltas, n, constrain=constrain)

            if n % 200 == 0:
                lengths = deltas.norm(dim=-1)
                print(f"  Step {n}: mean edge length = {lengths.mean():.4f}, "
                      f"std = {lengths.std():.4f}")

        return deltas


# =============================================================================
# Training
# =============================================================================

def train_epoch(model, dataloader, optimizer, schedule, device, grad_clip=1.0):
    """Train for one epoch."""
    model.train()
    total_loss = 0

    for batch_deltas in dataloader:
        batch_deltas = batch_deltas.to(device)
        B = batch_deltas.shape[0]

        # Sample noise and timesteps
        noise = torch.randn_like(batch_deltas)
        n = torch.randint(0, schedule.num_steps, (B,), device=device)

        # Forward process
        deltas_noisy = schedule.q_sample(batch_deltas, n, noise)

        # Predict noise
        predicted = model(deltas_noisy, n)

        # Loss
        loss = F.mse_loss(predicted, noise)

        optimizer.zero_grad()
        loss.backward()
        if grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step()

        total_loss += loss.item()

    return total_loss / len(dataloader)


def evaluate(model, dataloader, schedule, device):
    """Evaluate on validation set."""
    model.eval()
    total_loss = 0

    with torch.no_grad():
        for batch_deltas in dataloader:
            batch_deltas = batch_deltas.to(device)
            B = batch_deltas.shape[0]

            noise = torch.randn_like(batch_deltas)
            n = torch.randint(0, schedule.num_steps, (B,), device=device)

            deltas_noisy = schedule.q_sample(batch_deltas, n, noise)
            predicted = model(deltas_noisy, n)

            loss = F.mse_loss(predicted, noise)
            total_loss += loss.item()

    return total_loss / len(dataloader)


# =============================================================================
# Visualization
# =============================================================================

def plot_samples(positions, cells, title, output_path):
    """Plot generated mesh samples."""
    num_samples = min(len(positions), 6)

    fig, axes = plt.subplots(2, 3, figsize=(12, 8), subplot_kw={'projection': '3d'})
    axes = axes.flatten()

    for i in range(num_samples):
        ax = axes[i]
        pos = positions[i].cpu().numpy()

        triangles = pos[cells]
        mesh = Poly3DCollection(triangles, alpha=0.6, facecolor='steelblue',
                                edgecolor='k', linewidth=0.1)
        ax.add_collection3d(mesh)

        # Set limits
        ax.set_xlim(pos[:, 0].min(), pos[:, 0].max())
        ax.set_ylim(pos[:, 1].min(), pos[:, 1].max())
        ax.set_zlim(pos[:, 2].min(), pos[:, 2].max())

        # Mark anchor vertex
        ax.scatter([pos[0, 0]], [pos[0, 1]], [pos[0, 2]],
                   c='red', s=100, marker='o', zorder=10)

        ax.set_title(f'Sample {i+1}')
        ax.view_init(elev=20, azim=45)

    # Hide unused axes
    for i in range(num_samples, 6):
        axes[i].set_visible(False)

    plt.suptitle(title)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()


def plot_edge_length_histogram(deltas, rest_lengths, title, output_path):
    """Plot histogram of edge lengths vs rest lengths."""
    lengths = deltas.norm(dim=-1).cpu().numpy().flatten()
    rest = rest_lengths.cpu().numpy()

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

    # Histogram of generated lengths
    ax1.hist(lengths, bins=50, alpha=0.7, label='Generated')
    ax1.axvline(rest.mean(), color='r', linestyle='--', label=f'Rest mean: {rest.mean():.3f}')
    ax1.set_xlabel('Edge Length')
    ax1.set_ylabel('Count')
    ax1.set_title('Edge Length Distribution')
    ax1.legend()

    # Strain distribution (length / rest_length - 1)
    # Expand rest to match generated
    rest_expanded = np.tile(rest, len(deltas))
    strain = lengths / rest_expanded - 1

    ax2.hist(strain, bins=50, alpha=0.7)
    ax2.axvline(0, color='r', linestyle='--', label='Zero strain')
    ax2.axvline(-0.3, color='orange', linestyle=':', label='Max compression')
    ax2.axvline(0.3, color='orange', linestyle=':', label='Max stretch')
    ax2.set_xlabel('Strain (length/rest - 1)')
    ax2.set_ylabel('Count')
    ax2.set_title('Strain Distribution')
    ax2.legend()

    plt.suptitle(title)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()


# =============================================================================
# Main
# =============================================================================

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--resume', type=str, default=None,
                        help='Path to checkpoint to resume from')
    args = parser.parse_args()

    config = Config()
    print(f"Device: {config.device}")

    # Create output directory
    output_dir = Path(config.output_dir)
    output_dir.mkdir(exist_ok=True)

    # Load data
    print("\nLoading data...")
    data = np.load(config.data_path)
    world_pos = data['world_pos']  # (N, T, V, 3)
    cells = data['cells']
    mesh_pos = data['mesh_pos']

    N, T, V, C = world_pos.shape
    print(f"  Trajectories: {N}, Frames: {T}, Vertices: {V}")

    # Flatten to frames
    frames = world_pos.reshape(-1, V, C)
    print(f"  Total frames: {len(frames)}")

    # Build graph
    print("\nBuilding graph...")
    mesh_pos_3d = np.zeros((V, 3), dtype=np.float32)
    mesh_pos_3d[:, :2] = mesh_pos
    mesh_pos_tensor = torch.from_numpy(mesh_pos_3d).float()

    edge_index, rest_deltas, rest_lengths = mesh_to_graph(cells, mesh_pos_tensor)
    print(f"  Edges: {edge_index.shape[1]}")
    print(f"  Mean rest edge length: {rest_lengths.mean():.4f}")

    # Split train/val randomly
    n_train = int(config.train_split * len(frames))
    np.random.seed(42)
    indices = np.random.permutation(len(frames))
    train_idx = indices[:n_train]
    val_idx = indices[n_train:]

    train_frames = frames[train_idx]
    val_frames = frames[val_idx]

    # Create datasets
    print("\nCreating datasets...")
    train_dataset = EdgeDeltaDataset(train_frames, edge_index)
    val_dataset = EdgeDeltaDataset(val_frames, edge_index)

    train_loader = DataLoader(train_dataset, batch_size=config.batch_size,
                              shuffle=True, num_workers=0, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=config.batch_size,
                            shuffle=False, num_workers=0)

    print(f"  Train samples: {len(train_dataset)}")
    print(f"  Val samples: {len(val_dataset)}")

    # Create model
    print("\nCreating model...")
    model = EdgeDeltaDiffusion(
        edge_index=edge_index,
        rest_deltas=rest_deltas,
        rest_lengths=rest_lengths,
        num_vertices=V,
        hidden_dim=config.hidden_dim,
        num_layers=config.num_layers
    ).to(config.device)

    num_params = sum(p.numel() for p in model.parameters())
    print(f"  Parameters: {num_params:,}")

    # Create schedule and optimizer
    schedule = ConstrainedDiffusionSchedule(
        num_steps=config.num_steps,
        rest_lengths=rest_lengths,
        max_strain=config.max_strain,
        device=config.device
    )

    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate)

    # Resume from checkpoint if specified
    start_epoch = 1
    best_val_loss = float('inf')
    train_losses = []
    val_losses = []

    if args.resume:
        print(f"\nResuming from {args.resume}...")
        checkpoint = torch.load(args.resume, map_location=config.device)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
        best_val_loss = checkpoint.get('val_loss', float('inf'))
        print(f"  Resuming from epoch {start_epoch}, best val loss: {best_val_loss:.6f}")

    # Training loop
    print("\nTraining...")

    for epoch in range(start_epoch, config.num_epochs + 1):
        start_time = time.time()

        train_loss = train_epoch(model, train_loader, optimizer, schedule,
                                 config.device, config.grad_clip)
        val_loss = evaluate(model, val_loader, schedule, config.device)

        train_losses.append(train_loss)
        val_losses.append(val_loss)

        elapsed = time.time() - start_time
        print(f"Epoch {epoch:3d}/{config.num_epochs} | "
              f"Train: {train_loss:.6f} | Val: {val_loss:.6f} | "
              f"Time: {elapsed:.1f}s")

        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_loss': train_loss,
                'val_loss': val_loss,
                'config': vars(config),
                'edge_index': edge_index,
                'rest_deltas': rest_deltas,
                'rest_lengths': rest_lengths,
            }, output_dir / 'best_model.pt')

        # Periodic saves and samples
        if epoch % config.save_every == 0:
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_loss': train_loss,
                'val_loss': val_loss,
            }, output_dir / f'checkpoint_epoch{epoch}.pt')

        if epoch % config.sample_every == 0:
            print("  Generating samples...")
            model.eval()

            # Generate with constraints
            deltas = schedule.generate(model, num_samples=6,
                                       num_edges=edge_index.shape[1],
                                       constrain=True)

            # Convert to positions
            positions = deltas_to_positions_batch(deltas, edge_index, V, anchor_idx=0)

            # Plot meshes
            plot_samples(positions, cells, f'Epoch {epoch} (Constrained)',
                         output_dir / f'samples_epoch{epoch}.png')

            # Plot edge length distribution
            plot_edge_length_histogram(deltas, rest_lengths, f'Epoch {epoch}',
                                       output_dir / f'edge_lengths_epoch{epoch}.png')

            # Also generate without constraints for comparison
            deltas_unconstrained = schedule.generate(model, num_samples=6,
                                                      num_edges=edge_index.shape[1],
                                                      constrain=False)
            positions_unconstrained = deltas_to_positions_batch(
                deltas_unconstrained, edge_index, V, anchor_idx=0)
            plot_samples(positions_unconstrained, cells, f'Epoch {epoch} (Unconstrained)',
                         output_dir / f'samples_unconstrained_epoch{epoch}.png')

    # Plot training curves
    plt.figure(figsize=(10, 4))
    plt.plot(train_losses, label='Train')
    plt.plot(val_losses, label='Val')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training Curves')
    plt.legend()
    plt.savefig(output_dir / 'training_curves.png', dpi=150)
    plt.close()

    print(f"\nTraining complete! Best val loss: {best_val_loss:.6f}")
    print(f"Output saved to: {output_dir}")


if __name__ == '__main__':
    main()
