#!/usr/bin/env python3
"""
Train a trajectory diffusion model on flag simulation data.

This implements the architecture from docs/trajectory_diffusion.md:
- GNN encoder/decoder with edge features for direction-aware message passing
- Temporal transformer for sequence modeling
- DDPM diffusion for trajectory generation

Usage:
    python train_flag_diffusion.py

Requirements:
    - PyTorch with CUDA
    - PyTorch Geometric
    - numpy, matplotlib
"""

import os
import math
import time
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

# Check for PyTorch Geometric
try:
    from torch_geometric.nn import MessagePassing
    HAS_PYG = True
except ImportError:
    HAS_PYG = False
    print("Warning: torch_geometric not found. Using fallback implementation.")


# =============================================================================
# Configuration
# =============================================================================

class Config:
    # Data
    data_path = "flag_data/flag_test.npz"
    num_frames = 100        # Subsample trajectories to this many frames
    train_split = 0.8       # Fraction for training

    # Model
    hidden_dim = 128
    num_gnn_layers = 4
    num_transformer_layers = 4
    num_heads = 8
    edge_dim = 4

    # Diffusion
    num_diffusion_steps = 1000
    beta_start = 1e-4
    beta_end = 0.02

    # Training
    batch_size = 2          # Small due to memory constraints
    num_epochs = 100
    learning_rate = 1e-4
    grad_clip = 1.0

    # Output
    output_dir = "flag_diffusion_output"
    save_every = 10         # Save checkpoint every N epochs
    sample_every = 20       # Generate samples every N epochs

    # Device
    device = "cuda" if torch.cuda.is_available() else "cpu"


# =============================================================================
# Graph Construction
# =============================================================================

def mesh_to_graph(cells, pos):
    """
    Convert triangle mesh to graph edges WITH EDGE FEATURES.

    Args:
        cells: (F, 3) triangle vertex indices
        pos: (V, 3) vertex positions (rest state)

    Returns:
        edge_index: (2, E) for PyTorch Geometric
        edge_attr: (E, 4) edge features [direction_x, direction_y, direction_z, length]
    """
    edges = set()
    for tri in cells:
        # Add edges for each triangle edge (bidirectional)
        edges.add((int(tri[0]), int(tri[1])))
        edges.add((int(tri[1]), int(tri[0])))
        edges.add((int(tri[1]), int(tri[2])))
        edges.add((int(tri[2]), int(tri[1])))
        edges.add((int(tri[2]), int(tri[0])))
        edges.add((int(tri[0]), int(tri[2])))

    edge_list = list(edges)
    edge_index = torch.tensor(edge_list, dtype=torch.long).T  # (2, E)

    # Compute edge features (direction vectors)
    src, dst = edge_index
    edge_vec = pos[dst] - pos[src]  # (E, 3) vector from src to dst
    edge_len = edge_vec.norm(dim=-1, keepdim=True)  # (E, 1)
    edge_attr = torch.cat([edge_vec, edge_len], dim=-1)  # (E, 4)

    return edge_index, edge_attr


# =============================================================================
# Model Components
# =============================================================================

if HAS_PYG:
    class MeshConv(MessagePassing):
        """
        Graph convolution for mesh data WITH EDGE FEATURES.
        """
        def __init__(self, in_dim, out_dim, edge_dim=4):
            super().__init__(aggr='mean')

            self.mlp = nn.Sequential(
                nn.Linear(in_dim * 2 + edge_dim, out_dim),
                nn.LayerNorm(out_dim),
                nn.SiLU(),
                nn.Linear(out_dim, out_dim),
            )

        def forward(self, x, edge_index, edge_attr):
            return self.propagate(edge_index, x=x, edge_attr=edge_attr)

        def message(self, x_i, x_j, edge_attr):
            return self.mlp(torch.cat([x_i, x_j, edge_attr], dim=-1))
else:
    class MeshConv(nn.Module):
        """
        Fallback graph convolution without PyTorch Geometric.
        Less efficient but works without the dependency.
        """
        def __init__(self, in_dim, out_dim, edge_dim=4):
            super().__init__()

            self.mlp = nn.Sequential(
                nn.Linear(in_dim * 2 + edge_dim, out_dim),
                nn.LayerNorm(out_dim),
                nn.SiLU(),
                nn.Linear(out_dim, out_dim),
            )
            self.out_dim = out_dim

        def forward(self, x, edge_index, edge_attr):
            # x: (V, D), edge_index: (2, E), edge_attr: (E, edge_dim)
            src, dst = edge_index

            # Gather source and destination features
            x_src = x[src]  # (E, D)
            x_dst = x[dst]  # (E, D)

            # Compute messages
            messages = self.mlp(torch.cat([x_dst, x_src, edge_attr], dim=-1))  # (E, out_dim)

            # Aggregate (mean) messages to destination nodes
            out = torch.zeros(x.size(0), self.out_dim, device=x.device, dtype=x.dtype)
            count = torch.zeros(x.size(0), 1, device=x.device, dtype=x.dtype)

            out.scatter_add_(0, dst.unsqueeze(-1).expand(-1, self.out_dim), messages)
            count.scatter_add_(0, dst.unsqueeze(-1), torch.ones_like(dst, dtype=x.dtype).unsqueeze(-1))

            return out / count.clamp(min=1)


class MeshEncoder(nn.Module):
    """Encode per-vertex 3D positions to feature vectors."""

    def __init__(self, in_dim=3, hidden_dim=64, out_dim=128, num_layers=4, edge_dim=4):
        super().__init__()

        self.input_proj = nn.Linear(in_dim, hidden_dim)

        self.layers = nn.ModuleList([
            MeshConv(hidden_dim, hidden_dim, edge_dim=edge_dim)
            for _ in range(num_layers)
        ])

        self.output_proj = nn.Linear(hidden_dim, out_dim)

    def forward(self, x, edge_index, edge_attr):
        h = self.input_proj(x)

        for layer in self.layers:
            h = h + layer(h, edge_index, edge_attr)  # Residual

        return self.output_proj(h)


class MeshDecoder(nn.Module):
    """Decode feature vectors back to 3D positions (noise prediction)."""

    def __init__(self, in_dim=128, hidden_dim=64, out_dim=3, num_layers=4, edge_dim=4):
        super().__init__()

        self.input_proj = nn.Linear(in_dim, hidden_dim)

        self.layers = nn.ModuleList([
            MeshConv(hidden_dim, hidden_dim, edge_dim=edge_dim)
            for _ in range(num_layers)
        ])

        self.output_proj = nn.Linear(hidden_dim, out_dim)

    def forward(self, h, edge_index, edge_attr):
        h = self.input_proj(h)

        for layer in self.layers:
            h = h + layer(h, edge_index, edge_attr)

        return self.output_proj(h)


class TemporalTransformer(nn.Module):
    """Process temporal sequence of frame embeddings."""

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
        # x: (B, T, D), t_embed: (B, D)
        B, T, D = x.shape

        # Add positional embedding
        x = x + self.pos_embed[:, :T, :]

        # Add timestep embedding to all frames
        x = x + t_embed.unsqueeze(1)

        return self.transformer(x)


class TimestepEmbedding(nn.Module):
    """Sinusoidal timestep embedding (like in original DDPM)."""

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
        emb = t[:, None].float() * emb[None, :]
        emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=-1)
        return self.mlp(emb)


class TrajectoryDiffusion(nn.Module):
    """
    Full trajectory diffusion model with GNN + Transformer.
    Uses edge features for direction-aware message passing.
    """

    def __init__(self,
                 cells,
                 mesh_pos,
                 hidden_dim=128,
                 num_gnn_layers=4,
                 num_transformer_layers=4,
                 num_heads=8,
                 edge_dim=4):
        super().__init__()

        # Precompute graph structure and edge features
        edge_index, edge_attr = mesh_to_graph(cells, mesh_pos)
        self.register_buffer('edge_index', edge_index)
        self.register_buffer('edge_attr', edge_attr)

        self.num_vertices = mesh_pos.shape[0]

        self.encoder = MeshEncoder(
            in_dim=3,
            hidden_dim=hidden_dim // 2,
            out_dim=hidden_dim,
            num_layers=num_gnn_layers,
            edge_dim=edge_dim
        )

        self.time_embed = TimestepEmbedding(hidden_dim)

        # Pool per-vertex features to per-frame
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
            hidden_dim=hidden_dim // 2,
            out_dim=3,
            num_layers=num_gnn_layers,
            edge_dim=edge_dim
        )

    def forward(self, x, t):
        """
        x: (B, T, V, 3) noisy trajectory
        t: (B,) diffusion timestep (noise_step)

        returns: (B, T, V, 3) predicted noise
        """
        B, T, V, C = x.shape

        # Timestep embedding
        t_emb = self.time_embed(t)  # (B, D)

        # Encode each frame with GNN
        x_flat = x.reshape(B * T, V, C)

        h_list = []
        for i in range(B * T):
            h_frame = self.encoder(x_flat[i], self.edge_index, self.edge_attr)
            h_list.append(h_frame)
        h = torch.stack(h_list)  # (B*T, V, D)

        # Pool to per-frame embedding
        h = h.mean(dim=1)  # (B*T, D)
        h = self.frame_pool(h)
        h = h.reshape(B, T, -1)  # (B, T, D)

        # Temporal transformer
        h = self.temporal(h, t_emb)  # (B, T, D)

        # Unpool to per-vertex
        h = self.frame_unpool(h)
        h = h.unsqueeze(2).expand(-1, -1, V, -1)  # (B, T, V, D)
        h = h.reshape(B * T, V, -1)

        # Decode each frame with GNN
        out_list = []
        for i in range(B * T):
            out_frame = self.decoder(h[i], self.edge_index, self.edge_attr)
            out_list.append(out_frame)
        out = torch.stack(out_list)

        return out.reshape(B, T, V, C)


# =============================================================================
# Diffusion Utilities
# =============================================================================

def cosine_beta_schedule(timesteps, s=0.008):
    """Cosine schedule as proposed in https://arxiv.org/abs/2102.09672"""
    steps = timesteps + 1
    x = torch.linspace(0, timesteps, steps)
    alphas_cumprod = torch.cos(((x / timesteps) + s) / (1 + s) * math.pi * 0.5) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    return torch.clip(betas, 0.0001, 0.9999)


def linear_beta_schedule(timesteps, beta_start=1e-4, beta_end=0.02):
    """Linear schedule from original DDPM."""
    return torch.linspace(beta_start, beta_end, timesteps)


class DiffusionSchedule:
    """Precomputed diffusion schedule values."""

    def __init__(self, num_steps=1000, schedule='cosine', device='cpu'):
        if schedule == 'cosine':
            betas = cosine_beta_schedule(num_steps)
        else:
            betas = linear_beta_schedule(num_steps)

        alphas = 1.0 - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)
        alphas_cumprod_prev = F.pad(alphas_cumprod[:-1], (1, 0), value=1.0)

        # Precompute values for q(x_t | x_0)
        self.sqrt_alphas_cumprod = torch.sqrt(alphas_cumprod).to(device)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1.0 - alphas_cumprod).to(device)

        # Precompute values for posterior q(x_{t-1} | x_t, x_0)
        self.sqrt_recip_alphas = torch.sqrt(1.0 / alphas).to(device)
        self.posterior_variance = (betas * (1.0 - alphas_cumprod_prev) / (1.0 - alphas_cumprod)).to(device)
        self.betas = betas.to(device)

        self.num_steps = num_steps

    def q_sample(self, x_0, t, noise=None):
        """Forward diffusion: add noise to x_0 to get x_t."""
        if noise is None:
            noise = torch.randn_like(x_0)

        sqrt_alpha = self.sqrt_alphas_cumprod[t]
        sqrt_one_minus_alpha = self.sqrt_one_minus_alphas_cumprod[t]

        # Reshape for broadcasting: (B,) -> (B, 1, 1, 1)
        while sqrt_alpha.dim() < x_0.dim():
            sqrt_alpha = sqrt_alpha.unsqueeze(-1)
            sqrt_one_minus_alpha = sqrt_one_minus_alpha.unsqueeze(-1)

        return sqrt_alpha * x_0 + sqrt_one_minus_alpha * noise

    @torch.no_grad()
    def p_sample(self, model, x_t, t):
        """Reverse diffusion: denoise x_t to get x_{t-1}."""
        # Predict noise
        t_tensor = torch.full((x_t.shape[0],), t, device=x_t.device, dtype=torch.long)
        predicted_noise = model(x_t, t_tensor)

        # Get schedule values
        beta = self.betas[t]
        sqrt_recip_alpha = self.sqrt_recip_alphas[t]
        sqrt_one_minus_alpha = self.sqrt_one_minus_alphas_cumprod[t]

        # Compute mean
        model_mean = sqrt_recip_alpha * (x_t - beta / sqrt_one_minus_alpha * predicted_noise)

        if t == 0:
            return model_mean
        else:
            noise = torch.randn_like(x_t)
            posterior_std = torch.sqrt(self.posterior_variance[t])
            return model_mean + posterior_std * noise

    @torch.no_grad()
    def sample(self, model, shape, device='cuda', progress=True):
        """Generate samples from pure noise."""
        x = torch.randn(shape, device=device)

        steps = range(self.num_steps - 1, -1, -1)
        if progress:
            from tqdm import tqdm
            steps = tqdm(steps, desc="Sampling")

        for t in steps:
            x = self.p_sample(model, x, t)

        return x


# =============================================================================
# Dataset
# =============================================================================

class FlagTrajectoryDataset(Dataset):
    """Dataset of flag trajectories."""

    def __init__(self, trajectories, num_frames=100):
        """
        Args:
            trajectories: (N, T, V, 3) array of trajectories
            num_frames: number of frames to subsample to
        """
        self.num_frames = num_frames

        # Subsample trajectories to fixed number of frames
        N, T, V, C = trajectories.shape

        if T > num_frames:
            # Evenly sample frames
            indices = np.linspace(0, T - 1, num_frames, dtype=int)
            trajectories = trajectories[:, indices, :, :]

        # Normalize trajectories (zero mean, unit variance per trajectory)
        self.mean = trajectories.mean(axis=(1, 2, 3), keepdims=True)
        self.std = trajectories.std(axis=(1, 2, 3), keepdims=True) + 1e-8
        trajectories = (trajectories - self.mean) / self.std

        self.trajectories = torch.tensor(trajectories, dtype=torch.float32)

        print(f"Dataset: {len(self)} trajectories, {num_frames} frames, {V} vertices")

    def __len__(self):
        return len(self.trajectories)

    def __getitem__(self, idx):
        return self.trajectories[idx]


# =============================================================================
# Training
# =============================================================================

def train_epoch(model, dataloader, optimizer, schedule, device, grad_clip=1.0):
    """Train for one epoch."""
    model.train()
    total_loss = 0
    num_batches = 0

    for batch in dataloader:
        batch = batch.to(device)
        B = batch.shape[0]

        # Sample random timesteps
        t = torch.randint(0, schedule.num_steps, (B,), device=device)

        # Add noise
        noise = torch.randn_like(batch)
        x_t = schedule.q_sample(batch, t, noise)

        # Predict noise
        predicted_noise = model(x_t, t)

        # Loss
        loss = F.mse_loss(predicted_noise, noise)

        # Backward
        optimizer.zero_grad()
        loss.backward()

        if grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)

        optimizer.step()

        total_loss += loss.item()
        num_batches += 1

    return total_loss / num_batches


def evaluate(model, dataloader, schedule, device):
    """Evaluate on validation set."""
    model.eval()
    total_loss = 0
    num_batches = 0

    with torch.no_grad():
        for batch in dataloader:
            batch = batch.to(device)
            B = batch.shape[0]

            t = torch.randint(0, schedule.num_steps, (B,), device=device)
            noise = torch.randn_like(batch)
            x_t = schedule.q_sample(batch, t, noise)
            predicted_noise = model(x_t, t)
            loss = F.mse_loss(predicted_noise, noise)

            total_loss += loss.item()
            num_batches += 1

    return total_loss / num_batches


# =============================================================================
# Visualization
# =============================================================================

def visualize_trajectory(trajectory, cells, title="Flag Trajectory", save_path=None):
    """Visualize a single trajectory as a sequence of frames."""
    T = trajectory.shape[0]

    # Select frames to show
    num_frames = min(8, T)
    indices = np.linspace(0, T - 1, num_frames, dtype=int)

    fig, axes = plt.subplots(2, 4, figsize=(16, 8), subplot_kw={'projection': '3d'})
    axes = axes.flatten()

    for i, idx in enumerate(indices):
        ax = axes[i]
        pos = trajectory[idx]  # (V, 3)

        # Plot mesh
        ax.plot_trisurf(
            pos[:, 0], pos[:, 1], pos[:, 2],
            triangles=cells,
            cmap='viridis',
            alpha=0.8,
            linewidth=0.1,
            edgecolor='gray'
        )

        ax.set_title(f"Frame {idx}")
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')

        # Set consistent view
        ax.view_init(elev=20, azim=45)

    plt.suptitle(title)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150)
        print(f"Saved: {save_path}")

    plt.close()


def compare_real_vs_generated(real, generated, cells, save_path=None):
    """Compare real and generated trajectories side by side."""
    fig, axes = plt.subplots(2, 4, figsize=(16, 8), subplot_kw={'projection': '3d'})

    # Show 4 frames from each
    T = real.shape[0]
    indices = np.linspace(0, T - 1, 4, dtype=int)

    for i, idx in enumerate(indices):
        # Real
        ax = axes[0, i]
        pos = real[idx]
        ax.plot_trisurf(pos[:, 0], pos[:, 1], pos[:, 2], triangles=cells,
                       cmap='Blues', alpha=0.8, linewidth=0.1, edgecolor='gray')
        ax.set_title(f"Real - Frame {idx}")
        ax.view_init(elev=20, azim=45)

        # Generated
        ax = axes[1, i]
        pos = generated[idx]
        ax.plot_trisurf(pos[:, 0], pos[:, 1], pos[:, 2], triangles=cells,
                       cmap='Oranges', alpha=0.8, linewidth=0.1, edgecolor='gray')
        ax.set_title(f"Generated - Frame {idx}")
        ax.view_init(elev=20, azim=45)

    plt.suptitle("Real vs Generated Trajectories")
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150)
        print(f"Saved: {save_path}")

    plt.close()


# =============================================================================
# Main
# =============================================================================

def main():
    cfg = Config()

    print("=" * 60)
    print("Trajectory Diffusion Training")
    print("=" * 60)
    print(f"Device: {cfg.device}")
    print(f"Data: {cfg.data_path}")
    print()

    # Create output directory
    os.makedirs(cfg.output_dir, exist_ok=True)

    # Load data
    print("Loading data...")
    if not os.path.exists(cfg.data_path):
        print(f"Error: {cfg.data_path} not found!")
        print("Please run load_flag_data.py first to convert TFRecord to numpy.")
        return

    data = np.load(cfg.data_path)
    world_pos = data['world_pos']  # (N, T, V, 3)
    cells = data['cells']          # (F, 3)
    mesh_pos = data['mesh_pos']    # (V, 2) - rest state (2D)

    print(f"  Trajectories: {world_pos.shape}")
    print(f"  Mesh: {cells.shape[0]} triangles, {mesh_pos.shape[0]} vertices")

    # Convert 2D mesh_pos to 3D (add z=0)
    if mesh_pos.shape[1] == 2:
        mesh_pos_3d = np.zeros((mesh_pos.shape[0], 3), dtype=np.float32)
        mesh_pos_3d[:, :2] = mesh_pos
        mesh_pos = mesh_pos_3d

    # Split data
    N = len(world_pos)
    n_train = int(N * cfg.train_split)
    train_data = world_pos[:n_train]
    val_data = world_pos[n_train:]

    print(f"  Train: {len(train_data)}, Val: {len(val_data)}")

    # Create datasets
    train_dataset = FlagTrajectoryDataset(train_data, num_frames=cfg.num_frames)
    val_dataset = FlagTrajectoryDataset(val_data, num_frames=cfg.num_frames)

    train_loader = DataLoader(train_dataset, batch_size=cfg.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=cfg.batch_size, shuffle=False)

    # Create model
    print("\nBuilding model...")
    cells_tensor = torch.tensor(cells, dtype=torch.long)
    mesh_pos_tensor = torch.tensor(mesh_pos, dtype=torch.float32)

    model = TrajectoryDiffusion(
        cells=cells_tensor,
        mesh_pos=mesh_pos_tensor,
        hidden_dim=cfg.hidden_dim,
        num_gnn_layers=cfg.num_gnn_layers,
        num_transformer_layers=cfg.num_transformer_layers,
        num_heads=cfg.num_heads,
        edge_dim=cfg.edge_dim,
    ).to(cfg.device)

    num_params = sum(p.numel() for p in model.parameters())
    print(f"  Parameters: {num_params:,}")

    # Create diffusion schedule
    schedule = DiffusionSchedule(
        num_steps=cfg.num_diffusion_steps,
        schedule='cosine',
        device=cfg.device
    )

    # Optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.learning_rate)

    # Training loop
    print("\nTraining...")
    print("-" * 60)

    train_losses = []
    val_losses = []
    best_val_loss = float('inf')

    for epoch in range(1, cfg.num_epochs + 1):
        start_time = time.time()

        # Train
        train_loss = train_epoch(model, train_loader, optimizer, schedule, cfg.device, cfg.grad_clip)
        train_losses.append(train_loss)

        # Validate
        val_loss = evaluate(model, val_loader, schedule, cfg.device)
        val_losses.append(val_loss)

        elapsed = time.time() - start_time

        print(f"Epoch {epoch:3d}/{cfg.num_epochs} | "
              f"Train: {train_loss:.4f} | Val: {val_loss:.4f} | "
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
            }, os.path.join(cfg.output_dir, 'best_model.pt'))

        # Save periodic checkpoint
        if epoch % cfg.save_every == 0:
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_losses': train_losses,
                'val_losses': val_losses,
            }, os.path.join(cfg.output_dir, f'checkpoint_epoch{epoch}.pt'))

        # Generate samples
        if epoch % cfg.sample_every == 0:
            print("  Generating samples...")
            model.eval()

            # Get sample shape from dataset
            sample_shape = (1, cfg.num_frames, train_dataset.trajectories.shape[2], 3)

            with torch.no_grad():
                generated = schedule.sample(model, sample_shape, device=cfg.device, progress=False)

            # Denormalize
            generated = generated.cpu().numpy()[0]  # (T, V, 3)

            # Denormalize using first trajectory's stats
            generated = generated * train_dataset.std[0, 0, 0, 0] + train_dataset.mean[0, 0, 0, 0]

            # Compare with real
            real = train_data[0]
            if real.shape[0] > cfg.num_frames:
                indices = np.linspace(0, real.shape[0] - 1, cfg.num_frames, dtype=int)
                real = real[indices]

            compare_real_vs_generated(
                real, generated, cells,
                save_path=os.path.join(cfg.output_dir, f'comparison_epoch{epoch}.png')
            )

    # Plot training curves
    plt.figure(figsize=(10, 5))
    plt.plot(train_losses, label='Train')
    plt.plot(val_losses, label='Validation')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training Curves')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig(os.path.join(cfg.output_dir, 'training_curves.png'), dpi=150)
    plt.close()

    print("\n" + "=" * 60)
    print("Training complete!")
    print(f"Best validation loss: {best_val_loss:.4f}")
    print(f"Output directory: {cfg.output_dir}")
    print("=" * 60)


if __name__ == "__main__":
    main()
