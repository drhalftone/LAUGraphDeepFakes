#!/usr/bin/env python3
"""
Train a diffusion model to generate graph signals on a fixed mesh.

This implements single-frame generation (not trajectories):
- Train on individual frames from flag simulations
- Generate new frames from pure noise
- Augment existing frames via partial denoising

Usage:
    python train_flag_diffusion.py

See docs/graph_signal_diffusion.md for details.
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
    train_split = 0.9

    # Model
    hidden_dim = 128
    num_layers = 4
    edge_dim = 4

    # Diffusion
    num_steps = 1000

    # Training
    batch_size = 32
    num_epochs = 100
    learning_rate = 1e-4
    grad_clip = 1.0

    # Output
    output_dir = "flag_diffusion_output"
    save_every = 20
    sample_every = 20

    # Device
    device = "cuda" if torch.cuda.is_available() else "cpu"


# =============================================================================
# Graph Construction
# =============================================================================

def mesh_to_graph(cells, pos):
    """
    Convert triangle mesh to graph with edge features.

    Args:
        cells: (F, 3) triangle vertex indices
        pos: (V, 3) vertex positions (rest state)

    Returns:
        edge_index: (2, E) edges
        edge_attr: (E, 4) edge features [dx, dy, dz, length]
    """
    edges = set()
    for tri in cells:
        for i in range(3):
            for j in range(3):
                if i != j:
                    edges.add((int(tri[i]), int(tri[j])))

    edge_index = torch.tensor(list(edges), dtype=torch.long).T

    # Edge features: direction vector + length
    src, dst = edge_index
    edge_vec = pos[dst] - pos[src]
    edge_len = edge_vec.norm(dim=-1, keepdim=True)
    edge_attr = torch.cat([edge_vec, edge_len], dim=-1)

    return edge_index, edge_attr


def compute_laplacian(edge_index, num_nodes, normalize=True):
    """
    Compute graph Laplacian matrix from edge index.

    Args:
        edge_index: (2, E) edges
        num_nodes: number of vertices
        normalize: if True, compute normalized Laplacian L = I - D^{-1/2} A D^{-1/2}

    Returns:
        L: (V, V) Laplacian matrix
    """
    # Build adjacency matrix
    src, dst = edge_index
    A = torch.zeros(num_nodes, num_nodes)
    A[src, dst] = 1.0

    # Degree matrix
    D = A.sum(dim=1)

    if normalize:
        # Normalized Laplacian: L = I - D^{-1/2} A D^{-1/2}
        D_inv_sqrt = torch.zeros_like(D)
        mask = D > 0
        D_inv_sqrt[mask] = D[mask].pow(-0.5)
        D_inv_sqrt = torch.diag(D_inv_sqrt)
        L = torch.eye(num_nodes) - D_inv_sqrt @ A @ D_inv_sqrt
    else:
        # Unnormalized Laplacian: L = D - A
        L = torch.diag(D) - A

    return L


# =============================================================================
# Model Components
# =============================================================================

if HAS_PYG:
    class MeshConv(MessagePassing):
        """Graph convolution with edge features."""

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
        """Fallback graph convolution without PyTorch Geometric."""

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
            src, dst = edge_index
            x_src = x[src]
            x_dst = x[dst]

            messages = self.mlp(torch.cat([x_dst, x_src, edge_attr], dim=-1))

            out = torch.zeros(x.size(0), self.out_dim, device=x.device, dtype=x.dtype)
            count = torch.zeros(x.size(0), 1, device=x.device, dtype=x.dtype)

            out.scatter_add_(0, dst.unsqueeze(-1).expand(-1, self.out_dim), messages)
            count.scatter_add_(0, dst.unsqueeze(-1), torch.ones_like(dst, dtype=x.dtype).unsqueeze(-1))

            return out / count.clamp(min=1)


class TimestepEmbedding(nn.Module):
    """Sinusoidal embedding for noise step."""

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


class GraphSignalDiffusion(nn.Module):
    """Diffusion model for single-frame graph signals."""

    def __init__(self, cells, mesh_pos, hidden_dim=128, num_layers=4, edge_dim=4):
        super().__init__()

        # Precompute graph structure
        edge_index, edge_attr = mesh_to_graph(cells, mesh_pos)
        self.register_buffer('edge_index', edge_index)
        self.register_buffer('edge_attr', edge_attr)

        self.num_vertices = mesh_pos.shape[0]
        self.hidden_dim = hidden_dim

        # Encoder
        self.input_proj = nn.Linear(3, hidden_dim)
        self.encoder_layers = nn.ModuleList([
            MeshConv(hidden_dim, hidden_dim, edge_dim) for _ in range(num_layers)
        ])

        # Timestep embedding
        self.time_embed = TimestepEmbedding(hidden_dim)

        # Decoder
        self.decoder_layers = nn.ModuleList([
            MeshConv(hidden_dim, hidden_dim, edge_dim) for _ in range(num_layers)
        ])
        self.output_proj = nn.Linear(hidden_dim, 3)

    def forward(self, x, n):
        """
        x: (B, V, 3) noisy graph signal
        n: (B,) noise step

        returns: (B, V, 3) predicted noise
        """
        B, V, C = x.shape

        # Timestep embedding
        t_emb = self.time_embed(n)  # (B, D)

        # Batch all nodes together: (B, V, 3) -> (B*V, 3)
        x_flat = x.reshape(B * V, C)

        # Create batched edge_index with node offsets for each sample
        # This allows processing all batch items in a single GNN pass
        offsets = torch.arange(B, device=x.device) * V
        edge_index_b = torch.cat([
            self.edge_index + offset for offset in offsets
        ], dim=1)
        edge_attr_b = self.edge_attr.repeat(B, 1)

        # Encode (single pass for entire batch)
        h = self.input_proj(x_flat)  # (B*V, D)
        for layer in self.encoder_layers:
            h = h + layer(h, edge_index_b, edge_attr_b)

        # Add timestep embedding (broadcast to each node in each sample)
        h = h.reshape(B, V, -1)
        h = h + t_emb.unsqueeze(1)  # (B, V, D)
        h = h.reshape(B * V, -1)

        # Decode (single pass for entire batch)
        for layer in self.decoder_layers:
            h = h + layer(h, edge_index_b, edge_attr_b)

        out = self.output_proj(h)  # (B*V, 3)
        return out.reshape(B, V, C)


# =============================================================================
# Diffusion Schedule
# =============================================================================

def cosine_beta_schedule(num_steps, s=0.008):
    """Cosine schedule from improved DDPM."""
    steps = num_steps + 1
    x = torch.linspace(0, num_steps, steps)
    alphas_cumprod = torch.cos(((x / num_steps) + s) / (1 + s) * math.pi * 0.5) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    return torch.clip(betas, 0.0001, 0.9999)


class DiffusionSchedule:
    """Precomputed diffusion values."""

    def __init__(self, num_steps=1000, device='cpu'):
        betas = cosine_beta_schedule(num_steps)
        alphas = 1.0 - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)
        alphas_cumprod_prev = F.pad(alphas_cumprod[:-1], (1, 0), value=1.0)

        self.sqrt_alphas_cumprod = torch.sqrt(alphas_cumprod).to(device)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1.0 - alphas_cumprod).to(device)
        self.sqrt_recip_alphas = torch.sqrt(1.0 / alphas).to(device)
        self.posterior_variance = (betas * (1.0 - alphas_cumprod_prev) / (1.0 - alphas_cumprod)).to(device)
        self.betas = betas.to(device)

        self.num_steps = num_steps

    def q_sample(self, x_0, n, noise=None):
        """Add noise to x_0 at step n."""
        if noise is None:
            noise = torch.randn_like(x_0)

        sqrt_alpha = self.sqrt_alphas_cumprod[n]
        sqrt_one_minus = self.sqrt_one_minus_alphas_cumprod[n]

        while sqrt_alpha.dim() < x_0.dim():
            sqrt_alpha = sqrt_alpha.unsqueeze(-1)
            sqrt_one_minus = sqrt_one_minus.unsqueeze(-1)

        return sqrt_alpha * x_0 + sqrt_one_minus * noise

    @torch.no_grad()
    def p_sample(self, model, x_n, n):
        """Denoise one step."""
        n_tensor = torch.full((x_n.shape[0],), n, device=x_n.device, dtype=torch.long)
        predicted_noise = model(x_n, n_tensor)

        beta = self.betas[n]
        sqrt_recip = self.sqrt_recip_alphas[n]
        sqrt_one_minus = self.sqrt_one_minus_alphas_cumprod[n]

        mean = sqrt_recip * (x_n - beta / sqrt_one_minus * predicted_noise)

        if n == 0:
            return mean
        else:
            noise = torch.randn_like(x_n)
            std = torch.sqrt(self.posterior_variance[n])
            return mean + std * noise

    @torch.no_grad()
    def sample(self, model, shape, device='cuda'):
        """Generate from pure noise."""
        x = torch.randn(shape, device=device)

        for n in reversed(range(self.num_steps)):
            x = self.p_sample(model, x, n)
            if n % 100 == 0:
                print(f"  Step {n}...")

        return x

    @torch.no_grad()
    def augment(self, model, x_seed, start_step=500):
        """Generate variation of x_seed."""
        device = x_seed.device

        # Add noise to intermediate level
        noise = torch.randn_like(x_seed)
        alpha = self.sqrt_alphas_cumprod[start_step]
        one_minus = self.sqrt_one_minus_alphas_cumprod[start_step]
        x = alpha * x_seed + one_minus * noise

        # Denoise from start_step to 0
        for n in reversed(range(start_step + 1)):
            x = self.p_sample(model, x, n)

        return x


# =============================================================================
# GAD-Style Diffusion (Graph-Aware, using Heat Equation)
# =============================================================================

class GADSchedule:
    """
    Graph-Aware Diffusion schedule from arXiv:2510.05036.

    Key difference from standard DDPM:
    - Forward process uses heat equation on graph Laplacian
    - Noise spreads along graph edges, not isotropically
    - Smooth signals (low graph frequency) decay slower
    """

    def __init__(self, laplacian, num_steps=1000, sigma=1.0, gamma=0.1, device='cpu'):
        """
        Args:
            laplacian: (V, V) graph Laplacian matrix
            num_steps: number of diffusion steps
            sigma: noise scale
            gamma: regularization (L_gamma = L + gamma*I for stability)
        """
        self.num_steps = num_steps
        self.sigma = sigma
        self.gamma = gamma

        # L_gamma = L + gamma*I (ensures positive eigenvalues)
        V = laplacian.shape[0]
        L_gamma = laplacian + gamma * torch.eye(V)
        self.L_gamma = L_gamma.to(device)

        # Eigendecomposition for efficient computation
        # L_gamma = U @ diag(eigenvalues) @ U^T
        eigenvalues, U = torch.linalg.eigh(L_gamma)
        self.eigenvalues = eigenvalues.to(device)  # (V,)
        self.U = U.to(device)  # (V, V)

        # Precompute time schedule (FCPS - Floor Constrained Polynomial Schedule)
        # c_t controls how fast we add noise
        t = torch.linspace(0, 1, num_steps + 1)
        # Polynomial schedule: slower at start, faster at end
        self.c_t = (t ** 2).to(device)  # Simple quadratic, could use FCPS formula

        self.device = device

    def _heat_kernel(self, t_val):
        """
        Compute exp(-t * L_gamma) via eigendecomposition.

        This is the heat kernel - it smooths signals along graph edges.
        """
        # exp(-t * L_gamma) = U @ diag(exp(-t * eigenvalues)) @ U^T
        exp_eigenvalues = torch.exp(-t_val * self.eigenvalues)
        return self.U @ torch.diag(exp_eigenvalues) @ self.U.T

    def q_sample(self, x_0, n, noise=None):
        """
        Forward process: add graph-aware noise using heat equation.

        Unlike standard DDPM which adds isotropic noise,
        GAD adds noise that respects graph structure.
        """
        if noise is None:
            noise = torch.randn_like(x_0)

        # Time value for this step
        t_val = self.c_t[n].item() if isinstance(n, int) else self.c_t[n]

        # Heat kernel at time t
        H_t = self._heat_kernel(t_val)  # (V, V)

        # For batched input (B, V, 3)
        if x_0.dim() == 3:
            B, V, C = x_0.shape
            x_t = torch.zeros_like(x_0)
            for b in range(B):
                for c in range(C):
                    # Signal decays via heat equation
                    signal_part = H_t @ x_0[b, :, c]
                    # Noise is also filtered (correlated with graph structure)
                    noise_part = self.sigma * (torch.eye(V, device=self.device) - H_t @ H_t) @ noise[b, :, c]
                    x_t[b, :, c] = signal_part + noise_part.sqrt().abs() * noise[b, :, c]
            return x_t
        else:
            # Single sample (V, 3)
            V, C = x_0.shape
            x_t = torch.zeros_like(x_0)
            for c in range(C):
                signal_part = H_t @ x_0[:, c]
                x_t[:, c] = signal_part + self.sigma * torch.sqrt(1 - torch.exp(-2 * t_val * self.eigenvalues.mean())) * noise[:, c]
            return x_t

    @torch.no_grad()
    def p_sample(self, model, x_n, n):
        """Denoise one step (same as standard DDPM for now)."""
        n_tensor = torch.full((x_n.shape[0],), n, device=x_n.device, dtype=torch.long)
        predicted_noise = model(x_n, n_tensor)

        # Simple denoising step
        # In full GAD, this would use the score function with graph structure
        step_size = 1.0 / self.num_steps
        x_prev = x_n - step_size * predicted_noise

        if n > 0:
            noise = torch.randn_like(x_n)
            x_prev = x_prev + math.sqrt(step_size) * self.sigma * noise

        return x_prev

    @torch.no_grad()
    def sample(self, model, shape, device='cuda'):
        """Generate from pure noise."""
        x = torch.randn(shape, device=device)

        for n in reversed(range(self.num_steps)):
            x = self.p_sample(model, x, n)
            if n % 100 == 0:
                print(f"  Step {n}...")

        return x

    @torch.no_grad()
    def augment(self, model, x_seed, start_step=500):
        """Generate variation using graph-aware noise."""
        # Add graph-aware noise
        noise = torch.randn_like(x_seed)
        x = self.q_sample(x_seed, start_step, noise)

        # Denoise
        for n in reversed(range(start_step + 1)):
            x = self.p_sample(model, x, n)

        return x


class PolynomialGraphFilter(nn.Module):
    """
    Polynomial graph filter from GAD paper.

    H(L) = sum_{k=0}^{K} theta_k * L^k

    This is the denoiser architecture used in GAD.
    Simpler than message-passing GNN but respects graph structure.
    """

    def __init__(self, num_nodes, in_channels=3, out_channels=3, K=4, hidden_dim=64):
        """
        Args:
            num_nodes: number of vertices
            in_channels: input feature dimension (3 for xyz)
            out_channels: output dimension (3 for xyz noise)
            K: polynomial order (number of hops)
            hidden_dim: hidden layer dimension
        """
        super().__init__()
        self.K = K
        self.num_nodes = num_nodes

        # Learnable polynomial coefficients per layer
        self.theta = nn.ParameterList([
            nn.Parameter(torch.randn(K + 1) * 0.01)
            for _ in range(2)  # 2 layers
        ])

        # Feature transforms
        self.input_proj = nn.Linear(in_channels, hidden_dim)
        self.hidden_proj = nn.Linear(hidden_dim, hidden_dim)
        self.output_proj = nn.Linear(hidden_dim, out_channels)

        # Timestep embedding
        self.time_embed = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

        self.hidden_dim = hidden_dim
        self.L_powers = None  # Precomputed L^k

    def precompute_laplacian_powers(self, L):
        """Precompute L^0, L^1, ..., L^K for efficiency."""
        self.L_powers = [torch.eye(L.shape[0], device=L.device)]
        L_k = L.clone()
        for k in range(1, self.K + 1):
            self.L_powers.append(L_k.clone())
            L_k = L_k @ L

    def graph_filter(self, x, theta):
        """Apply polynomial graph filter: sum_k theta_k * L^k @ x"""
        # x: (V, D)
        out = torch.zeros_like(x)
        for k, L_k in enumerate(self.L_powers):
            out = out + theta[k] * (L_k @ x)
        return out

    def forward(self, x, n, L=None):
        """
        x: (B, V, 3) noisy signal
        n: (B,) noise step
        L: (V, V) Laplacian (optional, uses precomputed if available)
        """
        B, V, C = x.shape

        # Timestep embedding
        half = self.hidden_dim // 2
        freqs = torch.exp(-math.log(10000) * torch.arange(half, device=n.device) / half)
        args = n[:, None].float() * freqs[None, :]
        t_emb = torch.cat([torch.sin(args), torch.cos(args)], dim=-1)
        t_emb = self.time_embed(t_emb)  # (B, D)

        outputs = []
        for b in range(B):
            # Project to hidden dim
            h = self.input_proj(x[b])  # (V, D)

            # Add timestep
            h = h + t_emb[b].unsqueeze(0)

            # Apply polynomial graph filters
            for layer_idx, theta in enumerate(self.theta):
                h_filtered = self.graph_filter(h, theta)
                h = F.silu(self.hidden_proj(h_filtered)) if layer_idx == 0 else h_filtered

            out = self.output_proj(h)  # (V, 3)
            outputs.append(out)

        return torch.stack(outputs)


# =============================================================================
# Dataset
# =============================================================================

class FlagFrameDataset(Dataset):
    """Dataset of individual flag frames."""

    def __init__(self, frames):
        """
        Args:
            frames: (N, V, 3) individual frames
        """
        # Normalize
        self.mean = frames.mean()
        self.std = frames.std() + 1e-8
        frames = (frames - self.mean) / self.std

        self.frames = torch.tensor(frames, dtype=torch.float32)
        print(f"Dataset: {len(self)} frames, {frames.shape[1]} vertices")

    def __len__(self):
        return len(self.frames)

    def __getitem__(self, idx):
        return self.frames[idx]

    def denormalize(self, x):
        return x * self.std + self.mean


# =============================================================================
# Visualization
# =============================================================================

def visualize_frame(frame, cells, title="Frame", save_path=None):
    """Visualize a single frame."""
    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(111, projection='3d')

    ax.plot_trisurf(
        frame[:, 0], frame[:, 1], frame[:, 2],
        triangles=cells,
        cmap='viridis',
        alpha=0.8,
        linewidth=0.1,
        edgecolor='gray'
    )

    ax.set_title(title)
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.view_init(elev=20, azim=45)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150)
    plt.close()


def compare_frames(real, generated, augmented, cells, save_path=None):
    """Compare real, generated, and augmented frames."""
    fig, axes = plt.subplots(1, 3, figsize=(15, 5), subplot_kw={'projection': '3d'})

    titles = ['Real', 'Generated (from noise)', 'Augmented (from real)']
    frames = [real, generated, augmented]
    cmaps = ['Blues', 'Oranges', 'Greens']

    for ax, frame, title, cmap in zip(axes, frames, titles, cmaps):
        ax.plot_trisurf(
            frame[:, 0], frame[:, 1], frame[:, 2],
            triangles=cells,
            cmap=cmap,
            alpha=0.8,
            linewidth=0.1,
            edgecolor='gray'
        )
        ax.set_title(title)
        ax.view_init(elev=20, azim=45)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150)
        print(f"Saved: {save_path}")
    plt.close()


def plot_loss_live(train_losses, val_losses, save_path):
    """Plot training curves and save (called during training)."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    epochs = range(1, len(train_losses) + 1)

    # Linear scale
    ax1.plot(epochs, train_losses, 'b-', label='Train', linewidth=2)
    ax1.plot(epochs, val_losses, 'r-', label='Val', linewidth=2)
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.set_title('Training Curves (Linear)')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Log scale
    ax2.semilogy(epochs, train_losses, 'b-', label='Train', linewidth=2)
    ax2.semilogy(epochs, val_losses, 'r-', label='Val', linewidth=2)
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Loss (log scale)')
    ax2.set_title('Training Curves (Log)')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()


def plot_epoch_progression(progression_frames, epochs_recorded, cells, save_path):
    """
    Create a montage showing generation quality over epochs.

    Args:
        progression_frames: list of (V, 3) numpy arrays, one per recorded epoch
        epochs_recorded: list of epoch numbers
        cells: mesh triangles
        save_path: where to save
    """
    n_frames = len(progression_frames)
    if n_frames == 0:
        return

    # Determine grid size
    cols = min(5, n_frames)
    rows = (n_frames + cols - 1) // cols

    fig = plt.figure(figsize=(4 * cols, 4 * rows))

    for idx, (frame, epoch) in enumerate(zip(progression_frames, epochs_recorded)):
        ax = fig.add_subplot(rows, cols, idx + 1, projection='3d')
        ax.plot_trisurf(
            frame[:, 0], frame[:, 1], frame[:, 2],
            triangles=cells,
            cmap='viridis',
            alpha=0.8,
            linewidth=0.1,
            edgecolor='gray'
        )
        ax.set_title(f'Epoch {epoch}')
        ax.view_init(elev=20, azim=45)
        # Remove axis labels for cleaner look
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        ax.set_zticklabels([])

    plt.suptitle('Generation Quality Over Training (Fixed Seed)', fontsize=14)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    print(f"Saved: {save_path}")
    plt.close()


@torch.no_grad()
def generate_with_seed(model, schedule, shape, seed, device):
    """Generate a sample using a fixed seed for reproducibility."""
    torch.manual_seed(seed)
    if device == 'cuda':
        torch.cuda.manual_seed(seed)

    x = torch.randn(shape, device=device)

    for n in reversed(range(schedule.num_steps)):
        x = schedule.p_sample(model, x, n)

    return x


# =============================================================================
# Training
# =============================================================================

def train_epoch(model, dataloader, optimizer, schedule, device, grad_clip=1.0):
    model.train()
    total_loss = 0
    num_batches = 0

    for batch in dataloader:
        batch = batch.to(device)
        B = batch.shape[0]

        # Random noise steps
        n = torch.randint(0, schedule.num_steps, (B,), device=device)

        # Add noise
        noise = torch.randn_like(batch)
        x_n = schedule.q_sample(batch, n, noise)

        # Predict noise
        predicted = model(x_n, n)

        # Loss
        loss = F.mse_loss(predicted, noise)

        optimizer.zero_grad()
        loss.backward()
        if grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step()

        total_loss += loss.item()
        num_batches += 1

    return total_loss / num_batches


@torch.no_grad()
def evaluate(model, dataloader, schedule, device):
    model.eval()
    total_loss = 0
    num_batches = 0

    for batch in dataloader:
        batch = batch.to(device)
        B = batch.shape[0]

        n = torch.randint(0, schedule.num_steps, (B,), device=device)
        noise = torch.randn_like(batch)
        x_n = schedule.q_sample(batch, n, noise)
        predicted = model(x_n, n)
        loss = F.mse_loss(predicted, noise)

        total_loss += loss.item()
        num_batches += 1

    return total_loss / num_batches


# =============================================================================
# Main
# =============================================================================

def main():
    cfg = Config()

    print("=" * 60)
    print("Graph Signal Diffusion Training")
    print("=" * 60)
    print(f"Device: {cfg.device}")
    print(f"Data: {cfg.data_path}")
    print()

    os.makedirs(cfg.output_dir, exist_ok=True)

    # Load data
    print("Loading data...")
    if not os.path.exists(cfg.data_path):
        print(f"Error: {cfg.data_path} not found!")
        print("Run: python setup_flag_data.py")
        return

    data = np.load(cfg.data_path)
    world_pos = data['world_pos']  # (N, T, V, 3)
    cells = data['cells']
    mesh_pos = data['mesh_pos']

    # Flatten to individual frames
    N, T, V, C = world_pos.shape
    frames = world_pos.reshape(N * T, V, C)  # (N*T, V, 3)
    print(f"  Trajectories: {N}, Frames per trajectory: {T}")
    print(f"  Total frames: {len(frames)}")
    print(f"  Vertices: {V}, Mesh triangles: {len(cells)}")

    # Convert 2D mesh_pos to 3D
    if mesh_pos.shape[1] == 2:
        mesh_pos_3d = np.zeros((mesh_pos.shape[0], 3), dtype=np.float32)
        mesh_pos_3d[:, :2] = mesh_pos
        mesh_pos = mesh_pos_3d

    # Split
    n_train = int(len(frames) * cfg.train_split)
    np.random.seed(42)
    indices = np.random.permutation(len(frames))
    train_idx = indices[:n_train]
    val_idx = indices[n_train:]

    train_dataset = FlagFrameDataset(frames[train_idx])
    val_dataset = FlagFrameDataset(frames[val_idx])

    train_loader = DataLoader(train_dataset, batch_size=cfg.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=cfg.batch_size, shuffle=False)

    print(f"  Train: {len(train_dataset)}, Val: {len(val_dataset)}")

    # Model
    print("\nBuilding model...")
    cells_tensor = torch.tensor(cells, dtype=torch.long)
    mesh_pos_tensor = torch.tensor(mesh_pos, dtype=torch.float32)

    model = GraphSignalDiffusion(
        cells=cells_tensor,
        mesh_pos=mesh_pos_tensor,
        hidden_dim=cfg.hidden_dim,
        num_layers=cfg.num_layers,
        edge_dim=cfg.edge_dim,
    ).to(cfg.device)

    num_params = sum(p.numel() for p in model.parameters())
    print(f"  Parameters: {num_params:,}")

    schedule = DiffusionSchedule(num_steps=cfg.num_steps, device=cfg.device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.learning_rate)

    # Training
    print("\nTraining...")
    print("-" * 60)

    train_losses = []
    val_losses = []
    best_val_loss = float('inf')

    # For epoch progression visualization (fixed seed for consistency)
    progression_seed = 42
    progression_frames = []
    epochs_recorded = []

    for epoch in range(1, cfg.num_epochs + 1):
        start = time.time()

        train_loss = train_epoch(model, train_loader, optimizer, schedule, cfg.device, cfg.grad_clip)
        val_loss = evaluate(model, val_loader, schedule, cfg.device)

        train_losses.append(train_loss)
        val_losses.append(val_loss)

        elapsed = time.time() - start
        print(f"Epoch {epoch:3d}/{cfg.num_epochs} | Train: {train_loss:.4f} | Val: {val_loss:.4f} | {elapsed:.1f}s")

        # Save best
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'val_loss': val_loss,
                'mean': train_dataset.mean,
                'std': train_dataset.std,
            }, os.path.join(cfg.output_dir, 'best_model.pt'))

        # Checkpoint
        if epoch % cfg.save_every == 0:
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_losses': train_losses,
                'val_losses': val_losses,
            }, os.path.join(cfg.output_dir, f'checkpoint_epoch{epoch}.pt'))

        # Update loss plot every 5 epochs (live view of training progress)
        if epoch % 5 == 0 or epoch == 1:
            plot_loss_live(
                train_losses, val_losses,
                save_path=os.path.join(cfg.output_dir, 'training_curves.png')
            )

        # Sample and record for progression
        if epoch % cfg.sample_every == 0:
            print("  Generating samples...")
            model.eval()

            # Generate with fixed seed for progression montage
            generated_tensor = generate_with_seed(
                model, schedule, (1, V, 3), progression_seed, cfg.device
            )
            generated = train_dataset.denormalize(generated_tensor[0].cpu().numpy())

            # Store for progression montage
            progression_frames.append(generated.copy())
            epochs_recorded.append(epoch)

            # Also generate a random sample and augmentation for comparison
            random_generated = schedule.sample(model, (1, V, 3), device=cfg.device)
            random_generated = train_dataset.denormalize(random_generated[0].cpu().numpy())

            real_frame = train_dataset.frames[0:1].to(cfg.device)
            augmented = schedule.augment(model, real_frame, start_step=500)
            augmented = train_dataset.denormalize(augmented[0].cpu().numpy())

            real = train_dataset.denormalize(train_dataset.frames[0].numpy())

            compare_frames(
                real, random_generated, augmented, cells,
                save_path=os.path.join(cfg.output_dir, f'samples_epoch{epoch}.png')
            )

            # Update progression montage
            plot_epoch_progression(
                progression_frames, epochs_recorded, cells,
                save_path=os.path.join(cfg.output_dir, 'epoch_progression.png')
            )

    # Final training curves (both linear and log scale)
    plot_loss_live(
        train_losses, val_losses,
        save_path=os.path.join(cfg.output_dir, 'training_curves.png')
    )

    # Final epoch progression montage
    plot_epoch_progression(
        progression_frames, epochs_recorded, cells,
        save_path=os.path.join(cfg.output_dir, 'epoch_progression.png')
    )

    print("\n" + "=" * 60)
    print("Training complete!")
    print(f"Best val loss: {best_val_loss:.4f}")
    print(f"Output: {cfg.output_dir}")
    print("=" * 60)


if __name__ == "__main__":
    main()
