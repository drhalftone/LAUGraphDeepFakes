#!/usr/bin/env python3
"""
Image-based diffusion model for mesh generation (Ground Truth Baseline).

This implements the established approach: treat the mesh as an image and use
standard image diffusion with a U-Net architecture from Hugging Face diffusers.

The GNN approach (train_flag_diffusion.py) aims to match this baseline.

Usage:
    python train_image_diffusion.py

Requirements:
    pip install diffusers accelerate
"""

import os
import math
import time
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

from diffusers import UNet2DModel, DDPMScheduler


# =============================================================================
# Configuration
# =============================================================================

class Config:
    # Data
    data_path = "flag_data/flag_test.npz"
    train_split = 0.9

    # Grid dimensions (must be determined from data)
    # The mesh should be H x W = V vertices
    grid_height = None  # Set after loading data
    grid_width = None

    # Model (UNet2DModel from diffusers)
    base_channels = 128
    layers_per_block = 2
    attention_resolutions = (16, 8)  # Add attention at these resolutions

    # Diffusion
    num_steps = 1000
    beta_schedule = "squaredcos_cap_v2"  # Cosine schedule (Improved DDPM)

    # Training
    batch_size = 32
    num_epochs = 100
    learning_rate = 1e-4
    grad_clip = 1.0

    # Output
    output_dir = "image_diffusion_output"
    save_every = 20
    sample_every = 20

    # Device
    device = "cuda" if torch.cuda.is_available() else "cpu"


# =============================================================================
# Dataset
# =============================================================================

class MeshImageDataset(Dataset):
    """Dataset treating mesh frames as images."""

    def __init__(self, frames, H, W):
        """
        Args:
            frames: (N, V, 3) mesh frames where V = H * W
            H, W: grid dimensions
        """
        self.H = H
        self.W = W

        # Verify dimensions
        N, V, C = frames.shape
        assert V == H * W, f"V={V} != H*W={H*W}"
        assert C == 3, f"Expected 3 coordinates, got {C}"

        # Reshape to images: (N, V, 3) -> (N, H, W, 3)
        images = frames.reshape(N, H, W, 3)

        # Normalize to [-1, 1] (standard for diffusion)
        self.min_val = images.min()
        self.max_val = images.max()
        images = 2 * (images - self.min_val) / (self.max_val - self.min_val) - 1

        # Convert to channels-first for diffusers: (N, H, W, 3) -> (N, 3, H, W)
        self.images = torch.tensor(images, dtype=torch.float32).permute(0, 3, 1, 2)

        print(f"Dataset: {len(self)} images of size {H}x{W}")
        print(f"  Normalized to [-1, 1] (raw range: [{self.min_val:.3f}, {self.max_val:.3f}])")

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        return self.images[idx]

    def denormalize(self, x):
        """Convert from [-1, 1] back to original range."""
        return (x + 1) / 2 * (self.max_val - self.min_val) + self.min_val


# =============================================================================
# Model Creation
# =============================================================================

def create_model(H, W, cfg):
    """Create UNet2DModel from diffusers."""

    # Determine block types based on resolution
    # Add attention at lower resolutions
    num_down = int(math.log2(min(H, W))) - 2  # Number of downsampling levels
    num_down = min(num_down, 4)  # Cap at 4 levels

    down_block_types = []
    up_block_types = []

    channel_mult = [1, 2, 4, 4][:num_down]
    block_out_channels = tuple(cfg.base_channels * m for m in channel_mult)

    for i in range(num_down):
        res = min(H, W) // (2 ** i)
        if res <= 16:
            down_block_types.append("AttnDownBlock2D")
            up_block_types.insert(0, "AttnUpBlock2D")
        else:
            down_block_types.append("DownBlock2D")
            up_block_types.insert(0, "UpBlock2D")

    model = UNet2DModel(
        sample_size=min(H, W),
        in_channels=3,
        out_channels=3,
        layers_per_block=cfg.layers_per_block,
        block_out_channels=block_out_channels,
        down_block_types=tuple(down_block_types),
        up_block_types=tuple(up_block_types),
    )

    return model


def create_scheduler(cfg):
    """Create DDPM scheduler from diffusers."""
    return DDPMScheduler(
        num_train_timesteps=cfg.num_steps,
        beta_schedule=cfg.beta_schedule,
    )


# =============================================================================
# Training
# =============================================================================

def train_epoch(model, dataloader, optimizer, scheduler, device, grad_clip=1.0, scaler=None):
    """Train for one epoch."""
    model.train()
    total_loss = 0
    num_batches = 0
    use_amp = scaler is not None

    for batch in dataloader:
        batch = batch.to(device)
        B = batch.shape[0]

        # Sample random timesteps
        timesteps = torch.randint(
            0, scheduler.config.num_train_timesteps, (B,),
            device=device, dtype=torch.long
        )

        # Add noise using scheduler
        noise = torch.randn_like(batch)
        noisy = scheduler.add_noise(batch, noise, timesteps)

        optimizer.zero_grad()

        # Forward pass
        if use_amp:
            with torch.amp.autocast('cuda'):
                noise_pred = model(noisy, timesteps).sample
                loss = F.mse_loss(noise_pred, noise)
            scaler.scale(loss).backward()
            if grad_clip > 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            scaler.step(optimizer)
            scaler.update()
        else:
            noise_pred = model(noisy, timesteps).sample
            loss = F.mse_loss(noise_pred, noise)
            loss.backward()
            if grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            optimizer.step()

        total_loss += loss.item()
        num_batches += 1

    return total_loss / num_batches


@torch.no_grad()
def evaluate(model, dataloader, scheduler, device):
    """Evaluate on validation set."""
    model.eval()
    total_loss = 0
    num_batches = 0

    for batch in dataloader:
        batch = batch.to(device)
        B = batch.shape[0]

        timesteps = torch.randint(
            0, scheduler.config.num_train_timesteps, (B,),
            device=device, dtype=torch.long
        )

        noise = torch.randn_like(batch)
        noisy = scheduler.add_noise(batch, noise, timesteps)

        noise_pred = model(noisy, timesteps).sample
        loss = F.mse_loss(noise_pred, noise)

        total_loss += loss.item()
        num_batches += 1

    return total_loss / num_batches


# =============================================================================
# Generation
# =============================================================================

@torch.no_grad()
def generate(model, scheduler, H, W, device, num_samples=1, seed=None):
    """Generate new mesh images from noise."""
    if seed is not None:
        torch.manual_seed(seed)
        if device == 'cuda':
            torch.cuda.manual_seed(seed)

    # Start from pure noise
    sample = torch.randn(num_samples, 3, H, W, device=device)

    # Set timesteps for generation
    scheduler.set_timesteps(scheduler.config.num_train_timesteps)

    # Iterative denoising
    for t in scheduler.timesteps:
        noise_pred = model(sample, t).sample
        sample = scheduler.step(noise_pred, t, sample).prev_sample

    return sample


@torch.no_grad()
def augment(model, scheduler, x_real, start_step=500, seed=None):
    """Generate variation of real sample via partial denoising."""
    if seed is not None:
        torch.manual_seed(seed)

    # Add noise to intermediate level
    noise = torch.randn_like(x_real)
    timestep = torch.tensor([start_step], device=x_real.device)
    x_noisy = scheduler.add_noise(x_real, noise, timestep)

    # Denoise from start_step to 0
    scheduler.set_timesteps(scheduler.config.num_train_timesteps)
    timesteps = [t for t in scheduler.timesteps if t <= start_step]

    sample = x_noisy
    for t in timesteps:
        noise_pred = model(sample, t).sample
        sample = scheduler.step(noise_pred, t, sample).prev_sample

    return sample


# =============================================================================
# Visualization
# =============================================================================

def visualize_mesh_image(image, H, W, cells, title="Mesh", save_path=None):
    """Visualize a mesh image as 3D surface."""
    # image: (3, H, W) -> positions (H*W, 3)
    positions = image.permute(1, 2, 0).reshape(-1, 3).cpu().numpy()

    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(111, projection='3d')

    ax.plot_trisurf(
        positions[:, 0], positions[:, 1], positions[:, 2],
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


def compare_samples(real, generated, augmented, H, W, cells, save_path=None):
    """Compare real, generated, and augmented samples."""
    fig, axes = plt.subplots(1, 3, figsize=(15, 5), subplot_kw={'projection': '3d'})

    titles = ['Real', 'Generated (from noise)', 'Augmented (from real)']
    images = [real, generated, augmented]
    cmaps = ['Blues', 'Oranges', 'Greens']

    for ax, img, title, cmap in zip(axes, images, titles, cmaps):
        positions = img.permute(1, 2, 0).reshape(-1, 3).cpu().numpy()
        ax.plot_trisurf(
            positions[:, 0], positions[:, 1], positions[:, 2],
            triangles=cells,
            cmap=cmap,
            alpha=0.8,
            linewidth=0.1,
            edgecolor='gray'
        )
        ax.set_title(title)
        ax.view_init(elev=20, azim=45)

    plt.suptitle('Image Diffusion Baseline (U-Net)', fontsize=14)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150)
        print(f"Saved: {save_path}")
    plt.close()


def plot_training_curves(train_losses, val_losses, save_path):
    """Plot training curves."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    epochs = range(1, len(train_losses) + 1)

    ax1.plot(epochs, train_losses, 'b-', label='Train', linewidth=2)
    ax1.plot(epochs, val_losses, 'r-', label='Val', linewidth=2)
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.set_title('Training Curves (Linear)')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

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


# =============================================================================
# Main
# =============================================================================

def main():
    cfg = Config()

    print("=" * 60)
    print("Image Diffusion Training (Ground Truth Baseline)")
    print("Using Hugging Face diffusers (same as Stable Diffusion)")
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

    # Determine grid dimensions
    # Assume square grid (or find factors of V)
    sqrt_v = int(math.sqrt(V))
    if sqrt_v * sqrt_v == V:
        H = W = sqrt_v
    else:
        # Try to find factors
        for h in range(int(math.sqrt(V)), 0, -1):
            if V % h == 0:
                H = h
                W = V // h
                break
        else:
            raise ValueError(f"Cannot determine grid dimensions for V={V}")

    print(f"  Grid dimensions: {H} x {W}")
    cfg.grid_height = H
    cfg.grid_width = W

    # Split
    n_train = int(len(frames) * cfg.train_split)
    np.random.seed(42)
    indices = np.random.permutation(len(frames))
    train_idx = indices[:n_train]
    val_idx = indices[n_train:]

    train_dataset = MeshImageDataset(frames[train_idx], H, W)
    val_dataset = MeshImageDataset(frames[val_idx], H, W)

    train_loader = DataLoader(
        train_dataset, batch_size=cfg.batch_size, shuffle=True,
        num_workers=4, pin_memory=True, persistent_workers=True
    )
    val_loader = DataLoader(
        val_dataset, batch_size=cfg.batch_size, shuffle=False,
        num_workers=2, pin_memory=True, persistent_workers=True
    )

    print(f"  Train: {len(train_dataset)}, Val: {len(val_dataset)}")

    # Create model and scheduler
    print("\nBuilding model (diffusers UNet2DModel)...")
    model = create_model(H, W, cfg).to(cfg.device)
    scheduler = create_scheduler(cfg)

    num_params = sum(p.numel() for p in model.parameters())
    print(f"  Parameters: {num_params:,}")
    print(f"  Diffusion steps: {cfg.num_steps}")
    print(f"  Beta schedule: {cfg.beta_schedule}")

    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.learning_rate)

    # Training
    print("\nTraining...")
    print("-" * 60)

    scaler = torch.amp.GradScaler('cuda') if cfg.device == 'cuda' else None
    if scaler:
        print("Using mixed precision (AMP)")

    train_losses = []
    val_losses = []
    best_val_loss = float('inf')

    for epoch in range(1, cfg.num_epochs + 1):
        start = time.time()

        train_loss = train_epoch(model, train_loader, optimizer, scheduler, cfg.device, cfg.grad_clip, scaler)
        val_loss = evaluate(model, val_loader, scheduler, cfg.device)

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
                'H': H,
                'W': W,
                'data_min': train_dataset.min_val,
                'data_max': train_dataset.max_val,
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

        # Sample and visualize
        if epoch % cfg.sample_every == 0:
            print("  Generating samples...")
            model.eval()

            # Generate from noise
            generated = generate(model, scheduler, H, W, cfg.device, num_samples=1, seed=42)
            generated = train_dataset.denormalize(generated[0])

            # Augment from real
            real = train_dataset.images[0:1].to(cfg.device)
            augmented = augment(model, scheduler, real, start_step=500)
            augmented = train_dataset.denormalize(augmented[0])

            real_denorm = train_dataset.denormalize(train_dataset.images[0])

            compare_samples(
                real_denorm, generated, augmented, H, W, cells,
                save_path=os.path.join(cfg.output_dir, f'samples_epoch{epoch}.png')
            )

        # Plot losses
        if epoch % 5 == 0 or epoch == 1:
            plot_training_curves(
                train_losses, val_losses,
                save_path=os.path.join(cfg.output_dir, 'training_curves.png')
            )

    # Final plots
    plot_training_curves(
        train_losses, val_losses,
        save_path=os.path.join(cfg.output_dir, 'training_curves.png')
    )

    print("\n" + "=" * 60)
    print("Training complete!")
    print(f"Best val loss: {best_val_loss:.4f}")
    print(f"Output: {cfg.output_dir}")
    print("=" * 60)


if __name__ == "__main__":
    main()
