#!/usr/bin/env python3
"""
CNN-based diffusion for flag meshes using HuggingFace diffusers.

Converts flag mesh vertices (xyz) to images (treating xyz as RGB channels),
then uses the standard U-Net architecture from diffusers for DDPM.

This serves as the "ground truth" baseline to compare against the GNN approach.
"""

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from scipy.interpolate import griddata, LinearNDInterpolator
from scipy.spatial import Delaunay
import time

from diffusers import UNet2DModel, DDPMScheduler


# =============================================================================
# Mesh to Image Conversion
# =============================================================================

def mesh_to_image(vertices, mesh_pos, grid_size=64):
    """
    Convert mesh vertex positions to an image.

    Args:
        vertices: (V, 3) xyz positions
        mesh_pos: (V, 2) rest state UV coordinates
        grid_size: output image size (grid_size x grid_size)

    Returns:
        image: (3, H, W) xyz as RGB channels
    """
    # Normalize mesh_pos to [0, 1]
    uv = mesh_pos.copy()
    uv[:, 0] = (uv[:, 0] - uv[:, 0].min()) / (uv[:, 0].max() - uv[:, 0].min())
    uv[:, 1] = (uv[:, 1] - uv[:, 1].min()) / (uv[:, 1].max() - uv[:, 1].min())

    # Create regular grid
    grid_x, grid_y = np.meshgrid(
        np.linspace(0, 1, grid_size),
        np.linspace(0, 1, grid_size)
    )

    # Interpolate each channel
    image = np.zeros((3, grid_size, grid_size), dtype=np.float32)
    for c in range(3):
        image[c] = griddata(uv, vertices[:, c], (grid_x, grid_y), method='linear', fill_value=0)

    return image


def image_to_mesh(image, mesh_pos, grid_size=64):
    """
    Convert image back to mesh vertex positions via bilinear sampling.

    Args:
        image: (3, H, W) xyz as RGB channels
        mesh_pos: (V, 2) rest state UV coordinates
        grid_size: image size

    Returns:
        vertices: (V, 3) xyz positions
    """
    # Normalize mesh_pos to [0, 1]
    uv = mesh_pos.copy()
    uv[:, 0] = (uv[:, 0] - uv[:, 0].min()) / (uv[:, 0].max() - uv[:, 0].min())
    uv[:, 1] = (uv[:, 1] - uv[:, 1].min()) / (uv[:, 1].max() - uv[:, 1].min())

    V = mesh_pos.shape[0]
    vertices = np.zeros((V, 3), dtype=np.float32)

    for i in range(V):
        # Bilinear interpolation
        x = uv[i, 0] * (grid_size - 1)
        y = uv[i, 1] * (grid_size - 1)

        x0, y0 = int(x), int(y)
        x1, y1 = min(x0 + 1, grid_size - 1), min(y0 + 1, grid_size - 1)

        wx = x - x0
        wy = y - y0

        for c in range(3):
            v00 = image[c, y0, x0]
            v01 = image[c, y0, x1]
            v10 = image[c, y1, x0]
            v11 = image[c, y1, x1]

            vertices[i, c] = (v00 * (1 - wx) * (1 - wy) +
                              v01 * wx * (1 - wy) +
                              v10 * (1 - wx) * wy +
                              v11 * wx * wy)

    return vertices


# =============================================================================
# Dataset
# =============================================================================

class FlagImageDataset(Dataset):
    """Dataset of flag frames as images."""

    def __init__(self, frames, mesh_pos, grid_size=40):
        """
        Args:
            frames: (N, V, 3) flag frames
            mesh_pos: (V, 2) rest state positions
            grid_size: output image size
        """
        from scipy.spatial import cKDTree

        self.mesh_pos = mesh_pos
        self.grid_size = grid_size

        print(f"Converting {len(frames)} frames to {grid_size}x{grid_size} images...")
        print("  Precomputing interpolation weights...")

        # Normalize mesh_pos to [0, 1]
        uv = mesh_pos.copy()
        uv[:, 0] = (uv[:, 0] - uv[:, 0].min()) / (uv[:, 0].max() - uv[:, 0].min())
        uv[:, 1] = (uv[:, 1] - uv[:, 1].min()) / (uv[:, 1].max() - uv[:, 1].min())

        # Create grid points
        grid_x, grid_y = np.meshgrid(
            np.linspace(0, 1, grid_size),
            np.linspace(0, 1, grid_size)
        )
        grid_points = np.stack([grid_x.ravel(), grid_y.ravel()], axis=1)

        # Delaunay triangulation
        tri = Delaunay(uv)
        simplex_indices = tri.find_simplex(grid_points)
        valid_mask = simplex_indices >= 0

        # Barycentric coords for valid points
        simplices = tri.simplices[simplex_indices[valid_mask]]
        transforms = tri.transform[simplex_indices[valid_mask]]
        delta = grid_points[valid_mask] - transforms[:, 2, :]
        bary = np.einsum('ijk,ik->ij', transforms[:, :2, :2], delta)
        bary = np.column_stack([bary, 1 - bary.sum(axis=1)])

        # Nearest neighbor for invalid points
        tree = cKDTree(uv)
        _, nearest_idx = tree.query(grid_points[~valid_mask])

        print(f"  {valid_mask.sum()} valid pixels, {(~valid_mask).sum()} nearest-neighbor")
        print("  Converting frames...")

        # Convert all frames
        N = len(frames)
        images = np.zeros((N, 3, grid_size, grid_size), dtype=np.float32)
        n_grid = grid_size * grid_size
        invalid_indices = np.where(~valid_mask)[0]

        start_time = time.time()
        for i in range(N):
            frame = frames[i]
            for c in range(3):
                img_flat = np.zeros(n_grid, dtype=np.float32)
                img_flat[valid_mask] = np.sum(frame[simplices, c] * bary, axis=1)
                img_flat[invalid_indices] = frame[nearest_idx, c]
                images[i, c] = img_flat.reshape(grid_size, grid_size)

            if (i + 1) % 5000 == 0:
                elapsed = time.time() - start_time
                rate = (i + 1) / elapsed
                print(f"  {i + 1}/{N} ({rate:.0f} frames/sec)")

        # Normalize to [-1, 1]
        self.data_min = images.min()
        self.data_max = images.max()
        images = 2 * (images - self.data_min) / (self.data_max - self.data_min) - 1

        self.images = torch.tensor(images, dtype=torch.float32)
        print(f"Dataset: {len(self)} images, shape {self.images.shape[1:]}")
        print(f"  Normalized to [-1, 1] (raw range: [{self.data_min:.3f}, {self.data_max:.3f}])")

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        return self.images[idx]

    def denormalize(self, x):
        """Convert from [-1, 1] back to original range."""
        return (x + 1) / 2 * (self.data_max - self.data_min) + self.data_min


# =============================================================================
# Training
# =============================================================================

def train_epoch(model, loader, optimizer, noise_scheduler, device, scaler=None):
    """Train for one epoch."""
    model.train()
    total_loss = 0

    for batch in loader:
        batch = batch.to(device)
        B = batch.shape[0]

        # Sample random timesteps
        timesteps = torch.randint(
            0, noise_scheduler.config.num_train_timesteps, (B,),
            device=device
        ).long()

        # Add noise
        noise = torch.randn_like(batch)
        noisy_images = noise_scheduler.add_noise(batch, noise, timesteps)

        # Predict noise
        optimizer.zero_grad(set_to_none=True)

        if scaler:
            with torch.amp.autocast('cuda'):
                noise_pred = model(noisy_images, timesteps, return_dict=False)[0]
                loss = F.mse_loss(noise_pred, noise)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            noise_pred = model(noisy_images, timesteps, return_dict=False)[0]
            loss = F.mse_loss(noise_pred, noise)
            loss.backward()
            optimizer.step()

        total_loss += loss.item()

    return total_loss / len(loader)


@torch.no_grad()
def evaluate(model, loader, noise_scheduler, device):
    """Evaluate on validation set."""
    model.eval()
    total_loss = 0

    for batch in loader:
        batch = batch.to(device)
        B = batch.shape[0]

        timesteps = torch.randint(
            0, noise_scheduler.config.num_train_timesteps, (B,),
            device=device
        ).long()

        noise = torch.randn_like(batch)
        noisy_images = noise_scheduler.add_noise(batch, noise, timesteps)

        noise_pred = model(noisy_images, timesteps, return_dict=False)[0]
        loss = F.mse_loss(noise_pred, noise)
        total_loss += loss.item()

    return total_loss / len(loader)


@torch.no_grad()
def generate_samples(model, noise_scheduler, num_samples, grid_size, device):
    """Generate samples using DDPM sampling."""
    model.eval()

    # Start from pure noise
    samples = torch.randn(num_samples, 3, grid_size, grid_size, device=device)

    # Denoise step by step
    noise_scheduler.set_timesteps(noise_scheduler.config.num_train_timesteps)

    for i, t in enumerate(noise_scheduler.timesteps):
        # Predict noise
        timestep = torch.full((num_samples,), t, device=device, dtype=torch.long)
        noise_pred = model(samples, timestep, return_dict=False)[0]

        # Denoise one step
        samples = noise_scheduler.step(noise_pred, t, samples, return_dict=False)[0]

        if i % 200 == 0:
            print(f"  Step {i}/{len(noise_scheduler.timesteps)}...")

    return samples


# =============================================================================
# Visualization
# =============================================================================

def visualize_comparison(real_images, generated_images, output_path, dataset):
    """Create comparison grid."""
    num_samples = min(6, len(real_images), len(generated_images))

    fig, axes = plt.subplots(2, num_samples, figsize=(18, 6))

    for i in range(num_samples):
        # Real - normalize for visualization
        real = dataset.denormalize(real_images[i]).cpu().numpy()
        real_vis = np.transpose(real, (1, 2, 0))  # (H, W, 3)
        real_vis = (real_vis - real_vis.min()) / (real_vis.max() - real_vis.min() + 1e-8)
        axes[0, i].imshow(real_vis)
        axes[0, i].set_title(f'Real #{i+1}')
        axes[0, i].axis('off')

        # Generated
        gen = dataset.denormalize(generated_images[i]).cpu().numpy()
        gen_vis = np.transpose(gen, (1, 2, 0))
        gen_vis = (gen_vis - gen_vis.min()) / (gen_vis.max() - gen_vis.min() + 1e-8)
        axes[1, i].imshow(gen_vis)
        axes[1, i].set_title(f'Generated #{i+1}')
        axes[1, i].axis('off')

    axes[0, 0].set_ylabel('Real', fontsize=12, fontweight='bold')
    axes[1, 0].set_ylabel('Generated', fontsize=12, fontweight='bold')

    plt.suptitle('HuggingFace DDPM: Flag Images (XYZ as RGB)', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_path}")


def visualize_as_mesh(generated_images, mesh_pos, cells, output_path, dataset):
    """Convert generated images back to meshes and visualize."""
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection

    num_samples = min(6, len(generated_images))
    fig = plt.figure(figsize=(18, 6))

    grid_size = generated_images.shape[-1]

    for i in range(num_samples):
        # Convert image back to mesh
        img = dataset.denormalize(generated_images[i]).cpu().numpy()
        vertices = image_to_mesh(img, mesh_pos, grid_size)

        ax = fig.add_subplot(1, num_samples, i + 1, projection='3d')

        # Plot mesh
        triangles = vertices[cells]
        mesh_coll = Poly3DCollection(triangles, alpha=0.6, facecolor='darkorange',
                                      edgecolor='k', linewidth=0.1)
        ax.add_collection3d(mesh_coll)

        ax.set_xlim(vertices[:, 0].min(), vertices[:, 0].max())
        ax.set_ylim(vertices[:, 1].min(), vertices[:, 1].max())
        ax.set_zlim(vertices[:, 2].min(), vertices[:, 2].max())

        ax.set_title(f'Generated #{i+1}')
        ax.view_init(elev=20, azim=45)

    plt.suptitle('CNN Generated Samples (as Meshes)', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_path}")


# =============================================================================
# Main
# =============================================================================

def main():
    print("=" * 60)
    print("HuggingFace DDPM for Flag Meshes (Ground Truth Baseline)")
    print("=" * 60)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Device: {device}")

    # Config
    grid_size = 40  # Image size (40x40 = 1600 pixels, close to 1579 mesh vertices)
    batch_size = 32
    num_epochs = 100
    learning_rate = 1e-4
    num_train_timesteps = 1000

    output_dir = Path('flag_cnn_output')
    output_dir.mkdir(exist_ok=True)

    # Load data
    print("\nLoading data...")
    data_path = Path('flag_data/flag_test.npz')
    data = np.load(data_path)
    world_pos = data['world_pos']  # (N, T, V, 3)
    mesh_pos = data['mesh_pos']    # (V, 2)
    cells = data['cells']

    N, T, V, C = world_pos.shape
    frames = world_pos.reshape(-1, V, C)  # (N*T, V, 3)
    print(f"  Total frames: {len(frames)}")

    # Split train/val
    n_train = int(0.9 * len(frames))
    train_frames = frames[:n_train]
    val_frames = frames[n_train:]

    # Create datasets
    print("\nCreating datasets...")
    train_dataset = FlagImageDataset(train_frames, mesh_pos, grid_size)
    val_dataset = FlagImageDataset(val_frames, mesh_pos, grid_size)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True,
                              num_workers=0, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False,
                            num_workers=0, pin_memory=True)

    # Model - Standard UNet2DModel from HuggingFace
    print("\nBuilding HuggingFace UNet2DModel...")
    model = UNet2DModel(
        sample_size=grid_size,
        in_channels=3,
        out_channels=3,
        layers_per_block=2,
        block_out_channels=(64, 128, 256, 256),
        down_block_types=(
            "DownBlock2D",
            "DownBlock2D",
            "AttnDownBlock2D",
            "DownBlock2D",
        ),
        up_block_types=(
            "UpBlock2D",
            "AttnUpBlock2D",
            "UpBlock2D",
            "UpBlock2D",
        ),
    ).to(device)

    num_params = sum(p.numel() for p in model.parameters())
    print(f"  Parameters: {num_params:,}")

    # Noise scheduler - Standard DDPM scheduler
    noise_scheduler = DDPMScheduler(
        num_train_timesteps=num_train_timesteps,
        beta_schedule="squaredcos_cap_v2",  # Cosine schedule (improved DDPM)
    )

    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

    # Training
    print("\nTraining...")
    print("-" * 60)

    scaler = torch.amp.GradScaler('cuda') if device == 'cuda' else None
    if scaler:
        print("Using mixed precision (AMP)")

    train_losses = []
    val_losses = []
    best_val_loss = float('inf')

    for epoch in range(1, num_epochs + 1):
        start = time.time()

        train_loss = train_epoch(model, train_loader, optimizer, noise_scheduler, device, scaler)
        val_loss = evaluate(model, val_loader, noise_scheduler, device)

        train_losses.append(train_loss)
        val_losses.append(val_loss)

        elapsed = time.time() - start
        print(f"Epoch {epoch:3d}/{num_epochs} | Train: {train_loss:.4f} | Val: {val_loss:.4f} | {elapsed:.1f}s")

        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_loss': train_loss,
                'val_loss': val_loss,
                'grid_size': grid_size,
                'data_min': train_dataset.data_min,
                'data_max': train_dataset.data_max,
            }, output_dir / 'best_model.pt')

        # Generate samples periodically
        if epoch % 20 == 0 or epoch == 1:
            print("  Generating samples...")
            generated = generate_samples(model, noise_scheduler, 6, grid_size, device)

            # Get random real samples
            indices = np.random.choice(len(train_dataset), 6, replace=False)
            real = torch.stack([train_dataset[i] for i in indices])

            # Visualize as images
            visualize_comparison(real, generated,
                                output_dir / f'samples_epoch{epoch}.png',
                                train_dataset)

            # Visualize as meshes
            visualize_as_mesh(generated, mesh_pos, cells,
                             output_dir / f'meshes_epoch{epoch}.png',
                             train_dataset)

            # Save checkpoint
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_loss': train_loss,
                'val_loss': val_loss,
                'grid_size': grid_size,
                'data_min': train_dataset.data_min,
                'data_max': train_dataset.data_max,
            }, output_dir / f'checkpoint_epoch{epoch}.pt')

    # Plot training curves
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

    ax1.plot(train_losses, label='Train')
    ax1.plot(val_losses, label='Val')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.set_title('Training Curves')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    ax2.semilogy(train_losses, label='Train')
    ax2.semilogy(val_losses, label='Val')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Loss (log)')
    ax2.set_title('Training Curves (Log Scale)')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_dir / 'training_curves.png', dpi=150)
    plt.close()
    print(f"Saved: {output_dir / 'training_curves.png'}")

    print("\n" + "=" * 60)
    print("Training complete!")
    print(f"Best validation loss: {best_val_loss:.4f}")
    print(f"Output saved to: {output_dir}")
    print("=" * 60)


if __name__ == '__main__':
    main()
