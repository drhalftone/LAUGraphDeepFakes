#!/usr/bin/env python3
"""
CNN-based diffusion for flag meshes with CoordConv (positional encoding).

Adds row and column coordinates as extra input channels so the model knows
which pixel it's processing. This allows learning position-specific behavior
like keeping the anchor vertex fixed.

Input channels: 5 (xyz + row + col)
Output channels: 3 (noise prediction for xyz only)
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
    """Convert mesh vertex positions to an image."""
    uv = mesh_pos.copy()
    uv[:, 0] = (uv[:, 0] - uv[:, 0].min()) / (uv[:, 0].max() - uv[:, 0].min())
    uv[:, 1] = (uv[:, 1] - uv[:, 1].min()) / (uv[:, 1].max() - uv[:, 1].min())

    grid_x, grid_y = np.meshgrid(
        np.linspace(0, 1, grid_size),
        np.linspace(0, 1, grid_size)
    )

    image = np.zeros((3, grid_size, grid_size), dtype=np.float32)
    for c in range(3):
        image[c] = griddata(uv, vertices[:, c], (grid_x, grid_y), method='linear', fill_value=0)

    return image


def image_to_mesh(image, mesh_pos, grid_size=64):
    """Convert image back to mesh vertex positions via bilinear sampling."""
    uv = mesh_pos.copy()
    uv[:, 0] = (uv[:, 0] - uv[:, 0].min()) / (uv[:, 0].max() - uv[:, 0].min())
    uv[:, 1] = (uv[:, 1] - uv[:, 1].min()) / (uv[:, 1].max() - uv[:, 1].min())

    V = mesh_pos.shape[0]
    vertices = np.zeros((V, 3), dtype=np.float32)

    for i in range(V):
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
# CoordConv: Create coordinate channels
# =============================================================================

def create_coord_channels(grid_size, device='cpu'):
    """
    Create row and column coordinate channels normalized to [-1, 1].

    Returns:
        coords: (2, H, W) tensor with row and col coordinates
    """
    row_coords = torch.linspace(-1, 1, grid_size).view(1, grid_size, 1).expand(1, grid_size, grid_size)
    col_coords = torch.linspace(-1, 1, grid_size).view(1, 1, grid_size).expand(1, grid_size, grid_size)
    coords = torch.cat([row_coords, col_coords], dim=0)
    return coords.to(device)


# =============================================================================
# Dataset
# =============================================================================

class FlagImageDataset(Dataset):
    """Dataset of flag frames as images."""

    def __init__(self, frames, mesh_pos, grid_size=40):
        from scipy.spatial import cKDTree

        self.mesh_pos = mesh_pos
        self.grid_size = grid_size

        print(f"Converting {len(frames)} frames to {grid_size}x{grid_size} images...")
        print("  Precomputing interpolation weights...")

        uv = mesh_pos.copy()
        uv[:, 0] = (uv[:, 0] - uv[:, 0].min()) / (uv[:, 0].max() - uv[:, 0].min())
        uv[:, 1] = (uv[:, 1] - uv[:, 1].min()) / (uv[:, 1].max() - uv[:, 1].min())

        grid_x, grid_y = np.meshgrid(
            np.linspace(0, 1, grid_size),
            np.linspace(0, 1, grid_size)
        )
        grid_points = np.stack([grid_x.ravel(), grid_y.ravel()], axis=1)

        tri = Delaunay(uv)
        simplex_indices = tri.find_simplex(grid_points)
        valid_mask = simplex_indices >= 0

        simplices = tri.simplices[simplex_indices[valid_mask]]
        transforms = tri.transform[simplex_indices[valid_mask]]
        delta = grid_points[valid_mask] - transforms[:, 2, :]
        bary = np.einsum('ijk,ik->ij', transforms[:, :2, :2], delta)
        bary = np.column_stack([bary, 1 - bary.sum(axis=1)])

        tree = cKDTree(uv)
        _, nearest_idx = tree.query(grid_points[~valid_mask])

        print(f"  {valid_mask.sum()} valid pixels, {(~valid_mask).sum()} nearest-neighbor")
        print("  Converting frames...")

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
        return (x + 1) / 2 * (self.data_max - self.data_min) + self.data_min


# =============================================================================
# Training with CoordConv
# =============================================================================

def train_epoch(model, loader, optimizer, noise_scheduler, coord_channels, device, scaler=None):
    model.train()
    total_loss = 0

    for batch in loader:
        batch = batch.to(device)
        B = batch.shape[0]

        timesteps = torch.randint(0, noise_scheduler.config.num_train_timesteps, (B,), device=device).long()

        noise = torch.randn_like(batch)
        noisy_xyz = noise_scheduler.add_noise(batch, noise, timesteps)

        coords_batch = coord_channels.unsqueeze(0).expand(B, -1, -1, -1)
        noisy_with_coords = torch.cat([noisy_xyz, coords_batch], dim=1)

        optimizer.zero_grad(set_to_none=True)

        if scaler:
            with torch.amp.autocast('cuda'):
                noise_pred = model(noisy_with_coords, timesteps, return_dict=False)[0]
                loss = F.mse_loss(noise_pred, noise)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            noise_pred = model(noisy_with_coords, timesteps, return_dict=False)[0]
            loss = F.mse_loss(noise_pred, noise)
            loss.backward()
            optimizer.step()

        total_loss += loss.item()

    return total_loss / len(loader)


@torch.no_grad()
def evaluate(model, loader, noise_scheduler, coord_channels, device):
    model.eval()
    total_loss = 0

    for batch in loader:
        batch = batch.to(device)
        B = batch.shape[0]

        timesteps = torch.randint(0, noise_scheduler.config.num_train_timesteps, (B,), device=device).long()

        noise = torch.randn_like(batch)
        noisy_xyz = noise_scheduler.add_noise(batch, noise, timesteps)

        coords_batch = coord_channels.unsqueeze(0).expand(B, -1, -1, -1)
        noisy_with_coords = torch.cat([noisy_xyz, coords_batch], dim=1)

        noise_pred = model(noisy_with_coords, timesteps, return_dict=False)[0]
        loss = F.mse_loss(noise_pred, noise)
        total_loss += loss.item()

    return total_loss / len(loader)


@torch.no_grad()
def generate_samples(model, noise_scheduler, coord_channels, num_samples, grid_size, device):
    model.eval()

    samples = torch.randn(num_samples, 3, grid_size, grid_size, device=device)
    noise_scheduler.set_timesteps(noise_scheduler.config.num_train_timesteps)
    coords_batch = coord_channels.unsqueeze(0).expand(num_samples, -1, -1, -1)

    for i, t in enumerate(noise_scheduler.timesteps):
        samples_with_coords = torch.cat([samples, coords_batch], dim=1)
        timestep = torch.full((num_samples,), t, device=device, dtype=torch.long)
        noise_pred = model(samples_with_coords, timestep, return_dict=False)[0]
        samples = noise_scheduler.step(noise_pred, t, samples, return_dict=False)[0]

        if i % 200 == 0:
            print(f"  Step {i}/{len(noise_scheduler.timesteps)}...")

    return samples


# =============================================================================
# Visualization
# =============================================================================

def visualize_as_mesh(generated_images, mesh_pos, cells, output_path, dataset):
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection

    num_samples = min(6, len(generated_images))
    fig = plt.figure(figsize=(18, 6))
    grid_size = generated_images.shape[-1]

    for i in range(num_samples):
        img = dataset.denormalize(generated_images[i]).cpu().numpy()
        vertices = image_to_mesh(img, mesh_pos, grid_size)

        ax = fig.add_subplot(1, num_samples, i + 1, projection='3d')

        triangles = vertices[cells]
        mesh_coll = Poly3DCollection(triangles, alpha=0.6, facecolor='darkorange',
                                      edgecolor='k', linewidth=0.1)
        ax.add_collection3d(mesh_coll)

        ax.scatter([vertices[0, 0]], [vertices[0, 1]], [vertices[0, 2]],
                   c='red', s=100, marker='o', zorder=10)

        ax.set_xlim(vertices[:, 0].min(), vertices[:, 0].max())
        ax.set_ylim(vertices[:, 1].min(), vertices[:, 1].max())
        ax.set_zlim(vertices[:, 2].min(), vertices[:, 2].max())

        ax.set_title(f'Sample #{i+1}')
        ax.view_init(elev=20, azim=45)

    plt.suptitle('CoordConv CNN Generated Samples (anchor in red)', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_path}")


# =============================================================================
# Main
# =============================================================================

def main():
    print("=" * 60)
    print("CoordConv DDPM for Flag Meshes")
    print("  Input: 5 channels (xyz + row + col)")
    print("  Output: 3 channels (noise for xyz)")
    print("=" * 60)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Device: {device}")

    grid_size = 40
    batch_size = 32
    num_epochs = 100
    learning_rate = 1e-4
    num_train_timesteps = 1000

    output_dir = Path('flag_cnn_coordconv_output')
    output_dir.mkdir(exist_ok=True)

    print("\nLoading data...")
    data = np.load('flag_data/flag_test.npz')
    world_pos = data['world_pos']
    mesh_pos = data['mesh_pos']
    cells = data['cells']

    N, T, V, C = world_pos.shape
    frames = world_pos.reshape(-1, V, C)
    print(f"  Total frames: {len(frames)}")

    n_train = int(0.9 * len(frames))
    np.random.seed(42)
    indices = np.random.permutation(len(frames))
    train_frames = frames[indices[:n_train]]
    val_frames = frames[indices[n_train:]]

    print("\nCreating datasets...")
    train_dataset = FlagImageDataset(train_frames, mesh_pos, grid_size)
    val_dataset = FlagImageDataset(val_frames, mesh_pos, grid_size)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=0)

    print("\nCreating coordinate channels...")
    coord_channels = create_coord_channels(grid_size, device)
    print(f"  Shape: {coord_channels.shape}")

    print("\nBuilding CoordConv UNet2DModel...")
    model = UNet2DModel(
        sample_size=grid_size,
        in_channels=5,
        out_channels=3,
        layers_per_block=2,
        block_out_channels=(64, 128, 256, 256),
        down_block_types=("DownBlock2D", "DownBlock2D", "AttnDownBlock2D", "DownBlock2D"),
        up_block_types=("UpBlock2D", "AttnUpBlock2D", "UpBlock2D", "UpBlock2D"),
    ).to(device)

    print(f"  Parameters: {sum(p.numel() for p in model.parameters()):,}")

    noise_scheduler = DDPMScheduler(num_train_timesteps=num_train_timesteps, beta_schedule="squaredcos_cap_v2")
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

    print("\nTraining...")
    scaler = torch.amp.GradScaler('cuda') if device == 'cuda' else None
    best_val_loss = float('inf')

    for epoch in range(1, num_epochs + 1):
        start = time.time()

        train_loss = train_epoch(model, train_loader, optimizer, noise_scheduler, coord_channels, device, scaler)
        val_loss = evaluate(model, val_loader, noise_scheduler, coord_channels, device)

        print(f"Epoch {epoch:3d}/{num_epochs} | Train: {train_loss:.4f} | Val: {val_loss:.4f} | {time.time()-start:.1f}s")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save({
                'epoch': epoch, 'model_state_dict': model.state_dict(),
                'grid_size': grid_size, 'data_min': train_dataset.data_min, 'data_max': train_dataset.data_max,
            }, output_dir / 'best_model.pt')

        if epoch % 20 == 0 or epoch == 1:
            print("  Generating samples...")
            generated = generate_samples(model, noise_scheduler, coord_channels, 6, grid_size, device)
            visualize_as_mesh(generated, mesh_pos, cells, output_dir / f'meshes_epoch{epoch}.png', train_dataset)
            torch.save({'epoch': epoch, 'model_state_dict': model.state_dict(),
                        'grid_size': grid_size, 'data_min': train_dataset.data_min, 'data_max': train_dataset.data_max},
                       output_dir / f'checkpoint_epoch{epoch}.pt')

    print(f"\nTraining complete! Best val loss: {best_val_loss:.4f}")
    print(f"Output: {output_dir}")


if __name__ == '__main__':
    main()
