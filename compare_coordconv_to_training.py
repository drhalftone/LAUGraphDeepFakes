#!/usr/bin/env python3
"""
Compare CoordConv CNN generated samples to nearest training samples.
Shows generated "deep fakes" alongside the closest real training data.
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from pathlib import Path
from scipy.spatial import cKDTree
from diffusers import UNet2DModel, DDPMScheduler


def create_coord_channels(grid_size, device='cpu'):
    """Create row and column coordinate channels normalized to [-1, 1]."""
    row_coords = torch.linspace(-1, 1, grid_size).view(1, grid_size, 1).expand(1, grid_size, grid_size)
    col_coords = torch.linspace(-1, 1, grid_size).view(1, 1, grid_size).expand(1, grid_size, grid_size)
    coords = torch.cat([row_coords, col_coords], dim=0)
    return coords.to(device)


def image_to_mesh(image, mesh_pos, grid_size):
    """Convert image back to mesh vertices via bilinear interpolation."""
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
        wx, wy = x - x0, y - y0

        for c in range(3):
            v00 = image[c, y0, x0]
            v01 = image[c, y0, x1]
            v10 = image[c, y1, x0]
            v11 = image[c, y1, x1]
            vertices[i, c] = (v00 * (1-wx) * (1-wy) + v01 * wx * (1-wy) +
                              v10 * (1-wx) * wy + v11 * wx * wy)
    return vertices


@torch.no_grad()
def generate_coordconv_samples(model_path, num_samples, device='cuda'):
    """Generate samples from CoordConv CNN model."""
    checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    data_min = checkpoint['data_min']
    data_max = checkpoint['data_max']
    grid_size = checkpoint.get('grid_size', 40)

    # Build CoordConv model (5 input channels)
    model = UNet2DModel(
        sample_size=grid_size,
        in_channels=5,  # xyz + row + col
        out_channels=3,
        layers_per_block=2,
        block_out_channels=(64, 128, 256, 256),
        down_block_types=("DownBlock2D", "DownBlock2D", "AttnDownBlock2D", "DownBlock2D"),
        up_block_types=("UpBlock2D", "AttnUpBlock2D", "UpBlock2D", "UpBlock2D"),
    ).to(device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    scheduler = DDPMScheduler(num_train_timesteps=1000, beta_schedule="squaredcos_cap_v2")
    coord_channels = create_coord_channels(grid_size, device)

    print(f"Generating {num_samples} samples from CoordConv model...")
    samples = torch.randn(num_samples, 3, grid_size, grid_size, device=device)
    scheduler.set_timesteps(1000)
    coords_batch = coord_channels.unsqueeze(0).expand(num_samples, -1, -1, -1)

    for i, t in enumerate(scheduler.timesteps):
        samples_with_coords = torch.cat([samples, coords_batch], dim=1)
        timestep = torch.full((num_samples,), t, device=device, dtype=torch.long)
        noise_pred = model(samples_with_coords, timestep, return_dict=False)[0]
        samples = scheduler.step(noise_pred, t, samples, return_dict=False)[0]
        if i % 200 == 0:
            print(f"  Step {i}/1000...")

    # Denormalize
    samples = (samples + 1) / 2 * (data_max - data_min) + data_min
    return samples.cpu().numpy(), grid_size, data_min, data_max


def find_nearest_neighbors(generated_meshes, training_frames):
    """Find nearest training frame for each generated mesh using L2 distance."""
    gen_flat = generated_meshes.reshape(len(generated_meshes), -1)
    train_flat = training_frames.reshape(len(training_frames), -1)

    print(f"Building KD-tree for {len(training_frames)} training frames...")
    tree = cKDTree(train_flat)

    print("Finding nearest neighbors...")
    distances, indices = tree.query(gen_flat, k=1)

    return indices, distances


def plot_comparison(generated, nearest_training, cells, distances, output_path):
    """Create a figure comparing generated samples to nearest training samples."""
    num_samples = len(generated)

    fig = plt.figure(figsize=(16, 4 * ((num_samples + 1) // 2)))

    # Layout: pairs of (generated, nearest) side by side
    pairs_per_row = 2
    num_rows = (num_samples + pairs_per_row - 1) // pairs_per_row

    for i in range(num_samples):
        row = i // pairs_per_row
        col_in_row = i % pairs_per_row

        # Calculate subplot indices (4 columns per row: gen1, near1, gen2, near2)
        base_col = col_in_row * 2

        # Common limits for this pair
        all_verts = np.concatenate([generated[i], nearest_training[i]])
        xlim = (all_verts[:, 0].min() - 0.1, all_verts[:, 0].max() + 0.1)
        ylim = (all_verts[:, 1].min() - 0.1, all_verts[:, 1].max() + 0.1)
        zlim = (all_verts[:, 2].min() - 0.1, all_verts[:, 2].max() + 0.1)

        # Generated sample
        ax1 = fig.add_subplot(num_rows, 4, row * 4 + base_col + 1, projection='3d')
        triangles = generated[i][cells]
        mesh = Poly3DCollection(triangles, alpha=0.7, facecolor='darkorange',
                                edgecolor='k', linewidth=0.1)
        ax1.add_collection3d(mesh)
        ax1.scatter([generated[i][0, 0]], [generated[i][0, 1]], [generated[i][0, 2]],
                   c='red', s=80, marker='o', depthshade=False, zorder=10)
        ax1.set_xlim(xlim); ax1.set_ylim(ylim); ax1.set_zlim(zlim)
        ax1.set_title(f'Generated #{i+1}', fontsize=11, fontweight='bold', color='darkorange')
        ax1.view_init(elev=25, azim=45)
        ax1.set_xticklabels([]); ax1.set_yticklabels([]); ax1.set_zticklabels([])

        # Nearest training sample
        ax2 = fig.add_subplot(num_rows, 4, row * 4 + base_col + 2, projection='3d')
        triangles = nearest_training[i][cells]
        mesh = Poly3DCollection(triangles, alpha=0.7, facecolor='steelblue',
                                edgecolor='k', linewidth=0.1)
        ax2.add_collection3d(mesh)
        ax2.scatter([nearest_training[i][0, 0]], [nearest_training[i][0, 1]], [nearest_training[i][0, 2]],
                   c='red', s=80, marker='o', depthshade=False, zorder=10)
        ax2.set_xlim(xlim); ax2.set_ylim(ylim); ax2.set_zlim(zlim)
        ax2.set_title(f'Nearest Training (d={distances[i]:.2f})', fontsize=11, fontweight='bold', color='steelblue')
        ax2.view_init(elev=25, azim=45)
        ax2.set_xticklabels([]); ax2.set_yticklabels([]); ax2.set_zticklabels([])

    plt.suptitle('CoordConv CNN: Generated "Deep Fakes" vs Nearest Training Data\n(Orange = Generated, Blue = Nearest Real, Red dot = Anchor)',
                 fontsize=14, fontweight='bold')
    plt.tight_layout(rect=[0, 0, 1, 0.93])
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"\nSaved figure: {output_path}")


def main():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Device: {device}")

    model_path = Path('flag_cnn_coordconv_output/best_model.pt')
    output_path = Path('coordconv_generated_vs_nearest.png')
    num_samples = 6

    # Load training data
    print("\nLoading training data...")
    data = np.load('flag_data/flag_test.npz')
    world_pos = data['world_pos']  # (N, T, V, 3)
    cells = data['cells']
    mesh_pos = data['mesh_pos']

    N, T, V, C = world_pos.shape
    training_frames = world_pos.reshape(-1, V, C)
    print(f"  {len(training_frames)} training frames")

    # Generate samples
    generated_images, grid_size, data_min, data_max = generate_coordconv_samples(
        model_path, num_samples, device
    )

    # Convert images to meshes
    print("\nConverting generated images to meshes...")
    generated_meshes = np.array([
        image_to_mesh(img, mesh_pos, grid_size) for img in generated_images
    ])

    # Find nearest neighbors
    indices, distances = find_nearest_neighbors(generated_meshes, training_frames)
    nearest_training = training_frames[indices]

    print(f"\nNearest neighbor distances: {distances}")
    print(f"Mean distance: {distances.mean():.4f}")
    print(f"Min distance:  {distances.min():.4f}")
    print(f"Max distance:  {distances.max():.4f}")

    # Create comparison figure
    plot_comparison(generated_meshes, nearest_training, cells, distances, output_path)

    # Statistics
    print("\n" + "=" * 60)
    print("Statistics")
    print("=" * 60)
    print(f"{'Metric':<15} {'Generated':<20} {'Nearest Training':<20}")
    print("-" * 55)
    print(f"{'Mean':<15} {generated_meshes.mean():<20.4f} {nearest_training.mean():<20.4f}")
    print(f"{'Std':<15} {generated_meshes.std():<20.4f} {nearest_training.std():<20.4f}")
    print(f"{'Min':<15} {generated_meshes.min():<20.4f} {nearest_training.min():<20.4f}")
    print(f"{'Max':<15} {generated_meshes.max():<20.4f} {nearest_training.max():<20.4f}")


if __name__ == '__main__':
    main()
