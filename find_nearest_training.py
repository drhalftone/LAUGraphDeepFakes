#!/usr/bin/env python3
"""
Find nearest training samples for generated meshes.
Compares generated samples to training data and shows side-by-side.
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from pathlib import Path
from scipy.spatial import cKDTree
from scipy.interpolate import griddata
from scipy.spatial import Delaunay
import argparse


def load_cnn_checkpoint(checkpoint_path, device='cuda'):
    """Load CNN checkpoint and get normalization params."""
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    return checkpoint.get('data_min'), checkpoint.get('data_max'), checkpoint.get('grid_size', 40)


def image_to_mesh(image, mesh_pos, grid_size):
    """Convert image back to mesh vertices."""
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


def generate_cnn_samples(model_path, num_samples, device='cuda'):
    """Generate samples from CNN model."""
    from diffusers import UNet2DModel, DDPMScheduler

    # Load checkpoint
    checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    data_min = checkpoint['data_min']
    data_max = checkpoint['data_max']
    grid_size = checkpoint.get('grid_size', 40)

    # Rebuild model
    model = UNet2DModel(
        sample_size=grid_size,
        in_channels=3,
        out_channels=3,
        layers_per_block=2,
        block_out_channels=(64, 128, 256, 256),
        down_block_types=("DownBlock2D", "DownBlock2D", "AttnDownBlock2D", "DownBlock2D"),
        up_block_types=("UpBlock2D", "AttnUpBlock2D", "UpBlock2D", "UpBlock2D"),
    ).to(device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    # Scheduler
    scheduler = DDPMScheduler(num_train_timesteps=1000, beta_schedule="squaredcos_cap_v2")

    # Generate
    print(f"Generating {num_samples} samples...")
    samples = torch.randn(num_samples, 3, grid_size, grid_size, device=device)
    scheduler.set_timesteps(1000)

    with torch.no_grad():
        for i, t in enumerate(scheduler.timesteps):
            timestep = torch.full((num_samples,), t, device=device, dtype=torch.long)
            noise_pred = model(samples, timestep, return_dict=False)[0]
            samples = scheduler.step(noise_pred, t, samples, return_dict=False)[0]
            if i % 200 == 0:
                print(f"  Step {i}/1000...")

    # Denormalize
    samples = (samples + 1) / 2 * (data_max - data_min) + data_min
    return samples.cpu().numpy(), grid_size


def find_nearest_neighbors(generated_meshes, training_frames):
    """
    Find nearest training frame for each generated mesh.
    Uses L2 distance on flattened vertex positions.
    """
    # Flatten for distance computation
    gen_flat = generated_meshes.reshape(len(generated_meshes), -1)
    train_flat = training_frames.reshape(len(training_frames), -1)

    print(f"Building KD-tree for {len(training_frames)} training frames...")
    tree = cKDTree(train_flat)

    print("Finding nearest neighbors...")
    distances, indices = tree.query(gen_flat, k=1)

    return indices, distances


def plot_comparison(generated, nearest_training, cells, distances, output_path):
    """Plot generated vs nearest training side by side."""
    num_samples = len(generated)

    fig = plt.figure(figsize=(20, 6 * ((num_samples + 2) // 3)))

    for i in range(num_samples):
        # Generated
        ax1 = fig.add_subplot(num_samples, 2, i * 2 + 1, projection='3d')
        triangles = generated[i][cells]
        mesh = Poly3DCollection(triangles, alpha=0.6, facecolor='darkorange',
                                edgecolor='k', linewidth=0.1)
        ax1.add_collection3d(mesh)

        # Set same limits for both
        all_verts = np.concatenate([generated[i], nearest_training[i]])
        for ax in [ax1]:
            ax.set_xlim(all_verts[:, 0].min(), all_verts[:, 0].max())
            ax.set_ylim(all_verts[:, 1].min(), all_verts[:, 1].max())
            ax.set_zlim(all_verts[:, 2].min(), all_verts[:, 2].max())

        ax1.set_title(f'Generated #{i+1}', fontsize=12, fontweight='bold')
        ax1.view_init(elev=20, azim=45)

        # Nearest training
        ax2 = fig.add_subplot(num_samples, 2, i * 2 + 2, projection='3d')
        triangles = nearest_training[i][cells]
        mesh = Poly3DCollection(triangles, alpha=0.6, facecolor='steelblue',
                                edgecolor='k', linewidth=0.1)
        ax2.add_collection3d(mesh)
        ax2.set_xlim(all_verts[:, 0].min(), all_verts[:, 0].max())
        ax2.set_ylim(all_verts[:, 1].min(), all_verts[:, 1].max())
        ax2.set_zlim(all_verts[:, 2].min(), all_verts[:, 2].max())
        ax2.set_title(f'Nearest Training (dist={distances[i]:.2f})', fontsize=12, fontweight='bold')
        ax2.view_init(elev=20, azim=45)

    plt.suptitle('Generated (Orange) vs Nearest Training Sample (Blue)', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', default='flag_cnn_output/best_model.pt', help='Model checkpoint')
    parser.add_argument('--num-samples', type=int, default=6, help='Number of samples')
    parser.add_argument('--output', default='generated_vs_nearest.png', help='Output path')
    args = parser.parse_args()

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Device: {device}")

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
    generated_images, grid_size = generate_cnn_samples(args.model, args.num_samples, device)

    # Convert images to meshes
    print("\nConverting generated images to meshes...")
    generated_meshes = []
    for img in generated_images:
        mesh = image_to_mesh(img, mesh_pos, grid_size)
        generated_meshes.append(mesh)
    generated_meshes = np.array(generated_meshes)

    # Find nearest neighbors
    indices, distances = find_nearest_neighbors(generated_meshes, training_frames)
    nearest_training = training_frames[indices]

    print(f"\nNearest neighbor distances: {distances}")
    print(f"Mean distance: {distances.mean():.4f}")

    # Plot comparison
    output_path = Path(args.output)
    plot_comparison(generated_meshes, nearest_training, cells, distances, output_path)

    # Also compute some statistics
    print("\n" + "=" * 60)
    print("Statistics")
    print("=" * 60)
    print(f"Generated mesh stats:")
    print(f"  Mean: {generated_meshes.mean():.4f}")
    print(f"  Std:  {generated_meshes.std():.4f}")
    print(f"  Range: [{generated_meshes.min():.4f}, {generated_meshes.max():.4f}]")
    print(f"\nNearest training stats:")
    print(f"  Mean: {nearest_training.mean():.4f}")
    print(f"  Std:  {nearest_training.std():.4f}")
    print(f"  Range: [{nearest_training.min():.4f}, {nearest_training.max():.4f}]")


if __name__ == '__main__':
    main()
