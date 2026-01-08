#!/usr/bin/env python3
"""
Compare real training data vs generated samples from noise.
Creates a grid showing multiple examples of each.
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from pathlib import Path
import sys

# Add parent to path for imports
sys.path.insert(0, str(Path(__file__).parent))

from train_flag_diffusion import (
    GraphSignalDiffusion,
    DiffusionSchedule,
    FlagFrameDataset,
)


def load_model_and_data(checkpoint_path, data_path, device='cuda'):
    """Load trained model and dataset."""
    # Load data
    data = np.load(data_path)
    world_pos = data['world_pos']  # (N, T, V, 3)
    cells = data['cells']
    mesh_pos = data['mesh_pos']

    # Flatten to frames
    N, T, V, C = world_pos.shape
    frames = world_pos.reshape(-1, V, C)

    # Convert 2D mesh_pos to 3D (same as training)
    if mesh_pos.shape[1] == 2:
        mesh_pos_3d = np.zeros((mesh_pos.shape[0], 3), dtype=np.float32)
        mesh_pos_3d[:, :2] = mesh_pos
        mesh_pos = mesh_pos_3d

    # Create dataset (handles normalization)
    dataset = FlagFrameDataset(frames)

    # Load model with same config as training
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)

    cells_tensor = torch.tensor(cells, dtype=torch.long)
    mesh_pos_tensor = torch.tensor(mesh_pos, dtype=torch.float32)

    model = GraphSignalDiffusion(
        cells=cells_tensor,
        mesh_pos=mesh_pos_tensor,
        hidden_dim=128,
        num_layers=4,
        edge_dim=4,
        batch_size=32,  # Must match training batch size for checkpoint loading
    ).to(device)

    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    return model, dataset, cells, mesh_pos


@torch.no_grad()
def generate_from_noise(model, schedule, num_samples, num_vertices, device='cuda'):
    """Generate samples starting from pure noise (step 999)."""
    # Start from pure noise
    x = torch.randn(num_samples, num_vertices, 3, device=device)

    # Denoise from step 999 down to 0 using schedule's p_sample method
    for n in reversed(range(schedule.num_steps)):
        x = schedule.p_sample(model, x, n)

        if n % 200 == 0:
            print(f"  Step {n}...")

    return x


def plot_mesh(ax, vertices, cells, color='blue', alpha=0.7, limits=None):
    """Plot a triangular mesh."""
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection

    # Create triangles
    triangles = vertices[cells]

    # Plot
    mesh = Poly3DCollection(triangles, alpha=alpha, facecolor=color, edgecolor='k', linewidth=0.1)
    ax.add_collection3d(mesh)

    # Set limits (use global limits if provided, else per-mesh limits)
    if limits is not None:
        ax.set_xlim(limits[0])
        ax.set_ylim(limits[1])
        ax.set_zlim(limits[2])
    else:
        ax.set_xlim(vertices[:, 0].min(), vertices[:, 0].max())
        ax.set_ylim(vertices[:, 1].min(), vertices[:, 1].max())
        ax.set_zlim(vertices[:, 2].min(), vertices[:, 2].max())

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')


def main():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Device: {device}")

    # Paths
    checkpoint_path = Path('flag_diffusion_output/best_model.pt')
    data_path = Path('flag_data/flag_test.npz')
    output_path = Path('flag_diffusion_output/real_vs_generated_grid.png')

    if not checkpoint_path.exists():
        # Try latest checkpoint
        checkpoints = list(Path('flag_diffusion_output').glob('checkpoint_epoch*.pt'))
        if checkpoints:
            checkpoint_path = max(checkpoints, key=lambda p: int(p.stem.split('epoch')[1]))
            print(f"Using checkpoint: {checkpoint_path}")
        else:
            print("No checkpoint found! Train the model first.")
            return

    print(f"Loading model from {checkpoint_path}...")
    model, dataset, cells, mesh_pos = load_model_and_data(checkpoint_path, data_path, device)

    num_vertices = dataset.frames.shape[1]
    schedule = DiffusionSchedule(num_steps=1000, device=device)

    # Number of samples to show
    num_samples = 6

    # Get random real samples
    print(f"\nSelecting {num_samples} random real samples...")
    indices = np.random.choice(len(dataset), num_samples, replace=False)
    real_samples = torch.stack([dataset[i] for i in indices])

    # Generate from noise
    print(f"\nGenerating {num_samples} samples from noise...")
    generated_samples = generate_from_noise(model, schedule, num_samples, num_vertices, device)

    # Denormalize
    real_samples_denorm = dataset.denormalize(real_samples).cpu().numpy()
    generated_samples_denorm = dataset.denormalize(generated_samples).cpu().numpy()

    # Compute global axis limits across all samples
    all_samples = np.concatenate([real_samples_denorm, generated_samples_denorm], axis=0)
    global_limits = [
        (all_samples[:, :, 0].min(), all_samples[:, :, 0].max()),  # X
        (all_samples[:, :, 1].min(), all_samples[:, :, 1].max()),  # Y
        (all_samples[:, :, 2].min(), all_samples[:, :, 2].max()),  # Z
    ]
    print(f"Global axis limits: X={global_limits[0]}, Y={global_limits[1]}, Z={global_limits[2]}")

    # Create comparison grid
    print("\nCreating visualization...")
    fig = plt.figure(figsize=(20, 12))

    # Top row: Real samples
    for i in range(num_samples):
        ax = fig.add_subplot(2, num_samples, i + 1, projection='3d')
        plot_mesh(ax, real_samples_denorm[i], cells, color='steelblue', alpha=0.6, limits=global_limits)
        if i == 0:
            ax.set_title(f'Real #{i+1}', fontsize=12, fontweight='bold')
        else:
            ax.set_title(f'Real #{i+1}', fontsize=12)
        ax.view_init(elev=20, azim=45)

    # Bottom row: Generated samples
    for i in range(num_samples):
        ax = fig.add_subplot(2, num_samples, num_samples + i + 1, projection='3d')
        plot_mesh(ax, generated_samples_denorm[i], cells, color='darkorange', alpha=0.6, limits=global_limits)
        if i == 0:
            ax.set_title(f'Generated #{i+1}', fontsize=12, fontweight='bold')
        else:
            ax.set_title(f'Generated #{i+1}', fontsize=12)
        ax.view_init(elev=20, azim=45)

    # Add row labels
    fig.text(0.02, 0.75, 'REAL\n(Training Data)', ha='left', va='center', fontsize=14, fontweight='bold', color='steelblue')
    fig.text(0.02, 0.25, 'GENERATED\n(From Noise)', ha='left', va='center', fontsize=14, fontweight='bold', color='darkorange')

    plt.suptitle('Real Training Data vs Generated Samples (from pure noise)', fontsize=16, fontweight='bold')
    plt.tight_layout(rect=[0.05, 0, 1, 0.95])

    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"\nSaved: {output_path}")

    # Also print statistics
    print("\n" + "="*60)
    print("Statistics Comparison")
    print("="*60)
    print(f"{'Metric':<20} {'Real':<20} {'Generated':<20}")
    print("-"*60)
    print(f"{'Mean':<20} {real_samples_denorm.mean():<20.4f} {generated_samples_denorm.mean():<20.4f}")
    print(f"{'Std':<20} {real_samples_denorm.std():<20.4f} {generated_samples_denorm.std():<20.4f}")
    print(f"{'Min':<20} {real_samples_denorm.min():<20.4f} {generated_samples_denorm.min():<20.4f}")
    print(f"{'Max':<20} {real_samples_denorm.max():<20.4f} {generated_samples_denorm.max():<20.4f}")


if __name__ == '__main__':
    main()
