#!/usr/bin/env python3
"""
Load DeepMind's flag_simple dataset and convert to numpy.

This script loads the TFRecord data from MeshGraphNets and converts it
to numpy arrays for easier manipulation with PyTorch.
"""

import tensorflow as tf
import numpy as np
import json
from pathlib import Path
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


def load_metadata(data_dir='flag_data'):
    """Load dataset metadata."""
    with open(Path(data_dir) / 'meta.json') as f:
        return json.load(f)


def parse_tfrecord(serialized, meta):
    """Parse a single TFRecord example."""
    feature_spec = {k: tf.io.VarLenFeature(tf.string) for k in meta['features'].keys()}
    parsed = tf.io.parse_single_example(serialized, feature_spec)

    result = {}
    for key, field in meta['features'].items():
        data = tf.sparse.to_dense(parsed[key])
        dtype = getattr(tf, field['dtype'])
        data = tf.io.decode_raw(data[0], dtype)
        data = tf.reshape(data, field['shape'])
        result[key] = data
    return result


def load_trajectories(tfrecord_path, meta, max_trajectories=None):
    """Load trajectories from a TFRecord file."""
    dataset = tf.data.TFRecordDataset(tfrecord_path)

    trajectories = []
    mesh_info = None

    for i, raw_record in enumerate(dataset):
        if max_trajectories and i >= max_trajectories:
            break

        example = parse_tfrecord(raw_record, meta)

        # Store mesh info (same for all trajectories)
        if mesh_info is None:
            mesh_info = {
                'cells': example['cells'].numpy().squeeze(),      # (3028, 3)
                'mesh_pos': example['mesh_pos'].numpy().squeeze(), # (1579, 2)
            }

        # Store trajectory
        trajectories.append({
            'world_pos': example['world_pos'].numpy(),  # (401, 1579, 3)
            'node_type': example['node_type'].numpy(),  # (401, 1579, 1)
        })

        if (i + 1) % 10 == 0:
            print(f"  Loaded {i + 1} trajectories...")

    return trajectories, mesh_info


def convert_to_numpy(data_dir='flag_data', output_dir='flag_data', max_train=100, max_test=20):
    """Convert TFRecord data to numpy format."""
    meta = load_metadata(data_dir)
    output_dir = Path(output_dir)

    print(f"Dataset info:")
    print(f"  Timesteps: {meta['trajectory_length']}")
    print(f"  dt: {meta['dt']} seconds")
    print(f"  Vertices: {meta['features']['world_pos']['shape'][1]}")
    print(f"  Triangles: {meta['features']['cells']['shape'][1]}")

    # Load test data (smaller)
    print(f"\nLoading test trajectories (max {max_test})...")
    test_trajs, mesh_info = load_trajectories(
        output_dir / 'test.tfrecord', meta, max_trajectories=max_test
    )
    print(f"  Loaded {len(test_trajs)} test trajectories")

    # Stack into arrays
    test_world_pos = np.stack([t['world_pos'] for t in test_trajs])  # (N, 401, 1579, 3)

    # Save
    print("\nSaving to numpy format...")
    np.savez_compressed(
        output_dir / 'flag_test.npz',
        world_pos=test_world_pos,
        cells=mesh_info['cells'],
        mesh_pos=mesh_info['mesh_pos'],
        dt=meta['dt'],
    )
    print(f"  Saved flag_test.npz: {test_world_pos.shape}")

    return test_world_pos, mesh_info, meta


def visualize_trajectory(world_pos, cells, mesh_pos, trajectory_idx=0, timesteps=[0, 100, 200, 300, 400]):
    """Visualize a flag trajectory at different timesteps."""
    fig = plt.figure(figsize=(15, 4))

    traj = world_pos[trajectory_idx]  # (401, 1579, 3)

    for i, t in enumerate(timesteps):
        ax = fig.add_subplot(1, len(timesteps), i + 1, projection='3d')

        pos = traj[t]  # (1579, 3)

        # Plot mesh surface
        ax.plot_trisurf(
            pos[:, 0], pos[:, 1], pos[:, 2],
            triangles=cells,
            cmap='coolwarm',
            alpha=0.8,
            linewidth=0.1,
            edgecolor='gray'
        )

        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        ax.set_title(f't = {t}')

        # Set consistent axis limits
        ax.set_xlim([0, 4])
        ax.set_ylim([-2, 2])
        ax.set_zlim([-2, 1])
        ax.view_init(elev=20, azim=45)

    plt.tight_layout()
    plt.savefig('flag_data/flag_trajectory.png', dpi=150)
    plt.close()
    print("Saved flag_trajectory.png")


def visualize_vertex_timeseries(world_pos, trajectory_idx=0, vertex_indices=[0, 500, 1000, 1500]):
    """Visualize position time series for selected vertices."""
    fig, axes = plt.subplots(3, 1, figsize=(12, 8), sharex=True)

    traj = world_pos[trajectory_idx]  # (401, 1579, 3)
    t = np.arange(traj.shape[0])

    labels = ['X', 'Y', 'Z']

    for i, (ax, label) in enumerate(zip(axes, labels)):
        for v in vertex_indices:
            ax.plot(t, traj[:, v, i], label=f'Vertex {v}', alpha=0.7)
        ax.set_ylabel(f'{label} position')
        ax.legend(loc='upper right')
        ax.grid(True, alpha=0.3)

    axes[-1].set_xlabel('Timestep')
    axes[0].set_title('Flag Vertex Trajectories Over Time')

    plt.tight_layout()
    plt.savefig('flag_data/vertex_timeseries.png', dpi=150)
    plt.close()
    print("Saved vertex_timeseries.png")


def analyze_dynamics(world_pos):
    """Analyze the dynamics of the flag data."""
    print("\n=== Dynamics Analysis ===")

    # Velocity (finite difference)
    velocity = np.diff(world_pos, axis=1)  # (N, 400, 1579, 3)

    print(f"Position stats:")
    print(f"  Shape: {world_pos.shape}")
    print(f"  Range: [{world_pos.min():.3f}, {world_pos.max():.3f}]")

    print(f"\nVelocity stats:")
    print(f"  Shape: {velocity.shape}")
    print(f"  Mean: {velocity.mean():.6f}")
    print(f"  Std: {velocity.std():.6f}")
    print(f"  Max magnitude: {np.linalg.norm(velocity, axis=-1).max():.4f}")

    # Per-axis stats
    for i, axis in enumerate(['X', 'Y', 'Z']):
        print(f"\n  {axis}-axis:")
        print(f"    Position std: {world_pos[:,:,:,i].std():.4f}")
        print(f"    Velocity std: {velocity[:,:,:,i].std():.6f}")


if __name__ == '__main__':
    # Convert data
    world_pos, mesh_info, meta = convert_to_numpy(max_train=0, max_test=50)

    # Visualize
    print("\nGenerating visualizations...")
    visualize_trajectory(world_pos, mesh_info['cells'], mesh_info['mesh_pos'])
    visualize_vertex_timeseries(world_pos)

    # Analyze
    analyze_dynamics(world_pos)
