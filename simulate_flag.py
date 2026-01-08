#!/usr/bin/env python3
"""
GPU-accelerated flag simulation using PyTorch.

Generates training data for the diffusion model by simulating cloth dynamics.
Uses position-based dynamics with vectorized constraint solving on GPU.

Usage:
    python simulate_flag.py              # Preview mode (no recording)
    python simulate_flag.py --record     # Record frames to flag_data/
    python simulate_flag.py --record --stride 100  # Record every 100th step
"""

import argparse
import numpy as np
import torch
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import matplotlib.animation as animation
from pathlib import Path


def get_device():
    """Get best available device."""
    if torch.cuda.is_available():
        return torch.device('cuda')
    elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        return torch.device('mps')
    return torch.device('cpu')


class FlagSimulation:
    """GPU-accelerated cloth simulation using position-based dynamics."""

    def __init__(self, mesh_pos, cells, device='cuda'):
        self.device = torch.device(device)
        self.cells = torch.tensor(cells, dtype=torch.long, device=self.device)

        V = mesh_pos.shape[0]
        self.num_vertices = V

        # Initialize 3D positions (flag in XZ plane)
        pos = torch.zeros((V, 3), dtype=torch.float32, device=self.device)
        pos[:, 0] = torch.tensor(mesh_pos[:, 0], dtype=torch.float32, device=self.device)
        pos[:, 2] = torch.tensor(mesh_pos[:, 1], dtype=torch.float32, device=self.device)
        # Small Y perturbation to break symmetry
        pos[:, 1] = torch.randn(V, device=self.device) * 0.01

        self.pos = pos
        self.prev_pos = pos.clone()

        # Fixed vertices (left edge)
        x_min = pos[:, 0].min().item()
        self.fixed = pos[:, 0] < x_min + 0.05
        self.fixed_pos = pos.clone()

        # Build edges from triangles
        edge_set = set()
        cells_np = cells if isinstance(cells, np.ndarray) else cells.cpu().numpy()
        for tri in cells_np:
            for i in range(3):
                e = tuple(sorted([int(tri[i]), int(tri[(i+1)%3])]))
                edge_set.add(e)
        edges = torch.tensor(list(edge_set), dtype=torch.long, device=self.device)
        self.edges = edges

        # Rest lengths
        p0 = pos[edges[:, 0]]
        p1 = pos[edges[:, 1]]
        self.rest_len = torch.norm(p1 - p0, dim=1)

        # Physics params
        self.gravity = torch.tensor([0., 0., -1.5], device=self.device)
        self.wind_base = torch.tensor([1.5, 0.0, 0.], device=self.device)
        self.damping = 0.995
        self.time = 0.0

        # Precompute edge counts per vertex for averaging
        self.edge_count = torch.zeros(V, device=self.device)
        self.edge_count.scatter_add_(0, edges[:, 0], torch.ones(len(edges), device=self.device))
        self.edge_count.scatter_add_(0, edges[:, 1], torch.ones(len(edges), device=self.device))
        self.edge_count = self.edge_count.clamp(min=1).unsqueeze(1)

        print(f"Simulation initialized on {self.device}:")
        print(f"  {V} vertices, {len(cells)} triangles, {len(edges)} edges")
        print(f"  {self.fixed.sum().item()} fixed vertices")

    def step(self):
        """Advance simulation by one timestep."""
        self.time += 0.2

        # Chaotic wind
        gust_x = torch.sin(torch.tensor(self.time * 5.0)) * 3.0 + torch.sin(torch.tensor(self.time * 7.3)) * 2.0
        gust_y = torch.sin(torch.tensor(self.time * 6.1)) * 2.5 + torch.cos(torch.tensor(self.time * 4.7)) * 1.5
        gust_z = torch.cos(torch.tensor(self.time * 5.7)) * 1.0
        turbulence = torch.randn(3, device=self.device) * 1.0

        wind = self.wind_base + torch.tensor([gust_x, gust_y, gust_z], device=self.device) + turbulence

        # Verlet integration
        vel = (self.pos - self.prev_pos) * self.damping
        accel = self.gravity + wind
        new_pos = self.pos + vel + accel * 0.0005

        self.prev_pos = self.pos.clone()
        self.pos = new_pos

        # Fix boundary
        self.pos[self.fixed] = self.fixed_pos[self.fixed]

        # Vectorized constraint solving
        for _ in range(15):
            p0 = self.pos[self.edges[:, 0]]
            p1 = self.pos[self.edges[:, 1]]
            delta = p1 - p0
            dist = torch.norm(delta, dim=1, keepdim=True).clamp(min=1e-8)

            # Correction
            error = (dist - self.rest_len.unsqueeze(1)) / dist
            correction = delta * error * 0.4

            # Accumulate corrections
            corr_pos = torch.zeros_like(self.pos)
            corr_neg = torch.zeros_like(self.pos)
            corr_pos.scatter_add_(0, self.edges[:, 0:1].expand(-1, 3), correction)
            corr_neg.scatter_add_(0, self.edges[:, 1:2].expand(-1, 3), correction)

            # Apply averaged corrections
            self.pos += (corr_pos - corr_neg) / self.edge_count
            self.pos[self.fixed] = self.fixed_pos[self.fixed]

    def get_positions(self):
        """Get current positions as numpy array."""
        return self.pos.cpu().numpy()


def run_simulation(record=False, stride=100, num_frames=10000, output_dir='flag_data'):
    """
    Run flag simulation with optional recording.

    Args:
        record: If True, save frames to disk
        stride: Save every Nth frame (for statistical independence)
        num_frames: Total simulation steps to run
        output_dir: Where to save recorded data
    """
    # Load mesh
    print("Loading mesh...")
    data = np.load('flag_data/flag_test.npz')
    mesh_pos = data['mesh_pos']
    cells = data['cells']

    device = get_device()
    print(f"Using device: {device}")

    sim = FlagSimulation(mesh_pos, cells, device=device)

    if record:
        import time

        print(f"\nRecording mode: stride={stride}, target_frames={num_frames//stride}")
        print(f"Output: {output_dir}/flag_simulated.npz")

        recorded_frames = []
        frame_indices = []

        # Warm-up (let simulation settle)
        print("Warming up (500 steps)...")
        for _ in range(500):
            sim.step()

        print(f"Recording {num_frames} steps...")
        start_time = time.time()
        last_report_time = start_time
        report_interval = 5.0  # Report every 5 seconds

        for step in range(num_frames):
            sim.step()

            if step % stride == 0:
                recorded_frames.append(sim.get_positions().copy())
                frame_indices.append(step)

            # Progress report every 5 seconds
            current_time = time.time()
            if current_time - last_report_time >= report_interval:
                elapsed = current_time - start_time
                steps_per_sec = (step + 1) / elapsed
                remaining_steps = num_frames - step - 1
                eta_seconds = remaining_steps / steps_per_sec if steps_per_sec > 0 else 0

                print(f"  Step {step+1:,}/{num_frames:,} | "
                      f"{steps_per_sec:,.0f} steps/sec | "
                      f"Frames: {len(recorded_frames)} | "
                      f"ETA: {eta_seconds/60:.1f} min")
                last_report_time = current_time

        total_time = time.time() - start_time
        print(f"\nCompleted {num_frames:,} steps in {total_time:.1f}s ({num_frames/total_time:,.0f} steps/sec)")

        # Save
        frames = np.stack(recorded_frames, axis=0)  # (N, V, 3)
        output_path = Path(output_dir) / 'flag_simulated.npz'
        np.savez(output_path,
                 world_pos=frames[np.newaxis, ...],  # (1, N, V, 3) to match original format
                 cells=cells,
                 mesh_pos=mesh_pos,
                 stride=stride,
                 num_steps=num_frames)

        print(f"\nSaved {len(frames)} frames to {output_path}")
        print(f"  Shape: {frames.shape}")
        print(f"  Position range: [{frames.min():.3f}, {frames.max():.3f}]")
        return

    # Preview mode - animated visualization
    print("\nPreview mode (close window to stop)")

    fig = plt.figure(figsize=(12, 9))
    ax = fig.add_subplot(111, projection='3d')

    pos_np = sim.get_positions()
    tris = pos_np[cells]
    poly = Poly3DCollection(tris, alpha=0.9, facecolor='crimson',
                            edgecolor='darkred', linewidth=0.2)
    ax.add_collection3d(poly)

    ax.set_xlim(-2, 5)
    ax.set_ylim(-3, 3)
    ax.set_zlim(-3, 3)
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title('Flag Simulation (GPU)')
    ax.view_init(elev=20, azim=-60)

    frame_num = [0]

    def update(frame):
        sim.step()
        frame_num[0] += 1

        pos_np = sim.get_positions()
        poly.set_verts(pos_np[cells])

        ax.set_title(f'Flag Simulation - Frame {frame_num[0]} ({sim.device})')
        return [poly]

    ani = animation.FuncAnimation(fig, update, frames=500,
                                  interval=16, blit=False,
                                  cache_frame_data=False)
    plt.tight_layout()
    plt.show()
    print(f"Ended at frame {frame_num[0]}")


def main():
    parser = argparse.ArgumentParser(description='Flag simulation for training data generation')
    parser.add_argument('--record', action='store_true', help='Record frames to disk')
    parser.add_argument('--stride', type=int, default=100, help='Save every Nth frame (default: 100)')
    parser.add_argument('--frames', type=int, default=10000, help='Total simulation steps (default: 10000)')
    parser.add_argument('--output', type=str, default='flag_data', help='Output directory')
    args = parser.parse_args()

    run_simulation(
        record=args.record,
        stride=args.stride,
        num_frames=args.frames,
        output_dir=args.output
    )


if __name__ == "__main__":
    main()
