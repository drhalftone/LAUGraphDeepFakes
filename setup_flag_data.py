#!/usr/bin/env python3
"""
Download and prepare DeepMind's flag simulation data for training.

This script:
1. Downloads flag_simple data from Google Cloud Storage
2. Converts TFRecord format to numpy
3. Saves as flag_data/flag_test.npz (ready for train_flag_diffusion.py)

Usage:
    python setup_flag_data.py

Requirements:
    - tensorflow (for TFRecord parsing)
    - numpy
    - Internet connection (~1GB download for test.tfrecord)
"""

import os
import sys
import urllib.request
import json
import numpy as np
from pathlib import Path

# Data URLs
BASE_URL = "https://storage.googleapis.com/dm-meshgraphnets/flag_simple"
FILES = {
    "meta.json": f"{BASE_URL}/meta.json",
    "test.tfrecord": f"{BASE_URL}/test.tfrecord",
    # "train.tfrecord": f"{BASE_URL}/train.tfrecord",  # 9.5GB - uncomment if needed
}

DATA_DIR = Path("flag_data")
OUTPUT_FILE = DATA_DIR / "flag_test.npz"


def download_with_progress(url, filepath):
    """Download a file with progress indicator."""
    print(f"Downloading: {url}")
    print(f"        To: {filepath}")

    def progress_hook(block_num, block_size, total_size):
        downloaded = block_num * block_size
        if total_size > 0:
            percent = min(100, downloaded * 100 / total_size)
            mb_downloaded = downloaded / (1024 * 1024)
            mb_total = total_size / (1024 * 1024)
            sys.stdout.write(f"\r  Progress: {percent:5.1f}% ({mb_downloaded:.1f}/{mb_total:.1f} MB)")
            sys.stdout.flush()

    urllib.request.urlretrieve(url, filepath, progress_hook)
    print()  # newline after progress


def download_data():
    """Download all required files."""
    print("=" * 60)
    print("Step 1: Downloading flag_simple data")
    print("=" * 60)

    DATA_DIR.mkdir(exist_ok=True)

    for filename, url in FILES.items():
        filepath = DATA_DIR / filename

        if filepath.exists():
            size_mb = filepath.stat().st_size / (1024 * 1024)
            print(f"  {filename} already exists ({size_mb:.1f} MB), skipping")
        else:
            download_with_progress(url, filepath)

    print()


def convert_tfrecord():
    """Convert TFRecord to numpy format."""
    print("=" * 60)
    print("Step 2: Converting TFRecord to numpy")
    print("=" * 60)

    if OUTPUT_FILE.exists():
        print(f"  {OUTPUT_FILE} already exists, skipping conversion")
        print("  (Delete the file to reconvert)")
        return True

    # Check for TensorFlow
    try:
        import tensorflow as tf
        print(f"  Using TensorFlow {tf.__version__}")
    except ImportError:
        print("  ERROR: TensorFlow not found!")
        print("  Install with: pip install tensorflow")
        print()
        print("  Alternatively, if you have the .npz file from another source,")
        print("  place it at: flag_data/flag_test.npz")
        return False

    # Load metadata
    meta_path = DATA_DIR / "meta.json"
    if not meta_path.exists():
        print(f"  ERROR: {meta_path} not found!")
        return False

    with open(meta_path) as f:
        meta = json.load(f)

    print(f"  Metadata: {meta}")

    # Parse TFRecord
    tfrecord_path = DATA_DIR / "test.tfrecord"
    if not tfrecord_path.exists():
        print(f"  ERROR: {tfrecord_path} not found!")
        return False

    print(f"  Parsing: {tfrecord_path}")

    dataset = tf.data.TFRecordDataset(str(tfrecord_path))

    trajectories = []
    cells = None
    mesh_pos = None

    for i, raw_record in enumerate(dataset):
        # Parse the example
        example = tf.train.Example()
        example.ParseFromString(raw_record.numpy())
        features = example.features.feature

        # Extract fields
        world_pos_bytes = features['world_pos'].bytes_list.value[0]
        world_pos = tf.io.decode_raw(world_pos_bytes, tf.float32).numpy()

        cells_bytes = features['cells'].bytes_list.value[0]
        cells_data = tf.io.decode_raw(cells_bytes, tf.int32).numpy()

        mesh_pos_bytes = features['mesh_pos'].bytes_list.value[0]
        mesh_pos_data = tf.io.decode_raw(mesh_pos_bytes, tf.float32).numpy()

        # Reshape
        # world_pos: (T, V, 3) where T=timesteps, V=vertices
        # We need to figure out the shape from the data
        if cells is None:
            # cells is (F, 3) for triangles
            cells = cells_data.reshape(-1, 3)
            num_vertices = cells.max() + 1

            # mesh_pos is (V, 2) for 2D rest positions
            mesh_pos = mesh_pos_data.reshape(num_vertices, -1)

            print(f"  Mesh: {cells.shape[0]} triangles, {num_vertices} vertices")

        # world_pos is (T, V, 3)
        num_vertices = cells.max() + 1
        world_pos = world_pos.reshape(-1, num_vertices, 3)

        trajectories.append(world_pos)

        if (i + 1) % 10 == 0:
            print(f"  Loaded {i + 1} trajectories...")

    print(f"  Total: {len(trajectories)} trajectories")

    # Stack all trajectories
    # They should all have the same shape
    world_pos_all = np.stack(trajectories, axis=0)  # (N, T, V, 3)

    print(f"  Shape: {world_pos_all.shape}")
    print(f"    N = {world_pos_all.shape[0]} trajectories")
    print(f"    T = {world_pos_all.shape[1]} timesteps")
    print(f"    V = {world_pos_all.shape[2]} vertices")

    # Save
    print(f"  Saving to: {OUTPUT_FILE}")
    np.savez(
        OUTPUT_FILE,
        world_pos=world_pos_all,
        cells=cells,
        mesh_pos=mesh_pos
    )

    size_mb = OUTPUT_FILE.stat().st_size / (1024 * 1024)
    print(f"  Saved: {size_mb:.1f} MB")

    return True


def verify_data():
    """Verify the converted data."""
    print()
    print("=" * 60)
    print("Step 3: Verifying data")
    print("=" * 60)

    if not OUTPUT_FILE.exists():
        print(f"  ERROR: {OUTPUT_FILE} not found!")
        return False

    data = np.load(OUTPUT_FILE)

    print(f"  world_pos: {data['world_pos'].shape} - trajectory positions")
    print(f"  cells:     {data['cells'].shape} - mesh triangles")
    print(f"  mesh_pos:  {data['mesh_pos'].shape} - rest state positions")

    # Quick sanity checks
    world_pos = data['world_pos']
    print()
    print(f"  Position range: [{world_pos.min():.3f}, {world_pos.max():.3f}]")
    print(f"  Position std:   {world_pos.std():.3f}")

    # Check vertex 0 (should be fixed at pole)
    v0_movement = world_pos[:, :, 0, :].std()
    print(f"  Vertex 0 std:   {v0_movement:.6f} (should be ~0 if fixed)")

    return True


def main():
    print()
    print("=" * 60)
    print("Flag Data Setup")
    print("=" * 60)
    print()
    print("This will download ~1GB of data from Google Cloud Storage")
    print("and convert it to numpy format for training.")
    print()

    # Step 1: Download
    download_data()

    # Step 2: Convert
    success = convert_tfrecord()
    if not success:
        print()
        print("Conversion failed. See errors above.")
        return 1

    # Step 3: Verify
    verify_data()

    print()
    print("=" * 60)
    print("Setup complete!")
    print("=" * 60)
    print()
    print("Next steps:")
    print("  1. Install PyTorch with CUDA: pip install torch --index-url https://download.pytorch.org/whl/cu118")
    print("  2. Install PyTorch Geometric: pip install torch_geometric")
    print("  3. Run training: python train_flag_diffusion.py")
    print()

    return 0


if __name__ == "__main__":
    sys.exit(main())
