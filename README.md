# Graph Deep Fakes for FEA Simulations

Generate synthetic FEA (Finite Element Analysis) signals on meshes using graph neural networks and diffusion models.

## Overview

This project trains a generative model to create "deep fake" FEA fields on a fixed mesh. The approach:

1. **Graph Signal Processing**: Treat the FEA mesh as a graph with the cotangent-weighted Laplacian
2. **Spectral Graph VAE**: Compress high-dimensional fields (~6500 nodes) to a low-dimensional latent space (~64 dims)
3. **Latent Diffusion**: Train a DDPM-style diffusion model on the latent space
4. **Generation**: Sample new latent codes and decode to synthetic FEA fields

## Quick Start (GPU)

```bash
git clone git@github.com:drhalftone/LAUGraphDeepFakes.git
cd LAUGraphDeepFakes
chmod +x setup_gpu.sh && ./setup_gpu.sh
source gdf_gpu/bin/activate
python train_model.py
```

## Requirements

- Python 3.11 or 3.12 (PyTorch requirement)
- NVIDIA GPU with CUDA support (recommended)
- ~4GB disk space

## Project Structure

```
LAUGraphDeepFakes/
├── dataset/                    # Pre-generated training data
│   ├── mesh.npz               # Node positions, triangles, eigenvectors
│   ├── solutions.npz          # 245 FEA solutions
│   ├── laplacian.npz          # Graph Laplacian (sparse)
│   └── adjacency.npz          # Adjacency weights (sparse)
├── train_model.py             # Main training script
├── generate_dataset.py        # Dataset generation script
├── run_simulation.py          # Single simulation visualization
├── setup_gpu.sh               # GPU environment setup
├── setup.sh                   # CPU environment setup
└── requirements.txt           # Python dependencies
```

## Setup Options

### GPU Setup (Recommended)

For machines with NVIDIA GPU (tested on RTX 4070 Ti Super):

```bash
chmod +x setup_gpu.sh
./setup_gpu.sh
source gdf_gpu/bin/activate
```

### CPU Setup

For machines without GPU:

```bash
chmod +x setup.sh
./setup.sh
source gdf_env/bin/activate
```

Note: CPU training will be significantly slower.

### Conda Setup

If you prefer conda:

```bash
conda create -n gdf python=3.11 pytorch torchvision -c pytorch
conda activate gdf
pip install numpy scipy matplotlib scikit-learn
```

## Usage

### Train the Model

```bash
python train_model.py
```

This will:
- Load the pre-generated dataset (245 FEA solutions)
- Train a Graph VAE (~500 epochs)
- Train a Latent Diffusion model (~800 epochs)
- Generate synthetic samples
- Save visualizations to `training_output/`

Expected training time:
- GPU (RTX 4070 Ti): ~15-25 minutes
- CPU: ~2-4 hours

### Generate New Dataset

To regenerate the FEA dataset with different parameters:

```bash
python generate_dataset.py
```

Configurable parameters in the script:
- `DIFFUSIVITY_VALUES`: Thermal conductivity range
- `SOURCE_VALUES`: Heat source strength range
- `CYLINDER_Y_VALUES`: Cylinder position range
- `MESH_RESOLUTION`: Mesh density

### Run Single Simulation

To visualize a single FEA simulation:

```bash
python run_simulation.py
```

## Output Files

After training, `training_output/` contains:

| File | Description |
|------|-------------|
| `models_final.pt` | Trained VAE + Diffusion checkpoint |
| `synthetic_samples.npz` | Generated synthetic fields |
| `synthetic_samples.png` | Visualization of generated fields |
| `real_vs_synthetic.png` | Comparison of real vs generated |
| `training_curves.png` | Loss curves |
| `latent_space.png` | PCA of latent space |

## Model Architecture

### Graph VAE
- **Encoder**: Projects fields to spectral domain via Laplacian eigenvectors, then to latent space
- **Decoder**: Reconstructs fields from latent codes using spectral coefficients
- **Latent dim**: 64

### Latent Diffusion
- **Type**: DDPM (Denoising Diffusion Probabilistic Model)
- **Steps**: 100
- **Operates on**: 64-dim latent space (not full mesh)

## Dataset Details

The included dataset contains 245 steady-state heat equation solutions:

- **Mesh**: 6,523 nodes, 12,589 triangular elements
- **Domain**: Rectangular region with circular obstacle (cylinder)
- **Parameters varied**:
  - Diffusivity (k): 0.1 to 10.0
  - Source strength (Q): 0.5 to 7.0
  - Cylinder Y position: -0.03 to +0.03

## References

This project was inspired by discussions on:
- Graph Signal Processing on meshes
- Spectral mesh processing and manifold harmonics
- Graph-aware diffusion models for signal generation

See the included `.md` files for the original conversation notes.

## License

MIT
