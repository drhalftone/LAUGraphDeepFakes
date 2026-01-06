# Graph Deep Fakes for FEA Simulations

Generate synthetic FEA (Finite Element Analysis) signals on meshes using graph neural networks and diffusion models.

## Overview

This project trains a generative model to create "deep fake" FEA fields on a fixed mesh. The approach:

1. **Graph Signal Processing**: Treat the FEA mesh as a graph with the cotangent-weighted Laplacian
2. **Spectral Graph VAE**: Compress high-dimensional fields (~6500 nodes) to a low-dimensional latent space (~64 dims)
3. **Graph-Aware Diffusion (GAD)**: Train a diffusion model with polynomial graph filters on the latent space
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
├── dataset/                    # FEA training data (heat equation)
├── flag_data/                  # Flag simulation data (download separately)
├── docs/                       # Documentation and notes
│   ├── trajectory_diffusion.md # Main diffusion approach (START HERE)
│   ├── spectral_vae_augmentation.md
│   └── graph_signal_augmentation.md
├── reports/                    # LaTeX reports
│   └── diffusion_for_graph_signals.tex
├── train_model.py             # Original VAE+Diffusion for FEA
├── train_spectral_vae.py      # Spectral VAE approach
├── load_flag_data.py          # Load DeepMind flag data
├── generate_flag_noise.py     # Noise filtering baseline
└── setup_gpu.sh               # GPU environment setup
```

## Current Approach: Trajectory Diffusion

See **[docs/trajectory_diffusion.md](docs/trajectory_diffusion.md)** for the latest approach using diffusion models with GNNs for mesh trajectory generation.

**Key insight**: Diffusion models support data augmentation by controlling noise injection level - add less noise to stay closer to the original signal.

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
pip install numpy scipy matplotlib scikit-learn jupyter
```

## Usage

### Train the Model

**Command line:**
```bash
python train_model.py
```

**Jupyter notebook:**
```bash
jupyter notebook
# Open train_model.ipynb
```

This will:
- Load the pre-generated dataset (49 FEA solutions)
- Train a Graph VAE (~500 epochs)
- Train a Graph-Aware Diffusion model (~800 epochs)
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
- `DIFFUSIVITY_VALUES`: Thermal conductivity range (default: 7 values from 0.1 to 10.0)
- `SOURCE_VALUES`: Heat source strength range (default: 7 values from 0.5 to 7.0)
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
| `fcps_schedule.png` | FCPS vs linear noise schedule |
| `latent_space.png` | PCA of latent space |

## Model Architecture

### Graph VAE
- **Encoder**: Projects fields to spectral domain via Laplacian eigenvectors, then to latent space
- **Decoder**: Reconstructs fields from latent codes using spectral coefficients
- **Latent dim**: 64

### Graph-Aware Diffusion (GAD)
- **Schedule**: Floor Constrained Polynomial Schedule (FCPS)
- **Denoiser**: Polynomial graph filter H(S) = Σ θ_k S^k
- **Steps**: 100
- **Operates on**: 64-dim latent space (not full mesh)

Based on: [arXiv:2510.05036](https://arxiv.org/abs/2510.05036)

## Dataset Details

The included dataset contains 49 steady-state heat equation solutions:

- **Mesh**: 6,523 nodes, 12,589 triangular elements
- **Domain**: Rectangular region with circular obstacle (cylinder at origin)
- **Parameters varied**:
  - Diffusivity (k): 0.1, 0.2, 0.5, 1.0, 2.0, 5.0, 10.0
  - Source strength (Q): 0.5, 1.0, 2.0, 3.0, 4.0, 5.0, 7.0

## References

This project was inspired by:
- Graph Signal Processing on meshes
- Spectral mesh processing and manifold harmonics
- Graph-Aware Diffusion (GAD) for signal generation ([arXiv:2510.05036](https://arxiv.org/abs/2510.05036))

See the `docs/` folder for detailed documentation and the `reports/` folder for LaTeX reports.

## License

MIT
