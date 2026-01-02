# Comparison Plan: Our GAD Implementation vs Official GAD

## Goal
Compare our Python GAD implementation in `train_model.py` with the official GAD code from `github.com/vimalkb7/gad` using our PDE dataset (245 FEA solutions, 6523 nodes).

---

## Phase 1: Data Format Conversion

Create `export_for_official_gad.py` to convert our data format to official GAD format:

| Our Format | Official GAD Format |
|------------|---------------------|
| `dataset/solutions.npz['solutions']` (245, 6523) | `x_train.npy`, `x_test.npy` (N, num_nodes) |
| `dataset/laplacian.npz` (sparse CSR) | `L.npy` (dense combinatorial), `L_sym.npy` (dense symmetric normalized) |
| `dataset/adjacency.npz` (sparse CSR) | Computed from L |

**Tasks:**
1. Split solutions 80/20 into train/test
2. Convert sparse Laplacian to dense numpy arrays
3. Compute symmetric normalized Laplacian: `L_sym = D^(-1/2) L D^(-1/2)`
4. Save in official GAD expected directory structure

---

## Phase 2: Run Official GAD on Our Data

**Tasks:**
1. Clone official GAD repo to `comparison/official_gad/`
2. Install dependencies (PyTorch Geometric, torch-sparse)
3. Create config for our mesh (6523 nodes, heat equation data)
4. Train official GAD model with same hyperparameters:
   - 100 diffusion timesteps
   - Polynomial filter order K=3
   - ~1000 epochs
5. Save generated samples and training metrics

---

## Phase 3: Run Our Implementation

**Tasks:**
1. Ensure our `train_model.py` uses matching hyperparameters
2. Train GraphVAE + GraphAwareDiffusion
3. Save generated samples and training metrics
4. Export samples in compatible format for comparison

---

## Phase 4: Evaluation Metrics

Implement `compare_results.py` with metrics:

| Metric | Description |
|--------|-------------|
| **MSE to Training Data** | Average distance from generated samples to nearest training sample |
| **Spectral Distance** | Compare graph Fourier coefficients of real vs generated |
| **MMD (Maximum Mean Discrepancy)** | Distribution similarity in feature space |
| **Smoothness** | `x^T L x` measures signal smoothness on graph |
| **Visual Comparison** | Side-by-side plots of real vs generated fields |

---

## Phase 5: Visualization & Report

Create comparison visualizations:
1. Training curves (loss vs epoch) for both methods
2. Generated sample grids from both methods
3. Real vs Generated comparison panels
4. Histogram of smoothness values
5. PCA/t-SNE of latent spaces

---

## Key Differences to Document

| Aspect | Our Implementation | Official GAD |
|--------|-------------------|--------------|
| VAE | Spectral GraphVAE | None (direct diffusion) |
| Diffusion Space | 64-dim latent | Full graph signal |
| Filter | Polynomial on S | GraphFilterTap with GNN |
| Schedule | GASDE with eigenmode decay | GASDE |
| Sampling | Euler-Maruyama | PC sampler |

---

## Expected Outcome

A comprehensive comparison showing:
- Which approach generates more realistic FEA fields
- Training efficiency (time, convergence)
- Sample diversity and quality
- Whether latent space compression helps or hurts
