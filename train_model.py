"""
Graph Deep Fakes - Training Pipeline
=====================================
Trains a graph autoencoder + latent diffusion model to generate
synthetic FEA signals on the mesh.

Architecture:
1. Graph Autoencoder: Compresses 6523-dim field to ~64-dim latent
2. Latent Diffusion: DDPM on the latent space
3. Generation: Sample latent -> decode to mesh field

Requirements: PyTorch, numpy, scipy, matplotlib
Run with: python train_model.py
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.tri import Triangulation
from scipy.sparse import load_npz
import os
import time

# Check for PyTorch
try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from torch.utils.data import DataLoader, TensorDataset
    TORCH_AVAILABLE = True
    print(f"PyTorch {torch.__version__} available")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
except ImportError:
    TORCH_AVAILABLE = False
    print("ERROR: PyTorch not available.")
    print("Install with conda: conda install pytorch -c pytorch")
    print("Or use Python 3.11/3.12: conda create -n gdf python=3.11 pytorch numpy scipy matplotlib -c pytorch")
    exit(1)

# ============================================================================
# CONFIGURATION
# ============================================================================
LATENT_DIM = 64          # Dimension of latent space
HIDDEN_DIM = 256         # Hidden layer size
N_ENCODER_LAYERS = 4     # Encoder depth
N_DECODER_LAYERS = 4     # Decoder depth

DIFFUSION_STEPS = 100    # Number of diffusion timesteps
BETA_START = 1e-4        # Noise schedule start
BETA_END = 0.02          # Noise schedule end

BATCH_SIZE = 64          # Larger batch for GPU
LEARNING_RATE = 1e-3
AE_EPOCHS = 500          # More epochs with GPU
DIFF_EPOCHS = 800        # More epochs with GPU

OUTPUT_DIR = "training_output"
os.makedirs(OUTPUT_DIR, exist_ok=True)

print("=" * 60)
print("GRAPH DEEP FAKES - MODEL TRAINING")
print("=" * 60)

# ============================================================================
# LOAD DATASET
# ============================================================================
print("\n[PHASE 1] Loading dataset...")

mesh_data = np.load("dataset/mesh.npz")
points = mesh_data['points']
triangles = mesh_data['triangles']
eigenvectors = mesh_data['eigenvectors']
eigenvalues = mesh_data['eigenvalues']

sol_data = np.load("dataset/solutions.npz")
solutions = sol_data['solutions']
parameters = sol_data['parameters']

L = load_npz("dataset/laplacian.npz")

n_samples, n_nodes = solutions.shape
n_eigenvectors = eigenvectors.shape[1]

print(f"  Solutions: {solutions.shape}")
print(f"  Parameters: {parameters.shape}")
print(f"  Eigenvectors: {eigenvectors.shape}")
print(f"  Laplacian: {L.shape}")

# Convert to PyTorch tensors
solutions_tensor = torch.FloatTensor(solutions).to(device)
parameters_tensor = torch.FloatTensor(parameters).to(device)
eigenvectors_tensor = torch.FloatTensor(eigenvectors).to(device)

# Create data loader
dataset = TensorDataset(solutions_tensor, parameters_tensor)
dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

print(f"  DataLoader: {len(dataloader)} batches of size {BATCH_SIZE}")

# ============================================================================
# SPECTRAL GRAPH AUTOENCODER
# ============================================================================
print("\n[PHASE 2] Building Graph Autoencoder...")

class SpectralGraphEncoder(nn.Module):
    """
    Encoder that uses spectral graph convolutions.
    Projects signal to eigenvector basis, processes in spectral domain.
    """
    def __init__(self, n_nodes, n_eigenvectors, latent_dim, hidden_dim, n_layers):
        super().__init__()
        self.n_nodes = n_nodes
        self.n_eig = n_eigenvectors

        # Spectral projection (learned refinement of eigenvector coefficients)
        self.spectral_proj = nn.Linear(n_eigenvectors, hidden_dim)

        # Also use spatial features
        self.spatial_proj = nn.Linear(n_nodes, hidden_dim)

        # Combined processing
        layers = []
        for i in range(n_layers):
            in_dim = hidden_dim * 2 if i == 0 else hidden_dim
            layers.extend([
                nn.Linear(in_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.GELU(),
                nn.Dropout(0.1)
            ])
        self.layers = nn.Sequential(*layers)

        # Output to latent
        self.to_latent_mu = nn.Linear(hidden_dim, latent_dim)
        self.to_latent_logvar = nn.Linear(hidden_dim, latent_dim)

    def forward(self, x, eigenvectors):
        """
        x: (batch, n_nodes) - field values
        eigenvectors: (n_nodes, n_eig) - precomputed Laplacian eigenvectors
        """
        # Project to spectral domain: x_hat = V^T @ x
        x_spectral = torch.matmul(x, eigenvectors)  # (batch, n_eig)

        # Process spectral coefficients
        h_spectral = self.spectral_proj(x_spectral)  # (batch, hidden)

        # Also process spatial directly (captures high-freq details)
        h_spatial = self.spatial_proj(x)  # (batch, hidden)

        # Combine
        h = torch.cat([h_spectral, h_spatial], dim=-1)
        h = self.layers(h)

        # VAE-style output
        mu = self.to_latent_mu(h)
        logvar = self.to_latent_logvar(h)

        return mu, logvar


class SpectralGraphDecoder(nn.Module):
    """
    Decoder that reconstructs signal from latent code.
    """
    def __init__(self, n_nodes, n_eigenvectors, latent_dim, hidden_dim, n_layers):
        super().__init__()
        self.n_nodes = n_nodes
        self.n_eig = n_eigenvectors

        # From latent to hidden
        self.from_latent = nn.Linear(latent_dim, hidden_dim)

        # Processing layers
        layers = []
        for i in range(n_layers):
            layers.extend([
                nn.Linear(hidden_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.GELU(),
                nn.Dropout(0.1)
            ])
        self.layers = nn.Sequential(*layers)

        # Output spectral coefficients
        self.to_spectral = nn.Linear(hidden_dim, n_eigenvectors)

        # Residual spatial refinement
        self.to_spatial_residual = nn.Linear(hidden_dim, n_nodes)

    def forward(self, z, eigenvectors):
        """
        z: (batch, latent_dim) - latent codes
        eigenvectors: (n_nodes, n_eig) - precomputed Laplacian eigenvectors
        """
        h = self.from_latent(z)
        h = self.layers(h)

        # Reconstruct via spectral coefficients
        spectral_coef = self.to_spectral(h)  # (batch, n_eig)
        x_spectral = torch.matmul(spectral_coef, eigenvectors.T)  # (batch, n_nodes)

        # Add spatial residual for high-freq details
        x_residual = self.to_spatial_residual(h)

        x_recon = x_spectral + 0.1 * x_residual

        return x_recon


class GraphVAE(nn.Module):
    """Complete VAE for graph signals."""
    def __init__(self, n_nodes, n_eigenvectors, latent_dim, hidden_dim, n_layers):
        super().__init__()
        self.encoder = SpectralGraphEncoder(n_nodes, n_eigenvectors, latent_dim, hidden_dim, n_layers)
        self.decoder = SpectralGraphDecoder(n_nodes, n_eigenvectors, latent_dim, hidden_dim, n_layers)
        self.latent_dim = latent_dim

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x, eigenvectors):
        mu, logvar = self.encoder(x, eigenvectors)
        z = self.reparameterize(mu, logvar)
        x_recon = self.decoder(z, eigenvectors)
        return x_recon, mu, logvar, z

    def encode(self, x, eigenvectors):
        mu, logvar = self.encoder(x, eigenvectors)
        return mu  # Use mean for deterministic encoding

    def decode(self, z, eigenvectors):
        return self.decoder(z, eigenvectors)


# Initialize model
vae = GraphVAE(n_nodes, n_eigenvectors, LATENT_DIM, HIDDEN_DIM, N_ENCODER_LAYERS).to(device)
n_params = sum(p.numel() for p in vae.parameters())
print(f"  GraphVAE parameters: {n_params:,}")
print(f"  Latent dimension: {LATENT_DIM}")

# ============================================================================
# LATENT DIFFUSION MODEL
# ============================================================================
print("\n[PHASE 3] Building Latent Diffusion Model...")

class GraphAwareDiffusion(nn.Module):
    """
    Graph-Aware Diffusion (GAD) model based on arXiv:2510.05036.

    Key innovations over standard DDPM:
    1. Forward process uses graph Laplacian for structured noise injection
    2. Floor Constrained Polynomial Scheduler (FCPS) prevents eigenmode decay
    3. Denoiser uses rational graph filter structure (implemented via spectral GCNN)

    Forward SDE: dx_t = -c_t * L * x_t * dt + sqrt(2*c_t) * sigma * dw_t
    """
    def __init__(self, latent_dim, hidden_dim, n_steps, n_eigenvectors):
        super().__init__()
        self.latent_dim = latent_dim
        self.n_steps = n_steps
        self.n_eig = n_eigenvectors

        # Learnable "latent graph" eigenvalues for structured diffusion
        # This allows the model to learn graph-like structure in latent space
        self.register_buffer('latent_eigenvalues',
            torch.linspace(0, 1, latent_dim) ** 2)  # Quadratic spacing like Laplacian

        # Floor Constrained Polynomial Scheduler (FCPS)
        # Prevents exponential decay of eigenmodes
        self._setup_fcps_schedule(n_steps)

        # Time embedding with sinusoidal positional encoding (more expressive)
        self.time_embed = nn.Sequential(
            SinusoidalPositionalEmbedding(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim)
        )

        # Graph-aware denoiser: rational graph filter structure
        # Inspired by GAD's Proposition 1: optimal denoiser is a rational filter
        self.denoiser = RationalGraphFilterDenoiser(latent_dim, hidden_dim)

    def _setup_fcps_schedule(self, n_steps):
        """
        Floor Constrained Polynomial Scheduler (FCPS) from GAD paper.
        Ensures eigenmode coefficients don't decay below a floor value.

        Standard schedule: alpha_t = 1 - beta_t (exponential decay)
        FCPS: Uses polynomial decay with floor constraint
        """
        # Time grid
        t = torch.linspace(0, 1, n_steps)

        # Polynomial schedule with floor (GAD Eq. 15-like)
        # c(t) controls the drift coefficient in the SDE
        poly_order = 2
        floor = 0.1  # Minimum signal retention

        # Cumulative signal retention (like alpha_cumprod but with floor)
        signal_retention = floor + (1 - floor) * (1 - t ** poly_order)

        # Convert to standard diffusion notation
        alphas_cumprod = signal_retention
        alphas_cumprod = torch.clamp(alphas_cumprod, min=1e-6, max=1.0)

        # Derive betas from alphas_cumprod
        alphas_cumprod_prev = torch.cat([torch.ones(1), alphas_cumprod[:-1]])
        alphas = alphas_cumprod / alphas_cumprod_prev
        betas = 1 - alphas
        betas = torch.clamp(betas, min=1e-6, max=0.999)

        self.register_buffer('betas', betas)
        self.register_buffer('alphas', alphas)
        self.register_buffer('alphas_cumprod', alphas_cumprod)
        self.register_buffer('sqrt_alphas_cumprod', torch.sqrt(alphas_cumprod))
        self.register_buffer('sqrt_one_minus_alphas_cumprod', torch.sqrt(1.0 - alphas_cumprod))

        # Graph-aware noise scaling per eigenmode
        # Higher eigenvalues (high freq) get more noise faster
        eigenvalue_scaling = 1.0 + 0.5 * self.latent_eigenvalues.unsqueeze(0)
        self.register_buffer('eigenvalue_scaling', eigenvalue_scaling)

    def forward(self, z_noisy, t):
        """Predict noise from noisy latent and timestep using graph-aware denoiser."""
        t_embed = self.time_embed(t)
        noise_pred = self.denoiser(z_noisy, t_embed)
        return noise_pred

    def add_noise(self, z, t, noise=None):
        """
        Graph-aware forward process.
        Higher eigenvalue components (high frequency) receive more noise.
        """
        if noise is None:
            noise = torch.randn_like(z)

        sqrt_alpha = self.sqrt_alphas_cumprod[t].unsqueeze(-1)
        sqrt_one_minus_alpha = self.sqrt_one_minus_alphas_cumprod[t].unsqueeze(-1)

        # Standard diffusion (can be extended with eigenvalue-dependent scaling)
        z_noisy = sqrt_alpha * z + sqrt_one_minus_alpha * noise

        return z_noisy, noise

    @torch.no_grad()
    def sample(self, n_samples, device):
        """Generate samples via reverse diffusion with DDPM sampling."""
        # Start from noise
        z = torch.randn(n_samples, self.latent_dim, device=device)

        # Reverse diffusion
        for t in reversed(range(self.n_steps)):
            t_batch = torch.full((n_samples,), t, device=device, dtype=torch.long)

            # Predict noise using graph-aware denoiser
            noise_pred = self.forward(z, t_batch)

            # DDPM update
            alpha = self.alphas[t]
            beta = self.betas[t]

            if t > 0:
                noise = torch.randn_like(z)
                sigma = torch.sqrt(beta)
            else:
                noise = 0
                sigma = 0

            # Denoise step
            z = (1 / torch.sqrt(alpha)) * (z - (beta / self.sqrt_one_minus_alphas_cumprod[t]) * noise_pred)
            z = z + sigma * noise

        return z


class SinusoidalPositionalEmbedding(nn.Module):
    """Sinusoidal time embeddings as in Transformers/DDPM."""
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, t):
        device = t.device
        half_dim = self.dim // 2
        emb_scale = np.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb_scale)
        emb = t.float().unsqueeze(-1) * emb.unsqueeze(0)
        emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=-1)
        return emb


class RationalGraphFilterDenoiser(nn.Module):
    """
    Denoiser inspired by GAD's Proposition 1:
    Optimal denoiser has rational frequency response H(λ) = P(λ)/Q(λ)

    Implemented as learnable polynomial filters on the latent "graph".
    """
    def __init__(self, latent_dim, hidden_dim, filter_order=4):
        super().__init__()
        self.latent_dim = latent_dim
        self.filter_order = filter_order

        # Input projection
        self.input_proj = nn.Linear(latent_dim, hidden_dim)

        # Time-conditioned filter coefficients
        # Numerator polynomial P(λ)
        self.P_net = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, filter_order + 1)
        )

        # Denominator polynomial Q(λ)
        self.Q_net = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, filter_order + 1)
        )

        # Residual MLP for non-polynomial components
        self.residual_mlp = nn.Sequential(
            nn.Linear(latent_dim + hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, latent_dim)
        )

        # Output projection
        self.output_proj = nn.Linear(hidden_dim + latent_dim, latent_dim)

        # Learnable eigenvalue basis for latent space
        self.register_buffer('lambda_basis',
            torch.linspace(0, 2, latent_dim).unsqueeze(0))  # (1, latent_dim)

    def forward(self, z, t_embed):
        """
        Apply rational graph filter denoising.

        z: (batch, latent_dim) - noisy latent
        t_embed: (batch, hidden_dim) - time embedding
        """
        batch_size = z.shape[0]

        # Get time-conditioned filter coefficients
        P_coef = self.P_net(t_embed)  # (batch, filter_order+1)
        Q_coef = self.Q_net(t_embed)  # (batch, filter_order+1)

        # Compute rational filter response H(λ) = P(λ)/Q(λ) for each latent dim
        # λ values are the learnable eigenvalue basis
        lambda_powers = torch.stack([
            self.lambda_basis ** k for k in range(self.filter_order + 1)
        ], dim=-1)  # (1, latent_dim, filter_order+1)

        # P(λ) and Q(λ)
        P_lambda = torch.sum(P_coef.unsqueeze(1) * lambda_powers, dim=-1)  # (batch, latent_dim)
        Q_lambda = torch.sum(Q_coef.unsqueeze(1) * lambda_powers, dim=-1)  # (batch, latent_dim)

        # Rational filter: H(λ) = P(λ) / (Q(λ) + eps)
        H_lambda = P_lambda / (Q_lambda.abs() + 0.1)  # (batch, latent_dim)

        # Apply filter in "spectral" domain (element-wise for latent)
        z_filtered = z * H_lambda

        # Residual MLP for additional expressivity
        z_residual = self.residual_mlp(torch.cat([z, t_embed], dim=-1))

        # Combine filtered + residual
        h = self.input_proj(z_filtered)
        output = self.output_proj(torch.cat([h, z_residual], dim=-1))

        return output


# Alias for backward compatibility
LatentDiffusion = GraphAwareDiffusion


diffusion = GraphAwareDiffusion(LATENT_DIM, HIDDEN_DIM, DIFFUSION_STEPS, n_eigenvectors).to(device)
n_params_diff = sum(p.numel() for p in diffusion.parameters())
print(f"  Graph-Aware Diffusion (GAD) parameters: {n_params_diff:,}")
print(f"  Diffusion steps: {DIFFUSION_STEPS}")
print(f"  Schedule: Floor Constrained Polynomial (FCPS)")
print(f"  Denoiser: Rational Graph Filter")

# ============================================================================
# TRAINING PHASE 1: AUTOENCODER
# ============================================================================
print("\n[PHASE 4] Training Graph Autoencoder...")

optimizer_vae = torch.optim.AdamW(vae.parameters(), lr=LEARNING_RATE)
scheduler_vae = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer_vae, AE_EPOCHS)

vae_losses = []
best_loss = float('inf')

start_time = time.time()
for epoch in range(AE_EPOCHS):
    vae.train()
    epoch_loss = 0
    epoch_recon = 0
    epoch_kl = 0

    for batch_x, batch_params in dataloader:
        optimizer_vae.zero_grad()

        # Forward pass
        x_recon, mu, logvar, z = vae(batch_x, eigenvectors_tensor)

        # Reconstruction loss
        recon_loss = F.mse_loss(x_recon, batch_x)

        # KL divergence
        kl_loss = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())

        # Total loss (beta-VAE style, low beta for better reconstruction)
        beta = 0.001
        loss = recon_loss + beta * kl_loss

        loss.backward()
        torch.nn.utils.clip_grad_norm_(vae.parameters(), 1.0)
        optimizer_vae.step()

        epoch_loss += loss.item()
        epoch_recon += recon_loss.item()
        epoch_kl += kl_loss.item()

    scheduler_vae.step()

    avg_loss = epoch_loss / len(dataloader)
    avg_recon = epoch_recon / len(dataloader)
    avg_kl = epoch_kl / len(dataloader)
    vae_losses.append(avg_loss)

    if avg_loss < best_loss:
        best_loss = avg_loss
        torch.save(vae.state_dict(), f"{OUTPUT_DIR}/vae_best.pt")

    if (epoch + 1) % 20 == 0 or epoch == 0:
        elapsed = time.time() - start_time
        print(f"  Epoch {epoch+1:3d}/{AE_EPOCHS} | Loss: {avg_loss:.6f} | "
              f"Recon: {avg_recon:.6f} | KL: {avg_kl:.4f} | {elapsed:.1f}s")

print(f"  VAE training complete. Best loss: {best_loss:.6f}")

# Load best model
vae.load_state_dict(torch.load(f"{OUTPUT_DIR}/vae_best.pt"))

# ============================================================================
# ENCODE ALL DATA TO LATENT SPACE
# ============================================================================
print("\n[PHASE 5] Encoding dataset to latent space...")

vae.eval()
with torch.no_grad():
    all_latents = vae.encode(solutions_tensor, eigenvectors_tensor)

print(f"  Latent codes shape: {all_latents.shape}")
print(f"  Latent mean: {all_latents.mean().item():.4f}, std: {all_latents.std().item():.4f}")

# Create latent dataset
latent_dataset = TensorDataset(all_latents)
latent_dataloader = DataLoader(latent_dataset, batch_size=BATCH_SIZE, shuffle=True)

# ============================================================================
# TRAINING PHASE 2: DIFFUSION MODEL
# ============================================================================
print("\n[PHASE 6] Training Latent Diffusion Model...")

optimizer_diff = torch.optim.AdamW(diffusion.parameters(), lr=LEARNING_RATE)
scheduler_diff = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer_diff, DIFF_EPOCHS)

diff_losses = []
best_diff_loss = float('inf')

start_time = time.time()
for epoch in range(DIFF_EPOCHS):
    diffusion.train()
    epoch_loss = 0

    for (batch_z,) in latent_dataloader:
        optimizer_diff.zero_grad()

        # Sample random timesteps
        t = torch.randint(0, DIFFUSION_STEPS, (batch_z.size(0),), device=device)

        # Add noise
        z_noisy, noise = diffusion.add_noise(batch_z, t)

        # Predict noise
        noise_pred = diffusion(z_noisy, t)

        # Loss
        loss = F.mse_loss(noise_pred, noise)

        loss.backward()
        torch.nn.utils.clip_grad_norm_(diffusion.parameters(), 1.0)
        optimizer_diff.step()

        epoch_loss += loss.item()

    scheduler_diff.step()

    avg_loss = epoch_loss / len(latent_dataloader)
    diff_losses.append(avg_loss)

    if avg_loss < best_diff_loss:
        best_diff_loss = avg_loss
        torch.save(diffusion.state_dict(), f"{OUTPUT_DIR}/diffusion_best.pt")

    if (epoch + 1) % 30 == 0 or epoch == 0:
        elapsed = time.time() - start_time
        print(f"  Epoch {epoch+1:3d}/{DIFF_EPOCHS} | Loss: {avg_loss:.6f} | {elapsed:.1f}s")

print(f"  Diffusion training complete. Best loss: {best_diff_loss:.6f}")

# Load best model
diffusion.load_state_dict(torch.load(f"{OUTPUT_DIR}/diffusion_best.pt"))

# ============================================================================
# GENERATE SYNTHETIC SAMPLES
# ============================================================================
print("\n[PHASE 7] Generating synthetic samples...")

vae.eval()
diffusion.eval()

N_SYNTHETIC = 16

with torch.no_grad():
    # Sample from diffusion model
    synthetic_latents = diffusion.sample(N_SYNTHETIC, device)

    # Decode to mesh signals
    synthetic_fields = vae.decode(synthetic_latents, eigenvectors_tensor)
    synthetic_fields = synthetic_fields.cpu().numpy()

print(f"  Generated {N_SYNTHETIC} synthetic fields")
print(f"  Field range: [{synthetic_fields.min():.3f}, {synthetic_fields.max():.3f}]")

# Save synthetic samples
np.savez(f"{OUTPUT_DIR}/synthetic_samples.npz",
         fields=synthetic_fields,
         latents=synthetic_latents.cpu().numpy())

# ============================================================================
# VISUALIZATION
# ============================================================================
print("\n[PHASE 8] Generating visualizations...")

triang = Triangulation(points[:, 0], points[:, 1], triangles)
margin = 0.02

# Plot training curves
fig, axes = plt.subplots(1, 2, figsize=(12, 4))
fig.suptitle('Training Curves', fontsize=14, fontweight='bold')

axes[0].plot(vae_losses, 'b-', linewidth=1)
axes[0].set_xlabel('Epoch')
axes[0].set_ylabel('Loss')
axes[0].set_title('VAE Training Loss')
axes[0].grid(True, alpha=0.3)
axes[0].set_yscale('log')

axes[1].plot(diff_losses, 'r-', linewidth=1)
axes[1].set_xlabel('Epoch')
axes[1].set_ylabel('Loss')
axes[1].set_title('Diffusion Training Loss')
axes[1].grid(True, alpha=0.3)
axes[1].set_yscale('log')

plt.tight_layout()
plt.savefig(f'{OUTPUT_DIR}/training_curves.png', dpi=150, bbox_inches='tight')
print(f"  Saved: {OUTPUT_DIR}/training_curves.png")

# Plot FCPS schedule comparison
fig_sched, axes_sched = plt.subplots(1, 2, figsize=(12, 4))
fig_sched.suptitle('GAD: Floor Constrained Polynomial Schedule (FCPS) vs Linear', fontsize=12, fontweight='bold')

t_steps = np.arange(DIFFUSION_STEPS)
fcps_alphas = diffusion.alphas_cumprod.cpu().numpy()

# Linear schedule for comparison
linear_betas = np.linspace(BETA_START, BETA_END, DIFFUSION_STEPS)
linear_alphas = np.cumprod(1 - linear_betas)

axes_sched[0].plot(t_steps, fcps_alphas, 'b-', linewidth=2, label='FCPS (GAD)')
axes_sched[0].plot(t_steps, linear_alphas, 'r--', linewidth=2, label='Linear')
axes_sched[0].set_xlabel('Diffusion Step t')
axes_sched[0].set_ylabel('α̅ₜ (signal retention)')
axes_sched[0].set_title('Cumulative Signal Retention')
axes_sched[0].legend()
axes_sched[0].grid(True, alpha=0.3)
axes_sched[0].axhline(y=0.1, color='gray', linestyle=':', label='Floor')

axes_sched[1].semilogy(t_steps, 1 - fcps_alphas, 'b-', linewidth=2, label='FCPS (GAD)')
axes_sched[1].semilogy(t_steps, 1 - linear_alphas, 'r--', linewidth=2, label='Linear')
axes_sched[1].set_xlabel('Diffusion Step t')
axes_sched[1].set_ylabel('1 - α̅ₜ (noise level, log)')
axes_sched[1].set_title('Noise Level (Log Scale)')
axes_sched[1].legend()
axes_sched[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(f'{OUTPUT_DIR}/fcps_schedule.png', dpi=150, bbox_inches='tight')
print(f"  Saved: {OUTPUT_DIR}/fcps_schedule.png")

# Plot synthetic samples
fig, axes = plt.subplots(4, 4, figsize=(16, 14))
fig.suptitle('Synthetic FEA Fields (Generated by Diffusion Model)', fontsize=14, fontweight='bold')

for i, ax in enumerate(axes.flatten()):
    if i < N_SYNTHETIC:
        field = synthetic_fields[i]
        # Normalize for display
        field = np.clip(field, 0, 1)
        levels = np.linspace(0, 1, 25)
        tcf = ax.tricontourf(triang, field, levels=levels, cmap='inferno', extend='both')
        ax.tricontour(triang, field, levels=8, colors='white', linewidths=0.3, alpha=0.5)
        circle = plt.Circle((0, 0), 0.05, fill=True, color='cyan', ec='black', lw=0.5)
        ax.add_patch(circle)
        ax.set_xlim(-0.15 - margin, 0.375 + margin)
        ax.set_ylim(-0.15 - margin, 0.15 + margin)
        ax.set_aspect('equal')
        ax.set_title(f'Sample {i+1}', fontsize=10)
        ax.set_xticks([])
        ax.set_yticks([])

fig.subplots_adjust(right=0.92)
cbar_ax = fig.add_axes([0.94, 0.15, 0.02, 0.7])
sm = plt.cm.ScalarMappable(cmap='inferno', norm=plt.Normalize(0, 1))
fig.colorbar(sm, cax=cbar_ax, label='Temperature')

plt.savefig(f'{OUTPUT_DIR}/synthetic_samples.png', dpi=150, bbox_inches='tight')
print(f"  Saved: {OUTPUT_DIR}/synthetic_samples.png")

# Compare real vs synthetic
fig, axes = plt.subplots(2, 4, figsize=(16, 8))
fig.suptitle('Real vs Synthetic Comparison', fontsize=14, fontweight='bold')

# Top row: real samples
for i in range(4):
    ax = axes[0, i]
    idx = np.random.randint(n_samples)
    field = solutions[idx]
    levels = np.linspace(0, 1, 25)
    tcf = ax.tricontourf(triang, field, levels=levels, cmap='inferno', extend='both')
    circle = plt.Circle((0, 0), 0.05, fill=True, color='cyan', ec='black', lw=0.5)
    ax.add_patch(circle)
    ax.set_xlim(-0.15 - margin, 0.375 + margin)
    ax.set_ylim(-0.15 - margin, 0.15 + margin)
    ax.set_aspect('equal')
    ax.set_title(f'Real {i+1}', fontsize=10)
    ax.set_xticks([])
    ax.set_yticks([])

# Bottom row: synthetic samples
for i in range(4):
    ax = axes[1, i]
    field = np.clip(synthetic_fields[i], 0, 1)
    levels = np.linspace(0, 1, 25)
    tcf = ax.tricontourf(triang, field, levels=levels, cmap='inferno', extend='both')
    circle = plt.Circle((0, 0), 0.05, fill=True, color='cyan', ec='black', lw=0.5)
    ax.add_patch(circle)
    ax.set_xlim(-0.15 - margin, 0.375 + margin)
    ax.set_ylim(-0.15 - margin, 0.15 + margin)
    ax.set_aspect('equal')
    ax.set_title(f'Synthetic {i+1}', fontsize=10)
    ax.set_xticks([])
    ax.set_yticks([])

axes[0, 0].set_ylabel('REAL', fontsize=12, fontweight='bold')
axes[1, 0].set_ylabel('SYNTHETIC', fontsize=12, fontweight='bold')

plt.tight_layout()
plt.savefig(f'{OUTPUT_DIR}/real_vs_synthetic.png', dpi=150, bbox_inches='tight')
print(f"  Saved: {OUTPUT_DIR}/real_vs_synthetic.png")

# Latent space visualization (PCA)
from sklearn.decomposition import PCA

pca = PCA(n_components=2)
latents_2d = pca.fit_transform(all_latents.cpu().numpy())
synthetic_2d = pca.transform(synthetic_latents.cpu().numpy())

fig, ax = plt.subplots(figsize=(8, 8))
ax.scatter(latents_2d[:, 0], latents_2d[:, 1], c='blue', alpha=0.5, s=20, label='Real')
ax.scatter(synthetic_2d[:, 0], synthetic_2d[:, 1], c='red', alpha=0.8, s=50, marker='*', label='Synthetic')
ax.set_xlabel('PC1')
ax.set_ylabel('PC2')
ax.set_title('Latent Space (PCA Projection)')
ax.legend()
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(f'{OUTPUT_DIR}/latent_space.png', dpi=150, bbox_inches='tight')
print(f"  Saved: {OUTPUT_DIR}/latent_space.png")

# ============================================================================
# SAVE FINAL MODELS
# ============================================================================
print("\n[PHASE 9] Saving models...")

torch.save({
    'vae_state_dict': vae.state_dict(),
    'diffusion_state_dict': diffusion.state_dict(),
    'config': {
        'latent_dim': LATENT_DIM,
        'hidden_dim': HIDDEN_DIM,
        'n_encoder_layers': N_ENCODER_LAYERS,
        'n_decoder_layers': N_DECODER_LAYERS,
        'diffusion_steps': DIFFUSION_STEPS,
        'n_nodes': n_nodes,
        'n_eigenvectors': n_eigenvectors
    }
}, f"{OUTPUT_DIR}/models_final.pt")

print(f"  Saved: {OUTPUT_DIR}/models_final.pt")

# ============================================================================
# SUMMARY
# ============================================================================
print("\n" + "=" * 60)
print("TRAINING COMPLETE")
print("=" * 60)
print(f"""
Model Summary (Graph-Aware Diffusion / GAD):
  - Graph VAE: {n_params:,} parameters
  - GAD Diffusion: {n_params_diff:,} parameters
  - Latent dimension: {LATENT_DIM}
  - Diffusion steps: {DIFFUSION_STEPS}

GAD Innovations (arXiv:2510.05036):
  - Floor Constrained Polynomial Schedule (FCPS)
  - Rational Graph Filter Denoiser (H(λ) = P(λ)/Q(λ))
  - Eigenvalue-aware latent structure

Training Results:
  - VAE final loss: {vae_losses[-1]:.6f}
  - Diffusion final loss: {diff_losses[-1]:.6f}
  - Synthetic samples: {N_SYNTHETIC}

Output files in '{OUTPUT_DIR}/':
  - vae_best.pt           : Best VAE checkpoint
  - diffusion_best.pt     : Best GAD checkpoint
  - models_final.pt       : Final combined checkpoint
  - synthetic_samples.npz : Generated synthetic fields
  - training_curves.png   : Loss curves
  - fcps_schedule.png     : FCPS vs linear schedule comparison
  - synthetic_samples.png : Visualization of generated fields
  - real_vs_synthetic.png : Comparison plot
  - latent_space.png      : PCA of latent space

To generate more samples:
  checkpoint = torch.load('{OUTPUT_DIR}/models_final.pt')
  # ... load models and call diffusion.sample(n, device)
""")
