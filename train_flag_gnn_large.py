#!/usr/bin/env python3
"""
Large GNN-based diffusion that matches the HuggingFace UNet2D architecture.

Architecture mirrors the CNN:
- Channel progression: 64 → 128 → 256 → 256
- 2 layers per block
- Attention at 256-channel level
- Skip connections between encoder/decoder
- ~25M parameters (matching the CNN)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import time
import math

try:
    from torch_geometric.nn import MessagePassing
    HAS_PYG = True
except ImportError:
    HAS_PYG = False
    print("Warning: torch_geometric not found, using fallback implementation")


# =============================================================================
# Graph Construction
# =============================================================================

def mesh_to_graph(cells, pos):
    """Convert triangle mesh to graph with edge features."""
    edges = set()
    for tri in cells:
        for i in range(3):
            for j in range(3):
                if i != j:
                    edges.add((int(tri[i]), int(tri[j])))

    edge_index = torch.tensor(list(edges), dtype=torch.long).T
    src, dst = edge_index
    edge_vec = pos[dst] - pos[src]
    edge_len = edge_vec.norm(dim=-1, keepdim=True)
    edge_attr = torch.cat([edge_vec, edge_len], dim=-1)

    return edge_index, edge_attr


# =============================================================================
# Model Components
# =============================================================================

class MeshConv(nn.Module):
    """Graph convolution with edge features."""

    def __init__(self, in_dim, out_dim, edge_dim=4):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(in_dim * 2 + edge_dim, out_dim),
            nn.GroupNorm(8, out_dim),
            nn.SiLU(),
            nn.Linear(out_dim, out_dim),
        )
        self.out_dim = out_dim

    def forward(self, x, edge_index, edge_attr):
        src, dst = edge_index
        x_src = x[src]
        x_dst = x[dst]

        messages = self.mlp(torch.cat([x_dst, x_src, edge_attr], dim=-1))

        out = torch.zeros(x.size(0), self.out_dim, device=x.device, dtype=x.dtype)
        count = torch.zeros(x.size(0), 1, device=x.device, dtype=x.dtype)

        out.scatter_add_(0, dst.unsqueeze(-1).expand(-1, self.out_dim), messages)
        count.scatter_add_(0, dst.unsqueeze(-1), torch.ones_like(dst, dtype=x.dtype).unsqueeze(-1))

        return out / count.clamp(min=1)


class GraphAttention(nn.Module):
    """Multi-head graph attention."""

    def __init__(self, dim, num_heads=8, edge_dim=4):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5

        self.q_proj = nn.Linear(dim, dim)
        self.k_proj = nn.Linear(dim, dim)
        self.v_proj = nn.Linear(dim, dim)
        self.edge_proj = nn.Linear(edge_dim, num_heads)
        self.out_proj = nn.Linear(dim, dim)

        self.norm = nn.GroupNorm(8, dim)

    def forward(self, x, edge_index, edge_attr):
        B_V, D = x.shape
        src, dst = edge_index

        # Project queries, keys, values
        q = self.q_proj(x).view(B_V, self.num_heads, self.head_dim)
        k = self.k_proj(x).view(B_V, self.num_heads, self.head_dim)
        v = self.v_proj(x).view(B_V, self.num_heads, self.head_dim)

        # Compute attention scores for edges
        q_dst = q[dst]  # (E, H, D_h)
        k_src = k[src]  # (E, H, D_h)

        attn = (q_dst * k_src).sum(dim=-1) * self.scale  # (E, H)

        # Add edge bias
        edge_bias = self.edge_proj(edge_attr)  # (E, H)
        attn = attn + edge_bias

        # Softmax over incoming edges (per destination node)
        attn_max = torch.zeros(B_V, self.num_heads, device=x.device, dtype=x.dtype)
        attn_max.scatter_reduce_(0, dst.unsqueeze(-1).expand(-1, self.num_heads),
                                  attn, reduce='amax', include_self=False)
        attn = torch.exp(attn - attn_max[dst])

        attn_sum = torch.zeros(B_V, self.num_heads, device=x.device, dtype=x.dtype)
        attn_sum.scatter_add_(0, dst.unsqueeze(-1).expand(-1, self.num_heads), attn)
        attn = attn / attn_sum[dst].clamp(min=1e-6)

        # Aggregate values
        v_src = v[src]  # (E, H, D_h)
        weighted_v = attn.unsqueeze(-1) * v_src  # (E, H, D_h)

        out = torch.zeros(B_V, self.num_heads, self.head_dim, device=x.device, dtype=x.dtype)
        out.scatter_add_(0, dst.unsqueeze(-1).unsqueeze(-1).expand(-1, self.num_heads, self.head_dim), weighted_v)

        out = out.view(B_V, D)
        out = self.out_proj(out)
        out = self.norm(out)

        return out


class ResBlock(nn.Module):
    """Residual block with two conv layers and timestep conditioning."""

    def __init__(self, in_dim, out_dim, time_dim, edge_dim=4):
        super().__init__()
        self.conv1 = MeshConv(in_dim, out_dim, edge_dim)
        self.conv2 = MeshConv(out_dim, out_dim, edge_dim)
        self.time_mlp = nn.Sequential(
            nn.SiLU(),
            nn.Linear(time_dim, out_dim),
        )
        self.norm1 = nn.GroupNorm(8, out_dim)
        self.norm2 = nn.GroupNorm(8, out_dim)

        if in_dim != out_dim:
            self.skip = nn.Linear(in_dim, out_dim)
        else:
            self.skip = nn.Identity()

    def forward(self, x, edge_index, edge_attr, t_emb):
        h = self.conv1(x, edge_index, edge_attr)
        h = self.norm1(h)
        h = F.silu(h)

        # Add timestep
        h = h + self.time_mlp(t_emb)

        h = self.conv2(h, edge_index, edge_attr)
        h = self.norm2(h)
        h = F.silu(h)

        return h + self.skip(x)


class AttnBlock(nn.Module):
    """Residual block with attention."""

    def __init__(self, dim, time_dim, edge_dim=4, num_heads=8):
        super().__init__()
        self.res_block = ResBlock(dim, dim, time_dim, edge_dim)
        self.attn = GraphAttention(dim, num_heads, edge_dim)

    def forward(self, x, edge_index, edge_attr, t_emb):
        h = self.res_block(x, edge_index, edge_attr, t_emb)
        h = h + self.attn(h, edge_index, edge_attr)
        return h


class TimestepEmbedding(nn.Module):
    """Sinusoidal timestep embedding."""

    def __init__(self, dim, time_dim):
        super().__init__()
        self.dim = dim
        self.mlp = nn.Sequential(
            nn.Linear(dim, time_dim),
            nn.SiLU(),
            nn.Linear(time_dim, time_dim),
        )

    def forward(self, t):
        half = self.dim // 2
        freqs = torch.exp(-math.log(10000) * torch.arange(half, device=t.device) / half)
        args = t[:, None].float() * freqs[None, :]
        emb = torch.cat([torch.sin(args), torch.cos(args)], dim=-1)
        return self.mlp(emb)


# =============================================================================
# Large GNN Model (matching UNet2D architecture)
# =============================================================================

class GraphUNet(nn.Module):
    """
    GNN architecture matching HuggingFace UNet2DModel.

    Channels: 64 → 128 → 256 → 256
    2 ResBlocks per level
    Attention at 256-channel level
    Skip connections between encoder/decoder
    """

    def __init__(self, cells, mesh_pos,
                 block_channels=(64, 128, 256, 256),
                 layers_per_block=2,
                 time_dim=256,
                 edge_dim=4,
                 batch_size=32):
        super().__init__()

        # Precompute graph structure
        edge_index, edge_attr = mesh_to_graph(cells, mesh_pos)
        self.register_buffer('edge_index', edge_index)
        self.register_buffer('edge_attr', edge_attr)

        self.num_vertices = mesh_pos.shape[0]
        self.block_channels = block_channels

        # Precompute batched edges
        V = mesh_pos.shape[0]
        offsets = torch.arange(batch_size) * V
        edge_index_batched = torch.cat([edge_index + offset for offset in offsets], dim=1)
        edge_attr_batched = edge_attr.repeat(batch_size, 1)
        self.register_buffer('edge_index_batched', edge_index_batched)
        self.register_buffer('edge_attr_batched', edge_attr_batched)
        self.precomputed_batch_size = batch_size

        # Time embedding
        self.time_embed = TimestepEmbedding(128, time_dim)

        # Input projection
        self.input_proj = nn.Linear(3, block_channels[0])

        # Encoder blocks
        self.encoder_blocks = nn.ModuleList()
        in_ch = block_channels[0]
        for i, out_ch in enumerate(block_channels):
            block_list = nn.ModuleList()
            for j in range(layers_per_block):
                ch_in = in_ch if j == 0 else out_ch
                # Use attention at highest channel levels (256)
                if out_ch >= 256:
                    block_list.append(AttnBlock(ch_in if j == 0 else out_ch, time_dim, edge_dim))
                    if j == 0 and ch_in != out_ch:
                        block_list[-1] = nn.Sequential(
                            nn.Linear(ch_in, out_ch),
                            AttnBlock(out_ch, time_dim, edge_dim)
                        )
                else:
                    block_list.append(ResBlock(ch_in, out_ch, time_dim, edge_dim))
            self.encoder_blocks.append(block_list)
            in_ch = out_ch

        # Middle block
        mid_ch = block_channels[-1]
        self.mid_block1 = AttnBlock(mid_ch, time_dim, edge_dim)
        self.mid_block2 = AttnBlock(mid_ch, time_dim, edge_dim)

        # Decoder blocks (with skip connections)
        self.decoder_blocks = nn.ModuleList()
        reversed_channels = list(reversed(block_channels))
        for i, out_ch in enumerate(reversed_channels):
            block_list = nn.ModuleList()
            in_ch = reversed_channels[i-1] if i > 0 else mid_ch
            # Skip connection doubles the channels
            skip_ch = block_channels[len(block_channels) - 1 - i]

            for j in range(layers_per_block):
                if j == 0:
                    ch_in = in_ch + skip_ch  # Concatenate skip
                else:
                    ch_in = out_ch

                if out_ch >= 256:
                    block_list.append(AttnBlock(out_ch, time_dim, edge_dim))
                    if j == 0:
                        # Need projection for skip concatenation
                        block_list[-1] = nn.ModuleList([
                            nn.Linear(ch_in, out_ch),
                            AttnBlock(out_ch, time_dim, edge_dim)
                        ])
                else:
                    block_list.append(ResBlock(ch_in, out_ch, time_dim, edge_dim))
            self.decoder_blocks.append(block_list)

        # Output projection
        self.output_norm = nn.GroupNorm(8, block_channels[0])
        self.output_proj = nn.Linear(block_channels[0], 3)

    def forward(self, x, t):
        """
        x: (B, V, 3) noisy signal
        t: (B,) timestep
        """
        B, V, C = x.shape

        # Get batched edges
        if B == self.precomputed_batch_size:
            edge_index = self.edge_index_batched
            edge_attr = self.edge_attr_batched
        else:
            offsets = torch.arange(B, device=x.device) * V
            edge_index = torch.cat([self.edge_index + o for o in offsets], dim=1)
            edge_attr = self.edge_attr.repeat(B, 1)

        # Flatten batch
        x = x.reshape(B * V, C)

        # Time embedding (expand to all vertices)
        t_emb = self.time_embed(t)  # (B, time_dim)
        t_emb = t_emb.unsqueeze(1).expand(-1, V, -1).reshape(B * V, -1)  # (B*V, time_dim)

        # Input projection
        h = self.input_proj(x)

        # Encoder with skip connections
        skips = []
        for blocks in self.encoder_blocks:
            for block in blocks:
                if isinstance(block, nn.Sequential):
                    # Linear + AttnBlock
                    h = block[0](h)
                    h = block[1](h, edge_index, edge_attr, t_emb)
                elif isinstance(block, (ResBlock, AttnBlock)):
                    h = block(h, edge_index, edge_attr, t_emb)
            skips.append(h)

        # Middle
        h = self.mid_block1(h, edge_index, edge_attr, t_emb)
        h = self.mid_block2(h, edge_index, edge_attr, t_emb)

        # Decoder with skip connections
        for i, blocks in enumerate(self.decoder_blocks):
            skip = skips[len(skips) - 1 - i]
            h = torch.cat([h, skip], dim=-1)

            for j, block in enumerate(blocks):
                if isinstance(block, nn.ModuleList):
                    # Linear projection + AttnBlock
                    h = block[0](h)
                    h = block[1](h, edge_index, edge_attr, t_emb)
                elif isinstance(block, (ResBlock, AttnBlock)):
                    h = block(h, edge_index, edge_attr, t_emb)

        # Output
        h = self.output_norm(h)
        h = F.silu(h)
        out = self.output_proj(h)

        return out.reshape(B, V, 3)


# =============================================================================
# Dataset (same as original)
# =============================================================================

class FlagFrameDataset(Dataset):
    """Dataset of individual flag frames."""

    def __init__(self, frames):
        self.data_min = frames.min()
        self.data_max = frames.max()
        frames = 2 * (frames - self.data_min) / (self.data_max - self.data_min) - 1
        self.frames = torch.tensor(frames, dtype=torch.float32)
        print(f"Dataset: {len(self)} frames, {frames.shape[1]} vertices")
        print(f"  Normalized to [-1, 1] (raw range: [{self.data_min:.3f}, {self.data_max:.3f}])")

    def __len__(self):
        return len(self.frames)

    def __getitem__(self, idx):
        return self.frames[idx]

    def denormalize(self, x):
        return (x + 1) / 2 * (self.data_max - self.data_min) + self.data_min


# =============================================================================
# Diffusion Schedule
# =============================================================================

def cosine_beta_schedule(num_steps, s=0.008):
    steps = num_steps + 1
    x = torch.linspace(0, num_steps, steps)
    alphas_cumprod = torch.cos(((x / num_steps) + s) / (1 + s) * math.pi * 0.5) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    return torch.clip(betas, 0.0001, 0.9999)


class DiffusionSchedule:
    def __init__(self, num_steps=1000, device='cpu'):
        betas = cosine_beta_schedule(num_steps)
        alphas = 1.0 - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)
        alphas_cumprod_prev = F.pad(alphas_cumprod[:-1], (1, 0), value=1.0)

        self.sqrt_alphas_cumprod = torch.sqrt(alphas_cumprod).to(device)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1.0 - alphas_cumprod).to(device)
        self.sqrt_recip_alphas = torch.sqrt(1.0 / alphas).to(device)
        self.posterior_variance = (betas * (1.0 - alphas_cumprod_prev) / (1.0 - alphas_cumprod)).to(device)
        self.betas = betas.to(device)
        self.num_steps = num_steps

    def q_sample(self, x_0, n, noise=None):
        if noise is None:
            noise = torch.randn_like(x_0)
        sqrt_alpha = self.sqrt_alphas_cumprod[n]
        sqrt_one_minus = self.sqrt_one_minus_alphas_cumprod[n]
        while sqrt_alpha.dim() < x_0.dim():
            sqrt_alpha = sqrt_alpha.unsqueeze(-1)
            sqrt_one_minus = sqrt_one_minus.unsqueeze(-1)
        return sqrt_alpha * x_0 + sqrt_one_minus * noise

    @torch.no_grad()
    def p_sample(self, model, x_n, n):
        n_tensor = torch.full((x_n.shape[0],), n, device=x_n.device, dtype=torch.long)
        predicted_noise = model(x_n, n_tensor)
        beta = self.betas[n]
        sqrt_recip = self.sqrt_recip_alphas[n]
        sqrt_one_minus = self.sqrt_one_minus_alphas_cumprod[n]
        mean = sqrt_recip * (x_n - beta / sqrt_one_minus * predicted_noise)
        if n > 0:
            noise = torch.randn_like(x_n)
            std = self.posterior_variance[n].sqrt()
            return mean + std * noise
        return mean


# =============================================================================
# Training
# =============================================================================

def train_epoch(model, loader, optimizer, schedule, device, scaler=None):
    model.train()
    total_loss = 0
    for batch in loader:
        batch = batch.to(device)
        B = batch.shape[0]
        n = torch.randint(0, schedule.num_steps, (B,), device=device)
        noise = torch.randn_like(batch)
        x_n = schedule.q_sample(batch, n, noise)

        optimizer.zero_grad(set_to_none=True)
        if scaler:
            with torch.amp.autocast('cuda'):
                predicted = model(x_n, n)
                loss = F.mse_loss(predicted, noise)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            predicted = model(x_n, n)
            loss = F.mse_loss(predicted, noise)
            loss.backward()
            optimizer.step()

        total_loss += loss.item()
    return total_loss / len(loader)


@torch.no_grad()
def evaluate(model, loader, schedule, device):
    model.eval()
    total_loss = 0
    for batch in loader:
        batch = batch.to(device)
        B = batch.shape[0]
        n = torch.randint(0, schedule.num_steps, (B,), device=device)
        noise = torch.randn_like(batch)
        x_n = schedule.q_sample(batch, n, noise)
        predicted = model(x_n, n)
        loss = F.mse_loss(predicted, noise)
        total_loss += loss.item()
    return total_loss / len(loader)


@torch.no_grad()
def generate_samples(model, schedule, num_samples, num_vertices, device):
    model.eval()
    x = torch.randn(num_samples, num_vertices, 3, device=device)
    for n in reversed(range(schedule.num_steps)):
        x = schedule.p_sample(model, x, n)
        if n % 200 == 0:
            print(f"  Step {n}...")
    return x


# =============================================================================
# Main
# =============================================================================

def main():
    print("=" * 60)
    print("Large GNN Diffusion (Matching UNet2D Architecture)")
    print("=" * 60)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Device: {device}")

    # Config
    batch_size = 32
    num_epochs = 100
    learning_rate = 1e-4
    num_steps = 1000

    output_dir = Path('flag_gnn_large_output')
    output_dir.mkdir(exist_ok=True)

    # Load data
    print("\nLoading data...")
    data_path = Path('flag_data/flag_test.npz')
    data = np.load(data_path)
    world_pos = data['world_pos']
    cells = data['cells']
    mesh_pos = data['mesh_pos']

    # Convert 2D mesh_pos to 3D
    if mesh_pos.shape[1] == 2:
        mesh_pos_3d = np.zeros((mesh_pos.shape[0], 3), dtype=np.float32)
        mesh_pos_3d[:, :2] = mesh_pos
        mesh_pos = mesh_pos_3d

    N, T, V, C = world_pos.shape
    frames = world_pos.reshape(-1, V, C)
    print(f"  Total frames: {len(frames)}, Vertices: {V}")

    # Split randomly (not sequentially by trajectory)
    n_train = int(0.9 * len(frames))
    np.random.seed(42)
    indices = np.random.permutation(len(frames))
    train_idx = indices[:n_train]
    val_idx = indices[n_train:]
    train_dataset = FlagFrameDataset(frames[train_idx])
    val_dataset = FlagFrameDataset(frames[val_idx])

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True,
                              num_workers=0, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False,
                            num_workers=0, pin_memory=True)

    # Model
    print("\nBuilding Large GNN (matching UNet2D)...")
    cells_tensor = torch.tensor(cells, dtype=torch.long)
    mesh_pos_tensor = torch.tensor(mesh_pos, dtype=torch.float32)

    model = GraphUNet(
        cells=cells_tensor,
        mesh_pos=mesh_pos_tensor,
        block_channels=(64, 128, 256, 256),
        layers_per_block=2,
        time_dim=256,
        edge_dim=4,
        batch_size=batch_size,
    ).to(device)

    num_params = sum(p.numel() for p in model.parameters())
    print(f"  Parameters: {num_params:,}")

    schedule = DiffusionSchedule(num_steps=num_steps, device=device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

    # Training
    print("\nTraining...")
    print("-" * 60)

    scaler = torch.amp.GradScaler('cuda') if device == 'cuda' else None
    if scaler:
        print("Using mixed precision (AMP)")

    train_losses = []
    val_losses = []
    best_val_loss = float('inf')

    for epoch in range(1, num_epochs + 1):
        start = time.time()
        train_loss = train_epoch(model, train_loader, optimizer, schedule, device, scaler)
        val_loss = evaluate(model, val_loader, schedule, device)
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        elapsed = time.time() - start

        print(f"Epoch {epoch:3d}/{num_epochs} | Train: {train_loss:.4f} | Val: {val_loss:.4f} | {elapsed:.1f}s")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'train_loss': train_loss,
                'val_loss': val_loss,
            }, output_dir / 'best_model.pt')

        if epoch % 20 == 0 or epoch == 1:
            print("  Generating samples...")
            generated = generate_samples(model, schedule, 6, V, device)
            generated_np = train_dataset.denormalize(generated).cpu().numpy()

            # Visualize
            from mpl_toolkits.mplot3d.art3d import Poly3DCollection
            fig = plt.figure(figsize=(18, 6))
            for i in range(6):
                ax = fig.add_subplot(1, 6, i + 1, projection='3d')
                triangles = generated_np[i][cells]
                mesh = Poly3DCollection(triangles, alpha=0.6, facecolor='darkorange',
                                        edgecolor='k', linewidth=0.1)
                ax.add_collection3d(mesh)
                ax.set_xlim(generated_np[i, :, 0].min(), generated_np[i, :, 0].max())
                ax.set_ylim(generated_np[i, :, 1].min(), generated_np[i, :, 1].max())
                ax.set_zlim(generated_np[i, :, 2].min(), generated_np[i, :, 2].max())
                ax.set_title(f'#{i+1}')
                ax.view_init(elev=20, azim=45)
            plt.suptitle(f'Large GNN Generated (Epoch {epoch})')
            plt.tight_layout()
            plt.savefig(output_dir / f'samples_epoch{epoch}.png', dpi=150)
            plt.close()
            print(f"  Saved: {output_dir / f'samples_epoch{epoch}.png'}")

    # Training curves
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    ax1.plot(train_losses, label='Train')
    ax1.plot(val_losses, label='Val')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax2.semilogy(train_losses, label='Train')
    ax2.semilogy(val_losses, label='Val')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Loss (log)')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_dir / 'training_curves.png', dpi=150)
    plt.close()

    print("\n" + "=" * 60)
    print("Training complete!")
    print(f"Best validation loss: {best_val_loss:.4f}")
    print(f"Output: {output_dir}")
    print("=" * 60)


if __name__ == '__main__':
    main()
