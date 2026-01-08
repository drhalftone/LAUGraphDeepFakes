# Differential Coordinates for Physics-Informed Mesh Diffusion

## Motivation

### The Problem with Absolute Positions

The current approach models vertex positions directly in R^3:

```
Model predicts: pos_i ∈ R^3 for each vertex i
```

This has issues:
- **Not translation invariant**: Moving the mesh changes all positions
- **Edge constraints are global**: Enforcing edge lengths requires coordinating distant vertices
- **Physics is implicit**: The model must learn mesh structure from scratch

### The Key Insight

Instead of modeling absolute positions, model **relative positions between neighbors**:

```
Model predicts: δ_ij = pos_j - pos_i for each edge (i,j)
```

This makes physics constraints **local and easy to enforce**.

## Why Differential Coordinates Work Better

| Property | Absolute Positions | Edge Deltas |
|----------|-------------------|-------------|
| Translation invariant | No | Yes |
| Edge length constraint | Global, implicit | Local: \|\|δ_ij\|\| = length |
| Stretch limit | Hard to enforce | Just clamp \|\|δ\|\| |
| Boundary condition | Fix vertex at (x,y,z) | One anchor vertex |
| Physics meaning | None | Direct (local deformation) |

### Edge Length Becomes Trivial

```python
# With absolute positions - global constraint, hard to enforce
constraint: ||pos_i - pos_j|| ≈ rest_length_ij

# With edge deltas - just normalize!
δ_constrained = normalize(δ_predicted) * rest_length
```

## Architecture: Edge Delta Diffusion

### Overview

```
Training:
  positions → edge deltas → add noise → predict noise → loss

Generation:
  noise → denoise (with constraints) → edge deltas → reconstruct positions
```

### Data Representation

```python
# Current: vertex positions
pos: (V, 3)  # V vertices, xyz coordinates

# Proposed: edge deltas
deltas: (E, 3)  # E edges, delta xyz
```

### Converting Between Representations

```python
def positions_to_deltas(pos, edge_index):
    """Convert vertex positions to edge deltas."""
    src, dst = edge_index
    deltas = pos[dst] - pos[src]  # (E, 3)
    return deltas

def deltas_to_positions(deltas, edge_index, anchor_idx=0, anchor_pos=None):
    """Reconstruct positions from edge deltas via BFS."""
    if anchor_pos is None:
        anchor_pos = torch.zeros(3)

    num_vertices = edge_index.max() + 1
    pos = torch.zeros(num_vertices, 3)
    pos[anchor_idx] = anchor_pos
    visited = {anchor_idx}

    # Build adjacency for traversal
    adj = defaultdict(list)
    src, dst = edge_index
    for i, (s, d) in enumerate(zip(src, dst)):
        adj[s.item()].append((d.item(), i, +1))  # Forward edge
        adj[d.item()].append((s.item(), i, -1))  # Reverse edge

    # BFS reconstruction
    queue = [anchor_idx]
    while queue:
        i = queue.pop(0)
        for j, edge_idx, sign in adj[i]:
            if j not in visited:
                pos[j] = pos[i] + sign * deltas[edge_idx]
                visited.add(j)
                queue.append(j)

    return pos
```

### Model Architecture

```python
class EdgeDeltaDiffusion(nn.Module):
    """Diffusion model operating on edge deltas instead of vertex positions."""

    def __init__(self, cells, mesh_pos, hidden_dim=128, num_layers=4):
        super().__init__()

        # Build graph and compute rest state
        edge_index, edge_attr = mesh_to_graph(cells, mesh_pos)
        self.register_buffer('edge_index', edge_index)

        # Rest state deltas (what we want edges to be)
        src, dst = edge_index
        rest_deltas = mesh_pos[dst] - mesh_pos[src]
        self.register_buffer('rest_deltas', rest_deltas)
        self.register_buffer('rest_lengths', rest_deltas.norm(dim=-1))

        self.num_edges = edge_index.shape[1]
        self.num_vertices = mesh_pos.shape[0]

        # Edge-centric network
        # Edges communicate via shared vertices
        self.input_proj = nn.Linear(3, hidden_dim)  # Delta xyz → hidden

        self.layers = nn.ModuleList([
            EdgeConvLayer(hidden_dim) for _ in range(num_layers)
        ])

        self.time_embed = TimestepEmbedding(hidden_dim)
        self.output_proj = nn.Linear(hidden_dim, 3)  # hidden → delta noise

    def forward(self, delta_noisy, n):
        """
        delta_noisy: (B, E, 3) noisy edge deltas
        n: (B,) noise timestep

        returns: (B, E, 3) predicted noise on deltas
        """
        B, E, _ = delta_noisy.shape

        # Embed deltas
        h = self.input_proj(delta_noisy)  # (B, E, D)

        # Message passing between edges (via shared vertices)
        for layer in self.layers:
            h = h + layer(h, self.edge_index, self.num_vertices)

        # Add timestep
        t_emb = self.time_embed(n)  # (B, D)
        h = h + t_emb.unsqueeze(1)  # (B, E, D)

        # Predict noise
        noise_pred = self.output_proj(h)  # (B, E, 3)
        return noise_pred


class EdgeConvLayer(nn.Module):
    """Message passing between edges that share vertices."""

    def __init__(self, hidden_dim):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

    def forward(self, h_edges, edge_index, num_vertices):
        """
        h_edges: (B, E, D) edge features
        edge_index: (2, E) vertex indices for each edge

        Edges communicate through shared vertices:
        edge_ij aggregates info from all edges incident to i and j
        """
        B, E, D = h_edges.shape
        src, dst = edge_index

        # Aggregate edge features to vertices
        vertex_features = torch.zeros(B, num_vertices, D, device=h_edges.device)
        # Sum features of incident edges to each vertex
        vertex_features.scatter_add_(1, src.unsqueeze(0).unsqueeze(-1).expand(B, -1, D), h_edges)
        vertex_features.scatter_add_(1, dst.unsqueeze(0).unsqueeze(-1).expand(B, -1, D), h_edges)

        # Each edge reads from its two vertices
        h_src = vertex_features[:, src, :]  # (B, E, D)
        h_dst = vertex_features[:, dst, :]  # (B, E, D)

        # Combine
        h_new = self.mlp(torch.cat([h_src, h_dst], dim=-1))
        return h_new
```

### Constrained Denoising

The key advantage: we can enforce edge length constraints at every denoising step.

```python
class ConstrainedDiffusionSchedule(DiffusionSchedule):
    """Diffusion schedule with physics constraints on edge deltas."""

    def __init__(self, rest_lengths, num_steps=1000, max_strain=0.2):
        super().__init__(num_steps)
        self.rest_lengths = rest_lengths  # (E,)
        self.max_strain = max_strain  # Maximum stretch/compress (e.g., 0.2 = 20%)

        # Compute allowed length range
        self.min_length = rest_lengths * (1 - max_strain)
        self.max_length = rest_lengths * (1 + max_strain)

    def constrain_deltas(self, deltas):
        """Project deltas to satisfy edge length constraints."""
        # Get directions and lengths
        lengths = deltas.norm(dim=-1, keepdim=True)  # (B, E, 1)
        directions = deltas / lengths.clamp(min=1e-8)

        # Clamp lengths to valid range
        lengths_clamped = lengths.squeeze(-1).clamp(
            min=self.min_length,
            max=self.max_length
        ).unsqueeze(-1)

        # Reconstruct constrained deltas
        return directions * lengths_clamped

    @torch.no_grad()
    def p_sample(self, model, delta_n, n):
        """Denoise one step WITH constraint enforcement."""
        # Standard DDPM update
        n_tensor = torch.full((delta_n.shape[0],), n, device=delta_n.device, dtype=torch.long)
        predicted_noise = model(delta_n, n_tensor)

        # Compute mean
        beta = self.betas[n]
        sqrt_recip = self.sqrt_recip_alphas[n]
        sqrt_one_minus = self.sqrt_one_minus_alphas_cumprod[n]
        mean = sqrt_recip * (delta_n - beta / sqrt_one_minus * predicted_noise)

        if n == 0:
            delta_new = mean
        else:
            noise = torch.randn_like(delta_n)
            std = torch.sqrt(self.posterior_variance[n])
            delta_new = mean + std * noise

        # ENFORCE CONSTRAINTS
        delta_constrained = self.constrain_deltas(delta_new)

        return delta_constrained
```

### Training Loop

```python
def train_epoch_deltas(model, dataloader, optimizer, schedule, device):
    """Training loop for edge delta diffusion."""
    model.train()
    total_loss = 0

    for batch_pos in dataloader:
        # batch_pos: (B, V, 3) vertex positions
        batch_pos = batch_pos.to(device)
        B = batch_pos.shape[0]

        # Convert to edge deltas
        deltas = positions_to_deltas(batch_pos, model.edge_index)  # (B, E, 3)

        # Sample noise and timesteps
        noise = torch.randn_like(deltas)
        n = torch.randint(0, schedule.num_steps, (B,), device=device)

        # Forward process (add noise to deltas)
        deltas_noisy = schedule.q_sample(deltas, n, noise)

        # Predict noise
        predicted = model(deltas_noisy, n)

        # Loss
        loss = F.mse_loss(predicted, noise)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    return total_loss / len(dataloader)
```

### Generation

```python
@torch.no_grad()
def generate_mesh(model, schedule, device, anchor_pos=None):
    """Generate a new mesh from noise."""
    # Start with random noise in delta space
    deltas = torch.randn(1, model.num_edges, 3, device=device)

    # Denoise with constraints
    for n in reversed(range(schedule.num_steps)):
        deltas = schedule.p_sample(model, deltas, n)

        if n % 100 == 0:
            print(f"  Step {n}, mean edge length: {deltas.norm(dim=-1).mean():.4f}")

    # Reconstruct positions from deltas
    positions = deltas_to_positions(
        deltas[0],
        model.edge_index,
        anchor_idx=0,  # Fix first vertex
        anchor_pos=anchor_pos
    )

    return positions
```

## Handling Cycle Consistency

### The Problem

Edge deltas must be consistent around cycles:
```
δ_AB + δ_BC + δ_CA = 0  (for triangle ABC)
```

If the model predicts inconsistent deltas, reconstruction will accumulate errors.

### Solution 1: Least Squares Reconstruction

Instead of BFS (which picks arbitrary paths), solve for positions that best fit all deltas:

```python
def deltas_to_positions_least_squares(deltas, edge_index, anchor_idx=0):
    """Reconstruct positions via least squares (handles inconsistency)."""
    E = deltas.shape[0]
    V = edge_index.max().item() + 1
    src, dst = edge_index

    # Build system: for each edge, pos[dst] - pos[src] ≈ delta
    # This is: A @ pos = deltas, solve for pos

    # Sparse matrix A: (E, V) with +1 at dst, -1 at src
    row_idx = torch.arange(E).repeat_interleave(2)
    col_idx = torch.stack([src, dst]).T.flatten()
    values = torch.tensor([-1, 1]).repeat(E).float()

    A = torch.sparse_coo_tensor(
        torch.stack([row_idx, col_idx]),
        values,
        (E, V)
    ).to_dense()

    # Fix anchor (remove from unknowns)
    A_reduced = torch.cat([A[:, :anchor_idx], A[:, anchor_idx+1:]], dim=1)
    b = deltas  # (E, 3)

    # Least squares solve for each coordinate
    pos = torch.zeros(V, 3)
    for c in range(3):
        pos_reduced, _ = torch.lstsq(b[:, c:c+1], A_reduced)
        pos[:anchor_idx, c] = pos_reduced[:anchor_idx, 0]
        pos[anchor_idx+1:, c] = pos_reduced[anchor_idx:V-1, 0]

    return pos
```

### Solution 2: Cycle Consistency Loss

Add a loss term penalizing inconsistency:

```python
def cycle_consistency_loss(deltas, triangles, edge_to_idx):
    """Penalize deltas that don't sum to zero around triangles."""
    loss = 0
    for tri in triangles:
        a, b, c = tri
        # Get deltas around triangle
        d_ab = deltas[edge_to_idx[(a, b)]]
        d_bc = deltas[edge_to_idx[(b, c)]]
        d_ca = deltas[edge_to_idx[(c, a)]]

        # Should sum to zero
        cycle_error = d_ab + d_bc + d_ca
        loss += cycle_error.norm() ** 2

    return loss / len(triangles)
```

### Solution 3: Predict Per-Triangle, Share at Edges

Reformulate so consistency is built-in:

```python
# Instead of independent edge deltas, predict triangle deformations
# Each triangle: 3 vertices, 3 deltas, but only 2 DoF (third is determined)

def triangle_to_deltas(tri_deformation):
    """Extract edge deltas from triangle deformation (automatically consistent)."""
    # tri_deformation: (T, 2, 3) - two edge vectors per triangle
    # Third edge is: -edge1 - edge2 (closes the triangle)
    ...
```

## Comparison to Current Approach

| Aspect | Current (Vertex Positions) | Proposed (Edge Deltas) |
|--------|---------------------------|------------------------|
| Representation | (V, 3) positions | (E, 3) deltas |
| Translation | Must learn invariance | Built-in invariance |
| Edge lengths | Implicit constraint | Explicit, easy to enforce |
| Boundary | Fix vertex positions | Fix one anchor |
| Physics | Learned from data | Partially encoded |
| Reconstruction | Direct | Requires solve |
| Consistency | N/A | Must handle cycles |

## Implementation Plan

1. **Phase 1: Basic Implementation**
   - [ ] Implement `positions_to_deltas` and `deltas_to_positions`
   - [ ] Modify `FlagFrameDataset` to output deltas
   - [ ] Create `EdgeDeltaDiffusion` model
   - [ ] Test training loop with delta representation

2. **Phase 2: Constraints**
   - [ ] Implement `ConstrainedDiffusionSchedule`
   - [ ] Add edge length clamping during denoising
   - [ ] Add cycle consistency loss

3. **Phase 3: Evaluation**
   - [ ] Compare generation quality vs current approach
   - [ ] Measure edge length violations
   - [ ] Measure reconstruction error from cycles

4. **Phase 4: Extensions**
   - [ ] Per-triangle formulation (automatic consistency)
   - [ ] Variable stiffness per edge
   - [ ] Boundary handling (fix pole vertices)

## References

- Differential coordinates: Sorkine, "Differential Representations for Mesh Processing" (2006)
- Laplacian mesh editing: Sorkine et al., "Laplacian Surface Editing" (2004)
- Physics-based deformation: Terzopoulos et al., "Elastically Deformable Models" (1987)
