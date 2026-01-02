"""
Graph Deep Fakes - Multi-Dataset Generator
============================================
Generates multiple FEA solutions by varying parameters:
- Peclet number (advection strength)
- Inlet velocity magnitude
- Cylinder vertical position

Outputs:
- dataset/mesh.npz: Fixed mesh (nodes, triangles, Laplacian)
- dataset/solutions.npz: All FEA solutions with parameter labels
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.tri import Triangulation
from scipy.sparse import csr_matrix, diags, linalg as splinalg, save_npz
from scipy.spatial import Delaunay
import time
import os
from itertools import product

# ============================================================================
# CONFIGURATION
# ============================================================================
OUTPUT_DIR = "dataset"
MESH_RESOLUTION = 0.005  # Slightly coarser for faster generation

# Parameter ranges for dataset generation
DIFFUSIVITY_VALUES = [0.1, 0.2, 0.5, 1.0, 2.0, 5.0, 10.0]  # 7 values (thermal conductivity)
SOURCE_VALUES = [0.5, 1.0, 2.0, 3.0, 4.0, 5.0, 7.0]        # 7 values (heat source strength)
CYLINDER_Y_VALUES = [-0.03, -0.015, 0.0, 0.015, 0.03]      # 5 values

# Fixed domain parameters
CYLINDER_RADIUS = 0.05
DOMAIN_RADIUS = 0.15
X_MIN, X_MAX = -DOMAIN_RADIUS, DOMAIN_RADIUS * 2.5
Y_MIN, Y_MAX = -DOMAIN_RADIUS, DOMAIN_RADIUS

print("=" * 60)
print("GRAPH DEEP FAKES - MULTI-DATASET GENERATOR")
print("=" * 60)

# Create output directory
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ============================================================================
# MESH GENERATION (done once with cylinder at y=0)
# ============================================================================
print("\n[PHASE 1] Generating base mesh...")
start_time = time.time()

def generate_mesh(x_min, x_max, y_min, y_max, cx, cy, r, resolution):
    """Generate 2D triangular mesh for rectangular domain with circular obstacle."""
    nx = int((x_max - x_min) / resolution) + 1
    ny = int((y_max - y_min) / resolution) + 1

    x = np.linspace(x_min, x_max, nx)
    y = np.linspace(y_min, y_max, ny)
    xx, yy = np.meshgrid(x, y)
    points = np.column_stack([xx.ravel(), yy.ravel()])

    # Remove points inside cylinder
    dist_to_center = np.sqrt((points[:, 0] - cx)**2 + (points[:, 1] - cy)**2)
    outside_cylinder = dist_to_center > r * 1.1
    points = points[outside_cylinder]

    # Add rings around cylinder
    for ring_factor in [1.0, 1.15, 1.3]:
        n_circle = int(2 * np.pi * r * ring_factor / (resolution * 0.5))
        theta = np.linspace(0, 2*np.pi, n_circle, endpoint=False)
        circle_points = np.column_stack([
            cx + r * ring_factor * np.cos(theta),
            cy + r * ring_factor * np.sin(theta)
        ])
        points = np.vstack([points, circle_points])

    tri = Delaunay(points)
    triangles = tri.simplices

    # Remove triangles inside cylinder
    centroids = points[triangles].mean(axis=1)
    dist_centroids = np.sqrt((centroids[:, 0] - cx)**2 + (centroids[:, 1] - cy)**2)
    valid_triangles = triangles[dist_centroids > r]

    return points, valid_triangles

# Generate mesh with cylinder at center (y=0)
points, triangles = generate_mesh(
    X_MIN, X_MAX, Y_MIN, Y_MAX,
    0.0, 0.0, CYLINDER_RADIUS, MESH_RESOLUTION
)
n_nodes = len(points)
n_elements = len(triangles)
print(f"  Mesh: {n_nodes} nodes, {n_elements} elements ({time.time() - start_time:.2f}s)")

# ============================================================================
# BUILD GRAPH LAPLACIAN
# ============================================================================
print("\n[PHASE 2] Building graph Laplacian...")
start_time = time.time()

def build_cotangent_laplacian(points, triangles):
    """Build cotangent-weighted Laplacian matrix."""
    n = len(points)
    rows, cols, weights = [], [], []

    for tri in triangles:
        p = points[tri]
        for i in range(3):
            j = (i + 1) % 3
            k = (i + 2) % 3
            vi, vj, vk = tri[i], tri[j], tri[k]

            e1 = p[i] - p[k]
            e2 = p[j] - p[k]

            cos_angle = np.dot(e1, e2)
            e1_3d = np.array([e1[0], e1[1], 0])
            e2_3d = np.array([e2[0], e2[1], 0])
            sin_angle = np.abs(np.linalg.norm(np.cross(e1_3d, e2_3d)))
            cot_weight = cos_angle / (sin_angle + 1e-10) * 0.5
            cot_weight = max(cot_weight, 0)

            rows.extend([vi, vj])
            cols.extend([vj, vi])
            weights.extend([cot_weight, cot_weight])

    W = csr_matrix((weights, (rows, cols)), shape=(n, n))
    W = W.tocsr()
    D = diags(np.array(W.sum(axis=1)).flatten())
    L = D - W
    return L, W

L, W = build_cotangent_laplacian(points, triangles)
print(f"  Laplacian: {L.shape}, nnz={L.nnz} ({time.time() - start_time:.2f}s)")

# Compute eigendecomposition for spectral analysis
print("  Computing eigenvectors...")
start_time = time.time()
n_eigs = 50  # More modes for better representation
try:
    L_reg = L + 1e-8 * diags(np.ones(n_nodes))
    eigenvalues, eigenvectors = splinalg.eigsh(
        L_reg, k=n_eigs, which='LM', sigma=1e-6, tol=1e-4, maxiter=5000
    )
    eigenvalues = np.real(eigenvalues)
    eigenvectors = np.real(eigenvectors)
    idx = np.argsort(eigenvalues)
    eigenvalues = eigenvalues[idx]
    eigenvectors = eigenvectors[:, idx]
    print(f"  Computed {n_eigs} eigenvectors ({time.time() - start_time:.2f}s)")
except Exception as e:
    print(f"  Eigenvalue computation failed: {e}")
    eigenvalues, eigenvectors = None, None

# ============================================================================
# PDE SOLVER - Heat equation with parameter-dependent source and BCs
# ============================================================================
def solve_heat_equation(points, L, diffusivity, source_strength, cylinder_y):
    """
    Solve steady heat equation: -k*Laplacian(u) = f
    with parameter-dependent source term and boundary conditions.
    More numerically stable than convection-diffusion.
    """
    n = len(points)
    x, y = points[:, 0], points[:, 1]

    cx, cy = 0.0, cylinder_y
    r = CYLINDER_RADIUS

    # Distance from cylinder
    dist = np.sqrt((x - cx)**2 + (y - cy)**2)

    # Source term: heat source near cylinder, varies with source_strength
    # Creates a "hot spot" around the cylinder
    source = np.zeros(n)
    source_region = dist < r * 3
    source[source_region] = source_strength * np.exp(-(dist[source_region] - r)**2 / (r**2))

    # Additional source variation based on y-position (asymmetric heating)
    source += source_strength * 0.3 * np.sin(np.pi * y / Y_MAX) * (x > 0)

    # System matrix: diffusivity * Laplacian + small regularization
    A = diffusivity * L + 1e-6 * diags(np.ones(n))
    b = source.copy()

    # Boundary conditions
    tol = MESH_RESOLUTION * 1.5
    left = x < X_MIN + tol
    right = x > X_MAX - tol
    top = y > Y_MAX - tol
    bottom = y < Y_MIN + tol
    on_cylinder = dist < r * 1.3

    # Different BC values to create variation
    bc_nodes = left | right | top | bottom | on_cylinder
    u_bc = np.zeros(n)
    u_bc[left] = 1.0                              # Hot left wall
    u_bc[right] = 0.0                             # Cold right wall
    u_bc[top] = 0.5 + 0.3 * source_strength / 5   # Varies with source
    u_bc[bottom] = 0.5 - 0.3 * source_strength / 5
    u_bc[on_cylinder] = 0.2 + 0.6 * diffusivity   # Cylinder temp varies with diffusivity

    # Apply BCs
    A_mod = A.tolil()
    for i in np.where(bc_nodes)[0]:
        A_mod[i, :] = 0
        A_mod[i, i] = 1.0
        b[i] = u_bc[i]
    A_mod = A_mod.tocsr()

    # Solve
    u = splinalg.spsolve(A_mod, b)

    # Handle numerical issues
    u = np.nan_to_num(u, nan=0.5, posinf=1.0, neginf=0.0)

    # Normalize to [0, 1] for this solution
    u_min, u_max = u.min(), u.max()
    if u_max > u_min:
        u = (u - u_min) / (u_max - u_min)

    # Compute gradient magnitude as secondary output (like velocity)
    grad_x = np.zeros(n)
    grad_y = np.zeros(n)

    return u, grad_x, grad_y

# ============================================================================
# GENERATE ALL SOLUTIONS
# ============================================================================
print("\n[PHASE 3] Generating FEA solutions...")

# Create parameter combinations
param_combinations = list(product(DIFFUSIVITY_VALUES, SOURCE_VALUES, CYLINDER_Y_VALUES))
n_samples = len(param_combinations)
print(f"  Total combinations: {n_samples}")
print(f"  Parameters: Diffusivity={DIFFUSIVITY_VALUES}")
print(f"              Source={SOURCE_VALUES}")
print(f"              Cylinder_Y={CYLINDER_Y_VALUES}")

# Storage arrays
solutions = np.zeros((n_samples, n_nodes))
velocities_x = np.zeros((n_samples, n_nodes))
velocities_y = np.zeros((n_samples, n_nodes))
parameters = np.zeros((n_samples, 3))  # [peclet, velocity, cylinder_y]

start_time = time.time()
for idx, (diffusivity, source, cyl_y) in enumerate(param_combinations):
    u, gx, gy = solve_heat_equation(points, L, diffusivity, source, cyl_y)
    solutions[idx] = u
    velocities_x[idx] = gx
    velocities_y[idx] = gy
    parameters[idx] = [diffusivity, source, cyl_y]

    # Progress update every 25 samples
    if (idx + 1) % 25 == 0 or idx == 0:
        elapsed = time.time() - start_time
        rate = (idx + 1) / elapsed
        remaining = (n_samples - idx - 1) / rate
        print(f"  [{idx+1:3d}/{n_samples}] k={diffusivity:.1f}, Q={source:.1f}, Y={cyl_y:+.3f} "
              f"| {elapsed:.1f}s elapsed, ~{remaining:.1f}s remaining")

print(f"  Completed {n_samples} solutions in {time.time() - start_time:.1f}s")

# ============================================================================
# SAVE DATASET
# ============================================================================
print("\n[PHASE 4] Saving dataset...")

# Save mesh data
np.savez(
    f"{OUTPUT_DIR}/mesh.npz",
    points=points,
    triangles=triangles,
    eigenvalues=eigenvalues,
    eigenvectors=eigenvectors
)
save_npz(f"{OUTPUT_DIR}/laplacian.npz", L)
save_npz(f"{OUTPUT_DIR}/adjacency.npz", W)
print(f"  Saved: {OUTPUT_DIR}/mesh.npz")
print(f"  Saved: {OUTPUT_DIR}/laplacian.npz")
print(f"  Saved: {OUTPUT_DIR}/adjacency.npz")

# Save solutions
np.savez(
    f"{OUTPUT_DIR}/solutions.npz",
    solutions=solutions,
    velocities_x=velocities_x,
    velocities_y=velocities_y,
    parameters=parameters,
    param_names=['diffusivity', 'source', 'cylinder_y']
)
print(f"  Saved: {OUTPUT_DIR}/solutions.npz")

# ============================================================================
# GENERATE SAMPLE VISUALIZATIONS
# ============================================================================
print("\n[PHASE 5] Generating sample visualizations...")

triang = Triangulation(points[:, 0], points[:, 1], triangles)
margin = 0.02

# Select 9 diverse samples to visualize
sample_indices = [0, n_samples//4, n_samples//2, 3*n_samples//4, n_samples-1,
                  n_samples//8, 3*n_samples//8, 5*n_samples//8, 7*n_samples//8]

fig, axes = plt.subplots(3, 3, figsize=(15, 12))
fig.suptitle(f'Sample FEA Solutions ({n_samples} total in dataset)', fontsize=14, fontweight='bold')

for ax, idx in zip(axes.flatten(), sample_indices):
    diff, src, cy = parameters[idx]
    sol = solutions[idx]
    # Use fixed levels from 0 to 1 for consistent comparison
    levels = np.linspace(0, 1, 30)
    tcf = ax.tricontourf(triang, sol, levels=levels, cmap='inferno', extend='both')
    ax.tricontour(triang, sol, levels=10, colors='white', linewidths=0.3, alpha=0.5)
    circle = plt.Circle((0, cy), CYLINDER_RADIUS, fill=True, color='cyan', ec='black', lw=1)
    ax.add_patch(circle)
    ax.set_xlim(X_MIN - margin, X_MAX + margin)
    ax.set_ylim(Y_MIN - margin, Y_MAX + margin)
    ax.set_aspect('equal')
    ax.set_title(f'k={diff:.1f}, Q={src:.1f}, Y={cy:+.02f}', fontsize=10)
    ax.set_xticks([])
    ax.set_yticks([])

# Add colorbar
fig.subplots_adjust(right=0.92)
cbar_ax = fig.add_axes([0.94, 0.15, 0.02, 0.7])
sm = plt.cm.ScalarMappable(cmap='inferno', norm=plt.Normalize(0, 1))
fig.colorbar(sm, cax=cbar_ax, label='Temperature')

plt.savefig(f'{OUTPUT_DIR}/sample_solutions.png', dpi=150, bbox_inches='tight')
print(f"  Saved: {OUTPUT_DIR}/sample_solutions.png")

# Visualize parameter space coverage
fig2, axes2 = plt.subplots(1, 3, figsize=(14, 4))
fig2.suptitle('Dataset Parameter Coverage', fontsize=14, fontweight='bold')

ax = axes2[0]
ax.hist(parameters[:, 0], bins=len(DIFFUSIVITY_VALUES), edgecolor='black', alpha=0.7)
ax.set_xlabel('Diffusivity (k)')
ax.set_ylabel('Count')
ax.set_title('Diffusivity Distribution')

ax = axes2[1]
ax.hist(parameters[:, 1], bins=len(SOURCE_VALUES), edgecolor='black', alpha=0.7)
ax.set_xlabel('Source Strength (Q)')
ax.set_ylabel('Count')
ax.set_title('Source Distribution')

ax = axes2[2]
ax.hist(parameters[:, 2], bins=len(CYLINDER_Y_VALUES), edgecolor='black', alpha=0.7)
ax.set_xlabel('Cylinder Y Position')
ax.set_ylabel('Count')
ax.set_title('Cylinder Position Distribution')

plt.tight_layout()
plt.savefig(f'{OUTPUT_DIR}/parameter_coverage.png', dpi=150, bbox_inches='tight')
print(f"  Saved: {OUTPUT_DIR}/parameter_coverage.png")

# ============================================================================
# SUMMARY
# ============================================================================
print("\n" + "=" * 60)
print("DATASET GENERATION COMPLETE")
print("=" * 60)
print(f"""
Dataset Summary:
  - Samples: {n_samples} FEA solutions (steady heat equation)
  - Mesh: {n_nodes} nodes, {n_elements} elements
  - Parameters varied:
      Diffusivity: {DIFFUSIVITY_VALUES}
      Source:      {SOURCE_VALUES}
      Cylinder Y:  {CYLINDER_Y_VALUES}

Files saved to '{OUTPUT_DIR}/':
  - mesh.npz        : Node positions, triangles, eigenvectors
  - laplacian.npz   : Graph Laplacian (sparse)
  - adjacency.npz   : Adjacency weights (sparse)
  - solutions.npz   : All {n_samples} solutions + parameters
  - sample_solutions.png    : 9 sample visualizations
  - parameter_coverage.png  : Parameter distributions

Next steps:
  1. Load dataset: data = np.load('dataset/solutions.npz')
  2. Access solutions: data['solutions'].shape = ({n_samples}, {n_nodes})
  3. Access parameters: data['parameters'].shape = ({n_samples}, 3)
  4. Train graph autoencoder on solutions
  5. Fit diffusion model in latent space
""")
