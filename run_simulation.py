"""
Graph Deep Fakes - Pilot Simulation
====================================
Generates a 2D mesh with obstacle, solves a steady convection-diffusion equation
(simplified Navier-Stokes analog), and visualizes the mesh and solution.

This demonstrates the pipeline:
1. Mesh generation (channel with cylinder)
2. Graph Laplacian construction
3. PDE solve (convection-diffusion as NS proxy)
4. Visualization of mesh and field signals
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend - saves to file without displaying
import matplotlib.pyplot as plt
from matplotlib.tri import Triangulation
from scipy.sparse import csr_matrix, diags, linalg as splinalg
from scipy.spatial import Delaunay
import time

# ============================================================================
# CONFIGURATION
# ============================================================================
# Zoomed-in domain around cylinder
CYLINDER_CENTER = (0.0, 0.0)  # Center cylinder at origin for zoomed view
CYLINDER_RADIUS = 0.05

# Domain: small region around cylinder (±0.15 in each direction)
DOMAIN_RADIUS = 0.15
X_MIN, X_MAX = -DOMAIN_RADIUS, DOMAIN_RADIUS * 2.5  # Extend downstream for wake
Y_MIN, Y_MAX = -DOMAIN_RADIUS, DOMAIN_RADIUS

MESH_RESOLUTION = 0.004  # Much finer mesh for detailed view
PECLET_NUMBER = 50.0     # Controls convection strength (higher = more advection-dominated)

print("=" * 60)
print("GRAPH DEEP FAKES - PILOT SIMULATION")
print("=" * 60)


# ============================================================================
# STEP 1: MESH GENERATION
# ============================================================================
print("\n[STEP 1/5] Generating mesh...")
start_time = time.time()

def generate_mesh(x_min, x_max, y_min, y_max, cx, cy, r, resolution):
    """Generate 2D triangular mesh for rectangular domain with circular obstacle."""
    # Create point cloud
    nx = int((x_max - x_min) / resolution) + 1
    ny = int((y_max - y_min) / resolution) + 1

    # Regular grid points
    x = np.linspace(x_min, x_max, nx)
    y = np.linspace(y_min, y_max, ny)
    xx, yy = np.meshgrid(x, y)
    points = np.column_stack([xx.ravel(), yy.ravel()])

    # Remove points inside cylinder
    dist_to_center = np.sqrt((points[:, 0] - cx)**2 + (points[:, 1] - cy)**2)
    outside_cylinder = dist_to_center > r * 1.1
    points = points[outside_cylinder]

    # Add multiple rings of points around cylinder for better resolution
    for ring_factor in [1.0, 1.15, 1.3]:
        n_circle = int(2 * np.pi * r * ring_factor / (resolution * 0.5))
        theta = np.linspace(0, 2*np.pi, n_circle, endpoint=False)
        circle_points = np.column_stack([
            cx + r * ring_factor * np.cos(theta),
            cy + r * ring_factor * np.sin(theta)
        ])
        points = np.vstack([points, circle_points])

    # Triangulate
    tri = Delaunay(points)
    triangles = tri.simplices

    # Remove triangles whose centroid is inside cylinder
    centroids = points[triangles].mean(axis=1)
    dist_centroids = np.sqrt((centroids[:, 0] - cx)**2 + (centroids[:, 1] - cy)**2)
    valid_triangles = triangles[dist_centroids > r]

    return points, valid_triangles

points, triangles = generate_mesh(
    X_MIN, X_MAX, Y_MIN, Y_MAX,
    CYLINDER_CENTER[0], CYLINDER_CENTER[1],
    CYLINDER_RADIUS, MESH_RESOLUTION
)

n_nodes = len(points)
n_elements = len(triangles)
print(f"  Mesh generated in {time.time() - start_time:.2f}s")
print(f"  Nodes: {n_nodes}, Elements: {n_elements}")


# ============================================================================
# STEP 2: BUILD GRAPH LAPLACIAN
# ============================================================================
print("\n[STEP 2/5] Building graph Laplacian...")
start_time = time.time()

def build_cotangent_laplacian(points, triangles):
    """Build cotangent-weighted Laplacian matrix."""
    n = len(points)

    # Build adjacency with cotangent weights
    rows, cols, weights = [], [], []

    for tri in triangles:
        # Get triangle vertices
        p = points[tri]

        # Compute edges and cotangent weights
        for i in range(3):
            j = (i + 1) % 3
            k = (i + 2) % 3

            # Edge from vertex i to vertex j
            vi, vj, vk = tri[i], tri[j], tri[k]

            # Vectors for cotangent calculation
            e1 = p[i] - p[k]  # edge to vertex i from opposite
            e2 = p[j] - p[k]  # edge to vertex j from opposite

            # Cotangent of angle at vertex k
            cos_angle = np.dot(e1, e2)
            # Use 3D cross product to avoid deprecation warning
            e1_3d = np.array([e1[0], e1[1], 0])
            e2_3d = np.array([e2[0], e2[1], 0])
            sin_angle = np.abs(np.linalg.norm(np.cross(e1_3d, e2_3d)))
            cot_weight = cos_angle / (sin_angle + 1e-10) * 0.5
            cot_weight = max(cot_weight, 0)  # Clamp negative weights

            rows.extend([vi, vj])
            cols.extend([vj, vi])
            weights.extend([cot_weight, cot_weight])

    # Build sparse adjacency matrix
    W = csr_matrix((weights, (rows, cols)), shape=(n, n))

    # Combine duplicate entries (sum)
    W = W.tocsr()

    # Degree matrix
    D = diags(np.array(W.sum(axis=1)).flatten())

    # Laplacian: L = D - W
    L = D - W

    return L, W

L, W = build_cotangent_laplacian(points, triangles)
print(f"  Laplacian built in {time.time() - start_time:.2f}s")
print(f"  Laplacian shape: {L.shape}, nnz: {L.nnz}")


# ============================================================================
# STEP 3: SET UP AND SOLVE CONVECTION-DIFFUSION PDE
# ============================================================================
print("\n[STEP 3/5] Solving convection-diffusion equation...")
print("  (This is a simplified steady Navier-Stokes analog)")
start_time = time.time()

def solve_convection_diffusion(points, L, peclet):
    """
    Solve steady convection-diffusion: -div(grad(u)) + Pe * v·grad(u) = 0
    with Dirichlet BCs: u=1 at inlet, u=0 at outlet.

    This mimics temperature/concentration transport in a flow field.
    """
    n = len(points)
    x, y = points[:, 0], points[:, 1]

    # Assumed velocity field (parabolic profile, flow around cylinder)
    cx, cy = CYLINDER_CENTER
    r = CYLINDER_RADIUS
    domain_height = Y_MAX - Y_MIN

    # Parabolic inflow profile (centered at y=0)
    u_max = 1.5
    # Parabolic profile: max at center (y=0), zero at y=±Y_MAX
    vx = u_max * (1 - (y / Y_MAX)**2)
    vx = np.clip(vx, 0, u_max)  # Ensure non-negative
    vy = np.zeros_like(y)

    # Deflection around cylinder (potential flow approximation)
    dist = np.sqrt((x - cx)**2 + (y - cy)**2)
    mask = dist < 3 * r
    if np.any(mask):
        dx, dy = x[mask] - cx, y[mask] - cy
        factor = (r / dist[mask])**2
        vx[mask] = vx[mask] * (1 - factor * (dx**2 - dy**2) / dist[mask]**2)
        vy[mask] = -vx[mask] * factor * 2 * dx * dy / dist[mask]**2

    # Build convection matrix using upwind scheme (simplified)
    # We'll use the graph structure for a simple upwind approximation
    conv_rows, conv_cols, conv_vals = [], [], []
    W_coo = W.tocoo()

    for i, j, w in zip(W_coo.row, W_coo.col, W_coo.data):
        if i != j:
            # Direction from i to j
            dx = points[j, 0] - points[i, 0]
            dy = points[j, 1] - points[i, 1]
            edge_len = np.sqrt(dx**2 + dy**2) + 1e-10

            # Velocity at midpoint
            vx_mid = 0.5 * (vx[i] + vx[j])
            vy_mid = 0.5 * (vy[i] + vy[j])

            # Upwind contribution
            flow = (vx_mid * dx + vy_mid * dy) / edge_len
            upwind = peclet * max(flow, 0) * w

            conv_rows.append(i)
            conv_cols.append(j)
            conv_vals.append(-upwind)

            conv_rows.append(i)
            conv_cols.append(i)
            conv_vals.append(upwind)

    C = csr_matrix((conv_vals, (conv_rows, conv_cols)), shape=(n, n))

    # System matrix: diffusion + convection
    A = L + C

    # Right-hand side (zero for homogeneous)
    b = np.zeros(n)

    # Boundary conditions
    tol = MESH_RESOLUTION * 1.5
    inlet = x < X_MIN + tol
    outlet = x > X_MAX - tol
    walls = (y < Y_MIN + tol) | (y > Y_MAX - tol)
    cylinder = np.sqrt((x - cx)**2 + (y - cy)**2) < r * 1.2

    bc_nodes = inlet | outlet
    interior = ~bc_nodes

    # Dirichlet values
    u_bc = np.zeros(n)
    u_bc[inlet] = 1.0  # Hot at inlet
    u_bc[outlet] = 0.0  # Cold at outlet

    # Modify system for Dirichlet BCs
    A_mod = A.tolil()
    for i in np.where(bc_nodes)[0]:
        A_mod[i, :] = 0
        A_mod[i, i] = 1.0
        b[i] = u_bc[i]
    A_mod = A_mod.tocsr()

    # Solve
    u = splinalg.spsolve(A_mod, b)

    return u, vx, vy

u_field, vx_field, vy_field = solve_convection_diffusion(points, L, PECLET_NUMBER)
print(f"  PDE solved in {time.time() - start_time:.2f}s")
print(f"  Solution range: [{u_field.min():.3f}, {u_field.max():.3f}]")


# ============================================================================
# STEP 4: COMPUTE GRAPH SPECTRAL PROPERTIES
# ============================================================================
print("\n[STEP 4/5] Computing graph spectral properties...")
start_time = time.time()

# Compute a few smallest eigenvalues/vectors of the Laplacian
n_eigs = min(10, n_nodes - 2)
try:
    # Add small regularization for numerical stability
    L_reg = L + 1e-8 * diags(np.ones(n_nodes))
    # Use shift-invert mode with larger tolerance for convergence
    eigenvalues, eigenvectors = splinalg.eigsh(
        L_reg, k=n_eigs, which='LM', sigma=1e-6, tol=1e-4, maxiter=5000
    )
    eigenvalues = np.real(eigenvalues)
    eigenvectors = np.real(eigenvectors)
    # Sort by eigenvalue
    idx = np.argsort(eigenvalues)
    eigenvalues = eigenvalues[idx]
    eigenvectors = eigenvectors[:, idx]
    print(f"  Computed {n_eigs} smallest eigenvalues in {time.time() - start_time:.2f}s")
    print(f"  Eigenvalues: {eigenvalues[:5].round(4)}...")
except Exception as e:
    print(f"  Eigenvalue computation skipped: {e}")
    eigenvalues = None
    eigenvectors = None


# ============================================================================
# STEP 5: VISUALIZATION
# ============================================================================
print("\n[STEP 5/5] Generating visualizations...")
start_time = time.time()

# Create triangulation for matplotlib
triang = Triangulation(points[:, 0], points[:, 1], triangles)

# Figure 1: Mesh and solution field
fig1, axes1 = plt.subplots(2, 2, figsize=(14, 10))
fig1.suptitle('Graph Deep Fakes - Mesh and FEA Signal', fontsize=14, fontweight='bold')

# Panel 1: Mesh structure
ax = axes1[0, 0]
ax.triplot(triang, 'k-', linewidth=0.2, alpha=0.7)
circle = plt.Circle(CYLINDER_CENTER, CYLINDER_RADIUS, fill=True, color='gray')
ax.add_patch(circle)
margin = 0.02
ax.set_xlim(X_MIN - margin, X_MAX + margin)
ax.set_ylim(Y_MIN - margin, Y_MAX + margin)
ax.set_aspect('equal')
ax.set_title(f'Mesh Structure (Zoomed on Cylinder)\n({n_nodes} nodes, {n_elements} elements)')
ax.set_xlabel('x')
ax.set_ylabel('y')

# Panel 2: Solution field (temperature/concentration)
ax = axes1[0, 1]
tcf = ax.tricontourf(triang, u_field, levels=30, cmap='coolwarm')
ax.tricontour(triang, u_field, levels=15, colors='k', linewidths=0.3)
circle = plt.Circle(CYLINDER_CENTER, CYLINDER_RADIUS, fill=True, color='white', ec='black')
ax.add_patch(circle)
plt.colorbar(tcf, ax=ax, label='Field value')
ax.set_xlim(X_MIN - margin, X_MAX + margin)
ax.set_ylim(Y_MIN - margin, Y_MAX + margin)
ax.set_aspect('equal')
ax.set_title('FEA Signal (Convection-Diffusion Solution)\nAnalog to temperature/velocity magnitude')
ax.set_xlabel('x')
ax.set_ylabel('y')

# Panel 3: Velocity field
ax = axes1[1, 0]
# Subsample for quiver plot
step = max(1, n_nodes // 800)
idx = np.arange(0, n_nodes, step)
speed = np.sqrt(vx_field**2 + vy_field**2)
tcf = ax.tricontourf(triang, speed, levels=30, cmap='viridis')
ax.quiver(points[idx, 0], points[idx, 1], vx_field[idx], vy_field[idx],
          color='white', scale=20, width=0.003, alpha=0.8)
circle = plt.Circle(CYLINDER_CENTER, CYLINDER_RADIUS, fill=True, color='gray', ec='black')
ax.add_patch(circle)
plt.colorbar(tcf, ax=ax, label='Speed')
ax.set_xlim(X_MIN - margin, X_MAX + margin)
ax.set_ylim(Y_MIN - margin, Y_MAX + margin)
ax.set_aspect('equal')
ax.set_title('Velocity Field\n(Prescribed parabolic + cylinder deflection)')
ax.set_xlabel('x')
ax.set_ylabel('y')

# Panel 4: Graph structure visualization (node degree)
ax = axes1[1, 1]
degree = np.array(W.sum(axis=1)).flatten()
tcf = ax.tripcolor(triang, degree, cmap='plasma', shading='flat')
circle = plt.Circle(CYLINDER_CENTER, CYLINDER_RADIUS, fill=True, color='white', ec='black')
ax.add_patch(circle)
plt.colorbar(tcf, ax=ax, label='Weighted degree')
ax.set_xlim(X_MIN - margin, X_MAX + margin)
ax.set_ylim(Y_MIN - margin, Y_MAX + margin)
ax.set_aspect('equal')
ax.set_title('Graph Laplacian Node Weights\n(Cotangent-weighted degree)')
ax.set_xlabel('x')
ax.set_ylabel('y')

plt.tight_layout()
plt.savefig('output_mesh_and_signal.png', dpi=150, bbox_inches='tight')
print(f"  Saved: output_mesh_and_signal.png")

# Figure 2: Spectral analysis (if eigenvalues computed)
if eigenvectors is not None:
    fig2, axes2 = plt.subplots(2, 3, figsize=(15, 8))
    fig2.suptitle('Graph Laplacian Eigenvectors (Manifold Harmonics)', fontsize=14, fontweight='bold')

    for i, ax in enumerate(axes2.flatten()):
        if i < len(eigenvalues):
            ev = eigenvectors[:, i]
            tcf = ax.tricontourf(triang, ev, levels=25, cmap='RdBu_r')
            circle = plt.Circle(CYLINDER_CENTER, CYLINDER_RADIUS, fill=True, color='white', ec='black')
            ax.add_patch(circle)
            plt.colorbar(tcf, ax=ax)
            ax.set_xlim(X_MIN - margin, X_MAX + margin)
            ax.set_ylim(Y_MIN - margin, Y_MAX + margin)
            ax.set_aspect('equal')
            ax.set_title(f'Mode {i}: λ = {eigenvalues[i]:.4f}')
            ax.set_xlabel('x')
            ax.set_ylabel('y')

    plt.tight_layout()
    plt.savefig('output_spectral_modes.png', dpi=150, bbox_inches='tight')
    print(f"  Saved: output_spectral_modes.png")

# Figure 3: Signal in spectral domain
if eigenvectors is not None:
    fig3, axes3 = plt.subplots(1, 2, figsize=(12, 4))
    fig3.suptitle('Graph Signal Processing View', fontsize=14, fontweight='bold')

    # Graph Fourier Transform of the solution
    u_hat = eigenvectors.T @ u_field

    ax = axes3[0]
    ax.bar(range(len(eigenvalues)), np.abs(u_hat), color='steelblue', alpha=0.8)
    ax.set_xlabel('Eigenvalue index (frequency)')
    ax.set_ylabel('|Fourier coefficient|')
    ax.set_title('Graph Fourier Transform of FEA Signal')
    ax.grid(True, alpha=0.3)

    ax = axes3[1]
    ax.semilogy(eigenvalues, np.abs(u_hat), 'o-', color='steelblue', markersize=8)
    ax.set_xlabel('Eigenvalue λ')
    ax.set_ylabel('|Fourier coefficient| (log scale)')
    ax.set_title('Spectral Energy Distribution')
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('output_spectral_analysis.png', dpi=150, bbox_inches='tight')
    print(f"  Saved: output_spectral_analysis.png")

print(f"  Visualizations completed in {time.time() - start_time:.2f}s")

# ============================================================================
# SUMMARY
# ============================================================================
print("\n" + "=" * 60)
print("SIMULATION COMPLETE")
print("=" * 60)
domain_width = X_MAX - X_MIN
domain_height = Y_MAX - Y_MIN
print(f"""
Summary:
  - Mesh: {n_nodes} nodes, {n_elements} triangular elements
  - Domain: {domain_width:.2f} x {domain_height:.2f} (zoomed on cylinder, r={CYLINDER_RADIUS})
  - Resolution: {MESH_RESOLUTION} (high-resolution near cylinder)
  - PDE: Steady convection-diffusion (Peclet = {PECLET_NUMBER})
  - Graph Laplacian: Cotangent-weighted, {L.nnz} non-zeros

Output files:
  - output_mesh_and_signal.png   : Mesh structure and FEA solution
  - output_spectral_modes.png    : Laplacian eigenvectors (manifold harmonics)
  - output_spectral_analysis.png : Graph Fourier analysis of the signal

Next steps for deep fake generation:
  1. Generate multiple FEA solutions with varying parameters
  2. Train graph autoencoder to compress fields to latent space
  3. Fit diffusion model on latent representations
  4. Sample new latent codes and decode to synthetic FEA fields
""")

# Images saved to files - no interactive display (use 'open *.png' to view)
