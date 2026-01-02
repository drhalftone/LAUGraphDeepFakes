#!/bin/bash
# Setup script for Graph Deep Fakes on FEA Simulations
# macOS/Linux version

set -e  # Exit on error

echo "=============================================="
echo "Graph Deep Fakes - Environment Setup"
echo "=============================================="

# Configuration
VENV_NAME="gdf_env"
PYTHON_VERSION="python3"

# Check Python is available
echo ""
echo "[1/5] Checking Python installation..."
if ! command -v $PYTHON_VERSION &> /dev/null; then
    echo "ERROR: $PYTHON_VERSION not found. Please install Python 3.8+."
    exit 1
fi
$PYTHON_VERSION --version

# Create virtual environment
echo ""
echo "[2/5] Creating virtual environment '$VENV_NAME'..."
if [ -d "$VENV_NAME" ]; then
    echo "  Virtual environment already exists. Removing old one..."
    rm -rf "$VENV_NAME"
fi
$PYTHON_VERSION -m venv $VENV_NAME
echo "  Done."

# Activate virtual environment
echo ""
echo "[3/5] Activating virtual environment..."
source $VENV_NAME/bin/activate
echo "  Active Python: $(which python)"

# Upgrade pip
echo ""
echo "[4/5] Upgrading pip..."
pip install --upgrade pip --quiet

# Install dependencies
echo ""
echo "[5/5] Installing required packages..."
echo "  Installing numpy..."
pip install numpy --quiet
echo "  Installing scipy..."
pip install scipy --quiet
echo "  Installing matplotlib..."
pip install matplotlib --quiet
echo "  Installing networkx..."
pip install networkx --quiet

# PyTorch is optional - only needed for diffusion model training
echo "  Installing torch (optional, may fail on Python 3.13+)..."
pip install torch --quiet 2>/dev/null || echo "    [SKIP] PyTorch not available for this Python version - will install later"
echo "  Installing torch-geometric (optional)..."
pip install torch-geometric --quiet 2>/dev/null || echo "    [SKIP] PyTorch Geometric skipped"

echo "  Installing meshio..."
pip install meshio --quiet
echo "  Installing gmsh (for mesh generation)..."
pip install gmsh --quiet

echo ""
echo "=============================================="
echo "Environment setup complete!"
echo "=============================================="
echo ""
echo "To activate the environment manually:"
echo "  source $VENV_NAME/bin/activate"
echo ""
echo "Running simulation..."
echo ""

# Run the simulation
python run_simulation.py

echo ""
echo "Done! Check the generated images."
