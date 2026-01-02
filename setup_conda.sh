#!/bin/bash
# Setup conda environment for Graph Deep Fakes training
# PyTorch requires Python ≤3.12

set -e

ENV_NAME="gdf_torch"

echo "=============================================="
echo "Graph Deep Fakes - Conda Environment Setup"
echo "=============================================="

# Check if conda is available
if ! command -v conda &> /dev/null; then
    echo "ERROR: conda not found. Please install Miniconda or Anaconda."
    echo "  https://docs.conda.io/en/latest/miniconda.html"
    exit 1
fi

echo ""
echo "[1/4] Creating conda environment with Python 3.11..."
conda create -n $ENV_NAME python=3.11 -y

echo ""
echo "[2/4] Activating environment..."
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate $ENV_NAME

echo ""
echo "[3/4] Installing PyTorch and dependencies..."
# Install PyTorch (CPU version for compatibility)
conda install pytorch torchvision -c pytorch -y

# Install other dependencies
pip install numpy scipy matplotlib scikit-learn

echo ""
echo "[4/4] Verifying installation..."
python -c "import torch; print(f'PyTorch {torch.__version__} installed successfully')"
python -c "import numpy; print(f'NumPy {numpy.__version__}')"
python -c "import scipy; print(f'SciPy {scipy.__version__}')"

echo ""
echo "=============================================="
echo "Setup complete!"
echo "=============================================="
echo ""
echo "To use this environment:"
echo "  conda activate $ENV_NAME"
echo ""
echo "To train the model:"
echo "  conda activate $ENV_NAME"
echo "  python train_model.py"
echo ""
