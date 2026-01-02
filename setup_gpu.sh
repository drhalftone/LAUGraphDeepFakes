#!/bin/bash
# Setup for GPU training (RTX 4070 Ti Super)
# Run this on your GPU machine

set -e

echo "=============================================="
echo "Graph Deep Fakes - GPU Setup (CUDA)"
echo "=============================================="

# Create virtual environment
echo "[1/3] Creating Python environment..."
python3 -m venv gdf_gpu
source gdf_gpu/bin/activate

# Install PyTorch with CUDA support
echo "[2/3] Installing PyTorch with CUDA..."
pip install --upgrade pip
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121

# Install other dependencies
echo "[3/3] Installing dependencies..."
pip install numpy scipy matplotlib scikit-learn

# Verify CUDA
echo ""
echo "Verifying CUDA..."
python -c "import torch; print(f'PyTorch: {torch.__version__}'); print(f'CUDA available: {torch.cuda.is_available()}'); print(f'GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"None\"}')"

echo ""
echo "=============================================="
echo "Setup complete! To train:"
echo "  source gdf_gpu/bin/activate"
echo "  python train_model.py"
echo "=============================================="
