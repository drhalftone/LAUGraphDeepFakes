#!/bin/bash
# Setup and train trajectory diffusion model for flag data
# GPU version (Linux/Mac with NVIDIA GPU)

set -e

echo "=============================================="
echo "Flag Trajectory Diffusion - GPU Setup & Train"
echo "=============================================="

VENV_NAME="gdf_gpu"

# Check Python
echo ""
echo "[1/6] Checking Python..."
python3 --version

# Create virtual environment if it doesn't exist
echo ""
echo "[2/6] Setting up virtual environment '$VENV_NAME'..."
if [ -d "$VENV_NAME" ]; then
    echo "  Virtual environment already exists, reusing it."
else
    echo "  Creating new virtual environment..."
    python3 -m venv $VENV_NAME
fi

source $VENV_NAME/bin/activate

# Install dependencies
echo ""
echo "[3/6] Installing dependencies..."
pip install --upgrade pip

echo "  Installing PyTorch with CUDA..."
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121

echo "  Installing torch-geometric..."
pip install torch-geometric

echo "  Installing TensorFlow (for data loading)..."
pip install tensorflow

echo "  Installing numpy, matplotlib, tqdm..."
pip install numpy matplotlib tqdm

# Verify CUDA
echo ""
echo "[4/6] Verifying CUDA..."
python -c "import torch; print(f'PyTorch: {torch.__version__}'); print(f'CUDA available: {torch.cuda.is_available()}'); print(f'GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"None\"}')"

# Download and prepare data
echo ""
echo "[5/6] Downloading and preparing flag data..."
python setup_flag_data.py

# Train the model
echo ""
echo "[6/6] Training trajectory diffusion model..."
python train_flag_diffusion.py

echo ""
echo "=============================================="
echo "Training complete!"
echo "=============================================="
echo ""
echo "Output saved to: flag_diffusion_output/"
echo ""
