@echo off
REM Complete setup and training for flag trajectory diffusion
REM Run this on a fresh machine with an NVIDIA GPU
REM Skips any steps that have already been completed

echo.
echo ==============================================================
echo  Flag Trajectory Diffusion - Complete GPU Setup and Training
echo ==============================================================
echo.

set VENV_NAME=gdf_env

REM ---------------------------------------------------------------
REM Step 1: Check Python
REM ---------------------------------------------------------------
echo [1/6] Checking Python installation...
python --version >nul 2>&1
if errorlevel 1 (
    echo   ERROR: Python not found!
    echo   Please install Python 3.10+ from https://python.org
    pause
    exit /b 1
)
python --version
echo.

REM ---------------------------------------------------------------
REM Step 2: Create/reuse virtual environment
REM ---------------------------------------------------------------
echo [2/6] Setting up virtual environment '%VENV_NAME%'...
if exist %VENV_NAME%\Scripts\activate.bat (
    echo   Found existing venv, reusing it.
) else (
    echo   Creating new virtual environment...
    python -m venv %VENV_NAME%
    if errorlevel 1 (
        echo   ERROR: Failed to create virtual environment.
        pause
        exit /b 1
    )
)
call %VENV_NAME%\Scripts\activate.bat
echo.

REM ---------------------------------------------------------------
REM Step 3: Install dependencies
REM ---------------------------------------------------------------
echo [3/6] Installing dependencies...
pip install --upgrade pip --quiet

REM Check if PyTorch is already installed with CUDA
python -c "import torch; assert torch.cuda.is_available()" >nul 2>&1
if errorlevel 1 (
    echo   Installing PyTorch with CUDA 12.1...
    pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
) else (
    echo   PyTorch with CUDA already installed, skipping.
)

REM Check if torch_geometric is installed
python -c "import torch_geometric" >nul 2>&1
if errorlevel 1 (
    echo   Installing torch-geometric...
    pip install torch-geometric
) else (
    echo   torch-geometric already installed, skipping.
)

REM Check if TensorFlow is installed
python -c "import tensorflow" >nul 2>&1
if errorlevel 1 (
    echo   Installing TensorFlow...
    pip install tensorflow
) else (
    echo   TensorFlow already installed, skipping.
)

REM Install remaining dependencies (fast, always run)
echo   Installing numpy, matplotlib, tqdm...
pip install numpy matplotlib tqdm --quiet
echo.

REM ---------------------------------------------------------------
REM Step 4: Verify GPU
REM ---------------------------------------------------------------
echo [4/6] Verifying GPU setup...
python -c "import torch; print(f'  PyTorch {torch.__version__}'); print(f'  CUDA: {torch.cuda.is_available()}'); print(f'  GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"Not found\"}')"
echo.

REM ---------------------------------------------------------------
REM Step 5: Download and prepare data
REM ---------------------------------------------------------------
echo [5/6] Preparing training data...
if exist flag_data\flag_test.npz (
    echo   Data already prepared: flag_data\flag_test.npz
    echo   Skipping download and conversion.
) else (
    echo   Running setup_flag_data.py...
    python setup_flag_data.py
    if errorlevel 1 (
        echo   ERROR: Data setup failed.
        pause
        exit /b 1
    )
)
echo.

REM ---------------------------------------------------------------
REM Step 6: Train
REM ---------------------------------------------------------------
echo [6/6] Starting training...
echo.
echo ==============================================================
python train_flag_diffusion.py
echo ==============================================================
echo.

echo Training complete!
echo Output saved to: flag_diffusion_output\
echo.
pause
