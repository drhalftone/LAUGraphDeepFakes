@echo off
REM Setup and train trajectory diffusion model for flag data
REM Windows GPU version (NVIDIA CUDA)

echo ==============================================
echo Flag Trajectory Diffusion - GPU Setup ^& Train
echo ==============================================

set VENV_NAME=gdf_gpu

REM Check Python is available
echo.
echo [1/6] Checking Python installation...
python --version >nul 2>&1
if errorlevel 1 (
    echo ERROR: Python not found. Please install Python 3.8+.
    exit /b 1
)
python --version

REM Create virtual environment if it doesn't exist
echo.
echo [2/6] Setting up virtual environment '%VENV_NAME%'...
if exist %VENV_NAME% (
    echo   Virtual environment already exists, reusing it.
) else (
    echo   Creating new virtual environment...
    python -m venv %VENV_NAME%
)

REM Activate virtual environment
echo.
echo [3/6] Activating virtual environment...
call %VENV_NAME%\Scripts\activate.bat

REM Upgrade pip and install dependencies
echo.
echo [4/6] Installing dependencies...
pip install --upgrade pip --quiet

echo   Installing PyTorch with CUDA...
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121

echo   Installing torch-geometric...
pip install torch-geometric

echo   Installing TensorFlow (for data loading)...
pip install tensorflow

echo   Installing numpy, matplotlib, tqdm...
pip install numpy matplotlib tqdm

REM Verify CUDA
echo.
echo Verifying CUDA...
python -c "import torch; print(f'PyTorch: {torch.__version__}'); print(f'CUDA available: {torch.cuda.is_available()}'); print(f'GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"None\"}')"

REM Download and prepare data
echo.
echo [5/6] Downloading and preparing flag data...
python setup_flag_data.py
if errorlevel 1 (
    echo ERROR: Data setup failed.
    exit /b 1
)

REM Train the model
echo.
echo [6/6] Training trajectory diffusion model on GPU...
echo.
python train_flag_diffusion.py

echo.
echo ==============================================
echo Training complete!
echo ==============================================
echo.
echo Output saved to: flag_diffusion_output\
echo.
pause
