@echo off
REM Setup and train trajectory diffusion model for flag data
REM Windows version (CPU - for GPU use setup_gpu.sh on Linux)

echo ==============================================
echo Flag Trajectory Diffusion - Setup and Train
echo ==============================================

set VENV_NAME=gdf_env

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

REM Upgrade pip
echo.
echo [4/6] Upgrading pip and installing dependencies...
pip install --upgrade pip --quiet

echo   Installing PyTorch (CPU)...
pip install torch --quiet

echo   Installing torch-geometric...
pip install torch-geometric --quiet

echo   Installing TensorFlow (for data loading)...
pip install tensorflow --quiet

echo   Installing numpy, matplotlib, tqdm...
pip install numpy matplotlib tqdm --quiet

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
echo [6/6] Training trajectory diffusion model...
echo   (This will take a while on CPU - GPU recommended)
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
