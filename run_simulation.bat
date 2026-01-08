@echo off
REM Generate training data using GPU-accelerated flag simulation
REM Creates statistically independent frames by using large temporal stride

echo.
echo ==============================================================
echo  Flag Simulation - Generate Training Data
echo ==============================================================
echo.

set VENV_NAME=gdf_env

REM Activate virtual environment
if exist %VENV_NAME%\Scripts\activate.bat (
    call %VENV_NAME%\Scripts\activate.bat
) else (
    echo ERROR: Virtual environment not found.
    echo Run setup_gpu.bat first.
    pause
    exit /b 1
)

REM Configuration
set STRIDE=100
set FRAMES=100000
set OUTPUT=flag_data

echo Configuration:
echo   Stride: %STRIDE% (save every %STRIDE%th frame)
echo   Total steps: %FRAMES%
echo   Output frames: ~%FRAMES% / %STRIDE% = ~1000 frames
echo   Output: %OUTPUT%\flag_simulated.npz
echo.

REM Check GPU
echo Checking GPU...
python -c "import torch; print(f'  Device: {\"cuda\" if torch.cuda.is_available() else \"cpu\"}'); print(f'  GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"N/A\"}')"
echo.

REM Run simulation
echo Starting simulation...
echo ==============================================================
python simulate_flag.py --record --stride %STRIDE% --frames %FRAMES% --output %OUTPUT%
echo ==============================================================
echo.

if errorlevel 1 (
    echo ERROR: Simulation failed.
    pause
    exit /b 1
)

echo.
echo Simulation complete!
echo Training data saved to: %OUTPUT%\flag_simulated.npz
echo.
echo Next step: python train_flag_diffusion.py
echo.
pause
