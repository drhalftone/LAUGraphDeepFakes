@echo off
REM Setup script for Graph Deep Fakes on FEA Simulations
REM Windows version

echo ==============================================
echo Graph Deep Fakes - Environment Setup
echo ==============================================

set VENV_NAME=gdf_env

REM Check Python is available
echo.
echo [1/5] Checking Python installation...
python --version >nul 2>&1
if errorlevel 1 (
    echo ERROR: Python not found. Please install Python 3.8+.
    exit /b 1
)
python --version

REM Create virtual environment
echo.
echo [2/5] Creating virtual environment '%VENV_NAME%'...
if exist %VENV_NAME% (
    echo   Virtual environment already exists. Removing old one...
    rmdir /s /q %VENV_NAME%
)
python -m venv %VENV_NAME%
echo   Done.

REM Activate virtual environment
echo.
echo [3/5] Activating virtual environment...
call %VENV_NAME%\Scripts\activate.bat

REM Upgrade pip
echo.
echo [4/5] Upgrading pip...
pip install --upgrade pip --quiet

REM Install dependencies
echo.
echo [5/5] Installing required packages...
echo   Installing numpy...
pip install numpy --quiet
echo   Installing scipy...
pip install scipy --quiet
echo   Installing matplotlib...
pip install matplotlib --quiet
echo   Installing networkx...
pip install networkx --quiet
echo   Installing torch...
pip install torch --quiet
echo   Installing torch-geometric...
pip install torch-geometric --quiet
echo   Installing meshio...
pip install meshio --quiet
echo   Installing gmsh (for mesh generation)...
pip install gmsh --quiet

echo.
echo ==============================================
echo Environment setup complete!
echo ==============================================
echo.
echo To activate the environment manually:
echo   %VENV_NAME%\Scripts\activate.bat
echo.
echo Running simulation...
echo.

REM Run the simulation
python run_simulation.py

echo.
echo Done! Check the generated images.
pause
