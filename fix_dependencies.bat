@echo off
REM Fix MediaPipe and TensorFlow dependency conflicts
REM This script uninstalls conflicting packages and reinstalls compatible versions

echo ===========================================================================
echo Fixing MediaPipe + TensorFlow Dependency Conflicts
echo ===========================================================================
echo.

REM Check if we're in a conda environment
if "%CONDA_DEFAULT_ENV%"=="" (
    echo WARNING: No conda environment detected!
    echo Please run: conda activate iti123
    pause
    exit /b 1
)

echo Conda environment active: %CONDA_DEFAULT_ENV%
echo.

REM Step 1: Uninstall conflicting packages
echo Step 1: Removing conflicting packages...
pip uninstall -y tensorflow tensorflow-intel mediapipe protobuf

echo.
echo Step 2: Installing compatible versions...

REM Step 2: Install MediaPipe 0.10.9 (this will install protobuf 3.20.3)
pip install mediapipe==0.10.9

REM Step 3: Install TensorFlow 2.15.0 (compatible with protobuf 3.20.3)
pip install tensorflow==2.15.0

REM Step 4: Verify installations
echo.
echo Step 3: Verifying installations...
python -c "import mediapipe as mp; print(f'MediaPipe version: {mp.__version__}')"
python -c "import tensorflow as tf; print(f'TensorFlow version: {tf.__version__}')"
python -c "import google.protobuf; print(f'Protobuf version: {google.protobuf.__version__}')"

echo.
echo ===========================================================================
echo Dependency fix complete!
echo ===========================================================================
echo.
echo Compatible versions installed:
echo   - MediaPipe: 0.10.9
echo   - TensorFlow: 2.15.0
echo   - Protobuf: 3.20.3
echo.
pause
