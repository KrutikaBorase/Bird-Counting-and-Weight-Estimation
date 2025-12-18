@echo off
REM Quick setup script for Windows

echo ========================================
echo Bird Counting System - Quick Setup
echo ========================================
echo.

REM Check Python installation
python --version >nul 2>&1
if errorlevel 1 (
    echo ERROR: Python is not installed or not in PATH
    echo Please install Python 3.8 or higher from python.org
    pause
    exit /b 1
)

echo [1/4] Python found
python --version

REM Create virtual environment
echo.
echo [2/4] Creating virtual environment...
if not exist venv (
    python -m venv venv
    echo Virtual environment created
) else (
    echo Virtual environment already exists
)

REM Activate virtual environment and install dependencies
echo.
echo [3/4] Installing dependencies...
call venv\Scripts\activate.bat
pip install --upgrade pip
pip install -r requirements.txt

if errorlevel 1 (
    echo ERROR: Failed to install dependencies
    pause
    exit /b 1
)

echo.
echo [4/4] Creating directories...
if not exist data\input mkdir data\input
if not exist data\sample mkdir data\sample
if not exist outputs\videos mkdir outputs\videos
if not exist outputs\json mkdir outputs\json
if not exist models mkdir models

echo.
echo ========================================
echo Setup Complete!
echo ========================================
echo.
echo Next steps:
echo 1. Download a poultry farm video dataset (see DATASET_GUIDE.md)
echo 2. Place videos in data\sample\
echo 3. Start the API server: python main.py
echo 4. Test the system: python test_api.py data\sample\your_video.mp4
echo.
echo For more information, see README.md
echo.
pause
