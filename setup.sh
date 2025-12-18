#!/bin/bash
# Quick setup script for Linux/Mac

echo "========================================"
echo "Bird Counting System - Quick Setup"
echo "========================================"
echo ""

# Check Python installation
if ! command -v python3 &> /dev/null; then
    echo "ERROR: Python 3 is not installed"
    echo "Please install Python 3.8 or higher"
    exit 1
fi

echo "[1/4] Python found"
python3 --version

# Create virtual environment
echo ""
echo "[2/4] Creating virtual environment..."
if [ ! -d "venv" ]; then
    python3 -m venv venv
    echo "Virtual environment created"
else
    echo "Virtual environment already exists"
fi

# Activate virtual environment and install dependencies
echo ""
echo "[3/4] Installing dependencies..."
source venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt

if [ $? -ne 0 ]; then
    echo "ERROR: Failed to install dependencies"
    exit 1
fi

# Create directories
echo ""
echo "[4/4] Creating directories..."
mkdir -p data/input
mkdir -p data/sample
mkdir -p outputs/videos
mkdir -p outputs/json
mkdir -p models

echo ""
echo "========================================"
echo "Setup Complete!"
echo "========================================"
echo ""
echo "Next steps:"
echo "1. Download a poultry farm video dataset (see DATASET_GUIDE.md)"
echo "2. Place videos in data/sample/"
echo "3. Activate environment: source venv/bin/activate"
echo "4. Start the API server: python main.py"
echo "5. Test the system: python test_api.py data/sample/your_video.mp4"
echo ""
echo "For more information, see README.md"
echo ""
