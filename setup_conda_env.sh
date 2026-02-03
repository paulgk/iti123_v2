#!/bin/bash
#
# Setup Conda Environment for Pose Extraction
#

echo "=========================================="
echo "Setting up Conda Environment for ITI123"
echo "=========================================="
echo ""

# Check if conda is available
if ! command -v conda &> /dev/null; then
    echo "❌ Conda not found!"
    echo ""
    echo "Please install Miniconda or Anaconda first:"
    echo "  https://docs.conda.io/en/latest/miniconda.html"
    echo ""
    echo "After installation, restart your terminal and run this script again."
    exit 1
fi

echo "✓ Conda found: $(conda --version)"
echo ""

# Create environment
echo "Creating conda environment from environment.yml..."
conda env create -f environment.yml

echo ""
echo "=========================================="
echo "Setup Complete!"
echo "=========================================="
echo ""
echo "To activate the environment:"
echo "  conda activate iti123"
echo ""
echo "Then run pose extraction:"
echo "  python scripts/extract_poses_parallel.py --video-dir data/clips/ --output-dir data/processed/poses/ --num-workers 8"
echo ""
