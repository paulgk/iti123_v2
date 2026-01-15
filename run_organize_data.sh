#!/bin/bash
# Wrapper script to run data organization in conda environment

echo "Running data organization script..."
echo "Note: Make sure you have activated your conda environment (conda activate iti123)"
echo ""

# Check if we're in a conda environment
if [ -z "$CONDA_DEFAULT_ENV" ]; then
    echo "⚠️  WARNING: No conda environment detected!"
    echo "Please run: conda activate iti123"
    echo "Then run this script again, or run directly:"
    echo "  python src/data_processing/organize_data.py"
    exit 1
fi

echo "✅ Conda environment active: $CONDA_DEFAULT_ENV"
echo ""

# Run the script
python src/data_processing/organize_data.py
