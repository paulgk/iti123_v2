#!/bin/bash
# Colab Setup Script
# Run this at the start of each Colab session
# Usage: source scripts/colab_setup.sh

set -e

echo "=== Colab Environment Setup ==="

# Check if in Colab
if [ ! -d "/content" ]; then
    echo "Warning: Not in Colab environment. Some steps may fail."
fi

# Step 1: Check Python version
PYTHON_VERSION=$(python3 --version 2>&1 | awk '{print $2}' | cut -d. -f1,2)
echo "Current Python: $PYTHON_VERSION"

if [ "$PYTHON_VERSION" != "3.10" ]; then
    echo ""
    echo "=== Installing Python 3.10 ==="
    # Install Python 3.10 (Colab has it available but not default)
    apt-get update -qq
    apt-get install -y -qq python3.10 python3.10-venv python3.10-distutils

    # Create and activate virtual environment
    echo "Creating Python 3.10 virtual environment..."
    python3.10 -m venv /content/venv
    source /content/venv/bin/activate

    # Verify
    echo "Python version in venv: $(python --version)"
else
    echo "Python 3.10 already active"
fi

# Step 2: Install dependencies
echo ""
echo "=== Installing Dependencies ==="
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(dirname "$SCRIPT_DIR")"

if [ -f "$REPO_ROOT/requirements-colab.txt" ]; then
    pip install -q -r "$REPO_ROOT/requirements-colab.txt"
    echo "Dependencies installed from requirements-colab.txt"
else
    echo "Warning: requirements-colab.txt not found"
fi

# Step 3: Setup GCS authentication (if not already done)
echo ""
echo "=== GCS Authentication ==="
if [ -z "$GOOGLE_APPLICATION_CREDENTIALS" ]; then
    echo "Note: GOOGLE_APPLICATION_CREDENTIALS not set"
    echo "Run: export GOOGLE_APPLICATION_CREDENTIALS=/path/to/key.json"
    echo "Or authenticate with: gcloud auth application-default login"
else
    echo "Using credentials: $GOOGLE_APPLICATION_CREDENTIALS"
fi

# Step 4: Verify key imports
echo ""
echo "=== Verifying Imports ==="
python -c "import tensorflow as tf; print(f'TensorFlow: {tf.__version__}')" 2>/dev/null || echo "TensorFlow import failed"
python -c "import mediapipe as mp; print(f'MediaPipe: {mp.__version__}')" 2>/dev/null || echo "MediaPipe import failed"
python -c "import mlflow; print(f'MLflow: {mlflow.__version__}')" 2>/dev/null || echo "MLflow import failed"

echo ""
echo "=== Setup Complete ==="
echo "Next steps:"
echo "  1. Pull data: ./scripts/pull_data.sh --features"
echo "  2. Run training: python scripts/run_in_colab.py your_script.py"
echo "  3. Push results: ./scripts/push_results.sh --outputs"
