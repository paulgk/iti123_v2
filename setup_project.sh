#!/bin/bash

# ITI123 Project Setup Script
# AI-Based Badminton Stroke Technique Assessment
# This script creates the complete folder structure for the project

echo "=========================================="
echo "ITI123 Project Folder Structure Setup"
echo "=========================================="
echo ""

# Get the script's directory
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "$SCRIPT_DIR"

echo "Creating project folder structure in: $SCRIPT_DIR"
echo ""

# Create main data folders
echo "Creating data folders..."
mkdir -p data/raw_videos
mkdir -p data/annotations
mkdir -p data/processed/clips
mkdir -p data/processed/poses
mkdir -p data/processed/features

# Create notebooks folder
echo "Creating notebooks folder..."
mkdir -p notebooks

# Create source code folders
echo "Creating source code folders..."
mkdir -p src/data_processing
mkdir -p src/models
mkdir -p src/evaluation
mkdir -p src/deployment

# Create experiments folder (for MLflow)
echo "Creating experiments folder..."
mkdir -p experiments

# Create models folder (for saved models)
echo "Creating models folder..."
mkdir -p models

# Create outputs folder
echo "Creating outputs folder..."
mkdir -p outputs/plots
mkdir -p outputs/reports
mkdir -p outputs/visualizations

# Create docs folder
echo "Creating documentation folder..."
mkdir -p docs

# Create placeholder __init__.py files for Python packages
echo "Creating Python package files..."
touch src/__init__.py
touch src/data_processing/__init__.py
touch src/models/__init__.py
touch src/evaluation/__init__.py
touch src/deployment/__init__.py

# Create .gitkeep files to preserve empty folders in git
echo "Creating .gitkeep files for git..."
touch data/raw_videos/.gitkeep
touch data/annotations/.gitkeep
touch data/processed/clips/.gitkeep
touch data/processed/poses/.gitkeep
touch data/processed/features/.gitkeep
touch experiments/.gitkeep
touch models/.gitkeep
touch outputs/plots/.gitkeep
touch outputs/reports/.gitkeep
touch outputs/visualizations/.gitkeep

# Create a .gitignore file
echo "Creating .gitignore file..."
cat > .gitignore << 'EOF'
# Python
__pycache__/
*.py[cod]
*$py.class
*.so
.Python
env/
venv/
ENV/
build/
develop-eggs/
dist/
downloads/
eggs/
.eggs/
lib/
lib64/
parts/
sdist/
var/
wheels/
*.egg-info/
.installed.cfg
*.egg

# Jupyter Notebook
.ipynb_checkpoints
*.ipynb_checkpoints/

# Virtual Environment
venv/
ENV/
env/

# IDEs
.vscode/
.idea/
*.swp
*.swo
*~

# OS
.DS_Store
Thumbs.db

# Project specific - Large files
data/raw_videos/*.mp4
data/raw_videos/*.avi
data/raw_videos/*.mov
data/processed/clips/*.mp4
data/processed/clips/*.avi
data/processed/poses/*.npy
data/processed/poses/*.pkl
data/processed/features/*.npy
data/processed/features/*.pkl

# Models (keep only final model for submission)
models/*.h5
models/*.pth
models/*.pkl
!models/final_model.h5
!models/final_model.pth

# MLflow
mlruns/
experiments/

# Outputs (keep important visualizations only)
outputs/plots/*.png
outputs/plots/*.jpg
!outputs/plots/final_*.png
outputs/reports/*.html

# Temporary files
*.tmp
*.log
.cache/

# Large data files
*.zip
*.tar.gz
*.csv.gz
EOF

echo ""
echo "=========================================="
echo "Folder structure created successfully!"
echo "=========================================="
echo ""
echo "Project structure:"
tree -L 3 -I 'venv|__pycache__|.git' 2>/dev/null || find . -type d -maxdepth 3 | grep -v '.git' | grep -v 'venv' | sed 's|[^/]*/| |g'

echo ""
echo "Next steps:"
echo "1. Create virtual environment: python3 -m venv venv"
echo "2. Activate it: source venv/bin/activate  (or 'venv\\Scripts\\activate' on Windows)"
echo "3. Install requirements: pip install -r requirements.txt"
echo ""
echo "Note: Raw video files should be placed in: data/raw_videos/"
echo "      ShuttleSet annotations should be placed in: data/annotations/"
echo ""
