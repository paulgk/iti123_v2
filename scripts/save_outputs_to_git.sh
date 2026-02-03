#!/bin/bash
#
# Save Training Outputs to Git Repository
#
# This script copies outputs from Colab/external to the git repo and commits them
#

set -e

echo "========================================"
echo "SAVING OUTPUTS TO GIT REPOSITORY"
echo "========================================"
echo ""

# Configuration
REPO_DIR="/Volumes/Ext/GenAI/iti123_v2"
OUTPUTS_SRC="${1:-outputs}"  # First argument or default to 'outputs'

# Check if we're in the repo
cd "$REPO_DIR"

if [ ! -d ".git" ]; then
    echo "❌ Error: Not in a git repository"
    exit 1
fi

echo "Repository: $REPO_DIR"
echo "Source outputs: $OUTPUTS_SRC"
echo ""

# Create outputs directories in repo if they don't exist
mkdir -p outputs/models
mkdir -p outputs/reports
mkdir -p outputs/visualizations

# Check if source exists
if [ ! -d "$OUTPUTS_SRC" ]; then
    echo "❌ Error: Source directory not found: $OUTPUTS_SRC"
    echo ""
    echo "Usage: $0 [path/to/outputs]"
    echo ""
    echo "Example:"
    echo "  $0 /path/to/colab/outputs"
    echo ""
    exit 1
fi

# Copy files
echo "Copying files..."
echo ""

# Copy models (only .pth files, not large data files)
if [ -d "$OUTPUTS_SRC/models" ]; then
    echo "📦 Models:"
    rsync -av --include="*.pth" --include="*.pkl" --exclude="*" "$OUTPUTS_SRC/models/" outputs/models/
    ls -lh outputs/models/
    echo ""
fi

# Copy reports (text files, CSVs, etc.)
if [ -d "$OUTPUTS_SRC/reports" ]; then
    echo "📊 Reports:"
    rsync -av --include="*.txt" --include="*.md" --include="*.csv" --include="*.json" --exclude="*" "$OUTPUTS_SRC/reports/" outputs/reports/
    ls -lh outputs/reports/
    echo ""
fi

# Copy visualizations (images)
if [ -d "$OUTPUTS_SRC" ]; then
    echo "📈 Visualizations:"
    find "$OUTPUTS_SRC" -name "*.png" -o -name "*.jpg" -o -name "*.pdf" | while read img; do
        cp "$img" outputs/visualizations/
    done
    ls -lh outputs/visualizations/ 2>/dev/null || echo "  (no visualizations found)"
    echo ""
fi

# Check git status
echo "========================================"
echo "GIT STATUS"
echo "========================================"
echo ""

git status outputs/

echo ""
echo "========================================"
echo "FILES READY TO COMMIT"
echo "========================================"
echo ""

# Show what will be committed
git diff --stat outputs/
git diff --cached --stat outputs/

echo ""
read -p "Do you want to commit these changes? (y/n): " -n 1 -r
echo ""

if [[ $REPLY =~ ^[Yy]$ ]]; then
    # Stage files
    git add outputs/

    # Generate commit message
    TIMESTAMP=$(date +"%Y-%m-%d %H:%M:%S")
    COMMIT_MSG="chore: update training outputs - $TIMESTAMP

Updated files:
$(git diff --cached --name-status outputs/ | head -10)
"

    if [ $(git diff --cached --name-status outputs/ | wc -l) -gt 10 ]; then
        COMMIT_MSG="$COMMIT_MSG
... and $(( $(git diff --cached --name-status outputs/ | wc -l) - 10 )) more files"
    fi

    # Commit
    git commit -m "$COMMIT_MSG"

    echo ""
    echo "✓ Changes committed!"
    echo ""
    echo "To push to remote:"
    echo "  git push origin $(git branch --show-current)"
    echo ""
else
    echo ""
    echo "Commit cancelled. Files are staged but not committed."
    echo ""
    echo "To commit later:"
    echo "  git add outputs/"
    echo "  git commit -m 'chore: update training outputs'"
    echo ""
fi

echo "========================================"
