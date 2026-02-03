#!/bin/bash

# Repository Cleanup for Phase 1.5 ROI Extraction
# This script removes legacy files and creates a clean branch

set -e  # Exit on error

echo "========================================="
echo "Repository Cleanup for Phase 1.5"
echo "========================================="
echo ""

# Check if we're in the right directory
if [ ! -f "environment.yml" ]; then
    echo "❌ Error: Must run from project root directory"
    exit 1
fi

# Check current branch
CURRENT_BRANCH=$(git branch --show-current)
echo "Current branch: $CURRENT_BRANCH"
echo ""

# Confirm with user
read -p "Create new branch 'phase-1.5-roi-extraction' and cleanup? (y/n) " -n 1 -r
echo ""
if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    echo "Cancelled."
    exit 0
fi

# Create new branch
echo "Creating new branch: phase-1.5-roi-extraction"
git checkout -b phase-1.5-roi-extraction

echo ""
echo "Removing legacy files..."
echo ""

# Old feature engineering
echo "• Removing old feature engineering files..."
rm -f src/data_processing/feature_engineering_v2.py
rm -f src/data_processing/feature_engineering_v3.py
rm -f src/data_processing/angular_velocity_features.py
rm -f src/data_processing/contact_frame_features.py
rm -f src/data_processing/kinetic_chain_features.py
rm -f src/data_processing/phase_segmentation.py
rm -f src/data_processing/phase_specific_features.py
rm -f src/data_processing/feature_selection.py
rm -f src/data_processing/feature_versioning.py
rm -f src/data_processing/data_split.py

# Remove analysis directory
rm -rf src/analysis/

# Coaching system
echo "• Removing coaching system..."
rm -rf src/coaching/

# Deployment apps
echo "• Removing deployment apps..."
rm -rf src/deployment/

# Old models
echo "• Removing old model files..."
rm -f baseline_model_fixed.py
rm -rf src/models/
rm -rf src/evaluation/

# Old extraction scripts
echo "• Removing old extraction scripts..."
rm -f src/data_processing/extract_poses.py
rm -f scripts/extract_poses.py

# Phase 2 validation
echo "• Removing phase 2 validation files..."
rm -f scripts/validate_phase2.py
rm -f scripts/colab_phase2_validation.sh
rm -f notebooks/phase2_validation_colab.ipynb
rm -f notebooks/phase2_validation_workflow.md

# Old setup scripts
echo "• Removing old setup scripts..."
rm -f scripts/colab_setup.sh
rm -f scripts/verify_infra.py
rm -f scripts/gcs_setup.py
rm -f scripts/run_in_colab.py
rm -f scripts/sync_utils.py
rm -f scripts/pull_data.sh
rm -f scripts/push_results.sh
rm -f scripts/run_feature_selection.py
rm -f scripts/organize_videos.py
rm -f scripts/organize_gcs_videos.sh
rm -f setup_project.sh
rm -f setup_conda_env.sh
rm -f fix_dependencies.sh
rm -f run_local_extraction.sh
rm -f run_organize_data.sh
rm -f verify_milestone_simple.sh

# MLflow
echo "• Removing MLflow files..."
rm -rf mlruns/
rm -f config/mlflow.yaml
rm -f scripts/mlflow_config.py

# Old notebooks
echo "• Removing old notebooks..."
rm -f notebooks/00_initial_setup_colab.ipynb
rm -f notebooks/complete_workflow_colab.ipynb
rm -f notebooks/deep_learning_training_colab.ipynb
rm -f notebooks/phase3_model_training_colab.ipynb
rm -f notebooks/phase4_production_integration_colab.ipynb

# Old documentation
echo "• Removing old documentation..."
rm -f docs/GCS_DATASET_ANALYSIS.md
rm -f docs/CLIP_QUALITY_REVIEW.md
rm -f docs/VIDEO_ORGANIZATION_GUIDE.md
rm -f docs/RESUMABLE_POSE_EXTRACTION.md
rm -f docs/TOP_5_TRAINABLE_SHOTS_ANALYSIS.md
rm -f docs/MEDIAPIPE_VERSION_NOTES.md
rm -f outputs/reports/FINAL_PROJECT_REPORT.md
rm -f outputs/reports/feature_selection_report.md
rm -f outputs/reports/implementation_plan.md
rm -f outputs/reports/improvement_evaluation.md
rm -f outputs/reports/wrist_angle_analysis.md

# Old root files
echo "• Removing old root documentation..."
rm -f COLAB_QUICKSTART.md
rm -f QUICK_START.md
rm -f WORKFLOW_OVERVIEW.md
rm -f TRAINING_ANALYSIS.md
rm -f VIDEO_REQUIREMENTS.md

# Old analysis scripts
echo "• Removing old analysis scripts..."
rm -f analyze_stroke_types.py
rm -f analyze_video.py
rm -f diagnose.py
rm -f create_forehand_features_pkl.py
rm -f filter_backhand_and_regenerate.py
rm -f regenerate_features.py

# Tests (will add back later)
echo "• Removing tests (will add back later)..."
rm -rf tests/

# Planning files
echo "• Removing planning files..."
rm -rf .planning/

# Config files (keep only essential)
echo "• Cleaning config directory..."
rm -f config/mlflow.yaml
rm -f config/colab.yaml
rm -f config/paths.yaml

# Remove empty config directory if it exists
if [ -d "config" ] && [ -z "$(ls -A config)" ]; then
    rmdir config
fi

# Remove empty src subdirectories
echo "• Cleaning up empty directories..."
find src -type d -empty -delete 2>/dev/null || true

# If src is now empty, remove it
if [ -d "src" ] && [ -z "$(ls -A src)" ]; then
    rm -rf src
fi

# Remove pytest cache
rm -rf .pytest_cache/

# Remove notebooks README files that are no longer needed
rm -f notebooks/COLAB_UPLOAD_GUIDE.md

# Remove scripts README files that are outdated
rm -f scripts/README_organize_videos.md

echo ""
echo "========================================="
echo "Cleanup Summary"
echo "========================================="
echo ""

# Count remaining files
SCRIPT_COUNT=$(find scripts -name "*.py" -o -name "*.sh" | wc -l | tr -d ' ')
DOC_COUNT=$(find docs -name "*.md" | wc -l | tr -d ' ')
NOTEBOOK_COUNT=$(find notebooks -name "*.ipynb" | wc -l | tr -d ' ')

echo "Remaining essential files:"
echo "  Scripts:   $SCRIPT_COUNT"
echo "  Docs:      $DOC_COUNT"
echo "  Notebooks: $NOTEBOOK_COUNT"
echo ""

# Show what's left
echo "Core scripts:"
ls -1 scripts/*.py scripts/*.sh 2>/dev/null | sed 's|scripts/|  - |'
echo ""

echo "Documentation:"
ls -1 docs/*.md 2>/dev/null | sed 's|docs/|  - |'
echo ""

echo "Notebooks:"
ls -1 notebooks/*.ipynb 2>/dev/null | sed 's|notebooks/|  - |'
echo ""

# Stage all changes
echo "Staging changes..."
git add -A

# Show status
echo ""
echo "========================================="
echo "Git Status"
echo "========================================="
git status --short | head -20
echo ""

# Commit prompt
echo "Ready to commit changes."
read -p "Commit now? (y/n) " -n 1 -r
echo ""
if [[ $REPLY =~ ^[Yy]$ ]]; then
    git commit -m "chore: clean repository for Phase 1.5 ROI extraction

Removed legacy files:
- Old feature engineering (not used in deep learning)
- Coaching/feedback system (not needed for classification)
- Old phase 2 validation scripts
- MLflow integration files
- Old notebooks and documentation
- Tests (will add back later)
- Planning files
- Old setup and analysis scripts

Kept essential files:
- Core extraction and training scripts (5 scripts)
- Fixed training pipeline with normalization
- Essential documentation (8 docs)
- model_comparison_colab.ipynb (fixed version)
- environment.yml and .gitignore

Next: Implement ROI-based extraction using ShuttleSet player positions

Co-Authored-By: Claude Sonnet 4.5 <noreply@anthropic.com>"

    echo ""
    echo "✅ Cleanup complete!"
    echo ""
    echo "Next steps:"
    echo "1. Review remaining files: git log -1 --stat"
    echo "2. Update README.md for Phase 1.5"
    echo "3. Start ROI-based extraction implementation"
else
    echo ""
    echo "Changes staged but not committed."
    echo "Review with: git status"
    echo "Commit with: git commit"
fi

echo ""
echo "Branch: phase-1.5-roi-extraction"
echo "Status: Cleanup ready for Phase 1.5 work"
