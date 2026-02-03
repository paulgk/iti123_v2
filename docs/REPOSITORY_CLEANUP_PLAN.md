# Repository Cleanup Plan

**Date:** 2026-02-03
**Purpose:** Clean up repository for Phase 1.5 (ROI-based player-specific extraction)

---

## Current State

- **Total files:** 100+ files including legacy code, old notebooks, and planning files
- **Issue:** Mix of essential, legacy, and outdated files making navigation difficult
- **Goal:** Create clean branch with only files needed for Phase 1.5 work

---

## File Categorization

### ✅ ESSENTIAL - Keep These Files

#### Core Scripts (scripts/)
- `extract_shuttleset_clips.py` - Clip extraction from ShuttleSet videos (needs ROI modification)
- `extract_poses_parallel.py` - Pose extraction (needs ROI support)
- `train_models_fixed.py` - Fixed training script with normalization
- `upload_clips_to_gcs.sh` - Smart upload script for GCS
- `save_outputs_to_git.sh` - Save training outputs to repo

#### Documentation (docs/)
- `FIXES_APPLIED.md` - Critical fixes documentation
- `MULTI_PLAYER_ISSUE.md` - Multi-player detection problem analysis
- `TRAINING_WORKFLOW.md` - End-to-end workflow guide
- `CONDA_SETUP_GUIDE.md` - Local environment setup
- `SHUTTLESET_DATASET_ANALYSIS.md` - Dataset structure and analysis
- `SHUTTLESET_EXTRACTION_GUIDE.md` - Extraction documentation
- `COLAB_SETUP_GUIDE.md` - Colab setup for training
- `SMART_UPLOAD_GUIDE.md` - Upload script documentation

#### Notebooks (notebooks/)
- `model_comparison_colab.ipynb` - Fixed training notebook with proper normalization

#### Configuration
- `environment.yml` - Conda environment specification
- `.gitignore` - Git ignore rules

#### Root Documentation
- `README.md` - Project overview (needs update for Phase 1.5)

#### Data (keep structure, not files)
- `ShuttleSet/` - Source dataset directory structure
- `data/` - Processed data directory structure

---

### ❌ LEGACY - Remove or Archive These Files

#### Old Training Scripts
- `baseline_model_fixed.py` - Old baseline model (superseded by train_models_fixed.py)
- `src/models/baseline_model.py` - Old model code
- `src/models/lstm_model.py` - Old LSTM implementation (now in train_models_fixed.py)

#### Old Feature Engineering (Not Used for Deep Learning)
- `src/data_processing/feature_engineering_v2.py`
- `src/data_processing/feature_engineering_v3.py`
- `src/data_processing/angular_velocity_features.py`
- `src/data_processing/contact_frame_features.py`
- `src/data_processing/kinetic_chain_features.py`
- `src/data_processing/phase_segmentation.py`
- `src/data_processing/phase_specific_features.py`
- `src/data_processing/feature_selection.py`
- `src/data_processing/feature_versioning.py`
- `create_forehand_features_pkl.py`
- `filter_backhand_and_regenerate.py`
- `regenerate_features.py`

#### Coaching/Feedback System (Not Used for Shot Classification)
- `src/coaching/` - Entire directory
- `src/deployment/` - Streamlit apps (not needed yet)

#### Old Analysis Scripts
- `analyze_stroke_types.py`
- `analyze_video.py`
- `diagnose.py`
- `src/analysis/analyze_wrist_features.py`

#### Old Extraction Scripts
- `src/data_processing/extract_poses.py` - Old version (superseded by parallel script)
- `scripts/extract_poses.py` - Old non-parallel version

#### Phase 2 Validation (Completed)
- `scripts/validate_phase2.py`
- `scripts/colab_phase2_validation.sh`
- `notebooks/phase2_validation_colab.ipynb`
- `notebooks/phase2_validation_workflow.md`

#### Old Setup Scripts
- `scripts/colab_setup.sh` - Old Colab setup
- `scripts/verify_infra.py` - Old infrastructure verification
- `scripts/gcs_setup.py` - Old GCS setup
- `setup_project.sh` - Old setup script
- `fix_dependencies.sh` - Old dependency fix

#### MLflow Integration (Not Used)
- `config/mlflow.yaml`
- `scripts/mlflow_config.py`
- `mlruns/` - MLflow tracking data

#### Old Notebooks
- `notebooks/00_initial_setup_colab.ipynb` - Old setup
- `notebooks/complete_workflow_colab.ipynb` - Old combined workflow
- `notebooks/deep_learning_training_colab.ipynb` - Old training (superseded)
- `notebooks/phase3_model_training_colab.ipynb` - Old phase 3
- `notebooks/phase4_production_integration_colab.ipynb` - Future phase

#### Old Documentation
- `docs/GCS_DATASET_ANALYSIS.md` - Old GCS analysis
- `docs/CLIP_QUALITY_REVIEW.md` - Old clip review
- `docs/VIDEO_ORGANIZATION_GUIDE.md` - Old organization guide
- `docs/RESUMABLE_POSE_EXTRACTION.md` - Old resumable extraction
- `docs/TOP_5_TRAINABLE_SHOTS_ANALYSIS.md` - Old shot analysis
- `outputs/reports/*.md` - Old feature engineering reports

#### Old Root Files
- `COLAB_QUICKSTART.md` - Old quickstart
- `QUICK_START.md` - Old quickstart
- `WORKFLOW_OVERVIEW.md` - Old workflow
- `TRAINING_ANALYSIS.md` - Old analysis
- `VIDEO_REQUIREMENTS.md` - Old requirements

#### Tests (Not Currently Used)
- `tests/` - All test files (can add back later)

---

## ⚠️ UNCERTAIN - Review Before Deciding

#### Planning Files (.planning/)
- **Option 1:** Keep entire `.planning/` directory for project history
- **Option 2:** Remove `.planning/` to start fresh
- **Recommendation:** Remove for clean start, archive separately if needed

#### Scripts - Utility
- `scripts/create_metadata_from_poses.py` - May be useful for validation
- `scripts/translate_shuttleset_csvs.py` - Already completed, may archive
- `scripts/organize_videos.py` - May be useful for data organization
- `scripts/check_extraction_status.sh` - May be useful for monitoring

#### Config Files
- `config/paths.yaml` - May be useful for path management
- `config/colab.yaml` - May be useful for Colab configuration

---

## New Branch Structure

```
iti123_v2/
├── README.md                                   # Updated for Phase 1.5
├── environment.yml                             # Conda environment
├── .gitignore                                  # Git ignore rules
│
├── scripts/                                    # Core scripts
│   ├── extract_shuttleset_clips.py            # ROI-based clip extraction
│   ├── extract_poses_parallel.py              # Pose extraction with ROI
│   ├── train_models_fixed.py                  # Fixed training script
│   ├── upload_clips_to_gcs.sh                 # Smart GCS upload
│   └── save_outputs_to_git.sh                 # Save outputs to git
│
├── docs/                                       # Essential documentation
│   ├── FIXES_APPLIED.md
│   ├── MULTI_PLAYER_ISSUE.md
│   ├── TRAINING_WORKFLOW.md
│   ├── CONDA_SETUP_GUIDE.md
│   ├── SHUTTLESET_DATASET_ANALYSIS.md
│   ├── SHUTTLESET_EXTRACTION_GUIDE.md
│   ├── COLAB_SETUP_GUIDE.md
│   └── SMART_UPLOAD_GUIDE.md
│
├── notebooks/                                  # Training notebooks
│   └── model_comparison_colab.ipynb           # Fixed training notebook
│
├── ShuttleSet/                                 # Source dataset (structure only)
│   ├── match*/                                # Match directories
│   └── Annotations/                           # CSV annotations
│
├── data/                                       # Processed data (structure only)
│   ├── clips/                                 # Extracted clips
│   └── processed/                             # Processed features
│       └── poses/                             # Pose files
│
└── outputs/                                    # Training outputs
    ├── models/                                # Model weights
    ├── reports/                               # Training reports
    └── visualizations/                        # Training curves
```

---

## Migration Plan

### Step 1: Create New Branch
```bash
git checkout -b phase-1.5-roi-extraction
```

### Step 2: Remove Legacy Files
```bash
# Remove old feature engineering
rm -rf src/data_processing/feature_engineering*.py
rm -rf src/data_processing/angular_velocity_features.py
rm -rf src/data_processing/contact_frame_features.py
rm -rf src/data_processing/kinetic_chain_features.py
rm -rf src/data_processing/phase_*.py
rm -rf src/analysis/

# Remove coaching system
rm -rf src/coaching/
rm -rf src/deployment/

# Remove old models
rm baseline_model_fixed.py
rm -rf src/models/

# Remove old extraction
rm -rf src/data_processing/extract_poses.py

# Remove phase 2 validation
rm scripts/validate_phase2.py
rm scripts/colab_phase2_validation.sh
rm notebooks/phase2_validation*.ipynb
rm notebooks/phase2_validation_workflow.md

# Remove old setup scripts
rm scripts/colab_setup.sh
rm scripts/verify_infra.py
rm scripts/gcs_setup.py
rm setup_project.sh
rm fix_dependencies.sh

# Remove MLflow
rm -rf mlruns/
rm config/mlflow.yaml
rm scripts/mlflow_config.py

# Remove old notebooks
rm notebooks/00_initial_setup_colab.ipynb
rm notebooks/complete_workflow_colab.ipynb
rm notebooks/deep_learning_training_colab.ipynb
rm notebooks/phase3_model_training_colab.ipynb
rm notebooks/phase4_production_integration_colab.ipynb

# Remove old documentation
rm docs/GCS_DATASET_ANALYSIS.md
rm docs/CLIP_QUALITY_REVIEW.md
rm docs/VIDEO_ORGANIZATION_GUIDE.md
rm docs/RESUMABLE_POSE_EXTRACTION.md
rm docs/TOP_5_TRAINABLE_SHOTS_ANALYSIS.md

# Remove old root files
rm COLAB_QUICKSTART.md
rm QUICK_START.md
rm WORKFLOW_OVERVIEW.md
rm TRAINING_ANALYSIS.md
rm VIDEO_REQUIREMENTS.md

# Remove old analysis scripts
rm analyze_stroke_types.py
rm analyze_video.py
rm diagnose.py
rm create_forehand_features_pkl.py
rm filter_backhand_and_regenerate.py
rm regenerate_features.py

# Remove tests (add back later)
rm -rf tests/

# Remove planning files
rm -rf .planning/

# Remove old scripts
rm scripts/extract_poses.py
```

### Step 3: Update README.md
Update root README.md to reflect Phase 1.5 focus:
- ROI-based clip extraction using ShuttleSet player positions
- Proper player-to-shot alignment
- Both player extraction (2x dataset)
- Fixed normalization and training

### Step 4: Commit Clean Branch
```bash
git add -A
git commit -m "chore: clean repository for Phase 1.5 ROI extraction

Removed legacy files:
- Old feature engineering (not used in deep learning)
- Coaching/feedback system (not needed for classification)
- Old phase 2 validation scripts
- MLflow integration files
- Old notebooks and documentation
- Tests (will add back later)

Kept essential files:
- Core extraction and training scripts
- Fixed training pipeline with normalization
- Essential documentation
- model_comparison_colab.ipynb (fixed version)

Next: Implement ROI-based extraction using ShuttleSet player positions"
```

---

## Files to Keep Count

**Essential files: ~20 files**
- 5 core scripts
- 8 documentation files
- 1 notebook
- 3 config files (environment.yml, .gitignore, README.md)
- Directory structures (data/, ShuttleSet/, outputs/)

**Legacy files to remove: ~80+ files**

---

## Next Steps After Cleanup

1. **Update extract_shuttleset_clips.py**
   - Add `--player` argument (A, B, or both)
   - Use `player_location_x/y` for ROI calculation
   - Update clip naming: `{matchID}_set{N}_rally{R}_ball{B}_{shotType}_player{A/B}.mp4`

2. **Update extract_poses_parallel.py**
   - Add ROI support for player-specific extraction
   - Crop frame to player region before MediaPipe processing
   - Ensure pose coordinates remain in original frame space

3. **Create validation script**
   - Verify correct player-to-shot alignment
   - Check ROI covers player properly
   - Validate pose quality in ROI crops

4. **Re-extract full dataset**
   - ~27K clips → ~54K samples (both players)
   - Expected time: 8-12 hours locally with 8 workers
   - Upload to GCS with smart upload script

5. **Re-train models**
   - Expected improvement: Better accuracy with correct player data
   - Use fixed training script (already has normalization)
   - Expected results: 85-90% ST-GCN accuracy

---

**Status:** Plan ready for user approval
**Estimated cleanup time:** 10 minutes
**Risk:** Low (new branch, no data loss)
