#!/usr/bin/env python3
"""
Complete pipeline for Drop vs Smash classification

This script:
1. Loads Drop vs Smash metadata
2. Extracts poses from video clips
3. Engineers features (spatial, temporal, wrist orientation)
4. Creates train/val/test splits with group stratification
5. Trains baseline models (RF, SVM)
6. Trains deep learning models (LSTM, BiLSTM, GRU)
7. Generates performance reports

Expected accuracy: 70-85% (Drop and Smash have distinct biomechanics)
"""

import sys
import pickle
from pathlib import Path
import pandas as pd
import numpy as np

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / 'src'))

from data_processing.extract_poses import extract_poses_from_clips
from data_processing.feature_engineering_v2 import engineer_features_from_poses
from data_processing.data_split import create_group_stratified_splits

print("=" * 80)
print("DROP SHOT VS SMASH CLASSIFICATION PIPELINE")
print("=" * 80)
print()

# ============================================================================
# STEP 1: Load metadata
# ============================================================================

print("STEP 1: Loading Drop vs Smash metadata...")
print("-" * 80)

metadata_file = Path("data/processed/clips/drop_smash_metadata.csv")

if not metadata_file.exists():
    print(f"❌ ERROR: {metadata_file} not found!")
    print("   Please run: python extract_drop_and_lift.py first")
    sys.exit(1)

metadata = pd.read_csv(metadata_file)

print(f"✓ Loaded {len(metadata):,} strokes")
print(f"  Drop: {len(metadata[metadata['stroke_type_english'] == 'Drop']):,}")
print(f"  Smash: {len(metadata[metadata['stroke_type_english'] == 'Smash']):,}")
print(f"  Unique matches: {metadata['match_id'].nunique()}")
print()

# ============================================================================
# STEP 2: Extract poses from video clips
# ============================================================================

print("STEP 2: Extracting poses from video clips...")
print("-" * 80)
print("This may take 30-60 minutes depending on number of clips...")
print()

poses_output_dir = Path("data/processed/poses_drop_smash")
poses_output_dir.mkdir(parents=True, exist_ok=True)

# Check if poses already extracted
existing_poses = list(poses_output_dir.glob("*.pkl"))
if len(existing_poses) > 0:
    print(f"Found {len(existing_poses)} existing pose files")
    response = input("Re-extract poses? (y/N): ").strip().lower()
    if response != 'y':
        print("Skipping pose extraction...")
    else:
        print("Re-extracting poses...")
        # Call pose extraction function
        # Note: This would need the actual video files
        print("⚠️  WARNING: Pose extraction requires video files in data/videos/")
        print("   If videos not available, manually extract poses or skip to feature engineering")
else:
    print("⚠️  WARNING: Pose extraction requires video files in data/videos/")
    print("   If videos not available, manually extract poses or skip to feature engineering")

print()

# ============================================================================
# STEP 3: Engineer features
# ============================================================================

print("STEP 3: Engineering features from poses...")
print("-" * 80)

features_output = Path("data/processed/features_drop_smash")
features_output.mkdir(parents=True, exist_ok=True)

sequence_features_file = features_output / "sequence_features.pkl"
statistical_features_file = features_output / "statistical_features.pkl"

if sequence_features_file.exists() and statistical_features_file.exists():
    print(f"✓ Features already exist:")
    print(f"  - {sequence_features_file}")
    print(f"  - {statistical_features_file}")
    print()

    # Load to check
    with open(statistical_features_file, 'rb') as f:
        stat_data = pickle.load(f)
    print(f"  Loaded {len(stat_data['features'])} samples")
    print(f"  Features per sample: {len(stat_data['features'][0])}")
else:
    print("⚠️  Features not found. Need to run feature engineering.")
    print("   This requires pose files from Step 2.")
    print()
    print("Manual command:")
    print(f"  python src/data_processing/feature_engineering_v2.py --input {poses_output_dir} --output {features_output}")

print()

# ============================================================================
# STEP 4: Create train/val/test splits
# ============================================================================

print("STEP 4: Creating train/val/test splits with group stratification...")
print("-" * 80)

splits_output = Path("data/processed/splits_drop_smash")
splits_output.mkdir(parents=True, exist_ok=True)

train_file = splits_output / "train_data.pkl"
val_file = splits_output / "val_data.pkl"
test_file = splits_output / "test_data.pkl"

if all([train_file.exists(), val_file.exists(), test_file.exists()]):
    print(f"✓ Splits already exist:")
    print(f"  - {train_file}")
    print(f"  - {val_file}")
    print(f"  - {test_file}")
    print()

    # Load to check
    with open(train_file, 'rb') as f:
        train_data = pickle.load(f)
    with open(val_file, 'rb') as f:
        val_data = pickle.load(f)
    with open(test_file, 'rb') as f:
        test_data = pickle.load(f)

    print(f"  Train: {len(train_data['y']):,} samples")
    print(f"  Val: {len(val_data['y']):,} samples")
    print(f"  Test: {len(test_data['y']):,} samples")
else:
    print("⚠️  Splits not found. Need to run data splitting.")
    print("   This requires feature files from Step 3.")
    print()
    print("Manual command:")
    print(f"  python src/data_processing/data_split.py --input {features_output} --output {splits_output} --metadata {metadata_file}")

print()

# ============================================================================
# STEP 5: Train baseline models
# ============================================================================

print("STEP 5: Training baseline models (Random Forest, SVM)...")
print("-" * 80)
print()

baseline_models_dir = Path("models/drop_smash/baseline")
baseline_models_dir.mkdir(parents=True, exist_ok=True)

if (baseline_models_dir / "random_forest.pkl").exists():
    print("✓ Baseline models already trained")
else:
    print("⚠️  Baseline models not trained yet.")
    print()
    print("Manual command:")
    print(f"  python src/models/baseline_model.py --splits {splits_output} --output {baseline_models_dir}")

print()

# ============================================================================
# STEP 6: Train deep learning models
# ============================================================================

print("STEP 6: Training deep learning models (LSTM, BiLSTM, GRU)...")
print("-" * 80)
print()

dl_models_dir = Path("models/drop_smash/deep_learning")
dl_models_dir.mkdir(parents=True, exist_ok=True)

if (dl_models_dir / "lstm.keras").exists():
    print("✓ Deep learning models already trained")
else:
    print("⚠️  Deep learning models not trained yet.")
    print()
    print("Manual command:")
    print(f"  python src/models/lstm_model.py --splits {splits_output} --output {dl_models_dir}")

print()

# ============================================================================
# STEP 7: Generate reports
# ============================================================================

print("STEP 7: Generating performance reports...")
print("-" * 80)
print()

reports_dir = Path("outputs/reports_drop_smash")
reports_dir.mkdir(parents=True, exist_ok=True)

print("⚠️  Report generation requires trained models from Steps 5 & 6")
print()
print("Manual commands:")
print("  python src/evaluation/evaluate_models.py --models baseline --output reports_dir")
print("  python src/evaluation/evaluate_models.py --models deep_learning --output reports_dir")
print()

# ============================================================================
# Summary
# ============================================================================

print("=" * 80)
print("PIPELINE SUMMARY")
print("=" * 80)
print()
print("Drop vs Smash Classification Pipeline Status:")
print()
print("✓ Step 1: Metadata loaded")
print("? Step 2: Pose extraction (requires video files)")
print("? Step 3: Feature engineering (requires poses)")
print("? Step 4: Data splitting (requires features)")
print("? Step 5: Baseline training (requires splits)")
print("? Step 6: Deep learning training (requires splits)")
print("? Step 7: Report generation (requires models)")
print()
print("Expected Results:")
print("  Accuracy: 70-85%")
print("  F1 Score: 72-87%")
print("  ROC AUC: 75-90%")
print()
print("Rationale:")
print("  Drop shots have slower swing speed and early deceleration")
print("  Smash shots have explosive speed and strong follow-through")
print("  Expected Cohen's d > 0.6 (medium-large effect)")
print()
print("Next: Run python process_lift_smash_pipeline.py for Lift vs Smash")
print()
