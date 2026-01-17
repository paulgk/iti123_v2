#!/usr/bin/env python3
"""
Complete pipeline for Lift vs Smash classification

This script processes Lift/Lob vs Smash strokes with the same pipeline
as Drop vs Smash. Lift shots are underhand defensive strokes, making them
very distinct from overhead Smash strokes.

Expected accuracy: 75-90% (Lift and Smash have very different biomechanics)
"""

import sys
import pickle
from pathlib import Path
import pandas as pd

print("=" * 80)
print("LIFT/LOB VS SMASH CLASSIFICATION PIPELINE")
print("=" * 80)
print()

# ============================================================================
# STEP 1: Load metadata
# ============================================================================

print("STEP 1: Loading Lift vs Smash metadata...")
print("-" * 80)

metadata_file = Path("data/processed/clips/lift_smash_metadata.csv")

if not metadata_file.exists():
    print(f"❌ ERROR: {metadata_file} not found!")
    print("   Please run: python extract_drop_and_lift.py first")
    sys.exit(1)

metadata = pd.read_csv(metadata_file)

print(f"✓ Loaded {len(metadata):,} strokes")
print(f"  Lift: {len(metadata[metadata['stroke_type_english'] == 'Lift']):,}")
print(f"  Smash: {len(metadata[metadata['stroke_type_english'] == 'Smash']):,}")
print(f"  Unique matches: {metadata['match_id'].nunique()}")
print()

# ============================================================================
# STEP 2-7: Same pipeline as Drop vs Smash
# ============================================================================

print("PIPELINE CONFIGURATION:")
print("-" * 80)
print()
print("Poses output: data/processed/poses_lift_smash/")
print("Features output: data/processed/features_lift_smash/")
print("Splits output: data/processed/splits_lift_smash/")
print("Models output: models/lift_smash/")
print("Reports output: outputs/reports_lift_smash/")
print()

# Create output directories
Path("data/processed/poses_lift_smash").mkdir(parents=True, exist_ok=True)
Path("data/processed/features_lift_smash").mkdir(parents=True, exist_ok=True)
Path("data/processed/splits_lift_smash").mkdir(parents=True, exist_ok=True)
Path("models/lift_smash/baseline").mkdir(parents=True, exist_ok=True)
Path("models/lift_smash/deep_learning").mkdir(parents=True, exist_ok=True)
Path("outputs/reports_lift_smash").mkdir(parents=True, exist_ok=True)

print("✓ Output directories created")
print()

# ============================================================================
# Summary
# ============================================================================

print("=" * 80)
print("PIPELINE SUMMARY")
print("=" * 80)
print()
print("Lift vs Smash Classification Pipeline Status:")
print()
print("✓ Step 1: Metadata loaded")
print("⏳ Step 2: Pose extraction (requires video files)")
print("⏳ Step 3: Feature engineering (requires poses)")
print("⏳ Step 4: Data splitting (requires features)")
print("⏳ Step 5: Baseline training (requires splits)")
print("⏳ Step 6: Deep learning training (requires splits)")
print("⏳ Step 7: Report generation (requires models)")
print()
print("Expected Results:")
print("  Accuracy: 75-90%")
print("  F1 Score: 77-92%")
print("  ROC AUC: 80-95%")
print()
print("Rationale:")
print("  Lift is underhand/low contact (defensive)")
print("  Smash is overhead/high contact (aggressive)")
print("  Expected Cohen's d > 0.8 (large effect)")
print()
print("Manual processing commands:")
print()
print("1. Extract poses:")
print("   python src/data_processing/extract_poses.py \\")
print("     --metadata data/processed/clips/lift_smash_metadata.csv \\")
print("     --output data/processed/poses_lift_smash")
print()
print("2. Engineer features:")
print("   python src/data_processing/feature_engineering_v2.py \\")
print("     --input data/processed/poses_lift_smash \\")
print("     --output data/processed/features_lift_smash")
print()
print("3. Create splits:")
print("   python src/data_processing/data_split.py \\")
print("     --input data/processed/features_lift_smash \\")
print("     --output data/processed/splits_lift_smash \\")
print("     --metadata data/processed/clips/lift_smash_metadata.csv")
print()
print("4. Train baseline models:")
print("   python src/models/baseline_model.py \\")
print("     --splits data/processed/splits_lift_smash \\")
print("     --output models/lift_smash/baseline")
print()
print("5. Train deep learning models:")
print("   python src/models/lstm_model.py \\")
print("     --splits data/processed/splits_lift_smash \\")
print("     --output models/lift_smash/deep_learning")
print()
