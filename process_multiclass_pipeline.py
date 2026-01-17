#!/usr/bin/env python3
"""
Complete Multi-Class Pipeline for Game Analysis
4 strokes: Clear, Smash, Drop Shot, Lift

Expected accuracy: 50-70%
Use case: Automatic rally analysis, match statistics
"""

import sys
import pickle
from pathlib import Path
import pandas as pd

print("=" * 80)
print("MULTI-CLASS PIPELINE (4 STROKES)")
print("Clear, Smash, Drop, Lift - for Game Analysis")
print("=" * 80)
print()

# ============================================================================
# STEP 1: Load metadata
# ============================================================================

print("STEP 1: Loading multi-class metadata...")
print("-" * 80)

metadata_file = Path("data/processed/clips/multiclass_4stroke_metadata.csv")

if not metadata_file.exists():
    print(f"❌ ERROR: {metadata_file} not found!")
    print("   Please run: python extract_multiclass_dataset.py first")
    sys.exit(1)

metadata = pd.read_csv(metadata_file)

print(f"✓ Loaded {len(metadata):,} strokes")
print()

# Show distribution
for stroke in ['Clear', 'Smash', 'Drop', 'Lift']:
    count = len(metadata[metadata['stroke_type_english'] == stroke])
    pct = (count / len(metadata)) * 100
    print(f"  {stroke}: {count:,} ({pct:.1f}%)")

print(f"\n  Unique matches: {metadata['match_id'].nunique()}")
print()

# ============================================================================
# Configuration
# ============================================================================

print("PIPELINE CONFIGURATION:")
print("-" * 80)
print()
print("Output directories:")
print("  - Poses: data/processed/poses_multiclass/")
print("  - Features: data/processed/features_multiclass/")
print("  - Splits: data/processed/splits_multiclass/")
print("  - Models: models/multiclass/")
print("  - Reports: outputs/reports_multiclass/")
print()

# Create directories
Path("data/processed/poses_multiclass").mkdir(parents=True, exist_ok=True)
Path("data/processed/features_multiclass").mkdir(parents=True, exist_ok=True)
Path("data/processed/splits_multiclass").mkdir(parents=True, exist_ok=True)
Path("models/multiclass/baseline").mkdir(parents=True, exist_ok=True)
Path("models/multiclass/deep_learning").mkdir(parents=True, exist_ok=True)
Path("outputs/reports_multiclass").mkdir(parents=True, exist_ok=True)

print("✓ Directories created")
print()

# ============================================================================
# Processing steps
# ============================================================================

print("=" * 80)
print("PROCESSING STEPS")
print("=" * 80)
print()

print("1. Extract poses (30-60 minutes):")
print("   python src/data_processing/extract_poses.py \\")
print("     --metadata data/processed/clips/multiclass_4stroke_metadata.csv \\")
print("     --videos data/videos/ \\")
print("     --output data/processed/poses_multiclass/")
print()

print("2. Engineer features (15-20 minutes):")
print("   python src/data_processing/feature_engineering_v2.py \\")
print("     --input data/processed/poses_multiclass/ \\")
print("     --output data/processed/features_multiclass/")
print()

print("3. Create splits with stratification (1 minute):")
print("   python src/data_processing/data_split.py \\")
print("     --input data/processed/features_multiclass/ \\")
print("     --metadata data/processed/clips/multiclass_4stroke_metadata.csv \\")
print("     --output data/processed/splits_multiclass/")
print()

print("4. Train baseline with class weighting (10 minutes):")
print("   python src/models/baseline_model_multiclass.py \\")
print("     --splits data/processed/splits_multiclass/ \\")
print("     --output models/multiclass/baseline/ \\")
print("     --classes 4 \\")
print("     --class-weight balanced")
print()

print("5. Train deep learning with focal loss (30-40 minutes):")
print("   python src/models/lstm_model_multiclass.py \\")
print("     --splits data/processed/splits_multiclass/ \\")
print("     --output models/multiclass/deep_learning/ \\")
print("     --classes 4 \\")
print("     --loss focal")
print()

print("6. Evaluate and generate confusion matrix:")
print("   python src/evaluation/evaluate_multiclass.py \\")
print("     --models models/multiclass/ \\")
print("     --output outputs/reports_multiclass/")
print()

# ============================================================================
# Expected results
# ============================================================================

print("=" * 80)
print("EXPECTED RESULTS")
print("=" * 80)
print()

print("Baseline Models (Random Forest, SVM):")
print("  Accuracy: 55-65%")
print("  F1 Score (macro): 52-62%")
print()

print("Deep Learning (LSTM, BiLSTM, GRU):")
print("  Accuracy: 60-70%")
print("  F1 Score (macro): 58-68%")
print()

print("Likely confusion patterns:")
print("  - Clear ↔ Drop (both overhead)")
print("  - Drop ↔ Smash (both overhead, different speed)")
print("  - Lift correctly classified (underhand, distinct)")
print()

print("Success criteria:")
print("  ✓ Accuracy > 60% (better than random 25%)")
print("  ✓ F1 > 58% across all classes")
print("  ✓ Confusion matrix shows clear patterns")
print()

# ============================================================================
# Usage recommendations
# ============================================================================

print("=" * 80)
print("USAGE RECOMMENDATIONS")
print("=" * 80)
print()

print("✓ GOOD use cases:")
print("  - Automatic rally breakdown")
print("  - Match statistics (stroke distribution)")
print("  - Game pattern analysis")
print("  - Tournament data collection")
print()

print("✗ AVOID for:")
print("  - Real-time coaching feedback")
print("  - Technique analysis (use binary pairs)")
print("  - When high accuracy required (>80%)")
print()

print("Integration with binary pairs:")
print("  1. Use multi-class for initial stroke detection")
print("  2. If confidence low, ask user to specify stroke type")
print("  3. Then use binary pair classifier for detailed analysis")
print()

print("Example workflow:")
print("  Multi-class: \"70% Drop, 25% Smash, 5% Clear\" → Drop detected")
print("  Binary pair: Use Drop vs Smash classifier → 85% Drop (high confidence)")
print("  Coaching: Compare to professional Drop benchmarks")
print()

print("=" * 80)
print("Ready to process! Run the commands above step by step.")
print("=" * 80)
