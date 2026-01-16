#!/usr/bin/env python3
"""
Debug script to verify data splits in Colab

Run this in Colab to check what data is actually being loaded.
"""

import pickle
from pathlib import Path

print("=" * 70)
print("DEBUGGING DATA SPLITS")
print("=" * 70)
print()

# Check path resolution
BASE_DIR = Path(__file__).resolve().parents[0]
print(f"Script location: {__file__}")
print(f"BASE_DIR: {BASE_DIR}")
print()

# Try different possible split locations
possible_paths = [
    Path("/content/drive/MyDrive/ITI123/data/processed/splits"),
    BASE_DIR / "data" / "processed" / "splits",
    Path("./data/processed/splits"),
]

print("Checking possible split directory locations:")
for path in possible_paths:
    print(f"  {path}: {'EXISTS' if path.exists() else 'NOT FOUND'}")
print()

# Use the path that exists
SPLITS_DIR = None
for path in possible_paths:
    if path.exists():
        SPLITS_DIR = path
        break

if SPLITS_DIR is None:
    print("❌ ERROR: Could not find splits directory!")
    print("\nPlease verify the files are uploaded to:")
    print("  /content/drive/MyDrive/ITI123/data/processed/splits/")
    exit(1)

print(f"Using: {SPLITS_DIR}")
print()

# List files in directory
print("Files in splits directory:")
import os
for f in os.listdir(SPLITS_DIR):
    fpath = SPLITS_DIR / f
    size_mb = os.path.getsize(fpath) / (1024 * 1024)
    print(f"  {f}: {size_mb:.2f} MB")
print()

# Load and inspect each split file
for split_name in ['train', 'val', 'test']:
    print("=" * 70)
    print(f"{split_name.upper()} DATA")
    print("=" * 70)

    split_file = SPLITS_DIR / f"{split_name}_data.pkl"

    if not split_file.exists():
        print(f"❌ {split_file} not found!")
        continue

    with open(split_file, 'rb') as f:
        data = pickle.load(f)

    print(f"\nKeys in {split_name}_data.pkl:")
    for key, value in data.items():
        if hasattr(value, 'shape'):
            print(f"  {key}: {value.shape}")
        elif hasattr(value, '__len__'):
            print(f"  {key}: len={len(value)}")
        else:
            print(f"  {key}: {type(value).__name__}")

    print()

    # Check critical fields
    X_shape = data['X'].shape
    y_shape = data['y'].shape

    print(f"Sample counts:")
    print(f"  X (sequences): {X_shape[0]} samples")
    print(f"  y (labels): {y_shape[0]} samples")

    if 'X_stat_raw' in data:
        print(f"  X_stat_raw: {data['X_stat_raw'].shape[0]} samples")
    else:
        print(f"  X_stat_raw: NOT FOUND")

    if 'X_stat' in data:
        print(f"  X_stat: {data['X_stat'].shape[0]} samples")
    else:
        print(f"  X_stat: NOT FOUND")

    print()

print("=" * 70)
print("SUMMARY")
print("=" * 70)
print()

# Load train to check expected size
with open(SPLITS_DIR / "train_data.pkl", 'rb') as f:
    train = pickle.load(f)

train_samples = len(train['y'])

print(f"Training samples: {train_samples}")
print()

if train_samples < 100:
    print("❌ PROBLEM: Very few training samples!")
    print(f"   Expected: ~3,554 samples")
    print(f"   Found: {train_samples} samples")
    print()
    print("ACTION REQUIRED:")
    print("  1. The files in Colab are NOT the filtered forehand-only splits")
    print("  2. Download the correct files from local machine:")
    print("     - data/processed/splits/train_data.pkl (105 MB)")
    print("     - data/processed/splits/val_data.pkl (13 MB)")
    print("     - data/processed/splits/test_data.pkl (20 MB)")
    print("  3. Re-upload to Colab at:")
    print("     /content/drive/MyDrive/ITI123/data/processed/splits/")
elif train_samples > 3000:
    print("✅ CORRECT: Files appear to be the filtered forehand-only splits")
    print(f"   Expected: ~3,554 samples")
    print(f"   Found: {train_samples} samples")
    print()
    print("If baseline_model.py still shows 13 samples, there may be:")
    print("  1. Python import caching issues")
    print("  2. Multiple copies of split files in different locations")
    print("  3. The script is using a different BASE_DIR")
else:
    print("⚠️  UNCLEAR: Sample count is in between")
    print(f"   Expected: ~3,554 samples (forehand only)")
    print(f"   Found: {train_samples} samples")
    print()
    print("This may indicate partial data or corrupted upload")
