#!/usr/bin/env python3
"""
Baseline Model with Debug Output - Find where it's loading from
"""

import pickle
import os
from pathlib import Path

print("=" * 70)
print("BASELINE MODEL DEBUG - PATH RESOLUTION")
print("=" * 70)
print()

# Show where the script thinks it is
print(f"Current working directory: {os.getcwd()}")
print(f"Script file would be: {__file__ if '__file__' in dir() else 'N/A'}")
print()

# Try different BASE_DIR calculations
script_location = Path("/content/drive/MyDrive/ITI123/src/models/baseline_model.py")
print(f"Assuming script at: {script_location}")
print(f"  .parents[0]: {script_location.parents[0]}")
print(f"  .parents[1]: {script_location.parents[1]}")
print(f"  .parents[2]: {script_location.parents[2]}")
print()

# What the script SHOULD use
correct_base = Path("/content/drive/MyDrive/ITI123")
correct_splits = correct_base / "data" / "processed" / "splits"

print(f"CORRECT paths:")
print(f"  BASE_DIR: {correct_base}")
print(f"  SPLITS_DIR: {correct_splits}")
print(f"  Exists: {correct_splits.exists()}")
print()

# Check what baseline_model.py is actually using
# by importing it and checking its globals
import sys
sys.path.insert(0, str(correct_base))

# Before importing, let's check what would happen
BASE_DIR_test = script_location.resolve().parents[2]
SPLITS_DIR_test = BASE_DIR_test / "data" / "processed" / "splits"

print(f"What baseline_model.py calculates:")
print(f"  BASE_DIR: {BASE_DIR_test}")
print(f"  SPLITS_DIR: {SPLITS_DIR_test}")
print(f"  Exists: {SPLITS_DIR_test.exists()}")
print()

# Check if there are multiple split directories
print("Searching for all train_data.pkl files...")
os.system("find /content -name 'train_data.pkl' 2>/dev/null | head -10")
print()

# Load from each location and check size
potential_locations = [
    correct_splits,
    SPLITS_DIR_test,
    Path("/content/data/processed/splits"),
    Path("./data/processed/splits"),
]

print("=" * 70)
print("CHECKING ALL POTENTIAL SPLIT LOCATIONS")
print("=" * 70)

for loc in potential_locations:
    print(f"\n{loc}:")
    if loc.exists():
        train_file = loc / "train_data.pkl"
        if train_file.exists():
            with open(train_file, 'rb') as f:
                data = pickle.load(f)
            print(f"  ✓ EXISTS - {len(data['y'])} samples")
            print(f"  X_stat_raw: {data.get('X_stat_raw', 'N/A')}")
        else:
            print(f"  ✓ Dir exists but no train_data.pkl")
    else:
        print(f"  ✗ Does not exist")

print()
print("=" * 70)
print("SOLUTION")
print("=" * 70)
print()
print("The baseline_model.py needs to explicitly use:")
print(f"  SPLITS_DIR = Path('/content/drive/MyDrive/ITI123/data/processed/splits')")
print()
