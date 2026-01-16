#!/usr/bin/env python3
"""
Analyze all available stroke types in ShuttleSet annotations
"""

import pandas as pd
from pathlib import Path
from collections import Counter

# Find all set CSV files
annotation_dir = Path("data/annotations")
set_files = list(annotation_dir.glob("*/set*.csv"))

print(f"Found {len(set_files)} annotation files")
print()

# Collect all stroke types
all_strokes = []
for file in set_files:
    try:
        df = pd.read_csv(file)
        if 'type' in df.columns:
            all_strokes.extend(df['type'].dropna().tolist())
    except Exception as e:
        print(f"Error reading {file}: {e}")

# Count stroke types
stroke_counts = Counter(all_strokes)

print("=" * 70)
print("ALL STROKE TYPES IN SHUTTLESET DATASET")
print("=" * 70)
print()

# Common Chinese-English mappings for badminton strokes
translations = {
    '殺球': 'Smash',
    '長球': 'Clear',
    '挑球': 'Lift/Lob',
    '放小球': 'Drop Shot',
    '點扣': 'Drop Smash',
    '擋小球': 'Block',
    '推球': 'Push',
    '平球': 'Drive',
    '勾球': 'Cross Net Shot',
    '撲球': 'Rush/Kill',
    '發短球': 'Short Serve',
    '防守回挑': 'Defensive Lift',
    '防守回抽': 'Defensive Drive',
    '過度切球': 'Cut/Slice',
    '後場抽平球': 'Rear Court Drive',
}

print(f"{'Chinese':<20} {'English':<25} {'Count':>10}")
print("-" * 70)

for chinese, count in stroke_counts.most_common():
    english = translations.get(chinese, 'Unknown')
    print(f"{chinese:<20} {english:<25} {count:>10,}")

print()
print("=" * 70)
print("SUMMARY")
print("=" * 70)
print(f"Total strokes: {sum(stroke_counts.values()):,}")
print(f"Unique types: {len(stroke_counts)}")
print()

# Identify major stroke types (>1000 samples)
major_strokes = {k: v for k, v in stroke_counts.items() if v > 1000}
print("Major stroke types (>1,000 samples):")
for chinese, count in sorted(major_strokes.items(), key=lambda x: x[1], reverse=True):
    english = translations.get(chinese, 'Unknown')
    print(f"  {chinese} ({english}): {count:,}")
