#!/usr/bin/env python3
"""
Extract Multi-Class Dataset for Game Analysis
Clear, Smash, Drop Shot, and Lift

This creates a single 4-class dataset for automatic stroke detection in rallies.
Lower accuracy expected (50-70%) but useful for game statistics and analysis.
"""

import pandas as pd
from pathlib import Path
import numpy as np

# Chinese to English mappings
STROKE_TRANSLATIONS = {
    '殺球': 'Smash',
    '長球': 'Clear',
    '挑球': 'Lift',
    '放小球': 'Drop',
    '點扣': 'Drop_Smash',
    '擋小球': 'Block',
    '推球': 'Push',
    '平球': 'Drive',
    '勾球': 'Cross_Net',
    '撲球': 'Rush',
    '發短球': 'Short_Serve',
    '防守回挑': 'Defensive_Lift',
    '防守回抽': 'Defensive_Drive',
    '過度切球': 'Cut',
    '後場抽平球': 'Rear_Drive',
    '切球': 'Slice',
    '未知球種': 'Unknown',
    '發長球': 'Long_Serve',
    '小平球': 'Small_Drive',
}

print("=" * 80)
print("MULTI-CLASS DATASET EXTRACTION (4 STROKES)")
print("Clear, Smash, Drop Shot, Lift")
print("=" * 80)
print()

# ============================================================================
# STEP 1: Extract all strokes from annotations
# ============================================================================

print("STEP 1: Extracting strokes from ShuttleSet annotations...")
print("-" * 80)

annotation_dir = Path("data/annotations")
set_files = list(annotation_dir.glob("*/set*.csv"))

print(f"Found {len(set_files)} annotation files")
print()

all_strokes = []
match_counter = {}

for file in set_files:
    try:
        match_name = file.parent.name
        set_num = file.stem

        df = pd.read_csv(file)

        if 'type' not in df.columns:
            continue

        # Assign match ID
        if match_name not in match_counter:
            match_counter[match_name] = len(match_counter) + 1
        match_id = match_counter[match_name]

        # Add metadata
        df['match_id'] = match_id
        df['match_name'] = match_name
        df['set_num'] = int(set_num.replace('set', ''))

        # English translation
        df['stroke_type_english'] = df['type'].map(STROKE_TRANSLATIONS)
        df['stroke_type_english'] = df['stroke_type_english'].fillna('Unknown')

        # Create clip name
        df['clip_name'] = df.apply(
            lambda row: f"{str(match_id).zfill(2)}_set{row['set_num']}_rally{int(row['rally'])}_ball{int(row['ball_round'])}_{row['stroke_type_english']}.mp4",
            axis=1
        )

        all_strokes.append(df)

    except Exception as e:
        print(f"Error processing {file}: {e}")
        continue

all_data = pd.concat(all_strokes, ignore_index=True)

print(f"✓ Total strokes extracted: {len(all_data):,}")
print()

# ============================================================================
# STEP 2: Filter for 4 target classes
# ============================================================================

print("STEP 2: Filtering for Clear, Smash, Drop, and Lift...")
print("-" * 80)

target_classes = ['Clear', 'Smash', 'Drop', 'Lift']
multiclass_data = all_data[all_data['stroke_type_english'].isin(target_classes)].copy()

print(f"Total 4-class strokes: {len(multiclass_data):,}")
print()

# Show distribution
for stroke in target_classes:
    count = len(multiclass_data[multiclass_data['stroke_type_english'] == stroke])
    pct = (count / len(multiclass_data)) * 100
    print(f"  {stroke}: {count:,} ({pct:.1f}%)")

print()

# ============================================================================
# STEP 3: Filter forehand-only
# ============================================================================

print("STEP 3: Filtering forehand-only strokes...")
print("-" * 80)

forehand_only = multiclass_data[multiclass_data['backhand'] != 1.0].copy()

backhand_removed = len(multiclass_data) - len(forehand_only)
print(f"Backhand shots removed: {backhand_removed:,}")
print()

# Show backhand breakdown
for stroke in target_classes:
    backhand_count = len(multiclass_data[
        (multiclass_data['stroke_type_english'] == stroke) &
        (multiclass_data['backhand'] == 1.0)
    ])
    print(f"  {stroke} backhand: {backhand_count:,}")

print()

# ============================================================================
# STEP 4: Final dataset statistics
# ============================================================================

print("=" * 80)
print("FINAL MULTI-CLASS DATASET")
print("=" * 80)
print()

print(f"Total forehand-only strokes: {len(forehand_only):,}")
print()

print("Class distribution:")
print(f"{'Stroke':<12} {'Count':>8} {'Percentage':>12} {'Train(70%)':>12} {'Val(15%)':>10} {'Test(15%)':>10}")
print("-" * 80)

total = len(forehand_only)
for stroke in target_classes:
    count = len(forehand_only[forehand_only['stroke_type_english'] == stroke])
    pct = (count / total) * 100
    train_est = int(count * 0.70)
    val_est = int(count * 0.15)
    test_est = int(count * 0.15)

    print(f"{stroke:<12} {count:>8,} {pct:>11.1f}% {train_est:>12,} {val_est:>10,} {test_est:>10,}")

print("-" * 80)
print(f"{'TOTAL':<12} {total:>8,} {'100.0%':>12} {int(total*0.70):>12,} {int(total*0.15):>10,} {int(total*0.15):>10,}")
print()

# Calculate class imbalance
counts = [len(forehand_only[forehand_only['stroke_type_english'] == s]) for s in target_classes]
max_count = max(counts)
min_count = min(counts)
imbalance_ratio = max_count / min_count

print("Class balance analysis:")
print(f"  Most common: {target_classes[counts.index(max_count)]} ({max_count:,})")
print(f"  Least common: {target_classes[counts.index(min_count)]} ({min_count:,})")
print(f"  Imbalance ratio: {imbalance_ratio:.2f}x")

if imbalance_ratio > 2.0:
    print(f"  ⚠️  WARNING: High class imbalance (>{imbalance_ratio:.1f}x)")
    print("     Consider class weighting during training")
elif imbalance_ratio > 1.5:
    print(f"  ⚠️  Moderate class imbalance ({imbalance_ratio:.1f}x)")
    print("     May need class weighting")
else:
    print(f"  ✓ Good class balance (<1.5x)")

print()

# ============================================================================
# STEP 5: Save dataset
# ============================================================================

print("STEP 5: Saving multi-class dataset...")
print("-" * 80)

output_file = Path("data/processed/clips/multiclass_4stroke_metadata.csv")
output_file.parent.mkdir(parents=True, exist_ok=True)
forehand_only.to_csv(output_file, index=False)

print(f"✓ Saved to: {output_file}")
print()

# ============================================================================
# Expected performance and recommendations
# ============================================================================

print("=" * 80)
print("EXPECTED PERFORMANCE & RECOMMENDATIONS")
print("=" * 80)
print()

print("Multi-Class Classification (4 classes):")
print()
print("Expected Accuracy: 50-70%")
print("  - Much lower than binary pairs (70-85%)")
print("  - But useful for automatic game analysis")
print()

print("Why lower accuracy?")
print("  1. Four decision boundaries vs one (binary)")
print("  2. Confusion between similar strokes:")
print("     - Clear vs Drop (both overhead)")
print("     - Drop vs Smash (both overhead)")
print("  3. Class imbalance may bias predictions")
print()

print("Recommended training strategies:")
print("  1. Use class weights to handle imbalance:")
print("     - sklearn: class_weight='balanced'")
print("     - keras: class_weight={0: w0, 1: w1, 2: w2, 3: w3}")
print()
print("  2. Data augmentation for minority classes")
print()
print("  3. Ensemble methods:")
print("     - Random Forest with balanced trees")
print("     - Gradient Boosting with class weights")
print()
print("  4. Deep learning with focal loss:")
print("     - Focuses on hard-to-classify examples")
print("     - Reduces impact of easy examples")
print()

print("Best use cases:")
print("  ✓ Automatic rally analysis")
print("  ✓ Match statistics generation")
print("  ✓ Stroke distribution analysis")
print("  ✓ Game pattern recognition")
print()
print("  ✗ Real-time coaching (use binary pairs)")
print("  ✗ Technique feedback (ambiguous labels)")
print()

print("Next steps:")
print("  1. Run: python process_multiclass_pipeline.py")
print("  2. Train models with class weighting")
print("  3. Analyze confusion matrix to understand errors")
print("  4. Consider adding more features if accuracy < 60%")
print()

print("=" * 80)
print("Dataset ready for multi-class training!")
print("=" * 80)
