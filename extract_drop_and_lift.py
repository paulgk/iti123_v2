#!/usr/bin/env python3
"""
Extract Drop Shot and Lift/Lob strokes from ShuttleSet annotations
and create new datasets for Drop vs Smash and Lift vs Smash classification

This script:
1. Scans all annotation CSV files
2. Extracts Drop Shot (放小球) and Lift/Lob (挑球) strokes
3. Adds English translations
4. Creates metadata CSV files
5. Filters for forehand-only strokes
"""

import pandas as pd
from pathlib import Path
import shutil

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

def extract_strokes():
    """Extract all strokes from annotation files with English translations"""

    print("=" * 80)
    print("EXTRACTING DROP SHOT AND LIFT/LOB STROKES FROM SHUTTLESET")
    print("=" * 80)
    print()

    # Find all annotation files
    annotation_dir = Path("data/annotations")
    set_files = list(annotation_dir.glob("*/set*.csv"))

    print(f"Found {len(set_files)} annotation files")
    print()

    all_strokes = []
    match_counter = {}

    # Process each file
    for file in set_files:
        try:
            # Extract match info from path
            match_name = file.parent.name
            set_num = file.stem  # 'set1', 'set2', etc.

            # Load annotations
            df = pd.read_csv(file)

            if 'type' not in df.columns:
                continue

            # Assign match ID
            if match_name not in match_counter:
                match_counter[match_name] = len(match_counter) + 1
            match_id = match_counter[match_name]

            # Add metadata columns
            df['match_id'] = match_id
            df['match_name'] = match_name
            df['set_num'] = int(set_num.replace('set', ''))

            # Add English translation
            df['stroke_type_english'] = df['type'].map(STROKE_TRANSLATIONS)

            # Fill missing translations
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

    # Combine all data
    all_data = pd.concat(all_strokes, ignore_index=True)

    print(f"Total strokes extracted: {len(all_data):,}")
    print()

    return all_data

def create_drop_smash_dataset(all_data):
    """Create Drop vs Smash dataset"""

    print("=" * 80)
    print("CREATING DROP SHOT VS SMASH DATASET")
    print("=" * 80)
    print()

    # Filter for Drop and Smash only
    drop_smash = all_data[
        all_data['stroke_type_english'].isin(['Drop', 'Smash'])
    ].copy()

    print(f"Total Drop + Smash strokes: {len(drop_smash):,}")
    print(f"  Drop: {len(drop_smash[drop_smash['stroke_type_english'] == 'Drop']):,}")
    print(f"  Smash: {len(drop_smash[drop_smash['stroke_type_english'] == 'Smash']):,}")
    print()

    # Filter forehand-only (backhand != 1.0)
    forehand_only = drop_smash[drop_smash['backhand'] != 1.0].copy()

    backhand_removed = len(drop_smash) - len(forehand_only)
    print(f"Backhand shots removed: {backhand_removed}")
    print(f"  Drop backhand: {len(drop_smash[(drop_smash['stroke_type_english'] == 'Drop') & (drop_smash['backhand'] == 1.0)])}")
    print(f"  Smash backhand: {len(drop_smash[(drop_smash['stroke_type_english'] == 'Smash') & (drop_smash['backhand'] == 1.0)])}")
    print()

    print(f"Final forehand-only dataset: {len(forehand_only):,}")
    print(f"  Drop: {len(forehand_only[forehand_only['stroke_type_english'] == 'Drop']):,}")
    print(f"  Smash: {len(forehand_only[forehand_only['stroke_type_english'] == 'Smash']):,}")
    print()

    # Save
    output_file = Path("data/processed/clips/drop_smash_metadata.csv")
    output_file.parent.mkdir(parents=True, exist_ok=True)
    forehand_only.to_csv(output_file, index=False)

    print(f"✓ Saved to: {output_file}")
    print()

    return forehand_only

def create_lift_smash_dataset(all_data):
    """Create Lift vs Smash dataset"""

    print("=" * 80)
    print("CREATING LIFT/LOB VS SMASH DATASET")
    print("=" * 80)
    print()

    # Filter for Lift and Smash only
    lift_smash = all_data[
        all_data['stroke_type_english'].isin(['Lift', 'Smash'])
    ].copy()

    print(f"Total Lift + Smash strokes: {len(lift_smash):,}")
    print(f"  Lift: {len(lift_smash[lift_smash['stroke_type_english'] == 'Lift']):,}")
    print(f"  Smash: {len(lift_smash[lift_smash['stroke_type_english'] == 'Smash']):,}")
    print()

    # Filter forehand-only (backhand != 1.0)
    forehand_only = lift_smash[lift_smash['backhand'] != 1.0].copy()

    backhand_removed = len(lift_smash) - len(forehand_only)
    print(f"Backhand shots removed: {backhand_removed}")
    print(f"  Lift backhand: {len(lift_smash[(lift_smash['stroke_type_english'] == 'Lift') & (lift_smash['backhand'] == 1.0)])}")
    print(f"  Smash backhand: {len(lift_smash[(lift_smash['stroke_type_english'] == 'Smash') & (lift_smash['backhand'] == 1.0)])}")
    print()

    print(f"Final forehand-only dataset: {len(forehand_only):,}")
    print(f"  Lift: {len(forehand_only[forehand_only['stroke_type_english'] == 'Lift']):,}")
    print(f"  Smash: {len(forehand_only[forehand_only['stroke_type_english'] == 'Smash']):,}")
    print()

    # Save
    output_file = Path("data/processed/clips/lift_smash_metadata.csv")
    output_file.parent.mkdir(parents=True, exist_ok=True)
    forehand_only.to_csv(output_file, index=False)

    print(f"✓ Saved to: {output_file}")
    print()

    return forehand_only

def main():
    """Main execution"""

    # Extract all strokes with English translations
    all_data = extract_strokes()

    # Create Drop vs Smash dataset
    drop_smash_data = create_drop_smash_dataset(all_data)

    # Create Lift vs Smash dataset
    lift_smash_data = create_lift_smash_dataset(all_data)

    print("=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print()
    print("Created two new datasets:")
    print(f"1. Drop vs Smash: {len(drop_smash_data):,} forehand-only strokes")
    print(f"2. Lift vs Smash: {len(lift_smash_data):,} forehand-only strokes")
    print()
    print("Next steps:")
    print("  1. Run: python process_drop_smash_pipeline.py")
    print("  2. Run: python process_lift_smash_pipeline.py")
    print()

if __name__ == "__main__":
    main()
