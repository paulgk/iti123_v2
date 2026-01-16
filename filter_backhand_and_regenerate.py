#!/usr/bin/env python3
"""
Filter Backhand Shots and Regenerate Benchmarks

This script:
1. Filters out all backhand shots from the dataset
2. Updates the metadata CSV files
3. Recalculates professional benchmarks from forehand-only strokes
4. Updates technique_benchmarks.py with new values

Usage:
    python filter_backhand_and_regenerate.py
"""

import pandas as pd
import numpy as np
import pickle
from pathlib import Path
import shutil
from datetime import datetime

def filter_backhand_shots():
    """Remove backhand shots from clips_metadata.csv and save filtered version"""

    metadata_path = Path('data/processed/clips/clips_metadata.csv')
    backup_path = Path('data/processed/clips/clips_metadata_backup.csv')

    # Backup original
    print("=" * 70)
    print("STEP 1: FILTERING BACKHAND SHOTS")
    print("=" * 70)
    print()

    if not backup_path.exists():
        print(f"Creating backup: {backup_path}")
        shutil.copy(metadata_path, backup_path)
    else:
        print(f"Backup already exists: {backup_path}")

    # Load metadata
    print(f"Loading metadata from: {metadata_path}")
    df = pd.read_csv(metadata_path)

    total_before = len(df)
    print(f"\nBefore filtering:")
    print(f"  Total strokes: {total_before}")

    # Count by stroke type
    for stroke in ['Clear', 'Smash']:
        stroke_df = df[df['stroke_type_english'] == stroke]
        backhand_count = (stroke_df['backhand'] == 1.0).sum()
        total_count = len(stroke_df)
        print(f"  {stroke}: {total_count} total, {backhand_count} backhand ({backhand_count/total_count*100:.1f}%)")

    # Filter out backhand shots
    print(f"\nFiltering out backhand shots (where backhand == 1.0)...")
    df_filtered = df[df['backhand'] != 1.0].copy()

    total_after = len(df_filtered)
    removed = total_before - total_after

    print(f"\nAfter filtering:")
    print(f"  Total strokes: {total_after}")
    print(f"  Removed: {removed} backhand strokes ({removed/total_before*100:.1f}%)")

    # Count by stroke type after filtering
    for stroke in ['Clear', 'Smash']:
        stroke_df = df_filtered[df_filtered['stroke_type_english'] == stroke]
        total_count = len(stroke_df)
        print(f"  {stroke}: {total_count} forehand strokes")

    # Save filtered metadata
    print(f"\nSaving filtered metadata to: {metadata_path}")
    df_filtered.to_csv(metadata_path, index=False)
    print("✓ Saved")

    return df_filtered


def update_split_files(df_filtered):
    """Update train/val/test split files to remove backhand shots"""

    print("\n" + "=" * 70)
    print("STEP 2: UPDATING SPLIT FILES")
    print("=" * 70)
    print()

    splits_dir = Path('data/processed/splits')

    # Get list of clip names to keep (forehand only)
    valid_clips = set(df_filtered['clip_name'].values)

    for split_name in ['train', 'val', 'test']:
        split_path = splits_dir / f'{split_name}_metadata.csv'

        if not split_path.exists():
            print(f"⚠️  {split_name}_metadata.csv not found, skipping")
            continue

        # Backup
        backup_path = splits_dir / f'{split_name}_metadata_backup.csv'
        if not backup_path.exists():
            shutil.copy(split_path, backup_path)

        # Load split
        print(f"Processing {split_name} split...")
        df_split = pd.read_csv(split_path)
        before_count = len(df_split)

        # Filter to keep only forehand shots
        df_split_filtered = df_split[df_split['clip_name'].isin(valid_clips)].copy()
        after_count = len(df_split_filtered)
        removed = before_count - after_count

        print(f"  Before: {before_count} clips")
        print(f"  After: {after_count} clips")
        print(f"  Removed: {removed} backhand clips")

        # Save
        df_split_filtered.to_csv(split_path, index=False)
        print(f"  ✓ Saved to {split_path}")

    print("\n✓ All split files updated")


def recalculate_benchmarks():
    """Recalculate professional benchmarks from forehand-only strokes"""

    print("\n" + "=" * 70)
    print("STEP 3: RECALCULATING BENCHMARKS")
    print("=" * 70)
    print()

    features_path = Path('data/processed/features/statistical_features.pkl')

    if not features_path.exists():
        print(f"❌ Error: {features_path} not found")
        print("You need to regenerate features first by running:")
        print("  python src/data_processing/batch_extract_features.py")
        return None

    print(f"Loading features from: {features_path}")
    with open(features_path, 'rb') as f:
        data = pickle.load(f)

    features_df = pd.DataFrame(data['features'])
    labels = data['labels']
    clip_names = data['clip_names']

    print(f"Loaded {len(features_df)} samples")

    # Get forehand-only metadata
    metadata_df = pd.read_csv('data/processed/clips/clips_metadata.csv')
    forehand_clips = set(metadata_df['clip_name'].values)

    # Filter features to forehand only
    forehand_mask = [clip in forehand_clips for clip in clip_names]
    features_df_filtered = features_df[forehand_mask]
    labels_filtered = [labels[i] for i, is_forehand in enumerate(forehand_mask) if is_forehand]

    print(f"After filtering: {len(features_df_filtered)} forehand samples")

    # Calculate new benchmarks
    benchmarks = {}

    for stroke_type in ['Clear', 'Smash']:
        print(f"\nCalculating benchmarks for {stroke_type}...")

        # Get samples for this stroke type
        stroke_mask = [label == stroke_type for label in labels_filtered]
        stroke_features = features_df_filtered[stroke_mask]

        print(f"  Samples: {len(stroke_features)}")

        if len(stroke_features) == 0:
            print(f"  ⚠️  No samples found for {stroke_type}")
            continue

        # Calculate percentiles for each feature
        stroke_benchmarks = {}

        for col in stroke_features.columns:
            values = stroke_features[col].dropna()

            if len(values) > 0:
                p25 = np.percentile(values, 25)
                p50 = np.percentile(values, 50)
                p75 = np.percentile(values, 75)

                stroke_benchmarks[col] = {
                    'p25': float(p25),
                    'p50': float(p50),
                    'p75': float(p75)
                }

        benchmarks[stroke_type] = stroke_benchmarks
        print(f"  ✓ Calculated {len(stroke_benchmarks)} feature benchmarks")

    return benchmarks


def update_technique_benchmarks(benchmarks):
    """Update technique_benchmarks.py with new forehand-only benchmarks"""

    print("\n" + "=" * 70)
    print("STEP 4: UPDATING TECHNIQUE_BENCHMARKS.PY")
    print("=" * 70)
    print()

    if benchmarks is None:
        print("⚠️  Skipping - no benchmarks provided")
        return

    benchmarks_file = Path('src/coaching/technique_benchmarks.py')

    # Backup original
    backup_file = Path('src/coaching/technique_benchmarks_backup.py')
    if not backup_file.exists():
        print(f"Creating backup: {backup_file}")
        shutil.copy(benchmarks_file, backup_file)

    # Define key metrics we use for coaching
    key_metrics = {
        'max_velocity': 'Racket head velocity',
        'r_elbow_angle_mean': 'Elbow angle at contact',
        'r_shoulder_angle_mean': 'Shoulder angle',
        'r_forearm_vertical_angle_mean': 'Forearm vertical angle',
        'r_wrist_height_from_head_mean': 'Contact point height',
        'trunk_lean_forward_mean': 'Forward trunk lean',
        'hip_shoulder_separation_mean': 'Hip-shoulder separation'
    }

    print("Extracting key metrics from benchmarks...")

    new_ranges = {
        'Clear': {},
        'Smash': {}
    }

    for stroke_type in ['Clear', 'Smash']:
        if stroke_type not in benchmarks:
            continue

        stroke_benchmarks = benchmarks[stroke_type]

        for metric_key, metric_name in key_metrics.items():
            if metric_key in stroke_benchmarks:
                bench = stroke_benchmarks[metric_key]
                new_ranges[stroke_type][metric_key] = {
                    'min': bench['p25'],
                    'max': bench['p75'],
                    'target': bench['p50']
                }

        print(f"  {stroke_type}: {len(new_ranges[stroke_type])} key metrics")

    # Generate new file content
    content = f'''"""
Professional Technique Benchmarks - FOREHAND ONLY

Updated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
Source: ShuttleSet dataset (forehand strokes only, backhand filtered out)
Sample sizes:
  - Clear: {len([l for l in benchmarks.get('Clear', {}).keys()])} features
  - Smash: {len([l for l in benchmarks.get('Smash', {}).keys()])} features

These ranges represent the 25th-75th percentile of professional players.
"""

# Professional benchmarks for Clear stroke (forehand only)
CLEAR_BENCHMARKS = {{
    'max_velocity': {new_ranges['Clear'].get('max_velocity', {}).get('min', 70):.2f},  # Lower bound
    'max_velocity_target': {new_ranges['Clear'].get('max_velocity', {}).get('target', 80):.2f},  # Median
    'max_velocity_upper': {new_ranges['Clear'].get('max_velocity', {}).get('max', 90):.2f},  # Upper bound

    'elbow_angle_min': {new_ranges['Clear'].get('r_elbow_angle_mean', {}).get('min', 155):.2f},
    'elbow_angle_target': {new_ranges['Clear'].get('r_elbow_angle_mean', {}).get('target', 165):.2f},
    'elbow_angle_max': {new_ranges['Clear'].get('r_elbow_angle_mean', {}).get('max', 175):.2f},

    'forearm_angle_min': {new_ranges['Clear'].get('r_forearm_vertical_angle_mean', {}).get('min', 32):.2f},
    'forearm_angle_target': {new_ranges['Clear'].get('r_forearm_vertical_angle_mean', {}).get('target', 50):.2f},
    'forearm_angle_max': {new_ranges['Clear'].get('r_forearm_vertical_angle_mean', {}).get('max', 69):.2f},

    'contact_point_min': {new_ranges['Clear'].get('r_wrist_height_from_head_mean', {}).get('min', -0.15):.3f},
    'contact_point_target': {new_ranges['Clear'].get('r_wrist_height_from_head_mean', {}).get('target', -0.05):.3f},
    'contact_point_max': {new_ranges['Clear'].get('r_wrist_height_from_head_mean', {}).get('max', 0.05):.3f},

    'shoulder_angle_min': {new_ranges['Clear'].get('r_shoulder_angle_mean', {}).get('min', 60):.2f},
    'shoulder_angle_target': {new_ranges['Clear'].get('r_shoulder_angle_mean', {}).get('target', 75):.2f},
    'shoulder_angle_max': {new_ranges['Clear'].get('r_shoulder_angle_mean', {}).get('max', 90):.2f},

    'trunk_lean_min': {new_ranges['Clear'].get('trunk_lean_forward_mean', {}).get('min', -5):.2f},
    'trunk_lean_target': {new_ranges['Clear'].get('trunk_lean_forward_mean', {}).get('target', 5):.2f},
    'trunk_lean_max': {new_ranges['Clear'].get('trunk_lean_forward_mean', {}).get('max', 15):.2f},
}}

# Professional benchmarks for Smash stroke (forehand only)
SMASH_BENCHMARKS = {{
    'max_velocity': {new_ranges['Smash'].get('max_velocity', {}).get('min', 80):.2f},  # Lower bound
    'max_velocity_target': {new_ranges['Smash'].get('max_velocity', {}).get('target', 95):.2f},  # Median
    'max_velocity_upper': {new_ranges['Smash'].get('max_velocity', {}).get('max', 110):.2f},  # Upper bound

    'elbow_angle_min': {new_ranges['Smash'].get('r_elbow_angle_mean', {}).get('min', 160):.2f},
    'elbow_angle_target': {new_ranges['Smash'].get('r_elbow_angle_mean', {}).get('target', 170):.2f},
    'elbow_angle_max': {new_ranges['Smash'].get('r_elbow_angle_mean', {}).get('max', 178):.2f},

    'forearm_angle_min': {new_ranges['Smash'].get('r_forearm_vertical_angle_mean', {}).get('min', 15):.2f},
    'forearm_angle_target': {new_ranges['Smash'].get('r_forearm_vertical_angle_mean', {}).get('target', 30):.2f},
    'forearm_angle_max': {new_ranges['Smash'].get('r_forearm_vertical_angle_mean', {}).get('max', 47):.2f},

    'contact_point_min': {new_ranges['Smash'].get('r_wrist_height_from_head_mean', {}).get('min', -0.05):.3f},
    'contact_point_target': {new_ranges['Smash'].get('r_wrist_height_from_head_mean', {}).get('target', 0.05):.3f},
    'contact_point_max': {new_ranges['Smash'].get('r_wrist_height_from_head_mean', {}).get('max', 0.15):.3f},

    'shoulder_angle_min': {new_ranges['Smash'].get('r_shoulder_angle_mean', {}).get('min', 70):.2f},
    'shoulder_angle_target': {new_ranges['Smash'].get('r_shoulder_angle_mean', {}).get('target', 85):.2f},
    'shoulder_angle_max': {new_ranges['Smash'].get('r_shoulder_angle_mean', {}).get('max', 100):.2f},

    'trunk_lean_min': {new_ranges['Smash'].get('trunk_lean_forward_mean', {}).get('min', 5):.2f},
    'trunk_lean_target': {new_ranges['Smash'].get('trunk_lean_forward_mean', {}).get('target', 15):.2f},
    'trunk_lean_max': {new_ranges['Smash'].get('trunk_lean_forward_mean', {}).get('max', 25):.2f},
}}
'''

    print(f"Writing new benchmarks to: {benchmarks_file}")
    with open(benchmarks_file, 'w') as f:
        f.write(content)

    print("✓ Updated technique_benchmarks.py")

    # Display summary
    print("\n" + "=" * 70)
    print("NEW BENCHMARK SUMMARY")
    print("=" * 70)
    print()
    print("Clear (Forehand):")
    print(f"  Velocity: {new_ranges['Clear'].get('max_velocity', {}).get('min', 0):.1f} - {new_ranges['Clear'].get('max_velocity', {}).get('max', 0):.1f} (target: {new_ranges['Clear'].get('max_velocity', {}).get('target', 0):.1f})")
    print(f"  Forearm angle: {new_ranges['Clear'].get('r_forearm_vertical_angle_mean', {}).get('min', 0):.1f}° - {new_ranges['Clear'].get('r_forearm_vertical_angle_mean', {}).get('max', 0):.1f}° (target: {new_ranges['Clear'].get('r_forearm_vertical_angle_mean', {}).get('target', 0):.1f}°)")
    print(f"  Contact point: {new_ranges['Clear'].get('r_wrist_height_from_head_mean', {}).get('min', 0):.3f} - {new_ranges['Clear'].get('r_wrist_height_from_head_mean', {}).get('max', 0):.3f} (target: {new_ranges['Clear'].get('r_wrist_height_from_head_mean', {}).get('target', 0):.3f})")
    print()
    print("Smash (Forehand):")
    print(f"  Velocity: {new_ranges['Smash'].get('max_velocity', {}).get('min', 0):.1f} - {new_ranges['Smash'].get('max_velocity', {}).get('max', 0):.1f} (target: {new_ranges['Smash'].get('max_velocity', {}).get('target', 0):.1f})")
    print(f"  Forearm angle: {new_ranges['Smash'].get('r_forearm_vertical_angle_mean', {}).get('min', 0):.1f}° - {new_ranges['Smash'].get('r_forearm_vertical_angle_mean', {}).get('max', 0):.1f}° (target: {new_ranges['Smash'].get('r_forearm_vertical_angle_mean', {}).get('target', 0):.1f}°)")
    print(f"  Contact point: {new_ranges['Smash'].get('r_wrist_height_from_head_mean', {}).get('min', 0):.3f} - {new_ranges['Smash'].get('r_wrist_height_from_head_mean', {}).get('max', 0):.3f} (target: {new_ranges['Smash'].get('r_wrist_height_from_head_mean', {}).get('target', 0):.3f})")


def main():
    """Main execution"""
    print("\n")
    print("=" * 70)
    print("FILTER BACKHAND SHOTS & REGENERATE BENCHMARKS")
    print("=" * 70)
    print()
    print("This script will:")
    print("  1. Filter out all backhand shots from clips_metadata.csv")
    print("  2. Update train/val/test split files")
    print("  3. Recalculate benchmarks from forehand-only strokes")
    print("  4. Update technique_benchmarks.py")
    print()
    print("Note: Original files will be backed up before modification")
    print()

    # Step 1: Filter backhand shots
    df_filtered = filter_backhand_shots()

    # Step 2: Update split files
    update_split_files(df_filtered)

    # Step 3: Check if features need regeneration
    features_path = Path('data/processed/features/statistical_features.pkl')

    if not features_path.exists():
        print("\n" + "=" * 70)
        print("⚠️  FEATURES NOT FOUND")
        print("=" * 70)
        print()
        print("You need to regenerate features with the filtered dataset:")
        print("  python src/data_processing/batch_extract_features.py")
        print()
        print("After that, run this script again to update benchmarks.")
        print()
        return

    # Step 3: Recalculate benchmarks
    benchmarks = recalculate_benchmarks()

    # Step 4: Update technique_benchmarks.py
    update_technique_benchmarks(benchmarks)

    print("\n" + "=" * 70)
    print("✅ COMPLETE")
    print("=" * 70)
    print()
    print("Next steps:")
    print("  1. Test the system with a forehand video:")
    print("     python analyze_video.py <your_video.mp4> Clear")
    print()
    print("  2. If coaching feedback still seems off, you may want to:")
    print("     - Regenerate features: python src/data_processing/batch_extract_features.py")
    print("     - Then run this script again")
    print()
    print("Backup files created (in case you need to restore):")
    print("  - data/processed/clips/clips_metadata_backup.csv")
    print("  - data/processed/splits/*_backup.csv")
    print("  - src/coaching/technique_benchmarks_backup.py")
    print()


if __name__ == "__main__":
    main()
