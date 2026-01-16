#!/usr/bin/env python3
"""
Create Statistical Features PKL from Filtered Metadata

This script creates a combined statistical_features.pkl file using only
the forehand clips from the filtered metadata.

Usage:
    python create_forehand_features_pkl.py
"""

from pathlib import Path
import pandas as pd
import pickle
from tqdm import tqdm


def create_statistical_features_pkl():
    """
    Create a single statistical_features.pkl file from forehand-only clips
    """

    print("=" * 70)
    print("CREATING FOREHAND-ONLY STATISTICAL FEATURES PKL")
    print("=" * 70)
    print()

    metadata_file = Path('data/processed/clips/clips_metadata.csv')
    features_dir = Path('data/processed/features')
    output_file = features_dir / 'statistical_features.pkl'

    # Backup existing file
    if output_file.exists():
        backup_file = features_dir / 'statistical_features_backup.pkl'
        if not backup_file.exists():
            print(f"Creating backup: {backup_file}")
            import shutil
            shutil.copy(output_file, backup_file)

    # Load metadata (already filtered to forehand only)
    df = pd.read_csv(metadata_file)
    print(f"Metadata: {len(df)} forehand clips")

    # Count by stroke type
    clear_count = len(df[df['stroke_type_english'] == 'Clear'])
    smash_count = len(df[df['stroke_type_english'] == 'Smash'])
    print(f"  Clear: {clear_count}")
    print(f"  Smash: {smash_count}")
    print()

    print("Loading features from individual pickle files...")

    features_list = []
    labels_list = []
    clip_names_list = []
    successful = 0
    missing = 0
    failed = 0

    for idx, row in tqdm(df.iterrows(), total=len(df), desc="Loading features"):
        clip_name = row['clip_name']
        stroke_type = row['stroke_type_english']
        feature_path = features_dir / clip_name.replace('.mp4', '_features.pkl')

        if not feature_path.exists():
            missing += 1
            continue

        try:
            with open(feature_path, 'rb') as f:
                feature_data = pickle.load(f)

            # Get statistical summary
            stat_features = feature_data.get('statistical_summary', {})

            if stat_features:
                features_list.append(stat_features)
                labels_list.append(stroke_type)
                clip_names_list.append(clip_name)
                successful += 1
            else:
                failed += 1

        except Exception as e:
            tqdm.write(f"⚠️  Failed to load {clip_name}: {e}")
            failed += 1

    print()
    print("=" * 70)
    print("LOADING SUMMARY")
    print("=" * 70)
    print()
    print(f"Successfully loaded: {successful} clips")
    print(f"Missing feature files: {missing}")
    print(f"Failed to parse: {failed}")
    print()

    if successful == 0:
        print("❌ No features loaded. Cannot create statistical_features.pkl")
        return False

    # Save combined file
    print(f"Saving combined features to: {output_file}")

    combined_data = {
        'features': features_list,
        'labels': labels_list,
        'clip_names': clip_names_list,
        'num_samples': len(features_list)
    }

    with open(output_file, 'wb') as f:
        pickle.dump(combined_data, f)

    print("✓ Saved")
    print()

    # Display statistics
    print("=" * 70)
    print("FEATURE STATISTICS")
    print("=" * 70)
    print()
    print(f"Total samples: {len(features_list)}")
    print(f"Clear samples: {labels_list.count('Clear')}")
    print(f"Smash samples: {labels_list.count('Smash')}")
    print()

    # Sample feature names
    if len(features_list) > 0:
        sample_features = features_list[0]
        print(f"Features per sample: {len(sample_features)}")
        print()
        print("Sample feature names:")
        for i, key in enumerate(list(sample_features.keys())[:5]):
            print(f"  - {key}")
        print(f"  ... ({len(sample_features)} total)")
        print()

    return True


def main():
    """Main entry point"""

    print()
    print("This script creates statistical_features.pkl from forehand-only clips.")
    print()

    success = create_statistical_features_pkl()

    if success:
        print("=" * 70)
        print("✅ SUCCESS")
        print("=" * 70)
        print()
        print("Next step: Run filter script to update benchmarks")
        print("  python filter_backhand_and_regenerate.py")
        print()
    else:
        print("=" * 70)
        print("❌ FAILED")
        print("=" * 70)
        print()


if __name__ == "__main__":
    main()
