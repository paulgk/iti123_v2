#!/usr/bin/env python3
"""
Regenerate Features for Filtered Dataset

This script extracts pose features for all clips in the filtered metadata
(forehand-only strokes).

Usage:
    python regenerate_features.py
"""

import sys
from pathlib import Path
import pandas as pd
import pickle
import numpy as np
from tqdm import tqdm

# Add parent directory to path
sys.path.append(str(Path(__file__).resolve().parent))

from src.data_processing.extract_poses import PoseExtractor
from src.data_processing.feature_engineering_v2 import (
    extract_frame_features,
    extract_temporal_features,
    extract_statistical_summary
)


def regenerate_all_features():
    """Extract features for all clips in filtered metadata"""

    # Paths
    metadata_file = Path('data/processed/clips/clips_metadata.csv')
    clips_dir = Path('data/processed/clips')
    features_dir = Path('data/processed/features')

    # Load metadata
    print("=" * 70)
    print("REGENERATING FEATURES FOR FOREHAND-ONLY DATASET")
    print("=" * 70)
    print()

    if not metadata_file.exists():
        print(f"❌ Error: {metadata_file} not found")
        return False

    df = pd.read_csv(metadata_file)
    print(f"Loaded metadata: {len(df)} clips")

    # Count by stroke type
    clear_count = len(df[df['stroke_type_english'] == 'Clear'])
    smash_count = len(df[df['stroke_type_english'] == 'Smash'])
    print(f"  Clear: {clear_count}")
    print(f"  Smash: {smash_count}")
    print()

    # Initialize pose extractor
    print("Initializing MediaPipe Pose...")
    extractor = PoseExtractor()
    print("✓ Ready")
    print()

    # Process each clip
    print("=" * 70)
    print("EXTRACTING FEATURES")
    print("=" * 70)
    print()

    success_count = 0
    fail_count = 0
    already_exists = 0

    for idx, row in tqdm(df.iterrows(), total=len(df), desc="Processing clips"):
        clip_name = row['clip_name']
        stroke_type = row['stroke_type_english']
        clip_path = clips_dir / clip_name
        feature_path = features_dir / clip_name.replace('.mp4', '_features.pkl')

        # Skip if feature file already exists (optional - remove this to reprocess all)
        # if feature_path.exists():
        #     already_exists += 1
        #     continue

        # Check if clip exists
        if not clip_path.exists():
            tqdm.write(f"⚠️  Clip not found: {clip_name}")
            fail_count += 1
            continue

        try:
            # Extract poses
            pose_data = extractor.extract_from_video(str(clip_path), stroke_type)
            pose_sequence = pose_data['poses']
            quality = pose_data['quality']

            # Extract features
            per_frame_features, feature_names, temporal_metrics = extract_temporal_features(pose_sequence)

            if per_frame_features is None:
                tqdm.write(f"⚠️  Failed to extract features for {clip_name}")
                fail_count += 1
                continue

            # Extract statistical summary
            stat_features = extract_statistical_summary(per_frame_features, feature_names)
            stat_features.update(temporal_metrics)

            # Save feature data (V2 format)
            feature_data = {
                'sequence_features': per_frame_features,  # V2 format indicator
                'feature_names': feature_names,
                'statistical_summary': stat_features,
                'temporal_metrics': temporal_metrics,
                'quality': quality,
                'stroke_type': stroke_type,
                'num_frames': len(pose_sequence)
            }

            # Create features directory if needed
            features_dir.mkdir(parents=True, exist_ok=True)

            # Save
            with open(feature_path, 'wb') as f:
                pickle.dump(feature_data, f)

            success_count += 1

        except Exception as e:
            tqdm.write(f"❌ Error processing {clip_name}: {e}")
            fail_count += 1
            continue

    # Cleanup
    extractor.close()

    # Summary
    print()
    print("=" * 70)
    print("FEATURE EXTRACTION COMPLETE")
    print("=" * 70)
    print()
    print(f"Successful: {success_count}")
    print(f"Failed: {fail_count}")
    print(f"Already existed: {already_exists}")
    print()

    if success_count > 0:
        print("✓ Features regenerated successfully")
        print()
        print("Next step: Run filter script again to update benchmarks")
        print("  python filter_backhand_and_regenerate.py")
        print()
        return True
    else:
        print("❌ No features were generated")
        return False


def create_statistical_features_pkl():
    """
    Create a single statistical_features.pkl file from all individual feature files.
    This is needed for benchmark calculation.
    """

    print("=" * 70)
    print("CREATING STATISTICAL FEATURES PKL")
    print("=" * 70)
    print()

    metadata_file = Path('data/processed/clips/clips_metadata.csv')
    features_dir = Path('data/processed/features')
    output_file = features_dir / 'statistical_features.pkl'

    # Load metadata
    df = pd.read_csv(metadata_file)
    print(f"Loading features for {len(df)} clips...")

    features_list = []
    labels_list = []
    clip_names_list = []
    successful = 0
    failed = 0

    for idx, row in tqdm(df.iterrows(), total=len(df), desc="Loading features"):
        clip_name = row['clip_name']
        stroke_type = row['stroke_type_english']
        feature_path = features_dir / clip_name.replace('.mp4', '_features.pkl')

        if not feature_path.exists():
            failed += 1
            continue

        try:
            with open(feature_path, 'rb') as f:
                feature_data = pickle.load(f)

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
    print(f"Loaded: {successful} feature sets")
    print(f"Failed: {failed}")

    if successful == 0:
        print("❌ No features loaded")
        return False

    # Save combined file
    print()
    print(f"Saving to {output_file}...")

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

    return True


def main():
    """Main entry point"""

    print()
    print("This script will:")
    print("  1. Extract pose features for all forehand-only clips")
    print("  2. Create combined statistical_features.pkl file")
    print()
    print("This may take 30-60 minutes depending on dataset size.")
    print()

    # Step 1: Regenerate individual feature files
    success = regenerate_all_features()

    if not success:
        print("Failed to regenerate features. Exiting.")
        return

    # Step 2: Create combined statistical features file
    success = create_statistical_features_pkl()

    if not success:
        print("Failed to create statistical_features.pkl. Exiting.")
        return

    print("=" * 70)
    print("✅ ALL DONE")
    print("=" * 70)
    print()
    print("Next step: Update benchmarks")
    print("  python filter_backhand_and_regenerate.py")
    print()


if __name__ == "__main__":
    main()
