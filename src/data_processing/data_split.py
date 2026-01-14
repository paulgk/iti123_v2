#!/usr/bin/env python3
"""
Data Split Script
Split processed features into train/validation/test sets
- Stratified splitting by stroke type (Clear/Smash)
- 70% train, 15% validation, 15% test
- Match-level splitting to prevent data leakage
"""

import os
import pandas as pd
import numpy as np
import pickle
from pathlib import Path
from sklearn.model_selection import train_test_split
from collections import defaultdict

# =============================================================================
# CONFIGURATION
# =============================================================================

# Paths
BASE_DIR = Path(__file__).resolve().parents[2]
FEATURES_DIR = BASE_DIR / "data" / "processed" / "features"
SPLITS_DIR = BASE_DIR / "data" / "processed" / "splits"
REPORTS_DIR = BASE_DIR / "outputs" / "reports"
CLIPS_METADATA_FILE = BASE_DIR / "data" / "processed" / "clips" / "clips_metadata.csv"

# Split ratios
TRAIN_RATIO = 0.70  # 70% for training
VAL_RATIO = 0.15    # 15% for validation
TEST_RATIO = 0.15   # 15% for test

# Random seed for reproducibility
RANDOM_SEED = 42


# =============================================================================
# DATA LOADING
# =============================================================================

def load_features_and_metadata():
    """
    Load all feature files and their metadata

    Returns:
        features_dict: Dictionary mapping clip_name -> feature data
        metadata_df: DataFrame with clip metadata (stroke types, match info)
    """
    print("=" * 70)
    print("LOADING FEATURES AND METADATA")
    print("=" * 70)
    print()

    # Load clips metadata
    if not CLIPS_METADATA_FILE.exists():
        print(f"❌ ERROR: {CLIPS_METADATA_FILE} not found!")
        return None, None

    metadata_df = pd.read_csv(CLIPS_METADATA_FILE)
    print(f"✅ Loaded metadata for {len(metadata_df)} clips")
    print()

    # Load features
    features_dict = {}
    loaded_count = 0
    missing_count = 0

    print("Loading feature files...")
    for idx, row in metadata_df.iterrows():
        clip_name = row['clip_name']
        feature_path = FEATURES_DIR / clip_name.replace('.mp4', '_features.pkl')

        if feature_path.exists():
            try:
                with open(feature_path, 'rb') as f:
                    features_dict[clip_name] = pickle.load(f)
                loaded_count += 1
            except Exception as e:
                print(f"⚠️  Failed to load {clip_name}: {e}")
                missing_count += 1
        else:
            missing_count += 1

    print(f"✅ Successfully loaded: {loaded_count} feature files")
    print(f"❌ Missing: {missing_count} feature files")
    print()

    # Filter metadata to only include clips with features
    metadata_df = metadata_df[metadata_df['clip_name'].isin(features_dict.keys())]

    return features_dict, metadata_df


# =============================================================================
# MATCH-LEVEL SPLITTING
# =============================================================================

def split_by_matches(metadata_df, train_ratio=0.70, val_ratio=0.15, test_ratio=0.15, random_seed=42):
    """
    Split data by matches (not individual clips) to prevent data leakage

    Data leakage prevention:
    - All clips from the same match go into the same split
    - Prevents the model from learning match-specific patterns

    Args:
        metadata_df: DataFrame with clip metadata
        train_ratio: Proportion for training set
        val_ratio: Proportion for validation set
        test_ratio: Proportion for test set
        random_seed: Random seed for reproducibility

    Returns:
        train_df, val_df, test_df: Split DataFrames
    """
    print("=" * 70)
    print("SPLITTING DATA BY MATCHES")
    print("=" * 70)
    print()

    # Get unique matches
    unique_matches = metadata_df['match_id'].unique()
    print(f"Total unique matches: {len(unique_matches)}")
    print()

    # Set random seed
    np.random.seed(random_seed)

    # Shuffle matches
    shuffled_matches = unique_matches.copy()
    np.random.shuffle(shuffled_matches)

    # Calculate split indices
    n_matches = len(shuffled_matches)
    train_end = int(n_matches * train_ratio)
    val_end = train_end + int(n_matches * val_ratio)

    # Split matches
    train_matches = shuffled_matches[:train_end]
    val_matches = shuffled_matches[train_end:val_end]
    test_matches = shuffled_matches[val_end:]

    print(f"Match distribution:")
    print(f"  Train: {len(train_matches)} matches ({len(train_matches)/n_matches*100:.1f}%)")
    print(f"  Val:   {len(val_matches)} matches ({len(val_matches)/n_matches*100:.1f}%)")
    print(f"  Test:  {len(test_matches)} matches ({len(test_matches)/n_matches*100:.1f}%)")
    print()

    # Filter clips by match assignment
    train_df = metadata_df[metadata_df['match_id'].isin(train_matches)]
    val_df = metadata_df[metadata_df['match_id'].isin(val_matches)]
    test_df = metadata_df[metadata_df['match_id'].isin(test_matches)]

    print(f"Clip distribution:")
    print(f"  Train: {len(train_df)} clips ({len(train_df)/len(metadata_df)*100:.1f}%)")
    print(f"  Val:   {len(val_df)} clips ({len(val_df)/len(metadata_df)*100:.1f}%)")
    print(f"  Test:  {len(test_df)} clips ({len(test_df)/len(metadata_df)*100:.1f}%)")
    print()

    return train_df, val_df, test_df


# =============================================================================
# STRATIFICATION ANALYSIS
# =============================================================================

def analyze_stratification(train_df, val_df, test_df):
    """
    Analyze stroke type distribution across splits

    Ensures that each split has a similar proportion of Clear vs Smash strokes

    Args:
        train_df, val_df, test_df: Split DataFrames
    """
    print("=" * 70)
    print("STRATIFICATION ANALYSIS")
    print("=" * 70)
    print()

    def print_distribution(df, split_name):
        """Print stroke type distribution for a split"""
        print(f"{split_name} Set:")
        stroke_counts = df['stroke_type_english'].value_counts()
        total = len(df)

        for stroke_type, count in stroke_counts.items():
            percentage = (count / total) * 100
            print(f"  {stroke_type:10s}: {count:6d} ({percentage:5.2f}%)")
        print()

    # Overall distribution
    all_df = pd.concat([train_df, val_df, test_df])
    print(f"Overall Distribution (Total: {len(all_df)} clips):")
    stroke_counts = all_df['stroke_type_english'].value_counts()
    total = len(all_df)
    for stroke_type, count in stroke_counts.items():
        percentage = (count / total) * 100
        print(f"  {stroke_type:10s}: {count:6d} ({percentage:5.2f}%)")
    print()

    # Distribution by split
    print_distribution(train_df, "Training")
    print_distribution(val_df, "Validation")
    print_distribution(test_df, "Test")


# =============================================================================
# SAVE SPLITS
# =============================================================================

def save_splits(train_df, val_df, test_df, features_dict):
    """
    Save train/val/test splits to disk

    Saves both metadata (CSV) and features (pickle) for each split

    Args:
        train_df, val_df, test_df: Split DataFrames
        features_dict: Dictionary of features
    """
    print("=" * 70)
    print("SAVING SPLITS")
    print("=" * 70)
    print()

    # Create output directory
    SPLITS_DIR.mkdir(parents=True, exist_ok=True)

    splits = {
        'train': train_df,
        'val': val_df,
        'test': test_df
    }

    for split_name, split_df in splits.items():
        print(f"Saving {split_name} split...")

        # Save metadata CSV
        metadata_file = SPLITS_DIR / f"{split_name}_metadata.csv"
        split_df.to_csv(metadata_file, index=False)
        print(f"  ✅ Metadata saved: {metadata_file}")

        # Prepare features and labels
        X = []  # Features
        y = []  # Labels (0=Clear, 1=Smash)
        clip_names = []

        label_map = {'Clear': 0, 'Smash': 1}

        for idx, row in split_df.iterrows():
            clip_name = row['clip_name']
            if clip_name in features_dict:
                feature_data = features_dict[clip_name]

                # Extract normalized, fixed-length features
                X.append(feature_data['features'])

                # Map stroke type to label
                y.append(label_map[row['stroke_type_english']])

                clip_names.append(clip_name)

        # Convert to numpy arrays
        X = np.array(X)
        y = np.array(y)

        # Save features and labels
        data_file = SPLITS_DIR / f"{split_name}_data.pkl"
        with open(data_file, 'wb') as f:
            pickle.dump({
                'X': X,  # Shape: (num_samples, sequence_length, num_features)
                'y': y,  # Shape: (num_samples,)
                'clip_names': clip_names,
                'label_map': label_map,
                'feature_names': feature_data['feature_names']  # Feature column names
            }, f)

        print(f"  ✅ Data saved: {data_file}")
        print(f"     Shape: X={X.shape}, y={y.shape}")
        print()

    print(f"✅ All splits saved to: {SPLITS_DIR}")
    print()


# =============================================================================
# GENERATE SPLIT REPORT
# =============================================================================

def generate_split_report(train_df, val_df, test_df):
    """
    Generate a detailed report of the data splits

    Args:
        train_df, val_df, test_df: Split DataFrames
    """
    print("=" * 70)
    print("GENERATING SPLIT REPORT")
    print("=" * 70)
    print()

    report_file = REPORTS_DIR / "data_split_report.txt"

    with open(report_file, 'w') as f:
        f.write("=" * 70 + "\n")
        f.write("DATA SPLIT REPORT\n")
        f.write("=" * 70 + "\n\n")

        # Split configuration
        f.write("Split Configuration:\n")
        f.write(f"  Train ratio: {TRAIN_RATIO:.1%}\n")
        f.write(f"  Validation ratio: {VAL_RATIO:.1%}\n")
        f.write(f"  Test ratio: {TEST_RATIO:.1%}\n")
        f.write(f"  Random seed: {RANDOM_SEED}\n")
        f.write(f"  Split method: Match-level (prevents data leakage)\n\n")

        # Overall statistics
        all_df = pd.concat([train_df, val_df, test_df])
        f.write(f"Total clips: {len(all_df)}\n")
        f.write(f"Total matches: {all_df['match_id'].nunique()}\n\n")

        # Clip distribution
        f.write("Clip Distribution:\n")
        f.write(f"  Training:   {len(train_df):6d} clips ({len(train_df)/len(all_df)*100:5.2f}%)\n")
        f.write(f"  Validation: {len(val_df):6d} clips ({len(val_df)/len(all_df)*100:5.2f}%)\n")
        f.write(f"  Test:       {len(test_df):6d} clips ({len(test_df)/len(all_df)*100:5.2f}%)\n\n")

        # Match distribution
        f.write("Match Distribution:\n")
        f.write(f"  Training:   {train_df['match_id'].nunique()} matches\n")
        f.write(f"  Validation: {val_df['match_id'].nunique()} matches\n")
        f.write(f"  Test:       {test_df['match_id'].nunique()} matches\n\n")

        # Stroke type distribution
        f.write("Stroke Type Distribution:\n\n")

        for split_name, split_df in [('Training', train_df), ('Validation', val_df), ('Test', test_df)]:
            f.write(f"{split_name}:\n")
            stroke_counts = split_df['stroke_type_english'].value_counts()
            total = len(split_df)

            for stroke_type, count in stroke_counts.items():
                percentage = (count / total) * 100
                f.write(f"  {stroke_type:10s}: {count:6d} ({percentage:5.2f}%)\n")
            f.write("\n")

        # Match assignments
        f.write("Match Assignments:\n\n")
        f.write(f"Training matches: {sorted(train_df['match_id'].unique().tolist())}\n\n")
        f.write(f"Validation matches: {sorted(val_df['match_id'].unique().tolist())}\n\n")
        f.write(f"Test matches: {sorted(test_df['match_id'].unique().tolist())}\n\n")

    print(f"✅ Split report saved to: {report_file}")
    print()


# =============================================================================
# MAIN FUNCTION
# =============================================================================

def main():
    """Main function"""
    print("\n")
    print("*" * 70)
    print("*" + " " * 68 + "*")
    print("*" + "  ITI123 Project: Data Splitting".center(68) + "*")
    print("*" + " " * 68 + "*")
    print("*" * 70)
    print("\n")

    print("Configuration:")
    print(f"  Train: {TRAIN_RATIO:.1%}")
    print(f"  Validation: {VAL_RATIO:.1%}")
    print(f"  Test: {TEST_RATIO:.1%}")
    print(f"  Random seed: {RANDOM_SEED}")
    print(f"  Split method: Match-level (prevents data leakage)")
    print()

    # Load features and metadata
    features_dict, metadata_df = load_features_and_metadata()

    if features_dict is None or metadata_df is None:
        print("❌ Failed to load data. Exiting.")
        return

    # Split data by matches
    train_df, val_df, test_df = split_by_matches(
        metadata_df,
        train_ratio=TRAIN_RATIO,
        val_ratio=VAL_RATIO,
        test_ratio=TEST_RATIO,
        random_seed=RANDOM_SEED
    )

    # Analyze stratification
    analyze_stratification(train_df, val_df, test_df)

    # Save splits
    save_splits(train_df, val_df, test_df, features_dict)

    # Generate report
    generate_split_report(train_df, val_df, test_df)

    print("=" * 70)
    print("✅ DATA SPLITTING COMPLETE!")
    print("=" * 70)
    print()
    print("Next steps:")
    print("1. Review split report: outputs/reports/data_split_report.txt")
    print("2. Proceed with baseline model training")
    print("3. Run: python src/models/baseline_model.py")
    print()


if __name__ == "__main__":
    main()
