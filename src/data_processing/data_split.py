#!/usr/bin/env python3
"""
Data Split Script (Updated for V2 Features)
Split processed features into train/validation/test sets
- Stratified splitting by stroke type (Clear/Smash)
- 70% train, 15% validation, 15% test
- Match-level splitting to prevent data leakage
- Global normalization based on training set
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
FEATURES_DIR = BASE_DIR / "data" / "processed" / "features"  # Original MediaPipe features
SPLITS_DIR = BASE_DIR / "data" / "processed" / "splits"
REPORTS_DIR = BASE_DIR / "outputs" / "reports"
CLIPS_METADATA_FILE = BASE_DIR / "data" / "processed" / "clips" / "clips_metadata.csv"

# Split ratios
TRAIN_RATIO = 0.70  # 70% for training
VAL_RATIO = 0.15    # 15% for validation
TEST_RATIO = 0.15   # 15% for test

# Sequence length for LSTM models
SEQUENCE_LENGTH = 50

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
        print(f"ERROR: {CLIPS_METADATA_FILE} not found!")
        return None, None

    metadata_df = pd.read_csv(CLIPS_METADATA_FILE)
    print(f"Loaded metadata for {len(metadata_df)} clips")
    print()

    # Load features
    features_dict = {}
    loaded_count = 0
    missing_count = 0
    v2_count = 0
    v1_count = 0

    print("Loading feature files...")
    for idx, row in metadata_df.iterrows():
        clip_name = row['clip_name']
        feature_path = FEATURES_DIR / clip_name.replace('.mp4', '_features.pkl')

        if feature_path.exists():
            try:
                with open(feature_path, 'rb') as f:
                    feature_data = pickle.load(f)

                # Detect feature version
                if 'sequence_features' in feature_data:
                    v2_count += 1
                else:
                    v1_count += 1

                features_dict[clip_name] = feature_data
                loaded_count += 1
            except Exception as e:
                print(f"WARNING: Failed to load {clip_name}: {e}")
                missing_count += 1
        else:
            missing_count += 1

    print(f"Successfully loaded: {loaded_count} feature files")
    print(f"  V2 format (new): {v2_count}")
    print(f"  V1 format (old): {v1_count}")
    print(f"Missing: {missing_count} feature files")
    print()

    # Filter metadata to only include clips with features
    metadata_df = metadata_df[metadata_df['clip_name'].isin(features_dict.keys())]

    return features_dict, metadata_df


# =============================================================================
# MATCH-LEVEL SPLITTING
# =============================================================================

def split_by_clips(metadata_df, train_ratio=0.70, val_ratio=0.15, test_ratio=0.15, random_seed=42):
    """
    Split data by clips (for testing purposes only - may have data leakage)

    WARNING: This is for testing only. Use split_by_matches() for production.

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
    print("SPLITTING DATA BY CLIPS (TEST MODE - MAY HAVE DATA LEAKAGE)")
    print("=" * 70)
    print()

    # Stratified split by stroke type
    from sklearn.model_selection import train_test_split

    # First split: train vs (val + test)
    train_df, temp_df = train_test_split(
        metadata_df,
        test_size=(val_ratio + test_ratio),
        stratify=metadata_df['stroke_type_english'],
        random_state=random_seed
    )

    # Second split: val vs test
    val_df, test_df = train_test_split(
        temp_df,
        test_size=test_ratio / (val_ratio + test_ratio),
        stratify=temp_df['stroke_type_english'],
        random_state=random_seed
    )

    print(f"Clip distribution:")
    print(f"  Train: {len(train_df)} clips ({len(train_df)/len(metadata_df)*100:.1f}%)")
    print(f"  Val:   {len(val_df)} clips ({len(val_df)/len(metadata_df)*100:.1f}%)")
    print(f"  Test:  {len(test_df)} clips ({len(test_df)/len(metadata_df)*100:.1f}%)")
    print()

    return train_df, val_df, test_df


def split_by_matches(metadata_df, train_ratio=0.70, val_ratio=0.15, test_ratio=0.15, random_seed=42):
    """
    Split data by matches with stratification by stroke type (GROUP-STRATIFIED SPLIT)

    This ensures:
    1. All clips from the same match stay in the same split (prevents data leakage)
    2. Class balance is maintained across train/val/test (stratification)

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
    print("SPLITTING DATA BY MATCHES (GROUP-STRATIFIED)")
    print("=" * 70)
    print()

    # Set random seed
    np.random.seed(random_seed)

    # Calculate class distribution for each match
    match_stats = []
    for match_id in metadata_df['match_id'].unique():
        match_clips = metadata_df[metadata_df['match_id'] == match_id]
        n_clips = len(match_clips)
        n_clear = len(match_clips[match_clips['stroke_type_english'] == 'Clear'])
        n_smash = len(match_clips[match_clips['stroke_type_english'] == 'Smash'])
        clear_ratio = n_clear / n_clips if n_clips > 0 else 0

        match_stats.append({
            'match_id': match_id,
            'n_clips': n_clips,
            'n_clear': n_clear,
            'n_smash': n_smash,
            'clear_ratio': clear_ratio,
            'dominant_class': 'Clear' if n_clear > n_smash else 'Smash'
        })

    match_stats_df = pd.DataFrame(match_stats)

    print(f"Total unique matches: {len(match_stats_df)}")
    print(f"\nMatch statistics:")
    print(f"  Clear-dominant matches: {len(match_stats_df[match_stats_df['dominant_class'] == 'Clear'])}")
    print(f"  Smash-dominant matches: {len(match_stats_df[match_stats_df['dominant_class'] == 'Smash'])}")
    print()

    # Stratify matches by dominant class to ensure class balance
    clear_dominant = match_stats_df[match_stats_df['dominant_class'] == 'Clear']['match_id'].values
    smash_dominant = match_stats_df[match_stats_df['dominant_class'] == 'Smash']['match_id'].values

    # Shuffle within each stratum
    np.random.shuffle(clear_dominant)
    np.random.shuffle(smash_dominant)

    # Split each stratum according to ratios
    def split_stratum(matches, train_r, val_r, test_r):
        n = len(matches)
        train_end = int(n * train_r)
        val_end = train_end + int(n * val_r)
        return matches[:train_end], matches[train_end:val_end], matches[val_end:]

    clear_train, clear_val, clear_test = split_stratum(clear_dominant, train_ratio, val_ratio, test_ratio)
    smash_train, smash_val, smash_test = split_stratum(smash_dominant, train_ratio, val_ratio, test_ratio)

    # Combine strata
    train_matches = np.concatenate([clear_train, smash_train])
    val_matches = np.concatenate([clear_val, smash_val])
    test_matches = np.concatenate([clear_test, smash_test])

    print(f"Match distribution:")
    print(f"  Train: {len(train_matches)} matches ({len(clear_train)} Clear-dom, {len(smash_train)} Smash-dom)")
    print(f"  Val:   {len(val_matches)} matches ({len(clear_val)} Clear-dom, {len(smash_val)} Smash-dom)")
    print(f"  Test:  {len(test_matches)} matches ({len(clear_test)} Clear-dom, {len(smash_test)} Smash-dom)")
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
# FEATURE EXTRACTION HELPERS
# =============================================================================

def get_sequence_features(feature_data, target_length=50, expected_features=None, return_length=False):
    """
    Extract sequence features from feature data (handles both V1 and V2 formats)

    Args:
        feature_data: Dictionary with feature data
        target_length: Target sequence length
        expected_features: Expected number of features (for consistency check)
        return_length: If True, also return original sequence length

    Returns:
        If return_length=False: features array of shape (target_length, num_features) or None
        If return_length=True: (features, original_length) tuple or (None, 0)
    """
    # V2 format
    if 'sequence_features' in feature_data:
        features = feature_data['sequence_features']
    # V1 format (raw_features or features)
    elif 'raw_features' in feature_data:
        features = feature_data['raw_features']
    else:
        features = feature_data['features']

    # Ensure correct shape
    num_frames, num_features = features.shape
    original_length = num_frames

    # Skip if incompatible with expected features
    if expected_features is not None and num_features != expected_features:
        return (None, 0) if return_length else None

    if num_frames < target_length:
        # Pad with zeros at the end
        padding = np.zeros((target_length - num_frames, num_features))
        features = np.vstack([features, padding])
    elif num_frames > target_length:
        # Take center portion
        start = (num_frames - target_length) // 2
        features = features[start:start + target_length]
        original_length = target_length  # If truncated, effective length is target_length

    if return_length:
        return features, original_length
    return features


def get_statistical_features(feature_data):
    """
    Extract statistical summary features for baseline models

    Args:
        feature_data: Dictionary with feature data

    Returns:
        features: 1D numpy array of statistical features
    """
    # V2 format has pre-computed statistical summary
    if 'statistical_summary' in feature_data:
        summary = feature_data['statistical_summary']
        return np.array(list(summary.values()))

    # V1 format - compute statistics from sequence
    if 'raw_features' in feature_data:
        seq = feature_data['raw_features']
    else:
        seq = feature_data['features']

    # Compute statistics
    features = []
    for col in range(seq.shape[1]):
        col_data = seq[:, col]
        features.extend([
            np.mean(col_data),
            np.std(col_data),
            np.min(col_data),
            np.max(col_data),
        ])

    return np.array(features)


# =============================================================================
# SAVE SPLITS
# =============================================================================

def save_splits(train_df, val_df, test_df, features_dict):
    """
    Save train/val/test splits to disk with GLOBAL normalization.

    Uses raw features and applies normalization based on training set statistics.
    This preserves class-distinguishing information.

    Args:
        train_df, val_df, test_df: Split DataFrames
        features_dict: Dictionary of features
    """
    print("=" * 70)
    print("SAVING SPLITS (with Global Normalization)")
    print("=" * 70)
    print()

    # Create output directory
    SPLITS_DIR.mkdir(parents=True, exist_ok=True)

    label_map = {'Clear': 0, 'Smash': 1}

    # =========================================================================
    # STEP 1: Determine expected feature dimensions from V2 format
    # =========================================================================
    print("Determining feature dimensions from V2 format...")

    # Find a V2 format sample to get expected dimensions
    expected_seq_features = None
    expected_stat_features = None

    for clip_name, feature_data in features_dict.items():
        if 'sequence_features' in feature_data:
            expected_seq_features = feature_data['sequence_features'].shape[1]
            if 'statistical_summary' in feature_data:
                expected_stat_features = len(feature_data['statistical_summary'])
            break

    print(f"  Expected sequence features: {expected_seq_features}")
    print(f"  Expected statistical features: {expected_stat_features}")
    print()

    # =========================================================================
    # STEP 2: Collect features from training set
    # =========================================================================
    print("Collecting training features...")

    X_train_seq = []  # Sequence features for LSTM
    X_train_stat = []  # Statistical features for RF/SVM
    seq_lengths_train = []  # NEW: Original sequence lengths
    y_train = []
    train_clip_names = []
    skipped = 0

    for idx, row in train_df.iterrows():
        clip_name = row['clip_name']
        if clip_name in features_dict:
            feature_data = features_dict[clip_name]

            # Sequence features WITH original length - skip if incompatible
            seq_features, seq_length = get_sequence_features(
                feature_data, SEQUENCE_LENGTH, expected_seq_features, return_length=True
            )
            if seq_features is None:
                skipped += 1
                continue

            X_train_seq.append(seq_features)
            seq_lengths_train.append(seq_length)  # NEW: Save original length

            # Statistical features
            stat_features = get_statistical_features(feature_data)
            X_train_stat.append(stat_features)

            y_train.append(label_map[row['stroke_type_english']])
            train_clip_names.append(clip_name)

    print(f"  Skipped {skipped} incompatible clips (V1 format)")

    X_train_seq = np.array(X_train_seq)
    X_train_stat = np.array(X_train_stat)
    seq_lengths_train = np.array(seq_lengths_train)  # NEW
    y_train = np.array(y_train)

    print(f"  Sequence features shape: {X_train_seq.shape}")
    print(f"  Statistical features shape: {X_train_stat.shape}")
    print()

    # =========================================================================
    # STEP 2: Compute global normalization from training set (ONLY REAL FRAMES)
    # =========================================================================
    print("Computing global normalization statistics (excluding padding)...")

    # Collect only REAL frames (not padding) for normalization
    real_frames = []
    for seq, length in zip(X_train_seq, seq_lengths_train):
        real_frames.append(seq[:length])  # Only actual frames, no padding

    all_train_real = np.vstack(real_frames)  # Stack all real frames
    seq_mean = np.mean(all_train_real, axis=0)
    seq_std = np.std(all_train_real, axis=0) + 1e-8

    # For statistical features (no padding issue)
    stat_mean = np.mean(X_train_stat, axis=0)
    stat_std = np.std(X_train_stat, axis=0) + 1e-8

    print(f"  Computed from {len(all_train_real)} real frames (padding excluded)")
    print(f"  Sequence mean range: [{seq_mean.min():.4f}, {seq_mean.max():.4f}]")
    print(f"  Sequence std range:  [{seq_std.min():.4f}, {seq_std.max():.4f}]")
    print(f"  Statistical mean range: [{stat_mean.min():.4f}, {stat_mean.max():.4f}]")
    print()

    # Normalize training data AND re-pad with zeros AFTER normalization
    X_train_seq_norm = []
    for seq, length in zip(X_train_seq, seq_lengths_train):
        # Normalize only real frames
        normalized = (seq[:length] - seq_mean) / seq_std

        # Re-pad with zeros AFTER normalization
        if length < SEQUENCE_LENGTH:
            padding = np.zeros((SEQUENCE_LENGTH - length, seq.shape[1]))
            normalized = np.vstack([normalized, padding])

        X_train_seq_norm.append(normalized)

    X_train_seq_norm = np.array(X_train_seq_norm)
    X_train_stat_norm = (X_train_stat - stat_mean) / stat_std

    # Save normalization parameters
    norm_params = {
        'seq_mean': seq_mean,
        'seq_std': seq_std,
        'stat_mean': stat_mean,
        'stat_std': stat_std
    }
    with open(SPLITS_DIR / 'normalization_params.pkl', 'wb') as f:
        pickle.dump(norm_params, f)
    print(f"  Normalization params saved")
    print()

    # =========================================================================
    # STEP 3: Process validation and test sets
    # =========================================================================
    splits_data = {
        'train': {
            'X_seq': X_train_seq_norm,
            'X_stat': X_train_stat_norm,
            'X_stat_raw': X_train_stat,  # Raw features for baseline models
            'seq_lengths': seq_lengths_train,  # NEW: sequence lengths
            'y': y_train,
            'clip_names': train_clip_names,
            'df': train_df
        }
    }

    for split_name, split_df in [('val', val_df), ('test', test_df)]:
        X_seq = []
        X_stat = []
        seq_lengths = []  # NEW
        y = []
        clip_names = []
        split_skipped = 0

        for idx, row in split_df.iterrows():
            clip_name = row['clip_name']
            if clip_name in features_dict:
                feature_data = features_dict[clip_name]

                # Skip incompatible features - WITH length
                seq_features, seq_length = get_sequence_features(
                    feature_data, SEQUENCE_LENGTH, expected_seq_features, return_length=True
                )
                if seq_features is None:
                    split_skipped += 1
                    continue

                X_seq.append(seq_features)
                seq_lengths.append(seq_length)  # NEW

                stat_features = get_statistical_features(feature_data)
                X_stat.append(stat_features)

                y.append(label_map[row['stroke_type_english']])
                clip_names.append(clip_name)

        if split_skipped > 0:
            print(f"  {split_name}: Skipped {split_skipped} incompatible clips")

        X_seq = np.array(X_seq)
        X_stat = np.array(X_stat)
        seq_lengths = np.array(seq_lengths)  # NEW
        y = np.array(y)

        # Apply GLOBAL normalization (from training set) AND re-pad with zeros
        X_seq_norm = []
        for seq, length in zip(X_seq, seq_lengths):
            # Normalize only real frames
            normalized = (seq[:length] - seq_mean) / seq_std

            # Re-pad with zeros AFTER normalization
            if length < SEQUENCE_LENGTH:
                padding = np.zeros((SEQUENCE_LENGTH - length, seq.shape[1]))
                normalized = np.vstack([normalized, padding])

            X_seq_norm.append(normalized)

        X_seq_norm = np.array(X_seq_norm)
        X_stat_norm = (X_stat - stat_mean) / stat_std

        splits_data[split_name] = {
            'X_seq': X_seq_norm,
            'X_stat': X_stat_norm,
            'X_stat_raw': X_stat,  # Raw features for baseline models
            'seq_lengths': seq_lengths,  # NEW: sequence lengths
            'y': y,
            'clip_names': clip_names,
            'df': split_df
        }

    # =========================================================================
    # STEP 4: Save all splits (both normalized and raw)
    # =========================================================================
    for split_name, data in splits_data.items():
        print(f"Saving {split_name} split...")

        # Save metadata CSV
        metadata_file = SPLITS_DIR / f"{split_name}_metadata.csv"
        data['df'].to_csv(metadata_file, index=False)
        print(f"  Metadata saved: {metadata_file}")

        # Save features and labels (include raw features AND sequence lengths)
        data_file = SPLITS_DIR / f"{split_name}_data.pkl"
        with open(data_file, 'wb') as f:
            pickle.dump({
                'X': data['X_seq'],  # Normalized for LSTM: (num_samples, seq_length, num_features)
                'X_stat': data['X_stat'],  # Normalized statistical features
                'X_stat_raw': data.get('X_stat_raw', data['X_stat']),  # Raw statistical features
                'seq_lengths': data['seq_lengths'],  # NEW: Original sequence lengths for masking
                'y': data['y'],
                'clip_names': data['clip_names'],
                'label_map': label_map
            }, f)

        print(f"  Data saved: {data_file}")
        print(f"     Sequence: {data['X_seq'].shape}")
        print(f"     Statistical: {data['X_stat'].shape}")
        print(f"     Seq lengths: {data['seq_lengths'].shape} (min={data['seq_lengths'].min()}, max={data['seq_lengths'].max()})")
        print(f"     Labels: {data['y'].shape}")
        print()

    print(f"All splits saved to: {SPLITS_DIR}")
    print()

    # =========================================================================
    # STEP 5: Verify class separation
    # =========================================================================
    print("Verifying class separation after global normalization...")
    X_seq = splits_data['train']['X_seq']
    y = splits_data['train']['y']

    X_clear = X_seq[y == 0]
    X_smash = X_seq[y == 1]

    # Per-feature analysis
    print(f"\nPer-feature mean differences (sequence):")
    clear_means = np.mean(X_clear, axis=(0, 1))
    smash_means = np.mean(X_smash, axis=(0, 1))

    # Find most discriminative features
    diffs = np.abs(clear_means - smash_means)
    top_indices = np.argsort(diffs)[-5:][::-1]

    print(f"  Top 5 most discriminative features:")
    for idx in top_indices:
        print(f"    Feature {idx}: Clear={clear_means[idx]:.4f}, Smash={smash_means[idx]:.4f}, Diff={diffs[idx]:.4f}")

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

    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
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
        f.write(f"  Sequence length: {SEQUENCE_LENGTH}\n")
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

    print(f"Split report saved to: {report_file}")
    print()


# =============================================================================
# MAIN FUNCTION
# =============================================================================

def main():
    """Main function"""
    print("\n")
    print("*" * 70)
    print("*" + " " * 68 + "*")
    print("*" + "  ITI123 Project: Data Splitting (V2)".center(68) + "*")
    print("*" + " " * 68 + "*")
    print("*" * 70)
    print("\n")

    print("Configuration:")
    print(f"  Train: {TRAIN_RATIO:.1%}")
    print(f"  Validation: {VAL_RATIO:.1%}")
    print(f"  Test: {TEST_RATIO:.1%}")
    print(f"  Sequence length: {SEQUENCE_LENGTH}")
    print(f"  Random seed: {RANDOM_SEED}")
    print(f"  Split method: Match-level (prevents data leakage)")
    print()

    # Load features and metadata
    features_dict, metadata_df = load_features_and_metadata()

    if features_dict is None or metadata_df is None:
        print("Failed to load data. Exiting.")
        return

    # Split data by matches (prevents data leakage)
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
    print("DATA SPLITTING COMPLETE!")
    print("=" * 70)
    print()
    print("Next steps:")
    print("1. Review split report: outputs/reports/data_split_report.txt")
    print("2. Train models in Colab using the saved splits")
    print()


if __name__ == "__main__":
    main()
