#!/usr/bin/env python3
"""
Analyze Wrist Orientation Features - Cohen's d Analysis

Check if the new wrist/forearm orientation features separate Clear vs Smash
"""

import pickle
import numpy as np
import pandas as pd
from pathlib import Path
from scipy import stats

# Paths
BASE_DIR = Path(__file__).resolve().parents[2]
FEATURES_DIR = BASE_DIR / "data" / "processed" / "features"
CLIPS_METADATA_FILE = BASE_DIR / "data" / "processed" / "clips" / "clips_metadata.csv"
REPORTS_DIR = BASE_DIR / "outputs" / "reports"


def cohens_d(group1, group2):
    """Calculate Cohen's d effect size"""
    n1, n2 = len(group1), len(group2)
    var1, var2 = np.var(group1, ddof=1), np.var(group2, ddof=1)
    pooled_std = np.sqrt(((n1-1)*var1 + (n2-1)*var2) / (n1+n2-2))
    return (np.mean(group1) - np.mean(group2)) / (pooled_std + 1e-8)


def main():
    print("\n" + "="*70)
    print("WRIST ORIENTATION FEATURES - COHEN'S D ANALYSIS")
    print("="*70)
    print()

    # Load metadata
    metadata_df = pd.read_csv(CLIPS_METADATA_FILE)
    print(f"Total clips in metadata: {len(metadata_df)}")

    # Load features and separate by class
    clear_features = []
    smash_features = []
    feature_names = None

    print("Loading features...")
    for idx, row in metadata_df.iterrows():
        clip_name = row['clip_name']
        feature_file = FEATURES_DIR / clip_name.replace('.mp4', '_features.pkl')

        if not feature_file.exists():
            continue

        with open(feature_file, 'rb') as f:
            feature_data = pickle.load(f)

        # Get sequence features
        if 'sequence_features' in feature_data:
            seq_features = feature_data['sequence_features']  # (T, F)

            # Get feature names
            if feature_names is None and 'feature_names' in feature_data:
                feature_names = feature_data['feature_names']

            # Compute mean across time for each feature
            mean_features = np.mean(seq_features, axis=0)

            if row['stroke_type_english'] == 'Clear':
                clear_features.append(mean_features)
            elif row['stroke_type_english'] == 'Smash':
                smash_features.append(mean_features)

    clear_features = np.array(clear_features)
    smash_features = np.array(smash_features)

    print(f"Clear samples: {len(clear_features)}")
    print(f"Smash samples: {len(smash_features)}")
    print(f"Features per sample: {clear_features.shape[1]}")
    print()

    # Calculate Cohen's d for each feature
    results = []

    for i in range(clear_features.shape[1]):
        clear_vals = clear_features[:, i]
        smash_vals = smash_features[:, i]

        d = cohens_d(clear_vals, smash_vals)
        t_stat, p_value = stats.ttest_ind(clear_vals, smash_vals)

        feature_name = feature_names[i] if feature_names and i < len(feature_names) else f"feature_{i}"

        results.append({
            'feature_idx': i,
            'feature_name': feature_name,
            'cohens_d': d,
            'abs_cohens_d': abs(d),
            't_statistic': t_stat,
            'p_value': p_value,
            'clear_mean': np.mean(clear_vals),
            'smash_mean': np.mean(smash_vals),
            'clear_std': np.std(clear_vals),
            'smash_std': np.std(smash_vals)
        })

    results_df = pd.DataFrame(results)
    results_df = results_df.sort_values('abs_cohens_d', ascending=False)

    # Print top features
    print("="*70)
    print("TOP 20 MOST DISCRIMINATIVE FEATURES (by |Cohen's d|)")
    print("="*70)
    print()
    cohens_label = "Cohen's d"
    print(f"{'Rank':<6} {'Feature Name':<40} {cohens_label:<12} {'p-value':<12}")
    print("-"*70)

    for rank, (idx, row) in enumerate(results_df.head(20).iterrows(), 1):
        significance = "***" if row['p_value'] < 0.001 else "**" if row['p_value'] < 0.01 else "*" if row['p_value'] < 0.05 else ""
        print(f"{rank:<6} {row['feature_name']:<40} {row['cohens_d']:>10.4f}  {row['p_value']:>10.4e} {significance}")

    print()

    # Focus on wrist orientation features
    print("="*70)
    print("WRIST ORIENTATION FEATURES ANALYSIS")
    print("="*70)
    print()

    wrist_keywords = ['forearm', 'wrist_elbow', 'arm_plane', 'horizontal_reach', 'pitch']
    wrist_features = results_df[results_df['feature_name'].str.contains('|'.join(wrist_keywords), case=False, na=False)]

    if len(wrist_features) > 0:
        print(f"Found {len(wrist_features)} wrist/forearm orientation features:")
        print()
        print(f"{'Feature Name':<50} {cohens_label:<12} {'Clear Mean':<12} {'Smash Mean':<12}")
        print("-"*90)

        for idx, row in wrist_features.iterrows():
            print(f"{row['feature_name']:<50} {row['cohens_d']:>10.4f}  {row['clear_mean']:>10.4f}  {row['smash_mean']:>10.4f}")

        print()
        print("Key observations:")
        print(f"  Max |Cohen's d| in wrist features: {wrist_features['abs_cohens_d'].max():.4f}")
        print(f"  Min |Cohen's d| in wrist features: {wrist_features['abs_cohens_d'].min():.4f}")
        print(f"  Mean |Cohen's d| in wrist features: {wrist_features['abs_cohens_d'].mean():.4f}")
    else:
        print("WARNING: No wrist orientation features found!")
        print("Feature names may not match expected patterns.")

    print()

    # Interpretation guide
    print("="*70)
    print("COHEN'S D INTERPRETATION")
    print("="*70)
    print("  |d| < 0.2  : Negligible effect (features won't help)")
    print("  |d| = 0.2-0.5 : Small effect (may help slightly)")
    print("  |d| = 0.5-0.8 : Medium effect (good discriminator)")
    print("  |d| > 0.8  : Large effect (excellent discriminator)")
    print()

    # Save results
    results_df.to_csv(REPORTS_DIR / 'wrist_features_cohens_d.csv', index=False)
    print(f"Full results saved to: {REPORTS_DIR / 'wrist_features_cohens_d.csv'}")

    # Print summary statistics
    print()
    print("="*70)
    print("OVERALL SUMMARY")
    print("="*70)
    print(f"Features with |d| > 0.8 (large effect): {len(results_df[results_df['abs_cohens_d'] > 0.8])}")
    print(f"Features with |d| > 0.5 (medium effect): {len(results_df[results_df['abs_cohens_d'] > 0.5])}")
    print(f"Features with |d| > 0.2 (small effect): {len(results_df[results_df['abs_cohens_d'] > 0.2])}")
    print(f"Features with |d| < 0.2 (negligible): {len(results_df[results_df['abs_cohens_d'] < 0.2])}")
    print()


if __name__ == "__main__":
    main()
