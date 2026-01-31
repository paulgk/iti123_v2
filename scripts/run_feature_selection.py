#!/usr/bin/env python3
"""
Feature Selection Pipeline Runner

Extracts v3 features from pose sequences, runs two-stage feature selection
(filter -> wrapper), and saves selected features to manifest.

Usage:
    python scripts/run_feature_selection.py [--sample-size N] [--target-features N]

Options:
    --sample-size N      Number of samples to process (default: all)
    --target-features N  Maximum features to select (default: 254)
    --skip-extraction    Use cached features if available
    --verbose           Show detailed progress
"""

import argparse
import os
import pickle
import json
from pathlib import Path
from datetime import datetime
import pandas as pd
import numpy as np
from tqdm import tqdm

# Add project root to path
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data_processing.feature_engineering_v3 import extract_features_v3
from src.data_processing.feature_selection import (
    feature_selection_pipeline,
    save_feature_selection_mask,
    generate_report
)


def load_pose_sequence(pose_path):
    """Load pose sequence from pickle file"""
    with open(pose_path, 'rb') as f:
        return pickle.load(f)


def load_metadata(metadata_path='data/metadata.csv'):
    """Load metadata with stroke labels"""
    if not os.path.exists(metadata_path):
        raise FileNotFoundError(
            f"Metadata file not found: {metadata_path}\n"
            f"Expected CSV with columns: video_id, pose_file, stroke_type"
        )

    df = pd.read_csv(metadata_path)

    # Validate required columns
    required_cols = ['pose_file', 'stroke_type']
    missing = [col for col in required_cols if col not in df.columns]
    if missing:
        raise ValueError(f"Metadata missing columns: {missing}")

    # Filter to Clear and Smash only (binary classification)
    df = df[df['stroke_type'].isin(['clear', 'smash'])].copy()

    print(f"Loaded metadata: {len(df)} samples")
    print(f"  Clear: {sum(df['stroke_type'] == 'clear')}")
    print(f"  Smash: {sum(df['stroke_type'] == 'smash')}")

    return df


def extract_all_features(metadata, sample_size=None, verbose=False):
    """
    Extract v3 features from all pose sequences

    Args:
        metadata: DataFrame with pose_file and stroke_type columns
        sample_size: Optional limit on number of samples
        verbose: Show detailed progress

    Returns:
        X: DataFrame of features
        y: Series of labels (0=clear, 1=smash)
    """
    if sample_size:
        metadata = metadata.sample(n=min(sample_size, len(metadata)), random_state=42)
        print(f"Sampling {len(metadata)} clips")

    all_features = []
    labels = []
    failed = []

    iterator = tqdm(metadata.iterrows(), total=len(metadata), desc="Extracting features")

    for idx, row in iterator:
        try:
            # Load pose sequence
            pose_path = row['pose_file']
            if not os.path.isabs(pose_path):
                pose_path = os.path.join('data/processed/poses', pose_path)

            pose_sequence = load_pose_sequence(pose_path)

            # Extract features (without selection - we need all features for pipeline)
            features = extract_features_v3(pose_sequence, apply_selection=False)

            # Remove metadata fields
            features = {k: v for k, v in features.items()
                       if not k.startswith('_meta') and k != 'feature_version'}

            all_features.append(features)
            labels.append(1 if row['stroke_type'] == 'smash' else 0)

            if verbose and len(all_features) % 100 == 0:
                print(f"  Processed {len(all_features)} samples, {len(features)} features")

        except Exception as e:
            failed.append({'file': row['pose_file'], 'error': str(e)})
            if verbose:
                print(f"  Failed: {row['pose_file']} - {e}")

    if failed:
        print(f"\n⚠️  {len(failed)} samples failed extraction")
        with open('outputs/failed_extractions.json', 'w') as f:
            json.dump(failed, f, indent=2)
        print(f"   See outputs/failed_extractions.json for details")

    X = pd.DataFrame(all_features)
    y = pd.Series(labels, name='stroke_type')

    print(f"\n✓ Extracted features from {len(X)} samples")
    print(f"  Feature dimensions: {X.shape}")
    print(f"  Label distribution: {y.value_counts().to_dict()}")

    return X, y


def cache_features(X, y, cache_path='outputs/feature_cache.pkl'):
    """Cache extracted features for faster re-runs"""
    os.makedirs(os.path.dirname(cache_path), exist_ok=True)
    with open(cache_path, 'wb') as f:
        pickle.dump({'X': X, 'y': y, 'timestamp': datetime.now()}, f)
    print(f"✓ Cached features to {cache_path}")


def load_cached_features(cache_path='outputs/feature_cache.pkl'):
    """Load cached features if available"""
    if not os.path.exists(cache_path):
        return None, None

    with open(cache_path, 'rb') as f:
        cache = pickle.load(f)

    print(f"✓ Loaded cached features from {cache_path}")
    print(f"  Cached: {cache['timestamp']}")

    return cache['X'], cache['y']


def run_pipeline(X, y, target_features=254):
    """
    Run two-stage feature selection pipeline

    Args:
        X: Feature matrix
        y: Labels
        target_features: Maximum number of features to select

    Returns:
        pipeline_results: Dict with selected features and metrics
    """
    print(f"\n{'='*60}")
    print(f"FEATURE SELECTION PIPELINE")
    print(f"{'='*60}")
    print(f"Initial features: {len(X.columns)}")
    print(f"Target features: <{target_features}")
    print(f"Samples: {len(X)}")
    print()

    # Run pipeline
    pipeline_results = feature_selection_pipeline(
        X, y,
        target_features=target_features
    )

    # Print results
    print(f"\n{'='*60}")
    print(f"PIPELINE RESULTS")
    print(f"{'='*60}")

    stages = pipeline_results['stages']
    print(f"\nStage Summary:")
    print(f"  Initial features:           {stages['initial']}")
    print(f"  After zero-variance:        {stages['after_zero_var']} (-{stages['initial'] - stages['after_zero_var']})")
    print(f"  After Cohen's d >= 0.5:     {stages['after_cohens_d']} (-{stages['after_zero_var'] - stages['after_cohens_d']})")
    print(f"  After VIF < 10:             {stages['after_vif']} (-{stages['after_cohens_d'] - stages['after_vif']})")
    print(f"  After RFECV (final):        {stages['after_rfecv']} (-{stages['after_vif'] - stages['after_rfecv']})")

    print(f"\nFinal Feature Count: {len(pipeline_results['selected_features'])}")
    print(f"Target Met: {'✓ PASS' if len(pipeline_results['selected_features']) < target_features else '✗ FAIL'}")
    print(f"CV F1 Score: {pipeline_results['cv_f1_score']:.4f}")

    # Show top features by effect size
    print(f"\nTop 10 Features by Effect Size:")
    effect_sizes = pipeline_results['effect_sizes']
    top_features = sorted(effect_sizes.items(), key=lambda x: abs(x[1]), reverse=True)[:10]
    for feat, d in top_features:
        if feat in pipeline_results['selected_features']:
            print(f"  {feat:40s} d={d:6.3f}  ✓ selected")
        else:
            print(f"  {feat:40s} d={d:6.3f}  (removed)")

    return pipeline_results


def save_results(pipeline_results):
    """Save feature selection results"""
    # Save selected features manifest
    manifest_path = 'data/processed/features_v3/selected_features.json'
    save_feature_selection_mask(
        pipeline_results['selected_features'],
        manifest_path
    )
    print(f"\n✓ Saved feature manifest: {manifest_path}")

    # Generate report
    report_path = 'outputs/reports/feature_selection_report.md'
    generate_report(pipeline_results, report_path)
    print(f"✓ Saved selection report: {report_path}")

    # Save full results to pickle for analysis
    results_path = 'outputs/feature_selection_results.pkl'
    with open(results_path, 'wb') as f:
        pickle.dump(pipeline_results, f)
    print(f"✓ Saved full results: {results_path}")


def main():
    parser = argparse.ArgumentParser(
        description='Run feature selection pipeline',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    parser.add_argument('--sample-size', type=int, default=None,
                       help='Number of samples to process (default: all)')
    parser.add_argument('--target-features', type=int, default=254,
                       help='Maximum features to select (default: 254)')
    parser.add_argument('--skip-extraction', action='store_true',
                       help='Use cached features if available')
    parser.add_argument('--verbose', action='store_true',
                       help='Show detailed progress')
    parser.add_argument('--metadata', type=str, default='data/metadata.csv',
                       help='Path to metadata CSV (default: data/metadata.csv)')

    args = parser.parse_args()

    print(f"\n{'='*60}")
    print(f"FEATURE SELECTION PIPELINE RUNNER")
    print(f"{'='*60}")
    print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()

    # Create output directories
    os.makedirs('outputs/reports', exist_ok=True)
    os.makedirs('data/processed/features_v3', exist_ok=True)

    # Step 1: Extract features or load cached
    if args.skip_extraction:
        print("Attempting to load cached features...")
        X, y = load_cached_features()
        if X is None:
            print("⚠️  No cached features found, extracting from scratch")
            args.skip_extraction = False

    if not args.skip_extraction:
        print("Step 1: Loading metadata and extracting features")
        print("-" * 60)
        metadata = load_metadata(args.metadata)
        X, y = extract_all_features(metadata, args.sample_size, args.verbose)
        cache_features(X, y)

    # Step 2: Run feature selection pipeline
    print("\nStep 2: Running feature selection pipeline")
    print("-" * 60)
    pipeline_results = run_pipeline(X, y, args.target_features)

    # Step 3: Save results
    print("\nStep 3: Saving results")
    print("-" * 60)
    save_results(pipeline_results)

    print(f"\n{'='*60}")
    print(f"FEATURE SELECTION COMPLETE ✓")
    print(f"{'='*60}")
    print(f"Finished: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()
    print("Next steps:")
    print("  1. Review report: outputs/reports/feature_selection_report.md")
    print("  2. Verify manifest: data/processed/features_v3/selected_features.json")
    print("  3. Test v3 extraction: python -c 'from src.data_processing.feature_engineering_v3 import extract_features_v3; ...'")
    print()


if __name__ == '__main__':
    main()
