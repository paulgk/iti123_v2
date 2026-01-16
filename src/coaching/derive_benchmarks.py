"""
Derive Professional Technique Benchmarks from ShuttleSet Data

This script analyzes your v1.0 features to establish realistic benchmark ranges
for professional technique by looking at the top 20-30% performers.
"""

import pickle
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, Tuple
import matplotlib.pyplot as plt
import seaborn as sns

# Paths
BASE_DIR = Path(__file__).resolve().parents[2]
FEATURES_DIR = BASE_DIR / "data" / "processed" / "features"
SPLITS_DIR = BASE_DIR / "data" / "processed" / "splits"
OUTPUT_DIR = BASE_DIR / "outputs" / "reports"

# Key metrics to analyze
KEY_METRICS = [
    'r_arm_extension_mean',
    'max_velocity',
    'r_elbow_angle_mean',
    'torso_lean_mean',
    'r_wrist_height_rel_mean',
    'peak_velocity_timing',
    'r_forearm_vertical_angle_mean',
    'shoulder_rotation_mean'
]

def load_all_features(stroke_type: str = None) -> pd.DataFrame:
    """
    Load all feature files and combine into DataFrame

    Args:
        stroke_type: Filter by 'Clear' or 'Smash' (None = all)

    Returns:
        DataFrame with all features
    """
    print(f"Loading features from {FEATURES_DIR}...")

    # Load metadata to get stroke types
    train_meta = pd.read_csv(SPLITS_DIR / "train_metadata.csv")

    if stroke_type:
        train_meta = train_meta[train_meta['stroke_type_english'] == stroke_type]

    features_list = []

    for idx, row in train_meta.iterrows():
        clip_name = row['clip_name']
        feature_file = FEATURES_DIR / clip_name.replace('.mp4', '_features.pkl')

        if feature_file.exists():
            try:
                with open(feature_file, 'rb') as f:
                    data = pickle.load(f)

                stat_features = data['statistical_summary']
                temporal_features = data.get('temporal_features', {})

                # Combine
                all_features = {**stat_features, **temporal_features}
                all_features['clip_name'] = clip_name
                all_features['stroke_type'] = row['stroke_type_english']

                features_list.append(all_features)

            except Exception as e:
                print(f"Error loading {clip_name}: {e}")

    print(f"Loaded {len(features_list)} feature sets")

    return pd.DataFrame(features_list)


def calculate_percentile_ranges(df: pd.DataFrame, metric: str) -> Dict:
    """
    Calculate percentile-based ranges for a metric

    Professional range = 25th to 75th percentile (middle 50%)
    Optimal = median
    """
    values = df[metric].dropna()

    if len(values) == 0:
        return None

    return {
        'min': float(np.percentile(values, 25)),
        'max': float(np.percentile(values, 75)),
        'optimal': float(np.median(values)),
        'p10': float(np.percentile(values, 10)),
        'p90': float(np.percentile(values, 90)),
        'mean': float(np.mean(values)),
        'std': float(np.std(values))
    }


def derive_benchmarks(stroke_type: str) -> Dict:
    """
    Derive benchmark ranges for a stroke type

    Args:
        stroke_type: 'Clear' or 'Smash'

    Returns:
        Dictionary of benchmarks for each metric
    """
    print(f"\n{'='*70}")
    print(f"Deriving benchmarks for {stroke_type}")
    print(f"{'='*70}\n")

    # Load features
    df = load_all_features(stroke_type=stroke_type)

    if len(df) == 0:
        print(f"No data found for {stroke_type}")
        return {}

    print(f"Analyzing {len(df)} {stroke_type} strokes\n")

    benchmarks = {}

    for metric in KEY_METRICS:
        if metric not in df.columns:
            print(f"⚠️  Metric '{metric}' not found in features")
            continue

        ranges = calculate_percentile_ranges(df, metric)

        if ranges is None:
            print(f"⚠️  No valid data for '{metric}'")
            continue

        benchmarks[metric] = ranges

        print(f"✓ {metric}")
        print(f"   Range: {ranges['min']:.3f} - {ranges['max']:.3f}")
        print(f"   Optimal: {ranges['optimal']:.3f}")
        print(f"   Mean±SD: {ranges['mean']:.3f} ± {ranges['std']:.3f}\n")

    return benchmarks


def visualize_distributions(clear_df: pd.DataFrame, smash_df: pd.DataFrame):
    """Create visualization of metric distributions"""

    print("\nCreating distribution visualizations...")

    fig, axes = plt.subplots(4, 2, figsize=(14, 16))
    axes = axes.flatten()

    for i, metric in enumerate(KEY_METRICS):
        ax = axes[i]

        if metric in clear_df.columns and metric in smash_df.columns:
            # Plot distributions
            clear_vals = clear_df[metric].dropna()
            smash_vals = smash_df[metric].dropna()

            ax.hist(clear_vals, bins=30, alpha=0.6, label='Clear', color='blue', density=True)
            ax.hist(smash_vals, bins=30, alpha=0.6, label='Smash', color='red', density=True)

            # Mark percentiles
            clear_p25 = np.percentile(clear_vals, 25)
            clear_p75 = np.percentile(clear_vals, 75)
            smash_p25 = np.percentile(smash_vals, 25)
            smash_p75 = np.percentile(smash_vals, 75)

            ax.axvline(clear_p25, color='blue', linestyle='--', alpha=0.5)
            ax.axvline(clear_p75, color='blue', linestyle='--', alpha=0.5)
            ax.axvline(smash_p25, color='red', linestyle='--', alpha=0.5)
            ax.axvline(smash_p75, color='red', linestyle='--', alpha=0.5)

            ax.set_xlabel(metric, fontsize=10)
            ax.set_ylabel('Density', fontsize=10)
            ax.legend(fontsize=9)
            ax.grid(True, alpha=0.3)

    plt.tight_layout()
    save_path = OUTPUT_DIR / "benchmark_distributions.png"
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"✓ Saved visualization to {save_path}")


def generate_benchmark_code(clear_benchmarks: Dict, smash_benchmarks: Dict):
    """Generate Python code for technique_benchmarks.py"""

    print("\n" + "="*70)
    print("GENERATED CODE for technique_benchmarks.py")
    print("="*70 + "\n")

    print("PROFESSIONAL_CLEAR = {")
    for metric, ranges in clear_benchmarks.items():
        print(f"    '{metric}': {{")
        print(f"        'min': {ranges['min']:.4f},")
        print(f"        'max': {ranges['max']:.4f},")
        print(f"        'optimal': {ranges['optimal']:.4f},")
        print(f"        'unit': 'TODO',")
        print(f"        'description': 'TODO'")
        print(f"    }},\n")
    print("}\n")

    print("PROFESSIONAL_SMASH = {")
    for metric, ranges in smash_benchmarks.items():
        print(f"    '{metric}': {{")
        print(f"        'min': {ranges['min']:.4f},")
        print(f"        'max': {ranges['max']:.4f},")
        print(f"        'optimal': {ranges['optimal']:.4f},")
        print(f"        'unit': 'TODO',")
        print(f"        'description': 'TODO'")
        print(f"    }},\n")
    print("}")


def save_benchmarks(clear_benchmarks: Dict, smash_benchmarks: Dict):
    """Save benchmarks to file"""
    output = {
        'Clear': clear_benchmarks,
        'Smash': smash_benchmarks
    }

    save_path = OUTPUT_DIR / "derived_benchmarks.pkl"
    with open(save_path, 'wb') as f:
        pickle.dump(output, f)

    print(f"\n✓ Saved benchmarks to {save_path}")


def main():
    """Main execution"""
    print("\n" + "="*70)
    print("DERIVING PROFESSIONAL TECHNIQUE BENCHMARKS")
    print("="*70)

    # Create output directory
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Derive benchmarks for each stroke type
    clear_benchmarks = derive_benchmarks('Clear')
    smash_benchmarks = derive_benchmarks('Smash')

    # Load full datasets for visualization
    clear_df = load_all_features('Clear')
    smash_df = load_all_features('Smash')

    if len(clear_df) > 0 and len(smash_df) > 0:
        visualize_distributions(clear_df, smash_df)

    # Generate code
    generate_benchmark_code(clear_benchmarks, smash_benchmarks)

    # Save benchmarks
    save_benchmarks(clear_benchmarks, smash_benchmarks)

    print("\n" + "="*70)
    print("NEXT STEPS:")
    print("="*70)
    print("1. Review the generated code above")
    print("2. Copy it into src/coaching/technique_benchmarks.py")
    print("3. Add proper 'unit' and 'description' fields")
    print("4. Review benchmark_distributions.png to verify ranges make sense")
    print("\n")


if __name__ == "__main__":
    main()
