#!/usr/bin/env python3
"""
Pose Quality Analysis Script
=============================
Analyze pose extraction quality and identify problematic matches.

Provides detailed diagnostics for improving extraction quality.

Author: ITI123 Project
Date: January 2026
"""

import pandas as pd
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns

# Paths
BASE_DIR = Path(__file__).resolve().parents[2]
REPORTS_DIR = BASE_DIR / "outputs" / "reports"
LOG_FILE = REPORTS_DIR / "pose_extraction_log.csv"

# Create visualizations directory
VIZ_DIR = BASE_DIR / "outputs" / "visualizations"
VIZ_DIR.mkdir(parents=True, exist_ok=True)


def analyze_quality():
    """Analyze pose extraction quality"""

    print("=" * 70)
    print("POSE EXTRACTION QUALITY ANALYSIS")
    print("=" * 70)
    print()

    # Load log
    df = pd.read_csv(LOG_FILE)

    print(f"Total clips analyzed: {len(df)}")
    print()

    # ==========================================================================
    # OVERALL QUALITY DISTRIBUTION
    # ==========================================================================
    print("Quality Distribution:")
    print(f"  Excellent (>= 90% valid frames): {len(df[df['valid_percentage'] >= 90])}")
    print(f"  Good (80-90% valid frames):      {len(df[(df['valid_percentage'] >= 80) & (df['valid_percentage'] < 90)])}")
    print(f"  Fair (50-80% valid frames):      {len(df[(df['valid_percentage'] >= 50) & (df['valid_percentage'] < 80)])}")
    print(f"  Poor (< 50% valid frames):       {len(df[df['valid_percentage'] < 50])}")
    print()

    # ==========================================================================
    # LOW QUALITY ANALYSIS
    # ==========================================================================
    low_quality = df[df['valid_percentage'] < 50].copy()

    if len(low_quality) > 0:
        print(f"⚠️  LOW QUALITY CLIPS: {len(low_quality)} ({len(low_quality)/len(df)*100:.1f}%)")
        print()

        # Extract match ID
        low_quality['match_id'] = low_quality['clip_name'].str.extract(r'^(\d+)_')[0].astype(int)

        # Group by match
        match_quality = low_quality.groupby('match_id').agg({
            'clip_name': 'count',
            'valid_percentage': 'mean',
            'avg_confidence': 'mean'
        }).rename(columns={'clip_name': 'low_quality_count'})

        match_quality = match_quality.sort_values('low_quality_count', ascending=False)

        print("Matches with most low-quality clips:")
        print()
        print(match_quality.head(15).to_string())
        print()

        # =======================================================================
        # PROBLEM DIAGNOSIS
        # =======================================================================
        print("=" * 70)
        print("PROBLEM DIAGNOSIS")
        print("=" * 70)
        print()

        # Very low detection rate (<10% valid frames)
        very_low = low_quality[low_quality['valid_percentage'] < 10]
        if len(very_low) > 0:
            print(f"⚠️  CRITICAL: {len(very_low)} clips with < 10% valid frames")
            print("   Possible causes:")
            print("   - Camera angle issues (overhead/side view)")
            print("   - Poor lighting conditions")
            print("   - Extreme motion blur")
            print("   - Players too small in frame")
            print("   - Heavy occlusions")
            print()
            print("   Worst clips:")
            worst = very_low.nsmallest(10, 'valid_percentage')
            print(worst[['clip_name', 'valid_percentage', 'avg_confidence']].to_string())
            print()

        # Low confidence (<0.5 average)
        low_conf = low_quality[low_quality['avg_confidence'] < 0.5]
        if len(low_conf) > 0:
            print(f"⚠️  {len(low_conf)} clips with low average confidence (<0.5)")
            print("   Possible causes:")
            print("   - Poor video quality")
            print("   - Compressed/low resolution video")
            print("   - Motion blur")
            print()

        # =======================================================================
        # RECOMMENDATIONS
        # =======================================================================
        print("=" * 70)
        print("RECOMMENDATIONS")
        print("=" * 70)
        print()

        print("For matches with many low-quality clips:")
        print()

        # Top problematic matches
        top_problem_matches = match_quality.head(5).index.tolist()

        for match_id in top_problem_matches:
            match_clips = low_quality[low_quality['match_id'] == match_id]
            avg_valid = match_clips['valid_percentage'].mean()
            avg_conf = match_clips['avg_confidence'].mean()

            print(f"Match {match_id:02d}:")
            print(f"  Low-quality clips: {len(match_clips)}")
            print(f"  Average valid frames: {avg_valid:.1f}%")
            print(f"  Average confidence: {avg_conf:.3f}")
            print()

            # Diagnosis
            if avg_valid < 10:
                print(f"  ❌ CRITICAL - Very low detection rate")
                print(f"     Recommendation: Check video manually for:")
                print(f"     - Unusual camera angles")
                print(f"     - Very poor lighting")
                print(f"     - Players too far from camera")
            elif avg_conf < 0.4:
                print(f"  ⚠️  Low confidence - poor video quality")
                print(f"     Recommendation: Try robust extraction with preprocessing")
            else:
                print(f"  ⚠️  Moderate quality issues")
                print(f"     Recommendation: Try robust extraction with lower thresholds")
            print()

        # =======================================================================
        # SOLUTION STRATEGIES
        # =======================================================================
        print("=" * 70)
        print("SOLUTION STRATEGIES")
        print("=" * 70)
        print()

        print("Run robust pose extraction to improve quality:")
        print("  python src/data_processing/extract_poses_robust.py")
        print()
        print("This will:")
        print("  1. Use lower confidence thresholds")
        print("  2. Apply frame preprocessing (CLAHE, brightness adjustment)")
        print("  3. Try multiple detection attempts")
        print()
        print("Expected improvements:")
        print("  - Clips with 30-50% valid frames → 60-80% valid frames")
        print("  - Clips with 10-30% valid frames → 40-60% valid frames")
        print("  - Clips with < 10% valid frames → May need manual review")
        print()

    else:
        print("✅ All clips have good quality (>= 50% valid frames)!")
        print()

    # ==========================================================================
    # SAVE DETAILED ANALYSIS
    # ==========================================================================

    # Match-level statistics
    df['match_id'] = df['clip_name'].str.extract(r'^(\d+)_')[0].astype(int)

    match_stats = df.groupby('match_id').agg({
        'clip_name': 'count',
        'valid_percentage': ['mean', 'min', 'max'],
        'avg_confidence': 'mean'
    })

    match_stats.columns = ['total_clips', 'avg_valid_pct', 'min_valid_pct',
                           'max_valid_pct', 'avg_confidence']

    # Count low quality clips per match
    low_quality_counts = df[df['valid_percentage'] < 50].groupby('match_id').size()
    match_stats['low_quality_count'] = low_quality_counts
    match_stats['low_quality_count'] = match_stats['low_quality_count'].fillna(0).astype(int)

    # Save match statistics
    match_stats_file = REPORTS_DIR / "pose_quality_by_match.csv"
    match_stats.to_csv(match_stats_file)

    print(f"📊 Match-level statistics saved to: {match_stats_file}")
    print()

    # ==========================================================================
    # CREATE VISUALIZATIONS
    # ==========================================================================
    print("Creating visualizations...")

    # Plot 1: Valid percentage distribution
    plt.figure(figsize=(12, 6))
    plt.hist(df['valid_percentage'], bins=50, edgecolor='black', alpha=0.7)
    plt.axvline(x=50, color='r', linestyle='--', label='Minimum threshold (50%)')
    plt.xlabel('Valid Frame Percentage (%)')
    plt.ylabel('Number of Clips')
    plt.title('Distribution of Pose Detection Quality')
    plt.legend()
    plt.grid(axis='y', alpha=0.3)
    plt.savefig(VIZ_DIR / 'pose_quality_distribution.png', dpi=150, bbox_inches='tight')
    plt.close()

    # Plot 2: Match-level quality
    plt.figure(figsize=(14, 6))
    match_stats_sorted = match_stats.sort_values('avg_valid_pct')

    colors = ['red' if x < 50 else 'orange' if x < 80 else 'green'
              for x in match_stats_sorted['avg_valid_pct']]

    plt.bar(range(len(match_stats_sorted)), match_stats_sorted['avg_valid_pct'],
            color=colors, alpha=0.7, edgecolor='black')
    plt.axhline(y=50, color='r', linestyle='--', label='Minimum threshold')
    plt.axhline(y=80, color='orange', linestyle='--', label='Good quality')
    plt.xlabel('Match ID')
    plt.ylabel('Average Valid Frame Percentage (%)')
    plt.title('Pose Detection Quality by Match')
    plt.xticks(range(len(match_stats_sorted)), match_stats_sorted.index, rotation=90)
    plt.legend()
    plt.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    plt.savefig(VIZ_DIR / 'quality_by_match.png', dpi=150, bbox_inches='tight')
    plt.close()

    print(f"📊 Visualizations saved to: {VIZ_DIR}")
    print()


def main():
    """Main function"""
    print("\n")
    print("*" * 70)
    print("*" + " " * 68 + "*")
    print("*" + "  ITI123 Project: Pose Quality Analysis".center(68) + "*")
    print("*" + " " * 68 + "*")
    print("*" * 70)
    print("\n")

    analyze_quality()

    print("=" * 70)
    print("✅ ANALYSIS COMPLETE!")
    print("=" * 70)
    print()


if __name__ == "__main__":
    main()
