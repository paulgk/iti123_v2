#!/usr/bin/env python3
"""
Validate ROI-extracted poses for quality checks.

This script analyzes extracted pose files and checks for:
- Multi-player detection (x-range >60%)
- Short sequences (<30 frames)
- Data quality (mean, std, coordinate ranges)
- Missing or corrupted files

Usage:
    python scripts/validate_roi_poses.py --poses data/poses_test
    python scripts/validate_roi_poses.py --poses data/poses --metadata data/metadata.csv
"""

import argparse
import csv
import pickle
import sys
from pathlib import Path
from collections import Counter

import numpy as np
from tqdm import tqdm


def load_pose(pose_file: Path) -> np.ndarray:
    """Load pose from pickle file"""
    try:
        with open(pose_file, 'rb') as f:
            return pickle.load(f)
    except Exception as e:
        raise ValueError(f"Error loading {pose_file.name}: {e}")


def check_multi_player(pose: np.ndarray, threshold: float = 0.6) -> bool:
    """Check if pose likely contains multiple players"""
    x_coords = pose[:, :, 0]
    x_range = np.max(x_coords) - np.min(x_coords)
    return x_range > threshold


def analyze_pose(pose: np.ndarray) -> dict:
    """Analyze pose quality metrics"""
    return {
        'num_frames': len(pose),
        'shape': pose.shape,
        'mean': float(pose.mean()),
        'std': float(pose.std()),
        'x_range': float(pose[:, :, 0].max() - pose[:, :, 0].min()),
        'y_range': float(pose[:, :, 1].max() - pose[:, :, 1].min()),
        'z_range': float(pose[:, :, 2].max() - pose[:, :, 2].min()),
        'has_nan': bool(np.isnan(pose).any()),
        'has_inf': bool(np.isinf(pose).any()),
    }


def validate_poses(
    poses_dir: Path,
    metadata_file: Path = None,
    multi_player_threshold: float = 0.6,
    min_frames: int = 30
) -> None:
    """
    Validate extracted poses.

    Args:
        poses_dir: Directory containing pose .pkl files
        metadata_file: Optional metadata CSV for comparison
        multi_player_threshold: X-range threshold for multi-player detection
        min_frames: Minimum frames for valid sequence
    """
    print("=" * 80)
    print("ROI POSE VALIDATION")
    print("=" * 80)
    print(f"Poses dir:     {poses_dir}")
    if metadata_file:
        print(f"Metadata:      {metadata_file}")
    print(f"Multi-player:  x-range > {multi_player_threshold}")
    print(f"Min frames:    {min_frames}")
    print("=" * 80)
    print()

    # Find pose files
    pose_files = list(poses_dir.glob('*.pkl'))

    if len(pose_files) == 0:
        print(f"❌ No pose files found in {poses_dir}")
        return

    print(f"Found {len(pose_files)} pose files")
    print()

    # Load metadata if provided
    metadata_ids = set()
    if metadata_file and metadata_file.exists():
        with open(metadata_file, 'r') as f:
            reader = csv.DictReader(f)
            for row in reader:
                metadata_ids.add(row['video_id'])

        print(f"Metadata entries: {len(metadata_ids)}")
        print(f"Pose files:       {len(pose_files)}")
        print(f"Success rate:     {len(pose_files) / len(metadata_ids) * 100:.1f}%")
        print()

    # Analyze poses
    print("Analyzing poses...")
    print()

    valid_count = 0
    multi_player_count = 0
    short_sequence_count = 0
    corrupted_count = 0

    stats = {
        'num_frames': [],
        'x_range': [],
        'y_range': [],
        'mean': [],
        'std': [],
    }

    issues = {
        'multi_player': [],
        'short_sequence': [],
        'corrupted': [],
    }

    for pose_file in tqdm(pose_files, desc="Validating"):
        try:
            pose = load_pose(pose_file)

            # Check shape
            if len(pose.shape) != 3 or pose.shape[1] != 33 or pose.shape[2] != 3:
                corrupted_count += 1
                issues['corrupted'].append({
                    'file': pose_file.name,
                    'reason': f"Invalid shape: {pose.shape}"
                })
                continue

            # Analyze
            analysis = analyze_pose(pose)

            # Check for issues
            is_multi_player = check_multi_player(pose, multi_player_threshold)
            is_short = analysis['num_frames'] < min_frames

            if is_multi_player:
                multi_player_count += 1
                issues['multi_player'].append({
                    'file': pose_file.name,
                    'x_range': analysis['x_range'],
                    'frames': analysis['num_frames']
                })
            elif is_short:
                short_sequence_count += 1
                issues['short_sequence'].append({
                    'file': pose_file.name,
                    'frames': analysis['num_frames']
                })
            else:
                valid_count += 1

            # Collect stats
            stats['num_frames'].append(analysis['num_frames'])
            stats['x_range'].append(analysis['x_range'])
            stats['y_range'].append(analysis['y_range'])
            stats['mean'].append(analysis['mean'])
            stats['std'].append(analysis['std'])

        except Exception as e:
            corrupted_count += 1
            issues['corrupted'].append({
                'file': pose_file.name,
                'reason': str(e)
            })

    # Summary
    print()
    print("=" * 80)
    print("VALIDATION SUMMARY")
    print("=" * 80)
    print()

    total = len(pose_files)
    print(f"Total poses:       {total}")
    print(f"Valid:             {valid_count} ({valid_count/total*100:.1f}%)")
    print(f"Multi-player:      {multi_player_count} ({multi_player_count/total*100:.1f}%)")
    print(f"Short sequences:   {short_sequence_count} ({short_sequence_count/total*100:.1f}%)")
    print(f"Corrupted:         {corrupted_count} ({corrupted_count/total*100:.1f}%)")
    print()

    # Statistics
    if len(stats['num_frames']) > 0:
        print("POSE STATISTICS")
        print("-" * 80)
        print(f"{'Metric':<20} {'Mean':<12} {'Std':<12} {'Min':<12} {'Max':<12}")
        print("-" * 80)

        for metric in ['num_frames', 'x_range', 'y_range', 'mean', 'std']:
            values = stats[metric]
            print(f"{metric:<20} {np.mean(values):<12.4f} {np.std(values):<12.4f} {np.min(values):<12.4f} {np.max(values):<12.4f}")

        print()

    # Issue details
    if multi_player_count > 0:
        print("MULTI-PLAYER DETECTIONS (showing first 10)")
        print("-" * 80)
        for issue in issues['multi_player'][:10]:
            print(f"  {issue['file']:<50} x-range={issue['x_range']:.3f} frames={issue['frames']}")
        if len(issues['multi_player']) > 10:
            print(f"  ... and {len(issues['multi_player']) - 10} more")
        print()

    if short_sequence_count > 0:
        print("SHORT SEQUENCES (showing first 10)")
        print("-" * 80)
        for issue in issues['short_sequence'][:10]:
            print(f"  {issue['file']:<50} frames={issue['frames']}")
        if len(issues['short_sequence']) > 10:
            print(f"  ... and {len(issues['short_sequence']) - 10} more")
        print()

    if corrupted_count > 0:
        print("CORRUPTED FILES (showing first 10)")
        print("-" * 80)
        for issue in issues['corrupted'][:10]:
            print(f"  {issue['file']:<50} {issue['reason']}")
        if len(issues['corrupted']) > 10:
            print(f"  ... and {len(issues['corrupted']) - 10} more")
        print()

    # Assessment
    print("=" * 80)
    print("ASSESSMENT")
    print("=" * 80)
    print()

    if multi_player_count == 0:
        print("✓ No multi-player detections - ROI working correctly!")
    elif multi_player_count / total < 0.05:
        print(f"⚠️  {multi_player_count/total*100:.1f}% multi-player detections")
        print("   Acceptable (<5%), but ROI could be tighter")
    else:
        print(f"❌ {multi_player_count/total*100:.1f}% multi-player detections")
        print("   ROI may be too large or player positions inaccurate")

    print()

    if short_sequence_count / total < 0.20:
        print(f"✓ {short_sequence_count/total*100:.1f}% short sequences (<{min_frames} frames)")
        print("   Acceptable - will be filtered during training")
    else:
        print(f"⚠️  {short_sequence_count/total*100:.1f}% short sequences")
        print("   Many clips shorter than minimum - check clip extraction")

    print()

    if corrupted_count == 0:
        print("✓ No corrupted files")
    else:
        print(f"⚠️  {corrupted_count} corrupted files")
        print("   Check pose extraction logs for errors")

    print()

    # Recommendations
    expected_usable = valid_count
    print("EXPECTED TRAINING DATA")
    print("-" * 80)
    print(f"Usable samples:    {expected_usable} (after filtering)")
    print(f"Filtered out:      {multi_player_count + short_sequence_count} (multi-player + short)")
    print()

    if expected_usable < 10000:
        print("⚠️  Warning: Less than 10K usable samples")
        print("   Consider:")
        print("   - Extracting more matches")
        print("   - Reducing ROI size if multi-player rate is high")
        print("   - Lowering min_frames threshold (if sequences are consistently short)")
    elif expected_usable < 15000:
        print("✓ Good amount of training data (10-15K samples)")
    else:
        print("✓ Excellent amount of training data (>15K samples)")

    print()


def main():
    parser = argparse.ArgumentParser(
        description='Validate ROI-extracted poses',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Validate test poses
  python scripts/validate_roi_poses.py --poses data/poses_test

  # Validate with metadata comparison
  python scripts/validate_roi_poses.py --poses data/poses --metadata data/metadata.csv

  # Custom thresholds
  python scripts/validate_roi_poses.py --poses data/poses --multi-player-threshold 0.5 --min-frames 20
        """
    )

    parser.add_argument(
        '--poses',
        type=Path,
        required=True,
        help='Directory containing pose .pkl files'
    )

    parser.add_argument(
        '--metadata',
        type=Path,
        help='Optional metadata CSV for comparison'
    )

    parser.add_argument(
        '--multi-player-threshold',
        type=float,
        default=0.6,
        help='X-range threshold for multi-player detection (default: 0.6)'
    )

    parser.add_argument(
        '--min-frames',
        type=int,
        default=30,
        help='Minimum frames for valid sequence (default: 30)'
    )

    args = parser.parse_args()

    # Validate inputs
    if not args.poses.exists():
        print(f"Error: Poses directory not found: {args.poses}")
        return 1

    if args.metadata and not args.metadata.exists():
        print(f"Error: Metadata file not found: {args.metadata}")
        return 1

    # Validate
    validate_poses(
        args.poses,
        args.metadata,
        args.multi_player_threshold,
        args.min_frames
    )

    return 0


if __name__ == '__main__':
    sys.exit(main())
