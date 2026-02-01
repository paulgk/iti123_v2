#!/usr/bin/env python3
"""
Create metadata.csv from existing pose files

Use this when extraction completed but metadata.csv wasn't created
(e.g., process was interrupted)

Usage:
    python scripts/create_metadata_from_poses.py

Output:
    - data/metadata.csv
"""

import os
import pickle
from pathlib import Path
import pandas as pd
import numpy as np
from tqdm import tqdm


def infer_stroke_type_from_path(video_path):
    """Infer stroke type from video path"""
    path_parts = Path(video_path).parts

    for part in path_parts:
        part_lower = part.lower()
        if 'clear' in part_lower:
            return 'clear'
        elif 'smash' in part_lower:
            return 'smash'
        elif 'drop' in part_lower:
            return 'drop'
        elif 'lift' in part_lower:
            return 'lift'

    return 'unknown'


def infer_player_id_from_path(video_path):
    """Infer player ID from filename"""
    filename = Path(video_path).stem

    # Remove _pose suffix if present
    filename = filename.replace('_pose', '')

    # Try to extract player ID
    parts = filename.split('_')
    for part in parts:
        if part.startswith('p') or part.startswith('player'):
            return part

    # Default: use first part of filename
    return parts[0] if parts else filename


def create_metadata_from_poses(pose_dir='data/processed/poses',
                               output_path='data/metadata.csv'):
    """
    Create metadata.csv from existing pose files

    Args:
        pose_dir: Directory containing .pkl pose files
        output_path: Path to save metadata.csv
    """
    pose_dir = Path(pose_dir)

    if not pose_dir.exists():
        print(f"❌ Pose directory not found: {pose_dir}")
        return

    # Find all pose files
    pose_files = list(pose_dir.glob('*.pkl'))

    if len(pose_files) == 0:
        print(f"❌ No .pkl files found in {pose_dir}")
        return

    print(f"Found {len(pose_files)} pose files")
    print(f"Creating metadata...\n")

    metadata_records = []
    failed = []

    for pose_file in tqdm(pose_files, desc="Processing poses"):
        try:
            # Load pose to get shape info
            with open(pose_file, 'rb') as f:
                pose_data = pickle.load(f)

            # Get video name (remove _pose.pkl suffix)
            video_name = pose_file.stem.replace('_pose', '')

            # Infer labels from filename/path
            # Try to find original video path from pose filename
            stroke_type = infer_stroke_type_from_path(pose_file)
            player_id = infer_player_id_from_path(pose_file)

            # Create metadata record
            record = {
                'video_id': video_name,
                'video_path': f"data/videos/clips/{stroke_type}/{video_name}.mp4",  # Inferred
                'pose_file': str(pose_file.relative_to(pose_dir.parent.parent)),
                'stroke_type': stroke_type,
                'player_id': player_id,
                'extracted_frames': pose_data.shape[0],
                'shape': str(pose_data.shape)
            }

            metadata_records.append(record)

        except Exception as e:
            failed.append({
                'pose_file': str(pose_file),
                'error': str(e)
            })

    # Create DataFrame
    df = pd.DataFrame(metadata_records)

    # Save to CSV
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)

    print(f"\n{'='*60}")
    print(f"METADATA CREATION COMPLETE")
    print(f"{'='*60}")
    print(f"Total pose files: {len(pose_files)}")
    print(f"Successfully processed: {len(metadata_records)}")
    print(f"Failed: {len(failed)}")

    if len(failed) > 0:
        print(f"\n⚠️  Failed files:")
        for item in failed[:10]:
            print(f"  {item['pose_file']}: {item['error']}")
        if len(failed) > 10:
            print(f"  ... and {len(failed)-10} more")

    print(f"\n{'='*60}")
    print(f"METADATA SUMMARY")
    print(f"{'='*60}")
    print(f"\nStroke type distribution:")
    print(df['stroke_type'].value_counts())

    print(f"\nPlayer distribution (top 10):")
    print(df['player_id'].value_counts().head(10))

    # Check for unknown types
    unknown = df[df['stroke_type'] == 'unknown']
    if len(unknown) > 0:
        print(f"\n⚠️  {len(unknown)} videos with unknown stroke type")
        print(f"You may need to manually label these in {output_path}")
        print(f"\nSample unknown files:")
        print(unknown[['video_id', 'pose_file']].head(10))

    print(f"\n✓ Metadata saved to: {output_path}")
    print(f"\nNext steps:")
    print(f"  1. Review metadata: cat {output_path} | head -20")
    print(f"  2. Fix unknown stroke types if needed")
    print(f"  3. Run validation: python scripts/validate_phase2.py --sample-size 100")
    print()


def main():
    import argparse

    parser = argparse.ArgumentParser(
        description='Create metadata.csv from existing pose files',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    parser.add_argument('--pose-dir', type=str, default='data/processed/poses',
                       help='Directory containing pose .pkl files')
    parser.add_argument('--output', type=str, default='data/metadata.csv',
                       help='Output path for metadata.csv')

    args = parser.parse_args()

    create_metadata_from_poses(args.pose_dir, args.output)


if __name__ == '__main__':
    main()
