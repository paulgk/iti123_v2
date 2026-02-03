#!/usr/bin/env python3
"""
Create metadata.csv from ShuttleSet annotations for ROI-based extraction.

This script generates a metadata CSV file that includes player positions,
which is required for ROI-based pose extraction.

Usage:
    python scripts/create_metadata_csv.py --shuttleset ShuttleSet --clips data/clips --output data/metadata.csv
"""

import argparse
import csv
import sys
from pathlib import Path
from collections import Counter


# Shot type mapping (must match extract_shuttleset_clips.py)
SHOT_TYPE_MAPPING = {
    'Smash': 'Smash',
    'Steep_Smash': 'Smash',
    'Clear': 'Clear',
    'Clear (Long)': 'Clear',
    'Drop': 'Drop',
    'Drop Shot (Soft)': 'Drop',
    'Lift': 'Lift',
    'Defensive_Lift': 'Lift',
    'Drive': 'Drive',
    'Drive / Flat Shot': 'Drive',
}


def load_match_metadata(shuttleset_dir: Path) -> dict:
    """Load match metadata from match.csv"""
    match_csv = shuttleset_dir / 'match.csv'

    if not match_csv.exists():
        raise FileNotFoundError(f"Match CSV not found: {match_csv}")

    matches = {}

    with open(match_csv, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            match_id = int(row.get('match_id', 0))
            matches[match_id] = {
                'video_name': row.get('name', ''),
                'tournament': row.get('tournament', ''),
                'round': row.get('round', ''),
                'winner': row.get('winner', ''),
                'loser': row.get('loser', ''),
            }

    return matches


def create_metadata(
    shuttleset_dir: Path,
    clips_dir: Path,
    output_path: Path
) -> None:
    """
    Create metadata.csv with player positions for ROI extraction.

    Args:
        shuttleset_dir: ShuttleSet dataset directory
        clips_dir: Directory containing extracted clips
        output_path: Path to save metadata.csv
    """
    print("=" * 80)
    print("CREATING METADATA CSV")
    print("=" * 80)
    print(f"ShuttleSet:  {shuttleset_dir}")
    print(f"Clips:       {clips_dir}")
    print(f"Output:      {output_path}")
    print("=" * 80)
    print()

    # Load match metadata
    print("Loading match metadata...")
    matches = load_match_metadata(shuttleset_dir)
    print(f"✓ Loaded {len(matches)} matches")
    print()

    # Prepare metadata rows
    metadata_rows = []
    clips_found = 0
    clips_missing = 0
    shot_counts = Counter()

    print("Processing annotations...")

    for match_id, match_info in sorted(matches.items()):
        video_name = match_info['video_name']
        match_dir = shuttleset_dir / 'set' / video_name

        if not match_dir.exists():
            print(f"  ⚠️  Match directory not found: {match_dir}")
            continue

        # Load all set CSV files
        for set_csv in sorted(match_dir.glob('set*.csv')):
            set_num = int(set_csv.stem.replace('set', ''))

            with open(set_csv, 'r', encoding='utf-8') as f:
                reader = csv.DictReader(f)

                for row in reader:
                    shot_type = row.get('type', '').strip()
                    time_str = row.get('time', '').strip()

                    if not shot_type or not time_str:
                        continue

                    # Map to target class
                    target_class = SHOT_TYPE_MAPPING.get(shot_type, None)

                    if target_class is None:
                        continue  # Skip excluded shots

                    # Get player positions
                    player = row.get('player', '').strip()
                    player_x = row.get('player_location_x', '').strip()
                    player_y = row.get('player_location_y', '').strip()

                    if not player or not player_x or not player_y:
                        continue  # Skip if missing player data

                    # Generate video ID and clip path
                    rally = int(row.get('rally', 0))
                    ball_round = int(float(row.get('ball_round', 0)))
                    frame_num = float(row.get('frame_num', 0))

                    video_id = (
                        f"{match_id:02d}_"
                        f"set{set_num}_"
                        f"rally{rally:02d}_"
                        f"ball{ball_round:02d}_"
                        f"{target_class}"
                    )

                    clip_path = clips_dir / target_class / f"{video_id}.mp4"
                    pose_path = Path('data/poses') / f"{video_id}.pkl"

                    # Check if clip exists
                    if clip_path.exists():
                        clips_found += 1
                        shot_counts[target_class] += 1

                        metadata_rows.append({
                            'video_id': video_id,
                            'match_id': match_id,
                            'set_num': set_num,
                            'rally': rally,
                            'ball_round': ball_round,
                            'shot_type': target_class,
                            'original_type': shot_type,
                            'player': player,
                            'player_x': float(player_x),
                            'player_y': float(player_y),
                            'frame_num': int(frame_num),
                            'clip_path': str(clip_path),
                            'pose_path': str(pose_path),
                        })
                    else:
                        clips_missing += 1

    print(f"✓ Processed annotations")
    print(f"  Clips found:   {clips_found}")
    print(f"  Clips missing: {clips_missing}")
    print()

    # Write metadata.csv
    print(f"Writing metadata to {output_path}...")

    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, 'w', newline='', encoding='utf-8') as f:
        fieldnames = [
            'video_id',
            'match_id',
            'set_num',
            'rally',
            'ball_round',
            'shot_type',
            'original_type',
            'player',
            'player_x',
            'player_y',
            'frame_num',
            'clip_path',
            'pose_path',
        ]

        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(metadata_rows)

    print(f"✓ Saved {len(metadata_rows)} entries to {output_path}")
    print()

    # Summary
    print("=" * 80)
    print("METADATA SUMMARY")
    print("=" * 80)
    print(f"Total clips:  {len(metadata_rows)}")
    print()
    print("Clips by shot type:")
    for shot_type in ['Smash', 'Clear', 'Drop', 'Lift', 'Drive']:
        if shot_type in shot_counts:
            count = shot_counts[shot_type]
            pct = count / len(metadata_rows) * 100 if len(metadata_rows) > 0 else 0
            print(f"  {shot_type:<10} {count:>6} ({pct:>5.1f}%)")
    print()
    print(f"Output: {output_path}")
    print("=" * 80)


def main():
    parser = argparse.ArgumentParser(
        description='Create metadata.csv with player positions for ROI extraction',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Create metadata from extracted clips
  python scripts/create_metadata_csv.py --shuttleset ShuttleSet --clips data/clips --output data/metadata.csv

  # After running extract_shuttleset_clips.py
  python scripts/create_metadata_csv.py
        """
    )

    parser.add_argument(
        '--shuttleset',
        type=Path,
        default=Path('ShuttleSet'),
        help='ShuttleSet dataset directory (default: ShuttleSet)'
    )

    parser.add_argument(
        '--clips',
        type=Path,
        default=Path('data/clips'),
        help='Directory containing extracted clips (default: data/clips)'
    )

    parser.add_argument(
        '--output',
        type=Path,
        default=Path('data/metadata.csv'),
        help='Output metadata CSV path (default: data/metadata.csv)'
    )

    args = parser.parse_args()

    # Validate inputs
    if not args.shuttleset.exists():
        print(f"Error: ShuttleSet directory not found: {args.shuttleset}")
        return 1

    if not args.clips.exists():
        print(f"Error: Clips directory not found: {args.clips}")
        return 1

    # Create metadata
    try:
        create_metadata(args.shuttleset, args.clips, args.output)
        return 0
    except Exception as e:
        print(f"Error: {e}")
        return 1


if __name__ == '__main__':
    sys.exit(main())
