#!/usr/bin/env python3
"""
Extract badminton shot clips from ShuttleSet videos using CSV annotations.

This script extracts video clips for SMASH, CLEAR, DROP, LIFT, and DRIVE shots
from raw match videos based on timestamps in ShuttleSet CSV files.

Usage:
    python scripts/extract_shuttleset_clips.py --input data/raw_videos --output data/clips --dry-run
    python scripts/extract_shuttleset_clips.py --input data/raw_videos --output data/clips --execute
    python scripts/extract_shuttleset_clips.py --shot-types Smash Clear --match-ids 01 02 03
"""

import argparse
import csv
import subprocess
import sys
from pathlib import Path
from typing import Dict, List, Set, Tuple
from collections import Counter, defaultdict


# Shot type mapping from ShuttleSet labels to our target classes
SHOT_TYPE_MAPPING = {
    # SMASH class (overhead offensive)
    'Smash': 'Smash',
    'Steep_Smash': 'Smash',  # Merge steep smash with standard smash

    # CLEAR class (overhead defensive - high contact point)
    'Clear': 'Clear',
    'Clear (Long)': 'Clear',

    # DROP class (overhead offensive - deceptive)
    'Drop': 'Drop',
    'Drop Shot (Soft)': 'Drop',
    'Slice_Drop': 'Drop',  # Merge slice drop (pose estimation can't distinguish)
    'Slice Drop / Cut Drop': 'Drop',

    # LIFT class (defensive - low contact point)
    'Lift': 'Lift',
    'Lift / Clear (Defensive)': 'Lift',
    'Defensive_Lift': 'Lift',
    'Defensive Lift Return': 'Lift',

    # DRIVE class (mid-court attack)
    'Drive': 'Drive',
    'Drive / Flat Shot': 'Drive',
    'Rear_Drive': 'Drive',  # Merge rear-court drive
    'Rear-Court Drive': 'Drive',
    'Defensive_Drive': 'Drive',  # Merge defensive drive
    'Defensive Drive Return': 'Drive',
    'Push': 'Drive',  # Merge push with drive (weak biomechanical distinction)
    'Push Shot': 'Drive',
    'Short_Drive': 'Drive',

    # Excluded shot types (not in target classes)
    'Block': None,  # Net shot - Phase 5+
    'Block (Net)': None,
    'Net_Kill': None,  # Net shot - Phase 5+
    'Rush / Kill (Net)': None,
    'Cross_Net': None,  # Net shot - Phase 6+
    'Cross-Court Net Shot': None,
    'Short_Serve': None,  # Service - out of scope
    'Long_Serve': None,  # Service - out of scope
    'Overhead_Slice': None,  # Requires IMU sensors
    'Overhead Slice / Cross-Court Drop': None,
    'Unknown': None,  # Unknown shots
    'Unknown Shot': None,
}


def parse_timestamp(time_str: str) -> float:
    """
    Convert timestamp string to seconds.

    Args:
        time_str: Timestamp in format "HH:MM:SS" or "H:MM:SS" or "MM:SS"

    Returns:
        Total seconds as float

    Examples:
        "0:07:39" -> 459.0
        "00:07:43" -> 463.0
        "1:23:45" -> 5025.0
    """
    parts = time_str.strip().split(':')

    if len(parts) == 3:
        hours, minutes, seconds = parts
        return int(hours) * 3600 + int(minutes) * 60 + float(seconds)
    elif len(parts) == 2:
        minutes, seconds = parts
        return int(minutes) * 60 + float(seconds)
    else:
        raise ValueError(f"Invalid timestamp format: {time_str}")


def get_match_metadata(shuttleset_dir: Path) -> Dict[int, Dict]:
    """
    Load match metadata from match.csv.

    Returns:
        Dictionary mapping match ID to metadata (video name, tournament, etc.)
    """
    match_csv = shuttleset_dir / 'set' / 'match.csv'

    if not match_csv.exists():
        raise FileNotFoundError(f"Match CSV not found: {match_csv}")

    matches = {}
    with open(match_csv, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            match_id = int(row['id'])
            matches[match_id] = {
                'id': match_id,
                'video_name': row['video'],
                'tournament': row['tournament'],
                'round': row['round'],
                'year': row['year'],
                'sets': int(row['set']),
                'duration': int(row['duration']),
                'winner': row['winner'],
                'loser': row['loser'],
            }

    return matches


def load_shot_annotations(shuttleset_dir: Path, match_id: int, video_name: str) -> List[Dict]:
    """
    Load shot annotations from set CSV files for a specific match.

    Args:
        shuttleset_dir: ShuttleSet root directory
        match_id: Match ID from match.csv
        video_name: Video name (directory name for set CSVs)

    Returns:
        List of shot annotations with timestamp, type, rally info, etc.
    """
    match_dir = shuttleset_dir / 'set' / video_name

    if not match_dir.exists():
        print(f"⚠️  Warning: Match directory not found: {match_dir}")
        return []

    shots = []

    # Load all set CSV files (set1.csv, set2.csv, set3.csv)
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

                shots.append({
                    'match_id': match_id,
                    'set_num': set_num,
                    'rally': int(row.get('rally', 0)),
                    'ball_round': float(row.get('ball_round', 0)),
                    'timestamp': time_str,
                    'timestamp_seconds': parse_timestamp(time_str),
                    'frame_num': float(row.get('frame_num', 0)),
                    'original_type': shot_type,
                    'target_class': target_class,
                    'player': row.get('player', ''),
                    'backhand': row.get('backhand', ''),
                })

    return shots


def extract_clip(
    input_video: Path,
    output_path: Path,
    start_time: float,
    duration: float = 3.0,
    dry_run: bool = False
) -> bool:
    """
    Extract a video clip using ffmpeg.

    Args:
        input_video: Path to source video
        output_path: Path to output clip
        start_time: Start time in seconds
        duration: Clip duration in seconds (default: 3.0)
        dry_run: If True, only print command without executing

    Returns:
        True if extraction successful, False otherwise
    """
    # Ensure output directory exists
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Build ffmpeg command
    # -ss: start time (before -i for faster seek)
    # -i: input file
    # -t: duration
    # -c copy: copy codec (fast, no re-encoding) - if compatible
    # -c:v libx264 -c:a aac: re-encode if copy fails (compatibility)
    # -avoid_negative_ts make_zero: fix timestamp issues

    # Try fast copy first, fallback to re-encode if needed
    cmd = [
        'ffmpeg',
        '-y',  # Overwrite output file
        '-ss', f'{start_time:.3f}',
        '-i', str(input_video),
        '-t', f'{duration:.3f}',
        '-c', 'copy',
        '-avoid_negative_ts', 'make_zero',
        '-loglevel', 'error',  # Only show errors
        str(output_path)
    ]

    if dry_run:
        print(f"[DRY-RUN] Would extract: {output_path.name}")
        print(f"          Command: {' '.join(cmd)}")
        return True

    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)

        if result.returncode != 0:
            # Copy failed, try re-encoding
            cmd_reencode = [
                'ffmpeg',
                '-y',
                '-ss', f'{start_time:.3f}',
                '-i', str(input_video),
                '-t', f'{duration:.3f}',
                '-c:v', 'libx264',
                '-c:a', 'aac',
                '-preset', 'fast',
                '-crf', '23',
                '-avoid_negative_ts', 'make_zero',
                '-loglevel', 'error',
                str(output_path)
            ]

            result = subprocess.run(cmd_reencode, capture_output=True, text=True, timeout=60)

            if result.returncode != 0:
                print(f"✗ Error extracting {output_path.name}: {result.stderr}")
                return False

        return True

    except subprocess.TimeoutExpired:
        print(f"✗ Timeout extracting {output_path.name}")
        return False
    except Exception as e:
        print(f"✗ Exception extracting {output_path.name}: {e}")
        return False


def main():
    parser = argparse.ArgumentParser(
        description='Extract badminton shot clips from ShuttleSet videos',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Dry-run to preview extraction
  python scripts/extract_shuttleset_clips.py --input data/raw_videos --output data/clips --dry-run

  # Extract all shots from all matches
  python scripts/extract_shuttleset_clips.py --input data/raw_videos --output data/clips --execute

  # Extract only Smash and Clear shots
  python scripts/extract_shuttleset_clips.py --shot-types Smash Clear --execute

  # Extract from specific matches
  python scripts/extract_shuttleset_clips.py --match-ids 01 02 03 --execute

  # Custom clip duration (default 3.0 seconds)
  python scripts/extract_shuttleset_clips.py --duration 2.5 --execute
        """
    )

    parser.add_argument(
        '--input',
        type=Path,
        default=Path('data/raw_videos'),
        help='Input directory containing raw match videos (default: data/raw_videos)'
    )

    parser.add_argument(
        '--output',
        type=Path,
        default=Path('data/clips'),
        help='Output directory for extracted clips (default: data/clips)'
    )

    parser.add_argument(
        '--shuttleset',
        type=Path,
        default=Path('ShuttleSet'),
        help='ShuttleSet dataset directory (default: ShuttleSet)'
    )

    parser.add_argument(
        '--shot-types',
        nargs='+',
        choices=['Smash', 'Clear', 'Drop', 'Lift', 'Drive'],
        default=['Smash', 'Clear', 'Drop', 'Lift', 'Drive'],
        help='Shot types to extract (default: all 5 types)'
    )

    parser.add_argument(
        '--match-ids',
        nargs='+',
        help='Specific match IDs to process (e.g., 01 02 03). If not specified, processes all matches.'
    )

    parser.add_argument(
        '--duration',
        type=float,
        default=3.0,
        help='Clip duration in seconds (default: 3.0)'
    )

    parser.add_argument(
        '--pre-buffer',
        type=float,
        default=1.0,
        help='Seconds before contact frame to include (default: 1.0)'
    )

    parser.add_argument(
        '--post-buffer',
        type=float,
        default=2.0,
        help='Seconds after contact frame to include (default: 2.0)'
    )

    parser.add_argument(
        '--dry-run',
        action='store_true',
        help='Preview extraction without actually creating clips'
    )

    parser.add_argument(
        '--execute',
        action='store_true',
        help='Execute clip extraction (required for actual extraction)'
    )

    args = parser.parse_args()

    # Validate execution mode
    if not args.dry_run and not args.execute:
        print("Error: Must specify either --dry-run or --execute")
        parser.print_help()
        return 1

    # Validate directories
    if not args.input.exists():
        print(f"Error: Input directory not found: {args.input}")
        return 1

    if not args.shuttleset.exists():
        print(f"Error: ShuttleSet directory not found: {args.shuttleset}")
        return 1

    # Calculate clip duration from buffers
    clip_duration = args.pre_buffer + args.post_buffer
    if abs(clip_duration - args.duration) > 0.1:
        print(f"Note: Clip duration adjusted to {clip_duration:.1f}s (pre: {args.pre_buffer}s + post: {args.post_buffer}s)")

    print("=" * 80)
    print("SHUTTLESET VIDEO CLIP EXTRACTION")
    print("=" * 80)
    print(f"Input videos:  {args.input}")
    print(f"Output clips:  {args.output}")
    print(f"ShuttleSet:    {args.shuttleset}")
    print(f"Shot types:    {', '.join(args.shot_types)}")
    print(f"Clip duration: {clip_duration:.1f}s (pre: {args.pre_buffer}s, post: {args.post_buffer}s)")
    print(f"Mode:          {'DRY-RUN (preview only)' if args.dry_run else 'EXECUTE (extracting clips)'}")
    print("=" * 80)
    print()

    # Load match metadata
    print("Loading match metadata...")
    try:
        matches = get_match_metadata(args.shuttleset)
        print(f"✓ Loaded {len(matches)} matches from ShuttleSet")
    except Exception as e:
        print(f"✗ Error loading match metadata: {e}")
        return 1

    # Filter matches if specific IDs provided
    if args.match_ids:
        match_ids = [int(mid.lstrip('0')) for mid in args.match_ids]
        matches = {k: v for k, v in matches.items() if k in match_ids}
        print(f"✓ Filtered to {len(matches)} specified matches: {', '.join(args.match_ids)}")

    print()

    # Process each match
    total_clips_extracted = 0
    total_clips_failed = 0
    clips_per_type = Counter()
    skipped_matches = []

    for match_id, match_info in sorted(matches.items()):
        video_name = match_info['video_name']
        video_file = args.input / f"{match_id:02d}.mp4"

        if not video_file.exists():
            print(f"⚠️  Match {match_id:02d}: Video file not found: {video_file.name}")
            skipped_matches.append(match_id)
            continue

        print(f"Processing Match {match_id:02d}: {match_info['tournament']} - {match_info['round']}")
        print(f"  Players: {match_info['winner']} vs {match_info['loser']}")

        # Load shot annotations
        shots = load_shot_annotations(args.shuttleset, match_id, video_name)

        if not shots:
            print(f"  ⚠️  No shot annotations found")
            skipped_matches.append(match_id)
            continue

        # Filter shots by target class
        target_shots = [s for s in shots if s['target_class'] in args.shot_types]

        if not target_shots:
            print(f"  ⚠️  No shots matching target types: {', '.join(args.shot_types)}")
            continue

        # Count shots by type
        shot_counts = Counter(s['target_class'] for s in target_shots)
        print(f"  Shots found: {sum(shot_counts.values())} total")
        for shot_type in ['Smash', 'Clear', 'Drop', 'Lift', 'Drive']:
            if shot_type in shot_counts:
                print(f"    {shot_type}: {shot_counts[shot_type]}")

        # Extract clips
        match_extracted = 0
        match_failed = 0

        for shot in target_shots:
            shot_type = shot['target_class']

            # Generate output filename
            # Format: {match_id}_set{set}_rally{rally}_ball{ball}_{shot_type}.mp4
            filename = (
                f"{match_id:02d}_"
                f"set{shot['set_num']}_"
                f"rally{shot['rally']:02d}_"
                f"ball{int(shot['ball_round']):02d}_"
                f"{shot_type}.mp4"
            )

            output_path = args.output / shot_type / filename

            # Calculate start time (contact frame - pre_buffer)
            start_time = max(0, shot['timestamp_seconds'] - args.pre_buffer)

            # Extract clip
            success = extract_clip(
                input_video=video_file,
                output_path=output_path,
                start_time=start_time,
                duration=clip_duration,
                dry_run=args.dry_run
            )

            if success:
                match_extracted += 1
                clips_per_type[shot_type] += 1
            else:
                match_failed += 1

        total_clips_extracted += match_extracted
        total_clips_failed += match_failed

        if not args.dry_run:
            print(f"  ✓ Extracted: {match_extracted} clips, Failed: {match_failed}")

        print()

    # Summary
    print("=" * 80)
    print("EXTRACTION SUMMARY")
    print("=" * 80)
    print(f"Matches processed:     {len(matches) - len(skipped_matches)}/{len(matches)}")
    if skipped_matches:
        print(f"Matches skipped:       {len(skipped_matches)} (IDs: {', '.join([f'{m:02d}' for m in skipped_matches])})")
    print()
    print(f"Total clips extracted: {total_clips_extracted}")
    if total_clips_failed > 0:
        print(f"Total clips failed:    {total_clips_failed}")
    print()

    print("Clips by shot type:")
    for shot_type in ['Smash', 'Clear', 'Drop', 'Lift', 'Drive']:
        if shot_type in clips_per_type:
            print(f"  {shot_type:<10} {clips_per_type[shot_type]:>6} clips")
    print()

    if args.dry_run:
        print("⚠️  DRY-RUN mode - no clips were actually extracted")
        print("    Run with --execute to perform actual extraction")
    else:
        print(f"✓ Clips saved to: {args.output}")
        print()
        print("Next steps:")
        print("  1. Review extracted clips for quality")
        print("  2. Run pose extraction: python scripts/extract_poses_parallel.py")
        print("  3. Verify pose quality across all shot types")

    print("=" * 80)

    return 0


if __name__ == '__main__':
    sys.exit(main())
