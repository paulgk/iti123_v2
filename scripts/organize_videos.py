#!/usr/bin/env python3
"""
Organize Video Clips into Stroke Type Folders

Moves or copies video files into subdirectories based on stroke type detected in filename.

Supports:
- clear, smash, drop, lift stroke types
- Multiple detection patterns (case-insensitive)
- Copy or move operations
- Dry-run mode to preview changes

Usage:
    # Dry run (preview only)
    python scripts/organize_videos.py --input data/videos/clips --dry-run

    # Copy files to organized folders
    python scripts/organize_videos.py --input data/videos/clips --copy

    # Move files to organized folders
    python scripts/organize_videos.py --input data/videos/clips --move

    # GCS: Organize videos in cloud storage
    python scripts/organize_videos.py --gcs gs://BUCKET/videos/clips --dry-run
"""

import argparse
import os
import shutil
from pathlib import Path
import re
from typing import Dict, List, Tuple
import subprocess


def detect_stroke_type(filename: str) -> str:
    """
    Detect stroke type from filename

    Args:
        filename: Video filename (e.g., "player1_clear_001.mp4")

    Returns:
        Stroke type: 'clear', 'smash', 'drop', 'lift', or 'unknown'
    """
    filename_lower = filename.lower()

    # Pattern matching (in priority order)
    patterns = {
        'clear': [r'\bclear\b', r'\bclr\b', r'_c_', r'_c\d+'],
        'smash': [r'\bsmash\b', r'\bsmsh\b', r'_s_', r'_s\d+'],
        'drop': [r'\bdrop\b', r'\bdrp\b', r'_d_', r'_d\d+'],
        'lift': [r'\blift\b', r'\blft\b', r'_l_', r'_l\d+']
    }

    for stroke_type, pattern_list in patterns.items():
        for pattern in pattern_list:
            if re.search(pattern, filename_lower):
                return stroke_type

    return 'unknown'


def find_video_files(directory: Path) -> List[Path]:
    """Find all video files in directory"""
    video_extensions = ['.mp4', '.avi', '.mov', '.MP4', '.AVI', '.MOV']
    video_files = []

    for ext in video_extensions:
        video_files.extend(directory.glob(f'*{ext}'))
        video_files.extend(directory.glob(f'**/*{ext}'))

    # Remove duplicates and files already in subdirectories
    unique_files = []
    seen = set()

    for video_file in video_files:
        # Skip if already in a stroke type folder
        parent_name = video_file.parent.name.lower()
        if parent_name in ['clear', 'smash', 'drop', 'lift', 'unknown']:
            continue

        if video_file not in seen:
            unique_files.append(video_file)
            seen.add(video_file)

    return sorted(unique_files)


def organize_local_videos(input_dir: Path, dry_run: bool = True,
                          move: bool = False) -> Dict[str, List[str]]:
    """
    Organize local video files into stroke type folders

    Args:
        input_dir: Directory containing video files
        dry_run: If True, only preview changes
        move: If True, move files; if False, copy files

    Returns:
        Dictionary of stroke types and organized files
    """
    if not input_dir.exists():
        raise ValueError(f"Input directory not found: {input_dir}")

    # Find all video files
    video_files = find_video_files(input_dir)

    if len(video_files) == 0:
        print(f"No video files found in {input_dir}")
        return {}

    print(f"Found {len(video_files)} video files")
    print()

    # Categorize files
    categorized = {
        'clear': [],
        'smash': [],
        'drop': [],
        'lift': [],
        'unknown': []
    }

    for video_file in video_files:
        stroke_type = detect_stroke_type(video_file.name)
        categorized[stroke_type].append(video_file)

    # Show distribution
    print("Detected stroke types:")
    for stroke_type, files in categorized.items():
        if len(files) > 0:
            print(f"  {stroke_type:8s}: {len(files):4d} files")
    print()

    # Organize files
    organized = {k: [] for k in categorized.keys()}

    for stroke_type, files in categorized.items():
        if len(files) == 0:
            continue

        # Create output directory
        output_dir = input_dir / stroke_type

        if not dry_run:
            output_dir.mkdir(exist_ok=True)

        for video_file in files:
            dest_path = output_dir / video_file.name

            if dry_run:
                action = "MOVE" if move else "COPY"
                print(f"[{action}] {video_file.name} -> {stroke_type}/")
                organized[stroke_type].append(str(dest_path))
            else:
                try:
                    if move:
                        shutil.move(str(video_file), str(dest_path))
                        print(f"✓ Moved: {video_file.name} -> {stroke_type}/")
                    else:
                        shutil.copy2(str(video_file), str(dest_path))
                        print(f"✓ Copied: {video_file.name} -> {stroke_type}/")

                    organized[stroke_type].append(str(dest_path))

                except Exception as e:
                    print(f"✗ Error: {video_file.name}: {e}")

    return organized


def organize_gcs_videos(gcs_path: str, dry_run: bool = True) -> Dict[str, List[str]]:
    """
    Organize GCS video files into stroke type folders

    Args:
        gcs_path: GCS path (e.g., gs://bucket/videos/clips)
        dry_run: If True, only preview changes

    Returns:
        Dictionary of stroke types and organized files
    """
    if not gcs_path.startswith('gs://'):
        raise ValueError(f"GCS path must start with gs://: {gcs_path}")

    # List all video files
    print(f"Listing videos in {gcs_path}...")

    result = subprocess.run(
        ['gsutil', 'ls', f'{gcs_path}/*.mp4'],
        capture_output=True,
        text=True
    )

    if result.returncode != 0:
        # Try with recursive
        result = subprocess.run(
            ['gsutil', 'ls', '-r', f'{gcs_path}/**/*.mp4'],
            capture_output=True,
            text=True
        )

    if result.returncode != 0:
        print(f"Error listing GCS files: {result.stderr}")
        return {}

    video_files = [line.strip() for line in result.stdout.split('\n') if line.strip()]

    # Filter out files already in subdirectories
    filtered_files = []
    for video_file in video_files:
        path_parts = video_file.replace(gcs_path, '').strip('/').split('/')
        # Skip if already in a stroke type folder
        if len(path_parts) > 1 and path_parts[0] in ['clear', 'smash', 'drop', 'lift', 'unknown']:
            continue
        filtered_files.append(video_file)

    video_files = filtered_files

    if len(video_files) == 0:
        print(f"No video files found in {gcs_path}")
        return {}

    print(f"Found {len(video_files)} video files")
    print()

    # Categorize files
    categorized = {
        'clear': [],
        'smash': [],
        'drop': [],
        'lift': [],
        'unknown': []
    }

    for video_file in video_files:
        filename = video_file.split('/')[-1]
        stroke_type = detect_stroke_type(filename)
        categorized[stroke_type].append(video_file)

    # Show distribution
    print("Detected stroke types:")
    for stroke_type, files in categorized.items():
        if len(files) > 0:
            print(f"  {stroke_type:8s}: {len(files):4d} files")
    print()

    # Organize files
    organized = {k: [] for k in categorized.keys()}

    for stroke_type, files in categorized.items():
        if len(files) == 0:
            continue

        for video_file in files:
            filename = video_file.split('/')[-1]
            dest_path = f"{gcs_path}/{stroke_type}/{filename}"

            if dry_run:
                print(f"[MOVE] {video_file} -> {dest_path}")
                organized[stroke_type].append(dest_path)
            else:
                try:
                    # Use gsutil mv to move file
                    result = subprocess.run(
                        ['gsutil', 'mv', video_file, dest_path],
                        capture_output=True,
                        text=True
                    )

                    if result.returncode == 0:
                        print(f"✓ Moved: {filename} -> {stroke_type}/")
                        organized[stroke_type].append(dest_path)
                    else:
                        print(f"✗ Error moving {filename}: {result.stderr}")

                except Exception as e:
                    print(f"✗ Error: {filename}: {e}")

    return organized


def main():
    parser = argparse.ArgumentParser(
        description='Organize video clips into stroke type folders',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )

    # Input options
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument('--input', type=str,
                            help='Local directory containing video clips')
    input_group.add_argument('--gcs', type=str,
                            help='GCS path (e.g., gs://bucket/videos/clips)')

    # Operation options
    operation_group = parser.add_mutually_exclusive_group()
    operation_group.add_argument('--dry-run', action='store_true',
                                help='Preview changes without executing')
    operation_group.add_argument('--copy', action='store_true',
                                help='Copy files to organized folders (local only)')
    operation_group.add_argument('--move', action='store_true',
                                help='Move files to organized folders')

    args = parser.parse_args()

    # Default to dry-run if no operation specified
    if not args.dry_run and not args.copy and not args.move:
        args.dry_run = True
        print("⚠️  No operation specified - running in DRY-RUN mode")
        print("   Use --copy or --move to execute changes")
        print()

    print(f"{'='*60}")
    print("VIDEO ORGANIZATION")
    print(f"{'='*60}")

    if args.input:
        # Local files
        input_path = Path(args.input)
        mode = "DRY-RUN" if args.dry_run else ("COPY" if args.copy else "MOVE")
        print(f"Mode: {mode}")
        print(f"Input: {input_path}")
        print()

        organized = organize_local_videos(
            input_path,
            dry_run=args.dry_run,
            move=args.move
        )

    elif args.gcs:
        # GCS files
        mode = "DRY-RUN" if args.dry_run else "MOVE"
        print(f"Mode: {mode}")
        print(f"GCS Path: {args.gcs}")
        print()

        organized = organize_gcs_videos(
            args.gcs,
            dry_run=args.dry_run
        )

    # Summary
    print()
    print(f"{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")

    total = sum(len(files) for files in organized.values())
    print(f"Total files organized: {total}")

    for stroke_type, files in organized.items():
        if len(files) > 0:
            print(f"  {stroke_type:8s}: {len(files):4d} files")

    if args.dry_run:
        print()
        print("⚠️  DRY-RUN mode - no changes were made")
        print("   Run with --move or --copy to execute changes")
    else:
        print()
        print("✓ Organization complete")

    print()


if __name__ == '__main__':
    main()
