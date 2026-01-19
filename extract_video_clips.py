#!/usr/bin/env python3
"""
Extract Individual Video Clips from Full Match Videos

This script extracts short video clips for each stroke (Drop, Lift, etc.)
from full match videos based on the metadata CSV files.

Process:
1. Read metadata CSV (contains frame numbers and timestamps)
2. For each stroke, extract a 2-second clip centered on the stroke frame
3. Save clips to data/processed/clips/

Usage:
  python extract_video_clips.py --metadata data/processed/clips/drop_smash_metadata.csv
  python extract_video_clips.py --metadata data/processed/clips/lift_smash_metadata.csv
  python extract_video_clips.py --metadata data/processed/clips/multiclass_4stroke_metadata.csv
"""

import pandas as pd
import cv2
from pathlib import Path
import argparse
from tqdm import tqdm
import sys

def extract_clip(video_path, frame_num, output_path, fps=30, duration_sec=2.0):
    """
    Extract a short clip centered on a specific frame

    Args:
        video_path: Path to full match video
        frame_num: Frame number of the stroke
        output_path: Where to save the clip
        fps: Frames per second of the video
        duration_sec: Total duration of extracted clip in seconds
    """
    # Calculate frame range
    half_duration_frames = int((duration_sec / 2) * fps)
    start_frame = max(0, frame_num - half_duration_frames)
    end_frame = frame_num + half_duration_frames

    # Open video
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        print(f"❌ Error: Could not open video {video_path}")
        return False

    # Get video properties
    actual_fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # Set to start frame
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

    # Prepare video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(str(output_path), fourcc, actual_fps, (width, height))

    # Extract frames
    current_frame = start_frame
    while current_frame < end_frame:
        ret, frame = cap.read()
        if not ret:
            break
        out.write(frame)
        current_frame += 1

    # Cleanup
    cap.release()
    out.release()

    return True

def main():
    parser = argparse.ArgumentParser(description='Extract video clips from full match videos')
    parser.add_argument('--metadata', required=True, help='Path to metadata CSV file')
    parser.add_argument('--videos', default='data/raw_videos', help='Directory containing full match videos')
    parser.add_argument('--output', default='data/processed/clips', help='Output directory for clips')
    parser.add_argument('--fps', type=int, default=30, help='Video FPS (default: 30)')
    parser.add_argument('--duration', type=float, default=2.0, help='Clip duration in seconds (default: 2.0)')
    parser.add_argument('--force', action='store_true', help='Re-extract even if clip exists')

    args = parser.parse_args()

    # Load metadata
    print("=" * 80)
    print("VIDEO CLIP EXTRACTION")
    print("=" * 80)
    print()

    metadata_file = Path(args.metadata)
    if not metadata_file.exists():
        print(f"❌ Error: Metadata file not found: {metadata_file}")
        sys.exit(1)

    print(f"Loading metadata from: {metadata_file}")
    df = pd.read_csv(metadata_file)
    print(f"✓ Loaded {len(df):,} strokes")
    print()

    # Show stroke type distribution
    if 'stroke_type_english' in df.columns:
        print("Stroke types to extract:")
        for stroke_type in df['stroke_type_english'].unique():
            count = len(df[df['stroke_type_english'] == stroke_type])
            print(f"  {stroke_type}: {count:,}")
    print()

    # Create output directory
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    videos_dir = Path(args.videos)
    if not videos_dir.exists():
        print(f"❌ Error: Videos directory not found: {videos_dir}")
        print(f"   Please ensure full match videos are in {videos_dir}")
        sys.exit(1)

    # Map match_id to video files
    print("Finding match videos...")
    video_files = list(videos_dir.glob("*.mp4"))
    print(f"✓ Found {len(video_files)} video files in {videos_dir}")
    print()

    # Create mapping: match_id -> video_path
    # Assuming videos are named like: 01.mp4, 02.mp4, etc. (match_id)
    match_videos = {}
    for video_path in video_files:
        # Extract match ID from filename (e.g., "01.mp4" -> 1)
        match_id_str = video_path.stem
        try:
            match_id = int(match_id_str)
            match_videos[match_id] = video_path
        except ValueError:
            print(f"⚠️  Warning: Could not parse match ID from {video_path.name}")

    print(f"Mapped {len(match_videos)} matches to video files")
    print()

    # Extract clips
    print("Extracting clips...")
    print("-" * 80)

    extracted = 0
    skipped = 0
    failed = 0

    for idx, row in tqdm(df.iterrows(), total=len(df), desc="Extracting"):
        # Get clip information
        match_id = row['match_id']
        clip_name = row['clip_name']
        frame_num = row['frame_num']

        # Check if clip already exists
        output_path = output_dir / clip_name
        if output_path.exists() and not args.force:
            skipped += 1
            continue

        # Find corresponding video
        if match_id not in match_videos:
            if failed == 0:  # Only print first few errors
                print(f"\n⚠️  Warning: No video found for match_id {match_id}")
            failed += 1
            continue

        video_path = match_videos[match_id]

        # Extract clip
        success = extract_clip(
            video_path=video_path,
            frame_num=frame_num,
            output_path=output_path,
            fps=args.fps,
            duration_sec=args.duration
        )

        if success:
            extracted += 1
        else:
            failed += 1

    # Summary
    print()
    print("=" * 80)
    print("EXTRACTION COMPLETE")
    print("=" * 80)
    print()
    print(f"✓ Extracted: {extracted:,} clips")
    print(f"⏭️  Skipped (already exist): {skipped:,} clips")
    if failed > 0:
        print(f"❌ Failed: {failed:,} clips")
    print()
    print(f"Total clips in {output_dir}: {len(list(output_dir.glob('*.mp4'))):,}")
    print()

    if failed > 0:
        print("⚠️  Some clips failed to extract. Common reasons:")
        print("  - Video file not found for match_id")
        print("  - Frame number out of range")
        print("  - Corrupted video file")

    print("Next step: Extract poses from clips")
    print(f"  python src/data_processing/extract_poses.py")

if __name__ == "__main__":
    main()
