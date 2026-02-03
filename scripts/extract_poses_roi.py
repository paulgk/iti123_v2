#!/usr/bin/env python3
"""
Extract pose sequences with ROI (Region of Interest) cropping.

This script extracts poses from video clips using MediaPipe, but crops each frame
to a Region of Interest (ROI) around the player position BEFORE running pose detection.
This ensures only ONE player is detected, solving the multi-player skeleton merging issue.

Usage:
    python scripts/extract_poses_roi.py --clips data/clips --metadata data/metadata.csv --output data/poses
    python scripts/extract_poses_roi.py --clips data/clips --metadata data/metadata.csv --output data/poses --num-workers 8
"""

import argparse
import csv
import pickle
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import multiprocessing as mp
from functools import partial

import cv2
import numpy as np
import mediapipe as mp_module
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from tqdm import tqdm


def calculate_roi(
    player_x: float,
    player_y: float,
    frame_width: int = 1920,
    frame_height: int = 1080,
    roi_width: int = 600,
    roi_height: int = 800
) -> Dict[str, int]:
    """
    Calculate ROI bounding box around player position.

    Args:
        player_x: Player X coordinate in original frame (0-1920)
        player_y: Player Y coordinate in original frame (0-1080)
        frame_width: Video frame width (default: 1920)
        frame_height: Video frame height (default: 1080)
        roi_width: ROI width in pixels (default: 600)
        roi_height: ROI height in pixels (default: 800)

    Returns:
        Dictionary with ROI coordinates: {x1, y1, x2, y2, width, height}

    Notes:
        - ROI is centered on player position
        - ROI is clipped to frame boundaries
        - If ROI hits boundary, it's shifted to stay within frame
    """
    # Center ROI on player position
    roi_x1 = int(player_x - roi_width // 2)
    roi_y1 = int(player_y - roi_height // 2)
    roi_x2 = roi_x1 + roi_width
    roi_y2 = roi_y1 + roi_height

    # Clip to frame boundaries
    if roi_x1 < 0:
        roi_x1 = 0
        roi_x2 = min(roi_width, frame_width)
    elif roi_x2 > frame_width:
        roi_x2 = frame_width
        roi_x1 = max(0, frame_width - roi_width)

    if roi_y1 < 0:
        roi_y1 = 0
        roi_y2 = min(roi_height, frame_height)
    elif roi_y2 > frame_height:
        roi_y2 = frame_height
        roi_y1 = max(0, frame_height - roi_height)

    return {
        'x1': roi_x1,
        'y1': roi_y1,
        'x2': roi_x2,
        'y2': roi_y2,
        'width': roi_x2 - roi_x1,
        'height': roi_y2 - roi_y1,
    }


def extract_pose_with_roi(
    video_path: Path,
    player_x: float,
    player_y: float,
    model_path: Path,
    roi_width: int = 600,
    roi_height: int = 800
) -> Optional[np.ndarray]:
    """
    Extract pose sequence from video using ROI cropping.

    Args:
        video_path: Path to video clip
        player_x: Player X position in original frame
        player_y: Player Y position in original frame
        model_path: Path to MediaPipe pose model
        roi_width: ROI width in pixels
        roi_height: ROI height in pixels

    Returns:
        Pose sequence as numpy array (T, 33, 3) or None if extraction fails

    Process:
        1. Load video clip
        2. For each frame:
           a. Calculate ROI around player position
           b. Crop frame to ROI
           c. Run MediaPipe on cropped frame
           d. Transform pose coordinates back to original frame space
        3. Return pose sequence
    """
    if not video_path.exists():
        return None

    # Initialize MediaPipe Pose Landmarker
    base_options = python.BaseOptions(
        model_asset_path=str(model_path),
        delegate=python.BaseOptions.Delegate.CPU
    )

    options = vision.PoseLandmarkerOptions(
        base_options=base_options,
        output_segmentation_masks=False,
        num_poses=1,  # Only detect one pose in ROI
        min_pose_detection_confidence=0.5,
        min_pose_presence_confidence=0.5,
        min_tracking_confidence=0.5
    )

    try:
        with vision.PoseLandmarker.create_from_options(options) as landmarker:
            # Open video
            cap = cv2.VideoCapture(str(video_path))

            if not cap.isOpened():
                return None

            # Get video properties
            frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

            # Calculate ROI (same for all frames - player doesn't move far in 3 seconds)
            roi = calculate_roi(player_x, player_y, frame_width, frame_height, roi_width, roi_height)

            poses = []

            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break

                # Crop frame to ROI
                cropped_frame = frame[roi['y1']:roi['y2'], roi['x1']:roi['x2']]

                # Convert BGR to RGB for MediaPipe
                rgb_frame = cv2.cvtColor(cropped_frame, cv2.COLOR_BGR2RGB)

                # Create MediaPipe Image
                mp_image = mp_module.Image(image_format=mp_module.ImageFormat.SRGB, data=rgb_frame)

                # Run pose detection on cropped frame
                results = landmarker.detect(mp_image)

                if results and results.pose_landmarks and len(results.pose_landmarks) > 0:
                    # Extract landmarks (33 keypoints)
                    landmarks = results.pose_landmarks[0]

                    # Transform coordinates from ROI space to original frame space
                    pose_frame = []
                    for landmark in landmarks:
                        # Scale from ROI to original frame coordinates
                        x = landmark.x * roi['width'] + roi['x1']
                        y = landmark.y * roi['height'] + roi['y1']
                        z = landmark.z  # Depth remains relative
                        visibility = landmark.visibility

                        # Normalize to 0-1 range in original frame
                        x_norm = x / frame_width
                        y_norm = y / frame_height

                        pose_frame.append([x_norm, y_norm, z])

                    poses.append(pose_frame)

            cap.release()

            if len(poses) == 0:
                return None

            return np.array(poses, dtype=np.float32)

    except Exception as e:
        print(f"Error extracting pose from {video_path.name}: {e}")
        return None


def process_clip(
    row: Dict,
    clips_dir: Path,
    output_dir: Path,
    model_path: Path,
    roi_width: int,
    roi_height: int
) -> Tuple[str, bool]:
    """
    Process a single clip: extract pose with ROI and save.

    Args:
        row: Metadata row with clip info
        clips_dir: Directory containing video clips
        output_dir: Directory to save pose files
        model_path: Path to MediaPipe model
        roi_width: ROI width
        roi_height: ROI height

    Returns:
        Tuple of (clip_name, success)
    """
    video_id = row['video_id']
    shot_type = row['shot_type']
    player_x = float(row['player_x'])
    player_y = float(row['player_y'])

    # Find clip file
    clip_path = clips_dir / shot_type / f"{video_id}.mp4"

    if not clip_path.exists():
        return (video_id, False)

    # Extract pose with ROI
    pose_sequence = extract_pose_with_roi(
        clip_path,
        player_x,
        player_y,
        model_path,
        roi_width,
        roi_height
    )

    if pose_sequence is None:
        return (video_id, False)

    # Save pose
    output_file = output_dir / f"{video_id}.pkl"
    output_file.parent.mkdir(parents=True, exist_ok=True)

    with open(output_file, 'wb') as f:
        pickle.dump(pose_sequence, f)

    return (video_id, True)


def main():
    parser = argparse.ArgumentParser(
        description='Extract poses with ROI cropping (single player)',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Extract poses with default settings
  python scripts/extract_poses_roi.py --clips data/clips --metadata data/metadata.csv --output data/poses

  # Use 8 parallel workers
  python scripts/extract_poses_roi.py --clips data/clips --metadata data/metadata.csv --output data/poses --num-workers 8

  # Custom ROI size
  python scripts/extract_poses_roi.py --clips data/clips --metadata data/metadata.csv --output data/poses --roi-width 700 --roi-height 900
        """
    )

    parser.add_argument(
        '--clips',
        type=Path,
        default=Path('data/clips'),
        help='Directory containing video clips (default: data/clips)'
    )

    parser.add_argument(
        '--metadata',
        type=Path,
        default=Path('data/metadata.csv'),
        help='Metadata CSV with player positions (default: data/metadata.csv)'
    )

    parser.add_argument(
        '--output',
        type=Path,
        default=Path('data/poses'),
        help='Output directory for pose files (default: data/poses)'
    )

    parser.add_argument(
        '--model',
        type=Path,
        default=Path('models/pose_landmarker_heavy.task'),
        help='Path to MediaPipe pose model (default: models/pose_landmarker_heavy.task)'
    )

    parser.add_argument(
        '--roi-width',
        type=int,
        default=600,
        help='ROI width in pixels (default: 600)'
    )

    parser.add_argument(
        '--roi-height',
        type=int,
        default=800,
        help='ROI height in pixels (default: 800)'
    )

    parser.add_argument(
        '--num-workers',
        type=int,
        default=4,
        help='Number of parallel workers (default: 4)'
    )

    args = parser.parse_args()

    # Validate inputs
    if not args.clips.exists():
        print(f"Error: Clips directory not found: {args.clips}")
        return 1

    if not args.metadata.exists():
        print(f"Error: Metadata file not found: {args.metadata}")
        return 1

    if not args.model.exists():
        print(f"Error: MediaPipe model not found: {args.model}")
        print(f"Download from: https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_heavy/float16/latest/pose_landmarker_heavy.task")
        return 1

    # Create output directory
    args.output.mkdir(parents=True, exist_ok=True)

    print("=" * 80)
    print("ROI-BASED POSE EXTRACTION")
    print("=" * 80)
    print(f"Clips:      {args.clips}")
    print(f"Metadata:   {args.metadata}")
    print(f"Output:     {args.output}")
    print(f"Model:      {args.model}")
    print(f"ROI size:   {args.roi_width}x{args.roi_height} pixels")
    print(f"Workers:    {args.num_workers}")
    print("=" * 80)
    print()

    # Load metadata
    print("Loading metadata...")
    metadata_rows = []
    shot_type_counts = {}

    with open(args.metadata, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            metadata_rows.append(row)
            shot_type = row.get('shot_type', 'Unknown')
            shot_type_counts[shot_type] = shot_type_counts.get(shot_type, 0) + 1

    print(f"✓ Loaded {len(metadata_rows)} clips from metadata")
    print(f"\nBreakdown by shot type:")
    for shot_type in ['Smash', 'Clear', 'Drop', 'Lift', 'Drive']:
        if shot_type in shot_type_counts:
            print(f"  {shot_type:<10} {shot_type_counts[shot_type]:>6} clips")
    print()

    # Process clips
    import time
    start_time = time.time()

    print("Extracting poses with ROI...")
    print(f"Using {args.num_workers} worker(s) for parallel processing")
    print()

    process_func = partial(
        process_clip,
        clips_dir=args.clips,
        output_dir=args.output,
        model_path=args.model,
        roi_width=args.roi_width,
        roi_height=args.roi_height
    )

    successful = 0
    failed = 0
    last_report_time = time.time()
    report_interval = 60  # Report every 60 seconds

    if args.num_workers > 1:
        # Parallel processing
        with mp.Pool(args.num_workers) as pool:
            results_iter = pool.imap(process_func, metadata_rows)

            for idx, (video_id, success) in enumerate(tqdm(
                results_iter,
                total=len(metadata_rows),
                desc="Extracting poses",
                unit="clips"
            ), 1):
                if success:
                    successful += 1
                else:
                    failed += 1

                # Periodic status report
                current_time = time.time()
                if current_time - last_report_time >= report_interval:
                    elapsed = current_time - start_time
                    rate = idx / elapsed if elapsed > 0 else 0
                    remaining = len(metadata_rows) - idx
                    eta_seconds = remaining / rate if rate > 0 else 0

                    print(f"\nProgress: {idx}/{len(metadata_rows)} ({idx/len(metadata_rows)*100:.1f}%)")
                    print(f"Success rate: {successful}/{idx} ({successful/idx*100:.1f}%)")
                    print(f"Rate: {rate:.1f} clips/sec")
                    print(f"Estimated time remaining: {eta_seconds/60:.1f} minutes\n")

                    last_report_time = current_time
    else:
        # Sequential processing
        for idx, row in enumerate(tqdm(metadata_rows, desc="Extracting poses", unit="clips"), 1):
            video_id, success = process_func(row)
            if success:
                successful += 1
            else:
                failed += 1

            # Periodic status report
            current_time = time.time()
            if current_time - last_report_time >= report_interval:
                elapsed = current_time - start_time
                rate = idx / elapsed if elapsed > 0 else 0
                remaining = len(metadata_rows) - idx
                eta_seconds = remaining / rate if rate > 0 else 0

                print(f"\nProgress: {idx}/{len(metadata_rows)} ({idx/len(metadata_rows)*100:.1f}%)")
                print(f"Success rate: {successful}/{idx} ({successful/idx*100:.1f}%)")
                print(f"Rate: {rate:.1f} clips/sec")
                print(f"Estimated time remaining: {eta_seconds/60:.1f} minutes\n")

                last_report_time = current_time

    # Calculate final statistics
    end_time = time.time()
    total_duration = end_time - start_time
    hours = int(total_duration // 3600)
    minutes = int((total_duration % 3600) // 60)
    seconds = int(total_duration % 60)

    # Summary
    print()
    print("=" * 80)
    print("EXTRACTION SUMMARY")
    print("=" * 80)
    print(f"Total clips:      {len(metadata_rows)}")
    print(f"Successful:       {successful}")
    print(f"Failed:           {failed}")
    print(f"Success rate:     {successful / len(metadata_rows) * 100:.1f}%")
    print(f"Processing time:  {hours:02d}:{minutes:02d}:{seconds:02d}")
    if successful > 0:
        avg_time = total_duration / len(metadata_rows)
        print(f"Avg time/clip:    {avg_time:.2f} seconds")
    print()
    print(f"Output directory: {args.output}")
    print(f"Pose files saved: {successful}")
    print("=" * 80)

    return 0


if __name__ == '__main__':
    sys.exit(main())
