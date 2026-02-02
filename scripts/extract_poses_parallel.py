#!/usr/bin/env python3
"""
Parallel MediaPipe Pose Extraction

Uses multiprocessing to extract poses from multiple videos simultaneously.
Much faster than sequential processing.

Usage:
    python scripts/extract_poses_parallel.py \
        --video-dir data/videos/ \
        --output-dir data/processed/poses/ \
        --num-workers 4

Recommended workers:
    - CPU: 2-4 workers
    - GPU: 1-2 workers (MediaPipe GPU support limited)
"""

import argparse
import os
import pickle
from pathlib import Path
from datetime import datetime
import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import numpy as np
import pandas as pd
from tqdm import tqdm
from multiprocessing import Pool, cpu_count
import sys

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))


def extract_pose_from_video(args):
    """
    Wrapper for multiprocessing - takes tuple of arguments
    """
    video_path, output_dir, target_fps, min_confidence, model_complexity = args

    try:
        # Generate output filename
        video_name = Path(video_path).stem
        output_path = Path(output_dir) / f"{video_name}_pose.pkl"

        # Skip if already exists and is valid (non-empty)
        if output_path.exists():
            # Validate file is not empty/corrupted
            if output_path.stat().st_size > 100:  # At least 100 bytes
                try:
                    # Quick validation: try to load the pickle
                    with open(output_path, 'rb') as f:
                        _ = pickle.load(f)
                    return {'status': 'skipped', 'video': video_name}
                except:
                    # File corrupted, re-extract
                    print(f"⚠️  Corrupted pose file detected, re-extracting: {video_name}")
                    pass

        # Open video
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            return {'status': 'error', 'video': video_name, 'error': 'Cannot open video'}

        # Get video properties
        original_fps = cap.get(cv2.CAP_PROP_FPS)
        frame_skip = max(1, int(original_fps / target_fps))

        pose_sequence = []

        # Create PoseLandmarker options for new MediaPipe API (0.10.x)
        # Get model path from environment or use default location
        model_path = os.environ.get('MEDIAPIPE_POSE_MODEL', 'models/mediapipe/pose_landmarker.task')

        base_options = python.BaseOptions(
            model_asset_path=model_path,
            delegate=python.BaseOptions.Delegate.CPU
        )

        options = vision.PoseLandmarkerOptions(
            base_options=base_options,
            running_mode=vision.RunningMode.VIDEO,
            num_poses=1,
            min_pose_detection_confidence=min_confidence,
            min_pose_presence_confidence=min_confidence,
            min_tracking_confidence=min_confidence,
            output_segmentation_masks=False
        )

        # Create PoseLandmarker
        landmarker = vision.PoseLandmarker.create_from_options(options)

        frame_idx = 0
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            # Sample frames
            if frame_idx % frame_skip == 0:
                # Convert frame to MediaPipe Image format
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)

                # Calculate timestamp in milliseconds
                timestamp_ms = int(frame_idx * 1000 / original_fps)

                # Detect pose
                detection_result = landmarker.detect_for_video(mp_image, timestamp_ms)

                if detection_result.pose_landmarks:
                    # Extract landmarks (first pose only)
                    landmarks_list = detection_result.pose_landmarks[0]
                    landmarks = [[lm.x, lm.y, lm.visibility]
                               for lm in landmarks_list]
                    pose_sequence.append(landmarks)

            frame_idx += 1

        cap.release()
        landmarker.close()

        if len(pose_sequence) == 0:
            return {'status': 'error', 'video': video_name, 'error': 'No poses detected'}

        # Save pose sequence
        pose_array = np.array(pose_sequence, dtype=np.float32)
        with open(output_path, 'wb') as f:
            pickle.dump(pose_array, f)

        # Infer stroke type from path (supports all 5 shot types)
        path_parts = Path(video_path).parts
        stroke_type = 'unknown'
        for part in path_parts:
            part_lower = part.lower()
            if 'smash' in part_lower:
                stroke_type = 'smash'
                break
            elif 'clear' in part_lower:
                stroke_type = 'clear'
                break
            elif 'drop' in part_lower:
                stroke_type = 'drop'
                break
            elif 'lift' in part_lower:
                stroke_type = 'lift'
                break
            elif 'drive' in part_lower:
                stroke_type = 'drive'
                break

        # Infer player ID
        filename = Path(video_path).stem
        parts = filename.split('_')
        player_id = parts[0] if parts else filename

        return {
            'status': 'success',
            'video_id': video_name,
            'video_path': str(video_path),
            'pose_file': str(output_path.relative_to(Path(output_dir).parent.parent)),
            'stroke_type': stroke_type,
            'player_id': player_id,
            'extracted_frames': len(pose_sequence)
        }

    except Exception as e:
        return {'status': 'error', 'video': Path(video_path).name, 'error': str(e)}


def main():
    parser = argparse.ArgumentParser(description='Parallel pose extraction')
    parser.add_argument('--video-dir', type=str, required=True)
    parser.add_argument('--output-dir', type=str, required=True)
    parser.add_argument('--target-fps', type=int, default=30)
    parser.add_argument('--min-confidence', type=float, default=0.5)
    parser.add_argument('--model-complexity', type=int, default=1, choices=[0, 1, 2])
    parser.add_argument('--num-workers', type=int, default=None,
                       help='Number of parallel workers (default: CPU count - 1)')

    args = parser.parse_args()

    # Determine number of workers
    if args.num_workers is None:
        args.num_workers = max(1, cpu_count() - 1)

    print(f"\n{'='*60}")
    print(f"PARALLEL MEDIAPIPE POSE EXTRACTION")
    print(f"{'='*60}")
    print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Workers: {args.num_workers}")
    print(f"Model complexity: {args.model_complexity}")
    print()

    # Find all videos
    video_extensions = ['.mp4', '.avi', '.mov', '.MP4', '.AVI', '.MOV']
    video_files = []
    for ext in video_extensions:
        video_files.extend(Path(args.video_dir).rglob(f'*{ext}'))

    print(f"Found {len(video_files)} video files")

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    # Check existing pose files
    existing_poses = list(Path(args.output_dir).glob('*_pose.pkl'))
    print(f"Found {len(existing_poses)} existing pose files (will skip these)")

    # Estimate remaining work
    remaining = len(video_files) - len(existing_poses)
    if remaining > 0:
        print(f"Estimated remaining clips to process: {remaining}")
    else:
        print("All clips already processed!")

    # Prepare arguments for each video
    task_args = [(str(vf), args.output_dir, args.target_fps,
                  args.min_confidence, args.model_complexity)
                 for vf in video_files]

    # Process in parallel
    results = []
    processed_count = 0
    with Pool(processes=args.num_workers) as pool:
        for result in tqdm(pool.imap_unordered(extract_pose_from_video, task_args),
                          total=len(video_files),
                          desc="Extracting poses"):
            results.append(result)
            processed_count += 1

            # Checkpoint progress every 100 clips
            if processed_count % 100 == 0:
                success_so_far = len([r for r in results if r['status'] == 'success'])
                skipped_so_far = len([r for r in results if r['status'] == 'skipped'])
                failed_so_far = len([r for r in results if r['status'] == 'error'])
                print(f"\n  Checkpoint ({processed_count}/{len(video_files)}): "
                      f"{success_so_far} new, {skipped_so_far} skipped, {failed_so_far} failed")

    # Collect statistics
    successful = [r for r in results if r['status'] == 'success']
    skipped = [r for r in results if r['status'] == 'skipped']
    failed = [r for r in results if r['status'] == 'error']

    print(f"\n{'='*60}")
    print(f"EXTRACTION COMPLETE")
    print(f"{'='*60}")
    print(f"Total videos: {len(video_files)}")
    print(f"Successfully processed: {len(successful)}")
    print(f"Skipped (already exist): {len(skipped)}")
    print(f"Failed: {len(failed)}")

    # Save metadata (append to existing if present)
    if successful:
        df_new = pd.DataFrame(successful)
        metadata_path = Path(args.output_dir).parent.parent / 'data' / 'metadata.csv'
        os.makedirs(metadata_path.parent, exist_ok=True)

        # Merge with existing metadata if present
        if metadata_path.exists():
            try:
                df_existing = pd.read_csv(metadata_path)
                # Remove duplicates based on video_id (keep new ones)
                df_existing = df_existing[~df_existing['video_id'].isin(df_new['video_id'])]
                df_combined = pd.concat([df_existing, df_new], ignore_index=True)
                df_combined.to_csv(metadata_path, index=False)
                print(f"\n✓ Metadata updated: {metadata_path}")
                print(f"  Previous entries: {len(df_existing)}")
                print(f"  New entries: {len(df_new)}")
                print(f"  Total entries: {len(df_combined)}")
                print(f"\nStroke distribution:")
                print(df_combined['stroke_type'].value_counts())
            except Exception as e:
                # If merge fails, just save new data
                print(f"⚠️  Could not merge with existing metadata: {e}")
                df_new.to_csv(metadata_path, index=False)
                print(f"\n✓ Metadata saved (new): {metadata_path}")
                print(f"\nStroke distribution:")
                print(df_new['stroke_type'].value_counts())
        else:
            df_new.to_csv(metadata_path, index=False)
            print(f"\n✓ Metadata saved: {metadata_path}")
            print(f"\nStroke distribution:")
            print(df_new['stroke_type'].value_counts())

    if failed:
        print(f"\n⚠️  {len(failed)} failures:")
        for f in failed[:10]:
            print(f"  {f['video']}: {f['error']}")
        if len(failed) > 10:
            print(f"  ... and {len(failed)-10} more")

    print(f"\nFinished: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")


if __name__ == '__main__':
    main()
