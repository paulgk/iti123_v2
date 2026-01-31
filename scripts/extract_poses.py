#!/usr/bin/env python3
"""
MediaPipe Pose Extraction for Badminton Videos

Extracts pose sequences from video clips using MediaPipe Pose.
Outputs pickle files with landmark coordinates for downstream feature extraction.

Usage:
    python scripts/extract_poses.py --video-dir data/videos/clips --output-dir data/processed/poses

Options:
    --video-dir DIR      Directory containing video clips (mp4, avi, mov)
    --output-dir DIR     Directory to save pose pickle files
    --target-fps 30      Target FPS for pose extraction (default: 30)
    --min-confidence 0.5 Minimum detection confidence (default: 0.5)
    --create-metadata    Create metadata.csv automatically from folder structure
"""

import argparse
import os
import pickle
from pathlib import Path
from datetime import datetime
import cv2
import mediapipe as mp
import numpy as np
import pandas as pd
from tqdm import tqdm

# MediaPipe setup
mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils


def extract_pose_from_video(video_path, min_confidence=0.5, target_fps=30):
    """
    Extract pose landmarks from a single video file

    Args:
        video_path: Path to video file
        min_confidence: Minimum detection confidence (0-1)
        target_fps: Target frames per second for extraction

    Returns:
        pose_sequence: numpy array of shape (n_frames, 33, 3)
                      33 landmarks x (x, y, visibility)
        metadata: dict with video info
    """
    cap = cv2.VideoCapture(str(video_path))

    if not cap.isOpened():
        raise ValueError(f"Cannot open video: {video_path}")

    # Get video properties
    original_fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # Calculate frame sampling rate
    frame_skip = max(1, int(original_fps / target_fps))

    pose_sequence = []
    frame_indices = []

    with mp_pose.Pose(
        static_image_mode=False,
        model_complexity=2,  # 0=lite, 1=full, 2=heavy (best accuracy)
        min_detection_confidence=min_confidence,
        min_tracking_confidence=min_confidence
    ) as pose:

        frame_idx = 0

        while cap.isOpened():
            ret, frame = cap.read()

            if not ret:
                break

            # Sample frames based on target FPS
            if frame_idx % frame_skip == 0:
                # Convert BGR to RGB (MediaPipe expects RGB)
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

                # Process frame
                results = pose.process(frame_rgb)

                if results.pose_landmarks:
                    # Extract landmarks (33 points x 3 coordinates)
                    landmarks = []
                    for landmark in results.pose_landmarks.landmark:
                        landmarks.append([
                            landmark.x,
                            landmark.y,
                            landmark.visibility
                        ])
                    pose_sequence.append(landmarks)
                    frame_indices.append(frame_idx)
                else:
                    # No pose detected - add None or skip?
                    # For now, skip frames without detected pose
                    pass

            frame_idx += 1

    cap.release()

    if len(pose_sequence) == 0:
        raise ValueError(f"No poses detected in video: {video_path}")

    # Convert to numpy array
    pose_sequence = np.array(pose_sequence, dtype=np.float32)

    metadata = {
        'video_path': str(video_path),
        'original_fps': original_fps,
        'target_fps': target_fps,
        'frame_skip': frame_skip,
        'total_frames': total_frames,
        'extracted_frames': len(pose_sequence),
        'frame_indices': frame_indices,
        'resolution': (width, height),
        'shape': pose_sequence.shape
    }

    return pose_sequence, metadata


def infer_stroke_type_from_path(video_path):
    """
    Infer stroke type from video path
    Assumes structure like: videos/clips/clear/video.mp4 or videos/clips/smash/video.mp4
    """
    path_parts = Path(video_path).parts

    # Look for 'clear' or 'smash' in path
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

    # Default to unknown
    return 'unknown'


def infer_player_id_from_path(video_path):
    """
    Infer player ID from video filename
    Assumes format like: player1_shot001.mp4 or p1_clear_001.mp4
    """
    filename = Path(video_path).stem

    # Try to extract player ID from filename
    parts = filename.split('_')
    for part in parts:
        if part.startswith('p') or part.startswith('player'):
            return part

    # Default: use filename as unique ID
    return filename.split('_')[0] if '_' in filename else filename


def process_all_videos(video_dir, output_dir, target_fps=30, min_confidence=0.5,
                       create_metadata=True, verbose=False):
    """
    Process all videos in directory and extract pose sequences

    Args:
        video_dir: Directory containing video files
        output_dir: Directory to save pose pickle files
        target_fps: Target FPS for extraction
        min_confidence: Minimum detection confidence
        create_metadata: Whether to create metadata.csv
        verbose: Show detailed progress

    Returns:
        results: List of dicts with processing results
    """
    # Find all video files
    video_extensions = ['.mp4', '.avi', '.mov', '.MP4', '.AVI', '.MOV']
    video_files = []

    for ext in video_extensions:
        video_files.extend(Path(video_dir).rglob(f'*{ext}'))

    if len(video_files) == 0:
        raise ValueError(f"No video files found in {video_dir}")

    print(f"Found {len(video_files)} video files")

    # Create output directory
    os.makedirs(output_dir, exist_ok=True)

    results = []
    failed = []

    # Process each video
    for video_path in tqdm(video_files, desc="Extracting poses"):
        try:
            # Generate output filename
            video_name = Path(video_path).stem
            output_path = Path(output_dir) / f"{video_name}_pose.pkl"

            # Skip if already exists
            if output_path.exists():
                if verbose:
                    print(f"Skipping {video_name} (already exists)")
                continue

            # Extract pose
            pose_sequence, metadata = extract_pose_from_video(
                video_path,
                min_confidence=min_confidence,
                target_fps=target_fps
            )

            # Save pose sequence
            with open(output_path, 'wb') as f:
                pickle.dump(pose_sequence, f)

            # Infer labels from path
            stroke_type = infer_stroke_type_from_path(video_path)
            player_id = infer_player_id_from_path(video_path)

            results.append({
                'video_id': video_name,
                'video_path': str(video_path),
                'pose_file': str(output_path.relative_to(Path(output_dir).parent.parent)),
                'stroke_type': stroke_type,
                'player_id': player_id,
                'extracted_frames': metadata['extracted_frames'],
                'original_fps': metadata['original_fps'],
                'target_fps': metadata['target_fps'],
                'shape': str(metadata['shape'])
            })

            if verbose:
                print(f"✓ {video_name}: {metadata['extracted_frames']} frames, type={stroke_type}")

        except Exception as e:
            failed.append({
                'video_path': str(video_path),
                'error': str(e)
            })
            if verbose:
                print(f"✗ {video_path.name}: {e}")

    # Save results
    print(f"\n{'='*60}")
    print(f"EXTRACTION COMPLETE")
    print(f"{'='*60}")
    print(f"Total videos: {len(video_files)}")
    print(f"Successfully processed: {len(results)}")
    print(f"Failed: {len(failed)}")

    if failed:
        failed_path = Path(output_dir) / 'failed_extractions.txt'
        with open(failed_path, 'w') as f:
            for item in failed:
                f.write(f"{item['video_path']}: {item['error']}\n")
        print(f"\nFailed extractions saved to: {failed_path}")

    # Create metadata CSV
    if create_metadata and results:
        df = pd.DataFrame(results)

        # Add some statistics
        print(f"\nStroke Type Distribution:")
        print(df['stroke_type'].value_counts())
        print(f"\nPlayer Distribution:")
        print(df['player_id'].value_counts())

        # Save metadata
        metadata_path = Path(output_dir).parent.parent / 'data' / 'metadata.csv'
        os.makedirs(metadata_path.parent, exist_ok=True)
        df.to_csv(metadata_path, index=False)
        print(f"\n✓ Metadata saved to: {metadata_path}")

        return results, str(metadata_path)

    return results, None


def main():
    parser = argparse.ArgumentParser(
        description='Extract MediaPipe pose sequences from badminton videos',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    parser.add_argument('--video-dir', type=str, required=True,
                       help='Directory containing video clips')
    parser.add_argument('--output-dir', type=str, required=True,
                       help='Directory to save pose pickle files')
    parser.add_argument('--target-fps', type=int, default=30,
                       help='Target FPS for extraction (default: 30)')
    parser.add_argument('--min-confidence', type=float, default=0.5,
                       help='Minimum detection confidence (default: 0.5)')
    parser.add_argument('--create-metadata', action='store_true', default=True,
                       help='Create metadata.csv automatically')
    parser.add_argument('--verbose', action='store_true',
                       help='Show detailed progress')

    args = parser.parse_args()

    print(f"\n{'='*60}")
    print(f"MEDIAPIPE POSE EXTRACTION")
    print(f"{'='*60}")
    print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Video directory: {args.video_dir}")
    print(f"Output directory: {args.output_dir}")
    print(f"Target FPS: {args.target_fps}")
    print(f"Min confidence: {args.min_confidence}")
    print()

    # Process videos
    results, metadata_path = process_all_videos(
        args.video_dir,
        args.output_dir,
        target_fps=args.target_fps,
        min_confidence=args.min_confidence,
        create_metadata=args.create_metadata,
        verbose=args.verbose
    )

    print(f"\nFinished: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"\nNext steps:")
    print(f"  1. Review metadata: {metadata_path}")
    print(f"  2. Run feature engineering: python scripts/run_feature_selection.py")
    print(f"  3. Validate Phase 2: python scripts/validate_phase2.py")
    print()


if __name__ == '__main__':
    main()
