#!/usr/bin/env python3
"""
Simple Robust Pose Extraction (Memory Efficient)
=================================================
Reprocess failed clips with lower thresholds.
Uses single detector instance and minimal memory.

Author: ITI123 Project
Date: January 2026
"""

import gc
import pandas as pd
import cv2
import numpy as np
import mediapipe as mp
from pathlib import Path
from tqdm import tqdm
import pickle

# =============================================================================
# PATHS
# =============================================================================
BASE_DIR = Path(__file__).resolve().parents[2]
CLIPS_DIR = BASE_DIR / "data" / "processed" / "clips"
POSES_DIR = BASE_DIR / "data" / "processed" / "poses"
REPORTS_DIR = BASE_DIR / "outputs" / "reports"
LOG_FILE = REPORTS_DIR / "pose_extraction_log.csv"

# =============================================================================
# CONFIGURATION - Use lower thresholds for failed clips
# =============================================================================
MODEL_COMPLEXITY = 1          # Use lighter model (1 instead of 2) - saves memory
DETECTION_CONFIDENCE = 0.2    # Very low threshold
TRACKING_CONFIDENCE = 0.2     # Very low threshold
MIN_KEYPOINT_CONFIDENCE = 0.1

# Memory management
GC_EVERY_N_CLIPS = 20  # Force garbage collection every N clips


def extract_single_video(video_path, pose_detector):
    """
    Extract poses from a single video clip.

    Args:
        video_path: Path to video
        pose_detector: Initialized MediaPipe Pose detector

    Returns:
        poses: List of pose arrays (None for frames with no detection)
        stats: Dictionary with extraction statistics
    """
    cap = cv2.VideoCapture(str(video_path))

    if not cap.isOpened():
        return [], {'total_frames': 0, 'valid_frames': 0, 'avg_confidence': 0.0}

    poses = []
    confidences = []

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Convert to RGB (MediaPipe requirement)
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Detect pose
        results = pose_detector.process(frame_rgb)

        if results.pose_landmarks:
            # Extract landmarks as numpy array
            landmarks = []
            frame_conf = []

            for lm in results.pose_landmarks.landmark:
                landmarks.append([lm.x, lm.y, lm.z, lm.visibility])
                frame_conf.append(lm.visibility)

            poses.append(np.array(landmarks))
            confidences.append(np.mean(frame_conf))
        else:
            poses.append(None)
            confidences.append(0.0)

    cap.release()

    # Calculate statistics
    valid_frames = sum(1 for p in poses if p is not None)
    avg_conf = np.mean([c for c in confidences if c > 0]) if any(c > 0 for c in confidences) else 0.0

    return poses, {
        'total_frames': len(poses),
        'valid_frames': valid_frames,
        'avg_confidence': float(avg_conf)
    }


def main():
    """Main function - reprocess failed clips"""

    print("\n")
    print("=" * 60)
    print("SIMPLE ROBUST POSE EXTRACTION")
    print("=" * 60)
    print()
    print("Settings:")
    print(f"  Model complexity: {MODEL_COMPLEXITY} (lighter = less memory)")
    print(f"  Detection confidence: {DETECTION_CONFIDENCE}")
    print(f"  Tracking confidence: {TRACKING_CONFIDENCE}")
    print()

    # Load extraction log
    if not LOG_FILE.exists():
        print(f"ERROR: {LOG_FILE} not found!")
        return

    log_df = pd.read_csv(LOG_FILE)

    # Find failed clips
    failed_clips = log_df[log_df['status'] == 'extraction_failed'].copy()

    print(f"Total failed clips to reprocess: {len(failed_clips)}")
    print()

    if len(failed_clips) == 0:
        print("No failed clips to reprocess!")
        return

    # Create single detector instance (memory efficient)
    print("Initializing MediaPipe Pose detector...")
    mp_pose = mp.solutions.pose
    pose_detector = mp_pose.Pose(
        static_image_mode=False,
        model_complexity=MODEL_COMPLEXITY,
        smooth_landmarks=True,
        min_detection_confidence=DETECTION_CONFIDENCE,
        min_tracking_confidence=TRACKING_CONFIDENCE
    )
    print("Detector ready.")
    print()

    # Statistics
    fixed_count = 0
    partial_count = 0
    still_failed = 0
    results_log = []

    # Process clips
    print("Processing clips...")
    for idx, (_, row) in enumerate(tqdm(failed_clips.iterrows(),
                                         total=len(failed_clips),
                                         desc="Reprocessing")):

        clip_name = row['clip_name']
        stroke_type = row['stroke_type']
        clip_path = CLIPS_DIR / clip_name
        pose_path = POSES_DIR / clip_name.replace('.mp4', '_pose.pkl')

        if not clip_path.exists():
            continue

        try:
            # Extract poses
            poses, stats = extract_single_video(clip_path, pose_detector)

            valid_pct = (stats['valid_frames'] / stats['total_frames'] * 100
                        if stats['total_frames'] > 0 else 0)

            # Save if any improvement
            if stats['valid_frames'] > 0:
                pose_data = {
                    'poses': poses,
                    'total_frames': stats['total_frames'],
                    'valid_frames': stats['valid_frames'],
                    'avg_confidence': stats['avg_confidence'],
                    'method': 'simple_low_threshold'
                }

                with open(pose_path, 'wb') as f:
                    pickle.dump(pose_data, f)

            # Track results
            if valid_pct >= 50:
                status = 'fixed'
                fixed_count += 1
            elif valid_pct > 0:
                status = 'partial'
                partial_count += 1
            else:
                status = 'still_failed'
                still_failed += 1

            results_log.append({
                'clip_name': clip_name,
                'stroke_type': stroke_type,
                'valid_pct': valid_pct,
                'valid_frames': stats['valid_frames'],
                'total_frames': stats['total_frames'],
                'status': status
            })

        except Exception as e:
            results_log.append({
                'clip_name': clip_name,
                'stroke_type': stroke_type,
                'error': str(e),
                'status': 'error'
            })

        # Garbage collection every N clips
        if (idx + 1) % GC_EVERY_N_CLIPS == 0:
            gc.collect()

    # Close detector
    pose_detector.close()
    gc.collect()

    # Save results log
    results_df = pd.DataFrame(results_log)
    results_file = REPORTS_DIR / "simple_reprocessing_log.csv"
    results_df.to_csv(results_file, index=False)

    # Print summary
    print()
    print("=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"Total processed: {len(failed_clips)}")
    print(f"✅ Fixed (>=50% valid): {fixed_count}")
    print(f"⚙️  Partial (>0% valid): {partial_count}")
    print(f"❌ Still failed: {still_failed}")
    print()

    success_rate = (fixed_count + partial_count) / len(failed_clips) * 100 if len(failed_clips) > 0 else 0
    print(f"Success rate: {success_rate:.1f}%")
    print()
    print(f"Results saved to: {results_file}")
    print()


if __name__ == "__main__":
    main()
