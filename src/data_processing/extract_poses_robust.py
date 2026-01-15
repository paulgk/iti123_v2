#!/usr/bin/env python3
"""
Robust Pose Estimation Script (Memory Efficient)
=================================================
Enhanced version with multiple fallback strategies for difficult videos.
Processes clips in batches to manage memory usage.

Improvements over extract_poses.py:
1. Lower detection thresholds for difficult videos
2. Frame preprocessing (brightness/contrast adjustment)
3. Multiple detection attempts per frame
4. Better handling of occlusions and poor lighting
5. BATCH PROCESSING to manage memory

Author: ITI123 Project
Date: January 2026
"""

import os
import gc
import sys
import pandas as pd
import cv2
import numpy as np
import mediapipe as mp
from pathlib import Path
from tqdm import tqdm
import pickle

# =============================================================================
# PATH CONFIGURATION
# =============================================================================
BASE_DIR = Path(__file__).resolve().parents[2]
CLIPS_DIR = BASE_DIR / "data" / "processed" / "clips"
POSES_DIR = BASE_DIR / "data" / "processed" / "poses"
REPORTS_DIR = BASE_DIR / "outputs" / "reports"
CLIPS_METADATA_FILE = CLIPS_DIR / "clips_metadata.csv"
LOW_QUALITY_LOG = REPORTS_DIR / "pose_extraction_log.csv"
PROGRESS_FILE = REPORTS_DIR / "robust_extraction_progress.csv"

# =============================================================================
# ROBUST DETECTION CONFIGURATION
# =============================================================================
# Primary detection (stricter - for good quality videos)
PRIMARY_MODEL_COMPLEXITY = 1      # Reduced from 2 to save memory
PRIMARY_DETECTION_CONFIDENCE = 0.5
PRIMARY_TRACKING_CONFIDENCE = 0.5

# Fallback detection (more lenient - for difficult videos)
FALLBACK_MODEL_COMPLEXITY = 1     # Reduced from 2 to save memory
FALLBACK_DETECTION_CONFIDENCE = 0.3
FALLBACK_TRACKING_CONFIDENCE = 0.3

# Ultra fallback (very lenient)
ULTRA_MODEL_COMPLEXITY = 1
ULTRA_DETECTION_CONFIDENCE = 0.15
ULTRA_TRACKING_CONFIDENCE = 0.15

# Quality thresholds
MIN_KEYPOINT_CONFIDENCE = 0.2
MIN_VALID_FRAMES_PERCENTAGE = 30

# =============================================================================
# MEMORY MANAGEMENT CONFIGURATION
# =============================================================================
BATCH_SIZE = 100              # Process N clips per batch
GC_EVERY_N_CLIPS = 10         # Garbage collect every N clips within batch
ENABLE_PREPROCESSING = False  # Disable heavy preprocessing by default

# Temporal window for stroke detection
STROKE_WINDOW_START = 0
STROKE_WINDOW_END = 45

# MediaPipe landmarks
LANDMARK_LEFT_SHOULDER = 11
LANDMARK_RIGHT_SHOULDER = 12
LANDMARK_LEFT_ELBOW = 13
LANDMARK_RIGHT_ELBOW = 14
LANDMARK_LEFT_WRIST = 15
LANDMARK_RIGHT_WRIST = 16


# =============================================================================
# FRAME PREPROCESSING (Lightweight version)
# =============================================================================
def enhance_frame_light(frame):
    """
    Lightweight frame enhancement (less memory intensive).

    Uses only CLAHE without heavy noise reduction.

    Args:
        frame: BGR frame

    Returns:
        enhanced_frame: Enhanced BGR frame
    """
    # Convert to LAB color space
    lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)

    # Apply CLAHE to L channel (brightness)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    l_enhanced = clahe.apply(l)

    # Merge channels
    lab_enhanced = cv2.merge([l_enhanced, a, b])

    # Convert back to BGR
    enhanced = cv2.cvtColor(lab_enhanced, cv2.COLOR_LAB2BGR)

    return enhanced


# =============================================================================
# POSE EXTRACTION FUNCTIONS
# =============================================================================
def extract_poses_single_pass(video_path, pose_detector, preprocess=False):
    """
    Extract poses from video with a single detector pass.

    Memory efficient: processes frame by frame without storing all frames.

    Args:
        video_path: Path to video
        pose_detector: Initialized MediaPipe Pose object
        preprocess: Whether to apply frame preprocessing

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

        # Lightweight preprocessing if enabled
        if preprocess:
            frame = enhance_frame_light(frame)

        # Convert to RGB
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Detect pose
        results = pose_detector.process(frame_rgb)

        if results.pose_landmarks:
            # Extract landmarks
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

        # Clear frame from memory
        del frame, frame_rgb

    cap.release()

    # Calculate statistics
    valid_frames = sum(1 for p in poses if p is not None)
    avg_conf = np.mean([c for c in confidences if c > 0]) if any(c > 0 for c in confidences) else 0.0

    return poses, {
        'total_frames': len(poses),
        'valid_frames': valid_frames,
        'avg_confidence': float(avg_conf)
    }


def process_single_clip(clip_path, stroke_type):
    """
    Process a single clip with multiple fallback strategies.

    Creates and destroys detectors within function to manage memory.

    Args:
        clip_path: Path to video clip
        stroke_type: 'Clear' or 'Smash'

    Returns:
        pose_data: Dictionary with poses and metadata
        success: Boolean indicating if extraction met quality threshold
    """
    mp_pose = mp.solutions.pose
    best_result = None
    best_valid_pct = 0

    # =========================================================================
    # ATTEMPT 1: Fallback thresholds (skip primary since these are failed clips)
    # =========================================================================
    detector = mp_pose.Pose(
        static_image_mode=False,
        model_complexity=FALLBACK_MODEL_COMPLEXITY,
        smooth_landmarks=True,
        min_detection_confidence=FALLBACK_DETECTION_CONFIDENCE,
        min_tracking_confidence=FALLBACK_TRACKING_CONFIDENCE
    )

    poses, stats = extract_poses_single_pass(clip_path, detector, preprocess=False)
    detector.close()
    del detector
    gc.collect()

    valid_pct = (stats['valid_frames'] / stats['total_frames'] * 100
                 if stats['total_frames'] > 0 else 0)

    if valid_pct > best_valid_pct:
        best_valid_pct = valid_pct
        best_result = (poses, stats, 'fallback_low_threshold')

    # If good enough, return early
    if valid_pct >= 50:
        return {
            'poses': poses,
            'method': 'fallback_low_threshold',
            **stats
        }, True

    # =========================================================================
    # ATTEMPT 2: Ultra-low thresholds
    # =========================================================================
    detector = mp_pose.Pose(
        static_image_mode=False,
        model_complexity=ULTRA_MODEL_COMPLEXITY,
        smooth_landmarks=True,
        min_detection_confidence=ULTRA_DETECTION_CONFIDENCE,
        min_tracking_confidence=ULTRA_TRACKING_CONFIDENCE
    )

    poses, stats = extract_poses_single_pass(clip_path, detector, preprocess=False)
    detector.close()
    del detector
    gc.collect()

    valid_pct = (stats['valid_frames'] / stats['total_frames'] * 100
                 if stats['total_frames'] > 0 else 0)

    if valid_pct > best_valid_pct:
        best_valid_pct = valid_pct
        best_result = (poses, stats, 'ultra_low_threshold')

    # If good enough, return early
    if valid_pct >= 50:
        return {
            'poses': poses,
            'method': 'ultra_low_threshold',
            **stats
        }, True

    # =========================================================================
    # ATTEMPT 3: Ultra-low + preprocessing (only if enabled)
    # =========================================================================
    if ENABLE_PREPROCESSING and best_valid_pct < 30:
        detector = mp_pose.Pose(
            static_image_mode=False,
            model_complexity=ULTRA_MODEL_COMPLEXITY,
            smooth_landmarks=True,
            min_detection_confidence=ULTRA_DETECTION_CONFIDENCE,
            min_tracking_confidence=ULTRA_TRACKING_CONFIDENCE
        )

        poses, stats = extract_poses_single_pass(clip_path, detector, preprocess=True)
        detector.close()
        del detector
        gc.collect()

        valid_pct = (stats['valid_frames'] / stats['total_frames'] * 100
                     if stats['total_frames'] > 0 else 0)

        if valid_pct > best_valid_pct:
            best_valid_pct = valid_pct
            best_result = (poses, stats, 'ultra_preprocessed')

    # Return best result
    if best_result is not None:
        poses, stats, method = best_result
        return {
            'poses': poses,
            'method': method,
            **stats
        }, best_valid_pct >= MIN_VALID_FRAMES_PERCENTAGE

    # No detection at all
    return {
        'poses': [],
        'method': 'none',
        'total_frames': 0,
        'valid_frames': 0,
        'avg_confidence': 0.0
    }, False


# =============================================================================
# BATCH PROCESSING
# =============================================================================
def process_batch(batch_df, batch_num, total_batches):
    """
    Process a batch of clips.

    Args:
        batch_df: DataFrame with clips to process
        batch_num: Current batch number (1-indexed)
        total_batches: Total number of batches

    Returns:
        results: List of result dictionaries
        stats: Dictionary with batch statistics
    """
    print(f"\n{'='*60}")
    print(f"BATCH {batch_num}/{total_batches} ({len(batch_df)} clips)")
    print(f"{'='*60}")

    results = []
    fixed = 0
    partial = 0
    failed = 0

    for idx, (_, row) in enumerate(tqdm(batch_df.iterrows(),
                                         total=len(batch_df),
                                         desc=f"Batch {batch_num}")):

        clip_name = row['clip_name']
        stroke_type = row['stroke_type']
        clip_path = CLIPS_DIR / clip_name
        pose_path = POSES_DIR / clip_name.replace('.mp4', '_pose.pkl')

        if not clip_path.exists():
            results.append({
                'clip_name': clip_name,
                'stroke_type': stroke_type,
                'status': 'file_not_found'
            })
            continue

        try:
            # Process clip
            pose_data, success = process_single_clip(clip_path, stroke_type)

            valid_pct = (pose_data['valid_frames'] / pose_data['total_frames'] * 100
                        if pose_data['total_frames'] > 0 else 0)

            # Save result
            with open(pose_path, 'wb') as f:
                pickle.dump(pose_data, f)

            # Determine status
            old_valid_pct = row.get('valid_percentage', 0)
            if pd.isna(old_valid_pct):
                old_valid_pct = 0

            if valid_pct >= 50:
                status = 'fixed'
                fixed += 1
            elif valid_pct > 0:
                status = 'partial'
                partial += 1
            else:
                status = 'still_failed'
                failed += 1

            results.append({
                'clip_name': clip_name,
                'stroke_type': stroke_type,
                'old_valid_pct': old_valid_pct,
                'new_valid_pct': valid_pct,
                'method': pose_data['method'],
                'status': status
            })

        except Exception as e:
            results.append({
                'clip_name': clip_name,
                'stroke_type': stroke_type,
                'error': str(e),
                'status': 'error'
            })
            failed += 1

        # Garbage collection within batch
        if (idx + 1) % GC_EVERY_N_CLIPS == 0:
            gc.collect()

    # Force garbage collection after batch
    gc.collect()

    print(f"\nBatch {batch_num} complete: Fixed={fixed}, Partial={partial}, Failed={failed}")

    return results, {'fixed': fixed, 'partial': partial, 'failed': failed}


def reprocess_failed_and_low_quality_clips(min_valid_percentage=50, start_batch=1):
    """
    Reprocess clips in batches with memory management.

    Args:
        min_valid_percentage: Minimum valid frame percentage threshold
        start_batch: Batch number to start from (for resuming)
    """
    print("=" * 70)
    print("ROBUST POSE EXTRACTION (BATCH MODE)")
    print("=" * 70)
    print()
    print(f"Configuration:")
    print(f"  Batch size: {BATCH_SIZE} clips")
    print(f"  GC interval: every {GC_EVERY_N_CLIPS} clips")
    print(f"  Model complexity: {FALLBACK_MODEL_COMPLEXITY}")
    print(f"  Preprocessing: {'Enabled' if ENABLE_PREPROCESSING else 'Disabled'}")
    print()

    # Load previous extraction log
    if not LOW_QUALITY_LOG.exists():
        print(f"❌ ERROR: {LOW_QUALITY_LOG} not found!")
        return

    log_df = pd.read_csv(LOW_QUALITY_LOG)

    # Find clips to reprocess
    clips_to_reprocess = log_df[
        (log_df['status'] == 'extraction_failed') |
        (log_df['valid_percentage'] < min_valid_percentage)
    ].copy()

    failed_count = len(log_df[log_df['status'] == 'extraction_failed'])
    low_quality_count = len(clips_to_reprocess) - failed_count

    print(f"Clips to reprocess:")
    print(f"  Failed extractions: {failed_count}")
    print(f"  Low quality (<{min_valid_percentage}%): {low_quality_count}")
    print(f"  Total: {len(clips_to_reprocess)}")
    print()

    if len(clips_to_reprocess) == 0:
        print("✅ No clips to reprocess!")
        return

    # Split into batches
    num_batches = (len(clips_to_reprocess) + BATCH_SIZE - 1) // BATCH_SIZE
    print(f"Processing in {num_batches} batches of {BATCH_SIZE} clips each")
    print()

    # Overall statistics
    total_fixed = 0
    total_partial = 0
    total_failed = 0
    all_results = []

    # Process each batch
    for batch_num in range(start_batch, num_batches + 1):
        start_idx = (batch_num - 1) * BATCH_SIZE
        end_idx = min(batch_num * BATCH_SIZE, len(clips_to_reprocess))

        batch_df = clips_to_reprocess.iloc[start_idx:end_idx]

        # Process batch
        results, stats = process_batch(batch_df, batch_num, num_batches)

        # Accumulate results
        all_results.extend(results)
        total_fixed += stats['fixed']
        total_partial += stats['partial']
        total_failed += stats['failed']

        # Save progress after each batch
        progress_df = pd.DataFrame(all_results)
        progress_df.to_csv(PROGRESS_FILE, index=False)
        print(f"Progress saved to: {PROGRESS_FILE}")

        # Force garbage collection between batches
        gc.collect()

        # Print running totals
        print(f"\nRunning total: Fixed={total_fixed}, Partial={total_partial}, Failed={total_failed}")

    # Final summary
    print()
    print("=" * 70)
    print("FINAL SUMMARY")
    print("=" * 70)
    print(f"Total clips processed: {len(clips_to_reprocess)}")
    print(f"✅ Fixed (>=50% valid): {total_fixed}")
    print(f"⚙️  Partial (>0% valid): {total_partial}")
    print(f"❌ Still failed: {total_failed}")
    print()

    success_rate = (total_fixed + total_partial) / len(clips_to_reprocess) * 100
    print(f"Success rate: {success_rate:.1f}%")
    print()

    # Save final results
    final_log = REPORTS_DIR / "pose_reprocessing_log.csv"
    results_df = pd.DataFrame(all_results)
    results_df.to_csv(final_log, index=False)
    print(f"📊 Final results saved to: {final_log}")


# =============================================================================
# MAIN FUNCTION
# =============================================================================
def main():
    """Main function with optional batch resume"""
    print("\n")
    print("*" * 70)
    print("*" + " " * 68 + "*")
    print("*" + "  ITI123 Project: Robust Pose Extraction".center(68) + "*")
    print("*" + "        (Memory Efficient Batch Mode)".center(68) + "*")
    print("*" + " " * 68 + "*")
    print("*" * 70)
    print("\n")

    # Check for resume argument
    start_batch = 1
    if len(sys.argv) > 1:
        try:
            start_batch = int(sys.argv[1])
            print(f"Resuming from batch {start_batch}")
        except ValueError:
            print(f"Invalid batch number: {sys.argv[1]}")
            return

    print("This script reprocesses FAILED and low quality clips with:")
    print("  1. Lower confidence thresholds (0.5 → 0.3 → 0.15)")
    print("  2. Batch processing for memory efficiency")
    print("  3. Automatic progress saving after each batch")
    print()
    print("To resume from a specific batch:")
    print("  python extract_poses_robust.py <batch_number>")
    print()

    reprocess_failed_and_low_quality_clips(min_valid_percentage=50, start_batch=start_batch)

    print("=" * 70)
    print("✅ ROBUST POSE EXTRACTION COMPLETE!")
    print("=" * 70)
    print()


if __name__ == "__main__":
    main()
