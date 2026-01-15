#!/usr/bin/env python3
"""
Diagnose Failed Pose Extractions
=================================
Investigate why 1858 clips failed pose extraction and provide fixes.

Author: ITI123 Project
Date: January 2026
"""

import cv2
import numpy as np
import pandas as pd
import mediapipe as mp
from pathlib import Path
from tqdm import tqdm
import pickle

# Paths
BASE_DIR = Path(__file__).resolve().parents[2]
CLIPS_DIR = BASE_DIR / "data" / "processed" / "clips"
POSES_DIR = BASE_DIR / "data" / "processed" / "poses"
REPORTS_DIR = BASE_DIR / "outputs" / "reports"
LOG_FILE = REPORTS_DIR / "pose_extraction_log.csv"

# Very lenient detection settings for diagnosis
ULTRA_LOW_DETECTION_CONF = 0.1
ULTRA_LOW_TRACKING_CONF = 0.1


def sample_frames_from_video(video_path, num_samples=5):
    """Extract sample frames from video for visual inspection."""
    cap = cv2.VideoCapture(str(video_path))

    if not cap.isOpened():
        return []

    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # Sample frames evenly distributed
    frame_indices = np.linspace(0, frame_count-1, num_samples, dtype=int)

    frames = []
    for idx in frame_indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        if ret:
            frames.append(frame)

    cap.release()
    return frames


def try_ultra_lenient_detection(video_path):
    """
    Try pose detection with ULTRA LENIENT settings.

    This will help us understand if:
    - Poses can be detected with very low thresholds
    - Video is completely unsuitable for pose estimation
    """
    mp_pose = mp.solutions.pose

    # Ultra-lenient detector
    pose = mp_pose.Pose(
        static_image_mode=False,
        model_complexity=1,  # Use lighter model for speed
        smooth_landmarks=True,
        min_detection_confidence=ULTRA_LOW_DETECTION_CONF,
        min_tracking_confidence=ULTRA_LOW_TRACKING_CONF
    )

    cap = cv2.VideoCapture(str(video_path))

    if not cap.isOpened():
        pose.close()
        return False, 0, 0

    total_frames = 0
    detected_frames = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        total_frames += 1
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        results = pose.process(frame_rgb)

        if results.pose_landmarks:
            detected_frames += 1

    cap.release()
    pose.close()

    can_detect = detected_frames > 0
    detection_rate = detected_frames / total_frames if total_frames > 0 else 0

    return can_detect, detected_frames, total_frames


def diagnose_failed_clips(sample_size=50):
    """
    Diagnose a sample of failed clips to understand the issues.

    Args:
        sample_size: Number of failed clips to diagnose
    """
    print("=" * 70)
    print("DIAGNOSING FAILED POSE EXTRACTIONS")
    print("=" * 70)
    print()

    # Load log
    df = pd.read_csv(LOG_FILE)
    failed = df[df['status'] == 'extraction_failed']

    print(f"Total failed clips: {len(failed)}")
    print()

    # Sample from different matches
    failed['match_id'] = failed['clip_name'].str.extract(r'^(\d+)_')[0].astype(int)

    # Get top problematic matches
    match_failure_counts = failed['match_id'].value_counts()
    top_matches = match_failure_counts.head(10).index.tolist()

    print(f"Top 10 matches with failures:")
    for match_id in top_matches:
        count = match_failure_counts[match_id]
        print(f"  Match {match_id:02d}: {count} failures")
    print()

    # Sample clips from top matches
    sample_clips = []
    clips_per_match = max(1, sample_size // len(top_matches))

    for match_id in top_matches:
        match_failed = failed[failed['match_id'] == match_id]
        sample = match_failed.sample(min(clips_per_match, len(match_failed)))
        sample_clips.extend(sample['clip_name'].tolist())

    print(f"Diagnosing {len(sample_clips)} sample clips...")
    print()

    # Diagnose each clip
    diagnosis_results = []

    for clip_name in tqdm(sample_clips, desc="Diagnosing"):
        clip_path = CLIPS_DIR / clip_name

        if not clip_path.exists():
            diagnosis_results.append({
                'clip_name': clip_name,
                'issue': 'file_not_found'
            })
            continue

        # Try ultra-lenient detection
        can_detect, detected_frames, total_frames = try_ultra_lenient_detection(clip_path)

        detection_rate = detected_frames / total_frames if total_frames > 0 else 0

        diagnosis_results.append({
            'clip_name': clip_name,
            'can_detect_ultra_lenient': can_detect,
            'detection_rate_ultra_lenient': detection_rate * 100,
            'detected_frames': detected_frames,
            'total_frames': total_frames,
            'issue': 'no_detection' if not can_detect else 'low_confidence'
        })

    # Analyze results
    diagnosis_df = pd.DataFrame(diagnosis_results)

    print()
    print("=" * 70)
    print("DIAGNOSIS RESULTS")
    print("=" * 70)
    print()

    # Count issues
    no_detection = len(diagnosis_df[diagnosis_df['issue'] == 'no_detection'])
    low_confidence = len(diagnosis_df[diagnosis_df['issue'] == 'low_confidence'])

    print(f"Sample size: {len(diagnosis_df)}")
    print()
    print("Issue breakdown:")
    print(f"  Cannot detect even with ultra-low thresholds: {no_detection}")
    print(f"  Can detect with ultra-low thresholds: {low_confidence}")
    print()

    if low_confidence > 0:
        print(f"✅ GOOD NEWS: {low_confidence}/{len(diagnosis_df)} clips CAN be detected with lower thresholds!")
        print()
        print("   Average detection rate with ultra-low thresholds:")
        avg_rate = diagnosis_df[diagnosis_df['issue'] == 'low_confidence']['detection_rate_ultra_lenient'].mean()
        print(f"   {avg_rate:.1f}%")
        print()

    if no_detection > 0:
        print(f"⚠️  PROBLEM: {no_detection}/{len(diagnosis_df)} clips cannot be detected even with ultra-low thresholds")
        print()
        print("   Possible reasons:")
        print("   - Camera angle too extreme (overhead/bird's eye view)")
        print("   - Players too small/far away in frame")
        print("   - Heavy occlusions or visual artifacts")
        print("   - Video corruption or quality issues")
        print()

    # Save diagnosis
    diagnosis_file = REPORTS_DIR / "pose_diagnosis.csv"
    diagnosis_df.to_csv(diagnosis_file, index=False)
    print(f"📊 Diagnosis saved to: {diagnosis_file}")
    print()

    # Recommendations
    print("=" * 70)
    print("RECOMMENDATIONS")
    print("=" * 70)
    print()

    if low_confidence > 0:
        pct_fixable = low_confidence / len(diagnosis_df) * 100
        total_estimated_fixable = int(len(failed) * (low_confidence / len(diagnosis_df)))

        print(f"Estimated {total_estimated_fixable} / {len(failed)} failed clips can be fixed")
        print(f"({pct_fixable:.1f}% of failures)")
        print()
        print("Solution:")
        print("  1. Run extract_poses_robust.py with ultra-low thresholds")
        print("  2. This will reprocess failed clips with detection_conf=0.1")
        print()

    if no_detection > 0:
        pct_unfixable = no_detection / len(diagnosis_df) * 100
        total_estimated_unfixable = int(len(failed) * (no_detection / len(diagnosis_df)))

        print(f"Estimated {total_estimated_unfixable} / {len(failed)} failed clips are unfixable")
        print(f"({pct_unfixable:.1f}% of failures)")
        print()
        print("Options:")
        print("  1. Exclude these clips from training (recommended)")
        print("  2. Manually review videos to understand why")
        print("  3. Consider alternative pose estimation methods")
        print()

    return diagnosis_df


def main():
    """Main function"""
    print("\n")
    print("*" * 70)
    print("*" + " " * 68 + "*")
    print("*" + "  ITI123 Project: Diagnose Failed Pose Extractions".center(68) + "*")
    print("*" + " " * 68 + "*")
    print("*" * 70)
    print("\n")

    diagnose_failed_clips(sample_size=100)

    print("=" * 70)
    print("✅ DIAGNOSIS COMPLETE!")
    print("=" * 70)
    print()


if __name__ == "__main__":
    main()
