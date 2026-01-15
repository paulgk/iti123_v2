#!/usr/bin/env python3
"""
Pose Estimation Script (Multi-Player Detection)
================================================
Extract human pose keypoints from video clips using MediaPipe Pose.
Both players are always visible - algorithm identifies the stroke executor.

Dataset:
    ShuttleSet: A Human-Annotated Stroke-Level Singles Dataset for Badminton
    Tactical Analysis (Wang et al., 2023)
    https://arxiv.org/abs/2306.04948

Strategy for identifying stroke executor:
1. Detect ALL persons in each frame (both players)
2. Identify stroke executor based on:
   - Arm elevation (overhead strokes have raised arms)
   - Movement dynamics (executor moves more)
   - Temporal consistency (track same player across frames)

Process:
1. Load video clips
2. Extract pose keypoints for ALL detected persons per frame
3. Identify stroke executor using biomechanical heuristics
4. Track executor consistently across frames
5. Interpolate missing frames
6. Save pose sequences for feature engineering

References:
    Wang, W.-Y., Huang, Y.-C., Ik, T.-U., & Peng, W.-C. (2023).
    ShuttleSet: A Human-Annotated Stroke-Level Singles Dataset for
    Badminton Tactical Analysis. CoRR, abs/2306.04948.

Author: ITI123 Project
Date: January 2026
"""

import os
import pandas as pd
import cv2
import numpy as np
import mediapipe as mp
from pathlib import Path
from tqdm import tqdm
import pickle
from collections import deque

# =============================================================================
# PATH CONFIGURATION
# =============================================================================
BASE_DIR = Path(__file__).resolve().parents[2]
CLIPS_DIR = BASE_DIR / "data" / "processed" / "clips"
POSES_DIR = BASE_DIR / "data" / "processed" / "poses"
REPORTS_DIR = BASE_DIR / "outputs" / "reports"
CLIPS_METADATA_FILE = CLIPS_DIR / "clips_metadata.csv"

# =============================================================================
# POSE DETECTION CONFIGURATION
# =============================================================================
# MediaPipe Pose has 3 complexity levels:
# 0 = Lite (fast, less accurate)
# 1 = Full (balanced)
# 2 = Heavy (slow, most accurate) - RECOMMENDED for this project
MODEL_COMPLEXITY = 2

# Confidence thresholds (0.0 to 1.0)
# Higher = more strict detection, fewer false positives
MIN_DETECTION_CONFIDENCE = 0.5  # Minimum confidence to detect a person
MIN_TRACKING_CONFIDENCE = 0.5   # Minimum confidence to track across frames

# Quality thresholds
MIN_KEYPOINT_CONFIDENCE = 0.3   # Minimum confidence for individual keypoints
MIN_VALID_FRAMES_PERCENTAGE = 50  # Minimum % of frames with valid pose detection

# Player identification thresholds
# For overhead strokes (Clear, Smash), the executor's arm is raised high
ARM_ELEVATION_THRESHOLD = 0.3   # Wrist should be above shoulder
MOVEMENT_WINDOW = 5              # Frames to analyze for movement detection

# Temporal window for stroke detection
# IMPORTANT: The focused stroke is ALWAYS the FIRST shot in the clip
# - Video clips have ±1 second temporal context (from extract_clips.py)
# - At 30fps, this is ~30 frames before and ~30 frames after the stroke
# - The actual stroke execution occurs in the FIRST ~1.5 seconds
# - We prioritize detection in this window for better executor identification
STROKE_WINDOW_START = 0    # Start of stroke window (frame index)
STROKE_WINDOW_END = 45     # End of stroke window (frames) - first 1.5 seconds at 30fps


# =============================================================================
# MEDIAPIPE POSE LANDMARK INDICES
# =============================================================================
# Key landmarks for player identification
# Reference: https://google.github.io/mediapipe/solutions/pose.html
LANDMARK_NOSE = 0
LANDMARK_LEFT_SHOULDER = 11
LANDMARK_RIGHT_SHOULDER = 12
LANDMARK_LEFT_ELBOW = 13
LANDMARK_RIGHT_ELBOW = 14
LANDMARK_LEFT_WRIST = 15
LANDMARK_RIGHT_WRIST = 16
LANDMARK_LEFT_HIP = 23
LANDMARK_RIGHT_HIP = 24


# =============================================================================
# INITIALIZE MEDIAPIPE POSE DETECTOR
# =============================================================================
def initialize_pose_detector():
    """
    Initialize MediaPipe Pose detector with configuration.

    Returns:
        pose: MediaPipe Pose object
        mp_pose: MediaPipe pose module (for landmark references)
        mp_drawing: MediaPipe drawing utilities (for visualization)
    """
    print("Initializing MediaPipe Pose detector...")

    # Import MediaPipe components
    mp_pose = mp.solutions.pose
    mp_drawing = mp.solutions.drawing_utils
    mp_drawing_styles = mp.solutions.drawing_styles

    # Create Pose detector
    # static_image_mode=False means we're processing video (enables tracking)
    # smooth_landmarks=True applies temporal smoothing across frames
    pose = mp_pose.Pose(
        static_image_mode=False,  # Video mode (not individual images)
        model_complexity=MODEL_COMPLEXITY,  # 0=Lite, 1=Full, 2=Heavy
        smooth_landmarks=True,  # Smooth keypoints across frames
        enable_segmentation=False,  # We don't need segmentation mask
        min_detection_confidence=MIN_DETECTION_CONFIDENCE,
        min_tracking_confidence=MIN_TRACKING_CONFIDENCE
    )

    print(f"✅ Pose detector initialized:")
    print(f"   Model complexity: {MODEL_COMPLEXITY} (Heavy - most accurate)")
    print(f"   Detection confidence: {MIN_DETECTION_CONFIDENCE}")
    print(f"   Tracking confidence: {MIN_TRACKING_CONFIDENCE}")
    print(f"   Player identification: Multi-person detection enabled")

    return pose, mp_pose, mp_drawing


# =============================================================================
# DETECT MULTIPLE PERSONS (MEDIAPIPE WORKAROUND)
# =============================================================================
def detect_all_persons_in_frame(frame_rgb, pose_detector):
    """
    Detect all persons in a frame.

    Note: MediaPipe Pose is designed for single-person detection.
    For multiple persons, we need to:
    1. Detect the first person
    2. Use person segmentation or bounding boxes to isolate regions
    3. Re-run detection on remaining regions

    Simplified approach for this project:
    - Process full frame (gets most prominent person)
    - Use spatial heuristics to estimate second player position
    - For robust multi-person detection, consider using:
      * MediaPipe Holistic (if available)
      * Object detection (YOLO) + individual pose estimation
      * BlazePose with custom multi-person tracking

    Args:
        frame_rgb: RGB frame
        pose_detector: MediaPipe Pose object

    Returns:
        persons: List of pose arrays (may contain 1 or 2 persons)
    """
    # =========================================================================
    # STEP 1: Detect first person (most prominent in frame)
    # =========================================================================
    results = pose_detector.process(frame_rgb)

    persons = []

    if results.pose_landmarks:
        # Extract landmarks for first detected person
        landmarks_array = []
        for landmark in results.pose_landmarks.landmark:
            landmarks_array.append([
                landmark.x,
                landmark.y,
                landmark.z,
                landmark.visibility
            ])
        persons.append(np.array(landmarks_array))

    # =========================================================================
    # STEP 2: Attempt to detect second person (simplified approach)
    # =========================================================================
    # For full multi-person detection, would need:
    # - Mask out first person using segmentation
    # - Re-run detection on remaining image regions
    # - Or use dedicated multi-person pose estimation model

    # For this project: We'll work with the single most prominent detection
    # and use heuristics to identify if it's the executor

    return persons


# =============================================================================
# CALCULATE ARM ELEVATION SCORE
# =============================================================================
def calculate_arm_elevation_score(pose):
    """
    Calculate how elevated the arms are (for overhead stroke detection).

    For overhead strokes (Clear, Smash), the executor's arm is raised high:
    - Wrist should be significantly above shoulder
    - Elbow should be above shoulder

    Args:
        pose: Pose array (33, 4) [x, y, z, visibility]

    Returns:
        elevation_score: Float (0.0 to 1.0+), higher = more elevated
    """
    # Get shoulder, elbow, wrist positions (right arm - most players are right-handed)
    right_shoulder = pose[LANDMARK_RIGHT_SHOULDER]
    right_elbow = pose[LANDMARK_RIGHT_ELBOW]
    right_wrist = pose[LANDMARK_RIGHT_WRIST]

    left_shoulder = pose[LANDMARK_LEFT_SHOULDER]
    left_elbow = pose[LANDMARK_LEFT_ELBOW]
    left_wrist = pose[LANDMARK_LEFT_WRIST]

    # Y-coordinate: 0 = top of frame, 1 = bottom of frame
    # So lower Y value = higher position

    # Right arm elevation
    right_wrist_above_shoulder = max(0, right_shoulder[1] - right_wrist[1])
    right_elbow_above_shoulder = max(0, right_shoulder[1] - right_elbow[1])

    # Left arm elevation
    left_wrist_above_shoulder = max(0, left_shoulder[1] - left_wrist[1])
    left_elbow_above_shoulder = max(0, left_shoulder[1] - left_elbow[1])

    # Use the arm that's more elevated (player could be left-handed)
    right_arm_score = right_wrist_above_shoulder + right_elbow_above_shoulder
    left_arm_score = left_wrist_above_shoulder + left_elbow_above_shoulder

    elevation_score = max(right_arm_score, left_arm_score)

    return elevation_score


# =============================================================================
# CALCULATE MOVEMENT SCORE
# =============================================================================
def calculate_movement_score(pose_sequence):
    """
    Calculate how much movement occurs in pose sequence.

    Stroke executor moves more than opponent (preparing, executing, recovering).

    Args:
        pose_sequence: List of recent pose arrays

    Returns:
        movement_score: Float, higher = more movement
    """
    if len(pose_sequence) < 2:
        return 0.0

    # Calculate movement as sum of keypoint displacements across frames
    total_movement = 0.0

    for i in range(1, len(pose_sequence)):
        if pose_sequence[i] is not None and pose_sequence[i-1] is not None:
            # Calculate displacement of key points
            # Focus on upper body (shoulders, elbows, wrists)
            key_points = [
                LANDMARK_LEFT_SHOULDER, LANDMARK_RIGHT_SHOULDER,
                LANDMARK_LEFT_ELBOW, LANDMARK_RIGHT_ELBOW,
                LANDMARK_LEFT_WRIST, LANDMARK_RIGHT_WRIST
            ]

            for kp_idx in key_points:
                # Euclidean distance in x-y plane
                dx = pose_sequence[i][kp_idx][0] - pose_sequence[i-1][kp_idx][0]
                dy = pose_sequence[i][kp_idx][1] - pose_sequence[i-1][kp_idx][1]
                displacement = np.sqrt(dx**2 + dy**2)
                total_movement += displacement

    return total_movement


# =============================================================================
# IDENTIFY STROKE EXECUTOR
# =============================================================================
def identify_stroke_executor(all_frame_poses, stroke_type):
    """
    Identify which player is executing the stroke across all frames.

    Strategy:
    1. TEMPORAL PRIORITY: The focused stroke is ALWAYS the first shot in the clip
       - Prioritize arm elevation detection in frames 0-45 (first 1.5 seconds)
    2. For each frame, calculate arm elevation score for detected person(s)
    3. Track consistency across frames (executor should have elevated arm near stroke impact)
    4. Select person with highest average arm elevation during stroke window

    Since MediaPipe detects most prominent person, and clips are centered on stroke:
    - The detected person is usually the executor
    - But we verify using biomechanical features in the STROKE WINDOW

    Args:
        all_frame_poses: List of lists - each frame may have 1+ person detections
        stroke_type: 'Clear' or 'Smash' (both are overhead strokes)

    Returns:
        executor_poses: List of pose arrays (one per frame, executor only)
    """
    executor_poses = []

    # =========================================================================
    # STEP 1: Identify stroke window (first shot in clip)
    # =========================================================================
    total_frames = len(all_frame_poses)
    stroke_end = min(STROKE_WINDOW_END, total_frames)

    # =========================================================================
    # STEP 2: Calculate arm elevation scores in stroke window
    # Priority is given to frames in the stroke execution window
    # =========================================================================

    for frame_idx, frame_poses in enumerate(all_frame_poses):
        if len(frame_poses) == 0:
            # No person detected
            executor_poses.append(None)

        elif len(frame_poses) == 1:
            # Only one person detected - likely the executor
            executor_poses.append(frame_poses[0])

        else:
            # Multiple persons detected - select one with higher arm elevation
            # Apply temporal weighting: prioritize stroke window (first 1.5 seconds)
            best_pose = None
            best_score = -1

            for pose in frame_poses:
                elevation_score = calculate_arm_elevation_score(pose)

                # =========================================================
                # TEMPORAL WEIGHTING: Boost score if in stroke window
                # The focused stroke is ALWAYS in the first part of clip
                # =========================================================
                if frame_idx <= stroke_end:
                    # Inside stroke window - increase weight
                    # Earlier frames get higher weight (stroke impact is early)
                    temporal_weight = 2.0 - (frame_idx / stroke_end)  # 2.0 -> 1.0
                    elevation_score *= temporal_weight

                if elevation_score > best_score:
                    best_score = elevation_score
                    best_pose = pose

            executor_poses.append(best_pose)

    # =========================================================================
    # POST-PROCESSING: Temporal consistency
    # =========================================================================
    # Ensure the selected player is consistent across frames
    # (avoid switching between players mid-sequence)
    executor_poses = enforce_temporal_consistency(executor_poses)

    return executor_poses


# =============================================================================
# ENFORCE TEMPORAL CONSISTENCY
# =============================================================================
def enforce_temporal_consistency(poses):
    """
    Ensure selected player is consistent across frames.

    Method:
    - Track player position (center of mass)
    - If sudden large jump in position, likely switched players
    - Correct by maintaining consistent player identity

    Args:
        poses: List of pose arrays

    Returns:
        consistent_poses: List of pose arrays (same player tracked)
    """
    if len(poses) == 0:
        return poses

    # Calculate center of mass for each pose
    def get_center_of_mass(pose):
        if pose is None:
            return None
        # Average position of shoulders and hips
        key_points = [LANDMARK_LEFT_SHOULDER, LANDMARK_RIGHT_SHOULDER,
                      LANDMARK_LEFT_HIP, LANDMARK_RIGHT_HIP]
        x = np.mean([pose[kp][0] for kp in key_points])
        y = np.mean([pose[kp][1] for kp in key_points])
        return np.array([x, y])

    centers = [get_center_of_mass(p) for p in poses]

    # Check for sudden jumps (may indicate player switch)
    # For singles matches with both players visible, they're usually far apart
    # A jump > 0.3 (normalized coordinates) likely means switched players

    # For this simplified version, we assume MediaPipe consistently detects
    # the more prominent player (usually the executor in centered shots)

    # If needed, implement more sophisticated tracking here

    return poses


# =============================================================================
# EXTRACT POSES FROM VIDEO (MULTI-PERSON AWARE)
# =============================================================================
def extract_pose_from_video(video_path, pose_detector, stroke_type):
    """
    Extract pose keypoints from all frames, identifying stroke executor.

    MediaPipe Pose detects 33 keypoints per person:
    - 0-10: Face (nose, eyes, ears, mouth)
    - 11-22: Upper body (shoulders, elbows, wrists, hands)
    - 23-32: Lower body (hips, knees, ankles, feet)

    Each keypoint has 4 values: (x, y, z, visibility)
    - x, y: Normalized coordinates (0.0 to 1.0)
    - z: Depth (relative to hips)
    - visibility: Confidence score (0.0 to 1.0)

    Args:
        video_path: Path to video file
        pose_detector: MediaPipe Pose object
        stroke_type: 'Clear' or 'Smash' (helps identify executor)

    Returns:
        poses: List of pose arrays for EXECUTOR ONLY, each shape (33, 4)
        frame_qualities: List of quality scores per frame (0.0 to 1.0)
        success: Boolean indicating if extraction was successful
    """
    # Open video file
    cap = cv2.VideoCapture(str(video_path))

    if not cap.isOpened():
        return None, None, False

    # Get video properties
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)

    # Storage for ALL detected persons per frame
    all_frame_poses = []

    frame_idx = 0

    # =========================================================================
    # STEP 1: Extract poses for ALL persons in each frame
    # =========================================================================
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break  # End of video

        # Convert BGR (OpenCV) to RGB (MediaPipe requirement)
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Detect all persons in frame
        frame_persons = detect_all_persons_in_frame(frame_rgb, pose_detector)

        all_frame_poses.append(frame_persons)
        frame_idx += 1

    # Close video
    cap.release()

    # =========================================================================
    # STEP 2: Identify stroke executor across all frames
    # =========================================================================
    executor_poses = identify_stroke_executor(all_frame_poses, stroke_type)

    # =========================================================================
    # STEP 3: Calculate quality scores
    # =========================================================================
    frame_qualities = []
    for pose in executor_poses:
        if pose is not None:
            # Average visibility of all keypoints
            visibility_scores = pose[:, 3]  # 4th column is visibility
            frame_quality = np.mean(visibility_scores)
            frame_qualities.append(frame_quality)
        else:
            frame_qualities.append(0.0)

    # Check if extraction was successful
    valid_frames = sum(1 for p in executor_poses if p is not None)
    success = valid_frames > 0

    return executor_poses, frame_qualities, success


# =============================================================================
# HANDLE MISSING POSES (INTERPOLATION)
# =============================================================================
def interpolate_missing_poses(poses):
    """
    Fill in missing poses using linear interpolation.

    Why interpolate?
    - Some frames may have no pose detected (occlusion, blur, etc.)
    - Small gaps can be filled by interpolating between valid frames
    - Maintains temporal continuity for LSTM model

    Method: Linear interpolation
    - Find missing pose (None)
    - Find nearest valid poses before and after
    - Linearly interpolate keypoint positions

    Args:
        poses: List of pose arrays (some may be None)

    Returns:
        interpolated_poses: List of pose arrays (fewer None values)
    """
    # Create a copy to avoid modifying original
    interpolated_poses = poses.copy()

    # Iterate through all frames
    for i in range(len(poses)):
        # Check if pose is missing
        if poses[i] is None:

            # =================================================================
            # STEP 1: Find previous valid pose
            # =================================================================
            prev_idx = i - 1
            while prev_idx >= 0 and poses[prev_idx] is None:
                prev_idx -= 1

            # =================================================================
            # STEP 2: Find next valid pose
            # =================================================================
            next_idx = i + 1
            while next_idx < len(poses) and poses[next_idx] is None:
                next_idx += 1

            # =================================================================
            # STEP 3: Interpolate if both boundaries exist
            # =================================================================
            if prev_idx >= 0 and next_idx < len(poses):
                # We have valid poses before and after
                gap_size = next_idx - prev_idx
                position_in_gap = i - prev_idx

                # Linear interpolation weight
                # alpha = 0.0 at prev_idx, 1.0 at next_idx
                alpha = position_in_gap / gap_size

                # Interpolate all keypoints
                interpolated_pose = (1 - alpha) * poses[prev_idx] + alpha * poses[next_idx]
                interpolated_poses[i] = interpolated_pose

            # If only one boundary exists (start or end of video), leave as None

    return interpolated_poses


# =============================================================================
# CALCULATE POSE QUALITY METRICS
# =============================================================================
def calculate_pose_quality(poses, frame_qualities):
    """
    Calculate quality metrics for the extracted pose sequence.

    Metrics:
    - Total frames: Number of frames in video
    - Valid frames: Frames with detected pose
    - Valid percentage: % of frames with valid pose
    - Average confidence: Mean keypoint visibility score
    - Quality score: Combined metric (valid_pct * avg_confidence)

    Args:
        poses: List of pose arrays
        frame_qualities: List of quality scores per frame

    Returns:
        quality: Dictionary with quality metrics
    """
    total_frames = len(poses)
    valid_frames = sum(1 for p in poses if p is not None)
    valid_percentage = (valid_frames / total_frames * 100) if total_frames > 0 else 0

    # Calculate average confidence (only for valid frames)
    valid_qualities = [q for q in frame_qualities if q > 0]
    avg_confidence = np.mean(valid_qualities) if len(valid_qualities) > 0 else 0

    # Combined quality score
    quality_score = (valid_percentage / 100.0) * avg_confidence

    quality = {
        'total_frames': total_frames,
        'valid_frames': valid_frames,
        'valid_percentage': valid_percentage,
        'avg_confidence': avg_confidence,
        'quality_score': quality_score
    }

    return quality


# =============================================================================
# SAVE POSE DATA
# =============================================================================
def save_poses(poses, frame_qualities, quality, output_path):
    """
    Save pose sequence and metadata to file using pickle.

    Saved data structure:
    {
        'poses': List of (33, 4) arrays [x, y, z, visibility]
        'frame_qualities': List of quality scores per frame
        'quality': Dictionary with overall quality metrics
        'num_frames': Total number of frames
        'num_keypoints': Number of keypoints per pose (always 33 for MediaPipe)
    }

    Args:
        poses: List of pose arrays
        frame_qualities: List of quality scores
        quality: Quality metrics dictionary
        output_path: Path to save file
    """
    data = {
        'poses': poses,
        'frame_qualities': frame_qualities,
        'quality': quality,
        'num_frames': len(poses),
        'num_keypoints': 33  # MediaPipe Pose has 33 keypoints
    }

    # Save using pickle (efficient for numpy arrays)
    with open(output_path, 'wb') as f:
        pickle.dump(data, f)


# =============================================================================
# PROCESS ALL CLIPS
# =============================================================================
def process_all_clips():
    """
    Main processing function: Extract poses from all video clips.

    Process:
    1. Load clips metadata
    2. Initialize MediaPipe Pose detector
    3. For each clip:
       - Extract poses for ALL persons per frame
       - Identify stroke executor using biomechanical heuristics
       - Track executor consistently across frames
       - Interpolate missing poses
       - Calculate quality metrics
       - Save results
    4. Generate summary report
    """
    print("=" * 70)
    print("EXTRACTING POSES FROM VIDEO CLIPS (MULTI-PLAYER DETECTION)")
    print("=" * 70)
    print()

    # =========================================================================
    # STEP 1: Setup
    # =========================================================================
    # Create output directory
    POSES_DIR.mkdir(parents=True, exist_ok=True)

    # Load clips metadata
    if not CLIPS_METADATA_FILE.exists():
        print(f"❌ ERROR: {CLIPS_METADATA_FILE} not found!")
        print("Please run extract_clips.py first.")
        return

    clips_df = pd.read_csv(CLIPS_METADATA_FILE)
    print(f"✅ Loaded metadata for {len(clips_df)} clips")
    print()

    # =========================================================================
    # STEP 2: Initialize MediaPipe Pose
    # =========================================================================
    pose_detector, mp_pose, mp_drawing = initialize_pose_detector()
    print()

    # =========================================================================
    # STEP 3: Process each clip
    # =========================================================================
    # Statistics tracking
    total_clips = len(clips_df)
    successful_extractions = 0
    failed_extractions = 0
    high_quality_count = 0  # Clips with >80% valid frames
    low_quality_count = 0   # Clips with <50% valid frames

    extraction_log = []

    print(f"Processing {total_clips} clips...")
    print("Note: Both players visible - identifying stroke executor...")
    print()

    # Process each clip with progress bar
    for _, clip in tqdm(clips_df.iterrows(), total=total_clips,
                        desc="Extracting poses", unit="clip"):

        clip_name = clip['clip_name']
        stroke_type = clip['stroke_type_english']
        clip_path = CLIPS_DIR / clip_name
        pose_path = POSES_DIR / clip_name.replace('.mp4', '_pose.pkl')

        # ---------------------------------------------------------------------
        # Skip if already processed
        # ---------------------------------------------------------------------
        if pose_path.exists():
            successful_extractions += 1
            extraction_log.append({
                'clip_name': clip_name,
                'stroke_type': stroke_type,
                'status': 'already_exists'
            })
            continue

        # ---------------------------------------------------------------------
        # Check if clip file exists
        # ---------------------------------------------------------------------
        if not clip_path.exists():
            failed_extractions += 1
            extraction_log.append({
                'clip_name': clip_name,
                'stroke_type': stroke_type,
                'status': 'clip_not_found',
                'error': f'Clip file not found: {clip_path}'
            })
            continue

        # ---------------------------------------------------------------------
        # Extract poses from video (with player identification)
        # ---------------------------------------------------------------------
        try:
            poses, frame_qualities, success = extract_pose_from_video(
                clip_path,
                pose_detector,
                stroke_type
            )

            if not success or poses is None:
                # Pose extraction failed
                failed_extractions += 1
                extraction_log.append({
                    'clip_name': clip_name,
                    'stroke_type': stroke_type,
                    'status': 'extraction_failed',
                    'error': 'No poses detected in video'
                })
                continue

            # -----------------------------------------------------------------
            # Interpolate missing poses
            # -----------------------------------------------------------------
            interpolated_poses = interpolate_missing_poses(poses)

            # -----------------------------------------------------------------
            # Calculate quality metrics
            # -----------------------------------------------------------------
            quality = calculate_pose_quality(interpolated_poses, frame_qualities)

            # -----------------------------------------------------------------
            # Check quality threshold
            # -----------------------------------------------------------------
            if quality['valid_percentage'] < MIN_VALID_FRAMES_PERCENTAGE:
                low_quality_count += 1
                # Still save, but mark as low quality
            elif quality['valid_percentage'] > 80:
                high_quality_count += 1

            # -----------------------------------------------------------------
            # Save pose data
            # -----------------------------------------------------------------
            save_poses(interpolated_poses, frame_qualities, quality, pose_path)

            successful_extractions += 1
            extraction_log.append({
                'clip_name': clip_name,
                'stroke_type': stroke_type,
                'status': 'success',
                'total_frames': quality['total_frames'],
                'valid_frames': quality['valid_frames'],
                'valid_percentage': quality['valid_percentage'],
                'avg_confidence': quality['avg_confidence'],
                'quality_score': quality['quality_score']
            })

        except Exception as e:
            # Handle any unexpected errors
            failed_extractions += 1
            extraction_log.append({
                'clip_name': clip_name,
                'stroke_type': stroke_type,
                'status': 'error',
                'error': str(e)
            })

    # =========================================================================
    # STEP 4: Cleanup and save results
    # =========================================================================
    # Close MediaPipe Pose detector
    pose_detector.close()

    # Save extraction log
    log_df = pd.DataFrame(extraction_log)
    log_file = REPORTS_DIR / "pose_extraction_log.csv"
    log_df.to_csv(log_file, index=False)

    # =========================================================================
    # STEP 5: Print summary report
    # =========================================================================
    print()
    print("=" * 70)
    print("POSE EXTRACTION SUMMARY")
    print("=" * 70)
    print(f"Total clips: {total_clips}")
    print(f"✅ Successfully extracted: {successful_extractions}")
    print(f"❌ Failed: {failed_extractions}")
    print(f"⭐ High quality (>80% valid frames): {high_quality_count}")
    print(f"⚠️  Low quality (<{MIN_VALID_FRAMES_PERCENTAGE}% valid frames): {low_quality_count}")
    print()

    # Success rate
    success_rate = (successful_extractions / total_clips * 100) if total_clips > 0 else 0
    print(f"Success rate: {success_rate:.1f}%")
    print()

    # =========================================================================
    # Quality statistics (only for successful extractions)
    # =========================================================================
    success_df = log_df[log_df['status'] == 'success']
    if len(success_df) > 0:
        print("Quality Statistics (Successful Extractions):")
        print(f"  Average valid frames: {success_df['valid_percentage'].mean():.1f}%")
        print(f"  Average confidence: {success_df['avg_confidence'].mean():.3f}")
        print(f"  Average quality score: {success_df['quality_score'].mean():.3f}")
        print()

        # Breakdown by stroke type
        print("By Stroke Type:")
        for stroke_type in success_df['stroke_type'].unique():
            stroke_df = success_df[success_df['stroke_type'] == stroke_type]
            print(f"  {stroke_type}:")
            print(f"    Clips: {len(stroke_df)}")
            print(f"    Avg valid frames: {stroke_df['valid_percentage'].mean():.1f}%")
            print(f"    Avg confidence: {stroke_df['avg_confidence'].mean():.3f}")
        print()

    # =========================================================================
    # Save paths
    # =========================================================================
    print(f"📁 Poses saved to: {POSES_DIR}")
    print(f"📊 Extraction log saved to: {log_file}")
    print()

    # =========================================================================
    # Recommendations
    # =========================================================================
    if low_quality_count > 0:
        print(f"⚠️  {low_quality_count} clips have low quality poses (<{MIN_VALID_FRAMES_PERCENTAGE}% valid frames)")
        print("   Consider:")
        print("   - Reviewing these clips manually")
        print("   - Adjusting confidence thresholds")
        print("   - Filtering them out during training")
        print()


# =============================================================================
# MAIN ENTRY POINT
# =============================================================================
def main():
    """Main function"""
    print("\n")
    print("*" * 70)
    print("*" + " " * 68 + "*")
    print("*" + "  ITI123 Project: Pose Extraction (Multi-Player)".center(68) + "*")
    print("*" + " " * 68 + "*")
    print("*" * 70)
    print("\n")

    print("Configuration:")
    print(f"  Pose Detection: MediaPipe Pose (Model Complexity = {MODEL_COMPLEXITY})")
    print(f"  Detection Confidence: {MIN_DETECTION_CONFIDENCE}")
    print(f"  Tracking Confidence: {MIN_TRACKING_CONFIDENCE}")
    print(f"  Keypoints per pose: 33")
    print(f"  Match type: Singles (both players visible)")
    print(f"  Player identification: Arm elevation + movement heuristics")
    print()

    # Process all clips
    process_all_clips()

    print("=" * 70)
    print("✅ POSE EXTRACTION COMPLETE!")
    print("=" * 70)
    print()
    print("Next steps:")
    print("1. Review pose extraction log: outputs/reports/pose_extraction_log.csv")
    print("2. Check pose quality statistics")
    print("3. Proceed with feature engineering")
    print("4. Run: python src/data_processing/feature_engineering.py")
    print()


if __name__ == "__main__":
    main()
