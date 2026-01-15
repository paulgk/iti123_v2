#!/usr/bin/env python3
"""
Feature Engineering Script
Engineer biomechanical features from pose sequences
- Joint angles (shoulder, elbow, wrist)
- Angular velocities
- Spatial features (arm extension, body alignment)
- Temporal features
"""

import os
import pandas as pd
import numpy as np
from pathlib import Path
from tqdm import tqdm
import pickle

# Paths
BASE_DIR = Path(__file__).resolve().parents[2]
POSES_DIR = BASE_DIR / "data" / "processed" / "poses"
FEATURES_DIR = BASE_DIR / "data" / "processed" / "features"
REPORTS_DIR = BASE_DIR / "outputs" / "reports"
CLIPS_METADATA_FILE = BASE_DIR / "data" / "processed" / "clips" / "clips_metadata.csv"

# MediaPipe Pose landmark indices
# Reference: https://google.github.io/mediapipe/solutions/pose.html
LANDMARKS = {
    'nose': 0,
    'left_eye_inner': 1,
    'left_eye': 2,
    'left_eye_outer': 3,
    'right_eye_inner': 4,
    'right_eye': 5,
    'right_eye_outer': 6,
    'left_ear': 7,
    'right_ear': 8,
    'mouth_left': 9,
    'mouth_right': 10,
    'left_shoulder': 11,
    'right_shoulder': 12,
    'left_elbow': 13,
    'right_elbow': 14,
    'left_wrist': 15,
    'right_wrist': 16,
    'left_pinky': 17,
    'right_pinky': 18,
    'left_index': 19,
    'right_index': 20,
    'left_thumb': 21,
    'right_thumb': 22,
    'left_hip': 23,
    'right_hip': 24,
    'left_knee': 25,
    'right_knee': 26,
    'left_ankle': 27,
    'right_ankle': 28,
    'left_heel': 29,
    'right_heel': 30,
    'left_foot_index': 31,
    'right_foot_index': 32,
}


def calculate_angle(p1, p2, p3):
    """
    Calculate angle formed by three points

    Args:
        p1, p2, p3: Points as (x, y, z) arrays
        p2 is the vertex of the angle

    Returns:
        angle in degrees
    """
    # Vectors
    v1 = p1 - p2
    v2 = p3 - p2

    # Calculate angle
    cosine = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-6)
    cosine = np.clip(cosine, -1.0, 1.0)
    angle = np.arccos(cosine)

    return np.degrees(angle)


def calculate_distance(p1, p2):
    """Calculate Euclidean distance between two points"""
    return np.linalg.norm(p1 - p2)


def calculate_velocity(positions, dt=1.0):
    """Calculate velocity from position sequence"""
    if len(positions) < 2:
        return np.zeros_like(positions)

    velocities = np.diff(positions, axis=0) / dt
    # Pad with zero velocity for first frame
    velocities = np.vstack([np.zeros((1, positions.shape[1])), velocities])

    return velocities


def extract_biomechanical_features(pose_sequence):
    """
    Extract biomechanical features from pose sequence

    Args:
        pose_sequence: List of pose arrays, each (33, 4) with [x, y, z, visibility]
                       Some frames may be None (no detection)

    Returns:
        features: Array of shape (num_frames, num_features)
        feature_names: List of feature names
    """
    features_list = []
    num_features = 14  # Fixed number of features per frame

    for frame_idx, pose in enumerate(pose_sequence):
        if pose is None:
            # If pose is missing, use zeros
            features_list.append(np.zeros(num_features))
            continue

        # Handle both (33, 4) and (33, 3) formats
        # Extract only x, y, z coordinates (first 3 columns)
        if pose.shape[1] == 4:
            pose = pose[:, :3]  # Extract x, y, z only

        frame_features = {}

        # === JOINT ANGLES ===

        # Right arm angles (assuming right-handed player - can detect dominant hand later)
        right_shoulder = pose[LANDMARKS['right_shoulder']]
        right_elbow = pose[LANDMARKS['right_elbow']]
        right_wrist = pose[LANDMARKS['right_wrist']]
        right_hip = pose[LANDMARKS['right_hip']]

        # Shoulder angle (shoulder-elbow-wrist)
        shoulder_angle = calculate_angle(right_shoulder, right_elbow, right_wrist)
        frame_features['shoulder_angle'] = shoulder_angle

        # Elbow angle (shoulder-elbow-wrist)
        elbow_angle = calculate_angle(right_shoulder, right_elbow, right_wrist)
        frame_features['elbow_angle'] = elbow_angle

        # Torso angle (hip-shoulder-elbow)
        torso_angle = calculate_angle(right_hip, right_shoulder, right_elbow)
        frame_features['torso_angle'] = torso_angle

        # Left arm for comparison
        left_shoulder = pose[LANDMARKS['left_shoulder']]
        left_elbow = pose[LANDMARKS['left_elbow']]
        left_wrist = pose[LANDMARKS['left_wrist']]

        left_shoulder_angle = calculate_angle(left_shoulder, left_elbow, left_wrist)
        frame_features['left_shoulder_angle'] = left_shoulder_angle

        # === SPATIAL FEATURES ===

        # Arm extension (shoulder to wrist distance)
        arm_extension = calculate_distance(right_shoulder, right_wrist)
        frame_features['arm_extension'] = arm_extension

        # Body height (average shoulder height)
        body_height = (right_shoulder[1] + left_shoulder[1]) / 2
        frame_features['body_height'] = body_height

        # Shoulder width
        shoulder_width = calculate_distance(right_shoulder, left_shoulder)
        frame_features['shoulder_width'] = shoulder_width

        # Torso rotation (angle between shoulders and hips)
        left_hip = pose[LANDMARKS['left_hip']]
        shoulder_vector = right_shoulder[:2] - left_shoulder[:2]
        hip_vector = right_hip[:2] - left_hip[:2]

        shoulder_angle_2d = np.arctan2(shoulder_vector[1], shoulder_vector[0])
        hip_angle_2d = np.arctan2(hip_vector[1], hip_vector[0])
        torso_rotation = np.degrees(shoulder_angle_2d - hip_angle_2d)
        frame_features['torso_rotation'] = torso_rotation

        # === POSITION FEATURES ===

        # Wrist position relative to body center
        body_center = (right_shoulder + left_shoulder + right_hip + left_hip) / 4
        wrist_relative_x = right_wrist[0] - body_center[0]
        wrist_relative_y = right_wrist[1] - body_center[1]
        wrist_relative_z = right_wrist[2] - body_center[2]

        frame_features['wrist_relative_x'] = wrist_relative_x
        frame_features['wrist_relative_y'] = wrist_relative_y
        frame_features['wrist_relative_z'] = wrist_relative_z

        # Elbow position
        elbow_relative_x = right_elbow[0] - body_center[0]
        elbow_relative_y = right_elbow[1] - body_center[1]

        frame_features['elbow_relative_x'] = elbow_relative_x
        frame_features['elbow_relative_y'] = elbow_relative_y

        # === SYMMETRY FEATURES ===

        # Arm length ratio (left vs right)
        left_arm_length = calculate_distance(left_shoulder, left_wrist)
        right_arm_length = calculate_distance(right_shoulder, right_wrist)
        arm_length_ratio = left_arm_length / (right_arm_length + 1e-6)
        frame_features['arm_length_ratio'] = arm_length_ratio

        # Convert to array
        feature_vector = np.array(list(frame_features.values()))
        features_list.append(feature_vector)

    # Convert list to array - all elements should be same shape now
    features_array = np.vstack(features_list)

    # === TEMPORAL FEATURES (velocities) ===

    # Calculate velocities for key features
    velocities = calculate_velocity(features_array)

    # Combine spatial and temporal features
    combined_features = np.hstack([features_array, velocities])

    return combined_features, list(frame_features.keys())


def normalize_features(features, method='standardize'):
    """
    Normalize features

    Args:
        features: (num_frames, num_features)
        method: 'standardize' or 'minmax'

    Returns:
        normalized_features, mean, std (or min, max)
    """
    if method == 'standardize':
        mean = np.mean(features, axis=0)
        std = np.std(features, axis=0) + 1e-6
        normalized = (features - mean) / std
        return normalized, mean, std

    elif method == 'minmax':
        min_val = np.min(features, axis=0)
        max_val = np.max(features, axis=0)
        range_val = max_val - min_val + 1e-6
        normalized = (features - min_val) / range_val
        return normalized, min_val, max_val


def pad_or_truncate_sequence(features, target_length=75):
    """
    Pad or truncate feature sequence to fixed length

    Args:
        features: (num_frames, num_features)
        target_length: desired sequence length

    Returns:
        padded_features: (target_length, num_features)
    """
    num_frames, num_features = features.shape

    if num_frames == target_length:
        return features
    elif num_frames < target_length:
        # Pad with zeros
        padding = np.zeros((target_length - num_frames, num_features))
        return np.vstack([features, padding])
    else:
        # Truncate (take middle portion to keep stroke impact)
        start_idx = (num_frames - target_length) // 2
        return features[start_idx:start_idx + target_length]


def process_all_poses():
    """Process all poses to extract features"""
    print("=" * 70)
    print("ENGINEERING FEATURES FROM POSES")
    print("=" * 70)
    print()

    # Create output directory
    FEATURES_DIR.mkdir(parents=True, exist_ok=True)

    # Load clips metadata
    if not CLIPS_METADATA_FILE.exists():
        print(f"❌ ERROR: {CLIPS_METADATA_FILE} not found!")
        return

    clips_df = pd.read_csv(CLIPS_METADATA_FILE)
    print(f"✅ Loaded metadata for {len(clips_df)} clips")
    print()

    # Statistics
    total_clips = len(clips_df)
    successful_extractions = 0
    failed_extractions = 0

    feature_log = []

    print(f"Processing {total_clips} clips...")
    print()

    # Process each clip
    for idx, clip in tqdm(clips_df.iterrows(), total=total_clips, desc="Extracting features"):
        clip_name = clip['clip_name']
        pose_path = POSES_DIR / clip_name.replace('.mp4', '_pose.pkl')
        feature_path = FEATURES_DIR / clip_name.replace('.mp4', '_features.pkl')

        # Skip if already processed
        if feature_path.exists():
            successful_extractions += 1
            feature_log.append({
                'clip_name': clip_name,
                'stroke_type': clip['stroke_type_english'],
                'status': 'already_exists'
            })
            continue

        # Check if pose file exists
        if not pose_path.exists():
            failed_extractions += 1
            feature_log.append({
                'clip_name': clip_name,
                'stroke_type': clip['stroke_type_english'],
                'status': 'pose_file_not_found'
            })
            continue

        # Load poses
        try:
            with open(pose_path, 'rb') as f:
                pose_data = pickle.load(f)

            poses = pose_data['poses']

            # Extract features
            features, feature_names = extract_biomechanical_features(poses)

            # Normalize features
            normalized_features, mean, std = normalize_features(features)

            # Pad/truncate to fixed length
            fixed_length_features = pad_or_truncate_sequence(normalized_features, target_length=75)

            # Save features
            feature_data = {
                'features': fixed_length_features,
                'feature_names': feature_names,
                'raw_features': features,
                'normalization_mean': mean,
                'normalization_std': std,
                'original_length': len(poses),
                'num_features': features.shape[1]
            }

            with open(feature_path, 'wb') as f:
                pickle.dump(feature_data, f)

            successful_extractions += 1
            feature_log.append({
                'clip_name': clip_name,
                'stroke_type': clip['stroke_type_english'],
                'status': 'success',
                'num_frames': len(poses),
                'num_features': features.shape[1]
            })

        except Exception as e:
            failed_extractions += 1
            feature_log.append({
                'clip_name': clip_name,
                'stroke_type': clip['stroke_type_english'],
                'status': 'failed',
                'error': str(e)
            })

    # Save feature extraction log
    log_df = pd.DataFrame(feature_log)
    log_file = REPORTS_DIR / "feature_extraction_log.csv"
    log_df.to_csv(log_file, index=False)

    # Print summary
    print()
    print("=" * 70)
    print("FEATURE EXTRACTION SUMMARY")
    print("=" * 70)
    print(f"Total clips: {total_clips}")
    print(f"✅ Successfully extracted: {successful_extractions}")
    print(f"❌ Failed: {failed_extractions}")
    print()

    success_rate = (successful_extractions / total_clips * 100) if total_clips > 0 else 0
    print(f"Success rate: {success_rate:.1f}%")
    print()

    # Feature statistics
    success_df = log_df[log_df['status'] == 'success']
    if len(success_df) > 0:
        print("Feature Statistics:")
        print(f"  Average number of features per frame: {success_df['num_features'].mean():.0f}")
        print()

        print("By Stroke Type:")
        for stroke_type in success_df['stroke_type'].unique():
            stroke_df = success_df[success_df['stroke_type'] == stroke_type]
            print(f"  {stroke_type}: {len(stroke_df)} clips")

    print()
    print(f"📁 Features saved to: {FEATURES_DIR}")
    print(f"📊 Feature log saved to: {log_file}")
    print()


def main():
    """Main function"""
    print("\n")
    print("*" * 70)
    print("*" + " " * 68 + "*")
    print("*" + "  ITI123 Project: Feature Engineering".center(68) + "*")
    print("*" + " " * 68 + "*")
    print("*" * 70)
    print("\n")

    print("Configuration:")
    print("  Features extracted:")
    print("    - Joint angles (shoulder, elbow, torso)")
    print("    - Spatial measurements (arm extension, torso rotation)")
    print("    - Position features (wrist, elbow relative to body)")
    print("    - Temporal features (velocities)")
    print("  Sequence length: 75 frames (padded/truncated)")
    print("  Normalization: Standardization (zero mean, unit variance)")
    print()

    # Process all poses
    process_all_poses()

    print("=" * 70)
    print("✅ FEATURE ENGINEERING COMPLETE!")
    print("=" * 70)
    print()
    print("Next steps:")
    print("1. Review feature extraction log: outputs/reports/feature_extraction_log.csv")
    print("2. Proceed with data splitting and model training")
    print("3. Run: python src/data_processing/data_split.py")
    print()


if __name__ == "__main__":
    main()
