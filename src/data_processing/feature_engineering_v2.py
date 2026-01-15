#!/usr/bin/env python3
"""
Feature Engineering V2 - Improved Features for Clear vs Smash Classification

Key improvements based on diagnostic analysis:
1. Focus on Z-coordinate (depth) - shows medium effect size
2. Arm extension patterns - shows medium effect size
3. Acceleration and jerk (rate of velocity change)
4. Body posture (lean angle, torso position)
5. Temporal dynamics (peak velocity timing, stroke phase segmentation)
6. Relative metrics (wrist relative to shoulder, not body center)
"""

import os
import pandas as pd
import numpy as np
from pathlib import Path
from tqdm import tqdm
import pickle
from scipy import signal
from scipy.ndimage import gaussian_filter1d

# Paths
BASE_DIR = Path(__file__).resolve().parents[2]
POSES_DIR = BASE_DIR / "data" / "processed" / "poses"  # Original MediaPipe poses
FEATURES_DIR = BASE_DIR / "data" / "processed" / "features"
REPORTS_DIR = BASE_DIR / "outputs" / "reports"
CLIPS_METADATA_FILE = BASE_DIR / "data" / "processed" / "clips" / "clips_metadata.csv"

# MediaPipe Pose landmark indices
LANDMARKS = {
    'nose': 0,
    'left_shoulder': 11, 'right_shoulder': 12,
    'left_elbow': 13, 'right_elbow': 14,
    'left_wrist': 15, 'right_wrist': 16,
    'left_hip': 23, 'right_hip': 24,
    'left_knee': 25, 'right_knee': 26,
    'left_ankle': 27, 'right_ankle': 28,
}


def smooth_trajectory(positions, sigma=1.5):
    """Apply Gaussian smoothing to reduce noise"""
    if len(positions) < 3:
        return positions
    return gaussian_filter1d(positions, sigma=sigma, axis=0)


def calculate_velocity(positions, smooth=True):
    """Calculate velocity with optional smoothing"""
    if len(positions) < 2:
        return np.zeros_like(positions)

    if smooth:
        positions = smooth_trajectory(positions)

    velocities = np.diff(positions, axis=0)
    # Pad with zero for first frame
    velocities = np.vstack([np.zeros((1, positions.shape[1])), velocities])
    return velocities


def calculate_acceleration(velocities):
    """Calculate acceleration from velocities"""
    if len(velocities) < 2:
        return np.zeros_like(velocities)

    accelerations = np.diff(velocities, axis=0)
    accelerations = np.vstack([np.zeros((1, velocities.shape[1])), accelerations])
    return accelerations


def calculate_jerk(accelerations):
    """Calculate jerk (rate of change of acceleration)"""
    if len(accelerations) < 2:
        return np.zeros_like(accelerations)

    jerks = np.diff(accelerations, axis=0)
    jerks = np.vstack([np.zeros((1, accelerations.shape[1])), jerks])
    return jerks


def calculate_angle_3d(p1, p2, p3):
    """Calculate 3D angle at p2 formed by p1-p2-p3"""
    v1 = p1 - p2
    v2 = p3 - p2

    cosine = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-8)
    cosine = np.clip(cosine, -1.0, 1.0)
    return np.degrees(np.arccos(cosine))


def extract_frame_features(pose):
    """Extract features from a single frame"""
    if pose is None:
        return None

    # Ensure we have x, y, z only
    if pose.shape[1] == 4:
        pose = pose[:, :3]

    features = {}

    # Key landmarks
    nose = pose[LANDMARKS['nose']]
    l_shoulder = pose[LANDMARKS['left_shoulder']]
    r_shoulder = pose[LANDMARKS['right_shoulder']]
    l_elbow = pose[LANDMARKS['left_elbow']]
    r_elbow = pose[LANDMARKS['right_elbow']]
    l_wrist = pose[LANDMARKS['left_wrist']]
    r_wrist = pose[LANDMARKS['right_wrist']]
    l_hip = pose[LANDMARKS['left_hip']]
    r_hip = pose[LANDMARKS['right_hip']]

    # Body reference points
    shoulder_center = (l_shoulder + r_shoulder) / 2
    hip_center = (l_hip + r_hip) / 2

    # === ARM EXTENSION (Key discriminative feature) ===
    # Distance from shoulder to wrist (both arms)
    r_arm_extension = np.linalg.norm(r_wrist - r_shoulder)
    l_arm_extension = np.linalg.norm(l_wrist - l_shoulder)
    features['r_arm_extension'] = r_arm_extension
    features['l_arm_extension'] = l_arm_extension
    features['arm_extension_ratio'] = r_arm_extension / (l_arm_extension + 1e-8)

    # === DEPTH FEATURES (Z-coordinate - Key discriminative feature) ===
    # Wrist depth relative to shoulder
    features['r_wrist_depth'] = r_wrist[2] - r_shoulder[2]
    features['l_wrist_depth'] = l_wrist[2] - l_shoulder[2]

    # Wrist depth relative to hip (body depth reference)
    features['r_wrist_depth_from_hip'] = r_wrist[2] - hip_center[2]

    # Elbow depth
    features['r_elbow_depth'] = r_elbow[2] - r_shoulder[2]

    # Forward lean (shoulder center vs hip center in Z)
    features['body_forward_lean'] = shoulder_center[2] - hip_center[2]

    # === HEIGHT FEATURES (Y-coordinate) ===
    # Wrist height relative to shoulder (negative = above shoulder)
    features['r_wrist_height_rel'] = r_wrist[1] - r_shoulder[1]
    features['l_wrist_height_rel'] = l_wrist[1] - l_shoulder[1]

    # Wrist height relative to head
    features['r_wrist_height_from_head'] = r_wrist[1] - nose[1]

    # Elbow height relative to shoulder
    features['r_elbow_height_rel'] = r_elbow[1] - r_shoulder[1]

    # === LATERAL FEATURES (X-coordinate) ===
    # Wrist lateral position relative to shoulder
    features['r_wrist_lateral'] = r_wrist[0] - r_shoulder[0]

    # Body width
    shoulder_width = np.linalg.norm(r_shoulder[:2] - l_shoulder[:2])
    features['shoulder_width'] = shoulder_width

    # === JOINT ANGLES ===
    # Elbow angle (shoulder-elbow-wrist)
    r_elbow_angle = calculate_angle_3d(r_shoulder, r_elbow, r_wrist)
    features['r_elbow_angle'] = r_elbow_angle

    # Shoulder angle (elbow-shoulder-hip)
    r_shoulder_angle = calculate_angle_3d(r_elbow, r_shoulder, r_hip)
    features['r_shoulder_angle'] = r_shoulder_angle

    # === BODY POSTURE ===
    # Torso lean angle (in Y-Z plane)
    torso_vector = shoulder_center - hip_center
    torso_lean = np.arctan2(torso_vector[2], -torso_vector[1])  # Lean forward = positive
    features['torso_lean'] = np.degrees(torso_lean)

    # Shoulder rotation (in X-Z plane)
    shoulder_vector = r_shoulder - l_shoulder
    shoulder_rotation = np.arctan2(shoulder_vector[2], shoulder_vector[0])
    features['shoulder_rotation'] = np.degrees(shoulder_rotation)

    # === ARM CONFIGURATION ===
    # Upper arm length (shoulder to elbow)
    r_upper_arm = np.linalg.norm(r_elbow - r_shoulder)
    features['r_upper_arm_length'] = r_upper_arm

    # Forearm length (elbow to wrist)
    r_forearm = np.linalg.norm(r_wrist - r_elbow)
    features['r_forearm_length'] = r_forearm

    # Arm straightness (how extended is the arm)
    full_arm_length = r_upper_arm + r_forearm
    features['arm_straightness'] = r_arm_extension / (full_arm_length + 1e-8)

    # === WRIST/FOREARM ORIENTATION FEATURES (NEW - KEY FOR CLEAR VS SMASH) ===
    # These features capture wrist angle and pronation, which are critical
    # differences between Clear (upward wrist) and Smash (downward wrist)

    # Forearm vector (from elbow to wrist)
    r_forearm_vector = r_wrist - r_elbow
    r_forearm_length_check = np.linalg.norm(r_forearm_vector)

    if r_forearm_length_check > 1e-6:
        r_forearm_unit = r_forearm_vector / r_forearm_length_check

        # 1. Forearm vertical angle (KEY FEATURE)
        # Angle between forearm and vertical axis (gravity direction)
        # In image coordinates: Y-axis points DOWN (positive Y = lower in image)
        # Small angle = pointing down (Smash), Large angle = pointing up (Clear)
        vertical_down = np.array([0, 1, 0])
        cos_vert = np.dot(r_forearm_unit, vertical_down)
        r_forearm_vertical_angle = np.degrees(np.arccos(np.clip(cos_vert, -1, 1)))
        features['r_forearm_vertical_angle'] = r_forearm_vertical_angle

        # 2. Forearm horizontal angle (swing direction in X-Z plane)
        # Indicates lateral vs forward swing
        r_forearm_horizontal_angle = np.degrees(np.arctan2(r_forearm_unit[0], r_forearm_unit[2]))
        features['r_forearm_horizontal_angle'] = r_forearm_horizontal_angle

        # 3. Wrist-elbow vertical separation (flexion indicator)
        # Positive = wrist below elbow, Negative = wrist above elbow
        r_wrist_elbow_vertical = r_wrist[1] - r_elbow[1]
        features['r_wrist_elbow_vertical'] = r_wrist_elbow_vertical

        # 4. Forearm extension from body (reach distance)
        # Distance of wrist from shoulder in X-Z plane (horizontal plane)
        shoulder_wrist_horiz = np.array([r_wrist[0] - r_shoulder[0], 0, r_wrist[2] - r_shoulder[2]])
        r_wrist_horizontal_reach = np.linalg.norm(shoulder_wrist_horiz)
        features['r_wrist_horizontal_reach'] = r_wrist_horizontal_reach

        # 5. Arm plane pronation indicator
        # The plane formed by shoulder-elbow-wrist indicates pronation/supination
        r_upper_arm_vector = r_elbow - r_shoulder
        plane_normal = np.cross(r_upper_arm_vector, r_forearm_vector)
        plane_normal_length = np.linalg.norm(plane_normal)

        if plane_normal_length > 1e-6:
            plane_normal_unit = plane_normal / plane_normal_length

            # Lateral component (X-direction)
            # Positive = pronated (palm down), Negative = supinated (palm up)
            lateral_axis = np.array([1, 0, 0])
            r_arm_plane_pronation = np.dot(plane_normal_unit, lateral_axis)
            features['r_arm_plane_pronation'] = r_arm_plane_pronation

            # Vertical component (Y-direction)
            vertical_component = np.dot(plane_normal_unit, vertical_down)
            features['r_arm_plane_vertical'] = vertical_component
        else:
            features['r_arm_plane_pronation'] = 0
            features['r_arm_plane_vertical'] = 0

        # 6. Forearm pitch angle (elevation relative to horizontal)
        # Angle in Y-Z plane (sagittal plane)
        # Positive = pointing upward, Negative = pointing downward
        r_forearm_pitch = np.degrees(np.arctan2(-r_forearm_unit[1],
                                                  np.sqrt(r_forearm_unit[0]**2 + r_forearm_unit[2]**2)))
        features['r_forearm_pitch'] = r_forearm_pitch

    else:
        # Invalid forearm (too short or missing)
        features['r_forearm_vertical_angle'] = 0
        features['r_forearm_horizontal_angle'] = 0
        features['r_wrist_elbow_vertical'] = 0
        features['r_wrist_horizontal_reach'] = 0
        features['r_arm_plane_pronation'] = 0
        features['r_arm_plane_vertical'] = 0
        features['r_forearm_pitch'] = 0

    # Repeat for left arm (less important for right-handed players, but include for completeness)
    l_forearm_vector = l_wrist - l_elbow
    l_forearm_length_check = np.linalg.norm(l_forearm_vector)

    if l_forearm_length_check > 1e-6:
        l_forearm_unit = l_forearm_vector / l_forearm_length_check
        vertical_down = np.array([0, 1, 0])
        cos_vert = np.dot(l_forearm_unit, vertical_down)
        features['l_forearm_vertical_angle'] = np.degrees(np.arccos(np.clip(cos_vert, -1, 1)))
        features['l_wrist_elbow_vertical'] = l_wrist[1] - l_elbow[1]
    else:
        features['l_forearm_vertical_angle'] = 0
        features['l_wrist_elbow_vertical'] = 0

    return features


def extract_temporal_features(pose_sequence):
    """
    Extract temporal features from pose sequence
    Returns per-frame features and sequence-level statistics
    """
    # First pass: extract per-frame features
    frame_features_list = []
    valid_indices = []

    for i, pose in enumerate(pose_sequence):
        features = extract_frame_features(pose)
        if features is not None:
            frame_features_list.append(features)
            valid_indices.append(i)
        else:
            frame_features_list.append(None)

    if len(valid_indices) < 5:  # Need minimum frames
        return None, None

    # Get feature names from first valid frame
    feature_names = list(frame_features_list[valid_indices[0]].keys())
    num_spatial_features = len(feature_names)

    # Convert to array for valid frames
    valid_features = np.array([
        list(frame_features_list[i].values())
        for i in valid_indices
    ])

    # Interpolate missing frames
    full_features = np.zeros((len(pose_sequence), num_spatial_features))
    for i in valid_indices:
        full_features[i] = list(frame_features_list[i].values())

    # Linear interpolation for missing frames
    for col in range(num_spatial_features):
        valid_mask = np.zeros(len(pose_sequence), dtype=bool)
        valid_mask[valid_indices] = True

        if valid_mask.sum() >= 2:
            xp = np.where(valid_mask)[0]
            fp = full_features[valid_mask, col]
            full_features[:, col] = np.interp(np.arange(len(pose_sequence)), xp, fp)

    # === VELOCITY FEATURES ===
    velocities = calculate_velocity(full_features, smooth=True)
    velocity_magnitudes = np.linalg.norm(velocities, axis=1)

    # === ACCELERATION FEATURES ===
    accelerations = calculate_acceleration(velocities)
    acceleration_magnitudes = np.linalg.norm(accelerations, axis=1)

    # === KEY TEMPORAL METRICS ===
    temporal_features = {}

    # Peak velocity timing (normalized 0-1)
    peak_vel_frame = np.argmax(velocity_magnitudes)
    temporal_features['peak_velocity_timing'] = peak_vel_frame / len(pose_sequence)

    # Peak acceleration timing
    peak_acc_frame = np.argmax(acceleration_magnitudes)
    temporal_features['peak_acceleration_timing'] = peak_acc_frame / len(pose_sequence)

    # Max velocity and acceleration
    temporal_features['max_velocity'] = np.max(velocity_magnitudes)
    temporal_features['max_acceleration'] = np.max(acceleration_magnitudes)

    # Mean velocity in different phases
    mid_point = len(pose_sequence) // 2
    temporal_features['early_phase_velocity'] = np.mean(velocity_magnitudes[:mid_point])
    temporal_features['late_phase_velocity'] = np.mean(velocity_magnitudes[mid_point:])
    temporal_features['velocity_ratio'] = (
            temporal_features['late_phase_velocity'] /
            (temporal_features['early_phase_velocity'] + 1e-8)
    )

    # === COMBINE SPATIAL AND TEMPORAL ===
    # Per-frame features: spatial features + velocities
    per_frame_features = np.hstack([full_features, velocities])

    # Feature names for per-frame
    per_frame_names = feature_names + [f'{n}_vel' for n in feature_names]

    return per_frame_features, per_frame_names, temporal_features


def extract_statistical_summary(per_frame_features, feature_names):
    """
    Extract statistical summary of per-frame features.
    Returns fixed-length feature vector for traditional ML.
    """
    summary = {}

    for i, name in enumerate(feature_names):
        col = per_frame_features[:, i]

        # Basic statistics
        summary[f'{name}_mean'] = np.mean(col)
        summary[f'{name}_std'] = np.std(col)
        summary[f'{name}_min'] = np.min(col)
        summary[f'{name}_max'] = np.max(col)

        # Range
        summary[f'{name}_range'] = np.max(col) - np.min(col)

        # Percentiles
        summary[f'{name}_p25'] = np.percentile(col, 25)
        summary[f'{name}_p75'] = np.percentile(col, 75)

    return summary


def process_single_clip(pose_path, target_length=50):
    """
    Process a single clip and extract features

    Args:
        pose_path: Path to pose pickle file
        target_length: Target sequence length for LSTM

    Returns:
        dict with sequence features and statistical summary
    """
    with open(pose_path, 'rb') as f:
        pose_data = pickle.load(f)

    poses = pose_data['poses']

    # Extract temporal features
    result = extract_temporal_features(poses)
    if result[0] is None:
        return None

    per_frame_features, feature_names, temporal_features = result

    # Pad or truncate to target length
    num_frames, num_features = per_frame_features.shape

    if num_frames < target_length:
        # Pad with zeros at the end
        padding = np.zeros((target_length - num_frames, num_features))
        sequence_features = np.vstack([per_frame_features, padding])
    elif num_frames > target_length:
        # Take center portion (stroke impact is usually in middle)
        start = (num_frames - target_length) // 2
        sequence_features = per_frame_features[start:start + target_length]
    else:
        sequence_features = per_frame_features

    # Extract statistical summary for baseline models
    statistical_summary = extract_statistical_summary(per_frame_features, feature_names)

    # Add temporal features to summary
    statistical_summary.update(temporal_features)

    return {
        'sequence_features': sequence_features,  # For LSTM: (target_length, num_features)
        'feature_names': feature_names,
        'statistical_summary': statistical_summary,  # For RF/SVM
        'temporal_features': temporal_features,
        'original_length': num_frames,
    }


def process_all_clips():
    """Process all clips and save features"""
    print("=" * 70)
    print("FEATURE ENGINEERING V2 - IMPROVED DISCRIMINATIVE FEATURES")
    print("=" * 70)
    print()

    # Create output directory
    FEATURES_DIR.mkdir(parents=True, exist_ok=True)

    # Load clips metadata
    clips_df = pd.read_csv(CLIPS_METADATA_FILE)
    print(f"Loaded metadata for {len(clips_df)} clips")
    print()

    # Statistics
    successful = 0
    failed = 0
    feature_log = []

    print("Processing clips...")
    for idx, clip in tqdm(clips_df.iterrows(), total=len(clips_df), desc="Extracting features"):
        clip_name = clip['clip_name']
        pose_path = POSES_DIR / clip_name.replace('.mp4', '_pose.pkl')
        feature_path = FEATURES_DIR / clip_name.replace('.mp4', '_features.pkl')

        # Check pose file exists
        if not pose_path.exists():
            failed += 1
            feature_log.append({
                'clip_name': clip_name,
                'stroke_type': clip['stroke_type_english'],
                'status': 'pose_file_not_found'
            })
            continue

        try:
            result = process_single_clip(pose_path)

            if result is None:
                failed += 1
                feature_log.append({
                    'clip_name': clip_name,
                    'stroke_type': clip['stroke_type_english'],
                    'status': 'insufficient_valid_frames'
                })
                continue

            # Save features
            with open(feature_path, 'wb') as f:
                pickle.dump(result, f)

            successful += 1
            feature_log.append({
                'clip_name': clip_name,
                'stroke_type': clip['stroke_type_english'],
                'status': 'success',
                'num_features': result['sequence_features'].shape[1],
                'original_length': result['original_length']
            })

        except Exception as e:
            failed += 1
            feature_log.append({
                'clip_name': clip_name,
                'stroke_type': clip['stroke_type_english'],
                'status': 'error',
                'error': str(e)
            })

    # Save log
    log_df = pd.DataFrame(feature_log)
    log_file = REPORTS_DIR / "feature_extraction_v2_log.csv"
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    log_df.to_csv(log_file, index=False)

    # Print summary
    print()
    print("=" * 70)
    print("FEATURE EXTRACTION SUMMARY")
    print("=" * 70)
    print(f"Total clips: {len(clips_df)}")
    print(f"Successful: {successful} ({successful/len(clips_df)*100:.1f}%)")
    print(f"Failed: {failed}")
    print()

    # Feature info
    if successful > 0:
        sample_path = FEATURES_DIR / (clips_df.iloc[0]['clip_name'].replace('.mp4', '_features.pkl'))
        if sample_path.exists():
            with open(sample_path, 'rb') as f:
                sample = pickle.load(f)
            print("Feature dimensions:")
            print(f"  Sequence features: {sample['sequence_features'].shape}")
            print(f"  Statistical summary: {len(sample['statistical_summary'])} features")
            print()
            print("Spatial feature names:")
            for i, name in enumerate(sample['feature_names'][:22]):  # First 22 are spatial
                print(f"  {i+1}. {name}")

    print()
    print(f"Features saved to: {FEATURES_DIR}")
    print(f"Log saved to: {log_file}")


def main():
    print("\n")
    print("*" * 70)
    print("*" + " " * 68 + "*")
    print("*" + "  ITI123 Project: Feature Engineering V2".center(68) + "*")
    print("*" + " " * 68 + "*")
    print("*" * 70)
    print("\n")

    print("Key improvements in V2:")
    print("  1. Z-coordinate (depth) features - key discriminator")
    print("  2. Arm extension patterns")
    print("  3. Velocity, acceleration, and jerk")
    print("  4. Peak timing features")
    print("  5. Phase-based analysis")
    print()

    process_all_clips()

    print("=" * 70)
    print("FEATURE ENGINEERING V2 COMPLETE!")
    print("=" * 70)
    print()
    print("Next steps:")
    print("1. Run: python src/data_processing/data_split.py")
    print("2. Train models in Colab")
    print()


if __name__ == "__main__":
    main()
