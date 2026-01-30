#!/usr/bin/env python3
"""
Phase-Specific Features - FEAT-04

Extracts per-phase statistics and deceleration features.

Features extracted for EACH phase (preparation, backswing, forward_swing, follow_through):
- Mean wrist velocity
- Max wrist velocity
- Arm extension mean
- Elbow angle mean
- Body lean angle

Also includes deceleration features (FEAT-07) for future drop shot classification.

Output: ~20-25 features total (5 phases x 4-5 features each)
"""

import numpy as np
from scipy.ndimage import gaussian_filter1d


# MediaPipe Pose landmark indices
LANDMARKS = {
    'nose': 0,
    'left_shoulder': 11, 'right_shoulder': 12,
    'left_elbow': 13, 'right_elbow': 14,
    'left_wrist': 15, 'right_wrist': 16,
    'left_hip': 23, 'right_hip': 24,
}


def calculate_angle_3d(p1, p2, p3):
    """
    Calculate 3D angle at p2 formed by vectors p1-p2 and p3-p2.

    Args:
        p1, p2, p3: 3D points as numpy arrays (x, y, z)

    Returns:
        Angle in degrees
    """
    v1 = p1 - p2
    v2 = p3 - p2

    cosine = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-8)
    cosine = np.clip(cosine, -1.0, 1.0)

    return np.degrees(np.arccos(cosine))


def extract_phase_specific_features(pose_sequence, phases, fps=30, handedness='right'):
    """
    Extract statistics for each phase separately.

    For EACH phase (preparation, backswing, forward_swing, follow_through):
    - Mean wrist velocity
    - Max wrist velocity
    - Arm extension mean
    - Elbow angle mean
    - Body lean angle

    Args:
        pose_sequence: Array of shape (num_frames, 33, 3) - MediaPipe pose sequence
        phases: dict from phase_segmentation.segment_stroke_phases()
        fps: Frames per second (default 30)
        handedness: 'right' or 'left' (which hand holds racket)

    Returns:
        dict with phase-specific features:
            {phase}_{feature}_{stat} format
            e.g., 'preparation_wrist_vel_mean', 'forward_swing_elbow_angle_mean'
    """
    # Select landmarks based on handedness
    if handedness == 'right':
        shoulder_idx = LANDMARKS['right_shoulder']
        elbow_idx = LANDMARKS['right_elbow']
        wrist_idx = LANDMARKS['right_wrist']
        hip_idx = LANDMARKS['right_hip']
    else:
        shoulder_idx = LANDMARKS['left_shoulder']
        elbow_idx = LANDMARKS['left_elbow']
        wrist_idx = LANDMARKS['left_wrist']
        hip_idx = LANDMARKS['left_hip']

    # Extract positions
    shoulder_positions = pose_sequence[:, shoulder_idx, :]
    elbow_positions = pose_sequence[:, elbow_idx, :]
    wrist_positions = pose_sequence[:, wrist_idx, :]
    hip_positions = pose_sequence[:, hip_idx, :]

    # Smooth positions
    shoulder_smooth = gaussian_filter1d(shoulder_positions, sigma=1.5, axis=0)
    elbow_smooth = gaussian_filter1d(elbow_positions, sigma=1.5, axis=0)
    wrist_smooth = gaussian_filter1d(wrist_positions, sigma=1.5, axis=0)
    hip_smooth = gaussian_filter1d(hip_positions, sigma=1.5, axis=0)

    # Calculate wrist velocity
    wrist_velocities = np.diff(wrist_smooth, axis=0)
    wrist_velocity_mag = np.linalg.norm(wrist_velocities, axis=1) * fps
    wrist_velocity_mag = np.concatenate([[0], wrist_velocity_mag])

    # Calculate arm extension (distance from shoulder to wrist)
    arm_extension = np.linalg.norm(wrist_smooth - shoulder_smooth, axis=1)

    # Calculate elbow angles per frame
    elbow_angles = []
    for i in range(len(pose_sequence)):
        angle = calculate_angle_3d(
            shoulder_smooth[i],
            elbow_smooth[i],
            wrist_smooth[i]
        )
        elbow_angles.append(angle)
    elbow_angles = np.array(elbow_angles)

    # Calculate body lean angle (shoulder-hip vertical angle)
    body_lean_angles = []
    for i in range(len(pose_sequence)):
        # Vector from hip to shoulder
        torso_vector = shoulder_smooth[i] - hip_smooth[i]
        # Vertical reference (0, -1, 0) - negative Y is up
        vertical = np.array([0, -1, 0])
        # Angle from vertical
        cosine = np.dot(torso_vector, vertical) / (np.linalg.norm(torso_vector) + 1e-8)
        cosine = np.clip(cosine, -1.0, 1.0)
        angle = np.degrees(np.arccos(cosine))
        body_lean_angles.append(angle)
    body_lean_angles = np.array(body_lean_angles)

    # === Extract features for each phase ===
    features = {}

    phase_names = ['preparation', 'backswing', 'forward_swing', 'follow_through']

    for phase_name in phase_names:
        # Get phase boundaries
        phase_bounds = phases[phase_name]
        start_frame = phase_bounds[0]
        end_frame = phase_bounds[1]

        # Ensure indices are within bounds
        start_frame = max(0, start_frame)
        end_frame = min(len(wrist_velocity_mag) - 1, end_frame)

        # Skip if phase is too short
        if end_frame <= start_frame:
            features[f'{phase_name}_wrist_vel_mean'] = 0.0
            features[f'{phase_name}_wrist_vel_max'] = 0.0
            features[f'{phase_name}_arm_extension_mean'] = 0.0
            features[f'{phase_name}_elbow_angle_mean'] = 0.0
            features[f'{phase_name}_body_lean_mean'] = 0.0
            continue

        # Extract phase data
        phase_wrist_vel = wrist_velocity_mag[start_frame:end_frame+1]
        phase_arm_extension = arm_extension[start_frame:end_frame+1]
        phase_elbow_angles = elbow_angles[start_frame:end_frame+1]
        phase_body_lean = body_lean_angles[start_frame:end_frame+1]

        # Calculate statistics
        features[f'{phase_name}_wrist_vel_mean'] = float(np.mean(phase_wrist_vel))
        features[f'{phase_name}_wrist_vel_max'] = float(np.max(phase_wrist_vel))
        features[f'{phase_name}_arm_extension_mean'] = float(np.mean(phase_arm_extension))
        features[f'{phase_name}_elbow_angle_mean'] = float(np.mean(phase_elbow_angles))
        features[f'{phase_name}_body_lean_mean'] = float(np.mean(phase_body_lean))

    return features


def aggregate_phase_statistics(frame_features, phases):
    """
    Aggregate per-frame features by phase.

    Takes per-frame features (from v2 extract_frame_features) and aggregates
    by phase (mean, max, std within each phase).

    Args:
        frame_features: dict with per-frame feature arrays (num_frames,)
        phases: dict from phase_segmentation.segment_stroke_phases()

    Returns:
        dict keyed by {phase}_{featurename}_{stat}
        e.g., 'forward_swing_velocity_x_mean', 'backswing_acceleration_max'
    """
    aggregated = {}

    phase_names = ['preparation', 'backswing', 'forward_swing', 'follow_through']
    stats = ['mean', 'max', 'std']

    for phase_name in phase_names:
        # Get phase boundaries
        phase_bounds = phases[phase_name]
        start_frame = phase_bounds[0]
        end_frame = phase_bounds[1]

        for feature_name, feature_values in frame_features.items():
            # Extract phase data
            phase_data = feature_values[start_frame:end_frame+1]

            # Skip if phase is too short
            if len(phase_data) == 0:
                for stat in stats:
                    aggregated[f'{phase_name}_{feature_name}_{stat}'] = 0.0
                continue

            # Calculate statistics
            aggregated[f'{phase_name}_{feature_name}_mean'] = float(np.mean(phase_data))
            aggregated[f'{phase_name}_{feature_name}_max'] = float(np.max(phase_data))
            aggregated[f'{phase_name}_{feature_name}_std'] = float(np.std(phase_data))

    return aggregated


def calculate_deceleration_features(pose_sequence, phases, fps=30, handedness='right'):
    """
    Calculate deceleration control features (FEAT-07).

    For FEAT-07 (deceleration control) - deprioritized but implemented:
    - Calculate velocity reduction rate in follow-through
    - Note: Dataset is Clear+Smash only; drop shot deceleration may not discriminate
    - Include but expect low effect size

    Args:
        pose_sequence: Array of shape (num_frames, 33, 3) - MediaPipe pose sequence
        phases: dict from phase_segmentation.segment_stroke_phases()
        fps: Frames per second (default 30)
        handedness: 'right' or 'left' (which hand holds racket)

    Returns:
        dict with:
            - follow_through_decel_rate: float (rate of velocity decrease)
            - follow_through_decel_smoothness: float (jerk during deceleration)
    """
    # Select wrist landmark based on handedness
    if handedness == 'right':
        wrist_idx = LANDMARKS['right_wrist']
    else:
        wrist_idx = LANDMARKS['left_wrist']

    # Extract wrist positions
    wrist_positions = pose_sequence[:, wrist_idx, :]

    # Smooth positions
    wrist_smooth = gaussian_filter1d(wrist_positions, sigma=1.5, axis=0)

    # Calculate velocity
    wrist_velocities = np.diff(wrist_smooth, axis=0)
    wrist_velocity_mag = np.linalg.norm(wrist_velocities, axis=1) * fps
    wrist_velocity_mag = np.concatenate([[0], wrist_velocity_mag])

    # Calculate acceleration (for jerk calculation)
    wrist_acceleration = np.diff(wrist_velocity_mag)
    wrist_acceleration = np.concatenate([[0], wrist_acceleration])

    # Calculate jerk (rate of change of acceleration)
    wrist_jerk = np.diff(wrist_acceleration)
    wrist_jerk = np.concatenate([[0], wrist_jerk])

    # === Extract follow-through phase ===
    follow_through_start, follow_through_end = phases['follow_through']

    # Ensure indices are within bounds
    follow_through_start = max(0, follow_through_start)
    follow_through_end = min(len(wrist_velocity_mag) - 1, follow_through_end)

    # Extract follow-through data
    follow_velocity = wrist_velocity_mag[follow_through_start:follow_through_end+1]
    follow_jerk = wrist_jerk[follow_through_start:follow_through_end+1]

    # === Calculate deceleration rate ===
    # Rate of velocity decrease (negative slope = deceleration)
    if len(follow_velocity) > 1:
        # Linear fit to velocity over time
        x = np.arange(len(follow_velocity))
        # polyfit returns [slope, intercept]
        slope, _ = np.polyfit(x, follow_velocity, 1)
        decel_rate = -slope  # Negative slope = deceleration
    else:
        decel_rate = 0.0

    # === Calculate deceleration smoothness ===
    # Jerk magnitude (measure of smoothness - lower jerk = smoother)
    if len(follow_jerk) > 0:
        decel_smoothness = float(np.mean(np.abs(follow_jerk)))
    else:
        decel_smoothness = 0.0

    features = {
        'follow_through_decel_rate': float(decel_rate),
        'follow_through_decel_smoothness': decel_smoothness,
    }

    return features
