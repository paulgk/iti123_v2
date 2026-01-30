#!/usr/bin/env python3
"""
Angular Velocity Features - FEAT-05 and FEAT-06

Extracts angular velocity and racket head speed estimation features with proper noise handling.

Critical considerations:
- Angular velocity calculation is noise-sensitive
- MediaPipe pose jitter amplifies through differentiation
- MUST smooth at multiple stages: smooth-differentiate-smooth pipeline

Features extracted:
1. Elbow extension velocity (angular velocity of elbow joint)
2. Forearm rotation velocity (rotation in sagittal plane)
3. Racket head speed estimation (wrist velocity proxy)

Research reference values (elite badminton):
- Forearm rotation: 800-1200 deg/s during forward swing
- Elbow extension: 400-800 deg/s
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

# Research reference values for validation
MAX_REALISTIC_ANGULAR_VELOCITY = 5000  # deg/s (human joint maximum)


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


def validate_angular_velocity(velocities):
    """
    Check angular velocities for noise domination and unrealistic values.

    Validation criteria:
    1. Unrealistic values: |velocity| > 5000 deg/s
    2. Noise domination: std > 3 * abs(median)

    Args:
        velocities: Array of angular velocities (deg/s)

    Returns:
        dict with:
            - quality: 'ok' or 'suspect_noise'
            - max_velocity: float
            - median_velocity: float
            - std_velocity: float
    """
    # Calculate statistics
    max_vel = np.max(np.abs(velocities))
    median_vel = np.median(velocities)
    std_vel = np.std(velocities)

    # Check for noise domination
    if std_vel > abs(median_vel) * 3:
        quality = 'suspect_noise'
    elif max_vel > MAX_REALISTIC_ANGULAR_VELOCITY:
        quality = 'suspect_noise'
    else:
        quality = 'ok'

    return {
        'quality': quality,
        'max_velocity': max_vel,
        'median_velocity': median_vel,
        'std_velocity': std_vel
    }


def extract_angular_velocity_features(pose_sequence, fps=30, handedness='right'):
    """
    Extract angular velocity features with proper noise handling.

    Pipeline (smooth-differentiate-smooth):
    1. Smooth positions with gaussian_filter1d (sigma=1.5)
    2. Calculate joint angles from smoothed positions
    3. Smooth angles with gaussian_filter1d (sigma=1.0)
    4. Differentiate smoothed angles to get angular velocity
    5. Clip to realistic bounds (< 5000 deg/s)

    Args:
        pose_sequence: Array of shape (num_frames, 33, 3) - MediaPipe pose sequence
        fps: Frames per second (default 30)
        handedness: 'right' or 'left' (which hand holds racket)

    Returns:
        dict with:
            - elbow_extension_vel_max: float (max extension speed, deg/s)
            - elbow_extension_vel_median: float (median, robust to noise)
            - elbow_extension_peak_timing: float (normalized 0-1)
            - forearm_rotation_vel_max: float (peak rotation speed)
            - forearm_rotation_vel_range: float (total rotation range)
            - angular_velocity_quality: str ('ok' or 'suspect_noise')
    """
    # Select landmarks based on handedness
    if handedness == 'right':
        shoulder_idx = LANDMARKS['right_shoulder']
        elbow_idx = LANDMARKS['right_elbow']
        wrist_idx = LANDMARKS['right_wrist']
    else:
        shoulder_idx = LANDMARKS['left_shoulder']
        elbow_idx = LANDMARKS['left_elbow']
        wrist_idx = LANDMARKS['left_wrist']

    # Extract positions
    shoulder_positions = pose_sequence[:, shoulder_idx, :]
    elbow_positions = pose_sequence[:, elbow_idx, :]
    wrist_positions = pose_sequence[:, wrist_idx, :]

    # === STEP 1: Smooth positions ===
    shoulder_smooth = gaussian_filter1d(shoulder_positions, sigma=1.5, axis=0)
    elbow_smooth = gaussian_filter1d(elbow_positions, sigma=1.5, axis=0)
    wrist_smooth = gaussian_filter1d(wrist_positions, sigma=1.5, axis=0)

    # === STEP 2: Calculate elbow angles per frame ===
    elbow_angles = []
    for i in range(len(pose_sequence)):
        angle = calculate_angle_3d(
            shoulder_smooth[i],
            elbow_smooth[i],
            wrist_smooth[i]
        )
        elbow_angles.append(angle)

    elbow_angles = np.array(elbow_angles)

    # === STEP 3: Smooth angles before differentiation ===
    elbow_angles_smooth = gaussian_filter1d(elbow_angles, sigma=1.0)

    # === STEP 4: Calculate angular velocity ===
    # Elbow extension velocity (negative = extending, positive = flexing)
    elbow_angular_vel = np.diff(elbow_angles_smooth) * fps  # degrees/second

    # Pad first frame
    elbow_angular_vel = np.concatenate([[0], elbow_angular_vel])

    # === STEP 5: Clip to realistic bounds ===
    elbow_angular_vel = np.clip(
        elbow_angular_vel,
        -MAX_REALISTIC_ANGULAR_VELOCITY,
        MAX_REALISTIC_ANGULAR_VELOCITY
    )

    # === FOREARM ROTATION VELOCITY ===
    # Calculate forearm vector angle in sagittal plane (rotation around elbow)
    forearm_vectors = wrist_smooth - elbow_smooth
    forearm_angles = []

    for i in range(len(forearm_vectors)):
        # Angle of forearm in sagittal plane (Y-Z plane)
        # atan2(y, z) gives rotation angle
        angle = np.degrees(np.arctan2(forearm_vectors[i, 1], forearm_vectors[i, 2]))
        forearm_angles.append(angle)

    forearm_angles = np.array(forearm_angles)

    # Smooth forearm angles
    forearm_angles_smooth = gaussian_filter1d(forearm_angles, sigma=1.0)

    # Calculate forearm rotation velocity
    forearm_rotation_vel = np.diff(forearm_angles_smooth) * fps  # degrees/second
    forearm_rotation_vel = np.concatenate([[0], forearm_rotation_vel])

    # Clip to realistic bounds
    forearm_rotation_vel = np.clip(
        forearm_rotation_vel,
        -MAX_REALISTIC_ANGULAR_VELOCITY,
        MAX_REALISTIC_ANGULAR_VELOCITY
    )

    # === QUALITY VALIDATION ===
    elbow_quality = validate_angular_velocity(elbow_angular_vel)
    forearm_quality = validate_angular_velocity(forearm_rotation_vel)

    # Overall quality (worst of both)
    if elbow_quality['quality'] == 'suspect_noise' or forearm_quality['quality'] == 'suspect_noise':
        overall_quality = 'suspect_noise'
    else:
        overall_quality = 'ok'

    # === EXTRACT FEATURES ===
    # Use robust statistics (median, percentiles) instead of mean
    features = {
        # Elbow extension velocity
        'elbow_extension_vel_max': float(np.max(np.abs(elbow_angular_vel))),
        'elbow_extension_vel_median': float(np.median(elbow_angular_vel)),
        'elbow_extension_peak_timing': float(np.argmax(np.abs(elbow_angular_vel)) / len(elbow_angular_vel)),

        # Forearm rotation velocity
        'forearm_rotation_vel_max': float(np.max(np.abs(forearm_rotation_vel))),
        'forearm_rotation_vel_range': float(np.max(forearm_rotation_vel) - np.min(forearm_rotation_vel)),

        # Quality indicator
        'angular_velocity_quality': overall_quality,
    }

    return features


def estimate_racket_head_speed(pose_sequence, phases, fps=30, handedness='right'):
    """
    Estimate racket head speed from wrist velocity proxy.

    MediaPipe tracks body only, not racket. Research shows wrist velocity
    correlates r=0.72-0.74 with racket head speed.

    Args:
        pose_sequence: Array of shape (num_frames, 33, 3) - MediaPipe pose sequence
        phases: dict from phase_segmentation.segment_stroke_phases()
        fps: Frames per second (default 30)
        handedness: 'right' or 'left' (which hand holds racket)

    Returns:
        dict with:
            - estimated_racket_speed_max: float (peak wrist speed in forward swing)
            - estimated_racket_speed_mean: float (mean wrist speed in forward swing)
            - racket_speed_at_contact: float (wrist speed at contact frame)
            - speed_estimation_method: str ('wrist_proxy')
    """
    # Select wrist landmark based on handedness
    if handedness == 'right':
        wrist_idx = LANDMARKS['right_wrist']
    else:
        wrist_idx = LANDMARKS['left_wrist']

    # Extract wrist positions
    wrist_positions = pose_sequence[:, wrist_idx, :]

    # Smooth positions to reduce noise
    wrist_smooth = gaussian_filter1d(wrist_positions, sigma=1.5, axis=0)

    # Calculate velocity
    wrist_velocities = np.diff(wrist_smooth, axis=0)
    wrist_velocity_mag = np.linalg.norm(wrist_velocities, axis=1)

    # Pad first frame
    wrist_velocity_mag = np.concatenate([[0], wrist_velocity_mag])

    # Convert to speed (multiply by fps to get units/second)
    # Note: MediaPipe coordinates are normalized [0, 1], so this gives normalized speed
    wrist_speed = wrist_velocity_mag * fps

    # === Extract speeds within forward swing phase ===
    forward_swing_start, forward_swing_end = phases['forward_swing']
    contact_frame = phases['contact']

    # Ensure indices are within bounds
    forward_swing_start = max(0, forward_swing_start)
    forward_swing_end = min(len(wrist_speed) - 1, forward_swing_end)
    contact_frame = min(len(wrist_speed) - 1, contact_frame)

    # Extract forward swing speeds
    forward_swing_speeds = wrist_speed[forward_swing_start:forward_swing_end+1]

    features = {
        'estimated_racket_speed_max': float(np.max(forward_swing_speeds)),
        'estimated_racket_speed_mean': float(np.mean(forward_swing_speeds)),
        'racket_speed_at_contact': float(wrist_speed[contact_frame]),
        'speed_estimation_method': 'wrist_proxy',
    }

    return features
