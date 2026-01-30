#!/usr/bin/env python3
"""
Contact Frame and Intent Window Features - Priority 0 (P0)

Extracts biomechanical features at contact frame and from intent window.

Critical coaching insights:
- Most discriminative moment: 2-5 frames BEFORE contact (intent window)
- Contact frame itself looks similar across stroke types
- Pre-contact body loading reveals stroke INTENT (biomechanically unavoidable)

Smash Intent Score (SIS) formula:
    SIS = 0.35*E + 0.30*P + 0.15*N + 0.10*T + 0.10*C

    Where:
    E = Elbow lead (normalized 0-1)
    P = Pronation angular velocity
    N = Non-racket arm drop speed
    T = Torso rotation speed
    C = COM downward velocity

Expected accuracy boost: 15-20% based on coaching research

References:
    - Coach-validated SIS formula (02-RESEARCH.md)
    - Contact frame detection research
    - Intent window timing analysis
"""

import numpy as np
from scipy.ndimage import gaussian_filter1d
from src.data_processing.phase_segmentation import LANDMARKS, get_intent_window


def extract_contact_frame_features(pose_sequence, contact_frame):
    """
    Extract biomechanical features at the moment of contact.

    Key discriminative features from coaching research:
    - Forearm orientation: Clear > 90deg vertical angle, Smash < 90deg
    - Wrist-elbow relationship: Clear has wrist above elbow, Smash below
    - Arm extension: Smash has more extended arm at contact

    Args:
        pose_sequence: Array of shape (num_frames, 33, 3) - MediaPipe pose sequence
        contact_frame: int (frame index of contact)

    Returns:
        dict with contact frame features:
            - contact_forearm_vertical_angle: float (degrees, 0-180)
            - contact_forearm_pitch: float (elevation above horizontal)
            - contact_wrist_elbow_vertical: float (wrist Y - elbow Y)
            - contact_wrist_height: float (absolute wrist Y position)
            - contact_elbow_extension: float (elbow angle in degrees)
            - contact_arm_extension: float (shoulder-wrist distance)
            - contact_torso_lean: float (forward lean angle)
            - contact_shoulder_rotation: float (shoulder rotation angle)
    """
    if contact_frame >= len(pose_sequence):
        contact_frame = len(pose_sequence) - 1

    contact_pose = pose_sequence[contact_frame]

    # Detect dominant side
    r_wrist_heights = pose_sequence[:, LANDMARKS['right_wrist'], 1]
    l_wrist_heights = pose_sequence[:, LANDMARKS['left_wrist'], 1]
    handedness = 'right' if np.mean(r_wrist_heights) < np.mean(l_wrist_heights) else 'left'

    # Select landmarks based on handedness
    if handedness == 'right':
        shoulder_idx = LANDMARKS['right_shoulder']
        elbow_idx = LANDMARKS['right_elbow']
        wrist_idx = LANDMARKS['right_wrist']
    else:
        shoulder_idx = LANDMARKS['left_shoulder']
        elbow_idx = LANDMARKS['left_elbow']
        wrist_idx = LANDMARKS['left_wrist']

    # Extract joint positions
    r_shoulder = contact_pose[shoulder_idx]
    r_elbow = contact_pose[elbow_idx]
    r_wrist = contact_pose[wrist_idx]

    # === FOREARM ORIENTATION (KEY DISCRIMINATOR) ===
    forearm_vector = r_wrist - r_elbow
    forearm_mag = np.linalg.norm(forearm_vector)

    if forearm_mag > 1e-8:
        forearm_unit = forearm_vector / forearm_mag

        # Forearm vertical angle (Clear > 90deg, Smash < 90deg)
        vertical_down = np.array([0, 1, 0])  # Y-axis points down in MediaPipe
        dot_product = np.clip(np.dot(forearm_unit, vertical_down), -1, 1)
        forearm_vertical_angle = np.degrees(np.arccos(dot_product))

        # Forearm pitch (elevation above horizontal plane)
        horizontal_projection = np.sqrt(forearm_vector[0]**2 + forearm_vector[2]**2)
        forearm_pitch = np.degrees(np.arctan2(-forearm_vector[1], horizontal_projection))
    else:
        forearm_vertical_angle = np.nan
        forearm_pitch = np.nan

    # === WRIST-ELBOW RELATIONSHIP ===
    # Clear: wrist above elbow (negative Y diff), Smash: wrist below (positive Y diff)
    wrist_elbow_vertical = r_wrist[1] - r_elbow[1]
    wrist_height = r_wrist[1]

    # === ARM EXTENSION ===
    # Elbow angle (shoulder-elbow-wrist)
    elbow_extension = calculate_angle_3d(r_shoulder, r_elbow, r_wrist)

    # Arm extension distance (shoulder to wrist)
    arm_extension = np.linalg.norm(r_wrist - r_shoulder)

    # === BODY POSTURE ===
    # Torso lean (forward lean angle)
    left_shoulder = contact_pose[LANDMARKS['left_shoulder']]
    right_shoulder = contact_pose[LANDMARKS['right_shoulder']]
    left_hip = contact_pose[LANDMARKS['left_hip']]
    right_hip = contact_pose[LANDMARKS['right_hip']]

    shoulder_center = (left_shoulder + right_shoulder) / 2
    hip_center = (left_hip + right_hip) / 2
    torso_vector = shoulder_center - hip_center

    # Lean angle (deviation from vertical)
    vertical_up = np.array([0, -1, 0])  # Negative Y is up
    torso_mag = np.linalg.norm(torso_vector)
    if torso_mag > 1e-8:
        torso_unit = torso_vector / torso_mag
        torso_lean = np.degrees(np.arccos(np.clip(np.dot(torso_unit, vertical_up), -1, 1)))
    else:
        torso_lean = np.nan

    # Shoulder rotation (angle between shoulder line and frontal plane)
    shoulder_vector = right_shoulder - left_shoulder
    shoulder_rotation = np.degrees(np.arctan2(shoulder_vector[2], shoulder_vector[0]))

    return {
        'contact_forearm_vertical_angle': forearm_vertical_angle,
        'contact_forearm_pitch': forearm_pitch,
        'contact_wrist_elbow_vertical': wrist_elbow_vertical,
        'contact_wrist_height': wrist_height,
        'contact_elbow_extension': elbow_extension,
        'contact_arm_extension': arm_extension,
        'contact_torso_lean': torso_lean,
        'contact_shoulder_rotation': shoulder_rotation,
    }


def extract_intent_window_features(pose_sequence, contact_frame):
    """
    Extract features from intent window [contact-5 : contact-2] frames.

    This is the MOST discriminative window per coaching insights.
    Pre-contact body loading reveals stroke intent.

    Features computed within intent window:
    - Elbow lead (PRIMARY - 35% weight in SIS)
    - Forearm pronation velocity (PRIMARY - 30% weight)
    - Non-racket arm pull (SECONDARY - 15% weight)
    - Trunk rotation (TERTIARY - 10% weight)
    - COM direction (TERTIARY - 10% weight)

    Args:
        pose_sequence: Array of shape (num_frames, 33, 3) - MediaPipe pose sequence
        contact_frame: int (frame index of contact)

    Returns:
        dict with intent window features
    """
    # Get intent window using phase_segmentation function
    window_start, window_end = get_intent_window(contact_frame)

    # Handle edge case: window_end must be <= window_start if contact < 2
    if window_end < window_start:
        window_end = window_start

    # Extract window slice
    intent_window = pose_sequence[window_start:window_end+1]

    if len(intent_window) == 0:
        # Fallback: use single frame at contact if window empty
        intent_window = pose_sequence[max(0, contact_frame):contact_frame+1]

    # Detect dominant side
    r_wrist_heights = pose_sequence[:, LANDMARKS['right_wrist'], 1]
    l_wrist_heights = pose_sequence[:, LANDMARKS['left_wrist'], 1]
    handedness = 'right' if np.mean(r_wrist_heights) < np.mean(l_wrist_heights) else 'left'

    # Select landmarks
    if handedness == 'right':
        shoulder_idx = LANDMARKS['right_shoulder']
        elbow_idx = LANDMARKS['right_elbow']
        wrist_idx = LANDMARKS['right_wrist']
        non_racket_wrist_idx = LANDMARKS['left_wrist']
    else:
        shoulder_idx = LANDMARKS['left_shoulder']
        elbow_idx = LANDMARKS['left_elbow']
        wrist_idx = LANDMARKS['left_wrist']
        non_racket_wrist_idx = LANDMARKS['right_wrist']

    # === PRIMARY: ELBOW LEAD (35% weight in SIS) ===
    # Elbow forward position relative to shoulder
    elbow_positions = intent_window[:, elbow_idx, :]
    shoulder_positions = intent_window[:, shoulder_idx, :]

    elbow_forward = elbow_positions[:, 0] - shoulder_positions[:, 0]  # X-axis difference
    elbow_forward_mean = np.mean(elbow_forward)
    elbow_forward_max = np.max(elbow_forward)

    # === PRIMARY: FOREARM PRONATION VELOCITY (30% weight) ===
    # Angular velocity of forearm rotation
    wrist_positions = intent_window[:, wrist_idx, :]

    # Calculate forearm angles over time
    forearm_angles = []
    for i in range(len(intent_window)):
        forearm_vec = wrist_positions[i] - elbow_positions[i]
        # Angle in XY plane (pronation/supination)
        angle = np.arctan2(forearm_vec[1], forearm_vec[0])
        forearm_angles.append(angle)

    forearm_angles = np.array(forearm_angles)

    # Smooth before differentiation
    if len(forearm_angles) > 2:
        forearm_angles_smooth = gaussian_filter1d(forearm_angles, sigma=0.8)
        angular_velocity = np.diff(forearm_angles_smooth) * 30  # Assuming 30 fps
        forearm_rotation_vel = np.max(np.abs(angular_velocity)) if len(angular_velocity) > 0 else 0.0
    else:
        forearm_rotation_vel = 0.0

    # === SECONDARY: NON-RACKET ARM PULL-DOWN (15% weight) ===
    non_racket_wrist = intent_window[:, non_racket_wrist_idx, :]
    non_racket_arm_drop = 0.0

    if len(non_racket_wrist) > 1:
        # Vertical velocity (Y-axis)
        dy = np.diff(non_racket_wrist[:, 1])
        # Positive dy = downward motion (Y increases downward)
        non_racket_arm_drop = np.max(dy) if len(dy) > 0 else 0.0

    # === TERTIARY: TRUNK ROTATION (10% weight) ===
    left_shoulder_window = intent_window[:, LANDMARKS['left_shoulder'], :]
    right_shoulder_window = intent_window[:, LANDMARKS['right_shoulder'], :]
    left_hip_window = intent_window[:, LANDMARKS['left_hip'], :]
    right_hip_window = intent_window[:, LANDMARKS['right_hip'], :]

    # Calculate torso twist over time
    torso_twist_angles = []
    for i in range(len(intent_window)):
        shoulder_vec = right_shoulder_window[i] - left_shoulder_window[i]
        hip_vec = right_hip_window[i] - left_hip_window[i]

        shoulder_angle = np.arctan2(shoulder_vec[2], shoulder_vec[0])
        hip_angle = np.arctan2(hip_vec[2], hip_vec[0])

        torso_twist = shoulder_angle - hip_angle
        torso_twist_angles.append(torso_twist)

    torso_twist_angles = np.array(torso_twist_angles)

    if len(torso_twist_angles) > 2:
        torso_twist_smooth = gaussian_filter1d(torso_twist_angles, sigma=0.8)
        torso_rotation_vel = np.max(np.abs(np.diff(torso_twist_smooth))) * 30 if len(torso_twist_smooth) > 1 else 0.0
    else:
        torso_rotation_vel = 0.0

    # === TERTIARY: COM DIRECTION (10% weight) ===
    shoulder_center = (left_shoulder_window + right_shoulder_window) / 2
    hip_center = (left_hip_window + right_hip_window) / 2
    com = (shoulder_center + hip_center) / 2

    if len(com) > 1:
        com_velocity = np.diff(com, axis=0)
        com_velocity_y = np.mean(com_velocity[:, 1])  # Vertical component
    else:
        com_velocity_y = 0.0

    return {
        'intent_elbow_forward_mean': elbow_forward_mean,
        'intent_elbow_forward_max': elbow_forward_max,
        'intent_forearm_rotation_vel': forearm_rotation_vel,
        'intent_nonracket_arm_drop': non_racket_arm_drop,
        'intent_torso_rotation_vel': torso_rotation_vel,
        'intent_com_velocity_y': com_velocity_y,
    }


def calculate_smash_intent_score(intent_features):
    """
    Calculate Smash Intent Score (SIS) using coach-validated formula.

    Formula:
        SIS = 0.35*E + 0.30*P + 0.15*N + 0.10*T + 0.10*C

    Where (normalized 0-1 based on research reference values):
        E = Elbow lead
        P = Pronation angular velocity
        N = Non-racket arm drop speed
        T = Torso rotation speed
        C = COM downward velocity

    Thresholds:
        SIS >= 0.65: Smash
        0.40 <= SIS < 0.65: Deceptive/Half-smash
        SIS < 0.40: Clear

    Args:
        intent_features: dict from extract_intent_window_features()

    Returns:
        dict with:
            - smash_intent_score: float (0-1 scale)
            - classification_hint: str ('smash', 'deceptive', 'clear')
            - component_scores: dict (E, P, N, T, C individual scores)
    """
    # Reference values for normalization (from coaching research)
    # These are approximate values for elite players
    REFERENCE_VALUES = {
        'elbow_forward': (0.05, 0.20),  # (min, max) in normalized coordinates
        'pronation_vel': (5.0, 20.0),   # radians/second
        'arm_drop': (0.02, 0.15),       # normalized Y change
        'torso_rotation': (0.5, 3.0),   # radians/second
        'com_velocity_y': (0.01, 0.10), # normalized Y velocity
    }

    # Extract raw values
    elbow_forward = intent_features['intent_elbow_forward_max']
    pronation_vel = intent_features['intent_forearm_rotation_vel']
    arm_drop = intent_features['intent_nonracket_arm_drop']
    torso_rotation = intent_features['intent_torso_rotation_vel']
    com_velocity_y = intent_features['intent_com_velocity_y']

    # Normalize to 0-1 scale
    def normalize(value, min_val, max_val):
        """Normalize value to 0-1 scale, clipping extremes"""
        normalized = (value - min_val) / (max_val - min_val + 1e-8)
        return np.clip(normalized, 0.0, 1.0)

    E = normalize(elbow_forward, *REFERENCE_VALUES['elbow_forward'])
    P = normalize(pronation_vel, *REFERENCE_VALUES['pronation_vel'])
    N = normalize(arm_drop, *REFERENCE_VALUES['arm_drop'])
    T = normalize(torso_rotation, *REFERENCE_VALUES['torso_rotation'])
    C = normalize(com_velocity_y, *REFERENCE_VALUES['com_velocity_y'])

    # Calculate SIS with coach-validated weights
    sis = 0.35 * E + 0.30 * P + 0.15 * N + 0.10 * T + 0.10 * C

    # Classification hint based on thresholds
    if sis >= 0.65:
        classification_hint = 'smash'
    elif sis >= 0.40:
        classification_hint = 'deceptive'
    else:
        classification_hint = 'clear'

    return {
        'smash_intent_score': sis,
        'classification_hint': classification_hint,
        'component_scores': {
            'E_elbow_lead': E,
            'P_pronation': P,
            'N_nonracket_arm': N,
            'T_torso_rotation': T,
            'C_com_velocity': C
        }
    }


def calculate_angle_3d(p1, p2, p3):
    """
    Calculate 3D angle at p2 formed by p1-p2-p3.

    Args:
        p1, p2, p3: 3D points as numpy arrays

    Returns:
        float: Angle in degrees (0-180)
    """
    v1 = p1 - p2
    v2 = p3 - p2

    v1_mag = np.linalg.norm(v1)
    v2_mag = np.linalg.norm(v2)

    if v1_mag < 1e-8 or v2_mag < 1e-8:
        return np.nan

    cosine = np.dot(v1, v2) / (v1_mag * v2_mag)
    cosine = np.clip(cosine, -1.0, 1.0)

    return np.degrees(np.arccos(cosine))
