#!/usr/bin/env python3
"""
Kinetic Chain Timing Features - Priority 0 (P0)

Extracts sequential timing features through the kinetic chain:
    Hip -> Trunk -> Shoulder -> Elbow -> Wrist

Critical insight from coaching research:
- Kinetic chain timing MUST be measured within forward swing phase ONLY
- Measuring across full sequence finds preparation peaks, not power generation
- Sequential activation pattern is biomechanically unavoidable in power strokes

Expected accuracy boost: 15-20% based on coaching research

References:
    - Biomechanical Principles Applied to Badminton Power Strokes
    - Kinetic chain timing research (BetterMinton Service)
    - Coaching insights (02-RESEARCH.md)
"""

import numpy as np
from scipy.ndimage import gaussian_filter1d
from src.data_processing.phase_segmentation import LANDMARKS


def detect_dominant_side(pose_sequence):
    """
    Determine if player is right or left handed.

    Uses arm elevation comparison during stroke to identify racket-holding hand.
    For overhead strokes, racket arm typically has higher average elevation.

    Args:
        pose_sequence: Array of shape (num_frames, 33, 3) - MediaPipe pose sequence

    Returns:
        'right' or 'left'
    """
    # Average wrist heights across sequence
    r_wrist_heights = pose_sequence[:, LANDMARKS['right_wrist'], 1]
    l_wrist_heights = pose_sequence[:, LANDMARKS['left_wrist'], 1]

    # Racket-holding hand typically higher on average (overhead strokes)
    # Note: Y-axis in MediaPipe is inverted (lower Y = higher in image)
    r_avg_height = np.mean(r_wrist_heights)
    l_avg_height = np.mean(l_wrist_heights)

    # Lower Y value = higher position
    # If right wrist has lower Y (higher position), player is right-handed
    if r_avg_height < l_avg_height:
        return 'right'
    else:
        return 'left'


def extract_kinetic_chain_timing(pose_sequence, phases):
    """
    Extract sequential timing delays through kinetic chain.

    CRITICAL: Measures timing ONLY within forward swing phase, not full sequence.
    This prevents preparation movement peaks from contaminating power generation timing.

    Kinetic chain segments (sequential activation):
        1. Hip: Root of kinetic chain
        2. Trunk: Core rotation and transfer
        3. Shoulder: Upper body segment
        4. Elbow: Mid-arm segment
        5. Wrist: Terminal segment (racket head proxy)

    Args:
        pose_sequence: Array of shape (num_frames, 33, 3) - MediaPipe pose sequence
        phases: dict from phase_segmentation.segment_stroke_phases()
                Must contain 'forward_swing' and 'handedness' keys

    Returns:
        dict with:
            - hip_to_trunk_delay: int (frames between hip and trunk peaks)
            - trunk_to_shoulder_delay: int (frames)
            - shoulder_to_elbow_delay: int (frames)
            - elbow_to_wrist_delay: int (frames)
            - hip_to_wrist_total: int (total chain time in frames)
            - chain_sequence_valid: bool (True if delays are positive)
            - coordination_efficiency: float (0-1, based on sequence validity)
    """
    # Detect dominant side for segment selection
    handedness = phases.get('handedness')
    if handedness is None:
        handedness = detect_dominant_side(pose_sequence)

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

    # Extract forward swing phase boundaries
    forward_swing_start, forward_swing_end = phases['forward_swing']

    # Define segments to track
    segments = {
        'hip': hip_idx,
        'trunk': None,  # Will calculate as midpoint
        'shoulder': shoulder_idx,
        'elbow': elbow_idx,
        'wrist': wrist_idx,
    }

    # Calculate velocity peaks for each segment WITHIN forward swing only
    peak_times = {}

    for segment_name, landmark_idx in segments.items():
        if segment_name == 'trunk':
            # Trunk: midpoint between shoulders and hips
            left_shoulder = pose_sequence[:, LANDMARKS['left_shoulder'], :]
            right_shoulder = pose_sequence[:, LANDMARKS['right_shoulder'], :]
            left_hip = pose_sequence[:, LANDMARKS['left_hip'], :]
            right_hip = pose_sequence[:, LANDMARKS['right_hip'], :]

            shoulder_center = (left_shoulder + right_shoulder) / 2
            hip_center = (left_hip + right_hip) / 2
            positions = (shoulder_center + hip_center) / 2
        else:
            positions = pose_sequence[:, landmark_idx, :]

        # Smooth to reduce MediaPipe jitter
        positions_smooth = gaussian_filter1d(positions, sigma=1.5, axis=0)

        # Calculate velocity magnitude
        velocities = np.diff(positions_smooth, axis=0)
        velocity_mag = np.linalg.norm(velocities, axis=1)

        # Find peak ONLY in forward swing phase
        swing_velocities = velocity_mag[forward_swing_start:forward_swing_end]

        if len(swing_velocities) > 0:
            local_peak = np.argmax(swing_velocities)
            # Convert to global frame index
            peak_times[segment_name] = forward_swing_start + local_peak
        else:
            # Fallback if swing phase too short
            peak_times[segment_name] = forward_swing_start

    # Calculate sequential timing deltas (in frames)
    hip_to_trunk = peak_times['trunk'] - peak_times['hip']
    trunk_to_shoulder = peak_times['shoulder'] - peak_times['trunk']
    shoulder_to_elbow = peak_times['elbow'] - peak_times['shoulder']
    elbow_to_wrist = peak_times['wrist'] - peak_times['elbow']
    hip_to_wrist_total = peak_times['wrist'] - peak_times['hip']

    # Validate sequence (all delays should be non-negative for valid coordination)
    chain_sequence_valid = (
        hip_to_trunk >= 0 and
        trunk_to_shoulder >= 0 and
        shoulder_to_elbow >= 0 and
        elbow_to_wrist >= 0
    )

    # Calculate coordination efficiency
    coordination_efficiency = calculate_coordination_efficiency({
        'hip_to_trunk_delay': hip_to_trunk,
        'trunk_to_shoulder_delay': trunk_to_shoulder,
        'shoulder_to_elbow_delay': shoulder_to_elbow,
        'elbow_to_wrist_delay': elbow_to_wrist,
        'chain_sequence_valid': chain_sequence_valid
    })

    return {
        'hip_to_trunk_delay': hip_to_trunk,
        'trunk_to_shoulder_delay': trunk_to_shoulder,
        'shoulder_to_elbow_delay': shoulder_to_elbow,
        'elbow_to_wrist_delay': elbow_to_wrist,
        'hip_to_wrist_total': hip_to_wrist_total,
        'chain_sequence_valid': chain_sequence_valid,
        'coordination_efficiency': coordination_efficiency
    }


def calculate_coordination_efficiency(timing_features):
    """
    Calculate coordination efficiency score based on kinetic chain timing.

    A valid kinetic chain has positive sequential delays (hip < trunk < shoulder < elbow < wrist).
    Out-of-order peaks indicate coordination failure or poor technique.

    Args:
        timing_features: dict with delay features and chain_sequence_valid flag

    Returns:
        float: Efficiency score (0-1 scale)
            - 1.0: Perfect sequential activation
            - 0.5-0.9: Some delays negative but mostly sequential
            - 0.0-0.4: Multiple violations, poor coordination
    """
    if timing_features['chain_sequence_valid']:
        # Perfect sequence
        return 1.0

    # Calculate penalty based on number of negative delays
    delays = [
        timing_features['hip_to_trunk_delay'],
        timing_features['trunk_to_shoulder_delay'],
        timing_features['shoulder_to_elbow_delay'],
        timing_features['elbow_to_wrist_delay']
    ]

    # Count positive delays
    positive_count = sum(1 for d in delays if d >= 0)
    total_delays = len(delays)

    # Efficiency = ratio of valid transitions
    efficiency = positive_count / total_delays

    # Apply penalty for severe violations (large negative delays)
    max_negative = min(delays)
    if max_negative < -5:  # More than 5 frames out of order
        efficiency *= 0.5  # Heavy penalty

    return efficiency
