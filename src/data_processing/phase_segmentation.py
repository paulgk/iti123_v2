#!/usr/bin/env python3
"""
Phase Segmentation Module - Velocity-Based Detection

Identifies 5 stroke phases from pose sequence using velocity-based peak detection:
1. Preparation
2. Backswing
3. Forward swing
4. Contact (single frame at peak velocity)
5. Follow-through

Based on coaching insights and biomechanics research:
- Contact frame detected at peak velocity (NOT peak position)
- Intent window: [contact-5 : contact-2] frames (most discriminative)
- Phase boundaries validated against biomechanical timing constraints

References:
    - Biomechanical Principles Applied to Badminton Power Strokes
    - Coach-validated timing patterns (see 02-RESEARCH.md)
"""

import numpy as np
from scipy.signal import find_peaks
from scipy.ndimage import gaussian_filter1d


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

# Research reference timings (from biomechanics literature)
REFERENCE_TIMINGS = {
    'forward_swing_pct_min': 0.20,  # 20% of total duration
    'forward_swing_pct_max': 0.50,  # 50% of total duration
    'contact_position_min': 0.30,   # 30% of sequence position
    'contact_position_max': 0.80,   # 80% of sequence position
    'min_phase_duration': 3,        # Minimum frames per phase
}


def detect_handedness(pose_sequence):
    """
    Detect if player is left or right handed based on wrist height.

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
    if r_avg_height < l_avg_height:
        return 'right'
    else:
        return 'left'


def mirror_pose_for_lefthanded(pose_sequence):
    """
    Mirror X-coordinates for left-handed players.

    Args:
        pose_sequence: Array of shape (num_frames, 33, 3)

    Returns:
        Mirrored pose sequence
    """
    mirrored = pose_sequence.copy()
    mirrored[:, :, 0] = 1 - mirrored[:, :, 0]  # Mirror X-coordinates
    return mirrored


def detect_contact_frame(pose_sequence, fps=30, handedness='right'):
    """
    Detect contact frame at peak velocity (NOT peak position).

    Critical insight from coaching research:
    - Contact occurs at PEAK VELOCITY, not peak position
    - Peak position occurs 0.02-0.05s AFTER velocity peak

    Args:
        pose_sequence: Array of shape (num_frames, 33, 3) - MediaPipe pose sequence
        fps: Frames per second (default 30)
        handedness: 'right' or 'left' (which hand holds racket)

    Returns:
        dict with:
            - contact_frame: int (frame index)
            - contact_position_pct: float (position as % of sequence)
            - peak_velocity: float (velocity magnitude at contact)
            - valid: bool (whether contact position is reasonable)
    """
    # Extract wrist positions
    if handedness == 'right':
        wrist_landmark = LANDMARKS['right_wrist']
    else:
        wrist_landmark = LANDMARKS['left_wrist']

    wrist_positions = pose_sequence[:, wrist_landmark, :]  # (num_frames, 3)

    # Smooth positions to reduce MediaPipe jitter
    positions_smooth = gaussian_filter1d(wrist_positions, sigma=1.5, axis=0)

    # Calculate velocity magnitude
    velocities = np.diff(positions_smooth, axis=0)
    velocity_mag = np.linalg.norm(velocities, axis=1)

    # Contact frame = peak velocity
    contact_frame = np.argmax(velocity_mag)

    # Validation: contact should be in 30-80% of sequence
    total_frames = len(velocity_mag)
    contact_position_pct = contact_frame / total_frames

    valid = (REFERENCE_TIMINGS['contact_position_min'] <= contact_position_pct <=
             REFERENCE_TIMINGS['contact_position_max'])

    return {
        'contact_frame': contact_frame,
        'contact_position_pct': contact_position_pct,
        'peak_velocity': velocity_mag[contact_frame],
        'valid': valid
    }


def get_intent_window(contact_frame):
    """
    Get intent window: [contact-5 : contact-2] frames.

    Critical coaching insight:
    - Most discriminative moment is 2-5 frames BEFORE contact
    - Pre-contact body loading reveals stroke INTENT (unavoidable biomechanically)
    - At 30fps: 70-160ms before shuttle contact

    Args:
        contact_frame: int (frame index of contact)

    Returns:
        tuple (start_frame, end_frame) - inclusive range
    """
    # Clamp to prevent negative indices
    start_frame = max(0, contact_frame - 5)
    end_frame = max(0, contact_frame - 2)

    return (start_frame, end_frame)


def segment_stroke_phases(pose_sequence, fps=30, handedness=None):
    """
    Segment badminton stroke into 5 phases using velocity-based detection.

    Phases:
        1. Preparation: Start to backswing
        2. Backswing: First acceleration phase
        3. Forward swing: Peak acceleration to contact
        4. Contact: Single frame at peak velocity
        5. Follow-through: Post-contact deceleration

    Algorithm:
        - Contact frame: peak wrist velocity (NOT position)
        - Backswing start: first significant velocity peak before contact
        - Forward swing start: calculated from contact using research timing

    Args:
        pose_sequence: Array of shape (num_frames, 33, 3) - MediaPipe pose sequence
        fps: Frames per second (default 30)
        handedness: 'right' or 'left' (auto-detect if None)

    Returns:
        dict with phase boundaries:
            - preparation: (start_frame, end_frame)
            - backswing: (start_frame, end_frame)
            - forward_swing: (start_frame, end_frame)
            - contact: int (single frame)
            - follow_through: (start_frame, end_frame)
            - handedness: str (detected handedness)
    """
    # Auto-detect handedness if not provided
    if handedness is None:
        handedness = detect_handedness(pose_sequence)

    # Extract wrist positions
    if handedness == 'right':
        wrist_landmark = LANDMARKS['right_wrist']
    else:
        wrist_landmark = LANDMARKS['left_wrist']

    wrist_positions = pose_sequence[:, wrist_landmark, :]

    # Smooth positions to reduce noise
    positions_smooth = gaussian_filter1d(wrist_positions, sigma=1.5, axis=0)

    # Calculate velocity magnitude
    velocities = np.diff(positions_smooth, axis=0)
    velocity_mag = np.linalg.norm(velocities, axis=1)

    # === CONTACT FRAME (peak velocity) ===
    contact_frame = np.argmax(velocity_mag)

    # === FIND PHASE BOUNDARIES ===
    # Use find_peaks with biomechanically-informed parameters
    mean_vel = np.mean(velocity_mag)
    std_vel = np.std(velocity_mag)

    peaks, properties = find_peaks(
        velocity_mag,
        height=mean_vel * 0.3,      # Velocity > 30% of mean (filters noise)
        distance=5,                  # Min 5 frames between peaks
        prominence=std_vel * 0.5     # Significant peaks (0.5 * std)
    )

    # === BACKSWING START ===
    # First significant peak before contact
    backswing_candidates = peaks[peaks < contact_frame]

    if len(backswing_candidates) > 0:
        backswing_start = backswing_candidates[0]
    else:
        # Fallback: 30% of contact frame
        backswing_start = int(contact_frame * 0.3)

    # === FORWARD SWING START ===
    # Research: forward swing is ~30-40% of total duration
    # Start at approximately 35% before contact
    forward_swing_start = max(
        backswing_start + 3,  # At least 3 frames after backswing
        int(contact_frame - len(velocity_mag) * 0.35)
    )

    # === FOLLOW-THROUGH ===
    follow_through_start = contact_frame + 1

    # Construct phase dictionary
    phases = {
        'preparation': (0, backswing_start),
        'backswing': (backswing_start, forward_swing_start),
        'forward_swing': (forward_swing_start, contact_frame),
        'contact': contact_frame,
        'follow_through': (follow_through_start, len(velocity_mag)),
        'handedness': handedness
    }

    return phases


def validate_phase_boundaries(phases, total_frames):
    """
    Validate phase boundaries against biomechanical constraints.

    Validation checks:
        1. Minimum phase durations (>= 3 frames each)
        2. Contact frame position (30-80% of sequence)
        3. Forward swing duration (20-50% of total)
        4. Sequential ordering (prep < backswing < forward < contact < follow)

    Args:
        phases: dict from segment_stroke_phases()
        total_frames: int (total number of frames in sequence)

    Returns:
        dict with validation results:
            - valid: bool (overall validity)
            - checks: dict of individual check results
            - contact_position_pct: float
            - forward_swing_pct: float
            - phase_durations_frames: dict
            - warnings: list of warning messages
    """
    checks = {}
    warnings = []

    # Calculate phase durations
    prep_dur = phases['preparation'][1] - phases['preparation'][0]
    backswing_dur = phases['backswing'][1] - phases['backswing'][0]
    forward_dur = phases['forward_swing'][1] - phases['forward_swing'][0]
    follow_dur = phases['follow_through'][1] - phases['follow_through'][0]

    phase_durations = {
        'preparation': prep_dur,
        'backswing': backswing_dur,
        'forward_swing': forward_dur,
        'follow_through': follow_dur
    }

    # === CHECK 1: Minimum durations ===
    min_dur = REFERENCE_TIMINGS['min_phase_duration']

    min_durations_valid = True
    if prep_dur < min_dur:
        min_durations_valid = False
        warnings.append(f"Preparation too short: {prep_dur} frames (min: {min_dur})")
    if backswing_dur < min_dur:
        min_durations_valid = False
        warnings.append(f"Backswing too short: {backswing_dur} frames (min: {min_dur})")
    if forward_dur < min_dur:
        min_durations_valid = False
        warnings.append(f"Forward swing too short: {forward_dur} frames (min: {min_dur})")
    if follow_dur < min_dur:
        min_durations_valid = False
        warnings.append(f"Follow-through too short: {follow_dur} frames (min: {min_dur})")

    checks['min_durations'] = min_durations_valid

    # === CHECK 2: Contact position ===
    contact_frame = phases['contact']
    contact_pct = contact_frame / total_frames

    contact_valid = (REFERENCE_TIMINGS['contact_position_min'] <= contact_pct <=
                     REFERENCE_TIMINGS['contact_position_max'])

    if not contact_valid:
        warnings.append(
            f"Contact at {contact_pct*100:.1f}% of sequence "
            f"(expected {REFERENCE_TIMINGS['contact_position_min']*100:.0f}%-"
            f"{REFERENCE_TIMINGS['contact_position_max']*100:.0f}%)"
        )

    checks['contact_position'] = contact_valid

    # === CHECK 3: Forward swing duration ===
    forward_pct = forward_dur / total_frames

    forward_valid = (REFERENCE_TIMINGS['forward_swing_pct_min'] <= forward_pct <=
                     REFERENCE_TIMINGS['forward_swing_pct_max'])

    if not forward_valid:
        warnings.append(
            f"Forward swing {forward_pct*100:.1f}% of sequence "
            f"(expected {REFERENCE_TIMINGS['forward_swing_pct_min']*100:.0f}%-"
            f"{REFERENCE_TIMINGS['forward_swing_pct_max']*100:.0f}%)"
        )

    checks['forward_swing_duration'] = forward_valid

    # === CHECK 4: Sequential ordering ===
    prep_end = phases['preparation'][1]
    back_start = phases['backswing'][0]
    back_end = phases['backswing'][1]
    forward_start = phases['forward_swing'][0]
    forward_end = phases['forward_swing'][1]
    contact = phases['contact']
    follow_start = phases['follow_through'][0]

    sequential_valid = (
        prep_end == back_start and
        back_end == forward_start and
        forward_end == contact and
        contact + 1 == follow_start
    )

    if not sequential_valid:
        warnings.append("Phase boundaries not sequential")

    checks['sequential_order'] = sequential_valid

    # Overall validity
    overall_valid = all(checks.values())

    return {
        'valid': overall_valid,
        'checks': checks,
        'contact_position_pct': contact_pct,
        'forward_swing_pct': forward_pct,
        'phase_durations_frames': phase_durations,
        'warnings': warnings
    }


def segment_and_validate(pose_sequence, fps=30, handedness=None):
    """
    Combine segmentation and validation in single call.

    Args:
        pose_sequence: Array of shape (num_frames, 33, 3) - MediaPipe pose sequence
        fps: Frames per second (default 30)
        handedness: 'right' or 'left' (auto-detect if None)

    Returns:
        tuple (phases, validation_result)
            - phases: dict with phase boundaries
            - validation_result: dict with validation checks
    """
    phases = segment_stroke_phases(pose_sequence, fps=fps, handedness=handedness)

    # Total frames is length of velocity vector (one less than pose sequence)
    total_frames = len(pose_sequence) - 1

    validation_result = validate_phase_boundaries(phases, total_frames)

    return phases, validation_result


def calculate_phase_statistics(phases, total_frames):
    """
    Calculate phase duration statistics and compare to research reference timings.

    Args:
        phases: dict from segment_stroke_phases()
        total_frames: int (total number of frames in sequence)

    Returns:
        dict with:
            - duration_percentages: dict (each phase as % of total)
            - research_reference: dict (expected percentages from literature)
            - deviations: dict (actual - expected for each phase)
    """
    # Calculate actual durations
    prep_dur = phases['preparation'][1] - phases['preparation'][0]
    backswing_dur = phases['backswing'][1] - phases['backswing'][0]
    forward_dur = phases['forward_swing'][1] - phases['forward_swing'][0]
    follow_dur = phases['follow_through'][1] - phases['follow_through'][0]

    # Convert to percentages
    duration_percentages = {
        'preparation': (prep_dur / total_frames) * 100,
        'backswing': (backswing_dur / total_frames) * 100,
        'forward_swing': (forward_dur / total_frames) * 100,
        'follow_through': (follow_dur / total_frames) * 100
    }

    # Research reference (approximate ranges from biomechanics literature)
    research_reference = {
        'forward_swing': (30.0, 40.0),  # 30-40% expected
        'contact_position': (40.0, 70.0),  # 40-70% of sequence
        'backswing': 'variable',  # Skill-dependent
    }

    # Calculate deviations for forward swing (well-established in research)
    forward_swing_pct = duration_percentages['forward_swing']
    forward_expected_mid = (research_reference['forward_swing'][0] +
                            research_reference['forward_swing'][1]) / 2

    deviations = {
        'forward_swing': forward_swing_pct - forward_expected_mid
    }

    return {
        'duration_percentages': duration_percentages,
        'research_reference': research_reference,
        'deviations': deviations
    }
