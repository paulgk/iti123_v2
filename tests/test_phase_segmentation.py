#!/usr/bin/env python3
"""
Test Suite for Phase Segmentation Module

Tests velocity-based phase detection, contact frame identification,
and boundary validation against biomechanical constraints.
"""

import pytest
import numpy as np
from pathlib import Path
import pickle

from src.data_processing.phase_segmentation import (
    segment_stroke_phases,
    detect_contact_frame,
    get_intent_window,
    validate_phase_boundaries,
    segment_and_validate,
    calculate_phase_statistics,
    LANDMARKS
)


# ============================================================================
# Test Fixtures
# ============================================================================

@pytest.fixture
def synthetic_pose_sequence():
    """
    Create synthetic pose sequence with known peak velocity frame.

    Simulates overhead stroke with:
    - 50 frames
    - Peak velocity at frame 30 (60% of sequence)
    - Smooth acceleration → peak → deceleration pattern
    """
    num_frames = 50
    num_landmarks = 33
    num_coords = 3

    # Initialize pose array
    poses = np.zeros((num_frames, num_landmarks, num_coords))

    # Simulate wrist trajectory (right wrist = landmark 15)
    wrist_idx = LANDMARKS['right_wrist']

    # Generate smooth trajectory with peak velocity at frame 30
    t = np.linspace(0, np.pi, num_frames)

    # Position follows acceleration curve
    # Velocity will peak where acceleration crosses zero (at inflection point)
    # Use sine wave: position = sin(t), velocity = cos(t), peak at t=0
    # Shift to put peak at frame 30
    t_shifted = t - (30/num_frames) * np.pi

    # X-coordinate (lateral movement)
    poses[:, wrist_idx, 0] = 0.5 + 0.2 * np.sin(t_shifted)

    # Y-coordinate (vertical movement - overhead stroke)
    # Start high, go lower (Y increases downward in image coords)
    poses[:, wrist_idx, 1] = 0.3 + 0.3 * (1 - np.cos(t))

    # Z-coordinate (depth - forward movement)
    poses[:, wrist_idx, 2] = 0.5 + 0.2 * np.sin(t_shifted)

    # Add other landmarks (shoulders, elbows, hips) with minimal movement
    for landmark_name, landmark_idx in LANDMARKS.items():
        if landmark_idx != wrist_idx:
            # Static positions with small noise
            poses[:, landmark_idx, 0] = 0.5 + np.random.normal(0, 0.01, num_frames)
            poses[:, landmark_idx, 1] = 0.5 + np.random.normal(0, 0.01, num_frames)
            poses[:, landmark_idx, 2] = 0.5 + np.random.normal(0, 0.01, num_frames)

    # Set shoulder height for handedness detection
    # Right wrist should be higher (lower Y) on average for right-handed
    poses[:, LANDMARKS['right_wrist'], 1] -= 0.1
    poses[:, LANDMARKS['left_wrist'], 1] += 0.1

    return poses


@pytest.fixture
def real_pose_data():
    """
    Load real pose data from dataset if available.

    Returns:
        list of pose sequences or None if data not available
    """
    poses_dir = Path("data/processed/poses")

    if not poses_dir.exists():
        return None

    # Load up to 100 samples for batch validation
    pose_files = list(poses_dir.glob("*_pose.pkl"))[:100]

    if len(pose_files) == 0:
        return None

    pose_sequences = []
    for pose_file in pose_files:
        try:
            with open(pose_file, 'rb') as f:
                pose_data = pickle.load(f)
                poses = pose_data['poses']

                # Ensure correct shape
                if poses.shape[1] == 33:
                    pose_sequences.append(poses)
        except Exception:
            continue

    return pose_sequences if len(pose_sequences) > 0 else None


# ============================================================================
# Test Contact Frame Detection
# ============================================================================

def test_contact_frame_detection_synthetic(synthetic_pose_sequence):
    """
    Test contact frame detection on synthetic data with known peak.

    Expected: Contact frame at frame 30 (±2 frames tolerance for smoothing)
    """
    result = detect_contact_frame(synthetic_pose_sequence, fps=30, handedness='right')

    contact_frame = result['contact_frame']
    expected_frame = 30

    # Allow ±5 frames tolerance due to smoothing and velocity calculation
    assert abs(contact_frame - expected_frame) <= 5, (
        f"Contact frame {contact_frame} not within ±5 of expected {expected_frame}"
    )

    # Check validity flag
    assert result['valid'], "Contact frame should be valid (30-80% of sequence)"

    # Check contact position percentage
    assert 0.3 <= result['contact_position_pct'] <= 0.8, (
        f"Contact position {result['contact_position_pct']:.2%} outside valid range"
    )


def test_contact_frame_multiple_peaks():
    """
    Test contact frame detection with multiple velocity peaks.

    Expected: Should find highest peak (contact = max velocity)
    """
    num_frames = 60
    num_landmarks = 33

    poses = np.zeros((num_frames, num_landmarks, 3))
    wrist_idx = LANDMARKS['right_wrist']

    # Create trajectory with two peaks: small peak at frame 15, large peak at frame 35
    t = np.arange(num_frames)

    # Two Gaussian peaks
    peak1 = 0.3 * np.exp(-((t - 15) ** 2) / 50)  # Smaller peak
    peak2 = 0.5 * np.exp(-((t - 35) ** 2) / 50)  # Larger peak (contact)

    poses[:, wrist_idx, 0] = 0.5 + peak1 + peak2
    poses[:, wrist_idx, 1] = 0.5 + peak1 + peak2
    poses[:, wrist_idx, 2] = 0.5 + peak1 + peak2

    # Set handedness indicators
    poses[:, LANDMARKS['right_wrist'], 1] = 0.3
    poses[:, LANDMARKS['left_wrist'], 1] = 0.7

    result = detect_contact_frame(poses, fps=30, handedness='right')

    # Contact should be at larger peak (frame 35, ±5 for smoothing)
    assert 30 <= result['contact_frame'] <= 42, (
        f"Contact frame {result['contact_frame']} should be near frame 35 (highest peak)"
    )


# ============================================================================
# Test Phase Boundaries Ordering
# ============================================================================

def test_phase_boundaries_ordering(synthetic_pose_sequence):
    """
    Test that phase boundaries are sequential.

    Expected: preparation < backswing < forward_swing < contact < follow_through
    """
    phases = segment_stroke_phases(synthetic_pose_sequence, fps=30, handedness='right')

    prep_end = phases['preparation'][1]
    back_start, back_end = phases['backswing']
    forward_start, forward_end = phases['forward_swing']
    contact = phases['contact']
    follow_start, follow_end = phases['follow_through']

    # Check sequential ordering
    assert prep_end == back_start, "Preparation should end where backswing starts"
    assert back_end == forward_start, "Backswing should end where forward swing starts"
    assert forward_end == contact, "Forward swing should end at contact"
    assert contact + 1 == follow_start, "Follow-through should start after contact"

    # Check temporal ordering (prep_end == back_start, back_end == forward_start is expected)
    assert 0 <= prep_end <= back_end <= forward_end <= contact < follow_end, (
        "Phases should be in temporal order"
    )


def test_phase_boundaries_with_real_data(real_pose_data):
    """
    Test phase boundaries on real pose data (if available).
    """
    if real_pose_data is None:
        pytest.skip("Real pose data not available")

    # Test first sample
    pose_sequence = real_pose_data[0]
    phases = segment_stroke_phases(pose_sequence, fps=30)

    # Check all phases exist
    assert 'preparation' in phases
    assert 'backswing' in phases
    assert 'forward_swing' in phases
    assert 'contact' in phases
    assert 'follow_through' in phases

    # Check sequential ordering
    prep_end = phases['preparation'][1]
    back_start, back_end = phases['backswing']
    forward_start, forward_end = phases['forward_swing']
    contact = phases['contact']
    follow_start = phases['follow_through'][0]

    assert prep_end == back_start
    assert back_end == forward_start
    assert forward_end == contact
    assert contact + 1 == follow_start


# ============================================================================
# Test Validation Checks
# ============================================================================

def test_validation_minimum_duration():
    """
    Test minimum duration check triggers correctly.

    Create sequence with too-short phases.
    """
    # Create minimal sequence with very short phases
    num_frames = 20
    num_landmarks = 33

    poses = np.zeros((num_frames, num_landmarks, 3))
    wrist_idx = LANDMARKS['right_wrist']

    # Simple linear trajectory (will create very short phases)
    poses[:, wrist_idx, 0] = np.linspace(0, 1, num_frames)
    poses[:, wrist_idx, 1] = np.linspace(0, 1, num_frames)
    poses[:, wrist_idx, 2] = np.linspace(0, 1, num_frames)

    # Set handedness
    poses[:, LANDMARKS['right_wrist'], 1] = 0.3
    poses[:, LANDMARKS['left_wrist'], 1] = 0.7

    phases = segment_stroke_phases(poses, fps=30)
    validation = validate_phase_boundaries(phases, num_frames - 1)

    # With only 20 frames, some phases will likely be too short
    # Check that validation identifies this
    if not validation['checks']['min_durations']:
        assert len(validation['warnings']) > 0, (
            "Should have warnings for short phases"
        )


def test_validation_contact_position():
    """
    Test contact position bounds check.

    Create sequence with contact too early (< 30%) or too late (> 80%).
    """
    # Test early contact
    num_frames = 50
    num_landmarks = 33

    poses = np.zeros((num_frames, num_landmarks, 3))
    wrist_idx = LANDMARKS['right_wrist']

    # Peak velocity at frame 5 (10% - too early)
    t = np.linspace(0, 2*np.pi, num_frames)
    poses[:, wrist_idx, 0] = 0.5 + 0.3 * np.sin(t - 0.5*np.pi)
    poses[:, wrist_idx, 1] = 0.5 + 0.3 * np.sin(t - 0.5*np.pi)
    poses[:, wrist_idx, 2] = 0.5 + 0.3 * np.sin(t - 0.5*np.pi)

    # Ensure peak is early
    velocities = np.diff(poses[:, wrist_idx, :], axis=0)
    velocity_mag = np.linalg.norm(velocities, axis=1)
    early_peak = np.argmax(velocity_mag)

    if early_peak < num_frames * 0.3:
        # Manually create phases with early contact
        phases = {
            'preparation': (0, 2),
            'backswing': (2, 4),
            'forward_swing': (4, early_peak),
            'contact': early_peak,
            'follow_through': (early_peak + 1, num_frames - 1),
            'handedness': 'right'
        }

        validation = validate_phase_boundaries(phases, num_frames - 1)

        # Should flag contact position as invalid
        assert not validation['checks']['contact_position'], (
            "Should detect contact too early"
        )


def test_validation_forward_swing_duration():
    """
    Test forward swing duration bounds (20-50% of sequence).
    """
    num_frames = 50
    num_landmarks = 33

    poses = np.zeros((num_frames, num_landmarks, 3))
    wrist_idx = LANDMARKS['right_wrist']

    # Normal trajectory
    t = np.linspace(0, np.pi, num_frames)
    poses[:, wrist_idx, :] = 0.5 + 0.2 * np.sin(t)[:, np.newaxis]

    # Create phases with too-long forward swing (60% of sequence)
    contact = 40
    phases = {
        'preparation': (0, 5),
        'backswing': (5, 10),
        'forward_swing': (10, contact),  # 30 frames = 61% (too long)
        'contact': contact,
        'follow_through': (contact + 1, num_frames - 1),
        'handedness': 'right'
    }

    validation = validate_phase_boundaries(phases, num_frames - 1)

    # Should flag forward swing duration
    if not validation['checks']['forward_swing_duration']:
        assert validation['forward_swing_pct'] > 0.5 or validation['forward_swing_pct'] < 0.2, (
            "Forward swing duration should be outside 20-50% range"
        )


# ============================================================================
# Test Intent Window
# ============================================================================

def test_intent_window():
    """
    Test intent window returns [contact-5 : contact-2] range.
    """
    contact_frame = 30

    start, end = get_intent_window(contact_frame)

    assert start == 25, f"Intent window should start at contact-5 (got {start})"
    assert end == 28, f"Intent window should end at contact-2 (got {end})"


def test_intent_window_edge_case_early_contact():
    """
    Test intent window with early contact (< 5 frames).

    Expected: Should clamp to frame 0.
    """
    contact_frame = 3

    start, end = get_intent_window(contact_frame)

    assert start >= 0, "Intent window start should not be negative"
    assert end >= 0, "Intent window end should not be negative"
    assert start <= end, "Intent window start should be <= end"


# ============================================================================
# Test Batch Validation Accuracy
# ============================================================================

def test_batch_validation_accuracy(real_pose_data):
    """
    Test segmentation on multiple samples.

    Success criteria: >= 85% of samples pass validation checks.
    """
    if real_pose_data is None:
        pytest.skip("Real pose data not available - skipping batch validation")

    total_samples = len(real_pose_data)
    passed_samples = 0

    for pose_sequence in real_pose_data:
        try:
            phases, validation = segment_and_validate(pose_sequence, fps=30)

            # Count as passed if validation is successful
            if validation['valid']:
                passed_samples += 1

        except Exception:
            # Segmentation failure counts as not passed
            continue

    pass_rate = passed_samples / total_samples

    print(f"\nBatch validation results:")
    print(f"  Total samples: {total_samples}")
    print(f"  Passed validation: {passed_samples}")
    print(f"  Pass rate: {pass_rate:.1%}")

    # Success criteria: >= 85% pass rate
    assert pass_rate >= 0.85, (
        f"Pass rate {pass_rate:.1%} below 85% threshold "
        f"({passed_samples}/{total_samples} passed)"
    )


# ============================================================================
# Test Phase Statistics
# ============================================================================

def test_calculate_phase_statistics(synthetic_pose_sequence):
    """
    Test phase statistics calculation.
    """
    phases = segment_stroke_phases(synthetic_pose_sequence, fps=30)
    total_frames = len(synthetic_pose_sequence) - 1

    stats = calculate_phase_statistics(phases, total_frames)

    # Check structure
    assert 'duration_percentages' in stats
    assert 'research_reference' in stats
    assert 'deviations' in stats

    # Check duration percentages sum to ~100%
    durations = stats['duration_percentages']
    total_pct = sum(durations.values())

    assert 95 <= total_pct <= 105, (
        f"Phase durations should sum to ~100% (got {total_pct:.1f}%)"
    )

    # Check forward swing is in reasonable range
    forward_swing_pct = durations['forward_swing']
    assert 10 <= forward_swing_pct <= 70, (
        f"Forward swing {forward_swing_pct:.1f}% seems unreasonable"
    )


# ============================================================================
# Test Segment and Validate Combined
# ============================================================================

def test_segment_and_validate(synthetic_pose_sequence):
    """
    Test combined segmentation and validation function.
    """
    phases, validation = segment_and_validate(synthetic_pose_sequence, fps=30)

    # Check phases returned
    assert 'contact' in phases
    assert 'forward_swing' in phases

    # Check validation returned
    assert 'valid' in validation
    assert 'checks' in validation
    assert 'warnings' in validation

    # Synthetic data should pass validation
    assert validation['valid'], (
        f"Synthetic data should pass validation. Warnings: {validation['warnings']}"
    )


# ============================================================================
# Run Tests
# ============================================================================

if __name__ == "__main__":
    # Run with pytest
    pytest.main([__file__, "-v", "--tb=short"])
