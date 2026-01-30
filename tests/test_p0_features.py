#!/usr/bin/env python3
"""
Tests for Priority 0 (P0) Features

Tests kinetic chain timing, contact frame analysis, intent window features,
and Smash Intent Score (SIS) formula.
"""

import pytest
import numpy as np
from src.data_processing.kinetic_chain_features import (
    extract_kinetic_chain_timing,
    calculate_coordination_efficiency,
    detect_dominant_side
)
from src.data_processing.contact_frame_features import (
    extract_contact_frame_features,
    extract_intent_window_features,
    calculate_smash_intent_score,
    calculate_angle_3d
)
from src.data_processing.phase_segmentation import segment_stroke_phases


# === FIXTURES ===

@pytest.fixture
def synthetic_pose_sequential():
    """
    Create synthetic pose sequence with sequential kinetic chain activation.
    Hip peaks at frame 11, trunk at 13, shoulder at 15, elbow at 17, wrist at 19.
    All peaks within forward swing phase [8, 20].
    """
    from src.data_processing.phase_segmentation import LANDMARKS

    num_frames = 40
    pose_sequence = np.zeros((num_frames, 33, 3))

    # Create velocity profiles with sequential peaks
    time = np.arange(num_frames)

    # Hip peaks at frame 11 (within forward swing)
    hip_profile = np.exp(-((time - 11) ** 2) / 10)
    pose_sequence[:, LANDMARKS['left_hip'], :] = np.column_stack([hip_profile, hip_profile, hip_profile])
    pose_sequence[:, LANDMARKS['right_hip'], :] = np.column_stack([hip_profile, hip_profile, hip_profile])

    # Trunk calculated as midpoint - need separate profiles that peak at frame 13
    # We'll set shoulders and hips such that their midpoint peaks at 13
    trunk_profile = np.exp(-((time - 13) ** 2) / 10)

    # Shoulders peak at frame 15
    shoulder_profile = np.exp(-((time - 15) ** 2) / 10)
    pose_sequence[:, LANDMARKS['left_shoulder'], :] = np.column_stack([shoulder_profile, shoulder_profile, shoulder_profile])
    pose_sequence[:, LANDMARKS['right_shoulder'], :] = np.column_stack([shoulder_profile, shoulder_profile, shoulder_profile])

    # Elbow peaks at frame 17 (right elbow for right-handed)
    elbow_profile = np.exp(-((time - 17) ** 2) / 10)
    pose_sequence[:, LANDMARKS['right_elbow'], 0] = elbow_profile
    pose_sequence[:, LANDMARKS['right_elbow'], 1] = elbow_profile * 0.8  # Lower Y = higher position
    pose_sequence[:, LANDMARKS['right_elbow'], 2] = elbow_profile

    # Wrist peaks at frame 19 (right wrist - dominant)
    wrist_profile = np.exp(-((time - 19) ** 2) / 10)
    pose_sequence[:, LANDMARKS['right_wrist'], 0] = wrist_profile
    pose_sequence[:, LANDMARKS['right_wrist'], 1] = wrist_profile * 0.7  # Even higher (lower Y)
    pose_sequence[:, LANDMARKS['right_wrist'], 2] = wrist_profile

    # Left wrist lower (higher Y value) to indicate right-handed
    # Use constant profile with higher Y across all frames
    pose_sequence[:, LANDMARKS['left_wrist'], 1] = 0.9

    return pose_sequence


@pytest.fixture
def synthetic_phases_forward_swing():
    """Phase boundaries with forward swing from frame 8-20"""
    return {
        'preparation': (0, 5),
        'backswing': (5, 8),
        'forward_swing': (8, 20),
        'contact': 20,
        'follow_through': (21, 39),
        'handedness': 'right'
    }


@pytest.fixture
def synthetic_contact_pose():
    """
    Create synthetic contact frame pose.
    Smash characteristics: forearm vertical angle < 90deg, wrist below elbow
    """
    from src.data_processing.phase_segmentation import LANDMARKS

    pose = np.zeros((33, 3))

    # Right shoulder at origin
    pose[LANDMARKS['right_shoulder']] = [0.5, 0.3, 0.5]

    # Right elbow forward and down
    pose[LANDMARKS['right_elbow']] = [0.55, 0.5, 0.55]

    # Right wrist below elbow (smash characteristic)
    pose[LANDMARKS['right_wrist']] = [0.6, 0.6, 0.6]  # Y=0.6 > elbow Y=0.5 (below elbow)

    # Left wrist higher Y (non-dominant)
    pose[LANDMARKS['left_wrist']] = [0.4, 0.8, 0.4]

    # Hips for body posture
    pose[LANDMARKS['left_hip']] = [0.45, 0.7, 0.5]
    pose[LANDMARKS['right_hip']] = [0.55, 0.7, 0.5]

    # Left shoulder
    pose[LANDMARKS['left_shoulder']] = [0.45, 0.3, 0.5]

    return pose


# === KINETIC CHAIN TESTS ===

def test_kinetic_chain_sequential_order(synthetic_pose_sequential, synthetic_phases_forward_swing):
    """Test kinetic chain timing detects sequential activation."""
    features = extract_kinetic_chain_timing(
        synthetic_pose_sequential,
        synthetic_phases_forward_swing
    )

    # All delays should be non-negative (sequential order)
    assert features['hip_to_trunk_delay'] >= 0, "Hip should peak before trunk"
    assert features['trunk_to_shoulder_delay'] >= 0, "Trunk should peak before shoulder"
    assert features['shoulder_to_elbow_delay'] >= 0, "Shoulder should peak before elbow"
    assert features['elbow_to_wrist_delay'] >= 0, "Elbow should peak before wrist"

    # Chain should be valid
    assert features['chain_sequence_valid'] == True

    # Coordination efficiency should be perfect (1.0)
    assert features['coordination_efficiency'] == 1.0


def test_kinetic_chain_forward_swing_only(synthetic_pose_sequential, synthetic_phases_forward_swing):
    """Test timing measured only within forward swing phase."""
    features = extract_kinetic_chain_timing(
        synthetic_pose_sequential,
        synthetic_phases_forward_swing
    )

    # All peak times should be within forward swing phase [8, 20]
    forward_start = synthetic_phases_forward_swing['forward_swing'][0]
    forward_end = synthetic_phases_forward_swing['forward_swing'][1]

    # Since we can't directly access peak times, verify total chain time is reasonable
    # For forward swing of 12 frames, hip-to-wrist should be < 12
    assert features['hip_to_wrist_total'] <= (forward_end - forward_start), \
        "Total chain time exceeds forward swing duration"

    # Hip to wrist should be positive (wrist peaks after hip)
    assert features['hip_to_wrist_total'] >= 0


def test_coordination_efficiency():
    """Test coordination efficiency calculation."""
    # Perfect sequence
    perfect_timing = {
        'hip_to_trunk_delay': 2,
        'trunk_to_shoulder_delay': 2,
        'shoulder_to_elbow_delay': 2,
        'elbow_to_wrist_delay': 2,
        'chain_sequence_valid': True
    }
    efficiency = calculate_coordination_efficiency(perfect_timing)
    assert efficiency == 1.0, "Perfect sequence should have 1.0 efficiency"

    # Partial violation
    partial_violation = {
        'hip_to_trunk_delay': 2,
        'trunk_to_shoulder_delay': -1,  # One violation
        'shoulder_to_elbow_delay': 2,
        'elbow_to_wrist_delay': 2,
        'chain_sequence_valid': False
    }
    efficiency = calculate_coordination_efficiency(partial_violation)
    assert 0.5 < efficiency < 1.0, "Partial violation should have 0.5-1.0 efficiency"

    # Severe violation
    severe_violation = {
        'hip_to_trunk_delay': -10,  # Large negative
        'trunk_to_shoulder_delay': -5,
        'shoulder_to_elbow_delay': 2,
        'elbow_to_wrist_delay': 2,
        'chain_sequence_valid': False
    }
    efficiency = calculate_coordination_efficiency(severe_violation)
    assert efficiency < 0.5, "Severe violation should have <0.5 efficiency"


def test_detect_dominant_side():
    """Test dominant side detection."""
    from src.data_processing.phase_segmentation import LANDMARKS

    # Create pose with right wrist higher (lower Y in MediaPipe coordinates)
    pose = np.zeros((40, 33, 3))
    # Use correct LANDMARKS indices: right_wrist=16, left_wrist=15
    pose[:, LANDMARKS['right_wrist'], 1] = 0.3  # Right wrist high (lower Y value)
    pose[:, LANDMARKS['left_wrist'], 1] = 0.7   # Left wrist low (higher Y value)

    handedness = detect_dominant_side(pose)
    # r_avg_height (0.3) < l_avg_height (0.7) -> right hand higher -> right-handed
    assert handedness == 'right', "Right wrist higher (lower Y) should indicate right-handed"

    # Swap heights
    pose[:, LANDMARKS['right_wrist'], 1] = 0.7  # Right wrist low (higher Y value)
    pose[:, LANDMARKS['left_wrist'], 1] = 0.3   # Left wrist high (lower Y value)

    handedness = detect_dominant_side(pose)
    # r_avg_height (0.7) > l_avg_height (0.3) -> left hand higher -> left-handed
    assert handedness == 'left', "Left wrist higher (lower Y) should indicate left-handed"


# === CONTACT FRAME TESTS ===

def test_contact_frame_features_extraction():
    """Test contact frame features extraction."""
    from src.data_processing.phase_segmentation import LANDMARKS

    # Create full sequence with contact pose
    pose_sequence = np.zeros((30, 33, 3))

    # Right-handed setup
    pose_sequence[:, LANDMARKS['right_wrist'], 1] = 0.3  # Right wrist high
    pose_sequence[:, LANDMARKS['left_wrist'], 1] = 0.7   # Left wrist low

    # Set contact frame (frame 15)
    contact_frame = 15

    # Right shoulder, elbow, wrist
    pose_sequence[contact_frame, LANDMARKS['right_shoulder']] = [0.5, 0.3, 0.5]  # Shoulder
    pose_sequence[contact_frame, LANDMARKS['right_elbow']] = [0.55, 0.5, 0.55]  # Elbow
    pose_sequence[contact_frame, LANDMARKS['right_wrist']] = [0.6, 0.6, 0.6]  # Wrist below elbow

    # Left shoulder and hips for posture
    pose_sequence[contact_frame, LANDMARKS['left_shoulder']] = [0.45, 0.3, 0.5]
    pose_sequence[contact_frame, LANDMARKS['left_hip']] = [0.45, 0.7, 0.5]
    pose_sequence[contact_frame, LANDMARKS['right_hip']] = [0.55, 0.7, 0.5]

    features = extract_contact_frame_features(pose_sequence, contact_frame)

    # Check all expected features are present
    expected_features = [
        'contact_forearm_vertical_angle',
        'contact_forearm_pitch',
        'contact_wrist_elbow_vertical',
        'contact_wrist_height',
        'contact_elbow_extension',
        'contact_arm_extension',
        'contact_torso_lean',
        'contact_shoulder_rotation'
    ]

    for feature in expected_features:
        assert feature in features, f"Missing feature: {feature}"

    # Verify wrist is below elbow (positive vertical difference)
    assert features['contact_wrist_elbow_vertical'] > 0, "Wrist should be below elbow for smash setup"


def test_contact_frame_edge_case_missing_landmark():
    """Test contact frame handles missing/zero landmarks gracefully."""
    # Create sequence with zeros (missing landmarks)
    pose_sequence = np.zeros((20, 33, 3))
    contact_frame = 10

    # Extract features - should not raise exception
    features = extract_contact_frame_features(pose_sequence, contact_frame)

    # Features should contain NaN for angle calculations
    assert 'contact_forearm_vertical_angle' in features
    # With zero vectors, angle will be NaN
    assert np.isnan(features['contact_forearm_vertical_angle']) or features['contact_forearm_vertical_angle'] >= 0


# === INTENT WINDOW TESTS ===

def test_intent_window_features():
    """Test intent window features extraction."""
    from src.data_processing.phase_segmentation import LANDMARKS

    # Create pose sequence
    pose_sequence = np.zeros((30, 33, 3))

    # Right-handed
    pose_sequence[:, LANDMARKS['right_wrist'], 1] = 0.3  # Right wrist high
    pose_sequence[:, LANDMARKS['left_wrist'], 1] = 0.7   # Left wrist low

    contact_frame = 15

    # Set up landmarks for intent window [10:13] (contact-5:contact-2)
    for i in range(10, 14):
        # Right shoulder, elbow, wrist
        pose_sequence[i, LANDMARKS['right_shoulder']] = [0.5, 0.3, 0.5]
        pose_sequence[i, LANDMARKS['right_elbow']] = [0.55 + i*0.01, 0.5, 0.55]  # Elbow moving forward
        pose_sequence[i, LANDMARKS['right_wrist']] = [0.6, 0.6, 0.6]

        # Non-racket arm (left wrist) dropping
        pose_sequence[i, LANDMARKS['left_wrist']] = [0.4, 0.7 + i*0.02, 0.4]  # Dropping down

        # Shoulders and hips for rotation
        pose_sequence[i, LANDMARKS['left_shoulder']] = [0.45, 0.3, 0.5]
        pose_sequence[i, LANDMARKS['left_hip']] = [0.45, 0.7, 0.5]
        pose_sequence[i, LANDMARKS['right_hip']] = [0.55, 0.7, 0.5]

    features = extract_intent_window_features(pose_sequence, contact_frame)

    # Check all SIS component features are present
    expected_features = [
        'intent_elbow_forward_mean',
        'intent_elbow_forward_max',
        'intent_forearm_rotation_vel',
        'intent_nonracket_arm_drop',
        'intent_torso_rotation_vel',
        'intent_com_velocity_y'
    ]

    for feature in expected_features:
        assert feature in features, f"Missing intent window feature: {feature}"


def test_intent_window_edge_case_contact_near_start():
    """Test intent window when contact frame < 5."""
    from src.data_processing.phase_segmentation import LANDMARKS

    pose_sequence = np.zeros((10, 33, 3))

    # Right-handed
    pose_sequence[:, LANDMARKS['right_wrist'], 1] = 0.3
    pose_sequence[:, LANDMARKS['left_wrist'], 1] = 0.7

    contact_frame = 2  # Only frames 0-2 available

    # Set minimal valid pose data
    pose_sequence[:, LANDMARKS['right_shoulder']] = [0.5, 0.3, 0.5]
    pose_sequence[:, LANDMARKS['right_elbow']] = [0.55, 0.5, 0.55]
    pose_sequence[:, LANDMARKS['right_wrist']] = [0.6, 0.6, 0.6]
    pose_sequence[:, LANDMARKS['left_shoulder']] = [0.45, 0.3, 0.5]
    pose_sequence[:, LANDMARKS['left_hip']] = [0.45, 0.7, 0.5]
    pose_sequence[:, LANDMARKS['right_hip']] = [0.55, 0.7, 0.5]

    # Should not raise exception
    features = extract_intent_window_features(pose_sequence, contact_frame)

    # Features should exist (may be zero or minimal)
    assert 'intent_elbow_forward_mean' in features


# === SMASH INTENT SCORE TESTS ===

def test_smash_intent_score_formula():
    """Test SIS calculation with known inputs."""
    # Create features that should result in high SIS (smash-like)
    smash_features = {
        'intent_elbow_forward_max': 0.20,      # High elbow lead
        'intent_forearm_rotation_vel': 20.0,   # High pronation
        'intent_nonracket_arm_drop': 0.15,     # Strong arm pull
        'intent_torso_rotation_vel': 3.0,      # High rotation
        'intent_com_velocity_y': 0.10          # Downward COM
    }

    result = calculate_smash_intent_score(smash_features)

    # Check structure
    assert 'smash_intent_score' in result
    assert 'classification_hint' in result
    assert 'component_scores' in result

    # Check SIS is high (should be close to 1.0)
    assert result['smash_intent_score'] >= 0.65, "High values should give smash classification"
    assert result['classification_hint'] == 'smash'


def test_smash_intent_score_thresholds():
    """Test SIS thresholds for smash/deceptive/clear."""
    # Clear-like features (low values)
    clear_features = {
        'intent_elbow_forward_max': 0.05,
        'intent_forearm_rotation_vel': 5.0,
        'intent_nonracket_arm_drop': 0.02,
        'intent_torso_rotation_vel': 0.5,
        'intent_com_velocity_y': 0.01
    }

    result = calculate_smash_intent_score(clear_features)
    assert result['smash_intent_score'] < 0.65, "Low values should not be smash"
    assert result['classification_hint'] in ['clear', 'deceptive']

    # Mid-range (deceptive)
    deceptive_features = {
        'intent_elbow_forward_max': 0.12,
        'intent_forearm_rotation_vel': 12.0,
        'intent_nonracket_arm_drop': 0.08,
        'intent_torso_rotation_vel': 1.5,
        'intent_com_velocity_y': 0.05
    }

    result = calculate_smash_intent_score(deceptive_features)
    # Should be in deceptive range or boundary
    assert result['smash_intent_score'] >= 0.0 and result['smash_intent_score'] <= 1.0


def test_smash_intent_score_normalization():
    """Test SIS normalization clips extreme values to 0-1."""
    # Extreme high values (should clip to 1.0)
    extreme_features = {
        'intent_elbow_forward_max': 1.0,       # Way above max
        'intent_forearm_rotation_vel': 100.0,  # Way above max
        'intent_nonracket_arm_drop': 1.0,
        'intent_torso_rotation_vel': 100.0,
        'intent_com_velocity_y': 1.0
    }

    result = calculate_smash_intent_score(extreme_features)

    # SIS should be clipped to 1.0 max
    assert result['smash_intent_score'] <= 1.0, "SIS should not exceed 1.0"
    assert result['smash_intent_score'] >= 0.0, "SIS should not be negative"

    # All component scores should be 0-1
    for component, score in result['component_scores'].items():
        assert 0.0 <= score <= 1.0, f"Component {component} score {score} not in [0,1]"


# === INTEGRATION TEST ===

def test_p0_features_integration(synthetic_pose_sequential, synthetic_phases_forward_swing):
    """Integration test: extract all P0 features from single pose sequence."""
    # Get contact frame from phases
    contact_frame = synthetic_phases_forward_swing['contact']

    # Extract kinetic chain features
    kinetic_features = extract_kinetic_chain_timing(
        synthetic_pose_sequential,
        synthetic_phases_forward_swing
    )

    # Extract contact frame features
    contact_features = extract_contact_frame_features(
        synthetic_pose_sequential,
        contact_frame
    )

    # Extract intent window features
    intent_features = extract_intent_window_features(
        synthetic_pose_sequential,
        contact_frame
    )

    # Calculate SIS
    sis_result = calculate_smash_intent_score(intent_features)

    # Verify all feature categories present
    assert len(kinetic_features) > 0, "Kinetic features missing"
    assert len(contact_features) > 0, "Contact features missing"
    assert len(intent_features) > 0, "Intent features missing"
    assert 'smash_intent_score' in sis_result, "SIS missing"

    # Verify kinetic chain valid for sequential synthetic data
    assert kinetic_features['chain_sequence_valid'] == True


# === HELPER FUNCTION TEST ===

def test_calculate_angle_3d():
    """Test 3D angle calculation."""
    # 90 degree angle
    p1 = np.array([1, 0, 0])
    p2 = np.array([0, 0, 0])
    p3 = np.array([0, 1, 0])

    angle = calculate_angle_3d(p1, p2, p3)
    assert abs(angle - 90.0) < 0.1, "Should calculate 90 degree angle"

    # 180 degree angle (straight line)
    p1 = np.array([1, 0, 0])
    p2 = np.array([0, 0, 0])
    p3 = np.array([-1, 0, 0])

    angle = calculate_angle_3d(p1, p2, p3)
    assert abs(angle - 180.0) < 0.1, "Should calculate 180 degree angle"

    # Edge case: zero vector
    p1 = np.array([0, 0, 0])
    p2 = np.array([0, 0, 0])
    p3 = np.array([1, 1, 1])

    angle = calculate_angle_3d(p1, p2, p3)
    assert np.isnan(angle), "Zero vector should return NaN"


if __name__ == '__main__':
    pytest.main([__file__, '-v', '--tb=short'])
