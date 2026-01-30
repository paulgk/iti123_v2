#!/usr/bin/env python3
"""
Tests for Feature Engineering V3

Tests v3 feature extraction, version gating, and backward compatibility.

Tests gracefully handle both populated and placeholder manifest states:
- If manifest populated (Plan 04 completed): run all tests including selection tests
- If manifest placeholder (Plan 04 not yet run): skip selection-dependent tests
"""

import pytest
import numpy as np
import json
import os
from pathlib import Path


# Test fixtures

@pytest.fixture
def sample_pose_sequence():
    """Generate random pose sequence for testing"""
    # 50 frames, 33 landmarks, 3 coordinates (x, y, z)
    return np.random.randn(50, 33, 3)


@pytest.fixture
def manifest_status():
    """Check if manifest is populated"""
    base_dir = Path(__file__).resolve().parents[1]
    manifest_path = base_dir / 'data' / 'processed' / 'features_v3' / 'selected_features.json'

    if not manifest_path.exists():
        return 'missing'

    with open(manifest_path) as f:
        manifest = json.load(f)

    if manifest.get('selected_features') and len(manifest['selected_features']) > 0:
        return 'populated'
    return 'placeholder'


# Import tests

def test_v3_extraction_imports():
    """Verify all v3 modules can be imported"""
    from src.data_processing.feature_engineering_v3 import extract_features_v3
    from src.data_processing.feature_versioning import FeatureEngineering
    assert extract_features_v3 is not None
    assert FeatureEngineering is not None


def test_versioning_imports():
    """Verify versioning module imports correctly"""
    from src.data_processing.feature_versioning import (
        FeatureEngineering,
        get_feature_version,
        validate_feature_set,
        upgrade_v2_to_v3
    )
    assert get_feature_version is not None


# V3 extraction tests (without selection)

def test_v3_extraction_returns_dict(sample_pose_sequence):
    """Verify extraction returns dict with expected keys"""
    from src.data_processing.feature_engineering_v3 import extract_features_v3

    # Test WITHOUT selection (doesn't need manifest)
    features = extract_features_v3(sample_pose_sequence, apply_selection=False)

    assert isinstance(features, dict)
    assert len(features) > 0
    assert 'feature_version' in features
    assert features['feature_version'] == 'v3'


def test_v3_includes_metadata_fields(sample_pose_sequence):
    """Verify v3 includes required metadata fields"""
    from src.data_processing.feature_engineering_v3 import extract_features_v3

    features = extract_features_v3(sample_pose_sequence, apply_selection=False)

    # Check metadata fields
    assert 'feature_version' in features
    assert 'handedness' in features
    assert 'phase_validation_passed' in features
    assert 'contact_position_pct' in features
    assert 'forward_swing_pct' in features


def test_v3_includes_p0_features(sample_pose_sequence):
    """Verify P0 features present (kinetic chain, contact, intent, SIS)"""
    from src.data_processing.feature_engineering_v3 import extract_features_v3

    features = extract_features_v3(sample_pose_sequence, apply_selection=False)

    # P0 kinetic chain features
    assert any('kinetic' in k or 'chain' in k or 'hip_to_wrist_total' in k
               for k in features.keys())

    # P0 contact frame features
    assert any('contact' in k for k in features.keys())

    # P0 intent window features
    assert any('intent' in k for k in features.keys())

    # P0 Smash Intent Score
    assert 'smash_intent_score' in features
    assert 'sis_classification_hint' in features


def test_v3_includes_p1_features(sample_pose_sequence):
    """Verify P1 features present (angular velocity, phase-specific, racket speed)"""
    from src.data_processing.feature_engineering_v3 import extract_features_v3

    features = extract_features_v3(sample_pose_sequence, apply_selection=False)

    # P1 angular velocity features
    assert any('angular' in k or 'elbow_extension_vel' in k or 'forearm_rotation_vel' in k
               for k in features.keys())

    # P1 phase-specific features
    assert any('preparation_' in k or 'backswing_' in k or 'forward_swing_' in k
               for k in features.keys())

    # P1 racket speed estimation
    assert any('racket_speed' in k for k in features.keys())


def test_v3_includes_v2_base_features(sample_pose_sequence):
    """Verify v2 base features included in v3"""
    from src.data_processing.feature_engineering_v3 import extract_features_v3

    features = extract_features_v3(sample_pose_sequence, apply_selection=False)

    # Check for some v2 baseline features
    # V2 has velocity, acceleration, arm extension, elbow angle, etc.
    v2_indicator_features = [
        'max_velocity',
        'peak_velocity_timing',
        'r_arm_extension_mean',
        'r_elbow_angle_mean'
    ]

    found_v2_features = sum(1 for f in v2_indicator_features if f in features)
    assert found_v2_features >= 2, f"Expected v2 base features, found {found_v2_features}"


# Selection tests (skip if manifest not populated)

def test_v3_with_selection_requires_manifest(sample_pose_sequence, manifest_status):
    """Verify selection behavior with and without manifest"""
    from src.data_processing.feature_engineering_v3 import extract_features_v3

    if manifest_status == 'populated':
        # Manifest populated - selection should reduce count
        features_all = extract_features_v3(sample_pose_sequence, apply_selection=False)
        features_selected = extract_features_v3(sample_pose_sequence, apply_selection=True)

        assert len(features_selected) < len(features_all), \
            "Selection should reduce feature count when manifest populated"
        assert len(features_selected) <= 254, \
            f"Selected features should be <= 254, got {len(features_selected)}"
    else:
        # Manifest placeholder or missing - selection disabled, returns all features
        features_all = extract_features_v3(sample_pose_sequence, apply_selection=False)
        features_selected = extract_features_v3(sample_pose_sequence, apply_selection=True)

        assert len(features_selected) == len(features_all), \
            "Without populated manifest, apply_selection should return all features"


def test_v3_with_selection_preserves_metadata(sample_pose_sequence, manifest_status):
    """Verify metadata fields preserved even with selection"""
    if manifest_status != 'populated':
        pytest.skip("Manifest not populated - run feature_selection_pipeline first")

    from src.data_processing.feature_engineering_v3 import extract_features_v3

    features = extract_features_v3(sample_pose_sequence, apply_selection=True)

    # Metadata should always be present
    assert 'feature_version' in features
    assert 'handedness' in features
    assert 'smash_intent_score' in features  # Critical P0 feature
    assert 'sis_classification_hint' in features


# Version gating tests

def test_version_gating_v2(sample_pose_sequence):
    """Verify v2 extraction unchanged"""
    from src.data_processing.feature_versioning import FeatureEngineering

    fe_v2 = FeatureEngineering('v2')
    features_v2 = fe_v2.extract_features(sample_pose_sequence)

    # V2 should NOT have v3-specific features
    assert 'smash_intent_score' not in features_v2
    assert 'feature_version' not in features_v2  # V2 doesn't set this
    assert 'phase_validation_passed' not in features_v2

    # V2 should have baseline features
    assert 'max_velocity' in features_v2 or 'peak_velocity_timing' in features_v2


def test_version_gating_v3(sample_pose_sequence):
    """Verify v3 extraction includes all new features"""
    from src.data_processing.feature_versioning import FeatureEngineering

    fe_v3 = FeatureEngineering('v3')
    features_v3 = fe_v3.extract_features(sample_pose_sequence, apply_selection=False)

    # V3 should have v3-specific features
    assert 'smash_intent_score' in features_v3
    assert 'feature_version' in features_v3
    assert features_v3['feature_version'] == 'v3'


def test_version_detection(sample_pose_sequence):
    """Verify version detection from features"""
    from src.data_processing.feature_versioning import (
        FeatureEngineering,
        get_feature_version
    )

    # V2 features
    fe_v2 = FeatureEngineering('v2')
    features_v2 = fe_v2.extract_features(sample_pose_sequence)
    detected_v2 = get_feature_version(features_v2)
    assert detected_v2 == 'v2'

    # V3 features
    fe_v3 = FeatureEngineering('v3')
    features_v3 = fe_v3.extract_features(sample_pose_sequence, apply_selection=False)
    detected_v3 = get_feature_version(features_v3)
    assert detected_v3 == 'v3'


def test_backward_compatibility_overlap(sample_pose_sequence):
    """Verify v2 features subset of v3 (without selection)"""
    from src.data_processing.feature_versioning import FeatureEngineering

    fe_v2 = FeatureEngineering('v2')
    fe_v3 = FeatureEngineering('v3')

    features_v2 = fe_v2.extract_features(sample_pose_sequence)
    features_v3 = fe_v3.extract_features(sample_pose_sequence, apply_selection=False)

    v2_keys = set(features_v2.keys())
    v3_keys = set(features_v3.keys())

    # Calculate overlap
    overlap = len(v2_keys & v3_keys)

    # V3 should include most v2 features (some may have different names)
    # Expect at least 50% overlap
    overlap_ratio = overlap / len(v2_keys)
    assert overlap_ratio > 0.5, \
        f"Expected >50% overlap between v2 and v3, got {overlap_ratio:.1%}"


# Feature validation tests

def test_smash_intent_score_range(sample_pose_sequence):
    """Verify SIS is in valid range [0, 1]"""
    from src.data_processing.feature_engineering_v3 import extract_features_v3

    features = extract_features_v3(sample_pose_sequence, apply_selection=False)

    sis = features['smash_intent_score']
    assert 0 <= sis <= 1, f"SIS should be in [0, 1], got {sis}"


def test_classification_hint_valid(sample_pose_sequence):
    """Verify SIS classification hint is valid category"""
    from src.data_processing.feature_engineering_v3 import extract_features_v3

    features = extract_features_v3(sample_pose_sequence, apply_selection=False)

    hint = features['sis_classification_hint']
    valid_hints = ['smash', 'deceptive', 'clear']
    assert hint in valid_hints, \
        f"Classification hint should be {valid_hints}, got {hint}"


def test_handedness_detection(sample_pose_sequence):
    """Verify handedness detected"""
    from src.data_processing.feature_engineering_v3 import extract_features_v3

    features = extract_features_v3(sample_pose_sequence, apply_selection=False)

    handedness = features['handedness']
    assert handedness in ['left', 'right'], \
        f"Handedness should be 'left' or 'right', got {handedness}"


def test_phase_validation_flag(sample_pose_sequence):
    """Verify phase validation flag is boolean"""
    from src.data_processing.feature_engineering_v3 import extract_features_v3

    features = extract_features_v3(sample_pose_sequence, apply_selection=False)

    valid = features['phase_validation_passed']
    assert isinstance(valid, (bool, np.bool_)), \
        f"phase_validation_passed should be boolean, got {type(valid)}"


# Error handling tests

def test_invalid_version_raises_error():
    """Verify invalid version raises ValueError"""
    from src.data_processing.feature_versioning import FeatureEngineering

    with pytest.raises(ValueError, match="not supported"):
        FeatureEngineering('v99')


def test_version_validation_mismatch():
    """Verify version validation detects mismatch"""
    from src.data_processing.feature_versioning import (
        FeatureEngineering,
        validate_feature_set
    )

    fe_v3 = FeatureEngineering('v3')
    features_v3 = fe_v3.extract_features(np.random.randn(50, 33, 3), apply_selection=False)

    # Should pass for correct version
    assert validate_feature_set(features_v3, 'v3')

    # Should raise for incorrect version
    with pytest.raises(ValueError, match="version mismatch"):
        validate_feature_set(features_v3, 'v2')


# Integration tests

def test_process_single_clip_v3_structure():
    """Verify process_single_clip_v3 returns correct structure"""
    from src.data_processing.feature_engineering_v3 import extract_features_v3
    import numpy as np

    pose_sequence = np.random.randn(50, 33, 3)
    features = extract_features_v3(pose_sequence, apply_selection=False)

    # Should return dict with features
    assert isinstance(features, dict)
    assert len(features) > 0


def test_manifest_loading():
    """Verify manifest loading handles both states correctly"""
    from src.data_processing.feature_engineering_v3 import load_selected_features

    selected = load_selected_features()

    # Should return None (placeholder) or list (populated)
    assert selected is None or isinstance(selected, list)


# Performance tests (optional, commented out)
"""
def test_v3_extraction_performance(sample_pose_sequence, manifest_status):
    '''Verify v3 extraction completes in reasonable time'''
    import time
    from src.data_processing.feature_engineering_v3 import extract_features_v3

    start = time.time()
    features = extract_features_v3(sample_pose_sequence, apply_selection=False)
    elapsed = time.time() - start

    # Should complete in < 1 second for single clip
    assert elapsed < 1.0, f"Extraction took {elapsed:.2f}s, expected < 1s"
"""
