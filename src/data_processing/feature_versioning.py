#!/usr/bin/env python3
"""
Feature Versioning Compatibility Layer

Provides version-gated feature extraction to maintain backward compatibility
between v2 and v3 feature engineering pipelines.

Version differences:
- V2: 427 statistical features, no phase segmentation, no SIS
- V3: ~200-254 selected features (after feature selection), includes phase
      segmentation, P0 features (kinetic chain, contact, intent, SIS), and
      P1 features (angular velocity, phase-specific)

Usage:
    # Extract v2 features (backward compatible)
    fe = FeatureEngineering('v2')
    features_v2 = fe.extract_features(pose_sequence)

    # Extract v3 features (new coach-informed features)
    fe = FeatureEngineering('v3')
    features_v3 = fe.extract_features(pose_sequence, apply_selection=True)

    # Detect version from feature dict
    version = get_feature_version(features)

References:
    - Feature engineering v2: feature_engineering_v2.py
    - Feature engineering v3: feature_engineering_v3.py
"""

import numpy as np
from pathlib import Path
import json


def get_feature_version(features_dict):
    """
    Detect version from feature dictionary.

    Detection logic:
    1. If 'feature_version' key exists, use it
    2. If 'smash_intent_score' exists, it's v3 (SIS is v3-only)
    3. Otherwise assume v2 (default)

    Args:
        features_dict: Feature dictionary

    Returns:
        str: 'v2' or 'v3'
    """
    if 'feature_version' in features_dict:
        return features_dict['feature_version']
    elif 'smash_intent_score' in features_dict:
        return 'v3'  # SIS is v3-only
    else:
        return 'v2'  # Default to v2


def get_expected_feature_count(version):
    """
    Get expected feature count for version.

    Args:
        version: 'v2' or 'v3'

    Returns:
        int or tuple: Expected feature count (exact or range)
    """
    if version == 'v2':
        # V2 has fixed count (~427 features)
        # Exact count depends on temporal features
        return 427  # Approximate
    elif version == 'v3':
        # V3 count depends on feature selection
        # Load from manifest if available
        selection_path = Path(__file__).resolve().parents[2] / \
                        "data" / "processed" / "features_v3" / "selected_features.json"

        if selection_path.exists():
            with open(selection_path, 'r') as f:
                manifest = json.load(f)
            selected = manifest.get('selected_features', [])
            if len(selected) > 0:
                # Add metadata fields not in selection
                return len(selected) + 5  # Approximate (metadata fields)
            else:
                # Manifest placeholder - return all features count
                return 360  # Approximate (v2: ~315 + P0: ~20 + P1: ~32)
        else:
            # No manifest - return all features count
            return 360  # Approximate
    else:
        raise ValueError(f"Unknown version: {version}")


def validate_feature_set(features_dict, expected_version):
    """
    Validate features match expected version.

    Checks:
    1. Version detection matches expected
    2. Feature count within acceptable range

    Args:
        features_dict: Feature dictionary
        expected_version: 'v2' or 'v3'

    Returns:
        bool: True if valid

    Raises:
        ValueError: If validation fails
    """
    actual_version = get_feature_version(features_dict)

    if actual_version != expected_version:
        raise ValueError(
            f"Feature version mismatch: expected {expected_version}, "
            f"got {actual_version}"
        )

    expected_count = get_expected_feature_count(expected_version)
    actual_count = len(features_dict)

    # Allow tolerance (metadata fields, temporal features vary slightly)
    if isinstance(expected_count, int):
        tolerance = max(10, int(expected_count * 0.1))  # 10% tolerance or 10 features
        if abs(actual_count - expected_count) > tolerance:
            raise ValueError(
                f"Feature count mismatch for {expected_version}: "
                f"expected ~{expected_count}, got {actual_count}"
            )
    # For tuple (range), check within range
    elif isinstance(expected_count, tuple):
        min_count, max_count = expected_count
        if not (min_count <= actual_count <= max_count):
            raise ValueError(
                f"Feature count out of range for {expected_version}: "
                f"expected {min_count}-{max_count}, got {actual_count}"
            )

    return True


class FeatureEngineering:
    """
    Feature engineering with version compatibility.

    Supported versions:
    - 'v2': 427 statistical features, no phase segmentation
    - 'v3': ~200-254 selected features, includes P0+P1

    Usage:
        fe = FeatureEngineering('v3')
        features = fe.extract_features(pose_sequence)
    """

    SUPPORTED_VERSIONS = ['v2', 'v3']

    def __init__(self, version='v3'):
        """
        Initialize feature engineering with specified version.

        Args:
            version: 'v2' or 'v3' (default 'v3')

        Raises:
            ValueError: If version not supported
        """
        if version not in self.SUPPORTED_VERSIONS:
            raise ValueError(
                f"Version {version} not supported. "
                f"Use: {self.SUPPORTED_VERSIONS}"
            )
        self.version = version

    def extract_features(self, pose_sequence, **kwargs):
        """
        Extract features based on version.

        Args:
            pose_sequence: Array of shape (num_frames, 33, 3) - MediaPipe pose sequence
            **kwargs: Version-specific arguments (e.g., apply_selection for v3)

        Returns:
            dict with extracted features
        """
        if self.version == 'v2':
            return self._extract_v2(pose_sequence)
        elif self.version == 'v3':
            return self._extract_v3(pose_sequence, **kwargs)

    def _extract_v2(self, pose_sequence):
        """
        V2 extraction (427 features) - unchanged.

        Uses original v2 pipeline:
        - Per-frame spatial features
        - Temporal features (velocity, acceleration)
        - Statistical summary (mean, std, min, max, range, percentiles)

        Args:
            pose_sequence: Array of shape (num_frames, 33, 3)

        Returns:
            dict with v2 features
        """
        from src.data_processing.feature_engineering_v2 import (
            extract_temporal_features,
            extract_statistical_summary
        )

        # Extract temporal features
        result = extract_temporal_features(pose_sequence)

        if result[0] is None:
            return None

        per_frame, names, temporal = result

        # Extract statistical summary
        summary = extract_statistical_summary(per_frame, names)

        # Combine statistical summary and temporal features
        summary.update(temporal)

        return summary

    def _extract_v3(self, pose_sequence, apply_selection=True):
        """
        V3 extraction (< 254 selected features).

        Uses v3 pipeline:
        - Phase segmentation (5 phases)
        - V2 base features
        - P0 features (kinetic chain, contact, intent, SIS)
        - P1 features (angular velocity, phase-specific)
        - Feature selection applied (if enabled)

        Args:
            pose_sequence: Array of shape (num_frames, 33, 3)
            apply_selection: bool (default True) - Apply feature selection mask

        Returns:
            dict with v3 features
        """
        from src.data_processing.feature_engineering_v3 import extract_features_v3

        return extract_features_v3(pose_sequence, apply_selection=apply_selection)

    def get_expected_features(self):
        """
        Return expected feature count for version.

        Returns:
            int: Expected number of features
        """
        return get_expected_feature_count(self.version)

    def get_version(self):
        """
        Return current version.

        Returns:
            str: 'v2' or 'v3'
        """
        return self.version


def upgrade_v2_to_v3(features_v2, pose_sequence, apply_selection=True):
    """
    Upgrade v2 features to v3.

    Useful for migration scenarios where v2 features exist and you want
    to augment with v3 features.

    Strategy:
    1. Extract full v3 features from pose_sequence
    2. Combine v2 and v3 (v3 takes precedence for overlapping keys)
    3. Apply feature selection if enabled

    Args:
        features_v2: dict with v2 features
        pose_sequence: Array of shape (num_frames, 33, 3) - MediaPipe pose sequence
        apply_selection: bool (default True) - Apply feature selection mask

    Returns:
        dict with v3 features (augmented from v2)
    """
    from src.data_processing.feature_engineering_v3 import extract_features_v3

    # Extract v3 features
    features_v3 = extract_features_v3(pose_sequence, apply_selection=apply_selection)

    if features_v3 is None:
        # Fall back to v2 features if v3 extraction fails
        return features_v2

    # V3 includes v2 base features, so just return v3
    # (v3 already combines v2 base + P0 + P1)
    return features_v3


def compare_versions(pose_sequence):
    """
    Compare v2 and v3 feature extraction on same pose sequence.

    Useful for debugging and validation.

    Args:
        pose_sequence: Array of shape (num_frames, 33, 3) - MediaPipe pose sequence

    Returns:
        dict with:
            - v2_features: dict
            - v3_features: dict
            - v2_count: int
            - v3_count: int
            - overlap: int (number of common keys)
            - v3_only: list (keys in v3 but not v2)
            - v2_only: list (keys in v2 but not v3)
    """
    fe_v2 = FeatureEngineering('v2')
    fe_v3 = FeatureEngineering('v3')

    v2_features = fe_v2.extract_features(pose_sequence)
    v3_features = fe_v3.extract_features(pose_sequence, apply_selection=False)

    if v2_features is None or v3_features is None:
        return None

    v2_keys = set(v2_features.keys())
    v3_keys = set(v3_features.keys())

    overlap = len(v2_keys & v3_keys)
    v3_only = list(v3_keys - v2_keys)
    v2_only = list(v2_keys - v3_keys)

    return {
        'v2_features': v2_features,
        'v3_features': v3_features,
        'v2_count': len(v2_features),
        'v3_count': len(v3_features),
        'overlap': overlap,
        'v3_only': v3_only,
        'v2_only': v2_only
    }
