#!/usr/bin/env python3
"""
Feature Engineering V3 - Coach-Informed Biomechanical Features

Improvements over V2:
- Phase segmentation (5 phases: preparation, backswing, forward_swing, contact, follow_through)
- P0 features: kinetic chain timing, contact frame analysis, intent window, Smash Intent Score
- P1 features: angular velocity, phase-specific statistics, racket head speed estimation
- Feature selection applied (< 254 features via selection mask)

Maintains backward compatibility with V2 via feature_versioning.py

Integration of modules:
- phase_segmentation: 5-phase stroke detection
- kinetic_chain_features: Sequential timing through kinetic chain (P0)
- contact_frame_features: Contact and intent window features, SIS calculation (P0)
- angular_velocity_features: Elbow extension and forearm rotation velocities (P1)
- phase_specific_features: Per-phase statistics and deceleration (P1)
- feature_selection: Feature selection mask loading
- feature_engineering_v2: Baseline spatial and temporal features

References:
    - Coach-informed feature engineering research (02-RESEARCH.md)
    - Feature selection pipeline (02-04-SUMMARY.md)
"""

import os
import pandas as pd
import numpy as np
from pathlib import Path
from tqdm import tqdm
import pickle
import json

# Import v2 feature extraction for base features
from src.data_processing.feature_engineering_v2 import (
    extract_frame_features,
    extract_temporal_features,
    extract_statistical_summary
)

# Import phase segmentation
from src.data_processing.phase_segmentation import segment_and_validate

# Import P0 features
from src.data_processing.kinetic_chain_features import extract_kinetic_chain_timing
from src.data_processing.contact_frame_features import (
    extract_contact_frame_features,
    extract_intent_window_features,
    calculate_smash_intent_score
)

# Import P1 features
from src.data_processing.angular_velocity_features import (
    extract_angular_velocity_features,
    estimate_racket_head_speed
)
from src.data_processing.phase_specific_features import (
    extract_phase_specific_features,
    calculate_deceleration_features
)


# Paths
BASE_DIR = Path(__file__).resolve().parents[2]
POSES_DIR = BASE_DIR / "data" / "processed" / "poses"
FEATURES_V3_DIR = BASE_DIR / "data" / "processed" / "features_v3"
SELECTION_PATH = BASE_DIR / "data" / "processed" / "features_v3" / "selected_features.json"
CLIPS_METADATA_FILE = BASE_DIR / "data" / "processed" / "clips" / "clips_metadata.csv"
REPORTS_DIR = BASE_DIR / "outputs" / "reports"


def load_selected_features():
    """
    Load selected feature list from JSON manifest.

    Returns:
        list of selected feature names, or None if manifest not populated
    """
    if not SELECTION_PATH.exists():
        return None

    with open(SELECTION_PATH, 'r') as f:
        manifest = json.load(f)

    selected = manifest.get('selected_features')

    if not selected or len(selected) == 0:
        # Manifest exists but is placeholder (Plan 04 not yet executed)
        return None

    return selected


def extract_v2_base_features(pose_sequence):
    """
    Wrapper around v2 extraction for base spatial and temporal features.

    Args:
        pose_sequence: Array of shape (num_frames, 33, 3) - MediaPipe pose sequence

    Returns:
        dict with v2 baseline features
    """
    # Extract temporal features (includes per-frame and sequence-level)
    result = extract_temporal_features(pose_sequence)

    if result[0] is None:
        return None

    per_frame_features, feature_names, temporal_features = result

    # Extract statistical summary
    summary = extract_statistical_summary(per_frame_features, feature_names)

    # Combine statistical summary and temporal features
    base_features = {**summary, **temporal_features}

    return base_features


def extract_features_v3(pose_sequence, apply_selection=True):
    """
    Main v3 feature extraction combining all feature modules.

    Pipeline:
    1. Phase segmentation (5 phases)
    2. V2 base features (spatial + temporal statistical summary)
    3. P0 features (kinetic chain, contact frame, intent window, SIS)
    4. P1 features (angular velocity, phase-specific, racket speed)
    5. Combine all features
    6. Apply feature selection mask (if enabled and manifest populated)

    Args:
        pose_sequence: Array of shape (num_frames, 33, 3) - MediaPipe pose sequence
        apply_selection: bool (default True) - Apply feature selection mask

    Returns:
        dict with all v3 features (or selected subset if apply_selection=True)
    """
    # Step 1: Phase segmentation
    phases, validation = segment_and_validate(pose_sequence)

    # Step 2: V2 base features (spatial + temporal)
    base_features = extract_v2_base_features(pose_sequence)

    if base_features is None:
        return None

    # Step 3: P0 features (kinetic chain + contact frame + intent window)
    # Kinetic chain timing
    kinetic_chain = extract_kinetic_chain_timing(pose_sequence, phases)

    # Contact frame features
    contact_frame = phases['contact']
    contact_features = extract_contact_frame_features(pose_sequence, contact_frame)

    # Intent window features
    intent_features = extract_intent_window_features(pose_sequence, contact_frame)

    # Smash Intent Score (SIS)
    sis_result = calculate_smash_intent_score(intent_features)

    # Step 4: P1 features (angular velocity + phase-specific)
    handedness = phases['handedness']

    # Angular velocity features
    angular_vel = extract_angular_velocity_features(
        pose_sequence,
        fps=30,
        handedness=handedness
    )

    # Phase-specific features
    phase_specific = extract_phase_specific_features(
        pose_sequence,
        phases,
        fps=30,
        handedness=handedness
    )

    # Racket head speed estimation
    racket_speed = estimate_racket_head_speed(
        pose_sequence,
        phases,
        fps=30,
        handedness=handedness
    )

    # Deceleration features
    decel_features = calculate_deceleration_features(
        pose_sequence,
        phases,
        fps=30,
        handedness=handedness
    )

    # Step 5: Combine all features
    all_features = {
        # V2 base features
        **base_features,
        # P0 features
        **kinetic_chain,
        **contact_features,
        **intent_features,
        'smash_intent_score': sis_result['smash_intent_score'],
        'sis_classification_hint': sis_result['classification_hint'],
        # P1 features
        **angular_vel,
        **phase_specific,
        **racket_speed,
        **decel_features,
        # Metadata
        'phase_validation_passed': validation['valid'],
        'contact_position_pct': validation['contact_position_pct'],
        'forward_swing_pct': validation['forward_swing_pct'],
        'feature_version': 'v3',
        'handedness': handedness
    }

    # Step 6: Apply feature selection (if enabled and manifest populated)
    if apply_selection:
        selected = load_selected_features()

        if selected is not None:
            # Filter to selected features only (keep metadata fields)
            filtered = {}
            for k, v in all_features.items():
                if k in selected or k.startswith('_') or k in [
                    'feature_version', 'handedness', 'phase_validation_passed',
                    'sis_classification_hint', 'contact_position_pct', 'forward_swing_pct'
                ]:
                    filtered[k] = v

            return filtered
        else:
            # Manifest not populated - return all features without selection
            # This allows v3 extraction to work before Plan 04 runs
            return all_features
    else:
        # Selection disabled - return all features
        return all_features


def process_single_clip_v3(pose_path, target_length=50, apply_selection=True):
    """
    Process single clip with v3 feature extraction.

    Analogous to v2's process_single_clip, but uses v3 extraction pipeline.

    Args:
        pose_path: Path to pose pickle file
        target_length: Target sequence length for LSTM (default 50)
        apply_selection: bool (default True) - Apply feature selection mask

    Returns:
        dict with:
            - features: dict with all extracted features
            - statistical_summary: dict (same as features for compatibility)
            - original_length: int (number of frames in original sequence)
    """
    with open(pose_path, 'rb') as f:
        pose_data = pickle.load(f)

    poses = pose_data['poses']

    # Extract v3 features
    features = extract_features_v3(poses, apply_selection=apply_selection)

    if features is None:
        return None

    return {
        'features': features,
        'statistical_summary': features,  # For compatibility with v2 interface
        'original_length': len(poses)
    }


def process_all_clips_v3(output_dir=None, apply_selection=True):
    """
    Batch processing for all clips with v3 feature extraction.

    Analogous to v2's process_all_clips, but uses v3 extraction pipeline.

    Args:
        output_dir: Output directory for features (default: FEATURES_V3_DIR)
        apply_selection: bool (default True) - Apply feature selection mask

    Returns:
        None (saves features to disk and generates processing log)
    """
    if output_dir is None:
        output_dir = FEATURES_V3_DIR

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("FEATURE ENGINEERING V3 - COACH-INFORMED FEATURES")
    print("=" * 70)
    print()

    # Check if selection manifest exists
    selected = load_selected_features()
    if apply_selection:
        if selected is not None:
            print(f"Feature selection: ENABLED ({len(selected)} features selected)")
        else:
            print(f"Feature selection: DISABLED (manifest not populated - Plan 04 not yet run)")
            print(f"  Extracting all features without selection")
    else:
        print(f"Feature selection: DISABLED (apply_selection=False)")
    print()

    # Load clips metadata
    clips_df = pd.read_csv(CLIPS_METADATA_FILE)
    print(f"Loaded metadata for {len(clips_df)} clips")
    print()

    # Statistics
    successful = 0
    failed = 0
    feature_log = []

    print("Processing clips...")
    for idx, clip in tqdm(clips_df.iterrows(), total=len(clips_df), desc="Extracting v3 features"):
        clip_name = clip['clip_name']
        pose_path = POSES_DIR / clip_name.replace('.mp4', '_pose.pkl')
        feature_path = output_dir / clip_name.replace('.mp4', '_features_v3.pkl')

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
            result = process_single_clip_v3(pose_path, apply_selection=apply_selection)

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
                'num_features': len(result['features']),
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
    log_file = REPORTS_DIR / "feature_extraction_v3_log.csv"
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    log_df.to_csv(log_file, index=False)

    # Print summary
    print()
    print("=" * 70)
    print("FEATURE EXTRACTION V3 SUMMARY")
    print("=" * 70)
    print(f"Total clips: {len(clips_df)}")
    print(f"Successful: {successful} ({successful/len(clips_df)*100:.1f}%)")
    print(f"Failed: {failed}")
    print()

    # Feature info
    if successful > 0:
        sample_path = output_dir / (clips_df.iloc[0]['clip_name'].replace('.mp4', '_features_v3.pkl'))
        if sample_path.exists():
            with open(sample_path, 'rb') as f:
                sample = pickle.load(f)
            print("Feature dimensions:")
            print(f"  Features extracted: {len(sample['features'])} features")
            if apply_selection and selected is not None:
                print(f"  Selection applied: {len(selected)} features selected")
            print()

    print(f"Features saved to: {output_dir}")
    print(f"Log saved to: {log_file}")


def main():
    print("\n")
    print("*" * 70)
    print("*" + " " * 68 + "*")
    print("*" + "  ITI123 Project: Feature Engineering V3".center(68) + "*")
    print("*" + " " * 68 + "*")
    print("*" * 70)
    print("\n")

    print("Key improvements in V3:")
    print("  1. Phase segmentation (5 phases)")
    print("  2. P0 features: kinetic chain timing, contact frame, intent window")
    print("  3. Smash Intent Score (SIS) calculation")
    print("  4. P1 features: angular velocity, phase-specific statistics")
    print("  5. Racket head speed estimation")
    print("  6. Feature selection applied (< 254 features)")
    print()

    process_all_clips_v3()

    print("=" * 70)
    print("FEATURE ENGINEERING V3 COMPLETE!")
    print("=" * 70)
    print()
    print("Next steps:")
    print("1. Run: python src/data_processing/data_split.py")
    print("2. Train models in Colab with v3 features")
    print()


if __name__ == "__main__":
    main()
