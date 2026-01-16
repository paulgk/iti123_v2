#!/usr/bin/env python3
"""
AI Badminton Coach - Video Analysis

Analyze a badminton stroke video and get coaching feedback.

Usage:
    python analyze_video.py video.mp4 Clear
    python analyze_video.py video.mp4 Smash
"""

import sys
from pathlib import Path
import pickle
import numpy as np

# Add parent directory to path
sys.path.append(str(Path(__file__).resolve().parent))

from src.data_processing.extract_poses import PoseExtractor
from src.data_processing.feature_engineering_v2 import (
    extract_frame_features,
    extract_temporal_features,
    extract_statistical_summary
)
from src.coaching import CoachingFeedback, TechniqueVisualizer


def detect_stroke_type(stat_features):
    """
    Automatically detect stroke type based on biomechanical features

    Uses velocity, forearm angle, and contact point to discriminate
    between Clear and Smash strokes.
    """
    max_velocity = stat_features.get('max_velocity', 70)
    forearm_angle_mean = stat_features.get('r_forearm_vertical_angle_mean', 45)
    contact_point_mean = stat_features.get('r_wrist_height_from_head_mean', 0)

    # Scoring system
    smash_score = 0
    clear_score = 0

    # Velocity (Smash > 80, Clear < 80)
    if max_velocity > 90:
        smash_score += 3
    elif max_velocity > 80:
        smash_score += 2
    elif max_velocity < 70:
        clear_score += 2
    else:
        clear_score += 1

    # Forearm angle (Smash: 15-47°, Clear: 32-69°)
    if forearm_angle_mean < 30:
        smash_score += 2
    elif forearm_angle_mean > 50:
        clear_score += 2
    else:
        if forearm_angle_mean < 40:
            smash_score += 1
        else:
            clear_score += 1

    # Contact point (Clear is higher = more negative)
    if contact_point_mean < -0.01:
        clear_score += 2
    elif contact_point_mean > 0.02:
        smash_score += 1

    # Decision
    detected = 'Smash' if smash_score > clear_score else 'Clear'
    confidence = abs(smash_score - clear_score) / (smash_score + clear_score) * 100

    return detected, confidence


def analyze_video(video_path: str, stroke_type: str = None):
    """
    Complete pipeline: video -> poses -> features -> coaching feedback

    Args:
        video_path: Path to video file
        stroke_type: 'Clear' or 'Smash' (auto-detect if not specified)

    Returns:
        None (saves results to outputs/)
    """
    video_file = Path(video_path)

    if not video_file.exists():
        print(f"❌ Error: Video file not found: {video_path}")
        return

    # Handle stroke type
    auto_detect = False
    if stroke_type is None:
        print("🤖 Stroke type not specified. Will auto-detect from video features.")
        auto_detect = True
    elif stroke_type not in ['Clear', 'Smash']:
        print(f"❌ Error: Invalid stroke type '{stroke_type}'")
        print("   Must be 'Clear' or 'Smash'")
        return

    print("=" * 70)
    print("AI BADMINTON COACH - VIDEO ANALYSIS")
    print("=" * 70)
    print(f"\nVideo: {video_path}")
    if auto_detect:
        print(f"Stroke Type: Auto-detect")
    else:
        print(f"Stroke Type: {stroke_type}")
    print()

    # =========================================================================
    # STEP 1: Extract poses from video
    # =========================================================================
    print("Step 1/3: Extracting poses with MediaPipe...")
    try:
        extractor = PoseExtractor()
        pose_data = extractor.extract_from_video(video_path, stroke_type)
        extractor.close()

        num_frames = pose_data['num_frames']
        quality = pose_data['quality']['valid_percentage']

        print(f"✓ Extracted {num_frames} frames")
        print(f"✓ Quality: {quality:.1f}% valid frames")

    except Exception as e:
        print(f"❌ Error during pose extraction: {e}")
        print("\nTroubleshooting:")
        print("1. Ensure player is clearly visible throughout the video")
        print("2. Video should show one complete stroke (1-3 seconds)")
        print("3. Check video file is valid and not corrupted")
        print("4. If you see mutex error, run: pip install protobuf==3.20.3")
        print("5. Run diagnostic: python diagnose.py")
        print("\nTip: Try with a sample video from data/processed/clips/ first")
        return

    # =========================================================================
    # STEP 2: Extract biomechanical features
    # =========================================================================
    print("\nStep 2/3: Extracting biomechanical features...")
    try:
        pose_sequence = pose_data['poses']

        # Extract temporal features (includes velocity)
        per_frame_features, feature_names, temporal_metrics = extract_temporal_features(pose_sequence)

        if per_frame_features is None:
            print("❌ Failed to extract features (need at least 5 valid frames)")
            return

        # Extract statistical summary
        stat_features = extract_statistical_summary(per_frame_features, feature_names)

        # Combine temporal metrics
        stat_features.update(temporal_metrics)

        print(f"✓ Extracted {len(stat_features)} statistical features")
        print(f"✓ Sequence shape: {per_frame_features.shape}")

        # Auto-detect stroke type if needed
        if auto_detect:
            print("\n🤖 Auto-detecting stroke type...")
            detected_stroke, confidence = detect_stroke_type(stat_features)
            print(f"✓ Detected: {detected_stroke} (confidence: {confidence:.1f}%)")
            stroke_type = detected_stroke

    except Exception as e:
        print(f"❌ Error during feature extraction: {e}")
        import traceback
        traceback.print_exc()
        return

    # =========================================================================
    # STEP 3: Generate coaching feedback
    # =========================================================================
    print("\nStep 3/3: Generating coaching feedback...")
    try:
        # Initialize coach
        coach = CoachingFeedback()
        feedback_items = coach.analyze_technique(stat_features, stroke_type)

        # Setup output directory
        output_dir = Path("outputs") / "video_analysis" / video_file.stem
        output_dir.mkdir(parents=True, exist_ok=True)

        # Create visualizations
        viz = TechniqueVisualizer(output_dir=output_dir)
        viz.create_radar_chart(stat_features, stroke_type)
        viz.create_metrics_bar_chart(feedback_items, stroke_type)
        viz.create_score_gauge(coach.overall_score, stroke_type)
        viz.create_comprehensive_report(stat_features, stroke_type, coach)

        # Save text feedback
        feedback_text = coach.generate_summary()
        feedback_file = output_dir / "feedback.txt"
        with open(feedback_file, 'w') as f:
            f.write(feedback_text)

        print(f"✓ Generated {len(feedback_items)} feedback items")
        print(f"✓ Overall score: {coach.overall_score}/100")

    except Exception as e:
        print(f"❌ Error during coaching analysis: {e}")
        import traceback
        traceback.print_exc()
        return

    # =========================================================================
    # Display Results
    # =========================================================================
    print("\n" + "=" * 70)
    print("TECHNIQUE ANALYSIS RESULTS")
    print("=" * 70)
    print()

    # Overall score
    if coach.overall_score >= 85:
        grade = "Excellent ⭐⭐⭐"
    elif coach.overall_score >= 70:
        grade = "Good ⭐⭐"
    elif coach.overall_score >= 50:
        grade = "Needs Improvement ⭐"
    else:
        grade = "Needs Attention ⚠️"

    print(f"Overall Score: {coach.overall_score}/100 - {grade}")
    print()

    # Issue breakdown
    critical = sum(1 for item in feedback_items if item.severity == 'critical')
    major = sum(1 for item in feedback_items if item.severity == 'major')
    minor = sum(1 for item in feedback_items if item.severity == 'minor')
    good = sum(1 for item in feedback_items if item.severity == 'good')

    print("Issues Found:")
    if critical > 0:
        print(f"  🔴 Critical: {critical}")
    if major > 0:
        print(f"  ⚠️  Major: {major}")
    if minor > 0:
        print(f"  💡 Minor: {minor}")
    if good > 0:
        print(f"  ✅ Good: {good}")
    print()

    # Top 3 priority actions
    priority_items = [item for item in feedback_items if item.severity != 'good'][:3]

    if priority_items:
        print("=" * 70)
        print("TOP PRIORITY ACTIONS")
        print("=" * 70)
        print()

        for i, item in enumerate(priority_items, 1):
            icon = {'critical': '🔴', 'major': '⚠️', 'minor': '💡'}.get(item.severity, '📊')
            print(f"{i}. {icon} {item.message}")
            print(f"   🎯 {item.drill}")
            print()

    # Output location
    print("=" * 70)
    print("SAVED RESULTS")
    print("=" * 70)
    print(f"\n📁 All results saved to: {output_dir}/")
    print(f"\nFiles created:")
    print(f"  - feedback.txt")
    print(f"  - comprehensive_report_{stroke_type.lower()}.png")
    print(f"  - radar_chart_{stroke_type.lower()}.png")
    print(f"  - metrics_bar_chart_{stroke_type.lower()}.png")
    print(f"  - score_gauge_{stroke_type.lower()}.png")
    print()

    print("=" * 70)
    print("✅ ANALYSIS COMPLETE")
    print("=" * 70)
    print()


def main():
    """Main entry point"""
    if len(sys.argv) < 2:
        print("=" * 70)
        print("AI BADMINTON COACH - VIDEO ANALYSIS")
        print("=" * 70)
        print("\nUsage:")
        print("  python analyze_video.py <video_file> [stroke_type]")
        print("\nExamples:")
        print("  python analyze_video.py my_video.mp4              # Auto-detect stroke type")
        print("  python analyze_video.py my_clear.mp4 Clear        # Specify Clear")
        print("  python analyze_video.py my_smash.mp4 Smash        # Specify Smash")
        print("\nStroke Types:")
        print("  - Clear: Overhead defensive shot to back of court")
        print("  - Smash: Overhead attacking shot (downward)")
        print("  - (blank): Auto-detect from video features")
        print("\nNote: Video should show one badminton stroke with player visible")
        print()
        return

    video_path = sys.argv[1]
    stroke_type = sys.argv[2] if len(sys.argv) > 2 else None

    analyze_video(video_path, stroke_type)


if __name__ == "__main__":
    main()
