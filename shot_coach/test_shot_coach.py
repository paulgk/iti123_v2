"""
Test script for Shot Coach
Tests the complete pipeline with a sample video
"""

import sys
from pathlib import Path

# Add modules to path
sys.path.append(str(Path(__file__).parent / 'modules'))

from shot_classifier import ShotClassifier
from pose_extractor import PoseExtractor
from technique_analyzer import TechniqueAnalyzer


def test_shot_coach(video_path):
    """Test the shot coach pipeline"""

    print("=" * 70)
    print("SHOT COACH TEST")
    print("=" * 70)
    print()

    # Check video exists
    video_path = Path(video_path)
    if not video_path.exists():
        print(f"❌ Video not found: {video_path}")
        return

    print(f"📹 Video: {video_path.name}")
    print()

    # Initialize modules
    print("Loading models...")
    model_path = Path(__file__).parent.parent / 'outputs' / 'results_optionA' / 'best_model.pth'

    if not model_path.exists():
        print(f"❌ Model not found: {model_path}")
        return

    classifier = ShotClassifier(model_path)
    pose_extractor = PoseExtractor()
    analyzer = TechniqueAnalyzer()

    print("✅ Models loaded")
    print()

    # Step 1: Classify shot
    print("Step 1: Classifying shot type...")
    classification_result = classifier.predict(video_path, num_frames=16)

    if not classification_result['success']:
        print(f"❌ Error: {classification_result.get('error')}")
        return

    shot_type = classification_result['predicted_class']
    confidence = classification_result['confidence']

    print(f"✅ Shot Type: {shot_type} ({confidence*100:.1f}% confidence)")
    print()

    # Check if supported
    if shot_type not in ['Smash', 'Clear']:
        print(f"⚠️  Shot type '{shot_type}' not yet supported for technique analysis")
        print("   Only Smash and Clear are currently analyzed")
        return

    # Step 2: Extract pose
    print("Step 2: Extracting pose keypoints...")
    pose_result = pose_extractor.extract_from_video(video_path)

    if not pose_result['success']:
        print(f"❌ Error: {pose_result.get('error')}")
        return

    keypoints_list = pose_result['keypoints']
    valid_frames = pose_result['valid_frames']

    print(f"✅ Extracted {valid_frames} valid pose frames")

    # Find contact frame
    contact_frame_idx = pose_extractor.find_contact_frame(keypoints_list)
    print(f"   Contact frame: {contact_frame_idx}")
    print()

    # Step 3: Analyze technique
    print("Step 3: Analyzing technique...")
    technique_result = analyzer.analyze_shot(keypoints_list, shot_type, contact_frame_idx)

    if not technique_result['success']:
        print(f"❌ Error: {technique_result.get('error')}")
        return

    print(f"✅ Analysis complete")
    print()

    # Display results
    print("=" * 70)
    print("RESULTS")
    print("=" * 70)
    print()

    print(f"Shot Type: {shot_type}")
    print(f"Confidence: {confidence*100:.1f}%")
    print(f"Overall Score: {technique_result['overall_score']}/100")
    print()

    print("Feedback:")
    print("-" * 70)
    for feedback_section in technique_result['feedback']:
        category = feedback_section['category']
        print(f"\n{category}:")

        if category == 'Overall Assessment':
            print(f"  Rating: {feedback_section['rating']}")
            print(f"  {feedback_section['comment']}")
        else:
            for item in feedback_section['items']:
                print(f"  • {item}")

    print()
    print("Detailed Metrics:")
    print("-" * 70)
    metrics = technique_result['metrics']

    for metric_name, metric_data in metrics.items():
        metric_display = metric_name.replace('_', ' ').title()
        value = metric_data['value']
        score = metric_data['score']
        optimal = metric_data['optimal']
        unit = metric_data['unit']

        if unit == 'ratio':
            value_str = f"{value:.0%}"
            optimal_str = f"{optimal:.0%}"
        elif unit == 'degrees':
            value_str = f"{value:.1f}°"
            optimal_str = f"{optimal:.0f}°"
        else:
            value_str = f"{value:.2f}"
            optimal_str = f"{optimal:.2f}"

        # Status
        if score >= 85:
            status = "✅"
        elif score >= 65:
            status = "⚠️ "
        else:
            status = "❌"

        print(f"{status} {metric_display:20s}: {value_str:>8s} (target: {optimal_str:>8s}) - {score:>5.1f}/100")

    print()
    print("=" * 70)
    print("TEST COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python test_shot_coach.py <video_path>")
        print()
        print("Example:")
        print("  python test_shot_coach.py ../data/clips/Smash/01_set1_rally1_ball3_Smash.mp4")
    else:
        test_shot_coach(sys.argv[1])
