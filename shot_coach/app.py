"""
Shot Coach - Streamlit Web Application
Badminton shot technique analyzer
"""

import streamlit as st
import sys
from pathlib import Path
import tempfile
import shutil

# Add modules to path
sys.path.append(str(Path(__file__).parent / 'modules'))

from shot_classifier import ShotClassifier
from pose_extractor import PoseExtractor
from technique_analyzer import TechniqueAnalyzer


# Page config
st.set_page_config(
    page_title="Badminton Shot Coach",
    page_icon="🏸",
    layout="wide"
)


@st.cache_resource
def load_models():
    """Load models once and cache"""
    # Path to trained model
    model_path = Path(__file__).parent.parent / 'outputs' / 'results_optionA' / 'best_model.pth'

    if not model_path.exists():
        st.error(f"Model not found at: {model_path}")
        st.stop()

    classifier = ShotClassifier(model_path)
    pose_extractor = PoseExtractor()
    analyzer = TechniqueAnalyzer()

    return classifier, pose_extractor, analyzer


def format_report(shot_type, confidence, technique_result):
    """Format analysis report as text"""

    report = []
    report.append("=" * 70)
    report.append("BADMINTON SHOT COACH - ANALYSIS REPORT")
    report.append("=" * 70)
    report.append("")

    # Shot Classification
    report.append("SHOT CLASSIFICATION")
    report.append("-" * 70)
    report.append(f"Shot Type: {shot_type}")
    report.append(f"Confidence: {confidence*100:.1f}%")
    report.append("")

    # Technique Analysis
    if technique_result['success']:
        report.append("TECHNIQUE ANALYSIS")
        report.append("-" * 70)
        report.append(f"Overall Score: {technique_result['overall_score']}/100")
        report.append("")

        # Feedback sections
        for feedback_section in technique_result['feedback']:
            category = feedback_section['category']
            report.append(f"{category.upper()}")
            report.append("-" * 70)

            if category == 'Overall Assessment':
                report.append(f"Rating: {feedback_section['rating']}")
                report.append(f"{feedback_section['comment']}")
            else:
                for item in feedback_section['items']:
                    report.append(f"  • {item}")

            report.append("")

        # Detailed Metrics
        report.append("DETAILED METRICS")
        report.append("-" * 70)
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

            # Status indicator
            if score >= 85:
                status = "✅ Excellent"
            elif score >= 65:
                status = "⚠️  Good"
            else:
                status = "❌ Needs Work"

            report.append(f"{metric_display:20s}: {value_str:>8s} (target: {optimal_str:>8s}) - Score: {score:>5.1f}/100 {status}")

        report.append("")

    else:
        report.append("TECHNIQUE ANALYSIS")
        report.append("-" * 70)
        report.append(f"Error: {technique_result.get('error', 'Unknown error')}")
        report.append("")

    report.append("=" * 70)
    report.append("Thank you for using Badminton Shot Coach!")
    report.append("Keep practicing to improve your technique! 🏸")
    report.append("=" * 70)

    return "\n".join(report)


def main():
    # Header
    st.title("🏸 Badminton Shot Coach")
    st.markdown("### AI-Powered Shot Technique Analysis")
    st.markdown("Upload a video of your Smash or Clear shot for instant feedback!")

    st.markdown("---")

    # Load models
    with st.spinner("Loading AI models..."):
        try:
            classifier, pose_extractor, analyzer = load_models()
            st.success("✅ Models loaded successfully!")
        except Exception as e:
            st.error(f"Error loading models: {e}")
            st.stop()

    st.markdown("---")

    # Instructions
    with st.expander("📖 How to Use", expanded=False):
        st.markdown("""
        **Instructions:**
        1. Record a 2-3 second video of your shot (Smash or Clear)
        2. Make sure you're visible in the frame (full body preferred)
        3. Upload the video using the button below
        4. Wait for analysis (takes 10-30 seconds)
        5. Review your technique report!

        **Supported Shot Types:**
        - 🔥 **Smash** - Overhead attacking shot
        - 🎯 **Clear** - High defensive shot

        **What Gets Analyzed:**
        - Shot type detection (AI classification)
        - Technique metrics (arm extension, rotation, etc.)
        - Overall score (0-100)
        - Specific improvement recommendations
        """)

    # File upload
    st.markdown("### Upload Your Shot Video")
    uploaded_file = st.file_uploader(
        "Choose a video file (MP4, AVI, MOV)",
        type=['mp4', 'avi', 'mov', 'MP4', 'AVI', 'MOV']
    )

    if uploaded_file is not None:
        # Show video
        st.markdown("#### Your Video")
        st.video(uploaded_file)

        # Analyze button
        if st.button("🚀 Analyze Shot", type="primary"):
            # Save uploaded file temporarily
            with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as tmp_file:
                tmp_file.write(uploaded_file.read())
                tmp_video_path = tmp_file.name

            try:
                # Progress
                progress_bar = st.progress(0)
                status_text = st.empty()

                # Step 1: Classify shot type
                status_text.text("🔍 Classifying shot type...")
                progress_bar.progress(20)

                classification_result = classifier.predict(tmp_video_path, num_frames=16)

                if not classification_result['success']:
                    st.error(f"Error classifying shot: {classification_result.get('error')}")
                    return

                shot_type = classification_result['predicted_class']
                confidence = classification_result['confidence']

                progress_bar.progress(40)

                # Check if supported
                if shot_type not in ['Smash', 'Clear']:
                    st.warning(f"⚠️ Detected shot type: **{shot_type}**")
                    st.info("Currently, Shot Coach only supports **Smash** and **Clear** analysis.")
                    st.info("Support for Drive, Drop, and Lift coming soon!")
                    return

                # Step 2: Extract pose
                status_text.text("🦴 Extracting pose keypoints...")
                progress_bar.progress(60)

                pose_result = pose_extractor.extract_from_video(tmp_video_path)

                if not pose_result['success']:
                    st.error(f"Error extracting pose: {pose_result.get('error')}")
                    st.info("💡 Tip: Make sure you're clearly visible in the video!")
                    return

                keypoints_list = pose_result['keypoints']

                # Find contact frame
                contact_frame_idx = pose_extractor.find_contact_frame(keypoints_list)

                progress_bar.progress(75)

                # Step 3: Analyze technique
                status_text.text("📊 Analyzing technique...")
                progress_bar.progress(90)

                technique_result = analyzer.analyze_shot(
                    keypoints_list,
                    shot_type,
                    contact_frame_idx
                )

                progress_bar.progress(100)
                status_text.text("✅ Analysis complete!")

                # Generate report
                report_text = format_report(shot_type, confidence, technique_result)

                # Display results
                st.markdown("---")
                st.markdown("## 📋 Analysis Report")

                # Show in columns
                col1, col2 = st.columns([1, 1])

                with col1:
                    st.markdown("### Shot Classification")
                    st.metric("Shot Type", shot_type)
                    st.metric("Confidence", f"{confidence*100:.1f}%")

                with col2:
                    if technique_result['success']:
                        st.markdown("### Technique Score")
                        score = technique_result['overall_score']

                        # Color based on score
                        if score >= 85:
                            score_color = "🟢"
                        elif score >= 70:
                            score_color = "🟡"
                        else:
                            score_color = "🔴"

                        st.metric("Overall Score", f"{score_color} {score}/100")

                # Show feedback
                st.markdown("---")
                st.markdown("### 💡 Feedback")

                if technique_result['success']:
                    for feedback_section in technique_result['feedback']:
                        category = feedback_section['category']

                        if category == 'Overall Assessment':
                            st.success(f"**{feedback_section['rating']}**: {feedback_section['comment']}")
                        else:
                            with st.expander(f"📌 {category}", expanded=True):
                                for item in feedback_section['items']:
                                    st.markdown(f"- {item}")

                # Show detailed metrics
                st.markdown("---")
                st.markdown("### 📊 Detailed Metrics")

                if technique_result['success']:
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

                        col1, col2, col3 = st.columns([2, 1, 1])
                        with col1:
                            st.markdown(f"**{metric_display}**")
                        with col2:
                            st.markdown(f"{value_str} (target: {optimal_str})")
                        with col3:
                            # Progress bar for score
                            score_pct = score / 100
                            if score >= 85:
                                st.progress(score_pct, text=f"✅ {score:.0f}/100")
                            elif score >= 65:
                                st.progress(score_pct, text=f"⚠️ {score:.0f}/100")
                            else:
                                st.progress(score_pct, text=f"❌ {score:.0f}/100")

                # Download report
                st.markdown("---")
                st.markdown("### 📥 Download Report")

                st.download_button(
                    label="Download Text Report",
                    data=report_text,
                    file_name=f"shot_analysis_{shot_type.lower()}.txt",
                    mime="text/plain"
                )

            except Exception as e:
                st.error(f"Error during analysis: {e}")
                import traceback
                st.code(traceback.format_exc())

            finally:
                # Cleanup
                try:
                    Path(tmp_video_path).unlink()
                except:
                    pass

    else:
        st.info("👆 Upload a video to get started!")

    # Footer
    st.markdown("---")
    st.markdown("""
    <div style='text-align: center; color: gray;'>
    <p>Badminton Shot Coach v1.0 | Powered by AI & Computer Vision</p>
    <p>Built with ResNet18+BiLSTM (74.6% accuracy) + MediaPipe Pose</p>
    </div>
    """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()
