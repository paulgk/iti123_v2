"""
AI Badminton Coach - Streamlit Interface

A web application using Streamlit for analyzing badminton stroke technique.
Provides coaching feedback, visualizations, and practice drills.

Usage:
    streamlit run src/deployment/streamlit_app.py
"""

import streamlit as st
import pickle
import tempfile
from pathlib import Path
import sys

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).resolve().parents[2]))

from src.coaching import TechniqueBenchmarks, CoachingFeedback, TechniqueVisualizer
from src.data_processing.extract_poses import PoseExtractor
from src.data_processing.feature_engineering_v2 import (
    extract_temporal_features,
    extract_statistical_summary
)


# Page configuration
st.set_page_config(
    page_title="AI Badminton Coach",
    page_icon="🏸",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        color: #2E86AB;
        text-align: center;
        padding: 1rem;
    }
    .score-box {
        background-color: #f0f2f6;
        padding: 1.5rem;
        border-radius: 10px;
        border-left: 5px solid #2E86AB;
    }
    .critical-issue {
        background-color: #ffe6e6;
        padding: 0.5rem;
        border-radius: 5px;
        border-left: 4px solid #e74c3c;
        margin: 0.5rem 0;
    }
    .major-issue {
        background-color: #fff4e6;
        padding: 0.5rem;
        border-radius: 5px;
        border-left: 4px solid #f39c12;
        margin: 0.5rem 0;
    }
    .minor-issue {
        background-color: #fffde6;
        padding: 0.5rem;
        border-radius: 5px;
        border-left: 4px solid #f1c40f;
        margin: 0.5rem 0;
    }
    .good-technique {
        background-color: #e6f7e6;
        padding: 0.5rem;
        border-radius: 5px;
        border-left: 4px solid #2ecc71;
        margin: 0.5rem 0;
    }
</style>
""", unsafe_allow_html=True)


def detect_stroke_type(stat_features):
    """
    Automatically detect stroke type based on biomechanical features

    Args:
        stat_features: Dictionary of statistical features

    Returns:
        str: 'Clear' or 'Smash'
    """
    # Key discriminative features:
    # 1. Velocity: Smash is faster
    # 2. Forearm angle: Smash points more downward
    # 3. Contact point: Clear is higher

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
        # Overlap range, less weight
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


def process_video(video_path, stroke_type):
    """
    Process video: extract poses and features

    Args:
        video_path: Path to video file
        stroke_type: 'Clear', 'Smash', or None for auto-detect

    Returns:
        Dictionary with statistical_summary and detected_stroke, or None if failed
    """
    try:
        # Step 1: Extract poses (use 'Clear' as default for extraction)
        st.write("⏳ Extracting poses with MediaPipe...")
        extractor = PoseExtractor()
        pose_data = extractor.extract_from_video(video_path, stroke_type or 'Clear')
        extractor.close()

        # Check quality
        num_frames = pose_data['num_frames']
        valid_pct = pose_data['quality']['valid_percentage']

        st.write(f"✓ Extracted {num_frames} frames ({valid_pct:.1f}% valid)")

        if valid_pct < 50:
            st.warning(f"⚠️ Low quality: Only {valid_pct:.1f}% of frames have valid poses. Results may be inaccurate.")

        # Step 2: Extract features
        st.write("⏳ Extracting biomechanical features...")
        pose_sequence = pose_data['poses']
        per_frame_features, feature_names, temporal_metrics = extract_temporal_features(pose_sequence)

        if per_frame_features is None:
            st.error("❌ Not enough valid frames (need at least 5). Player must be clearly visible.")
            return None

        stat_features = extract_statistical_summary(per_frame_features, feature_names)
        stat_features.update(temporal_metrics)

        st.write(f"✓ Extracted {len(stat_features)} features")

        # Step 3: Auto-detect stroke type if not provided
        if stroke_type is None:
            st.write("⏳ Detecting stroke type...")
            detected_stroke, confidence = detect_stroke_type(stat_features)
            st.write(f"✓ Detected: **{detected_stroke}** (confidence: {confidence:.1f}%)")
        else:
            detected_stroke = stroke_type

        return {
            'statistical_summary': stat_features,
            'quality': pose_data['quality'],
            'detected_stroke': detected_stroke
        }

    except Exception as e:
        st.error(f"❌ Error processing video: {str(e)}")
        st.write("**Troubleshooting:**")
        st.write("1. Ensure player is clearly visible throughout the video")
        st.write("2. Video should show one complete stroke (1-3 seconds)")
        st.write("3. Check video is not corrupted")
        st.write("4. Try a different video clip")

        # Show detailed error in expander
        with st.expander("Show detailed error"):
            import traceback
            st.code(traceback.format_exc())

        return None


def analyze_features(features_data, stroke_type):
    """
    Analyze features and generate coaching feedback

    Args:
        features_data: Dictionary with statistical_summary
        stroke_type: 'Clear' or 'Smash'

    Returns:
        Tuple of (coach, output_dir)
    """
    # Get statistical features
    stat_features = features_data.get('statistical_summary', features_data)

    # Generate coaching feedback
    coach = CoachingFeedback()
    feedback_items = coach.analyze_technique(stat_features, stroke_type)

    # Create temporary directory for visualizations
    temp_dir = Path(tempfile.mkdtemp())

    # Generate visualizations
    viz = TechniqueVisualizer(output_dir=temp_dir)
    viz.create_radar_chart(stat_features, stroke_type)
    viz.create_metrics_bar_chart(feedback_items, stroke_type)
    viz.create_score_gauge(coach.overall_score, stroke_type)
    viz.create_comprehensive_report(stat_features, stroke_type, coach)

    return coach, temp_dir


def display_feedback_item(item):
    """Display a single feedback item with appropriate styling"""

    severity_icons = {
        'critical': '🔴',
        'major': '⚠️',
        'minor': '💡',
        'good': '✅'
    }

    severity_classes = {
        'critical': 'critical-issue',
        'major': 'major-issue',
        'minor': 'minor-issue',
        'good': 'good-technique'
    }

    icon = severity_icons.get(item.severity, '📊')
    css_class = severity_classes.get(item.severity, '')

    formatted_value = TechniqueBenchmarks.format_value(item.current_value, item.metric)
    min_val = TechniqueBenchmarks.format_value(item.target_range[0], item.metric)
    max_val = TechniqueBenchmarks.format_value(item.target_range[1], item.metric)

    st.markdown(f"""
    <div class="{css_class}">
        <strong>{icon} {item.message}</strong><br>
        Current: {formatted_value} | Target: {min_val} - {max_val}<br>
        → {item.impact}<br>
        🎯 Drill: {item.drill}
    </div>
    """, unsafe_allow_html=True)


def main():
    # Header
    st.markdown('<h1 class="main-header">🏸 AI Badminton Coach</h1>', unsafe_allow_html=True)

    st.markdown("""
    Upload a badminton stroke video or features file to receive professional coaching feedback
    on your technique across 8 biomechanical metrics.
    """)

    # Sidebar
    with st.sidebar:
        st.header("⚙️ Settings")

        mode = st.radio(
            "Analysis Mode",
            ["Upload Video", "Upload Features File", "Try Sample Clips"],
            help="Upload video, features, or test with sample clips"
        )

        stroke_type = st.selectbox(
            "Stroke Type",
            ["Auto-detect", "Clear", "Smash"],
            help="Select stroke type or let the system detect automatically"
        )

        st.markdown("---")
        st.markdown("### 📊 About")
        st.markdown("""
        This AI coaching system analyzes:
        - **Arm Extension**
        - **Velocity**
        - **Elbow Angle**
        - **Posture**
        - **Timing**
        - **Contact Point**
        - **Forearm Angle**
        - **Shoulder Rotation**

        Benchmarks derived from **3,347 professional strokes**.
        """)

    # Main content area
    features_data = None
    detected_stroke = None

    if mode == "Upload Video":
        st.header("🎥 Upload Video File")

        uploaded_video = st.file_uploader(
            "Upload your badminton stroke video",
            type=['mp4', 'avi', 'mov', 'mkv'],
            help="Upload a video showing one badminton stroke (Clear or Smash)"
        )

        if uploaded_video is not None:
            # Save uploaded video to temp file
            with tempfile.NamedTemporaryFile(delete=False, suffix=Path(uploaded_video.name).suffix) as tmp_file:
                tmp_file.write(uploaded_video.read())
                video_path = tmp_file.name

            st.success(f"✓ Uploaded video: {uploaded_video.name}")

            # Display video
            st.video(uploaded_video)

            # Determine stroke type for processing
            stroke_for_processing = None if stroke_type == "Auto-detect" else stroke_type

            if stroke_type == "Auto-detect":
                st.info("🤖 Stroke type will be auto-detected from video features")
            else:
                st.info(f"🎯 Stroke type: **{stroke_type}**")

            # Process video button
            if st.button("🔍 Process Video", type="primary", use_container_width=True):
                with st.spinner("Processing video... This may take 30-60 seconds."):
                    features_data = process_video(video_path, stroke_for_processing)

                    if features_data is not None:
                        quality = features_data.get('quality', {}).get('valid_percentage', 0)
                        detected_stroke = features_data.get('detected_stroke')

                        st.success(f"✓ Video processed successfully! Quality: {quality:.1f}% valid frames")
                        st.success(f"✓ Stroke type: **{detected_stroke}**")

                        # Store in session state
                        st.session_state['features_data'] = features_data
                        st.session_state['detected_stroke'] = detected_stroke
                        st.session_state['video_processed'] = True
                    else:
                        st.error("❌ Failed to process video. Check that player is visible.")

    # Check if video was processed from session state
    if st.session_state.get('video_processed', False) and features_data is None:
        features_data = st.session_state.get('features_data')
        detected_stroke = st.session_state.get('detected_stroke')
        st.info(f"✓ Video processed. Stroke type: **{detected_stroke}**")

    elif mode == "Upload Features File":
        st.header("📤 Upload Features File")

        uploaded_file = st.file_uploader(
            "Upload your .pkl features file",
            type=['pkl'],
            help="Upload a pre-extracted features file (.pkl format)"
        )

        if uploaded_file is not None:
            try:
                features_data = pickle.load(uploaded_file)
                st.success(f"✓ Loaded features from: {uploaded_file.name}")

                # Auto-detect stroke type
                if stroke_type == "Auto-detect":
                    if 'Clear' in uploaded_file.name:
                        detected_stroke = 'Clear'
                    elif 'Smash' in uploaded_file.name:
                        detected_stroke = 'Smash'
                    else:
                        stat_features = features_data.get('statistical_summary', features_data)
                        velocity = stat_features.get('max_velocity', 75)
                        detected_stroke = 'Smash' if velocity > 80 else 'Clear'
                    st.info(f"🎯 Auto-detected stroke type: **{detected_stroke}**")
                else:
                    detected_stroke = stroke_type

            except Exception as e:
                st.error(f"❌ Error loading file: {e}")
                st.stop()

    else:  # Try Sample Clips
        st.header("🎬 Sample Professional Clips")

        sample_options = {
            "Clear - Professional (Score: 60/100)": "data/processed/features/15_set1_rally22_ball21_Clear_features.pkl",
            "Smash - Professional (Score: 74/100)": "data/processed/features/44_set2_rally8_ball5_Smash_features.pkl"
        }

        selected_sample = st.selectbox(
            "Select a sample clip",
            list(sample_options.keys())
        )

        sample_path = Path(sample_options[selected_sample])

        if sample_path.exists():
            with open(sample_path, 'rb') as f:
                features_data = pickle.load(f)

            # Auto-detect from filename
            if 'Clear' in selected_sample:
                detected_stroke = 'Clear'
            elif 'Smash' in selected_sample:
                detected_stroke = 'Smash'
            else:
                detected_stroke = stroke_type if stroke_type != "Auto-detect" else 'Clear'

            st.success(f"✓ Loaded sample: {selected_sample}")
            st.info(f"🎯 Stroke type: **{detected_stroke}**")
        else:
            st.error(f"❌ Sample file not found: {sample_path}")
            st.info("Make sure you're running from the project root directory.")
            st.stop()

    # Analysis button
    if features_data is not None and detected_stroke is not None:
        st.markdown("---")

        if st.button("🔍 Analyze Technique", type="primary", use_container_width=True):
            with st.spinner("Analyzing technique... This may take a few seconds."):
                try:
                    # Run analysis
                    coach, output_dir = analyze_features(features_data, detected_stroke)

                    # Store results in session state
                    st.session_state['coach'] = coach
                    st.session_state['output_dir'] = output_dir
                    st.session_state['stroke_type'] = detected_stroke

                    st.success("✓ Analysis complete!")

                except Exception as e:
                    st.error(f"❌ Error during analysis: {e}")
                    import traceback
                    st.code(traceback.format_exc())
                    st.stop()

    # Display results if available
    if 'coach' in st.session_state:
        coach = st.session_state['coach']
        output_dir = st.session_state['output_dir']
        stroke_type = st.session_state['stroke_type']

        st.markdown("---")
        st.header("📊 Analysis Results")

        # Overall Score
        col1, col2, col3 = st.columns([2, 1, 1])

        with col1:
            st.markdown(f"""
            <div class="score-box">
                <h2>Overall Technique Score</h2>
                <h1 style="color: #2E86AB; font-size: 4rem; margin: 0;">{coach.overall_score}/100</h1>
            </div>
            """, unsafe_allow_html=True)

        with col2:
            # Grade
            if coach.overall_score >= 85:
                grade = "Excellent"
                color = "#2ecc71"
            elif coach.overall_score >= 70:
                grade = "Good"
                color = "#f1c40f"
            elif coach.overall_score >= 50:
                grade = "Needs Improvement"
                color = "#f39c12"
            else:
                grade = "Needs Attention"
                color = "#e74c3c"

            st.markdown(f"""
            <div style="text-align: center; padding-top: 1rem;">
                <h3>Grade</h3>
                <h2 style="color: {color};">{grade}</h2>
            </div>
            """, unsafe_allow_html=True)

        with col3:
            # Issue count
            critical = sum(1 for item in coach.feedback_items if item.severity == 'critical')
            major = sum(1 for item in coach.feedback_items if item.severity == 'major')
            minor = sum(1 for item in coach.feedback_items if item.severity == 'minor')
            good = sum(1 for item in coach.feedback_items if item.severity == 'good')

            st.markdown(f"""
            <div style="text-align: center; padding-top: 1rem;">
                <h3>Issues</h3>
                <p>🔴 Critical: {critical}<br>
                ⚠️ Major: {major}<br>
                💡 Minor: {minor}<br>
                ✅ Good: {good}</p>
            </div>
            """, unsafe_allow_html=True)

        # Tabs for different views
        tab1, tab2, tab3 = st.tabs(["📋 Detailed Feedback", "📊 Visualizations", "🎯 Priority Actions"])

        with tab1:
            st.subheader("Detailed Coaching Feedback")

            for i, item in enumerate(coach.feedback_items, 1):
                with st.expander(f"{i}. {item.message}", expanded=(item.severity in ['critical', 'major'])):
                    display_feedback_item(item)

        with tab2:
            st.subheader("Visual Analysis")

            # Display visualizations
            col1, col2 = st.columns(2)

            with col1:
                st.markdown("**Technique Radar**")
                radar_path = output_dir / f"radar_chart_{stroke_type.lower()}.png"
                if radar_path.exists():
                    st.image(str(radar_path), use_container_width=True)

                st.markdown("**Score Gauge**")
                gauge_path = output_dir / f"score_gauge_{stroke_type.lower()}.png"
                if gauge_path.exists():
                    st.image(str(gauge_path), use_container_width=True)

            with col2:
                st.markdown("**Metrics Breakdown**")
                bar_path = output_dir / f"metrics_bar_chart_{stroke_type.lower()}.png"
                if bar_path.exists():
                    st.image(str(bar_path), use_container_width=True)

            st.markdown("**Comprehensive Report**")
            report_path = output_dir / f"comprehensive_report_{stroke_type.lower()}.png"
            if report_path.exists():
                st.image(str(report_path), use_container_width=True)

                # Download button
                with open(report_path, 'rb') as f:
                    st.download_button(
                        label="📥 Download Report",
                        data=f.read(),
                        file_name=f"badminton_coaching_report_{stroke_type.lower()}.png",
                        mime="image/png",
                        use_container_width=True
                    )

        with tab3:
            st.subheader("Priority Actions")
            st.markdown("Focus on these areas for maximum improvement:")

            # Get priority items (exclude 'good')
            priority_items = [item for item in coach.feedback_items if item.severity != 'good'][:3]

            if priority_items:
                for i, item in enumerate(priority_items, 1):
                    st.markdown(f"### {i}. {item.message}")
                    st.markdown(f"**Current**: {TechniqueBenchmarks.format_value(item.current_value, item.metric)}")
                    st.markdown(f"**Target**: {TechniqueBenchmarks.format_value(item.target_range[0], item.metric)} - {TechniqueBenchmarks.format_value(item.target_range[1], item.metric)}")
                    st.markdown(f"**Impact**: {item.impact}")
                    st.markdown(f"**🎯 Drill**: {item.drill}")
                    st.markdown("---")
            else:
                st.success("✨ Great technique! Focus on maintaining consistency.")

        # Text feedback download
        st.markdown("---")
        feedback_text = coach.generate_summary()
        st.download_button(
            label="📄 Download Text Feedback",
            data=feedback_text,
            file_name=f"coaching_feedback_{stroke_type.lower()}.txt",
            mime="text/plain",
            use_container_width=True
        )


if __name__ == "__main__":
    main()
