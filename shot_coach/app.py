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


# Page config
st.set_page_config(
    page_title="Badminton Shot Classifier",
    page_icon="🏸",
    layout="wide"
)


@st.cache_resource
def load_classifier():
    """Load shot classifier once and cache"""
    # Path to trained model
    model_path = Path(__file__).parent.parent / 'outputs' / 'results_optionA' / 'best_model.pth'

    if not model_path.exists():
        st.error(f"Model not found at: {model_path}")
        st.stop()

    classifier = ShotClassifier(model_path)
    return classifier




def main():
    # Header
    st.title("🏸 Badminton Shot Classifier")
    st.markdown("### AI-Powered Shot Type Recognition")
    st.markdown("Upload a video of any badminton shot to instantly identify the shot type!")

    st.markdown("---")

    # Load classifier
    with st.spinner("Loading AI model..."):
        try:
            classifier = load_classifier()
            st.success("✅ Model loaded successfully!")
        except Exception as e:
            st.error(f"Error loading model: {e}")
            st.stop()

    st.markdown("---")

    # Instructions
    with st.expander("📖 How to Use", expanded=False):
        st.markdown("""
        **Instructions:**
        1. Record a 2-3 second video of **ONE** badminton shot
        2. Upload the video using the button below
        3. Wait for classification (takes 5-10 seconds)
        4. See which shot type was detected!

        **Supported Shot Types (5 classes):**
        - 🎯 **Clear** - High defensive shot to the back
        - ⚡ **Drive** - Fast, flat attacking shot
        - 🪶 **Drop** - Soft shot over the net
        - 📈 **Lift** - Defensive lob to the back
        - 💥 **Smash** - Powerful downward attack

        **What Gets Analyzed:**
        - Shot type classification (74.6% accuracy)
        - Confidence scores for all shot types
        - Video timing and frame analysis info
        - Alternative possibilities

        **Video Requirements:**
        - ⚠️ **ONE shot per video** (model samples frames across entire video)
        - Clear view of the shot execution (side-court angle works best)
        - 2-3 seconds duration (prep → contact → follow-through)
        - Full body visible in frame
        - Good lighting
        - MP4, AVI, or MOV format

        **Important Notes:**
        - Videos with multiple shots will produce confused/incorrect results
        - The app will warn you if your video is too long or confidence is low
        - Model trained on side-court broadcast angles - other angles may reduce accuracy
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
                metadata = classification_result.get('metadata', {})

                progress_bar.progress(40)

                progress_bar.progress(100)
                status_text.text("✅ Classification complete!")

                # Show all probabilities
                probabilities = classification_result['probabilities']

                # Display results
                st.markdown("---")
                st.markdown("## 🎯 Shot Classification Results")

                # Main result - large display
                st.markdown(f"### Detected Shot: **{shot_type}**")

                # Confidence meter
                col1, col2, col3 = st.columns([1, 2, 1])
                with col2:
                    st.metric("Confidence", f"{confidence*100:.1f}%", delta=None)
                    st.progress(confidence)

                # Video metadata and warnings
                if metadata:
                    st.markdown("---")
                    st.markdown("### ⏱️ Video Analysis Info")

                    duration = metadata.get('duration', 0)
                    fps = metadata.get('fps', 0)
                    total_frames = metadata.get('total_frames', 0)
                    num_analyzed = metadata.get('num_frames_analyzed', 16)

                    # Display metadata in columns
                    meta_col1, meta_col2, meta_col3 = st.columns(3)
                    with meta_col1:
                        st.metric("Video Duration", f"{duration:.2f}s")
                    with meta_col2:
                        st.metric("Total Frames", f"{total_frames}")
                    with meta_col3:
                        st.metric("Frames Analyzed", f"{num_analyzed}")

                    # Show frame sampling info
                    if 'sampled_times' in metadata and 'sampled_frames' in metadata:
                        sampled_times = metadata['sampled_times']
                        sampled_frames = metadata['sampled_frames']
                        time_range = f"{sampled_times[0]:.2f}s - {sampled_times[-1]:.2f}s"

                        st.caption(f"**Analyzed timeframe:** {time_range} (frames sampled uniformly across video)")

                        # Show visual timeline
                        with st.expander("🎬 See Frame Sampling Timeline", expanded=False):
                            st.markdown("**Frames analyzed by the model:**")

                            # Show frame numbers and times
                            st.markdown(f"**Total frames in video:** {total_frames}")
                            st.markdown(f"**Frames sampled for analysis:** {num_analyzed}")
                            st.markdown(f"**Sampling method:** Uniform distribution across entire video")

                            # Display sampled frame details in a more readable format
                            st.markdown("**Sample points:**")
                            sample_display = []
                            for i in range(0, len(sampled_frames), 4):  # Show 4 per row
                                batch = []
                                for j in range(i, min(i + 4, len(sampled_frames))):
                                    frame_num = int(sampled_frames[j])
                                    time_sec = sampled_times[j]
                                    batch.append(f"Frame {frame_num} ({time_sec:.2f}s)")
                                sample_display.append(" | ".join(batch))

                            for row in sample_display:
                                st.text(row)

                            st.caption(f"📊 The model analyzed **{num_analyzed} frames** distributed evenly across your **{duration:.2f}s** video.")

                            if duration > 3:
                                st.info("""
                                💡 **Why uniform sampling matters:**

                                The model samples frames evenly across your entire video. If your video contains:
                                - **One shot (2-3s)**: ✅ All frames capture that shot = Good
                                - **Multiple shots (5-10s)**: ❌ Frames capture different shots = Confused prediction

                                **Solution:** Trim your video to show only one complete shot execution.
                                """)

                    # Warnings based on video characteristics
                    if duration > 5:
                        st.warning("""
                        ⚠️ **Video is longer than 5 seconds**

                        Your video contains multiple shots or extended footage. The model samples frames uniformly
                        across the **entire video**, which may result in:
                        - Mixed signals from different shots
                        - Lower confidence scores
                        - Potentially incorrect classification

                        **Recommendation:** Trim your video to contain **only one shot** (2-3 seconds) for best results.
                        """)
                    elif duration < 1.5:
                        st.info("""
                        ℹ️ **Video is shorter than 1.5 seconds**

                        Short videos may not capture the full shot motion. For best results, record 2-3 seconds
                        including the preparation, contact, and follow-through.
                        """)

                    # Confidence-based warnings
                    if confidence < 0.70 and duration > 3:
                        st.warning("""
                        ⚠️ **Low confidence detected with long video**

                        The low confidence score combined with a longer video suggests the footage may contain
                        multiple shots or unclear motion. Consider:
                        - Trimming to a single shot
                        - Re-recording with clearer shot execution
                        - Using a side-court camera angle
                        """)
                    elif confidence < 0.60:
                        st.warning("""
                        ⚠️ **Low confidence score**

                        The model is uncertain about this classification. This may be due to:
                        - Video from an unusual camera angle (model trained on side-court views)
                        - Unclear or partial shot execution
                        - Multiple shots in one video
                        - Poor lighting or video quality

                        **Tips for better results:**
                        - Record from side-court view (like broadcast angle)
                        - Ensure full body is visible
                        - Use good lighting
                        - Trim to 2-3 seconds with one clear shot
                        """)
                    elif confidence > 0.85:
                        st.success("✅ High confidence - the video angle and shot execution match the training data well!")


                # Show all class probabilities
                st.markdown("---")
                st.markdown("### 📊 All Shot Type Probabilities")

                # Sort by probability
                sorted_probs = sorted(probabilities.items(), key=lambda x: x[1], reverse=True)

                for shot_name, prob in sorted_probs:
                    col1, col2 = st.columns([3, 1])
                    with col1:
                        # Highlight the predicted class
                        if shot_name == shot_type:
                            st.markdown(f"**🎯 {shot_name}** (Predicted)")
                        else:
                            st.markdown(f"{shot_name}")
                    with col2:
                        st.progress(prob, text=f"{prob*100:.1f}%")

                # Shot descriptions
                st.markdown("---")
                st.markdown("### 📖 Shot Type Information")

                shot_descriptions = {
                    'Clear': "🎯 **Clear** - High defensive shot to the back of the court. Used to push opponent back and buy time.",
                    'Drive': "⚡ **Drive** - Fast, flat shot parallel to the ground. Used for quick attacks and keeping opponent under pressure.",
                    'Drop': "🪶 **Drop** - Soft shot that barely clears the net. Used to make opponent move forward quickly.",
                    'Lift': "📈 **Lift** - Defensive shot hit high to the back. Similar to Clear but usually from a defensive position.",
                    'Smash': "💥 **Smash** - Powerful attacking shot hit downward. The primary attacking shot in badminton."
                }

                if shot_type in shot_descriptions:
                    st.info(shot_descriptions[shot_type])

                # Show alternatives
                st.markdown("---")
                st.markdown("### 💡 Alternative Possibilities")

                # Get top 3 alternatives
                alternatives = [s for s in sorted_probs if s[0] != shot_type][:3]

                if alternatives:
                    for shot_name, prob in alternatives:
                        if prob > 0.1:  # Only show if probability > 10%
                            st.markdown(f"- **{shot_name}**: {prob*100:.1f}% probability")
                            if shot_name in shot_descriptions:
                                st.caption(shot_descriptions[shot_name])

                # Model info
                st.markdown("---")
                st.markdown("### ℹ️ Model Information")
                st.markdown(f"""
                - **Model**: ResNet18 + BiLSTM
                - **Accuracy**: 74.6% (on test set)
                - **Classes**: 5 badminton shot types
                - **Training**: 22,302 video samples
                """)

                # Note about technique analysis
                st.markdown("---")
                st.info("""
                **Note**: Technique analysis (biomechanics, form scoring) is currently disabled.
                This version focuses on accurate shot classification only.
                """)

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
    <p>Badminton Shot Classifier v1.0 | Powered by AI</p>
    <p>Built with ResNet18+BiLSTM | 74.6% Test Accuracy</p>
    </div>
    """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()
