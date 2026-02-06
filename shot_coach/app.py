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
        1. Record a 2-3 second video of any badminton shot
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
        - Alternative possibilities

        **Video Requirements:**
        - Clear view of the shot execution
        - 2-3 seconds duration
        - MP4, AVI, or MOV format
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
