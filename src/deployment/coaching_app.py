"""
AI Badminton Coach - Enhanced Gradio Interface

A comprehensive web application that analyzes badminton stroke videos and provides:
- Stroke classification (Clear vs Smash)
- Technique analysis across 8 biomechanical metrics
- Personalized coaching feedback with practice drills
- Visual reports (radar charts, bar charts, score gauges)
- Overall technique score (0-100)
"""

import gradio as gr
import pickle
import cv2
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import sys
import tempfile
import shutil

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).resolve().parents[2]))

from src.coaching import TechniqueBenchmarks, CoachingFeedback, TechniqueVisualizer
from src.data_processing.extract_poses import PoseExtractor
from src.data_processing.feature_engineering_v2 import FeatureEngineering
from src.models.lstm_model import load_trained_model


class BadmintonCoachingApp:
    """Main application class for AI Badminton Coach"""

    def __init__(self, model_path: str = "outputs/models/bilstm_model.pth"):
        """
        Initialize the coaching app

        Args:
            model_path: Path to trained model weights
        """
        self.model_path = Path(model_path)
        self.pose_extractor = PoseExtractor()
        self.feature_engineer = FeatureEngineering()
        self.visualizer = TechniqueVisualizer()

        # Load model if exists
        self.model = None
        self.model_loaded = False
        if self.model_path.exists():
            try:
                self.model = load_trained_model(str(self.model_path))
                self.model_loaded = True
                print(f"✓ Model loaded from {self.model_path}")
            except Exception as e:
                print(f"⚠️  Could not load model: {e}")
                print("   Classification will use fallback method")

    def process_video(self, video_path: str, stroke_type_override: str = "Auto-detect"):
        """
        Process uploaded video and generate coaching feedback

        Args:
            video_path: Path to uploaded video file
            stroke_type_override: Manual stroke type or "Auto-detect"

        Returns:
            Tuple of (classification, score, feedback_text, radar_chart, bar_chart, gauge_chart, report)
        """
        try:
            # Step 1: Extract poses
            print(f"Processing video: {video_path}")
            poses = self.pose_extractor.extract_from_video(video_path)

            if poses is None or len(poses) == 0:
                return self._error_output("Failed to extract poses from video. Please ensure the video shows a clear badminton stroke.")

            # Step 2: Extract features
            print(f"Extracted {len(poses)} frames, extracting features...")
            features = self.feature_engineer.extract_features(poses)

            if features is None:
                return self._error_output("Failed to extract features from poses.")

            sequence_features = features['sequence_features']
            stat_features = features['statistical_summary']

            # Step 3: Classify stroke (if not overridden)
            if stroke_type_override == "Auto-detect":
                if self.model_loaded:
                    stroke_type, confidence = self._classify_with_model(sequence_features)
                    classification_text = f"**Detected Stroke**: {stroke_type} ({confidence:.1%} confidence)"
                else:
                    # Fallback: Use velocity heuristic
                    stroke_type = self._classify_fallback(stat_features)
                    classification_text = f"**Detected Stroke**: {stroke_type} (heuristic-based)"
            else:
                stroke_type = stroke_type_override
                classification_text = f"**Stroke Type**: {stroke_type} (manual)"

            # Step 4: Generate coaching feedback
            print(f"Generating feedback for {stroke_type}...")
            coach = CoachingFeedback()
            feedback_items = coach.analyze_technique(stat_features, stroke_type)
            overall_score = coach.overall_score

            # Step 5: Generate text feedback
            feedback_text = coach.generate_summary()

            # Step 6: Create visualizations
            print("Creating visualizations...")

            # Create temporary directory for charts
            temp_dir = Path(tempfile.mkdtemp())
            self.visualizer.output_dir = temp_dir

            # Generate all charts
            radar_fig = self.visualizer.create_radar_chart(stat_features, stroke_type)
            bar_fig = self.visualizer.create_metrics_bar_chart(feedback_items, stroke_type)
            gauge_fig = self.visualizer.create_score_gauge(overall_score, stroke_type)
            report_fig = self.visualizer.create_comprehensive_report(stat_features, stroke_type, coach)

            # Convert to image paths for Gradio
            radar_path = str(temp_dir / 'radar.png')
            bar_path = str(temp_dir / 'bar.png')
            gauge_path = str(temp_dir / 'gauge.png')
            report_path = str(temp_dir / 'report.png')

            radar_fig.savefig(radar_path, dpi=150, bbox_inches='tight')
            bar_fig.savefig(bar_path, dpi=150, bbox_inches='tight')
            gauge_fig.savefig(gauge_path, dpi=150, bbox_inches='tight')
            report_fig.savefig(report_path, dpi=150, bbox_inches='tight')

            # Close figures to free memory
            plt.close('all')

            print("✓ Analysis complete!")

            return (
                classification_text,
                f"**Overall Technique Score**: {overall_score}/100",
                feedback_text,
                radar_path,
                bar_path,
                gauge_path,
                report_path
            )

        except Exception as e:
            import traceback
            error_msg = f"Error processing video: {str(e)}\n\n{traceback.format_exc()}"
            print(error_msg)
            return self._error_output(error_msg)

    def _classify_with_model(self, sequence_features):
        """Classify stroke using trained model"""
        try:
            import torch

            # Prepare input
            x = torch.FloatTensor(sequence_features).unsqueeze(0)  # Add batch dimension

            # Get prediction
            self.model.eval()
            with torch.no_grad():
                output = self.model(x)
                probabilities = torch.softmax(output, dim=1)
                pred_class = torch.argmax(probabilities, dim=1).item()
                confidence = probabilities[0][pred_class].item()

            # Map class index to stroke type
            stroke_type = 'Clear' if pred_class == 0 else 'Smash'

            return stroke_type, confidence

        except Exception as e:
            print(f"Model classification failed: {e}")
            # Fallback
            return self._classify_fallback({'max_velocity': 75}), 0.5

    def _classify_fallback(self, stat_features):
        """Simple heuristic-based classification"""
        # Use velocity as heuristic (Clear typically higher arc, Smash more downward)
        # This is very approximate - better to use model
        velocity = stat_features.get('max_velocity', 75)

        # Simple threshold (this is a placeholder - model is better)
        if velocity > 80:
            return 'Smash'
        else:
            return 'Clear'

    def _error_output(self, error_msg: str):
        """Return error state for all outputs"""
        return (
            f"**Error**: Classification failed",
            f"**Score**: N/A",
            f"ERROR:\n{error_msg}",
            None,  # radar
            None,  # bar
            None,  # gauge
            None   # report
        )

    def analyze_sample(self, sample_name: str):
        """
        Analyze a sample clip from the dataset

        Args:
            sample_name: Name of sample clip

        Returns:
            Same as process_video()
        """
        # Map sample names to actual files
        samples = {
            "Clear - Professional": "data/processed/clips/01_set1_rally1_ball2_Clear.mp4",
            "Smash - Professional": "data/processed/clips/01_set1_rally2_ball3_Smash.mp4",
        }

        video_path = samples.get(sample_name)
        if video_path and Path(video_path).exists():
            # Extract stroke type from filename
            stroke_type = "Clear" if "Clear" in sample_name else "Smash"
            return self.process_video(video_path, stroke_type_override=stroke_type)
        else:
            return self._error_output(f"Sample clip not found: {video_path}")


def create_interface():
    """Create Gradio interface"""

    # Initialize app
    app = BadmintonCoachingApp()

    # Custom CSS for better styling
    custom_css = """
    .output-class {font-size: 1.2em; font-weight: bold;}
    .score-class {font-size: 1.5em; color: #2ecc71; font-weight: bold;}
    .feedback-class {font-family: monospace; font-size: 0.9em;}
    """

    # Create interface
    with gr.Blocks(css=custom_css, title="AI Badminton Coach") as demo:

        gr.Markdown("""
        # 🏸 AI Badminton Coach

        Upload a badminton stroke video to receive professional coaching feedback on your technique.

        **What you'll get**:
        - Stroke classification (Clear vs Smash)
        - Overall technique score (0-100)
        - Detailed feedback on 8 biomechanical metrics
        - Personalized practice drills
        - Visual analysis charts
        """)

        with gr.Tab("Upload Video"):
            with gr.Row():
                with gr.Column(scale=1):
                    video_input = gr.Video(label="Upload Your Stroke Video")

                    stroke_override = gr.Radio(
                        choices=["Auto-detect", "Clear", "Smash"],
                        value="Auto-detect",
                        label="Stroke Type",
                        info="Let AI detect or specify manually"
                    )

                    analyze_btn = gr.Button("Analyze Technique", variant="primary", size="lg")

                    gr.Markdown("""
                    ### Tips for best results:
                    - Video should show full stroke from preparation to follow-through
                    - Player should be clearly visible (side or back view preferred)
                    - 2-5 seconds duration
                    - Good lighting, minimal background clutter
                    """)

                with gr.Column(scale=1):
                    classification_output = gr.Markdown(label="Classification", elem_classes="output-class")
                    score_output = gr.Markdown(label="Overall Score", elem_classes="score-class")

            with gr.Row():
                with gr.Column():
                    feedback_output = gr.Textbox(
                        label="Detailed Coaching Feedback",
                        lines=20,
                        elem_classes="feedback-class"
                    )

            gr.Markdown("## Visual Analysis")

            with gr.Row():
                radar_output = gr.Image(label="Technique Radar", type="filepath")
                bar_output = gr.Image(label="Metrics Breakdown", type="filepath")
                gauge_output = gr.Image(label="Score Gauge", type="filepath")

            with gr.Row():
                report_output = gr.Image(label="Comprehensive Report", type="filepath")

        with gr.Tab("Try Sample Clips"):
            gr.Markdown("""
            ## Try with Professional Sample Clips

            Test the coaching system with professional badminton strokes from the ShuttleSet dataset.
            """)

            sample_selector = gr.Radio(
                choices=[
                    "Clear - Professional",
                    "Smash - Professional"
                ],
                label="Select Sample Clip",
                value="Clear - Professional"
            )

            sample_btn = gr.Button("Analyze Sample", variant="primary")

            with gr.Row():
                sample_classification = gr.Markdown(elem_classes="output-class")
                sample_score = gr.Markdown(elem_classes="score-class")

            with gr.Row():
                sample_feedback = gr.Textbox(label="Feedback", lines=20, elem_classes="feedback-class")

            with gr.Row():
                sample_radar = gr.Image(label="Technique Radar")
                sample_bar = gr.Image(label="Metrics Breakdown")
                sample_gauge = gr.Image(label="Score Gauge")

            with gr.Row():
                sample_report = gr.Image(label="Comprehensive Report")

        with gr.Tab("About"):
            gr.Markdown("""
            ## About AI Badminton Coach

            This application uses computer vision and machine learning to analyze badminton technique:

            ### How it works:
            1. **Pose Extraction**: MediaPipe detects 33 body keypoints per frame
            2. **Feature Engineering**: Calculates 60 sequence + 427 statistical features
            3. **Classification**: BiLSTM model predicts stroke type (Clear vs Smash)
            4. **Biomechanical Analysis**: Compares your technique to professional benchmarks
            5. **Feedback Generation**: Provides actionable coaching advice with practice drills
            6. **Visualization**: Creates professional-quality analysis charts

            ### Metrics Analyzed:
            - **Arm Extension**: Reach at contact point
            - **Velocity**: Racket head speed (via wrist tracking)
            - **Elbow Angle**: Joint positioning for power transfer
            - **Posture**: Body lean and balance
            - **Timing**: Peak velocity timing in stroke
            - **Contact Point**: Wrist height relative to shoulder
            - **Forearm Angle**: Vertical orientation
            - **Shoulder Rotation**: Frontal plane rotation

            ### Professional Benchmarks:
            - Derived from **3,347 professional strokes** (ShuttleSet dataset)
            - 1,781 Clear + 1,566 Smash strokes
            - Top players: Kento Momota, Chou Tien Chen, etc.
            - Ranges represent middle 50% (25th-75th percentile) of professionals

            ### Technology Stack:
            - **Pose Estimation**: MediaPipe Pose
            - **Classification**: PyTorch BiLSTM
            - **Visualization**: Matplotlib
            - **Interface**: Gradio
            - **Dataset**: ShuttleSet (Wang et al., 2023)

            ---

            **Version**: 2.0
            **Course**: ITI123 - Introduction to AI
            **Milestone Report**: January 29, 2026
            """)

        # Connect buttons to functions
        analyze_btn.click(
            fn=app.process_video,
            inputs=[video_input, stroke_override],
            outputs=[
                classification_output,
                score_output,
                feedback_output,
                radar_output,
                bar_output,
                gauge_output,
                report_output
            ]
        )

        sample_btn.click(
            fn=app.analyze_sample,
            inputs=[sample_selector],
            outputs=[
                sample_classification,
                sample_score,
                sample_feedback,
                sample_radar,
                sample_bar,
                sample_gauge,
                sample_report
            ]
        )

    return demo


if __name__ == "__main__":
    print("="*70)
    print("AI BADMINTON COACH - STARTING APPLICATION")
    print("="*70)
    print()

    # Create and launch interface
    demo = create_interface()

    print("✓ Interface created")
    print()
    print("Launching Gradio app...")
    print("Access at: http://localhost:7860")
    print()

    demo.launch(
        share=False,  # Set to True to create public link
        server_name="0.0.0.0",
        server_port=7860,
        show_error=True
    )
