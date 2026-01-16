# AI Badminton Coach v2.0

An AI-powered coaching system that analyzes badminton stroke technique using pose estimation and provides personalized feedback based on professional benchmarks.

---

## Quick Start

### 1. Installation

```bash
# Install dependencies
pip install -r requirements.txt

# Verify installation
python diagnose.py
```

### 2. Analyze a Video

```bash
python analyze_video.py your_video.mp4 Clear
# or
python analyze_video.py your_video.mp4 Smash
```

### 3. Web Interface

```bash
streamlit run src/deployment/streamlit_app.py
```

---

## Features

- **Pose Estimation**: MediaPipe-based biomechanical analysis
- **8 Key Metrics**: Arm extension, velocity, elbow angle, posture, timing, contact point, forearm angle, shoulder rotation
- **Professional Benchmarks**: Derived from 3,347 professional strokes
- **Personalized Feedback**: Severity-graded issues (critical/major/minor/good)
- **Practice Drills**: Specific exercises for each identified issue
- **Visualizations**: Radar charts, bar charts, score gauges, comprehensive reports

---

## Usage

### Command Line

**Analyze a video:**
```bash
python analyze_video.py video.mp4 Clear
```

**Check environment:**
```bash
python diagnose.py
```

### Web Interface

**Launch Streamlit app:**
```bash
streamlit run src/deployment/streamlit_app.py
```

Features:
- Upload video files or use sample clips
- Interactive visualizations
- Download comprehensive reports
- Three-tab view: Feedback / Visualizations / Priority Actions

---

## System Architecture

```
Video Input (.mp4, .avi, etc.)
    ↓
MediaPipe Pose Estimation (33 keypoints)
    ↓
Feature Engineering (427 statistical features)
    ↓
Technique Analysis (vs Professional Benchmarks)
    ↓
Coaching Feedback + Visualizations
```

### Components

1. **Pose Extraction** ([src/data_processing/extract_poses.py](src/data_processing/extract_poses.py))
   - MediaPipe Pose (Model Complexity: 2)
   - Multi-player detection with executor identification
   - Temporal interpolation for missing frames

2. **Feature Engineering** ([src/data_processing/feature_engineering_v2.py](src/data_processing/feature_engineering_v2.py))
   - 60 per-frame spatial features
   - Temporal features (velocity, acceleration, jerk)
   - 427 statistical aggregations

3. **Technique Benchmarks** ([src/coaching/technique_benchmarks.py](src/coaching/technique_benchmarks.py))
   - Professional ranges (25th-75th percentiles)
   - Separate benchmarks for Clear vs Smash
   - Based on 3,347 stroke dataset

4. **Coaching Feedback** ([src/coaching/feedback_generator.py](src/coaching/feedback_generator.py))
   - 6 analysis functions (arm, velocity, elbow, posture, timing, contact)
   - Severity classification
   - Specific practice drills

5. **Visualizations** ([src/coaching/visualizations.py](src/coaching/visualizations.py))
   - Radar charts (normalized technique profile)
   - Bar charts (metric-by-metric breakdown)
   - Score gauges (0-100 overall score)
   - Comprehensive reports (multi-panel view)

---

## Biomechanical Metrics

| Metric | Clear Target | Smash Target | Description |
|--------|--------------|--------------|-------------|
| **Arm Extension** | 0.062 - 0.121 | 0.077 - 0.146 | Racket-hand distance at contact |
| **Velocity** | 55.8 - 92.7 | 68.7 - 109.2 | Wrist movement speed |
| **Elbow Angle** | 116.9° - 141.6° | 121.3° - 148.0° | Joint angle at contact |
| **Posture** | -93.7 - 4.5 | -75.8 - 22.0 | Torso lean (forward/back) |
| **Timing** | 0.20 - 0.69 | 0.25 - 0.71 | Peak acceleration timing |
| **Contact Point** | -0.005 - 0.045 | -0.027 - 0.036 | Wrist height relative to head |
| **Forearm Angle** | 31.5° - 68.9° | 14.8° - 47.1° | Vertical orientation |
| **Shoulder Rotation** | -31.5° - 3.2° | -31.6° - 4.1° | Lateral rotation |

---

## Dataset

**ShuttleSet**: Professional badminton dataset
- **Citation**: Wang et al. (2023). "ShuttleSet: A Human-Annotated Stroke-Level Singles Dataset for Badminton Tactical Analysis"
- **Source**: [arXiv:2306.04948](https://arxiv.org/abs/2306.04948)
- **Clips Processed**: 4,983 professional strokes
- **Strokes Used for Benchmarks**: 3,347 (Clear + Smash only)
- **Players**: Professional singles matches

---

## Troubleshooting

### MediaPipe Mutex Error

**Symptom**: `[mutex.cc : 452] RAW: Lock blocking`

**Fix**:
```bash
pip uninstall protobuf -y
pip install protobuf==3.20.3
```

**Verify**:
```bash
python diagnose.py
```

### Video Processing Fails

1. **Check video format**: Supported formats: .mp4, .avi, .mov, .mkv
2. **Check player visibility**: Player must be visible throughout stroke
3. **Check video quality**: Higher resolution = better pose detection
4. **Run diagnostic**: `python diagnose.py`

### Dependencies

Required versions:
- Python: 3.8-3.10 (3.10 recommended)
- MediaPipe: 0.10.9
- Protobuf: 3.20.3 (critical!)
- OpenCV: 4.x
- NumPy, Pandas, Matplotlib, Streamlit

---

## Project Structure

```
iti123_v2/
├── analyze_video.py          # Main video analysis script
├── diagnose.py                # Environment diagnostic tool
├── requirements.txt           # Python dependencies
│
├── src/
│   ├── data_processing/
│   │   ├── extract_poses.py           # MediaPipe pose extraction
│   │   └── feature_engineering_v2.py   # Biomechanical features
│   │
│   ├── coaching/
│   │   ├── technique_benchmarks.py     # Professional ranges
│   │   ├── feedback_generator.py       # Coaching advice
│   │   └── visualizations.py           # Charts and reports
│   │
│   └── deployment/
│       └── streamlit_app.py            # Web interface
│
├── data/
│   └── processed/
│       ├── clips/              # Video clips (4,983)
│       └── features/           # Pre-extracted features
│
└── outputs/
    └── video_analysis/         # Analysis results
```

---

## Output Files

After running `analyze_video.py`, you'll get:

```
outputs/video_analysis/<video_name>/
├── feedback.txt                         # Text summary
├── comprehensive_report_<stroke>.png    # Multi-panel report
├── radar_chart_<stroke>.png             # Technique profile
├── metrics_bar_chart_<stroke>.png       # Metric breakdown
└── score_gauge_<stroke>.png             # Overall score
```

---

## Examples

### Example 1: Analyze Clear Stroke

```bash
python analyze_video.py my_clear.mp4 Clear
```

**Output**:
```
Overall Score: 74/100 - Good ⭐⭐

Issues Found:
  ⚠️  Major: 1
  💡 Minor: 2
  ✅ Good: 3

TOP PRIORITY ACTIONS
1. ⚠️ Velocity: Below target for Clear
   🎯 Resistance band training, 3 sets × 12 reps
```

### Example 2: Web Interface

```bash
streamlit run src/deployment/streamlit_app.py
```

1. Upload your video or select a sample
2. Choose stroke type (Clear/Smash)
3. Click "Analyze Technique"
4. View results in 3 tabs
5. Download comprehensive report

---

## Development

### Running Tests

```bash
# Verify complete system
python diagnose.py

# Test with sample clip
python analyze_video.py data/processed/clips/01_set1_rally1_ball2_Clear.mp4 Clear
```

### Extending the System

1. **Add New Metrics**: Edit [src/data_processing/feature_engineering_v2.py](src/data_processing/feature_engineering_v2.py)
2. **Modify Benchmarks**: Update [src/coaching/technique_benchmarks.py](src/coaching/technique_benchmarks.py)
3. **Customize Feedback**: Edit [src/coaching/feedback_generator.py](src/coaching/feedback_generator.py)
4. **Add Visualizations**: Extend [src/coaching/visualizations.py](src/coaching/visualizations.py)

---

## Citation

If you use this system in your research, please cite the underlying dataset:

```bibtex
@article{wang2023shuttleset,
  title={ShuttleSet: A Human-Annotated Stroke-Level Singles Dataset for Badminton Tactical Analysis},
  author={Wang, Wei-Yao and Huang, Yu-Chuan and Ik, Tsi-Ui and Peng, Wen-Chih},
  journal={arXiv preprint arXiv:2306.04948},
  year={2023}
}
```

---

## License

This project uses the ShuttleSet dataset. Please refer to the original dataset's license terms.

---

## Contact

For questions or issues, please open an issue on the project repository.

---

**Last Updated**: January 16, 2026
**Version**: 2.0
**Status**: Production Ready ✅
