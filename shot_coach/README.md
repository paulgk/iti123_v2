# 🏸 Badminton Shot Coach

AI-powered badminton shot technique analyzer. Get instant feedback on your Smash and Clear techniques!

## Features

- **Shot Type Detection**: Automatically classifies your shot (Smash, Clear, etc.)
- **Technique Analysis**: Analyzes 5-7 key metrics per shot
- **Overall Score**: Get a 0-100 score for your technique
- **Specific Feedback**: Receive actionable recommendations to improve
- **Easy to Use**: Simple web interface - just upload a video!

## Supported Shot Types

Currently supports:
- ✅ **Smash** - Overhead attacking shot
- ✅ **Clear** - High defensive shot

Coming soon:
- 🔜 Drive
- 🔜 Drop
- 🔜 Lift

## How It Works

1. **Upload Video**: 2-3 second clip of your shot
2. **AI Classification**: Uses trained ResNet18+BiLSTM model (74.6% accuracy)
3. **Pose Extraction**: MediaPipe extracts body keypoints
4. **Technique Analysis**: Rule-based analysis of biomechanics
5. **Get Feedback**: Instant report with scores and recommendations

## Installation

### Prerequisites

- Python 3.8 or higher
- pip

### Setup

1. Clone the repository and navigate to shot_coach:

```bash
cd shot_coach
```

2. Install dependencies:

```bash
pip install -r requirements.txt
```

3. Verify model exists:

Make sure the trained model is at:
```
../outputs/results_optionA/best_model.pth
```

## Usage

### Run the Streamlit App

```bash
streamlit run app.py
```

This will:
- Start a local web server (usually http://localhost:8501)
- Open the app in your browser automatically

### Using the App

1. **Upload Video**:
   - Click "Browse files" or drag-and-drop
   - Supported formats: MP4, AVI, MOV
   - Video should be 2-3 seconds long
   - Make sure you're fully visible in frame

2. **Analyze**:
   - Click "🚀 Analyze Shot" button
   - Wait 10-30 seconds for analysis

3. **Review Results**:
   - Shot classification with confidence
   - Overall technique score (0-100)
   - Detailed metric breakdown
   - Specific improvement recommendations

4. **Download Report**:
   - Click "Download Text Report" for full analysis

## Metrics Analyzed

### Smash Technique

1. **Arm Extension** - How fully extended is your arm at contact?
2. **Shoulder Rotation** - How much do you rotate through the shot?
3. **Knee Bend** - Are you loading power properly?
4. **Contact Height** - How high above your head is contact?
5. **Follow-through** - Do you complete the swing?

### Clear Technique

1. **Elbow Height** - Is your elbow high enough before contact?
2. **Arm Extension** - Full arm extension at contact?
3. **Swing Arc** - Complete swing motion?
4. **Contact Height** - Appropriate height for Clear?
5. **Follow-through** - Complete follow-through motion?

## Example Report

```
======================================================================
BADMINTON SHOT COACH - ANALYSIS REPORT
======================================================================

SHOT CLASSIFICATION
----------------------------------------------------------------------
Shot Type: Smash
Confidence: 94.3%

TECHNIQUE ANALYSIS
----------------------------------------------------------------------
Overall Score: 72.0/100

OVERALL ASSESSMENT
----------------------------------------------------------------------
Rating: Good
Solid Smash technique with room for minor improvements.

AREAS TO IMPROVE
----------------------------------------------------------------------
  • Rotate shoulders more (current: 95°, target: 120°+)
  • Bend knees deeper before jump (current: 25°, target: 30-45°)

DETAILED METRICS
----------------------------------------------------------------------
Arm Extension        :      89% (target:      95%) - Score:  85.0/100 ✅ Excellent
Shoulder Rotation    :     95.0° (target:     120°) - Score:  79.2/100 ⚠️  Good
Knee Bend            :     25.0° (target:      35°) - Score:  60.0/100 ❌ Needs Work
Contact Height       :      32% (target:      30%) - Score: 100.0/100 ✅ Excellent
Follow-Through       :     88.0° (target:      90°) - Score:  94.0/100 ✅ Excellent

======================================================================
```

## Technical Details

### Architecture

```
Video Input (2-3 seconds)
    ↓
Shot Classifier (ResNet18+BiLSTM)
    → Detects shot type
    ↓
Pose Extractor (MediaPipe)
    → Extracts body keypoints
    ↓
Technique Analyzer (Rule-based)
    → Calculates metrics
    → Generates scores
    → Produces feedback
    ↓
Report (Text output)
```

### Models Used

1. **Shot Classification**: ResNet18 + BiLSTM
   - Trained on 22,302 badminton shot videos
   - 74.6% test accuracy across 5 classes
   - 16 frames per video, 224×224 resolution

2. **Pose Extraction**: Google MediaPipe Pose
   - 33 body keypoints per frame
   - Real-time performance
   - Robust to lighting and background

3. **Technique Analysis**: Rule-based biomechanics
   - Calculates angles, extensions, rotations
   - Compares to optimal values
   - Generates actionable feedback

## Project Structure

```
shot_coach/
├── app.py                      # Streamlit web interface
├── modules/
│   ├── shot_classifier.py      # Shot type classification
│   ├── pose_extractor.py       # Pose keypoint extraction
│   └── technique_analyzer.py   # Technique metrics & scoring
├── outputs/                    # Example outputs (created on first run)
├── requirements.txt            # Python dependencies
└── README.md                   # This file
```

## Troubleshooting

### Model Not Found

**Error**: "Model not found at: ../outputs/results_optionA/best_model.pth"

**Solution**: Make sure you've trained the model and it exists at the correct path. If you moved the shot_coach folder, update the path in `app.py`:

```python
model_path = Path(__file__).parent.parent / 'outputs' / 'results_optionA' / 'best_model.pth'
```

### No Pose Detected

**Error**: "No pose detected in any frame"

**Causes**:
- Person not fully visible in frame
- Poor lighting
- Camera too far away
- Low video quality

**Solution**:
- Re-record video with better lighting
- Stand closer to camera
- Ensure full body is visible

### Low Confidence Score

If shot classification confidence is low (< 70%):
- Video might be too short or too long
- Shot might be partially visible
- Movement might be ambiguous
- Try re-recording with clearer shot execution

## Limitations

- Currently only supports Smash and Clear (other shots coming soon)
- Requires clear view of the player
- Best with full body visible in frame
- Lighting affects pose detection quality
- 2D analysis only (no depth information)

## Future Improvements

- [ ] Support for Drive, Drop, Lift shots
- [ ] Visual overlays on video (skeleton, annotations)
- [ ] Comparison with professional players
- [ ] Shot quality prediction (success rate)
- [ ] Mobile app version
- [ ] Real-time analysis (live camera feed)
- [ ] Multi-angle analysis
- [ ] Progress tracking over time

## Performance

- **Shot Classification**: ~2-5 seconds
- **Pose Extraction**: ~5-10 seconds
- **Technique Analysis**: <1 second
- **Total**: ~10-30 seconds per video

## Contributing

Built as part of the ITI123 badminton action recognition project.

## License

MIT License

## Acknowledgments

- Trained model based on ShuttleSet dataset
- Pose detection powered by Google MediaPipe
- Deep learning with PyTorch
- Web interface with Streamlit

---

**Happy practicing! 🏸**
