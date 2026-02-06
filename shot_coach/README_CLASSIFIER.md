# 🏸 Badminton Shot Classifier

AI-powered badminton shot type recognition. Upload a video and instantly identify which shot type was executed!

## What It Does

**Simple and focused:** Classifies badminton shots into 5 types with 74.6% accuracy.

### Supported Shot Types

1. **Clear** 🎯 - High defensive shot to the back of the court
2. **Drive** ⚡ - Fast, flat shot parallel to the ground
3. **Drop** 🪶 - Soft shot that barely clears the net
4. **Lift** 📈 - Defensive shot hit high to the back
5. **Smash** 💥 - Powerful attacking shot hit downward

## Quick Start

```bash
# Install
cd shot_coach
pip install streamlit torch torchvision opencv-python numpy Pillow

# Run
python -m streamlit run app.py
```

## How It Works

1. **Upload Video** (2-3 seconds, any shot)
2. **AI Classification** (ResNet18 + BiLSTM model)
3. **Get Results**:
   - Predicted shot type
   - Confidence score
   - Probabilities for all 5 classes
   - Shot descriptions

## Features

### ✅ What You Get

- **Shot Type Classification** - Identifies which of 5 shot types
- **Confidence Score** - How confident the model is (0-100%)
- **All Probabilities** - See scores for all 5 shot types
- **Shot Information** - Description of each shot type
- **Alternative Possibilities** - What else it might be

### 🎯 Model Performance

- **Accuracy**: 74.6% on test set
- **Training Data**: 22,302 video samples
- **Architecture**: ResNet18 (CNN) + BiLSTM (temporal)
- **Input**: 16 frames per video, 224×224 resolution

## Video Requirements

- **Duration**: 2-3 seconds
- **Format**: MP4, AVI, or MOV
- **Quality**: Any (doesn't need perfect lighting)
- **Content**: Clear view of the shot execution

## Example Output

```
🎯 Detected Shot: Smash

Confidence: 94.3%
████████████████████████ 94.3%

All Shot Type Probabilities:
🎯 Smash (Predicted)     ████████████████████████ 94.3%
Clear                    ████░░░░░░░░░░░░░░░░░░░  4.2%
Drive                    ░░░░░░░░░░░░░░░░░░░░░░░  1.1%
Drop                     ░░░░░░░░░░░░░░░░░░░░░░░  0.3%
Lift                     ░░░░░░░░░░░░░░░░░░░░░░░  0.1%

💥 Smash - Powerful attacking shot hit downward.
The primary attacking shot in badminton.
```

## Technical Details

### Architecture

```
Video (2-3 seconds)
    ↓
Frame Extraction (16 frames)
    ↓
ResNet18 CNN
    → Extract spatial features per frame
    ↓
BiLSTM
    → Learn temporal patterns
    ↓
Classification (5 classes)
```

### Model Specs

- **Backbone**: ResNet18 (pretrained on ImageNet)
- **Temporal**: Bidirectional LSTM (2 layers, 256 hidden)
- **Parameters**: ~14M
- **Training**: 44 epochs with early stopping
- **Best Val Accuracy**: 75.4%
- **Test Accuracy**: 74.6%

### Performance by Class

| Shot Type | Precision | Recall | F1-Score |
|-----------|-----------|--------|----------|
| Clear     | 67.6%     | 89.2%  | 76.9%    |
| Drive     | 58.3%     | 61.6%  | 59.9%    |
| Drop      | 90.7%     | 74.6%  | 81.9%    |
| Lift      | 71.0%     | 75.7%  | 73.3%    |
| Smash     | 76.5%     | 75.8%  | 76.1%    |

**Best at:** Drop (90.7% precision) and Clear (89.2% recall)
**Challenging:** Drive vs Lift confusion (similar trajectories)

## Installation

### Requirements

```bash
pip install streamlit torch torchvision opencv-python numpy Pillow
```

### Full Setup

```bash
# 1. Clone/navigate to project
cd shot_coach

# 2. Install dependencies
pip install -r requirements.txt

# 3. Verify model exists
ls ../outputs/results_optionA/best_model.pth

# 4. Run app
python -m streamlit run app.py
```

## Usage

### Web Interface

1. Start app: `python -m streamlit run app.py`
2. Upload video in browser
3. Click "🚀 Analyze Shot"
4. View results instantly

### Command Line (Alternative)

```bash
# Test with specific video
python test_shot_coach.py ../data/clips/Smash/video.mp4
```

## Troubleshooting

### Model Not Found

**Error**: "Model not found at: ../outputs/results_optionA/best_model.pth"

**Solution**: Make sure you have the trained model. Check path in `app.py` if you moved files.

### ImportError: torchvision

**Solution**: Make sure streamlit uses the correct Python environment:

```bash
# Use python -m to ensure correct environment
python -m streamlit run app.py
```

### Low Confidence (<70%)

**Possible causes:**
- Video too short or too long
- Shot partially visible
- Unusual shot execution
- Transition between shots

**Solution**: Re-record with clearer shot execution

## Limitations

- **5 classes only** - Cannot distinguish variations within shot types
- **2D video** - No depth information
- **Drive/Lift confusion** - Most common error (similar trajectories)
- **Single shot per video** - Cannot handle multiple shots in one video

## Future Improvements

- [ ] Support for shot variations (e.g., attacking clear, defensive drop)
- [ ] Multi-shot video analysis
- [ ] Real-time classification (camera feed)
- [ ] Shot quality scoring
- [ ] Trajectory visualization

## Model Training

Trained on ShuttleSet dataset:
- **22,302 total videos**
- **Train/Val/Test**: 70%/10%/20%
- **Class weights**: Used to handle imbalance
- **Augmentation**: Random flip, color jitter, rotation
- **Early stopping**: Patience of 15 epochs

## Citation

If you use this classifier, please cite:

```
Badminton Shot Classifier
Built with ResNet18 + BiLSTM
Test Accuracy: 74.6%
Training Dataset: ShuttleSet (22,302 samples)
```

---

**Classify your shots and improve your game! 🏸**
