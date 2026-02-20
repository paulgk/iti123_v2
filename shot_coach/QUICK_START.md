# 🚀 Shot Coach - Quick Start Guide

Get your Shot Coach running in 5 minutes!

## Prerequisites

- Python 3.8+ installed
- The trained model at `../outputs/results_optionA/best_model.pth`
- A video clip of a badminton shot (Smash or Clear)

## Installation (2 minutes)

1. **Navigate to shot\_coach directory:**

```bash
cd shot_coach
```

2. **Install dependencies:**

```bash
pip install -r requirements.txt
```

That's it! You're ready to go.

## Running the App (1 minute)

**Start Streamlit:**

```bash
streamlit run app.py
```

The app will:
- Open automatically in your browser at `http://localhost:8501`
- Show the Shot Coach interface

## Using the App (2 minutes)

### Step 1: Prepare Your Video

Record a 2-3 second video of:
- **Smash**: Overhead attacking shot
- **Clear**: High defensive shot

**Requirements:**
- You should be fully visible (full body preferred)
- Good lighting
- Clear background
- MP4, AVI, or MOV format

### Step 2: Upload and Analyze

1. Click "Browse files" in the app
2. Select your video
3. Click "🚀 Analyze Shot"
4. Wait 10-30 seconds

### Step 3: Review Results

You'll get:
- ✅ Shot type classification
- 📊 Overall score (0-100)
- 💡 Specific feedback
- 📋 Detailed metrics
- 📥 Downloadable report

## Test with Command Line (Alternative)

If you prefer command-line testing:

```bash
python test_shot_coach.py <path_to_video.mp4>
```

Example:

```bash
python test_shot_coach.py ../data/clips/Smash/01_set1_rally1_ball3_Smash.mp4
```

## Example Output

```
======================================================================
SHOT COACH - ANALYSIS REPORT
======================================================================

Shot Type: Smash
Confidence: 94.3%
Overall Score: 72.0/100

OVERALL ASSESSMENT
Rating: Good
Solid Smash technique with room for minor improvements.

AREAS TO IMPROVE
  • Rotate shoulders more (current: 95°, target: 120°+)
  • Bend knees deeper before jump (current: 25°, target: 30-45°)
```

## Common Issues

### "Model not found"

**Fix:** Make sure you're running from the `shot_coach` directory and the model exists:

```bash
ls ../outputs/results_optionA/best_model.pth
```

### "No pose detected"

**Fix:** Re-record video with:
- Better lighting
- You're fully visible
- Camera is stable

### "ModuleNotFoundError"

**Fix:** Install missing dependency:

```bash
pip install <missing_module>
```

## Tips for Best Results

1. **Video Quality:**
  - Use good lighting
  - Stable camera (no shaking)
  - Clear background
  - 1080p or 720p resolution

2. **Shot Execution:**
  - Full body in frame
  - Complete shot motion (prep → contact → follow-through)
  - Not too fast (normal speed)

3. **Camera Position:**
  - Side view works best
  - About 3-5 meters away
  - Eye level or slightly above

## Next Steps

- Try different shots and compare scores
- Track your improvement over time
- Focus on metrics with lower scores
- Practice recommended improvements

## Need Help?

Check the full [README.md](./README.md) for:
- Detailed documentation
- Troubleshooting guide
- Technical architecture
- API reference

---

**Ready to improve your badminton game? Let's go! 🏸**
