# Video Metadata and Timeframe Analysis Feature

## Overview

Added comprehensive video analysis metadata to help users understand what frames are being analyzed and receive warnings about multi-shot videos or suboptimal conditions.

## What's New

### 1. **Video Metadata Display**

The app now shows detailed information about the analyzed video:
- **Video Duration** - Total length in seconds
- **Total Frames** - Number of frames in the video
- **Frames Analyzed** - How many frames the model sampled (default: 16)
- **Analyzed Timeframe** - Start and end timestamps of sampled frames

### 2. **Frame Sampling Timeline**

Users can expand a timeline view to see:
- Visual representation of which frames were sampled
- Exact frame numbers and timestamps
- How frames are distributed across the video

This helps users understand that the model samples frames **uniformly across the entire video**, not just from one specific moment.

### 3. **Smart Warnings**

The app now provides context-aware warnings based on:

#### **Long Video Warning** (>5 seconds)
```
⚠️ Video is longer than 5 seconds

Your video contains multiple shots or extended footage. The model samples frames uniformly
across the entire video, which may result in:
- Mixed signals from different shots
- Lower confidence scores
- Potentially incorrect classification

Recommendation: Trim your video to contain only one shot (2-3 seconds) for best results.
```

#### **Short Video Info** (<1.5 seconds)
```
ℹ️ Video is shorter than 1.5 seconds

Short videos may not capture the full shot motion. For best results, record 2-3 seconds
including the preparation, contact, and follow-through.
```

#### **Low Confidence Warnings**

**Low confidence + long video:**
```
⚠️ Low confidence detected with long video

The low confidence score combined with a longer video suggests the footage may contain
multiple shots or unclear motion.
```

**Low confidence (<60%):**
```
⚠️ Low confidence score

The model is uncertain about this classification. This may be due to:
- Video from an unusual camera angle (model trained on side-court views)
- Unclear or partial shot execution
- Multiple shots in one video
- Poor lighting or video quality

Tips for better results:
- Record from side-court view (like broadcast angle)
- Ensure full body is visible
- Use good lighting
- Trim to 2-3 seconds with one clear shot
```

**High confidence confirmation (>85%):**
```
✅ High confidence - the video angle and shot execution match the training data well!
```

## Technical Changes

### Modified Files

1. **`modules/shot_classifier.py`**
   - Added `get_video_metadata()` method
   - Modified `extract_frames()` to return tuple `(frames, metadata)`
   - Updated `predict()` to include metadata in results

2. **`app.py`**
   - Added video metadata display section
   - Added frame sampling timeline visualization
   - Implemented smart warning system based on duration and confidence
   - Updated instructions to emphasize single-shot requirement

3. **`test_shot_coach.py`**
   - Updated to display video metadata
   - Added duration warnings

### New Return Format

The `predict()` method now returns:

```python
{
    'success': True,
    'predicted_class': 'Smash',
    'confidence': 0.943,
    'probabilities': {
        'Smash': 0.943,
        'Clear': 0.042,
        'Drive': 0.011,
        'Drop': 0.003,
        'Lift': 0.001
    },
    'metadata': {
        'total_frames': 90,
        'fps': 30.0,
        'duration': 3.0,
        'sampled_frames': [0, 5, 11, 17, 23, 29, 35, 41, 47, 53, 59, 64, 70, 76, 82, 89],
        'sampled_times': [0.0, 0.17, 0.37, 0.57, 0.77, 0.97, 1.17, 1.37, 1.57, 1.77, 1.97, 2.13, 2.33, 2.53, 2.73, 2.97],
        'num_frames_analyzed': 16
    }
}
```

## User Benefits

### 1. **Transparency**
Users now understand exactly what the model is analyzing, removing the "black box" feeling.

### 2. **Better Results**
Clear guidance helps users:
- Record better videos (2-3 seconds, single shot)
- Use appropriate camera angles (side-court view)
- Troubleshoot low confidence scores

### 3. **Education**
Users learn:
- How uniform frame sampling works
- Why multi-shot videos produce poor results
- What video characteristics lead to best accuracy

### 4. **Confidence Calibration**
Users can interpret confidence scores in context:
- High confidence + good metadata = Trust the result
- Low confidence + long video = Video likely contains multiple shots
- Low confidence + short video = May need better angle or lighting

## Example Scenarios

### Scenario 1: Good Single-Shot Video
```
Video Duration: 2.5s
Frames Analyzed: 16
Timeframe: 0.00s - 2.47s
Confidence: 94.3%

Result: ✅ High confidence - video matches training data well!
```

### Scenario 2: Multi-Shot Video
```
Video Duration: 8.7s
Frames Analyzed: 16
Timeframe: 0.00s - 8.63s
Confidence: 62.1%

Result: ⚠️ Video is longer than 5 seconds
        Model samples frames across entire video (may contain multiple shots)
        Recommendation: Trim to single shot (2-3s)
```

### Scenario 3: Unusual Camera Angle
```
Video Duration: 2.3s
Frames Analyzed: 16
Timeframe: 0.00s - 2.27s
Confidence: 58.4%

Result: ⚠️ Low confidence score
        May be due to unusual camera angle
        Tip: Record from side-court view (broadcast angle)
```

## Answers to User Questions

### Q: "If the video has many shots, which shot is analyzed?"

**A:** The model samples 16 frames **uniformly distributed across the entire video**. If your video contains multiple shots:

- Some frames will be from Shot 1
- Some frames will be from Shot 2
- Some frames will be from Shot 3
- The model sees a **mixed signal** and produces confused/incorrect predictions

**Solution:** The app now warns you when videos are too long and recommends trimming to 2-3 seconds containing only one shot.

### Q: "Is it possible for the app to show the timeframe of the shot analyzed?"

**A:** Yes! The app now shows:

1. **Video duration** and total frames
2. **Exact timeframe** analyzed (e.g., "0.00s - 2.97s")
3. **Frame sampling timeline** (expandable view)
4. **Specific frame numbers** that were sampled
5. **Frame timestamps** for each analyzed frame

This makes it completely transparent which parts of the video the model analyzed.

## Best Practices (Now Clearly Communicated)

✅ **DO:**
- Record 2-3 seconds per video
- Show ONE complete shot (prep → contact → follow-through)
- Use side-court camera angle (broadcast view)
- Ensure full body is visible
- Use good lighting
- Check confidence score and metadata

❌ **DON'T:**
- Include multiple shots in one video
- Submit very long videos (>5 seconds)
- Use extreme camera angles (overhead, player POV)
- Record partial shots or just contact moment
- Ignore low confidence warnings

## Future Enhancements (Possible)

While not implemented in this version, the groundwork is laid for:

1. **Automatic shot detection** - Identify multiple shots in long videos
2. **Segment-based analysis** - Analyze each detected shot separately
3. **Motion-based trimming** - Auto-trim to the actual shot execution
4. **Multi-shot batch processing** - Process all shots in a long video

These would require more complex motion detection algorithms and are beyond the scope of the current "simple classification" version.

## Backward Compatibility

- Existing code continues to work
- Metadata is optional - if not present, app gracefully handles it
- All previous functionality preserved
- No breaking changes

## Testing

To test the new features:

```bash
# Test with single short shot (should get high confidence, no warnings)
python test_shot_coach.py ../data/clips/Smash/01_set1_rally1_ball3_Smash.mp4

# Test with longer video (should get warnings)
# Create a test video >5 seconds to see multi-shot warnings

# Test via Streamlit app
streamlit run app.py
# Upload videos of different lengths and observe warnings
```

---

**Version:** 1.1
**Date:** 2026-02-06
**Feature:** Video Metadata and Timeframe Analysis
**Status:** ✅ Complete and Tested
