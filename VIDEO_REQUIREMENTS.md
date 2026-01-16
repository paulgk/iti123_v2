# Video Requirements for AI Badminton Coach

Guidelines for videos that work best with the pose detection system.

---

## ✅ Video Requirements

### Player Visibility
- **Player must be clearly visible** throughout the entire video
- **No occlusion**: Player should not be blocked by net, opponent, or other objects
- **Good lighting**: Well-lit court, avoid shadows
- **Minimal motion blur**: Player movements should be clear

### Video Specifications
- **Duration**: 1-3 seconds (showing one complete stroke)
- **Format**: .mp4, .avi, .mov, .mkv
- **Resolution**: Minimum 720p (1280x720), 1080p recommended
- **Frame rate**: 30fps or higher

### Camera Angle
- **Side view** or **diagonal view** preferred
- Player should be **5-15 meters** from camera
- Avoid extreme close-ups or distant shots
- Keep player in center of frame

### Stroke Focus
- Video should show **one complete stroke**:
  - Preparation phase
  - Execution (contact point)
  - Follow-through
- Avoid videos with multiple strokes

---

## ❌ Common Issues

### "Failed to extract poses"

**Causes:**
1. **Player not visible**: Blocked by objects or outside frame
2. **Video too short**: Less than 1 second
3. **Poor quality**: Low resolution, heavy compression
4. **Corrupted file**: Video file damaged

**Solutions:**
1. Use videos from `data/processed/clips/` (known to work)
2. Extract a better clip from original video
3. Ensure 720p+ resolution
4. Check player is visible throughout

### Low Quality Score (<50% valid frames)

**Causes:**
1. Player partially occluded in some frames
2. Fast movement causing motion blur
3. Poor lighting

**Solutions:**
- Results may still be useful if >30% valid
- Try increasing video quality
- Use better lighting conditions

---

## 📝 Testing

### Test with Sample Videos First

```bash
# These are guaranteed to work:
python analyze_video.py data/processed/clips/01_set1_rally1_ball2_Clear.mp4 Clear
python analyze_video.py data/processed/clips/44_set2_rally8_ball5_Smash.mp4 Smash
```

### Your Own Videos

1. **Start simple**: Test with one high-quality clip
2. **Check output**: Look at quality percentage
3. **Iterate**: Adjust camera angle, lighting if needed

---

## 🎥 Preparing Your Videos

### Using Existing Match Footage

If you have full match videos:

1. **Identify good strokes**:
   - Player clearly visible
   - Single stroke execution
   - Good angle

2. **Extract clip** (using ffmpeg):
   ```bash
   # Extract 3 seconds starting at 1:23
   ffmpeg -i match.mp4 -ss 00:01:23 -t 3 stroke_clip.mp4
   ```

3. **Test the clip**:
   ```bash
   python analyze_video.py stroke_clip.mp4 Clear
   ```

### Recording New Videos

**Setup:**
- Camera on tripod, stable position
- Side angle to court (45-90 degrees)
- Distance: 8-12 meters from player
- Settings: 1080p, 30fps minimum

**Recording:**
- Capture full stroke (prep → execution → follow-through)
- 2-3 second clips work best
- Multiple attempts recommended

---

## 💡 Tips for Best Results

1. **Use dataset clips**: The 4,983 clips in `data/processed/clips/` are pre-validated
2. **Quality over quantity**: One good clip > multiple poor clips
3. **Check visibility**: Can you clearly see the player's arms/shoulders?
4. **Proper lighting**: Outdoor daylight or well-lit indoor courts
5. **Stable camera**: Tripod or stable surface

---

## 🔧 Troubleshooting

### Video works in player but fails in analysis

1. **Check codec**: Convert to standard H.264:
   ```bash
   ffmpeg -i input.mp4 -c:v libx264 -crf 23 output.mp4
   ```

2. **Verify with sample**: Ensure system works with dataset clips first

3. **Run diagnostic**:
   ```bash
   python diagnose.py
   ```

### Results seem inaccurate

1. **Check quality score**: If <30%, redo the clip
2. **Verify stroke type**: Ensure you selected correct type (Clear/Smash)
3. **Multiple attempts**: Try different clips of same stroke

---

## 📊 Example Quality Scores

From dataset analysis:

- **Excellent (>80%)**: Player perfectly visible, ideal conditions
- **Good (50-80%)**: Minor occlusion, usable results
- **Fair (30-50%)**: Significant occlusion, results may vary
- **Poor (<30%)**: Not enough valid frames, retry needed

---

## ✅ Quick Checklist

Before uploading a video:

- [ ] Player visible throughout (check by scrubbing through video)
- [ ] Single stroke shown (1-3 seconds)
- [ ] Good lighting, minimal shadows
- [ ] Resolution 720p or higher
- [ ] File format: .mp4, .avi, .mov, or .mkv
- [ ] No corruption (plays smoothly in video player)

---

**Bottom Line**: Start with dataset clips to verify system works, then use similar quality videos for your own analysis.

Last updated: January 16, 2026
