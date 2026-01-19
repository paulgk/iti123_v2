# Video Clip Extraction Guide

## Current Status

✅ **Full match videos**: 44 videos in `data/raw_videos/` (01.mp4, 02.mp4, etc.)
✅ **Existing clips**: 4,983 clips for Clear/Smash already extracted
❌ **Missing clips**: Drop and Lift clips need to be extracted

---

## Quick Start - Extract Drop and Lift Clips

### For Drop vs Smash Dataset:
```bash
python extract_video_clips.py \
  --metadata data/processed/clips/drop_smash_metadata.csv \
  --videos data/raw_videos \
  --output data/processed/clips
```

**Expected output**: ~3,378 Drop clips + 2,575 Smash clips = 5,953 clips
**Time**: ~15-30 minutes

### For Lift vs Smash Dataset:
```bash
python extract_video_clips.py \
  --metadata data/processed/clips/lift_smash_metadata.csv \
  --videos data/raw_videos \
  --output data/processed/clips
```

**Expected output**: ~2,753 Lift clips + 2,575 Smash clips = 5,328 clips
**Time**: ~10-20 minutes

### For Multi-Class Dataset (4 strokes):
```bash
python extract_video_clips.py \
  --metadata data/processed/clips/multiclass_4stroke_metadata.csv \
  --videos data/raw_videos \
  --output data/processed/clips
```

**Expected output**: 11,307 clips (Clear, Smash, Drop, Lift)
**Time**: ~30-45 minutes

---

## How It Works

### Input:
1. **Metadata CSV**: Contains frame numbers and clip names
   - Example row: `match_id=1, frame_num=11921, clip_name=01_set1_rally2_ball3_Smash.mp4`

2. **Full match videos**: In `data/raw_videos/`
   - Named by match_id: `01.mp4`, `02.mp4`, etc.

### Process:
1. Script reads metadata CSV
2. For each stroke:
   - Finds corresponding full match video (by match_id)
   - Extracts 2-second clip centered on stroke frame
   - Saves to `data/processed/clips/`

### Output:
- Individual .mp4 clips for each stroke
- Named like: `01_set1_rally2_ball3_Smash.mp4`

---

## Advanced Options

### Skip Existing Clips (Default)
```bash
python extract_video_clips.py --metadata <file>
```
Only extracts clips that don't exist yet. Fast if some clips already extracted.

### Force Re-Extract All Clips
```bash
python extract_video_clips.py --metadata <file> --force
```
Re-extracts even if clips exist. Use if clips are corrupted.

### Custom Clip Duration
```bash
python extract_video_clips.py --metadata <file> --duration 3.0
```
Default is 2.0 seconds. Increase for longer context.

### Custom FPS
```bash
python extract_video_clips.py --metadata <file> --fps 25
```
Default is 30 FPS. Match your video's actual FPS.

---

## Verify Extraction

### Check clip counts:
```bash
echo "Total clips:"
find data/processed/clips -name "*.mp4" | wc -l

echo "By stroke type:"
find data/processed/clips -name "*Clear.mp4" | wc -l | xargs echo "Clear:"
find data/processed/clips -name "*Smash.mp4" | wc -l | xargs echo "Smash:"
find data/processed/clips -name "*Drop.mp4" | wc -l | xargs echo "Drop:"
find data/processed/clips -name "*Lift.mp4" | wc -l | xargs echo "Lift:"
```

### Expected counts:
- **After Drop vs Smash**: ~6,300 Drop + 2,575 Smash = ~8,875 total
- **After Lift vs Smash**: +2,753 Lift = ~11,628 total
- **After Multi-Class**: All 11,307 clips for 4 stroke types

---

## Common Issues

### Issue: "No video found for match_id X"
**Cause**: Video file missing in `data/raw_videos/`
**Solution**: Check if `XX.mp4` exists in `data/raw_videos/`

### Issue: "Could not open video"
**Cause**: Corrupted video file or wrong codec
**Solution**: Re-download video or convert with ffmpeg:
```bash
ffmpeg -i input.mp4 -c:v libx264 -c:a aac output.mp4
```

### Issue: Clips are too short/cut off
**Cause**: Frame number near start/end of video
**Solution**: Script automatically handles this by using `max(0, frame - duration/2)`

### Issue: Wrong clips extracted
**Cause**: Mismatch between match_id in metadata and video filename
**Solution**: Check that:
  - metadata `match_id=1` corresponds to `01.mp4`
  - metadata `match_id=10` corresponds to `10.mp4`

---

## After Clip Extraction

Once clips are extracted, proceed with pose extraction:

### Step 1: Extract Poses (30-60 min)
```bash
# The extract_poses.py script will automatically:
# 1. Read all .mp4 clips from data/processed/clips/
# 2. Extract MediaPipe poses
# 3. Save to data/processed/poses/

# Just run it (it handles all strokes):
python src/data_processing/extract_poses.py
```

### Step 2: Continue with Pipeline
```bash
python process_drop_smash_pipeline.py
# or
python process_lift_smash_pipeline.py
```

---

## File Structure After Extraction

```
data/
├── raw_videos/              # Full match videos
│   ├── 01.mp4              # Match 1
│   ├── 02.mp4              # Match 2
│   └── ...
│
└── processed/
    └── clips/              # Individual stroke clips
        ├── 01_set1_rally2_ball3_Smash.mp4
        ├── 01_set1_rally3_ball4_Smash.mp4
        ├── 01_set1_rally3_ball18_Clear.mp4
        ├── 02_set1_rally5_ball3_Drop.mp4    # NEW: Drop clips
        ├── 02_set1_rally7_ball2_Lift.mp4    # NEW: Lift clips
        └── ...
```

---

## Workflow Summary

```
Full Match Videos (data/raw_videos/)
          ↓
  extract_video_clips.py
          ↓
Individual Clips (data/processed/clips/)
          ↓
  src/data_processing/extract_poses.py
          ↓
Pose Files (data/processed/poses/)
          ↓
  Continue with pipeline...
```

---

## Troubleshooting

### Verify metadata has correct columns:
```bash
head -1 data/processed/clips/drop_smash_metadata.csv
```

Should have: `match_id`, `frame_num`, `clip_name`

### Check match video availability:
```bash
python -c "
import pandas as pd
df = pd.read_csv('data/processed/clips/drop_smash_metadata.csv')
matches = df['match_id'].unique()
print(f'Unique matches needed: {len(matches)}')
print(f'Matches: {sorted(matches)}')
"
```

Then verify each match has a corresponding video in `data/raw_videos/`

---

**Status**: Script ready to use!
**Next**: Run extraction command above
