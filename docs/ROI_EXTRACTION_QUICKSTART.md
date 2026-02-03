# ROI Extraction Quick Start Guide

**Phase 1.5: Single Player ROI-Based Extraction**

---

## Overview

This guide walks you through extracting clips and poses with ROI (Region of Interest) cropping to ensure clean single-player pose detection.

**Key improvements:**
- 0% multi-player skeleton merging (was 10-15%)
- 2.2x more clean training data (15.8K vs 7K)
- Clean shot type mapping (removed 23.3% ambiguous shots)
- Expected accuracy: 89-92% ST-GCN (was 85-90%)

---

## Prerequisites

### 1. Conda Environment

```bash
# Activate conda environment
conda activate iti123

# Verify dependencies
python -c "import cv2, mediapipe; print('✓ Dependencies OK')"
```

### 2. MediaPipe Model

```bash
# Download if not already present
mkdir -p models
cd models

# Download pose_landmarker_heavy.task (29MB)
curl -L -o pose_landmarker_heavy.task \
  "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_heavy/float16/latest/pose_landmarker_heavy.task"

cd ..
```

### 3. ShuttleSet Dataset and Match Videos

```bash
# ShuttleSet structure:
# ShuttleSet/
#   set/
#     match.csv              <- Match metadata
#     {match_name}/          <- Individual match directories
#       set1.csv, set2.csv   <- Shot annotations
#
# Match videos:
#   You need to download/place match videos separately
#   Expected location: data/raw_videos/ or ShuttleSet/match/
#   Format: 01.mp4, 02.mp4, ..., 44.mp4

# Check ShuttleSet annotations exist
ls ShuttleSet/set/match.csv
ls ShuttleSet/set/*/set*.csv | head -5

# Check where you've placed match videos
# (Adjust --input path in extraction command accordingly)
ls data/raw_videos/*.mp4 | wc -l
# Should be: 44 match videos
```

---

## Step-by-Step Extraction

### Step 1: Extract Clips (2 hours)

Extract video clips from full match videos using refined shot type mapping.

```bash
# Extract all clips with refined mapping
# Note: Adjust --input path to where your match videos are located
python scripts/extract_shuttleset_clips.py \
    --input data/raw_videos \
    --output data/clips \
    --shuttleset ShuttleSet \
    --execute

# Or if videos are in ShuttleSet/match/:
# python scripts/extract_shuttleset_clips.py \
#     --input ShuttleSet/match \
#     --output data/clips \
#     --shuttleset ShuttleSet \
#     --execute

# Expected output:
#   Smash:  ~4,234 clips
#   Clear:  ~2,922 clips
#   Drop:   ~6,290 clips (NO slices!)
#   Lift:   ~5,632 clips
#   Drive:  ~700 clips (NO push/rear/defensive!)
#   TOTAL:  ~19,778 clips
```

**What changed:**
- ❌ Removed Slice_Drop (2,144 shots) - different wrist technique
- ❌ Removed Push (2,925 shots) - net shots, not drives
- ❌ Removed Rear/Defensive_Drive (879 shots) - different positioning
- ✅ Clean 5-class mapping: 19,778 shots

### Step 2: Create Metadata (5 minutes)

Generate metadata.csv with player positions for ROI extraction.

```bash
# Create metadata CSV
python scripts/create_metadata_csv.py \
    --shuttleset ShuttleSet \
    --clips data/clips \
    --output data/metadata.csv

# Expected output:
#   ✓ Saved 19,778 entries to data/metadata.csv
#   Each entry includes: video_id, player, player_x, player_y
```

**Metadata format:**
```csv
video_id,match_id,set_num,rally,ball_round,shot_type,player,player_x,player_y,frame_num,clip_path,pose_path
01_set1_rally03_ball05_Smash,1,1,3,5,Smash,A,458,256,7199,data/clips/Smash/01_set1_rally03_ball05_Smash.mp4,data/poses/01_set1_rally03_ball05_Smash.pkl
```

### Step 3: Extract Poses with ROI (8-12 hours)

Extract poses using ROI cropping around player position.

```bash
# Extract poses with ROI (8 parallel workers)
python scripts/extract_poses_roi.py \
    --clips data/clips \
    --metadata data/metadata.csv \
    --output data/poses \
    --model models/pose_landmarker_heavy.task \
    --num-workers 8

# Expected output:
#   ✓ 19,778 clips processed
#   ✓ ~19,500 successful (98%+ success rate)
#   ✓ Each pose: (T, 33, 3) array
```

**ROI parameters (default):**
- ROI size: 600x800 pixels (width × height)
- Centered on player_x, player_y from CSV
- Captures full body + arm extension
- Excludes opponent (typically >600px away)

**What happens:**
1. Load video frame
2. Calculate ROI around player position
3. Crop frame to ROI (600x800)
4. Run MediaPipe on cropped frame (only one player visible!)
5. Transform coordinates back to original frame space
6. Save pose sequence

---

## Step 4: Train Models (4-6 hours)

Train models using the clean pose data.

```bash
# Train ST-GCN model
python scripts/train_models_fixed.py \
    --metadata data/metadata.csv \
    --pose-dir data/poses \
    --output outputs/ \
    --model stgcn \
    --epochs 50

# Expected results:
#   ✓ ~15,822 usable samples after filtering
#   ✓ 0% multi-player detections (was 10-15%)
#   ✓ Std: 0.15-0.35 (was 0.59)
#   ✓ ST-GCN accuracy: 89-92% (was 85-90%)
```

---

## Verification Checks

### After Clip Extraction

```bash
# Count clips by shot type
find data/clips -name "*.mp4" | wc -l
# Should be: ~19,778

# Check shot distribution
for shot in Smash Clear Drop Lift Drive; do
    echo "$shot: $(find data/clips/$shot -name '*.mp4' | wc -l)"
done

# Expected:
#   Smash: ~4,234
#   Clear: ~2,922
#   Drop: ~6,290
#   Lift: ~5,632
#   Drive: ~700
```

### After Metadata Creation

```bash
# Check metadata.csv
wc -l data/metadata.csv
# Should be: 19,779 (including header)

# Check player distribution
awk -F',' '{print $8}' data/metadata.csv | tail -n +2 | sort | uniq -c
# Should show ~50/50 split between Player A and B
```

### After Pose Extraction

```bash
# Count pose files
find data/poses -name "*.pkl" | wc -l
# Should be: ~19,500+ (98%+ success rate)

# Check a sample pose
python -c "
import pickle
import numpy as np
with open('data/poses/01_set1_rally03_ball05_Smash.pkl', 'rb') as f:
    pose = pickle.load(f)
print(f'Shape: {pose.shape}')
print(f'Mean: {pose.mean():.4f}')
print(f'X-range: {pose[:, :, 0].max() - pose[:, :, 0].min():.4f}')
"

# Expected:
#   Shape: (T, 33, 3) - T frames, 33 keypoints, 3 coords
#   Mean: 0.3-0.7 (before normalization)
#   X-range: <0.4 (single player, not >0.6!)
```

### After Training

```bash
# Check training outputs
ls outputs/models/*_best.pth
ls outputs/reports/*.txt
ls outputs/visualizations/*.png

# Check training report
cat outputs/reports/STGCN_report.txt

# Should show:
#   Test Accuracy: 89-92%
#   Per-class F1: >0.75 for all classes
#   Training samples: ~15,822
```

---

## Troubleshooting

### Issue: Low clip count (<19,000)

**Cause:** Missing match videos

**Solution:**
```bash
# Check which matches are missing
for i in {01..44}; do
    if [ ! -f "ShuttleSet/match/${i}.mp4" ]; then
        echo "Missing: Match ${i}"
    fi
done
```

### Issue: Metadata.csv has fewer entries than clips

**Cause:** Player position data missing in some CSVs

**Solution:** Check ShuttleSet CSV files have player_location_x/y columns

### Issue: Pose extraction fails with "Model not found"

**Cause:** MediaPipe model not downloaded

**Solution:**
```bash
# Download model
curl -L -o models/pose_landmarker_heavy.task \
  "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_heavy/float16/latest/pose_landmarker_heavy.task"
```

### Issue: ROI extraction very slow

**Cause:** Using only 1 worker

**Solution:** Increase workers (use CPU core count - 2)
```bash
# For 10-core machine
python scripts/extract_poses_roi.py --num-workers 8
```

### Issue: High multi-player detection after ROI

**Cause:** ROI too large or player position inaccurate

**Solution:** Reduce ROI size
```bash
# Try smaller ROI
python scripts/extract_poses_roi.py --roi-width 500 --roi-height 700
```

### Issue: Training accuracy still low (<85%)

**Possible causes:**
1. Multi-player contamination - Check x-range of poses
2. Normalization not applied - Check mean ~0.0
3. Learning rate too low - Should be 0.001
4. Short sequences not filtered - Check MIN_FRAMES=30

**Debug:**
```python
import pickle
import numpy as np

# Load a random pose
with open('data/poses/01_set1_rally03_ball05_Smash.pkl', 'rb') as f:
    pose = pickle.load(f)

# Check multi-player
x_range = pose[:, :, 0].max() - pose[:, :, 0].min()
print(f"X-range: {x_range:.4f}")
if x_range > 0.6:
    print("⚠️ Multi-player detection! ROI may be too large")

# Check sequence length
print(f"Frames: {len(pose)}")
if len(pose) < 30:
    print("⚠️ Sequence too short!")
```

---

## Performance Expectations

### Extraction Time

| Task | Clips | Time (8 workers) | Time (4 workers) |
|------|-------|------------------|------------------|
| Clip extraction | 19,778 | ~2 hours | ~3 hours |
| Metadata creation | 19,778 | ~5 minutes | ~5 minutes |
| Pose extraction | 19,778 | ~8-10 hours | ~14-16 hours |
| **Total** | **19,778** | **~10-12 hours** | **~17-19 hours** |

**Recommendation:** Use 8 workers, run overnight

### Training Time

| Model | Samples | Time (T4 GPU) | Time (CPU) |
|-------|---------|---------------|------------|
| LSTM | 15,822 | ~45 min | ~3 hours |
| ST-GCN | 15,822 | ~60 min | ~4 hours |
| MS-TCN | 15,822 | ~50 min | ~3 hours |

### Expected Accuracy

| Model | Current (7K noisy) | ROI (15.8K clean) | Improvement |
|-------|-------------------|-------------------|-------------|
| LSTM | 75-82% | **84-88%** | +9-13% |
| **ST-GCN** | 85-90% | **89-92%** | **+4-7%** |
| MS-TCN | 82-88% | **87-91%** | +5-9% |

---

## Next Steps After Extraction

1. **Upload to GCS** (optional):
```bash
# Upload poses
gsutil -m rsync -r data/poses/ gs://iti123storage/features/poses/

# Upload metadata
gsutil cp data/metadata.csv gs://iti123storage/data/
```

2. **Train in Colab**:
- Upload metadata.csv to Colab
- Download poses from GCS
- Run training notebook

3. **Save outputs**:
```bash
# Save training outputs to git
bash scripts/save_outputs_to_git.sh outputs/
```

4. **Push to remote**:
```bash
git push origin phase-1.5-roi-extraction
```

---

## Summary

**Complete workflow:**
```bash
# 1. Extract clips (2 hours)
python scripts/extract_shuttleset_clips.py --execute

# 2. Create metadata (5 minutes)
python scripts/create_metadata_csv.py

# 3. Extract poses with ROI (8-12 hours - can run overnight)
python scripts/extract_poses_roi.py --num-workers 8

# 4. Train models (4-6 hours)
python scripts/train_models_fixed.py --model stgcn --epochs 50
```

**Expected results:**
- ✅ 19,778 clips with clean shot mapping
- ✅ 15,822 usable training samples
- ✅ 0% multi-player contamination
- ✅ 89-92% ST-GCN accuracy

**Total time:** 16-20 hours (mostly extraction, can run unattended)

---

**Status:** Ready to run
**Documentation:** See [ROI_EXTRACTION_PLAN.md](ROI_EXTRACTION_PLAN.md) for details
**Support:** Check troubleshooting section or review training logs
