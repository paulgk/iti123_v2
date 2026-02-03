# Phase 1.5 ROI Extraction - Quick Reference

**Single Player ROI-based pose extraction with clean shot mapping**

---

## Test on One Match (Recommended First)

```bash
# Test complete pipeline on Match 01
bash scripts/test_roi_extraction.sh

# This will:
# - Extract clips from Match 01
# - Create metadata with player positions
# - Extract poses with ROI cropping
# - Validate results automatically
# - Show statistics and quality metrics
```

**Expected output:**
- ~400-500 clips from Match 01
- ~98%+ pose extraction success
- 0% multi-player detections (ROI working!)
- Validation summary with recommendations

---

## Full Extraction (All 44 Matches)

### Step 1: Extract Clips (2 hours)

```bash
python scripts/extract_shuttleset_clips.py \
    --input data/raw_videos \
    --output data/clips \
    --shuttleset ShuttleSet \
    --execute

# Expected: ~19,778 clips
# Smash: 4,234 | Clear: 2,922 | Drop: 6,290 | Lift: 5,632 | Drive: 700
```

### Step 2: Create Metadata (5 minutes)

```bash
python scripts/create_metadata_csv.py \
    --shuttleset ShuttleSet \
    --clips data/clips \
    --output data/metadata.csv

# Expected: 19,778 entries with player positions
```

### Step 3: Extract Poses with ROI (8-12 hours)

```bash
python scripts/extract_poses_roi.py \
    --clips data/clips \
    --metadata data/metadata.csv \
    --output data/poses \
    --model models/pose_landmarker_heavy.task \
    --num-workers 8

# Expected: ~19,500 poses (98%+ success)
# Can run overnight
```

### Step 4: Validate Results

```bash
python scripts/validate_roi_poses.py \
    --poses data/poses \
    --metadata data/metadata.csv

# Should show:
# - 0% multi-player detections
# - ~20% short sequences (will be filtered)
# - ~15,822 usable training samples
```

---

## Training

```bash
# Train ST-GCN model
python scripts/train_models_fixed.py \
    --metadata data/metadata.csv \
    --pose-dir data/poses \
    --output outputs/ \
    --model stgcn \
    --epochs 50

# Expected: 89-92% accuracy
```

---

## Troubleshooting Commands

### Check clip counts

```bash
find data/clips -name "*.mp4" | wc -l
# Should be: ~19,778

for shot in Smash Clear Drop Lift Drive; do
    echo "$shot: $(find data/clips/$shot -name '*.mp4' | wc -l)"
done
```

### Check metadata

```bash
wc -l data/metadata.csv
# Should be: 19,779 (including header)

head -3 data/metadata.csv
# Check player_x, player_y columns exist
```

### Check poses

```bash
find data/poses -name "*.pkl" | wc -l
# Should be: ~19,500

# Test load a pose
python -c "
import pickle
with open('data/poses/01_set1_rally03_ball05_Smash.pkl', 'rb') as f:
    pose = pickle.load(f)
print(f'Shape: {pose.shape}')
print(f'Frames: {len(pose)}')
"
```

### Check for multi-player contamination

```bash
python scripts/validate_roi_poses.py \
    --poses data/poses \
    --multi-player-threshold 0.6

# Should show 0% or <5% multi-player detections
```

---

## Configuration

### Default Settings

| Parameter | Default | Description |
|-----------|---------|-------------|
| ROI width | 600px | Captures full body + arm extension |
| ROI height | 800px | Full height from head to feet |
| Workers | 4-8 | Parallel processing (use CPU cores - 2) |
| Min frames | 30 | Minimum sequence length (1 second at 30 FPS) |
| Multi-player threshold | 0.6 | X-range >60% indicates multi-player |

### Adjust ROI Size

```bash
# If multi-player rate is high (>5%), try smaller ROI
python scripts/extract_poses_roi.py \
    --roi-width 500 \
    --roi-height 700 \
    --num-workers 8
```

### Adjust Workers

```bash
# For 10-core machine
python scripts/extract_poses_roi.py --num-workers 8

# For 6-core machine
python scripts/extract_poses_roi.py --num-workers 4

# Sequential (debugging)
python scripts/extract_poses_roi.py --num-workers 1
```

---

## File Locations

### Input
- Match videos: `data/raw_videos/*.mp4` (44 videos)
- ShuttleSet annotations: `ShuttleSet/set/` (CSV files)
- MediaPipe model: `models/pose_landmarker_heavy.task`

### Output
- Clips: `data/clips/{shot_type}/*.mp4`
- Metadata: `data/metadata.csv`
- Poses: `data/poses/*.pkl`

### Test Output (from test script)
- Test clips: `data/clips_test/`
- Test poses: `data/poses_test/`
- Test metadata: `data/metadata_test.csv`

---

## Expected Timeline

| Task | Time | Can Run Unattended? |
|------|------|---------------------|
| Test (1 match) | ~15 min | No (check output) |
| Clip extraction | ~2 hours | Yes |
| Metadata creation | ~5 min | Yes |
| Pose extraction | ~8-12 hours | Yes (overnight) |
| Validation | ~5 min | No (review results) |
| Training | ~4-6 hours | Yes (GPU recommended) |
| **Total** | **~16-20 hours** | **Mostly yes** |

---

## Expected Results

### Dataset
- Clean shots: 19,778 (removed 23.3% ambiguous)
- Clips extracted: ~19,778
- Poses extracted: ~19,500 (98% success)
- Usable samples: ~15,822 (after filtering short/multi-player)

### Quality Metrics
- Multi-player rate: 0% (ROI prevents merging)
- Short sequences: ~20% (filtered during training)
- Mean coordinates: 0.3-0.7 (before normalization)
- Std: 0.2-0.4 (after normalization should be 0.15-0.35)

### Model Performance
- ST-GCN: **89-92%** accuracy (was 85-90%)
- LSTM: 84-88% (was 75-82%)
- MS-TCN: 87-91% (was 82-88%)

---

## Quick Commands Summary

```bash
# 1. Test first
bash scripts/test_roi_extraction.sh

# 2. If test looks good, full extraction
python scripts/extract_shuttleset_clips.py --execute
python scripts/create_metadata_csv.py
python scripts/extract_poses_roi.py --num-workers 8

# 3. Validate
python scripts/validate_roi_poses.py --poses data/poses --metadata data/metadata.csv

# 4. Train
python scripts/train_models_fixed.py --model stgcn --epochs 50
```

---

## Documentation

- Full plan: [docs/ROI_EXTRACTION_PLAN.md](docs/ROI_EXTRACTION_PLAN.md)
- Quick start: [docs/ROI_EXTRACTION_QUICKSTART.md](docs/ROI_EXTRACTION_QUICKSTART.md)
- Shot mapping: [docs/SHOT_TYPE_MAPPING_REFINED.md](docs/SHOT_TYPE_MAPPING_REFINED.md)
- Dataset info: [docs/EXTRACTION_SUMMARY.md](docs/EXTRACTION_SUMMARY.md)

---

**Branch:** `phase-1.5-roi-extraction`
**Status:** Ready to run
**Last updated:** 2026-02-03
