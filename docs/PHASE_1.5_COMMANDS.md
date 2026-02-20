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

### Option 1: Extract All at Once (10-12 hours)

```bash
bash scripts/extract_full_pipeline.sh
# Extracts matches 01-44 in one run
```

### Option 2: Split into Two Phases (Recommended)

**Phase 1a: First Half (5-6 hours)**
```bash
bash scripts/extract_full_pipeline.sh 01 22
# Extracts matches 01-22
# Train model after this completes
```

**Phase 1b: Second Half (5-6 hours)**
```bash
bash scripts/extract_full_pipeline.sh 23 44
# Extracts matches 23-44
# Retrain model with combined data
```

---

## Expected Results

### After First Half (Matches 01-22)
- ~9,889 clips extracted
- ~9,700 poses extracted (98% success)
- ~7,911 usable training samples
- Ready for initial training

### After Full Dataset (Matches 01-44)
- ~19,778 clips extracted
- ~19,500 poses extracted (98% success)
- ~15,822 usable training samples
- Final training with full data

---

## Training Commands

### Option 1: Colab Notebook (Recommended)

Train 4 deep learning models in Google Colab:
- **ST-GCN** (Graph Convolutional Network) - 89-92% accuracy
- **MS-G3D** (Multi-Scale GCN) - 90-93% accuracy ⭐ Best
- **BiLSTM** (Temporal baseline) - 84-88% accuracy
- **Skeleton Transformer** (Attention-based) - 87-91% accuracy

```python
# 1. Open in Colab: notebooks/badminton_action_recognition_training.ipynb
# 2. Runtime → Change runtime type → GPU (T4)
# 3. Run all cells (~4-5 hours for all 4 models)
```

See [docs/COLAB_TRAINING_GUIDE.md](docs/COLAB_TRAINING_GUIDE.md) for complete guide.

### Option 2: Local Training (Scripts)

#### After First Half
```bash
python scripts/train_models_fixed.py \
    --metadata data/metadata.csv \
    --pose-dir data/poses \
    --output outputs/phase1a/ \
    --model stgcn \
    --epochs 50
```

#### After Second Half
```bash
python scripts/train_models_fixed.py \
    --metadata data/metadata.csv \
    --pose-dir data/poses \
    --output outputs/phase1b/ \
    --model stgcn \
    --epochs 50
```

---

## Monitoring Progress

### Enhanced Logging Features

All scripts now include detailed progress tracking:

**During Clip Extraction:**
- Match-by-match progress (e.g., `[5/44] Match 05: Tournament Name`)
- Per-match shot counts by type
- Progress updates every 50 clips
- Real-time failure tracking
- Estimated time remaining after each match

**During Metadata Creation:**
- Match-by-match progress with clip counts
- Total clips found vs missing
- Final breakdown by shot type

**During Pose Extraction:**
- Shot type breakdown at start
- Progress bar with clip count
- Status reports every 60 seconds showing:
  - Current progress percentage
  - Success rate
  - Processing rate (clips/sec)
  - Estimated time remaining
- Final summary with:
  - Total duration (HH:MM:SS)
  - Average time per clip
  - Success rate

### Check Running Pipeline

```bash
# View real-time log output (for extract_full_pipeline.sh)
tail -f logs/extraction_*_*.log

# Count extracted clips so far
find data/clips -name "*.mp4" | wc -l

# Count extracted poses so far
find data/poses -name "*.pkl" | wc -l

# Check poses by shot type
for shot in Smash Clear Drop Lift Drive; do
    echo "$shot: $(find data/poses -name "*_${shot}.pkl" | wc -l)"
done
```

### Validate After Completion

```bash
python scripts/validate_roi_poses.py \
    --poses data/poses \
    --metadata data/metadata.csv
```

---

## Troubleshooting

### Pipeline Stopped Unexpectedly

```bash
# Check last log for errors
tail -100 logs/extraction_*_*.log

# Resume from pose extraction step if clips already extracted
python scripts/extract_poses_roi.py \
    --clips data/clips \
    --metadata data/metadata.csv \
    --output data/poses \
    --model models/mediapipe/pose_landmarker_heavy.task \
    --num-workers 8
```

### Out of Memory

```bash
# Reduce number of workers
# Edit extract_full_pipeline.sh: NUM_WORKERS=4 (instead of 8)
bash scripts/extract_full_pipeline.sh 01 22
```

### Check Specific Match Progress

```bash
# Count clips for specific match
find data/clips -name "01_*.mp4" | wc -l

# Count poses for specific match
find data/poses -name "01_*.pkl" | wc -l
```

---

## File Locations

### Logs
- Location: `logs/extraction_01_to_22_YYYYMMDD_HHMMSS.log`
- Contains: Step-by-step execution, timings, success rates

### Outputs
- Clips: `data/clips/{Smash,Clear,Drop,Lift,Drive}/*.mp4`
- Metadata: `data/metadata.csv`
- Poses: `data/poses/*.pkl`

### Validation Reports
- Printed to console after validation step
- Shows: multi-player rate, short sequences, usable samples

---

## GCS Storage Management

### Clean Up Old Files First

Before uploading new ROI data, clean up old files:

```bash
# 1. Analyze current storage
bash scripts/list_gcs_contents.sh

# 2. Interactive cleanup (recommended)
bash scripts/clean_gcs_interactive.sh

# Delete:
#   - Old pose extractions (poses_old, poses_backup)
#   - Ambiguous shots (Slice_Drop, Push, Rear_Drive)
#   - Test files (clips_test, poses_test)
#   - Old outputs/models
```

**Expected savings:** 15-30 GB

See [docs/GCS_CLEANUP_GUIDE.md](docs/GCS_CLEANUP_GUIDE.md) for details.

### Upload New ROI Data

After cleanup, use the upload script:

```bash
# Quick upload (poses + metadata) - RECOMMENDED
bash scripts/quick_upload_gcs.sh

# Or full-featured upload with verification
bash scripts/upload_poses_to_gcs.sh

# Or manual upload
gsutil -m rsync -r data/poses/ gs://iti123storage/features/poses_roi/
gsutil cp data/metadata.csv gs://iti123storage/data/metadata_roi.csv
```

**Upload time:** 5-10 minutes for poses, <1 minute for metadata

See [docs/GCS_UPLOAD_DOWNLOAD_GUIDE.md](docs/GCS_UPLOAD_DOWNLOAD_GUIDE.md) for Colab download.

---

## Quick Commands Summary

```bash
# 1. Test first (15 min)
bash scripts/test_roi_extraction.sh

# 2. Extract first half (5-6 hours, run overnight)
bash scripts/extract_full_pipeline.sh 01 22

# 3. Clean GCS and upload
bash scripts/clean_gcs_interactive.sh
bash scripts/quick_upload_gcs.sh

# 4. Train on first half (4-6 hours)
python scripts/train_models_fixed.py --model stgcn --epochs 50

# 5. Extract second half (5-6 hours)
bash scripts/extract_full_pipeline.sh 23 44

# 6. Train on full dataset (4-6 hours)
python scripts/train_models_fixed.py --model stgcn --epochs 50
```

---

**Ready to run!** Start with the test script to verify everything works, then proceed with the first half extraction.
