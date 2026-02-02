# Colab Setup Guide for Pose Extraction

**Date:** 2026-02-02
**Purpose:** Run pose extraction in Google Colab with MediaPipe 0.10.x

---

## Overview

This guide explains how to set up and run pose extraction for all 23,531 badminton clips in Google Colab.

**Why Colab:**
- ✅ GPU acceleration (4-6 hours vs 12+ hours locally)
- ✅ Handles MediaPipe 0.10.x model downloads automatically
- ✅ Better for large datasets
- ✅ Free tier sufficient for this task

---

## Prerequisites

### 1. Upload Clips to GCS

First, upload your extracted clips to Google Cloud Storage:

```bash
# On your local machine
bash scripts/upload_clips_to_gcs.sh --execute
```

This will upload all 23,531 clips (26GB) to:
- `gs://iti123storage/videos/clips/smash/` (3,872 clips)
- `gs://iti123storage/videos/clips/clear/` (2,662 clips)
- `gs://iti123storage/videos/clips/drop/` (7,769 clips)
- `gs://iti123storage/videos/clips/lift/` (5,230 clips)
- `gs://iti123storage/videos/clips/drive/` (3,998 clips)

**Upload time:** 2-4 hours for 26GB
**Features:** Automatic chunking (200 clips/batch), resume capability

### 2. GCS Service Account Key

Download your service account JSON key from Google Cloud Console:
1. Go to IAM & Admin → Service Accounts
2. Create or select service account
3. Create Key → JSON
4. Download the JSON file

---

## Colab Workflow

### Step 1: Open Notebook

Open in Colab:
- **Main workflow:** [`complete_workflow_colab_cleaned.ipynb`](../notebooks/complete_workflow_colab_cleaned.ipynb)
- **Alternative:** [`complete_workflow_colab.ipynb`](../notebooks/complete_workflow_colab.ipynb)

### Step 2: Run Phase 1 (Setup) - ~10 minutes

Execute these cells in order:

**1.1: Clone Repository**
```python
# Clones repo to /content/iti123_v2
%cd /content/iti123_v2
```

**1.2: Authenticate GCS**
```python
# Upload your service account JSON key
from google.colab import files
uploaded = files.upload()
```

**1.3: Verify Videos**
```python
# Should show ~23,500 clips across 5 shot types
```

**1.4: Create Directories**
```python
# Creates data/videos/clips/{smash,clear,drop,lift,drive}/
```

### Step 3: Run Phase 2 (Pose Extraction) - ~4-6 hours

**2.1: Download Videos** (~30-60 minutes for 26GB)
```python
!gsutil -m rsync -r gs://iti123storage/videos/clips/ data/videos/clips/
```

**2.1.5: Setup MediaPipe** (~1 minute)
```python
# Downloads pose_landmarker_heavy.task (~30MB)
# This is NEW - required for MediaPipe 0.10.x
```

**2.2: Extract Poses** (~4-6 hours with GPU)
```python
!python scripts/extract_poses_parallel.py \
    --video-dir data/videos/clips \
    --output-dir data/processed/poses \
    --model-complexity 1 \
    --target-fps 20 \
    --num-workers 4
```

**Expected output:**
- ~23,500 pose files (`.pkl` format)
- `data/metadata.csv` with stroke types and player IDs

**2.5: Upload Results to GCS** (~30 minutes)
```python
# Backs up poses and metadata to GCS
!gsutil -m rsync -r data/processed/poses/ gs://iti123storage/features/poses/
```

---

## What's Different from Local Setup

### MediaPipe API Changes

**Local (doesn't work):**
- MediaPipe versions <0.10.13 no longer available
- Old API (`mp.solutions.pose`) removed

**Colab (works):**
- Uses MediaPipe 0.10.x with new task-based API
- Automatically downloads pose landmarker model
- Model stored at: `models/mediapipe/pose_landmarker.task`

### Script Updates

The pose extraction script ([extract_poses_parallel.py](../scripts/extract_poses_parallel.py)) has been updated:

**Old API (removed):**
```python
mp_pose = mp.solutions.pose
with mp_pose.Pose(...) as pose:
    results = pose.process(frame)
```

**New API (current):**
```python
from mediapipe.tasks.python import vision

# Download model first (done in notebook)
model_path = os.environ.get('MEDIAPIPE_POSE_MODEL')

options = vision.PoseLandmarkerOptions(
    base_options=python.BaseOptions(model_asset_path=model_path),
    running_mode=vision.RunningMode.VIDEO
)
landmarker = vision.PoseLandmarker.create_from_options(options)
result = landmarker.detect_for_video(mp_image, timestamp_ms)
```

---

## Performance Expectations

### Processing Speed

| Shot Type | Clips | Time (GPU) | Time (CPU) |
|-----------|-------|------------|------------|
| Smash | 3,872 | ~40 min | ~2 hours |
| Clear | 2,662 | ~30 min | ~1.5 hours |
| Drop | 7,769 | ~1.5 hours | ~4 hours |
| Lift | 5,230 | ~1 hour | ~2.5 hours |
| Drive | 3,998 | ~45 min | ~2 hours |
| **Total** | **23,531** | **~4-6 hours** | **~12+ hours** |

### Storage Requirements

| Item | Size | Location |
|------|------|----------|
| Input videos | 26GB | Colab: `data/videos/clips/` |
| Pose files | ~5GB | Colab: `data/processed/poses/` |
| MediaPipe model | 30MB | Colab: `models/mediapipe/` |
| **Total** | **~31GB** | Fits in Colab free tier |

---

## Troubleshooting

### Issue: "Runtime disconnected"

Colab free tier has a 12-hour maximum runtime. If extraction takes >12 hours:

**Solution:** Process in batches
```python
# Option 1: Download sample first
SAMPLE_SIZE = 20  # 100 clips total for testing

# Option 2: Process one shot type at a time
!python scripts/extract_poses_parallel.py \
    --video-dir data/videos/clips/smash \
    --output-dir data/processed/poses
```

### Issue: "Model file not found"

If pose extraction fails with model path error:

**Solution:** Run MediaPipe setup cell
```python
# Cell 2.1.5: Setup MediaPipe for Colab
# Downloads pose_landmarker_heavy.task
```

### Issue: "Out of memory"

If you run out of RAM:

**Solution:** Reduce workers
```python
# Change from 4 to 2 workers
!python scripts/extract_poses_parallel.py \
    --num-workers 2
```

### Issue: Upload to GCS fails

If backup to GCS has issues:

**Solution:** Use chunked upload
```bash
# Split into smaller batches
gsutil -m cp data/processed/poses/smash_*.pkl gs://iti123storage/features/poses/
```

---

## After Pose Extraction

### Verify Results

```python
# Check pose files
!find data/processed/poses -name "*.pkl" | wc -l
# Should show ~23,500

# Check metadata
!wc -l data/metadata.csv
# Should show ~23,500 rows

# Check distribution
import pandas as pd
df = pd.read_csv('data/metadata.csv')
print(df['stroke_type'].value_counts())
```

### Download Results (Optional)

If you want poses locally:
```bash
# On local machine
gsutil -m rsync -r gs://iti123storage/features/poses/ data/processed/poses/
```

### Next Steps

Continue with **Phase 3: Model Training** in the same notebook:
- Feature extraction from poses
- Train Random Forest and SVM
- 5-class classification (Smash, Clear, Drop, Lift, Drive)
- Expected accuracy: 40-60% (baseline: 20%)

---

## Cost Estimate

**Colab Free Tier:**
- ✅ Sufficient for this task
- Runtime: 4-6 hours < 12-hour limit
- Storage: 31GB < 100GB limit
- GPU: Optional (speeds up by 2-3x)

**GCS Storage:**
- Videos: 26GB × $0.02/GB/month = ~$0.52/month
- Poses: 5GB × $0.02/GB/month = ~$0.10/month
- **Total: ~$0.62/month**

**Data Transfer:**
- Download to Colab: 26GB (free within Google Cloud)
- Upload from Colab: 5GB (free within Google Cloud)

---

## Summary

1. ✅ **Upload clips to GCS** (local → GCS: 2-4 hours)
2. ✅ **Open Colab notebook** (complete_workflow_colab_cleaned.ipynb)
3. ✅ **Run Phase 1** (setup: ~10 minutes)
4. ✅ **Run Phase 2** (pose extraction: ~4-6 hours)
5. ✅ **Continue to Phase 3** (model training)

**Total time:** ~6-10 hours (mostly automated)

---

## References

- **MediaPipe Model:** [Google's official pose landmarker](https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_heavy/float16/latest/pose_landmarker_heavy.task)
- **MediaPipe Docs:** [Pose Landmarker Guide](https://developers.google.com/mediapipe/solutions/vision/pose_landmarker)
- **Script:** [extract_poses_parallel.py](../scripts/extract_poses_parallel.py)
- **Notebook:** [complete_workflow_colab_cleaned.ipynb](../notebooks/complete_workflow_colab_cleaned.ipynb)

---

**Last Updated:** 2026-02-02
**Status:** Ready for Colab execution
