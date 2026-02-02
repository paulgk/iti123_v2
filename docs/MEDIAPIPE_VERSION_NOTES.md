# MediaPipe Version Compatibility Notes

**Date:** 2026-02-02
**Issue:** MediaPipe API changes between versions

---

## Problem

MediaPipe 0.10.x introduced breaking API changes:
- **Old API (0.8.x - 0.9.x):** `mp.solutions.pose.Pose()`
- **New API (0.10.13+):** Task-based API using `mediapipe.tasks.python.vision`

Current scripts use the old `mp.solutions` API which is no longer available in MediaPipe 0.10.13+.

## Available MediaPipe Versions

PyPI currently offers:
- 0.10.32 (latest)
- 0.10.31, 0.10.30
- 0.10.21, 0.10.20, 0.10.18
- 0.10.15, 0.10.14, 0.10.13

**Note:** Versions 0.10.9 and earlier are no longer available on PyPI.

## Current Status

**Local Environment:**
- MediaPipe 0.10.32 installed
- Scripts updated to new API
- **Issue:** Requires manual model file download (pose_landmarker.task ~30MB)
- **Not recommended** for local extraction due to setup complexity

**Colab Environment:**
- Can handle model downloads automatically
- GPU acceleration available
- **Strongly recommended** for pose extraction

---

## Solutions

### Option 1: Run Pose Extraction in Colab (Recommended)

**Why this is better:**
- GPU acceleration available (faster)
- Can install any MediaPipe version
- Handles large dataset better (23,500 clips)
- Already set up in notebooks

**Steps:**
1. Upload clips to GCS: `bash scripts/upload_clips_to_gcs.sh --execute`
2. Run Colab notebook Phase 2 (pose extraction)
3. Downloads clips, extracts poses, uploads results back to GCS

**Colab MediaPipe setup:**
```bash
# In Colab, install compatible version
!pip install mediapipe==0.10.14
# OR use the latest and update scripts to new API
```

### Option 2: Update Local Scripts to New API

**Required changes:**

**Old API (doesn't work):**
```python
import mediapipe as mp
mp_pose = mp.solutions.pose

with mp_pose.Pose(
    static_image_mode=False,
    model_complexity=1,
    min_detection_confidence=0.5
) as pose:
    results = pose.process(frame)
```

**New API (0.10.13+):**
```python
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

# Create PoseLandmarker
base_options = python.BaseOptions(model_asset_path='pose_landmarker.task')
options = vision.PoseLandmarkerOptions(
    base_options=base_options,
    running_mode=vision.RunningMode.VIDEO
)
detector = vision.PoseLandmarker.create_from_options(options)

# Process frame
result = detector.detect_for_video(mp_image, timestamp_ms)
```

**Files that need updating:**
- `scripts/extract_poses.py`
- `scripts/extract_poses_parallel.py`
- `src/data_processing/extract_poses.py`
- Any validation/diagnostic scripts

### Option 3: Use Docker with Specific MediaPipe Version

Create a Docker container with MediaPipe 0.8.x or 0.9.x (if you can find the wheels).

---

## Recommended Workflow

**For your current project:**

1. ✅ **Upload clips to GCS** (ready to go)
   ```bash
   bash scripts/upload_clips_to_gcs.sh --execute
   ```

2. ✅ **Run pose extraction in Colab** (notebooks already updated)
   - Open `notebooks/complete_workflow_colab_cleaned.ipynb`
   - Run Phase 1 (setup)
   - Run Phase 2 (downloads clips from GCS, extracts poses, uploads results)

3. ✅ **Download results** (optional, if needed locally)
   ```bash
   gsutil -m rsync -r gs://iti123storage/features/poses/ data/processed/poses/
   ```

**Benefits:**
- No need to fix local scripts
- Faster processing (Colab GPUs)
- Scales better for 23,500 clips
- Already set up and tested

---

## Future Work

If you need local pose extraction later, consider:
1. Creating a separate Python environment with MediaPipe 0.8.x (if available)
2. Updating all scripts to MediaPipe 0.10.x new API
3. Using Colab exclusively for pose extraction (recommended)

---

## Quick Reference

| Environment | MediaPipe Version | API | Status |
|-------------|------------------|-----|--------|
| **Local Mac** | 0.10.32 | New task-based | Scripts need update |
| **Colab** | Flexible | Both supported | ✅ Recommended |
| **Production** | TBD | New API preferred | Future decision |

---

**Next Step:** Upload clips to GCS and use Colab for pose extraction (4-6 hours on GPU).
