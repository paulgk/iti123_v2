# Training Workflow Guide

**Complete workflow from pose extraction to trained models**

---

## Overview

This guide covers the end-to-end workflow for training badminton shot classification models.

**Steps:**
1. Extract pose features from videos
2. Upload to GCS
3. Train models in Colab
4. Save outputs to git

**Total time:** ~3-5 hours (depends on dataset size and GPU)

---

## Step 1: Pose Extraction (Local)

### Prerequisites
- Conda environment set up (see [CONDA_SETUP_GUIDE.md](CONDA_SETUP_GUIDE.md))
- Video clips in `data/clips/`
- MediaPipe model downloaded

### Run Extraction

```bash
# Activate conda environment
conda activate iti123

# Run parallel extraction (8 workers)
python scripts/extract_poses_parallel.py \
    --video-dir data/clips/ \
    --output-dir data/processed/poses/ \
    --num-workers 8

# Or use the training script directly
python scripts/train_models_fixed.py \
    --metadata data/data/metadata.csv \
    --output outputs/ \
    --model stgcn
```

**Expected time:** ~40-50 minutes for 23,531 clips (10-core Mac)

**Output:**
- Pose files: `data/processed/poses/*.pkl`
- Metadata: `data/data/metadata.csv`

### Verify Extraction

```python
import pickle
import numpy as np
from pathlib import Path

# Check a sample pose
pose_files = list(Path('data/processed/poses').glob('*.pkl'))
with open(pose_files[0], 'rb') as f:
    pose = pickle.load(f)

print(f"Shape: {pose.shape}")  # Should be (T, 33, 3)
print(f"Mean: {pose.mean():.3f}")  # Should be 0.3-0.7 (unnormalized)
print(f"Contains NaN: {np.isnan(pose).any()}")  # Should be False
```

---

## Step 2: Upload to GCS

### Upload Pose Features

```bash
bash upload_poses_to_gcs.sh
```

**Expected time:** ~10-15 minutes for 13,858 pose files

**Uploads to:**
- Poses: `gs://iti123storage/features/poses/`
- Metadata: `gs://iti123storage/data/metadata.csv`

### Verify Upload

```bash
# Count remote files
gsutil ls gs://iti123storage/features/poses/*.pkl | wc -l

# Should match local count
find data/processed/poses -name "*.pkl" | wc -l
```

---

## Step 3: Train Models in Colab

### Option A: Use Fixed Notebook (Recommended)

1. Open [model_comparison_colab.ipynb](../notebooks/model_comparison_colab.ipynb) in Colab
2. **Enable GPU:** Runtime → Change runtime type → T4 GPU
3. Run all cells

**Key changes in fixed notebook:**
- ✅ Proper normalization (torso-centered, height-scaled)
- ✅ Learning rate: 0.001 (was 0.0001)
- ✅ Filter short sequences (<30 frames)
- ✅ Normalized adjacency matrix for ST-GCN
- ✅ Class-weighted loss

### Option B: Use Standalone Script

```bash
# Train locally (if you have GPU)
python scripts/train_models_fixed.py \
    --metadata data/data/metadata.csv \
    --output outputs/ \
    --model stgcn \
    --epochs 50

# Or train LSTM
python scripts/train_models_fixed.py \
    --metadata data/data/metadata.csv \
    --output outputs/ \
    --model lstm \
    --epochs 50
```

**Expected time:**
- Colab (T4 GPU): ~30-45 minutes per model
- Local (CPU): ~2-3 hours per model
- Local (M1 GPU): ~1-2 hours per model

**Expected accuracy:**
- LSTM: 75-82%
- ST-GCN: 85-90% ⭐
- MS-TCN: 82-88%

---

## Step 4: Save Outputs to Git

### Download from Colab

In Colab, run:

```python
# Package outputs
!zip -r training_outputs.zip outputs/

# Download
from google.colab import files
files.download('training_outputs.zip')
```

### Extract and Save to Repo

```bash
# Extract downloaded file
unzip training_outputs.zip -d /path/to/extracted

# Save to git
bash scripts/save_outputs_to_git.sh /path/to/extracted/outputs

# Script will:
# 1. Copy models to outputs/models/
# 2. Copy reports to outputs/reports/
# 3. Copy visualizations to outputs/visualizations/
# 4. Stage files in git
# 5. Prompt for commit
```

### Commit and Push

```bash
# Commit is done automatically by script
# Just push to remote
git push origin main
```

---

## Expected Outputs

### Models

```
outputs/models/
├── LSTM_best.pth         (~2.5 MB)
├── STGCN_best.pth        (~3.2 MB)
└── MSTCN_best.pth        (~1.5 MB)
```

### Reports

```
outputs/reports/
├── model_comparison.csv
├── model_comparison_summary.txt
├── LSTM_report.txt
├── STGCN_report.txt
└── MSTCN_report.txt
```

### Visualizations

```
outputs/visualizations/
├── training_curves.png
├── confusion_matrices.png
└── per_class_performance.png
```

---

## Troubleshooting

### Issue: Low Accuracy (~30% instead of 85%)

**Cause:** Normalization not applied or learning rate too low

**Solution:**
1. Check preprocessing cell in notebook
2. Verify normalization check shows:
   - Mean: ~0.0 (not 0.3-0.7)
   - Std: ~0.1-0.3
3. Verify learning rate is 0.001 in training function

### Issue: Training Loss Not Decreasing

**Cause:** Learning rate too low or data not normalized

**Solution:**
1. Use fixed training script: `scripts/train_models_fixed.py`
2. Or use fixed notebook with corrections
3. Check training curves - loss should drop from 1.6 → 0.3-0.5

### Issue: Colab Keeps Crashing

**Cause:** Long-running session, memory issues, or runtime disconnects

**Solution:**
1. Use local extraction instead (faster and more stable)
2. Enable Colab Pro for longer runtimes
3. Use auto-clicker to keep session alive (Safari extension)

### Issue: Git Won't Commit Large Files

**Cause:** Model files too large (>100 MB)

**Solution:**
1. Git tracks only `*_best.pth` files (see `.gitignore`)
2. Large files are backed up to GCS automatically
3. Use `save_outputs_to_git.sh` script (handles size filtering)

---

## Quick Reference Commands

```bash
# Extract poses locally
conda activate iti123
python scripts/extract_poses_parallel.py --video-dir data/clips/ --output-dir data/processed/poses/ --num-workers 8

# Upload to GCS
bash upload_poses_to_gcs.sh

# Train model (local)
python scripts/train_models_fixed.py --metadata data/data/metadata.csv --output outputs/ --model stgcn

# Save outputs to git
bash scripts/save_outputs_to_git.sh outputs

# Push to remote
git push origin main
```

---

## Next Steps

After training:

1. **Deploy model** - Use best model for production
2. **Create ensemble** - Combine top 2 models for maximum accuracy
3. **Fine-tune** - Add sport-specific augmentation
4. **Integrate** - Connect to badminton analysis application

---

**Last updated:** 2026-02-03
**Status:** Production ready with normalization fixes
