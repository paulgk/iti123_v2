# Colab Training Guide - Video-Based Classification

**Notebook:** `notebooks/badminton_video_training_colab.ipynb`

**Expected:** 70-80% accuracy (vs 38% with pose-only)

---

## Quick Start

### 1. Upload Notebook to Colab

1. Go to [Google Colab](https://colab.research.google.com/)
2. File > Upload notebook
3. Select `badminton_video_training_colab.ipynb`

### 2. Set Runtime to GPU

1. Runtime > Change runtime type
2. Hardware accelerator: **GPU**
3. GPU type: **T4** (free tier)
4. Save

### 3. Run All Cells

- Runtime > Run all
- Downloads clips from GCS (~5-10 minutes)
- Trains model (~4-6 hours)

---

## Data Download from GCS

The notebook includes automatic download from `gs://iti123storage/videos/clips/`

**What happens:**
```
Downloading from gs://iti123storage/videos/clips/...
This will take ~5-10 minutes for 18,167 clips
======================================================================
✓ Download complete!
  Clear   :  2662 clips
  Drive   :   630 clips
  Drop    :  5773 clips
  Lift    :  5230 clips
  Smash   :  3872 clips

Total: 18167 clips downloaded

✓ DATA_ROOT set to: /content/data/clips
```

**If authentication needed:**
Uncomment in GCS download cell:
```python
from google.colab import auth
auth.authenticate_user()
```

---

## Expected Results

**Training progression:**
- Epoch 1: ~40% accuracy
- Epoch 10: ~65% accuracy
- Epoch 30: ~72% accuracy (best)

**Final test results:**
- Overall accuracy: 70-80%
- F1 score: 0.65-0.75
- Per-class:
  - Drop/Lift/Smash: 70-80%
  - Clear: 60-70%
  - Drive: 45-60%

---

## Files Downloaded

After training completes:
1. `best_model.pth` - Trained model
2. `training_history.csv` - Metrics per epoch
3. `training_curves.png` - Loss/accuracy plots
4. `confusion_matrix.png` - Per-class results

---

## Troubleshooting

**GPU not available:**
- Runtime > Change runtime type > GPU > Save

**GCS download fails:**
- Uncomment authentication in cell
- Or use Google Drive (Option B)

**Out of memory:**
- Reduce batch_size to 16 in config

---

**Last Updated:** 2026-02-04
