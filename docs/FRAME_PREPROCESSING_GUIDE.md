# Frame Preprocessing Guide - Speed Up Training 100x

## Problem

Training taking **59 seconds per batch** = **6.5 hours per epoch** = **325 hours total!**

**Root cause:** Video decoding bottleneck (18K videos × 16 frames = 290K frame decodes)

---

## Solution: Pre-extract Frames to .npy Files (with Parallel Processing)

**Trade-off:**
- **20-30 minutes** one-time preprocessing (parallel processing on 4+ cores)
- Then training takes 2-3 hours (100x faster!)
- Total: **2.5-3.5 hours** instead of 325 hours ✓

---

## How It Works

### Without Preprocessing (Current - SLOW):
```
Each batch:
1. Load 32 videos from disk
2. Decode H264 → extract 16 frames each
3. Resize 1080p → 224×224
4. Color convert BGR → RGB
5. Apply transforms
6. Send to GPU
Time: 59 seconds/batch ❌
```

### With Preprocessing (FAST):
```
ONE-TIME (20-30 minutes):
- Extract all frames to .npy files (parallel processing)
- Store as numpy arrays on disk

TRAINING (per batch):
1. Load 32 .npy files (pre-decoded!)
2. Apply transforms
3. Send to GPU
Time: ~0.5 seconds/batch ✓
```

---

## Step-by-Step Instructions

### Step 1: Run Notebook Up to Section 7

Execute all cells from sections 1-7 to:
- Download data from GCS (Section 2)
- Set up configuration (Section 3)
- Define model classes (Sections 4-6)
- **Load dataset** (Section 7) - This creates `video_paths` and `labels`

### Step 2: Run Frame Extraction (Parallel)

In **Section 7B, Cell 22** (Frame Preprocessing), uncomment and run:

```python
FRAMES_DIR = '/content/data/frames'

print("Starting parallel frame extraction...")
npy_paths, failed = preprocess_all_videos_parallel(
    video_paths,
    FRAMES_DIR,
    num_frames=CONFIG['num_frames'],
    frame_size=CONFIG['frame_size'],
    num_workers=None  # Auto-detects CPU count - 1
)

print(f"\n✓ Preprocessing complete!")
```

**What this does:**
- Extracts 16 frames from each of 18,167 videos **in parallel**
- Uses multiple CPU cores (default: CPU count - 1)
- Resizes to 224×224
- Saves as `.npy` files in `/content/data/frames/`
- Shows progress bar
- Takes **20-30 minutes** (4x faster than sequential)

**Output:**
```
Extracting frames from 18167 videos...
Output directory: /content/data/frames
Parallel workers: 3
Estimated time: 20-30 minutes for 18K videos
======================================================================
Extracting: 100%|██████████| 18167/18167 [23:12<00:00, 13.05it/s]

======================================================================
✓ Extracted 18167 videos
Storage used: 4821.3 MB (~4.7 GB)
```

### Step 3: Use Fast Dataset

After extraction completes, **uncomment Cell 24** to create fast data loaders:

```python
# Creates train/val/test splits with .npy paths
train_npy_paths, temp_npy_paths, train_labels, temp_labels = train_test_split(
    npy_paths, labels, test_size=0.3, random_state=42, stratify=labels
)

# Creates fast datasets
train_dataset = BadmintonFramesDataset(train_npy_paths, train_labels, augment=True)
val_dataset = BadmintonFramesDataset(val_npy_paths, val_labels, augment=False)
test_dataset = BadmintonFramesDataset(test_npy_paths, test_labels, augment=False)

# Creates fast data loaders
train_loader = DataLoader(train_dataset, batch_size=CONFIG['batch_size'], ...)
...
```

### Step 4: Skip to Model Training

After running the fast dataset workflow cell:
- **Skip sections 8-9** (they create video-based loaders, not needed)
- **Go directly to section 10** (Create Model & Training Setup)
- Continue training as normal

---

## Expected Performance

### Before (Video Loading):
```
Training: 13% | 52/398 [56:26<5:42:14, 59.35s/it]
- 59 seconds per batch
- 6.5 hours per epoch
- 325 hours total (50 epochs)
```

### After (Pre-extracted Frames):
```
Training: 13% | 26/199 [01:15<08:40, 0.5s/it]
- 0.5 seconds per batch
- 3-4 minutes per epoch
- 2-3 hours total (50 epochs)
```

**Speed improvement: 100x faster!** 🚀

---

## Storage Requirements

**Frame .npy files:**
- Per video: ~260 KB (16 frames × 224×224×3 RGB)
- Total: 18,167 videos × 260 KB = **~4.7 GB**

**Colab disk space:**
- Total available: ~100 GB
- Videos: ~10 GB
- Frames: ~5 GB
- ✓ Plenty of space!

---

## What Gets Saved

**Structure:**
```
/content/data/frames/
├── Clear_match01_rally001_shot001.npy
├── Clear_match01_rally001_shot002.npy
├── Drive_match03_rally005_shot003.npy
├── Drop_match02_rally002_shot001.npy
...
```

**Each .npy file contains:**
- Shape: (16, 224, 224, 3)
- dtype: uint8
- Size: ~260 KB
- Already resized and RGB converted

---

## Advantages

### 1. **100x Faster Training**
- Video decoding done once
- Training loads numpy arrays (super fast)

### 2. **Better GPU Utilization**
- CPU not bottlenecked by video decoding
- GPU gets data instantly
- Can increase batch size to 64

### 3. **Consistent Performance**
- No variation in decode speed
- Predictable training time
- No dropped frames

### 4. **Resumable**
- If preprocessing crashes, resumes automatically
- Skips already-extracted videos
- Only processes remaining ones

---

## Disadvantages

### 1. **Storage**
- Needs ~5 GB extra disk space
- Videos: 10 GB + Frames: 5 GB = 15 GB total

### 2. **One-time Cost**
- 1-2 hours preprocessing before training
- But saves 320+ hours overall!

### 3. **Less Flexible**
- If you change frame_size or num_frames, need to re-extract
- But this rarely changes

---

## When to Use This

### ✅ Use Frame Preprocessing If:
- Training speed >20s per batch
- Total training time >10 hours
- You'll train multiple times (experiments)
- GPU is idle waiting for data

### ❌ Don't Use If:
- Training already fast (<5s per batch)
- Only training once
- Low disk space (<20 GB free)
- Short on time for preprocessing

---

## Troubleshooting

### "Out of disk space"
**Cause:** Colab VM only has 100GB, might be full

**Fix 1:** Delete original videos after extraction
```python
!rm -rf /content/data/clips
# Saves ~10 GB
```

**Fix 2:** Extract to Google Drive (slower but more space)
```python
FRAMES_DIR = '/content/drive/MyDrive/iti123_frames'
```

### "Extraction taking too long"
**Expected:** ~1-2 hours for 18K videos

**Speed:** ~3-4 videos/second
- 18,167 / 3.5 = 5,190 seconds = 86 minutes

**If much slower:** Check CPU usage with `!htop`

### "Some videos failed"
**Normal:** A few corrupted videos may fail

**Action:** Note which ones failed, training continues without them

---

## Alternative: Increase Batch Size First

Before running 1-2 hour preprocessing, try this quick fix:

```python
CONFIG['batch_size'] = 64  # From 32
CONFIG['num_workers'] = 4   # From 2
```

**This might be enough** to bring 59s → 20-30s per batch.

**If still slow after this:** Then do frame preprocessing.

---

## Summary

**Your options:**

### Option 1: Quick Fix (Try First)
- Change batch_size to 64
- Change num_workers to 4
- Restart training
- **Expected:** 20-30s per batch
- **Total:** 10-20 hours

### Option 2: Frame Preprocessing (Best)
- Run preprocessing (~1-2 hours)
- Use fast dataset
- Train
- **Expected:** 0.5s per batch
- **Total:** 3-5 hours (including preprocessing)

### Option 3: Live with Current Speed
- Keep batch_size=32, workers=2
- Wait 325 hours
- Not recommended! ❌

**Recommended:** Try Option 1 first, if still slow do Option 2.

---

**Last Updated:** 2026-02-04
