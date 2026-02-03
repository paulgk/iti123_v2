# Multi-Player Detection Issue

**Critical Issue:** Badminton videos contain 2 players, and MediaPipe sometimes detects both players as a single skeleton.

---

## Problem Description

### What Happens

In badminton videos:
- **2 players** are always visible (one on each side of the court)
- MediaPipe's `num_poses=1` setting should detect **only one person**
- But MediaPipe sometimes:
  1. Merges keypoints from **both players** into one skeleton
  2. Creates a "skeleton" that **spans the entire court**
  3. Switches between detecting different players across frames

### Impact on Training

When both players are detected as one:
- **"Body height" is wrong** - spans entire court, not one player
- **Normalization breaks** - dividing by wrong body size
- **High variance** - std=0.59 instead of 0.1-0.3
- **Inconsistent data** - some clips = 1 player, some = 2 players merged

### Visual Example

```
Normal detection (1 player):
Frame width: 1.0
Player span: 0.2-0.3 (20-30% of frame)
Body height: 0.15-0.25

Multi-player detection (2 players merged):
Frame width: 1.0
"Player" span: 0.8-0.9 (80-90% of frame) ❌
Body height: 0.5-0.7 (entire court) ❌
```

### Evidence from Data

Sample analysis of extracted poses:

```
Good clips (single player):
  X-coord range: 0.17 - 0.37
  Body height: 0.15 - 0.25
  ✓ Represents single player

Bad clips (multi-player):
  X-coord range: 0.84 - 0.92 ❌
  Body height: 0.59 - 0.69 ❌
  ⚠️  Represents both players merged!
```

---

## Detection Logic

### How to Identify Multi-Player Clips

Check the spread of x-coordinates across all keypoints:

```python
def is_valid_single_person(pose_sequence, max_width=0.6):
    """
    Check if pose represents a single player

    A single person shouldn't span more than 60% of frame width.
    If x-coordinates span >60%, it's likely both players detected.
    """
    x_coords = pose_sequence[:, :, 0]
    x_range = np.max(x_coords) - np.min(x_coords)

    return x_range < max_width  # 0.6 = 60% of frame
```

### Why 60% Threshold?

**Single player characteristics:**
- Typical span: 15-35% of frame width
- With arm movements: up to 40-50% of frame
- **Safe threshold:** 60% (includes dynamic movements)

**Multi-player merged:**
- Typical span: 70-90% of frame width
- Clearly distinguishable from single player

---

## Solution Implemented

### 1. Filter Function

Added to preprocessing:

```python
def is_valid_single_person(pose_sequence, max_width=0.6):
    x_coords = pose_sequence[:, :, 0]
    x_range = np.max(x_coords) - np.min(x_coords)
    return x_range < max_width
```

### 2. Updated Preprocessing Pipeline

**Before (broken):**
```python
# Load → Normalize → Train
for pose in poses:
    normalized = normalize_pose(pose)  # Wrong normalization
    X.append(normalized)
```

**After (fixed):**
```python
# Load → Filter short → Filter multi-player → Normalize → Train
for pose in poses:
    if len(pose) < 30:  # Skip short clips
        continue

    if not is_valid_single_person(pose):  # Skip multi-player
        continue

    normalized = normalize_pose(pose)  # Correct normalization
    X.append(normalized)
```

### 3. Updated Files

1. **notebooks/model_comparison_colab.ipynb** - Cell 13
   - Added `is_valid_single_person()` function
   - Added multi-player filtering before normalization
   - Added statistics tracking

2. **scripts/train_models_fixed.py** - Lines 92-107, 478-479
   - Added `is_valid_single_person()` function
   - Integrated into loading pipeline

---

## Expected Impact

### Dataset Size

**Before filtering:**
- Original: 13,858 samples
- After short filter: ~7,988 samples

**After multi-player filtering:**
- Expected: ~6,500-7,000 samples (estimate)
- Filtered: ~1,000-1,500 multi-player clips (10-15%)

### Data Quality

**Before:**
- Mean: -0.03 ✓ (good)
- Std: 0.59 ⚠️ (too high)
- Reason: Multi-player clips inflating variance

**After:**
- Mean: ~0.0 ✓ (good)
- Std: 0.15-0.35 ✓ (improved)
- Cleaner, more consistent data

### Training Performance

**Expected improvement:**
- **Before:** 75-82% accuracy (with noisy data)
- **After:** 80-88% accuracy (with clean data)
- **Improvement:** +3-6% from cleaner training data

---

## Verification

### Check Your Data

After applying the filter, verify:

```python
# Should see in preprocessing output:
print("Filtering multi-player detections...")
print(f"Filtered {N} multi-player clips (x-range > 60%)")

# Check normalization stats:
print(f"Mean: {X.mean():.4f}")  # Should be ~0.0
print(f"Std: {X.std():.4f}")    # Should be 0.15-0.35 (not 0.59)
```

### Sample Analysis

Pick random samples and check:

```python
import random
sample_idx = random.randint(0, len(X))
sample = X[sample_idx]

x_range = sample[:, :, 0].max() - sample[:, :, 0].min()
print(f"X-range: {x_range:.3f}")

if x_range > 0.6:
    print("⚠️  Multi-player clip slipped through!")
else:
    print("✓ Valid single player clip")
```

---

## Alternative Solutions (Not Implemented)

### Option 1: Re-extract with Higher Confidence

**Approach:** Increase MediaPipe confidence thresholds

```python
# In extract_poses_parallel.py
options = vision.PoseLandmarkerOptions(
    min_pose_detection_confidence=0.7,  # Increase from 0.5
    min_pose_presence_confidence=0.7,
    min_tracking_confidence=0.7,
)
```

**Pros:** Better single-player detection at source
**Cons:** Takes 40-50 minutes to re-extract all clips
**Status:** Not needed if filtering works well

### Option 2: Crop to Primary Player

**Approach:** Identify and extract the most active player

```python
def crop_to_primary_player(pose_sequence):
    # Split frame into left/right regions
    # Calculate motion in each region
    # Keep the region with maximum variance
    pass
```

**Pros:** Recovers more data (doesn't discard multi-player clips)
**Cons:** Complex implementation, may introduce errors
**Status:** Can implement later if dataset too small

### Option 3: Use Region-Based Detection

**Approach:** Tell MediaPipe to only look at one side of court

```python
# Define region of interest (ROI)
roi = [0.0, 0.0, 0.5, 1.0]  # Left half of frame only
```

**Pros:** Cleanest solution at extraction time
**Cons:** Requires re-extraction with modified script
**Status:** Good for future datasets

---

## Recommendations

### Current Approach: Filtering (Implemented)

✅ **Use this now:**
- Quick to implement
- No re-extraction needed
- Effective for current dataset

### Future Improvement: Region-Based Detection

🔮 **For next dataset:**
- Modify extraction script to use ROI
- Extract left player and right player separately
- Double your dataset size (2 samples per video)

### Long-term: Ensemble Approach

🚀 **For production:**
- Train separate models for different camera angles
- Ensemble predictions from multiple models
- Handle edge cases better

---

## FAQ

### Q: Why not just use all the data?

**A:** Multi-player detections break normalization. The model would learn "a smash spans the entire court" which is wrong.

### Q: Isn't 60% threshold too strict?

**A:** No. Single players rarely span >50% even with full arm extension. 60% includes safety margin.

### Q: What if I lose too much data?

**A:** You can adjust threshold to 0.65 or 0.7, but check std doesn't go back up to 0.5+.

### Q: Should I re-extract with higher confidence?

**A:** Only if filtering removes >20% of your data. Current 10-15% loss is acceptable.

### Q: Can I recover the filtered clips?

**A:** Yes, implement Option 2 (crop to primary player), but it's complex. Try filtering first.

---

## Testing

Verify the fix works:

```bash
# 1. Check preprocessing output
#    Should see: "Filtered N multi-player clips"

# 2. Check normalization stats
#    Mean: ~0.0, Std: 0.15-0.35 (not 0.59)

# 3. Train model
#    Should achieve 80-88% accuracy (not 30%)

# 4. Check training curves
#    Loss should decrease smoothly to 0.3-0.5
```

---

## Files Modified

1. **notebooks/model_comparison_colab.ipynb**
   - Cell 13: Added `is_valid_single_person()` and filtering logic

2. **scripts/train_models_fixed.py**
   - Lines 92-107: Added `is_valid_single_person()` function
   - Lines 478-479: Added filtering in loading loop

3. **docs/MULTI_PLAYER_ISSUE.md** (this file)
   - Documentation of issue and solution

---

## Related Issues

- [FIXES_APPLIED.md](FIXES_APPLIED.md) - Overall training fixes
- [TRAINING_WORKFLOW.md](TRAINING_WORKFLOW.md) - Complete workflow guide
- [CONDA_SETUP_GUIDE.md](CONDA_SETUP_GUIDE.md) - Local extraction setup

---

**Status:** Fix implemented and ready for testing
**Expected:** Std improves from 0.59 → 0.15-0.35, accuracy improves +3-6%
