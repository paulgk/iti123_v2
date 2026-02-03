# Shot Type Mapping - Refined for Phase 1.5

**Date:** 2026-02-03
**Purpose:** Reduce shot type mappings to ensure clear, distinct classes for better model training

---

## Problem with Current Mapping

### Issue: Too Many Ambiguous Shots Merged

**Current mapping:** 25,794 shots across 5 classes

**Problems identified:**

1. **DRIVE class has 9 different shot types merged**
   - Push (2,925 shots) - Net shot, wrist-dominated, minimal arm movement
   - Drive (700 shots) - Mid-court, full arm extension, shoulder rotation
   - Rear_Drive (473 shots) - Rear-court position, different biomechanics
   - Defensive_Drive (406 shots) - Reactive, defensive posture
   - Short_Drive (68 shots) - Different power level
   - **Problem:** Model trying to learn "Drive" with 64% Push shots (wrong technique!)

2. **DROP class merges slice variants**
   - Drop (6,290 shots) - Standard linear drop
   - Slice_Drop (2,144 shots) - Requires wrist rotation (different technique)
   - **Problem:** Two distinct wrist mechanics merged into one class

3. **Ambiguous dual-label shots**
   - "Lift / Clear (Defensive)" - Annotators uncertain which shot it was
   - "Drive / Flat Shot" - Multiple names for same shot
   - **Problem:** Annotation uncertainty creates noisy labels

---

## Analysis Summary

### Current Mapping Breakdown

| Class | Total | Main Variants | Problem |
|-------|-------|---------------|---------|
| **Drop** | 8,434 | Drop (75%), Slice_Drop (25%) | Slice has different wrist mechanics |
| **Lift** | 5,632 | Lift (95%), Defensive_Lift (5%) | Clean ✓ |
| **Smash** | 4,234 | Smash (61%), Steep_Smash (39%) | Clean ✓ (same mechanics) |
| **Drive** | 4,572 | Push (64%), Drive (15%), Rear (10%), Defensive (9%) | **64% are Push shots!** ❌ |
| **Clear** | 2,922 | Clear (100%) | Clean ✓ |

**Key finding:** Drive class is 64% Push shots (net shots with different biomechanics)

---

## Proposed Solutions

### Option 1: Conservative (4 Classes) - Maximum Clarity

**Remove Drive entirely, focus on 4 most distinct classes**

```python
CONSERVATIVE_MAPPING = {
    'Smash': 'Smash',           # Overhead offensive
    'Steep_Smash': 'Smash',

    'Clear': 'Clear',           # Overhead defensive
    'Clear (Long)': 'Clear',

    'Drop': 'Drop',             # Overhead deceptive
    'Drop Shot (Soft)': 'Drop',

    'Lift': 'Lift',             # Defensive underhand
    'Defensive_Lift': 'Lift',
}
```

**Results:**
- Total shots: 19,078
- Training samples (with ROI): ~30,524
- Class balance: Excellent (15-33% per class)
- Biomechanical clarity: Maximum ✓✓✓

**Pros:**
- ✅ 4 extremely distinct classes
- ✅ No confusion between shot types
- ✅ Perfect for initial model validation
- ✅ Can add Drive later in Phase 2

**Cons:**
- ❌ Loses Drive class (5th target shot type)
- ❌ Slightly less training data

---

### Option 2: Moderate (5 Classes) - Remove Only Problematic Shots

**Keep 5 classes, but remove ambiguous variants**

```python
MODERATE_MAPPING = {
    # SMASH - Keep both variants (same biomechanics)
    'Smash': 'Smash',
    'Steep_Smash': 'Smash',

    # CLEAR - Keep all (clean labels)
    'Clear': 'Clear',
    'Clear (Long)': 'Clear',

    # DROP - Remove slice variants (different wrist mechanics)
    'Drop': 'Drop',
    'Drop Shot (Soft)': 'Drop',
    # REMOVE: Slice_Drop (2,144 shots) ❌
    # REMOVE: Slice Drop / Cut Drop ❌

    # LIFT - Keep all (clean labels)
    'Lift': 'Lift',
    'Defensive_Lift': 'Lift',
    # REMOVE: Lift / Clear (Defensive) - ambiguous ❌
    # REMOVE: Defensive Lift Return - too specific ❌

    # DRIVE - Keep only pure mid-court drives
    'Drive': 'Drive',
    'Drive / Flat Shot': 'Drive',
    # REMOVE: Push (2,925 shots) - net shot, not drive ❌
    # REMOVE: Push Shot ❌
    # REMOVE: Rear_Drive (473 shots) - different position ❌
    # REMOVE: Rear-Court Drive ❌
    # REMOVE: Defensive_Drive (406 shots) - reactive posture ❌
    # REMOVE: Defensive Drive Return ❌
    # REMOVE: Short_Drive (68 shots) - different power ❌
}
```

**Results:**
- Total shots: 19,778
- Training samples (with ROI): ~31,644
- Class balance: Good (Drive is 3.5%, others 15-32%)
- Biomechanical clarity: High ✓✓

**Removed shots:**
- Slice_Drop: 2,144 shots (different wrist technique)
- Push variants: 2,925 shots (net shots, not mid-court drives)
- Rear_Drive: 473 shots (rear court positioning)
- Defensive_Drive: 406 shots (reactive defensive posture)
- Short_Drive: 68 shots (different power level)
- **Total removed: 6,016 shots (23.3%)**

**Pros:**
- ✅ Keeps all 5 target classes
- ✅ Removes only problematic shots
- ✅ Pure Drive class (only 700 true mid-court drives)
- ✅ Pure Drop class (no slices)
- ✅ More training data than Option 1

**Cons:**
- ⚠️ Drive class is small (3.5% of dataset)
- ⚠️ May need class weighting to balance Drive

---

## Recommended Approach

### ✅ START WITH OPTION 2 (MODERATE - 5 CLASSES)

**Rationale:**

1. **Maintains original 5-class goal**
   - Smash, Clear, Drop, Lift, Drive as planned

2. **Removes only truly problematic shots**
   - Slice_Drop: Different wrist mechanics (not standard drop)
   - Push: Net shot, not mid-court drive (wrong class)
   - Defensive variants: Reactive posture (different from standard)

3. **Clean, unambiguous labels**
   - No dual-label shots (e.g., "Lift / Clear")
   - Each class has clear biomechanical identity
   - Reduces annotation noise

4. **Excellent training data volume**
   - 31,644 usable samples (with ROI extraction)
   - 4x more than current 7K samples
   - Sufficient for deep learning models

5. **Class imbalance is manageable**
   - Drive is only 3.5%, but class weighting solves this
   - Already using weighted loss in training script
   - ST-GCN handles imbalanced data well

---

## Updated Shot Type Mapping

### Final Mapping for Phase 1.5

```python
# scripts/extract_shuttleset_clips.py
# Update SHOT_TYPE_MAPPING to:

SHOT_TYPE_MAPPING = {
    # ===== TARGET CLASSES (5 SHOT TYPES) =====

    # SMASH - Overhead offensive attack
    'Smash': 'Smash',
    'Steep_Smash': 'Smash',  # Same biomechanics, steeper angle

    # CLEAR - Overhead defensive high shot
    'Clear': 'Clear',
    'Clear (Long)': 'Clear',  # Same technique, longer distance

    # DROP - Overhead deceptive soft shot (NO SLICES)
    'Drop': 'Drop',
    'Drop Shot (Soft)': 'Drop',  # Same technique, softer touch

    # LIFT - Defensive underhand high shot
    'Lift': 'Lift',
    'Defensive_Lift': 'Lift',  # Same technique, defensive context

    # DRIVE - Mid-court flat aggressive shot (PURE DRIVES ONLY)
    'Drive': 'Drive',
    'Drive / Flat Shot': 'Drive',  # Alternate name for same shot

    # ===== EXCLUDED - AMBIGUOUS OR DIFFERENT BIOMECHANICS =====

    # Slice variants (different wrist rotation technique)
    'Slice_Drop': None,  # ❌ Wrist rotation - different from standard drop
    'Slice Drop / Cut Drop': None,  # ❌ Ambiguous dual label

    # Push shots (net shots, not mid-court drives)
    'Push': None,  # ❌ Net shot, wrist-dominated, minimal arm movement
    'Push Shot': None,  # ❌ Net shot variant

    # Rear-court drives (different positioning/biomechanics)
    'Rear_Drive': None,  # ❌ Rear court position, defensive setup
    'Rear-Court Drive': None,  # ❌ Same as above

    # Defensive drive variants (reactive, different posture)
    'Defensive_Drive': None,  # ❌ Reactive defensive posture
    'Defensive Drive Return': None,  # ❌ Too specific, defensive context

    # Short drive (different power/distance)
    'Short_Drive': None,  # ❌ Different power level and distance

    # Ambiguous dual-label shots
    'Lift / Clear (Defensive)': None,  # ❌ Annotator uncertain - could be either
    'Defensive Lift Return': None,  # ❌ Too specific context

    # Net shots (Phase 5+)
    'Block': None,
    'Block (Net)': None,
    'Net_Kill': None,
    'Rush / Kill (Net)': None,
    'Cross_Net': None,
    'Cross-Court Net Shot': None,

    # Services (out of scope)
    'Short_Serve': None,
    'Long_Serve': None,

    # Overhead slices (requires IMU sensors)
    'Overhead_Slice': None,
    'Overhead Slice / Cross-Court Drop': None,

    # Unknown shots
    'Unknown': None,
    'Unknown Shot': None,
}
```

---

## Expected Results

### Class Distribution (Clean Mapping)

| Class | Shots | % of Dataset | Samples (ROI) | % of Training |
|-------|-------|--------------|---------------|---------------|
| **Drop** | 6,290 | 31.8% | ~10,064 | 31.8% |
| **Lift** | 5,632 | 28.5% | ~9,011 | 28.5% |
| **Smash** | 4,234 | 21.4% | ~6,774 | 21.4% |
| **Clear** | 2,922 | 14.8% | ~4,675 | 14.8% |
| **Drive** | 700 | 3.5% | ~1,120 | 3.5% |
| **TOTAL** | **19,778** | **100%** | **~31,644** | **100%** |

### Training Data Pipeline

```
19,778 clean shots
  × 2 players per shot (ROI extraction)
= 39,556 total clips

After filtering:
  - Short sequences (<30 frames): ~20% removed → 31,644 clips
  - Multi-player (>60% width): 0% with ROI → 31,644 clips

Final: ~31,644 usable training samples
```

### Expected Model Performance

**With clean data (31K samples):**

| Model | Expected Accuracy | Improvement from Current |
|-------|------------------|--------------------------|
| LSTM | 82-88% | +7-8% (was 75-82%) |
| **ST-GCN** | **88-93%** | +3-8% (was 85-90%) |
| MS-TCN | 87-92% | +5-10% (was 82-88%) |

**Improvements from:**
- ✅ 4x more training data (31K vs 7K)
- ✅ 0% multi-player contamination (ROI prevents merging)
- ✅ Clean labels (no ambiguous shots)
- ✅ Pure class definitions (no biomechanical confusion)
- ✅ Proper normalization (already implemented)

---

## Class Imbalance Handling

### Drive Class is Small (3.5%)

**Already handled in training script:**

```python
# scripts/train_models_fixed.py - Line 491
# Class weighting already implemented

class_weights = compute_class_weight(
    'balanced',
    classes=np.unique(y_train),
    y=y_train
)

# Drive will get ~9x weight vs Drop
# This balances the loss contribution
```

**Why this works:**
- Weighted loss gives Drive 9x importance per sample
- 700 Drive shots × 9 weight = equivalent to 6,300 shots
- Prevents model from ignoring Drive class
- ST-GCN paper shows this works well for imbalanced data

**Alternative if needed:**
- Oversample Drive class during training (duplicate clips)
- But try weighted loss first (cleaner approach)

---

## Migration Steps

### 1. Update Extraction Script

```bash
# Edit scripts/extract_shuttleset_clips.py
# Replace SHOT_TYPE_MAPPING with new version (lines 23-69)

# Verify new mapping
python scripts/extract_shuttleset_clips.py --dry-run

# Should show:
#   Smash: ~4,234 shots
#   Clear: ~2,922 shots
#   Drop:  ~6,290 shots (was 8,434)
#   Lift:  ~5,632 shots
#   Drive: ~700 shots (was 4,572)
```

### 2. Update Documentation

```bash
# Update docs/EXTRACTION_SUMMARY.md
# Update expected shot counts
# Document removed shots and rationale
```

### 3. Re-extract Clips (Phase 1.5)

```bash
# With updated mapping and ROI support
python scripts/extract_shuttleset_clips.py \
    --input ShuttleSet/match \
    --output data/clips \
    --execute

# Expected: 39,556 clips (19,778 × 2 players)
```

### 4. Train with Clean Data

```bash
# Use existing fixed training script
python scripts/train_models_fixed.py \
    --metadata data/metadata.csv \
    --output outputs/ \
    --model stgcn \
    --epochs 50

# Expected: 88-93% accuracy with clean data
```

---

## Validation Checklist

After re-extraction, validate:

### ✅ Mapping Correctness
```python
# Count shots per class
# Verify: Drop = 6,290 (not 8,434)
# Verify: Drive = 700 (not 4,572)
```

### ✅ No Slice Drops
```python
# Check no "Slice_Drop" in output clips
# Verify: All Drop clips are standard linear drops
```

### ✅ No Push Shots in Drive
```python
# Check no "Push" in Drive folder
# Verify: Only pure mid-court drives
```

### ✅ Class Balance
```python
# After training, check class weights applied
# Verify: Drive class not ignored despite 3.5% size
```

### ✅ Model Performance
```python
# After training
# Target: 88-93% accuracy for ST-GCN
# Check: Per-class metrics (Drive should be 70%+)
```

---

## Alternative: If Drive Performance is Poor

If Drive class performs poorly (<70% accuracy) despite weighting:

**Option A: Merge Drive into Smash (offensive overhead)**
```python
# Both are aggressive offensive shots
# Drive → mid-court, Smash → rear-court
# But similar intention (attack)
```

**Option B: Merge Drive into Lift (mid-height contact)**
```python
# Both have mid-height contact points
# But opposite intentions (offensive vs defensive)
```

**Option C: Remove Drive, use 4 classes**
```python
# Fall back to Option 1 (Conservative)
# Focus on 4 most distinct classes
# Add Drive back in Phase 2 with more data
```

**Recommendation:** Try Option 2 first, fall back to Option C only if needed

---

## Summary

### Current Mapping Issues
- ❌ Drive class 64% Push shots (wrong biomechanics)
- ❌ Drop class includes Slice_Drop (different wrist technique)
- ❌ Ambiguous dual-label shots create noise

### Proposed Clean Mapping
- ✅ Remove 6,016 ambiguous shots (23.3%)
- ✅ Pure Drive class (only 700 true mid-court drives)
- ✅ Pure Drop class (no slice variants)
- ✅ 31,644 training samples (4x more than current)
- ✅ Expected: 88-93% accuracy with clean data

### Next Steps
1. Update SHOT_TYPE_MAPPING in extraction script
2. Implement ROI extraction (Phase 1.5)
3. Re-extract with both players (39,556 clips)
4. Train with clean data (31,644 samples)
5. Validate performance (target: 88-93%)

---

**Status:** Ready to implement clean mapping
**Risk:** Low - removes only ambiguous shots
**Expected improvement:** +3-8% accuracy from cleaner labels
