# Improvement Suggestions Evaluation & Test Plan

**Date**: 2026-01-15
**Dataset**: MoveNet poses (19 clips: 13 train, 3 val, 3 test)
**Current Performance**: ~50% accuracy (random guessing level)

---

## 1. Balance the Splits

### Current Issue
- **Validation**: 67% Smash, 33% Clear (2 Smash, 1 Clear)
- **Test**: 67% Clear, 33% Smash (2 Clear, 1 Smash)
- **Training**: 54% Smash, 46% Clear (7 Smash, 6 Clear)

### Evaluation
**Impact**: ⚠️ **CRITICAL for current dataset**
- With only 3 samples in val/test, the imbalance is severe
- Models will show wildly different performance depending on which set is used
- Class weights can't compensate for such small samples

**Root Cause**: Only 1 match in the 19 clips → forced clip-level split
- Need clips from multiple matches for proper stratification
- Current workaround breaks match-level data leakage prevention

### Recommendation
**For 19-clip subset**:
- ❌ **Cannot fix** - need more data from multiple matches
- Current split is as good as it gets with 1 match

**For full dataset**:
- ✅ **MUST implement** group-stratified split by (match_id, stroke_type)
- Use `StratifiedGroupKFold` from sklearn

### Tests Required
```python
# Test 1: Verify current imbalance
def test_class_balance():
    # Load splits, check class distribution per split
    # Assert: train/val/test within ±10% of overall distribution
    pass

# Test 2: Implement group-stratified split
def test_group_stratified_split():
    # Use StratifiedGroupKFold with match_id as groups
    # Verify: no match appears in multiple splits
    # Verify: class balance maintained across splits
    pass
```

**Priority**: 🔴 **HIGH** (for full dataset with multiple matches)

---

## 2. Handle Padding Explicitly

### Current Issue
- Sequences padded to length=50 with zeros
- After normalization: padded frames become non-zero (mean/std applied)
- LSTM/GRU learn padding artifacts instead of masking them

### Evaluation
**Impact**: 🟡 **MEDIUM-HIGH**
- Models waste capacity learning padding patterns
- Shorter sequences get diluted by padding noise
- Affects gradient flow and training stability

**Evidence from current data**:
```python
# From data_split.py line 257-265:
# Padding is added but lengths are lost
if num_frames < target_length:
    padding = np.zeros((target_length - num_frames, num_features))
    features = np.vstack([features, padding])
```

After global normalization:
```
Sequence mean range: [-0.0383, 154.4777]
Sequence std range:  [0.0000, 87.8025]
```
→ Padded zeros become `(-0.0383 - mean) / std` (non-zero values)

### Recommendation
✅ **IMPLEMENT - Clear benefit**

Steps:
1. Save sequence lengths in split files
2. Apply normalization BEFORE padding
3. Pad with zeros AFTER normalization
4. Add `Masking(mask_value=0.0)` layer in LSTM models

### Tests Required
```python
# Test 1: Verify padding is properly masked
def test_padding_mask():
    # Create sample with known length
    # Verify: padded positions are exactly 0.0 after normalization
    # Verify: Masking layer correctly ignores padded timesteps
    pass

# Test 2: Compare model performance with/without masking
def test_masking_impact():
    # Train identical model with/without Masking layer
    # Expect: masked version has better validation F1
    pass

# Test 3: Verify gradient flow
def test_masked_gradients():
    # Check that gradients are 0 for padded timesteps
    pass
```

**Priority**: 🟠 **MEDIUM-HIGH** (should implement)

---

## 3. Tune Decision Threshold

### Current Issue
- Fixed threshold=0.5 for binary classification
- Optimal threshold may differ due to class imbalance
- Not maximizing F1 score potential

### Evaluation
**Impact**: 🟡 **MEDIUM**
- Can improve F1 by 5-15% without retraining
- Especially useful with imbalanced classes
- Quick win with minimal effort

**Current approach** (baseline_model.py line ~180):
```python
# Uses default threshold=0.5
y_pred = (y_pred_proba >= 0.5).astype(int)
```

### Recommendation
✅ **IMPLEMENT - Easy improvement**

Steps:
1. Keep sigmoid outputs (already done)
2. Sweep thresholds 0.1 to 0.9 in steps of 0.05 on validation set
3. Select threshold that maximizes F1
4. Report test metrics at optimal threshold

### Tests Required
```python
# Test 1: Threshold tuning on validation set
def test_threshold_tuning():
    # Sweep thresholds, find optimal F1
    # Verify: optimal threshold != 0.5 for imbalanced data
    pass

# Test 2: Compare fixed vs tuned threshold
def test_threshold_impact():
    # Compare F1 at 0.5 vs optimal threshold
    # Expect: 5-15% F1 improvement
    pass

# Test 3: Plot precision-recall curve
def test_pr_curve():
    # Visualize precision-recall tradeoff
    # Identify optimal operating point
    pass
```

**Priority**: 🟢 **LOW-MEDIUM** (nice-to-have, quick win)

---

## 4. Stabilize Training

### Current Issue
Potential training instability:
- Default learning rate may be too high
- No gradient clipping
- Small dataset → high variance in gradients

### Evaluation
**Impact**: 🟡 **MEDIUM** (for small dataset)

**Current hyperparameters** (likely defaults):
- LR: ~1e-3 (Adam default)
- No gradient clipping
- Hidden size: 128 (for 19 samples, may be too large)
- Batch size: Unknown (likely 32)

**For 19 training samples**:
- Batch size 32 → only 1 batch per epoch → high variance
- Model may overfit quickly
- Need smaller capacity or regularization

### Recommendation
✅ **IMPLEMENT for current dataset**

Suggested changes:
```python
# Lower learning rate
optimizer = Adam(learning_rate=3e-4, clipnorm=1.0)

# Smaller model for small dataset
hidden_size = 64  # down from 128
dropout = 0.3     # increased regularization

# Smaller batch size
batch_size = 4    # for 13 training samples
```

### Tests Required
```python
# Test 1: Learning rate sensitivity
def test_learning_rates():
    # Try: [1e-4, 3e-4, 1e-3, 3e-3]
    # Plot: training/val loss curves
    # Select: best validation F1
    pass

# Test 2: Gradient clipping impact
def test_gradient_clipping():
    # Train with/without clipnorm=1.0
    # Check: gradient magnitudes, training stability
    pass

# Test 3: Model capacity
def test_model_capacity():
    # Try hidden sizes: [32, 64, 128]
    # Check: overfitting (train vs val gap)
    pass

# Test 4: Batch size effect
def test_batch_sizes():
    # Try: [4, 8, 16] for 13 samples
    # Monitor: convergence speed, final performance
    pass
```

**Priority**: 🟠 **MEDIUM** (especially for small dataset)

---

## 5. Temporal Augmentation

### Current Issue
- No data augmentation
- Small dataset (19 clips) → severe overfitting risk
- Models can't generalize beyond exact training sequences

### Evaluation
**Impact**: 🟢 **HIGH for small dataset, LOW for large dataset**

**Why it matters for 19 clips**:
- 13 training samples is tiny for deep learning
- Augmentation can 5-10x effective dataset size
- Helps LSTM learn robust temporal patterns

**Proposed augmentations**:
1. **Frame dropout**: Randomly drop 10-20% of frames
2. **Gaussian noise**: Add σ=0.05 to velocities
3. **Time warping**: Stretch/compress by ±10%
4. **Random shift**: Shift sequence start by ±5 frames

### Recommendation
✅ **IMPLEMENT - Critical for 19-clip dataset**
⚠️ **Lower priority for full dataset**

### Tests Required
```python
# Test 1: Augmentation preserves class
def test_augmentation_validity():
    # Apply augmentations, verify sequences still look reasonable
    # Check: no NaNs, magnitudes in expected range
    pass

# Test 2: Augmentation diversity
def test_augmentation_diversity():
    # Apply 10x to same clip
    # Verify: sufficiently different sequences
    pass

# Test 3: Performance impact
def test_augmentation_impact():
    # Train with/without augmentation
    # Expect: 10-20% improvement on validation
    pass

# Test 4: Overfitting reduction
def test_overfitting_reduction():
    # Compare train/val gap with/without augmentation
    # Expect: smaller gap with augmentation
    pass
```

**Priority**: 🔴 **HIGH** (for current 19-clip dataset)

---

## 6. Enrich Signals

### Current Issue
Current features (42 per frame):
- Position, depth, arm extension
- Velocities (first derivative)
- ❌ Missing: Acceleration (second derivative)
- ❌ Missing: Momentum/magnitude cues
- ❌ Missing: Local motion patterns

### Evaluation
**Impact**: 🟡 **MEDIUM-HIGH**

**Why acceleration matters**:
- Smash: Rapid acceleration → peak velocity → deceleration
- Clear: Smoother acceleration curve
- Current velocity features miss this distinction

**Why Conv1D helps**:
- Captures local temporal patterns (3-5 frame windows)
- More robust to frame timing variations
- Reduces sequence length for LSTM → faster training

**From previous analysis**:
- Current features show Cohen's d < 0.1 (no separation)
- Adding acceleration may reveal differences in swing dynamics

### Recommendation
✅ **IMPLEMENT - Likely to help**

Steps:
1. Add acceleration features (diff of velocities)
2. Add momentum magnitude: `||velocity|| * mass_proxy`
3. Add Conv1D front-end:
   ```python
   Conv1D(64, kernel=3) → ReLU → MaxPool
   Conv1D(64, kernel=3) → ReLU → MaxPool
   → BiLSTM(64) → Dense
   ```

### Tests Required
```python
# Test 1: Feature separability
def test_acceleration_separability():
    # Compute Cohen's d for acceleration features
    # Expect: d > 0.3 for at least some features
    pass

# Test 2: Compare feature sets
def test_feature_ablation():
    # Train on: (a) position only, (b) +velocity, (c) +acceleration
    # Check: incremental F1 improvement
    pass

# Test 3: Conv1D architecture
def test_conv1d_frontend():
    # Compare: LSTM-only vs Conv1D+LSTM
    # Check: validation F1, training time
    pass

# Test 4: Momentum features
def test_momentum_features():
    # Add wrist/elbow momentum magnitude
    # Check: Cohen's d, model performance
    pass
```

**Priority**: 🟠 **MEDIUM** (worth trying)

---

## Summary & Prioritization

### For Current 19-Clip Dataset

**Immediate Actions** (1-2 days):
1. 🔴 **Temporal Augmentation** - Critical for small dataset
2. 🟠 **Training Stabilization** - LR=3e-4, clipnorm, smaller model
3. 🟠 **Padding Masking** - Easy fix, clear benefit
4. 🟢 **Threshold Tuning** - Quick win, 5 minutes

**Optional Improvements** (if time permits):
5. 🟡 **Acceleration Features** - May reveal swing dynamics
6. 🟡 **Conv1D Frontend** - Worth trying

**Cannot Fix**:
- ❌ Class balance - need more matches

### For Full Dataset (when ready)

**Critical**:
1. 🔴 **Group-Stratified Split** - Match-level + class balance
2. 🔴 **Padding Masking** - Essential for proper training
3. 🟠 **Acceleration Features** - Capture swing dynamics

**Nice-to-Have**:
4. 🟢 **Threshold Tuning** - Easy improvement
5. 🟢 **Training Stabilization** - Less critical with more data
6. 🟡 **Temporal Augmentation** - Lower priority with full data
7. 🟡 **Conv1D Frontend** - Experiment

---

## Test Execution Plan

### Phase 1: Quick Wins (Day 1)
```bash
# 1. Implement padding masking
python tests/test_padding_mask.py

# 2. Tune decision threshold
python tests/test_threshold_tuning.py

# 3. Stabilize training hyperparameters
python tests/test_training_stability.py
```

### Phase 2: Data Augmentation (Day 2)
```bash
# 4. Implement temporal augmentation
python tests/test_augmentation.py

# 5. Retrain with augmentation
python src/models/lstm_model.py --augment
```

### Phase 3: Feature Engineering (Day 3)
```bash
# 6. Add acceleration features
python tests/test_acceleration_features.py

# 7. Retrain with enriched features
python src/data_processing/feature_engineering_v3.py
python src/data_processing/data_split.py
python src/models/lstm_model.py
```

### Phase 4: Architecture (Day 4)
```bash
# 8. Test Conv1D frontend
python tests/test_conv1d_model.py
```

---

## Expected Outcomes

### Realistic Expectations
**Current**: ~50% accuracy (random guessing)

**After improvements**:
- **Optimistic**: 60-65% accuracy (~10-15% improvement)
- **Realistic**: 55-60% accuracy (~5-10% improvement)
- **Pessimistic**: Still ~50% (fundamental biomechanical similarity)

### Why Improvement May Be Limited
Even with all improvements, **Clear vs Smash may be inherently difficult**:
1. Both are overhead shots at similar heights
2. Main difference is racket angle (not visible in pose)
3. Shuttlecock trajectory differs (not in pose data)
4. Body mechanics are nearly identical

### Alternative Recommendation
If improvements don't help, consider:
1. **Different stroke pairs**: Clear vs Drive, Smash vs Drop (more distinct)
2. **Multi-class**: Classify all stroke types (easier than binary)
3. **Additional modalities**: Add racket detection, ball tracking
4. **Document findings**: Report that pose-only classification has fundamental limits

---

## Implementation Priority

**Start Here** (highest ROI for 19 clips):
1. Temporal augmentation
2. Padding masking
3. Training stabilization
4. Threshold tuning

**Then** (if still time):
5. Acceleration features
6. Conv1D frontend

**For Full Dataset**:
- Re-extract poses for more clips from multiple matches
- Implement group-stratified split
- Re-run all improvements on larger dataset
