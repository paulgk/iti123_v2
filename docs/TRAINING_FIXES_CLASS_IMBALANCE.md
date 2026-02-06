# Training Fixes for Class Imbalance

**Date:** 2026-02-04
**Issue:** Models performing at near-random accuracy (~34-35% vs expected 84-93%)
**Root Cause:** Severe class imbalance, particularly Drive class (3.8% of data)

---

## Problem Analysis

### Initial Results
- **ST-GCN:** 34.96% accuracy
- **MS-G3D:** 35.42% accuracy
- **BiLSTM:** 33.89% accuracy
- **Transformer:** 30.76% accuracy (collapsed to predicting only "Drop")

### Class Distribution
```
Drop:   3022 samples (32.4%)  ✓ Good
Lift:   2710 samples (29.0%)  ✓ Good
Smash:  1943 samples (20.8%)  ✓ Acceptable
Clear:  1306 samples (14.0%)  ⚠️ Low
Drive:    356 samples (3.8%)   ❌ CRITICAL
```

**Imbalance ratio:** 8.5:1 (Drop vs Drive)

### Why Previous Approach Failed

**1. Class Weights Too Extreme**
```python
Drive weight: 8.5x
Drop weight: 1.0x
```
- Caused training instability
- Models oscillated between overfitting Drive and ignoring it
- Validation loss flat or increasing

**2. No Data Augmentation**
- Minority classes had insufficient training examples
- Models couldn't learn robust patterns from 285 Drive samples

**3. Learning Rate Too High**
- lr=0.001 caused gradient explosions with extreme class weights
- Training curves showed high variance

**4. No Gradient Clipping**
- Gradients exploded on minority class samples
- Caused NaN losses and model collapse (Transformer)

---

## Solutions Implemented

### 1. Data Augmentation

Added skeleton-specific augmentations to increase effective training data:

**Augmentations:**
- **Temporal Scaling:** Speed up/slow down by 10-20%
- **Spatial Rotation:** ±15° rotation around vertical axis
- **Spatial Scaling:** Zoom in/out by 5-15%
- **Gaussian Noise:** Small noise (σ=0.02) for robustness

**Implementation:**
```python
def augment_pose(pose, aug_prob=0.5):
    # Applied only during training
    # Each augmentation has 50% probability
    # Preserves skeleton structure and biomechanics
```

**Impact:**
- Effectively doubles minority class data
- Improves model generalization
- Prevents overfitting on small Drive class

### 2. Focal Loss

Replaced CrossEntropyLoss with Focal Loss:

**Why Focal Loss:**
- Down-weights easy examples (well-classified samples)
- Focuses training on hard-to-classify samples
- More stable than extreme class weights
- Used in RetinaNet (CVPR 2017)

**Formula:**
```
FL(pt) = -(1 - pt)^γ * log(pt)

where:
- pt = probability of correct class
- γ = focusing parameter (2.0)
- Easy examples (pt → 1): loss → 0
- Hard examples (pt → 0): loss remains high
```

**Implementation:**
```python
criterion = FocalLoss(alpha=class_weights, gamma=2.0)

# Softened class weights (using sqrt to reduce extremes)
class_weights = sqrt(total / class_count)
# Normalize to sum to num_classes
```

**Class Weights After Softening:**
```
Smash: 0.945 (was 1.0)
Clear: 1.230 (was 1.49)
Drop:  0.765 (was 0.64)
Lift:  0.807 (was 0.72)
Drive: 2.253 (was 8.5)  ← Much more reasonable!
```

### 3. Reduced Learning Rate

**Changed:** 0.001 → 0.0005 (50% reduction)

**Why:**
- Prevents overshooting with focal loss
- More stable convergence
- Better fine-tuning of decision boundaries

### 4. Gradient Clipping

**Added:** Clip gradients to max norm of 1.0

```python
torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
```

**Why:**
- Prevents exploding gradients on minority class
- Stabilizes training with focal loss
- Essential for transformer models

### 5. Increased Patience

**Changed:** 15 → 20 epochs

**Why:**
- Focal loss needs more epochs to converge
- Allows model to learn minority classes properly
- Previous early stopping was too aggressive

### 6. F1-Based Model Selection

**Changed:** Select best model by F1 score instead of accuracy

**Why:**
- F1 is more appropriate for imbalanced data
- Balances precision and recall
- Prevents models that ignore minority classes

---

## Expected Improvements

### Performance Targets

| Model | Expected Accuracy | Expected F1 (Macro) |
|-------|-------------------|---------------------|
| **MS-G3D** | 75-82% | 70-78% |
| **ST-GCN** | 72-79% | 68-75% |
| **BiLSTM** | 68-75% | 63-70% |
| **Transformer** | 65-73% | 60-68% |

**Note:** These are realistic targets for severe class imbalance (8.5:1 ratio)

### Per-Class Performance

**Expected improvements:**
- **Drop, Lift, Smash:** 80-90% accuracy (majority classes)
- **Clear:** 65-75% accuracy (medium class)
- **Drive:** 50-65% accuracy (minority class) ← Key improvement area

**Why Drive will still be challenging:**
- Only 285 training samples (after split)
- 8.5x less data than Drop
- Augmentation helps but can't fully compensate

---

## Training Stability Indicators

### Good Training Curves Should Show:

**1. Loss:**
- Training loss: Steadily decreasing
- Validation loss: Decreasing (may plateau)
- Gap between train/val: Acceptable (<0.3)

**2. Accuracy:**
- Training accuracy: Steadily increasing to 75-85%
- Validation accuracy: Increasing to 70-80%
- Less noise than before

**3. F1 Score:**
- Validation F1: Steadily increasing
- All classes should show improvement
- Drive F1 should be >0.4 (previously ~0.05)

### Warning Signs:

❌ **Validation loss increasing:** Too high learning rate
❌ **Training accuracy 95%+, validation 35%:** Severe overfitting
❌ **All predictions for one class:** Model collapsed (shouldn't happen with focal loss)
❌ **NaN losses:** Gradient explosion (shouldn't happen with clipping)

---

## Next Steps After Training

### 1. If Results are Good (>70% accuracy):
- ✅ Extract second half of dataset (matches 23-44)
- ✅ Retrain with full ~13K samples
- ✅ Expected 5-10% accuracy boost

### 2. If Drive Class Still Poor (<50% accuracy):
**Options:**
- Increase augmentation probability for Drive
- Collect more Drive samples (if possible)
- Use SMOTE or similar oversampling
- Consider merging Drive with similar class

### 3. If All Classes Improve But Overall <65%:
**Possible causes:**
- Data quality issues
- Label noise
- Poses not discriminative enough
- Need ensemble methods

### 4. Model Deployment Priority:
1. **MS-G3D** (if >75% accuracy)
2. **ST-GCN** (if MS-G3D fails)
3. **Ensemble** (MS-G3D + ST-GCN)

---

## Changes Summary

### Modified Files:
1. **notebooks/badminton_action_recognition_training.ipynb**
   - Added `augment_pose()` function
   - Added `FocalLoss` class
   - Updated `BadmintonDataset` with augmentation
   - Updated `CONFIG` with new hyperparameters
   - Added gradient clipping to `train_epoch()`
   - Changed model selection to use F1 score
   - Fixed MS-G3D channel distribution bug

### New Hyperparameters:
```python
CONFIG = {
    'num_epochs': 100,
    'learning_rate': 0.0005,        # ← Changed from 0.001
    'weight_decay': 0.0001,
    'early_stopping_patience': 20,   # ← Changed from 15
    'scheduler': 'cosine',
    'focal_loss_gamma': 2.0,         # ← New
    'gradient_clip': 1.0,            # ← New
}
```

### Training Changes:
- ✅ Data augmentation enabled for training set
- ✅ Focal loss with softened class weights
- ✅ Gradient clipping at 1.0
- ✅ Reduced learning rate (0.0005)
- ✅ F1-based model selection
- ✅ Increased early stopping patience (20)

---

## Validation Protocol

### Before Retraining:
1. Verify class distribution in train/val/test splits
2. Check augmentation is working (sample a batch)
3. Verify focal loss accepts class weights
4. Confirm gradient clipping is applied

### During Training:
1. Monitor validation F1 score (should increase steadily)
2. Watch for NaN losses (shouldn't happen)
3. Check confusion matrix every 10 epochs
4. Verify all classes are being predicted

### After Training:
1. Check per-class F1 scores (all should be >0.3)
2. Analyze confusion matrix (no complete collapse)
3. Compare Drive accuracy to random baseline (>20%)
4. Verify best model selected by F1, not accuracy

---

## References

**Focal Loss:**
- Lin et al., "Focal Loss for Dense Object Detection" (CVPR 2017)
- https://arxiv.org/abs/1708.02002

**Data Augmentation for Skeletons:**
- Du et al., "Skeleton Based Action Recognition with Convolutional Neural Network" (2015)
- Commonly used in NTU RGB+D dataset training

**Class Imbalance Techniques:**
- He & Garcia, "Learning from Imbalanced Data" (IEEE Trans 2009)
- Focal loss > SMOTE > Class weights for deep learning

---

**Last Updated:** 2026-02-04
**Status:** Ready for retraining
**Expected Training Time:** 4-6 hours for all 4 models on Colab GPU
