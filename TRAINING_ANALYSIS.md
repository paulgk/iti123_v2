# Training Results Analysis - AI Badminton Coach

**Date**: January 16, 2026
**Status**: ✅ Data is clean, but model performance is poor

---

## Training Results (From Colab)

| Model | Accuracy | F1 Score | ROC AUC | Assessment |
|-------|----------|----------|---------|------------|
| **LSTM** | 45.19% | 55.31% | 48.02% | Poor |
| **BiLSTM** | 50.07% | 40.35% | 49.34% | Random |
| **GRU** | 50.67% | 33.27% | 49.09% | Random |

**Best Model**: LSTM with F1 score of 0.5531

---

## Data Verification ✅

### Splits Analysis

**Current splits (generated Jan 16, 21:45):**
- **Total samples**: 4,655 (from 4,682 clips, 27 failed to process)
- **Train**: 3,554 samples (76.3%)
- **Val**: 426 samples (9.2%)
- **Test**: 675 samples (14.5%)

**Source**: Filtered forehand-only metadata (0 backhand shots)

### Class Distribution ✅

**Training set:**
- Clear: 1,793 (50.5%)
- Smash: 1,761 (49.5%)
- **Status**: ✅ Perfectly balanced

**Test set:**
- Clear: 366 (54.2%)
- Smash: 309 (45.8%)
- **Status**: ✅ Well balanced

### Data Quality ✅

**Shape**: (samples, 50 timesteps, 60 features)
- NaN values: 0 ✅
- Inf values: 0 ✅
- Normalized: Mean=0.0, Std=1.0 ✅

---

## Why Performance is Poor (45-50% accuracy)

Despite having:
- ✅ Clean forehand-only data
- ✅ Balanced classes
- ✅ Good normalization
- ✅ No missing values

The models are performing at **random chance** (50% for binary classification).

### Possible Root Causes:

1. **🔴 Sequence Length Mismatch**
   - Training data: 50 timesteps
   - Real videos: Variable length (often < 50 frames)
   - Issue: Models may have learned padding artifacts

2. **🔴 Feature Selection Issues**
   - Using 60 features per timestep
   - Some features may not be discriminative
   - May need feature importance analysis

3. **🔴 Temporal Patterns Not Distinctive**
   - Clear vs Smash may be too similar in motion patterns
   - Key differences may be in single frames (contact point) not sequences
   - LSTM/GRU may be overcomplicating

4. **🔴 Model Architecture**
   - Current: 2-layer LSTM with 64/32 units
   - May need: More layers, attention mechanism, or different approach

5. **🔴 Training Strategy**
   - May need: Different learning rate, longer training, data augmentation

---

## Recommendations

### Immediate Actions:

#### 1. **Check Statistical Features Instead**

The statistical summary features (427 features) might work better than sequences:

```python
# Load statistical features
with open('data/processed/features/statistical_features.pkl', 'rb') as f:
    data = pickle.load(f)

# These are 427 aggregated features (mean, std, min, max of sequences)
# Try traditional ML: Random Forest, XGBoost, or simple neural network
```

**Why**: Statistical features capture the overall stroke characteristics without temporal complexity.

#### 2. **Verify Model Training Process**

Check if models are actually learning:
- Review training curves (loss/accuracy over epochs)
- Check for overfitting (train accuracy >> test accuracy)
- Verify early stopping isn't triggering too early

#### 3. **Try Simpler Approach First**

Before deep learning, try:
```python
# Use statistical features with Random Forest
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report

# This should get 70-85% accuracy if data is good
```

#### 4. **Feature Importance Analysis**

Identify which features actually discriminate Clear vs Smash:
```python
# Use the derive_benchmarks.py approach
# Cohen's d analysis to find most discriminative features
```

---

## Expected Performance (If everything works)

With good data and model:
- **Accuracy**: 70-85%
- **F1 Score**: 72-87%
- **ROC AUC**: 75-90%

Current performance suggests fundamental issue with either:
1. Sequence-based approach not suitable for this task
2. Model architecture needs redesign
3. Features not capturing discriminative information

---

## Next Steps

### Option A: Use Statistical Features (Recommended)

```bash
# Train with statistical features instead of sequences
python src/models/baseline_model.py  # Random Forest/SVM approach
```

Expected: 65-80% accuracy

### Option B: Debug Deep Learning

1. Visualize training curves
2. Check feature importance in sequences
3. Try attention mechanism
4. Reduce sequence length to 20-30 frames
5. Add dropout regularization

### Option C: Hybrid Approach

Use LSTM to learn which frames are important, then classify based on those frames.

---

## Data Status Summary

| Component | Status | Details |
|-----------|--------|---------|
| Metadata | ✅ Clean | 4,682 forehand-only strokes |
| Backhand filtering | ✅ Complete | 301 backhand shots removed |
| Data splits | ✅ Ready | 76/9/15% split, generated today |
| Class balance | ✅ Good | 50/50 Clear/Smash |
| Normalization | ✅ Good | Mean=0, Std=1 |
| Data quality | ✅ Good | No NaN/Inf |
| Benchmarks | ✅ Updated | Forehand-only ranges |

**Conclusion**: Data is excellent. The 45-50% accuracy is likely due to model architecture or feature representation issues, not data quality.
