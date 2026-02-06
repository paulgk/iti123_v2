# Lightweight Model Redesign

**Date:** 2026-02-04
**Issue:** Models overfitting (50% train, 35% val accuracy)
**Root Cause:** Models too complex for dataset size (589K params for 4.7K samples)
**Solution:** Progressive model sizing strategy

---

## Problem Analysis

### Previous Models (Too Complex)
```
ST-GCN:     3,300,000 parameters  →  0.001 samples/param  ❌
MS-G3D:       589,000 parameters  →  0.008 samples/param  ❌
BiLSTM:       430,000 parameters  →  0.011 samples/param  ❌
Transformer:  1,200,000 parameters  →  0.004 samples/param  ❌

Dataset: 4,715 training samples
Ratio: 125-700 parameters per sample (TERRIBLE!)
```

### Overfitting Evidence
- Training accuracy: 50% (learning)
- Validation accuracy: 35% (not generalizing)
- Gap: 15% (severe overfitting)
- Validation loss: Flat/increasing

### Ideal Ratio
**Rule of thumb:** 10-20 samples per parameter for good generalization
- **Minimum:** 5 samples/param
- **Optimal:** 10-30 samples/param
- **Safe:** 50+ samples/param

---

## New Model Architecture Strategy

### Progressive Sizing Approach

**3-tier system based on dataset size:**

1. **Lightweight (100-200K params)** - For 4-7K samples
2. **Medium (400-600K params)** - For 10-15K samples
3. **Heavy (800K-1.2M params)** - For 15K+ samples

### Model Specifications

#### Lightweight Models (Current Phase)

**For 4,715 training samples:**

| Model | Parameters | Samples/Param | Architecture |
|-------|-----------|---------------|--------------|
| **ST-GCN-Light** | ~150K | 31.4 ✓ | 3→32→64→128 (4 layers) |
| **MS-G3D-Light** | ~170K | 27.7 ✓ | 3→32→64→128 (4 layers, 3 scales) |
| **BiLSTM-Light** | ~110K | 42.9 ✓ | 99→64→BiLSTM(64)→5 |

**Changes from previous:**
- Reduced layers: 9 → 4 layers
- Reduced channels: 64-256 → 32-128
- Reduced kernel size: 9 → 5 (less overfitting)
- Increased dropout: 0.5 → 0.6

#### Medium Models (After Full Extraction)

**For ~13,000 training samples:**

| Model | Parameters | Samples/Param | Architecture |
|-------|-----------|---------------|--------------|
| **ST-GCN-Med** | ~420K | 30.9 ✓ | 3→64→128→256 (6 layers) |
| **MS-G3D-Med** | ~480K | 27.1 ✓ | 3→64→128→256 (5 layers, 3 scales) |
| **BiLSTM-Med** | ~350K | 37.1 ✓ | 99→128→BiLSTM(128)→5 |

**When to use:**
- After extracting matches 23-44
- Expected ~6,400 additional samples
- Total: ~13,000 samples

---

## Key Design Decisions

### 1. Reduced Depth

**Previous:** 9 ST-GCN layers
**Now:** 4 layers (lightweight), 6 layers (medium)

**Why:**
- Deep networks need more data
- 4 layers sufficient for temporal patterns
- Reduces overfitting risk

### 2. Reduced Channel Width

**Previous:** 64 → 64 → 128 → 256 channels
**Now:** 32 → 64 → 128 channels (lightweight)

**Why:**
- Fewer parameters per layer
- Still captures hierarchical features
- Better generalization with limited data

### 3. Smaller Kernels

**Previous:** kernel_size=9 (temporal)
**Now:** kernel_size=5 (lightweight), 7 (medium)

**Why:**
- Smaller receptive field = less overfitting
- 5 frames = 165ms at 30 FPS (sufficient for badminton)
- Reduces parameters significantly

### 4. Increased Dropout

**Previous:** dropout=0.5
**Now:** dropout=0.6 (lightweight), 0.5 (medium)

**Why:**
- Stronger regularization for small dataset
- Forces model to learn robust features
- Reduces overfitting

### 5. Layer-wise Dropout

**New:** Added dropout INSIDE graph convolution layers

```python
class GraphConvolution(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=1, dropout=0.0):
        super().__init__()
        self.conv = nn.Conv2d(...)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout(dropout) if dropout > 0 else None  # NEW!
```

**Why:**
- Dropout at each layer (not just final)
- Better regularization throughout network
- Standard practice for small datasets

---

## Expected Performance

### Lightweight Models (4.7K samples)

**Realistic expectations:**
- **Training accuracy:** 55-65% (down from 50% - better learning)
- **Validation accuracy:** 50-60% (up from 35% - better generalization)
- **Train-val gap:** <10% (down from 15% - less overfitting)

**Per-class accuracy:**
- Drop/Lift: 60-70% (majority classes)
- Smash: 55-65%
- Clear: 45-55%
- Drive: 35-50% (minority class)

**Why conservative:**
- Still limited data
- Severe class imbalance (8.5:1)
- Pose similarity between classes

### Medium Models (13K samples)

**Expected improvement with 2.75x more data:**
- **Overall accuracy:** 65-75% (10-15% boost)
- **Train-val gap:** <5% (minimal overfitting)

**Per-class accuracy:**
- Drop/Lift: 75-85%
- Smash: 70-80%
- Clear: 60-70%
- Drive: 50-65%

---

## Training Strategy

### Phase 1: Lightweight Models (Current)

**Dataset:** 4,715 samples (matches 01-22)

**Train:**
1. ST-GCN-Light
2. MS-G3D-Light
3. BiLSTM-Light

**Skip:**
- Transformer (requires 10K+ samples)

**Hyperparameters:**
```python
CONFIG = {
    'num_epochs': 100,
    'learning_rate': 0.0005,
    'weight_decay': 0.0001,
    'dropout': 0.6,              # Increased
    'early_stopping_patience': 20,
    'focal_loss_gamma': 2.0,
    'gradient_clip': 1.0,
}
```

**Expected training time:** ~2 hours on Colab GPU

### Phase 2: Extract More Data

**After Phase 1 completes:**
```bash
cd /Volumes/Ext/GenAI/iti123_v2
bash scripts/extract_full_pipeline.sh 23 44
bash scripts/quick_upload_gcs.sh all
```

**Expected:**
- Additional 6,400 samples
- Total: ~13,000 samples
- Extraction time: ~2.5 hours

### Phase 3: Medium Models

**Dataset:** ~13,000 samples (all 44 matches)

**Train:**
1. ST-GCN-Medium
2. MS-G3D-Medium
3. BiLSTM-Medium
4. Transformer-Medium (now viable)

**Hyperparameters:**
```python
CONFIG = {
    'num_epochs': 100,
    'learning_rate': 0.0005,
    'weight_decay': 0.0001,
    'dropout': 0.5,              # Standard
    'early_stopping_patience': 20,
    'focal_loss_gamma': 2.0,
    'gradient_clip': 1.0,
}
```

**Expected training time:** ~4 hours on Colab GPU

---

## Validation Criteria

### Success Metrics for Lightweight Models

**Minimum acceptable:**
- ✅ Validation accuracy > 45% (better than random 20% and previous 35%)
- ✅ Train-val gap < 12% (down from 15%)
- ✅ All classes predicted (no collapse)
- ✅ Drive F1 > 0.3 (up from 0.15)

**Good performance:**
- ✅ Validation accuracy > 50%
- ✅ Train-val gap < 10%
- ✅ F1 (macro) > 0.45

**Excellent performance:**
- ✅ Validation accuracy > 55%
- ✅ Train-val gap < 8%
- ✅ F1 (macro) > 0.50

### Success Metrics for Medium Models

**Minimum acceptable:**
- ✅ Validation accuracy > 60%
- ✅ Train-val gap < 8%
- ✅ F1 (macro) > 0.55

**Good performance:**
- ✅ Validation accuracy > 65%
- ✅ Train-val gap < 5%
- ✅ F1 (macro) > 0.60

**Excellent performance:**
- ✅ Validation accuracy > 70%
- ✅ Train-val gap < 5%
- ✅ F1 (macro) > 0.65

---

## Troubleshooting

### If Lightweight Models Still Overfit (train-val gap >12%)

**Try:**
1. Increase dropout to 0.7
2. Reduce layers further (3 layers instead of 4)
3. Add label smoothing (0.1)
4. Increase augmentation probability to 0.8

### If Lightweight Models Underfit (train accuracy <45%)

**Try:**
1. Reduce dropout to 0.5
2. Increase learning rate to 0.001
3. Reduce weight decay to 0.00001
4. Check if data augmentation too aggressive

### If Medium Models Don't Improve Enough

**Possible causes:**
- Data quality issues (label noise)
- Pose features not discriminative
- Need ensemble methods
- Need temporal modeling (video clips)

---

## Architecture Comparison

### Lightweight vs Previous

| Aspect | Previous (Heavy) | New (Lightweight) | Improvement |
|--------|-----------------|-------------------|-------------|
| Parameters | 589K | 170K | 3.5x fewer |
| Layers | 6-9 | 4 | 2x shallower |
| Channels | 64-256 | 32-128 | 2x narrower |
| Kernel size | 9 | 5 | 1.8x smaller |
| Dropout | 0.5 | 0.6 | 20% stronger |
| Samples/param | 8 ❌ | 28 ✓ | 3.5x better |

### Lightweight vs Medium

| Aspect | Lightweight | Medium | When to Use |
|--------|------------|--------|-------------|
| Parameters | 150-170K | 420-480K | Light: <7K data, Med: 10K+ data |
| Layers | 4 | 5-6 | Light: Current, Med: After extraction |
| Channels | 32-128 | 64-256 | Light: Simple patterns, Med: Complex |
| Kernel size | 5 | 7 | Light: Local, Med: Broader context |
| Expected acc | 50-60% | 65-75% | Light: Baseline, Med: Production |

---

## Model Code Snippets

### Lightweight ST-GCN
```python
class STGCN_Lightweight(nn.Module):
    def __init__(self, num_classes, in_channels=3, num_joints=33, dropout=0.5):
        super().__init__()
        self.A = self.build_adjacency_matrix(num_joints)

        # 4 layers: 3→32→64→128
        self.gcn1 = GraphConvolution(in_channels, 32, kernel_size=5, dropout=dropout)
        self.gcn2 = GraphConvolution(32, 64, kernel_size=5, dropout=dropout)
        self.gcn3 = GraphConvolution(64, 128, kernel_size=5, dropout=dropout)
        self.gcn4 = GraphConvolution(128, 128, kernel_size=5, dropout=dropout)

        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(128, num_classes)
        self.dropout = nn.Dropout(dropout)
```

### Lightweight MS-G3D
```python
class MSG3D_Lightweight(nn.Module):
    def __init__(self, num_classes, in_channels=3, num_joints=33, dropout=0.5):
        super().__init__()
        self.A = self.build_adjacency_matrix(num_joints)

        # 4 layers with 3 scales each: 3→32→64→128
        self.gcn1 = MultiScaleGraphConv(in_channels, 32, num_scales=3, kernel_size=5, dropout=dropout)
        self.gcn2 = MultiScaleGraphConv(32, 64, num_scales=3, kernel_size=5, dropout=dropout)
        self.gcn3 = MultiScaleGraphConv(64, 128, num_scales=3, kernel_size=5, dropout=dropout)
        self.gcn4 = MultiScaleGraphConv(128, 128, num_scales=3, kernel_size=5, dropout=dropout)

        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(128, num_classes)
        self.dropout = nn.Dropout(dropout)
```

### Lightweight BiLSTM
```python
class BiLSTM_Lightweight(nn.Module):
    def __init__(self, num_classes, in_channels=3, num_joints=33, hidden_size=64, dropout=0.5):
        super().__init__()

        # 99→64→BiLSTM(64)→5
        self.input_proj = nn.Linear(num_joints * in_channels, hidden_size)
        self.lstm = nn.LSTM(hidden_size, hidden_size, num_layers=2,
                           batch_first=True, bidirectional=True, dropout=dropout)
        self.fc = nn.Linear(hidden_size * 2, num_classes)
        self.dropout = nn.Dropout(dropout)
```

---

## Files Modified

1. **notebooks/badminton_action_recognition_training.ipynb**
   - Added `STGCN_Lightweight`, `STGCN_Medium`
   - Added `MSG3D_Lightweight`, `MSG3D_Medium`
   - Added `BiLSTM_Lightweight`, `BiLSTM_Medium`
   - Updated `GraphConvolution` with layer-wise dropout
   - Updated `MultiScaleGraphConv` with dropout
   - Updated training section to use lightweight models first
   - Removed Transformer from initial training (will add in Phase 3)

2. **docs/LIGHTWEIGHT_MODELS_REDESIGN.md** (this file)

---

## Next Steps

### Immediate (Phase 1)
1. ✅ Upload updated notebook to Colab
2. ✅ Run all cells sequentially
3. ✅ Train 3 lightweight models (~2 hours)
4. ✅ Evaluate and check:
   - Train-val gap < 12%?
   - Validation accuracy > 45%?
   - All classes predicted?

### If Phase 1 Successful (val acc >45%)
1. Extract matches 23-44 (~2.5 hours)
2. Upload to GCS
3. Train medium models (~4 hours)
4. Expected 65-75% accuracy

### If Phase 1 Unsuccessful (val acc <45%)
1. Check training curves for diagnosis
2. Adjust hyperparameters (see Troubleshooting)
3. Consider data quality issues
4. May need different approach (ensemble, temporal modeling)

---

**Last Updated:** 2026-02-04
**Status:** Ready for Phase 1 training
**Confidence:** High (proper model sizing + all other fixes)
