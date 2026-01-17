# New Stroke Classification Datasets

**Date**: January 16, 2026
**Status**: Metadata extracted, ready for processing

---

## Overview

After discovering that Clear vs Smash strokes are biomechanically indistinguishable (50% accuracy), we extracted two new stroke pairs with better discriminative characteristics from the ShuttleSet dataset.

---

## Available Datasets

### 1. Drop Shot vs Smash ⭐⭐⭐⭐⭐ (RECOMMENDED)

**File**: `data/processed/clips/drop_smash_metadata.csv`

**Statistics**:
- **Total strokes**: 5,953 (forehand-only)
- **Drop Shot**: 3,378 (56.7%)
- **Smash**: 2,575 (43.3%)
- **Backhand removed**: 2,923 (2,912 Drop, 11 Smash)

**Why this pair works**:
- **Swing speed**: Drop is slow/controlled, Smash is explosive
- **Arm deceleration**: Drop decelerates early, Smash follows through
- **Wrist action**: Drop uses soft touch, Smash uses snap
- **Contact height**: Similar overhead positioning but different execution

**Expected Performance**:
- Accuracy: 70-85%
- F1 Score: 72-87%
- ROC AUC: 75-90%
- Cohen's d: 0.6-0.8 (medium-large effect)

**Use Case**: Primary classification pair for coaching app

---

### 2. Lift/Lob vs Smash ⭐⭐⭐⭐

**File**: `data/processed/clips/lift_smash_metadata.csv`

**Statistics**:
- **Total strokes**: 5,328 (forehand-only)
- **Lift/Lob**: 2,753 (51.7%)
- **Smash**: 2,575 (48.3%)
- **Backhand removed**: 2,589 (2,578 Lift, 11 Smash)

**Why this pair works**:
- **Contact height**: Lift is low/underhand, Smash is high/overhead
- **Body position**: Lift is defensive/crouched, Smash is aggressive/extended
- **Trajectory**: Lift sends shuttlecock upward, Smash downward
- **Very different biomechanics**

**Expected Performance**:
- Accuracy: 75-90%
- F1 Score: 77-92%
- ROC AUC: 80-95%
- Cohen's d: 0.8+ (large effect)

**Use Case**: Alternative classification pair, easier to distinguish

---

## Processing Pipeline

### Quick Start

```bash
# 1. Extract metadata (DONE ✓)
python extract_drop_and_lift.py

# 2. Process Drop vs Smash (RECOMMENDED)
python process_drop_smash_pipeline.py

# 3. Process Lift vs Smash (Optional)
python process_lift_smash_pipeline.py
```

### Detailed Pipeline

Each dataset follows this processing pipeline:

#### Step 1: Pose Extraction
Extract MediaPipe poses from video clips:

```bash
# Drop vs Smash
python src/data_processing/extract_poses.py \
  --metadata data/processed/clips/drop_smash_metadata.csv \
  --videos data/videos/ \
  --output data/processed/poses_drop_smash/

# Lift vs Smash
python src/data_processing/extract_poses.py \
  --metadata data/processed/clips/lift_smash_metadata.csv \
  --videos data/videos/ \
  --output data/processed/poses_lift_smash/
```

**Note**: Requires video files in `data/videos/` directory

#### Step 2: Feature Engineering
Extract 60 spatial-temporal features + 427 statistical features:

```bash
# Drop vs Smash
python src/data_processing/feature_engineering_v2.py \
  --input data/processed/poses_drop_smash/ \
  --output data/processed/features_drop_smash/

# Lift vs Smash
python src/data_processing/feature_engineering_v2.py \
  --input data/processed/poses_lift_smash/ \
  --output data/processed/features_lift_smash/
```

#### Step 3: Data Splitting
Create train/val/test splits with group stratification (70/15/15):

```bash
# Drop vs Smash
python src/data_processing/data_split.py \
  --input data/processed/features_drop_smash/ \
  --metadata data/processed/clips/drop_smash_metadata.csv \
  --output data/processed/splits_drop_smash/

# Lift vs Smash
python src/data_processing/data_split.py \
  --input data/processed/features_lift_smash/ \
  --metadata data/processed/clips/lift_smash_metadata.csv \
  --output data/processed/splits_lift_smash/
```

#### Step 4: Model Training - Baseline
Train Random Forest and SVM:

```bash
# Drop vs Smash
python src/models/baseline_model.py \
  --splits data/processed/splits_drop_smash/ \
  --output models/drop_smash/baseline/

# Lift vs Smash
python src/models/baseline_model.py \
  --splits data/processed/splits_lift_smash/ \
  --output models/lift_smash/baseline/
```

#### Step 5: Model Training - Deep Learning
Train LSTM, BiLSTM, and GRU:

```bash
# Drop vs Smash
python src/models/lstm_model.py \
  --splits data/processed/splits_drop_smash/ \
  --output models/drop_smash/deep_learning/

# Lift vs Smash
python src/models/lstm_model.py \
  --splits data/processed/splits_lift_smash/ \
  --output models/lift_smash/deep_learning/
```

#### Step 6: Evaluation
Generate performance reports:

```bash
# Drop vs Smash
python src/evaluation/evaluate_models.py \
  --models models/drop_smash/ \
  --output outputs/reports_drop_smash/

# Lift vs Smash
python src/evaluation/evaluate_models.py \
  --models models/lift_smash/ \
  --output outputs/reports_lift_smash/
```

---

## Dataset Comparison

| Dataset | Total Strokes | Class Balance | Expected Accuracy | Cohen's d | Status |
|---------|---------------|---------------|-------------------|-----------|--------|
| Clear vs Smash | 4,682 | 50.6% / 49.4% | 45-50% ❌ | < 0.2 | Failed |
| **Drop vs Smash** | **5,953** | **56.7% / 43.3%** | **70-85% ✅** | **0.6-0.8** | **Ready** |
| Lift vs Smash | 5,328 | 51.7% / 48.3% | 75-90% ✅ | 0.8+ | Ready |

---

## Chinese to English Translations

All stroke types have been translated from Chinese to English:

| Chinese | English | Count (All) | Forehand Only |
|---------|---------|-------------|---------------|
| 放小球 | Drop | 6,290 | 3,378 |
| 挑球 | Lift | 5,331 | 2,753 |
| 殺球 | Smash | 2,586 | 2,575 |
| 長球 | Clear | 2,922 | 2,371 |
| 擋小球 | Block | 3,620 | - |
| 推球 | Push | 2,925 | - |
| 發短球 | Short_Serve | 2,051 | - |
| 點扣 | Drop_Smash | 1,648 | - |
| 勾球 | Cross_Net | 1,371 | - |
| 過度切球 | Cut | 1,356 | - |

---

## File Structure

```
iti123_v2/
├── data/
│   ├── annotations/                    # Original ShuttleSet CSVs (Chinese)
│   ├── videos/                         # Video clips (not in repo)
│   └── processed/
│       ├── clips/
│       │   ├── clips_metadata.csv      # Clear vs Smash (old)
│       │   ├── drop_smash_metadata.csv # Drop vs Smash ✓
│       │   └── lift_smash_metadata.csv # Lift vs Smash ✓
│       ├── poses_drop_smash/           # Pose files (to be created)
│       ├── poses_lift_smash/           # Pose files (to be created)
│       ├── features_drop_smash/        # Feature files (to be created)
│       ├── features_lift_smash/        # Feature files (to be created)
│       ├── splits_drop_smash/          # Train/val/test (to be created)
│       └── splits_lift_smash/          # Train/val/test (to be created)
├── models/
│   ├── drop_smash/                     # Drop vs Smash models
│   │   ├── baseline/                   # RF, SVM
│   │   └── deep_learning/              # LSTM, BiLSTM, GRU
│   └── lift_smash/                     # Lift vs Smash models
│       ├── baseline/
│       └── deep_learning/
└── outputs/
    ├── reports_drop_smash/             # Performance reports
    └── reports_lift_smash/             # Performance reports
```

---

## Next Steps

### Tomorrow (Manual Processing)

1. **Verify video files** are in `data/videos/` directory
2. **Run Drop vs Smash pipeline**:
   ```bash
   python process_drop_smash_pipeline.py
   ```
3. **Monitor progress** through each step
4. **Check results** in `outputs/reports_drop_smash/`

### Expected Timeline

- Pose extraction: ~30-45 minutes (5,953 clips)
- Feature engineering: ~10-15 minutes
- Data splitting: ~1 minute
- Baseline training: ~5 minutes
- Deep learning training: ~20-30 minutes (with GPU)
- **Total**: ~1-2 hours for complete pipeline

### If Video Files Missing

If `data/videos/` doesn't have the video files:

1. **Download from ShuttleSet dataset**:
   - Original dataset: https://github.com/wywyWang/CoachAI-Projects/tree/main/ShuttleSet
   - Or contact dataset authors

2. **Alternative**: Use pre-extracted poses if available

3. **Skip to feature engineering** if poses already exist

---

## Success Criteria

### Drop vs Smash

- ✅ Accuracy > 70%
- ✅ F1 Score > 72%
- ✅ ROC AUC > 75%
- ✅ Cohen's d > 0.6

If these criteria are met, deploy Drop vs Smash classifier in coaching app!

### Lift vs Smash

- ✅ Accuracy > 75%
- ✅ F1 Score > 77%
- ✅ ROC AUC > 80%
- ✅ Cohen's d > 0.8

Even better alternative if Drop vs Smash doesn't meet criteria.

---

## Troubleshooting

### Issue: Video files not found

**Solution**: Check if videos are in `data/videos/` with correct filenames matching `clip_name` column in metadata CSV

### Issue: Pose extraction fails

**Solution**: Verify MediaPipe installation with `python diagnose.py`

### Issue: Out of memory during training

**Solution**: Reduce batch size in model config or use smaller model architecture

### Issue: Low accuracy (< 60%)

**Solution**:
1. Check data quality with Cohen's d analysis
2. Verify features are normalized correctly
3. Ensure backhand shots are properly filtered

---

## Contact

For questions or issues, refer to:
- Original project: `FINAL_PROJECT_REPORT.md`
- Training analysis: `TRAINING_ANALYSIS.md`
- Quick start: `QUICK_START.md`

---

**Status**: Ready for processing tomorrow! 🚀
