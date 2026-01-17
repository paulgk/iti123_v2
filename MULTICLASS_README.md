# Multi-Class Game Analysis Branch

**Branch**: `multiclass-game-analysis`
**Date**: January 17, 2026
**Purpose**: Automatic stroke detection for rally analysis

---

## Overview

This branch contains a **4-class multi-class classifier** for automatic game analysis:
- **Clear** (23.0%)
- **Smash** (22.8%)
- **Drop** (29.9%)
- **Lift** (24.3%)

**Total**: 11,307 forehand-only strokes
**Class Balance**: 1.31x (excellent)

---

## Branch Strategy

### Main Branch (Binary Pairs for Coaching)
- Drop vs Smash (70-85% accuracy)
- Lift vs Smash (75-90% accuracy)
- **Use**: Real-time coaching feedback

### This Branch (Multi-Class for Game Analysis)
- 4-class classification (50-70% accuracy)
- **Use**: Automatic rally analysis, statistics

---

## Dataset Statistics

| Stroke | Total | Train (70%) | Val (15%) | Test (15%) |
|--------|-------|-------------|-----------|------------|
| Clear  | 2,601 | 1,820       | 390       | 390        |
| Smash  | 2,575 | 1,802       | 386       | 386        |
| Drop   | 3,378 | 2,364       | 506       | 506        |
| Lift   | 2,753 | 1,927       | 412       | 412        |
| **Total** | **11,307** | **7,914** | **1,696** | **1,696** |

**File**: `data/processed/clips/multiclass_4stroke_metadata.csv`

---

## Expected Performance

### Accuracy: 50-70%

**Why lower than binary?**
1. Four decision boundaries vs one
2. Confusion between similar overhead strokes (Clear, Drop, Smash)
3. Multi-class is inherently harder

**Is this good enough?**
✅ YES for game analysis:
- Random guessing = 25%
- 60% = **2.4x better than random**
- Sufficient for rally statistics

---

## Processing Pipeline

### Quick Start
```bash
# Switch to this branch
git checkout multiclass-game-analysis

# Extract dataset (DONE ✓)
python extract_multiclass_dataset.py

# Run pipeline
python process_multiclass_pipeline.py
```

### Detailed Steps

**1. Pose Extraction** (30-60 min)
```bash
python src/data_processing/extract_poses.py \
  --metadata data/processed/clips/multiclass_4stroke_metadata.csv \
  --videos data/videos/ \
  --output data/processed/poses_multiclass/
```

**2. Feature Engineering** (15-20 min)
```bash
python src/data_processing/feature_engineering_v2.py \
  --input data/processed/poses_multiclass/ \
  --output data/processed/features_multiclass/
```

**3. Data Splitting** (1 min)
```bash
python src/data_processing/data_split.py \
  --input data/processed/features_multiclass/ \
  --metadata data/processed/clips/multiclass_4stroke_metadata.csv \
  --output data/processed/splits_multiclass/
```

**4. Baseline Training** (10 min)
```bash
python src/models/baseline_model_multiclass.py \
  --splits data/processed/splits_multiclass/ \
  --output models/multiclass/baseline/ \
  --classes 4 \
  --class-weight balanced
```

**5. Deep Learning Training** (30-40 min with GPU)
```bash
python src/models/lstm_model_multiclass.py \
  --splits data/processed/splits_multiclass/ \
  --output models/multiclass/deep_learning/ \
  --classes 4 \
  --loss focal
```

**6. Evaluation**
```bash
python src/evaluation/evaluate_multiclass.py \
  --models models/multiclass/ \
  --output outputs/reports_multiclass/
```

---

## Training Strategies

### Class Weighting
Handle slight imbalance (1.31x):

```python
# Scikit-learn
RandomForestClassifier(class_weight='balanced')

# Keras
class_weights = {
    0: 1.15,  # Clear (underweighted)
    1: 1.16,  # Smash (underweighted)
    2: 1.00,  # Drop (baseline)
    3: 1.09,  # Lift (slightly underweighted)
}
model.fit(..., class_weight=class_weights)
```

### Focal Loss
Focus on hard examples:

```python
def focal_loss(gamma=2.0, alpha=0.25):
    def loss(y_true, y_pred):
        # Reduce loss for well-classified examples
        # Increase loss for misclassified examples
        pt = tf.where(tf.equal(y_true, 1), y_pred, 1 - y_pred)
        return -alpha * tf.pow(1 - pt, gamma) * tf.math.log(pt)
    return loss
```

---

## Expected Confusion Matrix

```
Predicted →
Actual ↓      Clear  Smash   Drop   Lift
Clear          180     40     110    60   (46% accuracy)
Smash           35    260      80    11   (67% accuracy)
Drop            90     80     280    56   (55% accuracy)
Lift            25     15      70   302   (73% accuracy)
```

**Insights**:
- **Lift**: Easiest to classify (underhand vs overhead)
- **Smash**: Good accuracy (explosive motion distinct)
- **Clear vs Drop**: Most confusion (both overhead, similar)

---

## Use Cases

### ✅ RECOMMENDED Use Cases

1. **Automatic Rally Analysis**
   ```
   Rally 1: Drop → Lift → Smash → Drop → Clear
   Pattern: Offensive start, defensive response
   ```

2. **Match Statistics**
   ```
   Player A: 45% Drop, 30% Clear, 20% Smash, 5% Lift
   Player B: 40% Clear, 35% Lift, 20% Smash, 5% Drop
   Style: A is aggressive, B is defensive
   ```

3. **Game Pattern Recognition**
   ```
   Winning rallies: Start with Drop (65%)
   Losing rallies: Start with Clear (60%)
   Recommendation: Use Drop shot more
   ```

4. **Tournament Data Collection**
   ```
   Match 1: 145 strokes analyzed
   Match 2: 167 strokes analyzed
   Aggregate statistics across tournament
   ```

### ❌ AVOID for

1. **Real-time Coaching** → Use binary pairs (70-85% accuracy)
2. **Technique Feedback** → Ambiguous labels, unclear benchmarks
3. **High-stakes Decisions** → 50-70% not reliable enough

---

## Integration with Binary Pairs

**Hybrid Approach** (Best of both worlds):

### Workflow
```
User uploads video
    ↓
Multi-class model: "70% Drop, 25% Smash, 5% Clear"
    ↓
If confidence > 80%:
    → Automatic stroke detection: DROP
    → Use Drop vs Smash binary pair
    → 85% confidence: DROP (validated)
    → Show technique feedback vs Drop benchmarks

If confidence < 80%:
    → Ask user: "Is this Drop or Smash?"
    → User selects: Drop
    → Use Drop vs Smash binary pair
    → Show technique feedback
```

### Benefits
- **Automatic** when confident (>80%)
- **User validation** when uncertain (<80%)
- **High accuracy feedback** using binary pairs

---

## File Structure

```
multiclass-game-analysis branch:

data/processed/
├── clips/
│   └── multiclass_4stroke_metadata.csv  ✓ (11,307 strokes)
├── poses_multiclass/                     (to be created)
├── features_multiclass/                  (to be created)
└── splits_multiclass/                    (to be created)

models/multiclass/
├── baseline/                             (RF, SVM)
└── deep_learning/                        (LSTM, BiLSTM, GRU)

outputs/reports_multiclass/               (confusion matrix, metrics)

Scripts:
├── extract_multiclass_dataset.py         ✓ (dataset ready)
├── process_multiclass_pipeline.py        ✓ (pipeline ready)
└── MULTICLASS_README.md                  ✓ (this file)
```

---

## Comparison: Binary vs Multi-Class

| Aspect | Binary Pairs | Multi-Class |
|--------|-------------|-------------|
| **Accuracy** | 70-85% | 50-70% |
| **Use Case** | Coaching feedback | Game analysis |
| **User Input** | Must specify stroke | Automatic detection |
| **Feedback** | Clear benchmarks | Ambiguous (which benchmark?) |
| **Training Time** | ~1hr per pair | ~2hrs for 4-class |
| **Storage** | 3 models | 1 model |
| **Deployment** | Separate endpoints | Single endpoint |

---

## Success Criteria

### Minimum Viable
- ✅ Accuracy > 60% (2.4x better than random)
- ✅ F1 (macro) > 58%
- ✅ Each class recall > 45%

### Good Performance
- ✅ Accuracy > 65%
- ✅ F1 (macro) > 62%
- ✅ Lift recall > 70% (easiest class)

### Excellent Performance
- ✅ Accuracy > 70%
- ✅ F1 (macro) > 68%
- ✅ All classes recall > 60%

---

## Next Steps

### Tomorrow
1. ✅ Dataset extracted (11,307 strokes)
2. ⏳ Extract poses from videos
3. ⏳ Train multi-class models
4. ⏳ Analyze confusion matrix
5. ⏳ If >60% accuracy, integrate with app

### Integration Plan
If multi-class succeeds (>60%):
1. Add automatic stroke detection endpoint
2. Use for rally analysis feature
3. Fallback to binary pairs for coaching

If multi-class fails (<55%):
1. Keep binary pairs only
2. User must specify stroke type
3. Focus on coaching, skip rally analysis

---

## Troubleshooting

### Q: Accuracy < 50%?
**A**: Check confusion matrix. If one class dominates predictions, adjust class weights.

### Q: Clear vs Drop confused?
**A**: Expected. Both overhead. Consider adding racket tracking features.

### Q: Out of memory?
**A**: Reduce batch size or use lighter model (GRU instead of LSTM).

### Q: Should I merge to main?
**A**: Only after achieving >60% accuracy and user testing.

---

## Merging Strategy

**When to merge**:
- ✅ Multi-class accuracy > 60%
- ✅ Binary pairs still work (70-85%)
- ✅ Integration tested

**How to merge**:
```bash
# On main branch
git checkout main

# Merge multi-class branch
git merge multiclass-game-analysis

# Resolve conflicts (keep both approaches)
# Test both binary and multi-class work

# Commit merge
git commit -m "Add multi-class for game analysis (60-70% acc)"
```

**Result**: App supports BOTH:
1. Binary pairs for coaching (high accuracy)
2. Multi-class for rally analysis (moderate accuracy)

---

## Contact

Questions about this branch:
- See `process_multiclass_pipeline.py` for processing
- See `FINAL_PROJECT_REPORT.md` for original analysis
- See main branch's `NEW_DATASETS_README.md` for binary pairs

---

**Status**: Dataset ready, awaiting processing 🚀
**Branch**: `multiclass-game-analysis`
**Keep separate until tested and validated!**
