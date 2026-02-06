# Training Quick Start Guide

**TL;DR:** Train small models first (2hrs), then get more data, then train bigger models (4hrs)

---

## The Problem We Fixed

**Before:**
- Models: 589K-3.3M parameters 😱
- Dataset: 4,715 samples
- Ratio: 125-700 params/sample ❌
- Result: 50% train, 35% val = Massive overfitting

**After:**
- Models: 110K-170K parameters ✓
- Dataset: 4,715 samples (same)
- Ratio: 28-43 params/sample ✓
- Expected: 55-60% train, 50-58% val = Good generalization

---

## Training Plan

### Phase 1: Lightweight Models (NOW)

**What:** Train 3 small models on current data
**Time:** ~2 hours
**Expected:** 50-60% accuracy

```
Models to train:
  ✓ ST-GCN-Light  (~150K params)
  ✓ MS-G3D-Light  (~170K params)
  ✓ BiLSTM-Light  (~110K params)
```

**Steps:**
1. Upload [badminton_action_recognition_training.ipynb](badminton_action_recognition_training.ipynb) to Colab
2. Run all cells
3. Wait ~2 hours
4. Check if val accuracy >45% ✓

### Phase 2: Get More Data

**What:** Extract second half of videos
**Time:** ~2.5 hours
**Result:** 4.7K → 13K samples (2.75x more!)

```bash
cd /Volumes/Ext/GenAI/iti123_v2
bash scripts/extract_full_pipeline.sh 23 44
bash scripts/quick_upload_gcs.sh all
```

### Phase 3: Medium Models

**What:** Train bigger models on full dataset
**Time:** ~4 hours
**Expected:** 65-75% accuracy

```
Models to train:
  ✓ ST-GCN-Medium  (~420K params)
  ✓ MS-G3D-Medium  (~480K params)
  ✓ BiLSTM-Medium  (~350K params)
  ✓ Transformer    (~600K params) - now viable!
```

---

## What Changed

### Model Sizes

| Model | Before | After (Light) | After (Med) |
|-------|--------|---------------|-------------|
| ST-GCN | 3.3M | 150K | 420K |
| MS-G3D | 589K | 170K | 480K |
| BiLSTM | 430K | 110K | 350K |

### Architecture Changes

**Lightweight models:**
- Layers: 9 → 4
- Channels: 64-256 → 32-128
- Kernel: 9 → 5
- Dropout: 0.5 → 0.6

**Medium models (after extraction):**
- Layers: 4 → 6
- Channels: 32-128 → 64-256
- Kernel: 5 → 7
- Dropout: 0.6 → 0.5

---

## Expected Results

### Phase 1 (Lightweight, 4.7K samples)

```
Best case:    55-60% accuracy
Realistic:    50-55% accuracy
Minimum OK:   >45% accuracy
Failure:      <45% accuracy
```

**Per-class (realistic):**
- Drop/Lift: 60-70%
- Smash: 55-65%
- Clear: 45-55%
- Drive: 35-50%

### Phase 3 (Medium, 13K samples)

```
Best case:    70-75% accuracy
Realistic:    65-70% accuracy
Minimum OK:   >60% accuracy
```

**Per-class (realistic):**
- Drop/Lift: 75-85%
- Smash: 70-80%
- Clear: 60-70%
- Drive: 50-65%

---

## Success Criteria

### ✅ Phase 1 Success

**Training curves:**
- [ ] Training loss decreasing
- [ ] Validation loss decreasing (or plateauing)
- [ ] Train-val gap <12% (down from 15%)

**Metrics:**
- [ ] Val accuracy >45%
- [ ] All 5 classes being predicted
- [ ] Drive F1 >0.3
- [ ] No NaN losses

**If successful:** Proceed to Phase 2 (extraction)

### ❌ Phase 1 Failure

**If val accuracy <45%:**
1. Check if training accuracy also low (<45%)
   - Yes → Model not learning → Increase LR to 0.001
   - No → Still overfitting → Increase dropout to 0.7

2. Check confusion matrix
   - One class dominates → Increase focal gamma to 3.0
   - Random predictions → Check data augmentation not too strong

3. Last resort
   - Reduce to 3 layers instead of 4
   - Add label smoothing (0.1)

---

## Files Reference

### Documentation
- [LIGHTWEIGHT_MODELS_REDESIGN.md](LIGHTWEIGHT_MODELS_REDESIGN.md) - Full technical details
- [TRAINING_FIXES_CLASS_IMBALANCE.md](TRAINING_FIXES_CLASS_IMBALANCE.md) - Focal loss & augmentation
- This file - Quick start guide

### Notebook
- [badminton_action_recognition_training.ipynb](badminton_action_recognition_training.ipynb) - Main training notebook

### Scripts
- `scripts/extract_full_pipeline.sh` - Extract poses from videos
- `scripts/quick_upload_gcs.sh` - Upload to Google Cloud Storage

---

## Timeline

**Total time:** 8.5 hours (can run overnight)

```
Phase 1: Train lightweight models       2 hours
Phase 2: Extract matches 23-44         2.5 hours
Phase 3: Upload to GCS                 0.5 hours
Phase 4: Train medium models           4 hours
         (Download data: 0.5 hours included)
---------------------------------------------------
Total:                                 8.5 hours
```

**Recommended schedule:**
- Day 1 afternoon: Start Phase 1 (2hrs) → check results → start Phase 2 extraction overnight
- Day 2 morning: Upload to GCS → start Phase 3 training → complete by afternoon

---

## Monitoring Training

### Good Signs ✅
- Training loss: 0.7 → 0.5 (decreasing)
- Validation loss: 0.7 → 0.6 (decreasing or flat)
- Train accuracy: 45% → 55% (increasing)
- Val accuracy: 40% → 52% (increasing)
- Gap: 8% (acceptable)

### Bad Signs ❌
- Training loss: Decreasing, Val loss: Increasing → Overfitting
- Training loss: Flat at 0.7 → Not learning
- NaN losses → Gradient explosion (shouldn't happen with clipping)
- Val acc stuck at 32% → Class imbalance not fixed

---

## FAQ

**Q: Why train small models if we know we'll get more data?**
A: Validates approach works. Better to find issues on 2hr training than 4hr training.

**Q: What if lightweight models get 60%+ accuracy?**
A: Great! Medium models should get 70-75% then. Proceed with extraction.

**Q: What if lightweight models still overfit?**
A: See troubleshooting in [LIGHTWEIGHT_MODELS_REDESIGN.md](LIGHTWEIGHT_MODELS_REDESIGN.md)

**Q: Can I skip Phase 1 and just train medium models after extraction?**
A: Not recommended. Phase 1 validates the approach in 2hrs vs 6.5hrs.

**Q: How much will accuracy improve in Phase 3?**
A: Expect 10-15% boost from 2.75x more data. Light 52% → Med 65%.

**Q: What's the theoretical maximum accuracy?**
A: With perfect model & infinite data: 85-90% (pose similarity limits accuracy)

---

**Last Updated:** 2026-02-04
**Status:** Ready to train
**Next Action:** Upload notebook to Colab and run Phase 1
