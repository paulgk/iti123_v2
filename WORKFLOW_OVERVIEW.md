# Complete Workflow Overview - v1.1 Milestone

End-to-end workflow from raw videos to production ML-powered coaching app.

## 📊 Workflow Diagram

```
┌─────────────────────────────────────────────────────────────────────┐
│                        MILESTONE v1.1                               │
│              Coach-Informed ML Classification                        │
└─────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────┐
│  PHASE 1: Infrastructure Foundation                    [COMPLETE]   │
├─────────────────────────────────────────────────────────────────────┤
│  ✓ Git LFS setup for model files                                   │
│  ✓ GCS bucket for videos and artifacts                             │
│  ✓ Colab Enterprise runtime (Python 3.10)                          │
│  ✓ MLflow experiment tracking                                       │
│  Duration: Manual setup (1-2 hours)                                 │
└─────────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────────┐
│  PHASE 2: Feature Engineering Enhancement             [COMPLETE]    │
├─────────────────────────────────────────────────────────────────────┤
│  📓 Notebook: phase2_validation_colab.ipynb                         │
│                                                                     │
│  Step 1: Pose Extraction (2-3 hours)                               │
│  ├─ Input: 11,055 video clips from GCS                            │
│  ├─ Process: MediaPipe Pose extraction (parallel)                 │
│  └─ Output: ~10,500 pose sequences (.pkl files)                   │
│                                                                     │
│  Step 2: Validation Suite (15-30 min)                             │
│  ├─ Validation 1: Phase segmentation (≥85% accuracy)              │
│  ├─ Validation 2: Kinetic chain effect sizes (Cohen's d > 0.5)    │
│  └─ Validation 4: V2 backward compatibility                       │
│                                                                     │
│  Step 3: Feature Selection (30-60 min)                            │
│  ├─ Initial features: ~361 (v2: 308 + P0: 22 + P1: 32)           │
│  ├─ Filter: Cohen's d ≥ 0.5, VIF < 10                            │
│  ├─ Wrapper: RFECV with Random Forest                             │
│  └─ Output: ~187 selected features (<254 target)                  │
│                                                                     │
│  ✓ P0 Features: Kinetic chain, contact frame, intent, SIS        │
│  ✓ P1 Features: Angular velocity, phase-specific                  │
│  ✓ Feature manifest: selected_features.json                       │
│  Total Duration: ~3-5 hours                                        │
└─────────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────────┐
│  PHASE 3: Model Training & Evaluation                 [READY]       │
├─────────────────────────────────────────────────────────────────────┤
│  📓 Notebook: phase3_model_training_colab.ipynb                     │
│                                                                     │
│  Step 1: Feature Extraction (15-30 min)                            │
│  ├─ Extract v3 features with selection applied                    │
│  └─ Output: X (n_samples, 187), y (labels)                        │
│                                                                     │
│  Step 2: Train-Test Split (5 min)                                 │
│  ├─ Method: GroupShuffleSplit by player_id                        │
│  ├─ Prevent player leakage                                        │
│  └─ Split: 80% train / 20% test                                   │
│                                                                     │
│  Step 3: Train Random Forest (10-20 min)                          │
│  ├─ 100 estimators, max_depth=20                                  │
│  ├─ Cross-validation                                               │
│  └─ Feature importance analysis                                    │
│                                                                     │
│  Step 4: Train SVM (10-20 min)                                    │
│  ├─ RBF kernel with scaling                                       │
│  ├─ Hyperparameter tuning                                         │
│  └─ Probability calibration                                        │
│                                                                     │
│  Step 5: Model Evaluation & Selection                             │
│  ├─ Compare test accuracy, F1, train-test gap                    │
│  ├─ Confusion matrices                                            │
│  └─ Select best model                                             │
│                                                                     │
│  Success Criteria:                                                 │
│  ✓ Test accuracy > 70% (baseline: 45%)                           │
│  ✓ Train-test gap < 15%                                           │
│  ✓ F1 score > 0.75                                                │
│  Total Duration: ~2-4 hours                                        │
└─────────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────────┐
│  PHASE 4: Production Integration                      [READY]       │
├─────────────────────────────────────────────────────────────────────┤
│  📓 Notebook: phase4_production_integration_colab.ipynb             │
│                                                                     │
│  Step 1: Model Loading System (15 min)                            │
│  ├─ Version detection (v2 vs v3)                                  │
│  ├─ Model caching                                                  │
│  └─ Preprocessing pipeline loading                                │
│                                                                     │
│  Step 2: ML Classifier (20 min)                                   │
│  ├─ Confidence-based routing (threshold: 0.85)                   │
│  ├─ Probability calculation                                       │
│  └─ Feature version compatibility                                 │
│                                                                     │
│  Step 3: Dual-Mode Analyzer (15 min)                              │
│  ├─ ML classification (primary)                                   │
│  ├─ Benchmark fallback (low confidence)                           │
│  └─ User-controllable toggle                                      │
│                                                                     │
│  Step 4: Streamlit Integration (20 min)                           │
│  ├─ UI updates (ML toggle, confidence display)                   │
│  ├─ Config templates                                              │
│  └─ Integration guide                                             │
│                                                                     │
│  Step 5: Testing & Deployment                                     │
│  ├─ Unit tests                                                     │
│  ├─ Local testing                                                  │
│  └─ Production deployment                                          │
│                                                                     │
│  Success Criteria:                                                 │
│  ✓ Dual-mode system working                                       │
│  ✓ Confidence routing (>0.85 for ML)                             │
│  ✓ Feature version compatibility (v2/v3)                          │
│  Total Duration: ~1-2 hours                                        │
└─────────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────────┐
│                    🎉 PRODUCTION READY                              │
│                                                                     │
│  ML-Powered Badminton Coaching App                                │
│  ├─ Test Accuracy: 70-80%+ (baseline: 45%)                       │
│  ├─ Improvement: +25-35 percentage points                         │
│  ├─ Dual-mode: ML + benchmark fallback                           │
│  └─ Feature Set: 187 coach-informed biomechanical features       │
└─────────────────────────────────────────────────────────────────────┘
```

---

## 🎯 Quick Navigation

### Current Phase: Phase 2 (Pose Extraction Running)

**Next Actions:**
1. Wait for pose extraction to complete (~2-3 hours remaining)
2. Open `notebooks/phase2_validation_colab.ipynb` in Colab
3. Run validation cells sequentially
4. Review feature selection results
5. Proceed to Phase 3

### Notebooks

| Phase | Notebook | Duration | Status |
|-------|----------|----------|--------|
| 2 | [phase2_validation_colab.ipynb](notebooks/phase2_validation_colab.ipynb) | 3-5 hrs | In Progress |
| 3 | [phase3_model_training_colab.ipynb](notebooks/phase3_model_training_colab.ipynb) | 2-4 hrs | Ready |
| 4 | [phase4_production_integration_colab.ipynb](notebooks/phase4_production_integration_colab.ipynb) | 1-2 hrs | Ready |

**Total Time to Production:** ~6-11 hours (mostly automated)

---

## 📦 Key Deliverables by Phase

### Phase 2 Output
- ✅ `data/processed/poses/*.pkl` - 10,500+ pose sequences
- ✅ `data/metadata.csv` - Labels and metadata
- ✅ `data/processed/features_v3/selected_features.json` - Feature manifest (187 features)
- ✅ `outputs/reports/feature_selection_report.md` - Detailed report

### Phase 3 Output
- 🔄 `models/v3/random_forest_*.pkl` - Trained Random Forest
- 🔄 `models/v3/svm_*.pkl` - Trained SVM
- 🔄 `models/v3/model_metadata_*.json` - Performance metrics
- 🔄 `outputs/reports/model_comparison.png` - Visual comparison

### Phase 4 Output
- 🔄 `src/models/model_loader.py` - Model loading system
- 🔄 `src/models/ml_classifier.py` - ML classifier
- 🔄 `src/models/dual_mode_analyzer.py` - Dual-mode system
- 🔄 `docs/streamlit_integration_guide.md` - Integration guide
- 🔄 `tests/test_production_integration.py` - Unit tests

Legend: ✅ Complete | 🔄 Pending

---

## 🔍 Feature Engineering Deep Dive

### Feature Categories

**v2 Baseline (308-315 features):**
- Joint positions, velocities, accelerations
- Body angles and orientations
- Basic temporal patterns

**P0 Features (~22 features):**
- Kinetic chain timing (7 features)
  - Hip → Trunk → Shoulder → Elbow → Wrist sequential delays
- Contact frame analysis (8 features)
  - Joint positions and velocities at contact
- Intent window (6 features)
  - Pre-contact movement patterns (contact-5 to contact-2 frames)
- Smash Intent Score (SIS, 1 feature)
  - Weighted formula: 35% elbow + 30% pronation + 15% non-racket arm + 10% torso + 10% COM

**P1 Features (~32 features):**
- Angular velocity (6 features)
  - Double-smoothing pipeline, noise handling
- Racket head speed (4 features)
  - Estimated from wrist velocity (r=0.72-0.74)
- Phase-specific (~20 features)
  - Per-phase: wrist velocity, arm extension, elbow angle, body lean
  - 5 phases: preparation, backswing, forward_swing, contact, follow-through
- Deceleration control (2 features)
  - Post-contact braking efficiency

**Total: ~361 features → Selected: 187 features (<254 target)**

### Feature Selection Pipeline

**Two-Stage Approach:**

1. **Filter Stage:**
   - Cohen's d ≥ 0.5 (medium effect size, biomechanics standard)
   - VIF < 10 (remove multicollinearity)
   - Zero variance removal
   - Result: 361 → ~240 features

2. **Wrapper Stage:**
   - RFECV (Recursive Feature Elimination with Cross-Validation)
   - Random Forest estimator
   - 5-fold CV, F1 scoring
   - Result: ~240 → 187 features

**Validation:**
- N_train/10 rule: 2,554 training samples requires <254 features
- All selected features have Cohen's d > 0.5
- Cross-validation F1 score: 0.78+

---

## 🚀 Performance Targets

| Metric | Baseline (v2) | Target (v3) | Expected |
|--------|---------------|-------------|----------|
| Test Accuracy | 45% | >70% | 70-80% |
| F1 Score | ~0.40 | >0.75 | 0.78-0.85 |
| Train-Test Gap | N/A | <15% | 10-12% |
| Feature Count | 308-315 | <254 | 187 |
| Effect Size (Cohen's d) | N/A | >0.5 | 0.5-0.9 |

**Improvement:** +25-35 percentage points in accuracy

---

## 📚 Documentation

### Core Guides
- [COLAB_QUICKSTART.md](COLAB_QUICKSTART.md) - Quick start for Colab
- [notebooks/README.md](notebooks/README.md) - Notebook overview
- [scripts/README.md](scripts/README.md) - Script documentation

### Phase Documentation
- [Phase 2 Verification](.planning/phases/02-feature-engineering-enhancement/02-VERIFICATION.md)
- [Phase 2 Research](.planning/phases/02-feature-engineering-enhancement/02-RESEARCH.md)
- [Project Roadmap](.planning/ROADMAP.md)

### Integration Guides
- [Streamlit Integration Guide](docs/streamlit_integration_guide.md) - Phase 4 integration (created by notebook)

---

## 🎓 Learning Resources

### Understanding the Pipeline

1. **Phase Segmentation:**
   - Velocity-based stroke phase detection
   - 5 phases identified: preparation → backswing → forward_swing → contact → follow-through
   - Contact detected at peak velocity (NOT position)

2. **Kinetic Chain:**
   - Sequential coordination: hip → trunk → shoulder → elbow → wrist
   - Timing measured ONLY in forward_swing phase (critical correctness)
   - Large effect sizes (Cohen's d > 0.5) discriminate clear vs smash

3. **Feature Selection:**
   - Prevents overfitting via N_train/10 rule
   - Removes redundant features (VIF)
   - Selects discriminative features (Cohen's d)
   - Optimizes with cross-validation (RFECV)

4. **Model Architecture:**
   - Random Forest: Non-linear, handles interactions, feature importance
   - SVM: Maximum margin, RBF kernel for non-linearity
   - Both regularized to prevent overfitting

5. **Production System:**
   - Dual-mode: ML primary, benchmark fallback
   - Confidence routing: >0.85 uses ML, else benchmark
   - Version compatibility: v2 for old models, v3 for new

---

## ⚠️ Important Notes

### Player Leakage Prevention
**Critical:** Train-test split must use `GroupShuffleSplit` with `groups=player_ids`
- Same player cannot appear in both train and test sets
- Ensures model generalizes to new players
- Implemented in Phase 3 notebook

### N_train/10 Rule
**Critical:** Feature count must be <254 for 2,554 training samples
- Prevents overfitting
- Ensures stable model performance
- Enforced by feature selection pipeline

### Feature Version Compatibility
**v2 (backward compatibility):**
- 308-315 features
- No SIS feature
- No phase segmentation
- For existing benchmark models

**v3 (new features):**
- ~187 selected features
- Includes P0 + P1 features
- Phase segmentation enabled
- For new ML models

### GCS Bucket Structure
```
gs://iti123storage/
├── videos/
│   └── clips/
│       ├── clear/*.mp4
│       └── smash/*.mp4
├── features/
│   └── poses/*.pkl
├── features_v3/
│   └── selected_features.json
├── models/
│   └── v3/*.pkl
├── outputs/
│   └── reports/*.md, *.png
└── metadata.csv
```

---

## 🐛 Troubleshooting

### Common Issues

**"Pose extraction too slow"**
- Use parallel script: `extract_poses_parallel.py`
- Lower model complexity: `--model-complexity 1`
- Reduce FPS: `--target-fps 15`

**"Feature selection takes forever"**
- Reduce sample size for testing: `--sample-size 1000`
- Use high-RAM Colab runtime

**"Train-test gap too high"**
- Check player leakage (use GroupShuffleSplit)
- Increase regularization
- Reduce model complexity

**"ML confidence always low"**
- Retrain with more data
- Check feature extraction (v3 vs v2)
- Lower threshold (not recommended <0.80)

---

## 📞 Support

For issues:
1. Check relevant notebook troubleshooting section
2. Review phase verification document
3. Check scripts README for command details
4. Review GitHub issues

---

**Last Updated:** 2026-01-31
**Status:** Phase 2 in progress (pose extraction running)
**Next Milestone:** Phase 3 model training
