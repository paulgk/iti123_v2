# Notebooks

Jupyter notebooks for complete project workflow - from pose extraction to production deployment.

## Overview

This directory contains interactive Jupyter notebooks for the complete project workflow:

- **Complete Workflow**: All phases in one notebook (recommended for first-time users)
- **Phase 2**: Feature engineering validation
- **Phase 3**: Model training and evaluation
- **Phase 4**: Production integration

Each notebook provides step-by-step guidance with checkpoints, expected outputs, and troubleshooting.

---

## 🌟 Complete Workflow (Recommended)

### [complete_workflow_colab.ipynb](complete_workflow_colab.ipynb)

**Purpose:** End-to-end workflow from Colab setup to production in one notebook

**Duration:** ~8-12 hours (mostly automated)

**What it covers:**
- ✅ Phase 1: Colab & GCS setup (10-20 min)
- ✅ Phase 2: Pose extraction & feature validation (3-5 hours)
- ✅ Phase 3: Model training & evaluation (2-4 hours)
- ✅ Phase 4: Production integration (30-60 min)

**Perfect for:**
- First-time users wanting complete workflow
- All-in-one execution
- Minimal context switching
- Full automation from start to finish

**Prerequisites:**
- Google Cloud account with GCS bucket created
- Service account JSON key
- Videos uploaded to `gs://YOUR_BUCKET/videos/clips/`

**What you get:**
- ML models with 70-80%+ accuracy (baseline: 45%)
- 187 coach-informed biomechanical features
- Production-ready integration code
- Complete documentation and reports

---

## Phase 2: Feature Engineering Validation

### [phase2_validation_colab.ipynb](phase2_validation_colab.ipynb)

**Purpose:** Complete Phase 2 validation workflow in Colab Enterprise

**Duration:** ~1-2 hours (after pose extraction completes)

**Prerequisites:**
- Pose extraction completed (~10,000+ samples)
- `data/processed/poses/*.pkl` files exist
- `data/metadata.csv` created

**What it validates:**
1. ✅ Phase segmentation boundary accuracy (≥85%)
2. ✅ Kinetic chain effect sizes (Cohen's d > 0.5)
3. ✅ Feature selection pipeline (<254 features)
4. ✅ V2 backward compatibility

**Sections:**
- **Setup**: Verify environment and extraction results (5 min)
- **Step 1**: Verify extraction results (5 min)
- **Step 2**: Run validation suite (15-30 min)
- **Step 3**: Run feature selection pipeline (30-60 min)
- **Step 4**: Test v3 feature extraction (5 min)
- **Step 5**: Upload results to GCS (10 min)
- **Step 6**: Generate phase 2 summary (2 min)

**Usage in Colab:**

```bash
# After pose extraction completes
cd /content/iti123_v2
source colab_venv/bin/activate

# Open notebook in Colab
# Upload to Colab or open from GitHub
```

**Expected outputs:**
- `data/processed/features_v3/selected_features.json` - Feature selection manifest
- `outputs/reports/feature_selection_report.md` - Detailed selection report
- `outputs/reports/phase2_validation_summary.txt` - Phase 2 summary
- All results backed up to GCS

---

---

## Phase 3: Model Training & Evaluation

### [phase3_model_training_colab.ipynb](phase3_model_training_colab.ipynb)

**Purpose:** Train Random Forest, SVM, and LSTM models on enhanced v3 features

**Duration:** ~2-4 hours

**Prerequisites:**
- Phase 2 complete (feature selection done)
- `data/processed/features_v3/selected_features.json` populated
- Selected features <254

**What it trains:**
1. ✅ Random Forest classifier
2. ✅ SVM classifier with RBF kernel
3. ✅ Cross-validation with player stratification
4. ✅ Model comparison and selection

**Sections:**
- **Step 1**: Load data and extract features (15-30 min)
- **Step 2**: Train-test split with player stratification (5 min)
- **Step 3**: Train Random Forest (10-20 min)
- **Step 4**: Train SVM (10-20 min)
- **Step 5**: Compare models (5 min)
- **Step 6**: Save models (5 min)
- **Step 7**: Upload to GCS (10 min)
- **Step 8**: Generate summary (2 min)

**Expected outputs:**
- `models/v3/random_forest_*.pkl` - Trained Random Forest
- `models/v3/svm_*.pkl` - Trained SVM
- `models/v3/model_metadata_*.json` - Model performance metrics
- `outputs/reports/model_comparison.png` - Visual comparison
- Test accuracy >70%, F1 >0.75

---

## Phase 4: Production Integration

### [phase4_production_integration_colab.ipynb](phase4_production_integration_colab.ipynb)

**Purpose:** Integrate trained ML models into Streamlit production interface

**Duration:** ~1-2 hours

**Prerequisites:**
- Phase 3 complete (models trained)
- Models in `models/v3/`
- Test accuracy >70%

**What it integrates:**
1. ✅ Model loading system with caching
2. ✅ ML classifier with confidence routing
3. ✅ Dual-mode analyzer (ML + benchmark fallback)
4. ✅ Streamlit configuration
5. ✅ Unit tests

**Sections:**
- **Step 1**: Verify trained models (5 min)
- **Step 2**: Create model loading system (15 min)
- **Step 3**: Create ML classifier (20 min)
- **Step 4**: Create dual-mode analyzer (15 min)
- **Step 5**: Update Streamlit interface (20 min)
- **Step 6**: Create unit tests (10 min)
- **Step 7**: Generate summary (2 min)

**Expected outputs:**
- `src/models/model_loader.py` - Model loading with caching
- `src/models/ml_classifier.py` - ML classification with confidence routing
- `src/models/dual_mode_analyzer.py` - Dual-mode system
- `docs/streamlit_integration_guide.md` - Integration instructions
- `tests/test_production_integration.py` - Unit tests

---

## Other Resources

### [phase2_validation_workflow.md](phase2_validation_workflow.md)

Markdown version of Phase 2 workflow with command-line instructions.

---

## Quick Start - Complete Workflow

### Phase 2: Feature Engineering Validation

1. **Extract poses** (in Colab terminal):
   ```bash
   python scripts/extract_poses_parallel.py \
       --video-dir data/videos/ \
       --output-dir data/processed/poses/ \
       --model-complexity 1 \
       --target-fps 20 \
       --num-workers 4
   ```

2. **Run Phase 2 notebook**:
   - Open `phase2_validation_colab.ipynb`
   - Run all cells sequentially
   - Check for ✅ checkpoints after each step

3. **Review results**:
   - Feature selection report: `outputs/reports/feature_selection_report.md`
   - Phase 2 summary: `outputs/reports/phase2_validation_summary.txt`
   - Feature manifest: `data/processed/features_v3/selected_features.json`

### Phase 3: Model Training

1. **Run Phase 3 notebook**:
   - Open `phase3_model_training_colab.ipynb`
   - Run all cells sequentially
   - Wait for training to complete (~2-4 hours)

2. **Review results**:
   - Model comparison: `outputs/reports/model_comparison.png`
   - Model metadata: `models/v3/model_metadata_*.json`
   - Phase 3 summary: `outputs/reports/phase3_training_summary.txt`

### Phase 4: Production Integration

1. **Run Phase 4 notebook**:
   - Open `phase4_production_integration_colab.ipynb`
   - Run all cells sequentially
   - Create production modules

2. **Integrate into Streamlit**:
   - Follow guide: `docs/streamlit_integration_guide.md`
   - Test locally
   - Deploy to production

---

## Troubleshooting

**"No pose files found"**
- Check extraction completed: `ls data/processed/poses/*.pkl | wc -l`
- Re-run extraction if needed

**"Not enough samples"**
- Feature selection needs ≥50 samples
- For robust results, aim for 1,000+ samples

**"Validation failed"**
- Check validation output for specific failures
- Review `outputs/reports/` for detailed error messages
- Common issues:
  - Low boundary accuracy: Check pose extraction quality
  - Low effect sizes: Check stroke type labels in metadata
  - Feature count >254: Adjust target in Step 3

**"Out of memory"**
- Use High-RAM runtime in Colab
- Reduce sample size in validation steps
- Process feature selection in batches

---

## File Structure

```
notebooks/
├── README.md                                    # This file
├── phase2_validation_colab.ipynb                # Phase 2: Feature validation
├── phase3_model_training_colab.ipynb            # Phase 3: Model training
├── phase4_production_integration_colab.ipynb    # Phase 4: Production integration
└── phase2_validation_workflow.md                # Command-line workflow reference
```

## Progress Tracking

Each notebook has built-in checkpoints:

**Phase 2 Checkpoints:**
1. ✅ Extraction results verified
2. ✅ Validation suite completed
3. ✅ Feature selection completed
4. ✅ Feature extraction verified
5. ✅ Results uploaded to GCS
6. ✅ Summary generated

**Phase 3 Checkpoints:**
1. ✅ Features extracted
2. ✅ Train-test split complete
3. ✅ Random Forest trained
4. ✅ SVM trained
5. ✅ Model comparison complete
6. ✅ Models saved
7. ✅ Results uploaded to GCS
8. ✅ Summary generated

**Phase 4 Checkpoints:**
1. ✅ Models verified
2. ✅ Model loader created
3. ✅ ML classifier created
4. ✅ Dual-mode system created
5. ✅ Streamlit integration prepared
6. ✅ Unit tests created
7. ✅ Summary generated

---

## Support

For issues or questions:
- Check [COLAB_QUICKSTART.md](../COLAB_QUICKSTART.md) for common issues
- Review [scripts/README.md](../scripts/README.md) for script documentation
- See [.planning/phases/02-feature-engineering-enhancement/02-VERIFICATION.md](../.planning/phases/02-feature-engineering-enhancement/02-VERIFICATION.md) for detailed validation requirements
