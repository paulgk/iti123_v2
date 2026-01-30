# Roadmap: AI Badminton Coaching App v1.1

## Overview

The v1.1 milestone transforms the current benchmark-based coaching system into an ML-powered platform with improved classification accuracy. Starting with reliable infrastructure (Git LFS, GCS, Colab Enterprise), we expand features based on biomechanics research, retrain models with improved accuracy, and integrate ML classification alongside the existing benchmark system as a dual-mode enhancement.

## Phases

**Phase Numbering:**
- Integer phases (1, 2, 3, 4): Planned milestone work
- Decimal phases (e.g., 2.1): Urgent insertions if needed

- [x] **Phase 1: Infrastructure Foundation** - Establish reliable Colab workflow with GCS and Git LFS
- [ ] **Phase 2: Feature Engineering Enhancement** - Add coach-informed biomechanical features
- [ ] **Phase 3: Model Training & Evaluation** - Train and validate improved ML models
- [ ] **Phase 4: Production Integration** - Integrate ML classification into production system

## Phase Details

### Phase 1: Infrastructure Foundation

**Goal**: Establish a reliable, loss-resistant data pipeline for ML training in Colab Enterprise with proper storage and experiment tracking.

**Depends on**: Nothing (first phase)

**Requirements**: Infrastructure & Workflow category
- Git LFS setup for models and metrics
- Google Cloud Storage for video files
- Colab Enterprise runtime with Python 3.10
- Bidirectional git-Colab sync scripts
- MLflow experiment tracking
- Terminal-based script execution

**Success Criteria** (what must be TRUE):
1. GCS bucket is accessible from Colab Enterprise with video files uploaded
2. Git LFS tracks model files without exceeding bandwidth limits (under 100MB in test week)
3. Colab runtime executes Python 3.10 terminal scripts successfully (TensorFlow 2.15 compatible)
4. Bidirectional sync workflow completes end-to-end (clone -> edit -> commit -> push -> verify)
5. MLflow logs experiments to GCS backend and can be queried

**Plans:** 4 plans

Plans:
- [x] 01-01-PLAN.md — Git LFS setup and configuration files (Wave 1)
- [x] 01-02-PLAN.md — GCS setup and MLflow configuration (Wave 1)
- [x] 01-03-PLAN.md — Bidirectional sync scripts (Wave 2)
- [x] 01-04-PLAN.md — Colab runtime and infrastructure verification (Wave 3)

---

### Phase 2: Feature Engineering Enhancement

**Goal**: Add 50-150 validated biomechanical features based on coaching research to improve model discriminative power without overfitting.

**Depends on**: Phase 1 (complete)

**Requirements**: Feature Engineering category
- FEAT-01: Phase segmentation algorithm (5 phases)
- FEAT-02: Kinetic chain timing features (hip->trunk->shoulder->elbow->wrist)
- FEAT-03: Contact frame-specific analysis with intent window
- FEAT-04: Phase-specific feature extraction
- FEAT-05: Angular velocity features
- FEAT-06: Racket head speed estimation
- FEAT-07: Deceleration control features (deprioritized - dataset is Clear+Smash)
- FEAT-08: Feature selection on current features
- FEAT-09: Reduce to <254 features (N_train/10 threshold)
- FEAT-10: Literature-validated prioritization

**Success Criteria** (what must be TRUE):
1. Phase segmentation algorithm identifies 5 stroke phases (preparation, backswing, forward swing, contact, follow-through) with 85%+ boundary accuracy
2. Kinetic chain timing features capture sequential coordination with measurable time deltas between segments
3. Feature set expanded with P0 and P1 features while maintaining total count under 254 (N_train/10 threshold)
4. Feature selection analysis shows each new feature has Cohen's d > 0.5 for medium effect
5. Feature engineering v3 maintains backward compatibility with v2 (427 features) through version gating

**Plans:** 5 plans

Plans:
- [ ] 02-01-PLAN.md — Phase segmentation algorithm with velocity-based detection (Wave 1)
- [ ] 02-02-PLAN.md — P0 features: kinetic chain timing + contact frame analysis (Wave 2)
- [ ] 02-03-PLAN.md — P1 features: angular velocity + phase-specific extraction (Wave 2)
- [ ] 02-04-PLAN.md — Feature selection pipeline: filter (Cohen's d, VIF) + wrapper (RFECV) (Wave 3)
- [ ] 02-05-PLAN.md — Feature engineering v3 integration with version compatibility (Wave 4)

---

### Phase 3: Model Training & Evaluation

**Goal**: Train Random Forest, SVM, and LSTM models on enhanced features achieving 70%+ accuracy with proper cross-validation and regularization.

**Depends on**: Phase 2

**Requirements**: Model Training & Architecture + Validation & Metrics categories
- Sparse categorical crossentropy implementation
- Stratified group K-fold cross-validation
- Regularization to prevent overfitting
- Random Forest, SVM, LSTM training
- External video evaluation
- MLflow experiment tracking
- Test accuracy > 70%
- Train-test gap < 15%
- F1 score > 0.75
- External video accuracy > 65%

**Success Criteria** (what must be TRUE):
1. Models trained with sparse categorical crossentropy show test accuracy above 70% (baseline was 45%)
2. Stratified group K-fold prevents player leakage with train-test accuracy gap under 15%
3. F1 score across Clear and Smash stroke types exceeds 0.75
4. External video validation (non-ShuttleSet) achieves accuracy above 65%
5. MLflow experiment logs capture all hyperparameters, metrics, and model artifacts for reproducibility

**Plans**: TBD

Plans:
- [ ] TBD during phase planning

---

### Phase 4: Production Integration

**Goal**: Integrate trained ML models into Streamlit interface as a dual-mode system with confidence-based fallback to benchmarks.

**Depends on**: Phase 3

**Requirements**: Production Integration category
- Feature version compatibility (v2 vs v3)
- ML classification as primary method
- Model versioning and loading system
- Streamlit integration
- Drop and Lift shot support

**Success Criteria** (what must be TRUE):
1. Streamlit interface offers dual-mode analysis (benchmark default, ML optional) with single config toggle
2. ML classification triggers only when confidence exceeds 0.85 threshold, otherwise falls back to benchmark
3. Feature version compatibility maintains v2 (427 features) for benchmarks and v3 (expanded) for ML
4. Model loading system selects appropriate model version based on feature set detected
5. Drop and Lift shot classifications integrated alongside existing Clear and Smash support

**Plans**: TBD

Plans:
- [ ] TBD during phase planning

---

## Progress

**Execution Order:**
Phases execute in numeric order: 1 -> 2 -> 3 -> 4

| Phase | Plans Complete | Status | Completed |
|-------|----------------|--------|-----------|
| 1. Infrastructure Foundation | 4/4 | Complete | 2026-01-30 |
| 2. Feature Engineering Enhancement | 0/5 | Planned | - |
| 3. Model Training & Evaluation | 0/TBD | Not started | - |
| 4. Production Integration | 0/TBD | Not started | - |

---

*Last updated: 2026-01-30 - Phase 2 planning complete*
