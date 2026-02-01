# Roadmap: AI Badminton Coaching App v1.1

## Overview

The v1.1 milestone transforms the current benchmark-based coaching system into an ML-powered platform with improved classification accuracy. Starting with reliable infrastructure (Git LFS, GCS, Colab Enterprise), we expand features based on biomechanics research, retrain models with improved accuracy, and integrate ML classification alongside the existing benchmark system as a dual-mode enhancement.

**Scope for Phases 1-4:** Clear and Smash overhead shots only. Drop and Lift shots will be addressed in Phase 5 after initial ML system is validated.

## Phases

**Phase Numbering:**
- Integer phases (1, 2, 3, 4): Planned milestone work
- Decimal phases (e.g., 2.1): Urgent insertions if needed

- [x] **Phase 1: Infrastructure Foundation** - Establish reliable Colab workflow with GCS and Git LFS (Clear + Smash only)
- [x] **Phase 2: Feature Engineering Enhancement** - Add coach-informed biomechanical features (Clear + Smash only)
- [ ] **Phase 3: Model Training & Evaluation** - Train and validate improved ML models (Clear + Smash only)
- [ ] **Phase 4: Production Integration** - Integrate ML classification into production system (Clear + Smash only)
- [ ] **Phase 5: Multi-Shot Type Expansion** - Extract and integrate Drop and Lift shots

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

**Scope**: Clear and Smash overhead shots only (7,303 videos total: 2,662 Clear + 4,641 Smash)

**Success Criteria** (what must be TRUE):
1. GCS bucket is accessible from Colab Enterprise with Clear and Smash video files uploaded
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

**Scope**: Clear and Smash overhead shots only (7,303 videos total: 2,662 Clear + 4,641 Smash)

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
- [x] 02-01-PLAN.md — Phase segmentation algorithm with velocity-based detection (Wave 1)
- [x] 02-02-PLAN.md — P0 features: kinetic chain timing + contact frame analysis (Wave 2)
- [x] 02-03-PLAN.md — P1 features: angular velocity + phase-specific extraction (Wave 2)
- [x] 02-04-PLAN.md — Feature selection pipeline: filter (Cohen's d, VIF) + wrapper (RFECV) (Wave 3)
- [x] 02-05-PLAN.md — Feature engineering v3 integration with version compatibility (Wave 4)

---

### Phase 3: Model Training & Evaluation

**Goal**: Train Random Forest, SVM, and LSTM models on enhanced features achieving 70%+ accuracy with proper cross-validation and regularization.

**Scope**: Clear and Smash binary classification only (7,303 videos)

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

**Scope**: Clear and Smash binary classification only

**Depends on**: Phase 3

**Requirements**: Production Integration category
- Feature version compatibility (v2 vs v3)
- ML classification as primary method
- Model versioning and loading system
- Streamlit integration
- Clear and Smash binary classification

**Success Criteria** (what must be TRUE):
1. Streamlit interface offers dual-mode analysis (benchmark default, ML optional) with single config toggle
2. ML classification triggers only when confidence exceeds 0.85 threshold, otherwise falls back to benchmark
3. Feature version compatibility maintains v2 (427 features) for benchmarks and v3 (expanded) for ML
4. Model loading system selects appropriate model version based on feature set detected
5. Clear and Smash binary classification integrated and validated in production

**Plans**: TBD

Plans:
- [ ] TBD during phase planning

---

### Phase 5: Multi-Shot Type Expansion

**Goal**: Extract and integrate Drop and Lift shot data to expand from binary (Clear vs Smash) to multi-class classification.

**Scope**: Add Drop (3,179 videos) and Lift (573 videos) to existing Clear + Smash dataset

**Depends on**: Phase 4 (Clear + Smash system validated in production)

**Requirements**: Data Extraction & Multi-Class Training
- Video clip extraction for Drop and Lift shots
- Pose extraction for new shot types
- Dataset rebalancing and augmentation (Lift is only 5.2% of data)
- Multi-class model retraining (4 classes: Clear, Smash, Drop, Lift)
- Feature engineering validation across all shot types
- Production integration for 4-class classification

**Success Criteria** (what must be TRUE):
1. Drop and Lift video clips extracted with same quality standards as Clear and Smash
2. Pose extraction completed for all Drop (3,179) and Lift (573) videos
3. Multi-class models achieve >65% accuracy across all 4 shot types
4. Class imbalance addressed (Lift is only 5.2% - consider oversampling or class weights)
5. Production system seamlessly handles 4-class classification with same confidence thresholds

**Plans**: TBD

Plans:
- [ ] TBD during phase planning (user will provide video extraction instructions)

**Notes:**
- Deferred until Clear + Smash system is validated
- User will provide specific instructions for Drop/Lift video extraction
- Class imbalance (Lift 5.2% vs Smash 42%) requires careful handling
- Consider starting with 3-class (Clear, Smash, Drop) before adding Lift

---

## Progress

**Execution Order:**
Phases execute in numeric order: 1 -> 2 -> 3 -> 4 -> 5

**Dataset Scope:**
- **Phases 1-4:** Clear and Smash only (7,303 videos: 2,662 Clear + 4,641 Smash)
- **Phase 5:** Add Drop and Lift (3,179 Drop + 573 Lift = 11,055 total)

| Phase | Plans Complete | Status | Completed |
|-------|----------------|--------|-----------|
| 1. Infrastructure Foundation (Clear+Smash) | 4/4 | Complete | 2026-01-30 |
| 2. Feature Engineering Enhancement (Clear+Smash) | 5/5 | Complete | 2026-01-31 |
| 3. Model Training & Evaluation (Clear+Smash) | 0/TBD | Not started | - |
| 4. Production Integration (Clear+Smash) | 0/TBD | Not started | - |
| 5. Multi-Shot Type Expansion (Add Drop+Lift) | 0/TBD | Not started | - |

---

*Last updated: 2026-02-01 - Added Phase 5 for Drop+Lift expansion; Phases 1-4 focus on Clear+Smash binary classification*
