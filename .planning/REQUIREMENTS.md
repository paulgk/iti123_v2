# Requirements: AI Badminton Coaching App v1.1

**Defined:** 2026-01-29
**Core Value:** Analyze an input video and give accurate, actionable feedback on how to improve stroke technique.

## v1.1 Requirements

Requirements for v1.1 Coach-Informed ML + Colab Infrastructure milestone.

### Infrastructure & Workflow

- INFRA-01: Set up Git LFS for model and metrics storage
- INFRA-02: Set up Google Cloud Storage (GCS) for video files (avoid bandwidth trap)
- INFRA-03: Configure Colab Enterprise runtime with Python 3.10 (TensorFlow 2.15 compatibility)
- INFRA-04: Create bidirectional git ↔ Colab sync scripts (pull data, push outputs)
- INFRA-05: Integrate MLflow for experiment tracking
- INFRA-06: Terminal-based script execution in Colab (no Jupyter notebooks)

### Feature Engineering

**Priority 0 (Critical - 15-20% accuracy boost expected):**
- FEAT-01: Implement phase segmentation algorithm (5 phases: preparation, backswing, forward swing, contact, follow-through)
- FEAT-02: Extract kinetic chain timing features (hip→trunk→shoulder→elbow→wrist sequential coordination)
- FEAT-03: Implement contact frame-specific analysis
- FEAT-04: Extract phase-specific features for each stroke phase

**Priority 1 (High value - 5-10% accuracy boost expected):**
- FEAT-05: Extract angular velocity features (forearm rotation velocity, elbow extension velocity)
- FEAT-06: Implement racket head speed estimation formula
- FEAT-07: Extract deceleration control features (for Drop shot detection)

**Feature Selection:**
- FEAT-08: Perform feature selection on current 427 features
- FEAT-09: Reduce feature count to <254 features (N_train/10 threshold to prevent overfitting)
- FEAT-10: Literature-validated feature prioritization based on coaching biomechanics research

### Model Training & Architecture

- MODEL-01: Change model to use sparse categorical crossentropy (remove one-hot encoding)
- MODEL-02: Implement stratified group K-fold cross-validation (prevent player leakage)
- MODEL-03: Add regularization to prevent overfitting on small dataset
- MODEL-04: Train Random Forest, SVM, and LSTM models with improved features
- MODEL-05: Evaluate on external videos (non-ShuttleSet validation)
- MODEL-06: Track experiments with MLflow

### Stroke Type Expansion

- STROKE-01: Add Drop shot classification (deceleration-based discrimination)
- STROKE-02: Add Lift shot classification
- STROKE-03: Extend feature engineering to support new stroke types
- STROKE-04: Validate on ShuttleSet annotated data for new stroke types

### Production Integration

- PROD-01: Implement feature version compatibility (v2 features: 427, v3 features: expanded set)
- PROD-02: Replace benchmark system with ML classification as primary method
- PROD-03: Integrate trained models into Streamlit interface
- PROD-04: Add model versioning and loading system

### Validation & Metrics

- VAL-01: Achieve test accuracy > 70% (baseline was 45%)
- VAL-02: Train-test accuracy gap < 15% (overfitting check)
- VAL-03: F1 score > 0.75 across stroke types
- VAL-04: External video accuracy > 65% (generalization check)
- VAL-05: Validate feedback quality improvements (qualitative assessment)

## Traceability

| Requirement | Phase | Status |
|-------------|-------|--------|
| INFRA-01 | Phase 1 | Pending |
| INFRA-02 | Phase 1 | Pending |
| INFRA-03 | Phase 1 | Pending |
| INFRA-04 | Phase 1 | Pending |
| INFRA-05 | Phase 1 | Pending |
| INFRA-06 | Phase 1 | Pending |
| FEAT-01 | Phase 2 | Pending |
| FEAT-02 | Phase 2 | Pending |
| FEAT-03 | Phase 2 | Pending |
| FEAT-04 | Phase 2 | Pending |
| FEAT-05 | Phase 2 | Pending |
| FEAT-06 | Phase 2 | Pending |
| FEAT-07 | Phase 2 | Pending |
| FEAT-08 | Phase 2 | Pending |
| FEAT-09 | Phase 2 | Pending |
| FEAT-10 | Phase 2 | Pending |
| MODEL-01 | Phase 3 | Pending |
| MODEL-02 | Phase 3 | Pending |
| MODEL-03 | Phase 3 | Pending |
| MODEL-04 | Phase 3 | Pending |
| MODEL-05 | Phase 3 | Pending |
| MODEL-06 | Phase 3 | Pending |
| STROKE-01 | Phase 2 | Pending |
| STROKE-02 | Phase 2 | Pending |
| STROKE-03 | Phase 2 | Pending |
| STROKE-04 | Phase 2 | Pending |
| VAL-01 | Phase 3 | Pending |
| VAL-02 | Phase 3 | Pending |
| VAL-03 | Phase 3 | Pending |
| VAL-04 | Phase 3 | Pending |
| VAL-05 | Phase 3 | Pending |
| PROD-01 | Phase 4 | Pending |
| PROD-02 | Phase 4 | Pending |
| PROD-03 | Phase 4 | Pending |
| PROD-04 | Phase 4 | Pending |

**Coverage:** 35/35 requirements mapped (100%)

## Out of Scope for v1.1

| Feature | Reason |
|---------|--------|
| Jupyter notebooks in Colab | Using terminal scripts only for reproducibility |
| Direct coach consultation | Using literature/videos instead (constraint) |
| Backhand-specific features | Cancelled for v1.1, focus on forehand strokes |
| Drive shot classification | Limited research available, deferred |
| Net shot classification | Minimal research available, deferred |
| Checkpoint-to-GCS system | Deprioritized, rely on git commits |
| Advanced movement efficiency metrics | P2 priority, deferred to future |
| Mobile app deployment | v2.0+ |
| Real-time video analysis | v2.0+ |
| Automated testing framework | Future work |

## Success Criteria

v1.1 is complete when:
- Colab Enterprise workflow is operational (Python 3.10, terminal scripts, git sync)
- Phase segmentation and kinetic chain features implemented and validated
- Model trained with sparse categorical crossentropy on expanded feature set
- Drop and Lift shot classification supported
- Test accuracy > 70% with F1 > 0.75
- ML classification integrated into Streamlit interface as primary method
- Feature version compatibility maintained between v2 and v3

---
*Requirements defined: 2026-01-29*
*Last updated: 2026-01-29 with REQ-IDs and traceability mapping*
