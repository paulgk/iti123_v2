# Requirements: AI Badminton Coaching App v1.1

**Defined:** 2026-01-29
**Core Value:** Analyze an input video and give accurate, actionable feedback on how to improve stroke technique.

## v1.1 Requirements

Requirements for v1.1 Coach-Informed ML + Colab Infrastructure milestone.

### Infrastructure & Workflow

- Set up Git LFS for model and metrics storage
- Set up Google Cloud Storage (GCS) for video files (avoid bandwidth trap)
- Configure Colab Enterprise runtime with Python 3.10 (TensorFlow 2.15 compatibility)
- Create bidirectional git ↔ Colab sync scripts (pull data, push outputs)
- Integrate MLflow for experiment tracking
- Terminal-based script execution in Colab (no Jupyter notebooks)

### Feature Engineering

**Priority 0 (Critical - 15-20% accuracy boost expected):**
- Implement phase segmentation algorithm (5 phases: preparation, backswing, forward swing, contact, follow-through)
- Extract kinetic chain timing features (hip→trunk→shoulder→elbow→wrist sequential coordination)
- Implement contact frame-specific analysis
- Extract phase-specific features for each stroke phase

**Priority 1 (High value - 5-10% accuracy boost expected):**
- Extract angular velocity features (forearm rotation velocity, elbow extension velocity)
- Implement racket head speed estimation formula
- Extract deceleration control features (for Drop shot detection)

**Feature Selection:**
- Perform feature selection on current 427 features
- Reduce feature count to <254 features (N_train/10 threshold to prevent overfitting)
- Literature-validated feature prioritization based on coaching biomechanics research

### Model Training & Architecture

- Change model to use sparse categorical crossentropy (remove one-hot encoding)
- Implement stratified group K-fold cross-validation (prevent player leakage)
- Add regularization to prevent overfitting on small dataset
- Train Random Forest, SVM, and LSTM models with improved features
- Evaluate on external videos (non-ShuttleSet validation)
- Track experiments with MLflow

### Stroke Type Expansion

- Add Drop shot classification (deceleration-based discrimination)
- Add Lift shot classification
- Extend feature engineering to support new stroke types
- Validate on ShuttleSet annotated data for new stroke types

### Production Integration

- Implement feature version compatibility (v2 features: 427, v3 features: expanded set)
- Replace benchmark system with ML classification as primary method
- Integrate trained models into Streamlit interface
- Add model versioning and loading system

### Validation & Metrics

- Achieve test accuracy > 70% (baseline was 45%)
- Train-test accuracy gap < 15% (overfitting check)
- F1 score > 0.75 across stroke types
- External video accuracy > 65% (generalization check)
- Validate feedback quality improvements (qualitative assessment)

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
*Last updated: 2026-01-29 after initial definition*
