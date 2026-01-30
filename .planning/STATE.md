# Project State

## Project Reference

See: .planning/PROJECT.md (updated 2026-01-29)

**Core value:** Analyze an input video and give accurate, actionable feedback on how to improve stroke technique.
**Current focus:** Phase 1 - Infrastructure Foundation

## Current Position

Phase: 2 of 4 (Feature Engineering Enhancement)
Plan: 5 of 9 (Feature Engineering v3 Integration)
Status: In progress
Last activity: 2026-01-30 - Completed 02-05-PLAN.md

Progress: [█████░░░░░] 50% (9 of 18 plans complete)

## Performance Metrics

**Velocity:**
- Total plans completed: 9
- Average duration: 2.1 hours
- Total execution time: 19.1 hours

**By Phase:**

| Phase | Plans | Total | Avg/Plan |
|-------|-------|-------|----------|
| 01-infrastructure-foundation | 4/4 | 18.7hr | 4.7hr |
| 02-feature-engineering-enhancement | 5/9 | 24min | 4.8min |

**Recent Trend:**
- Last 5 plans: 02-02 (7min), 02-03 (4min), 02-04 (4min), 02-05 (4min)
- Trend: Fully autonomous plans execute quickly; feature engineering plans averaging ~5min each

*Updated after each plan completion*

## Accumulated Context

### Decisions

Decisions are logged in PROJECT.md Key Decisions table.
Recent decisions affecting current work:

- v1.1 milestone: Use Colab Enterprise with terminal scripts (not notebooks) for reproducibility
- v1.1 milestone: Use GCS for bulk video storage, Git LFS only for models (bandwidth constraint)
- v1.1 milestone: ML augments benchmark system (dual-mode), not replacement (safe rollout)
- 01-01: Git LFS tracks only production models (experiments stay in GCS/MLflow)
- 01-01: Python 3.10 runtime enforced for TensorFlow 2.15 compatibility
- 01-01: Auto-checkpoint to GCS every 30 minutes to prevent data loss
- 01-02: GCS bucket structure uses separate prefixes (videos/, features/, models/, checkpoints/, mlflow/)
- 01-02: MLflow hybrid storage - metadata local SQLite, artifacts in GCS (gs://iti123storage/mlflow/)
- 01-02: Pin TensorFlow 2.15.0 and MediaPipe 0.10.9 for Colab Enterprise Python 3.10 compatibility
- 01-03: Manual sync scripts instead of automatic (user controls when to pull/push)
- 01-03: Environment detection via /content directory check (Colab-specific path)
- 01-03: Bucket name from paths.yaml with GCS_BUCKET_NAME env var override
- 01-03: gsutil rsync for idempotent incremental sync (re-running safe)
- 01-04: Python 3.10 venv required in Colab for TensorFlow 2.15 compatibility (Colab defaults to 3.12)
- 01-04: Terminal script execution wrapper needed for non-Jupyter workflows
- 01-04: Comprehensive verification script with granular flags (--python, --lfs, --gcs, --mlflow, --scripts)
- 01-04: Deferred full Colab verification to actual training sessions (infrastructure scripts ready for use)
- 02-01: Contact frame detected at peak velocity (NOT peak position) - critical coaching insight
- 02-01: Intent window [contact-5:contact-2] is most discriminative moment per coaching research
- 02-01: Phase boundaries use scipy.signal.find_peaks with biomechanically-informed parameters
- 02-01: Validation checks enforce research-based timing constraints without raising exceptions
- 02-02: Kinetic chain timing measured ONLY in forward swing phase (prevents preparation peak contamination)
- 02-02: SIS formula weights: 0.35 elbow lead, 0.30 pronation, 0.15 non-racket arm, 0.10 torso, 0.10 COM
- 02-02: SIS thresholds: >=0.65 smash, 0.40-0.65 deceptive, <0.40 clear
- 02-02: Edge case handling: missing landmarks return NaN, contact<5 uses available frames
- 02-03: Double-smoothing pipeline (sigma=1.5, 1.0) prevents MediaPipe jitter amplification in angular velocity
- 02-03: Angular velocities clipped to <5000 deg/s (realistic human joint maximum)
- 02-03: Racket head speed from wrist velocity proxy (research correlation r=0.72-0.74)
- 02-03: Robust statistics (median, percentiles) over mean for angular velocity features
- 02-03: Phase-specific features for all 5 phases with consistent naming: {phase}_{feature}_{stat}
- 02-03: Deceleration features implemented (FEAT-07) but expected low effect for Clear vs Smash
- 02-04: Filter methods run BEFORE wrapper methods (prevents overfitting on 658 features with 3,347 samples)
- 02-04: Cohen's d threshold=0.5 for medium effect (biomechanics research standard)
- 02-04: VIF threshold=10 for multicollinearity removal (research-validated acceptable threshold)
- 02-04: RFECV uses Random Forest with regularization (max_depth=10, min_samples_split=10)
- 02-04: F1 scoring for RFECV (balanced metric for binary classification)
- 02-04: Target 254 features (N_train/10 rule with 2,554 training samples)
- 02-04: Selected features saved to data/processed/features_v3/selected_features.json for v3 integration
- 02-05: extract_features_v3 combines v2 base + P0 + P1 features in unified pipeline
- 02-05: Feature selection applied via JSON manifest (graceful degradation if unpopulated)
- 02-05: Version gating via FeatureEngineering class maintains v2 backward compatibility
- 02-05: Metadata fields (feature_version, handedness, SIS) preserved during feature selection
- 02-05: V3 features saved to features_v3/ directory (separate from v2's features/ directory)

### Pending Todos

None yet.

### Blockers/Concerns

**Phase 1 complete - Infrastructure Foundation:**
- All 6 infrastructure requirements (INFRA-01 through INFRA-06) addressed
- Git LFS configured for model versioning (Plan 01-01)
- GCS bucket setup with MLflow integration (Plan 01-02)
- Bidirectional sync scripts for data transfer (Plan 01-03)
- Python 3.10 runtime setup for Colab (Plan 01-04)
- Terminal script execution framework (Plan 01-04)
- Infrastructure verification tools in place (Plan 01-04)

**Known issues:**
- Git LFS bandwidth limits (1GB/month free tier) - use Git LFS only for production models
- Colab defaults to Python 3.12; use colab_setup.sh to create Python 3.10 venv
- verify_infra.py may disconnect Colab session when importing TensorFlow (use granular flags)
- Full Colab verification deferred to actual training sessions

**Phase 2 progress:**
- Plan 02-01 complete: Phase segmentation with velocity-based detection
- Plan 02-02 complete: Kinetic chain timing and contact frame features (P0 features)
- Plan 02-03 complete: Angular velocity and phase-specific features (P1 features)
- Plan 02-04 complete: Two-stage feature selection pipeline (filter->wrapper)
- Plan 02-05 complete: Feature engineering v3 integration with version gating
- P0 features: ~20 features (7 kinetic + 8 contact + 6 intent + SIS)
- P1 features: ~32 features (6 angular + 4 racket + 20 phase + 2 deceleration)
- Current feature count: ~360-367 (v2 base: 308-315 + P0: 20 + P1: 32)
- Feature selection pipeline ready to reduce to <254 target (N_train/10 rule)
- V3 integration complete: unified extraction, version gating, backward compatibility
- Manifest-driven selection: graceful degradation if unpopulated (returns all features)
- Comprehensive test suite: 21 tests covering v3 extraction, version gating, validation
- Batch validation (85%+ pass rate) pending real dataset test in Colab
- Handedness detection assumes overhead strokes (may need refinement)
- Deceleration features may have low Cohen's d for Clear vs Smash (dataset is Clear+Smash only)

**Phase 3 readiness:**
- Train-test split must prevent player leakage (same player in both sets)
- External video generalization unknown (ShuttleSet is professional players only)

**Phase 4 readiness:**
- Model-benchmark integration requires careful version compatibility management

## Session Continuity

Last session: 2026-01-30T16:02:11Z
Stopped at: Completed 02-05-PLAN.md (Feature Engineering v3 Integration) - Phase 2 in progress (5/9 plans)
Resume file: None
Next: 02-06-PLAN.md (next feature engineering plan)

---

*This file tracks project state across all phases and sessions. Read first in every workflow.*
