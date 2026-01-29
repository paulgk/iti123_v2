# Research Summary: v1.1 Milestone

**Project:** AI Badminton Coaching App v1.1
**Milestone Goals:** Coach-Informed ML + Colab Infrastructure
**Research Date:** 2026-01-29
**Overall Confidence:** HIGH

---

## Executive Summary

The v1.1 milestone expansion presents a well-defined but technically constrained challenge. The core objective is to improve ML classification accuracy (currently 45% - barely above random chance) by adding coach-informed biomechanical features while establishing a sustainable Colab Enterprise training workflow integrated with Git LFS for dataset versioning.

**The Good News:** The technical stack is solid and requires minimal additions. The existing Python 3.10 + TensorFlow 2.15 + MediaPipe 0.10.9 foundation remains unchanged. Research reveals clear discriminative features from badminton biomechanics literature: kinetic chain timing (shoulder→elbow→wrist sequencing), phase-based analysis (preparation, backswing, forward swing, contact, follow-through), and contact point precision are the keys to distinguishing Clear vs Smash strokes.

**The Critical Risk:** Three interconnected pitfalls threaten the milestone: (1) Git LFS bandwidth exhaustion on GitHub's free tier (1GB/month limits), (2) feature engineering explosion on a small dataset (3,347 samples) leading to severe overfitting, and (3) Colab Enterprise's ephemeral sessions causing data loss without proper checkpointing. These risks compound each other - LFS costs discourage checkpointing, which increases data loss risk, forcing re-runs that consume more bandwidth.

**The Recommended Approach:** Use a layered integration pattern where Colab augments (not replaces) the existing benchmark-based system. Store videos in Google Cloud Storage (not Git LFS), add 50-150 high-value biomechanical features (not 200+), implement GCS checkpointing every 100 iterations, and maintain the benchmark analysis as the production default with ML as an experimental enhancement. Feature selection and domain validation must precede feature expansion - adding features without validation will worsen accuracy, not improve it.

---

## Key Findings

### From STACK.md: Infrastructure Additions (Confidence: HIGH)

**Core Stack Unchanged:**
- Python 3.10 (CRITICAL: Must not upgrade to 3.12 - breaks TensorFlow 2.15 compatibility)
- TensorFlow 2.15.x (pinned for MediaPipe 0.10.9 compatibility)
- MediaPipe 0.10.9 (stable pose estimation)
- All existing packages remain as-is

**Critical Additions:**
1. **Git LFS 3.7.1** - For video storage, BUT with severe bandwidth constraints (1GB/month free tier)
2. **MLflow 3.9.0** - Experiment tracking and model versioning
3. **Kineticstoolkit 0.17.0** - Optional biomechanical analysis helpers (can use existing SciPy instead)
4. **Colab Enterprise Runtime** - Must configure Python 3.10 explicitly (default 3.12 breaks stack)
5. **Keras 2.x constraint** - Pin `keras>=2.15.0,<3.0.0` to prevent MLflow incompatibility

**Critical Compatibility Note:**
Colab Enterprise defaults to Python 3.12 as of May 2025. TensorFlow 2.15 does NOT support Python 3.12. Custom runtime template with Python 3.10 is mandatory.

**Git LFS Decision:**
Git LFS is simpler than DVC for this use case, but GitHub's free tier (1GB storage + 1GB/month bandwidth) is insufficient for full video dataset. Recommendation: Store only 20-50 validation videos in LFS, use Google Cloud Storage for bulk dataset.

**Why NOT alternatives:**
- DVC: Overkill for simple versioning needs, adds pipeline complexity
- pyomeca: Too research-oriented, current SciPy approach sufficient
- TensorFlow 2.16+: Breaks MediaPipe compatibility
- Jupyter in Colab: Project uses terminal scripts for reproducibility

---

### From FEATURES.md: Coach-Informed Biomechanics (Confidence: HIGH for table stakes, MEDIUM for advanced)

**Current State:** 427 features with 45% accuracy (random chance). Feature extraction works but lacks discriminative power.

**Table Stakes Features (MUST ADD):**

1. **Kinetic Chain Sequencing** (CRITICAL MISSING):
   - Hip rotation timing → Trunk rotation → Shoulder rotation → Elbow extension → Wrist pronation
   - Elite players show proximal-to-distal energy transfer with precise timing
   - Implementation: Calculate peak velocity timing for each segment, measure time deltas
   - Expected impact: +15-20% accuracy improvement

2. **Phase-Based Analysis** (PARTIALLY MISSING):
   - Current system aggregates across entire stroke (mean, max, std)
   - Need: Phase segmentation (preparation → backswing → forward swing → contact → follow-through)
   - Different phases have different biomechanical requirements
   - Implementation: Velocity-based heuristics for phase boundaries
   - Expected impact: +10-15% accuracy improvement

3. **Contact Point Features** (PARTIALLY PRESENT):
   - Height (Y-axis): Clear higher than Smash
   - Forward reach (Z-axis): More forward = more power
   - Consistency: Low variance = better technique
   - Add: Contact point consistency metrics, stroke-specific optimal zones

4. **Racket Head Speed** (PRESENT but needs enhancement):
   - Current: Wrist velocity as proxy
   - Enhancement: Racket tip speed = wrist_velocity + (forearm_angular_velocity × racket_length)
   - Benchmark values: Elite smash 61-71 m/s, Clear 45-53 m/s (75% of smash)

5. **Wrist/Forearm Orientation** (PRESENT but needs expansion):
   - Forearm vertical angle: Upward for Clear (45-70°), Downward for Smash (110-140°)
   - Add: Forearm rotation velocity (pronation rate - critical for power)
   - Limitation: MediaPipe Pose doesn't track wrist flexion (needs hand landmarks)

6. **Lower Limb Features** (MISSING - MODERATE IMPORTANCE):
   - Ankle dorsiflexion, knee flexion at preparation, hip rotation
   - Power generation starts from ground, proper footwork enables upper body mechanics
   - Lower priority than upper body for overhead strokes

**Differentiators (Should-Have):**
- Stroke phase transition timing (smooth vs jerky)
- Symmetry analysis (left vs right arm)
- Movement efficiency metrics (jerk minimization)
- Deceleration control (follow-through smoothness)

**Anti-Features (DO NOT ADD):**
- Static pose features without temporal context (badminton is dynamic)
- Raw landmark coordinates (not biomechanically meaningful)
- Over-smoothing trajectories (loses high-frequency movements)
- Aggregate features only (need phase-specific + temporal)

**Stroke-Specific Considerations:**

| Stroke Type | Primary Discriminators | Current Coverage |
|-------------|------------------------|------------------|
| Clear | Contact height (moderate), forearm angle upward (45-70°), racket speed 45-53 m/s | Good |
| Smash | Contact height (very high), forearm angle downward (110-140°), explosive pronation, racket speed 61-71 m/s | Good for standing, missing jump features |
| Drop | Deceptive preparation (same as smash until late), rapid deceleration at contact, shorter follow-through | Missing - no deceleration features |
| Drive | Contact at shoulder height, horizontal swing plane, minimal trunk rotation | Missing - system designed for overhead strokes |
| Net | Very low contact (chest-waist), minimal arm extension, wrist-dominant motion | Missing - completely different biomechanics |

**Priority Matrix:**

| Feature Category | Impact on ML | Impact on Feedback | Complexity | Priority |
|------------------|--------------|-------------------|------------|----------|
| Kinetic chain sequencing | HIGH | HIGH | Medium-High | **P0 - CRITICAL** |
| Phase-based features | HIGH | HIGH | Medium | **P0 - CRITICAL** |
| Contact point enhancement | HIGH | HIGH | Low | **P0 - CRITICAL** |
| Wrist/forearm velocity | HIGH | MEDIUM | Medium | **P1 - HIGH** |
| Deceleration control | MEDIUM | HIGH | Medium | **P1 - HIGH** |
| Drop shot features | HIGH | MEDIUM | Medium | **P1 - HIGH** |
| Lower limb features | MEDIUM | MEDIUM | Low-Medium | **P2 - MEDIUM** |

**Research Gaps:**
- Drop shot biomechanics: Limited peer-reviewed research
- Drive & Net shots: Most research focuses on overhead power strokes
- Real-time phase segmentation: No standard algorithm, must use heuristics

---

### From ARCHITECTURE.md: Integration Patterns (Confidence: MEDIUM)

**Current Architecture (v1.0):**
- Local development only
- Benchmark-based analysis (NO ML in production)
- 427 features → percentile comparison → feedback generation
- Works well for demo and academic submission

**Target Architecture (v1.1):**
- **Layered integration**: Colab augments (not replaces) existing pipeline
- **Bidirectional git workflow**: Local → git → Colab → git → local
- **Script-based execution**: Terminal mode Python (not notebooks) for reproducibility
- **Dual-mode inference**: Benchmark as default, ML as experimental enhancement

**Key Architectural Patterns:**

1. **Version-Gated Feature Engineering:**
   ```python
   def extract_features(poses, version='v3'):
       if version == 'v2':
           return _extract_v2_features(poses)  # 427 features (production)
       elif version == 'v3':
           v2 = _extract_v2_features(poses)
           coach = _extract_coach_features(poses)
           return {**v2, **coach, '_version': 'v3'}  # 577 features (experimental)
   ```
   - Maintains backward compatibility
   - Allows A/B testing
   - Supports gradual rollout

2. **Selective Git LFS Pulling:**
   ```bash
   GIT_LFS_SKIP_SMUDGE=1 git clone repo  # Fast clone without LFS
   git lfs pull --include="data/processed/splits/*"  # Only training data
   ```
   - Reduces bandwidth costs
   - Only downloads required files
   - Critical for Colab workflow

3. **Dual-Mode Inference (Benchmark + ML):**
   ```python
   if mode == 'ml' and confidence > 0.85:
       return ml_feedback(prediction, features)
   else:
       return benchmark_feedback(features, stroke_type)  # Fallback
   ```
   - Safe rollout without breaking existing functionality
   - Easy rollback if ML underperforms
   - Production default remains benchmark-based

4. **Pipeline Orchestration via Terminal Scripts:**
   - `scripts/train/train_pipeline.py` - Main orchestrator
   - `scripts/evaluate/evaluate_model.py` - Validation
   - Colab wrapper notebook minimal (just runs scripts)
   - Version control friendly, reproducible, automatable

**Data Flow:**

```
LOCAL: Feature Engineering → Git push (code + metadata)
   ↓
GIT: Version control + Git LFS (minimal files only)
   ↓
COLAB: Clone repo → Train models → Push results
   ↓
LOCAL: Pull models → Evaluate → Integrate if accuracy > 85%
```

**Component Responsibilities:**

| Component | Responsibility | Location | Status |
|-----------|----------------|----------|--------|
| PoseExtractor | MediaPipe pose extraction | Local + Colab | Unchanged |
| FeatureEngineering V3 | Expanded biomechanical features | Local + Colab | NEW |
| CoachingFeedback | Benchmark-based analysis | Local only | Unchanged (production) |
| TrainingPipeline | Model retraining orchestration | Colab only | NEW |
| ClassificationModel | ML stroke classification | Colab (train) + Local (inference) | UPDATED |
| ModelEvaluator | Accuracy validation | Local + Colab | NEW |

**Recommended Project Structure:**

```
iti123_v2/
├── src/                     # Production code (unchanged)
│   ├── data_processing/
│   │   ├── feature_engineering_v2.py  # PRESERVED (427 features)
│   │   └── feature_engineering_v3.py  # NEW (577 features)
│   └── coaching/            # Benchmark system (unchanged)
├── scripts/                 # NEW: Training scripts for Colab
│   ├── train/
│   │   └── train_pipeline.py
│   ├── evaluate/
│   │   └── evaluate_model.py
│   └── setup/
│       └── setup_colab_env.sh
├── configs/                 # NEW: YAML configurations
│   └── train_config.yaml
├── data/                    # GCS storage (not Git LFS)
├── models/saved/            # Git LFS tracked
└── outputs/metrics/         # Regular git (JSON/CSV)
```

**Data Versioning Strategy:**

| Data Type | Storage | Versioning |
|-----------|---------|------------|
| Raw Videos | Google Cloud Storage | By filename |
| Extracted Poses | GCS | By clip name |
| Features | GCS + version metadata | `_version` field in dict |
| Trained Models | Git LFS + registry.json | Model ID + metadata |
| Benchmarks | Regular git | Small .pkl files |
| Metrics | Regular git | JSON files |

**Critical Decision: Use GCS, Not Git LFS for Bulk Data**

GitHub LFS free tier (1GB storage + 1GB/month bandwidth) is insufficient:
- 3,347 clips × 2MB/clip ≈ 6.7 GB (exceeds free storage)
- Colab runtime resets = multiple clones per day
- 5 clones/week × 6.7 GB = 33.5 GB/week (33× over bandwidth limit)
- Cost: ~$170/month for additional bandwidth

**Alternative (Recommended):**
- Store videos in Google Cloud Storage (native Colab integration)
- Use Git LFS only for: Final model files, 20-50 validation videos
- Estimated monthly cost: $0 (stays under free tier)

---

### From PITFALLS.md: Critical Integration Risks (Confidence: HIGH)

**Overall Risk Level:** HIGH - Three critical pitfalls can block progress

**CRITICAL PITFALLS (Must Fix Before Proceeding):**

1. **Git LFS Bandwidth Exhaustion** (Severity: CRITICAL):
   - GitHub free tier: 1GB storage + 1GB/month bandwidth
   - Colab workflow: Multiple clones per day consume bandwidth rapidly
   - ShuttleSet subset: ~6.7 GB (exceeds free tier)
   - Result: LFS disabled mid-experiment, files held "ransom"
   - **Prevention:**
     - Use Google Cloud Storage for bulk videos (NOT Git LFS)
     - Track only 20-50 validation videos with LFS
     - Shallow clone: `git clone --depth 1`
     - Selective LFS: `git lfs pull --include="data/validation_clips/*.mp4"`
   - **Detection:** `git push` slow, GitHub bandwidth warning emails
   - **Phase Impact:** Phase 1 (Infrastructure Setup) MUST solve this

2. **Feature Engineering Explosion Leading to Overfitting** (Severity: CRITICAL):
   - Current: 427 features, 3,347 samples (12.8% feature-to-sample ratio)
   - Common mistake: "More features = better model"
   - Small dataset (2,554 training samples after split) + high dimensionality = severe overfitting
   - Example: 600 features → 95% train accuracy, 42% test accuracy (worse than v1.0)
   - Badminton-specific: Clear and Smash have "kinematic homogeneity" - similar movement patterns
   - **Prevention:**
     - Feature selection BEFORE expansion (Cohen's d > 0.5 for medium effect)
     - Keep feature count < N_train/10 (< 254 for 2,554 samples)
     - Dimensionality reduction: PCA to 50-100 components
     - Cross-validation with stratification + player grouping
     - Regularization: L1 (Lasso) or L2 (Ridge), dropout 0.3-0.5
     - Data augmentation: Mirror flips, temporal jitter, Gaussian noise
     - Validate each feature: Physical interpretation, measurement reliability, discriminative power
   - **Detection:** Train accuracy > 90%, test accuracy < 60% (20%+ gap)
   - **Phase Impact:** Phase 2 (Feature Engineering) MUST include selection analysis

3. **Colab Enterprise Data Loss from Ephemeral Sessions** (Severity: CRITICAL):
   - Colab timeout: 90 minutes idle, 12 hours maximum
   - Kernel crash, network interruption → all in-memory data lost
   - Example: 2.5 hours of pose extraction lost on network hiccup
   - Git push failure: Large files (>100MB) block commit silently
   - **Prevention:**
     - Checkpoint to GCS every 100 iterations (native Colab integration)
     - Auto-commit in script (not manual): `git add; git commit; git push`
     - Separate data from code: Code in git, data in GCS
     - Model versioning with MLflow (GCS backend, not git)
     - Emergency checkpoint handler: `signal.signal(signal.SIGTERM, emergency_checkpoint)`
     - Validate push succeeded: Compare local HEAD with remote
   - **Detection:** Colab disconnection warnings, missing experiment results
   - **Phase Impact:** Phase 1 (Infrastructure Setup) MUST include checkpointing safeguards

**HIGH PRIORITY PITFALLS:**

4. **Git LFS .gitattributes Misconfiguration:**
   - Tracking files already in git history → file exists twice
   - `git lfs untrack` doesn't work as expected
   - Wildcard overmatch: `*.mp4` tracks ALL mp4 files everywhere
   - Forgetting to commit .gitattributes
   - **Prevention:** Use `git lfs migrate` for existing files, specific paths only, verify before adding

5. **Feature Leakage from Improper Train/Test Split:**
   - Player leakage: Same player in train and test sets
   - Temporal leakage: Consecutive strokes correlated
   - Global normalization leakage: Fit scaler on all data (including test)
   - **Prevention:** Group-based splitting by player, fit preprocessing on training only, stratified splits

6. **Model-Benchmark Integration Fragility:**
   - Dual analysis paths: Which result to trust when they disagree?
   - Feature compatibility: ML uses 600 features, benchmarks expect 427
   - Version mismatch: Benchmarks from forehand-only, ML trained on expanded dataset
   - **Prevention:** Unified feature interface, version-gated extraction, separate forehand/backhand benchmarks

7. **Terminal Script Reliability in Colab Enterprise:**
   - Paths break between notebook and terminal (cwd differences)
   - Environment variables not propagated
   - Dependency version conflicts (Colab defaults vs requirements.txt)
   - Script output not captured
   - **Prevention:** Absolute paths, environment setup script, redirect output to log files, preflight checks

**MEDIUM PRIORITY PITFALLS:**

8. **Insufficient Badminton Domain Knowledge:**
   - Copying tennis/baseball features (different biomechanics)
   - Teaching cues vs biomechanical reality
   - Unmeasurable from video (grip pressure, shuttle contact sound)
   - **Prevention:** Literature review BEFORE feature engineering, validate with domain expert, ablation testing

9. **No Experiment Tracking:**
   - Lose critical information: Which features? Which hyperparameters? Which data split?
   - Can't reproduce "best" model from last week
   - **Prevention:** MLflow or W&B, log everything, git tags for model versions

**Phase-Specific Warning Summary:**

| Phase | Critical Risks | Must Have Before Proceeding |
|-------|----------------|----------------------------|
| Phase 1: Infrastructure | LFS bandwidth, Colab data loss, .gitattributes | GCS setup, checkpoint function tested, LFS validated with small files |
| Phase 2: Feature Engineering | Feature explosion, feature leakage, domain gaps | Literature review, feature selection analysis, split strategy validated |
| Phase 3: Model Training | Overfitting, no experiment tracking, integration fragility | Regularization, cross-validation, MLflow setup, integration tests |
| Phase 4: Validation | Terminal reliability, feature leakage test set | Preflight checks, external video testing, accuracy threshold validation |

**Interconnected Risk Cycle:**

```
Git LFS bandwidth limits
    ↓
Discourage frequent checkpointing
    ↓
Increase Colab data loss risk
    ↓
Force re-runs of feature engineering
    ↓
Consume more LFS bandwidth
    ↓ (vicious cycle)
```

**Break the cycle:** Use GCS for data, git for code only.

---

## Implications for Roadmap

### Recommended Phase Structure

Based on combined research, the v1.1 milestone should be structured as 4 sequential phases with clear validation gates:

**Phase 1: Infrastructure Foundation (Week 1)**
- **What it delivers:** Reliable data pipeline with no loss risk
- **Features:**
  - Google Cloud Storage bucket setup (replace Git LFS for bulk data)
  - Git LFS configuration for models only (20-50 validation videos max)
  - Colab runtime template with Python 3.10
  - Checkpoint-to-GCS function (tested with SIGTERM simulation)
  - Terminal script workflow end-to-end (clone → run → commit → push)
  - MLflow experiment tracking configured (GCS backend)
- **Validation gate:**
  - Checkpoint function proven to work (simulate timeout)
  - Git push verified (local HEAD == remote HEAD)
  - Bandwidth usage < 500 MB in first week
  - Terminal scripts execute successfully in Colab
- **Pitfalls to avoid:** Git LFS bandwidth trap (P1), Colab data loss (P3), .gitattributes config (P4)
- **Rationale:** Infrastructure issues will block all subsequent work. Solve once, benefit throughout.

**Phase 2: Feature Engineering Enhancement (Week 2)**
- **What it delivers:** Validated set of 50-150 high-impact biomechanical features
- **Features from FEATURES.md:**
  - **P0 Features (CRITICAL):**
    - Kinetic chain timing: Hip → Trunk → Shoulder → Elbow → Wrist sequential peaks
    - Phase segmentation: Velocity-based detection of preparation/backswing/forward/contact/follow-through
    - Contact point consistency metrics
  - **P1 Features (HIGH):**
    - Forearm rotation velocity (pronation rate)
    - Racket head speed estimation (wrist velocity + angular velocity × racket length)
    - Deceleration control (follow-through smoothness)
  - Keep existing 427 features in `feature_engineering_v2.py` (backward compatibility)
  - Create `feature_engineering_v3.py` with version flag
- **Validation gate:**
  - Literature review completed (3+ badminton biomechanics papers)
  - Feature selection analysis: Each new feature has Cohen's d > 0.5
  - Feature count < N_train/10 (< 254 features for 2,554 samples)
  - No feature correlation > 0.95 (redundancy check)
  - Train/test split validated (no player leakage, group-based splitting)
  - Ablation test: Each feature improves accuracy by >1%
- **Pitfalls to avoid:** Feature explosion (P2), Feature leakage (P5), Domain knowledge gaps (P8)
- **Rationale:** Feature quality matters more than quantity. Discriminative features from literature are more valuable than 100 random features.
- **Research flags:** May need deeper research on Drop shot biomechanics (limited literature)

**Phase 3: Model Training & Evaluation (Week 3)**
- **What it delivers:** Trained models with accuracy > baseline, integration-ready
- **Features:**
  - Train Random Forest, SVM, LSTM on enhanced features (v3)
  - Cross-validation: Stratified + grouped by player ID (5-fold)
  - Regularization: L1 (Lasso) for feature selection, early stopping
  - Hyperparameter tuning (if promising results)
  - Model-benchmark integration testing
  - External validation videos (not from ShuttleSet)
- **Validation gate:**
  - Test accuracy > 70% (vs current 45%)
  - Train-test accuracy gap < 15% (overfitting check)
  - F1 score > 0.75
  - External video accuracy > 65% (generalization check)
  - Feature importance analysis shows kinetic chain timing in top 10
  - MLflow logs complete (reproducible experiment)
- **Pitfalls to avoid:** Overfitting (P2), No experiment tracking (P9), Integration fragility (P6)
- **Rationale:** Validate improvement before integrating into production. If accuracy doesn't improve significantly, stay with benchmark-based analysis.
- **Research flags:** Standard ML patterns well-documented, no additional research needed

**Phase 4: Production Integration & Validation (Week 4)**
- **What it delivers:** Dual-mode system (benchmark + ML) with confidence-based switching
- **Features:**
  - Create `model_loader.py` for production model loading
  - Update `streamlit_app.py` with optional ML mode
  - Implement confidence threshold (0.85) for ML classification
  - Fallback to benchmark analysis if confidence < threshold
  - A/B testing on user videos
  - Documentation and deployment guide
- **Validation gate:**
  - ML mode works alongside benchmark mode (no breaking changes)
  - Confidence threshold validated (precision > 90% above threshold)
  - User feedback: ML predictions match user expectations
  - Performance acceptable (inference < 5 seconds per video)
  - Rollback plan tested (can disable ML mode with single config change)
- **Pitfalls to avoid:** Terminal script reliability (P7), Integration breaking existing system
- **Rationale:** Safe rollout with fallback. ML augments rather than replaces existing system.

### Research Flags: Which Phases Need Deeper Research?

| Phase | Need `/gsd:research-phase`? | Rationale |
|-------|----------------------------|-----------|
| Phase 1: Infrastructure | NO | Standard patterns: GCS, Git LFS, Colab integration are well-documented |
| Phase 2: Feature Engineering | **YES - CONDITIONAL** | If expanding to Drop/Drive/Net shots: Limited literature on these stroke types. If staying with Clear/Smash: NO (well-researched) |
| Phase 3: Model Training | NO | Standard ML techniques: Cross-validation, regularization, hyperparameter tuning are well-documented |
| Phase 4: Production Integration | NO | Standard deployment patterns: Dual-mode systems, confidence thresholding are well-documented |

**Conditional research trigger:** If Phase 2 decides to add Drop shot features and accuracy is low, consider deeper research on Drop shot biomechanics (limited published research available).

### Dependencies & Critical Path

```
Phase 1 (Infrastructure) - CRITICAL PATH
    ↓ (blocks everything)
Phase 2 (Feature Engineering) - CRITICAL PATH
    ↓ (features needed for training)
Phase 3 (Model Training) - CRITICAL PATH
    ↓ (must validate accuracy before integration)
Phase 4 (Production Integration)
    ↓
COMPLETE (if accuracy > threshold)
    OR
ITERATE on Phase 2 (if accuracy < threshold)
```

**No parallel work possible** - Each phase depends on previous phase validation.

### Success Criteria by Phase

**Phase 1 Success:**
- [ ] GCS bucket accessible from Colab
- [ ] Checkpoint function saves every 100 iterations without failure
- [ ] Git workflow completes end-to-end (clone → edit → commit → push → verify)
- [ ] LFS bandwidth < 100 MB in test week
- [ ] Terminal scripts work identically in local and Colab

**Phase 2 Success:**
- [ ] 50-150 features added (not 200+)
- [ ] Feature count < N_train/10
- [ ] Each feature validated against literature
- [ ] No player leakage in train/test split
- [ ] Feature correlation matrix clean (no r > 0.95)
- [ ] Ablation test: Added features improve baseline

**Phase 3 Success:**
- [ ] Test accuracy > 70% (baseline 45%)
- [ ] Train-test gap < 15%
- [ ] F1 score > 0.75
- [ ] External video accuracy > 65%
- [ ] Experiment fully logged in MLflow
- [ ] Model artifacts < 50 MB each

**Phase 4 Success:**
- [ ] ML mode works alongside benchmark mode
- [ ] Confidence threshold > 0.85 gives precision > 90%
- [ ] Inference time < 5 seconds
- [ ] User acceptance: ML predictions feel accurate
- [ ] Rollback tested and documented

**Overall Milestone Success:**
- [ ] ML classification accuracy > 70% (vs baseline 45%)
- [ ] System maintains 100% backward compatibility (benchmark mode untouched)
- [ ] No data loss incidents during development
- [ ] Git LFS bandwidth < 1 GB/month
- [ ] Documentation complete for future iterations

---

## Confidence Assessment

| Research Area | Confidence Level | Evidence Quality | Gaps Identified |
|---------------|------------------|------------------|-----------------|
| **Technology Stack** | **HIGH** | Official documentation, version compatibility verified, existing codebase analysis | None - stack well-defined |
| **Biomechanical Features** | **HIGH** (table stakes)<br>**MEDIUM** (advanced) | Multiple peer-reviewed papers (2024-2025), badminton-specific research, coach training guides | Drop shot biomechanics (limited research), Drive/Net shots (minimal literature) |
| **Architecture Patterns** | **MEDIUM** | General MLOps patterns, Colab integration guides, Git LFS documentation | Colab Enterprise-specific edge cases, terminal script reliability in production |
| **Integration Risks** | **HIGH** | Verified against official GitHub LFS docs, ML overfitting research (2024), real Colab limitations | Small dataset overfitting mitigation (need experimentation) |

### Confidence by Decision Type

**Infrastructure Decisions (HIGH confidence):**
- Use Python 3.10, TensorFlow 2.15, MediaPipe 0.10.9: VERIFIED via version compatibility matrix
- Use GCS instead of Git LFS for bulk data: VERIFIED via bandwidth calculations and cost analysis
- Use MLflow for experiment tracking: STANDARD pattern in MLOps, well-documented
- Checkpoint to GCS every 100 iterations: PROVEN pattern for ephemeral environments

**Feature Engineering Decisions (HIGH-MEDIUM confidence):**
- Kinetic chain timing is discriminative: HIGH (multiple peer-reviewed studies 2024-2025)
- Phase-based analysis improves accuracy: HIGH (standard in sports biomechanics)
- Contact point precision matters: HIGH (validated in skill-level comparison studies)
- Drop shot features: MEDIUM (limited specific research, must infer from smash studies)
- Lower limb features less critical than upper body: MEDIUM (some research, but less emphasis)

**Architecture Decisions (MEDIUM confidence):**
- Layered integration pattern (Colab augments, not replaces): MEDIUM (logical but untested in this context)
- Dual-mode inference with confidence threshold: MEDIUM (standard pattern but threshold value needs tuning)
- Terminal scripts over notebooks in Colab: MEDIUM (best practice but edge cases possible)
- Model-benchmark integration strategy: MEDIUM (requires experimentation to validate)

**Risk Assessment Decisions (HIGH confidence):**
- Git LFS bandwidth exhaustion is real: HIGH (verified in GitHub docs, reported in multiple sources)
- Feature explosion causes overfitting on small datasets: HIGH (standard ML phenomenon, recent research 2024)
- Colab data loss is significant risk: HIGH (official Colab docs, widely reported issue)

### Gaps That Need Attention During Planning

1. **Drop Shot Feature Definition (MEDIUM GAP):**
   - Limited peer-reviewed research on drop shot biomechanics specifically
   - Must infer from smash studies + coaching videos
   - **Mitigation:** Start with Clear/Smash only (well-researched), defer Drop to v1.2 if needed
   - **When to address:** Phase 2 (Feature Engineering), during literature review

2. **Optimal Feature Count (SMALL GAP):**
   - Research says < N/10, but optimal count depends on feature quality
   - 254 features is upper bound for 2,554 samples, but what's the sweet spot?
   - **Mitigation:** Use cross-validation with varying feature counts, plot learning curves
   - **When to address:** Phase 3 (Model Training), during experimentation

3. **Confidence Threshold for Dual-Mode Inference (SMALL GAP):**
   - 0.85 is standard, but depends on model calibration
   - **Mitigation:** Validate threshold on held-out set, plot precision-recall curves
   - **When to address:** Phase 4 (Production Integration), during validation

4. **External Video Generalization (MEDIUM GAP):**
   - ShuttleSet is professional players only
   - Will features generalize to recreational players with different techniques?
   - **Mitigation:** Test on external videos (YouTube badminton tutorials), adjust features if needed
   - **When to address:** Phase 3 (Model Training), during external validation

5. **Colab Terminal Script Edge Cases (SMALL GAP):**
   - Official docs limited on terminal script reliability in Colab Enterprise
   - **Mitigation:** Extensive testing in Phase 1, preflight checks before expensive operations
   - **When to address:** Phase 1 (Infrastructure Setup), during terminal script testing

### Honest Assessment of Unknowns

**We Know:**
- TensorFlow 2.15 + MediaPipe 0.10.9 compatibility (verified)
- Git LFS bandwidth limits will be exceeded (calculated)
- Kinetic chain timing is discriminative (peer-reviewed)
- Small dataset + many features = overfitting (ML fundamentals)

**We Think We Know:**
- 50-150 features is safe range (based on N/10 rule + literature precedent)
- GCS checkpointing will prevent data loss (standard pattern, untested in this project)
- Phase-based analysis will improve accuracy by 10-15% (logical but not measured)
- Dual-mode inference at 0.85 confidence will work (standard threshold, needs tuning)

**We Don't Know:**
- Will enhanced features actually improve accuracy beyond 70%? (must experiment)
- What's the optimal feature count for this specific dataset? (must tune)
- Will external video generalization be acceptable? (must validate)
- Are terminal scripts reliable in Colab Enterprise for 12-hour runs? (must test)

**Honest Risk:** There's a 30-40% chance that enhanced features don't improve accuracy enough to justify ML integration. In that case, stay with benchmark-based analysis (existing v1.0) and defer ML to v2.0 with larger dataset.

---

## Key Sources

### Technology Stack
- [Git LFS Official Site](https://git-lfs.com/) - v3.7.1 installation and usage
- [GitHub Git LFS Billing](https://docs.github.com/billing/managing-billing-for-git-large-file-storage/about-billing-for-git-large-file-storage) - Storage and bandwidth limits
- [Colab Enterprise Runtimes](https://docs.cloud.google.com/colab/docs/runtimes) - Python 3.10 configuration
- [MLflow 3.9.0 Release](https://mlflow.org/releases) - Experiment tracking features
- [TensorFlow Python 3.12 Support Discussion](https://discuss.python.org/t/tensorflow-support-for-python-3-12/66346) - Version compatibility

### Biomechanical Features
- [Muscle Synergy Analysis (2025)](https://pmc.ncbi.nlm.nih.gov/articles/PMC12170632/) - Kinetic chain, sequential muscle activation
- [Novice vs. Skilled Player Comparison (2023)](https://www.mdpi.com/2076-3417/13/22/12488) - Kinematic differences, contact point
- [Racket Head Speed Study (2023)](https://www.nature.com/articles/s41598-023-37108-x) - Elite racket speeds (61-71 m/s smash)
- [Lower Limb Biomechanics in Clear Strokes](https://pmc.ncbi.nlm.nih.gov/articles/PMC6348812/) - Ankle, knee, hip features
- [Biomechanical Principles for Power Strokes](https://ojs.ub.uni-konstanz.de/cpa/article/download/2233/2089/) - Fundamental biomechanics, joint sequencing
- [Stroke Phases Diagram](https://www.researchgate.net/figure/Basic-phases-of-a-badminton-stroke-backswing-Frames-1-7-forward-swing-Frames-7-10_fig2_233782806) - Five-phase model

### Architecture & MLOps
- [Git LFS and DVC: Managing Large Artifacts](https://medium.com/@pablojusue/git-lfs-and-dvc-the-ultimate-guide-to-managing-large-artifacts-in-mlops-c1c926e6c5f4)
- [Combine GitHub and Google Colab](https://tilburgsciencehub.com/topics/automation/replicability/cloud-computing/colab-github/)
- [MLOps: Continuous Delivery Pipelines](https://docs.cloud.google.com/architecture/mlops-continuous-delivery-and-automation-pipelines-in-machine-learning)
- [Feature Store: The Definitive Guide](https://www.hopsworks.ai/dictionary/feature-store)

### Domain Pitfalls
- [GitHub LFS is Basically Paid Only (2024)](https://jamesoclaire.com/2024/12/06/github-large-file-storage-git-lfs-is-basically-paid-only/) - Bandwidth trap analysis
- [Estimation of Minimal Data Sets for ML (2024)](https://www.nature.com/articles/s41746-024-01360-w) - N=500-1000 mitigates overfitting
- [Badminton Stroke Kinematic Homogeneity (2025)](https://www.nature.com/articles/s41598-025-02771-9) - Clear/Smash similarity
- [Overfitting Prevention for Small Datasets](https://pmc.ncbi.nlm.nih.gov/articles/PMC8905023/)
- [Preventing Training Data Leakage](https://www.tonic.ai/blog/prevent-training-data-leakage-ai)

---

## Ready for Requirements Definition

This research synthesis provides:

1. **Clear technology choices:** Python 3.10 + TensorFlow 2.15 + MediaPipe 0.10.9 + MLflow 3.9.0 + GCS (not Git LFS for bulk data)

2. **Prioritized feature additions:** Kinetic chain timing (P0), phase segmentation (P0), contact point consistency (P0), deceleration control (P1), with clear discriminative power from literature

3. **Safe architectural pattern:** Layered integration where Colab augments (not replaces) existing system, with dual-mode inference and confidence-based fallback

4. **Critical risk mitigation:** GCS checkpointing every 100 iterations, feature selection before expansion (< N/10), group-based train/test split, MLflow experiment tracking

5. **Phased roadmap with validation gates:** 4-week timeline with clear success criteria per phase, sequential dependencies identified, conditional research triggers defined

6. **Honest assessment of gaps:** Drop shot biomechanics (limited research), optimal feature count (needs experimentation), external video generalization (needs validation), with 30-40% risk that ML doesn't outperform benchmarks

**Recommendation for orchestrator:** Proceed to requirements definition with confidence on infrastructure and table-stakes features. Consider iterative approach for Phase 2-3: Start with Clear/Smash only (well-researched), validate accuracy improvement, then expand to Drop/Drive/Net if successful.

**If timeline is constrained:** Use benchmark-based analysis only (existing v1.0), defer ML to v2.0 with larger dataset and more research. The existing system works well for demonstration and academic purposes.
