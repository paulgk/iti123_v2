# Domain Pitfalls: Colab Enterprise + Git LFS + Feature Engineering for Badminton ML

**Domain:** Sports analytics - Badminton biomechanics ML system expansion
**Researched:** 2026-01-29
**Confidence:** HIGH

---

## Executive Summary

Adding Colab Enterprise workflow, Git LFS for videos, and expanding from 427 to more features presents three critical integration risks:

1. **Git LFS bandwidth trap** - Free tier limits (1 GB/month) exhausted by CI/CD and collaborative workflows
2. **Feature engineering explosion** - Small dataset (3,347 samples) + high dimensionality = severe overfitting
3. **Colab-git sync failures** - Data loss from ephemeral sessions without checkpointing

These pitfalls are interconnected: Git LFS costs discourage frequent checkpointing, which increases data loss risk in Colab, while feature explosion makes model files larger, consuming more LFS bandwidth.

---

## Critical Pitfalls

Mistakes that cause rewrites, data loss, or major budget/technical issues.

---

### Pitfall 1: Git LFS Bandwidth Exhaustion (Financial & Access Loss)

**What goes wrong:**

GitHub's free tier includes only 1 GB/month of LFS bandwidth. For a badminton video dataset:
- ShuttleSet has 4,983 clips (~3,347 usable forehand strokes)
- Average video size: 10-50 MB per clip (estimated from typical sports video datasets)
- Single clone/pull operation: Downloads ALL LFS-tracked files
- Colab Enterprise workflow: Multiple clones per day (new runtime = new clone)
- CI/CD integration: Every build that checks out repo consumes bandwidth

**Bandwidth consumption pattern:**
```
Day 1: Initial clone (300 MB) + user experimentation (3 pulls) = 1.2 GB
Day 2: LFS DISABLED - exceeded free tier by 20%
Cost to restore: $5/month for 50 GB pack
```

**Why it happens:**

1. **Equal storage/bandwidth quotas don't match usage** - You download files far more often than you upload new ones
2. **Colab runtime resets** - Each new session clones the entire repo including LFS files
3. **Untracked file accumulation** - Adding more videos to LFS without pruning old experiments
4. **CI/CD multiplier** - Automated testing checkouts consume bandwidth invisibly

**Consequences:**

- LFS disabled mid-experiment (cannot pull new data)
- Files held "ransom" - must pay to access your own data
- .zip archive downloads exclude LFS files (only pointers)
- Team members blocked from accessing training videos
- Force-pushed fixes consume bandwidth on every collaborator's next pull

**Prevention:**

1. **DO NOT track all videos with LFS** - Only track essential subset:
   ```bash
   # WRONG: Track all videos
   git lfs track "data/**/*.mp4"

   # RIGHT: Track only benchmark validation set (20-50 videos max)
   git lfs track "data/validation_clips/*.mp4"
   ```

2. **Use external storage for bulk videos:**
   - **Google Cloud Storage (GCS)** - Colab Enterprise native integration
   - **Google Drive** - Mount in Colab, no LFS bandwidth
   - **Dataset hosting** - Kaggle, Hugging Face datasets (ShuttleSet is already published)

3. **Shallow clone in Colab:**
   ```bash
   # Reduce bandwidth by 60-80%
   git clone --depth 1 https://github.com/user/repo.git

   # Only fetch LFS files when needed
   GIT_LFS_SKIP_SMUDGE=1 git clone ...
   git lfs pull --include="data/validation_clips/*.mp4"
   ```

4. **Monitor bandwidth usage:**
   - GitHub Settings > Billing > Git LFS data usage
   - Set up alerts before hitting 80% of quota
   - Track Colab runtime resets (each reset = potential clone)

5. **Use .gitattributes selectively:**
   ```bash
   # Track only final model outputs, not intermediate checkpoints
   models/final_*.h5 filter=lfs diff=lfs merge=lfs -text

   # Do NOT track training checkpoints
   # models/checkpoint_*.h5  <- Keep in .gitignore
   ```

**Detection (warning signs):**

- `git push` suddenly slow (uploading large files)
- GitHub email: "You've used 80% of your LFS bandwidth"
- `git lfs ls-files` shows hundreds of tracked files
- Colab notebooks failing to clone with LFS errors
- Team members reporting "pointer files" instead of actual videos

**Phase recommendation:**

Phase 1 (Infrastructure Setup) MUST establish external storage pattern. Do not proceed to Phase 2 (Feature Engineering) without solving this.

**Sources:**
- [GitHub LFS billing documentation](https://docs.github.com/en/repositories/working-with-files/managing-large-files/about-storage-and-bandwidth-usage)
- [GitHub LFS is basically paid only (2024)](https://jamesoclaire.com/2024/12/06/github-large-file-storage-git-lfs-is-basically-paid-only/)
- [Why You Shouldn't Use Git LFS (2021)](https://gregoryszorc.com/blog/2021/05/12/why-you-shouldn't-use-git-lfs/)

---

### Pitfall 2: Feature Engineering Explosion Leading to Overfitting

**What goes wrong:**

Current system: 427 features from 3,347 training samples (13.5% feature-to-sample ratio)
Common mistake: "More features = better model" mindset

**The overfitting trap:**
```
v1.0: 427 features → 45% accuracy (random chance)
v1.1: Add 200+ "coach-informed" features → 600+ total features
Result: 95% training accuracy, 42% test accuracy (worse than v1.0)
```

**Why it happens:**

1. **Small dataset curse** - Research shows N=500-1000 mitigates overfitting, but performance converges at N=750-1500. With 3,347 samples split 76/9/15%, you have:
   - Training: 2,554 samples
   - Validation: 426 samples (too small for reliable early stopping)
   - Test: 675 samples

2. **Feature leakage from global statistics** - Calculating normalization parameters (mean, std) on full dataset before splitting gives model a "sneak peek" into test distribution

3. **Temporal leakage in badminton data** - If dataset contains multiple clips from same match or player:
   - Player 1's stroke in training set is correlated with Player 1's stroke in test set
   - Model learns player-specific quirks, not generalizable technique patterns

4. **Kinematic homogeneity** - Badminton research shows "occasional confusion in stroke classification with CNNs struggling to distinguish similar movement patterns" between Clear and Smash. Adding more features won't fix this if the motion patterns are inherently similar.

5. **Domain knowledge mismatch** - Coach-informed features may focus on **teaching cues** (e.g., "elbow high") rather than **discriminative biomechanics** (e.g., wrist angular velocity at shuttle contact)

**Specific to badminton biomechanics:**

Research on badminton clear vs smash shows discriminative features are:
- **Shuttle velocity**: 60.2% increase in smash (0.75 ratio clear/smash)
- **Wrist angular velocity** at impact
- **Proximal-to-distal kinetic chain timing** (shoulder → elbow → wrist sequence)
- **Racket angle at contact**
- **Full-body tension-arc formation** (multi-segment coordination)

Non-discriminative features often added:
- Static joint angles without temporal context
- Player height/anthropometrics (doesn't affect technique quality)
- Stance width (varies by player preference)
- Grip pressure (cannot be measured from video)

**Consequences:**

- Model memorizes training data, fails on new players
- Feature importance analysis shows noise features ranked high
- Model requires retraining for each new video (no generalization)
- Can't distinguish technique errors from player variation
- Feedback becomes player-specific, not actionable

**Prevention:**

1. **Feature selection BEFORE expansion:**
   ```python
   # Use Cohen's d to find discriminative features (already in codebase)
   # From derive_benchmarks.py approach

   # Only add features with |Cohen's d| > 0.5 (medium effect size)
   # Prioritize features with |Cohen's d| > 0.8 (large effect size)
   ```

2. **Dimensionality reduction:**
   - PCA to 50-100 components (preserve 95% variance)
   - Keep feature count < N/10 (< 254 features for 2,554 training samples)
   - Research shows high dimensionality risks overfitting even with 50,000 samples + 15,000 features

3. **Cross-validation with stratification:**
   - Stratified 5-fold CV (maintains 50/50 Clear/Smash balance)
   - Group by player ID (prevent player leakage across folds)
   - Temporal split if using match data (train on early matches, test on late matches)

4. **Regularization for small datasets:**
   - L1 (Lasso) for feature selection + regularization
   - L2 (Ridge) for correlated features
   - Dropout (0.3-0.5) for neural networks
   - Early stopping with validation loss (not validation accuracy)

5. **Data augmentation for biomechanics:**
   - Mirror flips (simulate opposite handedness)
   - Temporal jitter (±5 frames for stroke phase detection)
   - Gaussian noise on keypoints (±2% to simulate pose estimation variance)
   - Synthetic data via TVAEs (Tabular Variational Autoencoders) for underrepresented patterns

6. **Validate feature meaningfulness:**
   ```python
   # For each new feature:
   # 1. Physical interpretation - What biomechanical principle does it capture?
   # 2. Measurement reliability - Does MediaPipe capture this accurately?
   # 3. Discriminative power - Does it differ between Clear and Smash in literature?
   # 4. Correlation check - Is it redundant with existing features (r > 0.9)?
   ```

**Detection (warning signs):**

- Training accuracy > 90%, test accuracy < 60% (20%+ gap)
- Feature importance shows uninterpretable features ranked high
- Model performance degrades when adding external validation videos
- Correlation matrix shows many features with r > 0.95 (redundancy)
- Validation loss increases while training loss decreases (classic overfitting curve)
- Feature count > sample count / 10

**Phase recommendation:**

Phase 2 (Feature Engineering) MUST include:
- Feature selection analysis BEFORE adding new features
- Ablation study: Does adding feature X improve test accuracy?
- Literature review: Which features discriminate Clear vs Smash in published studies?

Do not proceed to Phase 3 (Model Training) until feature set is validated.

**Sources:**
- [Estimation of minimal data sets sizes for ML (2024)](https://www.nature.com/articles/s41746-024-01360-w) - N=500-1000 mitigates overfitting
- [Feature leakage in temporal data](https://towardsdatascience.com/two-rookie-mistakes-i-made-in-machine-learning-improper-data-splitting-and-data-leakage-3e33a99560ea/)
- [Badminton stroke kinematic homogeneity (2025)](https://www.nature.com/articles/s41598-025-02771-9)
- [Overfitting prevention for small datasets](https://pmc.ncbi.nlm.nih.gov/articles/PMC8905023/)

---

### Pitfall 3: Colab Enterprise Data Loss from Ephemeral Sessions

**What goes wrong:**

Colab Enterprise runtimes are ephemeral:
- Session timeout: 90 minutes idle, 12 hours maximum
- Kernel crash: All in-memory data lost
- Network interruption: Disconnects runtime
- Manual "Factory reset runtime": Clears all local files

**Data loss scenarios:**

1. **Feature engineering in progress:**
   ```
   Hour 1: Extract poses from 500 videos (1 hour runtime)
   Hour 2: Engineer features from poses (30 minutes)
   Hour 2.5: Network hiccup → Runtime disconnects
   Result: 2.5 hours of computation lost, no checkpoint saved
   ```

2. **Model training without checkpointing:**
   ```
   Epoch 1-20: Training progressing (loss decreasing)
   Epoch 21: Colab timeout (12 hour limit hit)
   Result: No model saved, must restart from epoch 0
   ```

3. **Git push failure:**
   ```
   # User runs feature engineering in Colab
   # Forgets to git add + commit + push before closing session
   # Next day: Opens new runtime, previous work gone
   ```

**Why it happens:**

1. **Jupyter notebook workflow assumption** - Colab users expect interactive notebook persistence, but you're using terminal scripts which don't auto-save state
2. **No git pull in Colab UI** - Colab's "Save a copy in GitHub" only works for notebooks, not for script outputs
3. **Large file commit failures** - Attempting to commit 105 MB file (like previous train_data.pkl) blocks push, user assumes it succeeded
4. **Authentication expiration** - GitHub PAT expires, push fails silently in background script

**Consequences:**

- Lose days of feature engineering work
- Model checkpoints not versioned (can't reproduce "best" model from 3 days ago)
- Team members can't access latest features (sync failure)
- Experiment results not tracked (which hyperparameters produced 65% accuracy?)
- Force to re-run expensive pose extraction (MediaPipe on 3,347 videos)

**Prevention:**

1. **Checkpoint to Google Cloud Storage (GCS) every N iterations:**
   ```python
   from google.cloud import storage

   def checkpoint_to_gcs(data, bucket_name, blob_name):
       """Save checkpoint to GCS (native Colab Enterprise integration)"""
       client = storage.Client()
       bucket = client.bucket(bucket_name)
       blob = bucket.blob(blob_name)
       blob.upload_from_string(pickle.dumps(data))
       print(f"Checkpointed to gs://{bucket_name}/{blob_name}")

   # In feature engineering loop:
   for i, video in enumerate(videos):
       features = extract_features(video)
       if i % 100 == 0:  # Checkpoint every 100 videos
           checkpoint_to_gcs(features, 'badminton-ml', f'checkpoints/features_{i}.pkl')
   ```

2. **Git commit + push in script (not manual):**
   ```bash
   #!/bin/bash
   # colab_workflow.sh - Run in Colab terminal

   # Pull latest
   git pull origin main

   # Run feature engineering
   python scripts/engineer_features.py --checkpoint-interval 100

   # Auto-commit outputs (excluding large files)
   git add data/processed/features/*.csv  # Only CSVs, not PKLs
   git add outputs/feature_importance.json
   git commit -m "Feature engineering run $(date +%Y%m%d_%H%M%S)"

   # Push with error handling
   if git push origin main; then
       echo "Successfully pushed to GitHub"
   else
       echo "Push failed - check .gitignore for large files"
       git status
       exit 1
   fi
   ```

3. **Separate data from code in git:**
   ```
   # Code repo (git, no LFS):
   - scripts/
   - src/
   - .gitignore (exclude all data)

   # Data storage (GCS):
   gs://badminton-ml-data/
   ├── raw_videos/           # ShuttleSet clips
   ├── processed/
   │   ├── poses/           # MediaPipe outputs
   │   └── features/        # Engineered features
   └── checkpoints/         # Training checkpoints

   # Colab workflow:
   1. Clone code repo (fast, <10 MB)
   2. Download data from GCS (controlled, only what's needed)
   3. Run scripts
   4. Upload outputs to GCS
   5. Push code changes to git
   ```

4. **Model versioning with MLflow/W&B (not git):**
   ```python
   import mlflow

   # Track experiments without bloating git repo
   mlflow.set_tracking_uri("gs://badminton-ml/mlruns")  # GCS backend

   with mlflow.start_run():
       mlflow.log_params({"n_features": 427, "model": "RandomForest"})
       mlflow.log_metrics({"test_accuracy": 0.65, "f1_score": 0.63})
       mlflow.sklearn.log_model(model, "model")  # Versioned automatically
   ```

5. **Colab-specific safeguards:**
   ```python
   # Add to start of every Colab script:

   import signal
   import sys

   def emergency_checkpoint(signum, frame):
       """Save work before timeout/crash"""
       print("Emergency checkpoint triggered!")
       checkpoint_to_gcs(current_state, 'badminton-ml', 'emergency_checkpoint.pkl')
       sys.exit(0)

   signal.signal(signal.SIGTERM, emergency_checkpoint)  # Timeout signal
   signal.signal(signal.SIGINT, emergency_checkpoint)   # Ctrl+C
   ```

6. **Validate pushes in script:**
   ```bash
   # After git push, verify it worked:
   PUSHED_COMMIT=$(git rev-parse HEAD)
   REMOTE_COMMIT=$(git ls-remote origin main | cut -f1)

   if [ "$PUSHED_COMMIT" != "$REMOTE_COMMIT" ]; then
       echo "ERROR: Push failed or not yet propagated"
       echo "Local commit: $PUSHED_COMMIT"
       echo "Remote commit: $REMOTE_COMMIT"
       exit 1
   fi
   ```

**Detection (warning signs):**

- Colab disconnection warnings ("You will lose all data...")
- Git push succeeds locally but changes not visible on GitHub
- Terminal scripts running >11 hours (approaching timeout)
- Large files in `git status` before push
- GitHub authentication prompts during automated push
- Missing experiment results from "yesterday's run"

**Phase recommendation:**

Phase 1 (Infrastructure Setup) MUST include:
- GCS bucket creation and access testing
- Checkpoint function implemented and tested
- Emergency checkpoint handler tested (SIGTERM simulation)
- Git workflow script tested end-to-end in Colab
- MLflow/W&B experiment tracking configured

Do not proceed to Phase 2 until data loss safeguards are proven to work.

**Sources:**
- [Troubleshooting Google Colab crashes and memory issues](https://www.mindfulchase.com/explore/troubleshooting-tips/data-science/troubleshooting-google-colab-crashes-and-memory-issues-in-data-science-workflows.html)
- [Colab-GitHub workflow best practices](https://tilburgsciencehub.com/topics/automation/replicability/cloud-computing/colab-github/)

---

## High Priority Pitfalls

Mistakes that cause delays, rework, or significant technical debt.

---

### Pitfall 4: Git LFS .gitattributes Misconfiguration

**What goes wrong:**

LFS tracking is managed by `.gitattributes` file. Common mistakes:

1. **Tracking files already in git history:**
   ```bash
   git lfs track "*.pkl"  # Adds to .gitattributes
   git add data/train_data.pkl
   git commit -m "Track with LFS"

   # But train_data.pkl already exists in history (committed without LFS)
   # Result: File exists TWICE - once in git, once in LFS
   # GitHub rejects push: "File exceeds 100 MB"
   ```

2. **Untrack command doesn't work:**
   - `git lfs untrack "*.mp4"` removes from `.gitattributes`
   - But doesn't remove files already tracked in LFS
   - Files still appear in `git lfs ls-files`
   - Manual `.gitattributes` editing more reliable than CLI

3. **Wildcard overmatch:**
   ```bash
   # WRONG: Tracks ALL mp4 files everywhere
   git lfs track "*.mp4"

   # Accidentally tracks:
   # - outputs/demo_video.mp4 (should be in git, only 2 MB)
   # - tests/fixtures/sample.mp4 (needed for CI, only 500 KB)

   # RIGHT: Specific paths only
   git lfs track "data/raw_videos/*.mp4"
   ```

4. **Forgetting to commit .gitattributes:**
   ```bash
   git lfs track "*.mp4"
   git add video.mp4
   git commit -m "Add video"
   # Did NOT commit .gitattributes
   # Other team members don't have LFS tracking configured
   # Their clones fail or get pointer files
   ```

**Prevention:**

1. **Use `git lfs migrate` for existing files:**
   ```bash
   # For files already in git history:
   git lfs migrate import --include="*.pkl" --everything
   # Rewrites history to track with LFS
   # Requires force push: git push --force
   ```

2. **Always commit .gitattributes changes:**
   ```bash
   git lfs track "data/validation_clips/*.mp4"
   git add .gitattributes  # CRITICAL
   git commit -m "Configure LFS tracking for validation videos"
   ```

3. **Verify LFS tracking before adding files:**
   ```bash
   git lfs track  # List all tracked patterns
   git lfs ls-files  # List files currently in LFS

   # Check if new file will be tracked:
   git check-attr filter data/new_video.mp4
   # Should output: data/new_video.mp4: filter: lfs
   ```

4. **Explicit .gitattributes patterns:**
   ```bash
   # Good: Specific paths
   data/validation_clips/*.mp4 filter=lfs diff=lfs merge=lfs -text
   models/final_model.h5 filter=lfs diff=lfs merge=lfs -text

   # Bad: Global wildcards
   *.mp4 filter=lfs diff=lfs merge=lfs -text
   ```

**Detection:**

- `git push` rejected with "file exceeds 100 MB"
- `git lfs ls-files` shows unexpected files
- Team members report "pointer files" (text files with `oid sha256:...`)
- LFS bandwidth consumed but files not downloading

**Phase:** Phase 1 (Infrastructure Setup) - Configure before adding any videos

**Sources:**
- [Git LFS untrack doesn't work (GitHub issue)](https://github.com/git-lfs/git-lfs/issues/4058)
- [Git LFS track documentation](https://www.mankier.com/1/git-lfs-track)

---

### Pitfall 5: Feature Leakage from Improper Train/Test Split

**What goes wrong:**

Badminton dataset has temporal and player-based dependencies that naive random split violates:

1. **Player leakage:**
   ```
   Player A: 100 Clear strokes, 100 Smash strokes
   Random split: 80 Clear in train, 20 Clear in test
                 80 Smash in train, 20 Smash in test

   Problem: Test set contains Player A's strokes
   Model learns Player A's idiosyncrasies (height, reach, style)
   Doesn't generalize to Player B (new player)
   ```

2. **Temporal leakage in match data:**
   ```
   Match 1: Stroke 1 → Stroke 2 → Stroke 3 → Stroke 4
   Random split: Stroke 1, 4 in train | Stroke 2, 3 in test

   Problem: Consecutive strokes are correlated
   Player fatigues over match (angles change)
   Model sees "future" data in training
   ```

3. **Global normalization leakage:**
   ```python
   # WRONG: Leakage from test set into training
   all_features = load_all_features()  # Train + validation + test
   scaler = StandardScaler()
   scaler.fit(all_features)  # Learns mean/std from test set!

   train_scaled = scaler.transform(train_features)
   test_scaled = scaler.transform(test_features)
   ```

4. **Feature engineering leakage:**
   ```python
   # WRONG: Target leakage
   features['avg_wrist_velocity_next_stroke'] = ...  # Uses future data

   # WRONG: Group statistics leakage
   features['player_avg_elbow_angle'] = ...  # Includes test set strokes
   ```

**Specific to badminton biomechanics:**

- **Fatigue effects:** Player's technique degrades over match (ankle joint parameters change, stability declines)
- **Warm-up effects:** First 5 strokes of match are different from strokes 50-100
- **Player anthropometrics:** Tall players have different joint angles than short players (but both can have good technique)
- **Court position correlation:** Forehand clear from backcourt vs midcourt have different characteristics

**Prevention:**

1. **Group-based splitting (by player):**
   ```python
   from sklearn.model_selection import GroupShuffleSplit

   # Ensure all strokes from same player stay in same split
   splitter = GroupShuffleSplit(n_splits=1, test_size=0.15, random_state=42)
   train_idx, test_idx = next(splitter.split(X, y, groups=player_ids))
   ```

2. **Temporal splitting (if match data):**
   ```python
   # Train on early matches, test on late matches
   # Simulates deployment: Train on past, predict future

   matches_sorted = matches.sort_values('match_date')
   train_matches = matches_sorted[:int(0.8 * len(matches_sorted))]
   test_matches = matches_sorted[int(0.8 * len(matches_sorted)):]
   ```

3. **Fit preprocessing on training only:**
   ```python
   # CORRECT: No leakage
   scaler = StandardScaler()
   scaler.fit(X_train)  # Learn from training only

   X_train_scaled = scaler.transform(X_train)
   X_test_scaled = scaler.transform(X_test)  # Apply same transformation
   ```

4. **Stratified splits for class balance + groups:**
   ```python
   from sklearn.model_selection import StratifiedGroupKFold

   # Maintains Clear/Smash balance AND player grouping
   cv = StratifiedGroupKFold(n_splits=5)
   for train_idx, val_idx in cv.split(X, y, groups=player_ids):
       # Each fold has balanced classes and no player overlap
       pass
   ```

5. **Feature engineering on train set only:**
   ```python
   # Calculate feature thresholds on training set
   velocity_threshold = np.percentile(X_train['velocity'], 90)

   # Apply same threshold to test set
   X_train['high_velocity'] = X_train['velocity'] > velocity_threshold
   X_test['high_velocity'] = X_test['velocity'] > velocity_threshold
   ```

**Detection:**

- Unrealistic test accuracy (>95% on small dataset)
- Performance drops dramatically on external videos (out-of-distribution)
- Feature importance shows player-specific features ranked high
- Model performs well on player in training set, fails on new player
- Scaler fitted on full dataset before split

**Phase:** Phase 2 (Feature Engineering) and Phase 3 (Model Training) - Validate split strategy before training

**Sources:**
- [Preventing training data leakage](https://www.tonic.ai/blog/prevent-training-data-leakage-ai)
- [Data leakage in cross-validation](https://medium.com/@silva.f.francis/avoiding-data-leakage-in-cross-validation-ba344d4d55c0)
- [Temporal leakage in time series](https://codecut.ai/cross-validation-with-time-series/)

---

### Pitfall 6: Model-Benchmark Integration Fragility

**What goes wrong:**

Current system uses benchmark-based analysis (no ML in production). Adding ML classification creates integration complexity:

1. **Dual analysis paths:**
   ```
   Video → Pose → Features → ML Classification → Label
                          └→ Benchmark Analysis → Feedback

   Problem: Which result to trust when ML says "Clear" but benchmark says "This looks like a Smash"?
   ```

2. **Feature compatibility:**
   ```
   ML model trained on: 600 features (expanded set)
   Benchmark system uses: 427 features (original set)

   Problem: Feature engineering diverges
   Pipeline breaks when benchmark expects features ML doesn't produce
   ```

3. **Version mismatch:**
   ```
   Benchmarks derived from: Forehand-only dataset (3,347 strokes, Jan 16)
   ML model trained on: Expanded dataset (4,983 strokes, includes backhand)

   Problem: Benchmark ranges don't apply to backhand strokes
   ML detects stroke type, but benchmark analysis is wrong
   ```

4. **Threshold drift:**
   ```python
   # Benchmark system: severity = 'critical' if >2 std dev
   # ML system: confidence threshold = 0.7 for "reliable" classification

   # What happens when ML says "80% confident it's a Clear"?
   # Do we:
   # - Run benchmark analysis for Clear (might be wrong)
   # - Run benchmark for both Clear and Smash (confusing output)
   # - Refuse to analyze (bad UX)
   ```

**Consequences:**

- Users get conflicting feedback (ML says Clear, benchmark analysis uses Smash ranges)
- Pipeline breaks when adding new features (benchmark code expects old feature names)
- Can't A/B test ML vs benchmark (no unified interface)
- Feature engineering codebase forks into "ML version" and "benchmark version"

**Prevention:**

1. **Unified feature interface:**
   ```python
   # Define feature schema BEFORE expanding features

   from dataclasses import dataclass
   from typing import Dict, List

   @dataclass
   class FeatureSet:
       """Versioned feature set"""
       version: str  # "v1.0" (427 features) or "v2.0" (600 features)
       core_features: Dict[str, float]  # Required by both ML and benchmarks
       extended_features: Dict[str, float]  # ML-only features

   def extract_features(poses, version="v1.0") -> FeatureSet:
       """Extract features compatible with specified version"""
       core = extract_core_features(poses)  # Always 427 features

       if version == "v2.0":
           extended = extract_extended_features(poses)
       else:
           extended = {}

       return FeatureSet(version=version, core_features=core, extended_features=extended)
   ```

2. **Benchmark validation before deployment:**
   ```python
   # Before using ML classification results:

   ml_label = ml_model.predict(features)
   ml_confidence = ml_model.predict_proba(features).max()

   if ml_confidence < 0.8:
       # Low confidence: Use auto-detection fallback or user input
       label = auto_detect_stroke_type(features)  # Existing heuristic
   else:
       label = ml_label

   # Run benchmark analysis with validated label
   feedback = benchmark_analysis(features.core_features, stroke_type=label)
   ```

3. **Separate forehand/backhand benchmarks:**
   ```python
   # Detect stroke side BEFORE benchmark analysis

   def detect_stroke_side(poses) -> str:
       """Detect forehand vs backhand from active arm"""
       # Current system: Uses arm elevation heuristic
       # Enhancement: Compare left vs right wrist velocity
       left_wrist_vel = calculate_velocity(poses[:, left_wrist_indices])
       right_wrist_vel = calculate_velocity(poses[:, right_wrist_indices])

       if left_wrist_vel.max() > right_wrist_vel.max():
           return "left"  # Left-handed or backhand
       else:
           return "right"  # Right-handed or forehand

   # Load appropriate benchmarks
   stroke_side = detect_stroke_side(poses)
   stroke_type = ml_model.predict(features)

   benchmarks = get_benchmarks(stroke_type, stroke_side)  # forehand or backhand
   ```

4. **Feature engineering version control:**
   ```python
   # Track which features were used in each model

   # models/model_v1.json
   {
       "model_file": "random_forest_v1.pkl",
       "features": ["max_velocity", "elbow_angle_mean", ...],  # 427 features
       "feature_version": "v1.0",
       "trained_date": "2026-01-16",
       "test_accuracy": 0.65
   }

   # models/model_v2.json
   {
       "model_file": "random_forest_v2.pkl",
       "features": ["max_velocity", ..., "wrist_pronation_range"],  # 600 features
       "feature_version": "v2.0",
       "trained_date": "2026-02-01",
       "test_accuracy": 0.68
   }

   # Load model + compatible features:
   model_meta = json.load("models/model_v2.json")
   features = extract_features(poses, version=model_meta["feature_version"])
   ```

5. **Integration testing:**
   ```python
   # tests/test_ml_benchmark_integration.py

   def test_ml_classification_feeds_benchmark_correctly():
       """Ensure ML classification result works with benchmark analysis"""
       poses = load_test_video_poses("clear_example.mp4")
       features = extract_features(poses, version="v2.0")

       # ML classification
       stroke_type = ml_model.predict(features.all_features)
       assert stroke_type in ["Clear", "Smash"]

       # Benchmark analysis should accept this label
       feedback = benchmark_analysis(features.core_features, stroke_type=stroke_type)
       assert len(feedback) > 0  # Should generate feedback
       assert feedback[0].metric in EXPECTED_METRICS
   ```

**Detection:**

- Error: "KeyError: 'wrist_pronation_range'" when benchmark code runs
- Benchmark analysis produces nonsensical feedback (e.g., "Your Smash is too slow" for a Clear)
- ML confidence low on most predictions (model-data mismatch)
- Feature extraction time doubles (computing unused features)

**Phase:** Phase 3 (Model Training) - Define integration contract BEFORE training models

---

### Pitfall 7: Terminal Script Reliability in Colab Enterprise

**What goes wrong:**

Project uses terminal scripts (not Jupyter notebooks) in Colab Enterprise. Specific failure modes:

1. **Paths break between notebook and terminal:**
   ```python
   # In notebook: Current directory is /content
   # In terminal: Current directory is /root

   # Script assumes:
   video_path = "data/raw_videos/video.mp4"  # Relative path

   # Fails with: FileNotFoundError
   ```

2. **Environment variables not propagated:**
   ```bash
   # Set in notebook:
   %env OPENAI_API_KEY=sk-...

   # Terminal script:
   api_key = os.getenv("OPENAI_API_KEY")  # Returns None
   ```

3. **Dependency version conflicts:**
   ```bash
   # Colab pre-installs:
   # - TensorFlow 2.18.0
   # - protobuf 4.x

   # Your requirements.txt:
   # - TensorFlow 2.15.x (for MediaPipe compatibility)
   # - protobuf 3.20.3

   # pip install -r requirements.txt
   # TensorFlow downgrade breaks Colab pre-installed packages
   ```

4. **Script output not captured:**
   ```python
   # Terminal script prints:
   print("Processing video 500/3347")

   # In Colab terminal: Output visible
   # In automated run: Output lost (not saved anywhere)
   ```

**Prevention:**

1. **Use absolute paths:**
   ```python
   # At start of every script:
   import os
   from pathlib import Path

   SCRIPT_DIR = Path(__file__).parent.absolute()
   PROJECT_ROOT = SCRIPT_DIR.parent
   DATA_DIR = PROJECT_ROOT / "data"

   # All file operations:
   video_path = DATA_DIR / "raw_videos" / "video.mp4"
   ```

2. **Environment setup script:**
   ```bash
   # setup_colab_env.sh - Run FIRST in every Colab session

   #!/bin/bash
   set -e  # Exit on error

   # Set environment variables for terminal
   export PROJECT_ROOT="/content/iti123_v2"
   export PYTHONPATH="$PROJECT_ROOT/src:$PYTHONPATH"
   export OPENAI_API_KEY="sk-..."  # Or load from GCS Secret Manager

   # Install dependencies in order
   pip uninstall -y protobuf tensorflow
   pip install protobuf==3.20.3
   pip install tensorflow==2.15.1
   pip install -r requirements.txt

   # Verify critical imports
   python -c "import mediapipe; print(f'MediaPipe {mediapipe.__version__}')"
   python -c "import google.protobuf; print(f'Protobuf {google.protobuf.__version__}')"

   echo "Environment setup complete"
   ```

3. **Redirect script output:**
   ```python
   import sys
   from datetime import datetime

   # At start of script:
   log_file = f"outputs/logs/run_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
   os.makedirs(os.path.dirname(log_file), exist_ok=True)

   # Duplicate output to file and console
   class Tee:
       def __init__(self, *files):
           self.files = files
       def write(self, data):
           for f in self.files:
               f.write(data)
       def flush(self):
           for f in self.files:
               f.flush()

   log_f = open(log_file, 'w')
   sys.stdout = Tee(sys.stdout, log_f)
   sys.stderr = Tee(sys.stderr, log_f)
   ```

4. **Script health checks:**
   ```python
   # scripts/preflight_check.py

   def preflight_check():
       """Verify environment before running expensive operations"""
       checks = {
           "MediaPipe version": check_mediapipe_version(),
           "Protobuf version": check_protobuf_version(),
           "Data directory exists": check_data_directory(),
           "GCS credentials": check_gcs_access(),
           "Git credentials": check_git_auth(),
           "Disk space": check_disk_space(required_gb=10),
       }

       failed = [name for name, passed in checks.items() if not passed]

       if failed:
           print(f"Preflight FAILED: {', '.join(failed)}")
           sys.exit(1)
       else:
           print("Preflight passed ✓")

   if __name__ == "__main__":
       preflight_check()
   ```

**Detection:**

- Script works in local dev, fails in Colab
- "Module not found" errors despite `pip install`
- Silent failures (script exits without error message)
- Different results between notebook and terminal execution

**Phase:** Phase 1 (Infrastructure Setup) - Test terminal scripts in Colab before building pipeline

**Sources:**
- [Colab-GitHub workflow](https://tilburgsciencehub.com/topics/automation/replicability/cloud-computing/colab-github/)
- Existing CONCERNS.md (Protobuf version dependency issue)

---

## Medium Priority Pitfalls

Mistakes that cause moderate delays or technical debt.

---

### Pitfall 8: Insufficient Badminton Domain Knowledge in Feature Engineering

**What goes wrong:**

Adding "coach-informed" features without understanding badminton biomechanics leads to non-discriminative or unmeasurable features.

**Common mistakes:**

1. **Copying tennis/baseball features:**
   - Racket head speed: Different meaning in badminton (lighter racket, wrist-driven)
   - Hip rotation: Less important in overhead badminton strokes vs baseball swing
   - Weight transfer: Important but measured differently (forward lunge vs lateral)

2. **Teaching cues vs biomechanical reality:**
   - Coach says: "Keep elbow high" → Feature: `elbow_height_above_shoulder`
   - Reality: Elite players vary elbow height; what matters is **elbow-wrist-racket alignment** at contact
   - Teaching cue for beginners ≠ discriminative feature for technique quality

3. **Unmeasurable from video:**
   - Grip pressure (requires sensors)
   - Shuttle contact sound (requires audio analysis)
   - Player intent (defensive clear vs attacking clear - look identical until shuttle trajectory)

4. **Ignoring kinematic chain sequencing:**
   - Badminton smash uses **proximal-to-distal kinetic chain**: shoulder → elbow → wrist sequence
   - Adding static joint angles misses the **temporal coordination** (timing between segments)
   - Feature: `elbow_angle_at_contact` vs `time_delta_shoulder_to_wrist_peak_velocity`

**Specific badminton biomechanics from research:**

| Feature Category | Discriminative for Clear vs Smash | Measurable from MediaPipe |
|------------------|-----------------------------------|---------------------------|
| Shuttle velocity | ✓ (60% increase in smash) | ✗ (requires shuttle tracking) |
| Wrist angular velocity | ✓ (higher in smash) | ✓ (from wrist position derivative) |
| Racket angle at contact | ✓ (different trajectory) | ~ (approximate from wrist-elbow vector) |
| Proximal-distal timing | ✓ (kinetic chain efficiency) | ✓ (from joint velocity peaks) |
| Shoulder flexibility | ✗ (affects quality, not type) | ~ (from range of motion) |
| Elbow height | ✗ (varies by player preference) | ✓ (but not discriminative) |

**Prevention:**

1. **Literature review BEFORE feature engineering:**
   - Read badminton biomechanics papers (not generic sports science)
   - Identify features that distinguish elite from recreational players
   - Identify features that distinguish Clear from Smash
   - Note measurement methods (optical markers, IMU sensors, video)

2. **Validate with domain expert (async):**
   ```markdown
   # Feature proposal document:

   Feature: wrist_pronation_range
   Hypothesis: Smash requires more wrist pronation than Clear
   Source: [Kinematic Analysis of Wrist and Elbow Angles (2025)](https://areste.org/index.php/oai/article/download/66/82)
   Measurement: Range of wrist z-rotation from MediaPipe (if available)
   Expected effect size: Cohen's d > 0.6 (medium)

   Request for expert feedback:
   - Is this hypothesis correct for advanced players?
   - Are there player-specific variations (e.g., by grip type)?
   ```

3. **Start with published discriminative features:**
   ```python
   # From literature review:
   # - Clear/Smash ratio = 0.75 for shuttle speed
   # - 60.2% increase in smash velocity
   # - Racket angle at contact differs by ~15 degrees

   # Implement these FIRST before adding new features:
   features['wrist_velocity_max'] = ...  # Proxy for racket speed
   features['wrist_height_at_max_velocity'] = ...  # Proxy for contact point
   features['elbow_extension_rate'] = ...  # Proxy for kinetic chain
   ```

4. **Ablation testing:**
   ```python
   # For each new feature:
   # 1. Train model WITHOUT feature → Baseline accuracy
   # 2. Train model WITH feature → New accuracy
   # 3. If improvement < 1%, feature is not useful

   baseline_features = ['max_velocity', 'elbow_angle_mean', ...]  # 427 features
   baseline_accuracy = cross_val_score(model, X[baseline_features], y).mean()

   new_features = baseline_features + ['wrist_pronation_range']
   new_accuracy = cross_val_score(model, X[new_features], y).mean()

   if new_accuracy - baseline_accuracy < 0.01:
       print(f"Feature 'wrist_pronation_range' not useful (Δ = {new_accuracy - baseline_accuracy:.3f})")
   ```

**Detection:**

- Feature importance analysis shows new features ranked low
- Feature correlation matrix shows new feature r > 0.95 with existing feature
- Literature review finds no papers mentioning the feature
- Expert says "We don't teach that" or "That's not how badminton works"

**Phase:** Phase 2 (Feature Engineering) - Literature review first, implementation second

**Sources:**
- [Kinematic Analysis of Wrist and Elbow Angles in Badminton (2025)](https://www.areste.org/index.php/oai/article/download/66/82/481)
- [Biomechanical Insights for Smash Development (2023)](https://www.mdpi.com/2076-3417/13/22/12488)
- [Badminton Clear vs Smash biomechanical differences](https://www.kheljournal.com/archives/2025/vol12issue3/PartC/12-3-23-144.pdf)

---

### Pitfall 9: No Experiment Tracking (MLOps Discipline)

**What goes wrong:**

Without experiment tracking, you lose critical information:

```
Week 1: Trained model, got 68% accuracy, forgot which hyperparameters
Week 2: Tried different features, got 65% accuracy, can't remember which features
Week 3: Need to reproduce Week 1 results for report - impossible
```

**What gets lost:**

- Which features were used in "best" model?
- Which hyperparameters produced 68% accuracy?
- Which data split was used? (Can't reproduce if random seed not logged)
- How many training samples had missing keypoints?
- What was the class distribution in that run?

**Specific to this project:**

- Benchmark ranges updated Jan 16 (forehand-only) - which models trained before vs after?
- Feature engineering v1 (427 features) vs v2 (600 features) - which results use which?
- MediaPipe version differences (0.10.9 vs 0.10.14) affect keypoint positions
- Protobuf version affects MediaPipe output (known issue from CONCERNS.md)

**Prevention:**

1. **MLflow for local experiment tracking:**
   ```python
   import mlflow

   # Configure to use GCS backend (accessible from Colab)
   mlflow.set_tracking_uri("gs://badminton-ml/mlruns")

   with mlflow.start_run(run_name="feature_v2_random_forest"):
       # Log parameters
       mlflow.log_params({
           "n_features": len(features.columns),
           "feature_version": "v2.0",
           "model_type": "RandomForest",
           "n_estimators": 100,
           "max_depth": 10,
           "train_samples": len(X_train),
           "test_samples": len(X_test),
       })

       # Log metrics
       mlflow.log_metrics({
           "train_accuracy": train_acc,
           "test_accuracy": test_acc,
           "f1_score": f1,
           "roc_auc": auc,
       })

       # Log artifacts
       mlflow.log_artifact("outputs/feature_importance.png")
       mlflow.log_artifact("outputs/confusion_matrix.png")

       # Log model
       mlflow.sklearn.log_model(model, "model")

       # Log dataset metadata
       mlflow.log_dict({
           "features": list(features.columns),
           "class_distribution": dict(pd.Series(y_train).value_counts()),
           "split_strategy": "stratified_group_shuffle",
           "random_seed": 42,
       }, "dataset_metadata.json")
   ```

2. **Weights & Biases for richer tracking:**
   ```python
   import wandb

   wandb.init(
       project="badminton-ml",
       config={
           "features": len(features.columns),
           "model": "RandomForest",
           "dataset": "ShuttleSet_forehand_only",
       }
   )

   # Automatic logging of system metrics (GPU, CPU, memory)
   # Table logging for predictions
   wandb.log({
       "predictions": wandb.Table(dataframe=predictions_df),
       "train_accuracy": train_acc,
   })

   # Artifact versioning
   artifact = wandb.Artifact('model-rf-v2', type='model')
   artifact.add_file('model.pkl')
   wandb.log_artifact(artifact)
   ```

3. **Minimal CSV logging (if no internet):**
   ```python
   # outputs/experiment_log.csv
   import csv
   from datetime import datetime

   log_entry = {
       "timestamp": datetime.now().isoformat(),
       "experiment_id": "exp_20260129_001",
       "features_version": "v2.0",
       "n_features": 600,
       "model_type": "RandomForest",
       "train_acc": 0.85,
       "test_acc": 0.68,
       "notes": "Added wrist pronation features",
   }

   with open("outputs/experiment_log.csv", "a") as f:
       writer = csv.DictWriter(f, fieldnames=log_entry.keys())
       if f.tell() == 0:  # New file
           writer.writeheader()
       writer.writerow(log_entry)
   ```

4. **Git tags for model versions:**
   ```bash
   # After training best model:
   git tag -a v1.1-model-rf-68pct -m "Random Forest 68% test accuracy, 600 features"
   git push origin v1.1-model-rf-68pct

   # Reproduce later:
   git checkout v1.1-model-rf-68pct
   python scripts/train_model.py  # Should produce same results (with same random seed)
   ```

**Detection:**

- Can't reproduce "best" model from last week
- Multiple model files with no metadata (model_v1.pkl, model_v2.pkl, model_final.pkl)
- Unsure which features were used in report results
- Forgot which hyperparameters to use for production

**Phase:** Phase 3 (Model Training) - Set up experiment tracking BEFORE first training run

**Sources:**
- [MLflow documentation](https://mlflow.org/docs/latest/genai/version-tracking/)
- [Weights & Biases experiment tracking](https://www.zenml.io/blog/mlflow-vs-weights-and-biases)

---

## Minor Pitfalls

Mistakes that cause annoyance but are easily fixable.

---

### Pitfall 10: Hardcoded Paths Break Across Environments

**What goes wrong:**

```python
# Script written on macOS:
video_path = "/Users/username/iti123_v2/data/videos/test.mp4"

# Runs in Colab:
FileNotFoundError: /Users/username/iti123_v2/data/videos/test.mp4
```

**Prevention:**

```python
# Use Path and relative paths:
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / "data"
video_path = DATA_DIR / "videos" / "test.mp4"
```

---

### Pitfall 11: Forgetting to Save Feature Names with Model

**What goes wrong:**

```python
# Training:
model.fit(X_train, y_train)
joblib.dump(model, "model.pkl")

# Inference (1 month later):
features = extract_features(video)  # 650 features (added 50 more)
predictions = model.predict(features)  # ERROR: expects 600 features
```

**Prevention:**

```python
# Save feature names with model:
model_bundle = {
    "model": model,
    "feature_names": list(X_train.columns),
    "feature_version": "v2.0",
    "scaler": scaler,
}
joblib.dump(model_bundle, "model_bundle.pkl")

# Inference:
bundle = joblib.load("model_bundle.pkl")
features_subset = features[bundle["feature_names"]]  # Only use trained features
predictions = bundle["model"].predict(features_subset)
```

---

### Pitfall 12: Not Handling MediaPipe Non-Determinism

**What goes wrong:**

Same video analyzed twice produces slightly different feedback scores.

**Prevention:**

- Document non-determinism in user-facing output
- Use percentile ranges instead of exact scores (65-70% instead of 67.3%)
- Average multiple runs for critical decisions

---

## Phase-Specific Warnings

Pitfalls organized by which milestone phase they affect most.

| Phase | Critical Pitfalls | Prevention |
|-------|------------------|------------|
| **Phase 1: Infrastructure Setup** | Git LFS bandwidth (P1), Colab data loss (P3), .gitattributes config (P4) | Set up GCS, test LFS with small files first, validate checkpointing works |
| **Phase 2: Feature Engineering** | Feature explosion (P2), Feature leakage (P5), Domain knowledge gaps (P8) | Literature review first, feature selection analysis, ablation testing |
| **Phase 3: Model Training** | Overfitting on small dataset (P2), No experiment tracking (P9), Model-benchmark integration (P6) | Regularization, cross-validation, MLflow setup, integration tests |
| **Phase 4: Validation** | Terminal script reliability (P7), Feature leakage in test set (P5) | Preflight checks, validate split strategy, external video testing |

---

## Quick Reference: Validation Checklist

Before proceeding to next phase:

### Phase 1 Checklist:
- [ ] GCS bucket created and accessible from Colab
- [ ] Git LFS configured for <50 validation videos only
- [ ] Shallow clone tested in Colab (< 100 MB download)
- [ ] Checkpoint-to-GCS function tested
- [ ] Emergency checkpoint handler tested (SIGTERM)
- [ ] Git push workflow tested end-to-end
- [ ] Terminal scripts work in Colab (absolute paths)
- [ ] Bandwidth usage monitored (< 500 MB first week)

### Phase 2 Checklist:
- [ ] Literature review completed (3+ badminton biomechanics papers)
- [ ] Discriminative features identified from research
- [ ] Feature selection analysis run on current 427 features
- [ ] Train/test split validated (no player leakage)
- [ ] Normalization fitted on training set only
- [ ] Feature count < N_train / 10 (< 254 for 2,554 samples)
- [ ] Ablation test: Each new feature improves accuracy by >1%
- [ ] Correlation matrix checked (no r > 0.95 redundancy)

### Phase 3 Checklist:
- [ ] MLflow or W&B experiment tracking configured
- [ ] Cross-validation strategy defined (stratified + grouped)
- [ ] Regularization enabled (L1/L2 or dropout)
- [ ] Early stopping enabled (validation loss, patience=10)
- [ ] Model-benchmark integration tested
- [ ] Feature version tracked with model
- [ ] External validation videos tested (not from ShuttleSet)
- [ ] Training/test accuracy gap < 15%

---

## Summary

### Critical (Must Fix Before Proceeding):
1. **Git LFS bandwidth trap** - Use GCS for videos, LFS for tiny validation set only
2. **Feature explosion overfitting** - Feature selection BEFORE expansion, keep count < N/10
3. **Colab data loss** - Checkpoint to GCS every 100 iterations, emergency handlers

### High Priority (Address in Each Phase):
4. **Git LFS .gitattributes** - Specific paths, commit .gitattributes, use git lfs migrate
5. **Feature leakage** - Group-based splits, fit scaler on train only, no global statistics
6. **Model-benchmark integration** - Unified feature interface, version compatibility
7. **Terminal script reliability** - Absolute paths, environment setup, preflight checks

### Medium Priority (Best Practices):
8. **Domain knowledge gaps** - Literature review, validate with badminton biomechanics research
9. **No experiment tracking** - MLflow or W&B, log everything, reproducibility

### Interconnected Risks:

```
Git LFS bandwidth limits
    ↓
Discourage frequent checkpointing
    ↓
Increase Colab data loss risk
    ↓
Re-run feature engineering
    ↓
Consume more LFS bandwidth (vicious cycle)
```

**Break the cycle:** Use GCS for data, git for code only.

---

## Risk Assessment

**Overall risk level for v1.1 milestone:** HIGH

- Three critical pitfalls that can block progress (LFS, overfitting, data loss)
- Integration complexity between existing system and new ML components
- Small dataset (3,347 samples) leaves little room for error in feature engineering
- Badminton-specific domain knowledge required (not generic ML)

**Recommended priority:**

1. Phase 1: Solve infrastructure issues (LFS, GCS, checkpointing) - **1 week**
2. Phase 2: Feature selection + domain research - **1 week**
3. Phase 3: Careful model training with regularization - **1 week**
4. Phase 4: Validation on external videos - **3 days**

**If timeline is tight:** Use benchmark-based analysis only (existing v1.0 approach), defer ML to v1.2. Expanding feature set and retraining has high risk/reward ratio.

---

*Research completed: 2026-01-29*
*Confidence: HIGH (verified with official documentation and recent badminton biomechanics research)*
*Primary sources: GitHub LFS docs, badminton biomechanics papers (2024-2025), ML overfitting research (2024)*
