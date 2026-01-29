# Technology Stack - v1.1 Additions

**Project:** AI Badminton Coaching App v1.1
**Focus:** Colab Enterprise workflow, Git LFS, ML training infrastructure
**Researched:** 2026-01-29
**Overall confidence:** HIGH

## Executive Summary

For v1.1 features (Colab Enterprise workflow, Git LFS video storage, ML training improvements), the stack additions are minimal and focused. The existing Python 3.10 + TensorFlow 2.15 + MediaPipe 0.10.9 foundation remains unchanged. Key additions: Git LFS 3.7.1 for video storage, MLflow 3.9.0 for experiment tracking, kineticstoolkit 0.17.0 for biomechanical analysis, and Colab Enterprise runtime configuration. Critical constraint: Python 3.10 must be maintained - Python 3.12 breaks TensorFlow 2.15 compatibility.

---

## Core Stack (Unchanged)

These components from v1.0 remain as-is:

| Technology | Version | Purpose | Status |
|------------|---------|---------|--------|
| Python | 3.10 | Runtime | **Pin to 3.10** - See compatibility note below |
| TensorFlow | 2.15.x | ML framework | **Pin to 2.15.x** - Required for MediaPipe |
| MediaPipe | 0.10.9 | Pose estimation | **Pin exactly** - Stability |
| Streamlit | Latest | Web interface | Keep current |
| NumPy | <2.0.0 | Numerical ops | Keep constraint |
| OpenCV | >=4.8.0 | Video processing | Keep current |
| SciPy | Latest | Signal processing | Keep current |

**CRITICAL COMPATIBILITY NOTE:**

Python 3.10 is mandatory. Colab Enterprise defaults to Python 3.12 as of May 2025, which breaks the stack:

- **TensorFlow 2.15 does NOT support Python 3.12**
- TensorFlow 2.16+ supports Python 3.12, but MediaPipe 0.10.9 requires TensorFlow 2.15.x
- **Action required:** Configure Colab runtime template to use Python 3.10, not the default 3.12

---

## New Additions for v1.1

### 1. Version Control - Git LFS

| Component | Version | Purpose |
|-----------|---------|---------|
| Git LFS | 3.7.1 | Video file storage |

**Why Git LFS:**
- Built-in GitHub integration (existing repo stays on GitHub)
- Pointer-based approach keeps git history lightweight
- No migration needed - just add tracking for video files
- 10 GB free storage + 10 GB free bandwidth/month (sufficient for ShuttleSet subset)

**Why NOT DVC:**
- DVC is ML-focused data versioning with pipeline management
- Overkill for simple "store videos, version outputs" use case
- Adds complexity (separate remote storage setup, learning curve)
- v1.1 doesn't need pipeline versioning yet

**Installation:**

```bash
# macOS
brew install git-lfs

# Linux
sudo apt-get install git-lfs

# Initialize once per user
git lfs install
```

**Repository setup:**

```bash
# Track video files
git lfs track "*.mp4"
git lfs track "*.avi"

# Track large model outputs
git lfs track "*.h5"
git lfs track "*.pkl"
git lfs track "models/**"

# Commit tracking config
git add .gitattributes
git commit -m "Configure Git LFS tracking"
```

**Storage considerations:**

GitHub Free tier provides:
- 10 GB storage (free)
- 10 GB bandwidth/month (free)
- Additional storage: $0.07/GB/month
- Additional bandwidth: $0.0875/GB download

ShuttleSet subset estimate (forehand Clear + Smash only):
- ~3,347 clips × ~2 MB/clip = ~6.7 GB (fits in free tier)
- If expanding to all stroke types (~4,983 clips) = ~10 GB (at limit)

**Recommendation:** Start with forehand subset (Clear, Smash, Drop) to stay under 10 GB. Expand to Drive/Net shots only if needed.

**References:**
- [Git LFS Official Site](https://git-lfs.com/) - v3.7.1 installation
- [GitHub Git LFS Billing](https://docs.github.com/billing/managing-billing-for-git-large-file-storage/about-billing-for-git-large-file-storage) - Storage limits
- [Git LFS Best Practices](https://lakefs.io/blog/dvc-vs-git-vs-dolt-vs-lakefs/) - When to use LFS vs DVC

---

### 2. Colab Enterprise Runtime

| Component | Configuration | Purpose |
|-----------|---------------|---------|
| Python version | **3.10** (NOT default 3.12) | TensorFlow 2.15 compatibility |
| Machine type | n1-standard-4 or better | Training workload |
| Accelerator | GPU (T4 or better) | Model training speedup |
| Idle shutdown | 180 minutes (default) | Cost control |

**Setup requirements:**

1. **Create custom runtime template** (cannot use default - wrong Python version):

```bash
# Via gcloud CLI or Colab Enterprise UI
# Set release_name to py310 (NOT default py312)
```

2. **Pre-install dependencies:**

Colab Enterprise doesn't specify pre-installed packages. Install everything explicitly:

```bash
# In Colab runtime startup script or first-run cell
pip install tensorflow==2.15.1
pip install mediapipe==0.10.9
pip install protobuf==3.20.3  # Required for MediaPipe
pip install mlflow==3.9.0
pip install kineticstoolkit==0.17.0
pip install scikit-learn pandas numpy matplotlib seaborn
```

**Git integration workflow:**

Colab Enterprise does NOT have built-in git pull/push UI. Use shell commands:

```bash
# Clone repository (first time)
!git clone https://github.com/username/repo.git
%cd repo

# Pull latest data (before training)
!git lfs pull  # Download LFS files
!git pull      # Get code changes

# Push outputs (after training)
!git add models/*.h5 outputs/metrics.json
!git commit -m "Training run results"
!git push origin main  # Requires token authentication
```

**Authentication:**

Push operations require Personal Access Token (PAT), not password:

```bash
# Set up token once
!git config --global credential.helper store
# First push will prompt for token, then cached
```

**Limitations:**

- Runtime resets on shutdown (re-clone repo each session)
- No persistent workspace across sessions
- Must re-install packages unless baked into runtime template

**References:**
- [Colab Enterprise Runtimes](https://docs.cloud.google.com/colab/docs/runtimes) - Python 3.10 configuration
- [Colab Git Integration](https://saturncloud.io/blog/methods-for-using-git-with-google-colab/) - Workflow patterns
- [Python Version Compatibility](https://discuss.python.org/t/tensorflow-support-for-python-3-12/66346) - TensorFlow 2.15 requires Python <=3.11

---

### 3. Experiment Tracking - MLflow

| Component | Version | Purpose |
|-----------|---------|---------|
| MLflow | 3.9.0 | Model versioning, metrics tracking |

**Why MLflow 3.9.0:**
- Latest stable release (Jan 29, 2026)
- Model Registry for versioning trained models
- Experiment tracking for comparing feature engineering iterations
- TensorFlow integration (with Keras 2.x compatibility)

**Key features for v1.1:**

1. **Experiment tracking:**
   - Log different feature engineering approaches (427 features → coach-informed features)
   - Track metrics: accuracy, precision, recall, F1 per stroke type
   - Compare runs to validate improvements

2. **Model versioning:**
   - Register trained models with versions
   - Tag models: "baseline", "coach-features-v1", "production"
   - Track which features were used for each model

3. **Artifact storage:**
   - Save feature importance plots
   - Store confusion matrices
   - Archive training/validation splits

**Installation:**

```bash
pip install mlflow==3.9.0
```

**Basic usage pattern:**

```python
import mlflow
import mlflow.tensorflow

# Start experiment
mlflow.set_experiment("badminton-classification")

with mlflow.start_run(run_name="coach-features-v1"):
    # Log parameters
    mlflow.log_param("num_features", 500)
    mlflow.log_param("feature_set", "coach-informed")

    # Train model
    model = train_model(...)

    # Log metrics
    mlflow.log_metric("accuracy", 0.85)
    mlflow.log_metric("f1_score", 0.82)

    # Save model (TensorFlow flavor)
    mlflow.tensorflow.log_model(model, "model")

    # Log artifacts
    mlflow.log_artifact("outputs/feature_importance.png")
```

**TensorFlow 2.15 compatibility warning:**

MLflow has known issues with TensorFlow 2.15 + Keras 3.0:

- **Problem:** TensorFlow 2.15 requires `keras<2.16,>=2.15.0`, but Keras 3.x may be installed
- **Symptom:** Model saving fails with "Invalid filepath extension" error
- **Solution:** Pin Keras to 2.x series OR use SavedModel format instead of Keras flavor

```bash
# Ensure Keras 2.x
pip install "keras<3.0,>=2.15.0"
```

**Storage:**

MLflow stores artifacts locally by default. For Colab → GitHub workflow:

```bash
# Set artifact location to repo directory
mlflow.set_tracking_uri("file:./mlruns")

# Commit MLflow runs to git
git add mlruns/
git commit -m "Add training run artifacts"
git push
```

**References:**
- [MLflow 3.9.0 Release](https://mlflow.org/releases) - Latest features
- [MLflow Model Registry](https://mlflow.org/docs/latest/model-registry/) - Versioning guide
- [TensorFlow Keras Compatibility](https://github.com/mlflow/mlflow/issues/11411) - Known issue with TF 2.15

---

### 4. Biomechanical Analysis - Kineticstoolkit

| Component | Version | Purpose |
|-----------|---------|---------|
| kineticstoolkit | 0.17.0 | Sports biomechanics feature engineering |

**Why kineticstoolkit:**
- Designed for biomechanics research (perfect fit for badminton analysis)
- ISB-standard joint angle calculations
- Signal processing routines (filtering, derivatives, onset detection)
- Works with NumPy arrays (integrates with MediaPipe output)

**Use cases in v1.1:**

1. **Joint angle calculations:**
   - Elbow flexion/extension angles
   - Shoulder rotation angles
   - Wrist angles during stroke
   - Hip/knee angles for lower body mechanics

2. **Signal processing:**
   - Smooth noisy pose data (Gaussian filtering)
   - Detect stroke phases (backswing, forward swing, follow-through)
   - Calculate derivatives (velocity, acceleration, jerk)

3. **Biomechanical metrics:**
   - Angular velocities (e.g., wrist angular velocity at impact)
   - Segment lengths (arm extension ratio)
   - Body center of mass calculations

**Installation:**

```bash
# Requires conda for full functionality (c3d file support)
conda install -c conda-forge kineticstoolkit==0.17.0

# OR pip (limited - no c3d support, but sufficient for this project)
pip install kineticstoolkit==0.17.0
```

**Note:** Pip installation lacks c3d file reading (requires ezc3d from conda-forge). Not needed for this project (using MediaPipe CSV/numpy data, not motion capture c3d files).

**Integration with existing feature engineering:**

Current `feature_engineering_v2.py` uses SciPy for signal processing:

```python
from scipy import signal
from scipy.ndimage import gaussian_filter1d
```

Kineticstoolkit provides higher-level biomechanics abstractions:

```python
import kineticstoolkit.lab as ktk

# Example: Calculate elbow angle from shoulder, elbow, wrist landmarks
def calculate_elbow_angle(shoulder, elbow, wrist):
    """
    Calculate elbow flexion angle using ISB standards.

    Args:
        shoulder, elbow, wrist: (x, y, z) coordinates

    Returns:
        angle in degrees
    """
    # Kineticstoolkit approach
    v1 = elbow - shoulder  # Upper arm vector
    v2 = wrist - elbow     # Forearm vector
    angle = ktk.geometry.get_angle(v1, v2)
    return angle
```

**Alternative: Continue with SciPy/NumPy**

Kineticstoolkit adds convenience, but isn't strictly necessary. Current SciPy-based approach works. Consider kineticstoolkit if:

- Need ISB-standard joint angle definitions
- Want built-in biomechanics functions (vs. manual NumPy)
- Plan to expand to more complex kinematics later

For v1.1, **kineticstoolkit is optional**. Existing SciPy + NumPy can handle coach-informed features (angles, velocities, accelerations).

**Recommendation:** Start without kineticstoolkit. Add only if manual calculations become cumbersome.

**References:**
- [Kineticstoolkit PyPI](https://pypi.org/project/kineticstoolkit/) - v0.17.0 release
- [Kineticstoolkit Documentation](https://kineticstoolkit.uqam.ca/doc/index.html) - Joint angle calculations
- [ISB Standards](https://kineticstoolkit.uqam.ca/doc/kinematics_joint_angles.html) - Biomechanical conventions

---

### 5. Supporting Libraries (Optional Enhancements)

| Library | Version | Purpose | Priority |
|---------|---------|---------|----------|
| tqdm | >=4.65.0 | Progress bars (already in requirements.txt) | Keep |
| python-dotenv | >=1.0.0 | Environment variables (already in requirements.txt) | Keep |
| pytest | >=7.4.0 | Testing framework (currently optional) | **Add for v1.1** |
| black | >=23.7.0 | Code formatting (currently optional) | Low |

**Add pytest for v1.1:**

ML experiments need regression testing. Add tests for:

- Feature engineering consistency (same input → same features)
- Model loading/prediction (trained model works after saving)
- Benchmark calculation correctness

```bash
pip install pytest>=7.4.0 pytest-cov>=4.1.0
```

---

## What NOT to Add

### pyomeca (biomechanics framework)

- **Why considered:** Comprehensive biomechanics toolbox with signal processing, matrix operations, file format support
- **Why NOT adding:**
  - Overkill for this project (designed for lab-grade motion capture data)
  - Requires xarray (adds dependency complexity)
  - Current SciPy + NumPy approach is simpler and sufficient
  - Latest version (v2024.0.2) targets research labs, not sports video analysis
- **Decision:** Use kineticstoolkit if needed; skip pyomeca

**Reference:** [pyomeca GitHub](https://github.com/pyomeca/pyomeca)

### Jupyter/IPython in Colab

- **Why NOT adding:**
  - v1.1 explicitly uses terminal-based scripts, not notebooks
  - Colab Enterprise supports notebooks, but project constraint is .py scripts for reproducibility
  - Current requirements.txt has `jupyter>=1.0.0` - **remove this for v1.1**
- **Decision:** Remove Jupyter from requirements.txt (not needed in Colab script workflow)

### DVC (Data Version Control)

- **Why NOT adding:**
  - Git LFS simpler for "store videos, version outputs" use case
  - DVC adds pipeline management complexity not needed yet
  - No multi-stage pipeline to track (just: extract poses → engineer features → train)
  - Would require separate remote storage setup (S3/GCS)
- **Decision:** Use Git LFS; revisit DVC if pipeline complexity grows in v2.0

**Reference:** [DVC vs Git LFS Comparison](https://lakefs.io/blog/dvc-vs-git-vs-dolt-vs-lakefs/)

### TensorFlow 2.16+ or PyTorch

- **Why NOT upgrading/switching:**
  - MediaPipe 0.10.9 requires TensorFlow 2.15.x
  - Upgrading TensorFlow breaks MediaPipe compatibility
  - PyTorch doesn't integrate with MediaPipe (TensorFlow-only)
- **Decision:** Stay on TensorFlow 2.15.x, Python 3.10

---

## Installation Summary

**New v1.1 dependencies:**

```bash
# Git LFS (system-level)
brew install git-lfs  # macOS
# OR
sudo apt-get install git-lfs  # Linux

git lfs install

# Python packages (add to requirements.txt)
mlflow==3.9.0
kineticstoolkit==0.17.0  # Optional
pytest>=7.4.0
pytest-cov>=4.1.0

# Keras version constraint (for TensorFlow 2.15 compatibility)
keras>=2.15.0,<3.0.0
```

**Updated requirements.txt for v1.1:**

```txt
# Core Deep Learning
tensorflow>=2.15.0,<2.16.0  # Pin to 2.15.x for MediaPipe
keras>=2.15.0,<3.0.0        # NEW: Ensure Keras 2.x for MLflow compatibility

# Computer Vision & Pose Estimation
opencv-python>=4.8.0
mediapipe==0.10.9
protobuf==3.20.3  # Required for MediaPipe

# Data Processing
numpy>=1.24.0,<2.0.0
pandas>=2.0.0
scikit-learn>=1.3.0

# Visualization
matplotlib>=3.7.0
seaborn>=0.12.0
plotly>=5.14.0

# Experiment Tracking
mlflow==3.9.0  # NEW: Model versioning and tracking

# Biomechanics (Optional)
kineticstoolkit==0.17.0  # NEW: Biomechanical analysis helpers

# Deployment
streamlit>=4.0.0  # Keep for web interface

# Testing (NEW: Was optional, now required)
pytest>=7.4.0
pytest-cov>=4.1.0

# Utilities
tqdm>=4.65.0
pillow>=10.0.0
python-dotenv>=1.0.0

# REMOVED: jupyter>=1.0.0, ipykernel>=6.25.0, notebook>=7.0.0
# Reason: Using terminal scripts in Colab, not notebooks
```

---

## Colab Enterprise Setup Checklist

1. **Create custom runtime template:**
   - [ ] Set Python version to 3.10 (NOT default 3.12)
   - [ ] Select machine type: n1-standard-4 or better
   - [ ] Add GPU accelerator (T4 recommended)
   - [ ] Set idle shutdown: 180 minutes

2. **Configure git integration:**
   - [ ] Generate GitHub Personal Access Token (PAT) with repo access
   - [ ] Configure git credentials in Colab runtime
   - [ ] Test clone/pull/push workflow

3. **Set up Git LFS:**
   - [ ] Install git-lfs in local environment
   - [ ] Configure `.gitattributes` for video files
   - [ ] Test LFS push/pull in Colab (use `git lfs pull`)

4. **Install dependencies:**
   - [ ] Create runtime startup script with pip installs
   - [ ] OR: Build custom Docker image with dependencies pre-installed
   - [ ] Verify TensorFlow 2.15 + MediaPipe 0.10.9 compatibility

5. **Set up MLflow tracking:**
   - [ ] Initialize MLflow experiment
   - [ ] Configure artifact location (repo directory for git commit)
   - [ ] Test run logging and model saving

---

## Version Compatibility Matrix

| Component | Version | Python 3.10 | Python 3.12 | Notes |
|-----------|---------|-------------|-------------|-------|
| TensorFlow | 2.15.x | ✓ Yes | ✗ No | TF 2.15 max Python 3.11 |
| MediaPipe | 0.10.9 | ✓ Yes | ✓ Yes | Works with both, but TF constraint |
| MLflow | 3.9.0 | ✓ Yes | ✓ Yes | Requires Python >=3.10 |
| Kineticstoolkit | 0.17.0 | ✓ Yes | ✓ Yes | Requires Python >=3.10 |
| Keras | 2.15.x | ✓ Yes | ✗ No | Must be <3.0 for MLflow + TF 2.15 |

**Conclusion:** Python 3.10 is the only safe choice. Python 3.12 breaks TensorFlow 2.15.

---

## Risk Assessment

| Risk | Severity | Mitigation |
|------|----------|------------|
| Colab defaults to Python 3.12 (breaks TensorFlow 2.15) | **HIGH** | Configure custom runtime template with Python 3.10 explicitly |
| Git LFS storage exceeds 10 GB free tier | Medium | Start with forehand subset only (Clear, Smash, Drop) |
| MLflow + TensorFlow 2.15 + Keras 3.x compatibility issue | Medium | Pin Keras to 2.x series in requirements.txt |
| Colab runtime resets lose git state | Low | Document clone/pull workflow clearly in scripts |
| GitHub PAT expiration breaks push workflow | Low | Use long-lived token, document renewal process |

---

## Sources

**Git LFS:**
- [Git LFS Official Site](https://git-lfs.com/) - Latest version and installation
- [GitHub Git LFS Billing](https://docs.github.com/billing/managing-billing-for-git-large-file-storage/about-billing-for-git-large-file-storage) - Storage limits
- [DVC vs Git LFS Comparison](https://lakefs.io/blog/dvc-vs-git-vs-dolt-vs-lakefs/) - Tool selection rationale

**Colab Enterprise:**
- [Colab Enterprise Runtimes Documentation](https://docs.cloud.google.com/colab/docs/runtimes) - Python version configuration
- [Colab Git Integration Guide](https://saturncloud.io/blog/methods-for-using-git-with-google-colab/) - Workflow patterns

**MLflow:**
- [MLflow Releases](https://mlflow.org/releases) - v3.9.0 release notes
- [TensorFlow + Keras Compatibility Issue](https://github.com/mlflow/mlflow/issues/11411) - Known bug with TF 2.15

**Biomechanical Libraries:**
- [Kineticstoolkit PyPI](https://pypi.org/project/kineticstoolkit/) - v0.17.0 release
- [Kineticstoolkit Documentation](https://kineticstoolkit.uqam.ca/doc/index.html) - Joint angle calculations
- [pyomeca GitHub](https://github.com/pyomeca/pyomeca) - Alternative library (not selected)

**Python Compatibility:**
- [TensorFlow Python 3.12 Support Discussion](https://discuss.python.org/t/tensorflow-support-for-python-3-12/66346) - Version constraints
- [MediaPipe PyPI](https://pypi.org/project/mediapipe/) - Python version support
