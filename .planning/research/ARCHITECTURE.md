# Architecture Research: Colab Enterprise Integration for v1.1

**Domain:** ML Training Infrastructure + Feature Engineering Pipeline
**Researched:** 2026-01-29
**Confidence:** MEDIUM

## Executive Summary

This research addresses how to integrate Colab Enterprise for ML model retraining while maintaining the existing local benchmark-based system. The architecture must support:

1. **Bidirectional git workflow**: Local development → git → Colab Enterprise → git → local deployment
2. **Data versioning**: Git LFS for videos, feature versioning for reproducibility
3. **Script-based execution**: Terminal mode Python scripts (not notebooks) in Colab
4. **Feature engineering expansion**: From 427 to enhanced feature set with coach-informed metrics
5. **Coexistence pattern**: Benchmark analysis (production) + ML classification (experimental)

**Key Finding**: Use a **layered integration pattern** where Colab augments (not replaces) the existing pipeline, with clear separation between training infrastructure and production inference.

---

## Current Architecture (v1.0 Baseline)

### System Overview (Existing)

```
┌─────────────────────────────────────────────────────────────────┐
│                    LOCAL DEVELOPMENT ENVIRONMENT                 │
├─────────────────────────────────────────────────────────────────┤
│  PRESENTATION LAYER                                              │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐          │
│  │  Streamlit   │  │   Gradio     │  │  analyze_    │          │
│  │    Web UI    │  │   Web UI     │  │  video.py    │          │
│  └──────┬───────┘  └──────┬───────┘  └──────┬───────┘          │
│         │                  │                  │                  │
├─────────┴──────────────────┴──────────────────┴──────────────────┤
│  ANALYSIS LAYER (Benchmark-based, NO ML in production)          │
│  ┌──────────────────────────────────────────────────────┐        │
│  │  CoachingFeedback (rule-based)                       │        │
│  │  ├─ TechniqueBenchmarks (percentile ranges)          │        │
│  │  ├─ FeedbackItem generation                          │        │
│  │  └─ Overall score calculation (0-100)                │        │
│  └──────────────────────────────────────────────────────┘        │
├─────────────────────────────────────────────────────────────────┤
│  FEATURE ENGINEERING LAYER                                       │
│  ┌──────────────────────────────────────────────────────┐        │
│  │  feature_engineering_v2.py                           │        │
│  │  ├─ extract_frame_features() → 60 spatial features   │        │
│  │  ├─ extract_temporal_features() → velocity, accel    │        │
│  │  └─ extract_statistical_summary() → 427 features     │        │
│  └──────────────────────────────────────────────────────┘        │
├─────────────────────────────────────────────────────────────────┤
│  POSE EXTRACTION LAYER                                           │
│  ┌──────────────────────────────────────────────────────┐        │
│  │  PoseExtractor (MediaPipe)                           │        │
│  │  ├─ 33 keypoints per frame                           │        │
│  │  ├─ Multi-player detection                           │        │
│  │  └─ Temporal interpolation                           │        │
│  └──────────────────────────────────────────────────────┘        │
├─────────────────────────────────────────────────────────────────┤
│  DATA LAYER (Local filesystem)                                  │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────┐        │
│  │  Videos  │  │  Poses   │  │ Features │  │Benchmarks│        │
│  │  .mp4    │  │  .pkl    │  │  .pkl    │  │  .pkl    │        │
│  └──────────┘  └──────────┘  └──────────┘  └──────────┘        │
└─────────────────────────────────────────────────────────────────┘
```

### Data Flow (Existing v1.0)

```
User Video (upload)
    ↓
PoseExtractor.extract_from_video()
    ↓ (MediaPipe processing)
pose_data = {'poses': ndarray (T×99), 'quality': dict}
    ↓
extract_temporal_features(poses)
    ↓ (per-frame + velocity/accel)
per_frame_features (T×60), temporal_metrics
    ↓
extract_statistical_summary()
    ↓ (aggregation: min/max/mean/std/percentiles)
stat_features = dict (427 features)
    ↓
CoachingFeedback.analyze_technique(stat_features, stroke_type)
    ↓ (benchmark comparison)
feedback_items = List[FeedbackItem], overall_score
    ↓
TechniqueVisualizer.create_*()
    ↓
PNG visualizations + text feedback
```

**Critical Constraint**: ML classification models exist (`baseline_model.py`, `lstm_model.py`) but are **NOT used in production** due to low accuracy. System uses benchmark-based analysis only.

---

## Target Architecture (v1.1 with Colab Enterprise)

### Integrated System Overview

```
┌─────────────────────────────────────────────────────────────────┐
│                    LOCAL DEVELOPMENT ENVIRONMENT                 │
│  (Production Inference - Benchmark-based)                        │
├─────────────────────────────────────────────────────────────────┤
│  ┌──────────────┐  ┌──────────────────────────────────┐         │
│  │  Streamlit   │  │  analyze_video.py                │         │
│  │   Web UI     │  │  (unchanged production flow)     │         │
│  └──────────────┘  └──────────────────────────────────┘         │
│                                                                  │
│  Feature Engineering V3 (expanded)                               │
│  ┌──────────────────────────────────────────────────────┐       │
│  │  - Existing 427 features (preserved)                 │       │
│  │  - NEW: Coach-informed biomechanics                  │       │
│  │  - NEW: Stroke-specific kinematics                   │       │
│  │  - NEW: Temporal phase segmentation                  │       │
│  └──────────────────────────────────────────────────────┘       │
└──────────────────────┬───────────────────────────────────────────┘
                       │
                       │ (1) Push code + metadata
                       ↓
┌─────────────────────────────────────────────────────────────────┐
│                         GIT REPOSITORY                           │
│  (Version Control + Git LFS Storage)                             │
├─────────────────────────────────────────────────────────────────┤
│  Code:                    Data (Git LFS):                        │
│  ├─ src/                  ├─ data/raw/videos/                   │
│  ├─ scripts/train/        ├─ data/processed/poses/              │
│  ├─ scripts/evaluate/     ├─ data/processed/features/           │
│  └─ configs/              └─ data/processed/splits/              │
│                                                                  │
│  Outputs (Git LFS):       Metadata (regular git):                │
│  ├─ models/saved/         ├─ data/processed/clips_metadata.csv  │
│  ├─ outputs/metrics/      ├─ data/processed/train_metadata.csv  │
│  └─ outputs/features/     └─ configs/feature_config.yaml        │
└──────────────────────┬───────────────────────────────────────────┘
                       │
                       │ (2) Clone/pull for training
                       ↓
┌─────────────────────────────────────────────────────────────────┐
│              COLAB ENTERPRISE TRAINING ENVIRONMENT               │
│  (Scalable GPU Training - Script Execution)                      │
├─────────────────────────────────────────────────────────────────┤
│  ORCHESTRATION LAYER (Terminal-based)                            │
│  ┌──────────────────────────────────────────────────────┐       │
│  │  train_pipeline.py (main orchestrator)               │       │
│  │  ├─ Load data from git LFS                           │       │
│  │  ├─ Feature engineering V3                           │       │
│  │  ├─ Train classification models                      │       │
│  │  ├─ Evaluate on validation set                       │       │
│  │  ├─ Generate metrics + reports                       │       │
│  │  └─ Export artifacts for git push                    │       │
│  └──────────────────────────────────────────────────────┘       │
│                                                                  │
│  TRAINING SCRIPTS (modular)                                      │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐          │
│  │ feature_     │  │ train_       │  │ evaluate_    │          │
│  │ engineering_ │  │ classifier.  │  │ model.py     │          │
│  │ v3.py        │  │ py           │  │              │          │
│  └──────────────┘  └──────────────┘  └──────────────┘          │
│                                                                  │
│  EXECUTION WORKFLOW                                              │
│  ┌──────────────────────────────────────────────────────┐       │
│  │  1. !git clone <repo> (with LFS)                     │       │
│  │  2. !cd repo && git lfs pull                         │       │
│  │  3. !pip install -r requirements.txt                 │       │
│  │  4. !python scripts/train/train_pipeline.py          │       │
│  │  5. !git add models/ outputs/                        │       │
│  │  6. !git commit -m "Training run [date]"             │       │
│  │  7. !git push                                         │       │
│  └──────────────────────────────────────────────────────┘       │
│                                                                  │
│  RUNTIME TEMPLATE (configuration)                                │
│  - Machine type: n1-standard-8 with T4 GPU                       │
│  - Environment: TensorFlow 2.15.x, MediaPipe 0.10.9              │
│  - Git credentials: service account or OAuth                     │
└──────────────────────┬───────────────────────────────────────────┘
                       │
                       │ (3) Pull trained models
                       ↓
┌─────────────────────────────────────────────────────────────────┐
│                    LOCAL DEVELOPMENT ENVIRONMENT                 │
│  (Model Evaluation + Integration)                                │
├─────────────────────────────────────────────────────────────────┤
│  ┌──────────────────────────────────────────────────────┐       │
│  │  1. git pull (fetch trained models)                  │       │
│  │  2. Evaluate model accuracy locally                  │       │
│  │  3. IF accuracy > threshold:                         │       │
│  │     - Integrate into production pipeline             │       │
│  │     - A/B test against benchmark analysis            │       │
│  │  4. ELSE:                                             │       │
│  │     - Continue using benchmark-based analysis        │       │
│  │     - Iterate on feature engineering                 │       │
│  └──────────────────────────────────────────────────────┘       │
└─────────────────────────────────────────────────────────────────┘
```

### Component Responsibilities

| Component | Responsibility | Location | Notes |
|-----------|----------------|----------|-------|
| **PoseExtractor** | MediaPipe pose extraction (33 keypoints) | Local + Colab | Unchanged - shared code |
| **FeatureEngineering V3** | Expanded biomechanical features | Local + Colab | NEW - enhanced from 427 features |
| **CoachingFeedback** | Benchmark-based analysis (production) | Local only | Unchanged - production system |
| **TrainingPipeline** | Model retraining orchestration | Colab only | NEW - terminal script |
| **ClassificationModel** | ML stroke classification | Colab (train) + Local (inference) | UPDATED - improved accuracy target |
| **Git LFS Manager** | Video/artifact versioning | Git | NEW - data versioning |
| **ModelEvaluator** | Accuracy validation | Local + Colab | NEW - validates before production |

---

## New Components for v1.1

### 1. Feature Engineering V3 Pipeline

**Location**: `src/data_processing/feature_engineering_v3.py`

**Enhancements**:
- Preserve existing 427 features for backward compatibility
- Add coach-informed features (from literature/video research):
  - Racket angle relative to forearm (pronation/supination indicators)
  - Hip rotation timing relative to shoulder rotation
  - Lower body weight transfer (ankle/knee/hip coordination)
  - Stroke phase segmentation (preparation, acceleration, contact, follow-through)
- Stroke-specific features (Clear, Smash, Drop, Drive, Net shots)
- Temporal phase detection using velocity peaks

**Integration Pattern**:
```python
# Backward compatible
def extract_features_v3(pose_sequence, version='v3'):
    """
    Extract features with version control

    Args:
        pose_sequence: MediaPipe poses
        version: 'v2' (427 features) or 'v3' (enhanced)

    Returns:
        features_dict with 'version' metadata
    """
    if version == 'v2':
        return extract_features_v2(pose_sequence)  # Existing

    # V3: Existing + new
    v2_features = extract_features_v2(pose_sequence)
    coach_features = extract_coach_informed_features(pose_sequence)
    phase_features = extract_phase_segmentation(pose_sequence)

    return {
        'version': 'v3',
        'v2_features': v2_features,  # 427 existing
        'coach_features': coach_features,  # ~100 new
        'phase_features': phase_features,  # ~50 new
        'combined': {**v2_features, **coach_features, **phase_features}
    }
```

**Feature Count**: ~577 total (427 existing + 150 new)

### 2. Git LFS Integration

**Setup**:
```bash
# Initialize Git LFS (one-time setup)
git lfs install

# Track large file types
git lfs track "data/raw/videos/**/*.mp4"
git lfs track "data/processed/poses/**/*.pkl"
git lfs track "data/processed/features/**/*.pkl"
git lfs track "models/saved/**/*.h5"
git lfs track "models/saved/**/*.pkl"

# Commit .gitattributes
git add .gitattributes
git commit -m "Configure Git LFS for videos and models"
```

**Data Organization**:
```
data/
├── raw/
│   └── videos/              # Git LFS tracked
│       ├── shuttleset/      # Original dataset
│       └── external/        # User-submitted for validation
├── processed/
│   ├── poses/               # Git LFS tracked (large .pkl files)
│   ├── features/            # Git LFS tracked (feature vectors)
│   ├── splits/              # Git LFS tracked (train/val/test)
│   └── clips_metadata.csv   # Regular git (small, text)
```

**Limitations**:
- GitHub free tier: 1GB storage, 1GB/month bandwidth
- GitHub LFS max file size: 2GB per file
- **Recommendation**: Use selective LFS pulling in Colab to minimize bandwidth

**Workflow Pattern**:
```bash
# Colab: Clone without LFS data initially
GIT_LFS_SKIP_SMUDGE=1 git clone <repo>

# Pull only required files
cd repo
git lfs pull --include="data/processed/splits/*"  # Training data only
git lfs pull --include="models/saved/*"           # Existing models

# After training: Push new models
git lfs push --all
```

### 3. Colab Training Pipeline

**Location**: `scripts/train/train_pipeline.py`

**Execution Model**: Terminal-based Python scripts (NOT notebooks)

**Why Scripts Over Notebooks**:
- Version control friendly (no JSON metadata noise)
- Reproducible (deterministic execution order)
- Automatable (can be scheduled/orchestrated)
- Testable (can import and unit test functions)
- CI/CD compatible (can run in pipelines)

**Structure**:
```
scripts/
├── train/
│   ├── train_pipeline.py        # Main orchestrator
│   ├── train_classifier.py      # Model training logic
│   ├── hyperparameter_search.py # Grid/random search
│   └── utils.py                 # Shared utilities
├── evaluate/
│   ├── evaluate_model.py        # Validation metrics
│   ├── compare_models.py        # Baseline vs new
│   └── generate_reports.py      # Training summary
└── setup/
    ├── setup_colab_env.sh       # Environment setup
    └── download_data.sh         # Git LFS selective pull
```

**Main Orchestrator**:
```python
# scripts/train/train_pipeline.py
"""
Main training pipeline for Colab Enterprise execution
Usage: python scripts/train/train_pipeline.py --config configs/train_config.yaml
"""

def main(config_path):
    # 1. Setup
    config = load_config(config_path)
    setup_logging(config['output_dir'])

    # 2. Data loading
    train_data = load_training_data(config['data_dir'])
    val_data = load_validation_data(config['data_dir'])

    # 3. Feature engineering V3
    train_features = extract_features_v3(train_data, version='v3')
    val_features = extract_features_v3(val_data, version='v3')

    # 4. Model training
    models = {
        'random_forest': train_random_forest(train_features, config['rf_params']),
        'svm': train_svm(train_features, config['svm_params']),
        'lstm': train_lstm(train_features, config['lstm_params'])
    }

    # 5. Evaluation
    results = {}
    for name, model in models.items():
        metrics = evaluate_model(model, val_features)
        results[name] = metrics
        save_model(model, f"models/saved/{name}_v3.pkl")

    # 6. Generate artifacts
    generate_training_report(results, config['output_dir'])
    save_feature_importance(models['random_forest'], config['output_dir'])

    # 7. Export for git
    export_metadata = {
        'timestamp': datetime.now().isoformat(),
        'config': config,
        'results': results,
        'feature_version': 'v3'
    }
    save_json(export_metadata, 'outputs/metrics/training_run.json')

    print("Training complete. Ready for git commit.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', required=True)
    args = parser.parse_args()
    main(args.config)
```

**Colab Execution Notebook** (minimal wrapper):
```python
# colab_runner.ipynb (single cell)
%%bash
# Setup
pip install -q -r requirements.txt

# Run training pipeline
python scripts/train/train_pipeline.py \
    --config configs/train_config.yaml

# Commit results
git config user.email "colab-bot@example.com"
git config user.name "Colab Training Bot"
git add models/saved/ outputs/metrics/
git commit -m "Training run $(date +%Y-%m-%d)"
git push
```

### 4. Colab Enterprise Setup

**Runtime Template Configuration**:
```yaml
# .colab/runtime_template.yaml
name: "badminton-training"
machine_type: "n1-standard-8"  # 8 vCPUs, 30GB RAM
accelerator:
  type: "NVIDIA_TESLA_T4"
  count: 1
boot_disk_size_gb: 100
environment:
  python_version: "3.10"
  packages:
    - tensorflow==2.15.0
    - mediapipe==0.10.9
    - scikit-learn>=1.3.0
    - protobuf==3.20.3
idle_timeout: "3600s"  # 1 hour
```

**Authentication**:
```python
# In Colab notebook (one-time setup)
from google.colab import auth
auth.authenticate_user()

# Configure git credentials
!git config --global credential.helper store
!git config --global user.email "colab@example.com"
!git config --global user.name "Colab Enterprise"

# Generate GitHub personal access token (PAT) and store
# OR use service account with repo access
```

---

## Data Flow Patterns

### Training Data Flow (Local → Colab → Local)

```
┌─────────────────────────────────────────────────────────────────┐
│ LOCAL: Initial Feature Engineering                               │
├─────────────────────────────────────────────────────────────────┤
│ 1. Process ShuttleSet videos locally                             │
│    → extract_poses.py (MediaPipe)                                │
│    → feature_engineering_v3.py                                   │
│                                                                  │
│ 2. Create train/val/test splits                                  │
│    → data_split.py                                               │
│                                                                  │
│ 3. Save to git LFS                                               │
│    → data/processed/poses/*.pkl (LFS)                            │
│    → data/processed/features/*.pkl (LFS)                         │
│    → data/processed/splits/*.pkl (LFS)                           │
│                                                                  │
│ 4. Commit and push                                               │
│    → git add data/ (auto-tracked by LFS)                         │
│    → git commit -m "Update features v3"                          │
│    → git push origin main                                        │
└──────────────────────┬───────────────────────────────────────────┘
                       │
                       ↓ (clone/pull)
┌─────────────────────────────────────────────────────────────────┐
│ COLAB: Model Training                                            │
├─────────────────────────────────────────────────────────────────┤
│ 1. Clone repository with selective LFS                           │
│    → GIT_LFS_SKIP_SMUDGE=1 git clone <repo>                      │
│    → git lfs pull --include="data/processed/splits/*"            │
│                                                                  │
│ 2. Execute training pipeline                                     │
│    → python scripts/train/train_pipeline.py                      │
│       ├─ Load splits from LFS                                    │
│       ├─ Train Random Forest                                     │
│       ├─ Train SVM                                               │
│       ├─ Train LSTM                                              │
│       └─ Generate evaluation metrics                             │
│                                                                  │
│ 3. Save trained models                                           │
│    → models/saved/rf_v3_[timestamp].pkl                          │
│    → models/saved/svm_v3_[timestamp].pkl                         │
│    → models/saved/lstm_v3_[timestamp].h5                         │
│    → outputs/metrics/training_run_[timestamp].json               │
│                                                                  │
│ 4. Commit and push results                                       │
│    → git add models/saved/ outputs/metrics/                      │
│    → git commit -m "Training run [timestamp]"                    │
│    → git lfs push                                                │
└──────────────────────┬───────────────────────────────────────────┘
                       │
                       ↓ (pull)
┌─────────────────────────────────────────────────────────────────┐
│ LOCAL: Model Evaluation & Integration                            │
├─────────────────────────────────────────────────────────────────┤
│ 1. Pull trained models                                           │
│    → git pull origin main                                        │
│    → git lfs pull --include="models/saved/*"                     │
│                                                                  │
│ 2. Evaluate accuracy locally                                     │
│    → python scripts/evaluate/evaluate_model.py                   │
│       ├─ Load test set                                           │
│       ├─ Load trained models                                     │
│       └─ Calculate accuracy, F1, confusion matrix                │
│                                                                  │
│ 3. Decision: Integrate or iterate                                │
│    IF accuracy > 85% AND F1 > 0.80:                              │
│       → Update production code to use classifier                 │
│       → A/B test: benchmark vs ML classification                 │
│    ELSE:                                                          │
│       → Analyze feature importance                               │
│       → Research additional coach-informed features              │
│       → Return to step 1 (feature engineering)                   │
└─────────────────────────────────────────────────────────────────┘
```

### Inference Data Flow (Production - Unchanged)

```
User uploads video (Streamlit)
    ↓
PoseExtractor.extract_from_video()
    ↓
extract_features_v3(poses, version='v3')
    ↓
┌─────────────────────────────────────────────┐
│  DECISION POINT (configurable)              │
├─────────────────────────────────────────────┤
│  IF use_ml_classifier AND model_available:  │
│     → Load trained model                    │
│     → Predict stroke type                   │
│     → IF confidence > threshold:            │
│        USE: ML classification               │
│     → ELSE:                                  │
│        FALLBACK: Benchmark analysis         │
│  ELSE:                                       │
│     → Use benchmark analysis (current v1.0) │
└─────────────────────────────────────────────┘
    ↓
CoachingFeedback.analyze_technique()
    ↓
TechniqueVisualizer.create_visualizations()
    ↓
Display feedback to user
```

**Key Design Decision**: ML classification is **additive**, not **replacement**. Benchmark analysis remains the production default, with ML as an experimental enhancement.

---

## Recommended Project Structure

```
iti123_v2/
├── src/                          # Existing production code
│   ├── data_processing/
│   │   ├── extract_poses.py      # UNCHANGED
│   │   ├── feature_engineering_v2.py  # PRESERVED (v1.0)
│   │   ├── feature_engineering_v3.py  # NEW (v1.1)
│   │   └── data_split.py         # UPDATED (support v3 features)
│   ├── coaching/
│   │   ├── technique_benchmarks.py  # UNCHANGED (production)
│   │   ├── feedback_generator.py    # UNCHANGED (production)
│   │   └── visualizations.py        # UNCHANGED
│   ├── models/
│   │   ├── baseline_model.py     # UPDATED (v3 features)
│   │   ├── lstm_model.py         # UPDATED (v3 features)
│   │   └── model_loader.py       # NEW (prod integration)
│   └── deployment/
│       ├── streamlit_app.py      # UPDATED (optional ML mode)
│       └── coaching_app.py       # UPDATED
│
├── scripts/                      # NEW: Colab training scripts
│   ├── train/
│   │   ├── train_pipeline.py     # Main orchestrator
│   │   ├── train_classifier.py   # Training logic
│   │   └── hyperparameter_search.py
│   ├── evaluate/
│   │   ├── evaluate_model.py     # Accuracy validation
│   │   ├── compare_models.py     # Baseline comparison
│   │   └── generate_reports.py   # Training summaries
│   ├── features/
│   │   ├── analyze_feature_importance.py
│   │   └── visualize_features.py
│   └── setup/
│       ├── setup_colab_env.sh    # Environment setup
│       └── download_data.sh      # Git LFS selective pull
│
├── configs/                      # NEW: Configuration files
│   ├── train_config.yaml         # Training hyperparameters
│   ├── feature_config_v3.yaml    # Feature engineering config
│   └── model_config.yaml         # Model architecture configs
│
├── data/
│   ├── raw/
│   │   └── videos/               # Git LFS tracked
│   ├── processed/
│   │   ├── poses/                # Git LFS tracked
│   │   ├── features/             # Git LFS tracked
│   │   ├── splits/               # Git LFS tracked
│   │   └── clips_metadata.csv    # Regular git
│
├── models/
│   ├── saved/                    # Git LFS tracked
│   │   ├── rf_v3_20260129.pkl
│   │   ├── svm_v3_20260129.pkl
│   │   └── lstm_v3_20260129.h5
│   └── registry.json             # Model metadata (regular git)
│
├── outputs/
│   ├── metrics/                  # Regular git (JSON/CSV)
│   │   └── training_run_*.json
│   ├── reports/                  # Regular git (text)
│   └── visualizations/           # Git LFS (images)
│
├── notebooks/                    # NEW: Colab notebooks
│   ├── colab_runner.ipynb        # Minimal training wrapper
│   └── analysis/                 # Exploratory notebooks
│       ├── feature_analysis.ipynb
│       └── error_analysis.ipynb
│
├── .colab/                       # NEW: Colab configuration
│   └── runtime_template.yaml
│
├── .gitattributes                # Git LFS configuration
├── requirements.txt              # Python dependencies (unchanged)
└── requirements-colab.txt        # NEW: Colab-specific deps
```

### Structure Rationale

- **`src/`**: Production code (runs in Streamlit). Unchanged for backward compatibility.
- **`scripts/`**: Training code (runs in Colab). Separated from production for clarity.
- **`configs/`**: YAML configurations for reproducibility. Version controlled.
- **`notebooks/`**: Colab notebooks (minimal wrappers). Not production-critical.
- **`.colab/`**: Colab runtime configuration. Separate from code for reusability.

---

## Architectural Patterns

### Pattern 1: Version-Gated Feature Engineering

**What**: Support multiple feature versions simultaneously with explicit version flags.

**When to use**: When upgrading feature engineering while maintaining backward compatibility.

**Trade-offs**:
- **Pros**: No breaking changes, can A/B test features, supports gradual rollout
- **Cons**: Code complexity, need to maintain multiple versions, storage overhead

**Example**:
```python
def extract_features(pose_sequence, version='v3'):
    """
    Version-gated feature extraction

    v2: 427 features (production baseline)
    v3: 577 features (experimental with coach-informed metrics)
    """
    if version == 'v2':
        return _extract_v2_features(pose_sequence)
    elif version == 'v3':
        v2 = _extract_v2_features(pose_sequence)
        coach = _extract_coach_features(pose_sequence)
        return {**v2, **coach, '_version': 'v3'}
    else:
        raise ValueError(f"Unknown version: {version}")

# Usage in production
features = extract_features(poses, version='v2')  # Safe, tested

# Usage in training
features = extract_features(poses, version='v3')  # Experimental
```

### Pattern 2: Selective Git LFS Pulling

**What**: Clone git repo without LFS data, then selectively pull only required files.

**When to use**: When working with large datasets in bandwidth-limited environments (Colab).

**Trade-offs**:
- **Pros**: Faster clones, reduced bandwidth costs, only download what's needed
- **Cons**: Manual management, need to know file structure, easy to miss dependencies

**Example**:
```bash
# Colab setup script
#!/bin/bash
# scripts/setup/download_data.sh

# 1. Clone WITHOUT LFS data (fast)
GIT_LFS_SKIP_SMUDGE=1 git clone https://github.com/user/repo.git
cd repo

# 2. Pull ONLY training data (not raw videos)
git lfs pull --include="data/processed/splits/train_data.pkl"
git lfs pull --include="data/processed/splits/val_data.pkl"

# 3. Pull ONLY metadata (small files, not LFS)
# These are regular git files, no LFS pull needed
ls data/processed/clips_metadata.csv

echo "Data ready for training"
```

### Pattern 3: Dual-Mode Inference (Benchmark + ML)

**What**: Production system supports both benchmark-based analysis (current) and ML classification (experimental) with runtime switching.

**When to use**: When introducing ML to a rule-based system without breaking existing functionality.

**Trade-offs**:
- **Pros**: Safe rollout, can A/B test, easy rollback, gradual migration
- **Cons**: Increased code complexity, need to maintain both paths, potential inconsistency

**Example**:
```python
class StrokeAnalyzer:
    def __init__(self, mode='benchmark', model_path=None):
        self.mode = mode
        self.benchmark = TechniqueBenchmarks()
        self.model = None

        if mode == 'ml' and model_path:
            self.model = load_model(model_path)

    def analyze(self, features, stroke_type):
        if self.mode == 'ml' and self.model is not None:
            # ML classification path
            prediction = self.model.predict(features)
            confidence = self.model.predict_proba(features).max()

            if confidence > 0.85:  # High confidence
                return self._ml_feedback(prediction, features)
            else:
                # Fallback to benchmark
                return self._benchmark_feedback(features, stroke_type)
        else:
            # Benchmark path (production default)
            return self._benchmark_feedback(features, stroke_type)

# Usage
analyzer = StrokeAnalyzer(mode='benchmark')  # Production
# analyzer = StrokeAnalyzer(mode='ml', model_path='models/saved/rf_v3.pkl')  # Experimental
```

### Pattern 4: Pipeline Orchestration via Terminal Scripts

**What**: Use executable Python scripts (not notebooks) for training pipelines, orchestrated via command line.

**When to use**: When reproducibility, automation, and CI/CD integration matter more than interactivity.

**Trade-offs**:
- **Pros**: Version control friendly, reproducible, automatable, testable, CI/CD ready
- **Cons**: Less interactive than notebooks, harder for exploratory work, requires discipline

**Example**:
```python
# scripts/train/train_pipeline.py
"""
Main training pipeline
Usage:
  python scripts/train/train_pipeline.py --config configs/train_config.yaml
  python scripts/train/train_pipeline.py --config configs/train_config.yaml --quick-test
"""
import argparse
import logging
from pathlib import Path

def setup_logging(output_dir):
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(output_dir / 'training.log'),
            logging.StreamHandler()
        ]
    )

def main(config_path, quick_test=False):
    logger = logging.getLogger(__name__)
    logger.info(f"Starting training pipeline with config: {config_path}")

    # Load config
    config = load_yaml(config_path)

    # Override for quick testing
    if quick_test:
        config['epochs'] = 2
        config['data_subset'] = 100

    # Run pipeline stages
    train_models(config)
    evaluate_models(config)
    export_artifacts(config)

    logger.info("Training complete")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', required=True, help='Path to config YAML')
    parser.add_argument('--quick-test', action='store_true', help='Quick test mode')
    args = parser.parse_args()

    main(args.config, args.quick_test)
```

**Colab Wrapper** (minimal notebook):
```python
# colab_runner.ipynb
# Cell 1: Setup
!git clone https://github.com/user/repo.git
%cd repo
!pip install -r requirements-colab.txt

# Cell 2: Run pipeline
!python scripts/train/train_pipeline.py --config configs/train_config.yaml

# Cell 3: Push results
!git add models/saved/ outputs/metrics/
!git commit -m "Training run $(date)"
!git push
```

---

## Integration Points

### Existing Components (Modified)

| Component | Modification | Reason |
|-----------|--------------|--------|
| `feature_engineering_v2.py` | **Preserved unchanged** | Backward compatibility for production |
| `streamlit_app.py` | **Add optional ML mode** | Allow users to opt-in to ML classification |
| `data_split.py` | **Support v3 features** | Load and split new feature format |
| `baseline_model.py` | **Update input shape** | Handle 577 features instead of 427 |
| `derive_benchmarks.py` | **No change required** | Operates on v2 features (production) |

### New Components

| Component | Purpose | Integration Point |
|-----------|---------|-------------------|
| `feature_engineering_v3.py` | Expanded features | Called by both local and Colab pipelines |
| `scripts/train/train_pipeline.py` | Training orchestrator | Runs in Colab, produces models for local |
| `scripts/evaluate/evaluate_model.py` | Model validation | Runs locally after pulling from git |
| `src/models/model_loader.py` | Production model loading | Used by Streamlit if ML mode enabled |
| `.gitattributes` | Git LFS configuration | Tracks large files automatically |

### External Services

| Service | Integration Pattern | Notes |
|---------|---------------------|-------|
| **Git LFS** | File tracking via `.gitattributes` | Automatic on `git add`, requires `git lfs pull` for retrieval |
| **Colab Enterprise** | Terminal script execution via `!python` | Runtime template specifies GPU/CPU resources |
| **GitHub** | Standard git workflow + LFS push/pull | Personal access token (PAT) for authentication |
| **MediaPipe** | Shared library (local + Colab) | Version pinned to 0.10.9 for consistency |

### Internal Boundaries

| Boundary | Communication | Notes |
|----------|---------------|-------|
| **Local ↔ Git** | `git push/pull` + `git lfs push/pull` | Bidirectional sync, LFS for large files |
| **Git ↔ Colab** | `git clone/pull` + `git lfs pull --include` | Selective LFS pulling to save bandwidth |
| **Training ↔ Production** | Trained models in `models/saved/`, loaded by `model_loader.py` | One-way flow (Colab trains, local deploys) |
| **v2 ↔ v3 Features** | Version flag in feature extraction | Isolated, can run both in parallel |

---

## Data Versioning Strategy

### What to Version

| Data Type | Versioning Approach | Storage |
|-----------|---------------------|---------|
| **Raw Videos** | Git LFS (by filename) | `data/raw/videos/` |
| **Extracted Poses** | Git LFS (by clip name) | `data/processed/poses/` |
| **Features** | Git LFS + version metadata | `data/processed/features/` with `_version` field |
| **Train/Val/Test Splits** | Git LFS + timestamp in filename | `data/processed/splits/split_v3_20260129.pkl` |
| **Trained Models** | Git LFS + registry.json | `models/saved/` with metadata in `registry.json` |
| **Benchmarks** | Regular git (small .pkl) | `outputs/benchmarks/` |
| **Metrics** | Regular git (JSON) | `outputs/metrics/` |

### Version Metadata Pattern

```python
# Embed version metadata in all artifacts
feature_data = {
    'features': stat_features,  # Actual feature dict
    'metadata': {
        'version': 'v3',
        'timestamp': '2026-01-29T10:30:00Z',
        'source_video': '15_set1_rally22_ball21_Clear.mp4',
        'feature_count': 577,
        'mediapipe_version': '0.10.9',
        'config': {
            'gaussian_sigma': 1.5,
            'min_confidence': 0.5
        }
    }
}
```

### Model Registry Pattern

```json
// models/registry.json (regular git)
{
  "models": [
    {
      "id": "rf_v3_20260129_001",
      "type": "RandomForest",
      "feature_version": "v3",
      "training_date": "2026-01-29T10:45:00Z",
      "accuracy": 0.87,
      "f1_score": 0.85,
      "file_path": "models/saved/rf_v3_20260129_001.pkl",
      "config": {
        "n_estimators": 200,
        "max_depth": 15
      },
      "status": "experimental"
    },
    {
      "id": "rf_v2_baseline",
      "type": "RandomForest",
      "feature_version": "v2",
      "training_date": "2026-01-20",
      "accuracy": 0.72,
      "f1_score": 0.68,
      "file_path": "models/saved/rf_v2_baseline.pkl",
      "status": "deprecated"
    }
  ],
  "active_production_model": null,  # No ML in production yet
  "active_experimental_model": "rf_v3_20260129_001"
}
```

---

## Scaling Considerations

| Scale | Architecture Adjustments |
|-------|--------------------------|
| **v1.0 (Current)** | Local development, benchmark-based analysis. No ML training. Works fine for demo and academic submission. |
| **v1.1 (Target)** | Add Colab for periodic retraining (weekly/monthly). Git LFS for dataset versioning. Still local inference. Suitable for 1-100 users. |
| **v2.0 (Future)** | Consider cloud deployment (Cloud Run, Lambda). Separate training and inference infrastructure. Feature store for consistency. Suitable for 100-10k users. |
| **v3.0 (Scale)** | Real-time pose extraction in cloud. Model serving via TFServing/Triton. Continuous retraining pipeline with Kubeflow. Suitable for 10k+ users. |

### Scaling Priorities (v1.1 Focus)

1. **First bottleneck**: Git LFS bandwidth limits
   - **Solution**: Selective LFS pulling, only download required files in Colab
   - **When**: Immediately (during Colab setup)

2. **Second bottleneck**: Local inference latency (if using ML models)
   - **Solution**: Model optimization (pruning, quantization), or keep benchmark-based
   - **When**: Only if ML classification accuracy justifies replacement

3. **Not a bottleneck yet**: Training time in Colab
   - **Current**: Training on ~3000 samples takes <30 min with GPU
   - **Action**: No optimization needed for v1.1

---

## Anti-Patterns to Avoid

### Anti-Pattern 1: Storing Large Files in Regular Git

**What people do**: Commit .mp4 videos, .pkl files, trained models directly to git without LFS.

**Why it's wrong**:
- Bloats repository size (clones become slow)
- Exceeds GitHub file size limits (100MB max without LFS)
- History contains large blobs forever (even after deletion)

**Do this instead**:
```bash
# Setup Git LFS BEFORE committing large files
git lfs install
git lfs track "data/raw/videos/**/*.mp4"
git lfs track "models/saved/**/*.pkl"

# Now commits are tracked by LFS automatically
git add data/raw/videos/video.mp4
git commit -m "Add training video"
git push  # Pushes pointer, actual file to LFS storage
```

### Anti-Pattern 2: Notebook-Driven Training Without Scripts

**What people do**: Put all training logic in Jupyter/Colab notebooks with complex execution order dependencies.

**Why it's wrong**:
- Hard to version control (JSON metadata noise)
- Non-reproducible (cell execution order matters)
- Can't automate (requires manual cell execution)
- Can't unit test (logic buried in notebook cells)

**Do this instead**:
```
# Put logic in scripts
scripts/train/train_pipeline.py  # ← All training logic here

# Use notebook as minimal wrapper
notebooks/colab_runner.ipynb:
  Cell 1: !pip install -r requirements.txt
  Cell 2: !python scripts/train/train_pipeline.py --config configs/train_config.yaml
  Cell 3: !git push
```

### Anti-Pattern 3: Breaking Changes to Feature Engineering

**What people do**: Modify `feature_engineering_v2.py` directly, changing feature order or count.

**Why it's wrong**:
- Breaks existing trained models (expect 427 features)
- Breaks benchmark analysis (hardcoded feature names)
- No rollback path if new features don't improve accuracy

**Do this instead**:
```python
# Create NEW version, preserve old
src/data_processing/
  feature_engineering_v2.py  # ← UNCHANGED (production)
  feature_engineering_v3.py  # ← NEW features

# Version-gated extraction
def extract_features(poses, version='v3'):
    if version == 'v2':
        return extract_features_v2(poses)  # Production
    else:
        return extract_features_v3(poses)  # Experimental
```

### Anti-Pattern 4: Replacing Benchmark System Prematurely

**What people do**: Remove benchmark-based analysis as soon as ML model is trained.

**Why it's wrong**:
- ML model may have lower accuracy than benchmarks (v1.0 experience)
- Lose interpretability (benchmarks show specific technique issues)
- No fallback if model fails or confidence is low

**Do this instead**:
```python
# Dual-mode: Benchmark as default, ML as optional enhancement
if config.USE_ML_CLASSIFICATION and model_available:
    prediction = model.predict(features)
    confidence = model.predict_proba(features).max()

    if confidence > 0.85:  # Only use ML if confident
        return ml_feedback(prediction)
    else:
        return benchmark_feedback(features)  # Fallback
else:
    return benchmark_feedback(features)  # Production default
```

### Anti-Pattern 5: No Model Evaluation Before Production

**What people do**: Train model in Colab, immediately deploy to production without validation.

**Why it's wrong**:
- Model may have lower accuracy than existing system
- May fail on edge cases not in training set
- User experience degrades without warning

**Do this instead**:
```python
# Evaluation gate before production
# scripts/evaluate/evaluate_model.py

def evaluate_before_production(model_path, test_data_path):
    model = load_model(model_path)
    test_data = load_test_data(test_data_path)

    metrics = calculate_metrics(model, test_data)

    print(f"Accuracy: {metrics['accuracy']:.2%}")
    print(f"F1 Score: {metrics['f1']:.2%}")

    # Production criteria
    ACCURACY_THRESHOLD = 0.85
    F1_THRESHOLD = 0.80

    if metrics['accuracy'] >= ACCURACY_THRESHOLD and metrics['f1'] >= F1_THRESHOLD:
        print("✓ Model meets production criteria")
        return True
    else:
        print("✗ Model below threshold, continue using benchmarks")
        return False
```

---

## Build Order for v1.1 Implementation

Recommended implementation sequence to minimize breaking changes:

### Phase 1: Git LFS Setup (Week 1)
1. **Initialize Git LFS** (`git lfs install`)
2. **Configure tracking** (`.gitattributes` for videos, poses, features, models)
3. **Migrate existing large files** (`git lfs migrate import`)
4. **Test selective pulling** (validate bandwidth optimization)

### Phase 2: Feature Engineering V3 (Week 2)
1. **Create `feature_engineering_v3.py`** (preserve v2)
2. **Research coach-informed features** (literature review, video analysis)
3. **Implement new features** (racket angle, phase segmentation, hip rotation)
4. **Validate backward compatibility** (v2 still works)
5. **Regenerate features** (process dataset with v3)

### Phase 3: Colab Training Pipeline (Week 3)
1. **Create script structure** (`scripts/train/`, `scripts/evaluate/`)
2. **Implement `train_pipeline.py`** (orchestrator)
3. **Create configs** (`configs/train_config.yaml`)
4. **Setup Colab runtime template** (`.colab/runtime_template.yaml`)
5. **Test end-to-end** (local → git → Colab → git → local)

### Phase 4: Model Training & Evaluation (Week 4)
1. **Train baseline models** (Random Forest, SVM with v3 features)
2. **Evaluate accuracy** (compare to v2 baseline)
3. **Feature importance analysis** (identify discriminative features)
4. **Hyperparameter tuning** (if accuracy promising)
5. **Document results** (`outputs/metrics/training_run.json`)

### Phase 5: Production Integration (Week 5)
1. **Create `model_loader.py`** (production model loading)
2. **Update `streamlit_app.py`** (add optional ML mode)
3. **Implement dual-mode inference** (benchmark + ML with fallback)
4. **A/B testing** (validate ML improves user experience)
5. **Decision: Deploy or iterate** (based on accuracy threshold)

### Dependency Graph

```
Phase 1 (Git LFS)
    ↓
Phase 2 (Features V3) ─┐
    ↓                  │
Phase 3 (Colab) ←──────┘
    ↓
Phase 4 (Training)
    ↓
Phase 5 (Integration)
```

**Critical path**: Git LFS → Features V3 → Colab → Training → Integration

---

## Sources

**Git LFS & ML Data Versioning**:
- [Git LFS and DVC: The Ultimate Guide to Managing Large Artifacts in MLOps](https://medium.com/@pablojusue/git-lfs-and-dvc-the-ultimate-guide-to-managing-large-artifacts-in-mlops-c1c926e6c5f4)
- [Managing Data and Model Artifacts with Git LFS](https://www.almabetter.com/bytes/tutorials/mlops/managing-data-and-model-artifacts-with-git-lfs)
- [Git Large File Storage for ML/DL projects](https://medium.com/@grof.attila9/git-large-file-storage-for-ml-dl-projects-df5e85995775)

**Google Colab Enterprise & Git Integration**:
- [Introduction to Colab Enterprise | Google Cloud Documentation](https://docs.cloud.google.com/colab/docs/introduction)
- [Combine GitHub and Google Colab for Collaborative Development](https://tilburgsciencehub.com/topics/automation/replicability/cloud-computing/colab-github/)
- [Using git & GitHub on Google Colaboratory](https://medium.com/geekculture/using-git-github-on-google-colaboratory-7ef3b76fe61b)

**Colab Terminal & Script Execution**:
- [How Can I Run Terminal in Google Colab?](https://www.analyticsvidhya.com/blog/2025/02/run-terminal-in-google-colab/)
- [How to Run a Python Script in a py File from a Google Colab Notebook](https://saturncloud.io/blog/how-to-run-a-python-script-in-a-py-file-from-a-google-colab-notebook/)

**ML Retraining Pipelines & Feature Stores**:
- [MLOps: Continuous delivery and automation pipelines in machine learning](https://docs.cloud.google.com/architecture/mlops-continuous-delivery-and-automation-pipelines-in-machine-learning)
- [Model Retraining [2026]: Why & How to Retrain ML Models?](https://research.aimultiple.com/model-retraining/)
- [Feature Store: The Definitive Guide - MLOps Dictionary](https://www.hopsworks.ai/dictionary/feature-store)
- [How to Build Machine Learning Systems With a Feature Store](https://neptune.ai/blog/building-ml-systems-with-feature-store)

---

*Architecture research for: Badminton Coaching System v1.1 (Colab Enterprise Integration)*
*Researched: 2026-01-29*
*Confidence: MEDIUM (Web search + existing codebase analysis, verified patterns)*
