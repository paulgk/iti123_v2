# Directory Structure

**Last Updated**: 2026-01-21
**Codebase**: AI Badminton Coaching System v2.0

---

## Overview

```
iti123_v2/
├── src/               # Core application code (modular, organized by function)
├── data/              # Dataset, annotations, processed artifacts
├── models/            # Trained models (unused in v2.0)
├── outputs/           # Analysis results, visualizations, reports
├── docs/              # Documentation (minimal)
├── experiments/       # MLflow experiments (unused in v2.0)
├── *.py               # Entry point scripts (root level)
├── requirements.txt   # Python dependencies
├── README.md          # Project documentation
└── .gitignore         # Git exclusions
```

---

## Root Level Files

### Entry Point Scripts
- [`analyze_video.py`](analyze_video.py) - **Main CLI** for video analysis
- [`diagnose.py`](diagnose.py) - Environment diagnostic tool

### Debug & Analysis Scripts (Development)
- [`analyze_stroke_types.py`](analyze_stroke_types.py) - Stroke type analysis
- [`analyze_video.py`](analyze_video.py) - Main analysis pipeline
- [`baseline_model_debug.py`](baseline_model_debug.py) - Model debugging
- [`baseline_model_fixed.py`](baseline_model_fixed.py) - Fixed baseline model
- [`create_forehand_features_pkl.py`](create_forehand_features_pkl.py) - Feature generation
- [`debug_splits_colab.py`](debug_splits_colab.py) - Colab-specific debugging
- [`filter_backhand_and_regenerate.py`](filter_backhand_and_regenerate.py) - Data filtering
- [`fix_baseline_paths.py`](fix_baseline_paths.py) - Path correction utility
- [`regenerate_features.py`](regenerate_features.py) - Feature regeneration

**Note**: Many scripts at root level - could be organized into `scripts/` or `tools/` directory.

### Configuration Files
- [`requirements.txt`](requirements.txt) - Python dependencies (TensorFlow, MediaPipe, etc.)
- [`.gitignore`](.gitignore) - Excludes large files, data, models

### Documentation
- [`README.md`](README.md) - **Primary documentation** (comprehensive, 314 lines)

---

## `/src/` - Application Source Code

**Core modules** organized by functional domain:

```
src/
├── __init__.py
├── data_processing/     # Video → Features
├── coaching/            # Analysis → Feedback
├── deployment/          # Web interfaces
├── models/              # ML models (unused in v2.0)
├── evaluation/          # Model evaluation (unused)
└── analysis/            # Exploratory analysis
```

### `/src/data_processing/` - Data Pipeline

```
src/data_processing/
├── __init__.py
├── extract_poses.py              # MediaPipe pose extraction (PoseExtractor class)
├── feature_engineering_v2.py     # Biomechanical feature engineering
└── data_split.py                 # Train/val/test splitting (unused in v2.0)
```

**Key responsibilities**:
- Video loading (OpenCV)
- Pose keypoint extraction (MediaPipe)
- Multi-player detection
- Feature derivation (spatial, temporal, statistical)

### `/src/coaching/` - Analysis & Feedback

```
src/coaching/
├── __init__.py
├── technique_benchmarks.py        # Professional ranges (CLEAR_BENCHMARKS, SMASH_BENCHMARKS)
├── feedback_generator.py          # CoachingFeedback, FeedbackItem classes
├── visualizations.py              # TechniqueVisualizer (radar, bar, gauge, report)
├── llm_enhancer.py                # LLMCoachingEnhancer (OpenAI integration - optional)
├── derive_benchmarks.py           # Benchmark derivation from dataset
└── technique_benchmarks_backup.py # Backup of old benchmarks
```

**Key responsibilities**:
- Benchmark comparison
- Feedback generation (severity classification)
- Practice drill recommendations
- Visualization creation
- Optional LLM enhancement

### `/src/deployment/` - User Interfaces

```
src/deployment/
├── __init__.py
├── streamlit_app.py    # **Primary web interface** (Streamlit)
└── coaching_app.py     # Alternative web interface (Gradio)
```

**Deployment options**:
- Streamlit: `streamlit run src/deployment/streamlit_app.py` (production)
- Gradio: `python src/deployment/coaching_app.py` (less maintained)

### `/src/models/` - ML Models (Unused in v2.0)

```
src/models/
├── __init__.py
├── baseline_model.py   # Baseline classifier (Clear vs Smash)
└── lstm_model.py       # LSTM temporal classifier
```

**Status**: Present but not used. System pivoted to benchmark-based analysis.

### `/src/evaluation/` - Model Evaluation (Unused)

```
src/evaluation/
└── __init__.py
```

**Status**: Empty directory, prepared for future model evaluation workflows.

### `/src/analysis/` - Exploratory Analysis

```
src/analysis/
└── analyze_wrist_features.py   # Wrist feature analysis script
```

**Purpose**: Ad-hoc analysis during development.

---

## `/data/` - Dataset & Processed Artifacts

```
data/
├── annotations/         # ShuttleSet CSV annotations (45 matches)
├── processed/           # Generated data from pipeline
│   ├── clips/          # Video clips (4,983 .mp4 files)
│   ├── poses/          # Extracted pose sequences (.pkl)
│   ├── features/       # Engineered features (.pkl)
│   ├── splits/         # Train/val/test splits (.pkl)
│   ├── poses_*/        # Alternative classification experiments
│   ├── features_*/     # Alternative feature sets
│   └── splits_*/       # Alternative data splits
└── raw_videos/          # Original full-length match videos
```

### `/data/annotations/` - Match Annotations (45 directories)

**Structure**: One directory per match
```
data/annotations/<Match_Name>/
├── match_info.csv       # Match metadata
├── rallies.csv          # Rally-level annotations
└── strokes.csv          # Stroke-level annotations
```

**Examples**:
- `Kento_MOMOTA_CHOU_Tien_Chen_Fuzhou_Open_2018_Finals/`
- `Viktor_Axelsen_Hans-Kristian_Solberg_VIittinghus_TOYOTA_THAILAND_OPEN_2021_Finals/`
- `Carolina_Marin_Pornpawee_Chochuwong_HSBC_BWF_WORLD_TOUR_FINALS_2020_SemiFinals/`

**Total**: 45 professional singles matches (male & female)

### `/data/processed/clips/` - Video Clips

**Files**:
- ~4,983 `.mp4` video clips
- `clips_metadata.csv` - Metadata (stroke type, match, rally, ball)

**Naming convention**: `{match_id}_set{set}_rally{rally}_ball{ball}_{stroke_type}.mp4`
- Example: `01_set1_rally1_ball2_Clear.mp4`

**Size**: Videos excluded from git via `.gitignore` (large files)

### `/data/processed/poses/` - Extracted Poses

**Files**: `.pkl` files (one per clip)
- Naming: Matches clip naming (e.g., `01_set1_rally1_ball2_Clear.pkl`)
- Content: Pickled `pose_data` dict
  ```python
  {
    'poses': np.ndarray (frames × 99),  # 33 keypoints × 3 coords
    'num_frames': int,
    'quality': {'valid_percentage': float, ...}
  }
  ```

**Size**: ~500 MB total (excluded from git)

### `/data/processed/features/` - Engineered Features

**Files**: `.pkl` files (one per clip)
- Naming: Matches clip naming
- Content: Pickled dict of 427 statistical features
  ```python
  {
    'max_velocity': float,
    'elbow_angle_mean': float,
    'forearm_angle_std': float,
    # ... 427 total
  }
  ```

**Size**: ~50 MB total (excluded from git)

### `/data/processed/splits/` - Train/Val/Test Splits

**Files**:
- `train_data.pkl` (was 105 MB - **removed from git history**)
- `val_data.pkl`
- `test_data.pkl`

**Content**: Pickled dicts with features + labels for ML training
- Unused in v2.0 (benchmark-based system, no model training)

**Note**: Historical ML experiments, not part of current workflow.

### Alternative Experiment Directories

```
data/processed/
├── poses_drop_smash/      # Drop vs Smash classification experiment
├── features_drop_smash/
├── splits_drop_smash/
├── poses_lift_smash/      # Lift vs Smash classification experiment
├── features_lift_smash/
├── splits_lift_smash/
├── poses_multiclass/      # Multiclass stroke classification experiment
├── features_multiclass/
└── splits_multiclass/
```

**Status**: Legacy experiments, not used in v2.0.

---

## `/models/` - Trained ML Models

```
models/
├── saved/              # Checkpoints from training
├── drop_smash/         # Drop vs Smash models
│   ├── baseline/
│   └── deep_learning/
├── lift_smash/         # Lift vs Smash models
│   ├── baseline/
│   └── deep_learning/
└── multiclass/         # Multiclass models
    ├── baseline/
    └── deep_learning/
```

**Status**: Unused in v2.0 (benchmark-based system, no ML inference)

**Contents**: `.h5` and `.pth` model checkpoint files (excluded from git)

---

## `/outputs/` - Analysis Results

```
outputs/
├── video_analysis/         # CLI output (analyze_video.py)
│   ├── <video_name>/      # One directory per analyzed video
│   │   ├── feedback.txt
│   │   ├── comprehensive_report_<stroke>.png
│   │   ├── radar_chart_<stroke>.png
│   │   ├── metrics_bar_chart_<stroke>.png
│   │   └── score_gauge_<stroke>.png
│   └── ...
├── simple_analysis/        # Simplified analysis outputs
├── test_clear/             # Test outputs for Clear strokes
├── test_smash/             # Test outputs for Smash strokes
├── visualizations/         # General visualizations
├── plots/                  # Training plots (unused in v2.0)
├── reports/                # Pose extraction reports
├── reports_drop_smash/     # Drop vs Smash experiment reports
├── reports_lift_smash/     # Lift vs Smash experiment reports
└── reports_multiclass/     # Multiclass experiment reports
```

### `/outputs/video_analysis/<video_name>/` - CLI Output Structure

**Generated by**: [`analyze_video.py`](analyze_video.py)

**Files**:
- `feedback.txt` - Text coaching feedback (severity breakdown, drills)
- `comprehensive_report_<stroke>.png` - Multi-panel visualization (4 subplots)
- `radar_chart_<stroke>.png` - 8-metric technique profile
- `metrics_bar_chart_<stroke>.png` - Metric-by-metric severity bar chart
- `score_gauge_<stroke>.png` - Overall score (0-100) speedometer

**Example**: `outputs/video_analysis/01_set1_rally1_ball2_Clear/`

---

## `/docs/` - Documentation

```
docs/
└── archive/   # Archived documentation
```

**Status**: Minimal. Primary documentation in [`README.md`](README.md).

---

## `/experiments/` - MLflow Experiments

```
experiments/   # Empty or minimal usage
```

**Status**: Prepared for MLflow tracking, but not actively used in v2.0.

---

## Key Locations Summary

| Purpose | Location |
|---------|----------|
| **Main CLI** | [`analyze_video.py`](analyze_video.py) |
| **Web UI (Primary)** | [`src/deployment/streamlit_app.py`](src/deployment/streamlit_app.py) |
| **Pose Extraction** | [`src/data_processing/extract_poses.py`](src/data_processing/extract_poses.py) |
| **Feature Engineering** | [`src/data_processing/feature_engineering_v2.py`](src/data_processing/feature_engineering_v2.py) |
| **Benchmarks** | [`src/coaching/technique_benchmarks.py`](src/coaching/technique_benchmarks.py) |
| **Feedback Logic** | [`src/coaching/feedback_generator.py`](src/coaching/feedback_generator.py) |
| **Visualizations** | [`src/coaching/visualizations.py`](src/coaching/visualizations.py) |
| **Dataset Clips** | `data/processed/clips/` |
| **Annotations** | `data/annotations/` |
| **Extracted Poses** | `data/processed/poses/` |
| **Engineered Features** | `data/processed/features/` |
| **Analysis Output** | `outputs/video_analysis/<video_name>/` |
| **Dependencies** | [`requirements.txt`](requirements.txt) |
| **Documentation** | [`README.md`](README.md) |

---

## Naming Conventions

### Files
- **Snake_case**: Python modules (`extract_poses.py`, `feature_engineering_v2.py`)
- **Lowercase scripts**: Root-level utilities (`diagnose.py`, `analyze_video.py`)
- **Version suffixes**: `_v2` indicates second iteration (e.g., `feature_engineering_v2.py`)
- **Backup suffix**: `_backup` for old versions (e.g., `technique_benchmarks_backup.py`)

### Directories
- **Lowercase with underscores**: `data_processing`, `video_analysis`
- **Plural for collections**: `poses`, `features`, `models`, `outputs`
- **Suffix for variants**: `_drop_smash`, `_multiclass` (experiment variants)

### Clips & Data Files
- **Match ID prefix**: `01_`, `02_`, etc. (match identifier)
- **Structured naming**: `{id}_set{n}_rally{n}_ball{n}_{StrokeType}.mp4`
- **Stroke types**: `Clear`, `Smash`, `Drop`, `Lift` (capitalized)

---

## Git Exclusions (`.gitignore`)

**Large files excluded**:
- `raw_videos/*.mp4` - Original match videos (GB-scale)
- `data/processed/clips/*.mp4` - Extracted clips (~5 GB)
- `data/processed/poses/*.pkl` - Pose sequences (~500 MB)
- `data/processed/features/*.pkl` - Feature files (~50 MB)
- `data/processed/splits/*.pkl` - Train/val/test splits (one file was 105 MB, removed from history)
- `data/annotations/*` - ShuttleSet annotations (CSV files)
- `models/*.h5`, `models/*.pth` - Trained model weights
- `outputs/plots/*.png` - Visualizations (except `final_*.png`)
- `mlruns/`, `experiments/` - MLflow tracking data

**Python standard exclusions**:
- `__pycache__/`, `*.pyc`, `*.egg-info/`
- `venv/`, `ENV/`, `env/`
- `.ipynb_checkpoints/`

**IDE & OS**:
- `.vscode/`, `.idea/`
- `.DS_Store`, `Thumbs.db`

---

## Observations & Recommendations

### Strengths
1. **Clean `/src/` structure**: Organized by domain (data, coaching, deployment)
2. **Comprehensive README**: Well-documented usage, features, troubleshooting
3. **Systematic data organization**: Separate directories for poses, features, splits

### Weaknesses
1. **Root-level clutter**: 10+ debug scripts at root (could move to `scripts/` or `tools/`)
2. **Duplicate experiment dirs**: `poses_*`, `features_*`, `splits_*` variants (could archive)
3. **No `tests/` directory**: No dedicated testing infrastructure
4. **Large binary files in history**: `train_data.pkl` (105 MB) was committed, later removed
5. **Mixed naming**: Some files use `CamelCase` (match names), others `snake_case`

### Recommendations
1. **Create `scripts/` directory**: Move debug/utility scripts from root
2. **Archive experiments**: Move `*_drop_smash`, `*_multiclass` dirs to `experiments/archive/`
3. **Add `tests/` directory**: Prepare for future test infrastructure
4. **Consolidate docs**: Move archived docs to `docs/archive/`, keep `docs/` for active content
5. **Add `config/` directory**: Centralize configuration (benchmark thresholds, MediaPipe settings)
6. **Create `notebooks/` directory**: If Jupyter notebooks are used for exploration
