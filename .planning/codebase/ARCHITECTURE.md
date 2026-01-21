# System Architecture

**Last Updated**: 2026-01-21
**Codebase**: AI Badminton Coaching System v2.0

---

## Architectural Pattern

**Pipeline Architecture** (Sequential Data Flow)

```
Video Input → Pose Estimation → Feature Engineering → Analysis → Visualization/Feedback
```

This is a classic **data processing pipeline** where each stage transforms data and passes it to the next stage. No complex state management, event-driven patterns, or distributed systems.

---

## High-Level System Flow

```
┌─────────────────┐
│  Video Upload   │ (.mp4, .avi, .mov, .mkv)
└────────┬────────┘
         │
         ▼
┌─────────────────────────────────────────────┐
│  STAGE 1: Pose Estimation                   │
│  Component: PoseExtractor                   │
│  Input: Video frames (cv2.VideoCapture)     │
│  Process:                                    │
│    - MediaPipe Pose (33 keypoints/person)   │
│    - Multi-player detection                 │
│    - Executor identification (arm elevation)│
│    - Temporal interpolation (missing frames)│
│  Output: pose_data dict                     │
│    - poses: np.array (frames × 99)          │
│    - quality metrics                        │
└────────┬────────────────────────────────────┘
         │
         ▼
┌─────────────────────────────────────────────┐
│  STAGE 2: Feature Engineering               │
│  Component: feature_engineering_v2          │
│  Input: pose_data['poses']                  │
│  Process:                                    │
│    - Per-frame spatial features (60)        │
│    - Temporal derivatives (velocity, accel) │
│    - Statistical aggregations (427 total)   │
│  Output: stat_features dict                 │
│    - 427 statistical features               │
│    - Temporal metrics (max_velocity, etc)   │
└────────┬────────────────────────────────────┘
         │
         ▼
┌─────────────────────────────────────────────┐
│  STAGE 3: Technique Analysis                │
│  Components:                                 │
│    - TechniqueBenchmarks (professional ranges)│
│    - CoachingFeedback (rule-based analysis) │
│  Input: stat_features, stroke_type          │
│  Process:                                    │
│    - Compare features vs benchmarks         │
│    - Generate FeedbackItem objects          │
│    - Calculate overall score (0-100)        │
│  Output: feedback_items (List[FeedbackItem])│
└────────┬────────────────────────────────────┘
         │
         ▼
┌─────────────────────────────────────────────┐
│  STAGE 4: Visualization & Output            │
│  Component: TechniqueVisualizer             │
│  Input: stat_features, feedback_items       │
│  Process:                                    │
│    - Radar chart (technique profile)        │
│    - Bar chart (metric breakdown)           │
│    - Score gauge (overall score)            │
│    - Comprehensive report (multi-panel)     │
│    - Text feedback                          │
│  Output: PNG images + feedback.txt          │
└─────────────────────────────────────────────┘
```

---

## Component Layers

### Layer 1: Entry Points

**Command-Line Interface**
- **Location**: [`analyze_video.py`](analyze_video.py)
- **Purpose**: Main script for video analysis
- **Usage**: `python analyze_video.py video.mp4 Clear`
- **Flow**: Orchestrates full pipeline (stages 1-4)

**Web Interface (Streamlit)**
- **Location**: [`src/deployment/streamlit_app.py`](src/deployment/streamlit_app.py)
- **Purpose**: Interactive web UI
- **Usage**: `streamlit run src/deployment/streamlit_app.py`
- **Flow**: Same pipeline as CLI, different presentation

**Web Interface (Gradio)**
- **Location**: [`src/deployment/coaching_app.py`](src/deployment/coaching_app.py)
- **Purpose**: Alternative web UI
- **Status**: Less actively maintained

**Diagnostic Tool**
- **Location**: [`diagnose.py`](diagnose.py)
- **Purpose**: Environment verification
- **Flow**: Checks dependencies, MediaPipe, file access

### Layer 2: Data Processing

**Pose Extraction Module**
- **Location**: [`src/data_processing/extract_poses.py`](src/data_processing/extract_poses.py)
- **Key Class**: `PoseExtractor`
- **Responsibilities**:
  - Load video with `cv2.VideoCapture`
  - Detect poses with MediaPipe (model complexity 2)
  - Identify stroke executor (multi-player scenes)
  - Interpolate missing frames
  - Return structured `pose_data` dict

**Feature Engineering Module**
- **Location**: [`src/data_processing/feature_engineering_v2.py`](src/data_processing/feature_engineering_v2.py)
- **Key Functions**:
  - `extract_frame_features(pose)` → 60 spatial features
  - `extract_temporal_features(poses)` → velocity, acceleration, jerk
  - `extract_statistical_summary(frames, names)` → 427 aggregated features
- **Responsibilities**:
  - Transform 33 keypoints → biomechanical features
  - Apply Gaussian smoothing
  - Calculate derivatives (velocity, acceleration)
  - Aggregate statistics (min, max, mean, std, percentiles)

**Data Splitting Module**
- **Location**: [`src/data_processing/data_split.py`](src/data_processing/data_split.py)
- **Purpose**: Train/val/test split for model training
- **Status**: Present but not used in current v2.0 (benchmark-based, no model training)

### Layer 3: Coaching Logic

**Technique Benchmarks**
- **Location**: [`src/coaching/technique_benchmarks.py`](src/coaching/technique_benchmarks.py)
- **Key Class**: `TechniqueBenchmarks`
- **Data**: Professional ranges (25th-75th percentile)
  - `CLEAR_BENCHMARKS` dict (forehand Clear strokes)
  - `SMASH_BENCHMARKS` dict (forehand Smash strokes)
- **Metrics Tracked**:
  - `max_velocity` (wrist speed)
  - `elbow_angle` (joint angle at contact)
  - `forearm_angle` (vertical orientation)
  - `contact_point` (wrist height relative to head)
  - `shoulder_angle`, `trunk_lean`
- **Methods**:
  - `get_benchmarks(stroke_type)` → dict
  - `format_value(value, metric)` → human-readable string

**Feedback Generator**
- **Location**: [`src/coaching/feedback_generator.py`](src/coaching/feedback_generator.py)
- **Key Classes**:
  - `FeedbackItem`: Single piece of feedback
  - `CoachingFeedback`: Analysis orchestrator
- **Analysis Functions** (all in `CoachingFeedback`):
  - `_analyze_arm_extension()` → racket-hand distance
  - `_analyze_velocity()` → wrist movement speed
  - `_analyze_elbow_angle()` → joint angle
  - `_analyze_posture()` → trunk lean
  - `_analyze_timing()` → peak acceleration timing
  - `_analyze_contact_point()` → wrist height
- **Severity Classification**:
  - `critical`: >2 std dev from benchmark
  - `major`: 1-2 std dev from benchmark
  - `minor`: 0.5-1 std dev from benchmark
  - `good`: within professional range
- **Output**: List of `FeedbackItem` objects, sorted by severity

**LLM Enhancer** (Optional)
- **Location**: [`src/coaching/llm_enhancer.py`](src/coaching/llm_enhancer.py)
- **Key Class**: `LLMCoachingEnhancer`
- **Purpose**: Natural language generation via OpenAI GPT-4o-mini
- **Status**: Experimental, not used in main workflows
- **Flow**: Rule-based feedback → LLM prompt → conversational text

### Layer 4: Visualization

**Visualization Module**
- **Location**: [`src/coaching/visualizations.py`](src/coaching/visualizations.py)
- **Key Class**: `TechniqueVisualizer`
- **Visualization Types**:
  - **Radar Chart**: `create_radar_chart(features, stroke_type)`
    - Normalized technique profile (8 metrics)
    - Compares user vs professional ranges
  - **Bar Chart**: `create_metrics_bar_chart(feedback_items, stroke_type)`
    - Metric-by-metric severity breakdown
    - Color-coded by severity
  - **Score Gauge**: `create_score_gauge(score, stroke_type)`
    - Overall score 0-100
    - Speedometer-style visualization
  - **Comprehensive Report**: `create_comprehensive_report(...)`
    - Multi-panel view (4 subplots)
    - Combines all visualizations
- **Technology**: Matplotlib, NumPy
- **Output**: PNG images saved to `outputs/video_analysis/<video_name>/`

### Layer 5: Models (Unused in v2.0)

**Baseline Model**
- **Location**: [`src/models/baseline_model.py`](src/models/baseline_model.py)
- **Status**: Present but not used in v2.0
- **Purpose**: Clear vs Smash classification
- **Note**: System pivoted from ML classification to benchmark-based analysis

**LSTM Model**
- **Location**: [`src/models/lstm_model.py`](src/models/lstm_model.py)
- **Status**: Present but not used in v2.0
- **Purpose**: Temporal sequence classification
- **Note**: Would enable advanced pattern recognition

---

## Data Flow (Detailed)

### Input: Video File
```
user_video.mp4
  ↓ cv2.VideoCapture
Frames (H × W × 3 RGB images)
```

### Stage 1: Pose Extraction
```
Frames
  ↓ MediaPipe Pose
Raw Keypoints (33 landmarks × (x, y, z, visibility))
  ↓ Multi-player detection (arm elevation heuristic)
Executor Keypoints (filtered)
  ↓ Temporal interpolation (fill missing frames)
Pose Sequence (T × 99 array)  # T=frames, 99=33*3 coords
  ↓
pose_data = {
  'poses': np.array (T × 99),
  'num_frames': int,
  'quality': {'valid_percentage': float}
}
```

### Stage 2: Feature Engineering
```
pose_data['poses'] (T × 99)
  ↓ extract_frame_features() per frame
Per-frame Features (T × 60)
  ↓ Gaussian smoothing
Smoothed Trajectories
  ↓ calculate_velocity(), calculate_acceleration()
Temporal Features (velocity, acceleration, jerk)
  ↓ extract_statistical_summary()
Statistical Features (427 aggregated metrics)
  ↓
stat_features = {
  'max_velocity': float,
  'elbow_angle_mean': float,
  'forearm_angle_std': float,
  ...  # 427 total
}
```

### Stage 3: Technique Analysis
```
stat_features + stroke_type
  ↓ TechniqueBenchmarks.get_benchmarks(stroke_type)
Professional Ranges (min, target, max)
  ↓ CoachingFeedback.analyze_technique()
[
  _analyze_arm_extension() → FeedbackItem,
  _analyze_velocity() → FeedbackItem,
  _analyze_elbow_angle() → FeedbackItem,
  _analyze_posture() → FeedbackItem,
  _analyze_timing() → FeedbackItem,
  _analyze_contact_point() → FeedbackItem
]
  ↓ Sort by severity
feedback_items = List[FeedbackItem]
overall_score = int (0-100)
```

### Stage 4: Visualization
```
stat_features + feedback_items + overall_score
  ↓ TechniqueVisualizer
[
  radar_chart_<stroke>.png,
  metrics_bar_chart_<stroke>.png,
  score_gauge_<stroke>.png,
  comprehensive_report_<stroke>.png
]
+
feedback.txt
```

---

## Key Abstractions

### 1. `pose_data` (Pose Extraction Output)
```python
{
  'poses': np.ndarray,        # (T, 99) - pose sequence
  'num_frames': int,           # Total frames
  'quality': {                 # Quality metrics
    'valid_percentage': float,
    'total_frames': int,
    'valid_frames': int
  }
}
```

### 2. `stat_features` (Feature Engineering Output)
```python
{
  # Velocity features
  'max_velocity': float,
  'velocity_mean': float,
  'velocity_std': float,

  # Angle features
  'elbow_angle_mean': float,
  'elbow_angle_min': float,
  'forearm_angle_max': float,

  # ... 427 total features
}
```

### 3. `FeedbackItem` (Coaching Output Unit)
```python
FeedbackItem(
  metric='velocity',                # Which metric
  severity='major',                 # critical/major/minor/good
  message='Velocity below target',  # Human-readable
  current_value=65.3,               # User's value
  target_range=(68.7, 109.2),       # Professional range
  drill='Resistance band training', # Practice recommendation
  impact='Increase power 15-20%'    # Expected improvement
)
```

### 4. Benchmarks (Professional Ranges)
```python
CLEAR_BENCHMARKS = {
  'max_velocity': 48.15,         # Lower bound (25th percentile)
  'max_velocity_target': 75.78,  # Median (50th percentile)
  'max_velocity_upper': 92.33,   # Upper bound (75th percentile)
  # ... more metrics
}
```

---

## Coupling & Dependencies

**Tight Coupling**:
- `CoachingFeedback` → `TechniqueBenchmarks` (direct dict access)
- `analyze_video.py` → All modules (orchestrator pattern)
- `streamlit_app.py` → All modules (same as CLI)

**Loose Coupling**:
- Stages communicate via dictionaries (not custom objects)
- No shared mutable state between stages
- Each module can be tested independently

**Dependency Direction**:
```
Visualization ←─ Coaching ←─ Features ←─ Poses ←─ Video
```
Clean one-way dependency (no circular dependencies).

---

## State Management

**No persistent state**:
- Each analysis is independent
- No session management
- No user profiles
- No history tracking

**Stateless processing**:
- All data flows through function parameters/returns
- No global variables (except constants)
- No caching or memoization

---

## Concurrency Model

**Single-threaded, synchronous**:
- No async/await
- No multiprocessing
- No threading
- Each video processed sequentially

**Parallelization opportunities** (not implemented):
- Batch processing multiple videos
- Parallel feature extraction (per-frame independence)

---

## Error Handling Strategy

**Try-catch at stage boundaries**:
- [`analyze_video.py:119-139`](analyze_video.py#L119-L139): Pose extraction
- [`analyze_video.py:143-175`](analyze_video.py#L143-L175): Feature engineering
- [`analyze_video.py:179-210`](analyze_video.py#L179-L210): Coaching analysis

**Fallback strategies**:
- LLM enhancement → Template-based feedback ([`llm_enhancer.py:89-92`](src/coaching/llm_enhancer.py#L89-L92))
- Stroke auto-detection → User must specify manually
- Missing frames → Temporal interpolation

**No retry logic**:
- Video processing failures are terminal
- User must fix input and re-run

---

## Configuration Management

**Hardcoded constants**:
- MediaPipe settings in [`extract_poses.py:59-77`](src/data_processing/extract_poses.py#L59-L77)
- Benchmark ranges in [`technique_benchmarks.py`](src/coaching/technique_benchmarks.py)
- Severity thresholds in [`feedback_generator.py`](src/coaching/feedback_generator.py)

**No config files**:
- No YAML, JSON, INI files for settings
- No environment-based configuration (dev/prod)

**Recommendation**: Extract magic numbers to a `config.py` module

---

## Testing Architecture

**Status**: No automated tests

**Testing approach**: Manual
- [`diagnose.py`](diagnose.py): Environment verification
- Debug scripts: [`baseline_model_debug.py`](baseline_model_debug.py), [`debug_splits_colab.py`](debug_splits_colab.py)
- Sample videos: `data/processed/clips/` for manual validation

**Testable components** (but no tests exist):
- Feature engineering (pure functions, deterministic)
- Benchmark comparisons (pure functions)
- Feedback severity classification (rule-based logic)

---

## Build Order for New Features

If extending this system, recommended build order:

1. **Data Layer**: Add new MediaPipe keypoints or external data sources
2. **Feature Layer**: Derive new biomechanical features from poses
3. **Benchmark Layer**: Define professional ranges for new metrics
4. **Analysis Layer**: Create new `_analyze_<metric>()` functions
5. **Visualization Layer**: Add charts for new metrics
6. **UI Layer**: Expose new features in Streamlit/Gradio

---

## Architectural Strengths

1. **Simple, linear pipeline**: Easy to understand and debug
2. **Clean stage separation**: Each stage has clear input/output
3. **No over-engineering**: No unnecessary abstractions
4. **Deterministic**: Same input always produces same output
5. **Extensible**: Easy to add new metrics or analysis functions

---

## Architectural Weaknesses

1. **No persistence**: Every analysis starts from scratch
2. **No caching**: Re-processing same video repeats all stages
3. **Synchronous only**: Can't process multiple videos concurrently
4. **Tight coupling in entry points**: CLI and Streamlit duplicate orchestration logic
5. **Hardcoded configuration**: Magic numbers scattered across modules
6. **No testing infrastructure**: Relies on manual validation
7. **No error recovery**: Failures require full restart

---

## Future Architectural Improvements

1. **Add service layer**: Extract orchestration logic from entry points
2. **Implement caching**: Store extracted poses and features
3. **Add database**: Track analysis history, user progress over time
4. **Enable batch processing**: Process multiple videos in parallel
5. **Configuration system**: Centralize constants in config file
6. **Add test suite**: Unit tests for feature engineering, analysis logic
7. **API layer**: RESTful API for programmatic access (beyond UI)
