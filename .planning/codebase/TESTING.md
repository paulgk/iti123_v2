# Testing Practices

**Last Updated**: 2026-01-21
**Codebase**: AI Badminton Coaching System v2.0

---

## Current State

**Automated Testing**: None

The codebase has **no automated test suite**. No unit tests, integration tests, or end-to-end tests.

---

## Testing Infrastructure

### Testing Frameworks

**Available but not used**:
```txt
# requirements.txt (commented out)
# pytest>=7.4.0
# pytest-cov>=4.1.0
```

**Status**: Testing dependencies are commented out, suggesting they were considered but never implemented.

### Test Files

**Location**: None

No `tests/` or `test_/` directory exists. No `test_*.py` or `*_test.py` files found in the codebase (grep search returned no test files).

---

## Current Testing Approach

### Manual Testing

**Primary method**: Run scripts manually and observe output

**Entry points for manual testing**:
1. **`diagnose.py`** - Environment verification
2. **`analyze_video.py`** - Full pipeline test
3. **Sample clips** - Professional strokes in `data/processed/clips/`
4. **Debug scripts** - Ad-hoc validation scripts at root level

### Diagnostic Tool

**File**: [`diagnose.py`](diagnose.py)

**Purpose**: Validates environment setup (not code correctness)

**Checks performed**:
- Python version (3.8-3.10 recommended)
- MediaPipe installation and version (0.10.9 required)
- Protobuf version (3.x required, critical!)
- OpenCV installation
- NumPy installation
- TensorFlow installation
- Matplotlib installation
- File system access (clips, poses, features directories)
- MediaPipe mutex test (known issue check)

**Usage**:
```bash
python diagnose.py
```

**Output format**:
```
Python: 3.10.x
  ✓ Compatible version

MediaPipe: 0.10.9
  ✓ Installed

Protobuf: 3.20.3
  ✓ Compatible version

[... more checks ...]

Overall: All checks passed ✓
```

**Not a test suite**: Checks environment, not code behavior. No assertions on function output.

### Sample Data Testing

**Location**: `data/processed/clips/`

**Clips used for manual validation**:
- `01_set1_rally1_ball2_Clear.mp4` - Clear stroke reference
- `01_set1_rally2_ball3_Smash.mp4` - Smash stroke reference
- ~4,983 professional clips total

**Testing workflow**:
1. Run `diagnose.py` to ensure environment is correct
2. Run `analyze_video.py` on sample clip
3. Visual inspection of output:
   - `feedback.txt` - Check feedback makes sense
   - `comprehensive_report_*.png` - Visual review of charts
   - Terminal output - Overall score, severity breakdown
4. Repeat for multiple stroke types (Clear, Smash)

**Problems**:
- No assertions on expected output
- No regression detection
- Time-consuming (manual review)
- Subjective (what looks "right"?)

### Debug Scripts (Ad-Hoc Testing)

**Purpose**: One-off validation during development

**Examples**:
- [`baseline_model_debug.py`](baseline_model_debug.py) - Model debugging
- [`debug_splits_colab.py`](debug_splits_colab.py) - Data split validation
- [`filter_backhand_and_regenerate.py`](filter_backhand_and_regenerate.py) - Data filtering test
- [`analyze_stroke_types.py`](analyze_stroke_types.py) - Stroke classification test

**Characteristics**:
- Not reusable tests
- Often contain hardcoded paths
- May be out of date with current code
- No standard structure

---

## Test-Like Code Patterns

### Manual Assertions in Functions

Some functions have basic validation checks (not formal tests):

**Example 1**: Defensive checks in feature engineering
```python
def calculate_velocity(positions, smooth=True):
    if len(positions) < 2:
        return np.zeros_like(positions)  # Edge case handling
    ...
```

**Example 2**: Input validation in coaching
```python
def _analyze_arm_extension(self, features: Dict, benchmarks: Dict):
    if metric not in features or metric not in benchmarks:
        return  # Silently skip if data missing
    ...
```

**Example 3**: Sanity checks in pose extraction
```python
if num_valid_frames < 5:
    raise ValueError("Need at least 5 valid frames for analysis")
```

**Not formal tests**: These are runtime checks, not test cases. No test harness, no failure reporting.

### Example Docstrings

Some functions have docstring examples (not doctests):

```python
def analyze_technique(self, features: Dict, stroke_type: str) -> List[FeedbackItem]:
    """
    Example:
        >>> feedback_gen = CoachingFeedback()
        >>> items = feedback_gen.analyze_technique(features, 'Clear')
        >>> for item in items[:3]:
        ...     print(item.format())
    """
```

**Status**: Documentation only, not executable tests. No `doctest` module usage.

### Main Blocks for Manual Testing

Many modules have `if __name__ == "__main__"` blocks:

**Example**: [`feedback_generator.py:502-505`](src/coaching/feedback_generator.py#L502-L505)
```python
if __name__ == "__main__":
    print("FEEDBACK GENERATOR TEST")
    # Manual testing code here
```

**Purpose**: Quick manual validation during development, not automated tests.

---

## Testing Gaps

### Unit Testing

**Status**: None

**What should be tested**:
1. **Feature engineering functions**:
   - `calculate_velocity()` - Input: position array → Output: velocity array
   - `calculate_acceleration()` - Input: velocity array → Output: acceleration array
   - `smooth_trajectory()` - Input: noisy positions → Output: smoothed positions
   - `extract_frame_features()` - Input: pose keypoints → Output: 60 spatial features

2. **Benchmark comparison logic**:
   - `TechniqueBenchmarks.get_metric_status()` - Input: value, benchmark → Output: status, distance
   - `TechniqueBenchmarks.format_value()` - Input: value, metric → Output: formatted string

3. **Feedback generation**:
   - `_analyze_velocity()` - Input: features, benchmarks → Output: FeedbackItem
   - `_calculate_overall_score()` - Input: features, benchmarks → Output: score (0-100)

4. **Data validation**:
   - Pose keypoint validation (33 landmarks, correct format)
   - Feature vector completeness (427 features)
   - Stroke type validation ('Clear', 'Smash' only)

**Why it matters**: Pure functions, deterministic, easy to test.

### Integration Testing

**Status**: None

**What should be tested**:
1. **Pose extraction pipeline**:
   - Input: Video file (.mp4)
   - Output: `pose_data` dict (correct structure, valid keypoints)
   - Expected: No crashes, quality metrics above threshold

2. **Feature engineering pipeline**:
   - Input: `pose_data` dict
   - Output: `stat_features` dict (427 keys, all numeric)
   - Expected: All required features present, no NaN values

3. **Coaching pipeline**:
   - Input: `stat_features` dict, stroke type
   - Output: List of `FeedbackItem` objects
   - Expected: Non-empty list, sorted by severity, overall score calculated

4. **End-to-end pipeline**:
   - Input: Video file path + stroke type
   - Output: Files in `outputs/video_analysis/<video_name>/`
   - Expected: All 5 files created (feedback.txt + 4 PNGs)

**Why it matters**: Catches integration bugs between modules.

### Regression Testing

**Status**: None

**Problem**: No way to detect when changes break existing functionality.

**What should be tested**:
1. **Benchmark stability**: Changes to benchmarks shouldn't accidentally shift all scores
2. **Feature consistency**: Same video should produce same features across runs (deterministic)
3. **Visualization output**: Charts should render correctly after Matplotlib updates
4. **Stroke detection**: Auto-detection logic should maintain accuracy

**Why it matters**: Prevents silent regressions during refactoring.

### Performance Testing

**Status**: None

**What should be measured**:
1. **Pose extraction speed**: Frames processed per second
2. **Feature engineering speed**: Time to compute 427 features
3. **End-to-end latency**: Video upload → feedback (user experience)
4. **Memory usage**: Peak memory during video processing

**Why it matters**: Ensure system remains responsive, detect performance degradation.

### Edge Case Testing

**Status**: Minimal (some defensive checks)

**What should be tested**:
1. **Missing keypoints**: What if MediaPipe fails to detect some landmarks?
2. **Short videos**: What if video has < 5 frames?
3. **No player visible**: What if pose detection fails entirely?
4. **Invalid stroke type**: What if user passes "Volley" instead of "Clear"?
5. **Corrupted video file**: What if video can't be read?
6. **Extreme feature values**: What if velocity is 1000x too high (sensor error)?

**Why it matters**: Graceful degradation, informative error messages.

---

## Test Coverage (Hypothetical)

If tests existed, expected coverage:

| Module | Testability | Priority | Reason |
|--------|-------------|----------|--------|
| **feature_engineering_v2.py** | High | Critical | Pure functions, deterministic, core logic |
| **technique_benchmarks.py** | High | Critical | Simple lookups, comparison logic |
| **feedback_generator.py** | High | Critical | Rule-based logic, deterministic |
| **extract_poses.py** | Medium | High | External dependency (MediaPipe), non-deterministic |
| **visualizations.py** | Medium | Medium | Output is images (hard to assert), but can test no crashes |
| **analyze_video.py** | Medium | High | Integration test, orchestrates all modules |
| **streamlit_app.py** | Low | Medium | UI testing (requires Streamlit test framework) |

---

## Recommended Testing Strategy

### Phase 1: Unit Tests (Start Here)

**Setup**:
```bash
pip install pytest pytest-cov
```

**Structure**:
```
tests/
├── __init__.py
├── test_feature_engineering.py    # Test feature extraction
├── test_benchmarks.py              # Test benchmark logic
├── test_feedback_generator.py      # Test feedback generation
└── fixtures/
    ├── sample_poses.pkl            # Known-good pose data
    └── sample_features.pkl         # Known-good features
```

**Example test** (feature engineering):
```python
# tests/test_feature_engineering.py
import numpy as np
from src.data_processing.feature_engineering_v2 import calculate_velocity

def test_calculate_velocity_basic():
    positions = np.array([[0, 0], [1, 1], [2, 2]])
    velocities = calculate_velocity(positions, smooth=False)
    assert velocities.shape == positions.shape
    assert np.allclose(velocities[1], [1, 1])  # Second velocity

def test_calculate_velocity_edge_case():
    positions = np.array([[0, 0]])  # Only 1 position
    velocities = calculate_velocity(positions)
    assert velocities.shape == (1, 2)
    assert np.allclose(velocities, [[0, 0]])  # Should return zeros
```

**Run**:
```bash
pytest tests/ -v
```

### Phase 2: Integration Tests

**Example test** (pose extraction):
```python
# tests/test_integration.py
from pathlib import Path
from src.data_processing.extract_poses import PoseExtractor

def test_pose_extraction_end_to_end():
    video_path = "data/processed/clips/01_set1_rally1_ball2_Clear.mp4"
    extractor = PoseExtractor()
    pose_data = extractor.extract_from_video(video_path, "Clear")
    extractor.close()

    assert pose_data is not None
    assert 'poses' in pose_data
    assert pose_data['poses'].shape[1] == 99  # 33 keypoints × 3 coords
    assert pose_data['quality']['valid_percentage'] > 50  # At least 50% valid
```

### Phase 3: Regression Tests

**Golden data approach**:
1. Run pipeline on sample videos, save outputs
2. Store expected outputs as "golden" references
3. Test that new code produces same outputs

**Example**:
```python
# tests/test_regression.py
import pickle
from src.coaching import CoachingFeedback

def test_feedback_unchanged_for_sample_clear():
    with open('tests/fixtures/sample_clear_features.pkl', 'rb') as f:
        features = pickle.load(f)

    coach = CoachingFeedback()
    feedback = coach.analyze_technique(features, 'Clear')

    # Check overall score hasn't drifted
    assert 70 <= coach.overall_score <= 80  # Expected range

    # Check critical issue detection
    critical_count = sum(1 for item in feedback if item.severity == 'critical')
    assert critical_count == 0  # This sample has no critical issues
```

### Phase 4: CI/CD Integration

**GitHub Actions workflow** (`.github/workflows/test.yml`):
```yaml
name: Tests

on: [push, pull_request]

jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v2
      - uses: actions/setup-python@v2
        with:
          python-version: '3.10'
      - run: pip install -r requirements.txt
      - run: pip install pytest pytest-cov
      - run: pytest tests/ --cov=src --cov-report=term
```

---

## Testing Tools Recommendation

| Tool | Purpose | Priority |
|------|---------|----------|
| **pytest** | Test runner, fixtures | Critical |
| **pytest-cov** | Coverage reporting | High |
| **hypothesis** | Property-based testing (edge cases) | Medium |
| **unittest.mock** | Mock MediaPipe, file I/O | High |
| **pytest-benchmark** | Performance regression tests | Low |

---

## Blockers to Testing

1. **MediaPipe dependency**: Hard to mock, non-deterministic
   - **Solution**: Record known-good pose outputs, test downstream logic only
2. **Large video files**: Can't commit to repo
   - **Solution**: Use small test clips (<1 MB), or generate synthetic pose data
3. **Visualization output**: Hard to assert correctness
   - **Solution**: Test no crashes, test data structures (not pixels)

---

## Summary

**Current state**: No automated testing. Relies entirely on manual validation.

**Strengths**:
- `diagnose.py` provides environment validation
- Many pure functions (easy to test)
- Sample data available for validation

**Weaknesses**:
- No test suite
- No CI/CD
- No regression detection
- Manual testing is time-consuming and error-prone

**Recommendation**: Start with unit tests for feature engineering and feedback generation (highest ROI, easiest to implement).
