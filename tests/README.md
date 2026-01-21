# Test Suite Documentation

## Overview

Comprehensive test suite for the AI Badminton Coaching System, covering core functionality including benchmarks, feature engineering, and feedback generation.

## Test Statistics

- **Total Tests**: 71
- **Passing**: 62 (87%)
- **Failing**: 9 (13%)
- **Code Coverage**: 11% overall (100% for benchmarks module)

## Test Modules

### test_benchmarks.py (30 tests, 100% passing)

Tests for the `TechniqueBenchmarks` class and professional benchmark data.

**Covered Functionality:**
- ✅ Benchmark retrieval for Clear and Smash strokes
- ✅ Metric status determination (optimal/acceptable/below/above)
- ✅ Target range and value retrieval
- ✅ Value formatting for display
- ✅ Benchmark data integrity validation
- ✅ Range ordering (min ≤ target ≤ max)
- ✅ Clear vs Smash similarity validation (Cohen's d findings)

**Key Test Classes:**
- `TestGetBenchmarks`: Stroke type validation
- `TestGetMetricStatus`: Status classification logic
- `TestGetTargetRange`: Range extraction
- `TestGetTargetValue`: Optimal value retrieval
- `TestFormatValue`: Display formatting
- `TestBenchmarkValues`: Data integrity checks

**Coverage**: 100% of technique_benchmarks.py

### test_feature_engineering.py (24 tests, 22 passing, 92%)

Tests for biomechanical feature extraction and calculations.

**Covered Functionality:**
- ✅ Velocity calculation from positions
- ✅ Acceleration calculation from velocities
- ✅ Jerk (rate of change of acceleration)
- ✅ Trajectory smoothing with Gaussian filter
- ✅ 3D angle calculations
- ✅ Edge case handling (NaN, empty arrays, large/small values)

**Key Test Classes:**
- `TestCalculateVelocity`: Position → velocity conversion
- `TestCalculateAcceleration`: Velocity → acceleration
- `TestCalculateJerk`: Acceleration → jerk
- `TestSmoothTrajectory`: Noise reduction
- `TestCalculateAngle3D`: Angle computation (2 failing tests)
- `TestFeatureEngineeringEdgeCases`: Robustness testing

**Known Issues:**
- ⚠️ Acute/obtuse angle test expectations need adjustment (angle calculation implementation differs)

**Coverage**: 16% of feature_engineering_v2.py (core functions covered)

### test_feedback_generator.py (17 tests, 10 passing, 59%)

Tests for the coaching feedback generation system.

**Covered Functionality:**
- ✅ CoachingFeedback analysis logic
- ✅ Severity classification (critical/minor/good)
- ✅ Feedback sorting by severity
- ✅ Missing feature handling
- ✅ Drill recommendations
- ✅ Overall score calculation
- ✅ Feedback message generation

**Key Test Classes:**
- `TestFeedbackItem`: FeedbackItem data structure (2 failing - structure mismatch)
- `TestCoachingFeedback`: Main analysis functionality (4 failing)
- `TestFeedbackSeverityLogic`: Severity classification (1 failing)
- `TestFeedbackMessages`: Message quality checks

**Known Issues:**
- ⚠️ FeedbackItem data class has different structure than expected
- ⚠️ Some tests expect specific score thresholds that don't match implementation

**Coverage**: 35% of feedback_generator.py

## Running Tests

### Run All Tests
```bash
pytest tests/
```

### Run Specific Module
```bash
pytest tests/test_benchmarks.py
pytest tests/test_feature_engineering.py
pytest tests/test_feedback_generator.py
```

### Run with Coverage Report
```bash
pytest tests/ --cov=src/coaching --cov=src/data_processing --cov-report=term-missing
```

### Run with Verbose Output
```bash
pytest tests/ -v
```

### Run Only Passing Tests
```bash
pytest tests/ -k "not (acute_angle or obtuse_angle or FeedbackItem or professional_technique or poor_technique or velocity_analysis or exactly_at_boundaries or good_severity)"
```

## Test Organization

```
tests/
├── __init__.py                   # Test package init
├── README.md                     # This file
├── test_benchmarks.py            # Benchmark tests (30 tests)
├── test_feature_engineering.py   # Feature engineering tests (24 tests)
├── test_feedback_generator.py    # Feedback generation tests (17 tests)
└── fixtures/                     # Test data (future)
```

## Writing New Tests

### Test Structure

```python
class TestFeatureName:
    """Test description for the feature."""

    @pytest.fixture
    def sample_data(self):
        """Provide test data."""
        return {...}

    def test_basic_functionality(self, sample_data):
        """Should do X when Y."""
        result = function_under_test(sample_data)
        assert result == expected_value

    def test_edge_case(self):
        """Should handle edge case gracefully."""
        ...
```

### Test Naming Convention

- **Classes**: `Test<FeatureName>` (e.g., `TestCalculateVelocity`)
- **Methods**: `test_<what_is_tested>` (e.g., `test_velocity_shape`)
- **Docstrings**: Start with "Should..." describing expected behavior

### Assertion Patterns

```python
# Exact equality
assert value == expected

# Floating point comparison
assert np.isclose(value, expected, atol=0.1)
assert np.allclose(array1, array2, atol=0.01)

# Range checking
assert min_val <= value <= max_val

# Type checking
assert isinstance(result, ExpectedType)

# Collection membership
assert 'key' in dictionary
assert item in list

# Exception testing
with pytest.raises(ValueError):
    function_that_should_fail()
```

## Coverage Goals

### Current Coverage
- `technique_benchmarks.py`: 100% ✅
- `feedback_generator.py`: 35%
- `feature_engineering_v2.py`: 16%

### Target Coverage
- **Core modules**: 80%+ (benchmarks, feature engineering, feedback)
- **Integration**: 60%+ (full pipeline tests)
- **Utilities**: 70%+

### Uncovered Areas
- Pose extraction (`extract_poses.py`): 0%
- Data splitting (`data_split.py`): 0%
- Visualizations (`visualizations.py`): 6%
- Benchmark derivation (`derive_benchmarks.py`): 0%

## Continuous Integration

### Future: GitHub Actions Workflow

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
      - run: pip install -r requirements.txt -r requirements-test.txt
      - run: pytest tests/ --cov=src --cov-report=xml
      - uses: codecov/codecov-action@v2
```

## Test Data Fixtures

### Current State
- Tests use inline test data (numpy arrays, dictionaries)
- No external test fixtures yet

### Future Additions
- `fixtures/sample_pose_data.pkl`: Known-good pose extraction output
- `fixtures/sample_features.pkl`: Reference feature vectors
- `fixtures/professional_benchmarks.json`: Expected benchmark values
- `fixtures/test_videos/`: Short sample clips for integration tests

## Known Test Issues

### 1. FeedbackItem Structure Mismatch
**Affected Tests**: 2 failures in `test_feedback_generator.py`

**Issue**: Test expects `user_value`, `status` attributes but actual FeedbackItem uses different structure.

**Fix**: Update tests to match actual FeedbackItem implementation or refactor FeedbackItem.

### 2. Angle Calculation Edge Cases
**Affected Tests**: 2 failures in `test_feature_engineering.py`

**Issue**: Acute/obtuse angle test expectations don't match actual implementation behavior.

**Fix**: Review `calculate_angle_3d` implementation and adjust test expectations.

### 3. Feedback Score Thresholds
**Affected Tests**: 5 failures in `test_feedback_generator.py`

**Issue**: Tests expect specific score thresholds (e.g., ≥90 for professional, <60 for poor) that don't match actual scoring logic.

**Fix**: Either adjust test expectations or tune scoring algorithm.

## Best Practices

1. **Test Independence**: Each test should run independently (no shared state)
2. **Clear Assertions**: One logical assertion per test (use multiple assert statements if needed)
3. **Descriptive Names**: Test names should describe what is being tested
4. **Edge Cases**: Test boundary conditions, empty inputs, NaN, large values
5. **Documentation**: Every test class and method should have docstrings
6. **Fast Execution**: Keep tests fast (<100ms each) by using minimal data
7. **Reproducibility**: Use `np.random.seed()` for tests involving randomness

## Next Steps

1. ✅ Fix FeedbackItem compatibility (9 tests)
2. ✅ Add integration tests (end-to-end pipeline)
3. ✅ Increase coverage to 80% for core modules
4. ✅ Set up GitHub Actions CI/CD
5. ✅ Add test fixtures for pose data and features
6. ✅ Test visualization outputs (no crashes, valid data structures)
7. ✅ Performance benchmarks (ensure tests complete in <5 seconds total)

## Contributing

When adding new features:
1. Write tests first (TDD approach preferred)
2. Aim for 80%+ coverage on new code
3. Run full test suite before committing: `pytest tests/`
4. Update this README if adding new test modules
