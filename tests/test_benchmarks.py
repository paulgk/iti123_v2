"""
Unit Tests for TechniqueBenchmarks

Tests the benchmark comparison logic, metric status determination,
and value formatting functionality.
"""

import pytest
from src.coaching.technique_benchmarks import (
    TechniqueBenchmarks,
    CLEAR_BENCHMARKS,
    SMASH_BENCHMARKS
)


class TestGetBenchmarks:
    """Test benchmark retrieval for different stroke types."""

    def test_get_clear_benchmarks(self):
        """Should return Clear benchmarks dictionary."""
        benchmarks = TechniqueBenchmarks.get_benchmarks('Clear')
        assert benchmarks == CLEAR_BENCHMARKS
        assert 'max_velocity' in benchmarks
        assert 'elbow_angle_target' in benchmarks

    def test_get_smash_benchmarks(self):
        """Should return Smash benchmarks dictionary."""
        benchmarks = TechniqueBenchmarks.get_benchmarks('Smash')
        assert benchmarks == SMASH_BENCHMARKS
        assert 'max_velocity' in benchmarks
        assert 'elbow_angle_target' in benchmarks

    def test_get_invalid_stroke_type(self):
        """Should return empty dict for invalid stroke type."""
        benchmarks = TechniqueBenchmarks.get_benchmarks('Invalid')
        assert benchmarks == {}

    def test_case_sensitive_stroke_type(self):
        """Stroke type should be case-sensitive."""
        assert TechniqueBenchmarks.get_benchmarks('clear') == {}
        assert TechniqueBenchmarks.get_benchmarks('CLEAR') == {}


class TestGetMetricStatus:
    """Test metric status determination (optimal/acceptable/below/above)."""

    @pytest.fixture
    def clear_benchmarks(self):
        """Provide Clear benchmarks for testing."""
        return CLEAR_BENCHMARKS

    def test_optimal_status_at_target(self, clear_benchmarks):
        """Value at target should be optimal."""
        target = clear_benchmarks['max_velocity_target']
        status, distance = TechniqueBenchmarks.get_metric_status(
            target, clear_benchmarks, 'max_velocity'
        )
        assert status == 'optimal'
        assert distance == 0.0

    def test_optimal_status_near_target(self, clear_benchmarks):
        """Value within 10% of range width should be optimal."""
        target = clear_benchmarks['max_velocity_target']  # 75.78
        min_val = clear_benchmarks['max_velocity']  # 48.15
        max_val = clear_benchmarks['max_velocity_upper']  # 92.33
        range_width = max_val - min_val  # 44.18

        # Value within 10% of range (4.418 units) from target
        near_target = target + 3.0
        status, distance = TechniqueBenchmarks.get_metric_status(
            near_target, clear_benchmarks, 'max_velocity'
        )
        assert status == 'optimal'

    def test_acceptable_status_in_range(self, clear_benchmarks):
        """Value within range but not near target should be acceptable."""
        min_val = clear_benchmarks['max_velocity']  # 48.15
        # Value in range but not near target
        value = min_val + 10.0  # 58.15
        status, distance = TechniqueBenchmarks.get_metric_status(
            value, clear_benchmarks, 'max_velocity'
        )
        assert status == 'acceptable'

    def test_below_status(self, clear_benchmarks):
        """Value below minimum should be 'below'."""
        min_val = clear_benchmarks['max_velocity']  # 48.15
        value = min_val - 5.0  # 43.15
        status, distance = TechniqueBenchmarks.get_metric_status(
            value, clear_benchmarks, 'max_velocity'
        )
        assert status == 'below'
        assert distance == 5.0

    def test_above_status(self, clear_benchmarks):
        """Value above maximum should be 'above'."""
        max_val = clear_benchmarks['max_velocity_upper']  # 92.33
        value = max_val + 10.0  # 102.33
        status, distance = TechniqueBenchmarks.get_metric_status(
            value, clear_benchmarks, 'max_velocity'
        )
        assert status == 'above'
        assert distance == 10.0

    def test_elbow_angle_metric(self, clear_benchmarks):
        """Test with elbow_angle metric (uses _min/_max pattern)."""
        target = clear_benchmarks['elbow_angle_target']  # 128.81
        status, distance = TechniqueBenchmarks.get_metric_status(
            target, clear_benchmarks, 'elbow_angle'
        )
        assert status == 'optimal'
        assert distance == 0.0

    def test_no_metric_name_fallback(self, clear_benchmarks):
        """Should return acceptable with 0 distance if no metric name."""
        status, distance = TechniqueBenchmarks.get_metric_status(
            50.0, clear_benchmarks, None
        )
        assert status == 'acceptable'
        assert distance == 0


class TestGetTargetRange:
    """Test target range (min, max) retrieval."""

    @pytest.fixture
    def clear_benchmarks(self):
        """Provide Clear benchmarks for testing."""
        return CLEAR_BENCHMARKS

    def test_max_velocity_range(self, clear_benchmarks):
        """Should return correct min/max for max_velocity."""
        min_val, max_val = TechniqueBenchmarks.get_target_range(
            clear_benchmarks, 'max_velocity'
        )
        assert min_val == 48.15
        assert max_val == 92.33

    def test_elbow_angle_range(self, clear_benchmarks):
        """Should return correct min/max for elbow_angle."""
        min_val, max_val = TechniqueBenchmarks.get_target_range(
            clear_benchmarks, 'elbow_angle'
        )
        assert min_val == 114.62
        assert max_val == 142.09

    def test_trunk_lean_range(self, clear_benchmarks):
        """Should return correct min/max for trunk_lean."""
        min_val, max_val = TechniqueBenchmarks.get_target_range(
            clear_benchmarks, 'trunk_lean'
        )
        assert min_val == -5.00
        assert max_val == 15.00

    def test_missing_metric_fallback(self, clear_benchmarks):
        """Should return default (0, 100) for missing metric."""
        min_val, max_val = TechniqueBenchmarks.get_target_range(
            clear_benchmarks, 'nonexistent_metric'
        )
        assert min_val == 0
        assert max_val == 100


class TestGetTargetValue:
    """Test target (optimal) value retrieval."""

    @pytest.fixture
    def clear_benchmarks(self):
        """Provide Clear benchmarks for testing."""
        return CLEAR_BENCHMARKS

    def test_max_velocity_target(self, clear_benchmarks):
        """Should return correct target for max_velocity."""
        target = TechniqueBenchmarks.get_target_value(
            clear_benchmarks, 'max_velocity'
        )
        assert target == 75.78

    def test_elbow_angle_target(self, clear_benchmarks):
        """Should return correct target for elbow_angle."""
        target = TechniqueBenchmarks.get_target_value(
            clear_benchmarks, 'elbow_angle'
        )
        assert target == 128.81

    def test_missing_metric_fallback(self, clear_benchmarks):
        """Should return 0 for missing metric."""
        target = TechniqueBenchmarks.get_target_value(
            clear_benchmarks, 'nonexistent_metric'
        )
        assert target == 0


class TestFormatValue:
    """Test value formatting for display."""

    def test_format_angle(self):
        """Angle metrics should format with degree symbol."""
        formatted = TechniqueBenchmarks.format_value(128.81, 'elbow_angle')
        assert formatted == "128.8°"

    def test_format_velocity(self):
        """Velocity metrics should format with units/s."""
        formatted = TechniqueBenchmarks.format_value(75.78, 'max_velocity')
        assert formatted == "75.8 units/s"

    def test_format_contact_point(self):
        """Contact point should format with meters (3 decimals)."""
        formatted = TechniqueBenchmarks.format_value(0.025, 'contact_point')
        assert formatted == "0.025m"

    def test_format_extension(self):
        """Extension metrics should format with meters (3 decimals)."""
        formatted = TechniqueBenchmarks.format_value(0.125, 'arm_extension')
        assert formatted == "0.125m"

    def test_format_generic_metric(self):
        """Generic metrics should format with 2 decimals."""
        formatted = TechniqueBenchmarks.format_value(1.2345, 'some_metric')
        assert formatted == "1.23"

    def test_format_case_insensitive(self):
        """Formatting should be case-insensitive."""
        assert TechniqueBenchmarks.format_value(90.0, 'Elbow_Angle') == "90.0°"
        assert TechniqueBenchmarks.format_value(50.0, 'MAX_VELOCITY') == "50.0 units/s"


class TestBenchmarkValues:
    """Test that benchmark values are reasonable and consistent."""

    def test_clear_benchmarks_completeness(self):
        """Clear benchmarks should have all required metrics."""
        required_metrics = [
            'max_velocity', 'max_velocity_target', 'max_velocity_upper',
            'elbow_angle_min', 'elbow_angle_target', 'elbow_angle_max',
            'forearm_angle_min', 'forearm_angle_target', 'forearm_angle_max',
            'contact_point_min', 'contact_point_target', 'contact_point_max',
            'shoulder_angle_min', 'shoulder_angle_target', 'shoulder_angle_max',
            'trunk_lean_min', 'trunk_lean_target', 'trunk_lean_max'
        ]
        for metric in required_metrics:
            assert metric in CLEAR_BENCHMARKS, f"Missing metric: {metric}"

    def test_smash_benchmarks_completeness(self):
        """Smash benchmarks should have all required metrics."""
        required_metrics = [
            'max_velocity', 'max_velocity_target', 'max_velocity_upper',
            'elbow_angle_min', 'elbow_angle_target', 'elbow_angle_max',
            'forearm_angle_min', 'forearm_angle_target', 'forearm_angle_max',
            'contact_point_min', 'contact_point_target', 'contact_point_max',
            'shoulder_angle_min', 'shoulder_angle_target', 'shoulder_angle_max',
            'trunk_lean_min', 'trunk_lean_target', 'trunk_lean_max'
        ]
        for metric in required_metrics:
            assert metric in SMASH_BENCHMARKS, f"Missing metric: {metric}"

    def test_min_target_max_ordering(self):
        """For all metrics, min <= target <= max."""
        for stroke_type in ['Clear', 'Smash']:
            benchmarks = TechniqueBenchmarks.get_benchmarks(stroke_type)

            # Test max_velocity (uses different key pattern)
            assert benchmarks['max_velocity'] <= benchmarks['max_velocity_target']
            assert benchmarks['max_velocity_target'] <= benchmarks['max_velocity_upper']

            # Test other metrics (use _min/_target/_max pattern)
            for metric_base in ['elbow_angle', 'forearm_angle', 'contact_point',
                                'shoulder_angle', 'trunk_lean']:
                min_val = benchmarks[f'{metric_base}_min']
                target = benchmarks[f'{metric_base}_target']
                max_val = benchmarks[f'{metric_base}_max']

                assert min_val <= target, f"{stroke_type} {metric_base}: min > target"
                assert target <= max_val, f"{stroke_type} {metric_base}: target > max"

    def test_angle_ranges_reasonable(self):
        """Angle values should be within 0-180 degrees."""
        for stroke_type in ['Clear', 'Smash']:
            benchmarks = TechniqueBenchmarks.get_benchmarks(stroke_type)

            for key in benchmarks:
                if 'angle' in key:
                    value = benchmarks[key]
                    assert 0 <= value <= 180, f"{stroke_type} {key}={value} outside [0, 180]"

    def test_velocity_ranges_positive(self):
        """Velocity values should be positive."""
        for stroke_type in ['Clear', 'Smash']:
            benchmarks = TechniqueBenchmarks.get_benchmarks(stroke_type)

            for key in benchmarks:
                if 'velocity' in key:
                    value = benchmarks[key]
                    assert value > 0, f"{stroke_type} {key}={value} should be positive"

    def test_clear_vs_smash_similarity(self):
        """Clear and Smash should have similar ranges (validates Cohen's d finding)."""
        clear = CLEAR_BENCHMARKS
        smash = SMASH_BENCHMARKS

        # Velocity targets should be within 5% of each other
        clear_vel = clear['max_velocity_target']
        smash_vel = smash['max_velocity_target']
        diff_pct = abs(clear_vel - smash_vel) / clear_vel * 100
        assert diff_pct < 5, f"Velocity targets differ by {diff_pct:.1f}%"

        # Elbow angle targets should be within 2 degrees
        clear_elbow = clear['elbow_angle_target']
        smash_elbow = smash['elbow_angle_target']
        assert abs(clear_elbow - smash_elbow) < 2, "Elbow angles too different"
