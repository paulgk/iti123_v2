"""
Unit Tests for Feature Engineering

Tests velocity calculation, smoothing, angle computation,
and other biomechanical feature extraction functions.
"""

import pytest
import numpy as np
from src.data_processing.feature_engineering_v2 import (
    calculate_velocity,
    calculate_acceleration,
    calculate_jerk,
    smooth_trajectory,
    calculate_angle_3d
)


class TestCalculateVelocity:
    """Test velocity calculation from position data."""

    def test_basic_velocity_calculation(self):
        """Should calculate velocity correctly for simple linear motion."""
        # Linear motion: moving 1 unit per frame
        positions = np.array([
            [0.0, 0.0],
            [1.0, 1.0],
            [2.0, 2.0],
            [3.0, 3.0]
        ])
        velocities = calculate_velocity(positions, smooth=False)

        # First velocity should be zero (no previous position)
        assert np.allclose(velocities[0], [0.0, 0.0])

        # Subsequent velocities should be [1, 1]
        for i in range(1, len(velocities)):
            assert np.allclose(velocities[i], [1.0, 1.0], atol=0.1)

    def test_velocity_shape(self):
        """Output shape should match input shape."""
        positions = np.random.rand(10, 3)
        velocities = calculate_velocity(positions, smooth=False)
        assert velocities.shape == positions.shape

    def test_single_position_edge_case(self):
        """Single position should return zero velocity."""
        positions = np.array([[1.0, 2.0, 3.0]])
        velocities = calculate_velocity(positions, smooth=False)
        assert velocities.shape == (1, 3)
        assert np.allclose(velocities[0], [0.0, 0.0, 0.0])

    def test_two_positions(self):
        """Two positions should calculate one velocity correctly."""
        positions = np.array([
            [0.0, 0.0],
            [5.0, 10.0]
        ])
        velocities = calculate_velocity(positions, smooth=False)
        assert np.allclose(velocities[0], [0.0, 0.0])
        assert np.allclose(velocities[1], [5.0, 10.0])

    def test_smoothing_reduces_noise(self):
        """Smoothing should reduce high-frequency noise."""
        # Create noisy signal
        np.random.seed(42)
        t = np.linspace(0, 10, 50)
        positions = np.column_stack([
            t + np.random.normal(0, 0.1, 50),
            2*t + np.random.normal(0, 0.1, 50)
        ])

        vel_noisy = calculate_velocity(positions, smooth=False)
        vel_smooth = calculate_velocity(positions, smooth=True)

        # Smoothed velocity should have lower variance
        assert np.var(vel_smooth) < np.var(vel_noisy)

    def test_zero_velocity_for_stationary(self):
        """Stationary positions should give zero velocity."""
        positions = np.array([
            [5.0, 3.0],
            [5.0, 3.0],
            [5.0, 3.0]
        ])
        velocities = calculate_velocity(positions, smooth=False)
        assert np.allclose(velocities, [[0.0, 0.0]] * 3)


class TestCalculateAcceleration:
    """Test acceleration calculation from velocity data."""

    def test_basic_acceleration_calculation(self):
        """Should calculate acceleration correctly."""
        # Constant acceleration: velocity increases by 1 per frame
        velocities = np.array([
            [0.0, 0.0],
            [1.0, 1.0],
            [2.0, 2.0],
            [3.0, 3.0]
        ])
        accelerations = calculate_acceleration(velocities)

        # First acceleration should be zero
        assert np.allclose(accelerations[0], [0.0, 0.0])

        # Subsequent accelerations should be [1, 1]
        for i in range(1, len(accelerations)):
            assert np.allclose(accelerations[i], [1.0, 1.0], atol=0.1)

    def test_acceleration_shape(self):
        """Output shape should match input shape."""
        velocities = np.random.rand(20, 3)
        accelerations = calculate_acceleration(velocities)
        assert accelerations.shape == velocities.shape

    def test_constant_velocity_zero_acceleration(self):
        """Constant velocity should give zero acceleration."""
        velocities = np.array([
            [5.0, 3.0],
            [5.0, 3.0],
            [5.0, 3.0]
        ])
        accelerations = calculate_acceleration(velocities)
        assert np.allclose(accelerations, [[0.0, 0.0]] * 3)


class TestCalculateJerk:
    """Test jerk (rate of change of acceleration) calculation."""

    def test_constant_acceleration_zero_jerk(self):
        """Constant acceleration should give zero jerk."""
        accelerations = np.array([
            [1.0, 1.0],
            [1.0, 1.0],
            [1.0, 1.0]
        ])
        jerk = calculate_jerk(accelerations)
        assert np.allclose(jerk, [[0.0, 0.0]] * 3)

    def test_jerk_shape(self):
        """Output shape should match input shape."""
        accelerations = np.random.rand(15, 3)
        jerk = calculate_jerk(accelerations)
        assert jerk.shape == accelerations.shape


class TestSmoothTrajectory:
    """Test trajectory smoothing functionality."""

    def test_smoothing_preserves_shape(self):
        """Smoothed output should have same shape as input."""
        trajectory = np.random.rand(50, 3)
        smoothed = smooth_trajectory(trajectory, sigma=1.5)
        assert smoothed.shape == trajectory.shape

    def test_smoothing_reduces_noise(self):
        """Smoothing should reduce variance."""
        np.random.seed(42)
        # Signal with noise
        t = np.linspace(0, 10, 100)
        clean = np.sin(t)
        noisy = clean + np.random.normal(0, 0.1, 100)
        trajectory = noisy.reshape(-1, 1)

        smoothed = smooth_trajectory(trajectory, sigma=2.0)

        # Smoothed should have lower variance
        assert np.var(smoothed) < np.var(trajectory)

    def test_smoothing_preserves_trend(self):
        """Smoothing should preserve overall trend."""
        # Linear trend
        trajectory = np.arange(100).reshape(-1, 1).astype(float)
        smoothed = smooth_trajectory(trajectory, sigma=1.5)

        # Mean should be approximately the same
        assert abs(np.mean(smoothed) - np.mean(trajectory)) < 1.0

    def test_edge_behavior(self):
        """Smoothing should handle edges gracefully."""
        trajectory = np.ones((10, 2))
        smoothed = smooth_trajectory(trajectory, sigma=1.0)

        # Constant signal should remain approximately constant
        assert np.allclose(smoothed, trajectory, atol=0.1)


class TestCalculateAngle3D:
    """Test 3D angle calculation between three points."""

    def test_90_degree_angle(self):
        """Perpendicular vectors should give 90 degrees."""
        p1 = np.array([0.0, 0.0, 0.0])  # Origin
        p2 = np.array([1.0, 0.0, 0.0])  # Vertex
        p3 = np.array([1.0, 1.0, 0.0])  # End point

        angle = calculate_angle_3d(p1, p2, p3)
        assert np.isclose(angle, 90.0, atol=1.0)

    def test_180_degree_angle(self):
        """Points in a straight line should give 180 degrees."""
        p1 = np.array([0.0, 0.0, 0.0])
        p2 = np.array([1.0, 0.0, 0.0])
        p3 = np.array([2.0, 0.0, 0.0])

        angle = calculate_angle_3d(p1, p2, p3)
        assert np.isclose(angle, 180.0, atol=1.0)

    def test_acute_angle(self):
        """Should calculate acute angles correctly."""
        p1 = np.array([0.0, 0.0, 0.0])
        p2 = np.array([1.0, 0.0, 0.0])
        p3 = np.array([1.5, 0.5, 0.0])  # Should give angle < 90

        angle = calculate_angle_3d(p1, p2, p3)
        assert 0 < angle < 90

    def test_obtuse_angle(self):
        """Should calculate obtuse angles correctly."""
        p1 = np.array([0.0, 0.0, 0.0])
        p2 = np.array([1.0, 0.0, 0.0])
        p3 = np.array([0.5, 0.5, 0.0])  # Should give angle > 90

        angle = calculate_angle_3d(p1, p2, p3)
        assert 90 < angle < 180

    def test_3d_angle_calculation(self):
        """Should work in 3D space."""
        p1 = np.array([1.0, 0.0, 0.0])
        p2 = np.array([0.0, 0.0, 0.0])
        p3 = np.array([0.0, 1.0, 0.0])

        angle = calculate_angle_3d(p1, p2, p3)
        assert np.isclose(angle, 90.0, atol=1.0)


class TestFeatureEngineeringEdgeCases:
    """Test edge cases and error handling."""

    def test_empty_array(self):
        """Empty arrays should be handled gracefully."""
        empty = np.array([]).reshape(0, 3)
        result = calculate_velocity(empty, smooth=False)
        assert result.shape == (0, 3)

    def test_nan_handling_velocity(self):
        """NaN values should propagate correctly."""
        positions = np.array([
            [1.0, 2.0],
            [np.nan, 3.0],
            [4.0, 5.0]
        ])
        velocities = calculate_velocity(positions, smooth=False)
        # NaN should propagate to velocity calculation
        assert np.isnan(velocities[2, 0])

    def test_large_values(self):
        """Should handle large coordinate values."""
        positions = np.array([
            [1e6, 1e6],
            [1e6 + 1, 1e6 + 1]
        ])
        velocities = calculate_velocity(positions, smooth=False)
        assert np.allclose(velocities[1], [1.0, 1.0])

    def test_very_small_differences(self):
        """Should handle very small position differences."""
        positions = np.array([
            [1.0, 1.0],
            [1.0 + 1e-10, 1.0 + 1e-10]
        ])
        velocities = calculate_velocity(positions, smooth=False)
        # Should calculate very small velocity
        assert velocities[1, 0] > 0
        assert velocities[1, 0] < 1e-5

    def test_negative_coordinates(self):
        """Should handle negative coordinates correctly."""
        positions = np.array([
            [-5.0, -3.0],
            [-3.0, -1.0]
        ])
        velocities = calculate_velocity(positions, smooth=False)
        assert np.allclose(velocities[1], [2.0, 2.0])
