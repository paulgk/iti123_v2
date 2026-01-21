"""
Professional Technique Benchmarks - FOREHAND ONLY

Updated: 2026-01-16 21:32:34
Source: ShuttleSet dataset (forehand strokes only, backhand filtered out)
Sample sizes:
  - Clear: 427 features
  - Smash: 427 features

These ranges represent the 25th-75th percentile of professional players.
"""

# Professional benchmarks for Clear stroke (forehand only)
CLEAR_BENCHMARKS = {
    'max_velocity': 48.15,  # Lower bound
    'max_velocity_target': 75.78,  # Median
    'max_velocity_upper': 92.33,  # Upper bound

    'elbow_angle_min': 114.62,
    'elbow_angle_target': 128.81,
    'elbow_angle_max': 142.09,

    'forearm_angle_min': 58.48,
    'forearm_angle_target': 80.28,
    'forearm_angle_max': 113.80,

    'contact_point_min': -0.010,
    'contact_point_target': 0.025,
    'contact_point_max': 0.060,

    'shoulder_angle_min': 43.92,
    'shoulder_angle_target': 62.66,
    'shoulder_angle_max': 87.11,

    'trunk_lean_min': -5.00,
    'trunk_lean_target': 5.00,
    'trunk_lean_max': 15.00,
}

# Professional benchmarks for Smash stroke (forehand only)
SMASH_BENCHMARKS = {
    'max_velocity': 43.78,  # Lower bound
    'max_velocity_target': 75.16,  # Median
    'max_velocity_upper': 91.70,  # Upper bound

    'elbow_angle_min': 115.29,
    'elbow_angle_target': 129.20,
    'elbow_angle_max': 142.39,

    'forearm_angle_min': 57.24,
    'forearm_angle_target': 79.97,
    'forearm_angle_max': 111.04,

    'contact_point_min': -0.012,
    'contact_point_target': 0.029,
    'contact_point_max': 0.065,

    'shoulder_angle_min': 43.34,
    'shoulder_angle_target': 61.99,
    'shoulder_angle_max': 84.83,

    'trunk_lean_min': 5.00,
    'trunk_lean_target': 15.00,
    'trunk_lean_max': 25.00,
}


class TechniqueBenchmarks:
    """
    Professional technique benchmarks for badminton overhead strokes.

    Provides utility methods for accessing and comparing against professional ranges.
    """

    @staticmethod
    def get_benchmarks(stroke_type: str) -> dict:
        """
        Get benchmark ranges for a specific stroke type.

        Args:
            stroke_type: 'Clear' or 'Smash'

        Returns:
            Dictionary of benchmark ranges
        """
        if stroke_type == 'Clear':
            return CLEAR_BENCHMARKS
        elif stroke_type == 'Smash':
            return SMASH_BENCHMARKS
        else:
            return {}

    @staticmethod
    def get_metric_status(value: float, benchmarks_dict: dict, metric_name: str = None) -> tuple:
        """
        Determine if a value is within acceptable, optimal, or outside professional range.

        Args:
            value: Measured value
            benchmarks_dict: Full benchmarks dictionary (CLEAR_BENCHMARKS or SMASH_BENCHMARKS)
            metric_name: Base metric name (e.g. 'max_velocity', 'elbow_angle')

        Returns:
            Tuple of (status, distance) where status is 'optimal', 'acceptable', 'below', or 'above'
        """
        # Extract min, target, max from flat benchmark structure
        # Keys follow pattern: metric_name, metric_name_target, metric_name_upper
        # OR: metric_name_min, metric_name_target, metric_name_max

        if not metric_name:
            # Fallback if no metric name provided
            return ('acceptable', 0)

        # Try different key patterns
        target_key = f"{metric_name}_target"
        min_key = f"{metric_name}_min" if f"{metric_name}_min" in benchmarks_dict else metric_name
        max_key = f"{metric_name}_max" if f"{metric_name}_max" in benchmarks_dict else f"{metric_name}_upper"

        target = benchmarks_dict.get(target_key, benchmarks_dict.get(metric_name, value))
        min_val = benchmarks_dict.get(min_key, target * 0.8)
        max_val = benchmarks_dict.get(max_key, target * 1.2)

        # Calculate distance from target
        distance = abs(value - target)

        # Determine status
        if value < min_val:
            return ('below', min_val - value)
        elif value > max_val:
            return ('above', value - max_val)
        elif abs(value - target) < (max_val - min_val) * 0.1:
            return ('optimal', distance)
        else:
            return ('acceptable', distance)

    @staticmethod
    def get_target_range(benchmarks_dict: dict, metric_name: str) -> tuple:
        """
        Get (min, max) target range for a metric.

        Args:
            benchmarks_dict: Full benchmarks dictionary
            metric_name: Base metric name

        Returns:
            Tuple of (min_value, max_value)
        """
        min_key = f"{metric_name}_min" if f"{metric_name}_min" in benchmarks_dict else metric_name
        max_key = f"{metric_name}_max" if f"{metric_name}_max" in benchmarks_dict else f"{metric_name}_upper"

        min_val = benchmarks_dict.get(min_key, 0)
        max_val = benchmarks_dict.get(max_key, 100)

        return (min_val, max_val)

    @staticmethod
    def get_target_value(benchmarks_dict: dict, metric_name: str) -> float:
        """
        Get target (optimal) value for a metric.

        Args:
            benchmarks_dict: Full benchmarks dictionary
            metric_name: Base metric name

        Returns:
            Target value
        """
        target_key = f"{metric_name}_target"
        return benchmarks_dict.get(target_key, 0)

    @staticmethod
    def format_value(value: float, metric: str) -> str:
        """
        Format a metric value for display.

        Args:
            value: Numeric value
            metric: Metric name (used to determine formatting)

        Returns:
            Formatted string
        """
        if 'angle' in metric.lower():
            return f"{value:.1f}°"
        elif 'velocity' in metric.lower():
            return f"{value:.1f} units/s"
        elif 'extension' in metric.lower() or 'contact' in metric.lower():
            return f"{value:.3f}m"
        else:
            return f"{value:.2f}"
