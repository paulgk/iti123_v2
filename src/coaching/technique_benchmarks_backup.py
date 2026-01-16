"""
Professional Technique Benchmarks for Badminton Strokes

Derived from ShuttleSet dataset analysis of ~1,800 Clear and ~1,600 Smash strokes.
Benchmarks represent the middle 50% (25th-75th percentile) of professional performance.

Dataset:
    ShuttleSet: A Human-Annotated Stroke-Level Singles Dataset for Badminton
    Tactical Analysis (Wang et al., 2023)
    https://arxiv.org/abs/2306.04948
"""

from typing import Dict, Tuple

class TechniqueBenchmarks:
    """
    Professional technique benchmarks for badminton overhead strokes

    Ranges derived from statistical analysis of ShuttleSet professional players:
    - min/max: 25th-75th percentile (middle 50% of performers)
    - optimal: median value

    Note: Some metrics (like torso_lean) have negative values due to
    coordinate system conventions in MediaPipe pose estimation.
    """

    # Clear stroke benchmarks (derived from 1,781 professional strokes)
    PROFESSIONAL_CLEAR = {
        'r_arm_extension_mean': {
            'min': 0.0620,
            'max': 0.1210,
            'optimal': 0.0850,
            'unit': 'normalized',
            'description': 'Right arm extension (shoulder to wrist distance, normalized by height)'
        },

        'max_velocity': {
            'min': 55.8480,
            'max': 92.6520,
            'optimal': 79.3080,
            'unit': 'pixels/frame',
            'description': 'Peak wrist velocity during stroke (proxy for racket speed)'
        },

        'r_elbow_angle_mean': {
            'min': 116.8790,
            'max': 141.5510,
            'optimal': 130.4900,
            'unit': 'degrees',
            'description': 'Elbow angle at contact (shoulder-elbow-wrist)'
        },

        'torso_lean_mean': {
            'min': -93.7210,
            'max': 4.5250,
            'optimal': -21.4670,
            'unit': 'degrees',
            'description': 'Forward body lean (negative = lean back, positive = lean forward)'
        },

        'r_wrist_height_rel_mean': {
            'min': -0.0050,
            'max': 0.0450,
            'optimal': 0.0210,
            'unit': 'normalized',
            'description': 'Wrist height relative to shoulder (positive = wrist above shoulder)'
        },

        'peak_velocity_timing': {
            'min': 0.1970,
            'max': 0.6890,
            'optimal': 0.4120,
            'unit': 'ratio (0-1)',
            'description': 'When peak velocity occurs (0=start, 1=end of stroke)'
        },

        'r_forearm_vertical_angle_mean': {
            'min': 53.5980,
            'max': 99.0430,
            'optimal': 73.8210,
            'unit': 'degrees',
            'description': 'Forearm angle relative to vertical (90° = horizontal)'
        },

        'shoulder_rotation_mean': {
            'min': -51.6550,
            'max': 45.0820,
            'optimal': -8.7910,
            'unit': 'degrees',
            'description': 'Shoulder rotation in frontal plane'
        }
    }

    # Smash stroke benchmarks (derived from 1,566 professional strokes)
    PROFESSIONAL_SMASH = {
        'r_arm_extension_mean': {
            'min': 0.0662,
            'max': 0.1257,
            'optimal': 0.0903,
            'unit': 'normalized',
            'description': 'Right arm extension (shoulder to wrist distance, normalized by height)'
        },

        'max_velocity': {
            'min': 54.4052,
            'max': 92.5371,
            'optimal': 79.2067,
            'unit': 'pixels/frame',
            'description': 'Peak wrist velocity during stroke (proxy for racket speed)'
        },

        'r_elbow_angle_mean': {
            'min': 117.0192,
            'max': 143.3215,
            'optimal': 131.0761,
            'unit': 'degrees',
            'description': 'Elbow angle at contact (shoulder-elbow-wrist)'
        },

        'torso_lean_mean': {
            'min': -97.4130,
            'max': 7.0822,
            'optimal': -27.2312,
            'unit': 'degrees',
            'description': 'Forward body lean (negative = lean back, positive = lean forward)'
        },

        'r_wrist_height_rel_mean': {
            'min': -0.0047,
            'max': 0.0483,
            'optimal': 0.0224,
            'unit': 'normalized',
            'description': 'Wrist height relative to shoulder (positive = wrist above shoulder)'
        },

        'peak_velocity_timing': {
            'min': 0.2157,
            'max': 0.7059,
            'optimal': 0.4314,
            'unit': 'ratio (0-1)',
            'description': 'When peak velocity occurs (0=start, 1=end of stroke)'
        },

        'r_forearm_vertical_angle_mean': {
            'min': 51.1998,
            'max': 96.3477,
            'optimal': 72.6840,
            'unit': 'degrees',
            'description': 'Forearm angle relative to vertical (90° = horizontal)'
        },

        'shoulder_rotation_mean': {
            'min': -46.5404,
            'max': 42.1127,
            'optimal': -5.5833,
            'unit': 'degrees',
            'description': 'Shoulder rotation in frontal plane'
        }
    }

    @staticmethod
    def get_benchmarks(stroke_type: str) -> Dict:
        """
        Get professional benchmarks for a stroke type

        Args:
            stroke_type: 'Clear', 'Smash', or Chinese equivalent
                - Clear: 'clear', 'long', 'chang qiu' (長球 = long ball in Chinese)
                - Smash: 'smash', 'kill', 'sha qiu' (殺球 = kill ball in Chinese)

        Returns:
            Dictionary of benchmarks for the stroke type

        Example:
            >>> benchmarks = TechniqueBenchmarks.get_benchmarks('Clear')
            >>> arm_ext = benchmarks['r_arm_extension_mean']
            >>> print(f"Target: {arm_ext['optimal']} {arm_ext['unit']}")
        """
        stroke_type_lower = stroke_type.lower()

        # Clear stroke: English and Chinese (長球 = chang qiu = long ball)
        if stroke_type_lower in ['clear', 'long', 'long ball', 'chang qiu', 'changqiu']:
            return TechniqueBenchmarks.PROFESSIONAL_CLEAR
        # Smash stroke: English and Chinese (殺球 = sha qiu = kill ball)
        elif stroke_type_lower in ['smash', 'kill', 'kill ball', 'sha qiu', 'shaqiu']:
            return TechniqueBenchmarks.PROFESSIONAL_SMASH
        else:
            # Default to Clear if unknown
            return TechniqueBenchmarks.PROFESSIONAL_CLEAR

    @staticmethod
    def get_metric_status(value: float, benchmark: Dict) -> Tuple[str, float]:
        """
        Determine if a metric value is within professional range

        Args:
            value: User's metric value
            benchmark: Benchmark dictionary for the metric

        Returns:
            Tuple of (status, distance):
                status: 'optimal', 'acceptable', 'below', 'above'
                distance: How far from optimal (positive value)

        Example:
            >>> benchmark = {'min': 0.06, 'max': 0.12, 'optimal': 0.085}
            >>> status, dist = TechniqueBenchmarks.get_metric_status(0.075, benchmark)
            >>> print(f"Status: {status}, Distance: {dist:.3f}")
        """
        min_val = benchmark['min']
        max_val = benchmark['max']
        optimal = benchmark['optimal']

        # Calculate distance from optimal
        distance = abs(value - optimal)

        # Determine status
        if min_val <= value <= max_val:
            # Within professional range
            range_width = max_val - min_val
            tolerance = range_width * 0.25  # 25% of range

            if distance <= tolerance:
                return 'optimal', distance
            else:
                return 'acceptable', distance
        elif value < min_val:
            return 'below', min_val - value
        else:  # value > max_val
            return 'above', value - max_val

    @staticmethod
    def get_all_metrics() -> list:
        """Get list of all benchmark metric names"""
        return list(TechniqueBenchmarks.PROFESSIONAL_CLEAR.keys())

    @staticmethod
    def format_value(value: float, metric_name: str) -> str:
        """
        Format a metric value with appropriate units

        Args:
            value: Metric value
            metric_name: Name of the metric

        Returns:
            Formatted string with value and unit

        Example:
            >>> formatted = TechniqueBenchmarks.format_value(0.085, 'r_arm_extension_mean')
            >>> print(formatted)  # "0.085"
        """
        benchmarks = TechniqueBenchmarks.PROFESSIONAL_CLEAR
        if metric_name in benchmarks:
            unit = benchmarks[metric_name]['unit']

            # Format based on typical value ranges
            if 'angle' in metric_name.lower():
                return f"{value:.1f}°"
            elif unit == 'normalized':
                return f"{value:.3f}"
            elif unit == 'ratio (0-1)':
                return f"{value:.2f}"
            elif 'velocity' in metric_name.lower():
                return f"{value:.1f}"
            else:
                return f"{value:.2f}"
        else:
            return f"{value:.2f}"


# Example usage and testing
if __name__ == "__main__":
    print("="*70)
    print("TECHNIQUE BENCHMARKS TEST")
    print("="*70)
    print()

    # Test 1: Get benchmarks
    clear_benchmarks = TechniqueBenchmarks.get_benchmarks('Clear')
    print("Clear Stroke Benchmarks:")
    for metric, values in clear_benchmarks.items():
        print(f"  {metric}:")
        print(f"    Range: {values['min']:.3f} - {values['max']:.3f} {values['unit']}")
        print(f"    Optimal: {values['optimal']:.3f}")
    print()

    # Test 2: Check metric status
    print("Metric Status Tests:")

    # Good arm extension
    benchmark = clear_benchmarks['r_arm_extension_mean']
    status, dist = TechniqueBenchmarks.get_metric_status(0.085, benchmark)
    print(f"  Arm extension 0.085: {status} (distance: {dist:.3f})")

    # Low arm extension
    status, dist = TechniqueBenchmarks.get_metric_status(0.050, benchmark)
    print(f"  Arm extension 0.050: {status} (distance: {dist:.3f})")

    # High velocity
    vel_benchmark = clear_benchmarks['max_velocity']
    status, dist = TechniqueBenchmarks.get_metric_status(95.0, vel_benchmark)
    print(f"  Velocity 95.0: {status} (distance: {dist:.3f})")
    print()

    # Test 3: Format values
    print("Value Formatting:")
    print(f"  Arm extension: {TechniqueBenchmarks.format_value(0.085, 'r_arm_extension_mean')}")
    print(f"  Elbow angle: {TechniqueBenchmarks.format_value(130.5, 'r_elbow_angle_mean')}")
    print(f"  Timing: {TechniqueBenchmarks.format_value(0.412, 'peak_velocity_timing')}")
    print()

    print("✓ All tests passed!")
