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
