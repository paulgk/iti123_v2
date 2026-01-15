# Wrist Angle Analysis - Clear vs Smash

## The Problem

**Current features miss wrist orientation**, which is likely THE key difference:
- **Smash**: Wrist pronated (palm down), racket angled ~45° downward
- **Clear**: Wrist neutral/supinated (palm up), racket angled upward

## What MediaPipe Provides

MediaPipe Pose gives us **33 keypoints** but:
- ✅ Tracks: Shoulder, Elbow, Wrist positions (x, y, z)
- ❌ Does NOT track: Hand/finger joints, wrist rotation, racket

## What We CAN Estimate

### 1. Elbow-Wrist Vector Angle (Forearm Orientation)

We already have this partially! The **forearm vector** direction can indicate wrist flexion:

```python
# Forearm vector
forearm_vector = wrist - elbow

# Angle relative to vertical (Y-axis)
vertical = np.array([0, 1, 0])  # Downward in image coordinates
forearm_vertical_angle = angle_between(forearm_vector, vertical)

# Angle relative to horizontal (X-Z plane)
forearm_horizontal_angle = np.arctan2(forearm_vector[0], forearm_vector[2])
```

**Hypothesis**:
- **Smash**: Forearm angled more downward at contact (smaller vertical angle)
- **Clear**: Forearm more horizontal/upward at contact (larger vertical angle)

### 2. Shoulder-Elbow-Wrist Plane Orientation

We can compute the **plane** formed by shoulder-elbow-wrist to detect pronation/supination:

```python
# Vectors
v1 = elbow - shoulder
v2 = wrist - elbow

# Normal to the plane (cross product)
plane_normal = np.cross(v1, v2)

# Rotation around forearm axis (pronation/supination indicator)
# Compare plane_normal direction to body reference frame
```

**Hypothesis**:
- **Smash**: Plane rotated (pronated) - normal points more laterally
- **Clear**: Plane more vertical - normal points more anteriorly

### 3. Wrist Trajectory Curvature

The **path** the wrist takes during the swing:

```python
# Compute wrist position over time
wrist_positions = [pose[t, wrist_idx] for t in range(T)]

# Compute curvature
velocity = np.gradient(wrist_positions, axis=0)
acceleration = np.gradient(velocity, axis=0)

# Curvature = |v × a| / |v|³
curvature = np.cross(velocity, acceleration) / np.linalg.norm(velocity)**3
```

**Hypothesis**:
- **Smash**: More linear trajectory (low curvature) - direct downward swing
- **Clear**: More curved trajectory (high curvature) - upward flick motion

---

## What We CANNOT Track (Without Additional Data)

### Hand/Racket Tracking (Need Different Model)

To truly track **racket angle**, we'd need:

1. **MediaPipe Hands** (tracks 21 hand keypoints including fingers)
2. **YOLOv8 + Pose** (object detection for racket + pose)
3. **Custom trained model** on badminton footage

---

## Proposed Solution: Add Forearm Orientation Features

Let me add these features to feature_engineering_v2.py:

```python
def extract_wrist_features(shoulder, elbow, wrist, prev_wrist=None):
    """
    Extract wrist/forearm orientation features

    Args:
        shoulder: (3,) shoulder position
        elbow: (3,) elbow position
        wrist: (3,) wrist position
        prev_wrist: (3,) previous frame wrist (for velocity)

    Returns:
        Dictionary of wrist features
    """
    features = {}

    # Forearm vector
    forearm = wrist - elbow
    forearm_length = np.linalg.norm(forearm)

    if forearm_length < 1e-6:
        # Invalid pose
        return {f'wrist_{k}': 0 for k in range(10)}

    forearm_unit = forearm / forearm_length

    # 1. Forearm vertical angle (flexion indicator)
    # Angle between forearm and vertical axis
    # Y-axis points DOWN in image coordinates
    vertical = np.array([0, 1, 0])
    cos_angle = np.dot(forearm_unit, vertical)
    forearm_vertical_angle = np.arccos(np.clip(cos_angle, -1, 1))
    features['forearm_vertical_angle'] = forearm_vertical_angle

    # 2. Forearm horizontal angle (lateral swing direction)
    # Angle in X-Z plane
    forearm_horizontal_angle = np.arctan2(forearm_unit[0], forearm_unit[2])
    features['forearm_horizontal_angle'] = forearm_horizontal_angle

    # 3. Shoulder-Elbow-Wrist plane normal (pronation indicator)
    upper_arm = elbow - shoulder
    plane_normal = np.cross(upper_arm, forearm)
    plane_normal_length = np.linalg.norm(plane_normal)

    if plane_normal_length > 1e-6:
        plane_normal_unit = plane_normal / plane_normal_length

        # Plane orientation relative to body midline
        # X-axis is lateral (left-right)
        lateral = np.array([1, 0, 0])
        plane_lateral_component = np.dot(plane_normal_unit, lateral)
        features['arm_plane_pronation'] = plane_lateral_component
    else:
        features['arm_plane_pronation'] = 0

    # 4. Wrist height relative to elbow (flexion indicator)
    wrist_elbow_height = wrist[1] - elbow[1]  # Y is down, so negative = wrist above
    features['wrist_elbow_height'] = wrist_elbow_height

    # 5. Forearm extension from body (reach)
    # Distance of wrist from shoulder in X-Z plane (horizontal)
    shoulder_wrist_horizontal = np.array([wrist[0] - shoulder[0], 0, wrist[2] - shoulder[2]])
    horizontal_reach = np.linalg.norm(shoulder_wrist_horizontal)
    features['wrist_horizontal_reach'] = horizontal_reach

    # 6. Wrist velocity direction (if prev_wrist available)
    if prev_wrist is not None:
        wrist_velocity = wrist - prev_wrist
        vel_magnitude = np.linalg.norm(wrist_velocity)

        if vel_magnitude > 1e-6:
            vel_unit = wrist_velocity / vel_magnitude

            # Velocity vertical component (downward for smash, upward for clear)
            vel_vertical = vel_unit[1]  # Y component (positive = downward)
            features['wrist_vel_vertical'] = vel_vertical

            # Velocity magnitude
            features['wrist_vel_magnitude'] = vel_magnitude
        else:
            features['wrist_vel_vertical'] = 0
            features['wrist_vel_magnitude'] = 0
    else:
        features['wrist_vel_vertical'] = 0
        features['wrist_vel_magnitude'] = 0

    return features
```

---

## Expected Impact

**If wrist angle is key discriminator**:
- Cohen's d for `forearm_vertical_angle`: **0.5-1.0** (medium-large effect)
- Model F1 improvement: **+15-25%** (significant!)

**Hypothesis**:
- **Smash at contact**: forearm_vertical_angle ~30-45° (downward)
- **Clear at contact**: forearm_vertical_angle ~120-150° (upward)

---

## Testing Plan

1. **Add wrist orientation features** to feature_engineering_v2.py
2. **Re-run feature extraction** on full dataset
3. **Analyze separability** (Cohen's d for new features)
4. **Train models** and compare F1 scores

---

## Alternative: Use MediaPipe Hands

If forearm orientation doesn't work, we could:

1. **Run MediaPipe Hands** on videos to track hand pose
2. **Extract hand orientation** (palm direction, finger positions)
3. **More accurate** racket angle estimation

**Tradeoff**: More complex, slower processing

---

## Recommendation

**Start with forearm orientation features** (easy to implement, fast to test).

If that doesn't help → Try MediaPipe Hands or YOLO racket detection.

Would you like me to implement the wrist/forearm orientation features now?
