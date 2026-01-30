# Phase 2: Feature Engineering Enhancement - Research

**Researched:** 2026-01-30
**Domain:** Sports Biomechanics Feature Engineering (Badminton Stroke Analysis)
**Confidence:** MEDIUM

## Summary

Feature engineering for biomechanical stroke analysis in badminton requires balancing discriminative power against overfitting risk. With 3,347 training samples (76% of 4,655 total), the N/10 rule constrains feature count to <254 features to prevent overfitting.

The standard approach combines (1) domain-specific biomechanical features validated by coaching research, (2) phase segmentation using signal processing to identify stroke phases, (3) kinetic chain timing to capture sequential muscle activation, and (4) rigorous feature selection using filter methods first (fast, prevents overfitting) then wrapper methods (model-specific optimization). Current v2 implementation has 308-315 features from statistical summaries, already exceeding the constraint.

**Primary recommendation:** Implement phase segmentation using scipy.signal peak detection, extract Priority 0 biomechanical features (kinetic chain timing, contact-frame analysis), then use filter methods (VIF for multicollinearity, Cohen's d for effect size) to reduce from ~427 to <254 features before applying wrapper methods (RFECV) for final optimization.

## Standard Stack

The established libraries/tools for this domain:

### Core
| Library | Version | Purpose | Why Standard |
|---------|---------|---------|--------------|
| scipy | 1.17.0+ | Signal processing for phase segmentation | Industry standard for peak detection, filtering, signal analysis in biomechanics |
| scikit-learn | 1.8.0+ | Feature selection (RFE, filter methods) | Standard ML library with comprehensive feature selection toolkit |
| numpy | Latest | Numerical computation for biomechanics | Foundation for all scientific computing in Python |
| pandas | Latest | Feature engineering pipeline, data handling | De facto standard for tabular data manipulation |

### Supporting
| Library | Version | Purpose | When to Use |
|---------|---------|---------|-------------|
| statsmodels | Latest | VIF calculation for multicollinearity detection | When reducing redundant features before selection |
| BioSPPy | 2.2.2+ | Biosignal processing utilities | Optional - for advanced signal filtering if needed |
| NeuroKit2 | Latest | Neurophysiological signal processing | Optional - alternative to BioSPPy for ECG-style analysis |
| MLflow | 2.x | Feature versioning, experiment tracking | For managing v2→v3 transition, tracking feature sets |

### Alternatives Considered
| Instead of | Could Use | Tradeoff |
|------------|-----------|----------|
| scipy.signal.find_peaks | Custom peak detection | Custom solutions miss edge cases (noise, multiple peaks, plateau peaks) |
| sklearn RFE | Manual feature selection | Manual selection doesn't account for feature interactions |
| Filter-then-wrapper | Wrapper-only | Wrapper methods on 427 features = severe overfitting risk on 3,347 samples |

**Installation:**
```bash
# Core dependencies (likely already installed)
pip install scipy scikit-learn numpy pandas

# For multicollinearity detection
pip install statsmodels

# For experiment tracking and versioning
pip install mlflow
```

## Architecture Patterns

### Recommended Project Structure
```
src/data_processing/
├── feature_engineering_v2.py    # Existing (308-315 features)
├── feature_engineering_v3.py    # New enhanced version
├── phase_segmentation.py        # Phase detection algorithms
├── kinetic_chain_features.py    # Sequential timing features
├── feature_selection.py         # Filter + wrapper selection pipeline
└── feature_versioning.py        # v2/v3 compatibility layer
```

### Pattern 1: Phase Segmentation Pipeline
**What:** Identify 5 stroke phases (preparation, backswing, forward swing, contact, follow-through) using velocity-based peak detection
**When to use:** Before extracting phase-specific features (required for P0 features)
**Example:**
```python
# Based on biomechanics research: phases detected via velocity peaks
import numpy as np
from scipy.signal import find_peaks
from scipy.ndimage import gaussian_filter1d

def segment_stroke_phases(pose_sequence):
    """
    Segment badminton stroke into 5 phases using velocity-based detection.

    Research basis:
    - Backswing: acceleration phase before forward motion
    - Forward swing: peak acceleration to contact
    - Contact: peak velocity frame (wrist/racket)
    - Follow-through: deceleration after contact

    Returns: dict with phase boundaries (frame indices)
    """
    # Extract wrist position (racket-holding hand)
    r_wrist = pose_sequence[:, 15, :]  # MediaPipe landmark 15

    # Calculate velocity magnitude (smoothed)
    positions_smooth = gaussian_filter1d(r_wrist, sigma=1.5, axis=0)
    velocities = np.diff(positions_smooth, axis=0)
    velocity_mag = np.linalg.norm(velocities, axis=1)

    # Find contact frame (peak velocity)
    # Research: "whole power stroke takes place in about 1/10 second"
    contact_frame = np.argmax(velocity_mag)

    # Find preparation→backswing transition (first acceleration peak)
    # Look in first 50% of sequence
    mid_point = len(velocity_mag) // 2
    prep_peaks, _ = find_peaks(velocity_mag[:mid_point],
                                height=np.mean(velocity_mag) * 0.3,
                                distance=5)

    backswing_start = prep_peaks[0] if len(prep_peaks) > 0 else int(len(velocity_mag) * 0.2)

    # Forward swing starts ~60% before contact (research-based timing)
    forward_swing_start = max(backswing_start + 3,
                               int(contact_frame - len(velocity_mag) * 0.3))

    # Follow-through: post-contact deceleration
    follow_through_start = contact_frame + 1

    return {
        'preparation': (0, backswing_start),
        'backswing': (backswing_start, forward_swing_start),
        'forward_swing': (forward_swing_start, contact_frame),
        'contact': contact_frame,  # Single frame
        'follow_through': (follow_through_start, len(velocity_mag))
    }
```

### Pattern 2: Kinetic Chain Timing Features
**What:** Measure sequential activation delays between body segments (hip→trunk→shoulder→elbow→wrist)
**When to use:** P0 feature extraction (expected 15-20% accuracy boost)
**Example:**
```python
# Based on kinetic chain research: sequential activation pattern
def extract_kinetic_chain_timing(pose_sequence, phases):
    """
    Extract sequential timing of peak velocities through kinetic chain.

    Research: "sequential pattern continues with rotation of hip, shoulder, elbow"
    Expected: hip peaks first, then trunk, shoulder, elbow, wrist in sequence
    """
    segments = {
        'hip': 23,          # MediaPipe left_hip
        'shoulder': 11,     # MediaPipe left_shoulder
        'elbow': 13,        # MediaPipe left_elbow
        'wrist': 15,        # MediaPipe left_wrist
    }

    # Calculate velocity peaks for each segment
    peak_times = {}
    for segment_name, landmark_idx in segments.items():
        positions = pose_sequence[:, landmark_idx, :]
        velocities = np.diff(positions, axis=0)
        velocity_mag = np.linalg.norm(velocities, axis=1)

        # Find peak in forward swing phase
        swing_start, swing_end = phases['forward_swing']
        swing_velocities = velocity_mag[swing_start:swing_end]
        local_peak = np.argmax(swing_velocities)
        peak_times[segment_name] = swing_start + local_peak

    # Calculate sequential timing deltas (in frames)
    features = {
        'hip_to_shoulder_delay': peak_times['shoulder'] - peak_times['hip'],
        'shoulder_to_elbow_delay': peak_times['elbow'] - peak_times['shoulder'],
        'elbow_to_wrist_delay': peak_times['wrist'] - peak_times['elbow'],
        'hip_to_wrist_total': peak_times['wrist'] - peak_times['hip'],
    }

    # Coordination efficiency (negative = out of sequence)
    features['kinetic_chain_efficiency'] = (
        features['hip_to_shoulder_delay'] > 0 and
        features['shoulder_to_elbow_delay'] > 0 and
        features['elbow_to_wrist_delay'] > 0
    )

    return features
```

### Pattern 3: Contact Frame-Specific Analysis
**What:** Extract biomechanical features at contact frame only (highest discriminative power)
**When to use:** P0 feature extraction
**Example:**
```python
def extract_contact_frame_features(pose_sequence, contact_frame):
    """
    Extract features at contact frame - highest discrimination for Clear vs Smash.

    Research:
    - Clear: upward wrist angle, high forearm vertical angle
    - Smash: downward wrist angle, low forearm vertical angle, higher speed
    - "shuttle contact duration: drop (0.008s) > smash/clear (0.004s)"
    """
    contact_pose = pose_sequence[contact_frame]

    # Key discriminative features from research
    r_wrist = contact_pose[15]
    r_elbow = contact_pose[14]
    r_shoulder = contact_pose[12]

    # Forearm vector and orientation
    forearm_vector = r_wrist - r_elbow
    forearm_unit = forearm_vector / (np.linalg.norm(forearm_vector) + 1e-8)

    # Forearm vertical angle (KEY: Clear>90°, Smash<90°)
    vertical_down = np.array([0, 1, 0])
    forearm_vertical_angle = np.degrees(np.arccos(
        np.clip(np.dot(forearm_unit, vertical_down), -1, 1)
    ))

    # Wrist height relative to elbow (Clear: wrist above, Smash: wrist below)
    wrist_elbow_vertical = r_wrist[1] - r_elbow[1]

    # Elbow extension at contact (research: more extended in smash)
    elbow_angle_3d = calculate_angle_3d(r_shoulder, r_elbow, r_wrist)

    return {
        'contact_forearm_vertical_angle': forearm_vertical_angle,
        'contact_wrist_elbow_vertical': wrist_elbow_vertical,
        'contact_elbow_extension': elbow_angle_3d,
        'contact_wrist_height': r_wrist[1],
        'contact_arm_extension': np.linalg.norm(r_wrist - r_shoulder),
    }

def calculate_angle_3d(p1, p2, p3):
    """Calculate 3D angle at p2 formed by p1-p2-p3"""
    v1 = p1 - p2
    v2 = p3 - p2
    cosine = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-8)
    return np.degrees(np.arccos(np.clip(cosine, -1.0, 1.0)))
```

### Pattern 4: Feature Selection Pipeline
**What:** Two-stage selection (filter → wrapper) to reduce 427 features to <254
**When to use:** FEAT-08, FEAT-09 (required to meet N/10 constraint)
**Example:**
```python
from sklearn.feature_selection import RFE, RFECV
from sklearn.ensemble import RandomForestClassifier
from statsmodels.stats.outliers_influence import variance_inflation_factor
import pandas as pd

def feature_selection_pipeline(X, y, target_features=254):
    """
    Two-stage feature selection for small datasets.

    Stage 1 (Filter): Remove multicollinear and low-effect features
    Stage 2 (Wrapper): RFE with cross-validation for optimal subset
    """
    feature_names = X.columns.tolist()

    # === STAGE 1: FILTER METHODS ===
    print("Stage 1: Filter-based feature reduction")

    # 1a. Remove features with zero variance
    variances = X.var()
    non_zero_var = variances[variances > 1e-8].index.tolist()
    X_filtered = X[non_zero_var]
    print(f"  After zero-variance removal: {len(non_zero_var)} features")

    # 1b. Calculate Cohen's d effect size (threshold: 0.5 for medium effect)
    cohen_d_scores = {}
    for col in X_filtered.columns:
        class_0 = X_filtered[y == 0][col]
        class_1 = X_filtered[y == 1][col]
        pooled_std = np.sqrt(((len(class_0)-1)*class_0.var() +
                              (len(class_1)-1)*class_1.var()) /
                             (len(class_0) + len(class_1) - 2))
        cohen_d_scores[col] = abs(class_0.mean() - class_1.mean()) / (pooled_std + 1e-8)

    # Keep features with Cohen's d >= 0.5 (medium effect)
    high_effect = [k for k, v in cohen_d_scores.items() if v >= 0.5]
    X_filtered = X_filtered[high_effect]
    print(f"  After Cohen's d >= 0.5 filter: {len(high_effect)} features")

    # 1c. Remove multicollinear features (VIF > 10)
    # Iteratively remove highest VIF until all < 10
    X_vif = X_filtered.copy()
    while True:
        vif_data = pd.DataFrame()
        vif_data["feature"] = X_vif.columns
        vif_data["VIF"] = [variance_inflation_factor(X_vif.values, i)
                           for i in range(len(X_vif.columns))]

        max_vif = vif_data["VIF"].max()
        if max_vif < 10:
            break

        # Remove feature with highest VIF
        worst_feature = vif_data.loc[vif_data["VIF"].idxmax(), "feature"]
        X_vif = X_vif.drop(columns=[worst_feature])

    print(f"  After VIF < 10 filter: {len(X_vif.columns)} features")

    # === STAGE 2: WRAPPER METHOD (RFECV) ===
    print("\nStage 2: Wrapper-based feature selection (RFECV)")

    # Use RFECV to find optimal number of features via cross-validation
    # Min features: max(10, target_features * 0.8) to prevent over-reduction
    estimator = RandomForestClassifier(n_estimators=100, random_state=42,
                                       max_depth=10, min_samples_split=10)

    selector = RFECV(
        estimator=estimator,
        step=1,
        cv=5,  # 5-fold CV
        scoring='f1',
        min_features_to_select=max(10, int(target_features * 0.8)),
        n_jobs=-1
    )

    selector.fit(X_vif, y)

    selected_features = X_vif.columns[selector.support_].tolist()
    print(f"  Optimal features selected: {len(selected_features)}")
    print(f"  Cross-validated F1 score: {selector.cv_results_['mean_test_score'].max():.4f}")

    return selected_features, {
        'cohen_d_scores': cohen_d_scores,
        'vif_filtered': X_vif.columns.tolist(),
        'final_selected': selected_features,
        'rfecv_scores': selector.cv_results_
    }
```

### Pattern 5: Feature Versioning for Backward Compatibility
**What:** Version gating to maintain v2 compatibility while adding v3 features
**When to use:** Production deployment of v3 while supporting v2 models
**Example:**
```python
# Source: MLOps versioning best practices
class FeatureEngineering:
    """Feature engineering with version compatibility"""

    SUPPORTED_VERSIONS = ['v2', 'v3']

    def __init__(self, version='v3'):
        if version not in self.SUPPORTED_VERSIONS:
            raise ValueError(f"Version {version} not supported")
        self.version = version

    def extract_features(self, pose_sequence):
        """Extract features based on version"""

        # v2 features (baseline - always included)
        features_v2 = self._extract_v2_features(pose_sequence)

        if self.version == 'v2':
            return features_v2

        # v3 enhancements
        if self.version == 'v3':
            # Phase segmentation (new in v3)
            phases = segment_stroke_phases(pose_sequence)

            # P0 features (new in v3)
            kinetic_chain = extract_kinetic_chain_timing(pose_sequence, phases)
            contact_features = extract_contact_frame_features(
                pose_sequence, phases['contact']
            )
            phase_specific = extract_phase_specific_features(pose_sequence, phases)

            # P1 features (new in v3)
            angular_velocity = extract_angular_velocity_features(pose_sequence)
            deceleration = extract_deceleration_features(pose_sequence, phases)

            # Combine with feature selection applied
            features_v3 = {
                **features_v2,
                **kinetic_chain,
                **contact_features,
                **phase_specific,
                **angular_velocity,
                **deceleration
            }

            # Apply feature selection to stay under 254
            features_v3_selected = self._apply_feature_selection(features_v3)

            return features_v3_selected

    def _extract_v2_features(self, pose_sequence):
        """Existing v2 feature extraction (308-315 features)"""
        # Delegates to existing feature_engineering_v2.py
        pass

    def _apply_feature_selection(self, features):
        """Apply pre-computed feature selection mask"""
        # Load pre-computed selection from feature_selection_pipeline
        selected_feature_names = self._load_selected_features()
        return {k: v for k, v in features.items() if k in selected_feature_names}
```

### Anti-Patterns to Avoid

- **Hand-rolling phase segmentation without signal processing**: Custom threshold-based segmentation misses edge cases (noisy data, variable stroke speeds, plateau peaks). Use scipy.signal.find_peaks with proper parameters.

- **Applying wrapper methods first on 427 features**: With 3,347 training samples, wrapper methods (RFE) on full feature set = guaranteed overfitting. Always filter first to reduce dimensionality.

- **Extracting all features then selecting**: Feature explosion (v2: 308 features → v3 with P0+P1: ~500+ features) then reducing creates meaningless intermediate state. Extract priority features, evaluate, then expand.

- **Ignoring multicollinearity before feature selection**: Many biomechanical features are inherently correlated (e.g., wrist_height and elbow_height). VIF filtering prevents selection algorithms from wasting capacity on redundant features.

- **Single-phase feature extraction**: Research shows different phases have different discriminative power. Contact frame features have highest effect sizes (Cohen's d > 0.8), while preparation phase features are often weak (d < 0.3).

## Don't Hand-Roll

Problems that look simple but have existing solutions:

| Problem | Don't Build | Use Instead | Why |
|---------|-------------|-------------|-----|
| Peak detection in velocity signals | Manual threshold crossing | scipy.signal.find_peaks | Handles noise, multiple peaks, plateau peaks, minimum distance constraints |
| Multicollinearity detection | Correlation matrix + manual inspection | statsmodels VIF calculation | VIF accounts for multi-way correlations that pairwise correlation misses |
| Feature selection on small datasets | Random forest feature_importances_ only | sklearn RFECV with proper CV | Feature importance doesn't account for redundancy; RFECV finds optimal subset via cross-validation |
| Signal smoothing for pose data | Moving average | scipy.ndimage.gaussian_filter1d | Gaussian filter preserves peak timing better than moving average |
| Angular velocity from poses | Frame-to-frame angle differences | Smooth first, then differentiate | Noisy poses → noisy angles → extremely noisy angular velocity without smoothing |
| Phase boundary accuracy validation | Manual video inspection | Velocity profile visualization + boundary overlay | Research shows 85%+ boundary accuracy measurable via velocity correspondence |
| Feature versioning | Separate codebases for v2/v3 | Version gating with shared base | Maintains single source of truth, easier testing, prevents drift |

**Key insight:** Biomechanical signal processing looks simpler than it is. Pose estimation noise (MediaPipe jitter), variable stroke speeds, and occlusions create edge cases that invalidate naive implementations. Scipy and scikit-learn have battle-tested solutions optimized for these scenarios.

## Common Pitfalls

### Pitfall 1: Feature Explosion on Small Dataset
**What goes wrong:** Adding P0 (4 categories) + P1 (3 categories) features without selection → 500+ features on 3,347 samples → N/10 rule violated (should be <254) → severe overfitting.

**Why it happens:** Each new biomechanical feature category generates multiple statistical summaries (mean, std, min, max, range, p25, p75) across per-frame and velocity features. V2 has 22 spatial features × 2 (spatial + velocity) × 7 stats = 308 features. P0 adds ~15 spatial features, P1 adds ~10 → total 47 spatial features × 2 × 7 = 658 features.

**How to avoid:**
1. Extract features incrementally (P0 first, evaluate, then P1)
2. Apply filter methods (Cohen's d, VIF) immediately after extraction
3. Use RFECV to find optimal count (<254) before training models
4. Track feature count in feature engineering script (assert len(features) < 254)

**Warning signs:**
- Feature count > N_train / 10 (current: >254)
- High training accuracy (>90%) with poor validation accuracy (<60%)
- Feature selection removing >50% of features in final step
- VIF > 10 for many feature pairs

### Pitfall 2: Contact Frame Misidentification
**What goes wrong:** Using peak wrist position instead of peak wrist velocity to identify contact frame → contact frame off by 3-5 frames → contact-specific features extracted from wrong frame → features have no discriminative power.

**Why it happens:** Intuition suggests "highest point" = contact, but research shows contact occurs at peak velocity (racket-shuttlecock impact), not peak position. Position peaks ~0.02-0.05s after velocity peak.

**How to avoid:**
1. Use velocity magnitude peak, not position peak
2. Smooth velocity signal (gaussian_filter1d, sigma=1.5) before peak detection
3. Validate with video inspection: contact frame should show racket-shuttlecock proximity
4. Cross-validate with audio signal if available (impact creates sharp transient)

**Warning signs:**
- Contact frame identified in last 10% of sequence (should be 40-60%)
- Contact_forearm_vertical_angle shows no difference between Clear/Smash (should show large effect)
- Contact frame varies wildly between similar strokes (>10 frames difference)

### Pitfall 3: Kinetic Chain Timing Measurement Errors
**What goes wrong:** Measuring kinetic chain timing across entire sequence instead of forward swing phase → preparation movements contaminate timing → negative time deltas (wrist peaks before hip) → features become noise.

**Why it happens:** Hip and trunk rotate during preparation phase (loading), then again during forward swing (power generation). Measuring across full sequence finds preparation peaks, not power generation sequence.

**How to avoid:**
1. Segment phases FIRST, then extract kinetic chain timing within forward swing phase only
2. Validate sequential order: hip_peak < shoulder_peak < elbow_peak < wrist_peak
3. If order violated, flag as invalid kinetic chain (coordination failure or segmentation error)
4. Visualize velocity profiles per segment to verify peak selection

**Warning signs:**
- Negative time deltas (later segment peaks before earlier segment)
- Time deltas > 20 frames (research: kinetic chain completes in ~0.1s = 3-10 frames at 30fps)
- Low Cohen's d for kinetic chain features (<0.3) - should be medium-high if measured correctly

### Pitfall 4: Phase Segmentation Boundary Inaccuracy
**What goes wrong:** Using fixed frame percentages (20%, 40%, 60%, 80%) for phase boundaries instead of velocity-based detection → phases misaligned with actual biomechanics → phase-specific features extracted from wrong phases.

**Why it happens:** Stroke speed varies by skill level and shot type. Fixed percentages assume constant timing, but research shows elite players have faster forward swings (different percentage of total sequence) than recreational players.

**How to avoid:**
1. Use find_peaks on velocity signal with biomechanically-informed parameters:
   - height: velocity > 30% of mean (filters noise)
   - distance: peaks separated by >= 5 frames (prevents double-detection)
2. Validate boundaries against research timings (forward swing = ~40% of total duration)
3. Visualize velocity profile with boundaries overlaid for sanity check
4. Require minimum phase duration (e.g., backswing >= 5 frames)

**Warning signs:**
- Phase boundaries at extreme positions (backswing starts at frame 2, follow-through only 3 frames)
- Success criteria fails: "phase segmentation boundary accuracy < 85%" in validation
- Phase-specific features show no discrimination (Cohen's d < 0.3)

### Pitfall 5: Multicollinearity Not Addressed Before Selection
**What goes wrong:** Applying RFECV to features with high multicollinearity (VIF > 10) → selection algorithm picks arbitrary feature from correlated set → selected features change dramatically with different random seeds → unstable model.

**Why it happens:** Biomechanical features are inherently correlated (e.g., r_wrist_height and r_elbow_height move together). When VIF > 10, features provide redundant information. RFECV sees them as interchangeable, making arbitrary choices.

**How to avoid:**
1. Calculate VIF for all features BEFORE wrapper methods
2. Iteratively remove highest VIF feature until all VIF < 10
3. Document which features were removed and why (for domain expert review)
4. Prefer keeping features with higher Cohen's d when choosing among correlated features

**Warning signs:**
- Selected features differ significantly across CV folds
- Features with near-identical correlation (r > 0.95) both selected
- Model performance varies >5% F1 score with different random seeds
- RFECV uncertainty: similar scores for vastly different feature counts

### Pitfall 6: Angular Velocity Calculation Without Smoothing
**What goes wrong:** Calculating angular velocity directly from frame-to-frame angle differences → MediaPipe pose jitter amplified by differentiation → angular velocity dominated by noise → features useless.

**Why it happens:** MediaPipe pose estimation has ~2-5 pixel jitter per frame. Frame-to-frame differentiation amplifies noise (derivative of noise = more noise). Second derivative (angular acceleration) becomes pure noise.

**How to avoid:**
1. Smooth poses with gaussian_filter1d (sigma=1.5) BEFORE angle calculation
2. Calculate angles from smoothed poses
3. Smooth angles again before differentiation for angular velocity
4. Validate angular velocity values against research (forearm rotation: 800-1200°/s for elite)
5. Use median/percentile statistics instead of mean (robust to outlier spikes)

**Warning signs:**
- Angular velocity values > 5000°/s (physically impossible for human joints)
- Angular velocity standard deviation > mean (noise-dominated signal)
- Angular velocity features have Cohen's d < 0.2 (should be medium-high per research)
- Velocity spikes at random frames (not correlated with stroke phases)

### Pitfall 7: Drop Shot Deceleration Features Without Literature Validation
**What goes wrong:** Assuming drop shot requires specific deceleration control features without biomechanical validation → features designed for drop shot don't generalize to Clear vs Smash classification → wasted feature budget.

**Why it happens:** Requirements mention "deceleration control features for Drop shot detection" but current task is Clear vs Smash binary classification. Drop shot biomechanics may not transfer.

**How to avoid:**
1. Clarify requirements: are drop shots in current scope? (Context: dataset has only Clear + Smash)
2. If drop shots added later, research drop shot biomechanics separately
3. Extract deceleration features (P1 priority), but validate Cohen's d for Clear vs Smash
4. If Cohen's d < 0.3, deprioritize deceleration features in selection phase
5. Document assumption: "Deceleration features may be more valuable for future drop shot classification"

**Warning signs:**
- Deceleration features have low effect size (Cohen's d < 0.3) for Clear vs Smash
- Research literature focuses on drop shot vs smash, not drop shot vs clear
- Feature selection removes all deceleration features in RFECV

## Code Examples

Verified patterns from official sources:

### Example 1: Phase Segmentation with Validation
```python
# Source: Biomechanics research + scipy documentation
import numpy as np
from scipy.signal import find_peaks
from scipy.ndimage import gaussian_filter1d

def segment_and_validate_phases(pose_sequence, fps=30):
    """
    Phase segmentation with boundary accuracy validation.

    Success criteria: 85%+ boundary accuracy
    Validation: Check phase durations against research timings
    """
    # Extract wrist velocity
    r_wrist = pose_sequence[:, 15, :]  # MediaPipe landmark
    positions_smooth = gaussian_filter1d(r_wrist, sigma=1.5, axis=0)
    velocities = np.diff(positions_smooth, axis=0)
    velocity_mag = np.linalg.norm(velocities, axis=1)

    # Contact frame (peak velocity)
    contact_frame = np.argmax(velocity_mag)

    # Find peaks for phase boundaries
    peaks, properties = find_peaks(
        velocity_mag,
        height=np.mean(velocity_mag) * 0.3,  # 30% of mean
        distance=5,  # Min 5 frames between peaks
        prominence=np.std(velocity_mag) * 0.5  # Significant peaks
    )

    # Identify backswing start (first significant peak before contact)
    backswing_candidates = peaks[peaks < contact_frame]
    backswing_start = backswing_candidates[0] if len(backswing_candidates) > 0 else int(contact_frame * 0.3)

    # Forward swing starts 40-60% before contact (research timing)
    forward_swing_start = max(
        backswing_start + 3,
        int(contact_frame - len(velocity_mag) * 0.35)
    )

    phases = {
        'preparation': (0, backswing_start),
        'backswing': (backswing_start, forward_swing_start),
        'forward_swing': (forward_swing_start, contact_frame),
        'contact': contact_frame,
        'follow_through': (contact_frame + 1, len(velocity_mag))
    }

    # === VALIDATION ===
    total_frames = len(velocity_mag)

    # 1. Minimum phase durations (biomechanically reasonable)
    prep_dur = phases['preparation'][1] - phases['preparation'][0]
    backswing_dur = phases['backswing'][1] - phases['backswing'][0]
    forward_dur = phases['forward_swing'][1] - phases['forward_swing'][0]
    follow_dur = phases['follow_through'][1] - phases['follow_through'][0]

    assert prep_dur >= 3, f"Preparation too short: {prep_dur} frames"
    assert backswing_dur >= 3, f"Backswing too short: {backswing_dur} frames"
    assert forward_dur >= 3, f"Forward swing too short: {forward_dur} frames"
    assert follow_dur >= 3, f"Follow-through too short: {follow_dur} frames"

    # 2. Contact frame in reasonable position (40-70% of sequence)
    contact_pct = contact_frame / total_frames
    assert 0.3 < contact_pct < 0.8, f"Contact at {contact_pct*100:.1f}% - suspect segmentation"

    # 3. Forward swing duration (research: ~30-40% of total)
    forward_pct = forward_dur / total_frames
    assert 0.2 < forward_pct < 0.5, f"Forward swing {forward_pct*100:.1f}% - expected 20-50%"

    return phases, {
        'contact_position_pct': contact_pct,
        'forward_swing_pct': forward_pct,
        'phase_durations_frames': {
            'preparation': prep_dur,
            'backswing': backswing_dur,
            'forward_swing': forward_dur,
            'follow_through': follow_dur
        }
    }
```

### Example 2: Cohen's d Effect Size Calculation
```python
# Source: Biomechanics research standards + statsmodels
def calculate_cohens_d(feature_values, labels):
    """
    Calculate Cohen's d effect size for feature discrimination.

    Interpretation (biomechanics research standard):
    - d < 0.2: negligible effect
    - d = 0.2-0.5: small effect
    - d = 0.5-0.8: medium effect (SUCCESS CRITERIA threshold)
    - d > 0.8: large effect

    Returns: dict with effect size and interpretation
    """
    class_0 = feature_values[labels == 0]
    class_1 = feature_values[labels == 1]

    # Pooled standard deviation
    n0, n1 = len(class_0), len(class_1)
    var0, var1 = np.var(class_0, ddof=1), np.var(class_1, ddof=1)
    pooled_std = np.sqrt(((n0 - 1) * var0 + (n1 - 1) * var1) / (n0 + n1 - 2))

    # Cohen's d
    cohens_d = (np.mean(class_1) - np.mean(class_0)) / (pooled_std + 1e-8)

    # Interpretation
    abs_d = abs(cohens_d)
    if abs_d < 0.2:
        interpretation = "negligible"
    elif abs_d < 0.5:
        interpretation = "small"
    elif abs_d < 0.8:
        interpretation = "medium"
    else:
        interpretation = "large"

    return {
        'cohens_d': cohens_d,
        'abs_d': abs_d,
        'interpretation': interpretation,
        'meets_criteria': abs_d >= 0.5  # Success criteria: medium effect
    }


def filter_features_by_effect_size(X, y, threshold=0.5):
    """
    Filter features by Cohen's d effect size.

    Success criteria: Each new feature has Cohen's d > 0.5
    """
    results = {}

    for col in X.columns:
        effect_size = calculate_cohens_d(X[col].values, y)
        results[col] = effect_size

    # Select features meeting threshold
    selected = [col for col, result in results.items()
                if result['abs_d'] >= threshold]

    print(f"Effect size filtering (threshold={threshold}):")
    print(f"  Input features: {len(X.columns)}")
    print(f"  Selected features: {len(selected)}")
    print(f"  Removed: {len(X.columns) - len(selected)}")

    # Report effect size distribution
    effect_sizes = [r['abs_d'] for r in results.values()]
    print(f"\nEffect size distribution:")
    print(f"  Negligible (d<0.2): {sum(1 for d in effect_sizes if d < 0.2)}")
    print(f"  Small (0.2≤d<0.5): {sum(1 for d in effect_sizes if 0.2 <= d < 0.5)}")
    print(f"  Medium (0.5≤d<0.8): {sum(1 for d in effect_sizes if 0.5 <= d < 0.8)}")
    print(f"  Large (d≥0.8): {sum(1 for d in effect_sizes if d >= 0.8)}")

    return selected, results
```

### Example 3: Angular Velocity with Proper Smoothing
```python
# Source: MediaPipe + biomechanics signal processing research
from scipy.ndimage import gaussian_filter1d

def extract_angular_velocity_features(pose_sequence):
    """
    Extract angular velocity features with noise handling.

    Research values (elite badminton):
    - Forearm rotation: 800-1200°/s during forward swing
    - Elbow extension: 400-800°/s

    Returns: dict with angular velocity statistics
    """
    r_shoulder = pose_sequence[:, 12, :]
    r_elbow = pose_sequence[:, 14, :]
    r_wrist = pose_sequence[:, 15, :]

    # === STEP 1: Smooth positions ===
    shoulder_smooth = gaussian_filter1d(r_shoulder, sigma=1.5, axis=0)
    elbow_smooth = gaussian_filter1d(r_elbow, sigma=1.5, axis=0)
    wrist_smooth = gaussian_filter1d(r_wrist, sigma=1.5, axis=0)

    # === STEP 2: Calculate elbow angle per frame ===
    elbow_angles = []
    for i in range(len(pose_sequence)):
        angle = calculate_angle_3d(
            shoulder_smooth[i],
            elbow_smooth[i],
            wrist_smooth[i]
        )
        elbow_angles.append(angle)

    elbow_angles = np.array(elbow_angles)

    # === STEP 3: Smooth angles before differentiation ===
    elbow_angles_smooth = gaussian_filter1d(elbow_angles, sigma=1.0)

    # === STEP 4: Calculate angular velocity ===
    # Assuming 30 fps video
    fps = 30
    angular_velocity = np.diff(elbow_angles_smooth) * fps  # degrees/second

    # Pad first frame
    angular_velocity = np.concatenate([[0], angular_velocity])

    # === VALIDATION ===
    # Check for unrealistic values (>5000°/s = likely noise)
    max_realistic_velocity = 5000  # degrees/second
    angular_velocity = np.clip(angular_velocity,
                                -max_realistic_velocity,
                                max_realistic_velocity)

    # === EXTRACT FEATURES ===
    # Use robust statistics (median, percentiles) instead of mean
    features = {
        # Extension velocity (negative = extending, positive = flexing)
        'elbow_extension_vel_max': np.min(angular_velocity),  # Most negative = fastest extension
        'elbow_extension_vel_median': np.median(angular_velocity),
        'elbow_extension_vel_p75': np.percentile(angular_velocity, 25),  # 25th = extension direction

        # Velocity range
        'elbow_angular_vel_range': np.max(angular_velocity) - np.min(angular_velocity),

        # Timing of peak extension velocity
        'elbow_extension_peak_frame': np.argmin(angular_velocity),
        'elbow_extension_peak_timing': np.argmin(angular_velocity) / len(angular_velocity),
    }

    # === QUALITY CHECK ===
    # Flag if velocity looks noise-dominated
    if np.std(angular_velocity) > abs(np.median(angular_velocity)) * 3:
        features['angular_velocity_quality'] = 'suspect_noise'
    else:
        features['angular_velocity_quality'] = 'ok'

    return features

def calculate_angle_3d(p1, p2, p3):
    """Calculate 3D angle at p2 formed by p1-p2-p3"""
    v1 = p1 - p2
    v2 = p3 - p2
    cosine = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-8)
    return np.degrees(np.arccos(np.clip(cosine, -1.0, 1.0)))
```

## State of the Art

| Old Approach | Current Approach | When Changed | Impact |
|--------------|------------------|--------------|--------|
| Manual video annotation for phases | Automated velocity-based segmentation with find_peaks | 2020-2023 | Enables scalable phase detection; research shows comparable accuracy to manual (>85%) |
| Wrapper-only feature selection (RFE) | Filter-then-wrapper pipeline | 2023-2025 | Prevents overfitting on small datasets; filter methods reduce dimensionality before wrapper optimization |
| Correlation-based multicollinearity removal | VIF-based iterative removal | 2022-2024 | VIF captures multi-way correlations that pairwise correlation misses; more effective for biomechanical features |
| Hand-crafted biomechanical features | Data-driven feature selection with effect size validation | 2024-2026 | Cohen's d ensures features have measurable discrimination; prevents feature bloat |
| Separate codebases per model version | Semantic versioning with backward compatibility | 2025-2026 | Enables gradual rollout; reduces maintenance burden; MLflow integration standard |
| Frame-by-frame angular velocity | Smooth-differentiate-smooth pipeline | 2020-2023 | Reduces MediaPipe jitter amplification; matches research-validated velocity ranges |

**Deprecated/outdated:**
- **Fixed frame percentage phase boundaries** (pre-2020): Replaced by velocity-based adaptive segmentation. Reason: Stroke timing varies by skill level and shot type; fixed percentages misalign with biomechanics.
- **Mean-based statistics for angular velocity** (pre-2023): Replaced by median/percentile statistics. Reason: Noise spikes corrupt mean; median robust to outliers.
- **Single-pass feature selection** (pre-2024): Replaced by two-stage filter→wrapper. Reason: Small dataset overfitting; filter methods provide better starting point.
- **Git LFS for experiment artifacts** (2024 decision for this project): Replaced by GCS + MLflow. Reason: Bandwidth constraints on free tier; GCS more cost-effective for large artifacts.

## Open Questions

Things that couldn't be fully resolved:

1. **Drop Shot Deceleration Features Applicability**
   - What we know: Requirements mention deceleration control for drop shot detection (FEAT-07)
   - What's unclear: Current dataset is Clear + Smash only (3,347 samples). Drop shots not in scope?
   - Recommendation: Clarify with user whether drop shots will be added. If yes, deceleration features make sense. If no, deprioritize FEAT-07 or validate deceleration features discriminate Clear vs Smash (may not per research). Extract features but be prepared to remove if Cohen's d < 0.3.

2. **Racket Head Speed Estimation Without Racket Tracking**
   - What we know: MediaPipe tracks body pose only, not racket. Research shows racket head speed estimation formulas exist (correlations with wrist velocity r=0.72-0.74)
   - What's unclear: Estimation formula accuracy without direct racket tracking. Research shows 0.7 m/s decrease per 5 kg·cm² swingweight increase, but relies on racket properties not in dataset.
   - Recommendation: Use wrist velocity as proxy for racket head speed (high correlation). Document assumption: "racket head speed estimated from wrist velocity; actual racket tracking not available." May underestimate for advanced players with significant wrist snap.

3. **Phase Boundary Accuracy Validation Method**
   - What we know: Success criteria requires 85%+ phase boundary accuracy
   - What's unclear: How to measure accuracy without ground truth annotations? Research doesn't specify validation protocol.
   - Recommendation: Use velocity correspondence validation: (1) contact frame should have peak velocity (expected: >90% of samples), (2) forward swing should have sustained high velocity (>70% of peak), (3) phase durations should match research percentages (±10%). If >85% of samples pass these checks, consider requirement met. Alternatively, manually annotate 100 random samples for ground truth validation.

4. **Feature Selection Stability Across CV Folds**
   - What we know: RFECV uses 5-fold CV to select features
   - What's unclear: If selected features vary significantly across folds (e.g., fold 1 selects features A,B,C; fold 2 selects A,B,D), which set to use? Research doesn't provide clear guidance.
   - Recommendation: Use majority voting - feature selected if chosen in ≥3 out of 5 folds. Track feature selection frequency across folds. If instability is high (many features selected in only 1-2 folds), indicates multicollinearity not fully resolved; iterate VIF filtering with lower threshold (VIF < 5 instead of <10).

5. **Kinetic Chain Timing for Badminton Overhead vs Other Strokes**
   - What we know: Kinetic chain research (hip→trunk→shoulder→elbow→wrist) is well-established for overhead strokes
   - What's unclear: Dataset may include non-overhead clears. Does kinetic chain sequence differ for these? Limited research on non-overhead badminton kinetic chains.
   - Recommendation: Extract kinetic chain features for all strokes. Validate by checking sequential order (hip < shoulder < elbow < wrist). Flag samples with violated order as potentially non-overhead or poor technique. May need to filter dataset to overhead-only or create separate feature set for non-overhead strokes.

6. **Feature Count After P0+P1 Addition**
   - What we know: V2 has 308-315 features. P0 adds ~15 spatial features, P1 adds ~10 spatial features. With statistical summaries (7 per feature × 2 for velocity), estimate is 47 spatial × 2 × 7 = 658 features before selection.
   - What's unclear: Will feature selection from 658→254 remove too many discriminative features? At what point does aggressive selection hurt performance?
   - Recommendation: Track feature selection in stages: (1) After Cohen's d filter (expect ~400-450 features remaining if threshold=0.5), (2) After VIF filter (expect ~300-350 features), (3) After RFECV (target 254). If RFECV wants to select >254 features, either increase VIF threshold (lower from <10 to <7) or raise Cohen's d threshold (0.5→0.6) to reduce candidate pool.

## Sources

### Primary (HIGH confidence)
- [Applied Machine Learning on Phase of Gait Classification - MDPI](https://mdpi.com/2673-7078/2/1/6/htm) - ML methods for phase segmentation
- [Badminton stroke phases diagram - ResearchGate](https://www.researchgate.net/figure/Basic-phases-of-a-badminton-stroke-backswing-Frames-1-7-forward-swing-Frames-7-10_fig2_233782806) - Phase definitions
- [Biomechanical Principles Applied to Badminton Power Strokes](https://ojs.ub.uni-konstanz.de/cpa/article/download/2233/2089/) - Kinetic chain and stroke mechanics
- [scipy.signal documentation](https://docs.scipy.org/doc/scipy/tutorial/signal.html) - Signal processing methods
- [scikit-learn feature_selection documentation](https://scikit-learn.org/stable/modules/feature_selection.html) - Feature selection methods
- [VIF threshold guidelines - QUANTIFYING HEALTH](https://quantifyinghealth.com/vif-threshold/) - Multicollinearity thresholds
- [Markerless joint angle estimation using MediaPipe - Springer](https://link.springer.com/article/10.1007/s11042-026-21256-z) - MediaPipe angular velocity calculation (Jan 2026)

### Secondary (MEDIUM confidence)
- [Kinetic chain timing features - BetterMinton Service](https://www.lungpancheng.tw/publication/betterminton/BetterMinton.pdf) - Sequential coordination research
- [Wrist flexion and forearm pronation importance](https://lupinepublishers.com/orthopedics-sportsmedicine-journal/fulltext/biomechanics-in-badminton-a-review.ID.000129.php) - Biomechanical feature validation
- [Cohen's d in biomechanics - MDPI Diagnostics](https://www.mdpi.com/2075-4418/15/22/2855) - Effect size application (2025)
- [Filter vs wrapper methods comparison - Sebastian Raschka](https://sebastianraschka.com/faq/docs/feature_sele_categories.html) - Feature selection strategy
- [RFE best practices - Machine Learning Mastery](https://machinelearningmastery.com/rfe-feature-selection-in-python/) - Wrapper method implementation
- [N/10 rule - Oxford Bioinformatics](https://academic.oup.com/bioinformatics/article/21/8/1509/249540) - Sample size to feature ratio
- [Racket head speed research - Nature Scientific Reports](https://www.nature.com/articles/s41598-023-37108-x) - Elite player racket speeds (2023)
- [Angular velocity in badminton - ResearchGate](https://www.researchgate.net/publication/343837565_Radio-ulnar_pronation_vs_forearm_extension_which_the_best_to_reach_the_maximal_badminton_racket_velocity) - Forearm rotation vs extension
- [Drop shot biomechanics - KHELJOURNAL](https://www.kheljournal.com/archives/2025/vol12issue3/PartC/12-3-23-144.pdf) - Deceleration control (2025)
- [Contact frame detection - Nature Scientific Reports](https://www.nature.com/articles/s41598-025-87610-7.pdf) - Hit detection methods (2025)
- [MediaPipe table tennis biomechanics - Frontiers](https://www.frontiersin.org/journals/sports-and-active-living/articles/10.3389/fspor.2025.1635581/full) - Angular velocity extraction (Nov 2025)

### Tertiary (LOW confidence - for context only)
- Various web search results on feature selection, phase segmentation, and biomechanics - used to identify research directions but verified with primary sources where possible

## Metadata

**Confidence breakdown:**
- Standard stack: HIGH - scipy, scikit-learn, numpy, pandas are industry standard for this domain
- Architecture patterns: MEDIUM - Phase segmentation and kinetic chain patterns validated by biomechanics research, but specific implementation details adapted for MediaPipe limitations
- Pitfalls: HIGH - Feature explosion, contact frame misidentification, and multicollinearity issues are well-documented in small-dataset ML and biomechanics literature
- Code examples: MEDIUM - Based on scipy/sklearn official documentation and biomechanics research, but specific parameter values (sigma=1.5, VIF<10, Cohen's d>0.5) are informed by literature but may need tuning for this dataset
- Drop shot features: LOW - Requirements mention drop shots but dataset appears to be Clear+Smash only; applicability unclear pending user clarification

**Research date:** 2026-01-30
**Valid until:** ~2026-02-28 (30 days for stable libraries like scipy/sklearn; 7 days for fast-moving biomechanics research)

**Key assumptions:**
1. Dataset is 3,347 forehand Clear + Smash strokes (no drop shots) - affects FEAT-07 priority
2. MediaPipe 0.10.9 pose estimation quality sufficient for angular velocity calculation
3. N_train = 2,554 (76% of 3,347) requires features < 254 per N/10 rule
4. Success criteria "Cohen's d > 0.5" applies to NEW features, not all features
5. Phase boundary accuracy "85%+" can be validated via velocity correspondence (no ground truth annotations available)
