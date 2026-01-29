# Feature Landscape: Coach-Informed Biomechanical Features

**Domain:** Badminton stroke analysis and coaching feedback
**Researched:** 2026-01-29
**Milestone:** v1.1 - Coach-informed feature engineering

---

## Executive Summary

This research identifies biomechanical features that badminton coaches use to analyze stroke technique, based on recent sports science literature (2024-2025) and established coaching practice. Current system extracts 427 features from MediaPipe keypoints. Research reveals critical gaps in kinetic chain sequencing, phase-specific analysis, and stroke-differentiating features that should be added for improved ML classification and coaching feedback.

**Key Finding:** Coaches focus on sequential coordination patterns (kinetic chain timing), wrist/forearm orientation at contact, and phase-specific metrics rather than just static angles and velocities. Current features miss temporal sequencing, muscle activation patterns, and stroke-specific discriminators.

**Confidence Level:** HIGH for table stakes features (grounded in multiple peer-reviewed studies), MEDIUM for advanced features (emerging research), LOW for some stroke-specific features (limited literature on Drop/Drive/Net shots).

---

## Table Stakes Features

Features that coaches expect for basic stroke analysis. Missing these = incomplete technique assessment.

### 1. Kinetic Chain Sequencing (MISSING - CRITICAL)

**What coaches look for:** Sequential activation from ground → legs → trunk → shoulder → elbow → wrist

**Why expected:** Elite players generate power through proximal-to-distal energy transfer. Timing matters more than peak values.

**Current gap:** System captures peak velocities but not sequential timing relationships between body segments.

| Feature | Why Critical | Complexity | Implementation Notes |
|---------|--------------|------------|---------------------|
| Hip rotation timing | Initiates kinetic chain | Medium | Track hip_center rotation relative to shoulders, measure time-to-peak |
| Trunk rotation timing | Transfers energy from lower to upper body | Medium | Shoulder_rotation peak timing relative to hip rotation |
| Shoulder internal rotation timing | Drives racket acceleration | High | Requires 3D joint angle calculation, peak timing relative to contact |
| Elbow extension timing | Final acceleration phase | Medium | Elbow angle derivative, peak extension velocity timing |
| Wrist pronation timing | Contact point control | High | Forearm_vertical_angle derivative, timing relative to contact frame |
| Sequential coordination index | Overall kinetic chain efficiency | High | Quantify time lag between segment peaks (ideal: proximal→distal cascade) |

**Research support:**
- [Muscle synergy study (2025)](https://pmc.ncbi.nlm.nih.gov/articles/PMC12170632/) found elite players show coordinated muscle activation: scapular stabilizers → power generators → deceleration controllers
- [Biomechanical principles](https://ojs.ub.uni-konstanz.de/cpa/article/download/2233/2089/) emphasize sequential joint action for power generation

**Actionability:** HIGH - Provides specific timing feedback ("Trunk rotation peaks too early, 0.05s before shoulder rotation")

---

### 2. Phase-Based Feature Extraction (PARTIALLY MISSING)

**What coaches look for:** Different metrics matter in different stroke phases

**Why expected:** Preparation, backswing, forward swing, contact, and follow-through have distinct biomechanical requirements.

**Current gap:** System aggregates across entire stroke (mean, max, std) without phase-specific analysis.

| Feature Category | Phase | Why Important | Complexity |
|------------------|-------|---------------|------------|
| Preparation stance | Pre-backswing | Balance, ready position | Low |
| Backswing range | Backswing | Energy storage (stretch-shortening cycle) | Medium |
| Forward acceleration | Forward swing | Power generation phase | Medium |
| Contact point precision | Impact | Determines shot trajectory | Low |
| Follow-through control | Post-contact | Deceleration, injury prevention | Medium |

**Phase segmentation approach:**
- **Preparation:** First 20% of frames (stance setup)
- **Backswing:** Frames until wrist reaches lowest/rearmost point
- **Forward swing:** Backswing end → peak wrist velocity frame
- **Contact:** Peak velocity frame ± 2 frames (impact window)
- **Follow-through:** Post-contact to end

**Specific phase features to add:**

| Feature | Phase | Calculation | Why Coaches Need It |
|---------|-------|-------------|---------------------|
| Backswing depth | Backswing | Max wrist distance behind shoulder | Deeper backswing = more power storage |
| Backswing duration | Backswing | Time from start to lowest point | Too fast = rushed, too slow = loss of momentum |
| Acceleration duration | Forward swing | Time from backswing end to peak velocity | Measures explosive power generation |
| Elbow angle at contact | Contact | Elbow angle at peak velocity frame | Optimal: 160-180° for full extension |
| Wrist angle at contact | Contact | Forearm_vertical_angle at impact | Clear: upward (~45-70°), Smash: downward (~110-140°) |
| Follow-through duration | Follow-through | Post-contact to end | Adequate deceleration prevents injury |
| Deceleration magnitude | Follow-through | Negative acceleration post-contact | Smooth deceleration indicates control |

**Research support:**
- [Stroke phases study](https://www.researchgate.net/figure/Basic-phases-of-a-badminton-stroke-backswing-Frames-1-7-forward-swing-Frames-7-10_fig2_233782806) documents standard five-phase model
- Phase-specific analysis standard in coaching literature

**Actionability:** HIGH - Phase-specific feedback more actionable than aggregate metrics

---

### 3. Contact Point Features (PARTIALLY PRESENT)

**What coaches look for:** Height, forward reach, lateral position of racket-shuttle contact

**Why expected:** Contact point determines shot trajectory, power transfer efficiency, and shot type.

**Current state:** System has `contact_point` (wrist height relative to head) but missing other dimensions.

| Feature | Current Status | Why Coaches Need It | Complexity |
|---------|----------------|---------------------|------------|
| Contact height (Y-axis) | ✓ PRESENT (`r_wrist_height_from_head`) | Higher = more downward angle for smash | Low |
| Contact forward reach (Z-axis) | ✓ PRESENT (`r_wrist_depth_from_hip`) | Forward contact = more power | Low |
| Contact lateral position (X-axis) | ✓ PRESENT (`r_wrist_lateral`) | Centered contact = better control | Low |
| Contact point consistency (std dev) | ✗ MISSING | Low variance = better technique | Low |
| Optimal contact zone indicator | ✗ MISSING | Binary: in/out of ideal zone | Medium |

**Enhancement needed:** Aggregate contact point features across phases, measure consistency.

**Stroke-specific optimal zones:**
- **Smash:** High (above head 10-20cm), forward (30-40cm in front of body)
- **Clear:** High (above head 5-15cm), slightly behind body (0-10cm forward)
- **Drop:** Similar to smash but requires deception (same contact zone as smash)
- **Drive:** Shoulder height, well forward (40-50cm)
- **Net:** Low (chest to waist height), close to body

**Research support:**
- [Smash biomechanics comparison](https://www.mdpi.com/2076-3417/13/22/12488) emphasizes contact point height differences between skill levels
- Contact point training emphasized in [coaching guides](https://badmintonprogress.app/en/technical/badminton-techniques-complete-guide/)

**Actionability:** HIGH - Visual/spatial feedback easy to understand

---

### 4. Racket Head Speed & Acceleration (PRESENT but needs enhancement)

**What coaches look for:** Peak racket speed, acceleration timing, deceleration control

**Why expected:** Fundamental indicator of power generation and technique efficiency.

**Current state:** System tracks `max_velocity` (wrist speed) as proxy for racket head speed.

| Feature | Current Status | Enhancement Needed | Complexity |
|---------|----------------|-------------------|------------|
| Peak racket head speed | ✓ PRESENT (via wrist velocity) | Estimate racket tip speed using forearm length + angular velocity | Medium |
| Time to peak speed | ✓ PRESENT (`peak_velocity_timing`) | Good | Low |
| Acceleration magnitude | ✓ PRESENT (acceleration features) | Phase-specific acceleration | Low |
| Deceleration control | ✗ MISSING | Follow-through deceleration smoothness | Medium |
| Speed consistency | ✗ MISSING | Std dev of peak speeds across similar strokes | Low |

**Enhancement formula for racket head speed:**
```
racket_tip_velocity = wrist_velocity + (forearm_angular_velocity × racket_length)
```
Where:
- `racket_length` ≈ 0.67m (standard badminton racket)
- `forearm_angular_velocity` = angular velocity of forearm rotation (rad/s)

**Benchmark values (from literature):**
- **Elite smash:** 118 m/s shuttle speed, 61-71 m/s racket head speed
- **Clear:** ~75% of smash speed (45-53 m/s racket head speed)
- **Drop:** Similar to clear but with controlled deceleration
- **Drive:** 30-40 m/s racket head speed

**Research support:**
- [Racket head speed study](https://www.nature.com/articles/s41598-023-37108-x): Elite Malaysian players achieve 61.2 m/s average, 68.5 m/s peak
- [Joint moments study](https://www.mdpi.com/2076-3417/12/2/880): Racket speed correlated with shoulder internal rotation moments

**Actionability:** MEDIUM - Speed targets useful but requires context (technique vs. strength)

---

### 5. Wrist/Forearm Orientation Features (PRESENT - needs expansion)

**What coaches look for:** Wrist angle at contact, forearm pronation, wrist flexion/extension

**Why expected:** Wrist orientation determines shot direction (up = clear, down = smash) and power transfer.

**Current state:** System has several forearm features but missing wrist-specific angles.

| Feature | Current Status | Why Critical | Complexity |
|---------|----------------|--------------|------------|
| Forearm vertical angle | ✓ PRESENT (`r_forearm_vertical_angle`) | Differentiates clear (upward) from smash (downward) | Low |
| Forearm pitch | ✓ PRESENT (`r_forearm_pitch`) | Elevation angle at contact | Low |
| Arm plane pronation | ✓ PRESENT (`r_arm_plane_pronation`) | Pronation = palm-down power generation | Medium |
| Wrist flexion angle | ✗ MISSING | Requires wrist landmark (not in MediaPipe) | High |
| Wrist ulnar/radial deviation | ✗ MISSING | Lateral wrist angle | High |
| Forearm rotation velocity | ✗ MISSING | Rate of pronation (critical for power) | Medium |

**Limitation:** MediaPipe Pose provides wrist landmark (single point), not wrist orientation. Cannot directly measure wrist flexion without hand landmarks.

**Workaround:** Infer wrist angle from elbow-wrist-hand vector (if hand landmark available in MediaPipe Holistic).

**Research support:**
- [Biomechanics review](https://lupinepublishers.com/orthopedics-sportsmedicine-journal/fulltext/biomechanics-in-badminton-a-review.ID.000129.php): 53% of racket head velocity from forearm rotation
- [Wrist pronation coaching](https://beastbadminton.com/badminton-wrist-training/): Pronation vs. supination critical for shot power

**Actionability:** HIGH - Wrist angle at contact highly actionable coaching cue

---

### 6. Lower Limb Features (MISSING - MODERATE IMPORTANCE)

**What coaches look for:** Footwork quality, balance, weight transfer, jump height (jump smash)

**Why expected:** Power generation starts from ground, proper footwork enables optimal upper body mechanics.

**Current gap:** System focuses on upper body (arm/shoulder/trunk) but minimal lower limb analysis.

| Feature | Why Important | Complexity | Implementation Notes |
|---------|---------------|------------|---------------------|
| Ankle dorsiflexion | Injury prevention, power generation | Low | Ankle-knee-hip angle in sagittal plane |
| Knee flexion at preparation | Loading for explosive movement | Low | Knee angle at preparation phase |
| Hip internal rotation | Trunk rotation initiation | Medium | Hip rotation timing relative to trunk |
| Foot pressure distribution | Balance and stability | N/A | Requires pressure sensors (out of scope for MediaPipe) |
| Jump height (jump smash) | Vertical reach for higher contact point | Medium | Estimate from hip center Y-displacement |
| Landing knee angle | Injury risk assessment | Low | Knee angle at landing frame |
| Weight transfer (rear to front foot) | Power generation | High | Requires bilateral foot tracking |

**Research support:**
- [Lower limb biomechanics study](https://pmc.ncbi.nlm.nih.gov/articles/PMC6348812/): Professionals show greater ankle dorsiflexion, knee abduction
- [Core stability study](https://www.researchgate.net/publication/321651217_The_Effect_of_Core_Stability_Training_on_Dynamic_Balance_and_Smash_Stroke_Performance_in_Badminton_Players): Balance critical for stroke performance

**Actionability:** MEDIUM - Footwork feedback valuable but harder to visualize than upper body

---

## Differentiators

Advanced features that improve analysis quality beyond basic coaching. Not expected but valued.

### 1. Stroke Phase Transition Timing

**Value proposition:** Elite players have consistent, efficient transitions between phases

**What it measures:** Duration and smoothness of phase transitions (backswing→forward swing, forward swing→contact, contact→follow-through)

**Why valuable:** Detects rushed or hesitant technique, indicates timing issues

| Feature | Calculation | Why Differentiating | Complexity |
|---------|-------------|---------------------|------------|
| Backswing-to-forward transition | Frame count where acceleration changes from negative to positive | Smooth transition = better rhythm | Medium |
| Contact anticipation | How far before peak velocity does elbow reach full extension | Elite players extend early, amateurs late | Medium |
| Follow-through smoothness | Standard deviation of deceleration rate | Jerky deceleration = poor control or injury risk | Low |

**Actionability:** HIGH - "Your transition from backswing to forward swing is too abrupt (0.08s, target: 0.12-0.15s)"

---

### 2. Symmetry Analysis (Left vs. Right Arm)

**Value proposition:** Balanced body mechanics indicate better overall technique

**What it measures:** Difference in motion between dominant and non-dominant arm

| Feature | Why Valuable | Complexity |
|---------|--------------|------------|
| Non-dominant arm position | Balance and counterweight | Low |
| Bilateral shoulder rotation | Symmetric rotation indicates good trunk engagement | Medium |
| Arm extension asymmetry | Large differences indicate compensation patterns | Low |

**Current state:** System tracks left arm features but doesn't compare left vs. right explicitly.

**Actionability:** MEDIUM - Useful for advanced players, less relevant for beginners

---

### 3. Movement Efficiency Metrics

**Value proposition:** Quantify "quality of movement" beyond just outcome metrics

| Feature | Calculation | Why Valuable | Complexity |
|---------|-------------|--------------|------------|
| Jerk (rate of acceleration change) | Already present but not aggregated | Smooth motion = low jerk, better technique | Low |
| Trajectory smoothness | Curvature of wrist path | Efficient motion follows smooth arc | Medium |
| Energy efficiency index | Racket speed / total body displacement | Elite players generate speed with less body movement | High |

**Research support:**
- Jerk minimization principle in motor control theory
- Smooth trajectories correlate with skill level in sports biomechanics

**Actionability:** MEDIUM - Conceptually complex for recreational players

---

### 4. Stroke Signature Analysis (Temporal Pattern Matching)

**Value proposition:** Compare entire temporal pattern to professional "signature" patterns

**What it measures:** Shape similarity of feature time-series curves (e.g., elbow angle curve, velocity curve) compared to elite players

**Implementation approach:**
- Dynamic Time Warping (DTW) between user stroke and professional reference
- Pattern correlation (Pearson R) for key feature curves
- Identify which phase deviates most from professional pattern

**Why valuable:** Goes beyond discrete metrics to capture "flow" of motion

**Complexity:** HIGH (requires DTW implementation, professional reference patterns)

**Actionability:** HIGH if visualized well - Overlay user vs. pro curves, highlight deviations

---

### 5. Deception Features (Advanced)

**Value proposition:** Differentiate between shots with similar preparation (e.g., smash vs. drop)

**What it measures:** How late in the stroke does preparation diverge between shot types

| Feature | Calculation | Why Valuable | Complexity |
|---------|-------------|--------------|------------|
| Deception point | Frame where smash/drop trajectories diverge | Late divergence = better deception | High |
| Preparation similarity | DTW distance between smash prep and drop prep | Similar prep = harder to read | High |

**Research support:**
- [Smash vs. drop study](https://www.mdpi.com/2076-3417/13/22/12488): Elite players show identical preparation until final 0.1s

**Actionability:** LOW - Useful for advanced players in competitive contexts, not beginners

---

## Anti-Features

Features to explicitly NOT implement. Common mistakes in stroke analysis systems.

### 1. Static Pose Features (No Temporal Context)

**What:** Features extracted from single frame without considering motion

**Why avoid:** Badminton is dynamic sport, static posture means little without velocity/acceleration context

**What to do instead:** Always pair static features (angles) with temporal derivatives (angular velocities)

**Example of mistake:** "Elbow angle is 145°" → meaningless without knowing if it's extending, flexing, or at peak

---

### 2. Raw Landmark Coordinates (X, Y, Z)

**What:** Using absolute pixel/normalized coordinates as features

**Why avoid:**
- Not invariant to camera angle, distance, or frame position
- No biomechanical meaning
- Hurts ML generalization

**What to do instead:** Use relative features (joint angles, distances between landmarks, body-centered coordinates)

**Current system:** ✓ GOOD - Uses relative features, not raw coordinates

---

### 3. Over-Smoothing of Trajectories

**What:** Applying heavy Gaussian smoothing (σ > 2.0) to remove all noise

**Why avoid:**
- Removes real high-frequency movements (wrist snap, rapid pronation)
- Critical timing information lost
- Can artificially improve-looking trajectories but hurt ML accuracy

**What to do instead:** Moderate smoothing (σ = 1.5, current system value is good) or adaptive smoothing based on velocity

---

### 4. Aggregate Features Only (Mean, Std, Min, Max)

**What:** Only statistical summaries without temporal sequence

**Why avoid:**
- Loses critical timing information
- Can't detect phase-specific issues
- Two completely different strokes could have same aggregate stats

**What to do instead:** Combine aggregate features with phase-specific and sequential features

**Current system:** ⚠️ MOSTLY AGGREGATE - v1.1 should add phase-specific features

---

### 5. Requiring Ground Truth Stroke Type for Analysis

**What:** System needs user to specify "Clear" or "Smash" before analysis

**Why avoid:**
- Poor UX (user may not know stroke type)
- Prevents auto-classification
- Limits deployment scenarios (batch processing)

**What to do instead:** Train classifier to detect stroke type from features, fall back to user input if confidence low

**Current system:** ⚠️ REQUIRES USER INPUT - v1.1 should improve ML classification to auto-detect

---

### 6. Binary "Good/Bad" Classification Without Actionable Feedback

**What:** System outputs "technique score: 45/100" without explaining why

**Why avoid:**
- Not actionable
- Frustrating for users
- Missed opportunity for coaching value

**What to do instead:** Always pair scores with specific, actionable feedback and drills

**Current system:** ✓ GOOD - Provides detailed FeedbackItem objects with drills

---

## Stroke-Specific Feature Considerations

Different stroke types emphasize different biomechanical features.

### Clear (Defensive and Offensive)

**Primary discriminators:**
- **Contact height:** Moderately high (5-15cm above head)
- **Forearm angle:** Upward orientation (45-70° from vertical)
- **Racket head speed:** 45-53 m/s (75% of smash)
- **Follow-through direction:** Upward and forward

**Key coaching cues:**
- Full arm extension at contact
- High contact point for trajectory
- Follow-through continues upward
- Weight transfer forward

**Current coverage:** ✓ GOOD - Existing features capture most discriminators

---

### Smash (Jump and Standing)

**Primary discriminators:**
- **Contact height:** Very high (10-20cm above head, higher for jump smash)
- **Forearm angle:** Downward orientation (110-140° from vertical)
- **Racket head speed:** 61-71 m/s (elite), 40-55 m/s (amateur)
- **Trunk rotation:** Rapid internal rotation
- **Pronation timing:** Peak pronation just before contact

**Key coaching cues:**
- Maximum arm extension
- Explosive shoulder internal rotation
- Pronation for power
- Steep downward angle

**Jump smash additions:**
- Jump height (15-30cm hip displacement)
- Landing knee angle (injury prevention)
- Mid-air trunk rotation

**Current coverage:** ✓ GOOD for standing smash, ⚠️ MISSING jump-specific features

---

### Drop Shot

**Primary discriminators:**
- **Deceptive preparation:** Nearly identical to smash until late
- **Contact height:** Similar to smash (10-20cm above head)
- **Deceleration:** Rapid deceleration just before contact (key difference)
- **Wrist angle:** Less pronation than smash, more "cutting" motion
- **Follow-through:** Shorter, more controlled than smash

**Key coaching cues:**
- Same preparation as smash (deception)
- Controlled contact with "slice" motion
- Minimize follow-through
- Racket face slightly open (compared to smash)

**Current coverage:** ⚠️ MISSING - No drop-specific features, need deceleration and deception metrics

**Research gap:** LIMITED - Few studies specifically on drop shot biomechanics

---

### Drive (Forehand and Backhand)

**Primary discriminators:**
- **Contact height:** Shoulder height (lower than overhead strokes)
- **Body position:** More upright, less trunk rotation
- **Arm angle:** Nearly horizontal at contact
- **Racket path:** Flat, parallel to ground
- **Speed:** 30-40 m/s racket head speed

**Key coaching cues:**
- Level swing path
- Quick wrist action
- Minimal backswing (fast reaction)
- Weight on front foot

**Current coverage:** ⚠️ MISSING - System designed for overhead strokes, needs drive-specific features

**Implementation notes:**
- Detect stroke type from contact height and swing plane
- Horizontal swing plane detection critical

---

### Net Shot (Spin, Tumble, Lift)

**Primary discriminators:**
- **Contact height:** Very low (chest to waist height)
- **Arm extension:** Minimal (bent elbow, close to body)
- **Wrist action:** Dominant (forearm relatively static)
- **Racket head speed:** Low (10-20 m/s for control)
- **Body position:** Lunging forward, low center of gravity

**Key coaching cues:**
- Soft touch, controlled speed
- Wrist-dominant motion
- Bent elbow for control
- Low body position

**Current coverage:** ⚠️ MISSING - Completely different biomechanics from overhead strokes

**Implementation challenges:**
- Lower body features critical (lunge depth, ankle angle)
- Fine motor control (wrist) hard to capture with MediaPipe resolution
- Short duration (0.5-1.0s vs. 1.5-2.5s for overhead strokes)

**Research gap:** LIMITED - Most literature focuses on power strokes (clear, smash)

---

## Feature Priority Matrix

Prioritization for v1.1 implementation based on impact vs. complexity.

| Feature Category | Impact on ML | Impact on Feedback | Complexity | Priority |
|------------------|--------------|-------------------|------------|----------|
| **Kinetic chain sequencing** | HIGH | HIGH | Medium-High | **P0 - CRITICAL** |
| **Phase-based features** | HIGH | HIGH | Medium | **P0 - CRITICAL** |
| **Contact point enhancement** | HIGH | HIGH | Low | **P0 - CRITICAL** |
| **Wrist/forearm velocity** | HIGH | MEDIUM | Medium | **P1 - HIGH** |
| **Racket head speed estimation** | MEDIUM | HIGH | Medium | **P1 - HIGH** |
| **Deceleration control** | MEDIUM | HIGH | Medium | **P1 - HIGH** |
| **Drop shot deception features** | HIGH | MEDIUM | Medium | **P1 - HIGH** |
| **Lower limb features** | MEDIUM | MEDIUM | Low-Medium | **P2 - MEDIUM** |
| **Jump smash features** | MEDIUM | MEDIUM | Medium | **P2 - MEDIUM** |
| **Symmetry analysis** | LOW | MEDIUM | Low | **P3 - LOW** |
| **Movement efficiency metrics** | LOW | LOW | High | **P3 - LOW** |
| **Stroke signature matching** | MEDIUM | HIGH | High | **P4 - FUTURE** |
| **Deception analysis** | LOW | LOW | High | **P4 - FUTURE** |

---

## Feature Engineering Implementation Roadmap

Recommended feature addition sequence for v1.1.

### Phase 1: Critical Discriminators (Week 1-2)

**Goal:** Add features that most differentiate Clear vs. Smash vs. Drop

1. **Phase segmentation logic**
   - Implement automatic phase detection
   - Extract phase-specific features for each existing metric

2. **Kinetic chain timing**
   - Hip rotation peak timing
   - Trunk rotation peak timing
   - Shoulder internal rotation peak timing
   - Sequential coordination index

3. **Contact frame analysis**
   - Refine contact point detection (peak velocity frame)
   - Extract all features specifically at contact frame
   - Add contact point consistency metrics

**Expected impact:** +15-20% ML classification accuracy

---

### Phase 2: Temporal Enhancement (Week 3)

**Goal:** Add velocity-based features for power and timing

1. **Angular velocities**
   - Forearm rotation velocity (pronation rate)
   - Elbow extension velocity
   - Shoulder rotation velocity

2. **Deceleration features**
   - Follow-through deceleration magnitude
   - Deceleration smoothness (jerk in follow-through)

3. **Racket head speed estimation**
   - Implement formula: wrist_velocity + (angular_velocity × racket_length)

**Expected impact:** +5-10% ML accuracy, improved power-related feedback

---

### Phase 3: Stroke-Specific Features (Week 4)

**Goal:** Enable Drop, Drive, Net shot classification and analysis

1. **Drop shot features**
   - Contact deceleration (smash accelerates through, drop decelerates)
   - Preparation similarity to smash (deception index)
   - Follow-through duration (shorter than smash)

2. **Drive detection features**
   - Swing plane angle (horizontal vs. overhead)
   - Contact height zone (shoulder level)
   - Trunk rotation magnitude (less than overhead)

3. **Net shot detection**
   - Contact height (chest-waist level)
   - Arm extension (minimal, bent elbow)
   - Stroke duration (short, 0.5-1.0s)

**Expected impact:** Enable multi-class classification (5 stroke types)

---

### Phase 4: Lower Limb & Balance (Week 5)

**Goal:** Add footwork and balance features for comprehensive analysis

1. **Lower limb angles**
   - Ankle dorsiflexion
   - Knee flexion at preparation
   - Hip internal rotation

2. **Jump smash features**
   - Jump height (hip displacement)
   - Landing knee angle
   - Mid-air trunk rotation

3. **Balance indicators**
   - Center of mass stability (hip center variation)
   - Foot separation width

**Expected impact:** +3-5% ML accuracy, richer feedback on footwork

---

## Data Collection & Labeling Considerations

To train models on new features, consider:

### New Labels Needed

| Label Type | Current Status | Need for v1.1 |
|------------|----------------|---------------|
| Stroke type (Clear/Smash) | ✓ HAS (from ShuttleSet) | Good |
| Drop shot | ✗ MISSING | CRITICAL for expansion |
| Drive | ✗ MISSING | CRITICAL for expansion |
| Net shot | ✗ MISSING | CRITICAL for expansion |
| Forehand/Backhand | ⚠️ PARTIAL (filtered to forehand) | NICE-TO-HAVE |
| Jump vs. standing | ✗ MISSING | NICE-TO-HAVE |
| Phase boundaries | ✗ MISSING | AUTO-DETECT via algorithm |

### ShuttleSet Dataset Coverage

**Current dataset (ShuttleSet):**
- 4,983 total clips
- 3,347 forehand Clear + Smash clips (used in v1.0)
- Unknown: Drop, Drive, Net shot distribution

**Action for v1.1:**
1. Re-examine full ShuttleSet dataset for Drop, Drive, Net shots
2. If insufficient, may need to collect supplementary video data
3. For MVP, can start with Clear/Smash/Drop (defer Drive/Net)

---

## Confidence Assessment

| Feature Category | Confidence Level | Rationale |
|------------------|------------------|-----------|
| Kinetic chain sequencing | **HIGH** | Multiple peer-reviewed studies (2024-2025) consistently emphasize sequential coordination |
| Phase-based features | **HIGH** | Standard model in sports biomechanics, well-documented |
| Contact point features | **HIGH** | Validated in multiple skill-level comparison studies |
| Wrist/forearm orientation | **HIGH** | Directly studied in smash biomechanics research, 53% of power generation |
| Racket head speed | **HIGH** | Well-measured in research (61-71 m/s elite smash) |
| Lower limb features | **MEDIUM** | Some research exists, but less emphasis in coaching practice for upper-level analysis |
| Drop shot features | **MEDIUM** | Limited specific research, mostly inferred from smash studies |
| Drive features | **LOW** | Very limited research on drive biomechanics |
| Net shot features | **LOW** | Minimal research, mostly coaching intuition |
| Deception features | **MEDIUM** | Some research on smash/drop similarity, but quantification difficult |

---

## Actionability for Coaching Feedback

Features ranked by how actionable they are for user improvement.

### Highly Actionable (Can be explained and practiced)

1. **Contact point height/position** - "Hit the shuttle 10cm higher, 5cm more forward"
2. **Phase timing** - "Your backswing is too fast, add 0.1s before forward swing"
3. **Elbow extension at contact** - "Extend your elbow more fully, aim for 170-180°"
4. **Wrist angle at contact** - "Angle your wrist more downward for steeper smash"
5. **Follow-through duration** - "Complete your follow-through, extend 0.2s longer"

### Moderately Actionable (Requires drill/practice)

6. **Kinetic chain timing** - "Start trunk rotation earlier, before shoulder rotation"
7. **Racket head speed** - "Increase speed through explosive pronation drills"
8. **Lower limb loading** - "Bend knees more in preparation phase for explosive power"
9. **Deceleration control** - "Smooth out your follow-through, avoid jerky stops"

### Less Actionable (Requires coaching intervention)

10. **Symmetry** - "Balance left and right arm positioning" (hard to self-correct)
11. **Movement efficiency** - "Reduce unnecessary body movement" (abstract concept)
12. **Stroke signature** - "Your elbow angle curve deviates from professional pattern" (too technical)

---

## Research Gaps & Future Work

Areas where research is limited and deeper investigation needed.

### 1. Drop Shot Biomechanics (CRITICAL GAP)

**Gap:** Limited peer-reviewed research on drop shot technique vs. smash/clear

**Impact:** Hard to define features that differentiate drop from smash (preparation nearly identical)

**Mitigation for v1.1:**
- Extract drop shots from ShuttleSet dataset (if available)
- Analyze deceleration patterns empirically
- Consult coaching videos/tutorials for qualitative features

**Future work:** Controlled study comparing smash vs. drop preparation timing

---

### 2. Drive & Net Shot Features (MODERATE GAP)

**Gap:** Most research focuses on overhead power strokes, minimal on drive/net shots

**Impact:** Feature engineering for drive/net will be more exploratory, less evidence-based

**Mitigation for v1.1:**
- Defer drive/net to v1.2 if time-constrained
- Start with drive (more common in singles play than net in dataset context)

**Future work:** Dedicated net shot analysis study with wearable sensors

---

### 3. Real-Time Phase Segmentation (TECHNICAL GAP)

**Gap:** No standard algorithm for automatic phase detection from pose keypoints alone

**Impact:** Phase boundaries must be estimated heuristically (may introduce errors)

**Mitigation for v1.1:**
- Use velocity-based heuristics (lowest point = backswing end, peak = contact)
- Validate on sample videos, iterate on algorithm
- Accept some phase boundary imprecision

**Future work:** Train LSTM to predict phase labels from pose sequences

---

### 4. MediaPipe Limitations for Fine Motor Control (TOOL GAP)

**Gap:** MediaPipe Pose doesn't track hand/finger positions (only wrist point)

**Impact:** Cannot measure actual wrist flexion, grip changes, finger pressure

**Mitigation for v1.1:**
- Use MediaPipe Holistic (includes hand landmarks) if available
- Infer wrist angle from elbow-wrist-inferred_hand_center vector
- Accept limitation for v1.1

**Future work:** Explore MediaPipe Holistic integration, or sensor-based measurement

---

## Sources

### Peer-Reviewed Research (HIGH Confidence)

- [Muscle Synergy Analysis (2025)](https://pmc.ncbi.nlm.nih.gov/articles/PMC12170632/) - Kinetic chain, sequential muscle activation
- [Biomechanical Principles for Power Strokes](https://ojs.ub.uni-konstanz.de/cpa/article/download/2233/2089/) - Fundamental biomechanics, joint sequencing
- [Novice vs. Skilled Player Comparison (2023)](https://www.mdpi.com/2076-3417/13/22/12488) - Kinematic differences, contact point
- [Lower Limb Biomechanics in Clear Strokes](https://pmc.ncbi.nlm.nih.gov/articles/PMC6348812/) - Ankle, knee, hip features
- [Racket Head Speed Study (2023)](https://www.nature.com/articles/s41598-023-37108-x) - Elite racket speeds, swingweight impact
- [MultiSenseBadminton Dataset (2024)](https://www.nature.com/articles/s41597-024-03144-z) - Multi-sensor features, stroke quality
- [Balance Training Effects (2022)](https://pmc.ncbi.nlm.nih.gov/articles/PMC9671355/) - Footwork and balance importance
- [Biomechanics in Badminton Review](https://lupinepublishers.com/orthopedics-sportsmedicine-journal/fulltext/biomechanics-in-badminton-a-review.ID.000129.php) - Comprehensive review of biomechanics

### Coaching Resources (MEDIUM Confidence)

- [Badminton Wrist Training](https://beastbadminton.com/badminton-wrist-training/) - Pronation vs. wrist snap myth
- [Common Amateur Errors](https://thebadmintonhub.com/10-common-errors-made-by-amateur-badminton-players/) - Technique mistakes
- [Badminton Techniques Guide](https://badmintonprogress.app/en/technical/badminton-techniques-complete-guide/) - Comprehensive coaching guide

### Technical Documentation (MEDIUM Confidence)

- [Stroke Phases Diagram](https://www.researchgate.net/figure/Basic-phases-of-a-badminton-stroke-backswing-Frames-1-7-forward-swing-Frames-7-10_fig2_233782806) - Five-phase model
- [ShuttleSet Dataset Paper (2023)](https://arxiv.org/abs/2306.04948) - Dataset characteristics

---

## Summary for Roadmap Creation

**Must-have features for v1.1:**
1. Phase segmentation and phase-specific features (impact: +15-20% ML accuracy)
2. Kinetic chain timing features (hip→trunk→shoulder→wrist sequence)
3. Contact frame-specific analysis (angles, position at moment of contact)
4. Deceleration and follow-through features

**Should-have features for v1.1:**
5. Angular velocity features (forearm rotation, elbow extension)
6. Drop shot discriminators (deceleration at contact)
7. Racket head speed estimation

**Nice-to-have for v1.1 (or defer to v1.2):**
8. Lower limb features (ankle, knee, hip)
9. Drive and net shot features
10. Jump smash specific features

**Future work (v2.0+):**
11. Stroke signature temporal matching
12. Deception analysis
13. MediaPipe Holistic integration for hand tracking

---

**Confidence in Roadmap:** HIGH - Clear prioritization based on research literature and impact/complexity trade-offs.
