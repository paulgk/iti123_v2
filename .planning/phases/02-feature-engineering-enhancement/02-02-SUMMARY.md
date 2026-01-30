---
phase: 02-feature-engineering-enhancement
plan: 02
subsystem: feature-engineering
tags: [kinetic-chain, contact-frame, intent-window, biomechanics, coaching-research, SIS]

# Dependency graph
requires:
  - phase: 02-feature-engineering-enhancement
    plan: 01
    provides: Phase segmentation, contact frame detection, intent window calculation
provides:
  - Kinetic chain timing features (hip->trunk->shoulder->elbow->wrist sequential delays)
  - Contact frame biomechanical features (forearm angle, wrist-elbow position, arm extension, body posture)
  - Intent window features ([contact-5:contact-2] discriminative features)
  - Smash Intent Score (SIS) formula with coach-validated weights
  - Comprehensive P0 feature test suite
affects: [02-03-phase-specific-features, feature-extraction-pipeline, ML-training]

# Tech tracking
tech-stack:
  added: []
  patterns: [kinetic-chain-timing, intent-window-analysis, coach-validated-SIS-formula]

key-files:
  created:
    - src/data_processing/kinetic_chain_features.py
    - src/data_processing/contact_frame_features.py
    - tests/test_p0_features.py
  modified: []

key-decisions:
  - "Kinetic chain timing measured ONLY within forward swing phase (not full sequence) - prevents preparation peak contamination"
  - "Intent window [contact-5:contact-2] uses get_intent_window from phase_segmentation.py for consistency"
  - "SIS formula weights: 0.35 elbow lead, 0.30 pronation, 0.15 non-racket arm, 0.10 torso, 0.10 COM"
  - "SIS thresholds: >=0.65 smash, 0.40-0.65 deceptive, <0.40 clear"
  - "Edge case handling: missing landmarks return NaN, contact<5 uses available frames"
  - "Auto-detect handedness from wrist height patterns for segment selection"

patterns-established:
  - "Kinetic chain pattern: smooth positions → calculate velocities → find peaks in forward swing only → calculate sequential delays"
  - "Contact frame pattern: extract biomechanical features at single critical frame"
  - "Intent window pattern: extract temporal features from discriminative window, aggregate with mean/max"
  - "SIS pattern: normalize components to 0-1, weighted sum with coach-validated weights, threshold-based classification"

# Metrics
duration: 7min
completed: 2026-01-30
---

# Phase 02 Plan 02: Kinetic Chain Timing and Contact Frame Features Summary

**Kinetic chain timing (hip->wrist sequential delays), contact frame biomechanics, intent window analysis with coach-validated Smash Intent Score (SIS)**

## Performance

- **Duration:** 7 min 10 sec
- **Started:** 2026-01-30T15:20:25Z
- **Completed:** 2026-01-30T15:27:35Z
- **Tasks:** 3
- **Files created:** 3

## Accomplishments

- Implemented kinetic chain timing features measuring sequential activation delays through hip->trunk->shoulder->elbow->wrist
- Contact frame feature extraction at peak velocity with 8 discriminative features (forearm angle, wrist-elbow position, arm extension, body posture)
- Intent window features from [contact-5:contact-2] frames capturing pre-contact body loading (most discriminative moment)
- Smash Intent Score (SIS) formula with coach-validated weights (35% elbow lead, 30% pronation, 15% non-racket arm, 10% torso, 10% COM)
- Comprehensive test suite with 13 passing tests covering all P0 features, edge cases, and integration

## Task Commits

Each task was committed atomically:

1. **Task 1: Implement kinetic chain timing features** - `a479f1f` (feat)
   - Created kinetic_chain_features.py with sequential timing extraction
   - Measures peak velocity timing ONLY within forward swing phase (critical requirement)
   - Coordination efficiency score based on sequence validity
   - Auto-detect dominant side for segment selection
   - Gaussian smoothing (sigma=1.5) to reduce MediaPipe jitter

2. **Task 2: Implement contact frame and intent window features** - `2facd7b` (feat)
   - Created contact_frame_features.py with contact and intent window extraction
   - Contact frame features: forearm_vertical_angle (Clear>90deg, Smash<90deg), wrist_elbow_vertical, elbow_extension, arm_extension, torso_lean, shoulder_rotation
   - Intent window features: elbow_forward (PRIMARY 35%), forearm_rotation_vel (PRIMARY 30%), nonracket_arm_drop (SECONDARY 15%), torso_rotation_vel (TERTIARY 10%), com_velocity_y (TERTIARY 10%)
   - SIS formula with normalization and threshold-based classification
   - Uses get_intent_window from phase_segmentation.py for consistency

3. **Task 3: Create P0 feature tests** - `6e0c028` (test)
   - 13 comprehensive tests covering kinetic chain timing, contact frame, intent window, SIS
   - Synthetic pose fixtures with sequential kinetic chain activation
   - Tests verify forward swing phase-only timing measurement
   - Contact frame and intent window edge case handling (missing landmarks, contact<5)
   - SIS formula and threshold validation
   - All tests passing

## Files Created/Modified

- `src/data_processing/kinetic_chain_features.py` - Kinetic chain timing feature extraction with sequential delay measurement
- `src/data_processing/contact_frame_features.py` - Contact frame and intent window feature extraction with SIS formula
- `tests/test_p0_features.py` - Comprehensive P0 feature test suite (13 tests, all passing)

## Decisions Made

1. **Kinetic chain timing ONLY in forward swing phase**
   - Rationale: CRITICAL coaching insight from RESEARCH.md. Measuring across full sequence finds preparation movement peaks, not power generation sequence. This is a documented pitfall.
   - Impact: Ensures kinetic chain features capture actual power transfer, not preparation loading. Prevents negative time deltas and invalid sequences.

2. **Intent window [contact-5:contact-2] using get_intent_window**
   - Rationale: Coaching research shows this is the MOST discriminative window (pre-contact body loading reveals stroke intent). Use phase_segmentation.py function for consistency.
   - Impact: Focused feature extraction on highest-value frames. Consistent with phase segmentation module.

3. **Smash Intent Score (SIS) with coach-validated weights**
   - Rationale: Formula validated by domain expert with specific weight distribution: 35% elbow lead (PRIMARY), 30% pronation (PRIMARY), 15% non-racket arm (SECONDARY), 10% torso, 10% COM (TERTIARY)
   - Thresholds: >=0.65 smash, 0.40-0.65 deceptive, <0.40 clear (based on coaching research reference values)
   - Impact: Provides interpretable smash-likelihood score grounded in biomechanics, not just data-driven black box

4. **Edge case handling with graceful degradation**
   - Rationale: Real-world pose data may have missing landmarks (occlusions) or short sequences (contact<5 frames)
   - Approach: Return NaN for affected features (not exceptions), use available frames when window truncated
   - Impact: Robustness to noisy/incomplete data, allows downstream processing to continue

5. **Auto-detect handedness from wrist height**
   - Rationale: Dataset contains both left and right-handed players. Racket-holding hand is typically higher (lower Y in MediaPipe) during overhead strokes.
   - Impact: Eliminates need for manual handedness annotation, enables automatic segment selection for kinetic chain

## Deviations from Plan

None - plan executed exactly as written.

All features extracted as specified:
- Kinetic chain: 7 features (4 sequential delays + total + valid flag + efficiency)
- Contact frame: 8 features (forearm orientation, wrist-elbow position, arm extension, body posture)
- Intent window: 6 features (elbow lead, pronation vel, non-racket arm, torso rotation, COM velocity)
- SIS: 1 composite score + classification hint + 5 component scores

All edge cases handled as planned.

## Issues Encountered

**Issue 1: MediaPipe landmark index confusion in tests**
- **Problem:** Test fixtures initially used wrong landmark indices (assumed right_wrist=15, but actually =16)
- **Root cause:** MediaPipe has right_wrist=16, left_wrist=15 (counterintuitive ordering)
- **Resolution:** Fixed all test fixtures to use LANDMARKS dictionary instead of hardcoded indices
- **Impact:** Tests now use correct landmarks, all 13 tests passing

**Issue 2: Numpy boolean vs Python boolean in assertions**
- **Problem:** `assert features['chain_sequence_valid'] is True` failed because numpy returns np.True_, not Python True
- **Root cause:** NumPy boolean types don't pass identity (`is`) checks with Python builtin True
- **Resolution:** Changed assertions from `is True` to `== True`
- **Impact:** Tests pass correctly with numpy boolean returns

**Issue 3: Synthetic pose fixture peak timing**
- **Problem:** Initial fixture had peaks outside forward swing phase [8,20], causing test failures
- **Root cause:** Peaks at frames 10-18, but forward swing started at frame 8, so hip peak at 10 was after start
- **Resolution:** Adjusted peak timings to 11-19 to ensure all peaks within forward swing [8,20]
- **Impact:** Kinetic chain timing tests validate correctly with sequential peaks in correct phase

## User Setup Required

None - no external service configuration required.

## Next Phase Readiness

**Ready for:**
- Plan 02-03: Phase-specific feature extraction (depends on kinetic chain and contact frame modules)
- Plan 02-04: Angular velocity features (can build on kinetic chain velocity calculation patterns)
- Feature integration: P0 features ready to integrate with feature_engineering_v2.py

**Key outputs for downstream plans:**
- `extract_kinetic_chain_timing()` returns 7 kinetic chain features
- `extract_contact_frame_features()` returns 8 contact frame features
- `extract_intent_window_features()` returns 6 intent window features
- `calculate_smash_intent_score()` returns SIS score + classification + components
- All functions handle edge cases (missing landmarks, short sequences) gracefully

**Expected impact:**
- 15-20% accuracy boost based on coaching research (P0 features)
- Kinetic chain timing: medium-large effect size expected (Cohen's d > 0.5)
- Contact frame features: large effect size expected (d > 0.8) per research
- SIS score: interpretable composite metric for model transparency

**Blockers/Concerns:**
None - all P0 features implemented and tested successfully.

**Integration notes:**
- P0 features add ~20 features (7 kinetic + 8 contact + 6 intent - 1 SIS overlap)
- Current v2 feature count: 308-315 features
- After P0: ~328-335 features (still need feature selection to reach <254 target)
- Must apply filter methods (Cohen's d, VIF) before wrapper methods (RFECV)

---
*Phase: 02-feature-engineering-enhancement*
*Completed: 2026-01-30*
