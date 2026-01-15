# Pose Extraction Issues & Solutions

## Current Status

### Overall Statistics
- **Total clips**: 4,983
- **Successfully extracted**: 3,125 (62.7%)
- **Failed (no poses detected)**: 1,858 (37.3%)
- **Low quality (<50% valid frames)**: 495 (9.9%)

### The Problem

The pose extraction has **two separate issues**:

1. **Complete Failures (1,858 clips)**: MediaPipe detected ZERO poses
   - Status: `extraction_failed`
   - Error: "No poses detected in video"
   - These clips have 0% valid frames

2. **Low Quality (495 clips)**: Poses detected but with poor quality
   - Status: `success` but valid_percentage < 50%
   - Partial pose detection (1-49% of frames)

## Most Affected Matches

### Matches with High Failure Rates

| Match | Failed | Total | Failure Rate |
|-------|--------|-------|--------------|
| 39 | 124 | 126 | 98.4% |
| 41 | 180 | 184 | 97.8% |
| 38 | 213 | 221 | 96.4% |
| 40 | 120 | 124 | 96.8% |
| 11 | 156 | 171 | 91.2% |
| 31 | 62 | 74 | 83.8% |
| 44 | 105 | 133 | 78.9% |
| 25 | 45 | 60 | 75.0% |

**Why are these matches failing?**

Possible causes:
1. **Different camera angles** - Overhead or extreme side views
2. **Players too small in frame** - Wide court shots
3. **Poor video quality** - Heavy compression, low resolution
4. **Lighting issues** - Dark gyms, shadows
5. **MediaPipe thresholds too strict** - Default 0.5 confidence may be too high

## Solutions

### Solution 1: Diagnose the Root Cause

Run the diagnostic script to understand WHY poses aren't detected:

```bash
python src/data_processing/diagnose_failed_poses.py
```

**What it does:**
- Tests 100 sample clips from problematic matches
- Tries ULTRA-LENIENT detection (confidence=0.1)
- Determines if clips are fixable or fundamentally unsuitable

**Expected output:**
- % of clips that CAN be detected with lower thresholds
- % of clips that CANNOT be detected even with ultra-low thresholds
- Specific recommendations

### Solution 2: Robust Re-extraction

Run enhanced extraction with multiple fallback strategies:

```bash
python src/data_processing/extract_poses_robust.py
```

**What it does:**
- Reprocesses ALL failed clips (1,858) and low-quality clips (495)
- **Strategy 1**: Lower thresholds (0.5 → 0.3)
- **Strategy 2**: Frame preprocessing (CLAHE, brightness/contrast)
- **Strategy 3**: Multiple attempts, picks best result

**Expected improvements:**
- 60-80% of failures → 40-60% valid frames
- 20-40% of failures → remain unfixable
- Low quality clips → 60-80% valid frames

### Solution 3: Quality Analysis

Understand quality distribution and problematic patterns:

```bash
python src/data_processing/analyze_pose_quality.py
```

**What it does:**
- Detailed quality statistics by match
- Identifies root causes
- Creates visualizations
- Provides specific recommendations per match

## Recommended Workflow

```bash
# Step 1: Diagnose to understand the problem
python src/data_processing/diagnose_failed_poses.py

# Step 2: Review diagnosis results
# Check: outputs/reports/pose_diagnosis.csv

# Step 3: Run robust re-extraction
python src/data_processing/extract_poses_robust.py

# Step 4: Analyze improved results
python src/data_processing/analyze_pose_quality.py

# Step 5: Re-run feature engineering with improved poses
python src/data_processing/feature_engineering.py
```

## Expected Outcomes

### Best Case Scenario
- Fix ~60% of failed clips (1,100 clips)
- Improve all low-quality clips to >50% valid frames
- Final success rate: ~85% (4,200+ usable clips)

### Realistic Scenario
- Fix ~40% of failed clips (750 clips)
- Improve 70% of low-quality clips
- Final success rate: ~78% (3,900+ usable clips)

### Worst Case Scenario
- Some matches (38, 39, 40, 41) may be fundamentally unsuitable
- These matches may have unusual camera setups
- Option: Exclude these matches from training dataset

## Decision Point

After running diagnostics, you'll need to decide:

### Option A: Keep All Clips (Recommended)
- Run robust re-extraction
- Accept some clips may remain low quality
- Use data augmentation to handle class imbalance if needed

### Option B: Exclude Problematic Matches
- If certain matches have >90% failure rate
- AND diagnosis shows they can't be fixed
- THEN exclude those entire matches
- This would reduce dataset size but improve quality

### Option C: Mixed Approach
- Reprocess all clips with robust extraction
- After reprocessing, exclude individual clips that still have <30% valid frames
- Maintains maximum dataset size while ensuring minimum quality

## Technical Details

### Why MediaPipe Fails

MediaPipe Pose works best when:
- ✅ Person is clearly visible
- ✅ Standard camera angles (front/side view)
- ✅ Good lighting and contrast
- ✅ Reasonable person size in frame (>100px height)

MediaPipe struggles with:
- ❌ Overhead/bird's eye camera angles
- ❌ Very distant/small players (<100px)
- ❌ Heavy motion blur
- ❌ Poor lighting/contrast
- ❌ Extreme occlusions

### Threshold Tuning

**Default thresholds (extract_poses.py):**
- min_detection_confidence: 0.5
- min_tracking_confidence: 0.5

**Fallback thresholds (extract_poses_robust.py):**
- min_detection_confidence: 0.3
- min_tracking_confidence: 0.3

**Ultra-lenient (diagnose_failed_poses.py):**
- min_detection_confidence: 0.1
- min_tracking_confidence: 0.1

Lower thresholds = more detections but potentially lower quality poses

## Next Steps

1. **Run diagnosis first** to understand the scope
2. **Review results** before deciding on approach
3. **Run robust extraction** for fixable clips
4. **Make informed decision** about problematic matches
5. **Proceed with feature engineering** using best available data

## Questions to Answer

1. What % of failed clips can be fixed with lower thresholds?
2. Are matches 38, 39, 40, 41 fundamentally different (camera setup)?
3. Should we exclude entire matches or individual clips?
4. What's the minimum acceptable quality threshold for training?

Run the diagnostic script to get answers to these questions.
