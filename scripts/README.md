# Phase 2 Validation Scripts

Scripts to validate Phase 2 (Feature Engineering Enhancement) completion.

## Quick Start

### Full Validation Suite

Run all validations except feature selection:

```bash
python scripts/validate_phase2.py --sample-size 100
```

### Feature Selection Pipeline

Run the two-stage feature selection pipeline:

```bash
python scripts/run_feature_selection.py
```

**Options:**
- `--sample-size N` - Process N samples (default: all)
- `--target-features N` - Maximum features to select (default: 254)
- `--skip-extraction` - Use cached features if available
- `--verbose` - Show detailed progress

**Output:**
- `data/processed/features_v3/selected_features.json` - Selected feature manifest
- `outputs/reports/feature_selection_report.md` - Detailed selection report
- `outputs/feature_selection_results.pkl` - Full results for analysis

### Running in Colab Enterprise

```bash
# SSH into Colab or use terminal in Colab UI
cd /content/iti123_v2

# Activate Python 3.10 environment
source colab_venv/bin/activate

# Sync data from GCS
bash scripts/sync_from_gcs.sh

# Run validation
python scripts/validate_phase2.py --sample-size 200

# Run feature selection
python scripts/run_feature_selection.py --verbose

# Sync results back to GCS
bash scripts/sync_to_gcs.sh
```

## Validation Checklist

### 1. Phase Segmentation Boundary Accuracy ✓
**Target:** 85%+ samples pass biomechanical validation checks

**Run:**
```bash
python scripts/validate_phase2.py --sample-size 100
```

**Checks:**
- Minimum phase durations (>= 3 frames)
- Contact position (30-80% of sequence)
- Forward swing duration (20-50%)
- Sequential ordering (prep < backswing < forward < contact < follow)

**Expected:** ≥85% pass rate

---

### 2. Kinetic Chain Feature Effect Sizes ✓
**Target:** Each kinetic chain feature shows Cohen's d > 0.5

**Run:**
```bash
python scripts/validate_phase2.py --sample-size 100
```

**Features tested:**
- `hip_to_trunk_delay`
- `trunk_to_shoulder_delay`
- `shoulder_to_elbow_delay`
- `elbow_to_wrist_delay`
- `hip_to_wrist_total`
- `coordination_efficiency`

**Expected:** All features show medium+ effect (d > 0.5)

---

### 3. Feature Selection Pipeline Execution ✓
**Target:** Final count < 254 features, all with Cohen's d > 0.5

**Run:**
```bash
python scripts/run_feature_selection.py
```

**Pipeline stages:**
1. **Zero variance filter** - Remove constant features
2. **Cohen's d filter** - Keep features with d ≥ 0.5
3. **VIF filter** - Remove multicollinear features (VIF < 10)
4. **RFECV wrapper** - Select optimal subset via cross-validation

**Expected:**
- Initial: ~360-367 features
- After Cohen's d: ~200-300 features
- After VIF: ~150-250 features
- Final (RFECV): <254 features
- CV F1 score: Report for baseline

---

### 4. V2 Backward Compatibility ✓
**Target:** V2 models load and predict without errors

**Run:**
```bash
python scripts/validate_phase2.py
```

**Checks:**
- V2 extraction returns ~427 features
- V2 features exclude SIS (v3-only)
- V3 extraction includes SIS
- Existing v2 models (if present) load and predict

**Expected:** Version gating works, v2 models unaffected

---

## Expected Output

### validate_phase2.py

```
============================================================
PHASE 2 VALIDATION SUITE
============================================================

VALIDATION 1: Phase Segmentation Boundary Accuracy
  Samples tested: 100
  Validation pass rate: 87.0%
  Target: 85%
  Status: ✓ PASS

VALIDATION 2: Kinetic Chain Feature Effect Sizes
  hip_to_trunk_delay                       d= 0.623 (medium    ) ✓
  trunk_to_shoulder_delay                  d= 0.581 (medium    ) ✓
  shoulder_to_elbow_delay                  d= 0.712 (medium    ) ✓
  elbow_to_wrist_delay                     d= 0.834 (large     ) ✓
  hip_to_wrist_total                       d= 0.905 (large     ) ✓
  coordination_efficiency                  d= 0.547 (medium    ) ✓
  Status: ✓ PASS - All features meet Cohen's d > 0.5

VALIDATION 4: V2 Backward Compatibility
  ✓ V2 feature extraction works
    Extracted 427 features
    Expected ~427 features: ✓
    V2 excludes SIS (v3-only): ✓
  ✓ V3 feature extraction works
    Extracted 361 features
    Has SIS: ✓
  Status: ✓ PASS

VALIDATION SUMMARY
1. Phase segmentation (85%+ accuracy):    ✓ PASS
2. Kinetic chain effect sizes (d > 0.5):  ✓ PASS
3. Feature selection (<254 features):     ℹ️  SKIP
4. V2 backward compatibility:             ✓ PASS

✓ ALL VALIDATIONS PASSED
```

### run_feature_selection.py

```
============================================================
FEATURE SELECTION PIPELINE RUNNER
============================================================

Step 1: Loading metadata and extracting features
  Loaded metadata: 3347 samples
    Clear: 1689
    Smash: 1658
  ✓ Extracted features from 3347 samples
    Feature dimensions: (3347, 361)

Step 2: Running feature selection pipeline
  Initial features: 361
  After zero-variance:        361 (-0)
  After Cohen's d >= 0.5:     243 (-118)
  After VIF < 10:             198 (-45)
  After RFECV (final):        187 (-11)

  Final Feature Count: 187
  Target Met: ✓ PASS
  CV F1 Score: 0.7823

Step 3: Saving results
  ✓ Saved feature manifest: data/processed/features_v3/selected_features.json
  ✓ Saved selection report: outputs/reports/feature_selection_report.md
  ✓ Saved full results: outputs/feature_selection_results.pkl

FEATURE SELECTION COMPLETE ✓
```

## Troubleshooting

### "Metadata file not found"
Create `data/metadata.csv` with columns:
- `pose_file` - Path to pose pickle file
- `stroke_type` - "clear" or "smash"

### "Not enough samples extracted"
Check pose file paths in metadata. Files should be in `data/processed/poses/` or use absolute paths.

### "Feature selection takes too long"
Use `--sample-size` to test on subset first:
```bash
python scripts/run_feature_selection.py --sample-size 500
```

### "Import errors"
Ensure all dependencies installed:
```bash
pip install -r requirements.txt
```

## Next Steps

Once all validations pass:

1. **Review reports:**
   - `outputs/reports/feature_selection_report.md`
   - `.planning/phases/02-feature-engineering-enhancement/02-VERIFICATION.md`

2. **Test v3 extraction:**
   ```python
   from src.data_processing.feature_versioning import FeatureEngineering
   fe = FeatureEngineering('v3')
   features = fe.extract_features(pose_sequence, apply_selection=True)
   # Should return ~187 selected features
   ```

3. **Proceed to Phase 3:**
   ```bash
   /gsd:plan-phase 3
   ```
