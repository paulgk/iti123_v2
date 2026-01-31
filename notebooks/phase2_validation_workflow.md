# Phase 2 Validation Workflow - Colab Enterprise

Complete workflow to pull videos from GCS, extract poses, run feature engineering, and validate Phase 2.

## Prerequisites

- GCS bucket: `gs://iti123storage/`
- Videos in: `gs://iti123storage/videos/` and `gs://iti123storage/videos/clips/`
- Service account with Storage Admin role
- Colab Enterprise runtime

---

## Step-by-Step Workflow

### Step 1: Setup Colab Environment (5 minutes)

```bash
# Clone repository
cd /content
git clone https://github.com/YOUR_USERNAME/iti123_v2.git
cd iti123_v2

# Create Python 3.10 virtual environment
bash scripts/colab_setup.sh

# Activate environment
source colab_venv/bin/activate

# Verify Python version
python --version  # Should be Python 3.10.x
```

### Step 2: Authenticate with GCS (2 minutes)

**Option A: Service Account Key (Recommended)**

```bash
# Upload your service account JSON key to Colab
# Then set environment variable
export GOOGLE_APPLICATION_CREDENTIALS="/content/iti123-service-key.json"

# Verify access
gsutil ls gs://iti123storage/
```

**Option B: Colab Default Credentials**

```python
from google.colab import auth
auth.authenticate_user()

# Test access
!gsutil ls gs://iti123storage/
```

### Step 3: Pull Videos from GCS (10-30 minutes)

```bash
# Create local directories
mkdir -p data/videos/clips
mkdir -p data/processed/poses

# Pull videos from GCS
# Option 1: Pull all videos
gsutil -m rsync -r gs://iti123storage/videos/ data/videos/

# Option 2: Pull only clips (faster)
gsutil -m rsync -r gs://iti123storage/videos/clips/ data/videos/clips/

# Check what was downloaded
find data/videos/clips -name "*.mp4" | wc -l
ls -lh data/videos/clips/
```

**Expected structure:**
```
data/videos/clips/
├── clear/
│   ├── clear_001.mp4
│   ├── clear_002.mp4
│   └── ...
└── smash/
    ├── smash_001.mp4
    ├── smash_002.mp4
    └── ...
```

### Step 4: Extract Pose Sequences (30-90 minutes)

```bash
# Extract poses with MediaPipe
python scripts/extract_poses.py \
    --video-dir data/videos/clips \
    --output-dir data/processed/poses \
    --target-fps 30 \
    --min-confidence 0.5 \
    --create-metadata \
    --verbose

# This will:
# - Process all .mp4 files in data/videos/clips/
# - Extract pose landmarks (33 keypoints per frame)
# - Save to data/processed/poses/ as pickle files
# - Auto-create data/metadata.csv with labels inferred from folder structure
```

**Expected output:**
```
Found 200 video files
Extracting poses: 100%|████████| 200/200 [30:00<00:00, 9.00s/it]

EXTRACTION COMPLETE
Total videos: 200
Successfully processed: 195
Failed: 5

Stroke Type Distribution:
clear    98
smash    97

✓ Metadata saved to: data/metadata.csv
```

### Step 5: Review & Fix Metadata (5 minutes)

```python
import pandas as pd

# Load metadata
df = pd.read_csv('data/metadata.csv')
print(df.head(10))
print(f"\nTotal samples: {len(df)}")
print(f"\nStroke types:\n{df['stroke_type'].value_counts()}")
print(f"\nPlayer IDs:\n{df['player_id'].value_counts()}")

# Check for 'unknown' stroke types (need manual labeling)
unknown = df[df['stroke_type'] == 'unknown']
if len(unknown) > 0:
    print(f"\n⚠️ {len(unknown)} videos with unknown stroke type")
    print("Please manually label these in metadata.csv")
    print(unknown[['video_id', 'video_path']])

# Fix player IDs if needed (important for train/test split)
# Ensure same player isn't in both train and test sets
```

### Step 6: Upload Extracted Poses to GCS (10-20 minutes)

```bash
# Upload poses to GCS for backup
gsutil -m rsync -r data/processed/poses/ gs://iti123storage/features/poses/

# Upload metadata
gsutil cp data/metadata.csv gs://iti123storage/metadata.csv

# Verify upload
gsutil ls gs://iti123storage/features/poses/ | wc -l
```

### Step 7: Run Phase 2 Validation (15-30 minutes)

```bash
# Validation 1, 2, 4: Phase segmentation, effect sizes, backward compatibility
python scripts/validate_phase2.py --sample-size 100 --metadata data/metadata.csv

# Expected output:
# VALIDATION 1: Phase Segmentation Boundary Accuracy
#   Validation pass rate: 87.0%
#   Status: ✓ PASS
#
# VALIDATION 2: Kinetic Chain Feature Effect Sizes
#   hip_to_wrist_total: d=0.905 (large) ✓
#   Status: ✓ PASS
#
# VALIDATION 4: V2 Backward Compatibility
#   Status: ✓ PASS
```

### Step 8: Run Feature Selection Pipeline (30-60 minutes)

```bash
# Validation 3: Feature selection
python scripts/run_feature_selection.py \
    --metadata data/metadata.csv \
    --target-features 254 \
    --verbose

# Expected output:
# Step 1: Loading metadata and extracting features
#   ✓ Extracted features from 195 samples
#
# Step 2: Running feature selection pipeline
#   Initial features: 361
#   After Cohen's d >= 0.5: 243
#   After VIF < 10: 198
#   After RFECV (final): 187
#   Final Feature Count: 187
#   Target Met: ✓ PASS
#   CV F1 Score: 0.7823
#
# Step 3: Saving results
#   ✓ Saved feature manifest: data/processed/features_v3/selected_features.json
#   ✓ Saved selection report: outputs/reports/feature_selection_report.md
```

### Step 9: Review Results

```bash
# View feature selection report
cat outputs/reports/feature_selection_report.md | head -100

# Check selected features
python -c "import json; f=open('data/processed/features_v3/selected_features.json'); d=json.load(f); print(f'Selected {len(d[\"selected_features\"])} features'); print(d['selected_features'][:20])"

# Verify v3 extraction works
python -c "
from src.data_processing.feature_versioning import FeatureEngineering
import pickle
fe = FeatureEngineering('v3')
pose = pickle.load(open('data/processed/poses/clear_001_pose.pkl', 'rb'))
features = fe.extract_features(pose, apply_selection=True)
print(f'✓ V3 extraction works: {len(features)} features extracted')
"
```

### Step 10: Upload Results to GCS (5-10 minutes)

```bash
# Upload feature selection results
gsutil -m rsync -r outputs/ gs://iti123storage/outputs/
gsutil -m rsync -r data/processed/features_v3/ gs://iti123storage/features_v3/

# Upload validation reports
gsutil cp .planning/phases/02-feature-engineering-enhancement/02-VERIFICATION.md \
    gs://iti123storage/reports/phase2_verification.md

# Commit to git and push
git add -A
git commit -m "feat(validation): complete phase 2 validation with real dataset"
git push origin milestone/v1.1-coach-informed-ml
```

---

## Quick Reference Commands

### Check Video Count
```bash
gsutil ls gs://iti123storage/videos/clips/**/*.mp4 | wc -l
```

### Download Sample for Testing
```bash
# Download just 10 videos for quick test
gsutil -m cp gs://iti123storage/videos/clips/clear/*.mp4 data/videos/clips/clear/ | head -10
gsutil -m cp gs://iti123storage/videos/clips/smash/*.mp4 data/videos/clips/smash/ | head -10
```

### Re-run Just Feature Selection
```bash
# If you already have poses extracted
python scripts/run_feature_selection.py --skip-extraction
```

### Check Progress
```bash
# Check extracted poses
ls data/processed/poses/*.pkl | wc -l

# Check metadata
wc -l data/metadata.csv
```

---

## Troubleshooting

### "No video files found"
```bash
# Check video directory structure
find data/videos -name "*.mp4"

# Verify GCS sync worked
gsutil ls gs://iti123storage/videos/clips/
```

### "MediaPipe import error"
```bash
# Reinstall MediaPipe
pip install --upgrade mediapipe==0.10.9

# Or use requirements
pip install -r requirements-colab.txt
```

### "Not enough samples extracted"
```bash
# Check failed extractions
cat data/processed/poses/failed_extractions.txt

# Lower confidence threshold
python scripts/extract_poses.py --min-confidence 0.3 ...
```

### "Feature selection takes too long"
```bash
# Test on smaller sample first
python scripts/run_feature_selection.py --sample-size 100
```

### "Out of memory"
```bash
# Use High-RAM runtime in Colab
# Runtime > Change runtime type > High-RAM

# Or process in batches
python scripts/run_feature_selection.py --sample-size 500
```

---

## Expected Timeline

| Step | Task | Duration |
|------|------|----------|
| 1 | Setup Colab environment | 5 min |
| 2 | Authenticate GCS | 2 min |
| 3 | Pull videos (200 clips) | 15-30 min |
| 4 | Extract poses (200 clips) | 30-90 min |
| 5 | Review metadata | 5 min |
| 6 | Upload poses to GCS | 10-20 min |
| 7 | Run validation suite | 15-30 min |
| 8 | Run feature selection | 30-60 min |
| 9 | Review results | 10 min |
| 10 | Upload results | 5-10 min |
| **Total** | | **~2-4 hours** |

---

## Success Criteria

After completing this workflow, you should have:

- ✅ Poses extracted for all video clips
- ✅ `data/metadata.csv` with stroke labels
- ✅ Phase 2 validation: 4/4 checks passed
- ✅ Feature selection manifest populated (<254 features)
- ✅ Feature selection report generated
- ✅ All results uploaded to GCS
- ✅ Ready for Phase 3 (Model Training)

---

## Next Steps

Once validation completes successfully:

1. **Review reports:**
   - `outputs/reports/feature_selection_report.md`
   - `.planning/phases/02-feature-engineering-enhancement/02-VERIFICATION.md`

2. **Verify manifest:**
   ```bash
   cat data/processed/features_v3/selected_features.json
   ```

3. **Proceed to Phase 3:**
   ```bash
   /gsd:discuss-phase 3
   ```

4. **Or plan directly:**
   ```bash
   /gsd:plan-phase 3
   ```
