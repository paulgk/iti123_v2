# Colab Enterprise Quick Start

Get from videos to validated Phase 2 in ~2-4 hours.

## 🚀 One-Command Validation (Recommended)

```bash
# In Colab Enterprise terminal
cd /content
git clone https://github.com/YOUR_USERNAME/iti123_v2.git
cd iti123_v2

# Set GCS credentials
export GOOGLE_APPLICATION_CREDENTIALS="/content/your-service-key.json"

# Run complete workflow
bash scripts/colab_phase2_validation.sh
```

This single command will:
1. ✅ Setup Python 3.10 environment
2. ✅ Pull videos from GCS (`gs://iti123storage/videos/clips/`)
3. ✅ Extract poses with MediaPipe
4. ✅ Create metadata.csv with auto-detected labels
5. ✅ Run Phase 2 validation suite
6. ✅ Run feature selection pipeline
7. ✅ Upload results back to GCS
8. ✅ Generate summary report

---

## 📋 Step-by-Step (Manual Control)

### 1. Setup (5 min)

```bash
cd /content
git clone https://github.com/YOUR_USERNAME/iti123_v2.git
cd iti123_v2

# Create Python 3.10 venv
bash scripts/colab_setup.sh
source colab_venv/bin/activate
```

### 2. Authenticate GCS (2 min)

```bash
# Option A: Service account key
export GOOGLE_APPLICATION_CREDENTIALS="/content/key.json"

# Option B: Colab auth
from google.colab import auth
auth.authenticate_user()
```

### 3. Pull Videos (15-30 min)

```bash
# Pull all clips
gsutil -m rsync -r gs://iti123storage/videos/clips/ data/videos/clips/

# Or just sample for testing
gsutil ls gs://iti123storage/videos/clips/clear/*.mp4 | head -10 | gsutil -m cp -I data/videos/clips/clear/
gsutil ls gs://iti123storage/videos/clips/smash/*.mp4 | head -10 | gsutil -m cp -I data/videos/clips/smash/
```

### 4. Extract Poses (30-90 min)

```bash
python scripts/extract_poses.py \
    --video-dir data/videos/clips \
    --output-dir data/processed/poses \
    --target-fps 30 \
    --create-metadata \
    --verbose
```

### 5. Run Validation (15-30 min)

```bash
python scripts/validate_phase2.py --sample-size 100
```

### 6. Run Feature Selection (30-60 min)

```bash
python scripts/run_feature_selection.py
```

### 7. Upload Results (10 min)

```bash
gsutil -m rsync -r outputs/ gs://iti123storage/outputs/
gsutil -m rsync -r data/processed/features_v3/ gs://iti123storage/features_v3/
```

---

## 🎯 Testing with Sample Data

For quick testing (10 videos):

```bash
# Pull sample
bash scripts/colab_phase2_validation.sh 10

# Or manually
python scripts/extract_poses.py --video-dir data/videos/clips --output-dir data/processed/poses
python scripts/validate_phase2.py --sample-size 10
```

---

## 📊 Expected Results

### After Extraction:
```
data/processed/poses/
├── clear_001_pose.pkl
├── clear_002_pose.pkl
├── smash_001_pose.pkl
└── ...

data/metadata.csv created with ~200 samples
```

### After Validation:
```
VALIDATION 1: Phase Segmentation ✓ PASS (87% accuracy)
VALIDATION 2: Effect Sizes ✓ PASS (d > 0.5)
VALIDATION 4: Backward Compat ✓ PASS
```

### After Feature Selection:
```
Initial features: 361
Final features: 187
Target (<254): ✓ PASS
CV F1 Score: 0.78

data/processed/features_v3/selected_features.json created
outputs/reports/feature_selection_report.md generated
```

---

## 🔧 Troubleshooting

### Videos not downloading
```bash
# Check GCS access
gsutil ls gs://iti123storage/videos/clips/

# Check bucket contents
gsutil ls -r gs://iti123storage/ | head -20
```

### MediaPipe errors
```bash
# Reinstall with correct version
pip install --upgrade mediapipe==0.10.9 opencv-python
```

### Out of memory
```bash
# Use High-RAM runtime
# Runtime > Change runtime type > High-RAM

# Or process in batches
bash scripts/colab_phase2_validation.sh 50  # Only 50 videos
```

### Feature selection fails
```bash
# Need minimum 50 samples
# Check if enough poses extracted
ls data/processed/poses/*.pkl | wc -l
```

---

## 📁 File Locations

### Inputs (from GCS):
- Videos: `gs://iti123storage/videos/clips/{clear,smash}/*.mp4`

### Outputs (local):
- Poses: `data/processed/poses/*.pkl`
- Metadata: `data/metadata.csv`
- Features v3: `data/processed/features_v3/selected_features.json`
- Reports: `outputs/reports/*.md`

### Outputs (uploaded to GCS):
- Poses backup: `gs://iti123storage/features/poses/`
- Features v3: `gs://iti123storage/features_v3/`
- Reports: `gs://iti123storage/outputs/reports/`

---

## ✅ Success Checklist

- [ ] Videos downloaded from GCS
- [ ] Poses extracted (≥50 samples)
- [ ] metadata.csv created with labels
- [ ] Validation 1: Phase segmentation ≥85% pass rate
- [ ] Validation 2: Kinetic chain d > 0.5
- [ ] Validation 4: V2 backward compatibility works
- [ ] Feature selection: <254 features selected
- [ ] Results uploaded to GCS
- [ ] Ready for Phase 3

---

## 🚀 Next Steps

After successful validation:

```bash
# Plan Phase 3 (Model Training)
/gsd:discuss-phase 3

# Or jump straight to planning
/gsd:plan-phase 3
```

---

## 📚 Detailed Documentation

- Full workflow: [notebooks/phase2_validation_workflow.md](notebooks/phase2_validation_workflow.md)
- Validation scripts: [scripts/README.md](scripts/README.md)
- Phase 2 verification: [.planning/phases/02-feature-engineering-enhancement/02-VERIFICATION.md](.planning/phases/02-feature-engineering-enhancement/02-VERIFICATION.md)
