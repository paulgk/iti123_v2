# GCS Scripts Summary

**Quick reference for all GCS management scripts**

---

## All Scripts

### Storage Management (3 scripts)

1. **[list_gcs_contents.sh](../scripts/list_gcs_contents.sh)** - Analyze storage
2. **[clean_gcs_interactive.sh](../scripts/clean_gcs_interactive.sh)** - Clean up interactively
3. **[clean_gcs_storage.sh](../scripts/clean_gcs_storage.sh)** - Automated cleanup

### Data Transfer (3 scripts)

4. **[upload_poses_to_gcs.sh](../scripts/upload_poses_to_gcs.sh)** - Full-featured upload
5. **[quick_upload_gcs.sh](../scripts/quick_upload_gcs.sh)** - Simple upload
6. **[download_from_gcs_colab.sh](../scripts/download_from_gcs_colab.sh)** - Colab download

---

## Quick Reference

### Before Uploading New Data

```bash
# Step 1: See what's in bucket
bash scripts/list_gcs_contents.sh

# Step 2: Clean up old files
bash scripts/clean_gcs_interactive.sh
```

### Upload Data

```bash
# Quick upload (RECOMMENDED)
bash scripts/quick_upload_gcs.sh

# Full upload with verification
bash scripts/upload_poses_to_gcs.sh
```

### Download in Colab

```bash
# In Colab cell
!bash scripts/download_from_gcs_colab.sh
```

---

## Script Comparison

### Storage Management

| Script | Type | Safety | Speed | Use Case |
|--------|------|--------|-------|----------|
| list_gcs_contents.sh | Read-only | ✅ Safe | Fast | Analysis, audits |
| clean_gcs_interactive.sh | Interactive | ✅ Confirms | Medium | First-time cleanup |
| clean_gcs_storage.sh | Automated | ⚠️ Dry-run first | Fast | Batch cleanup |

### Data Transfer

| Script | Features | Speed | Use Case |
|--------|----------|-------|----------|
| upload_poses_to_gcs.sh | Verification, dry-run | Medium | First upload, need verification |
| quick_upload_gcs.sh | Simple, no config | Fast | Regular uploads |
| download_from_gcs_colab.sh | Colab-optimized, verification | Medium | Training in Colab |

---

## Complete Workflow

### Phase 1: Extract Data Locally

```bash
# Extract first half
bash scripts/extract_full_pipeline.sh 01 22

# Verify extraction
python scripts/validate_roi_poses.py --poses data/poses --metadata data/metadata.csv
```

### Phase 2: Clean GCS Storage

```bash
# Analyze current storage
bash scripts/list_gcs_contents.sh > storage_before.txt

# Clean up interactively
bash scripts/clean_gcs_interactive.sh

# Verify cleanup
bash scripts/list_gcs_contents.sh > storage_after.txt
diff storage_before.txt storage_after.txt
```

**Expected cleanup:**
- Old poses: 8-12 GB
- Ambiguous shots: 5-8 GB
- Test files: 1-2 GB
- **Total saved: 15-30 GB**

### Phase 3: Upload to GCS

```bash
# Quick upload
bash scripts/quick_upload_gcs.sh

# Verify upload
gsutil ls gs://iti123storage/features/poses_roi/ | wc -l
# Should show: 19,423 files
```

### Phase 4: Train in Colab

```python
# Colab notebook

# Cell 1: Authenticate
from google.colab import auth
auth.authenticate_user()

# Cell 2: Clone repo
!git clone https://github.com/your-repo/iti123_v2.git
%cd iti123_v2

# Cell 3: Download data
!bash scripts/download_from_gcs_colab.sh

# Cell 4: Train
!python scripts/train_models_fixed.py --model stgcn --epochs 50
```

---

## File Locations

### Local (After Extraction)

```
data/
├── clips/
│   ├── Smash/ (4,234 clips)
│   ├── Clear/ (2,922 clips)
│   ├── Drop/ (6,290 clips)
│   ├── Lift/ (5,632 clips)
│   └── Drive/ (700 clips)
├── poses/ (19,423 .pkl files)
└── metadata.csv (19,778 entries)
```

### GCS (After Upload)

```
gs://iti123storage/
├── features/
│   └── poses_roi/        # 19,423 pose files, ~9 GB
├── data/
│   ├── metadata_roi.csv  # Current metadata, ~2 MB
│   └── clips_roi/        # Optional, ~15 GB
└── outputs/
    └── models/           # Trained models
```

### Colab (After Download)

```
./data/
├── poses/ (19,423 .pkl files)
└── metadata.csv (19,778 entries)
```

---

## Expected Times and Costs

### Times

| Task | Files | Time | Can Run Unattended? |
|------|-------|------|---------------------|
| List GCS contents | N/A | 30-60s | Yes |
| Clean GCS (interactive) | Varies | 5-10 min | No (confirms each) |
| Upload poses | 19,423 | 5-10 min | Yes |
| Upload metadata | 1 | <1 min | Yes |
| Download in Colab | 19,424 | 5-10 min | Yes |

### Costs

| Operation | Size | Cost |
|-----------|------|------|
| Upload (one-time) | 9 GB | ~$0.02 |
| Storage (monthly) | 9 GB | ~$0.18 |
| Download (Colab) | 9 GB | FREE (same region) |
| Cleanup (delete) | N/A | FREE |

---

## Common Issues and Solutions

### Issue: "gsutil: command not found"

```bash
# Install Google Cloud SDK
brew install --cask google-cloud-sdk
```

### Issue: "AccessDeniedException: 403"

```bash
# Authenticate
gcloud auth login

# Set project
gcloud config set project YOUR_PROJECT_ID
```

### Issue: Upload/download is slow

```bash
# Already using -m flag for parallelism
# Increase parallelism (advanced)
gsutil -o "GSUtil:parallel_process_count=16" -m rsync -r source/ dest/
```

### Issue: Verification fails (missing poses)

```bash
# Check which poses are missing
python - <<EOF
import csv
from pathlib import Path

with open('data/metadata.csv', 'r') as f:
    metadata_ids = {row['video_id'] for row in csv.DictReader(f)}

pose_files = {f.stem for f in Path('data/poses').glob('*.pkl')}
missing = metadata_ids - pose_files

print(f"Missing: {len(missing)}")
for vid in list(missing)[:10]:
    print(f"  {vid}")
EOF
```

---

## Documentation

- **Complete guides:**
  - [GCS_CLEANUP_GUIDE.md](GCS_CLEANUP_GUIDE.md) - Storage management
  - [GCS_UPLOAD_DOWNLOAD_GUIDE.md](GCS_UPLOAD_DOWNLOAD_GUIDE.md) - Data transfer
  - [GCS_CLEANUP_SUMMARY.md](GCS_CLEANUP_SUMMARY.md) - Quick cleanup reference

- **Phase guide:**
  - [PHASE_1.5_COMMANDS.md](../PHASE_1.5_COMMANDS.md) - Complete workflow

---

## Summary Commands

```bash
# STORAGE MANAGEMENT
bash scripts/list_gcs_contents.sh              # Analyze
bash scripts/clean_gcs_interactive.sh          # Clean up
bash scripts/clean_gcs_storage.sh              # Automated cleanup

# DATA UPLOAD
bash scripts/quick_upload_gcs.sh               # Quick upload
bash scripts/upload_poses_to_gcs.sh            # Full upload

# DATA DOWNLOAD (Colab)
!bash scripts/download_from_gcs_colab.sh       # Download for training
```

---

**All scripts are executable and ready to use!**

**Status:** Complete
**Last updated:** 2026-02-03
