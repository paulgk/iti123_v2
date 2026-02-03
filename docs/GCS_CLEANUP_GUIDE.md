# GCS Storage Cleanup Guide

**Clean up old and obsolete files from Google Cloud Storage**

---

## Overview

Three scripts are provided to help manage your GCS bucket storage:

1. **List Contents** - Analyze current storage usage
2. **Interactive Cleanup** - Choose what to delete with confirmation
3. **Automated Cleanup** - Batch delete with dry-run option

---

## Prerequisites

### 1. Install Google Cloud SDK

```bash
# macOS
brew install --cask google-cloud-sdk

# Or download from:
# https://cloud.google.com/sdk/docs/install
```

### 2. Authenticate

```bash
gcloud auth login
```

### 3. Test Connection

```bash
gsutil ls gs://iti123storage
```

---

## Script 1: List GCS Contents

**Purpose:** Analyze what's in your bucket before cleanup

```bash
bash scripts/list_gcs_contents.sh
```

**Output:**
- Total bucket usage
- Breakdown by directory (data, features, outputs, models)
- File counts and sizes for each category
- Sample files from each directory
- Storage recommendations

**Example output:**
```
================================================================================
GCS STORAGE CONTENTS ANALYSIS
================================================================================
Bucket: gs://iti123storage
================================================================================

✓ Connected to bucket successfully

TOTAL BUCKET USAGE
45.23 GB

MAIN DIRECTORIES
--------------------------------------------------------------------------------
[SECTION] Data Directory
Path: gs://iti123storage/data/
Files: 19,778
Size:  12.34 GB
Sample files:
  gs://iti123storage/data/clips/Smash/01_set1_rally03_ball05_Smash.mp4
  gs://iti123storage/data/clips/Clear/01_set1_rally03_ball06_Clear.mp4
  ...

[SECTION] Features Directory
Path: gs://iti123storage/features/
Files: 19,423
Size:  8.56 GB
...
```

---

## Script 2: Interactive Cleanup (Recommended)

**Purpose:** Choose what to delete with confirmation prompts

```bash
bash scripts/clean_gcs_interactive.sh
```

**What it does:**
- Shows size for each category
- Asks for confirmation before deleting
- Lets you skip items you want to keep
- Safe for first-time cleanup

**Categories presented:**
1. Old pose extractions (poses_old, poses_backup, poses_v1)
2. Old metadata files (metadata_old.csv, metadata_v1.csv)
3. Old clips (clips_old, clips_backup, clips_v1)
4. Test files (clips_test, poses_test)
5. Temporary files (tmp, debug)
6. Ambiguous shot types (Slice_Drop, Push, Rear_Drive, etc.)
7. Old training outputs (outputs/old, models/old)
8. MLflow artifacts (mlflow, mlruns)

**Example session:**
```
================================================================================
INTERACTIVE GCS STORAGE CLEANUP
================================================================================

Current bucket usage:
45.23 GB

This script will ask you to confirm each deletion.

================================================================================
[INFO] Old pose extractions (before ROI)
Path: gs://iti123storage/features/poses_old/
Size: 1,234 files, 4.56 GB
Delete this? (y/n): y
[WARN] Deleting gs://iti123storage/features/poses_old/...
[INFO] ✓ Deleted successfully

[INFO] Old metadata file
Path: gs://iti123storage/data/metadata_old.csv
Size: 1 files, 0.01 GB
Delete this? (y/n): y
...
```

---

## Script 3: Automated Cleanup (Advanced)

**Purpose:** Batch delete with dry-run preview

```bash
# Step 1: Dry-run (preview only)
bash scripts/clean_gcs_storage.sh

# Step 2: Review output, then edit script to execute
# Edit scripts/clean_gcs_storage.sh and change:
#   DRY_RUN=false

# Step 3: Execute cleanup
bash scripts/clean_gcs_storage.sh
```

**What it cleans:**

**Section 1: Old Pose Extractions**
- `features/poses_old/**`
- `features/poses_backup/**`
- `features/poses_v1/**`

**Section 2: Old Metadata**
- `data/metadata_old.csv`
- `data/metadata_v1.csv`
- `data/metadata_backup.csv`

**Section 3: Old Clips**
- `data/clips_old/**`
- `data/clips_backup/**`
- `data/clips_v1/**`

**Section 4: Old Training Outputs**
- `outputs/old/**`
- `outputs/backup/**`
- `outputs/v1/**`
- `models/old/**`

**Section 5: Temporary/Debug**
- `tmp/**`
- `debug/**`
- `test/**`
- `data/poses_test/**`
- `data/clips_test/**`

**Section 6: Ambiguous Shot Types** (removed from Phase 1.5 mapping)
- `data/clips/Slice_Drop/**`
- `data/clips/Push/**`
- `data/clips/Rear_Drive/**`
- `data/clips/Defensive_Drive/**`
- `data/clips/Short_Drive/**`
- Corresponding pose files (`*_Slice_Drop.pkl`, etc.)

**Section 7: MLflow Artifacts**
- `mlflow/**`
- `mlruns/**`

---

## Common Cleanup Scenarios

### Scenario 1: Clean up after re-extraction

**Problem:** Re-extracted clips/poses with new ROI method, old files taking up space

**Solution:**
```bash
# 1. List to verify old files exist
bash scripts/list_gcs_contents.sh

# 2. Interactive cleanup
bash scripts/clean_gcs_interactive.sh

# When prompted:
#   - Delete old pose extractions: YES
#   - Delete old clips: YES
#   - Delete test files: YES
#   - Keep current data: NO (skip)
```

**Expected savings:** 10-20 GB

---

### Scenario 2: Remove ambiguous shots

**Problem:** Phase 1.5 removed ambiguous shots from mapping, old files still in storage

**Solution:**
```bash
bash scripts/clean_gcs_interactive.sh

# When prompted:
#   - Delete ambiguous shot types: YES
#     (Slice_Drop, Push, Rear_Drive, Defensive_Drive, Short_Drive)
```

**Expected savings:** 5-8 GB (6,016 clips + poses removed)

---

### Scenario 3: Clean up MLflow artifacts

**Problem:** No longer using MLflow, artifacts taking up space

**Solution:**
```bash
bash scripts/clean_gcs_interactive.sh

# When prompted:
#   - Delete MLflow artifacts: YES
```

**Expected savings:** 2-5 GB

---

### Scenario 4: Full cleanup (start fresh)

**Problem:** Want to remove all old/test/backup files

**Solution:**
```bash
# 1. Review what will be deleted
bash scripts/list_gcs_contents.sh

# 2. Backup important data locally first
gsutil -m cp -r gs://iti123storage/data/metadata.csv ./backup/
gsutil -m cp -r gs://iti123storage/outputs/models/*.pth ./backup/

# 3. Interactive cleanup (answer YES to most prompts)
bash scripts/clean_gcs_interactive.sh
```

**Expected savings:** 20-30 GB

---

## Manual Deletion Commands

If you need to delete specific files manually:

### Delete a specific directory
```bash
gsutil -m rm -r gs://iti123storage/path/to/delete/
```

### Delete files matching a pattern
```bash
gsutil -m rm gs://iti123storage/features/poses/*_Push.pkl
```

### Delete a specific file
```bash
gsutil rm gs://iti123storage/data/metadata_old.csv
```

### Move files instead of deleting
```bash
# Move to backup location
gsutil -m mv gs://iti123storage/old/ gs://iti123storage/backup/old/
```

---

## Safety Tips

### 1. Always list before deleting
```bash
# See what's in a directory before deleting
gsutil ls -lh gs://iti123storage/features/poses_old/
```

### 2. Dry-run mode
```bash
# Preview what would be deleted
gsutil -m rm -n gs://iti123storage/test/
# -n flag shows what would be deleted without actually deleting
```

### 3. Backup critical files first
```bash
# Download current metadata and models before cleanup
gsutil cp gs://iti123storage/data/metadata.csv ./backup/
gsutil -m cp -r gs://iti123storage/outputs/models/ ./backup/models/
```

### 4. Use interactive mode for first cleanup
```bash
# Safer than automated cleanup
bash scripts/clean_gcs_interactive.sh
```

---

## What NOT to Delete

**Keep these directories:**
- `data/clips/{Smash,Clear,Drop,Lift,Drive}/` - Current clean clips
- `features/poses/` - Current ROI-extracted poses (NOT poses_old)
- `data/metadata.csv` - Current metadata with player positions
- `outputs/models/*_best.pth` - Best trained models
- `outputs/reports/` - Training reports

**Current clean structure:**
```
gs://iti123storage/
├── data/
│   ├── metadata.csv                    ← KEEP (current)
│   └── clips/
│       ├── Smash/                      ← KEEP (19,778 clean clips)
│       ├── Clear/
│       ├── Drop/
│       ├── Lift/
│       └── Drive/
├── features/
│   └── poses/                          ← KEEP (19,423 ROI poses)
└── outputs/
    ├── models/*_best.pth               ← KEEP (best models)
    └── reports/                        ← KEEP (training reports)
```

---

## Troubleshooting

### Error: "gsutil: command not found"
```bash
# Install Google Cloud SDK
brew install --cask google-cloud-sdk

# Or download from:
# https://cloud.google.com/sdk/docs/install
```

### Error: "AccessDeniedException: 403"
```bash
# Authenticate
gcloud auth login

# Verify access
gsutil ls gs://iti123storage
```

### Error: "BucketNotFoundException: 404"
```bash
# Check bucket name
gsutil ls

# Verify bucket exists in console:
# https://console.cloud.google.com/storage/browser
```

### Deletion is slow
```bash
# Use -m flag for parallel deletion (faster)
gsutil -m rm -r gs://iti123storage/large_directory/

# For very large directories, increase parallelism
gsutil -o "GSUtil:parallel_process_count=16" -m rm -r gs://iti123storage/large_directory/
```

---

## Cost Savings

**Storage pricing (us-central1):**
- Standard storage: $0.020/GB/month
- Nearline storage: $0.010/GB/month

**Example savings:**
- Delete 20 GB of old files → Save $0.40/month ($4.80/year)
- Delete 50 GB of old files → Save $1.00/month ($12.00/year)

**Operation costs:**
- Delete operations: Free
- List operations: $0.05 per 10,000 operations (negligible)

---

## Summary

### Recommended Workflow

1. **Analyze storage:**
   ```bash
   bash scripts/list_gcs_contents.sh
   ```

2. **Interactive cleanup:**
   ```bash
   bash scripts/clean_gcs_interactive.sh
   ```

3. **Verify results:**
   ```bash
   gsutil du -sh gs://iti123storage
   ```

### Quick Commands

```bash
# List contents
bash scripts/list_gcs_contents.sh

# Interactive cleanup
bash scripts/clean_gcs_interactive.sh

# Automated cleanup (dry-run)
bash scripts/clean_gcs_storage.sh

# Manual deletion
gsutil -m rm -r gs://iti123storage/path/to/delete/
```

---

**Status:** Ready to use
**Last updated:** 2026-02-03
