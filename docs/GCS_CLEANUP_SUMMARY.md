# GCS Cleanup Scripts - Quick Summary

**Three scripts to manage Google Cloud Storage efficiently**

---

## TL;DR

```bash
# 1. See what's in your bucket
bash scripts/list_gcs_contents.sh

# 2. Clean up interactively (RECOMMENDED)
bash scripts/clean_gcs_interactive.sh

# 3. Or automated cleanup
bash scripts/clean_gcs_storage.sh  # Dry-run first
```

---

## Scripts Overview

### 1. `list_gcs_contents.sh` - Analyze Storage

**What it does:**
- Shows total bucket size
- Lists all directories with file counts
- Displays storage breakdown by category
- Shows sample files from each directory

**When to use:**
- Before cleanup to see what's taking space
- After cleanup to verify results
- Regular storage audits

**Example:**
```bash
$ bash scripts/list_gcs_contents.sh

TOTAL BUCKET USAGE
45.23 GB

Data (clips + metadata):  12.34 GB
Features (poses):         18.56 GB
Outputs (training):        8.12 GB
Models (checkpoints):      6.21 GB
```

---

### 2. `clean_gcs_interactive.sh` - Safe Cleanup (RECOMMENDED)

**What it does:**
- Shows size for each category
- Asks YES/NO before deleting
- Lets you skip items
- Reports final savings

**When to use:**
- First-time cleanup
- When unsure what to delete
- When you want control over deletions

**What it cleans:**
1. ✅ Old pose extractions (poses_old, poses_backup, poses_v1)
2. ✅ Old metadata files (metadata_old.csv, metadata_v1.csv)
3. ✅ Old clips (clips_old, clips_backup, clips_v1)
4. ✅ Test files (clips_test, poses_test)
5. ✅ Temporary files (tmp, debug)
6. ✅ Ambiguous shots (Slice_Drop, Push, Rear_Drive, Defensive_Drive, Short_Drive)
7. ✅ Old training outputs (outputs/old, models/old)
8. ✅ MLflow artifacts (mlflow, mlruns)

**Example session:**
```bash
$ bash scripts/clean_gcs_interactive.sh

Current bucket usage: 45.23 GB

[INFO] Old pose extractions (before ROI)
Path: gs://iti123storage/features/poses_old/
Size: 1,234 files, 4.56 GB
Delete this? (y/n): y
✓ Deleted successfully

[INFO] Ambiguous shot types
Delete all? (y/n): y
✓ Deleted Slice_Drop clips
✓ Deleted Push clips
...

CLEANUP COMPLETE
New bucket usage: 28.15 GB
Saved: 17.08 GB
```

---

### 3. `clean_gcs_storage.sh` - Automated Cleanup

**What it does:**
- Batch deletes predefined paths
- Dry-run mode by default
- Reports all actions

**When to use:**
- After reviewing with list script
- When you know exactly what to delete
- For scripted/automated cleanup

**Safety:**
- Runs in dry-run mode by default
- Must edit script to enable deletion
- Shows what would be deleted first

**How to use:**
```bash
# Step 1: Preview (dry-run)
bash scripts/clean_gcs_storage.sh

# Step 2: Review output carefully

# Step 3: Edit script to enable deletion
# Change: DRY_RUN=false

# Step 4: Execute cleanup
bash scripts/clean_gcs_storage.sh
```

---

## What to Delete

### Safe to Delete ✅

**Old extractions (before Phase 1.5 ROI):**
- `features/poses_old/`
- `features/poses_backup/`
- `data/clips_old/`
- Old metadata files

**Ambiguous shot types (removed from mapping):**
- `data/clips/Slice_Drop/`
- `data/clips/Push/`
- `data/clips/Rear_Drive/`
- `data/clips/Defensive_Drive/`
- `data/clips/Short_Drive/`
- Corresponding pose files

**Test and temporary:**
- `data/clips_test/`
- `data/poses_test/`
- `tmp/`, `debug/`, `test/`

**Old training:**
- `outputs/old/`, `outputs/backup/`
- `models/old/`

**MLflow (if not using):**
- `mlflow/`, `mlruns/`

### Keep ❌

**Current Phase 1.5 data:**
- `data/clips/{Smash,Clear,Drop,Lift,Drive}/` - Clean 19,778 clips
- `features/poses/` - ROI-extracted 19,423 poses
- `data/metadata.csv` - Current with player positions

**Best models:**
- `outputs/models/*_best.pth`
- `outputs/reports/`

---

## Expected Savings

| Category | Files | Size | Savings |
|----------|-------|------|---------|
| Old pose extractions | ~20K | 8-12 GB | ✅ |
| Ambiguous shots | ~6K | 5-8 GB | ✅ |
| Test files | ~1K | 1-2 GB | ✅ |
| Old outputs | ~100 | 2-5 GB | ✅ |
| MLflow | ~1K | 2-5 GB | ✅ |
| **TOTAL** | **~28K** | **18-32 GB** | **✅** |

---

## Workflow Recommendation

### First-Time Cleanup

```bash
# Step 1: Analyze current storage
bash scripts/list_gcs_contents.sh > storage_before.txt

# Step 2: Interactive cleanup (answer YES to most prompts)
bash scripts/clean_gcs_interactive.sh

# Step 3: Verify results
bash scripts/list_gcs_contents.sh > storage_after.txt
diff storage_before.txt storage_after.txt
```

### Before Uploading New Data

```bash
# Clean up old data first
bash scripts/clean_gcs_interactive.sh

# Then upload new ROI data
gsutil -m rsync -r data/poses/ gs://iti123storage/features/poses_roi/
gsutil cp data/metadata.csv gs://iti123storage/data/metadata_roi.csv
```

### Regular Maintenance

```bash
# Monthly storage check
bash scripts/list_gcs_contents.sh

# Clean up if >50 GB
if [ $(gsutil du -s gs://iti123storage | awk '{print $1}') -gt 53687091200 ]; then
    bash scripts/clean_gcs_interactive.sh
fi
```

---

## Troubleshooting

### "gsutil: command not found"
```bash
brew install --cask google-cloud-sdk
```

### "AccessDeniedException: 403"
```bash
gcloud auth login
```

### Slow deletion
```bash
# Use parallel mode (-m)
gsutil -m rm -r gs://iti123storage/path/
```

---

## Manual Commands

```bash
# List specific directory
gsutil ls -lh gs://iti123storage/features/poses_old/

# Delete specific directory
gsutil -m rm -r gs://iti123storage/features/poses_old/

# Check total size
gsutil du -sh gs://iti123storage

# Copy before deleting (backup)
gsutil -m cp -r gs://iti123storage/important/ ./backup/
```

---

## Summary Table

| Script | Purpose | Safety | When to Use |
|--------|---------|--------|-------------|
| `list_gcs_contents.sh` | Analyze storage | ✅ Read-only | Before cleanup, audits |
| `clean_gcs_interactive.sh` | Safe cleanup | ✅ Confirms each | First-time, unsure |
| `clean_gcs_storage.sh` | Batch cleanup | ⚠️ Dry-run first | Know what to delete |

---

## Documentation

- Full guide: [docs/GCS_CLEANUP_GUIDE.md](GCS_CLEANUP_GUIDE.md)
- Quick reference: [PHASE_1.5_COMMANDS.md](../PHASE_1.5_COMMANDS.md)

---

**Recommended:** Start with `clean_gcs_interactive.sh` for safe, controlled cleanup.

**Status:** Ready to use
**Last updated:** 2026-02-03
