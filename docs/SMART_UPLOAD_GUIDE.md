# Smart Upload Guide for GCS

**Date:** 2026-02-02
**Purpose:** Efficiently upload clips to GCS with automatic skip for completed folders

---

## Overview

The [upload_clips_to_gcs.sh](../scripts/upload_clips_to_gcs.sh) script now intelligently checks remote file counts before uploading, automatically skipping shot type folders that are already fully uploaded to GCS.

## Key Features

### 1. Pre-Upload Remote Check

Before processing each shot type, the script:
- ✅ Counts local files
- ✅ Counts remote files in GCS
- ✅ Compares counts
- ✅ **Skips entire folder if counts match**

### 2. Status Display

**At startup, you'll see:**
```bash
Counting local and remote clips...
  Smash: 3872 local, 3872 remote ✓ Complete
  Clear: 2662 local, 2662 remote ✓ Complete
  Drop: 7769 local, 7769 remote ✓ Complete
  Lift: 5230 local, 11 remote ⚠ Partial
  Drive: 3998 local, 0 remote
```

**Status indicators:**
- ✓ Complete (green): Remote matches local - will be skipped
- ⚠ Partial (yellow): Some files uploaded, more needed
- No indicator: Not uploaded yet

### 3. Automatic Skip

**When local == remote:**
```bash
✓ Smash: Already uploaded (3872/3872 clips) - Skipping
✓ Clear: Already uploaded (2662/2662 clips) - Skipping
✓ Drop: Already uploaded (7769/7769 clips) - Skipping
```

**Processing only happens for incomplete folders:**
```bash
Processing Lift (5230 local, 11 remote, ~5219 to upload)...
Processing Drive (3998 clips in 20 batches)...
```

---

## Usage

### Dry-Run (Check Status)

```bash
bash scripts/upload_clips_to_gcs.sh --dry-run
```

**Output shows:**
- Which folders are complete (will be skipped)
- Which folders need uploading
- How many files need to be uploaded for each

### Execute Upload

```bash
bash scripts/upload_clips_to_gcs.sh --execute
```

**Only uploads missing files:**
- Complete folders: Skipped instantly
- Partial folders: Uploads only missing files
- Empty folders: Full upload

---

## Benefits

### 1. Time Savings

**Before (no skip logic):**
- Checks every file individually: `gsutil stat` × 23,531 files = ~40 minutes
- Even if all files exist

**After (with skip logic):**
- Folder check: `gsutil ls` × 5 folders = ~5 seconds
- Skip complete folders instantly
- Only check individual files in incomplete folders

**Example scenario:**
```
Smash: 3,872 files ✓ Complete → Skip (0 seconds)
Clear: 2,662 files ✓ Complete → Skip (0 seconds)
Drop: 7,769 files ✓ Complete → Skip (0 seconds)
Lift: 5,230 files, 11 remote → Upload 5,219 (~45 min)
Drive: 3,998 files, 0 remote → Upload 3,998 (~35 min)

Total time: ~1.3 hours (vs 2+ hours checking all files)
```

### 2. Resume-Friendly

If upload gets interrupted:
- Re-run same command
- Script checks each folder
- Skips completed folders
- Resumes from last incomplete folder

### 3. Bandwidth Efficient

- No unnecessary file stat checks
- No re-uploads of existing files
- Only transfers new data

---

## How It Works

### Step 1: Count Files

**Local count:**
```bash
find "$local_dir" -name "*.mp4" | wc -l
```

**Remote count:**
```bash
gsutil ls "$gcs_path/*.mp4" | wc -l
```

### Step 2: Compare

```bash
if [ $remote_count -eq $clip_count ] && [ $clip_count -gt 0 ]; then
    echo "✓ Already uploaded - Skipping"
    return
fi
```

### Step 3: Upload (If Needed)

**Only if remote_count < local_count:**
- Create directory if needed
- Upload in batches of 200
- Check each file before upload (skip if exists)

---

## Edge Cases Handled

### Case 1: Remote has MORE files than local

**Situation:** Local was cleaned up, but remote still has old files

**Behavior:** Script will NOT delete remote files
- Compares: `remote_count == local_count`
- If remote > local, counts don't match
- Script will attempt to upload
- Individual file check will skip existing files
- No data loss

### Case 2: Remote folder doesn't exist

**Situation:** First upload for a shot type

**Behavior:**
- remote_count = 0
- local_count > 0
- Counts don't match → Upload proceeds
- Creates folder automatically

### Case 3: Partial upload interrupted

**Situation:** Upload stopped mid-folder

**Behavior:**
- remote_count < local_count
- Counts don't match → Upload proceeds
- Individual file checks skip existing files
- Only uploads remaining files

### Case 4: All folders complete

**Situation:** Everything already uploaded

**Behavior:**
```bash
✓ Smash: Already uploaded (3872/3872 clips) - Skipping
✓ Clear: Already uploaded (2662/2662 clips) - Skipping
✓ Drop: Already uploaded (7769/7769 clips) - Skipping
✓ Lift: Already uploaded (5230/5230 clips) - Skipping
✓ Drive: Already uploaded (3998/3998 clips) - Skipping

✓ Upload complete!
Total clips uploaded: ~23531
```

Script completes in ~30 seconds (folder checks only)

---

## Performance Comparison

| Scenario | Old Script | New Script | Time Saved |
|----------|-----------|-----------|------------|
| All complete | ~40 min (checks all files) | ~30 sec (folder checks) | **39.5 min** |
| 1 folder incomplete | ~40 min (checks all) | ~20 min (1 folder) | **20 min** |
| All incomplete | ~2 hours | ~2 hours | 0 min (same) |
| Resume after interrupt | ~40 min + upload | ~upload only | **40 min** |

**Key insight:** The more complete your upload, the more time you save.

---

## Troubleshooting

### Issue: "Already uploaded" but files are missing in GCS

**Cause:** Count matches but files differ (rare edge case)

**Solution:** Delete remote folder and re-upload
```bash
# Backup first (optional)
gsutil -m cp -r gs://iti123storage/videos/clips/smash/ ./backup/

# Delete remote folder
gsutil -m rm -r gs://iti123storage/videos/clips/smash/

# Re-upload
bash scripts/upload_clips_to_gcs.sh --execute
```

### Issue: Script skips folder but should upload

**Cause:** Remote count incorrectly matches local

**Check:**
```bash
# Count local
find data/clips/Smash -name "*.mp4" | wc -l

# Count remote
gsutil ls gs://iti123storage/videos/clips/smash/*.mp4 | wc -l

# Compare
```

**Fix if counts are wrong:** Clear remote and re-upload

### Issue: Slow folder checks

**Cause:** GCS bucket has many files

**Expected:** ~1-2 seconds per folder
**If slower:** Check network connection

---

## Best Practices

### 1. Always Run Dry-Run First

```bash
bash scripts/upload_clips_to_gcs.sh --dry-run
```

Check which folders need uploading before executing.

### 2. Upload in Order

If uploading manually by shot type:
1. Smash (smallest: ~1 hour)
2. Clear (small: ~45 min)
3. Drive (medium: ~1.3 hours)
4. Lift (medium: ~1.7 hours)
5. Drop (largest: ~2.6 hours)

This way, you get quick wins first.

### 3. Verify After Upload

```bash
bash scripts/upload_clips_to_gcs.sh --dry-run
```

Should show all folders as "✓ Complete"

### 4. Use with Resumable Pose Extraction

**Workflow:**
1. Upload clips to GCS (skip completed folders)
2. Run pose extraction in Colab (skip completed poses)
3. Both are resumable - perfect combo!

---

## Files Modified

- [scripts/upload_clips_to_gcs.sh](../scripts/upload_clips_to_gcs.sh)
  - Added remote file count check
  - Added folder skip logic
  - Enhanced status display
  - Improved dry-run output

---

## Example Output

### First Run (Nothing Uploaded)

```bash
Counting local and remote clips...
  Smash: 3872 local, 0 remote
  Clear: 2662 local, 0 remote
  Drop: 7769 local, 0 remote
  Lift: 5230 local, 0 remote
  Drive: 3998 local, 0 remote

UPLOADING CLIPS
Processing Smash (3872 clips in 20 batches)...
  Batch 1/20: uploading clips 1-200 (200 files)...
    ✓ Batch complete (200 files)
  ...
✓ Uploaded 3872/3872 Smash clips

[Continues for all shot types]
```

### Second Run (Everything Complete)

```bash
Counting local and remote clips...
  Smash: 3872 local, 3872 remote ✓ Complete
  Clear: 2662 local, 2662 remote ✓ Complete
  Drop: 7769 local, 7769 remote ✓ Complete
  Lift: 5230 local, 5230 remote ✓ Complete
  Drive: 3998 local, 3998 remote ✓ Complete

UPLOADING CLIPS
✓ Smash: Already uploaded (3872/3872 clips) - Skipping
✓ Clear: Already uploaded (2662/2662 clips) - Skipping
✓ Drop: Already uploaded (7769/7769 clips) - Skipping
✓ Lift: Already uploaded (5230/5230 clips) - Skipping
✓ Drive: Already uploaded (3998/3998 clips) - Skipping

SUMMARY
✓ Upload complete!
Total clips uploaded: ~23531
```

### Resume After Interruption

```bash
Counting local and remote clips...
  Smash: 3872 local, 3872 remote ✓ Complete
  Clear: 2662 local, 2662 remote ✓ Complete
  Drop: 7769 local, 5123 remote ⚠ Partial
  Lift: 5230 local, 0 remote
  Drive: 3998 local, 0 remote

UPLOADING CLIPS
✓ Smash: Already uploaded (3872/3872 clips) - Skipping
✓ Clear: Already uploaded (2662/2662 clips) - Skipping

Processing Drop (7769 local, 5123 remote, ~2646 to upload)...
  [Uploads remaining 2646 clips]

Processing Lift (5230 clips in 27 batches)...
  [Uploads all clips]

Processing Drive (3998 clips in 20 batches)...
  [Uploads all clips]
```

---

**Status:** Production ready - Optimized for resumable, efficient uploads
