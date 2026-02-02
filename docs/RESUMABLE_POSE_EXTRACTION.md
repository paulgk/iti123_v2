# Resumable Pose Extraction

**Date:** 2026-02-02
**Purpose:** Resume pose extraction after interruptions without losing progress

---

## Problem

Colab runtime disconnections or errors during long pose extraction sessions (4-6 hours for 23,531 clips) can cause loss of progress. The script needed to be fully resumable.

## Solution

The [extract_poses_parallel.py](../scripts/extract_poses_parallel.py) script now includes:

### 1. Smart Skip Logic

**Checks existing pose files:**
```python
# Skip if already exists and is valid (non-empty)
if output_path.exists():
    # Validate file is not empty/corrupted
    if output_path.stat().st_size > 100:  # At least 100 bytes
        try:
            # Quick validation: try to load the pickle
            with open(output_path, 'rb') as f:
                _ = pickle.load(f)
            return {'status': 'skipped', 'video': video_name}
        except:
            # File corrupted, re-extract
            print(f"⚠️  Corrupted pose file detected, re-extracting: {video_name}")
            pass
```

**Benefits:**
- Validates pose file size (>100 bytes)
- Tests pickle file integrity
- Automatically re-extracts corrupted files
- Skips valid existing files

### 2. Progress Reporting

**Startup summary:**
```
Found 23531 video files
Found 1500 existing pose files (will skip these)
Estimated remaining clips to process: 22031
```

**Checkpoint updates every 100 clips:**
```
Checkpoint (100/23531): 95 new, 5 skipped, 0 failed
Checkpoint (200/23531): 190 new, 10 skipped, 0 failed
```

### 3. Metadata Merging

**Preserves existing metadata:**
```python
# Merge with existing metadata if present
if metadata_path.exists():
    df_existing = pd.read_csv(metadata_path)
    # Remove duplicates based on video_id (keep new ones)
    df_existing = df_existing[~df_existing['video_id'].isin(df_new['video_id'])]
    df_combined = pd.concat([df_existing, df_new], ignore_index=True)
```

**Benefits:**
- Appends new entries to existing metadata.csv
- Removes duplicates (keeps newest)
- Shows total entry count

---

## Usage in Colab

### First Run

```python
!python scripts/extract_poses_parallel.py \
    --video-dir data/videos/clips \
    --output-dir data/processed/poses \
    --model-complexity 1 \
    --target-fps 20 \
    --num-workers 4
```

**Output:**
```
Found 23531 video files
Found 0 existing pose files (will skip these)
Estimated remaining clips to process: 23531
Extracting poses: 0% 0/23531
```

### After Interruption (Resume)

Simply re-run the **exact same command**:

```python
!python scripts/extract_poses_parallel.py \
    --video-dir data/videos/clips \
    --output-dir data/processed/poses \
    --model-complexity 1 \
    --target-fps 20 \
    --num-workers 4
```

**Output:**
```
Found 23531 video files
Found 1500 existing pose files (will skip these)
Estimated remaining clips to process: 22031
Extracting poses: 0% 0/23531 [will skip 1500]
```

**Processing:**
- First 1500 clips: Skipped instantly (~0.01s each)
- Remaining 22031: Full extraction (~2.4s each)

---

## Benefits

✅ **No progress loss** - Resume from any interruption
✅ **Fast skip** - Existing files checked in milliseconds
✅ **Corruption detection** - Re-extracts invalid files
✅ **Metadata preservation** - Combines old + new entries
✅ **Progress visibility** - Checkpoints every 100 clips

---

## Expected Behavior

### Normal Run (No Interruptions)
```
Found 23531 video files
Found 0 existing pose files (will skip these)
Estimated remaining clips to process: 23531

Extracting poses: 100% 23531/23531 [4:32:15<00:00, 2.41s/it]

EXTRACTION COMPLETE
Total videos: 23531
Successfully processed: 23420
Skipped (already exist): 0
Failed: 111
```

### Resume After Interruption
```
Found 23531 video files
Found 10000 existing pose files (will skip these)
Estimated remaining clips to process: 13531

Extracting poses: 100% 23531/23531 [1:55:22<00:00, 1.82s/it]

EXTRACTION COMPLETE
Total videos: 23531
Successfully processed: 13450
Skipped (already exist): 10000
Failed: 81

✓ Metadata updated: data/metadata.csv
  Previous entries: 10000
  New entries: 13450
  Total entries: 23450
```

### Multiple Interruptions
```bash
# Run 1: Process 5000 clips, interrupted
# Run 2: Process 8000 more clips, interrupted
# Run 3: Process remaining 10531 clips, complete
```

**Final result:**
- All 23531 clips processed
- Single metadata.csv with all entries
- No duplicates

---

## Troubleshooting

### Issue: "File exists but re-extracting"

**Cause:** Pose file is corrupted or empty
**Solution:** Automatic re-extraction (working as intended)

### Issue: "Metadata has duplicates"

**Cause:** Multiple runs with different video sets
**Solution:** Script automatically deduplicates based on `video_id`

### Issue: "Skipped count doesn't match existing files"

**Cause:** Existing files in output directory from different video set
**Solution:** Normal - only videos in current video-dir are processed

---

## Performance

| Scenario | Time Saved |
|----------|------------|
| Resume after 1000 clips | ~40 minutes (skip 1000 × 2.4s) |
| Resume after 5000 clips | ~3.3 hours |
| Resume after 10000 clips | ~6.7 hours |

**Skip speed:** ~0.01s per file (240x faster than extraction)

---

## Files Modified

- [scripts/extract_poses_parallel.py](../scripts/extract_poses_parallel.py)
  - Added file size validation
  - Added pickle integrity check
  - Added progress reporting
  - Added metadata merging
  - Added checkpoint logging

---

**Status:** Production ready - Safe to use in Colab with interruptions
