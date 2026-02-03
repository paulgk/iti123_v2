# Logging Enhancements - Phase 1.5

**Enhanced progress tracking and status reporting for all extraction scripts**

---

## Overview

All three extraction scripts now include comprehensive logging to monitor progress during long-running overnight extractions.

---

## 1. Clip Extraction (`extract_shuttleset_clips.py`)

### Enhanced Features

**Match Progress Tracking:**
```
[5/44] Match 05: BWF World Championships
  Players: Player A vs Player B
  Shots to extract: 487
    Smash: 92
    Clear: 65
    Drop: 143
    Lift: 127
    Drive: 60
```

**Periodic Progress Updates:**
```
    Progress: 250/487 clips (245 ok, 5 failed)
```
- Updates every 50 clips
- Shows running success/failure count

**Per-Match Summary:**
```
  ✓ Extracted: 482/487 clips in 125.3s
  ✗ Failed: 5
  Estimated time remaining: 45.2 minutes (39 matches left)
```

**Time Estimation:**
- Calculates average time per match
- Estimates remaining time after each match
- Updates dynamically based on actual progress

---

## 2. Metadata Creation (`create_metadata_csv.py`)

### Enhanced Features

**Match-by-Match Progress:**
```
Processing annotations...

[1/44] Match 01: 2018_Denmark_Open_Oushuijun_CHEN_JIA → 825 clips
[2/44] Match 02: 2018_Japan_Open_MOMOTA_Kento_Jonatan_CHRISTIE → 512 clips
[3/44] Match 03: 2018_Korea_Open_CHOU_Tien_Chen_LEE_Chong_Wei → 467 clips
...
```

**Final Summary:**
```
✓ Processed 44 matches
  Clips found:   19,778
  Clips missing: 0

Clips by shot type:
  Smash        4,234 ( 21.4%)
  Clear        2,922 ( 14.8%)
  Drop         6,290 ( 31.8%)
  Lift         5,632 ( 28.5%)
  Drive          700 (  3.5%)
```

---

## 3. Pose Extraction (`extract_poses_roi.py`)

### Enhanced Features

**Initial Breakdown:**
```
✓ Loaded 19,778 clips from metadata

Breakdown by shot type:
  Smash        4,234 clips
  Clear        2,922 clips
  Drop         6,290 clips
  Lift         5,632 clips
  Drive          700 clips

Extracting poses with ROI...
Using 8 worker(s) for parallel processing
```

**Progress Bar with Stats:**
```
Extracting poses: 2,345/19,778 [=>-------------------] 11.9% [00:45<05:23, 54.2clips/s]
```

**Periodic Status Reports (every 60 seconds):**
```
Progress: 3,250/19,778 (16.4%)
Success rate: 3,185/3,250 (98.0%)
Rate: 53.8 clips/sec
Estimated time remaining: 308.5 minutes
```

**Final Summary:**
```
================================================================================
EXTRACTION SUMMARY
================================================================================
Total clips:      19,778
Successful:       19,423
Failed:           355
Success rate:     98.2%
Processing time:  05:28:37
Avg time/clip:    0.99 seconds

Output directory: data/poses
Pose files saved: 19,423
================================================================================
```

---

## 4. Full Pipeline (`extract_full_pipeline.sh`)

### Enhanced Features

The pipeline script already includes comprehensive logging:

**Timestamped Logs:**
```
[2026-02-03 22:15:30] Starting STEP 1: Extract Clips
[2026-02-03 22:17:45] ✓ Step 1 complete: 825 clips extracted in 135s
[2026-02-03 22:17:50] Starting STEP 2: Create Metadata
[2026-02-03 22:17:55] ✓ Step 2 complete: 825 entries in metadata.csv in 5s
```

**Duration Breakdown:**
```
Duration breakdown:
  Step 1 (Clips):    00:02:15
  Step 2 (Metadata): 00:00:05
  Step 3 (Poses):    00:15:23
  Step 4 (Validate): 00:00:08
  Total:             00:17:51
```

**Success Rates:**
```
Results:
  Clips extracted: 825
  Metadata entries: 825
  Poses extracted:  788
  Success rate:     95.5%
```

---

## Benefits

### 1. **Real-Time Monitoring**
- Know exactly where you are in the extraction process
- Identify stuck/slow matches immediately
- Monitor success rates as extraction progresses

### 2. **Time Management**
- Accurate time remaining estimates
- Plan when to check back on overnight runs
- Identify bottlenecks (which step takes longest)

### 3. **Quality Assurance**
- Track failure rates in real-time
- Identify problematic matches early
- Catch issues before full completion

### 4. **Resource Planning**
- Processing rate helps optimize worker count
- Duration tracking helps plan future extractions
- Average time per clip helps estimate costs

---

## Example Output Sessions

### Quick Test (1 Match)

```bash
$ bash scripts/test_roi_extraction.sh

==========================================
ROI EXTRACTION TEST - MATCH 01
==========================================

Checking prerequisites...
✓ Match video found: data/raw_videos/01.mp4
✓ ShuttleSet annotations found
✓ MediaPipe model found

==========================================
STEP 1: Extract Clips
==========================================

[1/1] Match 01: 2018 Denmark Open
  Players: Chen vs Christie
  Shots to extract: 825
    Smash: 156
    Clear: 98
    Drop: 287
    Lift: 234
    Drive: 50

  ✓ Extracted: 825/825 clips in 135.2s

==========================================
STEP 2: Create Metadata
==========================================

[1/1] Match 01: 2018_Denmark_Open_Oushuijun_CHEN_JIA → 825 clips

✓ Created metadata with 825 entries

==========================================
STEP 3: Extract Poses with ROI
==========================================

Breakdown by shot type:
  Smash        156 clips
  Clear         98 clips
  Drop         287 clips
  Lift         234 clips
  Drive         50 clips

Extracting poses: 825/825 [========================] 100% [00:15<00:00, 55.0clips/s]

✓ Extracted 788 pose sequences

✓ Test completed successfully!
```

### Full Extraction (First Half)

```bash
$ bash scripts/extract_full_pipeline.sh 01 22

================================================================================
COMPLETE EXTRACTION PIPELINE
================================================================================
Match range: 01 to 22
Input:       data/raw_videos
Output:      Clips: data/clips | Poses: data/poses
Workers:     8
================================================================================

================================================================================
STEP 1: EXTRACT CLIPS (matches 01-22)
================================================================================

[1/22] Match 01: 2018 Denmark Open
  Shots to extract: 825
  ✓ Extracted: 825/825 clips in 135.2s
  Estimated time remaining: 47.3 minutes (21 matches left)

[2/22] Match 02: 2018 Japan Open
  Shots to extract: 512
  ✓ Extracted: 512/512 clips in 82.5s
  Estimated time remaining: 36.3 minutes (20 matches left)

...

✓ Step 1 complete: 9,889 clips extracted in 1847s

================================================================================
STEP 2: CREATE METADATA
================================================================================

[1/22] Match 01: 2018_Denmark_Open_Oushuijun_CHEN_JIA → 825 clips
[2/22] Match 02: 2018_Japan_Open_MOMOTA_Kento_Jonatan_CHRISTIE → 512 clips
...

✓ Step 2 complete: 9,889 entries in metadata.csv in 15s

================================================================================
STEP 3: EXTRACT POSES WITH ROI (8 workers)
================================================================================

Breakdown by shot type:
  Smash      2,117 clips
  Clear      1,461 clips
  Drop       3,145 clips
  Lift       2,816 clips
  Drive        350 clips

Extracting poses: 4,945/9,889 [=============>------] 50.0% [02:45<02:45, 54.2clips/s]

Progress: 4,945/9,889 (50.0%)
Success rate: 4,850/4,945 (98.1%)
Rate: 54.2 clips/sec
Estimated time remaining: 152.3 minutes

...

✓ Step 3 complete: 9,701 poses extracted in 18425s

Total:             05:07:02
```

---

## Monitoring Tips

### 1. Keep Terminal Open
- Run in tmux/screen for remote sessions
- Or redirect output to file: `bash scripts/extract_full_pipeline.sh 01 22 2>&1 | tee extraction.log`

### 2. Check Progress Remotely
```bash
# Check log file
tail -f logs/extraction_01_to_22_*.log

# Count current progress
watch -n 30 'find data/clips -name "*.mp4" | wc -l'
watch -n 30 'find data/poses -name "*.pkl" | wc -l'
```

### 3. Estimate Completion Time
- First match gives you baseline time
- ETA updates after each match
- Pose extraction reports every 60 seconds

### 4. Identify Issues
- Failed clip extractions show immediately
- Low pose extraction rate (<90%) indicates problems
- Missing player positions show in metadata creation

---

## Summary

**Before:** Silent processing with no feedback until completion
**After:** Real-time progress tracking, time estimates, and quality monitoring

All scripts now provide professional-grade logging suitable for:
- Overnight unattended runs
- Production data pipelines
- Quality monitoring and debugging
- Resource planning and optimization

---

**Status:** Implemented and tested
**Last updated:** 2026-02-03
