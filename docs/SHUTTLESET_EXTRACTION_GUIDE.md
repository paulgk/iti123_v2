# ShuttleSet Video Clip Extraction Guide

**Date:** 2026-02-01
**Purpose:** Extract badminton shot clips from raw match videos using ShuttleSet CSV annotations

---

## Overview

This guide explains how to extract video clips for **Smash, Clear, Drop, Lift, and Drive** shots from downloaded ShuttleSet match videos.

### What Gets Extracted

The extraction script processes ShuttleSet CSV annotations and creates 3-second video clips centered around each shot's contact frame:

- **Pre-buffer:** 1.0 seconds before contact (preparation phase)
- **Contact frame:** Moment racket hits shuttle
- **Post-buffer:** 2.0 seconds after contact (follow-through + outcome)
- **Total duration:** 3.0 seconds per clip

### Shot Type Mapping

The script automatically maps ShuttleSet's 19 shot types to our 5 target classes:

| Target Class | ShuttleSet Types Included | Count in Dataset |
|--------------|---------------------------|------------------|
| **Smash** | Smash, Steep_Smash | 4,234 shots (11.6%) |
| **Clear** | Clear, Clear (Long) | 2,922 shots (8.0%) |
| **Drop** | Drop, Drop Shot (Soft), Slice_Drop | 10,082 shots (27.6%) |
| **Lift** | Lift, Lift / Clear (Defensive), Defensive_Lift | 5,632 shots (15.4%) |
| **Drive** | Drive, Rear_Drive, Defensive_Drive, Push, Short_Drive | 4,504 shots (12.3%) |

**Total extractable shots:** 27,374 (75% of ShuttleSet dataset)

**Excluded shot types:**
- Net shots (Block, Net_Kill, Cross_Net) - Phase 5+
- Service shots (Short_Serve, Long_Serve) - out of scope
- Slice shots requiring IMU (Overhead_Slice) - future research
- Unknown shots

---

## Prerequisites

### 1. Downloaded Match Videos

Videos should be in `data/raw_videos/` with match ID as filename:

```
data/raw_videos/
├── 01.mp4  # Kento MOMOTA vs CHOU Tien Chen - Fuzhou Open 2019 Finals
├── 02.mp4  # CHEN Long vs CHOU Tien Chen - World Tour Finals
├── 04.mp4  # CHEN Long vs CHOU Tien Chen - Denmark Open 2019
├── 05.mp4  # Kento MOMOTA vs CHOU Tien Chen - Fuzhou Open 2018
...
└── 44.mp4
```

**Note:** Match ID corresponds to the `id` field in `ShuttleSet/set/match.csv`

### 2. ShuttleSet CSV Annotations

The ShuttleSet dataset should be in the project root:

```
ShuttleSet/
└── set/
    ├── match.csv  # Match metadata (44 matches)
    └── {match_name}/
        ├── set1.csv  # Rally-level shot annotations
        ├── set2.csv
        └── set3.csv
```

### 3. FFmpeg Installed

```bash
# macOS
brew install ffmpeg

# Ubuntu/Debian
sudo apt-get install ffmpeg

# Verify installation
ffmpeg -version
```

---

## Usage

### Basic Commands

```bash
# 1. Preview extraction (dry-run) - see what will be extracted
python scripts/extract_shuttleset_clips.py --dry-run

# 2. Extract all shots from all available matches
python scripts/extract_shuttleset_clips.py --execute

# 3. Extract specific shot types only
python scripts/extract_shuttleset_clips.py --shot-types Smash Clear Drop --execute

# 4. Extract from specific matches
python scripts/extract_shuttleset_clips.py --match-ids 01 02 03 --execute

# 5. Custom clip duration
python scripts/extract_shuttleset_clips.py --duration 2.5 --execute

# 6. Adjust pre/post buffers
python scripts/extract_shuttleset_clips.py --pre-buffer 0.5 --post-buffer 2.5 --execute
```

### Command-Line Options

| Option | Default | Description |
|--------|---------|-------------|
| `--input` | `data/raw_videos` | Input directory with match videos |
| `--output` | `data/clips` | Output directory for extracted clips |
| `--shuttleset` | `ShuttleSet` | ShuttleSet dataset directory |
| `--shot-types` | All 5 types | Shot types to extract: Smash, Clear, Drop, Lift, Drive |
| `--match-ids` | All matches | Specific match IDs to process (e.g., 01 02 03) |
| `--duration` | `3.0` | Clip duration in seconds |
| `--pre-buffer` | `1.0` | Seconds before contact frame |
| `--post-buffer` | `2.0` | Seconds after contact frame |
| `--dry-run` | - | Preview extraction without creating clips |
| `--execute` | - | Execute actual extraction (required) |

---

## Output Structure

Clips are organized by shot type in separate folders:

```
data/clips/
├── Smash/
│   ├── 01_set1_rally02_ball03_Smash.mp4
│   ├── 01_set1_rally03_ball04_Smash.mp4
│   └── ... (4,234 clips expected)
├── Clear/
│   ├── 01_set1_rally01_ball02_Clear.mp4
│   └── ... (2,922 clips expected)
├── Drop/
│   ├── 01_set1_rally03_ball02_Drop.mp4
│   └── ... (10,082 clips expected)
├── Lift/
│   ├── 01_set1_rally03_ball03_Lift.mp4
│   └── ... (5,632 clips expected)
└── Drive/
    ├── 01_set1_rally02_ball02_Drive.mp4
    └── ... (4,504 clips expected)
```

### Filename Format

```
{match_id}_set{set_num}_rally{rally}_ball{ball_round}_{shot_type}.mp4
```

Examples:
- `01_set1_rally02_ball03_Smash.mp4` - Match 01, Set 1, Rally 2, Ball 3, Smash
- `13_set2_rally15_ball08_Drop.mp4` - Match 13, Set 2, Rally 15, Ball 8, Drop

---

## Example Workflow

### Step 1: Dry-Run Preview

```bash
# Preview extraction for a single match
python scripts/extract_shuttleset_clips.py --match-ids 01 --dry-run
```

**Expected output:**
```
================================================================================
SHUTTLESET VIDEO CLIP EXTRACTION
================================================================================
Input videos:  data/raw_videos
Output clips:  data/clips
ShuttleSet:    ShuttleSet
Shot types:    Smash, Clear, Drop, Lift, Drive
Clip duration: 3.0s (pre: 1.0s, post: 2.0s)
Mode:          DRY-RUN (preview only)
================================================================================

Loading match metadata...
✓ Loaded 44 matches from ShuttleSet
✓ Filtered to 1 specified matches: 01

Processing Match 01: Fuzhou Open 2019 - Finals
  Players: Kento MOMOTA vs CHOU Tien Chen
  Shots found: 1087 total
    Smash: 188
    Clear: 218
    Drop: 236
    Lift: 214
    Drive: 231
[DRY-RUN] Would extract: 01_set1_rally01_ball02_Clear.mp4
[DRY-RUN] Would extract: 01_set1_rally02_ball02_Drive.mp4
...
```

### Step 2: Extract Small Batch (Testing)

```bash
# Extract from 3 matches to verify quality
python scripts/extract_shuttleset_clips.py --match-ids 01 02 04 --execute
```

**Expected output:**
```
Processing Match 01: Fuzhou Open 2019 - Finals
  Players: Kento MOMOTA vs CHOU Tien Chen
  Shots found: 1087 total
    Smash: 188
    Clear: 218
    Drop: 236
    Lift: 214
    Drive: 231
  ✓ Extracted: 1087 clips, Failed: 0

Processing Match 02: World Tour Finals - Group-Stage
  Players: CHEN Long vs CHOU Tien Chen
  Shots found: 892 total
    Smash: 145
    Clear: 189
    Drop: 201
    Lift: 178
    Drive: 179
  ✓ Extracted: 892 clips, Failed: 0

...

================================================================================
EXTRACTION SUMMARY
================================================================================
Matches processed:     3/3
Total clips extracted: 2,847

Clips by shot type:
  Smash        512 clips
  Clear        623 clips
  Drop         701 clips
  Lift         556 clips
  Drive        455 clips

✓ Clips saved to: data/clips
```

### Step 3: Verify Quality

```bash
# Check a sample clip
ffplay data/clips/Smash/01_set1_rally02_ball03_Smash.mp4

# Check clip durations
for f in data/clips/Smash/*.mp4; do
    ffprobe -v error -show_entries format=duration -of default=noprint_wrappers=1:nokey=1 "$f" | head -5
done
```

### Step 4: Extract All Matches

```bash
# Extract all available matches (this will take time!)
python scripts/extract_shuttleset_clips.py --execute

# Or run in background
nohup python scripts/extract_shuttleset_clips.py --execute > extraction.log 2>&1 &

# Monitor progress
tail -f extraction.log
```

**Estimated time:** ~30-60 minutes for all 44 matches (depends on CPU and disk speed)

---

## Expected Results

### Match 01 Example (Fuzhou Open 2019 Finals)

From the dry-run output, Match 01 should yield:

- **Smash:** 188 clips
- **Clear:** 218 clips
- **Drop:** 236 clips
- **Lift:** 214 clips
- **Drive:** 231 clips
- **Total:** 1,087 clips from one 3-set match

### Full Dataset Projection

Based on ShuttleSet shot distribution (27,374 extractable shots across 44 matches):

| Shot Type | Expected Clips | Percentage |
|-----------|----------------|------------|
| **Drop** | ~10,082 | 36.8% |
| **Lift** | ~5,632 | 20.6% |
| **Drive** | ~4,504 | 16.5% |
| **Smash** | ~4,234 | 15.5% |
| **Clear** | ~2,922 | 10.7% |
| **Total** | **~27,374** | **100%** |

**Note:** Actual counts depend on available match videos. Some matches may be missing (e.g., Match ID 03, 12 per the current dataset).

---

## Troubleshooting

### Issue: "Video file not found"

```
⚠️  Match 03: Video file not found: 03.mp4
```

**Solution:** Download the missing match video or skip it. The script will continue with available matches.

### Issue: "No shot annotations found"

```
⚠️  Warning: Match directory not found: ShuttleSet/set/{match_name}
```

**Solution:** Ensure ShuttleSet CSV files are present. Check that directory names match `match.csv` video names exactly.

### Issue: FFmpeg extraction fails

```
✗ Error extracting 01_set1_rally02_ball03_Smash.mp4: ...
```

**Solution:** The script automatically retries with re-encoding if fast copy fails. If errors persist:
1. Check FFmpeg installation: `ffmpeg -version`
2. Verify input video is not corrupted: `ffplay data/raw_videos/01.mp4`
3. Check disk space: `df -h`

### Issue: Clips are corrupted or incomplete

**Possible causes:**
- Timestamp errors in CSV (contact frame beyond video duration)
- Video encoding issues in source file

**Solution:**
1. Verify source video plays correctly
2. Check clip with: `ffplay data/clips/Smash/problematic_clip.mp4`
3. Re-extract with re-encoding: The script will automatically attempt this

---

## Performance Tips

### Speed Up Extraction

```bash
# 1. Process specific shot types (less I/O)
python scripts/extract_shuttleset_clips.py --shot-types Smash Clear --execute

# 2. Process matches in parallel (if you have multiple CPUs)
python scripts/extract_shuttleset_clips.py --match-ids 01 02 03 --execute &
python scripts/extract_shuttleset_clips.py --match-ids 04 05 06 --execute &
python scripts/extract_shuttleset_clips.py --match-ids 07 08 09 --execute &
wait

# 3. Use SSD for output directory (faster writes)
python scripts/extract_shuttleset_clips.py --output /path/to/ssd/clips --execute
```

### Disk Space Requirements

- **Input videos:** ~50 GB (44 matches, avg 1-2 GB each)
- **Output clips:** ~15-20 GB (27,374 clips × 3 seconds × compression)
- **Total recommended:** 100 GB free space (buffer for processing)

---

## Next Steps

After extraction completes:

### 1. Verify Clip Quality

```bash
# Count extracted clips
find data/clips -name "*.mp4" | wc -l

# Check distribution
for shot in Smash Clear Drop Lift Drive; do
    count=$(find data/clips/$shot -name "*.mp4" | wc -l)
    echo "$shot: $count clips"
done
```

### 2. Run Pose Extraction

```bash
# Extract pose keypoints from all clips
python scripts/extract_poses_parallel.py \
    --input data/clips \
    --output data/pose_data \
    --parallel 4
```

### 3. Validate Pose Quality

```bash
# Check pose extraction success rate
python scripts/validate_pose_quality.py --input data/pose_data
```

### 4. Update Dataset Paths

Update feature extraction scripts to use new clip locations:

```python
# In feature extraction config
VIDEO_DIR = Path('data/clips')  # Instead of GCS path
SHOT_TYPES = ['Smash', 'Clear', 'Drop', 'Lift', 'Drive']
```

---

## Shot Type Mapping Details

### Why Merge Shot Types?

Some ShuttleSet shot types are merged into broader classes because:

1. **Biomechanical similarity:** Steep_Smash vs Smash differ only in angle (Cohen's d < 0.5)
2. **Pose estimation limitations:** Slice_Drop requires racket face tracking (not available)
3. **Training data strategy:** Start with clear classes, refine later if needed

### Merged Classes

| Our Class | Merged ShuttleSet Types | Rationale |
|-----------|-------------------------|-----------|
| **Smash** | Smash + Steep_Smash | Steep smash is angle variant, not distinct technique |
| **Drop** | Drop + Slice_Drop | Slice requires IMU sensors (future work) |
| **Drive** | Drive + Rear_Drive + Defensive_Drive + Push + Short_Drive | All mid-court flat shots with weak inter-class discrimination |
| **Lift** | Lift + Defensive_Lift | Emergency lift is stance variant, same biomechanics |

### Future Refinement (Phase 6+)

After base 5-class model is validated (accuracy > 75%), consider splitting:

- **Smash** → Standard Smash, Steep Smash (if d > 0.5)
- **Drop** → Soft Drop, Slice Drop (requires IMU sensors)
- **Drive** → Drive, Push (if d > 0.5 and >1,000 samples each)

---

## Script Implementation Details

### Timestamp Parsing

The script parses CSV timestamps in multiple formats:

```python
"0:07:39"   → 459.0 seconds (7 min 39 sec)
"00:07:43"  → 463.0 seconds
"1:23:45"   → 5025.0 seconds (1 hr 23 min 45 sec)
```

### FFmpeg Clip Extraction

Two-stage approach for robust extraction:

1. **Fast copy (no re-encoding):** Attempts `-c copy` first (instant extraction)
2. **Re-encode fallback:** If copy fails, re-encodes with `-c:v libx264` (slower but compatible)

### Error Handling

- **Missing videos:** Skips match, logs warning, continues with next
- **Missing CSVs:** Skips match, logs warning
- **Extraction timeouts:** 30s for copy, 60s for re-encode
- **All failures logged:** Summary shows failed clip count

---

## References

- ShuttleSet Dataset: 44 professional badminton matches (2018-2021)
- Shot type analysis: [docs/SHUTTLESET_DATASET_ANALYSIS.md](SHUTTLESET_DATASET_ANALYSIS.md)
- Top 5 trainable shots: [docs/TOP_5_TRAINABLE_SHOTS_ANALYSIS.md](TOP_5_TRAINABLE_SHOTS_ANALYSIS.md)
- Biomechanical features: [.planning/phases/02-feature-engineering-enhancement/02-BIOMECHANICS-SHOT-ANALYSIS.md](../.planning/phases/02-feature-engineering-enhancement/02-BIOMECHANICS-SHOT-ANALYSIS.md)

---

**Last Updated:** 2026-02-01
**Status:** Ready for clip extraction from local videos
