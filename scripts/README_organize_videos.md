# Video Organization Scripts - Quick Reference

## Problem Solved
Automatically organize video clips into stroke type folders (clear/smash/drop/lift) based on filename patterns.

## Which Script to Use?

### ✅ Use `organize_gcs_videos.sh` if:
- Running in Colab/GCS terminal
- Videos are already in GCS
- Want fastest execution (no downloads)

### ✅ Use `organize_videos.py` if:
- Working with local files
- Need more detailed progress reporting
- Want copy instead of move option

## Quick Start (Colab)

```bash
# Clone repo and navigate to directory
cd /content/iti123_v2

# Preview what will be organized
bash scripts/organize_gcs_videos.sh gs://iti123storage/videos/clips

# Review the output, then execute
bash scripts/organize_gcs_videos.sh gs://iti123storage/videos/clips --execute
```

## Filename Patterns Detected

The scripts automatically detect stroke types from these patterns:

| Stroke | Example Filenames (Actual Format) |
|--------|----------------------------------|
| **Clear** | `01_set1_rally11_ball19_Clear.mp4`, `02_set2_rally05_ball08_Clear.mp4` |
| **Smash** | `01_set1_rally10_ball15_Smash.mp4`, `03_set1_rally13_ball06_Smash.mp4` |
| **Drop** | `01_set1_rally10_ball10_Drop.mp4`, `02_set3_rally14_ball13_Drop.mp4` |
| **Lift** | `01_set1_rally10_ball12_Lift.mp4`, `04_set2_rally11_ball11_Lift.mp4` |

**Your actual GCS dataset contains:**
- 4,641 Smash videos
- 3,179 Drop videos
- 2,662 Clear videos
- 573 Lift videos
- **Total: 11,055 videos**

Detection is **case-insensitive** and looks for these patterns (in priority order):

1. **Primary pattern** (matches your actual files):
   - `_Clear.mp4`, `_Smash.mp4`, `_Drop.mp4`, `_Lift.mp4` at end of filename

2. **Fallback patterns** (for other naming conventions):
   - Full words: `clear`, `smash`, `drop`, `lift`
   - Abbreviations: `clr`, `smsh`, `drp`, `lft`
   - Delimited codes: `_c_`, `_s_`, `_d_`, `_l_`
   - Numbered codes: `_c001`, `_s042`, etc.

## Expected Output

### Before Organization
```
gs://iti123storage/videos/clips/
├── 01_set1_rally10_ball10_Drop.mp4
├── 01_set1_rally10_ball12_Lift.mp4
├── 01_set1_rally10_ball15_Smash.mp4
├── 01_set1_rally11_ball19_Clear.mp4
├── 01_set1_rally11_ball20_Clear.mp4
└── ... (11,055 files total)
```

### After Organization
```
gs://iti123storage/videos/clips/
├── clear/
│   ├── 01_set1_rally11_ball19_Clear.mp4
│   ├── 01_set1_rally11_ball20_Clear.mp4
│   └── ... (2,662 files)
├── smash/
│   ├── 01_set1_rally10_ball15_Smash.mp4
│   └── ... (4,641 files)
├── drop/
│   ├── 01_set1_rally10_ball10_Drop.mp4
│   └── ... (3,179 files)
└── lift/
    ├── 01_set1_rally10_ball12_Lift.mp4
    └── ... (573 files)
```

## Safety Features

1. **Dry-run by default** - Always previews changes first
2. **Skip organized files** - Won't move files already in stroke folders
3. **Unknown handling** - Files that don't match go to `unknown/` folder
4. **No data loss** - Uses `mv` (move), not delete

## Troubleshooting

### Error: "gsutil not found"
```bash
# Colab already has gsutil, but if needed:
pip install gsutil
```

### Error: "No video files found"
```bash
# Check if files exist
gsutil ls gs://iti123storage/videos/clips/*.mp4

# Check current organization
gsutil ls gs://iti123storage/videos/clips/clear/
gsutil ls gs://iti123storage/videos/clips/smash/
```

### Files detected incorrectly
The script looks for stroke type keywords in the filename. If detection is wrong:
1. Check the filename contains one of the expected patterns
2. Rename files to include clear patterns (e.g., `_clear_`, `_smash_`)
3. Manually move misclassified files after organization

### Want to undo organization
```bash
# Move all files back to parent directory
gsutil -m mv gs://iti123storage/videos/clips/clear/* gs://iti123storage/videos/clips/
gsutil -m mv gs://iti123storage/videos/clips/smash/* gs://iti123storage/videos/clips/
```

## Performance

| Dataset Size | Dry-run Time | Execution Time |
|--------------|--------------|----------------|
| 1,000 videos | ~30 sec | ~2 min |
| 10,000 videos | ~3 min | ~15 min |
| 11,055 videos | ~3.5 min | ~17 min |

## Complete Example Session

```bash
# Step 1: Navigate to project
cd /content/iti123_v2

# Step 2: Check current state
gsutil ls gs://iti123storage/videos/clips/ | head -10

# Step 3: Preview organization (DRY RUN)
bash scripts/organize_gcs_videos.sh gs://iti123storage/videos/clips

# Expected output:
# ============================================================
# VIDEO ORGANIZATION - GCS
# ============================================================
# GCS Path: gs://iti123storage/videos/clips
# Mode: DRY-RUN
#
# Listing video files...
# Found 11055 video files to organize
#
# Detected stroke types:
#   clear   : 5527 files
#   smash   : 5528 files
#
# Processing clear (5527 files)...
# [MOVE] player1_clear_001.mp4 -> clear/
# [MOVE] player2_clear_002.mp4 -> clear/
# ...

# Step 4: Review output and verify it looks correct

# Step 5: Execute organization
bash scripts/organize_gcs_videos.sh gs://iti123storage/videos/clips --execute

# Step 6: Verify organization
echo "Clear videos:"
gsutil ls gs://iti123storage/videos/clips/clear/ | wc -l

echo "Smash videos:"
gsutil ls gs://iti123storage/videos/clips/smash/ | wc -l

# Step 7: Continue with workflow
# Now your videos are organized and ready for pose extraction
```

## Integration with Workflow

After organizing videos, continue with the complete workflow:

```python
# In complete_workflow_colab.ipynb

# Phase 2.1: Download organized videos
!gsutil -m rsync -r gs://iti123storage/videos/clips/ data/videos/clips/

# Phase 2.2: Extract poses
!colab_venv/bin/python scripts/extract_poses_parallel.py \
    --video-dir data/videos/clips \
    --output-dir data/processed/poses \
    --model-complexity 1 \
    --target-fps 20 \
    --num-workers 4
```

## Need Help?

- Full documentation: [docs/VIDEO_ORGANIZATION_GUIDE.md](../docs/VIDEO_ORGANIZATION_GUIDE.md)
- Workflow overview: [WORKFLOW_OVERVIEW.md](../WORKFLOW_OVERVIEW.md)
- Complete notebook: [notebooks/complete_workflow_colab.ipynb](../notebooks/complete_workflow_colab.ipynb)

## Command Reference

```bash
# GCS Script (Recommended for Colab)
bash scripts/organize_gcs_videos.sh gs://BUCKET/PATH              # Preview
bash scripts/organize_gcs_videos.sh gs://BUCKET/PATH --execute    # Execute

# Python Script (Local files)
python scripts/organize_videos.py --input LOCAL/PATH --dry-run    # Preview
python scripts/organize_videos.py --input LOCAL/PATH --copy       # Copy files
python scripts/organize_videos.py --input LOCAL/PATH --move       # Move files

# Python Script (GCS files)
python scripts/organize_videos.py --gcs gs://BUCKET/PATH --dry-run  # Preview
python scripts/organize_videos.py --gcs gs://BUCKET/PATH --move     # Execute
```
