# Video Organization Guide

This guide explains how to organize video clips into stroke type folders (clear, smash, drop, lift) based on filename patterns.

## Quick Start

### In Colab/GCS Terminal (Recommended)

```bash
# Preview what will be organized
bash scripts/organize_gcs_videos.sh gs://iti123storage/videos/clips

# Execute the organization
bash scripts/organize_gcs_videos.sh gs://iti123storage/videos/clips --execute
```

### Local Files (Python)

```bash
# Preview what will be organized
python scripts/organize_videos.py --input data/videos/clips --dry-run

# Copy files to organized folders
python scripts/organize_videos.py --input data/videos/clips --copy

# Move files to organized folders
python scripts/organize_videos.py --input data/videos/clips --move
```

## How It Works

### Stroke Type Detection

The scripts detect stroke types from filenames using these patterns:

**Your Dataset (11,055 videos in `gs://iti123storage/videos/clips/`):**
- 4,641 Smash videos (format: `*_Smash.mp4`)
- 3,179 Drop videos (format: `*_Drop.mp4`)
- 2,662 Clear videos (format: `*_Clear.mp4`)
- 573 Lift videos (format: `*_Lift.mp4`)

| Stroke | Primary Pattern | Fallback Patterns |
|--------|----------------|-------------------|
| **Clear** | `_Clear.mp4` | `clear`, `clr`, `_c_`, `_c001` |
| **Smash** | `_Smash.mp4` | `smash`, `smsh`, `_s_`, `_s001` |
| **Drop** | `_Drop.mp4` | `drop`, `drp`, `_d_`, `_d001` |
| **Lift** | `_Lift.mp4` | `lift`, `lft`, `_l_`, `_l001` |
| **Unknown** | Files that don't match any pattern |

Detection is case-insensitive. The scripts check the primary pattern first (exact match at end of filename), then fallback to alternative patterns.

### File Organization

Before:
```
gs://iti123storage/videos/clips/
  01_set1_rally10_ball10_Drop.mp4
  01_set1_rally10_ball12_Lift.mp4
  01_set1_rally10_ball15_Smash.mp4
  01_set1_rally11_ball19_Clear.mp4
  ... (11,055 files)
```

After:
```
gs://iti123storage/videos/clips/
  clear/
    01_set1_rally11_ball19_Clear.mp4
    ... (2,662 files)
  smash/
    01_set1_rally10_ball15_Smash.mp4
    ... (4,641 files)
  drop/
    01_set1_rally10_ball10_Drop.mp4
    ... (3,179 files)
  lift/
    01_set1_rally10_ball12_Lift.mp4
    ... (573 files)
```

## Scripts Available

### 1. `organize_gcs_videos.sh` (Bash - GCS)

**Best for:** Running in Colab terminal or GCS console

**Features:**
- Works directly with GCS using `gsutil`
- Fast - no downloads required
- Dry-run mode by default (safe)
- Color-coded output
- Detects already-organized files

**Usage:**
```bash
# Dry run (preview only)
bash scripts/organize_gcs_videos.sh gs://bucket/videos/clips

# Execute
bash scripts/organize_gcs_videos.sh gs://bucket/videos/clips --execute
```

**Requirements:**
- `gsutil` installed (included in Colab)
- GCS authentication configured

### 2. `organize_videos.py` (Python - Local & GCS)

**Best for:** Local file organization or scripted workflows

**Features:**
- Supports both local files and GCS paths
- Copy or move modes
- Dry-run mode
- Detailed progress output
- Handles multiple video formats (.mp4, .avi, .mov)

**Usage:**

**Local files:**
```bash
# Preview
python scripts/organize_videos.py --input data/videos/clips --dry-run

# Copy files
python scripts/organize_videos.py --input data/videos/clips --copy

# Move files
python scripts/organize_videos.py --input data/videos/clips --move
```

**GCS files:**
```bash
# Preview
python scripts/organize_videos.py --gcs gs://bucket/videos/clips --dry-run

# Execute
python scripts/organize_videos.py --gcs gs://bucket/videos/clips --move
```

## Common Workflows

### Workflow 1: Organize Existing GCS Videos

```bash
# Step 1: Preview changes
bash scripts/organize_gcs_videos.sh gs://iti123storage/videos/clips

# Step 2: Review output and verify it looks correct

# Step 3: Execute
bash scripts/organize_gcs_videos.sh gs://iti123storage/videos/clips --execute
```

### Workflow 2: Organize Before Upload

```bash
# Step 1: Organize locally
python scripts/organize_videos.py --input data/videos/clips --move

# Step 2: Upload organized structure to GCS
gsutil -m rsync -r data/videos/clips/ gs://iti123storage/videos/clips/
```

### Workflow 3: Download, Organize, Upload

```bash
# Step 1: Download from GCS
gsutil -m rsync -r gs://iti123storage/videos/clips/ data/videos/clips/

# Step 2: Organize locally
python scripts/organize_videos.py --input data/videos/clips --move

# Step 3: Upload back to GCS
gsutil -m rsync -r data/videos/clips/ gs://iti123storage/videos/clips/
```

## Safety Features

### Dry-Run Mode

Both scripts default to **dry-run mode** - they show what will be done without making changes.

```bash
# These are SAFE - they only preview
bash scripts/organize_gcs_videos.sh gs://bucket/path
python scripts/organize_videos.py --input data/videos --dry-run
python scripts/organize_videos.py --gcs gs://bucket/path --dry-run
```

### Already-Organized Detection

The scripts automatically skip files that are already in stroke type folders:

```
✓ Skipped: clear/player1_clear_001.mp4 (already organized)
```

### Unknown Stroke Handling

Files that don't match any pattern are organized into an `unknown/` folder for manual review:

```
⚠️  Unknown stroke type detected: 5 files
Review these files and rename or move manually if needed
```

## Troubleshooting

### Issue: "No video files found"

**Solution:**
```bash
# Check if files exist
gsutil ls gs://bucket/videos/clips/

# Try with different patterns
gsutil ls gs://bucket/videos/clips/*.mp4
gsutil ls -r gs://bucket/videos/clips/**/*.mp4
```

### Issue: "gsutil not found"

**Solution:**
```bash
# Install gsutil (Colab)
pip install gsutil

# Or use gcloud SDK
gcloud auth login
```

### Issue: Files detected incorrectly

**Solution:**
1. Review the detection patterns in the script
2. Rename files to match expected patterns:
   - Use full words: `clear`, `smash`, `drop`, `lift`
   - Or use abbreviations: `clr`, `smsh`, `drp`, `lft`
   - Or use delimiters: `_c_`, `_s_`, `_d_`, `_l_`

### Issue: Want to undo organization

**Solution:**
```bash
# Move files back to parent directory
gsutil -m mv gs://bucket/videos/clips/clear/* gs://bucket/videos/clips/
gsutil -m mv gs://bucket/videos/clips/smash/* gs://bucket/videos/clips/
gsutil -m mv gs://bucket/videos/clips/drop/* gs://bucket/videos/clips/
gsutil -m mv gs://bucket/videos/clips/lift/* gs://bucket/videos/clips/
```

## Performance

| Operation | 1,000 files | 10,000 files |
|-----------|-------------|--------------|
| Dry run | ~30 sec | ~3 min |
| GCS organization | ~2 min | ~15 min |
| Local organization | ~1 min | ~10 min |

Performance varies based on:
- Network speed (for GCS)
- File sizes
- Number of parallel operations

## Integration with Complete Workflow

The video organization step is **optional** and included in the complete workflow notebook:

```python
# In complete_workflow_colab.ipynb
# Phase 2, Step 2.0 (Optional)

# Organize videos before extraction
!bash scripts/organize_gcs_videos.sh gs://{GCS_BUCKET}/videos/clips
# Review output, then execute:
!bash scripts/organize_gcs_videos.sh gs://{GCS_BUCKET}/videos/clips --execute
```

## Best Practices

1. **Always dry-run first** - Review output before executing
2. **Use GCS script for cloud files** - Faster than download + organize + upload
3. **Verify patterns** - Check a few files manually after organization
4. **Handle unknowns** - Review and manually organize files in `unknown/` folder
5. **Backup before moving** - Use `--copy` for local files if unsure

## Next Steps After Organization

Once videos are organized:

1. **Verify organization:**
   ```bash
   gsutil ls gs://bucket/videos/clips/clear/ | wc -l
   gsutil ls gs://bucket/videos/clips/smash/ | wc -l
   ```

2. **Proceed with pose extraction:**
   ```bash
   # Download organized videos
   gsutil -m rsync -r gs://bucket/videos/clips/ data/videos/clips/

   # Extract poses
   python scripts/extract_poses_parallel.py \
       --video-dir data/videos/clips \
       --output-dir data/processed/poses
   ```

3. **Continue with Phase 2 workflow** (feature engineering and validation)

---

## Reference

### File Paths

| Script | Path |
|--------|------|
| Bash (GCS) | `scripts/organize_gcs_videos.sh` |
| Python (All) | `scripts/organize_videos.py` |
| This guide | `docs/VIDEO_ORGANIZATION_GUIDE.md` |

### Related Documentation

- [COLAB_QUICKSTART.md](../notebooks/COLAB_QUICKSTART.md) - Complete workflow
- [WORKFLOW_OVERVIEW.md](../WORKFLOW_OVERVIEW.md) - Process diagram
- [complete_workflow_colab.ipynb](../notebooks/complete_workflow_colab.ipynb) - End-to-end notebook

---

**Questions?** Check the script help:
```bash
bash scripts/organize_gcs_videos.sh
python scripts/organize_videos.py --help
```
