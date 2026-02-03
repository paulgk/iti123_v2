# GCS Upload/Download Guide

**Upload poses to GCS and download in Colab for training**

---

## Overview

Three scripts for managing data transfer with Google Cloud Storage:

1. **upload_poses_to_gcs.sh** - Full-featured upload with verification
2. **quick_upload_gcs.sh** - Simple wrapper for common uploads
3. **download_from_gcs_colab.sh** - Download data in Colab for training

---

## Quick Start

### Upload (Local Machine)

```bash
# Quick upload (poses + metadata)
bash scripts/quick_upload_gcs.sh

# Or upload specific items
bash scripts/quick_upload_gcs.sh poses      # Only poses
bash scripts/quick_upload_gcs.sh metadata   # Only metadata
bash scripts/quick_upload_gcs.sh clips      # Only clips (large!)
```

### Download (Google Colab)

```bash
# In Colab cell
!bash scripts/download_from_gcs_colab.sh
```

---

## Script 1: Full Upload (`upload_poses_to_gcs.sh`)

### Features

- Uploads poses, metadata, and optionally clips
- Dry-run mode for preview
- Progress tracking and verification
- File count and size comparison
- Confirmation before upload

### Configuration

Edit the script to customize:

```bash
# Upload options
UPLOAD_POSES=true      # Upload pose files
UPLOAD_METADATA=true   # Upload metadata CSV
UPLOAD_CLIPS=false     # Upload clips (set to true if needed)
DRY_RUN=false          # Set to true to preview without uploading
```

### Usage

```bash
# Step 1: Preview (dry-run)
# Edit script: DRY_RUN=true
bash scripts/upload_poses_to_gcs.sh

# Step 2: Review output

# Step 3: Execute upload
# Edit script: DRY_RUN=false
bash scripts/upload_poses_to_gcs.sh
```

### Example Output

```
================================================================================
UPLOAD POSES TO GCS
================================================================================
Bucket:   gs://iti123storage
Mode:     EXECUTE (uploading)
================================================================================

Checking prerequisites...
✓ gsutil found
✓ Authenticated to GCS
✓ Found 19,423 pose files
✓ Found metadata file

Current GCS bucket usage:
28.15 GB

================================================================================
UPLOAD PLAN
================================================================================
✓ Poses:    data/poses -> gs://iti123storage/features/poses_roi
✓ Metadata: data/metadata.csv -> gs://iti123storage/data/metadata_roi.csv
================================================================================

Proceed with upload? (y/n): y

[SECTION] Uploading Poses (ROI-extracted)
Local:  data/poses
GCS:    gs://iti123storage/features/poses_roi
Size:   19,423 files, 8.9G

Starting upload...
Building synchronization state...
Starting synchronization...
Copying file://data/poses/01_set1_rally03_ball05_Smash.pkl...
...
✓ Upload complete in 287s
✓ Verified: 19,423 files in GCS

[SECTION] Uploading Metadata (with player positions)
Local:  data/metadata.csv
GCS:    gs://iti123storage/data/metadata_roi.csv
Size:   2.1M

Uploading...
✓ Upload complete

================================================================================
VERIFICATION
================================================================================

[SECTION] Poses Verification
Local:  19,423 files
GCS:    19,423 files
✓ Pose count matches

[SECTION] Metadata Verification
Local:  2,156,789 bytes
GCS:    2,156,789 bytes
✓ Metadata size matches

New GCS bucket usage:
37.28 GB

================================================================================
UPLOAD COMPLETE
================================================================================

Upload successful!

GCS Locations:
  Poses:    gs://iti123storage/features/poses_roi
  Metadata: gs://iti123storage/data/metadata_roi.csv

To download in Colab:
  # Download poses
  !gsutil -m rsync -r gs://iti123storage/features/poses_roi/ ./data/poses/

  # Download metadata
  !gsutil cp gs://iti123storage/data/metadata_roi.csv ./data/metadata.csv
```

---

## Script 2: Quick Upload (`quick_upload_gcs.sh`)

### Features

- Simple command-line interface
- No configuration needed
- Automatic file detection
- Progress tracking

### Usage

```bash
# Upload poses + metadata (default)
bash scripts/quick_upload_gcs.sh

# Upload specific items
bash scripts/quick_upload_gcs.sh poses
bash scripts/quick_upload_gcs.sh metadata
bash scripts/quick_upload_gcs.sh clips
bash scripts/quick_upload_gcs.sh all
```

### Example

```bash
$ bash scripts/quick_upload_gcs.sh

================================================================================
QUICK UPLOAD TO GCS
================================================================================
Upload type: all
Bucket:      gs://iti123storage
================================================================================

[SECTION] UPLOADING POSES
Found 19,423 pose files
Uploading to gs://iti123storage/features/poses_roi/
Building synchronization state...
✓ Poses uploaded successfully

[SECTION] UPLOADING METADATA
File size: 2.1M
Uploading to gs://iti123storage/data/metadata_roi.csv
✓ Metadata uploaded successfully

Skipping clips (too large). To upload clips, run:
  bash scripts/quick_upload_gcs.sh clips

================================================================================
UPLOAD COMPLETE
================================================================================

To download in Colab:

# Download poses
!gsutil -m rsync -r gs://iti123storage/features/poses_roi/ ./data/poses/

# Download metadata
!gsutil cp gs://iti123storage/data/metadata_roi.csv ./data/metadata.csv
```

---

## Script 3: Download in Colab (`download_from_gcs_colab.sh`)

### Features

- Optimized for Google Colab
- Downloads poses and metadata
- Automatic verification
- Format validation
- Metadata-pose matching check

### Usage in Colab

```python
# Cell 1: Upload script to Colab
from google.colab import files
files.upload()  # Upload download_from_gcs_colab.sh

# Cell 2: Make executable and run
!chmod +x download_from_gcs_colab.sh
!bash download_from_gcs_colab.sh
```

Or clone repository:

```python
# Cell 1: Clone repo
!git clone https://github.com/your-repo/iti123_v2.git
%cd iti123_v2

# Cell 2: Download data
!bash scripts/download_from_gcs_colab.sh
```

### Example Output

```
================================================================================
DOWNLOAD DATA FROM GCS
================================================================================
Source:  gs://iti123storage
Target:  ./data/
================================================================================

Running in Google Colab environment

[SECTION] DOWNLOADING METADATA
Source: gs://iti123storage/data/metadata_roi.csv
Target: ./data/metadata.csv

✓ Metadata downloaded

Metadata sample:
video_id,match_id,set_num,rally,ball_round,shot_type,...
01_set1_rally03_ball05_Smash,1,1,3,5,Smash,...

Metadata entries: 19,778

[SECTION] DOWNLOADING POSES
Source: gs://iti123storage/features/poses_roi
Target: ./data/poses

This may take 5-10 minutes...
Starting download...
Building synchronization state...
✓ Poses downloaded in 342s
Pose files: 19,423

Verifying pose format...
Sample pose: 01_set1_rally03_ball05_Smash.pkl
  Shape: (87, 33, 3)
  Frames: 87
  Keypoints: 33
  Coordinates: 3
  Mean: 0.4523
  Std: 0.2134
✓ Pose format valid

================================================================================
VERIFICATION
================================================================================

Metadata entries: 19,778
Pose files:       19,423
Matching:         19,423
Success rate:     98.2%
⚠️  355 poses missing (still acceptable)

================================================================================
DOWNLOAD COMPLETE
================================================================================

Data ready for training!

File locations:
  Metadata: ./data/metadata.csv
  Poses:    ./data/poses/

Next steps:
  1. Verify data: python scripts/validate_roi_poses.py
  2. Train model: python scripts/train_models_fixed.py --model stgcn --epochs 50
```

---

## GCS File Structure

### After Upload

```
gs://iti123storage/
├── data/
│   ├── metadata_roi.csv              # Current metadata with player positions
│   ├── clips_roi/                    # Optional: clean clips (19,778 files)
│   │   ├── Smash/
│   │   ├── Clear/
│   │   ├── Drop/
│   │   ├── Lift/
│   │   └── Drive/
│   └── [old files to delete]
├── features/
│   ├── poses_roi/                    # ROI-extracted poses (19,423 files)
│   │   └── *.pkl
│   └── [old poses_* to delete]
└── outputs/
    └── models/
```

### After Download in Colab

```
./data/
├── metadata.csv        # Downloaded from metadata_roi.csv
└── poses/             # Downloaded from poses_roi/
    └── *.pkl          # 19,423 pose files
```

---

## Common Workflows

### Workflow 1: Initial Upload After Extraction

```bash
# 1. Extract data locally
bash scripts/extract_full_pipeline.sh 01 22

# 2. Clean up old GCS files
bash scripts/clean_gcs_interactive.sh

# 3. Upload new data
bash scripts/quick_upload_gcs.sh

# 4. Verify upload
gsutil du -sh gs://iti123storage
```

---

### Workflow 2: Train in Colab

```python
# Colab Notebook

# Cell 1: Authenticate
from google.colab import auth
auth.authenticate_user()

# Cell 2: Clone repository
!git clone https://github.com/your-repo/iti123_v2.git
%cd iti123_v2

# Cell 3: Download data
!bash scripts/download_from_gcs_colab.sh

# Cell 4: Verify data
!python scripts/validate_roi_poses.py --poses data/poses --metadata data/metadata.csv

# Cell 5: Train model
!python scripts/train_models_fixed.py \
    --metadata data/metadata.csv \
    --pose-dir data/poses \
    --output outputs/ \
    --model stgcn \
    --epochs 50
```

---

### Workflow 3: Incremental Upload (After Second Half)

```bash
# 1. Extract second half
bash scripts/extract_full_pipeline.sh 23 44

# 2. Upload new poses (rsync only uploads new files)
bash scripts/quick_upload_gcs.sh poses

# 3. Update metadata
bash scripts/quick_upload_gcs.sh metadata

# 4. Verify
gsutil ls gs://iti123storage/features/poses_roi/ | wc -l
# Should show total count (first half + second half)
```

---

## Manual Commands

### Upload Commands

```bash
# Upload poses
gsutil -m rsync -r data/poses/ gs://iti123storage/features/poses_roi/

# Upload metadata
gsutil cp data/metadata.csv gs://iti123storage/data/metadata_roi.csv

# Upload clips (large)
gsutil -m rsync -r data/clips/ gs://iti123storage/data/clips_roi/

# Check upload
gsutil ls -lh gs://iti123storage/features/poses_roi/ | head -10
```

### Download Commands (Colab)

```bash
# Download poses
!gsutil -m rsync -r gs://iti123storage/features/poses_roi/ ./data/poses/

# Download metadata
!gsutil cp gs://iti123storage/data/metadata_roi.csv ./data/metadata.csv

# Verify download
!find ./data/poses -name "*.pkl" | wc -l
```

---

## Troubleshooting

### Upload Issues

**Error: "gsutil: command not found"**
```bash
# Install Google Cloud SDK
brew install --cask google-cloud-sdk
```

**Error: "AccessDeniedException: 403"**
```bash
# Authenticate
gcloud auth login

# Set project
gcloud config set project YOUR_PROJECT_ID
```

**Upload is slow**
```bash
# Use -m flag for parallel upload (already in scripts)
gsutil -m rsync -r data/poses/ gs://iti123storage/features/poses_roi/

# Increase parallelism (advanced)
gsutil -o "GSUtil:parallel_process_count=16" -m rsync -r data/poses/ gs://iti123storage/features/poses_roi/
```

---

### Download Issues

**Error: "No such file or directory"**
```bash
# Check if file exists in GCS
!gsutil ls gs://iti123storage/data/metadata_roi.csv

# Check bucket structure
!gsutil ls -r gs://iti123storage/
```

**Download is slow in Colab**
```bash
# Already using -m flag for parallel download
# Colab download speed depends on Google's network

# Check download progress
!watch -n 10 'find ./data/poses -name "*.pkl" | wc -l'
```

**Verification fails (missing poses)**
```bash
# Check which poses are missing
python - <<EOF
import csv
from pathlib import Path

with open('data/metadata.csv', 'r') as f:
    reader = csv.DictReader(f)
    metadata_ids = {row['video_id'] for row in reader}

pose_files = {f.stem for f in Path('data/poses').glob('*.pkl')}
missing = metadata_ids - pose_files

print(f"Missing {len(missing)} poses:")
for vid in list(missing)[:10]:
    print(f"  {vid}")
EOF

# Re-download specific poses
!gsutil cp gs://iti123storage/features/poses_roi/MISSING_FILE.pkl ./data/poses/
```

---

## Storage Costs

### Upload Costs

| Item | Files | Size | Upload Time | Cost (one-time) |
|------|-------|------|-------------|-----------------|
| Poses | 19,423 | ~9 GB | 5-10 min | ~$0.02 |
| Metadata | 1 | ~2 MB | <1 min | <$0.01 |
| Clips | 19,778 | ~15 GB | 15-30 min | ~$0.03 |

**Total upload cost:** ~$0.05

### Storage Costs (Monthly)

| Item | Size | Cost/Month |
|------|------|------------|
| Poses | 9 GB | $0.18 |
| Metadata | 2 MB | <$0.01 |
| Clips | 15 GB | $0.30 |
| **Total** | **24 GB** | **$0.48** |

### Download Costs (Colab)

- Download from same region: **FREE** (GCS to Google Colab)
- Download to different region: ~$0.12/GB

---

## Summary

### Upload Scripts

| Script | Purpose | When to Use |
|--------|---------|-------------|
| `upload_poses_to_gcs.sh` | Full-featured upload | First-time upload, need verification |
| `quick_upload_gcs.sh` | Simple upload | Regular uploads, incremental updates |

### Download Script

| Script | Purpose | When to Use |
|--------|---------|-------------|
| `download_from_gcs_colab.sh` | Colab download | Training in Colab |

### Quick Commands

```bash
# Upload (local)
bash scripts/quick_upload_gcs.sh

# Download (Colab)
!bash scripts/download_from_gcs_colab.sh

# Verify upload
gsutil du -sh gs://iti123storage/features/poses_roi/

# Verify download
!find ./data/poses -name "*.pkl" | wc -l
```

---

**Status:** Ready to use
**Last updated:** 2026-02-03
