---
created: 2026-02-01T00:00
title: Organize GCS videos into stroke type folders
area: data-preparation
files:
  - scripts/organize_gcs_videos.sh
  - scripts/organize_videos.py
  - docs/GCS_DATASET_ANALYSIS.md
---

## Problem

The GCS bucket (gs://iti123storage/videos/clips/) contains 11,055 video files in a flat structure. All videos follow the naming pattern `{match}_set{N}_rally{N}_ball{N}_{StrokeType}.mp4` but are not organized into subdirectories by stroke type.

Current state:
- All 11,055 videos in root clips/ folder
- Distribution: 4,641 Smash, 3,179 Drop, 2,662 Clear, 573 Lift
- Filenames already labeled with stroke type (_Clear.mp4, _Smash.mp4, _Drop.mp4, _Lift.mp4)

Required before Phase 2 pose extraction:
- Videos must be organized into clear/, smash/, drop/, lift/ subdirectories
- Pose extraction scripts expect organized structure
- Metadata creation relies on directory structure

## Solution

Scripts are ready and tested:

1. **organize_gcs_videos.sh** (recommended for Colab):
   - Bash script using gsutil for direct GCS operations
   - Pattern detection: 100% accuracy on actual filenames
   - Dry-run mode for preview
   - Estimated time: ~20-25 minutes for 11,055 videos

2. **organize_videos.py** (Python alternative):
   - Supports both local and GCS paths
   - Same pattern detection
   - Copy or move modes

Execution steps:
```bash
# Preview (dry-run)
bash scripts/organize_gcs_videos.sh gs://iti123storage/videos/clips

# Execute
bash scripts/organize_gcs_videos.sh gs://iti123storage/videos/clips --execute

# Verify
gsutil ls gs://iti123storage/videos/clips/clear/ | wc -l   # Should be 2,662
gsutil ls gs://iti123storage/videos/clips/smash/ | wc -l   # Should be 4,641
gsutil ls gs://iti123storage/videos/clips/drop/ | wc -l    # Should be 3,179
gsutil ls gs://iti123storage/videos/clips/lift/ | wc -l    # Should be 573
```

After organization, proceed with Phase 2 pose extraction workflow.

See: docs/GCS_DATASET_ANALYSIS.md for complete analysis and next steps.
