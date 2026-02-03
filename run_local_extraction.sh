#!/bin/bash
#
# Local Pose Extraction Runner
# Optimized for Mac with 10 cores
#

python3 scripts/extract_poses_parallel.py \
    --video-dir data/clips/ \
    --output-dir data/processed/poses/ \
    --target-fps 30 \
    --min-confidence 0.5 \
    --num-workers 8 \
    2>&1 | tee extraction_log.txt

echo ""
echo "Extraction log saved to extraction_log.txt"
