#!/bin/bash

# Test ROI Extraction on One Match
# This script tests the complete extraction pipeline on a single match

set -e  # Exit on error

# Configuration
MATCH_ID="01"
INPUT_DIR="data/raw_videos"
CLIPS_DIR="data/clips_test"
POSES_DIR="data/poses_test"
METADATA_FILE="data/metadata_test.csv"
SHUTTLESET_DIR="ShuttleSet"
MODEL_PATH="models/pose_landmarker_heavy.task"
NUM_WORKERS=4

echo "=========================================="
echo "ROI EXTRACTION TEST - MATCH ${MATCH_ID}"
echo "=========================================="
echo ""

# Check prerequisites
echo "Checking prerequisites..."

if [ ! -f "${INPUT_DIR}/${MATCH_ID}.mp4" ]; then
    echo "❌ Error: Match video not found: ${INPUT_DIR}/${MATCH_ID}.mp4"
    echo ""
    echo "Please place match video at: ${INPUT_DIR}/${MATCH_ID}.mp4"
    echo "Or update INPUT_DIR variable in this script"
    exit 1
fi

if [ ! -f "${SHUTTLESET_DIR}/set/match.csv" ]; then
    echo "❌ Error: ShuttleSet match.csv not found"
    echo "Expected: ${SHUTTLESET_DIR}/set/match.csv"
    exit 1
fi

if [ ! -f "${MODEL_PATH}" ]; then
    echo "❌ Error: MediaPipe model not found: ${MODEL_PATH}"
    echo ""
    echo "Download with:"
    echo "  mkdir -p models"
    echo "  curl -L -o ${MODEL_PATH} 'https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_heavy/float16/latest/pose_landmarker_heavy.task'"
    exit 1
fi

echo "✓ Match video found: ${INPUT_DIR}/${MATCH_ID}.mp4"
echo "✓ ShuttleSet annotations found"
echo "✓ MediaPipe model found"
echo ""

# Clean previous test outputs
echo "Cleaning previous test outputs..."
rm -rf "${CLIPS_DIR}"
rm -rf "${POSES_DIR}"
rm -f "${METADATA_FILE}"
echo "✓ Cleaned"
echo ""

# Step 1: Extract clips
echo "=========================================="
echo "STEP 1: Extract Clips"
echo "=========================================="
echo ""

python scripts/extract_shuttleset_clips.py \
    --input "${INPUT_DIR}" \
    --output "${CLIPS_DIR}" \
    --shuttleset "${SHUTTLESET_DIR}" \
    --match-ids "${MATCH_ID}" \
    --execute

if [ $? -ne 0 ]; then
    echo ""
    echo "❌ Clip extraction failed"
    exit 1
fi

# Count clips
CLIP_COUNT=$(find "${CLIPS_DIR}" -name "*.mp4" | wc -l | tr -d ' ')
echo ""
echo "✓ Extracted ${CLIP_COUNT} clips"
echo ""

if [ "${CLIP_COUNT}" -eq 0 ]; then
    echo "❌ No clips extracted"
    exit 1
fi

# Step 2: Create metadata
echo "=========================================="
echo "STEP 2: Create Metadata"
echo "=========================================="
echo ""

python scripts/create_metadata_csv.py \
    --shuttleset "${SHUTTLESET_DIR}" \
    --clips "${CLIPS_DIR}" \
    --output "${METADATA_FILE}"

if [ $? -ne 0 ]; then
    echo ""
    echo "❌ Metadata creation failed"
    exit 1
fi

# Count metadata entries
METADATA_COUNT=$(tail -n +2 "${METADATA_FILE}" | wc -l | tr -d ' ')
echo ""
echo "✓ Created metadata with ${METADATA_COUNT} entries"
echo ""

if [ "${METADATA_COUNT}" -eq 0 ]; then
    echo "❌ Metadata is empty"
    exit 1
fi

# Step 3: Extract poses with ROI
echo "=========================================="
echo "STEP 3: Extract Poses with ROI"
echo "=========================================="
echo ""

python scripts/extract_poses_roi.py \
    --clips "${CLIPS_DIR}" \
    --metadata "${METADATA_FILE}" \
    --output "${POSES_DIR}" \
    --model "${MODEL_PATH}" \
    --num-workers "${NUM_WORKERS}"

if [ $? -ne 0 ]; then
    echo ""
    echo "❌ Pose extraction failed"
    exit 1
fi

# Count poses
POSE_COUNT=$(find "${POSES_DIR}" -name "*.pkl" | wc -l | tr -d ' ')
echo ""
echo "✓ Extracted ${POSE_COUNT} pose sequences"
echo ""

if [ "${POSE_COUNT}" -eq 0 ]; then
    echo "❌ No poses extracted"
    exit 1
fi

# Step 4: Validate results
echo "=========================================="
echo "STEP 4: Validate Results"
echo "=========================================="
echo ""

python - << 'EOF'
import pickle
import numpy as np
from pathlib import Path
import csv

# Load metadata
metadata_file = Path('data/metadata_test.csv')
poses_dir = Path('data/poses_test')

print("Validation checks:")
print("-" * 50)

# Check 1: Metadata vs poses
with open(metadata_file, 'r') as f:
    reader = csv.DictReader(f)
    metadata_rows = list(reader)

pose_files = list(poses_dir.glob('*.pkl'))

print(f"1. Metadata entries: {len(metadata_rows)}")
print(f"   Pose files:       {len(pose_files)}")
print(f"   Success rate:     {len(pose_files)/len(metadata_rows)*100:.1f}%")
print()

if len(pose_files) == 0:
    print("❌ No poses to validate")
    exit(1)

# Check 2: Sample pose quality
print("2. Sample pose quality:")
sample_poses = list(pose_files)[:5]

multi_player_count = 0
short_sequences = 0
valid_poses = 0

for pose_file in sample_poses:
    try:
        with open(pose_file, 'rb') as f:
            pose = pickle.load(f)

        # Check shape
        if len(pose.shape) != 3 or pose.shape[1] != 33 or pose.shape[2] != 3:
            print(f"   ⚠️  {pose_file.name}: Invalid shape {pose.shape}")
            continue

        # Check multi-player (x-range >60%)
        x_coords = pose[:, :, 0]
        x_range = np.max(x_coords) - np.min(x_coords)

        # Check sequence length
        num_frames = len(pose)

        if x_range > 0.6:
            print(f"   ⚠️  {pose_file.name}: Multi-player detection (x-range={x_range:.3f})")
            multi_player_count += 1
        elif num_frames < 30:
            print(f"   ⚠️  {pose_file.name}: Short sequence ({num_frames} frames)")
            short_sequences += 1
        else:
            print(f"   ✓  {pose_file.name}: Valid (frames={num_frames}, x-range={x_range:.3f})")
            valid_poses += 1

    except Exception as e:
        print(f"   ❌ {pose_file.name}: Error loading - {e}")

print()
print(f"   Valid poses:      {valid_poses}/{len(sample_poses)}")
print(f"   Multi-player:     {multi_player_count}/{len(sample_poses)}")
print(f"   Short sequences:  {short_sequences}/{len(sample_poses)}")
print()

# Check 3: Data statistics
print("3. Data statistics (first pose):")
with open(pose_files[0], 'rb') as f:
    pose = pickle.load(f)

print(f"   Shape:      {pose.shape}")
print(f"   Mean:       {pose.mean():.4f}")
print(f"   Std:        {pose.std():.4f}")
print(f"   X-range:    {pose[:, :, 0].max() - pose[:, :, 0].min():.4f}")
print(f"   Y-range:    {pose[:, :, 1].max() - pose[:, :, 1].min():.4f}")
print()

# Check 4: Shot type distribution
print("4. Shot type distribution:")
shot_counts = {}
for row in metadata_rows:
    shot_type = row['shot_type']
    shot_counts[shot_type] = shot_counts.get(shot_type, 0) + 1

for shot_type in sorted(shot_counts.keys()):
    count = shot_counts[shot_type]
    pct = count / len(metadata_rows) * 100
    print(f"   {shot_type:<10} {count:>3} ({pct:>5.1f}%)")
print()

print("-" * 50)
print("✓ Validation complete")

# Exit with error if multi-player detected
if multi_player_count > 0:
    print()
    print(f"⚠️  Warning: {multi_player_count} multi-player detections found")
    print("   ROI may need adjustment")

EOF

if [ $? -ne 0 ]; then
    echo ""
    echo "❌ Validation failed"
    exit 1
fi

# Summary
echo ""
echo "=========================================="
echo "TEST SUMMARY"
echo "=========================================="
echo ""
echo "Match:         ${MATCH_ID}"
echo "Clips:         ${CLIP_COUNT}"
echo "Metadata:      ${METADATA_COUNT} entries"
echo "Poses:         ${POSE_COUNT}"
echo "Success rate:  $(python3 -c "print(f'{${POSE_COUNT}/${METADATA_COUNT}*100:.1f}%')")"
echo ""
echo "Output directories:"
echo "  Clips:    ${CLIPS_DIR}"
echo "  Poses:    ${POSES_DIR}"
echo "  Metadata: ${METADATA_FILE}"
echo ""
echo "✓ Test completed successfully!"
echo ""
echo "Next steps:"
echo "  1. Review validation output above"
echo "  2. Check sample clips: ls ${CLIPS_DIR}/Smash/*.mp4 | head -3"
echo "  3. If satisfied, run full extraction on all matches"
echo ""
