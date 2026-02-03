#!/bin/bash

# Complete Extraction Pipeline - Clips + Metadata + Poses
# Run this script overnight for unattended extraction
#
# Usage:
#   bash scripts/extract_full_pipeline.sh                    # Extract all matches (01-44)
#   bash scripts/extract_full_pipeline.sh 01 22              # Extract first half (matches 01-22)
#   bash scripts/extract_full_pipeline.sh 23 44              # Extract second half (matches 23-44)
#   bash scripts/extract_full_pipeline.sh 01 10              # Extract first 10 matches

set -e  # Exit on error

# Parse arguments
START_MATCH=${1:-01}
END_MATCH=${2:-44}

# Configuration
INPUT_DIR="data/raw_videos"
CLIPS_DIR="data/clips"
POSES_DIR="data/poses"
METADATA_FILE="data/metadata.csv"
SHUTTLESET_DIR="ShuttleSet"
MODEL_PATH="models/mediapipe/pose_landmarker_heavy.task"
NUM_WORKERS=8  # Adjust based on your CPU cores

# Logging
LOG_DIR="logs"
mkdir -p "${LOG_DIR}"
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
LOG_FILE="${LOG_DIR}/extraction_${START_MATCH}_to_${END_MATCH}_${TIMESTAMP}.log"

# Function to log with timestamp
log() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $1" | tee -a "${LOG_FILE}"
}

# Function to log and execute
run_cmd() {
    log "Executing: $1"
    eval "$1" 2>&1 | tee -a "${LOG_FILE}"
    if [ ${PIPESTATUS[0]} -ne 0 ]; then
        log "❌ Command failed: $1"
        return 1
    fi
    return 0
}

# Header
echo "================================================================================" | tee "${LOG_FILE}"
echo "COMPLETE EXTRACTION PIPELINE" | tee -a "${LOG_FILE}"
echo "================================================================================" | tee -a "${LOG_FILE}"
log "Match range: ${START_MATCH} to ${END_MATCH}"
log "Input:       ${INPUT_DIR}"
log "Output:      Clips: ${CLIPS_DIR} | Poses: ${POSES_DIR}"
log "Metadata:    ${METADATA_FILE}"
log "Workers:     ${NUM_WORKERS}"
log "Log file:    ${LOG_FILE}"
echo "================================================================================" | tee -a "${LOG_FILE}"
echo "" | tee -a "${LOG_FILE}"

# Check prerequisites
log "Checking prerequisites..."

if [ ! -d "${INPUT_DIR}" ]; then
    log "❌ Error: Input directory not found: ${INPUT_DIR}"
    exit 1
fi

if [ ! -f "${SHUTTLESET_DIR}/set/match.csv" ]; then
    log "❌ Error: ShuttleSet match.csv not found"
    exit 1
fi

if [ ! -f "${MODEL_PATH}" ]; then
    log "❌ Error: MediaPipe model not found: ${MODEL_PATH}"
    log "Download from: https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_heavy/float16/latest/pose_landmarker_heavy.task"
    exit 1
fi

# Count available videos
AVAILABLE_VIDEOS=$(ls ${INPUT_DIR}/*.mp4 2>/dev/null | wc -l | tr -d ' ')
log "✓ Found ${AVAILABLE_VIDEOS} video files in ${INPUT_DIR}"

# Generate match list
MATCH_LIST=""
for i in $(seq -f "%02g" ${START_MATCH} ${END_MATCH}); do
    if [ -f "${INPUT_DIR}/${i}.mp4" ]; then
        MATCH_LIST="${MATCH_LIST} ${i}"
    else
        log "⚠️  Warning: Video not found: ${INPUT_DIR}/${i}.mp4"
    fi
done

MATCH_COUNT=$(echo ${MATCH_LIST} | wc -w | tr -d ' ')

if [ ${MATCH_COUNT} -eq 0 ]; then
    log "❌ Error: No video files found for range ${START_MATCH}-${END_MATCH}"
    exit 1
fi

log "✓ Will process ${MATCH_COUNT} matches: ${MATCH_LIST}"
echo "" | tee -a "${LOG_FILE}"

# Start time
START_TIME=$(date +%s)

# ============================================================================
# STEP 1: EXTRACT CLIPS
# ============================================================================
log "================================================================================"
log "STEP 1: EXTRACT CLIPS (matches ${START_MATCH}-${END_MATCH})"
log "================================================================================"

STEP1_START=$(date +%s)

run_cmd "python scripts/extract_shuttleset_clips.py \
    --input ${INPUT_DIR} \
    --output ${CLIPS_DIR} \
    --shuttleset ${SHUTTLESET_DIR} \
    --match-ids ${MATCH_LIST} \
    --execute"

if [ $? -ne 0 ]; then
    log "❌ Clip extraction failed"
    exit 1
fi

STEP1_END=$(date +%s)
STEP1_DURATION=$((STEP1_END - STEP1_START))

# Count extracted clips
CLIP_COUNT=$(find "${CLIPS_DIR}" -name "*.mp4" 2>/dev/null | wc -l | tr -d ' ')
log "✓ Step 1 complete: ${CLIP_COUNT} clips extracted in ${STEP1_DURATION}s"
echo "" | tee -a "${LOG_FILE}"

if [ ${CLIP_COUNT} -eq 0 ]; then
    log "❌ Error: No clips were extracted"
    exit 1
fi

# ============================================================================
# STEP 2: CREATE METADATA
# ============================================================================
log "================================================================================"
log "STEP 2: CREATE METADATA"
log "================================================================================"

STEP2_START=$(date +%s)

run_cmd "python scripts/create_metadata_csv.py \
    --shuttleset ${SHUTTLESET_DIR} \
    --clips ${CLIPS_DIR} \
    --output ${METADATA_FILE}"

if [ $? -ne 0 ]; then
    log "❌ Metadata creation failed"
    exit 1
fi

STEP2_END=$(date +%s)
STEP2_DURATION=$((STEP2_END - STEP2_START))

# Count metadata entries
METADATA_COUNT=$(tail -n +2 "${METADATA_FILE}" 2>/dev/null | wc -l | tr -d ' ')
log "✓ Step 2 complete: ${METADATA_COUNT} entries in metadata.csv in ${STEP2_DURATION}s"
echo "" | tee -a "${LOG_FILE}"

if [ ${METADATA_COUNT} -eq 0 ]; then
    log "❌ Error: Metadata file is empty"
    exit 1
fi

# ============================================================================
# STEP 3: EXTRACT POSES WITH ROI
# ============================================================================
log "================================================================================"
log "STEP 3: EXTRACT POSES WITH ROI (${NUM_WORKERS} workers)"
log "================================================================================"
log "⏰ This step takes longest - estimated 8-12 hours for full dataset"
log "   Using ${NUM_WORKERS} workers. Progress will be displayed below."
echo "" | tee -a "${LOG_FILE}"

STEP3_START=$(date +%s)

run_cmd "python scripts/extract_poses_roi.py \
    --clips ${CLIPS_DIR} \
    --metadata ${METADATA_FILE} \
    --output ${POSES_DIR} \
    --model ${MODEL_PATH} \
    --num-workers ${NUM_WORKERS}"

if [ $? -ne 0 ]; then
    log "❌ Pose extraction failed"
    exit 1
fi

STEP3_END=$(date +%s)
STEP3_DURATION=$((STEP3_END - STEP3_START))

# Count extracted poses
POSE_COUNT=$(find "${POSES_DIR}" -name "*.pkl" 2>/dev/null | wc -l | tr -d ' ')
log "✓ Step 3 complete: ${POSE_COUNT} poses extracted in ${STEP3_DURATION}s"
echo "" | tee -a "${LOG_FILE}"

# ============================================================================
# STEP 4: VALIDATE RESULTS
# ============================================================================
log "================================================================================"
log "STEP 4: VALIDATE RESULTS"
log "================================================================================"

STEP4_START=$(date +%s)

run_cmd "python scripts/validate_roi_poses.py \
    --poses ${POSES_DIR} \
    --metadata ${METADATA_FILE}"

STEP4_END=$(date +%s)
STEP4_DURATION=$((STEP4_END - STEP4_START))

log "✓ Step 4 complete: Validation done in ${STEP4_DURATION}s"
echo "" | tee -a "${LOG_FILE}"

# ============================================================================
# FINAL SUMMARY
# ============================================================================
END_TIME=$(date +%s)
TOTAL_DURATION=$((END_TIME - START_TIME))
HOURS=$((TOTAL_DURATION / 3600))
MINUTES=$(((TOTAL_DURATION % 3600) / 60))
SECONDS=$((TOTAL_DURATION % 60))

echo "" | tee -a "${LOG_FILE}"
log "================================================================================"
log "EXTRACTION COMPLETE!"
log "================================================================================"
log ""
log "Match range:       ${START_MATCH} to ${END_MATCH}"
log "Matches processed: ${MATCH_COUNT}"
log ""
log "Results:"
log "  Clips extracted: ${CLIP_COUNT}"
log "  Metadata entries: ${METADATA_COUNT}"
log "  Poses extracted:  ${POSE_COUNT}"
log "  Success rate:     $(python3 -c "print(f'{${POSE_COUNT}/${METADATA_COUNT}*100:.1f}%')" 2>/dev/null || echo 'N/A')"
log ""
log "Duration breakdown:"
log "  Step 1 (Clips):    $(printf '%02d:%02d:%02d' $((STEP1_DURATION/3600)) $((STEP1_DURATION%3600/60)) $((STEP1_DURATION%60)))"
log "  Step 2 (Metadata): $(printf '%02d:%02d:%02d' $((STEP2_DURATION/3600)) $((STEP2_DURATION%3600/60)) $((STEP2_DURATION%60)))"
log "  Step 3 (Poses):    $(printf '%02d:%02d:%02d' $((STEP3_DURATION/3600)) $((STEP3_DURATION%3600/60)) $((STEP3_DURATION%60)))"
log "  Step 4 (Validate): $(printf '%02d:%02d:%02d' $((STEP4_DURATION/3600)) $((STEP4_DURATION%3600/60)) $((STEP4_DURATION%60)))"
log "  Total:             $(printf '%02d:%02d:%02d' ${HOURS} ${MINUTES} ${SECONDS})"
log ""
log "Output directories:"
log "  Clips:    ${CLIPS_DIR}"
log "  Poses:    ${POSES_DIR}"
log "  Metadata: ${METADATA_FILE}"
log "  Log:      ${LOG_FILE}"
log ""
log "Next steps:"
log "  1. Review validation output above"
log "  2. Train model: python scripts/train_models_fixed.py --model stgcn --epochs 50"

if [ "${END_MATCH}" -lt 44 ]; then
    NEXT_START=$((END_MATCH + 1))
    log "  3. Extract second half: bash scripts/extract_full_pipeline.sh ${NEXT_START} 44"
fi

log ""
log "================================================================================"
echo "" | tee -a "${LOG_FILE}"

exit 0
