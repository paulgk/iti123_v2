#!/bin/bash

# Upload Poses to GCS - Upload Phase 1.5 ROI-extracted poses to Google Cloud Storage
# This script uploads poses, metadata, and optionally clips to GCS

set -e

# Configuration
BUCKET="gs://iti123storage"
LOCAL_POSES_DIR="data/poses"
LOCAL_METADATA="data/metadata.csv"
LOCAL_CLIPS_DIR="data/clips"

# GCS paths
GCS_POSES_DIR="${BUCKET}/features/poses_roi"
GCS_METADATA="${BUCKET}/data/metadata_roi.csv"
GCS_CLIPS_DIR="${BUCKET}/data/clips_roi"

# Upload options
UPLOAD_POSES=true
UPLOAD_METADATA=true
UPLOAD_CLIPS=false  # Set to true if you want to upload clips (large)
DRY_RUN=false       # Set to true to preview without uploading

# Colors for output
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
RED='\033[0;31m'
NC='\033[0m' # No Color

# Function to print colored output
log_info() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

log_warn() {
    echo -e "${YELLOW}[WARN]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

log_section() {
    echo -e "${BLUE}[SECTION]${NC} $1"
}

# Function to get local directory stats
get_local_stats() {
    local dir=$1
    if [ -d "$dir" ]; then
        local file_count=$(find "$dir" -type f | wc -l | tr -d ' ')
        local size=$(du -sh "$dir" | awk '{print $1}')
        echo "$file_count files, $size"
    else
        echo "Not found"
    fi
}

# Function to upload with progress
upload_directory() {
    local local_path=$1
    local gcs_path=$2
    local description=$3

    echo ""
    log_section "$description"
    echo "Local:  $local_path"
    echo "GCS:    $gcs_path"

    local stats=$(get_local_stats "$local_path")
    echo "Size:   $stats"

    if [ "$DRY_RUN" = true ]; then
        log_warn "[DRY-RUN] Would upload: $local_path -> $gcs_path"
        return
    fi

    log_info "Starting upload..."
    local start_time=$(date +%s)

    # Use gsutil rsync for efficient upload (only uploads new/changed files)
    if gsutil -m rsync -r -d "$local_path" "$gcs_path" 2>&1; then
        local end_time=$(date +%s)
        local duration=$((end_time - start_time))
        log_info "✓ Upload complete in ${duration}s"

        # Verify upload
        local gcs_count=$(gsutil ls -r "$gcs_path" 2>/dev/null | grep -v ':$' | grep -v '^$' | wc -l | tr -d ' ')
        log_info "✓ Verified: $gcs_count files in GCS"
    else
        log_error "Upload failed"
        return 1
    fi
}

# Function to upload single file
upload_file() {
    local local_path=$1
    local gcs_path=$2
    local description=$3

    echo ""
    log_section "$description"
    echo "Local:  $local_path"
    echo "GCS:    $gcs_path"

    if [ -f "$local_path" ]; then
        local size=$(ls -lh "$local_path" | awk '{print $5}')
        echo "Size:   $size"
    else
        log_error "File not found: $local_path"
        return 1
    fi

    if [ "$DRY_RUN" = true ]; then
        log_warn "[DRY-RUN] Would upload: $local_path -> $gcs_path"
        return
    fi

    log_info "Uploading..."
    if gsutil cp "$local_path" "$gcs_path" 2>&1; then
        log_info "✓ Upload complete"
    else
        log_error "Upload failed"
        return 1
    fi
}

# Header
clear
echo "================================================================================"
echo "UPLOAD POSES TO GCS"
echo "================================================================================"
echo "Bucket:   $BUCKET"
echo "Mode:     $([ "$DRY_RUN" = true ] && echo 'DRY-RUN (preview only)' || echo 'EXECUTE (uploading)')"
echo "================================================================================"
echo ""

# Check prerequisites
log_info "Checking prerequisites..."

# Check gsutil
if ! command -v gsutil &> /dev/null; then
    log_error "gsutil not found. Please install Google Cloud SDK:"
    echo "  https://cloud.google.com/sdk/docs/install"
    exit 1
fi
log_info "✓ gsutil found"

# Check authentication
if ! gsutil ls "$BUCKET" >/dev/null 2>&1; then
    log_error "Cannot access bucket. Please authenticate:"
    echo "  gcloud auth login"
    exit 1
fi
log_info "✓ Authenticated to GCS"

# Check local files
if [ "$UPLOAD_POSES" = true ]; then
    if [ ! -d "$LOCAL_POSES_DIR" ]; then
        log_error "Poses directory not found: $LOCAL_POSES_DIR"
        exit 1
    fi
    local pose_count=$(find "$LOCAL_POSES_DIR" -name "*.pkl" | wc -l | tr -d ' ')
    if [ "$pose_count" -eq 0 ]; then
        log_error "No pose files found in $LOCAL_POSES_DIR"
        exit 1
    fi
    log_info "✓ Found $pose_count pose files"
fi

if [ "$UPLOAD_METADATA" = true ]; then
    if [ ! -f "$LOCAL_METADATA" ]; then
        log_error "Metadata file not found: $LOCAL_METADATA"
        exit 1
    fi
    log_info "✓ Found metadata file"
fi

if [ "$UPLOAD_CLIPS" = true ]; then
    if [ ! -d "$LOCAL_CLIPS_DIR" ]; then
        log_warn "Clips directory not found: $LOCAL_CLIPS_DIR"
        UPLOAD_CLIPS=false
    else
        log_info "✓ Found clips directory"
    fi
fi

echo ""

# Show current bucket usage
log_info "Current GCS bucket usage:"
gsutil du -sh "$BUCKET" 2>/dev/null || echo "Unable to get bucket size"
echo ""

# Confirm before upload
if [ "$DRY_RUN" = false ]; then
    echo "================================================================================"
    echo "UPLOAD PLAN"
    echo "================================================================================"
    [ "$UPLOAD_POSES" = true ] && echo "✓ Poses:    $LOCAL_POSES_DIR -> $GCS_POSES_DIR"
    [ "$UPLOAD_METADATA" = true ] && echo "✓ Metadata: $LOCAL_METADATA -> $GCS_METADATA"
    [ "$UPLOAD_CLIPS" = true ] && echo "✓ Clips:    $LOCAL_CLIPS_DIR -> $GCS_CLIPS_DIR"
    echo "================================================================================"
    echo ""

    read -p "Proceed with upload? (y/n): " -n 1 -r
    echo ""
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        log_warn "Upload cancelled"
        exit 0
    fi
    echo ""
fi

# ============================================================================
# UPLOAD 1: POSES
# ============================================================================
if [ "$UPLOAD_POSES" = true ]; then
    upload_directory "$LOCAL_POSES_DIR" "$GCS_POSES_DIR" "Uploading Poses (ROI-extracted)"
fi

# ============================================================================
# UPLOAD 2: METADATA
# ============================================================================
if [ "$UPLOAD_METADATA" = true ]; then
    upload_file "$LOCAL_METADATA" "$GCS_METADATA" "Uploading Metadata (with player positions)"
fi

# ============================================================================
# UPLOAD 3: CLIPS (Optional)
# ============================================================================
if [ "$UPLOAD_CLIPS" = true ]; then
    log_warn "Uploading clips (this may take a while - large files)"
    upload_directory "$LOCAL_CLIPS_DIR" "$GCS_CLIPS_DIR" "Uploading Clips (clean 5-class mapping)"
fi

# ============================================================================
# VERIFICATION
# ============================================================================
echo ""
echo "================================================================================"
echo "VERIFICATION"
echo "================================================================================"

if [ "$DRY_RUN" = false ]; then
    echo ""
    log_info "Verifying uploads..."

    if [ "$UPLOAD_POSES" = true ]; then
        echo ""
        log_section "Poses Verification"
        local gcs_pose_count=$(gsutil ls -r "$GCS_POSES_DIR" 2>/dev/null | grep '\.pkl$' | wc -l | tr -d ' ')
        local local_pose_count=$(find "$LOCAL_POSES_DIR" -name "*.pkl" | wc -l | tr -d ' ')
        echo "Local:  $local_pose_count files"
        echo "GCS:    $gcs_pose_count files"

        if [ "$gcs_pose_count" -eq "$local_pose_count" ]; then
            log_info "✓ Pose count matches"
        else
            log_warn "⚠️  Pose count mismatch"
        fi
    fi

    if [ "$UPLOAD_METADATA" = true ]; then
        echo ""
        log_section "Metadata Verification"
        if gsutil ls "$GCS_METADATA" >/dev/null 2>&1; then
            local gcs_size=$(gsutil ls -l "$GCS_METADATA" | awk '{print $1}')
            local local_size=$(wc -c < "$LOCAL_METADATA")
            echo "Local:  $local_size bytes"
            echo "GCS:    $gcs_size bytes"

            if [ "$gcs_size" -eq "$local_size" ]; then
                log_info "✓ Metadata size matches"
            else
                log_warn "⚠️  Metadata size mismatch"
            fi
        else
            log_error "✗ Metadata not found in GCS"
        fi
    fi

    echo ""
    log_info "New GCS bucket usage:"
    gsutil du -sh "$BUCKET" 2>/dev/null || echo "Unable to get bucket size"
fi

# ============================================================================
# SUMMARY
# ============================================================================
echo ""
echo "================================================================================"
echo "UPLOAD COMPLETE"
echo "================================================================================"
echo ""

if [ "$DRY_RUN" = true ]; then
    log_warn "DRY-RUN MODE: No files were uploaded"
    echo ""
    echo "To execute the upload, edit this script and set:"
    echo "  DRY_RUN=false"
    echo ""
    echo "Then run again:"
    echo "  bash scripts/upload_poses_to_gcs.sh"
else
    log_info "Upload successful!"
    echo ""
    echo "GCS Locations:"
    [ "$UPLOAD_POSES" = true ] && echo "  Poses:    $GCS_POSES_DIR"
    [ "$UPLOAD_METADATA" = true ] && echo "  Metadata: $GCS_METADATA"
    [ "$UPLOAD_CLIPS" = true ] && echo "  Clips:    $GCS_CLIPS_DIR"
    echo ""
    echo "To download in Colab:"
    echo "  # Download poses"
    echo "  !gsutil -m rsync -r $GCS_POSES_DIR ./data/poses/"
    echo ""
    echo "  # Download metadata"
    echo "  !gsutil cp $GCS_METADATA ./data/metadata.csv"
fi

echo ""
echo "================================================================================"

exit 0
