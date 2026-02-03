#!/bin/bash

# Clean GCS Storage - Remove old/obsolete files from Google Cloud Storage
# This script helps clean up previous extraction attempts and obsolete data

set -e

# Configuration
BUCKET="gs://iti123storage"
DRY_RUN=true  # Set to false to actually delete files

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
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

# Function to get file count and size
get_stats() {
    local path=$1
    echo "$(gsutil ls -l "$path" 2>/dev/null | grep -v TOTAL | wc -l | tr -d ' ') files, $(gsutil du -s "$path" 2>/dev/null | awk '{print $1/1024/1024/1024}' | xargs printf "%.2f") GB"
}

# Function to delete or simulate deletion
delete_path() {
    local path=$1
    local description=$2

    log_info "Analyzing: $description"
    log_info "Path: $path"

    # Check if path exists
    if gsutil ls "$path" >/dev/null 2>&1; then
        local stats=$(get_stats "$path")
        log_info "Found: $stats"

        if [ "$DRY_RUN" = true ]; then
            log_warn "[DRY-RUN] Would delete: $path"
        else
            log_warn "Deleting: $path"
            gsutil -m rm -r "$path" 2>/dev/null || log_error "Failed to delete $path"
            log_info "✓ Deleted: $path"
        fi
    else
        log_info "Not found (already clean): $path"
    fi

    echo ""
}

# Header
echo "================================================================================"
echo "GCS STORAGE CLEANUP"
echo "================================================================================"
echo "Bucket:  $BUCKET"
echo "Mode:    $([ "$DRY_RUN" = true ] && echo 'DRY-RUN (preview only)' || echo 'EXECUTE (will delete)')"
echo "================================================================================"
echo ""

# Check if gsutil is installed
if ! command -v gsutil &> /dev/null; then
    log_error "gsutil not found. Please install Google Cloud SDK first:"
    echo "  https://cloud.google.com/sdk/docs/install"
    exit 1
fi

# Check if authenticated
if ! gsutil ls "$BUCKET" >/dev/null 2>&1; then
    log_error "Cannot access bucket. Please authenticate:"
    echo "  gcloud auth login"
    exit 1
fi

log_info "Connected to bucket successfully"
echo ""

# Show current bucket usage
log_info "Current bucket usage:"
gsutil du -s "$BUCKET"
echo ""

# ============================================================================
# SECTION 1: OLD POSE EXTRACTIONS (before ROI)
# ============================================================================
echo "================================================================================"
echo "SECTION 1: OLD POSE EXTRACTIONS (before ROI implementation)"
echo "================================================================================"
echo ""

delete_path "$BUCKET/features/poses_old/**" "Old pose extractions (before ROI)"
delete_path "$BUCKET/features/poses_backup/**" "Backup pose files"
delete_path "$BUCKET/features/poses_v1/**" "Version 1 pose files"

# ============================================================================
# SECTION 2: OLD METADATA FILES
# ============================================================================
echo "================================================================================"
echo "SECTION 2: OLD METADATA FILES"
echo "================================================================================"
echo ""

delete_path "$BUCKET/data/metadata_old.csv" "Old metadata without player positions"
delete_path "$BUCKET/data/metadata_v1.csv" "Version 1 metadata"
delete_path "$BUCKET/data/metadata_backup.csv" "Backup metadata"

# ============================================================================
# SECTION 3: OLD CLIPS (if re-extracted with different mapping)
# ============================================================================
echo "================================================================================"
echo "SECTION 3: OLD CLIPS (if re-extracted)"
echo "================================================================================"
echo ""

delete_path "$BUCKET/data/clips_old/**" "Old clip extractions"
delete_path "$BUCKET/data/clips_backup/**" "Backup clips"
delete_path "$BUCKET/data/clips_v1/**" "Version 1 clips"

# ============================================================================
# SECTION 4: OLD TRAINING OUTPUTS
# ============================================================================
echo "================================================================================"
echo "SECTION 4: OLD TRAINING OUTPUTS"
echo "================================================================================"
echo ""

delete_path "$BUCKET/outputs/old/**" "Old training outputs"
delete_path "$BUCKET/outputs/backup/**" "Backup training outputs"
delete_path "$BUCKET/outputs/v1/**" "Version 1 training outputs"
delete_path "$BUCKET/models/old/**" "Old model checkpoints"

# ============================================================================
# SECTION 5: TEMPORARY/DEBUG FILES
# ============================================================================
echo "================================================================================"
echo "SECTION 5: TEMPORARY AND DEBUG FILES"
echo "================================================================================"
echo ""

delete_path "$BUCKET/tmp/**" "Temporary files"
delete_path "$BUCKET/debug/**" "Debug files"
delete_path "$BUCKET/test/**" "Test files"
delete_path "$BUCKET/data/poses_test/**" "Test pose extractions"
delete_path "$BUCKET/data/clips_test/**" "Test clip extractions"

# ============================================================================
# SECTION 6: AMBIGUOUS SHOT CLIPS (removed from mapping)
# ============================================================================
echo "================================================================================"
echo "SECTION 6: AMBIGUOUS SHOT TYPES (removed from clean mapping)"
echo "================================================================================"
echo ""

delete_path "$BUCKET/data/clips/Slice_Drop/**" "Slice_Drop clips (removed from mapping)"
delete_path "$BUCKET/data/clips/Push/**" "Push clips (removed from mapping)"
delete_path "$BUCKET/data/clips/Rear_Drive/**" "Rear_Drive clips (removed from mapping)"
delete_path "$BUCKET/data/clips/Defensive_Drive/**" "Defensive_Drive clips (removed from mapping)"
delete_path "$BUCKET/data/clips/Short_Drive/**" "Short_Drive clips (removed from mapping)"

delete_path "$BUCKET/features/poses/*_Slice_Drop.pkl" "Slice_Drop poses"
delete_path "$BUCKET/features/poses/*_Push.pkl" "Push poses"
delete_path "$BUCKET/features/poses/*_Rear_Drive.pkl" "Rear_Drive poses"
delete_path "$BUCKET/features/poses/*_Defensive_Drive.pkl" "Defensive_Drive poses"

# ============================================================================
# SECTION 7: MLflow ARTIFACTS (if migrated)
# ============================================================================
echo "================================================================================"
echo "SECTION 7: MLflow ARTIFACTS (if no longer using MLflow)"
echo "================================================================================"
echo ""

delete_path "$BUCKET/mlflow/**" "MLflow artifacts"
delete_path "$BUCKET/mlruns/**" "MLflow runs"

# ============================================================================
# FINAL SUMMARY
# ============================================================================
echo "================================================================================"
echo "CLEANUP SUMMARY"
echo "================================================================================"
echo ""

if [ "$DRY_RUN" = true ]; then
    log_warn "DRY-RUN MODE: No files were actually deleted"
    echo ""
    echo "To execute the cleanup, edit this script and set:"
    echo "  DRY_RUN=false"
    echo ""
    echo "Then run again:"
    echo "  bash scripts/clean_gcs_storage.sh"
else
    log_info "Cleanup completed!"
    echo ""
    log_info "New bucket usage:"
    gsutil du -s "$BUCKET"
fi

echo ""
echo "================================================================================"

exit 0
