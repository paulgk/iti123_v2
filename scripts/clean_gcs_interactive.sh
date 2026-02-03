#!/bin/bash

# Interactive GCS Cleanup - Choose what to delete
# This script presents options and confirms before deletion

set -e

# Configuration
BUCKET="gs://iti123storage"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
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

# Function to get storage stats
get_stats() {
    local path=$1
    if gsutil ls "$path" >/dev/null 2>&1; then
        local files=$(gsutil ls -r "$path" 2>/dev/null | grep -v ':$' | grep -v '^$' | wc -l | tr -d ' ')
        local size=$(gsutil du -s "$path" 2>/dev/null | awk '{printf "%.2f GB", $1/1024/1024/1024}')
        echo "$files files, $size"
    else
        echo "Not found"
    fi
}

# Function to ask yes/no question
ask_yes_no() {
    local question=$1
    local response

    while true; do
        read -p "$question (y/n): " response
        case $response in
            [Yy]* ) return 0;;
            [Nn]* ) return 1;;
            * ) echo "Please answer y or n.";;
        esac
    done
}

# Function to delete with confirmation
delete_with_confirm() {
    local path=$1
    local description=$2

    echo ""
    echo "================================================================================"
    log_info "$description"
    echo "Path: $path"
    local stats=$(get_stats "$path")
    echo "Size: $stats"

    if [ "$stats" = "Not found" ]; then
        log_info "Already clean, skipping"
        return
    fi

    if ask_yes_no "Delete this?"; then
        log_warn "Deleting $path..."
        gsutil -m rm -r "$path" 2>/dev/null && log_info "✓ Deleted successfully" || log_error "Failed to delete"
    else
        log_info "Skipped"
    fi
}

# Header
clear
echo "================================================================================"
echo "INTERACTIVE GCS STORAGE CLEANUP"
echo "================================================================================"
echo "Bucket: $BUCKET"
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

# Show current usage
log_info "Current bucket usage:"
gsutil du -sh "$BUCKET"
echo ""

log_warn "This script will ask you to confirm each deletion."
echo "Press Ctrl+C at any time to exit."
echo ""
read -p "Press Enter to continue..."

# ============================================================================
# CLEANUP OPTIONS
# ============================================================================

# 1. Old pose extractions
delete_with_confirm "$BUCKET/features/poses_old/" "Old pose extractions (before ROI)"
delete_with_confirm "$BUCKET/features/poses_backup/" "Backup pose files"
delete_with_confirm "$BUCKET/features/poses_v1/" "Version 1 pose files"

# 2. Old metadata
delete_with_confirm "$BUCKET/data/metadata_old.csv" "Old metadata file"
delete_with_confirm "$BUCKET/data/metadata_v1.csv" "Version 1 metadata"
delete_with_confirm "$BUCKET/data/metadata_backup.csv" "Backup metadata"

# 3. Old clips
delete_with_confirm "$BUCKET/data/clips_old/" "Old clip extractions"
delete_with_confirm "$BUCKET/data/clips_backup/" "Backup clips"
delete_with_confirm "$BUCKET/data/clips_v1/" "Version 1 clips"

# 4. Test files
delete_with_confirm "$BUCKET/data/poses_test/" "Test pose files"
delete_with_confirm "$BUCKET/data/clips_test/" "Test clip files"
delete_with_confirm "$BUCKET/test/" "Test directory"

# 5. Temporary files
delete_with_confirm "$BUCKET/tmp/" "Temporary files"
delete_with_confirm "$BUCKET/debug/" "Debug files"

# 6. Ambiguous shot types (removed from mapping)
echo ""
echo "================================================================================"
echo "AMBIGUOUS SHOT TYPES (removed from Phase 1.5 mapping)"
echo "================================================================================"
echo ""
echo "The following shot types were removed from the clean mapping:"
echo "  - Slice_Drop (different wrist technique)"
echo "  - Push (net shots, not drives)"
echo "  - Rear_Drive (different positioning)"
echo "  - Defensive_Drive (reactive posture)"
echo "  - Short_Drive (different power level)"
echo ""

if ask_yes_no "Delete all ambiguous shot clips and poses?"; then
    delete_with_confirm "$BUCKET/data/clips/Slice_Drop/" "Slice_Drop clips"
    delete_with_confirm "$BUCKET/data/clips/Push/" "Push clips"
    delete_with_confirm "$BUCKET/data/clips/Rear_Drive/" "Rear_Drive clips"
    delete_with_confirm "$BUCKET/data/clips/Defensive_Drive/" "Defensive_Drive clips"
    delete_with_confirm "$BUCKET/data/clips/Short_Drive/" "Short_Drive clips"

    log_info "Deleting corresponding pose files..."
    for shot in Slice_Drop Push Rear_Drive Defensive_Drive Short_Drive; do
        gsutil -m rm "$BUCKET/features/poses/*_${shot}.pkl" 2>/dev/null && \
            log_info "✓ Deleted ${shot} poses" || \
            log_info "No ${shot} poses found"
    done
else
    log_info "Skipped ambiguous shot types"
fi

# 7. Old training outputs
delete_with_confirm "$BUCKET/outputs/old/" "Old training outputs"
delete_with_confirm "$BUCKET/outputs/backup/" "Backup training outputs"
delete_with_confirm "$BUCKET/outputs/v1/" "Version 1 training outputs"
delete_with_confirm "$BUCKET/models/old/" "Old model checkpoints"

# 8. MLflow (if not using)
echo ""
echo "================================================================================"
echo "MLflow ARTIFACTS"
echo "================================================================================"
echo ""
echo "MLflow was used in earlier phases but may no longer be needed."
echo ""

delete_with_confirm "$BUCKET/mlflow/" "MLflow artifacts"
delete_with_confirm "$BUCKET/mlruns/" "MLflow runs"

# ============================================================================
# FINAL SUMMARY
# ============================================================================

echo ""
echo "================================================================================"
echo "CLEANUP COMPLETE"
echo "================================================================================"
echo ""

log_info "New bucket usage:"
gsutil du -sh "$BUCKET"
echo ""

log_info "Cleanup summary saved to: logs/gcs_cleanup_$(date +%Y%m%d_%H%M%S).log"
echo ""
echo "To see what's left in the bucket:"
echo "  bash scripts/list_gcs_contents.sh"
echo ""
echo "================================================================================"

exit 0
