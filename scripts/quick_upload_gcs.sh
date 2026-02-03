#!/bin/bash

# Quick Upload to GCS - Simple wrapper for common upload scenarios
# Usage: bash scripts/quick_upload_gcs.sh [poses|metadata|clips|all]

set -e

# Configuration
BUCKET="gs://iti123storage"

# Colors
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

log_info() { echo -e "${GREEN}[INFO]${NC} $1"; }
log_warn() { echo -e "${YELLOW}[WARN]${NC} $1"; }
log_section() { echo -e "${BLUE}[SECTION]${NC} $1"; }

# Parse arguments
UPLOAD_TYPE=${1:-all}

# Header
echo "================================================================================"
echo "QUICK UPLOAD TO GCS"
echo "================================================================================"
echo "Upload type: $UPLOAD_TYPE"
echo "Bucket:      $BUCKET"
echo "================================================================================"
echo ""

# Check gsutil
if ! command -v gsutil &> /dev/null; then
    echo "Error: gsutil not found. Install with:"
    echo "  brew install --cask google-cloud-sdk"
    exit 1
fi

# Check authentication
if ! gsutil ls "$BUCKET" >/dev/null 2>&1; then
    echo "Error: Cannot access bucket. Authenticate with:"
    echo "  gcloud auth login"
    exit 1
fi

# Function to upload with progress
upload_poses() {
    log_section "UPLOADING POSES"

    if [ ! -d "data/poses" ]; then
        log_warn "Poses directory not found: data/poses"
        return 1
    fi

    local count=$(find data/poses -name "*.pkl" | wc -l | tr -d ' ')
    log_info "Found $count pose files"

    log_info "Uploading to gs://iti123storage/features/poses_roi/"
    gsutil -m rsync -r data/poses/ gs://iti123storage/features/poses_roi/

    log_info "✓ Poses uploaded successfully"
    echo ""
}

upload_metadata() {
    log_section "UPLOADING METADATA"

    if [ ! -f "data/metadata.csv" ]; then
        log_warn "Metadata file not found: data/metadata.csv"
        return 1
    fi

    local size=$(ls -lh data/metadata.csv | awk '{print $5}')
    log_info "File size: $size"

    log_info "Uploading to gs://iti123storage/data/metadata_roi.csv"
    gsutil cp data/metadata.csv gs://iti123storage/data/metadata_roi.csv

    log_info "✓ Metadata uploaded successfully"
    echo ""
}

upload_clips() {
    log_section "UPLOADING CLIPS"

    if [ ! -d "data/clips" ]; then
        log_warn "Clips directory not found: data/clips"
        return 1
    fi

    local count=$(find data/clips -name "*.mp4" | wc -l | tr -d ' ')
    log_info "Found $count clip files"
    log_warn "This may take a while (large files)"

    log_info "Uploading to gs://iti123storage/data/clips_roi/"
    gsutil -m rsync -r data/clips/ gs://iti123storage/data/clips_roi/

    log_info "✓ Clips uploaded successfully"
    echo ""
}

# Execute based on upload type
case $UPLOAD_TYPE in
    poses)
        upload_poses
        ;;
    metadata)
        upload_metadata
        ;;
    clips)
        upload_clips
        ;;
    all)
        upload_poses
        upload_metadata
        log_warn "Skipping clips (too large). To upload clips, run:"
        echo "  bash scripts/quick_upload_gcs.sh clips"
        echo ""
        ;;
    *)
        echo "Error: Invalid upload type: $UPLOAD_TYPE"
        echo ""
        echo "Usage: bash scripts/quick_upload_gcs.sh [poses|metadata|clips|all]"
        echo ""
        echo "Examples:"
        echo "  bash scripts/quick_upload_gcs.sh         # Upload poses + metadata"
        echo "  bash scripts/quick_upload_gcs.sh poses   # Upload only poses"
        echo "  bash scripts/quick_upload_gcs.sh clips   # Upload only clips"
        exit 1
        ;;
esac

# Summary
echo "================================================================================"
echo "UPLOAD COMPLETE"
echo "================================================================================"
echo ""
echo "To download in Colab:"
echo ""
echo "# Download poses"
echo "!gsutil -m rsync -r gs://iti123storage/features/poses_roi/ ./data/poses/"
echo ""
echo "# Download metadata"
echo "!gsutil cp gs://iti123storage/data/metadata_roi.csv ./data/metadata.csv"
echo ""
echo "================================================================================"

exit 0
