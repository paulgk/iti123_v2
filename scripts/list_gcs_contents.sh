#!/bin/bash

# List GCS Storage Contents - Analyze what's in your GCS bucket
# This script helps you understand current storage usage before cleanup

set -e

# Configuration
BUCKET="gs://iti123storage"

# Colors for output
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Function to print colored output
log_info() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

log_section() {
    echo -e "${BLUE}[SECTION]${NC} $1"
}

# Function to analyze directory
analyze_dir() {
    local path=$1
    local name=$2

    echo ""
    log_section "$name"
    echo "Path: $path"

    # Check if path exists
    if gsutil ls "$path" >/dev/null 2>&1; then
        # Count files
        local file_count=$(gsutil ls -r "$path" 2>/dev/null | grep -v ':$' | grep -v '^$' | wc -l | tr -d ' ')

        # Get size
        local size_bytes=$(gsutil du -s "$path" 2>/dev/null | awk '{print $1}')
        local size_gb=$(echo "scale=2; $size_bytes / 1024 / 1024 / 1024" | bc)

        echo "Files: $file_count"
        echo "Size:  ${size_gb} GB (${size_bytes} bytes)"

        # Show sample files (first 5)
        echo "Sample files:"
        gsutil ls "$path" 2>/dev/null | head -5 | sed 's/^/  /'

        local remaining=$((file_count - 5))
        if [ $remaining -gt 0 ]; then
            echo "  ... and $remaining more files"
        fi
    else
        echo "✓ Not found (clean)"
    fi
}

# Header
echo "================================================================================"
echo "GCS STORAGE CONTENTS ANALYSIS"
echo "================================================================================"
echo "Bucket: $BUCKET"
echo "================================================================================"
echo ""

# Check if gsutil is installed
if ! command -v gsutil &> /dev/null; then
    echo "Error: gsutil not found. Please install Google Cloud SDK first:"
    echo "  https://cloud.google.com/sdk/docs/install"
    exit 1
fi

# Check if authenticated
if ! gsutil ls "$BUCKET" >/dev/null 2>&1; then
    echo "Error: Cannot access bucket. Please authenticate:"
    echo "  gcloud auth login"
    exit 1
fi

log_info "Connected to bucket successfully"
echo ""

# Show total bucket usage
log_section "TOTAL BUCKET USAGE"
gsutil du -sh "$BUCKET"
echo ""

# ============================================================================
# MAIN DIRECTORIES
# ============================================================================
echo "================================================================================"
echo "MAIN DIRECTORIES"
echo "================================================================================"

analyze_dir "$BUCKET/data/" "Data Directory"
analyze_dir "$BUCKET/features/" "Features Directory"
analyze_dir "$BUCKET/outputs/" "Outputs Directory"
analyze_dir "$BUCKET/models/" "Models Directory"

# ============================================================================
# CLIPS
# ============================================================================
echo ""
echo "================================================================================"
echo "CLIPS (by shot type)"
echo "================================================================================"

for shot in Smash Clear Drop Lift Drive Slice_Drop Push Rear_Drive Defensive_Drive Short_Drive; do
    analyze_dir "$BUCKET/data/clips/$shot/" "Clips: $shot"
done

# ============================================================================
# POSES
# ============================================================================
echo ""
echo "================================================================================"
echo "POSES"
echo "================================================================================"

analyze_dir "$BUCKET/features/poses/" "Current Poses"
analyze_dir "$BUCKET/features/poses_old/" "Old Poses"
analyze_dir "$BUCKET/features/poses_backup/" "Backup Poses"
analyze_dir "$BUCKET/data/poses_test/" "Test Poses"

# ============================================================================
# METADATA
# ============================================================================
echo ""
echo "================================================================================"
echo "METADATA FILES"
echo "================================================================================"

# List all CSV files in data directory
if gsutil ls "$BUCKET/data/*.csv" >/dev/null 2>&1; then
    gsutil ls -lh "$BUCKET/data/*.csv" 2>/dev/null | grep -v TOTAL
else
    echo "No CSV files found"
fi

# ============================================================================
# TEMPORARY/DEBUG
# ============================================================================
echo ""
echo "================================================================================"
echo "TEMPORARY AND DEBUG FILES"
echo "================================================================================"

analyze_dir "$BUCKET/tmp/" "Temporary Files"
analyze_dir "$BUCKET/debug/" "Debug Files"
analyze_dir "$BUCKET/test/" "Test Files"

# ============================================================================
# OLD VERSIONS
# ============================================================================
echo ""
echo "================================================================================"
echo "OLD VERSIONS AND BACKUPS"
echo "================================================================================"

analyze_dir "$BUCKET/data/clips_old/" "Old Clips"
analyze_dir "$BUCKET/data/clips_backup/" "Backup Clips"
analyze_dir "$BUCKET/data/clips_v1/" "Version 1 Clips"
analyze_dir "$BUCKET/outputs/old/" "Old Outputs"
analyze_dir "$BUCKET/outputs/backup/" "Backup Outputs"
analyze_dir "$BUCKET/models/old/" "Old Models"

# ============================================================================
# MLFLOW
# ============================================================================
echo ""
echo "================================================================================"
echo "MLflow ARTIFACTS"
echo "================================================================================"

analyze_dir "$BUCKET/mlflow/" "MLflow Directory"
analyze_dir "$BUCKET/mlruns/" "MLflow Runs"

# ============================================================================
# SUMMARY
# ============================================================================
echo ""
echo "================================================================================"
echo "SUMMARY"
echo "================================================================================"
echo ""

# Calculate total by category
echo "Storage breakdown by category:"
echo ""

# Data (clips + metadata)
data_size=$(gsutil du -s "$BUCKET/data/" 2>/dev/null | awk '{printf "%.2f GB\n", $1/1024/1024/1024}')
echo "Data (clips + metadata):  $data_size"

# Features (poses)
features_size=$(gsutil du -s "$BUCKET/features/" 2>/dev/null | awk '{printf "%.2f GB\n", $1/1024/1024/1024}')
echo "Features (poses):         $features_size"

# Outputs (training results)
outputs_size=$(gsutil du -s "$BUCKET/outputs/" 2>/dev/null | awk '{printf "%.2f GB\n", $1/1024/1024/1024}')
echo "Outputs (training):       $outputs_size"

# Models
models_size=$(gsutil du -s "$BUCKET/models/" 2>/dev/null | awk '{printf "%.2f GB\n", $1/1024/1024/1024}')
echo "Models (checkpoints):     $models_size"

echo ""
echo "================================================================================"
echo "RECOMMENDATIONS"
echo "================================================================================"
echo ""
echo "To clean up storage, run:"
echo "  bash scripts/clean_gcs_storage.sh"
echo ""
echo "To delete specific directories manually:"
echo "  gsutil -m rm -r gs://iti123storage/path/to/delete"
echo ""
echo "================================================================================"

exit 0
