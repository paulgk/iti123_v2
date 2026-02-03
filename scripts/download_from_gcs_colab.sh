#!/bin/bash

# Download Data from GCS - For use in Google Colab
# This script downloads poses and metadata from GCS for training

set -e

# Configuration
BUCKET="gs://iti123storage"
GCS_POSES="${BUCKET}/features/poses_roi"
GCS_METADATA="${BUCKET}/data/metadata_roi.csv"

# Local paths
LOCAL_POSES="./data/poses"
LOCAL_METADATA="./data/metadata.csv"

# Colors
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

log_info() { echo -e "${GREEN}[INFO]${NC} $1"; }
log_warn() { echo -e "${YELLOW}[WARN]${NC} $1"; }
log_section() { echo -e "${BLUE}[SECTION]${NC} $1"; }

# Header
echo "================================================================================"
echo "DOWNLOAD DATA FROM GCS"
echo "================================================================================"
echo "Source:  $BUCKET"
echo "Target:  ./data/"
echo "================================================================================"
echo ""

# Check if running in Colab
if [ -d "/content" ]; then
    log_info "Running in Google Colab environment"
else
    log_warn "Not running in Colab - are you sure you want to download locally?"
    read -p "Continue? (y/n): " -n 1 -r
    echo ""
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        exit 0
    fi
fi

# Check gsutil
if ! command -v gsutil &> /dev/null; then
    log_warn "gsutil not found. Installing..."
    if [ -d "/content" ]; then
        # In Colab, gsutil should be pre-installed
        log_warn "gsutil not available. Please authenticate first:"
        echo "  from google.colab import auth"
        echo "  auth.authenticate_user()"
        exit 1
    else
        echo "Install Google Cloud SDK:"
        echo "  brew install --cask google-cloud-sdk"
        exit 1
    fi
fi

# Create local directories
mkdir -p "$(dirname "$LOCAL_POSES")"
mkdir -p "$(dirname "$LOCAL_METADATA")"

# ============================================================================
# DOWNLOAD 1: METADATA
# ============================================================================
log_section "DOWNLOADING METADATA"
echo "Source: $GCS_METADATA"
echo "Target: $LOCAL_METADATA"
echo ""

if gsutil ls "$GCS_METADATA" >/dev/null 2>&1; then
    gsutil cp "$GCS_METADATA" "$LOCAL_METADATA"
    log_info "✓ Metadata downloaded"

    # Show sample
    echo ""
    log_info "Metadata sample:"
    head -3 "$LOCAL_METADATA"
    echo ""

    # Count entries
    local entry_count=$(($(wc -l < "$LOCAL_METADATA") - 1))
    log_info "Metadata entries: $entry_count"
else
    log_warn "Metadata not found in GCS"
    exit 1
fi

echo ""

# ============================================================================
# DOWNLOAD 2: POSES
# ============================================================================
log_section "DOWNLOADING POSES"
echo "Source: $GCS_POSES"
echo "Target: $LOCAL_POSES"
echo ""

log_info "This may take 5-10 minutes..."
log_info "Starting download..."

start_time=$(date +%s)

# Use rsync for efficient download
if gsutil -m rsync -r "$GCS_POSES" "$LOCAL_POSES" 2>&1; then
    end_time=$(date +%s)
    duration=$((end_time - start_time))

    log_info "✓ Poses downloaded in ${duration}s"

    # Count downloaded files
    local pose_count=$(find "$LOCAL_POSES" -name "*.pkl" | wc -l | tr -d ' ')
    log_info "Pose files: $pose_count"

    # Check sample pose
    echo ""
    log_info "Verifying pose format..."
    python3 - <<'EOF'
import pickle
import numpy as np
from pathlib import Path

poses_dir = Path('./data/poses')
pose_files = list(poses_dir.glob('*.pkl'))

if pose_files:
    sample_file = pose_files[0]
    with open(sample_file, 'rb') as f:
        pose = pickle.load(f)

    print(f"Sample pose: {sample_file.name}")
    print(f"  Shape: {pose.shape}")
    print(f"  Frames: {len(pose)}")
    print(f"  Keypoints: {pose.shape[1]}")
    print(f"  Coordinates: {pose.shape[2]}")
    print(f"  Mean: {pose.mean():.4f}")
    print(f"  Std: {pose.std():.4f}")
    print("✓ Pose format valid")
else:
    print("⚠️  No pose files found")
EOF

else
    log_warn "Download failed or incomplete"
    exit 1
fi

# ============================================================================
# VERIFICATION
# ============================================================================
echo ""
echo "================================================================================"
echo "VERIFICATION"
echo "================================================================================"
echo ""

# Check metadata vs poses match
python3 - <<'EOF'
import csv
from pathlib import Path

metadata_file = Path('./data/metadata.csv')
poses_dir = Path('./data/poses')

# Load metadata
with open(metadata_file, 'r') as f:
    reader = csv.DictReader(f)
    metadata_ids = {row['video_id'] for row in reader}

# Load pose files
pose_files = {f.stem for f in poses_dir.glob('*.pkl')}

# Compare
metadata_count = len(metadata_ids)
pose_count = len(pose_files)
match_count = len(metadata_ids & pose_files)

print(f"Metadata entries: {metadata_count}")
print(f"Pose files:       {pose_count}")
print(f"Matching:         {match_count}")
print(f"Success rate:     {match_count/metadata_count*100:.1f}%")

if match_count == metadata_count:
    print("✓ All metadata entries have corresponding poses")
elif match_count / metadata_count > 0.95:
    print(f"⚠️  {metadata_count - match_count} poses missing (still acceptable)")
else:
    print(f"⚠️  Warning: Only {match_count/metadata_count*100:.1f}% poses found")
EOF

# ============================================================================
# SUMMARY
# ============================================================================
echo ""
echo "================================================================================"
echo "DOWNLOAD COMPLETE"
echo "================================================================================"
echo ""

log_info "Data ready for training!"
echo ""
echo "File locations:"
echo "  Metadata: $LOCAL_METADATA"
echo "  Poses:    $LOCAL_POSES/"
echo ""
echo "Next steps:"
echo "  1. Verify data: python scripts/validate_roi_poses.py --poses data/poses --metadata data/metadata.csv"
echo "  2. Train model: python scripts/train_models_fixed.py --model stgcn --epochs 50"
echo ""
echo "================================================================================"

exit 0
