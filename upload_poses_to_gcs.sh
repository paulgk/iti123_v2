#!/bin/bash
#
# Upload extracted pose features to Google Cloud Storage
#

set -e

echo "=========================================="
echo "UPLOADING POSE FEATURES TO GCS"
echo "=========================================="
echo ""

GCS_BUCKET="gs://iti123storage"

# Check if gsutil is available
if ! command -v gsutil &> /dev/null; then
    echo "❌ Error: gsutil not found. Please install Google Cloud SDK."
    echo "  Install: https://cloud.google.com/sdk/docs/install"
    exit 1
fi

echo "Source: data/processed/poses/"
echo "Destination: ${GCS_BUCKET}/features/poses/"
echo ""

# Count local poses
LOCAL_COUNT=$(find data/processed/poses -name "*_pose.pkl" | wc -l | tr -d ' ')
echo "Local pose files: ${LOCAL_COUNT}"

# Upload poses
echo ""
echo "Uploading pose files..."
gsutil -m rsync -r data/processed/poses/ ${GCS_BUCKET}/features/poses/

# Upload metadata
echo ""
echo "Uploading metadata..."
gsutil cp data/data/metadata.csv ${GCS_BUCKET}/data/metadata.csv

# Verify upload
echo ""
echo "Verifying upload..."
REMOTE_COUNT=$(gsutil ls ${GCS_BUCKET}/features/poses/*.pkl 2>/dev/null | wc -l | tr -d ' ')
echo "Remote pose files: ${REMOTE_COUNT}"

if [ "$LOCAL_COUNT" -eq "$REMOTE_COUNT" ]; then
    echo "✓ Upload verified successfully!"
else
    echo "⚠️  Warning: File count mismatch (local: ${LOCAL_COUNT}, remote: ${REMOTE_COUNT})"
fi

echo ""
echo "=========================================="
echo "UPLOAD COMPLETE"
echo "=========================================="
echo ""
echo "GCS locations:"
echo "  Poses: ${GCS_BUCKET}/features/poses/"
echo "  Metadata: ${GCS_BUCKET}/data/metadata.csv"
echo ""
echo "Next steps:"
echo "  1. Open notebooks/model_comparison_colab.ipynb in Colab"
echo "  2. Enable T4 GPU runtime"
echo "  3. Run all cells to train models"
echo ""
