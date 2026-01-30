#!/bin/bash
# Pull data from GCS at session start
# Usage: ./scripts/pull_data.sh [--videos] [--features] [--all] [--dry-run]

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(dirname "$SCRIPT_DIR")"

# Load config
CONFIG_FILE="$REPO_ROOT/config/paths.yaml"
if [ ! -f "$CONFIG_FILE" ]; then
    echo "Error: Config file not found: $CONFIG_FILE"
    exit 1
fi

# Parse bucket name from config (simple grep, works for our format)
BUCKET="${GCS_BUCKET_NAME:-$(grep 'bucket:' "$CONFIG_FILE" | head -1 | awk '{print $2}' | tr -d '"')}"

# Detect environment
if [ -d "/content" ]; then
    ENV="colab"
    DATA_DIR="/content/data"
else
    ENV="local"
    DATA_DIR="$REPO_ROOT/data"
fi

echo "Environment: $ENV"
echo "GCS Bucket: $BUCKET"
echo "Data Dir: $DATA_DIR"

# Parse arguments
PULL_VIDEOS=false
PULL_FEATURES=false
DRY_RUN=""

for arg in "$@"; do
    case $arg in
        --videos) PULL_VIDEOS=true ;;
        --features) PULL_FEATURES=true ;;
        --all) PULL_VIDEOS=true; PULL_FEATURES=true ;;
        --dry-run) DRY_RUN="-n" ;;
        --help)
            echo "Usage: $0 [--videos] [--features] [--all] [--dry-run]"
            echo "  --videos    Pull video files from GCS"
            echo "  --features  Pull feature files from GCS"
            echo "  --all       Pull both videos and features"
            echo "  --dry-run   Show what would be synced without doing it"
            exit 0
            ;;
    esac
done

# Default to features if nothing specified
if [ "$PULL_VIDEOS" = false ] && [ "$PULL_FEATURES" = false ]; then
    PULL_FEATURES=true
fi

# Create directories
mkdir -p "$DATA_DIR/videos"
mkdir -p "$DATA_DIR/processed/features"

# Pull videos
if [ "$PULL_VIDEOS" = true ]; then
    echo ""
    echo "=== Pulling videos ==="
    gsutil -m rsync $DRY_RUN -r "gs://$BUCKET/videos/" "$DATA_DIR/videos/"
fi

# Pull features
if [ "$PULL_FEATURES" = true ]; then
    echo ""
    echo "=== Pulling features ==="
    gsutil -m rsync $DRY_RUN -r "gs://$BUCKET/features/" "$DATA_DIR/processed/features/"
fi

echo ""
echo "Pull complete!"
