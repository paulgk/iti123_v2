#!/bin/bash
# Push results to GCS at session end
# Usage: ./scripts/push_results.sh [--outputs] [--checkpoints] [--models] [--all] [--dry-run]

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(dirname "$SCRIPT_DIR")"

# Load config
CONFIG_FILE="$REPO_ROOT/config/paths.yaml"
if [ ! -f "$CONFIG_FILE" ]; then
    echo "Error: Config file not found: $CONFIG_FILE"
    exit 1
fi

# Parse bucket name from config
BUCKET="${GCS_BUCKET_NAME:-$(grep 'bucket:' "$CONFIG_FILE" | head -1 | awk '{print $2}' | tr -d '"')}"

# Detect environment
if [ -d "/content" ]; then
    ENV="colab"
    OUTPUT_DIR="/content/outputs"
    CHECKPOINT_DIR="/content/checkpoints"
else
    ENV="local"
    OUTPUT_DIR="$REPO_ROOT/outputs"
    CHECKPOINT_DIR="$REPO_ROOT/checkpoints"
fi

echo "Environment: $ENV"
echo "GCS Bucket: $BUCKET"

# Parse arguments
PUSH_OUTPUTS=false
PUSH_CHECKPOINTS=false
PUSH_MODELS=false
DRY_RUN=""

for arg in "$@"; do
    case $arg in
        --outputs) PUSH_OUTPUTS=true ;;
        --checkpoints) PUSH_CHECKPOINTS=true ;;
        --models) PUSH_MODELS=true ;;
        --all) PUSH_OUTPUTS=true; PUSH_CHECKPOINTS=true; PUSH_MODELS=true ;;
        --dry-run) DRY_RUN="-n" ;;
        --help)
            echo "Usage: $0 [--outputs] [--checkpoints] [--models] [--all] [--dry-run]"
            echo "  --outputs      Push outputs directory to GCS"
            echo "  --checkpoints  Push checkpoints to GCS"
            echo "  --models       Push experimental models to GCS"
            echo "  --all          Push all (outputs + checkpoints + models)"
            echo "  --dry-run      Show what would be synced without doing it"
            exit 0
            ;;
    esac
done

# Default to outputs if nothing specified
if [ "$PUSH_OUTPUTS" = false ] && [ "$PUSH_CHECKPOINTS" = false ] && [ "$PUSH_MODELS" = false ]; then
    PUSH_OUTPUTS=true
fi

# Push outputs
if [ "$PUSH_OUTPUTS" = true ]; then
    if [ -d "$OUTPUT_DIR" ]; then
        echo ""
        echo "=== Pushing outputs ==="
        gsutil -m rsync $DRY_RUN -r "$OUTPUT_DIR" "gs://$BUCKET/outputs/"
    else
        echo "Warning: Output directory not found: $OUTPUT_DIR"
    fi
fi

# Push checkpoints
if [ "$PUSH_CHECKPOINTS" = true ]; then
    if [ -d "$CHECKPOINT_DIR" ]; then
        echo ""
        echo "=== Pushing checkpoints ==="
        gsutil -m rsync $DRY_RUN -r "$CHECKPOINT_DIR" "gs://$BUCKET/checkpoints/"
    else
        echo "Note: No checkpoints directory: $CHECKPOINT_DIR"
    fi
fi

# Push experimental models (not production - those go via git lfs)
if [ "$PUSH_MODELS" = true ]; then
    MODEL_DIR="$REPO_ROOT/models/experiments"
    if [ -d "$MODEL_DIR" ]; then
        echo ""
        echo "=== Pushing experimental models ==="
        gsutil -m rsync $DRY_RUN -r "$MODEL_DIR" "gs://$BUCKET/models/experiments/"
    else
        echo "Note: No experimental models directory: $MODEL_DIR"
    fi
fi

echo ""
echo "Push complete!"
