#!/bin/bash
#
# Upload extracted video clips to Google Cloud Storage
#
# Usage:
#   bash scripts/upload_clips_to_gcs.sh --dry-run    # Preview upload
#   bash scripts/upload_clips_to_gcs.sh --execute    # Actually upload
#

set -e  # Exit on error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Configuration
LOCAL_CLIPS_DIR="data/clips"
GCS_BUCKET="gs://iti123storage/videos/clips"
DRY_RUN=true
BATCH_SIZE=200  # Process 200 clips at a time to prevent memory crashes

# Parse arguments
if [ "$1" == "--execute" ]; then
    DRY_RUN=false
elif [ "$1" == "--dry-run" ] || [ "$1" == "" ]; then
    DRY_RUN=true
else
    echo "Usage: $0 [--dry-run | --execute]"
    exit 1
fi

echo "================================================================="
echo "UPLOAD CLIPS TO GOOGLE CLOUD STORAGE"
echo "================================================================="
echo "Local source:  $LOCAL_CLIPS_DIR"
echo "GCS destination: $GCS_BUCKET"
echo "Mode:          $([ "$DRY_RUN" == true ] && echo 'DRY-RUN (preview only)' || echo 'EXECUTE (uploading)')"
echo "================================================================="
echo ""

# Check if local directory exists
if [ ! -d "$LOCAL_CLIPS_DIR" ]; then
    echo -e "${RED}✗ Error: Local directory not found: $LOCAL_CLIPS_DIR${NC}"
    exit 1
fi

# Check if gsutil is available
if ! command -v gsutil &> /dev/null; then
    echo -e "${RED}✗ Error: gsutil not found. Please install Google Cloud SDK.${NC}"
    echo "  Install: https://cloud.google.com/sdk/docs/install"
    exit 1
fi

# Count local clips
echo "Counting local clips..."
for shot_type in Smash Clear Drop Lift Drive; do
    shot_dir="$LOCAL_CLIPS_DIR/$shot_type"
    if [ -d "$shot_dir" ]; then
        count=$(find "$shot_dir" -name "*.mp4" | wc -l | tr -d ' ')
        echo "  $shot_type: $count clips"
    fi
done
echo ""

# Upload function with chunked batches
upload_shot_type() {
    local shot_type=$1
    local shot_type_lower=$(echo "$shot_type" | tr '[:upper:]' '[:lower:]')
    local local_dir="$LOCAL_CLIPS_DIR/$shot_type"
    local gcs_path="$GCS_BUCKET/$shot_type_lower"

    if [ ! -d "$local_dir" ]; then
        echo -e "${YELLOW}⚠️  Skipping $shot_type: directory not found${NC}"
        return
    fi

    # Get list of all clips
    local clips=()
    while IFS= read -r -d '' file; do
        clips+=("$file")
    done < <(find "$local_dir" -name "*.mp4" -print0 | sort -z)

    local clip_count=${#clips[@]}

    if [ $clip_count -eq 0 ]; then
        echo -e "${YELLOW}⚠️  Skipping $shot_type: no clips found${NC}"
        return
    fi

    local batch_count=$(( (clip_count + BATCH_SIZE - 1) / BATCH_SIZE ))

    echo -e "${BLUE}Processing $shot_type ($clip_count clips in $batch_count batches)...${NC}"

    if [ "$DRY_RUN" == true ]; then
        echo "[DRY-RUN] Would create directory: $gcs_path/"
        echo "[DRY-RUN] Would upload $clip_count clips in batches of $BATCH_SIZE"
        for ((batch=0; batch<batch_count; batch++)); do
            local start=$((batch * BATCH_SIZE))
            local end=$((start + BATCH_SIZE))
            if [ $end -gt $clip_count ]; then
                end=$clip_count
            fi
            echo "[DRY-RUN] Batch $((batch+1))/$batch_count: clips $((start+1))-$end"
        done
    else
        # Create destination directory if it doesn't exist
        echo "  Creating directory: $gcs_path/"
        if ! gsutil ls "$gcs_path/" >/dev/null 2>&1; then
            # Directory doesn't exist, create it with a placeholder
            echo "dummy" | gsutil cp - "$gcs_path/.keep"
            echo "  ✓ Created directory"
        else
            echo "  ✓ Directory exists"
        fi

        # Upload in batches to prevent memory crashes
        local uploaded=0
        local failed=0

        for ((batch=0; batch<batch_count; batch++)); do
            local start=$((batch * BATCH_SIZE))
            local end=$((start + BATCH_SIZE))
            if [ $end -gt $clip_count ]; then
                end=$clip_count
            fi
            local batch_size=$((end - start))

            echo -e "  ${BLUE}Batch $((batch+1))/$batch_count${NC}: uploading clips $((start+1))-$end ($batch_size files)..."

            # Upload batch
            local batch_failed=0
            for ((i=start; i<end; i++)); do
                local clip="${clips[$i]}"
                local filename=$(basename "$clip")

                # Check if file already exists in GCS
                if gsutil -q stat "$gcs_path/$filename" 2>/dev/null; then
                    # File exists, skip
                    ((uploaded++))
                else
                    # Upload file
                    if gsutil -q cp "$clip" "$gcs_path/$filename" 2>/dev/null; then
                        ((uploaded++))
                    else
                        ((failed++))
                        ((batch_failed++))
                        echo -e "    ${RED}✗${NC} Failed: $filename"
                    fi
                fi
            done

            if [ $batch_failed -eq 0 ]; then
                echo -e "    ${GREEN}✓${NC} Batch complete ($batch_size files)"
            else
                echo -e "    ${YELLOW}⚠️${NC} Batch complete with $batch_failed failures"
            fi

            # Small delay between batches to prevent overwhelming the API
            sleep 1
        done

        if [ $failed -eq 0 ]; then
            echo -e "${GREEN}✓${NC} Uploaded $uploaded/$clip_count $shot_type clips"
        else
            echo -e "${YELLOW}⚠️${NC} Uploaded $uploaded/$clip_count $shot_type clips ($failed failed)"
        fi
    fi

    echo ""
}

# Upload each shot type
echo "================================================================="
echo "UPLOADING CLIPS"
echo "================================================================="
echo ""

for shot_type in Smash Clear Drop Lift Drive; do
    upload_shot_type "$shot_type"
done

# Verify upload (if not dry-run)
if [ "$DRY_RUN" == false ]; then
    echo "================================================================="
    echo "VERIFYING UPLOAD"
    echo "================================================================="
    echo ""

    for shot_type in Smash Clear Drop Lift Drive; do
        shot_type_lower=$(echo "$shot_type" | tr '[:upper:]' '[:lower:]')
        gcs_path="$GCS_BUCKET/$shot_type_lower"

        if gsutil -q ls "$gcs_path" 2>/dev/null; then
            gcs_count=$(gsutil ls "$gcs_path/*.mp4" 2>/dev/null | wc -l | tr -d ' ')
            local_count=$(find "$LOCAL_CLIPS_DIR/$shot_type" -name "*.mp4" 2>/dev/null | wc -l | tr -d ' ')

            if [ "$gcs_count" -eq "$local_count" ]; then
                echo -e "${GREEN}✓${NC} $shot_type: $gcs_count/$local_count clips uploaded"
            else
                echo -e "${YELLOW}⚠️${NC} $shot_type: $gcs_count/$local_count clips (mismatch!)"
            fi
        else
            echo -e "${RED}✗${NC} $shot_type: GCS path not found"
        fi
    done
    echo ""
fi

# Summary
echo "================================================================="
echo "SUMMARY"
echo "================================================================="

if [ "$DRY_RUN" == true ]; then
    echo -e "${YELLOW}⚠️  DRY-RUN mode - no files were uploaded${NC}"
    echo ""
    echo "To execute upload, run:"
    echo "  bash $0 --execute"
else
    total_local=$(find "$LOCAL_CLIPS_DIR" -name "*.mp4" | wc -l | tr -d ' ')
    echo "✓ Upload complete!"
    echo "  Total clips uploaded: ~$total_local"
    echo "  GCS location: $GCS_BUCKET"
    echo ""
    echo "Next steps:"
    echo "  1. Verify in GCS Console: https://console.cloud.google.com/storage/browser/iti123storage/videos/clips"
    echo "  2. Update Colab scripts to use new GCS paths"
    echo "  3. Run pose extraction on Colab for better performance"
fi

echo "================================================================="
