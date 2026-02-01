#!/bin/bash
################################################################################
# Organize GCS Video Clips into Stroke Type Folders
#
# This script organizes video files in GCS into subdirectories based on
# stroke type detected in filename.
#
# Usage:
#   # Dry run (preview only)
#   bash scripts/organize_gcs_videos.sh gs://iti123storage/videos/clips
#
#   # Execute the organization
#   bash scripts/organize_gcs_videos.sh gs://iti123storage/videos/clips --execute
#
################################################################################

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Configuration
GCS_PATH="${1:-}"
EXECUTE_MODE="${2:-}"

# Validate inputs
if [ -z "$GCS_PATH" ]; then
    echo -e "${RED}Error: GCS path required${NC}"
    echo "Usage: bash $0 gs://bucket/path [--execute]"
    exit 1
fi

if [[ ! "$GCS_PATH" =~ ^gs:// ]]; then
    echo -e "${RED}Error: Path must start with gs://${NC}"
    exit 1
fi

# Check if gsutil is available
if ! command -v gsutil &> /dev/null; then
    echo -e "${RED}Error: gsutil not found${NC}"
    echo "Install with: pip install gsutil"
    exit 1
fi

# Determine mode
DRY_RUN=true
if [ "$EXECUTE_MODE" == "--execute" ]; then
    DRY_RUN=false
fi

################################################################################
# Functions
################################################################################

detect_stroke_type() {
    local filename="$1"
    local filename_lower=$(echo "$filename" | tr '[:upper:]' '[:lower:]')

    # Pattern matching (in priority order)
    if echo "$filename_lower" | grep -qE '\bclear\b|\\bclr\b|_c_|_c[0-9]+'; then
        echo "clear"
    elif echo "$filename_lower" | grep -qE '\bsmash\b|\bsmsh\b|_s_|_s[0-9]+'; then
        echo "smash"
    elif echo "$filename_lower" | grep -qE '\bdrop\b|\bdrp\b|_d_|_d[0-9]+'; then
        echo "drop"
    elif echo "$filename_lower" | grep -qE '\blift\b|\blft\b|_l_|_l[0-9]+'; then
        echo "lift"
    else
        echo "unknown"
    fi
}

################################################################################
# Main Script
################################################################################

echo "================================================================="
echo "VIDEO ORGANIZATION - GCS"
echo "================================================================="
echo "GCS Path: $GCS_PATH"
echo "Mode: $([ "$DRY_RUN" == true ] && echo 'DRY-RUN' || echo 'EXECUTE')"
echo ""

# List all video files
echo -e "${BLUE}Listing video files...${NC}"

# Try direct listing first
VIDEO_FILES=$(gsutil ls "${GCS_PATH}/*.mp4" 2>/dev/null || true)

# If no files found, try recursive
if [ -z "$VIDEO_FILES" ]; then
    VIDEO_FILES=$(gsutil ls -r "${GCS_PATH}/**/*.mp4" 2>/dev/null || true)
fi

if [ -z "$VIDEO_FILES" ]; then
    echo -e "${RED}No video files found in $GCS_PATH${NC}"
    exit 1
fi

# Filter out files already in subdirectories
FILTERED_FILES=""
while IFS= read -r video_file; do
    # Extract path relative to GCS_PATH
    relative_path="${video_file#$GCS_PATH/}"

    # Check if file is in a subdirectory
    if echo "$relative_path" | grep -qE '^(clear|smash|drop|lift|unknown)/'; then
        continue  # Skip files already organized
    fi

    FILTERED_FILES="${FILTERED_FILES}${video_file}"$'\n'
done <<< "$VIDEO_FILES"

# Remove trailing newline
FILTERED_FILES=$(echo "$FILTERED_FILES" | sed '/^$/d')

if [ -z "$FILTERED_FILES" ]; then
    echo -e "${YELLOW}No unorganized video files found${NC}"
    echo "All videos are already in stroke type folders"
    exit 0
fi

# Count files
FILE_COUNT=$(echo "$FILTERED_FILES" | wc -l | tr -d ' ')
echo -e "${GREEN}Found $FILE_COUNT video files to organize${NC}"
echo ""

# Categorize files using temporary files instead of associative arrays
TEMP_DIR=$(mktemp -d)
trap "rm -rf $TEMP_DIR" EXIT

clear_files="$TEMP_DIR/clear.txt"
smash_files="$TEMP_DIR/smash.txt"
drop_files="$TEMP_DIR/drop.txt"
lift_files="$TEMP_DIR/lift.txt"
unknown_files="$TEMP_DIR/unknown.txt"

touch "$clear_files" "$smash_files" "$drop_files" "$lift_files" "$unknown_files"

while IFS= read -r video_file; do
    filename=$(basename "$video_file")
    stroke_type=$(detect_stroke_type "$filename")

    case "$stroke_type" in
        clear)
            echo "$video_file" >> "$clear_files"
            ;;
        smash)
            echo "$video_file" >> "$smash_files"
            ;;
        drop)
            echo "$video_file" >> "$drop_files"
            ;;
        lift)
            echo "$video_file" >> "$lift_files"
            ;;
        unknown)
            echo "$video_file" >> "$unknown_files"
            ;;
    esac
done <<< "$FILTERED_FILES"

# Show distribution
echo "Detected stroke types:"
for stroke_type in clear smash drop lift unknown; do
    case "$stroke_type" in
        clear) file_list="$clear_files" ;;
        smash) file_list="$smash_files" ;;
        drop) file_list="$drop_files" ;;
        lift) file_list="$lift_files" ;;
        unknown) file_list="$unknown_files" ;;
    esac

    count=$(cat "$file_list" | wc -l | tr -d ' ')
    if [ "$count" -gt 0 ]; then
        printf "  %-8s: %4d files\n" "$stroke_type" "$count"
    fi
done
echo ""

# Organize files
total_moved=0

for stroke_type in clear smash drop lift unknown; do
    case "$stroke_type" in
        clear) file_list="$clear_files" ;;
        smash) file_list="$smash_files" ;;
        drop) file_list="$drop_files" ;;
        lift) file_list="$lift_files" ;;
        unknown) file_list="$unknown_files" ;;
    esac

    count=$(cat "$file_list" | wc -l | tr -d ' ')

    if [ "$count" -eq 0 ]; then
        continue
    fi

    echo -e "${BLUE}Processing $stroke_type ($count files)...${NC}"

    while IFS= read -r video_file; do
        if [ -z "$video_file" ]; then
            continue
        fi

        filename=$(basename "$video_file")
        dest_path="${GCS_PATH}/${stroke_type}/${filename}"

        if [ "$DRY_RUN" == true ]; then
            echo "[MOVE] $filename -> ${stroke_type}/"
        else
            # Execute the move
            if gsutil mv "$video_file" "$dest_path" 2>/dev/null; then
                echo -e "${GREEN}✓${NC} Moved: $filename -> ${stroke_type}/"
                total_moved=$((total_moved + 1))
            else
                echo -e "${RED}✗${NC} Error: Failed to move $filename"
            fi
        fi
    done < "$file_list"

    echo ""
done

# Summary
echo "================================================================="
echo "SUMMARY"
echo "================================================================="
echo "Total files processed: $FILE_COUNT"

for stroke_type in clear smash drop lift unknown; do
    case "$stroke_type" in
        clear) file_list="$clear_files" ;;
        smash) file_list="$smash_files" ;;
        drop) file_list="$drop_files" ;;
        lift) file_list="$lift_files" ;;
        unknown) file_list="$unknown_files" ;;
    esac

    count=$(cat "$file_list" | wc -l | tr -d ' ')
    if [ "$count" -gt 0 ]; then
        printf "  %-8s: %4d files\n" "$stroke_type" "$count"
    fi
done

if [ "$DRY_RUN" == true ]; then
    echo ""
    echo -e "${YELLOW}⚠️  DRY-RUN mode - no changes were made${NC}"
    echo "   Run with --execute to organize files:"
    echo "   bash $0 $GCS_PATH --execute"
else
    echo ""
    echo -e "${GREEN}✓ Organization complete${NC}"
    echo "   Files moved: $total_moved"
fi

echo ""
