#!/bin/bash
#
# Check Pose Extraction Status
#
# Usage: bash scripts/check_extraction_status.sh

echo "============================================================"
echo "POSE EXTRACTION STATUS CHECK"
echo "============================================================"
echo ""

# Check if pose directory exists
if [ ! -d "data/processed/poses" ]; then
    echo "❌ Pose directory not found: data/processed/poses"
    echo "   Run extraction first"
    exit 1
fi

# Count pose files
POSE_COUNT=$(find data/processed/poses -name "*.pkl" 2>/dev/null | wc -l | tr -d ' ')
echo "Pose Files"
echo "----------"
echo "Total .pkl files: $POSE_COUNT"

if [ "$POSE_COUNT" -eq 0 ]; then
    echo "❌ No pose files found!"
    echo ""
    echo "Troubleshooting:"
    echo "1. Check if extraction is still running:"
    echo "   ps aux | grep extract_poses"
    echo ""
    echo "2. Check extraction output/errors"
    echo "3. Try running extraction manually"
    exit 1
fi

echo "✓ Found $POSE_COUNT pose files"
echo ""

# Check for failed extractions
if [ -f "data/processed/poses/failed_extractions.txt" ]; then
    FAILED_COUNT=$(wc -l < data/processed/poses/failed_extractions.txt | tr -d ' ')
    if [ "$FAILED_COUNT" -gt 0 ]; then
        echo "⚠️  Failed Extractions"
        echo "-------------------"
        echo "Failed files: $FAILED_COUNT"
        echo ""
        echo "First 10 failures:"
        head -10 data/processed/poses/failed_extractions.txt
        echo ""
    fi
else
    echo "✓ No failed extractions file (all succeeded or none failed yet)"
    echo ""
fi

# Check for metadata.csv
echo "Metadata Status"
echo "---------------"
if [ -f "data/metadata.csv" ]; then
    METADATA_LINES=$(wc -l < data/metadata.csv | tr -d ' ')
    echo "✓ metadata.csv exists"
    echo "  Rows: $METADATA_LINES (including header)"
    echo ""

    # Show first few lines
    echo "First 5 rows:"
    head -5 data/metadata.csv
    echo ""
else
    echo "❌ metadata.csv NOT FOUND"
    echo ""
    echo "This means extraction either:"
    echo "1. Is still running (metadata created at the end)"
    echo "2. Failed before completion"
    echo "3. Was run with --create-metadata=False"
    echo ""
fi

# Check if extraction process is running
echo "Process Status"
echo "--------------"
if ps aux | grep -v grep | grep extract_poses > /dev/null; then
    echo "✓ Extraction process is RUNNING"
    echo ""
    ps aux | grep -v grep | grep extract_poses
    echo ""
    echo "⏳ Wait for extraction to complete before running validation"
else
    echo "❌ No extraction process running"
    echo ""
    if [ "$POSE_COUNT" -gt 0 ] && [ ! -f "data/metadata.csv" ]; then
        echo "⚠️  Pose files exist but no metadata.csv!"
        echo "   Extraction may have been interrupted."
        echo ""
        echo "Solutions:"
        echo "1. Re-run extraction (will skip existing files)"
        echo "2. Manually create metadata.csv from pose files"
        echo ""
    fi
fi

# Sample pose file check
echo "Sample Pose File Check"
echo "----------------------"
SAMPLE_POSE=$(find data/processed/poses -name "*.pkl" | head -1)
if [ -n "$SAMPLE_POSE" ]; then
    echo "Sample file: $(basename $SAMPLE_POSE)"

    # Check file size
    SIZE=$(ls -lh "$SAMPLE_POSE" | awk '{print $5}')
    echo "File size: $SIZE"

    # Try to read with Python
    python3 << EOF
import pickle
import sys

try:
    with open('$SAMPLE_POSE', 'rb') as f:
        pose = pickle.load(f)
    print(f"✓ File is valid pickle")
    print(f"  Shape: {pose.shape}")
    print(f"  Expected: (n_frames, 33, 3)")

    if len(pose.shape) == 3 and pose.shape[1:] == (33, 3):
        print(f"  ✓ Format is correct")
    else:
        print(f"  ⚠️  Unexpected shape")
except Exception as e:
    print(f"❌ Error reading pose file: {e}")
    sys.exit(1)
EOF
else
    echo "No pose files to sample"
fi

echo ""
echo "============================================================"
echo "SUMMARY"
echo "============================================================"

if [ -f "data/metadata.csv" ] && [ "$POSE_COUNT" -gt 0 ]; then
    echo "✓ Extraction COMPLETE"
    echo "  - Pose files: $POSE_COUNT"
    echo "  - Metadata: Ready"
    echo ""
    echo "Next steps:"
    echo "  1. Review metadata: cat data/metadata.csv | head -20"
    echo "  2. Run validation: python scripts/validate_phase2.py --sample-size 100"
    echo "  3. Or use notebook: notebooks/phase2_validation_colab.ipynb"
elif [ "$POSE_COUNT" -gt 0 ] && [ ! -f "data/metadata.csv" ]; then
    if ps aux | grep -v grep | grep extract_poses > /dev/null; then
        echo "⏳ Extraction IN PROGRESS"
        echo "  - Pose files so far: $POSE_COUNT"
        echo "  - Metadata: Will be created when complete"
        echo ""
        echo "Wait for completion or check progress periodically"
    else
        echo "⚠️  Extraction INCOMPLETE"
        echo "  - Pose files: $POSE_COUNT"
        echo "  - Metadata: Missing (extraction may have failed)"
        echo ""
        echo "Run this script to create metadata from existing poses:"
        echo "  python scripts/create_metadata_from_poses.py"
    fi
else
    echo "❌ Extraction NOT STARTED or FAILED"
    echo ""
    echo "Start extraction with:"
    echo "  python scripts/extract_poses_parallel.py \\"
    echo "      --video-dir data/videos/ \\"
    echo "      --output-dir data/processed/poses/ \\"
    echo "      --model-complexity 1 \\"
    echo "      --target-fps 20 \\"
    echo "      --num-workers 4"
fi

echo ""
