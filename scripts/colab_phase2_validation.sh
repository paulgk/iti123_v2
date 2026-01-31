#!/bin/bash
#
# Complete Phase 2 Validation Workflow for Colab Enterprise
#
# This script orchestrates the entire validation process:
#   1. Pull videos from GCS
#   2. Extract poses with MediaPipe
#   3. Run validation suite
#   4. Run feature selection
#   5. Upload results to GCS
#
# Usage:
#   bash scripts/colab_phase2_validation.sh [--sample-videos N] [--skip-upload]
#

set -e  # Exit on error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Configuration
GCS_BUCKET="${GCS_BUCKET:-iti123storage}"
SAMPLE_SIZE="${1:-all}"  # Number of videos to process (or 'all')
SKIP_UPLOAD="${2:-false}"

echo -e "${GREEN}======================================================"
echo "PHASE 2 VALIDATION WORKFLOW - COLAB ENTERPRISE"
echo -e "======================================================${NC}"
echo "Started: $(date '+%Y-%m-%d %H:%M:%S')"
echo "GCS Bucket: gs://$GCS_BUCKET"
echo ""

# Step 0: Environment Check
echo -e "${YELLOW}[Step 0/10] Checking environment...${NC}"

if [ ! -d "colab_venv" ]; then
    echo "Creating Python 3.10 virtual environment..."
    bash scripts/colab_setup.sh
fi

source colab_venv/bin/activate
python --version

# Verify GCS access
if ! gsutil ls "gs://$GCS_BUCKET/" > /dev/null 2>&1; then
    echo -e "${RED}ERROR: Cannot access GCS bucket gs://$GCS_BUCKET/${NC}"
    echo "Please set GOOGLE_APPLICATION_CREDENTIALS or authenticate with gcloud"
    exit 1
fi

echo -e "${GREEN}✓ Environment ready${NC}"
echo ""

# Step 1: Pull Videos from GCS
echo -e "${YELLOW}[Step 1/10] Pulling videos from GCS...${NC}"

mkdir -p data/videos/clips
mkdir -p data/processed/poses

if [ "$SAMPLE_SIZE" = "all" ]; then
    echo "Downloading all video clips..."
    gsutil -m rsync -r "gs://$GCS_BUCKET/videos/clips/" data/videos/clips/
else
    echo "Downloading sample of $SAMPLE_SIZE videos..."
    # Download limited number
    gsutil ls "gs://$GCS_BUCKET/videos/clips/clear/*.mp4" | head -$((SAMPLE_SIZE / 2)) | gsutil -m cp -I data/videos/clips/clear/
    gsutil ls "gs://$GCS_BUCKET/videos/clips/smash/*.mp4" | head -$((SAMPLE_SIZE / 2)) | gsutil -m cp -I data/videos/clips/smash/
fi

VIDEO_COUNT=$(find data/videos/clips -name "*.mp4" | wc -l | tr -d ' ')
echo -e "${GREEN}✓ Downloaded $VIDEO_COUNT video clips${NC}"
echo ""

# Step 2: Extract Pose Sequences
echo -e "${YELLOW}[Step 2/10] Extracting pose sequences with MediaPipe...${NC}"

if [ -f "data/processed/poses/failed_extractions.txt" ]; then
    rm data/processed/poses/failed_extractions.txt
fi

python scripts/extract_poses.py \
    --video-dir data/videos/clips \
    --output-dir data/processed/poses \
    --target-fps 30 \
    --min-confidence 0.5 \
    --create-metadata \
    --verbose

POSE_COUNT=$(ls data/processed/poses/*.pkl 2>/dev/null | wc -l | tr -d ' ')
echo -e "${GREEN}✓ Extracted $POSE_COUNT pose sequences${NC}"
echo ""

# Step 3: Review Metadata
echo -e "${YELLOW}[Step 3/10] Reviewing metadata...${NC}"

if [ -f "data/metadata.csv" ]; then
    python -c "
import pandas as pd
df = pd.read_csv('data/metadata.csv')
print(f'Total samples: {len(df)}')
print(f'\nStroke distribution:')
print(df['stroke_type'].value_counts())
unknown = df[df['stroke_type'] == 'unknown']
if len(unknown) > 0:
    print(f'\n⚠️  Warning: {len(unknown)} videos with unknown stroke type')
    print('Please manually label these in data/metadata.csv')
"
    echo -e "${GREEN}✓ Metadata created${NC}"
else
    echo -e "${RED}✗ Metadata file not created${NC}"
    exit 1
fi
echo ""

# Step 4: Upload Poses to GCS (backup)
echo -e "${YELLOW}[Step 4/10] Uploading poses to GCS (backup)...${NC}"

if [ "$SKIP_UPLOAD" = "false" ]; then
    gsutil -m rsync -r data/processed/poses/ "gs://$GCS_BUCKET/features/poses/"
    gsutil cp data/metadata.csv "gs://$GCS_BUCKET/metadata.csv"
    echo -e "${GREEN}✓ Poses backed up to GCS${NC}"
else
    echo "Skipping upload (--skip-upload flag)"
fi
echo ""

# Step 5: Run Phase 2 Validation Suite
echo -e "${YELLOW}[Step 5/10] Running Phase 2 validation suite...${NC}"

VALIDATION_SAMPLE=$((POSE_COUNT < 100 ? POSE_COUNT : 100))

python scripts/validate_phase2.py \
    --sample-size $VALIDATION_SAMPLE \
    --metadata data/metadata.csv \
    --skip-selection

VALIDATION_EXIT=$?

if [ $VALIDATION_EXIT -eq 0 ]; then
    echo -e "${GREEN}✓ Validation suite passed${NC}"
else
    echo -e "${YELLOW}⚠️  Some validations failed (check output above)${NC}"
fi
echo ""

# Step 6: Run Feature Selection Pipeline
echo -e "${YELLOW}[Step 6/10] Running feature selection pipeline...${NC}"

if [ $POSE_COUNT -lt 50 ]; then
    echo -e "${YELLOW}Warning: Only $POSE_COUNT samples available${NC}"
    echo "Feature selection may not be reliable with <50 samples"
    echo "Skipping feature selection (need more data)"
else
    python scripts/run_feature_selection.py \
        --metadata data/metadata.csv \
        --target-features 254

    if [ -f "data/processed/features_v3/selected_features.json" ]; then
        SELECTED_FEATURES=$(python -c "import json; f=open('data/processed/features_v3/selected_features.json'); d=json.load(f); print(len(d.get('selected_features', [])))")
        echo -e "${GREEN}✓ Feature selection complete: $SELECTED_FEATURES features selected${NC}"
    else
        echo -e "${RED}✗ Feature selection failed${NC}"
    fi
fi
echo ""

# Step 7: Test v3 Extraction
echo -e "${YELLOW}[Step 7/10] Testing v3 feature extraction...${NC}"

python -c "
from src.data_processing.feature_versioning import FeatureEngineering
import pickle
import os

# Find first pose file
pose_files = [f for f in os.listdir('data/processed/poses') if f.endswith('.pkl')]
if not pose_files:
    print('No pose files found')
    exit(1)

pose_path = os.path.join('data/processed/poses', pose_files[0])
pose = pickle.load(open(pose_path, 'rb'))

# Test v3 extraction
fe = FeatureEngineering('v3')
features = fe.extract_features(pose, apply_selection=True)
print(f'✓ V3 extraction successful: {len(features)} features')

# Test v2 extraction (backward compatibility)
fe_v2 = FeatureEngineering('v2')
features_v2 = fe_v2.extract_features(pose)
print(f'✓ V2 extraction successful: {len(features_v2)} features')
"

echo -e "${GREEN}✓ Feature extraction verified${NC}"
echo ""

# Step 8: Generate Summary Report
echo -e "${YELLOW}[Step 8/10] Generating summary report...${NC}"

REPORT_PATH="outputs/reports/phase2_validation_summary.txt"
mkdir -p outputs/reports

cat > "$REPORT_PATH" <<EOF
Phase 2 Validation Summary
==========================
Generated: $(date '+%Y-%m-%d %H:%M:%S')

Dataset Statistics
------------------
Total video clips: $VIDEO_COUNT
Pose sequences extracted: $POSE_COUNT
Validation sample size: $VALIDATION_SAMPLE

Validation Results
------------------
$(python scripts/validate_phase2.py --sample-size $VALIDATION_SAMPLE --metadata data/metadata.csv --skip-selection 2>&1 | grep -A 10 "VALIDATION SUMMARY" || echo "See validation output above")

Feature Selection
-----------------
$(if [ -f "data/processed/features_v3/selected_features.json" ]; then
    python -c "import json; f=open('data/processed/features_v3/selected_features.json'); d=json.load(f); print(f'Selected features: {len(d.get(\"selected_features\", []))}'); print(f'Selection method: {d.get(\"selection_method\", \"N/A\")}'); print(f'Date: {d.get(\"selection_date\", \"N/A\")}')"
else
    echo "Feature selection not run (insufficient data)"
fi)

Files Generated
---------------
- Metadata: data/metadata.csv
- Poses: data/processed/poses/*.pkl ($POSE_COUNT files)
- Feature manifest: data/processed/features_v3/selected_features.json
- Detailed report: outputs/reports/feature_selection_report.md
- Verification: .planning/phases/02-feature-engineering-enhancement/02-VERIFICATION.md

Next Steps
----------
1. Review detailed reports in outputs/reports/
2. Check verification status in .planning/phases/02-feature-engineering-enhancement/02-VERIFICATION.md
3. If all validations passed, proceed to Phase 3: /gsd:plan-phase 3
EOF

cat "$REPORT_PATH"
echo -e "${GREEN}✓ Summary report saved: $REPORT_PATH${NC}"
echo ""

# Step 9: Upload Results to GCS
echo -e "${YELLOW}[Step 9/10] Uploading results to GCS...${NC}"

if [ "$SKIP_UPLOAD" = "false" ]; then
    gsutil -m rsync -r outputs/ "gs://$GCS_BUCKET/outputs/"
    gsutil -m rsync -r data/processed/features_v3/ "gs://$GCS_BUCKET/features_v3/"
    gsutil cp "$REPORT_PATH" "gs://$GCS_BUCKET/reports/phase2_validation_summary.txt"
    echo -e "${GREEN}✓ Results uploaded to GCS${NC}"
else
    echo "Skipping upload (--skip-upload flag)"
fi
echo ""

# Step 10: Final Summary
echo -e "${YELLOW}[Step 10/10] Final summary${NC}"
echo ""
echo -e "${GREEN}======================================================"
echo "PHASE 2 VALIDATION COMPLETE"
echo -e "======================================================${NC}"
echo "Finished: $(date '+%Y-%m-%d %H:%M:%S')"
echo ""
echo "Results:"
echo "  Videos processed: $VIDEO_COUNT"
echo "  Poses extracted: $POSE_COUNT"
echo "  Validation: $([ $VALIDATION_EXIT -eq 0 ] && echo 'PASSED' || echo 'REVIEW NEEDED')"
echo ""
echo "Next steps:"
echo "  1. Review reports:"
echo "     - $REPORT_PATH"
echo "     - outputs/reports/feature_selection_report.md"
echo ""
echo "  2. If all validations passed, proceed to Phase 3:"
echo "     /gsd:plan-phase 3"
echo ""
