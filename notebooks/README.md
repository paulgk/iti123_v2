# Notebooks

Jupyter notebooks for Phase 2 validation and analysis.

## Phase 2 Validation

### [phase2_validation_colab.ipynb](phase2_validation_colab.ipynb)

**Purpose:** Complete Phase 2 validation workflow in Colab Enterprise

**Duration:** ~1-2 hours (after pose extraction completes)

**Prerequisites:**
- Pose extraction completed (~10,000+ samples)
- `data/processed/poses/*.pkl` files exist
- `data/metadata.csv` created

**What it validates:**
1. ✅ Phase segmentation boundary accuracy (≥85%)
2. ✅ Kinetic chain effect sizes (Cohen's d > 0.5)
3. ✅ Feature selection pipeline (<254 features)
4. ✅ V2 backward compatibility

**Sections:**
- **Setup**: Verify environment and extraction results (5 min)
- **Step 1**: Verify extraction results (5 min)
- **Step 2**: Run validation suite (15-30 min)
- **Step 3**: Run feature selection pipeline (30-60 min)
- **Step 4**: Test v3 feature extraction (5 min)
- **Step 5**: Upload results to GCS (10 min)
- **Step 6**: Generate phase 2 summary (2 min)

**Usage in Colab:**

```bash
# After pose extraction completes
cd /content/iti123_v2
source colab_venv/bin/activate

# Open notebook in Colab
# Upload to Colab or open from GitHub
```

**Expected outputs:**
- `data/processed/features_v3/selected_features.json` - Feature selection manifest
- `outputs/reports/feature_selection_report.md` - Detailed selection report
- `outputs/reports/phase2_validation_summary.txt` - Phase 2 summary
- All results backed up to GCS

---

## Other Notebooks

### [phase2_validation_workflow.md](phase2_validation_workflow.md)

Markdown version of the workflow with command-line instructions.

---

## Quick Start

1. **Extract poses** (in Colab terminal):
   ```bash
   python scripts/extract_poses_parallel.py \
       --video-dir data/videos/ \
       --output-dir data/processed/poses/ \
       --model-complexity 1 \
       --target-fps 20 \
       --num-workers 4
   ```

2. **Run validation notebook** (in Colab):
   - Open `phase2_validation_colab.ipynb`
   - Run all cells sequentially
   - Check for ✅ checkpoints after each step

3. **Review results**:
   - Feature selection report: `outputs/reports/feature_selection_report.md`
   - Phase 2 summary: `outputs/reports/phase2_validation_summary.txt`
   - Feature manifest: `data/processed/features_v3/selected_features.json`

4. **Proceed to Phase 3**:
   ```bash
   /gsd:plan-phase 3
   ```

---

## Troubleshooting

**"No pose files found"**
- Check extraction completed: `ls data/processed/poses/*.pkl | wc -l`
- Re-run extraction if needed

**"Not enough samples"**
- Feature selection needs ≥50 samples
- For robust results, aim for 1,000+ samples

**"Validation failed"**
- Check validation output for specific failures
- Review `outputs/reports/` for detailed error messages
- Common issues:
  - Low boundary accuracy: Check pose extraction quality
  - Low effect sizes: Check stroke type labels in metadata
  - Feature count >254: Adjust target in Step 3

**"Out of memory"**
- Use High-RAM runtime in Colab
- Reduce sample size in validation steps
- Process feature selection in batches

---

## File Structure

```
notebooks/
├── README.md                           # This file
├── phase2_validation_colab.ipynb       # Main validation notebook
└── phase2_validation_workflow.md       # Command-line workflow
```

---

## Support

For issues or questions:
- Check [COLAB_QUICKSTART.md](../COLAB_QUICKSTART.md) for common issues
- Review [scripts/README.md](../scripts/README.md) for script documentation
- See [.planning/phases/02-feature-engineering-enhancement/02-VERIFICATION.md](../.planning/phases/02-feature-engineering-enhancement/02-VERIFICATION.md) for detailed validation requirements
