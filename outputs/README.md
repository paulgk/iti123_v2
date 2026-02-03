# Training Outputs

This directory contains model training outputs, reports, and visualizations.

## Directory Structure

```
outputs/
├── models/          # Trained model weights (.pth files)
├── reports/         # Training reports and metrics
├── visualizations/  # Training curves, confusion matrices, etc.
└── README.md        # This file
```

## What's Tracked in Git

### ✅ Included in Git
- Training reports (`reports/*.txt`, `*.md`, `*.csv`, `*.json`)
- Visualizations (`visualizations/*.png`, `*.jpg`, `*.pdf`)
- Best model checkpoints (`models/*_best.pth`)

### ❌ Excluded from Git
- Intermediate checkpoints
- Large model files (backed up to GCS)
- Training logs

## Models

Model weights are saved in `.pth` format (PyTorch).

**Naming convention:**
- `LSTM_best.pth` - Best LSTM model
- `STGCN_best.pth` - Best ST-GCN model
- `MSTCN_best.pth` - Best MS-TCN model

**Location in GCS:**
```
gs://iti123storage/models/deep_learning/
```

## Reports

Training reports contain:
- Model architecture details
- Training configuration (epochs, batch size, learning rate)
- Dataset split information
- Test accuracy and F1 scores
- Per-class performance metrics
- Training time and inference speed

**Key files:**
- `model_comparison.csv` - Comparison table of all models
- `model_comparison_summary.txt` - Executive summary
- `LSTM_report.txt` - Detailed LSTM results
- `STGCN_report.txt` - Detailed ST-GCN results
- `MSTCN_report.txt` - Detailed MS-TCN results

## Visualizations

Training visualizations include:
- `training_curves.png` - Training and validation curves (4-panel)
- `confusion_matrices.png` - Confusion matrices for all models (side-by-side)
- `per_class_performance.png` - Per-class metrics comparison

## Usage

### Save Outputs from Colab

After training in Colab, download outputs and save to git:

```bash
# Copy outputs from Colab download location
bash scripts/save_outputs_to_git.sh /path/to/downloaded/outputs

# Or if outputs are already in project directory
bash scripts/save_outputs_to_git.sh outputs
```

### Load Trained Model

```python
import torch
from models import STGCN  # or LSTM, MSTCN

# Load model
model = STGCN(num_classes=5)
checkpoint = torch.load('outputs/models/STGCN_best.pth')
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()

print(f"Model trained for {checkpoint['epoch']} epochs")
print(f"Best validation accuracy: {checkpoint['val_acc']:.2f}%")
```

### View Training Results

```bash
# View summary
cat outputs/reports/model_comparison_summary.txt

# View detailed report
cat outputs/reports/STGCN_report.txt

# Open visualizations
open outputs/visualizations/training_curves.png
```

## Backup to GCS

All outputs are automatically backed up to Google Cloud Storage:

```bash
# Upload from Colab (done automatically in notebook)
!gsutil -m rsync -r outputs/models/ gs://iti123storage/models/deep_learning/
!gsutil -m rsync -r outputs/reports/ gs://iti123storage/outputs/reports/deep_learning/
!gsutil -m rsync -r outputs/visualizations/ gs://iti123storage/outputs/visualizations/deep_learning/

# Download from GCS (if needed)
gsutil -m rsync -r gs://iti123storage/models/deep_learning/ outputs/models/
gsutil -m rsync -r gs://iti123storage/outputs/reports/deep_learning/ outputs/reports/
```

## Expected Results

After training with proper normalization:

| Model | Test Accuracy | F1 Score | Parameters | Inference (ms) |
|-------|--------------|----------|------------|----------------|
| LSTM | 75-82% | 0.75-0.82 | ~660K | 1-2ms |
| ST-GCN | 85-90% | 0.85-0.90 | ~820K | 15-20ms |
| MS-TCN | 82-88% | 0.82-0.88 | ~385K | <1ms |

**Best model:** ST-GCN (85-90% accuracy)

## Troubleshooting

### Low Accuracy (<40%)

If models show poor performance:
1. Check normalization in preprocessing (mean should be ~0.0)
2. Verify learning rate is 0.001 (not 0.0001)
3. Ensure sequences are filtered (>30 frames)
4. Check class weights are applied

### Missing Files

If files are missing:
1. Check GCS backup: `gsutil ls gs://iti123storage/models/deep_learning/`
2. Re-run training in Colab
3. Use save script to commit outputs

---

**Last updated:** 2026-02-03
**Status:** Production ready - Fixed normalization and learning rate
