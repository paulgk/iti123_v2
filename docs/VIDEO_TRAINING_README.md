# Video-Based Training Guide

**Date:** 2026-02-04
**Status:** Ready for local training
**Expected Accuracy:** 70-80% (vs 38% with pose-only)

---

## Quick Start

### 1. Install Dependencies

```bash
pip install torch torchvision opencv-python numpy pandas scikit-learn tqdm
```

### 2. Train the Model

```bash
cd /Volumes/Ext/GenAI/iti123_v2/notebooks
python badminton_video_classification.py
```

**Expected training time:**
- **MPS (Mac M-series):** 6-8 hours
- **CUDA (NVIDIA GPU):** 4-6 hours
- **CPU:** 20-30 hours (not recommended)

### 3. Monitor Training

The script will show:
- Real-time progress bars for each epoch
- Training/validation loss and accuracy
- F1 score (macro average)
- Train-val gap (should be <10% for good generalization)

**Good training indicators:**
```
Epoch 10/50
Train Loss: 0.724 | Train Acc: 68.42%
Val Loss:   0.812 | Val Acc:   64.15% | Val F1: 0.6234
Gap:        4.27%
✓ Saved best model (F1: 0.6234)
```

### 4. Predict on New Videos

```bash
python predict_video.py --video path/to/video.mp4

# Example
python predict_video.py --video ../data/clips/Smash/match01_rally003_shot002.mp4
```

**Output:**
```
==================================================
Predicted Shot Type: Smash
==================================================

Probabilities:
  Clear   :  8.34% ████
  Drive   :  3.21% █
  Drop    : 12.45% ██████
  Lift    :  4.67% ██
  Smash   : 71.33% ███████████████████████████████████
==================================================
```

---

## Architecture Details

### Model: 2D CNN + LSTM

**Why this architecture?**
- **ResNet18 (pretrained):** Extracts spatial features from each frame
- **BiLSTM:** Captures temporal patterns across frames
- **Transfer learning:** Leverages ImageNet knowledge

**Architecture flow:**
```
Video (16 frames)
    ↓
ResNet18 CNN (per frame) → 512-dim features
    ↓
BiLSTM (temporal) → 256×2 hidden states
    ↓
Fully Connected → 5 classes
```

**Model size:**
- Total parameters: ~12.6M
- Trainable (LSTM + FC): ~1.2M (CNN frozen initially)
- Memory: ~2 GB GPU RAM at batch_size=16

---

## Dataset

### Structure
```
data/clips/
├── Clear/    (2,662 clips)
├── Drive/    (630 clips)
├── Drop/     (5,773 clips)
├── Lift/     (5,230 clips)
└── Smash/    (3,872 clips)
```

**Total:** 18,167 video clips

### Splits
- **Train:** 70% (12,717 clips)
- **Validation:** 15% (2,725 clips)
- **Test:** 15% (2,725 clips)

### Class Imbalance
- **Drop:** 31.8% (majority)
- **Lift:** 28.8%
- **Smash:** 21.3%
- **Clear:** 14.6%
- **Drive:** 3.5% (minority)

**Solution:** Focal Loss with softened class weights

---

## Configuration

Edit `badminton_video_classification.py` if needed:

```python
CONFIG = {
    # Data
    'num_frames': 16,              # Frames sampled per clip
    'frame_size': (224, 224),      # ResNet input size

    # Training
    'batch_size': 16,              # Reduce to 8 if GPU OOM
    'num_epochs': 50,
    'learning_rate': 0.0001,

    # Model
    'lstm_hidden_size': 256,
    'lstm_num_layers': 2,
    'lstm_dropout': 0.5,
    'freeze_cnn': True,            # Freeze CNN initially

    # Loss
    'use_focal_loss': True,
    'focal_gamma': 2.0,
}
```

**Tuning tips:**
- **GPU OOM:** Reduce `batch_size` to 8 or `num_frames` to 12
- **Underfitting (train acc <60%):** Set `freeze_cnn=False` to fine-tune CNN
- **Overfitting (gap >15%):** Increase `lstm_dropout` to 0.6
- **Drive class poor (<40%):** Increase `focal_gamma` to 2.5

---

## Expected Results

### Phase 1: Frozen CNN (First 20-30 epochs)

**Expected metrics:**
- **Train accuracy:** 65-75%
- **Val accuracy:** 60-70%
- **Val F1:** 0.58-0.68
- **Gap:** 5-10%

**Per-class accuracy (realistic):**
- Drop: 75-82%
- Lift: 72-78%
- Smash: 68-75%
- Clear: 60-68%
- Drive: 45-60%

### Phase 2: Fine-tuned CNN (Optional)

If Phase 1 results are good (>65% val acc), you can fine-tune:

```python
# After epoch 30, manually unfreeze CNN
CONFIG['freeze_cnn'] = False  # Set before training
# OR
model.unfreeze_cnn()  # Call during training
optimizer = optim.Adam(model.parameters(), lr=0.00001)  # Lower LR!
```

**Expected improvement:**
- **Val accuracy:** +3-5% boost
- **Final target:** 70-80%

---

## Output Files

After training, check `models/video_classification/`:

```
models/video_classification/
├── best_model.pth              # Best model checkpoint (by F1 score)
├── config.json                 # Training configuration
└── training_history.csv        # Loss/accuracy per epoch
```

### Analyze Training Curves

```python
import pandas as pd
import matplotlib.pyplot as plt

history = pd.read_csv('models/video_classification/training_history.csv')

plt.figure(figsize=(12, 4))

# Loss
plt.subplot(1, 3, 1)
plt.plot(history['train_loss'], label='Train')
plt.plot(history['val_loss'], label='Val')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.title('Loss Curves')

# Accuracy
plt.subplot(1, 3, 2)
plt.plot(history['train_acc'], label='Train')
plt.plot(history['val_acc'], label='Val')
plt.xlabel('Epoch')
plt.ylabel('Accuracy (%)')
plt.legend()
plt.title('Accuracy Curves')

# F1 Score
plt.subplot(1, 3, 3)
plt.plot(history['val_f1'])
plt.xlabel('Epoch')
plt.ylabel('F1 Score (Macro)')
plt.title('Validation F1')

plt.tight_layout()
plt.savefig('training_curves.png', dpi=150)
plt.show()
```

---

## Troubleshooting

### GPU Out of Memory

```python
# Reduce batch size
CONFIG['batch_size'] = 8  # or 4

# OR reduce frames
CONFIG['num_frames'] = 12
```

### MPS (Mac) Compatibility Issues

```python
# If MPS fails, force CPU
CONFIG['device'] = 'cpu'
CONFIG['num_workers'] = 0  # Disable multiprocessing on CPU
```

### Video Loading Errors

```bash
# Check corrupted videos
python -c "
import cv2
from pathlib import Path

for video in Path('data/clips').rglob('*.mp4'):
    cap = cv2.VideoCapture(str(video))
    frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if frames == 0:
        print(f'Corrupted: {video}')
    cap.release()
"
```

### Training Not Converging

**Symptoms:** Train/val accuracy stuck at ~32% (near random)

**Possible causes:**
1. **Learning rate too low:** Increase to 0.0005
2. **CNN frozen too long:** Unfreeze after 10 epochs
3. **Data loading issue:** Check video paths are correct
4. **Model not learning:** Check gradients are flowing

**Debug:**
```python
# Check if model is learning
print("Checking gradients...")
for name, param in model.named_parameters():
    if param.grad is not None:
        print(f"{name}: grad_norm={param.grad.norm().item():.4f}")
    else:
        print(f"{name}: NO GRADIENT")
```

---

## Comparison with Pose-Based Approach

| Metric | Pose-Only | Video-Based (Expected) |
|--------|-----------|------------------------|
| Overall Accuracy | 38.1% | 70-80% |
| Train-Val Gap | 0.4% | 5-10% |
| Drop Accuracy | 53% | 75-82% |
| Drive Accuracy | ~20% | 45-60% |
| F1 (Macro) | 0.34 | 0.65-0.75 |
| Training Time | 2 hrs | 6-8 hrs |

**Key insight:** Video captures shuttle trajectory implicitly through frame sequence, which pose data cannot.

---

## Next Steps

### After Training Completes

1. **Check test set results:**
   - Should be printed automatically
   - Look for confusion matrix patterns

2. **Analyze errors:**
   - Which classes are confused?
   - Is Drive still problematic?

3. **If results good (>70%):**
   - Deploy model for inference
   - Consider ensemble with pose model (vote or stack)

4. **If results poor (<65%):**
   - Try unfreezing CNN (fine-tune)
   - Increase num_frames to 24
   - Try X3D model (see VIDEO_BASED_APPROACH.md)

---

## File Reference

- **Training script:** `notebooks/badminton_video_classification.py`
- **Inference script:** `notebooks/predict_video.py`
- **Full guide:** `docs/VIDEO_BASED_APPROACH.md`
- **Dataset:** `data/clips/{Clear,Drive,Drop,Lift,Smash}/*.mp4`

---

**Last Updated:** 2026-02-04
**Status:** Ready to train
**Expected Completion:** 6-8 hours on MPS/CUDA
