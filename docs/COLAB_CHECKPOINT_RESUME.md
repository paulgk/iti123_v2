# Colab Training: Checkpoint & Resume Guide

## What's New

The Colab notebook now includes **automatic checkpoint saving** and **resume capability**!

### Benefits:
- ✅ **No data loss** if Colab disconnects
- ✅ **Automatic resume** from last epoch
- ✅ **Saves every epoch** (not just every 5)
- ✅ **Preserves training history** (loss/accuracy curves)
- ✅ **Tracks early stopping** progress

---

## How It Works

### Automatic Checkpoint Saving

After **every epoch**, the notebook saves:
- Current model weights
- Optimizer state (momentum, etc.)
- Learning rate scheduler state
- Best validation accuracy
- Complete training history
- Early stopping counter

**File saved:** `/content/results/checkpoint.pth`

### Automatic Resume

If training is interrupted and you re-run the training cell:
1. Notebook detects existing `checkpoint.pth`
2. Loads all saved state
3. Continues from the next epoch
4. Preserves all progress!

---

## Usage Example

### Scenario 1: Training Interrupted

**First run:**
```
Starting Training
======================================================================
Model: ResNet18_BiLSTM
Epochs: 0 to 100
...

Epoch 1/100
----------------------------------------------------------------------
Training: 100%|███████| ...
Validation: 100%|█████| ...

Epoch 1 Summary:
  Train Loss: 1.2345 | Train Acc: 45.67%
  Val Loss:   1.4567 | Val Acc:   42.34%
  ✓ Saved best model (val_acc: 42.34%)
  ✓ Checkpoint saved (can resume if interrupted)

Epoch 2/100
...
[Colab disconnects at epoch 15]
```

**Second run (same cell):**
```
======================================================================
RESUMING FROM CHECKPOINT
======================================================================
Resuming from epoch 16
Best val acc so far: 58.45%
Patience counter: 3/15
======================================================================

Starting Training
======================================================================
Model: ResNet18_BiLSTM
Epochs: 16 to 100
...

Epoch 16/100  ← Continues from where it stopped!
----------------------------------------------------------------------
```

### Scenario 2: Intentional Stop & Resume

You can stop training anytime and resume later:

**Stop training:**
- Click the stop button (⏹️) in Colab
- Or press Interrupt Runtime

**Resume later:**
- Just re-run the training cell
- It automatically continues from last epoch!

---

## Files Saved

| File | Purpose | When Saved |
| --- | --- | --- |
| `checkpoint.pth` | Resume state | Every epoch |
| `best_model.pth` | Best weights | When val accuracy improves |
| `results_summary.json` | Final results | After training complete |

**Both files work together:**
- `checkpoint.pth` → For resuming training
- `best_model.pth` → For evaluation/inference

---

## What Gets Preserved

When you resume, **everything** is preserved:

### ✅ Model State
- All weights and biases
- Exact model architecture

### ✅ Optimizer State
- Adam momentum buffers
- Learning rate
- Weight decay history

### ✅ Training Progress
- Current epoch number
- Best validation accuracy
- Training history (all epochs)
  - Train loss per epoch
  - Train accuracy per epoch
  - Val loss per epoch
  - Val accuracy per epoch

### ✅ Early Stopping
- Patience counter
- Epochs since last improvement

### ✅ Learning Rate Scheduler
- ReduceLROnPlateau state
- LR reduction history

---

## Important Notes

### 1. Don't Delete `checkpoint.pth`

If you want to start fresh training:
```python
# Add this cell BEFORE training cell
import os
checkpoint_path = f"{RESULTS_DIR}/checkpoint.pth"
if os.path.exists(checkpoint_path):
    os.remove(checkpoint_path)
    print("✓ Removed old checkpoint - will start fresh")
```

### 2. Checkpoint Saves Every Epoch

Unlike the old version (saved every 5 epochs), now it saves **after every single epoch**.

**Why?** To minimize data loss if Colab disconnects.

**Disk space:** Each checkpoint is ~60-170 MB depending on model.

### 3. Resume is Automatic

You don't need to do anything special:
- Just re-run the training cell
- It detects and loads `checkpoint.pth` automatically

### 4. Training History is Complete

The training history includes ALL epochs:
```python
history['train_loss']  # Contains losses from epoch 0 to current
history['train_acc']   # Contains accuracies from epoch 0 to current
```

So your plots will show the complete curve, not just the resumed portion!

---

## Troubleshooting

### Problem: "RESUMING FROM CHECKPOINT" but starts from wrong epoch

**Cause:** Old checkpoint from previous training run

**Solution:**
```python
# Check checkpoint info
import torch
checkpoint = torch.load(f"{RESULTS_DIR}/checkpoint.pth")
print(f"Checkpoint epoch: {checkpoint['epoch']}")
print(f"Best val acc: {checkpoint['best_val_acc']}")

# If wrong, delete it
import os
os.remove(f"{RESULTS_DIR}/checkpoint.pth")
```

### Problem: Want to start completely fresh training

**Solution:** Delete both checkpoint and best model
```python
import os
for file in ['checkpoint.pth', 'best_model.pth']:
    path = f"{RESULTS_DIR}/{file}"
    if os.path.exists(path):
        os.remove(path)
        print(f"Deleted {file}")
```

### Problem: Checkpoint file is corrupted

**Solution:** Delete it and restart
```python
import os
try:
    checkpoint = torch.load(f"{RESULTS_DIR}/checkpoint.pth")
    print("Checkpoint is valid")
except Exception as e:
    print(f"Checkpoint corrupted: {e}")
    os.remove(f"{RESULTS_DIR}/checkpoint.pth")
    print("Deleted corrupted checkpoint")
```

---

## Comparison: Old vs New

| Feature | Old Version | New Version |
| --- | --- | --- |
| **Checkpoint frequency** | Every 5 epochs | Every epoch |
| **Resume capability** | ❌ No | ✅ Yes |
| **Max data loss** | Up to 5 epochs | 1 epoch max |
| **History preservation** | ❌ Lost | ✅ Complete |
| **Early stopping state** | ❌ Lost | ✅ Preserved |
| **Manual intervention** | Required | Automatic |

---

## Best Practices

### 1. Upload Checkpoints to GCS Periodically

For very long training (>50 epochs), back up to GCS:

```python
# Add this as a new cell after training cell
# Run every 10-20 epochs

import os
if os.path.exists(f"{RESULTS_DIR}/checkpoint.pth"):
    !gsutil cp {RESULTS_DIR}/checkpoint.pth {GCS_BUCKET}/checkpoints/checkpoint_epoch{epoch}.pth
    print(f"✓ Backed up checkpoint to GCS")
```

### 2. Check Progress Before Resuming

```python
# Add this cell BEFORE resuming
import torch
import os

checkpoint_path = f"{RESULTS_DIR}/checkpoint.pth"
if os.path.exists(checkpoint_path):
    checkpoint = torch.load(checkpoint_path)
    print(f"Checkpoint found!")
    print(f"  Epoch: {checkpoint['epoch'] + 1}")
    print(f"  Best val acc: {checkpoint['best_val_acc']:.2f}%")
    print(f"  Patience: {checkpoint['patience_counter']}/{TRAIN_CONFIG['early_stopping_patience']}")
    print(f"  Total epochs so far: {len(checkpoint['history']['train_loss'])}")
else:
    print("No checkpoint found - will start fresh")
```

### 3. Save Final Model to GCS

After training completes:

```python
# Upload final results
!gsutil -m cp -r {RESULTS_DIR} {GCS_BUCKET}/outputs/
print("✓ Results backed up to GCS")
```

---

## Summary

### Key Changes:
1. ✅ Checkpoint saves **every epoch** (not every 5)
2. ✅ **Automatic resume** if re-run
3. ✅ **Complete history** preserved
4. ✅ **No data loss** on disconnect

### How to Use:
1. Run training cell normally
2. If interrupted: just re-run the same cell
3. It automatically resumes from last epoch!

### Files:
- `checkpoint.pth` → For resuming (saved every epoch)
- `best_model.pth` → Best weights (saved when improves)

**Your training is now Colab-disconnect-proof!** 🎉
