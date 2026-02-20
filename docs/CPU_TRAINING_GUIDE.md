# CPU Training Guide: Resource Control & Safe Resume

## Overview

The CPU training script (`badminton_training_cpu_local.py`) includes features to:
1. **Control CPU usage** to prevent system overload
2. **Gracefully handle interrupts** (Ctrl+C)
3. **Automatically save checkpoints** before exiting
4. **Resume training** from any checkpoint

---

## CPU Usage Control

### Problem: CPU Overloading
When training deep learning models on CPU, PyTorch can consume 100% CPU, causing:
- System slowdown/freezing
- Inability to use other applications
- Overheating on laptops
- Battery drain

### Solution: Thread Limiting

The script limits CPU threads based on a percentage:

```bash
# Use maximum 80% of CPU cores (default)
python badminton_training_cpu_local.py

# Use maximum 60% of CPU cores (lighter load)
python badminton_training_cpu_local.py --max-cpu-percent 60

# Use maximum 90% of CPU cores (faster training)
python badminton_training_cpu_local.py --max-cpu-percent 90
```

### How It Works

```python
# Calculate thread count from percentage
max_threads = int(os.cpu_count() * max_cpu_percent / 100)
torch.set_num_threads(max_threads)
```

**Example on 8-core CPU:**
- `--max-cpu-percent 80` → 6 threads (75% actual usage)
- `--max-cpu-percent 60` → 4 threads (50% actual usage)
- `--max-cpu-percent 100` → 8 threads (100% usage)

### Monitoring CPU Usage

**macOS:**
```bash
# Terminal
top -pid $(pgrep -f badminton_training)

# Activity Monitor (GUI)
open -a "Activity Monitor"
```

**Linux:**
```bash
htop -p $(pgrep -f badminton_training)
```

### Recommended Settings

| Use Case | CPU % | Threads (8-core) | Training Time |
|----------|-------|------------------|---------------|
| **Background training** | 60% | 4-5 threads | 20-30 hours |
| **Balanced** (default) | 80% | 6-7 threads | 12-18 hours |
| **Maximum speed** | 100% | 8 threads | 10-15 hours |
| **Laptop battery** | 40% | 3 threads | 30-40 hours |

---

## Graceful Interrupt Handling

### The Problem with Ctrl+C

**Without interrupt handling:**
```
[Training in progress...]
^C  # User presses Ctrl+C
Traceback (most recent call last):
  ...
KeyboardInterrupt
# Training stopped, no checkpoint saved
# All progress since last checkpoint (up to 5 epochs) is LOST
```

**With interrupt handling:**
```
[Training in progress...]
^C  # User presses Ctrl+C

======================================================================
🛑 INTERRUPT SIGNAL RECEIVED (Ctrl+C)
======================================================================
Finishing current batch and saving checkpoint...
Press Ctrl+C again to force quit (NOT RECOMMENDED)
======================================================================

⚠️  Training interrupted by user

======================================================================
SAVING CHECKPOINT BEFORE EXIT
======================================================================
✓ Checkpoint saved to: outputs/results_cpu_local/checkpoint.pth
  Epoch: 23/50
  Best val acc: 72.45%
  To resume: python badminton_training_cpu_local.py --resume checkpoint.pth
======================================================================
```

### How Graceful Interrupts Work

1. **First Ctrl+C**: Signals interrupt handler
2. **Finish current batch**: Completes forward/backward pass
3. **Save checkpoint**: Preserves all training state
4. **Clean exit**: No data loss

**Second Ctrl+C** (if pressed immediately): Forces immediate quit (may lose current batch)

### What Gets Saved in Checkpoint

```python
{
    'epoch': 23,                          # Last completed epoch
    'model_state_dict': <model weights>,  # Full model state
    'optimizer_state_dict': <optimizer>,  # Optimizer state (momentum, etc.)
    'scheduler_state_dict': <scheduler>,  # Learning rate scheduler
    'best_val_acc': 72.45,                # Best validation accuracy
    'history': {                          # Full training history
        'train_loss': [...],
        'train_acc': [...],
        'val_loss': [...],
        'val_acc': [...]
    },
    'patience_counter': 3,                # Early stopping counter
    'interrupted': True                   # Flag indicating manual stop
}
```

---

## Resuming Training

### Automatic Resume

If `checkpoint.pth` exists in the output directory, training automatically resumes:

```bash
# First run
python badminton_training_cpu_local.py
# ... trains epochs 0-22, then you press Ctrl+C

# Second run (automatically finds checkpoint)
python badminton_training_cpu_local.py
# Output:
# Found checkpoint: outputs/results_cpu_local/checkpoint.pth
# Resuming training...
#   Resuming from epoch 23
#   Best val acc so far: 72.45%
```

### Manual Resume from Specific Checkpoint

```bash
# Resume from specific checkpoint file
python badminton_training_cpu_local.py --resume /path/to/checkpoint.pth
```

### Resume After Crash/Power Loss

The script saves checkpoints:
1. **Every 5 epochs** (configurable via `save_every`)
2. **After each epoch** (automatic, overwrites checkpoint.pth)
3. **On Ctrl+C** (graceful interrupt)

**Scenarios:**

| Event | Max Data Loss | Resume Command |
|-------|---------------|----------------|
| **Ctrl+C interrupt** | 0 batches (graceful save) | `python badminton_training_cpu_local.py` |
| **Power loss** | 1 epoch max | `python badminton_training_cpu_local.py` |
| **System crash** | 1 epoch max | `python badminton_training_cpu_local.py` |
| **Out of memory** | 1 batch (saves after each epoch) | `python badminton_training_cpu_local.py` |

---

## Complete Usage Examples

### 1. Standard Training (80% CPU)

```bash
python badminton_training_cpu_local.py
```

**Expected output:**
```
======================================================================
Badminton Shot Classification - CPU Training
======================================================================
Data directory: /Volumes/Ext/GenAI/iti123_v2/data/frames
Results directory: /Volumes/Ext/GenAI/iti123_v2/outputs/results_cpu_local
Device: CPU (no GPU)
Batch size: 4
CPU threads: 6 / 8 (limit: 80%)
Skip ratio: 0% (focus on last 100% of frames)
Expected time: 12-24 hours
======================================================================
💡 Press Ctrl+C anytime to save checkpoint and exit gracefully
======================================================================

Starting Training
...
```

### 2. Lightweight Training (60% CPU)

For running in background while using your computer:

```bash
python badminton_training_cpu_local.py --max-cpu-percent 60
```

### 3. Training with Targeted Frame Sampling

Skip first 30% of frames (pre-shot preparation):

```bash
python badminton_training_cpu_local.py --skip-ratio 0.3
```

**Output shows:**
```
Skip ratio: 30% (focus on last 70% of frames)
```

### 4. Combined: Light CPU + Targeted Sampling

```bash
python badminton_training_cpu_local.py --max-cpu-percent 60 --skip-ratio 0.3
```

### 5. Resume After Interrupt

```bash
# First session
python badminton_training_cpu_local.py
# ... trains 0-22 epochs, press Ctrl+C

# Second session (automatic resume)
python badminton_training_cpu_local.py
# ... continues from epoch 23

# OR manual resume
python badminton_training_cpu_local.py --resume outputs/results_cpu_local/checkpoint.pth
```

---

## Training Progress Tracking

### During Training

```
Epoch 23/50
----------------------------------------------------------------------
Training: 100%|███████████| 3903/3903 [12:34<00:00, loss: 0.8234, acc: 68.45%]
Validation: 100%|█████████| 552/552 [01:23<00:00]

Epoch 23 Summary:
  Train Loss: 0.8234 | Train Acc: 68.45%
  Val Loss:   0.9123 | Val Acc:   65.32%
  Gap: 3.13%
  LR: 0.000100
  Time: 13.9 minutes
  Est. remaining: 6.3 hours
  No improvement (3/10)
  ✓ Checkpoint saved
```

### Key Metrics Explained

- **Train/Val Acc**: Accuracy on training/validation sets
- **Gap**: Overfitting indicator (train_acc - val_acc)
  - Gap < 5%: Good generalization
  - Gap 5-10%: Acceptable
  - Gap > 10%: Overfitting
- **LR**: Current learning rate (decreases when plateau)
- **Est. remaining**: Estimated hours until completion
- **No improvement (X/Y)**: Early stopping counter

---

## Troubleshooting

### Problem: Script Uses 100% CPU Despite Limit

**Cause**: Thread limit applies to PyTorch, but data loading may use additional threads.

**Solution**: Reduce num_workers (already set to 1 in script)

### Problem: Training Too Slow

**Symptoms:**
- > 30 minutes per epoch
- Est. remaining > 30 hours

**Solutions:**
1. Increase CPU limit:
   ```bash
   python badminton_training_cpu_local.py --max-cpu-percent 90
   ```

2. Use smaller dataset (for testing):
   ```python
   # Edit script: reduce samples
   train_paths = train_paths[:1000]  # Use only 1000 samples
   ```

3. Reduce batch size (if memory-bound):
   ```python
   CONFIG['batch_size'] = 2  # Instead of 4
   ```

### Problem: Out of Memory

**Symptoms:**
```
RuntimeError: [enforce fail at CPUAllocator.cpp:64] . DefaultCPUAllocator: can't allocate memory
```

**Solutions:**
1. Reduce batch size:
   ```python
   CONFIG['batch_size'] = 2  # Or even 1
   ```

2. Close other applications
3. Use targeted sampling to load fewer frames:
   ```bash
   python badminton_training_cpu_local.py --skip-ratio 0.3
   ```

### Problem: Checkpoint Not Found After Resume

**Cause**: Checkpoint saved in different directory

**Solution**: Check checkpoint location:
```bash
ls -lh outputs/results_cpu_local/checkpoint.pth

# If in different location, use --resume
python badminton_training_cpu_local.py --resume /path/to/checkpoint.pth
```

### Problem: Training Stops at Epoch 0 with Error

**Cause**: Data files not found

**Solution**: Verify data path:
```bash
# Check if .npy files exist
ls -lh /Volumes/Ext/GenAI/iti123_v2/data/frames/*.npy | head -5

# Edit script if path is different
DATA_ROOT = "/correct/path/to/frames"
```

---

## Best Practices

### 1. Monitor First Epoch

Don't start a 24-hour training session without monitoring the first epoch:

```bash
python badminton_training_cpu_local.py
# Wait for first epoch to complete (~15-30 minutes)
# Check metrics look reasonable
# Then let it run overnight
```

### 2. Use Screen/Tmux for Long Sessions

Keep training running even if you close terminal:

```bash
# Install screen (if not installed)
brew install screen  # macOS
sudo apt install screen  # Linux

# Start screen session
screen -S training

# Run training
python badminton_training_cpu_local.py --max-cpu-percent 70

# Detach: Press Ctrl+A then D
# Training continues in background

# Reattach later
screen -r training

# List sessions
screen -ls
```

### 3. Log Output to File

```bash
python badminton_training_cpu_local.py 2>&1 | tee training.log

# View log later
tail -f training.log
```

### 4. Periodic Checkpoints Strategy

For very long training (>24 hours), manually save checkpoints:

```bash
# Copy checkpoint every 10 epochs
cp outputs/results_cpu_local/checkpoint.pth \
   outputs/results_cpu_local/checkpoint_epoch10.pth
```

### 5. Test on Small Subset First

Before full training, test on 1000 samples:

```python
# Edit script temporarily
train_paths = train_paths[:1000]
val_paths = val_paths[:200]
test_paths = test_paths[:400]
```

Run 2-3 epochs to verify:
- Script works
- No errors
- Reasonable speed (~5-10 min/epoch)

Then remove limits and run full training.

---

## Performance Expectations

### Hardware vs Training Time

| Hardware | Threads | Time/Epoch | Total Time (50 epochs) |
|----------|---------|------------|------------------------|
| **M1 Mac (8 cores)** | 6 (80%) | 15 min | 12.5 hours |
| **Intel i7 (8 cores)** | 6 (80%) | 20 min | 16.7 hours |
| **Intel i5 (6 cores)** | 4 (80%) | 25 min | 20.8 hours |
| **Old dual-core** | 1 (80%) | 60 min | 50 hours |

### With Targeted Sampling (--skip-ratio 0.3)

Reduces data loading time slightly:
- **Speed improvement**: ~5-10% faster per epoch
- **M1 Mac**: 15 min → 13-14 min per epoch

### Early Stopping Impact

Training typically stops at 30-40 epochs (not full 50):
- **Expected completion**: 10-15 hours (instead of 20+ hours)

---

## Summary

### Key Commands

```bash
# Basic training (80% CPU)
python badminton_training_cpu_local.py

# Light load (60% CPU)
python badminton_training_cpu_local.py --max-cpu-percent 60

# With targeted sampling
python badminton_training_cpu_local.py --skip-ratio 0.3

# Resume after interrupt
python badminton_training_cpu_local.py  # Automatic
python badminton_training_cpu_local.py --resume checkpoint.pth  # Manual
```

### Safety Features

✅ **Graceful Ctrl+C handling** - No data loss on interrupt
✅ **Automatic checkpoints** - Every epoch + every 5 epochs
✅ **CPU throttling** - Prevent system overload
✅ **Resume capability** - Continue from any checkpoint
✅ **Progress tracking** - Time estimates and history

### When Training is Interrupted

1. **You press Ctrl+C**: Saves checkpoint immediately, no data loss
2. **Power loss**: Loses current epoch only (max 15-20 minutes)
3. **System crash**: Loses current epoch only
4. **Out of memory**: Saves checkpoint after each epoch

**In all cases**: Just run the script again, it will resume automatically!

---

## Next Steps

1. **Start training:**
   ```bash
   python badminton_training_cpu_local.py --max-cpu-percent 70
   ```

2. **Monitor first epoch** (~15-30 minutes)

3. **If satisfied, let it run overnight**

4. **Check results in:**
   ```
   outputs/results_cpu_local/
   ├── best_model.pth          # Best model weights
   ├── checkpoint.pth          # Resume checkpoint
   ├── results_summary.json    # Final results
   ├── classification_report.txt
   └── confusion_matrix.png
   ```

Good luck with your training! 🚀
