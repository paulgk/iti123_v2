# Colab Session Management Guide

**Problem:** Colab sessions expire, but training takes 2-3 hours on L4 GPU

**Solutions implemented in the notebook:**

---

## ✅ Solution 1: Auto-Reconnect (Cell 3)

Prevents 90-minute idle timeout by auto-clicking reconnect button.

**How it works:**
- JavaScript checks connection every 60 seconds
- Auto-clicks reconnect if needed
- Keeps session alive during training

**Usage:** Just run cell 3 once before training starts.

---

## ✅ Solution 2: Checkpoint Saving (Cell 29)

Saves training progress every 5 epochs + automatic resume.

**What's saved:**
- Model weights
- Optimizer state
- Learning rate scheduler
- Training history
- Current epoch
- Best F1 score

**How it works:**
1. **Every 5 epochs:** Saves `checkpoint.pth`
2. **If disconnected:** Re-run notebook → automatically resumes from last checkpoint
3. **On completion:** Final checkpoint saved

**Resume after disconnection:**
```
Found checkpoint! Resuming training...
✓ Resumed from epoch 16
```

No manual intervention needed - just "Run All" again!

---

## Session Limits

### Free Tier
- **Idle timeout:** 90 minutes (prevented by auto-reconnect)
- **Max runtime:** 12 hours (enough for L4: 2-3 hours)
- **Daily limit:** ~12 hours/day

### Colab Pro ($10/month)
- **Max runtime:** 24 hours
- **Priority GPU access:** L4/A100
- **Background execution:** Yes

---

## Timeline on L4 GPU

| Stage | Time | Total |
|-------|------|-------|
| GCS Download | 5-10 min | 0:10 |
| Training (50 epochs) | 2-3 hours | 2:30 |
| Evaluation | 5 min | 2:35 |
| **Total** | **~2.5-3 hours** | ✓ |

**Verdict:** Well within 12-hour limit! ✅

---

## What Happens If Disconnected?

### Scenario 1: Browser Tab Closed
- **Auto-reconnect prevents this**
- Training continues running
- Can reopen tab and see progress

### Scenario 2: Internet Connection Lost
- Training stops
- Checkpoint saved (last one: epoch 15, 20, 25, etc.)
- **To resume:**
  1. Reconnect internet
  2. Re-run all cells
  3. Automatically resumes from checkpoint

### Scenario 3: Runtime Disconnected/Crashed
- Last checkpoint available (every 5 epochs)
- **To resume:**
  1. Runtime > Connect to a hosted runtime
  2. Re-run all cells
  3. Skips GCS download (data still there)
  4. Resumes training from checkpoint

---

## Best Practices

### Before Training:
1. ✅ Close unnecessary tabs (free up RAM)
2. ✅ Disable browser sleep mode
3. ✅ Keep Colab tab visible (prevents throttling)
4. ✅ Run cell 3 (auto-reconnect)

### During Training:
1. ✅ Check progress every ~30 minutes
2. ✅ Don't close browser tab
3. ✅ Watch for checkpoint messages every 5 epochs

### After Disconnection:
1. ✅ Just re-run all cells
2. ✅ Will automatically resume
3. ✅ No need to re-download data

---

## Monitoring Progress

### In Colab:
Training progress shows in real-time:
```
Epoch 12/50
Training: 100%|██████████| 198/198 [02:15<00:00]
Validation: 100%|██████████| 43/43 [00:18<00:00]

Train Loss: 0.6234 | Train Acc: 72.45%
Val Loss:   0.7012 | Val Acc:   68.23% | Val F1: 0.6543
Gap:        4.22%
✓ Saved best model (F1: 0.6543)
✓ Checkpoint saved (epoch 12)
```

### Remote Monitoring (Optional):
You can check status from phone/another device:
1. Open Colab notebook URL
2. See current epoch in output
3. Training continues even if you're away

---

## If Training Takes Longer Than Expected

### Option 1: Reduce Epochs
```python
'num_epochs': 30,  # Instead of 50
```
Early stopping will likely trigger before 30 anyway.

### Option 2: Increase Batch Size (if memory allows)
```python
'batch_size': 80,  # From 64
```
Fewer batches = faster epochs (but watch for OOM).

### Option 3: Train in Multiple Sessions
With checkpointing, you can split across sessions:
- Session 1: Train 20 epochs (1 hour)
- Session 2: Resume and train 20 more (1 hour)
- Session 3: Finish remaining epochs

---

## Files Persisted

**In Colab VM (temporary):**
- `/content/data/clips/` - Downloaded videos
- `/content/models/checkpoint.pth` - Resume checkpoint
- `/content/models/best_model.pth` - Best model

**Downloaded to local (permanent):**
- `best_model.pth` - Trained model
- `training_history.csv` - Metrics
- `training_curves.png` - Plots
- `confusion_matrix.png` - Results

---

## Troubleshooting

### "Runtime disconnected" immediately
**Cause:** GPU quota exhausted
**Fix:** Wait 1-2 hours or upgrade to Colab Pro

### Checkpoint not loading
**Cause:** Model architecture changed
**Fix:** Delete checkpoint and restart:
```python
!rm /content/models/checkpoint.pth
```

### Training slower than expected
**Cause:** Not using GPU
**Fix:** Runtime > Change runtime type > GPU

### Out of memory
**Cause:** Batch size too large
**Fix:** Reduce batch_size to 48 or 32

---

## Summary

**For L4 GPU (2-3 hour training):**
- ✅ Auto-reconnect prevents idle timeout
- ✅ Checkpoints every 5 epochs (crash recovery)
- ✅ Well within 12-hour free tier limit
- ✅ Automatic resume if disconnected
- ✅ No manual intervention needed

**Just run the notebook and walk away!** 🚀

---

**Last Updated:** 2026-02-04
