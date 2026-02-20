# CPU Training Time Estimates

## Your System Performance

Based on actual measurements on your system:

**⏱️ 1 epoch = ~3 hours**

---

## Expected Training Time

### Full Training (50 epochs)

| Scenario | Epochs | Time per Epoch | Total Time |
|----------|--------|----------------|------------|
| **No early stopping** | 50 | 3 hours | **150 hours** (~6.3 days) |
| **With early stopping** | 30-40 | 3 hours | **90-120 hours** (~3.75-5 days) |
| **Likely completion** | ~35 | 3 hours | **~105 hours** (~4.4 days) |

### Recommended Strategy

Given the long training time, here are some strategies:

---

## Strategy 1: Reduce Epochs (Recommended)

Reduce `num_epochs` to get faster results:

```python
# Edit badminton_training_cpu_local.py, line 65
CONFIG = {
    ...
    'num_epochs': 30,  # Instead of 50
    ...
}
```

**Expected time:** 30 epochs × 3 hours = **90 hours** (~3.75 days)

With early stopping (patience=10), likely stops around epoch 25:
- **~75 hours** (~3.1 days)

---

## Strategy 2: Run Overnight/Weekend Sessions

**Overnight sessions:**
- 8 hours sleep = ~2.7 epochs
- 5 nights = ~13 epochs
- Need ~13 overnight sessions for 35 epochs

**Weekend session:**
- 48 hours = ~16 epochs
- 2-3 weekends needed

**Combined approach (Recommended):**
1. Start Friday evening
2. Run all weekend (48 hours = 16 epochs)
3. Continue 3-4 overnight sessions (24-32 hours = 8-11 more epochs)
4. Another weekend if needed
5. **Total: 8-10 days real time**

---

## Strategy 3: Reduce Dataset Size (For Testing)

Test with a smaller dataset first to verify everything works:

```python
# Edit badminton_training_cpu_local.py around line 296
print(f"Loaded {len(npy_paths)} samples")

# ADD THESE LINES for testing:
npy_paths = npy_paths[:5000]  # Use only 5000 samples instead of ~22k
labels = labels[:5000]
print(f"REDUCED to {len(npy_paths)} samples for testing")
```

**Expected speedup:** ~2-3x faster
- 1 epoch: 3 hours → **~1-1.5 hours**
- 30 epochs: **30-45 hours** (1.25-1.9 days)

Once satisfied, remove the reduction and run full training.

---

## Strategy 4: Use Screen/Tmux for Long Sessions

Keep training running even if you disconnect:

```bash
# Install screen (if needed)
brew install screen  # macOS

# Start screen session
screen -S training

# Activate conda and start training
conda activate iti123
python notebooks/badminton_training_cpu_local.py --max-cpu-percent 70

# Detach: Press Ctrl+A then D
# Training continues in background!

# Reattach later to check progress
screen -r training

# Check if still running
screen -ls
```

**Benefits:**
- Close laptop, training continues
- Survives terminal disconnects
- Can check progress anytime

---

## Strategy 5: Reduce Batch Size (If Memory-Limited)

If you're hitting memory limits, reduce batch size:

```python
# Edit line 64
CONFIG = {
    ...
    'batch_size': 2,  # Instead of 4
    ...
}
```

**Note:** This will make training **slower** (~1.5-2x), so only use if necessary.

---

## Progress Monitoring

### Check Progress Without Interrupting

```bash
# In another terminal, check the latest output
tail -f outputs/results_cpu_local/training_history.csv

# Or check checkpoint
ls -lh outputs/results_cpu_local/checkpoint.pth
```

### Estimate Remaining Time

The script shows estimates after each epoch:

```
Epoch 12/50
----------------------------------------------------------------------
...
Epoch 12 Summary:
  Train Loss: 0.8234 | Train Acc: 68.45%
  Val Loss:   0.9123 | Val Acc:   65.32%
  Gap: 3.13%
  LR: 0.000100
  Time: 3.1 hours  ← Actual time per epoch
  Est. remaining: 114 hours  ← Estimated time left
```

**After a few epochs, you'll see:**
- Actual time per epoch on your system
- Estimated remaining time
- Whether it's improving or plateauing

---

## Recommended Training Plan

### Phase 1: Quick Test (1 day)
```bash
# Reduce epochs to 10 and use small dataset
python notebooks/badminton_training_cpu_local.py
# Expected: 30 hours with reduced dataset
```

**Goal:** Verify everything works, no errors

### Phase 2: Full Training (4-5 days)
```bash
# Remove dataset reduction, run with num_epochs=35
# Use screen to run continuously
screen -S training
python notebooks/badminton_training_cpu_local.py --max-cpu-percent 80
# Ctrl+A then D to detach

# Check progress daily
screen -r training
```

**Goal:** Get final model with ~75% accuracy

### Phase 3: Monitor & Adjust

Check after each day:

**Day 1 (9 epochs completed):**
- Val accuracy improving? ✓ Continue
- Val accuracy plateaued? → Might stop early
- Training too slow? → Reduce epochs or dataset

**Day 2 (18 epochs):**
- Check val accuracy trend
- If plateaued for 10 epochs → will stop via early stopping

**Day 3-4 (27-36 epochs):**
- Likely completion point
- Check if early stopping triggered

---

## Time Breakdown

### Per Epoch (3 hours):
- Training: ~2.5 hours
- Validation: ~20 minutes
- Checkpoint saving: ~5 minutes
- Overhead: ~5 minutes

### Full Training (35 epochs):
- Training: ~87.5 hours
- Validation: ~12 hours
- Overhead: ~5.5 hours
- **Total: ~105 hours** (4.4 days)

---

## Interrupt & Resume Example

**Scenario:** You train for 2 days, then need to stop

**Day 1-2:** Train overnight, get 16 epochs done
```bash
python notebooks/badminton_training_cpu_local.py
# ... 48 hours later, 16 epochs completed
^C  # Press Ctrl+C

# Output:
# 🛑 INTERRUPT SIGNAL RECEIVED
# ✓ Checkpoint saved to: checkpoint.pth
#   Epoch: 16/50
```

**Day 3:** Resume training
```bash
python notebooks/badminton_training_cpu_local.py
# Output:
# Found checkpoint: checkpoint.pth
# Resuming from epoch 17
# ... continues training ...
```

**No data loss!** Training continues from where you left off.

---

## Cost-Benefit Analysis

### Option A: Full Training (35 epochs)
- Time: ~105 hours (4.4 days)
- Expected accuracy: 75-76%
- Best for production deployment

### Option B: Reduced Training (20 epochs)
- Time: ~60 hours (2.5 days)
- Expected accuracy: 73-74%
- Good enough for testing/demo

### Option C: Quick Training (10 epochs)
- Time: ~30 hours (1.25 days)
- Expected accuracy: 68-70%
- Good for validation, not production

**Recommendation:** Start with Option C (10 epochs) to validate, then run Option A (35 epochs) for final model.

---

## Optimization Tips

### Already Optimized:
✅ MobileNetV3 (lightweight model)
✅ Small batch size (4)
✅ Single LSTM layer
✅ CPU-optimized PyTorch
✅ Single worker (num_workers=1)

### Cannot Optimize Further Without:
- GPU (20-30x faster)
- Cloud instance (rent GPU for $1-2/hour)
- Reducing dataset size (affects accuracy)
- Reducing epochs (may underfit)

---

## Realistic Timeline

**Scenario:** You want to complete training with minimal disruption

**Week 1:**
- **Friday evening:** Start training (10pm)
- **Saturday-Sunday:** Run continuously (48 hours = 16 epochs)
- **Monday morning:** Check progress, pause if needed
- **Monday-Thursday evenings:** Run overnight (4 × 8 hours = 32 hours = 11 epochs)
- **Total after Week 1:** ~27 epochs (~81 hours)

**Week 2:**
- **Continue overnight sessions:** 3-4 more nights
- **Total:** ~35 epochs, training complete!

**Completion:** ~10-12 days real time, ~105 hours training time

---

## Summary

| Aspect | Your System |
|--------|-------------|
| **Time per epoch** | 3 hours |
| **Expected epochs** | 30-40 (with early stopping) |
| **Total time** | 90-120 hours (3.75-5 days) |
| **Realistic timeline** | 10-12 days (with overnight/weekend sessions) |
| **Recommended strategy** | Screen + overnight/weekend runs |
| **Test first** | 10 epochs with reduced dataset (~30 hours) |

**You're set up for success!** The training will take a while, but with the resume capability, you can stop/start as needed. 🚀
