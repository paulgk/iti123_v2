# CPU Training Quick Start

## Your System: 3 hours per epoch

**Full training:** ~105 hours (4.4 days continuous) or 10-12 days real time

---

## Quick Start (5 minutes)

```bash
# 1. Setup environment
cd /Volumes/Ext/GenAI/iti123_v2
conda env update -f environment.yml --prune
conda activate iti123

# 2. Verify dependencies
python verify_dependencies.py

# 3. Start training with screen
screen -S training
python notebooks/badminton_training_cpu_local.py --max-cpu-percent 80

# 4. Detach (training continues in background)
# Press: Ctrl+A then D

# 5. Reattach anytime to check progress
screen -r training
```

---

## Recommended: Quick Test First (30 hours)

Before full training, test with 10 epochs:

```python
# Edit notebooks/badminton_training_cpu_local.py, line 65
'num_epochs': 10,  # Change from 50 to 10

# And line 296 - add these lines:
print(f"Loaded {len(npy_paths)} samples")
npy_paths = npy_paths[:5000]  # Use 5000 samples
labels = labels[:5000]
print(f"REDUCED to {len(npy_paths)} samples for testing")
```

**Expected time:** ~30 hours (1.25 days)

Once satisfied, remove reduction and run full training.

---

## Time Estimates

| Training Mode | Epochs | Time | Accuracy |
|---------------|--------|------|----------|
| **Quick test** | 10 | 30 hours | 68-70% |
| **Reduced** | 20 | 60 hours | 73-74% |
| **Full** | 35 | 105 hours | 75-76% |

---

## Monitoring Progress

```bash
# Check if training is running
screen -ls

# Reattach to see progress
screen -r training

# In another terminal, tail the log
tail -f outputs/results_cpu_local/training_history.csv
```

---

## Interrupt & Resume

**Stop anytime:**
```bash
# Reattach to training
screen -r training

# Press Ctrl+C (once!)
# Checkpoint saves automatically
```

**Resume later:**
```bash
screen -S training
python notebooks/badminton_training_cpu_local.py
# Automatically resumes from checkpoint
```

---

## Realistic Timeline (Overnight/Weekend Strategy)

**Week 1:**
- Friday evening → Monday morning (60 hours) = 20 epochs
- Monday-Thursday nights (32 hours) = 11 epochs

**Week 2:**
- Continue 2-3 nights (16-24 hours) = 5-8 epochs
- **Total:** ~35 epochs, 10-12 days real time

---

## Commands Reference

```bash
# Start training
conda activate iti123
screen -S training
python notebooks/badminton_training_cpu_local.py

# Detach (Ctrl+A then D)

# Check status
screen -ls

# Reattach
screen -r training

# Stop training (inside screen)
Ctrl+C  # Press once, checkpoint saves

# Resume
python notebooks/badminton_training_cpu_local.py
```

---

## Files

- **Script:** [notebooks/badminton_training_cpu_local.py](notebooks/badminton_training_cpu_local.py)
- **Environment:** [environment.yml](environment.yml)
- **Dependencies:** [verify_dependencies.py](verify_dependencies.py)
- **Full guide:** [docs/CPU_TRAINING_TIME_ESTIMATES.md](docs/CPU_TRAINING_TIME_ESTIMATES.md)

---

## Need Help?

- [CPU Training Guide](docs/CPU_TRAINING_GUIDE.md) - Complete documentation
- [Dependencies Guide](docs/DEPENDENCIES_AND_VERSIONS.md) - Library setup
- [Multiprocessing Fix](docs/MULTIPROCESSING_FIX.md) - Why we fixed it

**You're ready to train!** 🚀
