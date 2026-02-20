# CPU Training - Quick Reference Card

## 🚀 Quick Start

```bash
# Standard training (recommended)
python badminton_training_cpu_local.py

# Light CPU load (background training)
python badminton_training_cpu_local.py --max-cpu-percent 60

# Targeted sampling (skip pre-shot frames)
python badminton_training_cpu_local.py --skip-ratio 0.3
```

---

## 🎛️ CPU Control

| Command | CPU Usage | When to Use |
|---------|-----------|-------------|
| `--max-cpu-percent 60` | 60% | Background, while working |
| `--max-cpu-percent 80` | 80% | Default, balanced |
| `--max-cpu-percent 100` | 100% | Maximum speed, overnight |

---

## 🛑 Interrupting Training

**Press `Ctrl+C` once**
- ✅ Saves checkpoint automatically
- ✅ No data loss
- ✅ Can resume anytime

**Press `Ctrl+C` twice**
- ⚠️ Forces immediate quit
- ⚠️ May lose current batch
- ❌ NOT recommended

---

## ♻️ Resuming Training

```bash
# Automatic resume (if checkpoint.pth exists)
python badminton_training_cpu_local.py

# Manual resume from specific checkpoint
python badminton_training_cpu_local.py --resume /path/to/checkpoint.pth
```

---

## 📊 What Gets Saved

| File | Content | When |
|------|---------|------|
| `checkpoint.pth` | Full training state | Every epoch + on Ctrl+C |
| `best_model.pth` | Best model weights | When validation improves |
| `results_summary.json` | Final results | After training completes |

---

## ⏱️ Expected Time

| Hardware | CPU % | Time per Epoch | Total (50 epochs) |
|----------|-------|----------------|-------------------|
| M1 Mac | 80% | ~15 min | ~12.5 hours |
| Intel i7 | 80% | ~20 min | ~16.7 hours |
| Intel i5 | 60% | ~25 min | ~20.8 hours |

**Note**: With early stopping, usually completes in **10-15 hours** (not full 50 epochs)

---

## 🔍 Monitoring Progress

```bash
# View live progress
# (script shows progress automatically)

# Check CPU usage (macOS)
top -pid $(pgrep -f badminton_training)

# Check CPU usage (Linux)
htop -p $(pgrep -f badminton_training)
```

---

## 🐛 Common Issues

### Training Too Slow
```bash
# Increase CPU limit
python badminton_training_cpu_local.py --max-cpu-percent 90
```

### Out of Memory
```python
# Edit script: reduce batch size
CONFIG['batch_size'] = 2  # Line 64
```

### Can't Find Data Files
```python
# Edit script: fix data path
DATA_ROOT = "/correct/path/to/frames"  # Line 53
```

---

## 🎯 Recommended Workflow

1. **Test first epoch** (~15 min)
   ```bash
   python badminton_training_cpu_local.py --max-cpu-percent 70
   ```

2. **If looks good, let it run overnight**

3. **Check results next day**
   ```bash
   cat outputs/results_cpu_local/results_summary.json
   ```

---

## 📂 Output Location

```
outputs/results_cpu_local/
├── best_model.pth              # ← Use this for inference
├── checkpoint.pth              # ← For resuming
├── results_summary.json        # ← Final accuracy
├── classification_report.txt   # ← Per-class metrics
└── confusion_matrix.png        # ← Visualization
```

---

## ✅ Safety Guarantees

| Scenario | Data Loss | Recovery |
|----------|-----------|----------|
| **Ctrl+C** | None | Automatic |
| **Power loss** | < 1 epoch | Automatic |
| **System crash** | < 1 epoch | Automatic |
| **Out of memory** | < 1 epoch | Automatic |

**Just run the script again - it always resumes from the last checkpoint!**

---

## 📖 Full Documentation

See [CPU_TRAINING_GUIDE.md](./CPU_TRAINING_GUIDE.md) for complete details.
