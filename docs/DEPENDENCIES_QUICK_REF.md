# Dependencies - Quick Reference

## 📦 One-Line Installation

```bash
pip install torch torchvision torchaudio opencv-python numpy pandas Pillow scikit-learn matplotlib seaborn tqdm
```

---

## 📋 Complete List

| Package | Version | Purpose |
|---------|---------|---------|
| torch | ≥2.0.0 | Deep learning |
| torchvision | ≥0.15.0 | Vision models |
| opencv-python | ≥4.8.0 | Video processing |
| numpy | ≥1.24.0 | Arrays |
| pandas | ≥2.0.0 | Data tables |
| Pillow | ≥10.0.0 | Images |
| scikit-learn | ≥1.3.0 | ML utilities |
| matplotlib | ≥3.7.0 | Plotting |
| seaborn | ≥0.12.0 | Plots |
| tqdm | ≥4.65.0 | Progress bars |

---

## 🚀 Quick Setup

```bash
# 1. Install from requirements file
pip install -r requirements_cpu_training.txt

# 2. Verify installation
python verify_dependencies.py

# 3. Run training
python notebooks/badminton_training_cpu_local.py
```

---

## 💻 CPU-Optimized Install (Saves 3GB!)

```bash
# Install PyTorch CPU version (smaller, no CUDA)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu

# Then install others
pip install opencv-python numpy pandas Pillow scikit-learn matplotlib seaborn tqdm
```

---

## 🐍 Python Version

- **Minimum**: 3.8
- **Recommended**: 3.10 or 3.11
- **Check**: `python --version`

---

## 💾 Disk Space

- **CPU version**: ~400 MB
- **With CUDA**: ~3.2 GB

---

## ✅ Verify Installation

```bash
python verify_dependencies.py
```

Expected output:
```
✓ torch           2.1.0
✓ torchvision     0.16.0
✓ cv2             4.8.1
✓ numpy           1.26.2
...
✅ All dependencies satisfied!
```

---

## 🔧 Common Fixes

### Missing cv2
```bash
pip install opencv-python
```

### Torch import error
```bash
pip uninstall torch torchvision
pip install torch torchvision
```

### NumPy version conflict
```bash
pip install --upgrade numpy
```

---

## 📁 Files

- `requirements_cpu_training.txt` - Pip requirements
- `verify_dependencies.py` - Dependency checker
- `DEPENDENCIES_AND_VERSIONS.md` - Full documentation

---

## 🆘 Still Having Issues?

Run the verification script for detailed diagnostics:
```bash
python verify_dependencies.py
```

It will tell you exactly what's missing or outdated!
