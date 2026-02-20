# Libraries and Versions for CPU Training

## Quick Summary

### Required Libraries (11 packages)

```bash
# One-line installation command:
pip install torch torchvision torchaudio opencv-python numpy pandas Pillow scikit-learn matplotlib seaborn tqdm
```

| # | Library | Min Version | Purpose |
|---|---------|-------------|---------|
| 1 | **torch** | 2.0.0 | Deep learning framework (neural networks, optimizers) |
| 2 | **torchvision** | 0.15.0 | Pre-trained models (MobileNetV3, ResNet18), transforms |
| 3 | **torchaudio** | 2.0.0 | PyTorch audio processing (dependency) |
| 4 | **opencv-python** | 4.8.0 | Video/image reading and processing (cv2) |
| 5 | **numpy** | 1.24.0 | Array operations, numerical computing |
| 6 | **pandas** | 2.0.0 | Data manipulation (reading metadata.csv) |
| 7 | **Pillow** | 10.0.0 | Image loading and PIL transforms |
| 8 | **scikit-learn** | 1.3.0 | Train/test split, metrics, confusion matrix |
| 9 | **matplotlib** | 3.7.0 | Plotting confusion matrix and training curves |
| 10 | **seaborn** | 0.12.0 | Enhanced statistical visualizations |
| 11 | **tqdm** | 4.65.0 | Progress bars during training |

### Built-in Libraries (No installation needed)

These come with Python 3.8+:
- `os`, `sys`, `time`, `json`, `signal`, `argparse`, `pathlib`, `datetime`, `collections`

---

## Installation Instructions

### Method 1: From Requirements File (Recommended)

```bash
# Navigate to project
cd /Volumes/Ext/GenAI/iti123_v2

# Install all dependencies
pip install -r requirements_cpu_training.txt

# Verify installation
python verify_dependencies.py
```

### Method 2: CPU-Optimized PyTorch (Saves 3GB)

For CPU-only training, install the lighter PyTorch CPU version:

```bash
# Step 1: Install PyTorch CPU-only (saves ~3GB disk space)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu

# Step 2: Install other dependencies
pip install opencv-python numpy pandas Pillow scikit-learn matplotlib seaborn tqdm
```

### Method 3: Conda Environment

```bash
# Create environment
conda create -n badminton python=3.10

# Activate
conda activate badminton

# Install PyTorch
conda install pytorch torchvision torchaudio cpuonly -c pytorch

# Install others
conda install numpy pandas scikit-learn matplotlib seaborn tqdm
pip install opencv-python
```

---

## Tested Working Versions

These specific versions are confirmed working in production:

```
torch==2.1.0
torchvision==0.16.0
torchaudio==2.1.0
opencv-python==4.8.1.78
numpy==1.26.2
pandas==2.1.3
Pillow==10.1.0
scikit-learn==1.3.2
matplotlib==3.8.2
seaborn==0.13.0
tqdm==4.66.1
```

To install exact versions:
```bash
pip install torch==2.1.0 torchvision==0.16.0 torchaudio==2.1.0 opencv-python==4.8.1.78 numpy==1.26.2 pandas==2.1.3 Pillow==10.1.0 scikit-learn==1.3.2 matplotlib==3.8.2 seaborn==0.13.0 tqdm==4.66.1
```

---

## What Each Library Does in the Training Script

### 1. PyTorch (torch, torchvision, torchaudio)

**Used for:**
- Neural network model definition (CNN + LSTM)
- Training loop (forward/backward propagation)
- Optimizers (Adam)
- Loss functions (CrossEntropyLoss, FocalLoss)
- Data loading (DataLoader, Dataset)
- Model saving/loading (torch.save, torch.load)

**Code examples:**
```python
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

import torchvision.transforms as transforms
import torchvision.models as models
```

### 2. OpenCV (opencv-python)

**Used for:**
- Reading video frames (not used in CPU script, but in preprocessing)
- Image processing operations

**Code example:**
```python
import cv2
# Note: In CPU script, cv2 is imported but .npy files are used directly
```

### 3. NumPy

**Used for:**
- Array operations on frames
- Loading .npy files (pre-extracted frames)
- Numerical computations
- Frame sampling (linspace)

**Code examples:**
```python
import numpy as np
frames = np.load('clip_000001.npy')  # Load frames
indices = np.linspace(0, 15, 16, dtype=int)  # Sample frames
```

### 4. Pandas

**Used for:**
- Reading metadata.csv
- Data manipulation
- Creating train/val/test splits

**Code example:**
```python
import pandas as pd
metadata = pd.read_csv('metadata.csv')
```

### 5. Pillow (PIL)

**Used for:**
- Converting numpy arrays to PIL Images
- Applying transforms (RandomHorizontalFlip, ColorJitter)
- Image augmentation

**Code example:**
```python
from PIL import Image
frame_pil = Image.fromarray(frame)
```

### 6. Scikit-learn

**Used for:**
- Train/test/validation split
- Classification metrics (precision, recall, F1)
- Confusion matrix generation
- Classification report

**Code examples:**
```python
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, f1_score

train_paths, test_paths = train_test_split(paths, test_size=0.2)
cm = confusion_matrix(y_true, y_pred)
```

### 7. Matplotlib

**Used for:**
- Plotting confusion matrix
- Plotting training curves (loss, accuracy)
- Saving plots as images

**Code example:**
```python
import matplotlib.pyplot as plt
plt.figure(figsize=(10, 8))
plt.imshow(confusion_matrix)
plt.savefig('confusion_matrix.png')
```

### 8. Seaborn

**Used for:**
- Enhanced visualization of confusion matrix
- Heatmaps with better styling

**Code example:**
```python
import seaborn as sns
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
```

### 9. Tqdm

**Used for:**
- Progress bars during training
- Progress bars during validation
- Time estimation

**Code example:**
```python
from tqdm import tqdm
for frames, labels in tqdm(dataloader, desc='Training'):
    # Training code
```

---

## Platform-Specific Notes

### macOS

```bash
# No special requirements
pip install -r requirements_cpu_training.txt

# On M1/M2 Macs, PyTorch will automatically use MPS (Metal Performance Shaders)
# for GPU acceleration even without CUDA
```

### Linux

```bash
# May need system packages
sudo apt update
sudo apt install python3-dev python3-pip

# Then install Python packages
pip install -r requirements_cpu_training.txt
```

### Windows

```bash
# May need Visual C++ Redistributable for OpenCV
# Download from: https://aka.ms/vs/17/release/vc_redist.x64.exe

pip install -r requirements_cpu_training.txt
```

---

## Verification

### Check Installed Packages

```bash
# Run verification script
python verify_dependencies.py
```

**Expected output:**
```
======================================================================
Dependency Check for badminton_training_cpu_local.py
======================================================================

Checking Python packages...
----------------------------------------------------------------------
✓ torch           2.1.0           (min: 2.0.0)
✓ torchvision     0.16.0          (min: 0.15.0)
✓ cv2             4.8.1           (min: 4.8.0)
✓ numpy           1.26.2          (min: 1.24.0)
✓ pandas          2.1.3           (min: 2.0.0)
✓ PIL             10.1.0          (min: 10.0.0)
✓ sklearn         1.3.2           (min: 1.3.0)
✓ matplotlib      3.8.2           (min: 3.7.0)
✓ seaborn         0.13.0          (min: 0.12.0)
✓ tqdm            4.66.1          (min: 4.65.0)

Checking built-in modules...
----------------------------------------------------------------------
✓ os
✓ sys
...

Python version check...
----------------------------------------------------------------------
Python 3.10.19
✓ Python version is compatible (≥3.8)

======================================================================
SUMMARY
======================================================================

✅ All dependencies satisfied!

   • 10 packages installed
   • 9 built-in modules available

You can now run:
   python notebooks/badminton_training_cpu_local.py
```

### Manual Check

```bash
# Check individual packages
python -c "import torch; print(f'torch: {torch.__version__}')"
python -c "import torchvision; print(f'torchvision: {torchvision.__version__}')"
python -c "import cv2; print(f'opencv: {cv2.__version__}')"
python -c "import numpy; print(f'numpy: {numpy.__version__}')"
python -c "import pandas; print(f'pandas: {pandas.__version__}')"
python -c "import PIL; print(f'Pillow: {PIL.__version__}')"
python -c "import sklearn; print(f'scikit-learn: {sklearn.__version__}')"
python -c "import matplotlib; print(f'matplotlib: {matplotlib.__version__}')"
python -c "import seaborn; print(f'seaborn: {seaborn.__version__}')"
python -c "import tqdm; print(f'tqdm: {tqdm.__version__}')"
```

---

## Disk Space Requirements

| Component | Size |
|-----------|------|
| PyTorch (CPU-only) | ~200 MB |
| TorchVision | ~15 MB |
| OpenCV | ~60 MB |
| NumPy | ~20 MB |
| Pandas | ~30 MB |
| Pillow | ~5 MB |
| Scikit-learn | ~40 MB |
| Matplotlib | ~20 MB |
| Seaborn | ~5 MB |
| Tqdm | ~2 MB |
| **Total** | **~400 MB** |

**Note:** With CUDA-enabled PyTorch, total size increases to ~3.2 GB.

---

## Python Version Requirements

| Python Version | Status |
|----------------|--------|
| 3.7 or earlier | ❌ Not supported |
| 3.8 | ✅ Minimum supported |
| 3.9 | ✅ Supported |
| 3.10 | ✅ **Recommended** |
| 3.11 | ✅ Supported |
| 3.12+ | ⚠️ Limited support (PyTorch compatibility) |

**Check your version:**
```bash
python --version
```

---

## Troubleshooting

### Problem: "No module named 'torch'"

```bash
pip install torch torchvision torchaudio
```

### Problem: "No module named 'cv2'"

```bash
pip install opencv-python
```

### Problem: "cannot import name 'xxx' from 'PIL'"

```bash
pip uninstall Pillow PIL
pip install Pillow
```

### Problem: NumPy version conflict

```bash
pip install --upgrade numpy
```

### Problem: "DLL load failed" (Windows)

Install Visual C++ Redistributable:
https://aka.ms/vs/17/release/vc_redist.x64.exe

---

## Summary

### Installation Checklist

- [ ] Python 3.8+ installed
- [ ] Run: `pip install -r requirements_cpu_training.txt`
- [ ] Run: `python verify_dependencies.py`
- [ ] See "✅ All dependencies satisfied!"
- [ ] Ready to train: `python notebooks/badminton_training_cpu_local.py`

### Total Requirements

- **Packages**: 11 (torch, torchvision, torchaudio, opencv-python, numpy, pandas, Pillow, scikit-learn, matplotlib, seaborn, tqdm)
- **Disk space**: ~400 MB (CPU) or ~3.2 GB (CUDA)
- **Installation time**: ~5 minutes
- **Python version**: 3.8 to 3.11 (3.10 recommended)

---

## Files Created

1. **requirements_cpu_training.txt** - Pip requirements file
2. **verify_dependencies.py** - Dependency checker script
3. **DEPENDENCIES_AND_VERSIONS.md** - Full documentation
4. **DEPENDENCIES_QUICK_REF.md** - Quick reference card
5. **LIBRARIES_AND_VERSIONS.md** - This file

---

## Next Steps

1. Install dependencies:
   ```bash
   pip install -r requirements_cpu_training.txt
   ```

2. Verify installation:
   ```bash
   python verify_dependencies.py
   ```

3. Run training:
   ```bash
   python notebooks/badminton_training_cpu_local.py
   ```

Good luck! 🚀
