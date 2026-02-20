# Dependencies and Versions for CPU Training

## Complete Library List

### Required Libraries

| Library | Version | Purpose | Import Statement |
|---------|---------|---------|------------------|
| **torch** | ≥2.0.0 | Deep learning framework | `import torch` |
| **torchvision** | ≥0.15.0 | Computer vision models & transforms | `import torchvision` |
| **torchaudio** | ≥2.0.0 | Audio processing (dependency) | - |
| **opencv-python** | ≥4.8.0 | Video/image processing | `import cv2` |
| **numpy** | ≥1.24.0 | Array operations | `import numpy as np` |
| **pandas** | ≥2.0.0 | Data manipulation | `import pandas as pd` |
| **Pillow** | ≥10.0.0 | Image loading/processing | `from PIL import Image` |
| **scikit-learn** | ≥1.3.0 | Train/test split, metrics | `from sklearn...` |
| **matplotlib** | ≥3.7.0 | Plotting & visualization | `import matplotlib` |
| **seaborn** | ≥0.12.0 | Enhanced plotting | `import seaborn` |
| **tqdm** | ≥4.65.0 | Progress bars | `from tqdm import tqdm` |

### Built-in Libraries (No Installation Needed)

These come with Python 3.8+:
- `os` - Operating system interface
- `sys` - System-specific parameters
- `time` - Time access and conversions
- `json` - JSON encoder/decoder
- `signal` - Signal handling (for Ctrl+C)
- `argparse` - Command-line argument parsing
- `pathlib` - Object-oriented filesystem paths
- `datetime` - Date and time manipulation
- `collections` - Container datatypes

---

## Installation Methods

### Method 1: Quick Install (Recommended)

```bash
# Navigate to project directory
cd /Volumes/Ext/GenAI/iti123_v2

# Install all dependencies
pip install -r requirements_cpu_training.txt
```

### Method 2: CPU-Optimized PyTorch (Smaller & Faster)

For CPU-only training, install CPU-specific PyTorch (no CUDA bloat):

```bash
# Install PyTorch CPU-only version (saves ~3GB disk space)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu

# Then install other dependencies
pip install opencv-python numpy pandas Pillow scikit-learn matplotlib seaborn tqdm
```

### Method 3: Individual Packages

```bash
# Core deep learning
pip install torch>=2.0.0 torchvision>=0.15.0 torchaudio>=2.0.0

# Computer vision
pip install opencv-python>=4.8.0 Pillow>=10.0.0

# Data science
pip install numpy>=1.24.0 pandas>=2.0.0 scikit-learn>=1.3.0

# Visualization
pip install matplotlib>=3.7.0 seaborn>=0.12.0 tqdm>=4.65.0
```

### Method 4: Anaconda/Conda

```bash
# Create new conda environment
conda create -n badminton python=3.10

# Activate environment
conda activate badminton

# Install PyTorch CPU
conda install pytorch torchvision torchaudio cpuonly -c pytorch

# Install other packages
conda install numpy pandas scikit-learn matplotlib seaborn tqdm
pip install opencv-python
```

---

## Tested Versions (Production)

These versions are confirmed working:

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
pip install torch==2.1.0 torchvision==0.16.0 torchaudio==2.1.0
pip install opencv-python==4.8.1.78 numpy==1.26.2 pandas==2.1.3
pip install Pillow==10.1.0 scikit-learn==1.3.2 matplotlib==3.8.2
pip install seaborn==0.13.0 tqdm==4.66.1
```

---

## Python Version Requirements

### Minimum: Python 3.8
### Recommended: Python 3.10 or 3.11
### Not supported: Python 3.7 or earlier, Python 3.12+ (PyTorch compatibility)

Check your Python version:
```bash
python --version
# Should show: Python 3.10.x or 3.11.x
```

---

## Disk Space Requirements

| Component | Size |
|-----------|------|
| **PyTorch (CPU)** | ~200 MB |
| **PyTorch (with CUDA)** | ~3 GB |
| **TorchVision** | ~15 MB |
| **OpenCV** | ~60 MB |
| **NumPy** | ~20 MB |
| **Pandas** | ~30 MB |
| **Scikit-learn** | ~40 MB |
| **Other libraries** | ~30 MB |
| **Total (CPU-only)** | ~400 MB |
| **Total (with CUDA)** | ~3.2 GB |

---

## Platform-Specific Notes

### macOS

```bash
# Install Xcode Command Line Tools (if not already installed)
xcode-select --install

# Install dependencies
pip install -r requirements_cpu_training.txt

# For M1/M2 Macs, use MPS acceleration (Metal Performance Shaders)
# PyTorch will automatically use MPS if available
```

**Note**: On Apple Silicon (M1/M2), PyTorch can use MPS backend for GPU acceleration even without CUDA.

### Linux

```bash
# Install system dependencies
sudo apt update
sudo apt install python3-dev python3-pip

# Install Python packages
pip install -r requirements_cpu_training.txt
```

### Windows

```bash
# Use PowerShell or Command Prompt
pip install -r requirements_cpu_training.txt
```

**Note**: OpenCV may require Microsoft Visual C++ Redistributable. Download from:
https://aka.ms/vs/17/release/vc_redist.x64.exe

---

## Verification Script

Run this to verify all dependencies are installed correctly:

```python
#!/usr/bin/env python3
"""Verify all dependencies for CPU training script"""

import sys

print("Checking dependencies...\n")

required = {
    'torch': '2.0.0',
    'torchvision': '0.15.0',
    'cv2': '4.8.0',
    'numpy': '1.24.0',
    'pandas': '2.0.0',
    'PIL': '10.0.0',
    'sklearn': '1.3.0',
    'matplotlib': '3.7.0',
    'seaborn': '0.12.0',
    'tqdm': '4.65.0',
}

missing = []
outdated = []

for module, min_version in required.items():
    try:
        if module == 'PIL':
            import PIL
            version = PIL.__version__
        elif module == 'cv2':
            import cv2
            version = cv2.__version__
        elif module == 'sklearn':
            import sklearn
            version = sklearn.__version__
        else:
            mod = __import__(module)
            version = mod.__version__

        print(f"✓ {module:15} {version:10} (min: {min_version})")

        # Simple version check
        if version.split('.')[0] < min_version.split('.')[0]:
            outdated.append((module, version, min_version))

    except ImportError:
        print(f"✗ {module:15} NOT FOUND")
        missing.append(module)
    except AttributeError:
        print(f"✓ {module:15} installed (version unknown)")

# Check built-in modules
print("\nBuilt-in modules:")
builtins = ['os', 'sys', 'time', 'json', 'signal', 'argparse', 'pathlib', 'datetime', 'collections']
for module in builtins:
    try:
        __import__(module)
        print(f"✓ {module}")
    except ImportError:
        print(f"✗ {module} (should be built-in!)")

print("\n" + "="*60)
if missing:
    print("❌ MISSING PACKAGES:")
    for module in missing:
        print(f"   - {module}")
    print("\nInstall with:")
    print(f"   pip install {' '.join(missing)}")
    sys.exit(1)
elif outdated:
    print("⚠️  OUTDATED PACKAGES:")
    for module, current, minimum in outdated:
        print(f"   - {module}: {current} (need ≥{minimum})")
    print("\nUpgrade with:")
    print(f"   pip install --upgrade {' '.join([m for m, _, _ in outdated])}")
    sys.exit(1)
else:
    print("✅ All dependencies satisfied!")
    print("\nYou can now run:")
    print("   python badminton_training_cpu_local.py")
    sys.exit(0)
```

Save as `verify_dependencies.py` and run:
```bash
python verify_dependencies.py
```

---

## Common Installation Issues

### Issue 1: Torch Import Error

**Error:**
```
ImportError: cannot import name 'xxx' from 'torch'
```

**Solution:**
```bash
# Uninstall and reinstall PyTorch
pip uninstall torch torchvision torchaudio
pip install torch torchvision torchaudio
```

### Issue 2: OpenCV Not Found

**Error:**
```
ModuleNotFoundError: No module named 'cv2'
```

**Solution:**
```bash
# Try different OpenCV package
pip uninstall opencv-python opencv-contrib-python
pip install opencv-python-headless
```

### Issue 3: NumPy Version Conflict

**Error:**
```
RuntimeError: module compiled against API version ... but this version of numpy is ...
```

**Solution:**
```bash
# Upgrade NumPy
pip install --upgrade numpy
```

### Issue 4: Pillow/PIL Import Error

**Error:**
```
ImportError: cannot import name 'Image' from 'PIL'
```

**Solution:**
```bash
# Reinstall Pillow
pip uninstall Pillow PIL
pip install Pillow
```

### Issue 5: Scikit-learn Version Too Old

**Error:**
```
AttributeError: module 'sklearn' has no attribute 'xxx'
```

**Solution:**
```bash
# Upgrade scikit-learn
pip install --upgrade scikit-learn
```

---

## Minimal Installation (Testing Only)

For quick testing without full dependencies:

```bash
# Absolute minimum (won't run full training)
pip install torch torchvision numpy pandas scikit-learn tqdm

# Missing: opencv, matplotlib, seaborn, Pillow
# Script will fail when trying to load frames or plot results
```

---

## Virtual Environment Setup (Recommended)

### Using venv (Python built-in)

```bash
# Create virtual environment
python -m venv badminton_env

# Activate (macOS/Linux)
source badminton_env/bin/activate

# Activate (Windows)
badminton_env\Scripts\activate

# Install dependencies
pip install -r requirements_cpu_training.txt

# Deactivate when done
deactivate
```

### Using conda

```bash
# Create environment
conda create -n badminton python=3.10

# Activate
conda activate badminton

# Install dependencies
pip install -r requirements_cpu_training.txt

# Deactivate
conda deactivate
```

---

## Docker Setup (Advanced)

For reproducible environment:

```dockerfile
# Dockerfile
FROM python:3.10-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements
COPY requirements_cpu_training.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements_cpu_training.txt

# Copy training script
COPY notebooks/badminton_training_cpu_local.py .

# Run training
CMD ["python", "badminton_training_cpu_local.py"]
```

Build and run:
```bash
docker build -t badminton-training .
docker run -v $(pwd)/data:/app/data -v $(pwd)/outputs:/app/outputs badminton-training
```

---

## Summary

### Quick Install Command (One-liner)

```bash
pip install torch>=2.0.0 torchvision>=0.15.0 torchaudio>=2.0.0 opencv-python>=4.8.0 numpy>=1.24.0 pandas>=2.0.0 Pillow>=10.0.0 scikit-learn>=1.3.0 matplotlib>=3.7.0 seaborn>=0.12.0 tqdm>=4.65.0
```

### Recommended Installation

```bash
# 1. Create virtual environment
python -m venv badminton_env
source badminton_env/bin/activate  # macOS/Linux
# or: badminton_env\Scripts\activate  # Windows

# 2. Install dependencies
pip install -r requirements_cpu_training.txt

# 3. Verify installation
python verify_dependencies.py

# 4. Run training
python badminton_training_cpu_local.py
```

---

## Total Installation Size

- **Disk space**: ~400 MB (CPU-only) or ~3.2 GB (with CUDA)
- **Download size**: ~150 MB (CPU-only) or ~1.5 GB (with CUDA)
- **Installation time**: ~5 minutes on fast connection

---

## Compatibility Matrix

| Python | PyTorch | TorchVision | Status |
|--------|---------|-------------|--------|
| 3.8 | 2.0.0 | 0.15.0 | ✅ Supported |
| 3.9 | 2.0.0 | 0.15.0 | ✅ Supported |
| 3.10 | 2.0.0 | 0.15.0 | ✅ Recommended |
| 3.11 | 2.0.0 | 0.15.0 | ✅ Supported |
| 3.12 | 2.1.0+ | 0.16.0+ | ⚠️ Limited support |
| 3.7 | - | - | ❌ Not supported |

---

## Need Help?

If you encounter installation issues:

1. Check [PyTorch Installation Guide](https://pytorch.org/get-started/locally/)
2. Check [OpenCV Installation Issues](https://github.com/opencv/opencv-python/issues)
3. Run `verify_dependencies.py` to diagnose problems
4. Search for error message on Stack Overflow

---

## Files Reference

- **requirements_cpu_training.txt** - Pip requirements file
- **verify_dependencies.py** - Dependency verification script (create if needed)
- **badminton_training_cpu_local.py** - Main training script
