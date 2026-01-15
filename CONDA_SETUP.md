# Conda Environment Setup Guide

## Current Status

You mentioned the dependency conflict with `pdflatex` - this means you **already ran pip install** in your conda environment. Great!

## To Complete Setup:

### 1. Activate Your Conda Environment

```bash
conda activate iti123
```

### 2. Verify the Environment is Active

After activation, you should see `(iti123)` at the beginning of your command prompt:
```
(iti123) paul@macbook ~ %
```

### 3. Verify Installation

Once the environment is activated, run:

```bash
python verify_installation.py
```

This will show you which packages are properly installed.

### 4. Fix the pdflatex Conflict (Optional)

The `attrs` conflict won't affect your project. You can safely ignore it, or remove pdflatex:

```bash
conda activate iti123
pip uninstall pdflatex
```

---

## Quick Reference Commands

### Activate environment:
```bash
conda activate iti123
```

### Deactivate environment:
```bash
conda deactivate
```

### Install missing packages (if needed):
```bash
conda activate iti123
pip install -r requirements.txt
```

### Check what's installed:
```bash
conda activate iti123
pip list
```

### Install specific packages with conda (preferred for some packages):
```bash
conda activate iti123
conda install -c conda-forge tensorflow
conda install -c conda-forge opencv
conda install -c conda-forge pandas
```

---

## Conda vs Pip

For this project, you can use either:
- **conda install** (preferred for scientific packages)
- **pip install** (works fine for everything)

Most packages in `requirements.txt` work with either method.

---

## Next Steps After Activation

Once your environment is active:

1. ✅ Verify installation: `python verify_installation.py`
2. ✅ Start Jupyter: `jupyter notebook`
3. ✅ Begin data exploration
4. ✅ Place videos in `data/raw_videos/`
5. ✅ Download annotations to `data/annotations/`

---

## Troubleshooting

**If packages are missing after pip install:**

```bash
conda activate iti123
pip install tensorflow opencv-python mediapipe pandas seaborn gradio mlflow jupyter
```

**If you prefer conda packages:**

```bash
conda activate iti123
conda install -c conda-forge tensorflow pandas opencv seaborn jupyter
pip install mediapipe gradio mlflow  # These are pip-only
```

---

## Environment Info

**Environment name:** iti123
**Python version:** Should be 3.9 or higher
**Package manager:** conda + pip

To check your conda environment details:
```bash
conda activate iti123
conda info
python --version
```
