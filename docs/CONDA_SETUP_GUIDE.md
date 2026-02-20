# Conda Environment Setup for Local Pose Extraction

**Issue:** MediaPipe multiprocessing fails on macOS with system Python due to mutex lock errors.

**Solution:** Use isolated conda environment with compatible dependencies.

---

## Quick Start

### 1. Install Conda (if not already installed)

**Download Miniconda:**
```bash
# For macOS (Apple Silicon/M1/M2)
curl -O https://repo.anaconda.com/miniconda/Miniconda3-latest-MacOSX-arm64.sh
bash Miniconda3-latest-MacOSX-arm64.sh

# For macOS (Intel)
curl -O https://repo.anaconda.com/miniconda/Miniconda3-latest-MacOSX-x86_64.sh
bash Miniconda3-latest-MacOSX-x86_64.sh
```

**Or download from:** https://docs.conda.io/en/latest/miniconda.html

After installation, **restart your terminal**.

### 2. Create Conda Environment

```bash
# Automated setup
bash setup_conda_env.sh

# Or manual setup
conda env create -f environment.yml
```

### 3. Activate Environment

```bash
conda activate iti123
```

### 4. Verify Installation

```bash
python --version          # Should show Python 3.10.x
python -c "import mediapipe; print(mediapipe.__version__)"  # Should show 0.10.9
python -c "import cv2; print(cv2.__version__)"              # Should show 4.8.0
```

### 5. Run Pose Extraction

```bash
python scripts/extract_poses_parallel.py \
    --video-dir data/clips/ \
    --output-dir data/processed/poses/ \
    --num-workers 8
```

---

## Why This Fixes the Issue

### The Problem

**Error encountered:**
```
libc++abi: terminating due to uncaught exception of type std::__1::system_error:
mutex lock failed: Invalid argument
```

**Root causes:**
1. **System Python + LibreSSL conflict**: macOS system Python uses LibreSSL instead of OpenSSL
2. **MediaPipe threading model**: MediaPipe's C++ backend creates thread pools that conflict with macOS system libraries
3. **Multiprocessing mutex locks**: Python's multiprocessing module creates mutex locks that fail with incompatible C++ standard libraries

### The Solution

**Conda environment provides:**
1. ✅ **Isolated Python 3.10**: Clean Python without system library conflicts
2. ✅ **OpenSSL instead of LibreSSL**: Proper SSL implementation for urllib3
3. ✅ **Compatible OpenCV build**: Compiled with matching C++ stdlib
4. ✅ **MediaPipe 0.10.9**: Tested version with stable multiprocessing
5. ✅ **Consistent C++ stdlib**: All packages use same libc++ version

---

## Environment Specification

**File:** `environment.yml`

```yaml
name: iti123
channels:
  - conda-forge
  - defaults
dependencies:
  - python=3.10
  - pip
  - numpy=1.24.3
  - opencv=4.8.0
  - pip:
    - mediapipe==0.10.9
    - pandas
    - tqdm
    - scikit-learn
    - pillow
```

**Why these versions:**
- **Python 3.10**: Most stable for MediaPipe on macOS
- **NumPy 1.24.3**: Compatible with MediaPipe 0.10.9
- **OpenCV 4.8.0**: Conda-forge build with proper C++ compatibility
- **MediaPipe 0.10.9**: Latest stable with task-based API

---

## Troubleshooting

### Issue: "conda: command not found"

**Solution:** Add conda to your PATH
```bash
# Add to ~/.zshrc (macOS default shell)
echo 'export PATH="$HOME/miniconda3/bin:$PATH"' >> ~/.zshrc
source ~/.zshrc
```

### Issue: Environment creation fails

**Solution:** Update conda first
```bash
conda update -n base -c defaults conda
conda env create -f environment.yml
```

### Issue: Still getting mutex errors

**Solution:** Reduce number of workers
```bash
# Try with fewer workers
python scripts/extract_poses_parallel.py \
    --video-dir data/clips/ \
    --output-dir data/processed/poses/ \
    --num-workers 4  # Reduced from 8
```

### Issue: "Module not found" errors

**Solution:** Verify environment is activated
```bash
conda activate iti123
which python  # Should show path to conda environment
```

---

## Performance Expectations

**With 8 workers on 10-core Mac:**
- Processing speed: ~8-10 clips/second
- Total time for 23,531 clips: **\~40-50 minutes**
- Much faster than Colab (2-3 hours)
- More stable (no crashes)

**Resource usage:**
- CPU: ~80% utilization
- RAM: ~4-6 GB
- Disk I/O: Moderate

---

## Alternative: Sequential Processing

If multiprocessing still has issues, use sequential processing:

```bash
# Single worker (no multiprocessing)
python scripts/extract_poses_parallel.py \
    --video-dir data/clips/ \
    --output-dir data/processed/poses/ \
    --num-workers 1
```

**Trade-off:**
- No mutex errors (100% stable)
- Much slower: ~8-10 hours for full dataset
- Lower resource usage

---

## Deactivating Environment

When done:
```bash
conda deactivate
```

## Removing Environment

If you need to recreate:
```bash
conda env remove -n iti123
bash setup_conda_env.sh  # Recreate
```

---

## Next Steps After Extraction

1. **Verify extraction:**
```bash
   find data/processed/poses -name "*_pose.pkl" | wc -l
   # Should show ~23,531
```

2. **Check metadata:**
```bash
   cat data/metadata.csv | head
```

3. **Upload to GCS (optional):**
```bash
   gsutil -m rsync -r data/processed/poses/ gs://iti123storage/features/poses/
```

4. **Train models in Colab:**
  - Use `notebooks/deep_learning_training_colab.ipynb`
  - Download poses from GCS or upload directly

---

**Status:** Production ready - solves macOS multiprocessing issues
