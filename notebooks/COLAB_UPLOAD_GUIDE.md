# How to Upload Notebooks to Colab

If you're having trouble opening notebooks directly in Colab, follow these steps:

## Method 1: Upload from GitHub (Recommended)

1. **Push notebooks to GitHub:**
   ```bash
   git push origin milestone/v1.1-coach-informed-ml
   ```

2. **Open in Colab:**
   - Go to https://colab.research.google.com/
   - Click "GitHub" tab
   - Enter your repository: `YOUR_USERNAME/iti123_v2`
   - Select branch: `milestone/v1.1-coach-informed-ml`
   - Click on the notebook you want to open

## Method 2: Upload Directly

1. **Download notebook from your local repo:**
   - Navigate to `notebooks/` folder
   - Copy the `.ipynb` file you want to use

2. **Upload to Colab:**
   - Go to https://colab.research.google.com/
   - Click "Upload" tab
   - Drag and drop the `.ipynb` file
   - Or click "Choose File" and select it

## Method 3: Open from Google Drive

1. **Upload to Google Drive:**
   - Upload the notebook to your Google Drive
   - Right-click the file
   - Select "Open with" → "Google Colaboratory"

## Method 4: Clone in Colab Terminal

If you just need to run the code:

1. **Open a blank Colab notebook**

2. **Clone the repository:**
   ```python
   !git clone https://github.com/YOUR_USERNAME/iti123_v2.git
   %cd iti123_v2
   ```

3. **Run commands manually:**
   Copy and paste the code cells from the notebook

---

## Troubleshooting "Unable to read file" Error

This error can happen due to:

### Issue 1: File Too Large
- **Solution:** Use individual phase notebooks instead of complete workflow
- Smaller notebooks: `phase2_validation_colab.ipynb`, `phase3_model_training_colab.ipynb`, etc.

### Issue 2: Browser Cache
- **Solution:**
  ```
  1. Clear browser cache
  2. Try in incognito/private mode
  3. Try a different browser (Chrome recommended)
  ```

### Issue 3: Colab Temporary Issue
- **Solution:** Wait a few minutes and try again

### Issue 4: Network Issue
- **Solution:** Check your internet connection and try again

---

## Recommended Approach for Large Workflows

For the complete workflow, instead of using the all-in-one notebook, use this sequence:

### Step 1: Setup (Manual in Terminal)
```bash
cd /content
git clone https://github.com/YOUR_USERNAME/iti123_v2.git
cd iti123_v2
bash scripts/colab_setup.sh
source colab_venv/bin/activate
```

### Step 2: GCS Authentication
```bash
export GOOGLE_APPLICATION_CREDENTIALS="/content/your-key.json"
gsutil ls gs://iti123storage/
```

### Step 3: Run Scripts Directly

Instead of notebooks, run scripts:

```bash
# Download videos
gsutil -m rsync -r gs://iti123storage/videos/clips/ data/videos/clips/

# Extract poses
python scripts/extract_poses_parallel.py \
    --video-dir data/videos/clips \
    --output-dir data/processed/poses \
    --model-complexity 1 \
    --target-fps 20 \
    --num-workers 4

# Check status
bash scripts/check_extraction_status.sh

# Create metadata if needed
python scripts/create_metadata_from_poses.py

# Run validation
python scripts/validate_phase2.py --sample-size 100

# Run feature selection
python scripts/run_feature_selection.py
```

### Step 4: Use Smaller Notebooks for Training

Open individual phase notebooks:
- `phase3_model_training_colab.ipynb` (smaller, more focused)
- `phase4_production_integration_colab.ipynb`

---

## Quick Commands Reference

### Check Current Status
```bash
# In Colab
!bash scripts/check_extraction_status.sh
```

### Run Complete Workflow (Command Line)
```bash
# All-in-one script
!bash scripts/colab_phase2_validation.sh
```

---

## File Sizes

| Notebook | Size | Recommended For |
|----------|------|-----------------|
| complete_workflow_colab.ipynb | 33KB | First-time users (may have issues in Colab) |
| phase2_validation_colab.ipynb | 29KB | Phase 2 only |
| phase3_model_training_colab.ipynb | 25KB | Phase 3 only |
| phase4_production_integration_colab.ipynb | 22KB | Phase 4 only |

**If complete_workflow_colab.ipynb has issues**, use the phase-specific notebooks or run scripts directly.

---

## Support

If you continue having issues:

1. **Check Colab status:** https://status.cloud.google.com/
2. **Try phase-specific notebooks** instead of complete workflow
3. **Run scripts directly** from Colab terminal
4. **Use the automated script:** `bash scripts/colab_phase2_validation.sh`

The scripts provide the same functionality as notebooks but may be more reliable in Colab.
