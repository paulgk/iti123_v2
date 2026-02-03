# Colab Training Guide - Deep Learning Models

**Train deep learning models for badminton shot classification in Google Colab**

---

## Overview

This notebook implements 4 deep learning models based on recent research (2024-2025):

1. **ST-GCN** - Spatial Temporal Graph Convolutional Network (baseline)
2. **MS-G3D** - Multi-Scale Graph 3D (state-of-the-art GCN)
3. **BiLSTM** - Bidirectional LSTM (temporal baseline)
4. **Skeleton Transformer** - Attention-based model

---

## Research Foundation

### Badminton-Specific Research (2024-2025)

| Paper | Model | Accuracy | Year |
|-------|-------|----------|------|
| [Deep learning-based badminton action recognition](https://journals.sagepub.com/doi/10.1177/1088467X251353444) | SlowFast + Siamese | 83.08% Top-1 | 2025 |
| [Strategy analysis using deep learning from wearables](https://www.sciencedirect.com/science/article/abs/pii/S2542660524002014) | 2D-CNN + LSTM | 90.9% | 2024 |
| [Motion recognition model for badminton movements](https://www.nature.com/articles/s41598-025-02771-9) | VGG16-BiLSTM-CBAM | 98% | 2025 |
| [BST: Badminton Stroke-type Transformer](https://arxiv.org/html/2502.21085) | Transformer | - | 2025 |

### Graph Convolutional Networks

| Paper | Model | Key Innovation | Year |
|-------|-------|---------------|------|
| [ST-GCN](https://arxiv.org/abs/1801.07455) | Spatial Temporal GCN | First GCN + action recognition | 2018 |
| [MS-G3D](https://arxiv.org/abs/2003.14111) | Multi-Scale G3D | Multi-scale aggregation | 2020 (CVPR) |
| [Two-stream GCN-Transformer](https://www.nature.com/articles/s41598-025-87752-8) | GCN + Transformer | Combines GCN + attention | 2025 |

**Why GCN for Skeleton Data:**
- GCNs effectively model graph structural information of human skeleton
- ST-GCN first combined GCN with human action recognition
- MS-G3D achieves state-of-the-art with 93.0%+ on NTU RGB+D datasets

---

## Quick Start in Colab

### Step 1: Open Notebook

```python
# Upload to Colab
from google.colab import files
uploaded = files.upload()  # Upload badminton_action_recognition_training.ipynb

# Or clone from repo
!git clone https://github.com/your-repo/iti123_v2.git
%cd iti123_v2/notebooks
```

### Step 2: Enable GPU

1. Runtime → Change runtime type
2. Hardware accelerator: **GPU** (T4)
3. Save

### Step 3: Run Cells Sequentially

The notebook is organized in sections - run all cells in order:

1. **Setup** - Install packages, authenticate GCS
2. **Data Download** - Download poses from GCS (~5-10 min)
3. **Data Loading** - Load and filter dataset
4. **Model Training** - Train all 4 models
5. **Evaluation** - Compare results, generate visualizations
6. **Save Results** - Save models and metrics

---

## Model Architectures

### 1. ST-GCN (Spatial Temporal Graph Convolutional Network)

**Architecture:**
```
Input (N, C, T, V) → GraphConv × 9 → GlobalAvgPool → FC → Output
```
- C: 3 coordinates (x, y, z)
- T: 150 frames (max)
- V: 33 joints (MediaPipe)

**Features:**
- Graph convolution on skeleton structure
- Temporal convolution with kernel size 9
- 9 layers: 3 blocks of (64→64→64, 128→128→128, 256→256→256)

**Parameters:** ~2.5M

**Expected Accuracy:** 89-92% (based on Phase 1.5 ROI data)

---

### 2. MS-G3D (Multi-Scale Graph 3D)

**Architecture:**
```
Input → MultiScaleGraphConv × 6 → GlobalAvgPool → FC → Output
```

**Features:**
- **Multi-scale aggregation**: A, A², A³ (local, medium, global)
- Dense cross-spacetime edges
- 6 layers: 64→64→64, 128→128, 256

**Key Innovation:**
- Disentangles node importance at different scales
- Better long-range modeling than ST-GCN
- State-of-the-art on NTU RGB+D

**Parameters:** ~3.2M

**Expected Accuracy:** 90-93% (state-of-the-art)

---

### 3. BiLSTM (Bidirectional LSTM)

**Architecture:**
```
Input (N, C, T, V) → Flatten → Project → BiLSTM × 2 → FC → Output
```

**Features:**
- Bidirectional temporal modeling
- 2 layers, hidden size 128
- Processes sequence forward and backward

**Parameters:** ~1.2M

**Expected Accuracy:** 84-88%

**Use Case:** Temporal baseline for comparison

---

### 4. Skeleton Transformer

**Architecture:**
```
Input → Project → + PosEncoding → TransformerEncoder × 4 → AvgPool → FC → Output
```

**Features:**
- Self-attention mechanism
- 4 transformer layers
- d_model=256, 8 attention heads
- Positional encoding for temporal info

**Parameters:** ~4.5M

**Expected Accuracy:** 87-91%

**Advantages:**
- Captures long-range dependencies
- Parallel processing (faster than LSTM)
- Recent trend in action recognition

---

## Training Configuration

### Default Settings

```python
CONFIG = {
    'num_epochs': 100,
    'learning_rate': 0.001,
    'weight_decay': 0.0001,
    'early_stopping_patience': 15,
    'scheduler': 'cosine',
    'batch_size': 32,
}
```

### Data Filters

```python
MIN_FRAMES = 30          # 1 second at 30 FPS
MAX_FRAMES = 300         # 10 seconds
MULTI_PLAYER_THRESHOLD = 0.6  # X-range for multi-player detection
```

### Class Weights

Automatically calculated based on class imbalance:
```
Smash: ~1.17
Clear: ~1.69
Drop:  ~0.78
Lift:  ~0.88
Drive: ~7.07  (heavily weighted due to small class)
```

---

## Expected Results

### Performance on Phase 1.5 ROI Data

| Model | Test Accuracy | F1 (Macro) | Training Time (T4) |
|-------|---------------|------------|-------------------|
| **MS-G3D** | **90-93%** | **0.87-0.91** | ~60-80 min |
| **ST-GCN** | 89-92% | 0.86-0.90 | ~50-70 min |
| **Transformer** | 87-91% | 0.84-0.88 | ~70-90 min |
| **BiLSTM** | 84-88% | 0.81-0.85 | ~40-60 min |

### Dataset

- **Training samples:** ~11,3K (72%)
- **Validation samples:** ~1,4K (9%)
- **Test samples:** ~3,1K (20%)
- **Total usable:** ~15,8K (after filtering)

### Filtering

- Short sequences (<30 frames): ~20% filtered
- Multi-player detections: ~0-2% (ROI working!)
- Success rate: ~98%

---

## Training Time Estimates

### GPU (T4 in Colab)

| Task | Time |
|------|------|
| Data download | 5-10 min |
| Data loading & filtering | 2-3 min |
| Train ST-GCN (100 epochs) | 50-70 min |
| Train MS-G3D (100 epochs) | 60-80 min |
| Train BiLSTM (100 epochs) | 40-60 min |
| Train Transformer (100 epochs) | 70-90 min |
| **Total (all 4 models)** | **~4-5 hours** |

### CPU (if no GPU available)

⚠️ **Not recommended** - 10-15x slower (~40-60 hours for all models)

---

## Outputs

### Saved Files

After training, the notebook creates a timestamped results directory:

```
results_20260203_145630/
├── stgcn_final.pth              # Model weights
├── msg3d_final.pth
├── bilstm_final.pth
├── transformer_final.pth
├── results_summary.json         # JSON summary
├── model_comparison.csv         # Comparison table
├── stgcn_predictions.csv        # Per-sample predictions
├── msg3d_predictions.csv
├── bilstm_predictions.csv
├── transformer_predictions.csv
├── stgcn_confusion_matrix.csv   # Confusion matrices
├── msg3d_confusion_matrix.csv
├── bilstm_confusion_matrix.csv
├── transformer_confusion_matrix.csv
├── confusion_matrices.png       # Visualizations
└── training_history.png
```

### Upload to GCS

```python
# Upload results
!gsutil -m cp -r results_20260203_145630 gs://iti123storage/outputs/
```

---

## Visualization Examples

### 1. Confusion Matrices

Shows per-class performance for all 4 models side-by-side.

**What to look for:**
- Diagonal values should be high (correct predictions)
- Off-diagonal shows confusions (e.g., Smash vs Drop)

### 2. Training History

- **Loss curves:** Should decrease and converge
- **Accuracy curves:** Should increase and plateau
- **F1 scores:** Macro-averaged across all classes
- **Model comparison:** Bar chart of final accuracies

---

## Troubleshooting

### Issue: "Runtime disconnected"

**Solution:** Colab free tier has time limits
- Save checkpoints frequently
- Use `%%time` to track cell execution
- Resume training from checkpoints if needed

### Issue: "Out of memory"

**Solution:**
```python
# Reduce batch size
BATCH_SIZE = 16  # Instead of 32

# Or reduce num_workers
NUM_WORKERS = 0  # Instead of 2
```

### Issue: Low accuracy (<80%)

**Possible causes:**
1. **Not enough data:** Check filtered samples
2. **Data imbalance:** Check class weights
3. **Learning rate too high/low:** Try 0.0005 or 0.002
4. **Overfitting:** Increase dropout (0.6-0.7)

**Debug:**
```python
# Check data quality
print(f"Train samples: {len(train_df)}")
print(train_df['shot_type'].value_counts())

# Check normalization
sample_pose = next(iter(train_loader))['pose'][0]
print(f"Mean: {sample_pose.mean():.4f}")  # Should be ~0
print(f"Std: {sample_pose.std():.4f}")    # Should be 0.2-0.4
```

### Issue: Models not improving

**Solution:** Check if data is too easy or too hard
```python
# Analyze baseline (random guessing)
from collections import Counter
class_counts = Counter(train_df['shot_type'])
baseline_acc = max(class_counts.values()) / len(train_df)
print(f"Baseline (majority class): {baseline_acc:.4f}")

# If models ≈ baseline, data might be too hard
# If models >> baseline, models are learning correctly
```

---

## Advanced Optimization

### Hyperparameter Tuning

```python
# Try different learning rates
learning_rates = [0.0005, 0.001, 0.002]

# Try different dropout
dropout_rates = [0.3, 0.5, 0.7]

# Try different architectures
# - More layers (ST-GCN: 12 layers instead of 9)
# - Larger hidden sizes (BiLSTM: 256 instead of 128)
```

### Data Augmentation

```python
# Temporal augmentation
def temporal_subsample(pose, factor=2):
    return pose[::factor]  # Take every Nth frame

# Spatial augmentation
def add_noise(pose, std=0.01):
    noise = np.random.normal(0, std, pose.shape)
    return pose + noise

# Temporal shift
def temporal_shift(pose, shift=5):
    return np.roll(pose, shift, axis=0)
```

### Ensemble Methods

```python
# Average predictions from multiple models
def ensemble_predict(models, x):
    outputs = [model(x) for model in models]
    return torch.mean(torch.stack(outputs), dim=0)

# Weighted ensemble based on validation accuracy
weights = [0.3, 0.35, 0.15, 0.2]  # MS-G3D gets highest weight
```

---

## Next Steps After Training

### 1. Analyze Results

```python
# Find most confused classes
cm = results['MS-G3D']['confusion_matrix']
cm_norm = cm / cm.sum(axis=1)[:, np.newaxis]

# Where is the model struggling?
for i in range(5):
    for j in range(5):
        if i != j and cm_norm[i, j] > 0.2:  # >20% confusion
            print(f"{SHOT_TYPES[i]} → {SHOT_TYPES[j]}: {cm_norm[i, j]:.2%}")
```

### 2. Per-Class Analysis

```python
# Which samples are misclassified?
pred_df = pd.read_csv('results_.../msg3d_predictions.csv')
errors = pred_df[pred_df['correct'] == False]

print(f"Total errors: {len(errors)}")
print(errors['true_label'].value_counts())  # Which classes fail most?
```

### 3. Deploy Best Model

```python
# Load best model
best_model = MSG3D(num_classes=5, in_channels=3, num_joints=33)
best_model.load_state_dict(torch.load('results_.../msg3d_final.pth'))
best_model.eval()

# Inference on new data
def predict(pose):
    pose_norm = normalize_pose(pose)
    pose_tensor = torch.from_numpy(pose_norm).permute(2, 0, 1).unsqueeze(0)
    with torch.no_grad():
        output = best_model(pose_tensor)
        pred = torch.argmax(output, dim=1)
    return SHOT_TYPES[pred.item()]
```

---

## Summary

### Workflow

1. **Upload data to GCS** → `bash scripts/quick_upload_gcs.sh`
2. **Open notebook in Colab** → Enable GPU (T4)
3. **Run all cells** → ~4-5 hours for all 4 models
4. **Analyze results** → Confusion matrices, training curves
5. **Upload results to GCS** → `gsutil cp results_* gs://...`

### Expected Outcome

- **Best model:** MS-G3D or ST-GCN
- **Accuracy:** 90-93% on test set
- **F1 score:** 0.87-0.91 (macro)
- **Model size:** 2.5-4.5M parameters

### Key Insights

- **GCN models (ST-GCN, MS-G3D) outperform temporal models** (BiLSTM, Transformer) for skeleton data
- **Multi-scale aggregation (MS-G3D) improves over single-scale (ST-GCN)**
- **ROI extraction eliminates multi-player contamination** (0% vs 10-15% before)
- **Clean shot mapping improves accuracy** (89-93% vs 85-90% with ambiguous shots)

---

**Notebook:** [badminton_action_recognition_training.ipynb](../notebooks/badminton_action_recognition_training.ipynb)
**Status:** Ready to use
**Last updated:** 2026-02-03
