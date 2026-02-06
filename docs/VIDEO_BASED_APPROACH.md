# Video-Based Badminton Shot Classification

**Decision:** Switch from pose-only to video-based approach
**Reason:** Pose similarity 0.975-0.994 proves body position alone cannot distinguish shots
**Expected:** 70-80% accuracy on 5 classes (vs 38% with pose-only)

---

## Why Video Works Better

**What video captures that pose doesn't:**
- ✅ Shuttle trajectory (visible in frames)
- ✅ Racket speed blur (motion blur indicates power)
- ✅ Contact point (shuttle position at impact)
- ✅ Follow-through motion (temporal dynamics)
- ✅ Shuttle deceleration (visible as it slows/falls)

**Research backing:**
- SlowFast + Siamese: 83.08% accuracy (your research)
- 2D-CNN + LSTM: 90.9% accuracy (your research)
- Both used **video frames**, not pose

---

## Architecture Options

### Option A: **SlowFast Network** (Recommended)

**What it is:**
- Dual-pathway 3D CNN
- Fast pathway: high frame rate, low channels (motion)
- Slow pathway: low frame rate, high channels (appearance)

**Pros:**
- State-of-the-art for action recognition
- Pretrained on Kinetics-400
- Efficient (fewer frames needed)
- Your research showed 83% accuracy

**Cons:**
- More complex architecture
- Requires understanding dual-pathway concept

**Expected accuracy:** 75-85%

### Option B: **X3D** (Efficient Alternative)

**What it is:**
- Efficient 3D CNN family
- Designed for mobile/edge deployment
- Multiple sizes (XS, S, M, L)

**Pros:**
- Very efficient (X3D-S has only 3.8M params)
- Fast training and inference
- Pretrained models available
- Good for your dataset size (18K samples)

**Cons:**
- Slightly lower accuracy than SlowFast

**Expected accuracy:** 70-80%

### Option C: **2D CNN + LSTM** (Simplest)

**What it is:**
- ResNet/EfficientNet extracts frame features
- LSTM aggregates temporal information

**Pros:**
- Simple to understand and implement
- Can use any 2D CNN backbone
- Your research showed 90.9% accuracy!
- Works well with limited data

**Cons:**
- Processes frames sequentially (slower)
- May miss fine temporal details

**Expected accuracy:** 70-78%

---

## Recommended: X3D (Best Balance)

**Why X3D:**
1. Efficient enough for your compute
2. Good enough accuracy (70-80%)
3. Simpler than SlowFast
4. Pretrained models available
5. Fast to train (~3-4 hours on Colab GPU)

---

## Implementation Plan

### Phase 1: Data Preparation (2-3 hours)

#### 1.1 Video Clip Format

You already have video clips! Just need to ensure format:

```python
# Verify clip format
import cv2
from pathlib import Path

clips_dir = Path('features/clips')
sample_clip = list(clips_dir.glob('*.mp4'))[0]

cap = cv2.VideoCapture(str(sample_clip))
fps = cap.get(cv2.CAP_PROP_FPS)
frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

print(f"Sample clip: {sample_clip.name}")
print(f"  FPS: {fps}")
print(f"  Frames: {frame_count}")
print(f"  Resolution: {width}x{height}")
print(f"  Duration: {frame_count/fps:.2f}s")

cap.release()
```

**Expected:**
- FPS: 30
- Frames: 45-150 (1.5-5 seconds)
- Resolution: 600x800 (from ROI extraction)

#### 1.2 Frame Sampling Strategy

**For X3D, sample 16 frames uniformly:**

```python
import cv2
import numpy as np
import torch

def load_video_clip(video_path, num_frames=16, target_size=(224, 224)):
    """Load video and sample frames uniformly

    Args:
        video_path: Path to video file
        num_frames: Number of frames to sample (16 for X3D)
        target_size: (H, W) to resize frames

    Returns:
        frames: (num_frames, H, W, 3) numpy array
    """
    cap = cv2.VideoCapture(str(video_path))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # Sample frame indices uniformly
    if total_frames >= num_frames:
        indices = np.linspace(0, total_frames - 1, num_frames, dtype=int)
    else:
        # Repeat frames if video too short
        indices = np.linspace(0, total_frames - 1, total_frames, dtype=int)
        indices = np.pad(indices, (0, num_frames - total_frames), mode='edge')

    frames = []
    for idx in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        if ret:
            # Convert BGR to RGB
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            # Resize
            frame = cv2.resize(frame, target_size)
            frames.append(frame)

    cap.release()

    frames = np.stack(frames)  # (num_frames, H, W, 3)
    return frames

# Test
sample_frames = load_video_clip(sample_clip, num_frames=16)
print(f"Loaded frames shape: {sample_frames.shape}")
print(f"  Expected: (16, 224, 224, 3)")
```

#### 1.3 Dataset Class

```python
import torch
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
import pandas as pd

class BadmintonVideoDataset(Dataset):
    """Dataset for video-based classification"""

    def __init__(self, df, clips_dir, label_to_idx, num_frames=16,
                 target_size=(224, 224), augment=False):
        self.df = df.reset_index(drop=True)
        self.clips_dir = Path(clips_dir)
        self.label_to_idx = label_to_idx
        self.num_frames = num_frames
        self.target_size = target_size
        self.augment = augment

        # Video normalization (ImageNet stats for pretrained models)
        self.mean = np.array([0.45, 0.45, 0.45])  # RGB
        self.std = np.array([0.225, 0.225, 0.225])

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        # Load video
        video_path = self.clips_dir / f"{row['video_id']}.mp4"
        frames = load_video_clip(video_path, self.num_frames, self.target_size)

        # Augmentation (if training)
        if self.augment:
            frames = self.augment_video(frames)

        # Normalize to [0, 1]
        frames = frames.astype(np.float32) / 255.0

        # Normalize with ImageNet stats
        frames = (frames - self.mean) / self.std

        # Convert to tensor: (C, T, H, W) for 3D CNN
        frames_tensor = torch.from_numpy(frames).permute(3, 0, 1, 2).float()

        # Label
        label = self.label_to_idx[row['shot_type']]

        return {
            'video': frames_tensor,
            'label': label,
            'video_id': row['video_id']
        }

    def augment_video(self, frames):
        """Simple video augmentation"""
        # Random horizontal flip
        if np.random.rand() > 0.5:
            frames = np.flip(frames, axis=2).copy()

        # Random brightness adjustment
        if np.random.rand() > 0.5:
            brightness_factor = np.random.uniform(0.8, 1.2)
            frames = np.clip(frames * brightness_factor, 0, 255)

        return frames

# Create datasets
train_dataset_video = BadmintonVideoDataset(
    train_df,
    'features/clips',
    label_to_idx,
    num_frames=16,
    augment=True
)

val_dataset_video = BadmintonVideoDataset(
    val_df,
    'features/clips',
    label_to_idx,
    num_frames=16,
    augment=False
)

test_dataset_video = BadmintonVideoDataset(
    test_df,
    'features/clips',
    label_to_idx,
    num_frames=16,
    augment=False
)

# Create dataloaders
BATCH_SIZE = 16  # Smaller batch size for video (memory intensive)

train_loader_video = DataLoader(
    train_dataset_video,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=2,
    pin_memory=True
)

val_loader_video = DataLoader(
    val_dataset_video,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=2,
    pin_memory=True
)

test_loader_video = DataLoader(
    test_dataset_video,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=2,
    pin_memory=True
)

print(f"Video datasets created:")
print(f"  Train: {len(train_dataset_video)} samples")
print(f"  Val:   {len(val_dataset_video)} samples")
print(f"  Test:  {len(test_dataset_video)} samples")

# Test loading
sample = train_dataset_video[0]
print(f"\nSample video shape: {sample['video'].shape}")
print(f"  Expected: (3, 16, 224, 224) = (C, T, H, W)")
```

---

### Phase 2: Model Implementation (1 hour)

#### Option 2A: Use PyTorch Video Models (Easiest)

```bash
# Install pytorchvideo
!pip install pytorchvideo
```

```python
import torch
import torch.nn as nn
from pytorchvideo.models import x3d

class X3D_Badminton(nn.Module):
    """X3D model for badminton shot classification"""

    def __init__(self, num_classes=5, model_size='s', pretrained=True):
        super().__init__()

        # Load pretrained X3D
        if model_size == 'xs':
            self.backbone = torch.hub.load('facebookresearch/pytorchvideo', 'x3d_xs', pretrained=pretrained)
        elif model_size == 's':
            self.backbone = torch.hub.load('facebookresearch/pytorchvideo', 'x3d_s', pretrained=pretrained)
        elif model_size == 'm':
            self.backbone = torch.hub.load('facebookresearch/pytorchvideo', 'x3d_m', pretrained=pretrained)

        # Replace final classification layer
        # X3D has different output dims based on size
        if model_size == 'xs':
            in_features = 2048
        elif model_size == 's':
            in_features = 2048
        elif model_size == 'm':
            in_features = 2048

        self.backbone.blocks[5].proj = nn.Linear(in_features, num_classes)

    def forward(self, x):
        # x: (B, C, T, H, W)
        return self.backbone(x)

# Create model
model_x3d = X3D_Badminton(num_classes=5, model_size='s', pretrained=True).to(device)

total_params = sum(p.numel() for p in model_x3d.parameters())
print(f"X3D-S parameters: {total_params:,}")
print(f"Samples/param ratio: {len(train_dataset_video)/total_params:.2f}")

# Test forward pass
sample_batch = next(iter(train_loader_video))
sample_video = sample_batch['video'].to(device)
output = model_x3d(sample_video)
print(f"\nOutput shape: {output.shape}")
print(f"  Expected: ({BATCH_SIZE}, 5)")
```

#### Option 2B: Simple 2D CNN + LSTM (If X3D too complex)

```python
from torchvision import models

class CNN_LSTM_Badminton(nn.Module):
    """ResNet18 + LSTM for badminton classification"""

    def __init__(self, num_classes=5, hidden_size=256, num_layers=2, dropout=0.5):
        super().__init__()

        # CNN backbone (pretrained ResNet18)
        resnet = models.resnet18(pretrained=True)
        # Remove final FC layer
        self.cnn = nn.Sequential(*list(resnet.children())[:-1])

        # LSTM temporal aggregation
        self.lstm = nn.LSTM(
            input_size=512,  # ResNet18 feature dim
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=True
        )

        # Classifier
        self.fc = nn.Linear(hidden_size * 2, num_classes)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # x: (B, C, T, H, W)
        B, C, T, H, W = x.shape

        # Reshape to (B*T, C, H, W) to process frames
        x = x.permute(0, 2, 1, 3, 4).contiguous()  # (B, T, C, H, W)
        x = x.view(B * T, C, H, W)

        # Extract CNN features
        features = self.cnn(x)  # (B*T, 512, 1, 1)
        features = features.view(B, T, -1)  # (B, T, 512)

        # LSTM temporal modeling
        lstm_out, _ = self.lstm(features)  # (B, T, hidden_size*2)

        # Take last hidden state
        last_hidden = lstm_out[:, -1, :]  # (B, hidden_size*2)

        # Classify
        output = self.dropout(last_hidden)
        output = self.fc(output)

        return output

# Create model
model_cnn_lstm = CNN_LSTM_Badminton(num_classes=5, hidden_size=256).to(device)

total_params = sum(p.numel() for p in model_cnn_lstm.parameters())
print(f"CNN-LSTM parameters: {total_params:,}")

# Test
output = model_cnn_lstm(sample_video)
print(f"Output shape: {output.shape}")
```

---

### Phase 3: Training Configuration (30 minutes)

```python
from torch.optim import Adam, AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR

# Training config for video models
CONFIG_VIDEO = {
    'num_epochs': 50,
    'learning_rate': 0.0001,  # Lower LR for pretrained models
    'weight_decay': 0.0001,
    'early_stopping_patience': 15,
    'scheduler': 'cosine',
    'gradient_clip': 1.0,
    'focal_loss_gamma': 2.0,
}

# Class weights (same as before)
class_counts = train_df['shot_type'].value_counts()
total_samples = len(train_df)

class_weights = torch.tensor([
    np.sqrt(total_samples / class_counts[SHOT_TYPES[i]])
    for i in range(len(SHOT_TYPES))
]).float().to(device)

class_weights = class_weights / class_weights.sum() * len(SHOT_TYPES)

print(f"Class weights:")
for i, shot in enumerate(SHOT_TYPES):
    print(f"  {shot}: {class_weights[i]:.3f}")

# Loss function
criterion_video = FocalLoss(alpha=class_weights, gamma=CONFIG_VIDEO['focal_loss_gamma'])

print(f"\nTraining configuration:")
for key, value in CONFIG_VIDEO.items():
    print(f"  {key}: {value}")
```

---

### Phase 4: Training (3-4 hours on Colab GPU)

```python
# Choose model (X3D recommended)
model = model_x3d  # or model_cnn_lstm

# Optimizer
optimizer = AdamW(model.parameters(), lr=CONFIG_VIDEO['learning_rate'],
                  weight_decay=CONFIG_VIDEO['weight_decay'])

# Scheduler
scheduler = CosineAnnealingLR(optimizer, T_max=CONFIG_VIDEO['num_epochs'])

print("Starting video model training...")
print("="*80)

# Train (reuse train_model function from before, just different data)
model_trained, history_video = train_model(
    model,
    train_loader_video,
    val_loader_video,
    criterion_video,
    optimizer,
    scheduler,
    CONFIG_VIDEO,
    'X3D_Badminton'  # or 'CNN_LSTM_Badminton'
)

print("\n✓ Video model training complete!")
```

---

### Phase 5: Evaluation & Comparison

```python
# Evaluate video model
def evaluate_video_model(model, test_loader, model_name):
    """Evaluate video model"""
    print(f"\nEvaluating {model_name}...")
    print("="*80)

    model.eval()
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for batch in tqdm(test_loader, desc='Testing'):
            videos = batch['video'].to(device)
            labels = batch['label']

            outputs = model(videos)
            preds = torch.argmax(outputs, dim=1).cpu()

            all_preds.extend(preds.numpy())
            all_labels.extend(labels.numpy())

    # Metrics
    test_acc = accuracy_score(all_labels, all_preds)
    test_f1 = f1_score(all_labels, all_preds, average='macro')

    print(f"\nTest Accuracy: {test_acc:.3f}")
    print(f"Test F1 (Macro): {test_f1:.3f}")

    print(f"\nClassification Report:")
    print(classification_report(all_labels, all_preds, target_names=SHOT_TYPES))

    # Confusion matrix
    cm = confusion_matrix(all_labels, all_preds)
    cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    return {
        'accuracy': test_acc,
        'f1_macro': test_f1,
        'confusion_matrix': cm,
        'confusion_matrix_normalized': cm_normalized
    }

# Evaluate
results_video = evaluate_video_model(model_trained, test_loader_video, 'X3D')

# Compare with pose-based
print("\n" + "="*80)
print("FINAL COMPARISON")
print("="*80)

print(f"\nPose-only (5-class):  {0.38:.1%}")
print(f"Pose-only (2-class):  {0.605:.1%}")
print(f"Video-based (5-class): {results_video['accuracy']:.1%}")

print(f"\nImprovement: {results_video['accuracy'] - 0.38:.1%} absolute")
print(f"Relative improvement: {(results_video['accuracy'] / 0.38 - 1) * 100:.0f}%")
```

---

## Expected Results

### Conservative Estimate
- **Accuracy: 70-75%**
- Per-class performance:
  - Drop: 75-80% (most samples)
  - Lift: 70-75%
  - Smash: 70-75%
  - Clear: 65-70%
  - Drive: 50-60% (still limited by sample size)

### Optimistic Estimate (with tuning)
- **Accuracy: 75-80%**
- Matches research paper results

### Comparison
```
Approach              | Accuracy | F1 (Macro) | Training Time
----------------------|----------|------------|---------------
Pose-only (5-class)   | 38%      | 34%        | 2 hours
Pose-only (2-class)   | 60.5%    | 58%        | 30 minutes
Video-based (5-class) | 70-80%   | 68-78%     | 3-4 hours
```

---

## Advantages of Video Approach

**vs Pose-only:**
1. ✅ Captures shuttle trajectory (implicitly)
2. ✅ Captures racket motion blur (speed indicator)
3. ✅ Captures contact point
4. ✅ Captures temporal dynamics
5. ✅ **No manual feature engineering needed**

**vs Shuttle tracking:**
1. ✅ Simpler implementation (no YOLOv8 training)
2. ✅ More robust (works even if shuttle detection fails)
3. ✅ Pretrained models available
4. ⚠️  Higher compute cost
5. ⚠️  Less interpretable

---

## Troubleshooting

### Issue: OOM (Out of Memory)

**Solutions:**
```python
# 1. Reduce batch size
BATCH_SIZE = 8  # Instead of 16

# 2. Use smaller model
model = X3D_Badminton(model_size='xs')  # Instead of 's'

# 3. Use gradient checkpointing
model.backbone.gradient_checkpointing = True

# 4. Mixed precision training
from torch.cuda.amp import autocast, GradScaler

scaler = GradScaler()

# In training loop:
with autocast():
    outputs = model(videos)
    loss = criterion(outputs, labels)

scaler.scale(loss).backward()
scaler.step(optimizer)
scaler.update()
```

### Issue: Video loading slow

**Solutions:**
```python
# 1. Pre-extract frames to disk (faster loading)
# 2. Increase num_workers
train_loader_video = DataLoader(..., num_workers=4)

# 3. Use GPU video decoding (if available)
# 4. Resize videos to 224x224 beforehand
```

### Issue: Training not converging

**Solutions:**
```python
# 1. Lower learning rate
CONFIG_VIDEO['learning_rate'] = 0.00005

# 2. Unfreeze backbone gradually
# Freeze backbone initially
for param in model.backbone.parameters():
    param.requires_grad = False

# Train classifier only for 10 epochs, then unfreeze

# 3. Use label smoothing
criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
```

---

## Next Steps After Success

Once you achieve 70-80% with video:

### 1. Error Analysis
- Which samples are still misclassified?
- Is it data quality or fundamental ambiguity?

### 2. Model Ensemble
- Combine video + pose models
- Expected: +3-5% accuracy boost

### 3. Temporal Segmentation
- Detect shot boundaries automatically
- End-to-end video analysis

### 4. Deployment
- Convert to ONNX for inference
- Optimize for real-time processing

---

## Summary

**Current status:** Pose-only hits 38% on 5-class (60% on 2-class)

**Root cause:** Pose similarity 0.975-0.994 - body positions nearly identical

**Solution:** Video-based approach captures shuttle trajectory implicitly

**Expected outcome:** 70-80% accuracy on 5-class

**Implementation time:** 1-2 days

**Next:** Start with Phase 1 (data preparation)

---

**Ready to implement? Start with Phase 1 - Data Preparation!**
