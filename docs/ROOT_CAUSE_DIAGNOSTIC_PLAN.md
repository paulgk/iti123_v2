# Root Cause Diagnostic Plan: <40% Accuracy Investigation

**Status:** Models performing at near-random levels despite all fixes
**Current:** Lightweight models ~35-40% accuracy (random baseline: 20%)
**Expected:** 60-70% with lightweight on 18K samples
**Gap:** 20-30% unexplained performance loss

---

## Diagnostic Framework

We'll investigate in order of likelihood and impact:

1. **Data Quality Issues** (60% probability) - Corrupted poses, label noise, wrong mappings
2. **Feature Discriminability** (25% probability) - Poses too similar between classes
3. **Model/Training Issues** (10% probability) - Implementation bugs, numerical instability
4. **Fundamental Limitations** (5% probability) - Task impossible with pose alone

---

## Phase 1: Training Curve Analysis (30 minutes)

**Goal:** Determine if models are learning at all

### 1.1 Check Training vs Validation Gap

**What to compute:**
```python
# From training history
best_train_acc = max(history['train_acc'])
best_val_acc = max(history['val_acc'])
final_train_acc = history['train_acc'][-1]
final_val_acc = history['val_acc'][-1]
gap = best_train_acc - best_val_acc

print(f"Best train accuracy: {best_train_acc:.3f}")
print(f"Best val accuracy: {best_val_acc:.3f}")
print(f"Train-val gap: {gap:.3f}")
print(f"Final train: {final_train_acc:.3f}, Final val: {final_val_acc:.3f}")
```

**Interpretation:**

| Scenario | Train Acc | Val Acc | Gap | Diagnosis |
|----------|-----------|---------|-----|-----------|
| A | <45% | <40% | <8% | **Not learning** - Model/data issue |
| B | >55% | <40% | >15% | **Overfitting** - Still too complex |
| C | 40-50% | 35-40% | 5-10% | **Learning ceiling** - Feature/data quality issue |

**Action based on result:**
- **Scenario A** → Go to Phase 2 (Data Quality)
- **Scenario B** → Go to Phase 5 (Model Simplification)
- **Scenario C** → Go to Phase 3 (Feature Analysis)

### 1.2 Check Loss Convergence

**What to look for:**
```python
import matplotlib.pyplot as plt

plt.figure(figsize=(12, 4))

# Plot loss
plt.subplot(1, 2, 1)
plt.plot(history['train_loss'], label='Train')
plt.plot(history['val_loss'], label='Val')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Loss Curves')
plt.legend()
plt.grid(True)

# Plot accuracy
plt.subplot(1, 2, 2)
plt.plot(history['train_acc'], label='Train')
plt.plot(history['val_acc'], label='Val')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.title('Accuracy Curves')
plt.legend()
plt.grid(True)

plt.savefig('training_diagnostics.png')
plt.show()

# Check for anomalies
print("\nLoss anomalies:")
print(f"  NaN losses: {sum(np.isnan(history['train_loss']))}")
print(f"  Infinite losses: {sum(np.isinf(history['train_loss']))}")
print(f"  Loss explosions (>10): {sum(np.array(history['train_loss']) > 10)}")

print("\nConvergence:")
print(f"  Final train loss: {history['train_loss'][-1]:.4f}")
print(f"  Final val loss: {history['val_loss'][-1]:.4f}")
print(f"  Minimum val loss: {min(history['val_loss']):.4f} at epoch {np.argmin(history['val_loss'])}")
```

**Red flags:**
- ❌ Loss not decreasing after epoch 10
- ❌ Loss oscillating wildly (std > 0.3)
- ❌ Validation loss increasing from start
- ❌ NaN or Inf values

**If red flags found** → Go to Phase 4 (Implementation Check)

---

## Phase 2: Data Quality Investigation (2-3 hours)

**Goal:** Find corrupted data, label errors, or systematic issues

### 2.1 Pose Sequence Statistics

**Compute distributions:**
```python
import pickle
import numpy as np
from pathlib import Path
from collections import defaultdict

poses_dir = Path('data/poses')
metadata = pd.read_csv('data/metadata.csv')

stats = {
    'lengths': [],
    'x_ranges': [],
    'y_ranges': [],
    'z_ranges': [],
    'nan_counts': [],
    'zero_counts': [],
    'by_class': defaultdict(list)
}

print("Analyzing all pose sequences...")
for idx, row in metadata.iterrows():
    video_id = row['video_id']
    shot_type = row['shot_type']
    pose_file = poses_dir / f"{video_id}.pkl"

    if not pose_file.exists():
        continue

    with open(pose_file, 'rb') as f:
        pose = pickle.load(f)

    # Basic stats
    stats['lengths'].append(len(pose))
    stats['x_ranges'].append(pose[:,:,0].max() - pose[:,:,0].min())
    stats['y_ranges'].append(pose[:,:,1].max() - pose[:,:,1].min())
    stats['z_ranges'].append(pose[:,:,2].max() - pose[:,:,2].min())
    stats['nan_counts'].append(np.isnan(pose).sum())
    stats['zero_counts'].append((pose == 0).sum())

    # Per-class
    stats['by_class'][shot_type].append({
        'length': len(pose),
        'x_range': pose[:,:,0].max() - pose[:,:,0].min(),
        'video_id': video_id
    })

# Print summary
print("\n" + "="*80)
print("POSE SEQUENCE STATISTICS")
print("="*80)

print(f"\nSequence lengths:")
print(f"  Mean: {np.mean(stats['lengths']):.1f} frames")
print(f"  Median: {np.median(stats['lengths']):.1f} frames")
print(f"  Min: {np.min(stats['lengths'])} frames")
print(f"  Max: {np.max(stats['lengths'])} frames")
print(f"  Std: {np.std(stats['lengths']):.1f} frames")

print(f"\nSpatial ranges (after normalization):")
print(f"  X-range mean: {np.mean(stats['x_ranges']):.3f}")
print(f"  Y-range mean: {np.mean(stats['y_ranges']):.3f}")
print(f"  Z-range mean: {np.mean(stats['z_ranges']):.3f}")

print(f"\nData quality:")
print(f"  Samples with NaN: {sum(c > 0 for c in stats['nan_counts'])} ({sum(c > 0 for c in stats['nan_counts'])/len(stats['nan_counts'])*100:.1f}%)")
print(f"  Samples with >50% zeros: {sum(c > len(pose)*33*3*0.5 for c in stats['zero_counts'])}")

print(f"\nPer-class length distribution:")
for shot_type in ['Smash', 'Clear', 'Drop', 'Lift', 'Drive']:
    if shot_type in stats['by_class']:
        lengths = [s['length'] for s in stats['by_class'][shot_type]]
        print(f"  {shot_type:6s}: mean={np.mean(lengths):5.1f}, median={np.median(lengths):5.1f}, std={np.std(lengths):5.1f}")
```

**Red flags to look for:**
- ❌ **>5% samples with NaN values** → Pose extraction failed
- ❌ **Mean sequence length <40 frames** → Too short to capture action
- ❌ **Std deviation >60 frames** → Inconsistent clip lengths
- ❌ **X/Y/Z range <0.1** → Poses barely moving (static)
- ❌ **Per-class length huge variance** → Inconsistent labeling

### 2.2 Label Consistency Check

**Find mislabeled samples:**
```python
# Strategy: Find outliers in each class
from scipy.stats import zscore

print("\n" + "="*80)
print("OUTLIER DETECTION (per class)")
print("="*80)

outliers = []

for shot_type in ['Smash', 'Clear', 'Drop', 'Lift', 'Drive']:
    class_data = stats['by_class'][shot_type]
    if len(class_data) < 10:
        continue

    # Compute z-scores for length and x-range
    lengths = np.array([s['length'] for s in class_data])
    x_ranges = np.array([s['x_range'] for s in class_data])

    length_z = np.abs(zscore(lengths))
    xrange_z = np.abs(zscore(x_ranges))

    # Find outliers (z-score > 3)
    outlier_mask = (length_z > 3) | (xrange_z > 3)

    if outlier_mask.sum() > 0:
        print(f"\n{shot_type} outliers ({outlier_mask.sum()}/{len(class_data)}):")
        for i, is_outlier in enumerate(outlier_mask):
            if is_outlier:
                video_id = class_data[i]['video_id']
                length = class_data[i]['length']
                x_range = class_data[i]['x_range']
                print(f"  {video_id}: length={length} (z={length_z[i]:.1f}), x_range={x_range:.3f} (z={xrange_z[i]:.1f})")
                outliers.append({
                    'video_id': video_id,
                    'shot_type': shot_type,
                    'length': length,
                    'x_range': x_range,
                    'reason': 'statistical_outlier'
                })

print(f"\n\nTotal outliers found: {len(outliers)}")
print("Action: Manually inspect these samples (see Phase 2.4)")
```

### 2.3 Class Similarity Analysis

**Measure how similar classes are:**
```python
from sklearn.metrics.pairwise import cosine_similarity

print("\n" + "="*80)
print("INTER-CLASS SIMILARITY")
print("="*80)

# Compute mean pose for each class
class_means = {}

for shot_type in ['Smash', 'Clear', 'Drop', 'Lift', 'Drive']:
    class_poses = []
    for idx, row in metadata[metadata['shot_type'] == shot_type].iterrows():
        pose_file = poses_dir / f"{row['video_id']}.pkl"
        if pose_file.exists():
            with open(pose_file, 'rb') as f:
                pose = pickle.load(f)
            # Flatten and normalize
            pose_norm = normalize_pose(pose)
            pose_flat = pose_norm.flatten()
            class_poses.append(pose_flat[:1000])  # First 1000 dims for speed

    if class_poses:
        class_means[shot_type] = np.mean(class_poses, axis=0)

# Compute pairwise similarities
shot_types = ['Smash', 'Clear', 'Drop', 'Lift', 'Drive']
similarities = np.zeros((5, 5))

for i, shot1 in enumerate(shot_types):
    for j, shot2 in enumerate(shot_types):
        if shot1 in class_means and shot2 in class_means:
            sim = cosine_similarity([class_means[shot1]], [class_means[shot2]])[0, 0]
            similarities[i, j] = sim

print("\nCosine similarity between class means:")
print("      ", " ".join(f"{s:6s}" for s in shot_types))
for i, shot1 in enumerate(shot_types):
    print(f"{shot1:6s}", " ".join(f"{similarities[i, j]:.3f}" for j in range(5)))

print("\nInterpretation:")
print("  >0.9: Extremely similar (hard to distinguish)")
print("  0.7-0.9: Similar (requires fine-grained features)")
print("  <0.7: Distinct (should be learnable)")

# Find most confused pairs
confused_pairs = []
for i in range(5):
    for j in range(i+1, 5):
        if similarities[i, j] > 0.85:
            confused_pairs.append((shot_types[i], shot_types[j], similarities[i, j]))

if confused_pairs:
    print("\n⚠️  Highly similar class pairs:")
    for shot1, shot2, sim in sorted(confused_pairs, key=lambda x: -x[2]):
        print(f"  {shot1} ↔ {shot2}: {sim:.3f}")
```

**Red flags:**
- ❌ **Any pair >0.90 similarity** → Classes too similar to distinguish
- ❌ **All classes >0.80 similarity** → Poses fundamentally not discriminative
- ❌ **Drop/Lift >0.85** → These are notoriously similar in badminton

### 2.4 Manual Inspection Protocol

**Sample 20 videos (4 per class) for manual review:**
```python
import random

print("\n" + "="*80)
print("MANUAL INSPECTION SAMPLES")
print("="*80)

samples_to_inspect = []

# Sample from outliers first
for outlier in outliers[:10]:
    samples_to_inspect.append(outlier['video_id'])

# Sample random from each class
for shot_type in ['Smash', 'Clear', 'Drop', 'Lift', 'Drive']:
    class_samples = metadata[metadata['shot_type'] == shot_type]['video_id'].tolist()
    if len(class_samples) >= 4:
        random_samples = random.sample(class_samples, 4)
        samples_to_inspect.extend(random_samples)

# Create inspection checklist
print("\nInspect these samples:")
print("For each sample, check:")
print("  1. Does the video match the label?")
print("  2. Is the pose extraction clean (no multi-player, occlusion)?")
print("  3. Is the action complete (pre-contact to follow-through)?")
print("  4. Would YOU be able to classify this from pose alone?")
print()

for i, video_id in enumerate(samples_to_inspect[:20], 1):
    row = metadata[metadata['video_id'] == video_id].iloc[0]
    print(f"{i:2d}. {video_id} - Label: {row['shot_type']}")
    print(f"    Match: {row['match_id']}, Frame: {row['start_frame']}-{row['end_frame']}")
    print(f"    Video: features/clips/{video_id}.mp4")
    print(f"    Pose:  features/poses/{video_id}.pkl")
    print()

print("\nCreate inspection results file:")
print("  File: data/manual_inspection_results.csv")
print("  Columns: video_id, labeled_class, actual_class, pose_quality (good/bad), notes")
```

**What to record:**
- Mislabeled samples (actual class != label)
- Ambiguous samples (hard to tell even for human)
- Bad pose extraction (occlusion, multi-player)
- Incomplete actions (clip cut off)

---

## Phase 3: Feature Discriminability Analysis (1-2 hours)

**Goal:** Determine if poses contain enough information to distinguish classes

### 3.1 Pose Visualization

**Visualize representative samples from each class:**
```python
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def visualize_pose_sequence(pose, title="", max_frames=5):
    """Visualize first N frames of pose sequence"""
    fig = plt.figure(figsize=(15, 3))

    frames_to_show = min(max_frames, len(pose))
    frame_indices = np.linspace(0, len(pose)-1, frames_to_show, dtype=int)

    for i, frame_idx in enumerate(frame_indices):
        ax = fig.add_subplot(1, frames_to_show, i+1, projection='3d')

        # Plot skeleton
        frame = pose[frame_idx]
        ax.scatter(frame[:, 0], frame[:, 1], frame[:, 2], c='blue', s=20)

        # Draw connections (MediaPipe skeleton)
        connections = [
            (11, 13), (13, 15),  # Left arm
            (12, 14), (14, 16),  # Right arm
            (11, 23), (12, 24),  # Shoulders to hips
            (23, 25), (25, 27),  # Left leg
            (24, 26), (26, 28),  # Right leg
        ]

        for start, end in connections:
            ax.plot([frame[start, 0], frame[end, 0]],
                   [frame[start, 1], frame[end, 1]],
                   [frame[start, 2], frame[end, 2]], 'r-', linewidth=1)

        ax.set_title(f"Frame {frame_idx}")
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')

        # Set equal aspect ratio
        ax.set_box_aspect([1,1,1])

    fig.suptitle(title)
    plt.tight_layout()
    return fig

# Visualize one sample from each class
for shot_type in ['Smash', 'Clear', 'Drop', 'Lift', 'Drive']:
    sample = metadata[metadata['shot_type'] == shot_type].iloc[0]
    pose_file = poses_dir / f"{sample['video_id']}.pkl"

    with open(pose_file, 'rb') as f:
        pose = pickle.load(f)

    pose_norm = normalize_pose(pose)

    fig = visualize_pose_sequence(pose_norm,
                                  title=f"{shot_type} - {sample['video_id']}",
                                  max_frames=5)
    plt.savefig(f'pose_viz_{shot_type.lower()}.png', dpi=150)
    plt.close()

print("✓ Saved pose visualizations to pose_viz_*.png")
print("\nLook for:")
print("  - Are body positions visually distinct between classes?")
print("  - Is racket arm clearly extended/bent differently?")
print("  - Is there temporal progression (not static)?")
```

### 3.2 Feature Importance via PCA

**Check if poses cluster by class:**
```python
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt

# Collect all poses
all_poses = []
all_labels = []

for idx, row in metadata.sample(min(2000, len(metadata))).iterrows():
    pose_file = poses_dir / f"{row['video_id']}.pkl"
    if pose_file.exists():
        with open(pose_file, 'rb') as f:
            pose = pickle.load(f)
        pose_norm = normalize_pose(pose)
        # Use mean pose as feature
        pose_mean = pose_norm.mean(axis=0).flatten()
        all_poses.append(pose_mean)
        all_labels.append(row['shot_type'])

X = np.array(all_poses)
y = np.array(all_labels)

# PCA
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X)

# Plot
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
for shot_type in ['Smash', 'Clear', 'Drop', 'Lift', 'Drive']:
    mask = y == shot_type
    plt.scatter(X_pca[mask, 0], X_pca[mask, 1], label=shot_type, alpha=0.5, s=20)
plt.xlabel(f'PC1 ({pca.explained_variance_ratio_[0]*100:.1f}%)')
plt.ylabel(f'PC2 ({pca.explained_variance_ratio_[1]*100:.1f}%)')
plt.title('PCA: Pose Features')
plt.legend()
plt.grid(True, alpha=0.3)

# t-SNE
tsne = TSNE(n_components=2, random_state=42, perplexity=30)
X_tsne = tsne.fit_transform(X)

plt.subplot(1, 2, 2)
for shot_type in ['Smash', 'Clear', 'Drop', 'Lift', 'Drive']:
    mask = y == shot_type
    plt.scatter(X_tsne[mask, 0], X_tsne[mask, 1], label=shot_type, alpha=0.5, s=20)
plt.xlabel('t-SNE 1')
plt.ylabel('t-SNE 2')
plt.title('t-SNE: Pose Features')
plt.legend()
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('pose_clustering.png', dpi=150)
plt.show()

print("✓ Saved clustering visualization")
print("\nInterpretation:")
print("  ✓ Distinct clusters → Poses are discriminative")
print("  ⚠️  Overlapping blobs → Classes similar but separable")
print("  ❌ Complete overlap → Poses NOT discriminative")
```

### 3.3 Baseline Classifier Test

**Train simple classifier to establish ceiling:**
```python
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score

# Use mean pose as features
X_simple = np.array([pose.mean(axis=0).flatten() for pose in all_poses])

# Train Random Forest (simple baseline)
rf = RandomForestClassifier(n_estimators=100, random_state=42, max_depth=10)
scores = cross_val_score(rf, X_simple, y, cv=5, scoring='accuracy')

print("\n" + "="*80)
print("BASELINE CLASSIFIER (Random Forest on mean poses)")
print("="*80)
print(f"CV Accuracy: {scores.mean():.3f} ± {scores.std():.3f}")
print(f"Best fold: {scores.max():.3f}")
print(f"Worst fold: {scores.min():.3f}")

print("\nInterpretation:")
print("  >70%: Features are good, deep model should work")
print("  50-70%: Features are weak, deep model will struggle")
print("  <50%: Features are very weak, task may be impossible")

if scores.mean() < 0.50:
    print("\n⚠️  WARNING: Even simple classifier can't beat 50%")
    print("   This suggests poses alone may not be sufficient for this task.")
```

---

## Phase 4: Implementation Verification (1 hour)

**Goal:** Rule out bugs in model/training code

### 4.1 Sanity Check: Overfit Single Batch

**Can model learn a tiny dataset perfectly?**
```python
# Create tiny dataset (1 batch)
tiny_dataset = BadmintonDataset(
    train_df.iloc[:32],  # Just 32 samples
    'data/poses',
    label_to_idx,
    normalize=True,
    augment=False
)

tiny_loader = DataLoader(tiny_dataset, batch_size=32, shuffle=False)

# Train lightweight model on this batch
model_test = MSG3D_Lightweight(num_classes=5, in_channels=3, num_joints=33, dropout=0.0).to(device)
optimizer = Adam(model_test.parameters(), lr=0.01)  # High LR
criterion_test = nn.CrossEntropyLoss()

print("Training on single batch (should reach 100% accuracy)...")
for epoch in range(100):
    model_test.train()
    for batch in tiny_loader:
        poses = batch['pose'].to(device)
        labels = batch['label'].to(device)

        optimizer.zero_grad()
        outputs = model_test(poses)
        loss = criterion_test(outputs, labels)
        loss.backward()
        optimizer.step()

        preds = torch.argmax(outputs, dim=1)
        acc = (preds == labels).float().mean().item()

        if epoch % 10 == 0:
            print(f"Epoch {epoch}: Loss={loss.item():.4f}, Acc={acc:.3f}")

print("\nExpected: Accuracy should reach 1.000 by epoch 50-100")
print("If NOT: Model architecture or forward pass has bugs")
```

### 4.2 Check Input/Output Shapes

**Verify data pipeline:**
```python
print("\n" + "="*80)
print("DATA PIPELINE VERIFICATION")
print("="*80)

sample_batch = next(iter(train_loader))

print(f"\nBatch shapes:")
print(f"  Pose: {sample_batch['pose'].shape}  (expected: (32, 3, 150, 33))")
print(f"  Label: {sample_batch['label'].shape}  (expected: (32,))")

print(f"\nPose statistics:")
pose_batch = sample_batch['pose']
print(f"  Mean: {pose_batch.mean():.4f}  (expected: ~0.0)")
print(f"  Std: {pose_batch.std():.4f}  (expected: ~0.5)")
print(f"  Min: {pose_batch.min():.4f}  (expected: >=-3.0)")
print(f"  Max: {pose_batch.max():.4f}  (expected: <=3.0)")
print(f"  NaN count: {torch.isnan(pose_batch).sum().item()}  (expected: 0)")

print(f"\nLabel distribution in batch:")
labels, counts = torch.unique(sample_batch['label'], return_counts=True)
for label, count in zip(labels, counts):
    print(f"  {idx_to_label[label.item()]}: {count.item()}")

# Test model forward pass
model_test = MSG3D_Lightweight(num_classes=5, in_channels=3, num_joints=33, dropout=0.5).to(device)
output = model_test(pose_batch.to(device))

print(f"\nModel output:")
print(f"  Shape: {output.shape}  (expected: (32, 5))")
print(f"  Output range: [{output.min():.4f}, {output.max():.4f}]")
print(f"  NaN in output: {torch.isnan(output).sum().item()}  (expected: 0)")

# Check gradients
loss = nn.CrossEntropyLoss()(output, sample_batch['label'].to(device))
loss.backward()

total_norm = 0
for p in model_test.parameters():
    if p.grad is not None:
        param_norm = p.grad.data.norm(2)
        total_norm += param_norm.item() ** 2
total_norm = total_norm ** 0.5

print(f"\nGradient norm: {total_norm:.4f}  (expected: >0.0, <1000)")

if total_norm == 0:
    print("  ❌ No gradients! Model not learning.")
if total_norm > 1000:
    print("  ⚠️  Exploding gradients! Reduce learning rate.")
```

---

## Phase 5: Model Simplification Experiment (1-2 hours)

**Goal:** Test if even simpler model can learn

### 5.1 Ultra-Lightweight Model

**Train smallest possible model:**
```python
class UltraLightGCN(nn.Module):
    """Minimal model: 2 layers, 32 channels only"""
    def __init__(self, num_classes=5, in_channels=3, num_joints=33, dropout=0.5):
        super().__init__()
        self.A = STGCN_Lightweight.build_adjacency_matrix(self, num_joints)

        # Only 2 layers
        self.gcn1 = GraphConvolution(in_channels, 32, kernel_size=3, dropout=dropout)
        self.gcn2 = GraphConvolution(32, 32, kernel_size=3, dropout=dropout)

        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(32, num_classes)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        A = self.A.to(x.device)
        x = self.gcn1(x, A)
        x = self.gcn2(x, A)
        x = self.pool(x)
        x = x.view(x.size(0), -1)
        x = self.dropout(x)
        x = self.fc(x)
        return x

model_ultra = UltraLightGCN(num_classes=5, dropout=0.7).to(device)
params = sum(p.numel() for p in model_ultra.parameters())
print(f"Ultra-light GCN: {params:,} parameters ({14533/params:.1f} samples/param)")

# Train
optimizer_ultra = Adam(model_ultra.parameters(), lr=0.001)
scheduler_ultra = CosineAnnealingLR(optimizer_ultra, T_max=50)

model_ultra, history_ultra = train_model(
    model_ultra, train_loader, val_loader, criterion,
    optimizer_ultra, scheduler_ultra,
    {**CONFIG, 'num_epochs': 50},
    'UltraLight'
)

print(f"\nResult: {max(history_ultra['val_acc']):.3f} val accuracy")
print("If still <45%: Problem is NOT model complexity")
```

### 5.2 Linear Baseline

**Ultimate simplicity test:**
```python
class LinearBaseline(nn.Module):
    """Linear classifier on mean pose"""
    def __init__(self, num_classes=5, num_joints=33, in_channels=3):
        super().__init__()
        self.fc = nn.Linear(num_joints * in_channels, num_classes)

    def forward(self, x):
        # x: (N, C, T, V)
        # Average over time
        x = x.mean(dim=2)  # (N, C, V)
        x = x.view(x.size(0), -1)  # (N, C*V)
        x = self.fc(x)
        return x

model_linear = LinearBaseline(num_classes=5).to(device)
params = sum(p.numel() for p in model_linear.parameters())
print(f"Linear baseline: {params:,} parameters")

optimizer_linear = Adam(model_linear.parameters(), lr=0.01)

# Train for 20 epochs
model_linear.train()
for epoch in range(20):
    for batch in train_loader:
        poses = batch['pose'].to(device)
        labels = batch['label'].to(device)

        optimizer_linear.zero_grad()
        outputs = model_linear(poses)
        loss = nn.CrossEntropyLoss()(outputs, labels)
        loss.backward()
        optimizer_linear.step()

    # Val accuracy
    model_linear.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for batch in val_loader:
            poses = batch['pose'].to(device)
            labels = batch['label'].to(device)
            outputs = model_linear(poses)
            preds = torch.argmax(outputs, dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)
    acc = correct / total
    print(f"Epoch {epoch+1}: Val Acc = {acc:.3f}")

print("\nLinear baseline result:")
print("  >50%: Problem is solvable, just need right model")
print("  40-50%: Difficult problem, may need feature engineering")
print("  <40%: Problem may be fundamentally difficult with pose alone")
```

---

## Phase 6: Alternative Approaches (2-3 hours)

**Goal:** Test if different features/approaches work better

### 6.1 Hand-Crafted Features

**Extract biomechanical features:**
```python
def extract_biomechanical_features(pose):
    """Extract interpretable features from pose"""
    # Pose shape: (T, 33, 3)

    features = {}

    # Key joints
    RIGHT_WRIST = 16
    RIGHT_ELBOW = 14
    RIGHT_SHOULDER = 12
    LEFT_HIP = 23
    RIGHT_HIP = 24

    # 1. Racket arm extension (wrist-shoulder distance)
    wrist = pose[:, RIGHT_WRIST, :]
    shoulder = pose[:, RIGHT_SHOULDER, :]
    arm_extension = np.linalg.norm(wrist - shoulder, axis=1)
    features['arm_extension_max'] = arm_extension.max()
    features['arm_extension_mean'] = arm_extension.mean()
    features['arm_extension_range'] = arm_extension.max() - arm_extension.min()

    # 2. Arm angle (elbow bend)
    elbow = pose[:, RIGHT_ELBOW, :]
    vec1 = shoulder - elbow
    vec2 = wrist - elbow
    cos_angle = np.sum(vec1 * vec2, axis=1) / (np.linalg.norm(vec1, axis=1) * np.linalg.norm(vec2, axis=1))
    arm_angle = np.arccos(np.clip(cos_angle, -1, 1))
    features['arm_angle_max'] = arm_angle.max()
    features['arm_angle_min'] = arm_angle.min()

    # 3. Wrist height (relative to hip)
    hip_center = (pose[:, LEFT_HIP, :] + pose[:, RIGHT_HIP, :]) / 2
    wrist_height = wrist[:, 1] - hip_center[:, 1]
    features['wrist_height_max'] = wrist_height.max()
    features['wrist_height_range'] = wrist_height.max() - wrist_height.min()

    # 4. Wrist speed (derivative)
    wrist_velocity = np.diff(wrist, axis=0)
    wrist_speed = np.linalg.norm(wrist_velocity, axis=1)
    features['wrist_speed_max'] = wrist_speed.max()
    features['wrist_speed_mean'] = wrist_speed.mean()

    # 5. Body lean (shoulder tilt)
    shoulder_tilt = shoulder[:, 0] - hip_center[:, 0]  # Forward/backward
    features['body_lean_max'] = shoulder_tilt.max()
    features['body_lean_range'] = shoulder_tilt.max() - shoulder_tilt.min()

    return features

# Extract for all samples
feature_list = []
labels_list = []

for idx, row in metadata.iterrows():
    pose_file = poses_dir / f"{row['video_id']}.pkl"
    if pose_file.exists():
        with open(pose_file, 'rb') as f:
            pose = pickle.load(f)
        pose_norm = normalize_pose(pose)
        feats = extract_biomechanical_features(pose_norm)
        feature_list.append(list(feats.values()))
        labels_list.append(row['shot_type'])

X_bio = np.array(feature_list)
y_bio = np.array(labels_list)

# Train Random Forest on biomechanical features
from sklearn.model_selection import train_test_split
X_train, X_val, y_train, y_val = train_test_split(X_bio, y_bio, test_size=0.2, random_state=42, stratify=y_bio)

rf_bio = RandomForestClassifier(n_estimators=200, max_depth=15, random_state=42)
rf_bio.fit(X_train, y_train)

train_acc = rf_bio.score(X_train, y_train)
val_acc = rf_bio.score(X_val, y_val)

print("\n" + "="*80)
print("BIOMECHANICAL FEATURES + RANDOM FOREST")
print("="*80)
print(f"Train accuracy: {train_acc:.3f}")
print(f"Val accuracy: {val_acc:.3f}")

# Feature importance
feature_names = ['arm_ext_max', 'arm_ext_mean', 'arm_ext_range',
                'arm_angle_max', 'arm_angle_min',
                'wrist_height_max', 'wrist_height_range',
                'wrist_speed_max', 'wrist_speed_mean',
                'body_lean_max', 'body_lean_range']

importances = rf_bio.feature_importances_
for name, imp in sorted(zip(feature_names, importances), key=lambda x: -x[1])[:5]:
    print(f"  {name}: {imp:.3f}")

print("\nIf this >50%: Hand-crafted features work better than learned features!")
```

### 6.2 Temporal Attention Model

**Try focusing on key frames:**
```python
class TemporalAttention(nn.Module):
    """Focus on important frames in sequence"""
    def __init__(self, num_classes=5, in_channels=3, num_joints=33, hidden_size=64):
        super().__init__()

        # Frame encoder
        self.frame_encoder = nn.Sequential(
            nn.Linear(num_joints * in_channels, hidden_size),
            nn.ReLU(),
            nn.Dropout(0.5)
        )

        # Attention
        self.attention = nn.Linear(hidden_size, 1)

        # Classifier
        self.classifier = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(hidden_size, num_classes)
        )

    def forward(self, x):
        # x: (N, C, T, V)
        N, C, T, V = x.shape

        # Reshape to (N, T, C*V)
        x = x.permute(0, 2, 1, 3).contiguous()
        x = x.view(N, T, C * V)

        # Encode each frame
        frame_features = self.frame_encoder(x)  # (N, T, hidden)

        # Compute attention weights
        attention_scores = self.attention(frame_features)  # (N, T, 1)
        attention_weights = torch.softmax(attention_scores, dim=1)

        # Weighted sum
        context = (frame_features * attention_weights).sum(dim=1)  # (N, hidden)

        # Classify
        output = self.classifier(context)

        return output

model_attn = TemporalAttention(num_classes=5, hidden_size=64).to(device)
# Train similar to above...

print("Testing if attention mechanism helps...")
```

---

## Phase 7: Decision Matrix (30 minutes)

**Goal:** Synthesize findings and decide next steps

### 7.1 Create Diagnostic Summary

Based on all tests above, fill in this table:

```
DIAGNOSTIC SUMMARY
═══════════════════════════════════════════════════════════

Test                          Result        Interpretation
────────────────────────────────────────────────────────────
Training Convergence          [____]%       ________________
Train-Val Gap                 [____]%       ________________
Data Quality Issues           [____]%       ________________
Label Errors Found            [____]        ________________
Inter-Class Similarity        [____]        ________________
PCA Clustering                [Good/Bad]    ________________
RF Baseline                   [____]%       ________________
Overfit Single Batch          [Yes/No]      ________________
Ultra-Light Model             [____]%       ________________
Linear Baseline               [____]%       ________________
Biomechanical Features        [____]%       ________________
═══════════════════════════════════════════════════════════
```

### 7.2 Root Cause Decision Tree

Follow this decision tree:

```
START
  │
  ├─ Training acc >55%?
  │  ├─ YES → OVERFITTING
  │  │   Action: Reduce model size further, increase dropout to 0.8
  │  │
  │  └─ NO → Continue
  │      │
  │      ├─ NaN/Inf in training?
  │      │  └─ YES → NUMERICAL INSTABILITY
  │      │      Action: Reduce LR to 0.0001, add gradient clipping
  │      │
  │      └─ NO → Continue
  │          │
  │          ├─ Can overfit single batch?
  │          │  ├─ NO → IMPLEMENTATION BUG
  │          │  │   Action: Debug model architecture
  │          │  │
  │          │  └─ YES → Continue
  │          │      │
  │          │      ├─ >10% label errors found?
  │          │      │  └─ YES → LABEL NOISE
  │          │      │      Action: Clean labels, retrain
  │          │      │
  │          │      └─ NO → Continue
  │          │          │
  │          │          ├─ Inter-class similarity >0.85?
  │          │          │  └─ YES → CLASSES TOO SIMILAR
  │          │          │      Action: Merge similar classes or add temporal features
  │          │          │
  │          │          └─ NO → Continue
  │          │              │
  │          │              ├─ RF baseline >50%?
  │          │              │  ├─ YES → FEATURE EXTRACTION ISSUE
  │          │              │  │   Action: Use hand-crafted features or different architecture
  │          │              │  │
  │          │              │  └─ NO → TASK IMPOSSIBLE
  │          │              │      Action: Need additional features (video, audio, trajectory)
```

### 7.3 Recommended Actions by Scenario

**Scenario A: Overfitting (train >55%, val <40%)**
```
Immediate actions:
1. Use ultra-lightweight model (2 layers, 32 channels)
2. Increase dropout to 0.8
3. Reduce augmentation probability (might be adding noise)
4. Try early stopping at epoch 10-15

Expected improvement: 40% → 50-55%
```

**Scenario B: Label Noise (>10% mislabeled)**
```
Immediate actions:
1. Create cleaned dataset (remove/relabel errors)
2. Consider using label smoothing (0.1)
3. Try co-teaching or other noise-robust methods

Expected improvement: Depends on noise level
```

**Scenario C: Feature Quality (RF baseline >50%, deep model <40%)**
```
Immediate actions:
1. Start with biomechanical features + RF
2. Try 1D CNN on pose sequences (simpler than GCN)
3. Ensemble: RF + lightweight GCN

Expected improvement: 40% → 50-60%
```

**Scenario D: Classes Too Similar (similarity >0.85)**
```
Immediate actions:
1. Merge similar classes (Drop+Lift → Defensive)
2. Add temporal derivative features (velocity, acceleration)
3. Try longer sequences (capture more context)

Expected improvement: Depends on new class structure
```

**Scenario E: Fundamental Limitation (RF baseline <40%)**
```
Hard truth: Pose alone may not be sufficient.

Options:
1. Add video frames as additional modality
2. Add racket trajectory (track racket separately)
3. Use temporal context (previous/next shots)
4. Change problem: Predict shot category (Attack/Defense) not specific type

Expected improvement: Requires different approach
```

---

## Execution Checklist

Execute phases in order. Check box when complete:

- [ ] **Phase 1:** Training Curve Analysis (30 min)
  - [ ] Compute train-val gap
  - [ ] Check loss convergence
  - [ ] Identify scenario (A/B/C)

- [ ] **Phase 2:** Data Quality (2-3 hrs)
  - [ ] Pose sequence statistics
  - [ ] Outlier detection
  - [ ] Class similarity analysis
  - [ ] Manual inspection (20 samples)

- [ ] **Phase 3:** Feature Analysis (1-2 hrs)
  - [ ] Visualize poses
  - [ ] PCA/t-SNE clustering
  - [ ] Random Forest baseline

- [ ] **Phase 4:** Implementation Check (1 hr)
  - [ ] Overfit single batch test
  - [ ] Shape verification
  - [ ] Gradient check

- [ ] **Phase 5:** Model Simplification (1-2 hrs)
  - [ ] Ultra-lightweight model
  - [ ] Linear baseline

- [ ] **Phase 6:** Alternative Approaches (2-3 hrs)
  - [ ] Hand-crafted features
  - [ ] Temporal attention

- [ ] **Phase 7:** Decision Matrix (30 min)
  - [ ] Fill diagnostic summary
  - [ ] Follow decision tree
  - [ ] Select recommended actions

---

## Expected Timeline

**Quick path (find obvious issue):** 2-3 hours
- Phase 1 → Phase 2 → Fix identified issue

**Thorough path (systematic investigation):** 6-8 hours
- All phases → comprehensive understanding

**Deep dive (multiple iterations):** 8-12 hours
- All phases → alternative approaches → retrain → analyze

---

## Output Artifacts

Create these files as you go:

1. **diagnostic_summary.txt** - Key findings from each phase
2. **training_diagnostics.png** - Loss/accuracy curves
3. **pose_clustering.png** - PCA/t-SNE visualization
4. **pose_viz_*.png** - Sample pose visualizations (5 classes)
5. **manual_inspection_results.csv** - Label verification results
6. **recommendations.md** - Final recommendations with evidence

---

**Ready to start?** Begin with Phase 1 and report back what you find!
