# Colab Training Performance Optimization

## Problem: Slow Training (3.32s/iteration)

**Current Performance:**
- 3.32 seconds per iteration
- ~260 iterations per epoch (with batch_size=64, ~16,500 samples)
- **~14.4 minutes per epoch** (260 × 3.32s ÷ 60)

**Expected Performance:**
- Should be ~0.5-1.0s per iteration
- ~4-8 minutes per epoch

**Root Cause:** Data loading bottleneck

---

## Bottleneck Analysis

### Current Data Pipeline Issues

**Cell 14 - Dataset `__getitem__`:**
```python
def __getitem__(self, idx):
    frames = np.load(self.npy_paths[idx])  # ✅ Fast (already numpy)

    # ❌ SLOW: Frame-by-frame processing
    for frame in frames:
        frame_pil = Image.fromarray(frame)  # ❌ NumPy → PIL conversion
        if self.transform:
            frame_pil = self.transform(frame_pil)  # ❌ PIL operations
        frame_tensor = transforms.ToTensor()(frame_pil)  # ❌ PIL → Tensor
        frame_tensor = self.normalize(frame_tensor)  # ✅ Fast
```

**Problems:**
1. **NumPy → PIL → Tensor conversion** (expensive per-frame)
2. **PIL-based augmentation** (slow for batch operations)
3. **Frame-by-frame loop** (no vectorization)
4. **Single worker** (`num_workers=1`)
5. **No persistent workers** (restart overhead each epoch)

---

## Optimizations

### 1. Vectorized Tensor Operations (Fastest)

**Replace PIL with native PyTorch:**

```python
class BadmintonFramesDatasetOptimized(Dataset):
    def __init__(self, npy_paths, labels, augment=False, skip_ratio=0.0):
        self.npy_paths = npy_paths
        self.labels = labels
        self.augment = augment
        self.skip_ratio = skip_ratio

        # Pre-compute normalization tensors
        self.mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
        self.std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)

    def __getitem__(self, idx):
        # Load frames
        frames = np.load(self.npy_paths[idx])  # (T, H, W, 3), uint8
        label = self.labels[idx]

        # Apply skip_ratio
        if self.skip_ratio > 0:
            skip_count = int(len(frames) * self.skip_ratio)
            frames = frames[skip_count:]

        # ✅ FAST: Direct NumPy → Tensor (no PIL!)
        # Convert to tensor: (T, H, W, 3) → (T, 3, H, W)
        frames_tensor = torch.from_numpy(frames).permute(0, 3, 1, 2).float() / 255.0

        # ✅ FAST: Vectorized augmentation (if needed)
        if self.augment:
            frames_tensor = self._augment_batch(frames_tensor)

        # ✅ FAST: Vectorized normalization
        frames_tensor = (frames_tensor - self.mean) / self.std

        return frames_tensor, label

    def _augment_batch(self, frames):
        """Vectorized augmentation on all frames at once"""
        # Random horizontal flip (50% chance)
        if torch.rand(1).item() > 0.5:
            frames = torch.flip(frames, dims=[3])  # Flip width dimension

        # Random brightness/contrast (simple version)
        if torch.rand(1).item() > 0.5:
            brightness_factor = 0.8 + torch.rand(1).item() * 0.4  # 0.8-1.2
            frames = torch.clamp(frames * brightness_factor, 0, 1)

        return frames
```

**Benefits:**
- ✅ No PIL conversion overhead
- ✅ Vectorized operations (process all frames together)
- ✅ ~3-5x faster per sample

---

### 2. Increase DataLoader Workers

**Current:**
```python
TRAIN_CONFIG = {
    'num_workers': 1,  # ❌ Only 1 worker (CPU idle)
    'prefetch_factor': 2,
    'persistent_workers': False,  # ❌ Restart overhead
}
```

**Optimized:**
```python
TRAIN_CONFIG = {
    'num_workers': 4,  # ✅ Use 4 workers (Colab has 2 CPUs)
    'prefetch_factor': 4,  # ✅ Prefetch more batches
    'persistent_workers': True,  # ✅ Keep workers alive
    'pin_memory': True,  # ✅ Already enabled (good!)
}
```

**Benefits:**
- ✅ Parallel data loading
- ✅ GPU never starved
- ✅ ~2-3x faster

---

### 3. Larger Batch Size (If RAM allows)

**Current:**
```python
'batch_size': 64,  # May be too small for T4 GPU
```

**Try:**
```python
'batch_size': 96,  # or 128 if RAM allows
```

**Check GPU memory usage:**
```python
# After first batch
print(f"GPU Memory Used: {torch.cuda.memory_allocated() / 1e9:.2f} GB")
print(f"GPU Memory Cached: {torch.cuda.memory_reserved() / 1e9:.2f} GB")
```

**Benefits:**
- ✅ Fewer iterations per epoch
- ✅ Better GPU utilization
- ✅ ~20-30% faster

---

### 4. Pre-load Data to RAM (If dataset fits)

**For small datasets (< 10GB):**

```python
class BadmintonFramesDatasetPreloaded(Dataset):
    def __init__(self, npy_paths, labels, augment=False, skip_ratio=0.0):
        print("Pre-loading dataset to RAM...")
        self.data = []
        self.labels = labels
        self.augment = augment
        self.skip_ratio = skip_ratio

        # Load all data once
        for path in tqdm(npy_paths, desc='Loading'):
            frames = np.load(path)
            if skip_ratio > 0:
                skip_count = int(len(frames) * skip_ratio)
                frames = frames[skip_count:]
            self.data.append(frames)

        print(f"✓ Loaded {len(self.data)} samples to RAM")

        # Pre-compute normalization
        self.mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
        self.std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)

    def __getitem__(self, idx):
        frames = self.data[idx]  # ✅ Already in RAM (no disk I/O!)
        label = self.labels[idx]

        # Convert to tensor
        frames_tensor = torch.from_numpy(frames).permute(0, 3, 1, 2).float() / 255.0

        # Augment if needed
        if self.augment:
            frames_tensor = self._augment_batch(frames_tensor)

        # Normalize
        frames_tensor = (frames_tensor - self.mean) / self.std

        return frames_tensor, label
```

**Benefits:**
- ✅ No disk I/O during training
- ✅ ~2x faster
- ⚠️ Requires RAM (check with `!free -h`)

---

## Complete Optimized Configuration

### Updated Cell 4 (Configuration)

```python
# Training settings (OPTIMIZED)
TRAIN_CONFIG = {
    'batch_size': 96,  # ⬆️ Increased from 64
    'num_epochs': 100,
    'learning_rate': 0.001,
    'weight_decay': 0.0001,
    'early_stopping_patience': 15,
    'num_workers': 4,  # ⬆️ Increased from 1
    'prefetch_factor': 4,  # ⬆️ Increased from 2
    'persistent_workers': True,  # ⬆️ Changed from False
    'checkpoint_path': f"{CHECKPOINT_DIR}/latest_checkpoint.pth",
    'resume_training': True,
}
```

### Updated Cell 14 (Optimized Dataset)

```python
import torch
from torch.utils.data import Dataset
import numpy as np

class BadmintonFramesDatasetOptimized(Dataset):
    """
    Optimized dataset with vectorized operations.
    ~3-5x faster than PIL-based version.
    """
    def __init__(self, npy_paths, labels, augment=False, skip_ratio=0.0):
        self.npy_paths = npy_paths
        self.labels = labels
        self.augment = augment
        self.skip_ratio = skip_ratio

        # Pre-compute normalization tensors (avoid repeated creation)
        self.mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
        self.std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)

    def __len__(self):
        return len(self.npy_paths)

    def __getitem__(self, idx):
        # Load frames
        frames = np.load(self.npy_paths[idx])  # (T, H, W, 3), uint8
        label = self.labels[idx]

        # Apply skip_ratio
        if self.skip_ratio > 0:
            skip_count = int(len(frames) * self.skip_ratio)
            frames = frames[skip_count:]

        # Direct NumPy → Tensor (no PIL conversion!)
        # (T, H, W, 3) → (T, 3, H, W), normalize to [0, 1]
        frames_tensor = torch.from_numpy(frames).permute(0, 3, 1, 2).float() / 255.0

        # Vectorized augmentation (process all frames together)
        if self.augment:
            frames_tensor = self._augment_batch(frames_tensor)

        # Vectorized normalization
        frames_tensor = (frames_tensor - self.mean) / self.std

        return frames_tensor, label

    def _augment_batch(self, frames):
        """
        Fast vectorized augmentation on all frames at once.
        Replaces slow PIL operations.
        """
        # Random horizontal flip (50% chance)
        if torch.rand(1).item() > 0.5:
            frames = torch.flip(frames, dims=[3])  # Flip width

        # Random brightness adjustment (simple, fast)
        if torch.rand(1).item() > 0.5:
            brightness = 0.8 + torch.rand(1).item() * 0.4  # 0.8 to 1.2
            frames = torch.clamp(frames * brightness, 0, 1)

        # Random contrast adjustment
        if torch.rand(1).item() > 0.5:
            contrast = 0.8 + torch.rand(1).item() * 0.4  # 0.8 to 1.2
            mean = frames.mean(dim=[2, 3], keepdim=True)
            frames = torch.clamp((frames - mean) * contrast + mean, 0, 1)

        return frames

print("✓ BadmintonFramesDatasetOptimized class defined")
```

### Updated Cell 15 (Use Optimized Dataset)

```python
# Create datasets with optimized class
SKIP_RATIO = CONFIG['skip_ratio']

train_dataset = BadmintonFramesDatasetOptimized(
    train_npy_paths,
    train_labels,
    augment=True,
    skip_ratio=SKIP_RATIO
)
val_dataset = BadmintonFramesDatasetOptimized(
    val_npy_paths,
    val_labels,
    augment=False,
    skip_ratio=SKIP_RATIO
)
test_dataset = BadmintonFramesDatasetOptimized(
    test_npy_paths,
    test_labels,
    augment=False,
    skip_ratio=SKIP_RATIO
)

print("Datasets created (OPTIMIZED):")
print(f"  Train: {len(train_dataset)} samples")
print(f"  Val:   {len(val_dataset)} samples")
print(f"  Test:  {len(test_dataset)} samples")
print("\n✓ Using vectorized tensor operations (3-5x faster)")
```

---

## Expected Performance Improvements

| Optimization | Speed Improvement | Cumulative |
|--------------|-------------------|------------|
| **Baseline** | 3.32s/it | 3.32s/it |
| + Vectorized tensors | ~4x faster | **0.83s/it** |
| + 4 workers | ~2x faster | **0.42s/it** |
| + Persistent workers | ~1.2x faster | **0.35s/it** |
| + Larger batch (96) | ~1.2x faster | **0.29s/it** |

**Final Expected:**
- **~0.3-0.5s per iteration** (vs current 3.32s)
- **~2-4 minutes per epoch** (vs current 14.4 minutes)
- **~10x total speedup**

---

## Implementation Steps

### Quick Fix (Immediate):

1. **Cell 4**: Change `num_workers=4`, `persistent_workers=True`, `prefetch_factor=4`
2. **Cell 4**: Try `batch_size=96` (monitor GPU memory)
3. Re-run from Cell 15 onwards

**Expected:** ~2-3x speedup immediately

### Full Optimization:

1. Replace Cell 14 with optimized dataset class
2. Update Cell 15 to use optimized class
3. Update Cell 4 with optimized config
4. Re-run from Cell 14 onwards

**Expected:** ~8-10x speedup

---

## Monitoring Performance

### Add this after first training batch:

```python
import time

# Time a few iterations
times = []
for i, (frames, labels) in enumerate(train_loader):
    if i >= 10:
        break
    start = time.time()

    frames = frames.to(device)
    labels = labels.to(device)

    with torch.cuda.amp.autocast():
        outputs = model(frames)
        loss = criterion(outputs, labels)

    scaler.scale(loss).backward()
    scaler.step(optimizer)
    scaler.update()
    optimizer.zero_grad()

    times.append(time.time() - start)

print(f"\nAverage iteration time: {np.mean(times):.3f}s")
print(f"Expected epoch time: {np.mean(times) * len(train_loader) / 60:.1f} minutes")
```

### Check GPU Utilization:

```python
# In Colab, run this in a separate cell
!nvidia-smi
```

**Look for:**
- GPU Utilization: Should be 80-100%
- GPU Memory: Should be 60-90% used
- If GPU util < 50% → Data loading is bottleneck

---

## Troubleshooting

### Issue: "Too many open files"
```python
# Reduce workers
TRAIN_CONFIG['num_workers'] = 2
```

### Issue: "CUDA out of memory"
```python
# Reduce batch size
TRAIN_CONFIG['batch_size'] = 64  # or 48
```

### Issue: Still slow after optimizations
```python
# Check disk I/O (is /content/data on slow disk?)
!df -h /content/data

# Try pre-loading to RAM (if < 10GB dataset)
# Use BadmintonFramesDatasetPreloaded class
```

---

## Summary

**Current:** 3.32s/it (very slow, data loading bottleneck)

**Optimizations:**
1. ✅ Vectorized tensor operations (no PIL) - 4x faster
2. ✅ Increase workers to 4 - 2x faster
3. ✅ Persistent workers - 1.2x faster
4. ✅ Larger batch size (96) - 1.2x faster

**Expected Result:** ~0.3-0.5s/it (~10x speedup)

**Next Step:** Apply optimizations to your v3 notebook!
