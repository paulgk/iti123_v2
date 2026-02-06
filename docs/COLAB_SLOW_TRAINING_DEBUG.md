# Colab Training Too Slow - Diagnosis

**Problem:** 1 epoch taking 6 hours (should be 3-4 minutes on L4!)

**Expected:**
- L4 GPU: ~3-4 minutes/epoch → 2-3 hours for 50 epochs
- Your speed: 6 hours/epoch → 300 hours total! ❌

---

## Root Causes (Most Likely → Least Likely)

### 1. ❌ GPU Not Being Used (MOST LIKELY)

**Symptom:** Training runs on CPU instead of GPU

**Check:**
```python
print(f"Device: {CONFIG['device']}")
print(f"CUDA available: {torch.cuda.is_available()}")
print(f"Model on GPU: {next(model.parameters()).is_cuda}")
```

**Expected output:**
```
Device: cuda
CUDA available: True
Model on GPU: True
```

**If shows `False`:** GPU not selected or runtime not set to GPU

**Fix:**
1. Runtime > Change runtime type
2. Hardware accelerator: **GPU**
3. GPU type: **L4** (or T4)
4. Save
5. Runtime > Restart runtime
6. Re-run all cells

---

### 2. ❌ Reading from Google Drive Instead of Local

**Symptom:** Data loading is bottleneck

**Check where data is:**
```python
print(f"Data root: {CONFIG['data_root']}")
!ls -lh {CONFIG['data_root']}/Clear/ | head -5
```

**Bad (slow):**
```
Data root: /content/drive/MyDrive/iti123_data/clips
```
Network I/O from Drive = SUPER SLOW (6 hours makes sense)

**Good (fast):**
```
Data root: /content/data/clips
```
Local SSD = fast

**Fix:** Use Option A (GCS download to local /content/data/clips), NOT Option B (Google Drive)

---

### 3. ❌ num_workers = 0 or Too High

**Check:**
```python
print(f"Num workers: {CONFIG['num_workers']}")
```

**Bad:**
- `num_workers = 0` → No parallel data loading
- `num_workers > 4` → Too much overhead

**Fix:** Set to 2-4 for Colab

---

### 4. ❌ Video Loading Issue

**Check if videos load properly:**
```python
import time
sample_video = train_paths[0]
start = time.time()
frames = train_dataset[0][0]
print(f"Loaded 1 clip in {time.time() - start:.2f} seconds")
print(f"Shape: {frames.shape}")
```

**Expected:** <1 second
**If >5 seconds:** Video decoding issue

---

### 5. ❌ Batch Size Too Small

**Check:**
```python
print(f"Batch size: {CONFIG['batch_size']}")
print(f"Batches per epoch: {len(train_loader)}")
```

**With 12,717 training samples:**
- Batch size 64: 199 batches/epoch ✓
- Batch size 8: 1,590 batches/epoch ❌ (too many iterations)

---

## Quick Debug Cell

Add this cell after configuration:

```python
# === DEBUG: Check training speed ===
import time

print("="*70)
print("PERFORMANCE DEBUG")
print("="*70)

# 1. Check GPU
print(f"\n1. GPU Status:")
print(f"   Device: {CONFIG['device']}")
print(f"   CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"   GPU name: {torch.cuda.get_device_name(0)}")
    print(f"   GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

# 2. Check data location
print(f"\n2. Data Location:")
print(f"   Data root: {CONFIG['data_root']}")
is_drive = '/drive/' in CONFIG['data_root']
print(f"   Using Google Drive: {is_drive} {'❌ SLOW!' if is_drive else '✓ Good'}")

# 3. Check data loader
print(f"\n3. Data Loader:")
print(f"   Batch size: {CONFIG['batch_size']}")
print(f"   Num workers: {CONFIG['num_workers']}")
print(f"   Batches/epoch: {len(train_loader)}")

# 4. Test single batch speed
print(f"\n4. Testing single batch speed...")
start = time.time()
batch = next(iter(train_loader))
frames, labels = batch
print(f"   Batch load time: {time.time() - start:.2f}s")
print(f"   Batch shape: {frames.shape}")

# 5. Test model forward pass
print(f"\n5. Testing model forward pass...")
model_on_gpu = next(model.parameters()).is_cuda
print(f"   Model on GPU: {model_on_gpu}")
frames = frames.to(CONFIG['device'])
start = time.time()
with torch.no_grad():
    output = model(frames)
forward_time = time.time() - start
print(f"   Forward pass time: {forward_time:.3f}s")
print(f"   Output shape: {output.shape}")

# 6. Estimate epoch time
batch_time = forward_time * 2  # Forward + backward ~2x
epoch_estimate = (batch_time * len(train_loader)) / 60
print(f"\n6. Estimated epoch time: {epoch_estimate:.1f} minutes")

print("\n" + "="*70)
print("DIAGNOSIS:")
print("="*70)

issues = []
if not torch.cuda.is_available():
    issues.append("❌ GPU NOT AVAILABLE - Runtime not set to GPU")
if not model_on_gpu:
    issues.append("❌ MODEL NOT ON GPU - Check device setting")
if is_drive:
    issues.append("❌ READING FROM GOOGLE DRIVE - Use GCS download instead")
if CONFIG['num_workers'] == 0:
    issues.append("⚠️  num_workers=0 - Data loading not parallelized")
if forward_time > 1.0:
    issues.append("⚠️  Forward pass slow - GPU may not be utilized")
if epoch_estimate > 10:
    issues.append("❌ WILL TAKE TOO LONG - See issues above")

if issues:
    for issue in issues:
        print(issue)
else:
    print("✓ All checks passed! Training should be fast.")
    print(f"✓ Expected: ~{epoch_estimate:.1f} min/epoch")

print("="*70)
```
