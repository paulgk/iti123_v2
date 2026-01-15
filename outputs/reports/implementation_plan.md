# Implementation Plan - All Improvements

**Status**: Ready to implement
**Dataset**: Original MediaPipe poses (~3500 clips, 40 matches)
**Goal**: Test all 6 suggested improvements systematically

---

## ✅ Completed

1. **Reverted to original MediaPipe dataset**
   - Changed paths back to `poses/`, `features/`, `splits/`
   - Removed MoveNet-related files
   - Restored match-level splitting

2. **Implemented group-stratified split**
   - Updated `split_by_matches()` in [data_split.py](src/data_processing/data_split.py:164)
   - Stratifies by match dominant class (Clear vs Smash)
   - Ensures class balance across splits

---

## 🔧 To Implement

### Priority 1: Data Split Improvements

#### 1.1 Padding Masking with Sequence Lengths

**Files to modify**: `data_split.py`, `lstm_model.py`

**Changes in data_split.py**:
```python
# Step 1: Collect features WITH original lengths
X_train_seq = []
seq_lengths_train = []

for clip in train_clips:
    features, length = get_sequence_features(
        clip_data,
        target_length=50,
        expected_features=42,
        return_length=True  # NEW
    )
    X_train_seq.append(features)  # Still padded for now
    seq_lengths_train.append(length)

# Step 2: Normalize BEFORE padding
# Only normalize non-padded portions
X_train_seq_unpadded = []
for seq, length in zip(X_train_seq, seq_lengths_train):
    X_train_seq_unpadded.append(seq[:length])  # Remove padding temporarily

# Compute stats on unpadded data
all_frames = np.vstack(X_train_seq_unpadded)
seq_mean = np.mean(all_frames, axis=0)
seq_std = np.std(all_frames, axis=0) + 1e-8

# Normalize and re-pad
X_train_seq_norm = []
for seq, length in zip(X_train_seq, seq_lengths_train):
    # Normalize actual frames
    normalized = (seq[:length] - seq_mean) / seq_std
    # Pad with zeros AFTER normalization
    if length < 50:
        padding = np.zeros((50 - length, seq.shape[1]))
        normalized = np.vstack([normalized, padding])
    X_train_seq_norm.append(normalized)

# Step 3: Save with lengths
pickle.dump({
    'X': np.array(X_train_seq_norm),
    'X_stat': X_train_stat_norm,
    'X_stat_raw': X_train_stat,
    'y': y_train,
    'seq_lengths': np.array(seq_lengths_train),  # NEW
    'clip_names': train_clip_names,
    'label_map': label_map
}, f)
```

**Changes in lstm_model.py**:
```python
# Load data WITH lengths
train_data = pickle.load(f'train_data.pkl')
X_train = train_data['X']
seq_lengths = train_data['seq_lengths']  # NEW

# Build model WITH masking
model = Sequential([
    Masking(mask_value=0.0, input_shape=(50, 42)),  # NEW - masks padded timesteps
    LSTM(128, return_sequences=True),
    Dropout(0.3),
    LSTM(64),
    Dropout(0.3),
    Dense(1, activation='sigmoid')
])
```

**Expected Impact**: 5-10% F1 improvement

---

### Priority 2: Training Improvements

#### 2.1 Threshold Tuning

**File to modify**: `lstm_model.py` (or create `threshold_tuner.py`)

```python
def tune_threshold(model, X_val, y_val, seq_lengths_val=None):
    """Find optimal classification threshold on validation set"""

    # Get probability predictions
    y_pred_proba = model.predict(X_val).flatten()

    # Sweep thresholds
    thresholds = np.arange(0.1, 0.9, 0.05)
    best_f1 = 0
    best_threshold = 0.5

    results = []
    for threshold in thresholds:
        y_pred = (y_pred_proba >= threshold).astype(int)
        f1 = f1_score(y_val, y_pred)

        results.append({
            'threshold': threshold,
            'f1': f1,
            'precision': precision_score(y_val, y_pred),
            'recall': recall_score(y_val, y_pred)
        })

        if f1 > best_f1:
            best_f1 = f1
            best_threshold = threshold

    print(f"Optimal threshold: {best_threshold:.2f} (F1: {best_f1:.4f})")

    # Plot precision-recall curve
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.plot([r['threshold'] for r in results], [r['f1'] for r in results])
    plt.xlabel('Threshold')
    plt.ylabel('F1 Score')
    plt.title('F1 vs Threshold')
    plt.grid(True)

    plt.subplot(1, 2, 2)
    plt.plot([r['recall'] for r in results], [r['precision'] for r in results])
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curve')
    plt.grid(True)
    plt.tight_layout()
    plt.savefig('outputs/visualizations/threshold_tuning.png')

    return best_threshold, results
```

**Usage**:
```python
# After training
optimal_threshold = tune_threshold(model, X_val, y_val)

# Evaluate on test set with optimal threshold
y_test_proba = model.predict(X_test).flatten()
y_test_pred = (y_test_proba >= optimal_threshold).astype(int)
test_f1 = f1_score(y_test, y_test_pred)
```

**Expected Impact**: 5-15% F1 improvement (free lunch)

---

#### 2.2 Training Stabilization

**File to modify**: `lstm_model.py`

```python
# Current (likely defaults):
optimizer = Adam(learning_rate=1e-3)
model.compile(optimizer=optimizer, ...)

# IMPROVED:
optimizer = Adam(
    learning_rate=3e-4,  # Lower LR
    clipnorm=1.0         # Gradient clipping
)

# Smaller model for regularization
model = Sequential([
    Masking(mask_value=0.0, input_shape=(50, 42)),
    LSTM(64, return_sequences=True),  # 128 → 64
    Dropout(0.4),                      # 0.3 → 0.4
    LSTM(32),                          # 64 → 32
    Dropout(0.4),
    Dense(1, activation='sigmoid')
])

# Smaller batch size
history = model.fit(
    X_train, y_train,
    batch_size=16,  # Down from likely 32
    epochs=50,
    validation_data=(X_val, y_val),
    callbacks=[
        EarlyStopping(patience=10, restore_best_weights=True),
        ReduceLROnPlateau(factor=0.5, patience=5)  # NEW
    ]
)
```

**Expected Impact**: More stable training, less overfitting

---

### Priority 3: Data Augmentation

#### 3.1 Temporal Augmentation

**Create new file**: `src/data_processing/temporal_augmentation.py`

```python
import numpy as np

def frame_dropout(sequence, dropout_rate=0.15):
    """
    Randomly drop frames and interpolate

    Args:
        sequence: (T, F) array
        dropout_rate: Fraction of frames to drop

    Returns:
        Augmented sequence of same shape
    """
    T, F = sequence.shape
    n_drop = int(T * dropout_rate)

    # Randomly select frames to drop
    keep_indices = np.sort(np.random.choice(T, T - n_drop, replace=False))

    # Interpolate dropped frames
    augmented = np.zeros_like(sequence)
    augmented[keep_indices] = sequence[keep_indices]

    # Linear interpolation for dropped frames
    for i in range(T):
        if i not in keep_indices:
            # Find nearest kept frames
            prev_idx = keep_indices[keep_indices < i][-1] if np.any(keep_indices < i) else keep_indices[0]
            next_idx = keep_indices[keep_indices > i][0] if np.any(keep_indices > i) else keep_indices[-1]

            # Interpolate
            if prev_idx == next_idx:
                augmented[i] = sequence[prev_idx]
            else:
                alpha = (i - prev_idx) / (next_idx - prev_idx)
                augmented[i] = (1 - alpha) * sequence[prev_idx] + alpha * sequence[next_idx]

    return augmented


def add_noise(sequence, noise_std=0.05):
    """Add Gaussian noise to velocities (features with high variance)"""
    noise = np.random.normal(0, noise_std, sequence.shape)
    return sequence + noise


def time_warp(sequence, warp_factor=0.1):
    """Stretch or compress sequence in time"""
    T, F = sequence.shape
    new_T = int(T * (1 + np.random.uniform(-warp_factor, warp_factor)))

    # Resample sequence
    old_indices = np.linspace(0, T-1, T)
    new_indices = np.linspace(0, T-1, new_T)

    warped = np.zeros((new_T, F))
    for f in range(F):
        warped[:, f] = np.interp(new_indices, old_indices, sequence[:, f])

    # Crop or pad to original length
    if new_T > T:
        warped = warped[:T]
    elif new_T < T:
        padding = np.zeros((T - new_T, F))
        warped = np.vstack([warped, padding])

    return warped


def augment_sequence(sequence, augment_prob=0.8):
    """
    Apply random augmentations

    Args:
        sequence: (T, F) array
        augment_prob: Probability of applying each augmentation

    Returns:
        Augmented sequence
    """
    aug_seq = sequence.copy()

    if np.random.random() < augment_prob:
        aug_seq = frame_dropout(aug_seq, dropout_rate=0.15)

    if np.random.random() < augment_prob:
        aug_seq = add_noise(aug_seq, noise_std=0.05)

    if np.random.random() < augment_prob:
        aug_seq = time_warp(aug_seq, warp_factor=0.1)

    return aug_seq


class TemporalAugmentationGenerator:
    """Data generator with on-the-fly augmentation"""

    def __init__(self, X, y, batch_size=32, augment=True):
        self.X = X
        self.y = y
        self.batch_size = batch_size
        self.augment = augment
        self.n_samples = len(X)

    def __len__(self):
        return int(np.ceil(self.n_samples / self.batch_size))

    def __getitem__(self, idx):
        start = idx * self.batch_size
        end = min((idx + 1) * self.batch_size, self.n_samples)

        batch_X = self.X[start:end]
        batch_y = self.y[start:end]

        if self.augment:
            batch_X_aug = np.array([augment_sequence(seq) for seq in batch_X])
            return batch_X_aug, batch_y

        return batch_X, batch_y
```

**Usage in training**:
```python
from temporal_augmentation import TemporalAugmentationGenerator

# Create augmented generator
train_gen = TemporalAugmentationGenerator(X_train, y_train, batch_size=16, augment=True)
val_gen = TemporalAugmentationGenerator(X_val, y_val, batch_size=16, augment=False)

# Train with augmentation
history = model.fit(
    train_gen,
    validation_data=val_gen,
    epochs=50,
    callbacks=[...]
)
```

**Expected Impact**: 10-20% F1 improvement (huge for small datasets)

---

### Priority 4: Feature Engineering

#### 4.1 Add Acceleration Features

**File to modify**: `feature_engineering_v2.py`

```python
def calculate_acceleration(positions, smooth=True):
    """
    Calculate acceleration (second derivative)

    Args:
        positions: (T, 3) array of [x, y, z]
        smooth: Apply Gaussian smoothing

    Returns:
        acceleration: (T, 3) array
    """
    velocity = calculate_velocity(positions, smooth=smooth)

    # Compute second derivative
    acceleration = np.gradient(velocity, axis=0)

    if smooth:
        acceleration = smooth_trajectory(acceleration, sigma=2.0)

    return acceleration


def extract_frame_features(pose):
    """Extract features from a single pose frame - WITH ACCELERATION"""
    features = {}

    # ... existing position, velocity features ...

    # NEW: Acceleration features
    r_wrist_accel = calculate_acceleration(r_wrist_history)  # Need window
    r_elbow_accel = calculate_acceleration(r_elbow_history)

    features['r_wrist_accel_mag'] = np.linalg.norm(r_wrist_accel)
    features['r_elbow_accel_mag'] = np.linalg.norm(r_elbow_accel)
    features['r_wrist_accel_y'] = r_wrist_accel[1]  # Vertical acceleration

    # Momentum proxy (velocity magnitude)
    features['r_wrist_momentum'] = np.linalg.norm(r_wrist_vel)
    features['r_elbow_momentum'] = np.linalg.norm(r_elbow_vel)

    return features
```

**Note**: Acceleration requires a time window (3-5 frames), so features need to be computed differently.

**Expected Impact**: 5-10% F1 improvement if Clear/Smash differ in swing dynamics

---

## 📊 Testing Protocol

### Test 1: Baseline with Improvements
```bash
# Step 1: Run improved data split
python src/data_processing/feature_engineering_v2.py  # Existing features
python src/data_processing/data_split.py              # Group-stratified + padding lengths

# Step 2: Train with padding masking
python src/models/lstm_model.py --mask-padding

# Step 3: Tune threshold
python src/models/threshold_tuner.py

# Step 4: Evaluate
python src/models/evaluate_models.py
```

### Test 2: With Temporal Augmentation
```bash
python src/models/lstm_model.py --mask-padding --augment
python src/models/threshold_tuner.py
python src/models/evaluate_models.py
```

### Test 3: With Acceleration Features
```bash
python src/data_processing/feature_engineering_v3.py  # Add acceleration
python src/data_processing/data_split.py
python src/models/lstm_model.py --mask-padding --augment
python src/models/threshold_tuner.py
python src/models/evaluate_models.py
```

---

## 📈 Expected Results

**Current Baseline** (MediaPipe, ~3500 clips):
- Accuracy: ~50%
- F1: ~45-50%

**After All Improvements**:
- **Optimistic**: 60-65% accuracy, 55-60% F1 (+10-15%)
- **Realistic**: 55-60% accuracy, 50-55% F1 (+5-10%)
- **Pessimistic**: 50-55% accuracy, 45-50% F1 (+0-5%)

**If still ~50%**: Confirms fundamental biomechanical similarity between Clear and Smash strokes. Pose-only classification has inherent limits.

---

## 🚀 Next Steps

1. **Implement Priority 1** (data split improvements)
2. **Run baseline test** (no augmentation)
3. **Implement Priority 2** (training improvements)
4. **Run augmentation test**
5. **Implement Priority 3** (feature engineering)
6. **Run final test**
7. **Document results** and compare with baseline

Would you like me to start implementing these changes?
