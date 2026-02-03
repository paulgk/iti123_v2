#!/usr/bin/env python3
"""
Fixed Training Script for Badminton Shot Classification

Fixes:
1. Proper pose normalization (torso-centered, height-scaled)
2. Filter short sequences (<30 frames)
3. Correct learning rate (0.001)
4. Proper early stopping
5. Class-weighted loss
6. Correct ST-GCN adjacency matrix

Usage:
    python scripts/train_models_fixed.py --metadata data/data/metadata.csv --output outputs/
"""

import argparse
import os
import pickle
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import GroupShuffleSplit
from sklearn.preprocessing import LabelEncoder
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import classification_report, confusion_matrix, f1_score
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm


# ============================================================================
# CONFIGURATION
# ============================================================================

TARGET_LENGTH = 90  # Pad/truncate to 90 frames (3 seconds @ 30fps)
MIN_FRAMES = 30  # Filter out sequences shorter than 1 second
BATCH_SIZE = 32
NUM_EPOCHS = 50
LEARNING_RATE = 0.001
WEIGHT_DECAY = 1e-4
PATIENCE = 10
NUM_WORKERS = 4


# ============================================================================
# DATA PREPROCESSING
# ============================================================================

def normalize_pose(pose_sequence):
    """
    Normalize pose sequence:
    - Center by torso (hip center)
    - Scale by body height

    Args:
        pose_sequence: numpy array of shape (T, 33, 3)

    Returns:
        normalized: numpy array of shape (T, 33, 3)
    """
    # MediaPipe keypoint indices
    LEFT_HIP = 23
    RIGHT_HIP = 24
    NOSE = 0

    # Calculate hip center (torso reference point)
    hip_center = (pose_sequence[:, LEFT_HIP, :2] + pose_sequence[:, RIGHT_HIP, :2]) / 2

    # Center all keypoints by hip
    centered = pose_sequence.copy()
    centered[:, :, :2] = centered[:, :, :2] - hip_center[:, np.newaxis, :]

    # Calculate body height (nose to hip distance)
    body_heights = np.linalg.norm(
        pose_sequence[:, NOSE, :2] - hip_center, axis=1
    )
    body_height = np.mean(body_heights)

    # Scale by body height
    if body_height > 0.01:  # Avoid division by zero
        centered[:, :, :2] = centered[:, :, :2] / body_height

    return centered


def is_valid_single_person(pose_sequence, max_width=0.6):
    """
    Filter out multi-player detections

    CRITICAL: Badminton videos have 2 players. MediaPipe sometimes detects both,
    creating a "skeleton" that spans the entire court. This breaks normalization.

    Args:
        pose_sequence: numpy array of shape (T, 33, 3)
        max_width: Maximum width ratio (0.6 = 60% of frame)

    Returns:
        True if sequence represents a single player
    """
    x_coords = pose_sequence[:, :, 0]
    x_range = np.max(x_coords) - np.min(x_coords)

    # Single person shouldn't span more than 60% of frame width
    return x_range < max_width


def pad_or_truncate(sequence, target_length):
    """
    Pad or truncate sequence to fixed length

    Args:
        sequence: numpy array of shape (T, 33, 3)
        target_length: int

    Returns:
        padded: numpy array of shape (target_length, 33, 3)
    """
    T, V, C = sequence.shape

    if T >= target_length:
        # Truncate (take middle frames to preserve motion)
        start = (T - target_length) // 2
        return sequence[start:start+target_length]
    else:
        # Pad with zeros
        padded = np.zeros((target_length, V, C), dtype=sequence.dtype)
        padded[:T] = sequence
        return padded


# ============================================================================
# DATASET
# ============================================================================

class PoseDataset(Dataset):
    """PyTorch Dataset for pose sequences"""

    def __init__(self, X, y):
        """
        Args:
            X: numpy array of shape (N, T, V, C)
            y: numpy array of labels (N,)
        """
        self.X = X
        self.y = y

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        pose = self.X[idx]
        label = self.y[idx]

        # Convert to tensor and reshape for GCN: (T, V, C) -> (C, T, V, M)
        pose_tensor = torch.FloatTensor(pose)  # (T, V, C)
        pose_tensor = pose_tensor.permute(2, 0, 1)  # (C, T, V)
        pose_tensor = pose_tensor.unsqueeze(-1)  # (C, T, V, 1)

        label_tensor = torch.LongTensor([label])[0]

        return pose_tensor, label_tensor


# ============================================================================
# MEDIAPIPE SKELETON GRAPH
# ============================================================================

# MediaPipe 33 keypoints connections
MEDIAPIPE_EDGES = [
    # Arms
    (11, 13), (13, 15), (15, 17), (15, 19), (15, 21),  # Left arm
    (12, 14), (14, 16), (16, 18), (16, 20), (16, 22),  # Right arm
    (11, 12),  # Shoulders
    # Torso
    (11, 23), (12, 24), (23, 24),  # Torso connections
    # Legs
    (23, 25), (25, 27), (27, 29), (27, 31),  # Left leg
    (24, 26), (26, 28), (28, 30), (28, 32),  # Right leg
    # Face (helps with orientation)
    (0, 1), (1, 2), (2, 3), (3, 7),  # Left face
    (0, 4), (4, 5), (5, 6), (6, 8),  # Right face
]


def get_adjacency_matrix(edges, num_nodes=33, self_loops=True):
    """Create adjacency matrix from edge list"""
    A = np.zeros((num_nodes, num_nodes), dtype=np.float32)

    for i, j in edges:
        A[i, j] = 1
        A[j, i] = 1

    if self_loops:
        A += np.eye(num_nodes)

    # Normalize adjacency matrix
    D = np.sum(A, axis=1)
    D_inv = np.diag(1.0 / np.sqrt(D))
    A_norm = D_inv @ A @ D_inv

    return A_norm


# ============================================================================
# MODEL ARCHITECTURES
# ============================================================================

class GraphConvolution(nn.Module):
    """Graph Convolution Layer"""

    def __init__(self, in_channels, out_channels, A):
        super(GraphConvolution, self).__init__()

        self.register_buffer('A', torch.FloatTensor(A))

        self.conv = nn.Conv2d(in_channels, out_channels, 1)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        # x shape: (N, C, T, V)
        N, C, T, V = x.shape

        # Apply graph convolution: X' = A @ X
        x = x.permute(0, 1, 3, 2)  # (N, C, V, T)
        x = torch.matmul(self.A, x)  # (N, C, V, T)
        x = x.permute(0, 1, 3, 2)  # (N, C, T, V)

        # Apply convolution
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)

        return x


class SimpleSTGCN(nn.Module):
    """Spatial-Temporal Graph Convolutional Network"""

    def __init__(self, in_channels=3, num_classes=5, A=None, dropout=0.5):
        super(SimpleSTGCN, self).__init__()

        if A is None:
            A = get_adjacency_matrix(MEDIAPIPE_EDGES)

        # Spatial-temporal blocks
        self.gcn1 = GraphConvolution(in_channels, 64, A)
        self.tcn1 = nn.Sequential(
            nn.Conv2d(64, 64, (9, 1), padding=(4, 0)),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout)
        )

        self.gcn2 = GraphConvolution(64, 128, A)
        self.tcn2 = nn.Sequential(
            nn.Conv2d(128, 128, (9, 1), padding=(4, 0)),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout)
        )

        self.gcn3 = GraphConvolution(128, 256, A)
        self.tcn3 = nn.Sequential(
            nn.Conv2d(256, 256, (9, 1), padding=(4, 0)),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout)
        )

        # Global pooling and classifier
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(256, num_classes)

    def forward(self, x):
        # x shape: (N, C, T, V, M)
        N, C, T, V, M = x.shape
        x = x.squeeze(-1)  # (N, C, T, V)

        # ST-GCN blocks
        x = self.gcn1(x)
        x = self.tcn1(x)

        x = self.gcn2(x)
        x = self.tcn2(x)

        x = self.gcn3(x)
        x = self.tcn3(x)

        # Global pooling
        x = self.pool(x)  # (N, 256, 1, 1)
        x = x.view(N, -1)  # (N, 256)

        # Classifier
        x = self.fc(x)

        return x


class SimpleLSTM(nn.Module):
    """Bidirectional LSTM baseline"""

    def __init__(self, input_size=99, hidden_size=128, num_layers=2, num_classes=5, dropout=0.3):
        super(SimpleLSTM, self).__init__()

        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0
        )

        self.fc = nn.Sequential(
            nn.Linear(hidden_size * 2, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        # x shape: (N, C, T, V, M)
        N, C, T, V, M = x.shape

        # Reshape to (N, T, C*V*M)
        x = x.squeeze(-1)  # (N, C, T, V)
        x = x.permute(0, 2, 1, 3)  # (N, T, C, V)
        x = x.reshape(N, T, -1)  # (N, T, C*V)

        # LSTM
        lstm_out, _ = self.lstm(x)

        # Use last timestep
        out = lstm_out[:, -1, :]

        # Classifier
        out = self.fc(out)

        return out


# ============================================================================
# TRAINING FUNCTIONS
# ============================================================================

def train_epoch(model, dataloader, criterion, optimizer, device):
    """Train for one epoch"""
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    pbar = tqdm(dataloader, desc="Training", leave=False)
    for inputs, labels in pbar:
        inputs, labels = inputs.to(device), labels.to(device)

        # Forward pass
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)

        # Backward pass
        loss.backward()
        optimizer.step()

        # Statistics
        running_loss += loss.item()
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()

        # Update progress bar
        pbar.set_postfix({
            'loss': f'{running_loss/len(dataloader):.4f}',
            'acc': f'{100.*correct/total:.2f}%'
        })

    epoch_loss = running_loss / len(dataloader)
    epoch_acc = correct / total

    return epoch_loss, epoch_acc


def evaluate(model, dataloader, criterion, device):
    """Evaluate model"""
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0

    all_preds = []
    all_labels = []

    with torch.no_grad():
        for inputs, labels in tqdm(dataloader, desc="Evaluating", leave=False):
            inputs, labels = inputs.to(device), labels.to(device)

            outputs = model(inputs)
            loss = criterion(outputs, labels)

            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    epoch_loss = running_loss / len(dataloader)
    epoch_acc = correct / total

    return epoch_loss, epoch_acc, np.array(all_preds), np.array(all_labels)


# ============================================================================
# MAIN TRAINING PIPELINE
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description='Train badminton shot classification models')
    parser.add_argument('--metadata', type=str, required=True, help='Path to metadata.csv')
    parser.add_argument('--output', type=str, default='outputs/', help='Output directory')
    parser.add_argument('--model', type=str, default='stgcn', choices=['stgcn', 'lstm'], help='Model to train')
    parser.add_argument('--epochs', type=int, default=NUM_EPOCHS, help='Number of epochs')
    parser.add_argument('--batch-size', type=int, default=BATCH_SIZE, help='Batch size')
    parser.add_argument('--lr', type=float, default=LEARNING_RATE, help='Learning rate')

    args = parser.parse_args()

    # Create output directories
    os.makedirs(args.output, exist_ok=True)
    os.makedirs(f'{args.output}/models', exist_ok=True)
    os.makedirs(f'{args.output}/reports', exist_ok=True)

    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\nUsing device: {device}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    print(f"\n{'='*60}")
    print("LOADING AND PREPROCESSING DATA")
    print(f"{'='*60}\n")

    # Load metadata
    df = pd.read_csv(args.metadata)
    print(f"Total entries in metadata: {len(df)}")

    # Filter for valid entries
    df_valid = df[df['status'] == 'success'].copy()
    print(f"Valid entries: {len(df_valid)}")

    # Load all poses
    print("\nLoading pose sequences...")
    pose_sequences = []
    labels = []
    player_ids = []

    for idx, row in tqdm(df_valid.iterrows(), total=len(df_valid), desc="Loading"):
        pose_file = Path(row['pose_file'])
        if not pose_file.is_absolute():
            pose_file = Path('data/processed/poses') / pose_file.name

        try:
            with open(pose_file, 'rb') as f:
                pose_data = pickle.load(f)

            # Filter short sequences
            if len(pose_data) < MIN_FRAMES:
                continue

            # Filter multi-player detections
            if not is_valid_single_person(pose_data):
                continue

            # Normalize pose
            normalized_pose = normalize_pose(pose_data)

            # Pad/truncate
            padded_pose = pad_or_truncate(normalized_pose, TARGET_LENGTH)

            pose_sequences.append(padded_pose)
            labels.append(row['stroke_type'])
            player_ids.append(row['player_id'])

        except Exception as e:
            print(f"\nError loading {row['video_id']}: {e}")
            continue

    filtered_count = len(df_valid) - len(pose_sequences)
    print(f"\n✓ Loaded {len(pose_sequences)} pose sequences")
    print(f"  Filtered: {filtered_count} clips (short sequences + multi-player detections)")

    # Convert to arrays
    X = np.array(pose_sequences, dtype=np.float32)
    labels = np.array(labels)
    player_ids = np.array(player_ids)

    print(f"Data shape: {X.shape}")
    print(f"Expected: (N, {TARGET_LENGTH}, 33, 3)")

    # Check normalization
    print(f"\nData statistics:")
    print(f"  Mean: {X[:, :, :, :2].mean():.4f} (should be ~0.0)")
    print(f"  Std: {X[:, :, :, :2].std():.4f} (should be ~0.1-0.3)")

    # Encode labels
    le = LabelEncoder()
    y = le.fit_transform(labels)

    print(f"\n{'='*60}")
    print("CLASS DISTRIBUTION")
    print(f"{'='*60}")
    for i, label in enumerate(le.classes_):
        count = np.sum(y == i)
        print(f"  {i}: {label.capitalize():10s} - {count:,} samples ({count/len(y)*100:.1f}%)")

    # Compute class weights
    class_weights = compute_class_weight(
        class_weight='balanced',
        classes=np.unique(y),
        y=y
    )
    class_weights_tensor = torch.FloatTensor(class_weights).to(device)

    print(f"\nClass weights:")
    for i, (class_name, weight) in enumerate(zip(le.classes_, class_weights)):
        print(f"  {class_name.capitalize():10s}: {weight:.3f}")

    # Train-val-test split (player-based)
    print(f"\n{'='*60}")
    print("SPLITTING DATA")
    print(f"{'='*60}")

    gss_test = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
    train_val_idx, test_idx = next(gss_test.split(X, y, groups=player_ids))

    gss_val = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
    train_idx, val_idx = next(gss_val.split(
        X[train_val_idx],
        y[train_val_idx],
        groups=player_ids[train_val_idx]
    ))

    train_idx = train_val_idx[train_idx]
    val_idx = train_val_idx[val_idx]

    X_train, y_train = X[train_idx], y[train_idx]
    X_val, y_val = X[val_idx], y[val_idx]
    X_test, y_test = X[test_idx], y[test_idx]

    print(f"Train: {len(X_train):,} ({len(X_train)/len(X)*100:.1f}%)")
    print(f"Val: {len(X_val):,} ({len(X_val)/len(X)*100:.1f}%)")
    print(f"Test: {len(X_test):,} ({len(X_test)/len(X)*100:.1f}%)")

    # Create datasets
    train_dataset = PoseDataset(X_train, y_train)
    val_dataset = PoseDataset(X_val, y_val)
    test_dataset = PoseDataset(X_test, y_test)

    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=NUM_WORKERS,
        pin_memory=True if torch.cuda.is_available() else False
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=True if torch.cuda.is_available() else False
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=True if torch.cuda.is_available() else False
    )

    print(f"\n{'='*60}")
    print(f"TRAINING {args.model.upper()} MODEL")
    print(f"{'='*60}")

    # Create model
    if args.model == 'stgcn':
        A = get_adjacency_matrix(MEDIAPIPE_EDGES)
        model = SimpleSTGCN(num_classes=len(le.classes_), A=A).to(device)
    else:
        model = SimpleLSTM(num_classes=len(le.classes_)).to(device)

    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Loss and optimizer
    criterion = nn.CrossEntropyLoss(weight=class_weights_tensor)
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=WEIGHT_DECAY)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', patience=5, factor=0.5)

    # Training loop
    history = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': []}
    best_val_acc = 0
    patience_counter = 0

    start_time = time.time()

    for epoch in range(args.epochs):
        print(f"\nEpoch {epoch+1}/{args.epochs}")
        print("-" * 40)

        # Train
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device)

        # Validate
        val_loss, val_acc, _, _ = evaluate(model, val_loader, criterion, device)

        # Update scheduler
        scheduler.step(val_acc)

        # Save history
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)

        # Print results
        print(f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc*100:.2f}%")
        print(f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc*100:.2f}%")
        print(f"Learning rate: {optimizer.param_groups[0]['lr']:.6f}")

        # Early stopping
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            patience_counter = 0
            # Save best model
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_acc': val_acc,
                'history': history,
                'label_encoder': le
            }, f'{args.output}/models/{args.model}_best.pth')
            print(f"✓ New best model saved (val_acc: {val_acc*100:.2f}%)")
        else:
            patience_counter += 1
            print(f"Patience: {patience_counter}/{PATIENCE}")

        if patience_counter >= PATIENCE:
            print(f"\nEarly stopping triggered at epoch {epoch+1}")
            break

    training_time = time.time() - start_time

    print(f"\n{'='*60}")
    print("TRAINING COMPLETE")
    print(f"{'='*60}")
    print(f"Total time: {training_time/60:.1f} minutes")
    print(f"Best val accuracy: {best_val_acc*100:.2f}%")

    # Load best model
    checkpoint = torch.load(f'{args.output}/models/{args.model}_best.pth')
    model.load_state_dict(checkpoint['model_state_dict'])

    # Test evaluation
    print(f"\n{'='*60}")
    print("TEST SET EVALUATION")
    print(f"{'='*60}\n")

    test_loss, test_acc, test_preds, test_labels = evaluate(model, test_loader, criterion, device)

    print(f"Test Loss: {test_loss:.4f}")
    print(f"Test Accuracy: {test_acc*100:.2f}%")

    # Classification report
    print(f"\n{'='*60}")
    print("CLASSIFICATION REPORT")
    print(f"{'='*60}\n")

    report = classification_report(
        test_labels,
        test_preds,
        target_names=le.classes_,
        digits=4
    )
    print(report)

    # F1 scores
    f1_macro = f1_score(test_labels, test_preds, average='macro')
    f1_weighted = f1_score(test_labels, test_preds, average='weighted')

    print(f"Macro F1 Score: {f1_macro:.4f}")
    print(f"Weighted F1 Score: {f1_weighted:.4f}")

    # Save final report
    report_text = f"""
# {args.model.upper()} Training Report
Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

## Model Configuration
- Model: {args.model.upper()}
- Parameters: {sum(p.numel() for p in model.parameters()):,}
- Input: MediaPipe pose landmarks (33 keypoints × 3 coordinates)
- Sequence length: {TARGET_LENGTH} frames
- Classes: {len(le.classes_)} ({', '.join([c.capitalize() for c in le.classes_])})

## Dataset
- Total samples: {len(X):,}
- Train: {len(X_train):,} ({len(X_train)/len(X)*100:.1f}%)
- Val: {len(X_val):,} ({len(X_val)/len(X)*100:.1f}%)
- Test: {len(X_test):,} ({len(X_test)/len(X)*100:.1f}%)
- Filtered sequences: {len(df_valid) - len(pose_sequences)} (< {MIN_FRAMES} frames)

## Training Configuration
- Epochs: {len(history['train_loss'])} (early stopping)
- Batch size: {args.batch_size}
- Learning rate: {args.lr}
- Weight decay: {WEIGHT_DECAY}
- Optimizer: Adam
- Loss: Cross-Entropy (class-weighted)
- Training time: {training_time/60:.1f} minutes

## Results
- Best val accuracy: {best_val_acc*100:.2f}%
- Test accuracy: {test_acc*100:.2f}%
- Macro F1: {f1_macro:.4f}
- Weighted F1: {f1_weighted:.4f}

## Classification Report
{report}

## Preprocessing Applied
✓ Torso-centered normalization (hip center)
✓ Height-scaled coordinates
✓ Sequence length: {TARGET_LENGTH} frames
✓ Filtered short clips (< {MIN_FRAMES} frames)
✓ Player-based train/val/test split (no leakage)
✓ Class-weighted loss
"""

    with open(f'{args.output}/reports/{args.model}_report.txt', 'w') as f:
        f.write(report_text)

    print(f"\n✓ Report saved to {args.output}/reports/{args.model}_report.txt")
    print(f"✓ Model saved to {args.output}/models/{args.model}_best.pth")
    print(f"\n{'='*60}\n")


if __name__ == '__main__':
    main()
