"""
Simple LSTM Baseline Model for Badminton Shot Classification
==============================================================

This script trains a simple LSTM-only model (no CNN) as a baseline comparison.

Purpose: Demonstrate that CNN+LSTM architecture is superior to vanilla LSTM.

Model Architecture:
    - NO CNN feature extraction
    - Direct frame pixel features → Flatten → LSTM → Classifier
    - Much simpler than ResNet18+BiLSTM

Expected Performance: ~40-50% accuracy (worse than 74.6% with CNN+LSTM)

Usage:
    python notebooks/badminton_training_simple_baseline.py

Author: ITI123 Badminton Shot Classification Project
Date: 2026-02-11
"""

import os
import sys
import time
import json
import signal
from datetime import datetime
from pathlib import Path
from collections import Counter

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns


# ============================================================================
# Configuration
# ============================================================================

CONFIG = {
    'data_root': '/Volumes/Ext/GenAI/iti123_v2',
    'frames_dir': 'data/frames_npy',
    'metadata_csv': 'data/metadata.csv',
    'output_dir': 'outputs/results_simple_baseline',

    # Model settings
    'num_frames': 16,
    'frame_size': (224, 224),
    'hidden_size': 128,       # Small LSTM hidden size
    'num_lstm_layers': 2,
    'dropout': 0.3,

    # Training settings
    'batch_size': 32,         # Smaller batch (simpler model)
    'num_epochs': 50,
    'learning_rate': 0.001,
    'weight_decay': 0.0001,
    'early_stopping_patience': 10,

    # Data split
    'test_size': 0.2,
    'val_size': 0.1,
    'random_state': 42,
}

SHOT_TYPES = ['Clear', 'Drive', 'Drop', 'Lift', 'Smash']


# ============================================================================
# Simple LSTM Model (No CNN)
# ============================================================================

class SimpleLSTMClassifier(nn.Module):
    """
    Simple LSTM-only baseline model.

    Architecture:
        Raw frames → Flatten → LSTM → Classifier

    No CNN feature extraction - just pure LSTM on pixel values.
    This will perform poorly compared to CNN+LSTM, which is the point!
    """
    def __init__(self, num_classes=5, frame_size=(224, 224),
                 hidden_size=128, num_lstm_layers=2, dropout=0.3):
        super(SimpleLSTMClassifier, self).__init__()

        self.frame_size = frame_size

        # Input: flattened frame pixels (3 * 224 * 224 = 150,528)
        self.input_size = 3 * frame_size[0] * frame_size[1]

        # Simple fully connected layer to reduce dimensionality
        self.fc_input = nn.Sequential(
            nn.Linear(self.input_size, 512),
            nn.ReLU(),
            nn.Dropout(dropout)
        )

        # LSTM (no bidirectional to keep it simple)
        self.lstm = nn.LSTM(
            input_size=512,
            hidden_size=hidden_size,
            num_layers=num_lstm_layers,
            batch_first=True,
            dropout=dropout if num_lstm_layers > 1 else 0
        )

        # Classifier
        self.fc = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(hidden_size, num_classes)
        )

    def forward(self, x):
        """
        Args:
            x: (batch_size, num_frames, channels, height, width)

        Returns:
            logits: (batch_size, num_classes)
        """
        batch_size, num_frames, c, h, w = x.size()

        # Flatten each frame: (batch * frames, 3*224*224)
        x = x.view(batch_size * num_frames, -1)

        # Reduce dimensionality
        x = self.fc_input(x)  # (batch * frames, 512)

        # Reshape for LSTM: (batch, frames, 512)
        x = x.view(batch_size, num_frames, -1)

        # LSTM
        x, _ = self.lstm(x)  # (batch, frames, hidden_size)

        # Use last timestep
        x = x[:, -1, :]  # (batch, hidden_size)

        # Classification
        x = self.fc(x)  # (batch, num_classes)

        return x


# ============================================================================
# Dataset
# ============================================================================

class BadmintonFramesDataset(Dataset):
    """Simple dataset for .npy frame loading"""

    def __init__(self, npy_paths, labels):
        self.npy_paths = npy_paths
        self.labels = labels

        # Normalization (ImageNet stats)
        self.mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
        self.std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)

    def __len__(self):
        return len(self.npy_paths)

    def __getitem__(self, idx):
        # Load frames
        frames = np.load(self.npy_paths[idx])  # (T, H, W, C)
        label = self.labels[idx]

        # Convert to tensor
        frames_tensor = torch.from_numpy(frames).permute(0, 3, 1, 2).float() / 255.0

        # Normalize
        frames_tensor = (frames_tensor - self.mean) / self.std

        return frames_tensor, label


# ============================================================================
# Training Functions
# ============================================================================

def train_one_epoch(model, train_loader, criterion, optimizer, device):
    """Train for one epoch"""
    model.train()

    running_loss = 0.0
    correct = 0
    total = 0

    pbar = tqdm(train_loader, desc='Training')

    for frames, labels in pbar:
        frames = frames.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()

        outputs = model(frames)
        loss = criterion(outputs, labels)

        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()

        pbar.set_postfix({
            'loss': running_loss / (pbar.n + 1),
            'acc': 100. * correct / total
        })

    epoch_loss = running_loss / len(train_loader)
    epoch_acc = 100. * correct / total

    return epoch_loss, epoch_acc


def validate(model, val_loader, criterion, device):
    """Validate the model"""
    model.eval()

    running_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for frames, labels in tqdm(val_loader, desc='Validation'):
            frames = frames.to(device)
            labels = labels.to(device)

            outputs = model(frames)
            loss = criterion(outputs, labels)

            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

    epoch_loss = running_loss / len(val_loader)
    epoch_acc = 100. * correct / total

    return epoch_loss, epoch_acc


# ============================================================================
# Main Training Loop
# ============================================================================

def main():
    """Main training function"""

    print("="*70)
    print("Simple LSTM Baseline Training")
    print("="*70)
    print(f"Model: Simple LSTM (no CNN)")
    print(f"Purpose: Baseline comparison to show CNN+LSTM is better")
    print(f"Expected accuracy: ~40-50% (vs 74.6% with CNN+LSTM)")
    print("="*70)
    print()

    # Create output directory
    output_dir = Path(CONFIG['output_dir'])
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load data
    print("Loading dataset...")
    data_root = Path(CONFIG['data_root'])
    frames_dir = data_root / CONFIG['frames_dir']

    # Get all .npy files
    all_npy_files = list(frames_dir.glob("*.npy"))
    print(f"Found {len(all_npy_files)} .npy files")

    # Extract labels from filenames
    npy_paths = []
    labels = []
    class_to_idx = {shot: idx for idx, shot in enumerate(SHOT_TYPES)}

    for npy_path in all_npy_files:
        filename = npy_path.stem
        class_name = filename.split('_')[0]

        if class_name in class_to_idx:
            npy_paths.append(str(npy_path))
            labels.append(class_to_idx[class_name])

    print(f"Usable samples: {len(npy_paths)}")

    # Class distribution
    label_counts = Counter(labels)
    print("\nClass distribution:")
    for idx, shot in enumerate(SHOT_TYPES):
        count = label_counts.get(idx, 0)
        print(f"  {shot:8s}: {count:5d}")
    print()

    # Train/val/test split
    train_paths, temp_paths, train_labels, temp_labels = train_test_split(
        npy_paths, labels, test_size=0.3, random_state=42, stratify=labels
    )

    val_paths, test_paths, val_labels, test_labels = train_test_split(
        temp_paths, temp_labels, test_size=0.67, random_state=42, stratify=temp_labels
    )

    print("Data splits:")
    print(f"  Train: {len(train_paths)}")
    print(f"  Val:   {len(val_paths)}")
    print(f"  Test:  {len(test_paths)}")
    print()

    # Create datasets
    train_dataset = BadmintonFramesDataset(train_paths, train_labels)
    val_dataset = BadmintonFramesDataset(val_paths, val_labels)
    test_dataset = BadmintonFramesDataset(test_paths, test_labels)

    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=CONFIG['batch_size'],
        shuffle=True,
        num_workers=2
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=CONFIG['batch_size'],
        shuffle=False,
        num_workers=2
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=CONFIG['batch_size'],
        shuffle=False,
        num_workers=2
    )

    # Create model
    print("Creating Simple LSTM model...")
    model = SimpleLSTMClassifier(
        num_classes=len(SHOT_TYPES),
        frame_size=CONFIG['frame_size'],
        hidden_size=CONFIG['hidden_size'],
        num_lstm_layers=CONFIG['num_lstm_layers'],
        dropout=CONFIG['dropout']
    )

    num_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {num_params:,}")
    print()

    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    model = model.to(device)

    # Class weights (for imbalanced dataset)
    class_counts = Counter(train_labels)
    total_samples = len(train_labels)
    class_weights = []
    for idx in range(len(SHOT_TYPES)):
        count = class_counts.get(idx, 0)
        if count > 0:
            weight = total_samples / (len(SHOT_TYPES) * count)
        else:
            weight = 1.0
        class_weights.append(weight)

    class_weights = torch.tensor(class_weights, dtype=torch.float32).to(device)

    # Loss, optimizer, scheduler
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = optim.Adam(
        model.parameters(),
        lr=CONFIG['learning_rate'],
        weight_decay=CONFIG['weight_decay']
    )
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='max', factor=0.5, patience=5
    )

    # Training loop
    print("="*70)
    print("Starting Training")
    print("="*70)
    print()

    history = {
        'train_loss': [],
        'train_acc': [],
        'val_loss': [],
        'val_acc': [],
    }

    best_val_acc = 0.0
    patience_counter = 0
    best_model_path = output_dir / 'best_model.pth'

    start_time = datetime.now()

    for epoch in range(CONFIG['num_epochs']):
        print(f"\nEpoch {epoch+1}/{CONFIG['num_epochs']}")
        print("-" * 70)

        # Train
        train_loss, train_acc = train_one_epoch(
            model, train_loader, criterion, optimizer, device
        )

        # Validate
        val_loss, val_acc = validate(model, val_loader, criterion, device)

        # Update history
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)

        # Print summary
        print(f"\nEpoch {epoch+1} Summary:")
        print(f"  Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}%")
        print(f"  Val Loss:   {val_loss:.4f} | Val Acc:   {val_acc:.2f}%")

        # Learning rate scheduler
        scheduler.step(val_acc)
        current_lr = optimizer.param_groups[0]['lr']
        print(f"  Learning rate: {current_lr:.6f}")

        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_acc': val_acc,
                'history': history
            }, best_model_path)
            print(f"  ✓ Saved best model (val_acc: {val_acc:.2f}%)")
            patience_counter = 0
        else:
            patience_counter += 1
            print(f"  No improvement ({patience_counter}/{CONFIG['early_stopping_patience']})")

        # Early stopping
        if patience_counter >= CONFIG['early_stopping_patience']:
            print(f"\n🛑 Early stopping triggered after {epoch+1} epochs")
            print(f"   Best val accuracy: {best_val_acc:.2f}%")
            break

    end_time = datetime.now()
    training_duration = end_time - start_time

    print("\n" + "="*70)
    print("Training Complete!")
    print("="*70)
    print(f"Total time: {training_duration}")
    print(f"Best val accuracy: {best_val_acc:.2f}%")
    print("="*70)
    print()

    # Evaluation
    print("Evaluating on test set...")
    checkpoint = torch.load(best_model_path)
    model.load_state_dict(checkpoint['model_state_dict'])

    model.eval()
    all_predictions = []
    all_labels = []
    test_correct = 0
    test_total = 0

    with torch.no_grad():
        for frames, labels in tqdm(test_loader, desc='Testing'):
            frames = frames.to(device)
            labels = labels.to(device)

            outputs = model(frames)
            _, predicted = outputs.max(1)

            all_predictions.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

            test_total += labels.size(0)
            test_correct += predicted.eq(labels).sum().item()

    test_acc = 100. * test_correct / test_total

    print(f"\nTest Accuracy: {test_acc:.2f}%")
    print(f"Correct: {test_correct}/{test_total}")
    print()

    # Confusion matrix
    cm = confusion_matrix(all_labels, all_predictions)
    print("Confusion Matrix:")
    cm_df = pd.DataFrame(cm, index=SHOT_TYPES, columns=SHOT_TYPES)
    print(cm_df)
    print()

    # Classification report
    report = classification_report(
        all_labels, all_predictions,
        target_names=SHOT_TYPES, digits=4
    )
    print("Classification Report:")
    print(report)

    # Save results
    with open(output_dir / 'classification_report.txt', 'w') as f:
        f.write(f"Test Accuracy: {test_acc:.2f}%\n\n")
        f.write("Confusion Matrix:\n")
        f.write(str(cm_df))
        f.write("\n\nClassification Report:\n")
        f.write(report)

    # Save summary
    results_summary = {
        'model': 'SimpleLSTM',
        'timestamp': datetime.now().isoformat(),
        'training': {
            'total_epochs': len(history['train_loss']),
            'best_val_acc': float(best_val_acc),
            'final_train_acc': float(history['train_acc'][-1]),
            'final_val_acc': float(history['val_acc'][-1]),
        },
        'test': {
            'accuracy': float(test_acc),
            'total_samples': int(test_total),
            'correct': int(test_correct),
        },
        'dataset': {
            'train_samples': len(train_dataset),
            'val_samples': len(val_dataset),
            'test_samples': len(test_dataset),
        },
        'config': CONFIG,
        'confusion_matrix': cm.tolist(),
        'class_names': SHOT_TYPES,
    }

    with open(output_dir / 'results_summary.json', 'w') as f:
        json.dump(results_summary, f, indent=2)

    # Plot confusion matrix
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=SHOT_TYPES, yticklabels=SHOT_TYPES)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title(f'Simple LSTM Baseline - Test Accuracy: {test_acc:.2f}%')
    plt.tight_layout()
    plt.savefig(output_dir / 'confusion_matrix.png', dpi=300, bbox_inches='tight')
    print(f"✓ Confusion matrix saved")

    # Plot training history
    fig, axes = plt.subplots(1, 2, figsize=(15, 5))

    axes[0].plot(history['train_loss'], label='Train Loss')
    axes[0].plot(history['val_loss'], label='Val Loss')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss')
    axes[0].set_title('Training and Validation Loss')
    axes[0].legend()
    axes[0].grid(True)

    axes[1].plot(history['train_acc'], label='Train Acc')
    axes[1].plot(history['val_acc'], label='Val Acc')
    axes[1].axhline(y=test_acc, color='r', linestyle='--',
                    label=f'Test Acc ({test_acc:.2f}%)')
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Accuracy (%)')
    axes[1].set_title('Training and Validation Accuracy')
    axes[1].legend()
    axes[1].grid(True)

    plt.tight_layout()
    plt.savefig(output_dir / 'training_history.png', dpi=300, bbox_inches='tight')
    print(f"✓ Training history saved")

    print(f"\n✓ All results saved to: {output_dir}")
    print("\nDone!")


if __name__ == '__main__':
    main()
