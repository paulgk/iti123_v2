"""
Simple LSTM Baseline Model - GPU OPTIMIZED Version
===================================================

This script trains a simple LSTM-only model with optimizations for better GPU utilization.

Optimizations:
    - Larger batch size (64 vs 32)
    - Mixed precision training (AMP)
    - Gradient accumulation
    - Multiple DataLoader workers
    - Persistent workers
    - Pin memory
    - Non-blocking transfers
    - Gradient clipping

Expected Performance: ~40-50% accuracy (baseline)
Expected Speed: ~2-3x faster than unoptimized version

Usage:
    python notebooks/badminton_training_simple_baseline_optimized.py

Author: ITI123 Badminton Shot Classification Project
Date: 2026-02-11
"""

import os
import sys
import time
import json
from datetime import datetime
from pathlib import Path
from collections import Counter

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.cuda.amp import autocast, GradScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns


# ============================================================================
# Configuration - OPTIMIZED
# ============================================================================

CONFIG = {
    'data_root': '/Volumes/Ext/GenAI/iti123_v2',
    'frames_dir': 'data/frames_npy',
    'output_dir': 'outputs/results_simple_baseline_optimized',

    # Model settings
    'num_frames': 16,
    'frame_size': (224, 224),
    'hidden_size': 128,
    'num_lstm_layers': 2,
    'dropout': 0.3,

    # Training settings - OPTIMIZED
    'batch_size': 64,              # ⚡ Increased from 32 (better GPU util)
    'gradient_accumulation': 2,    # ⚡ Effective batch = 128
    'num_epochs': 50,
    'learning_rate': 0.001,
    'weight_decay': 0.0001,
    'early_stopping_patience': 10,
    'gradient_clip': 1.0,          # ⚡ Prevent exploding gradients

    # DataLoader settings - OPTIMIZED
    'num_workers': 4,              # ⚡ Parallel data loading
    'prefetch_factor': 4,          # ⚡ Prefetch batches
    'persistent_workers': True,    # ⚡ Keep workers alive
    'pin_memory': True,            # ⚡ Faster GPU transfer

    # Mixed precision - OPTIMIZED
    'use_amp': True,               # ⚡ Automatic Mixed Precision

    'random_state': 42,
}

SHOT_TYPES = ['Clear', 'Drive', 'Drop', 'Lift', 'Smash']


# ============================================================================
# Optimized Dataset with Vectorized Operations
# ============================================================================

class BadmintonFramesDatasetOptimized(Dataset):
    """
    Optimized dataset with vectorized tensor operations.
    No PIL overhead - direct NumPy → Tensor conversion.
    """

    def __init__(self, npy_paths, labels):
        self.npy_paths = npy_paths
        self.labels = labels

        # Pre-compute normalization tensors
        self.mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
        self.std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)

    def __len__(self):
        return len(self.npy_paths)

    def __getitem__(self, idx):
        # Load frames
        frames = np.load(self.npy_paths[idx])  # (T, H, W, C), uint8
        label = self.labels[idx]

        # ⚡ Direct NumPy → Tensor (no PIL!)
        frames_tensor = torch.from_numpy(frames).permute(0, 3, 1, 2).float() / 255.0

        # ⚡ Vectorized normalization
        frames_tensor = (frames_tensor - self.mean) / self.std

        return frames_tensor, label


# ============================================================================
# Simple LSTM Model (No CNN) - Same as before
# ============================================================================

class SimpleLSTMClassifier(nn.Module):
    """
    Simple LSTM-only baseline model.

    Architecture:
        Raw frames → Flatten → FC → LSTM → Classifier

    No CNN feature extraction.
    """
    def __init__(self, num_classes=5, frame_size=(224, 224),
                 hidden_size=128, num_lstm_layers=2, dropout=0.3):
        super(SimpleLSTMClassifier, self).__init__()

        self.frame_size = frame_size
        self.input_size = 3 * frame_size[0] * frame_size[1]

        # FC layer to reduce dimensionality
        self.fc_input = nn.Sequential(
            nn.Linear(self.input_size, 512),
            nn.ReLU(),
            nn.Dropout(dropout)
        )

        # LSTM
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
        batch_size, num_frames, c, h, w = x.size()

        # Flatten each frame
        x = x.view(batch_size * num_frames, -1)

        # Reduce dimensionality
        x = self.fc_input(x)

        # Reshape for LSTM
        x = x.view(batch_size, num_frames, -1)

        # LSTM
        x, _ = self.lstm(x)

        # Use last timestep
        x = x[:, -1, :]

        # Classification
        x = self.fc(x)

        return x


# ============================================================================
# Optimized Training Functions
# ============================================================================

def train_one_epoch_optimized(model, train_loader, criterion, optimizer,
                              scaler, device, gradient_accumulation, gradient_clip):
    """
    Train for one epoch with GPU optimizations:
    - Mixed precision (AMP)
    - Gradient accumulation
    - Gradient clipping
    """
    model.train()

    running_loss = 0.0
    correct = 0
    total = 0

    optimizer.zero_grad()

    pbar = tqdm(train_loader, desc='Training')

    for batch_idx, (frames, labels) in enumerate(pbar):
        # ⚡ Non-blocking transfer
        frames = frames.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        # ⚡ Mixed precision forward pass
        with autocast(enabled=CONFIG['use_amp']):
            outputs = model(frames)
            loss = criterion(outputs, labels)
            loss = loss / gradient_accumulation  # Scale loss

        # ⚡ Mixed precision backward pass
        scaler.scale(loss).backward()

        # Gradient accumulation step
        if (batch_idx + 1) % gradient_accumulation == 0:
            # ⚡ Gradient clipping
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clip)

            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()

        # Statistics
        running_loss += loss.item() * gradient_accumulation
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()

        pbar.set_postfix({
            'loss': running_loss / (batch_idx + 1),
            'acc': 100. * correct / total
        })

    epoch_loss = running_loss / len(train_loader)
    epoch_acc = 100. * correct / total

    return epoch_loss, epoch_acc


def validate_optimized(model, val_loader, criterion, device):
    """Validate with mixed precision"""
    model.eval()

    running_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for frames, labels in tqdm(val_loader, desc='Validation'):
            # ⚡ Non-blocking transfer
            frames = frames.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            # ⚡ Mixed precision inference
            with autocast(enabled=CONFIG['use_amp']):
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
    print("Simple LSTM Baseline Training - GPU OPTIMIZED")
    print("="*70)
    print(f"Model: Simple LSTM (no CNN)")
    print(f"Optimizations: AMP, Gradient Accumulation, Parallel Loading")
    print(f"Expected: ~40-50% accuracy, 2-3x faster training")
    print("="*70)
    print()

    # Create output directory
    output_dir = Path(CONFIG['output_dir'])
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load data
    print("Loading dataset...")
    data_root = Path(CONFIG['data_root'])
    frames_dir = data_root / CONFIG['frames_dir']

    all_npy_files = list(frames_dir.glob("*.npy"))
    print(f"Found {len(all_npy_files)} .npy files")

    # Extract labels
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

    # ⚡ Create optimized datasets
    train_dataset = BadmintonFramesDatasetOptimized(train_paths, train_labels)
    val_dataset = BadmintonFramesDatasetOptimized(val_paths, val_labels)
    test_dataset = BadmintonFramesDatasetOptimized(test_paths, test_labels)

    # ⚡ Create optimized dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=CONFIG['batch_size'],
        shuffle=True,
        num_workers=CONFIG['num_workers'],
        pin_memory=CONFIG['pin_memory'],
        prefetch_factor=CONFIG['prefetch_factor'],
        persistent_workers=CONFIG['persistent_workers']
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=CONFIG['batch_size'],
        shuffle=False,
        num_workers=CONFIG['num_workers'],
        pin_memory=CONFIG['pin_memory'],
        prefetch_factor=CONFIG['prefetch_factor'],
        persistent_workers=CONFIG['persistent_workers']
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=CONFIG['batch_size'],
        shuffle=False,
        num_workers=CONFIG['num_workers'],
        pin_memory=CONFIG['pin_memory'],
        prefetch_factor=CONFIG['prefetch_factor'],
        persistent_workers=CONFIG['persistent_workers']
    )

    print("Optimized DataLoaders created:")
    print(f"  Batch size: {CONFIG['batch_size']}")
    print(f"  Gradient accumulation: {CONFIG['gradient_accumulation']}")
    print(f"  Effective batch size: {CONFIG['batch_size'] * CONFIG['gradient_accumulation']}")
    print(f"  Workers: {CONFIG['num_workers']}")
    print(f"  Prefetch factor: {CONFIG['prefetch_factor']}")
    print(f"  Persistent workers: {CONFIG['persistent_workers']}")
    print()

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

    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\nDevice: {device}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")

    model = model.to(device)

    # Class weights
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

    # ⚡ Mixed precision scaler
    scaler = GradScaler(enabled=CONFIG['use_amp'])

    print("\nOptimizations enabled:")
    print(f"  ✓ Mixed Precision (AMP): {CONFIG['use_amp']}")
    print(f"  ✓ Gradient Accumulation: {CONFIG['gradient_accumulation']}x")
    print(f"  ✓ Gradient Clipping: {CONFIG['gradient_clip']}")
    print(f"  ✓ Parallel Data Loading: {CONFIG['num_workers']} workers")
    print(f"  ✓ Persistent Workers: {CONFIG['persistent_workers']}")
    print()

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

        # Train with optimizations
        train_loss, train_acc = train_one_epoch_optimized(
            model, train_loader, criterion, optimizer, scaler, device,
            CONFIG['gradient_accumulation'], CONFIG['gradient_clip']
        )

        # Validate with optimizations
        val_loss, val_acc = validate_optimized(
            model, val_loader, criterion, device
        )

        # Update history
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)

        # Print summary
        print(f"\nEpoch {epoch+1} Summary:")
        print(f"  Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}%")
        print(f"  Val Loss:   {val_loss:.4f} | Val Acc:   {val_acc:.2f}%")

        # GPU memory usage
        if torch.cuda.is_available():
            allocated = torch.cuda.memory_allocated() / 1e9
            reserved = torch.cuda.memory_reserved() / 1e9
            print(f"  GPU Memory: {allocated:.2f}GB allocated, {reserved:.2f}GB reserved")

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

        # Clear cache
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

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
            frames = frames.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            with autocast(enabled=CONFIG['use_amp']):
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
    cm_df = pd.DataFrame(cm, index=SHOT_TYPES, columns=SHOT_TYPES)

    # Classification report
    report = classification_report(
        all_labels, all_predictions,
        target_names=SHOT_TYPES, digits=4
    )

    # Save results
    with open(output_dir / 'classification_report.txt', 'w') as f:
        f.write(f"Test Accuracy: {test_acc:.2f}%\n\n")
        f.write("Confusion Matrix:\n")
        f.write(str(cm_df))
        f.write("\n\nClassification Report:\n")
        f.write(report)

    # Save summary
    results_summary = {
        'model': 'SimpleLSTM_Optimized',
        'timestamp': datetime.now().isoformat(),
        'training': {
            'total_epochs': len(history['train_loss']),
            'best_val_acc': float(best_val_acc),
            'final_train_acc': float(history['train_acc'][-1]),
            'final_val_acc': float(history['val_acc'][-1]),
            'training_time': str(training_duration),
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
        'comparison': {
            'simple_lstm_accuracy': float(test_acc),
            'cnn_lstm_accuracy': 74.6,
            'improvement_from_cnn': float(74.6 - test_acc)
        }
    }

    with open(output_dir / 'results_summary.json', 'w') as f:
        json.dump(results_summary, f, indent=2)

    # Plot confusion matrix
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=SHOT_TYPES, yticklabels=SHOT_TYPES)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title(f'Simple LSTM Baseline (Optimized) - Test Accuracy: {test_acc:.2f}%')
    plt.tight_layout()
    plt.savefig(output_dir / 'confusion_matrix.png', dpi=300, bbox_inches='tight')

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

    print(f"\n✓ All results saved to: {output_dir}")
    print("\nOptimization Summary:")
    print(f"  Training time: {training_duration}")
    print(f"  Test accuracy: {test_acc:.2f}%")
    print(f"  Comparison to CNN+LSTM: +{74.6 - test_acc:.1f}pp from adding CNN")
    print("\nDone!")


if __name__ == '__main__':
    main()
