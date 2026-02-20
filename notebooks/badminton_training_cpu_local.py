#!/usr/bin/env python3
"""
Badminton Shot Classification - CPU Training (Local)

Optimized for local CPU training without GPU.
Expected time: 12-24 hours on modern CPU.

Usage:
    python badminton_training_cpu_local.py [--max-cpu-percent 80]

Features:
- Small batch size (4) for CPU
- Lightweight MobileNetV3 model
- Single worker data loading
- Resume capability
- Progress tracking
- CPU throttling to prevent overload
- Graceful interrupt handling (Ctrl+C)
"""

import os
import sys
import time
import json
import signal
import argparse
from pathlib import Path
from datetime import datetime
from collections import Counter, defaultdict

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

import torchvision.transforms as transforms
import torchvision.models as models

import cv2
import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm

from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, f1_score

# ============================================================================
# GRACEFUL INTERRUPT HANDLER
# ============================================================================

class GracefulInterruptHandler:
    """Handle Ctrl+C gracefully and save checkpoint before exiting."""
    def __init__(self):
        self.interrupted = False
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)

    def _signal_handler(self, sig, frame):
        print("\n\n" + "="*70)
        print("🛑 INTERRUPT SIGNAL RECEIVED (Ctrl+C)")
        print("="*70)
        print("Finishing current batch and saving checkpoint...")
        print("Press Ctrl+C again to force quit (NOT RECOMMENDED)")
        print("="*70 + "\n")
        self.interrupted = True
        # Restore default handler for second Ctrl+C
        signal.signal(signal.SIGINT, signal.SIG_DFL)
        signal.signal(signal.SIGTERM, signal.SIG_DFL)

# ============================================================================
# CONFIGURATION
# ============================================================================

# Parse command line arguments
parser = argparse.ArgumentParser(description='CPU Training with Resource Control')
parser.add_argument('--max-cpu-percent', type=int, default=80,
                    help='Maximum CPU usage percentage (default: 80)')
parser.add_argument('--skip-ratio', type=float, default=0.0,
                    help='Skip first X%% of frames (0.0-0.5, default: 0.0)')
parser.add_argument('--resume', type=str, default=None,
                    help='Path to checkpoint to resume from')
args = parser.parse_args()

# Paths (CHANGE THESE to match your setup)
DATA_ROOT = "/Volumes/Ext/GenAI/iti123_v2/data/frames"  # Where .npy files are
RESULTS_DIR = "/Volumes/Ext/GenAI/iti123_v2/outputs/results_cpu_local"

# Training settings (CPU-optimized)
CONFIG = {
    'num_frames': 16,
    'frame_size': (224, 224),
    'num_classes': 5,
    'class_names': ['Clear', 'Drive', 'Drop', 'Lift', 'Smash'],

    # CPU-optimized settings
    'batch_size': 4,              # Small batch for CPU
    'num_epochs': 50,
    'learning_rate': 0.0001,      # Lower LR for stability
    'weight_decay': 0.0001,
    'early_stopping_patience': 10,
    'num_workers': 1,             # Single worker for CPU

    # Model settings
    'lstm_hidden_size': 64,       # Smaller LSTM for CPU
    'lstm_num_layers': 1,         # Single layer
    'lstm_dropout': 0.3,

    'device': 'cpu',              # Force CPU
    'save_every': 5,              # Save checkpoint every 5 epochs

    # Resource control
    'max_cpu_percent': args.max_cpu_percent,
    'skip_ratio': args.skip_ratio,
}

# Set PyTorch thread count to control CPU usage
max_threads = max(1, int(os.cpu_count() * args.max_cpu_percent / 100))
torch.set_num_threads(max_threads)

# Create results directory
os.makedirs(RESULTS_DIR, exist_ok=True)

# Initialize interrupt handler
interrupt_handler = GracefulInterruptHandler()

print("="*70)
print("Badminton Shot Classification - CPU Training")
print("="*70)
print(f"Data directory: {DATA_ROOT}")
print(f"Results directory: {RESULTS_DIR}")
print(f"Device: CPU (no GPU)")
print(f"Batch size: {CONFIG['batch_size']}")
print(f"CPU threads: {max_threads} / {os.cpu_count()} (limit: {args.max_cpu_percent}%)")
print(f"Skip ratio: {args.skip_ratio*100:.0f}% (focus on last {(1-args.skip_ratio)*100:.0f}% of frames)")
print(f"Expected time: 12-24 hours")
if args.resume:
    print(f"Resuming from: {args.resume}")
print("="*70)
print("💡 Press Ctrl+C anytime to save checkpoint and exit gracefully")
print("="*70)
print()

# ============================================================================
# DATASET CLASS
# ============================================================================

class BadmintonFramesDataset(Dataset):
    """
    Dataset for pre-extracted frames (.npy files).
    Supports optional targeted sampling (skip first X% of frames).
    """
    def __init__(self, npy_paths, labels, augment=False, skip_ratio=0.3):
        self.npy_paths = npy_paths
        self.labels = labels
        self.augment = augment
        self.skip_ratio = skip_ratio

        # ImageNet normalization
        self.normalize = transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )

        # Light augmentation for CPU (no heavy transforms)
        if augment:
            self.transform = transforms.Compose([
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.ColorJitter(brightness=0.1, contrast=0.1),
            ])
        else:
            self.transform = None

    def __len__(self):
        return len(self.npy_paths)

    def __getitem__(self, idx):
        # Load frames
        frames = np.load(self.npy_paths[idx])  # (T, H, W, C)
        label = self.labels[idx]

        # Optional: Skip first skip_ratio % of frames
        if self.skip_ratio > 0:
            total_frames = len(frames)
            start_idx = int(total_frames * self.skip_ratio)
            selected_indices = np.linspace(start_idx, total_frames - 1,
                                          total_frames, dtype=int)
            frames = frames[selected_indices]

        # Convert to tensors
        frames_tensor = []
        for frame in frames:
            frame_pil = Image.fromarray(frame)

            if self.transform:
                frame_pil = self.transform(frame_pil)

            frame_tensor = transforms.ToTensor()(frame_pil)
            frame_tensor = self.normalize(frame_tensor)
            frames_tensor.append(frame_tensor)

        return torch.stack(frames_tensor), label


# ============================================================================
# MODEL (Lightweight MobileNetV3 + LSTM)
# ============================================================================

class MobileNet_LSTM_Classifier(nn.Module):
    """
    MobileNetV3-Small + LSTM
    Lightweight model optimized for CPU training.
    ~4M parameters (vs 14M for ResNet18+BiLSTM)
    """
    def __init__(self, num_classes=5, hidden_size=64, num_lstm_layers=1, dropout=0.3):
        super(MobileNet_LSTM_Classifier, self).__init__()

        # MobileNetV3-Small backbone (lightweight)
        mobilenet = models.mobilenet_v3_small(pretrained=True)
        self.cnn = mobilenet.features
        self.cnn_feature_size = 576

        self.pool = nn.AdaptiveAvgPool2d(1)

        # Single-layer LSTM (not bidirectional for speed)
        self.lstm = nn.LSTM(
            input_size=self.cnn_feature_size,
            hidden_size=hidden_size,
            num_layers=num_lstm_layers,
            batch_first=True,
            bidirectional=False,
            dropout=dropout if num_lstm_layers > 1 else 0
        )

        # Classifier
        self.fc = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(hidden_size, num_classes)
        )

    def forward(self, x):
        batch_size, num_frames, c, h, w = x.size()

        # Process frames
        x = x.view(batch_size * num_frames, c, h, w)
        x = self.cnn(x)
        x = self.pool(x)
        x = x.view(batch_size * num_frames, -1)
        x = x.view(batch_size, num_frames, -1)

        # LSTM
        x, _ = self.lstm(x)
        x = x[:, -1, :]  # Take last output

        # Classification
        x = self.fc(x)
        return x


# ============================================================================
# DATA LOADING
# ============================================================================

def main():
    """Main training function"""
    print("Loading dataset...")
    frames_dir = Path(DATA_ROOT)

    if not frames_dir.exists():
        print(f"ERROR: Data directory not found: {DATA_ROOT}")
        print("Please set DATA_ROOT to the directory containing .npy files")
        sys.exit(1)

    all_npy_files = list(frames_dir.glob("*.npy"))
    print(f"Found {len(all_npy_files)} .npy files")

    if len(all_npy_files) == 0:
        print("ERROR: No .npy files found!")
        print("Please extract frames first using the frame extraction script")
        sys.exit(1)

    # Extract labels from filenames (format: ClassName_videoname.npy)
    class_to_idx = {name: idx for idx, name in enumerate(CONFIG['class_names'])}

    npy_paths = []
    labels = []

    for npy_path in all_npy_files:
        filename = npy_path.stem
        class_name = filename.split('_')[0]

        if class_name in class_to_idx:
            npy_paths.append(str(npy_path))
            labels.append(class_to_idx[class_name])

    print(f"Loaded {len(npy_paths)} samples")

    # Class distribution
    label_counts = Counter(labels)
    print("\nClass distribution:")
    for class_name in CONFIG['class_names']:
        idx = class_to_idx[class_name]
        count = label_counts.get(idx, 0)
        pct = 100 * count / len(labels) if len(labels) > 0 else 0
        print(f"  {class_name:8s}: {count:5d} ({pct:5.1f}%)")

    # Train/val/test split (70/10/20)
    print("\nCreating train/val/test split (70/10/20)...")
    train_paths, temp_paths, train_labels, temp_labels = train_test_split(
        npy_paths, labels, test_size=0.3, random_state=42, stratify=labels
    )

    val_paths, test_paths, val_labels, test_labels = train_test_split(
        temp_paths, temp_labels, test_size=0.67, random_state=42, stratify=temp_labels
    )

    print(f"  Train: {len(train_paths):5d} samples ({100*len(train_paths)/len(npy_paths):.1f}%)")
    print(f"  Val:   {len(val_paths):5d} samples ({100*len(val_paths)/len(npy_paths):.1f}%)")
    print(f"  Test:  {len(test_paths):5d} samples ({100*len(test_paths)/len(npy_paths):.1f}%)")

    # Create datasets
    print("\nCreating datasets...")
    train_dataset = BadmintonFramesDataset(
        train_paths, train_labels,
        augment=True,      # Light augmentation for training
        skip_ratio=0.0     # Change to 0.3 to skip first 30% of frames
    )
    val_dataset = BadmintonFramesDataset(val_paths, val_labels, augment=False)
    test_dataset = BadmintonFramesDataset(test_paths, test_labels, augment=False)

    # Create dataloaders (CPU-optimized)
    train_loader = DataLoader(
        train_dataset,
        batch_size=CONFIG['batch_size'],
        shuffle=True,
        num_workers=CONFIG['num_workers'],
        pin_memory=False  # Disable for CPU
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=CONFIG['batch_size'],
        shuffle=False,
        num_workers=CONFIG['num_workers'],
        pin_memory=False
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=CONFIG['batch_size'],
        shuffle=False,
        num_workers=CONFIG['num_workers'],
        pin_memory=False
    )

    print(f"  Train batches: {len(train_loader)}")
    print(f"  Val batches:   {len(val_loader)}")
    print(f"  Test batches:  {len(test_loader)}")


    # ============================================================================
    # MODEL SETUP
    # ============================================================================

    print("\nCreating model...")
    model = MobileNet_LSTM_Classifier(
        num_classes=CONFIG['num_classes'],
        hidden_size=CONFIG['lstm_hidden_size'],
        num_lstm_layers=CONFIG['lstm_num_layers'],
        dropout=CONFIG['lstm_dropout']
    )

    model = model.to(CONFIG['device'])

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    print(f"  Total parameters: {total_params:,}")
    print(f"  Trainable parameters: {trainable_params:,}")

    # Class weights for imbalanced data
    class_counts = Counter(train_labels)
    class_weights = []
    for idx in range(CONFIG['num_classes']):
        count = class_counts.get(idx, 1)
        weight = len(train_labels) / (CONFIG['num_classes'] * count)
        class_weights.append(weight)

    class_weights = torch.tensor(class_weights, dtype=torch.float32).to(CONFIG['device'])

    print(f"\nClass weights:")
    for idx, name in enumerate(CONFIG['class_names']):
        print(f"  {name:8s}: {class_weights[idx]:.2f}")

    # Loss, optimizer, scheduler
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = optim.Adam(
        model.parameters(),
        lr=CONFIG['learning_rate'],
        weight_decay=CONFIG['weight_decay']
    )
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='max', factor=0.5, patience=3, verbose=True
    )


    # ============================================================================
    # TRAINING FUNCTIONS
    # ============================================================================

    def train_epoch(model, dataloader, criterion, optimizer, device, interrupt_handler):
        """Train for one epoch with interrupt checking"""
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        pbar = tqdm(dataloader, desc='Training')

        for frames, labels in pbar:
            # Check for interrupt signal
            if interrupt_handler.interrupted:
                print("\n⚠️  Training interrupted by user")
                return running_loss / max(1, pbar.n), 100. * correct / max(1, total), True

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

        return running_loss / len(dataloader), 100. * correct / total, False


    def validate(model, dataloader, criterion, device):
        """Validate the model"""
        model.eval()
        running_loss = 0.0
        correct = 0
        total = 0

        with torch.no_grad():
            for frames, labels in tqdm(dataloader, desc='Validation'):
                frames = frames.to(device)
                labels = labels.to(device)

                outputs = model(frames)
                loss = criterion(outputs, labels)

                running_loss += loss.item()
                _, predicted = outputs.max(1)
                total += labels.size(0)
                correct += predicted.eq(labels).sum().item()

        return running_loss / len(dataloader), 100. * correct / total


    # ============================================================================
    # TRAINING LOOP
    # ============================================================================

    # Check for existing checkpoint
    checkpoint_path = os.path.join(RESULTS_DIR, 'checkpoint.pth')
    best_model_path = os.path.join(RESULTS_DIR, 'best_model.pth')

    start_epoch = 0
    best_val_acc = 0.0
    patience_counter = 0
    history = defaultdict(list)

    if os.path.exists(checkpoint_path):
        print(f"\nFound checkpoint: {checkpoint_path}")
        print("Resuming training...")
        checkpoint = torch.load(checkpoint_path, map_location=CONFIG['device'])
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
        best_val_acc = checkpoint['best_val_acc']
        history = checkpoint['history']
        patience_counter = checkpoint['patience_counter']
        print(f"  Resuming from epoch {start_epoch}")
        print(f"  Best val acc so far: {best_val_acc:.2f}%")

    print("\n" + "="*70)
    print("Starting Training")
    print("="*70)
    print(f"Device: {CONFIG['device']}")
    print(f"Batch size: {CONFIG['batch_size']}")
    print(f"Epochs: {start_epoch} to {CONFIG['num_epochs']}")
    print(f"Early stopping patience: {CONFIG['early_stopping_patience']}")
    print("="*70)
    print()

    start_time = datetime.now()

    for epoch in range(start_epoch, CONFIG['num_epochs']):
        epoch_start = time.time()

        print(f"\nEpoch {epoch+1}/{CONFIG['num_epochs']}")
        print("-" * 70)

        # Train
        train_loss, train_acc, interrupted = train_epoch(
            model, train_loader, criterion, optimizer, CONFIG['device'], interrupt_handler
        )

        # Check for interrupt
        if interrupted:
            print("\n" + "="*70)
            print("SAVING CHECKPOINT BEFORE EXIT")
            print("="*70)
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'best_val_acc': best_val_acc,
                'history': history,
                'patience_counter': patience_counter,
                'interrupted': True,
            }, checkpoint_path)
            print(f"✓ Checkpoint saved to: {checkpoint_path}")
            print(f"  Epoch: {epoch+1}/{CONFIG['num_epochs']}")
            print(f"  Best val acc: {best_val_acc:.2f}%")
            print(f"  To resume: python {sys.argv[0]} --resume {checkpoint_path}")
            print("="*70)
            sys.exit(0)

        # Validate
        val_loss, val_acc = validate(model, val_loader, criterion, CONFIG['device'])

        # Update history
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)

        # Scheduler step
        scheduler.step(val_acc)
        current_lr = optimizer.param_groups[0]['lr']

        epoch_time = time.time() - epoch_start

        # Print summary
        print(f"\nEpoch {epoch+1} Summary:")
        print(f"  Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}%")
        print(f"  Val Loss:   {val_loss:.4f} | Val Acc:   {val_acc:.2f}%")
        print(f"  Gap: {train_acc - val_acc:.2f}%")
        print(f"  LR: {current_lr:.6f}")
        print(f"  Time: {epoch_time/60:.1f} minutes")

        # Estimate remaining time
        if epoch > start_epoch:
            avg_epoch_time = (time.time() - start_time.timestamp()) / (epoch - start_epoch + 1)
            remaining_epochs = CONFIG['num_epochs'] - epoch - 1
            est_remaining = avg_epoch_time * remaining_epochs / 3600
            print(f"  Est. remaining: {est_remaining:.1f} hours")

        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            patience_counter = 0
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'val_acc': val_acc,
            }, best_model_path)
            print(f"  ✓ Saved best model (val_acc: {val_acc:.2f}%)")
        else:
            patience_counter += 1
            print(f"  No improvement ({patience_counter}/{CONFIG['early_stopping_patience']})")

        # Save checkpoint (always save after successful epoch)
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'best_val_acc': best_val_acc,
            'history': history,
            'patience_counter': patience_counter,
            'interrupted': False,
        }, checkpoint_path)

        if (epoch + 1) % CONFIG['save_every'] == 0:
            print(f"  ✓ Checkpoint saved")

        # Early stopping
        if patience_counter >= CONFIG['early_stopping_patience']:
            print(f"\n🛑 Early stopping at epoch {epoch+1}")
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


    # ============================================================================
    # EVALUATION
    # ============================================================================

    print("\nEvaluating on test set...")

    # Load best model
    checkpoint = torch.load(best_model_path, map_location=CONFIG['device'])
    model.load_state_dict(checkpoint['model_state_dict'])

    model.eval()

    all_predictions = []
    all_labels = []
    test_correct = 0
    test_total = 0

    with torch.no_grad():
        for frames, labels in tqdm(test_loader, desc='Testing'):
            frames = frames.to(CONFIG['device'])
            labels = labels.to(CONFIG['device'])

            outputs = model(frames)
            _, predicted = outputs.max(1)

            all_predictions.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

            test_total += labels.size(0)
            test_correct += predicted.eq(labels).sum().item()

    test_acc = 100. * test_correct / test_total

    print(f"\n{'='*70}")
    print("Test Set Results")
    print(f"{'='*70}")
    print(f"Test Accuracy: {test_acc:.2f}%")
    print(f"Correct: {test_correct}/{test_total}")

    # Classification report
    print("\nClassification Report:")
    print("="*70)
    report = classification_report(
        all_labels,
        all_predictions,
        target_names=CONFIG['class_names'],
        digits=4
    )
    print(report)

    # Confusion matrix
    cm = confusion_matrix(all_labels, all_predictions)
    print("\nConfusion Matrix:")
    print("="*70)
    cm_df = pd.DataFrame(cm, index=CONFIG['class_names'], columns=CONFIG['class_names'])
    print(cm_df)


    # ============================================================================
    # SAVE RESULTS
    # ============================================================================

    print("\nSaving results...")

    # Save classification report
    with open(os.path.join(RESULTS_DIR, 'classification_report.txt'), 'w') as f:
        f.write(f"Test Accuracy: {test_acc:.2f}%\n\n")
        f.write("Confusion Matrix:\n")
        f.write(str(cm_df))
        f.write("\n\nClassification Report:\n")
        f.write(report)

    # Save results summary
    results_summary = {
        'model': 'MobileNetV3_LSTM_CPU',
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
            'total_samples': len(npy_paths),
        },
        'config': {
            'batch_size': CONFIG['batch_size'],
            'learning_rate': CONFIG['learning_rate'],
            'num_frames': CONFIG['num_frames'],
            'frame_size': CONFIG['frame_size'],
            'device': CONFIG['device'],
        },
        'confusion_matrix': cm.tolist(),
        'class_names': CONFIG['class_names'],
    }

    with open(os.path.join(RESULTS_DIR, 'results_summary.json'), 'w') as f:
        json.dump(results_summary, f, indent=2)

    # Save training history
    pd.DataFrame(history).to_csv(
        os.path.join(RESULTS_DIR, 'training_history.csv'),
        index=False
    )

    print(f"\n✓ Results saved to: {RESULTS_DIR}")
    print(f"  - best_model.pth")
    print(f"  - classification_report.txt")
    print(f"  - results_summary.json")
    print(f"  - training_history.csv")

    print("\n" + "="*70)
    print("DONE!")
    print("="*70)


if __name__ == '__main__':
    main()
