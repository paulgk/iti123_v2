"""
Badminton Shot Classification - Video-Based Approach
2D CNN + LSTM Architecture

This implementation uses video frames instead of pose data to classify badminton shots.
Expected accuracy: 70-80% (vs 38% with pose-only approach)

Author: Generated for ITI123 Project
Date: 2026-02-04
"""

import os
import cv2
import numpy as np
import pandas as pd
from pathlib import Path
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, f1_score
import torchvision.transforms as transforms
import torchvision.models as models
from collections import defaultdict
import json

# ==================== CONFIGURATION ====================

CONFIG = {
    # Data
    'data_root': 'data/clips',
    'num_frames': 16,              # Sample 16 frames per clip
    'frame_size': (224, 224),      # ResNet input size
    'num_classes': 5,
    'class_names': ['Clear', 'Drive', 'Drop', 'Lift', 'Smash'],

    # Training
    'batch_size': 16,
    'num_epochs': 50,
    'learning_rate': 0.0001,
    'weight_decay': 0.0001,
    'early_stopping_patience': 10,

    # Model
    'lstm_hidden_size': 256,
    'lstm_num_layers': 2,
    'lstm_dropout': 0.5,
    'freeze_cnn': True,            # Freeze CNN initially

    # Device
    'device': 'cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu',
    'num_workers': 4,

    # Focal Loss
    'use_focal_loss': True,
    'focal_gamma': 2.0,

    # Output
    'output_dir': 'models/video_classification',
    'save_best_model': True,
}

print(f"Device: {CONFIG['device']}")


# ==================== DATASET ====================

class BadmintonVideoDataset(Dataset):
    """
    Dataset for loading video clips organized in subdirectories by shot type.
    Structure: data/clips/{shot_type}/*.mp4
    """

    def __init__(self, video_paths, labels, num_frames=16, frame_size=(224, 224),
                 transform=None, augment=False):
        """
        Args:
            video_paths: List of video file paths
            labels: List of integer labels (0-4)
            num_frames: Number of frames to sample per video
            frame_size: Target frame size (H, W)
            transform: Optional torchvision transform
            augment: Apply data augmentation
        """
        self.video_paths = video_paths
        self.labels = labels
        self.num_frames = num_frames
        self.frame_size = frame_size
        self.augment = augment

        # Basic normalization (ImageNet stats)
        if transform is None:
            normalize = transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )

            if augment:
                self.transform = transforms.Compose([
                    transforms.ToPILImage(),
                    transforms.RandomHorizontalFlip(p=0.5),
                    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
                    transforms.RandomRotation(degrees=5),
                    transforms.ToTensor(),
                    normalize
                ])
            else:
                self.transform = transforms.Compose([
                    transforms.ToPILImage(),
                    transforms.ToTensor(),
                    normalize
                ])
        else:
            self.transform = transform

    def __len__(self):
        return len(self.video_paths)

    def load_video_frames(self, video_path):
        """
        Load and sample frames from video.
        Returns: (num_frames, C, H, W) tensor
        """
        cap = cv2.VideoCapture(str(video_path))

        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        if total_frames == 0:
            cap.release()
            # Return black frames if video is corrupted
            return torch.zeros(self.num_frames, 3, *self.frame_size)

        # Uniformly sample frame indices
        if total_frames < self.num_frames:
            # Repeat frames if video is too short
            indices = np.linspace(0, total_frames - 1, self.num_frames, dtype=int)
        else:
            indices = np.linspace(0, total_frames - 1, self.num_frames, dtype=int)

        frames = []
        for idx in indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ret, frame = cap.read()

            if not ret:
                # Use last valid frame if read fails
                if len(frames) > 0:
                    frames.append(frames[-1])
                else:
                    frames.append(np.zeros((*self.frame_size, 3), dtype=np.uint8))
                continue

            # Resize frame
            frame = cv2.resize(frame, self.frame_size)
            # Convert BGR to RGB
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            # Apply transform
            frame_tensor = self.transform(frame)
            frames.append(frame_tensor)

        cap.release()

        # Stack to (T, C, H, W)
        frames_tensor = torch.stack(frames)

        return frames_tensor

    def __getitem__(self, idx):
        video_path = self.video_paths[idx]
        label = self.labels[idx]

        # Load video frames: (T, C, H, W)
        frames = self.load_video_frames(video_path)

        return frames, label


# ==================== MODEL ====================

class CNN_LSTM_Classifier(nn.Module):
    """
    2D CNN + LSTM for video classification.

    Architecture:
    1. ResNet18 (pretrained) extracts features from each frame independently
    2. LSTM processes temporal sequence of features
    3. Final classifier predicts shot type
    """

    def __init__(self, num_classes=5, lstm_hidden_size=256, lstm_num_layers=2,
                 lstm_dropout=0.5, freeze_cnn=True):
        super().__init__()

        # CNN backbone (ResNet18 pretrained on ImageNet)
        resnet = models.resnet18(pretrained=True)

        # Remove final FC layer
        self.cnn = nn.Sequential(*list(resnet.children())[:-1])

        # Freeze CNN weights initially (fine-tune later)
        if freeze_cnn:
            for param in self.cnn.parameters():
                param.requires_grad = False

        # ResNet18 feature dimension
        self.cnn_feature_dim = 512

        # LSTM for temporal modeling
        self.lstm = nn.LSTM(
            input_size=self.cnn_feature_dim,
            hidden_size=lstm_hidden_size,
            num_layers=lstm_num_layers,
            batch_first=True,
            dropout=lstm_dropout if lstm_num_layers > 1 else 0,
            bidirectional=True
        )

        # Final classifier
        lstm_output_dim = lstm_hidden_size * 2  # Bidirectional
        self.classifier = nn.Sequential(
            nn.Dropout(lstm_dropout),
            nn.Linear(lstm_output_dim, 128),
            nn.ReLU(),
            nn.Dropout(lstm_dropout),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        """
        Args:
            x: (batch_size, num_frames, C, H, W)
        Returns:
            logits: (batch_size, num_classes)
        """
        batch_size, num_frames, C, H, W = x.shape

        # Reshape to (batch_size * num_frames, C, H, W)
        x = x.view(batch_size * num_frames, C, H, W)

        # Extract CNN features for each frame
        # (batch_size * num_frames, 512, 1, 1)
        cnn_features = self.cnn(x)

        # Reshape to (batch_size * num_frames, 512)
        cnn_features = cnn_features.view(batch_size * num_frames, -1)

        # Reshape to (batch_size, num_frames, 512)
        cnn_features = cnn_features.view(batch_size, num_frames, -1)

        # LSTM temporal modeling
        # lstm_out: (batch_size, num_frames, lstm_hidden_size * 2)
        lstm_out, (h_n, c_n) = self.lstm(cnn_features)

        # Use last time step output
        # (batch_size, lstm_hidden_size * 2)
        last_output = lstm_out[:, -1, :]

        # Classification
        logits = self.classifier(last_output)

        return logits

    def unfreeze_cnn(self):
        """Unfreeze CNN for fine-tuning."""
        for param in self.cnn.parameters():
            param.requires_grad = True


# ==================== LOSS FUNCTION ====================

class FocalLoss(nn.Module):
    """
    Focal Loss for handling class imbalance.
    FL(pt) = -alpha * (1 - pt)^gamma * log(pt)
    """

    def __init__(self, alpha=None, gamma=2.0, reduction='mean'):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):
        """
        Args:
            inputs: (batch_size, num_classes) logits
            targets: (batch_size,) class indices
        """
        ce_loss = nn.functional.cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = (1 - pt) ** self.gamma * ce_loss

        if self.alpha is not None:
            alpha_t = self.alpha[targets]
            focal_loss = alpha_t * focal_loss

        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss


# ==================== DATA LOADING ====================

def load_dataset(data_root, class_names):
    """
    Load video paths and labels from directory structure.

    Args:
        data_root: Path to data/clips
        class_names: List of class names (ordered)

    Returns:
        video_paths: List of Path objects
        labels: List of integers
        class_counts: Dict of class counts
    """
    data_root = Path(data_root)

    video_paths = []
    labels = []
    class_counts = defaultdict(int)

    for class_idx, class_name in enumerate(class_names):
        class_dir = data_root / class_name

        if not class_dir.exists():
            print(f"Warning: {class_dir} does not exist!")
            continue

        # Get all mp4 files
        mp4_files = list(class_dir.glob("*.mp4"))

        video_paths.extend(mp4_files)
        labels.extend([class_idx] * len(mp4_files))
        class_counts[class_name] = len(mp4_files)

    return video_paths, labels, class_counts


def compute_class_weights(labels, num_classes, method='sqrt'):
    """
    Compute class weights for focal loss.

    Args:
        labels: List of integer labels
        num_classes: Number of classes
        method: 'sqrt' (softened) or 'inverse' (standard)

    Returns:
        weights: Tensor of shape (num_classes,)
    """
    counts = np.bincount(labels, minlength=num_classes)
    total = len(labels)

    if method == 'sqrt':
        # Softened weights (less extreme)
        weights = np.sqrt(total / counts)
    else:
        # Standard inverse frequency
        weights = total / counts

    # Normalize to sum to num_classes
    weights = weights * num_classes / weights.sum()

    return torch.FloatTensor(weights)


# ==================== TRAINING ====================

def train_epoch(model, dataloader, criterion, optimizer, device):
    """Train for one epoch."""
    model.train()

    total_loss = 0
    correct = 0
    total = 0

    pbar = tqdm(dataloader, desc='Training')

    for frames, labels in pbar:
        frames = frames.to(device)  # (B, T, C, H, W)
        labels = labels.to(device)

        # Forward pass
        optimizer.zero_grad()
        logits = model(frames)
        loss = criterion(logits, labels)

        # Backward pass
        loss.backward()

        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

        optimizer.step()

        # Statistics
        total_loss += loss.item()
        _, predicted = logits.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()

        pbar.set_postfix({
            'loss': f'{loss.item():.4f}',
            'acc': f'{100. * correct / total:.2f}%'
        })

    epoch_loss = total_loss / len(dataloader)
    epoch_acc = 100. * correct / total

    return epoch_loss, epoch_acc


def validate_epoch(model, dataloader, criterion, device):
    """Validate for one epoch."""
    model.eval()

    total_loss = 0
    correct = 0
    total = 0

    all_preds = []
    all_labels = []

    with torch.no_grad():
        pbar = tqdm(dataloader, desc='Validation')

        for frames, labels in pbar:
            frames = frames.to(device)
            labels = labels.to(device)

            logits = model(frames)
            loss = criterion(logits, labels)

            total_loss += loss.item()
            _, predicted = logits.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

            pbar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'acc': f'{100. * correct / total:.2f}%'
            })

    epoch_loss = total_loss / len(dataloader)
    epoch_acc = 100. * correct / total
    f1 = f1_score(all_labels, all_preds, average='macro')

    return epoch_loss, epoch_acc, f1, all_preds, all_labels


# ==================== MAIN ====================

def main():
    """Main training function."""

    # Create output directory
    os.makedirs(CONFIG['output_dir'], exist_ok=True)

    # Save config
    with open(os.path.join(CONFIG['output_dir'], 'config.json'), 'w') as f:
        json.dump(CONFIG, f, indent=2)

    print("="*50)
    print("Badminton Video Classification")
    print("2D CNN + LSTM Architecture")
    print("="*50)

    # Load dataset
    print("\n1. Loading dataset...")
    video_paths, labels, class_counts = load_dataset(
        CONFIG['data_root'],
        CONFIG['class_names']
    )

    print(f"\nTotal videos: {len(video_paths)}")
    print("\nClass distribution:")
    for class_name in CONFIG['class_names']:
        count = class_counts[class_name]
        print(f"  {class_name:8s}: {count:5d} ({100*count/len(labels):.1f}%)")

    # Train/val/test split (70/15/15)
    print("\n2. Splitting dataset...")
    train_paths, temp_paths, train_labels, temp_labels = train_test_split(
        video_paths, labels, test_size=0.3, random_state=42, stratify=labels
    )

    val_paths, test_paths, val_labels, test_labels = train_test_split(
        temp_paths, temp_labels, test_size=0.5, random_state=42, stratify=temp_labels
    )

    print(f"  Train: {len(train_paths)} videos")
    print(f"  Val:   {len(val_paths)} videos")
    print(f"  Test:  {len(test_paths)} videos")

    # Compute class weights
    class_weights = compute_class_weights(train_labels, CONFIG['num_classes'], method='sqrt')
    print(f"\nClass weights (softened):")
    for i, name in enumerate(CONFIG['class_names']):
        print(f"  {name:8s}: {class_weights[i]:.3f}")

    # Create datasets
    print("\n3. Creating data loaders...")
    train_dataset = BadmintonVideoDataset(
        train_paths, train_labels,
        num_frames=CONFIG['num_frames'],
        frame_size=CONFIG['frame_size'],
        augment=True
    )

    val_dataset = BadmintonVideoDataset(
        val_paths, val_labels,
        num_frames=CONFIG['num_frames'],
        frame_size=CONFIG['frame_size'],
        augment=False
    )

    test_dataset = BadmintonVideoDataset(
        test_paths, test_labels,
        num_frames=CONFIG['num_frames'],
        frame_size=CONFIG['frame_size'],
        augment=False
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=CONFIG['batch_size'],
        shuffle=True,
        num_workers=CONFIG['num_workers'],
        pin_memory=True if CONFIG['device'] == 'cuda' else False
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=CONFIG['batch_size'],
        shuffle=False,
        num_workers=CONFIG['num_workers'],
        pin_memory=True if CONFIG['device'] == 'cuda' else False
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=CONFIG['batch_size'],
        shuffle=False,
        num_workers=CONFIG['num_workers'],
        pin_memory=True if CONFIG['device'] == 'cuda' else False
    )

    # Create model
    print("\n4. Creating model...")
    model = CNN_LSTM_Classifier(
        num_classes=CONFIG['num_classes'],
        lstm_hidden_size=CONFIG['lstm_hidden_size'],
        lstm_num_layers=CONFIG['lstm_num_layers'],
        lstm_dropout=CONFIG['lstm_dropout'],
        freeze_cnn=CONFIG['freeze_cnn']
    )

    model = model.to(CONFIG['device'])

    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Total parameters: {total_params:,}")
    print(f"  Trainable parameters: {trainable_params:,}")

    # Loss function
    if CONFIG['use_focal_loss']:
        criterion = FocalLoss(
            alpha=class_weights.to(CONFIG['device']),
            gamma=CONFIG['focal_gamma']
        )
        print(f"  Using Focal Loss (gamma={CONFIG['focal_gamma']})")
    else:
        criterion = nn.CrossEntropyLoss(weight=class_weights.to(CONFIG['device']))
        print("  Using CrossEntropy Loss")

    # Optimizer
    optimizer = optim.Adam(
        model.parameters(),
        lr=CONFIG['learning_rate'],
        weight_decay=CONFIG['weight_decay']
    )

    # Scheduler
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=CONFIG['num_epochs']
    )

    # Training loop
    print("\n5. Training...")
    print("="*50)

    best_val_f1 = 0
    patience_counter = 0
    history = defaultdict(list)

    for epoch in range(CONFIG['num_epochs']):
        print(f"\nEpoch {epoch+1}/{CONFIG['num_epochs']}")

        # Train
        train_loss, train_acc = train_epoch(
            model, train_loader, criterion, optimizer, CONFIG['device']
        )

        # Validate
        val_loss, val_acc, val_f1, val_preds, val_labels_list = validate_epoch(
            model, val_loader, criterion, CONFIG['device']
        )

        # Scheduler step
        scheduler.step()

        # Log
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)
        history['val_f1'].append(val_f1)

        print(f"\nTrain Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}%")
        print(f"Val Loss:   {val_loss:.4f} | Val Acc:   {val_acc:.2f}% | Val F1: {val_f1:.4f}")
        print(f"Gap:        {train_acc - val_acc:.2f}%")

        # Save best model
        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            patience_counter = 0

            if CONFIG['save_best_model']:
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'val_f1': val_f1,
                    'val_acc': val_acc,
                }, os.path.join(CONFIG['output_dir'], 'best_model.pth'))
                print(f"✓ Saved best model (F1: {val_f1:.4f})")
        else:
            patience_counter += 1

        # Early stopping
        if patience_counter >= CONFIG['early_stopping_patience']:
            print(f"\nEarly stopping triggered (patience={CONFIG['early_stopping_patience']})")
            break

    # Save training history
    pd.DataFrame(history).to_csv(
        os.path.join(CONFIG['output_dir'], 'training_history.csv'),
        index=False
    )

    # Load best model for testing
    print("\n6. Evaluating on test set...")
    checkpoint = torch.load(os.path.join(CONFIG['output_dir'], 'best_model.pth'))
    model.load_state_dict(checkpoint['model_state_dict'])

    test_loss, test_acc, test_f1, test_preds, test_labels_list = validate_epoch(
        model, test_loader, criterion, CONFIG['device']
    )

    print(f"\nTest Loss: {test_loss:.4f}")
    print(f"Test Acc:  {test_acc:.2f}%")
    print(f"Test F1:   {test_f1:.4f}")

    # Classification report
    print("\n" + "="*50)
    print("Classification Report:")
    print("="*50)
    print(classification_report(
        test_labels_list, test_preds,
        target_names=CONFIG['class_names'],
        digits=3
    ))

    # Confusion matrix
    cm = confusion_matrix(test_labels_list, test_preds)
    print("\nConfusion Matrix:")
    print("="*50)
    print(f"{'':8s}", end='')
    for name in CONFIG['class_names']:
        print(f"{name:8s}", end='')
    print()
    for i, name in enumerate(CONFIG['class_names']):
        print(f"{name:8s}", end='')
        for j in range(len(CONFIG['class_names'])):
            print(f"{cm[i,j]:8d}", end='')
        print()

    print("\n" + "="*50)
    print("Training complete!")
    print(f"Best model saved to: {CONFIG['output_dir']}/best_model.pth")
    print("="*50)


if __name__ == '__main__':
    main()
