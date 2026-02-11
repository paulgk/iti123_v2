"""
Shot Type Classifier Module
Uses trained ResNet18+BiLSTM model to classify shot type
"""

import torch
import torch.nn as nn
import torchvision.models as models
from torchvision import transforms
import cv2
import numpy as np
from pathlib import Path
from PIL import Image


class CNN_LSTM_Classifier(nn.Module):
    """
    ResNet18 (2D CNN) + Bidirectional LSTM
    Same architecture as trained model
    """
    def __init__(self, num_classes=5, hidden_size=256, num_lstm_layers=2, dropout=0.5):
        super(CNN_LSTM_Classifier, self).__init__()

        # ResNet18 backbone (pretrained)
        resnet = models.resnet18(pretrained=False)  # Will load our trained weights
        self.cnn = nn.Sequential(*list(resnet.children())[:-1])
        self.cnn_feature_size = 512

        # Bidirectional LSTM
        self.lstm = nn.LSTM(
            input_size=self.cnn_feature_size,
            hidden_size=hidden_size,
            num_layers=num_lstm_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_lstm_layers > 1 else 0
        )

        # Classifier
        self.fc = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(hidden_size * 2, num_classes)
        )

    def forward(self, x):
        batch_size, num_frames, c, h, w = x.size()

        # Process frames through CNN
        x = x.view(batch_size * num_frames, c, h, w)
        x = self.cnn(x)
        x = x.view(batch_size * num_frames, -1)
        x = x.view(batch_size, num_frames, -1)

        # LSTM
        x, _ = self.lstm(x)
        x = x[:, -1, :]

        # Classification
        x = self.fc(x)
        return x


class ShotClassifier:
    """Classify badminton shot type from video"""

    def __init__(self, model_path, device=None):
        """
        Args:
            model_path: Path to trained model (.pth file)
            device: torch device (cuda/cpu), auto-detect if None
        """
        self.model_path = Path(model_path)
        if not self.model_path.exists():
            raise FileNotFoundError(f"Model not found: {model_path}")

        # Device
        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = device

        # Class names
        self.class_names = ['Clear', 'Drive', 'Drop', 'Lift', 'Smash']

        # Load model
        self.model = self._load_model()

        # Transforms (same as training)
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])

    def _load_model(self):
        """Load trained model from checkpoint"""
        # Create model
        model = CNN_LSTM_Classifier(num_classes=5)

        # Load checkpoint
        checkpoint = torch.load(self.model_path, map_location=self.device)

        # Load state dict
        if 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
        else:
            model.load_state_dict(checkpoint)

        model = model.to(self.device)
        model.eval()

        return model

    def get_video_metadata(self, video_path):
        """
        Extract video metadata

        Args:
            video_path: Path to video file

        Returns:
            dict with video metadata or None if failed
        """
        cap = cv2.VideoCapture(str(video_path))

        if not cap.isOpened():
            return None

        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        duration = total_frames / fps if fps > 0 else 0

        cap.release()

        return {
            'total_frames': total_frames,
            'fps': fps,
            'duration': duration
        }

    def extract_frames(self, video_path, num_frames=16, frame_size=(224, 224)):
        """
        Extract frames from video (same as training preprocessing)

        Args:
            video_path: Path to video file
            num_frames: Number of frames to extract
            frame_size: Size to resize frames

        Returns:
            tuple: (frames_array, metadata_dict) or (None, None) if failed
            - frames_array: np.array of shape (num_frames, H, W, 3)
            - metadata_dict: dict with frame indices and timing info
        """
        cap = cv2.VideoCapture(str(video_path))

        if not cap.isOpened():
            return None, None

        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS)

        if total_frames == 0:
            cap.release()
            return None, None

        # Sample frames uniformly
        frame_indices = np.linspace(0, total_frames - 1, num_frames, dtype=int)

        # Calculate frame times
        frame_times = frame_indices / fps if fps > 0 else frame_indices

        frames = []
        for idx in frame_indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ret, frame = cap.read()

            if not ret or frame is None:
                cap.release()
                return None, None

            # Resize
            frame = cv2.resize(frame, frame_size)

            # BGR to RGB
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            frames.append(frame)

        cap.release()

        # Create metadata
        metadata = {
            'total_frames': total_frames,
            'fps': fps,
            'duration': total_frames / fps if fps > 0 else 0,
            'sampled_frames': frame_indices.tolist(),
            'sampled_times': frame_times.tolist(),
            'num_frames_analyzed': num_frames
        }

        return np.array(frames, dtype=np.uint8), metadata

    def predict(self, video_path, num_frames=16):
        """
        Predict shot type from video

        Args:
            video_path: Path to video file
            num_frames: Number of frames to extract

        Returns:
            dict with:
                - predicted_class: str (e.g., 'Smash')
                - confidence: float (0-1)
                - probabilities: dict of all class probabilities
                - metadata: dict with video and frame timing info
                - success: bool
        """
        # Extract frames with metadata
        frames, metadata = self.extract_frames(video_path, num_frames=num_frames)

        if frames is None or metadata is None:
            return {
                'success': False,
                'error': 'Failed to extract frames from video'
            }

        # Convert to tensor
        frames_tensor = []
        for frame in frames:
            frame_pil = Image.fromarray(frame)
            frame_tensor = self.transform(frame_pil)
            frames_tensor.append(frame_tensor)

        # Stack: (T, C, H, W)
        frames_tensor = torch.stack(frames_tensor)

        # Add batch dimension: (1, T, C, H, W)
        frames_tensor = frames_tensor.unsqueeze(0)
        frames_tensor = frames_tensor.to(self.device)

        # Predict
        with torch.no_grad():
            outputs = self.model(frames_tensor)
            probabilities = torch.softmax(outputs, dim=1)
            confidence, predicted = torch.max(probabilities, 1)

        predicted_idx = predicted.item()
        predicted_class = self.class_names[predicted_idx]
        confidence_val = confidence.item()

        # Get all probabilities
        probs_dict = {
            self.class_names[i]: probabilities[0][i].item()
            for i in range(len(self.class_names))
        }

        return {
            'success': True,
            'predicted_class': predicted_class,
            'confidence': confidence_val,
            'probabilities': probs_dict,
            'metadata': metadata
        }
