"""
Inference script for badminton shot classification.

Usage:
    python predict_video.py --video path/to/video.mp4
    python predict_video.py --video path/to/video.mp4 --model models/video_classification/best_model.pth
"""

import argparse
import cv2
import numpy as np
import torch
import torchvision.transforms as transforms
from pathlib import Path
import sys

# Import model from training script
from badminton_video_classification import CNN_LSTM_Classifier


def load_model(model_path, device):
    """Load trained model."""
    checkpoint = torch.load(model_path, map_location=device)

    model = CNN_LSTM_Classifier(
        num_classes=5,
        lstm_hidden_size=256,
        lstm_num_layers=2,
        lstm_dropout=0.5,
        freeze_cnn=True
    )

    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()

    return model


def load_video_frames(video_path, num_frames=16, frame_size=(224, 224)):
    """
    Load and preprocess video frames.

    Args:
        video_path: Path to video file
        num_frames: Number of frames to sample
        frame_size: Target frame size (H, W)

    Returns:
        frames_tensor: (1, num_frames, C, H, W) tensor
    """
    cap = cv2.VideoCapture(str(video_path))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    if total_frames == 0:
        print(f"Error: Could not read video {video_path}")
        return None

    # Uniformly sample frame indices
    indices = np.linspace(0, total_frames - 1, num_frames, dtype=int)

    # Normalization transform (ImageNet stats)
    normalize = transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )

    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.ToTensor(),
        normalize
    ])

    frames = []
    for idx in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()

        if not ret:
            if len(frames) > 0:
                frames.append(frames[-1])
            else:
                frames.append(torch.zeros(3, *frame_size))
            continue

        # Resize and convert to RGB
        frame = cv2.resize(frame, frame_size)
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Apply transform
        frame_tensor = transform(frame)
        frames.append(frame_tensor)

    cap.release()

    # Stack to (num_frames, C, H, W)
    frames_tensor = torch.stack(frames)

    # Add batch dimension: (1, num_frames, C, H, W)
    frames_tensor = frames_tensor.unsqueeze(0)

    return frames_tensor


def predict(model, video_path, device, class_names):
    """
    Predict shot type for a video.

    Args:
        model: Trained model
        video_path: Path to video file
        device: torch device
        class_names: List of class names

    Returns:
        predicted_class: Predicted class name
        probabilities: Softmax probabilities for all classes
    """
    # Load video
    frames = load_video_frames(video_path)

    if frames is None:
        return None, None

    frames = frames.to(device)

    # Predict
    with torch.no_grad():
        logits = model(frames)
        probabilities = torch.softmax(logits, dim=1)[0]
        predicted_idx = probabilities.argmax().item()

    predicted_class = class_names[predicted_idx]

    return predicted_class, probabilities.cpu().numpy()


def main():
    parser = argparse.ArgumentParser(description='Predict badminton shot type from video')
    parser.add_argument('--video', type=str, required=True, help='Path to video file')
    parser.add_argument('--model', type=str, default='models/video_classification/best_model.pth',
                        help='Path to trained model')
    parser.add_argument('--device', type=str, default='auto',
                        help='Device to use (cuda/mps/cpu/auto)')

    args = parser.parse_args()

    # Determine device
    if args.device == 'auto':
        if torch.cuda.is_available():
            device = 'cuda'
        elif torch.backends.mps.is_available():
            device = 'mps'
        else:
            device = 'cpu'
    else:
        device = args.device

    device = torch.device(device)
    print(f"Using device: {device}")

    # Check if video exists
    video_path = Path(args.video)
    if not video_path.exists():
        print(f"Error: Video not found: {video_path}")
        sys.exit(1)

    # Check if model exists
    model_path = Path(args.model)
    if not model_path.exists():
        print(f"Error: Model not found: {model_path}")
        sys.exit(1)

    # Load model
    print(f"Loading model from {model_path}...")
    model = load_model(model_path, device)

    # Class names
    class_names = ['Clear', 'Drive', 'Drop', 'Lift', 'Smash']

    # Predict
    print(f"Predicting shot type for {video_path}...")
    predicted_class, probabilities = predict(model, video_path, device, class_names)

    if predicted_class is None:
        print("Prediction failed!")
        sys.exit(1)

    # Display results
    print("\n" + "="*50)
    print(f"Predicted Shot Type: {predicted_class}")
    print("="*50)
    print("\nProbabilities:")
    for i, name in enumerate(class_names):
        bar = "█" * int(probabilities[i] * 50)
        print(f"  {name:8s}: {probabilities[i]*100:5.2f}% {bar}")
    print("="*50)


if __name__ == '__main__':
    main()
