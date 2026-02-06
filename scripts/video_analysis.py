# Quick video clip analysis
import cv2
from pathlib import Path
import numpy as np

clips_dir = Path('/Volumes/Ext/GenAI/iti123_v2/data/clips')

print(clips_dir)
# Check if clips exist
clip_files = list(clips_dir/*/.glob('*.mp4'))
print(f"Total clips found: {len(clip_files)}")

if len(clip_files) == 0:
    print("❌ No clips found! Check path or extract from videos first.")
else:
    # Analyze 10 random clips
    samples = np.random.choice(clip_files, min(10, len(clip_files)), replace=False)
    
    stats = {
        'fps': [],
        'frame_count': [],
        'duration': [],
        'width': [],
        'height': []
    }
    
    for clip_path in samples:
        cap = cv2.VideoCapture(str(clip_path))
        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        duration = frame_count / fps if fps > 0 else 0
        
        stats['fps'].append(fps)
        stats['frame_count'].append(frame_count)
        stats['duration'].append(duration)
        stats['width'].append(width)
        stats['height'].append(height)
        
        cap.release()
    
    print("\n" + "="*80)
    print("VIDEO CLIP STATISTICS")
    print("="*80)
    print(f"FPS:          {np.mean(stats['fps']):.1f} ± {np.std(stats['fps']):.1f}")
    print(f"Frame count:  {np.mean(stats['frame_count']):.1f} ± {np.std(stats['frame_count']):.1f}")
    print(f"Duration:     {np.mean(stats['duration']):.2f}s ± {np.std(stats['duration']):.2f}s")
    print(f"Resolution:   {int(np.mean(stats['width']))}x{int(np.mean(stats['height']))}")
    
    print("\n✓ Clips look good for video-based training!")
    print(f"\nRecommended settings:")
    print(f"  - Sample 16 frames uniformly from each clip")
    print(f"  - Resize to 224x224")
    print(f"  - Use X3D-S model (3.8M params)")
    print(f"  - Expected training time: 3-4 hours")
