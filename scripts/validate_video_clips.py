"""
Validate video clips before training.

Checks:
1. All videos are readable
2. Frame counts are reasonable (>0)
3. File sizes are reasonable (>1 KB)
4. No corrupted videos
5. Class distribution

Usage:
    python scripts/validate_video_clips.py
"""

import cv2
import numpy as np
from pathlib import Path
from tqdm import tqdm
from collections import defaultdict
import pandas as pd


def validate_single_video(video_path):
    """
    Validate a single video file.

    Returns:
        dict with validation results, or None if valid
    """
    # Check file exists
    if not video_path.exists():
        return {'error': 'File not found', 'path': str(video_path)}

    # Check file size
    file_size = video_path.stat().st_size
    if file_size < 1000:  # Less than 1 KB
        return {'error': 'File too small (<1 KB)', 'path': str(video_path), 'size': file_size}

    # Try to open video
    cap = cv2.VideoCapture(str(video_path))

    if not cap.isOpened():
        return {'error': 'Cannot open video', 'path': str(video_path)}

    # Get video properties
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # Check frame count
    if frame_count == 0:
        cap.release()
        return {'error': 'Zero frames', 'path': str(video_path)}

    # Try to read first frame
    ret, frame = cap.read()
    if not ret or frame is None:
        cap.release()
        return {'error': 'Cannot read first frame', 'path': str(video_path)}

    cap.release()

    # Valid video - return properties
    return {
        'valid': True,
        'frames': frame_count,
        'fps': fps,
        'width': width,
        'height': height,
        'size_kb': file_size / 1024,
        'duration_sec': frame_count / fps if fps > 0 else 0
    }


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--sample', type=int, default=100,
                       help='Number of clips to sample per class (default: 100, use -1 for all)')
    args = parser.parse_args()

    data_root = Path('data/clips')

    if not data_root.exists():
        print(f"Error: {data_root} not found!")
        return

    class_names = ['Clear', 'Drive', 'Drop', 'Lift', 'Smash']

    print("="*70)
    print("Video Clips Validation")
    print("="*70)
    print(f"Data root: {data_root.absolute()}")
    if args.sample > 0:
        print(f"Sampling: {args.sample} clips per class (for quick validation)")
    else:
        print(f"Validating ALL clips")
    print()

    # Collect all videos
    all_videos = []
    class_counts = defaultdict(int)
    class_total_counts = defaultdict(int)

    for class_name in class_names:
        class_dir = data_root / class_name
        if class_dir.exists():
            videos = list(class_dir.glob("*.mp4"))
            class_total_counts[class_name] = len(videos)

            # Sample if requested
            if args.sample > 0 and len(videos) > args.sample:
                import random
                random.seed(42)
                videos = random.sample(videos, args.sample)

            all_videos.extend([(v, class_name) for v in videos])
            class_counts[class_name] = len(videos)
        else:
            print(f"Warning: {class_dir} not found!")

    total_videos = len(all_videos)
    total_all_videos = sum(class_total_counts.values())

    if args.sample > 0:
        print(f"Total videos in dataset: {total_all_videos}")
        print(f"Validating sample: {total_videos} clips")
    else:
        print(f"Total videos found: {total_videos}")
    print()

    # Class distribution
    print("Class distribution (sample):" if args.sample > 0 else "Class distribution:")
    print("-"*70)
    for class_name in class_names:
        count = class_counts[class_name]
        total_count = class_total_counts[class_name]
        pct = 100 * count / total_videos if total_videos > 0 else 0
        bar = "█" * int(pct / 2)
        if args.sample > 0:
            print(f"  {class_name:8s}: {count:5d} / {total_count:5d} ({pct:5.2f}%) {bar}")
        else:
            print(f"  {class_name:8s}: {count:5d} ({pct:5.2f}%) {bar}")
    print()

    # Validate each video
    print("Validating videos...")
    print("-"*70)

    valid_videos = []
    corrupted_videos = []
    properties = defaultdict(list)

    for video_path, class_name in tqdm(all_videos, desc='Validating'):
        result = validate_single_video(video_path)

        if result.get('valid'):
            valid_videos.append((video_path, class_name))
            properties['class'].append(class_name)
            properties['frames'].append(result['frames'])
            properties['fps'].append(result['fps'])
            properties['width'].append(result['width'])
            properties['height'].append(result['height'])
            properties['size_kb'].append(result['size_kb'])
            properties['duration_sec'].append(result['duration_sec'])
        else:
            corrupted_videos.append(result)

    print()
    print("="*70)
    print("Validation Results")
    print("="*70)
    print(f"Valid videos:     {len(valid_videos):5d} ({100*len(valid_videos)/total_videos:.2f}%)")
    print(f"Corrupted videos: {len(corrupted_videos):5d} ({100*len(corrupted_videos)/total_videos:.2f}%)")
    print()

    # Show corrupted videos
    if corrupted_videos:
        print("Corrupted videos:")
        print("-"*70)
        for issue in corrupted_videos[:20]:  # Show first 20
            print(f"  {issue['error']:30s}: {issue['path']}")
        if len(corrupted_videos) > 20:
            print(f"  ... and {len(corrupted_videos) - 20} more")
        print()

    # Statistics on valid videos
    if valid_videos:
        df = pd.DataFrame(properties)

        print("Video Properties (Valid Videos Only)")
        print("-"*70)

        print("\nFrame counts:")
        print(f"  Min:    {df['frames'].min():6d} frames")
        print(f"  Max:    {df['frames'].max():6d} frames")
        print(f"  Mean:   {df['frames'].mean():6.1f} frames")
        print(f"  Median: {df['frames'].median():6.0f} frames")

        print("\nDuration:")
        print(f"  Min:    {df['duration_sec'].min():6.2f} seconds")
        print(f"  Max:    {df['duration_sec'].max():6.2f} seconds")
        print(f"  Mean:   {df['duration_sec'].mean():6.2f} seconds")
        print(f"  Median: {df['duration_sec'].median():6.2f} seconds")

        print("\nFPS:")
        print(f"  Min:    {df['fps'].min():6.1f} fps")
        print(f"  Max:    {df['fps'].max():6.1f} fps")
        print(f"  Mean:   {df['fps'].mean():6.1f} fps")
        print(f"  Median: {df['fps'].median():6.1f} fps")

        print("\nResolution:")
        resolutions = df.groupby(['width', 'height']).size().reset_index(name='count')
        resolutions = resolutions.sort_values('count', ascending=False)
        for _, row in resolutions.head(5).iterrows():
            count = row['count']
            pct = 100 * count / len(df)
            print(f"  {int(row['width'])}x{int(row['height'])}: {count:5d} ({pct:5.2f}%)")

        print("\nFile size:")
        print(f"  Min:    {df['size_kb'].min():8.2f} KB")
        print(f"  Max:    {df['size_kb'].max():8.2f} KB")
        print(f"  Mean:   {df['size_kb'].mean():8.2f} KB")
        print(f"  Median: {df['size_kb'].median():8.2f} KB")
        print(f"  Total:  {df['size_kb'].sum()/1024:.2f} MB")

        print()

        # Per-class statistics
        print("Per-class statistics:")
        print("-"*70)
        print(f"{'Class':8s}  {'Count':>6s}  {'Avg Frames':>11s}  {'Avg Duration':>12s}")
        print("-"*70)
        for class_name in class_names:
            class_df = df[df['class'] == class_name]
            if len(class_df) > 0:
                count = len(class_df)
                avg_frames = class_df['frames'].mean()
                avg_duration = class_df['duration_sec'].mean()
                print(f"{class_name:8s}  {count:6d}  {avg_frames:11.1f}  {avg_duration:12.2f}s")

    print()
    print("="*70)

    # Recommendations
    print("Recommendations:")
    print("-"*70)

    if len(corrupted_videos) > 0:
        print(f"⚠️  {len(corrupted_videos)} corrupted videos found - recommend removing them")
        print(f"   Run: python scripts/remove_corrupted_videos.py")
    else:
        print("✓ All videos are valid!")

    if len(valid_videos) > 0:
        df = pd.DataFrame(properties)

        # Check frame counts
        min_frames = df['frames'].min()
        max_frames = df['frames'].max()

        if min_frames < 10:
            print(f"⚠️  Some videos have very few frames (min={min_frames})")
            print(f"   This may cause issues with frame sampling")

        # Check resolution variance
        unique_resolutions = len(resolutions)
        if unique_resolutions > 5:
            print(f"⚠️  Multiple resolutions found ({unique_resolutions} different)")
            print(f"   Training script will resize all to 224x224")

        # Check FPS variance
        fps_std = df['fps'].std()
        if fps_std > 5:
            print(f"⚠️  High FPS variance (std={fps_std:.1f})")
            print(f"   Frame sampling will handle this automatically")

        # Check class imbalance
        drive_count = class_counts['Drive']
        drop_count = class_counts['Drop']
        imbalance_ratio = drop_count / drive_count if drive_count > 0 else 0

        if imbalance_ratio > 5:
            print(f"⚠️  Severe class imbalance (Drop/Drive = {imbalance_ratio:.1f}:1)")
            print(f"   Training uses Focal Loss to handle this")

        # Dataset size recommendation
        if total_videos < 5000:
            print(f"⚠️  Small dataset ({total_videos} videos)")
            print(f"   Recommend data augmentation and small models")
        elif total_videos > 15000:
            print(f"✓ Large dataset ({total_videos} videos) - good for training!")

    print()
    print("="*70)
    print("Validation complete!")
    print("="*70)


if __name__ == '__main__':
    main()
