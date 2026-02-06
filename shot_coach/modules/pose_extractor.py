"""
Pose Extraction Module for Shot Coach
Extracts pose keypoints from video clips on-the-fly
"""

import cv2
import numpy as np
import mediapipe as mp
from pathlib import Path


class PoseExtractor:
    """Extract pose keypoints from video using MediaPipe"""

    def __init__(self):
        self.mp_pose = mp.solutions.pose
        self.pose = self.mp_pose.Pose(
            static_image_mode=False,
            model_complexity=1,
            enable_segmentation=False,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )

    def extract_from_video(self, video_path, max_frames=None):
        """
        Extract pose keypoints from video.

        Args:
            video_path: Path to video file
            max_frames: Maximum number of frames to process (None = all)

        Returns:
            dict with:
                - keypoints: list of pose keypoints per frame
                - frame_count: total frames processed
                - fps: video fps
                - success: whether extraction succeeded
        """
        video_path = Path(video_path)
        if not video_path.exists():
            return {
                'success': False,
                'error': f'Video file not found: {video_path}'
            }

        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            return {
                'success': False,
                'error': f'Could not open video: {video_path}'
            }

        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        if max_frames:
            total_frames = min(total_frames, max_frames)

        keypoints_list = []
        frame_idx = 0

        while cap.isOpened() and (max_frames is None or frame_idx < max_frames):
            ret, frame = cap.read()
            if not ret:
                break

            # Convert BGR to RGB
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            # Process with MediaPipe
            results = self.pose.process(frame_rgb)

            if results.pose_landmarks:
                # Extract keypoints
                landmarks = results.pose_landmarks.landmark
                keypoints = {}

                # Store all 33 landmarks
                for idx, landmark in enumerate(landmarks):
                    keypoints[idx] = {
                        'x': landmark.x,
                        'y': landmark.y,
                        'z': landmark.z,
                        'visibility': landmark.visibility
                    }

                keypoints_list.append(keypoints)
            else:
                # No pose detected in this frame
                keypoints_list.append(None)

            frame_idx += 1

        cap.release()

        # Check if we got any valid poses
        valid_frames = [kp for kp in keypoints_list if kp is not None]

        if len(valid_frames) == 0:
            return {
                'success': False,
                'error': 'No pose detected in any frame'
            }

        return {
            'success': True,
            'keypoints': keypoints_list,
            'frame_count': len(keypoints_list),
            'valid_frames': len(valid_frames),
            'fps': fps,
            'total_frames': total_frames
        }

    def find_contact_frame(self, keypoints_list):
        """
        Find the frame where shot contact likely occurs.
        Uses wrist velocity as proxy for contact point.

        Args:
            keypoints_list: List of keypoint dicts per frame

        Returns:
            int: Frame index of likely contact point
        """
        if not keypoints_list:
            return 0

        # Calculate right wrist velocity across frames
        wrist_velocities = []

        for i in range(1, len(keypoints_list)):
            prev_frame = keypoints_list[i-1]
            curr_frame = keypoints_list[i]

            if prev_frame is None or curr_frame is None:
                wrist_velocities.append(0)
                continue

            # Right wrist is landmark 16
            if 16 not in prev_frame or 16 not in curr_frame:
                wrist_velocities.append(0)
                continue

            prev_wrist = prev_frame[16]
            curr_wrist = curr_frame[16]

            # Calculate velocity (Euclidean distance)
            velocity = np.sqrt(
                (curr_wrist['x'] - prev_wrist['x'])**2 +
                (curr_wrist['y'] - prev_wrist['y'])**2
            )
            wrist_velocities.append(velocity)

        # Contact is typically at peak velocity
        if wrist_velocities:
            contact_frame = np.argmax(wrist_velocities) + 1  # +1 because we started at frame 1
        else:
            contact_frame = len(keypoints_list) // 2  # Default to middle

        return contact_frame

    def get_keypoint_names(self):
        """Return mapping of keypoint indices to names"""
        return {
            0: 'nose',
            11: 'left_shoulder',
            12: 'right_shoulder',
            13: 'left_elbow',
            14: 'right_elbow',
            15: 'left_wrist',
            16: 'right_wrist',
            23: 'left_hip',
            24: 'right_hip',
            25: 'left_knee',
            26: 'right_knee',
            27: 'left_ankle',
            28: 'right_ankle'
        }

    def cleanup(self):
        """Release resources"""
        self.pose.close()
