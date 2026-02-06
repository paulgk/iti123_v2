"""
Technique Analyzer Module
Analyzes shot technique based on pose keypoints
Provides metrics and scoring for Smash and Clear
"""

import numpy as np
from typing import Dict, List, Optional


class TechniqueAnalyzer:
    """Analyze shot technique from pose keypoints"""

    def __init__(self):
        self.keypoint_names = {
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

    def analyze_shot(self, keypoints_list, shot_type, contact_frame_idx):
        """
        Analyze shot technique based on shot type

        Args:
            keypoints_list: List of keypoint dicts per frame
            shot_type: 'Smash' or 'Clear'
            contact_frame_idx: Frame index where contact occurs

        Returns:
            dict with metrics, scores, and feedback
        """
        if shot_type == 'Smash':
            return self.analyze_smash(keypoints_list, contact_frame_idx)
        elif shot_type == 'Clear':
            return self.analyze_clear(keypoints_list, contact_frame_idx)
        else:
            return {
                'success': False,
                'error': f'Shot type "{shot_type}" not supported yet'
            }

    def analyze_smash(self, keypoints_list, contact_frame_idx):
        """
        Analyze Smash technique

        Metrics:
        1. Arm extension (at contact)
        2. Shoulder rotation (pre-contact to follow-through)
        3. Knee bend (pre-contact)
        4. Contact point height (relative to head)
        5. Jump height (if jumping)
        6. Follow-through (post-contact)
        7. Balance/stability (foot positioning)
        """
        # Get key frames
        pre_contact_idx = max(0, contact_frame_idx - 3)
        post_contact_idx = min(len(keypoints_list) - 1, contact_frame_idx + 3)

        pre_contact = keypoints_list[pre_contact_idx]
        at_contact = keypoints_list[contact_frame_idx]
        post_contact = keypoints_list[post_contact_idx]

        if not all([pre_contact, at_contact, post_contact]):
            return {
                'success': False,
                'error': 'Missing pose data in key frames'
            }

        metrics = {}

        # 1. Arm Extension (at contact)
        arm_extension = self._calculate_arm_extension(at_contact)
        metrics['arm_extension'] = {
            'value': arm_extension,
            'score': self._score_arm_extension(arm_extension),
            'optimal': 0.95,
            'unit': 'ratio'
        }

        # 2. Shoulder Rotation
        shoulder_rotation = self._calculate_shoulder_rotation(pre_contact, post_contact)
        metrics['shoulder_rotation'] = {
            'value': shoulder_rotation,
            'score': self._score_shoulder_rotation(shoulder_rotation),
            'optimal': 120,
            'unit': 'degrees'
        }

        # 3. Knee Bend (pre-contact)
        knee_bend = self._calculate_knee_angle(pre_contact)
        metrics['knee_bend'] = {
            'value': knee_bend,
            'score': self._score_knee_bend(knee_bend),
            'optimal': 35,
            'unit': 'degrees'
        }

        # 4. Contact Height
        contact_height = self._calculate_contact_height(at_contact)
        metrics['contact_height'] = {
            'value': contact_height,
            'score': self._score_contact_height(contact_height),
            'optimal': 0.30,
            'unit': 'ratio'
        }

        # 5. Follow-through
        follow_through = self._calculate_follow_through(at_contact, post_contact)
        metrics['follow_through'] = {
            'value': follow_through,
            'score': self._score_follow_through(follow_through),
            'optimal': 90,
            'unit': 'degrees'
        }

        # Calculate overall score
        overall_score = np.mean([m['score'] for m in metrics.values()])

        # Generate feedback
        feedback = self._generate_smash_feedback(metrics, overall_score)

        return {
            'success': True,
            'shot_type': 'Smash',
            'overall_score': round(overall_score, 1),
            'metrics': metrics,
            'feedback': feedback
        }

    def analyze_clear(self, keypoints_list, contact_frame_idx):
        """
        Analyze Clear technique

        Metrics:
        1. Elbow height (pre-contact)
        2. Weight transfer (back to front foot)
        3. Swing arc (full rotation)
        4. Contact timing (late contact for height)
        5. Arm extension (at contact)
        6. Follow-through (complete swing)
        7. Body posture (upright for power)
        """
        pre_contact_idx = max(0, contact_frame_idx - 3)
        post_contact_idx = min(len(keypoints_list) - 1, contact_frame_idx + 3)

        pre_contact = keypoints_list[pre_contact_idx]
        at_contact = keypoints_list[contact_frame_idx]
        post_contact = keypoints_list[post_contact_idx]

        if not all([pre_contact, at_contact, post_contact]):
            return {
                'success': False,
                'error': 'Missing pose data in key frames'
            }

        metrics = {}

        # 1. Elbow Height (pre-contact)
        elbow_height = self._calculate_elbow_height(pre_contact)
        metrics['elbow_height'] = {
            'value': elbow_height,
            'score': self._score_elbow_height(elbow_height),
            'optimal': 0.25,
            'unit': 'ratio'
        }

        # 2. Arm Extension (at contact)
        arm_extension = self._calculate_arm_extension(at_contact)
        metrics['arm_extension'] = {
            'value': arm_extension,
            'score': self._score_arm_extension(arm_extension),
            'optimal': 0.92,
            'unit': 'ratio'
        }

        # 3. Swing Arc
        swing_arc = self._calculate_shoulder_rotation(pre_contact, post_contact)
        metrics['swing_arc'] = {
            'value': swing_arc,
            'score': self._score_swing_arc(swing_arc),
            'optimal': 100,
            'unit': 'degrees'
        }

        # 4. Contact Height
        contact_height = self._calculate_contact_height(at_contact)
        metrics['contact_height'] = {
            'value': contact_height,
            'score': self._score_contact_height_clear(contact_height),
            'optimal': 0.25,
            'unit': 'ratio'
        }

        # 5. Follow-through
        follow_through = self._calculate_follow_through(at_contact, post_contact)
        metrics['follow_through'] = {
            'value': follow_through,
            'score': self._score_follow_through(follow_through),
            'optimal': 85,
            'unit': 'degrees'
        }

        # Calculate overall score
        overall_score = np.mean([m['score'] for m in metrics.values()])

        # Generate feedback
        feedback = self._generate_clear_feedback(metrics, overall_score)

        return {
            'success': True,
            'shot_type': 'Clear',
            'overall_score': round(overall_score, 1),
            'metrics': metrics,
            'feedback': feedback
        }

    # ==================== Calculation Methods ====================

    def _calculate_arm_extension(self, keypoints):
        """Calculate arm extension ratio at contact"""
        if not all(k in keypoints for k in [12, 14, 16]):  # shoulder, elbow, wrist
            return 0.0

        shoulder = np.array([keypoints[12]['x'], keypoints[12]['y']])
        elbow = np.array([keypoints[14]['x'], keypoints[14]['y']])
        wrist = np.array([keypoints[16]['x'], keypoints[16]['y']])

        # Calculate arm segments
        upper_arm = np.linalg.norm(elbow - shoulder)
        forearm = np.linalg.norm(wrist - elbow)
        full_reach = np.linalg.norm(wrist - shoulder)

        # Extension ratio (1.0 = fully extended)
        max_reach = upper_arm + forearm
        if max_reach == 0:
            return 0.0

        extension_ratio = full_reach / max_reach
        return min(extension_ratio, 1.0)

    def _calculate_shoulder_rotation(self, pre_frame, post_frame):
        """Calculate shoulder rotation angle"""
        if not all(k in pre_frame for k in [11, 12]) or not all(k in post_frame for k in [11, 12]):
            return 0.0

        # Pre-contact shoulder line
        pre_left = np.array([pre_frame[11]['x'], pre_frame[11]['y']])
        pre_right = np.array([pre_frame[12]['x'], pre_frame[12]['y']])
        pre_vector = pre_right - pre_left

        # Post-contact shoulder line
        post_left = np.array([post_frame[11]['x'], post_frame[11]['y']])
        post_right = np.array([post_frame[12]['x'], post_frame[12]['y']])
        post_vector = post_right - post_left

        # Calculate angle between vectors
        cos_angle = np.dot(pre_vector, post_vector) / (np.linalg.norm(pre_vector) * np.linalg.norm(post_vector) + 1e-6)
        angle = np.degrees(np.arccos(np.clip(cos_angle, -1.0, 1.0)))

        return angle

    def _calculate_knee_angle(self, keypoints):
        """Calculate right knee angle"""
        if not all(k in keypoints for k in [24, 26, 28]):  # hip, knee, ankle
            return 180.0  # Straight leg

        hip = np.array([keypoints[24]['x'], keypoints[24]['y']])
        knee = np.array([keypoints[26]['x'], keypoints[26]['y']])
        ankle = np.array([keypoints[28]['x'], keypoints[28]['y']])

        # Vectors
        v1 = hip - knee
        v2 = ankle - knee

        # Angle
        cos_angle = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-6)
        angle = np.degrees(np.arccos(np.clip(cos_angle, -1.0, 1.0)))

        return 180 - angle  # Convert to bend angle

    def _calculate_contact_height(self, keypoints):
        """Calculate contact height relative to head"""
        if not all(k in keypoints for k in [0, 16]):  # nose, right wrist
            return 0.0

        nose_y = keypoints[0]['y']
        wrist_y = keypoints[16]['y']

        # Relative height (positive = above head)
        relative_height = (nose_y - wrist_y)  # MediaPipe y is inverted

        return max(relative_height, 0.0)

    def _calculate_elbow_height(self, keypoints):
        """Calculate elbow height relative to shoulder"""
        if not all(k in keypoints for k in [12, 14]):  # shoulder, elbow
            return 0.0

        shoulder_y = keypoints[12]['y']
        elbow_y = keypoints[14]['y']

        relative_height = (shoulder_y - elbow_y)  # Positive = above shoulder

        return max(relative_height, 0.0)

    def _calculate_follow_through(self, contact_frame, post_frame):
        """Calculate follow-through angle"""
        if not all(k in contact_frame for k in [12, 16]) or not all(k in post_frame for k in [12, 16]):
            return 0.0

        # Contact arm position
        contact_shoulder = np.array([contact_frame[12]['x'], contact_frame[12]['y']])
        contact_wrist = np.array([contact_frame[16]['x'], contact_frame[16]['y']])
        contact_vector = contact_wrist - contact_shoulder

        # Follow-through arm position
        post_shoulder = np.array([post_frame[12]['x'], post_frame[12]['y']])
        post_wrist = np.array([post_frame[16]['x'], post_frame[16]['y']])
        post_vector = post_wrist - post_shoulder

        # Angle
        cos_angle = np.dot(contact_vector, post_vector) / (np.linalg.norm(contact_vector) * np.linalg.norm(post_vector) + 1e-6)
        angle = np.degrees(np.arccos(np.clip(cos_angle, -1.0, 1.0)))

        return angle

    # ==================== Scoring Methods ====================

    def _score_arm_extension(self, value):
        """Score arm extension (0-100)"""
        if value >= 0.95:
            return 100
        elif value >= 0.85:
            return 80 + (value - 0.85) * 200
        elif value >= 0.75:
            return 60 + (value - 0.75) * 200
        else:
            return max(value * 80, 0)

    def _score_shoulder_rotation(self, value):
        """Score shoulder rotation (0-100)"""
        if value >= 120:
            return 100
        elif value >= 90:
            return 75 + (value - 90) * 0.83
        elif value >= 60:
            return 50 + (value - 60) * 0.83
        else:
            return max(value * 0.83, 0)

    def _score_knee_bend(self, value):
        """Score knee bend (0-100)"""
        optimal = 35
        if 30 <= value <= 45:
            return 100
        elif 20 <= value < 30:
            return 60 + (value - 20) * 4
        elif 45 < value <= 60:
            return 60 + (60 - value)
        else:
            return max(50 - abs(value - optimal), 0)

    def _score_contact_height(self, value):
        """Score contact height for Smash (0-100)"""
        if value >= 0.30:
            return 100
        elif value >= 0.20:
            return 70 + (value - 0.20) * 300
        elif value >= 0.10:
            return 40 + (value - 0.10) * 300
        else:
            return max(value * 400, 0)

    def _score_contact_height_clear(self, value):
        """Score contact height for Clear (0-100)"""
        if value >= 0.25:
            return 100
        elif value >= 0.15:
            return 70 + (value - 0.15) * 300
        elif value >= 0.10:
            return 50 + (value - 0.10) * 400
        else:
            return max(value * 500, 0)

    def _score_elbow_height(self, value):
        """Score elbow height (0-100)"""
        if value >= 0.25:
            return 100
        elif value >= 0.15:
            return 75 + (value - 0.15) * 250
        elif value >= 0.10:
            return 50 + (value - 0.10) * 500
        else:
            return max(value * 500, 0)

    def _score_follow_through(self, value):
        """Score follow-through (0-100)"""
        if value >= 80:
            return 100
        elif value >= 60:
            return 70 + (value - 60) * 1.5
        elif value >= 40:
            return 50 + (value - 40) * 1.0
        else:
            return max(value * 1.25, 0)

    def _score_swing_arc(self, value):
        """Score swing arc for Clear (0-100)"""
        if value >= 100:
            return 100
        elif value >= 80:
            return 80 + (value - 80)
        elif value >= 60:
            return 60 + (value - 60)
        else:
            return max(value, 0)

    # ==================== Feedback Generation ====================

    def _generate_smash_feedback(self, metrics, overall_score):
        """Generate feedback for Smash technique"""
        feedback = []

        # Categorize
        excellent = []
        good = []
        needs_work = []

        for metric_name, metric_data in metrics.items():
            score = metric_data['score']
            if score >= 85:
                excellent.append(metric_name)
            elif score >= 65:
                good.append(metric_name)
            else:
                needs_work.append(metric_name)

        # Build feedback list
        if excellent:
            feedback.append({
                'category': 'Strengths',
                'items': [self._format_metric_name(m) for m in excellent]
            })

        if needs_work:
            improvements = []
            for metric_name in needs_work:
                metric_data = metrics[metric_name]
                improvements.append(self._get_improvement_tip(metric_name, metric_data, 'Smash'))

            feedback.append({
                'category': 'Areas to Improve',
                'items': improvements
            })

        # Overall assessment
        if overall_score >= 85:
            rating = "Excellent"
            comment = "Your Smash technique is very strong!"
        elif overall_score >= 70:
            rating = "Good"
            comment = "Solid Smash technique with room for minor improvements."
        elif overall_score >= 55:
            rating = "Fair"
            comment = "Your Smash has potential, but needs work on fundamentals."
        else:
            rating = "Needs Improvement"
            comment = "Focus on building proper Smash technique from basics."

        feedback.insert(0, {
            'category': 'Overall Assessment',
            'rating': rating,
            'comment': comment
        })

        return feedback

    def _generate_clear_feedback(self, metrics, overall_score):
        """Generate feedback for Clear technique"""
        feedback = []

        # Categorize
        excellent = []
        good = []
        needs_work = []

        for metric_name, metric_data in metrics.items():
            score = metric_data['score']
            if score >= 85:
                excellent.append(metric_name)
            elif score >= 65:
                good.append(metric_name)
            else:
                needs_work.append(metric_name)

        # Build feedback list
        if excellent:
            feedback.append({
                'category': 'Strengths',
                'items': [self._format_metric_name(m) for m in excellent]
            })

        if needs_work:
            improvements = []
            for metric_name in needs_work:
                metric_data = metrics[metric_name]
                improvements.append(self._get_improvement_tip(metric_name, metric_data, 'Clear'))

            feedback.append({
                'category': 'Areas to Improve',
                'items': improvements
            })

        # Overall assessment
        if overall_score >= 85:
            rating = "Excellent"
            comment = "Your Clear technique is very strong!"
        elif overall_score >= 70:
            rating = "Good"
            comment = "Solid Clear technique with room for minor improvements."
        elif overall_score >= 55:
            rating = "Fair"
            comment = "Your Clear has potential, but needs work on fundamentals."
        else:
            rating = "Needs Improvement"
            comment = "Focus on building proper Clear technique from basics."

        feedback.insert(0, {
            'category': 'Overall Assessment',
            'rating': rating,
            'comment': comment
        })

        return feedback

    def _format_metric_name(self, metric_name):
        """Format metric name for display"""
        return metric_name.replace('_', ' ').title()

    def _get_improvement_tip(self, metric_name, metric_data, shot_type):
        """Get specific improvement tip for a metric"""
        tips = {
            'Smash': {
                'arm_extension': f"Extend your arm more at contact (current: {metric_data['value']:.0%}, target: 95%+)",
                'shoulder_rotation': f"Rotate shoulders more (current: {metric_data['value']:.0f}°, target: 120°+)",
                'knee_bend': f"Bend knees deeper before jump (current: {metric_data['value']:.0f}°, target: 30-45°)",
                'contact_height': "Hit the shuttlecock higher above your head",
                'follow_through': "Complete your follow-through motion (let racket cross body)"
            },
            'Clear': {
                'elbow_height': "Raise your elbow higher before contact",
                'arm_extension': f"Extend your arm more (current: {metric_data['value']:.0%}, target: 92%+)",
                'swing_arc': f"Complete full swing arc (current: {metric_data['value']:.0f}°, target: 100°+)",
                'contact_height': "Make contact higher for better trajectory",
                'follow_through': "Follow through completely after contact"
            }
        }

        return tips.get(shot_type, {}).get(metric_name, f"Improve {self._format_metric_name(metric_name)}")
