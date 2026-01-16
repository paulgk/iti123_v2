"""
Coaching Feedback Generator

Analyzes biomechanical features and generates personalized coaching feedback
with practice drills based on professional benchmarks.
"""

from typing import Dict, List, Tuple
from .technique_benchmarks import TechniqueBenchmarks


class FeedbackItem:
    """Single piece of coaching feedback"""

    def __init__(self,
                 metric: str,
                 severity: str,
                 message: str,
                 current_value: float,
                 target_range: Tuple[float, float],
                 drill: str,
                 impact: str):
        """
        Args:
            metric: Name of the biomechanical metric
            severity: 'critical', 'major', 'minor', 'good'
            message: Human-readable coaching message
            current_value: User's measured value
            target_range: (min, max) professional range
            drill: Practice drill recommendation
            impact: Expected improvement from fixing this issue
        """
        self.metric = metric
        self.severity = severity
        self.message = message
        self.current_value = current_value
        self.target_range = target_range
        self.drill = drill
        self.impact = impact

    def __repr__(self):
        return f"FeedbackItem({self.metric}, {self.severity})"

    def format(self) -> str:
        """Format feedback for display"""
        severity_icons = {
            'critical': '🔴',
            'major': '⚠️',
            'minor': '💡',
            'good': '✅'
        }

        icon = severity_icons.get(self.severity, '📊')
        formatted_value = TechniqueBenchmarks.format_value(self.current_value, self.metric)
        min_val = TechniqueBenchmarks.format_value(self.target_range[0], self.metric)
        max_val = TechniqueBenchmarks.format_value(self.target_range[1], self.metric)

        output = f"{icon} {self.message}\n"
        output += f"   Current: {formatted_value}  |  Target: {min_val} - {max_val}\n"
        output += f"   → {self.impact}\n"
        output += f"   🎯 Drill: {self.drill}\n"

        return output


class CoachingFeedback:
    """Generate personalized coaching feedback from biomechanical analysis"""

    def __init__(self):
        self.feedback_items = []
        self.overall_score = 0
        self.stroke_type = None

    def analyze_technique(self, features: Dict, stroke_type: str) -> List[FeedbackItem]:
        """
        Analyze technique and generate coaching feedback

        Args:
            features: Dictionary of biomechanical features (from feature_engineering_v2.py)
            stroke_type: 'Clear', 'Smash', or equivalent

        Returns:
            List of FeedbackItem objects, ordered by severity

        Example:
            >>> feedback_gen = CoachingFeedback()
            >>> items = feedback_gen.analyze_technique(features, 'Clear')
            >>> for item in items[:3]:  # Show top 3 issues
            ...     print(item.format())
        """
        self.stroke_type = stroke_type
        self.feedback_items = []

        # Get benchmarks for this stroke type
        benchmarks = TechniqueBenchmarks.get_benchmarks(stroke_type)

        # Run all analysis functions
        self._analyze_arm_extension(features, benchmarks)
        self._analyze_velocity(features, benchmarks)
        self._analyze_elbow_angle(features, benchmarks)
        self._analyze_posture(features, benchmarks)
        self._analyze_timing(features, benchmarks)
        self._analyze_contact_point(features, benchmarks)

        # Sort by severity (critical > major > minor > good)
        severity_order = {'critical': 0, 'major': 1, 'minor': 2, 'good': 3}
        self.feedback_items.sort(key=lambda x: severity_order.get(x.severity, 4))

        # Calculate overall score
        self.overall_score = self._calculate_overall_score(features, benchmarks)

        return self.feedback_items

    def _analyze_arm_extension(self, features: Dict, benchmarks: Dict):
        """Analyze arm extension at contact point"""
        metric = 'r_arm_extension_mean'

        if metric not in features or metric not in benchmarks:
            return

        value = features[metric]
        benchmark = benchmarks[metric]
        status, distance = TechniqueBenchmarks.get_metric_status(value, benchmark)

        if status == 'below':
            severity = 'critical' if distance > 0.02 else 'major'
            message = "Arm Extension: Insufficient reach"
            impact = "Extend arm fully at contact for 15% more power"
            drill = "10 overhead shadow swings, focus on full extension"

        elif status == 'above':
            severity = 'minor'
            message = "Arm Extension: Overextending"
            impact = "Slightly reduce extension to improve control"
            drill = "Mirror practice: Check arm position at contact point"

        elif status == 'acceptable':
            severity = 'minor'
            message = "Arm Extension: Near optimal range"
            impact = f"Fine-tune to reach optimal ({benchmark['optimal']:.3f})"
            drill = "Focus on consistency in extension"

        else:  # optimal
            severity = 'good'
            message = "Arm Extension: Excellent"
            impact = "Maintain this technique"
            drill = "Continue current practice"

        self.feedback_items.append(FeedbackItem(
            metric=metric,
            severity=severity,
            message=message,
            current_value=value,
            target_range=(benchmark['min'], benchmark['max']),
            drill=drill,
            impact=impact
        ))

    def _analyze_velocity(self, features: Dict, benchmarks: Dict):
        """Analyze racket head velocity (via wrist velocity proxy)"""
        metric = 'max_velocity'

        if metric not in features or metric not in benchmarks:
            return

        value = features[metric]
        benchmark = benchmarks[metric]
        status, distance = TechniqueBenchmarks.get_metric_status(value, benchmark)

        if status == 'below':
            severity = 'critical' if distance > 15 else 'major'
            message = f"Velocity: Below target for {self.stroke_type}"
            impact = "Increase wrist snap speed for more power"
            drill = "Resistance band training, 3 sets × 12 reps"

        elif status == 'above':
            severity = 'minor'
            message = "Velocity: Above professional average"
            impact = "Excellent power! Focus on accuracy now"
            drill = "Target practice: Aim for specific court areas"

        elif status == 'acceptable':
            severity = 'minor'
            message = "Velocity: Good speed"
            impact = f"Aim for optimal speed ({benchmark['optimal']:.1f})"
            drill = "Plyometric exercises for explosive power"

        else:  # optimal
            severity = 'good'
            message = "Velocity: Optimal racket speed"
            impact = "Perfect power generation"
            drill = "Maintain through regular practice"

        self.feedback_items.append(FeedbackItem(
            metric=metric,
            severity=severity,
            message=message,
            current_value=value,
            target_range=(benchmark['min'], benchmark['max']),
            drill=drill,
            impact=impact
        ))

    def _analyze_elbow_angle(self, features: Dict, benchmarks: Dict):
        """Analyze elbow angle at contact"""
        metric = 'r_elbow_angle_mean'

        if metric not in features or metric not in benchmarks:
            return

        value = features[metric]
        benchmark = benchmarks[metric]
        status, distance = TechniqueBenchmarks.get_metric_status(value, benchmark)

        if status == 'below':
            severity = 'major'
            message = "Elbow Angle: Too bent at contact"
            impact = "Straighten elbow for better reach and power transfer"
            drill = "Wall drills: Practice contact point with straighter arm"

        elif status == 'above':
            severity = 'major'
            message = "Elbow Angle: Too straight (locked)"
            impact = "Slight bend prevents injury and improves control"
            drill = "Partner throws: Focus on natural, relaxed contact"

        elif status == 'acceptable':
            severity = 'minor'
            message = "Elbow Angle: Good range"
            impact = "Fine-tune for optimal power transfer"
            drill = "Slow-motion swings focusing on elbow position"

        else:  # optimal
            severity = 'good'
            message = "Elbow Angle: Perfect positioning"
            impact = "Excellent power transfer"
            drill = "Continue current technique"

        self.feedback_items.append(FeedbackItem(
            metric=metric,
            severity=severity,
            message=message,
            current_value=value,
            target_range=(benchmark['min'], benchmark['max']),
            drill=drill,
            impact=impact
        ))

    def _analyze_posture(self, features: Dict, benchmarks: Dict):
        """Analyze body posture (torso lean)"""
        metric = 'torso_lean_mean'

        if metric not in features or metric not in benchmarks:
            return

        value = features[metric]
        benchmark = benchmarks[metric]
        status, distance = TechniqueBenchmarks.get_metric_status(value, benchmark)

        # Note: Negative values = leaning back (typical for overhead shots)
        if status == 'below':
            severity = 'major'
            message = "Posture: Leaning too far back"
            impact = "Reduce backward lean for better balance and power"
            drill = "Core strengthening: Planks and rotational exercises"

        elif status == 'above':
            severity = 'major'
            message = "Posture: Leaning too far forward"
            impact = "More backward lean for overhead power generation"
            drill = "Shadow practice: Focus on arching back at contact"

        elif status == 'acceptable':
            severity = 'minor'
            message = "Posture: Good body position"
            impact = "Slight adjustment for optimal balance"
            drill = "Video analysis: Compare with professional players"

        else:  # optimal
            severity = 'good'
            message = "Posture: Excellent body lean"
            impact = "Perfect balance for power"
            drill = "Maintain this positioning"

        self.feedback_items.append(FeedbackItem(
            metric=metric,
            severity=severity,
            message=message,
            current_value=value,
            target_range=(benchmark['min'], benchmark['max']),
            drill=drill,
            impact=impact
        ))

    def _analyze_timing(self, features: Dict, benchmarks: Dict):
        """Analyze peak velocity timing in stroke"""
        metric = 'peak_velocity_timing'

        if metric not in features or metric not in benchmarks:
            return

        value = features[metric]
        benchmark = benchmarks[metric]
        status, distance = TechniqueBenchmarks.get_metric_status(value, benchmark)

        if status == 'below':
            severity = 'minor'
            message = "Timing: Peak velocity too early"
            impact = "Delay power release for better contact"
            drill = "Rhythm drills: Count 1-2-3 with peak at contact"

        elif status == 'above':
            severity = 'minor'
            message = "Timing: Peak velocity too late"
            impact = "Earlier acceleration for better power transfer"
            drill = "Focus on explosive rotation before contact"

        elif status == 'acceptable':
            severity = 'minor'
            message = "Timing: Good acceleration pattern"
            impact = "Refine timing for maximum efficiency"
            drill = "Slow-motion practice: Feel the acceleration build-up"

        else:  # optimal
            severity = 'good'
            message = "Timing: Perfect velocity peak"
            impact = "Optimal power timing"
            drill = "Maintain this rhythm"

        self.feedback_items.append(FeedbackItem(
            metric=metric,
            severity=severity,
            message=message,
            current_value=value,
            target_range=(benchmark['min'], benchmark['max']),
            drill=drill,
            impact=impact
        ))

    def _analyze_contact_point(self, features: Dict, benchmarks: Dict):
        """Analyze contact point height (wrist height relative to shoulder)"""
        metric = 'r_wrist_height_rel_mean'

        if metric not in features or metric not in benchmarks:
            return

        value = features[metric]
        benchmark = benchmarks[metric]
        status, distance = TechniqueBenchmarks.get_metric_status(value, benchmark)

        if status == 'below':
            severity = 'major'
            message = "Contact Point: Too low"
            impact = "Raise contact point for better angle and power"
            drill = "Jump training: Practice contact at peak of jump"

        elif status == 'above':
            severity = 'minor'
            message = "Contact Point: Very high"
            impact = "Good height! Ensure you can control at this point"
            drill = "Consistency drills: Hit from same height repeatedly"

        elif status == 'acceptable':
            severity = 'minor'
            message = "Contact Point: Good height"
            impact = "Fine-tune for optimal contact"
            drill = "Mark optimal height with tape, practice hitting there"

        else:  # optimal
            severity = 'good'
            message = "Contact Point: Perfect height"
            impact = "Optimal contact positioning"
            drill = "Continue current technique"

        self.feedback_items.append(FeedbackItem(
            metric=metric,
            severity=severity,
            message=message,
            current_value=value,
            target_range=(benchmark['min'], benchmark['max']),
            drill=drill,
            impact=impact
        ))

    def _calculate_overall_score(self, features: Dict, benchmarks: Dict) -> int:
        """
        Calculate overall technique score (0-100)

        Scoring:
        - Each metric contributes equally
        - optimal = 100%, acceptable = 70-85%, below/above = 0-60%

        Returns:
            Score from 0-100
        """
        scores = []

        for metric, benchmark in benchmarks.items():
            if metric not in features:
                continue

            value = features[metric]
            status, distance = TechniqueBenchmarks.get_metric_status(value, benchmark)

            if status == 'optimal':
                metric_score = 100
            elif status == 'acceptable':
                # Score based on distance from optimal
                range_width = benchmark['max'] - benchmark['min']
                tolerance = range_width * 0.25
                distance_ratio = distance / tolerance
                metric_score = max(70, 100 - (distance_ratio * 15))
            else:  # below or above
                # Score based on how far outside range
                range_width = benchmark['max'] - benchmark['min']
                distance_ratio = distance / range_width
                metric_score = max(0, 60 - (distance_ratio * 40))

            scores.append(metric_score)

        if not scores:
            return 0

        overall = sum(scores) / len(scores)
        return int(round(overall))

    def generate_summary(self) -> str:
        """
        Generate text summary of analysis

        Returns:
            Formatted summary string with score and top issues

        Example:
            >>> feedback_gen = CoachingFeedback()
            >>> feedback_gen.analyze_technique(features, 'Clear')
            >>> print(feedback_gen.generate_summary())
        """
        # Score interpretation
        if self.overall_score >= 85:
            grade = "Excellent"
            emoji = "🌟"
        elif self.overall_score >= 70:
            grade = "Good"
            emoji = "✅"
        elif self.overall_score >= 50:
            grade = "Needs Improvement"
            emoji = "⚠️"
        else:
            grade = "Requires Attention"
            emoji = "🔴"

        summary = f"\n{'='*70}\n"
        summary += f"TECHNIQUE ANALYSIS - {self.stroke_type}\n"
        summary += f"{'='*70}\n\n"
        summary += f"{emoji} Overall Score: {self.overall_score}/100 - {grade}\n\n"

        # Count issues by severity
        critical = sum(1 for item in self.feedback_items if item.severity == 'critical')
        major = sum(1 for item in self.feedback_items if item.severity == 'major')
        minor = sum(1 for item in self.feedback_items if item.severity == 'minor')
        good = sum(1 for item in self.feedback_items if item.severity == 'good')

        summary += f"Issues Found:\n"
        if critical > 0:
            summary += f"  🔴 Critical: {critical}\n"
        if major > 0:
            summary += f"  ⚠️  Major: {major}\n"
        if minor > 0:
            summary += f"  💡 Minor: {minor}\n"
        if good > 0:
            summary += f"  ✅ Good: {good}\n"

        summary += f"\n{'='*70}\n"
        summary += "DETAILED FEEDBACK\n"
        summary += f"{'='*70}\n\n"

        # Show all feedback items
        for i, item in enumerate(self.feedback_items, 1):
            summary += f"{i}. {item.format()}\n"

        summary += f"{'='*70}\n"
        summary += "PRIORITY ACTIONS\n"
        summary += f"{'='*70}\n\n"

        # Top 3 priority items (excluding 'good' ones)
        priority_items = [item for item in self.feedback_items if item.severity != 'good'][:3]

        if priority_items:
            for i, item in enumerate(priority_items, 1):
                summary += f"{i}. {item.message}\n"
                summary += f"   🎯 {item.drill}\n\n"
        else:
            summary += "Great technique! Focus on maintaining consistency.\n\n"

        return summary


# Example usage and testing
if __name__ == "__main__":
    print("="*70)
    print("FEEDBACK GENERATOR TEST")
    print("="*70)
    print()

    # Mock features for testing
    test_features = {
        'r_arm_extension_mean': 0.072,  # Below optimal
        'max_velocity': 68.5,  # Below optimal
        'r_elbow_angle_mean': 128.0,  # Acceptable
        'torso_lean_mean': -25.0,  # Acceptable
        'r_wrist_height_rel_mean': 0.019,  # Acceptable
        'peak_velocity_timing': 0.35,  # Slightly early
        'r_forearm_vertical_angle_mean': 75.0,  # Good
        'shoulder_rotation_mean': -10.0  # Good
    }

    # Generate feedback
    coach = CoachingFeedback()
    feedback_items = coach.analyze_technique(test_features, 'Clear')

    # Print summary
    print(coach.generate_summary())

    print("\n✓ Feedback generator test complete!")
    print(f"Generated {len(feedback_items)} feedback items")
    print(f"Overall score: {coach.overall_score}/100")
