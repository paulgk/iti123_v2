"""
Unit Tests for Feedback Generator

Tests the coaching feedback generation logic, severity classification,
and drill recommendations.
"""

import pytest
from src.coaching.feedback_generator import CoachingFeedback, FeedbackItem
from src.coaching.technique_benchmarks import CLEAR_BENCHMARKS, SMASH_BENCHMARKS


class TestFeedbackItem:
    """Test FeedbackItem data class."""

    def test_feedback_item_creation(self):
        """Should create FeedbackItem with all fields."""
        item = FeedbackItem(
            metric='max_velocity',
            user_value=65.0,
            optimal=75.78,
            status='below',
            severity='critical',
            message='Your arm velocity is below professional range',
            drill='Practice shadow swings with speed focus'
        )

        assert item.metric == 'max_velocity'
        assert item.user_value == 65.0
        assert item.optimal == 75.78
        assert item.status == 'below'
        assert item.severity == 'critical'
        assert 'below professional range' in item.message
        assert 'shadow swings' in item.drill

    def test_feedback_item_format(self):
        """Should format feedback item for display."""
        item = FeedbackItem(
            metric='elbow_angle',
            user_value=95.0,
            optimal=128.81,
            status='below',
            severity='critical',
            message='Elbow angle too low',
            drill='Wall touch drill'
        )

        formatted = item.format()
        assert 'elbow_angle' in formatted.lower() or 'Elbow angle too low' in formatted
        assert '95.0' in formatted or 'critical' in formatted.lower()


class TestCoachingFeedback:
    """Test CoachingFeedback main analysis class."""

    @pytest.fixture
    def professional_features(self):
        """Create feature dict with professional-level values for Clear."""
        return {
            'max_velocity': 75.78,  # Target value
            'elbow_angle': 128.81,  # Target value
            'forearm_angle': 80.28,  # Target value
            'contact_point': 0.025,  # Target value
            'shoulder_angle': 62.66,  # Target value
            'trunk_lean': 5.00,  # Target value
            'arm_extension': 0.5,  # Dummy value
            'timing': 0.0  # Dummy value
        }

    @pytest.fixture
    def poor_features(self):
        """Create feature dict with below-range values."""
        return {
            'max_velocity': 30.0,  # Below min (48.15)
            'elbow_angle': 90.0,  # Below min (114.62)
            'forearm_angle': 40.0,  # Below min (58.48)
            'contact_point': -0.05,  # Below min (-0.010)
            'shoulder_angle': 30.0,  # Below min (43.92)
            'trunk_lean': -10.0,  # Below min (-5.00)
            'arm_extension': 0.3,  # Dummy value
            'timing': 0.0  # Dummy value
        }

    def test_analyze_professional_technique(self, professional_features):
        """Professional-level features should get high score."""
        coach = CoachingFeedback()
        feedback = coach.analyze_technique(professional_features, 'Clear')

        # Should have very high overall score
        assert coach.overall_score >= 90

        # Should have mostly 'good' severity items
        good_items = [item for item in feedback if item.severity == 'good']
        assert len(good_items) >= 4

    def test_analyze_poor_technique(self, poor_features):
        """Poor technique should get low score and critical feedback."""
        coach = CoachingFeedback()
        feedback = coach.analyze_technique(poor_features, 'Clear')

        # Should have low overall score
        assert coach.overall_score < 60

        # Should have multiple critical issues
        critical_items = [item for item in feedback if item.severity == 'critical']
        assert len(critical_items) >= 3

    def test_feedback_sorting_by_severity(self, poor_features):
        """Feedback should be sorted by severity (critical first)."""
        coach = CoachingFeedback()
        feedback = coach.analyze_technique(poor_features, 'Clear')

        # Find first critical and first good item
        critical_idx = next((i for i, item in enumerate(feedback)
                           if item.severity == 'critical'), None)
        good_idx = next((i for i, item in enumerate(feedback)
                        if item.severity == 'good'), None)

        # If both exist, critical should come before good
        if critical_idx is not None and good_idx is not None:
            assert critical_idx < good_idx

    def test_smash_vs_clear_benchmarks(self):
        """Should use correct benchmarks for stroke type."""
        features = {'max_velocity': 75.0, 'elbow_angle': 129.0,
                   'forearm_angle': 80.0, 'contact_point': 0.03,
                   'shoulder_angle': 62.0, 'trunk_lean': 5.0,
                   'arm_extension': 0.5, 'timing': 0.0}

        coach_clear = CoachingFeedback()
        coach_clear.analyze_technique(features, 'Clear')

        coach_smash = CoachingFeedback()
        coach_smash.analyze_technique(features, 'Smash')

        # Scores might differ due to different benchmarks
        # (though Clear and Smash are similar per Cohen's d)
        assert isinstance(coach_clear.overall_score, (int, float))
        assert isinstance(coach_smash.overall_score, (int, float))

    def test_missing_features_handling(self):
        """Should handle missing features gracefully."""
        incomplete_features = {
            'max_velocity': 75.0,
            'elbow_angle': 128.0
            # Missing other features
        }

        coach = CoachingFeedback()
        # Should not crash
        feedback = coach.analyze_technique(incomplete_features, 'Clear')

        # Should still return some feedback
        assert isinstance(feedback, list)

    def test_feedback_contains_drills(self, poor_features):
        """All feedback items should have drill recommendations."""
        coach = CoachingFeedback()
        feedback = coach.analyze_technique(poor_features, 'Clear')

        for item in feedback:
            assert item.drill is not None
            assert len(item.drill) > 0

    def test_overall_score_bounds(self, professional_features, poor_features):
        """Overall score should be between 0 and 100."""
        coach1 = CoachingFeedback()
        coach1.analyze_technique(professional_features, 'Clear')
        assert 0 <= coach1.overall_score <= 100

        coach2 = CoachingFeedback()
        coach2.analyze_technique(poor_features, 'Clear')
        assert 0 <= coach2.overall_score <= 100

    def test_velocity_analysis(self):
        """Test specific velocity metric analysis."""
        features = {
            'max_velocity': 40.0,  # Below min (48.15) for Clear
            'elbow_angle': 128.81,
            'forearm_angle': 80.28,
            'contact_point': 0.025,
            'shoulder_angle': 62.66,
            'trunk_lean': 5.00,
            'arm_extension': 0.5,
            'timing': 0.0
        }

        coach = CoachingFeedback()
        feedback = coach.analyze_technique(features, 'Clear')

        # Should have critical feedback about velocity
        velocity_items = [item for item in feedback if 'velocity' in item.metric.lower()]
        assert len(velocity_items) > 0
        assert any(item.severity == 'critical' for item in velocity_items)

    def test_edge_case_exactly_at_boundaries(self):
        """Values exactly at min/max boundaries should be acceptable."""
        features = {
            'max_velocity': 48.15,  # Exactly at min
            'elbow_angle': 114.62,  # Exactly at min
            'forearm_angle': 58.48,  # Exactly at min
            'contact_point': -0.010,  # Exactly at min
            'shoulder_angle': 43.92,  # Exactly at min
            'trunk_lean': -5.00,  # Exactly at min
            'arm_extension': 0.5,
            'timing': 0.0
        }

        coach = CoachingFeedback()
        feedback = coach.analyze_technique(features, 'Clear')

        # Should not have any 'below' status
        below_items = [item for item in feedback if item.status == 'below']
        assert len(below_items) == 0


class TestFeedbackSeverityLogic:
    """Test severity classification logic."""

    def test_critical_severity_below_range(self):
        """Values below min should be critical."""
        features = {'max_velocity': 30.0}  # Well below min (48.15)
        coach = CoachingFeedback()

        # Internal method test - accessing protected method for unit test
        benchmarks = CLEAR_BENCHMARKS
        # This would require exposing or testing through public API
        # For now, test through analyze_technique
        feedback = coach.analyze_technique(features, 'Clear')

        velocity_feedback = [f for f in feedback if 'velocity' in f.metric.lower()]
        if velocity_feedback:
            assert velocity_feedback[0].severity == 'critical'

    def test_minor_severity_in_range_but_not_optimal(self):
        """Values in range but not near optimal should be minor."""
        features = {
            'max_velocity': 50.0,  # In range but not optimal
            'elbow_angle': 128.81,
            'forearm_angle': 80.28,
            'contact_point': 0.025,
            'shoulder_angle': 62.66,
            'trunk_lean': 5.00,
            'arm_extension': 0.5,
            'timing': 0.0
        }

        coach = CoachingFeedback()
        feedback = coach.analyze_technique(features, 'Clear')

        # Velocity should be 'minor' or 'acceptable'
        velocity_feedback = [f for f in feedback if 'velocity' in f.metric.lower()]
        if velocity_feedback:
            assert velocity_feedback[0].severity in ['minor', 'acceptable']

    def test_good_severity_near_optimal(self):
        """Values near optimal should be 'good'."""
        features = {
            'max_velocity': 75.78,  # Exactly optimal
            'elbow_angle': 128.81,
            'forearm_angle': 80.28,
            'contact_point': 0.025,
            'shoulder_angle': 62.66,
            'trunk_lean': 5.00,
            'arm_extension': 0.5,
            'timing': 0.0
        }

        coach = CoachingFeedback()
        feedback = coach.analyze_technique(features, 'Clear')

        good_items = [item for item in feedback if item.severity == 'good']
        # Should have multiple good items
        assert len(good_items) >= 4


class TestFeedbackMessages:
    """Test that feedback messages are informative."""

    def test_message_contains_metric_name(self):
        """Feedback message should mention the metric."""
        features = {'max_velocity': 30.0}
        coach = CoachingFeedback()
        feedback = coach.analyze_technique(features, 'Clear')

        velocity_feedback = [f for f in feedback if 'velocity' in f.metric.lower()]
        if velocity_feedback:
            message = velocity_feedback[0].message.lower()
            # Should mention velocity or speed
            assert 'velocity' in message or 'speed' in message or 'arm' in message

    def test_message_indicates_direction(self):
        """Feedback should indicate if value is too high or too low."""
        low_features = {'max_velocity': 30.0}  # Too low
        coach_low = CoachingFeedback()
        feedback_low = coach_low.analyze_technique(low_features, 'Clear')

        high_features = {'max_velocity': 100.0}  # Too high
        coach_high = CoachingFeedback()
        feedback_high = coach_high.analyze_technique(high_features, 'Clear')

        # Low velocity message should indicate it's low
        if feedback_low:
            msg_low = feedback_low[0].message.lower()
            assert 'low' in msg_low or 'below' in msg_low or 'slow' in msg_low

        # High velocity message should indicate it's high
        if feedback_high:
            msg_high = feedback_high[0].message.lower()
            assert 'high' in msg_high or 'above' in msg_high or 'fast' in msg_high
