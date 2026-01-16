"""
Coaching Module - AI-Enhanced Badminton Coaching

Provides technique analysis, feedback generation, and visualization tools
for coaching feedback based on biomechanical analysis.
"""

from .technique_benchmarks import TechniqueBenchmarks
from .feedback_generator import CoachingFeedback, FeedbackItem
from .visualizations import TechniqueVisualizer

__all__ = ['TechniqueBenchmarks', 'CoachingFeedback', 'FeedbackItem', 'TechniqueVisualizer']
