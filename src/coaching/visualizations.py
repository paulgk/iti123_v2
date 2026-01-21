"""
Visualization Tools for Coaching Feedback

Creates visual representations of technique analysis:
- Radar charts comparing user vs professional benchmarks
- Bar charts showing metric performance by severity
- Score gauges for overall technique rating
- Comparison charts for progress tracking
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import Circle, Wedge, Rectangle
from typing import Dict, List, Optional, Tuple
from pathlib import Path

from .technique_benchmarks import TechniqueBenchmarks
from .feedback_generator import CoachingFeedback, FeedbackItem


class TechniqueVisualizer:
    """Generate visual charts for coaching feedback"""

    def __init__(self, output_dir: Optional[Path] = None):
        """
        Args:
            output_dir: Directory to save charts (None = don't save, just return figures)
        """
        self.output_dir = output_dir
        if output_dir:
            output_dir.mkdir(parents=True, exist_ok=True)

        # Set style
        plt.style.use('seaborn-v0_8-darkgrid')
        self.colors = {
            'user': '#3498db',      # Blue
            'optimal': '#2ecc71',   # Green
            'critical': '#e74c3c',  # Red
            'major': '#f39c12',     # Orange
            'minor': '#f1c40f',     # Yellow
            'good': '#2ecc71'       # Green
        }

    def create_radar_chart(self,
                          features: Dict,
                          stroke_type: str,
                          title: Optional[str] = None) -> plt.Figure:
        """
        Create radar chart comparing user metrics to professional benchmarks

        Args:
            features: User's biomechanical features
            stroke_type: 'Clear' or 'Smash'
            title: Optional custom title

        Returns:
            matplotlib Figure object

        Example:
            >>> viz = TechniqueVisualizer()
            >>> fig = viz.create_radar_chart(features, 'Clear')
            >>> plt.show()
        """
        benchmarks = TechniqueBenchmarks.get_benchmarks(stroke_type)

        # Get metrics that exist in both features and benchmarks
        metrics = []
        user_values = []
        optimal_values = []
        metric_labels = []

        for metric in benchmarks.keys():
            if metric in features:
                metrics.append(metric)
                user_values.append(features[metric])
                optimal = TechniqueBenchmarks.get_target_value(benchmarks, metric)
                optimal_values.append(optimal)

                # Create readable labels
                label = metric.replace('_', ' ').replace('mean', '').strip()
                label = label.replace('r ', '').title()
                metric_labels.append(label)

        # Normalize values to 0-1 scale for radar chart
        normalized_user = []
        normalized_optimal = []

        for i, metric in enumerate(metrics):
            min_val, max_val = TechniqueBenchmarks.get_target_range(benchmarks, metric)
            range_width = max_val - min_val

            if range_width > 0:
                # Normalize to 0-1 based on professional range
                user_norm = (user_values[i] - min_val) / range_width
                optimal_norm = (optimal_values[i] - min_val) / range_width

                # Clip to 0-1 range
                user_norm = max(0, min(1, user_norm))
                optimal_norm = max(0, min(1, optimal_norm))
            else:
                user_norm = 0.5
                optimal_norm = 0.5

            normalized_user.append(user_norm)
            normalized_optimal.append(optimal_norm)

        # Number of metrics
        num_vars = len(metrics)

        # Compute angle for each axis
        angles = np.linspace(0, 2 * np.pi, num_vars, endpoint=False).tolist()

        # Complete the circle
        normalized_user += normalized_user[:1]
        normalized_optimal += normalized_optimal[:1]
        angles += angles[:1]

        # Create figure
        fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(projection='polar'))

        # Plot data
        ax.plot(angles, normalized_optimal, 'o-', linewidth=2,
                label='Professional Optimal', color=self.colors['optimal'])
        ax.fill(angles, normalized_optimal, alpha=0.15, color=self.colors['optimal'])

        ax.plot(angles, normalized_user, 'o-', linewidth=2,
                label='Your Technique', color=self.colors['user'])
        ax.fill(angles, normalized_user, alpha=0.25, color=self.colors['user'])

        # Fix axis to go in the right order and start at 12 o'clock
        ax.set_theta_offset(np.pi / 2)
        ax.set_theta_direction(-1)

        # Set labels
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(metric_labels, size=10)

        # Set y-axis limits and labels
        ax.set_ylim(0, 1)
        ax.set_yticks([0.25, 0.5, 0.75, 1.0])
        ax.set_yticklabels(['25%', '50%', '75%', '100%'], size=8)

        # Add legend
        ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1), fontsize=10)

        # Add title
        if title is None:
            title = f'{stroke_type} Technique Analysis'
        plt.title(title, size=16, fontweight='bold', pad=20)

        # Add grid
        ax.grid(True, linestyle='--', alpha=0.5)

        plt.tight_layout()

        # Save if output directory specified
        if self.output_dir:
            save_path = self.output_dir / f'radar_chart_{stroke_type.lower()}.png'
            plt.savefig(save_path, dpi=150, bbox_inches='tight')

        return fig

    def create_metrics_bar_chart(self,
                                 feedback_items: List[FeedbackItem],
                                 stroke_type: str,
                                 title: Optional[str] = None) -> plt.Figure:
        """
        Create bar chart showing metric performance with severity colors

        Args:
            feedback_items: List of feedback items from CoachingFeedback
            stroke_type: 'Clear' or 'Smash'
            title: Optional custom title

        Returns:
            matplotlib Figure object
        """
        # Sort by severity
        severity_order = {'critical': 0, 'major': 1, 'minor': 2, 'good': 3}
        sorted_items = sorted(feedback_items,
                            key=lambda x: severity_order.get(x.severity, 4))

        # Prepare data
        metrics = []
        scores = []
        colors = []

        benchmarks = TechniqueBenchmarks.get_benchmarks(stroke_type)

        for item in sorted_items:
            # Create readable label
            label = item.metric.replace('_', ' ').replace('mean', '').strip()
            label = label.replace('r ', '').title()
            metrics.append(label)

            # Calculate score (0-100) based on distance from optimal
            metric_name = item.metric
            if metric_name in benchmarks or f"{metric_name}_target" in benchmarks:
                min_val, max_val = TechniqueBenchmarks.get_target_range(benchmarks, metric_name)
                range_width = max_val - min_val
                if range_width > 0:
                    # How close to optimal (0-100)
                    optimal = TechniqueBenchmarks.get_target_value(benchmarks, metric_name)
                    distance = abs(item.current_value - optimal)
                    max_distance = range_width / 2
                    score = max(0, 100 - (distance / max_distance * 100))
                else:
                    score = 50
            else:
                score = 50

            scores.append(score)
            colors.append(self.colors.get(item.severity, self.colors['user']))

        # Create figure
        fig, ax = plt.subplots(figsize=(12, 6))

        # Create bars
        bars = ax.barh(metrics, scores, color=colors, alpha=0.8, edgecolor='black')

        # Add value labels
        for i, (bar, score) in enumerate(zip(bars, scores)):
            width = bar.get_width()
            label = f'{score:.0f}%'
            ax.text(width + 2, bar.get_y() + bar.get_height()/2,
                   label, ha='left', va='center', fontsize=9, fontweight='bold')

        # Styling
        ax.set_xlabel('Performance Score (%)', fontsize=12, fontweight='bold')
        ax.set_xlim(0, 110)
        ax.set_xticks([0, 25, 50, 75, 100])
        ax.grid(axis='x', alpha=0.3, linestyle='--')

        # Add reference lines
        ax.axvline(x=85, color=self.colors['good'], linestyle='--',
                  alpha=0.5, linewidth=2, label='Excellent (85%)')
        ax.axvline(x=70, color=self.colors['minor'], linestyle='--',
                  alpha=0.5, linewidth=2, label='Good (70%)')
        ax.axvline(x=50, color=self.colors['major'], linestyle='--',
                  alpha=0.5, linewidth=2, label='Needs Work (50%)')

        # Title
        if title is None:
            title = f'{stroke_type} - Metric Performance Breakdown'
        ax.set_title(title, fontsize=14, fontweight='bold', pad=15)

        # Legend for severity
        severity_patches = [
            mpatches.Patch(color=self.colors['critical'], label='Critical Issue'),
            mpatches.Patch(color=self.colors['major'], label='Major Issue'),
            mpatches.Patch(color=self.colors['minor'], label='Minor Issue'),
            mpatches.Patch(color=self.colors['good'], label='Good Technique')
        ]
        ax.legend(handles=severity_patches, loc='lower right', fontsize=9)

        plt.tight_layout()

        # Save if output directory specified
        if self.output_dir:
            save_path = self.output_dir / f'metrics_bar_chart_{stroke_type.lower()}.png'
            plt.savefig(save_path, dpi=150, bbox_inches='tight')

        return fig

    def create_score_gauge(self,
                          overall_score: int,
                          stroke_type: str,
                          title: Optional[str] = None) -> plt.Figure:
        """
        Create gauge chart showing overall technique score

        Args:
            overall_score: Score from 0-100
            stroke_type: 'Clear' or 'Smash'
            title: Optional custom title

        Returns:
            matplotlib Figure object
        """
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.set_aspect('equal')

        # Define score ranges and colors
        ranges = [
            (0, 50, self.colors['critical'], 'Needs Attention'),
            (50, 70, self.colors['major'], 'Needs Improvement'),
            (70, 85, self.colors['minor'], 'Good'),
            (85, 100, self.colors['good'], 'Excellent')
        ]

        # Draw gauge segments
        start_angle = 180
        total_range = 100

        for min_val, max_val, color, label in ranges:
            range_size = max_val - min_val
            angle_size = (range_size / total_range) * 180

            wedge = Wedge((0.5, 0.3), 0.4, start_angle, start_angle + angle_size,
                         width=0.15, facecolor=color, edgecolor='black',
                         linewidth=2, alpha=0.8)
            ax.add_patch(wedge)
            start_angle += angle_size

        # Draw needle
        needle_angle = 180 - (overall_score / 100 * 180)
        needle_rad = np.radians(needle_angle)

        needle_x = 0.5 + 0.35 * np.cos(needle_rad)
        needle_y = 0.3 + 0.35 * np.sin(needle_rad)

        ax.plot([0.5, needle_x], [0.3, needle_y], 'k-', linewidth=3)
        ax.plot(0.5, 0.3, 'ko', markersize=10)

        # Add score text
        ax.text(0.5, 0.3, f'{overall_score}',
               ha='center', va='center', fontsize=40, fontweight='bold',
               bbox=dict(boxstyle='round,pad=0.3', facecolor='white',
                        edgecolor='black', linewidth=2))

        # Add labels
        ax.text(0.1, 0.32, '0', ha='center', va='center', fontsize=12, fontweight='bold')
        ax.text(0.5, 0.65, '50', ha='center', va='center', fontsize=12, fontweight='bold')
        ax.text(0.9, 0.32, '100', ha='center', va='center', fontsize=12, fontweight='bold')

        # Determine grade
        if overall_score >= 85:
            grade = 'Excellent'
            grade_color = self.colors['good']
        elif overall_score >= 70:
            grade = 'Good'
            grade_color = self.colors['minor']
        elif overall_score >= 50:
            grade = 'Needs Improvement'
            grade_color = self.colors['major']
        else:
            grade = 'Needs Attention'
            grade_color = self.colors['critical']

        # Add grade text
        ax.text(0.5, 0.05, grade, ha='center', va='center',
               fontsize=20, fontweight='bold', color=grade_color)

        # Set limits and remove axes
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 0.7)
        ax.axis('off')

        # Title
        if title is None:
            title = f'{stroke_type} - Overall Technique Score'
        ax.set_title(title, fontsize=16, fontweight='bold', pad=20)

        plt.tight_layout()

        # Save if output directory specified
        if self.output_dir:
            save_path = self.output_dir / f'score_gauge_{stroke_type.lower()}.png'
            plt.savefig(save_path, dpi=150, bbox_inches='tight')

        return fig

    def create_comprehensive_report(self,
                                   features: Dict,
                                   stroke_type: str,
                                   coach: CoachingFeedback) -> plt.Figure:
        """
        Create comprehensive 3-panel report with all visualizations

        Args:
            features: User's biomechanical features
            stroke_type: 'Clear' or 'Smash'
            coach: CoachingFeedback object with analyzed results

        Returns:
            matplotlib Figure object with 3 subplots
        """
        fig = plt.figure(figsize=(18, 6))

        # Panel 1: Radar chart
        ax1 = plt.subplot(131, projection='polar')
        self._draw_radar_on_axis(ax1, features, stroke_type)

        # Panel 2: Bar chart
        ax2 = plt.subplot(132)
        self._draw_bars_on_axis(ax2, coach.feedback_items, stroke_type)

        # Panel 3: Score gauge
        ax3 = plt.subplot(133)
        self._draw_gauge_on_axis(ax3, coach.overall_score, stroke_type)

        # Overall title
        fig.suptitle(f'{stroke_type} Stroke - Complete Technique Analysis',
                    fontsize=18, fontweight='bold', y=1.02)

        plt.tight_layout()

        # Save if output directory specified
        if self.output_dir:
            save_path = self.output_dir / f'comprehensive_report_{stroke_type.lower()}.png'
            plt.savefig(save_path, dpi=150, bbox_inches='tight')

        return fig

    def _draw_radar_on_axis(self, ax, features: Dict, stroke_type: str):
        """Helper to draw radar chart on existing axis"""
        benchmarks = TechniqueBenchmarks.get_benchmarks(stroke_type)

        metrics = []
        user_values = []
        optimal_values = []
        metric_labels = []

        for metric in benchmarks.keys():
            if metric in features:
                metrics.append(metric)
                user_values.append(features[metric])
                optimal = TechniqueBenchmarks.get_target_value(benchmarks, metric)
                optimal_values.append(optimal)

                label = metric.replace('_', ' ').replace('mean', '').strip()
                label = label.replace('r ', '').title()
                metric_labels.append(label)

        normalized_user = []
        normalized_optimal = []

        for i, metric in enumerate(metrics):
            min_val, max_val = TechniqueBenchmarks.get_target_range(benchmarks, metric)
            range_width = max_val - min_val

            if range_width > 0:
                user_norm = (user_values[i] - min_val) / range_width
                optimal_norm = (optimal_values[i] - min_val) / range_width
                user_norm = max(0, min(1, user_norm))
                optimal_norm = max(0, min(1, optimal_norm))
            else:
                user_norm = 0.5
                optimal_norm = 0.5

            normalized_user.append(user_norm)
            normalized_optimal.append(optimal_norm)

        num_vars = len(metrics)
        angles = np.linspace(0, 2 * np.pi, num_vars, endpoint=False).tolist()

        normalized_user += normalized_user[:1]
        normalized_optimal += normalized_optimal[:1]
        angles += angles[:1]

        ax.plot(angles, normalized_optimal, 'o-', linewidth=2,
                label='Professional', color=self.colors['optimal'])
        ax.fill(angles, normalized_optimal, alpha=0.15, color=self.colors['optimal'])

        ax.plot(angles, normalized_user, 'o-', linewidth=2,
                label='You', color=self.colors['user'])
        ax.fill(angles, normalized_user, alpha=0.25, color=self.colors['user'])

        ax.set_theta_offset(np.pi / 2)
        ax.set_theta_direction(-1)
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(metric_labels, size=8)
        ax.set_ylim(0, 1)
        ax.set_yticks([0.5, 1.0])
        ax.set_yticklabels(['50%', '100%'], size=7)
        ax.legend(loc='upper right', bbox_to_anchor=(1.15, 1.1), fontsize=8)
        ax.grid(True, linestyle='--', alpha=0.5)
        ax.set_title('Technique Radar', size=12, fontweight='bold', pad=10)

    def _draw_bars_on_axis(self, ax, feedback_items: List[FeedbackItem], stroke_type: str):
        """Helper to draw bar chart on existing axis"""
        severity_order = {'critical': 0, 'major': 1, 'minor': 2, 'good': 3}
        sorted_items = sorted(feedback_items,
                            key=lambda x: severity_order.get(x.severity, 4))

        metrics = []
        scores = []
        colors = []

        benchmarks = TechniqueBenchmarks.get_benchmarks(stroke_type)

        for item in sorted_items:
            label = item.metric.replace('_', ' ').replace('mean', '').strip()
            label = label.replace('r ', '').title()
            if len(label) > 20:
                label = label[:17] + '...'
            metrics.append(label)

            metric_name = item.metric
            if metric_name in benchmarks or f"{metric_name}_target" in benchmarks:
                min_val, max_val = TechniqueBenchmarks.get_target_range(benchmarks, metric_name)
                range_width = max_val - min_val
                if range_width > 0:
                    optimal = TechniqueBenchmarks.get_target_value(benchmarks, metric_name)
                    distance = abs(item.current_value - optimal)
                    max_distance = range_width / 2
                    score = max(0, 100 - (distance / max_distance * 100))
                else:
                    score = 50
            else:
                score = 50

            scores.append(score)
            colors.append(self.colors.get(item.severity, self.colors['user']))

        bars = ax.barh(metrics, scores, color=colors, alpha=0.8, edgecolor='black')

        for bar, score in zip(bars, scores):
            width = bar.get_width()
            ax.text(width + 2, bar.get_y() + bar.get_height()/2,
                   f'{score:.0f}%', ha='left', va='center', fontsize=7)

        ax.set_xlabel('Score (%)', fontsize=10)
        ax.set_xlim(0, 110)
        ax.grid(axis='x', alpha=0.3)
        ax.set_title('Metric Breakdown', size=12, fontweight='bold', pad=10)

    def _draw_gauge_on_axis(self, ax, overall_score: int, stroke_type: str):
        """Helper to draw gauge on existing axis"""
        ax.set_aspect('equal')

        ranges = [
            (0, 50, self.colors['critical']),
            (50, 70, self.colors['major']),
            (70, 85, self.colors['minor']),
            (85, 100, self.colors['good'])
        ]

        start_angle = 180
        for min_val, max_val, color in ranges:
            range_size = max_val - min_val
            angle_size = (range_size / 100) * 180
            wedge = Wedge((0.5, 0.3), 0.4, start_angle, start_angle + angle_size,
                         width=0.15, facecolor=color, edgecolor='black',
                         linewidth=1.5, alpha=0.8)
            ax.add_patch(wedge)
            start_angle += angle_size

        needle_angle = 180 - (overall_score / 100 * 180)
        needle_rad = np.radians(needle_angle)
        needle_x = 0.5 + 0.35 * np.cos(needle_rad)
        needle_y = 0.3 + 0.35 * np.sin(needle_rad)

        ax.plot([0.5, needle_x], [0.3, needle_y], 'k-', linewidth=2)
        ax.plot(0.5, 0.3, 'ko', markersize=8)

        ax.text(0.5, 0.3, f'{overall_score}',
               ha='center', va='center', fontsize=30, fontweight='bold',
               bbox=dict(boxstyle='round,pad=0.2', facecolor='white',
                        edgecolor='black', linewidth=1.5))

        ax.set_xlim(0, 1)
        ax.set_ylim(0, 0.7)
        ax.axis('off')
        ax.set_title('Overall Score', size=12, fontweight='bold', pad=10)


# Example usage and testing
if __name__ == "__main__":
    import pickle
    from pathlib import Path

    print("="*70)
    print("VISUALIZATION TEST")
    print("="*70)
    print()

    # Load sample feature file
    features_dir = Path('data/processed/features')
    feature_files = list(features_dir.glob('*Clear_features.pkl'))

    if feature_files:
        print(f"Testing with: {feature_files[0].name}\n")

        with open(feature_files[0], 'rb') as f:
            data = pickle.load(f)

        features = data['statistical_summary']

        # Generate feedback
        coach = CoachingFeedback()
        feedback_items = coach.analyze_technique(features, 'Clear')

        # Create visualizer
        viz = TechniqueVisualizer(output_dir=Path('outputs/visualizations'))

        # Test 1: Radar chart
        print("Creating radar chart...")
        fig1 = viz.create_radar_chart(features, 'Clear')
        print("✓ Radar chart created")

        # Test 2: Bar chart
        print("Creating bar chart...")
        fig2 = viz.create_metrics_bar_chart(feedback_items, 'Clear')
        print("✓ Bar chart created")

        # Test 3: Score gauge
        print("Creating score gauge...")
        fig3 = viz.create_score_gauge(coach.overall_score, 'Clear')
        print("✓ Score gauge created")

        # Test 4: Comprehensive report
        print("Creating comprehensive report...")
        fig4 = viz.create_comprehensive_report(features, 'Clear', coach)
        print("✓ Comprehensive report created")

        print()
        print("✓ All visualizations created successfully!")
        print(f"Saved to: outputs/visualizations/")

        # Show plots (comment out if running headless)
        # plt.show()

    else:
        print("No feature files found!")
