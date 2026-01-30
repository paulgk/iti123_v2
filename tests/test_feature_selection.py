#!/usr/bin/env python3
"""
Tests for feature selection pipeline.

Tests cover:
- Cohen's d calculation with known values
- VIF iterative removal of multicollinear features
- Pipeline feature count constraints
- Edge case handling (zero effect, empty features)
"""

import pytest
import numpy as np
import pandas as pd
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data_processing.feature_selection import (
    calculate_cohens_d,
    filter_by_effect_size,
    filter_by_vif,
    filter_zero_variance,
    rfecv_selection,
    feature_selection_pipeline
)


class TestCohensD:
    """Tests for Cohen's d effect size calculation"""

    def test_cohens_d_large_effect(self):
        """Test Cohen's d with known large separation"""
        # Two groups with large separation
        np.random.seed(42)
        n = 50
        values = np.concatenate([
            np.random.normal(0, 1, n),  # Class 0: mean=0
            np.random.normal(3, 1, n)   # Class 1: mean=3, d~3 (large)
        ])
        labels = np.array([0]*n + [1]*n)

        result = calculate_cohens_d(values, labels)

        assert result['interpretation'] == 'large'
        assert result['abs_d'] > 0.8
        assert result['meets_criteria'] is True

    def test_cohens_d_medium_effect(self):
        """Test Cohen's d with medium effect (threshold boundary)"""
        np.random.seed(42)
        n = 50
        values = np.concatenate([
            np.random.normal(0, 1, n),    # Class 0
            np.random.normal(0.6, 1, n)   # Class 1: d~0.6 (medium)
        ])
        labels = np.array([0]*n + [1]*n)

        result = calculate_cohens_d(values, labels)

        assert result['interpretation'] == 'medium'
        assert 0.5 <= result['abs_d'] < 0.8
        assert result['meets_criteria'] is True

    def test_cohens_d_small_effect(self):
        """Test Cohen's d with small effect (below threshold)"""
        np.random.seed(42)
        n = 50
        values = np.concatenate([
            np.random.normal(0, 1, n),    # Class 0
            np.random.normal(0.3, 1, n)   # Class 1: d~0.3 (small)
        ])
        labels = np.array([0]*n + [1]*n)

        result = calculate_cohens_d(values, labels)

        assert result['interpretation'] == 'small'
        assert 0.2 <= result['abs_d'] < 0.5
        assert result['meets_criteria'] is False

    def test_cohens_d_negligible_effect(self):
        """Test Cohen's d with negligible effect"""
        np.random.seed(42)
        n = 50
        values = np.concatenate([
            np.random.normal(0, 1, n),     # Class 0
            np.random.normal(0.1, 1, n)    # Class 1: d~0.1 (negligible)
        ])
        labels = np.array([0]*n + [1]*n)

        result = calculate_cohens_d(values, labels)

        assert result['interpretation'] == 'negligible'
        assert result['abs_d'] < 0.2
        assert result['meets_criteria'] is False

    def test_cohens_d_interpretation_thresholds(self):
        """Verify interpretation threshold boundaries"""
        # Test exact threshold values
        labels = np.array([0, 0, 1, 1])

        # d = 0 (no difference)
        values_zero = np.array([1.0, 1.0, 1.0, 1.0])
        result = calculate_cohens_d(values_zero, labels)
        assert result['interpretation'] == 'negligible'

        # d ~ 0.2 (boundary between negligible and small)
        # mean_diff = 0.2, std = 1 → d = 0.2
        values_small = np.array([0.0, 0.0, 0.2, 0.2])
        result = calculate_cohens_d(values_small, labels)
        assert result['interpretation'] in ['negligible', 'small']

        # d ~ 0.5 (boundary between small and medium)
        values_medium = np.array([0.0, 0.0, 0.5, 0.5])
        result = calculate_cohens_d(values_medium, labels)
        assert result['interpretation'] in ['small', 'medium']


class TestVIFFiltering:
    """Tests for VIF-based multicollinearity removal"""

    def test_vif_removes_perfect_correlation(self):
        """Test VIF removes perfectly correlated features"""
        # Create perfectly correlated features
        np.random.seed(42)
        n = 100
        X = pd.DataFrame({
            'a': np.random.randn(n),
            'b': np.random.randn(n),  # Independent
            'c': np.random.randn(n)   # Independent
        })
        # Add perfect correlation: d = 2*a
        X['d'] = X['a'] * 2

        selected = filter_by_vif(X, threshold=10)

        # Should remove one of a/d due to perfect multicollinearity
        assert len(selected) < len(X.columns)
        # Either 'a' or 'd' should be removed (but not both)
        assert ('a' in selected) != ('d' in selected)

    def test_vif_keeps_independent_features(self):
        """Test VIF keeps independent features"""
        np.random.seed(42)
        n = 100
        X = pd.DataFrame({
            'a': np.random.randn(n),
            'b': np.random.randn(n),
            'c': np.random.randn(n)
        })

        selected = filter_by_vif(X, threshold=10)

        # All independent features should be kept
        assert len(selected) == len(X.columns)

    def test_vif_iterative_removal(self):
        """Test VIF iteratively removes highest VIF until threshold met"""
        np.random.seed(42)
        n = 100
        # Create multiple correlated features
        base = np.random.randn(n)
        X = pd.DataFrame({
            'base': base,
            'corr1': base + np.random.randn(n) * 0.1,  # High correlation
            'corr2': base + np.random.randn(n) * 0.1,  # High correlation
            'indep': np.random.randn(n)                # Independent
        })

        selected = filter_by_vif(X, threshold=10)

        # Should keep independent feature
        assert 'indep' in selected
        # Should remove some correlated features
        assert len(selected) < len(X.columns)


class TestZeroVarianceFilter:
    """Tests for zero variance filtering"""

    def test_removes_constant_features(self):
        """Test removal of constant features"""
        X = pd.DataFrame({
            'constant': [1.0, 1.0, 1.0, 1.0],
            'variable': [1.0, 2.0, 3.0, 4.0]
        })

        selected = filter_zero_variance(X)

        assert 'constant' not in selected
        assert 'variable' in selected
        assert len(selected) == 1

    def test_keeps_varying_features(self):
        """Test keeping features with variance"""
        X = pd.DataFrame({
            'a': [1.0, 2.0, 3.0],
            'b': [4.0, 5.0, 6.0]
        })

        selected = filter_zero_variance(X)

        assert len(selected) == 2


class TestEffectSizeFiltering:
    """Tests for effect size filtering"""

    def test_filter_by_effect_size(self):
        """Test filtering by Cohen's d threshold"""
        np.random.seed(42)
        n = 100

        # Create features with different effect sizes
        X = pd.DataFrame({
            'large_effect': np.concatenate([
                np.random.normal(0, 1, n),
                np.random.normal(2, 1, n)  # d ~ 2
            ]),
            'medium_effect': np.concatenate([
                np.random.normal(0, 1, n),
                np.random.normal(0.6, 1, n)  # d ~ 0.6
            ]),
            'small_effect': np.concatenate([
                np.random.normal(0, 1, n),
                np.random.normal(0.3, 1, n)  # d ~ 0.3
            ])
        })
        y = np.array([0]*n + [1]*n)

        selected = filter_by_effect_size(X, y, threshold=0.5)

        # Should keep large and medium, remove small
        assert 'large_effect' in selected
        assert 'medium_effect' in selected
        assert 'small_effect' not in selected

    def test_filter_raises_if_all_removed(self):
        """Test error when all features filtered out"""
        # Features with no discriminative power
        X = pd.DataFrame({
            'a': [1, 2, 3, 4, 1, 2, 3, 4],
            'b': [5, 6, 7, 8, 5, 6, 7, 8]
        })
        y = np.array([0, 0, 0, 0, 1, 1, 1, 1])

        with pytest.raises(ValueError, match="All features removed"):
            filter_by_effect_size(X, y, threshold=0.5)


class TestPipeline:
    """Tests for full feature selection pipeline"""

    def test_pipeline_respects_target_features(self):
        """Verify pipeline respects target_features constraint"""
        np.random.seed(42)
        n_samples = 200
        n_features = 60

        # Create features with varying discriminative power
        X = pd.DataFrame()
        y = np.array([0]*(n_samples//2) + [1]*(n_samples//2))

        # Add features with medium-large effects (will pass Cohen's d filter)
        for i in range(n_features):
            effect = np.random.uniform(0.5, 1.5)  # Medium to large effect
            X[f'feat_{i}'] = np.concatenate([
                np.random.normal(0, 1, n_samples//2),
                np.random.normal(effect, 1, n_samples//2)
            ])

        target = 20
        result = feature_selection_pipeline(X, y, target_features=target)

        assert len(result['selected_features']) <= target

    def test_pipeline_stage_progression(self):
        """Verify pipeline shows decreasing feature counts through stages"""
        np.random.seed(42)
        n_samples = 150
        n_features = 40

        X = pd.DataFrame()
        y = np.array([0]*(n_samples//2) + [1]*(n_samples//2))

        # Mix of good and poor features
        for i in range(n_features):
            if i < 20:
                # Good features (medium effect)
                effect = np.random.uniform(0.5, 1.0)
            else:
                # Poor features (small effect)
                effect = np.random.uniform(0.1, 0.4)

            X[f'feat_{i}'] = np.concatenate([
                np.random.normal(0, 1, n_samples//2),
                np.random.normal(effect, 1, n_samples//2)
            ])

        result = feature_selection_pipeline(X, y, target_features=15)
        stages = result['stages']

        # Each stage should reduce or maintain count
        assert stages['after_zero_var'] <= stages['initial']
        assert stages['after_cohens_d'] <= stages['after_zero_var']
        assert stages['after_vif'] <= stages['after_cohens_d']
        assert stages['after_rfecv'] <= stages['after_vif']

    def test_pipeline_returns_required_keys(self):
        """Test pipeline returns all required result keys"""
        np.random.seed(42)
        n_samples = 100
        n_features = 30

        X = pd.DataFrame()
        y = np.array([0]*(n_samples//2) + [1]*(n_samples//2))

        for i in range(n_features):
            X[f'feat_{i}'] = np.concatenate([
                np.random.normal(0, 1, n_samples//2),
                np.random.normal(0.6, 1, n_samples//2)
            ])

        result = feature_selection_pipeline(X, y, target_features=10)

        # Check all required keys present
        assert 'selected_features' in result
        assert 'stages' in result
        assert 'effect_sizes' in result
        assert 'vif_removed' in result
        assert 'rfecv_ranking' in result
        assert 'cv_f1_score' in result

        # Check types
        assert isinstance(result['selected_features'], list)
        assert isinstance(result['stages'], dict)
        assert isinstance(result['cv_f1_score'], float)


class TestRFECV:
    """Tests for RFECV wrapper method"""

    def test_rfecv_selection(self):
        """Test RFECV selects features"""
        np.random.seed(42)
        n_samples = 150
        n_features = 30

        X = pd.DataFrame()
        y = np.array([0]*(n_samples//2) + [1]*(n_samples//2))

        for i in range(n_features):
            X[f'feat_{i}'] = np.concatenate([
                np.random.normal(0, 1, n_samples//2),
                np.random.normal(0.7, 1, n_samples//2)
            ])

        result = rfecv_selection(X, y, max_features=15)

        assert len(result['selected_features']) <= 15
        assert 'feature_ranking' in result
        assert 'best_cv_f1' in result
        assert 0.0 <= result['best_cv_f1'] <= 1.0


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
