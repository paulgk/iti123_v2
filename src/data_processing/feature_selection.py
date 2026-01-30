#!/usr/bin/env python3
"""
Feature Selection Pipeline for Feature Engineering V3

Two-stage feature selection to reduce feature count from ~658 to <254:
1. Filter methods: Cohen's d effect size, VIF multicollinearity, zero variance
2. Wrapper methods: RFECV with cross-validation

This module implements FEAT-08, FEAT-09, FEAT-10 requirements.

References:
- Cohen's d interpretation (biomechanics): d < 0.2 negligible, 0.2-0.5 small,
  0.5-0.8 medium (threshold), d >= 0.8 large
- VIF threshold: VIF < 10 acceptable, VIF >= 10 indicates high multicollinearity
- N/10 rule: With 3,347 samples and N_train ~2,554, need <254 features
"""

import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Tuple
import json
from datetime import datetime

from sklearn.feature_selection import RFECV
from sklearn.ensemble import RandomForestClassifier
from statsmodels.stats.outliers_influence import variance_inflation_factor


def calculate_cohens_d(feature_values: np.ndarray, labels: np.ndarray) -> Dict:
    """
    Calculate Cohen's d effect size for binary classification.

    Cohen's d measures the standardized difference between two group means.
    Interpretation for biomechanics research:
    - d < 0.2: negligible effect
    - 0.2 <= d < 0.5: small effect
    - 0.5 <= d < 0.8: medium effect (SUCCESS CRITERIA threshold)
    - d >= 0.8: large effect

    Args:
        feature_values: Feature values (n_samples,)
        labels: Binary class labels (n_samples,) - 0 or 1

    Returns:
        dict with:
            - cohens_d: signed Cohen's d value
            - abs_d: absolute Cohen's d value
            - interpretation: effect size category
            - meets_criteria: True if abs_d >= 0.5
    """
    class_0 = feature_values[labels == 0]
    class_1 = feature_values[labels == 1]

    n0, n1 = len(class_0), len(class_1)

    if n0 < 2 or n1 < 2:
        return {
            'cohens_d': 0.0,
            'abs_d': 0.0,
            'interpretation': 'insufficient_data',
            'meets_criteria': False
        }

    # Pooled standard deviation
    var0 = np.var(class_0, ddof=1)
    var1 = np.var(class_1, ddof=1)
    pooled_std = np.sqrt(((n0 - 1) * var0 + (n1 - 1) * var1) / (n0 + n1 - 2))

    # Cohen's d
    mean_diff = np.mean(class_1) - np.mean(class_0)
    cohens_d = mean_diff / (pooled_std + 1e-8)

    # Interpretation
    abs_d = abs(cohens_d)
    if abs_d < 0.2:
        interpretation = 'negligible'
    elif abs_d < 0.5:
        interpretation = 'small'
    elif abs_d < 0.8:
        interpretation = 'medium'
    else:
        interpretation = 'large'

    return {
        'cohens_d': cohens_d,
        'abs_d': abs_d,
        'interpretation': interpretation,
        'meets_criteria': abs_d >= 0.5
    }


def filter_by_effect_size(X: pd.DataFrame, y: np.ndarray, threshold: float = 0.5) -> List[str]:
    """
    Filter features by Cohen's d effect size.

    Args:
        X: Feature matrix (n_samples, n_features)
        y: Binary class labels (n_samples,)
        threshold: Cohen's d threshold (default 0.5 for medium effect)

    Returns:
        List of feature names meeting threshold

    Raises:
        ValueError: If all features filtered out
    """
    effect_sizes = {}

    for col in X.columns:
        result = calculate_cohens_d(X[col].values, y)
        effect_sizes[col] = result

    # Select features meeting threshold
    selected = [col for col, result in effect_sizes.items()
                if result['abs_d'] >= threshold]

    # Log distribution
    interpretations = [r['interpretation'] for r in effect_sizes.values()]
    print(f"\nEffect size filtering (threshold={threshold}):")
    print(f"  Input features: {len(X.columns)}")
    print(f"  Selected features: {len(selected)}")
    print(f"\nEffect size distribution:")
    print(f"  Negligible (d<0.2): {interpretations.count('negligible')}")
    print(f"  Small (0.2≤d<0.5): {interpretations.count('small')}")
    print(f"  Medium (0.5≤d<0.8): {interpretations.count('medium')}")
    print(f"  Large (d≥0.8): {interpretations.count('large')}")

    if len(selected) == 0:
        raise ValueError(
            f"All features removed by Cohen's d filter (threshold={threshold}). "
            f"Consider lowering threshold."
        )

    return selected


def calculate_vif(X: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate Variance Inflation Factor for each feature.

    VIF measures multicollinearity:
    - VIF < 5: low multicollinearity (ideal)
    - 5 <= VIF < 10: moderate multicollinearity (acceptable)
    - VIF >= 10: high multicollinearity (remove feature)

    Args:
        X: Feature matrix (n_samples, n_features)

    Returns:
        DataFrame with columns ['feature', 'VIF']
    """
    vif_data = pd.DataFrame()
    vif_data["feature"] = X.columns
    vif_data["VIF"] = [variance_inflation_factor(X.values, i)
                       for i in range(len(X.columns))]
    return vif_data


def filter_by_vif(X: pd.DataFrame, threshold: float = 10) -> List[str]:
    """
    Iteratively remove features with highest VIF until all VIF < threshold.

    Multicollinearity causes unstable feature selection (wrapper methods
    see correlated features as interchangeable). VIF filtering prevents this.

    Args:
        X: Feature matrix (n_samples, n_features)
        threshold: VIF threshold (default 10, research-validated)

    Returns:
        List of feature names with VIF < threshold
    """
    X_vif = X.copy()
    removed_features = []

    print(f"\nVIF-based multicollinearity filtering (threshold={threshold}):")
    print(f"  Initial features: {len(X_vif.columns)}")

    iteration = 0
    while True:
        vif_data = calculate_vif(X_vif)
        max_vif = vif_data["VIF"].max()

        if max_vif < threshold:
            break

        # Remove feature with highest VIF
        worst_idx = vif_data["VIF"].idxmax()
        worst_feature = vif_data.loc[worst_idx, "feature"]
        worst_vif = vif_data.loc[worst_idx, "VIF"]

        removed_features.append((worst_feature, worst_vif))
        X_vif = X_vif.drop(columns=[worst_feature])

        iteration += 1
        if iteration % 10 == 0:
            print(f"  Iteration {iteration}: {len(X_vif.columns)} features remaining")

    print(f"  Final features: {len(X_vif.columns)}")
    print(f"  Removed {len(removed_features)} features")

    if len(removed_features) > 0 and len(removed_features) <= 10:
        print("\nRemoved features (VIF):")
        for feat, vif in removed_features[:10]:
            print(f"    {feat}: VIF={vif:.2f}")

    return X_vif.columns.tolist()


def filter_zero_variance(X: pd.DataFrame, threshold: float = 1e-8) -> List[str]:
    """
    Remove features with near-zero variance.

    Features with zero variance provide no information for discrimination.

    Args:
        X: Feature matrix (n_samples, n_features)
        threshold: Variance threshold (default 1e-8)

    Returns:
        List of feature names with variance > threshold
    """
    variances = X.var()
    non_zero_var = variances[variances > threshold].index.tolist()

    removed = len(X.columns) - len(non_zero_var)
    if removed > 0:
        print(f"\nZero-variance filtering:")
        print(f"  Removed {removed} features with variance < {threshold}")

    return non_zero_var


def rfecv_selection(X: pd.DataFrame, y: np.ndarray, max_features: int = 254) -> Dict:
    """
    Recursive Feature Elimination with Cross-Validation.

    Uses Random Forest as base estimator with 5-fold CV and F1 scoring.
    If RFECV selects > max_features, takes top max_features by ranking.

    Args:
        X: Feature matrix (n_samples, n_features)
        y: Binary class labels (n_samples,)
        max_features: Maximum features to select (default 254, N_train/10)

    Returns:
        dict with:
            - selected_features: list of feature names
            - n_features_optimal: optimal feature count from CV
            - cv_scores: cross-validation scores per feature count
            - feature_ranking: dict mapping feature -> rank
    """
    print(f"\nRFECV feature selection:")
    print(f"  Input features: {len(X.columns)}")
    print(f"  Target max features: {max_features}")

    # Random Forest with regularization to prevent overfitting
    estimator = RandomForestClassifier(
        n_estimators=100,
        max_depth=10,
        min_samples_split=10,
        random_state=42,
        n_jobs=-1
    )

    # RFECV with 5-fold CV, F1 scoring
    min_features = max(10, int(max_features * 0.5))
    selector = RFECV(
        estimator=estimator,
        step=1,
        cv=5,
        scoring='f1',
        min_features_to_select=min_features,
        n_jobs=-1
    )

    selector.fit(X, y)

    # Get selected features
    selected_mask = selector.support_
    n_selected = selector.n_features_

    # Feature ranking (1 = best)
    feature_ranking = dict(zip(X.columns, selector.ranking_))

    # If selected > max_features, take top max_features by ranking
    if n_selected > max_features:
        print(f"  RFECV selected {n_selected} features, limiting to {max_features}")
        # Sort by ranking (lower = better)
        sorted_features = sorted(feature_ranking.items(), key=lambda x: x[1])
        selected_features = [feat for feat, rank in sorted_features[:max_features]]
    else:
        selected_features = X.columns[selected_mask].tolist()

    # Best CV score
    best_score = selector.cv_results_['mean_test_score'].max()

    print(f"  Selected features: {len(selected_features)}")
    print(f"  Best CV F1 score: {best_score:.4f}")

    return {
        'selected_features': selected_features,
        'n_features_optimal': n_selected,
        'cv_scores': selector.cv_results_['mean_test_score'],
        'feature_ranking': feature_ranking,
        'best_cv_f1': best_score
    }


def feature_selection_pipeline(X: pd.DataFrame, y: np.ndarray,
                               target_features: int = 254) -> Dict:
    """
    Full two-stage feature selection pipeline.

    Stage 1 (Filter methods - fast, prevents overfitting):
        a. Remove zero-variance features
        b. Filter by Cohen's d >= 0.5 (medium effect size)
        c. Filter by VIF < 10 (remove multicollinearity)

    Stage 2 (Wrapper method - model-specific optimization):
        d. RFECV with Random Forest, 5-fold CV, F1 scoring
        e. Limit to target_features if needed

    Args:
        X: Feature matrix (n_samples, n_features)
        y: Binary class labels (n_samples,)
        target_features: Target feature count (default 254, N_train/10 rule)

    Returns:
        dict with:
            - selected_features: final selected feature names
            - stages: dict with feature counts at each stage
            - effect_sizes: dict mapping feature -> Cohen's d result
            - vif_removed: list of features removed by VIF
            - rfecv_ranking: dict mapping feature -> RFECV rank
            - cv_f1_score: best cross-validation F1 score
    """
    print("\n" + "="*80)
    print("FEATURE SELECTION PIPELINE")
    print("="*80)
    print(f"\nDataset: {len(y)} samples, {len(X.columns)} features")
    print(f"Target: <{target_features} features (N_train/10 rule)")

    stage_counts = {'initial': len(X.columns)}

    # === STAGE 1: FILTER METHODS ===
    print("\n" + "-"*80)
    print("STAGE 1: FILTER METHODS")
    print("-"*80)

    # 1a. Zero variance removal
    non_zero_features = filter_zero_variance(X)
    X_filtered = X[non_zero_features]
    stage_counts['after_zero_var'] = len(X_filtered.columns)

    # 1b. Cohen's d effect size filtering
    effect_size_results = {}
    for col in X_filtered.columns:
        effect_size_results[col] = calculate_cohens_d(X_filtered[col].values, y)

    high_effect = [col for col, result in effect_size_results.items()
                   if result['abs_d'] >= 0.5]
    X_filtered = X_filtered[high_effect]
    stage_counts['after_cohens_d'] = len(X_filtered.columns)

    # Log effect size filtering
    interpretations = [r['interpretation'] for r in effect_size_results.values()]
    print(f"\nCohen's d filtering (threshold=0.5):")
    print(f"  Input: {len(effect_size_results)} features")
    print(f"  Selected: {len(high_effect)} features")
    print(f"  Distribution: {interpretations.count('large')} large, "
          f"{interpretations.count('medium')} medium, "
          f"{interpretations.count('small')} small, "
          f"{interpretations.count('negligible')} negligible")

    # 1c. VIF-based multicollinearity removal
    vif_features = filter_by_vif(X_filtered, threshold=10)
    vif_removed = [f for f in X_filtered.columns if f not in vif_features]
    X_filtered = X_filtered[vif_features]
    stage_counts['after_vif'] = len(X_filtered.columns)

    # === STAGE 2: WRAPPER METHOD ===
    print("\n" + "-"*80)
    print("STAGE 2: WRAPPER METHOD (RFECV)")
    print("-"*80)

    rfecv_result = rfecv_selection(X_filtered, y, max_features=target_features)
    stage_counts['after_rfecv'] = len(rfecv_result['selected_features'])

    # === SUMMARY ===
    print("\n" + "="*80)
    print("PIPELINE SUMMARY")
    print("="*80)
    print(f"\nStage progression:")
    for stage, count in stage_counts.items():
        reduction = stage_counts['initial'] - count if stage != 'initial' else 0
        print(f"  {stage:20s}: {count:4d} features (-{reduction})")

    print(f"\nFinal: {len(rfecv_result['selected_features'])} features selected")
    print(f"Target met: {len(rfecv_result['selected_features']) <= target_features}")
    print(f"Best CV F1: {rfecv_result['best_cv_f1']:.4f}")

    return {
        'selected_features': rfecv_result['selected_features'],
        'stages': stage_counts,
        'effect_sizes': effect_size_results,
        'vif_removed': vif_removed,
        'rfecv_ranking': rfecv_result['feature_ranking'],
        'cv_f1_score': rfecv_result['best_cv_f1']
    }


def save_feature_selection_mask(selected_features: List[str],
                                output_path: str = None) -> None:
    """
    Save selected feature names to JSON for use by feature_engineering_v3.

    Args:
        selected_features: List of selected feature names
        output_path: Output path (default: data/processed/features_v3/selected_features.json)
    """
    if output_path is None:
        base_dir = Path(__file__).resolve().parents[2]
        output_dir = base_dir / "data" / "processed" / "features_v3"
        output_dir.mkdir(parents=True, exist_ok=True)
        output_path = output_dir / "selected_features.json"

    mask = {
        'version': 'v3',
        'selected_features': selected_features,
        'selection_date': datetime.now().strftime('%Y-%m-%d'),
        'n_features': len(selected_features),
        'selection_method': 'filter_then_wrapper'
    }

    with open(output_path, 'w') as f:
        json.dump(mask, f, indent=2)

    print(f"\nFeature selection mask saved to: {output_path}")


def generate_report(pipeline_results: Dict,
                   output_path: str = 'outputs/reports/feature_selection_report.md') -> None:
    """
    Generate markdown report from pipeline results.

    Args:
        pipeline_results: Results from feature_selection_pipeline()
        output_path: Output path for report
    """
    # Ensure output directory exists
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)

    stages = pipeline_results['stages']
    effect_sizes = pipeline_results['effect_sizes']
    selected = pipeline_results['selected_features']

    # Calculate effect size distribution
    interpretations = [r['interpretation'] for r in effect_sizes.values()]
    dist_counts = {
        'negligible': interpretations.count('negligible'),
        'small': interpretations.count('small'),
        'medium': interpretations.count('medium'),
        'large': interpretations.count('large')
    }

    # Top features by effect size
    sorted_by_effect = sorted(effect_sizes.items(),
                             key=lambda x: x[1]['abs_d'],
                             reverse=True)

    # Generate report
    report = f"""# Feature Selection Report

**Status:** Analysis complete
**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

## Overview

This report documents the feature selection process for v3 feature engineering.

## Selection Pipeline Summary

| Stage | Features | Reduction |
|-------|----------|-----------|
| Initial | {stages['initial']} | - |
| After zero-variance removal | {stages['after_zero_var']} | -{stages['initial'] - stages['after_zero_var']} |
| After Cohen's d >= 0.5 | {stages['after_cohens_d']} | -{stages['after_zero_var'] - stages['after_cohens_d']} |
| After VIF < 10 | {stages['after_vif']} | -{stages['after_cohens_d'] - stages['after_vif']} |
| After RFECV (final) | {stages['after_rfecv']} | -{stages['after_vif'] - stages['after_rfecv']} |

## Effect Size Distribution

| Category | Count | Percentage |
|----------|-------|------------|
| Negligible (d < 0.2) | {dist_counts['negligible']} | {dist_counts['negligible']/len(effect_sizes)*100:.1f}% |
| Small (0.2 <= d < 0.5) | {dist_counts['small']} | {dist_counts['small']/len(effect_sizes)*100:.1f}% |
| Medium (0.5 <= d < 0.8) | {dist_counts['medium']} | {dist_counts['medium']/len(effect_sizes)*100:.1f}% |
| Large (d >= 0.8) | {dist_counts['large']} | {dist_counts['large']/len(effect_sizes)*100:.1f}% |

## Top 20 Features by Effect Size

| Rank | Feature | Cohen's d | Category |
|------|---------|-----------|----------|
"""

    for i, (feat, result) in enumerate(sorted_by_effect[:20], 1):
        report += f"| {i} | {feat} | {result['abs_d']:.3f} | {result['interpretation']} |\n"

    report += f"""
## Features Removed by VIF

{len(pipeline_results['vif_removed'])} features removed due to multicollinearity (VIF >= 10)

## Final Selected Features (N={len(selected)})

"""
    for feat in selected:
        effect = effect_sizes.get(feat, {})
        report += f"- {feat} (d={effect.get('abs_d', 0):.3f})\n"

    report += f"""
## Cross-Validation Performance

- **Optimal features:** {len(selected)}
- **CV F1 Score:** {pipeline_results['cv_f1_score']:.4f}

## Validation Against Requirements

- Final count < 254: **{'PASS' if len(selected) <= 254 else 'FAIL'}** ({len(selected)} features)
- Cohen's d >= 0.5 for selected features: **PASS** (filter enforced)

---

*Report generated by feature_selection.py*
*Selection method: filter_then_wrapper (Cohen's d + VIF + RFECV)*
"""

    with open(output_path, 'w') as f:
        f.write(report)

    print(f"\nReport saved to: {output_path}")
