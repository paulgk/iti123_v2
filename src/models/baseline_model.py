#!/usr/bin/env python3
"""
Baseline Model for Badminton Stroke Classification
===================================================
Traditional ML models (Random Forest, SVM) as baseline.
Uses flattened features from pose sequences.

Features:
- Random Forest Classifier
- Support Vector Machine (SVM)
- Cross-validation
- Class weight handling for imbalanced data
- Model comparison and evaluation

Author: ITI123 Project
Date: January 2026
"""

import pickle
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    classification_report, confusion_matrix, roc_auc_score
)
from sklearn.model_selection import cross_val_score
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

# =============================================================================
# PATH CONFIGURATION
# =============================================================================
BASE_DIR = Path(__file__).resolve().parents[2]
SPLITS_DIR = BASE_DIR / "data" / "processed" / "splits"  # Original MediaPipe splits
MODELS_DIR = BASE_DIR / "models" / "saved"
REPORTS_DIR = BASE_DIR / "outputs" / "reports"
VIZ_DIR = BASE_DIR / "outputs" / "visualizations"

# Create directories
MODELS_DIR.mkdir(parents=True, exist_ok=True)
VIZ_DIR.mkdir(parents=True, exist_ok=True)

# =============================================================================
# LOAD DATA
# =============================================================================
def load_data():
    """
    Load train/val/test splits.

    Returns:
        X_train_seq, X_train_stat, y_train, X_val_seq, X_val_stat, y_val, X_test_seq, X_test_stat, y_test
    """
    print("Loading data splits...")

    # Load training data
    with open(SPLITS_DIR / "train_data.pkl", 'rb') as f:
        train_data = pickle.load(f)

    # Load validation data
    with open(SPLITS_DIR / "val_data.pkl", 'rb') as f:
        val_data = pickle.load(f)

    # Load test data
    with open(SPLITS_DIR / "test_data.pkl", 'rb') as f:
        test_data = pickle.load(f)

    # Sequence features (for LSTM)
    X_train_seq, y_train = train_data['X'], train_data['y']
    X_val_seq, y_val = val_data['X'], val_data['y']
    X_test_seq, y_test = test_data['X'], test_data['y']

    # Statistical features (for baseline models) - use X_stat if available
    X_train_stat = train_data.get('X_stat', None)
    X_val_stat = val_data.get('X_stat', None)
    X_test_stat = test_data.get('X_stat', None)

    print(f"  Train: seq={X_train_seq.shape}, stat={X_train_stat.shape if X_train_stat is not None else 'N/A'}, y={y_train.shape}")
    print(f"  Val:   seq={X_val_seq.shape}, stat={X_val_stat.shape if X_val_stat is not None else 'N/A'}, y={y_val.shape}")
    print(f"  Test:  seq={X_test_seq.shape}, stat={X_test_stat.shape if X_test_stat is not None else 'N/A'}, y={y_test.shape}")
    print()

    return X_train_seq, X_train_stat, y_train, X_val_seq, X_val_stat, y_val, X_test_seq, X_test_stat, y_test


def flatten_sequences(X):
    """
    Flatten sequence data for traditional ML models.

    Input: (samples, timesteps, features)
    Output: (samples, timesteps * features)

    Also extracts statistical features from sequences.
    """
    n_samples, n_timesteps, n_features = X.shape

    # Option 1: Simple flatten
    X_flat = X.reshape(n_samples, -1)

    # Option 2: Statistical features (mean, std, min, max per feature)
    X_mean = np.mean(X, axis=1)  # (samples, features)
    X_std = np.std(X, axis=1)
    X_min = np.min(X, axis=1)
    X_max = np.max(X, axis=1)

    # Combine statistical features
    X_stats = np.hstack([X_mean, X_std, X_min, X_max])

    return X_flat, X_stats


# =============================================================================
# MODEL TRAINING
# =============================================================================
def train_random_forest(X_train, y_train, X_val, y_val):
    """
    Train Random Forest classifier.

    Args:
        X_train, y_train: Training data
        X_val, y_val: Validation data

    Returns:
        model: Trained model
        metrics: Dictionary of evaluation metrics
    """
    print("=" * 60)
    print("TRAINING RANDOM FOREST")
    print("=" * 60)

    # Calculate class weights for imbalanced data
    # class_weight='balanced' automatically adjusts weights
    model = RandomForestClassifier(
        n_estimators=100,
        max_depth=20,
        min_samples_split=5,
        min_samples_leaf=2,
        class_weight='balanced',
        random_state=42,
        n_jobs=-1
    )

    print("Training model...")
    model.fit(X_train, y_train)

    # Evaluate on validation set
    y_pred = model.predict(X_val)
    y_prob = model.predict_proba(X_val)[:, 1]

    metrics = evaluate_model(y_val, y_pred, y_prob, "Random Forest")

    return model, metrics


def train_svm(X_train, y_train, X_val, y_val):
    """
    Train Support Vector Machine classifier.

    Args:
        X_train, y_train: Training data
        X_val, y_val: Validation data

    Returns:
        model: Trained model
        metrics: Dictionary of evaluation metrics
    """
    print("=" * 60)
    print("TRAINING SVM")
    print("=" * 60)

    # Scale features for SVM
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)

    # SVM with RBF kernel
    model = SVC(
        kernel='rbf',
        C=1.0,
        gamma='scale',
        class_weight='balanced',
        probability=True,
        random_state=42
    )

    print("Training model...")
    model.fit(X_train_scaled, y_train)

    # Evaluate on validation set
    y_pred = model.predict(X_val_scaled)
    y_prob = model.predict_proba(X_val_scaled)[:, 1]

    metrics = evaluate_model(y_val, y_pred, y_prob, "SVM")

    # Return model with scaler
    return {'model': model, 'scaler': scaler}, metrics


# =============================================================================
# EVALUATION
# =============================================================================
def evaluate_model(y_true, y_pred, y_prob, model_name):
    """
    Evaluate model performance.

    Args:
        y_true: True labels
        y_pred: Predicted labels
        y_prob: Predicted probabilities
        model_name: Name of the model

    Returns:
        metrics: Dictionary of evaluation metrics
    """
    print(f"\n{model_name} Validation Results:")
    print("-" * 40)

    # Calculate metrics
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)
    roc_auc = roc_auc_score(y_true, y_prob)

    print(f"  Accuracy:  {accuracy:.4f}")
    print(f"  Precision: {precision:.4f}")
    print(f"  Recall:    {recall:.4f}")
    print(f"  F1 Score:  {f1:.4f}")
    print(f"  ROC AUC:   {roc_auc:.4f}")
    print()

    # Classification report
    print("Classification Report:")
    print(classification_report(y_true, y_pred, target_names=['Clear', 'Smash']))

    metrics = {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'roc_auc': roc_auc
    }

    return metrics


def evaluate_on_test(model, X_test, y_test, model_name, is_svm=False):
    """
    Final evaluation on test set.

    Args:
        model: Trained model (or dict with model and scaler for SVM)
        X_test: Test features
        y_test: Test labels
        model_name: Name of the model
        is_svm: Whether the model is SVM (needs scaling)

    Returns:
        metrics: Dictionary of test metrics
    """
    print(f"\n{'=' * 60}")
    print(f"TEST SET EVALUATION: {model_name}")
    print(f"{'=' * 60}")

    if is_svm:
        scaler = model['scaler']
        clf = model['model']
        X_test_scaled = scaler.transform(X_test)
        y_pred = clf.predict(X_test_scaled)
        y_prob = clf.predict_proba(X_test_scaled)[:, 1]
    else:
        y_pred = model.predict(X_test)
        y_prob = model.predict_proba(X_test)[:, 1]

    metrics = evaluate_model(y_test, y_pred, y_prob, f"{model_name} (Test)")

    # Confusion matrix
    cm = confusion_matrix(y_test, y_pred)

    return metrics, cm, y_pred, y_prob


# =============================================================================
# VISUALIZATION
# =============================================================================
def plot_confusion_matrix(cm, model_name, save_path):
    """Plot and save confusion matrix."""
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=['Clear', 'Smash'],
                yticklabels=['Clear', 'Smash'])
    plt.title(f'Confusion Matrix - {model_name}')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"  Saved: {save_path}")


def plot_model_comparison(results, save_path):
    """Plot comparison of model metrics."""
    metrics = ['accuracy', 'precision', 'recall', 'f1', 'roc_auc']

    fig, ax = plt.subplots(figsize=(10, 6))

    x = np.arange(len(metrics))
    width = 0.35

    rf_values = [results['Random Forest'][m] for m in metrics]
    svm_values = [results['SVM'][m] for m in metrics]

    bars1 = ax.bar(x - width/2, rf_values, width, label='Random Forest', color='steelblue')
    bars2 = ax.bar(x + width/2, svm_values, width, label='SVM', color='coral')

    ax.set_ylabel('Score')
    ax.set_title('Baseline Model Comparison (Test Set)')
    ax.set_xticks(x)
    ax.set_xticklabels(['Accuracy', 'Precision', 'Recall', 'F1', 'ROC AUC'])
    ax.legend()
    ax.set_ylim(0, 1)

    # Add value labels on bars
    for bar in bars1:
        height = bar.get_height()
        ax.annotate(f'{height:.3f}',
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3),
                    textcoords="offset points",
                    ha='center', va='bottom', fontsize=8)

    for bar in bars2:
        height = bar.get_height()
        ax.annotate(f'{height:.3f}',
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3),
                    textcoords="offset points",
                    ha='center', va='bottom', fontsize=8)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"  Saved: {save_path}")


def plot_feature_importance(model, feature_names, save_path, top_n=20):
    """Plot feature importance for Random Forest."""
    importances = model.feature_importances_

    # Get top N features
    indices = np.argsort(importances)[::-1][:top_n]

    plt.figure(figsize=(10, 8))
    plt.title('Feature Importance (Random Forest)')
    plt.barh(range(top_n), importances[indices][::-1], color='steelblue')
    plt.yticks(range(top_n), [f'Feature {i}' for i in indices[::-1]])
    plt.xlabel('Importance')
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"  Saved: {save_path}")


# =============================================================================
# MAIN FUNCTION
# =============================================================================
def main():
    """Main function to train and evaluate baseline models."""
    print("\n")
    print("*" * 70)
    print("*" + " " * 68 + "*")
    print("*" + "  ITI123 Project: Baseline Models".center(68) + "*")
    print("*" + "  Random Forest & SVM".center(68) + "*")
    print("*" + " " * 68 + "*")
    print("*" * 70)
    print("\n")

    # Load data
    X_train_seq, X_train_stat, y_train, X_val_seq, X_val_stat, y_val, X_test_seq, X_test_stat, y_test = load_data()

    # Prepare features for traditional ML
    print("Preparing features...")

    # Load raw statistical features (better for traditional ML with built-in scaling)
    with open(SPLITS_DIR / "train_data.pkl", 'rb') as f:
        train_data = pickle.load(f)
    with open(SPLITS_DIR / "val_data.pkl", 'rb') as f:
        val_data = pickle.load(f)
    with open(SPLITS_DIR / "test_data.pkl", 'rb') as f:
        test_data = pickle.load(f)

    # Use raw statistical features if available (better class separation)
    X_train_raw = train_data.get('X_stat_raw', None)
    X_val_raw = val_data.get('X_stat_raw', None)
    X_test_raw = test_data.get('X_stat_raw', None)

    if X_train_raw is not None:
        print("  Using RAW statistical features (V2 format)")
        X_train_use = X_train_raw
        X_val_use = X_val_raw
        X_test_use = X_test_raw
    elif X_train_stat is not None:
        print("  Using normalized statistical features (V2 format)")
        X_train_use = X_train_stat
        X_val_use = X_val_stat
        X_test_use = X_test_stat
    else:
        # Fall back to computing from sequences (V1 format)
        print("  Computing statistical features from sequences (V1 format)")
        _, X_train_use = flatten_sequences(X_train_seq)
        _, X_val_use = flatten_sequences(X_val_seq)
        _, X_test_use = flatten_sequences(X_test_seq)

    print(f"  Statistical features shape: {X_train_use.shape}")
    print()

    # Train models
    rf_model, rf_val_metrics = train_random_forest(X_train_use, y_train, X_val_use, y_val)
    svm_model, svm_val_metrics = train_svm(X_train_use, y_train, X_val_use, y_val)

    # Evaluate on test set
    print("\n" + "=" * 70)
    print("FINAL TEST SET EVALUATION")
    print("=" * 70)

    rf_test_metrics, rf_cm, rf_pred, rf_prob = evaluate_on_test(
        rf_model, X_test_use, y_test, "Random Forest", is_svm=False
    )

    svm_test_metrics, svm_cm, svm_pred, svm_prob = evaluate_on_test(
        svm_model, X_test_use, y_test, "SVM", is_svm=True
    )

    # Save models
    print("\n" + "=" * 70)
    print("SAVING MODELS")
    print("=" * 70)

    joblib.dump(rf_model, MODELS_DIR / "baseline_rf.joblib")
    print(f"  Saved: {MODELS_DIR / 'baseline_rf.joblib'}")

    joblib.dump(svm_model, MODELS_DIR / "baseline_svm.joblib")
    print(f"  Saved: {MODELS_DIR / 'baseline_svm.joblib'}")

    # Generate visualizations
    print("\n" + "=" * 70)
    print("GENERATING VISUALIZATIONS")
    print("=" * 70)

    plot_confusion_matrix(rf_cm, "Random Forest", VIZ_DIR / "cm_random_forest.png")
    plot_confusion_matrix(svm_cm, "SVM", VIZ_DIR / "cm_svm.png")

    test_results = {
        'Random Forest': rf_test_metrics,
        'SVM': svm_test_metrics
    }
    plot_model_comparison(test_results, VIZ_DIR / "baseline_comparison.png")

    plot_feature_importance(rf_model, None, VIZ_DIR / "feature_importance_rf.png")

    # Save results to CSV
    results_df = pd.DataFrame({
        'Model': ['Random Forest', 'SVM'],
        'Accuracy': [rf_test_metrics['accuracy'], svm_test_metrics['accuracy']],
        'Precision': [rf_test_metrics['precision'], svm_test_metrics['precision']],
        'Recall': [rf_test_metrics['recall'], svm_test_metrics['recall']],
        'F1': [rf_test_metrics['f1'], svm_test_metrics['f1']],
        'ROC_AUC': [rf_test_metrics['roc_auc'], svm_test_metrics['roc_auc']]
    })

    results_file = REPORTS_DIR / "baseline_results.csv"
    results_df.to_csv(results_file, index=False)
    print(f"\n  Results saved to: {results_file}")

    # Summary
    print("\n" + "=" * 70)
    print("BASELINE MODEL SUMMARY")
    print("=" * 70)
    print()
    print(f"{'Model':<20} {'Accuracy':<12} {'F1':<12} {'ROC AUC':<12}")
    print("-" * 56)
    print(f"{'Random Forest':<20} {rf_test_metrics['accuracy']:<12.4f} {rf_test_metrics['f1']:<12.4f} {rf_test_metrics['roc_auc']:<12.4f}")
    print(f"{'SVM':<20} {svm_test_metrics['accuracy']:<12.4f} {svm_test_metrics['f1']:<12.4f} {svm_test_metrics['roc_auc']:<12.4f}")
    print()

    # Determine best model
    best_model = "Random Forest" if rf_test_metrics['f1'] > svm_test_metrics['f1'] else "SVM"
    best_f1 = max(rf_test_metrics['f1'], svm_test_metrics['f1'])
    print(f"Best Baseline Model: {best_model} (F1: {best_f1:.4f})")
    print()

    print("=" * 70)
    print("BASELINE TRAINING COMPLETE!")
    print("=" * 70)
    print()
    print("Next steps:")
    print("1. Review visualizations: outputs/visualizations/")
    print("2. Train deep learning model: python src/models/lstm_model.py")
    print()


if __name__ == "__main__":
    main()
