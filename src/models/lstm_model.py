#!/usr/bin/env python3
"""
LSTM Model for Badminton Stroke Classification
===============================================
Deep learning model using LSTM/GRU for temporal sequence classification.
Better suited for pose sequences than traditional ML.

Author: ITI123 Project
Date: January 2026
"""

import pickle
import numpy as np
import pandas as pd
from pathlib import Path
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (
    LSTM, GRU, Dense, Dropout, BatchNormalization,
    Bidirectional, Input
)
from tensorflow.keras.callbacks import (
    EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
)
from tensorflow.keras.optimizers import Adam
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    classification_report, confusion_matrix, roc_auc_score
)
import matplotlib.pyplot as plt
import seaborn as sns

# =============================================================================
# PATH CONFIGURATION
# =============================================================================
BASE_DIR = Path(__file__).resolve().parents[2]
SPLITS_DIR = BASE_DIR / "data" / "processed" / "splits"
MODELS_DIR = BASE_DIR / "models" / "saved"
REPORTS_DIR = BASE_DIR / "outputs" / "reports"
VIZ_DIR = BASE_DIR / "outputs" / "visualizations"

MODELS_DIR.mkdir(parents=True, exist_ok=True)


# =============================================================================
# LOAD DATA
# =============================================================================
def load_data():
    """Load train/val/test splits."""
    print("Loading data splits...")

    with open(SPLITS_DIR / "train_data.pkl", 'rb') as f:
        train_data = pickle.load(f)

    with open(SPLITS_DIR / "val_data.pkl", 'rb') as f:
        val_data = pickle.load(f)

    with open(SPLITS_DIR / "test_data.pkl", 'rb') as f:
        test_data = pickle.load(f)

    X_train, y_train = train_data['X'], train_data['y']
    X_val, y_val = val_data['X'], val_data['y']
    X_test, y_test = test_data['X'], test_data['y']

    print(f"  Train: {X_train.shape}, {y_train.shape}")
    print(f"  Val:   {X_val.shape}, {y_val.shape}")
    print(f"  Test:  {X_test.shape}, {y_test.shape}")

    return X_train, y_train, X_val, y_val, X_test, y_test


# =============================================================================
# MODEL ARCHITECTURES
# =============================================================================
def build_lstm_model(input_shape, units=64):
    """
    Build LSTM model for sequence classification.

    Args:
        input_shape: (timesteps, features)
        units: Number of LSTM units

    Returns:
        model: Compiled Keras model
    """
    model = Sequential([
        Input(shape=input_shape),

        # First LSTM layer - returns sequences for stacking
        LSTM(units, return_sequences=True),
        BatchNormalization(),
        Dropout(0.3),

        # Second LSTM layer
        LSTM(units // 2, return_sequences=False),
        BatchNormalization(),
        Dropout(0.3),

        # Dense layers
        Dense(32, activation='relu'),
        Dropout(0.2),

        # Output layer (binary classification)
        Dense(1, activation='sigmoid')
    ])

    model.compile(
        optimizer=Adam(learning_rate=0.001),
        loss='binary_crossentropy',
        metrics=['accuracy']
    )

    return model


def build_bidirectional_lstm(input_shape, units=64):
    """
    Build Bidirectional LSTM model.
    Processes sequence in both directions for better context.

    Args:
        input_shape: (timesteps, features)
        units: Number of LSTM units

    Returns:
        model: Compiled Keras model
    """
    model = Sequential([
        Input(shape=input_shape),

        # Bidirectional LSTM
        Bidirectional(LSTM(units, return_sequences=True)),
        BatchNormalization(),
        Dropout(0.3),

        Bidirectional(LSTM(units // 2, return_sequences=False)),
        BatchNormalization(),
        Dropout(0.3),

        # Dense layers
        Dense(32, activation='relu'),
        Dropout(0.2),

        Dense(1, activation='sigmoid')
    ])

    model.compile(
        optimizer=Adam(learning_rate=0.001),
        loss='binary_crossentropy',
        metrics=['accuracy']
    )

    return model


def build_gru_model(input_shape, units=64):
    """
    Build GRU model (faster training than LSTM).

    Args:
        input_shape: (timesteps, features)
        units: Number of GRU units

    Returns:
        model: Compiled Keras model
    """
    model = Sequential([
        Input(shape=input_shape),

        GRU(units, return_sequences=True),
        BatchNormalization(),
        Dropout(0.3),

        GRU(units // 2, return_sequences=False),
        BatchNormalization(),
        Dropout(0.3),

        Dense(32, activation='relu'),
        Dropout(0.2),

        Dense(1, activation='sigmoid')
    ])

    model.compile(
        optimizer=Adam(learning_rate=0.001),
        loss='binary_crossentropy',
        metrics=['accuracy']
    )

    return model


# =============================================================================
# TRAINING
# =============================================================================
def train_model(model, X_train, y_train, X_val, y_val, model_name, epochs=50):
    """
    Train model with callbacks.

    Args:
        model: Keras model
        X_train, y_train: Training data
        X_val, y_val: Validation data
        model_name: Name for saving
        epochs: Maximum epochs

    Returns:
        history: Training history
    """
    print(f"\nTraining {model_name}...")
    print(f"  Epochs: {epochs}")
    print(f"  Training samples: {len(X_train)}")
    print(f"  Validation samples: {len(X_val)}")

    # Calculate class weights for imbalanced data
    n_pos = np.sum(y_train)
    n_neg = len(y_train) - n_pos
    class_weight = {0: len(y_train) / (2 * n_neg), 1: len(y_train) / (2 * n_pos)}
    print(f"  Class weights: {class_weight}")

    # Callbacks
    callbacks = [
        EarlyStopping(
            monitor='val_loss',
            patience=10,
            restore_best_weights=True,
            verbose=1
        ),
        ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=5,
            min_lr=1e-6,
            verbose=1
        ),
        ModelCheckpoint(
            filepath=str(MODELS_DIR / f"{model_name}.keras"),
            monitor='val_loss',
            save_best_only=True,
            verbose=0
        )
    ]

    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=epochs,
        batch_size=32,
        class_weight=class_weight,
        callbacks=callbacks,
        verbose=1
    )

    return history


# =============================================================================
# EVALUATION
# =============================================================================
def evaluate_model(model, X_test, y_test, model_name):
    """
    Evaluate model on test set.

    Returns:
        metrics: Dictionary of metrics
        cm: Confusion matrix
    """
    print(f"\nEvaluating {model_name} on test set...")

    y_prob = model.predict(X_test, verbose=0).flatten()
    y_pred = (y_prob > 0.5).astype(int)

    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    roc_auc = roc_auc_score(y_test, y_prob)

    print(f"\n{model_name} Test Results:")
    print("-" * 40)
    print(f"  Accuracy:  {accuracy:.4f}")
    print(f"  Precision: {precision:.4f}")
    print(f"  Recall:    {recall:.4f}")
    print(f"  F1 Score:  {f1:.4f}")
    print(f"  ROC AUC:   {roc_auc:.4f}")

    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=['Clear', 'Smash']))

    cm = confusion_matrix(y_test, y_pred)

    metrics = {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'roc_auc': roc_auc
    }

    return metrics, cm


# =============================================================================
# VISUALIZATION
# =============================================================================
def plot_training_history(history, model_name, save_path):
    """Plot training/validation loss and accuracy."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

    # Loss
    ax1.plot(history.history['loss'], label='Train')
    ax1.plot(history.history['val_loss'], label='Validation')
    ax1.set_title(f'{model_name} - Loss')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Accuracy
    ax2.plot(history.history['accuracy'], label='Train')
    ax2.plot(history.history['val_accuracy'], label='Validation')
    ax2.set_title(f'{model_name} - Accuracy')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"  Saved: {save_path}")


def plot_confusion_matrix(cm, model_name, save_path):
    """Plot confusion matrix."""
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


# =============================================================================
# MAIN
# =============================================================================
def main():
    """Main function."""
    print("\n")
    print("*" * 70)
    print("*" + " " * 68 + "*")
    print("*" + "  ITI123 Project: LSTM/GRU Models".center(68) + "*")
    print("*" + "  Deep Learning for Stroke Classification".center(68) + "*")
    print("*" + " " * 68 + "*")
    print("*" * 70)
    print("\n")

    # Load data
    X_train, y_train, X_val, y_val, X_test, y_test = load_data()

    input_shape = (X_train.shape[1], X_train.shape[2])  # (timesteps, features)
    print(f"\nInput shape: {input_shape}")

    # Store results
    results = {}

    # Train LSTM
    print("\n" + "=" * 70)
    print("LSTM MODEL")
    print("=" * 70)

    lstm_model = build_lstm_model(input_shape, units=64)
    lstm_model.summary()

    lstm_history = train_model(
        lstm_model, X_train, y_train, X_val, y_val,
        model_name="lstm", epochs=50
    )

    lstm_metrics, lstm_cm = evaluate_model(lstm_model, X_test, y_test, "LSTM")
    results['LSTM'] = lstm_metrics

    plot_training_history(lstm_history, "LSTM", VIZ_DIR / "lstm_training.png")
    plot_confusion_matrix(lstm_cm, "LSTM", VIZ_DIR / "cm_lstm.png")

    # Train Bidirectional LSTM
    print("\n" + "=" * 70)
    print("BIDIRECTIONAL LSTM MODEL")
    print("=" * 70)

    bilstm_model = build_bidirectional_lstm(input_shape, units=64)
    bilstm_model.summary()

    bilstm_history = train_model(
        bilstm_model, X_train, y_train, X_val, y_val,
        model_name="bilstm", epochs=50
    )

    bilstm_metrics, bilstm_cm = evaluate_model(bilstm_model, X_test, y_test, "BiLSTM")
    results['BiLSTM'] = bilstm_metrics

    plot_training_history(bilstm_history, "BiLSTM", VIZ_DIR / "bilstm_training.png")
    plot_confusion_matrix(bilstm_cm, "BiLSTM", VIZ_DIR / "cm_bilstm.png")

    # Train GRU
    print("\n" + "=" * 70)
    print("GRU MODEL")
    print("=" * 70)

    gru_model = build_gru_model(input_shape, units=64)
    gru_model.summary()

    gru_history = train_model(
        gru_model, X_train, y_train, X_val, y_val,
        model_name="gru", epochs=50
    )

    gru_metrics, gru_cm = evaluate_model(gru_model, X_test, y_test, "GRU")
    results['GRU'] = gru_metrics

    plot_training_history(gru_history, "GRU", VIZ_DIR / "gru_training.png")
    plot_confusion_matrix(gru_cm, "GRU", VIZ_DIR / "cm_gru.png")

    # Summary
    print("\n" + "=" * 70)
    print("DEEP LEARNING MODEL SUMMARY")
    print("=" * 70)
    print()
    print(f"{'Model':<15} {'Accuracy':<12} {'F1':<12} {'ROC AUC':<12}")
    print("-" * 51)
    for name, metrics in results.items():
        print(f"{name:<15} {metrics['accuracy']:<12.4f} {metrics['f1']:<12.4f} {metrics['roc_auc']:<12.4f}")
    print()

    # Best model
    best_model = max(results.keys(), key=lambda x: results[x]['f1'])
    best_f1 = results[best_model]['f1']
    print(f"Best Model: {best_model} (F1: {best_f1:.4f})")

    # Save results
    results_df = pd.DataFrame(results).T
    results_df.to_csv(REPORTS_DIR / "lstm_results.csv")
    print(f"\nResults saved to: {REPORTS_DIR / 'lstm_results.csv'}")

    print("\n" + "=" * 70)
    print("DEEP LEARNING TRAINING COMPLETE!")
    print("=" * 70)


if __name__ == "__main__":
    main()
