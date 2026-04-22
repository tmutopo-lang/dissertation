"""
model.py
─────────
Dissertation: Malicious Redirect Chain Detection
DSRM Activity 3 — Design and Development

Hybrid deep learning model:
  Bi-LSTM (256 hidden units/direction) + Multi-Head Transformer Attention (8 heads)
  Explicitly optimised for False Positive Rate (FPR) minimisation.

Architecture flow:
  Input → Embedding → Bi-LSTM → Multi-Head Attention → GlobalAvgPool
        → Dense(256) → Dropout → Dense(128) → Dropout → Sigmoid output

Usage:
  from scripts.model import build_model, train_model, evaluate_model
"""

import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, regularizers
from sklearn.metrics import (classification_report, confusion_matrix,
                              roc_auc_score, precision_recall_curve, roc_curve)
from sklearn.utils.class_weight import compute_class_weight
import matplotlib.pyplot as plt
from pathlib import Path
from loguru import logger


# ─── Architecture ────────────────────────────────────────────────────────────

def build_model(
    input_dim: int,
    sequence_length: int = 10,
    bilstm_units: int = 256,
    num_heads: int = 8,
    key_dim: int = 64,
    dense_units: tuple = (256, 128),
    dropout_rate: float = 0.3,
    learning_rate: float = 0.001,
) -> keras.Model:
    """
    Constructs the Bi-LSTM + Multi-Head Transformer Attention hybrid model.

    Args:
        input_dim:       Number of features per URL/hop in the sequence.
        sequence_length: Max redirect chain length (pad/truncate to this).
        bilstm_units:    Hidden units per direction in Bi-LSTM (512 total).
        num_heads:       Attention heads in MultiHeadAttention layer.
        key_dim:         Key dimensionality per attention head.
        dense_units:     Sizes of the two feedforward dense layers.
        dropout_rate:    Dropout applied after each dense layer.
        learning_rate:   Adam optimiser learning rate.

    Returns:
        Compiled Keras model.
    """
    inputs = keras.Input(shape=(sequence_length, input_dim), name="redirect_chain")

    # ── Layer 1: Bi-LSTM ───────────────────────────────────────────────────
    # Captures forward and backward sequential dependencies across redirect hops.
    # return_sequences=True passes the full sequence to the attention layer.
    x = layers.Bidirectional(
        layers.LSTM(bilstm_units, return_sequences=True,
                    kernel_regularizer=regularizers.l2(1e-4)),
        name="bi_lstm"
    )(inputs)

    # ── Layer 2: Multi-Head Self-Attention (Transformer block) ────────────
    # Dynamically weights the importance of each redirect hop and its
    # relationships to other hops — critical for catching late-stage payload.
    attn_output = layers.MultiHeadAttention(
        num_heads=num_heads, key_dim=key_dim, name="multi_head_attention"
    )(x, x)

    # Residual connection + Layer Normalisation (following Wang et al. 2023)
    x = layers.Add(name="residual_add")([x, attn_output])
    x = layers.LayerNormalization(name="layer_norm")(x)

    # ── Layer 3: Global Average Pooling ───────────────────────────────────
    # Aggregates attention-weighted representations across the sequence.
    x = layers.GlobalAveragePooling1D(name="global_avg_pool")(x)

    # ── Layer 4: Feedforward classifier ──────────────────────────────────
    for i, units in enumerate(dense_units):
        x = layers.Dense(units, activation="relu",
                         kernel_regularizer=regularizers.l2(1e-4),
                         name=f"dense_{i+1}")(x)
        x = layers.Dropout(dropout_rate, name=f"dropout_{i+1}")(x)

    # ── Output: Sigmoid for binary classification ────────────────────────
    outputs = layers.Dense(1, activation="sigmoid", name="output")(x)

    model = keras.Model(inputs=inputs, outputs=outputs, name="BiLSTM_Transformer")

    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
        loss="binary_crossentropy",
        metrics=[
            "accuracy",
            keras.metrics.Precision(name="precision"),
            keras.metrics.Recall(name="recall"),
            keras.metrics.AUC(name="auc_roc", curve="ROC"),
            keras.metrics.AUC(name="auc_pr",  curve="PR"),
            keras.metrics.FalsePositives(name="fp"),
            keras.metrics.TrueNegatives(name="tn"),
        ]
    )

    logger.info(f"Model built — parameters: {model.count_params():,}")
    model.summary()
    return model


# ─── Class-weighted training (FPR optimisation) ──────────────────────────────

def get_class_weights(y_train: np.ndarray) -> dict:
    """
    Computes class weights to handle real-world class imbalance.
    Upweights the malicious class to reduce false negatives,
    with a secondary effect of penalising false positives on the benign class.
    """
    classes = np.unique(y_train)
    weights = compute_class_weight("balanced", classes=classes, y=y_train)
    weight_dict = dict(zip(classes.astype(int), weights))
    logger.info(f"Class weights: {weight_dict}")
    return weight_dict


# ─── Training ─────────────────────────────────────────────────────────────────

def train_model(
    model: keras.Model,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    output_dir: str = "models",
    epochs: int = 100,
    batch_size: int = 64,
) -> keras.callbacks.History:
    """
    Trains the model with class weighting and early stopping.
    Saves the best checkpoint by val_auc_roc.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    class_weights = get_class_weights(y_train)

    callbacks = [
        # Early stopping — patience=10 as specified in proposal
        keras.callbacks.EarlyStopping(
            monitor="val_auc_roc", patience=10,
            restore_best_weights=True, mode="max"
        ),
        # Save best model checkpoint
        keras.callbacks.ModelCheckpoint(
            filepath=str(output_dir / "best_model.keras"),
            monitor="val_auc_roc", save_best_only=True, mode="max"
        ),
        # Reduce LR on plateau
        keras.callbacks.ReduceLROnPlateau(
            monitor="val_auc_roc", factor=0.5, patience=5,
            min_lr=1e-6, mode="max"
        ),
        # TensorBoard logging
        keras.callbacks.TensorBoard(
            log_dir=str(output_dir / "logs"),
            histogram_freq=1
        ),
    ]

    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=epochs,
        batch_size=batch_size,
        class_weight=class_weights,
        callbacks=callbacks,
        verbose=1,
    )

    return history


# ─── Evaluation (FPR-focused) ─────────────────────────────────────────────────

def evaluate_model(
    model: keras.Model,
    X_test: np.ndarray,
    y_test: np.ndarray,
    threshold: float = 0.5,
    output_dir: str = "models",
) -> dict:
    """
    Full evaluation suite aligned with the dissertation's primary metrics:
      - False Positive Rate (FPR) at multiple thresholds  ← primary metric
      - F1-score, AUC-ROC, AUC-PR
      - Precision-Recall curve, ROC curve plots

    Returns a dict of all metrics.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    y_proba = model.predict(X_test).flatten()
    y_pred  = (y_proba >= threshold).astype(int)

    # Confusion matrix
    tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
    fpr_at_threshold = fp / (fp + tn) if (fp + tn) > 0 else 0.0
    tpr_at_threshold = tp / (tp + fn) if (tp + fn) > 0 else 0.0

    metrics = {
        "threshold":   threshold,
        "accuracy":    (tp + tn) / (tp + tn + fp + fn),
        "precision":   tp / (tp + fp) if (tp + fp) > 0 else 0.0,
        "recall":      tpr_at_threshold,
        "fpr":         fpr_at_threshold,       # ← Primary metric
        "f1":          2 * tp / (2*tp + fp + fn) if (2*tp + fp + fn) > 0 else 0.0,
        "auc_roc":     roc_auc_score(y_test, y_proba),
        "tp": int(tp), "tn": int(tn), "fp": int(fp), "fn": int(fn),
    }

    logger.info("─── Evaluation Results ───────────────────────────────")
    for k, v in metrics.items():
        logger.info(f"  {k:<15}: {v:.4f}" if isinstance(v, float) else f"  {k:<15}: {v}")

    # FPR at multiple thresholds (for dissertation Table in Chapter 4)
    logger.info("\n  FPR at multiple decision thresholds:")
    for t in [0.3, 0.4, 0.5, 0.6, 0.7, 0.8]:
        yp = (y_proba >= t).astype(int)
        _, fp_t, _, _ = confusion_matrix(y_test, yp, labels=[0,1]).ravel() \
                        if len(np.unique(yp)) > 1 else (0,0,0,0)
        fpr_t = fp_t / (fp_t + (y_test == 0).sum() - fp_t) \
                if (y_test == 0).sum() > 0 else 0.0
        logger.info(f"    threshold={t:.1f}  FPR={fpr_t:.4f}")

    # ── Plots ──────────────────────────────────────────────────────────────
    _plot_roc(y_test, y_proba, metrics["auc_roc"], output_dir)
    _plot_pr(y_test, y_proba, output_dir)

    logger.success(f"Evaluation complete. Plots saved to {output_dir}")
    return metrics


def _plot_roc(y_test, y_proba, auc_val, output_dir: Path):
    fpr, tpr, _ = roc_curve(y_test, y_proba)
    plt.figure(figsize=(6, 5))
    plt.plot(fpr, tpr, lw=2, label=f"AUC-ROC = {auc_val:.4f}")
    plt.plot([0, 1], [0, 1], "k--", lw=1)
    plt.axvline(x=0.03, color="red", linestyle="--", lw=1, label="FPR target (3%)")
    plt.xlabel("False Positive Rate"); plt.ylabel("True Positive Rate")
    plt.title("ROC Curve — Bi-LSTM + Transformer")
    plt.legend(); plt.tight_layout()
    plt.savefig(output_dir / "roc_curve.png", dpi=150)
    plt.close()


def _plot_pr(y_test, y_proba, output_dir: Path):
    precision, recall, _ = precision_recall_curve(y_test, y_proba)
    plt.figure(figsize=(6, 5))
    plt.plot(recall, precision, lw=2)
    plt.xlabel("Recall"); plt.ylabel("Precision")
    plt.title("Precision-Recall Curve — Bi-LSTM + Transformer")
    plt.tight_layout()
    plt.savefig(output_dir / "pr_curve.png", dpi=150)
    plt.close()


# ─── CLI entry point ──────────────────────────────────────────────────────────

def main():
    import argparse
    import pandas as pd
    parser = argparse.ArgumentParser(description="Train or smoke-test the Bi-LSTM+Transformer model")
    parser.add_argument("--features", default="data/processed/features.csv",
                        help="Path to features CSV from extract_features.py")
    parser.add_argument("--output",   default="models", help="Output directory for checkpoints")
    parser.add_argument("--epochs",   type=int, default=50)
    parser.add_argument("--batch",    type=int, default=64)
    parser.add_argument("--smoke",    action="store_true",
                        help="Run a quick smoke-test with random data (no CSV needed)")
    args = parser.parse_args()

    if args.smoke:
        logger.info("Running smoke-test with random data...")
        SEQ_LEN    = 10
        N_FEATURES = 40
        N_SAMPLES  = 200
        X = np.random.rand(N_SAMPLES, SEQ_LEN, N_FEATURES).astype(np.float32)
        y = np.random.randint(0, 2, N_SAMPLES)
        model = build_model(input_dim=N_FEATURES, sequence_length=SEQ_LEN)
        split = int(0.8 * N_SAMPLES)
        train_model(model, X[:split], y[:split], X[split:], y[split:],
                    output_dir=args.output, epochs=3, batch_size=16)
        metrics = evaluate_model(model, X[split:], y[split:], output_dir=args.output)
        logger.success(f"Smoke-test passed. FPR = {metrics['fpr']:.4f}")
        return

    # ── Real training from features CSV ───────────────────────────────────
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import StandardScaler

    logger.info(f"Loading features from {args.features}...")
    df = pd.read_csv(args.features)

    # Drop non-numeric / identifier columns
    drop_cols = ["url", "source", "asn", "country_code", "tld",
                 "collected_at", "domain", "rank"]
    feature_cols = [c for c in df.columns
                    if c not in drop_cols + ["label"] and df[c].dtype != object]

    X_raw = df[feature_cols].fillna(0).values.astype(np.float32)
    y     = df["label"].values.astype(np.float32)

    # Scale features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_raw)

    # Reshape to (samples, sequence_length=1, features) for the LSTM
    # Single-hop sequences for now — multi-hop reshaping comes in Phase 4
    X_seq = X_scaled.reshape(X_scaled.shape[0], 1, X_scaled.shape[1])

    X_train, X_test, y_train, y_test = train_test_split(
        X_seq, y, test_size=0.2, stratify=y, random_state=42)
    X_train, X_val, y_train, y_val = train_test_split(
        X_train, y_train, test_size=0.125, stratify=y_train, random_state=42)

    logger.info(f"Train: {len(X_train):,}  Val: {len(X_val):,}  Test: {len(X_test):,}")

    model = build_model(input_dim=X_seq.shape[2], sequence_length=1)
    train_model(model, X_train, y_train, X_val, y_val,
                output_dir=args.output, epochs=args.epochs, batch_size=args.batch)
    metrics = evaluate_model(model, X_test, y_test, output_dir=args.output)
    logger.success(f"Training complete. FPR={metrics['fpr']:.4f}  AUC={metrics['auc_roc']:.4f}")


if __name__ == "__main__":
    main()
