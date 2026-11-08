"""
train_model.py
──────────────
Dissertation: Malicious Redirect Chain Detection
DSRM Activities 3-5 — Design, demonstration and evaluation (Chapters 3-5)

Trains one sequence model on data/dataset/dataset.npz and evaluates it with
the protocol of Chapter 3:

  * temporal split: oldest 80% train, next 10% validation, newest 10% test
  * features standardised with statistics from the training hops only
  * class-weighted binary cross-entropy (the malicious class is rarer)
  * early stopping on validation AUC
  * decision threshold calibrated on VALIDATION data so that validation
    FPR <= --target-fpr (default 3%); it is then frozen
  * the test partition is scored once at that threshold: FPR, recall,
    precision, F1, accuracy, AUC-ROC, with 95% bootstrap confidence intervals,
    plus precision re-weighted to a 1% malicious base rate
  * diagnostics: FPR per benign source (Tranco homepages vs outbound links)
    and recall per malicious source

Architectures (--arch), matching the baselines and ablations of Chapter 3:
  hybrid      Bi-LSTM -> 8-head self-attention -> pooling   (proposed model)
  bilstm      Bi-LSTM only                                   (ablation / baseline)
  lstm        unidirectional LSTM only                       (baseline)
  lstm_attn   unidirectional LSTM + attention                (ablation)
  attn        attention only, no recurrent layer             (ablation)
  cnn         1-D CNN                                        (baseline)

Usage (trial run on a laptop CPU):
  python scripts/train_model.py --arch hybrid --units 64 --layers 1 --epochs 15
Full configuration from the proposal (256 units per direction, 2 layers, 8 heads):
  python scripts/train_model.py --arch hybrid --units 256 --layers 2
"""

import argparse
import json
import os
import time
from datetime import datetime, timezone
from pathlib import Path

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
os.environ.setdefault("TF_ENABLE_ONEDNN_OPTS", "0")

import numpy as np

import keras
from keras import layers, ops
from eval_common import bootstrap, calibrate, metrics, temporal_split


# ─── data ────────────────────────────────────────────────────────────────────

def load(path):
    d = np.load(path, allow_pickle=False)
    return {k: d[k] for k in d.files}


def standardise(X, L, idx_train):
    mask = np.arange(X.shape[1])[None, :] < L[:, None]
    hops = X[idx_train][mask[idx_train]]
    mu, sd = hops.mean(0), hops.std(0)
    sd[sd < 1e-6] = 1.0
    Xs = (X - mu) / sd
    Xs[~mask] = 0.0
    return Xs.astype(np.float32), mask, mu, sd


# ─── model ───────────────────────────────────────────────────────────────────

class MaskedMeanPool(layers.Layer):
    def call(self, x, mask):
        m = ops.cast(ops.expand_dims(mask, -1), x.dtype)
        return ops.sum(x * m, axis=1) / ops.maximum(ops.sum(m, axis=1), 1.0)


def build(arch, T, F, units=64, layers_n=1, heads=8, dropout=0.3, lr=1e-3):
    x_in = keras.Input((T, F), name="hops")
    m_in = keras.Input((T,), dtype="bool", name="mask")
    x = layers.Dense(units * 2, activation="relu", name="hop_projection")(x_in)
    x = layers.Dropout(dropout)(x)

    if arch in ("hybrid", "bilstm"):
        for i in range(layers_n):
            x = layers.Bidirectional(layers.LSTM(units, return_sequences=True),
                                     name=f"bilstm_{i+1}")(x, mask=m_in)
    elif arch in ("lstm", "lstm_attn"):
        for i in range(layers_n):
            x = layers.LSTM(units * 2, return_sequences=True, name=f"lstm_{i+1}")(x, mask=m_in)
    elif arch == "attn":
        pos = layers.Embedding(T, units * 2, name="position")(ops.arange(T))
        x = x + pos
    elif arch == "cnn":
        for i in range(layers_n):
            x = layers.Conv1D(units * 2, 3, padding="same", activation="relu", name=f"conv_{i+1}")(x)
    else:
        raise ValueError(arch)

    if arch in ("hybrid", "lstm_attn", "attn"):
        att_mask = ops.logical_and(ops.expand_dims(m_in, 1), ops.expand_dims(m_in, 2))
        a = layers.MultiHeadAttention(num_heads=heads, key_dim=max((units * 2) // heads, 8),
                                      dropout=dropout, name="multi_head_attention")(
            x, x, attention_mask=att_mask)
        x = layers.LayerNormalization(name="attn_norm")(x + a)

    x = MaskedMeanPool(name="masked_mean_pool")(x, m_in)
    x = layers.Dense(64, activation="relu")(x)
    x = layers.Dropout(dropout)(x)
    out = layers.Dense(1, activation="sigmoid", name="p_malicious")(x)
    model = keras.Model([x_in, m_in], out, name=f"redirect_{arch}")
    model.compile(optimizer=keras.optimizers.Adam(lr), loss="binary_crossentropy",
                  metrics=[keras.metrics.AUC(name="auc")])
    return model


# ─── main ────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(description="Train and evaluate a redirect-chain model")
    ap.add_argument("--data", default="data/dataset/dataset.npz")
    ap.add_argument("--arch", default="hybrid",
                    choices=["hybrid", "bilstm", "lstm", "lstm_attn", "attn", "cnn"])
    ap.add_argument("--units", type=int, default=64, help="per direction (proposal: 256)")
    ap.add_argument("--layers", type=int, default=1, help="recurrent layers (proposal: 2)")
    ap.add_argument("--heads", type=int, default=8)
    ap.add_argument("--dropout", type=float, default=0.3)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--batch", type=int, default=128)
    ap.add_argument("--target-fpr", type=float, default=0.03)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--no-class-weight", action="store_true", help="ablation")
    ap.add_argument("--fixed-threshold", type=float, default=None,
                    help="ablation: skip calibration and use this threshold (e.g. 0.5)")
    ap.add_argument("--out", default="results")
    ap.add_argument("--tag", default="trial")
    args = ap.parse_args()

    keras.utils.set_random_seed(args.seed)
    d = load(args.data)
    X, L, y, src = d["X"], d["lengths"].astype(int), d["y"].astype(int), d["source"]
    N, T, F = X.shape
    tr, va, te = temporal_split(N, src)
    Xs, mask, mu, sd = standardise(X, L, tr)
    print(f"[*] {N:,} chains, {T} hops x {F} features | train {len(tr):,}  val {len(va):,}  "
          f"test {len(te):,} | malicious share train {y[tr].mean():.1%}  test {y[te].mean():.1%}")

    model = build(args.arch, T, F, args.units, args.layers, args.heads, args.dropout, args.lr)
    print(f"[*] {model.name}: {model.count_params():,} parameters")
    cw = None
    if not args.no_class_weight:
        pos = max(y[tr].mean(), 1e-6)
        cw = {0: 0.5 / (1 - pos), 1: 0.5 / pos}
    cb = [keras.callbacks.EarlyStopping(monitor="val_auc", mode="max", patience=5,
                                        restore_best_weights=True)]
    t0 = time.time()
    hist = model.fit([Xs[tr], mask[tr]], y[tr], validation_data=([Xs[va], mask[va]], y[va]),
                     epochs=args.epochs, batch_size=args.batch, class_weight=cw,
                     callbacks=cb, verbose=2)
    train_s = time.time() - t0

    s_va = model.predict([Xs[va], mask[va]], batch_size=512, verbose=0).ravel()
    thr = args.fixed_threshold if args.fixed_threshold is not None else calibrate(y[va], s_va, args.target_fpr)
    val_m = metrics(y[va], s_va, thr)
    s_te = model.predict([Xs[te], mask[te]], batch_size=512, verbose=0).ravel()
    test_m = metrics(y[te], s_te, thr)
    ci = bootstrap(y[te], s_te, thr)

    diag = {}
    for s_name in sorted(set(src[te])):
        i = src[te] == s_name
        yy, pp = y[te][i], s_te[i] >= thr
        if (yy == 0).any():
            diag[f"fpr[{s_name}]"] = float(pp[yy == 0].mean())
        if (yy == 1).any():
            diag[f"recall[{s_name}]"] = float(pp[yy == 1].mean())

    run = f"{args.tag}_{args.arch}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    out = Path(args.out) / run
    out.mkdir(parents=True, exist_ok=True)
    model.save(out / "model.keras")
    np.savez(out / "scaler.npz", mu=mu, sd=sd)
    np.savez(out / "test_scores.npz", y=y[te], s=s_te, seed_id=d["seed_id"][te], source=src[te])
    report = {"run": run, "arch": args.arch, "args": vars(args), "params": int(model.count_params()),
              "n": {"train": len(tr), "val": len(va), "test": len(te)},
              "threshold": thr, "validation": val_m, "test": test_m, "test_ci95": ci,
              "diagnostics": diag, "epochs_run": len(hist.history["loss"]),
              "train_seconds": round(train_s, 1),
              "created_utc": datetime.now(timezone.utc).isoformat()}
    (out / "report.json").write_text(json.dumps(report, indent=2))

    print(f"\n=== {run} ===")
    print(f"threshold (validation FPR <= {args.target_fpr:.0%}): {thr:.4f}")
    for k in ("fpr", "recall", "precision", "f1", "auc"):
        lo, hi = ci.get(k, [float("nan")] * 2)
        print(f"  test {k:<10} {test_m[k]:.4f}   95% CI [{lo:.4f}, {hi:.4f}]")
    print(f"  test accuracy   {test_m['accuracy']:.4f}   precision at 1% base rate {test_m['precision_at_1pct']:.4f}")
    print(f"  confusion: TP={test_m['tp']} FP={test_m['fp']} TN={test_m['tn']} FN={test_m['fn']}")
    target_met = ci.get("fpr", [1, 1])[1] < args.target_fpr
    print(f"  FPR target {args.target_fpr:.0%} met (upper CI bound below target): {'YES' if target_met else 'NO'}")
    for k, v in diag.items():
        print(f"  {k:<28} {v:.4f}")
    print(f"[✓] saved to {out}")


if __name__ == "__main__":
    main()
