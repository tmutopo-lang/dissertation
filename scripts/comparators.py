"""
comparators.py
──────────────
Dissertation: Malicious Redirect Chain Detection
DSRM Activity 5 — Evaluation (Chapter 2, Section 2.3.4; Chapter 3, Section 3.5.4)

The five published approaches named in the research proposal, re-implemented
or adapted so that they run on the same corpus, split and protocol as the
proposed model. Their published numbers were obtained on different datasets
and cannot be compared directly (Chapter 2), which is why they are re-run here.

Single-URL detectors (re-implemented on the URL text, as published):
  do2025      Do et al. [41]: character embedding -> Temporal Convolutional Network
              (dilated residual 1-D convolutions) -> multi-head self-attention
  asiri2024   Asiri et al. [42] PhishTransformer: character-level Transformer
              encoder over the URL, joined with page-content features of the
              landing page ("URL structure with collected page content")
  --url seed     uses the URL the user clicks (first hop)   [default]
  --url landing  uses the final URL of the chain

Flow-sequence intrusion detectors (adapted: each redirect hop plays the role of
a network flow, using the same 71 hop features as the proposed model):
  zhao2023    Zhao et al. [43] CNN-AttBiLSTM: 1-D CNN -> Bi-LSTM -> additive attention
  wang2023    Wang et al. [44] Res-TranBiLSTM: residual block -> Transformer encoder
              -> Bi-LSTM (imbalance handled by class weighting, as for every model)
  alhassan2024 Alhassan [45] SLWAM-BiLSTM: Bi-LSTM -> self-adaptive lightweight
              attention (feature gating + single-score temporal attention)

All layer sizes are documented below; where a paper leaves a detail open, the
closest standard choice is used and stated (Chapter 4 reports these choices).

Usage:
  python scripts/comparators.py --model all --tag full
  python scripts/comparators.py --model do2025 --url landing --tag full
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
from train_model import MaskedMeanPool, standardise

MAX_CHARS = 200
CONTENT_PREFIX = "content:"


# ─── inputs ──────────────────────────────────────────────────────────────────

def encode_urls(urls, max_len=MAX_CHARS):
    """Printable ASCII -> ids 1..95, 0 = padding; longer URLs are truncated."""
    out = np.zeros((len(urls), max_len), dtype=np.int32)
    for i, u in enumerate(urls):
        ids = [min(max(ord(c) - 31, 1), 96) for c in str(u)[:max_len]]
        out[i, : len(ids)] = ids
    return out


def landing_content(X, L, names):
    idx = [i for i, n in enumerate(names) if n.startswith(CONTENT_PREFIX)]
    last = X[np.arange(len(L)), np.maximum(L - 1, 0)]
    return last[:, idx]


# ─── single-URL models ───────────────────────────────────────────────────────

def build_do2025(emb=64, filters=64, heads=4, dropout=0.2):
    inp = keras.Input((MAX_CHARS,), dtype="int32", name="url_chars")
    mask = ops.not_equal(inp, 0)
    x = layers.Embedding(97, emb, name="char_embedding")(inp)
    for d in (1, 2, 4, 8):                                    # TCN residual blocks
        h = layers.Conv1D(filters, 3, padding="causal", dilation_rate=d, activation="relu")(x)
        h = layers.Dropout(dropout)(h)
        h = layers.Conv1D(filters, 3, padding="causal", dilation_rate=d, activation="relu")(h)
        res = x if x.shape[-1] == filters else layers.Conv1D(filters, 1)(x)
        x = layers.Activation("relu")(h + res)
    att_mask = ops.logical_and(ops.expand_dims(mask, 1), ops.expand_dims(mask, 2))
    a = layers.MultiHeadAttention(heads, key_dim=filters // heads, name="mhsa")(x, x, attention_mask=att_mask)
    x = layers.LayerNormalization()(x + a)
    x = MaskedMeanPool()(x, mask)
    x = layers.Dropout(dropout)(layers.Dense(64, activation="relu")(x))
    out = layers.Dense(1, activation="sigmoid")(x)
    return keras.Model(inp, out, name="do2025_tcn_mhsa")


def transformer_block(x, mask, dim, heads, dropout):
    att_mask = ops.logical_and(ops.expand_dims(mask, 1), ops.expand_dims(mask, 2))
    a = layers.MultiHeadAttention(heads, key_dim=dim // heads, dropout=dropout)(x, x, attention_mask=att_mask)
    x = layers.LayerNormalization()(x + a)
    f = layers.Dense(dim * 2, activation="relu")(x)
    f = layers.Dense(dim)(layers.Dropout(dropout)(f))
    return layers.LayerNormalization()(x + f)


def build_asiri2024(n_content, dim=64, heads=4, blocks=2, dropout=0.2):
    inp = keras.Input((MAX_CHARS,), dtype="int32", name="url_chars")
    cin = keras.Input((n_content,), name="landing_content")
    mask = ops.not_equal(inp, 0)
    x = layers.Embedding(97, dim)(inp) + layers.Embedding(MAX_CHARS, dim)(ops.arange(MAX_CHARS))
    for _ in range(blocks):
        x = transformer_block(x, mask, dim, heads, dropout)
    x = MaskedMeanPool()(x, mask)
    c = layers.Dense(32, activation="relu")(cin)
    x = layers.Concatenate()([x, c])
    x = layers.Dropout(dropout)(layers.Dense(64, activation="relu")(x))
    out = layers.Dense(1, activation="sigmoid")(x)
    return keras.Model([inp, cin], out, name="asiri2024_phishtransformer")


# ─── flow-sequence models (hop = flow) ───────────────────────────────────────

class AdditiveAttentionPool(layers.Layer):
    """Single-score temporal attention (Bahdanau style) with padding mask."""

    def __init__(self, units=64, **kw):
        super().__init__(**kw)
        self.w = layers.Dense(units, activation="tanh")
        self.v = layers.Dense(1)

    def call(self, x, mask):
        e = ops.squeeze(self.v(self.w(x)), -1)
        e = ops.where(mask, e, -1e9)
        a = ops.expand_dims(ops.softmax(e, axis=1), -1)
        return ops.sum(a * x, axis=1)


def build_zhao2023(T, F, filters=64, units=64, dropout=0.3):
    xin = keras.Input((T, F)); m = keras.Input((T,), dtype="bool")
    x = layers.Conv1D(filters, 3, padding="same", activation="relu")(xin)
    x = layers.Bidirectional(layers.LSTM(units, return_sequences=True))(x, mask=m)
    x = AdditiveAttentionPool(units)(x, m)
    x = layers.Dropout(dropout)(layers.Dense(64, activation="relu")(x))
    return keras.Model([xin, m], layers.Dense(1, activation="sigmoid")(x), name="zhao2023_cnn_attbilstm")


def build_wang2023(T, F, dim=64, heads=4, units=64, dropout=0.3):
    xin = keras.Input((T, F)); m = keras.Input((T,), dtype="bool")
    x = layers.Dense(dim)(xin)
    h = layers.Dense(dim, activation="relu")(x)                       # residual block
    h = layers.Dense(dim)(layers.Dropout(dropout)(h))
    x = layers.Activation("relu")(x + h)
    x = transformer_block(x, m, dim, heads, dropout)                   # Transformer encoder
    x = layers.Bidirectional(layers.LSTM(units, return_sequences=True))(x, mask=m)
    x = MaskedMeanPool()(x, m)
    x = layers.Dropout(dropout)(layers.Dense(64, activation="relu")(x))
    return keras.Model([xin, m], layers.Dense(1, activation="sigmoid")(x), name="wang2023_res_tranbilstm")


def build_alhassan2024(T, F, units=64, dropout=0.3):
    xin = keras.Input((T, F)); m = keras.Input((T,), dtype="bool")
    x = layers.Bidirectional(layers.LSTM(units, return_sequences=True))(xin, mask=m)
    d = units * 2
    g = MaskedMeanPool()(x, m)                                          # self-adaptive feature gate
    g = layers.Dense(d // 8, activation="relu")(g)
    g = layers.Dense(d, activation="sigmoid")(g)
    x = x * ops.expand_dims(g, 1)
    x = AdditiveAttentionPool(d // 4)(x, m)                             # lightweight temporal attention
    x = layers.Dropout(dropout)(layers.Dense(64, activation="relu")(x))
    return keras.Model([xin, m], layers.Dense(1, activation="sigmoid")(x), name="alhassan2024_slwam_bilstm")


# ─── run ─────────────────────────────────────────────────────────────────────

def run_one(name, d, names, args):
    X, L, y, src = d["X"], d["lengths"].astype(int), d["y"].astype(int), d["source"]
    N, T, F = X.shape
    tr, va, te = temporal_split(N, src)

    if name in ("do2025", "asiri2024"):
        if "seed_url" not in d:
            raise SystemExit("Rebuild the dataset with the latest build_dataset.py (it now stores URLs).")
        urls = d["seed_url"] if args.url == "seed" else d["landing_url"]
        U = encode_urls(urls)
        if name == "do2025":
            model = build_do2025()
            inputs = lambda i: U[i]
        else:
            C = landing_content(X, L, names)
            mu, sd = C[tr].mean(0), C[tr].std(0); sd[sd < 1e-6] = 1
            C = ((C - mu) / sd).astype(np.float32)
            model = build_asiri2024(C.shape[1])
            inputs = lambda i: [U[i], C[i]]
        label = f"{name}_{args.url}"
    else:
        Xs, mask, _, _ = standardise(X, L, tr)
        model = {"zhao2023": build_zhao2023, "wang2023": build_wang2023,
                 "alhassan2024": build_alhassan2024}[name](T, F)
        inputs = lambda i: [Xs[i], mask[i]]
        label = name

    model.compile(optimizer=keras.optimizers.Adam(args.lr), loss="binary_crossentropy",
                  metrics=[keras.metrics.AUC(name="auc")])
    pos = max(y[tr].mean(), 1e-6)
    cw = {0: 0.5 / (1 - pos), 1: 0.5 / pos}
    cb = [keras.callbacks.EarlyStopping(monitor="val_auc", mode="max", patience=5,
                                        restore_best_weights=True)]
    print(f"\n[*] {model.name}: {model.count_params():,} parameters")
    t0 = time.time()
    hist = model.fit(inputs(tr), y[tr], validation_data=(inputs(va), y[va]), epochs=args.epochs,
                     batch_size=args.batch, class_weight=cw, callbacks=cb, verbose=2)
    secs = time.time() - t0

    s_va = model.predict(inputs(va), batch_size=512, verbose=0).ravel()
    s_te = model.predict(inputs(te), batch_size=512, verbose=0).ravel()
    thr = calibrate(y[va], s_va, args.target_fpr)
    val_m, test_m = metrics(y[va], s_va, thr), metrics(y[te], s_te, thr)
    ci = bootstrap(y[te], s_te, thr)
    diag = {}
    for s_name in sorted(set(src[te])):
        i = src[te] == s_name
        yy, pp = y[te][i], s_te[i] >= thr
        if (yy == 0).any():
            diag[f"fpr[{s_name}]"] = float(pp[yy == 0].mean())
        if (yy == 1).any():
            diag[f"recall[{s_name}]"] = float(pp[yy == 1].mean())

    run = f"{args.tag}_{label}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    out = Path(args.out) / run
    out.mkdir(parents=True, exist_ok=True)
    np.savez(out / "test_scores.npz", y=y[te], s=s_te, seed_id=d["seed_id"][te], source=src[te])
    report = {"run": run, "arch": label, "args": vars(args), "params": int(model.count_params()),
              "n": {"train": len(tr), "val": len(va), "test": len(te)}, "threshold": float(thr),
              "validation": val_m, "test": test_m, "test_ci95": ci, "diagnostics": diag,
              "epochs_run": len(hist.history["loss"]), "train_seconds": round(secs, 1),
              "created_utc": datetime.now(timezone.utc).isoformat()}
    (out / "report.json").write_text(json.dumps(report, indent=2))
    print(f"=== {run} ===")
    for k in ("fpr", "recall", "precision", "f1", "auc"):
        lo, hi = ci.get(k, [float("nan")] * 2)
        print(f"  test {k:<10} {test_m[k]:.4f}   95% CI [{lo:.4f}, {hi:.4f}]")
    for k, v in diag.items():
        print(f"  {k:<28} {v:.4f}")


def main():
    ap = argparse.ArgumentParser(description="Five published comparators on the same protocol")
    ap.add_argument("--data", default="data/dataset/dataset.npz")
    ap.add_argument("--model", default="all",
                    choices=["do2025", "asiri2024", "zhao2023", "wang2023", "alhassan2024", "all"])
    ap.add_argument("--url", choices=["seed", "landing"], default="seed",
                    help="which URL the single-URL comparators read")
    ap.add_argument("--epochs", type=int, default=15)
    ap.add_argument("--batch", type=int, default=128)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--target-fpr", type=float, default=0.03)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", default="results")
    ap.add_argument("--tag", default="trial")
    args = ap.parse_args()

    keras.utils.set_random_seed(args.seed)
    d = dict(np.load(args.data, allow_pickle=False))
    names = json.load(open(Path(args.data).parent / "features.json"))["names"]
    todo = (["do2025", "asiri2024", "zhao2023", "wang2023", "alhassan2024"]
            if args.model == "all" else [args.model])
    for name in todo:
        run_one(name, d, names, args)


if __name__ == "__main__":
    main()
