"""
baselines.py
────────────
Dissertation: Malicious Redirect Chain Detection
DSRM Activity 5 — Evaluation (Chapter 3, Section 3.5.4: traditional baselines)

Traditional machine-learning baselines from the research proposal:
  rf    Random Forest
  gb    Gradient Boosting (histogram-based, scikit-learn)
  svm   Support Vector Machine (RBF kernel)

These models cannot read sequences, so each chain is summarised into one
fixed-length vector: the first hop, the last hop, and the mean / max / min of
every hop feature over the chain, plus the chain length.

Everything else is identical to train_model.py, so the comparison is like for
like: same dataset file, same temporal 80/10/10 split, same class weighting,
same threshold calibration on validation data (FPR <= target), same single
test run, same metrics and bootstrap confidence intervals.

Usage:
  python scripts/baselines.py --model rf
  python scripts/baselines.py --model all
  python scripts/baselines.py --model rf --data data/dataset_struct/dataset.npz --tag structure
"""

import argparse
import json
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from eval_common import bootstrap, calibrate, metrics, temporal_split


def summarise(X, L):
    """(N, T, F) hop sequences -> (N, 5F + 1) chain vectors."""
    N, T, F = X.shape
    mask = (np.arange(T)[None, :] < L[:, None])[:, :, None]
    first = X[:, 0, :]
    last = X[np.arange(N), np.maximum(L - 1, 0), :]
    cnt = np.maximum(L, 1)[:, None]
    mean = (X * mask).sum(1) / cnt
    big = np.where(mask, X, -np.inf).max(1)
    small = np.where(mask, X, np.inf).min(1)
    big[~np.isfinite(big)] = 0
    small[~np.isfinite(small)] = 0
    return np.hstack([first, last, mean, big, small, L[:, None] / T]).astype(np.float32)


def make(name, seed, pos_share):
    if name == "rf":
        return RandomForestClassifier(n_estimators=400, min_samples_leaf=2, n_jobs=-1,
                                      class_weight="balanced", random_state=seed)
    if name == "gb":
        w = {0: 0.5 / (1 - pos_share), 1: 0.5 / pos_share}
        return HistGradientBoostingClassifier(max_iter=400, learning_rate=0.08,
                                              early_stopping=True, random_state=seed,
                                              class_weight=w)
    if name == "svm":
        return SVC(kernel="rbf", C=1.0, gamma="scale", class_weight="balanced",
                   random_state=seed)
    raise ValueError(name)


def score(model, X):
    if hasattr(model, "predict_proba"):
        return model.predict_proba(X)[:, 1]
    return model.decision_function(X)


def run_one(name, d, args):
    X, L, y, src = d["X"], d["lengths"].astype(int), d["y"].astype(int), d["source"]
    tr, va, te = temporal_split(len(y), src)
    Z = summarise(X, L)
    sc = StandardScaler().fit(Z[tr])
    Z = sc.transform(Z).astype(np.float32)

    fit_idx = tr
    if name == "svm" and len(tr) > args.svm_max_train:   # RBF SVM is O(n^2): subsample train
        rng = np.random.default_rng(args.seed)
        fit_idx = np.sort(rng.choice(tr, args.svm_max_train, replace=False))

    model = make(name, args.seed, max(y[tr].mean(), 1e-6))
    t0 = time.time()
    model.fit(Z[fit_idx], y[fit_idx])
    secs = time.time() - t0

    s_va, s_te = score(model, Z[va]), score(model, Z[te])
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

    run = f"{args.tag}_{name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    out = Path(args.out) / run
    out.mkdir(parents=True, exist_ok=True)
    np.savez(out / "test_scores.npz", y=y[te], s=s_te, seed_id=d["seed_id"][te], source=src[te])
    report = {"run": run, "arch": name, "args": vars(args),
              "n": {"train": len(fit_idx), "val": len(va), "test": len(te)},
              "threshold": float(thr), "validation": val_m, "test": test_m, "test_ci95": ci,
              "diagnostics": diag, "train_seconds": round(secs, 1),
              "created_utc": datetime.now(timezone.utc).isoformat()}
    (out / "report.json").write_text(json.dumps(report, indent=2))

    print(f"\n=== {run} === (trained on {len(fit_idx):,} chains in {secs:.0f} s)")
    for k in ("fpr", "recall", "precision", "f1", "auc"):
        lo, hi = ci.get(k, [float("nan")] * 2)
        print(f"  test {k:<10} {test_m[k]:.4f}   95% CI [{lo:.4f}, {hi:.4f}]")
    print(f"  precision at 1% base rate {test_m['precision_at_1pct']:.4f}   "
          f"TP={test_m['tp']} FP={test_m['fp']} TN={test_m['tn']} FN={test_m['fn']}")
    for k, v in diag.items():
        print(f"  {k:<28} {v:.4f}")
    return report


def main():
    ap = argparse.ArgumentParser(description="Traditional ML baselines on the same protocol")
    ap.add_argument("--data", default="data/dataset/dataset.npz")
    ap.add_argument("--model", default="all", choices=["rf", "gb", "svm", "all"])
    ap.add_argument("--target-fpr", type=float, default=0.03)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--svm-max-train", type=int, default=15000)
    ap.add_argument("--out", default="results")
    ap.add_argument("--tag", default="trial")
    args = ap.parse_args()

    d = dict(np.load(args.data, allow_pickle=False))
    print(f"[*] {len(d['y']):,} chains from {args.data}")
    for name in (["rf", "gb", "svm"] if args.model == "all" else [args.model]):
        run_one(name, d, args)


if __name__ == "__main__":
    main()
