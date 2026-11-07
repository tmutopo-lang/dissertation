"""
eval_common.py
──────────────
Shared evaluation protocol (Chapter 3, Section 3.5.4), used by train_model.py,
baselines.py and the comparators, so that every model is judged identically:

  temporal_split  oldest 80% train, next 10% validation, newest 10% test
  calibrate       threshold from VALIDATION data: highest recall with FPR <= target,
                  then the lowest FPR among those thresholds
  metrics         FPR, recall, precision, F1, accuracy, AUC-ROC, precision at a 1% base rate
  bootstrap       95% percentile confidence intervals (1,000 resamples of the test set)
"""

import numpy as np
from sklearn.metrics import roc_auc_score


def temporal_split(n, frac=(0.8, 0.1, 0.1)):
    """Rows must already be sorted by crawl time (build_dataset.py does this)."""
    a = int(n * frac[0])
    b = int(n * (frac[0] + frac[1]))
    return np.arange(0, a), np.arange(a, b), np.arange(b, n)


def calibrate(y_val, s_val, target):
    cand = np.unique(s_val)
    if len(cand) == 0:
        return 0.5
    order = np.sort(cand)
    neg = np.sort(s_val[y_val == 0])
    pos = np.sort(s_val[y_val == 1])
    n_neg, n_pos = max(len(neg), 1), max(len(pos), 1)
    fpr = 1 - np.searchsorted(neg, order, side="left") / n_neg
    rec = 1 - np.searchsorted(pos, order, side="left") / n_pos
    ok = fpr <= target
    if not ok.any():
        return float(np.nextafter(order[-1], np.inf))
    best = rec[ok].max()
    return float(order[ok & (rec >= best - 1e-12)].max())


def metrics(y, s, t):
    p = s >= t
    tp = int(((p == 1) & (y == 1)).sum()); fp = int(((p == 1) & (y == 0)).sum())
    tn = int(((p == 0) & (y == 0)).sum()); fn = int(((p == 0) & (y == 1)).sum())
    fpr = fp / max(fp + tn, 1); rec = tp / max(tp + fn, 1)
    prec = tp / max(tp + fp, 1); f1 = 2 * prec * rec / max(prec + rec, 1e-9)
    acc = (tp + tn) / max(len(y), 1)
    auc = roc_auc_score(y, s) if len(set(y)) == 2 else float("nan")
    p1 = rec * 0.01 / max(rec * 0.01 + fpr * 0.99, 1e-12)
    return {"fpr": fpr, "recall": rec, "precision": prec, "f1": f1, "accuracy": acc,
            "auc": float(auc), "precision_at_1pct": p1, "tp": tp, "fp": fp, "tn": tn, "fn": fn}


def bootstrap(y, s, t, n=1000, seed=0):
    rng = np.random.default_rng(seed)
    keys = ("fpr", "recall", "precision", "f1", "auc")
    vals = {k: [] for k in keys}
    for _ in range(n):
        i = rng.integers(0, len(y), len(y))
        if len(set(y[i])) < 2:
            continue
        m = metrics(y[i], s[i], t)
        for k in keys:
            vals[k].append(m[k])
    return {k: [float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))]
            for k, v in vals.items() if v}
