"""
significance.py
───────────────
Dissertation: Malicious Redirect Chain Detection
DSRM Activity 5 — Evaluation (Chapter 3, Section 3.5.4: statistical comparison)

Paired comparison of a reference model (normally the proposed Bi-LSTM-MHA)
against every other run, on the SAME test chains at each model's own frozen
threshold:

  * McNemar's exact test on all test chains (are the two models' errors
    significantly different?)
  * McNemar's exact test on benign test chains only (is the difference in
    false positives significant?)
  * paired bootstrap 95% CI of the FPR difference (other - reference) and of
    the recall difference; an interval that excludes 0 indicates a real difference

Usage:
  python scripts/significance.py --ref full_hybrid --tag full
  (the newest run whose name starts with --ref is used as the reference)
"""

import argparse
import glob
import json
from math import comb
from pathlib import Path

import numpy as np


def mcnemar_exact(a_correct, b_correct):
    """Two-sided exact McNemar p-value from discordant pairs."""
    b = int(np.sum(a_correct & ~b_correct))   # ref right, other wrong
    c = int(np.sum(~a_correct & b_correct))   # ref wrong, other right
    n = b + c
    if n == 0:
        return 1.0, b, c
    k = min(b, c)
    p = 2 * sum(comb(n, i) for i in range(k + 1)) / 2 ** n
    return min(p, 1.0), b, c


def load_run(folder):
    r = json.load(open(Path(folder) / "report.json"))
    d = np.load(Path(folder) / "test_scores.npz", allow_pickle=False)
    return r, {k: d[k] for k in d.files}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default="results")
    ap.add_argument("--ref", default="full_hybrid", help="prefix of the reference run")
    ap.add_argument("--tag", default="full", help="compare only runs starting with this tag")
    ap.add_argument("--boot", type=int, default=2000)
    args = ap.parse_args()

    runs = sorted(glob.glob(f"{args.results}/{args.tag}*/report.json"))
    refs = sorted(p for p in runs if Path(p).parent.name.startswith(args.ref))
    if not refs:
        raise SystemExit(f"No run starting with '{args.ref}' under {args.results}")
    ref_dir = Path(refs[-1]).parent
    R, A = load_run(ref_dir)
    order = np.argsort(A["seed_id"])
    ya, pa = A["y"][order], (A["s"][order] >= R["threshold"])
    sid = A["seed_id"][order]
    print(f"Reference: {ref_dir.name}  (test chains: {len(ya):,})\n")
    print(f"{'compared run':<40}{'McNemar p':>10}{'benign p':>10}"
          f"{'dFPR (other-ref) 95% CI':>30}{'dRecall 95% CI':>26}")

    rng = np.random.default_rng(0)
    rows = []
    for p in runs:
        d = Path(p).parent
        if d == ref_dir:
            continue
        r, B = load_run(d)
        o = np.argsort(B["seed_id"])
        if not np.array_equal(B["seed_id"][o], sid):
            print(f"{d.name[:39]:<40}  (different test chains - skipped)")
            continue
        pb = B["s"][o] >= r["threshold"]
        a_ok, b_ok = pa == ya, pb == ya
        p_all, _, _ = mcnemar_exact(a_ok, b_ok)
        ben = ya == 0
        p_ben, _, _ = mcnemar_exact(a_ok[ben], b_ok[ben])
        mal = ya == 1
        dfpr, drec = [], []
        n = len(ya)
        for _ in range(args.boot):
            i = rng.integers(0, n, n)
            bi, mi = ben[i], mal[i]
            if bi.sum() == 0 or mi.sum() == 0:
                continue
            dfpr.append(pb[i][bi].mean() - pa[i][bi].mean())
            drec.append(pb[i][mi].mean() - pa[i][mi].mean())
        f_lo, f_hi = np.percentile(dfpr, [2.5, 97.5])
        r_lo, r_hi = np.percentile(drec, [2.5, 97.5])
        print(f"{d.name[:39]:<40}{p_all:>10.4f}{p_ben:>10.4f}"
              f"{f'[{f_lo:+.4f}, {f_hi:+.4f}]':>30}{f'[{r_lo:+.4f}, {r_hi:+.4f}]':>26}")
        rows.append({"run": d.name, "mcnemar_p": p_all, "mcnemar_p_benign": p_ben,
                     "dfpr_ci": [float(f_lo), float(f_hi)], "drecall_ci": [float(r_lo), float(r_hi)]})
    out = Path(args.results) / "significance.json"
    out.write_text(json.dumps({"reference": ref_dir.name, "comparisons": rows}, indent=2))
    print(f"\nRead: p < 0.05 means the two models' errors differ significantly. A dFPR interval")
    print(f"entirely above 0 means the other model has a significantly HIGHER FPR than the reference.")
    print(f"[✓] -> {out}")


if __name__ == "__main__":
    main()
