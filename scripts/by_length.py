"""
by_length.py
────────────
Dissertation: Malicious Redirect Chain Detection
DSRM Activity 5 — Evaluation (Chapter 5: analysis by chain length, SQ1)

Most redirect chains are short. Sequence architectures (Bi-LSTM, attention)
are designed for patterns spread across several hops, so differences between
architectures, if any, should appear on the longer chains. This script breaks
every run's test results down by the number of redirects in the chain, at each
run's own frozen threshold:

  1 redirect   (2 URLs)
  2 redirects  (3 URLs)
  3+ redirects (4 or more URLs)

Usage:
  python scripts/by_length.py --tag full
  python scripts/by_length.py --tag full --data data/dataset/dataset.npz
"""

import argparse
import glob
import json
from pathlib import Path

import numpy as np

BUCKETS = [("1 redirect", 2, 2), ("2 redirects", 3, 3), ("3+ redirects", 4, 99)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default="results")
    ap.add_argument("--tag", default="full")
    ap.add_argument("--data", default="data/dataset/dataset.npz",
                    help="dataset the runs were trained on (for chain lengths)")
    args = ap.parse_args()

    d = np.load(args.data, allow_pickle=False)
    length = dict(zip(d["seed_id"], d["lengths"].astype(int)))

    rows = []
    for f in sorted(glob.glob(f"{args.results}/{args.tag}*/report.json")):
        folder = Path(f).parent
        r = json.load(open(f))
        s = np.load(folder / "test_scores.npz", allow_pickle=False)
        L = np.array([length.get(x, 0) for x in s["seed_id"]])
        if (L == 0).any():
            continue                                  # run built on a different dataset
        y, p = s["y"].astype(int), s["s"] >= r["threshold"]
        row = {"run": folder.name}
        for name, lo, hi in BUCKETS:
            i = (L >= lo) & (L <= hi)
            ben, mal = i & (y == 0), i & (y == 1)
            row[name] = {"n_benign": int(ben.sum()), "n_malicious": int(mal.sum()),
                         "fpr": float(p[ben].mean()) if ben.any() else None,
                         "recall": float(p[mal].mean()) if mal.any() else None}
        rows.append(row)
    if not rows:
        raise SystemExit("No matching runs.")

    first = rows[0]
    print("Test chains per bucket: " + "   ".join(
        f"{n}: {first[n]['n_benign']:,} benign / {first[n]['n_malicious']:,} malicious"
        for n, _, _ in BUCKETS))
    print(f"\n{'run':<40}" + "".join(f"{n + ' FPR':>20}{'recall':>9}" for n, _, _ in BUCKETS))
    fmt = lambda v: f"{v:.3f}" if v is not None else "  -  "
    for row in sorted(rows, key=lambda x: (x["3+ redirects"]["fpr"] or 9)):
        print(f"{row['run'][:39]:<40}" + "".join(
            f"{fmt(row[n]['fpr']):>20}{fmt(row[n]['recall']):>9}" for n, _, _ in BUCKETS))
    out = Path(args.results) / "by_length.json"
    out.write_text(json.dumps(rows, indent=2))
    print(f"\nSmall buckets give unstable numbers: read them with the counts above.")
    print(f"[✓] -> {out}")


if __name__ == "__main__":
    main()
