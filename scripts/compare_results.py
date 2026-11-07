"""
compare_results.py
──────────────────
Collects every results/*/report.json into one comparison table
(results/comparison.csv) and prints it, sorted by test FPR.

Usage:
  python scripts/compare_results.py               # all runs
  python scripts/compare_results.py --tag full    # only runs whose tag starts with 'full'
"""

import argparse
import csv
import glob
import json
from pathlib import Path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default="results")
    ap.add_argument("--tag", default="")
    args = ap.parse_args()

    rows = []
    for f in glob.glob(f"{args.results}/*/report.json"):
        r = json.load(open(f))
        if args.tag and not r["run"].startswith(args.tag):
            continue
        t, ci = r["test"], r.get("test_ci95", {})
        rows.append({
            "run": r["run"], "model": r["arch"], "n_test": r["n"]["test"],
            "fpr": t["fpr"], "fpr_ci_hi": ci.get("fpr", [None, None])[1],
            "recall": t["recall"], "precision": t["precision"], "f1": t["f1"],
            "auc": t["auc"], "prec_at_1pct": t["precision_at_1pct"],
            "target_met": ci.get("fpr", [1, 1])[1] < r.get("args", {}).get("target_fpr", 0.03),
        })
    if not rows:
        print("No results found.")
        return
    rows.sort(key=lambda x: (x["fpr"], -x["recall"]))
    out = Path(args.results) / "comparison.csv"
    with open(out, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    print(f"{'run':<42}{'FPR':>8}{'(CI hi)':>9}{'Recall':>8}{'Prec':>8}{'F1':>8}{'AUC':>8}{'P@1%':>8}  met")
    for x in rows:
        print(f"{x['run'][:41]:<42}{x['fpr']:>8.4f}{(x['fpr_ci_hi'] if x['fpr_ci_hi'] is not None else float('nan')):>9.4f}{x['recall']:>8.4f}"
              f"{x['precision']:>8.4f}{x['f1']:>8.4f}{x['auc']:>8.4f}{x['prec_at_1pct']:>8.4f}  "
              f"{'yes' if x['target_met'] else 'no'}")
    print(f"\n[✓] {len(rows)} runs -> {out}")


if __name__ == "__main__":
    main()
