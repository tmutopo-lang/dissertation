"""
summarise_chains.py
───────────────────
Quick health check of a crawl output (JSONL from crawl_chains.py).
Prints the numbers you need after a pilot or at the end of each collection day:
success/failure reasons, how many seeds produced a real redirect chain,
hop-count and mechanism distributions, truncation, and speed.

Usage:
  python scripts/summarise_chains.py data/chains/chains_2026-11-03.jsonl
"""

import json
import sys
from collections import Counter

import pandas as pd


def main(path):
    rows = [json.loads(l) for l in open(path, encoding="utf-8") if l.strip()]
    if not rows:
        print("No records yet.")
        return
    df = pd.DataFrame([{
        "source": r.get("source"),
        "label_hint": str(r.get("label_hint")),
        "status": r.get("status"),
        "n_redirects": r.get("n_redirects", 0),
        "truncated": r.get("truncated", 0),
        "duration_s": r.get("duration_ms", 0) / 1000,
        "download": r.get("downloads_blocked", 0),
    } for r in rows])
    mech = Counter(m for r in rows for m in (r.get("mechanisms") or []))

    print(f"Records: {len(df):,}\n")
    print("Status by label hint (1 = feed says malicious, 0 = Tranco):")
    print(pd.crosstab(df["status"], df["label_hint"], margins=True).to_string(), "\n")

    ok = df[df["status"] == "ok"]
    usable = ok[ok["n_redirects"] >= 1]
    print("Usable chains (ok AND at least one redirect) by source:")
    print(pd.DataFrame({"crawled_ok": ok.groupby("source").size(),
                        "with_redirect": usable.groupby("source").size()}).fillna(0)
          .astype(int).to_string(), "\n")
    print(f"Share of ok crawls that redirect: {len(usable) / max(len(ok), 1):.1%}")
    print(f"Truncated at 15 hops: {int(df['truncated'].sum())}")
    print(f"Downloads blocked: {int(df['download'].sum())}\n")

    print("Redirects per usable chain:")
    print(usable["n_redirects"].value_counts().sort_index().to_string(), "\n")
    print("Redirect mechanisms (all hops):")
    for k, v in mech.most_common():
        print(f"  {k:<14}{v:>8,}")
    print(f"\nMean time per seed: {df['duration_s'].mean():.1f} s")


if __name__ == "__main__":
    main(sys.argv[1])
