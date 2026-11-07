"""
make_link_seeds.py
──────────────────
Builds extra benign seeds from the outbound links the crawler recorded on
Tranco pages (shorteners, click trackers, ad / affiliate links, other sites).
Crawling these gives benign chains with realistic multi-hop, cross-domain
redirects, so the model cannot learn "short chain = benign"
(Chapter 3, Section 3.5.2: observed CDN, advertising and e-commerce redirects).

They stay label_hint 0 but are only labelled benign if VirusTotal is clean in
both rounds and no hop appears in a threat feed (label_chains.py).

Usage:
  python scripts/make_link_seeds.py --chains data/chains/chains_2026-11-03.jsonl --max 8000
  python scripts/crawl_chains.py --seeds data/raw/2026-11-04/outlink_seeds.csv \
      --output data/chains/chains_2026-11-04.jsonl --workers 8 --shuffle
"""

import argparse
import csv
import glob
import hashlib
import json
import random
from datetime import datetime
from pathlib import Path


def sid(url):
    return hashlib.sha1(url.encode("utf-8", "ignore")).hexdigest()[:16]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--chains", nargs="+", required=True)
    ap.add_argument("--max", type=int, default=8000, help="outlink seeds to write")
    ap.add_argument("--per-page", type=int, default=3, help="max links taken from one page")
    ap.add_argument("--output", default="data/raw")
    args = ap.parse_args()

    seen = set()
    for f in glob.glob("data/raw/*/seeds*.csv") + glob.glob("data/raw/*/outlink_seeds*.csv"):
        with open(f, encoding="utf-8") as fh:
            seen.update(r["url"] for r in csv.DictReader(fh))

    rows = []
    for pat in args.chains:
        for path in glob.glob(pat):
            for line in open(path, encoding="utf-8"):
                if not line.strip():
                    continue
                r = json.loads(line)
                for url in (r.get("outlinks") or [])[: args.per_page]:
                    if url in seen:
                        continue
                    seen.add(url)
                    rows.append({"seed_id": sid(url), "url": url, "label_hint": 0,
                                 "source": "tranco_outlink", "rank": r.get("tranco_rank", ""),
                                 "parent_seed_id": r.get("seed_id", "")})
    random.Random(7).shuffle(rows)
    rows = rows[: args.max]
    today = datetime.now().strftime("%Y-%m-%d")
    out = Path(args.output) / today
    out.mkdir(parents=True, exist_ok=True)
    for r in rows:
        r["feed_date"] = today
    path, n = out / "outlink_seeds.csv", 1
    while path.exists():                     # second run the same day -> _b2, _b3 ...
        n += 1
        path = out / f"outlink_seeds_b{n}.csv"
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["seed_id", "url", "label_hint", "source", "rank",
                                          "feed_date", "parent_seed_id"])
        w.writeheader()
        w.writerows(rows)
    print(f"[✓] {len(rows):,} outlink seeds -> {path}")
    if not rows:
        print("    (no outlinks found: crawl Tranco seeds with the updated crawler first)")


if __name__ == "__main__":
    main()
