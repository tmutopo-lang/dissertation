"""
collect_feeds.py
────────────────
Dissertation: Malicious Redirect Chain Detection
DSRM Activity 3 — Data collection (Chapter 3, Section 3.5.2, pipeline stage 1)

Downloads today's threat-intelligence feeds and a Tranco top-sites list and
writes a dated snapshot that the crawler uses as its seed list:

  data/raw/<YYYY-MM-DD>/
      urlhaus_online.csv      raw feed as downloaded
      phishtank_online.csv    raw feed (if reachable)
      openphish_feed.txt      raw feed
      tranco_<LISTID>.csv     raw Tranco list
      seeds.csv               seed_id, url, label_hint, source, rank, feed_date
      snapshot.json           list ID, download times, counts (for Chapter 4)

label_hint is the feed's claim only. Final labels come later from the
two-source rule plus the 48-hour re-check (label_chains.py).

Run once per collection day, then crawl that day's seeds.csv straight away
(phishing pages go offline quickly).

Usage (Git Bash):
  python scripts/collect_feeds.py --benign 6000
  python scripts/collect_feeds.py --benign 300 --malicious 200   # pilot
"""

import argparse
import csv
import hashlib
import io
import json
import os
import random
import zipfile
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import requests

UA = {"User-Agent": "UJ-MSc-redirect-chain-research/1.0"}
RANK_BANDS = [(1, 1_000), (1_001, 10_000), (10_001, 100_000), (100_001, 1_000_000)]


def sid(url: str) -> str:
    return hashlib.sha1(url.encode("utf-8", "ignore")).hexdigest()[:16]


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def fetch_urlhaus(out: Path, meta: dict) -> pd.DataFrame:
    """URLhaus now needs a free Auth-Key (https://auth.abuse.ch).
    Set it with:  $env:URLHAUS_AUTH_KEY="yourkey"   (PowerShell)"""
    print("[*] URLhaus...")
    key = os.environ.get("URLHAUS_AUTH_KEY", "").strip()
    hdr = dict(UA, **({"Auth-Key": key} if key else {}))
    # 1) API: URLs added in the last 3 days (online ones only)
    if key:
        try:
            r = requests.get("https://urlhaus-api.abuse.ch/v1/urls/recent/", headers=hdr, timeout=60)
            r.raise_for_status()
            data = r.json()
            (out / "urlhaus_recent.json").write_text(json.dumps(data), encoding="utf-8")
            urls = [u["url"] for u in data.get("urls", []) if u.get("url_status") == "online"]
            meta["urlhaus"] = {"downloaded_utc": now(), "endpoint": "api v1/urls/recent",
                               "rows": len(data.get("urls", [])), "online": len(urls)}
            print(f"    {len(urls):,} online URLs (API)")
            return pd.DataFrame({"url": urls, "label_hint": 1, "source": "urlhaus"})
        except Exception as e:
            print(f"    API failed: {e}")
    # 2) bulk exports: newer abuse.ch exports put the key in the path,
    #    older ones used plain download URLs (now 410 Gone)
    eps = []
    if key:
        eps += [f"https://urlhaus-api.abuse.ch/v2/files/exports/{key}/recent.csv",
                f"https://urlhaus-api.abuse.ch/v2/files/exports/recent.csv?auth-key={key}"]
    eps += ["https://urlhaus.abuse.ch/downloads/csv_recent/",
            "https://urlhaus.abuse.ch/downloads/text_online/"]
    for ep in eps:
        try:
            r = requests.get(ep, headers=hdr, timeout=60)
            r.raise_for_status()
            name = "export_recent" if "/exports/" in ep else ep.rstrip("/").split("/")[-1]
            (out / f"urlhaus_{name}.txt").write_text(r.text, encoding="utf-8")
            urls = []
            lines = [l for l in r.text.splitlines() if l.strip() and not l.startswith("#")]
            for row in csv.reader(lines):
                if len(row) >= 4 and row[2].startswith("http"):
                    if row[3].strip().lower() == "online":       # id,dateadded,url,url_status,...
                        urls.append(row[2].strip())
                elif len(row) == 1 and row[0].startswith("http"):  # plain list of URLs
                    urls.append(row[0].strip())
            label = "v2 export" if "/exports/" in ep else ep.split('/')[-2]
            meta["urlhaus"] = {"downloaded_utc": now(), "endpoint": label, "rows": len(urls)}
            print(f"    {len(urls):,} online URLs ({label})")
            return pd.DataFrame({"url": urls, "label_hint": 1, "source": "urlhaus"})
        except Exception as e:
            label = "v2 export" if "/exports/" in ep else ep.split('/')[-2]
            msg = str(e).replace(key, "<key>") if key else str(e)
            print(f"    {label} failed: {msg[:120]}")
    print("    URLhaus skipped (get a free Auth-Key at https://auth.abuse.ch)")
    meta["urlhaus"] = {"error": "no endpoint reachable"}
    return pd.DataFrame(columns=["url", "label_hint", "source"])


def fetch_phishtank(out: Path, meta: dict) -> pd.DataFrame:
    print("[*] PhishTank (online, valid)...")
    key = os.environ.get("PHISHTANK_APP_KEY", "").strip()
    url = (f"http://data.phishtank.com/data/{key}/online-valid.csv" if key
           else "http://data.phishtank.com/data/online-valid.csv")
    try:
        r = requests.get(url, headers=UA, timeout=120)
        r.raise_for_status()
        (out / "phishtank_online.csv").write_text(r.text, encoding="utf-8")
        df_raw = pd.read_csv(io.StringIO(r.text))
        meta["phishtank"] = {"downloaded_utc": now(), "rows": len(df_raw), "app_key_used": bool(key)}
        print(f"    {len(df_raw):,} URLs")
        return pd.DataFrame({"url": df_raw["url"], "label_hint": 1, "source": "phishtank"})
    except Exception as e:
        print(f"    failed (set PHISHTANK_APP_KEY if you have one): {e}")
        meta["phishtank"] = {"error": str(e)[:200]}
        return pd.DataFrame(columns=["url", "label_hint", "source"])


def fetch_openphish(out: Path, meta: dict) -> pd.DataFrame:
    print("[*] OpenPhish (free feed)...")
    try:
        r = requests.get("https://openphish.com/feed.txt", headers=UA, timeout=60)
        r.raise_for_status()
        (out / "openphish_feed.txt").write_text(r.text, encoding="utf-8")
        urls = [l.strip() for l in r.text.splitlines() if l.strip()]
        meta["openphish"] = {"downloaded_utc": now(), "rows": len(urls)}
        print(f"    {len(urls):,} URLs")
        return pd.DataFrame({"url": urls, "label_hint": 1, "source": "openphish"})
    except Exception as e:
        print(f"    failed: {e}")
        meta["openphish"] = {"error": str(e)[:200]}
        return pd.DataFrame(columns=["url", "label_hint", "source"])


def fetch_tranco(out: Path, meta: dict, n: int, seed: int) -> pd.DataFrame:
    """Sample n domains equally across four rank bands (Chapter 3, Section 3.5.2)."""
    print("[*] Tranco (benign seeds)...")
    try:
        list_id = requests.get("https://tranco-list.eu/top-1m-id", headers=UA, timeout=30).text.strip()
        r = requests.get("https://tranco-list.eu/top-1m.csv.zip", headers=UA, timeout=180)
        r.raise_for_status()
        z = zipfile.ZipFile(io.BytesIO(r.content))
        with z.open(z.namelist()[0]) as f:
            df_raw = pd.read_csv(f, header=None, names=["rank", "domain"])
        df_raw.to_csv(out / f"tranco_{list_id}.csv", index=False)
        rng = random.Random(seed)
        per_band = n // len(RANK_BANDS)
        picks = []
        for lo, hi in RANK_BANDS:
            band = df_raw[(df_raw["rank"] >= lo) & (df_raw["rank"] <= hi)]
            idx = rng.sample(range(len(band)), min(per_band, len(band)))
            picks.append(band.iloc[idx])
        df = pd.concat(picks)
        meta["tranco"] = {"list_id": list_id, "downloaded_utc": now(),
                          "sampled": len(df), "per_band": per_band,
                          "bands": RANK_BANDS, "sample_seed": seed}
        print(f"    list ID {list_id}; sampled {len(df):,} domains ({per_band:,} per rank band)")
        return pd.DataFrame({"url": "https://" + df["domain"], "label_hint": 0,
                             "source": "tranco", "rank": df["rank"].astype(int).astype(str).values})
    except Exception as e:
        print(f"    failed: {e}")
        meta["tranco"] = {"error": str(e)[:200]}
        return pd.DataFrame(columns=["url", "label_hint", "source", "rank"])


def previously_seen(raw_root: Path, today: str) -> set:
    """URLs already seeded on earlier days, so each day only adds new seeds."""
    seen = set()
    for d in raw_root.glob("*/seeds.csv"):
        if d.parent.name != today:
            seen.update(pd.read_csv(d, usecols=["url"])["url"].astype(str))
    return seen


def main():
    ap = argparse.ArgumentParser(description="Download feeds and build today's seed list")
    ap.add_argument("--output", default="data/raw", help="root folder for dated snapshots")
    ap.add_argument("--benign", type=int, default=6000, help="Tranco domains to sample today")
    ap.add_argument("--malicious", type=int, default=0,
                    help="cap on malicious seeds today (0 = all new feed URLs)")
    ap.add_argument("--seed", type=int, default=0, help="sampling seed (default: derived from date)")
    args = ap.parse_args()

    today = datetime.now().strftime("%Y-%m-%d")
    root = Path(args.output)
    out = root / today
    out.mkdir(parents=True, exist_ok=True)
    seed = args.seed or int(today.replace("-", ""))
    meta = {"date": today, "started_utc": now()}

    mal = pd.concat([fetch_urlhaus(out, meta), fetch_phishtank(out, meta),
                     fetch_openphish(out, meta)], ignore_index=True)
    mal["url"] = mal["url"].astype(str).str.strip()
    mal = mal[mal["url"].str.match(r"^https?://", case=False)]
    mal = mal.drop_duplicates(subset="url")
    seen = previously_seen(root, today)
    before = len(mal)
    mal = mal[~mal["url"].isin(seen)]
    print(f"[*] Malicious: {before:,} unique feed URLs, {len(mal):,} new since earlier days")
    if args.malicious and len(mal) > args.malicious:
        mal = mal.sample(n=args.malicious, random_state=seed)

    ben = fetch_tranco(out, meta, args.benign, seed)
    ben = ben[~ben["url"].isin(seen)]

    seeds = pd.concat([mal, ben], ignore_index=True)
    seeds["rank"] = seeds.get("rank", pd.Series(dtype="float")).fillna("")
    seeds["seed_id"] = seeds["url"].map(sid)
    seeds["feed_date"] = today
    seeds = seeds[["seed_id", "url", "label_hint", "source", "rank", "feed_date"]]
    seeds.to_csv(out / "seeds.csv", index=False)

    meta["seeds"] = {"total": len(seeds),
                     "by_source": seeds["source"].value_counts().to_dict(),
                     "by_label_hint": {str(k): int(v) for k, v in
                                       seeds["label_hint"].value_counts().items()}}
    meta["finished_utc"] = now()
    (out / "snapshot.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(f"\n[✓] {len(seeds):,} seeds -> {out / 'seeds.csv'}")
    print(seeds["source"].value_counts().to_string())


if __name__ == "__main__":
    main()
