"""
collect_feeds.py
Phase 2 - Data Collection
Hits URLhaus, PhishTank, OpenPhish, and Tranco simultaneously,
deduplicates on URL, and saves a unified feeds_combined.csv with labels.
"""

import os
import argparse
import requests
import pandas as pd
from pathlib import Path
from tqdm import tqdm
import time

# ── URLhaus ────────────────────────────────────────────────────────────────────
def fetch_urlhaus(limit: int) -> pd.DataFrame:
    print("[*] Fetching URLhaus...")
    url = "https://urlhaus.abuse.ch/downloads/csv_recent/"
    try:
        r = requests.get(url, timeout=30)
        r.raise_for_status()
        lines = [l for l in r.text.splitlines() if not l.startswith("#") and l.strip()]
        rows = []
        for line in lines[:limit]:
            parts = line.split('","')
            if len(parts) >= 3:
                raw_url = parts[2].strip('"')
                rows.append({"url": raw_url, "label": 1, "source": "urlhaus"})
        df = pd.DataFrame(rows)
        print(f"    URLhaus: {len(df)} records")
        return df
    except Exception as e:
        print(f"    URLhaus failed: {e}")
        return pd.DataFrame(columns=["url", "label", "source"])


# ── PhishTank ──────────────────────────────────────────────────────────────────
def fetch_phishtank(limit: int) -> pd.DataFrame:
    print("[*] Fetching PhishTank...")
    url = "http://data.phishtank.com/data/online-valid.csv"
    try:
        r = requests.get(url, timeout=60)
        r.raise_for_status()
        from io import StringIO
        df_raw = pd.read_csv(StringIO(r.text))
        df = pd.DataFrame({
            "url": df_raw["url"].head(limit),
            "label": 1,
            "source": "phishtank"
        })
        print(f"    PhishTank: {len(df)} records")
        return df
    except Exception as e:
        print(f"    PhishTank failed: {e}")
        return pd.DataFrame(columns=["url", "label", "source"])


# ── OpenPhish ──────────────────────────────────────────────────────────────────
def fetch_openphish(limit: int) -> pd.DataFrame:
    print("[*] Fetching OpenPhish...")
    url = "https://openphish.com/feed.txt"
    try:
        r = requests.get(url, timeout=30)
        r.raise_for_status()
        urls = [line.strip() for line in r.text.splitlines() if line.strip()][:limit]
        df = pd.DataFrame({"url": urls, "label": 1, "source": "openphish"})
        print(f"    OpenPhish: {len(df)} records")
        return df
    except Exception as e:
        print(f"    OpenPhish failed: {e}")
        return pd.DataFrame(columns=["url", "label", "source"])


# ── Tranco (benign) ────────────────────────────────────────────────────────────
def fetch_tranco(limit: int) -> pd.DataFrame:
    print("[*] Fetching Tranco top domains (benign)...")
    url = "https://tranco-list.eu/top-1m.csv.zip"
    try:
        import zipfile
        from io import BytesIO
        r = requests.get(url, timeout=60)
        r.raise_for_status()
        z = zipfile.ZipFile(BytesIO(r.content))
        with z.open(z.namelist()[0]) as f:
            df_raw = pd.read_csv(f, header=None, names=["rank", "domain"])
        domains = df_raw["domain"].head(limit).tolist()
        urls = [f"https://{d}" for d in domains]
        df = pd.DataFrame({"url": urls, "label": 0, "source": "tranco"})
        print(f"    Tranco: {len(df)} records")
        return df
    except Exception as e:
        print(f"    Tranco failed: {e}")
        return pd.DataFrame(columns=["url", "label", "source"])


# ── Main ───────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="Collect threat intelligence feeds")
    parser.add_argument("--output", default="data/raw", help="Output directory")
    parser.add_argument("--limit", type=int, default=1000, help="Max records per source")
    args = parser.parse_args()

    Path(args.output).mkdir(parents=True, exist_ok=True)

    frames = []
    for fetcher in [fetch_urlhaus, fetch_phishtank, fetch_openphish, fetch_tranco]:
        df = fetcher(args.limit)
        if not df.empty:
            frames.append(df)
        time.sleep(1)  # be polite to APIs

    if not frames:
        print("[!] No data collected. Check your internet connection.")
        return

    combined = pd.concat(frames, ignore_index=True)

    # Deduplicate on URL
    before = len(combined)
    combined.drop_duplicates(subset="url", inplace=True)
    after = len(combined)
    print(f"\n[*] Deduplication: {before} → {after} records")

    # Save
    out_path = Path(args.output) / "feeds_combined.csv"
    combined.to_csv(out_path, index=False)
    print(f"[✓] Saved to {out_path}")
    print(f"\n    Label distribution:")
    print(combined["label"].value_counts().to_string())
    print(f"\n    Source distribution:")
    print(combined["source"].value_counts().to_string())


if __name__ == "__main__":
    main()
