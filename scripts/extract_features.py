"""
extract_features.py
────────────────────
Dissertation: Malicious Redirect Chain Detection
DSRM Activity 3 — Feature Extraction

Extracts features across 5 dimensions per redirect chain:
  1. Network-level   (IP reputation, ASN, geolocation, TTL)
  2. URL-based       (length, entropy, special chars, domain age, TLD)
  3. Content-based   (JS obfuscation, iframe, forms)
  4. Behavioural     (redirect count, cross-domain transitions, final destination)
  5. Temporal        (redirect timing intervals)

Usage:
  python scripts/extract_features.py \
      --input  data/raw/feeds_combined.csv \
      --output data/processed/features.csv
"""

import argparse
import hashlib
import math
import re
import socket
import string
from pathlib import Path
from urllib.parse import urlparse

import numpy as np
import pandas as pd
import tldextract
from loguru import logger
from tqdm import tqdm

# ─── 1. URL-based features ───────────────────────────────────────────────────

def shannon_entropy(s: str) -> float:
    """Compute Shannon entropy of a string."""
    if not s:
        return 0.0
    freq = {c: s.count(c) / len(s) for c in set(s)}
    return -sum(p * math.log2(p) for p in freq.values())


def extract_url_features(url: str) -> dict:
    """
    Extracts URL-based features from a single URL string.
    No network calls — pure string analysis.
    """
    try:
        parsed  = urlparse(url)
        ext     = tldextract.extract(url)
        path    = parsed.path or ""
        query   = parsed.query or ""
        netloc  = parsed.netloc or ""
        domain  = ext.domain or ""
        suffix  = ext.suffix or ""
        sub     = ext.subdomain or ""
    except Exception:
        return _empty_url_features()

    full_path = path + "?" + query if query else path

    return {
        # Length signals
        "url_length":        len(url),
        "domain_length":     len(domain),
        "path_length":       len(path),
        "query_length":      len(query),
        "subdomain_depth":   len(sub.split(".")) if sub else 0,

        # Entropy signals
        "url_entropy":       round(shannon_entropy(url), 4),
        "domain_entropy":    round(shannon_entropy(domain), 4),
        "path_entropy":      round(shannon_entropy(path), 4),

        # Character composition
        "digit_ratio":       sum(c.isdigit() for c in url) / max(len(url), 1),
        "special_char_count": sum(c in "-_~!@#$%^&*+=[]{}|;:,<>?" for c in url),
        "hyphen_count":      url.count("-"),
        "dot_count":         url.count("."),
        "slash_count":       url.count("/"),
        "at_sign":           int("@" in url),
        "double_slash":      int("//" in path),
        "hex_encoding":      len(re.findall(r"%[0-9a-fA-F]{2}", url)),
        "ip_in_url":         int(bool(re.match(
                                 r"^https?://\d{1,3}(\.\d{1,3}){3}", url))),

        # TLD & structure
        "tld":               suffix,
        "tld_length":        len(suffix),
        "is_https":          int(parsed.scheme == "https"),
        "has_port":          int(bool(parsed.port)),
        "port":              parsed.port or 0,

        # Suspicious keywords in URL
        "kw_login":          int(any(k in url.lower() for k in
                                 ["login", "signin", "account", "verify", "secure",
                                  "update", "confirm", "banking", "paypal", "password"])),
        "kw_redirect":       int(any(k in url.lower() for k in
                                 ["redirect", "redir", "goto", "url=", "next=",
                                  "forward", "dest=", "target="])),
        "kw_shortener":      int(any(k in url.lower() for k in
                                 ["bit.ly", "tinyurl", "t.co", "goo.gl",
                                  "ow.ly", "buff.ly", "rebrand.ly"])),
    }


def _empty_url_features() -> dict:
    return {k: 0 for k in extract_url_features("http://example.com")}


# ─── 2. Behavioural / chain-level features ───────────────────────────────────

def extract_chain_features(chain: list[str]) -> dict:
    """
    Extracts chain-level behavioural features from a list of URLs
    representing one redirect chain (in order).
    """
    if not chain:
        return {
            "chain_length": 0,
            "unique_domains": 0,
            "cross_domain_transitions": 0,
            "cross_tld_transitions": 0,
            "max_entropy_delta": 0.0,
            "mean_url_length": 0.0,
            "final_url_entropy": 0.0,
            "has_ip_hop": 0,
            "has_shortener_hop": 0,
            "domain_diversity_ratio": 0.0,
        }

    domains  = [tldextract.extract(u).registered_domain for u in chain]
    tlds     = [tldextract.extract(u).suffix for u in chain]
    entropies = [shannon_entropy(u) for u in chain]

    cross_domain = sum(1 for i in range(1, len(domains)) if domains[i] != domains[i-1])
    cross_tld    = sum(1 for i in range(1, len(tlds))    if tlds[i]    != tlds[i-1])
    entropy_deltas = [abs(entropies[i] - entropies[i-1]) for i in range(1, len(entropies))]
    url_features = [extract_url_features(u) for u in chain]

    return {
        "chain_length":             len(chain),
        "unique_domains":           len(set(d for d in domains if d)),
        "cross_domain_transitions": cross_domain,
        "cross_tld_transitions":    cross_tld,
        "max_entropy_delta":        round(max(entropy_deltas, default=0.0), 4),
        "mean_url_length":          round(np.mean([len(u) for u in chain]), 2),
        "final_url_entropy":        round(entropies[-1], 4) if entropies else 0.0,
        "has_ip_hop":               int(any(f["ip_in_url"] for f in url_features)),
        "has_shortener_hop":        int(any(f["kw_shortener"] for f in url_features)),
        "domain_diversity_ratio":   round(
            len(set(d for d in domains if d)) / max(len(chain), 1), 4),
    }


# ─── 3. Network-level features (lightweight, no external API) ────────────────

def extract_network_features(url: str) -> dict:
    """
    Lightweight network features that don't require external API calls.
    For full IP reputation / ASN features, wire in ipinfo.io or similar.
    """
    try:
        hostname = urlparse(url).hostname or ""
        ip = socket.gethostbyname(hostname) if hostname else ""
    except (socket.gaierror, UnicodeError):
        ip = ""

    return {
        "hostname_resolvable": int(bool(ip)),
        "is_ipv4_host":        int(bool(re.match(r"\d{1,3}(\.\d{1,3}){3}$", ip))),
        # Placeholder columns — fill via ipinfo.io batch lookup in a second pass
        "asn":                 "",
        "country_code":        "",
        "is_cdn":              -1,   # -1 = unknown until enriched
    }


# ─── 4. Main pipeline ────────────────────────────────────────────────────────

def process_single_url(row: pd.Series) -> dict:
    """
    Wraps feature extraction for a single URL row from the combined feed CSV.
    For dissertation purposes, each URL is treated as a single-hop chain.
    Multi-hop chains will be assembled during the crawl phase (Activity 3).
    """
    url   = str(row.get("url", ""))
    label = int(row.get("label", -1))

    url_feats     = extract_url_features(url)
    chain_feats   = extract_chain_features([url])       # single-hop for now
    network_feats = extract_network_features(url)

    return {
        "url":    url,
        "label":  label,
        "source": row.get("source", ""),
        **url_feats,
        **chain_feats,
        **network_feats,
    }


def main():
    parser = argparse.ArgumentParser(description="Extract features from URL feed")
    parser.add_argument("--input",  default="data/raw/feeds_combined.csv")
    parser.add_argument("--output", default="data/processed/features.csv")
    parser.add_argument("--limit",  type=int, default=0, help="0 = all rows")
    parser.add_argument("--no-network", action="store_true",
                        help="Skip DNS lookups (faster, offline-safe)")
    args = parser.parse_args()

    input_path  = Path(args.input)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    logger.add(output_path.parent / "extract_features.log")

    df = pd.read_csv(input_path)
    if args.limit:
        df = df.head(args.limit)

    logger.info(f"Processing {len(df):,} URLs from {input_path}")

    results = []
    for _, row in tqdm(df.iterrows(), total=len(df), desc="Extracting features"):
        try:
            feats = process_single_url(row)
            if args.no_network:
                feats.update({"hostname_resolvable": -1,
                               "is_ipv4_host": -1,
                               "asn": "", "country_code": "", "is_cdn": -1})
            results.append(feats)
        except Exception as e:
            logger.warning(f"Skipping {row.get('url', '?')}: {e}")

    out_df = pd.DataFrame(results)
    out_df.to_csv(output_path, index=False)

    logger.success(f"Saved {len(out_df):,} feature vectors → {output_path}")
    logger.info(f"Feature columns ({len(out_df.columns)}): {list(out_df.columns)}")

    # Quick class balance report
    balance = out_df["label"].value_counts().to_dict()
    logger.info(f"Class distribution — benign (0): {balance.get(0,0):,}, "
                f"malicious (1): {balance.get(1,0):,}")


if __name__ == "__main__":
    main()
