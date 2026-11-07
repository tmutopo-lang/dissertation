"""
build_dataset.py
────────────────
Dissertation: Malicious Redirect Chain Detection
DSRM Activity 3 — Feature extraction (Chapter 3, Section 3.5.2, pipeline stage 3)

Turns every labelled redirect chain into an ordered sequence of hop feature
vectors (one vector per URL in the chain), covering the five dimensions of
the feature framework:

  URL         lexical properties of each hop's URL
  Network     status codes, domain / IP changes between hops, TLS
  Behavioural how the hop was reached (HTTP 3xx, meta refresh, JavaScript),
              response headers
  Content     forms, password fields, iframes, scripts, obfuscation indicators
  Temporal    time since the previous hop, elapsed time, position in the chain

Labels come from data/labels/labels.csv (label_chains.py finalise).
VirusTotal verdicts, feed membership and Tranco rank are NEVER used as
features, because they are (or are close to) the label itself.

Exact duplicate chains (same sequence of hop URLs) are removed, keeping the
earliest crawl. The crawl time of each chain is stored for the temporal split.

Output: data/dataset/dataset.npz + features.json + build_report.json

Usage:
  python scripts/build_dataset.py --chains "data/chains/*.jsonl"
  python scripts/build_dataset.py --chains "data/chains/*.jsonl" --seed-url drop
"""

import argparse
import glob
import json
import math
import re
from collections import Counter
from datetime import datetime
from pathlib import Path
from urllib.parse import urlsplit, parse_qsl

import numpy as np
import pandas as pd

try:
    import tldextract
    _EXTRACT = tldextract.TLDExtract(suffix_list_urls=())   # bundled list, no network
except Exception:                                           # pragma: no cover
    _EXTRACT = None

MAX_URLS = 16            # 15 hops = 16 URLs (Chapter 3 hop limit)

SUSPICIOUS_WORDS = ("login", "signin", "sign-in", "logon", "verify", "verification", "account",
                    "secure", "security", "update", "bank", "confirm", "password", "wallet",
                    "webscr", "billing", "invoice", "support", "unlock", "auth", "session")
SHORTENERS = {"bit.ly", "t.co", "tinyurl.com", "goo.gl", "ow.ly", "buff.ly", "is.gd", "rebrand.ly",
              "cutt.ly", "lnkd.in", "t.ly", "rb.gy", "shorturl.at", "tiny.cc", "s.id", "bl.ink",
              "short.io", "trib.al", "dlvr.it", "amzn.to", "youtu.be", "fb.me", "wp.me"}
RISKY_TLDS = {"xyz", "top", "club", "online", "site", "icu", "live", "shop", "buzz", "cfd",
              "rest", "monster", "click", "link", "work", "fit", "tk", "ml", "ga", "cf", "gq",
              "cn", "ru", "zip", "mov", "info", "pw", "cam", "sbs", "lol", "quest"}
REDIRECT_PARAMS = {"url", "u", "redirect", "redirect_uri", "redirect_url", "next", "dest",
                   "destination", "target", "goto", "out", "continue", "return", "returnurl", "r"}
MECHANISMS = ["start", "http_301", "http_302", "http_303", "http_307", "http_308",
              "meta_refresh", "js", "other"]
CONTENT_KEYS = ["n_forms", "n_password_inputs", "n_iframes", "n_scripts", "n_external_scripts",
                "has_meta_refresh", "n_long_strings", "js_eval", "js_unescape", "js_atob",
                "js_fromcharcode", "js_document_write", "js_location_assign"]
IPV4 = re.compile(r"^\d{1,3}(\.\d{1,3}){3}$")


def entropy(s: str) -> float:
    if not s:
        return 0.0
    c = Counter(s)
    n = len(s)
    return -sum(v / n * math.log2(v / n) for v in c.values())


def reg_domain(host: str) -> str:
    host = (host or "").lower().strip(".")
    if not host or IPV4.match(host):
        return host
    if _EXTRACT:
        e = _EXTRACT(host)
        if e.domain:
            return f"{e.domain}.{e.suffix}" if e.suffix else e.domain
    parts = host.split(".")
    return ".".join(parts[-2:]) if len(parts) >= 2 else host


def l1p(x) -> float:
    try:
        return math.log1p(max(float(x), 0.0))
    except Exception:
        return 0.0


# ─── feature names (fixed order) ─────────────────────────────────────────────
URL_F = ["url_len", "host_len", "path_len", "query_len", "host_dots", "host_hyphens",
         "digit_ratio", "n_subdomains", "path_depth", "n_params", "host_is_ip", "has_at",
         "has_port", "is_https", "url_entropy", "host_entropy", "n_susp_words",
         "risky_tld", "is_shortener", "has_redirect_param"]
NET_F = ["status_2xx", "status_3xx", "status_4xx", "status_5xx", "status_none",
         "host_change", "regdom_change", "same_regdom_as_seed", "same_regdom_as_landing",
         "ip_change", "has_ip", "tls", "tls_letsencrypt"]
BEH_F = [f"via_{m}" for m in MECHANISMS] + ["hdr_set_cookie", "hdr_x_powered_by",
                                             "hdr_server", "hdr_hsts", "ctype_html"]
CON_F = ["has_content", "html_bytes"] + CONTENT_KEYS
TMP_F = ["dt_prev", "t_elapsed", "position", "is_first", "is_last", "chain_len"]
FEATURES = ([("url", f) for f in URL_F] + [("network", f) for f in NET_F] +
            [("behavioural", f) for f in BEH_F] + [("content", f) for f in CON_F] +
            [("temporal", f) for f in TMP_F])
NAMES = [f"{d}:{f}" for d, f in FEATURES]
URL_IDX = [i for i, (d, _) in enumerate(FEATURES) if d == "url"]


def hop_features(hops: list) -> np.ndarray:
    n = len(hops)
    seed_rd = reg_domain(urlsplit(hops[0]["url"]).hostname or "")
    land_rd = reg_domain(urlsplit(hops[-1]["url"]).hostname or "")
    out = np.zeros((n, len(FEATURES)), dtype=np.float32)
    prev_host = prev_rd = prev_ip = None
    prev_t = 0
    for i, h in enumerate(hops):
        url = h.get("url", "")
        sp = urlsplit(url)
        host = (sp.hostname or "").lower()
        rd = reg_domain(host)
        q = parse_qsl(sp.query, keep_blank_values=True)
        low = url.lower()
        status = h.get("status")
        hdr = h.get("headers") or {}
        via = h.get("via") or "start"
        if i == 0:
            via = "start"
        if via not in MECHANISMS:
            via = "other"
        tld = rd.rsplit(".", 1)[-1] if "." in rd else ""
        t = h.get("t_ms") or 0
        f = {
            # URL
            "url_len": l1p(len(url)), "host_len": l1p(len(host)), "path_len": l1p(len(sp.path)),
            "query_len": l1p(len(sp.query)), "host_dots": host.count("."),
            "host_hyphens": host.count("-"),
            "digit_ratio": sum(ch.isdigit() for ch in url) / max(len(url), 1),
            "n_subdomains": max(len(host.split(".")) - len(rd.split(".")), 0) if host else 0,
            "path_depth": len([p for p in sp.path.split("/") if p]),
            "n_params": l1p(len(q)), "host_is_ip": float(bool(IPV4.match(host))),
            "has_at": float("@" in sp.netloc), "has_port": float(sp.port not in (None, 80, 443)
                                                                 if _safe_port(sp) else 0),
            "is_https": float(sp.scheme == "https"), "url_entropy": entropy(url),
            "host_entropy": entropy(host),
            "n_susp_words": sum(w in low for w in SUSPICIOUS_WORDS),
            "risky_tld": float(tld in RISKY_TLDS), "is_shortener": float(host.removeprefix("www.") in SHORTENERS),
            "has_redirect_param": float(any(k.lower() in REDIRECT_PARAMS for k, _ in q)),
            # Network
            "status_2xx": float(status is not None and 200 <= status < 300),
            "status_3xx": float(status is not None and 300 <= status < 400),
            "status_4xx": float(status is not None and 400 <= status < 500),
            "status_5xx": float(status is not None and status >= 500),
            "status_none": float(status is None),
            "host_change": float(prev_host is not None and host != prev_host),
            "regdom_change": float(prev_rd is not None and rd != prev_rd),
            "same_regdom_as_seed": float(rd == seed_rd), "same_regdom_as_landing": float(rd == land_rd),
            "ip_change": float(prev_ip is not None and h.get("server_ip") not in (None, prev_ip)),
            "has_ip": float(bool(h.get("server_ip"))), "tls": float(bool(h.get("tls_protocol"))),
            "tls_letsencrypt": float("let's encrypt" in (h.get("tls_issuer") or "").lower()
                                     or (h.get("tls_issuer") or "").startswith(("R1", "E5", "E6", "R10", "R11"))),
            # Behavioural
            **{f"via_{m}": float(via == m) for m in MECHANISMS},
            "hdr_set_cookie": float("set-cookie" in hdr), "hdr_x_powered_by": float("x-powered-by" in hdr),
            "hdr_server": float("server" in hdr), "hdr_hsts": float("strict-transport-security" in hdr),
            "ctype_html": float("html" in (hdr.get("content-type") or "")),
            # Content
            "has_content": float("n_forms" in h), "html_bytes": l1p(h.get("html_bytes", 0)),
            **{k: l1p(h.get(k, 0)) for k in CONTENT_KEYS},
            # Temporal
            "dt_prev": l1p(t - prev_t), "t_elapsed": l1p(t),
            "position": i / max(n - 1, 1), "is_first": float(i == 0), "is_last": float(i == n - 1),
            "chain_len": n / MAX_URLS,
        }
        out[i] = [f[name] for _, name in FEATURES]
        prev_host, prev_rd, prev_t = host, rd, t
        prev_ip = h.get("server_ip") or prev_ip
    return out


def _safe_port(sp) -> bool:
    try:
        sp.port
        return True
    except ValueError:
        return False


def main():
    ap = argparse.ArgumentParser(description="Build hop-sequence dataset from labelled chains")
    ap.add_argument("--chains", nargs="+", required=True, help='e.g. "data/chains/*.jsonl"')
    ap.add_argument("--labels", default="data/labels/labels.csv")
    ap.add_argument("--output", default="data/dataset")
    ap.add_argument("--seed-url", choices=["keep", "drop"], default="keep",
                    help="'drop' zeroes the URL features of the first hop (shortcut check: "
                         "benign Tranco seeds are bare homepages, malicious seeds are deep links)")
    ap.add_argument("--url-features", choices=["all", "none"], default="all",
                    help="'none' zeroes the URL features of EVERY hop: the model then sees only "
                         "chain structure, network, behaviour, content and timing")
    args = ap.parse_args()

    labels = pd.read_csv(args.labels)
    lab = dict(zip(labels["seed_id"], labels["label"]))
    print(f"[*] {len(lab):,} labelled chains in {args.labels}")

    recs, seen_seq, seen_sid, dup, dup_seed, missing = [], {}, {}, 0, 0, 0
    for pat in args.chains:
        for path in sorted(glob.glob(pat)):
            for line in open(path, encoding="utf-8"):
                if not line.strip():
                    continue
                r = json.loads(line)
                if r.get("seed_id") not in lab:
                    continue
                hops = (r.get("hops") or [])[:MAX_URLS]
                if len(hops) < 2:
                    missing += 1
                    continue
                key = tuple(h["url"] for h in hops)
                t = r.get("crawl_started_utc") or ""
                sid_ = r["seed_id"]
                if sid_ in seen_sid:                  # same seed crawled twice: keep earliest
                    dup_seed += 1
                    j = seen_sid[sid_]
                    if recs[j] is not None and t < recs[j]["t"]:
                        old_key = tuple(h["url"] for h in recs[j]["hops"])
                        seen_seq.pop(old_key, None)
                        recs[j] = {"r": r, "hops": hops, "t": t}
                        seen_seq[key] = j
                    continue
                if key in seen_seq:
                    dup += 1
                    j = seen_seq[key]
                    if t < recs[j]["t"]:          # keep the earliest crawl
                        recs[j] = {"r": r, "hops": hops, "t": t}
                    continue
                seen_seq[key] = len(recs)
                seen_sid[sid_] = len(recs)
                recs.append({"r": r, "hops": hops, "t": t})

    recs.sort(key=lambda x: x["t"])
    N, F = len(recs), len(FEATURES)
    X = np.zeros((N, MAX_URLS, F), dtype=np.float32)
    L = np.zeros(N, dtype=np.int16)
    y = np.zeros(N, dtype=np.int8)
    ts = np.zeros(N, dtype=np.float64)
    sid, src, seed_urls, land_urls = [], [], [], []
    for i, x in enumerate(recs):
        feats = hop_features(x["hops"])
        if args.seed_url == "drop":
            feats[0, URL_IDX] = 0.0
        if args.url_features == "none":
            feats[:, URL_IDX] = 0.0
        X[i, : len(feats)] = feats
        L[i] = len(feats)
        y[i] = int(lab[x["r"]["seed_id"]])
        ts[i] = datetime.fromisoformat(x["t"]).timestamp() if x["t"] else 0
        sid.append(x["r"]["seed_id"])
        src.append(x["r"].get("source", ""))
        seed_urls.append(x["hops"][0]["url"][:500])     # for the single-URL comparators
        land_urls.append(x["hops"][-1]["url"][:500])

    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out / "dataset.npz", X=X, lengths=L, y=y, t=ts,
                        seed_id=np.array(sid), source=np.array(src),
                        seed_url=np.array(seed_urls), landing_url=np.array(land_urls))
    (out / "features.json").write_text(json.dumps({"names": NAMES, "max_urls": MAX_URLS,
                                                   "seed_url": args.seed_url,
                                                   "url_features": args.url_features}, indent=2))
    report = {"chains": N, "malicious": int(y.sum()), "benign": int(N - y.sum()),
              "malicious_share": round(float(y.mean()), 4) if N else 0,
              "duplicates_removed": dup, "repeat_seeds_removed": dup_seed, "too_short": missing, "features_per_hop": F,
              "by_source": dict(Counter(src)), "first_crawl": recs[0]["t"] if N else None,
              "last_crawl": recs[-1]["t"] if N else None, "seed_url": args.seed_url,
              "url_features": args.url_features}
    (out / "build_report.json").write_text(json.dumps(report, indent=2))
    print(f"[✓] {N:,} chains  ({report['malicious']:,} malicious, {report['benign']:,} benign, "
          f"{report['malicious_share']:.1%} malicious)  {F} features per hop")
    print(f"    duplicate chains removed: {dup:,}   repeat seeds removed: {dup_seed:,}")
    print(f"    by source: {report['by_source']}")
    print(f"    -> {out / 'dataset.npz'}")


if __name__ == "__main__":
    main()
