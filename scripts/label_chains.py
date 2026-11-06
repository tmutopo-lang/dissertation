"""
label_chains.py
───────────────
Dissertation: Malicious Redirect Chain Detection
DSRM Activity 3 — Labelling (Chapter 3, Section 3.5.2, pipeline stage 4)

Two steps:

  1) check   – query VirusTotal (API v3) for each usable chain and store the
               verdict counts. Run it twice per chain:
                 --round r1   the same day as the crawl
                 --round r2   at least 48 hours later (re-check)
               Lookups: the landing URL of every chain, plus the seed URL of
               chains whose seed came from a threat feed (if it differs).

  2) finalise – apply the labelling rule and write labels.csv:
       malicious  feed-listed seed AND VirusTotal flags the seed or landing URL
                  (>= --min-engines engines) in r1 or r2  -> two independent sources
       benign     Tranco seed (or Tranco outlink) AND zero malicious/suspicious
                  engines on every looked-up URL in BOTH rounds AND no hop URL in
                  any feed snapshot
       excluded   everything else (sources disagree, not yet re-checked,
                  unknown to VirusTotal) – written to review.csv with a reason

Usage (PowerShell, after $env:VT_API_KEY="..."):
  python scripts/label_chains.py check    --chains data/chains/chains_2026-11-03.jsonl --round r1
  python scripts/label_chains.py check    --chains data/chains/chains_2026-11-03.jsonl --round r2
  python scripts/label_chains.py finalise --chains data/chains/*.jsonl
  python scripts/label_chains.py stats                       # what VirusTotal said so far
"""

import argparse
import base64
import csv
import glob
import json
import os
import sys
import threading
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone, timedelta
from pathlib import Path

import requests

VT_URL = "https://www.virustotal.com/api/v3/urls/{}"
VT_SUBMIT = "https://www.virustotal.com/api/v3/urls"
LOOKUPS = Path("data/labels/vt_lookups.jsonl")


def now():
    return datetime.now(timezone.utc)


def vt_id(url: str) -> str:
    return base64.urlsafe_b64encode(url.encode()).decode().strip("=")


def usable(rec: dict) -> bool:
    return rec.get("status") == "ok" and rec.get("n_redirects", 0) >= 1 and rec.get("hops")


def landing(rec: dict) -> str:
    return rec.get("final_url") or rec["hops"][-1]["url"]


def is_feed_seed(rec: dict) -> bool:
    return rec.get("source") in ("phishtank", "openphish", "urlhaus")


def read_chains(patterns):
    for pat in patterns:
        for path in sorted(glob.glob(pat)):
            with open(path, encoding="utf-8") as f:
                for line in f:
                    if line.strip():
                        yield json.loads(line)


def load_lookups():
    """{(seed_id, round, role): result}"""
    done = {}
    if LOOKUPS.exists():
        for line in open(LOOKUPS, encoding="utf-8"):
            try:
                r = json.loads(line)
                done[(r["seed_id"], r["round"], r["role"])] = r
            except Exception:
                continue
    return done


class VT:
    """Thread-safe: a shared pacing lock keeps the overall rate at --rpm."""

    def __init__(self, key, rpm, daily_cap, submit_unknown=False):
        self.key = key
        self.submit_unknown = submit_unknown
        self.local = threading.local()
        self.min_gap = 60.0 / rpm
        self.next_slot = 0.0
        self.lock = threading.Lock()
        self.daily_cap = daily_cap
        self.used = 0
        self.stop = threading.Event()

    def _session(self):
        if not hasattr(self.local, "s"):
            self.local.s = requests.Session()
            self.local.s.headers.update({"x-apikey": self.key, "accept": "application/json"})
        return self.local.s

    def _pace(self):
        with self.lock:
            if self.stop.is_set():
                raise RuntimeError("stopped")
            if self.used >= self.daily_cap:
                self.stop.set()
                raise RuntimeError("daily cap reached")
            t = time.monotonic()
            slot = max(t, self.next_slot)
            self.next_slot = slot + self.min_gap
            self.used += 1
        if slot > t:
            time.sleep(slot - t)

    def submit(self, url: str) -> int:
        """Ask VirusTotal to scan a URL it has never seen, so the 48 h re-check has a verdict."""
        try:
            self._pace()
            r = self._session().post(VT_SUBMIT, data={"url": url}, timeout=30)
            return 1 if r.status_code == 200 else 0
        except RuntimeError:
            raise
        except Exception:
            return 0

    def lookup(self, url: str) -> dict:
        for attempt in range(5):
            self._pace()
            r = self._session().get(VT_URL.format(vt_id(url)), timeout=30)
            if r.status_code == 200:
                a = r.json()["data"]["attributes"]
                st = a.get("last_analysis_stats", {})
                return {"found": 1,
                        "malicious": st.get("malicious", 0),
                        "suspicious": st.get("suspicious", 0),
                        "harmless": st.get("harmless", 0),
                        "undetected": st.get("undetected", 0),
                        "last_analysis_utc": datetime.fromtimestamp(
                            a.get("last_analysis_date", 0), timezone.utc).isoformat(),
                        "categories": sorted(set((a.get("categories") or {}).values()))[:5]}
            if r.status_code == 404:
                if self.submit_unknown:
                    return {"found": 0, "submitted": self.submit(url)}
                return {"found": 0}
            if r.status_code == 429:
                # VirusTotal uses the same error for the per-minute and the daily
                # limit: wait out a minute a few times before calling it daily.
                if attempt >= 3:
                    self.stop.set()
                    raise RuntimeError("VirusTotal quota still exceeded after waiting "
                                       "(daily limit reached, or key on the free tier)")
                print("    rate limited, waiting 65 s...", flush=True)
                time.sleep(65)
                continue
            if r.status_code in (401, 403):
                self.stop.set()
                raise RuntimeError(f"VirusTotal refused the key ({r.status_code}): {r.text[:200]}")
            time.sleep(5 * (attempt + 1))
        return {"found": -1, "error": f"HTTP {r.status_code}"}


def cmd_check(args):
    key = os.environ.get("VT_API_KEY", "").strip()
    if not key:
        sys.exit('Set your key first:  $env:VT_API_KEY="..."')
    LOOKUPS.parent.mkdir(parents=True, exist_ok=True)
    done = load_lookups()

    # r2 is only allowed once the chain's r1 lookup is >= 48 h old
    r1_time = {k[0]: datetime.fromisoformat(v["checked_utc"])
               for k, v in done.items() if k[1] == "r1" and k[2] == "landing"}

    jobs, waiting = [], 0
    for rec in read_chains(args.chains):
        if not usable(rec):
            continue
        sid = rec["seed_id"]
        if args.round == "r2":
            t1 = r1_time.get(sid)
            if t1 is None or now() - t1 < timedelta(hours=args.gap_hours):
                waiting += 1
                continue
        targets = [("landing", landing(rec))]
        if is_feed_seed(rec) and rec["seed_url"] != landing(rec):
            targets.append(("seed", rec["seed_url"]))
        for role, url in targets:
            if (sid, args.round, role) not in done:
                jobs.append((sid, role, url))

    print(f"[*] Round {args.round}: {len(jobs):,} lookups to do"
          + (f"; {waiting:,} chains not yet {args.gap_hours} h past r1" if args.round == "r2" else ""))
    if not jobs:
        return
    vt = VT(key, args.rpm, args.daily_cap, submit_unknown=(args.round == "r1"))
    t0 = time.monotonic()
    stop_msg = None

    def work(job):
        sid, role, url = job
        try:
            res = vt.lookup(url)
        except RuntimeError as e:
            return job, None, str(e)
        except requests.RequestException as e:
            res = {"found": -1, "error": str(e)[:150]}
        return job, res, None

    n = 0
    with open(LOOKUPS, "a", encoding="utf-8") as fh, \
            ThreadPoolExecutor(max_workers=args.threads) as pool:
        futures = [pool.submit(work, j) for j in jobs]
        for fut in as_completed(futures):
            (sid, role, url), res, err = fut.result()
            if err:
                if stop_msg is None and err != "stopped":
                    stop_msg = err
                continue
            row = {"seed_id": sid, "round": args.round, "role": role, "url": url,
                   "checked_utc": now().isoformat(), **res}
            fh.write(json.dumps(row) + "\n")
            fh.flush()
            n += 1
            if n % 250 == 0 or n == len(jobs):
                rate = n / max(time.monotonic() - t0, 1) * 60
                print(f"    {n:,}/{len(jobs):,}  ~{rate:,.0f} lookups/min", flush=True)
    if stop_msg:
        print(f"[!] Stopped early: {stop_msg}. Re-run later; finished lookups are kept.")
    print(f"[✓] {n:,} lookups stored in {LOOKUPS}  (about {vt.used:,} requests this run)")


def cmd_submit(args):
    """One-off: submit r1 URLs that VirusTotal did not know and were never submitted."""
    key = os.environ.get("VT_API_KEY", "").strip()
    if not key:
        sys.exit('Set your key first:  $env:VT_API_KEY="..."')
    look = load_lookups()
    todo = [r for r in look.values() if r["round"] == "r1" and r.get("found") == 0
            and not r.get("submitted")]
    print(f"[*] {len(todo):,} unknown URLs to submit for scanning")
    vt = VT(key, args.rpm, args.daily_cap)
    ok = 0
    with ThreadPoolExecutor(max_workers=args.threads) as pool:
        for res in pool.map(lambda r: vt.submit(r["url"]) if not vt.stop.is_set() else 0, todo):
            ok += res
    # record submissions so they are not repeated
    with open(LOOKUPS, "a", encoding="utf-8") as fh:
        for r in todo:
            fh.write(json.dumps({**r, "submitted": 1, "submitted_utc": now().isoformat()}) + "\n")
    print(f"[✓] submitted {ok:,} of {len(todo):,}")


def cmd_stats(args):
    look = load_lookups()
    if not look:
        print("No lookups yet.")
        return
    rows = list(look.values())
    print(f"Lookups stored: {len(rows):,}\n")
    for rnd in ("r1", "r2"):
        sub = [r for r in rows if r["round"] == rnd]
        if not sub:
            continue
        print(f"Round {rnd}:")
        for role in ("landing", "seed"):
            rr = [r for r in sub if r["role"] == role]
            if not rr:
                continue
            c = Counter()
            for r in rr:
                if r.get("found") == 0:
                    c["not in VirusTotal"] += 1
                elif r.get("found") == -1:
                    c["lookup error"] += 1
                elif r.get("malicious", 0) >= 2:
                    c["flagged (2+ engines)"] += 1
                elif r.get("malicious", 0) == 1 or r.get("suspicious", 0) > 0:
                    c["weak (1 engine / suspicious)"] += 1
                else:
                    c["clean"] += 1
            print(f"  {role:<8} " + "  ".join(f"{k}: {v:,}" for k, v in c.most_common()))
        print()
    # by seed source (needs the chain files)
    src = {}
    for path in glob.glob("data/chains/*.jsonl"):
        for line in open(path, encoding="utf-8"):
            try:
                r = json.loads(line)
                src[r["seed_id"]] = r.get("source")
            except Exception:
                pass
    if src:
        print("Round r1, landing URL, by seed source:")
        by = {}
        for r in rows:
            if r["round"] == "r1" and r["role"] == "landing":
                s_ = src.get(r["seed_id"], "?")
                k = ("not in VT" if r.get("found") == 0 else "error" if r.get("found") == -1
                     else "flagged" if r.get("malicious", 0) >= 2 else "clean/weak")
                by.setdefault(s_, Counter())[k] += 1
        for s_, c in sorted(by.items()):
            print(f"  {s_:<15} " + "  ".join(f"{k}: {v:,}" for k, v in c.most_common()))


def load_feed_urls():
    """Every URL that appeared in any saved feed snapshot (for the hop check)."""
    urls = set()
    for f in glob.glob("data/raw/*/phishtank_online.csv"):
        try:
            with open(f, encoding="utf-8", errors="ignore") as fh:
                for row in csv.DictReader(fh):
                    if row.get("url"):
                        urls.add(row["url"].strip())
        except Exception:
            pass
    for f in glob.glob("data/raw/*/openphish_feed.txt"):
        urls.update(l.strip() for l in open(f, encoding="utf-8", errors="ignore") if l.strip())
    for f in glob.glob("data/raw/*/urlhaus_*.txt"):
        lines = [l for l in open(f, encoding="utf-8", errors="ignore")
                 if l.strip() and not l.startswith("#")]
        for row in csv.reader(lines):
            for cell in row[:4]:
                if cell.startswith("http"):
                    urls.add(cell.strip())
                    break
    for f in glob.glob("data/raw/*/urlhaus_recent.json"):
        try:
            urls.update(u["url"] for u in json.load(open(f, encoding="utf-8")).get("urls", []))
        except Exception:
            pass
    return urls


def cmd_finalise(args):
    look = load_lookups()
    feed_urls = load_feed_urls()
    out = Path("data/labels")
    out.mkdir(parents=True, exist_ok=True)
    labels, review = [], []
    counts = {"malicious": 0, "benign": 0, "excluded": 0}
    for rec in read_chains(args.chains):
        if not usable(rec):
            continue
        sid = rec["seed_id"]
        rounds = {}
        for rnd in ("r1", "r2"):
            rows = [look.get((sid, rnd, role)) for role in ("landing", "seed")]
            rows = [r for r in rows if r]
            if rows:
                rounds[rnd] = rows
        hop_in_feed = any(h["url"] in feed_urls for h in rec["hops"])
        reason = None
        if "r1" not in rounds or "r2" not in rounds:
            label, reason = None, "awaiting r1/r2"
        elif any(r.get("found") == -1 for rnd in rounds.values() for r in rnd):
            label, reason = None, "lookup error"
        else:
            vt_flag = any(r.get("malicious", 0) >= args.min_engines
                          for rnd in rounds.values() for r in rnd)
            # clean = no detections at crawl time (r1: clean, or not yet known to
            # VirusTotal and submitted) AND a real clean verdict at the re-check (r2)
            def clean(r, need_verdict):
                if r.get("found") == 0:
                    return not need_verdict
                return r.get("malicious", 0) == 0 and r.get("suspicious", 0) == 0
            vt_clean = (all(clean(r, False) for r in rounds["r1"])
                        and all(clean(r, True) for r in rounds["r2"]))
            if is_feed_seed(rec):
                # a phishing page that has been taken down often redirects to a
                # suspended / parked / real-brand page; if VirusTotal confirms the
                # landing page is clean at the re-check, the chain no longer shows
                # malicious behaviour and is excluded rather than labelled 1
                land_r2 = [r for r in rounds["r2"] if r.get("role") == "landing"]
                landing_clean = bool(land_r2) and all(
                    r.get("found") == 1 and r.get("malicious", 0) == 0
                    and r.get("suspicious", 0) == 0 for r in land_r2)
                if not vt_flag:
                    label, reason = None, "feed-listed but VirusTotal below threshold"
                elif landing_clean and landing(rec) != rec["seed_url"]:
                    label, reason = None, "taken down / ends on a clean page"
                else:
                    label = 1
            elif rec.get("source", "").startswith("tranco"):
                if vt_clean and not hop_in_feed:
                    label = 0
                elif hop_in_feed or vt_flag:
                    label, reason = None, "Tranco chain flagged by VirusTotal or a feed"
                else:
                    label, reason = None, "not clean in both rounds (unknown or suspicious)"
            else:
                label, reason = None, "unknown source"
        base = {"seed_id": sid, "seed_url": rec["seed_url"], "source": rec.get("source"),
                "crawl_started_utc": rec.get("crawl_started_utc"),
                "n_redirects": rec.get("n_redirects"), "landing_url": landing(rec)}
        if label is None:
            counts["excluded"] += 1
            review.append({**base, "reason": reason})
        else:
            counts["malicious" if label == 1 else "benign"] += 1
            labels.append({**base, "label": label})

    for name, rows in (("labels.csv", labels), ("review.csv", review)):
        with open(out / name, "w", newline="", encoding="utf-8") as f:
            if rows:
                w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
                w.writeheader()
                w.writerows(rows)
    print(f"[✓] malicious={counts['malicious']:,}  benign={counts['benign']:,}  "
          f"excluded={counts['excluded']:,}  -> {out}/labels.csv, review.csv")
    if review:
        from collections import Counter
        print("    Exclusion reasons:")
        for k, v in Counter(r["reason"] for r in review).most_common():
            print(f"      {k:<52}{v:>7,}")


def main():
    ap = argparse.ArgumentParser(description="VirusTotal labelling of crawled redirect chains")
    sub = ap.add_subparsers(dest="cmd", required=True)
    c = sub.add_parser("check")
    c.add_argument("--chains", nargs="+", required=True)
    c.add_argument("--round", choices=["r1", "r2"], required=True)
    c.add_argument("--gap-hours", type=float, default=48)
    c.add_argument("--rpm", type=float, default=500,
                   help="requests per minute (free tier: 4)")
    c.add_argument("--daily-cap", type=int, default=19500,
                   help="stop before the 20,000/day academic quota")
    c.add_argument("--threads", type=int, default=8,
                   help="parallel lookups (use 1 on the free tier)")
    f = sub.add_parser("finalise")
    f.add_argument("--chains", nargs="+", required=True)
    f.add_argument("--min-engines", type=int, default=2,
                   help="engines that must flag a URL for VirusTotal to count as a source")
    sub.add_parser("stats")
    sm = sub.add_parser("submit")
    sm.add_argument("--rpm", type=float, default=1000)
    sm.add_argument("--daily-cap", type=int, default=19500)
    sm.add_argument("--threads", type=int, default=8)
    args = ap.parse_args()
    {"check": cmd_check, "finalise": cmd_finalise, "stats": cmd_stats,
     "submit": cmd_submit}[args.cmd](args)


if __name__ == "__main__":
    main()
