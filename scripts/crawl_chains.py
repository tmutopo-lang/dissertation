"""
crawl_chains.py
───────────────
Dissertation: Malicious Redirect Chain Detection
DSRM Activity 3 — Data collection (Chapter 3, Section 3.5.2, pipeline stage 2)

Visits each seed URL in an instrumented Chromium browser (Playwright) and
records the full redirect chain, hop by hop:

  * HTTP 3xx redirects            (mechanism = "http_301", "http_302", ...)
  * HTML meta-refresh redirects   (mechanism = "meta_refresh")
  * JavaScript navigations        (mechanism = "js")

For every hop it stores the URL, status code, timing, server IP, TLS use,
a few response headers and content signals (forms, password fields, iframes,
script obfuscation indicators). Full page content is NOT stored (Section 3.4.1).

Rules from Chapter 3:
  * 30-second timeout per seed, maximum 15 hops (longer chains truncated + flagged)
  * failures (timeout, DNS error, blocked) are logged with a reason, never silently dropped
  * downloads are refused; each seed gets a fresh, isolated browser context

Output: one JSON object per line (JSONL), appended, so a crash or Ctrl+C
loses nothing and re-running the same command resumes where it stopped.

Usage (Git Bash):
  python scripts/crawl_chains.py --seeds data/raw/2026-11-03/seeds.csv \
      --output data/chains/chains_2026-11-03.jsonl --workers 6
"""

import argparse
import asyncio
import base64
import csv
import hashlib
import json
import logging
import random
import re
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import urljoin

from playwright.async_api import async_playwright

# Windows prints "socket.send() raised exception." for every browser pipe that
# closes during shutdown; it is harmless, so keep the console readable.
logging.getLogger("asyncio").setLevel(logging.CRITICAL)

# Outbound links worth crawling as benign chains: other sites, plus anything
# that looks like a redirector (shorteners, click trackers, ad or affiliate links).
REDIRECTOR_HINTS = ("bit.ly", "t.co/", "lnkd.in", "goo.gl", "tinyurl", "ow.ly", "buff.ly",
                    "rebrand.ly", "/redirect", "redir", "/out", "/go/", "/click", "click.",
                    "track", "url=", "u=http", "dest=", "target=", "doubleclick", "adclick",
                    "aff", "partner", "utm_")
MAX_OUTLINKS = 12

MAX_HOPS = 15
SEED_TIMEOUT_S = 30
QUIET_PERIOD_S = 4          # no new navigation for this long => chain finished
RESTART_BROWSER_EVERY = 300  # seeds per browser instance, avoids memory creep

USER_AGENT = ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
              "(KHTML, like Gecko) Chrome/130.0.0.0 Safari/537.36")

META_REFRESH_RE = re.compile(
    r"""<meta[^>]+http-equiv\s*=\s*["']?refresh["']?[^>]*>""", re.I)
OBFUSCATION_RE = {
    "js_eval": re.compile(r"\beval\s*\(", re.I),
    "js_unescape": re.compile(r"\bunescape\s*\(|\bdecodeURIComponent\s*\(", re.I),
    "js_atob": re.compile(r"\batob\s*\(", re.I),
    "js_fromcharcode": re.compile(r"fromCharCode", re.I),
    "js_document_write": re.compile(r"document\.write\s*\(", re.I),
    "js_location_assign": re.compile(
        r"(window\.|document\.|top\.)?location(\.href)?\s*=|location\.(replace|assign)\s*\(", re.I),
}
KEEP_HEADERS = ("server", "content-type", "location", "refresh", "x-powered-by",
                "cache-control", "strict-transport-security", "set-cookie")


def content_signals(html: str) -> dict:
    """Cheap content features computed in memory; the HTML itself is discarded."""
    low = html.lower()
    sig = {
        "html_bytes": len(html),
        "n_forms": low.count("<form"),
        "n_password_inputs": len(re.findall(r"""type\s*=\s*["']?password""", low)),
        "n_iframes": low.count("<iframe"),
        "n_scripts": low.count("<script"),
        "n_external_scripts": len(re.findall(r"<script[^>]+src\s*=", low)),
        "has_meta_refresh": int(bool(META_REFRESH_RE.search(html))),
        "n_long_strings": len(re.findall(r"[A-Za-z0-9+/=]{200,}", html)),
    }
    for name, rx in OBFUSCATION_RE.items():
        sig[name] = len(rx.findall(html))
    return sig


def seed_id(url: str) -> str:
    return hashlib.sha1(url.encode("utf-8", "ignore")).hexdigest()[:16]


class ChainRecorder:
    """Collects main-frame navigation requests/responses for one page."""

    def __init__(self, page, t0: float):
        self.page = page
        self.t0 = t0
        self.hops = []            # list of dicts, in order
        self.by_request = {}      # request object -> hop dict
        self.last_nav_time = time.monotonic()
        self.pending_bodies = []

    def _is_main_nav(self, request) -> bool:
        try:
            return request.is_navigation_request() and request.frame == self.page.main_frame
        except Exception:
            return False

    def on_request(self, request):
        if not self._is_main_nav(request):
            return
        self.last_nav_time = time.monotonic()
        prev = request.redirected_from
        hop = {
            "url": request.url,
            "t_ms": int((time.monotonic() - self.t0) * 1000),
            "method": request.method,
            "status": None,
            "via": None,          # mechanism that LED to this hop (filled below)
        }
        if self.hops:
            if prev is not None and prev in self.by_request:
                hop["via"] = f"http_{self.by_request[prev].get('status') or '3xx'}"
            else:
                # client-side: meta refresh or JS, decided in settle() once the
                # previous document's HTML has been analysed
                hop["via"] = "client"
        self.hops.append(hop)
        self.by_request[request] = hop

    def on_response(self, response):
        request = response.request
        hop = self.by_request.get(request)
        if hop is None:
            return
        hop["status"] = response.status
        hdrs = {k.lower(): v for k, v in response.headers.items()}
        hop["headers"] = {k: hdrs[k][:300] for k in KEEP_HEADERS if k in hdrs}
        hop["is_https"] = int(response.url.startswith("https://"))
        # status of a request that later redirects is final now; fix 'via' of next hop
        if 300 <= response.status < 400:
            hop["has_meta_refresh"] = 0
        else:
            self.pending_bodies.append(asyncio.ensure_future(self._read_body(response, hop)))
        self.pending_bodies.append(asyncio.ensure_future(self._server_info(response, hop)))

    async def _read_body(self, response, hop):
        if "n_forms" in hop:
            return
        try:
            ctype = response.headers.get("content-type", "")
            if "html" not in ctype and ctype:
                return
            body = await response.text()
            sig = content_signals(body[:2_000_000])
            hop.update(sig)
            hop["meta_refresh_header"] = int("refresh" in {k.lower() for k in response.headers})
            if hop["meta_refresh_header"]:
                hop["has_meta_refresh"] = 1
        except Exception:
            hop["body_unavailable"] = 1

    async def _server_info(self, response, hop):
        try:
            addr = await response.server_addr()
            if addr:
                hop["server_ip"] = addr.get("ipAddress")
        except Exception:
            pass
        try:
            sec = await response.security_details()
            if sec:
                hop["tls_protocol"] = sec.get("protocol")
                hop["tls_issuer"] = (sec.get("issuer") or "")[:120]
                hop["tls_valid_from"] = sec.get("validFrom")
        except Exception:
            pass

    async def attach_cdp(self, context):
        """Pause every document response (CDP Fetch domain) so its HTML can be
        analysed before the browser acts on it. The real network response is
        then released unchanged, so server IP and TLS details are preserved."""
        self.doc_signals = {}
        try:
            self.cdp = await context.new_cdp_session(self.page)
            await self.cdp.send("Fetch.enable", {"patterns": [
                {"resourceType": "Document", "requestStage": "Response"}]})
            self.cdp.on("Fetch.requestPaused",
                        lambda ev: self.pending_bodies.append(
                            asyncio.ensure_future(self._on_paused(ev))))
        except Exception:
            self.cdp = None

    async def _on_paused(self, ev):
        rid = ev["requestId"]
        try:
            status = ev.get("responseStatusCode") or 0
            if 200 <= status < 300:
                hdrs = {h["name"].lower(): h["value"] for h in ev.get("responseHeaders", [])}
                ctype = hdrs.get("content-type", "")
                if not ctype or "html" in ctype:
                    body = await self.cdp.send("Fetch.getResponseBody", {"requestId": rid})
                    raw = body.get("body", "")
                    if body.get("base64Encoded"):
                        raw = base64.b64decode(raw).decode("utf-8", "ignore")
                    sig = content_signals(raw[:2_000_000])
                    if "refresh" in hdrs:
                        sig["has_meta_refresh"] = 1
                    self.doc_signals.setdefault(ev["request"]["url"], []).append(sig)
        except Exception:
            pass
        finally:
            try:
                await self.cdp.send("Fetch.continueRequest", {"requestId": rid})
            except Exception:
                pass

    async def settle(self):
        if self.pending_bodies:
            await asyncio.gather(*self.pending_bodies, return_exceptions=True)
        # attach intercepted content signals to hops (in order, by URL)
        used = {}
        for hop in self.hops:
            lst = getattr(self, "doc_signals", {}).get(hop["url"])
            if lst:
                i = used.get(hop["url"], 0)
                if i < len(lst):
                    hop.update(lst[i])
                    used[hop["url"]] = i + 1
        # resolve client-side mechanisms now that every document is analysed
        for prev, hop in zip(self.hops, self.hops[1:]):
            if hop.get("via") == "client":
                hop["via"] = "meta_refresh" if prev.get("has_meta_refresh") else "js"


async def crawl_one(browser, row: dict) -> dict:
    url = row["url"].strip()
    if not re.match(r"^https?://", url, re.I):
        url = "http://" + url
    started = datetime.now(timezone.utc).isoformat()
    t0 = time.monotonic()
    record = {
        "seed_id": row.get("seed_id") or seed_id(url),
        "seed_url": url,
        "source": row.get("source", ""),
        "label_hint": row.get("label_hint", row.get("label", "")),
        "tranco_rank": row.get("rank", ""),
        "crawl_started_utc": started,
        "status": "ok",
        "error": None,
        "truncated": 0,
        "downloads_blocked": 0,
    }
    context = await browser.new_context(
        user_agent=USER_AGENT,
        viewport={"width": 1366, "height": 768},
        locale="en-ZA",
        timezone_id="Africa/Johannesburg",
        accept_downloads=False,
        ignore_https_errors=True,   # record bad TLS rather than fail on it
        java_script_enabled=True,
    )
    page = await context.new_page()
    rec = ChainRecorder(page, t0)
    await rec.attach_cdp(context)
    page.on("request", rec.on_request)
    page.on("response", rec.on_response)
    page.on("download", lambda d: record.__setitem__("downloads_blocked", 1))
    async def _dismiss(d):
        try:
            await d.dismiss()
        except Exception:
            pass   # page already closed; nothing to dismiss
    page.on("dialog", lambda d: asyncio.ensure_future(_dismiss(d)))

    try:
        try:
            await page.goto(url, wait_until="load", timeout=SEED_TIMEOUT_S * 1000)
        except Exception as e:
            msg = str(e).split("\n")[0]
            if not rec.hops or all(h.get("status") is None for h in rec.hops):
                raise
            record["error"] = "partial: " + msg[:200]
        # keep watching for client-side (meta/JS) redirects until quiet or timeout
        while True:
            elapsed = time.monotonic() - t0
            if elapsed >= SEED_TIMEOUT_S:
                record["error"] = (record["error"] or "") + " seed timeout reached while watching"
                break
            if len(rec.hops) > MAX_HOPS + 1:
                break
            if time.monotonic() - rec.last_nav_time >= QUIET_PERIOD_S:
                break
            await asyncio.sleep(0.5)
        try:
            await page.wait_for_load_state("load", timeout=3000)
        except Exception:
            pass
        await rec.settle()
        try:
            record["final_title"] = (await page.title())[:200]
        except Exception:
            record["final_title"] = None
        record["final_url"] = page.url
        if args_outlinks and str(record["source"]).startswith("tranco"):
            record["outlinks"] = await harvest_outlinks(page)
    except Exception as e:
        msg = str(e).split("\n")[0]
        record["status"] = classify_error(msg)
        record["error"] = msg[:300]
        await rec.settle()
    finally:
        try:
            await context.close()
        except Exception:
            pass

    hops = rec.hops
    if len(hops) > MAX_HOPS + 1:          # 15 hops = 16 URLs
        hops = hops[: MAX_HOPS + 1]
        record["truncated"] = 1
    record["hops"] = hops
    record["n_urls"] = len(hops)
    record["n_redirects"] = max(len(hops) - 1, 0)
    record["mechanisms"] = [h.get("via") for h in hops[1:]]
    record["duration_ms"] = int((time.monotonic() - t0) * 1000)
    return record


args_outlinks = True


async def harvest_outlinks(page) -> list:
    """Links on the landing page that leave the site, redirector-like links first."""
    try:
        hrefs = await page.eval_on_selector_all(
            "a[href]", "els => els.slice(0, 600).map(e => e.href)")
    except Exception:
        return []
    try:
        here = re.sub(r"^www\.", "", (page.url.split("/")[2] if "//" in page.url else "").lower())
    except Exception:
        here = ""
    seen, redir, other = set(), [], []
    for h in hrefs:
        if not isinstance(h, str) or not h.startswith(("http://", "https://")) or h in seen:
            continue
        seen.add(h)
        host = re.sub(r"^www\.", "", h.split("/")[2].lower()) if "//" in h else ""
        low = h.lower()
        if any(k in low for k in REDIRECTOR_HINTS):
            redir.append(h)
        elif host and here and not host.endswith(here) and not here.endswith(host):
            other.append(h)
    return (redir + other)[:MAX_OUTLINKS]


def classify_error(msg: str) -> str:
    m = msg.lower()
    if "err_name_not_resolved" in m:
        return "fail_dns"
    if "timeout" in m:
        return "fail_timeout"
    if "err_connection_refused" in m or "err_connection_reset" in m or "err_connection_closed" in m:
        return "fail_connection"
    if "err_cert" in m or "ssl" in m:
        return "fail_tls"
    if "err_aborted" in m or "download" in m:
        return "fail_aborted_or_download"
    if "err_blocked" in m:
        return "fail_blocked"
    return "fail_other"


def load_seeds(path: Path) -> list:
    with open(path, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    for r in rows:
        r.setdefault("seed_id", seed_id(r["url"]))
        if not r.get("seed_id"):
            r["seed_id"] = seed_id(r["url"])
    return rows


def load_done(path: Path) -> set:
    done = set()
    if path.exists():
        with open(path, encoding="utf-8") as f:
            for line in f:
                try:
                    done.add(json.loads(line)["seed_id"])
                except Exception:
                    continue
    return done


async def run(args):
    seeds = load_seeds(Path(args.seeds))
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    done = load_done(out)
    todo = [r for r in seeds if r["seed_id"] not in done]
    if args.shuffle:
        random.Random(42).shuffle(todo)
    if args.limit:
        todo = todo[: args.limit]
    print(f"[*] {len(seeds):,} seeds, {len(done):,} already crawled, {len(todo):,} to do "
          f"with {args.workers} workers")
    if not todo:
        return

    queue: asyncio.Queue = asyncio.Queue()
    for r in todo:
        queue.put_nowait(r)

    lock = asyncio.Lock()
    stats = {"done": 0, "ok": 0, "fail": 0, "redirected": 0}
    t_start = time.monotonic()
    fh = open(out, "a", encoding="utf-8")

    async with async_playwright() as p:
        async def worker(wid: int):
            browser = None
            used = 0
            while True:
                try:
                    row = queue.get_nowait()
                except asyncio.QueueEmpty:
                    break
                if browser is None or used >= RESTART_BROWSER_EVERY:
                    if browser:
                        await browser.close()
                    browser = await p.chromium.launch(
                        headless=not args.headed,
                        args=["--disable-blink-features=AutomationControlled",
                              "--no-first-run", "--disable-extensions"])
                    used = 0
                try:
                    rec = await asyncio.wait_for(crawl_one(browser, row),
                                                 timeout=SEED_TIMEOUT_S + 20)
                except Exception as e:
                    rec = {"seed_id": row["seed_id"], "seed_url": row["url"],
                           "source": row.get("source", ""),
                           "label_hint": row.get("label_hint", row.get("label", "")),
                           "crawl_started_utc": datetime.now(timezone.utc).isoformat(),
                           "status": "fail_hard_timeout", "error": str(e)[:300],
                           "hops": [], "n_urls": 0, "n_redirects": 0}
                    try:
                        await browser.close()
                    except Exception:
                        pass
                    browser = None
                used += 1
                async with lock:
                    fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
                    fh.flush()
                    stats["done"] += 1
                    if rec["status"] == "ok":
                        stats["ok"] += 1
                        if rec.get("n_redirects", 0) > 0:
                            stats["redirected"] += 1
                    else:
                        stats["fail"] += 1
                    if stats["done"] % 25 == 0 or stats["done"] == len(todo):
                        rate = stats["done"] / max(time.monotonic() - t_start, 1) * 3600
                        print(f"    {stats['done']:,}/{len(todo):,}  ok={stats['ok']:,}  "
                              f"with_redirect={stats['redirected']:,}  failed={stats['fail']:,}  "
                              f"~{rate:,.0f} seeds/hour", flush=True)
            if browser:
                await browser.close()

        await asyncio.gather(*(worker(i) for i in range(args.workers)))
    fh.close()
    print(f"[✓] Finished. Output: {out}")


def main():
    ap = argparse.ArgumentParser(description="Crawl redirect chains with Playwright")
    ap.add_argument("--seeds", required=True, help="seeds.csv from collect_feeds.py")
    ap.add_argument("--output", required=True, help="JSONL file (appended, resumable)")
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--limit", type=int, default=0, help="crawl at most N new seeds (0 = all)")
    ap.add_argument("--shuffle", action="store_true",
                    help="mix malicious and benign seeds so both are collected across time")
    ap.add_argument("--headed", action="store_true", help="show the browser (debugging)")
    ap.add_argument("--no-outlinks", action="store_true",
                    help="do not record outbound links on Tranco pages")
    args = ap.parse_args()
    global args_outlinks
    args_outlinks = not args.no_outlinks
    if sys.platform == "win32":
        asyncio.set_event_loop_policy(asyncio.WindowsProactorEventLoopPolicy())
    asyncio.run(run(args))


if __name__ == "__main__":
    main()
