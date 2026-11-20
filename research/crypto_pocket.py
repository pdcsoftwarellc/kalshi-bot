"""Out-of-sample test of the "97-99c favourite five minutes before close" lead in Kalshi 15-minute crypto markets.

Research only: public unauthenticated Kalshi reads, no orders. The rule, strata, statistics and verdict are frozen in
data/processed/crypto-pocket/prereg.txt (written before any new window was read).

  series   fee type and multiplier for the nine series
  listing  settled markets (live tier from 2026-09-10, historical tier pages for the original four coins)
  candles  one-minute bars close-6m..close-2m for every NEW live-tier market, batched across coins by clock window
  tape     trade prints in the minute after signal + 60 s for 300 random new bets
  hist     per-market historical candles for a seeded random order of pre-2026-09-10 markets, until the budget ends
  score    the frozen rule on new windows, then pooled with the original sample; writes report.txt and results.json

Every request is cached in slim form under data/raw/crypto-pocket/req, so each step resumes where it stopped.
At most one request every 5 s (shared through a lock file), 60-300 s back-off on HTTP 429.
"""
import argparse
from collections import defaultdict
from datetime import datetime, timezone
import fcntl
import hashlib
import json
import math
from pathlib import Path
import random
import sys
import time

import httpx
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data/raw/crypto-pocket"
OUT = ROOT / "data/processed/crypto-pocket"
OLD = ROOT / "data/raw/short-dated/v2"
BASE = "https://external-api.kalshi.com/trade-api/v2"
ORIGINAL = {"KXBTC15M": "btc15m", "KXETH15M": "eth15m", "KXSOL15M": "sol15m", "KXXRP15M": "xrp15m"}
OTHERS = ("KXDOGE15M", "KXBNB15M", "KXHYPE15M", "KXZEC15M", "KXNEAR15M")
SERIES = tuple(ORIGINAL) + OTHERS
LIVE_START = 1788998400          # 2026-09-10 00:00 UTC
LIST_END = 1795122000            # 2026-11-19 21:00 UTC, frozen in the prereg
HARD_STOP = 1795141560           # 2026-11-20 02:26 UTC: five hours after the work started
GAP = 5.0
BAND = (97.0, 99.0)
MAX_SPREAD = 5.0
HIST_PAGES = 10                  # 1,000 markets a page, about 10 days a coin

_stop = [HARD_STOP]


def utc(ts, fmt="%Y-%m-%d %H:%M"):
    return datetime.fromtimestamp(ts, timezone.utc).strftime(fmt)


def stamp(value):
    return datetime.fromisoformat(value.replace("Z", "+00:00")).timestamp()


def pace():
    """Shared 5 s spacing between requests of every copy of this script."""
    RAW.mkdir(parents=True, exist_ok=True)
    with open(RAW / ".pace", "a+") as f:
        fcntl.flock(f, fcntl.LOCK_EX)
        f.seek(0)
        text = f.read().strip()
        wait = (float(text) if text else 0.0) + GAP - time.time()
        if wait > 0:
            time.sleep(wait)
        f.seek(0)
        f.truncate()
        f.write(str(time.time()))
        f.flush()


def get(client, path, params, slim, ok404=False):
    key = hashlib.sha256(json.dumps([path, params], sort_keys=True).encode()).hexdigest()
    cache = RAW / "req" / (key + ".json")
    if cache.exists():
        return json.loads(cache.read_text())
    backoff = 60
    for attempt in range(12):
        if time.time() > _stop[0]:
            raise TimeoutError("download budget used")
        pace()
        try:
            resp = client.get(BASE + path, params=params)
        except httpx.RequestError:
            time.sleep(20)
            continue
        if resp.status_code == 429:
            print(f"  HTTP 429, sleeping {backoff}s", flush=True)
            time.sleep(backoff)
            backoff = min(backoff + 60, 300)
            continue
        if resp.status_code >= 500:
            time.sleep(30)
            continue
        if resp.status_code == 404 and ok404:
            data = None
        elif resp.status_code != 200:
            raise RuntimeError(f"{path} returned HTTP {resp.status_code}: {resp.text[:200]}")
        else:
            data = slim(resp.json())
        cache.parent.mkdir(parents=True, exist_ok=True)
        cache.write_text(json.dumps(data, separators=(",", ":")))
        return data
    raise RuntimeError(f"gave up: {path}")


def client():
    return httpx.Client(timeout=90, headers={"User-Agent": "research-client"})


# ---------------------------------------------------------------- download

def slim_series(data):
    s = data["series"]
    return dict(ticker=s["ticker"], fee_type=s.get("fee_type"), fee_multiplier=s.get("fee_multiplier"), title=s.get("title"))


def slim_markets(data):
    rows = []
    for m in data.get("markets", []):
        try:
            result = m.get("result")
            if result not in ("yes", "no"):
                continue
            rows.append(dict(ticker=m["ticker"], close=int(stamp(m["close_time"])), outcome=1.0 if result == "yes" else 0.0,
                             floor=m.get("floor_strike"), final=float(m["expiration_value"]) if m.get("expiration_value") else None,
                             volume=float(m.get("volume_fp") or 0)))
        except (KeyError, TypeError, ValueError):
            continue
    return dict(rows=rows, cursor=data.get("cursor") or "", count=len(data.get("markets", [])))


def num(obj, field):
    for name in (field + "_dollars", field):
        try:
            return round(100 * float(obj[name]), 2)
        except (KeyError, TypeError, ValueError):
            continue
    return None


def compact(bar):
    """[end ts, yes bid close, yes ask close, trade low, trade high, volume] in cents."""
    volume = bar.get("volume_fp", bar.get("volume")) or 0
    return [bar["end_period_ts"], num(bar.get("yes_bid") or {}, "close"), num(bar.get("yes_ask") or {}, "close"),
            num(bar.get("price") or {}, "low"), num(bar.get("price") or {}, "high"), float(volume)]


def slim_batch(data):
    return {m["market_ticker"]: [compact(b) for b in m["candlesticks"]] for m in data.get("markets", [])}


def slim_single(data):
    return [compact(b) for b in data.get("candlesticks", [])]


def slim_trades(data):
    return dict(rows=[[stamp(t["created_time"]), round(100 * float(t["yes_price_dollars"]), 2), float(t["count_fp"]), t.get("taker_side")]
                      for t in data.get("trades", [])], cursor=data.get("cursor") or "")


def do_series():
    with client() as c:
        rows = [get(c, f"/series/{s}", {}, slim_series) for s in SERIES]
    (RAW / "series.json").write_text(json.dumps(rows, indent=1))
    for r in rows:
        print(r)


def do_listing():
    with client() as c:
        for s in SERIES:
            out = RAW / f"listing-{s}.json"
            if out.exists():
                continue
            rows, cursor = [], ""
            while True:
                params = dict(series_ticker=s, status="settled", limit=1000, min_close_ts=LIVE_START, max_close_ts=LIST_END)
                page = get(c, "/markets", dict(params, cursor=cursor) if cursor else params, slim_markets)
                rows += page["rows"]
                cursor = page["cursor"]
                if not cursor or not page["count"]:
                    break
            out.write_text(json.dumps(rows, separators=(",", ":")))
            print(s, "live listing", len(rows), utc(min(r["close"] for r in rows)) if rows else "", flush=True)
        for s in ORIGINAL:
            out = RAW / f"hist-listing-{s}.json"
            if out.exists():
                continue
            rows, cursor = [], ""
            for _ in range(HIST_PAGES):
                params = dict(series_ticker=s, limit=1000)
                page = get(c, "/historical/markets", dict(params, cursor=cursor) if cursor else params, slim_markets)
                rows += page["rows"]
                cursor = page["cursor"]
                if not cursor or not page["count"]:
                    break
            out.write_text(json.dumps(rows, separators=(",", ":")))
            print(s, "historical listing", len(rows), utc(min(r["close"] for r in rows)) if rows else "", flush=True)


def original_tickers():
    return {m["ticker"] for name in ORIGINAL.values() for m in json.loads((OLD / f"{name}.json").read_text())}


def listed(prefix="listing"):
    rows = {}
    for s in SERIES:
        path = RAW / f"{prefix}-{s}.json"
        if path.exists():
            for m in json.loads(path.read_text()):
                rows[m["ticker"]] = dict(m, series=s)
    return rows


def jobs_live():
    """Requests grouped by clock window across coins: at most 100 tickers and about 9,000 requested bars."""
    old = original_tickers()
    by_close = defaultdict(list)
    for m in listed().values():
        if m["ticker"] not in old and LIVE_START <= m["close"] < LIST_END:
            by_close[m["close"]].append(m["ticker"])
    jobs, chunk, first = [], [], None
    for close in sorted(by_close):
        names = by_close[close]
        if chunk:
            size = len(chunk) + len(names)
            if size > 100 or size * ((close - first) // 60 + 6) > 9000:
                jobs.append((first, last, chunk))
                chunk = []
        if not chunk:
            first = close
        chunk = chunk + names
        last = close
    if chunk:
        jobs.append((first, last, chunk))
    return jobs


def do_candles():
    jobs = jobs_live()
    print("live candle requests", len(jobs), "markets", sum(len(j[2]) for j in jobs), flush=True)
    with client() as c:
        for n, (first, last, names) in enumerate(jobs):
            params = dict(market_tickers=",".join(names), start_ts=first - 360, end_ts=last - 120, period_interval=1)
            try:
                get(c, "/markets/candlesticks", params, slim_batch)
            except RuntimeError as exc:
                print("  skip", names[0], exc, flush=True)
            if n % 50 == 0:
                print(f"  candles {n}/{len(jobs)} {utc(first)}", flush=True)
    print("candles done", flush=True)


def live_bars():
    rows = {}
    for first, last, names in jobs_live():
        params = dict(market_tickers=",".join(names), start_ts=first - 360, end_ts=last - 120, period_interval=1)
        key = hashlib.sha256(json.dumps(["/markets/candlesticks", params], sort_keys=True).encode()).hexdigest()
        cache = RAW / "req" / (key + ".json")
        if cache.exists():
            rows.update(json.loads(cache.read_text()))
    return rows


def hist_order():
    pool = [m for m in listed("hist-listing").values() if m["close"] < LIVE_START]
    pool.sort(key=lambda m: m["ticker"])
    random.Random(13).shuffle(pool)
    return pool


def hist_params(m):
    return dict(start_ts=m["close"] - 360, end_ts=m["close"] - 120, period_interval=1)


def do_hist():
    pool = hist_order()
    print("historical pool", len(pool), flush=True)
    with client() as c:
        for n, m in enumerate(pool):
            try:
                get(c, f"/historical/markets/{m['ticker']}/candlesticks", hist_params(m), slim_single, ok404=True)
            except RuntimeError as exc:
                print("  skip", m["ticker"], exc, flush=True)
            if n % 100 == 0:
                print(f"  hist {n}", flush=True)


def hist_bars():
    rows = {}
    for m in hist_order():
        path = f"/historical/markets/{m['ticker']}/candlesticks"
        key = hashlib.sha256(json.dumps([path, hist_params(m)], sort_keys=True).encode()).hexdigest()
        cache = RAW / "req" / (key + ".json")
        if not cache.exists():
            continue                      # the order is random, so whatever was reached is a random sample
        bars = json.loads(cache.read_text())
        if bars:
            rows[m["ticker"]] = bars
    return rows


# ---------------------------------------------------------------- the frozen rule

def fee(price, mult=1.0):
    return 7 * price / 100 * (1 - price / 100) * mult


def bet(m, bars, mult=1.0):
    """The frozen rule on one market. Returns None (no signal) or a dict; `net` is None when there was no bet."""
    by = {b[0]: b for b in bars}
    s = m["close"] - 300
    b = by.get(s)
    if b is None or b[1] is None or b[2] is None:
        return None
    bid, ask = b[1], b[2]
    if not 0 < bid <= ask < 100 or ask - bid > MAX_SPREAD + 1e-9:
        return None
    for yes in (True, False):
        sb, sa = (bid, ask) if yes else (round(100 - ask, 2), round(100 - bid, 2))
        if not BAND[0] - 1e-9 <= sa <= BAND[1] + 1e-9:
            continue
        payout = m["outcome"] if yes else 1 - m["outcome"]
        row = dict(ticker=m["ticker"], series=m["series"], close=m["close"], yes=yes, sig_ask=sa, sig_bid=sb, payout=payout,
                   entry=None, net=None, same=100 * payout - sa - fee(sa, mult))
        e = by.get(s + 60)
        if e is not None:
            quote = e[2] if yes else (None if e[1] is None else round(100 - e[1], 2))
            if quote is not None and 0 < quote < 100:
                row["entry"] = quote
                row["net"] = 100 * payout - quote - fee(quote, mult)
                row["vol_next"] = e[5]
        limit = max(round(sa - 1.0, 2), sb)
        filled = False
        for t in (s + 60, s + 120):
            x = by.get(t)
            if x is None:
                continue
            through = x[3] if yes else (None if x[4] is None else 100 - x[4])
            if through is not None and through < limit - 1e-6:
                filled = True
        row.update(limit=limit, filled=filled, rest=100 * payout - limit)
        return row
    return None


def vol_table(markets):
    """series -> {close: trailing-24h std of 15-minute log returns over the previous 96 windows (>= 48 present)}."""
    rets = defaultdict(dict)
    for m in markets:
        if m.get("floor") and m.get("final") and m["floor"] > 0 and m["final"] > 0:
            rets[m["series"]][m["close"]] = math.log(m["final"] / m["floor"])
    out = {}
    for s, r in rets.items():
        table = {}
        for close in r:
            past = [r[close - 900 * k] for k in range(1, 97) if close - 900 * k in r]
            if len(past) >= 48:
                table[close] = float(np.std(past))
        out[s] = table
    return out


def cluster(r):
    return (r["close"] - 1) // 3600


def ratio(rows, key="net", price="entry"):
    return 100 * sum(r[key] for r in rows) / sum(r[price] for r in rows)


def stats(rows, key="net", price="entry", draws=2000):
    """Return in % of money put down, hour-clustered bootstrap interval, chronological halves."""
    rows = [r for r in rows if r.get(key) is not None]
    if not rows:
        return dict(n=0)
    groups = defaultdict(lambda: [0.0, 0.0])
    for r in rows:
        g = groups[cluster(r)]
        g[0] += r[key]
        g[1] += r[price]
    arr = np.array(list(groups.values()))
    ci = None
    if len(arr) >= 30:
        idx = np.random.default_rng(7).integers(0, len(arr), size=(draws, len(arr)))
        est = np.sort(100 * arr[idx, 0].sum(axis=1) / arr[idx, 1].sum(axis=1))
        ci = [float(est[int(.025 * draws)]), float(est[int(.975 * draws) - 1])]
    cut = float(np.median([r["close"] for r in rows]))
    early, late = [r for r in rows if r["close"] < cut], [r for r in rows if r["close"] >= cut]
    wins = sum(r["payout"] > .5 for r in rows)
    return dict(n=len(rows), hours=len(arr), windows=len({r["close"] for r in rows}), wins=wins, win_pct=100 * wins / len(rows),
                price=float(np.mean([r[price] for r in rows])), ret=ratio(rows, key, price), ci=ci,
                halves=[ratio(early, key, price) if early else None, ratio(late, key, price) if late else None],
                cents=float(np.mean([r[key] for r in rows])))


def line(label, s):
    if not s.get("n"):
        return f"  {label:44} no bets"
    ci = f"[{s['ci'][0]:+.2f}, {s['ci'][1]:+.2f}]" if s["ci"] else "[n/a: <30 hours]"
    h = " / ".join("n/a" if v is None else f"{v:+.2f}" for v in s["halves"])
    return (f"  {label:44} bets {s['n']:5} windows {s['windows']:5} hours {s['hours']:5} wins {s['wins']:5} ({s['win_pct']:6.2f}%) "
            f"price {s['price']:6.2f}c  ret {s['ret']:+6.2f}% {ci:17} halves {h}")


def verdict(s):
    if s["ci"] and s["ci"][0] > 0 and all(h is not None and h > 0 for h in s["halves"]):
        return "CONFIRMED"
    if s["ret"] <= 0 or (s["ci"] and s["ci"][1] < .35):
        return "KILLED"
    return "UNDECIDED"


def planted(rows, target, runs=1000, inner=300, seed=5):
    """Share of cluster-resampled datasets, recentred to a true return of `target` %, that pass the frozen test."""
    rows = [r for r in rows if r["net"] is not None]
    shift = (target - ratio(rows)) / 100
    groups = defaultdict(lambda: [0.0, 0.0, 0.0])
    for r in rows:
        g = groups[cluster(r)]
        g[0] += r["net"] + shift * r["entry"]
        g[1] += r["entry"]
        g[2] = r["close"]
    keys = sorted(groups)
    arr = np.array([groups[k] for k in keys])
    half = np.cumsum(arr[:, 1]) <= arr[:, 1].sum() / 2          # early half of the money, in time order
    rng = np.random.default_rng(seed)
    n, passed, ci_only, lows = len(arr), 0, 0, []
    for _ in range(runs):
        pick = rng.integers(0, n, n)
        a, early = arr[pick], half[pick]
        idx = rng.integers(0, n, size=(inner, n))
        est = np.sort(100 * a[idx, 0].sum(axis=1) / a[idx, 1].sum(axis=1))
        low = est[int(.025 * inner)]
        lows.append(low)
        h1 = a[early, 0].sum() / max(a[early, 1].sum(), 1e-9)
        h2 = a[~early, 0].sum() / max(a[~early, 1].sum(), 1e-9)
        ci_only += low > 0
        passed += low > 0 and h1 > 0 and h2 > 0
    return dict(target=target, pass_rate=passed / runs, ci_above_zero=ci_only / runs)


# ---------------------------------------------------------------- data assembly

def multipliers():
    path = RAW / "series.json"
    rows = json.loads(path.read_text()) if path.exists() else []
    return {r["ticker"]: float(r["fee_multiplier"] or 1) for r in rows}, rows


def new_bets():
    mult, _ = multipliers()
    old = original_tickers()
    old_closes = {m["close"] for name in ORIGINAL.values() for m in json.loads((OLD / f"{name}.json").read_text())}
    live, hist = listed(), listed("hist-listing")
    vol = vol_table(list({**hist, **live}.values()))
    bars, hbars = live_bars(), hist_bars()
    rows, count = [], defaultdict(int)
    for m in live.values():
        if m["ticker"] in old or not LIVE_START <= m["close"] < LIST_END:
            continue
        stratum = "A" if m["series"] in ORIGINAL else ("C" if m["close"] in old_closes else "B")
        count[stratum + " listed"] += 1
        got = bars.get(m["ticker"])
        if not got:
            count[stratum + " no bars"] += 1
            continue
        count[stratum + " with bars"] += 1
        r = bet(m, got, mult.get(m["series"], 1.0))
        if r:
            rows.append(dict(r, stratum=stratum, vol=vol.get(m["series"], {}).get(m["close"])))
    for m in hist.values():
        got = hbars.get(m["ticker"])
        if m["close"] >= LIVE_START or m["ticker"] in old or not got:
            continue
        count["H with bars"] += 1
        r = bet(m, got, mult.get(m["series"], 1.0))
        if r:
            rows.append(dict(r, stratum="H", vol=vol.get(m["series"], {}).get(m["close"])))
    rows.sort(key=lambda r: (r["close"], r["ticker"]))
    return rows, dict(count)


def old_bets():
    mult, _ = multipliers()
    rows = []
    for s, name in ORIGINAL.items():
        for m in json.loads((OLD / f"{name}.json").read_text()):
            r = bet(dict(m, series=s), m["bars"], mult.get(s, 1.0))
            if r:
                rows.append(dict(r, stratum="orig", vol=None))
    return rows


def tape_sample():
    rows, _ = new_bets()
    pool = [r for r in rows if r["net"] is not None and r["stratum"] in "ABC"]
    random.Random(11).shuffle(pool)
    return pool[:300]


def tape_params(r, cursor=""):
    s = r["close"] - 300
    p = dict(ticker=r["ticker"], min_ts=s + 60, max_ts=s + 120, limit=1000)
    return dict(p, cursor=cursor) if cursor else p


def do_tape(max_pages=10):
    sample = tape_sample()
    print("tape sample", len(sample), flush=True)
    with client() as c:
        for n, r in enumerate(sample):
            cursor = ""
            for _ in range(max_pages):
                page = get(c, "/markets/trades", tape_params(r, cursor), slim_trades)
                cursor = page["cursor"]
                if not cursor or not page["rows"]:
                    break
            if n % 25 == 0:
                print(f"  tape {n}/{len(sample)}", flush=True)
    print("tape done", flush=True)


def tape_read(r, max_pages=10):
    """Cached prints for one bet in [S+60, S+120), oldest first; None if the first page was never fetched."""
    trades, cursor, complete = [], "", False
    for page_no in range(max_pages):
        key = hashlib.sha256(json.dumps(["/markets/trades", tape_params(r, cursor)], sort_keys=True).encode()).hexdigest()
        cache = RAW / "req" / (key + ".json")
        if not cache.exists():
            if page_no == 0:
                return None
            break
        page = json.loads(cache.read_text())
        trades += page["rows"]
        cursor = page["cursor"]
        if not cursor or not page["rows"]:
            complete = True
            break
    s = r["close"] - 300
    trades = sorted(t for t in trades if s + 60 <= t[0] < s + 120)
    return dict(trades=trades, complete=complete)


# ---------------------------------------------------------------- report

def do_score():
    lines = []
    say = lambda text="": (lines.append(text), print(text, flush=True))
    mult, series = multipliers()
    rows, count = new_bets()
    orig = old_bets()
    bets = [r for r in rows if r["net"] is not None]
    say(f"CRYPTO POCKET - frozen rule scored {utc(time.time())} UTC (rule: data/processed/crypto-pocket/prereg.txt)")
    say("fees from /series: " + ", ".join(f"{r['ticker']} {r['fee_type']} x{r['fee_multiplier']}" for r in series))
    say("markets: " + ", ".join(f"{k} {v}" for k, v in sorted(count.items())))
    say(f"signals {len(rows)}; bets {len(bets)}; signals with no entry quote one minute later (no bet) {len(rows) - len(bets)}"
        + (f", of which won {sum(r['payout'] > .5 for r in rows if r['net'] is None)}" if len(rows) > len(bets) else ""))
    if not bets:
        return
    say(f"new bets span {utc(bets[0]['close'])} to {utc(bets[-1]['close'])} UTC")
    primary = stats(bets)
    say()
    say(f"VERDICT ON NEW DATA (primary = every new market, entry one minute late, after fee): {verdict(primary)}")
    say(line("PRIMARY new A+B+C+H", primary))
    say(line("  original coins only (A+H)", stats([r for r in bets if r["stratum"] in "AH"])))
    say(line("  time-disjoint from original (A+B+H)", stats([r for r in bets if r["stratum"] in "ABH"])))
    for name in "ABCH":
        say(line(f"  stratum {name}", stats([r for r in bets if r["stratum"] == name])))
    say(line("  same bets at the signal-bar ask (no latency)", stats(bets, key="same", price="sig_ask")))
    moved = np.array([r["entry"] - r["sig_ask"] for r in bets])
    say(f"  entry minus signal ask: mean {moved.mean():+.2f}c, share entered above 99c {100 * np.mean([r['entry'] > 99 for r in bets]):.1f}%, "
        f"below 97c {100 * np.mean([r['entry'] < 97 for r in bets]):.1f}%")
    say()
    say("ORIGINAL SAMPLE re-scored with this code, and POOLED")
    obets = [r for r in orig if r["net"] is not None]
    say(line("original, signal-bar ask (the +0.65% lead)", stats(orig, key="same", price="sig_ask")))
    say(line("original, entry one minute late", stats(obets)))
    say(line("POOLED original + new, one minute late", stats(obets + bets)))
    say(line("POOLED original coins only", stats(obets + [r for r in bets if r["stratum"] in "AH"])))
    say()
    say("BY COIN (new bets; then pooled with the original sample where the coin was in it)")
    for s in SERIES:
        say(line(s + " new", stats([r for r in bets if r["series"] == s])))
        if s in ORIGINAL:
            say(line(s + " pooled", stats([r for r in obets + bets if r["series"] == s])))
    say()
    say("BY UTC HOUR OF DAY (new bets, hour of the window close)")
    for lo in range(0, 24, 3):
        say(line(f"{lo:02}-{lo + 3:02}h", stats([r for r in bets if lo <= ((r['close'] - 1) % 86400) // 3600 < lo + 3])))
    say()
    say("BY REALISED-VOLATILITY TERCILE (coin's own trailing 24h, cut per coin on the new bets)")
    with_vol = [r for r in bets if r["vol"] is not None]
    cuts = {}
    for s in SERIES:
        v = sorted(r["vol"] for r in with_vol if r["series"] == s)
        if len(v) >= 3:
            cuts[s] = (v[len(v) // 3], v[2 * len(v) // 3])
    tier = lambda r: 0 if r["vol"] < cuts[r["series"]][0] else (1 if r["vol"] < cuts[r["series"]][1] else 2)
    for k, label in enumerate(("low vol", "mid vol", "high vol")):
        say(line(label, stats([r for r in with_vol if r["series"] in cuts and tier(r) == k])))
    say(f"  bets without 12h of prior windows (left out) {len(bets) - len(with_vol)}")
    say()

    # ---- fill realism
    say("FILL REALISM (random new bets, seed 11; prints in [signal + 60 s, signal + 120 s))")
    checked = []
    for r in tape_sample():
        t = tape_read(r)
        if t is None:
            continue
        side = "yes" if r["yes"] else "no"
        ours = [(ts, p if r["yes"] else round(100 - p, 2), n) for ts, p, n, taker in t["trades"] if taker == side]
        row = dict(r, complete=t["complete"], buys=sum(n for _, _, n in ours),
                   buyable=sum(n for _, p, n in ours if p <= r["entry"] + 1e-6), first=ours[0][1] if ours else None,
                   delay=ours[0][0] - (r["close"] - 240) if ours else None)
        if ours:
            row["tape_net"] = 100 * r["payout"] - row["first"] - fee(row["first"], mult.get(r["series"], 1.0))
        checked.append(row)
    tape = {}
    if checked:
        n = len(checked)
        any_buy = [r for r in checked if r["buyable"] > 0]
        printed = [r for r in checked if r["first"] is not None]
        size = np.array([r["buyable"] for r in checked])
        q = np.percentile(size, [25, 50, 75])
        say(f"  bets checked {n} (pages cut short {sum(not r['complete'] for r in checked)})")
        say(f"  a taker bought our side at or below the assumed ask: {len(any_buy)} of {n} ({100 * len(any_buy) / n:.1f}%)")
        say(f"  contracts bought at or below the assumed ask in that minute: quartiles {q[0]:,.0f} / {q[1]:,.0f} / {q[2]:,.0f}; "
            f">= 100 in {100 * np.mean(size >= 100):.1f}%, >= 1,000 in {100 * np.mean(size >= 1000):.1f}% of bets")
        say(f"  any taker buy of our side in that minute (any price): {len(printed)} of {n} ({100 * len(printed) / n:.1f}%); "
            f"first print minus bar ask: mean {np.mean([r['first'] - r['entry'] for r in printed]):+.2f}c, "
            f"median delay {np.median([r['delay'] for r in printed]):.1f}s")
        say(line("sample, bar ask (all checked)", stats(checked)))
        say(line("sample, bar ask (those with a print)", stats(printed)))
        say(line("sample, FIRST PRINT at/after signal+60s", stats(printed, key="tape_net", price="first")))
        none = [r for r in checked if r["first"] is None]
        if none:
            say(f"  no taker buy in the minute: {len(none)} bets, won {sum(r['payout'] > .5 for r in none)} "
                f"(bar-ask return {ratio(none):+.2f}%)")
        for s in SERIES:
            part = [r for r in checked if r["series"] == s]
            if part:
                sz = np.array([r["buyable"] for r in part])
                say(f"    {s:10} checked {len(part):3} median contracts at/below ask {np.median(sz):8,.0f}  >=100: {100 * np.mean(sz >= 100):5.1f}%  "
                    f">=1,000: {100 * np.mean(sz >= 1000):5.1f}%")
        tape = dict(n=n, any=len(any_buy), quartiles=[float(x) for x in q], ge100=float(np.mean(size >= 100)),
                    ge1000=float(np.mean(size >= 1000)), mean_cap1000=float(np.mean(np.minimum(size, 1000))),
                    mean_cap100=float(np.mean(np.minimum(size, 100))))
    else:
        say("  no tape downloaded")
    say()

    # ---- resting bid
    say("RESTING BID one cent under the signal ask for 2 minutes (strict print-through, no fee), new signals")
    hit, miss = [r for r in rows if r["filled"]], [r for r in rows if not r["filled"]]
    say(f"  orders {len(rows)}, filled {len(hit)} ({100 * len(hit) / len(rows):.1f}%), mean bid {np.mean([r['limit'] for r in rows]):.2f}c")
    if hit:
        say(line("filled bids (return on the bid)", stats(hit, key="rest", price="limit")))
        say(f"  win rate filled {100 * np.mean([r['payout'] > .5 for r in hit]):.2f}% against unfilled "
            f"{100 * np.mean([r['payout'] > .5 for r in miss]):.2f}% (all signals {100 * np.mean([r['payout'] > .5 for r in rows]):.2f}%)")
        both = [r for r in hit if r["net"] is not None]
        say(line("same filled orders as a taker one minute late", stats(both)))
        say(line("unfilled orders as a taker one minute late", stats([r for r in miss if r["net"] is not None])))
        ohit = [r for r in orig if r["filled"]]
        say(line("original sample, filled bids", stats(ohit, key="rest", price="limit")))
        say(line("pooled, filled bids", stats(ohit + hit, key="rest", price="limit")))
    say()

    # ---- economics
    say("ECONOMICS (new bets, live-tier strata A+B+C; one minute late, after fee)")
    live = [r for r in bets if r["stratum"] in "ABC"]
    if live:
        days = (live[-1]["close"] - live[0]["close"]) / 86400
        s_live = stats(live)
        five = [r for r in live if r["series"] in OTHERS]
        odays = (max(r["close"] for r in obets) - min(r["close"] for r in obets if r["series"] != "KXBTC15M")) / 86400
        four = len([r for r in obets if r["close"] >= min(x["close"] for x in obets if x["series"] != "KXBTC15M")]) / odays
        say(f"  {len(live)} new bets over {days:.1f} days; the five other coins alone give {len(five) / days:.1f} bets a day, the original four "
            f"gave {four:.1f} a day in the original sample: about {len(five) / days + four:.0f} bets a day across all nine")
        per_day = len(five) / days + four
        say(f"  net per contract {s_live['cents']:+.3f}c at a mean price of {s_live['price']:.2f}c")
        price = s_live["price"]
        cost = price + fee(price)
        say(f"  one loss costs {cost:.2f}c; one win pays {100 - cost:.2f}c; one loss erases {cost / (100 - cost):.0f} wins "
            f"(break-even win rate {cost:.2f}%, observed {s_live['win_pct']:.2f}%)")
        say(f"  at 100 contracts a bet: ${per_day * 100 * s_live['cents'] / 100:+,.2f} a day on about ${per_day * price:,.0f} staked a day "
            f"(${price:,.0f} a bet), if every bet filled in full")
        if tape:
            say(f"  tape-capped: mean contracts actually bought at or below the ask, capped at 100: {tape['mean_cap100']:.0f}; capped at 1,000: {tape['mean_cap1000']:.0f}")
            say(f"  at 100 a bet (tape-capped): ${per_day * tape['mean_cap100'] * s_live['cents'] / 100:+,.2f} a day; "
                f"at 1,000 a bet (tape-capped): ${per_day * tape['mean_cap1000'] * s_live['cents'] / 100:+,.2f} a day; "
                f"at 1,000 uncapped: ${per_day * 1000 * s_live['cents'] / 100:+,.2f} a day")
        vol_next = np.array([r.get("vol_next") or 0 for r in live])
        say(f"  contracts traded (both sides) in the entry minute, all bets: quartiles "
            f"{np.percentile(vol_next, 25):,.0f} / {np.percentile(vol_next, 50):,.0f} / {np.percentile(vol_next, 75):,.0f}")
    say()

    # ---- planted edge
    say("PLANTED EDGE (real bets recentred to a known true return, hour clusters resampled 1,000 times, frozen test)")
    for label, part in (("new bets", bets), ("pooled original + new", obets + bets)):
        for target in (0.0, .35, .65):
            p = planted(part, target)
            say(f"  {label:24} n {len(part):5} true {target:+.2f}%: interval above zero {100 * p['ci_above_zero']:5.1f}%, "
                f"full test (interval and both halves) {100 * p['pass_rate']:5.1f}%")
    s_all = stats(obets + bets)
    if s_all["ci"]:
        width = (s_all["ci"][1] - s_all["ci"][0]) / 2
        say(f"  pooled interval half-width {width:.2f}%: the smallest true edge with 80% power is about {width * 2.8 / 1.96:.2f}%; "
            f"detecting +0.35% at 80% power needs about {max((width * 2.8 / 1.96 / .35) ** 2, 0):.1f}x this sample")
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "report.txt").write_text("\n".join(lines) + "\n")
    (OUT / "results.json").write_text(json.dumps(dict(
        verdict=verdict(primary), primary=primary, counts=count, pooled=stats(obets + bets), tape=tape,
        by_coin={s: stats([r for r in bets if r["series"] == s]) for s in SERIES}), indent=1))
    (OUT / "bets.json").write_text(json.dumps(rows, separators=(",", ":")))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("steps", nargs="+", choices=("series", "listing", "candles", "tape", "hist", "score"))
    parser.add_argument("--minutes", type=float, default=50, help="request budget for this run (never past the 5-hour stop)")
    args = parser.parse_args()
    _stop[0] = min(HARD_STOP, time.time() + 60 * args.minutes)
    for step in args.steps:
        try:
            dict(series=do_series, listing=do_listing, candles=do_candles, tape=do_tape, hist=do_hist, score=do_score)[step]()
        except TimeoutError:
            print(step, "stopped: budget used; run again to resume", flush=True)
            return


if __name__ == "__main__":
    main()
