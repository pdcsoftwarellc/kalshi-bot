"""Favorite-longshot test on non-sports Kalshi markets with days or weeks left.

Public Kalshi reads only. Steps (run in order):
  universe  list settled markets of the highest-volume non-sports series per category
  bars      hourly quote bars for a seeded sample of those markets
  analyze   part A: buy the 90-99c side D days before the scheduled close, hold to settlement
  fills     part B: would a bid resting one cent under the ask have been filled (public trades)

Scheduled close. Kalshi overwrites close_time when a market closes early, so a
close that is not on a whole minute, or that precedes the latest close among the
event's markets, is treated as an early close and the reference becomes the
market's latest_expiration_time, which is published in advance. Entries are only
taken while the market was really open.

Hourly bars are emitted only for hours in which the quote or tape changed, so the
quote at time T is the close of the last bar ending at or before T. The decision
uses the quote at T = reference - D days; the simulated order pays the ask one
hour later (the next hourly bar) if it is still at or under the top of the band.
"""
import argparse
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from datetime import date
import hashlib
import json
from pathlib import Path
import random
from statistics import mean, median
import sys
import threading
import time

import httpx

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import config
from research.api_cost import ROOT, midnight, save, timestamp

RAW = ROOT / "data/raw/favorites"
OUT = ROOT / "data/processed/favorites"
START, END = date(2025, 7, 1), date(2026, 11, 14)
CUTOFF = timestamp("2026-09-18T00:00:00Z")
SKIP_CATEGORIES = {"Sports", "Crypto", "Exotics"}
INTRADAY = {"hourly", "fifteen_min"}
NO_DAILY = {"Financials", "Commodities"}       # daily index, currency and commodity ranges are intraday bets
TOP_SERIES = 30
MAX_PAGES = 12
MIN_VOLUME = 1000
PER_EVENT, PER_SERIES = 2, 20
DAYS = (1, 3, 7, 14, 30)
BANDS = ((90, 94), (95, 97), (98, 99))
HOUR, DAY = 3600, 86400
GAP = .5                                        # seconds between request starts, shared by both workers

_lock, _next = threading.Lock(), [0.0]
deadline = [float("inf")]


def get(client, path, params, slim=None, folder="misc"):
    """Throttled GET with a slimmed disk cache. Returns None for a 4xx other than 429."""
    key = hashlib.sha256(json.dumps([path, params], sort_keys=True).encode()).hexdigest()
    cache = RAW / folder / (key + ".json")
    if cache.exists():
        return json.loads(cache.read_text())
    if time.time() > deadline[0]:
        raise TimeoutError("download budget used")
    for attempt in range(10):
        with _lock:
            wait = _next[0] - time.time()
            _next[0] = max(time.time(), _next[0]) + GAP
        if wait > 0:
            time.sleep(wait)
        try:
            resp = client.get(config.KALSHI_PROD_BASE + path, params=params)
        except httpx.RequestError:
            time.sleep(2 ** attempt)
            continue
        if resp.status_code == 429 or resp.status_code >= 500:
            with _lock:                          # everyone pauses briefly; long exponential pauses here
                _next[0] = max(_next[0], time.time() + 2 + attempt)   # stalled both workers for minutes
            continue
        if resp.status_code != 200:
            data = None
        else:
            data = resp.json()
            if slim:
                data = slim(data)
        cache.parent.mkdir(parents=True, exist_ok=True)
        cache.write_text(json.dumps(data, separators=(",", ":")))
        return data
    raise RuntimeError(f"Read failed after retries: {path}")


MARKET_FIELDS = ("ticker", "event_ticker", "open_time", "close_time", "expiration_time", "latest_expiration_time",
                 "expected_expiration_time", "settlement_ts", "settlement_value_dollars", "result", "volume_fp",
                 "market_type", "can_close_early")


def slim_markets(data):
    return dict(cursor=data.get("cursor"), markets=[{k: m.get(k) for k in MARKET_FIELDS} for m in data["markets"]])


def slim_bars(data):
    def c(bar, side, field):
        v = (bar.get(side) or {})
        v = v.get(field + "_dollars", v.get(field))
        return None if v is None else round(100 * float(v))
    return [[b["end_period_ts"], c(b, "yes_bid", "open"), c(b, "yes_ask", "open"), c(b, "yes_bid", "close"),
             c(b, "yes_ask", "close"), float(b.get("volume_fp") or b.get("volume") or 0),
             c(b, "price", "low"), c(b, "price", "high")] for b in data.get("candlesticks", [])]


def slim_trades(data):
    out = []
    for t in data.get("trades", []):
        price = t.get("yes_price_dollars")
        price = round(100 * float(price)) if price is not None else t.get("yes_price")
        out.append([timestamp(t["created_time"]), price, float(t.get("count_fp") or t.get("count") or 0),
                    t.get("taker_side") or t.get("taker_outcome_side")])
    return dict(cursor=data.get("cursor"), trades=out)


def series_list(client):
    path = RAW / "series.json"
    if path.exists():
        return json.loads(path.read_text())
    rows = client.get(config.KALSHI_PROD_BASE + "/series", params=dict(include_volume="true")).json()["series"]
    rows = [dict(ticker=s["ticker"], category=s.get("category") or "", frequency=s.get("frequency"),
                 fee_type=s.get("fee_type"), fee_multiplier=s.get("fee_multiplier"), title=s.get("title"),
                 volume=float(s.get("volume_fp") or 0)) for s in rows]
    save(path, rows)
    return rows


def chosen(series):
    by = defaultdict(list)
    for s in series:
        if s["category"] in SKIP_CATEGORIES or not s["category"] or s["frequency"] in INTRADAY or s["volume"] <= 0:
            continue
        if s["category"] in NO_DAILY and s["frequency"] == "daily":
            continue
        by[s["category"]].append(s)
    return {k: sorted(v, key=lambda s: -s["volume"])[:TOP_SERIES] for k, v in by.items()}


def list_series(client, ticker):
    found, truncated = {}, False
    for path, extra in (("/historical/markets", {}),
                        ("/markets", dict(status="settled", min_close_ts=midnight(START)))):
        cursor = ""
        for page in range(MAX_PAGES):
            params = dict(series_ticker=ticker, limit=1000, **extra)
            if cursor:
                params["cursor"] = cursor
            data = get(client, path, params, slim_markets, "markets")
            if not data or not data["markets"]:
                break
            for m in data["markets"]:
                found[m["ticker"]] = m
            cursor = data.get("cursor")
            # both listings run newest first; stop once a whole page is older than the window
            if not cursor or all(timestamp(m["close_time"]) < midnight(START) for m in data["markets"]):
                break
        else:
            truncated = True
    return list(found.values()), truncated


def universe(client):
    picks = chosen(series_list(client))
    # best series of every category first, so a run that hits its time budget still covers each category
    jobs = [(cat, group[rank]) for rank in range(TOP_SERIES) for cat, group in picks.items() if rank < len(group)]
    print(f"{len(jobs)} series in {len(picks)} categories", flush=True)
    out, seen, unlisted = [], set(), []

    def listing(job):
        try:
            return list_series(client, job[1]["ticker"])
        except TimeoutError:
            return None
    with ThreadPoolExecutor(2) as pool:
        for (cat, s), result in zip(jobs, pool.map(listing, jobs)):
            if result is None:
                unlisted.append(s["ticker"])
                continue
            markets, truncated = result
            keep = []
            for m in markets:
                if m["ticker"] in seen or not m.get("close_time"):
                    continue
                if not midnight(START) <= timestamp(m["close_time"]) < midnight(END):
                    continue
                seen.add(m["ticker"])
                keep.append(dict(m, series=s["ticker"], category=cat))
            out += keep
            print(f"  {cat:24} {s['ticker']:28} {len(keep):6}{' truncated' if truncated else ''}", flush=True)
    save(RAW / "universe.json", out)
    save(RAW / "unlisted.json", unlisted)
    print(len(out), "markets;", len(unlisted), "series not listed within the time budget")


def load_universe():
    return json.loads((RAW / "universe.json").read_text())


def references(markets):
    """Attach the scheduled reference time and the early-close flag to each market."""
    latest = defaultdict(float)
    for m in markets:
        latest[m["event_ticker"]] = max(latest[m["event_ticker"]], timestamp(m["close_time"]))
    for m in markets:
        close = timestamp(m["close_time"])
        m["close_ts"], m["open_ts"] = close, timestamp(m["open_time"])
        m["early"] = close % 60 != 0 or close < latest[m["event_ticker"]] - 60
        ref = m.get("latest_expiration_time") or m.get("expiration_time")
        m["ref_ts"] = timestamp(ref) if m["early"] and ref else close
        m["early"] = bool(m["early"])
    return markets


def usable(m):
    try:
        value = float(m["settlement_value_dollars"])
    except (TypeError, ValueError):
        return False
    return (m.get("market_type") == "binary" and m.get("settlement_ts") and 0 <= value <= 1
            and float(m.get("volume_fp") or 0) >= MIN_VOLUME
            and min(m["close_ts"], m["ref_ts"]) - m["open_ts"] >= DAY + 2 * HOUR   # alive at the one-day snapshot
            and m["ref_ts"] - 30 * DAY < m["close_ts"])                            # some snapshot falls inside its life


def sample(markets):
    """Seeded sample: PER_SERIES events per series, PER_EVENT markets per event."""
    by_series = defaultdict(lambda: defaultdict(list))
    for m in markets:
        if usable(m):
            by_series[m["series"]][m["event_ticker"]].append(m)
    picked = []
    for series, events in sorted(by_series.items()):
        rng = random.Random(series)
        names = sorted(events)
        for event in sorted(rng.sample(names, min(PER_SERIES, len(names)))):
            group = sorted(events[event], key=lambda m: m["ticker"])
            picked += rng.sample(group, min(PER_EVENT, len(group)))
    return picked


def hindsight(markets, category="Elections"):
    """Supplement only. Election markets close when the race is called, far ahead of their latest
    expiration, so the plain rule never reaches them. Here the reference is the ACTUAL close, which
    for an election falls shortly after a polling day known in advance, but is still hindsight."""
    group = [dict(m) for m in markets if m["category"] == category]
    for m in group:
        m["ref_ts"], m["early"] = m["close_ts"], True
    return sample(group)


def bar_request(m):
    start = int(max(m["open_ts"], m["ref_ts"] - 45 * DAY))   # quiet markets emit few bars; look back for a quote
    end = int(min(m["close_ts"], m["ref_ts"]) + HOUR)
    params = dict(start_ts=start, end_ts=end, period_interval=60)
    old = f"/historical/markets/{m['ticker']}/candlesticks"
    new = f"/series/{m['series']}/markets/{m['ticker']}/candlesticks"
    return (old, new) if timestamp(m["settlement_ts"]) < CUTOFF else (new, old), params


def bars_for(client, m, download=True):
    paths, params = bar_request(m)
    for path in paths:
        if not download:
            key = hashlib.sha256(json.dumps([path, params], sort_keys=True).encode()).hexdigest()
            if not (RAW / "bars" / (key + ".json")).exists():
                continue
        rows = get(client, path, params, slim_bars, "bars")
        if rows:
            return rows
    return None


def fetch_bars(client, supplement=False):
    markets = references(load_universe())
    picked = hindsight(markets) if supplement else sample(markets)
    random.Random(5).shuffle(picked)             # an interrupted run is still a fair sample
    print(len(picked), "markets to fetch", flush=True)
    done = Counter()

    def one(m):
        try:
            done["bars" if bars_for(client, m) else "empty"] += 1
        except TimeoutError:
            done["not fetched"] += 1
        if sum(done.values()) % 250 == 0:
            print(dict(done), flush=True)
    with ThreadPoolExecutor(2) as pool:
        list(pool.map(one, picked))
    print(dict(done))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("step", choices=("universe", "bars", "elections", "analyze", "fills"))
    parser.add_argument("--minutes", type=float, default=30, help="download budget for this step")
    args = parser.parse_args()
    deadline[0] = time.time() + 60 * args.minutes
    with httpx.Client(timeout=60) as client:
        if args.step == "universe":
            universe(client)
        elif args.step in ("bars", "elections"):
            fetch_bars(client, args.step == "elections")
        else:
            from research import favorites_report
            getattr(favorites_report, args.step)(client)


if __name__ == "__main__":
    main()
