"""Does the "nearly decided, shortly before resolution" premium exist in short-dated non-sports markets?

Kalshi public reads plus public Coinbase minute candles. Research only; no orders.

Rule under test: inside the last N minutes before a market's SCHEDULED close
(`close_time`, published in advance), when a side's ask closes a minute inside
a price band, buy that side at the NEXT minute's opening ask (YES ask, or
100 minus YES bid for NO) if it is still inside the band, pay the taker fee
0.07*p*(1-p) and hold to the exchange settlement value.

  download   list settled markets and pull one-minute bars (throttled, cached)
  underlying pull Coinbase BTC/ETH minute candles for the state screen
  analyze    build entries, pick rules on the earliest 60% of events, score on the latest 40%

Second study (November 19): is the sports "underpriced favourite" gap present here?
  fetch2     pull the extra families (SOL/XRP/FX 15-minute, Nasdaq hourly, daily ladders), 2 s between requests
  prints2    trade prints by taker side in the minute after the decision time, to check the candle ask and size
  calibrate  per family and time-to-close bucket: favourite-side calibration against mid and ask net of the
             taker fee, a resting-bid version filled only on a strict print-through, and a distance-to-strike split
"""
import argparse
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from datetime import date, datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import random
from statistics import mean, median
import sys
import threading
import time

import httpx
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from research.api_cost import COLLECTION, ROOT, fetch, midnight, pages, save, timestamp

OUT = ROOT / "data/processed/short-dated"
RAW = ROOT / "data/raw/short-dated"
END = date(2026, 11, 18)
# family -> (series, first day, minutes of bars before close, strike filter as a fraction of the final value)
FAMILIES = {       # ordered by priority; windows were cut to fit a 30-minute shared-API download budget
    "btc15m": (("KXBTC15M",), date(2026, 9, 18), 15, None),
    "btc_hourly": (("KXBTCD",), date(2026, 11, 8), 60, .025),
    "eth15m": (("KXETH15M",), date(2026, 10, 19), 15, None),
    "index_hourly": (("KXINXU",), date(2026, 9, 18), 60, None),
    "weather_daily": (("KXHIGHNY", "KXHIGHLAX"), date(2026, 10, 19), 360, None),
    "eth_hourly": (("KXETHD",), date(2026, 11, 8), 60, .04),
}
# Earlier slices added once the first pass left budget: family -> [(first day, end day)]
EXTRA = {"btc_hourly": [(date(2026, 10, 29), date(2026, 11, 8))]}
WINDOWS = {15: (2, 5, 10), 60: (5, 15, 30, 60), 360: (60, 180, 360)}
BANDS = ((95, 99), (97, 99), (98, 99), (99, 99.9))
PLAIN = (97, 99)
Z_GRID = (1.5, 2.5, 4.0)
COIN = {"btc15m": "BTC-USD", "btc_hourly": "BTC-USD", "eth15m": "ETH-USD", "eth_hourly": "ETH-USD"}

_gate, _last = threading.Lock(), [0.0]
_deadline = [None]


def polite(client, path, params):
    """`fetch` with a shared throttle (about three uncached requests a second) and a download deadline."""
    key = hashlib.sha256(json.dumps([path, params], sort_keys=True).encode()).hexdigest()
    if not (COLLECTION / (key + ".json")).exists():
        if _deadline[0] and time.time() > _deadline[0]:
            raise TimeoutError("download budget used")
        with _gate:
            wait = _last[0] + .35 - time.time()
            if wait > 0:
                time.sleep(wait)
            _last[0] = time.time()
    for attempt in range(4):
        try:
            return fetch(client, path, params)
        except RuntimeError as exc:
            if "HTTP 4" in str(exc):
                raise
            time.sleep(10 * (attempt + 1))
    raise RuntimeError(f"gave up: {path}")


def listing(client, series, start, end=END):
    """Slim settled-market records; the full pages stay in the `fetch` cache."""
    slim = RAW / (f"{series}-markets.json" if end == END else f"{series}-{start}-markets.json")
    if slim.exists():
        return json.loads(slim.read_text())
    rows, cursor = [], ""
    params = dict(series_ticker=series, status="settled", limit=1000,
                  min_close_ts=midnight(start), max_close_ts=midnight(end))
    while True:
        data = polite(client, "/markets", dict(params, cursor=cursor) if cursor else params)
        for m in data["markets"]:
            try:
                rows.append(dict(
                    ticker=m["ticker"], event=m["event_ticker"], close=int(timestamp(m["close_time"])),
                    open=int(timestamp(m["open_time"])), outcome=float(m["settlement_value_dollars"]),
                    settle=timestamp(m["settlement_ts"]), floor=m.get("floor_strike"), cap=m.get("cap_strike"),
                    kind=m.get("strike_type"), final=float(m["expiration_value"]) if m.get("expiration_value") else None,
                    volume=float(m.get("volume_fp") or 0)))
            except (KeyError, TypeError, ValueError):
                continue
        cursor = data.get("cursor")
        if not cursor or not data["markets"]:
            break
    save(slim, rows)
    return rows


def near(m, width):
    if width is None:
        return True
    if not m["final"] or m["floor"] is None:
        return False
    return abs(m["floor"] / m["final"] - 1) <= width


def compact(bar):
    def px(side, field):
        try:
            return round(100 * float(bar[side][field + "_dollars"]), 1)
        except (KeyError, TypeError, ValueError):
            return None
    return [bar["end_period_ts"], px("yes_bid", "open"), px("yes_ask", "open"), px("yes_bid", "close"),
            px("yes_ask", "close"), float(bar.get("volume_fp") or 0)]


def download(client, family, minutes_budget):
    series_list, start, window, width = FAMILIES[family]
    for first, end in [(start, END)] + EXTRA.get(family, []):
        download_slice(client, family, first, end)


def download_slice(client, family, start, end):
    series_list, _, window, width = FAMILIES[family]
    out = RAW / (f"{family}-bars.json" if end == END else f"{family}-bars-{start}.json")
    if out.exists():
        print(family, start, "already downloaded", flush=True)
        return
    markets = [m for s in series_list for m in listing(client, s, start, end)]
    print(family, "listed", len(markets), "markets", flush=True)
    wanted = [m for m in markets if near(m, width)]
    # One request takes one time range and counts tickers x minutes against a 10,000-bar cap.
    jobs, chunk = [], []
    for m in sorted(wanted, key=lambda m: (m["close"], m["ticker"])):
        span = (m["close"] - chunk[0]["close"]) // 60 + window + 2 if chunk else 0
        if chunk and ((len(chunk) + 1) * span > 9500 or len(chunk) >= 100):      # at most 100 tickers a request
            jobs.append(chunk)
            chunk = []
        chunk.append(m)
    if chunk:
        jobs.append(chunk)

    def bars(job):
        group = job
        try:
            data = polite(client, "/markets/candlesticks", dict(
                market_tickers=",".join(m["ticker"] for m in group),
                start_ts=group[0]["close"] - window * 60, end_ts=group[-1]["close"], period_interval=1))
        except TimeoutError:
            return None
        except RuntimeError as exc:
            print("  skip", group[0]["ticker"], exc, flush=True)
            return {}
        return {m["market_ticker"]: [compact(b) for b in m["candlesticks"]] for m in data.get("markets", [])}
    got, complete = {}, True
    with ThreadPoolExecutor(2) as pool:
        for n, result in enumerate(pool.map(bars, jobs)):
            if result is None:
                complete = False
                continue
            got.update(result)
            if n % 100 == 0:
                print(f"  {family} {n}/{len(jobs)} requests", flush=True)
    rows = []
    for m in wanted:
        kept = [b for b in got.get(m["ticker"], []) if m["close"] - window * 60 < b[0] <= m["close"]]
        if kept:
            rows.append(dict(m, bars=kept))
    print(family, "bars for", len(rows), "of", len(wanted), "selected markets;", "complete" if complete else "PARTIAL", flush=True)
    if rows:
        save(out if complete else out.with_name(out.stem + "-partial.json"), rows)


def underlying(product, start):
    """Coinbase one-minute candles as {bucket start: close}. Generic client, no account data."""
    path = RAW / f"{product}-minutes.json"
    if path.exists():
        return {int(k): v for k, v in json.loads(path.read_text()).items()}
    rows, at, end = {}, midnight(start) - 7200, midnight(END)
    with httpx.Client(timeout=30, headers={"User-Agent": "research-client"}) as client:
        while at < end:
            stop = min(at + 300 * 60, end)
            for attempt in range(6):
                resp = client.get(f"https://api.exchange.coinbase.com/products/{product}/candles",
                                  params=dict(granularity=60, start=datetime.fromtimestamp(at, timezone.utc).isoformat(),
                                              end=datetime.fromtimestamp(stop - 60, timezone.utc).isoformat()))
                if resp.status_code == 200:
                    break
                time.sleep(2 ** attempt)
            else:
                raise RuntimeError("Coinbase candles unavailable")
            for t, low, high, first, last, volume in resp.json():
                rows[int(t)] = last
            at = stop
            time.sleep(.15)
    save(path, rows)
    return rows


class State:
    """Distance to strike in volatility units, using only prices printed before the entry time."""

    def __init__(self, minutes, basis_points):
        self.px = minutes
        self.basis = sorted(basis_points)          # (time known, index value minus Coinbase close)

    def price(self, ts):                           # last Coinbase close of a minute that ended at or before ts
        for back in range(1, 6):
            v = self.px.get(ts - 60 * back)
            if v is not None:
                return v
        return None

    def sigma(self, ts, span=60):                  # dollars per sqrt(minute)
        steps = []
        for i in range(1, span + 1):
            a, b = self.px.get(ts - 60 * i), self.px.get(ts - 60 * (i + 1))
            if a and b:
                steps.append(a - b)
        return math.sqrt(mean(s * s for s in steps)) if len(steps) >= span // 2 else None

    def offset(self, ts):
        lo, hi = 0, len(self.basis)
        while lo < hi:
            mid = (lo + hi) // 2
            if self.basis[mid][0] <= ts:
                lo = mid + 1
            else:
                hi = mid
        recent = [b for _, b in self.basis[max(0, lo - 4):lo]]
        return median(recent) if recent else 0.0

    def z(self, ts, strike, close, yes):
        spot, vol = self.price(ts), self.sigma(ts)
        if spot is None or not vol:
            return None
        # The index is averaged over the final minute, which removes about two thirds of a minute of variance.
        left = max((close - ts) / 60 - 2 / 3, 1 / 3)
        gap = spot + self.offset(ts) - strike
        return (gap if yes else -gap) / (vol * math.sqrt(left))


def entries(m, windows, state=None):
    """First qualifying entry per side, band and window for one market."""
    found, bars = [], sorted(m["bars"])
    for yes in (True, False):
        def ask(bid, offer):
            if bid is None or offer is None:
                return None, False
            mine = offer if yes else round(100 - bid, 1)
            # Two-sided: a real offer on the side bought and a real bid behind it.
            return mine, 0 < bid <= offer < 100
        pending = {(band, n) for band in BANDS for n in windows}
        for bar, nxt in zip(bars, bars[1:]):
            ts = bar[0]                            # decision time; the next bar opens here
            left = (m["close"] - ts) / 60
            if left <= 0 or nxt[0] - ts > 120 or nxt[0] > m["close"]:
                continue
            seen, ok_seen = ask(bar[3], bar[4])
            price, ok = ask(nxt[1], nxt[2])
            if seen is None or price is None or not ok_seen:
                continue
            for band, n in sorted(pending):
                if left <= n and band[0] <= seen <= band[1] and band[0] <= price <= band[1]:
                    pending.discard((band, n))
                    p = price / 100
                    payout = m["outcome"] if yes else 1 - m["outcome"]
                    z = None
                    if state and m["floor"] is not None and m["kind"] in ("greater", "greater_or_equal"):
                        z = state.z(ts, m["floor"], m["close"], yes)
                    # Robustness: the same signal filled a minute later at whatever the ask is by then.
                    late, late_ok = ask(nxt[3], nxt[4])
                    late_net = None
                    if late is not None and late_ok and nxt[0] < m["close"]:
                        late_net = 100 * (payout - late / 100 - .07 * late / 100 * (1 - late / 100))
                    found.append(dict(late_net=late_net, band=list(band), window=n, side="yes" if yes else "no", price=price, ts=ts,
                                      minutes_left=left, two_sided=ok, volume=nxt[5], prior_volume=bar[5],
                                      spread=(nxt[2] - nxt[1]) if ok else None, z=z, payout=payout,
                                      net=100 * (payout - p - .07 * p * (1 - p)), event=m["event"], ticker=m["ticker"],
                                      close=m["close"]))
    return found


def interval(rows, draws=2000, key=lambda r: r["event"]):
    """95% bootstrap interval for mean net cents, resampling whole events (or whole UTC days)."""
    by_event = defaultdict(list)
    for r in rows:
        by_event[key(r)].append(r["net"])
    groups = list(by_event.values())
    if len(groups) < 10:
        return None
    total, count = np.array([sum(g) for g in groups]), np.array([len(g) for g in groups])
    pick = np.random.default_rng(11).integers(0, len(groups), size=(draws, len(groups)))
    means = np.sort(total[pick].sum(axis=1) / count[pick].sum(axis=1))
    return [round(float(means[int(.025 * draws)]), 3), round(float(means[int(.975 * draws)]), 3)]


def one_per_event(rows):
    """Earliest entry per event; ties go to the busiest minute."""
    best = {}
    for r in sorted(rows, key=lambda r: (r["ts"], -r["volume"], r["ticker"])):
        best.setdefault(r["event"], r)
    return list(best.values())


def score(rows, days):
    if not rows:
        return dict(trades=0)
    p = mean(r["price"] for r in rows) / 100
    return dict(
        trades=len(rows), events=len({r["event"] for r in rows}), full_losses=sum(r["payout"] < .5 for r in rows),
        mean_price=round(100 * p, 2), win_rate=round(mean(r["payout"] for r in rows), 4),
        breakeven=round(mean(r["price"] / 100 + .07 * r["price"] / 100 * (1 - r["price"] / 100) for r in rows), 4),
        net_cents=round(mean(r["net"] for r in rows), 3), ci95=interval(rows),
        filled_one_minute_late=sum(r["late_net"] is not None for r in rows),
        losses_one_minute_late=sum(r["late_net"] is not None and r["payout"] < .5 for r in rows),
        net_one_minute_late=round(mean([r["late_net"] for r in rows if r["late_net"] is not None] or [0]), 3),
        ci95_by_day=interval(rows, key=lambda r: r["close"] // 86400),
        trades_per_day=round(len(rows) / days, 1), median_minute_volume=median(r["volume"] for r in rows),
        share_minute_volume_under_100=round(mean(r["volume"] < 100 for r in rows), 3),
        share_two_sided=round(mean(r["two_sided"] for r in rows), 3),
        median_spread=median([r["spread"] for r in rows if r["spread"] is not None] or [0]))


def analyze(family):
    series_list, start, window, width = FAMILIES[family]
    paths = sorted(RAW.glob(f"{family}-bars*.json"))
    if not paths:
        return None
    partial = any("partial" in p.name for p in paths)
    markets = list({m["ticker"]: m for p in paths for m in json.loads(p.read_text())}.values())
    state = None
    if family in COIN:
        minutes = underlying(COIN[family], date(2026, 9, 18))
        fifteen = RAW / ("KXBTC15M-markets.json" if COIN[family] == "BTC-USD" else "KXETH15M-markets.json")
        basis = []
        for m in json.loads(fifteen.read_text()) if fifteen.exists() else []:
            # A 15-minute market's strike is the index average over the minute before it opens.
            ref = minutes.get(m["open"] - 60)
            if m["floor"] and ref:
                basis.append((m["open"] + 60, m["floor"] - ref))
        state = State(minutes, basis)
    events = sorted({(m["close"], m["event"]) for m in markets})
    cut = events[int(.6 * len(events))][0]
    rows = [e for m in markets for e in entries(m, WINDOWS[window], state)]
    span = [events[0][0], cut, events[-1][0] + 1]
    days = [(span[1] - span[0]) / 86400, (span[2] - span[1]) / 86400]
    part = lambda r: 0 if r["close"] < cut else 1

    def both(keep, single=False):
        out = []
        for half in (0, 1):
            hits = [r for r in rows if part(r) == half and keep(r)]
            out.append(score(one_per_event(hits) if single else hits, days[half]))
        return out
    tried, rules = 0, {}
    for band in BANDS:
        for n in WINDOWS[window]:
            tried += 1
            keep = lambda r, band=band, n=n: r["band"] == list(band) and r["window"] == n and r["two_sided"]
            rules[f"ask {band[0]}-{band[1]}c last {n}m"] = dict(clustered=both(keep), one_per_event=both(keep, True))
    screens = {}
    if state:
        for band in BANDS:
            for n in WINDOWS[window]:
                for z in Z_GRID:
                    tried += 1
                    keep = lambda r, band=band, n=n, z=z: (r["band"] == list(band) and r["window"] == n
                                                           and r["two_sided"] and r["z"] is not None and r["z"] >= z)
                    screens[f"ask {band[0]}-{band[1]}c last {n}m z>={z}"] = dict(clustered=both(keep), one_per_event=both(keep, True))
    # Choose on the earlier part only: highest lower bootstrap bound among rules with at least 100 trades there.
    def best(table):
        ok = [(v["one_per_event"][0]["ci95"][0], k) for k, v in table.items()
              if v["one_per_event"][0].get("trades", 0) >= 100 and v["one_per_event"][0]["ci95"]]
        return max(ok)[1] if ok else None
    plain = f"ask {PLAIN[0]}-{PLAIN[1]}c last {WINDOWS[window][-1]}m"
    # How well the screen sorts risk: loss rate by z bucket for every 95-99.9c entry in the widest window.
    wide = {}
    if state:
        seen = set()
        for r in sorted(rows, key=lambda r: r["ts"]):
            if r["window"] != WINDOWS[window][-1] or not r["two_sided"] or r["z"] is None or (r["ticker"], r["side"]) in seen:
                continue
            seen.add((r["ticker"], r["side"]))
            bucket = "z<1.5" if r["z"] < 1.5 else "1.5-2.5" if r["z"] < 2.5 else "2.5-4" if r["z"] < 4 else "z>=4"
            wide.setdefault((part(r), bucket), []).append(r)
    report = dict(
        family=family, series=series_list, partial_download=partial, strike_filter=width,
        markets_with_bars=len(markets), events=len(events),
        first_close=datetime.fromtimestamp(span[0], timezone.utc).isoformat(),
        split_close=datetime.fromtimestamp(cut, timezone.utc).isoformat(),
        last_close=datetime.fromtimestamp(span[2], timezone.utc).isoformat(),
        days=[round(d, 1) for d in days], rules_tried=tried, plain_rule=plain, plain=rules[plain],
        chosen_price_rule=best(rules), chosen_screen_rule=best(screens) if screens else None,
        price_rules=rules, screen_rules=screens,
        z_buckets={f"{'early' if half == 0 else 'late'} {bucket}": dict(
            trades=len(v), losses=sum(r["payout"] < .5 for r in v), mean_price=round(mean(r["price"] for r in v), 2),
            net_cents=round(mean(r["net"] for r in v), 3)) for (half, bucket), v in sorted(wide.items())})
    save(OUT / f"{family}.json", report)
    save(OUT / f"{family}-trades.json", [r for r in rows if r["band"] == list(PLAIN) and r["two_sided"]])
    return report


def show(report):
    def line(name, pair):
        for label, s in zip(("early", "late "), pair):
            if not s.get("trades"):
                print(f"  {name:34} {label} no trades")
                continue
            print(f"  {name:34} {label} n {s['trades']:5} loss {s['full_losses']:4} win {s['win_rate']:.4f} need {s['breakeven']:.4f} "
                  f"net {s['net_cents']:+.3f}c ci {s['ci95']} dayci {s['ci95_by_day']} /day {s['trades_per_day']:6} vol {s['median_minute_volume']:.0f} "
                  f"thin {s['share_minute_volume_under_100']:.2f} spread {s['median_spread']:.1f} "
                  f"| 1m late n {s['filled_one_minute_late']} loss {s['losses_one_minute_late']} net {s['net_one_minute_late']:+.3f}")
    print(f"== {report['family']} {report['series']} events {report['events']} markets {report['markets_with_bars']} "
          f"{report['first_close'][:16]} | {report['split_close'][:16]} | {report['last_close'][:16]} tried {report['rules_tried']}")
    for name, v in report["price_rules"].items():
        line(name + " 1/ev", v["one_per_event"])
    line("PLAIN clustered", report["plain"]["clustered"])
    for key in ("chosen_price_rule", "chosen_screen_rule"):
        name = report[key]
        if name:
            table = report["price_rules"] if key == "chosen_price_rule" else report["screen_rules"]
            line(f"CHOSEN {name}", table[name]["one_per_event"])
    for k, v in report["z_buckets"].items():
        print("  z", k, v)


# ---------------------------------------------------------------------------------------------------------
# Second study: favourite calibration by time to close, taker and resting-bid versions.
V2 = RAW / "v2"
END2 = date(2026, 11, 19)
BINS = ((80, 90), (90, 95), (95, 97), (97, 99))
MAX_SPREAD = 5.0
MINUTE = dict(step=1, window=15, buckets=(10, 5, 2))
HOUR = dict(step=1, window=60, buckets=(55, 30, 15, 5))
# `old` re-reads the first study's raw candles from the fetch cache (no requests). `step` is the bar length in
# minutes, `window` the minutes of bars kept before close, `buckets` the decision times in minutes before close.
FAM2 = {
    "btc15m": dict(MINUTE, series="KXBTC15M", old="btc15m", coin="BTC-USD"),
    "eth15m": dict(MINUTE, series="KXETH15M", old="eth15m", coin="ETH-USD"),
    "sol15m": dict(MINUTE, series="KXSOL15M", start=date(2026, 10, 19)),
    "xrp15m": dict(MINUTE, series="KXXRP15M", start=date(2026, 10, 19)),
    "btc_hourly": dict(HOUR, series="KXBTCD", old="btc_hourly", coin="BTC-USD"),
    "eth_hourly": dict(HOUR, series="KXETHD", old="eth_hourly", coin="ETH-USD"),
    "sp_hourly": dict(HOUR, series="KXINXU", old="index_hourly"),
    # About 125,000 markets in 60 days: kept to 30 days and strikes within 2% of the final index value.
    "nasdaq_hourly": dict(HOUR, series="KXNASDAQ100U", start=date(2026, 10, 19), listed_from=date(2026, 9, 18), width=.02),
    "btc_daily": dict(series="KXBTCD", start=date(2026, 9, 18), step=60, window=1500, buckets=(1440, 360, 60), daily="17"),
    "eth_daily": dict(series="KXETHD", start=date(2026, 9, 18), step=60, window=1500, buckets=(1440, 360, 60), daily="17"),
    "sp_daily": dict(series="KXINXU", start=date(2026, 9, 18), step=60, window=1140, buckets=(1080, 360, 60), daily="H1600",
                     listed="KXINXU-markets.json"),
    "sp_range_daily": dict(series="KXINX", start=date(2026, 9, 18), step=60, window=1140, buckets=(1080, 360, 60)),
    "nasdaq_range_daily": dict(series="KXNASDAQ100", start=date(2026, 9, 18), step=60, window=1140, buckets=(1080, 360, 60)),
    "eurusd15m": dict(MINUTE, series="KXEURUSD15M", start=date(2026, 10, 19)),
    "usdjpy15m": dict(MINUTE, series="KXUSDJPY15M", start=date(2026, 10, 19)),
}
_stop2, _last2 = [None], [0.0]


def get2(client, path, params, slim):
    """One public read, at least 2 s after the previous one, cached in slim form under data/raw/short-dated/v2/req."""
    import config
    key = hashlib.sha256(json.dumps([path, params], sort_keys=True).encode()).hexdigest()
    cache = V2 / "req" / (key + ".json")
    if cache.exists():
        return json.loads(cache.read_text())
    if _stop2[0] and time.time() > _stop2[0]:
        raise TimeoutError("download budget used")
    for attempt in range(6):
        wait = _last2[0] + 2.0 - time.time()
        if wait > 0:
            time.sleep(wait)
        _last2[0] = time.time()
        try:
            resp = client.get(config.KALSHI_PROD_BASE + path, params=params)
        except httpx.RequestError:
            time.sleep(10)
            continue
        if resp.status_code == 429 or resp.status_code >= 500:
            time.sleep(15 * (attempt + 1))
            continue
        if resp.status_code != 200:
            raise RuntimeError(f"{path} returned HTTP {resp.status_code}")
        data = slim(resp.json())
        cache.parent.mkdir(parents=True, exist_ok=True)
        cache.write_text(json.dumps(data, separators=(",", ":")))
        return data
    raise RuntimeError(f"gave up: {path}")


def slim_markets(data):
    rows = []
    for m in data.get("markets", []):
        try:
            rows.append(dict(
                ticker=m["ticker"], event=m["event_ticker"], close=int(timestamp(m["close_time"])),
                open=int(timestamp(m["open_time"])), outcome=float(m["settlement_value_dollars"]),
                floor=m.get("floor_strike"), cap=m.get("cap_strike"), kind=m.get("strike_type"),
                final=float(m["expiration_value"]) if m.get("expiration_value") else None,
                volume=float(m.get("volume_fp") or 0)))
        except (KeyError, TypeError, ValueError):
            continue
    return dict(rows=rows, cursor=data.get("cursor") or "", count=len(data.get("markets", [])))


def compact2(bar):
    def px(side, field):
        try:
            return round(100 * float(bar[side][field + "_dollars"]), 1)
        except (KeyError, TypeError, ValueError):
            return None
    return [bar["end_period_ts"], px("yes_bid", "close"), px("yes_ask", "close"), px("price", "low"), px("price", "high"),
            float(bar.get("volume_fp") or 0)]


def slim_bars(data):
    return {m["market_ticker"]: [compact2(b) for b in m["candlesticks"]] for m in data.get("markets", [])}


def listing2(client, **params):
    rows, cursor = [], ""
    while True:
        page = get2(client, "/markets", dict(params, limit=1000, cursor=cursor) if cursor else dict(params, limit=1000), slim_markets)
        rows += page["rows"]
        cursor = page["cursor"]
        if not cursor or not page["count"]:
            return rows


def chunks2(markets, window, step):
    """Requests of at most 100 tickers and about 9,500 bars, in close-time order."""
    jobs, chunk = [], []
    for m in sorted(markets, key=lambda m: (m["close"], m["ticker"])):
        span = ((m["close"] - chunk[0]["close"]) // 60 + window) // step + 2 if chunk else 0
        if chunk and ((len(chunk) + 1) * span > 9500 or len(chunk) >= 100):
            jobs.append(chunk)
            chunk = []
        chunk.append(m)
    if chunk:
        jobs.append(chunk)
    return jobs


def old_bars(family):
    """The first study's raw one-minute candles, re-read from its fetch cache with the trade high and low kept."""
    series_list, start, window, width = FAMILIES[family]
    rows = {}
    for first, end in [(start, END)] + EXTRA.get(family, []):
        wanted = [m for s in series_list for m in listing(None, s, first, end) if near(m, width)]
        for group in chunks2(wanted, window, 1):               # the same requests the first study made
            params = dict(market_tickers=",".join(m["ticker"] for m in group),
                          start_ts=group[0]["close"] - window * 60, end_ts=group[-1]["close"], period_interval=1)
            key = hashlib.sha256(json.dumps(["/markets/candlesticks", params], sort_keys=True).encode()).hexdigest()
            cache = COLLECTION / (key + ".json")
            if not cache.exists():
                continue
            got = slim_bars(json.loads(cache.read_text()))
            for m in group:
                kept = [b for b in got.get(m["ticker"], []) if m["close"] - window * 60 < b[0] <= m["close"]]
                if kept:
                    rows[m["ticker"]] = dict(m, bars=kept)
    return list(rows.values())


def fetch2(families, minutes):
    _stop2[0] = time.time() + 60 * minutes
    with httpx.Client(timeout=60, headers={"User-Agent": "research-client"}) as client:
        for name in families:
            spec, out = FAM2[name], V2 / f"{name}.json"
            if out.exists():
                print(name, "already built", flush=True)
                continue
            try:
                if spec.get("old"):
                    rows = old_bars(spec["old"])
                else:
                    rows = new_bars(client, name, spec)
            except TimeoutError:
                print(name, "not finished: time budget used; run again to resume", flush=True)
                return
            V2.mkdir(parents=True, exist_ok=True)
            out.write_text(json.dumps(rows, separators=(",", ":")))
            print(name, "markets with bars", len(rows), "events", len({m["event"] for m in rows}), flush=True)


def new_bars(client, name, spec):
    series, start, step, window = spec["series"], spec["start"], spec["step"], spec["window"]
    if spec.get("listed"):                                   # a listing the first study already saved
        markets = [m for m in json.loads((RAW / spec["listed"]).read_text()) if m["event"].endswith(spec["daily"])]
    elif spec.get("daily"):                                  # one long-lived event a day, listed by its ticker
        markets, day = [], start
        while day < END2:
            event = f"{series}-{day.strftime('%y%b%d').upper()}{spec['daily']}"
            markets += listing2(client, event_ticker=event, status="settled")
            day = date.fromordinal(day.toordinal() + 1)
        markets = [m for m in markets if m["close"] - m["open"] <= 30 * 3600]      # drop the week-long strikes
    else:
        markets = listing2(client, series_ticker=series, status="settled", min_close_ts=midnight(spec.get("listed_from", start)),
                           max_close_ts=midnight(END2))
        markets = [m for m in markets if m["close"] >= midnight(start)]
    print(name, "listed", len(markets), "markets", flush=True)
    markets = [m for m in markets if near(m, spec.get("width"))]
    jobs, rows = chunks2(markets, window, step), []
    for n, group in enumerate(jobs):
        try:
            got = get2(client, "/markets/candlesticks", dict(
                market_tickers=",".join(m["ticker"] for m in group), start_ts=group[0]["close"] - window * 60,
                end_ts=group[-1]["close"], period_interval=step), slim_bars)
        except RuntimeError as exc:
            print("  skip", group[0]["ticker"], exc, flush=True)
            continue
        for m in group:
            kept = [b for b in got.get(m["ticker"], []) if m["close"] - window * 60 < b[0] <= m["close"]]
            if kept:
                rows.append(dict(m, bars=kept))
        if n % 50 == 0:
            print(f"  {name} {n}/{len(jobs)} requests", flush=True)
    return rows


def slim_trades(data):
    return [[timestamp(t["created_time"]), round(100 * float(t["yes_price_dollars"]), 1), float(t["count_fp"]), t.get("taker_side")]
            for t in data.get("trades", [])]


def prints2(families, minutes, sample=60):
    """Is the candle ask a price takers really paid? Trade prints in the minute after the decision time for the
    latest `sample` favourites (ask 90-99c) per family and bucket, split by taker side."""
    _stop2[0] = time.time() + 60 * minutes
    with httpx.Client(timeout=60, headers={"User-Agent": "research-client"}) as client:
        for name in families:
            spec = FAM2[name]
            markets = json.loads((V2 / f"{name}.json").read_text())
            for m in markets:
                m["bars"].sort()
                m["tick"] = 1.0
            for left in spec["buckets"][:2]:
                picked = []
                for m in sorted(markets, key=lambda m: -m["close"]):
                    for i, side in enumerate(observations(m, left)):
                        if 90 <= side["ask"] <= 99 and m["event"] not in {p[0]["event"] for p in picked}:
                            picked.append((side, i == 0))
                    if len(picked) >= sample:
                        break
                rows = []
                for side, yes in picked:
                    start = side["close"] - 60 * left
                    try:
                        trades = get2(client, "/markets/trades", dict(ticker=side["ticker"], min_ts=start, max_ts=start + 60 * spec["step"],
                                                                      limit=1000), slim_trades)
                    except TimeoutError:
                        break
                    buys = [(p if yes else 100 - p, n) for _, p, n, taker in trades if taker == ("yes" if yes else "no")]
                    size = sum(n for _, n in buys)
                    rows.append(dict(ask=side["ask"], size=size, at_or_below=sum(n for p, n in buys if p <= side["ask"] + 1e-6),
                                     vwap=sum(p * n for p, n in buys) / size if size else None, full=len(trades) >= 1000))
                hit = [r for r in rows if r["size"]]
                if not hit:
                    continue
                print(f"{name} {left}m: {len(rows)} favourites, {len(hit)} with taker buys in the next bar; median contracts bought "
                      f"{median(r['size'] for r in rows):,.0f} (about ${median(r['size'] * r['ask'] / 100 for r in rows):,.0f}), at or below the candle ask "
                      f"{median(r['at_or_below'] for r in rows):,.0f}; mean taker price minus candle ask {mean(r['vwap'] - r['ask'] for r in hit):+.2f}c; "
                      f"pages cut at 1,000 trades {sum(r['full'] for r in rows)}", flush=True)


def fee_cents(price):
    """Taker fee in cents per contract for fee_type quadratic, multiplier 1 (every series used here)."""
    return 7 * price / 100 * (1 - price / 100)


def observations(m, minutes_before, state=None):
    """Both sides of one market at one decision time: quote, payout, next-bar volume and a resting-bid outcome."""
    want, bars = m["close"] - 60 * minutes_before, m["bars"]
    at = next((i for i, b in enumerate(bars) if b[0] == want), None)
    if at is None:
        return []
    bid, ask = bars[at][1], bars[at][2]
    if bid is None or ask is None or not 0 < bid <= ask < 100 or ask - bid > MAX_SPREAD:
        return []
    after = [b for b in bars[at + 1:] if b[0] <= m["close"]]
    tick = m["tick"]
    out = []
    for yes in (True, False):
        b, a = (bid, ask) if yes else (round(100 - ask, 1), round(100 - bid, 1))
        limit = min(b + 1, round(a - tick, 1))             # 1c inside the spread, or join the bid when there is no room
        if limit <= b:
            limit = b
        if yes:                                            # strict print-through, as `prints` in paper/rules.py
            filled = any(x[3] is not None and x[3] < limit - 1e-6 for x in after)
        else:
            filled = any(x[4] is not None and 100 - x[4] < limit - 1e-6 for x in after)
        z = None
        if state and m["floor"] is not None and m["kind"] in ("greater", "greater_or_equal"):
            z = state.z(want, m["floor"], m["close"], yes)
        payout = m["outcome"] if yes else 1 - m["outcome"]
        out.append(dict(event=m["event"], close=m["close"], ticker=m["ticker"], ask=a, bid=b, mid=(a + b) / 2, payout=payout,
                        net=100 * payout - a - fee_cents(a), limit=limit, filled=filled, rest=100 * payout - limit,
                        dollars=(after[0][5] if after else 0.0) * a / 100, z=z))
    return out


def clustered(rows, value, cut):
    """Event-weighted mean (each event counts once), bootstrap 95% interval over events, and both halves."""
    by_event, when = defaultdict(list), {}
    for r in rows:
        v = value(r)
        if v is not None:
            by_event[r["event"]].append(v)
            when[r["event"]] = r["close"]
    if not by_event:
        return dict(n=0, mean=None, ci=None, halves=[None, None], t=0.0)
    names = list(by_event)
    x = np.array([mean(by_event[e]) for e in names])
    ci = None
    if len(x) >= 30:
        draws = np.sort(x[np.random.default_rng(7).integers(0, len(x), size=(2000, len(x)))].mean(axis=1))
        ci = [round(float(draws[50]), 2), round(float(draws[1949]), 2)]
    halves = []
    for late in (False, True):
        part = [v for e, v in zip(names, x) if (when[e] >= cut) == late]
        halves.append(round(float(np.mean(part)), 2) if part else None)
    t = float(x.mean() / (x.std(ddof=1) / math.sqrt(len(x)))) if len(x) >= 30 and x.std() > 0 else 0.0
    return dict(n=len(x), mean=round(float(x.mean()), 3), ci=ci, halves=halves, t=round(t, 2))


def passes(s, rows=None):
    """Interval above zero and both halves positive. A sample with fewer than three losing events gets no
    information about the loss rate from a bootstrap, so it must also survive three more full losses."""
    ok = bool(s["ci"] and s["ci"][0] > 0 and all(h is not None and h > 0 for h in s["halves"]))
    if ok and rows is not None and len({r["event"] for r in rows if r["payout"] < .5}) < 3:
        ok = s["mean"] - 100 * 3 / s["n"] > 0
    return ok


BONFERRONI_T = 3.5        # one-sided 5% shared across roughly 200 cells


def calibrate(families):
    pool = defaultdict(list)
    report, tested = {}, dict(taker=0, taker_pass=0, maker=0, maker_pass=0, control=0, control_pass=0, z=0, z_pass=0)
    for name in families:
        spec, path = FAM2[name], V2 / f"{name}.json"
        if not path.exists():
            print(name, "not downloaded")
            continue
        markets = json.loads(path.read_text())
        for m in markets:
            m["bars"].sort()
            quotes = [v for b in m["bars"] for v in b[1:3] if v is not None]
            m["tick"] = .1 if any(abs(v - round(v)) > 1e-6 for v in quotes) else 1.0
        state = None
        if spec.get("coin"):
            minutes = underlying(spec["coin"], date(2026, 9, 18))
            fifteen = RAW / ("KXBTC15M-markets.json" if spec["coin"] == "BTC-USD" else "KXETH15M-markets.json")
            basis = [(m["open"] + 60, m["floor"] - minutes[m["open"] - 60]) for m in json.loads(fifteen.read_text())
                     if m["floor"] and minutes.get(m["open"] - 60)]
            state = State(minutes, basis)
        closes = sorted({(m["close"], m["event"]) for m in markets})
        cut, days = closes[len(closes) // 2][0], max((closes[-1][0] - closes[0][0]) / 86400, 1)
        print(f"\n== {name} {spec['series']} events {len(closes)} markets {len(markets)} "
              f"{datetime.fromtimestamp(closes[0][0], timezone.utc):%Y-%m-%d} to {datetime.fromtimestamp(closes[-1][0], timezone.utc):%Y-%m-%d} "
              f"halves split {datetime.fromtimestamp(cut, timezone.utc):%Y-%m-%d}")
        print("  taker: left bin(ask)  events  obs    mid    ask    win   gap_mid   net_c   ret%   ci95(net)        halves(net)     $/day_next_bar")
        fam, maker_lines, z_lines = dict(events=len(closes), days=round(days, 1), taker=[], maker=[], z=[]), [], []
        for left in spec["buckets"]:
            pairs = [o for m in markets for o in [observations(m, left, state)] if o]
            for lo, hi in BINS:
                top = lambda a: lo <= a < hi or (hi == 99 and a == 99)
                fav = [s for p in pairs for s in p if top(s["ask"])]
                dog = [p[1 - i] for p in pairs for i in (0, 1) if top(p[i]["ask"])]
                if not fav:
                    continue
                group = "15-minute" if name.endswith("15m") else "hourly" if name.endswith("hourly") else "daily"
                kind = "crypto" if name[:3] in ("btc", "eth", "sol", "xrp") else "index and FX"
                for key in ((name, "all buckets", f"{lo}-{hi}c"), (f"{kind} {group}", "all buckets", f"{lo}-{hi}c")) + (
                        ((name, "all buckets", "90-99c"), (f"{kind} {group}", "all buckets", "90-99c")) if lo >= 90 else ()):
                    pool[key] += fav
                # Across families the same clock window is one cluster: coins and indices move together.
                window = [dict(r, event=str(r["close"])) for r in fav]
                pool[(f"{kind} {group}", f"{left}m", f"{lo}-{hi}c")] += window
                pool[(f"{kind} {group}", f"{left}m dog bid", f"{lo}-{hi}c")] += [
                    dict(r, event=str(r["close"]), net=r["rest"], ask=r["limit"]) for r in dog if r["filled"]]
                pool[(f"{kind} {group}", f"{left}m fav bid", f"{lo}-{hi}c")] += [
                    dict(r, event=str(r["close"]), net=r["rest"], ask=r["limit"]) for r in fav if r["filled"]]
                net = clustered(fav, lambda r: r["net"], cut)
                stat = lambda f: clustered(fav, f, cut)["mean"]
                mid, ask, win = stat(lambda r: r["mid"]), stat(lambda r: r["ask"]), stat(lambda r: 100 * r["payout"])
                gap = clustered(fav, lambda r: 100 * r["payout"] - r["mid"], cut)
                dollars = sum(r["dollars"] for r in fav) / days
                ok = passes(net, fav)
                tested["taker"] += net["n"] >= 30
                tested["taker_pass"] += ok
                row = dict(left=left, bin=[lo, hi], events=net["n"], obs=len(fav), mid=mid, ask=ask, win=win, gap_mid=gap["mean"],
                           gap_mid_ci=gap["ci"], net=net["mean"], ret_pct=round(100 * net["mean"] / ask, 2), ci=net["ci"],
                           halves=net["halves"], t=net["t"], dollars_per_day=round(dollars), passed=ok)
                fam["taker"].append(row)
                print(f"  {left:5}m {lo}-{hi}c {net['n']:7} {len(fav):6} {mid:6.2f} {ask:6.2f} {win:6.2f} {gap['mean']:+7.2f} {net['mean']:+7.2f} "
                      f"{row['ret_pct']:+6.2f} {str(net['ci']):16} {str(net['halves']):16} {dollars:12,.0f} t {net['t']:+5.2f} {'PASS' if ok else ''}")
                for role, group in (("fav", fav), ("dog", dog)):
                    hit = [r for r in group if r["filled"]]
                    rest = clustered(hit, lambda r: r["rest"], cut)
                    kind = "maker" if role == "fav" else "control"
                    tested[kind] += rest["n"] >= 30
                    tested[kind + "_pass"] += passes(rest, hit)
                    miss = [r["payout"] for r in group if not r["filled"]]
                    mrow = dict(left=left, bin=[lo, hi], role=role, orders=len(group), fills=len(hit),
                                limit=round(mean(r["limit"] for r in group), 2),
                                win_filled=round(100 * mean(r["payout"] for r in hit), 2) if hit else None,
                                win_unfilled=round(100 * mean(miss), 2) if miss else None,
                                net_per_fill=rest["mean"], ret_pct=round(100 * rest["mean"] / mean(r["limit"] for r in hit), 2) if hit else None,
                                ci=rest["ci"], halves=rest["halves"], events=rest["n"], t=rest["t"], passed=passes(rest, hit))
                    fam["maker"].append(mrow)
                    if hit:
                        maker_lines.append(
                            f"  {left:5}m {lo}-{hi}c {role} orders {len(group):6} fill {len(hit) / len(group):5.2f} bid {mrow['limit']:6.2f} "
                            f"win|fill {mrow['win_filled']:6.2f} win|none {str(mrow['win_unfilled']):6} net/fill {rest['mean']:+7.2f} "
                            f"({mrow['ret_pct']:+.2f}%) ci {str(rest['ci']):16} halves {str(rest['halves']):16} ev {rest['n']:5} t {rest['t']:+5.2f} {'PASS' if mrow['passed'] else ''}")
            if state:
                wide = [s for p in pairs for s in p if 90 <= s["ask"] <= 99 and s["z"] is not None]
                for label, a, b in (("z<1.5", -99, 1.5), ("1.5-2.5", 1.5, 2.5), ("2.5-4", 2.5, 4), ("z>=4", 4, 1e9)):
                    part = [r for r in wide if a <= r["z"] < b]
                    if not part:
                        continue
                    net = clustered(part, lambda r: r["net"], cut)
                    tested["z"] += net["n"] >= 30
                    tested["z_pass"] += passes(net, part)
                    ask = clustered(part, lambda r: r["ask"], cut)["mean"]
                    win = clustered(part, lambda r: 100 * r["payout"], cut)["mean"]
                    fam["z"].append(dict(left=left, z=label, events=net["n"], ask=ask, win=win, net=net["mean"], ci=net["ci"],
                                         halves=net["halves"], t=net["t"], passed=passes(net, part)))
                    z_lines.append(f"  {left:5}m ask 90-99c {label:8} events {net['n']:5} ask {ask:6.2f} win {win:6.2f} net {net['mean']:+6.2f} "
                                   f"ci {str(net['ci']):16} halves {net['halves']} t {net['t']:+5.2f} {'PASS' if passes(net, part) else ''}")
        print("  resting bid 1c inside (or joined), strict print-through; fav = favourite side, dog = other side of the same quote")
        print("\n".join(maker_lines))
        if z_lines:
            print("  distance to strike in volatility units (favourite side, taker at ask, net of fee)")
            print("\n".join(z_lines))
        report[name] = fam
    print("\n== pooled (an event counts once; family groups cluster by clock window; 'bid' rows are filled resting bids, "
          "net of nothing, price column = bid): events  price  win  gap_mid  ci  net_c  ci95(net)  halves  t")
    pooled = []
    for (who, bucket, band), rows in sorted(pool.items()):
        who = f"{who} {bucket}" if bucket != "all buckets" else who
        times = sorted({(r["close"], r["event"]) for r in rows})
        cut = times[len(times) // 2][0]
        net, gap = clustered(rows, lambda r: r["net"], cut), clustered(rows, lambda r: 100 * r["payout"] - r["mid"], cut)
        ask, win = clustered(rows, lambda r: r["ask"], cut)["mean"], clustered(rows, lambda r: 100 * r["payout"], cut)["mean"]
        ok = passes(net, rows)
        pooled.append(dict(who=who, band=band, events=net["n"], ask=ask, win=win, gap_mid=gap["mean"], gap_mid_ci=gap["ci"],
                           net=net["mean"], ci=net["ci"], halves=net["halves"], t=net["t"], passed=ok))
        print(f"  {who:34} {band:7} {net['n']:6} {ask:6.2f} {win:6.2f} {gap['mean']:+6.2f} {str(gap['ci']):15} {net['mean']:+6.2f} "
              f"{str(net['ci']):15} {str(net['halves']):15} {net['t']:+5.2f} {'PASS' if ok else ''}")
    report["pooled"] = pooled
    print("\ncells with at least 30 events and passes:", tested)
    save(OUT / "v2-calibration.json", dict(tested=tested, families=report))



def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("step", choices=("download", "underlying", "analyze", "fetch2", "calibrate", "prints2"))
    parser.add_argument("--families", nargs="+", choices=tuple(FAMILIES), default=list(FAMILIES))
    parser.add_argument("--minutes", type=float, default=30, help="Kalshi download budget")
    parser.add_argument("--v2", nargs="+", choices=tuple(FAM2), default=list(FAM2), help="families for fetch2 and calibrate")
    args = parser.parse_args()
    if args.step == "fetch2":
        fetch2(args.v2, args.minutes)
    elif args.step == "calibrate":
        calibrate(args.v2)
    elif args.step == "prints2":
        prints2(args.v2, args.minutes)
    elif args.step == "download":
        _deadline[0] = time.time() + 60 * args.minutes
        with httpx.Client(timeout=60, headers={"User-Agent": "research-client"}) as client:
            for family in args.families:
                try:
                    download(client, family, args.minutes)
                except TimeoutError:
                    print(family, "not reached: download budget used", flush=True)
    elif args.step == "underlying":
        for product in ("BTC-USD", "ETH-USD"):
            print(product, len(underlying(product, date(2026, 9, 18))), "minutes", flush=True)
    else:
        for family in args.families:
            report = analyze(family)
            if report:
                show(report)


if __name__ == "__main__":
    main()
