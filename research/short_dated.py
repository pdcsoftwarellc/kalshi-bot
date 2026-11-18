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


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("step", choices=("download", "underlying", "analyze"))
    parser.add_argument("--families", nargs="+", choices=tuple(FAMILIES), default=list(FAMILIES))
    parser.add_argument("--minutes", type=float, default=30, help="Kalshi download budget")
    args = parser.parse_args()
    if args.step == "download":
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
