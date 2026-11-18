"""Test buying a near-certain side late in a sports market and holding to settlement.

Kalshi data only. For each settled event take one market and its one-minute
quote bars over the six hours before settlement. A side qualifies when its ask
closes a minute at or above a threshold (and at most 99c). The simulated order
buys at the *next* minute's opening ask if that is still inside the same band,
pays the taker fee, and collects the exchange settlement value.

"risen" trades are the in-play subset that can be recognised at the time: the
same side's ask was at least ten cents lower earlier in the window. Each trade
also records minutes to Kalshi's expected end time, which is published before
the game; minutes to settlement is known only afterwards and must not be used
to select trades.
Minute bars cannot show displayed size or sub-minute moves.
"""
import argparse
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from datetime import date, timedelta
import json
from pathlib import Path
import random
from statistics import mean, median
import sys

import httpx

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from research.api_cost import ROOT, fetch, midnight, pages, save, timestamp

OUT = ROOT / "data/processed/near-end"
START, END = date(2026, 9, 1), date(2026, 11, 16)
SERIES = {"tennis": ("KXATPMATCH", "KXATPCHALLENGERMATCH", "KXWTAMATCH"),
          "nfl": ("KXNFLGAME",), "nba": ("KXNBAGAME",), "nhl": ("KXNHLGAME",), "mlb": ("KXMLBGAME",),
          "college football": ("KXNCAAFGAME",), "college basketball": ("KXNCAAMBGAME",)}
THRESHOLDS = (90, 93, 95, 97, 98, 99)
WINDOW, CAP, RISE = 6*3600, 99, 10


def cents(bar, side, field):
    try:      # archived bars drop the "_dollars" suffix
        value = bar[side].get(field+"_dollars", bar[side].get(field))
        return round(100*float(value))
    except (KeyError, TypeError, ValueError, AttributeError):
        return None


def entries(bars, settle_ts, outcome, expected_ts=None):
    """First qualifying entry per side and threshold."""
    found = []
    bars = sorted(bars, key=lambda b: b["end_period_ts"])
    for yes in (True, False):
        def ask(bar, field):
            if yes:
                return cents(bar, "yes_ask", field)
            bid = cents(bar, "yes_bid", field)
            return None if bid is None else 100-bid
        def tight(bar, field):
            # Any spread is allowed: one side's ask is executable against the other side's bid.
            bid, offer = cents(bar, "yes_bid", field), cents(bar, "yes_ask", field)
            return bid is not None and offer is not None and 0 < bid <= offer < 100
        low, pending = 100, set(THRESHOLDS)
        for bar, nxt in zip(bars, bars[1:]):
            close = ask(bar, "close")
            if close is None or not tight(bar, "close"):
                continue
            first = ask(nxt, "open")
            for t in sorted(pending):
                if (t <= close <= CAP and first is not None and t <= first <= CAP and tight(nxt, "open")
                        and nxt["end_period_ts"]-bar["end_period_ts"] <= 120):
                    pending.discard(t)
                    price = first/100
                    fee = .07*price*(1-price)
                    found.append(dict(threshold=t, side="yes" if yes else "no", price=first, risen=low <= t-RISE,
                                      ts=nxt["end_period_ts"], minute_volume=float(nxt.get("volume_fp") or nxt.get("volume") or 0),
                                      minutes_to_expected_end=None if expected_ts is None else (expected_ts-nxt["end_period_ts"])/60,
                                      minutes_to_settlement=(settle_ts-nxt["end_period_ts"])/60,
                                      payout=outcome if yes else 1-outcome,
                                      net_cents=100*((outcome if yes else 1-outcome)-price-fee)))
            low = min(low, close)
    return found


def collect(client, series, archived=False):
    events = defaultdict(list)
    if archived:      # markets settled before Kalshi's historical cutoff live on separate endpoints
        listing = pages(client, "/historical/markets", "markets", series_ticker=series, limit=1000)
        candle_path = "/historical/markets/{ticker}/candlesticks"
    else:
        listing = pages(client, "/markets", "markets", series_ticker=series, status="settled", limit=1000,
                        min_close_ts=midnight(START), max_close_ts=midnight(END+timedelta(days=7)))
        candle_path = f"/series/{series}/markets/{{ticker}}/candlesticks"
    for m in listing:
        events[m["event_ticker"]].append(m)
    skip, jobs = Counter(), []
    for event, group in events.items():
        m = min(group, key=lambda m: m["ticker"])
        try:
            outcome, settle = float(m["settlement_value_dollars"]), int(timestamp(m["settlement_ts"]))
        except (KeyError, TypeError, ValueError):
            skip["no_settlement"] += 1
            continue
        if not midnight(START) <= settle < midnight(END):
            continue
        expected = m.get("expected_expiration_time") or m.get("occurrence_datetime")
        jobs.append((event, m["ticker"], settle, outcome, int(timestamp(expected)) if expected else None))

    def bars(job):
        try:
            return fetch(client, candle_path.format(ticker=job[1]),
                         dict(start_ts=job[2]-WINDOW, end_ts=job[2], period_interval=1))["candlesticks"]
        except (RuntimeError, KeyError):
            return None
    rows = []
    with ThreadPoolExecutor(8) as pool:
        for (event, ticker, settle, outcome, expected), candles in zip(jobs, pool.map(bars, jobs)):
            if not candles:
                skip["no_bars"] += 1
                continue
            for e in entries(candles, settle, outcome, expected):
                rows.append(dict(e, series=series, event=event, ticker=ticker))
    return rows, dict(skip, events=len(jobs))


def interval(by_event, draws=2000):
    """95% bootstrap interval for mean net cents, resampling whole events."""
    groups = list(by_event.values())
    if len(groups) < 10:
        return None
    rng = random.Random(11)
    means = []
    for _ in range(draws):
        pick = [v for g in rng.choices(groups, k=len(groups)) for v in g]
        means.append(mean(pick))
    means.sort()
    return [means[int(.025*draws)], means[int(.975*draws)]]


def table(rows):
    out = {}
    for t in THRESHOLDS:
        for label, keep in (("all", lambda r: True), ("risen", lambda r: r["risen"])):
            hits = [r for r in rows if r["threshold"] == t and keep(r)]
            if not hits:
                continue
            by_event = defaultdict(list)
            for r in hits:
                by_event[r["event"]].append(r["net_cents"])
            price = mean(r["price"] for r in hits)/100
            out[f">={t}c {label}"] = dict(
                trades=len(hits), mean_price_cents=100*price, win_rate=mean(r["payout"] for r in hits),
                breakeven_win_rate=price+.07*price*(1-price), losses=sum(r["payout"] < .5 for r in hits),
                net_cents_per_contract=mean(r["net_cents"] for r in hits), ci95=interval(by_event),
                net_with_1c_slippage=mean(r["net_cents"] for r in hits)-1,
                median_minutes_to_settlement=median(r["minutes_to_settlement"] for r in hits))
    return out


def main():
    global START, END
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--groups", nargs="+", choices=tuple(SERIES), default=list(SERIES))
    parser.add_argument("--start", type=date.fromisoformat, default=START)
    parser.add_argument("--end", type=date.fromisoformat, default=END)
    parser.add_argument("--name", default="summary", help="output file stem")
    parser.add_argument("--archived", action="store_true", help="use Kalshi's historical endpoints")
    args = parser.parse_args()
    START, END = args.start, args.end
    report, everything = dict(window=[str(START), str(END)], coverage={}, groups={}), []
    with httpx.Client(timeout=40) as client:
        for group in args.groups:
            rows = []
            for series in SERIES[group]:
                try:
                    found, cover = collect(client, series, args.archived)
                except RuntimeError as exc:
                    report["coverage"][series] = dict(error=str(exc))
                    continue
                report["coverage"][series] = cover
                rows += found
            everything += rows
            report["groups"][group] = table(rows)
            save(OUT/f"{args.name}-trades.json", everything)
            save(OUT/f"{args.name}.json", report)
            print(group, {s: report["coverage"].get(s) for s in SERIES[group]}, flush=True)
            for name, row in report["groups"][group].items():
                ci = row["ci95"] and [round(v, 2) for v in row["ci95"]]
                print(f'  {name:14} n {row["trades"]:5} price {row["mean_price_cents"]:.1f} win {row["win_rate"]:.4f} '
                      f'need {row["breakeven_win_rate"]:.4f} net {row["net_cents_per_contract"]:+.2f}c ci {ci} '
                      f'losses {row["losses"]} mins {row["median_minutes_to_settlement"]:.0f}', flush=True)
    report["groups"]["everything"] = table(everything)
    save(OUT/f"{args.name}.json", report)


if __name__ == "__main__":
    main()
