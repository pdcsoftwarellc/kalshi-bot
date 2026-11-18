"""Rest a bid below a strong pre-game favourite, get filled on an in-play dip, hold to settlement.

Kalshi one-minute bars only (the six hours before settlement). The pre-game
price is the median midpoint of the first 15 bars, which must be stable (a
range of at most 3c) so that the window really starts before play. For each
favourite, a bid rests at a fixed level at least 8c below that price.

A bar fills the bid in the strict sense only when a trade printed BELOW the
bid in that minute (someone sold through us); "touch" also counts a print at
the bid, where queue position would decide. The order pays the limit price.
Maker fee is taken as zero, with the full taker fee shown as a stress case.
"""
import argparse
from collections import defaultdict
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
from research.near_end import WINDOW, cents

OUT = ROOT / "data/processed/favorite-dip"
GROUPS = {"tennis": ("KXATPMATCH", "KXATPCHALLENGERMATCH", "KXWTAMATCH"), "nba": ("KXNBAGAME",),
          "mlb": ("KXMLBGAME",), "nhl": ("KXNHLGAME",), "nfl": ("KXNFLGAME",)}
START, END = date(2026, 9, 1), date(2026, 11, 16)       # the listing window already cached by near_end.py
BUCKETS = ((70, 80), (80, 90), (90, 101))
LIMITS = (50, 60, 65, 70, 75, 80)
WARMUP, MIN_DIP = 15, 8


def matches(client, series, archived=False):
    events = defaultdict(list)
    if archived:      # settled before Kalshi's historical cutoff (2026-09-18)
        listing = pages(client, "/historical/markets", "markets", series_ticker=series, limit=1000)
        path, lo, hi = "/historical/markets/{t}/candlesticks", midnight(date(2026, 1, 1)), midnight(date(2026, 9, 18))
    else:
        listing = pages(client, "/markets", "markets", series_ticker=series, status="settled", limit=1000,
                        min_close_ts=midnight(START), max_close_ts=midnight(END+timedelta(days=7)))
        path, lo, hi = "/series/"+series+"/markets/{t}/candlesticks", midnight(START), midnight(END)
    for m in listing:
        events[m["event_ticker"]].append(m)
    jobs = []
    for event, group in events.items():
        m = min(group, key=lambda m: m["ticker"])
        try:
            outcome, settle = float(m["settlement_value_dollars"]), int(timestamp(m["settlement_ts"]))
        except (KeyError, TypeError, ValueError):
            continue
        if lo <= settle < hi:
            jobs.append((event, m["ticker"], settle, outcome))

    def bars(job):
        try:
            return fetch(client, path.format(t=job[1]),
                         dict(start_ts=job[2]-WINDOW, end_ts=job[2], period_interval=1))["candlesticks"]
        except (RuntimeError, KeyError):
            return None
    with ThreadPoolExecutor(2) as pool:
        for job, candles in zip(jobs, pool.map(bars, jobs)):
            if candles:
                yield job, sorted(candles, key=lambda b: b["end_period_ts"])


def orders(job, bars):
    """One simulated resting bid per limit level for this match's pre-game favourite."""
    event, ticker, settle, outcome = job
    mids = []
    for b in bars[:WARMUP]:
        bid, ask = cents(b, "yes_bid", "close"), cents(b, "yes_ask", "close")
        if bid is None or ask is None or not 0 < bid <= ask < 100 or ask-bid > 4:
            return []
        mids.append((bid+ask)/2)
    if len(mids) < WARMUP or max(mids)-min(mids) > 3:
        return []
    yes = median(mids) > 50
    start = median(mids) if yes else 100-median(mids)
    payout = outcome if yes else 1-outcome
    out = []
    for limit in LIMITS:
        if limit > start-MIN_DIP:
            continue
        row = dict(event=event, settle=settle, start=start, limit=limit, payout=payout, strict=False, touch=False)
        for i, b in enumerate(bars[WARMUP:], WARMUP):
            # Lowest print on the favourite's side this minute.
            low = cents(b, "price", "low") if yes else (None if cents(b, "price", "high") is None else 100-cents(b, "price", "high"))
            if low is None:
                continue
            if low <= limit and not row["touch"]:
                row["touch"] = True
            if low < limit:
                row.update(strict=True, through=limit-low, minutes_before_settlement=(settle-b["end_period_ts"])/60)
                # Where the favourite was quoted a minute after the fill: negative means we were run over.
                if i+1 < len(bars):
                    bid, ask = cents(bars[i+1], "yes_bid", "close"), cents(bars[i+1], "yes_ask", "close")
                    if bid is not None and ask is not None:
                        after = (bid+ask)/2 if yes else 100-(bid+ask)/2
                        row["mid_after_minus_limit"] = after-limit
                break
        out.append(row)
    return out


def interval(nets, draws=2000):
    if len(nets) < 10:
        return None
    rng = random.Random(9)
    means = sorted(mean(rng.choices(nets, k=len(nets))) for _ in range(draws))
    return [means[int(.025*draws)], means[int(.975*draws)]]


def score(rows, limit, fill="strict"):
    filled = [r for r in rows if r[fill]]
    if not filled:
        return dict(orders=len(rows), fills=0)
    nets = [100*r["payout"]-limit for r in filled]
    fee = 7*limit/100*(1-limit/100)
    after = [r["mid_after_minus_limit"] for r in filled if "mid_after_minus_limit" in r]
    return dict(orders=len(rows), fills=len(filled), fill_rate=len(filled)/len(rows),
                win_rate=mean(r["payout"] for r in filled), needed=limit/100, net_cents=mean(nets),
                ci95=interval(nets), net_with_taker_fee=mean(nets)-fee, losses=sum(r["payout"] < .5 for r in filled),
                mid_one_minute_later_vs_limit=mean(after) if after else None,
                profit_per_order_cents=sum(nets)/len(rows))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--groups", nargs="+", choices=tuple(GROUPS), default=["tennis"])
    parser.add_argument("--archived", action="store_true", help="January to mid-September 2026 from Kalshi's historical endpoints")
    args = parser.parse_args()
    report, tag = {}, "-archived" if args.archived else ""
    with httpx.Client(timeout=40) as client:
        for group in args.groups:
            rows, seen = [], 0
            for series in GROUPS[group]:
                for job, bars in matches(client, series, args.archived):
                    seen += 1
                    rows += orders(job, bars)
            save(OUT/f"{group}{tag}-orders.json", rows)
            events = sorted({(r["settle"], r["event"]) for r in rows})
            cut = events[int(len(events)*.6)][0] if events else 0
            report[group] = dict(matches_with_bars=seen, favourites_with_clean_pregame_price=len(events), table={})
            print(f"\n{group.upper()}: {seen} matches with bars, {len(events)} with a stable pre-game price and a favourite at 70c+")
            for lo, hi in BUCKETS:
                for limit in LIMITS:
                    sub = [r for r in rows if lo <= r["start"] < hi and r["limit"] == limit]
                    if len(sub) < 30:
                        continue
                    parts = {"earlier": [r for r in sub if r["settle"] < cut], "later": [r for r in sub if r["settle"] >= cut], "all": sub}
                    cell = {name: score(part, limit) for name, part in parts.items()}
                    cell["all_touch"] = score(sub, limit, "touch")
                    report[group]["table"][f"start {lo}-{hi}c, bid {limit}c"] = cell
                    for name in ("earlier", "later", "all"):
                        s = cell[name]
                        if not s["fills"]:
                            continue
                        ci = s["ci95"] and [round(v, 1) for v in s["ci95"]]
                        print(f'  pre-game {lo}-{min(hi, 100)}c bid {limit}c {name:7} orders {s["orders"]:4} fills {s["fills"]:4} '
                              f'({s["fill_rate"]:.0%}) win {s["win_rate"]:.3f} need {s["needed"]:.2f} net {s["net_cents"]:+.1f}c ci {ci} '
                              f'run-over {s["mid_one_minute_later_vs_limit"] or 0:+.1f}c per order {s["profit_per_order_cents"]:+.2f}c')
    save(OUT/f"summary{tag}.json", report)


if __name__ == "__main__":
    main()
