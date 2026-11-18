"""Archive holdout of the tennis dip rule on WTA Challenger, ITF men and ITF women.

The rule (research/favorite_dip.py): a player quoted at 90c or more before play
gets a resting bid below that price; a fill is held to settlement. This script
scores it on markets settled before 2026-09-18 (Kalshi's /historical endpoints),
which the recent-window discovery never touched.

It reuses favorite_dip's `orders`, `score` and `interval` unchanged, but
downloads strictly one request at a time, paced to about one per second, because
a live trader shares the rate limit. Every response is cached by
research.api_cost.fetch, so a rerun costs nothing and resumes where it stopped.

The archive listing comes newest first. `--max-matches` takes that many of the
most recent matches and works backwards, so a capped sample is the latest slice
of the archive, not a random one. `--minutes` stops new downloads after a time
budget (cached matches are still scored).

A bid fills only when a trade prints strictly below it in a later one-minute
bar. One order set per match. Nothing known after the match selects a trade.
"""
import argparse
from datetime import date, datetime, timezone
import hashlib
import json
from pathlib import Path
from statistics import mean, median
import sys
import time

import httpx

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from research import favorite_dip
from research.api_cost import COLLECTION, ROOT, fetch, midnight, save, timestamp
from research.favorite_dip import GROUPS, interval, orders, score
from research.near_end import WINDOW

OUT = ROOT / "data/processed/favorite-dip"
TOURS = ("wta-challenger", "itf-men", "itf-women")
FIXED = (80, 75, 70, 65)
OFFSET = 10                      # the relative bid sits this far below the rounded pre-match price
LO, HI = midnight(date(2026, 1, 1)), midnight(date(2026, 9, 18))
PACE = 1.0
last_request = [0.0]


def cached(path, params):
    key = hashlib.sha256(json.dumps([path, params], sort_keys=True).encode()).hexdigest()
    return (COLLECTION / (key + ".json")).exists()


def paced(client, path, params):
    """fetch(), one request at a time, at most one per PACE seconds, with a long pause after a 429 storm."""
    if cached(path, params):
        return fetch(client, path, params)
    for attempt in range(4):
        wait = last_request[0] + PACE - time.monotonic()
        if wait > 0:
            time.sleep(wait)
        began = last_request[0] = time.monotonic()
        try:
            data = fetch(client, path, params)
        except RuntimeError as error:
            last_request[0] = time.monotonic()
            if "returned HTTP" in str(error):      # a definite answer such as 404: do not retry
                raise
            time.sleep(60*(attempt+1))             # fetch already backed off five times on 429 or 5xx
            continue
        last_request[0] = time.monotonic()
        if last_request[0]-began > 3:              # fetch slept inside, so we were throttled: cool down
            time.sleep(10)
        return data
    raise RuntimeError(f"Read failed after long back-off: {path}")


def listing(client, series, want):
    """Archived matches, newest first, until `want` usable matches are found. Returns jobs and match volumes."""
    events, order, cursor = {}, [], ""
    while True:
        params = dict(series_ticker=series, limit=1000)
        data = paced(client, "/historical/markets", dict(params, cursor=cursor) if cursor else params)
        for m in data["markets"]:
            if m["event_ticker"] not in events:
                events[m["event_ticker"]] = []
                order.append(m["event_ticker"])
            events[m["event_ticker"]].append(m)
        cursor = data.get("cursor")
        # One page of slack so that the last wanted match has both of its markets.
        if not cursor or (want and len(order) > want+600):
            break
    jobs, volume = [], {}
    for event in order:
        group = events[event]
        m = min(group, key=lambda m: m["ticker"])
        try:
            outcome, settle = float(m["settlement_value_dollars"]), int(timestamp(m["settlement_ts"]))
        except (KeyError, TypeError, ValueError):
            continue
        if LO <= settle < HI:
            jobs.append((event, m["ticker"], settle, outcome))
            volume[event] = sum(float(x.get("volume_fp") or 0) for x in group)
    jobs.sort(key=lambda j: -j[2])
    return (jobs[:want] if want else jobs), volume, not cursor


def relative(rows, n_orders):
    """Score for bids whose level differs by match (favorite_dip.score assumes one level)."""
    filled = [r for r in rows if r["strict"]]
    if not filled:
        return dict(orders=n_orders, fills=0)
    nets = [100*r["payout"]-r["limit"] for r in filled]
    return dict(orders=n_orders, fills=len(filled), fill_rate=len(filled)/n_orders,
                win_rate=mean(r["payout"] for r in filled), needed=mean(r["limit"] for r in filled)/100,
                net_cents=mean(nets), ci95=interval(nets), losses=sum(r["payout"] < .5 for r in filled),
                profit_per_order_cents=sum(nets)/n_orders)


def quartiles(values):
    values = sorted(values)
    return None if not values else dict(p25=values[len(values)//4], median=median(values), p75=values[3*len(values)//4],
                                        mean=mean(values), under_1000=sum(v < 1000 for v in values)/len(values))


def day(ts):
    return datetime.fromtimestamp(ts, timezone.utc).date().isoformat()


def run(client, group, want, minutes):
    series = GROUPS[group][0]
    jobs, volume, whole = listing(client, series, want)
    deadline = time.monotonic()+60*minutes if minutes else None
    rows, quotes, done, no_bars, stopped = [], {}, [], 0, False
    for n, job in enumerate(jobs):
        path = f"/historical/markets/{job[1]}/candlesticks"
        params = dict(start_ts=job[2]-WINDOW, end_ts=job[2], period_interval=1)
        if deadline and time.monotonic() > deadline and not cached(path, params):
            stopped = True
            break                                   # contiguous newest-first slice: stop at the first uncached match
        try:
            bars = paced(client, path, params)["candlesticks"]
        except (RuntimeError, KeyError):
            bars = None
        done.append(job)
        if n % 200 == 0:
            print(f"  {group}: {n}/{len(jobs)} matches, back to {day(job[2])}", flush=True)
        if not bars:
            no_bars += 1
            continue
        bars = sorted(bars, key=lambda b: b["end_period_ts"])
        # A 0c bid never fills, so it returns a row exactly when the pre-match quote was two-sided and stable.
        favorite_dip.LIMITS = (0,)+FIXED
        got = orders(job, bars)
        if not got:
            continue
        start = got[0]["start"]
        quotes[job[0]] = start
        if start < 90:
            continue
        rows += [dict(r, kind="fixed") for r in got if r["limit"]]
        favorite_dip.LIMITS = (round(start)-OFFSET,)
        rows += [dict(r, kind="relative") for r in orders(job, bars)]
    save(OUT/f"{group}-archive-holdout-orders.json", rows)

    heavy = sorted({(r["settle"], r["event"]) for r in rows})
    table = {}
    for limit in FIXED:
        table[f"bid {limit}c"] = score([r for r in rows if r["kind"] == "fixed" and r["limit"] == limit], limit)
    table[f"bid {OFFSET}c below pre-match"] = relative([r for r in rows if r["kind"] == "relative"], len(heavy))
    report = dict(
        series=series, sample="every archived match" if whole and not want and not stopped else "most recent archived matches, not random",
        matches_covered=len(done), first_settle=day(done[-1][2]) if done else None, last_settle=day(done[0][2]) if done else None,
        matches_without_bars=no_bars, stable_two_sided_pregame_quote=len(quotes),
        stable_share=len(quotes)/len(done) if done else None,
        favourites_70c_plus=sum(s >= 70 for s in quotes.values()), favourites_90c_plus=len(heavy),
        contracts_traded_per_match=quartiles([volume[j[0]] for j in done]),
        contracts_traded_per_match_90c_plus=quartiles([volume[e] for _, e in heavy]),
        table=table)
    print(f"\n{group.upper()} ({series}): {len(done)} matches settled {report['first_settle']} to {report['last_settle']} "
          f"({report['sample']}); {no_bars} without bars; {len(quotes)} with a stable two-sided pre-match quote; "
          f"{len(heavy)} favourites at 90c+")
    v = report["contracts_traded_per_match"]
    if v:
        print(f"  contracts traded per match: median {v['median']:,.0f} (quartiles {v['p25']:,.0f} to {v['p75']:,.0f}), "
              f"{v['under_1000']:.0%} under 1,000")
    for name, s in table.items():
        if not s["fills"]:
            print(f"  {name:26} orders {s['orders']:4} fills 0")
            continue
        ci = s["ci95"] and [round(x, 1) for x in s["ci95"]]
        print(f"  {name:26} orders {s['orders']:4} fills {s['fills']:4} ({s['fill_rate']:.0%}) win {s['win_rate']:.3f} "
              f"need {s['needed']:.3f} net {s['net_cents']:+.1f}c ci {ci} per order {s['profit_per_order_cents']:+.2f}c")
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--groups", nargs="+", choices=TOURS, default=list(TOURS))
    parser.add_argument("--max-matches", type=int, default=0, help="most recent archived matches per tour (0 = all)")
    parser.add_argument("--minutes", type=float, default=0, help="download budget per tour (0 = none)")
    args = parser.parse_args()
    with httpx.Client(timeout=40) as client:
        for group in args.groups:
            save(OUT/f"summary-archive-holdout-{group}.json", run(client, group, args.max_matches, args.minutes))


if __name__ == "__main__":
    main()
