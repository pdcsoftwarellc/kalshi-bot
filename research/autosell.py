"""Take-profit ("auto-sell") and two-sided dip-bid tests on Kalshi sports winner markets.

Reads ONLY the on-disk cache written by research/favorite_dip.py's matches()
(same request parameters, so the same cache keys); it never touches the network.

Pre-game price: median midpoint of the first 15 one-minute bars of the six-hour
window before settlement (two-sided, spread <= 4c, range <= 3c), as in favorite_dip.py.

H1/H2  buy one side at bar 15 (the first bar after the warm-up) and rest a sell at a target.
       entry "ask": pay the ask at that bar's open (taker fee).
       entry "bid": rest a bid 1c below that ask for at most 60 minutes; it fills only when
                    a trade prints strictly below it in a later minute (maker, zero fee).
       The sell fills only when a trade prints strictly above the target in a minute after
       the entry. Unfilled positions collect the settlement value.
H3     in matches starting 40-60c, rest a bid at the same level on both sides for the whole
       match; strict fills only; both filled = locked 100 - 2*level.

The other side of each market is the complement of the recorded one.
"""
import argparse
from collections import defaultdict
from datetime import date, datetime, timedelta, timezone
import hashlib
import json
from pathlib import Path
import random
from statistics import mean, median
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from research.api_cost import COLLECTION, ROOT, midnight, save, timestamp
from research.favorite_dip import END, GROUPS, START
from research.near_end import WINDOW, cents

OUT = ROOT / "data/processed/autosell"
WARMUP, ENTRY_WAIT = 15, 60
# The whole grid, fixed before any result was seen.
DOG_BUCKETS = ((10, 20), (20, 35), (35, 50))
DOG_TARGETS = ("+10", "+15", "+20", "x1.5", "x2")
FAV_BUCKETS = ((55, 65), (65, 80))
FAV_TARGETS = ("+8", "+12", "+15")
STRADDLE_BUCKETS = (("45-55", 45, 55), ("40-45/55-60", 40, 60))
STRADDLE_LEVELS = (35, 30, 25)


def cached(path, params):
    key = hashlib.sha256(json.dumps([path, params], sort_keys=True).encode()).hexdigest()
    file = COLLECTION / (key + ".json")
    if not file.exists():
        return None
    try:
        return json.loads(file.read_text())
    except ValueError:      # being written right now by another job
        return None


def listing(path, **params):
    cursor = ""
    while True:
        data = cached(path, dict(params, cursor=cursor) if cursor else params)
        if data is None:
            return
        yield from data["markets"]
        cursor = data.get("cursor")
        if not cursor:
            return


def matches(series, archived):
    """favorite_dip.matches without the network: yields only what is already on disk."""
    events = defaultdict(list)
    if archived:
        rows = listing("/historical/markets", series_ticker=series, limit=1000)
        path, lo, hi = "/historical/markets/{t}/candlesticks", midnight(date(2026, 1, 1)), midnight(date(2026, 9, 18))
    else:
        rows = listing("/markets", series_ticker=series, status="settled", limit=1000,
                       min_close_ts=midnight(START), max_close_ts=midnight(END+timedelta(days=7)))
        path, lo, hi = "/series/"+series+"/markets/{t}/candlesticks", midnight(START), midnight(END)
    for m in rows:
        events[m["event_ticker"]].append(m)
    listed = missing = 0
    for event, group in events.items():
        m = min(group, key=lambda m: m["ticker"])
        try:
            outcome, settle = float(m["settlement_value_dollars"]), int(timestamp(m["settlement_ts"]))
        except (KeyError, TypeError, ValueError):
            continue
        if not lo <= settle < hi:
            continue
        listed += 1
        data = cached(path.format(t=m["ticker"]), dict(start_ts=settle-WINDOW, end_ts=settle, period_interval=1))
        if not data or not data.get("candlesticks"):
            missing += 1
            continue
        yield (event, m["ticker"], settle, outcome), sorted(data["candlesticks"], key=lambda b: b["end_period_ts"])
    print(f"    {series}{' archived' if archived else ''}: {listed} listed, {missing} without cached bars", file=sys.stderr)


def fee(price):
    return 7*price/100*(1-price/100)


def target_price(rule, entry):
    return entry+int(rule[1:]) if rule[0] == "+" else round(entry*float(rule[1:]))


def study(job, bars):
    """Everything the three hypotheses need from one match, or None without a clean pre-game price."""
    event, ticker, settle, outcome = job
    mids = []
    for b in bars[:WARMUP]:
        bid, ask = cents(b, "yes_bid", "close"), cents(b, "yes_ask", "close")
        if bid is None or ask is None or not 0 < bid <= ask < 100 or ask-bid > 4:
            return None
        mids.append((bid+ask)/2)
    if len(mids) < WARMUP or max(mids)-min(mids) > 3 or len(bars) <= WARMUP+1:
        return None
    first = bars[WARMUP]
    bid, ask = cents(first, "yes_bid", "open"), cents(first, "yes_ask", "open")
    quoted = bid is not None and ask is not None and 0 < bid <= ask < 100 and ask-bid <= 4
    rest = bars[WARMUP+1:]
    out = dict(event=event, settle=settle, trades=[], straddle=[])
    for yes in (True, False):
        start = median(mids) if yes else 100-median(mids)
        payout = 100*(outcome if yes else 1-outcome)
        highs, lows = [], []       # trade prints on this side, minute by minute after the entry bar
        for b in rest:
            hi, lo = cents(b, "price", "high"), cents(b, "price", "low")
            if hi is None or lo is None:
                highs.append(None), lows.append(None)
            else:
                highs.append(hi if yes else 100-lo), lows.append(lo if yes else 100-hi)
        out.setdefault("sides", []).append(dict(start=start, payout=payout, lows=lows))
        if not quoted:
            continue
        price = ask if yes else 100-bid
        family = ("dog", DOG_BUCKETS, DOG_TARGETS) if start < 50 else ("fav", FAV_BUCKETS, FAV_TARGETS)
        bucket = next((b for b in family[1] if b[0] <= start < b[1]), None)
        if bucket is None or not 1 < price < 99:
            continue
        for mode in ("ask", "bid"):
            if mode == "ask":
                entry, at, touched_entry = price, 0, True
            else:
                entry = price-1
                window = lows[:ENTRY_WAIT]
                at = next((i+1 for i, low in enumerate(window) if low is not None and low < entry), None)
                touched_entry = any(low is not None and low <= entry for low in window)
            for rule in family[2]:
                target = target_price(rule, entry)
                if target > 99:
                    continue
                row = dict(family=family[0], bucket=bucket, mode=mode, rule=rule, entry=entry, target=target,
                           filled=at is not None, entry_touch=touched_entry, payout=payout, exit=False, exit_touch=False)
                if at is not None:
                    later = highs[at:]
                    row["exit"] = any(h is not None and h > target for h in later)
                    row["exit_touch"] = any(h is not None and h >= target for h in later)
                out["trades"].append(row)
    start = median(mids)
    if 40 <= start <= 60:
        label = "45-55" if 45 <= start <= 55 else "40-45/55-60"
        for level in STRADDLE_LEVELS:
            legs = []
            for side in out["sides"]:
                legs.append(dict(strict=any(low is not None and low < level for low in side["lows"]),
                                 touch=any(low is not None and low <= level for low in side["lows"]),
                                 payout=side["payout"]))
            out["straddle"].append(dict(bucket=label, level=level, legs=legs))
    del out["sides"]
    return out


def interval(values, draws=2000):
    if len(values) < 10:
        return None
    rng = random.Random(9)
    means = sorted(mean(rng.choices(values, k=len(values))) for _ in range(draws))
    return [round(means[int(.025*draws)], 2), round(means[int(.975*draws)], 2)]


def score_trades(rows, mode):
    """rows: one per match in the bucket (an order). Net is per filled contract."""
    live = [r for r in rows if r["filled"]]
    if not live:
        return dict(matches=len(rows), trades=0)
    entry_fee = (lambda r: fee(r["entry"])) if mode == "ask" else (lambda r: 0)
    nets = [(r["target"] if r["exit"] else r["payout"])-r["entry"]-entry_fee(r) for r in live]
    hold = [r["payout"]-r["entry"]-entry_fee(r) for r in live]
    stress = [(r["target"]-fee(r["target"]) if r["exit"] else r["payout"])-r["entry"]-fee(r["entry"]) for r in live]
    edge = [a-b for a, b in zip(nets, hold)]
    return dict(matches=len(rows), trades=len(live), entry_fill_rate=len(live)/len(rows),
                entry_touch_rate=mean(r["entry_touch"] for r in rows),
                exit_rate=mean(r["exit"] for r in live), exit_touch_rate=mean(r["exit_touch"] for r in live),
                win_rate_if_held=mean(r["payout"] for r in live)/100, mean_entry=mean(r["entry"] for r in live),
                net=mean(nets), ci95=interval(nets), hold_net=mean(hold), hold_ci95=interval(hold),
                vs_hold=mean(edge), vs_hold_ci95=interval(edge), net_all_taker=mean(stress),
                per_match=sum(nets)/len(rows))


def score_straddle(rows, level):
    if not rows:
        return dict(matches=0)
    nets, stress, counts, touch = [], [], [0, 0, 0], [0, 0, 0]
    for r in rows:
        filled = [leg for leg in r["legs"] if leg["strict"]]
        counts[len(filled)] += 1
        touch[sum(leg["touch"] for leg in r["legs"])] += 1
        nets.append(sum(leg["payout"]-level for leg in filled))
        stress.append(sum(leg["payout"]-level-fee(level) for leg in filled))
    n = len(rows)
    one = [x for r, x in zip(rows, nets) if sum(leg["strict"] for leg in r["legs"]) == 1]
    return dict(matches=n, none=counts[0]/n, one=counts[1]/n, both=counts[2]/n,
                touch_none=touch[0]/n, touch_one=touch[1]/n, touch_both=touch[2]/n,
                locked_per_pair=100-2*level, single_leg_net=mean(one) if one else None,
                single_leg_win_rate=mean(x > 0 for x in one) if one else None,
                per_match=mean(nets), ci95=interval(nets), per_match_all_taker=mean(stress),
                per_active_match=sum(nets)/max(1, n-counts[0]))


def months(rows):
    span = (max(r["settle"] for r in rows)-min(r["settle"] for r in rows))/86400/30.4
    return max(span, .5)


def report(name, studies):
    studies.sort(key=lambda s: s["settle"])
    cut = studies[int(len(studies)*.6)]["settle"]
    day = lambda ts: datetime.fromtimestamp(ts, timezone.utc).strftime("%Y-%m-%d")
    span = months(studies)
    head = dict(matches=len(studies), first=day(studies[0]["settle"]), split=day(cut), last=day(studies[-1]["settle"]), months=round(span, 2))
    print(f"\n=== {name.upper()}: {len(studies)} matches with a clean pre-game price, {head['first']} to {head['last']}, later part starts {head['split']}")
    parts = lambda rows: (("early", [r for r in rows if r["settle"] < cut]), ("late", [r for r in rows if r["settle"] >= cut]), ("all", rows))
    table = {}
    for family, buckets, rules in (("dog", DOG_BUCKETS, DOG_TARGETS), ("fav", FAV_BUCKETS, FAV_TARGETS)):
        print(f"--- {'H1 underdog' if family == 'dog' else 'H2 favourite'} auto-sell")
        for bucket in buckets:
            for mode in ("ask", "bid"):
                for rule in rules:
                    rows = [dict(t, settle=s["settle"]) for s in studies for t in s["trades"]
                            if t["family"] == family and t["bucket"] == bucket and t["mode"] == mode and t["rule"] == rule]
                    if len(rows) < 30:
                        continue
                    cell = {part: score_trades(sub, mode) for part, sub in parts(rows)}
                    cell["trades_per_month"] = cell["all"]["trades"]/span
                    table[f"{family} {bucket[0]}-{bucket[1]}c entry={mode} sell {rule}"] = cell
                    for part in ("early", "late", "all"):
                        s = cell[part]
                        if not s["trades"]:
                            continue
                        print(f'  {bucket[0]}-{bucket[1]}c {mode} {rule:5} {part:5} m {s["matches"]:4} tr {s["trades"]:4} exit {s["exit_rate"]:4.0%} '
                              f'(touch {s["exit_touch_rate"]:4.0%}) net {s["net"]:+6.2f} ci {s["ci95"]} hold {s["hold_net"]:+6.2f} '
                              f'vs hold {s["vs_hold"]:+6.2f} ci {s["vs_hold_ci95"]} taker {s["net_all_taker"]:+6.2f} per match {s["per_match"]:+6.2f}')
    print("--- H3 two-sided dip bids")
    for label, _, _ in STRADDLE_BUCKETS:
        for level in STRADDLE_LEVELS:
            rows = [dict(t, settle=s["settle"]) for s in studies for t in s["straddle"] if t["bucket"] == label and t["level"] == level]
            if len(rows) < 30:
                continue
            cell = {part: score_straddle(sub, level) for part, sub in parts(rows)}
            cell["matches_per_month"] = len(rows)/span
            table[f"straddle start {label} bids {level}c"] = cell
            for part in ("early", "late", "all"):
                s = cell[part]
                if not s["matches"]:
                    continue
                single = "n/a" if s["single_leg_net"] is None else f'{s["single_leg_net"]:+.1f}'
                print(f'  start {label:11} bids {level} {part:5} m {s["matches"]:4} none {s["none"]:4.0%} one {s["one"]:4.0%} both {s["both"]:4.0%} '
                      f'(touch both {s["touch_both"]:4.0%}) single-leg net {single} per match {s["per_match"]:+6.2f} ci {s["ci95"]} '
                      f'taker {s["per_match_all_taker"]:+6.2f}')
    return dict(head, table=table)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--groups", nargs="+", choices=tuple(GROUPS), default=list(GROUPS))
    args = parser.parse_args()
    summary = {}
    for group in args.groups:
        sets = {"live": [], "archived": []}
        for kind in sets:
            if kind == "archived" and group == "nfl":
                continue
            seen = 0
            for series in GROUPS[group]:
                for job, bars in matches(series, kind == "archived"):
                    seen += 1
                    s = study(job, bars)
                    if s:
                        sets[kind].append(s)
            print(f"  {group} {kind}: {seen} matches with cached bars, {len(sets[kind])} usable", file=sys.stderr)
        # Tennis: the autumn sample is the test and the archive, as far as it is cached, a holdout.
        # Other sports: one chronological sample, January to November.
        if group == "tennis":
            samples = {"tennis": sets["live"], "tennis-archive-holdout": sets["archived"]}
        else:
            samples = {group: sets["live"]+sets["archived"]}
        for name, studies in samples.items():
            if len(studies) >= 50:
                summary[name] = report(name, studies)
    save(OUT/"summary.json", summary)


if __name__ == "__main__":
    main()
