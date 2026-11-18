"""Backtest bookmaker fair prices against Kalshi's prematch quote on settled matches.

For every settled ATP, Challenger and WTA match that links to one provider
fixture, compare the provider's stored bookmaker odds with Kalshi's one-minute
quote shortly before the scheduled start, then settle at the exchange payout.

Two cautions shape the reading. The provider stores one odds snapshot per
match with no timestamp, so it may be later than the Kalshi quote (lookahead
that flatters the bookmaker). And the scheduled start can be revised, so a
"prematch" quote can occasionally be in play.
"""
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from datetime import date, datetime, timedelta
import hashlib
import json
import math
from pathlib import Path
import random
from statistics import mean
import sys

import httpx

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from research.api_cost import ROOT, UTC, fetch, link_market, midnight, pages, save, timestamp
from research.forward_odds import tennis

RAW = ROOT / "data/raw/consensus-history"
OUT = ROOT / "data/processed/consensus-history"
TOURS = {"atp": ("KXATPMATCH", 265), "challenger": ("KXATPCHALLENGERMATCH", 281), "wta": ("KXWTAMATCH", 266)}
START, END = date(2026, 9, 1), date(2026, 11, 16)
HORIZONS = (60, 10)                 # minutes before the scheduled start
THRESHOLDS = (0, 1, 2, 3, 5)        # cents of edge after the entry fee
SOURCES = ("pinnacle", "median")


def provider(client, method, **params):
    key = hashlib.sha256(json.dumps([method, params], sort_keys=True).encode()).hexdigest()
    cache = RAW / f"{method}-{key[:16]}.json"
    if cache.exists():
        return json.loads(cache.read_text())
    data = tennis(client, method, **params)
    save(cache, data)
    return data


def weeks():
    day = START-timedelta(days=1)
    while day <= END+timedelta(days=1):
        last = min(day+timedelta(days=6), END+timedelta(days=1))
        yield str(day), str(last)
        day = last+timedelta(days=1)


def fair(home, away):
    try:
        a, b = float(home), float(away)
    except (TypeError, ValueError):
        return None
    return b/(a+b) if min(a, b) > 1 else None


def bookmaker(market):
    """(Pinnacle, median of at least three books) fair probability for the first player."""
    home, away = market.get("Home") or {}, market.get("Away") or {}
    each = sorted(p for b in home.keys() & away.keys() if (p := fair(home[b], away[b])) is not None)
    med = None
    if len(each) >= 3:
        mid = len(each)//2
        med = each[mid] if len(each) % 2 else (each[mid-1]+each[mid])/2
    return (fair(home.get("Pncl"), away.get("Pncl")), med)


def quote(bars, at):
    valid = [b for b in bars if at-300 < b["end_period_ts"] <= at]
    if not valid:
        return None
    bar = max(valid, key=lambda b: b["end_period_ts"])
    try:
        bid, ask = (float(bar[f"yes_{s}"]["close_dollars"]) for s in ("bid", "ask"))
    except (KeyError, TypeError, ValueError):
        return None
    return (bid, ask) if 0 < bid <= ask < 1 else None


def collect(client, tour):
    series, event_type = TOURS[tour]
    by_day, odds = defaultdict(list), {}
    for a, b in weeks():
        for f in provider(client, "get_fixtures", date_start=a, date_stop=b, timezone="UTC",
                          event_type_key=event_type) or []:
            if "Singles" in str(f.get("event_type_type")) and f.get("event_time"):
                by_day[date.fromisoformat(f["event_date"])].append(
                    {k: f.get(k) for k in ("event_key", "event_date", "event_time", "event_first_player",
                                           "event_second_player", "first_player_key", "second_player_key",
                                           "event_status")})
        got = provider(client, "get_odds", date_start=a, date_stop=b, event_type_key=event_type)
        if isinstance(got, dict):
            odds.update(got)
    pairs = defaultdict(dict)
    for m in pages(client, "/markets", "markets", series_ticker=series, status="settled", limit=1000,
                   min_close_ts=midnight(START), max_close_ts=midnight(END+timedelta(days=7))):
        pairs[m["event_ticker"]][m["ticker"]] = m
    skip, jobs = Counter(), []
    for group in pairs.values():
        pair = sorted(group.values(), key=lambda m: m["ticker"])
        m = pair[0]
        if len(pair) != 2 or not m.get("occurrence_datetime"):
            skip["incomplete_pair"] += 1
            continue
        day = datetime.fromtimestamp(timestamp(m["occurrence_datetime"]), UTC).date()
        link = link_market(pair, [f for off in (-1, 0, 1) for f in by_day[day+timedelta(days=off)]])
        if not link:
            skip["no_unique_fixture"] += 1
            continue
        f, yes_id, _ = link
        start = int(timestamp(f["event_date"]+"T"+f["event_time"]+"+00:00"))
        if not START <= datetime.fromtimestamp(start, UTC).date() < END:
            continue
        if f.get("event_status") != "Finished":
            skip["not_finished_normally"] += 1       # retirements and walkovers settle by special rules
            continue
        pinnacle, med = bookmaker((odds.get(str(f["event_key"])) or {}).get("Home/Away") or {})
        if pinnacle is None and med is None:
            skip["no_bookmaker_odds"] += 1
            continue
        try:
            outcome = float(m["settlement_value_dollars"])
        except (KeyError, TypeError, ValueError):
            skip["no_settlement"] += 1
            continue
        if outcome not in (0, 1) or start-max(HORIZONS)*60 <= timestamp(m["open_time"]):
            skip["fractional_settlement_or_listed_late"] += 1
            continue
        flip = (lambda p: p) if str(f["first_player_key"]) == yes_id else (lambda p: 1-p)
        jobs.append(dict(tour=tour, ticker=m["ticker"], fixture=f["event_key"], start=start, outcome=outcome,
                         pinnacle=None if pinnacle is None else flip(pinnacle),
                         median=None if med is None else flip(med)))

    def bars(job):
        try:
            return fetch(client, f"/series/{series}/markets/{job['ticker']}/candlesticks",
                         dict(start_ts=job["start"]-max(HORIZONS)*60-300, end_ts=job["start"], period_interval=1))["candlesticks"]
        except (RuntimeError, KeyError):
            return None
    with ThreadPoolExecutor(6) as pool:
        for job, candles in zip(jobs, pool.map(bars, jobs)):
            if candles is None:
                skip["candles_unavailable"] += 1
                continue
            job["quotes"] = {str(h): quote(candles, job["start"]-h*60) for h in HORIZONS}
    return [j for j in jobs if "quotes" in j], skip


def interval(values, draws=2000):
    """95% bootstrap interval for the mean, resampling matches."""
    if len(values) < 5:
        return None
    rng = random.Random(7)
    means = sorted(mean(rng.choices(values, k=len(values))) for _ in range(draws))
    return [means[int(.025*draws)], means[int(.975*draws)]]


def trade(row, source, horizon):
    """(edge, pnl) per contract for the better side at a large-order fee, or None."""
    q, p = row["quotes"][str(horizon)], row[source]
    if not q or p is None:
        return None
    bid, ask = q
    sides = [(p, ask, row["outcome"]), (1-p, 1-bid, 1-row["outcome"])]
    prob, price, payout = max(sides, key=lambda s: s[0]-s[1])
    fee = .07*price*(1-price)
    return prob-price-fee, payout-price-fee


def analyse(rows):
    out = {}
    for horizon in HORIZONS:
        both = [r for r in rows if r["quotes"][str(horizon)] and r["pinnacle"] is not None]
        block = dict(matches=len(both))
        if both:
            mids = [sum(r["quotes"][str(horizon)])/2 for r in both]
            brier = lambda ps: mean((p-r["outcome"])**2 for p, r in zip(ps, both))
            gap = [(m-r["outcome"])**2-(r["pinnacle"]-r["outcome"])**2 for m, r in zip(mids, both)]
            block["forecast"] = dict(kalshi_brier=brier(mids), pinnacle_brier=brier([r["pinnacle"] for r in both]),
                                     kalshi_minus_pinnacle=mean(gap), ci95=interval(gap),
                                     mean_abs_gap_cents=100*mean(abs(m-r["pinnacle"]) for m, r in zip(mids, both)),
                                     mean_spread_cents=100*mean(r["quotes"][str(horizon)][1]-r["quotes"][str(horizon)][0] for r in both))
        for source in SOURCES:
            table = {}
            scored = [(r["tour"], t) for r in rows if (t := trade(r, source, horizon))]
            for cents in THRESHOLDS:
                hits = [(tour, pnl) for tour, (edge, pnl) in scored if edge >= cents/100]
                if hits:
                    pnl = [100*p for _, p in hits]
                    table[f">={cents}c"] = dict(trades=len(hits), by_tour=dict(Counter(t for t, _ in hits)),
                                                net_cents_per_contract=mean(pnl), ci95=interval(pnl),
                                                win_rate=mean(p > 0 for p in pnl))
            block[source] = dict(priced=len(scored), thresholds=table)
        out[f"{horizon} min before start"] = block
    return out


def main():
    rows, skips = [], {}
    with httpx.Client(timeout=40) as client:
        for tour in TOURS:
            try:
                found, skip = collect(client, tour)
            except RuntimeError as exc:
                skips[tour] = dict(error=str(exc))
                continue
            rows += found
            skips[tour] = dict(skip, kept=len(found))
            print(tour, skips[tour], flush=True)
    report = dict(window=[str(START), str(END)], coverage=skips, all=analyse(rows),
                  by_tour={t: analyse([r for r in rows if r["tour"] == t]) for t in TOURS})
    save(OUT/"matches.json", rows)
    save(OUT/"summary.json", report)
    print(json.dumps(report["all"], indent=2))


if __name__ == "__main__":
    main()
