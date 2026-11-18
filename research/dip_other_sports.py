"""Does the tennis "dip bid" exist in other Kalshi sports?

The rule under test (see favorite_dip.py): rest a bid below a strong pre-game
favourite, get filled when the price dips in play, hold to settlement. Here it
is applied to esports, combat sports, college sports, cricket, table tennis,
Euroleague and (from bars that were already cached) European soccer.

Method, kept as close to favorite_dip.py as possible:

* One canonical market per two-outcome event (the lowest ticker); the favourite
  is whichever side has a pre-game midpoint above 50c. Three-way events (soccer)
  use the YES side of the outcome market with the highest pre-game midpoint.
* Pre-game price = median midpoint of the first 15 one-minute bars of the
  window, which must have a spread of at most 4c and a range of at most 3c.
* The window is the six hours before settlement. Where the event ticker carries
  a scheduled start (esports, cricket, table tennis) and that start is less than
  an hour after the window opens, the window is extended back to one hour before
  the scheduled start, so long matches are kept rather than dropped (dropping
  them would remove exactly the matches in which the favourite struggled). The
  15 warm-up bars must then end before the scheduled start.
* Favourites at 90c+: bids at 80, 70, 60 and at 10c / 15c below the pre-game
  price. Favourites at 80-90c: bids 10c and 15c below. One order per level per
  match; the levels are fixed in advance.
* Strict fill: a trade printed strictly below the bid in a bar after the
  warm-up. The order pays its limit; maker fee zero. Held to settlement value.
* Matches are ordered by settlement time; every cell is also shown on the
  earliest 60% and the latest 40% of the family's matches.

Public Kalshi reads only, strictly one request at a time and at most about one
per second (other processes share the rate limit). Every response is cached by
research.api_cost.fetch, so an interrupted run resumes for free.

    python research/dip_other_sports.py plan
    python research/dip_other_sports.py run --families ufc cs2 --minutes 9
    python research/dip_other_sports.py report
"""
import argparse
from collections import defaultdict
from datetime import date, datetime
import hashlib
import json
import math
from pathlib import Path
import re
from statistics import mean, median, quantiles
import sys
import time
from zoneinfo import ZoneInfo

import httpx

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from research.api_cost import COLLECTION, ROOT, fetch, midnight, save, timestamp
from research.favorite_dip import interval
from research.near_end import WINDOW, cents

OUT = ROOT / "data/processed/favorite-dip"
CUTOFF = date(2026, 9, 18)                              # earlier settlements live on /historical
START, END = date(2026, 9, 1), date(2026, 11, 18)
LIVE = dict(status="settled", limit=1000, min_close_ts=midnight(CUTOFF)-7*86400, max_close_ts=midnight(END)+7*86400)
WARMUP, PACE = 15, 1.05
EASTERN = ZoneInfo("America/New_York")
MONTHS = {m: i+1 for i, m in enumerate("JAN FEB MAR APR MAY JUN JUL AUG SEP OCT NOV DEC".split())}

# family: series, cap (most recent matches), archived (also read /historical, back to `since`),
# cached_only (never download bars: reuse what sports_survey.py already fetched).
FAMILIES = {
    "ufc": dict(series=["KXUFCFIGHT"], cap=800, archived=True, since=date(2026, 1, 1)),
    "boxing": dict(series=["KXBOXING"], cap=800, archived=True, since=date(2026, 1, 1)),
    "cs2": dict(series=["KXCS2GAME"], cap=800),
    "lol": dict(series=["KXLOLGAME"], cap=800),
    "valorant": dict(series=["KXVALORANTGAME"], cap=800),
    "dota2": dict(series=["KXDOTA2GAME"], cap=800),
    "ncaamb": dict(series=["KXNCAAMBGAME"], cap=800),
    # Holdout for the one family that looked positive: last season's games, most recent first.
    "ncaamb-last-season": dict(series=["KXNCAAMBGAME"], cap=1000, archived=True, since=date(2026, 1, 1), before=date(2026, 9, 1)),
    "ncaaf": dict(series=["KXNCAAFGAME"], cap=800),
    "cricket-t20": dict(series=["KXT20MATCH", "KXWT20MATCH"], cap=800),
    "cricket-odi": dict(series=["KXODIMATCH"], cap=800),
    "wnba": dict(series=["KXWNBAGAME"], cap=800, archived=True, since=date(2026, 1, 1)),
    "ncaawb": dict(series=["KXNCAAWBGAME"], cap=800),
    "table-tennis": dict(series=["KXLIGAPROMATCH"], cap=800),
    "euroleague": dict(series=["KXEUROLEAGUEGAME"], cap=800),
    "soccer-europe": dict(series=["KXEPLGAME", "KXLALIGAGAME", "KXSERIEAGAME", "KXBUNDESLIGAGAME", "KXLIGUE1GAME",
                                  "KXUCLGAME", "KXUELGAME"], cap=2000, archived=True, since=date(2026, 1, 1), cached_only=True),
}
last_request = [0.0]
counts = defaultdict(int)


def cached(path, params):
    key = hashlib.sha256(json.dumps([path, params], sort_keys=True).encode()).hexdigest()
    return (COLLECTION / (key + ".json")).exists()


def get(client, path, params):
    """research.api_cost.fetch, but never more than one network request per PACE seconds."""
    if cached(path, params):
        counts["cache"] += 1
        return fetch(client, path, params)
    for pause in (30, 90, None):
        wait = last_request[0]+PACE-time.time()
        if wait > 0:
            time.sleep(wait)
        last_request[0] = time.time()      # paced from the start of each request
        try:
            counts["network"] += 1
            return fetch(client, path, params)
        except RuntimeError as exc:      # fetch has already backed off five times on 429/5xx
            counts["errors"] += 1
            if pause is None or "HTTP 4" in str(exc):
                raise
            time.sleep(pause)


def pages(client, path, **params):
    cursor, seen = "", set()
    while True:
        data = get(client, path, dict(params, cursor=cursor) if cursor else params)
        yield from data["markets"]
        cursor = data.get("cursor")
        if not cursor or cursor in seen:
            return
        seen.add(cursor)


def scheduled(event):
    """Scheduled start from tickers like KXCS2GAME-26NOV171600UNDQUIN (US Eastern), else None."""
    hit = re.search(r"-(\d{2})([A-Z]{3})(\d{2})(\d{2})(\d{2})(?=[A-Z])", event)
    if not hit or hit[2] not in MONTHS or int(hit[4]) > 23 or int(hit[5]) > 59:
        return None
    try:
        return int(datetime(2000+int(hit[1]), MONTHS[hit[2]], int(hit[3]), int(hit[4]), int(hit[5]), tzinfo=EASTERN).timestamp())
    except ValueError:
        return None


def listing(client, name):
    """Settled events of a family, newest first, capped: (settle, series, event, archived, markets)."""
    spec, events = FAMILIES[name], {}
    lo = midnight(spec.get("since", START))
    for series in spec["series"]:
        sources = [(False, pages(client, "/markets", series_ticker=series, **LIVE))]
        if spec.get("archived"):
            sources.append((True, pages(client, "/historical/markets", series_ticker=series, limit=1000)))
        for archived, rows in sources:
            group = defaultdict(list)
            for m in rows:
                try:
                    m["_settle"], m["_outcome"] = int(timestamp(m["settlement_ts"])), float(m["settlement_value_dollars"])
                except (KeyError, TypeError, ValueError):
                    continue
                group[m["event_ticker"]].append(m)
            for event, ms in group.items():
                ms.sort(key=lambda m: m["ticker"])
                if lo <= ms[0]["_settle"] < midnight(spec.get("before", END)) and len(ms) >= 2:
                    events.setdefault(event, (ms[0]["_settle"], series, event, archived, ms))
    return sorted(events.values(), key=lambda e: (-e[0], e[2]))[:spec["cap"]]


def request(series, archived, m, sched):
    start = m["_settle"]-WINDOW
    if sched and sched-3600 < start:      # long match (or one settled late): open the window an hour before the scheduled start
        start = max(sched-3600, m["_settle"]-48*3600)
    path = f"/historical/markets/{m['ticker']}/candlesticks" if archived else f"/series/{series}/markets/{m['ticker']}/candlesticks"
    return path, dict(start_ts=start, end_ts=m["_settle"], period_interval=1)


def jobs(name, events):
    """Bar requests per event: one canonical market, or every outcome market of a three-way event."""
    for settle, series, event, archived, ms in events:
        sched = scheduled(event)
        yield (settle, series, event, sched, ms), [request(series, archived, m, sched) for m in (ms if len(ms) >= 3 else ms[:1])]


def pregame(bars, sched):
    """(median midpoint of the warm-up bars, None) or (None, reason)."""
    if len(bars) < WARMUP+1:
        return None, "few_bars"
    mids = []
    for b in bars[:WARMUP]:
        bid, ask = cents(b, "yes_bid", "close"), cents(b, "yes_ask", "close")
        if bid is None or ask is None or not 0 < bid <= ask < 100 or ask-bid > 4:
            return None, "no_quote_or_wide"
        mids.append((bid+ask)/2)
    if max(mids)-min(mids) > 3:
        return None, "unstable"
    if sched and bars[WARMUP-1]["end_period_ts"] > sched:
        return None, "warmup_after_scheduled_start"
    return median(mids), None


def levels(start):
    out = {}
    if start >= 90:
        out.update({"80": 80, "70": 70, "60": 60})
    if start >= 80:
        out.update({"pre-10": math.floor(start)-10, "pre-15": math.floor(start)-15})
    return out


def orders(info, bars, yes, start, three_way):
    settle, series, event, sched, ms = info
    m = bars["market"]
    payout = m["_outcome"] if yes else 1-m["_outcome"]
    drift = None      # minutes after the warm-up until the favourite's midpoint first leaves a 5c band round the pre-game price
    for b in bars["bars"][WARMUP:]:
        bid, ask = cents(b, "yes_bid", "close"), cents(b, "yes_ask", "close")
        if bid is not None and ask is not None and abs(((bid+ask)/2 if yes else 100-(bid+ask)/2)-start) > 5:
            drift = (b["end_period_ts"]-bars["bars"][WARMUP-1]["end_period_ts"])/60
            break
    base = dict(event=event, series=series, ticker=m["ticker"], settle=settle, sched=sched, start=start, payout=payout,
                three_way=three_way, window_hours=(settle-bars["bars"][0]["end_period_ts"])/3600, minutes_to_first_move=drift,
                volume=float(m.get("volume_fp") or m.get("volume") or 0))
    out = []
    for label, limit in levels(start).items():
        row = dict(base, level=label, limit=limit, strict=False, touch=False)
        for i, b in enumerate(bars["bars"][WARMUP:], WARMUP):
            high = cents(b, "price", "high")
            low = cents(b, "price", "low") if yes else (None if high is None else 100-high)
            if low is None:
                continue
            if low <= limit:
                row["touch"] = True
            if low < limit:
                row.update(strict=True, through=limit-low, minutes_before_settlement=(settle-b["end_period_ts"])/60,
                           before_scheduled_start=bool(sched and b["end_period_ts"] <= sched))
                if i+1 < len(bars["bars"]):
                    nxt = bars["bars"][i+1]
                    bid, ask = cents(nxt, "yes_bid", "close"), cents(nxt, "yes_ask", "close")
                    if bid is not None and ask is not None:
                        row["mid_after_minus_limit"] = ((bid+ask)/2 if yes else 100-(bid+ask)/2)-limit
                break
        out.append(row)
    return out or [dict(base, level=None)]      # keeps the match in the count of clean pre-game quotes


def collect(client, name, deadline):
    spec = FAMILIES[name]
    events = listing(client, name)
    rows, cover = [], defaultdict(int)
    cover["listed"] = len(events)
    for info, reqs in jobs(name, events):
        if spec.get("cached_only") and not all(cached(*r) for r in reqs):
            cover["not_cached"] += 1
            continue
        if time.time() > deadline and not all(cached(*r) for r in reqs):
            cover["cut_by_deadline"] += 1
            continue
        best = None
        for m, (path, params) in zip(info[4], reqs):
            try:
                bars = sorted(get(client, path, params)["candlesticks"], key=lambda b: b["end_period_ts"])
            except (RuntimeError, KeyError):
                bars = []
            if not bars:
                cover["no_bars"] += 1
                best = None
                break
            mid, why = pregame(bars, info[3])
            if mid is None:
                if len(reqs) == 1:
                    cover[why] += 1
                continue
            three_way = len(reqs) >= 3
            start = mid if three_way or mid > 50 else 100-mid
            if best is None or start > best[0]:
                best = (start, mid > 50 or three_way, dict(market=m, bars=bars))
        if best is None:
            cover["no_clean_pregame"] += 1
            continue
        cover["clean_pregame"] += 1
        rows += orders(info, best[2], best[1], best[0], len(reqs) >= 3)
    return rows, dict(cover)


def score(rows):
    filled = [r for r in rows if r["strict"]]
    out = dict(orders=len(rows), fills=len(filled))
    if filled:
        nets = [100*r["payout"]-r["limit"] for r in filled]
        after = [r["mid_after_minus_limit"] for r in filled if "mid_after_minus_limit" in r]
        out.update(fill_rate=len(filled)/len(rows), win_rate=mean(r["payout"] for r in filled),
                   needed=mean(r["limit"] for r in filled)/100, net_cents=mean(nets), ci95=interval(nets),
                   losses=sum(r["payout"] < .5 for r in filled), profit_per_order_cents=sum(nets)/len(rows),
                   touch_fills=sum(r["touch"] for r in rows), fills_before_scheduled_start=sum(r.get("before_scheduled_start", False) for r in filled),
                   mid_one_minute_later_vs_limit=mean(after) if after else None)
    return out


CELLS = [("90+", "80"), ("90+", "70"), ("90+", "60"), ("90+", "pre-10"), ("90+", "pre-15"), ("80-90", "pre-10"), ("80-90", "pre-15")]


def summarise(rows):
    matches = {r["event"]: r for r in rows}
    ordered = sorted(matches.values(), key=lambda r: (r["settle"], r["event"]))
    if not ordered:
        return dict(clean_matches=0, table={})
    cut = ordered[int(.6*len(ordered))]["settle"]
    moves = [r["minutes_to_first_move"] for r in ordered if r["minutes_to_first_move"] is not None]
    iso = lambda ts: datetime.fromtimestamp(ts, ZoneInfo("UTC")).isoformat()[:10]
    fav = [r for r in ordered if r["start"] >= 80]
    out = dict(clean_matches=len(ordered), first=iso(ordered[0]["settle"]), last=iso(ordered[-1]["settle"]), split_at=iso(cut),
               favourites_90_plus=sum(r["start"] >= 90 for r in ordered), favourites_80_90=sum(80 <= r["start"] < 90 for r in ordered),
               pregame_win_rate_90_plus=mean(r["payout"] for r in ordered if r["start"] >= 90) if any(r["start"] >= 90 for r in ordered) else None,
               pregame_mean_price_90_plus=mean(r["start"] for r in ordered if r["start"] >= 90) if any(r["start"] >= 90 for r in ordered) else None,
               pregame_win_rate_80_90=mean(r["payout"] for r in fav if r["start"] < 90) if any(r["start"] < 90 for r in fav) else None,
               pregame_mean_price_80_90=mean(r["start"] for r in fav if r["start"] < 90) if any(r["start"] < 90 for r in fav) else None,
               extended_windows=sum(r["window_hours"] > 6.1 for r in ordered),
               minutes_from_warmup_to_first_5c_move=dict(zip(("p10", "p50", "p90"), (quantiles(moves, n=10)[i] for i in (0, 4, 8)))) if len(moves) >= 10 else None,
               median_market_volume=median(r["volume"] for r in ordered), table={})
    for bucket, level in CELLS:
        lo, hi = (90, 101) if bucket == "90+" else (80, 90)
        sub = [r for r in rows if r.get("level") == level and lo <= r["start"] < hi]
        if sub:
            out["table"][f"{bucket} bid {level}"] = dict(all=score(sub), earlier=score([r for r in sub if r["settle"] < cut]),
                                                         later=score([r for r in sub if r["settle"] >= cut]))
    return out


def show(name, s, cover=None):
    print(f"\n{name.upper()} coverage {cover}" if cover else f"\n{name.upper()}")
    if not s["clean_matches"]:
        return print("  nothing clean")
    print(f'  clean {s["clean_matches"]} ({s["first"]}..{s["last"]}, split {s["split_at"]}) 90+: {s["favourites_90_plus"]} 80-90: {s["favourites_80_90"]} '
          f'extended windows {s["extended_windows"]} first-move min {s["minutes_from_warmup_to_first_5c_move"]} median volume {s["median_market_volume"]:.0f}')
    for lab in ("90_plus", "80_90"):
        if s[f"pregame_win_rate_{lab}"] is not None:
            print(f'  buy-and-hold pre-game {lab}: mean price {s[f"pregame_mean_price_{lab}"]:.1f} win {100*s[f"pregame_win_rate_{lab}"]:.1f}%')
    for cell, parts in s["table"].items():
        for part in ("all", "earlier", "later"):
            c = parts[part]
            if not c["fills"]:
                print(f'  {cell:16} {part:7} orders {c["orders"]:4} fills 0')
                continue
            ci = c["ci95"] and [round(v, 1) for v in c["ci95"]]
            print(f'  {cell:16} {part:7} orders {c["orders"]:4} fills {c["fills"]:4} ({c["fill_rate"]:.0%}) win {c["win_rate"]:.3f} need {c["needed"]:.3f} '
                  f'net {c["net_cents"]:+.1f}c ci {ci} per order {c["profit_per_order_cents"]:+.2f}c L {c["losses"]} '
                  f'pre-start fills {c["fills_before_scheduled_start"]} after {c["mid_one_minute_later_vs_limit"] or 0:+.1f}')


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("cmd", choices=("plan", "run", "report"))
    parser.add_argument("--families", nargs="+", choices=tuple(FAMILIES), default=list(FAMILIES))
    parser.add_argument("--minutes", type=float, default=9, help="stop downloading after this long; rerun to resume from the cache")
    args = parser.parse_args()
    path = OUT/"other-sports-summary.json"
    report = json.loads(path.read_text()) if path.exists() else {}
    if args.cmd == "report":
        for name in args.families:
            if name in report:
                show(name, report[name]["summary"], report[name]["coverage"])
        return
    deadline = time.time()+60*args.minutes
    with httpx.Client(timeout=40) as client:
        for name in args.families:
            if args.cmd == "plan":
                events = listing(client, name)
                reqs = [r for _, rs in jobs(name, events) for r in rs]
                sched = [(timestamp(i[4][0]["expected_expiration_time"])-i[3])/3600 for i, _ in jobs(name, events)
                         if i[3] and i[4][0].get("expected_expiration_time")]
                long = sum(1 for i, _ in jobs(name, events) if i[3] and i[3]-3600 < i[0]-WINDOW)
                print(f'{name:14} events {len(events):5} requests {len(reqs):5} uncached {sum(not cached(*r) for r in reqs):5} '
                      f'{datetime.fromtimestamp(events[-1][0]).date() if events else None}..{datetime.fromtimestamp(events[0][0]).date() if events else None} '
                      f'ticker-timed {len(sched)} expected-minus-scheduled h {[round(q, 1) for q in quantiles(sched, n=4)] if len(sched) > 3 else None} '
                      f'need-extended-window {long}', flush=True)
                continue
            began = time.time()
            rows, cover = collect(client, name, deadline)
            save(OUT/f"other-sports-{name}-orders.json", rows)
            report[name] = dict(series=FAMILIES[name]["series"], coverage=cover, summary=summarise(rows))
            save(path, report)
            show(name, report[name]["summary"], cover)
            print(f"  {time.time()-began:.0f}s, requests so far {dict(counts)}", flush=True)


if __name__ == "__main__":
    main()
