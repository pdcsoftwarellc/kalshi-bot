"""The 80c dip bid on heavy favourites in WOMEN'S college basketball, on all of last season.

The rule (live on men's college basketball): a team is priced 90c or more just
before tip-off; rest a bid at 80c once the game starts; a fill counts only when
a trade prints strictly below the bid in a later one-minute bar; hold to
settlement. For the women's series KXNCAAWBGAME the repo had only 1-18
November 2026. Kalshi's archive holds the whole 2025-26 season (4,078 games);
this script downloads it and scores the rule on it.

Conventions are those of research/favourite_contexts.py, whose functions are
called directly (study, start_of, Table, events_of, chunk):

* One market per game, the alphabetically first; the favourite is the side
  whose pre-game midpoint is above 50c.
* Window: the six hours before settlement, one-minute bars.
* "3h" price: median midpoint of the first 15 bars of the window (each
  two-sided, spread at most 4c, all within 3c), which must end before the
  start. A game without it has no row at all (favourite_contexts' rule).
* "tip" price: median midpoint of the last 10-15 bars ending 25 minutes before
  the estimated start (last season) or two minutes before the scheduled start
  (this season), same spread and range limits.
* Start: last season's listings carry no schedule, so the start is the close
  minus 130 minutes, known to about 20 minutes. This season it is Kalshi's
  expected expiration minus three hours.
* A resting bid fills when a bar ending more than a minute after the start
  has a trade strictly below the bid. "Touch" also counts a print at the bid.
  Fixed bids need to be at least 5c below the price.
* The series' fee type is "quadratic": makers pay nothing. Net per fill is
  100 x settlement - bid. Prices of 99c and above are excluded.
* A second, looser definition ("tip-only") drops the requirement of a clean
  quote three hours before the game, as the live rule (paper/rules.py) only
  looks at the last quotes before play. It is shown as a sensitivity.
* A third ("tip-cal") corrects last season's estimated start. The first
  visible price movement (`onset`) is measured in both seasons against the
  start; this season's schedule is known, so the difference between the two
  medians is the error of "close minus 130 minutes" on last season's women's
  games, and the start is moved by it.
* Games settled more than an hour after their close (late settlements) are
  read on a window placed on the close instead and flagged.

Network: public reads only, strictly one at a time, --pace seconds apart
(1.2 asked for; run at 2.0), a 30-second pause (then longer) after any HTTP
429: a live trader and a recorder share the limit. Every 429 arrived at
second 42-44 of a minute, at paces of 1.2, 2.0, 2.5 and 3.1 seconds alike, so
they come from another process's burst: after one, the reader stays silent
for seven seconds around that second of every later minute, and it slows its
own pace only when two 429s come less than 150 seconds apart. Responses go into the shared cache
data/raw/cost-research-v1 in the layout of research.api_cost.fetch. Archived
games are read in a fixed pseudo-random order (sha256 of the event ticker), so
a run cut short leaves a uniform sample of the season.

Result of the run of 18 November 2026 (2,292 of last season's 4,078 games, a
uniform 56% in sha256 order, and all 742 of 1-18 November 2026): 80c bid on
90-99c favourites at the tip, last season 288 orders, 80 fills, 83.8% won,
+3.8c a fill (bootstrap -5.0 to +11.2); with November 2026, 118 fills, 86.4%,
+6.4c (-0.3 to +12.4); first half +5.2c (-4.6 to +13.4), second +7.7c. The
first-half interval includes zero, so the rule does not pass. Last season's
estimated start proved about 13 minutes late for women's games; with it
corrected the season gives 99 fills at +5.9c (-1.2 to +11.9).

    .venv/bin/python research/wcbb_backfill.py download --minutes 9     (repeat until nothing is left; 1,786 games remain)
    .venv/bin/python research/wcbb_backfill.py recent --minutes 9       (November 2026, batch endpoint)
    .venv/bin/python research/wcbb_backfill.py build                    (offline: cache -> rows.json)
    .venv/bin/python research/wcbb_backfill.py trades --minutes 9       (trade record around the 80c fills)
    .venv/bin/python research/wcbb_backfill.py report
"""
import argparse
from collections import Counter, defaultdict
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import sys
import time

import httpx
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import config
from research.api_cost import ROOT, timestamp
from research.calibration_map import cache_file, quote, ts, volume
from research.near_end import WINDOW, cents
from research import favourite_contexts as fc
from research.dip_new_families import measure, wilson

OUT = ROOT / "data/processed/wcbb-backfill"
SERIES, FAM = "KXNCAAWBGAME", "cbb-women"
WARMUP = 15
LEVELS = ("85", "80", "75", "70", "price-10")
RNG = np.random.default_rng(41)
SPAN = 6*3600+120
pace = [1.2]
throttle_at = [0.0]
QUIET = 3.5      # observed: every HTTP 429 arrived at second 42-44 of a minute, whatever our pace (another process's burst)
last_request = [0.0]
used = Counter()


# ---------------------------------------------------------------- network

def read(client, path, params):
    """One paced public read into the shared cache (the file layout of research.api_cost.fetch, whose own retry would
    come back within seconds). 30 seconds (then 60, 120, 240) after HTTP 429 or a server error."""
    file = cache_file(path, params)
    if file.exists():
        used["cache"] += 1
        return json.loads(file.read_text())
    for pause in (30, 60, 120, 240):
        wait = last_request[0]+pace[0]-time.time()
        if wait > 0:
            time.sleep(wait)
        if throttle_at[0]:      # stay silent around the second of the minute at which the other process bursts
            phase = (time.time()-throttle_at[0]+QUIET) % 60
            if phase < 2*QUIET:
                used["quiet windows"] += 1
                time.sleep(2*QUIET-phase)
        last_request[0] = time.time()
        used["network"] += 1
        try:
            resp = client.get(config.KALSHI_PROD_BASE+path, params=params)
        except httpx.RequestError:
            resp = None
        if resp is not None and resp.status_code == 200:
            data = resp.json()
            temp = file.with_suffix(".tmp")
            temp.write_text(json.dumps(data))
            temp.replace(file)
            return data
        if resp is not None and resp.status_code != 429 and resp.status_code < 500:
            used["http4xx"] += 1
            raise RuntimeError(f"{path} returned HTTP {resp.status_code}")
        used["throttled"] += 1
        if throttle_at[0] and time.time()-throttle_at[0] < 150:      # 429s closer together than the other process's cycle: our own pace is too fast
            pace[0] = min(4.0, pace[0]*1.25)
        throttle_at[0] = time.time()
        print(f"   {time.strftime('%H:%M:%S')} after {used['network']} requests: throttled ({resp.status_code if resp is not None else 'network error'}): pausing {pause}s, pace now {pace[0]:.2f}s", flush=True)
        time.sleep(pause)
    raise RuntimeError(f"Read failed after long back-off: {path}")


def games(archived):
    return sorted((e for e in fc.events_of(SERIES, archive=True).values() if e[3] == archived), key=lambda e: fc.order_key(e[2]))


def late(e):
    close = ts(e[4][0], "close_time")
    return bool(close and e[0]-close > 3600)


def request(e):
    """The archive read of one game: favourite_contexts' six hours before settlement, or, for a late settlement, six hours
    ending 31 minutes after the close (the usual gap between close and settlement last season) to an hour after the close."""
    m = e[4][0]
    path = f"/historical/markets/{m['ticker']}/candlesticks"
    if late(e):
        close = ts(m, "close_time")
        return path, dict(start_ts=close+1860-WINDOW, end_ts=close+3600, period_interval=1)
    return path, dict(start_ts=e[0]-WINDOW, end_ts=e[0], period_interval=1)


def download(minutes):
    deadline, todo = time.time()+60*minutes, [e for e in games(True) if not cache_file(*request(e)).exists()]
    print(f"archived games {len(games(True))}, still to read {len(todo)}", flush=True)
    with httpx.Client(timeout=40) as client:
        for n, e in enumerate(todo):
            if time.time() > deadline:
                break
            try:
                read(client, *request(e))
            except RuntimeError as exc:
                used["failed"] += 1
                print("   read failed", e[2], exc, flush=True)
                if used["failed"] >= 5 and used["failed"] > n/2:
                    break
    left = sum(not cache_file(*request(e)).exists() for e in games(True))
    print(f"done: {dict(used)}; pace {pace[0]:.2f}s; still to read {left}")


def recent_requests():
    """Batch requests for this season's games (one-minute bars of the canonical market), grouped as favourite_contexts does."""
    jobs = [(e[0]-WINDOW, [e[4][0]["ticker"]]) for e in games(False)]
    return [("/markets/candlesticks", dict(market_tickers=",".join(t for _, ts_ in g for t in ts_), start_ts=g[0][0], end_ts=g[-1][0]+SPAN,
                                           period_interval=1)) for g in fc.chunk(jobs, SPAN, 60)]


def recent(minutes):
    deadline, reqs = time.time()+60*minutes, recent_requests()
    todo = [r for r in reqs if not cache_file(*r).exists()]
    print(f"this season's games {len(games(False))}, batch requests {len(reqs)}, still to read {len(todo)}", flush=True)
    with httpx.Client(timeout=90) as client:
        for r in todo:
            if time.time() > deadline:
                break
            try:
                read(client, *r)
                time.sleep(2)      # a batch is a heavier read
            except RuntimeError as exc:
                print("   batch failed", exc, flush=True)
    print(f"done: {dict(used)}; still to read {sum(not cache_file(*r).exists() for r in reqs)}")


# ---------------------------------------------------------------- rows

def bars_of(e, batches):
    """One-minute bars of the game's canonical market from the cache, or None when not downloaded."""
    if e[3]:
        file = cache_file(*request(e))
        if not file.exists():
            return None
        bars = sorted(json.loads(file.read_text()).get("candlesticks") or [], key=lambda b: b["end_period_ts"])
        return [b for b in bars if b["end_period_ts"] <= e[0]]
    bars = batches.get(e[4][0]["ticker"])
    if bars is None:
        return None
    anchor = e[0]-WINDOW
    return [b for b in bars if anchor < b["end_period_ts"] <= min(anchor+SPAN, e[0])]


def tip_only(e, bars, start):
    """The price as the live rule sees it, without requiring a clean quote three hours earlier: (price, spread, yes) or None."""
    margin = 25*60 if e[3] else 120
    quotes = [quote(b) for b in bars if b["end_period_ts"] <= start-margin][-WARMUP:]
    if len(quotes) < 10 or not all(q and q[1]-q[0] <= 4 for q in quotes):
        return None
    mids = [(q[0]+q[1])/2 for q in quotes]
    if max(mids)-min(mids) > 3:
        return None
    mid = float(np.median(mids))
    yes = mid > 50
    return (mid if yes else 100-mid), float(np.mean([q[1]-q[0] for q in quotes])), yes


def plays(bars, yes, start, skip, archived):
    """Bars with a trade from the end of the tip quote (25 minutes before last season's estimated start, two minutes before
    this season's scheduled one). favourite_contexts counts those ending more than a minute after the start; fill_of does too
    unless told otherwise. [end of the bar (unix), favourite low, favourite high, contracts, favourite midpoint at its close]."""
    out = []
    for b in bars[skip:]:
        lo, hi = cents(b, "price", "low"), cents(b, "price", "high")
        if lo is None or hi is None or b["end_period_ts"] <= start-(25*60 if archived else 120):
            continue
        q = quote(b)
        mid = None if q is None else ((q[0]+q[1])/2 if yes else 100-(q[0]+q[1])/2)
        out.append([b["end_period_ts"], lo if yes else 100-hi, hi if yes else 100-lo, volume(b), mid])
    return out


def onset(bars, start):
    """Minutes from the estimated start to the first bar at which the midpoint has travelled 4c or more within ten minutes
    (searched from two hours before): where play visibly begins. Used only to check last season's start estimate against
    this season's, whose schedule is known."""
    path = []
    for b in bars:
        q = quote(b)
        if q is None or q[1]-q[0] > 6 or b["end_period_ts"] < start-7200:
            continue
        mid = (q[0]+q[1])/2
        if path:
            path.append((b["end_period_ts"], abs(mid-last)))
        else:
            path.append((b["end_period_ts"], 0.0))
        last = mid
        if sum(d for t, d in path if t > b["end_period_ts"]-600) >= 4:
            return (b["end_period_ts"]-start)/60
    return None


def build():
    batches = {}
    for r in recent_requests():
        if cache_file(*r).exists():
            for x in json.loads(cache_file(*r).read_text()).get("markets", []):
                batches[x["market_ticker"]] = sorted(x.get("candlesticks") or [], key=lambda b: b["end_period_ts"])
    rows, cover = [], Counter()
    for archived in (True, False):
        for e in games(archived):
            season = "2025-26" if archived else "2026-27"
            cover[season, "listed"] += 1
            bars = bars_of(e, batches)
            if bars is None:
                cover[season, "not downloaded"] += 1
                continue
            m = e[4][0]
            start = fc.start_of(FAM, e)
            row = fc.study(FAM, e, {m["ticker"]: bars}, True)
            row.update(event=e[2], settle=e[0], archived=e[3], season=season, binary=fc.binary(e), late=late(e), bars=len(bars), ticker=m["ticker"], est_start=start,
                       pre_bars=sum(b["end_period_ts"] <= start for b in bars), pre_quoted=sum(b["end_period_ts"] <= start and quote(b) is not None for b in bars),
                       total_volume=float(m.get("volume_fp") or m.get("volume") or 0), outcome_first=m["_outcome"],
                       codes=[x["ticker"].rsplit("-", 1)[-1] for x in e[4]], close=ts(m, "close_time"), expected=ts(m, "expected_expiration_time"))
            row["onset"] = onset(bars, start)
            if not row.get("skip"):
                row["plays"] = plays(bars, row["yes"], start, WARMUP, e[3])
                row["before"] = min((x for x in ((cents(b, "price", "low") if row["yes"] else (None if cents(b, "price", "high") is None else 100-cents(b, "price", "high")))
                                                 for b in bars[WARMUP:] if b["end_period_ts"] <= start+60) if x is not None), default=None)
            alt = tip_only(e, bars, start)
            if alt:
                row.update(alt_price=alt[0], alt_spread=alt[1], alt_yes=alt[2], alt_payout=m["_outcome"] if alt[2] else 1-m["_outcome"],
                           alt_plays=plays(bars, alt[2], start, 0, e[3]))
            rows.append(row)
            cover[season, "read"] += 1
    # Second pass: last season's start moved by the difference between the two seasons' median onset of play (see `onset`).
    med = {season: [r["onset"] for r in rows if r["season"] == season and r["onset"] is not None and 60 <= r.get("alt_price", 0) < 97]
           for season in ("2025-26", "2026-27")}
    shift = round(float(np.median(med["2025-26"])-np.median(med["2026-27"]))) if len(med["2026-27"]) >= 50 and med["2025-26"] else 0
    events = {e[2]: e for archived in (True, False) for e in games(archived)}
    for row in rows:
        e = events[row["event"]]
        start = row["est_start"]+(60*shift if e[3] else 0)
        bars = bars_of(e, batches)
        cal = tip_only(e, bars, start)
        row["shift"] = shift
        if cal and not row.get("skip"):
            row.update(cal_start=start, cal_price=cal[0], cal_spread=cal[1], cal_yes=cal[2], cal_payout=row["outcome_first"] if cal[2] else 1-row["outcome_first"],
                       cal_plays=plays(bars, cal[2], start, 0, e[3]))
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT/"rows.json").write_text(json.dumps(rows))
    (OUT/"coverage.json").write_text(json.dumps({f"{k[0]} {k[1]}": v for k, v in cover.items()}, indent=1))
    print({f"{k[0]} {k[1]}": v for k, v in cover.items()})



# ---------------------------------------------------------------- views and cells

def load():
    return json.loads((OUT/"rows.json").read_text())


def month(r):
    return datetime.fromtimestamp(r["est_start"], fc.EASTERN).strftime("%Y-%m")


def view(rows, kind, fractional=False):
    """Priced games under one definition of the pre-game price: 'tip' (favourite_contexts' "at tip"), '3h' (its first 15
    bars, which must end before the start) or 'tip-only' (last quotes before play, no three-hour requirement)."""
    out = []
    for r in rows:
        if not fractional and not r["binary"]:
            continue
        if kind in ("tip-only", "tip-cal"):
            k = "alt" if kind == "tip-only" else "cal"
            if k+"_price" not in r:
                continue
            v = dict(price=r[k+"_price"], spread=r[k+"_spread"], payout=r[k+"_payout"], plays=r[k+"_plays"], yes=r[k+"_yes"],
                     est_start=r["est_start"] if k == "alt" else r["cal_start"])
        else:
            if r.get("skip") or (kind == "tip" and "tip_price" not in r) or (kind == "3h" and r["warm_end"] > r["est_start"]):
                continue
            v = dict(price=r["tip_price"] if kind == "tip" else r["start"], spread=r["tip_spread"] if kind == "tip" else r["spread"],
                     payout=r["payout"], plays=r["plays"], yes=r["yes"], est_start=r["est_start"])
        if v["price"] >= 99:
            continue
        v.update(event=r["event"], ticker=r["ticker"], settle=r["settle"], season=r["season"], month=month(r),
                 archived=r["archived"], binary=r["binary"], late=r["late"], before=r.get("before"), total_volume=r["total_volume"],
                 fav_code=r["codes"][0 if v["yes"] else 1])
        out.append(v)
    out.sort(key=lambda v: (v["settle"], v["event"]))
    return out


def band(vs, lo, hi):
    return [v for v in vs if lo <= v["price"] < hi]


def limit_of(v, level):
    """The bid in cents, or None when it would sit less than 5c below the price."""
    limit = math.floor(v["price"])-10 if level == "price-10" else int(level)
    return limit if limit <= v["price"]-5 else None


def fill_of(v, limit, delay=0):
    """First in-play bar with a print strictly below the bid (from `delay` minutes after the estimated start; negative
    values start before it, for the case that the estimate is late):
    dict(minute, bar volume, volume of the following five minutes in bars with a print below / wholly below the bid, mid after)."""
    p = [x for x in v["plays"] if x[0] > v["est_start"]+(60*delay if delay else 60)]
    for i, x in enumerate(p):
        if x[1] < limit:
            span = [y for y in p[i:] if y[0] < x[0]+300]
            later = next((y[4] for y in p[i+1:] if y[4] is not None), None)
            return dict(ts=x[0], minute=(x[0]-v["est_start"])/60, bar_volume=x[3], below5=sum(y[3] for y in span if y[1] < limit),
                        wholly5=sum(y[3] for y in span if y[2] < limit), low=x[1], mid_close=x[4], mid_next=later)
    return None


def touched(v, limit, delay=0):
    return any(x[1] <= limit for x in v["plays"] if x[0] > v["est_start"]+(60*delay if delay else 60))


def boot(values, draws=4000):
    if len(values) < 5:
        return None
    a = np.asarray(values, float)
    return [float(x) for x in np.percentile(a[RNG.integers(0, len(a), (draws, len(a)))].mean(1), [2.5, 97.5])]


def cell(vs, level, delay=0):
    orders = [(v, limit_of(v, level)) for v in vs]
    orders = [(v, l) for v, l in orders if l is not None]
    fills = [(v, l, f) for v, l in orders for f in [fill_of(v, l, delay)] if f]
    out = dict(orders=len(orders), fills=len(fills), touch=sum(touched(v, l, delay) for v, l in orders))
    if not fills:
        return out
    nets = [100*v["payout"]-l for v, l, _ in fills]
    limit = float(np.mean([l for _, l, _ in fills]))
    wins = sum(v["payout"] for v, _, _ in fills)
    w = wilson(wins, len(fills))
    out.update(rate=len(fills)/len(orders), win=100*wins/len(fills), needed=limit, net=float(np.mean(nets)), per_order=sum(nets)/len(orders),
               ret=100*float(np.mean(nets))/limit, boot=boot(nets), wilson=[100*w[0]-limit, 100*w[1]-limit], losses=sum(v["payout"] < .5 for v, _, _ in fills))
    return out


def fmt(c):
    if not c["fills"]:
        return f'orders {c["orders"]:4} fills    0 touch {c["touch"]:3}'
    b = "[   n<5    ]" if c["boot"] is None else f'[{c["boot"][0]:+5.1f},{c["boot"][1]:+5.1f}]'
    return (f'orders {c["orders"]:4} fills {c["fills"]:3} touch {c["touch"]:3} rate {c["rate"]:4.0%} win {c["win"]:5.1f}% need {c["needed"]:4.1f}% '
            f'net/fill {c["net"]:+6.1f}c ({c["ret"]:+5.1f}%) boot {b} wilson [{c["wilson"][0]:+5.1f},{c["wilson"][1]:+5.1f}] /order {c["per_order"]:+5.2f}c losses {c["losses"]}')


def halves(vs):
    return vs[:len(vs)//2], vs[len(vs)//2:]


def gap(vs):
    """Pre-game gap with favourite_contexts' interval (the largest of the sample, price-implied and team-clustered errors)."""
    if len(vs) < 2:
        return None
    t = fc.Table([dict(start=v["price"], payout=v["payout"], spread=v["spread"], settle=v["settle"], fav_code=v["fav_code"]) for v in vs], 0)
    return t.stats(np.ones(len(vs), bool))


def gap_text(g):
    return "n    0" if not g else f'n {g["n"]:4} price {g["price"]:4.1f}c win {g["win"]:5.1f}% gap {g["gap"]:+5.1f}c [{g["lo"]:+5.1f},{g["hi"]:+5.1f}]'


def iso(t):
    return datetime.fromtimestamp(t, timezone.utc).isoformat()[:10]


# ---------------------------------------------------------------- trade record

def fills_at_80(rows, kind="tip"):
    out = []
    for v in band(view(rows, kind), 90, 99):
        f = fill_of(v, 80)
        if f:
            out.append((v, f))
    return out


def tape_requests(v, f, cutoff):
    begin = f["ts"]-60
    return ("/historical/trades" if begin < cutoff else "/markets/trades"), dict(ticker=v["ticker"], min_ts=begin, max_ts=begin+360, limit=1000)


def cutoff_of(client=None):
    path = "/historical/cutoff"
    if cache_file(path, {}).exists():
        data = json.loads(cache_file(path, {}).read_text())
    elif client is not None:
        data = read(client, path, {})
    else:
        return None
    return timestamp(data["trades_created_ts"])


def trades(minutes):
    deadline, rows = time.time()+60*minutes, load()
    with httpx.Client(timeout=40) as client:
        cutoff = cutoff_of(client)
        todo = [(v, f) for v, f in fills_at_80(rows) if not cache_file(*tape_requests(v, f, cutoff)).exists()]
        print(f"80c fills on 90c+ favourites at tip: {len(fills_at_80(rows))}, trade records still to read {len(todo)}", flush=True)
        for v, f in todo:
            if time.time() > deadline:
                break
            try:
                read(client, *tape_requests(v, f, cutoff))
            except RuntimeError as exc:
                print("   read failed", v["event"], exc, flush=True)
    print(f"done: {dict(used)}")


def tapes(rows):
    """dip_new_families.measure on every cached trade record of an 80c fill."""
    cutoff, out = cutoff_of(), []
    if cutoff is None:
        return out
    for v, f in fills_at_80(rows):
        file = cache_file(*tape_requests(v, f, cutoff))
        if not file.exists():
            continue
        data = json.loads(file.read_text())
        got = measure(dict(levels={"80": dict(limit=80, ts=f["ts"])}, yes=v["yes"]), data.get("trades") or [])
        out.append(dict(got, event=v["event"], season=v["season"], truncated=bool(data.get("cursor")) and len(data.get("trades") or []) >= 1000,
                        bar_volume=f["bar_volume"], payout=v["payout"]))
    return out


# ---------------------------------------------------------------- report

def pass_test(vs, level):
    """First-half interval above zero (bootstrap and Wilson shown) and second half positive, halves by settlement over the orders."""
    a, b = halves(vs)
    ca, cb = cell(a, level), cell(b, level)
    ok_w = bool(ca["fills"] >= 5 and ca["wilson"][0] > 0 and cb.get("net", -1) > 0)
    ok_b = bool(ca["fills"] >= 5 and ca["boot"][0] > 0 and cb.get("net", -1) > 0)
    return ca, cb, ok_b, ok_w


def drop_best_month(vs, level):
    by = defaultdict(list)
    for v in vs:
        l = limit_of(v, level)
        if l is not None and fill_of(v, l):
            by[v["month"]].append(100*v["payout"]-l)
    if len(by) < 2:
        return None
    best = max(by, key=lambda k: sum(by[k]))
    rest = [x for k, xs in by.items() if k != best for x in xs]
    return dict(dropped=best, dropped_fills=len(by[best]), dropped_net=float(np.mean(by[best])), fills=len(rest), net=float(np.mean(rest)), boot=boot(rest))


def report():
    rows, lines = load(), []

    def say(text=""):
        print(text)
        lines.append(text)
    summary = {}
    # ---- coverage
    say("=== COVERAGE (one market per game, the alphabetically first)")
    listed = Counter()
    for archived in (True, False):
        for e in games(archived):
            listed[datetime.fromtimestamp(fc.start_of(FAM, e), fc.EASTERN).strftime("%Y-%m"), archived] += 1
    got = Counter((month(r), r["archived"]) for r in rows)
    for (mo, archived), n in sorted(listed.items(), key=lambda kv: (not kv[0][1], kv[0][0])):
        say(f'  {"last season (archive)" if archived else "this season        "} {mo}: listed {n:5} read {got[mo, archived]:5} ({got[mo, archived]/n:4.0%})')
    for season in ("2025-26", "2026-27"):
        mine = [r for r in rows if r["season"] == season]
        say(f'  {season}: games read {len(mine)}, first start {iso(min(r["est_start"] for r in mine))}, last {iso(max(r["est_start"] for r in mine))}')
    # ---- tradeability
    say("\n=== TRADEABLE AT ALL? (share of games read)")
    for season in ("2025-26", "2026-27"):
        mine = [r for r in rows if r["season"] == season]
        n = len(mine)
        skips = Counter(r["skip"] for r in mine if r.get("skip"))
        tip = [r for r in mine if not r.get("skip") and "tip_price" in r]
        alt = [r for r in mine if "alt_price" in r]
        early = [r for r in mine if not r.get("skip") and r["warm_end"] <= r["est_start"]]
        say(f"  {season}: games {n}; no trade or quote bar before the estimated start {sum(r['pre_bars'] == 0 for r in mine)} ({sum(r['pre_bars'] == 0 for r in mine)/n:.0%}); "
            f"fewer than 10 two-sided bars before the start {sum(r['pre_quoted'] < 10 for r in mine)} ({sum(r['pre_quoted'] < 10 for r in mine)/n:.0%})")
        say(f"     clean quote three hours out (favourite_contexts row): {n-sum(skips.values())} ({(n-sum(skips.values()))/n:.0%}); of those ending before the start {len(early)} ({len(early)/n:.0%}); "
            f"no row because: {dict(skips)}")
        say(f"     clean quote at the tip, favourite_contexts' definition (needs the three-hour row too): {len(tip)} ({len(tip)/n:.0%}); "
            f"clean quote at the tip alone (tip-only): {len(alt)} ({len(alt)/n:.0%})")
        say(f"     median contracts traded per game (whole life of the market): {np.median([r['total_volume'] for r in mine]):.0f}; "
            f"with a tip price {np.median([r['total_volume'] for r in tip]) if tip else 0:.0f}; without {np.median([r['total_volume'] for r in mine if 'alt_price' not in r] or [0]):.0f}")
        for name, lo, hi in (("90-99c", 90, 99), ("80-90c", 80, 90), ("99c+", 99, 101)):
            say(f"     favourites {name}: at tip {sum(lo <= r['tip_price'] < hi for r in tip)}, tip-only {sum(lo <= r['alt_price'] < hi for r in alt)}, "
                f"three hours out {sum(lo <= r['start'] < hi for r in early)}")
        summary[f"tradeable {season}"] = dict(games=n, tip=len(tip), tip_only=len(alt), three_hour=len(early))
    # ---- start-time check on this season's games, where the schedule is known
    now = [r for r in rows if r["season"] == "2026-27" and r.get("close") and r.get("expected")]
    if now:
        d = np.array([(r["close"]-(r["expected"]-3*3600))/60 for r in now])
        say(f"\n=== START ESTIMATE: this season the close falls a median {np.median(d):.0f} minutes after the scheduled start (10th-90th percentile "
            f"{np.percentile(d, 10):.0f}-{np.percentile(d, 90):.0f}; n {len(d)}). Last season's start is the close minus 130 minutes, so it is early by about "
            f"{130-np.median(d):.0f} minutes at the median; |error| over 20 minutes in {np.mean(np.abs(d-130) > 20):.0%} of games, over 25 minutes late in {np.mean(d-130 > 25):.0%}.")
    for season in ("2025-26", "2026-27"):
        on = np.array([r["onset"] for r in rows if r["season"] == season and r["onset"] is not None and "alt_price" in r and 60 <= r["alt_price"] < 97])
        if len(on):
            say(f"  {season}: first visible price movement (midpoint travels 4c within ten minutes), minutes after the {'estimated' if season == '2025-26' else 'scheduled'} start, "
                f"games with a 60-97c favourite at the tip: n {len(on)}, 10th pct {np.percentile(on, 10):+.0f}, 25th {np.percentile(on, 25):+.0f}, median {np.median(on):+.0f}, "
                f"75th {np.percentile(on, 75):+.0f}; before the start in {np.mean(on < 0):.0%}, more than 25 minutes before in {np.mean(on < -25):.0%}")
    # ---- gap and dip tables
    for kind, title in (("tip", "PRICE AT THE TIP (favourite_contexts' definition; the live rule's)"), ("3h", "PRICE ABOUT THREE HOURS BEFORE THE TIP"),
                        ("tip-only", "PRICE AT THE TIP WITHOUT THE THREE-HOUR QUOTE REQUIREMENT (sensitivity)"),
                        ("tip-cal", f"PRICE AT THE TIP, LAST SEASON'S START MOVED BY {rows[0]['shift']:+d} MINUTES TO MATCH THE ONSET OF PLAY (sensitivity)")):
        vs = view(rows, kind)
        say(f"\n{'='*30} {title}: {len(vs)} priced games")
        for name, lo, hi in (("90-99c", 90, 99), ("80-90c (control)", 80, 90)):
            sub = band(vs, lo, hi)
            if not sub:
                continue
            parts = [("all", sub), ("last season 2025-26", [v for v in sub if v["season"] == "2025-26"]), ("November 2026", [v for v in sub if v["season"] == "2026-27"])]
            h1, h2 = halves(sub)
            parts += [(f"first half (to {iso(h1[-1]['settle'])})", h1), (f"second half (from {iso(h2[0]['settle'])})", h2)]
            last = parts[1][1]
            if len(last) >= 20:
                a, b = halves(last)
                parts += [(f"last season, first half (to {iso(a[-1]['settle'])})", a), (f"last season, second half", b)]
            months = sorted({v["month"] for v in sub})
            say(f"\n  -- favourites {name}: pre-game gap (win rate minus price)")
            for label, part in parts+[(mo, [v for v in sub if v["month"] == mo]) for mo in months]:
                say(f"     {label:42} {gap_text(gap(part))}")
            summary[f"{kind} {name} gap"] = gap(sub)
            for level in LEVELS:
                if not any(limit_of(v, level) is not None for v in sub):
                    continue
                say(f"  -- favourites {name}, bid {level if level == 'price-10' else level+'c'}")
                for label, part in parts+[(mo, [v for v in sub if v["month"] == mo]) for mo in months]:
                    say(f"     {label:42} {fmt(cell(part, level))}")
                ca, cb, ok_b, ok_w = pass_test(sub, level)
                d = drop_best_month(sub, level)
                say(f"     PASS TEST (first-half interval above zero, second half positive): bootstrap {'PASS' if ok_b else 'FAIL'}, Wilson {'PASS' if ok_w else 'FAIL'}"
                    + (f" | best month {d['dropped']} dropped ({d['dropped_fills']} fills at {d['dropped_net']:+.1f}c): {d['fills']} fills, net {d['net']:+.1f}c "
                       f"{'[%+.1f,%+.1f]' % tuple(d['boot']) if d['boot'] else ''}" if d else ""))
                if last and level == "80" and lo == 90:
                    la, lb, okb, okw = pass_test(last, level)
                    say(f"     last season alone: bootstrap {'PASS' if okb else 'FAIL'}, Wilson {'PASS' if okw else 'FAIL'}")
                summary[f"{kind} {name} bid {level}"] = dict(all=cell(sub, level), last_season=cell(parts[1][1], level), this_season=cell(parts[2][1], level),
                                                              first_half=ca, second_half=cb, pass_boot=ok_b, pass_wilson=ok_w, drop_best_month=d,
                                                              months={mo: cell([v for v in sub if v["month"] == mo], level) for mo in months})
    # ---- sanity checks on the main cell
    vs = view(rows, "tip")
    main = band(vs, 90, 99)
    fills = [(v, f) for v in main for f in [fill_of(v, 80)] if f]
    say(f"\n{'='*30} SANITY CHECKS: 80c bid, favourites 90-99c at the tip ({len(main)} orders, {len(fills)} fills)")
    for season in ("2025-26", "2026-27"):
        mine = [(v, f) for v, f in fills if v["season"] == season]
        if not mine:
            continue
        mins = np.array([f["minute"] for _, f in mine])
        to_settle = np.array([(v["settle"]-f["ts"])/60 for v, f in mine])
        say(f"  {season}: fill minutes after the (estimated) start: min {mins.min():.0f}, 10th pct {np.percentile(mins, 10):.0f}, median {np.median(mins):.0f}, 90th pct {np.percentile(mins, 90):.0f}, "
            f"max {mins.max():.0f}; in the first 10 minutes {int((mins <= 10).sum())}, first 20 {int((mins <= 20).sum())}; later than 150 minutes {int((mins > 150).sum())}; "
            f"minutes before settlement: median {np.median(to_settle):.0f}")
        vol = np.array([f["bar_volume"] for _, f in mine])
        b5, w5 = np.array([f["below5"] for _, f in mine]), np.array([f["wholly5"] for _, f in mine])
        say(f"  {season}: contracts traded in the fill minute (all prices): median {np.median(vol):.0f}, 25th pct {np.percentile(vol, 25):.0f}, 10th pct {np.percentile(vol, 10):.0f}; "
            f"under 3 contracts in {np.mean(vol < 3):.0%}, under 10 in {np.mean(vol < 10):.0%}, under 50 in {np.mean(vol < 50):.0%}")
        say(f"  {season}: next five minutes: contracts in bars with a print under 80c: median {np.median(b5):.0f}; in bars wholly under 80c (a lower bound on what a resting "
            f"80c bid could buy): median {np.median(w5):.0f}, 25th pct {np.percentile(w5, 25):.0f}, zero in {np.mean(w5 == 0):.0%}, under 3 in {np.mean(w5 < 3):.0%}")
        after = [f["mid_close"]-80 for _, f in mine if f["mid_close"] is not None]
        nxt = [f["mid_next"]-80 for _, f in mine if f["mid_next"] is not None]
        say(f"  {season}: favourite's midpoint minus the bid at the end of the fill minute {np.mean(after):+.1f}c (n {len(after)}), at the next traded minute {np.mean(nxt):+.1f}c; "
            f"low of the fill minute: median {np.median([f['low'] for _, f in mine]):.0f}c")
    for season in ("2025-26", "2026-27"):
        mine = [v for v in main if v["season"] == season]
        for delay in (-20, -10, 10, 20, 30):
            if delay > 0 or season == "2025-26":
                say(f"  {season}: fills counted from {delay:+d} minutes relative to the {'estimated' if season == '2025-26' else 'scheduled'} start: {fmt(cell(mine, '80', delay))}")
    for season in ("2025-26", "2026-27"):
        mine = [v for v in main if v["season"] == season]
        early = [v for v in mine if any(x[1] < 80 for x in v["plays"] if x[0] <= v["est_start"]+60)]
        say(f"  {season}: a print under 80c between the tip quote and the {'estimated' if season == '2025-26' else 'scheduled'} start (not counted as a fill): {len(early)} of {len(mine)} orders; "
            f"{sum(fill_of(v, 80) is not None for v in early)} of them also filled after the start; the {len(early)} won {100*np.mean([v['payout'] for v in early]) if early else 0:.0f}%")
    frac = [v for v in band(view(rows, 'tip', fractional=True), 90, 99) if not v["binary"]]
    allrows = [r for r in rows if not r["binary"]]
    say(f"  fractional settlements: {len(allrows)} games read settled away from 0 or 1 ({sorted(Counter(round(r['outcome_first'], 2) for r in allrows).items())}); "
        f"{len(frac)} of them were 90-99c favourites at the tip, {sum(fill_of(v, 80) is not None for v in frac)} filled at 80c"
        + ("; with them at their fractional value: "+fmt(cell(band(view(rows, 'tip', fractional=True), 90, 99), '80')) if frac else ""))
    lates = [r for r in rows if r["late"]]
    say(f"  late settlements (more than an hour after the close): {len(lates)} games read; {sum(v['late'] for v in main)} among the 90-99c favourites, "
        f"{sum(v['late'] for v, _ in fills)} among the fills. They are included, on a window placed on the close.")
    gaps = np.array([r["settle"]-r["close"] for r in rows if r.get("close")])
    say(f"  settlement minus close: median {np.median(gaps)/60:.0f} minutes, 95th pct {np.percentile(gaps, 95)/60:.0f} minutes")
    tp = tapes(rows)
    if tp:
        found = [t for t in tp if t["found"]]
        say(f"  trade record read for {len(tp)} of {len(fills)} fills; a print under 80c found in {len(found)} ({len(found)/len(tp):.0%})")
        for season in ("2025-26", "2026-27"):
            mine = [t for t in found if t["season"] == season]
            if not mine:
                continue
            c = np.array([t["contracts_below"] for t in mine])
            s_ = np.array([t["sold_by_taker"] for t in mine])
            say(f"  {season} ({len(mine)} fills): contracts printed strictly under 80c within five minutes of the first such print: median {np.median(c):.0f}, 25th pct {np.percentile(c, 25):.0f}, "
                f"10th pct {np.percentile(c, 10):.0f}; fewer than 3 in {np.mean(c < 3):.0%}, fewer than 10 in {np.mean(c < 10):.0%}, fewer than 50 in {np.mean(c < 50):.0%}; "
                f"of them sold by the taker (what a resting bid buys): median {np.median(s_):.0f}, fewer than 3 in {np.mean(s_ < 3):.0%}, fewer than 10 in {np.mean(s_ < 10):.0%}")
            for name, mask in (("at least 3 taker-sold contracts under 80c", s_ >= 3), ("at least 10", s_ >= 10), ("fewer than 3", s_ < 3)):
                pay = np.array([t["payout"] for t in mine])[mask]
                if len(pay):
                    say(f"       fills with {name}: {len(pay)}, won {100*pay.mean():.1f}%, net {100*pay.mean()-80:+.1f}c")
        summary["tape"] = dict(read=len(tp), found=len(found))
    # ---- combined verdict numbers
    say(f"\n{'='*30} VERDICT NUMBERS: 80c bid, 90-99c at the tip, last season + November 2026")
    c = cell(main, "80")
    ca, cb, ok_b, ok_w = pass_test(main, "80")
    say(f"  all          {fmt(c)}")
    say(f"  first half   {fmt(ca)}")
    say(f"  second half  {fmt(cb)}")
    say(f"  pass: bootstrap {'PASS' if ok_b else 'FAIL'}, Wilson {'PASS' if ok_w else 'FAIL'}")
    wide = band(vs, 80, 99)      # favourite_contexts' own test: halves cut at the median settlement of the 80-99c rows, normal interval
    cut = float(np.median([v["settle"] for v in wide]))
    t = fc.Table([dict(start=v["price"], payout=v["payout"], spread=v["spread"], settle=v["settle"], fav_code=v["fav_code"], full=True, limit=80,
                       fill=fill_of(v, 80) is not None) for v in main], cut)
    whole, ok = t.dip_test(np.ones(len(main), bool))
    parts = [t.dip_test(m)[0] for m in (t.first, ~t.first)]
    say(f"  favourite_contexts' test (the one the men's rule passed): net {whole[0]:+.1f}c se {whole[1]:.1f} on {whole[2]} fills; halves cut at {iso(cut)}: "
        + " / ".join(f"{x[2]} fills {x[0]:+.1f}c [{x[0]-1.96*x[1]:+.1f},{x[0]+1.96*x[1]:+.1f}]" if x else "too few" for x in parts) + f" -> {'PASS' if ok else 'FAIL'}")
    for season, days in (("2025-26", None), ("2026-27", None)):
        mine = [v for v in main if v["season"] == season]
        if not mine:
            continue
        span = max(14, (max(v["settle"] for v in mine)-min(v["settle"] for v in mine))/86400)
        cs = cell(mine, "80")
        share = sum(r["season"] == season for r in rows)/len(games(season == "2025-26"))
        say(f"  {season}: {len(mine)} orders and {cs['fills']} fills over {span:.0f} days on {share:.0%} of the games = {len(mine)/span*30/share:.0f} orders and "
            f"{cs['fills']/span*30/share:.0f} fills per 30 days on every game")
    from research.dip_new_families import shrink
    men = json.loads((fc.OUT/"summary.json").read_text())["families"].get("cbb-men at tip 90-99") if (fc.OUT/"summary.json").exists() else None
    cells = {"women's college basketball (this study)": (round(c["win"]*c["fills"]/100), c["fills"], 80)}
    if men:
        cells["men's college basketball (favourite_contexts)"] = (round(men["dip_win"]*men["dip_fills"]/100), men["dip_fills"], 80)
        other = json.loads((fc.OUT/"summary.json").read_text())["families"]
        for k in ("soccer 90-99", "esports 90-99", "basketball-other 90-99"):
            if k in other and other[k].get("dip_fills"):
                cells[k] = (round(other[k]["dip_win"]*other[k]["dip_fills"]/100), other[k]["dip_fills"], 80)
    sh = shrink(cells)
    for k, v in sh.items():
        say(f"  shrunk with the other families (dip_new_families.shrink): {k:48} fills {v['fills']:4} raw {v['raw']:+5.1f}c shrunk {v['shrunk']:+5.1f}c ({100*v['shrunk']/80:+.1f}%)")
    summary["verdict"] = dict(all=c, first_half=ca, second_half=cb, pass_boot=ok_b, pass_wilson=ok_w, shrunk=sh)
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT/"report.txt").write_text("\n".join(lines)+"\n")
    (OUT/"summary.json").write_text(json.dumps(summary, indent=1, default=float))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("cmd", choices=("download", "recent", "build", "trades", "report"))
    parser.add_argument("--minutes", type=float, default=9)
    parser.add_argument("--pace", type=float, default=1.2, help="seconds between requests (raised automatically after HTTP 429)")
    args = parser.parse_args()
    pace[0] = args.pace
    if args.cmd == "download":
        download(args.minutes)
    elif args.cmd == "recent":
        recent(args.minutes)
    elif args.cmd == "build":
        build()
    elif args.cmd == "trades":
        trades(args.minutes)
    else:
        report()
