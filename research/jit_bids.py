"""Just-in-time dip bids: place the 80c bid only once the favourite has fallen to a trigger, instead of resting it from the start.

The live rule (paper/trader.py): a favourite quoted 90c+ before play gets a
resting 80c bid from the first live pass, held to settlement; real money on
first-round tennis qualifying and men's college basketball. About one bid in
four fills, yet every bid reserves 80c for the whole event. This script asks
whether the same trades can be had on less capital by placing the bid only
when the price reaches a trigger (90, 88, 86, 85, 84, 82, 81c), what that costs in
missed fills and queue position, and what it does to the return on capital.

Orders. The rows of data/processed/favourite-basket/table.json with a price of
90c to under 99c at the pricing moment (tennis: last bar before the detected
start; college basketball: the last 10-15 bars before the tip), i.e. the same
orders as the basket study's dip table. Groups: "FRQ" (tennis, first-round
qualifying), "CBB" (men's college basketball), "LIVE" (both), and every tennis
favourite by tour. Nothing is selected on anything known after the start.

Steps

    build     replay the cached one-minute bars of every order (no network)
              into one row: when the favourite's lowest print / lowest ask of a
              minute first reached each trigger in play, the first minute with
              a print below 80c (the strict fill of research/favorite_dip.py),
              at 80c (touch), and the minute-by-minute path used by the
              capital simulation.
    download  the public trade record (GET /markets/trades, /historical/trades
              for events settled before 2026-09-18) around every strict fill
              of FRQ and CBB (11 minutes before the fill bar ends to 31 minutes
              after, the window of research/dip_flow.py, so tennis fills it
              already read are reused), then around touch-only events (a print
              at 80c, never below), then other tennis fills. One request at a
              time, PACE seconds apart, 30 seconds off on HTTP 429, a time
              budget (`--minutes`). Cached in the shared cache, so a rerun
              resumes.
    analyse   the tables (speed, capital, queue, alternatives, combined).

Definitions

* Prices are the favourite's, in cents. t0 = the first print strictly below
  80c in play (from the trade record where read, else the fill bar).
* Trigger: the favourite's last trade OR its ask at or below the trigger level.
  On the bars: the first in-play minute whose lowest print or lowest ask is at
  or below it (the time is taken as the end of that minute, the latest it can
  have been; the capital tables place the bid a minute earlier, the earliest).
  On the trade record: the first trade at or below the trigger in the window,
  or the end of such a minute if that is earlier. When that minute closed
  before the trade window opens, the lead is taken from the bars alone (a
  lower bound). Sub-minute quotes are not on record, so leads are lower bounds
  for a trader who also watches the ask.
  "Final descent" = the same, counted from the last trade ABOVE the trigger
  before t0 (what a trader who cancels on a recovery would face).
* In time: a trader polling every P seconds sees the trigger U(0,P) seconds
  after it happens and places REACT seconds later; share = P(delay < lead).
* Late bid: placed after t0. It still fills if, after it is placed, any trade
  prints at 79c or less (seller-initiated: it hits our bid first; buyer-
  initiated: an offer stood at 79c or less, which a marketable 80c bid takes),
  or a seller-initiated trade prints at 80c. The outcome of the event is the
  same, so a late fill earns the same as an early one; only dips that are back
  above 80c before the bid arrives are lost.
* Queue. A print strictly below 80c means every bid at 80c on that book was
  filled, so strict fills do not depend on the queue. The queue decides only
  "touch-only" events (prints at 80c, none below): a bid with Q contracts
  ahead fills when seller-initiated volume at exactly 80c after its placement
  exceeds Q (cancellations ahead of us are ignored, which is conservative).
  "Depth at 80c" = seller-initiated volume at exactly 80c before t0 in strict
  fills: what was resting there, an upper bound on the queue a late bid joins.
* Capital. Each order reserves its limit price from placement to settlement
  (the trader's own exposure rule; nothing is cancelled). "Cancel on recovery"
  cancels when the favourite's bid closes a minute at trigger+5c or more and
  places again at the next trigger. Bankroll simulation: K slots, each 1/K of
  the bankroll, stake = slot / limit contracts. An order that finds no free
  slot is skipped for good ("always") or waits for one while the price is
  still at or below the trigger ("just in time"). Return on bankroll per day =
  sum of slot returns / K over calendar days (US Eastern date of the start
  minus 5 hours); a month is 30 such days. Maker fee taken as zero (ATP and
  WTA main-tour series charge one; FRQ on those tours is a small share).
* Fill conventions. "Bar": every order with an in-play print strictly below
  the bid is one fill (the historical figures). "Trade record": a bid of 3
  contracts fills when seller-initiated volume below the bid, plus the part of
  the volume at the bid beyond the queue ahead, reaches 3 contracts inside the
  window; this drops prints too small to fill us and adds touch-only events
  the queue lets through. Sections 5A and 5B give the bid resting from the
  start nothing ahead of it and the just-in-time bid 1,000 contracts ahead: the
  comparison least favourable to just in time. Section 3 shows every queue.
* Not covered: 14 of this November's college games have no per-market bar
  read in the cache and are left out; the live recorder (two days) holds only
  seven 90c+ favourites with a pre-start quote and no dip, so the 15-second
  quote path of a dip could not be checked against the trade record.
* Halves: orders sorted by settlement, split at the median, per group.
* Intervals: 95% bootstrap, resampling days for daily returns and orders or
  fills elsewhere.

    .venv/bin/python research/jit_bids.py build
    .venv/bin/python research/jit_bids.py download --minutes 35
    .venv/bin/python research/jit_bids.py analyse

Public market data only; no orders, no authentication. Outputs under
data/processed/jit-bids/.
"""
import argparse
from collections import Counter, defaultdict
from datetime import date, datetime, timedelta
import hashlib
import io
import json
from pathlib import Path
import pickle
import sys
import time
from zoneinfo import ZoneInfo

import httpx
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import config
from research.api_cost import COLLECTION, ROOT, midnight, save, timestamp
from research.near_end import WINDOW, cents

PROC = ROOT / "data/processed"
OUT = PROC / "jit-bids"
CUTOFF = midnight(date(2026, 9, 18))
EASTERN = ZoneInfo("America/New_York")
TRIGGERS = (90, 88, 86, 85, 84, 82, 81)
LIMIT, BEFORE, AFTER = 80, 660, 1860
PACE, REACT, POLL = 1.5, 1.0, 20.0
RNG = np.random.default_rng(41)
last_request = [0.0]
buffer = io.StringIO()


def say(text=""):
    print(text, flush=True)
    buffer.write(text+"\n")


def cache_file(path, params):
    return COLLECTION / (hashlib.sha256(json.dumps([path, params], sort_keys=True).encode()).hexdigest() + ".json")


# ---------------------------------------------------------------- build

def bar_file(series, ticker, settle):
    params = dict(start_ts=settle-WINDOW, end_ts=settle, period_interval=1)
    for path in (f"/historical/markets/{ticker}/candlesticks", f"/series/{series}/markets/{ticker}/candlesticks"):
        if cache_file(path, params).exists():
            return cache_file(path, params)
    return None


def side(bars, yes):
    """Per bar on the favourite's side: end time, closing bid, lowest ask of the minute, lowest print, volume.

    The ask (not the bid) carries the quote trigger: a bid vanishes when a book empties, an offer at a low price is real.
    """
    out = []
    for b in sorted(bars, key=lambda b: b["end_period_ts"]):
        if yes:
            bid, ask_low, low = cents(b, "yes_bid", "close"), cents(b, "yes_ask", "low"), cents(b, "price", "low")
        else:
            ask, bid_high, high = cents(b, "yes_ask", "close"), cents(b, "yes_bid", "high"), cents(b, "price", "high")
            bid, ask_low, low = (None if ask is None else 100-ask), (None if bid_high is None else 100-bid_high), (None if high is None else 100-high)
        if ask_low is not None and not 0 < ask_low < 100:
            ask_low = None
        out.append((b["end_period_ts"], bid, ask_low, low, float(b.get("volume_fp") or b.get("volume") or 0)))
    return out


def build():
    table = [r for r in json.loads((PROC/"favourite-basket/table.json").read_text())
             if 90 <= r["mid"] < 99 and (r["family"] == "tennis" or r["cell"] == "cbb-men")]
    tennis = {r["event"]: r for r in json.loads((PROC/"tennis-contexts/rows.json").read_text())}
    college = json.loads((PROC/"favourite-contexts/rows-cbb-men.json").read_text())
    rows, missing = [], Counter()
    for n, r in enumerate(table):
        if r["family"] == "tennis":
            x = tennis[r["event"]]
            ticker, yes, tour = x["ticker"], x["yes"], x["tour"]
        else:
            x = college[r["event"]]
            ticker, yes, tour = x["bar_ticker"], x["yes"], "CBB men"
        file = bar_file(r["series"], ticker, x["settle"])
        if file is None:
            missing[r["cell"]] += 1
            continue
        path = [p for p in side(json.loads(file.read_text()).get("candlesticks") or [], yes) if p[0] > r["start"]+60]
        row = dict(event=r["event"], series=r["series"], ticker=ticker, yes=yes, tour=tour, cell=r["cell"], start=r["start"],
                   settle=x["settle"], mid=r["mid"], payout=r["payout"], day=r["day"], archived=r["archived"],
                   low_table=r["low_in_play"], path=path)
        lows = [p[3] for p in path if p[3] is not None]
        row["low"] = min(lows) if lows else None
        first = lambda test: next((p[0] for p in path if test(p)), None)
        row["fill_ts"] = first(lambda p: p[3] is not None and p[3] < LIMIT)
        row["touch_ts"] = first(lambda p: p[3] is not None and p[3] <= LIMIT)
        row["print_ts"] = {t: first(lambda p: p[3] is not None and p[3] <= t) for t in TRIGGERS}
        row["quote_ts"] = {t: first(lambda p: (p[2] is not None and p[2] <= t) or (p[3] is not None and p[3] <= t)) for t in TRIGGERS}
        rows.append(row)
        if n % 500 == 0:
            print(f"  {n}/{len(table)}", flush=True)
    OUT.mkdir(parents=True, exist_ok=True)
    with (OUT/"orders.pkl").open("wb") as out:
        pickle.dump(rows, out)
    print(f"{len(rows)} orders built; without cached bars: {dict(missing)}")
    differ = sum(r["low"] != r["low_table"] for r in rows)
    print(f"in-play low differs from the basket table on {differ} orders; "
          f"fills here {sum(r['fill_ts'] is not None for r in rows)}, in the table {sum((r['low_table'] or 100) < LIMIT for r in rows)}")


def load():
    with (OUT/"orders.pkl").open("rb") as file:
        rows = pickle.load(file)
    for r in rows:
        r["group"] = "FRQ" if r["cell"] == "tennis: qualifying, first round" else "CBB" if r["cell"] == "cbb-men" else "other tennis"
        r["net"] = 100*r["payout"]-LIMIT
    for group in ("FRQ", "CBB"):
        part = sorted((r for r in rows if r["group"] == group), key=lambda r: r["settle"])
        for i, r in enumerate(part):
            r["half"] = 1 if i < len(part)/2 else 2
    live = sorted((r for r in rows if r["group"] != "other tennis"), key=lambda r: r["settle"])
    for i, r in enumerate(live):
        r["live_half"] = 1 if i < len(live)/2 else 2
    return rows


# ---------------------------------------------------------------- download

def read(client, path, params, download, deadline):
    """Cached read, else one paced network read. None when not cached and downloading is off or out of time."""
    file = cache_file(path, params)
    if file.exists():
        return json.loads(file.read_text())
    if not download or time.monotonic() > deadline:
        return None
    for attempt in range(6):
        wait = last_request[0]+PACE-time.monotonic()
        if wait > 0:
            time.sleep(wait)
        try:
            resp = client.get(config.KALSHI_PROD_BASE+path, params=params)
        except httpx.RequestError:
            resp = None
        last_request[0] = time.monotonic()
        if resp is not None and resp.status_code == 200:
            data = resp.json()
            temp = file.with_suffix(".tmp")
            temp.write_text(json.dumps(data))
            temp.replace(file)
            return data
        if resp is not None and resp.status_code != 429 and resp.status_code < 500:
            return dict(trades=[], error=resp.status_code)        # a definite answer such as 404: not cached, not retried
        time.sleep(30)                                            # HTTP 429 or a server error: the live trader shares the limit
        last_request[0] = time.monotonic()
    raise RuntimeError(f"Read failed after long back-off: {path}")


def trade_pages(client, ticker, settle, lo, hi, download, deadline):
    """All trades of one market between lo and hi. None if not cached and not downloaded."""
    paths = ["/historical/trades", "/markets/trades"] if settle < CUTOFF else ["/markets/trades", "/historical/trades"]
    for path in paths:
        trades, cursor = [], ""
        while True:
            params = dict(ticker=ticker, min_ts=lo, max_ts=hi, limit=1000)
            if cursor:
                params["cursor"] = cursor
            data = read(client, path, params, download, deadline)
            if data is None:
                return None
            trades += data.get("trades") or []
            cursor = data.get("cursor")
            if not cursor or not data.get("trades"):
                break
        if trades:
            return trades
    return []


def windows(r):
    """(lo, hi) of the trade window for an order: around the fill bar, or around its prints at 80c when it never traded below."""
    if r["fill_ts"] is not None:
        return r["fill_ts"]-BEFORE, r["fill_ts"]+AFTER
    if r["touch_ts"] is not None:
        last = max(p[0] for p in r["path"] if p[3] is not None and p[3] <= LIMIT)
        return r["touch_ts"]-BEFORE, min(last+120, r["touch_ts"]+3600)
    return None


def parse(r, raw):
    """(time, favourite's price, contracts, seller-initiated?) sorted by time."""
    out = []
    for t in raw:
        price = float(t.get("yes_price_dollars", t.get("yes_price")))
        price = price*100 if "yes_price_dollars" in t else price
        taker = t.get("taker_side") or t.get("taker_outcome_side")
        out.append((timestamp(t["created_time"]), round(price) if r["yes"] else 100-round(price),
                    float(t.get("count_fp", t.get("count"))), (taker == "no") if r["yes"] else (taker == "yes")))
    return sorted(out)


def queue(rows):
    """Download order: strict fills of the live groups, their touch-only events, then other tennis fills (newest first)."""
    rank = lambda r: (0 if r["group"] != "other tennis" and r["fill_ts"] else 1 if r["group"] != "other tennis" else 2, -r["settle"])
    return sorted((r for r in rows if windows(r)), key=rank)


def download(rows, minutes):
    deadline, got, began = time.monotonic()+60*minutes, Counter(), time.monotonic()
    with httpx.Client(timeout=40, headers={"User-Agent": "research-script"}) as client:
        for n, r in enumerate(queue(rows)):
            lo, hi = windows(r)
            raw = trade_pages(client, r["ticker"], r["settle"], lo, hi, True, deadline)
            got[(r["group"], "fill" if r["fill_ts"] else "touch", "read" if raw is not None else "not read")] += 1
            if n % 40 == 0:
                print(f"  {n} {(time.monotonic()-began)/60:.1f} min {dict(got)}", flush=True)
    print("download:", {" | ".join(k): v for k, v in sorted(got.items())})


# ---------------------------------------------------------------- trade-record measures

def lead(r, trades, lo, target, trigger, recover=None):
    """(seconds from the trigger to `target`, "trades" or "bars", trigger time or None when it came before the window).

    recover=None: the first trade at or below the trigger in play. recover=c: the first such trade after the last trade at
    trigger+c or more before the target (the final descent, for a bid cancelled on a recovery to trigger+c).
    """
    before = [t for t in trades if r["start"] < t[0] <= target]
    if recover is None:
        bar = r["quote_ts"][trigger]
        if bar is not None and bar <= lo:
            return target-bar, "bars", None
        when = next((t[0] for t in before if t[1] <= trigger), target)
        quote = r["quote_ts"][trigger]              # the offer reached the trigger in a minute that closed before the target
        if quote is not None and quote < when:
            return target-quote, "quote bar", quote
        return target-when, "trades", when
    above = [i for i, t in enumerate(before) if t[1] >= trigger+recover]
    if not above:                                   # never back up to trigger+c inside the window
        bar = r["quote_ts"][trigger]
        if bar is not None and bar <= lo:           # the bid has been resting since before the window opened
            return target-bar, "bars", None
        when = next((t[0] for t in before if t[1] <= trigger), target)
        return target-when, "trades", when
    when = next((t[0] for t in before[above[-1]+1:] if t[1] <= trigger), target)
    return target-when, "trades", when


def filled(trades, placed, level, ahead, size, sellers_only=True):
    """Does a bid of `size` at `level`, placed at `placed` with `ahead` contracts in front of it, fill inside the window?"""
    at = sum(t[2] for t in trades if t[0] > placed and t[1] == level and t[3])
    below = sum(t[2] for t in trades if t[0] > placed and t[1] < level and (t[3] or not sellers_only))
    return max(0.0, at-ahead)+below >= size


def chance(trades, lo, when, level, ahead, size, poll=POLL, sellers_only=True, steps=20):
    """Probability that a bid placed after a trigger at `when` (None: resting since before the window) fills."""
    if when is None:
        return float(filled(trades, lo-1, level, ahead, size, sellers_only))
    return float(np.mean([filled(trades, when+REACT+poll*(i+.5)/steps, level, ahead, size, sellers_only) for i in range(steps)]))


def measure(rows):
    """Trade-record measures for every order whose window is cached. Adds r["flow"]."""
    with httpx.Client(transport=httpx.MockTransport(lambda request: httpx.Response(404))) as client:
        for r in rows:
            r["flow"] = None
            if not windows(r):
                continue
            lo, hi = windows(r)
            raw = trade_pages(client, r["ticker"], r["settle"], lo, hi, False, 0)
            if not raw:
                continue
            trades = parse(r, raw)
            anchor = r["fill_ts"] or r["touch_ts"]
            first = lambda level: next((t[0] for t in trades if t[0] > max(anchor-60.5, r["start"]) and t[1] <= level), None)
            t0, t80 = (first(LIMIT-1) if r["fill_ts"] else None), first(LIMIT)
            if (r["fill_ts"] and t0 is None) or t80 is None:
                continue
            f = dict(trades=trades, lo=lo, t0=t0, t80=t80, lead={}, descent={}, lead81={})
            for trigger in TRIGGERS:
                if t0 is not None:
                    f["lead"][trigger] = lead(r, trades, lo, t0, trigger)
                    f["descent"][trigger] = lead(r, trades, lo, t0, trigger, 5)
                f["lead81"][trigger] = lead(r, trades, lo, t80, trigger)       # an 81c bid must be in before the first print at 80c
            end = t0 if t0 is not None else hi
            f["depth_at_80"] = sum(t[2] for t in trades if r["start"] < t[0] < end and t[1] == LIMIT and t[3])
            after = [t for t in trades if t0 is not None and t0 <= t[0] <= t0+1800]
            f["sold_below"] = sum(t[2] for t in after if t[1] < LIMIT and t[3])
            f["any_below"] = sum(t[2] for t in after if t[1] < LIMIT)
            f["sold_at_80"] = sum(t[2] for t in trades if t[0] > r["start"] and t[1] == LIMIT and t[3])
            back = next((t[0]-t0 for t in after if t[1] > LIMIT), None) if t0 is not None else None
            f["seconds_below"] = back
            r["flow"] = f


def in_time(seconds, poll):
    return float(np.clip((np.asarray(seconds, float)-REACT)/poll, 0, 1).mean())


def boot(values, weights=None, draws=2000):
    values = np.asarray(values, float)
    if len(values) < 8:
        return None
    pick = RNG.integers(0, len(values), (draws, len(values)))
    if weights is None:
        means = values[pick].mean(axis=1)
    else:
        weights = np.asarray(weights, float)
        means = (values[pick]*weights[pick]).sum(axis=1)/np.maximum(weights[pick].sum(axis=1), 1e-9)
    return [float(v) for v in np.percentile(means, [2.5, 97.5])]


def band(ci, digits=1, unit=""):
    return "[n/a]" if ci is None else f"[{ci[0]:+.{digits}f},{ci[1]:+.{digits}f}]{unit}"


SETS = (("FRQ (tennis first-round qualifying)", lambda r: r["group"] == "FRQ"), ("CBB (men's college basketball)", lambda r: r["group"] == "CBB"),
        ("LIVE (FRQ + CBB)", lambda r: r["group"] != "other tennis"), ("every tennis favourite at 90c+", lambda r: r["group"] != "CBB"),
        ("  Challenger", lambda r: r["tour"] == "Challenger"), ("  WTA", lambda r: r["tour"] == "WTA"),
        ("  WTA Challenger", lambda r: r["tour"] == "WTA Challenger"), ("  ATP", lambda r: r["tour"] == "ATP"),
        ("  ITF men", lambda r: r["tour"] == "ITF men"), ("  ITF women", lambda r: r["tour"] == "ITF women"))


def speed(rows, out):
    say("\n1. SPEED OF THE DIP: seconds from the first trade at or below the trigger to the first print below 80c (t0)")
    say("   trade timestamps; 'bars' = the trigger printed in a minute that closed before the trade window opened (11 minutes before the")
    say("   fill bar ends), so the lead is at least that long. 'in time' = share of fills where a trader polling every P seconds, placing")
    say(f"   {REACT:.0f}s after it sees the trigger, has the bid in before t0. 'descent' = the same from the last trade at trigger+5c or more.")
    for name, test in SETS:
        fills = [r for r in rows if test(r) and r["fill_ts"]]
        part = [r for r in fills if r["flow"]]
        if len(part) < 15:
            continue
        say(f"\n  {name}: {len(fills)} fills at 80c, {len(part)} with a trade record")
        say("    trigger |  10%   25%   50%   75% (seconds) | <5s  <20s  <60s  from bars | in time: poll 40s  20s   5s | descent: median  poll 20s  5s | gap-through")
        for trigger in TRIGGERS:
            secs = np.array([r["flow"]["lead"][trigger][0] for r in part])
            src = np.array([r["flow"]["lead"][trigger][1] == "bars" for r in part])
            fall = np.array([r["flow"]["descent"][trigger][0] for r in part])
            q = np.percentile(secs, [10, 25, 50, 75])
            cell = dict(fills=len(part), seconds_quantiles=[float(v) for v in q], under_5s=float((secs < 5).mean()), under_20s=float((secs < 20).mean()),
                        under_60s=float((secs < 60).mean()), from_bars=float(src.mean()), gap_through=float((secs == 0).mean()),
                        in_time={p: in_time(secs, p) for p in (40, 20, 5)}, descent_median=float(np.median(fall)),
                        descent_in_time={p: in_time(fall, p) for p in (20, 5)})
            out[f"speed | {name.strip()} | {trigger}"] = cell
            say(f'      {trigger}c   | {q[0]:5.0f} {q[1]:5.0f} {q[2]:5.0f} {q[3]:5.0f}           | {cell["under_5s"]:4.0%} {cell["under_20s"]:4.0%} {cell["under_60s"]:5.0%} {cell["from_bars"]:7.0%}    |'
                f'        {cell["in_time"][40]:6.1%} {cell["in_time"][20]:6.1%} {cell["in_time"][5]:5.1%} | {cell["descent_median"]:10.0f} {cell["descent_in_time"][20]:9.1%} {cell["descent_in_time"][5]:5.1%} | {cell["gap_through"]:5.1%}')
    say("\n  by half (LIVE groups), share in time at a 20s poll, first touch:")
    for name, test in SETS[:2]:
        for half in (1, 2):
            part = [r for r in rows if test(r) and r["fill_ts"] and r["flow"] and r["half"] == half]
            cell = {t: in_time([r["flow"]["lead"][t][0] for r in part], POLL) for t in TRIGGERS}
            out[f"speed | {name} | half {half}"] = dict(fills=len(part), in_time_20s=cell)
            say(f"    {name[:3]} half {half} (n {len(part):3}): "+"  ".join(f"{t}c {v:.1%}" for t, v in cell.items()))
    say("\n  bars only, every order: was the trigger's minute earlier than the fill minute? (share of fills; the rest shared one bar)")
    for name, test in SETS:
        fills = [r for r in rows if test(r) and r["fill_ts"]]
        if len(fills) < 15:
            continue
        cell = {t: float(np.mean([r["quote_ts"][t] < r["fill_ts"] for r in fills])) for t in TRIGGERS}
        out[f"speed bars | {name.strip()}"] = dict(fills=len(fills), earlier_minute=cell)
        say(f"    {name:36} n {len(fills):3}: "+"  ".join(f"{t}c {v:.0%}" for t, v in cell.items()))
    say("\n  are the fast dips the better ones? (LIVE fills with trades; lead from the 85c trigger)")
    part = [r for r in rows if r["group"] != "other tennis" and r["fill_ts"] and r["flow"]]
    for label, test in (("lead under 20s", lambda s: s < 20), ("lead 20s to 2 min", lambda s: 20 <= s < 120), ("lead 2 min or more", lambda s: s >= 120)):
        sub = [r for r in part if test(r["flow"]["lead"][85][0])]
        nets = [r["net"] for r in sub]
        out[f"speed | outcome | {label}"] = dict(fills=len(sub), win=float(np.mean([r["payout"] for r in sub])) if sub else None,
                                                 net=float(np.mean(nets)) if sub else None, ci=boot(nets))
        if sub:
            say(f"    {label:20} fills {len(sub):3} won {np.mean([r['payout'] for r in sub]):.1%} net {np.mean(nets):+5.1f}c {band(boot(nets))} = {np.mean(nets)/.8:+.1f}%")


# ---------------------------------------------------------------- capital

def segments(r, plan):
    """[(on, off)] reservation intervals of one order under a plan, and the index of the interval that holds the fill (or None).

    plan: ("always", level) | ("jit", level, trigger) | ("jit-cancel", level, trigger). The fill is the first in-play minute
    with a print strictly below the level; the interval that holds it runs to settlement.
    """
    kind, level = plan[0], plan[1]
    hit = next((p[0] for p in r["path"] if p[3] is not None and p[3] < level), None)
    if kind == "always":
        return [(r["start"]+60, r["settle"])], (0 if hit else None), hit
    trigger, out, on = plan[2], [], None
    for p in r["path"]:
        down = (p[2] is not None and p[2] <= trigger) or (p[3] is not None and p[3] <= trigger)
        if on is None and down:
            on = p[0]-60 if p[0]-60 > r["start"] else p[0]      # the trigger came some time inside this minute
        if on is not None and p[3] is not None and p[3] < level:
            return out+[(on, r["settle"])], len(out), hit
        if kind == "jit-cancel" and on is not None and not down and p[1] is not None and p[1] >= trigger+5:
            out.append((on, p[0]))
            on = None
    if on is not None:
        out.append((on, r["settle"]))
    return out, None, hit


def value(r, plan, ahead, size, poll=POLL):
    """Expected return on the money reserved by a filled order (profit / limit), times the chance the bid is in and fills.

    80c and 81c bids use the trade record where the order has one (touch-only events included: there the queue decides);
    other levels, and orders without a trade record, use the bar convention (a print strictly below the bid fills).
    """
    kind, level = plan[0], plan[1]
    net = (100*r["payout"]-level)/level
    f = r["flow"]
    if level > LIMIT+1 or f is None:
        return net if r["low"] is not None and r["low"] < level else 0.0
    if kind == "always":
        return net*chance(f["trades"], f["lo"], None, level, ahead, size)
    if level == LIMIT and f["t0"] is not None:
        when = (f["descent"] if kind == "jit-cancel" else f["lead"])[plan[2]][2]
    else:                                            # an 81c bid, or a touch-only event: in before the first print at 80c
        when = f["lead81"][plan[2]][2]
    return net*chance(f["trades"], f["lo"], when, level, ahead, size, poll)


def prepare(rows, plan, ahead, size, poll=POLL):
    """Per order: reservation intervals, the interval holding the fill, and the expected return of a fill."""
    jobs = []
    for r in rows:
        segs, holds, hit = segments(r, plan)
        worth = value(r, plan, ahead, size, poll) if segs else 0.0
        if worth and holds is None:                  # a touch-only event the queue let through: the last interval holds it
            holds = len(segs)-1
        jobs.append(dict(r=r, segs=segs, holds=holds, worth=worth, hit=hit))
    return jobs


def run(jobs, slots, wait):
    """Bankroll simulation with `slots` equal slots. Returns per-day returns on the bankroll and counts."""
    import heapq
    stream, n = [], 0
    for j in jobs:
        for i, (on, off) in enumerate(j["segs"]):
            heapq.heappush(stream, (on, n, j, i, off))
            n += 1
    busy, days, count = [], defaultdict(float), Counter()
    peak = defaultdict(int)
    while stream:
        now, _, j, i, off = heapq.heappop(stream)
        while busy and busy[0] <= now:
            heapq.heappop(busy)
        r = j["r"]
        if slots is None or len(busy) < slots:
            heapq.heappush(busy, off)
            peak[r["day"]] = max(peak[r["day"]], len(busy))
            count["placed"] += 1
            if j["holds"] == i:
                late = j["hit"] is not None and now > j["hit"]-60 and now > j["segs"][i][0]
                if not late:
                    days[r["day"]] += j["worth"]/(slots or 1)
                    count["fills"] += j["worth"] != 0
            continue
        if wait and busy[0] < off-60:
            n += 1
            heapq.heappush(stream, (busy[0], n, j, i, off))
            continue
        count["skipped"] += 1
    return days, count, peak


def calendar(rows):
    first, last = min(r["day"] for r in rows), max(r["day"] for r in rows)
    day, out = date.fromisoformat(first), []
    while day <= date.fromisoformat(last):
        out.append(day.isoformat())
        day += timedelta(days=1)
    return out


def daily(days, span):
    """Mean daily return (percent of bankroll) over the calendar days of `span`, interval over days, worst day, sd."""
    values = np.array([100*days.get(d, 0.0) for d in span])
    return dict(mean=float(values.mean()), ci=boot(values), worst=float(values.min()), sd=float(values.std()), days=len(span))


def spans(rows):
    """Order sets for the simulation: the full-coverage window, both halves of the live history, and all of it."""
    live = [r for r in rows if r["group"] != "other tennis"]
    begin = min(r["day"] for r in live if r["group"] == "CBB" and not r["archived"])
    end = min(max(r["day"] for r in live if r["group"] == g) for g in ("CBB", "FRQ"))
    return {"window": [r for r in live if begin <= r["day"] <= end], "half 1": [r for r in live if r["live_half"] == 1],
            "half 2": [r for r in live if r["live_half"] == 2], "all": live}


PLANS = [("rest from the start (live rule)", ("always", 80))]+[(f"just in time at {t}c", ("jit", 80, t)) for t in (90, 88, 86, 85, 84, 82)]+[
    ("just in time at 85c, cancel at 90c", ("jit-cancel", 80, 85)), ("just in time at 82c, cancel at 87c", ("jit-cancel", 80, 82)),
    ("bid 81c from the start", ("always", 81)), ("bid 81c just in time at 88c", ("jit", 81, 88)), ("bid 81c just in time at 85c", ("jit", 81, 85))]


def capital(rows, out):
    say("\n2. CAPITAL: how often the bid is placed, how long it rests, money-hours reserved (bars, every order; no slot limit)")
    say("   'placed' = share of favourites whose bid or print reached the trigger in play; 'hours' = mean hours reserved per ORDER OF THE RULE")
    say("   (zero when never placed; from placement to settlement, nothing cancelled unless stated); 'vs start' = reserved money-hours relative")
    say("   to resting from the start; 'per fill' and 'per placed' use the bar convention (a print below the bid fills, whatever the queue).")
    for name, test in SETS[:4]:
        part = [r for r in rows if test(r)]
        say(f"\n  {name}: {len(part)} orders, median event {np.median([(r['settle']-r['start'])/3600 for r in part]):.2f}h")
        base = None
        for label, plan in PLANS:
            got = [segments(r, plan) for r in part]
            hours = np.array([sum(off-on for on, off in segs)/3600 for segs, _, _ in got])
            placed = np.array([bool(segs) for segs, _, _ in got])
            hold = np.array([holds is not None for _, holds, _ in got])
            nets = np.array([100*r["payout"]-plan[1] for r in part])
            base = base or hours.sum()
            rest = [h for h, p, f in zip(hours, placed, hold) if p and not f]
            cell = dict(orders=len(part), placed=float(placed.mean()), fills=int(hold.sum()), hours_per_order=float(hours.mean()),
                        hours_unfilled_placed_median=float(np.median(rest)) if rest else None, relative_money_hours=float(hours.sum()/base),
                        net_per_fill=float(nets[hold].mean()) if hold.any() else None, cents_per_placed=float(nets[hold].sum()/max(placed.sum(), 1)),
                        pct_per_placed=float(100*nets[hold].sum()/plan[1]/max(placed.sum(), 1)),
                        pct_per_reserved_day=float(100*nets[hold].sum()/plan[1]/(hours.sum()/24)))
            out[f"capital | {name.strip()} | {label}"] = cell
            say(f'    {label:36} placed {cell["placed"]:4.0%} fills {cell["fills"]:3} | hours/order {cell["hours_per_order"]:5.2f} (unfilled, once placed: median {cell["hours_unfilled_placed_median"] or 0:4.2f}h) '
                f'vs start {cell["relative_money_hours"]:5.0%} | per fill {cell["net_per_fill"]:+5.1f}c | per bid placed {cell["pct_per_placed"]:+5.2f}% | per reserved dollar-day {cell["pct_per_reserved_day"]:+6.1f}%')

    say("\n   concurrency without a slot limit: orders reserved at once (daily peak), full-coverage window and all history")
    sets = spans(rows)
    for span in ("window", "all"):
        for label, plan in PLANS[:9]:
            jobs = prepare(sets[span], plan, 0, 3)
            _, count, peak = run(jobs, None, False)
            peaks = [peak.get(d, 0) for d in calendar(sets[span])]
            cell = dict(days=len(peaks), median_peak=float(np.median(peaks)), p90_peak=float(np.percentile(peaks, 90)), max_peak=int(max(peaks)))
            out[f"concurrency | {span} | {label}"] = cell
            say(f'    {span:6} {label:36} daily peak: median {cell["median_peak"]:4.0f}  90% {cell["p90_peak"]:4.0f}  max {cell["max_peak"]:3}')


def efficiency(rows, out):
    say("\n5A. RETURN ON THE MONEY ACTUALLY RESERVED (LIVE groups; trade record for 3 contracts where read, 20s poll)")
    say("   'rest from the start' with nothing ahead in the queue, just-in-time bids with 1,000 ahead (the comparison least favourable to just in time).")
    say("   per order = profit / the 80c a bid reserves, over every order of the rule; dollar-day = profit per dollar reserved for a day")
    say("   (money-hours from placement to settlement). Intervals: bootstrap over orders. This is what a bankroll earns only if it is kept fully")
    say("   reserved, i.e. when more favourites are on offer than slots; section 5B shows when that is.")
    sets = spans(rows)
    for span in ("all", "half 1", "half 2", "window"):
        part = sets[span]
        say(f"\n  {span}: {len(part)} orders")
        base = None
        for label, plan in PLANS:
            jobs = prepare(part, plan, 0 if plan[0] == "always" else 1000, 3)
            worth = np.array([100*j["worth"] for j in jobs])
            hours = np.array([sum(off-on for on, off in j["segs"])/3600 for j in jobs])
            fills = sum(abs(j["worth"])/abs((100*j["r"]["payout"]-plan[1])/plan[1]) for j in jobs if j["worth"])
            pick = RNG.integers(0, len(jobs), (2000, len(jobs)))
            ratio = np.percentile(worth[pick].sum(axis=1)/(hours[pick].sum(axis=1)/24), [2.5, 97.5])
            base = base or hours.sum()
            cell = dict(orders=len(jobs), fills=float(fills), pct_per_order=float(worth.mean()), ci_per_order=boot(worth),
                        money_hours_vs_start=float(hours.sum()/base), pct_per_dollar_day=float(worth.sum()/(hours.sum()/24)),
                        ci_per_dollar_day=[float(v) for v in ratio], pct_per_dollar_month=float(30*worth.sum()/(hours.sum()/24)))
            out[f"efficiency | {span} | {label}"] = cell
            say(f'    {label:36} fills {fills:6.1f} | per order {cell["pct_per_order"]:+.2f}% {band(cell["ci_per_order"], 2, "%")} | money-hours {cell["money_hours_vs_start"]:4.0%} of start '
                f'| per reserved dollar-day {cell["pct_per_dollar_day"]:+6.1f}% {band(cell["ci_per_dollar_day"], 0, "%")}')


def simulate(rows, out):
    say("\n5B. BANKROLL SIMULATION (LIVE groups): K equal slots, each 1/K of the bankroll; percent of bankroll per calendar day")
    say("   fills from the trade record for 3 contracts: 'rest from the start' with nothing ahead in the queue (most favourable to it), just-in-time bids")
    say("   with 1,000 contracts ahead (least favourable), a 20s poll. An order with no free slot is skipped (start) or waits (just in time).")
    say("   'skip' = orders that never got a slot. Month = 30 calendar days, not compounded. Loss on one losing fill = 1/K of the bankroll.")
    sets = spans(rows)
    for span, part in sets.items():
        cal = calendar(part)
        say(f"\n  {span}: {len(part)} orders over {len(cal)} calendar days ({cal[0]} to {cal[-1]})")
        say("    plan                                 |"+"".join(f" K={k:<2}: %/day (skipped; worst day, sd) " for k in (25, 12, 8, 5))+"| 95% interval of %/day (over days); %/month")
        for label, plan in PLANS:
            wait = plan[0] != "always"
            jobs = prepare(part, plan, 0 if plan[0] == "always" else 1000, 3)
            line, cell = f"    {label:36} |", {}
            for k in (25, 12, 8, 5):
                days, count, _ = run(jobs, k, wait)
                s = daily(days, cal)
                s["skipped"] = count["skipped"]/max(1, count["skipped"]+count["placed"])
                s["month"] = 30*s["mean"]
                cell[k] = s
                line += f' {s["mean"]:+6.2f}% ({s["skipped"]:3.0%}; {s["worst"]:+5.1f}%, {s["sd"]:4.1f}%)    '
            line += "| "+"; ".join(f'K={k}: {band(cell[k]["ci"], 2, "%")} {cell[k]["month"]:+5.1f}%/mo' for k in (25, 12, 8))
            out[f"simulation | {span} | {label}"] = cell
            say(line)


def queue_cost(rows, out):
    say("\n3. QUEUE COST (trade record, LIVE groups; seller-initiated prints only; a bid of 3 contracts, and of 100)")
    live = [r for r in rows if r["group"] != "other tennis"]
    fills = [r for r in live if r["fill_ts"] and r["flow"]]
    touch = [r for r in live if not r["fill_ts"] and r["touch_ts"] and r["flow"]]
    say(f"  strict fills with a trade record: {len(fills)} of {sum(bool(r['fill_ts']) for r in live)}; touch-only events (a print at 80c, none below): "
        f"{len(touch)} read of {sum(1 for r in live if not r['fill_ts'] and r['touch_ts'])}, out of {len(live)} orders")
    for name, test in SETS[:2]:
        part = [r for r in fills if test(r)]
        if not part:
            continue
        below, at, anyb = (np.array([r["flow"][k] for r in part]) for k in ("sold_below", "depth_at_80", "any_below"))
        cell = dict(fills=len(part), sold_below_quartiles=[float(v) for v in np.percentile(below, [10, 25, 50, 75])],
                    any_below_quartiles=[float(v) for v in np.percentile(anyb, [10, 25, 50, 75])],
                    depth_at_80_quartiles=[float(v) for v in np.percentile(at, [25, 50, 75, 90])],
                    share_no_print_at_80_first=float((at == 0).mean()),
                    share_sold_below_under={n: float((below < n).mean()) for n in (3, 10, 100)})
        out[f"queue | sizes | {name}"] = cell
        say(f'  {name}: seller-initiated contracts strictly below 80c within 30 min of t0 [10%,25%,50%,75%] {[round(v) for v in cell["sold_below_quartiles"]]} '
            f'(any initiator {[round(v) for v in cell["any_below_quartiles"]]}); under 3: {cell["share_sold_below_under"][3]:.0%}, under 10: {cell["share_sold_below_under"][10]:.0%}, under 100: {cell["share_sold_below_under"][100]:.0%}')
        say(f'     seller-initiated contracts AT 80c before t0 (the depth that was resting there) [25%,50%,75%,90%] {[round(v) for v in cell["depth_at_80_quartiles"]]}; '
            f'no print at 80c before the print below it: {cell["share_no_print_at_80_first"]:.0%}')
    say("\n  fills and return per ORDER OF THE RULE (percent of the 80c reserved by a bid), by contracts ahead of us at 80c; 20s poll for just-in-time")
    say("  ('bar' = the historical convention: every event with a print below 80c is one fill, touch-only events none)")
    for name, test in SETS[:3]:
        part = [r for r in live if test(r)]
        nets = np.array([r["net"] for r in part])
        bar = np.array([float(r["fill_ts"] is not None) for r in part])
        say(f"\n  {name}: {len(part)} orders; bar convention fills {bar.sum():.0f}, per fill {nets[bar > 0].mean():+.1f}c = {nets[bar > 0].mean()/.8:+.1f}%, per order {100*(bar*nets).sum()/80/len(part):+.2f}%")
        for size in (3, 100):
            for label, plan in (PLANS[0], PLANS[2], PLANS[4], PLANS[6], PLANS[7]):
                line = f"    {size:3} contracts, {label:34}"
                for ahead in (0, 50, 200, 500, 1000):
                    p = np.array([value(r, plan, ahead, size)*80/r["net"] for r in part])      # the chance of a fill (net is never zero)
                    cell = dict(fills=float(p.sum()), win=float((p*[r["payout"] for r in part]).sum()/max(p.sum(), 1e-9)),
                                per_fill=float((p*nets).sum()/max(p.sum(), 1e-9)), pct_per_order=float(100*(p*nets).sum()/80/len(part)),
                                ci=boot(100*p*nets/80))
                    out[f"queue | {name} | {size} | {label} | Q={ahead}"] = cell
                    line += f' | Q={ahead:<4} fills {cell["fills"]:5.1f} won {cell["win"]:.1%} {cell["pct_per_order"]:+.2f}%'
                say(line)
    say("\n  touch-only events (queue-dependent): fills by contracts ahead, bid resting from the start")
    for ahead in (0, 50, 200, 500, 1000):
        got = [r for r in touch if filled(r["flow"]["trades"], r["flow"]["lo"]-1, LIMIT, ahead, 3)]
        nets = [r["net"] for r in got]
        out[f"queue | touch-only | Q={ahead}"] = dict(events=len(touch), fills=len(got), wins=sum(r["payout"] for r in got))
        say(f"    Q={ahead:<4}: {len(got):2} of {len(touch)} fill, {sum(r['payout'] for r in got):.0f} won"+(f", net {np.mean(nets):+.1f}c" if nets else ""))
    say(f"    all touch-only events on the bars ({sum(1 for r in live if not r['fill_ts'] and r['touch_ts'])}): "
        f"{sum(r['payout'] for r in live if not r['fill_ts'] and r['touch_ts']):.0f} won")

    say("\n  which fills does a just-in-time bid lose? (LIVE strict fills; chance the bid fills, 20s poll, 3 contracts, 1,000 ahead)")
    for trigger in (90, 88, 85, 82):
        p = np.array([chance(r["flow"]["trades"], r["flow"]["lo"], r["flow"]["lead"][trigger][2], LIMIT, 1000, 3) for r in fills])
        base = np.array([chance(r["flow"]["trades"], r["flow"]["lo"], None, LIMIT, 1000, 3) for r in fills])
        pay, nets = np.array([r["payout"] for r in fills]), np.array([r["net"] for r in fills])
        late = np.array([1-in_time([r["flow"]["lead"][trigger][0]], POLL) for r in fills])
        lost = base-p
        cell = dict(fills=len(fills), fills_resting=float(base.sum()), fills_jit=float(p.sum()), arrives_late=float(late.sum()),
                    lost=float(lost.sum()), lost_win=float((lost*pay).sum()/max(lost.sum(), 1e-9)), kept_win=float((p*pay).sum()/p.sum()),
                    lost_net=float((lost*nets).sum()/max(lost.sum(), 1e-9)), kept_net=float((p*nets).sum()/p.sum()))
        out[f"queue | lost | {trigger}"] = cell
        say(f'    trigger {trigger}c: of {cell["fills_resting"]:.1f} fills resting from the start, the bid arrives after t0 on {cell["arrives_late"]:.1f} and still fills on all but '
            f'{cell["lost"]:.1f} ({cell["lost"]/cell["fills_resting"]:.1%}); lost fills won {cell["lost_win"]:.0%} (net {cell["lost_net"]:+.1f}c), kept fills won {cell["kept_win"]:.1%} (net {cell["kept_net"]:+.1f}c)')
    quick = [r for r in fills if r["flow"]["seconds_below"] is not None and r["flow"]["seconds_below"] <= 20]
    slow = [r for r in fills if r not in quick]
    for label, sub in (("back above 80c within 20s of t0", quick), ("below 80c for longer", slow)):
        if sub:
            nets = [r["net"] for r in sub]
            out[f"queue | {label}"] = dict(fills=len(sub), win=float(np.mean([r["payout"] for r in sub])), net=float(np.mean(nets)), ci=boot(nets))
            say(f"    {label:32} fills {len(sub):3} won {np.mean([r['payout'] for r in sub]):.1%} net {np.mean(nets):+5.1f}c {band(boot(nets))}")


def alternatives(rows, out):
    say("\n4. ALTERNATIVES")
    live = [r for r in rows if r["group"] != "other tennis"]
    say("  (a) a higher bid, bar convention (a print strictly below the bid fills; an 81c bid fills on any print at 80c, ahead of every 80c bid)")
    for name, test in SETS[:3]:
        part = [r for r in live if test(r)]
        for level in (80, 81, 82, 83, 85):
            line = f"    {name[:4]} bid {level}c"
            for half in (1, 2, 0):
                sub = [r for r in part if half in (0, r["half" if name[:4] != "LIVE" else "live_half"])]
                hit = [r for r in sub if r["low"] is not None and r["low"] < level]
                nets = [100*r["payout"]-level for r in hit]
                per = [((100*r["payout"]-level)/level*100 if r["low"] is not None and r["low"] < level else 0.0) for r in sub]
                cell = dict(orders=len(sub), fills=len(hit), win=float(np.mean([r["payout"] for r in hit])), net=float(np.mean(nets)), ci=boot(nets),
                            pct_per_order=float(np.mean(per)), pct_ci=boot(per))
                out[f"alt | bid {level} | {name.strip()} | {'all' if not half else 'half '+str(half)}"] = cell
                line += f' | {"all" if not half else "half "+str(half)}: fills {len(hit):3} ({len(hit)/len(sub):3.0%}) won {cell["win"]:.1%} per fill {cell["net"]:+5.1f}c per order {cell["pct_per_order"]:+.2f}% {band(cell["pct_ci"], 2, "%")}'
            say(line)
    have = [r for r in live if r["flow"]]
    cell = {t: in_time([r["flow"]["lead81"][t][0] for r in have], POLL) for t in TRIGGERS}
    out["alt | bid 81 | in time before the first print at 80c, 20s poll"] = dict(events=len(have), in_time=cell)
    say(f"    an 81c bid must be in before the first print at 80c: in time at a 20s poll on "+"  ".join(f"{t}c {v:.1%}" for t, v in cell.items())+f"  (n {len(have)})")
    extra = [r for r in live if r["low"] == 80]
    say(f"    the events an 81c bid adds (low exactly 80c): {len(extra)}, won {np.mean([r['payout'] for r in extra]):.0%}; it pays 1c more on the other "
        f"{sum(r['fill_ts'] is not None for r in live)} fills")

    say("\n  (b) a bid on BOTH books (research/favourite_gap.py q2 rows: Challenger, WTA, WTA Challenger favourites at 90c+ by the warm-up price)")
    path = PROC/"favourite-gap/q2-rows.json"
    if path.exists():
        q = json.loads(path.read_text())
        for period in ("holdout", "recent", "both"):
            sub = [x for x in q if period in (x["period"], "both")]
            a = [x for x in sub if x["dip_a"] is not None]
            b = [x for x in sub if x["dip_b"] is not None]
            either = [x for x in sub if x["dip_a"] is not None or x["dip_b"] is not None]
            only = [x for x in either if x["dip_a"] is None or x["dip_b"] is None]
            nets = lambda xs: [100*x["payout"]-80 for x in xs]
            cell = dict(matches=len(sub), fills_cached_book=len(a), fills_other_book=len(b), fills_either=len(either), only_one_book=len(only),
                        net_cached=float(np.mean(nets(a))), net_either=float(np.mean(nets(either))), net_only_one=float(np.mean(nets(only))) if only else None,
                        ci_only_one=boot(nets(only)), per_order_one_book=float(sum(nets(a))/80/len(sub)*100),
                        per_order_both_books=float(sum(nets(either))/160/len(sub)*100))
            out[f"alt | both books | {period}"] = cell
            say(f'    {period:7} matches {len(sub):4}: fills on the cached book {len(a)} (net {cell["net_cached"]:+.1f}c), other book {len(b)}, either {len(either)} (net {cell["net_either"]:+.1f}c); '
                f'on one book only {len(only)}: net {cell["net_only_one"]:+.1f}c {band(cell["ci_only_one"])} | per order on reserved money: one book {cell["per_order_one_book"]:+.2f}%, both books (twice the money) {cell["per_order_both_books"]:+.2f}%')

    say("\n  (c) is there a pre-event predictor of dipping? fill rate and return per order by pre-event price band, both halves (bar convention)")
    for name, test in SETS[:4]:
        part = [r for r in rows if test(r)]
        key = "live_half" if name.startswith("LIVE") else "half"
        cut = sorted(r["settle"] for r in part)[len(part)//2]
        for lo, hi in ((90, 93), (93, 96), (96, 99)):
            line = f"    {name[:5]:5} {lo}-{hi}c"
            for half in (1, 2):
                sub = [r for r in part if lo <= r["mid"] < hi and (r["settle"] < cut) == (half == 1)]
                hit = [r for r in sub if r["fill_ts"]]
                per = [r["net"]/.8 if r["fill_ts"] else 0.0 for r in sub]
                hours = sum(r["settle"]-r["start"] for r in sub)/3600
                cell = dict(orders=len(sub), fill_rate=len(hit)/max(len(sub), 1), win=float(np.mean([r["payout"] for r in hit])) if hit else None,
                            pct_per_order=float(np.mean(per)) if sub else None, ci=boot(per),
                            pct_per_reserved_day=float(sum(per)/(hours/24)) if hours else None)
                out[f"alt | band | {name.strip()} | {lo}-{hi} | half {half}"] = cell
                line += (f' | half {half}: orders {len(sub):4} fill rate {cell["fill_rate"]:4.0%} won {(cell["win"] or 0):.0%} per order {(cell["pct_per_order"] or 0):+.2f}% {band(cell["ci"], 2, "%")} '
                         f'per reserved dollar-day {(cell["pct_per_reserved_day"] or 0):+5.1f}%')
            say(line)
    say("  (d) a smaller stake on every event against a full stake on fewer: the K columns of section 5B (K slots = a stake of 1/K of the bankroll).")


def analyse(rows):
    measure(rows)
    out = {}
    have = Counter((r["group"], "fill" if r["fill_ts"] else "touch", bool(r["flow"])) for r in rows if windows(r))
    say("JUST-IN-TIME DIP BIDS")
    say(f"orders: {dict(Counter(r['group'] for r in rows))}; trade windows read: "+", ".join(f"{g} {k} {'yes' if ok else 'no'} {n}" for (g, k, ok), n in sorted(have.items())))
    speed(rows, out)
    capital(rows, out)
    queue_cost(rows, out)
    alternatives(rows, out)
    efficiency(rows, out)
    simulate(rows, out)
    save(OUT/"summary.json", out)
    (OUT/"report.txt").write_text(buffer.getvalue())
    keep = ("event", "ticker", "tour", "cell", "group", "start", "settle", "mid", "payout", "day", "low", "fill_ts", "touch_ts", "print_ts", "quote_ts")
    flat = []
    for r in rows:
        x = {k: r[k] for k in keep}
        if r["flow"]:
            f = r["flow"]
            x.update(t0=f["t0"], t80=f["t80"], lead={t: v[:2] for t, v in f["lead"].items()}, descent={t: v[:2] for t, v in f["descent"].items()},
                     depth_at_80=f["depth_at_80"], sold_below=f["sold_below"], sold_at_80=f["sold_at_80"], seconds_below=f["seconds_below"])
        flat.append(x)
    save(OUT/"rows.json", flat)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("steps", nargs="+", choices=("build", "download", "analyse"))
    parser.add_argument("--minutes", type=float, default=35, help="stop new downloads after this long")
    args = parser.parse_args()
    if "build" in args.steps:
        build()
    rows = load()
    if "download" in args.steps:
        download(rows, args.minutes)
    if "analyse" in args.steps:
        analyse(rows)


if __name__ == "__main__":
    main()
