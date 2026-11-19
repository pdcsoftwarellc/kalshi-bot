"""Just-in-time dip bid with the trigger the live trader actually uses: the favourite's ASK, not its last trade.

research/dip_funding.py placed the 80c bid when the favourite first TRADED at
or below T. paper/trader.py `fills` instead calls a bid "near" when the
favourite's ask in the /markets listing (polled every 20 seconds) is at or
below bid+8c. The cached one-minute bars carry yes_bid and yes_ask
(open/high/low/close) for every order of the study, so the quote trigger can be
replayed without any network read. This script does that, with the same orders,
prints and accounting as dip_funding (imported; nothing re-derived).

Favourite's ask per bar: yes_ask when the bid rests on the favourite's own
market as YES; 100 - yes_bid when it rests on the underdog's market as NO (so
the ask LOW of the minute is 100 - yes_bid HIGH). An empty side (0 or 100) is
no quote and cannot trigger.

Trigger conventions
  low     the bid is placed at the END of the first in-play minute whose ask
          low is at or below T (the ask touched T at some moment of that
          minute; a 20-second poll can miss a touch shorter than 20 seconds).
  close   the same with the ask at the minute's close (what a poll once a
          minute, at the minute's end, would see). low and close bound the
          20-second poll.
  low-opt the low trigger timed at the START of its minute and a bar's
          through-print at its end: the optimistic bound on timing.
  If the last bar before play already closes with the ask at or below T, the
  bid is placed at the start (it is the resting bid).
  No cancel: the trader's simulated "_jit" order is never cancelled.
  kept / late / missed, strict fills, funding and peaks: as dip_funding.
  Funding runs from the START of the trigger minute (low) or its end (close).

Latency (section 2): the ask first reached T somewhere in its minute; the
moment is bracketed by the minute's start and by the first buyer-initiated
trade at or below T in that minute where the trade record covers it (a buyer
lifts the ask, so the ask was at or below T by then), else the minute's end.
The first print under 80c is exact where the trade record covers it, else
bracketed by its minute. A fill is lost at delay D when trigger + D is after
that print: reported with the trigger at its latest (pessimistic), earliest
(optimistic) and uniform between (expected).

    .venv/bin/python research/dip_funding_ask.py

Offline. Output: data/processed/dip-funding-ask/report.txt and summary.json.
"""
import io
import json
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from research.api_cost import ROOT
from research.near_end import cents
from research import dip_capacity as dc
from research import dip_funding as df
from research import jit_bids as jit
from research import wcbb_backfill as w

OUT = ROOT / "data/processed/dip-funding-ask"
TRIGGERS = (90, 88, 86)
DELAYS = (0, 20, 60)
BID, DRAWS = 80, 4000
GROUPS = dc.GROUPS
RNG = np.random.default_rng(29)
buffer = io.StringIO()


def say(text=""):
    print(text, flush=True)
    buffer.write(text+"\n")


# ---------------------------------------------------------------- quotes

def women_bars():
    batches = {}
    for r in w.recent_requests():
        if w.cache_file(*r).exists():
            for x in json.loads(w.cache_file(*r).read_text()).get("markets", []):
                batches[x["market_ticker"]] = sorted(x.get("candlesticks") or [], key=lambda b: b["end_period_ts"])
    events = {e[2]: e for archived in (True, False) for e in w.games(archived)}
    return lambda o: w.bars_of(events[o["event"]], batches)


def quotes(o, women):
    """[(end of minute, favourite's ask low, ask close)] for every cached bar, None where that side of the book is empty."""
    if o["group"] == "women":
        bars = women(o)
    else:
        file = jit.bar_file(o["series"], o["ticker"], o["settle"])
        bars = None if file is None else json.loads(file.read_text()).get("candlesticks")
    out = []
    for b in sorted(bars or [], key=lambda b: b["end_period_ts"]):
        if o["yes"]:
            low, close = cents(b, "yes_ask", "low"), cents(b, "yes_ask", "close")
        else:
            high, last = cents(b, "yes_bid", "high"), cents(b, "yes_bid", "close")
            low, close = (None if high is None else 100-high), (None if last is None else 100-last)
        out.append((b["end_period_ts"], low if low is not None and 0 < low < 100 else None, close if close is not None and 0 < close < 100 else None))
    return out


def trigger(o, level, mode):
    """(end of the trigger minute, True if the ask was already there before play) or (None, False)."""
    before = [q for q in o["quotes"] if q[0] <= o["start"]+60]
    if before and before[-1][2] is not None and before[-1][2] <= level:
        return o["start"], True
    for end, low, close in o["quotes"]:
        value = low if mode == "low" else close
        if end > o["start"]+60 and value is not None and value <= level:
            return end, False
    return None, False


def replay(o, placed, optimistic=False):
    """Strict / through for a bid live from `placed` (None = never), on dip_capacity's prints."""
    strict, through, seen = False, 0.0, False
    for when, lo, hi, size, seller in o["stream"]:
        at = when if (seller is not None or optimistic) else when-60
        if hi < BID:
            if placed is not None and at >= placed:
                strict = strict or not seen
                through += size
            seen = True
    return dict(strict=strict, through=through)


def plan(o, level, mode, delay=0):
    end, pre = o["trig"][level, mode.split("-")[0]]
    if end is None:
        return dict(strict=False, through=0.0, spans=[], placed=None)
    if pre:
        placed, fund = o["start"], o["start"]
    elif mode == "low-opt":
        placed, fund = end-60+delay, end-60
    else:
        placed, fund = end+delay, (end-60 if mode == "low" else end)
    r = replay(o, placed, mode == "low-opt")
    r.update(spans=[(max(o["start"], fund), max(o["settle"], fund))], placed=placed)
    return r


# ---------------------------------------------------------------- statistics

def net(o):
    return 100*o["payout"]-BID


def cell(part):
    if not part:
        return "  0     n/a              "
    nets = [net(o) for o in part]
    return f"{len(part):3} win {100*np.mean([o['payout'] for o in part]):5.1f}% {np.mean(nets):+5.1f}c={100*np.mean(nets)/BID:+5.1f}%"


def share_ci(flags):
    """Share of baseline fills kept, with an interval that resamples games (each order is one game)."""
    flags = np.asarray(flags, float)
    pick = RNG.integers(0, len(flags), (DRAWS, len(flags)))
    lo, hi = np.percentile(flags[pick].mean(axis=1), [2.5, 97.5])
    return f"{100*flags.mean():5.1f}% [{100*lo:4.0f},{100*hi:4.0f}]"


def profit(base, flags):
    """Profit of the kept fills as a share of the resting bid's profit, same games resampled together."""
    nets, flags = np.array([net(o) for o in base]), np.asarray(flags, float)
    pick = RNG.integers(0, len(base), (DRAWS, len(base)))
    whole, part = nets[pick].sum(axis=1), (nets*flags)[pick].sum(axis=1)
    ok = whole > 0
    lo, hi = np.percentile(100*part[ok]/whole[ok], [2.5, 97.5])
    dl, dh = np.percentile((part-whole)/len(base), [2.5, 97.5])
    return 100*(nets*flags).sum()/nets.sum(), lo, hi, ((nets*flags).sum()-nets.sum())/len(base), dl, dh


def halves(base):
    return base[:len(base)//2], base[len(base)//2:]


def needed(p1, p2, alpha=1.96, power=0.8416):
    """Games per arm for a two-sided 5% test with 80% power between two fill rates."""
    return (alpha+power)**2*(p1*(1-p1)+p2*(1-p2))/(p1-p2)**2


# ---------------------------------------------------------------- report

def report():
    women = women_bars()
    rows = [dc.measure(o) for o in dc.orders()]
    for o in rows:
        o["stream"] = dc.prints(o)[0]
        o["quotes"] = quotes(o, women)
        o["trig"] = {(t, m): trigger(o, t, m) for t in TRIGGERS for m in ("low", "close")}
        o["ask"] = {(t, m, d): plan(o, t, m, d) for t in TRIGGERS for m in ("low", "close", "low-opt") for d in DELAYS}
        o["trade"] = {(t, d): df.replay(o, t, d, False) for t in TRIGGERS for d in (20, 60)}
    OUT.mkdir(parents=True, exist_ok=True)
    by_group = {key: [o for o in rows if o["group"] == key] for key, _ in GROUPS}
    hits = {key: sorted(dc.fills(by_group[key], BID), key=lambda o: (o["settle"], o["event"])) for key, _ in GROUPS}
    summary = {}

    say("=== 0. WHAT THE CACHED BARS HOLD (no network read was needed)")
    for key, name in GROUPS:
        part = by_group[key]
        have = sum(bool(o["quotes"]) for o in part)
        inplay = [q for o in part for q in o["quotes"] if o["start"]+60 < q[0] <= o["settle"]]
        gaps = [b[0]-a[0] for o in part for a, b in zip(o["quotes"], o["quotes"][1:]) if a[0] > o["start"]]
        say(f"  {name:32} orders {len(part):4} with quote bars {have:4} | in-play minutes {len(inplay):7,}, favourite's ask present in {100*np.mean([q[1] is not None for q in inplay]):5.1f}% "
            f"| consecutive bars one minute apart {100*np.mean(np.array(gaps) == 60):5.1f}% | NO-side orders {100*np.mean([not o['yes'] for o in part]):3.0f}% "
            f"| baseline fills {len(hits[key])} {cell(hits[key])[4:]}")
    say("\n=== 1. ASK TRIGGER AGAINST TRADE TRIGGER (80c bid, strict print-through fills, no cancel; returns per fill on 80c)")
    say("    kept = bid live at or before the game's first print under 80c. late = arrives after it, a later print goes through. missed = no fill.")
    say("    trade 60s / 20s = dip_funding's figures (first print at or below T, timed to the end of its minute, bid live 60 / 20 seconds later).")
    say("    ask low / close = bid live at the END of the first minute whose ask low / close is at or below T. ask low-opt = the low trigger at the START of its minute.")
    say("    hours = bid-hours funded as % of the resting bid's (every order). placed = % of orders that ever get a bid.")
    variants = [("trade 60s", "trade", 60), ("trade 20s", "trade", 20), ("ask low", "low", 0), ("ask close", "close", 0), ("ask low-opt", "low-opt", 0)]

    def result(o, t, kind, d):
        return o["trade"][t, d] if kind == "trade" else o["ask"][t, kind, d]

    for key, name in GROUPS:
        base, part = hits[key], by_group[key]
        total = sum(o["settle"]-o["start"] for o in part)
        say(f"  {name}: baseline fills {cell(base)}")
        for t in TRIGGERS:
            for label, kind, d in variants:
                res = [result(o, t, kind, d) for o in base]
                keep = [o for o, r in zip(base, res) if r["strict"]]
                late = [o for o, r in zip(base, res) if not r["strict"] and r["through"] > 0]
                miss = [o for o, r in zip(base, res) if not r["strict"] and r["through"] <= 0]
                a, b = halves([r["strict"] for r in res])
                spans = [s for o in part for s in result(o, t, kind, d)["spans"]]
                placed = np.mean([bool(result(o, t, kind, d)["spans"]) for o in part])
                say(f"    T {t}c {label:11}: kept {share_ci([r['strict'] for r in res])} halves {100*np.mean(a):3.0f}/{100*np.mean(b):3.0f}% | kept {cell(keep)} | late {cell(late)} | missed {cell(miss)} "
                    f"| placed {100*placed:4.1f}% hours {100*sum(y-x for x, y in spans)/total:4.1f}%")
                summary[f"{key}|T{t}|{label}"] = dict(baseline=len(base), kept=len(keep), late=len(late), missed=len(miss), net_kept=float(np.mean([net(o) for o in keep])) if keep else None,
                                                      hours=sum(y-x for x, y in spans)/total)
            both = np.array([[o["trade"][t, 60]["strict"], o["ask"][t, "low", 0]["strict"]] for o in base], float)
            pick = RNG.integers(0, len(base), (DRAWS, len(base)))
            diff = (both[:, 1]-both[:, 0])[pick].mean(axis=1)
            say(f"      ask low minus trade 60s, kept share: {100*(both[:, 1]-both[:, 0]).mean():+5.1f} points [{100*np.percentile(diff, 2.5):+5.1f},{100*np.percentile(diff, 97.5):+5.1f}]; "
                f"kept by ask only {int(((both[:, 1] == 1) & (both[:, 0] == 0)).sum())}, by trade only {int(((both[:, 0] == 1) & (both[:, 1] == 0)).sum())}")

    say("\n  1b. BIDS FUNDED AT ONCE, November 2026, all three sports (and per sport); whole-sample peak per sport in brackets")
    now = [o for o in rows if o["month"] == "2026-11"]

    def peaks(spans_of):
        top = df.most([s for o in now for s in spans_of(o)])[0]
        each = {key: df.most([s for o in now if o["group"] == key for s in spans_of(o)])[0] for key, _ in GROUPS}
        whole = {key: df.most([s for o in by_group[key] for s in spans_of(o)])[0] for key, _ in GROUPS}
        days = sorted({o["day"] for o in now})
        daily = [df.most([s for o in now if o["day"] == day for s in spans_of(o)])[0] for day in days]
        return top, each, whole, max(daily[:len(days)//2]), max(daily[len(days)//2:])

    lines = [("resting all game", lambda o: [(o["start"], o["settle"])])]
    for t in TRIGGERS:
        lines += [(f"T {t}c trade 60s", lambda o, t=t: o["trade"][t, 60]["spans"]), (f"T {t}c ask low", lambda o, t=t: o["ask"][t, "low", 0]["spans"]),
                  (f"T {t}c ask close", lambda o, t=t: o["ask"][t, "close", 0]["spans"])]
    for label, fn in lines:
        top, each, whole, first, second = peaks(fn)
        say(f"    {label:18} most at once {top:3} (men {each['men']:2}, women {each['women']:2}, tennis {each['tennis']:2}) [whole sample: men {whole['men']:2}, women {whole['women']:2}, tennis {whole['tennis']:2}] "
            f"| peak in first / second half of the month {first:2}/{second:2}")
        summary[f"peak|{label}"] = dict(peak=top, by_sport=each, whole=whole)

    say("\n=== 2. LATENCY: fills lost when the bid is live D seconds after the ask first reaches T (ask-low trigger)")
    say("    pess = trigger at the latest moment the record allows, opt = at the start of its minute, exp = uniform between. A bid whose ask was already at T before play is the resting bid: never lost. 'timed' = fills whose first print under 80c is timed")
    say("    to the second by the trade record. Shares are of the baseline fills.")
    lead = {}
    for key, name in GROUPS:
        base = hits[key]
        say(f"  {name} ({len(base)} baseline fills)")
        for t in TRIGGERS:
            out = []
            for o in base:
                first = next((p for p in o["stream"] if p[2] < BID), None)
                end, pre = o["trig"][t, "low"]
                if first is None:
                    out.append(None)
                    continue
                t0 = (first[0], first[0]) if first[4] is not None else (first[0]-60, first[0])
                if end is None:
                    g = (np.inf, np.inf)
                elif pre:
                    g = (o["start"], o["start"])
                else:
                    lift = [p[0] for p in o["stream"] if p[4] is False and end-60 < p[0] <= end and p[1] <= t]
                    g = (end-60, min(lift) if lift else end)
                out.append(dict(t0=t0, g=g, exact=first[4] is not None, o=o, pre=pre))
            lead[key, t] = out
            ok = [x for x in out if x]
            text = f"    T {t}c: first print timed to the second {sum(x['exact'] for x in ok):3} of {len(ok)}"
            for d in DELAYS:
                pess = [not x["pre"] and x["g"][1]+d > x["t0"][0] for x in ok]
                opt = [not x["pre"] and x["g"][0]+d > x["t0"][1] for x in ok]
                exp = []
                for x in ok:
                    (a, b), (c, e) = x["g"], x["t0"]
                    if x["pre"]:
                        exp.append(0.0)
                    elif not np.isfinite(a):
                        exp.append(1.0)
                    elif b == a and e == c:
                        exp.append(float(a+d > c))
                    else:
                        gs, ts = np.linspace(a, b, 61), np.linspace(c, e, 61)
                        exp.append(float(np.mean(gs[:, None]+d > ts[None, :])))
                for x, v in zip(ok, exp):
                    x["o"].setdefault("lost", {})[t, d] = v
                a, b = halves(exp)
                lost = [x["o"] for x, flag in zip(ok, pess) if flag]
                text += (f" | D {d:2}s: lost pess {100*np.mean(pess):4.1f}% exp {100*np.mean(exp):4.1f}% (halves {100*np.mean(a):4.1f}/{100*np.mean(b):4.1f}) opt {100*np.mean(opt):4.1f}%"
                         f"; pess-lost fills net {np.mean([net(o) for o in lost]) if lost else float('nan'):+5.1f}c")
                summary[f"latency|{key}|T{t}|D{d}"] = dict(pess=float(np.mean(pess)), exp=float(np.mean(exp)), opt=float(np.mean(opt)), n=len(ok))
            say(text)
            if t == 88:
                gaps = np.array([x["t0"][0]-x["g"][1] for x in ok if np.isfinite(x["g"][1]) and not x["pre"]])
                say(f"           seconds from the trigger (latest) to the first print under 80c (earliest): percentiles 5/10/25/50/75 = "
                    + " / ".join(f"{v:,.0f}" for v in np.percentile(gaps, [5, 10, 25, 50, 75]))+f" ({len(gaps)} fills whose ask was above T at the start); under 0 / 20 / 60 / 120 s: "
                    + " / ".join(f"{100*np.mean(gaps < v):.0f}%" for v in (0.001, 20, 60, 120)))
                extra = [100*np.mean([not x["pre"] and x["g"][1]+d > x["t0"][0] for x in ok])-100*np.mean([not x["pre"] and x["g"][1] > x["t0"][0] for x in ok]) for d in (20, 60)]
                say(f"           cost of the delay alone (pessimistic timing): 20s loses {extra[0]:.1f} points of the baseline fills, 60s loses {extra[1]:.1f}")

    say("\n=== 3. NO WARNING: the ask is above T until the minute of the first print under 80c (or later) - fills only a resting bid gets")
    say("    one bar = the ask first reaches T in the same minute as the first print under 80c, or after it, or never. 20s sure / possible = under 20 seconds between them")
    say("    whatever the timing inside the minute / at the least favourable timing. 'never' = the ask (as the bars record it) was not at or below T before the print at all.")
    for key, name in GROUPS:
        base = hits[key]
        say(f"  {name} ({len(base)} baseline fills, {cell(base)[4:]})")
        for t in TRIGGERS:
            ok = [x for x in lead[key, t] if x]
            minute = lambda v: np.ceil(v/60)*60
            bar = [x for x in ok if not x["pre"] and (not np.isfinite(x["g"][0]) or x["g"][0]+60 >= minute(x["t0"][1]))]
            never = [x for x in ok if not x["pre"] and (not np.isfinite(x["g"][0]) or x["g"][0] >= x["t0"][1])]
            sure = [x for x in ok if not x["pre"] and x["g"][0]+20 > x["t0"][1]]
            maybe = [x for x in ok if not x["pre"] and x["g"][1]+20 > x["t0"][0]]
            pre = sum(x["pre"] for x in ok)
            a, b = halves([x in bar for x in ok])
            say(f"    T {t}c: within one bar {100*len(bar)/len(ok):4.1f}% (halves {100*np.mean(a):3.0f}/{100*np.mean(b):3.0f}%) {cell([x['o'] for x in bar])} | ask after the print or never {cell([x['o'] for x in never])} "
                f"| under 20s sure {cell([x['o'] for x in sure])} | under 20s possible {cell([x['o'] for x in maybe])} | ask already at T before play {pre}")
            warned = [x["o"] for x in ok if x not in bar]
            nets = [net(o) for o in warned]
            say(f"           the fills WITH more than a bar of warning: {cell(warned)} {dc.band(dc.boot(nets))}")
            summary[f"nowarning|{key}|T{t}"] = dict(n=len(ok), one_bar=len(bar), never=len(never), sure20=len(sure), possible20=len(maybe),
                                                    net_one_bar=float(np.mean([net(x["o"]) for x in bar])) if bar else None)

    say("\n=== 4. SHARE OF THE RESTING BID'S PROFIT KEPT (sum of cents over kept fills / sum over all baseline fills; games resampled)")
    say("    'cost' = cents per BASELINE fill given up against the resting bid (negative = just-in-time earns less per game that would have filled).")
    say("    Profit share is a ratio of small noisy sums: read the cost column and its interval first. 'bar end' rows time the trigger to the end of its minute (as section 1);")
    say("    'timed' rows use section 2's bracket of the trigger inside its minute, each fill weighted by its chance of being kept (uniform timing).")
    for key, name in GROUPS:
        base = hits[key]
        say(f"  {name}: resting bid {cell(base)}")
        for t in TRIGGERS:
            for label, flag in (("trade 60s strict", lambda o: o["trade"][t, 60]["strict"]), ("ask low, bar end", lambda o: o["ask"][t, "low", 0]["strict"]),
                                ("ask low, bar end + 20s", lambda o: o["ask"][t, "low", 20]["strict"]), ("ask low, bar end + 60s", lambda o: o["ask"][t, "low", 60]["strict"]),
                                ("ask low timed, D 0 exp", lambda o: 1-o.get("lost", {}).get((t, 0), 1)), ("ask low timed, D 20s exp", lambda o: 1-o.get("lost", {}).get((t, 20), 1)),
                                ("ask low timed, D 60s exp", lambda o: 1-o.get("lost", {}).get((t, 60), 1)),
                                ("ask close", lambda o: o["ask"][t, "close", 0]["strict"]), ("ask low-opt", lambda o: o["ask"][t, "low-opt", 0]["strict"]),
                                ("ask low, late fills too", lambda o: o["ask"][t, "low", 0]["through"] > 0)):
                flags = [float(flag(o)) for o in base]
                share, lo, hi, cost, dl, dh = profit(base, flags)
                parts = []
                for half, fl in zip(halves(base), halves(flags)):
                    nets = np.array([net(o) for o in half])
                    parts.append(f"{100*(nets*np.array(fl)).sum()/nets.sum():4.0f}% of {nets.mean():+5.1f}c")
                say(f"    T {t}c {label:24}: fills kept {100*np.mean(flags):5.1f}% | profit kept {share:5.1f}% [{lo:4.0f},{hi:4.0f}] | cost {cost:+5.2f}c [{dl:+5.2f},{dh:+5.2f}] per baseline fill = {100*cost/BID:+5.2f}% "
                    f"| halves {parts[0]} / {parts[1]}")
                summary[f"profit|{key}|T{t}|{label}"] = dict(kept=float(np.mean(flags)), profit_share=share, interval=[lo, hi], cost=cost, cost_interval=[dl, dh])

    say("\n  4b. WOMEN'S COLLEGE BASKETBALL BY SEASON (last season's start is estimated and its books were thin; November 2026 is the season being traded)")
    for label, test in (("2025-26", lambda o: o["month"] < "2026-07"), ("2026-27 (November)", lambda o: o["month"] >= "2026-07")):
        base = [o for o in hits["women"] if test(o)]
        say(f"    {label}: resting bid {cell(base)}")
        for t in TRIGGERS:
            text = f"      T {t}c"
            for name, flag in (("trade 60s", lambda o: o["trade"][t, 60]["strict"]), ("ask low bar end", lambda o: o["ask"][t, "low", 0]["strict"]), ("ask low timed D 20s exp", lambda o: 1-o.get("lost", {}).get((t, 20), 1))):
                flags = [float(flag(o)) for o in base]
                share, lo, hi, cost, dl, dh = profit(base, flags)
                text += f" | {name}: kept {100*np.mean(flags):5.1f}% profit {share:5.1f}% cost {cost:+5.2f}c [{dl:+5.2f},{dh:+5.2f}]"
            say(text)

    say("\n=== 5. A REAL-MONEY A/B TEST: alternate games between a resting bid and a just-in-time bid (3 contracts each)")
    say("    Two-sided 5% test, 80% power, independent arms. Fill rate = fills per game with a bid (resting: baseline fills / orders).")
    for key, name in GROUPS:
        p = len(hits[key])/len(by_group[key])
        text = f"  {name}: resting fill rate {100*p:4.1f}% of games"
        for points in (10, 20):
            q = p*(1-points/100)
            n = needed(p, q)
            text += f" | just-in-time keeps {100-points}% of fills ({100*q:4.1f}% of games): {n:6,.0f} games per arm = {n*p:5,.0f} resting-arm fills, {n*(p+q):5,.0f} fills in all"
        say(text)
        text = "      if the difference is in fills per game outright"
        for points in (10, 20):
            q = max(p-points/100, .001)
            n = needed(p, q)
            text += f" | {100*p:4.1f}% against {100*q:4.1f}%: {n:5,.0f} games per arm = {n*(p+q):4,.0f} fills in all"
        say(text)
    say("  Paired alternative (both rules judged on the SAME game, as the paper trader's '_jit' shadow order does): among N resting fills, count those the just-in-time bid also got.")
    for p0 in (.90, .80):
        for points in (10, 20):
            p1 = p0-points/100
            n = (1.96*(p0*(1-p0))**.5+.8416*(p1*(1-p1))**.5)**2/(p0-p1)**2
            say(f"    expected kept share {100*p0:.0f}%, true {100*p1:.0f}%: {n:4.0f} resting fills to detect it; a 95% interval of +/-10 points on the kept share needs {1.96**2*p1*(1-p1)/.01:4.0f} fills, +/-5 points {1.96**2*p1*(1-p1)/.0025:4.0f}")
    freq = df.frequency(rows)
    say("  Fills per 30 days at the current frequency (dip_capacity 4b): "+", ".join(f"{key} {freq[key]['fills']:.0f} of {freq[key]['orders']:.0f} orders" for key, _ in GROUPS))
    (OUT/"summary.json").write_text(json.dumps(summary, indent=1, default=float))
    (OUT/"report.txt").write_text(buffer.getvalue())


if __name__ == "__main__":
    report()
