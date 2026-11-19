"""Reconcile "+4c a fill" with "+10c a fill" for the 80c dip bid in first-round tennis qualifying.

Offline unless `download` is run. Reads the listings and one-minute bars other
studies cached (data/raw/cost-research-v1) and replays the rule the way
paper/trader.py `dips` applies it, as closely as minute bars allow:

  * round: paper/rules.py `qualifying(market) == "first"` on the first-listed
    market's rules text, on the four series the trader watches (ATP, WTA, ATP
    Challenger, WTA Challenger). ITF series are not traded.
  * pre-match price: the last QUOTES one-minute closes of the first-listed
    market before the start, each with a two-sided quote at most 4c wide,
    their midpoints within 3c; the median picks the favourite, 90c or more.
    (The trader keeps 15 quotes 20 seconds apart: five minutes.)
  * start: Kalshi's live status is not on record. The start is the bar
    research/favourite_gap.py `detect_start` flags (first 2c move in ten
    minutes, or an open-interest jump). `--shift k` variants place the bid k
    minutes before that bar, to measure what a late detection does.
  * bid: 80c on the favourite, as NO on the underdog's market (the other
    market when the favourite is YES on the first-listed one). Placed at the
    close of the start bar if the favourite's ask there is above 80c.
  * fill: a later minute whose prints on that book go strictly below 80c.
  * payout: the settlement value of the contract held. Matches settled at a
    "fair price" (never played) are kept as orders; they cannot fill unless a
    print went below 80c after the start bar.

    .venv/bin/python research/tennis_reconcile.py download     (public bars of underdog markets not yet cached; 2.5 s apart)
    .venv/bin/python research/tennis_reconcile.py report

Public market data only; no orders, no authentication. Output:
data/processed/tennis-reconcile/.
"""
import argparse
from collections import Counter, defaultdict
from datetime import datetime, timezone
import hashlib
import io
import json
from pathlib import Path
import re
from statistics import median
import sys
import time

import httpx
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import config
from paper import rules
from research.api_cost import COLLECTION, ROOT, save, timestamp
from research.favourite_gap import STUB, TOURS, bar_params, detect_start, listing, side_path

OUT = ROOT / "data/processed/tennis-reconcile"
BARS = OUT / "bars"
SERIES = ("KXATPCHALLENGERMATCH", "KXWTAMATCH", "KXWTACHALLENGERMATCH", "KXATPMATCH")
BID, QUOTES, PACE = 80, 5, 2.5
RNG = np.random.default_rng(23)
buffer = io.StringIO()


def say(text=""):
    print(text, flush=True)
    buffer.write(text+"\n")


def shared(path, params):
    return COLLECTION / (hashlib.sha256(json.dumps([path, params], sort_keys=True).encode()).hexdigest()+".json")


def own(ticker):
    return BARS / (ticker+".json")


def bars_of(path, ticker, settle):
    """Cached one-minute bars of one market over the six hours before `settle`, or None."""
    for file in (shared(path.format(t=ticker), bar_params(settle)), own(ticker)):
        if file.exists():
            return sorted(json.loads(file.read_text()).get("candlesticks") or [], key=lambda b: b["end_period_ts"])
    return None


def events():
    """First-round qualifying events by the trader's own test, archive copy first."""
    out, seen, tally = [], set(), Counter()
    with httpx.Client(transport=STUB) as client:
        for archived in (True, False):
            for series in SERIES:
                found, path, lo, hi = listing(client, series, archived)
                for event, group in found.items():
                    group = sorted(group, key=lambda m: m["ticker"])
                    stage = rules.qualifying(group[0])
                    tally[(TOURS[series], stage or "main draw")] += event not in seen
                    if stage != "first" or event in seen:
                        continue
                    seen.add(event)
                    try:
                        values = [float(m["settlement_value_dollars"]) for m in group]
                        settle = int(timestamp(group[0]["settlement_ts"]))
                    except (KeyError, TypeError, ValueError):
                        tally["no settlement"] += 1
                        continue
                    if len(group) != 2:
                        tally["not two markets"] += 1
                        continue
                    text = group[0].get("rules_primary") or ""
                    label = re.search(r"\bQualif\w*( Round \d+)?", text).group(0)
                    name = re.search(r" in the (?:\d{4} )?(.+?) after a ball", text)
                    out.append(dict(event=event, series=series, tour=TOURS[series], archived=archived, path=path, settle=settle,
                                    tickers=[m["ticker"] for m in group], values=values, label=label,
                                    tournament=re.sub(r"\s*Qualif.*$", "", name.group(1)) if name else event,
                                    team_event="Qualifiers" in label))
    return out, tally


def quote(p):
    """Midpoint of one bar if its quote is usable the way rules.listed wants it."""
    bid, ask = p[1], p[2]
    return (bid+ask)/2 if bid is not None and ask is not None and 0 < bid <= ask < 100 and ask-bid <= rules.MAX_SPREAD else None


def replay(e, a_bars, b_bars, shift=0, book="live", price="live", warm=False):
    """One order of the rule, or a reason there is none.

    book: "live" = NO on the underdog's market; "first" = the first-listed market whichever side the favourite is on.
    price: "live" = the last QUOTES closes before the start; "warm" = the first 15 bars; "bar" = the single bar before the start.
    """
    a = side_path(a_bars, True)
    began = detect_start(a)
    if began is None:
        return "no start seen"
    at = began-shift
    if at < QUOTES:
        return "start too early in the window"
    if warm:                                   # research/tennis_contexts.py also wants the first 15 bars of the window stable
        mids = [quote(p) for p in a[:15]]
        if None in mids or max(mids)-min(mids) > 3:
            return "warm-up quote not stable"
    if price == "live":
        mids = [quote(p) for p in a[at-QUOTES:at]]
        if None in mids or a[at-1][0]-a[at-QUOTES][0] > 600:
            return "pre-match quote wide or missing"
        if max(mids)-min(mids) > rules.MAX_RANGE:
            return "pre-match quote not stable"
        mid = median(mids)
    elif price == "warm":
        mids = [quote(p) for p in a[:15]]
        if None in mids or max(mids)-min(mids) > 3:
            return "pre-match quote not stable"
        mid = median(mids)
    else:
        mid = quote(a[at-1])
        if mid is None:
            return "pre-match quote wide or missing"
    yes, start = (True, mid) if mid > 50 else (False, 100-mid)
    if start < rules.DIP_START:
        return "favourite under 90c"
    # The book the bid rests on, as YES bars of that market; the favourite is NO there unless book == "first" and yes.
    if yes and book == "live":
        if b_bars is None:
            return "underdog market bars not cached"
        target, fav_yes, held = side_path(b_bars, True), False, 1-e["values"][1]
    else:
        target, fav_yes, held = a, yes, (e["values"][0] if yes else 1-e["values"][0])
    path = {p[0]: p for p in target}
    now = path.get(a[at][0])
    ask = None if now is None else (now[2] if fav_yes else (None if not now[1] else 100-now[1]))
    order = dict(event=e["event"], tour=e["tour"], series=e["series"], label=e["label"], tournament=e["tournament"], archived=e["archived"],
                 start_ts=a[at][0], settle=e["settle"], start=start, favourite_first_listed=yes, payout=held,
                 own_market_payout=e["values"][0] if yes else e["values"][1], ask_at_placement=ask, fill_ts=None, touch_ts=None, low=None)
    if ask is None or ask <= BID:
        order["skipped"] = "no offer" if ask is None else "price already at the bid"
        return order
    for p in target:
        if p[0] <= a[at][0]:
            continue
        low = p[3] if fav_yes else (None if p[4] is None else 100-p[4])
        if low is None:
            continue
        order["low"] = low if order["low"] is None else min(order["low"], low)
        if low <= BID and order["touch_ts"] is None:
            order["touch_ts"] = p[0]
        if low < BID and order["fill_ts"] is None:
            order["fill_ts"] = p[0]
    return order


def run(evs, **how):
    orders, why = [], Counter()
    for e in evs:
        if e["a"] is None:
            why["first market bars not cached"] += 1
            continue
        got = replay(e, e["a"], e["b"], **how)
        if isinstance(got, str):
            why[got] += 1
        elif "skipped" in got:
            why[got["skipped"]] += 1
        else:
            orders.append(got)
    return sorted(orders, key=lambda o: (o["settle"], o["event"])), why


# ---------------------------------------------------------------- statistics

def boot(nets, clusters, draws=4000):
    """95% interval of the mean net per fill, resampling clusters."""
    groups = defaultdict(list)
    for n, c in zip(nets, clusters):
        groups[c].append(n)
    sums = np.array([sum(v) for v in groups.values()])
    counts = np.array([len(v) for v in groups.values()])
    if len(sums) < 8:
        return None
    pick = RNG.integers(0, len(sums), (draws, len(sums)))
    return [float(v) for v in np.percentile(sums[pick].sum(axis=1)/counts[pick].sum(axis=1), [2.5, 97.5])]


def score(orders, cluster="event"):
    fills = [o for o in orders if o["fill_ts"]]
    nets = [100*o["payout"]-BID for o in fills]
    out = dict(orders=len(orders), fills=len(fills), touch_only=sum(1 for o in orders if o["touch_ts"] and not o["fill_ts"]))
    if fills:
        out.update(win=float(np.mean([o["payout"] for o in fills])), losses=sum(o["payout"] < .5 for o in fills), net=float(np.mean(nets)),
                   ci=boot(nets, [o[cluster] for o in fills]), per_order=float(sum(nets)/len(orders)))
    return out


def line(name, s):
    if not s.get("fills"):
        return f"  {name:44} orders {s['orders']:4} fills   0"
    ci = "[   n/a    ]" if s["ci"] is None else f"[{s['ci'][0]:+5.1f},{s['ci'][1]:+5.1f}]"
    return (f"  {name:44} orders {s['orders']:4} fills {s['fills']:3} ({s['fills']/s['orders']:3.0%}) won {s['win']:6.1%} lost {s['losses']:2} "
            f"net {s['net']:+5.1f}c {ci} = {s['net']/.8:+5.1f}% | per order {s['per_order']:+5.2f}c")


def month(ts):
    return datetime.fromtimestamp(ts, timezone.utc).strftime("%Y-%m")


def shrink(cells):
    """Normal-normal shrinkage of cell means towards their precision-weighted mean; tau by the method of moments."""
    m, v = np.array([c[1] for c in cells]), np.array([c[2]**2 for c in cells])
    w = 1/v
    centre = float((w*m).sum()/w.sum())
    q = float((w*(m-centre)**2).sum())
    tau2 = max(0.0, (q-(len(m)-1))/(w.sum()-(w**2).sum()/w.sum()))
    out = {}
    for (name, mean, se), var in zip(cells, v):
        # Centre and tau from the OTHER cells would be stricter; shown as well for the cell of interest.
        b = tau2/(tau2+var) if tau2+var else 0.0
        post_sd = (1/(1/var+1/tau2))**.5 if tau2 else 0.0
        out[name] = dict(raw=mean, se=se, shrunk=centre+b*(mean-centre), sd=post_sd, weight=b)
    return centre, tau2**.5, out


# ---------------------------------------------------------------- steps

def load():
    evs, tally = events()
    for e in evs:
        e["a"] = bars_of(e["path"], e["tickers"][0], e["settle"])
        e["b"] = bars_of(e["path"], e["tickers"][1], e["settle"])
    return evs, tally


def download(minutes):
    """Bars of the underdog's market for favourites that are YES on the first-listed market and have no cached second book."""
    evs, _ = load()
    todo = []
    for e in evs:
        if e["a"] is None or e["b"] is not None:
            continue
        for shift in (0, 10, 20, 30):
            if replay(e, e["a"], None, shift=shift) == "underdog market bars not cached":
                todo.append(e)
                break
    print(f"underdog-market bar reads to make: {len(todo)}", flush=True)
    BARS.mkdir(parents=True, exist_ok=True)
    deadline, last, done = time.monotonic()+60*minutes, 0.0, 0
    with httpx.Client(timeout=40, headers={"User-Agent": "research-script"}) as client:
        for e in todo:
            if time.monotonic() > deadline:
                break
            ticker = e["tickers"][1]
            for pause in (30, 60, 120, 240):
                time.sleep(max(0, last+PACE-time.monotonic()))
                try:
                    resp = client.get(config.KALSHI_PROD_BASE+e["path"].format(t=ticker), params=bar_params(e["settle"]))
                except httpx.RequestError:
                    resp = None
                last = time.monotonic()
                if resp is not None and resp.status_code == 200:
                    temp = own(ticker).with_suffix(".tmp")
                    temp.write_text(json.dumps(resp.json()))
                    temp.replace(own(ticker))
                    done += 1
                    break
                if resp is not None and resp.status_code != 429 and resp.status_code < 500:
                    print(f"  {ticker}: HTTP {resp.status_code}", flush=True)
                    break
                print(f"  throttled ({resp.status_code if resp is not None else 'network'}): pausing {pause}s", flush=True)
                time.sleep(pause)
                last = time.monotonic()
    print(f"read {done} of {len(todo)}")


def report():
    evs, tally = load()
    out = {}
    say("TENNIS FIRST-ROUND QUALIFYING, 80c DIP BID: RECONCILIATION")
    say(f"\n1. SAMPLE. Events by the trader's round test (paper/rules.py qualifying), four traded series, archive copy first")
    for tour in ("Challenger", "WTA", "WTA Challenger", "ATP"):
        say(f"  {tour:15} first-round qualifying {tally[(tour, 'first')]:5} | later qualifying {tally[(tour, 'later')]:5} | main draw {tally[(tour, 'main draw')]:5}")
    labels = Counter((e["tour"], e["label"]) for e in evs)
    say("  what the 'first' test matched: "+"; ".join(f"{t} '{l}' {n}" for (t, l), n in sorted(labels.items())))
    say(f"  with first-market bars {sum(e['a'] is not None for e in evs)} of {len(evs)}; with underdog-market bars as well {sum(e['a'] is not None and e['b'] is not None for e in evs)}")
    say(f"  settled between {month(min(e['settle'] for e in evs))} and {month(max(e['settle'] for e in evs))}; "
        f"fractional settlement (never played, 'fair price') {sum(e['values'][0] not in (0.0, 1.0) for e in evs)}")

    say("\n2. THE RULE REPLAYED UNDER EACH DEFINITION (strict fills; net per fill on the 80c paid; interval = bootstrap over matches)")
    variants = (
        ("A0 jit_bids / dip_capacity order set rebuilt", dict(book="first", price="bar", warm=True), lambda o: o["start"] < 99 and not o["team"]),
        ("A  = A0 without the stable warm-up hours before", dict(book="first", price="bar"), lambda o: o["start"] < 99 and not o["team"]),
        ("B  = A, 5 stable quotes instead of 1 bar", dict(book="first", price="live"), lambda o: o["start"] < 99 and not o["team"]),
        ("C  = B, bid on the underdog's market (LIVE)", dict(book="live", price="live"), lambda o: o["start"] < 99 and not o["team"]),
        ("D  = C, no 99c cap, team events in (as coded)", dict(book="live", price="live"), lambda o: True),
        ("E  = C, bid 10 min before the detected start", dict(book="live", price="live", shift=10), lambda o: o["start"] < 99 and not o["team"]),
        ("F  = C, bid 20 min before", dict(book="live", price="live", shift=20), lambda o: o["start"] < 99 and not o["team"]),
        ("G  = C, bid 30 min before", dict(book="live", price="live", shift=30), lambda o: o["start"] < 99 and not o["team"]),
        ("H  warm-up price (hours before), first book", dict(book="first", price="warm"), lambda o: o["start"] < 99 and not o["team"]),
    )
    team = {e["event"] for e in evs if e["team_event"]}
    runs = {}
    for name, how, keep in variants:
        orders, why = run(evs, **how)
        for o in orders:
            o["team"] = o["event"] in team
        orders = [o for o in orders if keep(o)]
        runs[name.split()[0]] = (orders, why)
        out[f"2 | {name}"] = dict(score(orders), dropped=dict(why))
        say(line(name, score(orders)))
    say("  dropped before an order, definition C: "+", ".join(f"{k} {v}" for k, v in runs["C"][1].most_common()))
    live = runs["C"][0]

    say("\n3. WHICH BOOK (definition B/C orders, same matches)")
    first = {o["event"]: o for o in runs["B"][0]}
    both = [(first[o["event"]], o) for o in live if o["event"] in first]
    for label, test in (("favourite is NO on the first-listed market (same book either way)", lambda f: not f["favourite_first_listed"]),
                        ("favourite is YES on the first-listed market", lambda f: f["favourite_first_listed"])):
        part = [(f, o) for f, o in both if test(f)]
        say(f"  {label}: {len(part)} matches")
        say(line("    bid on the first-listed (favourite's) book", score([f for f, _ in part])))
        say(line("    bid on the underdog's book as NO (live)", score([o for _, o in part])))
        only_u = [o for f, o in part if o["fill_ts"] and not f["fill_ts"]]
        only_f = [f for f, o in part if f["fill_ts"] and not o["fill_ts"]]
        say(f"    filled on the underdog's book only {len(only_u)} (won {sum(o['payout'] for o in only_u):.0f}); on the favourite's book only {len(only_f)} (won {sum(f['payout'] for f in only_f):.0f})")
    say(f"  one order per match everywhere: {len(live)} orders, {len({o['event'] for o in live})} distinct matches")

    say("\n4. THE LIVE DEFINITION (C) IN DETAIL")
    s = score(live)
    out["4 | all"] = s
    say(line("all", s))
    say(line("  interval resampling tournaments instead", score(live, "tournament")))
    half = len(live)//2
    for name, part in (("first half (by settlement)", live[:half]), ("second half", live[half:]),
                       ("archive (settled before 2026-09-18)", [o for o in live if o["archived"]]), ("recent", [o for o in live if not o["archived"]])):
        out[f"4 | {name}"] = score(part)
        say(line(name, score(part)))
    say("  by tour")
    for tour in ("Challenger", "WTA", "WTA Challenger", "ATP"):
        out[f"4 | tour | {tour}"] = score([o for o in live if o["tour"] == tour])
        say(line("  "+tour, out[f"4 | tour | {tour}"]))
    say("  by pre-match price")
    for lo, hi in ((90, 93), (93, 96), (96, 99)):
        say(line(f"  {lo}-{hi}c", score([o for o in live if lo <= o["start"] < hi])))
    say("  by month of the start (UTC). Listings: every tour from 2026-04-11; recent bars end 2026-11-16")
    months = sorted({month(o["start_ts"]) for o in live})
    for m in months:
        part = [o for o in live if month(o["start_ts"]) == m]
        out[f"4 | month | {m}"] = dict(score(part), by_tour=dict(Counter(o["tour"] for o in part)))
        say(line("  "+m, score(part))+"  "+" ".join(f"{t[:3]}{'+' if t.endswith('Challenger') and t != 'Challenger' else ''} {n}" for t, n in sorted(Counter(o["tour"] for o in part).items())))
    full = [m for m in months if "2026-05" <= m <= "2026-10"]
    per = [sum(1 for o in live if month(o["start_ts"]) == m and o["fill_ts"]) for m in full]
    orders_per = [sum(1 for o in live if month(o["start_ts"]) == m) for m in full]
    say(f"  complete months May-October: fills a month {per} (mean {np.mean(per):.1f}), orders a month {orders_per} (mean {np.mean(orders_per):.1f})")
    for name, lo, hi in (("April-October", "2026-04", "2026-10"), ("May-October", "2026-05", "2026-10"), ("May-July", "2026-05", "2026-07")):
        part = [o for o in live if lo <= month(o["start_ts"]) <= hi]
        rest = [o for o in live if not lo <= month(o["start_ts"]) <= hi]
        out[f"4 | season | {name}"] = dict(inside=score(part), outside=score(rest))
        say(line("  "+name, score(part)))
        say(line("    every other month", score(rest)))
    from scipy.stats import fisher_exact
    inside = [o for o in live if o["fill_ts"] and "2026-05" <= month(o["start_ts"]) <= "2026-07"]
    outside = [o for o in live if o["fill_ts"] and not "2026-05" <= month(o["start_ts"]) <= "2026-07"]
    lost = lambda xs: sum(o["payout"] < .5 for o in xs)
    p = fisher_exact([[lost(inside), len(inside)-lost(inside)], [lost(outside), len(outside)-lost(outside)]])[1]
    say(f"  May-July against the rest, losses among fills {lost(inside)}/{len(inside)} v {lost(outside)}/{len(outside)}: Fisher p {p:.4f} for a window chosen after looking "
        f"(any 3 consecutive months of 11: about 9 windows, so roughly p {min(1, 9*p):.2f})")
    old = runs["A0"][0]
    say("  the jit_bids order set (A0) by month, orders/fills: "+"  ".join(
        f"{m[5:]} {sum(1 for o in old if month(o['start_ts']) == m)}/{sum(1 for o in old if month(o['start_ts']) == m and o['fill_ts'])}" for m in months))
    fills = [o for o in live if o["fill_ts"]]
    say(f"  touch-only orders (a print at 80c, none below): {s['touch_only']}, won {sum(o['payout'] for o in live if o['touch_ts'] and not o['fill_ts']):.0f}")
    say(f"  fractional payouts among orders: {sum(o['payout'] not in (0.0, 1.0) for o in live)}; among fills: "
        f"{[(o['event'], round(o['payout'], 2)) for o in fills if o['payout'] not in (0.0, 1.0)]}")
    say(f"  minutes from the bid to the fill: "+" / ".join(f"{v:.0f}" for v in np.percentile([(o['fill_ts']-o['start_ts'])/60 for o in fills], [5, 25, 50, 75, 95]))+" (5/25/50/75/95%)")
    say(f"  ask when the bid was placed: "+" / ".join(f"{v:.0f}" for v in np.percentile([o['ask_at_placement'] for o in live], [1, 5, 25, 50]))+" (1/5/25/50%)")

    say("\n5. ARITHMETIC: the pre-match gap and the net per fill are one edge")
    won = float(np.mean([o["own_market_payout"] for o in live]))
    price = float(np.mean([o["start"] for o in live]))
    losses = sum(o["own_market_payout"] < .5 for o in live)
    lost_unfilled = sum(o["own_market_payout"] < .5 and not o["fill_ts"] for o in live)
    fair_losses = len(live)*(1-price/100)
    gap_ci = boot([100*o["own_market_payout"]-o["start"] for o in live], [o["event"] for o in live])
    out["5"] = dict(orders=len(live), mean_price=price, won=won, gap=100*won-price, gap_ci=gap_ci, losses=losses, fair_losses=fair_losses)
    say(f"  {len(live)} favourites, mean pre-match price {price:.1f}c, won {won:.1%}: gap {100*won-price:+.2f}c per FAVOURITE [{gap_ci[0]:+.2f},{gap_ci[1]:+.2f}]")
    say(f"  losses {losses} (a fair price implies {fair_losses:.1f}); losses that never filled the bid {lost_unfilled}")
    say(f"  dip bid: {s['fills']} fills x {s['net']:+.1f}c = {s['fills']*s['net']:+.0f}c over {len(live)} orders = {s['per_order']:+.2f}c per ORDER")
    fair = 100*(1-fair_losses/s["fills"])-BID
    say(f"  the same {s['fills']} fills with the losses a fair price implies ({fair_losses:.1f}): net {fair:+.1f}c per fill. "
        f"Each loss avoided is worth {100/s['fills']:.2f}c per fill.")

    say("\n6. SHRUNK, FORWARD-LOOKING. Same replay (definition B: first book, so no second book is needed) on every round of the four series")
    cells = rungs()
    centre, tau, post = shrink([(k, v["net"], v["se"]) for k, v in cells.items() if v["fills"] >= 20])
    for k, v in cells.items():
        p = post.get(k)
        say(f"  {k:36} orders {v['orders']:5} fills {v['fills']:4} won {v['win']:6.1%} net {v['net']:+5.1f}c (se {v['se']:.1f})"
            + (f" -> shrunk {p['shrunk']:+5.1f}c (weight on its own data {p['weight']:.0%})" if p else ""))
    say(f"  precision-weighted centre {centre:+.1f}c, between-cell sd {tau:.1f}c")
    key = "qualifying, first round"
    others = [(k, v["net"], v["se"]) for k, v in cells.items() if v["fills"] >= 20 and k != key]
    w = np.array([1/c[2]**2 for c in others])
    c0 = float((w*np.array([c[1] for c in others])).sum()/w.sum())
    raw, se = s["net"], (s["ci"][1]-s["ci"][0])/3.92
    table = {}
    for t in (tau, 3.0, 4.3, 6.0):
        b = t**2/(t**2+se**2) if t else 0.0
        mean, sd = c0+b*(raw-c0), (b*se**2)**.5
        table[round(t, 1)] = dict(mean=mean, low=mean-1.96*sd, high=mean+1.96*sd)
        say(f"  live figure {raw:+.1f}c (se {se:.1f}) shrunk towards the OTHER rounds' mean {c0:+.1f}c with between-cell sd {t:.1f}c: "
            f"{mean:+.1f}c [{mean-1.96*sd:+.1f},{mean+1.96*sd:+.1f}] = {mean/.8:+.1f}%")
    out["6"] = dict(cells=cells, centre=centre, tau=tau, others_mean=c0, raw=raw, se=se, shrunk=table)

    save(OUT/"summary.json", out)
    save(OUT/"orders.json", live)
    (OUT/"report.txt").write_text(buffer.getvalue())


def rungs():
    """The 80c bid on 90-99c favourites by round, four series, definition B (first-listed book, five stable quotes)."""
    cells = defaultdict(list)
    with httpx.Client(transport=STUB) as client:
        seen = set()
        for archived in (True, False):
            for series in SERIES:
                found, path, lo, hi = listing(client, series, archived)
                for event, group in found.items():
                    group = sorted(group, key=lambda m: m["ticker"])
                    if event in seen or len(group) != 2:
                        continue
                    seen.add(event)
                    try:
                        values = [float(m["settlement_value_dollars"]) for m in group]
                        settle = int(timestamp(group[0]["settlement_ts"]))
                    except (KeyError, TypeError, ValueError):
                        continue
                    text = group[0].get("rules_primary") or ""
                    stage = rules.qualifying(group[0])
                    if "Qualifiers" in text:
                        name = "team event (Billie Jean King Cup)"
                    elif stage:
                        name = f"qualifying, {stage} round" if stage == "first" else "qualifying, later rounds"
                    elif re.search(r"(Quarter-?[Ff]inals?|Semi-?[Ff]inals?|Finals?|1/[24][- ][Ff]inals?) after a ball", text):
                        name = "main draw, quarterfinal on"
                    else:
                        name = "main draw, before the quarterfinal"
                    a = bars_of(path, group[0]["ticker"], settle)
                    if a is None:
                        continue
                    e = dict(event=event, series=series, tour=TOURS[series], archived=archived, settle=settle, values=values, label="", tournament="")
                    got = replay(e, a, None, book="first", price="live")
                    if isinstance(got, dict) and "skipped" not in got and got["start"] < 99:
                        cells[name].append(got)
    out = {}
    for name, orders in sorted(cells.items()):
        fills = [o for o in orders if o["fill_ts"]]
        nets = np.array([100*o["payout"]-BID for o in fills])
        if len(fills) > 1:
            out[name] = dict(orders=len(orders), fills=len(fills), win=float(np.mean([o["payout"] for o in fills])),
                             net=float(nets.mean()), se=float(nets.std(ddof=1)/len(nets)**.5))
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("step", choices=("download", "report"))
    parser.add_argument("--minutes", type=float, default=8)
    args = parser.parse_args()
    if args.step == "download":
        download(args.minutes)
    else:
        report()


if __name__ == "__main__":
    main()
