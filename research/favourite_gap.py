"""The pre-match gap on heavy tennis favourites: is it stable, what predicts it, and how best to buy it.

Background (research/dip_flow.py, research/edge_reality.py): on some Kalshi
tennis tours a player quoted at 90c or more before the match wins more often
than the price says. This script asks four things about that gap.

  build   Replay the cached one-minute bars (six hours before settlement, the
          alphabetically-first market of each event) for ATP, ATP Challenger,
          WTA, WTA Challenger, ITF men and ITF women, both periods, and write
          one row per match with a stable pre-match price of 70c or more:
          the favourite, its price, the settlement value, the detected start
          of play (as in research/dip_filters.py), features known before play,
          the tournament and round (parsed from the market's rules text, which
          is already in the cached listing), and the simulated pre-match bids.
          No network: the HTTP client is a stub that answers 404.
  q3      Gap (settlement value minus pre-match price) for 90c+ favourites by
          month and quarter per tour, a trend test, price bands, and 80-90c
          favourites as a control.
  q4      Which pre-match features separate large-gap from small-gap matches,
          in BOTH periods. Cuts are holdout terciles applied unchanged.
  q5      A bid resting before the match: at the bid, and 1c, 2c, 3c below the
          midpoint, placed at the end of the 15-bar warm-up, filled when a later
          pre-match trade prints strictly below it ("touch": at or below),
          cancelled when play is detected, held to settlement. Compared with
          buying at the ask at the same moment and paying the taker fee. Also
          the combined plan with the 80c dip bid.
  q2      The other book. Downloads the one-minute bars of the second market of
          each 90c+ Challenger / WTA / WTA Challenger match (one request at a
          time, one per second, long pause on HTTP 429) and compares the two
          books before play and at the dip.

Conventions. Prices are the favourite's, in cents. The pre-match price `start`
is the median midpoint of the first 15 bars (favorite_dip.py). Holdout is
settled before 2026-09-18, recent is after; a match listed in both is kept in
the holdout only. Fractional settlements count as fractional wins. A resting
bid in the detection bar itself still counts as resting (the start is known
only once that bar has closed). Taker fee 0.07*p*(1-p); maker fee zero on the
Challenger series and, as an assumption, 0.0175*p*(1-p) elsewhere (shown with
and without). Intervals are 95% bootstrap intervals over matches.

Limits. The cached window opens six hours before settlement, so a pre-match
bid is simulated for the few hours before play only, not from listing. Minute
bars carry the low and high print of the minute but not volume by price, so
"filled" means one contract would have filled.

    .venv/bin/python research/favourite_gap.py build q3 q4 q5
    .venv/bin/python research/favourite_gap.py q2 --minutes 40

Public market data only; no orders, no authentication.
"""
import argparse
from collections import defaultdict
from datetime import date, datetime, timedelta, timezone
import json
import math
from pathlib import Path
import re
from statistics import median
import sys
import time

import httpx
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from research.api_cost import ROOT, fetch, midnight, pages, save, timestamp
from research.dip_archive_tours import cached, paced
from research.favorite_dip import END, START, WARMUP
from research.near_end import WINDOW, cents

OUT = ROOT / "data/processed/favourite-gap"
CUTOFF = midnight(date(2026, 9, 18))
TOURS = {"KXATPCHALLENGERMATCH": "Challenger", "KXWTAMATCH": "WTA", "KXWTACHALLENGERMATCH": "WTA Challenger",
         "KXATPMATCH": "ATP", "KXITFMATCH": "ITF men", "KXITFWMATCH": "ITF women"}
EDGE = ("Challenger", "WTA", "WTA Challenger")          # the tours the dip rule trades
NO_MAKER_FEE = ("Challenger", "WTA Challenger")
LEVELS = ("at bid", "mid-1", "mid-2", "mid-3")
DIP = 80
RNG = np.random.default_rng(9)
STUB = httpx.MockTransport(lambda request: httpx.Response(404))


def number(bar, key):
    value = bar.get(key, bar.get(key+"_fp"))      # recent bars carry a "_fp" suffix
    return float(value) if value is not None else 0.0


def taker_fee(price):
    return 7*price/100*(1-price/100)


def maker_fee(price):
    return 1.75*price/100*(1-price/100)


def ci(values, draws=4000):
    values = np.asarray(values, float)
    if len(values) < 10:
        return None
    means = values[RNG.integers(0, len(values), (draws, len(values)))].mean(axis=1)
    return [round(float(v), 2) for v in np.percentile(means, [2.5, 97.5])]


def fmt(band):
    return "      n/a      " if band is None else f"[{band[0]:+6.2f},{band[1]:+6.2f}]"


# ---------------------------------------------------------------- build

def listing(client, series, archived):
    """Events of one series from the cached listing: event -> its markets, plus the candlestick path and settle range."""
    events = defaultdict(list)
    if archived:
        rows = pages(client, "/historical/markets", "markets", series_ticker=series, limit=1000)
        path, lo, hi = "/historical/markets/{t}/candlesticks", midnight(date(2026, 1, 1)), CUTOFF
    else:
        rows = pages(client, "/markets", "markets", series_ticker=series, status="settled", limit=1000,
                     min_close_ts=midnight(START), max_close_ts=midnight(END+timedelta(days=7)))
        path, lo, hi = "/series/"+series+"/markets/{t}/candlesticks", midnight(START), midnight(END)
    try:
        for m in rows:
            events[m["event_ticker"]].append(m)
    except RuntimeError as error:
        print(f"  {series} archived={archived}: listing not fully cached ({error}); using {len(events)} events")
    return events, path, lo, hi


def bar_params(settle):
    return dict(start_ts=settle-WINDOW, end_ts=settle, period_interval=1)


def side_path(bars, yes):
    """Per bar, on the chosen side: time, bid, ask, low print, high print, volume, open interest."""
    out = []
    for b in bars:
        bid, ask = cents(b, "yes_bid", "close"), cents(b, "yes_ask", "close")
        low, high = cents(b, "price", "low"), cents(b, "price", "high")
        if not yes:
            bid, ask = (None if ask is None else 100-ask), (None if bid is None else 100-bid)
            low, high = (None if high is None else 100-high), (None if low is None else 100-low)
        out.append((b["end_period_ts"], bid, ask, low, high, number(b, "volume"), number(b, "open_interest")))
    return out


def mid(p):
    return None if p[1] is None or p[2] is None else (p[1]+p[2])/2


def detect_start(path):
    """First bar after the warm-up where the midpoint moved more than 2c in ten minutes, or open interest jumped."""
    for i in range(WARMUP, len(path)):
        j = next((j for j in range(i, -1, -1) if path[j][0] <= path[i][0]-600), None)
        if j is None or mid(path[i]) is None or mid(path[j]) is None:
            continue
        if abs(mid(path[i])-mid(path[j])) > 2 or (path[i][6]-path[j][6] >= 2000 and path[i][6] >= 1.5*path[j][6]):
            return i
    return None


def parse_rules(text):
    """Tournament and round from 'in the 2026 ATP Challenger Tiburon Round Of 16 after a ball has been played'."""
    found = re.search(r" in the (?:\d{4} )?(.+?) after a ball", text or "")
    if not found:
        return None, None
    name = found.group(1).strip()
    stage = re.search(r"\b(Round [Oo]f \d+|Quarter-?[Ff]inals?|Semi-?[Ff]inals?|Finals?|Qualif\w+(?: Round \d+)?|"
                      r"(?:First|Second|Third|Fourth|1st|2nd|3rd|4th) Round|Round \d+|Group \w+|Round Robin)\s*$", name)
    if not stage:
        return name, None
    return name[:stage.start()].strip(" ,-:"), stage.group(1).title()


def study(event, group, settle, outcome, bars):
    """One match: favourite, pre-match price, detected start, pre-play features, simulated pre-match bids."""
    first = min(group, key=lambda m: m["ticker"])
    mids = []
    for b in bars[:WARMUP]:
        bid, ask = cents(b, "yes_bid", "close"), cents(b, "yes_ask", "close")
        if bid is None or ask is None or not 0 < bid <= ask < 100 or ask-bid > 4:
            return None
        mids.append((bid+ask)/2)
    if len(mids) < WARMUP or max(mids)-min(mids) > 3:
        return None
    yes = median(mids) > 50
    start = median(mids) if yes else 100-median(mids)
    if start < 70:
        return None
    path = side_path(bars, yes)
    began = detect_start(path)
    place = WARMUP-1                                   # the order goes in on the close of the last warm-up bar
    bid0, ask0 = path[place][1], path[place][2]
    mid0 = (bid0+ask0)/2
    tournament, stage = parse_rules(first.get("rules_primary"))
    other = [m["ticker"] for m in group if m["ticker"] != first["ticker"]]
    row = dict(event=event, series=first["ticker"].split("-")[0], tour=TOURS[first["ticker"].split("-")[0]],
               ticker=first["ticker"], other_ticker=other[0] if len(other) == 1 else None, settle=settle,
               period="holdout" if settle < CUTOFF else "recent", yes=yes, start=start, payout=outcome if yes else 1-outcome,
               bid0=bid0, ask0=ask0, tournament=tournament, round=stage,
               match_volume=sum(float(m.get("volume_fp") or m.get("volume") or 0) for m in group),
               detected=began is not None, bars=len(path))
    # The pre-match bids rest from the bar after placement to the detection bar inclusive (or 4 hours if none is seen).
    stop = began if began is not None else min(len(path)-1, place+240)
    rest = path[place+1:stop+1]
    lows = [(i, p[3]) for i, p in enumerate(rest, place+1) if p[3] is not None]
    quiet = path[place+1:began] if began is not None else rest
    row.update(rest_minutes=len(rest), prematch_volume=sum(p[5] for p in quiet), prematch_trade_minutes=sum(p[5] > 0 for p in quiet),
               prematch_low=min((low for _, low in lows), default=None),
               prematch_low_before_detection=min((low for i, low in lows if began is None or i < began), default=None))
    levels = {"at bid": bid0, "mid-1": math.floor(mid0-1), "mid-2": math.floor(mid0-2), "mid-3": math.floor(mid0-3)}
    row["bids"] = {}
    for name, level in levels.items():
        strict = next((i for i, low in lows if low < level), None)
        touch = next((i for i, low in lows if low <= level), None)
        row["bids"][name] = dict(level=level, strict=strict is not None, touch=touch is not None,
                                 strict_in_detection_bar=strict is not None and strict == began,
                                 minutes_to_fill=None if strict is None else strict-place)
    # The 80c dip bid rests from the bar after the detected start (dip_filters.py), or from the warm-up if none is seen.
    after = [p[3] for p in path[(began+1 if began is not None else WARMUP):] if p[3] is not None]
    row["low_in_play"] = min(after) if after else None
    row["dip_fill"] = bool(after) and min(after) < DIP and start >= 90
    if began is None:
        row["features"] = None
        return row
    last = began-1
    now = next((mid(path[j]) for j in range(last, -1, -1) if mid(path[j]) is not None), None)
    hour_back = [p for p in path[:last+1] if p[0] > path[last][0]-3600]
    quoted = [p[2]-p[1] for p in hour_back if mid(p) is not None]
    when = datetime.fromtimestamp(path[began][0], timezone.utc)
    opened = first.get("open_time")
    row["features"] = dict(
        price_before_play=now, drift_since_warmup=now-start,
        spread_warmup=float(np.mean([p[2]-p[1] for p in path[:WARMUP]])),
        spread_60m=float(np.mean(quoted)) if quoted else None,
        volume_60m=sum(p[5] for p in hour_back), volume_warmup=sum(p[5] for p in path[:WARMUP]),
        open_interest=path[last][6], hour_utc=when.hour, weekday=when.weekday(),
        hours_listed=None if not opened else (path[began][0]-timestamp(opened))/3600,
        minutes_before_settlement=(settle-path[began][0])/60)   # known only afterwards: a check, never a feature
    return row


def build():
    rows, seen, counts = [], set(), defaultdict(lambda: [0, 0])
    with httpx.Client(transport=STUB) as client:
        for archived in (True, False):               # archive first, so a match listed twice keeps its archive copy
            for series in TOURS:
                events, path, lo, hi = listing(client, series, archived)
                for event, group in events.items():
                    first = min(group, key=lambda m: m["ticker"])
                    try:
                        outcome, settle = float(first["settlement_value_dollars"]), int(timestamp(first["settlement_ts"]))
                    except (KeyError, TypeError, ValueError):
                        continue
                    if not lo <= settle < hi or event in seen:
                        continue
                    key = (TOURS[series], "holdout" if archived else "recent")
                    counts[key][0] += 1
                    params = bar_params(settle)
                    if not cached(path.format(t=first["ticker"]), params):
                        continue
                    candles = fetch(client, path.format(t=first["ticker"]), params).get("candlesticks")
                    if not candles:
                        continue
                    counts[key][1] += 1
                    row = study(event, group, settle, outcome, sorted(candles, key=lambda b: b["end_period_ts"]))
                    if row:
                        seen.add(event)
                        row["archived"] = archived
                        rows.append(row)
    for (tour, period), (listed, with_bars) in sorted(counts.items()):
        kept = [r for r in rows if r["tour"] == tour and r["archived"] == (period == "holdout")]
        print(f"  {tour:15} {period:8} listed {listed:6} with cached bars {with_bars:6} stable 70c+ {len(kept):5} "
              f"90c+ {sum(r['start'] >= 90 for r in kept):4}")
    save(OUT/"rows.json", rows)
    return rows


def load():
    return json.loads((OUT/"rows.json").read_text())



# ---------------------------------------------------------------- shared helpers

def heavy(rows, lo=90, hi=101):
    return [r for r in rows if lo <= r["start"] < hi]


def gap(rows):
    """Settlement value minus pre-match price, in cents, for a set of matches."""
    if not rows:
        return dict(matches=0)
    res = [100*r["payout"]-r["start"] for r in rows]
    return dict(matches=len(rows), price=float(np.mean([r["start"] for r in rows])), won=float(np.mean([r["payout"] for r in rows])),
                gap=float(np.mean(res)), ci95=ci(res))


def line(name, g):
    if not g["matches"]:
        return f"  {name:34} none"
    return f'  {name:34} n {g["matches"]:5} price {g["price"]:5.1f} won {100*g["won"]:5.1f}% gap {g["gap"]:+6.2f} {fmt(g.get("ci95"))}'


def groups(rows):
    """Each tour, the three traded tours together, and the rest together."""
    out = {tour: [r for r in rows if r["tour"] == tour] for tour in TOURS.values()}
    out["Chall+WTA+WTA Chall"] = [r for r in rows if r["tour"] in EDGE]
    out["ATP+ITF men+ITF women"] = [r for r in rows if r["tour"] not in EDGE]
    return out


# ---------------------------------------------------------------- Q3: stability

def trend(rows):
    """Slope of the per-match gap against time (cents per month), a bootstrap interval, and a permutation p-value."""
    if len(rows) < 40:
        return None
    x = np.array([r["settle"] for r in rows], float)/(30.44*86400)
    y = np.array([100*r["payout"]-r["start"] for r in rows])
    x = x-x.mean()
    slope = float((x*y).sum()/(x*x).sum())
    pick = RNG.integers(0, len(x), (4000, len(x)))
    xs, ys = x[pick], y[pick]
    xs = xs-xs.mean(axis=1, keepdims=True)
    boot = (xs*ys).sum(axis=1)/(xs*xs).sum(axis=1)
    perm = np.array([(x*RNG.permutation(y)).sum()/(x*x).sum() for _ in range(4000)])
    return dict(matches=len(rows), months_spanned=float(x.max()-x.min()), slope_cents_per_month=slope,
                ci95=[round(float(v), 2) for v in np.percentile(boot, [2.5, 97.5])],
                p_two_sided=float((np.abs(perm) >= abs(slope)).mean()))


def q3(rows):
    report = {}
    top = heavy(rows)
    month = lambda r: datetime.fromtimestamp(r["settle"], timezone.utc).strftime("%Y-%m")
    quarter = lambda r: f'Q{(datetime.fromtimestamp(r["settle"], timezone.utc).month-1)//3+1}'
    print("\n(Q3) IS THE GAP STABLE? 90c+ favourites, gap = settlement value minus pre-match price (cents)")
    for name, part in groups(top).items():
        print(f" {name}")
        print(line("all", gap(part)))
        report[f"{name} | all"] = gap(part)
        for label, key in (("quarter", quarter), ("month", month)):
            for value in sorted({key(r) for r in part}):
                sub = [r for r in part if key(r) == value]
                g = gap(sub)
                g["median_match_volume"] = float(np.median([r["match_volume"] for r in sub]))
                report[f"{name} | {label} {value}"] = g
                print(line(f"{label} {value}", g)+f'  median match volume {g["median_match_volume"]:9.0f}')
        t = trend(part)
        report[f"{name} | trend"] = t
        if t:
            print(f'  trend {t["slope_cents_per_month"]:+.2f}c per month {fmt(t["ci95"])} permutation p {t["p_two_sided"]:.2f} '
                  f'over {t["months_spanned"]:.1f} months')
        for period in ("holdout", "recent"):
            g = gap([r for r in part if r["period"] == period])
            report[f"{name} | {period}"] = g
            print(line(period, g))
    print(" price bands (pre-match price, cents)")
    bands = (("90-92", 90, 93), ("93-95", 93, 96), ("96-98", 96, 99), ("99", 99, 101), ("control 80-84", 80, 85), ("control 85-89", 85, 90),
             ("control 80-89", 80, 90), ("control 70-79", 70, 80))
    for name, part in groups(rows).items():
        print(f" {name}")
        for label, lo, hi in bands:
            for period in ("holdout", "recent", "both"):
                sub = [r for r in heavy(part, lo, hi) if period == "both" or r["period"] == period]
                report[f"{name} | band {label} | {period}"] = gap(sub)
                print(line(f"{label} {period}", gap(sub)))
    # Does the gap shrink where there is more money? Open interest before play, cut at the holdout terciles of the traded tours.
    print(" gap against the money in the match before play (open interest, traded tours, 90c+)")
    have = [r for r in top if r["tour"] in EDGE and r["features"]]
    cuts = np.percentile([r["features"]["open_interest"] for r in have if r["period"] == "holdout"], [100/3, 200/3])
    for period in ("holdout", "recent"):
        for label, lo, hi in (("low", -1, cuts[0]), ("mid", cuts[0], cuts[1]), ("high", cuts[1], 1e18)):
            sub = [r for r in have if r["period"] == period and lo < r["features"]["open_interest"] <= hi]
            report[f"open interest {label} | {period}"] = gap(sub)
            print(line(f"open interest {label} {period}", gap(sub)))
    report["open interest cuts"] = [float(c) for c in cuts]
    save(OUT/"q3-stability.json", report)


# ---------------------------------------------------------------- Q4: what predicts the gap

NUMERIC = ("open_interest", "volume_60m", "volume_warmup", "spread_warmup", "spread_60m", "hours_listed", "drift_since_warmup",
           "price_before_play")


def tier(r):
    name = r["tournament"] or ""
    found = re.match(r"([MW])(\d+)\b", name)
    if found:
        return found.group(1)+found.group(2)
    if re.search(r"Open (Men|Women) Singles|Wimbledon", name):
        return "Grand Slam"
    return "tour event"


def stage(r):
    text = r["round"] or ""
    if text.startswith("Qualif"):
        return "qualifying"
    if text in ("Quarterfinal", "Quarterfinals", "Semifinal", "Semifinals", "Final", "Finals"):
        return "quarterfinal on"
    return "early main draw"


CATEGORICAL = {
    "favourite is the first-listed (YES) market": lambda r: "yes side" if r["yes"] else "no side",
    "round": stage,
    "tournament tier": tier,
    "weekend": lambda r: "Sat/Sun" if r["features"]["weekday"] >= 5 else "Mon-Fri",
    "weekday": lambda r: ("Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun")[r["features"]["weekday"]],
    "hour of start (UTC)": lambda r: ("00-05", "06-11", "12-17", "18-23")[r["features"]["hour_utc"]//6],
}


def q4(rows):
    report, tried, consistent = {}, 0, []
    top = [r for r in heavy(rows) if r["features"]]
    pools = {"Chall+WTA+WTA Chall": [r for r in top if r["tour"] in EDGE], "all six tours": top}
    print("\n(Q4) WHAT PREDICTS THE GAP? 90c+ favourites with a detected start; gap in cents; cuts are holdout terciles")
    for pool, part in pools.items():
        print(f" {pool}: holdout {sum(r['period'] == 'holdout' for r in part)}, recent {sum(r['period'] == 'recent' for r in part)}")
        for name in NUMERIC:
            tried += 1
            have = [r for r in part if r["features"][name] is not None]
            lo, hi = np.percentile([r["features"][name] for r in have if r["period"] == "holdout"], [100/3, 200/3])
            print(f"  {name}: cuts {lo:.2f} and {hi:.2f}")
            signs = []
            for period in ("holdout", "recent"):
                sub = [r for r in have if r["period"] == period]
                parts = {"low": [r for r in sub if r["features"][name] <= lo], "mid": [r for r in sub if lo < r["features"][name] <= hi],
                         "high": [r for r in sub if r["features"][name] > hi]}
                for k, rs in parts.items():
                    report[f"{pool} | {name} | {period} | {k}"] = gap(rs)
                    print(line(f"  {period} {k}", gap(rs)))
                x = np.argsort(np.argsort([r["features"][name] for r in sub])).astype(float)
                y = np.array([100*r["payout"]-r["start"] for r in sub])
                rho, band = float(np.corrcoef(x, y)[0, 1]), 2/len(sub)**.5
                report[f"{pool} | {name} | {period} | rank correlation"] = dict(matches=len(sub), rho=rho, noise_band=band)
                print(f"    {period} rank correlation with the gap {rho:+.3f} (noise band +/-{band:.3f})")
                signs.append((rho, band))
            if signs[0][0]*signs[1][0] > 0 and all(abs(rho) > band for rho, band in signs):
                consistent.append(f"{pool} | {name}")
        for name, key in CATEGORICAL.items():
            tried += 1
            print(f"  {name}")
            spread = {}
            for period in ("holdout", "recent"):
                sub = [r for r in part if r["period"] == period]
                for value in sorted({key(r) for r in sub}):
                    g = gap([r for r in sub if key(r) == value])
                    report[f"{pool} | {name} | {period} | {value}"] = g
                    spread.setdefault(value, {})[period] = g
                    print(line(f"  {period} {value}", g))
            # A level counts as consistent when its interval excludes the pooled gap of the other levels on the same side in both periods.
            for value, both in spread.items():
                if len(both) < 2 or any(g["ci95"] is None for g in both.values()):
                    continue
                diffs = []
                for period, g in both.items():
                    others = [r for r in part if r["period"] == period and key(r) != value]
                    mine = [r for r in part if r["period"] == period and key(r) == value]
                    a = np.array([100*r["payout"]-r["start"] for r in mine])
                    b = np.array([100*r["payout"]-r["start"] for r in others])
                    if len(a) < 20 or len(b) < 20:
                        break
                    boot = a[RNG.integers(0, len(a), (4000, len(a)))].mean(axis=1)-b[RNG.integers(0, len(b), (4000, len(b)))].mean(axis=1)
                    diffs.append((float(a.mean()-b.mean()), [round(float(v), 2) for v in np.percentile(boot, [2.5, 97.5])]))
                if len(diffs) == 2:
                    report[f"{pool} | {name} | {value} | minus the rest"] = dict(holdout=diffs[0], recent=diffs[1])
                    print(f"    {value} minus the rest: holdout {diffs[0][0]:+.2f} {fmt(diffs[0][1])}, recent {diffs[1][0]:+.2f} {fmt(diffs[1][1])}")
                    if diffs[0][0]*diffs[1][0] > 0 and all(d[1][0]*d[1][1] > 0 for d in diffs):
                        consistent.append(f"{pool} | {name} = {value}")
    # Tournaments: is the spread of per-tournament gaps wider than chance? (traded tours, tournaments with 8+ favourites)
    part = pools["Chall+WTA+WTA Chall"]
    by = defaultdict(list)
    for r in part:
        by[r["tournament"]].append(100*r["payout"]-r["start"])
    big = {k: v for k, v in by.items() if len(v) >= 8}
    res = np.array([x for v in big.values() for x in v])
    sizes = [len(v) for v in big.values()]
    seen = float(np.average([(np.mean(v)-res.mean())**2 for v in big.values()], weights=sizes))
    sims = []
    for _ in range(2000):
        shuffled, at, total = RNG.permutation(res), 0, 0.0
        for n in sizes:
            total += n*(shuffled[at:at+n].mean()-res.mean())**2
            at += n
        sims.append(total/sum(sizes))
    tried += 1
    report["tournament spread"] = dict(tournaments=len(big), matches=len(res), between_variance=seen,
                                       chance_median=float(np.median(sims)), p=float((np.array(sims) >= seen).mean()))
    print(f"  tournaments with 8+ favourites: {len(big)} ({len(res)} matches); between-tournament variance of the gap {seen:.1f} "
          f"against {np.median(sims):.1f} by chance, p {report['tournament spread']['p']:.2f}")
    # The round, looked at more closely because it is the one feature that repeats: within each tour, and for the 80c dip bid.
    print("  qualifying against the main draw within each tour (gap, cents), and the 80c dip bid by round")
    weights, diffs = [], []
    for tour in TOURS.values():
        for period in ("holdout", "recent"):
            q = [100*r["payout"]-r["start"] for r in top if r["tour"] == tour and r["period"] == period and stage(r) == "qualifying"]
            m = [100*r["payout"]-r["start"] for r in top if r["tour"] == tour and r["period"] == period and stage(r) != "qualifying"]
            if len(q) >= 20 and len(m) >= 20:
                weights.append(1/(np.var(q)/len(q)+np.var(m)/len(m)))
                diffs.append(np.mean(q)-np.mean(m))
                report[f"round within tour | {tour} | {period}"] = dict(qualifying=len(q), main_draw=len(m), difference=float(diffs[-1]))
                print(f"    {tour:15} {period:8} qualifying {len(q):4} gap {np.mean(q):+6.2f} | main draw {len(m):4} gap {np.mean(m):+6.2f} | difference {diffs[-1]:+6.2f}")
    pooled, error = float(np.average(diffs, weights=weights)), float(1/np.sqrt(sum(weights)))
    report["round within tour | pooled"] = dict(cells=len(diffs), difference=pooled, standard_error=error, same_sign=int(sum(d > 0 for d in diffs)))
    print(f"    pooled within-tour difference {pooled:+.2f}c (standard error {error:.2f}); {sum(d > 0 for d in diffs)} of {len(diffs)} tour-period cells positive")
    for pool, part in pools.items():
        for period in ("holdout", "recent"):
            for value in ("qualifying", "early main draw", "quarterfinal on"):
                sub = [r for r in part if r["period"] == period and stage(r) == value]
                nets = [100*r["payout"]-DIP for r in sub if r["dip_fill"]]
                s = dict(matches=len(sub), fills=len(nets), net_per_fill=float(np.mean(nets)) if nets else None, ci95=ci(nets))
                report[f"dip bid by round | {pool} | {period} | {value}"] = s
                if nets:
                    print(f"    80c dip bid, {pool}, {period:8} {value:16} matches {len(sub):4} fills {len(nets):3} net/fill {s['net_per_fill']:+6.2f} {fmt(s['ci95'])}")
    report["features tried per pool"] = tried
    report["consistent in both periods"] = consistent
    print(f" {tried} feature-by-pool tests were run; consistent in both periods (same sign, both outside the noise band): {consistent or 'none'}")
    save(OUT/"q4-predictors.json", report)


# ---------------------------------------------------------------- Q5: a resting bid before the match

def bid_score(part, name, fill="strict", fee=False, skip_detection_bar=False):
    """One pre-match bid level over a set of matches placed. Nets in cents per contract."""
    filled = [r for r in part if r["bids"][name][fill] and not (skip_detection_bar and r["bids"][name]["strict_in_detection_bar"])]
    if not part:
        return dict(orders=0)
    nets = [100*r["payout"]-r["bids"][name]["level"]-(maker_fee(r["bids"][name]["level"]) if fee and r["tour"] not in NO_MAKER_FEE else 0)
            for r in filled]
    out = dict(orders=len(part), fills=len(filled), fill_rate=len(filled)/len(part))
    if filled:
        per_order = [100*r["payout"]-r["bids"][name]["level"] if r in filled else 0.0 for r in part] if not fee else None
        out.update(mean_level=float(np.mean([r["bids"][name]["level"] for r in filled])), won=float(np.mean([r["payout"] for r in filled])),
                   net_per_fill=float(np.mean(nets)), ci95=ci(nets), net_per_order=sum(nets)/len(part),
                   per_order_ci95=ci(per_order) if per_order else None,
                   median_minutes_to_fill=float(np.median([r["bids"][name]["minutes_to_fill"] for r in filled])) if fill == "strict" else None)
    return out


def taker_score(part):
    nets = [100*r["payout"]-r["ask0"]-taker_fee(r["ask0"]) for r in part]
    return dict(orders=len(part), mean_ask=float(np.mean([r["ask0"] for r in part])), won=float(np.mean([r["payout"] for r in part])),
                fee=float(np.mean([taker_fee(r["ask0"]) for r in part])), net_per_order=float(np.mean(nets)), ci95=ci(nets),
                net_before_fee=float(np.mean(nets))+float(np.mean([taker_fee(r["ask0"]) for r in part])))


def combined(part, name):
    """Pre-match bid plus the 80c dip bid on the same match; profit per match placed, one contract each."""
    if name == "take":
        pre = np.array([100*r["payout"]-r["ask0"]-taker_fee(r["ask0"]) for r in part])
    elif name:
        pre = np.array([100*r["payout"]-r["bids"][name]["level"] if r["bids"][name]["strict"] else 0.0 for r in part])
    else:
        pre = np.zeros(len(part))
    dip = np.array([100*r["payout"]-DIP if r["dip_fill"] else 0.0 for r in part])
    total = pre+dip
    both = sum(1 for r in part if r["dip_fill"] and (name == "take" or (name and r["bids"][name]["strict"])))
    return dict(matches=len(part), per_match=float(total.mean()), ci95=ci(total), worst_seen=float(total.min()),
                both_filled=both, dip_fills=int(sum(r["dip_fill"] for r in part)), pre_part=float(pre.mean()), dip_part=float(dip.mean()))


def q5(rows):
    report = {}
    top = [r for r in heavy(rows) if r["bid0"] is not None and r["ask0"] is not None]
    print("\n(Q5) A RESTING BID BEFORE THE MATCH. 90c+ favourites; order placed on the close of the 15th bar")
    print(" how much pre-match trading there is to fill a bid (between placement and the detected start)")
    for name, part in groups(top).items():
        for period in ("holdout", "recent"):
            sub = [r for r in part if r["period"] == period]
            if not sub:
                continue
            vol = [r["prematch_volume"] for r in sub]
            s = dict(matches=len(sub), median_rest_minutes=float(np.median([r["rest_minutes"] for r in sub])),
                     rest_minutes_quartiles=[float(v) for v in np.percentile([r["rest_minutes"] for r in sub], [25, 75])],
                     median_prematch_contracts=float(np.median(vol)), quartiles=[float(v) for v in np.percentile(vol, [25, 75])],
                     median_minutes_with_trades=float(np.median([r["prematch_trade_minutes"] for r in sub])),
                     share_no_trade=float(np.mean([v == 0 for v in vol])), median_spread=float(np.median([r["ask0"]-r["bid0"] for r in sub])),
                     share_any_print_below_bid=float(np.mean([r["prematch_low"] is not None and r["prematch_low"] < r["bid0"] for r in sub])))
            report[f"activity | {name} | {period}"] = s
            print(f'  {name:22} {period:8} n {s["matches"]:4} rest {s["median_rest_minutes"]:4.0f} min, contracts traded median {s["median_prematch_contracts"]:8.0f} '
                  f'(IQR {s["quartiles"][0]:.0f}-{s["quartiles"][1]:.0f}), minutes with a trade {s["median_minutes_with_trades"]:3.0f}, '
                  f'no trade at all {100*s["share_no_trade"]:3.0f}%, spread {s["median_spread"]:.0f}c')
    print(" bids (strict = a later pre-match print strictly below the bid; fee rows add the assumed maker fee where one applies)")
    for name, part in groups(top).items():
        for period in ("holdout", "recent", "both"):
            sub = [r for r in part if period == "both" or r["period"] == period]
            if len(sub) < 20:
                continue
            t = taker_score(sub)
            report[f"take the ask | {name} | {period}"] = t
            print(f'  {name:22} {period:8} take ask   orders {t["orders"]:4} at {t["mean_ask"]:5.1f} won {100*t["won"]:5.1f}% '
                  f'net/order {t["net_per_order"]:+6.2f} {fmt(t["ci95"])} (before fee {t["net_before_fee"]:+.2f}, fee {t["fee"]:.2f})')
            for level in LEVELS:
                s, touch = bid_score(sub, level), bid_score(sub, level, "touch")
                fee, clean = bid_score(sub, level, fee=True), bid_score(sub, level, skip_detection_bar=True)
                report[f"bid {level} | {name} | {period}"] = dict(strict=s, touch=touch, strict_with_maker_fee=fee, strict_excluding_detection_bar=clean)
                if not s["fills"]:
                    print(f'  {name:22} {period:8} {level:10} orders {s["orders"]:4} no strict fills; touch {touch["fills"]}')
                    continue
                print(f'  {name:22} {period:8} {level:10} fills {s["fills"]:4} ({100*s["fill_rate"]:4.1f}%) at {s["mean_level"]:5.1f} won {100*s["won"]:5.1f}% '
                      f'net/fill {s["net_per_fill"]:+6.2f} {fmt(s["ci95"])} net/order {s["net_per_order"]:+5.2f} {fmt(s["per_order_ci95"])} '
                      f'| with fee {fee["net_per_fill"]:+6.2f} | touch {touch["fills"]:4} ({100*touch["fill_rate"]:4.1f}%) net/fill {touch["net_per_fill"]:+6.2f} {fmt(touch["ci95"])} '
                      f'| outside detection bar {clean["fills"]:4} net/fill {clean.get("net_per_fill", float("nan")):+6.2f}')
    print(" favourites the bid missed (unfilled), for comparison: what the unfilled matches paid at their price")
    for name, part in groups(top).items():
        for level in ("at bid", "mid-1"):
            got = [r for r in part if r["bids"][level]["strict"]]
            missed = [r for r in part if not r["bids"][level]["strict"]]
            report[f"selection | {level} | {name}"] = dict(filled=gap(got), unfilled=gap(missed))
            print(line(f"{name[:20]} {level} filled", gap(got)))
            print(line(f"{name[:20]} {level} unfilled", gap(missed)))
    print(" by round (the one feature that repeated in Q4): take the ask, and the bid at the bid")
    for pool, part in (("Chall+WTA+WTA Chall", [r for r in top if r["tour"] in EDGE]), ("all six tours", top)):
        for value in ("qualifying", "early main draw", "quarterfinal on"):
            for period in ("holdout", "recent", "both"):
                sub = [r for r in part if stage(r) == value and (period == "both" or r["period"] == period)]
                if len(sub) < 20:
                    continue
                t, s = taker_score(sub), bid_score(sub, "at bid")
                report[f"by round | {pool} | {value} | {period}"] = dict(take=t, at_bid=s)
                print(f'  {pool:20} {value:16} {period:8} n {len(sub):4} take ask net/order {t["net_per_order"]:+6.2f} {fmt(t["ci95"])} | '
                      f'at bid fills {s["fills"]:4} net/fill {s["net_per_fill"]:+6.2f} {fmt(s["ci95"])} net/order {s["net_per_order"]:+5.2f} {fmt(s["per_order_ci95"])}')
    print(" combined plan: pre-match order plus the 80c dip bid, one contract each, profit per match placed (cents)")
    for name, part in groups(top).items():
        for period in ("holdout", "recent", "both"):
            sub = [r for r in part if period == "both" or r["period"] == period]
            if len(sub) < 20:
                continue
            for plan in (None, "take", "at bid", "mid-1", "mid-2"):
                c = combined(sub, plan)
                label = {None: "dip only", "take": "take ask + dip"}.get(plan, f"bid {plan} + dip")
                report[f"combined | {label} | {name} | {period}"] = c
                worst = -DIP if plan is None else -(DIP+max((r["ask0"] if plan == "take" else r["bids"][plan]["level"]) for r in sub))
                print(f'  {name:22} {period:8} {label:16} per match {c["per_match"]:+6.2f} {fmt(c["ci95"])} (pre {c["pre_part"]:+5.2f}, dip {c["dip_part"]:+5.2f}) '
                      f'dip fills {c["dip_fills"]:3} both filled {c["both_filled"]:3} worst seen {c["worst_seen"]:+7.1f} worst possible {worst:+.0f}')
    save(OUT/"q5-prematch-bid.json", report)


# ---------------------------------------------------------------- Q2: the other book

def bar_path(row, ticker):
    if row["archived"]:
        return f"/historical/markets/{ticker}/candlesticks"
    return f'/series/{row["series"]}/markets/{ticker}/candlesticks'


def two_books(row, own, other):
    """Compare the favourite on the cached market (A) and on the other player's market (B), minute by minute."""
    a = side_path(own, row["yes"])
    b = {p[0]: p for p in side_path(other, not row["yes"])}
    began = detect_start(a)
    out = dict(event=row["event"], tour=row["tour"], period=row["period"], payout=row["payout"], start=row["start"],
               favourite_market="A" if row["yes"] else "B", bars_b=len(b),
               volume_a=sum(p[5] for p in a), volume_b=sum(p[5] for p in b.values()))

    def quotes(i):
        pa, pb = a[i], b.get(a[i][0])
        if pb is None or None in (pa[1], pa[2], pb[1], pb[2]) or not (0 < pa[1] <= pa[2] < 100 and 0 < pb[1] <= pb[2] < 100):
            return None
        return dict(ask_a=pa[2], ask_b=pb[2], bid_a=pa[1], bid_b=pb[1], spread_a=pa[2]-pa[1], spread_b=pb[2]-pb[1])

    out["placement"] = quotes(WARMUP-1)
    out["before_play"] = quotes(began-1) if began is not None else None
    # Share of pre-match minutes where the two asks differ, and where the two books cross (one's bid above the other's ask).
    pre = [q for q in (quotes(i) for i in range(WARMUP-1, began if began is not None else WARMUP)) if q]
    out["prematch_minutes"] = len(pre)
    out["prematch_ask_differs"] = float(np.mean([q["ask_a"] != q["ask_b"] for q in pre])) if pre else None
    out["prematch_crossed"] = float(np.mean([q["bid_a"] > q["ask_b"] or q["bid_b"] > q["ask_a"] for q in pre])) if pre else None
    # Pre-match resting bid at the placement bid, on each book (strict), resting to the detection bar inclusive.
    stop = began if began is not None else min(len(a)-1, WARMUP-1+240)
    place = out["placement"]
    if place:
        for name, level, lows in (("a", place["bid_a"], [p[3] for p in a[WARMUP:stop+1]]),
                                  ("b", place["bid_b"], [b[p[0]][3] for p in a[WARMUP:stop+1] if p[0] in b])):
            out[f"prematch_fill_{name}"] = any(low is not None and low < level for low in lows)
    # The 80c dip bid on each book, resting from the bar after the detected start (from the warm-up if none is seen).
    first = began+1 if began is not None else WARMUP
    dip_a = next((i for i in range(first, len(a)) if a[i][3] is not None and a[i][3] < DIP), None)
    dip_b = next((i for i in range(first, len(a)) if a[i][0] in b and b[a[i][0]][3] is not None and b[a[i][0]][3] < DIP), None)
    out.update(dip_a=dip_a, dip_b=dip_b)
    for name, at in (("a", dip_a), ("b", dip_b)):
        if at is not None:
            out[f"dip_{name}_minutes_before_settlement"] = (row["settle"]-a[at][0])/60
    at = min((i for i in (dip_a, dip_b) if i is not None), default=None)
    if at is not None:
        out["at_dip_previous_bar"] = quotes(at-1)
        out["at_dip_close"] = quotes(at)
        pb = b.get(a[at][0])
        out["dip_low_a"], out["dip_low_b"] = a[at][3], pb[3] if pb else None
    return out


def q2(rows, minutes, download):
    todo = sorted((r for r in heavy(rows) if r["tour"] in EDGE and r["other_ticker"]), key=lambda r: -r["settle"])
    deadline, fetched, missing, studies = time.monotonic()+60*minutes, 0, 0, []
    print(f"\n(Q2) THE OTHER BOOK. {len(todo)} favourites at 90c+ on {', '.join(EDGE)}")
    with httpx.Client(timeout=40) as live, httpx.Client(transport=STUB) as stub:
        for n, row in enumerate(todo):
            params = bar_params(row["settle"])
            path = bar_path(row, row["other_ticker"])
            if not cached(path, params):
                if not download or time.monotonic() > deadline:
                    missing += 1
                    continue
                try:
                    paced(live, path, params)          # one request at a time, one per second, long pause on 429
                    fetched += 1
                except RuntimeError as error:
                    print(f"  {row['other_ticker']}: {error}")
                    missing += 1
                    continue
                if fetched % 100 == 0:
                    print(f"  downloaded {fetched} ({n+1} of {len(todo)} matches reached)", flush=True)
            other = fetch(stub, path, params).get("candlesticks") or []
            own = fetch(stub, bar_path(row, row["ticker"]), params).get("candlesticks") or []
            if other and own:
                studies.append(two_books(row, sorted(own, key=lambda x: x["end_period_ts"]), sorted(other, key=lambda x: x["end_period_ts"])))
            else:
                missing += 1
    print(f" downloaded {fetched} this run; {len(studies)} matches have both books, {missing} do not")
    save(OUT/"q2-rows.json", studies)
    report = dict(matches=len(studies), without_both_books=missing)

    def stats(values):
        values = [v for v in values if v is not None]
        return None if not values else dict(n=len(values), mean=float(np.mean(values)), ci95=ci(values))

    for scope in ("all", "holdout", "recent") + EDGE:
        part = [s for s in studies if scope == "all" or s["period"] == scope or s["tour"] == scope]
        if len(part) < 20:
            continue
        print(f" {scope}: {len(part)} matches")
        for moment in ("placement", "before_play", "at_dip_previous_bar", "at_dip_close"):
            have = [s for s in part if s.get(moment)]
            if len(have) < 10:
                continue
            q = [s[moment] for s in have]
            own = [x["ask_a"] if s["favourite_market"] == "A" else x["ask_b"] for s, x in zip(have, q)]     # YES on the favourite's own market
            under = [x["ask_b"] if s["favourite_market"] == "A" else x["ask_a"] for s, x in zip(have, q)]   # NO on the underdog's market
            best = [min(x["ask_a"], x["ask_b"]) for x in q]
            diff = np.array(own)-np.array(under)
            r = dict(matches=len(have), mean_ask_favourite_yes=float(np.mean(own)), mean_ask_underdog_no=float(np.mean(under)),
                     mean_best=float(np.mean(best)), saving_vs_favourite_yes=float(np.mean(np.array(own)-np.array(best))),
                     saving_vs_underdog_no=float(np.mean(np.array(under)-np.array(best))),
                     saving_vs_cached_book=float(np.mean([x["ask_a"]-min(x["ask_a"], x["ask_b"]) for x in q])),
                     share_differ_1c=float(np.mean(np.abs(diff) >= 1)), share_differ_2c=float(np.mean(np.abs(diff) >= 2)),
                     share_underdog_no_cheaper=float(np.mean(diff >= 1)), share_favourite_yes_cheaper=float(np.mean(diff <= -1)),
                     spread_favourite_market=float(np.mean([x["spread_a"] if s["favourite_market"] == "A" else x["spread_b"] for s, x in zip(have, q)])),
                     spread_underdog_market=float(np.mean([x["spread_b"] if s["favourite_market"] == "A" else x["spread_a"] for s, x in zip(have, q)])),
                     share_crossed=float(np.mean([x["bid_a"] > x["ask_b"] or x["bid_b"] > x["ask_a"] for x in q])))
            report[f"{scope} | quotes | {moment}"] = r
            print(f'  {moment:20} n {r["matches"]:4} ask: favourite YES {r["mean_ask_favourite_yes"]:5.2f}, underdog NO {r["mean_ask_underdog_no"]:5.2f}, best {r["mean_best"]:5.2f} '
                  f'(saves {r["saving_vs_favourite_yes"]:.2f} vs favourite YES, {r["saving_vs_underdog_no"]:.2f} vs underdog NO) | differ by 1c+ {100*r["share_differ_1c"]:3.0f}%, 2c+ {100*r["share_differ_2c"]:3.0f}% '
                  f'(underdog NO cheaper {100*r["share_underdog_no_cheaper"]:3.0f}%, favourite YES cheaper {100*r["share_favourite_yes_cheaper"]:3.0f}%) | '
                  f'spread {r["spread_favourite_market"]:.2f} / {r["spread_underdog_market"]:.2f} | crossed {100*r["share_crossed"]:.0f}%')
        share = [s["volume_a"]/(s["volume_a"]+s["volume_b"]) if s["favourite_market"] == "A" else s["volume_b"]/(s["volume_a"]+s["volume_b"])
                 for s in part if s["volume_a"]+s["volume_b"] > 0]
        report[f"{scope} | favourite market's share of the six-hour volume"] = float(np.median(share))
        print(f"  favourite's own market carries a median {100*np.median(share):.0f}% of the two books' volume in the window; "
              f"asks differ in {100*np.mean([s['prematch_ask_differs'] for s in part if s['prematch_ask_differs'] is not None]):.0f}% of pre-match minutes, "
              f"books crossed in {100*np.mean([s['prematch_crossed'] for s in part if s['prematch_crossed'] is not None]):.1f}%")
        # Buying before the match at the better ask, after the taker fee.
        have = [s for s in part if s.get("placement")]
        for label, pick in (("favourite YES", lambda s, x: x["ask_a"] if s["favourite_market"] == "A" else x["ask_b"]),
                            ("underdog NO", lambda s, x: x["ask_b"] if s["favourite_market"] == "A" else x["ask_a"]),
                            ("better of the two", lambda s, x: min(x["ask_a"], x["ask_b"]))):
            nets = [100*s["payout"]-pick(s, s["placement"])-taker_fee(pick(s, s["placement"])) for s in have]
            report[f"{scope} | take the ask | {label}"] = stats(nets)
            print(f"  take the ask before the match, {label:18} n {len(nets):4} net/order {np.mean(nets):+6.2f} {fmt(ci(nets))}")
        for name in ("a", "b"):
            got = [s for s in have if s.get(f"prematch_fill_{name}")]
            label = "cached (first-listed) book" if name == "a" else "other book"
            report[f"{scope} | pre-match bid at the bid | {label}"] = dict(orders=len(have), fills=len(got))
            print(f"  pre-match bid at the bid, {label:26} fills {len(got):4} of {len(have)} ({100*len(got)/max(1, len(have)):.0f}%)")
        either = sum(1 for s in have if s.get("prematch_fill_a") or s.get("prematch_fill_b"))
        print(f"  pre-match bid at the bid, either book {either} of {len(have)}")
        # The 80c dip bid on each book.
        fa, fb = [s for s in part if s["dip_a"] is not None], [s for s in part if s["dip_b"] is not None]
        both = [s for s in part if s["dip_a"] is not None and s["dip_b"] is not None]
        either = [s for s in part if s["dip_a"] is not None or s["dip_b"] is not None]
        fav = [s for s in part if s["dip_a" if s["favourite_market"] == "A" else "dip_b"] is not None]
        dog = [s for s in part if s["dip_b" if s["favourite_market"] == "A" else "dip_a"] is not None]
        lead = [s["dip_a"]-s["dip_b"] for s in both]                 # positive: the other book printed first
        lead_dog = [(s["dip_a"]-s["dip_b"])*(1 if s["favourite_market"] == "A" else -1) for s in both]
        d = dict(matches=len(part))
        for label, group in (("cached book only", fa), ("other book only", fb), ("favourite's YES market", fav), ("underdog's market (NO)", dog),
                             ("either book, first fill", either), ("both books filled", both)):
            nets = [100*s["payout"]-DIP for s in group]
            d[label] = dict(fills=len(group), fill_rate=len(group)/len(part), net_per_fill=float(np.mean(nets)) if nets else None, ci95=ci(nets),
                            per_match=sum(nets)/len(part))
            print(f"  80c dip bid, {label:24} fills {len(group):4} ({100*len(group)/len(part):4.1f}%) net/fill {np.mean(nets) if nets else float('nan'):+6.2f} {fmt(ci(nets))} "
                  f"per match {sum(nets)/len(part):+5.2f}")
        only_a = [s for s in fa if s["dip_b"] is None]
        only_b = [s for s in fb if s["dip_a"] is None]
        d.update(only_cached=len(only_a), only_other=len(only_b),
                 only_cached_net=float(np.mean([100*s["payout"]-DIP for s in only_a])) if only_a else None,
                 only_other_net=float(np.mean([100*s["payout"]-DIP for s in only_b])) if only_b else None,
                 same_minute=int(sum(v == 0 for v in lead)), other_first=int(sum(v > 0 for v in lead)), cached_first=int(sum(v < 0 for v in lead)),
                 underdog_market_first=int(sum(v > 0 for v in lead_dog)), favourite_market_first=int(sum(v < 0 for v in lead_dog)),
                 median_lead_minutes_when_different=float(np.median([abs(v) for v in lead if v])) if any(lead) else None)
        report[f"{scope} | dip bid"] = d
        print(f"  filled on one book only: cached {len(only_a)} (net {d['only_cached_net']}), other {len(only_b)} (net {d['only_other_net']}); "
              f"both filled {len(both)}: same minute {d['same_minute']}, underdog's market first {d['underdog_market_first']}, "
              f"favourite's market first {d['favourite_market_first']}, median lead {d['median_lead_minutes_when_different']} min")
    save(OUT/"q2-other-book.json", report)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("steps", nargs="+", choices=("build", "q3", "q4", "q5", "q2"))
    parser.add_argument("--minutes", type=float, default=40, help="download budget for q2")
    parser.add_argument("--no-download", action="store_true", help="q2 from the cache only")
    args = parser.parse_args()
    for step in args.steps:
        if step == "build":
            build()
        elif step == "q2":
            q2(load(), args.minutes, not args.no_download)
        else:
            dict(q3=q3, q4=q4, q5=q5)[step](load())
