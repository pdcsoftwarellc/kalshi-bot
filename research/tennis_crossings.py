"""In-play "first crossing" calibration map for Kalshi tennis: every match, both players, every price level.

Question. Once play has started, is a player's contract mispriced the first time
its price falls through, or rises through, a given level? The one live rule
(pre-match favourite at 90c+, resting 80c bid) is a single cell of this map.

Data. One-minute bars already cached under data/raw/cost-research-v1 (the six
hours before settlement, the alphabetically-first market of each event) for
ATP, WTA, ATP Challenger, WTA Challenger, ITF men and ITF women. No network:
the HTTP client is a stub that answers 404. Archive = settled before
2026-09-18, recent = after; an event listed in both is kept in the archive.

Definitions
* Player A is the YES side of the cached market, player B its complement (the
  same book read from the other side: B's ask is 100 minus A's bid, B's low
  print is 100 minus A's high print). Both players of every match are studied.
* P0 = median midpoint of the first 15 bars, each with a two-sided quote at most
  4c wide, the 15 midpoints within 3c (favorite_dip.py). B's P0 is 100 - P0.
* Start of play = favourite_gap.detect_start (midpoint moved more than 2c in ten
  minutes, or open interest jumped). Matches with no detected start are left
  out. Orders are decided on the close of the detection bar; only later bars
  can fill them. `ask_s` is the player's ask on that close.
* DOWN (resting bid at L): placed when the player's bid on that close is at or
  above L (the price has not yet come down to the level). Filled in the first later bar
  with a trade print strictly below L. Pays L. Fee: none on the Challenger
  series; elsewhere shown with no fee and with an assumed maker fee
  0.0175*p*(1-p) ("realistic" = that assumption).
* UP (buy on strength at L): placed when ask_s < L. Signal = first later bar
  whose closing ask is at or above L. Entry = the NEXT bar's opening ask,
  taken only if it lies in [L, L+4] (otherwise the price jumped past the level
  and the order for this level is dropped). Pays that ask plus the taker fee
  0.07*p*(1-p).
* Held to settlement; fractional settlements count as fractional wins. Net per
  fill = 100 x settlement value - price paid - fee.
* Cells: P0 band (under 20, 20-35, 35-50, 50-65, 65-80, 80-90, 90+) x level
  (every 5c, 5..95 down, 10..95 up) x direction, by period and by tour.
* Round from the rules text (favourite_gap.parse_rules). Time since the detected
  start: 0-20, 20-60, 60+ minutes. Fast move: the midpoint three bars before
  the fill (or signal) bar was 5c or more on the far side of L.

The standard a cell must meet (step `map`): archive net positive with a 95%
interval above zero and at least 30 fills; recent net positive with at least 15
fills; and every neighbouring cell (adjacent level, adjacent P0 band, same
direction) with 20+ pooled fills is positive on both periods pooled, with at
least two such neighbours. The pass counts use a normal-approximation interval
so that the identical test can be run on simulated fair markets; the tables
also carry bootstrap intervals (resampling matches).

Chance rate (step `null`): every real match is replayed as a fair market. The
YES price follows the exact martingale p_k = Phi(W_k / sqrt(1 - k/N)), W a
random walk started at Phi^-1(P0), N steps, settling at 0 or 100. The same
order rules, fees, tours, periods and P0 bands are applied, and the whole map
is scored with the same standard. Each replication redraws every match once,
so fills within a match stay dependent. The walk has jumps, so a resting bid
filled "through" its level is adversely selected, as in the real data; 700
steps reproduce the real average cost of a resting fill (about -1.4c to -1.8c),
120 steps overstate it and so understate the chance rate.

Special structures (step `special`): underdogs that rally or collapse, coin-flip
matches that reach 80-90 and dip back to 65-75, and second dips below 80 on
heavy favourites that had recovered above 90.

    .venv/bin/python research/tennis_crossings.py build map null cuts special rows rank
    .venv/bin/python research/tennis_crossings.py null --reps 200 --walk-steps 700 120

Outputs under data/processed/tennis-crossings/. Public market data only; no
orders, no authentication, nothing fetched.
"""
import argparse
from collections import defaultdict
import json
from pathlib import Path
import sys

import httpx
import numpy as np
import pandas as pd
from scipy.special import ndtr, ndtri

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from research.api_cost import ROOT, fetch, timestamp
from research.dip_archive_tours import cached
from research.favorite_dip import WARMUP
from research.favourite_gap import CUTOFF, STUB, TOURS, bar_params, detect_start, listing, parse_rules, side_path

OUT = ROOT / "data/processed/tennis-crossings"
LEVELS = np.arange(5, 100, 5)                         # index 0..18
BANDS = ((0, 20), (20, 35), (35, 50), (50, 65), (65, 80), (80, 90), (90, 101))
BAND_NAMES = ("<20", "20-35", "35-50", "50-65", "65-80", "80-90", "90+")
TOUR_NAMES = ("Challenger", "WTA", "WTA Challenger", "ATP", "ITF men", "ITF women")
NO_MAKER_FEE = ("Challenger", "WTA Challenger")
UNITS = {**{t: (t,) for t in TOUR_NAMES}, "Chall+WTA+WTA Chall": ("Challenger", "WTA", "WTA Challenger"),
         "ITF both": ("ITF men", "ITF women"), "ALL": TOUR_NAMES}
PERIODS = ("archive", "recent")
DIRS = ("down", "up")
MIN_ARCHIVE, MIN_RECENT, MIN_NEIGHBOUR = 30, 15, 20
RECENT_MONTHS, ARCHIVE_MONTHS = 59/30.44, 260/30.44    # 18 Sep - 16 Nov; 1 Jan - 17 Sep
RNG = np.random.default_rng(17)
SHAPE = (len(PERIODS), len(BANDS), len(DIRS), len(LEVELS))


def band_of(p0):
    return next(i for i, (lo, hi) in enumerate(BANDS) if lo <= p0 < hi)


def taker_fee(p):
    return 7*p/100*(1-p/100)


def maker_fee(p):
    return 1.75*p/100*(1-p/100)


# ---------------------------------------------------------------- build

def arrays(bars):
    """YES-side arrays per bar; missing values are NaN."""
    path = side_path(bars, True)
    f = lambda k: np.array([np.nan if p[k] is None else p[k] for p in path], float)
    ts, bid, ask, low, high, vol = (f(k) for k in range(6))

    def opening(side):
        out = []
        for b in bars:
            v = b.get(side, {})
            v = v.get("open_dollars", v.get("open"))
            out.append(np.nan if v is None else round(100*float(v)))
        return np.array(out, float)
    return path, ts, bid, ask, low, high, vol, opening("yes_bid"), opening("yes_ask")


def first(mask):
    """Index of the first True, or None."""
    i = int(np.argmax(mask)) if mask.size else 0
    return i if mask.size and mask[i] else None


def study(mid, bars, fills, coin, reentry):
    """Append this match's fills; return (began, quotes at the start) or None when there is no usable start."""
    path, ts, bid, ask, low, high, vol, obid, oask = arrays(bars)
    began = detect_start(path)
    if began is None or began+1 >= len(path):
        return None
    ok = np.where(~np.isnan(bid[:began+1]) & ~np.isnan(ask[:began+1]))[0]
    if not len(ok):
        return None
    bid_s, ask_s = bid[ok[-1]], ask[ok[-1]]
    mid_yes = (bid+ask)/2
    n = len(ts)
    for player in (0, 1):
        if player == 0:
            p_low, p_bid, p_ask, p_oask, p_mid, a_s, b_s = low, bid, ask, oask, mid_yes, ask_s, bid_s
        else:
            p_low, p_bid, p_ask, p_oask, p_mid, a_s, b_s = 100-high, 100-ask, 100-bid, 100-obid, 100-mid_yes, 100-bid_s, 100-ask_s
        lows = np.where(np.isnan(p_low), 999, p_low)
        asks = np.where(np.isnan(p_ask), -1, p_ask)
        bids = np.where(np.isnan(p_bid), -1, p_bid)
        after = slice(began+1, n)

        def meta(i, level, sign):
            back = p_mid[max(began, i-3)]
            fast = bool(not np.isnan(back) and sign*(back-level) >= 5)
            later = p_mid[i+1] if i+1 < n else np.nan
            return (ts[i]-ts[began])/60, fast, 0.0 if np.isnan(vol[i]) else vol[i], later

        for li, level in enumerate(LEVELS):
            if b_s >= level:                                             # resting bid at `level`
                j = first(lows[after] < level)
                if j is not None:
                    i = began+1+j
                    minutes, fast, volume, later = meta(i, level, 1)
                    fills.append((mid, player, 0, li, float(level), minutes, fast, volume, level-lows[i], later))
            if a_s < level and level >= 10:                              # buy on strength through `level`
                j = first(asks[after] >= level)
                if j is not None and began+2+j < n:
                    i = began+1+j
                    paid = p_oask[i+1]
                    if not np.isnan(paid) and level <= paid <= level+4:
                        minutes, fast, volume, later = meta(i, level, -1)
                        fills.append((mid, player, 1, li, float(paid), minutes, fast, volume, paid-level, later))
        # (b) reach H on the bid, then a resting bid at L below it
        for high_level in (80, 85, 90):
            if a_s >= high_level:
                continue
            i = first(bids[after] >= high_level)
            if i is None:
                continue
            i += began+1
            for level in (65, 70, 75):
                j = first(lows[i+1:] < level)
                coin.append((mid, player, high_level, level, j is not None,
                             np.nan if j is None else (ts[i+1+j]-ts[i])/60, np.nan if j is None else vol[i+1+j]))
        # (c) dip below 80, recover to a 90 bid, dip below 80 again
        if b_s >= 80:
            j1 = first(lows[after] < 80)
            if j1 is not None:
                j1 += began+1
                k = first(bids[j1+1:] >= 90)
                k = None if k is None else j1+1+k
                j2 = None if k is None else first(lows[k+1:] < 80)
                reentry.append((mid, player, True, k is not None, j2 is not None))
    return began, bid_s, ask_s, ts[began]


def build():
    matches, fills, coin, reentry, seen = [], [], [], [], set()
    counts = defaultdict(lambda: [0, 0, 0, 0])
    with httpx.Client(transport=STUB) as client:
        for archived in (True, False):
            for series, tour in TOURS.items():
                events, path, lo, hi = listing(client, series, archived)
                for event, group in events.items():
                    m = min(group, key=lambda m: m["ticker"])
                    try:
                        outcome, settle = float(m["settlement_value_dollars"]), int(timestamp(m["settlement_ts"]))
                    except (KeyError, TypeError, ValueError):
                        continue
                    if not lo <= settle < hi or event in seen:
                        continue
                    key = counts[tour, PERIODS[0] if archived else PERIODS[1]]
                    key[0] += 1
                    params = bar_params(settle)
                    if not cached(path.format(t=m["ticker"]), params):
                        continue
                    bars = fetch(client, path.format(t=m["ticker"]), params).get("candlesticks")
                    if not bars:
                        continue
                    key[1] += 1
                    seen.add(event)
                    bars = sorted(bars, key=lambda b: b["end_period_ts"])
                    mids = []
                    for b in bars[:WARMUP]:
                        try:
                            bid = round(100*float(b["yes_bid"].get("close_dollars", b["yes_bid"].get("close"))))
                            ask = round(100*float(b["yes_ask"].get("close_dollars", b["yes_ask"].get("close"))))
                        except (KeyError, TypeError, ValueError, AttributeError):
                            break
                        if not 0 < bid <= ask < 100 or ask-bid > 4:
                            break
                        mids.append((bid+ask)/2)
                    if len(mids) < WARMUP or max(mids)-min(mids) > 3:
                        continue
                    key[2] += 1
                    mid = len(matches)
                    got = study(mid, bars, fills, coin, reentry)
                    if got is None:
                        continue
                    key[3] += 1
                    began, bid_s, ask_s, start_ts = got
                    tournament, stage = parse_rules(m.get("rules_primary"))
                    matches.append(dict(mid=mid, event=event, tour=tour, period=PERIODS[0] if archived else PERIODS[1], settle=settle,
                                        p0=float(np.median(mids)), outcome=outcome, bid_s=bid_s, ask_s=ask_s, start_ts=start_ts,
                                        minutes_played=(settle-start_ts)/60, bars=len(bars), tournament=tournament, round=stage,
                                        qualifying=None if stage is None else stage.startswith("Qualif"),
                                        volume=sum(float(x.get("volume_fp") or x.get("volume") or 0) for x in group)))
    OUT.mkdir(parents=True, exist_ok=True)
    print("BUILD  listed / with cached bars / stable pre-match quote / start of play detected")
    for (tour, period), c in sorted(counts.items()):
        print(f"  {tour:15} {period:8} {c[0]:6} {c[1]:6} {c[2]:6} {c[3]:6}")
    pd.DataFrame(matches).to_pickle(OUT/"matches.pkl")
    pd.DataFrame(fills, columns=["mid", "player", "dir", "li", "paid", "minutes", "fast", "bar_volume", "through", "mid_next"]).to_pickle(OUT/"fills.pkl")
    pd.DataFrame(coin, columns=["mid", "player", "high", "level", "filled", "minutes_after_high", "bar_volume"]).to_pickle(OUT/"coin.pkl")
    pd.DataFrame(reentry, columns=["mid", "player", "first", "recovered", "second"]).to_pickle(OUT/"reentry.pkl")
    (OUT/"build-counts.json").write_text(json.dumps({f"{t} | {p}": dict(zip(("listed", "bars", "stable", "started"), c))
                                                     for (t, p), c in sorted(counts.items())}, indent=1))


# ---------------------------------------------------------------- shared

def load():
    m = pd.read_pickle(OUT/"matches.pkl").set_index("mid")
    f = pd.read_pickle(OUT/"fills.pkl").join(m[["tour", "period", "p0", "outcome", "bid_s", "ask_s", "qualifying", "settle"]], on="mid")
    f["p0p"] = np.where(f.player == 0, f.p0, 100-f.p0)
    f["payout"] = np.where(f.player == 0, f.outcome, 1-f.outcome)
    f["band"] = f.p0p.map(band_of)
    f["pi"] = (f.period == "recent").astype(int)
    fee = np.where(f.dir == 1, taker_fee(f.paid), np.where(f.tour.isin(NO_MAKER_FEE), 0.0, maker_fee(f.paid)))
    f["gross"] = 100*f.payout-f.paid
    f["net_nofee"] = f.gross-np.where(f.dir == 1, taker_fee(f.paid), 0.0)
    f["net"] = f.gross-fee
    return m, f


def players(m):
    """One row per match and player: P0, band, ask at the start."""
    a = m[["tour", "period", "qualifying"]].assign(player=0, p0p=m.p0, ask=m.ask_s, bid=m.bid_s)
    b = m[["tour", "period", "qualifying"]].assign(player=1, p0p=100-m.p0, ask=100-m.bid_s, bid=100-m.ask_s)
    out = pd.concat([a, b]).reset_index()
    out["band"] = out.p0p.map(band_of)
    out["pi"] = (out.period == "recent").astype(int)
    return out


def placed_grid(pl):
    """Orders placed per cell: down when bid >= L, up when ask < L (levels from 10)."""
    grid = np.zeros(SHAPE)
    for li, level in enumerate(LEVELS):
        down = pl[pl.bid >= level].groupby(["pi", "band"]).size()
        for (pi, band), n in down.items():
            grid[pi, band, 0, li] = n
        if level >= 10:
            up = pl[pl.ask < level].groupby(["pi", "band"]).size()
            for (pi, band), n in up.items():
                grid[pi, band, 1, li] = n
    return grid


def sums(f, col):
    """n, sum, sum of squares of `col` on the (period, band, direction, level) grid."""
    flat = np.ravel_multi_index((f.pi.values, f.band.values, f.dir.values, f.li.values), SHAPE)
    size = int(np.prod(SHAPE))
    v = f[col].values
    return tuple(np.bincount(flat, w, size).reshape(SHAPE) for w in (None, v, v*v))


def moments(n, s, ss):
    with np.errstate(divide="ignore", invalid="ignore"):
        mean = s/n
        se = np.sqrt(np.maximum(ss/n-mean**2, 0)/np.maximum(n-1, 1))
    return mean, se


def standard(n, s, ss):
    """The three-part standard on one unit's grid. Returns boolean grids: archive significant, +recent, +neighbours."""
    mean, se = moments(n, s, ss)
    a = (n[0] >= MIN_ARCHIVE) & (mean[0]-1.96*se[0] > 0)
    b = a & (n[1] >= MIN_RECENT) & (mean[1] > 0)
    pn, ps = n.sum(0), s.sum(0)
    with np.errstate(divide="ignore", invalid="ignore"):
        pooled = ps/pn
    counted = pn >= MIN_NEIGHBOUR
    good, bad = counted & (pooled > 0), counted & ~(pooled > 0)
    agree, against = np.zeros(pn.shape, int), np.zeros(pn.shape, int)
    for axis in (0, 2):                                 # adjacent P0 band, adjacent level
        for shift in (1, -1):
            g, w = np.roll(good, shift, axis), np.roll(bad, shift, axis)
            edge = [slice(None)]*3
            edge[axis] = 0 if shift == 1 else -1
            g[tuple(edge)], w[tuple(edge)] = False, False
            agree += g
            against += w
    c = b & (agree >= 2) & (against == 0)
    return a, b, c


def boot(values, draws=2000):
    values = np.asarray(values, float)
    if len(values) < 10:
        return (np.nan, np.nan)
    means = values[RNG.integers(0, len(values), (draws, len(values)))].mean(axis=1)
    return tuple(np.percentile(means, [2.5, 97.5]))


def brief(values):
    values = np.asarray(values, float)
    if not len(values):
        return "   0 fills"
    lo, hi = boot(values)
    return f"{len(values):4} fills {values.mean():+6.2f} [{lo:+6.2f},{hi:+6.2f}]"


# ---------------------------------------------------------------- map

def cell_table(m, f, unit, col="net"):
    tours = UNITS[unit]
    fu, pl = f[f.tour.isin(tours)], players(m[m.tour.isin(tours)])
    placed = placed_grid(pl)
    rows = []
    for (pi, band, d, li), g in fu.groupby(["pi", "band", "dir", "li"]):
        lo, hi = boot(g[col].values)
        rows.append(dict(unit=unit, period=PERIODS[pi], band=BAND_NAMES[band], direction=DIRS[d], level=int(LEVELS[li]),
                         placed=int(placed[pi, band, d, li]), fills=len(g), fill_rate=len(g)/max(1, placed[pi, band, d, li]),
                         paid=g.paid.mean(), won=g.payout.mean(), net=g[col].mean(), ci_lo=lo, ci_hi=hi,
                         net_no_maker_fee=g.net_nofee.mean(), gross=g.gross.mean(),
                         per_placed=g[col].sum()/max(1, placed[pi, band, d, li]), median_bar_volume=g.bar_volume.median(),
                         mean_through=g.through.mean()))
    return pd.DataFrame(rows)


def show_grid(n, s, ss, title):
    mean, se = moments(n, s, ss)
    for d, name in enumerate(DIRS):
        for pi, period in enumerate(PERIODS):
            print(f"  {title} | {name} | {period}: net cents per fill (fills); * = normal 95% interval excludes zero")
            print("   P0\\L  "+"".join(f"{int(level):>11}" for level in LEVELS))
            for b, band in enumerate(BAND_NAMES):
                cells = []
                for li in range(len(LEVELS)):
                    k = n[pi, b, d, li]
                    if k < 10:
                        cells.append(f"{'.':>11}")
                        continue
                    star = "*" if abs(mean[pi, b, d, li]) > 1.96*se[pi, b, d, li] else " "
                    cells.append(f"{mean[pi, b, d, li]:+5.1f}{star}({int(k):3})"[:11].rjust(11))
                print(f"   {band:6}"+"".join(cells))


def do_map():
    m, f = load()
    print(f"\nMAP  {len(m)} matches with a stable pre-match quote and a detected start; {len(f)} fills "
          f"({(f.dir == 0).sum()} resting-bid, {(f.dir == 1).sum()} buy-on-strength)")
    tables, report = [], {}
    for unit, tours in UNITS.items():
        fu = f[f.tour.isin(tours)]
        report[unit] = {}
        for col, label in (("net", "realistic fees"), ("net_nofee", "no maker fee anywhere")):
            n, s, ss = sums(fu, col)
            a, b, c = standard(n, s, ss)
            tested = int(((n[0] >= MIN_ARCHIVE)).sum())
            mean, _ = moments(n, s, ss)
            positive = int(((n[0] >= MIN_ARCHIVE) & (mean[0] > 0)).sum())
            report[unit][label] = dict(cells_with_30_archive_fills=tested, archive_positive=positive, archive_significant=int(a.sum()),
                                       and_recent_positive=int(b.sum()), and_neighbours_agree=int(c.sum()),
                                       down=dict(a=int(a[:, 0].sum()), b=int(b[:, 0].sum()), c=int(c[:, 0].sum())),
                                       up=dict(a=int(a[:, 1].sum()), b=int(b[:, 1].sum()), c=int(c[:, 1].sum())),
                                       passing=[dict(band=BAND_NAMES[i], direction=DIRS[d], level=int(LEVELS[li]), neighbours=bool(c[i, d, li]))
                                                for i, d, li in zip(*np.where(b))])
            if col == "net":
                np.savez(OUT/f"grid-{unit.replace('+', '_').replace(' ', '_')}.npz", n=n, s=s, ss=ss)
        r = report[unit]["realistic fees"]
        q = report[unit]["no maker fee anywhere"]
        print(f" {unit:20} cells tested {r['cells_with_30_archive_fills']:3}  archive>0 {r['archive_positive']:3}  archive sig {r['archive_significant']:2} "
              f"-> +recent {r['and_recent_positive']:2} -> +neighbours {r['and_neighbours_agree']:2}   "
              f"(no maker fee: {q['archive_significant']}/{q['and_recent_positive']}/{q['and_neighbours_agree']})")
        for p in r["passing"]:
            print(f"      {p['band']:6} {p['direction']:4} L={p['level']:2}{'  <- neighbours agree' if p['neighbours'] else ''}")
        tables.append(cell_table(m, f, unit))
    pd.concat(tables).to_csv(OUT/"cells.csv", index=False, float_format="%.3f")
    (OUT/"map-pass-counts.json").write_text(json.dumps(report, indent=1))
    for unit in ("ALL", "Chall+WTA+WTA Chall", "ITF both", "ATP"):
        fu = f[f.tour.isin(UNITS[unit])]
        show_grid(*sums(fu, "net"), unit)
    # Overall: is a resting bid adversely selected, and what does buying strength cost?
    print(" totals by direction and period (realistic fees), all tours")
    for d, name in enumerate(DIRS):
        for period in PERIODS:
            g = f[(f.dir == d) & (f.period == period)]
            by_match = g.groupby("mid").net.sum()
            print(f"   {name:4} {period:8} {len(g):6} fills, net/fill {g.net.mean():+.2f}, gross {g.gross.mean():+.2f}, "
                  f"mean through/overshoot {g.through.mean():.2f}c; per match sum {by_match.mean():+.1f} "
                  f"[{boot(by_match.values)[0]:+.1f},{boot(by_match.values)[1]:+.1f}]")


# ---------------------------------------------------------------- null

def simulate(m, reps, steps, spread=1):
    """Replay every match as a fair market `reps` times; return pass counts per unit and the mean null grid."""
    pl = players(m)
    tour_i = m.tour.map({t: i for i, t in enumerate(TOUR_NAMES)}).values
    pi = (m.period == "recent").astype(int).values
    p0 = m.p0.values
    z0 = ndtri(np.clip(p0, 1, 99)/100)
    n_m = len(m)
    start_ask = np.stack([np.rint(p0)+spread, 100-np.rint(p0)+spread])            # per player
    start_bid = start_ask-2*spread
    bands = np.stack([np.array([band_of(v) for v in p0]), np.array([band_of(100-v) for v in p0])])
    no_fee = np.isin(m.tour.values, NO_MAKER_FEE)
    scale = np.sqrt(1-np.arange(1, steps)/steps)
    full = (len(TOUR_NAMES),)+SHAPE
    size = int(np.prod(full))
    counts = {u: np.zeros((reps, 3), int) for u in UNITS}
    by_dir = {u: np.zeros((reps, 2, 3), int) for u in UNITS}
    mean_grid, fills_grid = np.zeros(SHAPE), np.zeros(SHAPE)
    for rep in range(reps):
        w = z0[:, None]+np.cumsum(RNG.standard_normal((n_m, steps), dtype=np.float32)/np.sqrt(steps), axis=1)
        won = w[:, -1] > 0
        price = np.clip(np.rint(100*ndtr(w[:, :-1]/scale)), 1, 99)                   # steps-1 live prices, YES side
        idx, val = [], []
        for player in (0, 1):
            p = price if player == 0 else 100-price
            y = won if player == 0 else ~won
            lowest = np.minimum(p.min(axis=1), np.where(y, 99, 0))                  # the loser's price ends at zero
            ask = p+spread
            for li, level in enumerate(LEVELS):
                hit = (start_bid[player] >= level) & (lowest < level)
                fee = np.where(no_fee[hit], 0.0, maker_fee(level))
                idx.append(np.ravel_multi_index((tour_i[hit], pi[hit], bands[player][hit], np.zeros(hit.sum(), int), np.full(hit.sum(), li)), full))
                val.append(100*y[hit]-level-fee)
                if level >= 10:
                    cross = ask >= level
                    at = cross.argmax(axis=1)
                    paid = ask[np.arange(n_m), at]
                    hit = (start_ask[player] < level) & cross.any(axis=1) & (paid <= level+4) & (paid <= 99)
                    idx.append(np.ravel_multi_index((tour_i[hit], pi[hit], bands[player][hit], np.ones(hit.sum(), int), np.full(hit.sum(), li)), full))
                    val.append(100*y[hit]-paid[hit]-taker_fee(paid[hit]))
        idx, val = np.concatenate(idx), np.concatenate(val)
        n = np.bincount(idx, None, size).reshape(full)
        s = np.bincount(idx, val, size).reshape(full)
        ss = np.bincount(idx, val*val, size).reshape(full)
        for unit, tours in UNITS.items():
            pick = [TOUR_NAMES.index(t) for t in tours]
            a, b, c = standard(n[pick].sum(0), s[pick].sum(0), ss[pick].sum(0))
            counts[unit][rep] = a.sum(), b.sum(), c.sum()
            for d in (0, 1):
                by_dir[unit][rep, d] = a[:, d].sum(), b[:, d].sum(), c[:, d].sum()
        mean_grid += s.sum(0)
        fills_grid += n.sum(0)
    with np.errstate(divide="ignore", invalid="ignore"):
        return counts, by_dir, mean_grid/fills_grid, fills_grid/reps


def do_null(reps, steps):
    m, f = load()
    seen = json.loads((OUT/"map-pass-counts.json").read_text())
    report = {}
    for n_steps in steps:
        counts, by_dir, null_mean, null_fills = simulate(m, reps, n_steps)
        print(f"\nNULL  fair-market replay, {reps} replications, {n_steps} steps per match")
        print("  unit                  real: archive sig / +recent / +neighbours   |  fair market mean (95th pct) of the same three  |  P(fair >= real)")
        report[n_steps] = {}
        for unit in UNITS:
            r = seen[unit]["realistic fees"]
            real = np.array([r["archive_significant"], r["and_recent_positive"], r["and_neighbours_agree"]])
            c = counts[unit]
            p = (c >= real).mean(axis=0)
            report[n_steps][unit] = dict(real=real.tolist(), fair_mean=c.mean(axis=0).round(2).tolist(),
                                         fair_p95=np.percentile(c, 95, axis=0).tolist(), p_fair_at_least_real=p.round(3).tolist(),
                                         fair_mean_down=by_dir[unit][:, 0].mean(axis=0).round(2).tolist(),
                                         fair_mean_up=by_dir[unit][:, 1].mean(axis=0).round(2).tolist(),
                                         real_down=[r["down"][k] for k in "abc"], real_up=[r["up"][k] for k in "abc"])
            print(f"  {unit:20}  {real[0]:3} / {real[1]:3} / {real[2]:3}   |  {c[:, 0].mean():5.2f} ({np.percentile(c[:, 0], 95):.0f}) / "
                  f"{c[:, 1].mean():5.2f} ({np.percentile(c[:, 1], 95):.0f}) / {c[:, 2].mean():5.2f} ({np.percentile(c[:, 2], 95):.0f})  |  "
                  f"{p[0]:.3f} / {p[1]:.3f} / {p[2]:.3f}    down real {report[n_steps][unit]['real_down']} fair {report[n_steps][unit]['fair_mean_down']}; "
                  f"up real {report[n_steps][unit]['real_up']} fair {report[n_steps][unit]['fair_mean_up']}")
        # How the real map compares with the fair one, cell by cell (pooled periods, all tours).
        n, s, ss = sums(f, "net")
        real_fills = n.sum(0)
        with np.errstate(divide="ignore", invalid="ignore"):
            real_mean = s.sum(0)/real_fills
        fair = (null_mean*null_fills).sum(0)/null_fills.sum(0)
        fair_fills = null_fills.sum(0)
        for d, name in enumerate(DIRS):
            ok = (real_fills[:, d] >= 30) & (fair_fills[:, d] >= 1)
            wr, wf = real_fills[:, d][ok], fair_fills[:, d][ok]
            print(f"  {name}: real fills {real_fills[:, d].sum():.0f} vs fair {fair_fills[:, d].sum():.0f} per replication; "
                  f"fill-weighted net real {np.average(real_mean[:, d][ok], weights=wr):+.2f}c vs fair {np.average(fair[:, d][ok], weights=wf):+.2f}c")
        report[n_steps]["fair_mean_net_pooled"] = np.nan_to_num(fair).round(2).tolist()
        report[n_steps]["fair_fills_pooled"] = fair_fills.round(1).tolist()
    (OUT/"null.json").write_text(json.dumps(report, indent=1))


# ---------------------------------------------------------------- cuts

def candidates():
    """Cells that pass archive + recent in any unit (realistic fees), survivors flagged."""
    seen = json.loads((OUT/"map-pass-counts.json").read_text())
    out = []
    for unit, r in seen.items():
        for p in r["realistic fees"]["passing"]:
            out.append((unit, p["band"], p["direction"], p["level"], p["neighbours"]))
    return out


def select(f, unit, band, direction, level):
    return f[f.tour.isin(UNITS[unit]) & (f.band == BAND_NAMES.index(band)) & (f.dir == DIRS.index(direction)) & (f.li == list(LEVELS).index(level))]


def do_cuts():
    m, f = load()
    f = f.assign(stage=np.where(f.qualifying.isna(), "unknown", np.where(f.qualifying == True, "qualifying", "main draw")),   # noqa: E712
                 when=pd.cut(f.minutes, [-1, 20, 60, 1e6], labels=["0-20 min", "20-60 min", "60+ min"]).astype(str),
                 speed=np.where(f.fast, "fast", "slow"))
    print("\nCUTS  cells passing archive + recent (realistic fees); net cents per fill with bootstrap intervals")
    rows = []
    for unit, band, direction, level, neighbours in candidates():
        g = select(f, unit, band, direction, level)
        print(f" {unit} | P0 {band} | {direction} L={level}{' | SURVIVOR' if neighbours else ''}")
        for name, key in (("all", None), ("round", "stage"), ("time since start", "when"), ("move into the level", "speed")):
            for value in ([None] if key is None else sorted(g[key].unique())):
                sub = g if key is None else g[g[key] == value]
                parts = [f"{p}: {brief(sub[sub.period == p].net.values)}" for p in PERIODS]
                print(f"    {name+(' = '+str(value) if value else ''):32} {parts[0]} | {parts[1]} | both: {brief(sub.net.values)}")
                rows.append(dict(unit=unit, band=band, direction=direction, level=level, cut=name, value=value,
                                 **{f"{p}_{k}": v for p in PERIODS for k, v in
                                    (("fills", int((sub.period == p).sum())), ("net", sub[sub.period == p].net.mean()))}))
    pd.DataFrame(rows).to_csv(OUT/"cuts.csv", index=False, float_format="%.3f")
    # The same three cuts over the whole map, so that a cut seen in a surviving cell can be compared with its base rate.
    print(" the same cuts over ALL resting-bid fills and ALL buy-on-strength fills (every tour, band and level)")
    for d, name in enumerate(DIRS):
        for key in ("stage", "when", "speed"):
            for value in sorted(f[key].unique()):
                sub = f[(f.dir == d) & (f[key] == value)]
                print(f"    {name:4} {key:6} = {value:10} archive {sub[sub.period == 'archive'].net.mean():+6.2f} ({(sub.period == 'archive').sum():6}) "
                      f"recent {sub[sub.period == 'recent'].net.mean():+6.2f} ({(sub.period == 'recent').sum():6})")


# ---------------------------------------------------------------- special structures

def line(label, g, col="net"):
    parts = [brief(g[g.period == p][col].values) for p in PERIODS]
    return f"    {label:46} archive {parts[0]} | recent {parts[1]} | both {brief(g[col].values)}"


def do_special():
    m, f = load()
    pl = players(m)
    report = {}
    groups = (("ALL", TOUR_NAMES), ("Chall+WTA+WTA Chall", UNITS["Chall+WTA+WTA Chall"]), ("ITF both", UNITS["ITF both"]), ("ATP", ("ATP",)))
    print("\nSPECIAL (a) UNDERDOGS: pre-match 10-35c")
    dog = f[(f.p0p >= 10) & (f.p0p < 35)]
    for unit, tours in groups:
        print(f"  {unit}")
        placed = pl[pl.tour.isin(tours) & (pl.p0p >= 10) & (pl.p0p < 35)]
        for direction, levels in ((1, (50, 60, 70)), (0, (5, 10))):
            for level in levels:
                g = dog[dog.tour.isin(tours) & (dog.dir == direction) & (dog.li == list(LEVELS).index(level))]
                n = {p: int(((placed.period == p) & ((placed.ask < level) if direction else (placed.bid >= level))).sum()) for p in PERIODS}
                label = f"{'buy on strength at' if direction else 'lottery bid at'} {level}c (placed {n['archive']}/{n['recent']})"
                print(line(label, g))
                report[f"a | {unit} | {DIRS[direction]} {level}"] = {p: dict(placed=n[p], fills=int((g.period == p).sum()),
                                                                          net=float(g[g.period == p].net.mean()) if (g.period == p).any() else None,
                                                                          ci=boot(g[g.period == p].net.values)) for p in PERIODS}
    print("  lottery bids, all pre-match prices (down L=5, L=10), all tours: won share against price")
    for level in (5, 10):
        g = f[(f.dir == 0) & (f.li == list(LEVELS).index(level))]
        print(line(f"bid {level}c, any P0 (won {100*g.payout.mean():.1f}% of {len(g)})", g))

    coin = pd.read_pickle(OUT/"coin.pkl").join(m[["tour", "period", "p0", "outcome", "qualifying"]], on="mid")
    coin["p0p"] = np.where(coin.player == 0, coin.p0, 100-coin.p0)
    coin["payout"] = np.where(coin.player == 0, coin.outcome, 1-coin.outcome)
    coin["net"] = 100*coin.payout-coin.level-np.where(coin.tour.isin(NO_MAKER_FEE), 0.0, maker_fee(coin.level))
    coin = coin[(coin.p0p >= 35) & (coin.p0p <= 65)]
    print("\nSPECIAL (b) COIN-FLIP MATCHES (P0 35-65): a player's bid reaches H, then a resting bid at L")
    for unit, tours in groups:
        print(f"  {unit}")
        for high in (80, 85, 90):
            for level in (65, 70, 75):
                sub = coin[coin.tour.isin(tours) & (coin.high == high) & (coin.level == level)]
                g = sub[sub.filled]
                n = {p: int((sub.period == p).sum()) for p in PERIODS}
                print(line(f"reach {high}, bid {level} (reached {n['archive']}/{n['recent']})", g))
                report[f"b | {unit} | reach {high} bid {level}"] = {p: dict(reached=n[p], fills=int((g.period == p).sum()),
                                                                           net=float(g[g.period == p].net.mean()) if (g.period == p).any() else None,
                                                                           ci=boot(g[g.period == p].net.values)) for p in PERIODS}
    print("  control: the plain first crossing down through L for the same players (no requirement to have reached H)")
    for level in (65, 70, 75):
        g = f[(f.dir == 0) & (f.p0p >= 35) & (f.p0p <= 65) & (f.li == list(LEVELS).index(level))]
        print(line(f"P0 35-65, plain bid {level}", g))

    re = pd.read_pickle(OUT/"reentry.pkl").join(m[["tour", "period", "p0", "outcome"]], on="mid")
    re["p0p"] = np.where(re.player == 0, re.p0, 100-re.p0)
    re["payout"] = np.where(re.player == 0, re.outcome, 1-re.outcome)
    re["net"] = 100*re.payout-80-np.where(re.tour.isin(NO_MAKER_FEE), 0.0, maker_fee(80))
    print("\nSPECIAL (c) RE-ENTRY: dipped below 80, recovered to a 90 bid, dips below 80 again (80c bid each time)")
    for lo, hi in ((90, 101), (80, 90)):
        for unit, tours in groups:
            sub = re[re.tour.isin(tours) & (re.p0p >= lo) & (re.p0p < hi)]
            print(f"  P0 {lo}-{min(hi, 100)} | {unit}")
            for label, g in (("first dip, all", sub), ("first dip, later recovered to 90 (hindsight)", sub[sub.recovered]),
                             ("first dip, never recovered to 90 (hindsight)", sub[~sub.recovered]),
                             ("SECOND dip after the recovery", sub[sub.second]),
                             ("recovered and never dipped again (hindsight)", sub[sub.recovered & ~sub.second])):
                print(line(label, g))
                report[f"c | P0 {lo}-{hi} | {unit} | {label}"] = {p: dict(fills=int((g.period == p).sum()),
                                                                         net=float(g[g.period == p].net.mean()) if (g.period == p).any() else None,
                                                                         ci=boot(g[g.period == p].net.values)) for p in PERIODS}
    (OUT/"special.json").write_text(json.dumps(report, indent=1, default=float))


# ---------------------------------------------------------------- rows

def do_rows():
    """Context for the favourite rows: the pre-match gap by band, the rows by tour, and bid ladders per player placed."""
    m, f = load()
    pl = players(m).join(m[["outcome"]], on="mid")
    pl["gap"] = 100*np.where(pl.player == 0, pl.outcome, 1-pl.outcome)-pl.p0p
    report = {}
    print("\nROWS  pre-match gap (settlement value minus P0, cents) by P0 band")
    for unit, tours in UNITS.items():
        for period in PERIODS:
            sub = pl[pl.tour.isin(tours) & (pl.period == period)]
            cells = []
            for b in range(3, len(BANDS)):
                g = sub[sub.band == b].gap.values
                lo, hi = boot(g, 1000)
                cells.append(f"{BAND_NAMES[b]} {g.mean() if len(g) else float('nan'):+5.2f} [{lo:+5.1f},{hi:+5.1f}] ({len(g)})")
                report[f"gap | {unit} | {period} | {BAND_NAMES[b]}"] = dict(n=len(g), gap=float(g.mean()) if len(g) else None, ci=[lo, hi])
            print(f"  {unit:20} {period:8} "+" | ".join(cells))
    print(" resting bids on favourites by tour (net cents per fill)")
    for unit in TOUR_NAMES:
        for band, levels in (("80-90", (50, 70)), ("90+", (50, 70, 80, 85))):
            for level in levels:
                g = select(f, unit, band, "down", level)
                print(f"  {unit:15} P0 {band:6} L={level} archive {brief(g[g.period == 'archive'].net.values)} | recent {brief(g[g.period == 'recent'].net.values)}")
    print(" ladders on Chall+WTA+WTA Chall: one contract at each level, cents per player placed (unfilled players count as zero)")
    tours = UNITS["Chall+WTA+WTA Chall"]
    fu = f[f.tour.isin(tours) & (f.dir == 0)]
    for band, levels in ((6, (80,)), (6, (85,)), (6, (85, 80, 75, 70)), (5, (70,)), (5, (50,)), (5, (70, 60, 50))):
        for stage in ("all rounds", "qualifying"):
            for period in PERIODS:
                keep = (pl.tour.isin(tours)) & (pl.band == band) & (pl.period == period)
                g = fu[(fu.band == band) & (fu.period == period) & fu.paid.isin(levels)]
                if stage == "qualifying":
                    keep, g = keep & (pl.qualifying == True), g[g.qualifying == True]   # noqa: E712
                total = g.groupby([g.mid, g.player]).net.sum().values
                per = np.zeros(int(keep.sum()))
                per[:len(total)] = total
                lo, hi = boot(per)
                fills = {int(level): int((g.paid == level).sum()) for level in levels}
                report[f"ladder | {BAND_NAMES[band]} | {levels} | {stage} | {period}"] = dict(players=len(per), per_player=float(per.mean()), ci=[lo, hi], fills=fills)
                print(f"  P0 {BAND_NAMES[band]:6} bids {str(levels):18} {stage:10} {period:8} players {len(per):5} fills {fills} "
                      f"per player {per.mean():+6.2f} [{lo:+6.2f},{hi:+6.2f}] worst {per.min():+.0f}")
    (OUT/"rows.json").write_text(json.dumps(report, indent=1, default=float))



# ---------------------------------------------------------------- rank

def shrink(f, unit, direction):
    """Empirical-Bayes shrinkage of pooled cell nets toward the unit's mean for that direction (DerSimonian-Laird)."""
    fu = f[f.tour.isin(UNITS[unit]) & (f.dir == direction)]
    n, s, ss = sums(fu, "net")
    n, s, ss = n.sum(0)[:, direction], s.sum(0)[:, direction], ss.sum(0)[:, direction]
    mean, se = moments(n, s, ss)
    ok = (n >= MIN_NEIGHBOUR) & (se > 0)
    w = 1/se[ok]**2
    centre = (w*mean[ok]).sum()/w.sum()
    q = (w*(mean[ok]-centre)**2).sum()
    tau2 = max(0.0, (q-(ok.sum()-1))/(w.sum()-(w**2).sum()/w.sum()))
    with np.errstate(divide="ignore", invalid="ignore"):
        out = centre+tau2/(tau2+se**2)*(mean-centre)
    return out, centre, tau2**.5, mean, se


def do_rank():
    m, f = load()
    print("\nRANK  candidate rules: cells passing archive + recent, with shrunk forward expectations")
    rows = []
    for unit, band, direction, level, neighbours in candidates():
        d, b, li = DIRS.index(direction), BAND_NAMES.index(band), list(LEVELS).index(level)
        g = select(f, unit, band, direction, level)
        est, centre, tau, mean, se = shrink(f, unit, d)
        arch, rec = g[g.period == "archive"], g[g.period == "recent"]
        rows.append(dict(unit=unit, band=band, direction=direction, level=level, survivor=neighbours,
                         archive_fills=len(arch), archive_net=arch.net.mean(), archive_ci=boot(arch.net.values),
                         recent_fills=len(rec), recent_net=rec.net.mean(), recent_ci=boot(rec.net.values),
                         pooled_net=g.net.mean(), shrunk=float(est[b, li]), unit_centre=float(centre), between_cell_sd=float(tau),
                         fills_per_month_recent=len(rec)/RECENT_MONTHS, fills_per_month_archive=len(arch)/ARCHIVE_MONTHS,
                         median_fill_bar_volume=float(g.bar_volume.median()), p25_fill_bar_volume=float(g.bar_volume.quantile(.25)),
                         cents_per_month_at_shrunk=float(est[b, li])*len(rec)/RECENT_MONTHS))
    rows.sort(key=lambda r: -r["cents_per_month_at_shrunk"])
    for r in rows:
        print(f" {r['unit']:20} P0 {r['band']:6} {r['direction']:4} L={r['level']:2} {'SURVIVOR' if r['survivor'] else '        '} "
              f"archive {r['archive_fills']:4} {r['archive_net']:+6.2f} [{r['archive_ci'][0]:+.1f},{r['archive_ci'][1]:+.1f}] | "
              f"recent {r['recent_fills']:4} {r['recent_net']:+6.2f} [{r['recent_ci'][0]:+.1f},{r['recent_ci'][1]:+.1f}] | "
              f"shrunk {r['shrunk']:+5.2f} (unit mean {r['unit_centre']:+.2f}, cell sd {r['between_cell_sd']:.2f}) | "
              f"{r['fills_per_month_recent']:5.1f} fills/month | fill-bar volume median {r['median_fill_bar_volume']:.0f} (p25 {r['p25_fill_bar_volume']:.0f})")
    (OUT/"rank.json").write_text(json.dumps(rows, indent=1, default=float))
    print(" between-cell spread of true edge by unit and direction (0 = the map is indistinguishable from one common value)")
    for unit in UNITS:
        for d, name in enumerate(DIRS):
            _, centre, tau, _, _ = shrink(f, unit, d)
            print(f"   {unit:20} {name:4} common value {centre:+.2f}c, between-cell sd {tau:.2f}c")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("steps", nargs="+", choices=("build", "map", "null", "cuts", "special", "rows", "rank"))
    parser.add_argument("--reps", type=int, default=200, help="fair-market replications for `null`")
    parser.add_argument("--walk-steps", type=int, nargs="+", default=[700], help="steps per simulated match")
    args = parser.parse_args()
    for step in args.steps:
        if step == "null":
            do_null(args.reps, args.walk_steps)
        else:
            dict(build=build, map=do_map, cuts=do_cuts, special=do_special, rows=do_rows, rank=do_rank)[step]()
