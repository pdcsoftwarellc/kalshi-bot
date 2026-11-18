"""Do Kalshi tennis match-winner prices mean-revert in play? Martingale tests, microstructure checks, score-aware checks.

A "buy the dip, sell the rebound" loop can only pay if a fall in the in-play
price tends to be followed by a rise that the price does not already contain.
This script measures that directly, with no trading rule in mind.

Data
* One-minute bars cached under data/raw/cost-research-v1 (the six hours before
  settlement, the alphabetically-first market of each event; the second market
  where an earlier study cached it) for ATP, WTA, ATP Challenger, WTA
  Challenger, ITF men and ITF women. Archive = settled before 2026-09-18,
  recent = after. Nothing is fetched for the bars (the HTTP client is a stub).
* The public trade record (/markets/trades) for a sample of recent matches,
  downloaded by step `fetch`: one request at a time, one per second, long pause
  on HTTP 429, time-capped.
* The live_games.py recording (data/raw/live-games/*.jsonl): Kalshi's point
  score beside the listing quote, polled about every 25 seconds.

Definitions
* P0 = median midpoint of the first 15 bars (two-sided, at most 4c wide, within
  3c): favorite_dip.py. Favourite = the side with P0 above 50. Bands: "even"
  (favourite under 60, where both players are studied by symmetry), 60-70,
  70-80, 80-90, 90+.
* Start of play = favourite_gap.detect_start. The in-play path runs on a regular
  one-minute grid from the detection bar to the last bar with a two-sided quote
  (0 < bid <= ask < 100). Minutes with no bar carry the last quote forward (and
  are flagged). One terminal point is appended holding the settlement value, so
  a change that runs past the end of the path ends at settlement: under the
  null "price = fair probability" every change, including that one, has mean
  zero given the past. All prices are the favourite's, in cents.
* (a) slope of the next-h change on the last-h change, intervals clustered by
  match. Variants: `raw` (shares the midpoint at t, so a noisy or stale quote
  at t fakes reversion); `skip` (next change measured from t+1); `tight`
  (skip, and the quotes at t-h, t, t+1 are at most 2c wide and t is a real
  bar); `other` (next change measured on the OTHER market's book).
* (b) variance ratio: sum of squared h-minute changes over the sum of the
  squared one-minute changes inside the same windows; 1 for a martingale,
  below 1 = reversion. Bootstrap over matches.
* (c) events: the first bar where the favourite's midpoint is X cents (5, 10,
  15, 20) or more below (above) its level five minutes earlier, then a
  five-bar pause. Mean midpoint change over the next 1, 5, 15, 30 minutes and
  to settlement. Tradeable: buy at the NEXT bar's opening ask, sell at the
  closing bid k minutes later, taker fee 0.07*p*(1-p) each side (no exit fee
  at settlement). Maker: a bid resting at the closing bid of the event bar for
  five bars, filled by a print strictly below it; exits at the bid (taker) or
  by an offer 3c higher resting 30 bars (print strictly above), else the bid.
  Maker fee zero on Challenger / WTA Challenger / ITF, 0.0175*p*(1-p) assumed
  on ATP / WTA.

    .venv/bin/python research/tennis_reversion.py build
    .venv/bin/python research/tennis_reversion.py martingale events
    .venv/bin/python research/tennis_reversion.py fetch --minutes 18
    .venv/bin/python research/tennis_reversion.py trades scores

Two traps this script avoids, both of which manufacture reversion:
* Survivors. Keeping only windows that fit inside the quoted path (the match
  did not end in the next h minutes), or only events with a trade print k
  minutes later, drops the cases where the price ran away. The variance ratio
  falls from about 0.98-1.04 to 0.83-0.90 on that selection alone.
* Stale quotes. A midpoint that has not finished repricing looks like it
  "recovers" (or keeps falling) a minute later; hence `skip`, `tight`, the
  other book and the trade prints. In the score recording the first quote read
  beside a new point score has made only about two thirds of its move.

Outputs under data/processed/tennis-reversion/. Public market data only; no
orders, no authentication.
"""
import argparse
from collections import defaultdict
import json
from pathlib import Path
import sys
import time

import httpx
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from research.api_cost import ROOT, fetch, timestamp
from research.dip_archive_tours import cached, paced
from research.favorite_dip import WARMUP
from research.favourite_gap import STUB, TOURS, bar_params, detect_start, listing, side_path
from research.near_end import cents

OUT = ROOT / "data/processed/tennis-reversion"
LIVE = ROOT / "data/raw/live-games"
GROUPS = {"ATP+WTA": ("ATP", "WTA"), "Challenger": ("Challenger", "WTA Challenger"), "ITF": ("ITF men", "ITF women"),
          "ALL": tuple(TOURS.values())}
NO_MAKER_FEE = ("Challenger", "WTA Challenger", "ITF men", "ITF women")
PERIODS = ("archive", "recent")
BANDS = (("even", 50, 60), ("60-70", 60, 70), ("70-80", 70, 80), ("80-90", 80, 90), ("90+", 90, 101))
HORIZONS = (1, 2, 5, 10, 20, 30)
MOVES = (5, 10, 15, 20)
AFTER = (1, 5, 15, 30)
RNG = np.random.default_rng(23)


def fee(p, rate=7.0):
    """Fee in cents at price p cents."""
    return rate*p/100*(1-p/100)


# ---------------------------------------------------------------- build

def quote_arrays(bars):
    f = lambda side, field: np.array([np.nan if (v := cents(b, side, field)) is None else v for b in bars], float)
    return f("yes_bid", "close"), f("yes_ask", "close"), f("yes_bid", "open"), f("yes_ask", "open")


def on_grid(ts, values, t0, n, fill):
    """Place per-bar values on the minute grid starting at t0; forward-fill when `fill`."""
    out = np.full(n, np.nan)
    k = ((ts-t0)//60).astype(int)
    before = k < 0
    seed = values[before][-1] if before.any() else np.nan
    ok = (k >= 0) & (k < n)
    out[k[ok]] = values[ok]
    seen = np.zeros(n, bool)
    seen[k[ok]] = True
    if fill:
        last = seed
        for i in range(n):
            if seen[i]:
                last = out[i]
            else:
                out[i] = last
    return out, seen


def build():
    cols = defaultdict(list)
    matches, seen = [], set()
    with httpx.Client(transport=STUB) as client:
        for archived in (True, False):
            for series, tour in TOURS.items():
                events, path, lo, hi = listing(client, series, archived)
                kept = 0
                for event, group in events.items():
                    group = sorted(group, key=lambda m: m["ticker"])
                    m = group[0]
                    try:
                        outcome, settle = float(m["settlement_value_dollars"]), int(timestamp(m["settlement_ts"]))
                    except (KeyError, TypeError, ValueError):
                        continue
                    if not lo <= settle < hi or event in seen:
                        continue
                    params = bar_params(settle)
                    if not cached(path.format(t=m["ticker"]), params):
                        continue
                    bars = fetch(client, path.format(t=m["ticker"]), params).get("candlesticks")
                    if not bars:
                        continue
                    seen.add(event)
                    bars = sorted(bars, key=lambda b: b["end_period_ts"])
                    bid, ask, obid, oask = quote_arrays(bars)
                    warm = (bid[:WARMUP]+ask[:WARMUP])/2
                    if len(bars) <= WARMUP or np.isnan(warm).any() or not ((bid[:WARMUP] > 0) & (bid[:WARMUP] <= ask[:WARMUP]) & (ask[:WARMUP] < 100)
                                                                             & (ask[:WARMUP]-bid[:WARMUP] <= 4)).all() or warm.max()-warm.min() > 3:
                        continue
                    sp = side_path(bars, True)
                    began = detect_start(sp)
                    two = (bid > 0) & (bid <= ask) & (ask < 100)
                    if began is None or not two[began+1:].any():
                        continue
                    last = int(np.where(two)[0][-1])
                    ts = np.array([b["end_period_ts"] for b in bars], float)
                    n = int((ts[last]-ts[began])//60)+1
                    for a in (bid, ask, obid, oask):
                        a[~two] = np.nan                                   # a one-sided or empty quote is no price
                    g = {}
                    g["bid"], obs = on_grid(ts, bid, ts[began], n, True)
                    g["ask"], _ = on_grid(ts, ask, ts[began], n, True)
                    ob, _ = on_grid(ts, obid, ts[began], n, False)
                    oa, _ = on_grid(ts, oask, ts[began], n, False)
                    prev_b, prev_a = np.r_[np.nan, g["bid"][:-1]], np.r_[np.nan, g["ask"][:-1]]
                    g["obid"], g["oask"] = np.where(obs, ob, prev_b), np.where(obs, oa, prev_a)
                    low = np.array([np.nan if p[3] is None else p[3] for p in sp], float)
                    high = np.array([np.nan if p[4] is None else p[4] for p in sp], float)
                    vol = np.array([p[5] for p in sp], float)
                    g["low"], _ = on_grid(ts, low, ts[began], n, False)
                    g["high"], _ = on_grid(ts, high, ts[began], n, False)
                    g["vol"], _ = on_grid(ts, vol, ts[began], n, False)
                    g["vol"] = np.nan_to_num(g["vol"])
                    g["obs"] = obs.astype(float)
                    # The other market of the event, read as this market's YES side.
                    g["bid2"], g["ask2"] = np.full(n, np.nan), np.full(n, np.nan)
                    other = group[1]["ticker"] if len(group) == 2 else None
                    has_other = False
                    if other and cached(path.format(t=other), params):
                        ob2 = fetch(client, path.format(t=other), params).get("candlesticks") or []
                        if ob2:
                            ob2 = sorted(ob2, key=lambda b: b["end_period_ts"])
                            b2, a2, _, _ = quote_arrays(ob2)
                            bad = ~((b2 > 0) & (b2 <= a2) & (a2 < 100))
                            b2[bad], a2[bad] = np.nan, np.nan
                            ts2 = np.array([b["end_period_ts"] for b in ob2], float)
                            g["bid2"], _ = on_grid(ts2, 100-a2, ts[began], n, True)
                            g["ask2"], _ = on_grid(ts2, 100-b2, ts[began], n, True)
                            has_other = True
                    mid = len(matches)
                    final = 100*outcome
                    for key, a in g.items():                               # terminal point: the settlement value
                        end = final if key in ("bid", "ask", "obid", "oask", "bid2", "ask2") else (0.0 if key in ("vol", "obs") else np.nan)
                        cols[key].append(np.r_[a, end].astype(np.float32))
                    cols["m"].append(np.full(n+1, mid, np.int32))
                    cols["term"].append(np.r_[np.zeros(n, bool), True])
                    matches.append(dict(mid=mid, event=event, ticker=m["ticker"], other_ticker=other, has_other=has_other, tour=tour,
                                        period=PERIODS[0] if archived else PERIODS[1], settle=settle, start_ts=float(ts[began]),
                                        p0=float(np.median(warm)), outcome=outcome, n=n,
                                        volume=sum(float(x.get("volume_fp") or x.get("volume") or 0) for x in group)))
                    kept += 1
                print(f"  {tour:15} {'archive' if archived else 'recent':8} {kept:6} matches", flush=True)
    OUT.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(matches).to_pickle(OUT/"matches.pkl")
    np.savez_compressed(OUT/"grid.npz", **{k: np.concatenate(v) for k, v in cols.items()})
    print(f"BUILD {len(matches)} matches, {sum(len(v) for v in cols['m'])} grid points")


# ---------------------------------------------------------------- shared

class Data:
    """The grid in the favourite's terms."""

    def __init__(self):
        self.M = pd.read_pickle(OUT/"matches.pkl")
        z = np.load(OUT/"grid.npz")
        self.g = z["m"]
        yes = (self.M.p0.values > 50)[self.g]
        flip = lambda a, b: np.where(yes, a, 100-b).astype(np.float64)
        self.bid, self.ask = flip(z["bid"], z["ask"]), flip(z["ask"], z["bid"])
        self.obid, self.oask = flip(z["obid"], z["oask"]), flip(z["oask"], z["obid"])
        self.bid2, self.ask2 = flip(z["bid2"], z["ask2"]), flip(z["ask2"], z["bid2"])
        self.low, self.high = flip(z["low"], z["high"]), flip(z["high"], z["low"])
        self.vol, self.obs, self.term = z["vol"].astype(float), z["obs"] > 0, z["term"]
        self.mid, self.mid2 = (self.bid+self.ask)/2, (self.bid2+self.ask2)/2
        self.spread, self.spread2 = self.ask-self.bid, self.ask2-self.bid2
        n = len(self.g)
        change = np.r_[True, self.g[1:] != self.g[:-1]]
        self.first = np.where(change)[0][self.g]                       # index of the match's first grid point
        self.end = (np.r_[np.where(change)[0][1:], n]-1)[self.g]       # index of its terminal point
        self.i = np.arange(n)
        M = self.M
        self.fav0 = np.maximum(M.p0.values, 100-M.p0.values)
        self.payout = np.where(M.p0.values > 50, M.outcome.values, 1-M.outcome.values)
        self.band = np.array([next(k for k, (_, lo, hi) in enumerate(BANDS) if lo <= v < hi) for v in self.fav0])
        self.period = (M.period.values == "recent").astype(int)
        self.tour = M.tour.values
        self.maker_rate = np.where(np.isin(self.tour, NO_MAKER_FEE), 0.0, 1.75)

    def fwd(self, i, h):
        return np.minimum(i+h, self.end[i])

    def cells(self):
        """(group, period, band name, boolean mask over matches)."""
        for group, tours in GROUPS.items():
            in_group = np.isin(self.tour, tours)
            for p, period in enumerate(PERIODS):
                for b, name in list(enumerate(n for n, _, _ in BANDS))+[(None, "all")]:
                    yield group, period, name, in_group & (self.period == p) & ((self.band == b) if b is not None else True)


def cluster_mean(v, g):
    """n, clusters, mean, 95% interval with errors clustered by g."""
    ok = ~np.isnan(v)
    v, g = v[ok], g[ok]
    if len(v) < 5:
        return dict(n=len(v), matches=len(set(g)), mean=np.nan, lo=np.nan, hi=np.nan)
    mean = v.mean()
    _, inv = np.unique(g, return_inverse=True)
    sums = np.bincount(inv, v-mean)
    k = len(sums)
    se = np.sqrt((sums**2).sum()*k/max(k-1, 1))/len(v)
    return dict(n=len(v), matches=k, mean=mean, lo=mean-1.96*se, hi=mean+1.96*se)


def cluster_slope(x, y, g):
    ok = ~(np.isnan(x) | np.isnan(y))
    x, y, g = x[ok], y[ok], g[ok]
    if len(x) < 50:
        return dict(n=len(x), matches=len(set(g)), slope=np.nan, lo=np.nan, hi=np.nan)
    xc, yc = x-x.mean(), y-y.mean()
    sxx = (xc*xc).sum()
    slope = (xc*yc).sum()/sxx
    _, inv = np.unique(g, return_inverse=True)
    score = np.bincount(inv, xc*(yc-slope*xc))
    k = len(score)
    se = np.sqrt((score**2).sum()*k/max(k-1, 1))/sxx
    return dict(n=len(x), matches=k, slope=slope, lo=slope-1.96*se, hi=slope+1.96*se, sd_x=float(x.std()))


# ---------------------------------------------------------------- (a) and (b)

def martingale():
    d = Data()
    live = ~d.term
    rows = []
    level = (d.mid >= 5) & (d.mid <= 95)
    print("\n(a) SLOPE of the next-h change on the last-h change (favourite's midpoint, cents); negative = reversion")
    store = {}
    for h in HORIZONS:
        back = d.i-h
        ok = live & (back >= d.first) & level
        i = d.i[ok]
        x = d.mid[i]-d.mid[i-h]
        nxt = d.fwd(i, 1)
        y_raw = d.mid[d.fwd(i, h)]-d.mid[i]
        y_skip = d.mid[d.fwd(nxt, h)]-d.mid[nxt]
        tight = (d.spread[i] <= 2) & (d.spread[i-h] <= 2) & d.obs[i] & (d.spread[nxt] <= 2)
        m2_i, m2_f = d.mid2[i], d.mid2[d.fwd(i, h)]
        y_other = np.where((d.spread2[i] <= 4) & (d.spread2[d.fwd(i, h)] <= 4), m2_f-m2_i, np.nan)
        y_other_skip = np.where((d.spread2[nxt] <= 4) & (d.spread2[d.fwd(nxt, h)] <= 4), d.mid2[d.fwd(nxt, h)]-d.mid2[nxt], np.nan)
        store[h] = (i, x, y_raw, y_skip, tight, y_other, y_other_skip)
        for group, period, band, mask in d.cells():
            sel = mask[d.g[i]]
            gi = d.g[i][sel]
            for name, yy, extra in (("raw", y_raw, None), ("skip", y_skip, None), ("tight", y_skip, tight),
                                    ("other", y_other, None), ("other_skip", y_other_skip, None)):
                s = sel if extra is None else sel & extra
                r = cluster_slope(x[s], yy[s], d.g[i][s])
                rows.append(dict(test="slope", group=group, period=period, band=band, h=h, variant=name, **r))
            # the same matches for own-book vs other-book comparison
            both = sel & ~np.isnan(y_other)
            r = cluster_slope(x[both], y_raw[both], d.g[i][both])
            rows.append(dict(test="slope", group=group, period=period, band=band, h=h, variant="raw_on_other_sample", **r))
        # by price level at t (all tours)
        for lo, hi in ((5, 30), (30, 50), (50, 70), (70, 85), (85, 95.01)):
            for p, period in enumerate(PERIODS):
                s = (d.mid[i] >= lo) & (d.mid[i] < hi) & (d.period[d.g[i]] == p)
                for name, yy, extra in (("raw", y_raw, None), ("skip", y_skip, None), ("tight", y_skip, tight)):
                    ss = s if extra is None else s & extra
                    rows.append(dict(test="slope_by_level", group="ALL", period=period, band=f"price {lo}-{int(hi)}", h=h, variant=name,
                                     **cluster_slope(x[ss], yy[ss], d.g[i][ss])))
    S = pd.DataFrame(rows)

    def show(group, variants=("raw", "skip", "tight")):
        for variant in variants:
            print(f"  {group} | {variant}: slope [95% interval] by horizon (minutes); obs at h=5")
            for band in [n for n, _, _ in BANDS]+["all"]:
                for period in PERIODS:
                    q = S[(S.test == "slope") & (S.group == group) & (S.variant == variant) & (S.band == band) & (S.period == period)].set_index("h")
                    cells = " ".join(f"{q.slope[h]:+.3f}[{q.lo[h]:+.3f},{q.hi[h]:+.3f}]" for h in HORIZONS)
                    print(f"   {band:6} {period:8} {cells}  n={int(q.n[5])} m={int(q.matches[5])}")
    for group in GROUPS:
        show(group)
    print("  other book (follow-up leg measured on the second market): ALL tours, band all")
    for variant in ("raw_on_other_sample", "other", "other_skip"):
        for period in PERIODS:
            q = S[(S.test == "slope") & (S.group == "ALL") & (S.variant == variant) & (S.band == "all") & (S.period == period)].set_index("h")
            print(f"   {variant:20} {period:8} "+" ".join(f"{q.slope[h]:+.3f}[{q.lo[h]:+.3f},{q.hi[h]:+.3f}]" for h in HORIZONS)+f"  n={int(q.n[5])} m={int(q.matches[5])}")
    print("  by price level at t, all tours")
    for variant in ("raw", "skip", "tight"):
        for band in S[S.test == "slope_by_level"].band.unique():
            for period in PERIODS:
                q = S[(S.test == "slope_by_level") & (S.variant == variant) & (S.band == band) & (S.period == period)].set_index("h")
                print(f"   {variant:5} {band:12} {period:8} "+" ".join(f"{q.slope[h]:+.3f}[{q.lo[h]:+.3f},{q.hi[h]:+.3f}]" for h in HORIZONS))

    # (b) variance ratios, in-path (terminal excluded) and with the settlement point
    print("\n(b) VARIANCE RATIO of h-minute to one-minute changes (1 = martingale, <1 = reversion); bootstrap over matches")
    d1 = np.r_[np.nan, np.diff(d.mid)]
    d1[d.i == d.first] = np.nan                                        # no change into a match's first point
    vrows = []
    usable = ~np.isnan(d1)
    sq = np.where(usable, d1**2, 0.0)
    c_sq, c_bad = np.r_[0, np.cumsum(sq)], np.r_[0, np.cumsum(~usable)]
    # "to settlement": every window starting at a live point with the price in 5-95; a window that runs past the end of
    # the path stops at the settlement point. "survivors only": windows that fit inside the quoted path, which
    # conditions on the match NOT ending in the next h minutes and so biases the ratio downwards (shown as a warning).
    for label in ("to settlement", "survivors only"):
        for h in HORIZONS[1:]:
            ok = live & ~np.isnan(d.mid) & (d.mid >= 5) & (d.mid <= 95)
            if label == "survivors only":
                ok &= d.i+h < d.end
            i = d.i[ok]
            j = d.fwd(i, h)
            ok2 = (c_bad[j+1]-c_bad[i+1]) == 0
            i, j = i[ok2], j[ok2]
            num, den = (d.mid[j]-d.mid[i])**2, c_sq[j+1]-c_sq[i+1]
            for group, period, band, mask in d.cells():
                if band not in ("all",) and group != "ALL":
                    continue
                sel = mask[d.g[i]]
                if sel.sum() < 100:
                    continue
                _, inv = np.unique(d.g[i][sel], return_inverse=True)
                A, B = np.bincount(inv, num[sel]), np.bincount(inv, den[sel])
                w = RNG.multinomial(len(A), np.full(len(A), 1/len(A)), size=300)
                boot = (w@A)/(w@B)
                vrows.append(dict(test="vr", kind=label, group=group, period=period, band=band, h=h, vr=A.sum()/B.sum(),
                                  lo=np.percentile(boot, 2.5), hi=np.percentile(boot, 97.5), matches=len(A)))
    V = pd.DataFrame(vrows)
    for kind in ("to settlement", "survivors only"):
        print(f"  {kind}")
        for (group, band, period), q in V[V.kind == kind].groupby(["group", "band", "period"], sort=False):
            q = q.set_index("h")
            print(f"   {group:10} {band:6} {period:8} "+" ".join(f"h{h}:{q.vr[h]:.3f}[{q.lo[h]:.3f},{q.hi[h]:.3f}]" for h in HORIZONS[1:])+f" m={q.matches.iloc[0]}")
    S.to_csv(OUT/"slopes.csv", index=False, float_format="%.5f")
    V.to_csv(OUT/"variance-ratios.csv", index=False, float_format="%.5f")


# ---------------------------------------------------------------- (c) events

def find_events(d):
    """First bar where the 5-minute change reaches X (per threshold and direction), then a 5-bar pause."""
    live = ~d.term
    back = d.i-5
    ok = live & (back >= d.first)
    ch = np.full(len(d.i), np.nan)
    ch[ok] = d.mid[ok]-d.mid[back[ok]]
    ch[~((d.mid >= 5) & (d.mid <= 95))] = np.nan
    out = []
    for X in MOVES:
        for sign, name in ((-1, "fall"), (1, "rise")):
            cand = np.where(sign*ch >= X)[0]
            last_g, last_i = -1, -99
            for i in cand:
                if d.g[i] != last_g or i-last_i > 5:
                    out.append((i, X, name))
                    last_g, last_i = d.g[i], i
                # a bar inside the pause is dropped, and does not extend the pause
    E = pd.DataFrame(out, columns=["i", "X", "dir"])
    E["move"] = ch[E.i.values]
    return E


def events():
    d = Data()
    E = find_events(d)
    i = E.i.values
    g = d.g[i]
    E["m"], E["tour"], E["period"], E["band"] = g, d.tour[g], np.array(PERIODS)[d.period[g]], np.array([n for n, _, _ in BANDS])[d.band[g]]
    E["minute"], E["price"], E["spread"], E["obs"] = i-d.first[i], d.mid[i], d.spread[i], d.obs[i]
    E["vol_bar"], E["vol_next"] = d.vol[i], d.vol[d.fwd(i, 1)]
    final = 100*d.payout[g]
    for k in AFTER:
        E[f"ch{k}"] = d.mid[d.fwd(i, k)]-d.mid[i]
        E[f"ch{k}_skip"] = d.mid[d.fwd(d.fwd(i, 1), k)]-d.mid[d.fwd(i, 1)]
        E[f"ch{k}_other"] = np.where((d.spread2[i] <= 4) & (d.spread2[d.fwd(i, k)] <= 4), d.mid2[d.fwd(i, k)]-d.mid2[i], np.nan)
    E["chS"], E["chS_skip"] = final-d.mid[i], final-d.mid[d.fwd(i, 1)]
    # Taker round trips, entered at the next bar's opening ask. `fav`: buy the favourite; `dog`: buy the other player.
    e = d.fwd(i, 1)
    can = ~d.term[e]
    for who in ("fav", "dog"):
        a = np.where(can, d.oask[e] if who == "fav" else 100-d.obid[e], np.nan)
        a[(a < 1) | (a > 99)] = np.nan
        E[f"{who}_ask"] = a
        E[f"{who}_entry_spread"] = np.where(can, d.oask[e]-d.obid[e], np.nan)
        for k in AFTER:
            x = d.fwd(e, k)
            b = d.bid[x] if who == "fav" else 100-d.ask[x]
            E[f"{who}_take{k}"] = np.where(d.term[x], b, b-fee(b))-a-fee(a)
        end = final if who == "fav" else 100-final
        E[f"{who}_takeS"] = end-a-fee(a)
        # Maker: bid at the closing bid of the event bar, five bars, strict print-through.
        L = d.bid[i] if who == "fav" else 100-d.ask[i]
        fill = np.full(len(i), -1)
        for j in range(5, 0, -1):
            x = d.fwd(i, j)
            low = d.low[x] if who == "fav" else 100-d.high[x]
            hit = ~d.term[x] & (x == i+j) & (low < L)
            fill[hit] = x[hit]
        got = (fill >= 0) & (L >= 1) & (L <= 99)
        f = np.where(got, fill, 0)
        mfee = fee(L, d.maker_rate[g])
        E[f"{who}_make_fill"] = got
        E[f"{who}_make_L"] = np.where(got, L, np.nan)
        nxt = d.fwd(f, 1)
        E[f"{who}_make_after1"] = np.where(got, (d.mid[nxt] if who == "fav" else 100-d.mid[nxt])-L, np.nan)
        for k in (5, 15, 30):
            x = d.fwd(f, k)
            b = d.bid[x] if who == "fav" else 100-d.ask[x]
            E[f"{who}_make_take{k}"] = np.where(got, np.where(d.term[x], b, b-fee(b))-L-mfee, np.nan)
        E[f"{who}_makeS"] = np.where(got, end-L-mfee, np.nan)
        for T in (3, 5, 10):                                           # resting offer T above the fill, 30 bars
            sold = np.zeros(len(i), bool)
            for j in range(1, 31):
                x = d.fwd(f, j)
                high = d.high[x] if who == "fav" else 100-d.low[x]
                sold |= ~d.term[x] & (x == f+j) & (high > L+T)
            x = d.fwd(f, 30)
            b = d.bid[x] if who == "fav" else 100-d.ask[x]
            miss = np.where(d.term[x], b, b-fee(b))-L-mfee
            hit = T-mfee-fee(L+T, d.maker_rate[g])
            E[f"{who}_make_offer{T}"] = np.where(got, np.where(sold, hit, miss), np.nan)
            E[f"{who}_make_offer{T}_sold"] = np.where(got, sold, np.nan)
    E.to_pickle(OUT/"events.pkl")
    report_events(E)


def stat(v, g, digits=2):
    r = cluster_mean(np.asarray(v, float), np.asarray(g))
    return "      n/a        " if np.isnan(r["mean"]) else f"{r['mean']:+6.{digits}f}[{r['lo']:+6.{digits}f},{r['hi']:+6.{digits}f}]"


def report_events(E=None):
    E = pd.read_pickle(OUT/"events.pkl") if E is None else E
    rows = []
    names = [n for n, _, _ in BANDS]
    measures = ([f"ch{k}" for k in AFTER]+["chS"]+[f"ch{k}_skip" for k in AFTER]+["chS_skip"]+[f"ch{k}_other" for k in AFTER]
                + [f"{w}_take{k}" for w in ("fav", "dog") for k in list(AFTER)+["S"]]
                + [f"{w}_make_{x}" for w in ("fav", "dog") for x in ("after1", "take5", "take15", "take30", "offer3", "offer5", "offer10")]+["fav_makeS", "dog_makeS"])
    for group, tours in GROUPS.items():
        for period in PERIODS:
            for band in names+["all"]:
                for X in MOVES:
                    for direction in ("fall", "rise"):
                        for flt in ("any", "tight"):
                            q = E[E.tour.isin(tours) & (E.period == period) & (E.X == X) & (E.dir == direction)]
                            if band != "all":
                                q = q[q.band == band]
                            if flt == "tight":
                                q = q[(q.spread <= 2) & q.obs & (q.fav_entry_spread <= 2)]
                            if len(q) < 20:
                                continue
                            row = dict(group=group, period=period, band=band, X=X, dir=direction, filter=flt, events=len(q), matches=q.m.nunique(),
                                       mean_move=q.move.mean(), price=q.price.mean(), spread_close=q.spread.mean(), spread_entry=q.fav_entry_spread.mean(),
                                       vol_bar_median=q.vol_bar.median(), vol_next_median=q.vol_next.median(),
                                       fav_fill_rate=q.fav_make_fill.mean(), dog_fill_rate=q.dog_make_fill.mean(),
                                       **{f"{w}_offer{T}_sold": q[f"{w}_make_offer{T}_sold"].mean() for w in ("fav", "dog") for T in (3, 5, 10)})
                            for c in measures:
                                r = cluster_mean(q[c].values.astype(float), q.m.values)
                                row.update({c: r["mean"], c+"_lo": r["lo"], c+"_hi": r["hi"], c+"_n": r["n"]})
                            rows.append(row)
    R = pd.DataFrame(rows)
    R.to_csv(OUT/"event-cells.csv", index=False, float_format="%.3f")

    def table(group, flt, cols, title):
        print(f"  {group} | {flt} | {title}")
        q = R[(R.group == group) & (R["filter"] == flt)]
        for direction in ("fall", "rise"):
            for band in names+["all"]:
                for X in MOVES:
                    line = []
                    for period in PERIODS:
                        r = q[(q.period == period) & (q.band == band) & (q.X == X) & (q.dir == direction)]
                        if r.empty:
                            line.append(f"{period}: -")
                            continue
                        r = r.iloc[0]
                        line.append(f"{period} n={int(r.events):5} "+" ".join(f"{c.split('_')[-1] if '_' in c else c}:{r[c]:+5.2f}[{r[c+'_lo']:+5.2f},{r[c+'_hi']:+5.2f}]" for c in cols))
                    print(f"   {direction} {X:2}c+ {band:6} "+" | ".join(line))
    print("\n(c) CONDITIONAL DRIFT after a move of X cents or more in five minutes (favourite's price; cents; intervals clustered by match)")
    for group in GROUPS:
        table(group, "any", ["ch1", "ch5", "ch15", "ch30", "chS"], "midpoint change after the event bar")
    table("ALL", "any", ["ch1_skip", "ch5_skip", "ch15_skip", "ch30_skip", "chS_skip"], "midpoint change measured from the NEXT bar (free of a stale quote at t)")
    table("ALL", "tight", ["ch1_skip", "ch5_skip", "ch15_skip", "ch30_skip", "chS_skip"], "tight quotes only, from the next bar")
    table("ALL", "any", ["ch1_other", "ch5_other", "ch15_other", "ch30_other"], "follow-up measured on the other market's book")
    print("  the other book is cached only for 90c+ favourites on Challenger / WTA / WTA Challenger, so compare like with like:")
    print("  the same events, follow-up on the market's own book vs on the other market's book (all periods)")
    for direction in ("fall", "rise"):
        for X in MOVES:
            q = E[(E.dir == direction) & (E.X == X) & E.ch5_other.notna() & E.ch15_other.notna()]
            if len(q) >= 20:
                print(f"   {direction} {X:2}c+ n={len(q):5} in {q.m.nunique():4} matches | 5m own {stat(q.ch5, q.m)} other {stat(q.ch5_other, q.m)} | "
                      f"15m own {stat(q.ch15, q.m)} other {stat(q.ch15_other, q.m)} | other minus own at the event bar's close: "
                      f"{stat((q.ch5-q.ch5_other), q.m)} (5m difference)")
    print("  TRADEABLE, taker: buy at the next bar's opening ask, sell at the bid k minutes later (S = hold to settlement), both fees")
    for group in GROUPS:
        table(group, "any", ["fav_take5", "fav_take15", "fav_take30", "fav_takeS"], "BUY THE FAVOURITE (fade a fall / follow a rise)")
        table(group, "any", ["dog_take5", "dog_take15", "dog_take30", "dog_takeS"], "BUY THE OTHER PLAYER (follow a fall / fade a rise)")
    print("  TRADEABLE, maker: bid at the event bar's closing bid, filled by a strict print-through within five bars")
    for group in GROUPS:
        table(group, "any", ["fav_make_after1", "fav_make_take15", "fav_make_offer3", "fav_makeS"], "bid for the favourite; after1 = midpoint one bar after the fill minus the bid")
        table(group, "any", ["dog_make_after1", "dog_make_take15", "dog_make_offer3", "dog_makeS"], "bid for the other player")
    print("  capacity and cost at the event (ALL tours and by group): spread at the event close, at entry, bar volume, maker fill rate")
    for group in GROUPS:
        q = R[(R.group == group) & (R["filter"] == "any") & (R.band == "all")]
        for _, r in q.iterrows():
            print(f"   {group:10} {r.period:8} {r['dir']} {int(r.X):2}c+ events {int(r.events):6} in {int(r.matches):5} matches | price {r.price:4.1f} spread close {r.spread_close:4.2f} entry {r.spread_entry:4.2f} | "
                  f"volume event bar median {r.vol_bar_median:8.0f} next {r.vol_next_median:8.0f} | maker fill fav {100*r.fav_fill_rate:3.0f}% dog {100*r.dog_fill_rate:3.0f}% | offer+3 sold fav {100*r.fav_offer3_sold:3.0f}% dog {100*r.dog_offer3_sold:3.0f}%")
    print("  THE LOOP IN ONE TABLE: resting bid filled after a 10c+ fall, then an offer T cents higher for 30 minutes (else sell at the bid)")
    print("   share of cycles that 'work' (the offer fills) beside the average result per cycle, cents")
    for group in GROUPS:
        for band in names+["all"]:
            line = []
            for period in PERIODS:
                r = R[(R.group == group) & (R["filter"] == "any") & (R.band == band) & (R.X == 10) & (R.dir == "fall") & (R.period == period)]
                if r.empty:
                    continue
                r = r.iloc[0]
                line.append(f"{period} fills {int(r.fav_make_offer3_n):5} "+" ".join(
                    f"+{T}: {100*r[f'fav_offer{T}_sold']:3.0f}% {r[f'fav_make_offer{T}']:+5.2f}[{r[f'fav_make_offer{T}_lo']:+5.2f},{r[f'fav_make_offer{T}_hi']:+5.2f}]" for T in (3, 5, 10))
                    + f" hold: {r.fav_makeS:+5.2f}[{r.fav_makeS_lo:+5.2f},{r.fav_makeS_hi:+5.2f}]")
            print(f"   {group:10} {band:6} "+" | ".join(line))
    # The search for anything positive after costs in both periods.
    trade_cols = [c for c in measures if "take" in c or "make_offer" in c or "makeS" in c]
    wide = R[R["filter"] == "any"].pivot_table(index=["group", "band", "X", "dir"], columns="period", values=[c for c in trade_cols]+[c+"_lo" for c in trade_cols]+[c+"_n" for c in trade_cols])
    hits, tested = [], 0
    for key, r in wide.iterrows():
        for c in trade_cols:
            try:
                a, alo, an, b, bn = r[(c, "archive")], r[(c+"_lo", "archive")], r[(c+"_n", "archive")], r[(c, "recent")], r[(c+"_n", "recent")]
            except KeyError:
                continue
            if np.isnan(a) or np.isnan(b) or an < 30 or bn < 15:
                continue
            tested += 1
            if alo > 0 and b > 0:
                hits.append((key, c, a, alo, int(an), b, r[(c+"_lo", "recent")], int(bn)))
    print(f"\n  SEARCH: {tested} tradeable cells with 30+ archive and 15+ recent trades; {len(hits)} have an archive interval above zero AND a positive recent mean")
    for key, c, a, alo, an, b, blo, bn in sorted(hits, key=lambda h: -h[2]):
        print(f"   {key} {c}: archive {a:+.2f} (lo {alo:+.2f}, n {an}) recent {b:+.2f} (lo {blo:+.2f}, n {bn})")
    (OUT/"event-search.json").write_text(json.dumps(dict(tested=tested, hits=[dict(cell=list(map(str, k)), rule=c, archive=a, archive_lo=alo, archive_n=an,
                                                                                     recent=b, recent_lo=blo, recent_n=bn) for k, c, a, alo, an, b, blo, bn in hits]), indent=1))


# ---------------------------------------------------------------- trade record

QUOTA = (("Challenger", 110), ("WTA", 50), ("ATP", 50), ("WTA Challenger", 35), ("ITF men", 35), ("ITF women", 20))


def trade_sample():
    """A fixed random sample of recent matches, interleaved across tours so a time cap cuts evenly."""
    M = pd.read_pickle(OUT/"matches.pkl")
    rng = np.random.default_rng(5)
    picks = []
    for tour, n in QUOTA:
        q = M[(M.period == "recent") & (M.tour == tour)]
        picks.append(list(q.mid.values[rng.permutation(len(q))[:n]]))
    order = []
    for k in range(max(len(p) for p in picks)):
        for p, (_, n) in zip(picks, QUOTA):
            j = int(k*len(p)/max(len(x) for x in picks))            # spread each tour evenly through the order
            if j < len(p) and (k == 0 or j != int((k-1)*len(p)/max(len(x) for x in picks))):
                order.append(p[j])
    return M.set_index("mid").loc[order].reset_index()


def trade_pages(client, row, download, deadline, max_pages=14):
    """In-play trades of one market, or None when not fully cached and the budget is spent (or it needs too many pages)."""
    trades, cursor, n = [], "", 0
    while True:
        params = dict(ticker=row["ticker"], min_ts=int(row["start_ts"])-360, max_ts=int(row["settle"]), limit=1000)
        if cursor:
            params["cursor"] = cursor
        if not cached("/markets/trades", params) and (not download or time.monotonic() > deadline):
            return None
        try:
            data = paced(client, "/markets/trades", params)
        except RuntimeError as error:
            print(f"  {row['ticker']}: {error}")
            return None
        n += 1
        trades += data.get("trades") or []
        cursor = data.get("cursor")
        if not cursor or not data.get("trades"):
            return trades
        if n >= max_pages:
            return None


def fetch_trades(minutes):
    S = trade_sample()
    deadline, done, short = time.monotonic()+60*minutes, 0, 0
    with httpx.Client(timeout=40) as client:
        for _, row in S.iterrows():
            got = trade_pages(client, row, True, deadline)
            done += got is not None
            short += got is None
            if time.monotonic() > deadline:
                break
            if done % 25 == 0 and got is not None:
                print(f"  {done} matches complete, {short} incomplete, {int(deadline-time.monotonic())}s left", flush=True)
    print(f"FETCH {done} matches with a complete in-play trade record, {short} incomplete (too many pages or out of time)")


def trades():
    d = Data()
    E = pd.read_pickle(OUT/"events.pkl")
    S = trade_sample()
    rows, regs, used = [], defaultdict(list), []
    with httpx.Client(transport=STUB) as client:
        for _, row in S.iterrows():
            raw = trade_pages(client, row, False, 0)
            if not raw:
                continue
            yes = row.p0 > 50
            t = np.array([timestamp(x["created_time"]) for x in raw])
            p = np.array([100*float(x["yes_price_dollars"]) for x in raw])
            size = np.array([float(x.get("count_fp", x.get("count", 0))) for x in raw])
            buys_yes = np.array([(x.get("taker_side") or x.get("taker_outcome_side")) == "yes" for x in raw])
            o = np.argsort(t)
            t, size = t[o], size[o]
            price = np.round(p[o]) if yes else 100-np.round(p[o])          # favourite's price
            buy = buys_yes[o] if yes else ~buys_yes[o]                     # taker bought the favourite
            used.append(row.mid)
            first = int(np.where(d.g == row.mid)[0][0])
            n = int(row.n)
            t0 = row.start_ts                                              # grid point k is the minute ending at t0+60k
            # Per-minute last print on each side: buyer-initiated (at the ask) and seller-initiated (at the bid).
            k = np.ceil((t-t0)/60).astype(int)
            series = {}
            for name, mask in (("ask_print", buy), ("bid_print", ~buy), ("any_print", np.ones(len(t), bool))):
                s = np.full(n, np.nan)
                ok = mask & (k >= 0) & (k < n)
                s[k[ok]] = price[ok]                                       # sorted by time, so the last print of the minute wins
                series[name] = s
            series["mid"] = d.mid[first:first+n]
            for h in (1, 2, 5, 10):
                for name, s in series.items():
                    i = np.arange(h, n-h-1)
                    regs[h, name].append((np.full(len(i), row.mid), s[i]-s[i-h], s[i+h]-s[i], s[i+1+h]-s[i+1]))
                # last-h change on the midpoint, follow-up on same-side prints: no shared observation at all
                i = np.arange(h, n-h-1)
                regs[h, "mid_then_bid_print"].append((np.full(len(i), row.mid), series["mid"][i]-series["mid"][i-h],
                                                      series["bid_print"][i+h]-series["bid_print"][i], series["bid_print"][i+1+h]-series["bid_print"][i+1]))
            # Events of this match, re-priced on actual prints.
            for _, ev in E[E.m == row.mid].iterrows():
                T = t0+60*(ev.i-first)                                     # end of the event bar
                for who, faller_buy in (("fav", buy), ("dog", ~buy)):
                    pr = price if who == "fav" else 100-price
                    ent = np.where(faller_buy & (t > T+1) & (t <= T+60))[0]
                    out = dict(m=row.mid, tour=row.tour, X=ev.X, dir=ev.dir, who=who, band=ev.band, bar_ask=ev[f"{who}_ask"],
                               bar_take5=ev[f"{who}_take5"], bar_take15=ev[f"{who}_take15"])
                    L = ev[f"{who}_make_L"] if ev[f"{who}_make_fill"] else (d.bid[ev.i] if who == "fav" else 100-d.ask[ev.i])
                    sell = ~faller_buy
                    w1, w5 = (t > T) & (t <= T+60), (t > T) & (t <= T+300)
                    out.update(sold_below_bid_1m=size[sell & w1 & (pr < L)].sum(), sold_below_bid_5m=size[sell & w5 & (pr < L)].sum(),
                               sold_at_or_below_bid_1m=size[sell & w1 & (pr <= L)].sum(),
                               bought_1m=size[faller_buy & w1].sum(), all_1m=size[w1].sum(), prints_1m=int(w1.sum()))
                    last_bid = np.where(sell & (t <= T) & (t > T-60))[0]
                    out["bid_print_T"] = pr[last_bid[-1]] if len(last_bid) else np.nan
                    if len(ent):
                        a = pr[ent[0]]
                        out.update(paid=a, entry_delay=t[ent[0]]-T, entry_size=size[ent[0]])
                        for kk in (5, 15):
                            ex = np.where(sell & (t > T+60*kk) & (t <= T+60*kk+90))[0]
                            if len(ex):
                                b = pr[ex[0]]
                                out[f"print_take{kk}"] = b-fee(b)-a-fee(a)
                    for kk in (1, 5, 15):
                        ex = np.where(sell & (t > T+60*kk) & (t <= T+60*kk+90))[0]
                        if len(ex) and len(last_bid):
                            out[f"bid_print_ch{kk}"] = pr[ex[0]]-out["bid_print_T"]
                    rows.append(out)
    print(f"\nTRADE RECORD: {len(used)} recent matches with a complete in-play trade record "
          f"({dict(pd.Series(d.tour[used]).value_counts())})")
    print("  slope of the next-h change on the last-h change, the same matches, by price series (cluster by match)")
    print("   series: mid = quote midpoint; any_print = last trade of the minute (bounces between bid and ask);")
    print("           bid_print / ask_print = last seller- / buyer-initiated trade of the minute (one side of the book, no bounce)")
    out = []
    for (h, name), parts in sorted(regs.items()):
        g, x, y, ys = (np.concatenate(v) for v in zip(*parts))
        a, b = cluster_slope(x, y, g), cluster_slope(x, ys, g)
        out.append(dict(h=h, series=name, **{f"raw_{k}": v for k, v in a.items()}, **{f"skip_{k}": v for k, v in b.items()}))
        print(f"   h={h:2} {name:18} raw {a['slope']:+.3f}[{a['lo']:+.3f},{a['hi']:+.3f}] n={a['n']:6}   skip {b['slope']:+.3f}[{b['lo']:+.3f},{b['hi']:+.3f}] n={b['n']:6}")
    pd.DataFrame(out).to_csv(OUT/"trade-slopes.csv", index=False, float_format="%.5f")
    T = pd.DataFrame(rows)
    T.to_pickle(OUT/"trade-events.pkl")
    fade = T[((T.dir == "fall") & (T.who == "fav")) | ((T.dir == "rise") & (T.who == "dog"))]
    print("  events re-priced on prints, buying the player who FELL (taker; first buyer-initiated print in the next minute, first seller-initiated print k minutes on)")
    for X in MOVES:
        q = fade[fade.X == X]
        if len(q) < 10:
            continue
        both = q.dropna(subset=["paid", "bar_ask"])
        for kk in (5, 15):
            if f"print_take{kk}" not in q:
                q = q.assign(**{f"print_take{kk}": np.nan})
        s5, s15 = q.dropna(subset=["print_take5", "bar_take5"]), q.dropna(subset=["print_take15", "bar_take15"])
        print(f"   {X:2}c+ events {len(q):5} matches {q.m.nunique():3} | a buy printed within 60s: {100*q.paid.notna().mean():3.0f}% | paid minus the bar's opening ask {stat(both.paid-both.bar_ask, both.m)} | "
              f"round trip on prints vs on bars, the same events: 5m {stat(s5.print_take5, s5.m)} vs {stat(s5.bar_take5, s5.m)} (n={len(s5)}); 15m {stat(s15.print_take15, s15.m)} vs {stat(s15.bar_take15, s15.m)} (n={len(s15)})")
        hybrid = {kk: q[f"print_take{kk}"].fillna(q[f"bar_take{kk}"]) for kk in (5, 15)}
        none = {kk: q[q[f"print_take{kk}"].isna()] for kk in (5, 15)}
        print(f"        every event (the print-only figures keep only events with a sell print k minutes on, i.e. matches still trading): prints where they exist, bars otherwise: "
              f"5m {stat(hybrid[5], q.m)} 15m {stat(hybrid[15], q.m)} | bars on all events: 5m {stat(q.bar_take5, q.m)} 15m {stat(q.bar_take15, q.m)} | "
              f"bars on the events with no print exit: 5m {stat(none[5].bar_take5, none[5].m)} 15m {stat(none[15].bar_take15, none[15].m)}")
        print(f"        bid-side print drift after the event (same survivor caveat): 1m {stat(q.get('bid_print_ch1', pd.Series(dtype=float)), q.m)} 5m {stat(q.get('bid_print_ch5', pd.Series(dtype=float)), q.m)} 15m {stat(q.get('bid_print_ch15', pd.Series(dtype=float)), q.m)}")
        print(f"        capacity in the minute after: contracts sold strictly below the closing bid median {q.sold_below_bid_1m.median():.0f} (p25 {q.sold_below_bid_1m.quantile(.25):.0f}, zero in {100*(q.sold_below_bid_1m == 0).mean():.0f}%), "
              f"at or below {q.sold_at_or_below_bid_1m.median():.0f}; bought by takers {q.bought_1m.median():.0f}; all {q.all_1m.median():.0f} in {q.prints_1m.median():.0f} prints; within 5m below bid {q.sold_below_bid_5m.median():.0f}")
    for tour, q in fade[fade.X == 10].groupby("tour"):
        print(f"   10c+ {tour:15} events {len(q):4} sold below bid 1m median {q.sold_below_bid_1m.median():7.0f} zero {100*(q.sold_below_bid_1m == 0).mean():3.0f}% | all volume 1m {q.all_1m.median():7.0f}")


# ---------------------------------------------------------------- score feed

POINT = {0: 0, 15: 1, 30: 2, 40: 3, 50: 4}


def hold_prob(a, b, p):
    """P(server wins the game) from a points to b, each point won with probability p."""
    if a >= 3 and b >= 3:
        deuce = p*p/(p*p+(1-p)*(1-p))
        return deuce if a == b else (p+(1-p)*deuce if a > b else p*deuce)
    if a == 4:
        return 1.0
    if b == 4:
        return 0.0
    return p*hold_prob(a+1, b, p)+(1-p)*hold_prob(a, b+1, p)


def scores():
    from research.dip_scores import code_side, direction_side, phase, result_side, sets, winner, yes_quote
    events = defaultdict(list)
    for path in sorted(LIVE.glob("*.jsonl")):
        with path.open() as lines:
            for line in lines:
                try:
                    r = json.loads(line)
                    if r["event"].split("-")[0] in TOURS and isinstance(r.get("state"), dict):
                        events[r["event"]].append(r)
                except (ValueError, KeyError, TypeError):
                    continue
    games, counts = [], defaultdict(int)
    for event, rows in events.items():
        rows.sort(key=lambda r: r["ts"])
        counts["events"] += 1
        code, result = code_side(event, rows[0]["ticker"]), result_side(rows)
        direction, _, _ = direction_side(rows)
        firm = [v for v in (code, result) if v]
        if not firm or len(set(firm)) > 1 or (direction and direction != firm[0]):
            counts["yes side unclear"] += 1
            continue
        yes = firm[0]
        tour = TOURS[event.split("-")[0]]
        won = next((winner(r["state"]) for r in reversed(rows) if phase(r["state"]) == "done" and winner(r["state"])), None)
        seq = []
        for r in rows:
            s, q = r["state"], yes_quote(r)
            if phase(s) != "live" or q is None or not 0 < q[0] <= q[1] < 100:
                continue
            ids = {s.get("competitor1_id"): 1, s.get("competitor2_id"): 2}
            server, adv = ids.get(s.get("server") or None), ids.get(s.get("advantage") or None)
            rounds = sets(s)
            cur = next((x for x in rounds if not x[2]), None)
            if not server or cur is None:
                continue
            pts = (s.get("competitor1_current_round_score"), s.get("competitor2_current_round_score"))
            if pts[0] not in POINT or pts[1] not in POINT:
                continue
            a, b = POINT[pts[server-1]], POINT[pts[2-server]]
            if a == 3 and b == 3 and adv:
                a, b = (4, 3) if adv == server else (3, 4)
            mid = (q[0]+q[1])/2
            if "other" in r:                                           # both books: average the two readings of the same price
                try:
                    o = r["other"]["quote"]
                    ob, oa = 100*float(o["yes_bid_dollars"]), 100*float(o["yes_ask_dollars"])
                    if 0 < ob <= oa < 100 and oa-ob <= 6:
                        mid = (mid+100-(ob+oa)/2)/2
                except (KeyError, TypeError, ValueError):
                    pass
            total = (sum(x[0] for x in rounds), sum(x[1] for x in rounds))
            seq.append(dict(ts=r["ts"], key=(len(rounds), cur[0], cur[1]), server=server, a=a, b=b, tiebreak=cur[0] == cur[1] >= 6,
                            mid=mid if server == yes else 100-mid, spread=q[1]-q[0], total=total))
        if len(seq) < 10:
            counts["too few live rows"] += 1
            continue
        counts["matches used"] += 1
        # split into games
        blocks, start = [], 0
        for i in range(1, len(seq)+1):
            if i == len(seq) or seq[i]["key"] != seq[start]["key"] or seq[i]["server"] != seq[start]["server"]:
                blocks.append(seq[start:i])
                start = i
        for n, block in enumerate(blocks[:-1]):
            nxt = blocks[n+1][0]
            first = block[0]
            if first["tiebreak"]:
                continue
            srv = first["server"]
            gain = (nxt["total"][0]-first["total"][0], nxt["total"][1]-first["total"][1])
            if sorted(gain) != [0, 1] or nxt["ts"]-block[-1]["ts"] > 240:
                counts["game outcome unclear"] += 1
                continue
            held = gain[srv-1] == 1
            flip = lambda x: x if nxt["server"] == srv else 100-x      # the next game is read from this game's server's side
            if max(x["spread"] for x in block) > 6 or nxt["spread"] > 6:
                counts["wide quote in game"] += 1
                continue

            def readings(rows_, stop):
                """Per score state: (first quote, last quote, settled quote or NaN).

                The recorder writes a row when the state or the quote changes and polls about every 25 seconds, and
                the listing quote lags the book. A reading is "settled" when it is the last one of its state and was
                taken 15s or more after the state first appeared, or nothing at all changed for 45s after it appeared.
                """
                seen = {}
                for n_, x in enumerate(rows_):
                    after = rows_[n_+1]["ts"] if n_+1 < len(rows_) else stop
                    cell = seen.setdefault((x["a"], x["b"]), dict(first=x["mid"], t0=x["ts"]))
                    cell.update(last=x["mid"], t1=x["ts"], after=after)
                return {k: (v["first"], v["last"], v["last"] if v["t1"]-v["t0"] >= 15 or v["after"]-v["t0"] >= 45 else np.nan)
                        for k, v in seen.items()}
            stop = blocks[n+2][0]["ts"] if n+2 < len(blocks) else blocks[n+1][-1]["ts"]
            seen = readings(block, nxt["ts"])
            opening = readings(blocks[n+1], stop)[(blocks[n+1][0]["a"], blocks[n+1][0]["b"])]
            p_end = tuple(flip(v) for v in opening)
            p_start = seen[(0, 0)] if (0, 0) in seen else (np.nan,)*3
            won_match = None if won is None else float(won == srv)
            games.append(dict(event=event, tour=tour, women=tour in ("WTA", "WTA Challenger", "ITF women"), held=held, p_start=p_start,
                              p_end=p_end, fav_serving=block[0]["mid"] > 50, won_match=won_match,
                              states={f"{a}-{b}": v for (a, b), v in seen.items()}))
    G = pd.DataFrame(games)
    print(f"\nSCORE FEED: {dict(counts)}; {len(G)} service games with a clear outcome in {G.event.nunique()} matches "
          f"({dict(G.groupby('tour').event.nunique())})")
    name = {0: "0", 1: "15", 2: "30", 3: "40", 4: "A"}
    report = {}
    for women, label in ((False, "men"), (True, "women")):
        q = G[G.women == women]
        if len(q) < 30:
            print(f"  {label}: {len(q)} games, too few")
            continue
        rate = q.held.mean()
        grid = np.linspace(.45, .8, 351)
        p_fit = grid[np.argmin([abs(hold_prob(0, 0, p)-rate) for p in grid])]
        print(f"  {label}: {len(q)} games in {q.event.nunique()} matches, server held {100*rate:.1f}% -> serve-point win rate {p_fit:.3f} if points are independent")
        print("   Each cell: first reading / last reading / settled reading of the state (the first is stale, see the docstring of `readings`).")
        print("   state | games (settled) | price move 0-0 -> state | held: actual, model at fitted p, at "
              f"{'0.62, 0.65' if not women else '0.55, 0.58'} | market-implied hold | price change state -> start of next game [95%], settled: if held, if broken | favourite serving, settled")
        for a, b in ((0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3), (3, 4), (3, 3), (1, 0), (2, 0), (3, 0), (0, 0)):
            key = f"{a}-{b}"
            sub = q[q.states.map(lambda s: key in s)]
            if len(sub) < 8:
                continue
            held = sub.held.values
            lo_p, hi_p = (.62, .65) if not women else (.55, .58)
            cells, moves, implieds = [], [], []
            for which in (0, 1, 2):
                ps = sub.states.map(lambda s: s[key][which]).values.astype(float)
                to_end = sub.p_end.map(lambda v: v[which]).values-ps
                ok = ~np.isnan(to_end)
                up = to_end[held & ok].mean() if (held & ok).any() else np.nan
                down = to_end[~held & ok].mean() if (~held & ok).any() else np.nan
                implied = -down/(up-down) if up > down else np.nan
                r = cluster_mean(to_end, sub.event.values)
                fav = sub.fav_serving.values
                fr = cluster_mean(to_end[fav], sub.event.values[fav])
                moves.append(np.nanmean(ps-sub.p_start.map(lambda v: v[which]).values))
                implieds.append(implied)
                cells.append(f"{r['mean']:+5.2f} [{r['lo']:+5.2f},{r['hi']:+5.2f}]")
                report[f"{label} | {key} | {('first', 'last', 'settled')[which]}"] = dict(
                    games=len(sub), usable=int(ok.sum()), matches=sub.event.nunique(), held=float(held.mean()), implied_hold=float(implied),
                    model_fit=hold_prob(a, b, p_fit), drift=r, if_held=float(up), if_broken=float(down), favourite_serving=fr)
            print(f"   {name[a]:>2}-{name[b]:<2} | {len(sub):4} ({int(ok.sum()):4}) | {moves[0]:+5.1f}/{moves[1]:+5.1f}/{moves[2]:+5.1f} | {100*held.mean():3.0f}%, {100*hold_prob(a, b, p_fit):3.0f}%, "
                  f"{100*hold_prob(a, b, lo_p):3.0f}%, {100*hold_prob(a, b, hi_p):3.0f}% | {100*implieds[0]:3.0f}%/{100*implieds[1]:3.0f}%/{100*implieds[2]:3.0f}% | "
                  f"{cells[0]} / {cells[1]} / {cells[2]}: {up:+5.1f}, {down:+5.1f} | n={fr['n']} {fr['mean']:+5.2f} [{fr['lo']:+5.2f},{fr['hi']:+5.2f}]")
    # Pooled "server behind" states, and what sample would detect a 1c mispricing (settled readings).
    behind = []
    for _, r in G.iterrows():
        for key, v in r.states.items():
            a, b = map(int, key.split("-"))
            if b > a:
                behind.append(dict(event=r.event, key=key, d=r.p_end[2]-v[2], d_last=r.p_end[1]-v[1], d_first=r.p_end[0]-v[0], held=r.held,
                                   fav=r.fav_serving, women=r.women, bp=(b >= 3 and b > a),
                                   settle=np.nan if r.won_match is None else 100*r.won_match-v[2]))
    B = pd.DataFrame(behind)
    print("  pooled, server BEHIND in the game: price change from the state to the start of the next game")
    for label, q in (("server behind, any state", B), ("favourite serving and behind", B[B.fav]), ("facing break point", B[B.bp]),
                     ("favourite facing break point", B[B.bp & B.fav]), ("server at 0-30", B[B.key == "0-2"]), ("favourite at 0-30", B[(B.key == "0-2") & B.fav]),
                     ("favourite at 0-30 or 0-40 or 15-40", B[B.key.isin(("0-2", "0-3", "1-3")) & B.fav])):
        r, r1, r0 = (cluster_mean(q[c].values, q.event.values) for c in ("d", "d_last", "d_first"))
        st = cluster_mean(q.settle.values.astype(float), q.event.values)
        se = (r["hi"]-r["lo"])/3.92
        need = r["n"]*(se*2.8/1.0)**2
        report[f"pooled | {label}"] = dict(readings=len(q), settled=r["n"], matches=q.event.nunique(), held=float(q.held.mean()), drift_settled=r,
                                           drift_last=r1, drift_first=r0, to_settlement=st, settled_readings_to_detect_1c=float(need))
        print(f"   {label:36} readings {len(q):4} (settled {r['n']:4}) in {q.event.nunique():3} matches, held {100*q.held.mean():3.0f}% | first {r0['mean']:+5.2f} [{r0['lo']:+5.2f},{r0['hi']:+5.2f}] "
              f"last {r1['mean']:+5.2f} [{r1['lo']:+5.2f},{r1['hi']:+5.2f}] settled {r['mean']:+5.2f} [{r['lo']:+5.2f},{r['hi']:+5.2f}] | to settlement {st['mean']:+5.1f} [{st['lo']:+5.1f},{st['hi']:+5.1f}] n={st['n']} | "
              f"settled readings to see 1c at 80% power: {need:,.0f} ({need/max(r['n'], 1):.1f}x)")
    (OUT/"scores.json").write_text(json.dumps(report, indent=1, default=float))
    G.drop(columns=["states", "p_start", "p_end"]).to_csv(OUT/"score-games.csv", index=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("steps", nargs="+", choices=("build", "martingale", "events", "report", "fetch", "trades", "scores"))
    parser.add_argument("--minutes", type=float, default=18, help="time budget for `fetch`")
    args = parser.parse_args()
    for step in args.steps:
        if step == "fetch":
            fetch_trades(args.minutes)
        else:
            dict(build=build, martingale=martingale, events=events, report=report_events, trades=trades, scores=scores)[step]()
