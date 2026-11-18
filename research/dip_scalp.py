"""Buy the dip, sell the rebound, repeat: an in-play scalping loop on Kalshi tennis favourites.

The idea (from one ITF match whose favourite went 85 -> 64 -> 87 -> 49 -> 73 -> 32
and then won): rest a bid some way below the pre-match price, and when it fills
rest a sell a few cents higher instead of holding to settlement; when the sell
fills, put the bid back and go again.

Data. The one-minute bars cached under data/raw/cost-research-v1 (six hours
before settlement, the alphabetically-first market of each event), exactly the
27,824 matches of research/tennis_crossings.py: ATP, WTA, ATP Challenger, WTA
Challenger, ITF men, ITF women. Archive = settled before 2026-09-18, recent =
after. No network: the HTTP client is a stub that answers 404.

Definitions (all fixed before any result was looked at)
* Player price P0 = median midpoint of the first 15 bars (stable within 3c);
  the other player is 100 minus. Bands: 45-60 ("either player": every player
  whose own P0 is in the band, so both players of a 45-55 match), and the
  favourite at 60-70, 70-80, 80-90, 90+.
* Start of play = favourite_gap.detect_start. Orders are decided on the close
  of the detection bar; only later bars can fill.
* Entry level L = round(P0) - d for d in 10, 15, 20, 30, or round(f*P0) for f in
  0.85, 0.75, 0.60 (L at least 5).
* Target T = L + 5, 10, 15, 20, or round(P0) ("back to the pre-match price"),
  capped at 99.
* Stop: none, or 15c / 25c below L. When the player's bid closes a bar at or
  below it, sell at the next bar's opening bid and pay the taker fee.
* The loop. IDLE -> the bid is placed at the first bar close on which the
  player's bid is at or above L (so the order rests below the market). RESTING
  -> filled by a later bar. LONG -> the sell rests from the bar after the fill.
  After an exit the loop returns to IDLE. A position still open when the bars
  end settles at 100 x settlement value.
* Three executions, to bracket the truth:
    strict  maker both legs. The bid fills only when a later bar has a print
            strictly below L; the sell only on a later print strictly above T.
    touch   maker both legs; a print AT the price also fills (we are always
            first in the queue: an upper bound).
    taker   both legs cross. Entry signal = a bar closing with the ask at or
            below L; buy the next bar's opening ask. Exit signal = a bar closing
            with the bid at or above T; sell the next bar's opening bid. Taker
            fee 0.07*p*(1-p) on each leg.
  Maker fee: zero on Challenger, WTA Challenger and ITF; 0.0175*p*(1-p) per leg
  (an assumption) on ATP and WTA. Settlement is free.
* Max cycles per match 1, 2, 3 or unlimited, and the time rule "no new entry
  once the player has printed below 30c" are both truncations of the unlimited
  run (a later cycle never changes an earlier one), so one run scores all eight.
* Grid: 5 bands x 7 entry levels x 5 targets x 3 stops x 4 cycle caps x 2 time
  rules = 4,200 rules, per execution, per tour group (Challenger + WTA
  Challenger; ATP + WTA; ITF; pooled).
* Selection: within a tour group, the rule with the highest archive t-statistic
  of net cents per match placed (strict execution, at least 100 archive
  cycles). It is scored once on the recent period. Because a target at 99c (or
  far above the entry) turns the "loop" back into the old hold-to-settlement
  rule, the selection is repeated among rules that really sell: half or more of
  their archive cycles end at the target.
* Fair-market replay (step `null`): every real match is redrawn as the
  martingale p_k = Phi(W_k / sqrt(1 - k/N)) started at its P0 (as in
  tennis_crossings.py; 700 steps reproduce the real cost of a first resting
  fill), quoted 1c either side, ending at 99 or 1, and the whole grid and the
  same selection are run on it. A take-profit loop wins most cycles in a fair
  market too; the replay says how much of what is seen is that. The replay has
  no favourite underpricing, so a real rule can beat it simply by holding a
  90c+ favourite; the decomposition separates the two. Its cost per fill grows
  with the coarseness of the walk (a continuous fair market costs a maker
  nothing), so `--walk-steps 2000` is run as a sensitivity check.
* Decomposition per match placed: hold = the first entry held to settlement;
  exit = what selling at the target changes on that first entry; re-entry =
  the sum of cycles 2 and later.
* Intervals: the grid table (grid.csv) carries a standard error and t-statistic
  per rule, clustered by match; bootstrap 95% intervals resampling matches are
  computed for the selected and reference rules (step `verdict`).
* Choppiness: completed swings of 10c in the midpoint (a move of 10c from the
  start, then each reversal of 10c from the running extreme). Hindsight splits
  use the whole match; the honest versions arm the loop only from minute 20
  (split by the range of the first 20 minutes) or only after two swings have
  already completed.

Steps
    build      bars -> compact arrays (data/processed/dip-scalp/bars.npz)
    grid       the whole grid on the real matches
    null       the whole grid on fair-market replays (--reps)
    headline   grid against the replay; selection; chronology
    decompose  hold-to-settlement vs exit rule vs re-entry
    choppy     which matches pay, and whether that can be known in time
    verdict    the selected rule in full: intervals, capacity, trades per month
    sensitivity  the replay's cost per cycle at each walk length on disk

    .venv/bin/python research/dip_scalp.py build grid
    .venv/bin/python research/dip_scalp.py null --reps 24
    .venv/bin/python research/dip_scalp.py null --reps 4 --walk-steps 2000
    .venv/bin/python research/dip_scalp.py headline decompose choppy verdict sensitivity

Outputs under data/processed/dip-scalp/. Public market data only; no orders,
no authentication, nothing fetched.
"""
import argparse
import itertools
import json
import multiprocessing as mp
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
from research.favourite_gap import STUB, TOURS, bar_params, detect_start, listing, parse_rules
from research.tennis_crossings import arrays

OUT = ROOT / "data/processed/dip-scalp"
PERIODS = ("archive", "recent")
GROUP_OF = {"Challenger": "Chall+WTA Chall", "WTA Challenger": "Chall+WTA Chall", "ATP": "ATP+WTA", "WTA": "ATP+WTA",
            "ITF men": "ITF", "ITF women": "ITF"}
GROUPS = ("Chall+WTA Chall", "ATP+WTA", "ITF")
UNITS = GROUPS+("ALL",)
MAKER_FEE_TOURS = ("ATP", "WTA")
BANDS = ((45, 60), (60, 70), (70, 80), (80, 90), (90, 101))
BAND_NAMES = ("45-60 either", "60-70", "70-80", "80-90", "90+")
DIPS = (("c", 10), ("c", 15), ("c", 20), ("c", 30), ("f", .85), ("f", .75), ("f", .60))
DIP_NAMES = ("-10c", "-15c", "-20c", "-30c", "x0.85", "x0.75", "x0.60")
TARGETS = (5, 10, 15, 20, "P0")
TARGET_NAMES = ("+5", "+10", "+15", "+20", "P0")
STOPS = (0, 15, 25)
STOP_NAMES = ("none", "15c", "25c")
MODES = ("strict", "touch", "taker")
CAPS = (1, 2, 3, 99)
CAP_NAMES = ("1", "2", "3", "unl")
RULES = ("off", "on")                                   # no new entry after a print below 30c
MIN_CYCLES = 100
RECENT_MONTHS, ARCHIVE_MONTHS = 59/30.44, 260/30.44     # 18 Sep - 16 Nov; 1 Jan - 17 Sep
NB, NO = 8, 4                                           # buckets: cycle 1,2,3,4+ x (before / after a sub-30 print); outcomes
OUTCOMES = ("target", "stopped", "settled win", "settled loss")
SHAPE = (len(DIPS), len(TARGETS), len(STOPS), len(MODES))
RNG = np.random.default_rng(23)


# ---------------------------------------------------------------- build

def build():
    """The match set and order of tennis_crossings.build, keeping the bars."""
    rows, chunks, seen = [], [], set()
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
                    params = bar_params(settle)
                    if not cached(path.format(t=m["ticker"]), params):
                        continue
                    bars = fetch(client, path.format(t=m["ticker"]), params).get("candlesticks")
                    if not bars:
                        continue
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
                    path_, ts, bid, ask, low, high, vol, obid, oask = arrays(bars)
                    began = detect_start(path_)
                    if began is None or began+1 >= len(path_):
                        continue
                    ok = np.where(~np.isnan(bid[:began+1]) & ~np.isnan(ask[:began+1]))[0]
                    if not len(ok):
                        continue
                    tournament, stage = parse_rules(m.get("rules_primary"))
                    rows.append(dict(mid=len(rows), event=event, ticker=m["ticker"], tour=tour, period=PERIODS[0] if archived else PERIODS[1],
                                     settle=settle, p0=float(np.median(mids)), outcome=outcome, began=began, n=len(ts),
                                     start_ts=float(ts[began]), tournament=tournament, round=stage))
                    chunks.append(np.stack([ts-ts[began], bid, ask, low, high, vol, obid, oask]).astype(np.float32)[:, began:])
                    if len(rows) % 2000 == 0:
                        print(f"  {len(rows)} matches", flush=True)
    OUT.mkdir(parents=True, exist_ok=True)
    m = pd.DataFrame(rows)
    m["len"] = [c.shape[1] for c in chunks]
    m.to_pickle(OUT/"matches.pkl")
    np.savez(OUT/"bars.npz", data=np.concatenate(chunks, axis=1), offsets=np.concatenate([[0], np.cumsum(m.len.values)]))
    print(f"BUILD {len(m)} matches")
    print(m.groupby(["tour", "period"]).size().unstack())
    old = ROOT/"data/processed/tennis-crossings/matches.pkl"
    if old.exists():
        o = pd.read_pickle(old)
        same = len(o) == len(m) and (o.event.values == m.event.values).all() and np.allclose(o.p0.values, m.p0.values)
        print(f"  same match set and order as tennis_crossings: {same}")


# ---------------------------------------------------------------- players and padded arrays

class Data:
    """One row per (match, player) in a band, bars from the detection bar on, padded; rows sorted by length, longest first."""


def players(m):
    """Rows: every player with own P0 >= 45. Columns: mid, side (0 = YES), p0p, band, payout, fee multiplier."""
    a = m.assign(side=0, p0p=m.p0, payout=m.outcome)
    b = m.assign(side=1, p0p=100-m.p0, payout=1-m.outcome)
    pl = pd.concat([a, b], ignore_index=True)
    pl = pl[pl.p0p >= 45].copy()
    pl["band"] = np.searchsorted([60, 70, 80, 90], pl.p0p.values, side="right")
    pl["unit"] = pl.tour.map(GROUP_OF).map({g: i for i, g in enumerate(GROUPS)})
    pl["pi"] = (pl.period == "recent").astype(int)
    pl["mm"] = np.where(pl.tour.isin(MAKER_FEE_TOURS), 1.75, 0.0)
    return pl.sort_values(["len", "mid", "side"], ascending=[False, True, True]).reset_index(drop=True)


def real_data(m=None, pl=None):
    m = pd.read_pickle(OUT/"matches.pkl") if m is None else m
    pl = players(m) if pl is None else pl
    z = np.load(OUT/"bars.npz")
    data, off = z["data"], z["offsets"]
    n, width = len(pl), int(pl.len.max())
    d = Data()
    low, high = np.full((n, width), np.nan, np.float32), np.full((n, width), np.nan, np.float32)
    bid, ask = low.copy(), low.copy()
    obid, oask, vol, ts = low.copy(), low.copy(), np.zeros((n, width), np.float32), low.copy()
    for r, (mid, side, length) in enumerate(zip(pl.mid.values, pl.side.values, pl.len.values)):
        x = data[:, off[mid]:off[mid]+length]
        ts[r, :length], vol[r, :length] = x[0], np.nan_to_num(x[5])
        if side == 0:
            bid[r, :length], ask[r, :length], low[r, :length], high[r, :length], obid[r, :length], oask[r, :length] = x[1], x[2], x[3], x[4], x[6], x[7]
        else:
            bid[r, :length], ask[r, :length], low[r, :length], high[r, :length], obid[r, :length], oask[r, :length] = \
                100-x[2], 100-x[1], 100-x[4], 100-x[3], 100-x[7], 100-x[6]
    d.low, d.high, d.bid, d.ask, d.obid, d.oask, d.vol, d.ts = low, high, bid, ask, obid, oask, vol, ts
    d.alive = (pl.len.values[:, None] > np.arange(width)[None, :]).sum(axis=0)
    finish(d, pl)
    return m, pl, d


def finish(d, pl):
    d.n, d.width = len(pl), d.low.shape[1]
    d.p0 = np.rint(pl.p0p.values)
    d.payout, d.mm = pl.payout.values.astype(float), pl.mm.values
    codes, d.cluster = np.unique(pl.mid.values, return_inverse=True)
    d.n_clusters = len(codes)
    # one group id per cluster: unit x period x band (both players of a match always share them)
    gid = (pl.unit.values*2+pl.pi.values)*len(BANDS)+pl.band.values
    d.cgid = np.zeros(d.n_clusters, int)
    d.cgid[d.cluster] = gid
    d.n_gid = len(GROUPS)*2*len(BANDS)


def levels(d, dip, target):
    kind, x = DIPS[dip]
    L = d.p0-x if kind == "c" else np.rint(x*d.p0)
    T = np.minimum(99, d.p0 if TARGETS[target] == "P0" else L+TARGETS[target])
    valid = (L >= 5) & (T > L)
    return L, T, valid


def fee(mult, p):
    return mult*p/100*(1-p/100)


# ---------------------------------------------------------------- the loop

def run(d, L, T, valid, stop, mode, gate=None, detail=False):
    """Simulate the unlimited loop for every row. Returns per-row accumulators by bucket, and optionally one row per cycle.

    Bucket = min(cycle, 4)-1 + 4*(the player had printed below 30c in a bar before the entry bar).
    """
    n = d.n
    state = np.where(valid, 0, -1).astype(np.int8)      # 0 idle, 1 resting / watching, 2 buy at open, 3 long, 4 sell at open
    cyc = np.zeros(n, np.int16)
    cost, bucket, reason = np.zeros(n), np.zeros(n, np.int8), np.zeros(n, np.int8)
    sub30, placed = np.zeros(n, bool), np.zeros(n, bool)
    net, hold, outc = np.zeros((n, NB)), np.zeros((n, NB)), np.zeros((n, NB, NO), np.int32)
    S = np.where(L-stop >= 1, L-stop, -1) if stop else None
    maker = mode != "taker"
    rows = [] if detail else None
    t_in, v_in = (np.zeros(n, np.int32), np.zeros(n)) if detail else (None, None)

    def close(idx, proceeds, outcome, t):
        b = bucket[idx]
        np.add.at(net, (idx, b), proceeds-cost[idx])
        np.add.at(hold, (idx, b), 100*d.payout[idx]-cost[idx])
        np.add.at(outc, (idx, b, outcome), 1)
        if detail and len(idx):
            exit_volume = d.vol[idx, t] if t is not None and hasattr(d, "vol") else np.zeros(len(idx))
            rows.append(np.column_stack([idx, cyc[idx], b, t_in[idx], np.full(len(idx), -1 if t is None else t), cost[idx], proceeds,
                                         np.broadcast_to(outcome, idx.shape), v_in[idx], exit_volume, 100*d.payout[idx]]))

    def enter(idx, price, mult, t):
        cost[idx] = price+fee(mult, price)
        cyc[idx] += 1
        bucket[idx] = np.minimum(cyc[idx], 4)-1+4*sub30[idx]
        state[idx] = 3
        if detail:
            t_in[idx] = t
            v_in[idx] = d.vol[idx, t] if hasattr(d, "vol") else 0

    for t in range(d.width):
        k = int(d.alive[t])
        if k == 0:
            break
        st = state[:k]
        lo, hi, bid = d.low[:k, t], d.high[:k, t], d.bid[:k, t]
        fresh = np.zeros(k, bool)
        if t:
            ob = d.obid[:k, t]
            idx = np.flatnonzero((st == 4) & (ob > 0))
            if len(idx):
                close(idx, ob[idx]-fee(7, ob[idx]), reason[idx], t)
                st[idx] = 0
            if maker:
                idx = np.flatnonzero((st == 1) & ((lo < L[:k]) if mode == "strict" else (lo <= L[:k])))
                if len(idx):
                    enter(idx, L[idx], d.mm[idx], t)
                    fresh[idx] = True
            else:
                oa = d.oask[:k, t]
                idx = np.flatnonzero((st == 2) & (oa > 0) & (oa < 100))
                if len(idx):
                    enter(idx, oa[idx], 7, t)
                idx = np.flatnonzero((st == 1) & (d.ask[:k, t] <= L[:k]))
                st[idx] = 2
            long = (st == 3) & ~fresh
            if maker:
                hit = long & ((hi > T[:k]) if mode == "strict" else (hi >= T[:k]))
                idx = np.flatnonzero(hit)
                if len(idx):
                    close(idx, T[idx]-fee(d.mm[idx], T[idx]), 0, t)
                    st[idx] = 0
            else:
                hit = long & (bid >= T[:k])
                idx = np.flatnonzero(hit)
                st[idx], reason[idx] = 4, 0
            if stop:
                idx = np.flatnonzero(long & ~hit & (bid <= S[:k]))
                st[idx], reason[idx] = 4, 1
        arm = (st == 0) & (bid >= L[:k])
        if gate is not None:
            arm &= t >= gate[:k]
        st[arm] = 1
        placed[:k] |= arm
        if t:
            sub30[:k] |= lo < 30
    idx = np.flatnonzero(state >= 3)
    won = d.payout[idx] >= .5
    for outcome, part in ((2, idx[won]), (3, idx[~won])):
        close(part, 100*d.payout[part], outcome, None)
    if detail:
        cols = ["row", "cycle", "bucket", "t_in", "t_out", "cost", "proceeds", "outcome", "vol_in", "vol_out", "settle_value"]
        return net, hold, outc, placed, pd.DataFrame(np.concatenate(rows) if rows else np.zeros((0, len(cols))), columns=cols)
    return net, hold, outc, placed


def variant_masks():
    """For each (cap, rule): which of the 8 buckets count."""
    out = np.zeros((len(CAPS), len(RULES), NB), bool)
    for ci, cap in enumerate(CAPS):
        for ri in range(len(RULES)):
            for b in range(NB):
                out[ci, ri, b] = (b % 4)+1 <= min(cap, 4) and (ri == 0 or b < 4)
    return out


VARIANTS = variant_masks()
STATS = ("placed", "entered", "cycles", "target", "stopped", "win", "loss", "net", "net_sq", "worst", "hold", "net1", "hold1", "d1", "d1_sq")


def aggregate(d, net, hold, outc, placed):
    """Sums per group id for each (cap, rule): see STATS. Clustered by match."""
    out = np.zeros((len(CAPS), len(RULES), d.n_gid, len(STATS)))
    g, nc, ng = d.cgid, d.n_clusters, d.n_gid
    by = lambda v: np.bincount(d.cluster, v, nc)
    out[:, :, :, 0] = np.bincount(g, by(placed) > 0, ng)
    for ci in range(len(CAPS)):
        for ri in range(len(RULES)):
            keep = VARIANTS[ci, ri]
            cycles = outc[:, keep].sum(axis=1)
            x = by(net[:, keep].sum(axis=1))
            o = out[ci, ri]
            o[:, 1] = np.bincount(g, by(cycles.sum(axis=1)) > 0, ng)
            o[:, 2] = np.bincount(g, by(cycles.sum(axis=1)), ng)
            for j in range(NO):
                o[:, 3+j] = np.bincount(g, by(cycles[:, j]), ng)
            o[:, 7], o[:, 8] = np.bincount(g, x, ng), np.bincount(g, x*x, ng)
            worst = np.zeros(ng)
            np.minimum.at(worst, g, x)
            o[:, 9] = worst
            o[:, 10] = np.bincount(g, by(hold[:, keep].sum(axis=1)), ng)
            first = keep & (np.arange(NB) % 4 == 0)
            n1, h1 = by(net[:, first].sum(axis=1)), by(hold[:, first].sum(axis=1))
            o[:, 11], o[:, 12] = np.bincount(g, n1, ng), np.bincount(g, h1, ng)
            o[:, 13], o[:, 14] = np.bincount(g, n1-h1, ng), np.bincount(g, (n1-h1)**2, ng)
    return out


DATA = None


def one_config(key):
    dip, target, stop, mode = key
    L, T, valid = levels(DATA, dip, target)
    return key, aggregate(DATA, *run(DATA, L, T, valid, STOPS[stop], MODES[mode]))


def whole_grid(d, workers):
    """All configs -> array [dip, target, stop, mode, cap, rule, gid, stat]."""
    global DATA
    DATA = d
    keys = list(itertools.product(*(range(s) for s in SHAPE)))
    out = np.zeros(SHAPE+(len(CAPS), len(RULES), d.n_gid, len(STATS)))
    with mp.get_context("fork").Pool(workers) as pool:
        for key, value in pool.imap_unordered(one_config, keys, chunksize=2):
            out[key] = value
    return out


def do_grid(workers):
    m, pl, d = real_data()
    print(f"GRID {len(m)} matches, {d.n} player rows, {int(np.prod(SHAPE))} runs")
    np.save(OUT/"grid.npy", whole_grid(d, workers))


# ---------------------------------------------------------------- fair-market replay

def fair_data(m, pl, steps, rng):
    """Every match redrawn as a fair market. One 'bar' per step: a single print, quoted 1c either side; a final bar at 99 or 1."""
    nm = len(m)
    z0 = ndtri(np.clip(m.p0.values, 1, 99)/100)
    w = z0[:, None]+np.cumsum(rng.standard_normal((nm, steps), dtype=np.float32)/np.sqrt(steps), axis=1)
    won = w[:, -1] > 0
    scale = np.sqrt(1-np.arange(1, steps)/steps)
    price = np.empty((nm, steps+1), np.float32)
    price[:, 0] = np.clip(np.rint(m.p0.values), 1, 99)
    price[:, 1:steps] = np.clip(np.rint(100*ndtr(w[:, :-1]/scale)), 1, 99)
    price[:, steps] = np.where(won, 99, 1)
    del w
    yes = price[pl.mid.values]
    p = np.where(pl.side.values[:, None] == 0, yes, 100-yes)
    d = Data()
    d.low = d.high = p
    d.bid, d.ask = p-1, p+1
    d.obid, d.oask = np.roll(d.bid, 1, axis=1), np.roll(d.ask, 1, axis=1)
    d.alive = np.full(steps+1, len(pl))
    fair = pl.assign(payout=np.where(pl.side.values == 0, won[pl.mid.values], ~won[pl.mid.values]).astype(float))
    finish(d, fair)
    return d


def do_null(reps, steps, workers, seed):
    m = pd.read_pickle(OUT/"matches.pkl")
    pl = players(m)
    folder = OUT/f"null-{steps}"
    folder.mkdir(parents=True, exist_ok=True)
    for rep in range(reps):
        file = folder/f"rep-{seed+rep:03d}.npy"
        if file.exists():
            continue
        d = fair_data(m, pl, steps, np.random.default_rng(1000+seed+rep))
        np.save(file, whole_grid(d, workers).astype(np.float32))
        print(f"  null {steps} steps: replication {seed+rep} done", flush=True)


# ---------------------------------------------------------------- tables from a grid array

def by_unit(grid):
    """[..., gid, stat] -> [..., unit(4, last = ALL), period, band, stat]; 'worst' is a minimum, the rest are sums."""
    g = grid.reshape(grid.shape[:-2]+(len(GROUPS), 2, len(BANDS), len(STATS)))
    total = g.sum(axis=-4, keepdims=True)
    total[..., 9] = g[..., 9].min(axis=-3, keepdims=True)
    return np.concatenate([g, total], axis=-4)


def measures(u):
    """Per-cell means from the summed stats: returns dict of arrays over the leading axes."""
    s = {name: u[..., i] for i, name in enumerate(STATS)}
    with np.errstate(divide="ignore", invalid="ignore"):
        per_match = s["net"]/s["placed"]
        var = s["net_sq"]/s["placed"]-per_match**2
        se = np.sqrt(np.maximum(var, 0)/np.maximum(s["placed"]-1, 1))
        out = dict(per_match=per_match, se=se, t=per_match/se, per_cycle=s["net"]/s["cycles"], hold_per_cycle=s["hold"]/s["cycles"],
                   hit=s["target"]/s["cycles"], cycles=s["cycles"], placed=s["placed"], worst=s["worst"],
                   d1=s["d1"]/s["placed"], net1=s["net1"]/s["placed"], hold1=s["hold1"]/s["placed"])
        d1var = s["d1_sq"]/s["placed"]-out["d1"]**2
        out["d1_t"] = out["d1"]/np.sqrt(np.maximum(d1var, 0)/np.maximum(s["placed"]-1, 1))
    return out


def name_of(ix):
    dip, target, stop, cap, rule, band = ix
    return (f"P0 {BAND_NAMES[band]:12} bid {DIP_NAMES[dip]:5} sell {TARGET_NAMES[target]:3} stop {STOP_NAMES[stop]:4} "
            f"cycles {CAP_NAMES[cap]:3} sub-30 rule {RULES[rule]:3}")


def rule_axes(u):
    """[dip, target, stop, mode, cap, rule, unit, period, band, stat] -> [mode, unit, period, (dip, target, stop, cap, rule, band), stat]."""
    v = np.moveaxis(u, (3, 6, 7), (0, 1, 2))            # mode, unit, period, dip, target, stop, cap, rule, band, stat
    return v.reshape(v.shape[:3]+(-1, len(STATS))), v.shape[3:9]


def select(flat, unit, mode=0, loop=False):
    """Index of the rule with the best archive t-statistic (at least MIN_CYCLES archive cycles), or None.

    loop=True keeps only rules that really sell: at least half of their archive cycles end at the target. (With a target at
    99c, or far above the entry, a "loop" is the old hold-to-settlement rule under another name.)
    """
    ms = measures(flat[mode, unit, 0])
    ok = (ms["cycles"] >= MIN_CYCLES) & np.isfinite(ms["t"])
    if loop:
        ok &= ms["hit"] >= .5
    t = np.where(ok, ms["t"], -np.inf)
    i = int(np.argmax(t))
    return i if np.isfinite(t[i]) else None


def null_reps(steps):
    files = sorted((OUT/f"null-{steps}").glob("rep-*.npy"))
    return [np.load(f).astype(float) for f in files]


def do_headline(steps):
    real = by_unit(np.load(OUT/"grid.npy"))
    flat, dims = rule_axes(real)
    fair = [rule_axes(by_unit(r))[0] for r in null_reps(steps)]
    fsum = np.sum(fair, axis=0)
    print(f"\nHEADLINE  {flat.shape[3]} rules per execution and tour group; fair-market replay: {len(fair)} replications of {steps} steps")
    report = {}
    print(" A. Everything pooled (all rules in the grid, cycles weighted equally): net cents per cycle")
    print("    unit             execution  period    cycles  target-hit  net/cycle  hold-same-entries |  FAIR: target-hit  net/cycle  hold")
    for ui, unit in enumerate(UNITS):
        for mi, mode in enumerate(MODES):
            for pi, period in enumerate(PERIODS):
                r, f = flat[mi, ui, pi].sum(axis=0), fsum[mi, ui, pi].sum(axis=0)
                s = lambda x, k: x[STATS.index(k)]
                row = dict(cycles=s(r, "cycles"), hit=s(r, "target")/s(r, "cycles"), net=s(r, "net")/s(r, "cycles"), hold=s(r, "hold")/s(r, "cycles"),
                           fair_hit=s(f, "target")/s(f, "cycles"), fair_net=s(f, "net")/s(f, "cycles"), fair_hold=s(f, "hold")/s(f, "cycles"),
                           fair_cycles=s(f, "cycles")/len(fair))
                report[f"pooled | {unit} | {mode} | {period}"] = row
                print(f"    {unit:16} {mode:7}    {period:8} {row['cycles']:8.0f}  {row['hit']:9.3f}  {row['net']:+8.2f}  {row['hold']:+8.2f}          |"
                      f"  {row['fair_hit']:9.3f}       {row['fair_net']:+7.2f}  {row['fair_hold']:+6.2f}   (fair cycles {row['fair_cycles']:.0f})")
    print(" B. Rules with 100+ cycles in each period: how many are positive, real against fair (mean of replications, 95th percentile)")
    print("    unit             execution   rules   archive>0   both>0   archive t>2   archive t>2 & recent>0  | FAIR: archive>0  both>0  t>2  t>2&recent>0")

    def counts(x):
        a, b = measures(x[0]), measures(x[1])
        ok = (a["cycles"] >= MIN_CYCLES) & (b["cycles"] >= MIN_CYCLES)
        return np.array([ok.sum(), (ok & (a["per_match"] > 0)).sum(), (ok & (a["per_match"] > 0) & (b["per_match"] > 0)).sum(),
                         (ok & (a["t"] > 2)).sum(), (ok & (a["t"] > 2) & (b["per_match"] > 0)).sum()])
    for ui, unit in enumerate(UNITS):
        for mi, mode in enumerate(MODES):
            r = counts(flat[mi, ui])
            f = np.array([counts(x[mi, ui]) for x in fair])
            p = (f >= r).mean(axis=0)
            report[f"counts | {unit} | {mode}"] = dict(real=r.tolist(), fair_mean=f.mean(axis=0).tolist(), fair_p95=np.percentile(f, 95, axis=0).tolist(),
                                                      p_fair_at_least_real=p.tolist())
            print(f"    {unit:16} {mode:7}    {r[0]:5}   {r[1]:7}   {r[2]:6}   {r[3]:9}   {r[4]:12}            | "
                  f"{f[:, 1].mean():7.0f} ({np.percentile(f[:, 1], 95):.0f})  {f[:, 2].mean():5.0f} ({np.percentile(f[:, 2], 95):.0f})  "
                  f"{f[:, 3].mean():4.0f} ({np.percentile(f[:, 3], 95):.0f})  {f[:, 4].mean():4.0f} ({np.percentile(f[:, 4], 95):.0f})   "
                  f"P(fair>=real) {p[1]:.2f}/{p[2]:.2f}/{p[3]:.2f}/{p[4]:.2f}")
    print(" C. Selection: best archive t-statistic (strict), scored on the recent period; the same selection on each fair replay")
    for ui, unit in enumerate(UNITS):
        for loop, tag in ((False, "selected"), (True, "selected loop")):
            i = select(flat, ui, loop=loop)
            if i is None:
                continue
            ix = np.unravel_index(i, dims)
            a, b = measures(flat[0, ui, 0, i]), measures(flat[0, ui, 1, i])
            null = []
            for x in fair:
                j = select(x, ui, loop=loop)
                if j is not None:
                    na, nb = measures(x[0, ui, 0, j]), measures(x[0, ui, 1, j])
                    null.append((na["t"], na["per_match"], nb["per_match"], nb["t"]))
            null = np.array(null).reshape(-1, 4)
            others = {mode: (float(measures(flat[mi, ui, 0, i])["per_match"]), float(measures(flat[mi, ui, 1, i])["per_match"])) for mi, mode in enumerate(MODES)}
            keys = ("placed", "cycles", "per_match", "t", "per_cycle", "hit", "worst", "hold1", "net1")
            report[f"{tag} | {unit}"] = dict(rule=name_of(ix), index=[int(v) for v in ix], archive={k: float(a[k]) for k in keys}, recent={k: float(b[k]) for k in keys},
                                             by_execution=others,
                                             fair=dict(reps=len(null), best_archive_t_mean=float(null[:, 0].mean()), best_archive_t_max=float(null[:, 0].max()),
                                                       p_fair_t_at_least_real=float((null[:, 0] >= a["t"]).mean()),
                                                       recent_per_match_mean=float(null[:, 2].mean()), recent_t_max=float(null[:, 3].max()),
                                                       p_fair_recent_t_at_least_real=float((null[:, 3] >= b["t"]).mean())) if len(null) else None)
            print(f"  {unit}, {'rules that really sell (half or more of archive cycles end at the target)' if loop else 'any rule in the grid'}: {name_of(ix)}")
            for label, st in (("archive", a), ("recent", b)):
                print(f"      {label:8} matches placed {st['placed']:6.0f} cycles {st['cycles']:6.0f} target-hit {st['hit']:.3f} net/cycle {st['per_cycle']:+6.2f} "
                      f"net/match placed {st['per_match']:+6.2f} (t {st['t']:+.2f}) worst match {st['worst']:+.0f} | first entry held to settlement instead {st['hold1']:+.2f}/match")
            print("      same rule, net per match placed (archive, recent): "+"; ".join(f"{k} {v[0]:+.2f}, {v[1]:+.2f}" for k, v in others.items()))
            if len(null):
                print(f"      fair replay, same selection: best archive t mean {null[:, 0].mean():+.2f} (max {null[:, 0].max():+.2f}), "
                      f"P(fair best t >= real) {(null[:, 0] >= a['t']).mean():.2f}; its recent net/match mean {null[:, 2].mean():+.2f}, "
                      f"P(fair recent t >= real recent t) {(null[:, 3] >= b['t']).mean():.2f}")
            else:
                print("      fair replay, same selection: no rule qualifies in any replication")
    print(" D. Chronology across the grid (strict, rules with 100+ cycles in each period): does archive rank predict recent?")
    for ui, unit in enumerate(UNITS):
        a, b = measures(flat[0, ui, 0]), measures(flat[0, ui, 1])
        ok = (a["cycles"] >= MIN_CYCLES) & (b["cycles"] >= MIN_CYCLES)
        if ok.sum() < 20:
            continue
        x, y = a["per_match"][ok], b["per_match"][ok]
        slope, intercept = np.polyfit(x, y, 1)
        top = np.argsort(-a["t"][ok])[:20]
        report[f"chronology | {unit}"] = dict(rules=int(ok.sum()), corr=float(np.corrcoef(x, y)[0, 1]), slope=float(slope), intercept=float(intercept),
                                              archive_mean=float(x.mean()), recent_mean=float(y.mean()),
                                              top20_archive=float(x[top].mean()), top20_recent=float(y[top].mean()))
        print(f"    {unit:16} rules {ok.sum():4}  mean net/match archive {x.mean():+.2f} recent {y.mean():+.2f}  corr {np.corrcoef(x, y)[0, 1]:+.2f}  "
              f"recent = {intercept:+.2f} {slope:+.2f} x archive   top-20 by archive t: archive {x[top].mean():+.2f} -> recent {y[top].mean():+.2f}")
    print(" E. Where the positive rules are (strict, ALL tours, 100+ cycles in each period): by band, and by whether the rule really sells")
    print("    band           kind                rules  archive>0  both>0   mean net/match archive  recent | mean first-entry hold archive  recent")
    idx = np.array(np.unravel_index(np.arange(flat.shape[3]), dims)).T
    for ui in (len(GROUPS), 0):
        a, b = measures(flat[0, ui, 0]), measures(flat[0, ui, 1])
        ok = (a["cycles"] >= MIN_CYCLES) & (b["cycles"] >= MIN_CYCLES)
        print(f"   {UNITS[ui]}")
        for bi, band in enumerate(BAND_NAMES):
            for kind, mask in (("sells (hit>=50%)", a["hit"] >= .5), ("mostly holds", a["hit"] < .5)):
                k = ok & (idx[:, 5] == bi) & mask
                if not k.any():
                    continue
                row = dict(rules=int(k.sum()), archive_pos=int((k & (a["per_match"] > 0)).sum()), both_pos=int((k & (a["per_match"] > 0) & (b["per_match"] > 0)).sum()),
                           archive=float(a["per_match"][k].mean()), recent=float(b["per_match"][k].mean()),
                           hold_archive=float(a["hold1"][k].mean()), hold_recent=float(b["hold1"][k].mean()))
                report[f"where | {UNITS[ui]} | {band} | {kind}"] = row
                print(f"    {band:13} {kind:18} {row['rules']:5}  {row['archive_pos']:7}  {row['both_pos']:6}        {row['archive']:+6.2f}        {row['recent']:+6.2f}    |"
                      f"        {row['hold_archive']:+6.2f}          {row['hold_recent']:+6.2f}")
    print(" F. Marginal effect of each knob (ALL tours, cycles pooled over the rest of the grid): net cents per cycle, archive / recent | fair")
    full = np.load(OUT/"grid.npy")
    u = by_unit(full)                                    # dip, target, stop, mode, cap, rule, unit, period, band, stat
    fu = np.mean([by_unit(r) for r in null_reps(steps)], axis=0)
    ci, ni, pmi = STATS.index("cycles"), STATS.index("net"), STATS.index("placed")
    for axis, label, names in ((0, "entry level", DIP_NAMES), (1, "target", TARGET_NAMES), (2, "stop", STOP_NAMES), (4, "max cycles", CAP_NAMES),
                               (5, "sub-30 rule", RULES), (8, "band", BAND_NAMES)):
        for mi, mode in enumerate(MODES):
            cells = []
            for j, nm in enumerate(names):
                def take(x, pi):
                    pick = [slice(None)]*x.ndim
                    pick[3], pick[6], pick[7], pick[axis] = mi, len(GROUPS), pi, j
                    y = x[tuple(pick)].reshape(-1, len(STATS)).sum(axis=0)
                    return y[ni]/y[ci], y[ni]/y[pmi]
                ra, rb, fa = take(u, 0), take(u, 1), take(fu, 0)
                report[f"marginal | {label} | {nm} | {mode}"] = dict(archive_per_cycle=ra[0], recent_per_cycle=rb[0], fair_per_cycle=fa[0],
                                                                     archive_per_match=ra[1], recent_per_match=rb[1], fair_per_match=fa[1])
                cells.append(f"{nm} {ra[0]:+.2f}/{rb[0]:+.2f}|{fa[0]:+.2f} (per match {ra[1]:+.2f}/{rb[1]:+.2f}|{fa[1]:+.2f})")
            print(f"    {label:12} {mode:7} "+"   ".join(cells))
    (OUT/"headline.json").write_text(json.dumps(report, indent=1, default=float))
    table(flat, dims, fsum/len(fair))


def table(flat, dims, fair_mean):
    """Every rule, unit, execution: both periods and the fair-replay mean, to CSV."""
    rows = []
    idx = np.array(np.unravel_index(np.arange(flat.shape[3]), dims)).T
    for mi, mode in enumerate(MODES):
        for ui, unit in enumerate(UNITS):
            ms = [measures(flat[mi, ui, pi]) for pi in (0, 1)]
            fs = [measures(fair_mean[mi, ui, pi]) for pi in (0, 1)]
            raw = [flat[mi, ui, pi] for pi in (0, 1)]
            frame = pd.DataFrame(dict(unit=unit, execution=mode, band=[BAND_NAMES[i] for i in idx[:, 5]], bid=[DIP_NAMES[i] for i in idx[:, 0]],
                                      sell=[TARGET_NAMES[i] for i in idx[:, 1]], stop=[STOP_NAMES[i] for i in idx[:, 2]],
                                      max_cycles=[CAP_NAMES[i] for i in idx[:, 3]], sub30_rule=[RULES[i] for i in idx[:, 4]]))
            for pi, period in enumerate(PERIODS):
                for k, name in enumerate(STATS[:7]):
                    frame[f"{period}_{name}"] = raw[pi][:, k]
                for name in ("per_cycle", "per_match", "se", "t", "worst", "hold_per_cycle", "d1"):
                    frame[f"{period}_{name}"] = ms[pi][name]
                for name in ("per_cycle", "per_match", "hit", "hold_per_cycle"):
                    frame[f"{period}_fair_{name}"] = fs[pi][name]
            rows.append(frame[(frame.archive_cycles > 0) | (frame.recent_cycles > 0)])
    pd.concat(rows).to_csv(OUT/"grid.csv", index=False, float_format="%.3f")


# ---------------------------------------------------------------- decomposition

def do_decompose(steps):
    real = np.load(OUT/"grid.npy")
    u = by_unit(real)                                    # dip, target, stop, mode, cap, rule, unit, period, band, stat
    fair = np.mean([by_unit(r) for r in null_reps(steps)], axis=0)
    report = {}
    s = lambda k: STATS.index(k)
    print("\nDECOMPOSE  strict execution, no stop, no sub-30 rule. Cents per match placed, summed over the 7 entry levels of each band.")
    print(" hold = first entry held to settlement; exit = what selling at the target changes on that first entry; re-entry = cycles 2+.")
    for ti, target in enumerate(TARGET_NAMES):
        print(f" target {target}")
        print("    unit             band           period   placed  entries/match |   hold   +exit  = cycle1 | +re-entry = loop  | FAIR: hold  +exit  +re-entry = loop")
        for ui, unit in enumerate(UNITS):
            for bi, band in enumerate(BAND_NAMES):
                for pi, period in enumerate(PERIODS):
                    def parts(x):
                        one, unl = x[:, ti, 0, 0, 0, 0, ui, pi, bi].sum(axis=0), x[:, ti, 0, 0, 3, 0, ui, pi, bi].sum(axis=0)
                        n = one[s("placed")]
                        if n == 0:
                            return None
                        return n, unl[s("cycles")]/n, one[s("hold1")]/n, one[s("d1")]/n, one[s("net1")]/n, (unl[s("net")]-one[s("net1")])/n, unl[s("net")]/n
                    r, f = parts(u), parts(fair)
                    if r is None or f is None or r[0] < 50*len(DIPS):
                        continue
                    report[f"{target} | {unit} | {band} | {period}"] = dict(zip(("placed", "cycles_per_match", "hold", "exit", "cycle1", "reentry", "loop"), r),
                                                                           fair=dict(zip(("hold", "exit", "reentry", "loop"), (f[2], f[3], f[5], f[6]))))
                    print(f"    {unit:16} {band:13} {period:8} {r[0]/len(DIPS):6.0f}  {r[1]:6.2f}        | {r[2]:+6.2f} {r[3]:+6.2f}  = {r[4]:+6.2f} | {r[5]:+6.2f}  = {r[6]:+6.2f} |"
                          f"     {f[2]:+6.2f} {f[3]:+6.2f}  {f[5]:+6.2f}  = {f[6]:+6.2f}")
    # Does selling on the rebound beat holding, cell by cell (cap 1, no stop, rule off)?
    print(" Sell-on-rebound minus hold-to-settlement on the first entry, per cell (band x entry level x target), strict, no stop:")
    print("    unit             cells(50+ entries each period)  exit>hold archive  recent  both  | both & loop>0 both | FAIR both (mean, 95th pct)")
    reps = [by_unit(r) for r in null_reps(steps)]

    def cells(x, ui):
        c = x[:, :, 0, 0, 0, 0, ui]                      # dip, target, period, band, stat
        ok = (c[:, :, 0, :, s("cycles")] >= 50) & (c[:, :, 1, :, s("cycles")] >= 50)
        da, db = c[:, :, 0, :, s("d1")], c[:, :, 1, :, s("d1")]
        la, lb = c[:, :, 0, :, s("net1")], c[:, :, 1, :, s("net1")]
        return ok, da, db, la, lb
    for ui, unit in enumerate(UNITS):
        ok, da, db, la, lb = cells(u, ui)
        both = ok & (da > 0) & (db > 0)
        fb = [(lambda o, a, b, *_: (o & (a > 0) & (b > 0)).sum())(*cells(x, ui)) for x in reps]
        report[f"exit beats hold | {unit}"] = dict(cells=int(ok.sum()), archive=int((ok & (da > 0)).sum()), recent=int((ok & (db > 0)).sum()), both=int(both.sum()),
                                                   both_and_positive=int((both & (la > 0) & (lb > 0)).sum()), fair_both_mean=float(np.mean(fb)),
                                                   fair_both_p95=float(np.percentile(fb, 95)),
                                                   which=[f"{BAND_NAMES[b]} bid {DIP_NAMES[d_]} sell {TARGET_NAMES[t]}: exit-hold {da[d_, t, b]/c1:+.2f}/{db[d_, t, b]/c2:+.2f}, "
                                                          f"cycle {la[d_, t, b]/c1:+.2f}/{lb[d_, t, b]/c2:+.2f} per entry"
                                                          for d_, t, b in zip(*np.where(both))
                                                          for c1, c2 in [(u[d_, t, 0, 0, 0, 0, ui, 0, b, s("cycles")], u[d_, t, 0, 0, 0, 0, ui, 1, b, s("cycles")])]])
        print(f"    {unit:16} {ok.sum():5}                            {(ok & (da > 0)).sum():5}          {(ok & (db > 0)).sum():5}  {both.sum():4}  | "
              f"{(both & (la > 0) & (lb > 0)).sum():4}                | {np.mean(fb):.1f} ({np.percentile(fb, 95):.0f})")
        for line in report[f"exit beats hold | {unit}"]["which"]:
            print(f"        {line}")
    # The win rate the loop needs against the one it gets (strict, cap 1, no stop): needed = share of cycles that must hit the target.
    print(" Win rate needed against win rate achieved (strict, first cycle only, no stop), by band, entry -15c, all units pooled and by unit:")
    print("    unit             band           target  period    cycles   hit    settled-win  settled-loss")
    for ui, unit in enumerate(UNITS):
        for bi, band in enumerate(BAND_NAMES):
            for ti in (1, 3, 4):
                for pi, period in enumerate(PERIODS):
                    c, f = u[1, ti, 0, 0, 0, 0, ui, pi, bi], fair[1, ti, 0, 0, 0, 0, ui, pi, bi]
                    n = c[s("cycles")]
                    if n < 50:
                        continue
                    hit, win, loss = c[s("target")]/n, c[s("win")]/n, c[s("loss")]/n
                    report[f"winrate | {unit} | {band} | {TARGET_NAMES[ti]} | {period}"] = dict(cycles=float(n), hit=float(hit), settled_win=float(win), settled_loss=float(loss),
                                                                                             net=float(c[s("net")]/n), fair_hit=float(f[s("target")]/max(f[s("cycles")], 1)),
                                                                                             fair_net=float(f[s("net")]/max(f[s("cycles")], 1)))
                    print(f"    {unit:16} {band:13} {TARGET_NAMES[ti]:5}   {period:8} {n:6.0f}  {hit:.3f}    {win:.3f}       {loss:.3f}       net/cycle {c[s('net')]/n:+6.2f}"
                          f"     fair: hit {f[s('target')]/max(f[s('cycles')], 1):.3f} net/cycle {f[s('net')]/max(f[s('cycles')], 1):+.2f}")
    (OUT/"decompose.json").write_text(json.dumps(report, indent=1, default=float))


# ---------------------------------------------------------------- one rule in full

def boot(values, draws=4000):
    values = np.asarray(values, float)
    if len(values) < 10:
        return (np.nan, np.nan)
    means = values[RNG.integers(0, len(values), (draws, len(values)))].mean(axis=1)
    return tuple(np.percentile(means, [2.5, 97.5]))


def rule_detail(d, pl, ix, mode="strict", gate=None):
    """Run one rule; return per-cycle rows (kept by the rule's cap and time rule) and the per-row 'placed' flag."""
    dip, target, stop, cap, rule, band = ix
    L, T, valid = levels(d, dip, target)
    valid = valid & (pl.band.values == band)
    net, hold, outc, placed, cycles = run(d, L, T, valid, STOPS[stop], mode, gate=gate, detail=True)
    cycles = cycles[(cycles.cycle <= CAPS[cap]) & ((cycles.bucket < 4) | (rule == 0))].copy()
    cycles["net"] = cycles.proceeds-cycles.cost
    cycles["hold"] = cycles.settle_value-cycles.cost
    for col in ("mid", "unit", "pi", "tour"):
        cycles[col] = pl[col].values[cycles.row.values.astype(int)]
    return cycles, placed & valid


def per_match(cycles, placed, pl, unit, pi, col="net"):
    """Net per match placed (zeros for matches placed but never filled)."""
    keep = placed & (pl.pi.values == pi) & ((pl.unit.values == unit) if unit < len(GROUPS) else True)
    mids = np.unique(pl.mid.values[keep])
    sums = cycles[cycles.mid.isin(mids)].groupby("mid")[col].sum()
    return pd.Series(0.0, index=mids).add(sums, fill_value=0.0)


def summary(cycles, placed, pl, unit, pi):
    x = per_match(cycles, placed, pl, unit, pi)
    c = cycles[cycles.mid.isin(x.index)]
    if not len(x):
        return None
    lo, hi = boot(x.values)
    counts = c.outcome.value_counts()
    up, down = c.net[c.net > 0], c.net[c.net <= 0]
    arithmetic = dict(won=float(len(up)/len(c)), avg_win=float(up.mean()), avg_loss=float(-down.mean()),
                      needed=float(-down.mean()/(up.mean()-down.mean()))) if len(up) and len(down) else None
    return dict(arithmetic=arithmetic, placed=len(x), matches_entered=int(c.mid.nunique()), cycles=len(c), **{name: int(counts.get(i, 0)) for i, name in enumerate(OUTCOMES)},
                per_cycle=float(c.net.mean()) if len(c) else None, per_match=float(x.mean()), ci=[float(lo), float(hi)], worst=float(x.min()),
                hold_per_cycle=float(c.hold.mean()) if len(c) else None,
                vol_in_median=float(c.vol_in.median()) if len(c) else None, vol_in_p25=float(c.vol_in.quantile(.25)) if len(c) else None,
                vol_out_median=float(c[c.outcome == 0].vol_out.median()) if (c.outcome == 0).any() else None)


def show(label, s):
    if s is None:
        print(f"      {label:34} no matches")
        return
    print(f"      {label:34} placed {s['placed']:5} entered {s['matches_entered']:5} cycles {s['cycles']:5} "
          f"[target {s['target']} / stopped {s['stopped']} / settled win {s['settled win']} / settled loss {s['settled loss']}] "
          f"net/cycle {s['per_cycle'] if s['per_cycle'] is not None else float('nan'):+6.2f} net/match placed {s['per_match']:+6.2f} "
          f"[{s['ci'][0]:+.2f},{s['ci'][1]:+.2f}] worst {s['worst']:+.0f}")


def paired(cycles, placed, pl, unit, pi):
    """Loop net minus 'hold the first entry to settlement', per match placed, with a bootstrap interval."""
    x = per_match(cycles, placed, pl, unit, pi)
    h = per_match(cycles[cycles.cycle == 1], placed, pl, unit, pi, col="hold")
    if len(x) < 10:
        return None
    diff = (x-h).values
    return dict(loop=float(x.mean()), hold=float(h.mean()), hold_ci=list(boot(h.values)), diff=float(diff.mean()), diff_ci=list(boot(diff)))


def full_rule(d, pl, ix, ui, label, report):
    print(f"  {label}: {name_of(ix)}")
    rep = report[label] = dict(rule=name_of(ix))
    for mode in MODES:
        cycles, placed = rule_detail(d, pl, ix, mode)
        for pi, period in enumerate(PERIODS):
            st = summary(cycles, placed, pl, ui, pi)
            rep[f"{mode} | {period}"] = st
            show(f"{mode} {period}", st)
            if mode == "strict" and st and st["cycles"]:
                months = RECENT_MONTHS if pi else ARCHIVE_MONTHS
                pr = paired(cycles, placed, pl, ui, pi)
                rep[f"paired | {period}"] = pr
                ar = st["arithmetic"]
                print(f"          {st['cycles']/months:.1f} cycles a month in {st['matches_entered']/months:.1f} matches a month; contracts traded in the fill minute: "
                      f"entry median {st['vol_in_median']:.0f} (p25 {st['vol_in_p25']:.0f}), target exit median {st['vol_out_median'] or float('nan'):.0f}")
                if ar:
                    print(f"          cycles won {ar['won']:.1%}, average win {ar['avg_win']:+.1f}c, average loss -{ar['avg_loss']:.1f}c -> needs {ar['needed']:.1%} to break even")
                if pr:
                    print(f"          against holding the first entry to settlement: loop {pr['loop']:+.2f}, hold {pr['hold']:+.2f} [{pr['hold_ci'][0]:+.2f},{pr['hold_ci'][1]:+.2f}], "
                          f"loop minus hold {pr['diff']:+.2f} [{pr['diff_ci'][0]:+.2f},{pr['diff_ci'][1]:+.2f}] per match placed")
    return rep


def do_verdict(steps):
    m, pl, d = real_data()
    head = json.loads((OUT/"headline.json").read_text())
    report = {}
    print("\nVERDICT  rules selected on the archive, in full (bootstrap 95% intervals resampling matches)")
    for ui, unit in enumerate(UNITS):
        for tag in ("selected", "selected loop"):
            sel = head.get(f"{tag} | {unit}")
            if not sel:
                continue
            rep = full_rule(d, pl, tuple(sel["index"]), ui, f"{tag} | {unit}", report)
            chron = head.get(f"chronology | {unit}")
            if chron:
                a = sel["archive"]["per_match"]
                forward = chron["intercept"]+chron["slope"]*a
                rep["forward_regression"] = forward
                print(f"      forward guess from the grid's own archive->recent regression: {chron['intercept']:+.2f} {chron['slope']:+.2f} x {a:+.2f} = {forward:+.2f}c per match placed "
                      f"(recent actual {sel['recent']['per_match']:+.2f})")
    print("  The live rule's home turf: 90c+ favourites, bid 10c below the pre-match price, Challenger + WTA Challenger, and all tours")
    for ui in (0, len(GROUPS)):
        for name, ix in (("hold to settlement (target 99)", (0, 3, 0, 0, 0, 4)), ("sell at pre-match price, once", (0, 4, 0, 0, 0, 4)),
                         ("sell +5, repeat", (0, 0, 0, 3, 0, 4)), ("sell +10, repeat", (0, 1, 0, 3, 0, 4)), ("sell at pre-match price, repeat", (0, 4, 0, 3, 0, 4))):
            full_rule(d, pl, ix, ui, f"home | {UNITS[ui]} | {name}", report)
    print("  Reference loops in the weaker bands (bid 15c below the pre-match price, sell +10, no stop, unlimited cycles), all tours")
    for band in range(4):
        full_rule(d, pl, (1, 1, 0, 3, 0, band), len(GROUPS), f"reference | {BAND_NAMES[band]}", report)
    (OUT/"verdict.json").write_text(json.dumps(report, indent=1, default=float))


# ---------------------------------------------------------------- choppiness

def swings(d, size):
    """Per row: completed swings of `size` cents in the midpoint over the match, and the bar at which the 2nd completed.

    The first swing is a move of `size` from the midpoint at the start; each later one is a reversal of `size` from the running extreme.
    """
    n = d.n
    mid = (d.bid+d.ask)/2
    ext, direction = mid[:, 0].copy(), np.zeros(n, np.int8)
    count, second = np.zeros(n, np.int32), np.full(n, 10**6, np.int32)
    for t in range(1, d.width):
        k = int(d.alive[t])
        x, e, dr = mid[:k, t], ext[:k], direction[:k]
        blank = np.isnan(e) & ~np.isnan(x)
        e[blank] = x[blank]
        up = ((dr == 0) | (dr == -1)) & (x-e >= size)        # NaN compares False
        down = ((dr == 0) | (dr == 1)) & (e-x >= size)
        done = up | down
        count[:k] += done
        second[np.flatnonzero(done & (count[:k] == 2))] = t
        dr[up], dr[down] = 1, -1
        move = done | ((dr == 1) & (x > e)) | ((dr == -1) & (x < e))
        e[move] = x[move]
    return count, second


def early_range(d, minutes=20):
    """Midpoint range over the first `minutes` after the detected start, and the first bar after them."""
    mid = (d.bid+d.ask)/2
    early = d.ts <= 60*minutes
    lo = np.nanmin(np.where(early, mid, np.nan), axis=1)
    hi = np.nanmax(np.where(early, mid, np.nan), axis=1)
    gate = early.sum(axis=1)                              # index of the first bar later than `minutes`
    return hi-lo, gate


def do_choppy():
    m, pl, d = real_data()
    head = json.loads((OUT/"headline.json").read_text())
    report = {}
    count10, second10 = swings(d, 10)
    count15, second15 = swings(d, 15)
    rng20, gate20 = early_range(d, 20)
    rules = {"selected loop ALL": tuple(head["selected loop | ALL"]["index"])}
    for band in range(len(BANDS)):                        # one plain reference rule per band: bid -15c, sell +10, no stop, unlimited
        rules[f"reference {BAND_NAMES[band]}"] = (1, 1, 0, 3, 0, band)
    print("\nCHOPPY  strict execution. 'swings' = completed 10c reversals of the midpoint over the whole match (hindsight).")
    for name, ix in rules.items():
        print(f"  {name}: {name_of(ix)}")
        cycles, placed = rule_detail(d, pl, ix)
        rowsum = cycles.groupby("row").net.sum().reindex(range(d.n), fill_value=0.0).values
        ncyc = cycles.groupby("row").size().reindex(range(d.n), fill_value=0).values
        base = placed
        rep = report[name] = {}
        print("    (a) hindsight: by number of 10c swings in the match")
        for lo, hi in ((0, 2), (2, 4), (4, 7), (7, 999)):
            for pi, period in enumerate(PERIODS):
                k = base & (count10 >= lo) & (count10 < hi) & (pl.pi.values == pi)
                if k.sum() < 20:
                    continue
                ci = boot(rowsum[k])
                rep[f"swings {lo}-{hi} | {period}"] = dict(n=int(k.sum()), cycles=float(ncyc[k].mean()), net=float(rowsum[k].mean()), ci=ci)
                print(f"        swings {lo}-{hi if hi < 999 else '+'}  {period:8} players {k.sum():5} cycles/player {ncyc[k].mean():.2f} net/player {rowsum[k].mean():+6.2f} [{ci[0]:+.2f},{ci[1]:+.2f}]")
        tot = rowsum[base]
        order = np.sort(tot)[::-1]
        rep["concentration"] = dict(total=float(tot.sum()), top_decile_sum=float(order[:len(order)//10].sum()), bottom_decile_sum=float(order[-(len(order)//10):].sum()),
                                    share_losing=float((tot < 0).mean()), share_winning=float((tot > 0).mean()))
        print(f"        all placed: total {tot.sum():+.0f}c; best 10% of players {order[:len(order)//10].sum():+.0f}c, worst 10% {order[-(len(order)//10):].sum():+.0f}c; "
              f"{(tot > 0).mean():.0%} of players end up, {(tot < 0).mean():.0%} down")
        print("    (b) known in time: by tour group")
        for ui, unit in enumerate(GROUPS):
            for pi, period in enumerate(PERIODS):
                k = base & (pl.unit.values == ui) & (pl.pi.values == pi)
                if k.sum() < 20:
                    continue
                ci = boot(rowsum[k])
                rep[f"unit {unit} | {period}"] = dict(n=int(k.sum()), swings=float(count10[k].mean()), net=float(rowsum[k].mean()), ci=ci)
                print(f"        {unit:16} {period:8} players {k.sum():5} mean 10c swings {count10[k].mean():.2f} net/player {rowsum[k].mean():+6.2f} [{ci[0]:+.2f},{ci[1]:+.2f}]")
        print("    (c) known in time: loop armed only from minute 20, split by the midpoint range of the first 20 minutes (terciles of the archive)")
        cycles_g, placed_g = rule_detail(d, pl, ix, gate=gate20)
        rs = cycles_g.groupby("row").net.sum().reindex(range(d.n), fill_value=0.0).values
        cuts = np.nanpercentile(rng20[placed_g & (pl.pi.values == 0)], [33.3, 66.7]) if (placed_g & (pl.pi.values == 0)).sum() > 30 else (np.nan, np.nan)
        for label, k0 in (("quiet", rng20 <= cuts[0]), ("middle", (rng20 > cuts[0]) & (rng20 <= cuts[1])), ("wild", rng20 > cuts[1])):
            for pi, period in enumerate(PERIODS):
                k = placed_g & k0 & (pl.pi.values == pi)
                if k.sum() < 20:
                    continue
                ci = boot(rs[k])
                rep[f"early range {label} | {period}"] = dict(n=int(k.sum()), cuts=[float(c) for c in cuts], swings_whole_match=float(count10[k].mean()), net=float(rs[k].mean()), ci=ci)
                print(f"        first-20-min range {label:6} (cuts {cuts[0]:.0f}c/{cuts[1]:.0f}c) {period:8} players {k.sum():5} whole-match 10c swings {count10[k].mean():.2f} "
                      f"net/player {rs[k].mean():+6.2f} [{ci[0]:+.2f},{ci[1]:+.2f}]")
        print("    (d) known in time: loop armed only after the midpoint has already completed two swings of X cents")
        for size, second in ((10, second10), (15, second15)):
            cyc_s, placed_s = rule_detail(d, pl, ix, gate=second)
            rs = cyc_s.groupby("row").net.sum().reindex(range(d.n), fill_value=0.0).values
            nc = cyc_s.groupby("row").size().reindex(range(d.n), fill_value=0).values
            for pi, period in enumerate(PERIODS):
                k = placed_s & (pl.pi.values == pi)
                if k.sum() < 20:
                    continue
                ci = boot(rs[k])
                per_cycle = rs[k].sum()/max(nc[k].sum(), 1)
                rep[f"after two {size}c swings | {period}"] = dict(n=int(k.sum()), cycles=int(nc[k].sum()), per_cycle=float(per_cycle), net=float(rs[k].mean()), ci=ci)
                print(f"        after two {size}c swings  {period:8} players placed {k.sum():5} cycles {nc[k].sum():5} net/cycle {per_cycle:+6.2f} "
                      f"net/player {rs[k].mean():+6.2f} [{ci[0]:+.2f},{ci[1]:+.2f}]")
    # Is choppiness itself predictable? Later swings against the early range, the pre-match price and the tour.
    print("  Is choppiness predictable? Completed 10c swings over the whole match (all players with P0 >= 45)")
    c_all = count10
    for label, key in (("pre-match band", pl.band.map(lambda b: BAND_NAMES[b]).values), ("tour group", pl.unit.map(lambda u_: GROUPS[u_]).values)):
        for value in pd.unique(key):
            k = key == value
            print(f"        {label:14} {value:16} players {k.sum():6}  10c swings per match {c_all[k].mean():.2f}  share with 4+ {np.mean(c_all[k] >= 4):.2f}")
            report[f"swings by {label} | {value}"] = dict(n=int(k.sum()), swings=float(c_all[k].mean()), share4=float(np.mean(c_all[k] >= 4)))
    ok = ~np.isnan(rng20)
    r = np.corrcoef(rng20[ok], c_all[ok])[0, 1]
    report["corr early range vs whole-match swings"] = float(r)
    print(f"        correlation of first-20-minute range with whole-match 10c swings: {r:+.2f} (the early range is part of the whole match, so this overstates it)")
    (OUT/"choppy.json").write_text(json.dumps(report, indent=1, default=float))


def do_sensitivity():
    """The pooled fair-market cost per cycle for every walk length that has replications on disk, beside the real figure."""
    real = rule_axes(by_unit(np.load(OUT/"grid.npy")))[0]
    ui, k = len(GROUPS), lambda name: STATS.index(name)
    report = {}
    print("\nSENSITIVITY  all tours, all rules pooled: target-hit share and net cents per cycle (archive / recent)")
    for mi, mode in enumerate(MODES):
        r = [real[mi, ui, pi].sum(axis=0) for pi in (0, 1)]
        line = f"  {mode:7} real hit {r[0][k('target')]/r[0][k('cycles')]:.3f}/{r[1][k('target')]/r[1][k('cycles')]:.3f} net {r[0][k('net')]/r[0][k('cycles')]:+.2f}/{r[1][k('net')]/r[1][k('cycles')]:+.2f}"
        for folder in sorted(OUT.glob("null-*"), key=lambda f: int(f.name.split("-")[1])):
            steps = int(folder.name.split("-")[1])
            reps = null_reps(steps)
            if not reps:
                continue
            f = np.sum([rule_axes(by_unit(x))[0][mi, ui].sum(axis=1) for x in reps], axis=0)      # period, stat
            report[f"{mode} | {steps}"] = dict(reps=len(reps), hit=float(f[:, k("target")].sum()/f[:, k("cycles")].sum()),
                                               net=float(f[:, k("net")].sum()/f[:, k("cycles")].sum()), hold=float(f[:, k("hold")].sum()/f[:, k("cycles")].sum()))
            line += f" | fair {steps} steps ({len(reps)} reps): hit {report[f'{mode} | {steps}']['hit']:.3f} net {report[f'{mode} | {steps}']['net']:+.2f}"
        print(line)
    (OUT/"sensitivity.json").write_text(json.dumps(report, indent=1))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("steps", nargs="+", choices=("build", "grid", "null", "headline", "decompose", "choppy", "verdict", "sensitivity"))
    parser.add_argument("--reps", type=int, default=24, help="fair-market replications for `null`")
    parser.add_argument("--seed", type=int, default=0, help="first replication number (replications are resumable)")
    parser.add_argument("--walk-steps", type=int, default=700, help="steps per simulated match")
    parser.add_argument("--workers", type=int, default=12)
    args = parser.parse_args()
    for step in args.steps:
        if step == "build":
            build()
        elif step == "grid":
            do_grid(args.workers)
        elif step == "null":
            do_null(args.reps, args.walk_steps, args.workers, args.seed)
        elif step == "headline":
            do_headline(args.walk_steps)
        elif step == "decompose":
            do_decompose(args.walk_steps)
        elif step == "choppy":
            do_choppy()
        elif step == "sensitivity":
            do_sensitivity()
        else:
            do_verdict(args.walk_steps)
