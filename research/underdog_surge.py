"""Momentum in underdog surges: does buying a side whose in-play price has just surged pay on Kalshi game-winner markets?

The idea under test (the owner's words): "momentum swings where the underdog
sweeps up within a threshold that closes out for a win". Rule: when a side's
price surges in play, BUY it at the next minute's opening ask and hold to
settlement (exits are tested too). Offline only: every bar comes from the
cache under data/raw/cost-research-v1; nothing is fetched, no orders, no
authentication, nothing under paper/ is read or run.

Data (one-minute in-play grids, YES side of the alphabetically-first market,
one terminal point per game holding the settlement value)
* tennis: data/processed/tennis-reversion/grid.npz (27,741 matches; groups
  ATP+WTA, Challenger incl. WTA Challenger, ITF). Start of play = detected.
* basketball: data/processed/basketball-reversion/grid.npz (men's and women's
  college, NBA; period and clock from play-by-play where linked. The SCORE is
  never used here, so the archive's score lookahead flaw cannot matter).
* nhl, mlb, nfl, college football: built here (step `build`) from the cached
  six-hour windows; start = time in the ticker, else expected expiration - 3h.
Minutes with no bar, or with a one-sided quote, carry the last two-sided closing quote forward. Both sides of every game are
studied (the other side is the same book read from the other side).

Definitions (fixed before any result was looked at)
* Pre-game price P0 of a side: as built by the source grids (tennis: median of
  the first 15 bars of the window; others: the last quotes before the start).
  Class of the SURGING side: heavy underdog (P0 < 20), underdog (20-35), mild
  underdog (35-50), favourite (50+, the control). "dog<50" pools the first three.
* Surge, on the side's midpoint at a bar's close:
  (a) rise X/M: the midpoint is X cents (10, 15, 20, 30) or more above its
      level M minutes (2, 5, 10) earlier. Event = the first bar of an episode
      (no such bar in the previous M minutes).
  (b) cross L: the midpoint closes at or above L (40, 50, 60, 70, 80; and 85,
      90, 95 for the late close-outs) having closed the previous minute below
      it. Speed classes: f10in2, f10in5, f15in5, f15in10, f20in10 = it was at
      least D cents below L at some close within the last M minutes; slow = at
      least ten minutes of play and never 10c below L in the last ten minutes;
      any. One event per side, game, level and class: the first that qualifies.
  (c) sweep: the midpoint closes above 65 having been below 35 at a close
      within the last 15 minutes (first per side and game).
* Execution. FOLLOW buys the surging side at the NEXT minute's opening ask
  (the last closing ask if that minute has no bar), ask in 1..99, taker fee
  0.07*p*(1-p). No trade if the game is already over. Outcomes: hold to
  settlement; sell at the closing bid 5, 15, 30 minutes after entry (taker
  fee; if the game ends first the position settles: nothing is conditioned on
  the market still being quoted); a resting offer 5c or 10c above the entry
  filled only by a LATER bar with a print strictly above it (maker fee
  0.0175*p*(1-p) assumed), else settlement. FADE buys the other side the same way.
* Net cents per contract; % return = mean of net/price paid (equal dollars per
  trade; `pct_w` in the CSV is total net over total paid). `need` = the win
  rate that breaks even at the price paid (ask + fee). Intervals are 95%,
  clustered by game. Halves: each group's games split at the median settlement.
* Phase: thirds of the group's median in-play length by minutes elapsed
  (knowable live); NBA quarters and college halves from the play-by-play clock.
* Fair-market replay (step 5): every real game redrawn as the martingale
  p_k = Phi(W_k / sqrt(1 - k/N)) started at its P0, N = its real number of
  in-play minutes, quoted 1c either side, settling at 0 or 100; the identical
  detection, execution, cells and pass standard are applied. In it the price IS
  the probability, so it shows what "the surging side usually wins" looks like
  when there is no edge, and how many cells pass by chance.
* Pass standard (step 7): 30+ trades in each half, first-half interval above
  zero (test interval: clustered, plus one unseen full loss, and for holds at
  least the binomial variance at the price paid, so that a cell with no losers
  yet is not read as certain), second-half mean positive, and every neighbouring threshold (adjacent
  X or M; adjacent L at the same speed) with 20+ trades positive on both
  halves pooled, with at least one such neighbour (sweep has none: waived).
  Survivors are shrunk towards the mean of all cells (empirical Bayes: true
  spread = observed variance of cell means minus their sampling variance).

    .venv/bin/python research/underdog_surge.py build
    .venv/bin/python research/underdog_surge.py run --reps 10

Outputs under data/processed/underdog-surge/: cells.csv (every cell), report.txt
(the printed report), fair-cells.csv, null.json, summary.json, events.pkl.

Result of the 18 November 2026 run (27,741 tennis matches, 5,805 basketball
games, 4,602 nhl/mlb/nfl/college football games; 2.4 million surge events): no
momentum. Underdog surges held to settlement return about -5% in tennis, -1% to
-4% in men's college basketball, the NBA, NHL and MLB, -22% in women's college
basketball; every exit is worse. Fast crossers win no more than slow crossers
at the same price (slope of excess win on speed -0.4c per 10c in tennis, zero
elsewhere). Former underdogs bought at 85-95c lose about 2%. Of 17,868 follow
cells one passes (a favourite control cell), against 2.7 in fair-market
replays. Fading loses about 3c too, except where the faller was a 90c+ favourite.
"""
import argparse
from collections import Counter
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd
from scipy.special import ndtr, ndtri

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "data/processed/underdog-surge"
TENNIS = ROOT / "data/processed/tennis-reversion"
BBALL = ROOT / "data/processed/basketball-reversion"
OTHER = {"nhl": "KXNHLGAME", "mlb": "KXMLBGAME", "nfl": "KXNFLGAME", "ncaaf": "KXNCAAFGAME"}
TOUR_GROUP = {"ATP": "ATP+WTA", "WTA": "ATP+WTA", "Challenger": "Challenger", "WTA Challenger": "Challenger",
              "ITF men": "ITF", "ITF women": "ITF"}
GROUPS = ["ATP+WTA", "Challenger", "ITF", "cbb-men", "cbb-women", "nba", "nhl", "mlb", "nfl", "ncaaf"]
POOLED_GROUP = "tennis"                                    # the three tennis groups pooled
ALL_GROUPS = GROUPS+[POOLED_GROUP]
CLASSES = ["heavy<20", "dog20-35", "mild35-50", "fav50+"]
ALL_CLASSES = CLASSES+["dog<50"]
HALVES = ["half1", "half2", "both"]
RISE_X, RISE_M = (10, 15, 20, 30), (2, 5, 10)
LEVELS = (40, 50, 60, 70, 80, 85, 90, 95)
SPEEDS = {"f10in2": (10, 2), "f10in5": (10, 5), "f15in5": (15, 5), "f15in10": (15, 10), "f20in10": (20, 10)}
SPEED_NAMES = list(SPEEDS)+["slow", "any"]
DEFS = [("rise", x, m) for x in RISE_X for m in RISE_M]+[("cross", L, s) for L in LEVELS for s in SPEED_NAMES]+[("sweep", 65, 15)]
DEF_NAME = [f"rise {a}c/{b}m" if k == "rise" else f"cross {a} {b}" if k == "cross" else "sweep 35>65/15m" for k, a, b in DEFS]
DEF_ID = {d: i for i, d in enumerate(DEFS)}
EXITS = ["hold", "ex5", "ex15", "ex30", "tp5", "tp10"]
SIDES = ["follow", "fade"]
MAKER = 1.75
SEED = 41


def fee(p, rate=7.0):
    """Fee in cents at price p cents."""
    return rate*p/100*(1-p/100)


class Tee:
    def __init__(self, path):
        self.file, self.out = open(path, "w"), sys.stdout

    def write(self, s):
        self.file.write(s)
        self.out.write(s)

    def flush(self):
        self.file.flush()
        self.out.flush()


# ---------------------------------------------------------------- build (nhl, mlb, nfl, college football)

def build():
    from research.basketball_reversion import pre_price
    from research.calibration_map import bar_requests, cache_file, listing, scheduled, ts as field_ts, volume
    from research.near_end import cents
    from research.tennis_reversion import on_grid
    OUT.mkdir(parents=True, exist_ok=True)
    cols, games, cover = {}, [], Counter()
    for sport, series in OTHER.items():
        for e in sorted(listing(None, series).values(), key=lambda x: (x[0], x[2])):
            settle, _, event, archived, ms = e
            cover[sport, "listed"] += 1
            if len(ms) != 2 or not all(m.get("_outcome") in (0.0, 1.0) for m in ms):
                cover[sport, "not two markets settled at 0 or 1"] += 1
                continue
            m = ms[0]
            sched = scheduled(event)
            got = {}
            for path, params in bar_requests(series, archived, m, sched):
                file = cache_file(path, params)
                if file.exists():
                    try:
                        for b in json.loads(file.read_text()).get("candlesticks") or []:
                            got[b["end_period_ts"]] = b
                    except ValueError:
                        continue
            bars = [got[k] for k in sorted(got) if settle-8*3600 < k <= settle]
            if not bars:
                cover[sport, "no bars on disk"] += 1
                continue
            expected = field_ts(m, "expected_expiration_time")
            start = sched or (expected-3*3600 if expected else None)
            if not start or not 1800 <= settle-start <= 7*3600:
                cover[sport, "no usable start"] += 1
                continue
            p0, kind = pre_price(bars, start, 120 if sched else 600)
            if p0 is None:
                cover[sport, "no pre-game quote on disk"] += 1
                continue
            tsb = np.array([b["end_period_ts"] for b in bars], float)
            get = lambda side, f: np.array([np.nan if (v := cents(b, side, f)) is None else v for b in bars], float)
            bid, ask, obid, oask = get("yes_bid", "close"), get("yes_ask", "close"), get("yes_bid", "open"), get("yes_ask", "open")
            low, high = get("price", "low"), get("price", "high")
            vol = np.array([volume(b) for b in bars], float)
            two = (bid > 0) & (bid <= ask) & (ask < 100)
            t0 = float(np.ceil(start/60)*60)
            inplay = two & (tsb >= t0)
            if inplay.sum() < 20:
                cover[sport, "fewer than 20 in-play quotes"] += 1
                continue
            n = int((tsb[inplay][-1]-t0)//60)+1
            for a in (bid, ask):
                a[~two] = np.nan
            otwo = (obid > 0) & (obid <= oask) & (oask < 100)
            obid[~otwo], oask[~otwo] = np.nan, np.nan
            g = {}
            g["bid"], obs = on_grid(tsb[two], bid[two], t0, n, True)
            g["ask"], _ = on_grid(tsb[two], ask[two], t0, n, True)
            ob, _ = on_grid(tsb, obid, t0, n, False)
            oa, _ = on_grid(tsb, oask, t0, n, False)
            prev_b, prev_a = np.r_[np.nan, g["bid"][:-1]], np.r_[np.nan, g["ask"][:-1]]
            g["obid"], g["oask"] = np.where(np.isnan(ob), prev_b, ob), np.where(np.isnan(oa), prev_a, oa)
            g["low"], _ = on_grid(tsb, low, t0, n, False)
            g["high"], _ = on_grid(tsb, high, t0, n, False)
            v, _ = on_grid(tsb, vol, t0, n, False)
            g["vol"], g["obs"] = np.nan_to_num(v), obs.astype(float)
            final = 100*m["_outcome"]
            for key, a in g.items():
                end = final if key in ("bid", "ask", "obid", "oask") else (0.0 if key in ("vol", "obs") else np.nan)
                cols.setdefault(key, []).append(np.r_[a, end].astype(np.float32))
            cols.setdefault("g", []).append(np.full(n+1, len(games), np.int32))
            cols.setdefault("term", []).append(np.r_[np.zeros(n, bool), True])
            games.append(dict(grp=sport, event=event, settle=settle, p0=p0, outcome=m["_outcome"], n=n, start_kind="ticker" if sched else "expected-3h",
                              p0_kind=kind))
            cover[sport, "built"] += 1
            cover[sport, f"p0: {kind}"] += 1
    pd.DataFrame(games).to_pickle(OUT/"other-games.pkl")
    np.savez_compressed(OUT/"other-grid.npz", **{k: np.concatenate(v) for k, v in cols.items()})
    text = {f"{a} | {b}": v for (a, b), v in sorted(cover.items())}
    (OUT/"other-coverage.json").write_text(json.dumps(text, indent=1))
    print(json.dumps(text, indent=1))


# ---------------------------------------------------------------- data

class Set:
    """All games on one grid. Side arrays have shape (2, points): 0 = the YES side, 1 = the other side."""

    def __init__(self, parts):
        games, off, arr = [], 0, {k: [] for k in ("bid", "ask", "obid", "oask", "low", "high", "vol", "obs", "term", "g", "per", "left")}
        for G, z, gkey in parts:
            n = len(z[gkey])
            for k in arr:
                if k == "g":
                    arr[k].append(z[gkey].astype(np.int64)+off)
                elif k in ("per", "left"):
                    arr[k].append(z["period" if k == "per" else k].astype(np.float32) if "left" in z else np.full(n, np.nan, np.float32))
                else:
                    arr[k].append(z[k])
            games.append(G)
            off += len(G)
        self.M = M = pd.concat(games, ignore_index=True)
        a = {k: np.concatenate(v) for k, v in arr.items()}
        self.finish(a["g"], a["term"], a["bid"], a["ask"], a["obid"], a["oask"], a["low"], a["high"])
        self.vol, self.obs, self.per, self.left = a["vol"], a["obs"] > 0, a["per"], a["left"]
        self.grp = np.array([GROUPS.index(x) for x in M.grp.values])
        self.p0 = M.p0.values.astype(float)
        self.win = M.outcome.values.astype(float)
        self.half = np.zeros(len(M), int)
        self.third = np.zeros(len(M))
        for k in range(len(GROUPS)):
            s = self.grp == k
            if s.any():
                self.half[s] = (M.settle.values[s] > np.median(M.settle.values[s])).astype(int)
                self.third[s] = np.median(M.n.values[s])/3
        self.day = (M.settle.values//86400).astype(int)

    def finish(self, g, term, bid, ask, obid, oask, low, high):
        f = lambda a, b: np.stack([a, 100-b]).astype(np.float32)
        self.g, self.term = g, term
        # a minute whose bar had a one-sided quote (only at the extremes) carries the last two-sided closing quote, as an empty minute
        # does; the OPENING quotes are left as they are, so no entry is ever priced on a carried quote of that kind
        start = np.where(np.r_[True, g[1:] != g[:-1]])[0][g]
        last = np.maximum.accumulate(np.where(np.isnan(bid) | np.isnan(ask), -1, np.arange(len(g))))
        ok = last >= start
        bid, ask = np.where(ok, bid[np.maximum(last, 0)], np.nan), np.where(ok, ask[np.maximum(last, 0)], np.nan)
        self.bid, self.ask, self.obid, self.oask = f(bid, ask), f(ask, bid), f(obid, oask), f(oask, obid)
        self.low, self.high = f(low, high), f(high, low)
        self.mid = (self.bid+self.ask)/2
        n = len(g)
        change = np.r_[True, g[1:] != g[:-1]]
        self.first = np.where(change)[0][g]
        self.end = (np.r_[np.where(change)[0][1:], n]-1)[g]
        self.i = np.arange(n)
        # running maximum of later prints inside the game: sufmax[s, x] = highest print in bars x..end-1
        self.sufmax = np.empty_like(self.high)
        for s in (0, 1):
            h = np.where(np.isnan(self.high[s]) | term, -1.0, self.high[s]).astype(np.float64)[::-1]
            offs = (g.max()-g[::-1])*1000.0
            self.sufmax[s] = (np.maximum.accumulate(h+offs)-offs)[::-1]


def load():
    parts = []
    M = pd.read_pickle(TENNIS/"matches.pkl")
    G = pd.DataFrame(dict(grp=M.tour.map(TOUR_GROUP).values, event=M.event.values, settle=M.settle.values, p0=M.p0.values, outcome=M.outcome.values,
                          n=M.n.values))
    parts.append((G, np.load(TENNIS/"grid.npz"), "m"))
    M = pd.read_pickle(BBALL/"games.pkl")
    parts.append((M[["league", "event", "settle", "p0", "outcome", "n"]].rename(columns=dict(league="grp")), np.load(BBALL/"grid.npz"), "g"))
    if (OUT/"other-grid.npz").exists():
        M = pd.read_pickle(OUT/"other-games.pkl")
        parts.append((M[["grp", "event", "settle", "p0", "outcome", "n"]], np.load(OUT/"other-grid.npz"), "g"))
    return Set(parts)


def replay(d, rng):
    """The same games redrawn as fair markets (see the docstring). Returns a Set-like object sharing d's game table."""
    r = object.__new__(Set)
    for k in ("M", "grp", "p0", "half", "third", "day", "per", "left"):
        setattr(r, k, getattr(d, k))
    g, term, first = d.g, d.term, d.first
    n = d.M.n.values.astype(float)[g]
    k = (d.i-first).astype(float)
    eps = rng.standard_normal(len(g))/np.sqrt(n)
    eps[d.i == first] = 0.0
    c = np.cumsum(eps)
    w = ndtri(np.clip(d.p0, 1, 99)/100)[g]+c-c[first]
    final = w > 0                                                   # at the terminal point k = n: W_n decides
    p = np.where(term, np.where(final, 100.0, 0.0), 100*ndtr(w/np.sqrt(np.maximum(1-k/n, 1e-9))))
    mid = np.clip(np.round(p), 2, 98)
    bid, ask = np.where(term, p, mid-1), np.where(term, p, mid+1)
    prev = lambda a: np.where(d.i == first, a, np.r_[a[0], a[:-1]])
    pm = prev(mid)
    high, low = np.where(term, np.nan, np.maximum(mid, pm)), np.where(term, np.nan, np.minimum(mid, pm))
    r.finish(g, term, bid, ask, prev(bid), prev(ask), low, high)
    r.vol, r.obs = np.zeros(len(g), np.float32), ~term
    win = np.zeros(len(d.M))
    win[g[term]] = final[term]
    r.win = win
    return r


# ---------------------------------------------------------------- surges

def prev_min(d, M):
    """Lowest closing midpoint of each side over the previous M minutes of the same game (inf where there is none)."""
    out = np.full(d.mid.shape, np.inf, np.float32)
    for k in range(1, M+1):
        j = np.maximum(d.i-k, 0)
        ok = (d.i-k >= d.first)
        v = np.where(ok & ~np.isnan(d.mid[:, j]), d.mid[:, j], np.inf)
        np.minimum(out, v, out=out)
    return out


def first_per_side(d, cond):
    """Grid indices and sides of the first True of each game and side."""
    I, S = [], []
    for s in (0, 1):
        hits = np.where(cond[s])[0]
        _, idx = np.unique(d.g[hits], return_index=True)
        I.append(hits[idx])
        S.append(np.full(len(idx), s))
    return np.concatenate(I), np.concatenate(S)


def find(d):
    """Every surge event: arrays (definition id, grid index, side)."""
    live = ~d.term
    mid = d.mid
    D, I, S = [], [], []

    def add(k, i, s):
        D.append(np.full(len(i), k))
        I.append(i)
        S.append(s)
    for M in RISE_M:
        back = np.maximum(d.i-M, 0)
        ch = np.where((d.i-M >= d.first) & live, mid-mid[:, back], np.nan)
        for X in RISE_X:
            hit = ch >= X
            run = np.zeros(hit.shape, bool)                              # a hit in the previous M minutes of the same game
            for k in range(1, M+1):
                run |= hit[:, np.maximum(d.i-k, 0)] & (d.i-k >= d.first)
            s, i = np.where(hit & ~run)
            add(DEF_ID["rise", X, M], i, s)
    mins = {M: prev_min(d, M) for M in (2, 5, 10, 15)}
    prev = mid[:, np.maximum(d.i-1, 0)]
    has_prev = (d.i-1 >= d.first) & live
    for L in LEVELS:
        up = has_prev & (mid >= L) & (prev < L)
        for name in SPEED_NAMES:
            if name == "any":
                cond = up
            elif name == "slow":
                cond = up & (d.i-10 >= d.first) & (mins[10] > L-10)
            else:
                dd, M = SPEEDS[name]
                cond = up & (mins[M] <= L-dd)
            i, s = first_per_side(d, cond)
            add(DEF_ID["cross", L, name], i, s)
    i, s = first_per_side(d, has_prev & (mid > 65) & (prev <= 65) & (mins[15] < 35))
    add(DEF_ID["sweep", 65, 15], i, s)
    E = dict(d=np.concatenate(D), i=np.concatenate(I), s=np.concatenate(S))
    E["rise5"] = (mid[E["s"], E["i"]]-np.where(np.isinf(mins[5][E["s"], E["i"]]), np.nan, mins[5][E["s"], E["i"]])).astype(np.float32)
    E["rise10"] = (mid[E["s"], E["i"]]-np.where(np.isinf(mins[10][E["s"], E["i"]]), np.nan, mins[10][E["s"], E["i"]])).astype(np.float32)
    return E


def price(d, E):
    """Add the game, class, entry and every trade outcome (follow and fade) to the events."""
    i, s = E["i"], E["s"]
    g = d.g[i]
    e = np.minimum(i+1, d.end[i])
    can = ~d.term[e]
    p0 = np.where(s == 0, d.p0[g], 100-d.p0[g])
    E.update(g=g, grp=d.grp[g], half=d.half[g], p0=p0, cls=np.digitize(p0, (20, 35, 50)), day=d.day[g], el=i-d.first[i],
             mid=d.mid[s, i], per=d.per[i], left=d.left[i], obs_e=d.obs[e], vol_e=d.vol[e])
    E["third"] = np.minimum((E["el"]/d.third[g]).astype(int), 2)
    for name, side in (("follow", s), ("fade", 1-s)):
        a = np.where(can, d.oask[side, e], np.nan).astype(np.float64)
        a[~((a >= 1) & (a <= 99))] = np.nan
        won = np.where(side == 0, d.win[g], 1-d.win[g])
        final = 100*won
        E[name+"_ask"], E[name+"_won"] = a, np.where(np.isnan(a), np.nan, won)
        E[name+"_emid"] = np.where(np.isnan(a), np.nan, (d.oask[side, e]+d.obid[side, e])/2)
        E[name+"_hold"] = final-a-fee(a)
        for N in (5, 15, 30):
            x = np.minimum(e+N, d.end[i])
            b = d.bid[side, x].astype(np.float64)
            E[f"{name}_ex{N}"] = np.where(d.term[x], final, b-fee(b))-a-fee(a)
        later = np.where(e+1 < d.end[i], d.sufmax[side, np.minimum(e+1, len(d.g)-1)], -1.0)
        for Y in (5, 10):
            sold = (later > a+Y) & (a+Y <= 99)
            E[f"{name}_tp{Y}"] = np.where(sold, Y-fee(a)-fee(a+Y, MAKER), E[name+"_hold"])
            E[f"{name}_tp{Y}_sold"] = np.where(np.isnan(a), np.nan, sold)
    return E


# ---------------------------------------------------------------- cells

def cmean(v, g):
    """n, clusters, mean and a 95% interval clustered by g."""
    v, g = np.asarray(v, float), np.asarray(g)
    ok = ~np.isnan(v)
    v, g = v[ok], g[ok]
    if len(v) < 5:
        return dict(n=len(v), k=len(set(g)), mean=np.nan, lo=np.nan, hi=np.nan, se=np.nan)
    mean = v.mean()
    _, inv = np.unique(g, return_inverse=True)
    sums = np.bincount(inv, v-mean)
    k = len(sums)
    se = np.sqrt((sums**2).sum()*k/max(k-1, 1))/len(v)
    return dict(n=len(v), k=k, mean=mean, lo=mean-1.96*se, hi=mean+1.96*se, se=se)


def cells(E, sides=SIDES):
    """Every cell (group x definition x class x half, with the pooled group, class and half) for each side and exit."""
    ng, nd, nc, nh = len(ALL_GROUPS), len(DEFS), len(ALL_CLASSES), 3
    NG = int(E["g"].max())+1
    cols, names = [], []
    for side in sides:
        a, ok = E[side+"_ask"], ~np.isnan(E[side+"_ask"])
        z = lambda v: np.where(ok, v, 0.0)
        cols += [ok.astype(float), z(a), z(E[side+"_won"]), z(a+fee(a))]
        names += [f"{side}|n", f"{side}|ask", f"{side}|won", f"{side}|need"]
        for x in EXITS:
            v = z(E[f"{side}_{x}"])
            cols += [v, z(v/a), (v > 0) & ok, np.where(v <= 0, v, 0.0), (v <= 0) & ok]
            names += [f"{side}|{x}|{k}" for k in ("sum", "ret", "wins", "losssum", "losses")]
    S = np.stack(cols, 1).astype(np.float64)
    ix = {n: k for k, n in enumerate(names)}
    out = []
    tennis = np.array([g in TOUR_GROUP.values() for g in GROUPS])
    for pg in (False, True):
        for pc in (False, True):
            for ph in (False, True):
                keep = np.ones(len(E["g"]), bool)
                kg, kc, kh = E["grp"], E["cls"], E["half"]
                if pg:
                    keep &= tennis[kg]
                    kg = np.full(len(kg), len(GROUPS))
                if pc:
                    keep &= kc < 3
                    kc = np.full(len(kc), 4)
                if ph:
                    kh = np.full(len(kh), 2)
                key = (((kg*nd+E["d"])*nc+kc)*nh+kh)[keep]
                u, inv = np.unique(key*NG+E["g"][keep], return_inverse=True)
                P = np.stack([np.bincount(inv, S[keep, c], len(u)) for c in range(S.shape[1])], 1)      # per cell and game
                uc, inv2 = np.unique(u//NG, return_inverse=True)
                T = np.stack([np.bincount(inv2, P[:, c], len(uc)) for c in range(P.shape[1])], 1)        # per cell
                base = dict(grp=np.array(ALL_GROUPS)[uc//(nd*nc*nh)], defn=np.array(DEF_NAME)[uc//(nc*nh) % nd], d=uc//(nc*nh) % nd,
                            cls=np.array(ALL_CLASSES)[uc//nh % nc], half=np.array(HALVES)[uc % nh])
                for side in sides:
                    n = T[:, ix[f"{side}|n"]]
                    nn = np.maximum(n, 1)
                    pn = P[:, ix[f"{side}|n"]]
                    k = np.bincount(inv2, pn > 0, len(uc))
                    for x in EXITS:
                        mean = T[:, ix[f"{side}|{x}|sum"]]/nn
                        resid = P[:, ix[f"{side}|{x}|sum"]]-pn*mean[inv2]
                        se = np.sqrt(np.bincount(inv2, resid**2, len(uc))*k/np.maximum(k-1, 1))/nn
                        losses = T[:, ix[f"{side}|{x}|losses"]]
                        # test interval: a cell with no (or almost no) losers has no visible variance, so add one unseen full loss to
                        # the variance and, for holds, never use less than the binomial variance at the price paid (edge_reality.py)
                        need = T[:, ix[f"{side}|need"]]/nn
                        se_t = np.sqrt(se**2+((need+mean)/nn)**2)
                        if x == "hold":
                            pn_ = np.clip(need/100, 0.01, 0.99)
                            se_t = np.maximum(se_t, 100*np.sqrt(pn_*(1-pn_)/nn))
                        out.append(pd.DataFrame(dict(base, side=side, exit=x, n=n, games=k, ask=T[:, ix[f"{side}|ask"]]/nn,
                                                     win=100*T[:, ix[f"{side}|won"]]/nn, need=T[:, ix[f"{side}|need"]]/nn, net=mean, lo=mean-1.96*se,
                                                     hi=mean+1.96*se, se=se, lo_t=mean-1.96*se_t, se_t=se_t, pct=100*T[:, ix[f"{side}|{x}|ret"]]/nn,
                                                     pct_w=100*T[:, ix[f"{side}|{x}|sum"]]/np.maximum(T[:, ix[f"{side}|ask"]], 1e-9),
                                                     winshare=100*T[:, ix[f"{side}|{x}|wins"]]/nn,
                                                     loser=T[:, ix[f"{side}|{x}|losssum"]]/np.maximum(losses, 1))))
    C = pd.concat(out, ignore_index=True)
    return C[C.n > 0].reset_index(drop=True)


def neighbours(k):
    kind, a, b = DEFS[k]
    if kind == "rise":
        xs, ms = RISE_X.index(a), RISE_M.index(b)
        out = [("rise", RISE_X[j], b) for j in (xs-1, xs+1) if 0 <= j < len(RISE_X)]+[("rise", a, RISE_M[j]) for j in (ms-1, ms+1) if 0 <= j < len(RISE_M)]
    elif kind == "cross":
        ls = LEVELS.index(a)
        out = [("cross", LEVELS[j], b) for j in (ls-1, ls+1) if 0 <= j < len(LEVELS)]
    else:
        out = []
    return [DEF_ID[x] for x in out]


def passes(C, side="follow"):
    """Cells meeting the standard. Returns (tested, passing without the neighbour rule, passing with it, the rows)."""
    C = C[C.side == side]
    key = ["grp", "d", "cls", "exit"]
    h1, h2, both = (C[C.half == h].set_index(key) for h in HALVES)
    j = h1.join(h2, rsuffix="_2", how="inner")
    j = j[(j.n >= 30) & (j.n_2 >= 30)]
    first = j[(j.lo_t > 0) & (j.net_2 > 0)]
    pooled = both.net.to_dict()
    pooled_n = both.n.to_dict()
    rows = []
    for (grp, k, cls, ex), r in first.iterrows():
        nb = [(grp, q, cls, ex) for q in neighbours(k)]
        nb = [q for q in nb if pooled_n.get(q, 0) >= 20]
        agree = DEFS[k][0] == "sweep" or (len(nb) >= 1 and all(pooled[q] > 0 for q in nb))
        b = both.loc[(grp, k, cls, ex)]
        rows.append(dict(grp=grp, defn=DEF_NAME[k], cls=cls, exit=ex, n1=int(r.n), net1=r.net, lo1=r.lo_t, pct1=r.pct, n2=int(r.n_2), net2=r.net_2,
                         lo2=r.lo_t_2, pct2=r.pct_2, net=b.net, se=b.se_t, pct=b.pct, ask=b.ask, win=b.win, need=b.need, neighbours=agree))
    return len(j), len(first), sum(r["neighbours"] for r in rows), rows


# ---------------------------------------------------------------- report

def pick(C, **kw):
    q = C
    for k, v in kw.items():
        q = q[q[k] == v]
    return q


def line(r, pct=True):
    return f"n={int(r.n):5} ask {r.ask:4.1f} win {r.win:4.1f}% need {r.need:4.1f}% net {r.net:+6.2f}c[{r.lo:+6.2f},{r.hi:+6.2f}]" + (f" {r.pct:+6.1f}%" if pct else "")


def q(v, g):
    r = cmean(v, g)
    return f"n={r['n']:5} {r['mean']:+6.2f}[{r['lo']:+6.2f},{r['hi']:+6.2f}]" if r["n"] >= 5 else f"n={r['n']:5}    n/a               "


def diff(v1, g1, v2, g2):
    a, b = cmean(v1, g1), cmean(v2, g2)
    if a["n"] < 5 or b["n"] < 5:
        return "n/a"
    dd, se = a["mean"]-b["mean"], np.hypot(a["se"], b["se"])
    return f"{dd:+6.2f}[{dd-1.96*se:+6.2f},{dd+1.96*se:+6.2f}]"


def groups_of(E, name):
    if name == POOLED_GROUP:
        return np.isin(E["grp"], [GROUPS.index(g) for g in set(TOUR_GROUP.values())])
    return E["grp"] == GROUPS.index(name)


def report(d, E, C, null):
    T = lambda s: print("\n"+"="*8+" "+s)
    isd = lambda k, a, b: E["d"] == DEF_ID[k, a, b]
    dog = E["cls"] < 3
    M = d.M
    T("DATA")
    for k, gname in enumerate(GROUPS):
        s = d.grp == k
        if s.any():
            days = len(set(d.day[s]))
            print(f"  {gname:10} games {s.sum():6}  days {days:4}  median in-play minutes {np.median(M.n.values[s]):4.0f}  pre-game underdog (YES or NO side under 50) wins "
                  f"{100*np.mean(np.where(d.p0[s] < 50, d.win[s], 1-d.win[s])):.1f}% at mean price {np.mean(np.minimum(d.p0[s], 100-d.p0[s])):.1f}")
    ok = ~np.isnan(E["follow_ask"])
    print(f"  events {len(E['d'])}, of which {ok.sum()} could be bought; entry spread (ask - bid at the next open) by group, rise 15c/5m:")
    for gname in GROUPS:
        s = groups_of(E, gname) & isd("rise", 15, 5) & ok
        if s.sum() > 20:
            sp = 2*(E["follow_ask"][s]-E["follow_emid"][s])
            print(f"     {gname:10} mean {np.nanmean(sp):4.1f}c median {np.nanmedian(sp):3.0f}c | next bar is a real bar {100*E['obs_e'][s].mean():3.0f}% | "
                  f"median contracts in the entry minute {np.median(E['vol_e'][s]):7.0f} | events per day {s.sum()/len(set(d.day[d.grp == GROUPS.index(gname)])):5.1f}")

    T("2. FOLLOW THE SURGE, hold to settlement. Underdogs pooled (pre-game price under 50) and the favourite control; both halves. % = mean return on the price paid")
    for gname in ALL_GROUPS:
        print(f"  --- {gname}")
        for k, name in enumerate(DEF_NAME):
            if DEFS[k][0] == "cross" and (DEFS[k][1] >= 85 or DEFS[k][2] not in ("f10in5", "f15in10", "slow", "any")):
                continue
            for cls in ("dog<50", "fav50+"):
                r = pick(C, grp=gname, d=k, cls=cls, side="follow", exit="hold")
                if r.empty or pick(r, half="both").n.iloc[0] < 30:
                    continue
                b, h1, h2 = (pick(r, half=h) for h in HALVES[2:]+HALVES[:2])
                f = lambda x: f"{x.net.iloc[0]:+6.2f}c {x.pct.iloc[0]:+6.1f}% (n {int(x.n.iloc[0])})" if len(x) else "n/a"
                print(f"   {name:17} {cls:7} {line(b.iloc[0])} | winners {b.winshare.iloc[0]:3.0f}% avg loser {b.loser.iloc[0]:+6.1f}c | h1 {f(h1)} h2 {f(h2)}")
    T("2b. By underdog class and exit (both halves pooled): net cents and % return for hold / sell after 5, 15, 30 min / take-profit +5c, +10c")
    main_defs = [("rise", 10, 5), ("rise", 15, 5), ("rise", 20, 5), ("rise", 30, 10), ("cross", 50, "f10in5"), ("cross", 60, "f10in5"), ("cross", 70, "f10in5"),
                 ("cross", 50, "f15in10"), ("sweep", 65, 15)]
    for gname in ALL_GROUPS:
        print(f"  --- {gname}")
        for key in main_defs:
            for cls in ALL_CLASSES:
                r = pick(C, grp=gname, d=DEF_ID[key], cls=cls, side="follow", half="both").set_index("exit")
                if r.empty or r.n.iloc[0] < 30:
                    continue
                h = r.loc["hold"]
                print(f"   {DEF_NAME[DEF_ID[key]]:17} {cls:9} n={int(h.n):5} ask {h.ask:4.1f} win {h.win:4.1f}% need {h.need:4.1f}% | "
                      + " ".join(f"{x}:{r.net[x]:+5.2f}c({r.pct[x]:+5.1f}%)" for x in EXITS) + f" | winners hold {h.winshare:2.0f}% tp5 {r.winshare['tp5']:2.0f}% tp10 {r.winshare['tp10']:2.0f}%"
                      f" loser hold {h.loser:+5.1f} tp5 {r.loser['tp5']:+5.1f}")

    T("3. FAST versus SLOW first crossings of the same level: does speed predict the win beyond the price?")
    print("   excess = 100 x win - midpoint at the entry bar's open (no costs: pure calibration); paid = 100 x win - ask - fee. Underdogs (P0 under 50) only.")
    cross = {(L, sp): isd("cross", L, sp) for L in LEVELS for sp in SPEED_NAMES}
    exc = 100*E["follow_won"]-E["follow_emid"]
    for gname in ALL_GROUPS:
        gm = groups_of(E, gname) & dog
        printed = False
        for L in LEVELS[:5]:
            sl = gm & cross[L, "slow"]
            for sp in ("f10in5", "f15in10", "f20in10"):
                fa = gm & cross[L, sp]
                if (fa & ok).sum() < 30 or (sl & ok).sum() < 30:
                    continue
                if not printed:
                    print(f"  --- {gname}")
                    printed = True
                print(f"   L={L} {sp:8} fast: ask {np.nanmean(E['follow_ask'][fa]):4.1f} excess {q(exc[fa], E['g'][fa])} paid {q(E['follow_hold'][fa], E['g'][fa])} | "
                      f"slow: ask {np.nanmean(E['follow_ask'][sl]):4.1f} excess {q(exc[sl], E['g'][sl])} | fast minus slow {diff(exc[fa], E['g'][fa], exc[sl], E['g'][sl])}")
    print("  Pooled over levels 40-80 (first crossing, any speed): slope of excess (cents) on the rise over the previous 10 minutes (per 10c), clustered by game")
    anyc = np.zeros(len(E["d"]), bool)
    for L in LEVELS[:5]:
        anyc |= cross[L, "any"]
    summary = {}
    for gname in ALL_GROUPS:
        for cname, cm in (("dog<50", dog), ("fav50+", ~dog)):
            for h in (0, 1, None):
                s = groups_of(E, gname) & cm & anyc & ~np.isnan(exc) & ~np.isnan(E["rise10"]) & ((E["half"] == h) if h is not None else True)
                if s.sum() < 200:
                    continue
                x, y, g = E["rise10"][s].astype(float)/10, exc[s], E["g"][s]
                xc, yc = x-x.mean(), y-y.mean()
                slope = (xc*yc).sum()/(xc*xc).sum()
                _, inv = np.unique(g, return_inverse=True)
                sc = np.bincount(inv, xc*(yc-slope*xc))
                se = np.sqrt((sc**2).sum()*len(sc)/(len(sc)-1))/(xc*xc).sum()
                hn = "both" if h is None else HALVES[h]
                summary[f"{gname}|{cname}|{hn}"] = [float(slope), float(slope-1.96*se), float(slope+1.96*se), int(s.sum())]
                if h is None:
                    a, b = summary.get(f"{gname}|{cname}|half1"), summary.get(f"{gname}|{cname}|half2")
                    print(f"   {gname:10} {cname:7} n={s.sum():6} slope {slope:+5.2f}c per 10c [{slope-1.96*se:+5.2f},{slope+1.96*se:+5.2f}]"
                          + (f" | h1 {a[0]:+5.2f} h2 {b[0]:+5.2f}" if a and b else ""))
    print("  By phase (thirds of the group's median in-play length, by minutes elapsed): fast crossings (f10in5) of 40-80 by an underdog; excess and paid (hold)")
    fastc = np.zeros(len(E["d"]), bool)
    slowc = np.zeros(len(E["d"]), bool)
    for L in LEVELS[:5]:
        fastc |= cross[L, "f10in5"]
        slowc |= cross[L, "slow"]
    for gname in ALL_GROUPS:
        for t, tn in enumerate(("early", "middle", "late")):
            fa, sl = groups_of(E, gname) & dog & fastc & (E["third"] == t), groups_of(E, gname) & dog & slowc & (E["third"] == t)
            if (fa & ok).sum() < 30:
                continue
            print(f"   {gname:10} {tn:6} fast: ask {np.nanmean(E['follow_ask'][fa]):4.1f} excess {q(exc[fa], E['g'][fa])} paid {q(E['follow_hold'][fa], E['g'][fa])} "
                  f"({100*np.nanmean(E['follow_hold'][fa]/E['follow_ask'][fa]):+5.1f}%) | slow excess {q(exc[sl], E['g'][sl])} | fast-slow {diff(exc[fa], E['g'][fa], exc[sl], E['g'][sl])}")
    print("  By the play-by-play clock (linked games): underdog rise 15c/5m and fast crossings (f10in5, 40-80); excess and paid (hold)")
    per, left = E["per"], E["left"]
    clock = {"nba": (("Q1", per == 1), ("Q2", per == 2), ("Q3", per == 3), ("Q4 before last 5", (per >= 4) & (left > 300)), ("last 5 min + OT", (per >= 4) & (left <= 300))),
             "cbb-men": (("1st half", per == 1), ("2nd half before last 5", (per >= 2) & (left > 300)), ("last 5 min + OT", (per >= 2) & (left <= 300)))}
    for gname, phases in clock.items():
        for label, sel in (("rise 15c/5m", isd("rise", 15, 5)), ("fast cross 40-80", fastc)):
            for pn, pm in phases:
                s = groups_of(E, gname) & dog & sel & pm
                if (s & ok).sum() < 20:
                    continue
                print(f"   {gname:8} {label:16} {pn:22} ask {np.nanmean(E['follow_ask'][s]):4.1f} win {100*np.nanmean(E['follow_won'][s]):4.1f}% excess {q(exc[s], E['g'][s])} "
                      f"paid {q(E['follow_hold'][s], E['g'][s])} ({100*np.nanmean(E['follow_hold'][s]/E['follow_ask'][s]):+5.1f}%)")

    T("4. LATE CLOSE-OUTS: a pre-game underdog first crosses 85 / 90 / 95; buy at the next open, hold. Fast = 15c+ lower within the last 10 minutes")
    for gname in ALL_GROUPS:
        print(f"  --- {gname}")
        for L in (85, 90, 95):
            for cls in ("dog<50", "fav50+"):
                for sp in ("f15in10", "f10in5", "slow", "any"):
                    r = pick(C, grp=gname, d=DEF_ID["cross", L, sp], cls=cls, side="follow", exit="hold")
                    b = pick(r, half="both")
                    if b.empty or b.n.iloc[0] < 30:
                        continue
                    f = lambda x: f"{x.net.iloc[0]:+5.2f}c {x.pct.iloc[0]:+5.1f}% (n {int(x.n.iloc[0])}, win {x.win.iloc[0]:.1f}%)" if len(x) else "n/a"
                    print(f"   cross {L} {sp:8} {cls:7} {line(b.iloc[0])} | avg loser {b.loser.iloc[0]:+6.1f}c | h1 {f(pick(r, half='half1'))} h2 {f(pick(r, half='half2'))}")

    T("5. FAIR-MARKET REPLAY of the same games (price = probability, 2c spread): what the same rules show with no edge. Underdogs pooled, hold")
    N = null["cells"]
    for gname in ALL_GROUPS:
        print(f"  --- {gname}")
        for key in main_defs+[("cross", 85, "f15in10"), ("cross", 90, "any")]:
            r = pick(C, grp=gname, d=DEF_ID[key], cls="dog<50", side="follow", half="both").set_index("exit")
            f = pick(N, grp=gname, d=DEF_ID[key], cls="dog<50", side="follow", half="both").set_index("exit")
            if r.empty or f.empty or r.n.iloc[0] < 30:
                continue
            h, fh = r.loc["hold"], f.loc["hold"]
            print(f"   {DEF_NAME[DEF_ID[key]]:17} REAL n={int(h.n):5} ask {h.ask:4.1f} win {h.win:4.1f}% (win-ask {h.win-h.ask:+5.2f}) hold {h.net:+5.2f}c winners {h.winshare:2.0f}% "
                  f"tp5 {r.net['tp5']:+5.2f}c winners {r.winshare['tp5']:2.0f}% | FAIR n/rep={fh.n:7.0f} ask {fh.ask:4.1f} win {fh.win:4.1f}% (win-ask {fh.win-fh.ask:+5.2f}) "
                  f"hold {fh.net:+5.2f}c winners {fh.winshare:2.0f}% tp5 {f.net['tp5']:+5.2f}c winners {f.winshare['tp5']:2.0f}%")

    T("6. THE MIRROR: fade the surge (buy the side that just fell at the next open ask), hold. By the FALLER's pre-game price")
    fp0 = 100-E["p0"]
    bands = (("90+", 90, 101), ("80-90", 80, 90), ("65-80", 65, 80), ("50-65", 50, 65), ("under 50", 0, 50))
    okf = ~np.isnan(E["fade_ask"])
    for gname in ALL_GROUPS:
        print(f"  --- {gname}")
        for key in (("rise", 10, 5), ("rise", 15, 5), ("rise", 20, 5), ("cross", 50, "f10in5"), ("sweep", 65, 15)):
            base = groups_of(E, gname) & isd(*key)
            if (base & okf).sum() < 30:
                continue
            r = cmean(E["fade_hold"][base], E["g"][base])
            parts = []
            for bn, lo, hi in bands:
                s = base & (fp0 >= lo) & (fp0 < hi)
                if (s & okf).sum() >= 15:
                    c = cmean(E["fade_hold"][s], E["g"][s])
                    parts.append(f"{bn}: n {c['n']} ask {np.nanmean(E['fade_ask'][s]):4.1f} {c['mean']:+5.2f}c[{c['lo']:+5.1f},{c['hi']:+5.1f}] {100*np.nanmean(E['fade_hold'][s]/E['fade_ask'][s]):+5.1f}%")
            ex = base & (fp0 < 90)
            c2 = cmean(E["fade_hold"][ex], E["g"][ex])
            print(f"   {DEF_NAME[DEF_ID[key]]:17} all fallers n {r['n']} {r['mean']:+5.2f}c[{r['lo']:+5.2f},{r['hi']:+5.2f}] {100*np.nanmean(E['fade_hold'][base]/E['fade_ask'][base]):+5.1f}%"
                  f" | without 90+ favourites n {c2['n']} {c2['mean']:+5.2f}c[{c2['lo']:+5.2f},{c2['hi']:+5.2f}] {100*np.nanmean(E['fade_hold'][ex]/E['fade_ask'][ex]):+5.1f}%")
            print("        "+" | ".join(parts))
        for hh in (0, 1):
            s = groups_of(E, gname) & isd("rise", 15, 5) & (fp0 >= 90) & (E["half"] == hh)
            s2 = groups_of(E, gname) & isd("rise", 15, 5) & (fp0 < 90) & (E["half"] == hh)
            print(f"        rise 15c/5m {HALVES[hh]}: faller 90+ {q(E['fade_hold'][s], E['g'][s])} | faller under 90 {q(E['fade_hold'][s2], E['g'][s2])}")

    T("7. MULTIPLICITY")
    out = {}
    for side in SIDES:
        tested, first, full, rows = passes(C, side)
        nt = [x[side] for x in null["passes"]]
        out[side] = dict(tested=tested, pass_without_neighbours=first, pass_full=full, naive_chance=tested*0.025*0.5,
                         fair_replays=[list(map(int, x[:3])) for x in nt], rows=rows)
        print(f"  {side.upper()}: {tested} cells with 30+ trades in each half (group x definition x class x exit; pooled groups and classes included, so cells overlap)")
        print(f"     first-half interval above zero AND second half positive: {first}; also neighbours agree: {full}")
        print(f"     expected if every cell had zero true edge BEFORE costs are even counted (2.5% x 50%): {tested*0.0125:.0f}; "
              f"fair-market replays (costs included), {len(nt)} runs: mean {np.mean([x[1] for x in nt]):.1f} / {np.mean([x[2] for x in nt]):.1f} (max {max(x[1] for x in nt)} / {max(x[2] for x in nt)})")
        H = C[(C.side == side) & (C.half == "both") & (C.n >= 60)]
        for ex in EXITS:
            h = H[H.exit == ex]
            print(f"     {ex:5}: {len(h):5} pooled cells with 60+ trades; mean net {h.net.mean():+5.2f}c, median {h.net.median():+5.2f}c, share positive {100*(h.net > 0).mean():4.1f}%, "
                  f"interval above zero {100*(h.lo > 0).mean():4.1f}%, interval below zero {100*(h.hi < 0).mean():4.1f}%")
        # empirical-Bayes shrinkage of the survivors towards the mean of comparable cells (same side and exit)
        for r in sorted(rows, key=lambda r: -r["net"]):
            h = H[H.exit == r["exit"]]
            mu, tau2 = h.net.mean(), max(h.net.var()-np.mean(h.se_t**2), 0.0)
            w = tau2/(tau2+r["se"]**2) if tau2+r["se"]**2 > 0 else 0.0
            r["shrunk"], r["weight"] = mu+w*(r["net"]-mu), w
            print(f"     {'PASS' if r['neighbours'] else 'no-nb'} {r['grp']:10} {r['defn']:17} {r['cls']:9} {r['exit']:5} ask {r['ask']:4.1f} win {r['win']:4.1f}% need {r['need']:4.1f}% | "
                  f"h1 {r['net1']:+5.2f}c (lo {r['lo1']:+5.2f}, n {r['n1']}, {r['pct1']:+5.1f}%) h2 {r['net2']:+5.2f}c (lo {r['lo2']:+5.2f}, n {r['n2']}, {r['pct2']:+5.1f}%) | "
                  f"pooled {r['net']:+5.2f}c {r['pct']:+5.1f}% -> shrunk {r['shrunk']:+5.2f}c (weight {w:.2f}; cell mean {mu:+.2f}, true spread {np.sqrt(tau2):.2f}c)")
    out["speed_slopes"] = summary
    return out


def run(reps):
    OUT.mkdir(parents=True, exist_ok=True)
    sys.stdout = Tee(OUT/"report.txt")
    d = load()
    E = price(d, find(d))
    C = cells(E)
    C.to_csv(OUT/"cells.csv", index=False, float_format="%.4f")
    rng = np.random.default_rng(SEED)
    acc, npass = None, []
    for rep in range(reps):
        r = replay(d, rng)
        Cn = cells(price(r, find(r)))
        key = ["grp", "d", "cls", "half", "side", "exit"]
        num = ["n", "ask", "win", "need", "net", "pct", "winshare", "loser"]
        part = Cn[Cn.side == "follow"].set_index(key)[num]
        w = part.mul(part.n, axis=0).assign(n=part.n, reps=1.0)
        acc = w if acc is None else acc.add(w, fill_value=0)
        npass.append({s: passes(Cn, s)[:3] for s in SIDES})
        del r
    N = acc[["ask", "win", "need", "net", "pct", "winshare", "loser"]].div(acc.n, axis=0).assign(n=acc.n/reps).reset_index()
    N.to_csv(OUT/"fair-cells.csv", index=False, float_format="%.4f")
    summary = report(d, E, C, dict(cells=N, passes=npass))
    (OUT/"null.json").write_text(json.dumps(dict(reps=reps, passes=[{k: list(map(int, v)) for k, v in x.items()} for x in npass]), indent=1))
    (OUT/"summary.json").write_text(json.dumps(summary, indent=1, default=float))
    keep = ["d", "i", "s", "g", "grp", "half", "p0", "cls", "day", "el", "third", "mid", "per", "left", "rise5", "rise10", "obs_e", "vol_e"]
    keep += [k for k in E if k.startswith(("follow_", "fade_"))]
    pd.DataFrame({k: E[k] for k in keep}).to_pickle(OUT/"events.pkl")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("step", choices=("build", "run"))
    parser.add_argument("--reps", type=int, default=10, help="fair-market replays")
    args = parser.parse_args()
    sys.path.insert(0, str(ROOT))
    build() if args.step == "build" else run(args.reps)
