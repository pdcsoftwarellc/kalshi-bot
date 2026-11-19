"""Do in-play basketball winner prices on Kalshi overreact to scoring runs? Martingale tests, tradeable fades, microstructure.

In tennis the in-play price is a fair martingale (research/tennis_reversion.py,
research/dip_scalp.py): after a 10c fall the average rebound is 0.0-0.2c against
a 3-5c round trip. College basketball books are thinner and move fast on runs.
This script repeats the tennis method on men's college basketball
(KXNCAAMBGAME), women's college basketball (KXNCAAWBGAME) and the NBA
(KXNBAGAME), and adds the structure specific to basketball.

Data (all already on disk unless `fetch` is run)
* One-minute bars of the alphabetically-first market of each game, from every
  per-market read and batch read cached under data/raw/cost-research-v1 by
  earlier scripts (the batch reads are found by scanning the cache once).
* ESPN public play-by-play cached under data/raw/game-state/{ncaamb,nba}
  (research/college_basketball.py, research/game_state.py): wall-clock time,
  period, clock and score of every play. Linked to Kalshi events by those
  scripts' own rules; the archive's winner must agree with the settlement.
  The archive carries a score LOOKAHEAD flaw (a row can show a score that has
  not happened yet), so every use of the SCORE here reads the score as of 120
  seconds before the bar's close. The clock (period, time left) is used at the
  bar's close only to label the phase of the game.
* The public trade record around a sample of events (`fetch`): one request at
  a time, 1.6 seconds apart, 30 seconds (then longer) after HTTP 429.

Definitions
* Start of play: the first play-by-play row where the game is linked;
  otherwise the scheduled start (this season: Kalshi's expected expiration
  minus three hours; NBA: the ticker or the archive's schedule); otherwise
  (last season's college games) the close minus 130 minutes, known only to
  about 20 minutes. `start_kind` records which.
* P0 = median midpoint of the last 10-15 one-minute quotes ending two minutes
  before the start (25 minutes for an estimated start), each at most 4c wide,
  all within 3c; failing that the first 15 bars of the six-hour window
  (favourite_contexts' "3h" price); failing that the median of the last
  two-sided quotes before the start ("loose"). Favourite = the side with P0
  above 50. All prices are the favourite's, in cents.
* The in-play path runs on a regular one-minute grid from the start to the
  last bar with a two-sided quote (0 < bid <= ask < 100); minutes with no bar
  carry the last quote forward (flagged). One terminal point holds the
  settlement value, so a change that runs past the end of the path ends at
  settlement: under "price = fair probability" every change, including that
  one, has mean zero given the past. Nothing is conditioned on the game still
  being quoted later (the survivor trap; the survivors-only variance ratio is
  printed once as a warning).
* Slopes, variance ratios and events are those of tennis_reversion.py. An
  event is the first bar where the midpoint is X cents or more below (above)
  its level M minutes earlier (M = 3, 5), then an M-bar pause.
* Tradeable versions. Taker: buy the team that fell at the NEXT bar's opening
  ask, sell at the closing bid N minutes later (fee 0.07*p*(1-p) per side, no
  fee at settlement), or rest an offer Y cents above the entry for 10 bars
  (sold only by a print strictly above it; else the bid), or hold to
  settlement. Maker: a bid resting at the event bar's closing bid for five
  bars, filled only by a print strictly below it. Maker fee 0.0175*p*(1-p)
  on men's college basketball and the NBA (unconfirmed), zero on women's.
  "Follow" buys the other team instead.
* Halves: each league's games ordered by settlement and split at the median.

    .venv/bin/python research/basketball_reversion.py build
    .venv/bin/python research/basketball_reversion.py martingale events special
    .venv/bin/python research/basketball_reversion.py fetch --minutes 30
    .venv/bin/python research/basketball_reversion.py trades

Outputs under data/processed/basketball-reversion/ (martingale.txt, events.txt,
special.txt, trades.txt are the printed reports of the 18 November 2026 run).
Public market data only; no orders, no authentication.

Result of that run (2,965 men's, 1,447 women's, 1,393 NBA games): no reversion.
Slopes from the next bar are -0.01 to +0.03 at every horizon, variance ratios
0.98-1.07 (women's last season 0.80-0.88 on raw midpoints, which is quote noise
in books 10c+ wide: it vanishes from the next bar and on tight quotes). After a
10c fall the faller's rebound from the next bar is -0.2 to +0.1c; a taker fade
loses 4-5c (12-16%) in men's and the NBA and 14c (34%) in women's; none of 6,820
short-horizon cells with 200+ trades is positive. The only positive holds are
the known 80c dip bid on 90c+ favourites.
"""
import argparse
from bisect import bisect_right
from collections import Counter, defaultdict
from datetime import date, timedelta
import json
import os
from pathlib import Path
import re
import sys
import time

import httpx
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import config
from research.api_cost import COLLECTION, ROOT, timestamp
from research.calibration_map import cache_file, listing, quote, scheduled, ts as field_ts, volume
from research.near_end import WINDOW, cents
from research.tennis_reversion import cluster_mean, cluster_slope, on_grid

OUT = ROOT / "data/processed/basketball-reversion"
RAW = ROOT / "data/raw/game-state"
LEAGUES = {"cbb-men": "KXNCAAMBGAME", "cbb-women": "KXNCAAWBGAME", "nba": "KXNBAGAME"}
MAKER_RATE = {"cbb-men": 1.75, "cbb-women": 0.0, "nba": 1.75}
BANDS = (("50-60", 50, 60), ("60-70", 60, 70), ("70-80", 70, 80), ("80-90", 80, 90), ("90+", 90, 101))
BAND_NAMES = [b[0] for b in BANDS]
HORIZONS = (1, 2, 5, 10)
MOVES = (5, 10, 15, 20)
WITHIN = (3, 5)
AFTER = (1, 3, 5, 10)
LEVELS = ((5, 30), (30, 50), (50, 70), (70, 85), (85, 95.01))
MONTHS = {m: i+1 for i, m in enumerate("JAN FEB MAR APR MAY JUN JUL AUG SEP OCT NOV DEC".split())}
SCORE_DELAY = 120
RNG = np.random.default_rng(23)
PACE = 1.6


def fee(p, rate=7.0):
    """Fee in cents at price p cents."""
    return rate*p/100*(1-p/100)


# ---------------------------------------------------------------- build

def batch_index():
    """{ticker: [cache file names]} for every cached batch read (/markets/candlesticks) holding a basketball market."""
    path = OUT/"batch-index.json"
    if path.exists():
        return json.loads(path.read_text())
    index, pat = defaultdict(list), re.compile(rb'"market_ticker": ?"(KX(?:NCAAMB|NCAAWB|NBA)GAME-[^"]+)"')
    for entry in os.scandir(COLLECTION):
        with open(entry.path, "rb") as f:
            head = f.read(40)
            if not head.startswith(b'{"markets": [{"') or b"candlesticks" not in head:
                continue
            for hit in set(pat.findall(head+f.read())):
                index[hit.decode()].append(entry.name)
    OUT.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(index))
    return index


class Bars:
    def __init__(self):
        self.index, self.loaded = batch_index(), {}

    def batch(self, name):
        if name not in self.loaded:
            if len(self.loaded) > 6:
                self.loaded.clear()
            data = json.loads((COLLECTION/name).read_text())
            self.loaded[name] = {x["market_ticker"]: x.get("candlesticks") or [] for x in data.get("markets", [])}
        return self.loaded[name]

    def get(self, series, e):
        """Every cached one-minute bar of the game's canonical market in the eight hours before settlement, by bar end."""
        settle, _, event, archived, ms = e
        m = ms[0]
        t, sched, close = m["ticker"], scheduled(event), field_ts(m, "close_time")
        expected = field_ts(m, "expected_expiration_time")
        starts = {settle-WINDOW}
        for s in (sched, expected-3*3600 if expected else None):
            if s and s-3600 < settle-WINDOW:
                starts.add(max(s-3600, settle-48*3600))
        reqs = [dict(start_ts=s, end_ts=settle, period_interval=1) for s in starts]
        if close:
            reqs.append(dict(start_ts=close+1860-WINDOW, end_ts=close+3600, period_interval=1))      # wcbb_backfill's late settlements
        got = {}
        for path in (f"/historical/markets/{t}/candlesticks", f"/series/{series}/markets/{t}/candlesticks"):
            for params in reqs:
                file = cache_file(path, params)
                if file.exists():
                    try:
                        for b in json.loads(file.read_text()).get("candlesticks") or []:
                            got[b["end_period_ts"]] = b
                    except ValueError:
                        continue
        for name in self.index.get(t, []):
            for b in self.batch(name).get(t, []):
                got.setdefault(b["end_period_ts"], b)
        return [got[k] for k in sorted(got) if settle-8*3600 < k <= settle]


def events_of(league):
    series = LEAGUES[league]
    if league == "nba":
        return {k: v for k, v in listing(None, series, archive=True).items() if len(v[4]) >= 2}
    from research import favourite_contexts as fc
    return fc.events_of(series, archive=True)


def slates_of(sport):
    out = {}
    for path in (RAW/sport).glob("day-*.json"):
        try:
            out[path.stem[4:]] = json.loads(path.read_text())
        except ValueError:
            continue
    return out


def play_log(sport, game):
    path = RAW/sport/f"game-{game}.json"
    if not path.exists():
        return None
    try:
        log = json.loads(path.read_text())
    except ValueError:
        return None
    return log if len(log) >= 50 else None


def day_of(event):
    stamp = event.split("-")[1]
    return date(2000+int(stamp[:2]), MONTHS[stamp[2:5]], int(stamp[5:7]))


def link_plays(league, e, slates, linker):
    """(log, yes_home, scheduled start) from the play-by-play archive, or (None, None, schedule or None)."""
    settle, _, event, _, ms = e
    m = ms[0]
    yes = m["ticker"].rsplit("-", 1)[1]
    try:
        day = day_of(event)
    except (KeyError, ValueError, IndexError):
        return None, None, None
    if league == "nba":
        teams = {x["ticker"].rsplit("-", 1)[1] for x in ms}
        match = [g for g in slates.get(f"{day:%Y%m%d}", []) if set(g["teams"].values()) == teams and g["status"] == "STATUS_FINAL"
                 and g["start"] < settle]
        if not match:
            return None, None, None
        g = max(match, key=lambda x: x["start"])
        yes_home = g["teams"]["home"] == yes
    else:
        names = {x["ticker"].rsplit("-", 1)[1]: x.get("yes_sub_title") or "" for x in ms}
        if len(names) != 2:
            return None, None, None
        by_day = {date(int(k[:4]), int(k[4:6]), int(k[6:])): v for k, v in slates.items()} if not linker.get("days") else linker["days"]
        linker["days"] = by_day
        hit, _ = linker["link"](dict(names=names, day=day, settle=settle), by_day)
        if not hit:
            return None, None, None
        g, yes_home = hit[1], hit[2][yes]
    log = play_log("nba" if league == "nba" else "ncaamb", g["id"])
    if log is None:
        return None, None, g["start"]
    if (log[-1][3] > log[-1][4]) != ((m["_outcome"] == 1) == yes_home):
        return None, None, g["start"]
    if not -900 <= log[0][0]-g["start"] <= 5400 or not 1800 <= settle-log[0][0] <= 6*3600:
        return None, None, g["start"]
    return log, yes_home, g["start"]


def pre_price(bars, start, margin):
    """(P0 on the YES side, kind) from the quotes before the start."""
    late = [quote(b) for b in bars if b["end_period_ts"] <= start-margin][-15:]
    if len(late) >= 10 and all(q and q[1]-q[0] <= 4 for q in late):
        mids = [(q[0]+q[1])/2 for q in late]
        if max(mids)-min(mids) <= 3:
            return float(np.median(mids)), "tip"
    first = [quote(b) for b in bars[:15]]
    if len(first) == 15 and bars[14]["end_period_ts"] <= start-margin and all(q and q[1]-q[0] <= 4 for q in first):
        mids = [(q[0]+q[1])/2 for q in first]
        if max(mids)-min(mids) <= 3:
            return float(np.median(mids)), "3h"
    loose = [q for q in (quote(b) for b in bars if b["end_period_ts"] <= start-margin) if q][-15:]
    if len(loose) >= 5:
        return float(np.median([(q[0]+q[1])/2 for q in loose])), "loose"
    return None, None


def build():
    from research import favourite_contexts as fc
    from research import college_basketball as cb
    OUT.mkdir(parents=True, exist_ok=True)
    store = Bars()
    cols, games, cover = defaultdict(list), [], Counter()
    for league, series in LEAGUES.items():
        fam = league
        slates = slates_of("nba" if league == "nba" else "ncaamb") if league != "cbb-women" else {}
        linker = dict(link=cb.link)
        for e in sorted(events_of(league).values(), key=lambda x: (x[0], x[2])):
            settle, _, event, archived, ms = e
            cover[league, "listed"] += 1
            if len(ms) < 2 or not all(m.get("_outcome") in (0.0, 1.0) for m in ms):
                cover[league, "not settled at 0 or 1"] += 1
                continue
            bars = store.get(series, e)
            if not bars:
                cover[league, "no bars on disk"] += 1
                continue
            m = ms[0]
            log, yes_home, sched = (None, None, None) if league == "cbb-women" else link_plays(league, e, slates, linker)
            if log is not None:
                start, kind = log[0][0], "plays"
            elif league == "nba":
                start = scheduled(event) or sched
                kind = "schedule"
                if not start:
                    expected = field_ts(m, "expected_expiration_time")
                    start = expected-3*3600 if expected and 2700 <= settle-(expected-3*3600) <= 8*3600 else None
            else:
                start = fc.start_of(fam, e)
                kind = "estimate" if archived or not field_ts(m, "expected_expiration_time") else "schedule"
            if not start or not 1800 <= settle-start <= 7*3600:
                cover[league, "no usable start"] += 1
                continue
            margin = 25*60 if kind == "estimate" else 120
            p0, p0_kind = pre_price(bars, start, margin)
            if p0 is None:
                cover[league, "no pre-game quote on disk"] += 1
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
                cover[league, "fewer than 20 in-play quotes"] += 1
                continue
            last = tsb[inplay][-1]
            n = int((last-t0)//60)+1
            for a in (bid, ask):
                a[~two] = np.nan
            otwo = (obid > 0) & (obid <= oask) & (oask < 100)
            obid[~otwo], oask[~otwo] = np.nan, np.nan
            g = {}
            g["bid"], obs = on_grid(tsb[two], bid[two], t0, n, True)
            g["ask"], _ = on_grid(tsb[two], ask[two], t0, n, True)
            ob, seen = on_grid(tsb, obid, t0, n, False)
            oa, _ = on_grid(tsb, oask, t0, n, False)
            prev_b, prev_a = np.r_[np.nan, g["bid"][:-1]], np.r_[np.nan, g["ask"][:-1]]
            g["obid"], g["oask"] = np.where(np.isnan(ob), prev_b, ob), np.where(np.isnan(oa), prev_a, oa)
            g["low"], _ = on_grid(tsb, low, t0, n, False)
            g["high"], _ = on_grid(tsb, high, t0, n, False)
            v, _ = on_grid(tsb, vol, t0, n, False)
            g["vol"] = np.nan_to_num(v)
            g["obs"] = obs.astype(float)
            # game state at each grid minute: period and seconds left in regulation at the bar's close; YES margin 120s earlier
            period, left, margin_yes = np.full(n, np.nan), np.full(n, np.nan), np.full(n, np.nan)
            half_end = second_start = np.nan
            if log is not None:
                times = [p[0] for p in log]
                grid_t = t0+60*np.arange(n)
                sign = 1 if yes_home else -1
                for k, t in enumerate(grid_t):
                    i = bisect_right(times, t)-1
                    if i >= 0 and t <= times[-1]+180:
                        period[k], left[k] = log[i][1], log[i][2]
                    j = bisect_right(times, t-SCORE_DELAY)-1
                    if j >= 0:
                        margin_yes[k] = sign*(log[j][3]-log[j][4])
                mid_period = 2 if league == "nba" else 1
                first = [p[0] for p in log if p[1] <= mid_period]
                second = [p[0] for p in log if p[1] > mid_period]
                if first and second:
                    half_end, second_start = first[-1], second[0]
            g["period"], g["left"], g["margin"] = period, left, margin_yes
            gid = len(games)
            final = 100*m["_outcome"]
            for key, a in g.items():
                end = final if key in ("bid", "ask", "obid", "oask") else (0.0 if key in ("vol", "obs") else np.nan)
                cols[key].append(np.r_[a, end].astype(np.float32))
            cols["g"].append(np.full(n+1, gid, np.int32))
            cols["term"].append(np.r_[np.zeros(n, bool), True])
            games.append(dict(gid=gid, league=league, event=event, ticker=m["ticker"], archived=bool(archived), settle=settle, t0=t0, start=float(start),
                              start_kind=kind, p0=p0, p0_kind=p0_kind, outcome=m["_outcome"], n=n, plays=log is not None,
                              yes_home=yes_home, half_end=half_end, second_start=second_start,
                              volume=sum(float(x.get("volume_fp") or x.get("volume") or 0) for x in ms)))
            cover[league, "built"] += 1
            cover[league, f"start: {kind}"] += 1
            cover[league, f"p0: {p0_kind}"] += 1
    pd.DataFrame(games).to_pickle(OUT/"games.pkl")
    np.savez_compressed(OUT/"grid.npz", **{k: np.concatenate(v) for k, v in cols.items()})
    text = {f"{a} | {b}": v for (a, b), v in sorted(cover.items())}
    (OUT/"coverage.json").write_text(json.dumps(text, indent=1))
    print(json.dumps(text, indent=1))
    print(f"BUILD {len(games)} games, {sum(len(v) for v in cols['g'])} grid points")


# ---------------------------------------------------------------- shared

class Data:
    """The grid in the favourite's terms."""

    def __init__(self):
        M = self.M = pd.read_pickle(OUT/"games.pkl")
        z = np.load(OUT/"grid.npz")
        self.g = z["g"]
        yes = (M.p0.values > 50)[self.g]
        flip = lambda a, b: np.where(yes, a, 100-b).astype(np.float64)
        self.bid, self.ask = flip(z["bid"], z["ask"]), flip(z["ask"], z["bid"])
        self.obid, self.oask = flip(z["obid"], z["oask"]), flip(z["oask"], z["obid"])
        self.low, self.high = flip(z["low"], z["high"]), flip(z["high"], z["low"])
        self.margin = np.where(yes, z["margin"], -z["margin"]).astype(float)
        self.vol, self.obs, self.term = z["vol"].astype(float), z["obs"] > 0, z["term"]
        self.mid, self.spread = (self.bid+self.ask)/2, self.ask-self.bid
        n = len(self.g)
        change = np.r_[True, self.g[1:] != self.g[:-1]]
        self.first = np.where(change)[0][self.g]
        self.end = (np.r_[np.where(change)[0][1:], n]-1)[self.g]
        self.i = np.arange(n)
        self.league = M.league.values
        self.fav0 = np.maximum(M.p0.values, 100-M.p0.values)
        self.payout = np.where(M.p0.values > 50, M.outcome.values, 1-M.outcome.values)
        self.band = np.array([next(k for k, (_, lo, hi) in enumerate(BANDS) if lo <= v < hi) for v in self.fav0])
        self.half = np.zeros(len(M), int)
        for league in LEAGUES:
            s = self.league == league
            self.half[s] = (M.settle.values[s] > np.median(M.settle.values[s])).astype(int)
        self.maker_rate = np.array([MAKER_RATE[x] for x in self.league])
        # phase of the game at each grid point, from the clock where the game is linked to play-by-play
        t = M.t0.values[self.g]+60*(self.i-self.first)
        per, left = z["period"].astype(float), z["left"].astype(float)
        gap = (M.second_start.values-M.half_end.values)/60
        ok_half = (gap >= 8) & (gap <= 30)
        self.ok_half = ok_half
        ht = ok_half[self.g] & (t >= M.half_end.values[self.g]+60) & (t < M.second_start.values[self.g])
        nba, plays = (self.league == "nba")[self.g], M.plays.values[self.g]
        phase = np.full(n, "no clock", dtype=object)
        el = self.i-self.first
        for lo, hi, name in ((0, 50, "no clock: min 0-50"), (50, 100, "no clock: min 50-100"), (100, 999, "no clock: min 100+")):
            phase[~plays & (el >= lo) & (el < hi)] = name
        c = plays & ~nba
        phase[c & (per == 1)] = "1st half"
        phase[c & (per >= 2) & (left > 300)] = "2nd half before last 5"
        phase[c & (per >= 2) & (left <= 300)] = "last 5 min + OT"
        c = plays & nba
        for q in (1, 2, 3):
            phase[c & (per == q)] = f"Q{q}"
        phase[c & (per >= 4) & (left > 300)] = "Q4 before last 5"
        phase[c & (per >= 4) & (left <= 300)] = "last 5 min + OT"
        phase[plays & ht] = "halftime"
        phase[plays & np.isnan(per)] = "after final play"
        phase[self.term] = "terminal"
        self.phase = phase

    def fwd(self, i, h):
        return np.minimum(i+h, self.end[i])

    def cells(self, bands=True):
        """(league, half name, band name, boolean mask over games)."""
        for league in LEAGUES:
            for h, half in ((0, "half1"), (1, "half2"), (None, "both")):
                for b, name in ([(None, "all")]+(list(enumerate(BAND_NAMES)) if bands else [])):
                    yield league, half, name, ((self.league == league) & ((self.half == h) if h is not None else True)
                                               & ((self.band == b) if b is not None else True))


def fmt(r, key="slope", digits=3):
    return "      n/a          " if np.isnan(r[key]) else f"{r[key]:+.{digits}f}[{r['lo']:+.{digits}f},{r['hi']:+.{digits}f}]"


def stat(v, g, digits=2):
    r = cluster_mean(np.asarray(v, float), np.asarray(g))
    return "      n/a        " if np.isnan(r["mean"]) else f"{r['mean']:+6.{digits}f}[{r['lo']:+6.{digits}f},{r['hi']:+6.{digits}f}]"


# ---------------------------------------------------------------- (a) slopes and (b) variance ratios

def martingale():
    d = Data()
    live = ~d.term
    level = (d.mid >= 5) & (d.mid <= 95)
    rows = []
    for h in HORIZONS:
        ok = live & (d.i-h >= d.first) & level
        i = d.i[ok]
        x = d.mid[i]-d.mid[i-h]
        nxt = d.fwd(i, 1)
        ys = dict(raw=d.mid[d.fwd(i, h)]-d.mid[i], skip=d.mid[d.fwd(nxt, h)]-d.mid[nxt])
        tight = (d.spread[i] <= 2) & (d.spread[i-h] <= 2) & d.obs[i] & (d.spread[nxt] <= 2)
        gi = d.g[i]

        def add(test, league, half, cut, sel):
            for name, yy, extra in (("raw", ys["raw"], None), ("skip", ys["skip"], None), ("tight", ys["skip"], tight)):
                s = sel if extra is None else sel & extra
                rows.append(dict(test=test, league=league, half=half, cut=cut, h=h, variant=name, **cluster_slope(x[s], yy[s], gi[s])))
        for league, half, band, mask in d.cells():
            add("band", league, half, band, mask[gi])
        for league in LEAGUES:
            for hh, half in ((0, "half1"), (1, "half2"), (None, "both")):
                base = (d.league[gi] == league) & ((d.half[gi] == hh) if hh is not None else True)
                for ph in sorted(set(d.phase[i][base])):
                    add("phase", league, half, ph, base & (d.phase[i] == ph))
                for lo, hi in LEVELS:
                    add("level", league, half, f"price {lo}-{int(hi)}", base & (d.mid[i] >= lo) & (d.mid[i] < hi))
    S = pd.DataFrame(rows)
    S.to_csv(OUT/"slopes.csv", index=False, float_format="%.5f")
    print("\n(a) SLOPE of the next-h change on the last-h change (favourite's midpoint, cents); negative = reversion, positive = momentum")
    print("    raw = shares the midpoint at t; skip = follow-up measured from t+1; tight = skip and quotes at t-h, t, t+1 at most 2c wide")
    for test in ("band", "phase", "level"):
        print(f"  --- by {test}")
        for league in LEAGUES:
            q0 = S[(S.test == test) & (S.league == league)]
            for cut in q0.cut.unique():
                for variant in ("raw", "skip", "tight"):
                    if test != "band" and variant == "raw":
                        continue
                    for half in (("half1", "half2", "both") if cut == "all" or test != "band" else ("both",)):
                        q = q0[(q0.cut == cut) & (q0.variant == variant) & (q0.half == half)].set_index("h")
                        if q.empty or q.n.max() < 200:
                            continue
                        print(f"   {league:9} {cut:24} {variant:5} {half:5} "+" ".join(f"h{h}:{fmt(q.loc[h])}" for h in HORIZONS)
                              + f"  n(h5)={int(q.n[5])} games={int(q.matches[5])}")

    print("\n(b) VARIANCE RATIO of h-minute to one-minute changes (1 = martingale, <1 = reversion, >1 = momentum); bootstrap over games")
    d1 = np.r_[np.nan, np.diff(d.mid)]
    d1[d.i == d.first] = np.nan
    usable = ~np.isnan(d1)
    sq = np.where(usable, d1**2, 0.0)
    c_sq, c_bad = np.r_[0, np.cumsum(sq)], np.r_[0, np.cumsum(~usable)]
    vrows = []
    for label in ("to settlement", "survivors only"):
        for h in HORIZONS[1:]:
            ok = live & ~np.isnan(d.mid) & level
            if label == "survivors only":
                ok &= d.i+h < d.end
            i = d.i[ok]
            j = d.fwd(i, h)
            ok2 = (c_bad[j+1]-c_bad[i+1]) == 0
            i, j = i[ok2], j[ok2]
            num, den = (d.mid[j]-d.mid[i])**2, c_sq[j+1]-c_sq[i+1]
            for league, half, band, mask in d.cells():
                if band != "all" and half != "both":
                    continue
                sel = mask[d.g[i]]
                if sel.sum() < 100:
                    continue
                _, inv = np.unique(d.g[i][sel], return_inverse=True)
                A, B = np.bincount(inv, num[sel]), np.bincount(inv, den[sel])
                w = RNG.multinomial(len(A), np.full(len(A), 1/len(A)), size=300)
                boot = (w@A)/(w@B)
                vrows.append(dict(kind=label, league=league, half=half, band=band, h=h, vr=A.sum()/B.sum(),
                                  lo=np.percentile(boot, 2.5), hi=np.percentile(boot, 97.5), games=len(A)))
    V = pd.DataFrame(vrows)
    V.to_csv(OUT/"variance-ratios.csv", index=False, float_format="%.5f")
    for kind in ("to settlement", "survivors only"):
        print(f"  {kind}" + ("  (WARNING: conditions on the game still being quoted h minutes later; biased towards reversion)" if kind != "to settlement" else ""))
        for (league, band, half), q in V[V.kind == kind].groupby(["league", "band", "half"], sort=False):
            if kind == "survivors only" and band != "all":
                continue
            q = q.set_index("h")
            print(f"   {league:9} {band:6} {half:5} "+" ".join(f"h{h}:{q.vr[h]:.3f}[{q.lo[h]:.3f},{q.hi[h]:.3f}]" for h in HORIZONS[1:])+f" games={q.games.iloc[0]}")


# ---------------------------------------------------------------- (c) events and tradeable versions

def find_events(d):
    """First bar where the M-minute change of the favourite's midpoint reaches X (per M, threshold and direction), then an M-bar pause."""
    live, out = ~d.term, []
    for M in WITHIN:
        back = d.i-M
        ok = live & (back >= d.first)
        ch = np.full(len(d.i), np.nan)
        ch[ok] = d.mid[ok]-d.mid[back[ok]]
        ch[~((d.mid >= 5) & (d.mid <= 95))] = np.nan
        for X in MOVES:
            for sign, name in ((-1, "fav fell"), (1, "dog fell")):
                last_g, last_i = -1, -99
                for i in np.where(sign*ch >= X)[0]:
                    if d.g[i] != last_g or i-last_i > M:
                        out.append((i, M, X, name, ch[i]))
                        last_g, last_i = d.g[i], i
    return pd.DataFrame(out, columns=["i", "M", "X", "dir", "move"])


def price_event(d, E, i=None):
    """Add drifts and trade results to events at grid index E.i (or `i`). `fade` buys the team that fell, `follow` the other."""
    i = E.i.values if i is None else i
    g = d.g[i]
    fall = (E["dir"] == "fav fell").values
    sgn = np.where(fall, 1.0, -1.0)                                    # faller's price change = sgn * favourite's change
    E["g"], E["league"], E["half"], E["band"] = g, d.league[g], d.half[g], np.array(BAND_NAMES)[d.band[g]]
    E["phase"], E["minute"], E["obs"] = d.phase[i], i-d.first[i], d.obs[i]
    E["fav_price"], E["spread"] = d.mid[i], d.spread[i]
    E["price"] = np.where(fall, d.mid[i], 100-d.mid[i])                # the faller's midpoint after the fall
    E["vol_bar"], E["vol_next"] = d.vol[i], d.vol[d.fwd(i, 1)]
    final = 100*d.payout[g]
    e = d.fwd(i, 1)
    can = ~d.term[e]
    E["entry_spread"] = np.where(can, d.oask[e]-d.obid[e], np.nan)
    for k in AFTER:
        E[f"reb{k}"] = sgn*(d.mid[d.fwd(i, k)]-d.mid[i])
        E[f"reb{k}_skip"] = sgn*(d.mid[d.fwd(e, k)]-d.mid[e])
    E["rebS"], E["rebS_skip"] = sgn*(final-d.mid[i]), sgn*(final-d.mid[e])
    rate = d.maker_rate[g]
    res = {}
    for who in ("fav", "dog"):
        a = np.where(can, d.oask[e] if who == "fav" else 100-d.obid[e], np.nan)
        a[(a < 1) | (a > 99)] = np.nan
        res[who, "ask"] = a
        end = final if who == "fav" else 100-final
        bidx = lambda x: d.bid[x] if who == "fav" else 100-d.ask[x]
        net_bid = lambda x: np.where(d.term[x], bidx(x), bidx(x)-fee(bidx(x)))
        for k in AFTER:
            res[who, f"take{k}"] = net_bid(d.fwd(e, k))-a-fee(a)
        res[who, "takeS"] = end-a-fee(a)
        for Y in (3, 5):                                               # resting offer Y above the entry price for 10 bars
            sold = np.zeros(len(i), bool)
            for j in range(1, 11):
                x = d.fwd(e, j)
                high = d.high[x] if who == "fav" else 100-d.low[x]
                sold |= ~d.term[x] & (x == e+j) & (high > a+Y)
            res[who, f"offer{Y}"] = np.where(sold, Y-fee(a)-fee(np.minimum(a+Y, 99), rate), net_bid(d.fwd(e, 10))-a-fee(a))
            res[who, f"offer{Y}_sold"] = np.where(np.isnan(a), np.nan, sold)
        # maker entry: bid at the event bar's closing bid, five bars, strict print-through
        L = d.bid[i] if who == "fav" else 100-d.ask[i]
        fill = np.full(len(i), -1)
        for j in range(5, 0, -1):
            x = d.fwd(i, j)
            low = d.low[x] if who == "fav" else 100-d.high[x]
            hit = ~d.term[x] & (x == i+j) & (low < L)
            fill[hit] = x[hit]
        got = (fill >= 0) & (L >= 1) & (L <= 99)
        f = np.where(got, fill, 0)
        mfee = fee(L, rate)
        res[who, "make_fill"] = got.astype(float)
        res[who, "make_L"] = np.where(got, L, np.nan)
        res[who, "make_after1"] = np.where(got, (d.mid[d.fwd(f, 1)] if who == "fav" else 100-d.mid[d.fwd(f, 1)])-L, np.nan)
        res[who, "make_take5"] = np.where(got, net_bid(d.fwd(f, 5))-L-mfee, np.nan)
        res[who, "makeS"] = np.where(got, end-L-mfee, np.nan)
        sold = np.zeros(len(i), bool)
        for j in range(1, 11):
            x = d.fwd(f, j)
            high = d.high[x] if who == "fav" else 100-d.low[x]
            sold |= ~d.term[x] & (x == f+j) & (high > L+3)
        res[who, "make_offer3"] = np.where(got, np.where(sold, 3-mfee-fee(np.minimum(L+3, 99), rate), net_bid(d.fwd(f, 10))-L-mfee), np.nan)
    keys = sorted({k for _, k in res})
    for k in keys:
        E["fade_"+k] = np.where(fall, res["fav", k], res["dog", k])
        E["follow_"+k] = np.where(fall, res["dog", k], res["fav", k])
    return E


TRADES = ["take1", "take3", "take5", "take10", "takeS", "offer3", "offer5", "make_take5", "make_offer3", "makeS"]
DRIFTS = [f"reb{k}" for k in AFTER]+["rebS"]+[f"reb{k}_skip" for k in AFTER]+["rebS_skip"]


def cell_row(q, extra=()):
    row = dict(events=len(q), games=q.g.nunique(), mean_move=q.move.abs().mean(), price=q.price.mean(), spread=q.spread.mean(),
               entry_spread=q.entry_spread.mean(), vol_bar=q.vol_bar.median(), vol_next=q.vol_next.median(),
               fade_fill=q.fade_make_fill.mean(), follow_fill=q.follow_make_fill.mean(), offer3_sold=q.fade_offer3_sold.mean())
    cols = DRIFTS+["fade_make_after1", "follow_make_after1"]+[f"{s}_{t}" for s in ("fade", "follow") for t in TRADES]+list(extra)
    for c in cols:
        v = q[c].values.astype(float)
        r = cluster_mean(v, q.g.values)
        row.update({c: r["mean"], c+"_lo": r["lo"], c+"_hi": r["hi"], c+"_n": r["n"]})
        if c.startswith(("fade_", "follow_")) and "after1" not in c:
            ok = ~np.isnan(v)
            paid = q["fade_make_L" if "make" in c and c.startswith("fade") else "follow_make_L" if "make" in c else c.split("_")[0]+"_ask"].values[ok]
            row[c+"_pct"] = 100*np.mean(v[ok]/paid) if ok.any() else np.nan
            row[c+"_win"] = np.mean(v[ok] > 0) if ok.any() else np.nan
            row[c+"_loser"] = v[ok][v[ok] <= 0].mean() if (v[ok] <= 0).any() else np.nan
    return row


def events():
    d = Data()
    E = price_event(d, find_events(d))
    E.to_pickle(OUT/"events.pkl")
    report_events(E)


def report_events(E=None):
    E = pd.read_pickle(OUT/"events.pkl") if E is None else E
    rows = []
    halves = ((0, "half1"), (1, "half2"), (None, "both"))
    for league in LEAGUES:
        L = E[E.league == league]
        for M in WITHIN:
            for X in MOVES:
                base = L[(L.M == M) & (L.X == X)]
                for h, half in halves:
                    qh = base if h is None else base[base.half == h]
                    for flt in ("any", "tight"):
                        qf = qh if flt == "any" else qh[(qh.spread <= 2) & qh.obs & (qh.entry_spread <= 2)]
                        cuts = [("all", "all", qf)]
                        cuts += [("who", w, qf[qf["dir"] == w]) for w in ("fav fell", "dog fell")]
                        if flt == "any":
                            cuts += [("band", f"{b} {w}", qf[(qf.band == b) & (qf["dir"] == w)]) for b in BAND_NAMES for w in ("fav fell", "dog fell")]
                            cuts += [("phase", p, qf[qf.phase == p]) for p in sorted(qf.phase.unique())]
                            cuts += [("level", f"faller at {lo}-{int(hi)}", qf[(qf.price >= lo) & (qf.price < hi)]) for lo, hi in LEVELS]
                        for kind, cut, q in cuts:
                            if len(q) >= 20:
                                rows.append(dict(league=league, M=M, X=X, half=half, filter=flt, kind=kind, cut=cut, **cell_row(q)))
    R = pd.DataFrame(rows)
    R.to_csv(OUT/"event-cells.csv", index=False, float_format="%.3f")

    def ci(r, c, pct=False):
        if r is None or np.isnan(r[c]):
            return "   n/a"
        s = f"{r[c]:+5.2f}[{r[c+'_lo']:+5.2f},{r[c+'_hi']:+5.2f}]"
        return s+f"({r[c+'_pct']:+.1f}%)" if pct else s

    def pick(league, M, X, half, flt="any", kind="all", cut="all"):
        q = R[(R.league == league) & (R.M == M) & (R.X == X) & (R.half == half) & (R["filter"] == flt) & (R.kind == kind) & (R.cut == cut)]
        return None if q.empty else q.iloc[0]

    print("\n(c) CONDITIONAL DRIFT after a team's price falls X cents or more within M minutes: the FALLER's midpoint rebound (cents; + = reversion)")
    print("    measured from the event bar (reb) and from the next bar (skip: free of a stale quote at t); S = to settlement; intervals clustered by game")
    for league in LEAGUES:
        for M in WITHIN:
            for X in MOVES:
                for half in ("half1", "half2", "both"):
                    r = pick(league, M, X, half)
                    if r is None:
                        continue
                    print(f"   {league:9} {X:2}c in {M}m {half:5} n={int(r.events):5} games={int(r.games):4} | "
                          + " ".join(f"{k}:{ci(r, 'reb'+str(k))}" for k in list(AFTER)+["S"]) + " | skip "
                          + " ".join(f"{k}:{ci(r, f'reb{k}_skip')}" for k in list(AFTER)+["S"]))
    print("  tight quotes only (event bar real, spread at t and at entry at most 2c), from the next bar")
    for league in LEAGUES:
        for M in WITHIN:
            for X in MOVES:
                r = pick(league, M, X, "both", "tight")
                a = pick(league, M, X, "both")
                if r is not None:
                    print(f"   {league:9} {X:2}c in {M}m n={int(r.events):5} of {int(a.events):5} | " + " ".join(f"{k}:{ci(r, f'reb{k}_skip')}" for k in list(AFTER)+["S"]))
    for kind in ("who", "band", "phase", "level"):
        print(f"  cut by {kind}: 10c in 5 minutes, rebound from the next bar (both halves pooled; then half1 / half2 for the 5-minute rebound)")
        for league in LEAGUES:
            q = R[(R.league == league) & (R.M == 5) & (R.X == 10) & (R["filter"] == "any") & (R.kind == kind)]
            for cut in q.cut.unique():
                r = pick(league, 5, 10, "both", "any", kind, cut)
                if r is None:
                    continue
                h1, h2 = pick(league, 5, 10, "half1", "any", kind, cut), pick(league, 5, 10, "half2", "any", kind, cut)
                print(f"   {league:9} {cut:26} n={int(r.events):5} " + " ".join(f"{k}:{ci(r, f'reb{k}_skip')}" for k in list(AFTER)+["S"])
                      + f" | 5m h1 {ci(h1, 'reb5_skip')} h2 {ci(h2, 'reb5_skip')}")

    print("\n  TRADEABLE. FADE = buy the team that fell; FOLLOW = buy the other team. Net cents per trade after fees [95%] (mean % of price paid)")
    print("   take N = next bar's opening ask, sell at the bid N minutes later; S = hold to settlement; offer Y = resting offer Y above entry, 10 bars")
    for side in ("fade", "follow"):
        for league in LEAGUES:
            for M in WITHIN:
                for X in MOVES:
                    for half in ("half1", "half2", "both"):
                        r = pick(league, M, X, half)
                        if r is None:
                            continue
                        print(f"   {side:6} {league:9} {X:2}c in {M}m {half:5} n={int(r[side+'_take5_n']):5} | "
                              + " ".join(f"{t}:{ci(r, f'{side}_{t}', True)}" for t in ("take1", "take3", "take5", "take10", "takeS", "offer3", "offer5")))
    print("   win share and average loser, both halves pooled")
    for side in ("fade", "follow"):
        for league in LEAGUES:
            for X in (10, 15):
                r = pick(league, 5, X, "both")
                if r is not None:
                    print(f"   {side:6} {league:9} {X:2}c in 5m | " + " ".join(
                        f"{t}: win {100*r[f'{side}_{t}_win']:3.0f}% loser {r[f'{side}_{t}_loser']:+6.1f}c" for t in ("take5", "take10", "takeS", "offer3", "offer5"))
                        + f" | offer3 sold {100*r.offer3_sold:3.0f}%")
    print("  MAKER entry: bid at the event bar's closing bid for five bars, strict print-through. after1 = midpoint one bar after the fill minus the bid")
    for side in ("fade", "follow"):
        for league in LEAGUES:
            for M in WITHIN:
                for X in MOVES:
                    for half in ("half1", "half2", "both"):
                        r = pick(league, M, X, half)
                        if r is None:
                            continue
                        print(f"   {side:6} {league:9} {X:2}c in {M}m {half:5} fills={int(r[side+'_makeS_n']):5} ({100*r[side+'_fill']:3.0f}%) | after1 {ci(r, side+'_make_after1')} "
                              + " ".join(f"{t}:{ci(r, f'{side}_{t}', True)}" for t in ("make_take5", "make_offer3", "makeS")))
    print("  COSTS AND CAPACITY at the event (both halves): spread at the event bar's close and at entry, contracts in the event bar and the next")
    for league in LEAGUES:
        for M in WITHIN:
            for X in MOVES:
                r = pick(league, M, X, "both")
                if r is not None:
                    rt = r.entry_spread+2*fee(r.price)
                    print(f"   {league:9} {X:2}c in {M}m events {int(r.events):5} games {int(r.games):4} | faller price {r.price:4.1f} | spread close {r.spread:5.2f} entry {r.entry_spread:5.2f} | "
                          f"contracts/min median: event bar {r.vol_bar:8.0f} next {r.vol_next:8.0f} | taker round trip about {rt:4.1f}c ({100*rt/r.price:4.1f}% of price)")
    # search for anything positive in both halves
    hits, tested = [], 0
    A = R[(R["filter"] == "any")]
    key = ["league", "M", "X", "kind", "cut"]
    h1, h2 = A[A.half == "half1"].set_index(key), A[A.half == "half2"].set_index(key)
    for k in h1.index.intersection(h2.index):
        a, b = h1.loc[k], h2.loc[k]
        for side in ("fade", "follow"):
            for t in TRADES:
                c = f"{side}_{t}"
                if a[c+"_n"] < 30 or b[c+"_n"] < 30 or np.isnan(a[c]) or np.isnan(b[c]):
                    continue
                tested += 1
                if a[c+"_lo"] > 0 and b[c] > 0:
                    hits.append(dict(cell=" | ".join(map(str, k)), rule=c, half1=a[c], half1_lo=a[c+"_lo"], half1_n=int(a[c+"_n"]), half1_pct=a[c+"_pct"],
                                     half2=b[c], half2_lo=b[c+"_lo"], half2_n=int(b[c+"_n"]), half2_pct=b[c+"_pct"], both_lo=bool(b[c+"_lo"] > 0)))
    print(f"\n  SEARCH: {tested} tradeable cells with 30+ trades in each half; {len(hits)} have a first-half interval above zero AND a positive second-half mean"
          f" ({sum(h['both_lo'] for h in hits)} with both intervals above zero); chance alone would give roughly {tested*0.025*0.5:.0f}")
    for h in sorted(hits, key=lambda h: -h["half2"]):
        print(f"   {h['cell']:55} {h['rule']:18} half1 {h['half1']:+6.2f} (lo {h['half1_lo']:+5.2f}, n {h['half1_n']:4}, {h['half1_pct']:+5.1f}%)"
              f" half2 {h['half2']:+6.2f} (lo {h['half2_lo']:+5.2f}, n {h['half2_n']:4}, {h['half2_pct']:+5.1f}%)")
    (OUT/"event-search.json").write_text(json.dumps(dict(tested=tested, hits=hits), indent=1))


# ---------------------------------------------------------------- structure specific to basketball

def simple(v):
    """mean [95%] treating rows as independent (one row per game)."""
    v = np.asarray(v, float)
    v = v[~np.isnan(v)]
    if len(v) < 5:
        return f"n={len(v):4} n/a"
    se = v.std(ddof=1)/np.sqrt(len(v))
    return f"n={len(v):4} {v.mean():+6.2f}[{v.mean()-1.96*se:+6.2f},{v.mean()+1.96*se:+6.2f}]"


def special():
    d = Data()
    M = d.M
    first_of = np.r_[np.where(np.r_[True, d.g[1:] != d.g[:-1]])[0]]          # first grid index of each game
    end_of = np.r_[first_of[1:], len(d.g)]-1
    names = {0: "half1", 1: "half2"}

    # (1) halftime: is the price at the break biased for teams that closed the half on a run?
    print("\n(4a) HALFTIME. Price two minutes into the break against the final result. Run = change in the score margin over the last six")
    print("     wall-clock minutes of the half (score read once the half is over, so the lookahead flaw cannot matter). One row per game.")
    rows = []
    for gid in np.where(M.plays.values & d.ok_half)[0]:
        k = first_of[gid]+int(np.ceil((M.half_end.values[gid]+120-M.t0.values[gid])/60))
        if k-6 < first_of[gid] or k+1 >= end_of[gid] or np.isnan(d.margin[k]) or np.isnan(d.margin[k-6]) or np.isnan(d.mid[k]):
            continue
        t_k = M.t0.values[gid]+60*(k-first_of[gid])
        if t_k >= M.second_start.values[gid]-120:
            continue
        j = min(first_of[gid]+int(np.ceil((M.second_start.values[gid]-M.t0.values[gid])/60)), end_of[gid])
        rows.append(dict(g=gid, league=d.league[gid], half=d.half[gid], mid=d.mid[k], spread=d.spread[k], run=d.margin[k]-d.margin[k-6],
                         prun=d.mid[k]-d.mid[k-6], margin=d.margin[k], win=d.payout[gid], ask=d.oask[k+1], bid=d.obid[k+1],
                         drift=d.mid[j]-d.mid[k], band=BAND_NAMES[d.band[gid]]))
    H = pd.DataFrame(rows)
    H = H[(H.mid >= 3) & (H.mid <= 97)]
    H.to_pickle(OUT/"halftime.pkl")
    for league in ("cbb-men", "nba"):
        L = H[H.league == league]
        print(f"   {league}: {len(L)} games with a clean halftime; favourite midpoint at the break {L.mid.mean():.1f}, wins {100*L.win.mean():.1f}%; "
              f"final minus price, all {simple(100*L.win-L.mid)}; spread {L.spread.mean():.2f}c; drift across the break (to the second-half tip) {simple(L.drift)}")
        for label, col, cut in (("score run 6+", "run", 6), ("score run 8+", "run", 8), ("price run 8c+", "prun", 8), ("price run 12c+", "prun", 12)):
            for hname, q in (("both", L), ("half1", L[L.half == 0]), ("half2", L[L.half == 1])):
                up, dn = q[q[col] >= cut], q[q[col] <= -cut]
                # in the RUNNER's terms: the team that closed the half on the run
                price = np.r_[up.mid.values, 100-dn.mid.values]
                win = np.r_[up.win.values, 1-dn.win.values]
                ask = np.r_[up.ask.values, 100-dn.bid.values]
                other_ask = np.r_[100-up.bid.values, dn.ask.values]
                follow = 100*win-ask-fee(ask)
                fade = 100*(1-win)-other_ask-fee(other_ask)
                print(f"     {label:14} {hname:5} runner's price {price.mean() if len(price) else np.nan:5.1f} | final minus midpoint (runner) {simple(100*win-price)} | "
                      f"buy the runner at the ask, hold {simple(follow)} ({100*np.nanmean(follow/ask) if len(ask) else np.nan:+.1f}%) | buy the other team {simple(fade)} ({100*np.nanmean(fade/other_ask) if len(ask) else np.nan:+.1f}%)")
        print("     by the favourite's price at the break (final minus midpoint): " + " | ".join(
            f"{lo}-{hi}: {simple(100*L.win[(L.mid >= lo) & (L.mid < hi)]-L.mid[(L.mid >= lo) & (L.mid < hi)])}" for lo, hi in ((3, 30), (30, 50), (50, 70), (70, 85), (85, 97.01))))

    # (2) scoring runs from play-by-play (score delayed 120 seconds)
    print(f"\n(4b) SCORING RUNS from play-by-play, score read {SCORE_DELAY}s before the bar's close. Event: the margin moved P points or more against a team")
    print("     within three minutes (first bar, then a three-bar pause), price in 5-95. FADE buys the team that suffered the run at the next bar's opening ask.")
    live = ~d.term
    ok = live & (d.i-3 >= d.first)
    run = np.full(len(d.i), np.nan)
    run[ok] = d.margin[ok]-d.margin[d.i[ok]-3]
    run[~((d.mid >= 5) & (d.mid <= 95))] = np.nan
    out = []
    for P in (6, 8, 10, 12):
        for sign, name in ((-1, "fav fell"), (1, "dog fell")):
            last_g, last_i = -1, -99
            for i in np.where(sign*run >= P)[0]:
                if d.g[i] != last_g or i-last_i > 3:
                    out.append((i, 3, P, name, run[i]))
                    last_g, last_i = d.g[i], i
    R = price_event(d, pd.DataFrame(out, columns=["i", "M", "X", "dir", "move"]))
    i = R.i.values
    R["price_move5"] = np.where(R["dir"] == "fav fell", 1, -1)*(d.mid[i]-d.mid[np.maximum(i-5, d.first[i])])
    R.to_pickle(OUT/"runs.pkl")
    for league in ("cbb-men", "nba"):
        for P in (6, 8, 10, 12):
            for h, hname in ((None, "both"), (0, "half1"), (1, "half2")):
                q = R[(R.league == league) & (R.X == P) & ((R.half == h) if h is not None else True)]
                if len(q) < 30:
                    continue
                c = cell_row(q)
                print(f"   {league:8} {P:2}+ points {hname:5} n={len(q):5} games={q.g.nunique():4} price already moved {q.price_move5.mean():+5.1f}c | rebound from next bar "
                      + " ".join(f"{k}:{c[f'reb{k}_skip']:+5.2f}[{c[f'reb{k}_skip_lo']:+5.2f},{c[f'reb{k}_skip_hi']:+5.2f}]" for k in list(AFTER)+["S"])
                      + f" | fade take5 {c['fade_take5']:+5.2f}({c['fade_take5_pct']:+.1f}%) takeS {c['fade_takeS']:+5.2f}[{c['fade_takeS_lo']:+5.2f},{c['fade_takeS_hi']:+5.2f}]({c['fade_takeS_pct']:+.1f}%)"
                      + f" makeS {c['fade_makeS']:+5.2f}[{c['fade_makeS_lo']:+5.2f},{c['fade_makeS_hi']:+5.2f}] | follow take5 {c['follow_take5']:+5.2f} takeS {c['follow_takeS']:+5.2f}[{c['follow_takeS_lo']:+5.2f},{c['follow_takeS_hi']:+5.2f}]")
        # every minute: does the delayed run predict the next five minutes of price (from the next bar)?
        s = ok & M.plays.values[d.g] & (d.league[d.g] == league) & ~np.isnan(run)
        ii = d.i[s]
        nxt = d.fwd(ii, 1)
        for h, hname in ((None, "both"), (0, "half1"), (1, "half2")):
            ss = np.ones(len(ii), bool) if h is None else d.half[d.g[ii]] == h
            r5 = cluster_slope(run[ii][ss], (d.mid[d.fwd(nxt, 5)]-d.mid[nxt])[ss], d.g[ii][ss])
            r1 = cluster_slope(run[ii][ss], (d.mid[d.fwd(nxt, 1)]-d.mid[nxt])[ss], d.g[ii][ss])
            print(f"   {league:8} every minute {hname:5}: cents of price change per point of delayed three-minute run: next 1m {fmt(r1)} next 5m {fmt(r5)} (n={r5['n']}); negative = overreaction")

    # (3) a 90c+ favourite that falls to 80c: fast versus slow
    print("\n(4c) A 90c+ PRE-GAME FAVOURITE THAT TRADES BELOW 80c IN PLAY (the live rule's fill: first in-play bar with a print strictly below 80).")
    print("     Net = 100 x win - 80 - maker fee. Speed A: the midpoint five minutes before the fill bar (88+ = fast). Speed B: minutes since the midpoint was last 90+.")
    rows = []
    for gid in np.where((d.fav0 >= 90) & (d.fav0 < 99) & np.isin(M.p0_kind.values, ("tip", "3h")))[0]:
        a, b = first_of[gid]+1, end_of[gid]
        hit = np.where(d.low[a:b] < 80)[0]
        if not len(hit):
            rows.append(dict(g=gid, league=d.league[gid], half=d.half[gid], fill=False, win=d.payout[gid]))
            continue
        k = a+hit[0]
        before = d.mid[max(k-5, first_of[gid])]
        high = np.where(d.mid[first_of[gid]:k] >= 90)[0]
        since = k-(first_of[gid]+high[-1]) if len(high) else k-first_of[gid]+1
        rows.append(dict(g=gid, league=d.league[gid], half=d.half[gid], fill=True, win=d.payout[gid], minute=k-first_of[gid], before=before, since=since,
                         phase=d.phase[k], low=d.low[k], mid_after=d.mid[k], vol=d.vol[k], p0=d.fav0[gid]))
    F = pd.DataFrame(rows)
    F.to_pickle(OUT/"fav-dip-speed.pkl")

    def line(label, q):
        if len(q) < 5:
            return f"     {label:38} n={len(q):3}"
        rate = MAKER_RATE[q.league.iloc[0]] if q.league.nunique() == 1 else 1.75
        net = 100*q.win-80-fee(80, rate)
        se = net.std(ddof=1)/np.sqrt(len(q))
        return (f"     {label:38} n={len(q):3} win {100*q.win.mean():5.1f}% net {net.mean():+6.2f}c [{net.mean()-1.96*se:+6.2f},{net.mean()+1.96*se:+6.2f}] ({100*net.mean()/80:+5.1f}%)"
                f" | half1 n={int((q.half == 0).sum()):3} win {100*q.win[q.half == 0].mean():5.1f}% | half2 n={int((q.half == 1).sum()):3} win {100*q.win[q.half == 1].mean():5.1f}%")
    for league in list(LEAGUES)+["all three"]:
        L = F if league == "all three" else F[F.league == league]
        Q = L[L.fill]
        print(f"   {league}: 90c+ favourites {len(L)}, win {100*L.win.mean():.1f}%; never below 80: {int((~L.fill).sum())}, win {100*L.win[~L.fill].mean():.1f}%; filled at 80: {len(Q)}")
        if len(Q) < 10:
            continue
        print(line("all fills", Q))
        print(line("A fast: mid 88+ five minutes before", Q[Q.before >= 88]))
        print(line("A medium: 84-88", Q[(Q.before >= 84) & (Q.before < 88)]))
        print(line("A slow: 80-84", Q[(Q.before >= 80) & (Q.before < 84)]))
        print(line("A already under 80 (start misdated)", Q[Q.before < 80]))
        print(line("fill bar closed with the mid at 82+", Q[Q.mid_after >= 82]))
        print(line("fill bar closed with the mid under 79", Q[Q.mid_after < 79]))
        print(line("B fast: 90+ within 5 minutes", Q[Q.since <= 5]))
        print(line("B medium: 6-15 minutes", Q[(Q.since > 5) & (Q.since <= 15)]))
        print(line("B slow: over 15 minutes", Q[Q.since > 15]))
        print(line("gap: bar's low under 75", Q[Q.low < 75]))
        print(line("fill in the first 30 minutes", Q[Q.minute <= 30]))
        print(line("fill at 31-75 minutes", Q[(Q.minute > 30) & (Q.minute <= 75)]))
        print(line("fill after 75 minutes", Q[Q.minute > 75]))
        for ph in sorted(Q.phase.unique()):
            print(line(f"phase: {ph}", Q[Q.phase == ph]))
    print("   IMPLEMENTABLE version, all three leagues: the 80c bid is live only while the midpoint LB minutes ago was TH or more (first print below 80 while live)")
    for LB in (3, 5, 10):
        for TH in (82, 84, 86, 88):
            w, hh = [], []
            for gid in F.g.values:
                a, b = first_of[gid]+1, end_of[gid]
                for k in a+np.where(d.low[a:b] < 80)[0]:
                    if d.mid[max(k-LB, first_of[gid])] >= TH:
                        w.append(d.payout[gid])
                        hh.append(d.half[gid])
                        break
            w, hh = np.array(w), np.array(hh)
            print(f"     LB {LB:2} TH {TH}: fills {len(w):3} win {100*w.mean():5.1f}% net {100*w.mean()-80-fee(80, 1.75):+6.2f}c ({(100*w.mean()-80-fee(80, 1.75))/.8:+5.1f}%)"
                  f" | half1 n={int((hh == 0).sum()):3} win {100*w[hh == 0].mean():5.1f}% | half2 n={int((hh == 1).sum()):3} win {100*w[hh == 1].mean():5.1f}%")


# ---------------------------------------------------------------- trade record

SAMPLE = (("cbb-men", 120), ("cbb-women", 80), ("nba", 50))
last_request = [0.0]
used = Counter()


def read(client, path, params):
    """One paced public read into the shared cache: PACE seconds apart, 30s (then 60, 120, 240) after HTTP 429 or a server error."""
    file = cache_file(path, params)
    if file.exists():
        used["cache"] += 1
        return json.loads(file.read_text())
    for pause in (30, 60, 120, 240):
        wait = last_request[0]+PACE-time.time()
        if wait > 0:
            time.sleep(wait)
        last_request[0] = time.time()
        used["network"] += 1
        try:
            resp = client.get(config.KALSHI_PROD_BASE+path, params=params)
        except httpx.RequestError:
            resp = None
        last_request[0] = time.time()
        if resp is not None and resp.status_code == 200:
            data = resp.json()
            temp = file.with_suffix(".tmp")
            temp.write_text(json.dumps(data))
            temp.replace(file)
            return data
        if resp is not None and resp.status_code != 429 and resp.status_code < 500:
            raise RuntimeError(f"{path} returned HTTP {resp.status_code}")
        used["throttled"] += 1
        print(f"   {time.strftime('%H:%M:%S')} throttled ({resp.status_code if resp is not None else 'network error'}): pausing {pause}s", flush=True)
        time.sleep(pause)
    raise RuntimeError(f"Read failed after long back-off: {path}")


def trade_sample():
    """The first 10c-in-5-minutes event of a fixed random sample of games, leagues interleaved so a time cap cuts evenly."""
    d = Data()
    E = pd.read_pickle(OUT/"events.pkl")
    E = E[(E.M == 5) & (E.X == 10)].sort_values("i").groupby("g", as_index=False).first()
    E["T"] = d.M.t0.values[E.g.values]+60*(E.i.values-d.first[E.i.values])
    E["ticker"], E["yes"] = d.M.ticker.values[E.g.values], (d.M.p0.values > 50)[E.g.values]
    rng = np.random.default_rng(11)
    picks = []
    for league, n in SAMPLE:
        q = E[E.league == league]
        picks.append(q.iloc[rng.permutation(len(q))[:n]])
    order = []
    for k in range(max(n for _, n in SAMPLE)):
        for p in picks:
            j = k*len(p)/max(n for _, n in SAMPLE)
            if int(j) < len(p) and (k == 0 or int(j) != int((k-1)*len(p)/max(n for _, n in SAMPLE))):
                order.append(p.iloc[int(j)])
    return pd.DataFrame(order), d


def tape(client, ev, cutoff, deadline, pages=8):
    """Trades from a minute before the event bar's close to 12 minutes after (7 for the NBA): list, or None if incomplete."""
    span = 420 if ev.league == "nba" else 720
    path = "/historical/trades" if ev["T"] < cutoff else "/markets/trades"
    base, out, cursor = dict(ticker=ev.ticker, min_ts=int(ev["T"])-60, max_ts=int(ev["T"])+span, limit=1000), [], ""
    for _ in range(pages):
        params = dict(base, cursor=cursor) if cursor else base
        if not cache_file(path, params).exists() and (client is None or time.time() > deadline):
            return None
        try:
            data = read(client, path, params)
        except RuntimeError as exc:
            print("   read failed", ev.ticker, exc, flush=True)
            return None
        out += data.get("trades") or []
        cursor = data.get("cursor")
        if not cursor or not data.get("trades"):
            return out
    return None


def cutoff_ts():
    file = cache_file("/historical/cutoff", {})
    return timestamp(json.loads(file.read_text())["trades_created_ts"]) if file.exists() else 1789689600.0


def fetch_trades(minutes):
    S, _ = trade_sample()
    deadline, done, short = time.time()+60*minutes, Counter(), 0
    with httpx.Client(timeout=40) as client:
        for _, ev in S.iterrows():
            if time.time() > deadline:
                break
            got = tape(client, ev, cutoff_ts(), deadline)
            done[ev.league] += got is not None
            short += got is None
    print(f"FETCH complete tapes {dict(done)}, incomplete {short}, requests {dict(used)}")


def trades():
    S, d = trade_sample()
    rows = []
    for _, ev in S.iterrows():
        raw = tape(None, ev, cutoff_ts(), 0)
        if raw is None:
            continue
        faller_yes = ev.yes == (ev["dir"] == "fav fell")
        recs = []
        for x in raw:
            try:
                price = round(100*float(x["yes_price_dollars"])) if "yes_price_dollars" in x else round(float(x["yes_price"]))
                size = float(x.get("count_fp", x.get("count", 0)))
                recs.append((timestamp(x["created_time"]), price if faller_yes else 100-price, size,
                             ((x.get("taker_side") or x.get("taker_outcome_side")) == "yes") == faller_yes))
            except (KeyError, TypeError, ValueError):
                continue
        recs.sort()
        t, pr, size, buy = (np.array([r[k] for r in recs]) for k in range(4)) if recs else (np.array([]),)*4
        buy = buy.astype(bool) if len(buy) else np.zeros(0, bool)
        T, i = ev["T"], int(ev.i)
        L = d.bid[i] if ev["dir"] == "fav fell" else 100-d.ask[i]                # the faller's closing bid at the event bar
        out = dict(g=ev.g, league=ev.league, half=ev.half, prints=len(recs), bar_ask=ev.fade_ask, L=L)
        for k in (1, 3, 5, 10):
            out[f"bar_take{k}"], out[f"bar_reb{k}"] = ev[f"fade_take{k}"], ev[f"reb{k}_skip"]
        w1 = (t > T) & (t <= T+60)
        out.update(all_1m=size[w1].sum(), prints_1m=int(w1.sum()), bought_1m=size[w1 & buy].sum(), sold_below_bid_1m=size[w1 & ~buy & (pr < L)].sum(),
                   sold_at_or_below_1m=size[w1 & ~buy & (pr <= L)].sum(), sold_below_bid_5m=size[(t > T) & (t <= T+300) & ~buy & (pr < L)].sum())
        ent = np.where(buy & (t > T+1) & (t <= T+60))[0]
        last_bid = np.where(~buy & (t <= T) & (t > T-60))[0]
        first_bid = np.where(~buy & (t > T+60) & (t <= T+150))[0]                 # bid-side print about a bar after the event
        if len(ent):
            out.update(paid=pr[ent[0]], entry_delay=t[ent[0]]-T, entry_size=size[ent[0]])
        for k in (1, 3, 5, 10):
            if ev.league == "nba" and k == 10:
                continue
            ex = np.where(~buy & (t > T+60*k+60) & (t <= T+60*k+150))[0]
            if len(ex) and len(ent):
                b = pr[ex[0]]
                out[f"print_take{k}"] = b-fee(b)-pr[ent[0]]-fee(pr[ent[0]])
            if len(ex) and len(first_bid):
                out[f"bid_reb{k}_skip"] = pr[ex[0]]-pr[first_bid[0]]            # same-side prints, from a bar after the event
            ex0 = np.where(~buy & (t > T+60*k) & (t <= T+60*k+90))[0]
            if len(ex0) and len(last_bid):
                out[f"bid_reb{k}"] = pr[ex0[0]]-pr[last_bid[-1]]
        rows.append(out)
    X = pd.DataFrame(rows)
    X.to_pickle(OUT/"trade-events.pkl")
    print(f"\nTRADE RECORD: {len(X)} sampled games with a complete tape around their first 10c-in-5-minutes fall ({dict(X.league.value_counts())})")
    print("  FADE on prints: pay the first buyer-initiated print in the next minute; sell at the first seller-initiated print k+1 minutes after the event bar")
    for league in list(LEAGUES)+["all"]:
        q = X if league == "all" else X[X.league == league]
        if len(q) < 10:
            continue
        both = q.dropna(subset=["paid", "bar_ask"])
        print(f"   {league:9} events {len(q):4} | prints in the next minute: median {q.prints_1m.median():.0f}, contracts {q.all_1m.median():.0f} (bought by takers {q.bought_1m.median():.0f}); "
              f"a buy printed within 60s: {100*q.paid.notna().mean():3.0f}% | paid minus the bar's opening ask {stat(both.paid-both.bar_ask, both.g)}")
        for k in (1, 3, 5, 10):
            c = f"print_take{k}"
            if c not in q or q[c].notna().sum() < 10:
                continue
            s = q.dropna(subset=[c, f"bar_take{k}"])
            hybrid = q[c].fillna(q[f"bar_take{k}"])
            none = q[q[c].isna()]
            print(f"      {k:2}m round trip: prints {stat(s[c], s.g)} vs bars on the same events {stat(s[f'bar_take{k}'], s.g)} (n={len(s)}) | every event, prints where they exist else bars "
                  f"{stat(hybrid, q.g)} | bars on all {stat(q[f'bar_take{k}'], q.g)} | bars on events with no print exit {stat(none[f'bar_take{k}'], none.g)} (n={len(none)})")
        line = []
        for k in (1, 3, 5, 10):
            a, b = f"bid_reb{k}", f"bid_reb{k}_skip"
            if a in q and q[a].notna().sum() >= 10:
                line.append(f"{k}m {stat(q[a], q.g)} skip {stat(q[b], q.g) if b in q else 'n/a'} (bars skip, same events: {stat(q[f'bar_reb{k}'][q[a].notna()], q.g[q[a].notna()])})")
        print("      rebound on SELLER-initiated prints only (one side of the book, no bounce; survivors only): " + " | ".join(line))
        print(f"      maker capacity in the next minute: contracts sold strictly below the event bar's closing bid: median {q.sold_below_bid_1m.median():.0f} "
              f"(p25 {q.sold_below_bid_1m.quantile(.25):.0f}, zero in {100*(q.sold_below_bid_1m == 0).mean():.0f}%); at or below {q.sold_at_or_below_1m.median():.0f}; within five minutes below {q.sold_below_bid_5m.median():.0f}")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("steps", nargs="+", choices=("build", "martingale", "events", "special", "fetch", "trades"))
    parser.add_argument("--minutes", type=float, default=30)
    args = parser.parse_args()
    for step in args.steps:
        if step == "build":
            build()
        elif step == "martingale":
            martingale()
        elif step == "events":
            events()
        elif step == "special":
            special()
        elif step == "fetch":
            fetch_trades(args.minutes)
        elif step == "trades":
            trades()


if __name__ == "__main__":
    main()
