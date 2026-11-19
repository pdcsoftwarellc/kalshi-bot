"""The favourite-dip bid as a SURFACE: pre-game price band x bid depth, by family. Does any cell below 90c pass?

The traded cell: a favourite priced 90-99c before play gets a resting 80c bid
once the game is live, held to settlement. This script scores every cell of

    pre-game favourite price P:  60-70, 70-80, 80-85, 85-90, 90-95, 95-99c
    bid:                         floor(P - D) for D = 5, 10, 15, 20, 25, 30c (each game's own P),
                                 and the absolute bids 80, 70, 60c (only where the bid is 5c or more below P)

for men's college basketball, women's college basketball, tennis first-round
qualifying, tennis all other rounds and the NBA, from bars already on disk.

Orders and conventions (those of the earlier studies; nothing is fetched by `build` or `report`)
  men     KXNCAAMBGAME. Every game with one-minute bars in the shared cache (research.basketball_reversion.Bars finds
          per-market and batch reads). Start = favourite_contexts.start_of: this season Kalshi's expected expiration
          minus three hours; last season the close minus 130 minutes, an ESTIMATE known to about 20 minutes. Price =
          median midpoint of the last 10-15 one-minute quotes ending two minutes before the start (25 minutes before an
          estimated start), each at most 4c wide, all within 3c (the live rule's "at tip" price).
  men*    the same games with the start taken from ESPN's first play where basketball_reversion linked one (price from
          the quotes ending two minutes before it). A sensitivity check only; not counted in the multiplicity.
  women   KXNCAAWBGAME, research/wcbb_backfill.py rows, view "tip" (its start: close minus 130 minutes last season).
  nba     KXNBAGAME, the games of data/processed/basketball-reversion/games.pkl with its start (ESPN first play, else
          the schedule) and the same "at tip" price with a two-minute margin.
  tennis  the matches of research/tennis_contexts.py with a detected start (favourite_gap.detect_start) and a quote
          at most 4c wide on the last bar before it (the price; favourite_basket's convention), bars from
          data/processed/dip-scalp/bars.npz. "FRQ" = first-round qualifying, "other" = every other round.
  A bid fills only when a one-minute bar ending more than a minute after the start has a print STRICTLY below it
  (paper/rules.py `prints`). Net per fill = 100 x settlement - bid; maker fee taken as zero (ATP/WTA main tour, men's
  college basketball and the NBA may charge 0.0175*p*(1-p), at most 0.44c). Prices of 99c and above are excluded.

Statistics
  * Null: a fill at price b wins with probability b if the in-play price is a fair martingale, so the test of a cell
    is mean(100 x win - bid) against zero. Interval: bootstrap over games (one fill per game per cell), 4,000 draws.
  * Halves: the orders of a family x band sorted by settlement and split at the median.
  * PASS = at least MIN_FILLS fills, interval above zero, and positive in both halves.
  * Multiplicity: Bonferroni and Benjamini-Hochberg on one-sided z-tests, and a fair-market replay: every game is
    redrawn as a continuous martingale from its own P (win with probability P; a winner's lowest price M has
    P(M < b) = b(100-P) / (P(100-b)); a loser fills every bid), which keeps the dependence between cells that share
    games. The replay says how many cells pass by chance.
  * Capacity: contracts in in-play minutes wholly below the bid (a lower bound on what printed through it) and in
    minutes with any print below it (an upper bound); the trade record was not read for the new cells.
  * Fills per 30 days: college basketball = 1-18 November 2026 scaled to 30 days (and last season's December-February
    divided by the share of listed games with bars on disk); tennis = mean of May-October 2026; NBA = November 2025 to
    March 2026 divided by the share of games on disk.

    .venv/bin/python research/dip_surface.py build      (offline: caches -> data/processed/dip-surface/orders.pkl)
    .venv/bin/python research/dip_surface.py report

Public market data only; no orders, no authentication. Output: data/processed/dip-surface/.
"""
import argparse
from collections import Counter, defaultdict
from datetime import datetime, timedelta
import io
import json
import math
from pathlib import Path
import pickle
import sys
import time
from zoneinfo import ZoneInfo

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from research.api_cost import ROOT

RAW = ROOT / "data/raw/dip-surface"
OUT = ROOT / "data/processed/dip-surface"
EASTERN = ZoneInfo("America/New_York")
BANDS = ((60, 70), (70, 80), (80, 85), (85, 90), (90, 95), (95, 99))
DEPTHS = (5, 10, 15, 20, 25, 30)
ABSOLUTE = (80, 70, 60)
LEVELS = [("d", d) for d in DEPTHS]+[("a", a) for a in ABSOLUTE]
FAMILIES = (("men", "men's college basketball"), ("women", "women's college basketball"), ("frq", "tennis first-round qualifying"),
            ("tennis-other", "tennis, all other rounds"), ("nba", "NBA"))
CHECK = (("men*", "men's college basketball, ESPN start (sensitivity)"),)
MIN_FILLS, DRAWS, SIMS = 30, 4000, 1000
RNG = np.random.default_rng(11)
buffer = io.StringIO()


def say(text=""):
    print(text, flush=True)
    buffer.write(text+"\n")


# ---------------------------------------------------------------- build

def tip_price(bars, start, margin):
    """(favourite's price, yes) from the last 10-15 quotes ending `margin` seconds before the start, or None."""
    from research.calibration_map import quote
    late = [quote(b) for b in bars if b["end_period_ts"] <= start-margin][-15:]
    if len(late) < 10 or not all(q and q[1]-q[0] <= 4 for q in late):
        return None
    mids = [(q[0]+q[1])/2 for q in late]
    if max(mids)-min(mids) > 3:
        return None
    mid = float(np.median(mids))
    return (mid if mid > 50 else 100-mid), mid > 50


def in_play(bars, yes, start):
    """[end of minute, favourite low, favourite high, contracts] for bars with a trade ending more than a minute after the start."""
    from research.calibration_map import volume
    from research.near_end import cents
    out = []
    for b in bars:
        lo, hi = cents(b, "price", "low"), cents(b, "price", "high")
        if lo is None or hi is None or b["end_period_ts"] <= start+60:
            continue
        out.append((b["end_period_ts"], lo if yes else 100-hi, hi if yes else 100-lo, volume(b)))
    return np.array(out, float).reshape(-1, 4)


def order(fam, event, ticker, settle, start, price, payout, plays, **extra):
    day = (datetime.fromtimestamp(start, EASTERN)-timedelta(hours=5)).strftime("%Y-%m-%d")
    return dict(fam=fam, event=event, ticker=ticker, settle=settle, start=start, price=float(price), payout=float(payout), plays=plays,
                low=float(plays[:, 1].min()) if len(plays) else 999.0, day=day, month=day[:7], **extra)


def build_basketball(meta):
    import pandas as pd
    from research import basketball_reversion as br
    from research import favourite_contexts as fc
    store = br.Bars()
    games = pd.read_pickle(br.OUT/"games.pkl")
    espn = {g.event: g for g in games.itertuples()}
    out = []
    for league, fam in (("cbb-men", "men"), ("nba", "nba")):
        series, cover = br.LEAGUES[league], Counter()
        for e in sorted(br.events_of(league).values(), key=lambda x: (x[0], x[2])):
            settle, _, event, archived, ms = e
            month = datetime.fromtimestamp(settle, EASTERN).strftime("%Y-%m")
            meta[fam, "listed", month] += 1
            if len(ms) < 2 or not all(m.get("_outcome") in (0.0, 1.0) for m in ms):
                continue
            g = espn.get(event)
            if fam == "nba":
                starts = [("nba", float(g.start), 120)] if g is not None else []
            else:
                start = fc.start_of("cbb-men", e)
                starts = [("men", start, 25*60 if archived else 120)] if start else []
                if g is not None and g.start_kind == "plays":
                    starts.append(("men*", float(g.start), 120))
            if not starts:
                continue
            bars = store.get(series, e)
            if not bars:
                cover["no bars on disk"] += 1
                continue
            meta[fam, "read", month] += 1
            m = ms[0]
            for name, start, margin in starts:
                if not 1800 <= settle-start <= 7*3600:
                    cover[name, "no usable start"] += 1
                    continue
                tip = tip_price(bars, start, margin)
                if tip is None:
                    cover[name, "no clean quote at the tip"] += 1
                    continue
                price, yes = tip
                if price >= 99:
                    continue
                out.append(order(name, event, m["ticker"], settle, start, price, m["_outcome"] if yes else 1-m["_outcome"],
                                 in_play(bars, yes, start), archived=bool(archived), yes=yes))
                cover[name, "orders"] += 1
        print(league, dict(cover), flush=True)
    return out


def build_women(meta):
    from research import wcbb_backfill as w
    rows = w.load()
    for r in rows:
        meta["women", "read", datetime.fromtimestamp(r["settle"], EASTERN).strftime("%Y-%m")] += 1
    for archived in (True, False):
        for e in w.games(archived):
            meta["women", "listed", datetime.fromtimestamp(e[0], EASTERN).strftime("%Y-%m")] += 1
    out = []
    for v in w.view(rows, "tip"):
        plays = np.array([x[:4] for x in v["plays"] if x[0] > v["est_start"]+60], float).reshape(-1, 4)
        out.append(order("women", v["event"], v["ticker"], v["settle"], v["est_start"], v["price"], v["payout"], plays, archived=v["archived"], yes=v["yes"]))
    print("women", len(out), flush=True)
    return out


def build_tennis(meta):
    import pandas as pd
    from research import tennis_contexts
    matches = pd.read_pickle(ROOT/"data/processed/dip-scalp/matches.pkl")
    z = np.load(ROOT/"data/processed/dip-scalp/bars.npz")
    data, off = z["data"], z["offsets"]
    where = {event: i for i, event in enumerate(matches.event.values)}
    out, cover = [], Counter()
    for r in tennis_contexts.prepare():
        if not r.get("stable") or r.get("start_ts") is None or r.get("ask_before_play") is None or r.get("bid_before_play") is None:
            continue
        bid, ask = r["bid_before_play"], r["ask_before_play"]
        if not 0 < bid <= ask < 100 or ask-bid > 4 or r["pay_fav"] is None:
            continue
        price = (bid+ask)/2
        if price >= 99:
            continue
        i = where.get(r["event"])
        if i is None or matches.start_ts.values[i] != r["start_ts"] or matches.ticker.values[i] != r["ticker"]:
            cover["no bars in dip-scalp/bars.npz or a different start"] += 1
            continue
        x = data[:, off[i]:off[i+1]].astype(float)
        keep = (x[0] > 60) & ~np.isnan(x[3]) & ~np.isnan(x[4])
        t, lo, hi, vol = x[0][keep]+r["start_ts"], x[3][keep], x[4][keep], np.nan_to_num(x[5][keep])
        if not r["yes"]:
            lo, hi = 100-hi, 100-lo
        fam = "frq" if r["qual"] and r["stage"] in ("qual Q1", "qual Q") else "tennis-other"
        out.append(order(fam, r["event"], r["ticker"], r["settle"], r["start_ts"], price, r["pay_fav"], np.stack([t, lo, hi, vol], axis=1),
                         archived=r["archived"], yes=r["yes"], tour=r["tour"], stage=r["stage"]))
        cover[fam] += 1
    print("tennis", dict(cover), flush=True)
    return out


def build():
    meta = Counter()
    orders = build_basketball(meta)+build_women(meta)+build_tennis(meta)
    orders.sort(key=lambda o: (o["settle"], o["event"]))
    OUT.mkdir(parents=True, exist_ok=True)
    with (OUT/"orders.pkl").open("wb") as f:
        pickle.dump(dict(orders=orders, meta={"|".join(k): v for k, v in meta.items()}), f)
    print(f"{len(orders)} orders: {dict(Counter(o['fam'] for o in orders))}")


def load():
    with (OUT/"orders.pkl").open("rb") as f:
        d = pickle.load(f)
    return d["orders"], {tuple(k.split("|")): v for k, v in d["meta"].items()}


# ---------------------------------------------------------------- cells

def bid_of(o, level):
    kind, x = level
    if kind == "d":
        bid = math.floor(o["price"]-x)
    else:
        bid = x if x <= o["price"]-5 else None
    return bid if bid is not None and bid >= 5 else None


def name_of(level):
    return f"P-{level[1]}" if level[0] == "d" else f"{level[1]}c"


def placed(part, level):
    """[(order, bid)] for the orders on which this bid exists."""
    return [(o, b) for o in part for b in [bid_of(o, level)] if b is not None]


def filled(pairs):
    return [(o, b) for o, b in pairs if o["low"] < b]


def boot(nets):
    nets = np.asarray(nets, float)
    if len(nets) < 10:
        return None
    means = nets[RNG.integers(0, len(nets), (DRAWS, len(nets)))].mean(axis=1)
    return [float(v) for v in np.percentile(means, [2.5, 97.5])]


def halves(part):
    part = sorted(part, key=lambda o: (o["settle"], o["event"]))
    return part[:len(part)//2], part[len(part)//2:]


def stat(pairs):
    """Fills, win rate, mean bid, net c per fill, % return, z against zero."""
    hit = filled(pairs)
    out = dict(orders=len(pairs), fills=len(hit))
    if not hit:
        return out
    nets = np.array([100*o["payout"]-b for o, b in hit])
    bids = np.array([b for _, b in hit], float)
    se = nets.std(ddof=1)/len(nets)**.5 if len(nets) > 1 else float("nan")
    # the standard error a fair market implies (bid x (100 - bid)) guards cells without a loss
    se = max(se, float(np.sqrt((bids*(100-bids)).mean()/len(nets)))) if len(nets) > 1 else se
    out.update(win=100*float(np.mean([o["payout"] for o, _ in hit])), need=float(bids.mean()), net=float(nets.mean()), ret=100*float(nets.mean()/bids.mean()),
               z=float(nets.mean()/se) if se and se > 0 else float("nan"), nets=nets)
    return out


def through(o, bid):
    """Contracts in in-play minutes wholly below the bid (lower bound) and in minutes with a print below it (upper bound)."""
    p = o["plays"]
    return float(p[p[:, 2] < bid, 3].sum()), float(p[p[:, 1] < bid, 3].sum())


def frequency(fam, part, meta):
    """Callable: fills (a list of orders) -> fills per 30 days in the family's reference window, and the window's label."""
    def share(months):
        listed, read = sum(meta.get((fam, "listed", m), 0) for m in months), sum(meta.get((fam, "read", m), 0) for m in months)
        return read/listed if listed else 1.0
    if fam in ("men", "men*", "women"):
        key = "men" if fam == "men*" else fam
        days = sorted({o["day"] for o in part if o["month"] == "2026-11"})
        span = (datetime.fromisoformat(days[-1])-datetime.fromisoformat(days[0])).days+1 if days else 1
        listed, read = meta.get((key, "listed", "2026-11"), 0), meta.get((key, "read", "2026-11"), 0)
        cover = read/listed if listed else 1.0
        return (lambda hit: sum(o["month"] == "2026-11" for o in hit)*30/span/cover), f"November 2026 ({span} days, {cover:.0%} of listed games on disk)"
    if fam == "nba":
        months = ("2025-11", "2025-12", "2026-01", "2026-02", "2026-03")
        cover = share(months)
        return (lambda hit: sum(o["month"] in months for o in hit)/cover*30/151), f"November 2025-March 2026 ({cover:.0%} of listed games on disk)"
    months = ("2026-05", "2026-06", "2026-07", "2026-08", "2026-09", "2026-10")
    return (lambda hit: sum(o["month"] in months for o in hit)*30/184), "May-October 2026 (matches on disk, no coverage correction)"


def last_season(fam, hit, meta):
    """College basketball: fills per 30 days in December 2025-February 2026, divided by the share of listed games on disk."""
    months = ("2025-12", "2026-01", "2026-02")
    key = "men" if fam == "men*" else fam
    listed, read = sum(meta.get((key, "listed", m), 0) for m in months), sum(meta.get((key, "read", m), 0) for m in months)
    return sum(o["month"] in months for o in hit)/(read/listed if listed else 1)*30/90


def cell(fam, band, level, part, freq):
    pairs = placed(part, level)
    s = stat(pairs)
    s.update(fam=fam, band=band, level=level, pairs=pairs)
    if not s["fills"]:
        return s
    a, b = halves(part)
    s["h1"], s["h2"] = stat(placed(a, level)), stat(placed(b, level))
    s["ci"] = boot(s["nets"])
    hit = [o for o, _ in filled(pairs)]
    s["per30"] = freq(hit)
    s["pass"] = bool(s["fills"] >= MIN_FILLS and s["ci"] and s["ci"][0] > 0 and s["h1"].get("net", -1) > 0 and s["h2"].get("net", -1) > 0)
    return s


def fmt(s):
    if not s.get("fills"):
        return f'{name_of(s["level"]):>5} games {s["orders"]:5} fills    0'
    ci = "[   n<10    ]" if not s["ci"] else f'[{s["ci"][0]:+5.1f},{s["ci"][1]:+5.1f}]'
    h = lambda x: f'{x["net"]:+5.1f}c({x["fills"]:3})' if x.get("fills") else "   n/a    "
    small = " too small" if s["fills"] < MIN_FILLS else ""
    return (f'{name_of(s["level"]):>5} games {s["orders"]:5} fills {s["fills"]:4} ({s["fills"]/s["orders"]:4.0%}) win {s["win"]:5.1f}% need {s["need"]:4.1f}% '
            f'net {s["net"]:+5.1f}c = {s["ret"]:+5.1f}% {ci} halves {h(s["h1"])} / {h(s["h2"])} per30d {s["per30"]:6.1f}'
            f'{"  PASS" if s["pass"] else ""}{small}')


# ---------------------------------------------------------------- fair-market replay

def replay(cells, orders):
    """Number of cells passing (normal interval above zero, both halves positive, MIN_FILLS fills) when every game is a
    fair martingale from its own price: per simulation, [all cells, cells below 90c, cells at 90c+]."""
    index = {id(o): i for i, o in enumerate(orders)}
    price = np.array([o["price"] for o in orders])
    prepared = []
    for c in cells:
        if not c["pairs"]:
            continue
        idx = np.array([index[id(o)] for o, _ in c["pairs"]])
        bids = np.array([b for _, b in c["pairs"]], float)
        rank = np.argsort(np.argsort([(o["settle"], o["event"]) for o, _ in c["pairs"]], axis=0)[:, 0]) if False else None
        order_ = sorted(range(len(idx)), key=lambda k: (c["pairs"][k][0]["settle"], c["pairs"][k][0]["event"]))
        first = np.zeros(len(idx), bool)
        first[order_[:len(idx)//2]] = True
        prepared.append((idx, bids, first, c["band"][0] < 90))
    counts = np.zeros((SIMS, 3), int)
    for s in range(SIMS):
        win = RNG.random(len(orders)) < price/100
        u = RNG.random(len(orders))
        low = np.where(win, 100*u*price/((100-price)+u*price), 0.0)      # a winner's lowest price; a loser reaches zero
        for idx, bids, first, below in prepared:
            hit = low[idx] < bids
            if hit.sum() < MIN_FILLS:
                continue
            nets = 100*win[idx][hit]-bids[hit]
            f = first[hit]
            if f.sum() == 0 or f.sum() == len(f):
                continue
            se = max(nets.std(ddof=1), 1e-9)/len(nets)**.5
            if nets.mean()-1.96*se > 0 and nets[f].mean() > 0 and nets[~f].mean() > 0:
                counts[s, 0] += 1
                counts[s, 1 if below else 2] += 1
    return counts


def p_one_sided(z):
    return 0.5*math.erfc(z/math.sqrt(2)) if z == z else 1.0


# ---------------------------------------------------------------- report

def report():
    orders, meta = load()
    names = dict(FAMILIES+CHECK)
    cells, by_fam = [], {}
    say("=== ORDERS ON DISK (favourites with a clean pre-game price under 99c)")
    for fam, name in FAMILIES+CHECK:
        part = [o for o in orders if o["fam"] == fam]
        by_fam[fam] = part
        text = " ".join(f"{lo}-{hi}: {sum(lo <= o['price'] < hi for o in part):5}" for lo, hi in BANDS)
        say(f"  {name:52} {len(part):6} | {text} | {min(o['day'] for o in part)} to {max(o['day'] for o in part)}")
    say("  share of listed games with one-minute bars on disk, by month of settlement:")
    for fam in ("men", "women", "nba"):
        months = sorted({k[2] for k in meta if k[0] == fam})
        say(f"    {fam:6} "+" ".join(f"{m}: {meta.get((fam, 'read', m), 0)}/{meta.get((fam, 'listed', m), 0)}" for m in months))
    say("  tennis matches by month: "+" ".join(f"{m}: {n}" for m, n in sorted(Counter(o["month"] for o in orders if o["fam"] in ("frq", "tennis-other")).items())))

    say("\n=== 0. THE TRADED CELL REBUILT (90-99c, 80c bid) against the earlier backtests (men 116 fills +9.9%, women 176 +9.4%, tennis FRQ 165 +13.5%)")
    for fam, name in FAMILIES+CHECK:
        part = [o for o in by_fam[fam] if 90 <= o["price"] < 99]
        freq, _ = frequency(fam, by_fam[fam], meta)
        s = cell(fam, (90, 99), ("a", 80), part, freq)
        say(f"  {name:52} {fmt(s)}")

    say("\n=== 1. THE SURFACE. One line per cell: games with the bid, fills (rate), win rate, win rate needed (mean bid), net per fill and")
    say("       % return on the bid, bootstrap 95% interval over games, halves (net, fills), fills per 30 days in the reference window.")
    for fam, name in FAMILIES+CHECK:
        freq, label = frequency(fam, by_fam[fam], meta)
        say(f"\n--- {name}   [fills per 30 days: {label}]")
        for band in BANDS:
            part = [o for o in by_fam[fam] if band[0] <= o["price"] < band[1]]
            wins = np.mean([o["payout"] for o in part]) if part else float("nan")
            gap = 100*wins-np.mean([o["price"] for o in part]) if part else float("nan")
            se = 100*np.std([o["payout"] for o in part])/max(len(part), 1)**.5 if part else float("nan")
            say(f"  favourites {band[0]}-{band[1]}c: games {len(part)}, pre-game gap (win rate minus price) {gap:+.1f}c (se {se:.1f})")
            for level in LEVELS:
                s = cell(fam, band, level, part, freq)
                if s["orders"]:
                    say("    "+fmt(s))
                    if fam != "men*":
                        cells.append(s)

    # ---- the question: overshoot in general, or heavy favourites only?
    say("\n=== 2. DIPS OVERSHOOT IN GENERAL, OR HEAVY FAVOURITES ONLY? Net per fill pooled over the families, by band and depth")
    say("       (win rate minus the bid; a fair market gives zero in every cell). Sets: college = men + women; tennis = both; all = five families.")
    sets = (("college basketball", ("men", "women")), ("tennis", ("frq", "tennis-other")), ("NBA", ("nba",)), ("all five", tuple(f for f, _ in FAMILIES)))
    pooled = {}
    for label, fams in sets:
        say(f"  {label}")
        for band in BANDS+((60, 90), (90, 99)):
            part = [o for f in fams for o in by_fam[f] if band[0] <= o["price"] < band[1]]
            text = f"    {band[0]}-{band[1]}c"
            for level in LEVELS:
                s = stat(placed(part, level))
                pooled[label, band, level] = s
                if s.get("fills", 0) >= 10:
                    ci = boot(s["nets"])
                    star = "*" if ci[0] > 0 else ("-" if ci[1] < 0 else " ")
                    text += f" | {name_of(level)} {s['net']:+5.1f}c{star}({s['fills']:4})"
                else:
                    text += f" | {name_of(level)}   n/a  ({s.get('fills', 0):4})"
            say(text)
    say("    * interval above zero, - interval below zero (unadjusted)")
    say("  Fill rate against the fair-market fill rate (100-P)/(100-bid) per game, depth P-10 and P-20:")
    for fam, name in FAMILIES:
        text = f"    {name:32}"
        for band in BANDS:
            part = [o for o in by_fam[fam] if band[0] <= o["price"] < band[1]]
            for level in (("d", 10), ("d", 20)):
                pairs = placed(part, level)
                if len(pairs) >= 20:
                    fair = np.mean([(100-o["price"])/(100-b) for o, b in pairs])
                    text += f" | {band[0]}-{band[1]} {name_of(level)} {len(filled(pairs))/len(pairs):4.0%} v {fair:4.0%}"
        say(text)

    # ---- multiplicity
    tested = [c for c in cells if c.get("fills", 0) >= MIN_FILLS]
    passing = [c for c in tested if c["pass"]]
    say(f"\n=== 3. MULTIPLICITY. Cells with a bid: {len(cells)}; with {MIN_FILLS}+ fills (tested): {len(tested)} ({sum(c['band'][0] < 90 for c in tested)} below 90c, "
        f"{sum(c['band'][0] >= 90 for c in tested)} at 90c+)")
    say(f"  PASS (interval above zero, both halves positive): {len(passing)} cells; below 90c: {sum(c['band'][0] < 90 for c in passing)}; at 90c+: {sum(c['band'][0] >= 90 for c in passing)}")
    for c in tested:
        c["p"] = p_one_sided(c["z"])
    ranked = sorted(tested, key=lambda c: c["p"])
    bonf = [c for c in tested if c["p"] < .05/len(tested)]
    bh = []
    for k, c in enumerate(ranked, 1):
        if c["p"] <= .05*k/len(tested):
            bh = ranked[:k]
    say(f"  one-sided z-test of net > 0: Bonferroni (p < 0.05/{len(tested)} = {.05/len(tested):.5f}) keeps {len(bonf)} cells, {sum(c['band'][0] < 90 for c in bonf)} below 90c; "
        f"Benjamini-Hochberg at 5% keeps {len(bh)}, {sum(c['band'][0] < 90 for c in bh)} below 90c")
    say(f"  uncorrected one-sided p < 0.025: {sum(c['p'] < .025 for c in tested)} cells (chance alone on independent cells: {.025*len(tested):.1f}); "
        f"below 90c: {sum(c['p'] < .025 and c['band'][0] < 90 for c in tested)} of {sum(c['band'][0] < 90 for c in tested)} (chance {.025*sum(c['band'][0] < 90 for c in tested):.1f}); "
        f"significantly NEGATIVE below 90c (p > 0.975): {sum(c['p'] > .975 and c['band'][0] < 90 for c in tested)}")
    for tag, group in (("Bonferroni", bonf), ("Benjamini-Hochberg only", [c for c in bh if c not in bonf])):
        for c in sorted(group, key=lambda c: (c["fam"], c["band"], LEVELS.index(c["level"]))):
            say(f"    {tag:24} {c['fam']:13} {c['band'][0]}-{c['band'][1]}c {fmt(c)}  p {c['p']:.5f}")
    counts = replay(cells, [o for o in orders if o["fam"] != "men*"])
    say(f"  fair-market replay ({SIMS} seasons, every game a martingale from its own price, same games and cells): cells passing per season "
        f"mean {counts[:, 0].mean():.1f} (95th pct {np.percentile(counts[:, 0], 95):.0f}, max {counts[:, 0].max()}); "
        f"below 90c mean {counts[:, 1].mean():.1f} (95th pct {np.percentile(counts[:, 1], 95):.0f}); at 90c+ mean {counts[:, 2].mean():.1f} (95th pct {np.percentile(counts[:, 2], 95):.0f})")
    say(f"  seasons with at least as many passes as observed: all {np.mean(counts[:, 0] >= len(passing)):.1%}; below 90c {np.mean(counts[:, 1] >= sum(c['band'][0] < 90 for c in passing)):.1%}; "
        f"at 90c+ {np.mean(counts[:, 2] >= sum(c['band'][0] >= 90 for c in passing)):.1%}")
    say("  cells that PASS:")
    for c in sorted(passing, key=lambda c: (c["fam"], c["band"], LEVELS.index(c["level"]))):
        say(f"    {c['fam']:13} {c['band'][0]}-{c['band'][1]}c {fmt(c)}  p {c['p']:.5f}")

    # ---- pick on the first half, confirm on the second
    say(f"\n=== 4. PICK ON THE FIRST HALF, CONFIRM ON THE SECOND (first-half interval above zero with {MIN_FILLS//2}+ fills; then the second half alone)")
    picks = []
    for c in cells:
        h1 = c.get("h1")
        if not h1 or h1.get("fills", 0) < MIN_FILLS//2:
            continue
        ci1 = boot(h1["nets"])
        if ci1 and ci1[0] > 0:
            h2 = c["h2"]
            ci2 = boot(h2["nets"]) if h2.get("fills") else None
            c["confirmed"] = bool(ci2 and ci2[0] > 0)
            picks.append((c, ci1, ci2))
    eligible = sum(1 for c in cells if c.get("h1") and c["h1"].get("fills", 0) >= MIN_FILLS//2)
    say(f"  cells with {MIN_FILLS//2}+ first-half fills: {eligible}; picked: {len(picks)} (chance alone about {.025*eligible:.1f}); below 90c picked: {sum(c['band'][0] < 90 for c, _, _ in picks)}")
    for c, ci1, ci2 in sorted(picks, key=lambda x: (x[0]["fam"], x[0]["band"], LEVELS.index(x[0]["level"]))):
        h2 = c["h2"]
        second = "no fills" if not h2.get("fills") else (f'{h2["fills"]:3} fills win {h2["win"]:5.1f}% need {h2["need"]:4.1f}% net {h2["net"]:+5.1f}c = {h2["ret"]:+5.1f}% '
                                                         + (f'[{ci2[0]:+5.1f},{ci2[1]:+5.1f}]' if ci2 else "[n<10]"))
        verdict = "CONFIRMED (interval above zero)" if c["confirmed"] else ("positive, interval includes zero" if h2.get("net", -1) > 0 else "NOT confirmed")
        say(f"    {c['fam']:13} {c['band'][0]}-{c['band'][1]}c {name_of(c['level']):>5} first half {c['h1']['fills']:3} fills net {c['h1']['net']:+5.1f}c [{ci1[0]:+5.1f},{ci1[1]:+5.1f}] "
            f"-> second half {second}: {verdict}")

    # ---- overlap, widened rule, capacity
    say("\n=== 5. WHAT THE PASSING CELLS ADD to the current rule (90-99c, 80c bid): new games, fills per 30 days, blended return, capacity")
    for fam, name in FAMILIES:
        freq, label = frequency(fam, by_fam[fam], meta)
        base_pairs = placed([o for o in by_fam[fam] if 90 <= o["price"] < 99], ("a", 80))
        base_hit = filled(base_pairs)
        base_events = {o["event"] for o, _ in base_hit}
        base_net = [100*o["payout"]-b for o, b in base_hit]
        if not base_hit:
            say(f"  {name}: the current rule has no fills on disk")
            continue
        lo_hi = np.array([through(o, b) for o, b in base_hit])
        text = (f"  {name}: current rule {len(base_hit)} fills, {np.mean(base_net):+.1f}c = {100*np.mean(base_net)/80:+.1f}%, {freq([o for o, _ in base_hit]):.1f} fills per 30 days"
                + (f" (last season December-February, corrected for coverage: {last_season(fam, [o for o, _ in base_hit], meta):.1f})" if fam in ("men", "women") else "")
                + f"; contracts through the bid, median {np.median(lo_hi[:, 0]):,.0f} (minutes wholly below) to {np.median(lo_hi[:, 1]):,.0f} (minutes with a print below)")
        say(text)
        mine = [c for c in cells if c["fam"] == fam and c.get("pass")]
        if not mine:
            say("     no other cell passes: no widening")
        for c in sorted(mine, key=lambda c: (c["band"], LEVELS.index(c["level"]))):
            hit = filled(c["pairs"])
            new = [(o, b) for o, b in hit if o["event"] not in base_events]
            cap = np.array([through(o, b) for o, b in hit])
            new_net = [100*o["payout"]-b for o, b in new]
            both_net = base_net+new_net
            both_bid = [80.0]*len(base_net)+[b for _, b in new]
            extra = (f"; last season December-February {last_season(fam, [o for o, _ in new], meta):.1f} new fills per 30 days" if fam in ("men", "women") else "")
            say(f"     + {c['band'][0]}-{c['band'][1]}c {name_of(c['level']):>5}: {len(hit)} fills, {len(new)} in games the current rule did not fill "
                f"({freq([o for o, _ in new]):.1f} new fills per 30 days{extra}); new fills alone "
                + (f"{np.mean(new_net):+.1f}c = {100*np.mean(new_net)/np.mean([b for _, b in new]):+.1f}%" if new else "n/a")
                + f"; current + new: {len(both_net)} fills, {np.mean(both_net):+.1f}c = {100*np.mean(both_net)/np.mean(both_bid):+.1f}% per fill"
                + f"; contracts through the bid, median {np.median(cap[:, 0]):,.0f} to {np.median(cap[:, 1]):,.0f}")

    summary = [dict(fam=c["fam"], band=list(c["band"]), level=name_of(c["level"]), orders=c["orders"], fills=c.get("fills", 0), win=c.get("win"), need=c.get("need"),
                    net=c.get("net"), ret=c.get("ret"), ci=c.get("ci"), h1=[c["h1"].get("fills", 0), c["h1"].get("net")] if c.get("h1") else None,
                    h2=[c["h2"].get("fills", 0), c["h2"].get("net")] if c.get("h2") else None, per30=c.get("per30"), passed=c.get("pass", False), p=c.get("p"))
               for c in cells]
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT/"cells.json").write_text(json.dumps(summary, indent=1))
    (OUT/"report.txt").write_text(buffer.getvalue())
    return cells


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("command", choices=("build", "report"))
    args = parser.parse_args()
    if args.command == "build":
        build()
    else:
        report()


if __name__ == "__main__":
    main()
