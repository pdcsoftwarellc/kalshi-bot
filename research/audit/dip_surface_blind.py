"""AUDIT: blind rebuild of the dip surface (pre-game favourite price band x bid depth, per family).

Written from the first 30 lines of research/dip_surface.py's docstring only (hypothesis + data sources), before its
code body or its outputs were read. Shares only the low-level bar readers with the repo; favourite, pre-game price,
fill, fee, halves and bootstrap are this file's own.

Rules of evidence
  favourite      the side whose pre-game midpoint is above 50c (the canonical market's YES side or its mirror).
  pre-game price basketball: median midpoint of the quotes (spread <= 5c, at least 5 of them, range <= 5c) on bars ending
                 in the 30 minutes before (start - margin); margin 2 minutes when the start is known (this season: expected
                 expiration - 3h; NBA: games.pkl start), 25 minutes when it is estimated (archived college: close - 130m).
                 tennis: midpoint on the last bar before the detected start (tennis-contexts rows), spread <= 4c.
  bid            floor(P) - D for D = 5, 10, 15, 20; fixed 80 / 75 / 70 when at least 5c below P.
  fill           a bar ending more than 60s after the start whose low print on the favourite's side is STRICTLY below
                 the bid. One fill per game per cell. Held to settlement.
  fee            maker fee 1.75*p*(1-p) cents on series whose fee_type is quadratic_with_maker_fees (men's college
                 basketball, NBA, ATP, WTA; checked on the public /series endpoint 2026-11-19), else zero.
  return         (100*payout - bid - fee) / (bid + fee), in % of stake, per fill.
  interval       95% bootstrap over games (4,000 draws). Halves: fills split at the median start time.

Run: .venv/bin/python research/audit/dip_surface_blind.py        (no network)
"""
from collections import defaultdict
import json
from pathlib import Path
import pickle
import sys

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
OUT = ROOT/"data/processed/audit/dip_surface"
MAKER = {"cbb-men": 1.75, "cbb-women": 0.0, "nba": 1.75, "ATP": 1.75, "WTA": 1.75, "Challenger": 0.0, "WTA Challenger": 0.0,
         "ITF men": 0.0, "ITF women": 0.0}
BANDS = ((60, 70), (70, 80), (80, 85), (85, 90), (90, 95), (95, 99), (80, 90), (90, 99))
BIDS = ("P-5", "P-10", "P-15", "P-20", "80", "75", "70")
RNG = np.random.default_rng(1119)


def basketball():
    """One record per game: league, event, start, archived, P (favourite), yes?, payout, and favourite-side in-play lows."""
    cache = OUT/"blind-basketball.pkl"
    if cache.exists():
        return pickle.loads(cache.read_bytes())
    from research import basketball_reversion as br
    from research import favourite_contexts as fc
    from research.calibration_map import quote
    from research.near_end import cents
    nba_start = {}
    g = pd.read_pickle(ROOT/"data/processed/basketball-reversion/games.pkl")
    for r in g[g.league == "nba"].itertuples():
        nba_start[r.event] = float(r.start)
    bars_of, out, skip = br.Bars(), [], defaultdict(int)
    for league, series in br.LEAGUES.items():
        for event, e in br.events_of(league).items():
            settle, _, _, archived, ms = e
            m = ms[0]
            if m["_outcome"] not in (0.0, 1.0):
                skip[league, "not a 0/1 settlement"] += 1
                continue
            if league == "nba":
                start, estimated = nba_start.get(event), False
            else:
                start, estimated = fc.start_of(league, e), bool(archived)
            if not start:
                skip[league, "no start"] += 1
                continue
            bars = bars_of.get(series, e)
            if not bars:
                skip[league, "no bars"] += 1
                continue
            edge = start-(1500 if estimated else 120)
            qs = [q for q in (quote(b) for b in bars if edge-1800 < b["end_period_ts"] <= edge) if q and q[1]-q[0] <= 5]
            if len(qs) < 5:
                skip[league, "fewer than 5 tight pre-game quotes"] += 1
                continue
            mids = [(a+b)/2 for a, b in qs]
            if max(mids)-min(mids) > 5:
                skip[league, "pre-game price unstable"] += 1
                continue
            p = float(np.median(mids))
            yes = p > 50
            lows = []
            for b in bars:
                if b["end_period_ts"] <= start+60:
                    continue
                low = cents(b, "price", "low") if yes else cents(b, "price", "high")
                if low is not None:
                    lows.append(low if yes else 100-low)
            out.append(dict(family=league, event=event, start=float(start), estimated=estimated, p=p if yes else 100-p,
                            payout=m["_outcome"] if yes else 1-m["_outcome"], low=min(lows) if lows else 999, fee=MAKER[league]))
    for k, v in sorted(skip.items()):
        print("  skipped", k, v)
    cache.write_bytes(pickle.dumps(out))
    return out


def tennis():
    m = pd.read_pickle(ROOT/"data/processed/dip-scalp/matches.pkl")
    z = np.load(ROOT/"data/processed/dip-scalp/bars.npz")
    data, off = z["data"], z["offsets"]
    ctx = {r["event"]: r for r in json.loads((ROOT/"data/processed/tennis-contexts/rows.json").read_text()) if r.get("detected")}
    out, skip = [], defaultdict(int)
    for i, r in enumerate(m.itertuples()):
        c = ctx.get(r.event)
        if c is None or c.get("price_before_play") is None or c["ticker"] != r.ticker or abs(c["start_ts"]-r.start_ts) > 1:
            skip["no context row / different start"] += 1
            continue
        if c["ask_before_play"]-c["bid_before_play"] > 4 or r.outcome not in (0.0, 1.0):
            skip["wide quote before play or void"] += 1
            continue
        fav_of_ctx = c["price_before_play"]          # on the side the warm-up favoured
        p_yes = fav_of_ctx if c["yes"] else 100-fav_of_ctx
        yes = p_yes > 50
        d = data[:, off[i]:off[i+1]]
        col = d[3, 1:] if yes else 100-d[4, 1:]      # bars after the detection bar
        col = col[~np.isnan(col)]
        first = r.round in ("Qualification Round 1", "Qualification", "Qualifying")
        out.append(dict(family="tennis-FRQ" if first else "tennis-other", tour=r.tour, event=r.event, start=float(r.start_ts),
                        p=p_yes if yes else 100-p_yes, payout=r.outcome if yes else 1-r.outcome,
                        low=float(col.min()) if len(col) else 999, fee=MAKER[r.tour]))
    for k, v in skip.items():
        print("  tennis skipped", k, v)
    return out


def bid_of(p, name):
    if name.startswith("P-"):
        return int(np.floor(p))-int(name[2:])
    b = int(name)
    return b if b <= p-5 else None


def boot(v):
    v = np.asarray(v, float)
    if len(v) < 2:
        return float("nan"), float("nan")
    means = v[RNG.integers(0, len(v), (4000, len(v)))].mean(axis=1)
    return tuple(np.percentile(means, [2.5, 97.5]))


def cell(games, lo, hi, name):
    placed, fills = 0, []
    for g in games:
        if not lo <= g["p"] < hi:
            continue
        b = bid_of(g["p"], name)
        if b is None or b < 2:
            continue
        placed += 1
        if g["low"] < b:
            fee = g["fee"]*b/100*(1-b/100)
            fills.append((g["start"], 100*(100*g["payout"]-b-fee)/(b+fee), g["payout"], b))
    if not fills:
        return dict(placed=placed, n=0)
    fills.sort()
    r = np.array([f[1] for f in fills])
    h = len(r)//2
    lo95, hi95 = boot(r)
    return dict(placed=placed, n=len(r), mean=r.mean(), lo=lo95, hi=hi95, h1=r[:h].mean() if h else float("nan"),
                h2=r[h:].mean(), win=100*np.mean([f[2] for f in fills]), bid=np.mean([f[3] for f in fills]))


def line(c):
    if not c["n"]:
        return f"placed {c['placed']:5}  fills    0"
    return (f"placed {c['placed']:5}  fills {c['n']:4}  avg bid {c['bid']:5.1f}c  win {c['win']:5.1f}%  "
            f"ret {c['mean']:+6.1f}% [{c['lo']:+6.1f}, {c['hi']:+6.1f}]  halves {c['h1']:+6.1f}% / {c['h2']:+6.1f}%")


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    games = basketball()+tennis()
    lines = ["BLIND REBUILD of the dip surface. Return per fill in % of stake, net of maker fee; 95% game bootstrap; "
             "chronological halves of the fills.", ""]
    fams = ["cbb-men", "cbb-women", "nba", "tennis-FRQ", "tennis-other"]
    for fam in fams:
        part = [g for g in games if g["family"] == fam]
        lines.append(f"== {fam}: {len(part)} games with a pre-game price")
        for lo, hi in BANDS:
            for name in BIDS:
                c = cell(part, lo, hi, name)
                if c["placed"]:
                    lines.append(f"  {lo}-{hi}c  bid {name:5} {line(c)}")
            lines.append("")
    for label, tours in (("tennis-FRQ Challenger+WTA+WTA Challenger", ("Challenger", "WTA", "WTA Challenger")),
                         ("tennis-FRQ ITF", ("ITF men", "ITF women")), ("tennis-FRQ ATP", ("ATP",))):
        part = [g for g in games if g["family"] == "tennis-FRQ" and g["tour"] in tours]
        lines.append(f"== {label}: {len(part)} matches")
        for lo, hi in ((80, 85), (85, 90), (80, 90), (90, 99)):
            for name in ("P-5", "P-10", "80", "75"):
                c = cell(part, lo, hi, name)
                if c["placed"]:
                    lines.append(f"  {lo}-{hi}c  bid {name:5} {line(c)}")
        lines.append("")
    text = "\n".join(lines)
    (OUT/"blind.txt").write_text(text)
    pickle.dump(games, open(OUT/"blind-games.pkl", "wb"))
    print(text)


if __name__ == "__main__":
    main()
