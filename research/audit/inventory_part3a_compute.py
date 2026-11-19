"""Inventory audit, part 3a: recompute per-bet returns with match-clustered intervals from SAVED rows only.

Read-only on the research outputs; no network; nothing is rerun. Reads
  data/processed/tennis-crossings/{matches,fills,coin,reentry}.pkl   (written by research/tennis_crossings.py)
  data/processed/tennis-reversion/events.pkl                         (written by research/tennis_reversion.py)
and prints, per grouped idea: bets, independent matches, mean % per bet (sum of net cents / sum of cents paid, i.e.
one contract per bet), a 95% bootstrap interval resampling MATCHES (2,000 draws), the two chronological periods
(archive = settled before 2026-09-18, recent = after), and a capacity guide (bets per month in the recent period,
median contracts traded in the fill bar).

Fees are exactly the studies' own columns unless a line says "no maker fee": taker 0.07*p*(1-p); maker 0 on
Challenger / WTA Challenger, 0.0175*p*(1-p) elsewhere in tennis_crossings (which includes ITF), and in
tennis_reversion 0 on Challenger / WTA Challenger / ITF, 0.0175*p*(1-p) on ATP / WTA.

    .venv/bin/python research/audit/inventory_part3a_compute.py > data/processed/audit/inventory/part-3a-computed.txt
"""
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
CROSS = ROOT/"data/processed/tennis-crossings"
REV = ROOT/"data/processed/tennis-reversion"
RNG = np.random.default_rng(20261119)
DRAWS = 2000
RECENT_MONTHS = 59/30.44          # 18 Sep - 16 Nov, as in tennis_crossings.py
LEVELS = np.arange(5, 100, 5)
BANDS = ((0, 20), (20, 35), (35, 50), (50, 65), (65, 80), (80, 90), (90, 101))
BAND_NAMES = ("<20", "20-35", "35-50", "50-65", "65-80", "80-90", "90+")
NO_MAKER_FEE = ("Challenger", "WTA Challenger")
UNITS = {"Challenger": ("Challenger",), "WTA Challenger": ("WTA Challenger",), "WTA": ("WTA",), "ATP": ("ATP",),
         "ITF men": ("ITF men",), "ITF women": ("ITF women",), "Chall+WTA Chall": ("Challenger", "WTA Challenger"),
         "ATP+WTA": ("ATP", "WTA"), "ITF both": ("ITF men", "ITF women")}


def ratio_boot(net, stake, cluster):
    """100 * sum(net) / sum(stake) with a 95% interval resampling clusters. Returns (pct, lo, hi, clusters)."""
    net, stake = np.asarray(net, float), np.asarray(stake, float)
    keep = ~np.isnan(net) & ~np.isnan(stake)
    net, stake, cluster = net[keep], stake[keep], np.asarray(cluster)[keep]
    if not len(net):
        return np.nan, np.nan, np.nan, 0
    codes, _ = pd.factorize(cluster)
    k = codes.max()+1
    a, b = np.bincount(codes, net, k), np.bincount(codes, stake, k)
    pct = 100*a.sum()/b.sum()
    if k < 10:
        return pct, np.nan, np.nan, k
    out = np.empty(DRAWS)
    step = max(1, int(4e6//k))
    for s in range(0, DRAWS, step):
        idx = RNG.integers(0, k, (min(step, DRAWS-s), k))
        out[s:s+len(idx)] = 100*a[idx].sum(1)/b[idx].sum(1)
    lo, hi = np.percentile(out, [2.5, 97.5])
    return pct, lo, hi, k


def line(label, g, net="net", stake="paid", cluster="mid", vol=None):
    pct, lo, hi, k = ratio_boot(g[net], g[stake], g[cluster])
    ok = g[net].notna()
    n = int(ok.sum())
    if n == 0:
        print(f"{label:74} n 0")
        return
    parts = []
    for p in ("archive", "recent"):
        h = g[ok & (g.period == p)]
        parts.append(f"{p[:4]} {100*h[net].sum()/h[stake].sum():+6.2f}% (n {len(h)})" if len(h) else f"{p[:4]} n/a")
    rec = int((ok & (g.period == "recent")).sum())
    extra = f" | vol med {g.loc[ok, vol].median():.0f}" if vol else ""
    print(f"{label:74} n {n:6} matches {k:5} | c/bet {g.loc[ok, net].mean():+6.2f} at {g.loc[ok, stake].mean():5.1f}c = {pct:+6.2f}% [{lo:+6.2f},{hi:+6.2f}] | "
          f"{parts[0]} | {parts[1]} | {rec/RECENT_MONTHS:6.1f}/mo recent{extra}")


def crossings():
    m = pd.read_pickle(CROSS/"matches.pkl").set_index("mid")
    f = pd.read_pickle(CROSS/"fills.pkl").join(m[["tour", "period", "p0", "outcome", "settle"]], on="mid")
    f["p0p"] = np.where(f.player == 0, f.p0, 100-f.p0)
    f["payout"] = np.where(f.player == 0, f.outcome, 1-f.outcome)
    f["band"] = pd.cut(f.p0p, [b[0] for b in BANDS]+[101], right=False, labels=False)
    f["level"] = LEVELS[f.li.values]
    taker = 7*f.paid/100*(1-f.paid/100)
    maker = 1.75*f.paid/100*(1-f.paid/100)
    f["gross"] = 100*f.payout-f.paid
    f["net"] = f.gross-np.where(f.dir == 1, taker, np.where(f.tour.isin(NO_MAKER_FEE), 0.0, maker))
    f["net_nofee"] = f.gross-np.where(f.dir == 1, taker, 0.0)
    print(f"CROSSINGS {len(m)} matches, {len(f)} fills; net columns rebuilt exactly as tennis_crossings.load()")
    print("\n== DOWN: resting bid at L (strict print-through), held to settlement. 'near' = L within 15c of the band's lower edge, 'deep' = further below")
    for unit, tours in UNITS.items():
        for b, name in enumerate(BAND_NAMES):
            g = f[f.tour.isin(tours) & (f.dir == 0) & (f.band == b)]
            cut = BANDS[b][0]-15
            for depth, sub in (("near", g[g.level >= cut]), ("deep", g[g.level < cut])):
                if len(sub) < 20:
                    continue
                lv = f"L {int(sub.level.min())}-{int(sub.level.max())}"
                line(f"down | {unit} | P0 {name} | {depth} {lv} | study fees", sub, vol="bar_volume")
                if not set(tours) <= set(NO_MAKER_FEE):
                    line(f"down | {unit} | P0 {name} | {depth} {lv} | no maker fee", sub, net="net_nofee", vol="bar_volume")
    print("\n== DOWN, one level per row for the favourite bands (study fees)")
    for unit in ("Challenger", "WTA Challenger", "WTA", "ATP", "ITF men", "ITF women", "Chall+WTA Chall", "ATP+WTA", "ITF both"):
        for b in (4, 5, 6):
            for level in (50, 60, 70, 75, 80, 85, 90):
                g = f[f.tour.isin(UNITS[unit]) & (f.dir == 0) & (f.band == b) & (f.level == level)]
                if len(g) >= 20:
                    line(f"down | {unit} | P0 {BAND_NAMES[b]} | L={level}", g, vol="bar_volume")
    print("\n== UP: buy on strength (taker at the next bar's opening ask in [L, L+4], fee 0.07*p*(1-p)), held to settlement")
    for unit, tours in UNITS.items():
        for b, name in enumerate(BAND_NAMES):
            g = f[f.tour.isin(tours) & (f.dir == 1) & (f.band == b)]
            if len(g) >= 20:
                line(f"up | {unit} | P0 {name} | all levels", g, vol="bar_volume")
    print("\n== UP, underdogs P0 10-35 buying strength at 50/60/70 (special a), and lottery bids at 5/10")
    dog = f[(f.p0p >= 10) & (f.p0p < 35)]
    for unit in ("Chall+WTA Chall", "ATP+WTA", "ITF both"):
        for level in (50, 60, 70):
            line(f"up | {unit} | P0 10-35 | L={level}", dog[dog.tour.isin(UNITS[unit]) & (dog.dir == 1) & (dog.level == level)], vol="bar_volume")
        for level in (5, 10):
            g = f[f.tour.isin(UNITS[unit]) & (f.dir == 0) & (f.level == level)]
            line(f"down | {unit} | any P0 | lottery bid L={level} | study fees", g, vol="bar_volume")
            line(f"down | {unit} | any P0 | lottery bid L={level} | no maker fee", g, net="net_nofee", vol="bar_volume")

    coin = pd.read_pickle(CROSS/"coin.pkl").join(m[["tour", "period", "p0", "outcome"]], on="mid")
    coin["p0p"] = np.where(coin.player == 0, coin.p0, 100-coin.p0)
    coin["payout"] = np.where(coin.player == 0, coin.outcome, 1-coin.outcome)
    coin["paid"] = coin.level.astype(float)
    coin["net_nofee"] = 100*coin.payout-coin.level
    coin["net"] = coin.net_nofee-np.where(coin.tour.isin(NO_MAKER_FEE), 0.0, 1.75*coin.level/100*(1-coin.level/100))
    coin = coin[(coin.p0p >= 35) & (coin.p0p <= 65) & coin.filled]
    print("\n== SPECIAL (b): coin-flip match (P0 35-65), bid reaches H, then a resting bid at L (strict fill), held to settlement")
    for unit in ("Chall+WTA Chall", "ATP+WTA", "ITF both"):
        for high in (80, 85, 90):
            g = coin[coin.tour.isin(UNITS[unit]) & (coin.high == high)]
            line(f"coin | {unit} | reach {high} then bid 65/70/75 pooled | study fees", g, vol="bar_volume")
        g = coin[coin.tour.isin(UNITS[unit])]
        line(f"coin | {unit} | all H, all L pooled | study fees", g, vol="bar_volume")
        if unit != "Chall+WTA Chall":
            line(f"coin | {unit} | all H, all L pooled | no maker fee", g, net="net_nofee", vol="bar_volume")

    re = pd.read_pickle(CROSS/"reentry.pkl").join(m[["tour", "period", "p0", "outcome"]], on="mid")
    re["p0p"] = np.where(re.player == 0, re.p0, 100-re.p0)
    re["payout"] = np.where(re.player == 0, re.outcome, 1-re.outcome)
    re["paid"] = 80.0
    re["net"] = 100*re.payout-80-np.where(re.tour.isin(NO_MAKER_FEE), 0.0, 1.75*.8*.2)
    print("\n== SPECIAL (c): the SECOND dip below 80 after a recovery to a 90 bid (80c bid, strict fill implied by the first-dip rule)")
    for lo, hi in ((90, 101), (80, 90)):
        for unit in ("Chall+WTA Chall", "WTA", "ATP", "ITF both"):
            g = re[re.tour.isin(UNITS.get(unit, (unit,))) & (re.p0p >= lo) & (re.p0p < hi) & re.second]
            line(f"reentry | {unit} | P0 {lo}-{min(hi, 100)} | second dip", g)


def reversion():
    E = pd.read_pickle(REV/"events.pkl")
    groups = {"ATP+WTA": ("ATP", "WTA"), "Challenger": ("Challenger", "WTA Challenger"), "ITF": ("ITF men", "ITF women")}
    print(f"\n\nREVERSION {len(E)} event rows in {E.m.nunique()} matches (events are nested: a 20c move is also a 5/10/15c event)")
    rules = (("taker buy fav, sell at bid +15m", "fav_take15", "fav_ask"), ("taker buy fav, hold to settle", "fav_takeS", "fav_ask"),
             ("taker buy dog, sell at bid +15m", "dog_take15", "dog_ask"), ("taker buy dog, hold to settle", "dog_takeS", "dog_ask"),
             ("maker bid fav, sell at bid +15m", "fav_make_take15", "fav_make_L"), ("maker bid fav, offer +3c 30m", "fav_make_offer3", "fav_make_L"),
             ("maker bid fav, offer +5c 30m", "fav_make_offer5", "fav_make_L"), ("maker bid fav, hold to settle", "fav_makeS", "fav_make_L"),
             ("maker bid dog, offer +3c 30m", "dog_make_offer3", "dog_make_L"), ("maker bid dog, hold to settle", "dog_makeS", "dog_make_L"))
    for X in (10, 20):
        for direction in ("fall", "rise"):
            print(f"\n== after a {direction} of {X}c+ in five minutes, all favourite bands pooled")
            for group, tours in groups.items():
                q = E[E.tour.isin(tours) & (E.X == X) & (E.dir == direction)]
                for name, col, stake in rules:
                    line(f"{direction} {X}c+ | {group} | {name}", q, net=col, stake=stake, cluster="m", vol="vol_next")
    print("\n== after a fall of 10c+, by favourite band: maker bid for the favourite, held to settlement / offer +3c")
    for group, tours in groups.items():
        for band in ("even", "60-70", "70-80", "80-90", "90+"):
            q = E[E.tour.isin(tours) & (E.X == 10) & (E.dir == "fall") & (E.band == band)]
            line(f"fall 10c+ | {group} | band {band} | maker bid fav, hold", q, net="fav_makeS", stake="fav_make_L", cluster="m", vol="vol_next")
            line(f"fall 10c+ | {group} | band {band} | maker bid fav, offer +3c", q, net="fav_make_offer3", stake="fav_make_L", cluster="m", vol="vol_next")
            line(f"fall 10c+ | {group} | band {band} | taker buy fav, hold", q, net="fav_takeS", stake="fav_ask", cluster="m", vol="vol_next")


if __name__ == "__main__":
    pd.set_option("display.width", 250)
    crossings()
    reversion()
