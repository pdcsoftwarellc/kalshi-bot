"""AUDIT step 3b: can research/dip_filters.py part (A) see a planted pre-play feature that truly separates dip fills?

Uses the study's own rows (data/processed/favorite-dip/filters-rows.json: 705 holdout and 212 recent WTA / ATP Challenger
favourites at 90c+, 228 and 62 fills at 80c) and its own functions (ladder, ci), unchanged. Per simulation:
  * a new feature X is drawn for every match (uniform, independent of everything);
  * every FILLED match's payout is redrawn: win with probability base + delta/2 when X is above the median, base - delta/2
    below, base = the period's real win rate of a fill. So fills with high X truly win `delta` points more often.
  * the original analysis is run: holdout tercile cuts applied to both periods, net per fill with its bootstrap
    interval per tercile, and the rank correlation between X and winning a fill against the "noise band" 2/sqrt(fills).
dip_filters states no decision rule, so three readings of its tables are scored:
  rho-holdout   |rho| beyond the noise band on the holdout, right sign
  rho-both      the same on the holdout AND on the recent period (the study's two-period habit)
  terciles      holdout: the high tercile's interval lies wholly above the low tercile's
  bonf-11       holdout rho beyond the band widened for the study's 11 features (2.84/sqrt(fills))
delta = 0 gives the false-positive rate. A second block repeats it with the fills scaled up, to find the sample needed.

    .venv/bin/python research/audit/dip_filters_power.py [sims]
"""
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from research import dip_filters as df

OUT = ROOT/"data/processed/audit/dip_surface"
RNG = np.random.default_rng(303)
lines = []


def say(t=""):
    print(t, flush=True)
    lines.append(t)


def filled(r):
    return r["low_in_play"] is not None and r["low_in_play"] < 80


def plant(data, delta, scale=1):
    out = {}
    for p, rows in data.items():
        rows = [r for r in rows if r["features"]]
        if scale != 1:
            rows = [rows[i] for i in RNG.integers(0, len(rows), int(scale*len(rows)))]
        base = float(np.mean([r["payout"] for r in rows if filled(r)]))
        x = RNG.random(len(rows))
        new = []
        for r, v in zip(rows, x):
            r = dict(r, x=float(v))
            if filled(r):
                r["payout"] = float(RNG.random() < min(1.0, base+(delta/2 if v > .5 else -delta/2)))
            new.append(r)
        out[p] = new
    return out


def analyse(data):
    """The original part (A) for one feature: terciles with holdout cuts, rank correlation on the fills."""
    lo, hi = np.percentile([r["x"] for r in data["holdout"]], [100/3, 200/3])
    res = {}
    for p, rows in data.items():
        low = df.ladder([r for r in rows if r["x"] <= lo], (80,), "low_in_play")["per_fill"]
        high = df.ladder([r for r in rows if r["x"] > hi], (80,), "low_in_play")["per_fill"]
        fills = [(r["x"], r["payout"]) for r in rows if filled(r)]
        xr = np.argsort(np.argsort([f[0] for f in fills])).astype(float)
        y = np.array([f[1] for f in fills])
        rho = float(np.corrcoef(xr, y)[0, 1]) if y.std() > 0 else 0.0
        res[p] = dict(rho=rho, n=len(fills), low=low.get("ci95"), high=high.get("ci95"))
    return res


def power(data, deltas, sims, scale=1):
    for delta in deltas:
        got = dict(rho_h=0, rho_both=0, terc=0, bonf=0)
        ns = []
        for _ in range(sims):
            df.RNG = RNG
            a = analyse(plant(data, delta/100, scale))
            h, r = a["holdout"], a["recent"]
            ns.append((h["n"], r["n"]))
            hit_h = h["rho"] > 2/h["n"]**.5
            got["rho_h"] += hit_h
            got["rho_both"] += hit_h and r["rho"] > 2/r["n"]**.5
            got["bonf"] += h["rho"] > 2.84/h["n"]**.5
            got["terc"] += bool(h["low"] and h["high"] and h["high"][0] > h["low"][1])
        n = np.mean(ns, axis=0)
        say(f"    true difference {delta:4.0f} points (fills {n[0]:.0f} holdout, {n[1]:.0f} recent): found by rho-holdout {got['rho_h']/sims:4.0%} | "
            f"rho-both {got['rho_both']/sims:4.0%} | terciles {got['terc']/sims:4.0%} | bonf-11 {got['bonf']/sims:4.0%}")


def main():
    sims = int(sys.argv[1]) if len(sys.argv) > 1 else 400
    data = json.loads((ROOT/"data/processed/favorite-dip/filters-rows.json").read_text())
    say(f"PLANTED-FILTER POWER of dip_filters part (A) ({sims} simulations per line)")
    say("  at the study's own sample")
    power(data, (0, 8, 12, 16, 20, 25, 30), sims)
    for scale in (2, 4, 8):
        say(f"  with {scale}x the matches (resampled)")
        power(data, (8, 12), sims, scale)
    (OUT/"power-filters.txt").write_text("\n".join(lines))


if __name__ == "__main__":
    main()
