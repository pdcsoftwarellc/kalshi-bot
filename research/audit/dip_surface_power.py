"""AUDIT step 3a/3c: can research/dip_surface.py's pass rule see a planted edge?

The original's own functions are imported unchanged (placed, filled, cell, stat, halves, boot, p_one_sided). Each
simulated season keeps the band's real games and pre-game prices; fills come from the original's own fair-market replay
model and the edge is planted by turning would-be losers into winners: see season(). edge = 0 is the fair market (the
false-positive rate of the rule). Two earlier attempts were dropped: redrawing outcomes on the REAL lows cannot be made
consistent (a real low already says who lost), and deflating the price below a martingale changes the fill counts.

Detection, per simulated season:
  cell      the original PASS (30+ fills, bootstrap interval above zero, both halves positive) at the named cell
  any       at least one of the six levels P-5, P-10, P-15, P-20, 80c, 70c passes in the band
  bonf      the P-10 cell has one-sided p < 0.05/225 (the original's Bonferroni line)
Bands scored: the original's 80-85c and 85-90c, the pooled 80-90c it did not score, and 90-99c (the known edge).
ds.DRAWS is lowered from 4,000 to 1,000 bootstrap draws for speed (the interval's end moves by its own noise only).

    .venv/bin/python research/audit/dip_surface_power.py [sims]
"""
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from research import dip_surface as ds

OUT = ROOT/"data/processed/audit/dip_surface"
RNG = np.random.default_rng(77)
DEPTHS = (5, 10, 15, 20, 25, 30)
LEVELS = (("d", 5), ("d", 10), ("d", 15), ("d", 20), ("a", 80), ("a", 70))
lines = []


def say(t=""):
    print(t, flush=True)
    lines.append(t)


def season(part, edge):
    """One simulated season of the band's real games. Fills come from the original's fair-market replay model
    (dip_surface.replay: win with probability P; a winner's lowest price M has P(M < b) = b(100-P) / (P(100-b)); a loser
    fills every bid), so fill counts stay those of a fair market. The edge is planted as a dip that overshoots: a game
    that would have lost is turned into a win with probability phi = edge x b / (100 - b), b = its own P-10 bid, its low
    unchanged. Then P(win | filled at P-10) = b(1 + edge)/100: a true `edge` return per fill at the P-10 bid (a little
    less at P-5, more at deeper bids: edge x [b/(100-b)] x [(100-b2)/b2] at a bid b2)."""
    price = np.array([o["price"] for o in part])
    bid = np.floor(price-10)
    win = RNG.random(len(part)) < price/100
    u = RNG.random(len(part))
    low = np.where(win, 100*u*price/((100-price)+u*price), 0.0)
    win = win | (RNG.random(len(part)) < edge*bid/(100-bid))
    return [dict(o, payout=float(w), low=float(x)) for o, w, x in zip(part, win, low)]


def run(fam, orders, band, edges, sims, ref=("d", 10)):
    part = [o for o in orders if o["fam"] == fam and band[0] <= o["price"] < band[1]]
    base = ds.stat(ds.placed(part, ref))
    sd = base["nets"].std(ddof=1)
    mde = 2.8*sd/base["fills"]**.5/base["need"]*100
    say(f"  {fam:13} {band[0]}-{band[1]}c: games {len(part)}, P-10 fills {base['fills']}, sd of a fill {100*sd/base['need']:.0f}% of stake, "
        f"standard error {100*sd/base['fills']**.5/base['need']:.1f}% -> analytic minimum detectable edge (80% power, interval above zero) {mde:.1f}%; "
        f"fills needed to detect +3%: {int((2.8*100*sd/base['need']/3)**2):,}, +1.5%: {int((2.8*100*sd/base['need']/1.5)**2):,}")
    freq = lambda hit: 0.0
    for edge in edges:
        got = dict(cell=0, any=0, bonf=0)
        rets, fills = [], []
        for _ in range(sims):
            sim = season(part, edge/100)
            passed = []
            for level in LEVELS:
                s = ds.cell(fam, band, level, sim, freq)
                passed.append(bool(s.get("pass")))
                if level == ref:
                    rets.append(s.get("ret", float("nan")))
                    fills.append(s.get("fills", 0))
                    got["cell"] += passed[-1]
                    got["bonf"] += ds.p_one_sided(s.get("z", float("nan"))) < .05/225
            got["any"] += any(passed)
        say(f"      planted {edge:+5.1f}%: P-10 fills {np.mean(fills):5.0f}, measured return mean {np.nanmean(rets):+5.1f}% | P-10 cell passes {got['cell']/sims:4.0%} | "
            f"any of 6 levels passes {got['any']/sims:4.0%} | P-10 survives Bonferroni/225 {got['bonf']/sims:4.0%}")


def main():
    sims = int(sys.argv[1]) if len(sys.argv) > 1 else 200
    ds.DRAWS = 1000
    orders, _ = ds.load()
    say(f"PLANTED-EDGE POWER of dip_surface's PASS rule ({sims} simulated seasons per line; real games and prices, simulated outcome and in-play low)")
    say("\n=== below 90c")
    for fam in ("men", "frq", "tennis-other", "women"):
        for band in ((80, 85), (85, 90), (80, 90)):
            run(fam, orders, band, (0, 1.5, 3, 5, 8, 12, 16) if fam != "tennis-other" else (0, 1.5, 3, 5, 8), sims)
    say("\n=== the known edge: 90-99c (planted at about the measured size)")
    for fam in ("men", "women", "frq"):
        run(fam, orders, (90, 99), (0, 5, 9.5, 13), sims)
    (OUT/"power-surface.txt").write_text("\n".join(lines))


if __name__ == "__main__":
    main()
