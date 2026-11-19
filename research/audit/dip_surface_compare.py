"""AUDIT step 2: the blind rebuild against research/dip_surface.py's orders, game by game and cell by cell.

Reads data/processed/dip-surface/orders.pkl (the original's orders) and data/processed/audit/dip_surface/blind-games.pkl.
Scores both with the ORIGINAL's cell() (no fee, halves of the orders by settlement, bootstrap over games), also on the
pooled 80-90c band the original never scored within a family, and rebuilds the women's orders without the
"clean quote three hours earlier" requirement that wcbb_backfill's 'tip' view inherits.

    .venv/bin/python research/audit/dip_surface_compare.py
"""
from collections import Counter
from pathlib import Path
import pickle
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from research import dip_surface as ds

OUT = ROOT/"data/processed/audit/dip_surface"
FAM = {"men": "cbb-men", "women": "cbb-women", "nba": "nba", "frq": "tennis-FRQ", "tennis-other": "tennis-other"}
lines = []


def say(t=""):
    print(t, flush=True)
    lines.append(t)


def women_tip_only():
    """Women's orders built exactly as the original builds the men's (tip_price + in_play on Bars), no 3h requirement."""
    cache = OUT/"women-tip-only.pkl"
    if cache.exists():
        return pickle.loads(cache.read_bytes())
    from research import basketball_reversion as br
    from research import favourite_contexts as fc
    store, out = br.Bars(), []
    for e in sorted(br.events_of("cbb-women").values(), key=lambda x: (x[0], x[2])):
        settle, _, event, archived, ms = e
        if len(ms) < 2 or not all(m.get("_outcome") in (0.0, 1.0) for m in ms):
            continue
        start = fc.start_of("cbb-women", e)
        bars = store.get("KXNCAAWBGAME", e)
        if not start or not bars or not 1800 <= settle-start <= 7*3600:
            continue
        tip = ds.tip_price(bars, start, 25*60 if archived else 120)
        if tip is None or tip[0] >= 99:
            continue
        price, yes = tip
        m = ms[0]
        out.append(ds.order("women+", event, m["ticker"], settle, start, price, m["_outcome"] if yes else 1-m["_outcome"],
                            ds.in_play(bars, yes, start), archived=bool(archived), yes=yes))
    cache.write_bytes(pickle.dumps(out))
    return out


def pct(s):
    """Original cell -> text in % of stake with the interval converted from cents."""
    if not s.get("fills"):
        return "no fills"
    ci = s.get("ci")
    k = 100/s["need"]
    h = lambda x: f'{100*x["net"]/x["need"]:+5.1f}%({x["fills"]})' if x.get("fills") else "n/a"
    return (f'games {s["orders"]:5} fills {s["fills"]:4} win {s["win"]:5.1f}% ret {s["ret"]:+5.1f}% '
            + (f'[{ci[0]*k:+5.1f},{ci[1]*k:+5.1f}]' if ci else "[n<10]") + f' halves {h(s["h1"])} / {h(s["h2"])}'
            + ("  PASS" if s.get("pass") else ""))


def main():
    orders, _ = ds.load()
    blind = pickle.load(open(OUT/"blind-games.pkl", "rb"))
    b_by = {(g["family"], g["event"]): g for g in blind}
    say("=== A. GAME BY GAME: original orders against the blind rebuild")
    for fam, bfam in FAM.items():
        part = [o for o in orders if o["fam"] == fam]
        mine = [g for g in blind if g["family"] == bfam]
        both = [(o, b_by[bfam, o["event"]]) for o in part if (bfam, o["event"]) in b_by]
        dp = np.array([o["price"]-b["p"] for o, b in both])
        dl = np.array([o["low"]-b["low"] for o, b in both])
        pay = sum(o["payout"] != b["payout"] for o, b in both)
        say(f"  {fam:13} original {len(part):6} blind {len(mine):6} in both {len(both):6} | price differs by >1c in {np.mean(np.abs(dp) > 1):.1%}, "
            f">3c in {np.mean(np.abs(dp) > 3):.1%} | in-play low differs in {np.mean(dl != 0):.1%} | favourite/payout differs in {pay}")
    say("\n=== B. CELL BY CELL, ORIGINAL METHOD (ds.cell) on the original orders; then the same on the pooled 80-90c band")
    by = {fam: [o for o in orders if o["fam"] == fam] for fam in FAM}
    by["women+"] = women_tip_only()
    say(f"  women+ = women's games rebuilt like the men's (tip price only, no 3h-quote requirement): {len(by['women+'])} orders against {len(by['women'])}; "
        f"80-90c games {sum(80 <= o['price'] < 90 for o in by['women+'])} against {sum(80 <= o['price'] < 90 for o in by['women'])}")
    freq = lambda hit: 0.0
    for fam in ("men", "women", "women+", "frq", "tennis-other", "nba"):
        say(f"  --- {fam}")
        for band in ((80, 85), (85, 90), (80, 90), (90, 99)):
            part = [o for o in by[fam] if band[0] <= o["price"] < band[1]]
            for level in (("d", 5), ("d", 10), ("d", 15), ("d", 20), ("a", 80), ("a", 70)):
                ds.RNG = np.random.default_rng(11)
                s = ds.cell(fam, band, level, part, freq)
                if s["orders"]:
                    say(f"    {band[0]}-{band[1]}c {ds.name_of(level):>5} {pct(s)}")
    say("\n=== C. FRQ by tour group (original orders), pooled 80-90c")
    for label, tours in (("Challenger + WTA + WTA Challenger", ("Challenger", "WTA", "WTA Challenger")), ("ATP", ("ATP",)), ("ITF", ("ITF men", "ITF women"))):
        part = [o for o in by["frq"] if o.get("tour") in tours and 80 <= o["price"] < 90]
        say(f"  {label}: {len(part)} matches at 80-90c ({dict(Counter(o['tour'] for o in part))})")
        for level in (("d", 5), ("d", 10), ("d", 15), ("d", 20)):
            ds.RNG = np.random.default_rng(11)
            s = ds.cell("frq", (80, 90), level, part, freq)
            if s["orders"]:
                say(f"    80-90c {ds.name_of(level):>5} {pct(s)}")
    say("\n=== D. FRQ monthly, 80-90c P-10 (original orders): fills, win rate, return")
    part = [o for o in by["frq"] if 80 <= o["price"] < 90]
    for month in sorted({o["month"] for o in part}):
        hit = ds.filled(ds.placed([o for o in part if o["month"] == month], ("d", 10)))
        if hit:
            nets = [100*o["payout"]-b for o, b in hit]
            say(f"    {month} fills {len(hit):3} win {100*np.mean([o['payout'] for o, _ in hit]):5.1f}% ret {100*np.mean(nets)/np.mean([b for _, b in hit]):+6.1f}%")
    (OUT/"compare.txt").write_text("\n".join(lines))


if __name__ == "__main__":
    main()
