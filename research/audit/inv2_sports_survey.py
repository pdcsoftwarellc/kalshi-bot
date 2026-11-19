"""Pooled event-clustered intervals and halves for sports_survey saved trades (taker, next-minute open ask, held).

Reproduces one_per_event (earliest entry per event; three-way events YES side only). "liquid+risen" is a
diagnostic computed here, not in the study: the ask had been 10c lower earlier and the entry minute traded >= 50.
"""
import json, sys
from collections import defaultdict
from statistics import median
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
from inv2_common import boot, halves, fmt
D = Path(__file__).resolve().parents[2]/"data/processed/sports-survey"


def one(rows, t):
    best = {}
    for r in rows:
        if r["threshold"] != t or (r["markets"] >= 3 and r["side"] == "no"):
            continue
        key = (r["ts"], r["ticker"], r["side"])
        if r["event"] not in best or key < best[r["event"]][0]:
            best[r["event"]] = (key, r)
    return [r for _, r in best.values()]


pct = lambda r: 100*r["net_cents"]/r["price"]
for name in ("survey", "survey2", "holdout"):
    rows = json.load(open(D/f"{name}-trades.json"))
    summ = json.load(open(D/f"{name}.json"))
    fam = defaultdict(list)
    for r in rows:
        fam[r["family"]].append(r)
    print("==", name)
    for f, rs in fam.items():
        print(f' {f}: events complete {summ[f]["events"]} three-way {summ[f]["three_way_events"]} coverage {summ[f]["coverage"]}')
        for t in (95, 97, 98):
            for lab, keep in (("all", lambda r: True), ("liquid+risen", lambda r: r["risen"] and r["minute_volume"] >= 50)):
                hit = [r for r in one(rs, t) if keep(r)]
                if not hit:
                    continue
                b = boot(hit, lambda r: r["event"], pct)
                h = halves(hit, lambda r: r["event"], lambda r: r["ts"], pct)
                c = sum(r["net_cents"] for r in hit)/len(hit)
                wk = (max(r["ts"] for r in hit)-min(r["ts"] for r in hit))/604800 or 1
                print(f'   >={t} {lab:12} price {sum(r["price"] for r in hit)/len(hit):.1f} L {sum(r["payout"] < .5 for r in hit):3} '
                      f'cents {c:+.2f} medvol {median(r["minute_volume"] for r in hit):.0f} /wk {len(hit)/wk:.0f} | {fmt(b, h)}')
