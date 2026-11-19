"""Pooled interval for decided-market taker entries (ask <= 99c, deciding play >= 15 s old), clustered by game."""
import json, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
from inv2_common import boot, halves, fmt
rows = json.load(open(Path(__file__).resolve().parents[2]/"data/processed/decided/trades.json"))
for rule in ("entry99_15", "entry98_15", "entry99_120"):
    for lab, keep in (("all", lambda r: True), ("verified only", lambda r: r["verified"])):
        hit = [dict(r, **{"e": r[rule]}) for r in rows if r.get(rule) and keep(r)]
        if not hit:
            continue
        pct = lambda r: 100*r["e"]["net"]/r["e"]["ask"]
        ev = lambda r: r["sport"]+r["key"]
        b = boot(hit, ev, pct)
        h = halves(hit, ev, lambda r: r["e"]["ts"], pct)
        print(f'{rule} {lab}: ask {sum(r["e"]["ask"] for r in hit)/len(hit):.1f} losses {sum(r["payout"] < .5 for r in hit)} '
              f'cents {sum(r["e"]["net"] for r in hit)/len(hit):+.2f} at 98-99c only n={sum(r["e"]["ask"] >= 98 for r in hit)} | {fmt(b, h)}')
    hit = [dict(r, e=r[rule]) for r in rows if r.get(rule) and r["verified"] and r[rule]["ask"] >= 96]
    if hit:
        pct = lambda r: 100*r["e"]["net"]/r["e"]["ask"]
        print("   verified, ask 96-99 only:", fmt(boot(hit, lambda r: r["sport"]+r["key"], pct)), "losses", sum(r["payout"] < .5 for r in hit))
print("rows", len(rows), "keys", sorted({k for r in rows for k in r if k.startswith("entry")}))
