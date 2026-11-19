"""Audit helper (read-only): condense data/processed/thin-book/simulate.json and trades.json.
All profit numbers and intervals are COPIED from those files (the script bootstraps over matches).
COMPUTED here: the average fill price per fills-*.json file, used only to turn cents per fill into % of stake
(stake of a buy = price, of a sell = 100-price)."""
import json
from pathlib import Path
import sys
ROOT = Path(__file__).resolve().parents[2]
D = ROOT/"data/processed/thin-book"
s = json.load(open(D/"simulate.json"))
f = lambda c: "None" if c is None else f"[{c[0]:+.2f},{c[1]:+.2f}]"
g = lambda x: "na" if x is None else f"{x:+.2f}"
stake = {}
for p in sorted(D.glob("fills-*.json")):
    rows = json.load(open(p))
    st = [pr if side > 0 else 100-pr for r in rows for _, side, pr in r["fills"]]
    by = {}
    for r in rows:
        by.setdefault(r["tour"], []).extend(pr if side > 0 else 100-pr for _, side, pr in r["fills"])
    stake[p.stem] = sum(st)/max(1, len(st))
    print(f"{p.stem:28} matches {len(rows)} with-fill {sum(bool(r['fills']) for r in rows)} fills {len(st)} mean stake {stake[p.stem]:.1f}c  by tour "
          + " ".join(f"{t}:{len(v)}/{(sum(v)/len(v) if v else 0):.0f}c" for t, v in sorted(by.items())))
want = sys.argv[1:] or ["all tours | all"]
for k, v in s.items():
    if k.startswith("thick") or not any(k.endswith(w) for w in want):
        continue
    if not v.get("fills"):
        print(f"{k:50} matches {v['matches']} no fills")
        continue
    e, l = v["early60"], v["late40"]
    print(f"{k:50} m {v['matches']} fills {v['fills']} mfill {v['matches_with_a_fill']:.2f} RT {v['round_trips']} c/RT {g(v['spread_per_round_trip'])}{f(v['spread_per_round_trip_ci'])} edge {v['edge_at_quote']:.2f} mk5 {v['markout_5m']:+.2f}{f(v['markout_5m_ci'])} "
          f"exit {v['exit_pnl_per_fill']:+.2f}{f(v['exit_pnl_per_fill_ci'])} mid {v['mid_pnl_per_fill']:+.2f}{f(v['mid_pnl_per_fill_ci'])} settle {v['settle_pnl_per_fill']:+.2f}{f(v['settle_pnl_per_fill_ci'])} "
          f"E exit/settle {g(e['exit_per_fill'])}/{g(e['settle_per_fill'])} (n={e['fills']}) L {g(l['exit_per_fill'])}/{g(l['settle_per_fill'])} (n={l['fills']}) maxinv p95 {v['max_inventory_p95']:.0f}")
t = json.load(open(D/"trades.json"))
for k, v in t.items():
    if k.startswith("queue") and isinstance(v, dict):
        if not v.get("fills"):
            print(f"{k:70} m {v.get('matches')} no fills")
            continue
        e, l = v["early60"], v["late40"]
        print(f"{k:70} m {v['matches']} fills {v['fills']} mk5 {v['markout_5m']:+.2f} exit {v['exit_pnl_per_fill']:+.2f}{f(v['exit_pnl_per_fill_ci'])} mid {v['mid_pnl_per_fill']:+.2f}{f(v['mid_pnl_per_fill_ci'])} "
              f"settle {v['settle_pnl_per_fill']:+.2f}{f(v['settle_pnl_per_fill_ci'])} E {g(e['exit_per_fill'])}/{g(e['settle_per_fill'])}({e['fills']}) L {g(l['exit_per_fill'])}/{g(l['settle_per_fill'])}({l['fills']})")
