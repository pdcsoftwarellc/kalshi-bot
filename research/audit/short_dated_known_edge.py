"""Audit step 3b: would the short-dated study's statistical gates have passed the repo's validated edge?

Feeds the filled 80c-bid orders from data/processed/dip-capacity/rows.json (men's college basketball and tennis
first-round qualifying, pre-game favourite 90-99c, strict fills) through the ORIGINAL functions `clustered`,
`passes` (second study) and `interval` (first study, 60/40 chronological split), imported from the audit copy
with no planted edge. The rule construction itself (fixed minutes before a scheduled close, bid 1c inside the
spread) is not a dip rule and is not applied; only the evidence gates are.
"""
import json, os, sys
from pathlib import Path
os.environ["PLANT"] = "0"
sys.path.insert(0, str(Path(__file__).resolve().parent))
import short_dated_power as sd

ROOT = Path(__file__).resolve().parents[2]
rows_all = json.load(open(ROOT / "data/processed/dip-capacity/rows.json"))
out = []
for group, label in (("men", "men's college basketball"), ("tennis", "tennis first-round qualifying"), ("women", "women's college basketball")):
    orders = [r for r in rows_all if r["group"] == group and 90 <= r["price"] <= 99]
    fills = [dict(event=r["event"], close=r["settle"], payout=r["payout"], rest=100 * r["payout"] - 80, net=100 * r["payout"] - 80)
             for r in orders if r["levels"]["80"]["filled"]]
    times = sorted({(r["close"], r["event"]) for r in fills})
    cut = times[len(times) // 2][0]
    s = sd.clustered(fills, lambda r: r["rest"], cut)
    ok = sd.passes(s, fills)
    cut60 = times[int(.6 * len(times))][0]
    early, late = [r for r in fills if r["close"] < cut60], [r for r in fills if r["close"] >= cut60]
    mean = lambda v: sum(r["net"] for r in v) / len(v)
    out.append(f"{label}: orders {len(orders)} fills {len(fills)} events {s['n']} | second-study gate: net {s['mean']:+.2f}c "
               f"({s['mean'] / 80 * 100:+.1f}%) ci {s['ci']} halves {s['halves']} t {s['t']:+.2f} -> "
               f"{'PASS' if ok else 'FAIL'}; Bonferroni t>=3.5 -> {'PASS' if s['t'] >= sd.BONFERRONI_T else 'FAIL'}\n"
               f"    first-study gate (60/40): early n {len(early)} net {mean(early):+.2f}c ci {sd.interval(early)} | "
               f"late n {len(late)} net {mean(late):+.2f}c ({mean(late) / 80 * 100:+.1f}%) ci {sd.interval(late)} -> "
               f"{'PASS' if (sd.interval(late) or [-1])[0] > 0 else 'FAIL (late interval includes zero)'}")
text = "\n".join(out)
(ROOT / "data/processed/audit/short_dated/known_edge.txt").write_text(text + "\n")
print(text)
