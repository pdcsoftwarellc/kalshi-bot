"""Audit step 2: independent re-computation of the original v2 taker cells (research/short_dated.py `calibrate`),
plus two sensitivities the original did not run:
  late1  = same signal, bought at the NEXT bar's closing ask (one bar of latency), any price < 100
  ffill  = missing minutes filled forward with the last quote (original needs a bar exactly at the decision minute)
Reads data/raw/short-dated/v2/*.json and data/processed/short-dated/v2-calibration.json only.
"""
import json
from collections import defaultdict
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
V2 = ROOT / "data/raw/short-dated/v2"
ORIG = json.load(open(ROOT / "data/processed/short-dated/v2-calibration.json"))["families"]
OUT = ROOT / "data/processed/audit/short_dated/compare.txt"
BINS = ((80, 90), (90, 95), (95, 97), (97, 99))
BUCKETS = {"15m": (10, 5, 2), "hourly": (55, 30, 15, 5)}
FAMS = ["btc15m", "eth15m", "sol15m", "xrp15m", "btc_hourly", "eth_hourly", "sp_hourly", "nasdaq_hourly"]
fee = lambda a: 7 * a / 100 * (1 - a / 100)


def ev_stats(rows):  # rows: (event, close, value) -> event-weighted mean, ci, halves
    by, when = defaultdict(list), {}
    for e, c, v in rows:
        by[e].append(v); when[e] = c
    if not by:
        return None
    names = list(by); x = np.array([np.mean(by[e]) for e in names]); t = np.array([when[e] for e in names])
    ci = (float("nan"),) * 2
    if len(x) >= 30:
        d = x[np.random.default_rng(3).integers(0, len(x), (2000, len(x)))].mean(1)
        ci = tuple(np.percentile(d, [2.5, 97.5]))
    cut = np.median(t)
    return dict(n=len(x), mean=x.mean(), lo=ci[0], hi=ci[1], h1=x[t < cut].mean() if (t < cut).any() else np.nan,
                h2=x[t >= cut].mean() if (t >= cut).any() else np.nan)


def main():
    lines, worst = [], 0.0
    pooled = defaultdict(list)
    for name in FAMS:
        markets = json.load(open(V2 / f"{name}.json"))
        buckets = BUCKETS["15m" if name.endswith("15m") else "hourly"]
        orig = {(r["left"], tuple(r["bin"])): r for r in ORIG[name]["taker"]}
        cells = defaultdict(lambda: defaultdict(list))
        for m in markets:
            bars = sorted(m["bars"]); idx = {b[0]: i for i, b in enumerate(bars)}
            for left in buckets:
                want = m["close"] - 60 * left
                for variant in ("same", "ffill"):
                    if variant == "same":
                        i = idx.get(want)
                    else:
                        prev = [j for j, b in enumerate(bars) if b[0] <= want]
                        i = prev[-1] if prev else None
                    if i is None:
                        continue
                    bid, ask = bars[i][1], bars[i][2]
                    if bid is None or ask is None or not 0 < bid <= ask < 100 or ask - bid > 5:
                        continue
                    for yes in (True, False):
                        a = ask if yes else round(100 - bid, 1)
                        win = m["outcome"] if yes else 1 - m["outcome"]
                        for lo, hi in BINS:
                            if lo <= a < hi or (hi == 99 and a == 99):
                                cells[(left, (lo, hi))][variant].append((m["event"], m["close"], 100 * win - a - fee(a), a))
                                if variant == "same":
                                    nb = bars[i + 1] if i + 1 < len(bars) and bars[i + 1][0] == want + 60 else None
                                    if nb and nb[1] is not None and nb[2] is not None and 0 < nb[1] <= nb[2] < 100:
                                        a2 = nb[2] if yes else round(100 - nb[1], 1)
                                        cells[(left, (lo, hi))]["late1"].append((m["event"], m["close"], 100 * win - a2 - fee(a2), a2))
        lines.append(f"== {name}: left bin | original n net ci halves | mine n net% ci halves | late1 n net% ci | ffill n net% ci")
        for (left, b), v in sorted(cells.items(), key=lambda kv: (-kv[0][0], kv[0][1])):
            def f(key):
                rows = v.get(key)
                if not rows:
                    return "n=0"
                s = ev_stats([r[:3] for r in rows]); ask = np.mean([r[3] for r in rows])
                return (f"n={s['n']:>4} {s['mean']:+5.2f}c ({100 * s['mean'] / ask:+5.2f}%) [{s['lo']:+5.2f},{s['hi']:+5.2f}] "
                        f"h {s['h1']:+5.2f}/{s['h2']:+5.2f}")
            o = orig.get((left, b))
            mine = ev_stats([r[:3] for r in v["same"]]) if v.get("same") else None
            if o and mine:
                worst = max(worst, abs(o["net"] - mine["mean"]))
            ot = f"n={o['events']:>4} {o['net']:+5.2f}c {o['ci']} {o['halves']} {'PASS' if o['passed'] else ''}" if o else "absent"
            lines.append(f"  {left:>3}m {b[0]}-{b[1]} | {ot} | {f('same')} | late1 {f('late1')} | ffill {f('ffill')}")
            if name.endswith("15m"):
                for key in ("same", "late1"):
                    pooled[(left, b, key)] += [(str(c), c, val) for e, c, val, a in v.get(key, [])]
    lines.append("== pooled crypto 15-minute, clustered by clock window: same-bar ask vs one bar late")
    for (left, b, key), rows in sorted(pooled.items(), key=lambda kv: (-kv[0][0], kv[0][1], kv[0][2])):
        s = ev_stats(rows)
        lines.append(f"  {left:>3}m {b[0]}-{b[1]} {key:5} n={s['n']:>5} {s['mean']:+5.2f}c [{s['lo']:+5.2f},{s['hi']:+5.2f}] h {s['h1']:+5.2f}/{s['h2']:+5.2f}")
    lines.insert(0, f"largest |original net - my net| over all matched cells: {worst:.3f}c")
    OUT.write_text("\n".join(lines) + "\n"); print("\n".join(lines))


main()
