"""Audit step 3a summary: what the original method reports with no plant and with +1.5% / +3% planted."""
import json
from collections import defaultdict
from pathlib import Path
A = Path(__file__).resolve().parents[2] / "data/processed/audit/short_dated"
RUNS = [("none", "plant-price-0"), ("price +1.5%", "plant-price-1.5"), ("price +3%", "plant-price-3"),
        ("outcome +1.5%", "plant-outcome-1.5"), ("outcome +3%", "plant-outcome-3")]
L = []
def group(name):
    return ("crypto 15m" if name.endswith("15m") else "crypto hourly" if name in ("btc_hourly", "eth_hourly") else
            "index hourly" if name.endswith("hourly") else "daily")
data = {label: json.load(open(A / d / "v2-calibration.json")) for label, d in RUNS if (A / d / "v2-calibration.json").exists()}
base = data["none"]["families"]
L.append("SECOND STUDY (calibrate): share of taker cells (>=30 events) that pass the study's own gate (interval > 0, both halves > 0)")
L.append(f"  {'plant':14} {'group':14} {'bin':7} cells  pass  bonf(t>=3.5)  median half-width of ci (c)  mean realised shift (c)")
for label, rep in data.items():
    agg = defaultdict(lambda: [0, 0, 0, [], []])
    for name, fam in rep["families"].items():
        if name == "pooled":
            continue
        b0 = {(r["left"], tuple(r["bin"])): r for r in base[name]["taker"]}
        for r in fam["taker"]:
            if r["events"] < 30 or not r["ci"]:
                continue
            for key in ((group(name), f"{r['bin'][0]}-{r['bin'][1]}"), (group(name), "all"), ("ALL", "all")):
                a = agg[key]; a[0] += 1; a[1] += r["passed"]; a[2] += r["t"] >= 3.5
                a[3].append((r["ci"][1] - r["ci"][0]) / 2); a[4].append(r["net"] - b0[(r["left"], tuple(r["bin"]))]["net"])
    for (g, b), a in sorted(agg.items()):
        med = sorted(a[3])[len(a[3]) // 2]
        L.append(f"  {label:14} {g:14} {b:7} {a[0]:5} {a[1]:5} {a[2]:5}         {med:5.2f}                        {sum(a[4]) / len(a[4]):+5.2f}")
L.append("")
L.append("SECOND STUDY pooled rows (the largest samples the method has): net c, ci, halves, t, PASS")
want = [("crypto 15-minute", "90-99c"), ("crypto 15-minute", "97-99c"), ("crypto 15-minute", "90-95c"), ("crypto 15-minute", "80-90c"),
        ("crypto hourly", "90-99c"), ("crypto hourly", "80-90c"), ("index and FX hourly", "90-99c"), ("index and FX hourly", "80-90c"),
        ("crypto daily", "90-99c"), ("index and FX daily", "90-99c"), ("btc15m", "90-99c"), ("btc_hourly", "90-99c"),
        ("sp_hourly", "90-99c"), ("nasdaq_hourly", "90-99c"), ("crypto 15-minute 5m fav bid", "90-95c"), ("crypto hourly 30m fav bid", "90-95c")]
for who, band in want:
    for label, rep in data.items():
        row = next((r for r in rep["families"]["pooled"] if r["who"] == who and r["band"] == band), None)
        if row:
            L.append(f"  {who:28} {band:7} {label:14} ev {row['events']:5} ask {row['ask']:6.2f} net {row['net']:+6.2f}c ({100 * row['net'] / row['ask']:+5.2f}%) "
                     f"ci {str(row['ci']):16} halves {str(row['halves']):16} t {row['t']:+6.2f} {'PASS' if row['passed'] else ''}")
L.append("")
L.append("FIRST STUDY (analyze): plain rule (ask 97-99c, widest window), early 60% / late 40%; detect = late interval > 0 and early net > 0")
for fam in ("btc15m", "eth15m", "btc_hourly", "eth_hourly", "index_hourly"):
    for label, d in RUNS:
        p = A / d / f"{fam}.json"
        if not p.exists():
            continue
        rep = json.load(open(p)); e, l = rep["plain"]["clustered"]
        hit = bool(l.get("ci95") and l["ci95"][0] > 0 and e["net_cents"] > 0)
        ch = rep["chosen_price_rule"]; c = rep["price_rules"][ch]["one_per_event"] if ch else None
        chs = f"| chosen '{ch}' late n {c[1]['trades']} net {c[1]['net_cents']:+.2f} ci {c[1]['ci95']}" if c and c[1].get("trades") else ""
        L.append(f"  {fam:13} {label:14} early n {e['trades']:5} net {e['net_cents']:+.2f}c ci {e['ci95']} | late n {l['trades']:5} net {l['net_cents']:+.2f}c "
                 f"ci {l['ci95']} {'DETECTED' if hit else 'missed'} {chs}")
(A / "power_summary.txt").write_text("\n".join(L) + "\n"); print("\n".join(L))
