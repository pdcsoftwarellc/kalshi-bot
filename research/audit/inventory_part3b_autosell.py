"""Audit helper (read-only): condense data/processed/autosell/summary.json to one line per
sample x family x bucket x entry mode, converting cents per contract to % of the mean entry price.
Nothing is recomputed: every number is copied from summary.json and divided by mean_entry."""
import json
from pathlib import Path
ROOT = Path(__file__).resolve().parents[2]
s = json.load(open(ROOT/"data/processed/autosell/summary.json"))
for name, v in s.items():
    print("==", name, {k: v[k] for k in v if k != "table"})
    groups = {}
    for k, c in v["table"].items():
        a = c["all"]
        if "trades" in a:
            if not a["trades"]:
                continue
            fam, bucket, mode = k.split()[0], k.split()[1], k.split()[2]
            groups.setdefault((fam, bucket, mode), []).append((k, c))
        else:
            lvl = int(k.split()[-1][:-1])
            pc = lambda x: 100*x/lvl
            print(f"  {k:36} matches {a['matches']} (E {c['early']['matches']}/L {c['late']['matches']}) both {a['both']:.2f} one {a['one']:.2f} "
                  f"per-match {a['per_match']:+.2f}c ci {a['ci95']} = {pc(a['per_match']):+.1f}% of one {lvl}c stake [{pc(a['ci95'][0]):+.1f},{pc(a['ci95'][1]):+.1f}] "
                  f"E {c['early']['per_match']:+.2f}c L {c['late']['per_match']:+.2f}c all-taker {a['per_match_all_taker']:+.2f}c /mo {c['matches_per_month']:.0f}")
    for (fam, bucket, mode), cells in groups.items():
        pct = lambda cell, part: 100*cell[part]["net"]/cell[part]["mean_entry"]
        cells.sort(key=lambda kc: pct(kc[1], "all"))
        for tag, (k, c) in (("worst", cells[0]), ("best", cells[-1])):
            a, e = c["all"], c["all"]["mean_entry"]
            ci = a["ci95"]
            citxt = "n<10" if ci is None else f"[{100*ci[0]/e:+.1f},{100*ci[1]/e:+.1f}]"
            half = lambda part: (f"{pct(c, part):+.1f}%(n={c[part]['trades']})" if c[part].get("trades") else "n=0")
            vh = a["vs_hold_ci95"]
            print(f"  {fam} {bucket} {mode} {tag:5} {k.split('sell ')[1]:5} matches {a['matches']} fills {a['trades']} entry {e:.1f}c net {a['net']:+.2f}c = {pct(c,'all'):+.1f}% {citxt} "
                  f"E {half('early')} L {half('late')} exit {a['exit_rate']:.2f} hold {a['hold_net']:+.2f}c {a['hold_ci95']} vs-hold {a['vs_hold']:+.2f}c {vh} all-taker {a['net_all_taker']:+.2f}c /mo {c['trades_per_month']:.0f}")
