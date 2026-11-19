"""Audit helper (read-only): bootstrap by match of data/processed/dip-scores/dips.json.
Same row filter as dip_scores.summarise (not mid_match reference, depth >= 8c, in play), outcome known.
Per-bet return = (100*won - level) / level, no fee (as the script; these are quote-based dips, not proven fills).
COMPUTED here: mean % per bet, 95% bootstrap interval resampling events (4000 draws), chronological halves by dip time."""
import json
from collections import defaultdict
from pathlib import Path
import numpy as np
ROOT = Path(__file__).resolve().parents[2]
rows = json.load(open(ROOT/"data/processed/dip-scores/dips.json"))
rows = [d for d in rows if d["reference_kind"] != "mid_match" and d["depth"] >= 8 and d["in_play"]]
print("qualifying dips", len(rows), "with outcome", sum(d["favourite_won"] is not None for d in rows),
      "events", len({d["event"] for d in rows}), "first", min(d["time"] for d in rows), "last", max(d["time"] for d in rows))
rng = np.random.default_rng(7)


def stat(sel, label):
    sel = sorted((d for d in sel if d["favourite_won"] is not None), key=lambda d: d["ts"])
    if not sel:
        print(f"{label:60} no settled dips")
        return
    ev = defaultdict(list)
    for d in sel:
        ev[d["event"]].append(100*(100*d["favourite_won"]-d["level"])/d["level"])
    groups = [np.array(v) for v in ev.values()]
    allv = np.concatenate(groups)
    if len(groups) >= 5:
        means = []
        for _ in range(4000):
            pick = rng.integers(0, len(groups), len(groups))
            means.append(np.concatenate([groups[i] for i in pick]).mean())
        ci = f"[{np.percentile(means, 2.5):+.1f},{np.percentile(means, 97.5):+.1f}]"
    else:
        ci = "too few events for an interval"
    r = [100*(100*d["favourite_won"]-d["level"])/d["level"] for d in sel]
    h = len(r)//2
    halves = f"{np.mean(r[:h]):+.1f}% (n={h}) / {np.mean(r[h:]):+.1f}% (n={len(r)-h})" if h else "n/a"
    print(f"{label:60} dips {len(sel):3} events {len(groups):3} won {sum(d['favourite_won'] for d in sel):3} mean level {np.mean([d['level'] for d in sel]):.1f}c "
          f"mean {allv.mean():+.1f}% ci {ci} halves {halves}")


for b in ("90c+", "80-90c", "under 80c"):
    for lv in ("80", "70", "65", "pre-10"):
        sel = [d for d in rows if d["bucket"] == b and d["level_name"] == lv]
        if sel:
            stat(sel, f"fav {b} level {lv} ALL")
for lv in ("pre-10",):
    for key in ("sets_situation", "break_situation"):
        for sit in sorted({d["score"][key] for d in rows}):
            stat([d for d in rows if d["level_name"] == lv and d["score"][key] == sit], f"all favourites level {lv} | {sit}")
stat([d for d in rows if d["level_name"] == "pre-10"], "all favourites level pre-10 ALL")
