"""Score research/favorites.py's taker rule on the part of its seeded sample the original run never fetched.

Bars come from data/processed/audit/inventory/favorites-holdout/bars (see favorites_holdout_download.py). The
decisions, the one-entry-per-event choice, the 60/40 split and the pass rule are the original's functions,
unmodified. Offline.
"""
import json, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from research import favorites as f, favorites_report as fr
from favorites_audit import line, pct, boot

SRC = ROOT/"data/processed/audit/inventory/favorites-holdout/bars"
OUT = ROOT/"data/processed/audit/inventory/favorites-holdout-audit.json"


def main():
    series = {s["ticker"]: s for s in fr.series_list(None)}
    picked = f.sample(f.references(f.load_universe()))
    rows, got, empty = [], 0, 0
    for m in picked:
        p = SRC/(m["ticker"]+".json")
        if not p.exists():
            continue
        bars = json.loads(p.read_text())
        if not bars:
            empty += 1
            continue
        got += 1
        rows += fr.decisions(m, bars, series[m["series"]])
    taken = fr.one_per_event([r for r in rows if r["net_cents"] is not None])
    orig = json.loads((ROOT/"data/processed/favorites/entries.json").read_text())
    seen = {r["event"] for r in orig}
    fresh = [r for r in taken if r["event"] not in seen]
    print(f"holdout markets with bars {got} (empty {empty}); decisions {len(rows)}; entries {len(taken)};"
          f" entries in events the original never entered {len(fresh)}")
    out = {}
    for label, base in (("HOLDOUT, every entry", taken), ("HOLDOUT, only events absent from the original entries", fresh),
                        ("ORIGINAL + HOLDOUT (one entry per event and rule, re-chosen over both)", fr.one_per_event(orig+taken))):
        print(f"\n{label}")
        cells = {"all rules": base}
        for b in ("90-94", "95-97", "98-99"):
            cells[f"band {b}"] = [r for r in base if r["band"] == b]
        for d in fr.DAYS:
            cells[f"D{d}"] = [r for r in base if r["days"] == d]
        cells["D1 90-94"] = [r for r in base if r["days"] == 1 and r["band"] == "90-94"]
        for c in ("Politics", "Climate and Weather", "Commodities", "Economics", "Mentions", "Entertainment", "Science and Technology"):
            cells[c] = [r for r in base if r["category"] == c]
        cells["liquid (vol>=100, spread<=3)"] = [r for r in base if r["volume_24h"] >= 100 and r["spread"] <= 3]
        out[label] = {}
        for k, v in cells.items():
            s = line(k, v)
            if s:
                s["losses"] = sum(r["payout"] < .5 for r in v)
            out[label][k] = s
        d1 = cells["D1 90-94"]
        print(f"   D1 90-94 full losses: {sum(r['payout'] < .5 for r in d1)} of {len(d1)}; script's own table verdicts that PASS:",
              [k for k, v in fr.rule_table(base).items() if v["passes"]])
    OUT.write_text(json.dumps(out, indent=1, default=str))
    print("\nsaved", OUT)


if __name__ == "__main__":
    main()
