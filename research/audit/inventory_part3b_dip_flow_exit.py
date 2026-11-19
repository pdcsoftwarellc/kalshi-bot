"""Audit helper (read-only): is the 'did not recover' signal in dip_flow an untested EXIT rule?
From data/processed/favorite-dip/flow-rows.json (one filled 80c bid per match). For the fills flagged by a signal known
N minutes after the fill, compare (a) holding to settlement with (b) selling at the last traded price at that time.
(b) is a ROUGH proxy: last print, not the bid; shown with no fee and with the taker fee 0.07*p*(1-p) plus 1c of spread.
COMPUTED here; 95% intervals resample matches (4000 draws). % is of the 80c stake."""
import json
from pathlib import Path
import numpy as np
ROOT = Path(__file__).resolve().parents[2]
rows = [r for r in json.load(open(ROOT/"data/processed/favorite-dip/flow-rows.json")) if r.get("found")]
rng = np.random.default_rng(5)
fee = lambda p: 7*p/100*(1-p/100)


def ci(x):
    x = np.asarray(x, float)
    m = x[rng.integers(0, len(x), (4000, len(x)))].mean(axis=1)
    return f"{x.mean():+.1f}% [{np.percentile(m, 2.5):+.1f},{np.percentile(m, 97.5):+.1f}]"


SIGNALS = (("no print at 83c+ within 30 min -> sell at the 30-min price", lambda r: r["seconds_to_83"] is None, "price_after_30m"),
           ("price 5 min after the fill < 80c -> sell at the 5-min price", lambda r: r["price_after_5m"] < 80, "price_after_5m"),
           ("price 5 min after the fill < 70c -> sell at the 5-min price", lambda r: r["price_after_5m"] < 70, "price_after_5m"))
GROUPS = (("all tours", None), ("Challenger+WTA+WTA Challenger", ("Challenger", "WTA", "WTA Challenger")), ("ATP+ITF men", ("ATP", "ITF men")))
for label, test, key in SIGNALS:
    print(label)
    for name, tours in GROUPS:
        for period in ("holdout", "recent", "both"):
            part = [r for r in rows if (tours is None or r["tour"] in tours) and period in (r["period"], "both")]
            hit = [r for r in part if test(r)]
            if len(hit) < 10:
                print(f"  {name:30} {period:7} flagged {len(hit)} of {len(part)}: too few")
                continue
            hold = [100*(100*r["payout"]-80)/80 for r in hit]
            sell = [100*(r[key]-80)/80 for r in hit]
            sell_fee = [100*(r[key]-1-fee(r[key]-1)-80)/80 for r in hit]
            diff = [b-a for a, b in zip(hold, sell_fee)]
            whole_hold = [100*(100*r["payout"]-80)/80 for r in part]
            whole_rule = [100*((r[key]-1-fee(r[key]-1)) if test(r) else 100*r["payout"])/80-100 for r in part]
            print(f"  {name:30} {period:7} flagged {len(hit):3} of {len(part):3} | win rate {np.mean([r['payout'] for r in hit]):.3f} mean exit price {np.mean([r[key] for r in hit]):.1f}c "
                  f"| hold {ci(hold)} | sell no fee {ci(sell)} | sell -1c -taker fee {ci(sell_fee)} | sell minus hold {ci(diff)} "
                  f"| WHOLE RULE all fills: hold {ci(whole_hold)} with exit {ci(whole_rule)} diff {ci(np.array(whole_rule)-np.array(whole_hold))}")
