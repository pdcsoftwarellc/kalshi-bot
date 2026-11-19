"""Day-clustered intervals for calibration_map's non-sports 'close' families: buy the 90-99c dearer side at the ask.

Cost = midpoint + half the warm-up spread + 0.07 p (1-p) (the study's 'net@ask'); return = net / cost. Cluster = family + UTC day of close.
"""
import json, glob, sys
from statistics import median
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
from inv2_common import boot, halves, fmt
D = Path(__file__).resolve().parents[2]/"data/processed/calibration-map"


def pct(r):
    ask = (r["start"]+r["spread"]/2)/100
    cost = ask+.07*ask*(1-ask)
    return 100*(r["payout"]-cost)/cost


for f in sorted(glob.glob(str(D/"close-rows-*.json"))):
    rows = json.load(open(f))
    for hours in sorted({r["hours"] for r in rows}):
        hit = [r for r in rows if r["hours"] == hours and 90 <= r["start"] < 99 and r["payout"] in (0, 1, 0.0, 1.0)]
        if len(hit) < 10:
            continue
        day = lambda r: r["close"]//86400
        b = boot(hit, day, pct)
        h = halves(hit, day, lambda r: r["close"], pct)
        days = len({day(r) for r in hit})
        span = (max(r["close"] for r in hit)-min(r["close"] for r in hit))/86400/30 or 1
        print(f'{hit[0]["family"]:24} @{hours}h price {sum(r["start"] for r in hit)/len(hit):.1f} losses {sum(r["payout"] < .5 for r in hit)} '
              f'warm-vol median {median(r["warm_volume"] for r in hit):.0f} mkt-vol median {median(r["market_volume"] for r in hit):.0f} rows/month {len(hit)/span:.0f} days {days} | {fmt(b, h)}')
