"""Does an exit improve the tennis dip bid? Hold to settlement against take-profit and stop-loss exits.

Same entry as favorite_dip.py: a favourite at 90c+ before play, a resting 80c
bid, filled when a trade prints below it. After the fill, from the next bar:

  take-profit  a resting sell at T fills when a trade prints above T (no fee)
  stop-loss    once the favourite's bid closes a bar at or below S, sell at the
               next bar's opening bid and pay the taker fee

Results are split by tour and by period (the archive before 2026-09-18 is the
holdout; the recent window is where the rule was found).
"""
from collections import defaultdict
from statistics import mean
import sys
from pathlib import Path

import httpx

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from research.api_cost import ROOT, save
from research.favorite_dip import GROUPS, WARMUP, interval, matches, orders
from research.near_end import cents

OUT = ROOT / "data/processed/favorite-dip"
LIMIT = 80
TAKES, STOPS = (None, 88, 92, 96), (None, 65, 50, 35)


def path(job, bars):
    """Favourite-side (high print, closing bid, opening bid) per bar after the fill, or None if not filled."""
    row = next((r for r in orders(job, bars) if r["limit"] == LIMIT and r["start"] >= 90 and r["strict"]), None)
    if not row:
        return None
    mids = [(cents(b, "yes_bid", "close")+cents(b, "yes_ask", "close"))/2 for b in bars[:WARMUP]]
    yes = sorted(mids)[len(mids)//2] > 50
    fill = next(i for i, b in enumerate(bars) if i >= WARMUP and b["end_period_ts"] == job[2]-60*row["minutes_before_settlement"])

    def side(b, field, part):
        if yes:
            return cents(b, field, part)
        other = cents(b, {"yes_bid": "yes_ask", "price": "price"}[field], {"high": "low", "close": "close", "open": "open"}[part])
        return None if other is None else 100-other
    return row, [(side(b, "price", "high"), side(b, "yes_bid", "close"), side(b, "yes_bid", "open")) for b in bars[fill+1:]]


def result(row, after, take, stop):
    """Profit per contract in cents for one filled order under an exit plan."""
    for i, (high, bid, _) in enumerate(after):
        if take and high is not None and high > take:
            return take-LIMIT
        if stop and bid is not None and bid <= stop and i+1 < len(after) and after[i+1][2]:
            out = after[i+1][2]
            return out-LIMIT-7*out/100*(1-out/100)
    return 100*row["payout"]-LIMIT


def main():
    filled = defaultdict(list)
    with httpx.Client(timeout=40) as client:
        for archived in (True, False):
            for series in GROUPS["tennis"]:
                for job, bars in matches(client, series, archived):
                    found = path(job, bars)
                    if found:
                        filled[(series, "holdout" if archived else "recent")].append(found)
    report = {}
    groups = dict(filled)
    for period in ("holdout", "recent"):
        groups[("ALL TOURS", period)] = [f for (s, p), rows in filled.items() if p == period for f in rows]
    for (series, period), rows in sorted(groups.items()):
        print(f"\n{series} {period}: {len(rows)} fills")
        for take in TAKES:
            for stop in STOPS:
                nets = [result(row, after, take, stop) for row, after in rows]
                if not nets:
                    continue
                ci = interval(nets)
                name = f'take {take or "-"} stop {stop or "-"}'
                report[f"{series} {period} {name}"] = dict(fills=len(nets), net_cents=mean(nets), ci95=ci,
                                                           wins=sum(n > 0 for n in nets)/len(nets))
                print(f'  {name:18} net {mean(nets):+6.2f}c  won {sum(n > 0 for n in nets)/len(nets):.0%}  '
                      f'ci {ci and [round(v, 1) for v in ci]}')
    save(OUT/"exits.json", report)


if __name__ == "__main__":
    main()
