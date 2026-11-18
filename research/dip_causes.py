"""What kind of dip pays in the tennis dip rule, and would a bid set relative to the pre-match price do better?

Kalshi keeps no past scores, so the "cause" of a dip is read from the price
path alone: how long after play began the 80c bid filled, how fast the price
fell into it, how far the fill bar traded through it, and the pre-match price.
Splits use the holdout archive (before 2026-09-18) and the recent window
separately. Every split is descriptive: none was chosen in advance.
"""
from collections import defaultdict
from datetime import datetime, timezone
from statistics import mean, median
import sys
from pathlib import Path

import httpx

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from research.api_cost import ROOT, save
from research.favorite_dip import GROUPS, WARMUP, interval, matches
from research.near_end import cents

OUT = ROOT / "data/processed/favorite-dip"
OFFSETS = (8, 10, 12, 15, 20, 25)


def study(job, bars):
    """Pre-match price, and for each bid level the features of its fill (None when unfilled)."""
    event, ticker, settle, outcome = job
    mids = []
    for b in bars[:WARMUP]:
        bid, ask = cents(b, "yes_bid", "close"), cents(b, "yes_ask", "close")
        if bid is None or ask is None or not 0 < bid <= ask < 100 or ask-bid > 4:
            return None
        mids.append((bid+ask)/2)
    if len(mids) < WARMUP or max(mids)-min(mids) > 3:
        return None
    yes = median(mids) > 50
    start = median(mids) if yes else 100-median(mids)
    if start < 85:
        return None
    path = []                                  # favourite-side (time, low print, mid) per bar after the warm-up
    for b in bars[WARMUP:]:
        low = cents(b, "price", "low") if yes else (None if cents(b, "price", "high") is None else 100-cents(b, "price", "high"))
        bid, ask = cents(b, "yes_bid", "close"), cents(b, "yes_ask", "close")
        mid = None if bid is None or ask is None else ((bid+ask)/2 if yes else 100-(bid+ask)/2)
        path.append((b["end_period_ts"], low, mid))
    began = next((i for i, (_, _, mid) in enumerate(path) if mid is not None and abs(mid-start) > 2), None)
    fills = {}
    for limit in {80, *(round(start)-k for k in OFFSETS)}:
        i = next((i for i, (_, low, _) in enumerate(path) if low is not None and low < limit), None)
        if i is None:
            fills[limit] = None
            continue
        before = next((mid for _, _, mid in reversed(path[max(0, i-5):i]) if mid is not None), None)
        fills[limit] = dict(minutes_in=None if began is None else i-began, mid_5_min_before=before,
                            through=limit-path[i][1], hour=datetime.fromtimestamp(path[i][0], timezone.utc).hour)
    return dict(event=event, series=ticker.split("-")[0], settle=settle, start=start,
                payout=outcome if yes else 1-outcome, fills=fills)


def line(name, rows, limit=None):
    nets = [100*r["payout"]-(limit or r["limit"]) for r in rows]
    if not nets:
        return None
    ci = interval(nets)
    print(f'  {name:26} fills {len(nets):4}  won {mean(r["payout"] for r in rows):.1%}  net {mean(nets):+6.2f}c  ci {ci and [round(v, 1) for v in ci]}')
    return dict(fills=len(nets), win_rate=mean(r["payout"] for r in rows), net_cents=mean(nets), ci95=ci)


def main():
    data = defaultdict(list)
    with httpx.Client(timeout=40) as client:
        for archived in (True, False):
            for series in GROUPS["tennis"]:
                for job, bars in matches(client, series, archived):
                    row = study(job, bars)
                    if row:
                        data["holdout" if archived else "recent"].append(row)
    save(OUT/"causes-rows.json", data)
    report = {}
    cuts = {
        "pre-match price": lambda r, f: "90-92c" if r["start"] < 93 else "93-95c" if r["start"] < 96 else "96c+",
        "minutes of play before fill": lambda r, f: "unknown" if f["minutes_in"] is None else "under 20" if f["minutes_in"] < 20
        else "20-45" if f["minutes_in"] < 45 else "45-90" if f["minutes_in"] < 90 else "90+",
        "price 5 minutes before fill": lambda r, f: "unknown" if f["mid_5_min_before"] is None else "88c+ (sudden)"
        if f["mid_5_min_before"] >= 88 else "84-88c" if f["mid_5_min_before"] >= 84 else "under 84c (gradual)",
        "traded through the bid by": lambda r, f: "1-2c" if f["through"] <= 2 else "3-9c" if f["through"] <= 9 else "10c+",
        "tour": lambda r, f: r["series"],
    }
    for period, rows in data.items():
        heavy = [r for r in rows if r["start"] >= 90 and r["series"] != "KXATPMATCH"]
        filled = [(r, r["fills"][80]) for r in heavy if r["fills"].get(80)]
        print(f"\n=== {period}: {len(heavy)} WTA and Challenger favourites at 90c+, {len(filled)} filled at 80c")
        for name, key in cuts.items():
            print(f" by {name}")
            groups = defaultdict(list)
            for r, f in filled:
                groups[key(r, f)].append(r)
            for k in sorted(groups):
                report[f"{period} | {name} | {k}"] = line(k, groups[k], 80)
        print(" bid set relative to the pre-match price (profit per order placed in brackets)")
        for k in OFFSETS:
            hit = [dict(r, limit=round(r["start"])-k) for r in heavy if r["fills"].get(round(r["start"])-k)]
            got = line(f"{k}c below", hit)
            if got:
                got["per_order_cents"] = got["net_cents"]*got["fills"]/len(heavy)
                print(f'  {"":26} [{got["per_order_cents"]:+.3f}c per order]')
                report[f"{period} | relative bid | {k}c below"] = got
        flat = [r for r in heavy if r["fills"].get(80)]
        print(f'  flat 80c bid for comparison   [{sum(100*r["payout"]-80 for r in flat)/len(heavy):+.3f}c per order]')
        mid = [r for r in rows if 85 <= r["start"] < 90 and r["series"] != "KXATPMATCH"]
        print(f" favourites at 85-90c before play ({len(mid)} matches)")
        for k in (10, 12, 15, 20):
            line(f"{k}c below", [dict(r, limit=round(r["start"])-k) for r in mid if r["fills"].get(round(r["start"])-k)])
    save(OUT/"causes.json", report)


if __name__ == "__main__":
    main()
