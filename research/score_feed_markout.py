"""Measure the residual price move after one independently received tennis point.

Each observation is a book markout. Overlapping observations are never summed
as an attainable portfolio return. Missing future quotes count against coverage.
"""
import argparse
import asyncio
from bisect import bisect_right
from collections import Counter, defaultdict
from decimal import Decimal as D
import json
from pathlib import Path
from statistics import mean, median
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from research.api_cost import ROOT, fee, save
from research.native_replay import convert
from tennis.engine import PredictionEngine

OUT = ROOT / "data/processed/native-atp/score-feed-markout"


def touch(q, yes):
    if not q or min(q["bid_size"], q["ask_size"]) < 1 or not 0 < q["yes_bid"] <= q["yes_ask"] < 100:
        return None
    if yes:
        return q["yes_bid"], q["yes_ask"]
    return 100-q["yes_ask"], 100-q["yes_bid"]


def observe(records, row, end_ms):
    quotes, times = records
    now = row["ts_ms"]
    def at(offset):
        target = now+offset
        i = bisect_right(times, target)-1
        if i < 0 or target > end_ms or target-times[i] > 5000:
            return None
        return touch(quotes[i], row["point_winner_yes"])
    origin = dict(yes_bid=row["bid"],yes_ask=row["ask"],bid_size=row["bid_size"],ask_size=row["ask_size"])
    before, entry, exit_quote = at(-5000), at(2000), at(30000)
    here = touch(origin, row["point_winner_yes"]) if row["book_age_ms"] is not None and row["book_age_ms"] <= 5000 else None
    if row["decoded_in_receipt"] != 1:
        return {"status":"batched_points"}
    if not here or not entry:
        return {"status":"unpriced_at_receipt_or_entry"}
    if not exit_quote:
        return {"status":"missing_exit_or_window_ended"}
    buy, sell = D(entry[1])/100, D(exit_quote[0])/100
    cost = fee(buy)+fee(sell)
    return dict(status="measured", entry=float(buy), exit=float(sell), fees=float(cost),
                pre_move_cents=mean(here)-mean(before) if before else None,
                post_move_cents=mean(exit_quote)-mean(here),
                gross_cents=float(100*(sell-buy)), net_cents=float(100*(sell-buy-cost)))


def summarize(rows):
    measured = [r for r in rows if r["status"]=="measured"]
    fixtures = {r["fixture"] for r in measured}
    out = dict(coverage=dict(Counter(r["status"] for r in rows)), distinct_measured_matches=len(fixtures),
               scored_windows=len(measured))
    if measured:
        out.update(mean_cents_after_fees=mean(r["net_cents"] for r in measured),
                   median_cents_after_fees=median(r["net_cents"] for r in measured),
                   profitable_windows=sum(r["net_cents"] > 0 for r in measured),
                   mean_gross_cents=mean(r["gross_cents"] for r in measured),
                   mean_point_winner_mid_move_after_receipt_cents=mean(r["post_move_cents"] for r in measured),
                   match_weighted_mean_net_cents=mean(mean(r["net_cents"] for r in measured if r["fixture"]==f) for f in fixtures))
    return out


async def study(tapes):
    model = PredictionEngine.load(str(ROOT/"data/artifacts/native-atp"))
    fixtures = {}
    for f in (ROOT/"data/raw/challenger-rebound").glob("fixtures-*.json"):
        fixtures.update({str(r["event_key"]):r for r in json.loads(f.read_text())})
    rows, seen = [], set()
    for i,tape in enumerate(tapes):
        with tape.open() as stream:
            start = json.loads(next(stream))
            last = None
            for line in stream:
                last = line
        end = json.loads(last)
        if end["kind"] != "end":
            raise ValueError(f"Incomplete tape: {tape}")
        end_ms = (start["utc_ns"]+end["mono_ns"]-start["mono_ns"])//1_000_000
        events, forecasts, _ = await convert(tape, model, fixtures)
        books = defaultdict(list)
        for e in events:
            if e["kind"]=="quote":
                books[e["ticker"]].append(e)
        index = {t:(qs,[q["ts_ms"] for q in qs]) for t,qs in books.items()}
        for row in forecasts:
            key = row["fixture"], row["points_observed"]
            if key in seen:
                continue
            seen.add(key)
            result = observe(index.get(row["ticker"],([],[])), row, end_ms)
            rows.append(dict(tape=tape.name, period="fresh" if i==len(tapes)-1 else "previous",
                             fixture=key[0], points_observed=key[1], ts_ms=row["ts_ms"],
                             point_winner_yes=row["point_winner_yes"], source=row["score_source"], **result))
    report={k:summarize([r for r in rows if r["period"]==k]) for k in ("previous","fresh")}
    report["all"]=summarize(rows)
    save(OUT/"point-windows.json",rows)
    save(OUT/"summary.json",report)
    print(json.dumps(report,indent=2))


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("historical",type=Path,nargs="+")
    parser.add_argument("--fresh",type=Path,required=True)
    args=parser.parse_args()
    if args.fresh in args.historical:
        parser.error("Fresh capture must be separate from the historical sample")
    asyncio.run(study([*args.historical,args.fresh]))
