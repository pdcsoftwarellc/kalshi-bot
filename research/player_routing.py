"""Paired execution test for the same ATP match-winner signal.

At confirmed set checkpoints, try the original ticker or take the cheaper
representation of the same player winning under ordinary match settlement.
Both arms use the same original signal times. Price ties stay on the original.
"""
from collections import Counter
from datetime import date
from decimal import Decimal
from heapq import heappop, heappush
import json
from pathlib import Path
import sys

import httpx

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from research.api_cost import PERIODS, ROOT, SERIES, fee, fetch, pages, save, timestamp
from research.set_frequency import MODEL, collect_forecasts, competitor, evaluate_records, groups, print_hit, quoted_close
from tennis.engine import PredictionEngine

D = Decimal
OUT = ROOT / "data/processed/parallel-round3/player_routing"
NATIVE = ROOT / "data/processed/native-atp/set-frequency"
CHECK = (date(2026, 9, 22), date(2026, 10, 1))


def comparable_pair(pair, event):
    if len(pair) != 2 or not event.get("mutually_exclusive") or event.get("series_ticker") != SERIES:
        return False
    named_rules = []
    for m in pair:
        name = m.get("yes_sub_title")
        if (not name or m.get("market_type") != "binary" or D(m.get("notional_value_dollars") or 0) != 1
                or m["event_ticker"] != event["event_ticker"] or not competitor(m)):
            return False
        prefix = "If " + name + " wins "
        if not m.get("rules_primary","").startswith(prefix):
            return False
        named_rules.append((m["rules_primary"][len(prefix):], m.get("rules_secondary") or ""))
    return named_rules[0] == named_rules[1] and competitor(pair[0]) != competitor(pair[1])


def pick_route(original_side, limit, other_quote):
    """Choose solely on fee-inclusive quoted cost; express price in dollars."""
    if other_quote is None:
        return False, original_side, limit
    bid, ask = other_quote
    if not 0 < bid <= ask < 100:
        return False, original_side, limit
    alternate_side = "no" if original_side == "yes" else "yes"
    alternate = D(100-bid if alternate_side == "no" else ask) / 100
    if alternate+fee(alternate) < limit+fee(limit):
        return True, alternate_side, alternate
    return False, original_side, limit


def account(client, orders, frames, use_other, initial_cash):
    cash = initial_cash
    locked = D(0)
    releases = []
    rows = []
    for order in sorted(orders, key=lambda r: (r["t"],r["ticker"])):
        t = order["t"]
        while releases and releases[0][0] <= t:
            _,returned,cost = heappop(releases)
            cash += returned
            locked -= cost
        original, other, quote = frames[order["ticker"],t]
        initial_limit = D(order["limit"])
        changed, side, limit = pick_route(order["side"], initial_limit, quote if use_other else None)
        market = other if changed else original
        cost = limit+fee(limit)
        row = dict(fixture=order["fixture"], set_number=order["set_number"], ts=t,
                   original=original["ticker"], ticker=market["ticker"], other_quote=quote,
                   switched=changed, side=side, original_limit=str(initial_limit), limit=str(limit),
                   cost=str(cost), quoted_saving=str(initial_limit+fee(initial_limit)-cost))
        rows.append(row)
        if cash < cost or initial_cash-locked < cost:
            row["decision"] = "cash_blocked"
            continue
        cash -= cost
        locked += cost
        prints = pages(client,"/markets/trades","trades",ticker=market["ticker"],limit=1000,min_ts=t+2,max_ts=t+5)
        hit = print_hit(prints,side,limit,t,5)
        if hit is None:
            row["decision"] = "unfilled"
            heappush(releases,(t+5,cost,cost))
            continue
        paid = D(market["settlement_value_dollars"])
        settled = timestamp(market["settlement_ts"])
        if not 0 <= paid <= 1 or settled <= hit["ts"]:
            raise ValueError("Invalid payout for a filled route")
        proceeds = paid if side == "yes" else 1-paid
        heappush(releases,(settled,proceeds,cost))
        row.update(decision="paper_fill",proof=hit,settled=settled,payout=str(proceeds),
                   fee=str(fee(limit)),pnl=str(proceeds-cost))
    fills = [row for row in rows if "pnl" in row]
    return rows, dict(attempts=len(rows), both_books=sum(row["other_quote"] is not None for row in rows),
                      switches=sum(row["switched"] for row in rows),fills=len(fills),
                      fees=str(sum((D(row["fee"]) for row in fills),D(0))),
                      pnl=str(sum((D(row["pnl"]) for row in fills),D(0))))


def run(client, start, end, settings):
    path = NATIVE/f"{start}.json"
    if path.is_file():
        previous = json.loads(path.read_text())
        records = previous["ledger"]
        control = previous["summary"]
    else:
        engine = PredictionEngine.load(str(MODEL),n_mc_samples=200)
        records, counts = collect_forecasts(client,engine,start,end,settings)
        control = evaluate_records(client,records,start,end,settings,counts)
    orders = [r for r in records if r["decision"] in ("paper_fill","unfilled")]
    by_event = groups(client,SERIES,start,end)
    frames = {}
    exclusions = Counter()
    for order in orders:
        ticker,t = order["ticker"],order["t"]
        event_name = ticker.rsplit("-",1)[0]
        pair = by_event[event_name]
        original = pair[ticker]
        frames[ticker,t] = original, None, None
        response = fetch(client,"/events/"+event_name,{})
        if ({m["ticker"] for m in response.get("markets",[])} != set(pair)
                or not comparable_pair(list(pair.values()),response["event"])):
            exclusions["ambiguous_contracts"] += 1
            continue
        other = next(m for m in pair.values() if m["ticker"] != ticker)
        if not timestamp(other["open_time"]) <= t < timestamp(other["close_time"])-5:
            exclusions["other_book_closed"] += 1
            continue
        q = quoted_close(client,other["ticker"],t)
        if q is None:
            exclusions["no_same_minute_close_on_other_book"] += 1
        frames[ticker,t] = original, other, q
    direct, original = account(client,orders,frames,False,D(str(settings["bankroll"])))
    routed, improved = account(client,orders,frames,True,D(str(settings["bankroll"])))
    if D(original["pnl"]) != D(control["net_trading_dollars"]) or original["fills"] != control["fills"]:
        raise ValueError("Control failed to reproduce the native study")
    if any(row["decision"] != source["decision"] for row,source in zip(direct,sorted(orders,key=lambda x:(x["t"],x["ticker"])),strict=True)):
        raise ValueError("Control differs at the order level")
    result = dict(start=str(start),end_exclusive=str(end),subset="all original qualifying signals",
                  total_forecasts=len(records),control=original,routed=improved, exclusions=dict(exclusions),
                  incremental_pnl=str(D(improved["pnl"])-D(original["pnl"])),
                  quoted_savings_if_all_fill=str(sum((D(r["quoted_saving"]) for r in routed),D(0))))
    save(OUT/f"{start}-ledger.json",dict(summary=result,control=direct,routed=routed))
    print(json.dumps(result),flush=True)
    return result


def main():
    settings = json.loads((NATIVE/"summary.json").read_text())["settings"]
    if (settings["per_signal"], settings["latency_ms"], settings["max_age_ms"]) != (1,2000,5000):
        raise ValueError("Expected previous one-contract, two-second-latency and five-second-expiry settings")
    with httpx.Client(timeout=40) as client:
        results = [run(client,start,end,settings) for start,end in [*PERIODS,CHECK]]
    save(OUT/"summary.json",dict(settings=settings,periods=results))


if __name__ == "__main__":
    main()
