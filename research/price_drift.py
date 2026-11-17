"""Paper test of prematch ATP price momentum. Reads public market data only.

The same fixed thresholds apply to both periods. Times are UTC and a quote
means the latest completed one-minute candle close within five minutes.
"""
from collections import Counter, defaultdict
from datetime import date, datetime, timedelta
from decimal import Decimal
from pathlib import Path
import json
import sys

import httpx

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from research.api_cost import (
    PERIODS, ROOT, SERIES, STARTING_CASH, UTC, fee, fetch, history, ledger,
    link_market, midnight, pages, qualifying_print, save, timestamp,
)

D = Decimal
OUTPUT = ROOT / "data/processed/parallel-research/price_drift"
RULE = {
    "side": "direction of the 120-to-30-minute change in midpoint",
    "min_change_dollars": "0.05",
    "max_spread_dollars": "0.04",
    "max_quote_age_s": 300,
    "entry": "one contract at the later ask, first same-side non-block trade of size >= 1 at/below limit, 2 <= elapsed seconds < 300",
    "exit": "hold to published exchange settlement",
    "fee": "quadratic taker, multiplier 1, rounded up to whole cent per contract",
    "starting_cash_dollars": "25.00",
    "monthly_overhead_dollars": "80.00",
}


def read_quote(client, ticker, decision):
    candles = fetch(client, f"/series/{SERIES}/markets/{ticker}/candlesticks",
                    dict(start_ts=decision-300, end_ts=decision, period_interval=1))["candlesticks"]
    visible = [bar for bar in candles if decision-300 <= bar["end_period_ts"] <= decision]
    if not visible:
        return None
    bar = max(visible, key=lambda b: b["end_period_ts"])
    bid = bar.get("yes_bid", {}).get("close_dollars")
    ask = bar.get("yes_ask", {}).get("close_dollars")
    if bid is None or ask is None:
        return None
    bid, ask = D(bid), D(ask)
    if not 0 < bid <= ask < 1:
        return None
    return bid, ask, int(bar["end_period_ts"])


def signal(early, late):
    if any(q is None or q[1]-q[0] > D(".04") for q in (early, late)):
        return None
    move = (late[0] + late[1] - early[0] - early[1])/2
    if move >= D(".05"):
        return "yes", late[1], move
    if move <= D("-.05"):
        return "no", 1-late[0], move
    return None


def matched_events(client, fixtures, start, end, counts):
    by_day = defaultdict(list)
    for fixture in fixtures:
        by_day[date.fromisoformat(fixture["event_date"])].append(fixture)
    events = defaultdict(dict)
    for m in pages(client, "/markets", "markets", series_ticker=SERIES, status="settled",
                   limit=1000, min_close_ts=midnight(start), max_close_ts=midnight(end+timedelta(days=7))):
        events[m["event_ticker"]][m["ticker"]] = m

    counts["events_in_close_time_query_including_seven_day_buffer"] = len(events)
    by_fixture = defaultdict(list)
    for event, pair_map in sorted(events.items()):
        pair = sorted(pair_map.values(), key=lambda m:m["ticker"])
        if len(pair) != 2 or any(not m.get("occurrence_datetime") for m in pair):
            counts["unpaired_or_missing_exchange_schedule_in_buffer"] += 1
            continue
        planned = min(timestamp(m["occurrence_datetime"]) for m in pair)
        day = datetime.fromtimestamp(planned, UTC).date()
        nearby = [f for offset in (-1,0,1) for f in by_day[day+timedelta(days=offset)]]
        match = link_market(pair, nearby)
        if not match or not match[0].get("event_time"):
            counts["no_unique_fixture_or_provider_time_in_buffer"] += 1
            continue
        f = match[0]
        provider = timestamp(f["event_date"]+"T"+f["event_time"]+"+00:00")
        reference = int(min(planned, provider))
        decision = reference-1800
        if not midnight(start) <= decision < midnight(end):
            counts["matched_outside_decision_period"] += 1
            continue
        by_fixture[str(f["event_key"])].append((event, pair[0], int(planned), int(provider), reference))
    for fixture_id, rows in sorted(by_fixture.items()):
        if len(rows) != 1:
            counts["duplicate_events_per_fixture"] += len(rows)
            continue
        yield fixture_id, *rows[0]


def run_period(client, fixtures, start, end):
    counts = Counter()
    observations = []
    orders = []
    for fixture, event, market, kalshi_start, provider_start, reference in matched_events(client, fixtures, start, end, counts):
        counts["matched_events_in_period"] += 1
        early_ts = reference - 7200
        t = reference - 1800
        ticker = market["ticker"]
        if timestamp(market["open_time"]) > early_ts:
            counts["opened_after_first_observation"] += 1
            continue
        if timestamp(market["close_time"]) <= t+300:
            counts["closed_before_order_expiry"] += 1
            continue
        settlement = market.get("settlement_value_dollars")
        settle_time = market.get("settlement_ts")
        if settlement is None or not settle_time or timestamp(settle_time) <= t+300:
            counts["missing_or_early_settlement"] += 1
            continue
        value = D(settlement)
        if not 0 <= value <= 1:
            raise ValueError("Settlement outside $0..$1 for " + ticker)
        early = read_quote(client, ticker, early_ts)
        if early is None:
            counts["missing_early_quote"] += 1
            continue
        if early[1]-early[0] > D(".04"):
            counts["wide_early_spread"] += 1
            continue
        late = read_quote(client, ticker, t)
        if late is None:
            counts["missing_late_quote"] += 1
            continue
        if late[1]-late[0] > D(".04"):
            counts["wide_late_spread"] += 1
            continue
        counts["eligible_quote_pairs"] += 1
        row = dict(fixture=fixture, event=event, ticker=ticker, ts=t, kalshi_start=kalshi_start,
                   provider_start=provider_start, early_target_ts=early_ts, early_quote_ts=early[2],
                   late_quote_ts=late[2], early_bid=str(early[0]), early_ask=str(early[1]),
                   late_bid=str(late[0]), late_ask=str(late[1]))
        observations.append(row)
        trade = signal(early, late)
        if trade is None:
            counts["change_below_threshold"] += 1
            continue
        side, limit, move = trade
        prints = pages(client, "/markets/trades", "trades", ticker=ticker, min_ts=t+2, max_ts=t+300, limit=1000)
        hit = qualifying_print(prints, side, limit, t)
        orders.append(dict(row, side=side, change_yes_mid=str(move), limit=str(limit), fee=str(fee(limit)),
                           payout=str(value if side=="yes" else 1-value), settlement_ts=timestamp(settle_time),
                           print_ts=hit[0] if hit else None, trade_id=hit[1] if hit else None))

    fills, totals = ledger(orders, STARTING_CASH)
    pnls = [D(r["pnl"]) for r in fills]
    bill = D(80)*(end-start).days/((start.replace(day=28)+timedelta(days=4)).replace(day=1)-start).days
    summary = dict(start=str(start), end_exclusive=str(end), coverage=dict(counts),
                   signals=len(orders), **totals, gross_trading_pnl_dollars=float(sum(pnls,D(0))+sum((D(r["fee"]) for r in fills),D(0))),
                   winning_fills=sum(p>0 for p in pnls), api_overhead_dollars=float(bill),
                   net_after_subscription_dollars=float(sum(pnls,D(0))-bill),
                   all_signals_filled_net_dollars=float(sum((D(r["payout"])-D(r["limit"])-D(r["fee"]) for r in orders),D(0))),
                   all_signals_filled_and_won_ceiling_dollars=float(sum((1-D(r["limit"])-D(r["fee"]) for r in orders),D(0))))
    moves = [(D(r["late_bid"])+D(r["late_ask"])-D(r["early_bid"])-D(r["early_ask"]))/2
             for r in observations]
    summary["unchanged_midpoints"] = sum(m == 0 for m in moves)
    summary["max_abs_midpoint_move_dollars"] = float(max(map(abs, moves))) if moves else None
    save(OUTPUT/f"{start}-ledger.json", dict(eligible_observations=observations, orders=orders, fills=fills))
    return summary


def main():
    periods = []
    with httpx.Client(timeout=25) as client:
        fixtures = history(client, PERIODS[-1][1]+timedelta(days=4))
        for start, end in PERIODS:
            report = run_period(client, fixtures, start, end)
            periods.append(report)
            print(json.dumps(report), flush=True)
    save(OUTPUT/"summary.json", dict(rule=RULE, periods=periods))


if __name__ == "__main__":
    main()
