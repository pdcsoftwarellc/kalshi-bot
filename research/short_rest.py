"""Fixed ATP short-rest experiment using verified prior settlements, quoted limits and public prints."""
from collections import Counter, defaultdict
from datetime import date, datetime, timedelta
from decimal import Decimal
import json
from pathlib import Path
import re
import sys

import httpx

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from research.api_cost import (ROOT, SERIES, UTC, PERIODS, STARTING_CASH, fee, ledger,
                               link_market, midnight, pages, qualifying_print, quotes_at, save, timestamp)
from research.forward_odds import tennis

OUT = ROOT / "data/processed/parallel-research/short_rest"
FIXTURES = ROOT / "data/raw/short-rest-v1"
ONE_DAY = timedelta(days=1)
KEEP = ("event_key", "event_date", "event_time", "event_first_player", "event_second_player",
        "first_player_key", "second_player_key", "event_status", "event_winner",
        "event_type_type", "event_final_result")


def get_fixtures(client, start, end, player=None):
    tag = f"player-{player}" if player is not None else "ATP"
    cache = FIXTURES/f"{tag}-{start}-{end}.json"
    if cache.exists():
        return json.loads(cache.read_text())
    selector = {"event_type_key": 265} if player is None else {"player_key": player}
    rows = tennis(client, "get_fixtures", date_start=str(start), date_stop=str(end),
                  timezone="UTC", **selector)
    if not isinstance(rows, list) and rows:
        raise ValueError("Unrecognized fixtures response")
    result = [{k:r.get(k) for k in KEEP} for r in rows]
    save(cache, result)
    return result


def player_ids(fixture):
    return (str(fixture["first_player_key"]), str(fixture["second_player_key"]))


def deciding_set(fixture):
    """A completed full-length singles match whose winner needed the final set."""
    if fixture["event_status"] != "Finished":
        return False
    score = re.fullmatch(r"\s*([123])\s*-\s*([0123])\s*", fixture.get("event_final_result") or "")
    if not score:
        return False
    a,b = map(int,score.groups())
    return ((a,b) in {(2,1),(3,2)} and fixture["event_winner"] == "First Player"
            or (a,b) in {(1,2),(2,3)} and fixture["event_winner"] == "Second Player")


def rested_opponent(fixture, previous):
    """Return an identified deciding-set fixture and the player to buy."""
    players = player_ids(fixture)
    appearances = [[p for p in previous if pid in player_ids(p)] for pid in players]
    for i in (0,1):
        if len(appearances[i]) == 1 and not appearances[1-i] and deciding_set(appearances[i][0]):
            return appearances[i][0], players[1-i]
    return None


def confirmation(linked, prior, decision):
    """Verify both market outcomes match the recorded winner and were published before entry."""
    if linked is None:
        return None
    pair, _, yes_id, _ = linked
    winner_idx = {"First Player":0, "Second Player":1}.get(prior["event_winner"])
    if winner_idx is None:
        return None
    yes_won = player_ids(prior)[winner_idx] == yes_id
    settled = []
    for i, m in enumerate(pair):
        if not m.get("settlement_ts") or m.get("settlement_value_dollars") is None:
            return None
        if Decimal(m["settlement_value_dollars"]) != int(yes_won if i==0 else not yes_won):
            return None
        settled.append(timestamp(m["settlement_ts"]))
    confirmed = max(settled)
    return confirmed if confirmed < decision else None


def universe(client):
    first = PERIODS[0][0] - timedelta(days=4)
    end = PERIODS[-1][1] + timedelta(days=3)
    by_date = defaultdict(list)
    fixtures = {}
    day = first
    while day < end:
        last = min(end-ONE_DAY, day+timedelta(days=6))
        for f in get_fixtures(client, day, last):
            if f["event_type_type"] == "Atp Singles":
                fixtures[str(f["event_key"])] = f
        day = last + ONE_DAY
    for f in fixtures.values():
        by_date[date.fromisoformat(f["event_date"])].append(f)
    pairs = defaultdict(dict)
    for start, stop in [(first,PERIODS[0][0])] + [(a,b+timedelta(days=7)) for a,b in PERIODS]:
        for m in pages(client, "/markets", "markets", series_ticker=SERIES, status="settled",
                       min_close_ts=midnight(start), max_close_ts=midnight(stop), limit=1000):
            pairs[m["event_ticker"]][m["ticker"]] = m
    by_fixture = defaultdict(list)
    prejoin = []
    for tickers in pairs.values():
        pair = sorted(tickers.values(), key=lambda m:m["ticker"])
        m = pair[0]
        if not m.get("occurrence_datetime"):
            continue
        scheduled = timestamp(m["occurrence_datetime"])
        day = datetime.fromtimestamp(scheduled, UTC).date()
        match = None
        if len(pair) == 2:
            nearby = [f for offset in (-1,0,1) for f in by_date[day+timedelta(days=offset)]]
            match = link_market(pair,nearby)
        if match is not None:
            f,yes_id,other_id = match
            linked = pair,f,yes_id,other_id
            by_fixture[str(f["event_key"])].append(linked)
        prejoin.append((scheduled,match))
    # Multiple event pairs mapping to one tennis fixture are ambiguous.
    unique = {key:matches[0] for key,matches in by_fixture.items() if len(matches)==1}
    return by_date, prejoin, unique


def study(client, by_date, prejoin, unique, start, end):
    counters = Counter()
    for occurrence,link in prejoin:
        if midnight(start) <= occurrence-1800 < midnight(end):
            counters["exchange_events_before_linking_by_exchange_start"] += 1
            if link is None:
                counters["unlinked_or_incomplete_at_exchange_start"] += 1
    orders = []
    for key,(pair,fixture,yes_id,other_id) in sorted(unique.items()):
        m = pair[0]
        if not fixture["event_time"]:
            continue
        fday = date.fromisoformat(fixture["event_date"])
        fixture_start = timestamp(fixture["event_date"]+"T"+fixture["event_time"]+"+00:00")
        occurrence = timestamp(m["occurrence_datetime"])
        decision = int(min(occurrence, fixture_start)//60)*60 - 1800
        if not midnight(start) <= decision < midnight(end):
            continue
        counters["uniquely_linked_decision_in_period"] += 1
        if decision < midnight(fday) or decision < max(timestamp(v["open_time"]) for v in pair):
            counters["decision_before_fixture_date_or_listing"] += 1
            continue
        if not m.get("settlement_ts") or m.get("settlement_value_dollars") is None or timestamp(m["settlement_ts"]) <= decision+300:
            counters["early_or_missing_settlement"] += 1
            continue
        previous_day = fday - ONE_DAY
        trigger = rested_opponent(fixture,by_date[previous_day])
        if trigger is None:
            continue
        prior,pick = trigger
        counters["prior_decider_and_no_previous_atp_record"] += 1
        verified = confirmation(unique.get(str(prior["event_key"])), prior, decision)
        if verified is None:
            counters["previous_finish_unverified_by_decision"] += 1
            continue
        # The index covers ATP singles. Use a player-level, all-event-types lookup
        # for the prospective rested player to detect e.g. Challenger/ITF appearances.
        all_appearances = get_fixtures(client,previous_day,previous_day,pick)
        if any(pick in player_ids(p) and "singles" in str(p["event_type_type"]).lower()
               and p["event_date"] == str(previous_day) for p in all_appearances):
            counters["rested_player_has_other_singles_fixture"] += 1
            continue
        counters["verified_rest_mismatch"] += 1
        quote = quotes_at(client,m["ticker"],decision)
        if quote is None:
            counters["missing_two_sided_quote"] += 1
            continue
        bid,ask,observed = quote
        side = "yes" if pick == yes_id else "no"
        limit = ask if side=="yes" else 1-bid
        trade = qualifying_print(pages(client, "/markets/trades", "trades", ticker=m["ticker"],
                                       min_ts=decision+2, max_ts=decision+300, limit=1000),side,limit,decision)
        payout = Decimal(m["settlement_value_dollars"])
        orders.append(dict(fixture=key, ticker=m["ticker"], ts=decision, side=side,
                           limit=str(limit), fee=str(fee(limit)), quote_ts=observed,
                           rested_player=pick, prior_fixture=str(prior["event_key"]),
                           prior_final=prior["event_final_result"], prior_confirmed_ts=verified,
                           payout=str(payout if side=="yes" else 1-payout),
                           settlement_ts=timestamp(m["settlement_ts"]),
                           print_ts=trade[0] if trade else None, trade_id=trade[1] if trade else None))
    filled, totals = ledger(orders,STARTING_CASH)
    end_month = (start.replace(day=28)+timedelta(days=4)).replace(day=1)
    bill = Decimal(80)*(end-start).days/(end_month-start).days
    costs = [Decimal(r["limit"])+Decimal(r["fee"]) for r in orders]
    return dict(start=str(start),end_exclusive=str(end),coverage=dict(counters),orders=len(orders),
                **totals,gross_pnl_dollars=round(totals["trading_pnl_dollars"]+totals["fees_dollars"],4),
                api_cost_dollars=float(bill),pnl_after_api=round(totals["trading_pnl_dollars"]-float(bill),4),
                all_quoted_fill_scenario=float(sum((Decimal(r["payout"])-cost for r,cost in zip(orders,costs)),Decimal(0))),
                all_win_ceiling_at_limits=float(sum((1-cost for cost in costs),Decimal(0)))), orders, filled


def main():
    directory = OUT
    reports = []
    with httpx.Client(timeout=45) as client:
        by_date,prejoin,unique = universe(client)
        for start,end in PERIODS:
            report,orders,fills = study(client,by_date,prejoin,unique,start,end)
            reports.append(report)
            save(directory/f"{start}-ledger.json",dict(orders=orders,fills=fills))
            print(json.dumps(report),flush=True)
    save(directory/"summary.json",reports)


if __name__ == "__main__":
    main()
