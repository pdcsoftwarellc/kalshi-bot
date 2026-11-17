"""Paper screen for a match/decider pricing difference after a confirmed one-set tie."""
import argparse
from collections import Counter, defaultdict
from datetime import date, timedelta
from decimal import Decimal as D, ROUND_CEILING
from heapq import heappop, heappush
import json
import math
from pathlib import Path
import sys

import httpx

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from research.api_cost import ROOT, SERIES, PERIODS, fetch, pages, fee, save, midnight, timestamp
from research.set_frequency import SET_SERIES, groups, confirmation, competitor
from research.paired_prices import cash_ledger

EXACT = "KXATPEXACTMATCH"
OUT = ROOT / "data/processed/parallel-round3/decider_basis"
LIMIT = D(".98")
ARMS = ("third_set", "exact_2_1")


def direct_fee(price, multiplier):
    """Fee for one whole-contract buy in a single fill with a new accumulator."""
    model = D(".07") * multiplier * price * (1-price)
    fee_6dp = model.quantize(D(".000001"), rounding=ROUND_CEILING)
    return (price + fee_6dp).quantize(D(".0001"), rounding=ROUND_CEILING) - price


def public_fees(client):
    series = {}
    for ticker in (SERIES, SET_SERIES, EXACT):
        current = fetch(client, "/series/"+ticker, {})["series"]
        changes = fetch(client, "/series/fee_changes", dict(series_ticker=ticker, show_historical="true"))
        series[ticker] = dict(current_type=current["fee_type"], current_multiplier=current["fee_multiplier"],
                              changes=changes["series_fee_change_arr"])
    events = defaultdict(list)
    count = 0
    for row in pages(client, "/events/fee_changes", "event_fee_changes", limit=1000):
        count += 1
        if row["series_ticker"] in series:
            events[row["event_ticker"]].append(row)
    return dict(series=series,event_changes=dict(events),global_event_changes_returned=count)


def multiplier_at(meta, event, ts):
    series = event.partition("-")[0]
    data = meta["series"][series]
    kind, multiplier = data["current_type"], data["current_multiplier"]
    changes = sorted(data["changes"],key=lambda r:timestamp(r["scheduled_ts"]))
    effective = [r for r in changes if timestamp(r["scheduled_ts"]) <= ts]
    if effective:
        kind, multiplier = effective[-1]["fee_type"], effective[-1]["fee_multiplier"]
    elif changes:
        raise ValueError("No verifiable prior fee before first reported series change")
    # No published series changes: the sensitivity uses today's series setting
    # conditionally. An empty override feed cannot prove every historical fee.
    overrides = [r for r in meta["event_changes"].get(event,[]) if timestamp(r["scheduled_ts"]) <= ts]
    if overrides:
        last = max(overrides,key=lambda r:timestamp(r["scheduled_ts"]))
        kind = last["fee_type_override"] if last["fee_type_override"] is not None else kind
        multiplier = last["fee_multiplier_override"] if last["fee_multiplier_override"] is not None else multiplier
    if kind not in ("quadratic", "quadratic_with_maker_fees"):
        raise ValueError("Series or override has a different fee model: "+str(kind))
    return D(str(multiplier))


def direct_cash_ledger(signals):
    cash = D(25)
    releases, counts = [], Counter()
    fees = gross = paired = stranded = D(0)
    for s in sorted(signals,key=lambda row:(row["ts"], row["event"])):
        while releases and releases[0][0] <= s["ts"]:
            cash += heappop(releases)[1]
        prices, leg_fees = list(map(D,s["prices"])), list(map(D,s["leg_fees"]))
        costs = [p+f for p,f in zip(prices,leg_fees,strict=True)]
        if sum(costs)>cash:
            counts["cash_blocked"]+=1
            continue
        cash -= sum(costs)
        fills=sum(bool(p) for p in s["prints"])
        counts[("no_fills","single_legs","both_legs")[fills]]+=1
        for i, cost in enumerate(costs):
            if s["prints"][i] is None:
                heappush(releases,(s["ts"]+7,cost))
                continue
            payout = D(s["settlement"][i])
            heappush(releases,(timestamp(s["settlement_ts"][i]),payout))
            fees += leg_fees[i]
            gross += payout-prices[i]
            if fills==2:
                paired += payout-cost
            else:
                stranded += payout-cost
    metrics={k:counts[k] for k in ("both_legs","single_legs","no_fills","cash_blocked")}
    return dict(metrics,gross_dollars=float(gross),fees_dollars=float(fees),net_dollars=float(gross-fees),
                paired_net=float(paired),unhedged_net=float(stranded))


def named_pair(markets):
    if len(markets) != 2 or any(not competitor(m) or not m.get("yes_sub_title") for m in markets):
        return None
    result = {competitor(m): m for m in markets}
    return result if len(result) == 2 else None


def tie_confirmation(first, second):
    """The second distinct set result must be public as well as the first."""
    a, b = confirmation(first), confirmation(second)
    if not a or not b or a[2] == b[2] or {competitor(m) for m in first} != {competitor(m) for m in second}:
        return None
    confirmed = max(a[1], b[1])
    return int(60 * (math.floor(confirmed / 60) + 1)), confirmed


def exact_slate(markets, players, t):
    """Only four prelisted 2-0/2-1 strikes positively establish best-of-three."""
    expected = {(pid, score) for pid in players for score in ("2-0", "2-1")}
    strikes = {}
    for m in markets:
        if timestamp(m["open_time"]) > t:
            continue
        pid = competitor(m)
        score = (m.get("custom_strike") or {}).get("Set Score")
        if pid not in players or not score or (pid,score) in strikes:
            return None
        name = players[pid]["yes_sub_title"]
        if m["market_type"] != "binary" or D(m["notional_value_dollars"]) != 1:
            return None
        if not (m["rules_primary"].startswith(f"If {name} wins ") and
                f"by a set score of {score}," in m["rules_primary"]):
            return None
        strikes[pid,score] = m
    if set(strikes) != expected:
        return None
    return {pid: strikes[pid,"2-1"] for pid in players}


def minute_quotes(client, markets, decision):
    """Use the close on each observed side (one-sided books are supported)."""
    query = dict(market_tickers=",".join(sorted(m["ticker"] for m in markets)),
                 start_ts=decision-60, end_ts=decision, period_interval=1)
    response = fetch(client, "/markets/candlesticks", query)
    result = {}
    for book in response["markets"]:
        for candle in book["candlesticks"]:
            if candle["end_period_ts"] != decision:
                continue
            raw_ask = (candle.get("yes_ask") or {}).get("close_dollars")
            raw_bid = (candle.get("yes_bid") or {}).get("close_dollars")
            ask = D(raw_ask) if raw_ask else None
            bid = D(raw_bid) if raw_bid else None
            if ask is not None and bid is not None and ask < bid:
                continue
            sides = {}
            if ask is not None and 0 < ask < 1:
                sides["yes"] = ask
            if bid is not None and 0 < bid < 1:
                sides["no"] = 1-bid
            result[book["market_ticker"]] = sides
    return result


def cheapest_pair(match, decider, quotes, multipliers=None):
    """Choose 2 opposite player exposures in different books for a normally completed match."""
    players = sorted(match)
    if len(players) != 2 or set(players) != set(decider):
        raise ValueError("Different participant IDs")

    def legs(book):
        for i,pid in enumerate(players):
            ticker = book[pid]["ticker"]
            for side,price in quotes.get(ticker,{}).items():
                yield pid if side=="yes" else players[1-i], ticker, side, price

    def cost(ticker,price):
        return price + (fee(price) if multipliers is None else direct_fee(price,multipliers[ticker]))

    choices = []
    for player, ticker, side, price in legs(match):
        for other, ticker2, side2, price2 in legs(decider):
            if player != other:
                combined = cost(ticker,price) + cost(ticker2,price2)
                choices.append((combined,ticker,side,ticker2,side2,price,price2))
    if not choices:
        return None
    cost,t1,s1,t2,s2,p1,p2 = min(choices)
    row = dict(cost=str(cost),tickers=[t1,t2],sides=[s1,s2],prices=[str(p1),str(p2)])
    if multipliers is not None:
        row["leg_fees"]=[str(direct_fee(p1,multipliers[t1])),str(direct_fee(p2,multipliers[t2]))]
        row["multipliers"]=[str(multipliers[t1]),str(multipliers[t2])]
    return row


def proof(trades, ticker, side, limit, decision, close):
    for trade in sorted(trades,key=lambda x:timestamp(x["created_time"])):
        ts = timestamp(trade["created_time"])
        if (trade.get("ticker") == ticker and decision+2 <= ts < min(decision+7,close)
                and not trade.get("is_block_trade") and D(trade["count_fp"]) >= 1
                and (trade.get("taker_outcome_side") or trade.get("taker_side")) == side
                and D(trade[side+"_price_dollars"]) <= limit):
            return trade["trade_id"]
    return None


def study(client, start, end, fee_metadata=None, destination=OUT):
    sets = groups(client, SET_SERIES, start, end)
    matches = groups(client, SERIES, start, end)
    exacts = groups(client, EXACT, start, end)
    counts = Counter()
    arm_counts = {a:Counter() for a in ARMS}
    ledgers = {a:[] for a in ARMS}
    costs = {a:[] for a in ARMS}
    signals = {a:[] for a in ARMS}
    for event, set_markets in sets.items():
        if not event.endswith("-2"):
            continue
        set2 = list(set_markets.values())
        first_event = event[:-1]+"1"
        first = list(sets.get(first_event,{}).values())
        deadline = tie_confirmation(first,set2)
        if not deadline or not midnight(start) <= deadline[0] < midnight(end):
            continue
        decision,confirmed = deadline
        counts["confirmed_ties"] += 1
        root = event[len(SET_SERIES)+1:-2]
        match_event = SERIES+"-"+root
        match = named_pair(list(matches.get(match_event,{}).values()))
        if not match or set(match) != {competitor(m) for m in set2}:
            counts["no_verified_match_pair"] += 1
            continue
        if not all(m["market_type"]=="binary" and D(m["notional_value_dollars"]) == 1 and
                   m["rules_primary"].startswith("If "+m["yes_sub_title"]+" wins ") for m in match.values()):
            counts["match_rule_mismatch"] += 1
            continue
        if not all(timestamp(m["open_time"]) <= decision < timestamp(m["close_time"]) for m in match.values()):
            counts["match_not_open"] += 1
            continue
        counts["ties_with_open_match_pair"] += 1
        exact = exact_slate(exacts.get(EXACT+"-"+root,{}).values(), match, decision)
        direct = named_pair(list(sets.get(event[:-1]+"3",{}).values()))
        if direct:
            counts["direct_third_set_pair_listed"] += 1
        if not exact:
            counts["missing_format_certificate"] += 1
            continue
        counts["best_of_three_verified"] += 1
        for arm, second in (("third_set",direct),("exact_2_1",exact)):
            coverage = arm_counts[arm]
            if not second:
                coverage["not_listed"] += 1
                continue
            if set(second)!=set(match) or (arm=="third_set" and not all(
                    m["market_type"]=="binary" and D(m["notional_value_dollars"])==1 and
                    m["rules_primary"].startswith("If "+match[pid]["yes_sub_title"]+" wins set 3 ")
                    for pid,m in second.items())):
                coverage["rule_mismatch"]+=1
                continue
            markets = [*match.values(),*second.values()]
            open_markets = [m for m in markets if timestamp(m["open_time"]) <= decision < timestamp(m["close_time"])]
            if len(open_markets) <= 2:
                coverage["alternative_not_open"]+=1
                continue
            coverage["overlapping_listings"] += 1
            quotes = minute_quotes(client,open_markets,decision)
            multipliers = None
            if fee_metadata is not None:
                multipliers = {m["ticker"]:multiplier_at(fee_metadata,m["event_ticker"],decision) for m in open_markets}
            row = cheapest_pair(match,second,quotes,multipliers)
            if row is None:
                coverage["no_opposite_prices_at_minute"] += 1
                continue
            row.update(event=match_event,ts=decision,confirmed_ts=confirmed)
            coverage["executable_quote_comparisons"] += 1
            costs[arm].append(D(row["cost"]))
            if D(row["cost"]) <= LIMIT:
                coverage["discount_signals"] += 1
                by_ticker={m["ticker"]:m for m in markets}
                prints, settlements, settlements_ts = [], [], []
                for ticker,side,p in zip(row["tickers"],row["sides"],row["prices"],strict=True):
                    m=by_ticker[ticker]
                    trades=pages(client,"/markets/trades","trades",ticker=ticker,
                                 min_ts=decision+2,max_ts=decision+7,limit=1000)
                    prints.append(proof(trades,ticker,side,D(p),decision,timestamp(m["close_time"])))
                    val=m.get("settlement_value_dollars")
                    if val is not None:
                        payout=D(val)
                        if not 0<=payout<=1:
                            raise ValueError("Invalid YES settlement")
                        val=str(payout if side=="yes" else 1-payout)
                    if prints[-1] and (not m.get("settlement_ts") or timestamp(m["settlement_ts"]) < decision+2):
                        raise ValueError("No valid later settlement for a filled contract")
                    settlements.append(val)
                    settlements_ts.append(m.get("settlement_ts"))
                row.update(prints=prints,settlement=settlements,settlement_ts=settlements_ts)
                signals[arm].append(row)
            ledgers[arm].append(row)
    days_in_month=((start.replace(day=28)+timedelta(days=4)).replace(day=1)-start.replace(day=1)).days
    bill=80*(end-start).days/days_in_month
    summary=dict(start=str(start),end_exclusive=str(end),coverage=dict(counts),arms={})
    for arm in ARMS:
        pnl=(direct_cash_ledger if fee_metadata is not None else cash_ledger)(signals[arm])
        summary["arms"][arm]=dict(coverage=dict(arm_counts[arm]),signals=len(signals[arm]),
            minimum_all_in=str(min(costs[arm])) if costs[arm] else None,**pnl,
            bill=round(bill,2),after_bill=round(pnl["net_dollars"]-bill,4))
    save(destination/f"{start}-ledger.json",dict(summary=summary,comparisons=ledgers))
    return summary


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--direct-fees",action="store_true",help="Separate sensitivity for $0.0001 balance precision")
    args=parser.parse_args()
    destination = OUT / "direct_fee_sensitivity" if args.direct_fees else OUT
    method=dict(
        rule="After verified opposite set 1 and 2 results, at the first minute strictly after both pairs settle, choose the cheapest pair of opposite player exposures across the match and deciding instruments. Maximum combined limit and fees $0.98. One attempt per event per arm.",
        arm1="Direct third-set winner vs match winner.",
        arm2="A separate extension: exact 2–1 score vs match winner after 1–1. The prelisted four-strike, two-player 2–0/2–1 slate verifies best-of-three.",
        selection="All eligible ATP markets in the listing/closing query with 7-day buffer. Exact participant UUIDs and prompt text checked. Limits chosen without either leg's final payout. Same-time historical one-minute top quotes; NO ask = 1 minus YES bid. Missing quoted sides excluded. No forward filling.",
        fills="Submit both chosen limits concurrently. Each leg independently requires its own same-side ordinary public taker print >=1 contract at/better than limit from +2s inclusive to +7s exclusive and before that leg closes. Credit the submitted limit. No full basket fill inferred from price touches.",
        cash="Separate $25 simulated cash account per period per arm, reserve both legs at signal; release misses at expiry and actual side-specific payouts at settlement. General taker fee with multiplier 1 rounded up to cent per leg.",
        settlement="Equivalent winner only for a normally completed three-set match. Match-specific withdrawal rules, incomplete sets, disqualification, abandonment, cancellations and discretionary fair-value settlements may produce different payouts. A one-leg fill is directional inventory. A minute quote provides no proof of simultaneous depth.",
        sources=["https://assets.kalshi.com/contract_terms/TENNISSETWINNER.pdf","https://assets.kalshi.com/contract_terms/TENNISSETSCORE.pdf","https://assets.kalshi.com/contract_terms/ACHIEVEMENTS.pdf","https://kalshi.com/markets/kxatpmatch/atp-tennis-match/kxatpmatch-26nov08shealc","https://kalshi.com/fee-schedule"],
    )
    results=[]
    with httpx.Client(timeout=35) as client:
        metadata = public_fees(client) if args.direct_fees else None
        if metadata is not None:
            method["cash"] = ("One contract per leg, single fill and new fee accumulator. General quadratic taker fee "
                              "rounded up to six decimals; round position cost plus fee up to $0.0001. "
                              "Direct-member scenario only. Use reported series/override metadata at decision time "
                              "and current series fees if no change is reported. Historical private fees are unverified.")
            method["sensitivity"] = "Chosen after inspecting the 99-cent near miss. Retained original 98-cent threshold."
            method["sources"].extend(["https://docs.kalshi.com/getting_started/fee_rounding",
                                      "https://kalshi.com/docs/kalshi-fee-schedule.pdf",
                                      "https://docs.kalshi.com/api-reference/exchange/get-series-fee-changes",
                                      "https://docs.kalshi.com/api-reference/events/get-event-fee-changes"])
            save(destination/"fee_metadata.json",metadata)
        save(destination/"method.json",method)
        for start,end in [*PERIODS,(date(2026,9,22),date(2026,10,1))]:
            summary=study(client,start,end,metadata,destination)
            results.append(summary)
            print(json.dumps(summary),flush=True)
            save(destination/"summary.json",results)


if __name__=="__main__":
    main()
