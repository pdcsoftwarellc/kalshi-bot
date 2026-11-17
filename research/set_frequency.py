"""Measure trade frequency using publicly confirmed nonterminal ATP sets."""
from collections import Counter, defaultdict
from datetime import date, datetime, timedelta
from decimal import Decimal, ROUND_CEILING, ROUND_FLOOR
import hashlib
import json
import math
from heapq import heappop, heappush
from pathlib import Path
import sys

import httpx
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from integration.edge import compute_edge, evaluate_signal
from research.api_cost import ROOT, PERIODS, SERIES, UTC, fee, fetch, link_market, midnight, pages, save, timestamp
from tennis.engine import PredictionEngine
from tennis.fixture_history import MAJOR_SURFACES
from tennis.live_score import observed_points
from tennis.types import MatchFormat, MatchState

D = Decimal
SET_SERIES = "KXATPSETWINNER"
OUT = ROOT / "data/processed/native-atp/set-frequency"
MODEL = ROOT / "data/artifacts/native-atp"
DECISION_SECONDS = 60


def competitor(market):
    return (market.get("custom_strike") or {}).get("tennis_competitor")


def confirmation(pair):
    if len(pair) != 2 or not all(competitor(m) and m.get("settlement_ts") for m in pair):
        return None
    by_result = {m.get("result"): m for m in pair}
    if set(by_result) != {"yes", "no"} or len({competitor(m) for m in pair}) != 2:
        return None
    for side, expected in (("yes",1),("no",0)):
        value = by_result[side].get("settlement_value_dollars")
        if value is None or D(value) != expected:
            return None
    confirmed = max(timestamp(m["settlement_ts"]) for m in pair)
    decision = DECISION_SECONDS * math.ceil(confirmed / DECISION_SECONDS)
    return decision, confirmed, competitor(by_result["yes"])


def initial_state(fixture, server):
    name = str(fixture.get("tournament_name", "")).lower()
    major = name in MAJOR_SURFACES
    qualifying = str(fixture.get("event_qualification")).lower() in ("1", "true")
    return MatchState(player1_id=str(fixture["first_player_key"]), player2_id=str(fixture["second_player_key"]),
                      surface=MAJOR_SURFACES.get(name, fixture.get("surface") or "Unknown"),
                      tournament_level="G" if major else "O", match_date=fixture["event_date"],
                      server=server, match_format=MatchFormat(best_of=5 if major and not qualifying else 3))


def forecast(engine, fixture, first_winner, second_winner=None):
    # Only the confirmed set prefix is eligible for prediction.
    # No later set, final score, winner or postmatch statistics enter it.
    allowed = {"Set 1", "Set 1 TieBreak"}
    if second_winner is not None:
        allowed.update(("Set 2", "Set 2 TieBreak"))
    history = [g for g in fixture.get("pointbypoint") or [] if g.get("set_number") in allowed]
    state = initial_state(fixture, 0)
    best_of = state.match_format.best_of
    points = observed_points(history, best_of)
    winners = [first_winner] if second_winner is None else [first_winner, second_winner]
    previous = (0,0)
    transitions = []
    for point in points:
        if point.sets != previous:
            transitions.append(0 if point.sets[0] > previous[0] else 1)
            previous = point.sets
    if not points or transitions != winners or points[-1].games != (0,0) or max(points[-1].sets) >= best_of//2+1:
        raise ValueError("Incomplete or terminal score prefix")
    state.server = points[0].served_by
    seed = int.from_bytes(hashlib.sha256(str(fixture["event_key"]).encode()).digest()[:8],"big")
    engine._rng = np.random.default_rng(seed)
    engine.new_match(state)
    for p in points:
        engine.record_point(p.won, server=p.served_by)
    state.sets_p1, state.sets_p2 = points[-1].sets
    state.server = points[-1].next_server
    return engine.predict(state)


def quoted_close(client, ticker, decision):
    data = fetch(client, f"/series/{SERIES}/markets/{ticker}/candlesticks",
                 dict(start_ts=decision-60, end_ts=decision, period_interval=1))
    for bar in data["candlesticks"]:
        if bar["end_period_ts"] != decision:
            continue
        b = bar.get("yes_bid",{}).get("close_dollars")
        a = bar.get("yes_ask",{}).get("close_dollars")
        if b and a and 0 < D(b) <= D(a) < 1:
            return int((D(b)*100).to_integral_value(rounding=ROUND_FLOOR)), int((D(a)*100).to_integral_value(rounding=ROUND_CEILING))
    return None


def print_hit(trades, side, limit, t, seconds):
    for r in sorted(trades, key=lambda row:timestamp(row["created_time"])):
        when = timestamp(r["created_time"])
        if (t+2 <= when < t+seconds and not r.get("is_block_trade")
                and (r.get("taker_outcome_side") or r.get("taker_side")) == side
                and D(r["count_fp"]) >= 1 and D(r[f"{side}_price_dollars"]) <= limit):
            return dict(id=r["trade_id"],ts=when)
    return None


def groups(client, series, start, end):
    grouped = defaultdict(dict)
    for m in pages(client,"/markets","markets",series_ticker=series,status="settled",limit=1000,
                   min_close_ts=midnight(start-timedelta(days=2)),max_close_ts=midnight(end+timedelta(days=7))):
        grouped[m["event_ticker"]][m["ticker"]]=m
    return grouped


def collect_forecasts(client, engine, start, end, settings):
    source = ROOT/"data/raw/native-atp"
    by_day = defaultdict(list)
    day = start-timedelta(days=2)
    while day < end+timedelta(days=1):
        path = source / f"{day}.json"
        if not path.exists():
            path = ROOT/"data/raw/set-frequency-atp"/f"{day}.json"
        if path.exists():
            by_day[day].extend(json.loads(path.read_text()))
        elif start <= day < end:
            raise ValueError(f"Missing fixture date {day}")
        day += timedelta(days=1)

    matches = groups(client,SERIES,start,end)
    sets = groups(client,SET_SERIES,start,end)
    coverage = Counter()
    records, linked = [], defaultdict(list)
    for event, markets in sets.items():
        if not event.startswith(SET_SERIES+"-") or event.rsplit("-",1)[-1] not in {"1", "2"}:
            continue
        set_pair=list(markets.values())
        known=confirmation(set_pair)
        if not known:
            continue
        decision, confirmed, winner=known
        if not midnight(start)<=decision<midnight(end):
            continue
        set_number = int(event.rsplit("-", 1)[-1])
        coverage[f"confirmed_set_{set_number}"] += 1
        first_known = known
        if set_number == 2:
            prior_pair = list(sets.get(event[:-1]+"1", {}).values())
            first_known = confirmation(prior_pair)
            if {competitor(m) for m in prior_pair} != {competitor(m) for m in set_pair}:
                coverage["mismatched_set_participants"] += 1
                continue
            if not first_known or first_known[1] > confirmed or first_known[2] == winner:
                coverage["not_a_confirmed_one_set_tie"] += 1
                continue
        event_id = SERIES + event[len(SET_SERIES):-2]
        pair = sorted(matches.get(event_id,{}).values(),key=lambda m:m["ticker"])
        if len(pair)!=2 or {competitor(m) for m in pair} != {competitor(m) for m in set_pair}:
            coverage["missing_matching_binary_market_pair"]+=1
            continue
        market = pair[0]
        if not timestamp(market["open_time"])<=decision<timestamp(market["close_time"])-settings["max_age_ms"]/1000:
            coverage["match_not_open_for_window"]+=1
            continue
        current = datetime.fromtimestamp(confirmed,UTC).date()
        pool = [f for days in (-1,0,1) for f in by_day[current+timedelta(days=days)]]
        match = link_market(pair,pool)
        if match is None:
            coverage["no_unique_fixture"]+=1
            continue
        f, yes_id, _ = match
        last_book_close = max(timestamp(leg["close_time"]) for leg in set_pair)
        linked[str(f["event_key"]),set_number].append((market,f,yes_id,first_known[2],winner,confirmed,decision,last_book_close))

    for (fid,set_number), group in linked.items():
        if len(group)!=1:
            coverage["ambiguous_pair_for_fixture"]+=len(group)
            continue
        m,f,yes_id,first,winner,confirmed,t,last_book_close=group[0]
        yes_player = 0 if yes_id==str(f["first_player_key"]) else 1
        first_winner = yes_player if first==competitor(m) else 1-yes_player
        second_winner = None if set_number == 1 else (yes_player if winner==competitor(m) else 1-yes_player)
        try:
            pred = forecast(engine,f,first_winner,second_winner)
        except (ValueError, KeyError, TypeError, IndexError):
            coverage["incomplete_or_inconsistent_score_prefix"]+=1
            continue
        q=quoted_close(client,m["ticker"],t)
        if not q or not 0 < q[0] <= q[1] < 100:
            coverage["no_valid_boundary_quote"]+=1
            continue
        bid,ask=q
        p=pred.win_prob_p1 if yes_player==0 else pred.win_prob_p2
        prior_cutoff = None
        if m.get("occurrence_datetime") and f.get("event_time"):
            scheduled = timestamp(f["event_date"]+"T"+f["event_time"]+"+00:00")
            prior_cutoff = int(min(scheduled, timestamp(m["occurrence_datetime"]))//60)*60-1800
        records.append(dict(fixture=fid,set_number=set_number,ticker=m["ticker"],t=t,confirmed=confirmed,points_observed=pred.points_observed,
                            yes_player=yes_player,first_set_winner=first_winner,model=p,confidence=pred.confidence,
                            bid=bid,ask=ask,set_book_close=last_book_close,prematch_cutoff=prior_cutoff,opened=timestamp(m["open_time"]),
                            settlement=m.get("settlement_value_dollars"),settled=m.get("settlement_ts")))
    return records, coverage


def evaluate_records(client, records, start, end, settings, coverage):
    """Apply the same signal gates and chronological cash ledger to any YES probabilities."""
    cash=D(str(settings["bankroll"]))
    releases=[]
    locked=D(0)
    pnl=D(0)
    all_win=D(0)
    for r in sorted(records,key=lambda r:(r["t"],r["ticker"])):
        while releases and releases[0][0]<=r["t"]:
            _, returned, cost_released = heappop(releases)
            cash += returned
            locked -= cost_released
        side, edge, cents = compute_edge(r["model"], r["ask"], r["bid"])
        limit = D(cents)/100
        ticket_fee = fee(limit) if side else D(0)
        r.update(side=side,gross_edge=edge,limit=str(limit),fee=str(ticket_fee))
        sig=evaluate_signal(r["ticker"],r["model"],r["confidence"],r["points_observed"],r["ask"],r["bid"],float(min(cash,D(str(settings["bankroll"]))-locked)),
            min_edge=settings["min_edge"],min_confidence=settings["min_confidence"],min_points=settings["min_points"],
            net_edge_min=settings["net_edge_min"],max_contracts=1,fee_per_contract=float(ticket_fee))
        if not sig:
            if r["confidence"]<settings["min_confidence"]:
                reason="low_confidence"
            elif not r["side"] or r["gross_edge"]<settings["min_edge"]:
                reason="gross_edge"
            elif r["gross_edge"]-float(ticket_fee)<settings["net_edge_min"]:
                reason="net_edge"
            else:
                reason="point_or_size_limit"
            coverage[reason]+=1
            r["decision"]=reason
            continue
        coverage["signals"]+=1
        all_win+=1-limit-ticket_fee
        cost=limit+ticket_fee
        cash-=cost
        locked+=cost
        ttl=settings["max_age_ms"] / 1000
        hit=print_hit(pages(client,"/markets/trades","trades",ticker=r["ticker"],limit=1000,
                         min_ts=r["t"]+2,max_ts=int(r["t"]+ttl)),sig.side,limit,r["t"],ttl)
        if not hit:
            r["decision"]="unfilled"
            heappush(releases,(r["t"]+ttl,cost,cost))
            continue
        if r["settlement"] is None or not r["settled"] or timestamp(r["settled"])<=hit["ts"]:
            raise ValueError("Missing or invalid settlement of a filled contract")
        yes_payout=D(r["settlement"])
        payout=yes_payout if sig.side=="yes" else 1-yes_payout
        heappush(releases,(timestamp(r["settled"]),payout,cost))
        pnl+=payout-cost
        coverage["fills"]+=1
        r.update(decision="paper_fill",proof=hit,pnl=str(payout-cost))
    days=(end-start).days
    month_days=((start.replace(day=28)+timedelta(days=4)).replace(day=1)-start.replace(day=1)).days
    bill=D(80)*days/month_days
    summary=dict(start=str(start),end_exclusive=str(end),paired_forecasts=len(records),coverage=dict(coverage),
                 qualified=coverage["signals"],fills=coverage["fills"],net_trading_dollars=str(pnl),
                 all_signals_all_win_dollars=str(all_win),allocated_api_dollars=str(bill),after_api_dollars=str(pnl-bill))
    binary = [r for r in records if r["settlement"] is not None and r["settled"]
              and D(r["settlement"]) in (D(0), D(1)) and timestamp(r["settled"]) > r["t"]]
    if binary:
        summary["binary_outcomes"] = len(binary)
        summary["model_brier"] = sum((r["model"]-float(r["settlement"]))**2 for r in binary)/len(binary)
        summary["market_brier"] = sum(((r["bid"]+r["ask"])/200-float(r["settlement"]))**2 for r in binary)/len(binary)
    fills = [r for r in records if r["decision"] == "paper_fill"]
    summary["winning_fills"] = sum(D(r["pnl"]) > 0 for r in fills)
    summary["fees_on_fills"] = str(sum((D(r["fee"]) for r in fills), D(0)))
    by_set = {}
    for n in (1,2):
        group = [r for r in records if r["set_number"]==n]
        filled = [r for r in group if r["decision"]=="paper_fill"]
        sent = [r for r in group if r["decision"] in ("paper_fill","unfilled")]
        by_set[str(n)] = dict(forecasts=len(group), signals=len(sent), fills=len(filled),
                              net=str(sum((D(r["pnl"]) for r in filled), D(0))),
                              ceiling=str(sum((1-D(r["limit"])-D(r["fee"]) for r in sent), D(0))))
    summary["distinct_matches_with_quotes"] = len({r["fixture"] for r in records})
    summary["by_set"] = by_set
    return summary


def run_period(client, engine, start, end, settings):
    records, coverage = collect_forecasts(client, engine, start, end, settings)
    summary = evaluate_records(client, records, start, end, settings, coverage)
    save(OUT/f"{start}.json",dict(summary=summary,ledger=records))
    print(json.dumps(summary),flush=True)
    return summary


def main():
    settings=json.loads((ROOT/"data/processed/native-atp/tape-replay/summary.json").read_text())[0]["thresholds"]
    if (settings["per_signal"],settings["latency_ms"],settings["taker_multiplier"],settings["fee_precision"])!=(1,2000,1,"0.01"):
        raise ValueError("The frozen study requires one contract, 2s latency and cent-rounded general taker fees.")
    meta=json.loads((MODEL/"fit_metadata.json").read_text())
    if meta["calibration_through"]>=str(PERIODS[0][0]):
        raise ValueError("Evaluation overlaps model fitting")
    engine=PredictionEngine.load(str(MODEL),n_mc_samples=200)
    with httpx.Client(timeout=45) as client:
        reports=[run_period(client,engine,s,e,settings) for s,e in PERIODS]
    save(OUT/"summary.json",dict(settings=settings,periods=reports))

if __name__=="__main__":
    main()
