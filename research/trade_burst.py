"""Test one-sided aggressive trade flow in ATP singles, one signal opportunity per match."""
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from datetime import date, timedelta
from decimal import Decimal as D
from functools import partial
from math import ceil
from pathlib import Path
import sys

import httpx

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from research.api_cost import ROOT, PERIODS, fee, history, midnight, pages, save, timestamp
from research.price_drift import matched_events, read_quote
from research.favorite_rebound import cash_ledger

OUT = ROOT / "data/processed/parallel-round2/trade_burst"
CHECK = (date(2026,9,19),date(2026,10,1))
RULE = {
    "universe":"Uniquely linked ATP singles, provider/exchange schedules within 60 minutes",
    "decision":"10 minutes after ATP fixture scheduled start (in-play status unverified)",
    "signal":"At least five non-block prints totaling >=100 contracts in the preceding five minutes, >=80% on one taker side",
    "execution":"Enter at that side's ask, fresh <=60s, spread <=3c. Require >=1 contract later same-side print at or better, latency 2s, TTL 60s.",
    "exit":"10 minutes after fill, attempt sale at fresh bid. Reprice at subsequent minute boundaries up to 15 minutes after first exit attempt, then hold to settlement.",
    "sizing":"$25 starting cash per independent period, one contract per entry. One-cent-rounded quadratic taker fee on each exchange fill.",
}


def direction(trades, decision):
    volume = Counter()
    seen = set()
    for trade in trades:
        side = trade.get("taker_outcome_side") or trade.get("taker_side")
        if side not in ("yes","no") or trade.get("is_block_trade"):
            continue
        if not decision-300 <= timestamp(trade["created_time"]) < decision:
            continue
        qty = D(trade["count_fp"])
        if trade["trade_id"] in seen or qty <= 0:
            continue
        seen.add(trade["trade_id"])
        volume[side] += qty
    total = sum(volume.values(), D(0))
    chosen = None
    if len(seen) >= 5 and total >= 100:
        side, count = volume.most_common(1)[0]
        if count / total >= D(".80"):
            chosen = side
    return chosen, len(seen), total, volume["yes"]


def trade_list(client, ticker, start, stop):
    return pages(client,"/markets/trades","trades",ticker=ticker,limit=1000,min_ts=int(start),max_ts=int(stop))


def print_after(client, ticker, side, buy, limit, when, close):
    stop = min(when+62, close)
    if stop <= when+2:
        return None
    taker = side if buy else ("no" if side=="yes" else "yes")
    for trade in sorted(trade_list(client,ticker,when+2,stop),key=lambda t:timestamp(t["created_time"])):
        ts = timestamp(trade["created_time"])
        if not when+2 <= ts < stop or trade.get("is_block_trade") or D(trade["count_fp"]) < 1:
            continue
        if (trade.get("taker_outcome_side") or trade.get("taker_side")) != taker:
            continue
        price = D(trade[f"{side}_price_dollars"])
        if (buy and price <= limit) or (not buy and price >= limit):
            return ts, trade["trade_id"]
    return None


def replay(client, event):
    fixture, event_id, market, kalshi, provider, _ = event
    decision = provider + 600
    ticker = market["ticker"]
    close = timestamp(market["close_time"])
    row = dict(fixture=fixture,event=event_id,ticker=ticker,decision=decision)
    if abs(kalshi-provider)>3600:
        return dict(row,status="schedule_disagreement")
    if timestamp(market["open_time"]) > decision-300 or close <= decision+2:
        return dict(row,status="market_not_open_during_window")
    trades = trade_list(client,ticker,decision-300,decision)
    side, n, vol, yes = direction(trades,decision)
    row.update(prints=n, contracts=str(vol), yes_contracts=str(yes))
    if side is None:
        return dict(row,status="flow_filter")
    row.update(side=side)
    quote = read_quote(client,ticker,decision)
    if quote is None or decision-quote[2]>60 or quote[1]-quote[0]>D(".03"):
        return dict(row,status="entry_quote_filter")
    bid, ask, end = quote
    limit = ask if side=="yes" else 1-bid
    row.update(ts=decision,limit=str(limit),entry_quote_ts=end)
    hit = print_after(client,ticker,side,True,limit,decision,close)
    if not hit:
        return dict(row,status="entry_unfilled")
    row.update(entry_ts=hit[0],entry_print=hit[1])
    # A fractional-second fill should not shorten the specified holding time.
    first_exit = ceil(hit[0])+600
    for check in range(first_exit,first_exit+901,60):
        if check+2 >= close:
            break
        quote = read_quote(client,ticker,check)
        if quote is None or check-quote[2]>60:
            continue
        exit_limit = quote[0] if side=="yes" else 1-quote[1]
        sold = print_after(client,ticker,side,False,exit_limit,check,close)
        if sold:
            gross = exit_limit-limit
            fees = fee(limit)+fee(exit_limit)
            return dict(row,status="sold",exit_ts=sold[0],exit_print=sold[1],exit_limit=str(exit_limit),
                        gross=str(gross),fees=str(fees),pnl=str(gross-fees),exit_attempt=(check-first_exit)//60+1)
    payout = market.get("settlement_value_dollars")
    settled = market.get("settlement_ts")
    if payout is None or not settled or timestamp(settled)<hit[0]:
        return dict(row,status="open")
    proceeds = D(payout) if side=="yes" else 1-D(payout)
    if not 0 <= proceeds <= 1:
        raise ValueError("Invalid settlement payout")
    gross = proceeds-limit
    return dict(row,status="settled",exit_ts=timestamp(settled),gross=str(gross),
                fees=str(fee(limit)),pnl=str(gross-fee(limit)))


def summarize(rows, start, end):
    cash_ledger(rows)
    realized = [r for r in rows if r.get("funded") and "pnl" in r]
    n = len(realized)
    net = sum((D(r["pnl"]) for r in realized),D(0))
    covered_days = (end-start).days
    month_length = ((start.replace(day=28)+timedelta(days=4)).replace(day=1)-start.replace(day=1)).days
    bill = D(80)*covered_days/month_length
    stress = sum((max(D(0),min(D(1),D(r["limit"])+D(".01"))-D(r["limit"])) +
                  (min(D(".01"),D(r["exit_limit"])) if r["status"]=="sold" else 0) for r in realized),D(0))
    ceiling = sum((1-D(r["limit"])-fee(D(r["limit"])) for r in rows if "ts" in r),D(0))
    return dict(start=str(start),end_exclusive=str(end),matched=len(rows),all_wins_ceiling=str(ceiling),
                coverage=dict(Counter(r["status"] for r in rows)),orders=sum("ts" in r for r in rows),
                cash_blocked=sum(r.get("funded") is False for r in rows),realized=n,
                wins=sum(D(r["pnl"])>0 for r in realized),net_dollars=str(net),
                fees=str(sum((D(r["fees"]) for r in realized),D(0))),
                cashout_net=str(sum((D(r["pnl"]) for r in realized if r["status"]=="sold"),D(0))),
                net_less_1c_per_fill=str(net-stress),allocated_bill=str(bill),
                pnl_after_bill=str(net-bill))


def main():
    # Include date buffers to account for the different decision anchor in the
    # shared event matcher. The membership test is on our decision timestamp.
    results = []
    with httpx.Client(timeout=45) as client:
        fixtures = history(client, PERIODS[-1][1]+timedelta(days=4))
        for start,end in [*PERIODS,CHECK]:
            cohort = matched_events(client,fixtures,start-timedelta(days=1),end+timedelta(days=1),Counter())
            eligible = [ev for ev in cohort if midnight(start)<=ev[4]+600<midnight(end)]
            print(f"{start}: reading public flow for {len(eligible)} distinct scheduled matches",flush=True)
            with ThreadPoolExecutor(max_workers=3) as workers:
                rows = list(workers.map(partial(replay,client),eligible))
            result = summarize(rows,start,end)
            save(OUT/f"{start}-ledger.json",rows)
            results.append(result)
            print(result,flush=True)
            save(OUT/"summary.json",dict(rule=RULE,periods=results))


if __name__=="__main__":
    main()
