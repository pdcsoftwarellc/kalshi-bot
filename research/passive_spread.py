"""ATP passive spread capture with trade-through evidence, inventory settlement and a taker control."""
from collections import Counter
from datetime import date, timedelta
from decimal import Decimal, ROUND_CEILING
import heapq
import json
import math
from pathlib import Path
import sys

import httpx

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from research.api_cost import ROOT, PERIODS, SERIES, D, fee, fetch, history, midnight, pages, save, timestamp
from research.price_drift import matched_events, read_quote

OUT = ROOT/"data/processed/parallel-round2/passive_spread"
WINDOWS = [*PERIODS, (date(2026,9,19), date(2026,10,1))]
CENT = D(".01")
METHOD = dict(
    signal="One YES contract per lexicographically first ticker in a matched ATP pair. Observe 30m before earlier verified schedule. Require a fresh <=5m minute quote and 3–6c spread inclusive.",
    maker_entry="Post initial best bid after 2s for remainder of 5-minute window. Require non-block NO taker print of >=1 contract with YES equivalent price at least 1c below bid.",
    maker_exit="Original ask offered after first complete two-sided quote from entry+2s to entry+62s only if bid < original ask. Require a YES taker print >=1c above original ask and >=1 contract from post+2s until 10min after original entry. Otherwise settle the inventory.",
    taker_control="One YES at same event's original ask on later YES taker print <= limit (+2s to +5min). At fresh post-entry minute bids at/above original ask, attempt to sell at observed bid on NO taker print >= that bid within the next minute after 2s, through entry+10min. Otherwise settle inventory.",
    fees="Maker 0.0175 * P * (1-P), multiplier 1, rounded up to whole cent each leg. Taker control and stressed fees 0.07 * P * (1-P), likewise rounded up. No exit fee on settlement.",
    limitations="Later public trades and historical minute closes do not reveal queue priority, successful post-only acceptance or book state at the exact simulated arrival. The maker and taker arms can fill different events.",
    fees_source="https://kalshi.com/docs/kalshi-fee-schedule.pdf",
)


def maker_fee(price):
    return (D(".0175")*price*(1-price)).quantize(CENT,rounding=ROUND_CEILING)


def price_hit(trades, *, side, bound, at_or_below, earliest, end):
    """Eligible trade evidence, timestamp-sorted. Caller chooses sided price and threshold."""
    for ts, direction, price, ident in trades:
        if earliest <= ts < end and direction == side and (price<=bound if at_or_below else price>=bound):
            return ts,ident
    return None


def tape(client,ticker,decision):
    out=[]
    for r in pages(client,"/markets/trades","trades",ticker=ticker,min_ts=decision+2,
                   max_ts=decision+900,limit=1000):
        if r.get("is_block_trade") or D(r["count_fp"])<1:
            continue
        out.append((timestamp(r["created_time"]),r.get("taker_outcome_side") or r.get("taker_side"),
                    D(r["yes_price_dollars"]),r["trade_id"]))
    return sorted(out)


def completed_quotes(client,ticker,t):
    data=fetch(client,f"/series/{SERIES}/markets/{ticker}/candlesticks",
               dict(start_ts=t,end_ts=t+900,period_interval=1))
    result=[]
    for bar in data["candlesticks"]:
        when=bar["end_period_ts"]
        bid=bar.get("yes_bid",{}).get("close_dollars")
        ask=bar.get("yes_ask",{}).get("close_dollars")
        if t < when <= t+900 and bid and ask and 0<D(bid)<=D(ask)<1:
            result.append((when,D(bid),D(ask)))
    return sorted(result)


def order(mode,base,bid,ask,trades,quotes,settlement,settle_time):
    passive=mode=="maker"
    t=base["ts"]
    limit=bid if passive else ask
    entry=price_hit(trades, side="no" if passive else "yes", bound=bid-CENT if passive else ask,
                    at_or_below=True, earliest=t+2, end=t+300)
    row=dict(base,mode=mode,limit=str(limit),entry_print=None,entry_fill_ts=None)
    if not entry:
        return dict(row,status="missed_entry",release=t+300)
    entry_time,tid=entry
    row.update(entry_print=tid,entry_fill_ts=entry_time)
    deadline=entry_time+600
    proceeds=settlement
    exit_found=None
    reason="no_exit_trade"
    observations=[q for q in quotes if entry_time+2<=q[0]<deadline]
    if passive:
        if not observations or observations[0][0]>entry_time+62:
            reason="no_post_entry_quote_within_one_minute"
        elif observations[0][1]>=ask:
            reason="post_only_exit_would_cross"
        else:
            when=observations[0][0]
            row["offered_at"]=when
            exit_found=price_hit(trades,side="yes",bound=ask+CENT,at_or_below=False,
                                 earliest=when+2,end=deadline)
            if exit_found:
                proceeds=ask
    else:
        for when,market_bid,_ in observations:
            if market_bid>=ask:
                exit_found=price_hit(trades,side="no",bound=market_bid,at_or_below=False,
                                     earliest=when+2,end=min(when+62,deadline))
                if exit_found:
                    proceeds=market_bid
                    break
    if exit_found:
        exit_t,exit_id=exit_found
        row.update(status="sold",exit_print=exit_id,release=exit_t)
    else:
        row.update(status="settled",exit_reason=reason,release=settle_time)
    row["proceeds"]=str(proceeds)
    return row


def portfolio(rows,fee_fn):
    cash=D("25")
    releases=[]
    fills=[]
    blocked=0
    for r in sorted(rows,key=lambda r:(r["ts"],r["ticker"])):
        while releases and releases[0][0]<=r["ts"]:
            cash+=heapq.heappop(releases)[1]
        limit=D(r["limit"])
        cost=limit+fee_fn(limit)
        if cash<cost:
            blocked+=1
            continue
        cash-=cost
        if r["status"]=="missed_entry":
            heapq.heappush(releases,(r["release"],cost))
            continue
        proceeds=D(r["proceeds"])
        sell_fee=fee_fn(proceeds) if r["status"]=="sold" else D(0)
        heapq.heappush(releases,(r["release"],proceeds-sell_fee))
        fills.append(dict(r,entry_fee=str(fee_fn(limit)),exit_fee=str(sell_fee),pnl=str(proceeds-sell_fee-cost)))
    return fills, blocked


def money(fills,field):
    return sum((D(r[field]) for r in fills),D(0))


def summarize(rows,arm,bill):
    fills,blocked=portfolio(rows,maker_fee if arm=="maker" else fee)
    roundtrips=[r for r in fills if r["status"]=="sold"]
    inventory=[r for r in fills if r["status"]=="settled"]
    pnl=money(fills,"pnl")
    fees=money(fills,"entry_fee")+money(fills,"exit_fee")
    stress,_=portfolio(rows,fee)
    summary=dict(attempts=len(rows),fills=len(fills),missed_entry=len(rows)-len(fills)-blocked,
                 cash_blocked=blocked,roundtrips=len(roundtrips),inventory_settled=len(inventory),
                 inventory_wins=sum(D(r["pnl"])>0 for r in inventory),roundtrip_net=str(money(roundtrips,"pnl")),
                 unmatched_inventory_net=str(money(inventory,"pnl")),gross=str(pnl+fees),fees=str(fees),
                 net=str(pnl),taker_fee_stress_net=str(money(stress,"pnl")),net_after_api=str(pnl-bill))
    return fills,summary


def run_period(client,fixtures,start,end):
    counts=Counter()
    arms={"maker":[],"taker":[]}
    for key,event,m,planned,provider,ref in matched_events(client,fixtures,start,end,counts):
        t=ref-1800
        ticker=m["ticker"]
        counts["linked_in_period"]+=1
        if timestamp(m["open_time"])>t or timestamp(m["close_time"])<=t+900:
            counts["unavailable_during_window"]+=1
            continue
        if not m.get("settlement_ts") or m.get("settlement_value_dollars") is None or timestamp(m["settlement_ts"])<=t+900:
            counts["no_valid_settlement"]+=1
            continue
        val=D(m["settlement_value_dollars"])
        if not 0<=val<=1:
            raise ValueError("Nonbinary market or invalid settlement")
        q=read_quote(client,ticker,t)
        if not q:
            counts["no_current_two_sided_quote"]+=1
            continue
        counts["fresh_quotes"]+=1
        bid,ask,quote_ts=q
        if not D(".03")<=ask-bid<=D(".06"):
            counts["outside_spread_band"]+=1
            continue
        tr=tape(client,ticker,t)
        qs=completed_quotes(client,ticker,t) if tr else []
        base=dict(fixture=key,event=event,ticker=ticker,ts=t,bid=str(bid),ask=str(ask),
                  quote_ts=quote_ts,provider=provider,exchange=planned)
        for arm in arms:
            arms[arm].append(order(arm,base,bid,ask,tr,qs,val,timestamp(m["settlement_ts"])))
    month_end=(start.replace(day=28)+timedelta(days=4)).replace(day=1)
    bill=D(80)*(end-start).days/(month_end-start.replace(day=1)).days
    report=dict(start=str(start),end_exclusive=str(end),coverage=dict(counts),bill=str(bill))
    ledgers={}
    for k,rows in arms.items():
        fills,report[k]=summarize(rows,k,bill)
        ledgers[k]=dict(orders=rows,fills=fills)
    save(OUT/f"{start}-ledger.json",dict(summary=report,**ledgers))
    return report


def main():
    results=[]
    with httpx.Client(timeout=40) as c:
        details=fetch(c,f"/series/{SERIES}",{})["series"]
        if details.get("fee_type")!="quadratic_with_maker_fees" or D(str(details.get("fee_multiplier")))!=1:
            raise RuntimeError("Check fee model before running")
        save(OUT/"method.json",dict(METHOD,observed_fee_type=details["fee_type"],observed_multiplier=details["fee_multiplier"]))
        fixtures=history(c,PERIODS[-1][1]+timedelta(days=4))
        for start,end in WINDOWS:
            r=run_period(c,fixtures,start,end)
            print(json.dumps(r),flush=True)
            results.append(r)
            save(OUT/"summary.json",results)


if __name__=="__main__":
    main()
