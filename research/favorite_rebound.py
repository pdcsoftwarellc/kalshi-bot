"""One-contract, two-sided execution study of a dip and rebound in Challenger favorites."""
import argparse
from collections import Counter, defaultdict
from datetime import date, timedelta
from decimal import Decimal as D
import hashlib
import json
from pathlib import Path
import sys

import httpx

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from research.api_cost import ROOT, PERIODS, fee, fetch, link_market, midnight, pages, save, timestamp
from research.forward_odds import tennis

SERIES = "KXATPCHALLENGERMATCH"
CLIP = SERIES+"-26NOV17ROCAGU"
CLIP_END = int(timestamp("2026-11-17T19:51:08+00:00"))
OUT = ROOT/"data/processed/parallel-research/favorite_rebound"
CACHE = ROOT/"data/raw/challenger-rebound"
ZERO = D("0")
ONE = D("1")


def fixtures(client, start, end):
    rows = {}
    while start <= end:
        last = min(start+timedelta(days=6),end)
        cache = CACHE / f"fixtures-{start}-{last}.json"
        if not cache.exists():
            source = tennis(client,"get_fixtures",event_type_key=281,date_start=str(start),date_stop=str(last),timezone="UTC")
            if not isinstance(source, (list,dict)) or isinstance(source,dict) and source:
                raise ValueError("Unexpected fixtures response")
            fields = ("event_key","event_date","event_time","event_first_player","event_second_player",
                      "first_player_key","second_player_key","tournament_name")
            save(cache,[{k:r.get(k) for k in fields} for r in source if r.get("event_type_type")=="Challenger Men Singles"])
        rows.update({r["event_key"]:r for r in json.loads(cache.read_text())})
        start=last+timedelta(days=1)
    return list(rows.values())


def sample(event):
    return int.from_bytes(hashlib.sha256(event.encode()).digest()[:4],"big") % 8 == 0


def market_pairs(client,start,end):
    by_event = defaultdict(dict)
    for m in pages(client, "/markets","markets",series_ticker=SERIES,status="settled",limit=1000,
                   min_close_ts=midnight(start),max_close_ts=midnight(end+timedelta(days=7))):
        by_event[m["event_ticker"]][m["ticker"]] = m
    return by_event


def quotes(client,ticker,start,end):
    if end<=start:
        return []
    candles=fetch(client,f"/series/{SERIES}/markets/{ticker}/candlesticks",
                  dict(start_ts=int(start),end_ts=int(end),period_interval=1))["candlesticks"]
    result=[]
    for bar in sorted(candles,key=lambda c:c["end_period_ts"]):
        t=bar["end_period_ts"]
        a=bar.get("yes_ask",{}).get("close_dollars")
        b=bar.get("yes_bid",{}).get("close_dollars")
        if start<=t<=end and a and b and ZERO<D(b)<=D(a)<ONE:
            result.append((t,D(b),D(a)))
    return result


def choose_favorite(client, pair, scheduled):
    anchor = scheduled - 600
    eligible = []
    for m in pair:
        bars = quotes(client,m["ticker"],max(anchor-300,int(timestamp(m["open_time"]))),anchor)
        if not bars:
            continue
        _,bid,ask = bars[-1]
        mid = (bid+ask)/2
        if mid>=D(".70") and ask-bid<=D(".04"):
            eligible.append((m,mid,bars[-1][0]))
    return eligible[0] if len(eligible)==1 else (None,None,None)


def signal(bars):
    """First confirmed bounce off a running trough; gaps reset the pattern."""
    history=[]
    peak=trough=None
    trough_ts=None
    for t,bid,ask in bars:
        if history and t-history[-1][0]>300:
            history=[]
            peak=trough=None
        history=[v for v in history if v[0]>=t-1800]
        mid=(ask+bid)/2
        if peak and (not history or peak[0]<t-1800):
            peak=trough=None
        if not peak and history:
            high=max(history,key=lambda x:x[1])
            if high[1]-mid>=D(".10"):
                peak,trough,trough_ts=high,mid,t
        if peak:
            if mid<trough:
                trough,trough_ts=mid,t
            elif mid>=peak[1]:
                peak=trough=None
            elif (t>trough_ts and mid-trough>=D(".02") and peak[1]-mid>=D(".05")
                  and ask-bid<=D(".04") and ask<=D(".91")):
                return dict(ts=t,limit=ask,peak=peak[1],trough=trough,trough_ts=trough_ts)
        history.append((t,mid))
    return None


def evidence(client,ticker,action,limit,decision,end):
    """Buy YES on YES taker prints, sell YES on NO taker prints at/through limit."""
    stop=min(decision+62,end)
    if stop<=decision+2:
        return None
    stream=pages(client,"/markets/trades","trades",ticker=ticker,min_ts=decision+2,max_ts=int(stop),limit=1000)
    for trade in sorted(stream,key=lambda x:timestamp(x["created_time"])):
        t=timestamp(trade["created_time"])
        if (not decision+2<=t<stop or trade.get("is_block_trade") or D(trade["count_fp"])<1):
            continue
        side=trade.get("taker_outcome_side") or trade.get("taker_side")
        price=D(trade["yes_price_dollars"])
        if (action=="buy" and side=="yes" and price<=limit) or (action=="sell" and side=="no" and price>=limit):
            return (t,trade["trade_id"])
    return None


def replay(client,pair,fixture,cutoff=None,fill=evidence):
    scheduled=int(timestamp(fixture["event_date"]+"T"+fixture["event_time"]+"+00:00"))
    m,prior,prior_ts=choose_favorite(client,pair,scheduled)
    if not m:
        return dict(status="no_valid_favorite")
    ticker=m["ticker"]
    end=int(min(timestamp(m["close_time"]),cutoff or timestamp(m["close_time"])))
    # Observe up to 3h post-start for entry, plus an hour to work an exit.
    bars=quotes(client,ticker,scheduled,min(scheduled+4*3600,end))
    entry=signal([v for v in bars if v[0]<=scheduled+3*3600])
    row=dict(ticker=ticker,prior=str(prior),prior_quote_ts=prior_ts,bars=len(bars),scheduled_ts=scheduled)
    if not entry:
        return dict(row,status="no_entry_signal")
    row.update({k:str(v) if isinstance(v,D) else v for k,v in entry.items()})
    hit=fill(client,ticker,"buy",entry["limit"],entry["ts"],end)
    if not hit:
        return dict(row,status="entry_unfilled")
    buy=entry["limit"]
    row.update(entry_fill_ts=hit[0],entry_print=hit[1],entry_fee=str(fee(buy)))
    trigger=None
    for t,bid,ask in bars:
        if t<=hit[0]:
            continue
        if not trigger:
            if bid>=buy+D(".08"):
                trigger="target"
            elif bid<=buy-D(".08"):
                trigger="stop"
            elif t>=hit[0]+1800:
                trigger="timeout"
        if trigger:
            # One working sell at a time; latest bid replaces the preceding
            # limit after its one-minute evidence window.
            sold=fill(client,ticker,"sell",bid,t,end)
            if sold:
                return dict(row,status="sold",exit_reason=trigger,exit_ts=sold[0],exit_limit=str(bid),
                            exit_print=sold[1],gross=str(bid-buy),fees=str(fee(buy)+fee(bid)),pnl=str(bid-buy-fee(buy)-fee(bid)))
    if cutoff or not m.get("settlement_ts") or m.get("settlement_value_dollars") is None:
        return dict(row,status="position_open",exit_reason=trigger)
    payout=D(m["settlement_value_dollars"])
    if not 0<=payout<=1 or timestamp(m["settlement_ts"])<=hit[0]:
        raise ValueError("Invalid settlement")
    return dict(row,status="settled",exit_reason=trigger or "no_exit_before_close",exit_ts=timestamp(m["settlement_ts"]),
                exit_limit=str(payout),gross=str(payout-buy),fees=str(fee(buy)),pnl=str(payout-buy-fee(buy)))


def cash_ledger(rows):
    """Reserve the limit cost when the signal fires and release at expiry/exit."""
    from heapq import heappop, heappush
    releases=[]
    cash=D("25")
    for row in sorted((r for r in rows if "ts" in r),key=lambda r:(r["ts"],r["ticker"])):
        while releases and releases[0][0]<=row["ts"]:
            cash+=heappop(releases)[1]
        cost=D(row["limit"])+fee(D(row["limit"]))
        row["funded"]=cash>=cost
        if row["funded"]:
            cash-=cost
            if "pnl" in row:
                heappush(releases,(row["exit_ts"],cost+D(row["pnl"])))
            elif row["status"]=="entry_unfilled":
                heappush(releases,(row["ts"]+62,cost))


def historical(client,start,end):
    calendar=fixtures(client,start-timedelta(days=1),end)
    pairs=market_pairs(client,start,end)
    rows=[]
    counts=Counter()
    for event,mm in sorted(pairs.items()):
        if not sample(event) or len(mm)!=2:
            continue
        pair=sorted(mm.values(),key=lambda v:v["ticker"])
        if not pair[0].get("occurrence_datetime"):
            continue
        approx=timestamp(pair[0]["occurrence_datetime"])
        nearby=[r for r in calendar if r.get("event_time") and
                abs(timestamp(r["event_date"]+"T"+r["event_time"]+"+00:00")-approx)<=36*3600]
        link=link_market(pair,nearby)
        if not link:
            counts["unlinked_in_buffer"]+=1
            continue
        f=link[0]
        if not str(start)<=f["event_date"]<str(end):
            continue
        rows.append(dict(replay(client,pair,f),event=event,fixture=f["event_key"]))
    cash_ledger(rows)
    closed=[r for r in rows if r.get("funded") and "pnl" in r]
    counts.update(r["status"] for r in rows)
    result=dict(start=str(start),end_exclusive=str(end),sample="SHA256 first four bytes mod 8 == 0",
                matched_events=len(rows),coverage=dict(counts),signals=sum("ts" in r for r in rows),
                blocked=sum(r.get("funded") is False for r in rows),closed=len(closed),
                cashouts=sum(r["status"]=="sold" for r in closed),wins=sum(D(r["pnl"])>0 for r in closed),
                gross=str(sum((D(r["gross"]) for r in closed),ZERO)),
                fees=str(sum((D(r["fees"]) for r in closed),ZERO)),net=str(sum((D(r["pnl"]) for r in closed),ZERO)))
    save(OUT/f"{start}.json",dict(summary=result,ledger=rows))
    print(json.dumps(result),flush=True)


def clip(client):
    pair=[fetch(client,"/markets/"+CLIP+"-"+side,{})["market"] for side in ("ROC","AGU")]
    f=link_market(pair,fixtures(client,date(2026,11,17),date(2026,11,17)))
    if not f:
        raise ValueError("Recording fixture did not uniquely match")
    report=replay(client,pair,f[0],CLIP_END)
    save(OUT/"rocha-clip.json",report)
    print("Clip",json.dumps(report),flush=True)


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--clip-only",action="store_true")
    args=parser.parse_args()
    with httpx.Client(timeout=45) as client:
        clip(client)
        if not args.clip_only:
            for s,e in PERIODS:
                historical(client,s,e)
