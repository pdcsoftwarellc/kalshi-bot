"""Fixed one-contract WTA favorite study using quotes, later public prints, and settlements."""
from collections import Counter, defaultdict
from datetime import date, datetime, timedelta
from decimal import Decimal
import json
from pathlib import Path
import sys

import httpx
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from research.api_cost import (ROOT, UTC, PERIODS, STARTING_CASH, fee, fetch, ledger,
                              link_market, midnight, pages, qualifying_print, save, timestamp)
from research.forward_odds import tennis

D = Decimal
DAY = timedelta(days=1)
SERIES = "KXWTAMATCH"
BANDS = (D(".80"), D(".90"))
WINDOWS = [*PERIODS, (date(2026, 9, 19), date(2026, 10, 1))]
OUT = ROOT / "data/processed/parallel-round2/wta_tail"
RAW = ROOT / "data/raw/wta-tail"
METHOD = dict(
    hypothesis="Prematch longshot demand underprices WTA favorites.",
    entry="Buy one direct player YES at a fresh 80–90 cent ask, inclusive, spread <=3 cents.",
    decision="30 minutes before the earlier dated API-Tennis and Kalshi start.",
    quote="Last completed 1-minute candle at decision, no more than 300s old, valid two-sided market.",
    fill="YES, non-block, >=1 contract, at/below limit, +2s inclusive through +300s exclusive.",
    limits="One attempt per match, $25 starting cash per period, reserve until expiry or settlement.",
    fee="General quadratic taker x1, rounded up to a cent; settlement has no sell fee.",
    stress="Pay 1 cent more plus recalculated fee on the same print-supported orders; reapply cash constraints.",
    uncertainty="95% resampling interval of total net P&L from equal-sized resamples of UTC entry days, including inactive days. Conditional on historical period and fill model.",
    periods="October primary; November 1–15 later comparison; September 19–30 separate earlier comparison.",
)


def fixture_rows(client, start, end):
    rows = {}
    while start < end:
        stop = min(start+timedelta(days=6), end-DAY)
        cache = RAW / f"fixtures-{start}-{stop}.json"
        if not cache.exists():
            payload = tennis(client, "get_fixtures", event_type_key=266, timezone="UTC",
                             date_start=str(start), date_stop=str(stop))
            if not isinstance(payload, list) and payload != {}:
                raise ValueError("Unrecognized WTA fixture response")
            fields = ("event_key", "event_type_type", "event_date", "event_time", "event_first_player",
                      "event_second_player", "first_player_key", "second_player_key", "tournament_name")
            save(cache, [{key: f.get(key) for key in fields} for f in payload
                         if f.get("event_type_type") == "Wta Singles"])
        for f in json.loads(cache.read_text()):
            rows[str(f["event_key"])] = f
        start = stop+DAY
    return rows.values()


def paired_fixtures(client):
    start = min(a for a,b in WINDOWS)
    end = max(b for a,b in WINDOWS)
    by_day = defaultdict(list)
    for f in fixture_rows(client,start-2*DAY,end+8*DAY):
        by_day[date.fromisoformat(f["event_date"])].append(f)
    pairs = defaultdict(dict)
    for a,b in WINDOWS:
        for m in pages(client, "/markets", "markets", status="settled", series_ticker=SERIES,
                       min_close_ts=midnight(a), max_close_ts=midnight(b+7*DAY), limit=1000):
            pairs[m["event_ticker"]][m["ticker"]] = m
    linked = defaultdict(list)
    exchange_decisions = []
    for event, group in pairs.items():
        pair = sorted(group.values(),key=lambda m:m["ticker"])
        if not all(m.get("occurrence_datetime") for m in pair):
            continue
        planned = min(timestamp(m["occurrence_datetime"]) for m in pair)
        exchange_decisions.append(int(planned)-1800)
        if len(pair) != 2:
            continue
        day = datetime.fromtimestamp(planned, UTC).date()
        match = link_market(pair,[f for off in (-1,0,1) for f in by_day[day+off*DAY]])
        if not match or not match[0].get("event_time"):
            continue
        f = match[0]
        provider = timestamp(f["event_date"]+"T"+f["event_time"]+"+00:00")
        if abs(planned-provider) > 86400:
            continue
        t = int(min(planned,provider))-1800
        linked[str(f["event_key"])].append((event,pair,t,int(planned),int(provider)))
    return exchange_decisions, linked


def fresh_quote(candles,t):
    valid = [bar for bar in candles if t-300 <= bar["end_period_ts"] <= t]
    if not valid:
        return None
    bar = max(valid,key=lambda x:x["end_period_ts"])
    bid, ask = (bar.get(f"yes_{side}",{}).get("close_dollars") for side in ("bid","ask"))
    if bid is None or ask is None:
        return None
    bid, ask = D(bid), D(ask)
    if not D(0)<bid<=ask<D(1):
        return None
    return bid, ask, bar["end_period_ts"]


def select(quotes):
    high = [(ticker, q) for ticker,q in quotes.items() if q and q[0]>D(".50")]
    if len(high) != 1:
        return None
    ticker,(bid,ask,stamp) = high[0]
    if BANDS[0] <= ask <= BANDS[1] and ask-bid <= D(".03"):
        return ticker, bid, ask, stamp
    return None


def sensitivity(orders):
    rows = []
    for order in orders:
        price = D(order["limit"])+D(".01")
        rows.append(dict(order,limit=str(price),fee=str(fee(price))))
    return ledger(rows, STARTING_CASH)[1]["trading_pnl_dollars"]


def run_period(client,a,b,exchange_times,linked):
    exclusions=Counter()
    attempts=[]
    evaluated=[]
    for key,links in linked.items():
        if len(links)!=1:
            if any(midnight(a)<=x[2]<midnight(b) for x in links):
                exclusions["ambiguous_repeat_fixture"] += 1
            continue
        event,pair,t,planned,provider=links[0]
        if not midnight(a)<=t<midnight(b):
            continue
        if any(timestamp(m["open_time"])>t or timestamp(m["close_time"])<=t+300 for m in pair):
            exclusions["not_open_for_full_order_window"]+=1
            continue
        tickers=[m["ticker"] for m in pair]
        data=fetch(client,"/markets/candlesticks",dict(market_tickers=",".join(tickers),
                   start_ts=t-300,end_ts=t,period_interval=1))
        quotes={item["market_ticker"]:fresh_quote(item["candlesticks"],t) for item in data["markets"]}
        q={ticker:quotes.get(ticker) for ticker in tickers}
        row=dict(event=event,fixture=key,ts=t,exchange_start=planned,provider_start=provider,
                 quotes={k: [str(v[0]),str(v[1]),v[2]] if v else None for k,v in q.items()})
        evaluated.append(row)
        if not any(q.values()):
            exclusions["no_fresh_two_sided_quote"]+=1
            continue
        pick=select(q)
        if not pick:
            exclusions["no_unique_qualifying_ask"]+=1
            continue
        ticker,bid,ask,stamp=pick
        m=next(m for m in pair if m["ticker"]==ticker)
        settle=m.get("settlement_value_dollars")
        if settle is None or not D(0)<=D(settle)<=D(1) or not m.get("settlement_ts") or timestamp(m["settlement_ts"])<=t+300:
            exclusions["unusable_settlement"]+=1
            continue
        trades=pages(client,"/markets/trades","trades",ticker=ticker,min_ts=t+2,max_ts=t+300,limit=1000)
        hit=qualifying_print(trades,"yes",ask,t)
        attempts.append(dict(event=event,fixture=key,ts=t,ticker=ticker,side="yes",bid=str(bid),limit=str(ask),
                             fee=str(fee(ask)),quote_ts=stamp,payout=str(settle),
                             settlement_ts=timestamp(m["settlement_ts"]),print_ts=hit[0] if hit else None,
                             trade_id=hit[1] if hit else None))

    fills,totals=ledger(attempts,STARTING_CASH)
    all_fill=[dict(o,print_ts=o["ts"]+2) for o in attempts]
    pnl=D(str(totals["trading_pnl_dollars"]))
    last=(a.replace(day=28)+4*DAY).replace(day=1)
    cost=D(80)*(b-a).days/(last-a.replace(day=1)).days
    daily=np.zeros((b-a).days)
    for r in fills:
        daily[(datetime.fromtimestamp(r["ts"],UTC).date()-a).days]+=float(r["pnl"])
    rng=np.random.default_rng(12937)
    sampled=rng.choice(daily,size=(5000,len(daily)),replace=True).sum(axis=1)
    report=dict(start=str(a),end_exclusive=str(b),listed_pairs_or_unpaired_at_exchange_decision=sum(midnight(a)<=t<midnight(b) for t in exchange_times),
                linked_matches=len(evaluated)+exclusions["not_open_for_full_order_window"],evaluated=len(evaluated),
                exclusions=dict(exclusions),fresh_two_sided=sum(any(r["quotes"].values()) for r in evaluated),
                orders=len(attempts),**totals,winning_fills=sum(D(r["pnl"])>0 for r in fills),
                missed_orders=len(attempts)-len(fills)-totals["cash_blocked"],
                gross_dollars=float(pnl+D(str(totals["fees_dollars"]))),api_overhead=float(cost),
                after_api_dollars=float(pnl-cost),one_cent_worse_net=sensitivity(attempts),
                all_fill_scenario_dollars=ledger(all_fill,STARTING_CASH)[1]["trading_pnl_dollars"],
                all_fill_win_upper_dollars=float(sum((D(1)-D(o["limit"])-D(o["fee"]) for o in attempts),D(0))),
                resampled_day_total_interval=np.quantile(sampled,[.025,.975]).round(2).tolist())
    save(OUT/f"{a}-ledger.json",dict(summary=report,quotes=evaluated,orders=attempts,fills=fills))
    return report


def main():
    save(OUT/"method.json",METHOD)
    results=[]
    with httpx.Client(timeout=40) as c:
        cutoff=fetch(c,"/historical/cutoff",{})
        if max(timestamp(cutoff[k]) for k in ("market_settled_ts","trades_created_ts"))>=midnight(min(w[0] for w in WINDOWS)):
            raise SystemExit("A fresh run needs archive tier endpoints for these dates.")
        times,linked=paired_fixtures(c)
        print(f"Uniquely paired in buffered study universe: {sum(len(v)==1 for v in linked.values())}",flush=True)
        for a,b in WINDOWS:
            r=run_period(c,a,b,times,linked)
            results.append(r)
            save(OUT/"summary.json",results)
            print(json.dumps(r),flush=True)


if __name__=="__main__":
    main()
