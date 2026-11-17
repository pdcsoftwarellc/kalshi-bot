"""Screen for discounted two-player YES baskets using public ATP market data.

Periods are assigned by common opening time. SHA-256 selection is independent
of settlement and price history. Each observed pair is a separate two-leg
position. A pre-start cancellation can settle each leg at its own fair price.
"""
from collections import Counter, defaultdict
from datetime import timedelta
from decimal import Decimal as D
import hashlib
import heapq
import json
import math
from pathlib import Path
import sys
import time

import httpx

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from research.api_cost import ROOT, SERIES, PERIODS, config, fee, midnight, pages, save, timestamp

OUT = ROOT / "data/processed/parallel-research/paired_prices"
CACHE = OUT / "inputs"
SAMPLE = 40
LIMIT = D(".98")


def public_read(client, path, params=None):
    for attempt in range(5):
        response = client.get(config.KALSHI_PROD_BASE + path, params=params)
        if response.status_code == 429 or response.status_code >= 500:
            time.sleep(2 ** attempt)
            continue
        response.raise_for_status()
        return response.json()
    raise RuntimeError(f"Could not read {path} after retries")


def sample(client, start, end):
    groups = defaultdict(dict)
    # Recover period listings from the settled-market universe, with a seven-day close buffer.
    for m in pages(client, "/markets", "markets", series_ticker=SERIES, status="settled", limit=1000,
                   min_close_ts=midnight(start), max_close_ts=midnight(end + timedelta(days=7))):
        groups[m["event_ticker"]][m["ticker"]] = m
    pairs = []
    for event, by_ticker in groups.items():
        if len(by_ticker) != 2:
            continue
        pair = sorted(by_ticker.values(), key=lambda m: m["ticker"])
        opening = max(timestamp(m["open_time"]) for m in pair)
        if midnight(start) <= opening < midnight(end):
            pairs.append((event, pair))
    pairs.sort(key=lambda p: hashlib.sha256(p[0].encode()).digest())
    return len(pairs), pairs[:SAMPLE]


def inputs(client, event, pair, end):
    start_ts = math.ceil(max(timestamp(m["open_time"]) for m in pair))
    end_ts = int(min(start_ts + 48*3600, midnight(end), *(timestamp(m["close_time"]) for m in pair)))
    cache = CACHE / f"{event}-{start_ts}-{end_ts}.json"
    if cache.exists():
        return json.loads(cache.read_text())
    ev = public_read(client, "/events/" + event)
    tickers = [m["ticker"] for m in pair]
    e = ev["event"]
    ok = (e["series_ticker"] == SERIES and e["event_ticker"] == event and e["mutually_exclusive"] is True
          and sorted(m["ticker"] for m in ev["markets"]) == tickers
          and len({m.get("custom_strike",{}).get("tennis_competitor") for m in pair}) == 2
          and all(m.get("custom_strike",{}).get("tennis_competitor") and m["market_type"] == "binary"
                  and D(m["notional_value_dollars"]) == 1 and m["yes_sub_title"]
                  and m["rules_primary"].startswith("If " + m["yes_sub_title"] + " wins ")
                  for m in pair))
    values = dict(rule_verified=bool(ok), event=event, tickers=tickers,
                  settlement=[m.get("settlement_value_dollars") for m in pair],
                  settlement_ts=[m.get("settlement_ts") for m in pair], start_ts=start_ts, end_ts=end_ts,
                  rules=[(m["rules_primary"],m["rules_secondary"]) for m in pair], minute_asks={})
    if ok and start_ts < end_ts:
        response = public_read(client, "/markets/candlesticks", dict(
            market_tickers=",".join(tickers), start_ts=start_ts, end_ts=end_ts, period_interval=1))
        for market in response["markets"]:
            # A missing bar on one leg cannot be filled in from an earlier close.
            bars = {}
            for c in market["candlesticks"]:
                t = c["end_period_ts"]
                a = c.get("yes_ask",{}).get("close_dollars")
                b = c.get("yes_bid",{}).get("close_dollars")
                if start_ts < t and t+7 < end_ts and a is not None and 0 < D(a) < 1 and (b is None or 0 <= D(b) <= D(a)):
                    bars[str(t)] = a
            values["minute_asks"][market["market_ticker"]] = bars
    save(cache,values)
    return values


def scan(values):
    books = [values["minute_asks"].get(t,{}) for t in values["tickers"]]
    common = sorted(books[0].keys() & books[1].keys(),key=int)
    best = None
    signal = None
    # Track coverage and minimum even after the first decision. Only that first
    # decision enters the simulated ledger.
    for key in common:
        prices = [D(levels[key]) for levels in books]
        cost = sum((p + fee(p) for p in prices),D(0))
        best = min(best,cost) if best is not None else cost
        if signal is None and cost <= LIMIT:
            signal = dict(ts=int(key), prices=list(map(str,prices)), total_limit_cost=str(cost))
    return len(common),best,signal


def fill_proxy(rows, price, ts, ticker):
    for row in sorted(rows,key=lambda r:timestamp(r["created_time"])):
        if (row.get("ticker") == ticker and not row.get("is_block_trade")
            and (row.get("taker_outcome_side") or row.get("taker_side")) == "yes"
            and ts + 2 <= timestamp(row["created_time"]) < ts + 7
            and D(row["count_fp"]) >= 1 and D(row["yes_price_dollars"]) <= price):
            return row["trade_id"]
    return None


def cash_ledger(signals):
    cash = D(25)
    releases = []
    gross = fees = paired = single = D(0)
    counts = Counter()
    for s in sorted(signals,key=lambda s:(s["ts"],s["event"])):
        ts = s["ts"]
        while releases and releases[0][0] <= ts:
            cash += heapq.heappop(releases)[1]
        costs = [D(p)+fee(D(p)) for p in s["prices"]]
        if cash < sum(costs):
            counts["cash_blocked"] += 1
            continue
        cash -= sum(costs)
        fills = sum(bool(x) for x in s["prints"])
        counts[("no_fill","one_leg","two_legs")[fills]] += 1
        for i in range(2):
            if not s["prints"][i]:
                heapq.heappush(releases,(ts+7,costs[i]))
                continue
            if s["settlement"][i] is None or s["settlement_ts"][i] is None:
                raise ValueError("A simulated fill is missing an exchange payout")
            payout = D(s["settlement"][i])
            if not 0 <= payout <= 1:
                raise ValueError("Invalid payout")
            price = D(s["prices"][i])
            gross += payout-price
            fees += fee(price)
            if fills == 2:
                paired += payout-costs[i]
            else:
                single += payout-costs[i]
            heapq.heappush(releases,(timestamp(s["settlement_ts"][i]),payout))
    return dict(both_legs=counts["two_legs"],single_legs=counts["one_leg"],no_fills=counts["no_fill"],
                cash_blocked=counts["cash_blocked"],gross_dollars=float(gross), fees_dollars=float(fees),
                net_dollars=float(gross-fees),paired_net=float(paired),unhedged_net=float(single))


def study(client,start,end):
    eligible,pairs = sample(client,start,end)
    sampled = []
    signals = []
    for event,pair in pairs:
        v = inputs(client,event,pair,end)
        n,best,signal = scan(v)
        sampled.append(dict(event=event,verified=v["rule_verified"],minutes=n,
                            min_combined_cost=str(best) if best is not None else None,
                            observed_payouts=v["settlement"]))
        if signal is None:
            continue
        ts = signal["ts"]
        trades = []
        for i,ticker in enumerate(v["tickers"]):
            rows = pages(client,"/markets/trades","trades",ticker=ticker,min_ts=ts+2,max_ts=ts+7,limit=1000)
            trades.append(fill_proxy(rows,D(signal["prices"][i]),ts,ticker))
        signals.append(dict(signal,event=event,tickers=v["tickers"],settlement=v["settlement"],
                            settlement_ts=v["settlement_ts"],prints=trades))
    pnl = cash_ledger(signals)
    days_in_month = ((start.replace(day=28)+timedelta(days=4)).replace(day=1)-start).days
    bill = D(80)*(end-start).days/days_in_month
    costs = [D(r["min_combined_cost"]) for r in sampled if r["min_combined_cost"] is not None]
    result = dict(start=str(start),end_exclusive=str(end),paired_listings=eligible,sampled=len(sampled),
                  verified=sum(r["verified"] for r in sampled),events_with_paired_quotes=len(costs),
                  matched_minute_closes=sum(r["minutes"] for r in sampled),
                  lowest_combined_cost=str(min(costs)) if costs else None,signals=len(signals),
                  **pnl, bill=float(bill),after_bill=pnl["net_dollars"]-float(bill))
    save(OUT/f"{start}-ledger.json",dict(result=result,sample=sampled,signals=signals))
    return result


def main():
    with httpx.Client(timeout=30) as client:
        reports = []
        for start,end in PERIODS:
            report = study(client,start,end)
            print(json.dumps(report),flush=True)
            reports.append(report)
        save(OUT/"summary.json",reports)
        save(OUT/"method.json",dict(
            rule="One YES per player, first common minute close with total asks plus both fees at most $0.98.",
            universe="First 40 SHA-256-sorted event tickers per period. Both markets open in the period by the later listing time. Settled markets fetched with a seven-day close buffer.",
            horizon="First 48h while both markets open, ending at period boundary. Only minutes with both observed close asks. No forward-filling.",
            simulated_fills="Each leg independently needs an ordinary YES taker print of >=1 contract at/below its limit [2s, 7s) after the minute close. Buy at the limit. This cannot verify actual displayed depth or simultaneity.",
            pricing="General quadratic taker fee times one, rounded up separately to one cent. Each period starts with $25. Reserve both limits and fees, release unfilled legs at expiry and filled legs at settlement.",
            bill="The $80 October and $40 Nov 1–15 hurdles are standalone comparisons for this sampled strategy. The strategy itself uses public Kalshi data.",
            sources=["https://assets.kalshi.com/contract_terms/ACHIEVEMENTS.pdf",
                     "https://kalshi.com/markets/kxatpmatch/atp-tennis-match/kxatpmatch-26nov08shealc",
                     "https://docs.kalshi.com/api-reference/market/batch-get-market-candlesticks",
                     "https://kalshi.com/fee-schedule"],
        ))


if __name__ == "__main__":
    main()
