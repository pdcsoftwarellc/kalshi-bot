"""Historical candidates for the consistency relations in consistency.py, from one-minute bars.

A minute counts only if every leg has a bar with a two-sided quote at that minute,
the basket is profitable after exact fees at the bar-close quotes, AND it is still
profitable at the next minute's opening quotes (which must also exist for every leg).
Bars carry no sizes and their quotes need not have coexisted: these are candidates.
Every counted basket is settled at the markets' actual settlement values.

Families: soccer (three-way result), nba (winner + spread ladder, totals ladder),
temp (daily high-temperature range ladders). One worker, cached reads.
"""
import argparse
from collections import defaultdict
from datetime import date, timedelta
from pathlib import Path
from statistics import mean, median
import sys
import time

import httpx

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from research.api_cost import COLLECTION, ROOT, fetch, midnight, pages, save, timestamp
from research.consistency import basket, evaluate, relations
from research.near_end import cents

OUT = ROOT / "data/processed/consistency/history"
SOCCER = ["KXEPLGAME", "KXUCLGAME", "KXLALIGAGAME", "KXSERIEAGAME", "KXBUNDESLIGAGAME", "KXLIGUE1GAME",
          "KXMLSGAME", "KXLIGAMXGAME"]
TEMP = ["KXHIGHNY", "KXHIGHLAX", "KXHIGHMIA", "KXHIGHCHI", "KXHIGHAUS", "KXHIGHDEN", "KXHIGHPHIL"]
PAUSE = .3


def cached(path, params):
    import hashlib, json
    key = hashlib.sha256(json.dumps([path, params], sort_keys=True).encode()).hexdigest()
    return (COLLECTION / (key + ".json")).exists()


def get(client, path, params):
    hit = cached(path, params)
    data = fetch(client, path, params)
    if not hit:
        time.sleep(PAUSE)
    return data


def settled(client, series, start, end):
    events = defaultdict(list)
    for m in pages(client, "/markets", "markets", series_ticker=series, status="settled", limit=1000,
                   min_close_ts=midnight(start), max_close_ts=midnight(end)):
        events[m["event_ticker"]].append(m)
    time.sleep(PAUSE)
    return events


def bars(client, tickers, start_ts, end_ts):
    """ticker -> {minute end ts: (close bid, close ask, open bid, open ask)} in cents; None when one-sided."""
    out = {t: {} for t in tickers}
    minutes = max(1, (end_ts - start_ts) // 60)
    chunk = max(1, 9000 // minutes)
    for i in range(0, len(tickers), chunk):
        data = get(client, "/markets/candlesticks", dict(market_tickers=",".join(tickers[i:i + chunk]),
                   start_ts=start_ts, end_ts=end_ts, period_interval=1))
        for mk in data.get("markets", []):
            for b in mk["candlesticks"]:
                out[mk["market_ticker"]][b["end_period_ts"]] = tuple(
                    cents(b, side, field) for field in ("close", "open") for side in ("yes_bid", "yes_ask"))
    return out


def quote(bid, ask):
    ok = bid is not None and ask is not None and 0 < bid <= ask < 100
    return (bid, 1, ask, 1) if ok else (None, 0, None, 0)


def study(client, name, groups, hours, limit, flags=()):
    """groups: list of market lists that belong together (one game or one event)."""
    rows, episodes = [], []
    done = minutes_seen = 0
    for ms in groups[:limit]:
        close = int(max(timestamp(m["close_time"]) for m in ms))
        tickers = sorted(m["ticker"] for m in ms)
        try:
            B = bars(client, tickers, close - hours * 3600, close)
        except RuntimeError as err:
            print("skip", ms[0]["event_ticker"], err, flush=True)
            continue
        done += 1
        rel = relations(ms, set(flags))
        value = {m["ticker"]: (None if m.get("settlement_value_dollars") in (None, "") else
                               100 * float(m["settlement_value_dollars"])) for m in ms}
        result = {m["ticker"]: m.get("result") for m in ms}
        stamps = sorted({ts for t in tickers for ts in B[t]})
        minutes_seen += len(stamps)
        runs = defaultdict(list)                  # basket id -> [(ts, profit close, profit next open, basket)]
        for ts in stamps:
            Q = {t: quote(B[t][ts][0], B[t][ts][1]) for t in tickers if ts in B[t]}
            for b in evaluate(rel, Q, 0.0):
                if b["profit_exact"] <= 0:
                    continue
                legs = [(l["ticker"], l["side"]) for l in b["legs"]]
                nxt = {t: quote(B[t][ts + 60][2], B[t][ts + 60][3]) for t, _ in legs if ts + 60 in B[t]}
                after = basket(b["kind"], b["key"], legs, b["payout"], nxt)
                if after is None or after["profit_exact"] <= 0:
                    continue
                runs[b["id"]].append((ts, b, after))
        for bid_, seen in runs.items():
            seen.sort(key=lambda r: r[0])
            first = seen[0][1]
            legs = [(l["ticker"], l["side"]) for l in first["legs"]]
            pay = None
            if all(value[t] is not None for t, _ in legs):
                pay = sum(value[t] if s == "yes" else 100 - value[t] for t, s in legs)
            run, last = [], None
            spans = []
            for ts, b, after in seen:
                if last is not None and ts - last > 60:
                    spans.append(run)
                    run = []
                run.append((ts, after))
                last = ts
            spans.append(run)
            for run in spans:
                entry = run[0][1]                 # filled at the next-minute opening quotes of the first minute
                episodes.append(dict(family=name, series=first["family"], kind=first["kind"], key=first["key"],
                                     legs=entry["legs"], start_ts=run[0][0], minutes=len(run),
                                     minutes_to_close=(close - run[0][0]) // 60,
                                     entry_cost=entry["cost"], promised=first["payout"],
                                     profit_exact=entry["profit_exact"], profit_rounded=entry["profit_rounded"],
                                     best_profit_exact=max(a["profit_exact"] for _, a in run),
                                     settled_payout=pay, results=[result[t] for t, _ in legs],
                                     realised_exact=None if pay is None else round(pay - entry["cost"] - (entry["gross"] - entry["profit_exact"]), 3),
                                     short_pay=pay is not None and pay < first["payout"] - 1e-6))
        rows.append(dict(event=ms[0]["event_ticker"], markets=len(ms), minutes=len(stamps),
                         baskets=len(runs)))
        if done % 20 == 0:
            print(name, done, "groups", len(episodes), "episodes", flush=True)
    return dict(family=name, groups=done, minutes=minutes_seen, episodes=episodes, coverage=rows)


def report(res, days):
    ep = res["episodes"]
    out = dict(family=res["family"], groups=res["groups"], group_minutes=res["minutes"], episodes=len(ep),
               groups_with_episode=len({e["key"].split("|")[0] for e in ep}))
    if ep:
        durations = [e["minutes"] for e in ep]
        out.update(episodes_per_week=round(len(ep) / days * 7, 1),
                   median_profit_exact=median(e["profit_exact"] for e in ep),
                   mean_profit_exact=round(mean(e["profit_exact"] for e in ep), 3),
                   max_profit_exact=max(e["profit_exact"] for e in ep),
                   positive_after_rounded_fees=sum(e["profit_rounded"] > 0 for e in ep),
                   median_minutes=median(durations), max_minutes=max(durations),
                   one_minute_only=sum(d == 1 for d in durations),
                   settled=sum(e["settled_payout"] is not None for e in ep),
                   short_pay=sum(e["short_pay"] for e in ep),
                   mean_realised_exact=round(mean(e["realised_exact"] for e in ep if e["realised_exact"] is not None), 3),
                   by_kind={k: sum(e["kind"] == k for e in ep) for k in sorted({e["kind"] for e in ep})},
                   by_series={k: sum(e["series"] == k for e in ep) for k in sorted({e["series"] for e in ep})},
                   last_two_hours=sum(e["minutes_to_close"] <= 150 for e in ep))
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--families", default="soccer,nba,temp")
    parser.add_argument("--start", default="2026-10-20")
    parser.add_argument("--end", default="2026-11-16")
    parser.add_argument("--soccer", type=int, default=120)
    parser.add_argument("--nba", type=int, default=40)
    parser.add_argument("--temp", type=int, default=120)
    args = parser.parse_args()
    start, end = date.fromisoformat(args.start), date.fromisoformat(args.end)
    days = (end - start).days
    summary = []
    with httpx.Client(timeout=60) as client:
        for family in args.families.split(","):
            groups = []
            if family == "soccer":
                for s in SOCCER:
                    groups += [ms for ms in settled(client, s, start, end).values() if len(ms) == 3]
                groups.sort(key=lambda ms: -sum(float(m.get("volume_fp") or 0) for m in ms))
                res = study(client, family, groups, 8, args.soccer)
            elif family == "nba":
                nba_start = max(start, end - timedelta(days=12))
                games = defaultdict(list)
                for s in ("KXNBAGAME", "KXNBASPREAD", "KXNBATOTAL"):
                    for event, ms in settled(client, s, nba_start, end).items():
                        games[event.split("-", 1)[1]] += ms
                groups = [ms for ms in games.values() if len({m["event_ticker"].split("-")[0] for m in ms}) == 3]
                groups.sort(key=lambda ms: -sum(float(m.get("volume_fp") or 0) for m in ms))
                days_nba = (end - nba_start).days
                res = study(client, family, groups, 6, args.nba)
                rep = dict(report(res, days_nba), population=len(groups), sampled_days=days_nba)
                save(OUT / f"{family}.json", res)
                summary.append(rep)
                print(rep, flush=True)
                save(OUT / "summary.json", summary)
                continue
            else:
                for s in TEMP:
                    groups += list(settled(client, s, start, end).values())
                groups.sort(key=lambda ms: (ms[0]["close_time"], ms[0]["event_ticker"]), reverse=True)
                res = study(client, family, groups, 24, args.temp)
            rep = dict(report(res, days), population=len(groups), sampled_days=days)
            save(OUT / f"{family}.json", res)
            summary.append(rep)
            print(rep, flush=True)
            save(OUT / "summary.json", summary)


if __name__ == "__main__":
    main()
