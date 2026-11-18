"""Log prematch bookmaker consensus beside the Kalshi quote, then score the gaps.

Each pass makes three reads per tour (fixtures, bulk odds, open markets) and
appends one compact row per matched, not-yet-started match. The report asks
two questions of the log: how often does a sized Kalshi ask sit below the
bookmaker fair price by more than the taker fee, and does Kalshi's midpoint
then move toward the bookmakers before the match starts. Read-only; no orders.
"""
import argparse
from collections import Counter, defaultdict
from datetime import datetime, timedelta
from decimal import Decimal as D
import json
from pathlib import Path
from statistics import mean, median
import sys
import time

import httpx

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from research.api_cost import ROOT, UTC, link_market, save, timestamp
from research.forward_odds import consensus, exchange, tennis

LOG = ROOT / "data/raw/consensus-log"
OUT = ROOT / "data/processed/consensus-log"
TOURS = {"atp": ("KXATPMATCH", 265), "challenger": ("KXATPCHALLENGERMATCH", 281), "wta": ("KXWTAMATCH", 266)}
SHARP = "Pncl"
THRESHOLDS = (0, 1, 2, 3, 5)      # cents of edge after the entry fee


def fair(home, away):
    a, b = float(home), float(away)
    return b/(a+b) if min(a, b) > 1 else None


def snapshot(client, tour):
    series, event_type = TOURS[tour]
    today = datetime.now(UTC).date()
    window = dict(date_start=str(today), date_stop=str(today+timedelta(days=3)), event_type_key=event_type)
    fixtures = tennis(client, "get_fixtures", timezone="UTC", **window) or []
    odds = tennis(client, "get_odds", **window) or {}
    odds_at = time.time()
    pairs, cursor = defaultdict(list), ""
    while True:
        page = exchange(client, "/markets", status="open", series_ticker=series, limit=1000, cursor=cursor)
        for m in page["markets"]:
            pairs[m["event_ticker"]].append(m)
        cursor = page.get("cursor")
        if not cursor:
            break
    book_at = time.time()
    rows, skip = [], Counter()
    for event, pair in pairs.items():
        pair.sort(key=lambda m: m["ticker"])
        link = link_market(pair, fixtures) if len(pair) == 2 else None
        if not link:
            skip["no_unique_fixture"] += 1
            continue
        f, yes_id, _ = link
        m = pair[0]
        if not f.get("event_time"):
            skip["no_start_time"] += 1
            continue
        start = timestamp(f["event_date"]+"T"+f["event_time"]+"+00:00")
        if (str(f.get("event_live")) == "1" or f.get("event_winner") or start <= book_at
                or f.get("event_status") not in (None, "", "Scheduled")):
            skip["started_or_finished"] += 1
            continue
        market = (odds.get(str(f["event_key"])) or {}).get("Home/Away") or {}
        home, away = market.get("Home") or {}, market.get("Away") or {}
        mid, books = consensus(home, away)
        if mid is None:
            skip["fewer_than_three_books"] += 1
            continue
        first_is_yes = str(f["first_player_key"]) == yes_id
        sharp = fair(home[SHARP], away[SHARP]) if SHARP in home and SHARP in away else None
        orient = lambda p: None if p is None else float(p) if first_is_yes else 1-float(p)
        rows.append(dict(ts=book_at, odds_ts=odds_at, tour=tour, ticker=m["ticker"], fixture=f["event_key"],
                         yes=m["yes_sub_title"], start_ts=start, opened_ts=timestamp(m["open_time"]),
                         books=len(books), fair_yes=orient(mid), sharp_yes=orient(sharp),
                         yes_bid=float(m["yes_bid_dollars"]), yes_ask=float(m["yes_ask_dollars"]),
                         bid_size=float(m["yes_bid_size_fp"]), ask_size=float(m["yes_ask_size_fp"]),
                         volume=float(m["volume_fp"])))
    return rows, skip


def record(every, hours, tours):
    LOG.mkdir(parents=True, exist_ok=True)
    stop = time.time()+hours*3600
    with httpx.Client(timeout=30) as client:
        while True:
            began = time.time()
            for tour in tours:
                try:
                    rows, skip = snapshot(client, tour)
                except (RuntimeError, httpx.HTTPError, ValueError, KeyError) as exc:
                    print(f"{datetime.now(UTC):%H:%M:%S} {tour}: skipped pass ({type(exc).__name__})", flush=True)
                    continue
                path = LOG / f"{datetime.now(UTC):%Y-%m-%d}.jsonl"
                with path.open("a") as out:
                    out.writelines(json.dumps(r, separators=(",", ":"))+"\n" for r in rows)
                print(f"{datetime.now(UTC):%H:%M:%S} {tour}: {len(rows)} rows {dict(skip)}", flush=True)
            if time.time()+every > stop:
                return
            time.sleep(max(0, every-(time.time()-began)))


def edges(row, source):
    """Edge per contract after the entry fee for (YES, NO), or None without a sized two-sided quote."""
    p = row[source]
    bid, ask = row["yes_bid"], row["yes_ask"]
    if p is None or not 0 < bid < ask < 1 or min(row["bid_size"], row["ask_size"]) < 1:
        return None
    cost = lambda price: price+.07*price*(1-price)     # unrounded: the fee on a large order
    return p-cost(ask), (1-p)-cost(1-bid)


def report():
    series = defaultdict(list)
    for path in sorted(LOG.glob("*.jsonl")):
        for line in path.read_text().splitlines():
            row = json.loads(line)
            series[row["ticker"]].append(row)
    out = dict(matches=len(series), observations=sum(map(len, series.values())), sources={})
    if not series:
        raise SystemExit("No log yet. Run with --record first.")
    out["hours_logged"] = (max(r[-1]["ts"] for r in series.values())-min(r[0]["ts"] for r in series.values()))/3600
    out["odds_changes_per_match"] = mean(sum(a["fair_yes"] != b["fair_yes"] for a, b in zip(r, r[1:])) for r in series.values())
    signals = []
    for source in ("fair_yes", "sharp_yes"):
        best, table = [], {}
        hits = defaultdict(list)
        for ticker, rows in series.items():
            priced = [(r, e) for r in rows if (e := edges(r, source))]
            if not priced:
                continue
            best.append(100*max(max(e) for _, e in priced))
            for cents in THRESHOLDS:
                # First qualifying observation only: one decision per match.
                for i, (r, e) in enumerate(priced):
                    if max(e) >= cents/100:
                        side = 1 if e[0] >= e[1] else -1
                        last = priced[-1][0]
                        drift = side*100*((last["yes_bid"]+last["yes_ask"])/2-(r["yes_bid"]+r["yes_ask"])/2)
                        hit = dict(source=source, threshold=cents, ticker=ticker, tour=r["tour"], side="yes" if side > 0 else "no",
                                   edge_cents=100*max(e), price=r["yes_ask"] if side > 0 else 1-r["yes_bid"],
                                   hours_to_start=(r["start_ts"]-r["ts"])/3600, later_observations=len(priced)-1-i,
                                   kalshi_mid_drift_cents=drift, ts=r["ts"])
                        hits[cents].append(hit)
                        break
        for cents, rows in hits.items():
            followed = [h for h in rows if h["later_observations"]]
            table[f">={cents}c"] = dict(
                matches=len(rows), by_tour=dict(Counter(h["tour"] for h in rows)),
                mean_edge_cents=mean(h["edge_cents"] for h in rows),
                with_later_quote=len(followed),
                mean_drift_toward_us_cents=mean(h["kalshi_mid_drift_cents"] for h in followed) if followed else None,
                share_drifting_toward_us=mean(h["kalshi_mid_drift_cents"] > 0 for h in followed) if followed else None)
            signals += rows
        out["sources"][source] = dict(matches_priced=len(best), median_best_edge_cents=median(best) if best else None,
                                      thresholds=table)
    save(OUT/"summary.json", out)
    save(OUT/"signals.json", signals)
    print(json.dumps(out, indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--record", action="store_true", help="append snapshots instead of reporting")
    parser.add_argument("--every", type=int, default=300, help="seconds between passes")
    parser.add_argument("--hours", type=float, default=336)
    parser.add_argument("--tours", nargs="+", choices=tuple(TOURS), default=list(TOURS))
    args = parser.parse_args()
    record(args.every, args.hours, args.tours) if args.record else report()


if __name__ == "__main__":
    main()
