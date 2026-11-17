"""Save one timestamped prematch bookmaker/Kalshi comparison per open ATP event."""
import argparse
from collections import Counter, defaultdict
from datetime import datetime, timedelta
from decimal import Decimal as D, InvalidOperation
import json
from pathlib import Path
import statistics
import sys
import time

import httpx

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from research.api_cost import ROOT, SERIES, UTC, config, fee, link_market, save, timestamp

OUT = ROOT / "data/processed/api-cost-research/forward"
MIN_BOOKS = 3
MIN_EDGE = D(".05")


def consensus(home, away):
    """Remove the two-way overround separately at each bookmaker."""
    prices = []
    for book in sorted(home.keys() & away.keys()):
        try:
            a, b = D(str(home[book])), D(str(away[book]))
        except InvalidOperation:
            continue
        if a.is_finite() and b.is_finite() and min(a, b) > 1:
            prices.append(dict(book=book, home=str(a), away=str(b), fair_home=str(b/(a+b))))
    if len(prices) < MIN_BOOKS:
        return None, prices
    return statistics.median(D(p["fair_home"]) for p in prices), prices


def book_offers(book):
    offers = {}
    for side, levels in (("yes", book["no_dollars"]), ("no", book["yes_dollars"])):
        if levels:
            bid, size = max(((D(p), D(q)) for p, q in levels), key=lambda level:level[0])
            if 0 < bid < 1 and size >= 1:
                offers[side] = 1 - bid
    return offers


def tennis(client, method, **params):
    # Raising HTTPStatusError would include the access key in the URL.
    try:
        r = client.get(config.API_TENNIS_REST, params=dict(params, method=method, APIkey=config.API_TENNIS_KEY))
        if r.status_code != 200:
            raise RuntimeError(f"{method} HTTP {r.status_code}")
        payload = r.json()
    except (httpx.RequestError, ValueError) as exc:
        raise RuntimeError(f"{method}: {type(exc).__name__}") from None
    if str(payload.get("success")) != "1":
        raise RuntimeError(f"{method} rejected by provider")
    return payload.get("result") or {}


def exchange(client, path, **params):
    r = client.get(config.KALSHI_PROD_BASE + path, params=params)
    r.raise_for_status()
    return r.json()


def capture(client):
    now = datetime.now(UTC)
    end = now.date() + timedelta(days=3)
    fixtures = tennis(client, "get_fixtures", date_start=str(now.date()), date_stop=str(end),
                      timezone="UTC", event_type_key=265)
    if not isinstance(fixtures, list) and fixtures:
        raise ValueError("Invalid fixtures")
    pairs = defaultdict(list)
    cursor = ""
    while True:
        data = exchange(client, "/markets", status="open", series_ticker=SERIES, limit=1000, cursor=cursor)
        for m in data["markets"]:
            pairs[m["event_ticker"]].append(m)
        cursor = data.get("cursor")
        if not cursor:
            break

    skip = Counter()
    rows = []
    for event, markets in sorted(pairs.items()):
        pair = sorted(markets, key=lambda m:m["ticker"])
        if len(pair) != 2:
            skip["incomplete_pair"] += 1
            continue
        link = link_market(pair, fixtures)
        if not link:
            skip["no_unique_fixture"] += 1
            continue
        f, yes_id, _ = link
        m = pair[0]
        if str(f.get("event_live")) == "1" or f.get("event_winner") or not f.get("event_time"):
            skip["live_or_finished_fixture"] += 1
            continue
        fixture_start = timestamp(f["event_date"] + "T" + f["event_time"] + "+00:00")
        if fixture_start <= time.time():
            skip["fixture_time_passed"] += 1
            continue
        occur = m.get("occurrence_datetime")
        exchange_start = timestamp(occur) if occur else None
        start = min(fixture_start, exchange_start) if occur else fixture_start
        # Record conflicting schedules for later audit. They cannot trigger
        # a simulated entry even when a fresh book and provider odds exist.
        eligible = bool(occur and f.get("event_status") in (None, "", "Scheduled")
                        and time.time()+1800 < start)
        ticker = m["ticker"]
        odds = tennis(client, "get_odds", match_key=f["event_key"])
        odds_at = time.time()
        market = odds.get(str(f["event_key"]), {}).get("Home/Away", {}) if isinstance(odds, dict) else {}
        home, books = consensus(market.get("Home", {}), market.get("Away", {}))
        if home is None:
            skip["fewer_than_three_paired_books"] += 1
            continue
        book = exchange(client, "/markets/"+ticker+"/orderbook", depth=1)["orderbook_fp"]
        seen = time.time()
        offers = book_offers(book)
        p = home if str(f["first_player_key"]) == yes_id else 1-home
        edges = {s:(p if s=="yes" else 1-p)-price-fee(price) for s,price in offers.items()}
        side = max(edges, key=edges.get) if edges else None
        candidate = bool(eligible and side and edges[side] >= MIN_EDGE and seen-odds_at <= 10 and seen+1800 < start)
        row = dict(ticker=ticker, fixture=f["event_key"], event=event, yes_player=m["yes_sub_title"],
                   first_player=f["event_first_player"], second_player=f["event_second_player"],
                   odds_received_ts=odds_at, book_received_ts=seen, planned_start_ts=start,
                   fixture_status=f.get("event_status"), fixture_start_ts=fixture_start,
                   exchange_start_ts=exchange_start, eligible_prematch=eligible,
                   books=books, fair_yes=str(p), orderbook=book, candidate=candidate,
                   side=side, modeled_net_edge=str(edges[side]) if side else None,
                   paper_fill=False)
        if candidate:
            limit = offers[side]
            time.sleep(2)
            second = exchange(client, "/markets/"+ticker+"/orderbook", depth=1)["orderbook_fp"]
            checked = time.time()
            px = book_offers(second).get(side)
            row.update(check_ts=checked, second_book=second, limit=str(limit),
                       paper_fill=px is not None and px <= limit and checked-seen <= 10 and checked < start)
        rows.append(row)
    return dict(started=now.isoformat(), ended=datetime.now(UTC).isoformat(), open_events=len(pairs),
                assumptions=dict(min_books=MIN_BOOKS, min_net_edge=str(MIN_EDGE), contracts=1,
                                 fee="quadratic taker multiplier 1, rounded to cent"),
                exclusions=dict(skip), observations=rows)


def main():
    argparse.ArgumentParser(description=__doc__).parse_args()
    with httpx.Client(timeout=25) as client:
        snapshot = capture(client)
    path = OUT / f"{time.time_ns()}.json"
    save(path, snapshot)
    rows = snapshot["observations"]
    print(json.dumps(dict(path=str(path), open_events=snapshot["open_events"], observations=len(rows),
                          eligible_prematch=sum(r["eligible_prematch"] for r in rows),
                          candidates=sum(r["candidate"] for r in rows), paper_fills=sum(r["paper_fill"] for r in rows),
                          exclusions=snapshot["exclusions"]), indent=2))


if __name__ == "__main__":
    main()
