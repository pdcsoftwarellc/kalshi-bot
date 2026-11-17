"""Bounded, read-only two-feed capture for the first-break paper experiment."""
import argparse
import asyncio
from collections import Counter, defaultdict
from datetime import datetime, timedelta
import json
from pathlib import Path
import sys
import time
from urllib.parse import urlencode, urlsplit

import httpx
import websockets

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from integration.kalshi_client import KalshiClient
from research.api_cost import ROOT, SERIES, UTC, config, link_market, timestamp
from research.forward_odds import exchange, tennis

OUT = ROOT / "data/raw/first-break"


TOURS = {"atp": (SERIES, 265), "challenger": ("KXATPCHALLENGERMATCH", 281)}


def discover(client, tour="atp"):
    series, event_type = TOURS[tour]
    today = datetime.now(UTC).date()
    fixtures = tennis(client, "get_fixtures", date_start=str(today-timedelta(days=1)),
                      date_stop=str(today+timedelta(days=3)), timezone="UTC", event_type_key=event_type)
    pairs = defaultdict(list)
    cursor = ""
    while True:
        result = exchange(client, "/markets", status="open", series_ticker=series, limit=1000, cursor=cursor)
        for m in result["markets"]:
            pairs[m["event_ticker"]].append(m)
        cursor = result.get("cursor")
        if not cursor:
            break
    matches = {}
    duplicates = set()
    for event, pair in pairs.items():
        if len(pair) != 2:
            continue
        pair.sort(key=lambda m:m["ticker"])
        link = link_market(pair, fixtures)
        if not link:
            continue
        f, yes_id, _ = link
        if f.get("event_winner") or f.get("event_status") == "Finished" or not f.get("event_time"):
            continue
        m = pair[0]
        key = str(f["event_key"])
        if key in matches:
            duplicates.add(key)
        matches[key] = dict(ticker=m["ticker"], event=event, series=series, tour=tour,
            market_pair=[dict(ticker=v["ticker"],open_time=v["open_time"]) for v in pair],
            discovery_status=f.get("event_status"), discovery_live=f.get("event_live"),
            already_in_play=str(f.get("event_live"))=="1" or bool(f.get("pointbypoint")) or bool(f.get("scores") and f.get("event_status") in ("Set 1","Set 2","Set 3","Set 4","Set 5")),

            player1=f["event_first_player"], player2=f["event_second_player"],
            yes_player=1 if str(f["first_player_key"]) == yes_id else 2,
            open_ts=max(timestamp(x["open_time"]) for x in pair),
            fixture_start=f['event_date']+"T"+f['event_time']+"+00:00",
            exchange_start=m.get("occurrence_datetime"))
    for key in duplicates:
        matches.pop(key)
    return matches


def tennis_rows(payload):
    if isinstance(payload, dict):
        if "event_key" in payload:
            return [payload]
        payload = payload.get("result", payload)
        if isinstance(payload, dict):
            if "event_key" in payload:
                return [payload]
            payload = list(payload.values())
    return [r for r in payload if isinstance(r,dict) and "event_key" in r] if isinstance(payload,list) else []


async def record(matches, seconds, path, tour="atp"):
    series, event_type = TOURS[tour]
    counters = Counter()
    with path.open("x", buffering=1) as out:
        def emit(kind, data, stamp=None):
            epoch, mono = stamp or (time.time_ns(), time.monotonic_ns())
            out.write(json.dumps(dict(kind=kind, utc_ns=epoch, mono_ns=mono, data=data), separators=(",",":"))+"\n")
            counters[kind] += 1

        emit("start", dict(matches=matches, seconds=seconds, tour=tour, series=series))
        def emit_scores(payload, stamp, source):
            for row in tennis_rows(payload):
                key = str(row["event_key"])
                if key not in matches:
                    continue
                games = [{**{k:g.get(k) for k in ("number_game","player_served","serve_winner","score")},
                          "n_points":len(g.get("points") or []),
                          "last_point":(g.get("points") or [{}])[-1].get("score"),
                          "points":[{"number_point":p.get("number_point"),"score":p.get("score")}
                                    for p in g.get("points") or []]}
                         for g in row.get("pointbypoint") or [] if g.get("set_number") == "Set 1"]
                native = {name:row.get(name) for name in (
                    "event_key","event_type_type","event_status","event_live","event_winner",
                    "event_date","event_first_player","event_second_player","first_player_key","second_player_key",
                    "event_ground","tournament_name","event_qualification","event_serve","event_game_result",
                    "event_final_result","scores","pointbypoint")}
                emit("score", dict(fixture=key, native=native, status=row.get("event_status"), source=source,
                                   live=row.get("event_live"), winner=row.get("event_winner"),
                                   games=games, set_scores=row.get("scores"),
                                   game_score=row.get("event_game_result")), stamp)

        async def scores():
            async with websockets.connect(config.API_TENNIS_WS+"?"+urlencode(
                    {"APIkey":config.API_TENNIS_KEY,"timezone":"UTC"}), max_size=32*1024*1024) as ws:
                emit("tennis_connected", {})
                async for raw in ws:
                    stamp = time.time_ns(), time.monotonic_ns()
                    counters["tennis_messages"] += 1
                    emit_scores(json.loads(raw), stamp, "websocket")

        async def poll_scores():
            # A second receipt channel supplies a baseline even when the
            # provider sends no WebSocket update for a particular fixture.
            async with httpx.AsyncClient(timeout=8) as client:
                while True:
                    try:
                        response = await client.get(config.API_TENNIS_REST, params=dict(
                            method="get_livescore",timezone="UTC",event_type_key=event_type,APIkey=config.API_TENNIS_KEY))
                        stamp = time.time_ns(), time.monotonic_ns()
                        response.raise_for_status()
                        payload = response.json()
                        if str(payload.get("success")) != "1":
                            raise ValueError("Livescore request rejected")
                        emit_scores(payload, stamp, "rest")
                    except (httpx.HTTPError,ValueError):
                        emit("tennis_poll_failed", {})
                    await asyncio.sleep(4)

        async def books():
            client = KalshiClient(config.KALSHI_API_KEY, config.KALSHI_PRIVATE_KEY_PATH, demo=False)
            headers = client._auth_headers("GET", urlsplit(config.KALSHI_WS_PROD).path)
            async with websockets.connect(config.KALSHI_WS_PROD, additional_headers=headers) as ws:
                await ws.send(json.dumps(dict(id=1, cmd="subscribe", params=dict(
                    channels=["orderbook_delta"], market_tickers=[m["ticker"] for m in matches.values()],
                    use_yes_price=False))))
                async for raw in ws:
                    stamp = time.time_ns(), time.monotonic_ns()
                    frame = json.loads(raw)
                    kind = frame["type"]
                    if kind == "error":
                        raise RuntimeError("Exchange rejected orderbook subscription")
                    if kind in ("orderbook_snapshot","orderbook_delta"):
                        keys = ("market_ticker","side","price_dollars","delta_fp","yes_dollars_fp",
                                "no_dollars_fp","clock","ts_ms")
                        emit(kind, dict(seq=frame["seq"],sid=frame["sid"],
                                        **{k:v for k,v in frame["msg"].items() if k in keys}), stamp)
                    elif kind == "subscribed":
                        emit("kalshi_subscribed", {"sid":frame["msg"]["sid"]}, stamp)

        tasks = [asyncio.create_task(coro()) for coro in (scores,books,poll_scores)]
        try:
            done, _ = await asyncio.wait(tasks, timeout=seconds, return_when=asyncio.FIRST_COMPLETED)
            for task in done:
                task.result()
                raise RuntimeError("Feed closed early")
        finally:
            for task in tasks:
                task.cancel()
            await asyncio.gather(*tasks, return_exceptions=True)
            emit("end", dict(counters))
    return counters


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seconds", type=int, default=120)
    parser.add_argument("--tour", choices=tuple(TOURS), default="atp")
    args = parser.parse_args()
    if args.seconds < 1:
        parser.error("Duration must be positive")
    with httpx.Client(timeout=20) as client:
        matches = discover(client, args.tour)
    if not matches:
        raise SystemExit("No uniquely matched open ATP events.")
    directory = OUT if args.tour == "atp" else OUT.with_name("first-break-challenger")
    directory.mkdir(exist_ok=True, parents=True)
    path = directory/f"{time.time_ns()}.jsonl"
    print(f"Recording {len(matches)} fixtures to {path}", flush=True)
    try:
        print(dict(asyncio.run(record(matches, args.seconds, path,args.tour))))
    except Exception as exc:
        raise SystemExit(f"Capture stopped ({type(exc).__name__}). Inspect the partial tape.") from None


if __name__ == "__main__":
    main()
