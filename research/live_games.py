"""Record Kalshi's own live game state beside the winner-market quote.

Kalshi publishes score, period, clock, inning and tennis set scores for its
sports events (milestones and live data). Past games keep only their final
state, so rules that need "how late is it and by how much" can only be tested
on a recording. Each pass reads the open markets of every series and one
batched live-data call, and appends a row whenever an event's state or quote
changes. Public reads only; no orders.
"""
import argparse
from collections import defaultdict
from datetime import datetime, timedelta
import json
from pathlib import Path
import sys
import time

import httpx

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import config
from research.api_cost import ROOT, UTC

OUT = ROOT / "data/raw/live-games"
SERIES = ("KXNBAGAME", "KXNHLGAME", "KXNFLGAME", "KXMLBGAME", "KXNCAAMBGAME", "KXNCAAFGAME",
          "KXATPMATCH", "KXATPCHALLENGERMATCH", "KXWTAMATCH")
DROP = {"pitchers", "away_last_games", "home_last_games", "last_play", "current_batter"}
QUOTE = ("yes_bid_dollars", "yes_ask_dollars", "yes_bid_size_fp", "yes_ask_size_fp", "volume_fp")


def get(client, path, **params):
    for attempt in range(10):
        r = client.get(config.KALSHI_PROD_BASE+path, params=params)
        if r.status_code == 429 or r.status_code >= 500:
            time.sleep(.5+attempt)
            continue
        r.raise_for_status()
        return r.json()
    raise RuntimeError(f"{path}: no response after retries")


def milestones(client, since):
    """Map event ticker -> (milestone id, type, scheduled start) for recent and upcoming games."""
    found, cursor = {}, ""
    while True:
        page = get(client, "/milestones", limit=200, category="Sports", minimum_start_date=since, cursor=cursor)
        for m in page.get("milestones", []):
            for event in m.get("primary_event_tickers") or []:
                if event.startswith(SERIES):
                    found[event] = (m["id"], m["type"], m.get("start_date"))
        cursor = page.get("cursor")
        if not cursor or not page.get("milestones"):
            return found


def quotes(client):
    """Canonical (alphabetically first) market per open event with its top of book."""
    events = defaultdict(list)
    for series in SERIES:
        cursor = ""
        while True:
            page = get(client, "/markets", status="open", series_ticker=series, limit=1000, cursor=cursor)
            for m in page["markets"]:
                events[m["event_ticker"]].append(m)
            cursor = page.get("cursor")
            if not cursor:
                break
    return {event: min(group, key=lambda m: m["ticker"]) for event, group in events.items()}


def record(every, hours):
    OUT.mkdir(parents=True, exist_ok=True)
    stop, known, refreshed, last = time.time()+hours*3600, {}, 0, {}
    with httpx.Client(timeout=30) as client:
        while time.time() < stop:
            began = time.time()
            try:
                if began-refreshed > 300:
                    since = (datetime.now(UTC)-timedelta(hours=8)).strftime("%Y-%m-%dT%H:%M:%SZ")
                    known, refreshed = milestones(client, since), began
                books = quotes(client)
                ids = {known[e][0]: e for e in books if e in known}
                live = {}
                for i in range(0, len(ids), 40):
                    chunk = list(ids)[i:i+40]
                    for row in get(client, "/live_data/batch", milestone_ids=chunk).get("live_datas", []):
                        live[ids[row["milestone_id"]]] = {k: v for k, v in row["details"].items() if k not in DROP}
                seen = time.time()
                wrote = 0
                with (OUT / f"{datetime.now(UTC):%Y-%m-%d}.jsonl").open("a") as out:
                    for event, state in live.items():
                        m = books[event]
                        quote = {k: m.get(k) for k in QUOTE}
                        # Skip games that have not started and rows identical to the last one written.
                        status = str(state.get("status", "")).lower()
                        idle = status in ("scheduled", "not_started", "created", "sch", "")
                        stable = {k: v for k, v in state.items() if not k.endswith("_ts") and "seq" not in k and "cursor" not in k}
                        key = json.dumps([stable, quote[QUOTE[0]], quote[QUOTE[1]]], sort_keys=True)
                        if idle or last.get(event) == key:
                            continue
                        last[event] = key
                        out.write(json.dumps(dict(ts=seen, event=event, ticker=m["ticker"], yes=m.get("yes_sub_title"),
                                                  type=known[event][1], start=known[event][2], expected_end=m.get("expected_expiration_time"),
                                                  quote=quote, state=state), separators=(",", ":"))+"\n")
                        wrote += 1
                print(f"{datetime.now(UTC):%H:%M:%S} open events {len(books)} with live data {len(live)} rows {wrote}", flush=True)
            except (httpx.HTTPError, RuntimeError, KeyError, ValueError) as exc:
                print(f"{datetime.now(UTC):%H:%M:%S} pass skipped ({type(exc).__name__})", flush=True)
            time.sleep(max(1, every-(time.time()-began)))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--every", type=int, default=15, help="seconds between passes")
    parser.add_argument("--hours", type=float, default=24)
    args = parser.parse_args()
    record(args.every, args.hours)


if __name__ == "__main__":
    main()
