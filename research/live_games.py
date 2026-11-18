"""Record Kalshi's own live game state beside the winner-market quote.

Kalshi publishes score, period, clock, inning and tennis set scores for its
sports events (milestones and live data). Past games keep only their final
state, so rules that need "how late is it and by how much" can only be tested
on a recording. Each pass reads the open markets of every series and one
batched live-data call, and appends a row whenever an event's state or quote
changes. Public reads only; no orders.

Rows start 45 minutes before the scheduled start, so the pre-game price is on
record. Each row carries both markets of a two-sided event (`other`), the time
the quote was read (`quote_ts`) beside the time the state was read (`ts`). One
line per pass goes to data/raw/live-games-heartbeat/, so a quiet spell can be
told apart from an outage.
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
from research.api_cost import ROOT, UTC, timestamp

OUT = ROOT / "data/raw/live-games"
BEAT = ROOT / "data/raw/live-games-heartbeat"
SERIES = ("KXNBAGAME", "KXNHLGAME", "KXNFLGAME", "KXMLBGAME", "KXNCAAMBGAME", "KXNCAAFGAME",
          "KXATPMATCH", "KXATPCHALLENGERMATCH", "KXWTAMATCH", "KXWTACHALLENGERMATCH", "KXITFMATCH", "KXITFWMATCH")
EARLY, AHEAD = 45*60, 3600                    # write rows this long before the start; ask for state this long before
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
    """Every open event's markets, first-listed first, and the time each listing was read."""
    events, read = defaultdict(list), {}
    for series in SERIES:
        cursor = ""
        while True:
            page = get(client, "/markets", status="open", series_ticker=series, limit=1000, cursor=cursor)
            at = time.time()
            for m in page["markets"]:
                events[m["event_ticker"]].append(m)
                read[m["event_ticker"]] = at
            cursor = page.get("cursor")
            if not cursor:
                break
    return {event: sorted(group, key=lambda m: m["ticker"]) for event, group in events.items()}, read


def starts(known, event):
    try:
        return timestamp(known[event][2])
    except (KeyError, TypeError, ValueError, AttributeError):
        return None


def record(every, hours):
    OUT.mkdir(parents=True, exist_ok=True)
    BEAT.mkdir(parents=True, exist_ok=True)
    stop, known, refreshed, last = time.time()+hours*3600, {}, 0, {}
    with httpx.Client(timeout=30) as client:
        while time.time() < stop:
            began = time.time()
            try:
                if began-refreshed > 300:
                    since = (datetime.now(UTC)-timedelta(hours=8)).strftime("%Y-%m-%dT%H:%M:%SZ")
                    known, refreshed = milestones(client, since), began
                books, read = quotes(client)
                # State is only asked for events that have started or start within the hour.
                ids = {known[e][0]: e for e in books if e in known and (starts(known, e) or 0) <= began+AHEAD}
                live = {}
                for i in range(0, len(ids), 40):
                    chunk = list(ids)[i:i+40]
                    for row in get(client, "/live_data/batch", milestone_ids=chunk).get("live_datas", []):
                        live[ids[row["milestone_id"]]] = {k: v for k, v in row["details"].items() if k not in DROP}
                seen = time.time()
                wrote = 0
                with (OUT / f"{datetime.now(UTC):%Y-%m-%d}.jsonl").open("a") as out:
                    for event, state in live.items():
                        m, rest = books[event][0], books[event][1:]
                        quote = {k: m.get(k) for k in QUOTE}
                        other = dict(ticker=rest[0]["ticker"], yes=rest[0].get("yes_sub_title"),
                                     quote={k: rest[0].get(k) for k in QUOTE}) if len(rest) == 1 else None
                        # Skip games more than 45 minutes from their start and rows identical to the last one written.
                        status = str(state.get("status", "")).lower()
                        idle = status in ("scheduled", "not_started", "created", "sch", "")
                        start = starts(known, event)
                        far = idle and not (start and start-seen <= EARLY)
                        stable = {k: v for k, v in state.items() if not k.endswith("_ts") and "seq" not in k and "cursor" not in k}
                        key = json.dumps([stable, quote[QUOTE[0]], quote[QUOTE[1]],
                                          other and [other["quote"][QUOTE[0]], other["quote"][QUOTE[1]]]], sort_keys=True)
                        if far or last.get(event) == key:
                            continue
                        last[event] = key
                        out.write(json.dumps(dict(ts=seen, quote_ts=read.get(event), event=event, ticker=m["ticker"],
                                                  yes=m.get("yes_sub_title"), type=known[event][1], start=known[event][2],
                                                  expected_end=m.get("expected_expiration_time"), quote=quote, other=other,
                                                  state=state), separators=(",", ":"))+"\n")
                        wrote += 1
                (BEAT / f"{datetime.now(UTC):%Y-%m-%d}.jsonl").open("a").write(
                    json.dumps(dict(ts=seen, open_events=len(books), with_state=len(live), rows=wrote))+"\n")
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
