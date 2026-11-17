"""Record and replay a prospective Challenger cohort across entire matches."""
import argparse
import asyncio
from collections import Counter, defaultdict
from contextlib import contextmanager
from datetime import datetime, timezone
import fcntl
import json
import math
import os
from pathlib import Path
import signal
import sys
import time

import httpx

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

SECOND = 10**9
TERMINAL = {"finished", "retired", "cancelled", "canceled", "walk over", "walkover",
            "awarded", "abandoned", "defaulted"}
PREMATCH = {"", "scheduled", "not started"}
LEAD_SECONDS = 20 * 60
MISSING_AFTER = 48 * 3600


def start_time(meta):
    return datetime.fromisoformat(meta["fixture_start"].replace("Z", "+00:00")).timestamp()


def entrants(candidates, enrolled, now, target):
    """Select upcoming fixtures chronologically and independently of their odds."""
    events = {m["event"] for m in enrolled.values()}
    selected = {}
    for key, meta in sorted(candidates.items(), key=lambda item: (start_time(item[1]), item[0])):
        if len(enrolled) + len(selected) >= target:
            break
        if not meta.get("first_player_key") or not meta.get("second_player_key") or meta["first_player_key"] == meta["second_player_key"]:
            continue
        if key in enrolled or meta["event"] in events or meta["already_in_play"]:
            continue
        if str(meta.get("discovery_status") or "").strip().lower() not in PREMATCH:
            continue
        if 0 < start_time(meta) - now <= LEAD_SECONDS:
            selected[key] = meta
            events.add(meta["event"])
    return selected


def sessions(directory):
    return sorted(directory.glob("session-*.jsonl"))


def frames(path, byte_limit=None):
    with path.open("rb") as stream:
        for line in stream:
            # A crash can tear only the final append. Each resume opens a new file.
            if byte_limit is not None and stream.tell() > byte_limit:
                break
            if not line.endswith(b"\n"):
                break
            yield json.loads(line)


def roster(paths):
    enrolled, ended = {}, {}
    for path in paths:
        for r in frames(path):
            if r["kind"] == "enroll":
                enrolled[r["data"]["fixture"]] = r["data"]["meta"]
            elif r["kind"] == "terminal":
                ended[r["data"]["fixture"]] = r["data"]["status"]
    return enrolled, ended


@contextmanager
def single_writer(directory):
    directory.mkdir(parents=True, exist_ok=True)
    with (directory / ".writer.lock").open("a") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise ValueError("This trial already has a recorder") from None
        yield


class Journal:
    """Append-only source of truth. Lifecycle transitions survive a process restart."""
    def __init__(self, stream):
        self.stream = stream
        self.seq = 0
        self.last = None
        self.busy = False
        self.last_pause_seq = 0

    def write(self, kind, data):
        wall, mono = time.time_ns(), time.monotonic_ns()
        if self.last:
            wall_delta, mono_delta = wall-self.last[0], mono-self.last[1]
            if mono_delta < 0 or abs(wall_delta-mono_delta) > 15*SECOND or (self.busy and mono_delta > 15*SECOND):
                self.last_pause_seq = self._append(
                    "gap", dict(channel="all", reason="clock_or_capture_pause", after=self.seq),
                    wall, mono,
                )
        return self._append(kind, data, wall, mono)

    def _append(self, kind, data, wall, mono):
        self.seq += 1
        self.stream.write(json.dumps(dict(seq=self.seq, kind=kind, data=data, utc_ns=wall,
                                          mono_ns=mono), separators=(",", ":"))+"\n")
        self.last = (wall, mono)
        if kind in {"enroll", "terminal", "session_open", "session_stop", "gap"}:
            self.stream.flush()
            os.fsync(self.stream.fileno())
        return self.seq


def protocol_for(directory, source, target):
    from research.api_cost import save
    from research.live_odds_revision import POINT_RULE

    dest = directory / "protocol.json"
    supplied = json.loads(source.read_text()) if source else None
    existing = json.loads(dest.read_text()) if dest.exists() else None
    if existing:
        if (existing["target"] != target or existing["rule"] != POINT_RULE
                or supplied is not None and existing["registration"] != supplied):
            raise ValueError("Trial target, registration or strategy changed; use a new trial directory")
        return existing
    if sessions(directory):
        raise ValueError("Trial tapes exist without protocol.json")
    if supplied and (supplied.get("rule") != POINT_RULE or target != 30):
        raise ValueError("The supplied 30-match registration must match the point-confirmed rule")
    protocol = dict(target=target, tour="challenger", lead_seconds=LEAD_SECONDS,
                    rule=POINT_RULE, registration=supplied,
                    created=datetime.now(timezone.utc).isoformat())
    save(dest, protocol)
    return protocol


async def discover_status():
    from research.record_break import discover
    def read():
        statuses = {}
        with httpx.Client(timeout=20) as client:
            candidates = discover(client, "challenger", fixture_statuses=statuses)
        return candidates, statuses
    return await asyncio.to_thread(read)


class _SnapshotRequired(Exception):
    """The journal invalidated this book during a capture pause."""


async def fixture_book(key, ticker, journal, read_book):
    """Each reconnect receives a fresh snapshot and an independent sequence space."""
    from research.score_break import Books
    delay = 1
    while True:
        validator = Books()
        last_seq = None
        anchor = journal.seq
        sid = f"{key}:{anchor}"
        def accept(kind, data):
            nonlocal last_seq, anchor, delay
            if (data["market_ticker"] != ticker or last_seq is None and kind != "orderbook_snapshot"
                    or last_seq is not None and data["seq"] != last_seq + 1):
                raise ValueError("Invalid book sequence")
            validator.apply(kind, data, time.monotonic_ns())
            if ticker not in validator.levels:
                raise ValueError("Delta implies negative depth")
            last_seq = data["seq"]
            previous_anchor = anchor
            anchor = journal.write(kind, dict(data, sid=sid, fixture=key))
            # A pause may have been detected by another socket, or by this
            # very write. Replay ignores deltas until it sees a fresh snapshot.
            if kind == "orderbook_delta" and journal.last_pause_seq > previous_anchor:
                raise _SnapshotRequired
            delay = 1
        try:
            await read_book(accept, [ticker])
            reason = "socket_closed"
        except _SnapshotRequired:
            # The global gap is already durable. Resubscribe for a snapshot.
            continue
        except (OSError, TimeoutError, ValueError, RuntimeError) as exc:
            reason = type(exc).__name__
        except Exception as exc:
            if getattr(getattr(exc, "response", None), "status_code", None) in (401, 403):
                raise RuntimeError("Kalshi refused WebSocket authentication") from None
            from websockets.exceptions import WebSocketException
            if not isinstance(exc, WebSocketException):
                raise
            reason = type(exc).__name__
        journal.write("gap", dict(fixture=key, channel="book", reason=reason, after=anchor))
        await asyncio.sleep(delay)
        delay = min(30, 2*delay)


async def monitor(journal, enrolled, ended, target, stop, *, discover_fn=None, odds_fn=None,
                  book_fn=None, refresh=60):
    from research.live_odds_revision import odds_loop, book_stream
    discover_fn = discover_fn or discover_status
    odds_fn = odds_fn or odds_loop
    book_fn = book_fn or book_stream
    active = {k:m for k,m in enrolled.items() if k not in ended}
    workers = {}
    last_odds = journal.seq
    last_poll_time = None
    interrupted_odds = False

    def finish(key, status):
        if key in active:
            journal.write("terminal", dict(fixture=key, status=status))
            ended[key] = status
            del active[key]
            journal.busy = bool(active)
            workers[key].cancel()

    def scores(received):
        for key, row in received.items():
            if key not in active:
                continue
            meta = active[key]
            if any(str(row.get(k) or "") != meta[k] for k in ("first_player_key","second_player_key")):
                continue
            status = str(row.get("event_status") or "").strip().lower()
            if status in TERMINAL:
                finish(key, status)
            elif str(row.get("event_live")) == "1":
                journal.write("live", dict(fixture=key, status=status, scores=row.get("scores"),
                                           point=row.get("event_game_result")))

    def poll_event(kind, data):
        nonlocal last_odds, last_poll_time, interrupted_odds
        now = time.monotonic()
        late = last_poll_time is not None and now-last_poll_time > 10
        if (kind=="odds_error" or late) and not interrupted_odds:
            journal.write("gap", dict(channel="odds",reason="poll_error" if kind=="odds_error" else "poll_late",
                                      after=last_odds))
            interrupted_odds = True
        if kind=="odds":
            # Bind Home/Away to the identities captured at enrollment.
            data = {k:v for k,v in data.items() if k in active and all(
                str(v.get(field) or "") == active[k][field] for field in ("first_player_key","second_player_key"))}
        seq = journal.write(kind, data)
        if kind=="odds":
            last_odds, last_poll_time, interrupted_odds = seq, now, False

    def follow(key, meta):
        journal.busy = True
        workers[key] = asyncio.create_task(fixture_book(key,meta["ticker"],journal,book_fn))

    for key, meta in active.items():
        follow(key, meta)
    polling = asyncio.create_task(odds_fn(poll_event,active,"challenger",True,scores))
    stopping = asyncio.create_task(stop.wait())
    try:
        next_discovery = 0.
        while not stop.is_set() and len(ended) < target:
            if time.monotonic() >= next_discovery:
                try:
                    candidates, statuses = await discover_fn()
                except (httpx.HTTPError, RuntimeError, ValueError, OSError) as exc:
                    journal.write("discovery_error", dict(error_type=type(exc).__name__))
                else:
                    if stop.is_set():
                        break
                    now = time.time()
                    journal.write("discovery", dict(linked=len(candidates)))
                    for key,m in entrants(candidates,enrolled,now,target).items():
                        journal.write("enroll", dict(fixture=key,meta=m))
                        enrolled[key] = active[key] = m
                        follow(key,m)
                    for key in list(active):
                        status = str(statuses.get(key) or "").strip().lower()
                        if status in TERMINAL:
                            finish(key,status)
                        elif now > start_time(active[key])+MISSING_AFTER:
                            finish(key,"terminal_unconfirmed")
                next_discovery = time.monotonic()+refresh
            if len(ended) >= target:
                break
            watched = {stopping, polling, *(workers[k] for k in active)}
            done, _ = await asyncio.wait(watched, timeout=max(0,next_discovery-time.monotonic()),
                                         return_when=asyncio.FIRST_COMPLETED)
            for task in done - {stopping}:
                if task.cancelled() and task not in (workers.get(k) for k in active):
                    continue
                task.result()
                raise RuntimeError("Recorder worker stopped unexpectedly")
    finally:
        tasks = [polling,stopping,*workers.values()]
        for task in tasks:
            task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)


def trial_frames(paths):
    """Stream a unified replay. Invalidate a broken interval at its last trusted frame."""
    last = None
    limits = {path:path.stat().st_size for path in paths}
    for path in paths:
        # Only gap markers are buffered, not the potentially multi-hour book data.
        markers = defaultdict(list)
        for r in frames(path, limits[path]):
            if r["kind"]=="gap":
                markers[r["data"]["after"]].append(r["data"])
        offset = None
        for r in frames(path, limits[path]):
            if offset is None:
                offset = 0 if last is None else last["mono_ns"]+1-r["mono_ns"]
                if last is None:
                    yield dict(r, kind="start", data=dict(matches={},point_mode=True))
                else:
                    yield dict(last,kind="invalidate",data=dict(channel="all",reason="recorder_restart"))
            row = dict(r, mono_ns=r["mono_ns"]+offset)
            if row["kind"] in ("orderbook_snapshot","orderbook_delta"):
                row["data"] = dict(row["data"],sid=f"{path.stem}:{row['data']['sid']}")
            yield row
            for marker in markers.get(r["seq"], []):
                yield dict(row,kind="invalidate",data=marker)
            last = row
    if last is not None:
        yield dict(last,kind="end",data={"normal":True})


def summarize(directory, filename="report.json"):
    from research.api_cost import save
    from research.live_odds_revision import replay, usable
    from research.score_break import Books

    paths = sessions(directory)
    protocol = json.loads((directory/"protocol.json").read_text())
    fixtures = {}
    active = set()
    books = Books()
    gaps, windows = Counter(), []
    def inspect():
        for r in trial_frames(paths):
            k, m = r["kind"], r["data"]
            if k=="session_open":
                windows.append(dict(opened_utc_ns=r["utc_ns"], last_receipt_utc_ns=r["utc_ns"], stopped_utc_ns=None))
            elif k=="session_stop":
                windows[-1].update(stopped_utc_ns=r["utc_ns"],reason=m["reason"])
            elif k=="enroll":
                key = m["fixture"]
                active.add(key)
                fixtures[key] = dict(ticker=m["meta"]["ticker"], event=m["meta"]["event"],
                    yes_player=m["meta"]["yes_player"], scheduled=m["meta"]["fixture_start"],
                    enrolled_utc_ns=r["utc_ns"], terminal=None, live_seen=False, opening_observed=False,
                    book_ready=False, gaps=0, polls=0, usable_odds=0, live_polls=0,
                    priced_live_polls=0, orders=0, fills=0, closed=0)
            elif k=="terminal":
                key = m["fixture"]
                fixtures[key].update(terminal=m["status"],terminal_utc_ns=r["utc_ns"])
                active.discard(key)
            elif k in ("orderbook_snapshot", "orderbook_delta"):
                books.apply(k, m, r["mono_ns"])
                if k == "orderbook_snapshot":
                    fixtures[m["fixture"]]["book_ready"] = True
            elif k=="live":
                f = fixtures[m["fixture"]]
                if not f["live_seen"]:
                    first_set = [s for s in (m.get("scores") or []) if str(s.get("score_set"))=="1"]
                    f["opening_observed"] = (f["book_ready"] and m["status"]=="set 1"
                        and len(first_set)==1 and all(str(first_set[0].get(s))=="0" for s in ("score_first","score_second"))
                        and str(m.get("point")).replace(" ","") in {"0-0","-"})
                f["live_seen"] = True
            elif k=="invalidate":
                reason = m["reason"]
                gaps[reason] += 1
                keys = active if not m.get("fixture") else active & {m["fixture"]}
                for key in keys:
                    fixtures[key]["gaps"] += 1
                    if m.get("channel") != "odds":
                        fixtures[key]["book_ready"] = False
                        books.levels.pop(fixtures[key]["ticker"], None)
            elif k=="discovery_error":
                gaps["discovery_error"] += 1
            elif k=="odds":
                for key in active:
                    f = fixtures[key]
                    f["polls"] += 1
                    verified = usable(m.get(key),f["yes_player"],True) is not None
                    f["usable_odds"] += int(verified)
                    if f["live_seen"]:
                        f["live_polls"] += 1
                        sized = verified and all(books.ask(f["ticker"],side,r["mono_ns"]) is not None
                                                 for side in ("yes","no"))
                        f["priced_live_polls"] += int(sized)
            if windows:
                windows[-1]["last_receipt_utc_ns"] = r["utc_ns"]
            yield r

    if not any(p.stat().st_size for p in paths):
        raise ValueError("No capture sessions yet")
    paper = replay(inspect())
    for revision in paper["revisions"]:
        f = fixtures[revision["fixture"]]
        f["revisions"] = f.get("revisions",0) + 1
    for o in paper["ledger"]:
        f = fixtures[o["fixture"]]
        f["orders"] += 1
        f["fills"] += int("buy" in o)
        f["closed"] += int(o["state"]=="closed")
        if o["state"] in ("open_at_end","unpriced_exit","entry_unknown","incomplete_entry_window"):
            f["unresolved_attempts"] = f.get("unresolved_attempts",0) + 1
    for f in fixtures.values():
        if f["terminal"] is None:
            state = "pending" if not f["gaps"] else "interrupted_pending"
        elif f["terminal"]=="terminal_unconfirmed" or f["gaps"] or not f["opening_observed"]:
            state = "incomplete"
        elif not f["priced_live_polls"]:
            state = "no_verified_prices"
        elif f["priced_live_polls"] < f["live_polls"]:
            state = "partial_pricing"
        else:
            state = "covered"
        f["coverage"] = state
        f["continuous_capture"] = bool(f["opening_observed"] and f["terminal"] and
                                       f["terminal"] != "terminal_unconfirmed" and not f["gaps"])
        f["usable_odds_fraction_of_polls"] = f["usable_odds"]/f["polls"] if f["polls"] else None
    window_seconds = sum(max(0, w["last_receipt_utc_ns"]-w["opened_utc_ns"]) for w in windows)/SECOND
    from decimal import Decimal
    economics = dict(qualified=paper["qualified"], entered=paper["entered"], closed=paper["closed"],
                     unresolved=paper["unresolved"], unresolved_cost_dollars=paper["unresolved_cost"],
                     pending_entry_windows=sum(r["state"]=="incomplete_entry_window" for r in paper["ledger"]),
                     closed_pnl_dollars=paper["net_closed_dollars"], fees_on_closed_dollars=paper["fees_closed_dollars"],
                     dollars_short_of_one_month_cost=str(Decimal(80)-Decimal(paper["net_closed_dollars"])))
    report = dict(target=protocol["target"],enrolled=len(fixtures),terminal=sum(f["terminal"] is not None for f in fixtures.values()),
                  monitored_window_seconds=window_seconds, economics=economics, coverage=dict(Counter(
        f["coverage"] for f in fixtures.values())),gaps=dict(gaps),windows=windows,fixtures=fixtures,
        paper=paper, monthly_api_cost_dollars=80)
    save(directory/filename,report)
    return report


def configure(path):
    from dotenv import dotenv_values
    if path:
        if not path.is_file():
            raise ValueError("Credentials file does not exist")
        values = dotenv_values(path)
        for key in ("API_TENNIS_KEY","KALSHI_API_KEY","KALSHI_PRIVATE_KEY_PATH"):
            if not os.getenv(key) and values.get(key):
                value = values[key]
                if key.endswith("_PATH"):
                    value = str((path.resolve().parent / Path(value).expanduser()).resolve())
                os.environ[key] = value
    import config
    if not (config.API_TENNIS_KEY and config.KALSHI_API_KEY and config.KALSHI_PRIVATE_KEY_PATH):
        raise ValueError("Set API-Tennis and production Kalshi credentials or supply --env-file")
    from integration.kalshi_client import KalshiClient
    KalshiClient(config.KALSHI_API_KEY,config.KALSHI_PRIVATE_KEY_PATH,demo=False)


async def run(args):
    with single_writer(args.directory):
        protocol_for(args.directory,args.protocol,args.target)
        paths = sessions(args.directory)
        enrolled, ended = roster(paths)
        if len(ended) < args.target:
            path = args.directory/f"session-{len(paths)+1:06d}.jsonl"
            stop = asyncio.Event()
            reason = "target_reached"
            def request_stop(why):
                nonlocal reason
                reason = why
                stop.set()
            loop = asyncio.get_running_loop()
            for sig in (signal.SIGINT, signal.SIGTERM):
                loop.add_signal_handler(sig,request_stop,sig.name)
            timer = loop.call_later(args.window_minutes*60,request_stop,"window_limit") if args.window_minutes else None
            try:
                with path.open("x", buffering=1) as stream:
                    journal = Journal(stream)
                    journal.write("session_open",dict(previous_pending=len(enrolled)-len(ended)))
                    try:
                        await monitor(journal,enrolled,ended,args.target,stop)
                    except BaseException:
                        reason = "interrupted"
                        raise
                    finally:
                        journal.write("session_stop",dict(enrolled=len(enrolled),terminal=len(ended),reason=reason))
            finally:
                if timer:
                    timer.cancel()
                for sig in (signal.SIGINT,signal.SIGTERM):
                    loop.remove_signal_handler(sig)
        return summarize(args.directory)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command",choices=("record","report"))
    parser.add_argument("--directory",type=Path,required=True,help="Durable trial directory; reuse to resume")
    parser.add_argument("--protocol",type=Path,help="Previously frozen registration JSON")
    parser.add_argument("--target",type=int,default=30)
    parser.add_argument("--env-file",type=Path,help="Credentials file for record; key paths are relative to it")
    parser.add_argument("--window-minutes",type=float,help="End this monitored window early; fixture enrollment survives")
    args = parser.parse_args()
    try:
        if args.command=="record":
            if args.target<1 or args.window_minutes is not None and (
                    not math.isfinite(args.window_minutes) or args.window_minutes<=0):
                parser.error("Target and optional window duration must be positive")
            configure(args.env_file)
            report = asyncio.run(run(args))
        else:
            report = summarize(args.directory, "status.json")
    except (OSError,ValueError,RuntimeError) as exc:
        parser.exit(1, f"Trial failed: {exc}\n")
    print(json.dumps({k:report[k] for k in ("target","enrolled","terminal","coverage","gaps","monitored_window_seconds","economics")},indent=2))


if __name__=="__main__":
    main()
