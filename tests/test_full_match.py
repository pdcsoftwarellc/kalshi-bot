import asyncio
import json
from pathlib import Path
import time

import pytest

from research.full_match import (Journal, entrants, fixture_book, frames, monitor, protocol_for,
                                 roster, single_writer, summarize, trial_frames)
from research.live_odds_revision import SECOND, replay
from tests.test_live_odds_revision import crosscheck, sample


def meta(key, delta=60, **changes):
    from datetime import datetime, timezone
    return dict(dict(event="E"+key, ticker="T"+key, first_player_key="1", second_player_key="2",
                     fixture_start=datetime.fromtimestamp(time.time()+delta,timezone.utc).isoformat(),
                     already_in_play=False, discovery_status="", yes_player=1), **changes)


def test_prospective_enrollment_is_ordered_deduplicated_and_limited():
    candidates = dict(old=meta("old",-1), live=meta("live",already_in_play=True),
                      unknown=meta("unknown",discovery_status="1"), future=meta("far",1500),
                      later=meta("later",200), first=meta("first",100), duplicate=meta("first",101),
                      invalid=meta("no-keys",first_player_key=""))
    assert list(entrants(candidates,{},time.time(),2)) == ["first","later"]
    assert list(entrants(candidates,{"prior":candidates["first"]},time.time(),2)) == ["later"]


def test_single_writer_and_frozen_settings(tmp_path):
    with single_writer(tmp_path):
        with pytest.raises(ValueError,match="already has"):
            with single_writer(tmp_path):
                pass
    original = protocol_for(tmp_path,None,2)
    assert protocol_for(tmp_path,None,2) == original
    with pytest.raises(ValueError,match="changed"):
        protocol_for(tmp_path,None,30)


@pytest.mark.asyncio
async def test_monitor_follows_to_terminal_even_if_absent_from_open_market_list(tmp_path):
    protocol_for(tmp_path,None,1)
    observed = asyncio.Event()
    ready = asyncio.Event()
    enrolled, ended = {}, {}
    started = meta("A")
    async def discover():
        if not enrolled:
            return {"A":started}, {"A":""}
        await asyncio.wait_for(observed.wait(),2)
        return {}, {"A":"Finished"}

    async def books(emit,tickers):
        emit("orderbook_snapshot",dict(market_ticker=tickers[0],sid=1,seq=1))
        ready.set()
        await asyncio.Future()

    async def odds(emit,active,tour,point_mode,scores):
        await ready.wait()
        scores({"A":{"event_status":"Set 1","event_live":"1","event_game_result":"0 - 0",
                     "scores":[{"score_set":"1","score_first":"0","score_second":"0"}],
                     "first_player_key":"1","second_player_key":"2"}})
        emit("odds",{})
        observed.set()
        await asyncio.Future()

    path = tmp_path/"session-000001.jsonl"
    with path.open("w",buffering=1) as stream:
        j=Journal(stream)
        j.write("session_open",{})
        await asyncio.wait_for(monitor(j,enrolled,ended,1,asyncio.Event(),discover_fn=discover,
                                       odds_fn=odds,book_fn=books,refresh=.001),3)
        j.write("session_stop",{"reason":"target_reached"})
    assert ended == {"A":"finished"}
    assert roster([path]) == (enrolled,ended)
    report = summarize(tmp_path)
    assert report["fixtures"]["A"]["orders"] == report["paper"]["entered"] == 0
    assert report["fixtures"]["A"]["coverage"]=="no_verified_prices"
    assert report["fixtures"]["A"]["continuous_capture"]
    assert report["fixtures"]["A"]["usable_odds"]==0


@pytest.mark.asyncio
async def test_book_reconnect_requires_snapshot_and_rejects_skipped_sequence(tmp_path):
    path = tmp_path/"raw.jsonl"
    connected = asyncio.Event()
    tries = []
    async def book(emit,tickers):
        tries.append(1)
        emit("orderbook_snapshot",dict(market_ticker="T",sid=1,seq=1))
        if len(tries)==1:
            emit("orderbook_delta",dict(market_ticker="T",sid=1,seq=3,side="yes",
                                        delta_fp="2",price_dollars=".45"))
        connected.set()
        await asyncio.Future()
    with path.open("w",buffering=1) as stream:
        task=asyncio.create_task(fixture_book("A","T",Journal(stream),book))
        try:
            await asyncio.wait_for(connected.wait(),3)
        finally:
            task.cancel()
            await asyncio.gather(task,return_exceptions=True)
    rows=list(frames(path))
    assert [r["kind"] for r in rows]==["orderbook_snapshot","gap","orderbook_snapshot"]
    assert rows[1]["data"]["after"]==rows[0]["seq"]
    assert rows[0]["data"]["sid"]!=rows[2]["data"]["sid"]


def write_session(path, rows, *, torn=False):
    with path.open("w") as f:
        for seq,(seconds,kind,data) in enumerate(rows,1):
            f.write(json.dumps(dict(seq=seq,mono_ns=int(seconds*SECOND),utc_ns=int(seconds*SECOND)+200*SECOND,
                                    kind=kind,data=data))+"\n")
        if torn:
            f.write('{"kind":')


def book(seq,bid=".47",opposite=".48"):
    return dict(fixture="A",market_ticker="TA",sid="A:2",seq=seq,
                yes_dollars_fp=[[bid,"3"]],no_dollars_fp=[[opposite,"3"]])


def paper_window():
    return [(0,"session_open",{}),(.1,"enroll",dict(fixture="A",meta=meta("A"))),
            (1,"orderbook_snapshot",book(1)),(1.1,"odds",{"A":crosscheck(sample())}),
            (5.9,"orderbook_snapshot",book(2)),
            (6,"odds",{"A":crosscheck(sample("1.5","3",tag="00:02",score="30 - 0"))}),
            (7.8,"orderbook_snapshot",book(3,opposite=".49"))]


def test_late_gap_detection_cannot_simulate_a_fill_inside_the_missing_interval(tmp_path):
    tape=tmp_path/"session-000001.jsonl"
    rows=paper_window()+[(9,"odds",{}),
        (10,"gap",dict(fixture="A",reason="seq_gap",channel="book",after=7)),
        (11,"orderbook_snapshot",book(4)),(15,"session_stop",{"reason":"window_limit"})]
    write_session(tape,rows)
    paper=replay(trial_frames([tape]))
    assert paper["entered"]==0 and paper["unresolved"]==1
    assert paper["ledger"][0]["state"]=="entry_unknown"


def test_crash_resume_carries_inventory_and_missing_coverage_without_double_count(tmp_path):
    protocol_for(tmp_path,None,1)
    first,second=[tmp_path/f"session-{i:06d}.jsonl" for i in (1,2)]
    write_session(first,paper_window()+[(9,"discovery",{})],torn=True)
    write_session(second,[(0,"session_open",{}),(.1,"orderbook_snapshot",book(1)),
                          (3600,"terminal",dict(fixture="A",status="finished")),
                          (3601,"session_stop",dict(reason="target_reached"))])
    report=summarize(tmp_path)
    assert report["enrolled"]==1 and report["coverage"]=={"incomplete":1}
    assert report["paper"]["entered"]==report["paper"]["unresolved"]==1
    assert report["paper"]["closed"]==0 and report["paper"]["net_closed_dollars"]=="0"
    assert report["paper"]["ledger"][0]["state"]=="unpriced_exit"
    assert report["paper"]["ledger"][0]["entry_size"]=="3"
    assert report["windows"][0]["stopped_utc_ns"] is None
    assert summarize(tmp_path)==report


def test_valid_capture_exceeding_ten_minutes_keeps_one_order_and_exit(tmp_path):
    path=tmp_path/"session-000001.jsonl"
    rows=paper_window()+[(37.8,"orderbook_snapshot",book(4,bid=".62",opposite=".37")),
        (39,"odds",{}),(8000,"terminal",dict(fixture="A",status="finished")),
        (8001,"session_stop",dict(reason="target_reached"))]
    write_session(path,rows)
    paper=replay(trial_frames([path]))
    assert paper["entered"]==paper["closed"]==1
    assert float(paper["net_closed_dollars"])==pytest.approx(.06)
    assert paper["ledger"][0]["exit_size"]=="3"


@pytest.mark.asyncio
@pytest.mark.parametrize("pause_on_other_stream", [False, True])
async def test_connected_socket_resubscribes_after_global_capture_pause(tmp_path, pause_on_other_stream):
    path = tmp_path / "session-000001.jsonl"
    recovered = asyncio.Event()
    connections = 0
    with path.open("w", buffering=1) as stream:
        journal = Journal(stream)
        journal.busy = True
        journal.write("session_open", {})

        async def book_stream(emit, tickers):
            nonlocal connections
            connections += 1
            emit("orderbook_snapshot", book(1, opposite=".49" if connections == 2 else ".48"))
            if connections == 1:
                journal.last = tuple(t - 20 * SECOND for t in journal.last)
                if pause_on_other_stream:
                    journal.write("discovery", {})
                emit("orderbook_delta", dict(market_ticker="TA", sid=1, seq=2, side="no",
                                             price_dollars=".48", delta_fp="1"))
            else:
                recovered.set()
            await asyncio.Future()

        task = asyncio.create_task(fixture_book("A", "TA", journal, book_stream))
        try:
            await asyncio.wait_for(recovered.wait(), 2)
        finally:
            task.cancel()
            await asyncio.gather(task, return_exceptions=True)

    from research.score_break import Books
    books = Books()
    gaps = 0
    for frame in trial_frames([path]):
        kind, data = frame["kind"], frame["data"]
        if kind == "invalidate":
            gaps += 1
            books.levels.clear()
        elif kind in ("orderbook_snapshot", "orderbook_delta"):
            books.apply(kind, data, frame["mono_ns"])
            if kind == "orderbook_delta":
                assert "TA" not in books.levels
    assert gaps == 1 and connections == 2
    assert float(books.ask("TA", "yes", books.updated["TA"])) == pytest.approx(.51)
