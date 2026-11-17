from decimal import Decimal as D

from research.depth_pressure import SECOND, pressure, replay, totals


def frame(t, kind, data):
    return dict(mono_ns=int(t * SECOND), utc_ns=100 * SECOND + int(t * SECOND), kind=kind, data=data)


def snapshot(seq=1):
    return dict(seq=seq, sid=1, market_ticker="T", yes_dollars_fp=[[".50", "80"], [".46", "9999"]],
                no_dollars_fp=[[".49", "20"]])


def tape(*more):
    return [
        frame(0, "start", dict(matches={"m": dict(ticker="T")})),
        frame(0, "score", dict(fixture="m", live="1", winner=None, status="Set 2")),
        frame(0, "orderbook_snapshot", snapshot()),
        *more,
    ]


def delta(seq=2, side="yes", price=".50", size="-1"):
    return dict(seq=seq, sid=1, market_ticker="T", side=side, price_dollars=price, delta_fp=size)


def test_continuous_pressure_delay_and_take_both_fees():
    rows = tape(frame(2, "orderbook_delta", delta()), frame(4, "orderbook_delta", delta(seq=3, size="1")),
                frame(42, "end", {}))
    result = replay(rows)
    trade, = result["attempts"]
    assert trade["signal_utc_ns"] == 107 * SECOND
    assert trade["entry_utc_ns"] == 109 * SECOND
    assert trade["exit_utc_ns"] == 141 * SECOND
    assert D(trade["buy"]) == D(".51")
    assert D(trade["pnl"]) == D("-.05")


def test_no_retroactive_live_label_and_gap_breaks_streak():
    rows = tape(frame(2, "orderbook_delta", delta(seq=2, size="1")),
                frame(10, "orderbook_delta", delta(seq=4)), frame(11, "end", {}))
    assert replay(rows)["attempts"] == []
    rows.pop(1)
    assert replay(rows)["fixtures_with_observed_live_score"] == []


def test_deteriorated_entry_and_unpriced_exit():
    rows = tape(frame(5, "orderbook_delta", delta(side="no", price=".49", size="-20")), frame(50, "end", {}))
    trade, = replay(rows)["attempts"]
    assert trade["status"] == "entry_unfilled"
    rows = tape(frame(36, "orderbook_delta", delta(size="-80")), frame(36, "orderbook_delta", delta(seq=3,price=".46",size="-9999")), frame(40, "end", {}))
    trade, = replay(rows)["attempts"]
    assert trade["status"] == "unpriced_exit" and "pnl" not in trade
    assert totals([trade])["open_cost_dollars"] == "0.53"


def test_far_depth_excluded_and_partial_size_at_touch():
    assert pressure({"yes": {D(".50"): D(1), D(".47"):D(1000)}, "no": {D(".49"):D(5)}}) == "no"
    assert pressure({"yes": {D(".50"): D(".5")}, "no": {D(".49"):D(5)}}) is None
