import json

from research.fifteen_forty import fair_trade, fee
from research.fifteen_forty_tape import SECOND, trades


def _tape(path, points):
    """One market where player 2 is YES; player 1 serves at 2-2."""
    rows = [dict(kind="start", mono_ns=0, data=dict(matches={"7": dict(
        ticker="T", player1="A", player2="B", yes_player=2)}))]
    seq = 0
    for second, game, sets, yes_bid in points:
        seq += 1
        rows.append(dict(kind="orderbook_snapshot", mono_ns=second*SECOND-1, data=dict(
            sid=1, seq=seq, market_ticker="T", yes_dollars_fp=[[f"{yes_bid/100:.2f}", "10"]],
            no_dollars_fp=[[f"{(98-yes_bid)/100:.2f}", "10"]])))
        seq += 1                      # refresh the same book just after the simulated latency
        rows.append(dict(kind="orderbook_snapshot", mono_ns=second*SECOND+2*SECOND-1, data=dict(
            sid=1, seq=seq, market_ticker="T", yes_dollars_fp=[[f"{yes_bid/100:.2f}", "10"]],
            no_dollars_fp=[[f"{(98-yes_bid)/100:.2f}", "10"]])))
        rows.append(dict(kind="score", mono_ns=second*SECOND, data=dict(
            fixture="7", status="Set 1", winner=None, game_score=game,
            set_scores=[dict(score_first=str(sets[0]), score_second=str(sets[1]))],
            native=dict(event_serve="First Player" if sum(sets) % 2 == 0 else "Second Player"))))
    path.write_text("\n".join(json.dumps(r) for r in rows)+"\n")
    return path


def test_break_closes_at_bid_after_fees(tmp_path):
    tape = _tape(tmp_path/"t.jsonl", [(10, "15 - 30", (2, 2), 55), (30, "15 - 40", (2, 2), 60),
                                      (50, "0 - 0", (2, 3), 66)])
    (row,), gaps = trades(tape)
    assert gaps == 0 and row["status"] == "closed" and row["result"] == "break"
    assert (row["entry_ask"], row["exit_bid"]) == (62, 66)
    assert row["net_cents"] == 4-2-2


def test_deuce_exit_and_skipped_point(tmp_path):
    tape = _tape(tmp_path/"t.jsonl", [(10, "0 - 40", (2, 2), 60), (30, "15 - 40", (2, 2), 58),
                                      (50, "40 - 40", (2, 2), 50)])
    (row,), _ = trades(tape)
    assert row["result"] == "deuce" and row["gross_cents"] == 50-60
    tape = _tape(tmp_path/"u.jsonl", [(10, "15 - 15", (2, 2), 50), (30, "15 - 40", (2, 2), 60)])
    assert trades(tape)[0][0]["status"] == "not_a_clean_transition"


def test_fair_round_trip_only_loses_costs():
    t = fair_trade(.64, .64, (2, 2))
    assert abs(t["gross_at_fair_mid"]) < 1e-12
    assert t["net_per_contract"] < -.04
    assert fee(.5, 1) == .02 and fee(.5, 100) == .0175
