import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from integration.edge import evaluate_signal
from scripts.build_features import build_training_rows_from_matches
from scripts.replay_trades import replay
from scripts.train import build_match_win_probs_on_val
from tennis.elo import EloTracker
from tennis.fixture_history import _point_winners
from tennis.research import parse_pbp_games
from tennis.types import MatchFormat


@pytest.mark.parametrize("p,ask,bid", [(.55, 50, 49), (.45, 51, 50)])
def test_fee_can_reverse_gross_edge(p, ask, bid):
    assert evaluate_signal("T", p, .9, 15, ask, bid, 1000, min_edge=0,
                           net_edge_min=0, fee_per_contract=.07) is None


def test_score_features_precede_points_and_respect_service_and_dates():
    # Six breaks/holds all won by player two, then alternating holds to 6-6.
    games = parse_pbp_games("RRRD;SSSA;RRRR;SSSS;RRRR;SSSS." + "SSSS;" * 12
                            + "S/RR/SS/RR.SSSS")
    elo = EloTracker()
    elo._update_elo("2", "1", "Hard", 32, "20240101")
    elo._update_elo("1", "2", "Hard", 32, "20240102")
    match = dict(match_id="m", games=games, player1_id="1", player2_id="2",
                 match_date="2024-01-02", surface="Hard", tourney_level="A", best_of=3)
    df = build_training_rows_from_matches([match], None, elo, None)
    assert df.iloc[0].elo_diff == -32  # same-day result excluded
    assert df.iloc[4].elo_diff == 32
    assert df.iloc[4].games_server == 1
    assert (df.iloc[24].sets_server, df.iloc[24].sets_returner) == (0, 1)
    tb = df[df.is_tiebreak.eq(1)]
    assert tb.server.tolist() == [0, 1, 1, 0, 0, 1, 1]
    assert tb.points_p1.tolist() == list(range(7))
    last = df.iloc[-4]
    assert (last.sets_server, last.sets_returner, last.is_final_set, last.server) == (1, 1, 1, 1)
    assert last.games_server == last.games_returner == last.points_p1 == 0


def test_api_tennis_terminal_point_and_deuce():
    game = {"points": [{"number_point": str(i), "score": score} for i, score in enumerate(
        ["15 - 0", "15 - 15", "30 - 15", "40 - 15", "40 - 30", "40 - 40", "40 - A",
         "40 - 40", "A - 40"], 1)]}
    pts = _point_winners(game, 2, 1, False, 4)
    assert len(pts) == 10 and pts[-1] == (2, 0)
    with pytest.raises(ValueError, match="incomplete"):
        _point_winners(game, 2, 2, False, 4)


def test_calibration_with_sliced_indices_and_unknown_outcomes():
    from tennis.features import FEATURE_COLUMNS

    frame = pd.DataFrame([{**dict.fromkeys(FEATURE_COLUMNS, 0.), "point_idx": 0,
                           "match_id": str(idx), "match_winner": winner, "server": 1}
                          for idx, winner in enumerate([1, 0, 2])], index=[185, 200, 334])
    class Model:
        def predict_proba(self, features):
            return np.full(len(features), .60)
    probs, outcomes = build_match_win_probs_on_val(Model(), frame, MatchFormat())
    assert len(probs) == 2 and list(outcomes) == [1, 0]


def events():
    return [json.loads(line) for line in Path(__file__).with_name("fixtures").joinpath("replay.jsonl").read_text().splitlines()]


def test_replay_timing_partial_fills_fees_and_unsettled_exposure():
    tape = events()
    out = replay(tape, fee=.02)
    assert out["filled_contracts"] == 3 and out["partial_fills"] == 1
    assert out["settled_pnl_dollars"] == 1.44
    assert out["settled_fee_dollars"] == .06
    assert out["cash_dollars"] == 501.44
    open_result = replay(tape[:-1], fee=.02)
    assert open_result["unsettled_cost_dollars"] == 1.56 and open_result["settled_roi"] is None
    tape[3]["yes_ask"] = 51
    assert replay(tape, fee=.07)["unfilled_orders"] == 1


def test_no_side_and_settlement_ordering():
    tape = events()
    tape[1]["model_prob"] = .10
    tape[3]["yes_bid"] = 45
    tape[3]["bid_size"] = 2
    tape[-1]["yes_cents"] = 0
    assert replay(tape, fee=.02)["settled_pnl_dollars"] == .86
    with pytest.raises(ValueError, match="ordered"):
        replay([tape[1], tape[0]], fee=.02)
    with pytest.raises(ValueError, match="after settlement"):
        replay(tape + [tape[-1]], fee=.02)

def test_fee_rounds_once_per_fill_at_selected_precision():
    tape = events()
    result = replay(tape, taker_multiplier=1, fee_precision="0.01")
    assert result["settled_fee_dollars"] == .06  # 3 × .0175 rounded to one cent
    assert replay(tape, taker_multiplier=1)["settled_fee_dollars"] == .0525


def test_profitable_but_subthreshold_net_edge_is_rejected():
    # Gross $0.15, fee $0.07, net $0.08. Kelly would size a nonzero trade.
    args = dict(ticker="T", model_prob=.75, confidence=.80, points_observed=20,
                best_ask_cents=60, best_bid_cents=55, bankroll=500., fee_per_contract=.07)
    assert evaluate_signal(**args, net_edge_min=.10) is None
    assert evaluate_signal(**args, net_edge_min=.08) is not None


def test_fixture_tiebreak_records_and_missing_points():
    from tennis.fixture_history import parse_fixture

    # First set: twelve holds and a 7-0 tiebreak. Second set: P1 wins 6-0.
    games = []
    for sn, winners in [(1, [1, 2] * 6), (2, [1] * 6)]:
        counts = [0, 0]
        for gn, winner in enumerate(winners, 1):
            counts[winner - 1] += 1
            server = 1 if (gn + sn) % 2 == 0 else 2
            pts = [f"{v} - 0" if winner == 1 else f"0 - {v}" for v in (15, 30, 40)]
            games.append(dict(set_number=f"Set {sn}", number_game=str(gn),
                              score=f"{counts[0]} - {counts[1]}",
                              player_served=("First Player", "Second Player")[server - 1],
                              points=[dict(number_point=str(i), score=v) for i, v in enumerate(pts, 1)]))
    for i, srv in enumerate([1, 2, 2, 1, 1, 2, 2], 1):
        games.append(dict(set_number="Set 1 TieBreak", number_game=str(i),
                          score=f"{i} - 0", player_served=("First Player", "Second Player")[srv - 1]))
    games.append(dict(set_number="Set 1", number_game="13", score="7 - 6", points=[]))
    fixture = dict(event_key="m", tournament_name="Other", event_winner="First Player",
                   first_player_key=10, second_player_key=11, event_final_result="2 - 0",
                   scores=[dict(score_set=1, score_first="7.7", score_second="6.0"),
                           dict(score_set=2, score_first="6", score_second="0")],
                   pointbypoint=games, event_date="2025-01-02")
    _, match = parse_fixture(fixture)
    tb = match["games"][12]
    assert tb.tiebreak and tb.points[-1] == (2, 0)
    assert len(match["games"]) == 19
    del games[-3]
    with pytest.raises(ValueError, match="incomplete tiebreak"):
        parse_fixture(fixture)
