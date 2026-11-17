from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from integration.tennis_feed import TennisFeed
from tennis.calibration import MatchWinCalibrator
from tennis.elo import EloTracker
from tennis.engine import PredictionEngine
from tennis.markov import game_win_prob_from_score, match_win_prob, tiebreak_win_prob_from_score
from tennis.research import checkpoint_predictions
from tennis.types import MatchState, MatchFormat


def test_recurrences_and_player_swap():
    p, other = .65, .57
    assert game_win_prob_from_score(p,4,3) > game_win_prob_from_score(p,3,3)
    # Tiebreak first-server P1 currently receives: one point ahead and one down.
    deuce = tiebreak_win_prob_from_score(p,7,7,1-other)
    assert tiebreak_win_prob_from_score(p,7,6,1-other) == pytest.approx(1-other + other*deuce)
    assert tiebreak_win_prob_from_score(p,6,7,1-other) == pytest.approx((1-other)*deuce)
    for s in [MatchState(server=1, games_p1=2,games_p2=3,points_p1=3,points_p2=4),
              MatchState(server=1,games_p1=6,games_p2=6,points_p1=2,points_p2=3,is_tiebreak=True)]:
        swap = replace(s, server=1-s.server, games_p1=s.games_p2,games_p2=s.games_p1,
                       points_p1=s.points_p2,points_p2=s.points_p1)
        assert match_win_prob(s,p,other) + match_win_prob(swap,p,other) == pytest.approx(1.)
    s = MatchState()
    assert match_win_prob(s,.63) == pytest.approx(.5)
    assert match_win_prob(s,.65,.57) > .5
    assert match_win_prob(replace(s,server=1),.57,.65) == pytest.approx(match_win_prob(s,.65,.57))


def test_deciding_slam_tiebreak_plays_to_ten():
    s = MatchState(sets_p1=2,sets_p2=2,games_p1=6,games_p2=6,is_tiebreak=True,
                   points_p1=7,points_p2=5,match_format=MatchFormat.best_of_5_tiebreak())
    assert .5 < match_win_prob(s,.5) < 1
    assert match_win_prob(replace(s,points_p1=10),.5) == 1


def test_point_updates_do_not_transfer_to_other_server():
    eng = PredictionEngine(None,MatchWinCalibrator(),None,None,n_mc_samples=100,rng_seed=1)
    eng.start_with_priors(.65,.52)
    for _ in range(5):
        eng.record_point(True,server=0)
    assert eng.predict(MatchState(server=0)).p_serve_estimate > .65
    other = eng.predict(MatchState(server=1))
    assert other.p_serve_estimate == pytest.approx(.52)
    assert other.points_observed == 5


def test_checkpoint_uses_only_past_points_and_equal_match_weight():
    from tennis.features import FEATURE_COLUMNS
    frame = pd.DataFrame([{**dict.fromkeys(FEATURE_COLUMNS,0), "point_idx":i,"match_id":mid,"match_winner":1,
                           "best_of":3,"server":0,"server_won":int(i%2==0),"points_p1":0,"points_p2":0}
                          for mid, n in [("a",14),("b",24)] for i in range(n)])
    model = SimpleNamespace(predict_proba=lambda X: np.full(len(X),.62))
    raw = checkpoint_predictions(model,frame)
    assert raw.groupby("match_id").weight.sum().to_dict() == {"a":1.,"b":1.}
    frame.loc[frame.point_idx>=10,"server_won"] = 0
    changed = checkpoint_predictions(model,frame)
    np.testing.assert_array_equal(changed.raw[changed.points_observed<=10],raw.raw[raw.points_observed<=10])
    assert not np.allclose(changed.raw[changed.points_observed==20],raw.raw[raw.points_observed==20])


def test_inplay_calibration_roundtrip_and_complement(tmp_path):
    cal = MatchWinCalibrator()
    cal.fit_inplay([.1,.4,.6,.9],[0,1,0,1],[1,1,1,1])
    cal.player_id_source = "api-tennis"
    filename = tmp_path/"calibrator.pkl"
    cal.save(str(filename))
    loaded = MatchWinCalibrator.load(str(filename))
    assert loaded.player_id_source == "api-tennis"
    for p in (.02,.5,.98):
        assert loaded.calibrate(p) == pytest.approx(1-cal.calibrate(1-p))
    assert loaded.calibrate(.99) > loaded.calibrate(.5) > loaded.calibrate(.01)


@pytest.mark.asyncio
async def test_backlog_only_warms_model_and_new_point_preserves_score_and_server():
    events = []
    async def collect(e):
        events.append(e)
    feed = TennisFeed("",collect)
    m = dict(event_key="1", event_first_player="A",event_second_player="B",first_player_key=11,
             second_player_key=12,event_status="Set 1",event_type_type="Atp Singles",event_date="2026-10-01",
             event_serve="Second Player",event_game_result="0 - 0",scores=[dict(score_set=1,score_first="0",score_second="1")],
             pointbypoint=[dict(set_number="Set 1",number_game=1,player_served="First Player",
                                score="0 - 1",serve_winner="Second Player",
                                points=[dict(number_point=i,score=s) for i,s in enumerate(("0 - 15","0 - 30","0 - 40"),1)])])
    await feed._process_match(m)
    assert len(events)==4 and not any(e.actionable for e in events)
    assert (events[-1].games_p1,events[-1].games_p2,events[-1].point_server,events[-1].server)==(0,1,0,1)
    m["pointbypoint"].append(dict(set_number="Set 1",number_game=2,player_served="Second Player",points=[dict(number_point=1,score="0 - 15")]))
    m["event_game_result"]="0 - 15"
    await feed._process_match(m)
    await feed._process_match(m)
    assert len(events)==5 and events[-1].actionable and events[-1].server_won_last_point
    assert events[-1].player1_key=="11"


def test_unknown_surface_uses_only_earlier_overall_elo():
    rows = pd.DataFrame([dict(tourney_date=20260902,winner_id=1,loser_id=2,surface="Unknown")])
    elo = EloTracker()
    elo.process_matches(rows)
    assert elo.get_rating("1","Unknown","20260901")==1500
    assert elo.get_rating("1","Clay","20260903")>1500
