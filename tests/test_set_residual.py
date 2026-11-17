from decimal import Decimal as D

import pytest

from research.set_residual import first_set_feature, fit, market_mid, pick


def fixture():
    games=[]
    for i in range(1,7):
        games.append(dict(set_number="Set 1",number_game=str(i),player_served="First Player" if i%2 else "Second Player",
                          serve_winner="First Player",score=f"{i} - 0",
                          points=[dict(number_point=str(j),score=f"{v} - 0") for j,v in enumerate((15,30,40),1)]))
    return dict(tournament_name="ATP Test",event_date="2026-10-01",first_player_key=1,second_player_key=2,
                pointbypoint=games+[{"set_number":"Set 2","score":"unreadable"}])


def test_feature_reverses_with_ticker_orientation_and_ignores_future_points():
    f=fixture()
    x=first_set_feature(f,0,0)
    assert x["yes_served"]==12 and x["yes_serve_won"]==12
    assert x["x"]==pytest.approx(12/32)
    assert first_set_feature(f,1,0)["x"]==-x["x"]
    with pytest.raises(ValueError):
        first_set_feature(f,1,1)


def test_wrong_direction_leaves_market_baseline_intact():
    rows=[dict(bid=45,ask=55,x=.2,settlement="0"),dict(bid=45,ask=55,x=-.2,settlement="1")]
    assert market_mid(rows[0])==.5
    assert fit(rows)==0
    assert fit([dict(r,x=-r["x"]) for r in rows])>0


def test_spread_fee_and_five_cent_threshold():
    assert pick(.57,49,50)==("yes",D(".50"),D(".05"))
    assert pick(.5699,49,50) is None
    assert pick(.42,50,51)==("no",D(".5"),D(".06"))
