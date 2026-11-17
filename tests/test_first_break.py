from decimal import Decimal as D
from research.break_study import first_break
from research.record_break import tennis_rows
from research.score_break import Books, replay, SECOND


def game(n, serving, winner=None, score=None, pts=0):
    return dict(number_game=str(n),player_served=serving,serve_winner=winner,score=score,n_points=pts)


def test_confirmed_break_requires_contiguous_completed_games():
    hold = game(1, "First Player", "First Player", "1 - 0")
    broken = game(2, "Second Player", "First Player", "2 - 0")
    assert first_break([hold,broken]) == dict(game=2, broken=2, games_down=2, next_server=1)
    assert first_break([broken]) is None
    assert first_break([hold,game(2,"Second Player")]) is None
    assert first_break([hold,dict(broken,score="1 - 1")]) is None


def test_sequence_gap_requires_new_snapshot():
    books = Books()
    snap = dict(sid=1, seq=1, market_ticker="T", yes_dollars_fp=[[".37","5"]],no_dollars_fp=[[".60","2"]])
    books.apply("orderbook_snapshot",snap,0)
    assert books.ask("T","yes",0) == D(".40")
    books.apply("orderbook_delta",dict(sid=1, seq=3, market_ticker="T", side="no", price_dollars=".60",delta_fp="-1"),1)
    assert books.ask("T","yes",1) is None and books.gaps==1
    books.apply("orderbook_snapshot",dict(snap,seq=4),2)
    assert books.ask("T","no",2) == D(".63")


def test_delayed_ask_excludes_future_quote_and_duplicate_break():
    def row(at,kind,data):
        return dict(utc_ns=at*SECOND,mono_ns=at*SECOND,kind=kind,data=data)
    def frame(game_row):
        return dict(fixture="M", status="Set 1", live="1", winner=None, games=[game_row], game_score="0 - 0",
                    set_scores=[dict(score_set="1", score_first="0", score_second="1")])
    start = row(0,"start",dict(matches={"M":dict(ticker="T",yes_player=1)}))
    snap = row(1,"orderbook_snapshot",dict(market_ticker="T",sid=1,seq=1,
                    yes_dollars_fp=[[".25","5"]],no_dollars_fp=[[".68","2"]]))
    playing = row(1,"score",frame(game(1,"First Player")))
    lost = frame(game(1,"First Player","Second Player","0 - 1"))
    observed = row(2,"score",lost)
    later = row(5,"orderbook_delta",dict(market_ticker="T",sid=1,seq=2,side="no",price_dollars=".68",delta_fp="-2"))
    priors = {"M":dict(mid=".62",anchor="early_listing")}
    def checks(rows):
        return replay(rows,priors,{"early_listing":[0.,0.,0.]})["checks"]
    result = checks([start,snap,playing,observed,later,row(6,"score",lost)])
    assert len(result)==1 and result[0]["decision"]=="paper_fill" and D(result[0]["ask"])==D(".32")
    assert checks([start,snap,observed,later])[0]["decision"]=="catchup_or_late_break"
    assert checks([start,snap,playing,observed])[0]["decision"]=="capture_ended_in_delay"
    changed = dict(lost,games=lost["games"]+[game(2,"Second Player",pts=1)])
    assert checks([start,snap,playing,observed,row(3,"score",changed),later])[0]["decision"]=="score_changed_during_delay"

    in_next_game = dict(lost, game_score="15 - 0")
    assert checks([start,snap,playing,row(2,"score",in_next_game),later])[0]["decision"]=="catchup_or_late_break"


def test_feed_envelopes_include_all_matches():
    m={"event_key":1}
    for data in ([m],m,{"result":[m]},{"result":m},{"1":m}):
        assert tennis_rows(data)==[m]
