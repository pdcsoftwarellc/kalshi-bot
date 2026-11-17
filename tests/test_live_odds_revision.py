from decimal import Decimal as D

from research.live_odds_revision import SECOND, replay, revision, usable


def sample(home="2", away="2", tag="00:01", score="15 - 0"):
    return dict(event_status="Set 1",event_winner=None,event_game_result=score,
                event_serve="First Player",event_final_result="0 - 0",first_player_key=1,second_player_key=2,
                lines=[dict(odd_name="To Win",type=side,value=price,suspended="No",handicap=None,upd=tag)
                       for side,price in (("Home",home),("Away",away))])


def test_same_context_revision_and_orientation():
    a,b=sample(), sample("1.5","3",tag="00:02")
    assert revision(usable(a,1), usable(b,1), 5*SECOND) == D(2)/3-D(".5")
    assert usable(b,2)["p"] == D(1)/3
    assert revision(usable(a,1),usable(b,1),11*SECOND) is None
    assert revision(usable(a,1),usable(dict(b,event_game_result="30 - 0"),1),5*SECOND) is None
    assert revision(usable(a,1),usable(sample("1.5","3"),1),5*SECOND) is None
    b["lines"][0]["suspended"]="Yes"
    assert usable(b,1) is None


def record(t,kind,data):
    return dict(mono_ns=int(t*SECOND),utc_ns=int(t*SECOND)+2_000_000_000,kind=kind,data=data)


def snap(t,seq,yes,no):
    return record(t,"orderbook_snapshot",dict(sid=1,seq=seq,market_ticker="T",
        yes_dollars_fp=[[yes,"3"]],no_dollars_fp=[[no,"3"]]))


def trial():
    return [record(0,"start",dict(matches={"F":dict(ticker="T",yes_player=1)})),
            snap(0,1,".47",".48"),record(1,"odds",{"F":sample()}),
            snap(5.9,2,".47",".48"),record(6,"odds",{"F":sample("1.5","3","00:02")}),
            snap(7.8,3,".48",".49"),snap(37.8,4,".62",".37"),record(39,"end",dict(normal=True))]


def test_replay_uses_sized_quotes_at_due_times_and_roundtrip_fees():
    report=replay(trial()); row=report["ledger"][0]
    assert report["revision_count"]==report["qualified"]==report["entered"]==report["closed"]==1
    assert D(row["buy"])==D(".52") and D(row["sell"])==D(".62")
    assert D(row["net"])==D(".06") and report["unresolved"]==0


def test_unpriced_exit_and_suspension_are_visible():
    frames=trial()
    result=replay(frames[:-2]+[frames[-1]])
    assert result["entered"]==result["unresolved"]==1 and result["closed"]==0
    assert D(result["unresolved_cost"])==D(".54")
    frames.insert(4,record(5.95,"odds",{}))
    assert replay(frames)["revision_count"]==0


def crosscheck(row, set_games=("2","1"), stamped=5, odds_at=5):
    verified={k:row[k] for k in ("event_status","event_game_result","event_serve","first_player_key","second_player_key")}
    verified.update(receipt_utc_ns=stamped*SECOND,event_winner=None,event_live="1",event_final_result="0 - 0",
                    scores=[{"score_set":"1","score_first":set_games[0],"score_second":set_games[1]}])
    return dict(row,score_verification=verified,odds_receipt_utc_ns=odds_at*SECOND)


def test_point_revision_must_match_direction_at_same_game_with_verified_score():
    before,after=(crosscheck(sample()),crosscheck(sample("1.5","3","00:02",score="30 - 0")))
    def check(a,b,yes):
        return revision(usable(a,yes,True),usable(b,yes,True),5*SECOND,point_mode=True,yes_player=yes)
    assert check(before,after,1)>D(".03")
    assert check(before,after,2)<D("-.03")
    assert check(before,crosscheck(sample("1.5","3","00:02",score="15 - 15")),1) is None
    assert check(before,crosscheck(sample("1.5","3","00:02",score="40 - 0")),1) is None
    assert check(before,crosscheck(sample("1.5","3","00:02",score="30 - 0"),set_games=("3","1")),1) is None
    assert usable(crosscheck(sample(),set_games=("6","6")),1,True) is None
    assert usable(crosscheck(sample(),stamped=10),1,True) is None
    assert usable(sample(),1,True) is None


def test_point_mode_advantage_winner_and_no_game_reset():
    from research.live_odds_revision import ordinary_point
    def ctx(s):
        return ("Set 1",s,"Second Player","0 - 0",1,2,0,0,2,2)
    assert ordinary_point(ctx("40 - A"),ctx("40 - 40"))==1
    assert ordinary_point(ctx("A - 40"),ctx("40 - 40"))==2
    assert ordinary_point(ctx("A - 40"),ctx("0 - 0")) is None
    assert ordinary_point(ctx("7 - 7"),ctx("8 - 7")) is None


def test_point_capture_requires_verified_move_and_reprices_after_delay():
    frames=trial()
    frames[0]["data"]["point_mode"]=True
    frames[2]["data"]["F"]=crosscheck(sample())
    frames[4]["data"]["F"]=crosscheck(sample("1.5","3","00:02",score="30 - 0"))
    report=replay(frames)
    assert report["coverage"]["legal_point_transitions"]==1
    assert report["qualified"]==report["closed"]==1
    assert D(report["net_closed_dollars"])==D(".06")
    # A new score receipt before the entry invalidates the old opportunity.
    frames.insert(5,record(7,"odds",{"F":crosscheck(sample("1.5","3","00:02",score="40 - 0"))}))
    report=replay(frames)
    assert report["entered"]==0 and report["ledger"][0]["state"]=="intervening_odds_unusable"
