from research.break_hold import ServiceGameWatch, break_points, fixture_labels, is_pressure


def game(n,server,winner,score,states):
    return dict(set_number="Set 1",number_game=str(n),player_served=server,serve_winner=winner,
                score=score,points=[dict(number_point=str(i),score=s) for i,s in enumerate(states,1)])


def test_defended_game_keeps_prepoint_orientation():
    # From 30–40, first player saves a break point, then wins at advantage.
    states=["0 - 15","15 - 15","15 - 30","30 - 30","30 - 40","40 - 40","A - 40"]
    fixture={"pointbypoint":[game(1,"First Player","First Player","1 - 0",states)]}
    labels,_=fixture_labels(fixture)
    assert labels["1"]=={"game":1,"break_points":1,"result":"hold"}
    assert is_pressure("A - 40",2)
    assert not is_pressure("40 - A",2)


def test_first_break_point_game_and_incomplete_prefix():
    prior=game(1,"Second Player","Second Player","0 - 1",["0 - 15","0 - 30","0 - 40"])
    broken=game(2,"First Player","Second Player","0 - 2",["0 - 15","0 - 30","0 - 40"])
    labels,_=fixture_labels({"pointbypoint":[prior,broken]})
    assert labels["1"]["result"]=="break"
    labels,status=fixture_labels({"pointbypoint":[broken]})
    assert labels=={} and status=="incomplete_game_history"


def test_one_receiver_lead_from_deuce_counts():
    won=[1,0,1,0,1,0,0,1,0,0]
    assert break_points([(1,v) for v in won],1)==2


def test_live_hold_needs_observed_pressure_and_game_boundary():
    watcher=ServiceGameWatch(1)
    g=dict(number_game="1",player_served="First Player",serve_winner=None,n_points=5,last_point="30 - 40")
    msg=dict(live="1",winner=None,status="Set 1",game_score="30 - 40",games=[g],set_scores=[])
    assert watcher.process(msg,1) is None
    g=dict(g,serve_winner="First Player",score="1 - 0",n_points=7,last_point="A - 40")
    confirmed=dict(msg,games=[g],game_score="0 - 0",set_scores=[dict(score_set=1,score_first=1,score_second=0)])
    assert watcher.process(confirmed,2)=="hold"
    assert watcher.process(confirmed,3) is None
    assert ServiceGameWatch(1).process(confirmed,2)=="prior_game_incomplete"


def test_forward_replay_uses_point_then_price_and_charges_both_legs():
    import io
    import json

    from research.break_hold import tape_replay, SECOND

    rows=[]
    def append(t,kind,data):
        rows.append(dict(kind=kind,utc_ns=t*SECOND,mono_ns=t*SECOND,data=data))
    append(0,"start",dict(matches={"fixture":dict(ticker="T",yes_player=1)}))
    def score(t,games,score,total):
        append(t,"score",dict(fixture="fixture",live="1",winner=None,status="Set 1",games=games,
                               game_score=score,set_scores=[dict(score_set="1",score_first=total,score_second="0")]))
    playing=dict(number_game="1",player_served="First Player",serve_winner=None,n_points=2,last_point="15 - 15")
    score(10,[playing],"15 - 15","0")
    def book(t,seq,bid,no_bid):
        append(t,"orderbook_snapshot",dict(sid=1,seq=seq,market_ticker="T",
               yes_dollars_fp=[[bid,"10"]],no_dollars_fp=[[no_bid,"10"]]))
    book(59,1,".90",".08")
    pressure=dict(playing,n_points=5,last_point="30 - 40")
    score(119,[pressure],"30 - 40","0")
    book(119,2,".77",".21")
    won=dict(pressure,serve_winner="First Player",score="1 - 0",n_points=7,last_point="A - 40")
    score(170,[won],"0 - 0","1")
    book(179,3,".80",".18")
    append(183,"heartbeat",{})
    book(239,4,".90",".08")
    append(243,"end",{})
    tape=io.StringIO("\n".join(json.dumps(r) for r in rows)+"\n")
    result=tape_replay(tape,{"fixture":{"side":"yes","prior":".75"}})
    assert result["active_qualifying_games"]==1
    assert result["orders"][0]["status"]=="sold"
    assert result["orders"][0]["group"]=="hold"
    assert result["orders"][0]["entry"]=="0.82"
    assert result["orders"][0]["pnl"]=="0.05"


def test_challenger_reference_uses_correct_market_and_side(monkeypatch):
    from decimal import Decimal
    from research.break_hold import at_reference
    from research import break_hold
    import datetime
    pair=[{"ticker":"KXATPCHALLENGERMATCH-AB-A"},{"ticker":"KXATPCHALLENGERMATCH-AB-B"}]
    scheduled=datetime.datetime.fromisoformat("2026-11-17T12:30:00+00:00").timestamp()
    def quote(client, markets, anchor):
        assert markets==pair and anchor==scheduled
        return pair[1], Decimal(".80"),int(scheduled)-600
    monkeypatch.setattr(break_hold,"choose_favorite",quote)
    m={"series":"KXATPCHALLENGERMATCH","ticker":pair[0]["ticker"],"market_pair":pair,
       "fixture_start":"2026-11-17T12:30:00+00:00"}
    prior=at_reference(None,{"M":m},scheduled)
    assert prior["M"]["side"]=="no" and prior["M"]["prior"]=="0.80"
    assert at_reference(None,{"M":m},scheduled-900)=={}


def test_full_prefix_does_not_retroactively_trade_past_game():
    g=dict(number_game="1",player_served="First Player",serve_winner="First Player",score="1 - 0",
           n_points=7,last_point="A - 40",points=[
               {"number_point":str(i),"score":s} for i,s in enumerate(
                   ["0 - 15","15 - 15","15 - 30","30 - 30","30 - 40","40 - 40","A - 40"],1)])
    msg=dict(live="1",winner=None,status="Set 1",game_score="0 - 0",games=[g],set_scores=[
        dict(score_set="1",score_first="1",score_second="0")])
    assert ServiceGameWatch(1).process(msg,1)=="left_censored_game"


def test_tour_recorder_keeps_preexisting_matches_observation_only(monkeypatch):
    from research import record_break
    values=[]
    pair=[dict(ticker="KXATPCHALLENGERMATCH-E-A",event_ticker="E",yes_sub_title="Andy Player",
               occurrence_datetime="2026-11-17T14:00:00Z",open_time="2026-11-17T08:00:00Z"),
          dict(ticker="KXATPCHALLENGERMATCH-E-B",event_ticker="E",yes_sub_title="Bill Test",
               occurrence_datetime="2026-11-17T14:00:00Z",open_time="2026-11-17T08:00:00Z")]
    def tennis(client,method,**params):
        values.append(params["event_type_key"])
        return [dict(event_key=3,first_player_key=1,second_player_key=2,event_live="1",event_status="Set 1",
                     event_first_player="A. Player",event_second_player="B. Test",event_date="2026-11-17",event_time="14:00")]
    def exchange(client,path,**params):
        values.append(params["series_ticker"])
        return {"markets":pair}
    monkeypatch.setattr(record_break,"tennis",tennis)
    monkeypatch.setattr(record_break,"exchange",exchange)
    m=record_break.discover(None,"challenger")["3"]
    assert m["already_in_play"] and m["yes_player"]==1
    assert values==[281,"KXATPCHALLENGERMATCH"]
