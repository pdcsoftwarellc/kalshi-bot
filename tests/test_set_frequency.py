from decimal import Decimal

from research.set_frequency import confirmation, print_hit


def test_first_set_decision_waits_for_both_settlements():
    def market(result, val, player, when):
        return dict(result=result,settlement_value_dollars=val,settlement_ts=when,
                    custom_strike={"tennis_competitor":player})
    pair=[market("yes","1","A","2026-10-03T18:00:00Z"),market("no","0","B","2026-10-03T18:00:01Z")]
    decision,confirmed,winner=confirmation(pair)
    assert decision-confirmed==59 and winner=="A"
    assert confirmation([pair[0]]) is None
    assert confirmation([pair[0],dict(pair[1],settlement_value_dollars=".50")]) is None


def test_trade_window_price_and_side_are_observable():
    def trade(second, side, price, size="1"):
        return dict(created_time=f"1970-01-01T00:00:{second:02d}Z",taker_outcome_side=side,
                    yes_price_dollars=price,count_fp=size,trade_id=str(second))
    rows=[trade(1,"yes",".50"),trade(2,"no",".49"),trade(3,"yes",".51"),trade(5,"yes",".50")]
    assert print_hit(rows,"yes",Decimal(".50"),0,5) is None
    assert print_hit(rows+[trade(4,"yes",".49")],"yes",Decimal(".50"),0,5)["ts"]==4


def test_forecast_cannot_consume_later_sets():
    from research.set_frequency import forecast
    from unittest.mock import Mock
    games=[]
    for i in range(1,7):
        games.append(dict(set_number="Set 1",number_game=str(i),player_served="First Player" if i%2 else "Second Player",
                          serve_winner="First Player",score=f"{i} - 0",
                          points=[dict(number_point=str(j),score=f"{v} - 0") for j,v in enumerate((15,30,40),1)]))
    fixture=dict(event_date="2026-10-01",first_player_key=1,second_player_key=2,event_key=100,
                 pointbypoint=[*games, {"set_number":"Set 2", "score":"inaccessible"}])
    engine=Mock()
    forecast(engine,fixture,0)
    state=engine.predict.call_args.args[0]
    assert (state.sets_p1,state.sets_p2,state.games_p1,state.games_p2)==(1,0,0,0)
    assert engine.record_point.call_count==24
