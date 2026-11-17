from datetime import datetime
from research.short_rest import confirmation, deciding_set, rested_opponent


def fixture(a,b,score="2 - 1",winner="First Player",status="Finished"):
    return dict(first_player_key=a,second_player_key=b,event_final_result=score,
                event_winner=winner,event_status=status)


def test_decider_must_be_a_completed_competitive_match():
    assert deciding_set(fixture("a","b"))
    assert deciding_set(fixture("a","b","2 - 3","Second Player"))
    for f in [fixture("a","b","2 - 0"),fixture("a","b","2 - 1","Second Player"),
              fixture("a","b",status="Retired")]:
        assert not deciding_set(f)


def test_absence_and_unambiguous_prior_appearance_required():
    current = fixture("a","b")
    previous = fixture("c","b","1 - 2","Second Player")
    assert rested_opponent(current,[previous]) == (previous,"a")
    assert rested_opponent(current,[previous,fixture("a","d","2 - 0")]) is None
    assert rested_opponent(current,[previous,fixture("b","d","2 - 0")]) is None


def test_both_prior_markets_must_confirm_winner_before_entry():
    prev = fixture("c","b","1 - 2","Second Player")
    time = '2026-10-14T18:00:00+00:00'
    pair = [dict(settlement_ts=time,settlement_value_dollars=str(v)) for v in (1,0)]
    linked = pair,prev,"b","c"
    cutoff = datetime.fromisoformat(time).timestamp()
    assert confirmation(linked,prev,cutoff) is None
    assert confirmation(linked,prev,cutoff+1) == cutoff
    assert confirmation((pair,prev,"c","b"),prev,cutoff+1) is None
    assert confirmation((pair[:1]+[dict(pair[1],settlement_ts=None)],prev,"b","c"),prev,cutoff+1) is None
