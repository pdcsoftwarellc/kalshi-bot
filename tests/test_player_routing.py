from decimal import Decimal as D

from research.player_routing import comparable_pair, pick_route


def test_cheapest_direction_preserves_normal_match_exposure():
    assert pick_route("yes", D(".70"), (33,35)) == (True,"no",D(".67"))
    assert pick_route("no", D(".30"), (25,28)) == (True,"yes",D(".28"))
    assert pick_route("yes", D(".70"), (30,32)) == (False,"yes",D(".70"))
    assert pick_route("no", D(".30"), None) == (False,"no",D(".30"))


def test_requires_distinct_participants_and_same_contract_terms():
    event = dict(event_ticker="ATP-E", series_ticker="KXATPMATCH", mutually_exclusive=True)
    def market(name):
        return dict(event_ticker="ATP-E", yes_sub_title=name, market_type="binary",
                    notional_value_dollars="1", custom_strike={"tennis_competitor":name},
                    rules_primary=f"If {name} wins the match, Yes.", rules_secondary="Retirement terms")
    a,b = market("A"),market("B")
    assert comparable_pair([a,b],event)
    assert not comparable_pair([a,a],event)
    assert not comparable_pair([a,dict(b,rules_secondary="Different terms")],event)
    assert not comparable_pair([a,b],dict(event,mutually_exclusive=False))
