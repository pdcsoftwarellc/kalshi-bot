from decimal import Decimal as D

from research.decider_basis import cheapest_pair, exact_slate, proof, tie_confirmation
from research.paired_prices import cash_ledger


def settled(pid, winner, stamp):
    return dict(custom_strike={"tennis_competitor":pid}, result="yes" if winner else "no",
                settlement_ts=stamp, settlement_value_dollars="1.00" if winner else "0.00")


def test_tie_requires_later_public_confirmation_and_opposite_winners():
    first=[settled("A",True,"1970-01-01T00:01:00Z"),settled("B",False,"1970-01-01T00:01:00Z")]
    second=[settled("A",False,"1970-01-01T00:02:00Z"),settled("B",True,"1970-01-01T00:02:00Z")]
    assert tie_confirmation(first,second)==(180,120.0)
    assert tie_confirmation(first,first) is None
    assert tie_confirmation(first,second[:1]) is None


def test_optimizes_sides_without_double_exposure_to_same_winner():
    match={"A":{"ticker":"ma"},"B":{"ticker":"mb"}}
    set3={"A":{"ticker":"sa"},"B":{"ticker":"sb"}}
    # Cheapest isolated prices both back A: ma YES and sb NO. That basket is unhedged.
    q={"ma":{"yes":D(".30")},"sb":{"no":D(".35")},"sa":{"no":D(".55")}}
    result=cheapest_pair(match,set3,q)
    assert result["tickers"]==["ma","sa"] and result["sides"]==["yes","no"]
    assert D(result["cost"])==D(".89")
    assert cheapest_pair(match,set3,{"ma":q["ma"],"sb":q["sb"]}) is None


def test_format_requires_both_players_and_prelisted_winning_scores():
    pair={"a":{"yes_sub_title":"Alice"},"b":{"yes_sub_title":"Betty"}}
    markets=[dict(custom_strike={"tennis_competitor":pid,"Set Score":s}, open_time="1970-01-01T00:00:00Z", market_type="binary",
                  notional_value_dollars="1",rules_primary=f"If {player['yes_sub_title']} wins the match by a set score of {s}, then yes.")
             for pid,player in pair.items() for s in ("2-0","2-1")]
    assert set(exact_slate(markets,pair,30))=={"a","b"}
    assert exact_slate(markets[:-1],pair,30) is None
    assert exact_slate([*markets[:-1],dict(markets[-1],open_time="1970-01-01T00:01:00Z")],pair,30) is None


def test_only_correct_side_print_in_window_supports_a_leg_and_miss_can_lose():
    example=dict(created_time="1970-01-01T00:00:04Z",ticker="s",taker_outcome_side="no",
                 no_price_dollars=".53",count_fp="1",trade_id="x")
    assert proof([example],"s","no",D(".54"),0,10)=="x"
    assert proof([example],"s","yes",D(".54"),0,10) is None
    assert proof([example],"s","no",D(".52"),0,10) is None
    assert proof([example],"s","no",D(".54"),0,4) is None
    ledger=cash_ledger([dict(ts=0,event="m",prices=[".4",".53"],prints=[None,"x"],settlement=["1","0"],
                             settlement_ts=["1970-01-01T01:00:00Z","1970-01-01T01:00:00Z"])])
    assert ledger["single_legs"]==1 and ledger["unhedged_net"]==-.55


def test_direct_member_fee_reranks_at_original_threshold():
    from research.decider_basis import direct_fee
    assert direct_fee(D(".01"),D(1)) == D(".0007")
    assert direct_fee(D(".96"),D(1)) == D(".0027")
    # Price outside the cent grid still aligns total cost to centicent.
    assert direct_fee(D(".055"),D(1)) == D(".0037")
    m={"A":{"ticker":"mA"},"B":{"ticker":"mB"}}
    s={"A":{"ticker":"sA"},"B":{"ticker":"sB"}}
    quotes={"mA":{"yes":D(".01")},"sB":{"yes":D(".96")}}
    assert D(cheapest_pair(m,s,quotes)["cost"])==D(".99")
    direct=cheapest_pair(m,s,quotes,{"mA":D(1),"sB":D(1)})
    assert D(direct["cost"])==D(".9734")
    assert [D(f) for f in direct["leg_fees"]]==[D(".0007"),D(".0027")]


def test_direct_single_leg_loss_keeps_actual_payout_and_fee():
    from research.decider_basis import direct_cash_ledger
    signal=dict(ts=0,event="m",prices=[".01",".96"],leg_fees=[".0007",".0027"],
                prints=["x",None],settlement=["0","1"],settlement_ts=["1970-01-01T00:01:00Z"]*2)
    result=direct_cash_ledger([signal])
    assert result["single_legs"]==1 and result["unhedged_net"]==-.0107
