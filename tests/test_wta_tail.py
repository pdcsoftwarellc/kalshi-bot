from decimal import Decimal as D

from research.wta_tail import fresh_quote, select, sensitivity


def test_select_80_to_90_cent_unique_favorite_inclusive():
    def pair(low, high):
        return {"A":(D(low),D(high),200),"B":(D(".12"),D(".13"),200)}
    assert select(pair(".77",".80"))[0]=="A"
    assert select(pair(".89",".90"))[0]=="A"
    assert select(pair(".76",".80")) is None
    assert select(pair(".77",".79")) is None
    assert select(pair(".90",".91")) is None
    assert select({"A":(D(".82"),D(".84"),200),"B":(D(".81"),D(".83"),200)}) is None


def test_quote_does_not_look_ahead_or_resurrect_bad_last_bar():
    def bar(t,bid,ask):
        return {"end_period_ts":t,"yes_bid":{"close_dollars":bid},"yes_ask":{"close_dollars":ask}}
    bars=[bar(100,".82",".84"),bar(401,".01",".03")]
    assert fresh_quote(bars,400)==(D(".82"),D(".84"),100)
    assert fresh_quote(bars[:1],401) is None
    assert fresh_quote(bars+[bar(399,None,".86")],400) is None


def test_adverse_cent_keeps_original_fills_and_recomputes_fee():
    base=dict(ts=100,ticker="test",print_ts=103,limit=".83",fee=".01",payout="1",
              settlement_ts=500)
    assert sensitivity([base,dict(base,ticker="C",print_ts=None)]) == .15
