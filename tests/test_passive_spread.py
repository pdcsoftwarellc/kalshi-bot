from decimal import Decimal as D

from research.passive_spread import maker_fee, order, summarize


def simulate(trades,quotes,mode="maker",settle="0"):
    return order(mode,{"ts":0,"ticker":"T"},D(".45"),D(".50"),trades,quotes,D(settle),2000)


def test_maker_needs_opposite_taker_and_trade_through_on_both_legs():
    rows=[(1,"no",D(".43"),"early"),(3,"yes",D(".41"),"wrong"),(5,"no",D(".45"),"touch"),
          (7,"no",D(".44"),"buy"),(61,"yes",D(".53"),"too-early"),(65,"yes",D(".50"),"touch2"),
          (68,"yes",D(".51"),"sale")]
    r=simulate(rows,[(60,D(".46"),D(".48"))])
    assert (r["entry_print"],r["exit_print"],r["status"])==("buy","sale","sold")
    _,s=summarize([r],"maker",D("80"))
    assert D(s["net"])==D(".03") and D(s["fees"])==D(".02") and D(s["taker_fee_stress_net"])==D(".01")


def test_post_only_would_cross_keeps_real_inventory_loss():
    r=simulate([(4,"no",D(".42"),"buy"),(63,"yes",D(".56"),"sale")],
               [(60,D(".52"),D(".56"))])
    assert r["exit_reason"]=="post_only_exit_would_cross"
    _,s=summarize([r],"maker",D("80"))
    assert D(s["unmatched_inventory_net"])==D("-.46") and s["inventory_settled"]==1


def test_stale_exit_or_post_horizon_print_cannot_fill():
    rows=[(5,"no",D(".43"),"buy"),(605,"yes",D(".55"),"late")]
    assert simulate(rows,[(60,D(".48"),D(".50"))])["status"]=="settled"
    assert simulate(rows,[(120,D(".48"),D(".50"))])["exit_reason"]=="no_post_entry_quote_within_one_minute"
    assert maker_fee(D(".99"))==D(".01")
