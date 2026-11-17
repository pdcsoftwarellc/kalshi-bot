from research.paired_prices import scan, cash_ledger, fill_proxy
from decimal import Decimal as D


def test_synchronized_closes_and_cost_after_fees():
    v = dict(tickers=["A","B"], minute_asks={"A":{"60":".46","120":".47","180":".45","240":".44"},
                                              "B":{"120":".49","180":".48","240":".48"}})
    n, best, signal = scan(v)
    assert n==3 and best==D(".96") and signal["ts"]==180


def test_one_leg_exposes_full_downside():
    s=dict(event="E",ts=1000,prices=[".45",".46"],prints=["t",None],settlement=["0","1"],
           settlement_ts=["2026-01-01T00:00:00Z"]*2)
    p=cash_ledger([s])
    assert p["single_legs"]==1 and p["gross_dollars"]==-.45 and p["net_dollars"]==-.47
    p=cash_ledger([dict(s,prints=["t","u"])])
    assert p["both_legs"]==1 and p["net_dollars"]==.05


def test_trade_proxy_excludes_outside_window_and_wrong_side():
    t=dict(ticker="A", trade_id="available",yes_price_dollars=".43",count_fp="2.00",
           taker_outcome_side="yes",created_time="1970-01-01T00:00:02Z")
    assert fill_proxy([dict(t,created_time="1970-01-01T00:00:07Z"),dict(t,taker_outcome_side="no"),
                       dict(t,count_fp=".1")],D(".44"),0,"A") is None
    assert fill_proxy([t],D(".44"),0,"A")=="available"
