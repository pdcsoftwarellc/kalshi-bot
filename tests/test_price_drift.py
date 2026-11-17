from decimal import Decimal as D

import pytest

from research import price_drift


def q(bid, ask):
    return D(bid), D(ask), 0


@pytest.mark.parametrize("early,late,expected", [
    (q(".48",".52"), q(".53",".57"), ("yes",D(".57"),D(".05"))),
    (q(".48",".52"), q(".43",".47"), ("no",D(".57"),D("-.05"))),
    (q(".48",".52"), q(".52",".56"), None),
    (q(".47",".52"), q(".53",".57"), None),
    (q(".47",".50"), q(".52",".57"), None),
])
def test_fixed_momentum_and_complementary_limit(early, late, expected):
    assert price_drift.signal(early, late) == expected


def test_quote_freshness_boundary_no_future_and_no_forward_filling(monkeypatch):
    def bar(ts,bid,ask):
        return {"end_period_ts":ts, "yes_bid":{"close_dollars":bid}, "yes_ask":{"close_dollars":ask}}
    bars = [bar(699,".41",".42"), bar(700,".47",".49"), bar(1001,".69",".70")]
    monkeypatch.setattr(price_drift, "fetch", lambda *a, **k: {"candlesticks":bars})
    assert price_drift.read_quote(None, "T", 1000) == (D(".47"),D(".49"),700)
    bars.append(bar(995,None,None))
    assert price_drift.read_quote(None, "T", 1000) is None
