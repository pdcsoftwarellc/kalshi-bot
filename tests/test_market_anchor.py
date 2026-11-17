from decimal import Decimal as D
from unittest.mock import patch

import pytest

from research.market_anchor import anchored_probability, prematch_reference
from research.score_feed_markout import observe


def test_market_odds_move_only_with_new_evidence():
    assert anchored_probability(.7,.4,.4) == pytest.approx(.7)
    # Native odds doubled from 1:1 to 2:1, taking market odds from 3:1 to 6:1.
    assert anchored_probability(.75,.5,2/3) == pytest.approx(6/7)
    assert anchored_probability(.25,.5,1/3) == pytest.approx(1/7)


def test_reference_cannot_use_live_or_stale_quote():
    row = dict(ticker="T", prematch_cutoff=600, confirmed=1800, opened=200)
    q = (D(".49"),D(".51"),601)
    with patch("research.market_anchor.read_quote",return_value=q):
        assert prematch_reference(None,row) is None
    with patch("research.market_anchor.read_quote",return_value=(*q[:2],600)):
        assert prematch_reference(None,row) == (.5,600)
        assert prematch_reference(None,dict(row,confirmed=500)) is None


def test_markout_uses_executable_side_and_ignores_later_updates():
    def q(t,b,a):
        return dict(ts_ms=t,yes_bid=b,yes_ask=a,bid_size=3,ask_size=3)
    qs=[q(0,49,51),q(1900,45,47),q(2100,40,42),q(30000,30,32),q(30001,20,22)]
    base=dict(ts_ms=0,point_winner_yes=False,bid=49,ask=51,bid_size=2,ask_size=2,book_age_ms=0,decoded_in_receipt=1)
    index=qs,[r["ts_ms"] for r in qs]
    result=observe(index,base,30001)
    assert result["status"]=="measured" and result["entry"]==.55 and result["exit"]==.68
    assert result["gross_cents"]==13 and result["net_cents"]==9
    assert observe(index,base,29000)["status"]=="missing_exit_or_window_ended"
    assert observe(index,dict(base,decoded_in_receipt=2),30001)["status"]=="batched_points"
