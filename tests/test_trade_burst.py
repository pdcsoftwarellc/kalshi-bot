from datetime import datetime, timezone
from decimal import Decimal as D
from unittest.mock import patch

from research import trade_burst as study


def iso(t):
    return datetime.fromtimestamp(t, timezone.utc).isoformat()


def trade(t, idx, side="yes", qty="20"):
    return dict(ticker="T", created_time=iso(t), trade_id=str(idx),
                taker_outcome_side=side, count_fp=qty, yes_price_dollars=".55", no_price_dollars=".45")


def test_burst_requires_unique_nonblock_volume_strictly_before_decision():
    valid = [trade(700+i, i, "yes" if i<4 else "no") for i in range(5)]
    extras = [trade(699, 8, "no", "100"), trade(1000,9,"no","100"),
              dict(trade(900,10,"no","100"),is_block_trade=True), valid[-1]]
    assert study.direction(valid+extras,1000)==("yes",5,D(100),D(80))
    assert study.direction(valid[:-1]+extras[:-1],1000)[0] is None


def test_sale_of_no_checks_opposing_taker_and_no_equivalent_price():
    prints = [trade(1001,1),trade(1003,2,"no"),trade(1004,3,"yes",".99"),trade(1005,4)]
    with patch.object(study,"trade_list",return_value=prints):
        assert study.print_after(None,"T","no",False,D(".45"),1000,2000)==(1005,"4")
        assert study.print_after(None,"T","no",False,D(".46"),1000,2000) is None


def test_missing_exit_exposes_buyer_to_settlement_loss():
    m=dict(ticker="T",close_time=iso(5000),open_time=iso(0),settlement_ts=iso(5100),
           settlement_value_dollars="0")
    with patch.object(study,"trade_list",return_value=[trade(1010+i,i) for i in range(5)]), \
            patch.object(study,"read_quote",side_effect=lambda c,t,s:(D(".48"),D(".50"),s)), \
            patch.object(study,"print_after",side_effect=lambda c,t,s,b,l,when,end:(when+3,"print") if b else None):
        res=study.replay(None,("fixture","event",m,700,700,700))
    assert res["status"]=="settled" and D(res["pnl"])==D("-.52")
