from decimal import Decimal as D
from unittest.mock import patch

from research import favorite_rebound as experiment


def quote(t, midpoint):
    midpoint = D(midpoint)
    return (t, midpoint - D(".01"), midpoint + D(".01"))


def test_waits_for_confirmation_without_using_future_low():
    bars = [quote(0, ".92"), quote(60, ".81"), quote(120, ".80"), quote(180, ".81")]
    assert experiment.signal(bars) is None
    bars.append(quote(240, ".82"))
    entry = experiment.signal(bars)
    assert entry["ts"] == 240 and entry["limit"] == D(".83")
    assert experiment.signal([*bars, quote(300, ".50")]) == entry


def test_gap_resets_peak_and_wide_ask_cannot_trigger():
    assert experiment.signal([quote(0,".92"), quote(600,".75"), quote(660,".80")]) is None
    assert experiment.signal([quote(0,".92"), quote(60,".75"), (120,D(".74"),D(".86"))]) is None


def test_buy_and_sell_need_opposite_taker_sides_and_correct_limit_direction():
    trades = [dict(created_time="1970-01-01T00:00:01Z", trade_id="early", count_fp="2",
                   taker_outcome_side="no",yes_price_dollars=".91"),
              dict(created_time="1970-01-01T00:00:05Z", trade_id="sell", count_fp="2",
                   taker_outcome_side="no",yes_price_dollars=".90")]
    with patch.object(experiment,"pages",return_value=trades):
        assert experiment.evidence(None,"T","buy",D(".91"),0,100) is None
        assert experiment.evidence(None,"T","sell",D(".91"),0,100) is None
        assert experiment.evidence(None,"T","sell",D(".89"),0,100)==(5,"sell")


def test_failed_sell_stays_exposed_to_actual_settlement():
    bars = [quote(0,".92"),quote(60,".80"),quote(120,".82"),quote(240,".95")]
    m = dict(ticker="T",close_time="1970-01-01T01:00:00Z",settlement_ts="1970-01-01T01:01:00Z",
             settlement_value_dollars="0.00")
    def prints(client,ticker,side,limit,t,end):
        return (t+5,"buy") if side=="buy" else None
    with patch.object(experiment,"choose_favorite",return_value=(m,D(".8"),-600)), patch.object(experiment,"quotes",return_value=bars):
        row = experiment.replay(None,[m],dict(event_date="1970-01-01",event_time="00:00"),fill=prints)
        assert row["status"]=="settled" and row["exit_reason"]=="target" and D(row["pnl"])==D("-.84")
