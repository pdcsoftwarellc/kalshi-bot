from research.native_replay import market_scorecard


def test_scorecard_selects_one_premarked_forecast_per_match_and_keeps_open_outcomes_unscored():
    thresholds = dict(min_points=10, max_age_ms=5000, min_confidence=.60, min_edge=.05,
                      net_edge_min=.10, taker_multiplier=1, fee_precision=".01")
    row = dict(fixture="m", ticker="T", ts_ms=100, bid=38, ask=42, bid_size=3, ask_size=1,
               points_observed=20, model_prob=.7, confidence=.5, book_age_ms=100)
    pending = dict(row, fixture="other",ticker="U")
    rows = [dict(row,ts_ms=90,ask_size=0), pending, row, row, dict(row,ts_ms=200,model_prob=.95)]
    card = market_scorecard(rows, {"T":dict(ts_ms=900, yes_cents=100)}, thresholds)
    assert (card["forecasts"],card["eligible_matches"],card["scored_matches"]) == (4,2,1)
    assert abs(card["model_brier"]-.09) < 1e-10
    assert abs(card["market_brier"]-.36) < 1e-10
    assert card["forecast_gates"] == {"missing_fresh_two_sided_depth":1, "low_confidence":3}
    assert market_scorecard(rows,{},thresholds)["model_brier"] is None
