from datetime import date
from decimal import Decimal as D

from research.api_cost import fee, ledger, link_market, qualifying_print, rating_at


def test_results_not_used_during_embargo():
    fixture = dict(first_player_key=1, second_player_key=2, event_status="Finished",
                   event_winner="First Player", event_date="2026-10-01")
    _, counts = rating_at([fixture], date(2026, 10, 7))
    assert not counts
    ratings, counts = rating_at([fixture], date(2026, 10, 8))
    assert ratings["1"] == 1516 and counts["2"] == 1


def test_ambiguous_pair_skipped_and_side_mapping_reversed():
    fixture = dict(first_player_key=1, second_player_key=2, event_first_player="A. Zverev",
                   event_second_player="F. Auger-Aliassime")
    pair = [dict(yes_sub_title="Félix Auger-Aliassime"), dict(yes_sub_title="Alexander Zverev")]
    assert link_market(pair, [fixture])[1:] == ("2", "1")
    assert link_market(pair, [fixture, fixture]) is None


def test_print_needs_side_size_price_and_order_lifetime():
    trade = dict(created_time="1970-01-01T00:00:05Z", taker_outcome_side="no",
                 count_fp="1.00", no_price_dollars=".47", trade_id="hit")
    rejected = [dict(trade, created_time="1970-01-01T00:00:01Z"), dict(trade, count_fp=".99"),
                dict(trade, taker_outcome_side="yes"), dict(trade, no_price_dollars=".49"),
                dict(trade, is_block_trade=True), dict(trade, created_time="1970-01-01T00:05:00Z")]
    assert qualifying_print(rejected, "no", D(".48"), 0) is None
    assert qualifying_print([*rejected, trade], "no", D(".48"), 0) == (5, "hit")


def test_fee_rounding_and_capital_release():
    assert fee(D(".50")) == D(".02")
    row = dict(ts=2, print_ts=5, settlement_ts=400, limit=".75", fee=".02", payout="1", ticker="X")
    orders = [dict(row, ts=1, ticker="A", print_ts=None), row, dict(row, ts=302),
              dict(row, ts=350), dict(row, ts=405, settlement_ts=600, payout="0")]
    fills, summary = ledger(orders, D("1"))
    assert len(fills) == 2 and summary["cash_blocked"] == 2
    assert summary["trading_pnl_dollars"] == -.54
    assert summary["fees_dollars"] == .04


def test_forward_consensus_pairs_each_book_and_removes_overround():
    from research.forward_odds import consensus
    home = {"a": "1.40", "b": "1.5", "c": "1.70", "d": "n/a"}
    away = {"a": "2.80", "b": "3", "c": "2", "d": "2"}
    estimate, books = consensus(home, away)
    assert len(books) == 3 and estimate == D(2)/3
    assert consensus(home, {"a": "2.80", "b": "3"})[0] is None


def test_forward_buy_needs_one_contract_on_opposite_bid():
    from research.forward_odds import book_offers
    book = {"yes_dollars": [["0.49", "0.5"]], "no_dollars": [["0.48", "2"]]}
    assert book_offers(book) == {"yes": D(".52")}
