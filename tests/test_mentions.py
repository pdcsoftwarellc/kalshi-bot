from datetime import datetime, timezone

from paper.mentions import Mentions, event_day, family, summary

T0 = datetime(2026, 12, 3, 12, tzinfo=timezone.utc).timestamp()


def iso(ts):
    return datetime.fromtimestamp(ts, timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.%fZ")


class FakeApi:
    def __init__(self):
        self.markets, self.book, self.tape, self.settled = [], {}, [], {}

    def get(self, path, **params):
        if path == "/series":
            return dict(series=[dict(ticker="KXEARNINGSMENTIONKR")])
        if path == "/markets":
            return dict(markets=self.markets)
        if path == "/markets/trades":
            return dict(trades=self.tape)
        if path.endswith("/orderbook"):
            return self.book
        return dict(market=self.settled)


def test_event_day_and_family():
    assert str(event_day("KXEARNINGSMENTIONKR-26DEC03")) == "2026-12-03"
    assert event_day("KXSOMETHING-NEXT") is None
    assert family("KXEARNINGSMENTIONKR-26DEC03-TARIFF") == "earnings calls"
    assert family("KXFEDMENTION-26DEC-VOLA") == "politicians and officials"


def test_mention_market_from_quote_to_settlement(tmp_path):
    ticker = "KXEARNINGSMENTIONKR-26DEC03-TARIFF"
    api = FakeApi()
    api.markets = [dict(ticker=ticker, event_ticker="KXEARNINGSMENTIONKR-26DEC03"),
                   dict(ticker="KXEARNINGSMENTIONKR-27MAR04-TARIFF", event_ticker="KXEARNINGSMENTIONKR-27MAR04")]
    api.book = dict(orderbook_fp=dict(yes_dollars=[["0.4000", "50"]], no_dollars=[["0.5200", "80"]]))   # 8c wide
    book = Mentions(api, tmp_path)
    book.step(T0)                                    # lists the series
    book.step(T0+4)                                  # lists its markets: only the one dated today is queued
    assert list(book.state["queue"]) == [ticker]
    book.step(T0+8)                                  # quotes it
    no, mirror = book.state["orders"]
    assert (no["rule"], no["side"], no["price"], mirror["rule"], mirror["side"], mirror["price"]) == \
        ("mention_no", "no", 53, "mention_mirror", "yes", 41)

    # A buyer lifts YES at 48c: that is NO at 52c, strictly below our 53c bid, so the NO bid fills. The mirror does not.
    api.tape = [dict(created_time=iso(T0+100), yes_price_dollars="0.4800", no_price_dollars="0.5200", count_fp="25")]
    api.settled = dict(status="finalized", settlement_value_dollars="0.0000", settlement_ts=iso(T0+5000))
    book.step(T0+2000)                               # settlement check, which also reads the last prints
    assert (no["status"], no["net"], mirror["status"]) == ("settled", 47, "expired")
    table = summary(book.state["orders"])
    assert table["all families | mention_no"]["fills"] == 1 and table["all families | mention_mirror"]["fills"] == 0


def test_tight_or_one_sided_books_are_retried_then_dropped(tmp_path):
    ticker = "KXEARNINGSMENTIONKR-26DEC03-TARIFF"
    api = FakeApi()
    api.markets = [dict(ticker=ticker, event_ticker="KXEARNINGSMENTIONKR-26DEC03")]
    api.book = dict(orderbook_fp=dict(yes_dollars=[["0.4000", "50"]], no_dollars=[["0.5900", "80"]]))   # 1c wide
    book = Mentions(api, tmp_path)
    for i in range(3):
        book.step(T0+4*i)
    assert not book.state["orders"] and book.state["queue"][ticker]["looks"] == 1
