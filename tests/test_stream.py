from paper.stream import Stream


class Signer:
    def _auth_headers(self, method, path):
        return {}


def test_ticker_messages_become_quotes_only_while_connected():
    stream = Stream(Signer())
    stream.watch({"A"})
    sid = stream.handle(dict(type="subscribed", msg=dict(channel="ticker", sid=7)), None)
    assert sid == 7
    stream.handle(dict(type="ticker", sid=7, msg=dict(market_ticker="A", yes_bid_dollars="0.8500", yes_ask_dollars="0.8700")), sid)
    stream.handle(dict(type="ticker", sid=7, msg=dict(market_ticker="B", yes_bid_dollars="0.1000", yes_ask_dollars="0.1200")), sid)
    assert stream.quote("A") is None                             # not connected: the trader falls back to polling
    stream.connected = True
    assert stream.quote("A")[:2] == (85, 87) and stream.quote("B") is None
    stream.watch(set())
    assert stream.quote("A") is None


def test_an_exchange_error_drops_the_connection():
    stream = Stream(Signer())
    try:
        stream.handle(dict(type="error", msg=dict(code=2, msg="Params required")), 7)
    except RuntimeError as exc:
        assert "Params required" in str(exc)
    else:
        raise AssertionError("expected the error to end the session")


def test_subscription_commands_follow_the_wanted_markets():
    stream = Stream(Signer())
    cmds, have = stream.commands({"A", "B"}, set(), None)
    assert [c["cmd"] for c in cmds] == ["subscribe"] and cmds[0]["params"]["market_tickers"] == ["A", "B"] and have == {"A", "B"}
    cmds, have = stream.commands({"B", "C"}, have, 7)
    assert [(c["params"]["action"], c["params"]["market_tickers"]) for c in cmds] == [("add_markets", ["C"]), ("delete_markets", ["A"])]
    cmds, have = stream.commands(set(), have, 7)
    assert cmds == [dict(cmd="unsubscribe", params=dict(sids=[7]))] and have == set()
    assert stream.commands(set(), set(), None) == ([], set())   # never subscribe with no markets: that means all of them
