from datetime import datetime, timezone

import pytest

from paper import rules
from paper.trader import Trader, summary

T0 = 1_800_000_000.0


def iso(ts):
    return datetime.fromtimestamp(ts, timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.%fZ")


def trade(ts, yes, count=10):
    return dict(created_time=iso(ts), yes_price_dollars=f"{yes/100:.4f}", no_price_dollars=f"{1-yes/100:.4f}", count_fp=str(count))


def book(yes_bid, yes_ask, size=500):
    return dict(orderbook_fp=dict(yes_dollars=[["0.0100", "9"], [f"{yes_bid/100:.4f}", str(size)]],
                                  no_dollars=[[f"{1-yes_ask/100:.4f}", str(size)]]))


def test_top_of_book_for_both_sides():
    top = rules.top(book(96, 98, 250))
    assert top["yes"] == dict(bid=96, bid_size=250, ask=98, ask_size=250)
    assert top["no"] == dict(bid=2, bid_size=250, ask=4, ask_size=250)
    assert rules.top({})["yes"]["ask"] is None


def test_favourite_needs_a_stable_price_at_ninety():
    steady = [(T0+20*i, 8.0) for i in range(12)]
    assert rules.favourite(steady) == ("no", 92.0)
    assert rules.favourite(steady[:5]) is None                                  # too few quotes
    assert rules.favourite([(t, 85.0) for t, _ in steady]) is None              # favourite below 90c
    assert rules.favourite([(t, 91.0+(i % 2)*5) for i, (t, _) in enumerate(steady)]) is None    # price moving


def test_resting_bid_fills_only_on_a_print_below_it():
    found = rules.prints([trade(T0+5, 80, 7), trade(T0+9, 79, 3), trade(T0-5, 60)], "yes", 80, T0)
    assert (found["touched_at"], found["filled_at"], found["size_through"], found["size_at"]) == (T0+5, T0+9, 3, 7)
    assert rules.prints([trade(T0+5, 81)], "yes", 80, T0)["filled_at"] is None
    assert rules.prints([trade(T0+5, 25)], "no", 80, T0)["filled_at"] == T0+5   # yes 25c is no 75c


def test_nba_leader():
    state = dict(status="inprogress", period=4, home_points=100, away_points=90)
    assert rules.nba_leader("KXNBAGAME-26NOV17CHALAL", "KXNBAGAME-26NOV17CHALAL-CHA", state) == "no"
    assert rules.nba_leader("KXNBAGAME-26NOV17CHALAL", "KXNBAGAME-26NOV17CHALAL-LAL", state) == "yes"
    assert rules.nba_leader("KXNBAGAME-26NOV17CHALAL", "KXNBAGAME-26NOV17CHALAL-CHA", dict(state, period=3)) is None
    assert rules.nba_leader("KXNBAGAME-26NOV17CHALAL", "KXNBAGAME-26NOV17CHALAL-CHA", dict(state, away_points=100)) is None


def test_net_per_contract():
    assert rules.net(80, 1, taker=False) == 20
    assert rules.net(98, 1, taker=True) == pytest.approx(2-.1372)
    assert rules.net(98, 0, taker=True) == pytest.approx(-98.1372)


class FakeApi:
    """Canned responses keyed by path; tests mutate the attributes between passes."""

    def __init__(self):
        self.listing, self.states, self.books, self.tape, self.settled, self.start = {}, {}, {}, {}, {}, {}

    def get(self, path, **params):
        if path == "/milestones":
            return dict(milestones=[dict(id=f"id-{e}", primary_event_tickers=[e], start_date=self.start.get(e))
                                    for e in self.states])
        if path == "/markets":
            return dict(markets=[m for m in self.listing.values() if m["ticker"].startswith(params["series_ticker"]+"-")])
        if path == "/live_data/batch":
            return dict(live_datas=[dict(milestone_id=i, details=self.states[i[3:]]) for i in params["milestone_ids"]])
        if path == "/markets/trades":
            return dict(trades=self.tape.get(params["ticker"], []))
        if path.endswith("/orderbook"):
            return self.books[path.split("/")[2]]
        return dict(market=self.settled.get(path.split("/")[2], {}))


def market(ticker, bid, ask):
    return dict(ticker=ticker, event_ticker=ticker.rsplit("-", 1)[0], yes_bid_dollars=f"{bid/100:.4f}", yes_ask_dollars=f"{ask/100:.4f}")


def test_tennis_dip_from_pre_match_to_settlement(tmp_path):
    event, ticker = "KXWTAMATCH-26NOV18AAABBB", "KXWTAMATCH-26NOV18AAABBB-AAA"
    api = FakeApi()
    api.listing[ticker] = market(ticker, 92, 94)
    api.states[event] = dict(status="not_started")
    trader = Trader(api, tmp_path)
    for i in range(12):
        trader.step(T0+20*i)
    assert not trader.state["orders"]

    api.states[event] = dict(status="live")
    api.books[ticker] = book(91, 93)
    start = T0+260
    trader.step(start)
    order, relative, deep = trader.state["orders"]
    assert (deep["rule"], deep["price"]) == ("tennis_dip_deep", 65)
    assert (relative["rule"], relative["price"]) == ("tennis_dip_rel", 83)
    assert (order["rule"], order["tier"], order["side"], order["price"], order["start"], order["status"]) == \
        ("tennis_dip", "main-draw", "yes", 80, 93.0, "open")

    api.listing[ticker] = market(ticker, 82, 84)                 # near the bid: the tape is read every pass
    api.tape[ticker] = [trade(start+30, 80, 40)]
    trader.step(start+40)
    assert order["status"] == "open" and order["touched_at"] == start+30

    api.tape[ticker].append(trade(start+50, 78, 25))
    trader.step(start+60)
    assert order["status"] == "filled" and order["size_through"] == 25

    del api.listing[ticker]
    api.settled[ticker] = dict(status="finalized", settlement_value_dollars="1.0000", settlement_ts=iso(start+900))
    trader.step(start+1000)
    assert (order["status"], order["payout"], order["net"], order["size"]) == ("settled", 1.0, 20, 25)

    # A restart reads the same state and does not place a second order for the match.
    again = Trader(api, tmp_path)
    again.step(start+1100)
    assert len(again.state["orders"]) == 3
    row = summary(again.state["orders"])["tennis_dip (main-draw)"]
    assert (row["fills"], row["win_rate"], row["net_cents"], row["dollars_at_clip"]) == (1, 1.0, 20, 5.0)


def test_tennis_match_already_live_is_skipped(tmp_path):
    event, ticker = "KXWTAMATCH-26NOV18AAABBB", "KXWTAMATCH-26NOV18AAABBB-AAA"
    api = FakeApi()
    api.listing[ticker] = market(ticker, 92, 94)
    api.states[event] = dict(status="live")
    trader = Trader(api, tmp_path)
    trader.step(T0)
    assert not trader.state["orders"] and event in trader.handled


def test_nba_take_and_rest(tmp_path):
    event, ticker = "KXNBAGAME-26NOV17CHALAL", "KXNBAGAME-26NOV17CHALAL-CHA"
    api = FakeApi()
    api.listing[ticker] = market(ticker, 2, 3)                   # Charlotte (away) trails: the leader is the NO side
    api.states[event] = dict(status="inprogress", period=3, home_points=90, away_points=70, period_remaining_time="1:00")
    api.books[ticker] = book(2, 3, 800)
    trader = Trader(api, tmp_path)
    trader.step(T0)
    assert not trader.state["orders"]                            # third quarter

    api.states[event]["period"] = 4
    trader.step(T0+20)
    take, rest = trader.state["orders"]
    assert (take["rule"], take["side"], take["price"], take["status"], take["size"]) == ("nba_take", "no", 98, "filled", 800)
    assert (rest["rule"], rest["price"], rest["status"]) == ("nba_rest", 97, "open")

    trader.step(T0+40)
    assert len(trader.state["orders"]) == 2                      # one entry per game

    del api.listing[ticker]
    api.settled[ticker] = dict(status="finalized", settlement_value_dollars="0.0000", settlement_ts=iso(T0+600))
    trader.step(T0+700)
    assert (take["status"], take["payout"]) == ("settled", 1.0) and take["net"] == pytest.approx(2-.1372)
    assert rest["status"] == "expired" and "net" not in rest


class FakeBroker:
    contracts, max_exposure, max_loss, rules = 5, 10, 30, ("tennis_dip", "cbb_dip", "nba_take")

    def __init__(self):
        self.sent, self.fills = [], {}

    def buy(self, ticker, side, price, count, rest, post_only=False):
        self.sent.append((ticker, side, price, count, rest))
        order_id = f"o{len(self.sent)}"
        self.fills[order_id] = 0 if rest else count
        return dict(order_id=order_id, fill_count=str(self.fills[order_id]))

    def queue(self, order_id):
        return 40.0

    def order(self, order_id):
        filled = self.fills[order_id]
        return dict(status="executed", filled=filled, remaining=0, cost=filled*.98, fees=.01 if filled else 0)


def test_live_orders_follow_the_validated_rules_and_limits(tmp_path):
    event, ticker = "KXNBAGAME-26NOV17CHALAL", "KXNBAGAME-26NOV17CHALAL-CHA"
    api, broker = FakeApi(), FakeBroker()
    api.listing[ticker] = market(ticker, 2, 3)
    api.states[event] = dict(status="inprogress", period=4, home_points=90, away_points=70)
    api.books[ticker] = book(2, 3, 3)                            # only three contracts displayed at the ask
    trader = Trader(api, tmp_path, broker)
    trader.step(T0)
    take, rest = trader.state["orders"]
    assert broker.sent == [(ticker, "no", 98, 3, False)]         # the resting version stays simulated
    assert take["live"]["filled"] == 3 and "live" not in rest
    assert trader.exposure() == pytest.approx(2.94)

    del api.listing[ticker]
    api.settled[ticker] = dict(status="finalized", settlement_value_dollars="0.0000", settlement_ts=iso(T0+600))
    trader.step(T0+700)
    assert take["live"]["pnl"] == pytest.approx(3-2.94-.01)
    assert summary(trader.state["orders"])["nba_take (validated)"]["live"]["simulated_fill_agrees"] == 1

    # A second game would pass the $10 exposure cap only if the first were still open; a STOP file blocks it anyway.
    (tmp_path/"STOP").touch()
    event2, ticker2 = "KXNBAGAME-26NOV17BOSNYK", "KXNBAGAME-26NOV17BOSNYK-BOS"
    api.listing[ticker2] = market(ticker2, 2, 3)
    api.states[event2] = dict(status="inprogress", period=4, home_points=90, away_points=70)
    api.books[ticker2] = book(2, 3, 800)
    trader.known, trader.refreshed = {}, 0
    trader.step(T0+800)
    assert len(trader.state["orders"]) == 4 and len(broker.sent) == 1


def test_small_nba_leads_are_simulated_only_until_the_lead_grows(tmp_path):
    event, ticker = "KXNBAGAME-26NOV17CHALAL", "KXNBAGAME-26NOV17CHALAL-CHA"
    api, broker = FakeApi(), FakeBroker()
    api.listing[ticker] = market(ticker, 2, 3)
    api.states[event] = dict(status="inprogress", period=4, home_points=90, away_points=87)
    api.books[ticker] = book(2, 3, 800)
    trader = Trader(api, tmp_path, broker)
    trader.step(T0)
    trader.step(T0+20)
    assert [o["tier"] for o in trader.state["orders"]] == ["shadow", "shadow"] and not broker.sent

    api.states[event]["home_points"] = 95
    trader.step(T0+40)
    assert [o["tier"] for o in trader.state["orders"]] == ["shadow", "shadow", "validated", "validated"]
    assert broker.sent == [(ticker, "no", 98, 5, False)]


def test_college_basketball_sends_only_the_80c_bid(tmp_path):
    event, ticker = "KXNCAAMBGAME-26NOV18DUKEUNC", "KXNCAAMBGAME-26NOV18DUKEUNC-DUKE"
    api, broker = FakeApi(), FakeBroker()
    api.listing[ticker] = market(ticker, 93, 94)
    api.states[event] = dict(status="scheduled")
    trader = Trader(api, tmp_path, broker)
    for i in range(12):
        trader.step(T0+20*i)
    api.states[event] = dict(status="inprogress", period=1, home_points=2, away_points=0)
    api.books[ticker] = book(92, 94)
    trader.step(T0+260)
    assert [(o["rule"], o["tier"], o["price"]) for o in trader.state["orders"]] == \
        [("cbb_dip", "validated", 80), ("cbb_dip_rel", "validated", 83), ("cbb_dip_deep", "validated", 65)]
    assert broker.sent == [(ticker, "yes", 80, 5, True)]


def test_dip_bid_rests_on_the_underdogs_market(tmp_path):
    event, fav, dog = "KXWTAMATCH-26NOV18AAABBB", "KXWTAMATCH-26NOV18AAABBB-AAA", "KXWTAMATCH-26NOV18AAABBB-BBB"
    api = FakeApi()
    api.listing[fav], api.listing[dog] = market(fav, 92, 94), market(dog, 6, 8)
    api.states[event] = dict(status="not_started")
    trader = Trader(api, tmp_path)
    for i in range(12):
        trader.step(T0+20*i)
    api.states[event] = dict(status="live")
    api.books[dog] = book(6, 8)                                  # buying NO there costs 100-6 = 94c
    start = T0+260
    trader.step(start)
    order = trader.state["orders"][0]
    assert (order["ticker"], order["side"], order["price"], order["ask_when_placed"]) == (dog, "no", 80, 94)

    api.listing[dog] = market(dog, 19, 22)
    api.tape[dog] = [trade(start+30, 21, 12)]                    # the underdog's YES at 21c is the favourite at 79c
    trader.step(start+40)
    assert order["status"] == "filled" and order["size_through"] == 12

    del api.listing[fav], api.listing[dog]
    api.settled[dog] = dict(status="finalized", settlement_value_dollars="0.0000", settlement_ts=iso(start+900))
    trader.step(start+1000)
    assert (order["status"], order["payout"], order["net"]) == ("settled", 1.0, 20)


def test_only_qualifying_tennis_matches_send_real_orders(tmp_path):
    event, fav, dog = "KXWTAMATCH-26NOV18AAABBB", "KXWTAMATCH-26NOV18AAABBB-AAA", "KXWTAMATCH-26NOV18AAABBB-BBB"
    api, broker = FakeApi(), FakeBroker()
    text = "If A wins the A vs B professional tennis match in the 2026 WTA Tokyo Qualification Round 1 ..."
    api.listing[fav], api.listing[dog] = dict(market(fav, 92, 94), rules_primary=text), dict(market(dog, 6, 8), rules_primary=text)
    api.books[fav], api.books[dog] = book(92, 94), book(7, 9)   # NO on the underdog's book costs 93c, a cent cheaper
    api.states[event] = dict(status="not_started")
    trader = Trader(api, tmp_path, broker)
    for i in range(12):
        trader.step(T0+20*i)
    (early,) = trader.state["orders"]                            # the simulated pre-match buy, once
    assert (early["rule"], early["ticker"], early["side"], early["price"], early["status"]) == \
        ("tennis_prematch", dog, "no", 93, "filled")
    assert not broker.sent

    api.states[event] = dict(status="live")
    trader.step(T0+260)
    dip = trader.state["orders"][1]
    assert (dip["rule"], dip["tier"], dip["round"]) == ("tennis_dip", "validated", "qualifying, first round")
    assert broker.sent == [(dog, "no", 80, 5, True)]             # only the 80c bid is real
    assert [o["rule"] for o in trader.state["orders"]][1:] == ["tennis_dip", "tennis_dip_rel", "tennis_dip_deep", "tennis_dip_85"]
    api.listing[dog] = dict(market(dog, 17, 19), rules_primary=text)      # near the bid: the exchange is read each pass
    broker.order = lambda order_id: dict(status="resting", filled=0, remaining=5, cost=0, fees=0)
    trader.step(T0+280)
    assert dip["live"]["queue_first"] == 40
    del broker.order

    api.listing.clear()
    api.settled[dog] = dict(status="finalized", settlement_value_dollars="0.0000", settlement_ts=iso(T0+900))
    trader.step(T0+1000)
    from paper.trader import gaps
    assert gaps(trader.state["orders"])[("tennis, first-round qualifying", "all weeks")]["gap_cents"] == 7.0


def test_nba_stays_simulated_unless_named_as_a_live_rule(tmp_path):
    event, ticker = "KXNBAGAME-26NOV17CHALAL", "KXNBAGAME-26NOV17CHALAL-CHA"
    api, broker = FakeApi(), FakeBroker()
    broker.rules = ("tennis_dip",)
    api.listing[ticker] = market(ticker, 2, 3)
    api.states[event] = dict(status="inprogress", period=4, home_points=90, away_points=70)
    api.books[ticker] = book(2, 3, 800)
    trader = Trader(api, tmp_path, broker)
    trader.step(T0)
    assert len(trader.state["orders"]) == 2 and not broker.sent


def test_only_first_round_qualifying_counts():
    stage = lambda text: rules.qualifying(dict(rules_primary=f"... in the 2026 ATP Challenger Kobe {text} after a ball ..."))
    assert stage("Qualification Round 1") == "first" and stage("Qualifying") == "first"
    assert stage("Qualifiers Round Robin") is None               # Billie Jean King Cup team ties
    assert stage("Qualification Round 2") == "later" and stage("Qualification Final") == "later"
    assert stage("Round Of 16") is None and rules.qualifying({}) is None


def test_first_round_qualifying_is_real_on_a_shadow_tour(tmp_path):
    event, ticker = "KXATPMATCH-26NOV18AAABBB", "KXATPMATCH-26NOV18AAABBB-AAA"
    api, broker = FakeApi(), FakeBroker()
    text = "... in the 2026 ATP Paris Qualification Round 1 after a ball ..."
    api.listing[ticker] = dict(market(ticker, 92, 94), rules_primary=text)
    api.books[ticker] = book(92, 94)
    api.states[event] = dict(status="not_started")
    trader = Trader(api, tmp_path, broker)
    for i in range(12):
        trader.step(T0+20*i)
    api.states[event] = dict(status="live")
    trader.step(T0+260)
    assert broker.sent == [(ticker, "yes", 80, 5, True)]


def test_basket_buys_a_basketball_favourite_just_before_the_start(tmp_path):
    event, ticker = "KXNCAAWBGAME-26NOV18UCONNLSU", "KXNCAAWBGAME-26NOV18UCONNLSU-LSU"
    api, broker = FakeApi(), FakeBroker()
    api.listing[ticker] = market(ticker, 6, 8)                   # the listed team is the underdog: favourite is NO at 93c
    api.books[ticker] = book(6, 8, 400)
    api.states[event] = dict(status="scheduled")
    api.start[event] = iso(T0+1000)
    trader = Trader(api, tmp_path, broker)
    for i in range(12):
        trader.step(T0+20*i)
    assert not trader.state["orders"]                            # more than ten minutes before the start
    trader.step(T0+500)
    (buy,) = trader.state["orders"]
    assert (buy["rule"], buy["tier"], buy["side"], buy["price"], buy["band"], buy["status"], buy["size"]) == \
        ("basket_prematch", "wcbb", "no", 94, "93-96", "filled", 400)

    api.states[event] = dict(status="inprogress", period=1, home_points=0, away_points=0)
    trader.step(T0+1010)
    assert [(o["rule"], o["tier"]) for o in trader.state["orders"]][1:] == \
        [("wcbb_dip", "validated"), ("wcbb_dip_rel", "validated"), ("wcbb_dip_deep", "validated")]
    assert not broker.sent                                       # this broker was not given the women's rule


def test_womens_college_basketball_sends_the_80c_bid_and_volleyball_stays_simulated(tmp_path):
    api, broker = FakeApi(), FakeBroker()
    broker.rules = ("tennis_dip", "cbb_dip", "wcbb_dip")
    tickers = ["KXNCAAWBGAME-26NOV18UCONNLSU-UCONN", "KXNCAAWVMATCH-26NOV18NEBWIS-NEB"]
    for ticker in tickers:
        api.listing[ticker] = market(ticker, 93, 94)
        api.states[ticker.rsplit("-", 1)[0]] = dict(status="scheduled")
        api.books[ticker] = book(92, 94)
    trader = Trader(api, tmp_path, broker)
    for i in range(12):
        trader.step(T0+20*i)
    for ticker in tickers:
        api.states[ticker.rsplit("-", 1)[0]] = dict(status="inprogress", period=1, home_points=2, away_points=0)
    trader.step(T0+260)
    assert {(o["rule"], o["tier"]) for o in trader.state["orders"]} >= {("wcbb_dip", "validated"), ("vball_dip", "simulated")}
    assert broker.sent == [(tickers[0], "yes", 80, 5, True)]


def test_mild_college_basketball_underdog_is_bought_on_paper_before_the_start(tmp_path):
    event, ticker = "KXNCAAMBGAME-26NOV18DUKEUNC", "KXNCAAMBGAME-26NOV18DUKEUNC-DUKE"
    api, broker = FakeApi(), FakeBroker()
    api.listing[ticker] = market(ticker, 56, 58)                 # the other team is the underdog: NO at about 43c
    api.books[ticker] = book(56, 58, 200)
    api.states[event] = dict(status="scheduled")
    api.start[event] = iso(T0+1000)
    trader = Trader(api, tmp_path, broker)
    for i in range(12):
        trader.step(T0+20*i)
    assert not trader.state["orders"]                            # more than ten minutes before the start
    trader.step(T0+500)
    trader.step(T0+520)
    (dog,) = trader.state["orders"]                              # once, and never a real order
    assert (dog["rule"], dog["tier"], dog["side"], dog["price"], dog["status"]) == ("cbb_underdog", "simulated", "no", 44, "filled")
    assert not broker.sent


def test_nba_dip_and_late_rule_do_not_block_each_other(tmp_path):
    event, ticker = "KXNBAGAME-26NOV17CHALAL", "KXNBAGAME-26NOV17CHALAL-CHA"
    api, broker = FakeApi(), FakeBroker()
    api.listing[ticker] = market(ticker, 6, 8)
    api.books[ticker] = book(6, 8, 500)
    api.states[event] = dict(status="scheduled")
    trader = Trader(api, tmp_path, broker)
    for i in range(12):
        trader.step(T0+20*i)
    api.states[event] = dict(status="inprogress", period=1, home_points=2, away_points=0)
    trader.step(T0+260)
    assert [o["rule"] for o in trader.state["orders"]] == ["nba_dip", "nba_dip_rel", "nba_dip_deep"]
    api.states[event] = dict(status="inprogress", period=4, home_points=100, away_points=80)
    api.listing[ticker] = market(ticker, 2, 3)
    api.books[ticker] = book(2, 3, 500)
    trader.step(T0+280)
    assert [o["rule"] for o in trader.state["orders"]][3:] == ["nba_take", "nba_rest"]
