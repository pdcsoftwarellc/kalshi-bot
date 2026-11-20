"""Trader for the tennis dip rule and the NBA late-favourite rule: simulated, or live with `--live`.

Without `--live` it makes public Kalshi reads only and cannot place an order.
Every pass it reads the open markets, Kalshi's live game state, and (for the
few markets with an order) the order book and the trade record.

  tennis dip       favourite at 90c+ before play, a resting 80c bid from the
                   first live pass, filled only when a trade prints below 80c.
                   The bid rests on the underdog's market (as NO) when there is one.
  tennis dip rel   the same, with the bid 10c below the pre-match price
  tennis dip deep  the same, with the bid at 65c
  tennis dip 85    first-round qualifying only: the bid at 85c
  ... dip 75/70/60 the lower rungs of a ladder, for every dip family
  ... dip jit      the 80c bid placed only once the ask falls to 88c ("just in time"),
                   which ties up far less cash; filled only by later prints. Also a
                   real order in men's college basketball and qualifying tennis
  tennis dip low   first-round qualifying favourite at 85-90c before play: the 80c bid
  tennis prematch  first-round qualifying favourite at 90c+: buy at the ask before play
  nba take         4th quarter or overtime, leader ahead by 6+ with its ask at
                   97-99c: buy at the ask, limited to the size displayed there
  nba rest         the same signal, a bid 1c below the ask instead
  cbb dip (...)    men's college basketball, the same three dip bids as tennis
  wcbb dip (...)   women's college basketball, the same three dip bids
  nba / vball dip  the same on NBA pre-game favourites and women's college volleyball
  cbb underdog     men's college basketball underdog priced 35-50c: buy at the ask
                   within ten minutes of the scheduled start
  basket prematch  the nightly basket: buy every 90c+ basketball favourite at the
                   ask within ten minutes of the scheduled start

Tiers. "validated" rows are the ones the research supports: first-round
qualifying matches on every tennis tour, men's and women's college basketball,
and NBA leads of 6+. "later-qualifying" and "main-draw" are the other rounds on WTA
and ATP Challenger. "simulated" marks families that are never sent live
(NBA dips, college volleyball, college basketball underdogs). Basket rows carry the family as tier. "shadow" is everything the research did not support or
did not test: other rounds on ATP main tour and WTA Challenger, NBA leads
under 6.

With `--live`, a validated row of a rule named in --live-rules (default: the
80c dip bid in tennis and men's and women's college basketball) is also sent to the production
Kalshi account as a real order
(real money). Everything else stays simulated. New live orders stop when open
exposure would pass --max-exposure, when settled live losses reach --max-loss,
or while a file named STOP exists in the state folder.

Just-in-time bids need the quote quickly. With credentials present the trader
also follows those markets on Kalshi's live price stream (paper/stream.py)
between passes and acts within about a second; if the stream is down, or run
with --no-stream, the pass-by-pass polling does the same job more slowly.

State is saved after every pass, so a restart resumes where it stopped.

  .venv/bin/python -m paper.trader run --hours 24
  .venv/bin/python -m paper.trader run --hours 336 --live --contracts 3
  .venv/bin/python -m paper.trader report
"""
import argparse
from collections import defaultdict
from datetime import datetime, timedelta, timezone
import json
import math
from pathlib import Path
import random
from statistics import mean, median
import sys
import time

import httpx

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import config
from paper import rules
from paper.broker import Broker, BrokerError
from paper.stream import Stream

OUT = ROOT / "data/paper"
# Series that get dip bids: (rule prefix, tier). Only "validated" rows of a rule in --live-rules are sent live.
DIPS = {"KXWTAMATCH": ("tennis", "validated"), "KXATPCHALLENGERMATCH": ("tennis", "validated"),
        "KXATPMATCH": ("tennis", "shadow"), "KXWTACHALLENGERMATCH": ("tennis", "shadow"),
        "KXNCAAMBGAME": ("cbb", "validated"),
        "KXNCAAWBGAME": ("wcbb", "validated"), "KXNBAGAME": ("nba", "simulated"),
        "KXNCAAWVMATCH": ("vball", "simulated")}
TRIAL_RULES = ("tennis_low_rel10", "tennis_low_rel20", "wcbb_low_rel15", "cbb_underdog")   # real at the smallest size, on their own budget
OWN_BUDGET = ("cbb_underdog",)                  # trial rules that fire often enough to crowd the others: capped apart
LIVE_RULES = ("tennis_dip", "cbb_dip", "wcbb_dip", "tennis_dip_jit", "cbb_dip_jit", *TRIAL_RULES)       # sent as real orders unless --live-rules says otherwise


def budget_of(rule):
    """Which exposure cap a rule draws on: the proven rules', the trial rules', or the underdog rule's own."""
    return "underdog" if rule in OWN_BUDGET else "trial" if rule in TRIAL_RULES else "proven"


LADDER = (75, 70, 60)                         # simulated lower rungs; 65c is the "deep" row
WINDOW, SIZED, SHARE = 1200, 300, .25         # fill window after the first print; order size and share of prints for sized returns
UNDERDOG = (35, 50)                             # men's college basketball underdogs bought on paper before the start
LOW = {"tennis": (10, 20), "wcbb": (15,)}       # trial rules: cents below the pre-match price, for 80-90c favourites
NBA = "KXNBAGAME"
BASKET, BASKET_LEAD = ("cbb", "wcbb", "nba"), 600     # families bought before the start, and how long before
SLOW_POLL, NEAR, STALE_QUOTES, SETTLE_POLL, HORIZON = 300, 8, 600, 60, 12*3600
CLIP = 100                                    # contracts per trade for the dollar figures


class Api:
    """Rate-limited public reads."""

    def __init__(self, per_second=3):
        self.client, self.gap, self.last = httpx.Client(timeout=30), 1/per_second, 0.0

    def get(self, path, **params):
        for attempt in range(8):
            time.sleep(max(0, self.last+self.gap-time.time()))
            self.last = time.time()
            try:
                r = self.client.get(config.KALSHI_PROD_BASE+path, params=params)
            except httpx.RequestError:
                time.sleep(1+attempt)
                continue
            if r.status_code == 429 or r.status_code >= 500:
                time.sleep(.5+attempt)
                continue
            r.raise_for_status()
            return r.json()
        raise RuntimeError(f"{path}: no response after retries")


def stamp(text):
    return datetime.fromisoformat(text.replace("Z", "+00:00")).timestamp()


class Trader:
    def __init__(self, api, out=OUT, broker=None, stream=None):
        self.api, self.out, self.broker, self.stream = api, Path(out), broker, stream
        self.out.mkdir(parents=True, exist_ok=True)
        file = self.out/"state.json"
        self.state = json.loads(file.read_text()) if file.exists() else dict(watch={}, handled=[], orders=[])
        self.handled = set(self.state["handled"])
        self.known, self.refreshed, self.pairs, self.by_ticker, self.starts = {}, 0.0, {}, {}, {}

    # ---- reads -------------------------------------------------------------------------------

    def milestones(self, now):
        since = (datetime.fromtimestamp(now, timezone.utc)-timedelta(hours=8)).strftime("%Y-%m-%dT%H:%M:%SZ")
        found, cursor = {}, ""
        while True:
            page = self.api.get("/milestones", limit=200, category="Sports", minimum_start_date=since, cursor=cursor)
            for m in page.get("milestones", []):
                for event in m.get("primary_event_tickers") or []:
                    if event.startswith((*DIPS, NBA)):
                        found[event] = m["id"]
                        try:
                            self.starts[event] = stamp(m["start_date"])
                        except (KeyError, TypeError, ValueError, AttributeError):
                            pass
            cursor = page.get("cursor")
            if not cursor or not page.get("milestones"):
                return found

    def markets(self):
        """The alphabetically first open market of every event, as the research used."""
        events = defaultdict(list)
        for series in dict.fromkeys((*DIPS, NBA)):
            cursor = ""
            while True:
                page = self.api.get("/markets", status="open", series_ticker=series, limit=1000, cursor=cursor)
                for m in page["markets"]:
                    events[m["event_ticker"]].append(m)
                cursor = page.get("cursor")
                if not cursor:
                    break
        self.pairs = dict(events)
        self.by_ticker = {m["ticker"]: m for group in events.values() for m in group}
        return {event: min(group, key=lambda m: m["ticker"]) for event, group in events.items()}

    def live(self, events):
        ids, out = {self.known[e]: e for e in events if e in self.known}, {}
        for i in range(0, len(ids), 40):
            chunk = list(ids)[i:i+40]
            for row in self.api.get("/live_data/batch", milestone_ids=chunk).get("live_datas", []):
                out[ids[row["milestone_id"]]] = row.get("details") or {}
        return out

    def trades(self, ticker, since):
        rows, cursor = [], ""
        while True:
            page = self.api.get("/markets/trades", ticker=ticker, min_ts=int(since), limit=1000, cursor=cursor)
            rows += page.get("trades", [])
            cursor = page.get("cursor")
            if not cursor or not page.get("trades"):
                return rows

    # ---- bookkeeping -------------------------------------------------------------------------

    def log(self, kind, **fields):
        with (self.out/"journal.jsonl").open("a") as out:
            out.write(json.dumps(dict(kind=kind, **fields), separators=(",", ":"))+"\n")
        print(f'{datetime.now(timezone.utc):%H:%M:%S} {kind} '+" ".join(f"{k}={v}" for k, v in fields.items()), flush=True)

    def save(self):
        self.state["handled"] = sorted(self.handled)
        temp = self.out/"state.json.tmp"
        temp.write_text(json.dumps(self.state, indent=1))
        temp.replace(self.out/"state.json")

    def order(self, now, **fields):
        row = dict(id=len(self.state["orders"])+1, placed=now, seen=now, polled=now, status="open", **fields)
        self.state["orders"].append(row)
        self.log("order", **{k: row[k] for k in ("id", "rule", "tier", "event", "side", "price", "status")})
        return row

    # ---- live orders -------------------------------------------------------------------------

    def exposure(self, budget="proven"):
        """Dollars committed to unsettled live orders drawing on one budget, resting orders in full."""
        total = 0.0
        for o in self.state["orders"]:
            live = o.get("live") or {}
            if "order_id" in live and "pnl" not in live and budget_of(o["rule"]) == budget:
                total += (live["count"] if o["kind"] == "rest" and o["status"] in ("open", "filled") else live["filled"])*o["price"]/100
        return total

    def realized(self):
        return sum((o.get("live") or {}).get("pnl", 0) for o in self.state["orders"])

    def send(self, o, available=None):
        """Place the real order behind a simulated one, when live trading is on and the limits allow."""
        budget = budget_of(o["rule"])                   # unproven rules: real, but they cannot use the proven rules' cash
        trial = budget != "proven"
        if not self.broker or o["rule"] not in self.broker.rules or not (trial or o["tier"] == "validated"):
            return                                     # all else is simulated
        count = self.broker.contracts if available is None else min(self.broker.contracts, int(available))
        cost = count*o["price"]/100
        room = dict(proven=self.broker.max_exposure, trial=self.broker.max_trial, underdog=self.broker.max_underdog)[budget]
        why = ("STOP file present" if (self.out/"STOP").exists()
               else "loss limit reached" if self.realized() <= -self.broker.max_loss
               else "exposure limit" if self.exposure(budget)+cost > room
               else "nothing displayed" if count < 1 else None)
        if why:
            self.log("live_skip", id=o["id"], event=o["event"], why=why)
            return
        try:
            sent = self.broker.buy(o["ticker"], o["side"], o["price"], count, rest=o["kind"] == "rest")
        except (BrokerError, httpx.HTTPError, KeyError, ValueError) as exc:
            o["live"] = dict(error=str(exc)[:200])
            self.log("live_error", id=o["id"], event=o["event"], error=o["live"]["error"])
            return
        o["live"] = dict(order_id=sent["order_id"], count=count, filled=float(sent.get("fill_count") or 0))
        self.log("live_order", id=o["id"], event=o["event"], side=o["side"], price=o["price"], count=count,
                 filled=o["live"]["filled"])

    def refresh(self, o):
        """Read a live order's fills from the exchange. Returns False when the read failed."""
        live = o.get("live") or {}
        if "order_id" not in live:
            return True
        try:
            seen = self.broker.order(live["order_id"])
        except (BrokerError, httpx.HTTPError, KeyError, ValueError) as exc:
            self.log("live_error", id=o["id"], event=o["event"], error=str(exc)[:200])
            return False
        if seen["filled"] > live["filled"]:
            self.log("live_fill", id=o["id"], event=o["event"], filled=seen["filled"], cost=seen["cost"])
        live.update(filled=seen["filled"], cost=seen["cost"], fees=seen["fees"], exchange_status=seen["status"])
        if seen["status"] == "resting" and seen["filled"] < live["count"]:
            try:                                       # how many contracts sit ahead of us: the simulation cannot know this
                live["queue"] = self.broker.queue(live["order_id"])
                live.setdefault("queue_first", live["queue"])
            except (BrokerError, httpx.HTTPError, KeyError, ValueError):
                pass
        return True

    # ---- rules -------------------------------------------------------------------------------

    def dips(self, markets, live, now):
        watch = self.state["watch"]
        for event, m in markets.items():
            series = m["ticker"].split("-")[0]
            if series not in DIPS or event in self.handled or event not in live:
                continue
            status = str(live[event].get("status", "")).lower()
            if status in rules.PRE:
                quote = rules.listed(m)
                if quote is None:
                    watch.pop(event, None)             # a wide or one-sided quote restarts the pre-match price
                else:
                    watch[event] = (watch.get(event, [])+[[now, sum(quote)/2]])[-rules.KEEP_QUOTES:]
                    self.prematch(event, m, now)
                continue
            quotes = watch.get(event, [])
            fresh = quotes and now-quotes[-1][0] <= STALE_QUOTES
            pick = rules.favourite(quotes) if status in rules.DIP_LIVE and fresh else None
            low = ()                                   # trial rules: favourites at 80-90c, bids set below the price
            if not pick and status in rules.DIP_LIVE and fresh:
                family = DIPS[series][0]
                low = LOW.get(family, ()) if family != "tennis" or rules.qualifying(m) == "first" else ()
                pick = rules.favourite(quotes, rules.DIP_FLOOR) if low else None
            # Buy the favourite as NO on the underdog's market when there is one: it carries most of the
            # trading and prints the dip first. The pre-match price still comes from the first-listed market.
            target, side = m, pick and pick[0]
            others = [x for x in self.pairs.get(event, []) if x["ticker"] != m["ticker"]]
            if side == "yes" and len(others) == 1:
                target, side = others[0], "no"
            # Read the book before marking the match handled, so a failed read is retried next pass.
            book = rules.top(self.api.get(f'/markets/{target["ticker"]}/orderbook'))[side] if pick else None
            watch.pop(event, None)
            self.handled.add(event)
            if status in rules.DIP_LIVE and not fresh:
                self.log("skip", event=event, why="start not observed")
            if not pick:
                continue                               # not live, or no stable favourite at 90c+
            start = pick[1]
            prefix, tier = DIPS[series]
            stage = rules.qualifying(m) if prefix == "tennis" else None
            where = dict(kind="rest", event=event, ticker=target["ticker"], side=side, start=start,
                         ask_when_placed=book["ask"], round=f"qualifying, {stage} round" if stage else "main draw")
            if low:
                bids = {f"{prefix}_low_rel{depth}": math.floor(start)-depth for depth in low}
                if prefix == "tennis" and start >= rules.DIP_LOW:
                    bids["tennis_dip_low"] = rules.DIP_BID
                for rule, price in bids.items():
                    if book["ask"] is not None and book["ask"] > price:
                        self.send(self.order(now, rule=rule, price=price, tier="simulated", **where))   # real only for trial rules
                continue
            if book["ask"] is None or book["ask"] <= rules.DIP_BID:
                self.log("skip", event=event, why="price already at the bid", ask=book["ask"])
                continue
            if stage == "first":
                tier = "validated"                     # first-round qualifying is supported on every tour
            elif prefix == "tennis" and tier == "validated":
                tier = "later-qualifying" if stage else "main-draw"
            common = dict(tier=tier, **where)
            self.send(self.order(now, rule=f"{prefix}_dip", price=rules.DIP_BID, **common))
            near = math.floor(start)-rules.DIP_RELATIVE     # simulated only: a bid set relative to the pre-match price
            if near != rules.DIP_BID and book["ask"] > near:
                self.order(now, rule=f"{prefix}_dip_rel", price=near, **common)
            self.order(now, rule=f"{prefix}_dip_deep", price=rules.DIP_DEEP, **common)   # simulated only: a second, deeper bid
            for rung in LADDER:                        # simulated only: the rest of a ladder
                self.order(now, rule=f"{prefix}_dip_{rung}", price=rung, **common)
            if stage == "first" and book["ask"] > rules.DIP_HIGH:
                self.order(now, rule="tennis_dip_85", price=rules.DIP_HIGH, **common)   # simulated only: fills more often

    def prematch(self, event, m, now):
        """Simulated only: buy a 90c+ favourite before play, on the cheaper of the two books.

        Tennis: first-round qualifying, as soon as the favourite is established. Basketball (the nightly
        basket): every game, within ten minutes of the scheduled start.
        """
        prefix, key = DIPS[m["ticker"].split("-")[0]][0], event+":pre"
        if key in self.handled:
            return
        if prefix == "tennis":
            if rules.qualifying(m) != "first":
                return
            rule, tier = "tennis_prematch", "validated"
        elif prefix in BASKET:
            if event not in self.starts or now < self.starts[event]-BASKET_LEAD:
                return
            rule, tier = "basket_prematch", prefix
            if prefix == "cbb":
                self.underdog(event, m, now)
        else:
            return
        pick = rules.favourite(self.state["watch"][event])
        if not pick or (prefix != "tennis" and pick[1] >= 99):
            return
        offers = [(m["ticker"], pick[0])]
        others = [x for x in self.pairs.get(event, []) if x["ticker"] != m["ticker"]]
        if len(others) == 1:
            offers.append((others[0]["ticker"], "no" if pick[0] == "yes" else "yes"))
        books = [(rules.top(self.api.get(f"/markets/{ticker}/orderbook"))[side], ticker, side) for ticker, side in offers]
        books = [b for b in books if b[0]["ask"] is not None and b[0]["ask"] < 100]
        self.handled.add(key)
        if not books:
            return
        book, ticker, side = min(books, key=lambda b: b[0]["ask"])
        if prefix != "tennis" and (book["ask"] >= 99 or book["ask"] > pick[1]+2):
            return                                     # nothing sensible on offer
        band = "90-93" if pick[1] < 93 else "93-96" if pick[1] < 96 else "96-99"
        o = self.order(now, rule=rule, tier=tier, kind="take", event=event, ticker=ticker, side=side,
                       price=book["ask"], start=pick[1], band=band,
                       **(dict(round="qualifying") if prefix == "tennis" else {}))
        o.update(status="filled", filled_at=now, size=book["ask_size"])

    def underdog(self, event, m, now):
        """A trial rule: buy a men's college basketball underdog priced 35-50c at the ask, just before the start.

        research/college_sports.py saw these win more often than their price, but not by enough to pass.
        """
        key, quotes = event+":dog", self.state["watch"][event][-rules.KEEP_QUOTES:]
        if key in self.handled or len(quotes) < rules.MIN_QUOTES:
            return
        mid = median(q[1] for q in quotes)
        side, price = ("yes", mid) if mid <= 50 else ("no", 100-mid)
        if UNDERDOG[0] <= price <= UNDERDOG[1]:
            book = rules.top(self.api.get(f'/markets/{m["ticker"]}/orderbook'))[side]
            if book["ask"] is not None and book["ask"] <= price+2:
                o = self.order(now, rule="cbb_underdog", tier="simulated", kind="take", event=event, ticker=m["ticker"],
                               side=side, price=book["ask"], start=price)
                o.update(status="filled", filled_at=now, size=book["ask_size"])
                self.send(o, available=book["ask_size"])
        self.handled.add(key)

    def nba(self, markets, live, now):
        for event, m in markets.items():
            if not m["ticker"].startswith(NBA) or event not in live:
                continue
            side = rules.nba_leader(event, m["ticker"], live[event])
            if not side:
                continue
            # Small leads lost money in research: they are simulated once as "shadow" and never sent live,
            # and the game stays eligible for a real entry if the lead grows.
            small = abs(int(live[event]["home_points"])-int(live[event]["away_points"])) < rules.NBA_MIN_LEAD
            key = event+(":small" if small else ":late")
            if key in self.handled:
                continue
            try:                                       # the cached listing only decides whether a book read is worth it
                rough = 100*float(m["yes_ask_dollars"]) if side == "yes" else 100-100*float(m["yes_bid_dollars"])
            except (KeyError, TypeError, ValueError):
                continue
            if rough < rules.NBA_BAND[0]-2:
                continue
            book = rules.top(self.api.get(f'/markets/{m["ticker"]}/orderbook'))[side]
            if book["ask"] is None or not rules.NBA_BAND[0] <= book["ask"] <= rules.NBA_BAND[1]:
                continue
            s = live[event]
            seen = dict(period=s.get("period"), clock=s.get("period_remaining_time"),
                        score=f'{s.get("away_points")}-{s.get("home_points")}', ask_size=book["ask_size"], bid=book["bid"])
            self.handled.add(key)
            common = dict(tier="shadow" if small else "validated", event=event, ticker=m["ticker"], side=side, **seen)
            take = self.order(now, rule="nba_take", kind="take", price=book["ask"], **common)
            take.update(status="filled", filled_at=now, size=book["ask_size"])
            self.send(take, available=book["ask_size"])
            self.order(now, rule="nba_rest", kind="rest", price=book["ask"]-1, **common)

    def near(self, o, ask, now, by):
        """The ask has come within reach of a resting bid: note it and place the just-in-time version of the bid.

        The just-in-time bid is real as well in men's college basketball and qualifying tennis: a second
        bid behind our own resting one, to measure the queue a late bid meets.
        """
        o.update(near_at=now, near_by=by)
        if o["rule"].endswith("_dip"):
            extra = {k: o[k] for k in ("tier", "kind", "event", "ticker", "side", "price", "start", "round") if k in o}
            self.send(self.order(now, rule=o["rule"]+"_jit", ask_when_placed=ask, trigger=by, **extra))

    def waiting(self):
        """Resting dip bids whose ask has not yet come near: the ones a just-in-time bid is waiting on."""
        return [o for o in self.state["orders"]
                if o["status"] == "open" and o["rule"].endswith("_dip") and "near_at" not in o]

    def watch(self, until, pause=.25):
        """Between passes, act on the live stream's quotes. Without a stream this only waits."""
        while True:
            fired = False
            for o in self.waiting() if self.stream else ():
                quote = self.stream.quote(o["ticker"])
                if not quote:
                    continue
                ask = quote[1] if o["side"] == "yes" else 100-quote[0]
                if 0 < ask < 100 and ask <= o["price"]+NEAR:
                    self.near(o, ask, time.time(), "stream")
                    fired = True
            if fired:
                self.save()
            if time.time() >= until:
                return
            time.sleep(max(0, min(pause, until-time.time())))

    def apply(self, o, trades):
        """Update one resting bid from the trade record. Returns what the new trades showed."""
        since = o["seen"]
        found = rules.prints(trades, o["side"], o["price"], since)
        o["seen"] = found["latest"]
        o["size_through"] = o.get("size_through", 0)+found["size_through"]
        o["size_at"] = o.get("size_at", 0)+found["size_at"]
        o["touched_at"] = o.get("touched_at") or found["touched_at"]
        first = o.get("filled_at") or found["filled_at"]
        if first:                                      # size in the first minutes of the dip, for the fill-window test
            o["size_window"] = o.get("size_window", 0)+rules.through(trades, o["side"], o["price"], since, first+WINDOW)
        if o["status"] == "open" and found["filled_at"]:
            o.update(status="filled", filled_at=found["filled_at"])
        return found

    def fills(self, markets, now):
        """Check the trade record for every resting bid; near the bid every pass, otherwise every few minutes.

        One read per market serves all of its bids.
        """
        groups = defaultdict(list)
        for o in self.state["orders"]:
            if o["kind"] == "rest" and o["status"] in ("open", "filled"):
                groups[o["ticker"]].append(o)
        for ticker, group in groups.items():
            m, due = self.by_ticker.get(ticker), False
            for o in group:
                near = o["rule"] == "nba_rest"
                if m and o["status"] == "open" and not near:
                    try:
                        ask = 100*float(m["yes_ask_dollars"]) if o["side"] == "yes" else 100-100*float(m["yes_bid_dollars"])
                        near = ask <= o["price"]+NEAR
                        if near:
                            o.setdefault("poll_near_at", now)   # when polling alone would have seen it
                            if "near_at" not in o:
                                self.near(o, ask, now, "poll")
                    except (KeyError, TypeError, ValueError):
                        near = True
                due = due or (near and o["status"] == "open") or now-o["polled"] >= SLOW_POLL
            if not due:
                continue
            trades = self.trades(ticker, min(o["seen"] for o in group)-1)
            for o in group:
                was = o["status"]
                self.apply(o, trades)
                o["polled"] = now
                if self.broker:
                    self.refresh(o)
                if was == "open" and o["status"] == "filled":
                    self.log("fill", id=o["id"], rule=o["rule"], event=o["event"], price=o["price"],
                             size_through=round(o["size_through"], 2))

    def settle(self, markets, live, now):
        for o in self.state["orders"]:
            if o["status"] not in ("open", "filled"):
                continue
            ended = o["event"] not in markets or str(live.get(o["event"], {}).get("status", "")).lower() in rules.DONE
            if not ended or now-o.get("settle_polled", 0) < SETTLE_POLL:
                continue
            o["settle_polled"] = now
            m = self.api.get(f'/markets/{o["ticker"]}').get("market", {})
            try:
                value = float(m["settlement_value_dollars"])
            except (KeyError, TypeError, ValueError):
                continue
            if m.get("status") not in ("finalized", "settled") or not m.get("settlement_ts"):
                continue
            live = o.get("live") or {}
            if "order_id" in live:                     # the exchange's own record decides the live result
                if not self.broker or not self.refresh(o):
                    continue
                payout = value if o["side"] == "yes" else 1-value
                live["pnl"] = live["filled"]*payout-live["cost"]-live["fees"]
            if o["kind"] == "rest":                    # trades that printed between the last check and the close
                self.apply(o, self.trades(o["ticker"], o["seen"]-1))
            o["payout"] = value if o["side"] == "yes" else 1-value
            o["settled_at"] = stamp(m["settlement_ts"])
            if o["status"] == "filled":
                o["net"] = rules.net(o["price"], o["payout"], taker=o["kind"] == "take")
                if o["kind"] == "rest":
                    o["size"] = o["size_through"]
            o["status"] = "settled" if o["status"] == "filled" else "expired"
            self.log("settle", id=o["id"], rule=o["rule"], event=o["event"], status=o["status"],
                     payout=o["payout"], net=o.get("net"), live_pnl=live.get("pnl"))

    def step(self, now=None):
        now = now or time.time()
        if now-self.refreshed > 300:
            self.known, self.refreshed = self.milestones(now), now
        markets = self.markets()
        # Game state is only needed for events about to start or under way, and for events holding an order.
        soon = {e for e, m in markets.items()
                if not m.get("expected_expiration_time") or stamp(m["expected_expiration_time"]) < now+HORIZON}
        live = self.live(soon | {o["event"] for o in self.state["orders"] if o["status"] in ("open", "filled")})
        self.dips(markets, live, now)
        self.nba(markets, live, now)
        self.fills(markets, now)
        self.settle(markets, live, now)
        if self.stream:
            self.stream.watch({o["ticker"] for o in self.waiting()})
        self.save()
        return len(markets), len(live)


# ---- report ----------------------------------------------------------------------------------

def interval(nets, draws=2000):
    if len(nets) < 10:
        return None
    rng = random.Random(7)
    means = sorted(mean(rng.choices(nets, k=len(nets))) for _ in range(draws))
    return [round(means[int(.025*draws)], 2), round(means[int(.975*draws)], 2)]


def summary(orders):
    groups = defaultdict(list)
    for o in orders:
        groups[(o["rule"], o["tier"])].append(o)
    out = {}
    for (rule, tier), rows in sorted(groups.items()):
        done = [o for o in rows if o["status"] in ("settled", "expired")]
        won = [o for o in done if o["status"] == "settled"]
        missed = [o for o in done if o["status"] == "expired"]
        row = dict(orders=len(rows), waiting=sum(o["status"] == "open" for o in rows),
                   filled_unsettled=sum(o["status"] == "filled" for o in rows), finished=len(done), fills=len(won))
        if done:
            row["fill_rate"] = len(won)/len(done)
        if missed:                                     # if unfilled orders win far more often, fills are adversely selected
            row["unfilled_win_rate"] = mean(o["payout"] for o in missed)
        if won:
            nets = [o["net"] for o in won]
            price = mean(o["price"] for o in won)
            row.update(win_rate=mean(o["payout"] for o in won), needed=price/100, mean_price=price,
                       net_cents=mean(nets), ci95=interval(nets), losses=sum(o["payout"] < .5 for o in won),
                       net_with_taker_fee=mean(o["net"]-(0 if o["kind"] == "take" else rules.taker_fee(o["price"])) for o in won),
                       median_size=median(o.get("size") or 0 for o in won),
                       dollars_at_clip=sum(o["net"]*min(CLIP, o.get("size") or 0) for o in won)/100)
        sized = [o for o in won if "size_window" in o]
        if sized:                                      # a larger order that takes a share of the prints, with and without the window
            for key, field in (("sized_return", "size_through"), ("sized_return_window", "size_window")):
                held = [(min(SIZED, SHARE*o[field]), o) for o in sized]
                paid = sum(n*o["price"] for n, o in held)
                row[key] = sum(n*o["net"] for n, o in held)/paid if paid else None
        placed = [o for o in done if o["kind"] == "rest" and "settled_at" in o]
        if placed:
            row["hours_funded"] = mean((o["settled_at"]-o["placed"])/3600 for o in placed)
        sent = [o["live"] for o in rows if "order_id" in (o.get("live") or {})]
        if sent:                                       # real orders, in dollars, from the exchange's own figures
            closed = [l for l in sent if "pnl" in l]
            row["live"] = dict(orders=len(sent), with_fills=sum(l["filled"] > 0 for l in sent),
                               contracts_filled=sum(l["filled"] for l in sent), settled=len(closed),
                               paid=sum(l.get("cost", 0) for l in closed), fees=sum(l.get("fees", 0) for l in closed),
                               pnl=sum(l["pnl"] for l in closed),
                               simulated_fill_agrees=sum((l["filled"] > 0) == (o["status"] in ("filled", "settled"))
                                                         for o in rows for l in [o.get("live") or {}] if "order_id" in l))
        out[f"{rule} ({tier})"] = row
    return out


def gaps(orders):
    """Weekly pre-game gap of every 90c+ favourite the trader priced: settlement value minus pre-game price.

    The dip rules only work while this stays positive, and it shows up long before the fills can.
    """
    weeks = defaultdict(list)
    for o in orders:
        if o["rule"] not in ("tennis_dip", "cbb_dip", "wcbb_dip") or "payout" not in o:
            continue
        group = ("men's college basketball" if o["rule"] == "cbb_dip" else
                 "women's college basketball" if o["rule"] == "wcbb_dip" else
                 "tennis, first-round qualifying" if str(o.get("round", "")).startswith("qualifying, first") else "tennis, other rounds")
        week = datetime.fromtimestamp(o["placed"], timezone.utc).strftime("%G-W%V")
        for key in ((group, week), (group, "all weeks")):
            weeks[key].append((o["start"], o["payout"]))
    return {key: dict(favourites=len(rows), price=mean(s for s, _ in rows), won=mean(p for _, p in rows),
                      gap_cents=mean(100*p-s for s, p in rows)) for key, rows in sorted(weeks.items())}


def report(out=OUT):
    file = Path(out)/"state.json"
    if not file.exists():
        print("No paper trades recorded yet.")
        return {}
    state = json.loads(file.read_text())
    table = summary(state["orders"])
    print(f'{len(state["orders"])} simulated orders; {len(state["watch"])} tennis matches being priced before play')
    for name, r in table.items():
        print(f'\n{name}: orders {r["orders"]}, waiting {r["waiting"]}, filled and not yet settled {r["filled_unsettled"]}, '
              f'finished {r["finished"]}, fills {r["fills"]}')
        if "fill_rate" in r:
            print(f'  fill rate {r["fill_rate"]:.0%}'+(f', unfilled orders won {r["unfilled_win_rate"]:.1%}' if "unfilled_win_rate" in r else ""))
        if "win_rate" in r:
            print(f'  won {r["win_rate"]:.1%} against {r["needed"]:.1%} needed, losses {r["losses"]}, net {r["net_cents"]:+.2f}c per contract '
                  f'(95% interval {r["ci95"]}), {r["net_with_taker_fee"]:+.2f}c with the full taker fee')
            print(f'  median size available {r["median_size"]:.0f} contracts; ${r["dollars_at_clip"]:+.2f} at up to {CLIP} contracts a trade')
        if r.get("sized_return") is not None:
            window = r.get("sized_return_window")
            print(f'  at {SIZED} contracts taking {SHARE:.0%} of the prints: {r["sized_return"]:+.1%}'
                  +(f', {window:+.1%} with fills limited to the first {WINDOW//60} minutes' if window is not None else ""))
        if "hours_funded" in r:
            print(f'  cash reserved for {r["hours_funded"]:.1f} hours per bid on average')
        if "live" in r:
            l = r["live"]
            print(f'  LIVE: {l["orders"]} real orders, {l["with_fills"]} with fills ({l["contracts_filled"]:.0f} contracts), '
                  f'{l["settled"]} settled: paid ${l["paid"]:.2f}, fees ${l["fees"]:.2f}, profit ${l["pnl"]:+.2f}; '
                  f'simulated fill agreed with the exchange on {l["simulated_fill_agrees"]} of {l["orders"]}')
    weekly = gaps(state["orders"])
    if weekly:
        print("\nPre-game gap of 90c+ favourites (history: about +3.5c first-round qualifying, +2.5c college basketball, +1c other)")
        for (group, week), g in weekly.items():
            print(f'  {group:31} {week:9} favourites {g["favourites"]:4}  price {g["price"]:.1f}c  won {g["won"]:.1%}  gap {g["gap_cents"]:+.1f}c')
    leads = [o["poll_near_at"]-o["near_at"] for o in state["orders"] if o.get("near_by") == "stream" and "poll_near_at" in o]
    by = defaultdict(int)
    for o in state["orders"]:
        if "near_by" in o and o["rule"].endswith("_dip"):
            by[o["near_by"]] += 1
    if by:
        print(f'\nJust-in-time triggers: {by["stream"]} from the live stream, {by["poll"]} from polling'
              +(f'; the stream was ahead of polling by a median {median(leads):.0f} seconds ({len(leads)} games)' if leads else ""))
    for name, late in (("Real resting bids", False), ("Real just-in-time bids", True)):
        queued = [o["live"] for o in state["orders"]
                  if "queue_first" in (o.get("live") or {}) and o["rule"].endswith("_jit") == late]
        if queued:
            print(f'\n{name}: median {median(l["queue_first"] for l in queued):.0f} contracts ahead in the queue when first read '
                  f'({len(queued)} orders)')
    return table


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("command", choices=("run", "report"))
    parser.add_argument("--hours", type=float, default=24)
    parser.add_argument("--every", type=int, default=20, help="seconds between passes")
    parser.add_argument("--dir", default=str(OUT), help="where state and the journal are kept")
    parser.add_argument("--live", action="store_true", help="also place real orders on the production account")
    parser.add_argument("--contracts", type=int, default=3, help="contracts per live order")
    parser.add_argument("--max-exposure", type=float, default=60, help="dollars in unsettled live orders")
    parser.add_argument("--max-underdog-exposure", type=float, default=20, help="dollars in unsettled live underdog orders")
    parser.add_argument("--max-trial-exposure", type=float, default=20, help="dollars in unsettled live orders of the trial rules")
    parser.add_argument("--max-loss", type=float, default=30, help="settled live loss in dollars that stops new orders")
    parser.add_argument("--no-stream", action="store_true", help="do not use the live price stream; poll only")
    parser.add_argument("--live-rules", nargs="+", choices=(*LIVE_RULES, "nba_take"), default=list(LIVE_RULES),
                        help="rules allowed to place real orders; everything else is simulated")
    args = parser.parse_args()
    if args.command == "report":
        report(args.dir)
        return
    broker = None
    if args.live:
        broker = Broker(args.contracts, args.max_exposure, args.max_loss, args.live_rules, args.max_trial_exposure, args.max_underdog_exposure)
        print(f"LIVE ({', '.join(args.live_rules)}): real orders on the production account, balance ${broker.balance():.2f}; {args.contracts} contracts "
              f"an order, at most ${args.max_exposure:.0f} unsettled (${args.max_trial_exposure:.0f} more for trial rules, ${args.max_underdog_exposure:.0f} more for underdogs), stop after ${args.max_loss:.0f} lost", flush=True)
    stream = None
    if not args.no_stream and config.KALSHI_API_KEY and config.KALSHI_PRIVATE_KEY_PATH:
        stream = Stream()
        stream.start()
    trader, stop, passes = Trader(Api(), args.dir, broker, stream), time.time()+args.hours*3600, 0
    while time.time() < stop:
        began = time.time()
        try:
            open_events, with_state = trader.step()
            passes += 1
            if passes % 15 == 1:
                orders = trader.state["orders"]
                print(f'{datetime.now(timezone.utc):%H:%M:%S} pass {passes}: open events {open_events}, with live state {with_state}, '
                      f'pricing {len(trader.state["watch"])} matches before play, orders {len(orders)} '
                      f'({sum(o["status"] == "open" for o in orders)} waiting)', flush=True)
        except (httpx.HTTPError, RuntimeError, KeyError, ValueError, TypeError) as exc:
            print(f"{datetime.now(timezone.utc):%H:%M:%S} pass skipped ({type(exc).__name__}: {exc})", flush=True)
        trader.watch(max(time.time()+1, began+args.every))


if __name__ == "__main__":
    main()
