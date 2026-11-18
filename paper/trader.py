"""Trader for the tennis dip rule and the NBA late-favourite rule: simulated, or live with `--live`.

Without `--live` it makes public Kalshi reads only and cannot place an order.
Every pass it reads the open markets, Kalshi's live game state, and (for the
few markets with an order) the order book and the trade record.

  tennis dip      favourite at 90c+ before play, a resting 80c bid from the
                  first live pass, filled only when a trade prints below 80c
  tennis dip rel  the same, with the bid 10c below the pre-match price
  nba take        4th quarter or overtime, leader ahead by 6+ with its ask at
                  97-99c: buy at the ask, limited to the size displayed there
  nba rest        the same signal, a bid 1c below the ask instead

WTA and ATP Challenger are the tennis tours the research supports. ATP main
tour and WTA Challenger are simulated too, labelled "shadow": the first lost
in research and the second was never tested. NBA leads under 6 points are
also "shadow": they carried five of the eight losses in research.

With `--live`, every simulated order on a supported tour also goes to the
production Kalshi account as a real order (real money): the tennis bid rests
until the market closes, and the NBA buy is immediate-or-cancel at the ask.
The relative tennis bid, the NBA resting version and every shadow row stay
simulated. New live orders
stop when open exposure would pass --max-exposure, when settled live losses
reach --max-loss, or while a file named STOP exists in the state folder.

State is saved after every pass, so a restart resumes where it stopped.

  .venv/bin/python -m paper.trader run --hours 24
  .venv/bin/python -m paper.trader run --hours 24 --live --contracts 5
  .venv/bin/python -m paper.trader report
"""
import argparse
from collections import defaultdict
from datetime import datetime, timedelta, timezone
import json
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

OUT = ROOT / "data/paper"
TENNIS = {"KXWTAMATCH": "validated", "KXATPCHALLENGERMATCH": "validated",
          "KXATPMATCH": "shadow", "KXWTACHALLENGERMATCH": "shadow"}
NBA = "KXNBAGAME"
SLOW_POLL, NEAR, STALE_QUOTES, SETTLE_POLL = 300, 8, 600, 60
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
    def __init__(self, api, out=OUT, broker=None):
        self.api, self.out, self.broker = api, Path(out), broker
        self.out.mkdir(parents=True, exist_ok=True)
        file = self.out/"state.json"
        self.state = json.loads(file.read_text()) if file.exists() else dict(watch={}, handled=[], orders=[])
        self.handled = set(self.state["handled"])
        self.known, self.refreshed = {}, 0.0

    # ---- reads -------------------------------------------------------------------------------

    def milestones(self, now):
        since = (datetime.fromtimestamp(now, timezone.utc)-timedelta(hours=8)).strftime("%Y-%m-%dT%H:%M:%SZ")
        found, cursor = {}, ""
        while True:
            page = self.api.get("/milestones", limit=200, category="Sports", minimum_start_date=since, cursor=cursor)
            for m in page.get("milestones", []):
                for event in m.get("primary_event_tickers") or []:
                    if event.startswith((*TENNIS, NBA)):
                        found[event] = m["id"]
            cursor = page.get("cursor")
            if not cursor or not page.get("milestones"):
                return found

    def markets(self):
        """The alphabetically first open market of every event, as the research used."""
        events = defaultdict(list)
        for series in (*TENNIS, NBA):
            cursor = ""
            while True:
                page = self.api.get("/markets", status="open", series_ticker=series, limit=1000, cursor=cursor)
                for m in page["markets"]:
                    events[m["event_ticker"]].append(m)
                cursor = page.get("cursor")
                if not cursor:
                    break
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

    def exposure(self):
        """Dollars committed to live orders that have not settled, counting resting orders in full."""
        total = 0.0
        for o in self.state["orders"]:
            live = o.get("live") or {}
            if "order_id" in live and "pnl" not in live:
                total += (live["count"] if o["kind"] == "rest" and o["status"] in ("open", "filled") else live["filled"])*o["price"]/100
        return total

    def realized(self):
        return sum((o.get("live") or {}).get("pnl", 0) for o in self.state["orders"])

    def send(self, o, available=None):
        """Place the real order behind a simulated one, when live trading is on and the limits allow."""
        if not self.broker or o["tier"] != "validated" or o["rule"] not in ("tennis_dip", "nba_take"):   # all else is simulated
            return
        count = self.broker.contracts if available is None else min(self.broker.contracts, int(available))
        cost = count*o["price"]/100
        why = ("STOP file present" if (self.out/"STOP").exists()
               else "loss limit reached" if self.realized() <= -self.broker.max_loss
               else "exposure limit" if self.exposure()+cost > self.broker.max_exposure
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
        return True

    # ---- rules -------------------------------------------------------------------------------

    def tennis(self, markets, live, now):
        watch = self.state["watch"]
        for event, m in markets.items():
            series = m["ticker"].split("-")[0]
            if series not in TENNIS or event in self.handled or event not in live:
                continue
            status = str(live[event].get("status", "")).lower()
            if status in rules.PRE:
                quote = rules.listed(m)
                if quote is None:
                    watch.pop(event, None)             # a wide or one-sided quote restarts the pre-match price
                else:
                    watch[event] = (watch.get(event, [])+[[now, sum(quote)/2]])[-rules.KEEP_QUOTES:]
                continue
            quotes = watch.get(event, [])
            fresh = quotes and now-quotes[-1][0] <= STALE_QUOTES
            pick = rules.favourite(quotes) if status in rules.TENNIS_LIVE and fresh else None
            # Read the book before marking the match handled, so a failed read is retried next pass.
            book = rules.top(self.api.get(f'/markets/{m["ticker"]}/orderbook'))[pick[0]] if pick else None
            watch.pop(event, None)
            self.handled.add(event)
            if status in rules.TENNIS_LIVE and not fresh:
                self.log("skip", event=event, why="start not observed")
            if not pick:
                continue                               # not live, or no stable favourite at 90c+
            side, start = pick
            if book["ask"] is None or book["ask"] <= rules.DIP_BID:
                self.log("skip", event=event, why="price already at the bid", ask=book["ask"])
                continue
            common = dict(tier=TENNIS[series], kind="rest", event=event, ticker=m["ticker"], side=side, start=start,
                          ask_when_placed=book["ask"])
            self.send(self.order(now, rule="tennis_dip", price=rules.DIP_BID, **common))
            near = round(start)-rules.DIP_RELATIVE     # simulated only: a bid set relative to the pre-match price
            if near != rules.DIP_BID and book["ask"] > near:
                self.order(now, rule="tennis_dip_rel", price=near, **common)

    def nba(self, markets, live, now):
        for event, m in markets.items():
            if not m["ticker"].startswith(NBA) or event in self.handled or event not in live:
                continue
            side = rules.nba_leader(event, m["ticker"], live[event])
            if not side:
                continue
            # Small leads lost money in research: they are simulated once as "shadow" and never sent live,
            # and the game stays eligible for a real entry if the lead grows.
            small = abs(int(live[event]["home_points"])-int(live[event]["away_points"])) < rules.NBA_MIN_LEAD
            key = event+":small" if small else event
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

    def fills(self, markets, now):
        """Check the trade record for every resting bid; near the bid every pass, otherwise every few minutes."""
        for o in self.state["orders"]:
            if o["kind"] != "rest" or o["status"] not in ("open", "filled"):
                continue
            m, near = markets.get(o["event"]), o["rule"] == "nba_rest"
            if m and o["status"] == "open" and not near:
                try:
                    ask = 100*float(m["yes_ask_dollars"]) if o["side"] == "yes" else 100-100*float(m["yes_bid_dollars"])
                    near = ask <= o["price"]+NEAR
                except (KeyError, TypeError, ValueError):
                    near = True
            if not (near and o["status"] == "open") and now-o["polled"] < SLOW_POLL:
                continue
            found = rules.prints(self.trades(o["ticker"], o["seen"]-1), o["side"], o["price"], o["seen"])
            o["polled"], o["seen"] = now, found["latest"]
            o["size_through"] = o.get("size_through", 0)+found["size_through"]
            o["size_at"] = o.get("size_at", 0)+found["size_at"]
            o["touched_at"] = o.get("touched_at") or found["touched_at"]
            if self.broker:
                self.refresh(o)
            if o["status"] == "open" and found["filled_at"]:
                o.update(status="filled", filled_at=found["filled_at"])
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
                found = rules.prints(self.trades(o["ticker"], o["seen"]-1), o["side"], o["price"], o["seen"])
                o["size_through"] = o.get("size_through", 0)+found["size_through"]
                if o["status"] == "open" and found["filled_at"]:
                    o.update(status="filled", filled_at=found["filled_at"])
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
        live = self.live(set(markets) | {o["event"] for o in self.state["orders"] if o["status"] in ("open", "filled")})
        self.tennis(markets, live, now)
        self.nba(markets, live, now)
        self.fills(markets, now)
        self.settle(markets, live, now)
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
        if "live" in r:
            l = r["live"]
            print(f'  LIVE: {l["orders"]} real orders, {l["with_fills"]} with fills ({l["contracts_filled"]:.0f} contracts), '
                  f'{l["settled"]} settled: paid ${l["paid"]:.2f}, fees ${l["fees"]:.2f}, profit ${l["pnl"]:+.2f}; '
                  f'simulated fill agreed with the exchange on {l["simulated_fill_agrees"]} of {l["orders"]}')
    return table


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("command", choices=("run", "report"))
    parser.add_argument("--hours", type=float, default=24)
    parser.add_argument("--every", type=int, default=20, help="seconds between passes")
    parser.add_argument("--dir", default=str(OUT), help="where state and the journal are kept")
    parser.add_argument("--live", action="store_true", help="also place real orders on the production account")
    parser.add_argument("--contracts", type=int, default=5, help="contracts per live order")
    parser.add_argument("--max-exposure", type=float, default=60, help="dollars in unsettled live orders")
    parser.add_argument("--max-loss", type=float, default=30, help="settled live loss in dollars that stops new orders")
    args = parser.parse_args()
    if args.command == "report":
        report(args.dir)
        return
    broker = None
    if args.live:
        broker = Broker(args.contracts, args.max_exposure, args.max_loss)
        print(f"LIVE: real orders on the production account, balance ${broker.balance():.2f}; {args.contracts} contracts "
              f"an order, at most ${args.max_exposure:.0f} unsettled, stop after ${args.max_loss:.0f} lost", flush=True)
    trader, stop, passes = Trader(Api(), args.dir, broker), time.time()+args.hours*3600, 0
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
        time.sleep(max(1, args.every-(time.time()-began)))


if __name__ == "__main__":
    main()
