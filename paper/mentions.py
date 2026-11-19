"""Resting NO bids in "mention" markets ("will the speaker say X?"): simulated, or real with `--live`.

research/hype_no.py found that takers in these markets mostly buy YES and that
a NO bid resting one cent inside the spread earned about +4c per fill over two
months, while the same order on the YES side lost. research/resting_no.py then
confirmed it on January-September 2026 with this program's own rule (+6.5c per
fill, 995 fills in 233 events), outside sports broadcasts, mostly for words
priced 20-80c in books 10c or wider, and fading in the latest weeks. This
program collects forward evidence.

Kalshi gives no start time for a speech, but most event tickers carry its
date (KXEARNINGSMENTIONKR-26DEC03). So for every open mention market whose
event is dated today or tomorrow (UTC) and whose book is two-sided and at
least 2c wide, once:

  mention_no      a NO bid one cent above the best NO bid
  mention_mirror  the control: a YES bid one cent above the best YES bid

Both rest until the market closes and are held to settlement. A bid counts as
filled only when a trade prints strictly below it. If YES is overpriced the NO
side should win and the mirror lose; a wide book alone would pay both.

With `--live`, the NO bid is also sent to the production Kalshi account as a
real post-only order (real money) when the market is not a sports broadcast
and YES is priced 20-80c, for at most five words per event. The mirror is
never real. New real orders stop at --max-exposure, at --max-loss, or while a
file named STOP exists in the state folder (the same file stops the trader).

It makes one public request every few seconds, so it can run beside the
trader without crowding it. State is saved after every change.

  .venv/bin/python -m paper.mentions run --hours 336
  .venv/bin/python -m paper.mentions run --hours 336 --live --contracts 3
  .venv/bin/python -m paper.mentions report
"""
import argparse
from collections import defaultdict
from datetime import datetime, timedelta, timezone
import json
from pathlib import Path
import re
from statistics import mean
import sys
import time

import httpx

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from paper import rules
from paper.broker import Broker, BrokerError
from paper.trader import Api, interval, stamp

OUT = ROOT / "data/paper"
MONTHS = dict(JAN=1, FEB=2, MAR=3, APR=4, MAY=5, JUN=6, JUL=7, AUG=8, SEP=9, OCT=10, NOV=11, DEC=12)
MIN_SPREAD, FILL_POLL, SETTLE_POLL, SERIES_REFRESH, LOOKS, LOOK_AGAIN = 2, 600, 900, 6*3600, 6, 1800
LIVE_YES, LIVE_PER_EVENT = (20, 80), 5         # real orders: mid-priced words only, a few per speech
FAMILIES = (("earnings calls", r"EARN"), ("Trump", r"TRUMP|DJT|PRESMENTION"),
            ("TV and media", r"SNL|KIMMEL|COLBERT|FALLON|MADDOW|ROGAN|TUCKER|NEWS|SHOW|WEEKNIGHT|FNS|AWARD|ENTMENTION"),
            ("sports broadcasts", r"NBA|NFL|MLB|NCAA|SNF|TNF|MNF|NHL|CFB|WNBA|UFC"))


def family(ticker):
    return next((name for name, pattern in FAMILIES if re.search(pattern, ticker)), "politicians and officials")


def event_day(event):
    """The date carried by an event ticker such as KXFEDMENTION-26DEC10, or None."""
    found = re.search(r"-(\d{2})([A-Z]{3})(\d{2})(?:\D|$)", event)
    try:
        return datetime(2000+int(found.group(1)), MONTHS[found.group(2)], int(found.group(3)), tzinfo=timezone.utc).date()
    except (AttributeError, KeyError, ValueError):
        return None


class Mentions:
    def __init__(self, api, out=OUT, broker=None):
        self.api, self.file, self.broker = api, Path(out)/"mentions.json", broker
        self.file.parent.mkdir(parents=True, exist_ok=True)
        self.state = (json.loads(self.file.read_text()) if self.file.exists()
                      else dict(series=[], listed=0, next=0, queue={}, done=[], orders=[]))
        self.done = set(self.state["done"])

    def save(self):
        self.state["done"] = sorted(self.done)
        temp = self.file.with_suffix(".tmp")
        temp.write_text(json.dumps(self.state, indent=1))
        temp.replace(self.file)

    def log(self, text):
        print(f"{datetime.now(timezone.utc):%H:%M:%S} {text}", flush=True)

    # ---- real orders -------------------------------------------------------------------------

    def send(self, o):
        """Place the real NO bid behind a simulated one, when live trading is on and the limits allow."""
        if (not self.broker or o["family"] == "sports broadcasts"
                or not LIVE_YES[0] <= o["yes_mid"] <= LIVE_YES[1]):
            return
        sent = [x for x in self.state["orders"] if "order_id" in (x.get("live") or {})]
        if sum(x["event"] == o["event"] for x in sent) >= LIVE_PER_EVENT:
            return
        count = self.broker.contracts
        open_cost = sum(x["live"]["count"]*x["price"]/100 for x in sent if "pnl" not in x["live"])
        why = ("STOP file present" if (self.file.parent/"STOP").exists()
               else "loss limit reached" if sum(x["live"].get("pnl", 0) for x in sent) <= -self.broker.max_loss
               else "exposure limit" if open_cost+count*o["price"]/100 > self.broker.max_exposure else None)
        if why:
            self.log(f'live skip {o["ticker"]}: {why}')
            return
        try:
            placed = self.broker.buy(o["ticker"], "no", o["price"], count, rest=True, post_only=True)
        except (BrokerError, httpx.HTTPError, KeyError, ValueError) as exc:
            o["live"] = dict(error=str(exc)[:200])
            self.log(f'live error {o["ticker"]}: {o["live"]["error"]}')
            return
        o["live"] = dict(order_id=placed["order_id"], count=count, filled=float(placed.get("fill_count") or 0))
        self.log(f'LIVE order {o["ticker"]} NO bid {o["price"]}c x{count}')

    def refresh(self, o):
        """Read a real order's fills from the exchange. Returns False when the read failed."""
        live = o.get("live") or {}
        if "order_id" not in live:
            return True
        if not self.broker:
            return False
        try:
            seen = self.broker.order(live["order_id"])
        except (BrokerError, httpx.HTTPError, KeyError, ValueError) as exc:
            self.log(f'live error {o["ticker"]}: {str(exc)[:200]}')
            return False
        if seen["filled"] > live["filled"]:
            self.log(f'LIVE fill {o["ticker"]} {seen["filled"]:.0f} contracts, paid ${seen["cost"]:.2f}')
        live.update(filled=seen["filled"], cost=seen["cost"], fees=seen["fees"], exchange_status=seen["status"])
        return True

    # ---- one request per call ----------------------------------------------------------------

    def sweep(self, now):
        """List the open markets of the next mention series and queue the ones whose event is today or tomorrow."""
        if not self.state["series"] or now-self.state["listed"] > SERIES_REFRESH:
            rows = self.api.get("/series", category="Mentions").get("series", [])
            self.state.update(series=sorted(s["ticker"] for s in rows), listed=now, next=0)
            return
        series = self.state["series"][self.state["next"] % len(self.state["series"])]
        self.state["next"] += 1
        today = datetime.fromtimestamp(now, timezone.utc).date()
        for m in self.api.get("/markets", status="open", series_ticker=series, limit=1000).get("markets", []):
            day = event_day(m["event_ticker"])
            if m["ticker"] in self.done or m["ticker"] in self.state["queue"] or day is None:
                continue
            if today <= day <= today+timedelta(days=1):
                self.state["queue"][m["ticker"]] = dict(event=m["event_ticker"], day=str(day), looks=0, look_at=0)

    def quote(self, ticker, now):
        """Place the simulated NO bid and its YES mirror if the book is two-sided and wide enough."""
        item = self.state["queue"][ticker]
        top = rules.top(self.api.get(f"/markets/{ticker}/orderbook"))
        yes, no = top["yes"]["bid"], top["no"]["bid"]
        if yes is None or no is None or 100-yes-no < MIN_SPREAD:
            item.update(looks=item["looks"]+1, look_at=now+LOOK_AGAIN)   # one-sided or tight: look again later, then give up
            if item["looks"] >= LOOKS:
                self.done.add(ticker)
                del self.state["queue"][ticker]
            return
        for rule, side, bid in (("mention_no", "no", no+1), ("mention_mirror", "yes", yes+1)):
            self.state["orders"].append(dict(id=len(self.state["orders"])+1, rule=rule, family=family(ticker), ticker=ticker,
                                             event=item["event"], day=item["day"], side=side, price=bid, spread=100-yes-no,
                                             yes_mid=(yes+100-no)/2,
                                             placed=now, seen=now, polled=now, checked=0, status="open", size_through=0.0))
            if rule == "mention_no":
                self.send(self.state["orders"][-1])
        self.done.add(ticker)
        del self.state["queue"][ticker]
        self.log(f'quote {ticker} NO bid {no+1}c, mirror YES bid {yes+1}c, spread {100-yes-no}c')

    def fills(self, ticker, orders, now):
        trades, cursor = [], ""
        while True:
            page = self.api.get("/markets/trades", ticker=ticker, min_ts=int(min(o["seen"] for o in orders))-1, limit=1000, cursor=cursor)
            trades += page.get("trades", [])
            cursor = page.get("cursor")
            if not cursor or not page.get("trades"):
                break
        for o in orders:
            found = rules.prints(trades, o["side"], o["price"], o["seen"])
            o.update(polled=now, seen=found["latest"], size_through=o["size_through"]+found["size_through"])
            if o["status"] == "open" and found["filled_at"]:
                o.update(status="filled", filled_at=found["filled_at"])
                self.log(f'fill {o["rule"]} {ticker} at {o["price"]}c')
        return all([self.refresh(o) for o in orders])

    def settle(self, ticker, orders, now):
        m = self.api.get(f"/markets/{ticker}").get("market", {})
        for o in orders:
            o["checked"] = now
        try:
            value = float(m["settlement_value_dollars"])
        except (KeyError, TypeError, ValueError):
            return
        if m.get("status") not in ("finalized", "settled") or not m.get("settlement_ts"):
            return
        if not self.fills(ticker, orders, now):        # prints between the last check and the close
            return                                     # a real order could not be read: try again later
        for o in orders:
            live = o.get("live") or {}
            if "order_id" in live:                     # the exchange's own record decides the real result
                live["pnl"] = live["filled"]*(1-value)-live["cost"]-live["fees"]
            o["payout"] = value if o["side"] == "yes" else 1-value
            if o["status"] == "filled":
                o["net"] = rules.net(o["price"], o["payout"], taker=False)
            o["status"] = "settled" if o["status"] == "filled" else "expired"
        self.log(f'settle {ticker} value {value}')

    def step(self, now=None):
        """Do the most urgent single job: quote, then check fills, then settlements, then keep sweeping."""
        now = now or time.time()
        waiting = defaultdict(list)
        for o in self.state["orders"]:
            if o["status"] in ("open", "filled"):
                waiting[o["ticker"]].append(o)
        fresh = [t for t, item in self.state["queue"].items() if now >= item["look_at"]]
        due_fill = [t for t, group in waiting.items() if now-min(o["polled"] for o in group) > FILL_POLL]
        due_settle = [t for t, group in waiting.items() if now-min(o["checked"] for o in group) > SETTLE_POLL]
        if fresh:
            self.quote(fresh[0], now)
        elif due_settle:
            ticker = min(due_settle, key=lambda t: min(o["checked"] for o in waiting[t]))
            self.settle(ticker, waiting[ticker], now)
        elif due_fill:
            ticker = min(due_fill, key=lambda t: min(o["polled"] for o in waiting[t]))
            self.fills(ticker, waiting[ticker], now)
        else:
            self.sweep(now)
        self.save()


def summary(orders):
    groups = defaultdict(list)
    for o in orders:
        groups[(o["family"], o["rule"])].append(o)
        if o["family"] == "sports broadcasts":         # lost in the holdout: YES was underpriced there
            continue
        groups[("all non-sports", o["rule"])].append(o)
        # Where the edge sat in both research periods: mid-priced words and wide books.
        if 20 <= o.get("yes_mid", 50) <= 80:
            groups[("non-sports, YES 20-80c", o["rule"])].append(o)
        if o["spread"] >= 10:
            groups[("non-sports, spread 10c+", o["rule"])].append(o)
    out = {}
    for (name, rule), rows in sorted(groups.items()):
        done = [o for o in rows if o["status"] in ("settled", "expired")]
        won = [o for o in done if o["status"] == "settled"]
        row = dict(orders=len(rows), finished=len(done), fills=len(won), events=len({o["event"] for o in rows}))
        if done:
            row["fill_rate"] = len(won)/len(done)
        if won:
            nets = [o["net"] for o in won]
            row.update(net_cents=mean(nets), ci95=interval(nets), win_rate=mean(o["payout"] for o in won),
                       return_on_price=sum(nets)/sum(o["price"] for o in won))
        unfilled = [100*o["payout"]-o["price"] for o in done if o["status"] == "expired"]
        if unfilled:
            row["unfilled_would_have_made_cents"] = mean(unfilled)
        out[f"{name} | {rule}"] = row
    return out


def report(out=OUT):
    file = Path(out)/"mentions.json"
    if not file.exists():
        print("No mention markets recorded yet.")
        return {}
    state = json.loads(file.read_text())
    table = summary(state["orders"])
    print(f'{len(state["orders"])//2} mention markets quoted; {len(state["queue"])} waiting for a usable book; '
          f'{len(state["series"])} mention series watched')
    print("Target from research: the NO side nets above +1.5c a fill and beats the mirror, judged after about 300 fills over 30+ events.")
    sent = [o for o in state["orders"] if "order_id" in (o.get("live") or {})]
    if sent:
        closed = [o["live"] for o in sent if "pnl" in o["live"]]
        paid = sum(l["cost"]+l["fees"] for l in closed)
        agree = sum((o["live"]["filled"] > 0) == (o["status"] in ("filled", "settled")) for o in sent)
        print(f'LIVE: {len(sent)} real NO bids, {sum(o["live"]["filled"] > 0 for o in sent)} with fills, {len(closed)} settled: '
              f'paid ${paid:.2f}, profit ${sum(l["pnl"] for l in closed):+.2f}'
              +(f' ({sum(l["pnl"] for l in closed)/paid:+.1%})' if paid else "")
              +f'; simulated fill agreed with the exchange on {agree} of {len(sent)}')
    for name, r in table.items():
        line = f'  {name:48} orders {r["orders"]:4} finished {r["finished"]:4} fills {r["fills"]:4} events {r["events"]:3}'
        if "net_cents" in r:
            line += (f'  fill rate {r["fill_rate"]:.0%}  won {r["win_rate"]:.1%}  return {r["return_on_price"]:+.1%}  '
                     f'net {r["net_cents"]:+.2f}c {r["ci95"] or ""}')
        if "unfilled_would_have_made_cents" in r:
            line += f'  unfilled would have made {r["unfilled_would_have_made_cents"]:+.1f}c'
        print(line)
    return table


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("command", choices=("run", "report"))
    parser.add_argument("--hours", type=float, default=24)
    parser.add_argument("--every", type=float, default=4, help="seconds between requests")
    parser.add_argument("--dir", default=str(OUT))
    parser.add_argument("--live", action="store_true", help="also place real NO bids on the production account")
    parser.add_argument("--contracts", type=int, default=3, help="contracts per real order")
    parser.add_argument("--max-exposure", type=float, default=25, help="dollars in unsettled real orders")
    parser.add_argument("--max-loss", type=float, default=15, help="settled real loss in dollars that stops new orders")
    args = parser.parse_args()
    if args.command == "report":
        report(args.dir)
        return
    broker = None
    if args.live:
        broker = Broker(args.contracts, args.max_exposure, args.max_loss, ("mention_no",))
        print(f"LIVE: real NO bids on the production account, balance ${broker.balance():.2f}; {args.contracts} contracts an order, "
              f"at most ${args.max_exposure:.0f} unsettled, stop after ${args.max_loss:.0f} lost", flush=True)
    book, stop = Mentions(Api(per_second=1/args.every), args.dir, broker), time.time()+args.hours*3600
    while time.time() < stop:
        try:
            book.step()
        except (httpx.HTTPError, RuntimeError, KeyError, ValueError, TypeError) as exc:
            book.log(f"step skipped ({type(exc).__name__}: {str(exc)[:120]})")
            time.sleep(10)


if __name__ == "__main__":
    main()
