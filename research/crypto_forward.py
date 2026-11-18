"""Forward paper test of the 15-minute crypto screen from research/short_dated.py. Read-only; no orders.

Frozen rule (do not tune here): on KXBTC15M / KXETH15M, at a minute boundary with 10 to 1 whole
minutes left before the scheduled close, a side whose ask is 97-99c with a two-sided book and
z >= 2.5 is bought on paper, at most once per event. z is the distance from spot to the strike on
the side bought, divided by (volatility x sqrt(minutes left - 2/3, floored at 1/3)); volatility is
the root mean square of the previous 60 one-minute changes in Coinbase closes; spot is shifted by
the median of the last four (strike - Coinbase close of the minute before open) gaps.
Control: the same rule without the z condition ("plain").

Live differences from the minute-bar backtest, all logged so they can be re-scored:
  - the ask comes from the live order book (with displayed size), read just after the boundary;
  - spot is Coinbase's last trade at that moment rather than the finished minute's close;
  - an entry counts as filled only if the same price or better is still offered 2 seconds later.

  .venv/bin/python research/crypto_forward.py --hours 24      run the recorder
  .venv/bin/python research/crypto_forward.py --report        summarise the log
"""
import argparse
from collections import defaultdict, deque
from datetime import datetime, timezone
import json
import math
from pathlib import Path
from statistics import mean, median
import sys
import time

import httpx
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import config

ROOT = Path(__file__).resolve().parents[1]
LOG = ROOT / "data/raw/crypto-forward"
SERIES = {"KXBTC15M": "BTC-USD", "KXETH15M": "ETH-USD"}
BAND, WINDOW_MIN, Z_MIN, SPAN = (97.0, 99.0), 10, 2.5, 60        # frozen
TICK, RECHECK, BOUNDARY_GRACE = 5, 2.0, 10
COINBASE = "https://api.exchange.coinbase.com"
HEADERS = {"User-Agent": "research-client"}


def write(row):
    LOG.mkdir(parents=True, exist_ok=True)
    day = datetime.fromtimestamp(row["t"], timezone.utc).strftime("%Y-%m-%d")
    with open(LOG / f"{day}.jsonl", "a") as out:
        out.write(json.dumps(row, separators=(",", ":")) + "\n")


def stamp(value):
    value = value.replace("Z", "+00:00")
    if "." in value:                               # Coinbase sends nanoseconds
        head, tail = value.split(".", 1)
        zone = tail[-6:] if tail[-6] in "+-" else ""
        value = f"{head}.{tail[:len(tail) - len(zone)][:6]}{zone}"
    return datetime.fromisoformat(value).timestamp()


def fee(price):
    p = price / 100
    return 100 * .07 * p * (1 - p)


class Kalshi:
    """Public reads with a shared cool-off after HTTP 429."""

    def __init__(self, client):
        self.client, self.quiet_until, self.backoff = client, 0.0, 5.0

    def get(self, path, **params):
        if time.time() < self.quiet_until:
            return None
        try:
            resp = self.client.get(config.KALSHI_PROD_BASE + path, params=params)
        except httpx.RequestError:
            return None
        if resp.status_code == 429:
            self.quiet_until = time.time() + self.backoff
            self.backoff = min(60.0, self.backoff * 2)
            write(dict(t=time.time(), type="error", what="kalshi 429", wait=self.backoff))
            return None
        if resp.status_code != 200:
            return None
        self.backoff = 5.0
        return resp.json()

    def book(self, ticker):
        """Top of book in cents with displayed sizes, plus the local receipt time."""
        data = self.get(f"/markets/{ticker}/orderbook", depth=5)
        got = time.time()
        if not data:
            return None
        levels = data.get("orderbook_fp") or {}

        def best(side):
            rows = [(round(100 * float(p), 1), float(q)) for p, q in levels.get(side) or [] if float(q) > 0]
            return max(rows) if rows else (None, 0.0)
        (yes_bid, yes_bid_size), (no_bid, no_bid_size) = best("yes_dollars"), best("no_dollars")
        return dict(recv=got, yes_bid=yes_bid, yes_bid_size=yes_bid_size,
                    yes_ask=None if no_bid is None else round(100 - no_bid, 1), yes_ask_size=no_bid_size)


def side_quote(book, yes):
    """(ask, displayed size at the ask, two-sided?) for the side bought."""
    bid, offer = book["yes_bid"], book["yes_ask"]
    two_sided = bid is not None and offer is not None and 0 < bid <= offer < 100
    if yes:
        return offer, book["yes_ask_size"], two_sided
    return (None if bid is None else round(100 - bid, 1)), book["yes_bid_size"], two_sided


class Spot:
    def __init__(self, client, product):
        self.client, self.product, self.closes, self.minute = client, product, {}, 0

    def refresh(self, now):
        """Finished one-minute closes, fetched once a minute."""
        if int(now // 60) == self.minute:
            return
        try:
            resp = self.client.get(f"{COINBASE}/products/{self.product}/candles", params=dict(granularity=60))
            if resp.status_code == 200:
                current = int(now // 60) * 60
                self.closes.update({int(r[0]): float(r[4]) for r in resp.json() if int(r[0]) < current})
                self.minute = int(now // 60)
                for old in [k for k in self.closes if k < now - 6 * 3600]:
                    del self.closes[old]
        except (httpx.RequestError, ValueError, IndexError):
            pass

    def last(self):
        try:
            resp = self.client.get(f"{COINBASE}/products/{self.product}/ticker")
            got = time.time()
            if resp.status_code != 200:
                return None
            data = resp.json()
            return dict(price=float(data["price"]), trade_time=stamp(data["time"]), recv=got)
        except (httpx.RequestError, ValueError, KeyError):
            return None

    def sigma(self, boundary):
        steps = []
        for i in range(1, SPAN + 1):
            a, b = self.closes.get(boundary - 60 * i), self.closes.get(boundary - 60 * (i + 1))
            if a and b:
                steps.append(a - b)
        return math.sqrt(mean(s * s for s in steps)) if len(steps) >= SPAN // 2 else None


class Track:
    """One series: its open market, the basis history and the paper entries."""

    def __init__(self, series, kalshi, spot):
        self.series, self.kalshi, self.spot = series, kalshi, spot
        self.market, self.basis, self.seen = None, deque(maxlen=4), set()
        self.minutes_done, self.entered, self.watching, self.next_lookup = set(), {}, [], 0.0
        self.seeded, self.next_seed = False, 0.0

    def add_basis(self, m):
        ref = self.spot.closes.get(int(m["open"]) - 60)
        if m["strike"] and ref and m["ticker"] not in self.seen:
            self.seen.add(m["ticker"])
            self.basis.append(m["strike"] - ref)

    def slim(self, m):
        return dict(ticker=m["ticker"], event=m["event_ticker"], open=stamp(m["open_time"]),
                    close=stamp(m["close_time"]), strike=m.get("floor_strike"))

    def seed(self, now):
        data = self.kalshi.get("/markets", series_ticker=self.series, limit=20, min_close_ts=int(now - 4500),
                               max_close_ts=int(now))
        self.seeded, self.next_seed = data is not None, now + 20
        for m in sorted((self.slim(m) for m in (data or {}).get("markets", [])), key=lambda m: m["close"]):
            self.add_basis(m)

    def find(self, now):
        """The market open right now. The listing is cached upstream, so it is used only for discovery."""
        if self.market and now < self.market["close"] and self.market["strike"]:
            return
        if now < self.next_lookup:
            return
        self.next_lookup = now + 4
        data = self.kalshi.get("/markets", series_ticker=self.series, status="open", limit=5)
        live = [self.slim(m) for m in (data or {}).get("markets", [])]
        live = [m for m in live if m["open"] <= now < m["close"]]
        if not live:
            if self.market and now >= self.market["close"]:
                self.market = None
            return
        m = min(live, key=lambda m: m["close"])
        if not self.market or self.market["ticker"] != m["ticker"]:
            self.minutes_done, self.entered, self.watching = set(), {}, []
        self.market = m
        if m["strike"] and now >= m["open"] + 60:
            self.add_basis(m)

    def z(self, price, boundary, now):
        m, vol = self.market, self.spot.sigma(boundary)
        if price is None or not vol or not m["strike"] or not self.basis:
            return None, vol
        left = max((m["close"] - now) / 60 - 2 / 3, 1 / 3)
        return (price + median(self.basis) - m["strike"]) / (vol * math.sqrt(left)), vol

    def tick(self, slot):
        now = time.time()
        self.spot.refresh(now)
        if not self.seeded and now >= self.next_seed:      # the start-up read can lose to a rate limit
            self.seed(now)
        self.find(now)
        m = self.market
        if not m or now >= m["close"]:
            return
        if m["strike"] and now >= m["open"] + 60:
            self.add_basis(m)
        book, last = self.kalshi.book(m["ticker"]), self.spot.last()
        if not book:
            return
        boundary = int(book["recv"] // 60) * 60
        z_yes, vol = self.z(last and last["price"], boundary, book["recv"]) if last else (None, None)
        row = dict(t=book["recv"], type="obs", series=self.series, ticker=m["ticker"], close=m["close"], strike=m["strike"],
                   yes_bid=book["yes_bid"], yes_ask=book["yes_ask"], yes_bid_size=book["yes_bid_size"],
                   yes_ask_size=book["yes_ask_size"], spot=last and last["price"], spot_trade_time=last and last["trade_time"],
                   spot_recv=last and last["recv"], sigma=vol, basis=median(self.basis) if self.basis else None, z_yes=z_yes)
        write(row)
        self.follow(book)
        # Frozen rule: checked once per minute boundary, 10 to 1 whole minutes before the scheduled close.
        minutes_left = round((m["close"] - boundary) / 60)
        if (boundary in self.minutes_done or book["recv"] - boundary > BOUNDARY_GRACE
                or not 1 <= minutes_left <= WINDOW_MIN):
            return
        self.minutes_done.add(boundary)
        fired = []
        for yes in (True, False):
            ask, size, two_sided = side_quote(book, yes)
            if ask is None or not two_sided or not BAND[0] <= ask <= BAND[1]:
                continue
            z = None if z_yes is None else (z_yes if yes else -z_yes)
            for rule in ("plain", "screen"):
                if rule in self.entered or (rule == "screen" and (z is None or z < Z_MIN)):
                    continue
                entry = dict(t=book["recv"], type="entry", rule=rule, series=self.series, ticker=m["ticker"], event=m["event"],
                             side="yes" if yes else "no", ask=ask, ask_size=size, z=z, seconds_to_close=m["close"] - book["recv"],
                             minutes_left=minutes_left, spot=last and last["price"], strike=m["strike"], sigma=vol,
                             spot_age=None if not last else book["recv"] - last["trade_time"])
                self.entered[rule] = entry
                write(entry)
                fired.append(entry)
        if fired:
            time.sleep(max(0.0, fired[0]["t"] + RECHECK - time.time()))
            again = self.kalshi.book(m["ticker"])
            for entry in fired:
                ask, size, _ = side_quote(again, entry["side"] == "yes") if again else (None, 0.0, False)
                filled = ask is not None and ask <= entry["ask"] and size > 0
                write(dict(t=again["recv"] if again else time.time(), type="recheck", rule=entry["rule"], ticker=entry["ticker"],
                           event=entry["event"], read_ok=bool(again), filled=filled, ask=ask, ask_size=size,
                           delay=(again["recv"] - entry["t"]) if again else None))
                if filled:
                    entry["last_ok"] = again["recv"]
                    self.watching.append(entry)
                elif again:
                    write(dict(t=again["recv"], type="gone", rule=entry["rule"], ticker=entry["ticker"], event=entry["event"],
                               seconds_available=0.0, upper_bound=again["recv"] - entry["t"]))

    def follow(self, book):
        """How long the entry price stayed on offer (resolution: one poll)."""
        for entry in list(self.watching):
            ask, size, _ = side_quote(book, entry["side"] == "yes")
            if ask is None or ask > entry["ask"] or size <= 0:
                self.watching.remove(entry)
                write(dict(t=book["recv"], type="gone", rule=entry["rule"], ticker=entry["ticker"], event=entry["event"],
                           seconds_available=entry.get("last_ok", entry["t"]) - entry["t"], upper_bound=book["recv"] - entry["t"]))
            else:
                entry["last_ok"] = book["recv"]


def rows():
    for path in sorted(LOG.glob("*.jsonl")):
        with open(path) as lines:
            for line in lines:
                try:
                    yield json.loads(line)
                except ValueError:
                    continue


def unsettled():
    need, done = {}, set()
    for r in rows():
        if r["type"] == "entry":
            need[r["ticker"]] = r["t"]
        elif r["type"] == "settle":
            done.add(r["ticker"])
    return {t: ts for t, ts in need.items() if t not in done}


def settle(kalshi, pending, skip=()):
    for ticker in [t for t in pending if t not in skip]:
        data = kalshi.get(f"/markets/{ticker}")
        m = (data or {}).get("market") or {}
        value = m.get("settlement_value_dollars")
        if m.get("result") in ("yes", "no") and value not in (None, ""):
            write(dict(t=time.time(), type="settle", ticker=ticker, value=float(value), result=m["result"],
                       final=m.get("expiration_value"), settled_at=m.get("settlement_ts")))
            del pending[ticker]


def run(hours):
    stop = time.time() + 3600 * hours
    with httpx.Client(timeout=4, headers=HEADERS) as client:
        kalshi = Kalshi(client)
        tracks = [Track(series, kalshi, Spot(client, product)) for series, product in SERIES.items()]
        for track in tracks:
            track.spot.refresh(time.time())
            track.seed(time.time())
        pending, next_settle = unsettled(), 0.0
        write(dict(t=time.time(), type="start", band=BAND, window=WINDOW_MIN, z=Z_MIN, span=SPAN, pending=len(pending)))
        while time.time() < stop:
            slot = math.floor(time.time() / TICK) * TICK + TICK
            time.sleep(max(0.0, slot + .3 - time.time()))
            for track in tracks:
                try:
                    track.tick(slot)
                    for entry in track.entered.values():
                        pending.setdefault(entry["ticker"], entry["t"])
                except Exception as exc:      # keep recording through any single bad read
                    write(dict(t=time.time(), type="error", what=f"{type(exc).__name__}: {exc}"[:200]))
            if time.time() >= next_settle:
                next_settle = time.time() + 30
                live = {t.market["ticker"] for t in tracks if t.market}
                try:
                    settle(kalshi, pending, live)
                except Exception as exc:
                    write(dict(t=time.time(), type="error", what=f"settle {type(exc).__name__}: {exc}"[:200]))


def interval(values, draws=4000):
    if len(values) < 10:
        return None
    data = np.array(values)
    means = np.sort(data[np.random.default_rng(11).integers(0, len(data), size=(draws, len(data)))].mean(axis=1))
    return [round(float(means[int(.025 * draws)]), 2), round(float(means[int(.975 * draws)]), 2)]


def report():
    entries, checks, gone, settled, first, last, ticks, errors = {}, {}, {}, {}, None, None, 0, 0
    for r in rows():
        kind = r["type"]
        if kind == "obs":
            first, last, ticks = first or r["t"], r["t"], ticks + 1
        elif kind == "entry":
            entries.setdefault((r["event"], r["rule"]), r)          # at most one per event per rule
        elif kind == "recheck":
            checks.setdefault((r["event"], r["rule"]), r)
        elif kind == "gone":
            gone.setdefault((r["event"], r["rule"]), r)
        elif kind == "settle":
            settled[r["ticker"]] = r["value"]
        elif kind == "error":
            errors += 1
    if not first:
        print("no observations logged yet")
        return
    days = ticks * TICK / len(SERIES) / 86400                     # time actually observed, not wall time
    print(f"observed {days * 24:.2f} hours of book time between {datetime.fromtimestamp(first, timezone.utc):%Y-%m-%d %H:%M} "
          f"and {datetime.fromtimestamp(last, timezone.utc):%Y-%m-%d %H:%M} UTC; {ticks} book reads; {errors} logged errors")
    out = {}
    for rule in ("screen", "plain"):
        for series in list(SERIES) + ["both"]:
            mine = [(k, e) for k, e in entries.items() if k[1] == rule and series in (e["series"], "both")]
            done = [(k, e) for k, e in mine if e["ticker"] in settled and k in checks and checks[k]["read_ok"]]
            fills = [(k, e) for k, e in done if checks[k]["filled"]]

            def net(k, e, price):
                payout = settled[e["ticker"]] if e["side"] == "yes" else 1 - settled[e["ticker"]]
                return 100 * payout - price - fee(price), payout
            got = [net(k, e, checks[k]["ask"]) + (checks[k]["ask"],) for k, e in fills]      # pays the ask seen 2s later
            paper = [net(k, e, e["ask"])[0] for k, e in done]                                 # every signal at its own ask
            stay = [gone[k]["seconds_available"] for k, e in fills if k in gone]
            line = dict(
                entries=len(mine), settled_and_rechecked=len(done), filled=len(fills),
                fill_rate=round(len(fills) / len(done), 3) if done else None,
                full_losses=sum(p < .5 for _, p, _ in got),
                win_rate=round(mean(p for _, p, _ in got), 4) if got else None,
                break_even=round(mean((a + fee(a)) / 100 for _, _, a in got), 4) if got else None,
                net_cents=round(mean(n for n, _, _ in got), 3) if got else None,
                ci95=interval([n for n, _, _ in got]),
                net_cents_if_every_signal_filled=round(mean(paper), 3) if paper else None,
                median_size_at_signal=median(e["ask_size"] for _, e in mine) if mine else None,
                median_size_2s_later=median(checks[k]["ask_size"] for k, _ in fills) if fills else None,
                median_seconds_price_stayed=median(stay) if stay else None,
                still_offered_at_close=sum(k not in gone for k, _ in fills),
                median_z=round(median(e["z"] for _, e in mine if e["z"] is not None), 2) if any(e["z"] is not None for _, e in mine) else None,
                entries_per_day=round(len(mine) / days, 1) if days else None)
            out[f"{rule} {series}"] = line
            print(f"\n{rule:6} {series}")
            for key, value in line.items():
                print(f"  {key:34} {value}")
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--hours", type=float, default=24)
    parser.add_argument("--report", action="store_true")
    args = parser.parse_args()
    if args.report:
        report()
    else:
        run(args.hours)


if __name__ == "__main__":
    main()
