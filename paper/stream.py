"""Live best bid and ask from Kalshi's WebSocket "ticker" channel, kept on a background thread.

The trader polls the market listing every pass; this gives it the same quote
within about a second of a change (Kalshi flushes the channel once a second),
so a just-in-time bid can go in sooner. It only ever reads: no order goes
through this connection.

It is an addition, not a dependency. `quote` returns None whenever the
connection is down or a market has not been heard from, and the trader's own
polling carries on regardless. A dropped connection is retried with a growing
pause, and every subscription is rebuilt after a reconnect.

  stream = Stream(); stream.start()
  stream.watch({"KXNCAAMBGAME-26NOV18DUKEUNC-DUKE"})
  stream.quote("KXNCAAMBGAME-26NOV18DUKEUNC-DUKE")   # (yes bid, yes ask, received at) in cents, or None

  .venv/bin/python -m paper.stream TICKER [TICKER ...]   # print quotes for 30 seconds
"""
import asyncio
from datetime import datetime, timezone
import json
import sys
import threading
import time

import websockets

import config
from integration.kalshi_client import KalshiClient

URL, PATH = "wss://external-api-ws.kalshi.com/trade-api/ws/v2", "/trade-api/ws/v2"


def log(text):
    print(f"{datetime.now(timezone.utc):%H:%M:%S} stream {text}", flush=True)


class Stream:
    def __init__(self, signer=None, url=URL):
        self.signer = signer or KalshiClient(config.KALSHI_API_KEY, config.KALSHI_PRIVATE_KEY_PATH, demo=False)
        self.url, self.lock = url, threading.Lock()
        self.want, self.quotes, self.connected, self.stopped, self.thread = set(), {}, False, False, None

    # ---- called from the trader's thread -------------------------------------------------------

    def start(self):
        self.thread = threading.Thread(target=lambda: asyncio.run(self.run()), name="kalshi-stream", daemon=True)
        self.thread.start()

    def stop(self):
        self.stopped = True

    def watch(self, tickers):
        """Set the markets to follow. Quotes for markets no longer wanted are dropped."""
        with self.lock:
            self.want = set(tickers)
            self.quotes = {t: q for t, q in self.quotes.items() if t in self.want}

    def quote(self, ticker):
        """(yes bid, yes ask, received at) in whole cents, or None when the stream cannot vouch for this market."""
        with self.lock:
            return self.quotes.get(ticker) if self.connected else None

    # ---- the connection ------------------------------------------------------------------------

    def handle(self, message, sid):
        """Apply one message from the exchange. Returns the ticker subscription's id (None when there is none)."""
        kind, body = message.get("type"), message.get("msg") or {}
        if kind == "ticker":
            try:
                bid, ask = round(100*float(body["yes_bid_dollars"])), round(100*float(body["yes_ask_dollars"]))
            except (KeyError, TypeError, ValueError):
                return sid
            with self.lock:
                if body.get("market_ticker") in self.want:
                    self.quotes[body["market_ticker"]] = (bid, ask, time.time())
        elif kind == "subscribed" and body.get("channel") == "ticker":
            return body.get("sid")
        elif kind == "unsubscribed":
            return None
        elif kind == "error":                          # simplest recovery: drop the connection and rebuild it
            raise RuntimeError(f'exchange error {body.get("code")}: {body.get("msg")}')
        return sid

    def commands(self, want, have, sid):
        """The commands that turn the subscribed set `have` into `want`, and the set they leave subscribed."""
        add, drop = sorted(want-have), sorted(have-want)
        if sid is None:                                # nothing confirmed yet: (re)subscribe; drops wait for the id
            cmds = [dict(cmd="subscribe", params=dict(channels=["ticker"], market_tickers=add, send_initial_snapshot=True))] if add else []
            return cmds, have | want
        if not want:
            return [dict(cmd="unsubscribe", params=dict(sids=[sid]))], set()
        cmds = [dict(cmd="update_subscription", params=dict(sids=[sid], market_tickers=group, action=action))
                for group, action in ((add, "add_markets"), (drop, "delete_markets")) if group]
        return cmds, set(want)

    async def session(self):
        headers = self.signer._auth_headers("GET", PATH)
        async with websockets.connect(self.url, additional_headers=headers, open_timeout=15) as ws:
            sid, have, sent = None, set(), 0
            self.connected = True
            log("connected")
            while not self.stopped:
                with self.lock:
                    want = set(self.want)
                if want != have:
                    cmds, have = self.commands(want, have, sid)
                    for cmd in cmds:
                        sent += 1
                        await ws.send(json.dumps(dict(id=sent, **cmd)))
                try:
                    raw = await asyncio.wait_for(ws.recv(), .5)
                except asyncio.TimeoutError:
                    continue
                sid = self.handle(json.loads(raw), sid)

    async def run(self):
        pause = 1
        while not self.stopped:
            began = time.time()
            try:
                await self.session()
            except Exception as exc:                   # any failure means: no quotes until we are back
                log(f"down ({type(exc).__name__}: {str(exc)[:120]}); polling carries on")
            finally:
                with self.lock:
                    self.connected, self.quotes = False, {}
            pause = 1 if time.time()-began > 60 else min(60, pause*2)
            await asyncio.sleep(pause)


def main():
    tickers = sys.argv[1:]
    if not tickers:
        sys.exit(__doc__)
    stream, seen = Stream(), {}
    stream.start()
    stream.watch(tickers)
    stop = time.time()+30
    while time.time() < stop:
        for t in tickers:
            q = stream.quote(t)
            if q and q != seen.get(t):
                seen[t] = q
                log(f"{t} bid {q[0]}c ask {q[1]}c")
        time.sleep(.2)
    log(f"heard from {len(seen)} of {len(tickers)} markets")


if __name__ == "__main__":
    main()
