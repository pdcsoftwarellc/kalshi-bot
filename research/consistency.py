"""Arithmetic consistency between linked Kalshi markets (no crypto). Public reads only.

Relations, all evaluated as taker baskets with the fee 0.07*p*(1-p) on every leg:
  exh   exhaustive event: buy YES on every outcome, payout 100c
  me    mutually exclusive event: buy NO on outcomes, payout (n-1)*100c
  imp   A implies B (higher strike implies lower strike; "wins by over k" implies
        "wins"): buy YES B and NO A, payout >= 100c
  excl  A and B cannot both happen (team X by over k vs team Y wins): buy both NO,
        payout >= 100c

live:  each cycle refreshes the cached /markets listing for everything closing
       soon, screens every relation on the listing quotes, then re-prices each
       candidate, plus a fixed panel of the busiest events, from fresh order books.
       Only order-book results count. Confirmed baskets are re-polled to time them.
"""
import argparse
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
import json
import math
from pathlib import Path
import re
import sys
import threading
import time

import httpx

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from research.api_cost import ROOT, save
import config

OUT = ROOT / "data/processed/consistency"
BASE = config.KALSHI_PROD_BASE
CRYPTO = re.compile(r"^KX?(BTC|ETH|SOL|XRP|DOGE|ADA|BNB|LTC|AVAX|LINK|SHIB|HYPE|SUI|DOT|BCH|CRYPTO|COIN)")
PAUSE = .22          # per worker after each read; two workers
WORKERS = 2
_lock = threading.Lock()
STATS = dict(requests=0, throttled=0)


def read(client, path, **params):
    for attempt in range(7):
        try:
            resp = client.get(BASE + path, params=params)
        except httpx.RequestError:
            time.sleep(2 ** attempt)
            continue
        with _lock:
            STATS["requests"] += 1
        if resp.status_code == 429 or resp.status_code >= 500:
            with _lock:
                STATS["throttled"] += 1
            time.sleep(1.5 * 2 ** attempt)
            continue
        if resp.status_code == 404:
            return None
        resp.raise_for_status()
        time.sleep(PAUSE)
        return resp.json()
    raise RuntimeError("read failed: " + path)


def fee(p):
    """Taker fee in cents for one contract bought at p cents."""
    return .07 * p * (100 - p) / 100


def fee_up(p):
    return math.ceil(fee(p) - 1e-9)


def listing_quote(m):
    bid, ask = 100 * float(m["yes_bid_dollars"]), 100 * float(m["yes_ask_dollars"])
    bs, az = float(m.get("yes_bid_size_fp") or 0), float(m.get("yes_ask_size_fp") or 0)
    return (bid if bs > 0 and bid > 0 else None, bs, ask if az > 0 and ask < 100 else None, az)


def book_quote(client, ticker):
    data = read(client, f"/markets/{ticker}/orderbook", depth=1)
    if not data:
        return ticker, (None, 0, None, 0), time.time()
    ob = data["orderbook_fp"]
    yes, no = ob.get("yes_dollars") or [], ob.get("no_dollars") or []
    yb = max(yes, key=lambda x: float(x[0])) if yes else None
    nb = max(no, key=lambda x: float(x[0])) if no else None
    return ticker, (100 * float(yb[0]) if yb else None, float(yb[1]) if yb else 0,
                    100 - 100 * float(nb[0]) if nb else None, float(nb[1]) if nb else 0), time.time()


def books(client, tickers):
    with ThreadPoolExecutor(WORKERS) as pool:
        got = list(pool.map(lambda t: book_quote(client, t), tickers))
    return {t: q for t, q, _ in got}, {t: ts for t, _, ts in got}


def basket(kind, key, legs, payout, Q):
    """legs: (ticker, 'yes'|'no'). None unless every leg has displayed size."""
    cost = exact = up = 0
    size, detail = math.inf, []
    for ticker, side in legs:
        bid, bs, ask, az = Q.get(ticker, (None, 0, None, 0))
        price, sz = (ask, az) if side == "yes" else (None if bid is None else 100 - bid, bs)
        if price is None or sz <= 0:
            return None
        cost += price
        exact += fee(price)
        up += fee_up(price)
        size = min(size, sz)
        detail.append(dict(ticker=ticker, side=side, price=round(price, 2), size=sz))
    return dict(kind=kind, key=key, legs=detail, cost=round(cost, 2), payout=payout, size=size,
                gross=round(payout - cost, 3), profit_exact=round(payout - cost - exact, 3),
                profit_rounded=round(payout - cost - up, 3),
                id=kind + "|" + "|".join(sorted(t + ":" + s for t, s in legs)))


def series_of(m):
    return m["event_ticker"].split("-")[0]


def game_of(m):
    return m["event_ticker"].split("-", 1)[1] if "-" in m["event_ticker"] else ""


def relations(markets, me_events):
    """Static structure. Returns lists of groups; see module docstring."""
    by_event = defaultdict(list)
    for m in markets:
        by_event[m["event_ticker"]].append(m)
    imp, excl, exh, me = [], [], [], []
    ladders = defaultdict(list)                  # (base, prefix, direction) -> [(strike, ticker, title-shape)]
    for m in markets:
        st = m.get("strike_type")
        base, last = m["ticker"].rsplit("-", 1)
        prefix = re.match(r"[A-Z]*", last).group()
        shape = re.sub(r"[\d.,$]+", "#", m["title"])
        if st in ("greater", "greater_or_equal") and m.get("floor_strike") is not None:
            ladders[(base, prefix, "up", shape)].append((float(m["floor_strike"]), m["ticker"]))
        elif st in ("less", "less_or_equal") and m.get("cap_strike") is not None:
            ladders[(base, prefix, "down", shape)].append((-float(m["cap_strike"]), m["ticker"]))
    winners = {}                                 # (series, game) -> {team: ticker}; also three-way and two-way sets
    for event, ms in by_event.items():
        s = series_of(ms[0])
        types = {m.get("strike_type") for m in ms}
        lasts = {m["ticker"].rsplit("-", 1)[1]: m["ticker"] for m in ms}
        if types == {"structured"} and len(ms) == 3 and "TIE" in lasts:
            exh.append((s, event, sorted(lasts.values()), "three-way"))
            me.append((s, event, sorted(lasts.values())))
            winners[(s, game_of(ms[0]))] = {k: v for k, v in lasts.items() if k != "TIE"}
        elif types == {"structured"} and len(ms) == 2 and re.search(r"(GAME|MATCH|FIGHT)$", s):
            exh.append((s, event, sorted(lasts.values()), "two-way"))
            winners[(s, game_of(ms[0]))] = dict(lasts)
        elif "between" in types and types <= {"between", "less", "greater"} and {"less", "greater"} <= types \
                and re.match(r"KX(HIGH|LOW)", s):
            exh.append((s, event, sorted(lasts.values()), "range-ladder"))
            me.append((s, event, sorted(lasts.values())))
        elif event in me_events and len(ms) >= 2:
            me.append((s, event, sorted(m["ticker"] for m in ms)))
            exh.append((s, event, sorted(m["ticker"] for m in ms), "flagged-exclusive-unverified-exhaustive"))
    sides = defaultdict(dict)                    # (spread series, game) -> team -> ordered weak..strong tickers
    for (base, prefix, direction, shape), rows in ladders.items():
        rows.sort()
        chain = [t for _, t in rows]
        s = base.split("-")[0]
        if s.endswith("SPREAD") and prefix:
            game = base.split("-", 1)[1]
            for ws in (s.replace("SPREAD", "GAME"), s.replace("SPREAD", "")):
                w = winners.get((ws, game), {}).get(prefix)
                if w:
                    chain = [w] + chain
                    break
            sides[(s, game)][prefix] = chain
        if len(chain) >= 2:
            imp.append((s, base + "|" + prefix, chain))
    for (s, game), teams in sides.items():
        names = sorted(teams)
        if len(names) == 2:
            excl.append((s, s + "-" + game, teams[names[0]], teams[names[1]]))
    return imp, excl, exh, me


def evaluate(rel, Q, floor):
    """Every basket with gross profit >= floor cents. At most three per group."""
    imp, excl, exh, me = rel
    found = []
    for s, key, chain in imp:
        rows = []
        quotes = [Q.get(t, (None, 0, None, 0)) for t in chain]
        for i, weak in enumerate(chain):
            ask = quotes[i][2]
            if ask is None:
                continue
            for j in range(i + 1, len(chain)):
                bid = quotes[j][0]
                if bid is not None and bid - ask >= floor:
                    b = basket("imp", key, [(weak, "yes"), (chain[j], "no")], 100, Q)
                    if b:
                        rows.append(dict(b, family=s))
        found += sorted(rows, key=lambda b: -b["gross"])[:3]
    for s, key, a, bside in excl:
        rows = []
        for x in a:
            bx = Q.get(x, (None,))[0]
            if bx is None:
                continue
            for y in bside:
                by = Q.get(y, (None,))[0]
                if by is not None and bx + by - 100 >= floor:
                    b = basket("excl", key, [(x, "no"), (y, "no")], 100, Q)
                    if b:
                        rows.append(dict(b, family=s))
        found += sorted(rows, key=lambda b: -b["gross"])[:3]
    for s, key, tickers, note in exh:
        b = basket("exh", key, [(t, "yes") for t in tickers], 100, Q)
        if b and b["gross"] >= floor:
            found.append(dict(b, family=s, note=note))
    for s, key, tickers in me:
        legs = [(t, "no") for t in tickers if (Q.get(t, (None,))[0] or 0) >= 1]
        if len(legs) >= 2:
            b = basket("me", key, legs, 100 * (len(legs) - 1), Q)
            if b and b["gross"] >= floor:
                found.append(dict(b, family=s))
    return found


def near_listing(client, days):
    now = int(time.time())
    out, cursor = [], ""
    while True:
        q = dict(status="open", limit=1000, min_close_ts=now, max_close_ts=now + int(days * 86400), mve_filter="exclude")
        if cursor:
            q["cursor"] = cursor
        data = read(client, "/markets", **q)
        out += data["markets"]
        cursor = data.get("cursor")
        if not cursor:
            return [m for m in out if not CRYPTO.match(m["event_ticker"])]


def exclusive_events(client, markets, long_dated):
    """mutually_exclusive flags for events that are not ladders or winners; cached once."""
    path = OUT / "exclusive_events.json"
    if path.exists():
        return json.loads(path.read_text())
    count = defaultdict(lambda: defaultdict(int))
    for m in markets:
        if m.get("strike_type") in ("custom", "structured", "functional", None):
            count[series_of(m)][m["event_ticker"]] += 1
    wanted = sorted(s for s, ev in count.items() if max(ev.values()) >= 3)
    flags = {e["event_ticker"]: e["series_ticker"] for e in long_dated}
    for s in wanted:
        data = read(client, "/events", series_ticker=s, status="open", limit=200) or {"events": []}
        for e in data["events"]:
            if e.get("mutually_exclusive"):
                flags[e["event_ticker"]] = s
    save(path, flags)
    return flags


def long_dated_candidates():
    """Flagged-exclusive events from the one-off open-events dump whose listing sums were close to a lock."""
    path = OUT / "open_events.json"
    if not path.exists():
        return [], []
    events, markets = [], []
    for e in json.loads(path.read_text()):
        ms = [m for m in e.get("markets", []) if m["status"] == "active"]
        if not e.get("mutually_exclusive") or len(ms) < 2 or CRYPTO.match(e["event_ticker"]):
            continue
        bids = sum(float(m["yes_bid_dollars"]) for m in ms)
        volume = sum(float(m.get("volume_24h_fp") or 0) for m in ms)
        if bids >= .99 and volume > 0:
            events.append(e)
            markets += ms
    return events, markets


def live(args):
    live_dir = OUT / "live"
    live_dir.mkdir(parents=True, exist_ok=True)
    log = (live_dir / "violations.jsonl").open("a")
    watch = {}                                   # id -> dict(first, last, legs..., seen)
    summary = []
    start = time.time()
    with httpx.Client(timeout=30) as client:
        long_events, long_markets = long_dated_candidates()
        for cycle in range(args.first_cycle, args.first_cycle + args.cycles):
            t0 = time.time()
            markets = near_listing(client, args.days)
            flags = exclusive_events(client, markets, long_events)
            near = {m["ticker"] for m in markets}
            universe = markets + [m for m in long_markets if m["ticker"] not in near]
            rel = relations(universe, flags)
            L = {m["ticker"]: listing_quote(m) for m in universe}
            screen = evaluate(rel, L, args.floor)
            screen.sort(key=lambda b: -b["gross"])
            # Panel: busiest events, whole books, so the listing screen is not the only gate.
            volume = defaultdict(float)
            members = defaultdict(list)
            for m in markets:
                if args.core and not re.search(r"(GAME|MATCH|SPREAD|TOTAL)$", series_of(m)):
                    continue                     # winner, spread and total books only: props swamp a game
                if args.core and re.search(r"TEAMTOTAL$", series_of(m)):
                    continue
                g = game_of(m)
                volume[g] += float(m.get("volume_24h_fp") or 0)
                members[g].append(m["ticker"])
            panel = []
            for g in sorted(volume, key=lambda g: -volume[g]):
                if len(panel) + len(members[g]) > args.panel:
                    continue
                panel += members[g]
                if len(panel) > args.panel - 20:
                    break
            panel += [m["ticker"] for m in long_markets]
            need = []
            for b in screen:
                for leg in b["legs"]:
                    if leg["ticker"] not in need and len(need) < args.confirm:
                        need.append(leg["ticker"])
            for w in watch.values():
                need += [leg["ticker"] for leg in w["legs"]]
            tickers = list(dict.fromkeys(need + panel))
            Q, _ = books(client, tickers)
            stamp = time.time()
            fresh = evaluate(rel, Q, -1.0)
            hits = [b for b in fresh if b["profit_exact"] > 0]
            near_miss = sorted((b for b in fresh if b["profit_exact"] <= 0), key=lambda b: -b["profit_exact"])[:40]
            for b in hits:
                w = watch.setdefault(b["id"], dict(first=stamp, legs=b["legs"], kind=b["kind"], key=b["key"],
                                                    payout=b["payout"], family=b["family"], seen=[]))
                w["last"] = stamp
                w["seen"].append((round(stamp, 1), b["profit_exact"], b["profit_rounded"], b["size"]))
                log.write(json.dumps(dict(b, ts=stamp, cycle=cycle)) + "\n")
            log.flush()
            # Time the confirmed ones: re-poll their legs until they vanish or the window ends.
            active = {b["id"] for b in hits}
            polls = 0
            while active and polls < args.polls:
                time.sleep(args.poll_gap)
                polls += 1
                legs = list(dict.fromkeys(l["ticker"] for i in active for l in watch[i]["legs"]))[:120]
                Qp, _ = books(client, legs)
                now = time.time()
                for i in list(active):
                    w = watch[i]
                    b = basket(w["kind"], w["key"], [(l["ticker"], l["side"]) for l in w["legs"]], w["payout"], Qp)
                    if b and b["profit_exact"] > 0:
                        w["last"] = now
                        w["seen"].append((round(now, 1), b["profit_exact"], b["profit_rounded"], b["size"]))
                        log.write(json.dumps(dict(b, family=w["family"], ts=now, cycle=cycle, poll=polls)) + "\n")
                    else:
                        w.setdefault("gone", now)
                        active.discard(i)
                log.flush()
            row = dict(cycle=cycle, started=t0, priced=stamp, markets_listed=len(markets), universe=len(universe),
                       groups=dict(imp=len(rel[0]), excl=len(rel[1]), exh=len(rel[2]), me=len(rel[3])),
                       listing_candidates=len(screen), listing_positive_after_fees=sum(b["profit_exact"] > 0 for b in screen),
                       books_read=len(tickers), book_baskets_within_1c=len(fresh), book_violations=len(hits),
                       requests=STATS["requests"], throttled=STATS["throttled"], polls=polls)
            summary.append(row)
            save(live_dir / f"cycle-{cycle:02d}.json", dict(row, hits=hits, near_miss=near_miss,
                 listing_top=screen[:60]))
            save(live_dir / "summary.json", dict(cycles=summary, watch=watch))
            print(json.dumps(row), flush=True)
            for b in hits[:12]:
                print("  HIT", b["kind"], b["family"], b["profit_exact"], b["profit_rounded"], b["size"],
                      [(l["ticker"], l["side"], l["price"], l["size"]) for l in b["legs"]], flush=True)
            if time.time() - start > args.budget * 60:
                break
            wait = args.gap - (time.time() - t0)
            if wait > 0 and cycle + 1 < args.first_cycle + args.cycles:
                time.sleep(wait)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cycles", type=int, default=6)
    parser.add_argument("--gap", type=float, default=300, help="seconds between cycle starts")
    parser.add_argument("--days", type=float, default=4)
    parser.add_argument("--panel", type=int, default=350)
    parser.add_argument("--confirm", type=int, default=250)
    parser.add_argument("--floor", type=float, default=0.0, help="listing gross cents needed to confirm on books")
    parser.add_argument("--polls", type=int, default=8)
    parser.add_argument("--poll-gap", type=float, default=10)
    parser.add_argument("--budget", type=float, default=32, help="minutes")
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--core", action="store_true", help="panel = winner/spread/total books of the busiest games")
    parser.add_argument("--first-cycle", type=int, default=0, help="numbering offset when resuming")
    args = parser.parse_args()
    global WORKERS
    WORKERS = args.workers
    live(args)


if __name__ == "__main__":
    main()
