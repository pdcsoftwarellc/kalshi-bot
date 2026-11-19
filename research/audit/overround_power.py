"""Overround in events with many mutually exclusive outcomes: can a NO buyer or a resting NO bid harvest it?

Research only. Public Kalshi reads, no orders, no authentication.

In an event flagged mutually_exclusive at most one outcome settles YES, so NO
on k outcomes pays at least (k-1)*100. If the YES prices sum to more than 100
the NO side collects the excess.

  A. Taker: at one instant buy NO at the ask (100 - YES bid) on every outcome
     whose YES bid exceeds its taker fee. Locked profit per set of one contract
     each = sum(YES bid - fee) - 100. Fee 7*p*(1-p) cents times the series
     fee_multiplier (the exchange rounds up per order, so this is the
     large-order limit).
  B. Maker: a NO bid one cent above the best NO bid (a YES offer at ask - 1)
     on every outcome quoted two-sided (1 <= bid < ask <= 99) and >= 2c wide,
     resting to the close, held to settlement. Filled only when a LATER hourly
     bar prints a trade strictly above the offer (the rule of paper/rules.py
     `prints`, on bars). Mirror control: a YES bid at bid + 1, filled by a
     later print strictly below it. One contract per order. Maker fee zero on
     "quadratic" series, 1.75*p*(1-p)*multiplier on "quadratic_with_maker_fees".

Fixed before any result was read

* Universe: events with mutually_exclusive = true, 4+ settled outcomes and
  5,000+ contracts traded, from series with 1,000,000+ lifetime contracts
  (sports: series whose ticker does not look like a game, prop, spread or
  total). No Mentions, no hourly or 15-minute series.
* Clocks: 24h before the event's close (headline), 7 days before (events that
  lived that long), 1h before (quotes only). The event's close is the modal
  close of its outcomes. The quote at T is the close of the last hourly bar
  ending at or before T, looking back at most 7 days (bars exist only for
  hours in which something changed).
* Recent tier: closes 2026-09-19 to 2026-11-18, every outcome, batch bars; at
  most PER_SERIES_RECENT events per series (pseudo-random). Archive tier:
  closes 2025-01-01 to 2026-09-18, one request per market, so at most
  ARCHIVE_OUTCOMES outcomes per event (the most traded) and events taken
  round-robin over category x quarter x series.
* Evidence: NO must beat zero and the mirror in both chronological halves,
  intervals from a bootstrap over EVENTS.

Commands (resumable; --minutes bounds network time)

    series    GET /series with lifetime volume
    discover  GET /events per candidate series (nested markets: recent settled and open)
    listings  GET /historical/markets per mutually exclusive series
    rbars     recent hourly bars (batch)
    abars     archive hourly bars (one market per request)
    dbars     daily bars over the whole life (long-dated clock)
    trades    trade record for a sample of recent orders (capacity)
    report    offline -> data/processed/overround/report.txt
"""
import argparse
from collections import Counter, defaultdict
from datetime import date, datetime, timezone
import hashlib
import json
from pathlib import Path
import re
import sys
import time

import httpx
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))  # AUDIT: copy lives one level deeper
import config
from research.api_cost import ROOT, midnight, timestamp

RAW = ROOT / "data/raw/overround"
OUT = ROOT / "data/processed/overround"
PACE = 2.0
CUTOFF = midnight(date(2026, 9, 19))
HOLD_FROM = midnight(date(2025, 1, 1))
NOW = midnight(date(2026, 11, 19))
WINDOW = 10*86400
MIN_SERIES_VOLUME, MIN_EVENT_VOLUME, MIN_OUTCOMES = 1e6, 5000, 4
PER_SERIES_RECENT, ARCHIVE_OUTCOMES, PER_SERIES_ARCHIVE = 30, 10, 12
RNG = np.random.default_rng(11)
SPORT_NOISE = re.compile(r"GAME|MATCH|SPREAD|TOTAL|PTS|REB|AST|3PT|TD|YDS|HIT|HR|KS|GOAL|SHOT|BTTS|1H|2H|FIRST|ADVANCE|PROP|MVE|SACK|REC|RUSH|PASS|"
                         r"SOG|ACE|SET|MAP|ROUND|FIGHT|MOV|H2H|TOP|CUT|R1|R2|R3|LEAD|DOUBLE|CORNER|CARD|SCORE|HALF|QUARTER|INNING|RUNS|WINS|MAKE|"
                         r"PLAYOFF|SERIES|PREPACK|1Q|2Q|3Q|4Q|RFI|F5|F3|F7|1P|2P|3P|MARGIN|MOF|DISTANCE|EXACT")
GROUPS = {"Politics": "Politics+Elections", "Elections": "Politics+Elections", "World": "Politics+Elections", "Entertainment": "Entertainment",
          "Sports": "Sports winners", "Climate and Weather": "Weather ranges", "Financials": "Financials+Crypto", "Crypto": "Financials+Crypto",
          "Commodities": "Financials+Crypto", "Economics": "Economics", "Science and Technology": "Other", "Companies": "Other", "Health": "Other",
          "Social": "Other", "Transportation": "Other"}

last_request = [0.0]
deadline = [float("inf")]
log = Counter()


class OutOfTime(Exception):
    pass


def read(client, path, params):
    """One paced public read. Parsed JSON, or None on a 4xx other than 429."""
    for pause in (30, 60, 120, 240, None):
        if time.time() > deadline[0]:
            raise OutOfTime
        wait = last_request[0]+PACE-time.time()
        if wait > 0:
            time.sleep(wait)
        log["requests"] += 1
        try:
            resp = client.get(config.KALSHI_PROD_BASE+path, params=params, headers={"User-Agent": "research-client"})
        except httpx.RequestError:
            resp = None
        last_request[0] = time.time()
        if resp is not None and resp.status_code == 200:
            return resp.json()
        if resp is not None and resp.status_code != 429 and resp.status_code < 500:
            log[f"http_{resp.status_code}"] += 1
            return None
        log["http_429" if resp is not None and resp.status_code == 429 else "server_or_network_error"] += 1
        if pause is None:
            raise RuntimeError("read failed after long back-off: "+path)
        time.sleep(pause)


def load(path, default):
    return json.loads(path.read_text()) if path.exists() else default


def dump(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(".tmp")
    temp.write_text(json.dumps(value, default=lambda o: o.item() if hasattr(o, "item") else str(o)))
    temp.replace(path)


def pick_hash(*parts):
    return hashlib.sha256("|".join(parts).encode()).hexdigest()


def stamp(m, key):
    try:
        return int(timestamp(m[key]))
    except (KeyError, TypeError, ValueError, AttributeError):
        return None


def cents(m, key):
    try:
        return round(100*float(m[key]), 2)
    except (KeyError, TypeError, ValueError):
        return None


def slim(m):
    return dict(ticker=m["ticker"], event=m["event_ticker"], name=(m.get("yes_sub_title") or m.get("subtitle") or "")[:40], open=stamp(m, "open_time"),
                close=stamp(m, "close_time"), settle=stamp(m, "settlement_ts"), result=m.get("result"), status=m.get("status"),
                volume=float(m.get("volume_fp") or 0), early=bool(m.get("can_close_early")), bid=cents(m, "yes_bid_dollars"), ask=cents(m, "yes_ask_dollars"),
                bid_size=float(m.get("yes_bid_size_fp") or 0), ask_size=float(m.get("yes_ask_size_fp") or 0))


# ---------------------------------------------------------------- listings

def series_step(client):
    data = read(client, "/series", dict(include_volume="true")) or {}
    rows = [dict(ticker=s["ticker"], title=s["title"], category=s.get("category"), frequency=s.get("frequency"), volume=float(s.get("volume_fp") or 0),
                 fee_type=s.get("fee_type"), fee_multiplier=s.get("fee_multiplier")) for s in data.get("series") or []]
    dump(RAW/"series.json", rows)
    print(f"{len(rows)} series", flush=True)


def candidates():
    out = []
    for s in load(RAW/"series.json", []):
        if s["volume"] < MIN_SERIES_VOLUME or s["category"] not in GROUPS or s["frequency"] in ("hourly", "fifteen_min"):
            continue
        if s["category"] == "Sports" and SPORT_NOISE.search(s["ticker"]):
            continue
        out.append(s)
    out.sort(key=lambda s: -s["volume"])
    # round-robin over categories so that a short run still covers every category
    by = defaultdict(list)
    for s in out:
        by[GROUPS[s["category"]]].append(s)
    order, depth = [], 0
    while any(len(v) > depth for v in by.values()):
        order += [v[depth] for _, v in sorted(by.items()) if len(v) > depth]
        depth += 1
    return order


def discover_step(client):
    """First page of events per series (newest first): the flag, recent settled outcomes and open outcomes with their quotes."""
    have = load(RAW/"events.json", {})
    todo = [s for s in candidates() if s["ticker"] not in have]
    print(f"discover: {len(have)} series on disk, {len(todo)} to read", flush=True)
    try:
        for n, s in enumerate(todo):
            data = read(client, "/events", dict(series_ticker=s["ticker"], with_nested_markets="true", limit=200)) or {}
            rows = []
            for e in data.get("events") or []:
                rows.append(dict(event=e["event_ticker"], me=bool(e.get("mutually_exclusive")), title=(e.get("title") or "")[:70],
                                 markets=[slim(m) for m in e.get("markets") or []]))
            have[s["ticker"]] = dict(events=rows, more=bool(data.get("cursor")))
            if n % 20 == 19:
                dump(RAW/"events.json", have)
    finally:
        dump(RAW/"events.json", have)
    me = [k for k, v in have.items() if any(e["me"] for e in v["events"])]
    print(f"{len(have)} series read; {len(me)} with a mutually exclusive event", flush=True)


def listing_step(client, pages=3):
    """Archive listings (newest first) for series with a mutually exclusive event of 4+ outcomes or archived events."""
    events = load(RAW/"events.json", {})
    wanted = [k for k, v in events.items() if any(e["me"] and (len(e["markets"]) >= MIN_OUTCOMES or not e["markets"]) for e in v["events"])]
    for n, series in enumerate(wanted):
        path = RAW/"archive-markets"/f"{series}.json"
        have = load(path, dict(markets=[], pages=0, cursor="", done=False))
        while not have["done"]:
            data = read(client, "/historical/markets", dict(series_ticker=series, limit=1000, **({"cursor": have["cursor"]} if have["cursor"] else {}))) or {}
            have["pages"] += 1
            rows = data.get("markets") or []
            oldest = None
            for m in rows:
                v = slim(m)
                if v["close"]:
                    oldest = v["close"] if oldest is None else min(oldest, v["close"])
                if m.get("result") in ("yes", "no") and v["close"] and v["open"] and HOLD_FROM <= v["close"] < CUTOFF:
                    have["markets"].append(v)
            have["cursor"] = data.get("cursor") or ""
            have["done"] = bool(not have["cursor"] or not rows or have["pages"] >= pages or (oldest is not None and oldest < HOLD_FROM))
            dump(path, have)
        if n % 25 == 0:
            print(f"  {n+1}/{len(wanted)} {series:28} {len(have['markets']):5} archived markets", flush=True)


def universe():
    """Eligible settled events: {event: dict(series, category, group, tier, close, markets=[...], thinned)}."""
    meta = {s["ticker"]: s for s in load(RAW/"series.json", [])}
    events = load(RAW/"events.json", {})
    out = {}
    for series, v in events.items():
        s = meta[series]
        if s["category"] == "Sports" and SPORT_NOISE.search(series):
            continue
        flag = {e["event"]: e for e in v["events"]}
        groups = defaultdict(list)
        for e in v["events"]:
            for m in e["markets"]:
                groups[e["event"]].append(dict(m, archived=False))
        for m in load(RAW/"archive-markets"/f"{series}.json", dict(markets=[]))["markets"]:
            if m["event"] in flag and not flag[m["event"]]["markets"]:
                groups[m["event"]].append(dict(m, archived=True))
        for event, ms in groups.items():
            if not flag[event]["me"] or len(ms) < MIN_OUTCOMES or any(m["result"] not in ("yes", "no") or not m["close"] or not m["open"] for m in ms):
                continue
            if sum(m["volume"] for m in ms) < MIN_EVENT_VOLUME:
                continue
            closes = Counter(m["close"]//600 for m in ms)
            close = 600*min(closes, key=lambda k: (-closes[k], k))
            archived = ms[0]["archived"]
            if not (HOLD_FROM <= close < NOW) or (archived and close >= CUTOFF+86400):
                continue
            out[event] = dict(event=event, series=series, category=s["category"], group=GROUPS[s["category"]], title=flag[event]["title"],
                              tier="archive" if archived else "recent", close=close, n=len(ms), yes=sum(m["result"] == "yes" for m in ms),
                              volume=sum(m["volume"] for m in ms), early=any(m["early"] for m in ms), markets=ms,
                              maker_fee=s["fee_type"] == "quadratic_with_maker_fees", mult=float(s["fee_multiplier"] if s["fee_multiplier"] is not None else 1))
    return out


def quarter(ts):
    d = datetime.fromtimestamp(ts, timezone.utc)
    return f"{d.year}Q{(d.month-1)//3+1}"


def sample():
    """Events to fetch bars for, in order. Recent: up to PER_SERIES_RECENT a series. Archive: round-robin over group x quarter, series capped."""
    uni = universe()
    recent, per = [], Counter()
    for e in sorted((e for e in uni.values() if e["tier"] == "recent"), key=lambda e: pick_hash("recent", e["event"])):
        if per[e["series"]] < PER_SERIES_RECENT:
            per[e["series"]] += 1
            recent.append(e)
    cells, per = defaultdict(list), Counter()
    for e in sorted((e for e in uni.values() if e["tier"] == "archive"), key=lambda e: pick_hash("archive", e["event"])):
        if per[e["series"]] < PER_SERIES_ARCHIVE:
            per[e["series"]] += 1
            cells[(e["group"], quarter(e["close"]))].append(e)
    archive, depth = [], 0
    while any(len(v) > depth for v in cells.values()):
        archive += [v[depth] for _, v in sorted(cells.items()) if len(v) > depth]
        depth += 1
    for e in archive:
        e["picked"] = sorted(e["markets"], key=lambda m: (-m["volume"], m["ticker"]))[:ARCHIVE_OUTCOMES]
    for e in recent:
        e["picked"] = e["markets"]
    return recent, archive


# ---------------------------------------------------------------- bars

def money(group, field):
    value = group.get(field+"_dollars", group.get(field)) if group else None
    return None if value is None else round(100*float(value), 2)


def compact(candles):
    """[end_ts, yes bid close, yes ask close, trade low, trade high, volume] per bar."""
    return [[b["end_period_ts"], money(b.get("yes_bid"), "close"), money(b.get("yes_ask"), "close"), money(b.get("price"), "low"),
             money(b.get("price"), "high"), float(b.get("volume_fp", b.get("volume")) or 0)] for b in sorted(candles or [], key=lambda b: b["end_period_ts"])]


def span(e, m):
    """Hourly window: the last 10 days before the event's close (or the outcome's own earlier close) to two hours after."""
    end = max(m["close"], e["close"])//3600*3600+7200
    return max(m["open"]//3600*3600-3600, min(m["close"], e["close"])-WINDOW)//3600*3600, end


def rbars_step(client):
    have = load(RAW/"recent-bars.json", {})
    recent, _ = sample()
    jobs = sorted(((span(e, m), m["ticker"]) for e in recent for m in e["picked"] if m["ticker"] not in have), key=lambda j: (j[0][1], j[0][0]))
    chunks, cur = [], []
    for j in jobs:
        trial = cur+[j]
        hours = (max(x[0][1] for x in trial)-min(x[0][0] for x in trial))//3600+2
        if cur and (len(trial) > 100 or len(trial)*hours > 9000):
            chunks.append(cur)
            cur = []
        cur.append(j)
    chunks += [cur] if cur else []
    print(f"recent bars: {len(have)} on disk, {len(jobs)} markets in {len(chunks)} requests", flush=True)
    try:
        for n, group in enumerate(chunks):
            lo, hi = min(x[0][0] for x in group), max(x[0][1] for x in group)
            data = read(client, "/markets/candlesticks", dict(market_tickers=",".join(x[1] for x in group), start_ts=lo, end_ts=hi, period_interval=60))
            if data is None:
                log["batch_refused"] += 1
                for x in group:                                  # fall back to one at a time later
                    log["batch_refused_markets"] += 1
                continue
            got = {x["market_ticker"]: compact(x.get("candlesticks")) for x in data.get("markets", [])}
            for (a, b), t in group:
                have[t] = [bar for bar in got.get(t, []) if a <= bar[0] <= b]
            if n % 10 == 9:
                dump(RAW/"recent-bars.json", have)
                print(f"  {n+1}/{len(chunks)}", flush=True)
    finally:
        dump(RAW/"recent-bars.json", have)
    print(f"recent bars for {len(have)} markets", flush=True)


def abars_step(client, cap):
    have = load(RAW/"archive-bars.json", {})
    _, archive = sample()
    jobs = [(span(e, m), m["ticker"]) for e in archive[:cap] for m in e["picked"] if m["ticker"] not in have]
    print(f"archive bars: {len(have)} on disk, {len(jobs)} to fetch for the first {cap} events", flush=True)
    try:
        for n, ((a, b), t) in enumerate(jobs):
            data = read(client, f"/historical/markets/{t}/candlesticks", dict(start_ts=a, end_ts=b, period_interval=60))
            if data is None:
                log["no_bars"] += 1
            have[t] = compact((data or {}).get("candlesticks"))
            if n % 25 == 24:
                dump(RAW/"archive-bars.json", have)
            if n % 200 == 199:
                print(f"  {n+1}/{len(jobs)}", flush=True)
    finally:
        dump(RAW/"archive-bars.json", have)
    print(f"archive bars for {len(have)} markets", flush=True)


def dbars_step(client, cap):
    """Daily bars over the whole life, for events that lived 30+ days: recent by batch, archive one market per request (first `cap` events)."""
    have = load(RAW/"daily-bars.json", {})
    recent, archive = sample()
    hourly = load(RAW/"archive-bars.json", {})
    long_ = lambda e: e["close"]-min(m["open"] for m in e["picked"]) >= 30*86400
    jobs = sorted(((m["open"]//86400*86400-86400, max(m["close"], e["close"])//86400*86400+2*86400, m["ticker"]) for e in recent if long_(e)
                   for m in e["picked"] if m["ticker"] not in have), key=lambda j: (j[1], j[0]))
    chunks, cur = [], []
    for j in jobs:
        trial = cur+[j]
        days = (max(x[1] for x in trial)-min(x[0] for x in trial))//86400+2
        if cur and (len(trial) > 100 or len(trial)*days > 9000):
            chunks.append(cur)
            cur = []
        cur.append(j)
    chunks += [cur] if cur else []
    old = [(m["open"]//86400*86400-86400, max(m["close"], e["close"])//86400*86400+2*86400, m["ticker"]) for e in archive[:cap] if long_(e)
           for m in e["picked"] if m["ticker"] not in have and m["ticker"] in hourly]
    print(f"daily bars: {len(have)} on disk, {len(jobs)} recent markets in {len(chunks)} requests, {len(old)} archive markets", flush=True)
    try:
        for n, group in enumerate(chunks):
            data = read(client, "/markets/candlesticks", dict(market_tickers=",".join(x[2] for x in group), start_ts=min(x[0] for x in group),
                                                              end_ts=max(x[1] for x in group), period_interval=1440))
            if data is None:
                log["batch_refused"] += 1
                continue
            got = {x["market_ticker"]: compact(x.get("candlesticks")) for x in data.get("markets", [])}
            for a, b, t in group:
                have[t] = got.get(t, [])
        dump(RAW/"daily-bars.json", have)
        for n, (a, b, t) in enumerate(old):
            data = read(client, f"/historical/markets/{t}/candlesticks", dict(start_ts=a, end_ts=b, period_interval=1440))
            have[t] = compact((data or {}).get("candlesticks"))
            if n % 25 == 24:
                dump(RAW/"daily-bars.json", have)
    finally:
        dump(RAW/"daily-bars.json", have)
    print(f"daily bars for {len(have)} markets", flush=True)


# ---------------------------------------------------------------- simulation

def taker_fee(p, mult=1.0):
    return 7*mult*p/100*(1-p/100)


def maker_fee(p, mult=1.0):
    return 1.75*mult*p/100*(1-p/100)


def quote_at(bars, t, back=7*86400):
    """Index of the last bar ending at or before t (within `back`), or None."""
    i = int(np.searchsorted([b[0] for b in bars], t, side="right"))-1
    return i if i >= 0 and t-bars[i][0] <= back else None


def two_sided(bid, ask):
    return bid is not None and ask is not None and 1 <= bid < ask <= 99 and ask-bid >= 2


def event_rows(e, bars, t, back=7*86400, step=3600):
    """Quotes and orders for every picked outcome of one event at instant t. Returns (outcome rows, summary) or None when nothing is quoted."""
    rows = []
    for m in e["picked"]:
        b = bars.get(m["ticker"])
        if b is None or m["open"] > t or m["close"] <= t:
            continue
        i = quote_at(b, t, back) if b else None
        if i is None:
            rows.append(dict(m=m, bid=None, ask=None))
            continue
        bid, ask = b[i][1], b[i][2]
        later = [x for x in b[i+1:] if x[0] <= max(m["close"], e["close"])+step]
        r = dict(m=m, bid=bid if bid is not None else 0.0, ask=ask if ask is not None else 100.0, y=m["result"] == "yes", order=two_sided(bid, ask))
        if r["order"]:
            f = next((x for x in later if x[5] > 0 and x[4] is not None and x[4] > ask-1), None)
            yf = next((x for x in later if x[5] > 0 and x[3] is not None and x[3] < bid+1), None)
            r.update(fill=f is not None, yfill=yf is not None, fill_ts=f[0] if f else None, yfill_ts=yf[0] if yf else None,
                     touch=any(x[5] > 0 and x[4] is not None and x[4] >= ask-1 for x in later),
                     ytouch=any(x[5] > 0 and x[3] is not None and x[3] <= bid+1 for x in later),
                     vol_through=sum(x[5] for x in later if x[4] is not None and x[4] > ask-1),
                     yvol_through=sum(x[5] for x in later if x[3] is not None and x[3] < bid+1))
        rows.append(r)
    quoted = [r for r in rows if r["bid"] is not None]
    if not quoted:
        return None
    mult = e["mult"]
    edge = sum(max(0.0, r["bid"]-taker_fee(100-r["bid"], mult)) for r in quoted if r["bid"] >= 1)
    legs = [r for r in quoted if r["bid"] >= 1 and r["bid"] > taker_fee(100-r["bid"], mult)]
    asks = [r["ask"] for r in quoted]
    summary = dict(open=len(rows), quoted=len(quoted), bids=sum(r["bid"] for r in quoted), mids=sum((r["bid"]+r["ask"])/2 for r in quoted if r["ask"] <= 99),
                   all_asked=all(a <= 99 for a in asks), asks=sum(asks), a_edge=edge-100, a_legs=len(legs),
                   a_cost=sum(100-r["bid"]+taker_fee(100-r["bid"], mult) for r in legs),
                   buy_all_yes=100-sum(a+taker_fee(a, mult) for a in asks) if all(a <= 99 for a in asks) and len(quoted) == len(rows) == e["n"] else None)
    return quoted, summary


def build(clock_hours, daily=False):
    """Orders at `clock_hours` before the event's close (hourly bars), or `clock_hours` after the event's listing when daily (daily bars)."""
    recent, archive = sample()
    rb, ab = load(RAW/"recent-bars.json", {}), load(RAW/"archive-bars.json", {})
    db = load(RAW/"daily-bars.json", {}) if daily else None
    orders, events = [], []
    for e in recent+archive:
        if daily:
            bars = db
            start = min(m["open"] for m in e["picked"])
            if e["close"]-start < 30*86400:
                continue
            t = -(-(start+clock_hours*3600)//86400)*86400
            if t >= e["close"]-2*86400:
                continue
        else:
            bars = rb if e["tier"] == "recent" else ab
            t = e["close"]-clock_hours*3600
        if not all(m["ticker"] in bars for m in e["picked"]):
            continue
        if not daily and t < min(m["open"] for m in e["picked"])+3600:
            continue
        got = event_rows(e, bars, t, back=30*86400 if daily else 7*86400, step=86400 if daily else 3600)
        if got is None:
            continue
        quoted, summary = got
        rank = {id(r): k for k, r in enumerate(sorted(quoted, key=lambda r: -((r["bid"]+r["ask"])/2 if r["ask"] <= 99 else r["bid"])))}
        ev = dict(e, t=t, **summary)
        ev.pop("markets"), ev.pop("picked")
        ev["thinned"] = e["n"] > len(e["picked"])
        ev["ge"] = len(events)
        events.append(ev)
        for r in quoted:
            settle = r["m"]["settle"] or max(r["m"]["close"], e["close"])
            orders.append(dict(ge=ev["ge"], rank=rank[id(r)], bid=r["bid"], ask=r["ask"], y=r["y"], order=r["order"], fill=r.get("fill", False),
                               yfill=r.get("yfill", False), touch=r.get("touch", False), ytouch=r.get("ytouch", False),
                               hold=(settle-r["fill_ts"])/86400 if r.get("fill_ts") else np.nan, yhold=(settle-r["yfill_ts"])/86400 if r.get("yfill_ts") else np.nan,
                               vol=r.get("vol_through", 0.0), yvol=r.get("yvol_through", 0.0), ticker=r["m"]["ticker"], t=t))
    o = {k: np.array([x[k] for x in orders]) for k in orders[0]} if orders else {}
    if not orders:
        return o, events
    mk = np.array([events[g]["maker_fee"] for g in o["ge"]])
    mult = np.array([events[g]["mult"] for g in o["ge"]])
    o["price"] = 100-(o["ask"]-1)
    o["net"] = 100*(~o["y"])-o["price"]-np.where(mk, maker_fee(o["price"], mult), 0)
    o["yprice"] = o["bid"]+1
    o["ynet"] = 100*o["y"]-o["yprice"]-np.where(mk, maker_fee(o["yprice"], mult), 0)
    o["mid"] = (o["bid"]+o["ask"])/2
    for key in ("group", "tier", "close", "series", "early"):
        o[key] = np.array([events[g][key] for g in o["ge"]])
    return o, events


def boot(parts, codes, draws=2000):
    """Event bootstrap of sum(num)/sum(den), or the difference of two such ratios. parts = [(num, den, mask), ...] -> est, lo, hi, clusters."""
    used = np.zeros(len(codes), bool)
    for _, _, mask in parts:
        used |= mask
    if not used.any():
        return None
    _, inv = np.unique(codes[used], return_inverse=True)
    k = inv.max()+1
    sums = []
    for num, den, mask in parts:
        w = mask[used]
        sums.append((np.bincount(inv, num[used]*w, k), np.bincount(inv, den[used]*w, k)))
        if sums[-1][1].sum() == 0:
            return None
    est = [a.sum()/b.sum() for a, b in sums]
    est = est[0]-est[1] if len(est) == 2 else est[0]
    if k < 8:
        return float(est), None, None, int(k)
    pick = RNG.integers(0, k, (draws, k))
    v = [a[pick].sum(-1)/np.where(b[pick].sum(-1) == 0, np.nan, b[pick].sum(-1)) for a, b in sums]
    v = v[0]-v[1] if len(v) == 2 else v[0]
    lo, hi = np.nanpercentile(v, [2.5, 97.5])
    return float(est), float(lo), float(hi), int(k)


def show(v, scale=1.0, width=5):
    if v is None:
        return f"{'n/a':>{width+16}}"
    if v[1] is None:
        return f"{scale*v[0]:+{width}.1f} [{'n/a':^14}]"
    return f"{scale*v[0]:+{width}.1f} [{scale*v[1]:+6.1f},{scale*v[2]:+6.1f}]"


def halves(o, mask):
    """Split the events under mask into chronological halves by close."""
    order = sorted({(c, g) for c, g in zip(o["close"][mask], o["ge"][mask])})
    first = {g for _, g in order[:len(order)//2]}
    h1 = mask & np.isin(o["ge"], list(first))
    return h1, mask & ~h1


def maker_line(label, o, mask, mode="through"):
    one = np.ones(len(mask))
    orders = mask & o["order"]
    f, yf = orders & o["touch" if mode == "touch" else "fill"], orders & o["ytouch" if mode == "touch" else "yfill"]
    if orders.sum() < 10:
        return None, None
    r = dict(events=len(set(o["ge"][mask])), orders=int(orders.sum()), fills=int(f.sum()), yfills=int(yf.sum()),
             events_filled=len(set(o["ge"][f])), net=boot([(o["net"], one, f)], o["ge"]), ret=boot([(o["net"], o["price"], f)], o["ge"]),
             ynet=boot([(o["ynet"], one, yf)], o["ge"]), yret=boot([(o["ynet"], o["yprice"], yf)], o["ge"]),
             diff=boot([(o["net"], one, f), (o["ynet"], one, yf)], o["ge"]) if f.any() and yf.any() else None,
             unfilled=float(o["net"][orders & ~f].mean()) if (orders & ~f).any() else None,
             hold=float(np.nanmedian(o["hold"][f])) if f.any() else None, paid=float(o["price"][f].mean()) if f.any() else None,
             lost=float(o["y"][f].mean()) if f.any() else None)
    hold = np.nanmean(o["hold"][f]) if f.any() else np.nan
    num = lambda v: "   n/a" if v is None else f"{v:+6.1f}"
    text = (f"{label:34} {r['events']:5} {r['orders']:6} {r['fills']:5} {show(r['net'])} {show(r['ret'], 100)} {num(r['unfilled'])} "
            f"{(r['paid'] or 0):5.1f} {100*(r['lost'] or 0):5.1f} {hold:6.2f} | {r['yfills']:5} {show(r['ynet'])} {show(r['yret'], 100)} | {show(r['diff'])}")
    return r, text


MAKER_HEAD = (f"{'':34} {'ev':>5} {'orders':>6} {'fills':>5} {'NO net c / fill':>21} {'NO return %':>21} {'unfill':>6} {'paid':>5} {'YES%':>5} {'hold d':>6} | "
              f"{'fills':>5} {'MIRROR net c / fill':>21} {'MIRROR return %':>21} | {'NO minus mirror c':>21}")
GROUP_ORDER = ("Politics+Elections", "Entertainment", "Sports winners", "Weather ranges", "Financials+Crypto", "Economics", "Other")


# ---------------------------------------------------------------- trades (capacity)

TRADE_SAMPLE = 160


def trades_step(client, hours=24):
    o, events = build(hours)
    have = load(RAW/"trades.json", {})
    pick = [i for i in np.flatnonzero(o["order"] & (o["tier"] == "recent"))]
    pick.sort(key=lambda i: pick_hash("trade", o["ticker"][i]))
    todo = [i for i in pick[:TRADE_SAMPLE] if o["ticker"][i] not in have]
    print(f"trades: {len(have)} on disk, {len(todo)} to fetch", flush=True)
    try:
        for i in todo:
            rows, cursor, pages = [], "", 0
            while pages < 3:
                data = read(client, "/markets/trades", dict(ticker=o["ticker"][i], limit=1000, min_ts=int(o["t"][i]), **({"cursor": cursor} if cursor else {}))) or {}
                pages += 1
                for t in data.get("trades") or []:
                    rows.append([int(timestamp(t["created_time"])), round(100*float(t["yes_price_dollars"]), 2), float(t.get("count_fp", t.get("count", 0))),
                                 1 if (t.get("taker_side") or t.get("taker_outcome_side")) == "yes" else 0])
                cursor = data.get("cursor")
                if not cursor or not data.get("trades"):
                    break
            have[o["ticker"][i]] = dict(complete=not cursor, t=int(o["t"][i]), trades=rows)
    finally:
        dump(RAW/"trades.json", have)
    print(f"trades for {len(have)} markets", flush=True)


# ---------------------------------------------------------------- report

def report():
    OUT.mkdir(parents=True, exist_ok=True)
    lines, summary = [], {}

    def say(text=""):
        if text is not None:
            print(text, flush=True)
            lines.append(text)
    uni = universe()
    recent, archive = sample()
    events_disk = load(RAW/"events.json", {})
    say("(0) COVERAGE")
    say(f"    candidate series {len(candidates())}; read {len(events_disk)}; with a mutually exclusive event {sum(any(e['me'] for e in v['events']) for v in events_disk.values())};"
        f" eligible settled events (ME, 4+ outcomes, 5,000+ contracts, closing 2025-01-01..2026-11-18): {len(uni)}"
        f" ({sum(e['tier'] == 'recent' for e in uni.values())} recent, {sum(e['tier'] == 'archive' for e in uni.values())} archive);"
        f" sampled for bars: {len(recent)} recent, {len(archive)} archive in order.")
    yes = Counter(e["yes"] for e in uni.values())
    say(f"    YES outcomes per eligible event: none {yes[0]}, one {yes[1]}, two or more {sum(v for k, v in yes.items() if k > 1)}"
        f"  (two or more would break the NO set: {[e['event'] for e in uni.values() if e['yes'] > 1][:12]})")
    o24, ev24 = build(24)
    o168, ev168 = build(168)
    o1, ev1 = build(1)
    if not ev24:
        say("no bars yet")
        return
    say(f"    events with bars for every picked outcome and a quote 24h before close: {len(ev24)}; 7d before: {len(ev168)}; 1h before: {len(ev1)}")
    say(f"    {'group':20} {'tier':8} {'events':>6} {'series':>6} {'outcomes/ev':>11} {'thinned':>7} {'can close early':>15} {'first close':>11} {'last close':>11}")
    day = lambda t: datetime.fromtimestamp(t, timezone.utc).strftime("%Y-%m-%d")
    for g in GROUP_ORDER:
        for tier in ("recent", "archive"):
            sel = [e for e in ev24 if e["group"] == g and e["tier"] == tier]
            if sel:
                say(f"    {g:20} {tier:8} {len(sel):6} {len({e['series'] for e in sel}):6} {np.mean([e['n'] for e in sel]):11.1f} {sum(e['thinned'] for e in sel):7} "
                    f"{sum(e['early'] for e in sel):15} {day(min(e['close'] for e in sel)):>11} {day(max(e['close'] for e in sel)):>11}")

    say("\n(1) OVERROUND at fixed times before the event's close. Sums over the outcomes still open and quoted (cents). 'bids' = what a NO taker faces (100 - YES bid"
        "\n    is the NO ask); 'asks' only for events where every quoted outcome has an offer. Archive events carry at most 10 outcomes, so their sums are LOWER bounds;"
        "\n    the recent tier (every outcome) is the clean read.")
    say(f"    {'':30} {'events':>6} | {'sum YES bids: mean':>18} {'median':>6} {'>100':>5} | {'sum mids':>8} {'median':>6} | {'all asked':>9} {'sum asks: mean':>14} {'median':>6} {'<100':>5}")
    for label, evs in (("24h", ev24), ("7d", ev168), ("1h", ev1)):
        for g in ("ALL",)+GROUP_ORDER:
            for tier in ("recent", "archive"):
                sel = [e for e in evs if g in ("ALL", e["group"]) and e["tier"] == tier]
                if len(sel) < 5:
                    continue
                bids, mids = np.array([e["bids"] for e in sel]), np.array([e["mids"] for e in sel])
                asked = np.array([e["asks"] for e in sel if e["all_asked"]])
                say(f"    {label+' '+g+' '+tier:30} {len(sel):6} | {bids.mean():18.1f} {np.median(bids):6.1f} {100*(bids > 100).mean():4.0f}% | {mids.mean():8.1f} {np.median(mids):6.1f} | "
                    f"{len(asked):9} {asked.mean() if len(asked) else float('nan'):14.1f} {np.median(asked) if len(asked) else float('nan'):6.1f} "
                    f"{100*(asked < 100).mean() if len(asked) else float('nan'):4.0f}%")
                summary.setdefault("overround", {})[f"{label}|{g}|{tier}"] = dict(events=len(sel), bids=float(bids.mean()), bids_median=float(np.median(bids)),
                                                                                  mids=float(mids.mean()), asks=float(asked.mean()) if len(asked) else None)

    say("\n(2) VERSION A, TAKER. Locked profit of buying NO at the ask on every outcome whose YES bid beats its fee = sum(YES bid - fee) - 100, cents per set of"
        "\n    one contract each (at most one outcome can settle YES). Also the opposite set where every outcome has an offer: 100 - sum(YES ask + fee), which"
        "\n    additionally needs exactly one outcome to settle YES (counted only when every outcome is open, quoted and offered; 'won' = one did settle YES). Hourly quote closes; displayed size is NOT in the history.")
    say(f"    {'':30} {'events':>6} | {'NO set > 0':>10} {'mean edge c when > 0':>20} {'max':>6} {'return % on cost':>16} | {'YES set: events':>15} {'> 0':>5} {'mean c':>7} {'won':>4}")
    for label, evs in (("24h", ev24), ("7d", ev168), ("1h", ev1)):
        for g in ("ALL",)+GROUP_ORDER:
            for tier in ("recent", "archive"):
                sel = [e for e in evs if g in ("ALL", e["group"]) and e["tier"] == tier]
                if len(sel) < 5:
                    continue
                pos = [e for e in sel if e["a_edge"] > 0]
                ys = [e["buy_all_yes"] for e in sel if e["buy_all_yes"] is not None and not e["thinned"]]
                ypos = [v for v in ys if v > 0]
                ywon = sum(e["yes"] == 1 for e in sel if e["buy_all_yes"] is not None and not e["thinned"] and e["buy_all_yes"] > 0)
                say(f"    {label+' '+g+' '+tier:30} {len(sel):6} | {len(pos):4} {100*len(pos)/len(sel):4.1f}% {np.mean([e['a_edge'] for e in pos]) if pos else float('nan'):20.2f} "
                    f"{max([e['a_edge'] for e in pos], default=float('nan')):6.2f} {100*np.mean([e['a_edge']/e['a_cost'] for e in pos]) if pos else float('nan'):16.2f} | "
                    f"{len(ys):15} {len(ypos):5} {np.mean(ypos) if ypos else float('nan'):7.2f} {ywon:4}")
                summary.setdefault("taker", {})[f"{label}|{g}|{tier}"] = dict(events=len(sel), positive=len(pos), mean=float(np.mean([e["a_edge"] for e in pos])) if pos else None)
    pos = sorted((e for e in ev24+ev168+ev1 if e["a_edge"] > 0), key=lambda e: -e["a_edge"])
    say("    largest NO-set edges: "+"; ".join(f"{e['event']} {e['a_edge']:+.1f}c ({e['a_legs']} legs, {e['yes']} YES)" for e in pos[:10]))
    scan_hours(say, summary)
    live(say, summary)

    say("\n(3) VERSION B, MAKER. A NO bid one cent above the best NO bid on every outcome quoted two-sided and >= 2c wide at the clock, resting to the close, held to"
        "\n    settlement; filled by a later hourly bar printing strictly through it. MIRROR = a YES bid at bid + 1. One contract per order. 'unfill' = net per contract the"
        "\n    unfilled NO orders would have made; 'paid' = mean NO price of fills; 'YES%' = share of NO fills that lost; 'hold d' = mean days from fill to settlement."
        "\n    Intervals: 95%, bootstrap over EVENTS.")
    for label, o in (("24h before close (headline)", o24), ("7d before close", o168)):
        if not len(o):
            continue
        say(f"\n  CLOCK {label}")
        say("  "+MAKER_HEAD)
        everything = np.ones(len(o["ge"]), bool)
        for name, mask in [("ALL", everything), ("ALL recent tier", o["tier"] == "recent"), ("ALL archive tier", o["tier"] == "archive")]+[
                (g, o["group"] == g) for g in GROUP_ORDER]:
            if not mask.any():
                continue
            r, text = maker_line(name, o, mask)
            say(text and "  "+text)
            summary.setdefault("maker", {})[f"{label}|{name}"] = r
            if r is None:
                continue
            h1, h2 = halves(o, mask)
            for hn, hm in (("  first half", h1), ("  second half", h2)):
                r, text = maker_line(hn, o, hm)
                say(text and "  "+text)
                summary["maker"][f"{label}|{name}|{hn.strip()}"] = r
        say("  robustness (decided after looking at the layout, before the numbers were interpreted):")
        for name, mask, mode in (("fills at the price too (touch)", everything, "touch"), ("scheduled close (cannot close early)", ~o["early"], "through"),
                                 ("can close early", o["early"], "through"), ("favourite (rank 1)", o["rank"] == 0, "through"),
                                 ("rank 2-3", (o["rank"] >= 1) & (o["rank"] <= 2), "through"), ("rank 4+", o["rank"] >= 3, "through"),
                                 ("YES mid < 10", o["mid"] < 10, "through"), ("YES mid 10-40", (o["mid"] >= 10) & (o["mid"] < 40), "through"),
                                 ("YES mid 40+", o["mid"] >= 40, "through"), ("without weather", o["group"] != "Weather ranges", "through")):
            r, text = maker_line(name, o, mask, mode)
            say(text and "  "+text)
            summary["maker"][f"{label}|robust|{name}"] = r
        per_event(o, say, summary, label)
    long_dated(say, summary)

    say("\n(4) CALIBRATION BY RANK within the event, 24h before close (rank by YES midpoint among quoted outcomes; outcomes with an offer and a spread <= 10c)."
        "\n    'freq - mid' negative = YES overpriced. Interval: 95% over events.")
    calibration(o24, say, summary)
    capacity(o24, ev24, say, summary)
    dump(OUT/"summary.json", summary)
    (OUT/"report.txt").write_text("\n".join(lines)+"\n")


def scan_hours(say, summary):
    """Version A at every hour of the last 7 days before close (not the last hour): how often is the NO set locked in profit?"""
    recent, archive = sample()
    rb, ab = load(RAW/"recent-bars.json", {}), load(RAW/"archive-bars.json", {})
    out = defaultdict(lambda: [0, 0, 0, [], set(), set()])
    for e in recent+archive:
        bars = rb if e["tier"] == "recent" else ab
        if not all(m["ticker"] in bars for m in e["picked"]):
            continue
        start = max(min(m["open"] for m in e["picked"])+3600, e["close"]-7*86400)
        for t in range(-(-start//3600)*3600, e["close"]-3600, 3600):
            got = quick(e, bars, t)
            if got is None:
                continue
            for key in ("ALL "+e["tier"], e["group"]+" "+e["tier"]):
                c = out[key]
                c[0] += 1
                c[4].add(e["event"])
                if got > 0:
                    c[1] += 1
                    c[3].append(got)
                    c[5].add(e["event"])
    say("    Every hour of the last 7 days before close except the last (event-hours with a quote):")
    say(f"    {'':30} {'events':>6} {'event-hours':>11} {'NO set > 0':>10} {'share':>6} {'events ever':>11} {'mean c':>7} {'median c':>8} {'max c':>6}")
    for key in sorted(out):
        c = out[key]
        say(f"    {key:30} {len(c[4]):6} {c[0]:11} {c[1]:10} {100*c[1]/max(1, c[0]):5.2f}% {len(c[5]):11} {np.mean(c[3]) if c[3] else float('nan'):7.2f} "
            f"{np.median(c[3]) if c[3] else float('nan'):8.2f} {max(c[3], default=float('nan')):6.2f}")
        summary.setdefault("taker_hours", {})[key] = dict(events=len(c[4]), hours=c[0], positive=c[1], events_ever=len(c[5]), mean=float(np.mean(c[3])) if c[3] else None)


_ends = {}


def quick(e, bars, t):
    """sum(YES bid - fee) - 100 over open outcomes at t, or None when nothing is quoted."""
    total, any_ = 0.0, False
    for m in e["picked"]:
        b = bars[m["ticker"]]
        if not b or m["open"] > t or m["close"] <= t:
            continue
        ends = _ends.get(m["ticker"])
        if ends is None:
            ends = _ends[m["ticker"]] = np.array([x[0] for x in b])
        i = int(np.searchsorted(ends, t, side="right"))-1
        if i < 0 or t-ends[i] > 7*86400:
            continue
        any_ = True
        bid = b[i][1] or 0.0
        if bid >= 1:
            total += max(0.0, bid-taker_fee(100-bid, e["mult"]))
    return total-100 if any_ else None


def live(say, summary):
    """The open mutually exclusive events as listed at discovery: sum of YES bids and the size displayed at those bids."""
    meta = {s["ticker"]: s for s in load(RAW/"series.json", [])}
    rows = []
    for series, v in load(RAW/"events.json", {}).items():
        for e in v["events"]:
            ms = [m for m in e["markets"] if m["status"] == "active"]
            if not e["me"] or len(ms) < MIN_OUTCOMES or any(m["result"] == "yes" for m in e["markets"]) or len(ms) != len([m for m in e["markets"] if m["status"] in ("active", "initialized")]):
                continue
            mult = float(meta[series]["fee_multiplier"] if meta[series]["fee_multiplier"] is not None else 1)
            legs = [m for m in ms if m["bid"] and m["bid"] >= 1 and m["bid"] > taker_fee(100-m["bid"], mult)]
            edge = sum(m["bid"]-taker_fee(100-m["bid"], mult) for m in legs)-100
            rows.append(dict(event=e["event"], group=GROUPS[meta[series]["category"]], n=len(ms), bids=sum(m["bid"] or 0 for m in ms),
                             asks=sum(m["ask"] if m["ask"] else 100 for m in ms), all_asked=all(m["ask"] and m["ask"] <= 99 for m in ms), edge=edge,
                             size=min((m["bid_size"] for m in legs), default=0), volume=sum(m["volume"] for m in ms)))
    say("\n    LIVE SNAPSHOT (listing quotes at discovery, 2026-11-19; the only place displayed size is visible). Open mutually exclusive events with 4+ active outcomes:")
    say(f"    {'':22} {'events':>6} {'sum YES bids mean':>17} {'median':>6} {'>100':>5} {'NO set > 0 after fee':>20} {'sum asks median (all offered)':>30}")
    for g in ("ALL",)+GROUP_ORDER:
        sel = [r for r in rows if g in ("ALL", r["group"])]
        if len(sel) < 3:
            continue
        bids = np.array([r["bids"] for r in sel])
        asked = [r["asks"] for r in sel if r["all_asked"]]
        say(f"    {g:22} {len(sel):6} {bids.mean():17.1f} {np.median(bids):6.1f} {100*(bids > 100).mean():4.0f}% {sum(r['edge'] > 0 for r in sel):20} "
            f"{np.median(asked) if asked else float('nan'):24.1f} ({len(asked)})")
    pos = sorted((r for r in rows if r["edge"] > 0), key=lambda r: -r["edge"])
    for r in pos[:15]:
        say(f"      {r['event']:34} {r['n']:3} outcomes  sum bids {r['bids']:6.1f}  locked {r['edge']:+5.2f}c per set  smallest displayed bid size {r['size']:8.0f} contracts")
    summary["live"] = dict(events=len(rows), positive=len(pos), rows=pos[:30])


def per_event(o, say, summary, label):
    """Per-event result of the whole basket of fills (the unit a trader experiences)."""
    f = o["order"] & o["fill"]
    yf = o["order"] & o["yfill"]
    k = int(o["ge"].max())+1
    net, cost, n = np.bincount(o["ge"], o["net"]*f, k), np.bincount(o["ge"], o["price"]*f, k), np.bincount(o["ge"], f, k)
    ynet, yn = np.bincount(o["ge"], o["ynet"]*yf, k), np.bincount(o["ge"], yf, k)
    has = n > 0
    if has.sum() < 5:
        return
    lostyes = np.bincount(o["ge"], (o["y"] & f), k)
    say(f"  per event (one contract per order): events with a NO fill {int(has.sum())}; fills per such event mean {n[has].mean():.1f}; net per event mean {net[has].mean():+.1f}c, "
        f"median {np.median(net[has]):+.1f}c, worst {net[has].min():+.1f}c, best {net[has].max():+.1f}c; events that lose {100*(net[has] < 0).mean():.0f}%; "
        f"events where the winner was among the NO fills {100*(lostyes[has] > 0).mean():.0f}%; return on money put in {100*net[has].sum()/cost[has].sum():+.2f}%."
        f" Mirror: events with a fill {int((yn > 0).sum())}, net per event mean {ynet[yn > 0].mean():+.1f}c.")
    summary.setdefault("per_event", {})[label] = dict(events=int(has.sum()), mean=float(net[has].mean()), median=float(np.median(net[has])), worst=float(net[has].min()),
                                                      lose=float((net[has] < 0).mean()))


def long_dated(say, summary):
    """Long-dated clock on daily bars: orders 7 days after the listing in events that lived 30+ days, resting to the close."""
    o, events = build(7*24, daily=True)
    if not len(o) or len(events) < 5:
        say("\n  LONG-DATED CLOCK: no daily bars on disk")
        return
    say("\n  LONG-DATED CLOCK (daily bars; events that lived 30+ days; order at the first daily close 7 days after the listing, resting to the close; a fill needs a LATER"
        "\n  daily bar printing through the price). Hold in days is fill to settlement: the capital is tied up that long.")
    say("  "+MAKER_HEAD)
    everything = np.ones(len(o["ge"]), bool)
    for name, mask in [("ALL long-dated", everything)]+[(g, o["group"] == g) for g in GROUP_ORDER]:
        if not mask.any():
            continue
        r, text = maker_line(name, o, mask)
        say(text and "  "+text)
        summary.setdefault("maker_long", {})[name] = r
        if name == "ALL long-dated" and r:
            for hn, hm in zip(("  first half", "  second half"), halves(o, mask)):
                r2, text = maker_line(hn, o, hm)
                say(text and "  "+text)
                summary["maker_long"][hn.strip()] = r2
    f = o["order"] & o["fill"]
    if f.any():
        hold = o["hold"][f]
        ret = o["net"][f].sum()/o["price"][f].sum()
        say(f"  holding time of NO fills: mean {np.nanmean(hold):.0f} days, median {np.nanmedian(hold):.0f}, 90% {np.nanpercentile(hold, 90):.0f}; return on money put in "
            f"{100*ret:+.2f}% over a mean {np.nanmean(hold):.0f} days = {100*ret*365/max(1, np.nanmean(hold)):+.1f}% a year simple (capital-days weighted: "
            f"{100*o['net'][f].sum()/np.nansum(o['price'][f]*hold)*365:+.1f}% a year).")
    per_event(o, say, summary, "long-dated")


def calibration(o, say, summary):
    ok = (o["ask"] <= 99) & (o["ask"]-o["bid"] <= 10)
    one = np.ones(len(ok))
    y = 100.0*o["y"]
    say(f"    {'':26} {'outcomes':>8} {'events':>6} {'mean mid':>8} {'mean bid':>8} {'mean ask':>8} {'YES %':>6} {'freq - mid c':>22} {'freq - bid c':>12} {'freq - ask c':>12}")
    cells = [("rank 1 (favourite)", o["rank"] == 0), ("rank 2", o["rank"] == 1), ("rank 3", o["rank"] == 2), ("rank 4-6", (o["rank"] >= 3) & (o["rank"] <= 5)),
             ("rank 7+", o["rank"] >= 6), ("all ranks", o["rank"] >= 0), ("mid < 3", o["mid"] < 3), ("mid 3-10", (o["mid"] >= 3) & (o["mid"] < 10)),
             ("mid 10-25", (o["mid"] >= 10) & (o["mid"] < 25)), ("mid 25-50", (o["mid"] >= 25) & (o["mid"] < 50)), ("mid 50-75", (o["mid"] >= 50) & (o["mid"] < 75)),
             ("mid 75+", o["mid"] >= 75)]
    for tier in ("recent", "archive", "both"):
        say(f"    -- {tier} tier" + (" (every outcome)" if tier == "recent" else " (top 10 outcomes by volume)" if tier == "archive" else ""))
        for name, mask in cells:
            m = mask & ok & ((o["tier"] == tier) | (tier == "both"))
            if m.sum() < 20:
                continue
            gap = boot([(y-o["mid"], one, m)], o["ge"])
            say(f"    {name:26} {int(m.sum()):8} {len(set(o['ge'][m])):6} {o['mid'][m].mean():8.1f} {o['bid'][m].mean():8.1f} {o['ask'][m].mean():8.1f} {y[m].mean():6.1f} "
                f"{show(gap)} {(y[m]-o['bid'][m]).mean():+12.1f} {(y[m]-o['ask'][m]).mean():+12.1f}")
            summary.setdefault("calibration", {})[f"{tier}|{name}"] = dict(n=int(m.sum()), mid=float(o["mid"][m].mean()), yes=float(y[m].mean()), gap=gap)
    for g in GROUP_ORDER:
        parts = []
        for name, mask in (("fav", o["rank"] == 0), ("2-3", (o["rank"] >= 1) & (o["rank"] <= 2)), ("4+", o["rank"] >= 3)):
            m = mask & ok & (o["group"] == g)
            if m.sum() >= 20:
                parts.append(f"{name}: {show(boot([(y-o['mid'], one, m)], o['ge']))} (n {int(m.sum())}, mid {o['mid'][m].mean():.1f})")
        if parts:
            say(f"    {g:20} freq - mid by rank  "+"   ".join(parts))


def capacity(o, events, say, summary):
    say("\n(5) CAPACITY of version B (24h clock).")
    recent = o["order"] & (o["tier"] == "recent")
    f = recent & o["fill"]
    days = (NOW-CUTOFF)/86400
    uni = universe()
    all_recent = sum(e["tier"] == "recent" for e in uni.values())
    got = len({g for g in o["ge"][o["tier"] == "recent"]})
    say(f"    recent tier: {got} events with bars of {all_recent} eligible over {days:.0f} days; {int(recent.sum())} orders, {int(f.sum())} filled.")
    say(f"    upper bound from bars (all volume of every later hour that printed through the price, which includes prints not through it): "
        f"{o['vol'][f].sum():,.0f} contracts over the sample; median per filled order {np.median(o['vol'][f]) if f.any() else 0:,.0f}.")
    have = load(RAW/"trades.json", {})
    rows = []
    for i in np.flatnonzero(recent):
        v = have.get(o["ticker"][i])
        if not v or not v["complete"] or v["t"] != int(o["t"][i]):
            continue
        offer, ybid = o["ask"][i]-1, o["bid"][i]+1
        after = [t for t in v["trades"] if t[0] > o["t"][i]+0]
        rows.append((sum(t[2] for t in after if t[1] > offer), sum(t[2] for t in after if t[1] > offer and t[3] == 1), o["net"][i], o["price"][i], o["fill"][i],
                     sum(t[2] for t in after if t[1] < ybid), o["ynet"][i], o["yprice"][i], any(t[1] > offer for t in after)))
    if len(rows) >= 20:
        a = np.array(rows, float)
        thr = a[:, 0]
        say(f"    trade record for {len(a)} sampled recent orders (every trade from the clock to the close): bar-rule fill {100*a[:, 4].mean():.0f}%, tape fill {100*a[:, 8].mean():.0f}%,"
            f" agree {100*(a[:, 4] == a[:, 8]).mean():.0f}%; contracts through the price per order: mean {thr.mean():,.0f}, median of filled {np.median(thr[thr > 0]) if (thr > 0).any() else 0:,.0f},"
            f" >=10 in {100*(thr >= 10).mean():.0f}%, >=100 in {100*(thr >= 100).mean():.0f}% of orders.")
        scale = recent.sum()/len(a)*all_recent/max(1, got)*30/days
        for cap in (10, 100, 1000, None):
            q = thr if cap is None else np.minimum(cap, thr)
            yq = a[:, 5] if cap is None else np.minimum(cap, a[:, 5])
            if q.sum() == 0:
                continue
            say(f"      order size {'unlimited' if cap is None else cap:>9}: filled {q.mean():8.1f} contracts/order, size-weighted NO net {(q*a[:, 2]).sum()/q.sum():+5.2f}c "
                f"({100*(q*a[:, 2]).sum()/(q*a[:, 3]).sum():+5.2f}% on money in); mirror {(yq*a[:, 6]).sum()/max(1e-9, yq.sum()):+5.2f}c; scaled to every eligible event: "
                f"{q.mean()*scale:12,.0f} contracts/month, ${(q*a[:, 3]).mean()*scale/100:12,.0f}/month put in; at 25%: ${0.25*(q*a[:, 3]).mean()*scale/100:10,.0f}/month in, "
                f"${0.25*(q*a[:, 2]).mean()*scale/100:+10,.0f}/month net")
            summary.setdefault("capacity", {})[str(cap)] = dict(contracts_month=float(q.mean()*scale), dollars_month=float((q*a[:, 3]).mean()*scale/100),
                                                                net_month_25=float(0.25*(q*a[:, 2]).mean()*scale/100))
        say(f"      (scale: {recent.sum()} orders in {got} sampled events -> {all_recent} eligible events in {days:.0f} days; series beyond the candidate list are not counted)")
    else:
        say(f"    trade record on disk for {len(rows)} sampled orders: not enough for a tape-based capacity figure")


# ================================================================ AUDIT ADDITIONS (everything above is the original, unmodified except sys.path)
AUDIT_OUT = ROOT / "data/processed/audit/overround"


def verdict(o, mask=None):
    """The original's own evidence on a set of orders: NO return on fills, event bootstrap, chronological halves."""
    mask = np.ones(len(o["ge"]), bool) if mask is None else mask
    f = mask & o["order"] & o["fill"]
    yf = mask & o["order"] & o["yfill"]
    one = np.ones(len(mask))
    h1, h2 = halves(o, mask)
    out = {}
    for name, m in (("all", mask), ("h1", h1), ("h2", h2)):
        out[name] = boot([(o["net"], o["price"], f & m)], o["ge"])
        out[name+"_diff"] = boot([(o["net"], one, f & m), (o["ynet"], one, yf & m)], o["ge"]) if (f & m).any() and (yf & m).any() else None
    out["fills"] = int(f.sum())
    return out


def renet(o, events, y):
    mk = np.array([events[g]["maker_fee"] for g in o["ge"]])
    mult = np.array([events[g]["mult"] for g in o["ge"]])
    o = dict(o, y=y)
    o["net"] = 100*(~y)-o["price"]-np.where(mk, maker_fee(o["price"], mult), 0)
    o["ynet"] = 100*y-o["yprice"]-np.where(mk, maker_fee(o["yprice"], mult), 0)
    return o


def plant_probs(o, events, edge, target=None):
    """Winner probabilities per outcome such that the NO-fill rule has TRUE expected return `edge` (sum net / sum price).

    Filled NO orders get P(YES) = theta * breakeven, breakeven = (offer - fee)/100; the rest of each event's mass goes to its other
    quoted outcomes in proportion to their midpoints (or to 'nobody wins'). theta is solved so the expectation is exactly `edge`.
    `target` restricts the tilted orders (default: every filled order)."""
    mk = np.array([events[g]["maker_fee"] for g in o["ge"]])
    mult = np.array([events[g]["mult"] for g in o["ge"]])
    fee = np.where(mk, maker_fee(o["price"], mult), 0)
    f = o["order"] & o["fill"]
    tgt = f if target is None else f & target
    k = int(o["ge"].max())+1
    brk = np.where(f, (100-o["price"]-fee)/100, 0.0)

    def probs(theta):
        q = np.where(tgt, theta*brk, brk)*f
        tot = np.bincount(o["ge"], q, k)
        q = q/np.maximum(1.0, tot)[o["ge"]]
        left = 1-np.bincount(o["ge"], q, k)
        w = np.where(f, 0.0, np.maximum(o["mid"], 0.0))
        wt = np.bincount(o["ge"], w, k)
        q = q+np.where(wt[o["ge"]] > 0, w/np.where(wt[o["ge"]] > 0, wt[o["ge"]], 1)*left[o["ge"]], 0)
        return q

    def expect(theta):
        q = probs(theta)
        return ((100*(1-q)-o["price"]-fee)*tgt).sum()/(o["price"]*tgt).sum()
    lo, hi = 0.0, 1.5
    for _ in range(60):
        mid = (lo+hi)/2
        lo, hi = (lo, mid) if expect(mid) < edge else (mid, hi)
    return probs(lo), expect(lo), lo


def draw(o, q, rng):
    """One winner per event (or none) from q."""
    y = np.zeros(len(q), bool)
    order = np.argsort(o["ge"], kind="stable")
    ge = o["ge"][order]
    starts = np.flatnonzero(np.r_[True, ge[1:] != ge[:-1]])
    ends = np.r_[starts[1:], len(ge)]
    u = rng.random(len(starts))
    for a, b, x in zip(starts, ends, u):
        c = np.cumsum(q[order[a:b]])
        j = int(np.searchsorted(c, x))
        if j < b-a:
            y[order[a+j]] = True
    return y


def power(label, o, events, edge, say, reps=300, mask=None, target=None):
    rng = np.random.default_rng(2026)
    q, true, theta = plant_probs(o, events, edge, target)
    hits = Counter()
    est, lo_, hi_ = [], [], []
    for _ in range(reps):
        v = verdict(renet(o, events, draw(o, q, rng)), mask)
        a, h1, h2 = v["all"], v["h1"], v["h2"]
        est.append(a[0]); lo_.append(a[1]); hi_.append(a[2])
        hits["interval excludes zero"] += a[1] > 0
        hits["both halves positive (point)"] += h1[0] > 0 and h2[0] > 0
        hits["interval > 0 AND both halves positive"] += a[1] > 0 and h1[0] > 0 and h2[0] > 0
        hits["both halves' intervals exclude zero"] += h1[1] > 0 and h2[1] > 0
        d1, d2 = v["h1_diff"], v["h2_diff"]
        hits["both halves: NO interval > 0 and beats mirror (point)"] += h1[1] > 0 and h2[1] > 0 and d1 is not None and d2 is not None and d1[0] > 0 and d2[0] > 0
    say(f"  {label}: planted true edge {100*true:+.2f}% (theta {theta:.4f}), {reps} redraws of the outcomes; fills {v['fills']}")
    say(f"    estimate mean {100*np.mean(est):+.2f}%, sd across redraws {100*np.std(est):.2f}; mean interval [{100*np.mean(lo_):+.2f}, {100*np.mean(hi_):+.2f}] (half-width {50*np.mean(np.array(hi_)-np.array(lo_)):.2f})")
    for k_, n in hits.items():
        say(f"    detected, {k_:55}: {100*n/reps:5.1f}%")
    return {k_: n/reps for k_, n in hits.items()} | dict(true=true, sd=float(np.std(est)))


def audit():
    AUDIT_OUT.mkdir(parents=True, exist_ok=True)
    lines = []

    def say(t=""):
        print(t, flush=True)
        lines.append(t)
    o, events = build(24)
    fmt = lambda v: "n/a" if v is None else f"{100*v[0]:+.2f}% [{100*v[1]:+.2f}, {100*v[2]:+.2f}]" if v[1] is not None else f"{100*v[0]:+.2f}%"
    say("ORIGINAL METHOD, real data, 24h clock (reproduction of the headline and halves of selected cells)")
    cells = [("ALL", None), ("recent tier", o["tier"] == "recent"), ("favourite (rank 1)", o["rank"] == 0), ("YES mid 40+", o["mid"] >= 40),
             ("rank 2-3", (o["rank"] >= 1) & (o["rank"] <= 2)), ("YES mid 10-40", (o["mid"] >= 10) & (o["mid"] < 40)),
             ("favourite, without weather", (o["rank"] == 0) & (o["group"] != "Weather ranges")), ("favourite, weather", (o["rank"] == 0) & (o["group"] == "Weather ranges"))]
    for name, mask in cells:
        v = verdict(o, mask)
        say(f"  {name:28} fills {v['fills']:5}  ret {fmt(v['all'])}  H1 {fmt(v['h1'])}  H2 {fmt(v['h2'])}  NO-mirror c: all {v['all_diff'] and round(v['all_diff'][0], 1)} H1 {v['h1_diff'] and round(v['h1_diff'][0], 1)} H2 {v['h2_diff'] and round(v['h2_diff'][0], 1)}")
    say("\nPOWER: PLANTED EDGE. Outcomes redrawn (one winner per event) so that the filled NO orders truly earn the stated return; fills and prices untouched.")
    res = {}
    for edge in (0.0, 0.015, 0.03, 0.05):
        res[str(edge)] = power(f"ALL fills, edge {100*edge:.1f}%", o, events, edge, say)
    say("\n  Same, only the recent tier counted (edge planted on all fills)")
    for edge in (0.015, 0.03):
        res["recent"+str(edge)] = power(f"recent tier, edge {100*edge:.1f}%", o, events, edge, say, mask=o["tier"] == "recent")
    say("\n  Same, planted and read on the favourite cell only (rank 1)")
    for edge in (0.03, 0.073):
        res["fav"+str(edge)] = power(f"favourite cell, edge {100*edge:.1f}%", o, events, edge, say, mask=o["rank"] == 0, target=o["rank"] == 0)

    say("\nKNOWN EDGE: the validated dip bid (80c resting bid, pre-game 90-99c favourite, strict print-through fills) read with this study's evidence code")
    say("  (boot over events of sum net / sum price, chronological halves). The overround ORDER rule itself does not apply to two-outcome game markets.")
    for label, files in (("men's college basketball, this season", ["other-sports-ncaamb-orders.json"]),
                         ("men's college basketball, last season", ["other-sports-ncaamb-last-season-orders.json"]),
                         ("men's college basketball, both", ["other-sports-ncaamb-orders.json", "other-sports-ncaamb-last-season-orders.json"])):
        rows = [r for f_ in files for r in load(ROOT/"data/processed/favorite-dip"/f_, []) if r.get("level") == "80" and r.get("strict") and 90 <= r["start"] <= 99]
        if len(rows) < 10:
            say(f"  {label}: {len(rows)} strict fills, too few")
            continue
        names = sorted({r["event"] for r in rows})
        code = {n: i for i, n in enumerate(names)}
        d = dict(ge=np.array([code[r["event"]] for r in rows]), close=np.array([r["settle"] for r in rows]), net=np.array([100*r["payout"]-r["limit"] for r in rows]),
                 price=np.array([float(r["limit"]) for r in rows]))
        allm = np.ones(len(rows), bool)
        h1, h2 = halves(d, allm)
        v = [boot([(d["net"], d["price"], m)], d["ge"]) for m in (allm, h1, h2)]
        say(f"  {label}: fills {len(rows)}, ret {fmt(v[0])}  H1 {fmt(v[1])}  H2 {fmt(v[2])}  -> passes (interval > 0 overall, both halves positive): "
            f"{bool(v[0][1] is not None and v[0][1] > 0 and v[1][0] > 0 and v[2][0] > 0)}; both halves' intervals exclude zero: {bool(v[1][1] is not None and v[2][1] is not None and v[1][1] > 0 and v[2][1] > 0)}")
    (AUDIT_OUT/"power.txt").write_text("\n".join(lines)+"\n")
    dump(AUDIT_OUT/"power.json", res)


def main():
    audit()
    return
    # original main follows (unreachable in the audit copy: no network commands here)

    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=["series", "discover", "listings", "rbars", "abars", "dbars", "trades", "report", "plan"])
    parser.add_argument("--minutes", type=float, default=9)
    parser.add_argument("--cap", type=int, default=250)
    args = parser.parse_args()
    if args.command == "report":
        return report()
    if args.command == "plan":
        recent, archive = sample()
        print(f"recent {len(recent)} events, {sum(len(e['picked']) for e in recent)} markets; archive {len(archive)} events, "
              f"{sum(len(e['picked']) for e in archive)} markets; first {args.cap}: {sum(len(e['picked']) for e in archive[:args.cap])} markets")
        for tier in (recent, archive):
            print(sorted(Counter(e["group"] for e in tier).items()))
        return
    deadline[0] = time.time()+60*args.minutes
    began = time.time()
    try:
        with httpx.Client(timeout=60) as client:
            dict(series=lambda: series_step(client), discover=lambda: discover_step(client), listings=lambda: listing_step(client),
                 rbars=lambda: rbars_step(client), abars=lambda: abars_step(client, args.cap), dbars=lambda: dbars_step(client, args.cap),
                 trades=lambda: trades_step(client))[args.command]()
    except OutOfTime:
        print("out of time; run again to resume", flush=True)
    finally:
        history = load(RAW/"download-log.json", [])
        history.append(dict(step=args.command, seconds=round(time.time()-began), **log))
        dump(RAW/"download-log.json", history)
        print(dict(log), flush=True)


if __name__ == "__main__":
    main()
