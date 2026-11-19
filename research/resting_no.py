"""Resting NO bids in "mention" markets: a larger out-of-period test with real-time rules only.

Research only. Public Kalshi reads, no orders, no authentication.

research/hype_no.py found that a NO bid resting one cent inside the spread of a
mention market ("will the speaker say X?") earned about +4c a fill from
2026-09-18 to 2026-11-18 while the same order on the YES side lost. Its event
clock was known only afterwards and its holdout was 520 markets of 41 events.
This script repeats the test with rules a trader could follow live, on a much
larger archive sample.

Fixed before any result was read

* Order: a NO bid one cent above the best NO bid (a YES offer at ask - 1) when
  the quote is two-sided (1 <= bid < ask <= 99) and at least 2c wide. Mirror
  control: a YES bid at bid + 1. Both rest until the MARKET CLOSES (no cancel
  at a speech start nobody announces) and are held to settlement. Filled only
  if a LATER hourly bar prints a trade strictly through the price (above the
  offer for NO, below the bid for the mirror). Maker fee: none (every mention
  series is fee type "quadratic"); 0.0175*p*(1-p) on "quadratic_with_maker_fees".
* Placement clocks, all knowable live. The quote is the close of the hourly
  bar ending at the placement instant; if it is not two-sided and 2c wide the
  next three hourly closes are tried (the paper program looks again for 3h).
    date-1   00:00 UTC of the day before the date in the event ticker, or the
             first hour after the market opens if it opens later, as long as
             that is before 00:00 UTC of the day after the date. This is what
             paper/mentions.py does.
    date     the same from 00:00 UTC of the ticker date itself.
    date12   the same from 12:00 UTC (8am Eastern) of the ticker date.
    open+1h, open+6h, open+24h   that long after open_time, rounded up to the hour.
    first2c  the first hourly close after listing with a two-sided book >= 2c.
* Families by series ticker (hype_no.mention_family): Trump, politicians and
  officials, TV and media, earnings calls, sports broadcasts.
* Recent period: every settled mention market closing 2026-09-18 to 2026-11-18
  (listings of hype_no; hourly bars from open to close through the batch
  endpoint). Holdout: archived mention markets closing 2026-01-01 to
  2026-09-17 from GET /historical/markets, series taken in order of lifetime
  volume; the archive gives candlesticks one market per request, so up to
  PER_EVENT markets per event (pseudo-random) with events taken round-robin
  over family x month. The 520 markets of the hype_no holdout are added where
  their cached bars reach back to the listing.
* Dependence: bootstraps over events and, separately, over calendar days
  (US Eastern date of the event's close); the wider interval is quoted.
* Extension families: the same order and mirror at 24h / 6h / 1h before the
  scheduled close (these markets close on a schedule), resting to the close.

Decided after looking (say so when quoting)

* Sensitivities to the fill rule: fills at the price as well as through it;
  every order still resting when its market settles against it counted as
  filled. The row "holdout without sports", the pooled row, the by-month table.
* Fill realism sample: 75 recent and 75 holdout markets with an order under
  the paper program's rule, every trade from placement to close.
* Extension holdout families (see EXT_HOLD): the archive costs one request per
  market and most chart markets never show a two-sided book, so recurring
  politics (approval ratings) and Netflix charts were the ones worth the
  requests; both samples are small.
* What the archive offers: /historical/markets (listing by series, newest
  first, no date filter), /historical/markets/{ticker}/candlesticks (one
  market per request, hourly bars back to at least January 2026) and
  /historical/trades?ticker= (the full public trade record).

Commands (each resumable; --minutes bounds the network time of that call)

    series    GET /series?category=Mentions with lifetime volume
    listings  archive listings, series by volume
    pick      offline: the holdout sample
    abars     archive hourly bars, one market per request
    rbars     recent hourly bars, batch endpoint
    trades    trade record for the fill-realism sample
    ext       archive listings and bars for two extension families
    report    offline -> data/processed/resting-no/report.txt and summary.json

Network: one request at a time, at least 2.5 seconds apart, 30 seconds or more
of back-off on HTTP 429 (a live trader, a recorder and a paper program share
the limit). A lock file refuses a second download running at the same time.
"""
import argparse
from collections import Counter, defaultdict
from datetime import date, datetime, timedelta, timezone
import hashlib
import json
import os
from pathlib import Path
import re
import sys
import time
from zoneinfo import ZoneInfo

import httpx
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import config
from research import hype_no
from research.api_cost import ROOT, midnight
from research.hype_no import slim, mention_family

RAW = ROOT / "data/raw/resting-no"
OLD = ROOT / "data/raw/hype-no"
OUT = ROOT / "data/processed/resting-no"
PACE = 2.5
RECENT_FROM, RECENT_TO = midnight(date(2026, 9, 18)), midnight(date(2026, 11, 19))
HOLD_FROM = midnight(date(2026, 1, 1))
EASTERN = ZoneInfo("America/New_York")
MONTHS = dict(JAN=1, FEB=2, MAR=3, APR=4, MAY=5, JUN=6, JUL=7, AUG=8, SEP=9, OCT=10, NOV=11, DEC=12)
PER_EVENT = 5
PAGES_PER_SERIES = 6
MENTION_FAMILIES = ("Mentions Trump", "Mentions politicians and officials", "Mentions TV and media", "Mentions earnings calls",
                    "Mentions sports broadcasts")
RNG = np.random.default_rng(31)

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


def event_day(event):
    """The date carried by an event ticker such as KXFEDMENTION-26DEC10 (the rule of paper/mentions.py), or None."""
    found = re.search(r"-(\d{2})([A-Z]{3})(\d{2})(?:\D|$)", event)
    try:
        return date(2000+int(found.group(1)), MONTHS[found.group(2)], int(found.group(3)))
    except (AttributeError, KeyError, ValueError):
        return None


# ---------------------------------------------------------------- listings

def series_step(client):
    data = read(client, "/series", dict(category="Mentions", include_volume="true")) or {}
    rows = [dict(ticker=s["ticker"], title=s["title"], volume=float(s.get("volume_fp") or 0), fee_type=s.get("fee_type"),
                 fee_multiplier=s.get("fee_multiplier")) for s in data.get("series") or []]
    rows.sort(key=lambda s: -s["volume"])
    dump(RAW/"series.json", rows)
    print(f"{len(rows)} mention series; fee types {Counter(s['fee_type'] for s in rows)}", flush=True)


def listing_step(client, series_list, folder, most_pages=PAGES_PER_SERIES):
    """Archive listings (newest first, 1,000 a page) back to HOLD_FROM, slim, resumable page by page."""
    for n, series in enumerate(series_list):
        path = RAW/folder/f"{series}.json"
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
                if m.get("result") in ("yes", "no") and v["close"] and v["open"] and HOLD_FROM <= v["close"] < RECENT_FROM:
                    have["markets"].append(dict(v, exp=v["exp"] or v["close"]))
            have["cursor"] = data.get("cursor") or ""
            have["done"] = bool(not have["cursor"] or not rows or have["pages"] >= most_pages or (oldest is not None and oldest < HOLD_FROM))
            have["reached"] = oldest
            dump(path, have)
        print(f"  {n+1}/{len(series_list)} {series:28} {len(have['markets']):5} archived markets, {have['pages']} pages, "
              f"{len({m['event'] for m in have['markets']})} events", flush=True)


def archive_markets(folder="archive-markets"):
    out = []
    meta = {s["ticker"]: s for s in load(RAW/"series.json", [])}
    for path in sorted((RAW/folder).glob("*.json")):
        series = path.stem
        for m in load(path, dict(markets=[]))["markets"]:
            out.append(dict(m, series=series, maker_fee=meta.get(series, {}).get("fee_type") == "quadratic_with_maker_fees"))
    return out


def old_holdout():
    """The 520 markets of the hype_no holdout whose cached bars (48h before the event's close) reach back to the listing."""
    bars = load(OLD/"bars/mentions-before-sep-18-holdout.json", {})
    out = {}
    for series in load(OLD/"archive-series.json", []):
        ms = load(OLD/"markets-archive"/f"{series}.json", dict(markets=[]))["markets"]
        closes = defaultdict(Counter)
        for m in ms:
            closes[m["event"]][m["close"]//600] += 1
        mode = {g: 600*min(c, key=lambda k: (-c[k], k)) for g, c in closes.items()}
        for m in ms:
            if m["ticker"] in bars and m["open"] >= mode[m["event"]]-48*3600 and m["close"] <= mode[m["event"]]+3600:
                out[m["ticker"]] = [b+[None, None] for b in bars[m["ticker"]]]
    return out


def pick():
    """The holdout sample: up to PER_EVENT markets per event, events round-robin over family x month in a fixed pseudo-random order."""
    ms = [m for m in archive_markets() if event_day(m["event"]) is not None]
    by_event = defaultdict(list)
    for m in ms:
        by_event[m["event"]].append(m)
    cells = defaultdict(list)
    for event, rows in by_event.items():
        cells[(mention_family(rows[0]["series"]), datetime.fromtimestamp(max(m["close"] for m in rows), timezone.utc).strftime("%Y-%m"))].append(event)
    for key in cells:
        cells[key].sort(key=lambda e: pick_hash("event", e))
    order, depth = [], 0
    while any(len(v) > depth for v in cells.values()):
        order += [v[depth] for key, v in sorted(cells.items()) if len(v) > depth]
        depth += 1
    sample = []
    for event in order:
        sample += [m["ticker"] for m in sorted(by_event[event], key=lambda m: pick_hash("market", m["ticker"]))[:PER_EVENT]]
    dump(RAW/"sample.json", sample)
    print(f"{len(ms)} archived markets with a ticker date in {len(by_event)} events; {len(cells)} family x month cells; sample order of {len(sample)} markets", flush=True)
    for fam in MENTION_FAMILIES:
        print(f"  {fam:36} events by month: "+" ".join(f"{k[1][5:]}:{len(v)}" for k, v in sorted(cells.items()) if k[0] == fam), flush=True)


# ---------------------------------------------------------------- bars

def money(group, field):
    value = group.get(field+"_dollars", group.get(field)) if group else None
    return None if value is None else round(100*float(value), 2)


def compact(candles):
    """[end_ts, yes bid close, yes ask close, trade low, trade high, volume, yes bid high, yes ask low] per hourly bar."""
    return [[b["end_period_ts"], money(b.get("yes_bid"), "close"), money(b.get("yes_ask"), "close"), money(b.get("price"), "low"),
             money(b.get("price"), "high"), float(b.get("volume_fp", b.get("volume")) or 0), money(b.get("yes_bid"), "high"),
             money(b.get("yes_ask"), "low")] for b in sorted(candles or [], key=lambda b: b["end_period_ts"])]


def abars_step(client, cap, file="archive-bars.json", sample="sample.json", folder="archive-markets"):
    have = load(RAW/file, {})
    by = {m["ticker"]: m for m in archive_markets(folder)}
    old = old_holdout() if file == "archive-bars.json" else {}
    todo = [t for t in load(RAW/sample, [])[:cap] if t not in have and t not in old]
    print(f"archive bars: {len(have)} on disk, {len(todo)} to fetch", flush=True)
    try:
        for n, t in enumerate(todo):
            m = by[t]
            data = read(client, f"/historical/markets/{t}/candlesticks",
                        dict(start_ts=m["open"]//3600*3600-3600, end_ts=m["close"]//3600*3600+7200, period_interval=60))
            if data is None:
                log["no_bars"] += 1
            have[t] = compact((data or {}).get("candlesticks"))
            if n % 25 == 24:
                dump(RAW/file, have)
    finally:
        dump(RAW/file, have)
    print(f"archive bars for {len(have)} markets", flush=True)


def recent_markets():
    fam = hype_no.families()
    out = []
    for name in MENTION_FAMILIES:
        for series in fam["families"].get(name, []):
            meta = fam["meta"][series]
            for m in hype_no.markets_of(series):
                if m["open"] and m["close"]:
                    out.append(dict(m, series=series, maker_fee=meta.get("fee_type") == "quadratic_with_maker_fees"))
    return out


def rbars_step(client, markets=None, file="recent-bars.json"):
    """Hourly bars from open to close through the batch endpoint (at most 100 markets and about 9,000 bars a request)."""
    have = load(RAW/file, {})
    todo = sorted((m for m in (markets or recent_markets()) if m["ticker"] not in have), key=lambda m: (m["open"]//3600, m["close"], m["ticker"]))
    chunks, cur = [], []
    for m in todo:
        trial = cur+[m]
        hours = (max(x["close"] for x in trial)-trial[0]["open"])//3600+4
        if cur and (len(trial) > 100 or len(trial)*hours > 9000):
            chunks.append(cur)
            cur = []
        cur.append(m)
    chunks += [cur] if cur else []
    print(f"recent bars: {len(have)} on disk, {len(todo)} markets in {len(chunks)} requests", flush=True)
    try:
        for n, group in enumerate(chunks):
            data = read(client, "/markets/candlesticks", dict(market_tickers=",".join(m["ticker"] for m in group),
                                                              start_ts=group[0]["open"]//3600*3600-3600,
                                                              end_ts=max(m["close"] for m in group)//3600*3600+7200, period_interval=60))
            if data is None:
                log["batch_refused"] += 1
                continue
            got = {x["market_ticker"]: compact(x.get("candlesticks")) for x in data.get("markets", [])}
            for m in group:
                have[m["ticker"]] = [b for b in got.get(m["ticker"], []) if m["open"]-3600 <= b[0] <= m["close"]+7200]
            if n % 10 == 9:
                dump(RAW/file, have)
    finally:
        dump(RAW/file, have)
    print(f"recent bars for {len(have)} markets", flush=True)


# ---------------------------------------------------------------- orders

RULES = ("date-1", "date", "date12", "open+1h", "open+6h", "open+24h", "first2c")
HEAD = "date-1"                      # the paper program's rule
LOOK_HOURS = 3


def up_hour(ts):
    return -(-int(ts)//3600)*3600


def utc_midnight(d):
    return int(datetime(d.year, d.month, d.day, tzinfo=timezone.utc).timestamp())


def two_sided(bar):
    return bar[1] is not None and bar[2] is not None and 1 <= bar[1] < bar[2] <= 99 and bar[2]-bar[1] >= 2


def place_times(m, day):
    """Earliest placement instant per rule (an hour boundary), or None where the rule gives no order."""
    first = up_hour(m["open"])
    out = {"open+1h": up_hour(m["open"]+3600), "open+6h": up_hour(m["open"]+6*3600), "open+24h": up_hour(m["open"]+24*3600), "first2c": first}
    if day is not None:
        for name, start in (("date-1", utc_midnight(day)-86400), ("date", utc_midnight(day)), ("date12", utc_midnight(day)+12*3600)):
            t = max(first, start)
            out[name] = t if t < utc_midnight(day)+86400 else None
    return out


def simulate(m, bars, t, patient):
    """The order placed at the first hourly close at or after t with a two-sided book >= 2c (within LOOK_HOURS unless patient)."""
    if t is None or t >= m["close"]:
        return None
    ends = [b[0] for b in bars]
    i = int(np.searchsorted(ends, t, side="right"))-1          # last bar ending at or before t
    if i < 0 or t-ends[i] > 86400:
        i += 1
        if i >= len(bars) or bars[i][0] > t+LOOK_HOURS*3600 and not patient:
            return None
    stop = float("inf") if patient else t+LOOK_HOURS*3600
    while i < len(bars) and not two_sided(bars[i]):
        i += 1
        if i >= len(bars) or bars[i][0] > stop:
            return None
    if i >= len(bars) or bars[i][0] >= m["close"]:
        return None
    at, bid, ask = max(bars[i][0], t), bars[i][1], bars[i][2]
    later = bars[i+1:]
    fill = next((k for k, b in enumerate(later) if b[5] > 0 and b[4] is not None and b[4] > ask-1), None)
    yfill = next((k for k, b in enumerate(later) if b[5] > 0 and b[3] is not None and b[3] < bid+1), None)
    touch = next((k for k, b in enumerate(later) if b[5] > 0 and b[4] is not None and b[4] >= ask-1), None)
    ytouch = next((k for k, b in enumerate(later) if b[5] > 0 and b[3] is not None and b[3] <= bid+1), None)
    beat = next((k for k, b in enumerate(later) if len(b) > 7 and b[7] is not None and b[7] < ask-1), None)
    ybeat = next((k for k, b in enumerate(later) if len(b) > 6 and b[6] is not None and b[6] > bid+1), None)
    detail = len(later) > 0 and len(later[0]) > 7 and later[0][7] is not None
    return dict(at=at, bid=bid, ask=ask, fill=fill, yfill=yfill, touch=touch, ytouch=ytouch, beat=beat, ybeat=ybeat, detail=detail,
                fill_ts=None if fill is None else later[fill][0], yfill_ts=None if yfill is None else later[yfill][0],
                beat_ts=None if beat is None else later[beat][0])


def universe():
    """Every mention market with bars: recent (all) and holdout (sample), with event-level facts taken from the full listings."""
    recent, archive = recent_markets(), archive_markets()
    rb, ab, ob = load(RAW/"recent-bars.json", {}), load(RAW/"archive-bars.json", {}), old_holdout()
    listed = {m["ticker"]: m for m in archive}
    for series in load(OLD/"archive-series.json", []):            # the hype_no holdout listings (June onward), for markets not re-listed
        for m in load(OLD/"markets-archive"/f"{series}.json", dict(markets=[]))["markets"]:
            listed.setdefault(m["ticker"], dict(m, series=series, maker_fee=False))
    events = defaultdict(list)
    for m in recent+list(listed.values()):
        events[m["event"]].append(m)
    facts = {}
    for event, rows in events.items():
        closes = Counter(m["close"]//600 for m in rows)
        mode = 600*min(closes, key=lambda k: (-closes[k], k))
        said = [m["close"] for m in rows if m["result"] == "yes" and m["close"] < mode-600]
        facts[event] = dict(mode=mode, start=min(said) if said else None, n=len(rows), day=event_day(event))
    out = []
    for period, ms, sources in (("recent", recent, (rb,)), ("holdout", list(listed.values()), (ab, ob))):
        for m in ms:
            bars = next((src[m["ticker"]] for src in sources if m["ticker"] in src), None)
            if bars is None:
                continue
            out.append(dict(m, period=period, bars=bars, family=mention_family(m["series"]).replace("Mentions ", "")))
    return out, facts


COLS = ("bid", "ask", "y", "filled", "yfilled", "fee", "yfee", "g", "d", "m", "at", "lead", "beat_first", "beat_same", "ybeat_first", "detail", "eventsize",
        "fill_lag", "t_event", "beat_lag", "touched", "ytouched")


def build():
    ms, facts = universe()
    gcode, dcode = {}, {}
    rows = {rule: [] for rule in RULES}
    meta = {rule: [] for rule in RULES}
    cover = defaultdict(Counter)
    for mi, m in enumerate(ms):
        f = facts[m["event"]]
        times = place_times(m, f["day"])
        key = (m["period"], m["family"])
        cover[key]["markets with bars"] += 1
        cover[key]["no ticker date"] += f["day"] is None
        cover[key]["empty bars"] += not m["bars"]
        g = gcode.setdefault(m["event"], len(gcode))
        d = dcode.setdefault(str(datetime.fromtimestamp(f["mode"], EASTERN).date()), len(dcode))
        t_event = f["start"] or f["mode"]
        for rule in RULES:
            o = simulate(m, m["bars"], times.get(rule), rule == "first2c") if m["bars"] else None
            if o is None:
                continue
            cover[key][rule] += 1
            y = m["result"] == "yes"
            offer, ybid = o["ask"]-1, o["bid"]+1
            fee = 1.75*offer/100*(1-offer/100) if m["maker_fee"] else 0.0
            yfee = 1.75*ybid/100*(1-ybid/100) if m["maker_fee"] else 0.0
            rows[rule].append((o["bid"], o["ask"], y, o["fill"] is not None, o["yfill"] is not None, fee, yfee, g, d, mi, o["at"], (t_event-o["at"])/3600,
                               o["beat"] is not None and (o["fill"] is None or o["beat"] < o["fill"]),
                               o["beat"] is not None and o["fill"] is not None and o["beat"] == o["fill"],
                               o["ybeat"] is not None and (o["yfill"] is None or o["ybeat"] < o["yfill"]), o["detail"], f["n"],
                               np.nan if o["fill_ts"] is None else (o["fill_ts"]-o["at"])/3600, t_event,
                               np.nan if o["beat_ts"] is None else (o["beat_ts"]-o["at"])/3600, o["touch"] is not None, o["ytouch"] is not None))
            meta[rule].append((m["ticker"], m["event"], m["period"], m["family"]))
    out = {}
    for rule in RULES:
        a = np.array(rows[rule], float).reshape(-1, len(COLS))
        r = {c: a[:, i] for i, c in enumerate(COLS)}
        for c in ("g", "d", "m"):
            r[c] = r[c].astype(int)
        for c in ("y", "filled", "yfilled", "beat_first", "beat_same", "ybeat_first", "detail", "touched", "ytouched"):
            r[c] = r[c] == 1
        r["ticker"] = np.array([x[0] for x in meta[rule]])
        r["event"] = np.array([x[1] for x in meta[rule]])
        r["period"] = np.array([x[2] for x in meta[rule]])
        r["family"] = np.array([x[3] for x in meta[rule]])
        r["mid"], r["spread"] = (r["bid"]+r["ask"])/2, r["ask"]-r["bid"]
        r["price"] = 100-(r["ask"]-1)                          # NO bid
        r["net"] = 100*(~r["y"])-r["price"]-r["fee"]
        r["yprice"] = r["bid"]+1                               # mirror: YES bid
        r["ynet"] = 100*r["y"]-r["yprice"]-r["yfee"]
        r["gap"] = 100*r["y"]-r["mid"]
        hold = r["period"] == "holdout"
        order = np.argsort(r["t_event"][hold], kind="stable")
        events_in_order = list(dict.fromkeys(r["g"][hold][order]))
        first = set(events_in_order[:len(events_in_order)//2])
        r["h1"] = hold & np.isin(r["g"], list(first))
        r["h2"] = hold & ~r["h1"]
        out[rule] = r
    return out, ms, facts, {f"{k[0]} | {k[1]}": dict(v) for k, v in cover.items()}, sorted(dcode, key=dcode.get)


# ---------------------------------------------------------------- statistics

def boot(parts, codes, draws=1000):
    """Bootstrap over clusters of f(sum_1/den_1, sum_2/den_2, ...). parts = [(num, den, mask), ...]; returns estimate, lo, hi, clusters."""
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
        sums.append((np.bincount(inv, (num[used]*w)), np.bincount(inv, (den[used]*w))))
        if sums[-1][1].sum() == 0:
            return None
    value = lambda pick: [a[pick].sum(-1)/np.where(b[pick].sum(-1) == 0, np.nan, b[pick].sum(-1)) for a, b in sums]
    est = [a.sum()/b.sum() for a, b in sums]
    est = est[0]-est[1] if len(est) == 2 else est[0]
    if k < 5:
        return est, None, None, k
    pick = RNG.integers(0, k, (draws, k))
    v = value(pick)
    v = v[0]-v[1] if len(v) == 2 else v[0]
    lo, hi = np.nanpercentile(v, [2.5, 97.5])
    return float(est), float(lo), float(hi), k


def wide(r, parts):
    """The estimate with the wider of the event-clustered and day-clustered 95% intervals."""
    a, b = boot(parts, r["g"]), boot(parts, r["d"])
    if a is None:
        return None
    if a[1] is None or b[1] is None:
        return a[0], None, None, a[3], b[3]
    pick = a if a[2]-a[1] >= b[2]-b[1] else b
    return a[0], pick[1], pick[2], a[3], b[3]


def show(v, scale=1.0, width=5):
    if v is None:
        return f"{'n/a':>{width+17}}"
    if v[1] is None:
        return f"{scale*v[0]:+{width}.1f} [{'n/a':^{15}}]"
    return f"{scale*v[0]:+{width}.1f} [{scale*v[1]:+6.1f},{scale*v[2]:+6.1f}]"


def line(r, mask, mode="through"):
    """One row of results: NO side, mirror, difference, unfilled, calibration gap.
    mode "touch": fills at the price count, not only through it. "worst": as through, and every order still resting when the market
    settles AGAINST it is counted as filled (picked off at resolution without a print through the price)."""
    n = int(mask.sum())
    if n < 20:
        return None
    one = np.ones(len(mask))
    f, yf = mask & r["touched" if mode == "touch" else "filled"], mask & r["ytouched" if mode == "touch" else "yfilled"]
    if mode == "worst":
        f, yf = f | (mask & r["y"]), yf | (mask & ~r["y"])
    o = dict(orders=n, events=len(set(r["g"][mask])), days=len(set(r["d"][mask])), fill_rate=float(f.sum()/n), mirror_fill_rate=float(yf.sum()/n),
             spread=float(r["spread"][mask].mean()), mid=float(r["mid"][mask].mean()), gap=wide(r, [(r["gap"], one, mask)]),
             lead=float(np.median(r["lead"][mask])), late=float((r["lead"][mask] <= 0).mean()))
    if f.sum() >= 10:
        o.update(net=wide(r, [(r["net"], one, f)]), ret=wide(r, [(r["net"], r["price"], f)]), paid=float(r["price"][f].mean()),
                 won=float((~r["y"][f]).mean()))
    if yf.sum() >= 10:
        o.update(mirror_net=wide(r, [(r["ynet"], one, yf)]), mirror_ret=wide(r, [(r["ynet"], r["yprice"], yf)]))
    if f.sum() >= 10 and yf.sum() >= 10:
        o["diff"] = wide(r, [(r["net"], one, f), (r["ynet"], one, yf)])
    u, yu = mask & ~f, mask & ~yf
    o["unfilled"] = float(r["net"][u].mean()) if u.any() else None
    o["mirror_unfilled"] = float(r["ynet"][yu].mean()) if yu.any() else None
    return o


HEADER = (f"{'':38} {'orders':>6} {'ev':>4} {'days':>4} {'sprd':>4} {'fill%':>5} {'NO net c / fill':>22} {'NO return %':>22} {'un-':>6} | "
          f"{'fill%':>5} {'MIRROR net c / fill':>22} {'ret %':>6} {'un-':>6} | {'NO minus mirror c':>22} | {'YES freq - mid c':>22}")


def text(label, o):
    if o is None:
        return None
    num = lambda v: f"{v:+6.1f}" if v is not None else "   n/a"
    return (f"{label:38} {o['orders']:6} {o['events']:4} {o['days']:4} {o['spread']:4.1f} {100*o['fill_rate']:5.0f} {show(o.get('net'))} {show(o.get('ret'), 100)} "
            f"{num(o['unfilled'])} | {100*o['mirror_fill_rate']:5.0f} {show(o.get('mirror_net'))} {num(100*o['mirror_ret'][0] if o.get('mirror_ret') else None)} "
            f"{num(o['mirror_unfilled'])} | {show(o.get('diff'))} | {show(o['gap'])}")


PERIODS = (("recent Sep18-Nov18", lambda r: r["period"] == "recent"), ("HOLDOUT Jan1-Sep17", lambda r: r["period"] == "holdout"),
           ("  holdout first half", lambda r: r["h1"]), ("  holdout second half", lambda r: r["h2"]),
           ("  holdout without sports", lambda r: (r["period"] == "holdout") & (r["family"] != "sports broadcasts")),
           ("both periods pooled", lambda r: np.ones(len(r["period"]), bool)))
FAMILIES = ("Trump", "politicians and officials", "TV and media", "earnings calls", "sports broadcasts")
LEVELS = (("YES under 20", 0, 20), ("YES 20-40", 20, 40), ("YES 40-60", 40, 60), ("YES 60-80", 60, 80), ("YES over 80", 80, 101))
WIDTHS = (("spread 2-4c", 2, 5), ("spread 5-9c", 5, 10), ("spread 10c+", 10, 101))


# ---------------------------------------------------------------- report

def report():
    OUT.mkdir(parents=True, exist_ok=True)
    lines, summary = [], {}

    def say(text_=""):
        if text_ is not None:
            print(text_, flush=True)
            lines.append(text_)
    rs, ms, facts, cover, days = build()
    coverage(rs, ms, facts, cover, say, summary)
    clocks(rs, ms, facts, say, summary)
    say("\n(3) RESULTS BY RULE. A NO bid one cent above the best NO bid (book two-sided, >= 2c), resting until the market closes, held to settlement; filled by a later"
        "\n    hourly bar printing strictly through it. MIRROR = the same on the YES side. 'un-' = what the UNFILLED orders would have made per contract at the same price"
        "\n    (adverse selection: fills are the ones the other side wanted). Intervals: 95%, the wider of event-clustered and day-clustered bootstraps."
        "\n    'YES freq - mid' = calibration at placement over all orders: negative means YES was overpriced, whatever the fill model.")
    for rule in RULES:
        r = rs[rule]
        say(f"\n  RULE {rule}" + ("   <- the paper program's rule" if rule == HEAD else ""))
        say("  "+HEADER)
        for pname, pick in PERIODS:
            o = line(r, pick(r))
            say(text("  "+pname+", all families", o) and "  "+text(pname+", all families", o))
            summary.setdefault("rules", {})[f"{rule}|{pname.strip()}|all"] = o
        for fam in FAMILIES:
            for pname, pick in PERIODS[:4]:
                o = line(r, pick(r) & (r["family"] == fam))
                if o is not None:
                    say("  "+text(f"{fam[:16]}, {pname.strip()[:20]}", o))
                    summary["rules"][f"{rule}|{pname.strip()}|{fam}"] = o
    say("\n(3b) BY YES PRICE AND BY SPREAD AT PLACEMENT (all families)")
    for rule in (HEAD, "date", "open+1h", "first2c"):
        r = rs[rule]
        say(f"\n  RULE {rule}")
        say("  "+HEADER)
        for pname, pick in PERIODS[:2]:
            for label, lo, hi in LEVELS:
                o = line(r, pick(r) & (r["mid"] >= lo) & (r["mid"] < hi))
                if o is not None:
                    say("  "+text(f"{pname.split()[0]}, {label}", o))
                    summary.setdefault("levels", {})[f"{rule}|{pname.strip()}|{label}"] = o
            for label, lo, hi in WIDTHS:
                o = line(r, pick(r) & (r["spread"] >= lo) & (r["spread"] < hi))
                if o is not None:
                    say("  "+text(f"{pname.split()[0]}, {label}", o))
                    summary.setdefault("levels", {})[f"{rule}|{pname.strip()}|{label}"] = o
    say("\n(3d) SENSITIVITY: FILLED AT THE PRICE, NOT ONLY THROUGH IT. Our bid makes a new price level, so the first contract to trade AT our price is ours."
        "\n     Same table with a fill whenever a later hourly bar prints at or through the price (all families).")
    say("  "+HEADER)
    for rule in RULES:
        for pname, pick in PERIODS[:2]+PERIODS[4:5]:
            o = line(rs[rule], pick(rs[rule]), "touch")
            if o is not None:
                say("  "+text(f"{rule}, {pname.strip()[:24]}", o))
                summary.setdefault("touch", {})[f"{rule}|{pname.strip()}"] = o
    say("\n(3e) WORST CASE FOR FILLS: strictly-through fills, PLUS every order still resting when its market settles against it is counted as filled"
        "\n     (a NO bid in a market that resolved YES is taken to have been picked off at the resolution even though no print went through it).")
    say("  "+HEADER)
    for rule in RULES:
        for pname, pick in PERIODS[:2]+PERIODS[4:5]:
            o = line(rs[rule], pick(rs[rule]), "worst")
            if o is not None:
                say("  "+text(f"{rule}, {pname.strip()[:24]}", o))
                summary.setdefault("worst", {})[f"{rule}|{pname.strip()}"] = o
    say("\n(3f) BY MONTH of the event's close (US Eastern), sports broadcasts left out: fills, NO net c per fill / mirror net c per fill."
        "\n     January to mid-September is the holdout sample (about 5 words per event); from 18 September every market.")
    months = sorted({d[:7] for d in days})
    code = np.array([d[:7] for d in days])
    say("     "+f"{'rule':9}"+"".join(f"{m[2:]:>17}" for m in months))
    for rule in (HEAD, "date", "open+1h", "first2c"):
        r = rs[rule]
        cells = []
        for m in months:
            mask = (code[r["d"]] == m) & (r["family"] != "sports broadcasts")
            f, yf = mask & r["filled"], mask & r["yfilled"]
            cells.append(f"{int(f.sum()):5} {r['net'][f].mean() if f.any() else float('nan'):+5.1f}/{r['ynet'][yf].mean() if yf.any() else float('nan'):+5.1f}")
            summary.setdefault("months", {})[f"{rule}|{m}"] = dict(fills=int(f.sum()), net=float(r["net"][f].mean()) if f.any() else None,
                                                                   mirror=float(r["ynet"][yf].mean()) if yf.any() else None)
        say("     "+f"{rule:9}"+"".join(f"{c:>17}" for c in cells))
    timing(rs, say, summary)
    realism(rs, say, summary)
    tail(rs, days, say, summary)
    extension(say, summary)
    forward(rs, say, summary)
    summary["requests"] = load(RAW/"download-log.json", [])
    dump(OUT/"summary.json", summary)
    (OUT/"report.txt").write_text("\n".join(lines)+"\n")


def coverage(rs, ms, facts, cover, say, summary):
    say("(1) COVERAGE. Recent = every settled mention market closing 2026-09-18 to 2026-11-18 (batch bars, open to close)."
        "\n    Holdout = archived markets closing 2026-01-01 to 2026-09-17: a sample of up to 5 markets per event (plus the 520 markets of the earlier holdout).")
    series = load(RAW/"series.json", [])
    listed = [(p.stem, load(p, {})) for p in sorted((RAW/"archive-markets").glob("*.json"))]
    total = sum(s["volume"] for s in series) or 1
    vol = {s["ticker"]: s["volume"] for s in series}
    say(f"    {len(series)} mention series exist; archive listings read for {len(listed)} of them holding {100*sum(vol.get(k, 0) for k, _ in listed)/total:.0f}% of lifetime volume;"
        f" {sum(len(v['markets']) for _, v in listed)} archived markets in {len({m['event'] for _, v in listed for m in v['markets']})} events listed.")
    short = [k for k, v in listed if v.get("reached") and v["reached"] >= HOLD_FROM and v.get("cursor")]
    say(f"    series whose listing stopped at the page cap before reaching January: {', '.join(short) or 'none'}")
    table = {}
    say(f"    {'period':8} {'family':28} {'series':>6} {'events':>6} {'markets':>7} {'YES%':>5} {'first close':>11} {'last close':>11} {'no date in ticker':>17}")
    for period in ("recent", "holdout"):
        for fam in FAMILIES+("ALL",):
            sel = [m for m in ms if m["period"] == period and fam in (m["family"], "ALL")]
            if not sel:
                continue
            day = lambda t: datetime.fromtimestamp(t, timezone.utc).strftime("%Y-%m-%d")
            o = dict(series=len({m["series"] for m in sel}), events=len({m["event"] for m in sel}), markets=len(sel), yes=float(np.mean([m["result"] == "yes" for m in sel])),
                     first=day(min(m["close"] for m in sel)), last=day(max(m["close"] for m in sel)),
                     undated=sum(facts[m["event"]]["day"] is None for m in sel))
            table[f"{period}|{fam}"] = o
            say(f"    {period:8} {fam:28} {o['series']:6} {o['events']:6} {o['markets']:7} {100*o['yes']:5.1f} {o['first']:>11} {o['last']:>11} {o['undated']:17}")
    months = Counter((datetime.fromtimestamp(m["close"], timezone.utc).strftime("%Y-%m")) for m in ms if m["period"] == "holdout")
    evm = defaultdict(set)
    for m in ms:
        if m["period"] == "holdout":
            evm[datetime.fromtimestamp(m["close"], timezone.utc).strftime("%Y-%m")].add(m["event"])
    say("    holdout by month of close (markets / events): "+"  ".join(f"{k} {months[k]}/{len(evm[k])}" for k in sorted(months)))
    say("    orders per rule (markets where the rule found a two-sided book >= 2c): "
        + "  ".join(f"{rule} {int((rs[rule]['period'] == 'recent').sum())}/{int((rs[rule]['period'] == 'holdout').sum())}" for rule in RULES)+"  (recent/holdout)")
    summary["coverage"] = dict(table=table, by_month={k: [months[k], len(evm[k])] for k in months}, detail=cover)


def clocks(rs, ms, facts, say, summary):
    say("\n(2) DOES THE DATE IN THE TICKER MATCH THE EVENT DAY? Event time = the earliest early close of a word that resolved YES (it closes when said), from the full"
        "\n    listing of the event; where no word closed early, the event's common close. Compared with the ticker date, per event.")
    say(f"    {'period':8} {'family':28} {'events':>6} {'no date':>7} {'same UTC day':>12} {'same US-East day':>16} {'event later':>11} {'event earlier':>13} "
        f"{'h from 00:00 UTC of date, 10/50/90%':>36}")
    seen = {}
    for m in ms:
        seen.setdefault(m["event"], (m["period"], m["family"]))
    out = {}
    for period in ("recent", "holdout"):
        for fam in FAMILIES+("ALL",):
            evs = [e for e, (p, f) in seen.items() if p == period and fam in (f, "ALL")]
            if not evs:
                continue
            dated = [e for e in evs if facts[e]["day"] is not None]
            t = np.array([facts[e]["start"] or facts[e]["mode"] for e in dated])
            d0 = np.array([utc_midnight(facts[e]["day"]) for e in dated])
            utc = np.array([datetime.fromtimestamp(x, timezone.utc).date() == facts[e]["day"] for x, e in zip(t, dated)])
            east = np.array([datetime.fromtimestamp(x, EASTERN).date() == facts[e]["day"] for x, e in zip(t, dated)])
            eastd = np.array([(datetime.fromtimestamp(x, EASTERN).date()-facts[e]["day"]).days for x, e in zip(t, dated)])
            q = np.percentile((t-d0)/3600, [10, 50, 90]) if len(dated) else [np.nan]*3
            o = dict(events=len(evs), undated=len(evs)-len(dated), same_utc=float(utc.mean()) if len(dated) else None, same_east=float(east.mean()) if len(dated) else None,
                     later=float((eastd > 0).mean()) if len(dated) else None, earlier=float((eastd < 0).mean()) if len(dated) else None, hours=[float(x) for x in q])
            out[f"{period}|{fam}"] = o
            if dated:
                say(f"    {period:8} {fam:28} {o['events']:6} {o['undated']:7} {100*o['same_utc']:11.0f}% {100*o['same_east']:15.0f}% {100*o['later']:10.0f}% {100*o['earlier']:12.0f}% "
                    f"{q[0]:14.0f} {q[1]:6.0f} {q[2]:6.0f}")
    summary["clock"] = out


def timing(rs, say, summary):
    say("\n(3c) WHEN THE ORDER GOES IN, relative to the event (known only afterwards; shown to explain the rules, not to trade on)."
        "\n     lead = hours from placement to the event time; 'after start' = orders placed once the event had begun; fill lag = hours from placement to the fill bar.")
    say(f"     {'rule':10} {'period':8} {'orders':>6} {'lead h 10/50/90%':>20} {'after start':>11} {'median fill lag h':>17} {'fills after start':>17} | NO net c: filled before / after the event began")
    for rule in RULES:
        r = rs[rule]
        for period in ("recent", "holdout"):
            mask = r["period"] == period
            if mask.sum() < 20:
                continue
            q = np.percentile(r["lead"][mask], [10, 50, 90])
            f = mask & r["filled"]
            after = f & (r["fill_lag"] > r["lead"])
            before = f & ~after
            o = dict(orders=int(mask.sum()), lead=[float(x) for x in q], placed_after_start=float((r["lead"][mask] <= 0).mean()),
                     median_fill_lag=float(np.nanmedian(r["fill_lag"][f])) if f.any() else None, fills_after_start=float(after.sum()/max(1, f.sum())),
                     net_filled_before=float(r["net"][before].mean()) if before.any() else None, net_filled_after=float(r["net"][after].mean()) if after.any() else None)
            summary.setdefault("timing", {})[f"{rule}|{period}"] = o
            say(f"     {rule:10} {period:8} {o['orders']:6} {q[0]:7.0f}{q[1]:6.0f}{q[2]:7.0f} {100*o['placed_after_start']:10.0f}% {o['median_fill_lag'] or float('nan'):17.1f} "
                f"{100*o['fills_after_start']:16.0f}% | {o['net_filled_before'] if o['net_filled_before'] is not None else float('nan'):+7.1f} ({int(before.sum())})"
                f" / {o['net_filled_after'] if o['net_filled_after'] is not None else float('nan'):+7.1f} ({int(after.sum())})")


# ---------------------------------------------------------------- fill realism (trade record)

TRADE_SAMPLE, TRADE_PAGES = 75, 4    # markets per period; pages of 1,000 trades per market


def trade_sample():
    """Markets with an order under the headline rule, round-robin over families in a fixed pseudo-random order, per period."""
    r = build()[0][HEAD]
    out = []
    for period in ("recent", "holdout"):
        cells = defaultdict(list)
        for i in np.flatnonzero(r["period"] == period):
            cells[r["family"][i]].append((pick_hash("trade", r["ticker"][i]), r["ticker"][i], float(r["at"][i])))
        for v in cells.values():
            v.sort()
        depth, got = 0, []
        while len(got) < TRADE_SAMPLE and any(len(v) > depth for v in cells.values()):
            got += [v[depth][1:]+(period,) for _, v in sorted(cells.items()) if len(v) > depth]
            depth += 1
        out += got[:TRADE_SAMPLE]
    return out


def trades_step(client):
    have = load(RAW/"trades.json", {})
    todo = [x for x in trade_sample() if x[0] not in have]
    print(f"trades: {len(have)} markets on disk, {len(todo)} to fetch", flush=True)
    try:
        for ticker, at, period in todo:
            rows, cursor, pages = [], "", 0
            while pages < TRADE_PAGES:
                data = read(client, "/markets/trades" if period == "recent" else "/historical/trades",
                            dict(ticker=ticker, limit=1000, min_ts=int(at)-7200, **({"cursor": cursor} if cursor else {}))) or {}
                pages += 1
                for t in data.get("trades") or []:
                    rows.append([int(hype_no.timestamp(t["created_time"])), round(100*float(t["yes_price_dollars"]), 2), float(t.get("count_fp", t.get("count", 0))),
                                 1 if (t.get("taker_side") or t.get("taker_outcome_side")) == "yes" else 0])
                cursor = data.get("cursor")
                if not cursor or not data.get("trades"):
                    break
            have[ticker] = dict(period=period, complete=not cursor, trades=rows)
    finally:
        dump(RAW/"trades.json", have)
    print(f"trades for {len(have)} markets", flush=True)


def realism(rs, say, summary):
    have = load(RAW/"trades.json", {})
    say("\n(4) FILL REALISM.")
    say("  (4a) Against the trade record: a pseudo-random sample of markets with an order under the headline rule (every trade from placement to close)."
        "\n       'through' = contracts a taker bought on the other side at a price strictly through ours after placement. Capped result: we are filled on"
        "\n       min(cap, contracts through) and the result is weighted by contracts.")
    out = {}
    for rule in (HEAD, "date", "open+1h"):
        r = rs[rule]
        for period in ("recent", "holdout", "both"):
            rows = []
            for i in range(len(r["ticker"])):
                v = have.get(r["ticker"][i])
                if not v or not v["complete"] or period not in (v["period"], "both"):
                    continue
                after = [t for t in v["trades"] if t[0] > r["at"][i]]
                offer, ybid = r["ask"][i]-1, r["bid"][i]+1
                rows.append((r["filled"][i], sum(t[2] for t in after if t[3] == 1 and t[1] > offer), any(t[1] > offer for t in after), r["net"][i], r["price"][i],
                             sum(t[2] for t in after if t[3] == 1 and t[1] >= offer),
                             r["yfilled"][i], sum(t[2] for t in after if t[3] == 0 and t[1] < ybid), any(t[1] < ybid for t in after), r["ynet"][i], r["yprice"][i],
                             sum(t[2] for t in after if t[3] == 0 and t[1] <= ybid)))
            if len(rows) < 15:
                continue
            a = np.array(rows, float)
            o = dict(markets=len(a))
            for side, k in (("NO", 0), ("mirror", 6)):
                bar, thr, tape, net, price, at = a[:, k], a[:, k+1], a[:, k+2], a[:, k+3], a[:, k+4], a[:, k+5]
                d = dict(bar_fill=float(bar.mean()), tape_fill=float(tape.mean()), agree=float((bar == tape).mean()),
                         median_through=float(np.median(thr[thr > 0])) if (thr > 0).any() else 0.0,
                         through_10=float((thr >= 10).mean()), through_100=float((thr >= 100).mean()),
                         bar_net=float(net[bar == 1].mean()) if bar.any() else None)
                for cap in (1, 10, 100):
                    q = np.minimum(cap, thr)
                    d[f"cap{cap}"] = dict(net=float((q*net).sum()/q.sum()), ret=float((q*net).sum()/(q*price).sum()), mean_filled=float(q.mean())) if q.sum() else None
                    q = np.minimum(cap, at)
                    d[f"at{cap}"] = dict(net=float((q*net).sum()/q.sum()), ret=float((q*net).sum()/(q*price).sum()), mean_filled=float(q.mean())) if q.sum() else None
                d["at_fill"] = float((at > 0).mean())
                o[side] = d
            out[f"{rule}|{period}"] = o
            for side in ("NO", "mirror"):
                d = o[side]
                cap = lambda c, k="cap": "n/a" if d[f"{k}{c}"] is None else f"{d[f'{k}{c}']['net']:+5.1f}c ({100*d[f'{k}{c}']['ret']:+5.1f}%, {d[f'{k}{c}']['mean_filled']:.1f}/order)"
                say(f"       {rule:8} {period:8} {side:6} markets {o['markets']:3}  bar-rule fill {100*d['bar_fill']:3.0f}%  tape fill {100*d['tape_fill']:3.0f}%  agree {100*d['agree']:3.0f}%  "
                    f"median through {d['median_through']:5.0f}  >=10: {100*d['through_10']:3.0f}%  >=100: {100*d['through_100']:3.0f}%  | bar-rule net {d['bar_net'] if d['bar_net'] is not None else float('nan'):+5.1f}c"
                    f"  cap 10: {cap(10)}  cap 100: {cap(100)}  || at or through: fill {100*d['at_fill']:3.0f}%  cap 10: {cap(10, 'at')}  cap 100: {cap(100, 'at')}")
    incomplete = sum(not v["complete"] for v in have.values())
    say(f"       trade record on disk for {len(have)} markets ({incomplete} longer than {TRADE_PAGES},000 trades after placement and left out)")
    summary["tape"] = out
    say("  (4b) Outbid. From the hourly bars that carry the lowest YES ask of each hour (recent markets and the new archive sample): a later best NO bid above ours"
        "\n       (a YES ask below our offer). 'before fill' = in an hour before the hour of our fill, or never filled; 'same hour' = cannot be ordered within the hour.")
    say(f"       {'rule':10} {'period':8} {'orders':>6} {'improved on, ever':>17} {'median h to it':>14} {'before fill / unfilled':>22} {'same hour as fill':>17} | "
        f"NO net c of fills: not improved on first / improved on first | unfilled: improved on %")
    for rule in (HEAD, "date", "open+1h", "first2c"):
        r = rs[rule]
        for period in ("recent", "holdout"):
            mask = (r["period"] == period) & r["detail"]
            if mask.sum() < 30:
                continue
            f = mask & r["filled"]
            ever = mask & ~np.isnan(r["beat_lag"])
            first = f & r["beat_first"]
            clean = f & ~r["beat_first"] & ~r["beat_same"]
            o = dict(orders=int(mask.sum()), ever=float(ever.sum()/mask.sum()), median_hours=float(np.nanmedian(r["beat_lag"][ever])) if ever.any() else None,
                     before=float((mask & r["beat_first"]).sum()/mask.sum()), same=float((mask & r["beat_same"]).sum()/mask.sum()),
                     net_clean=float(r["net"][clean].mean()) if clean.any() else None, net_beaten=float(r["net"][first].mean()) if first.any() else None,
                     n_clean=int(clean.sum()), n_beaten=int(first.sum()), unfilled_beaten=float((mask & ~r["filled"] & r["beat_first"]).sum()/max(1, (mask & ~r["filled"]).sum())))
            summary.setdefault("outbid", {})[f"{rule}|{period}"] = o
            nz = lambda v: float("nan") if v is None else v
            say(f"       {rule:10} {period:8} {o['orders']:6} {100*o['ever']:16.0f}% {nz(o['median_hours']):14.1f} {100*o['before']:21.0f}% {100*o['same']:16.0f}% | "
                f"{nz(o['net_clean']):+6.1f} ({o['n_clean']}) / {nz(o['net_beaten']):+6.1f} ({o['n_beaten']}) | {100*o['unfilled_beaten']:3.0f}%")


# ---------------------------------------------------------------- tail

def capped(r, mask, most, by):
    """Keep at most `most` orders per event, ranked by a quantity known at placement."""
    key = dict(widest=-r["spread"], lowest=r["mid"], random=np.array([int(pick_hash("cap", t)[:8], 16) for t in r["ticker"]], float))[by]
    keep = np.zeros(len(mask), bool)
    idx = np.flatnonzero(mask)
    order = idx[np.lexsort((r["at"][idx], key[idx], r["g"][idx]))]
    rank = np.zeros(len(order), int)
    g = r["g"][order]
    start = np.r_[0, np.flatnonzero(g[1:] != g[:-1])+1]
    rank = np.arange(len(order))-np.repeat(start, np.diff(np.r_[start, len(order)]))
    keep[order[rank < most]] = True
    return keep


def tail(rs, days, say, summary):
    say("\n(6) TAIL. A book of 10 contracts per market: filled orders only, dollars. Recent period = whole events (every word of every event);"
        "\n    the holdout is a sample of at most 5 words per new event (about half the words for the 41 events of the earlier holdout), so its event totals are already capped.")
    say(f"    {'rule':8} {'period':8} {'cap per event':16} {'fills':>5} {'net c':>6} {'ret %':>22} {'total $':>8} {'$ per day':>9} {'worst event':>42} {'worst day':>28} {'losing days':>11} {'worst 5-day run $':>17}")
    out = {}
    for rule in (HEAD, "date", "open+1h"):
        r = rs[rule]
        for period in ("recent", "holdout"):
            base = r["period"] == period
            for label, most, by in (("none", 10**6, "random"), ("5 widest spread", 5, "widest"), ("10 widest spread", 10, "widest"), ("5 lowest YES", 5, "lowest"),
                                    ("10 lowest YES", 10, "lowest"), ("5 random", 5, "random"), ("10 random", 10, "random")):
                if period == "holdout" and label != "none":
                    continue
                f = capped(r, base, most, by) & r["filled"]
                if f.sum() < 20:
                    continue
                pnl, stake = r["net"][f]/10, r["price"][f]/10
                ev, ev_stake, ev_n, ev_lost = (np.bincount(r["g"][f], w) for w in (pnl, stake, np.ones(f.sum()), r["y"][f].astype(float)))
                day, day_stake = np.bincount(r["d"][f], pnl), np.bincount(r["d"][f], stake)
                wi = int(np.argmin(np.where(ev_n > 0, ev, np.inf)))
                used = day_stake > 0
                wd = int(np.argmin(np.where(used, day, np.inf)))
                name = r["event"][f][r["g"][f] == wi][0]
                in_order = sorted(np.flatnonzero(used), key=lambda k: days[k])
                run = min(sum(day[k] for k in in_order[j:j+5]) for j in range(max(1, len(in_order)-4)))
                ret = wide(r, [(r["net"], r["price"], f)])
                o = dict(fills=int(f.sum()), net=float(r["net"][f].mean()), ret=ret, total=float(pnl.sum()), days=int(used.sum()), per_day=float(pnl.sum()/used.sum()),
                         worst_event=dict(name=name, pnl=float(ev[wi]), stake=float(ev_stake[wi]), fills=int(ev_n[wi]), lost=int(ev_lost[wi])),
                         worst_day=dict(day=days[wd], pnl=float(day[wd]), stake=float(day_stake[wd])), losing_days=float((day[used] < 0).mean()), worst_run=float(run),
                         event_percentiles=[float(x) for x in np.percentile(ev[ev_n > 0], [1, 5, 25, 50, 75, 95])])
                out[f"{rule}|{period}|{label}"] = o
                say(f"    {rule:8} {period:8} {label:16} {o['fills']:5} {o['net']:+6.1f} {show(ret, 100)} {o['total']:+8.0f} {o['per_day']:+9.2f} "
                    f"{name[-24:]:>24} {ev[wi]:+5.0f} on {ev_stake[wi]:4.0f} ({int(ev_lost[wi])}/{int(ev_n[wi])} lost) {days[wd]:>10} {day[wd]:+6.0f} on {day_stake[wd]:5.0f} "
                    f"{100*o['losing_days']:10.0f}% {run:+17.0f}")
    summary["tail"] = out


# ---------------------------------------------------------------- extension families

EXT = ("Netflix charts", "Music charts", "Rotten Tomatoes scores", "Reality TV", "Recurring politics", "Entertainment other", "Apps search and social counts",
       "Box office", "Social and viral")
EXT_REFS = (24, 6, 1)
# Archive holdout: chosen AFTER reading the recent table. family -> (series, markets to fetch, minimum lifetime volume of a sampled market).
# The archive costs one request per market and most chart markets never have a two-sided book, so only families with a usable
# yield were worth the requests: recurring politics (the approval-rating series) and the Netflix charts (active markets only).
EXT_HOLD = {"Recurring politics": (("KXTRUMPAPPROVE", "KXAPRPOTUS"), 140, 0),
            "Netflix charts": (("KXNETFLIXTOPVIEWSMOVIE", "KXNETFLIXTOPVIEWSTV", "KXNETFLIXRANKMOVIEGLOBAL"), 80, 1000)}


def ext_rows(ms, bars, period):
    """Orders at 24h / 6h / 1h before the event's scheduled (most common) close, resting to that close. Same arithmetic as the mention rules."""
    closes = defaultdict(Counter)
    for m in ms:
        closes[m["event"]][m["close"]//600] += 1
    mode = {g: 600*min(c, key=lambda k: (-c[k], k)) for g, c in closes.items()}
    rows = []
    for m in ms:
        bs = bars.get(m["ticker"])
        if not bs:
            continue
        t0 = mode[m["event"]]
        for h in EXT_REFS:
            tq = (t0-3600*h)//3600*3600
            if m["open"] > tq or m["close"] <= tq:
                continue
            past = [b for b in bs if tq-86400 <= b[0] <= tq]
            if not past or not two_sided(past[-1]):
                continue
            bid, ask = past[-1][1], past[-1][2]
            later = [b for b in bs if tq < b[0] <= t0+3600 and b[5] > 0]
            y = m["result"] == "yes"
            fee = lambda p: 1.75*p/100*(1-p/100) if m.get("maker_fee") else 0.0
            rows.append(dict(h=h, bid=bid, ask=ask, y=y, filled=any(b[4] is not None and b[4] > ask-1 for b in later),
                             yfilled=any(b[3] is not None and b[3] < bid+1 for b in later), fee=fee(ask-1), yfee=fee(bid+1), event=m["event"],
                             day=str(datetime.fromtimestamp(t0, EASTERN).date()), t=t0, period=period, volume=m["volume"]))
    return rows


def ext_table(rows):
    if not rows:
        return None
    r = {k: np.array([x[k] for x in rows]) for k in rows[0]}
    r["g"] = np.unique(r["event"], return_inverse=True)[1]
    r["d"] = np.unique(r["day"], return_inverse=True)[1]
    r["mid"], r["spread"] = (r["bid"]+r["ask"])/2, r["ask"]-r["bid"]
    r["price"], r["yprice"] = 100-(r["ask"]-1), r["bid"]+1
    r["net"], r["ynet"] = 100*(~r["y"])-r["price"]-r["fee"], 100*r["y"]-r["yprice"]-r["yfee"]
    r["gap"], r["lead"] = 100*r["y"]-r["mid"], r["h"].astype(float)
    return r


def ext_recent(name):
    fam = hype_no.families()
    ms = [dict(m, maker_fee=m["maker_fee"]) for m in hype_no.family_markets(name, fam)]
    return ext_rows(ms, load(OLD/"bars"/f"{hype_no.slug(name)}.json", {}), "recent")


def ext_step(client):
    for name, (series, cap, least) in EXT_HOLD.items():
        listing_step(client, list(series), f"ext-markets/{hype_no.slug(name)}", most_pages=2)
        ms = [dict(m, series=p.stem) for p in sorted((RAW/f"ext-markets/{hype_no.slug(name)}").glob("*.json")) for m in load(p, dict(markets=[]))["markets"]]
        by_event = defaultdict(list)
        for m in ms:
            if m["volume"] >= least:
                by_event[m["event"]].append(m)
        events = sorted(by_event, key=lambda e: pick_hash("ext", e))
        order, depth = [], 0
        while len(order) < cap and any(len(by_event[e]) > depth for e in events):      # one market per event per pass
            order += [sorted(by_event[e], key=lambda m: pick_hash("extm", m["ticker"]))[depth] for e in events if len(by_event[e]) > depth]
            depth += 1
        path = RAW/"ext-bars"/f"{hype_no.slug(name)}.json"
        have = load(path, {})
        closes = defaultdict(Counter)
        for m in ms:
            closes[m["event"]][m["close"]//600] += 1
        mode = {g: 600*min(c, key=lambda k: (-c[k], k)) for g, c in closes.items()}
        try:
            for m in order[:cap]:
                if m["ticker"] in have:
                    continue
                t0 = mode[m["event"]]
                data = read(client, f"/historical/markets/{m['ticker']}/candlesticks", dict(start_ts=t0-49*3600, end_ts=t0+7200, period_interval=60))
                have[m["ticker"]] = compact((data or {}).get("candlesticks"))
        finally:
            dump(path, have)
        print(f"  {name}: archive bars for {len(have)} markets of {len(ms)} listed in {len(by_event)} events", flush=True)


def extension(say, summary):
    say("\n(5) EXTENSION: the same resting NO bid and mirror in other wide-book, retail-heavy families. These markets close on a schedule, so the order goes in"
        "\n    24h, 6h or 1h before the scheduled close and rests to the close. Recent = bars already on disk from research/hype_no.py; holdout = archive sample.")
    say("  "+HEADER)
    meta = hype_no.families()["meta"]
    pooled = defaultdict(list)
    for name in EXT:
        sets = [("recent", ext_recent(name))]
        folder = RAW/f"ext-markets/{hype_no.slug(name)}"
        if folder.exists():
            ms = [dict(m, series=p.stem, maker_fee=meta.get(p.stem, {}).get("fee_type") == "quadratic_with_maker_fees")
                  for p in sorted(folder.glob("*.json")) for m in load(p, dict(markets=[]))["markets"]]
            sets.append(("HOLDOUT", ext_rows([m for m in ms if m["volume"] >= EXT_HOLD[name][2]], load(RAW/"ext-bars"/f"{hype_no.slug(name)}.json", {}), "holdout")))
            only = [x for x in sets[0][1] if x["event"].split("-")[0] in EXT_HOLD[name][0] and x["volume"] >= EXT_HOLD[name][2]]
            sets.insert(1, ("recent, holdout series", only))
        for label, rows in sets:
            if label == "recent":
                for x in rows:
                    pooled[x["h"]].append(x)
            r = ext_table(rows)
            for h in EXT_REFS:
                o = line(r, r["h"] == h) if r is not None else None
                if o is not None:
                    say("  "+text(f"{name[:20]} {h}h {label[:14]}", o))
                    summary.setdefault("extension", {})[f"{name}|{h}|{label}"] = o
    r = ext_table([x for v in pooled.values() for x in v])
    for h in EXT_REFS:
        o = line(r, r["h"] == h)
        say("  "+text(f"ALL EXTENSION FAMILIES {h}h recent", o))
        summary["extension"][f"ALL|{h}|recent"] = o


def forward(rs, say, summary):
    say("\n(7) SCALE. Fills per calendar day in the recent period (every mention market, 62 days) and what 10 contracts a fill would make per day at the"
        "\n    holdout result, at the recent result, and at the lower end of the holdout interval.")
    say(f"    {'rule':10} {'orders/day':>10} {'fills/day':>9} {'paid c':>6} {'holdout c':>22} {'recent c':>9} {'$ / day at holdout':>18} {'at recent':>9} {'at holdout low end':>18}")
    for rule in RULES:
        r = rs[rule]
        rec, hold = r["period"] == "recent", r["period"] == "holdout"
        one = np.ones(len(rec))
        h = wide(r, [(r["net"], one, hold & r["filled"])])
        per_day = (rec & r["filled"]).sum()/62
        recent = float(r["net"][rec & r["filled"]].mean())
        o = dict(orders_per_day=float(rec.sum()/62), fills_per_day=float(per_day), paid=float(r["price"][rec & r["filled"]].mean()), holdout=h, recent=recent,
                 dollars_holdout=float(per_day*h[0]/10), dollars_recent=float(per_day*recent/10), dollars_low=float(per_day*h[1]/10) if h[1] is not None else None)
        summary.setdefault("forward", {})[rule] = o
        say(f"    {rule:10} {o['orders_per_day']:10.1f} {per_day:9.1f} {o['paid']:6.1f} {show(h)} {recent:+9.1f} {o['dollars_holdout']:+18.2f} {o['dollars_recent']:+9.2f} "
            f"{o['dollars_low'] if o['dollars_low'] is not None else float('nan'):+18.2f}")


# ---------------------------------------------------------------- main

def locked(run):
    lock = RAW/"download.lock"
    RAW.mkdir(parents=True, exist_ok=True)
    if lock.exists():
        try:
            os.kill(int(lock.read_text()), 0)
            sys.exit("another download is running (download.lock)")
        except (ValueError, ProcessLookupError):
            pass
    lock.write_text(str(os.getpid()))
    try:
        with httpx.Client(timeout=120) as client:
            run(client)
    finally:
        lock.unlink(missing_ok=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("steps", nargs="+", choices=("series", "listings", "pick", "abars", "rbars", "trades", "ext", "report"))
    parser.add_argument("--minutes", type=float, default=9)
    parser.add_argument("--top", type=int, default=50, help="`listings`: how many series by lifetime volume")
    parser.add_argument("--earnings", type=int, default=25, help="`listings`: plus this many further earnings-call series")
    parser.add_argument("--cap", type=int, default=1200, help="`abars`: markets of the sample order to fetch")
    args = parser.parse_args()
    for step in args.steps:
        began = time.time()
        deadline[0] = began+60*args.minutes
        log.clear()
        if step == "pick":
            pick()
        elif step == "report":
            report()
        else:
            def run(client):
                try:
                    if step == "series":
                        series_step(client)
                    elif step == "listings":
                        rows = load(RAW/"series.json", [])
                        listing_step(client, [s["ticker"] for s in rows[:args.top]]
                                     + [s["ticker"] for s in rows[args.top:] if "EARNINGS" in s["ticker"]][:args.earnings], "archive-markets")
                    elif step == "abars":
                        abars_step(client, args.cap)
                    elif step == "rbars":
                        rbars_step(client)
                    elif step == "trades":
                        trades_step(client)
                    elif step == "ext":
                        ext_step(client)
                except OutOfTime:
                    print(f"[{step}] out of time; run again to resume", flush=True)
                finally:
                    old = load(RAW/"download-log.json", [])
                    old.append(dict(step=step, seconds=round(time.time()-began), **log))
                    dump(RAW/"download-log.json", old)
                    print(f"[{step}] {dict(log)} in {time.time()-began:.0f}s", flush=True)
            locked(run)


if __name__ == "__main__":
    main()
