"""AUDIT COPY of research/resting_no_wide.py for power checks (planted edge, minimum detectable edge, mention markets).
Offline only: the download steps are not reachable from this copy's main(). Changes are marked AUDIT.

Original docstring follows.

Resting NO bids outside mention markets: does the edge exist where there is more money?

Research only. Public Kalshi reads, no orders, no authentication.

research/resting_no.py found that in "mention" markets a NO bid resting one cent
above the best NO bid earned about +16% a fill while the same order on the YES
side (the mirror) lost. That edge holds about ten contracts a word. This script
applies the same order, the same mirror control and the same fill rule to other
non-sports categories: Entertainment, Politics, Elections, Science and
Technology, Companies, and the small World / Social / Health / Culture lists.
Sports, Mentions, weather, gas, Economics, Financials and Crypto are left out.

Fixed before any result was read

* Order: a NO bid one cent above the best NO bid (a YES offer at ask - 1) when
  the quote is two-sided (1 <= bid < ask <= 99) and at least 2c wide, held to
  settlement. Mirror: a YES bid at bid + 1. A bid is filled only when a LATER
  bar prints a trade strictly through it (the `prints` rule of paper/rules.py
  applied to bars: trade high above the offer for NO, trade low below the bid
  for the mirror). Maker fee none on "quadratic" series, 0.0175*p*(1-p) on
  "quadratic_with_maker_fees".
* Rules, all knowable live and none anchored on the close (many of these markets
  close early when the thing happens, so "hours before the close" would select
  on the outcome):
    first2c   the first bar close after listing with a two-sided book >= 2c;
              the order rests until the market closes.            <- headline
    open+24h  24 hours after open_time (the next 3 hourly closes are tried);
              rests until the market closes.
    weekly    every Monday 05:00 UTC (midnight Eastern, where daily bars end)
              while the market is open: a fresh order that rests SEVEN DAYS and
              is then cancelled. One market gives many orders; they are
              clustered by event. This is the rule that matters for markets
              that live for months, and the one used for dollars per month.
* Families (`family` below): by category and series tags, 20 of them (no box
  office series has 1M lifetime contracts, so there is no such family).
* Markets: lifetime volume of at least 500 contracts (a market nobody trades
  cannot fill an order and holds no money). Recent = settled markets closing
  2026-09-18 to 2026-11-18, whole events in a fixed pseudo-random order up to a
  cap per family. Holdout = archived markets closing 2026-01-01 to 2026-09-17,
  at most 2 markets per event, events round-robin over family x month.
* Evidence: a family passes only if NO net per fill is above zero AND above the
  mirror in both chronological halves of its events, and the 95% interval of
  NO net per fill (the wider of the event-clustered and the day-clustered
  bootstrap) excludes zero. With about twenty families and two rules, one pass
  is what chance gives.

Quotes are reconstructed, not recorded. The best bid and ask at the time of the
order are the yes_bid / yes_ask CLOSE of the candlestick ending at the placement
instant: hourly bars for the first week of a market's life, DAILY bars (ending
05:00 UTC) after that. See the report text for what that can bias.

Commands (each resumable; --minutes bounds the network time of that call)

    series    GET /series per category -> families
    listings  settled markets per series, recent (GET /markets) and archive
    pick      offline: the samples
    bars      candlesticks: recent through the batch endpoint, archive one a market
    trades    trade record for a sample of filled orders (capacity)
    report    offline -> data/processed/resting-no-wide/report.txt, summary.json

Network: one request at a time, 2 seconds apart, 30 seconds or more of back-off
on HTTP 429 (live programs share the limit).
"""
import argparse
from collections import Counter, defaultdict
from datetime import datetime, timezone
import os
from pathlib import Path
import re
import sys
import time

import httpx
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))   # AUDIT: one level deeper
from research import hype_no, resting_no
from research.api_cost import ROOT
from research.hype_no import slim, entertainment_family
from research.resting_no import (read, load, dump, compact, two_sided, wide, show, pick_hash, deadline, log, OutOfTime, EASTERN,
                                 RECENT_FROM, RECENT_TO, HOLD_FROM)

resting_no.PACE = 2.0
RAW = ROOT / "data/raw/resting-no-wide"
OUT = ROOT / "data/processed/audit/resting_no_wide"   # AUDIT
OLD = ROOT / "data/raw/hype-no"
CATEGORIES = ("Entertainment", "Politics", "Elections", "Science and Technology", "Companies", "World", "Social", "Culture", "Health")
BIG = ("Entertainment", "Politics", "Elections", "Science and Technology")     # series need 1M lifetime contracts; the others 100k
MIN_VOLUME = 500
RECENT_CAP, HOLD_CAP, PER_EVENT = 1200, 120, 2
HOURLY_DAYS = 7
DAY_END = 5*3600                     # daily bars end at 05:00 UTC
SKIP = re.compile(r"^KX(RAIN|HIGH|LOW|TEMP|SNOW|HURR|AAAGAS|FLIGHT|ORDDLY)|MENTION")


# ---------------------------------------------------------------- families

def family(s):
    cat, tags, t = s["category"], set(s.get("tags") or []), s["ticker"]
    title = s["title"].lower()
    if cat == "Entertainment":
        if tags & {"Awards", "Oscars", "Grammys", "Emmys", "The Game Awards", "Tonys", "VMA", "Eurovision"}:
            return "Ent awards"
        name = entertainment_family(s)
        return {"Rotten Tomatoes scores": "Ent Rotten Tomatoes", "Netflix charts": "Ent Netflix charts",
                "Music charts": "Ent music charts", "Reality TV": "Ent reality TV"}.get(name, "Ent other")
    if cat == "Politics":
        if "Recurring" in tags or s.get("frequency") in ("daily", "weekly", "monthly"):
            return "Pol recurring (approval, counts)"
        if tags & {"International", "Iran", "Shipping lanes", "Venezuela", "Global institutions", "World leaders", "Latin America", "Trade & tariffs"}:
            return "Pol international"
        if tags & {"Congress", "Budget & shutdowns", "Other legislation", "Tax & economy bills", "SCOTUS & courts", "Impeachment", "Supreme Court & judges",
                   "Health & social bills", "Congress & courts", "Taxes & spending", "Filibuster & vetoes", "Crypto policy"}:
            return "Pol Congress and courts"
        if tags & {"Trump", "Trump administration", "Cabinet & staff", "Departures", "Executive actions", "Travel & meetings", "SOTU", "Territory & renaming",
                   "Prosecutions", "Media & press"}:
            return "Pol Trump and administration"
        return "Pol other"
    if cat == "Elections":
        if tags & {"International elections", "Foreign Elections", "Brazil", "Iran", "Peru", "Germany & France", "Hungary", "Africa & Middle East",
                   "South Asia & Pacific", "East Asia", "International"}:
            return "Elec international"
        if tags & {"Election Combos", "Mixed Combos", "Senate Combos", "House Combos"} or re.search(r"margin|combo|how many|seats|turnout|vote share", title):
            return "Elec combos and margins"
        if tags & {"Primaries", "Nomination race", "2028", "Who runs", "Leadership races"} or re.search(r"primary|nomin", title):
            return "Elec primaries and nominations"
        if tags & {"Senate", "Senate races", "House", "House races", "House control", "Governor", "Governor races", "Presidential winner", "State legislatures"}:
            return "Elec Senate House Governor"
        return "Elec other (mayors, local)"
    if cat == "Science and Technology":
        if tags & {"Space", "SpaceX", "Public Health", "Energy"}:
            return "Science space and other"
        return "Tech AI and products" if tags & {"AI", "Product launches", "Market Share", "Google", "Tesla", "Companies", "Big Tech & Business"} else "Science space and other"
    if cat == "Companies":
        return "Companies"
    return "World Social Health Culture"


FAMILIES = ("Ent Rotten Tomatoes", "Ent Netflix charts", "Ent music charts", "Ent reality TV", "Ent awards", "Ent other",
            "Pol recurring (approval, counts)", "Pol Trump and administration", "Pol Congress and courts", "Pol international", "Pol other",
            "Elec Senate House Governor", "Elec primaries and nominations", "Elec combos and margins", "Elec international", "Elec other (mayors, local)",
            "Tech AI and products", "Science space and other", "Companies", "World Social Health Culture")


def series_step(client):
    raw = load(RAW/"series-raw.json", {})
    for cat in CATEGORIES:
        if cat not in raw:
            raw[cat] = (read(client, "/series", dict(category=cat, include_volume="true")) or {}).get("series") or []
            dump(RAW/"series-raw.json", raw)
    rows = []
    for cat in CATEGORIES:
        for s in raw[cat]:
            vol = float(s.get("volume_fp") or 0)
            if vol < (1e6 if cat in BIG else 1e5) or SKIP.search(s["ticker"]):
                continue
            rows.append(dict(ticker=s["ticker"], title=s["title"], volume=vol, category=cat, tags=s.get("tags") or [], frequency=s.get("frequency"),
                             fee_type=s.get("fee_type"), family=family(s)))
    rows.sort(key=lambda s: -s["volume"])
    dump(RAW/"series.json", rows)
    count, vol = Counter(s["family"] for s in rows), Counter()
    for s in rows:
        vol[s["family"]] += s["volume"]
    print(f"{len(rows)} series; fee types {dict(Counter(s['fee_type'] for s in rows))}")
    for f in FAMILIES:
        print(f"  {f:36} {count[f]:4} series {vol[f]/1e6:8.0f}M lifetime contracts | "+", ".join(s["ticker"] for s in rows if s["family"] == f)[:150], flush=True)


def series_rows():
    return load(RAW/"series.json", [])


# ---------------------------------------------------------------- listings

def recent_listing(client, order):
    for n, series in enumerate(order):
        path = RAW/"recent-markets"/f"{series}.json"
        if path.exists():
            continue
        old = load(OLD/"markets"/f"{series}.json", None)            # listed by research/hype_no.py with the same window and fields
        if old is not None and old.get("done") and not old.get("truncated"):
            dump(path, dict(markets=old["markets"], pages=0, reused=True))
            continue
        have, cursor, seen = dict(markets=[], pages=0, truncated=False), "", set()
        while True:
            data = read(client, "/markets", dict(status="settled", limit=1000, min_close_ts=RECENT_FROM, max_close_ts=RECENT_TO, series_ticker=series,
                                                 **({"cursor": cursor} if cursor else {}))) or {}
            have["pages"] += 1
            have["markets"] += [slim(m) for m in data.get("markets", []) if m.get("result") in ("yes", "no")]
            cursor = data.get("cursor")
            if not cursor or cursor in seen or not data.get("markets"):
                break
            if have["pages"] >= 5:
                have["truncated"] = True
                break
            seen.add(cursor)
        dump(path, have)
        if n % 20 == 0 or have["markets"]:
            print(f"  recent {n+1}/{len(order)} {series:30} {len(have['markets']):5} settled markets", flush=True)


def archive_listing(client, order, most_pages=2):
    for n, series in enumerate(order):
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
                if m.get("result") in ("yes", "no") and v["close"] and v["open"] and HOLD_FROM <= v["close"] < RECENT_FROM:
                    have["markets"].append(dict(v, exp=v["exp"] or v["close"]))
            have["cursor"] = data.get("cursor") or ""
            have["done"] = bool(not have["cursor"] or not rows or have["pages"] >= most_pages or (oldest is not None and oldest < HOLD_FROM))
            dump(path, have)
        if n % 20 == 0 or have["markets"]:
            print(f"  archive {n+1}/{len(order)} {series:30} {len(have['markets']):5} markets closing Jan 1 - Sep 17", flush=True)


def listed(period):
    meta = {s["ticker"]: s for s in series_rows()}
    out = []
    for path in sorted((RAW/f"{period}-markets").glob("*.json")):
        s = meta.get(path.stem)
        if s is None:
            continue
        for m in load(path, dict(markets=[]))["markets"]:
            if m["open"] and m["close"] and m["close"] > m["open"]:
                out.append(dict(m, series=path.stem, family=s["family"], maker_fee=s["fee_type"] == "quadratic_with_maker_fees", period=period))
    return out


# ---------------------------------------------------------------- pick

def pick():
    sample, counts = {}, {}
    for period in ("recent", "archive"):
        ms = listed(period)
        events = defaultdict(list)
        for m in ms:
            events[m["event"]].append(m)
        size = {e: len(v) for e, v in events.items()}
        chosen = []
        print(f"{period}: {len(ms)} settled markets in {len(events)} events listed")
        print(f"  {'family':36} {'series':>6} {'markets':>7} {'events':>6} {'vol>=500':>8} {'events':>6} {'contracts M':>11} {'picked':>6} {'events':>6} {'median life d':>13}")
        for fam in FAMILIES:
            rows = [m for m in ms if m["family"] == fam]
            live = [m for m in rows if m["volume"] >= MIN_VOLUME]
            by = defaultdict(list)
            for m in live:
                by[m["event"]].append(m)
            got = []
            if period == "recent":
                for e in sorted(by, key=lambda e: pick_hash("event", e)):
                    if len(got) >= RECENT_CAP:
                        break
                    got += by[e]
            else:
                cells = defaultdict(list)
                for e, v in by.items():
                    cells[datetime.fromtimestamp(max(m["close"] for m in v), timezone.utc).strftime("%Y-%m")].append(e)
                for v in cells.values():
                    v.sort(key=lambda e: pick_hash("event", e))
                depth = 0
                while len(got) < HOLD_CAP and any(len(v) > depth for v in cells.values()):
                    for _, v in sorted(cells.items()):
                        if len(v) > depth and len(got) < HOLD_CAP:
                            got += sorted(by[v[depth]], key=lambda m: pick_hash("market", m["ticker"]))[:PER_EVENT]
                    depth += 1
            chosen += got
            life = np.median([(m["close"]-m["open"])/86400 for m in got]) if got else float("nan")
            print(f"  {fam:36} {len({m['series'] for m in rows}):6} {len(rows):7} {len({m['event'] for m in rows}):6} {len(live):8} {len(by):6} "
                  f"{sum(m['volume'] for m in rows)/1e6:11.1f} {len(got):6} {len({m['event'] for m in got}):6} {life:13.1f}", flush=True)
            if period == "recent":
                counts[fam] = len(live)
        sample[period] = [dict(m, eventsize=size[m["event"]]) for m in chosen]
    dump(RAW/"sample.json", sample)
    dump(RAW/"listed-counts.json", counts)
    print({k: len(v) for k, v in sample.items()})


def sample():
    return load(RAW/"sample.json", dict(recent=[], archive=[]))


# ---------------------------------------------------------------- bars

def hourly_end(m):
    """Hourly bars are kept up to the first 05:00 UTC at least HOURLY_DAYS after the open (or the close); daily bars after that."""
    t = m["open"]+HOURLY_DAYS*86400
    return min(-(-(t-DAY_END)//86400)*86400+DAY_END, m["close"]//3600*3600+7200)


def needs_daily(m):
    return m["close"] > hourly_end(m)


def batches(todo, start, end, unit, most):
    """Groups for the batch endpoint: at most 100 markets and `most` bar periods a request."""
    todo = sorted(todo, key=lambda m: (start(m), end(m), m["ticker"]))
    chunks, cur = [], []
    for m in todo:
        trial = cur+[m]
        span = (max(end(x) for x in trial)-start(trial[0]))//unit+3
        if cur and (len(trial) > 100 or len(trial)*span > most):
            chunks.append(cur)
            cur = []
        cur.append(m)
    return chunks+([cur] if cur else [])


def recent_bars(client):
    ms = sample()["recent"]
    for kind, interval, start, end, pool in (("hourly", 60, lambda m: m["open"]//3600*3600-3600, hourly_end, ms),
                                             ("daily", 1440, lambda m: m["open"]//86400*86400-86400, lambda m: m["close"]//86400*86400+2*86400,
                                              [m for m in ms if needs_daily(m)])):
        path = RAW/f"recent-{kind}.json"
        have = load(path, {})
        chunks = batches([m for m in pool if m["ticker"] not in have], start, end, 60*interval, 9000)
        print(f"recent {kind}: {len(have)} on disk, {sum(len(c) for c in chunks)} markets in {len(chunks)} requests", flush=True)
        try:
            for n, group in enumerate(chunks):
                data = read(client, "/markets/candlesticks", dict(market_tickers=",".join(m["ticker"] for m in group), start_ts=start(group[0]),
                                                                  end_ts=max(end(m) for m in group), period_interval=interval))
                if data is None:
                    log["batch_refused"] += 1
                    if len(group) > 1:                                # try the markets one by one next time round
                        continue
                    have[group[0]["ticker"]] = []
                    continue
                got = {x["market_ticker"]: compact(x.get("candlesticks")) for x in data.get("markets", [])}
                for m in group:
                    have[m["ticker"]] = [b for b in got.get(m["ticker"], []) if start(m) <= b[0] <= end(m)]
                if n % 10 == 9:
                    dump(path, have)
        finally:
            dump(path, have)


def archive_bars(client):
    ms = sample()["archive"]
    path = RAW/"archive-bars.json"
    have = load(path, {})
    todo = [m for m in ms if m["ticker"] not in have]
    todo.sort(key=lambda m: (pick_hash("order", m["event"]), m["ticker"]))     # families interleaved, so a short run is still balanced
    print(f"archive bars: {len(have)} on disk, {len(todo)} to fetch", flush=True)
    try:
        for n, m in enumerate(todo):
            url = f"/historical/markets/{m['ticker']}/candlesticks"
            if m["close"]-m["open"] <= 150*86400:                    # one hourly request covers the whole life
                data = read(client, url, dict(start_ts=m["open"]//3600*3600-3600, end_ts=m["close"]//3600*3600+7200, period_interval=60))
                have[m["ticker"]] = dict(hourly=compact((data or {}).get("candlesticks")), daily=None)
            else:
                a = read(client, url, dict(start_ts=m["open"]//3600*3600-3600, end_ts=hourly_end(m), period_interval=60))
                b = read(client, url, dict(start_ts=m["open"]//86400*86400-86400, end_ts=m["close"]//86400*86400+2*86400, period_interval=1440))
                have[m["ticker"]] = dict(hourly=compact((a or {}).get("candlesticks")), daily=compact((b or {}).get("candlesticks")))
            if n % 25 == 24:
                dump(path, have)
    finally:
        dump(path, have)
    print(f"archive bars for {len(have)} markets", flush=True)


def bars_of(m, stores):
    """(bars, cut) per market: hourly bars, then daily bars from `cut` where the hourly ones stop (cut None: hourly throughout).
    (None, None) if nothing was downloaded."""
    if m["period"] == "recent":
        if m["ticker"] not in stores["rh"] or (needs_daily(m) and m["ticker"] not in stores["rd"]):
            return None, None
        hourly, daily = stores["rh"][m["ticker"]], stores["rd"].get(m["ticker"]) if needs_daily(m) else None
    else:
        v = stores["a"].get(m["ticker"])
        if v is None:
            return None, None
        hourly, daily = v["hourly"], v["daily"]
    if daily is None:
        return [b for b in hourly if b[0] <= m["close"]+7200], None
    cut = hourly_end(m)
    return [b for b in hourly if b[0] <= cut]+[b for b in daily if b[0] > cut and b[0]-86400 < m["close"]], cut


# ---------------------------------------------------------------- orders

RULES = ("first2c", "open+24h", "weekly")
HEAD = "first2c"
WEEK = 7*86400
MONDAY = 4*86400+DAY_END             # 1970-01-05 05:00 UTC, a Monday


def order_at(m, bars, ends, t, patient, rest, cut):
    """The order placed at the first bar close at or after t with a two-sided book >= 2c (within 3 hours unless patient), and what the
    later bars say (within `rest` seconds of placement, or to the close). The arithmetic of resting_no.simulate."""
    if t >= m["close"]:
        return None
    i = int(np.searchsorted(ends, t, side="right"))-1
    if i < 0 or t-ends[i] > 86400:
        i += 1
        if i >= len(bars) or (bars[i][0] > t+3*3600 and not patient):
            return None
    stop = float("inf") if patient else t+3*3600
    while i < len(bars) and not two_sided(bars[i]):
        i += 1
        if i >= len(bars) or bars[i][0] > stop:
            return None
    if i >= len(bars) or bars[i][0] >= m["close"]:
        return None
    at, bid, ask = max(bars[i][0], t), bars[i][1], bars[i][2]
    later = [b for b in bars[i+1:] if b[5] > 0 and (rest is None or b[0] <= at+rest)]
    fill = next((b for b in later if b[4] is not None and b[4] > ask-1), None)
    yfill = next((b for b in later if b[3] is not None and b[3] < bid+1), None)
    return dict(at=at, bid=bid, ask=ask, filled=fill is not None, yfilled=yfill is not None, touched=any(b[4] is not None and b[4] >= ask-1 for b in later),
                ytouched=any(b[3] is not None and b[3] <= bid+1 for b in later), fill_ts=fill[0] if fill else None,
                daily=cut is not None and bars[i][0] > cut)


def orders(m, bars, cut):
    ends = [b[0] for b in bars]
    first = -(-m["open"]//3600)*3600
    out = []
    o = order_at(m, bars, ends, first, True, None, cut)
    if o:
        out.append(("first2c", o))
    o = order_at(m, bars, ends, -(-(m["open"]+86400)//3600)*3600, False, None, cut)
    if o:
        out.append(("open+24h", o))
    t = -(-(max(first, m["open"]+3600)-MONDAY)//WEEK)*WEEK+MONDAY
    while t < m["close"]:
        o = order_at(m, bars, ends, t, False, WEEK, cut)
        if o:
            out.append(("weekly", o))
        t += WEEK
    return out


def build(ms=None, getbars=None):
    # AUDIT: optional market list and bar source (for the mention-market run); fee columns kept so outcomes can be replaced
    if ms is None:
        stores = dict(rh=load(RAW/"recent-hourly.json", {}), rd=load(RAW/"recent-daily.json", {}), a=load(RAW/"archive-bars.json", {}))
        s = sample()
        ms = [dict(m, period="recent") for m in s["recent"]]+[dict(m, period="archive") for m in s["archive"]]
        getbars = lambda m: bars_of(m, stores)
    rows, cover = [], defaultdict(Counter)
    for m in ms:
        bars, cut = getbars(m)
        key = (m["period"], m["family"])
        cover[key]["picked"] += 1
        if bars is None:
            continue
        cover[key]["with bars"] += 1
        cover[key]["empty"] += not bars
        y = m["result"] == "yes"
        for rule, o in orders(m, bars, cut) if bars else []:
            cover[key][rule] += 1
            offer, ybid = o["ask"]-1, o["bid"]+1
            fee = (lambda p: 1.75*p/100*(1-p/100)) if m["maker_fee"] else (lambda p: 0.0)
            rows.append(dict(rule=rule, ticker=m["ticker"], event=m["event"], series=m["series"], family=m["family"], period=m["period"], y=y,
                             bid=o["bid"], ask=o["ask"], at=o["at"], filled=o["filled"], yfilled=o["yfilled"], touched=o["touched"], ytouched=o["ytouched"],
                             fill_ts=o["fill_ts"] or 0, daily=o["daily"], close=m["close"], open=m["open"], volume=m["volume"], eventsize=m["eventsize"],
                             price=100-offer, net=100*(not y)-(100-offer)-fee(offer), yprice=ybid, ynet=100*y-ybid-fee(ybid), fee=fee(offer), yfee=fee(ybid),
                             day=str(datetime.fromtimestamp(m["close"], EASTERN).date())))
    r = {k: np.array([x[k] for x in rows]) for k in rows[0]}
    r["g"] = np.unique(r["event"], return_inverse=True)[1]
    r["d"] = np.unique(r["day"], return_inverse=True)[1]
    r["mid"], r["spread"] = (r["bid"]+r["ask"])/2, r["ask"]-r["bid"]
    r["gap"] = 100*r["y"]-r["mid"]
    r["lead"] = (r["close"]-r["at"])/86400
    return r, ms, {f"{k[0]} | {k[1]}": dict(v) for k, v in cover.items()}


def halves(r, mask):
    """First and second chronological half of the EVENTS under the mask (by the event's last close)."""
    last = {}
    for g, c in zip(r["g"][mask], r["close"][mask]):
        last[g] = max(last.get(g, 0), c)
    order = sorted(last, key=lambda g: (last[g], g))
    first = np.isin(r["g"], order[:len(order)//2])
    return mask & first, mask & ~first


def line(r, mask, least=20):
    n = int(mask.sum())
    if n < least:
        return None
    one = np.ones(len(mask))
    f, yf = mask & r["filled"], mask & r["yfilled"]
    o = dict(orders=n, markets=len(set(r["ticker"][mask])), events=len(set(r["g"][mask])), days=len(set(r["d"][mask])), fills=int(f.sum()), mirror_fills=int(yf.sum()),
             spread=float(r["spread"][mask].mean()), mid=float(r["mid"][mask].mean()), gap=wide(r, [(r["gap"], one, mask)]))
    if f.sum() >= 10:
        o.update(net=wide(r, [(r["net"], one, f)]), ret=wide(r, [(r["net"], r["price"], f)]), paid=float(r["price"][f].mean()), won=float((~r["y"][f]).mean()),
                 fill_events=len(set(r["g"][f])))
    if yf.sum() >= 10:
        o.update(mirror_net=wide(r, [(r["ynet"], one, yf)]), mirror_ret=wide(r, [(r["ynet"], r["yprice"], yf)]), mirror_won=float(r["y"][yf].mean()))
    if f.sum() >= 10 and yf.sum() >= 10:
        o["diff"] = wide(r, [(r["net"], one, f), (r["ynet"], one, yf)])
    return o


HEADER = (f"{'':40} {'ord':>5} {'mkt':>5} {'ev':>4} {'sprd':>4} | {'fills':>5} {'win%':>4} {'NO net c / fill':>22} {'NO return %':>22} | "
          f"{'fills':>5} {'win%':>4} {'MIRROR net c / fill':>22} {'ret %':>6} | {'NO minus mirror c':>22} | {'YES freq - mid c':>22}")


def text(label, o):
    if o is None:
        return None
    pc = lambda v: " n/a" if v is None else f"{100*v:4.0f}"
    num = lambda v: f"{v:+6.1f}" if v is not None else "   n/a"
    return (f"{label:40} {o['orders']:5} {o['markets']:5} {o['events']:4} {o['spread']:4.1f} | {o['fills']:5} {pc(o.get('won'))} {show(o.get('net'))} {show(o.get('ret'), 100)} | "
            f"{o['mirror_fills']:5} {pc(o.get('mirror_won'))} {show(o.get('mirror_net'))} {num(100*o['mirror_ret'][0] if o.get('mirror_ret') else None)} | "
            f"{show(o.get('diff'))} | {show(o['gap'])}")


def verdict(r, mask):
    """The pass rule. Returns (passed, reason, halves)."""
    o = line(r, mask, 1)
    hs = [line(r, h, 1) for h in halves(r, mask)]
    if o is None or "net" not in o or "mirror_net" not in o:
        return None, "too few fills", hs
    if any(h is None or "net" not in h or "mirror_net" not in h for h in hs):
        return None, "a half has under 10 fills on a side", hs
    why = []
    if not (o["net"][1] is not None and o["net"][1] > 0):
        why.append("interval includes zero")
    for k, h in enumerate(hs):
        if h["net"][0] <= 0:
            why.append(f"half {k+1} NO <= 0")
        if h["net"][0] <= h["mirror_net"][0]:
            why.append(f"half {k+1} NO <= mirror")
    return (not why), "; ".join(why) or "PASS", hs


# ---------------------------------------------------------------- trades (capacity)

TRADE_PER_FAMILY, TRADE_PAGES = 30, 3


def trade_sample():
    """Filled weekly orders (else first2c), one per market, round-robin over events, per family and period-blind."""
    r = build()[0]
    out = []
    for fam in FAMILIES:
        seen, cells = set(), defaultdict(list)
        for rule in ("weekly", "first2c"):
            for i in np.flatnonzero((r["family"] == fam) & (r["rule"] == rule) & r["filled"]):
                if r["ticker"][i] in seen:
                    continue
                if rule == "weekly" and pick_hash("week", r["ticker"][i], str(r["at"][i]))[0] > "7":   # one filled week in two, at random
                    continue
                seen.add(r["ticker"][i])
                cells[r["event"][i]].append((pick_hash("trade", r["ticker"][i]), str(r["ticker"][i]), int(r["at"][i]), rule, str(r["period"][i]),
                                             float(r["bid"][i]), float(r["ask"][i])))
        for v in cells.values():
            v.sort()
        got, depth = [], 0
        events = sorted(cells, key=lambda e: pick_hash("tevent", e))
        while len(got) < TRADE_PER_FAMILY and any(len(cells[e]) > depth for e in events):
            got += [cells[e][depth][1:] for e in events if len(cells[e]) > depth]
            depth += 1
        out += [(fam,)+x for x in got[:TRADE_PER_FAMILY]]
    return out


def trades_step(client):
    have = load(RAW/"trades.json", {})
    todo = [x for x in trade_sample() if f"{x[1]}|{x[2]}" not in have]
    todo.sort(key=lambda x: pick_hash("torder", x[1]))
    print(f"trades: {len(have)} orders on disk, {len(todo)} to fetch", flush=True)
    try:
        for n, (fam, ticker, at, rule, period, bid, ask) in enumerate(todo):
            rows, cursor, pages = [], "", 0
            window = dict(min_ts=at, **({"max_ts": at+WEEK} if rule == "weekly" else {}))
            while pages < TRADE_PAGES:
                data = read(client, "/markets/trades" if period == "recent" else "/historical/trades",
                            dict(ticker=ticker, limit=1000, **window, **({"cursor": cursor} if cursor else {}))) or {}
                pages += 1
                for t in data.get("trades") or []:
                    rows.append([int(hype_no.timestamp(t["created_time"])), round(100*float(t["yes_price_dollars"]), 2), float(t.get("count_fp", t.get("count", 0))),
                                 1 if (t.get("taker_side") or t.get("taker_outcome_side")) == "yes" else 0])
                cursor = data.get("cursor")
                if not cursor or not data.get("trades"):
                    break
            have[f"{ticker}|{at}"] = dict(family=fam, rule=rule, period=period, bid=bid, ask=ask, complete=not cursor, trades=rows)
            if n % 25 == 24:
                dump(RAW/"trades.json", have)
    finally:
        dump(RAW/"trades.json", have)
    print(f"trades for {len(have)} orders", flush=True)


def capacity(say):
    """Per family: contracts a YES taker bought strictly through our offer inside the resting window (and the mirror), from the trade record."""
    have = load(RAW/"trades.json", {})
    out = {}
    say("\n(5) CAPACITY FROM THE TRADE RECORD. A sample of FILLED orders (weekly rule where there is one, else first2c; one per market). 'through' = contracts"
        "\n    printed strictly through our price after placement inside the resting window, any taker side; 'YES-taker' = the part where the taker bought YES"
        "\n    (the flow a NO bid actually meets). tape fill% = share of the bar-rule fills the trade record confirms (a print strictly through, later than the bar close)."
        "\n    'cut' = orders whose record ran past 3,000 trades (their size is a lower bound). GET /markets/trades returned nothing for most orders placed more than"
        "\n    about two months ago in recent-tier markets, so a low tape fill% in the long-lived families is missing data, not a refuted fill; sizes use orders with prints.")
    say(f"    {'family':36} {'orders':>6} {'tape fill%':>10} {'through: 25%':>12} {'median':>7} {'75%':>7} {'mean':>8} {'YES-taker median':>16} {'share YES-taker':>15} {'>=100':>6} {'>=1000':>6} {'cut':>4}")
    for fam in FAMILIES+("ALL",):
        rows = [v for v in have.values() if fam in (v["family"], "ALL")]
        if len(rows) < 5:
            continue
        thr, ythr, ok = [], [], []
        for v in rows:
            offer = v["ask"]-1
            thr.append(sum(t[2] for t in v["trades"] if t[1] > offer))
            ythr.append(sum(t[2] for t in v["trades"] if t[1] > offer and t[3] == 1))
            ok.append(thr[-1] > 0)
        thr, ythr = np.array(thr), np.array(ythr)
        pos = thr[thr > 0] if (thr > 0).any() else np.array([0.0])
        q = np.percentile(pos, [25, 50, 75])
        out[fam] = dict(orders=len(rows), tape_fill=float(np.mean(ok)), q25=float(q[0]), median=float(q[1]), q75=float(q[2]), mean=float(pos.mean()),
                        yes_median=float(np.median(ythr[thr > 0])) if (thr > 0).any() else 0.0, yes_share=float(ythr.sum()/max(1, thr.sum())),
                        over100=float((thr >= 100).mean()), over1000=float((thr >= 1000).mean()), cut=int(sum(not v["complete"] for v in rows)))
        o = out[fam]
        say(f"    {fam:36} {o['orders']:6} {100*o['tape_fill']:9.0f}% {o['q25']:12.0f} {o['median']:7.0f} {o['q75']:7.0f} {o['mean']:8.0f} {o['yes_median']:16.0f} "
            f"{100*o['yes_share']:14.0f}% {100*o['over100']:5.0f}% {100*o['over1000']:5.0f}% {o['cut']:4}")
    return out


# ---------------------------------------------------------------- report

BANDS = (("YES under 20c", 0, 20), ("YES 20-80c", 20, 80), ("YES over 80c", 80, 101))
WIDTHS = (("spread 2-4c", 2, 5), ("spread 5-9c", 5, 10), ("spread 10c+", 10, 101))
SIZES = (("1 market in event", 1, 2), ("2 markets in event", 2, 3), ("3-9 markets", 3, 10), ("10+ markets", 10, 10**6))


def report():
    OUT.mkdir(parents=True, exist_ok=True)
    lines, summary = [], {}

    def say(t=""):
        if t is not None:
            print(t, flush=True)
            lines.append(t)
    r, ms, cover = build()
    say("(1) COVERAGE. Markets with lifetime volume >= 500 contracts. recent = closing 2026-09-18 to 2026-11-18 (whole events, capped per family);"
        "\n    archive = closing 2026-01-01 to 2026-09-17 (at most 2 markets per event). Orders per rule = markets (weekly: market-weeks) with a two-sided book >= 2c.")
    say(f"    {'family':36} {'period':8} {'picked':>6} {'bars':>5} {'empty':>5} {'first2c':>7} {'open+24h':>8} {'weekly':>7}")
    for fam in FAMILIES:
        for period in ("recent", "archive"):
            c = cover.get(f"{period} | {fam}")
            if c:
                say(f"    {fam:36} {period:8} {c.get('picked', 0):6} {c.get('with bars', 0):5} {c.get('empty', 0):5} {c.get('first2c', 0):7} {c.get('open+24h', 0):8} {c.get('weekly', 0):7}")
    summary["coverage"] = cover
    say("\n    HOW THE QUOTE WAS INFERRED. The best YES bid and ask at the order are the bid/ask CLOSE of the candlestick ending at the placement instant (hourly bars in a"
        "\n    market's first week, daily bars ending 05:00 UTC after it). A fill is a LATER bar whose trade high is strictly above our YES offer (NO side) or whose trade"
        "\n    low is strictly below our YES bid (mirror). Biases: (a) a bar close is one snapshot, so a quote that flickered wide for a moment counts as a 2c+ book;"
        "\n    (b) with daily bars the quote is up to a day coarse and a fill is any print through the price in a later daily bar: nothing says the book was still"
        "\n    where the close put it when a live order would have gone in; (c) the bars do not say how many contracts printed through the price,"
        "\n    which is why capacity comes from the trade record of a sample; (d) our own order is assumed not to move anyone: a competitor may step in front of a 1c"
        "\n    improvement, so real fills skew to the adverse ones; (e) markets under 500 lifetime contracts are left out, a filter known only afterwards.")
    share = float(r["daily"][r["rule"] == "weekly"].mean()) if (r["rule"] == "weekly").any() else 0
    say(f"    Share of weekly orders quoted from a daily bar: {100*share:.0f}%.")

    passes = {}
    for rule in RULES:
        rr = r["rule"] == rule
        say(f"\n(2) RULE {rule}" + ("   <- headline (the mention-market rule)" if rule == HEAD else "") + ". Intervals: 95%, the wider of event- and day-clustered bootstraps."
            "\n    win% = share of fills that won. 'YES freq - mid' = calibration over all orders: negative means YES was overpriced, whatever the fill model.")
        say("  "+HEADER)
        for fam in FAMILIES+("ALL FAMILIES",):
            base = rr & ((r["family"] == fam) if fam != "ALL FAMILIES" else True)
            o = line(r, base)
            if o is None:
                say(f"  {fam:40} {int(base.sum()):5}   under 20 orders")
                continue
            say("  "+text(fam, o))
            ok, why, hs = verdict(r, base)
            for k, h in enumerate(hs):
                if h is not None:
                    say("  "+text(f"    half {k+1}", h))
            for label, pick in (("recent", r["period"] == "recent"), ("archive", r["period"] == "archive")):
                say(text("  "+f"    {label}", line(r, base & pick)) and "  "+text(f"    {label}", line(r, base & pick)))
            say(f"      -> {'PASS' if ok else 'not tested' if ok is None else 'FAIL'}: {why}")
            passes[f"{rule}|{fam}"] = dict(passed=ok, why=why)
            summary.setdefault("families", {})[f"{rule}|{fam}"] = dict(all=o, halves=hs, verdict=ok, why=why)
    tested = [k for k, v in passes.items() if v["passed"] is not None and "ALL" not in k]
    passed = [k for k in tested if passes[k]["passed"]]
    say(f"\n    MULTIPLICITY. {len(tested)} family x rule cells had enough fills to test; {len(passed)} pass: {', '.join(passed) or 'none'}."
        "\n    A cell with no edge passes about 1 time in 40 on the interval alone (one-sided 2.5%), less with the half conditions; the cells of one family across"
        "\n    rules are not independent.")
    summary["passes"] = passes

    say("\n(3) BY YES PRICE, SPREAD AND EVENT SIZE AT PLACEMENT")
    for rule in ("weekly", "first2c"):
        rr = r["rule"] == rule
        say(f"\n  RULE {rule}")
        say("  "+HEADER)
        groups = (("ALL", np.ones(len(rr), bool)), ("Entertainment", np.char.startswith(r["family"], "Ent")), ("Politics", np.char.startswith(r["family"], "Pol")),
                  ("Elections", np.char.startswith(r["family"], "Elec")), ("Tech+Sci+Co+World", ~np.char.startswith(r["family"], "Ent") & ~np.char.startswith(r["family"], "Pol")
                                                                           & ~np.char.startswith(r["family"], "Elec")))
        for gname, gm in groups:
            for label, lo, hi in BANDS:
                o = line(r, rr & gm & (r["mid"] >= lo) & (r["mid"] < hi))
                say(text("", o) and "  "+text(f"{gname}, {label}", o))
                summary.setdefault("bands", {})[f"{rule}|{gname}|{label}"] = o
            for label, lo, hi in WIDTHS:
                o = line(r, rr & gm & (r["spread"] >= lo) & (r["spread"] < hi))
                say(text("", o) and "  "+text(f"{gname}, {label}", o))
                summary.setdefault("bands", {})[f"{rule}|{gname}|{label}"] = o
            for label, lo, hi in SIZES:
                o = line(r, rr & gm & (r["eventsize"] >= lo) & (r["eventsize"] < hi))
                say(text("", o) and "  "+text(f"{gname}, {label}", o))
                summary.setdefault("bands", {})[f"{rule}|{gname}|{label}"] = o
    say("\n  BY FAMILY AND YES PRICE BAND, rule weekly then first2c (families with 20+ orders in the band)")
    say("  "+HEADER)
    for rule in ("weekly", "first2c"):
        for fam in FAMILIES:
            for label, lo, hi in BANDS:
                o = line(r, (r["rule"] == rule) & (r["family"] == fam) & (r["mid"] >= lo) & (r["mid"] < hi))
                if o is not None and o["fills"] >= 10:
                    say("  "+text(f"{rule[:6]} {fam[:20]}, {label[4:]}", o))
                    summary.setdefault("family_bands", {})[f"{rule}|{fam}|{label}"] = o

    say("\n(4) SENSITIVITY: fills at the price as well as through it (rule weekly, then first2c): fills, NO net c / mirror net c")
    for rule in ("weekly", "first2c"):
        cells = []
        for fam in FAMILIES:
            base = (r["rule"] == rule) & (r["family"] == fam)
            f, yf = base & r["touched"], base & r["ytouched"]
            if f.sum() >= 10 and yf.sum() >= 10:
                cells.append(f"{fam[:22]} {int(f.sum())}: {r['net'][f].mean():+.1f}/{r['ynet'][yf].mean():+.1f}")
        say(f"  {rule}: "+" | ".join(cells))

    cap = capacity(say)
    summary["capacity"] = cap

    say("\n(6) DOLLARS PER MONTH at 25% of the size printing through, rule weekly, RECENT period (62 days, every picked market: whole events up to the family cap)."
        "\n    fills/month = NO fills in the recent period x 30/62. contracts = 25% of the median contracts through the price per filled order (trade-record sample)."
        "\n    $ = fills x contracts x net c / 100, at the family's net per fill over BOTH periods, and at the lower end of its interval. Stake = contracts x price paid,"
        "\n    summed over a month of fills (money committed, not all at once). Families ranked by $ per month. A family that FAILS has no claim to its number.")
    say(f"    {'family':36} {'verdict':>10} {'fills/mo':>8} {'x scale':>7} {'median through':>14} {'contracts':>9} {'net c':>6} {'ret %':>6} {'$ / month':>10} {'at low end':>10} {'stake $/mo':>10}")
    rank = []
    for fam in FAMILIES:
        base = (r["rule"] == "weekly") & (r["family"] == fam)
        o = line(r, base)
        rec = base & (r["period"] == "recent") & r["filled"]
        c = cap.get(fam)
        if o is None or "net" not in o or c is None:
            continue
        picked = cover.get(f"recent | {fam}", {}).get("picked", 0)
        listed_n = load(RAW/"listed-counts.json", {}).get(fam, picked)
        scale = listed_n/picked if picked else 1.0
        fills = rec.sum()*30/62*scale
        size = 0.25*c["median"]
        dollars = fills*size*o["net"][0]/100
        low = fills*size*o["net"][1]/100 if o["net"][1] is not None else float("nan")
        rank.append((dollars, fam, passes.get(f"weekly|{fam}", {}).get("passed"), fills, scale, c["median"], size, o["net"][0], 100*o["ret"][0], low, fills*size*o["paid"]/100))
    for dollars, fam, ok, fills, scale, med, size, net, ret, low, stake in sorted(rank, reverse=True):
        say(f"    {fam:36} {'PASS' if ok else 'untested' if ok is None else 'FAIL':>10} {fills:8.0f} {scale:7.2f} {med:14.0f} {size:9.0f} {net:+6.1f} {ret:+6.1f} {dollars:+10.0f} {low:+10.0f} {stake:10.0f}")
    summary["rank"] = [dict(family=x[1], dollars=float(x[0]), verdict=x[2], fills_per_month=float(x[3]), median_through=float(x[5]), net=float(x[7]), ret=float(x[8]),
                            low=float(x[9]), stake=float(x[10])) for x in sorted(rank, reverse=True)]
    dump(OUT/"summary.json", summary)
    (OUT/"report.txt").write_text("\n".join(lines)+"\n")


# AUDIT: the original main() and download lock are removed; see the audit section below.


# ================================================================ AUDIT SECTION (new code)
# Everything above is the original method (order_at, orders, build, halves, line, verdict) apart from the marked lines.

AUD = ROOT / "data/processed/audit/resting_no_wide"


def reseed(seed=31):
    resting_no.RNG = np.random.default_rng(seed)


def both(r, parts, draws=4000):
    """Event- and day-clustered intervals separately: (est, (elo, ehi), (dlo, dhi))."""
    a, b = resting_no.boot(parts, r["g"], draws), resting_no.boot(parts, r["d"], draws)
    if a is None:
        return None
    return a[0], (a[1], a[2]), (b[1], b[2])


def lean(r, mask, draws=1000):
    """The pass rule of verdict() with only the pieces it uses: whole-sample NO net interval (wider of event/day), half means of NO and mirror.
    Returns (passed or None, dict of the three conditions)."""
    one = np.ones(len(mask))
    f, yf = mask & r["filled"], mask & r["yfilled"]
    if f.sum() < 10 or yf.sum() < 10:
        return None, {}
    a, b = resting_no.boot([(r["net"], one, f)], r["g"], draws), resting_no.boot([(r["net"], one, f)], r["d"], draws)
    if a[1] is None or b[1] is None:
        lo = None
    else:
        lo = (a if a[2]-a[1] >= b[2]-b[1] else b)[1]
    c = dict(interval=lo is not None and lo > 0, halves_pos=True, halves_gt_mirror=True)
    for h in halves(r, mask):
        hf, hy = h & r["filled"], h & r["yfilled"]
        if hf.sum() < 10 or hy.sum() < 10:
            return None, {}
        if r["net"][hf].mean() <= 0:
            c["halves_pos"] = False
        if r["net"][hf].mean() <= r["ynet"][hy].mean():
            c["halves_gt_mirror"] = False
    return all(c.values()), c


def with_outcomes(r, y):
    q = dict(r)
    q["y"] = y
    q["net"] = 100*(~y)-r["price"]-r["fee"]
    q["ynet"] = 100*y-r["yprice"]-r["yfee"]
    q["gap"] = 100*y-r["mid"]
    return q


def fmt(v, scale=1.0):
    return "   n/a" if v is None else f"{scale*v:+6.1f}"


def recheck(say):
    """Every family x rule with ONE seeded bootstrap (4000 draws) used for both the printed interval and the verdict."""
    r, ms, cover = build()
    orig = load(ROOT/"data/processed/resting-no-wide/summary.json", {}).get("passes", {})
    one = np.ones(len(r["y"]))
    say("(A) RECHECK of the original pass rule with a seeded bootstrap, 4000 draws. NO net c/fill with the event-clustered and the day-clustered 95% interval;"
        "\n    return % of stake with the wider interval; halves; mirror. 'orig' = verdict in the original summary.json (its verdict interval came from a second,"
        "\n    different random draw than the one it printed). 'flip' marks a disagreement.")
    say(f"    {'rule|family':46} {'ev':>4} {'days':>4} {'fills':>5} {'net c':>6} {'event CI':>15} {'day CI':>15} {'ret %':>6} {'ret CI (wider)':>15} {'h1 net/mir':>12} {'h2 net/mir':>12} {'mirror c':>8} seeded orig")
    out = {}
    for rule in RULES:
        for fam in FAMILIES+("ALL FAMILIES",):
            mask = (r["rule"] == rule) & ((r["family"] == fam) if fam != "ALL FAMILIES" else True)
            f, yf = mask & r["filled"], mask & r["yfilled"]
            if f.sum() < 10 or yf.sum() < 10:
                continue
            reseed(1000+len(out))
            net = both(r, [(r["net"], one, f)])
            ret = both(r, [(r["net"], r["price"], f)])
            pick = lambda v: v[1] if (v[1][0] is None or v[2][0] is None or v[1][1]-v[1][0] >= v[2][1]-v[2][0]) else v[2]
            lo = pick(net)[0]
            hs = []
            okh = True
            small = False
            for h in halves(r, mask):
                hf, hy = h & r["filled"], h & r["yfilled"]
                if hf.sum() < 10 or hy.sum() < 10:
                    small = True
                    hs.append((None, None))
                    continue
                a, b = float(r["net"][hf].mean()), float(r["ynet"][hy].mean())
                hs.append((a, b))
                okh &= a > 0 and a > b
            ok = None if small else bool(lo is not None and lo > 0 and okh)
            o = orig.get(f"{rule}|{fam}", {}).get("passed", "n/a")
            w = pick(ret)
            name = lambda v: "PASS" if v is True else "FAIL" if v is False else "untested" if v is None else str(v)
            say(f"    {rule+'|'+fam:46} {len(set(r['g'][f])):4} {len(set(r['d'][f])):4} {int(f.sum()):5} {net[0]:+6.1f} [{fmt(net[1][0])},{fmt(net[1][1])}] [{fmt(net[2][0])},{fmt(net[2][1])}] "
                f"{100*ret[0]:+6.1f} [{fmt(w[0], 100)},{fmt(w[1], 100)}] {fmt(hs[0][0])}/{fmt(hs[0][1])} {fmt(hs[1][0])}/{fmt(hs[1][1])} {float(r['ynet'][yf].mean()):+8.1f} "
                f"{name(ok):8} {name(o):8} {'flip' if ok != o else ''}")
            out[f"{rule}|{fam}"] = dict(fills=int(f.sum()), events=len(set(r["g"][f])), net=net, ret=ret, ret_wider=w, halves=hs, seeded=ok, original=o)
    return r, out


def stability(r, say, cells, runs=40):
    say("\n(B) STABILITY of the original verdict() across random draws (it uses 1000 draws and an unseeded shared generator): share of 40 reruns that PASS.")
    for key in cells:
        rule, fam = key.split("|")
        mask = (r["rule"] == rule) & ((r["family"] == fam) if fam != "ALL FAMILIES" else True)
        got = []
        for k in range(runs):
            reseed(k)
            got.append(lean(r, mask)[0])
        say(f"    {key:46} pass in {sum(g is True for g in got):2}/{runs}")


def by_series(r, say, least=60):
    say("\n(C) INSIDE THE FAMILIES: the same rule per SERIES (60+ fills, 8+ events). Looked at after the fact; many cells, so a lone pass means little.")
    say(f"    {'rule':8} {'family':34} {'series':28} {'ev':>4} {'fills':>5} {'paid':>5} {'net c':>6} {'wider CI':>15} {'ret %':>6} {'mirror':>6}  seeded verdict")
    one = np.ones(len(r["y"]))
    n = 0
    for rule in ("first2c", "weekly"):
        for fam in FAMILIES:
            base = (r["rule"] == rule) & (r["family"] == fam)
            for series in sorted(set(r["series"][base])):
                mask = base & (r["series"] == series)
                f, yf = mask & r["filled"], mask & r["yfilled"]
                if f.sum() < least or len(set(r["g"][f])) < 8 or yf.sum() < 10:
                    continue
                n += 1
                reseed(5000+n)
                ok, c = lean(r, mask, 4000)
                reseed(5000+n)
                net = wide(r, [(r["net"], one, f)])
                say(f"    {rule:8} {fam[:34]:34} {series[:28]:28} {len(set(r['g'][f])):4} {int(f.sum()):5} {r['price'][f].mean():5.1f} {net[0]:+6.1f} [{fmt(net[1])},{fmt(net[2])}] "
                    f"{100*r['net'][f].sum()/r['price'][f].sum():+6.1f} {r['ynet'][yf].mean():+6.1f}  {'PASS' if ok else 'untested' if ok is None else 'FAIL'}")
    say(f"    {n} series x rule cells.")


def mde(r, say, res):
    say("\n(D) MINIMUM DETECTABLE EDGE, return % of stake per fill, from the wider (event/day) interval of the realised sample. half-width = edge whose"
        "\n    interval would just touch zero (about 50% power on the interval alone); 80% = 2.8 standard errors. 'x fills for 3%' = how many times the present"
        "\n    number of fills AND independent events/days an 80%-power test of +3% would need (same mix).")
    say(f"    {'rule|family':46} {'events':>6} {'fills':>6} {'ret %':>6} {'half-width %':>12} {'MDE 80% %':>10} {'x fills for 3%':>14} {'x for 1.5%':>10}")
    out = {}
    for key, v in res.items():
        w = v["ret_wider"]
        if w[0] is None:
            continue
        half = 100*(w[1]-w[0])/2
        m80 = half*2.8/1.96
        say(f"    {key:46} {v['events']:6} {v['fills']:6} {100*v['ret'][0]:+6.1f} {half:12.1f} {m80:10.1f} {(m80/3)**2:14.1f} {(m80/1.5)**2:10.1f}")
        out[key] = dict(half=half, mde80=m80)
    return out


def planted(r, say, fam, edges=(0.03, 0.015, 0.06, 0.10, 0.20), sims=200, rules=("first2c", "weekly")):
    """Replace the family's outcomes so that every filled NO order has a true expected return of `edge` of its stake, then apply the pass rule.
    Per market: P(NO) = (1+edge) x (mean NO price of that market's filled orders under the rule) / 100, capped at 1. Outcomes are drawn independently per market
    ('indep') or with one uniform draw per event ('event': all markets of an event move together, the most clustered case)."""
    out = {}
    gen = np.random.default_rng(2026)
    for rule in rules:
        mask = (r["rule"] == rule) & (r["family"] == fam)
        idx = np.flatnonzero(mask)
        sub = {k: v[idx] for k, v in r.items()}
        m = np.ones(len(idx), bool)
        tick, tinv = np.unique(sub["ticker"], return_inverse=True)
        f = sub["filled"]
        cost = sub["price"]+sub["fee"]
        csum, cn = np.bincount(tinv, cost*f, len(tick)), np.bincount(tinv, f, len(tick))
        asum, an = np.bincount(tinv, cost, len(tick)), np.bincount(tinv, minlength=len(tick))
        cbar = np.where(cn > 0, csum/np.maximum(cn, 1), asum/an)
        ev_of = np.zeros(len(tick), int)
        ev_of[tinv] = np.unique(sub["g"], return_inverse=True)[1]
        for edge in edges:
            p = np.minimum(1.0, (1+edge)*cbar/100)
            true = float(((100*p[tinv]-cost)*f).sum()/(sub["price"]*f).sum())
            for mode in ("indep", "event"):
                tally, rets = Counter(), []
                for k in range(sims):
                    u = gen.random(len(tick)) if mode == "indep" else gen.random(ev_of.max()+1)[ev_of]
                    no = u < p
                    q = with_outcomes(sub, ~no[tinv])
                    reseed(k)
                    ok, c = lean(q, m)
                    tally["pass"] += ok is True
                    tally["untested"] += ok is None
                    for name, val in c.items():
                        tally[name] += val
                    rets.append(100*q["net"][f].sum()/q["price"][f].sum())
                out[f"{rule}|{edge}|{mode}"] = dict(true=true, **{k: v/sims for k, v in tally.items()})
                say(f"    {rule:8} planted {100*edge:4.1f}% (true after the cap {100*true:+5.2f}%) {mode:6} | realised return mean {np.mean(rets):+6.1f}% sd {np.std(rets):5.1f}% | "
                    f"PASS {100*tally['pass']/sims:5.1f}% | interval excludes 0: {100*tally['interval']/sims:5.1f}%  both halves > 0: {100*tally['halves_pos']/sims:5.1f}%  "
                    f"both halves > mirror: {100*tally['halves_gt_mirror']/sims:5.1f}%  untested {100*tally['untested']/sims:4.1f}%")
    return out


def shifted(r, say, fams, edges=(0.015, 0.03, 0.06, 0.10, 0.20), sims=200, rules=("first2c", "weekly")):
    """Second planting, which leaves the real quotes, fills, clustering and mirror relation alone: resample the family's EVENTS with replacement, then flip just enough
    randomly chosen outcomes (YES -> NO, or NO -> YES when the real return is above the target) for the expected stake-weighted return of the filled NO orders to be
    exactly `edge`. A flip applies to a market, so the mirror loses what the NO side gains."""
    out = {}
    gen = np.random.default_rng(606)
    for fam in fams:
        for rule in rules:
            mask = (r["rule"] == rule) & (r["family"] == fam)
            idx = np.flatnonzero(mask)
            if not len(idx):
                continue
            sub = {k: v[idx] for k, v in r.items()}
            f = sub["filled"]
            ev, einv = np.unique(sub["g"], return_inverse=True)
            members = [np.flatnonzero(einv == k) for k in range(len(ev))]
            s_net, s_price = sub["net"][f].sum(), sub["price"][f].sum()
            for edge in edges:
                need = edge*s_price-s_net                       # cents to add to the filled NO orders
                pool = f & (sub["y"] if need > 0 else ~sub["y"])
                q = min(1.0, abs(need)/(100*max(1, pool.sum())))
                tally, rets = Counter(), []
                for k in range(sims):
                    pick = gen.integers(0, len(ev), len(ev))
                    rows = np.concatenate([members[j] for j in pick])
                    g = np.concatenate([np.full(len(members[j]), n) for n, j in enumerate(pick)])
                    q_ = {key: v[rows] for key, v in sub.items()}
                    q_["g"] = g
                    _, tk = np.unique(np.char.add(np.char.add(g.astype(str), "|"), q_["ticker"]), return_inverse=True)
                    flip = (gen.random(tk.max()+1) < q)[tk]
                    y = q_["y"].copy()
                    if need > 0:
                        y[flip & y] = False
                    else:
                        y[flip & ~y] = True
                    q_ = with_outcomes(q_, y)
                    reseed(k)
                    ok, c = lean(q_, np.ones(len(rows), bool))
                    tally["pass"] += ok is True
                    tally["untested"] += ok is None
                    for name, val in c.items():
                        tally[name] += val
                    ff = q_["filled"]
                    rets.append(100*q_["net"][ff].sum()/q_["price"][ff].sum())
                out[f"{fam}|{rule}|{edge}"] = dict(mean=float(np.mean(rets)), sd=float(np.std(rets)), **{k: v/sims for k, v in tally.items()})
                say(f"    {fam[:32]:32} {rule:8} planted {100*edge:4.1f}% | realised return mean {np.mean(rets):+6.1f}% sd {np.std(rets):5.1f}% | PASS {100*tally['pass']/sims:5.1f}% | "
                    f"interval excludes 0: {100*tally['interval']/sims:5.1f}%  halves > 0: {100*tally['halves_pos']/sims:5.1f}%  halves > mirror: {100*tally['halves_gt_mirror']/sims:5.1f}%  "
                    f"untested {100*tally['untested']/sims:4.1f}%")
    return out


def mentions(say):
    """The wide method (first2c / open+24h / weekly, same fills, same verdict) on the mention markets of research/resting_no.py."""
    ms, facts = resting_no.universe()
    rows = []
    for m in ms:
        if m.get("result") not in ("yes", "no") or not m.get("open") or not m.get("close") or m["close"] <= m["open"]:
            continue
        rows.append(dict(m, family="Mentions "+m["family"], period="recent" if m["period"] == "recent" else "archive", eventsize=facts[m["event"]]["n"],
                         volume=m.get("volume") or 0))
    bars = {m["ticker"]: m.pop("bars") for m in rows}
    out = {}
    for label, pool in (("every mention market with bars", rows), ("lifetime volume >= 500 (the wide study's filter)", [m for m in rows if m["volume"] >= MIN_VOLUME])):
        r, _, _ = build(pool, lambda m: ([b for b in bars[m["ticker"]] if b[0] <= m["close"]+7200], None))
        one = np.ones(len(r["y"]))
        say(f"\n  {label}: {len(pool)} markets")
        say(f"    {'rule|family':46} {'ev':>4} {'fills':>5} {'paid':>5} {'net c':>6} {'wider CI':>15} {'ret %':>6} {'ret CI':>15} {'h1 net/mir':>12} {'h2 net/mir':>12} verdict (original verdict(), seeded)")
        fams = sorted(set(r["family"]))
        for rule in RULES:
            for fam in fams+["ALL MENTIONS", "ALL MENTIONS ex sports broadcasts"]:
                mask = (r["rule"] == rule) & ((r["family"] == fam) if not fam.startswith("ALL") else (r["family"] != "Mentions sports broadcasts") if "ex" in fam else True)
                f = mask & r["filled"]
                if f.sum() < 10:
                    continue
                reseed(77)
                ok, why, hs = verdict(r, mask)
                reseed(77)
                net, ret = wide(r, [(r["net"], one, f)]), wide(r, [(r["net"], r["price"], f)])
                h = lambda x: f"{fmt(x['net'][0])}/{fmt(x['mirror_net'][0])}" if x and "net" in x and "mirror_net" in x else "     n/a    "
                say(f"    {rule+'|'+fam:46} {len(set(r['g'][f])):4} {int(f.sum()):5} {r['price'][f].mean():5.1f} {net[0]:+6.1f} [{fmt(net[1])},{fmt(net[2])}] {100*ret[0]:+6.1f} "
                    f"[{fmt(ret[1], 100)},{fmt(ret[2], 100)}] {h(hs[0])} {h(hs[1])} {'PASS' if ok else 'untested' if ok is None else 'FAIL'}: {why}")
                out[f"{label}|{rule}|{fam}"] = dict(fills=int(f.sum()), net=net, ret=ret, verdict=ok, why=why)
    return out


def main():
    parser = argparse.ArgumentParser(description="audit power checks of research/resting_no_wide.py (offline)")
    parser.add_argument("steps", nargs="+", choices=("recheck", "planted", "shifted", "mentions"))
    parser.add_argument("--family", default="Pol recurring (approval, counts)")
    parser.add_argument("--sims", type=int, default=200)
    args = parser.parse_args()
    AUD.mkdir(parents=True, exist_ok=True)
    for step in args.steps:
        lines = []

        def say(t=""):
            print(t, flush=True)
            lines.append(t)
        if step == "recheck":
            r, res = recheck(say)
            stability(r, say, [k for k, v in res.items() if v["seeded"] != v["original"] or v["seeded"]]+["first2c|Elec combos and margins", "weekly|Elec combos and margins"])
            by_series(r, say)
            m = mde(r, say, res)
            dump(AUD/"recheck.json", dict(cells=res, mde=m))
        elif step == "planted":
            r = build()[0]
            say(f"(E) PLANTED EDGE in '{args.family}', {args.sims} simulated outcome sets per line; pass rule = the original verdict() conditions.")
            dump(AUD/f"planted-{args.family.split()[0]}-{args.family.split()[1]}.json", planted(r, say, args.family, sims=args.sims))
        elif step == "shifted":
            r = build()[0]
            big = ("Elec Senate House Governor", "Elec combos and margins", "Elec other (mayors, local)", "Elec international", "Ent Rotten Tomatoes")
            say(f"(E2) PLANTED EDGE BY SHIFTING REAL OUTCOMES, {args.sims} event-resampled sets per line. First the mid-sized family, then the five families with the most"
                "\n     contracts traded in the recent sample.")
            dump(AUD/"shifted.json", shifted(r, say, (args.family,)+big, sims=args.sims))
        else:
            say("(F) KNOWN EDGE: the wide study's rules, fills and pass rule applied to the mention markets of research/resting_no.py (cached bars only).")
            dump(AUD/"mentions.json", mentions(say))
        tag = step if step != "planted" else "planted-"+"-".join(args.family.split()[:2])
        (AUD/f"{tag}.txt").write_text("\n".join(lines)+"\n")


if __name__ == "__main__":
    main()
