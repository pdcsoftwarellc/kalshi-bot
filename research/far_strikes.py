"""Far strikes in daily ladders: is the near-certain side underpriced, and what does the bad day cost?

Research only. Public Kalshi reads, no orders, no authentication. Background:
research/calibration_map.py found that contracts at about 96c in daily gas and
temperature ladders won more often than their price on the recent tier (since
2026-09-18). Those are mostly the NO side of far strikes, and 56-61 days with
zero or one loss cannot say what a spike day costs. This script extends the
history through the archive, measures the tail, and checks whether the 96c
contracts can actually be bought.

Fixed in advance (before any result was read)

* Reference times, one hourly bar close each:
    gas / diesel / state gas / weekly gas / rain / rates and FX / other daily:
        the last hourly bar ending at or before 6 hours before the scheduled close
        (national gas: 21:00 UTC, about 5pm Eastern the evening before the AAA print);
    daily high temperature: 07:00 and 13:00 local clock time on the day
        (the close moved from 11:59pm to 1am local during the sample, so
        "hours before close" is not a fixed time of day);
    daily low temperature: 23:00 local the evening before and 05:00 local.
* The quote at T is the close of the last hourly bar ending at or before T
  (bars are only emitted for hours in which something changed), provided the
  market had opened. A row is the dearer side of one market: YES if the
  midpoint is above 50c, else NO. Needs a two-sided quote, spread <= 4c,
  midpoint below 99c. Bands 80-90, 90-95, 95-99 on the midpoint.
* Tradeable versions. Taker: pay the ask (YES ask, or 100 - YES bid for NO)
  plus 0.07*p*(1-p), ask <= 99c. Maker: rest one cent above the best bid of
  that side (never at or through the ask), counted as filled only if a later
  hourly bar prints strictly below it (for NO: a YES print strictly above
  100 - bid), no fee (every series here has fee_type "quadratic"), held to
  settlement.
* Intervals: bootstrap over calendar days and over ISO weeks (all strikes,
  cities and series of one day move together).

Archive coverage is bought with one request per market, so it is thinned by
rules that use only information public before the reference time:
* national gas: whole-cent strikes only (the ladder steps by half a cent). A
  ring of strikes around the previous day's print was tried first and thinned
  nothing: the 80-99c contracts sit anywhere from 0 to 5c from yesterday's
  price because the price trends. Archived diesel (44 days) was not read;
* temperature highs: one of the seven original cities per calendar day in
  rotation (day ordinal mod 7), all six strikes; plus New York at 13:00 on
  every day from the minute bars research/weather_state_download.py left on disk.

Commands (run in order; every one is resumable and respects --minutes)

    list      listings and fee types
    recent    hourly bars for the recent tier, batch endpoint
    archive   hourly bars for the thinned archive, one market per request
    eia       weekly US retail gasoline prices from the EIA (public, no key)
    report    everything else, offline -> data/processed/far-strikes/

Network: one request at a time, at most one per PACE seconds, long back-off on
HTTP 429 (a live trader shares the limit). Raw compact caches live under
data/raw/far-strikes/; a request log is kept in download-log.json.
"""
import argparse
import hashlib
from collections import Counter, defaultdict
from datetime import date, datetime, timedelta, timezone
import json
import math
from pathlib import Path
import re
import sys
import time
from zoneinfo import ZoneInfo

import httpx
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import config
from research.api_cost import ROOT, midnight, timestamp

RAW = ROOT / "data/raw/far-strikes"
OUT = ROOT / "data/processed/far-strikes"
CUTOFF = midnight(date(2026, 9, 18))
RECENT = dict(status="settled", limit=1000, min_close_ts=CUTOFF-86400, max_close_ts=midnight(date(2026, 11, 19)))
PACE = 1.5
UTC = timezone.utc
MONTHS = {m: i+1 for i, m in enumerate("JAN FEB MAR APR MAY JUN JUL AUG SEP OCT NOV DEC".split())}
BANDS = ((80, 90), (90, 95), (95, 99), (90, 99))
RNG = np.random.default_rng(7)

EAST, CENT, MOUNT, PAC = "America/New_York", "America/Chicago", "America/Denver", "America/Los_Angeles"
ZONES = {"NY": EAST, "NYC": EAST, "PHIL": EAST, "MIA": EAST, "ATL": EAST, "BOS": EAST, "DC": EAST, "CHI": CENT, "AUS": CENT,
         "DAL": CENT, "HOU": CENT, "MIN": CENT, "OKC": CENT, "NOLA": CENT, "SATX": CENT, "DEN": MOUNT, "PHX": "America/Phoenix",
         "LAX": PAC, "SFO": PAC, "SEA": PAC, "LV": PAC}
ORIGINAL = ["KXHIGHNY", "KXHIGHCHI", "KXHIGHMIA", "KXHIGHAUS", "KXHIGHLAX", "KXHIGHDEN", "KXHIGHPHIL"]
HIGHS = ORIGINAL + ["KXHIGHT"+c for c in "SFO PHX ATL SEA BOS DAL DC LV HOU MIN OKC NOLA SATX".split()]
LOWS = ["KXLOWT"+c for c in "NYC CHI MIA AUS DEN LAX PHIL PHX OKC ATL DAL MIN DC HOU SEA NOLA LV SATX SFO BOS".split()]
STATES = ["KXAAAGASD"+s for s in "FL OH IL NJ".split()]
FAMILIES = {
    # name: (series, reference times, archive?)
    "Gas+diesel daily (US)": dict(series=["KXAAAGASD", "KXDIESELD"], refs=("6h",), explore_refs=("12h", "9h", "3h", "1h"), archive=True),
    "Temp high daily": dict(series=HIGHS, refs=("07:00", "13:00"), implied_refs=("20:00-1",), archive=True),
    "Temp low daily": dict(series=LOWS, refs=("23:00-1", "05:00")),
    "Gas daily (4 states)": dict(series=STATES, refs=("6h",)),
    "Gas+diesel weekly": dict(series=["KXAAAGASW", "KXDIESELW"], refs=("6h",), archive=True),
    "Rain daily": dict(series=["KXRAINNYC", "KXRAINSEA"], refs=("6h",)),
    "Rates+FX daily": dict(series=["KXTNOTED", "KXUSDJPY", "KXEURUSD", "KXSOFRD"], refs=("6h",)),
    "Other daily": dict(series=["KXTXERCOTPEAKD", "KXTRUEV"], refs=("6h",)),
    "Other weekly": dict(series=["KXTNOTEW", "KXTSAW", "KX30YMORTW", "KXJOBLESSCLAIMS"], refs=("6h",)),
}
SERIES_FAMILY = {s: name for name, spec in FAMILIES.items() for s in spec["series"]}

log = Counter()
last_request = [0.0]
deadline = [float("inf")]


# ---------------------------------------------------------------- network

class OutOfTime(Exception):
    pass


def read(client, path, params):
    """One paced public read. Returns parsed JSON, or None on a 4xx other than 429."""
    for pause in (30, 60, 120, 240, None):
        if time.time() > deadline[0]:
            raise OutOfTime
        wait = last_request[0]+PACE-time.time()
        if wait > 0:
            time.sleep(wait)
        last_request[0] = time.time()
        log["requests"] += 1
        try:
            resp = client.get(config.KALSHI_PROD_BASE+path, params=params, headers={"User-Agent": "research-client"})
        except httpx.RequestError:
            resp = None
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
    temp.write_text(json.dumps(value))
    temp.replace(path)


def finish_log(step, began):
    rows = load(OUT/"download-log.json", [])
    rows.append(dict(step=step, began=datetime.fromtimestamp(began, UTC).isoformat(), seconds=round(time.time()-began), **log))
    dump(OUT/"download-log.json", rows)
    print(step, dict(log), f"{(time.time()-began)/60:.1f} min", flush=True)


# ---------------------------------------------------------------- listings

def ticker_day(ticker):
    part = ticker.split("-")[1]
    try:
        return date(2000+int(part[:2]), MONTHS[part[2:5]], int(part[5:7]))
    except (KeyError, ValueError):
        return None


def slim(m, archived):
    def num(key):
        try:
            return float(m.get(key))
        except (TypeError, ValueError):
            return None
    return dict(ticker=m["ticker"], event=m["event_ticker"], archived=archived, kind=m.get("strike_type"), floor=m.get("floor_strike"),
                cap=m.get("cap_strike"), result=m.get("result"), final=num("expiration_value"), open=int(timestamp(m["open_time"])),
                close=int(timestamp(m["close_time"])), settle=int(timestamp(m["settlement_ts"])) if m.get("settlement_ts") else None,
                volume=float(m.get("volume_fp") or m.get("volume") or 0))


def list_series(client, series, archive):
    path = RAW/f"markets-{series}.json"
    have = load(path, None)
    if have is not None and (have["archive_done"] or not archive):
        return have
    have = have or dict(fee=None, markets={}, archive_done=False, recent_done=False)
    if have["fee"] is None:
        data = read(client, f"/series/{series}", {})
        s = (data or {}).get("series") or {}
        have["fee"] = dict(fee_type=s.get("fee_type"), fee_multiplier=s.get("fee_multiplier"), title=s.get("title"))
    for archived, route, extra in ((False, "/markets", RECENT), (True, "/historical/markets", dict(limit=1000))):
        if (archived and not archive) or have["archive_done" if archived else "recent_done"]:
            continue
        cursor, seen = "", set()
        while True:
            data = read(client, route, dict(extra, series_ticker=series, **({"cursor": cursor} if cursor else {})))
            for m in (data or {}).get("markets", []):
                if m.get("result") in ("yes", "no") and m["ticker"] not in have["markets"]:
                    have["markets"][m["ticker"]] = slim(m, archived)
            cursor = (data or {}).get("cursor")
            if not cursor or cursor in seen or not data["markets"]:
                break
            seen.add(cursor)
        have["archive_done" if archived else "recent_done"] = True
        dump(path, have)
    return have


def cmd_list(client):
    for name, spec in FAMILIES.items():
        for series in spec["series"]:
            have = list_series(client, series, bool(spec.get("archive")) and series not in HIGHS)
            ms = have["markets"].values()
            days = {m["event"] for m in ms}
            print(f"{name:24} {series:16} {len(ms):6} markets {len(days):5} events fee {have['fee']['fee_type']} "
                  f"{min((m['close'] for m in ms), default=0) and datetime.fromtimestamp(min(m['close'] for m in ms), UTC).date()}", flush=True)


def markets_of(series):
    """Listing for one series; the seven original high-temperature cities also get the archive listing left on disk."""
    ms = dict(load(RAW/f"markets-{series}.json", dict(markets={}))["markets"])
    old = ROOT/f"data/raw/weather-state/{series}-markets.json"
    if series in ORIGINAL and old.exists():
        for m in json.loads(old.read_text()):
            if m["ticker"] not in ms and m["archived"]:
                ms[m["ticker"]] = dict(ticker=m["ticker"], event=m["event"], archived=True, kind=m["kind"], floor=m["floor"], cap=m["cap"],
                                       result=m["result"], final=m["final"], open=int(m["open"]), close=int(m["close"]), settle=None,
                                       volume=m["volume"])
    return ms


# ---------------------------------------------------------------- bars

def money(group, field):
    value = group.get(field+"_dollars", group.get(field)) if group else None
    return None if value is None else round(100*float(value), 2)


def compact(bars):
    """[end_ts, yes bid close, yes ask close, trade low, trade high, volume] per hourly bar."""
    out = []
    for b in sorted(bars or [], key=lambda b: b["end_period_ts"]):
        out.append([b["end_period_ts"], money(b.get("yes_bid"), "close"), money(b.get("yes_ask"), "close"),
                    money(b.get("price"), "low"), money(b.get("price"), "high"), float(b.get("volume_fp", b.get("volume")) or 0)])
    return out


def bars_of(series):
    return load(RAW/f"bars-{series}.json", {})


def cmd_recent(client, first=()):
    order = sorted(FAMILIES, key=lambda name: (first.index(name) if name in first else len(first), list(FAMILIES).index(name)))
    for name in order:
        spec = FAMILIES[name]
        for series in spec["series"]:
            have = bars_of(series)
            todo = sorted((m for m in markets_of(series).values() if not m["archived"] and m["ticker"] not in have),
                          key=lambda m: (m["close"], m["ticker"]))
            groups, cur = [], []
            for m in todo:
                trial = cur+[m]
                hours = (max(x["close"] for x in trial)-min(x["open"] for x in trial))//3600+3
                if cur and (len(trial) > 100 or len(trial)*hours > 9000):
                    groups.append(cur)
                    cur = []
                cur.append(m)
            groups += [cur] if cur else []
            try:
                for n, group in enumerate(groups):
                    start, end = min(m["open"] for m in group), max(m["close"] for m in group)
                    if (end-start)//3600+3 > 9000:
                        log["window_too_long"] += len(group)
                        continue
                    data = read(client, "/markets/candlesticks", dict(market_tickers=",".join(m["ticker"] for m in group),
                                                                      start_ts=start, end_ts=end+3600, period_interval=60))
                    if data is None:
                        log["batch_refused"] += 1
                        continue
                    got = {x["market_ticker"]: compact(x.get("candlesticks")) for x in data.get("markets", [])}
                    for m in group:
                        have[m["ticker"]] = got.get(m["ticker"], [])
                    if n % 20 == 19:
                        dump(RAW/f"bars-{series}.json", have)
            finally:
                dump(RAW/f"bars-{series}.json", have)
            print(f"{name:24} {series:16} recent bars for {sum(1 for m in markets_of(series).values() if not m['archived'] and m['ticker'] in have)}"
                  f" markets, {len(groups)} requests", flush=True)


def previous_final(ms):
    """{event: settled value of the previous event of the series} - public before the next ladder trades."""
    finals = {}
    for m in ms.values():
        if m["final"] is not None:
            finals[m["event"]] = (max(m["close"], finals.get(m["event"], (0, 0))[0]), m["final"])
    order = sorted(finals, key=lambda e: finals[e][0])
    return {e: finals[p][1] for p, e in zip(order, order[1:])}, {e: finals[e][1] for e in order}


def strike(m):
    return m["floor"] if m["floor"] is not None else m["cap"]


def whole_cent(m):
    return round(strike(m)*1000) % 10 == 0


def shuffle_key(event):
    return hashlib.sha256(event.encode()).hexdigest()


def archive_plan(weekly=False):
    """Archive markets to read, whole events in a fixed pseudo-random order so that a cut-off still spans the calendar.
    Gas: every daily ladder, whole-cent strikes only (ladders step by half a cent). Temperature: one original city per day."""
    gas = []
    for series in ("KXAAAGASD",) + (("KXDIESELD", "KXAAAGASW", "KXDIESELW") if weekly else ()):
        ms = markets_of(series)
        sizes = Counter(m["event"] for m in ms.values())
        gas += [(series, m) for m in ms.values() if m["archived"] and sizes[m["event"]] >= 5 and whole_cent(m)]
    temp = []
    for i, series in enumerate(ORIGINAL):
        for m in markets_of(series).values():
            day = ticker_day(m["ticker"])
            if m["archived"] and day and day.toordinal() % 7 == i:
                temp.append((series, m))
    for q in (gas, temp):
        q.sort(key=lambda x: (shuffle_key(x[1]["event"]), x[1]["ticker"]))
    plan = []
    while gas or temp:                                  # alternate whole events
        for q in (gas, temp):
            if q:
                event = q[0][1]["event"]
                while q and q[0][1]["event"] == event:
                    plan.append(q.pop(0))
    return plan


def cmd_archive(client, weekly_share):
    plan = archive_plan(weekly_share)
    stores = {}
    todo = [(s, m) for s, m in plan if m["ticker"] not in stores.setdefault(s, bars_of(s))]
    print(f"archive plan {len(plan)} markets, {len(todo)} still to read: {dict(Counter(s for s, _ in todo))}", flush=True)
    try:
        for n, (series, m) in enumerate(todo):
            data = read(client, f"/historical/markets/{m['ticker']}/candlesticks",
                        dict(start_ts=m["open"], end_ts=m["close"]+3600, period_interval=60))
            stores[series][m["ticker"]] = compact((data or {}).get("candlesticks"))
            if n % 100 == 99:
                for s, store in stores.items():
                    dump(RAW/f"bars-{s}.json", store)
                print(f"  {n+1} of {len(todo)} {dict(log)}", flush=True)
    finally:
        for s, store in stores.items():
            dump(RAW/f"bars-{s}.json", store)



# ---------------------------------------------------------------- rows

def city_zone(series):
    return ZONES[re.sub(r"^KX(HIGHT|HIGH|LOWT)", "", series)]


def ref_ts(ref, m, series, close):
    if ref.endswith("h"):
        return close-int(ref[:-1])*3600
    hit = re.match(r"(\d\d):(\d\d)(-1)?", ref)
    day = ticker_day(m["ticker"])
    if day is None:
        return None
    day -= timedelta(days=1 if hit[3] else 0)
    return int(datetime(day.year, day.month, day.day, int(hit[1]), int(hit[2]), tzinfo=ZoneInfo(city_zone(series))).timestamp())


def quote_at(bars, when):
    """Index of the last bar ending at or before `when`, or None."""
    idx = None
    for i, b in enumerate(bars):
        if b[0] <= when:
            idx = i
        else:
            break
    return idx


def temp_bounds(m):
    if m["kind"] == "less":
        return -999, int(m["cap"])-1
    if m["kind"] == "greater":
        return int(m["floor"])+1, 999
    return int(m["floor"]), int(m["cap"])


def disk_minute_bars():
    """New York archive days, 13:00 only: minute bars left by weather_state_download.py, folded into hourly bars."""
    path = ROOT/"data/raw/weather-state/KXHIGHNY-bars.json"
    out = {}
    if not path.exists():
        return out
    for m in json.loads(path.read_text()):
        if not m["archived"]:
            continue
        hours = {}
        for end, _, bid, _, ask, vol, price in m["bars"]:
            h = hours.setdefault(-(-end//3600)*3600, [0, None, None, None, None, 0.0])
            h[0] = -(-end//3600)*3600
            if bid is not None and ask is not None:
                h[1], h[2] = bid, ask
            if price is not None and vol:
                h[3] = price if h[3] is None else min(h[3], price)
                h[4] = price if h[4] is None else max(h[4], price)
            h[5] += vol
        bars, last = [], (None, None)
        for key in sorted(hours):                      # carry the quote into hours whose minutes had none
            h = hours[key]
            if h[1] is None:
                h[1], h[2] = last
            last = (h[1], h[2])
            bars.append(h)
        out[m["ticker"]] = bars
    return out


def build_rows():
    rows, cover, ladders, explore = [], Counter(), {}, []
    minute = disk_minute_bars()
    for family, spec in FAMILIES.items():
        for series in spec["series"]:
            ms, bars = markets_of(series), bars_of(series)
            prev, finals = previous_final(ms)
            events = defaultdict(list)
            for m in ms.values():
                events[m["event"]].append(m)
            for event, group in events.items():
                close = max(m["close"] for m in group)
                day = ticker_day(group[0]["ticker"]) or datetime.fromtimestamp(close, UTC).date()
                if len(group) < 3:
                    continue
                for ref in tuple(spec["refs"])+tuple(spec.get("implied_refs", ()))+tuple(spec.get("explore_refs", ())):
                    quotes = []
                    for m in group:
                        source = "hourly"
                        b = bars.get(m["ticker"])
                        if b is None and ref == "13:00" and m["ticker"] in minute:
                            b, source = minute[m["ticker"]], "minute"
                        if b is None:
                            cover[family, ref, "not downloaded"] += 1
                            continue
                        when = ref_ts(ref, m, series, close)
                        i = quote_at(b, when) if when and m["open"] < when < m["close"] else None
                        if i is None:
                            cover[family, ref, "no bar before the reference time"] += 1
                            continue
                        bid, ask = b[i][1], b[i][2]
                        if bid is None or ask is None or not 0 <= bid <= ask <= 100:
                            cover[family, ref, "no quote"] += 1
                            continue
                        quotes.append((m, (bid+ask)/2))
                        if ref in spec.get("implied_refs", ()):
                            continue
                        yes = bid+ask > 100
                        sbid, sask = (bid, ask) if yes else (100-ask, 100-bid)
                        mid = (sbid+sask)/2
                        if not (bid >= 1 and ask <= 99 and sask-sbid <= 4 and 50 < mid < 99):
                            cover[family, ref, "one-sided, wide or 99c+"] += 1
                            continue
                        later = [x for x in b[i+1:] if x[0] <= close+3600]
                        lows = [(x[3] if yes else 100-x[4]) for x in later if x[3] is not None and x[5] > 0]
                        nxt = [x for x in later if x[0] <= when+3600]
                        nlow = [(x[3] if yes else 100-x[4]) for x in nxt if x[3] is not None and x[5] > 0]
                        rest = sbid+1 if sbid+1 < sask else sbid
                        win = (m["result"] == "yes") == yes
                        before = None
                        if i and b[i-1][1] is not None and b[i-1][2] is not None and b[i][0]-b[i-1][0] <= 7200:
                            before = (b[i-1][1]+b[i-1][2])/2 if yes else 100-(b[i-1][1]+b[i-1][2])/2
                        row = dict(family=family, hour_change=None if before is None else mid-before, series=series, event=event, ticker=m["ticker"], ref=ref, day=str(day), when=when,
                                   side="YES" if yes else "NO", mid=mid, bid=sbid, ask=sask, win=int(win), source=source,
                                   age=(when-b[i][0])/3600, rest=rest, filled=int(any(v < rest for v in lows)),
                                   print_next_hour=int(any(v <= sask for v in nlow)), next_hour_volume=sum(x[5] for x in nxt),
                                   day_volume=m["volume"], strike=strike(m), final=m["final"], prev=prev.get(event),
                                   kind=m["kind"], floor=m["floor"], cap=m["cap"], archived=m["archived"])
                        (rows if ref in spec["refs"] else explore).append(row)
                        cover[family, ref, "rows"] += 1
                    if family.startswith("Temp") and len(quotes) == len(group) and all(q[0]["kind"] in ("less", "greater", "between") for q in quotes):
                        total = sum(q[1] for q in quotes)
                        if 60 <= total <= 140:
                            centre = lambda b: b[1]-1 if b[0] < -900 else b[0]+1 if b[1] > 900 else (b[0]+b[1])/2
                            ladders[event, ref] = sum(q[1]*centre(temp_bounds(q[0])) for q in quotes)/total
                    elif len(quotes) >= 5 and all(q[0]["kind"] == "greater" for q in quotes):
                        quotes.sort(key=lambda q: q[0]["floor"])       # cumulative ladder: strike where the YES midpoint crosses 50c
                        for (a, pa), (b, pb) in zip(quotes, quotes[1:]):
                            if pa >= 50 > pb:
                                ladders[event, ref] = a["floor"]+(b["floor"]-a["floor"])*(pa-50)/(pa-pb)
                                break
    for r in rows+explore:
        r["implied"] = ladders.get((r["event"], r["ref"]))
        r["evening"] = ladders.get((r["event"], "20:00-1"))
    return rows, cover, explore


# ---------------------------------------------------------------- statistics

def week_of(day):
    y, w, _ = date.fromisoformat(day).isocalendar()
    return f"{y}-W{w:02d}"


def boot(rows, value, unit, draws=2000):
    """Mean of value(row) with a 95% interval from resampling whole days or weeks."""
    if not rows:
        return None, (None, None), 0
    by = defaultdict(lambda: [0.0, 0])
    for r in rows:
        c = by[r["day"] if unit == "day" else week_of(r["day"])]
        c[0] += value(r)
        c[1] += 1
    c = np.array(list(by.values()))
    est = c[:, 0].sum()/c[:, 1].sum()
    if len(c) < 3:
        return est, (None, None), len(c)
    pick = RNG.integers(0, len(c), (draws, len(c)))
    b = c[pick, 0].sum(1)/c[pick, 1].sum(1)
    return est, tuple(float(v) for v in np.percentile(b, [2.5, 97.5])), len(c)


def fee(price):
    return 7*(price/100)*(1-price/100)


def taker_net(r):
    return 100*r["win"]-r["ask"]-fee(r["ask"])


def maker_net(r):
    return 100*r["win"]-r["rest"]


def f(v, width=5, digits=1):
    return " "*width if v is None else f"{v:+{width}.{digits}f}"


def ci(pair):
    return "[   n/a    ]" if pair[0] is None else f"[{pair[0]:+5.1f},{pair[1]:+5.1f}]"


def split_days(rows):
    days = sorted({r["day"] for r in rows})
    cut = set(days[:len(days)//2])
    return [r for r in rows if r["day"] in cut], [r for r in rows if r["day"] not in cut]


def band(rows, lo, hi):
    return [r for r in rows if lo <= r["mid"] < hi]


def calibration_line(rows):
    gap = lambda r: 100*r["win"]-r["mid"]
    g, cd, nd = boot(rows, gap, "day")
    _, cw, nw = boot(rows, gap, "week")
    a, b = split_days(rows)
    ga, gb = boot(a, gap, "day")[0], boot(b, gap, "day")[0]
    return (f"n {len(rows):5} days {nd:4} wk {nw:3} price {np.mean([r['mid'] for r in rows]):5.1f} win {100*np.mean([r['win'] for r in rows]):5.1f} "
            f"losses {sum(1-r['win'] for r in rows):3} gap {f(g)} day-ci {ci(cd)} week-ci {ci(cw)} | half1 {len(a):4} {f(ga)} | half2 {len(b):4} {f(gb)}"), g, cd


def trade_line(rows):
    t = [r for r in rows if r["ask"] <= 99]
    out = dict(n=len(t))
    if not t:
        return "no rows", out
    net, cd, nd = boot(t, taker_net, "day")
    _, cw, _ = boot(t, taker_net, "week")
    paid = np.mean([r["ask"]+fee(r["ask"]) for r in t])
    filled = [r for r in t if r["filled"]]
    unfilled = [r for r in t if not r["filled"]]
    mnet, mcd, _ = boot(filled, maker_net, "day")
    mpaid = np.mean([r["rest"] for r in filled]) if filled else None
    a, b = split_days(t)
    out.update(taker=net, taker_ci=cd, taker_week_ci=cw, taker_pct=100*net/paid, days=nd, paid=paid, fills=len(filled),
               maker=mnet, maker_ci=mcd, maker_pct=100*mnet/mpaid if filled else None,
               maker_per_order=mnet*len(filled)/len(t) if filled else 0.0,
               half1=boot(a, taker_net, "day")[0], half2=boot(b, taker_net, "day")[0])
    text = (f"TAKER n {len(t):5} ask {np.mean([r['ask'] for r in t]):5.1f} fee {np.mean([fee(r['ask']) for r in t]):4.2f} net {f(net, 6, 2)}c "
            f"({100*net/paid:+5.2f}%) day-ci {ci(cd)} week-ci {ci(cw)} halves {f(out['half1'], 6, 2)}/{f(out['half2'], 6, 2)}"
            f" | MAKER fills {len(filled):4} ({100*len(filled)/len(t):3.0f}%) "
            + (f"bid {mpaid:5.1f} won {100*np.mean([r['win'] for r in filled]):5.1f}% vs unfilled {100*np.mean([r['win'] for r in unfilled]) if unfilled else float('nan'):5.1f}% "
               f"net {f(mnet, 6, 2)}c ({100*mnet/mpaid:+5.2f}%) day-ci {ci(mcd)} per order {f(out['maker_per_order'], 6, 2)}c" if filled else ""))
    return text, out


def daily(rows, net=taker_net):
    """{day: (profit in cents at one contract per strike, strikes, losers)}"""
    by = defaultdict(lambda: [0.0, 0, 0])
    for r in rows:
        d = by[r["day"]]
        d[0] += net(r)
        d[1] += 1
        d[2] += 1-r["win"]
    return by


# ---------------------------------------------------------------- outside data

def cmd_eia():
    """Weekly US regular gasoline retail price (EIA series, served as CSV by the St. Louis Fed). No key, generic client."""
    resp = httpx.get("https://fred.stlouisfed.org/graph/fredgraph.csv", params={"id": "GASREGW"}, timeout=60,
                     headers={"User-Agent": "research-client"})
    resp.raise_for_status()
    (RAW/"GASREGW.csv").write_text(resp.text)
    print(len(resp.text.splitlines()), "weekly rows saved")


def weekly_gas():
    path = RAW/"GASREGW.csv"
    out = []
    if path.exists():
        for line in path.read_text().splitlines()[1:]:
            d, _, v = line.partition(",")
            try:
                out.append((d, float(v)))
            except ValueError:
                pass
    return out


def temperature_state(rows):
    """Adds r['room']: degrees between the highest reading so far (hourly METARs, usable 20 minutes after their stamp) and the
    strike range. Negative room on a NO row = the range is already below the day's maximum, so it can no longer win."""
    try:
        from research.weather_state import observations
        from research.weather_state_download import CITIES
    except Exception as exc:        # the observation files are optional
        print("no observations:", exc)
        return
    cache = {}
    for r in rows:
        if r["family"] != "Temp high daily" or r["series"] not in CITIES or r["day"] > "2026-11-15":
            continue
        _, station, _, _, offset = CITIES[r["series"]]
        if station not in cache:
            try:
                cache[station] = observations(station)
            except Exception:
                cache[station] = []
        d = date.fromisoformat(r["day"])
        start = datetime(d.year, d.month, d.day, tzinfo=UTC).timestamp()-3600*offset
        seen = [max(o[1], o[2] or -999) if o[0] > start+6*3600 else o[1] for o in cache[station] if start <= o[0] <= r["when"]-1200]
        if not seen:
            continue
        top = max(seen)
        lo, hi = temp_bounds(r)
        r["max_so_far"] = top
        r["room"] = lo-top if lo > top else (hi-top if hi < top else 0)      # >0: range above the max; <0: range already passed; 0: inside


# ---------------------------------------------------------------- report

def cmd_report():
    rows, cover, explore = build_rows()
    temperature_state(rows)
    lines = []
    say = lambda text="": lines.append(text)
    summary = dict(cells={}, tails={}, coverage={f"{k[0]} @{k[1]}: {k[2]}": v for k, v in sorted(cover.items())})
    keys = sorted({(r["family"], r["ref"]) for r in rows}, key=lambda k: (list(FAMILIES).index(k[0]), k[1]))

    say("COVERAGE (markets x reference times)")
    for k, v in sorted(cover.items(), key=lambda kv: (list(FAMILIES).index(kv[0][0]), kv[0][1], kv[0][2])):
        say(f"  {k[0]:24} @{k[1]:8} {k[2]:36} {v:6}")
    for fam, ref in keys:
        sub = [r for r in rows if (r["family"], r["ref"]) == (fam, ref)]
        days = sorted({r["day"] for r in sub})
        say(f"  {fam:24} @{ref:8} rows {len(sub):5}, {len(days)} days {days[0]}..{days[-1]}; archive days {len({r['day'] for r in sub if r['archived']})}, "
            f"recent days {len({r['day'] for r in sub if not r['archived']})}; sources {dict(Counter(r['source'] for r in sub))}")

    say("\n1. CALIBRATION by band of the midpoint. gap = win% - price. Intervals resample whole days / ISO weeks.")
    for fam, ref in keys:
        sub = [r for r in rows if (r["family"], r["ref"]) == (fam, ref)]
        say(f"\n{fam} @{ref}")
        for side in ("NO", "YES", "both"):
            for lo, hi in BANDS:
                cell = [r for r in band(sub, lo, hi) if side in (r["side"], "both")]
                if len(cell) >= 5:
                    text, g, cd = calibration_line(cell)
                    say(f"  {side:4} {lo}-{hi}: {text}")
        slices = [("archive (settled before 2026-09-18)", lambda r: r["archived"]), ("recent tier", lambda r: not r["archived"])]
        if fam == "Temp high daily":
            slices += [("recent tier, 7 original cities", lambda r: not r["archived"] and r["series"] in ORIGINAL),
                       ("recent tier, 13 newer cities", lambda r: not r["archived"] and r["series"] not in ORIGINAL),
                       ("New York only (all days)", lambda r: r["series"] == "KXHIGHNY")]
        if fam == "Gas+diesel daily (US)":
            slices += [("gas only", lambda r: r["series"] == "KXAAAGASD"), ("diesel only", lambda r: r["series"] == "KXDIESELD")]
        for label, pick in slices:
            for side in ("NO", "YES", "both"):
                cell = [r for r in band(sub, 90, 99) if pick(r) and side in (r["side"], "both")]
                if len(cell) >= 5:
                    say(f"  {side:4} 90-99 {label}: {calibration_line(cell)[0]}")
                    say(f"  {'':4}       {'':{len(label)}}  {trade_line(cell)[0]}")

    say("\n2. TRADEABLE. TAKER pays the ask + 0.07p(1-p). MAKER rests 1c above the best bid, filled only by a later print strictly below, no fee."
        "\n   Percent = net / price paid. halves = taker net on the earlier / later half of days.")
    for fam, ref in keys:
        sub = [r for r in rows if (r["family"], r["ref"]) == (fam, ref)]
        say(f"\n{fam} @{ref}")
        for side in ("NO", "YES", "both"):
            for lo, hi in BANDS:
                cell = [r for r in band(sub, lo, hi) if side in (r["side"], "both")]
                if len(cell) >= 5:
                    text, out = trade_line(cell)
                    say(f"  {side:4} {lo}-{hi}: {text}")
                    if side == "both":
                        summary["cells"][f"{fam} @{ref} {lo}-{hi}"] = out

    say("\n3. TAIL. Rule: buy every 90-99c contract at the ask at the reference time, one contract each, hold. Profit in cents per day."
        "\n   (National gas uses whole-cent strikes only, so that archive and recent days are comparable; the full ladder is about twice that.)")
    for fam, ref in keys:
        sub = [r for r in band(rows, 90, 99) if (r["family"], r["ref"]) == (fam, ref) and r["ask"] <= 99]
        if fam == "Gas+diesel daily (US)":
            sub = [r for r in sub if whole_cent(r)]
        if len(sub) < 20:
            continue
        by = daily(sub)
        days = sorted(by)
        pnl = np.array([by[d][0] for d in days])
        count = np.array([by[d][1] for d in days])
        losers = np.array([by[d][2] for d in days])
        weeks = defaultdict(float)
        for d in days:
            weeks[week_of(d)] += by[d][0]
        worst_week = min(weeks, key=weeks.get)
        span = (date.fromisoformat(days[-1])-date.fromisoformat(days[0])).days+1
        say(f"\n{fam} @{ref}: {len(sub)} contracts on {len(days)} days (calendar span {span}); strikes per day mean {count.mean():.1f} max {count.max()}")
        say(f"  daily profit c: mean {pnl.mean():+.1f} median {np.median(pnl):+.1f} p5 {np.percentile(pnl, 5):+.1f} p1 {np.percentile(pnl, 1):+.1f} "
            f"min {pnl.min():+.1f}; losing days {int((pnl < 0).sum())} ({100*(pnl < 0).mean():.1f}%); days with any losing strike {int((losers > 0).sum())}")
        say(f"  worst week {worst_week}: {weeks[worst_week]:+.1f}c; weeks {len(weeks)}, losing weeks {sum(v < 0 for v in weeks.values())}; total {pnl.sum():+.0f}c")
        say(f"  losers per day when at least one loses: {dict(sorted(Counter(int(x) for x in losers[losers > 0]).items()))}")
        good = pnl[losers == 0].mean() if (losers == 0).any() else 0.0
        cost = np.mean([r["ask"]+fee(r["ask"]) for r in sub])
        say(f"  mean profit on a day with no loser {good:+.1f}c; a day on which EVERY strike held loses {count.mean()*cost:.0f}c on average "
            f"(max {max(sum(r['ask']+fee(r['ask']) for r in sub if r['day'] == d) for d in days):.0f}c)"
            f" -> full-wipe break-even: one such day per {1+count.mean()*cost/max(good, 1e-9):.0f} trading days")
        for side in ("NO", "YES"):
            part = [r for r in sub if r["side"] == side]
            if part:
                k = len(part)/len(days)
                c1 = np.mean([r["ask"]+fee(r["ask"]) for r in part])
                g1 = np.mean([100-r["ask"]-fee(r["ask"]) for r in part])
                say(f"    {side} side: {k:.1f} contracts per day, wins {g1:.2f}c each, a loss costs {c1:.1f}c -> break-even loss rate {100*g1/(g1+c1):.2f}% of contracts "
                    f"(observed {100*np.mean([1-r['win'] for r in part]):.2f}%)")
        say("  ten worst days (date, strikes, losers, profit c, underlying):")
        for i in np.argsort(pnl)[:10]:
            d = days[i]
            lost = [r for r in sub if r["day"] == d and not r["win"]]
            any_row = [r for r in sub if r["day"] == d]
            if fam.startswith("Gas"):
                r0 = any_row[0]
                extra = "; ".join(f"{s}: prev {x['prev']} final {x['final']} move {100*(x['final']-x['prev']):+.1f}c, ladder median {x['implied'] and round(x['implied'], 4)}"
                                  for s, x in {r['series']: r for r in any_row if r['prev'] is not None and r['final'] is not None}.items())
            else:
                extra = "; ".join(f"{r['series'][2:]} {r['side']} {r['floor']}-{r['cap']} at {r['ask']:.0f}c, final {r['final']}, ladder mean {r['implied'] and round(r['implied'], 1)}"
                                  f" (evening {r.get('evening') and round(r['evening'], 1)})" for r in lost[:6])
            say(f"    {d} {by[d][1]:3} {by[d][2]:2} {by[d][0]:+8.1f}  {extra}")
        summary["tails"][f"{fam} @{ref}"] = dict(days=len(days), contracts=len(sub), mean_day=float(pnl.mean()), worst_day=float(pnl.min()),
                                                  worst_week=float(weeks[worst_week]), strikes_per_day=float(count.mean()),
                                                  loser_days=int((losers > 0).sum()), max_losers=int(losers.max()))

    gas_tail(rows, say, summary)
    say("\n   EXPLORATORY, national gas and diesel at other hours before the close (not fixed in advance; shows when the ladder learns the print):")
    for ref in ("12h", "9h", "6h", "3h", "1h"):
        part = [r for r in band(explore+rows, 90, 99) if r["family"] == "Gas+diesel daily (US)" and r["ref"] == ref and r["ask"] <= 99]
        seen = {r["event"]: abs(100*(r["final"]-r["implied"])) for r in explore+rows
                if r["family"] == "Gas+diesel daily (US)" and r["ref"] == ref and r["implied"] is not None and r["final"] is not None}
        if part and seen:
            net, cd, nd = boot(part, taker_net, "day")
            v = np.array(list(seen.values()))
            say(f"     {ref:>3} before close: |settled - ladder median| median {np.median(v):.2f}c p90 {np.percentile(v, 90):.2f}c | 90-99c contracts n {len(part):4} on {nd} days, "
                f"losses {sum(1-r['win'] for r in part):2} NO {sum(1-r['win'] for r in part if r['side'] == 'NO')} YES {sum(1-r['win'] for r in part if r['side'] == 'YES')}, "
                f"taker net {net:+.2f}c day-ci {ci(cd)}")
    temp_tail(rows, say, summary)

    say("\n4. BUYABILITY of the 90-99c contracts")
    say("   spread = displayed; print = a trade at or below our ask in the hour after the reference time; vol = contracts traded in that hour (all prices);"
        "\n   age = hours since the quote last changed.")
    for fam, ref in keys:
        sub = [r for r in band(rows, 90, 99) if (r["family"], r["ref"]) == (fam, ref) and r["source"] == "hourly"]
        if len(sub) < 20:
            continue
        by = defaultdict(float)
        for r in sub:
            by[r["day"]] += r["next_hour_volume"] if r["print_next_hour"] else 0
        sp = np.array([r["ask"]-r["bid"] for r in sub])
        vol = np.array([r["next_hour_volume"] for r in sub])
        hit = np.array([r["print_next_hour"] for r in sub], bool)
        age = np.array([r["age"] for r in sub])
        say(f"  {fam:24} @{ref:8} n {len(sub):5} spread mean {sp.mean():.2f} (1c: {100*(sp <= 1).mean():.0f}%) | print within the hour {100*hit.mean():.0f}% | "
            f"next-hour vol median {np.median(vol):.0f} mean {vol.mean():.0f} | per day, hour volume on strikes that printed: median {np.median(list(by.values())):.0f} | "
            f"quote age median {np.median(age):.0f}h, >3h {100*(age > 3).mean():.0f}%")
        for label, pick in (("printed", hit), ("no print", ~hit)):
            part = [r for r, p_ in zip(sub, pick) if p_ and r["ask"] <= 99]
            if len(part) >= 10:
                net, cd, _ = boot(part, taker_net, "day")
                say(f"      {label:9} n {len(part):5} win {100*np.mean([r['win'] for r in part]):5.1f}% taker net {net:+.2f}c day-ci {ci(cd)}")
    say("\n   Temperature highs, seven original cities with hourly observations (to 2026-11-15): where does the gap sit?")
    say("   room = degrees between the highest reading so far and the strike range (NO rows): <0 already passed (cannot win), >0 still above.")
    for ref in ("07:00", "13:00"):
        sub = [r for r in band(rows, 90, 99) if r["family"] == "Temp high daily" and r["ref"] == ref and "room" in r and r["ask"] <= 99]
        groups = [("NO, range already below the max so far", lambda r: r["side"] == "NO" and r["room"] < 0),
                  ("NO, max so far inside the range", lambda r: r["side"] == "NO" and r["room"] == 0),
                  ("NO, range 1-3F above the max so far", lambda r: r["side"] == "NO" and 1 <= r["room"] <= 3),
                  ("NO, range 4-7F above", lambda r: r["side"] == "NO" and 4 <= r["room"] <= 7),
                  ("NO, range 8F+ above", lambda r: r["side"] == "NO" and r["room"] >= 8),
                  ("YES rows", lambda r: r["side"] == "YES")]
        for label, pick in groups:
            part = [r for r in sub if pick(r)]
            if not part:
                say(f"    @{ref} {label:40} n    0")
            if len(part) >= 5:
                net, cd, _ = boot(part, taker_net, "day")
                say(f"    @{ref} {label:40} n {len(part):4} ask {np.mean([r['ask'] for r in part]):.1f} win {100*np.mean([r['win'] for r in part]):5.1f}% "
                    f"taker net {net:+.2f}c day-ci {ci(cd)}")
    against_map(rows, say)
    quiet_quotes(rows, say)
    verdict(rows, say, summary)
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT/"report.txt").write_text("\n".join(lines)+"\n")
    dump(OUT/"summary.json", json.loads(json.dumps(summary, default=float)))
    dump(OUT/"rows.json", rows)
    print("\n".join(lines))


def gas_tail(rows, say, summary):
    sub = [r for r in rows if r["family"] == "Gas+diesel daily (US)" and r["series"] == "KXAAAGASD"]
    ms = markets_of("KXAAAGASD")
    prev, finals = previous_final(ms)
    days = {e: ticker_day(e+"-x") for e in finals}
    moves = sorted((days[e], 100*(finals[e]-prev[e])) for e in prev if days[e] and days.get(e))
    order = sorted((d, e) for e, d in days.items() if d)
    daily_moves = np.array([100*(finals[b]-finals[a]) for (da, a), (db, b) in zip(order, order[1:]) if (db-da).days == 1])
    say("\n   NATIONAL GAS: the underlying move")
    if len(daily_moves):
        a = np.abs(daily_moves)
        say(f"   AAA day-over-day change on {len(a)} consecutive-day pairs in the Kalshi listing ({order[0][0]}..{order[-1][0]}): "
            f"mean abs {a.mean():.2f}c, median {np.median(a):.2f}c, p90 {np.percentile(a, 90):.2f}c, p99 {np.percentile(a, 99):.2f}c, max {a.max():.2f}c")
    seen = {}
    for r in sub:
        if r["implied"] is not None and r["final"] is not None:
            seen[r["day"]] = (100*(r["final"]-r["implied"]), 100*(r["final"]-r["prev"]) if r["prev"] is not None else None)
    if seen:
        s = np.array([v[0] for v in seen.values()])
        say(f"   surprise = settled price - the ladder's own median at the reference time, {len(s)} days: sd {s.std():.2f}c, mean {s.mean():+.2f}c, "
            f"abs median {np.median(np.abs(s)):.2f}c, p95 {np.percentile(np.abs(s), 95):.2f}c, p99 {np.percentile(np.abs(s), 99):.2f}c, max {np.abs(s).max():.2f}c")
        both = [(abs(v[0]), abs(v[1])) for v in seen.values() if v[1] is not None]
        if len(both) > 20:
            x, y = np.array([b[1] for b in both]), np.array([b[0] for b in both])
            slope, inter = np.polyfit(x, y, 1)
            say(f"   |surprise| against the size of the day's move: |surprise| = {inter:.2f}c + {slope:.2f} x |move| (corr {np.corrcoef(x, y)[0, 1]:.2f})")
        # distance, in cents from the ladder median, of the 90-99c contracts
        held = [r for r in band(sub, 90, 99) if r["implied"] is not None and r["ask"] <= 99]
        near, far, n_side = [], [], []
        for d in sorted({r["day"] for r in held}):
            for side in ("NO", "YES"):
                dist = [abs(100*(r["strike"]-r["implied"])) for r in held if r["day"] == d and r["side"] == side]
                if dist:
                    near.append(min(dist)); far.append(max(dist)); n_side.append(len(dist))
        near, far = np.array(near), np.array(far)
        say(f"   distance of the 90-99c strikes from the ladder median, per day and side: nearest median {np.median(near):.2f}c (p10 {np.percentile(near, 10):.2f}c), "
            f"farthest median {np.median(far):.2f}c (p90 {np.percentile(far, 90):.2f}c); strikes per side mean {np.mean(n_side):.1f}")
        say("   Beyond the farthest 90-99c strike the ladder is at 99c+ and is not bought, so a 'wipe-out' = a surprise larger than that distance.")
        mad = np.median(np.abs(s-np.median(s)))*1.4826
        say(f"   chance per day that |surprise| exceeds the nearest / farthest 90-99c strike on one side:")
        emp_near = np.mean([np.mean(np.abs(s) > v) for v in near])
        emp_far = np.mean([np.mean(np.abs(s) > v) for v in far])
        say(f"     empirical surprises x empirical distances (independent draw): first loss {100*emp_near:.1f}%, whole side {100*emp_far:.2f}% of days")
        from math import erfc, sqrt
        for label, scale in (("normal, robust sd", mad), ("normal, sample sd", s.std())):
            p_near = np.mean([erfc(v/scale/sqrt(2)) for v in near])
            p_far = np.mean([erfc(v/scale/sqrt(2)) for v in far])
            say(f"     {label} ({scale:.2f}c): first loss {100*p_near:.1f}%, whole side {100*p_far:.3f}% of days")
        t3 = RNG.standard_t(3, 400000)*mad
        say(f"     Student-t 3 d.f. scaled to the robust sd: first loss {100*np.mean([np.mean(np.abs(t3) > v) for v in near]):.1f}%, "
            f"whole side {100*np.mean([np.mean(np.abs(t3) > v) for v in far]):.2f}% of days")
        summary["gas_surprise"] = dict(days=len(s), sd=float(s.std()), robust_sd=float(mad), max_abs=float(np.abs(s).max()),
                                       nearest_median=float(np.median(near)), farthest_median=float(np.median(far)))
    wk = weekly_gas()
    if wk:
        v = np.array([x[1] for x in wk])
        ch = 100*np.diff(v)
        dates = [x[0] for x in wk][1:]
        say(f"   EIA weekly US regular gasoline, {wk[0][0]}..{wk[-1][0]} ({len(ch)} weekly changes): sd {ch.std():.1f}c; "
            + "; ".join(f"|change| >= {t}c in {100*np.mean(np.abs(ch) >= t):.1f}% of weeks" for t in (5, 10, 15, 20, 30)))
        recent = [c for d, c in zip(dates, ch) if d >= "2026-01-01"]
        if recent:
            say(f"   same series over the Kalshi sample (2026): sd {np.std(recent):.1f}c, max abs {np.max(np.abs(recent)):.1f}c over {len(recent)} weeks")
        big = sorted(zip(np.abs(ch), dates), reverse=True)[:8]
        say("   largest weekly moves: " + ", ".join(f"{d} {c:.0f}c" for c, d in big))


def temp_tail(rows, say, summary):
    say("\n   TEMPERATURE HIGHS: forecast error inferred from the ladder (settled high minus the ladder's probability-weighted mean)")
    for ref, key in (("07:00", "evening"), ("07:00", "implied"), ("13:00", "implied")):
        seen = {}
        for r in rows:
            if r["family"] == "Temp high daily" and r["ref"] == ref and r.get(key) is not None and r["final"] is not None:
                seen[r["event"]] = (r["final"]-r[key], r["day"])
        if not seen:
            continue
        e = np.array([v[0] for v in seen.values()])
        a = np.abs(e)
        label = "evening before (20:00 local)" if key == "evening" else f"at {ref} local"
        say(f"   {label:30} {len(e)} city-days: mean {e.mean():+.2f}F sd {e.std():.2f}F; |error| >= 3F {100*np.mean(a >= 3):.1f}%, >= 5F {100*np.mean(a >= 5):.1f}%, "
            f">= 7F {100*np.mean(a >= 7):.2f}%, max {a.max():.1f}F")
    for ref in ("07:00", "13:00"):
        sub = [r for r in band(rows, 90, 99) if r["family"] == "Temp high daily" and r["ref"] == ref and r["ask"] <= 99 and not r["archived"]]
        if not sub:
            continue
        by_city = defaultdict(lambda: [0, 0])
        by_day = defaultdict(set)
        for r in sub:
            by_city[r["event"]][0] += 1
            by_city[r["event"]][1] += 1-r["win"]
            if not r["win"]:
                by_day[r["day"]].add(r["series"])
        say(f"   @{ref}, recent tier with all 20 cities: {len(by_city)} city-ladders, {sum(1 for v in by_city.values() if v[1])} with a losing contract, "
            f"most losers in one ladder {max(v[1] for v in by_city.values())}; cities losing on the same day: "
            f"{dict(sorted(Counter(len(v) for v in by_day.values()).items()))} (days with 1, 2, ... losing cities)")


def against_map(rows, say):
    """Same contracts, two price definitions: the map's 15-minute stable median (>= 3 one-minute bars within 3c) and this
    script's single hourly close. Shows how much of the map's gap came from which rows it kept."""
    say("\n5. AGAINST research/calibration_map.py (recent tier, 90-99c, same reference time)")
    for name, file, hours, fam, ref in (("Weather daily high @18h", "close-rows-weather-daily-high.json", 18, "Temp high daily", "07:00"),
                                        ("Weather daily high @12h", "close-rows-weather-daily-high.json", 12, "Temp high daily", "13:00"),
                                        ("Gas and diesel daily @6h", "close-rows-gas-and-diesel-daily.json", 6, "Gas+diesel daily (US)", "6h")):
        path = ROOT/"data/processed/calibration-map"/file
        if not path.exists():
            continue
        old = {r["ticker"]: r for r in json.loads(path.read_text()) if r["hours"] == hours and 90 <= r["start"] < 99}
        series = {r["series"] for r in old.values()}
        mine = {r["ticker"]: r for r in band(rows, 90, 99) if r["family"] == fam and r["ref"] == ref and not r["archived"] and r["series"] in series}
        both = set(old) & set(mine)
        line = lambda label, wins, prices: say(f"     {label:44} n {len(wins):4} price {np.mean(prices):5.1f} win {100*np.mean(wins):5.1f}% gap {100*np.mean(wins)-np.mean(prices):+.1f}c"
                                               f" losses {len(wins)-int(sum(wins))}") if wins else None
        say(f"   {name}: map rows {len(old)}, this script's rows in the same series {len(mine)}, in both {len(both)}")
        line("map rows, map price", [old[t]["payout"] for t in old], [old[t]["start"] for t in old])
        line("in both, map price", [old[t]["payout"] for t in both], [old[t]["start"] for t in both])
        line("in both, hourly close", [mine[t]["win"] for t in both], [mine[t]["mid"] for t in both])
        line("map only (not 90-99c at the hourly close)", [old[t]["payout"] for t in set(old)-both], [old[t]["start"] for t in set(old)-both])
        line("this script only (map dropped or other band)", [mine[t]["win"] for t in set(mine)-both], [mine[t]["mid"] for t in set(mine)-both])


def quiet_quotes(rows, say):
    say("\n   EXPLORATORY (suggested by the comparison above, not fixed in advance): 90-99c rows split by how far the midpoint moved"
        "\n   over the hour before the reference time. 'new' = no bar in the previous two hours.")
    for fam, ref in sorted({(r["family"], r["ref"]) for r in rows}, key=lambda k: (list(FAMILIES).index(k[0]), k[1])):
        sub = [r for r in band(rows, 90, 99) if (r["family"], r["ref"]) == (fam, ref) and r["ask"] <= 99]
        if len(sub) < 100:
            continue
        groups = (("moved <= 1c", lambda r: r["hour_change"] is not None and abs(r["hour_change"]) <= 1),
                  ("rose > 1c (towards certainty)", lambda r: r["hour_change"] is not None and r["hour_change"] > 1),
                  ("fell > 1c (away from certainty)", lambda r: r["hour_change"] is not None and r["hour_change"] < -1),
                  ("new", lambda r: r["hour_change"] is None))
        for label, pick in groups:
            part = [r for r in sub if pick(r)]
            if len(part) >= 10:
                net, cd, _ = boot(part, taker_net, "day")
                a, b = split_days(part)
                say(f"     {fam:22} @{ref:8} {label:32} n {len(part):5} win {100*np.mean([r['win'] for r in part]):5.1f}% taker net {net:+.2f}c day-ci {ci(cd)}"
                    f" halves {f(boot(a, taker_net, 'day')[0], 6, 2)}/{f(boot(b, taker_net, 'day')[0], 6, 2)}")


def verdict(rows, say, summary):
    from research.calibration_map import shrink
    say("\n6. VERDICT INPUTS. Taker net on the 90-99c band, each family x reference time, shrunk together (normal-normal, as calibration_map.shrink).")
    cells, info = {}, {}
    for fam, ref in sorted({(r["family"], r["ref"]) for r in rows}, key=lambda k: (list(FAMILIES).index(k[0]), k[1])):
        sub = [r for r in band(rows, 90, 99) if (r["family"], r["ref"]) == (fam, ref) and r["ask"] <= 99]
        if len(sub) < 30:
            continue
        net, cd, nd = boot(sub, taker_net, "day")
        if cd[0] is None:
            continue
        se = (cd[1]-cd[0])/3.92
        cells[f"{fam} @{ref}"] = (net, max(se, 0.05))
        days = sorted({r["day"] for r in sub})
        recent = [r for r in sub if not r["archived"]]
        rdays = len({r["day"] for r in recent}) or 1
        filled = [r for r in sub if r["filled"]]
        mnet = boot(filled, maker_net, "day")
        info[f"{fam} @{ref}"] = dict(n=len(sub), days=len(days), per_day_recent=len(recent)/rdays, paid=np.mean([r["ask"]+fee(r["ask"]) for r in sub]),
                                     maker=mnet[0], maker_ci=mnet[1], fill=len(filled)/len(sub))
    if len(cells) < 2:
        return
    out = shrink(cells)
    say(f"   {len(cells)} cells; overall mean {out['overall_mean']:+.2f}c (se {out['overall_se']}), between-cell sd {out['between_family_sd']}c")
    say(f"   {'cell':36} {'n':>5} {'days':>4} {'raw c':>6} {'se':>5} {'shrunk':>6} {'sd':>5} {'P>0':>5} {'% paid':>6} {'trades/day':>10} {'$/day @10':>9} {'$/day @100':>10} | maker: fill, net/fill, net/order")
    for k, (net, se) in cells.items():
        sh, i = out["families"][k], info[k]
        per = i["per_day_recent"]
        say(f"   {k:36} {i['n']:5} {i['days']:4} {net:+6.2f} {se:5.2f} {sh['shrunk']:+6.2f} {sh['sd']:5.2f} {sh['prob_positive']:5.2f} {100*sh['shrunk']/i['paid']:+6.2f} "
            f"{per:10.1f} {sh['shrunk']*per*10/100:+9.2f} {sh['shrunk']*per*100/100:+10.2f} | {100*i['fill']:3.0f}% {f(i['maker'], 6, 2)}c {ci(i['maker_ci'])} "
            f"{f(i['maker']*i['fill'] if i['maker'] is not None else None, 6, 2)}c")
    summary["verdict"] = dict(shrink=out, cells={k: dict(raw=v[0], se=v[1], **info[k]) for k, v in cells.items()})


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=("list", "recent", "archive", "eia", "report"))
    parser.add_argument("--minutes", type=float, default=9)
    parser.add_argument("--weekly-share", action="store_true", help="archive: also read archived diesel and weekly gas")
    parser.add_argument("--first", nargs="*", default=(), help="recent: family names to download first")
    args = parser.parse_args()
    began = time.time()
    deadline[0] = began+60*args.minutes
    if args.command in ("list", "recent", "archive"):
        with httpx.Client(timeout=40) as http:
            try:
                {"list": cmd_list, "recent": lambda c: cmd_recent(c, args.first)}.get(args.command, lambda c: cmd_archive(c, args.weekly_share))(http)
            except OutOfTime:
                print("out of time; run again to resume", flush=True)
        finish_log(args.command, began)
    elif args.command == "eia":
        cmd_eia()
    else:
        cmd_report()
