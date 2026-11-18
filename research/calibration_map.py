"""Calibration map of pre-event favourites across Kalshi: where do favourites win more often than their price?

Background (research/dip_flow.py, research/edge_reality.py): on some tennis
tours and in men's college basketball a side quoted at 90c or more before the
event wins more often than its price. This script measures the same thing for
every family it can reach, with the same definition, and then asks how much of
the picture survives the fact that dozens of cells were looked at.

Definitions (as in research/favorite_dip.py and research/dip_other_sports.py)

* One canonical market per two-outcome event (the lowest ticker); the favourite
  is the side whose pre-event midpoint is above 50c. Events with three or more
  outcome markets (soccer) use the YES side of the outcome with the highest
  pre-event midpoint.
* Pre-event price = median midpoint of the first 15 one-minute bars of the six
  hours before settlement. Each of those bars needs a two-sided quote with a
  spread of at most 4c, and the 15 midpoints must stay within 3c.
* Gap = 100 x settlement value - pre-event price. "Buy at the ask" pays the
  midpoint plus half the mean warm-up spread plus the taker fee 0.07*p*(1-p).
* Is the window really before the event? Kalshi publishes
  `expected_expiration_time` before the event (scheduled start plus a fixed
  allowance, about three hours for most sports). A row is "timing ok" when the
  warm-up ends before the start in the ticker (where the ticker carries one), or
  otherwise when settlement came no later than SLACK hours after the expected
  expiration. Rows that fail are either long events or, worse, markets that
  settled many hours after the event ended, where the "pre-event" quote is a
  stale post-event one. Tables are shown on all rows and on timing-ok rows.
  Tennis has no usable schedule (expected_expiration is an order-of-play
  placeholder); its rows are taken as pre-match because favourite_gap.py
  detected the start of play after the warm-up in every 90c+ match.
* Non-sports ("close" families): there is no event start, so the price is the
  median midpoint of the bars (at least three, spread at most 4c, range at most
  3c) in the 15 minutes ending 6 hours and 1 hour before the market's close.
  Daily temperature markets close at 1am local time the next day, long after
  the reading is known, so they use 18h and 12h (highs: about 7am and 1pm
  local) and 26h and 20h (lows: about 11pm the evening before and 5am).
  Every market of an event whose dearer side is 50-99c is a row, so a strike
  ladder gives several rows; intervals and tests treat the calendar day as the
  unit, because one weather or price move decides every ladder that day.

Commands

    scan      no network. Replays one-minute bars already cached under
              data/raw/cost-research-v1 by earlier scripts.
    discover  one GET /series per category; lists series not yet covered.
    download  capped, paced download for the families in
              data/processed/calibration-map/new-families.json
              ({name: {series, cap, archive}}, written by hand after
              `discover`). Recent markets go through the batch endpoint
              GET /markets/candlesticks (up to 100 tickers and 10,000 bar
              periods per request) and ask only for the first 150 minutes of
              each window (the first 15 bars are all that is needed), so a
              family of 400 events costs a few dozen requests instead of 400.
              Markets settled before 2026-09-18 have no batch endpoint: with
              "archive" set they cost one short read each. A family that is
              already on disk can be listed again to top it up. Rows built this
              way lack the fields that need the whole window (`full` is false).
    closing   the non-sports families in close-families.json, batch endpoint.
    report    band tables (all rows, chronological halves, timing-ok rows),
              the count of 90-99c cells whose interval excludes zero against
              the count expected by chance, one-sided p-values with
              Benjamini-Hochberg, a hierarchical (partial pooling) estimate of
              each family's 90-99c gap, and capacity. Writes summary.json.

Statistics. One row per event: Wilson interval on the win rate minus the mean
price, and a z-test of outcomes against their own prices (variance p(1-p)), so
a cell with no losses still gets an honest interval. Non-sports rows are
clustered by calendar day (bootstrap). Shrinkage: gap_i ~ N(theta_i, se_i),
theta_i ~ N(mu, tau), flat prior on mu, half-normal(3c) on tau, tau on a grid.
Prices of 99c and above are excluded everywhere.

Network use: public reads only, strictly one request at a time, at most one
per PACE seconds, long pause after HTTP 429 (a live trader shares the limit).
Every response is cached by research.api_cost.fetch, and rows are saved per
family, so an interrupted run resumes. No orders, no authentication.

    .venv/bin/python research/calibration_map.py scan
    .venv/bin/python research/calibration_map.py discover
    .venv/bin/python research/calibration_map.py download --minutes 9
    .venv/bin/python research/calibration_map.py closing --minutes 9
    .venv/bin/python research/calibration_map.py report
"""
import argparse
from collections import Counter, defaultdict
from datetime import date, datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import re
from statistics import mean, median
import sys
import time
from zoneinfo import ZoneInfo

import httpx
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from research.api_cost import COLLECTION, ROOT, fetch, midnight, save, timestamp
from research.near_end import WINDOW, cents

OUT = ROOT / "data/processed/calibration-map"
CUTOFF = date(2026, 9, 18)                              # earlier settlements live on /historical
LIVE_A = dict(status="settled", limit=1000, min_close_ts=midnight(date(2026, 9, 1)), max_close_ts=midnight(date(2026, 11, 23)))
LIVE_B = dict(status="settled", limit=1000, min_close_ts=midnight(CUTOFF)-7*86400, max_close_ts=midnight(date(2026, 11, 18))+7*86400)
WARMUP, PACE, SLACK = 15, 1.1, 2.0
AHEAD = 150      # minutes of each window that `download` asks for: bars exist only for active minutes, so 15 bars can take a while
BANDS = ((50, 60), (60, 70), (70, 80), (80, 90), (90, 95), (95, 99), (90, 99))
EASTERN = ZoneInfo("America/New_York")
MONTHS = {m: i+1 for i, m in enumerate("JAN FEB MAR APR MAY JUN JUL AUG SEP OCT NOV DEC".split())}
RNG = np.random.default_rng(5)

# Families whose bars earlier scripts already cached (scan), and new ones (download).
# kind: "known" = the favourite gap was already reported there, "disk" = bars on disk, "new" = downloaded here.
ON_DISK = {
    "ATP": ["KXATPMATCH"], "WTA": ["KXWTAMATCH"], "ATP Challenger": ["KXATPCHALLENGERMATCH"],
    "WTA Challenger": ["KXWTACHALLENGERMATCH"], "ITF men": ["KXITFMATCH"], "ITF women": ["KXITFWMATCH"],
    "NBA": ["KXNBAGAME"], "NHL": ["KXNHLGAME"], "NFL": ["KXNFLGAME"], "MLB": ["KXMLBGAME"],
    "College basketball men": ["KXNCAAMBGAME"], "College basketball women": ["KXNCAAWBGAME"],
    "College football": ["KXNCAAFGAME"], "WNBA": ["KXWNBAGAME"], "Euroleague": ["KXEUROLEAGUEGAME"],
    "Soccer Europe": ["KXEPLGAME", "KXLALIGAGAME", "KXSERIEAGAME", "KXBUNDESLIGAGAME", "KXLIGUE1GAME", "KXUCLGAME", "KXUELGAME"],
    "Soccer Americas": ["KXMLSGAME", "KXLIGAMXGAME", "KXBRASILEIROGAME", "KXARGPREMDIVGAME", "KXDIMAYORGAME"],
    "Soccer internationals": ["KXUEFANLGAME", "KXINTLFRIENDLYGAME", "KXCONCACAFNLGAME", "KXAFCONGAME"],
    "CS2": ["KXCS2GAME"], "League of Legends": ["KXLOLGAME"], "Valorant": ["KXVALORANTGAME"], "Dota 2": ["KXDOTA2GAME"],
    "UFC": ["KXUFCFIGHT"], "Boxing": ["KXBOXING"], "Cricket T20": ["KXT20MATCH", "KXWT20MATCH"], "Cricket ODI": ["KXODIMATCH"],
    "Table tennis": ["KXLIGAPROMATCH"], "Asian baseball": ["KXKBOGAME", "KXNPBGAME"], "eBasketball": ["KXEBASKETBALLGAME"],
}
KNOWN = {"ATP", "WTA", "ATP Challenger", "WTA Challenger", "ITF men", "ITF women", "College basketball men",
         "College football", "WNBA", "UFC"}
TENNIS = {"ATP", "WTA", "ATP Challenger", "WTA Challenger", "ITF men", "ITF women", "Tennis doubles", "Tennis team events"}
NEW_FILE, CLOSE_FILE = OUT/"new-families.json", OUT/"close-families.json"      # written by hand after `discover`

last_request = [0.0]
counts = Counter()


def cache_file(path, params):
    return COLLECTION / (hashlib.sha256(json.dumps([path, params], sort_keys=True).encode()).hexdigest() + ".json")


def get(client, path, params):
    """Cached read, or (when a client is given) one paced network read. Returns None when offline and not cached."""
    file = cache_file(path, params)
    if file.exists():
        counts["cache"] += 1
        return json.loads(file.read_text())
    if client is None:
        return None
    for pause in (45, 120, None):
        wait = last_request[0]+PACE-time.time()
        if wait > 0:
            time.sleep(wait)
        began = last_request[0] = time.time()
        try:
            counts["network"] += 1
            data = fetch(client, path, params)
            if time.time()-began > 4:      # fetch slept inside: we were throttled, cool down
                counts["throttled"] += 1
                time.sleep(15)
            return data
        except RuntimeError as exc:        # fetch has already backed off five times on 429/5xx
            counts["errors"] += 1
            if pause is None or "HTTP 4" in str(exc):
                raise
            time.sleep(pause)


def pages(client, path, key="markets", **params):
    cursor, seen, out = "", set(), []
    while True:
        data = get(client, path, dict(params, cursor=cursor) if cursor else params)
        if data is None:
            return out or None
        out += data[key]
        cursor = data.get("cursor")
        if not cursor or cursor in seen:
            return out
        seen.add(cursor)


def scheduled(event):
    """Scheduled start from tickers like KXCS2GAME-26NOV171600UNDQUIN (US Eastern), else None."""
    hit = re.search(r"-(\d{2})([A-Z]{3})(\d{2})(\d{2})(\d{2})(?=[A-Z])", event)
    if not hit or hit[2] not in MONTHS or int(hit[4]) > 23 or int(hit[5]) > 59:
        return None
    try:
        return int(datetime(2000+int(hit[1]), MONTHS[hit[2]], int(hit[3]), int(hit[4]), int(hit[5]), tzinfo=EASTERN).timestamp())
    except ValueError:
        return None


def ts(m, field):
    try:
        return int(timestamp(m[field]))
    except (KeyError, TypeError, ValueError, AttributeError):
        return None


def listing(client, series, archive=True, live=True):
    """Settled events of a series: {event: (settle, series, event, archived, markets)}. Archive rows win a tie."""
    sources = []
    if archive:
        sources.append((True, pages(client, "/historical/markets", series_ticker=series, limit=1000)))
    if live:
        rows = [pages(None, "/markets", series_ticker=series, **v) for v in (LIVE_A, LIVE_B)]
        if not any(rows) and client is not None:
            rows = [pages(client, "/markets", series_ticker=series, **LIVE_B)]
        sources += [(False, r) for r in rows]
    events = {}
    for archived, rows in sources:
        group = defaultdict(list)
        for m in rows or []:
            try:
                m["_settle"], m["_outcome"] = int(timestamp(m["settlement_ts"])), float(m["settlement_value_dollars"])
            except (KeyError, TypeError, ValueError):
                continue
            group[m["event_ticker"]].append(m)
        for event, ms in group.items():
            ms.sort(key=lambda m: m["ticker"])
            if event not in events or len(ms) > len(events[event][4]):
                events[event] = (ms[0]["_settle"], series, event, archived, ms)
    return events


def bar_requests(series, archived, m, sched):
    """The per-market reads earlier scripts may have cached for this market (extended window first)."""
    out = []
    for arch in (archived, not archived):
        path = f"/historical/markets/{m['ticker']}/candlesticks" if arch else f"/series/{series}/markets/{m['ticker']}/candlesticks"
        if sched and sched-3600 < m["_settle"]-WINDOW:
            out.append((path, dict(start_ts=max(sched-3600, m["_settle"]-48*3600), end_ts=m["_settle"], period_interval=1)))
        out.append((path, dict(start_ts=m["_settle"]-WINDOW, end_ts=m["_settle"], period_interval=1)))
    return out


def quote(bar):
    bid, ask = cents(bar, "yes_bid", "close"), cents(bar, "yes_ask", "close")
    if bid is None or ask is None or not 0 < bid <= ask < 100:
        return None
    return bid, ask


def volume(bar):
    try:
        return float(bar.get("volume_fp") or bar.get("volume") or 0)
    except (TypeError, ValueError):
        return 0.0


def pregame(bars):
    """(median midpoint, mean spread) of the warm-up bars, or (None, reason)."""
    if len(bars) < WARMUP:
        return None, "few_bars"
    mids, spreads = [], []
    for b in bars[:WARMUP]:
        q = quote(b)
        if q is None or q[1]-q[0] > 4:
            return None, "no_quote_or_wide"
        mids.append((q[0]+q[1])/2)
        spreads.append(q[1]-q[0])
    if max(mids)-min(mids) > 3:
        return None, "unstable"
    return median(mids), mean(spreads)


def fixed_time_quote(bars, least=3):
    """Non-sports price: median midpoint of the bars in the 15 minutes before the reference time. Bars exist only for
    minutes in which something changed, so at least `least` are required rather than all 15."""
    mids, spreads = [], []
    for b in bars:
        q = quote(b)
        if q is None or q[1]-q[0] > 4:
            return None, "no_quote_or_wide"
        mids.append((q[0]+q[1])/2)
        spreads.append(q[1]-q[0])
    if len(mids) < least:
        return None, "few_bars"
    if max(mids)-min(mids) > 3:
        return None, "unstable"
    return median(mids), mean(spreads)


def build_row(family, info, per_market, full):
    """One row per event from {ticker: bars}; (row, None) or (None, reason)."""
    settle, series, event, archived, ms = info
    best, why = None, None
    three_way = len(per_market) >= 3
    for m in ms:
        bars = per_market.get(m["ticker"])
        if bars is None:
            continue
        mid, spread = pregame(bars)
        if mid is None:
            why = spread
            if three_way:
                continue
            return None, why
        start = mid if three_way or mid > 50 else 100-mid
        if best is None or start > best[0]:
            best = (start, three_way or mid > 50, spread, m, bars)
    if best is None:
        return None, why or "no_bars"
    start, yes, spread, m, bars = best
    warm = bars[:WARMUP]
    row = dict(family=family, series=series, event=event, ticker=m["ticker"], settle=settle, archived=archived, three_way=three_way,
               start=start, spread=spread, payout=m["_outcome"] if yes else 1-m["_outcome"], sched=scheduled(event),
               expected=ts(m, "expected_expiration_time"), close=ts(m, "close_time"), window_open=bars[0]["end_period_ts"]-60,
               warm_end=warm[-1]["end_period_ts"], warm_volume=sum(volume(b) for b in warm),
               warm_active_minutes=sum(volume(b) > 0 for b in warm), market_volume=float(m.get("volume_fp") or m.get("volume") or 0),
               open_interest=float(warm[-1].get("open_interest_fp") or warm[-1].get("open_interest") or 0), full=full)
    if full:      # how long the price then sat still, and how much traded meanwhile (needs the whole window)
        moved, vol = None, 0.0
        for b in bars[WARMUP:]:
            q = quote(b)
            if q is not None and abs(((q[0]+q[1])/2 if yes else 100-(q[0]+q[1])/2)-start) > 5:
                moved = (b["end_period_ts"]-warm[-1]["end_period_ts"])/60
                break
            vol += volume(b)
        row.update(minutes_to_first_move=moved, volume_before_move=vol, bars=len(bars))
    return row, None


def scan(names):
    """Rows for the on-disk families, from cached per-market bars only."""
    for name in names:
        rows, cover = [], Counter()
        for series in ON_DISK[name]:
            events = listing(None, series)
            cover["listed"] += len(events)
            for info in events.values():
                sched, per_market = scheduled(info[2]), {}
                markets = info[4] if len(info[4]) >= 3 else info[4][:1]
                for m in markets:
                    for path, params in bar_requests(series, info[3], m, sched):
                        file = cache_file(path, params)
                        if file.exists():
                            data = json.loads(file.read_text()).get("candlesticks") or []
                            per_market[m["ticker"]] = sorted(data, key=lambda b: b["end_period_ts"])
                            break
                if len(per_market) < len(markets):
                    cover["not_cached"] += 1
                    continue
                if not all(per_market.values()):
                    cover["no_bars"] += 1
                    continue
                row, why = build_row(name, info, per_market, True)
                cover["clean" if row else why] += 1
                if row:
                    rows.append(row)
        save(OUT/f"rows-{slug(name)}.json", rows)
        print(f"{name:26} {dict(cover)}", flush=True)
        update_coverage(name, ON_DISK[name], cover, "disk")


def slug(name):
    return re.sub(r"[^a-z0-9]+", "-", name.lower()).strip("-")


def update_coverage(name, series, cover, kind):
    path = OUT/"coverage.json"
    data = json.loads(path.read_text()) if path.exists() else {}
    old = data.get(name, {})
    if old.get("kind", kind) != kind and kind == "new":      # an on-disk family topped up by `download`
        data[name] = dict(old, topped_up=dict(cover))
    else:
        data[name] = dict(series=series, kind=kind, coverage=dict(cover))
    save(path, data)


# ---------- network: discovery and the capped download ----------

def discover(client):
    covered = {s for v in ON_DISK.values() for s in v}
    out = {}
    for category in ("Sports", "Climate and Weather", "Financials", "Economics", "Crypto", "Companies"):
        data = get(client, "/series", dict(category=category, include_volume="true"))
        rows = [dict(ticker=s["ticker"], title=s["title"], tags=s.get("tags") or [], frequency=s.get("frequency"),
                     volume=float(s.get("volume_fp") or 0)) for s in data.get("series") or []]
        rows.sort(key=lambda s: -s["volume"])
        out[category] = [dict(s, covered=s["ticker"] in covered) for s in rows]
        print(f"\n{category}: {len(rows)} series")
        for s in rows[:400 if category == "Sports" else 60]:
            if s["volume"] < 2e6 or s["ticker"] in covered:
                continue
            print(f'  {s["ticker"]:28} {s["volume"]/1e6:9.1f}M {s["frequency"] or "":10} {s["title"][:60]:60} {",".join(s["tags"])[:40]}')
    save(OUT/"discovery.json", out)


def chunks(jobs, span):
    """Group (anchor_ts, tickers) jobs, sorted by anchor, so that tickers x minutes stays under the 10,000-period cap."""
    out, cur, n = [], [], 0
    for anchor, tickers in sorted(jobs, key=lambda j: j[0]):
        size = n+len(tickers)
        minutes = ((anchor-cur[0][0])//60 if cur else 0)+span+2
        if cur and (size > 100 or size*minutes > 9500):
            out.append(cur)
            cur, n = [], 0
        cur.append((anchor, tickers))
        n += len(tickers)
    return out+[cur] if cur else out


def batch_bars(client, group, before, after, deadline):
    """{ticker: bars} for one chunk: one request covering [first anchor - before, last anchor + after]."""
    params = dict(market_tickers=",".join(t for _, ts_ in group for t in ts_), start_ts=group[0][0]-before,
                  end_ts=group[-1][0]+after, period_interval=1)
    if time.time() > deadline and not cache_file("/markets/candlesticks", params).exists():
        return None
    try:
        data = get(client, "/markets/candlesticks", params)
    except RuntimeError as exc:
        print("   batch failed", exc, flush=True)
        return {}
    return {x["market_ticker"]: sorted(x.get("candlesticks") or [], key=lambda b: b["end_period_ts"]) for x in data.get("markets", [])}


def download(client, families, cap, deadline):
    """New sports families: listing, then the first AHEAD minutes of each event's six-hour window through the batch endpoint."""
    for name, spec in families.items():
        path = OUT/f"rows-{slug(name)}.json"
        done = {r["event"]: r for r in json.loads(path.read_text())} if path.exists() else {}
        events, cover = {}, Counter()
        for series in spec["series"]:
            if time.time() > deadline:
                break
            try:
                events.update(listing(client, series, archive=bool(spec.get("archive")), live=not spec.get("archive_only")))
            except RuntimeError as exc:
                print("  listing failed", series, exc, flush=True)
        recent = sorted(events.values(), key=lambda e: (-e[0], e[2]))[:spec.get("cap", cap)]
        cover["listed"], cover["taken"] = len(events), len(recent)
        for e in recent:      # archived events: no batch endpoint, one short read per market
            if not e[3] or e[2] in done:
                continue
            per_market = {}
            for m in (e[4] if len(e[4]) >= 3 else e[4][:1]):
                req = (f"/historical/markets/{m['ticker']}/candlesticks", dict(start_ts=e[0]-WINDOW, end_ts=e[0]-WINDOW+AHEAD*60, period_interval=1))
                if time.time() > deadline and not cache_file(*req).exists():
                    break
                try:
                    data = get(client, *req)
                except RuntimeError:
                    data = {}
                per_market[m["ticker"]] = sorted(data.get("candlesticks") or [], key=lambda b: b["end_period_ts"])
            else:
                if not all(per_market.values()):
                    cover["no_bars"] += 1
                    continue
                row, why = build_row(name, e, per_market, False)
                cover["clean" if row else why] += 1
                if row:
                    done[e[2]] = row
                continue
            cover["cut_by_deadline"] += 1
        todo = {e[2]: e for e in recent if e[2] not in done and not e[3]}
        jobs = [(e[0]-WINDOW, [m["ticker"] for m in (e[4] if len(e[4]) >= 3 else e[4][:1])]) for e in todo.values()]
        by_anchor = {(e[0]-WINDOW, e[4][0]["ticker"]): e for e in todo.values()}
        for group in chunks(jobs, AHEAD):
            got = batch_bars(client, group, 0, AHEAD*60, deadline)
            if got is None:
                cover["cut_by_deadline"] += len(group)
                continue
            for anchor, tickers in group:
                info = by_anchor[anchor, tickers[0]]
                per_market = {t: [b for b in got.get(t, []) if anchor <= b["end_period_ts"] <= anchor+AHEAD*60] for t in tickers}
                if not all(per_market.values()):
                    cover["no_bars"] += 1
                    continue
                row, why = build_row(name, info, per_market, False)
                cover["clean" if row else why] += 1
                if row:
                    done[info[2]] = row
        save(path, list(done.values()))
        update_coverage(name, spec["series"], cover, "new")
        print(f"{name:26} {dict(cover)} rows {len(done)} requests {dict(counts)}", flush=True)


def closing(client, families, cap, deadline, offsets=(6, 1)):
    """Non-sports: every market of an event, priced over the 15 bars ending 6h and 1h before the scheduled close."""
    for name, spec in families.items():
        path = OUT/f"close-rows-{slug(name)}.json"
        done = defaultdict(list)
        for r in (json.loads(path.read_text()) if path.exists() else []):
            done[r["event"], r["hours"]].append(r)
        events, cover = {}, Counter()
        for series in spec["series"]:
            if time.time() > deadline:
                break
            try:
                events.update(listing(client, series, archive=False))
            except RuntimeError as exc:
                print("  listing failed", series, exc, flush=True)
        recent = sorted(events.values(), key=lambda e: (-e[0], e[2]))[:spec.get("cap", cap)]
        cover["listed"], cover["taken"] = len(events), len(recent)
        for hours in spec.get("offsets", offsets):
            jobs, lookup = [], {}
            for e in recent:
                close = ts(e[4][0], "close_time")
                if close is None or (e[2], hours) in done or e[3]:
                    continue
                ms = [m for m in e[4] if ts(m, "close_time") == close][:spec.get("max_markets", 60)]
                anchor = close-hours*3600
                jobs.append((anchor, [m["ticker"] for m in ms]))
                lookup[anchor, ms[0]["ticker"]] = (e, {m["ticker"]: m for m in ms})
            for group in chunks(jobs, 16):
                got = batch_bars(client, group, 16*60, 0, deadline)
                if got is None:
                    cover["cut_by_deadline"] += len(group)
                    continue
                for anchor, tickers in group:
                    e, ms = lookup[anchor, tickers[0]]
                    done[e[2], hours] = []      # marks the event as read even when no market qualifies
                    for t in tickers:
                        bars = [b for b in got.get(t, []) if anchor-15*60 < b["end_period_ts"] <= anchor]
                        mid, spread = fixed_time_quote(bars)
                        if mid is None:
                            cover[f"{hours}h {spread}"] += 1
                            continue
                        yes, m = mid > 50, ms[t]
                        done[e[2], hours].append(dict(
                            family=name, series=e[1], event=e[2], ticker=t, hours=hours, settle=e[0], close=anchor+hours*3600,
                            start=mid if yes else 100-mid, spread=spread, payout=m["_outcome"] if yes else 1-m["_outcome"],
                            warm_volume=sum(volume(b) for b in bars[-WARMUP:]), market_volume=float(m.get("volume_fp") or 0),
                            markets_in_event=len(tickers)))
                        cover[f"{hours}h clean"] += 1
        save(path, [r for rows in done.values() for r in rows])
        update_coverage(name, spec["series"], cover, "close")
        print(f"{name:26} {dict(cover)} requests {dict(counts)}", flush=True)


# ---------- report ----------

def timing_ok(r):
    if r["family"] in TENNIS:      # order-of-play sports: expected_expiration is a placeholder; favourite_gap.py detected the
        return True                # start of play after the warm-up in every 90c+ match, so these rows are taken as pre-match
    if r.get("sched"):
        return r["warm_end"] <= r["sched"]
    if r.get("expected"):
        return r["settle"] <= r["expected"]+SLACK*3600
    return False


def wilson(wins, n, z=1.96):
    p = wins/n
    centre, half = (p+z*z/(2*n))/(1+z*z/n), z*math.sqrt(p*(1-p)/n+z*z/(4*n*n))/(1+z*z/n)
    return centre-half, centre+half


def clusters(rows, unit="event"):
    """Per-unit (sum of payout, sum of price/100, sum of p(1-p), n). The unit is the event for sports and the calendar
    day of the close for non-sports, where every city or strike ladder on a day shares the same weather or market move."""
    by = defaultdict(lambda: [0.0, 0.0, 0.0, 0])
    for r in rows:
        c = by[r["event"] if unit == "event" else iso(r["close"])]
        p = r["start"]/100
        c[0] += r["payout"]; c[1] += p; c[2] += p*(1-p); c[3] += 1
    return np.array(list(by.values()))


def cell(rows, unit="event"):
    """Calibration of a set of rows. Interval: Wilson on the win rate minus the mean price (one market per event),
    or a bootstrap over units when a unit carries several markets. p-value: outcomes against their own prices."""
    n = len(rows)
    if not n:
        return dict(n=0)
    price, win = mean(r["start"] for r in rows), mean(r["payout"] for r in rows)
    spread = mean(r["spread"] for r in rows)
    nets = [100*r["payout"]-(r["start"]+r["spread"]/2)-7*((r["start"]+r["spread"]/2)/100)*(1-(r["start"]+r["spread"]/2)/100) for r in rows]
    c = clusters(rows, unit)
    if len(c) == n:
        lo, hi = wilson(win*n, n)
        ci = [100*lo-price, 100*hi-price]
        se0 = 100*math.sqrt(c[:, 2].sum())/n
    else:
        pick = RNG.integers(0, len(c), (2000, len(c)))
        boot = 100*(c[pick, 0]-c[pick, 1]).sum(1)/c[pick, 3].sum(1)
        ci = [float(v) for v in np.percentile(boot, [2.5, 97.5])]
        se0 = 100*math.sqrt(((c[:, 0]-c[:, 1])**2).sum())/n      # clustered, robust
    gap = 100*win-price
    z = gap/se0 if se0 else 0.0
    return dict(n=n, events=len(c), price=round(price, 2), win=round(100*win, 2), gap=round(gap, 2), ci=[round(ci[0], 2), round(ci[1], 2)],
                se=round(se0, 3), z=round(z, 2), p_two_sided=float(math.erfc(abs(z)/math.sqrt(2))), spread=round(spread, 2),
                net_at_ask=round(mean(nets), 2), losses=sum(r["payout"] < .5 for r in rows))


def chance_of_exclusion(rows, sims=4000):
    """If prices were exactly right, how often would this cell's interval exclude zero (either side, and above)?"""
    p = np.array([r["start"]/100 for r in rows])
    n = len(p)
    wins = (RNG.random((sims, n)) < p).sum(1)
    centre = (wins/n+1.96**2/(2*n))/(1+1.96**2/n)
    half = 1.96*np.sqrt(wins/n*(1-wins/n)/n+1.96**2/(4*n*n))/(1+1.96**2/n)
    above, below = centre-half > p.mean(), centre+half < p.mean()
    return float(above.mean()), float(below.mean())


def halves(rows):
    rows = sorted(rows, key=lambda r: (r["settle"], r["event"]))
    cut = len(rows)//2
    return rows[:cut], rows[cut:]


def in_band(rows, lo, hi):
    return [r for r in rows if lo <= r["start"] < hi]


def shrink(cells, prior_sd=3.0):
    """Normal-normal hierarchical estimate of each family's gap (cents). cells: {name: (gap, se)}.
    gap_i ~ N(theta_i, se_i), theta_i ~ N(mu, tau); flat prior on mu, half-normal(prior_sd) on tau, tau on a grid."""
    names = list(cells)
    g, s = np.array([cells[k][0] for k in names]), np.array([cells[k][1] for k in names])
    taus = np.linspace(0, 8, 321)
    logp, means, variances, mus = [], [], [], []
    for tau in taus:
        w = 1/(s**2+tau**2)
        mu, v = (w*g).sum()/w.sum(), 1/w.sum()
        logp.append(.5*np.log(w).sum()-.5*(w*(g-mu)**2).sum()+.5*math.log(v)-.5*(tau/prior_sd)**2)
        b = tau**2/(tau**2+s**2)
        means.append(mu+b*(g-mu)); variances.append(b*s**2+(1-b)**2*v); mus.append((mu, v))
    p = np.exp(np.array(logp)-max(logp)); p /= p.sum()
    means, variances, mus = np.array(means), np.array(variances), np.array(mus)
    post = (p[:, None]*means).sum(0)
    sd = np.sqrt((p[:, None]*(variances+means**2)).sum(0)-post**2)
    mu = float((p*mus[:, 0]).sum())
    mu_sd = math.sqrt(float((p*(mus[:, 1]+mus[:, 0]**2)).sum())-mu**2)
    w = 1/s**2
    q = float((w*(g-(w*g).sum()/w.sum())**2).sum())
    cdf = np.cumsum(p)
    out = {k: dict(shrunk=round(float(m), 2), sd=round(float(d), 2), prob_positive=round(float(.5*math.erfc(-m/d/math.sqrt(2))), 3))
           for k, m, d in zip(names, post, sd)}
    return dict(overall_mean=round(mu, 2), overall_se=round(mu_sd, 2), between_family_sd=round(float((p*taus).sum()), 2),
                between_family_sd_95=[round(float(taus[np.searchsorted(cdf, .025)]), 2), round(float(taus[min(np.searchsorted(cdf, .975), len(taus)-1)]), 2)],
                q=round(q, 1), df=len(g)-1, families=out)


def capacity(rows):
    if not rows:
        return None
    span = max((max(r["settle"] for r in rows)-min(r["settle"] for r in rows))/86400, 1)
    out = dict(events=len(rows), days=round(span, 1), per_month=round(30.4*len(rows)/span, 1),
               median_spread=median(r["spread"] for r in rows), share_spread_1c=round(mean(r["spread"] <= 1.01 for r in rows), 2),
               median_contracts_in_15_warmup_minutes=round(median(r["warm_volume"] for r in rows)),
               mean_contracts_in_15_warmup_minutes=round(mean(r["warm_volume"] for r in rows)),
               share_with_any_warmup_trade=round(mean(r["warm_volume"] > 0 for r in rows), 2),
               median_market_volume=round(median(r["market_volume"] for r in rows)))
    full = [r for r in rows if r.get("full")]
    if full:
        still = [r["minutes_to_first_move"] for r in full if r["minutes_to_first_move"] is not None]
        out.update(median_contracts_before_first_5c_move=round(median(r["volume_before_move"] for r in full)),
                   median_minutes_to_first_5c_move=median(still) if still else None,
                   share_never_moved_5c=round(1-len(still)/len(full), 2))
    return out


def report():
    cover = json.loads((OUT/"coverage.json").read_text())
    fams, close = {}, {}
    for name, c in cover.items():
        file = OUT/(("close-rows-" if c["kind"] == "close" else "rows-")+slug(name)+".json")
        rows = json.loads(file.read_text()) if file.exists() else []
        if c["kind"] == "close":
            for hours in sorted({r["hours"] for r in rows}):
                close[f"{name} @{hours}h"] = [r for r in rows if r["hours"] == hours and r["start"] < 99]
        else:
            fams[name] = [r for r in rows if r["start"] < 99]
    out = dict(families={}, close={})
    print("\nCALIBRATION BY BAND (all rows | first half | second half | timing-ok rows)")
    for group, target in ((fams, out["families"]), (close, out["close"])):
        for name, rows in group.items():
            ok = [r for r in rows if timing_ok(r)] if group is fams else rows
            entry = dict(rows=len(rows), timing_ok_share=round(len(ok)/len(rows), 3) if rows else None,
                         first=iso(min(r["settle"] for r in rows)) if rows else None, last=iso(max(r["settle"] for r in rows)) if rows else None, bands={})
            a, b = halves(rows)
            if rows:
                entry["split_at"] = iso(b[0]["settle"])
            print(f'\n{name}: {len(rows)} rows {entry["first"]}..{entry["last"]} timing ok {entry["timing_ok_share"]}')
            for lo, hi in BANDS:
                oa, ob = halves(ok)
                unit = "event" if group is fams else "day"
                parts = {k: cell(in_band(v, lo, hi), unit) for k, v in (("all", rows), ("first_half", a), ("second_half", b), ("timing_ok", ok),
                                                                       ("timing_ok_first_half", oa), ("timing_ok_second_half", ob))}
                if parts["all"]["n"]:
                    parts["chance_interval_above_zero"], parts["chance_interval_below_zero"] = chance_of_exclusion(in_band(rows, lo, hi))
                entry["bands"][f"{lo}-{hi}"] = parts
                c = parts["all"]
                if c["n"]:
                    def short(x):
                        return f'{x["n"]:4} {x["gap"]:+5.1f}' if x["n"] else "   0      "
                    print(f'  {lo}-{hi}: n {c["n"]:5} price {c["price"]:5.1f} win {c["win"]:5.1f} gap {c["gap"]:+5.1f} ci [{c["ci"][0]:+5.1f},{c["ci"][1]:+5.1f}] '
                          f'spread {c["spread"]:.1f} net@ask {c["net_at_ask"]:+5.1f} | {short(parts["first_half"])} | {short(parts["second_half"])} | {short(parts["timing_ok"])}')
            if group is fams:
                entry["capacity_90_99"] = capacity(in_band(ok, 90, 99))
                entry["capacity_90_99_all_rows"] = capacity(in_band(rows, 90, 99))
            else:
                top = in_band(rows, 90, 99)
                days = max((max(r["settle"] for r in top)-min(r["settle"] for r in top))/86400, 1) if top else 1
                entry["capacity_90_99"] = dict(
                    rows=len(top), events=len({r["event"] for r in top}), days=len({iso(r["close"]) for r in top}), rows_per_month=round(30.4*len(top)/days),
                    median_spread=median(r["spread"] for r in top), median_contracts_in_15_minutes=round(median(r["warm_volume"] for r in top)),
                    mean_contracts_in_15_minutes=round(mean(r["warm_volume"] for r in top)), share_with_any_trade=round(mean(r["warm_volume"] > 0 for r in top), 2),
                    median_market_volume=round(median(r["market_volume"] for r in top)), losses=sum(r["payout"] < .5 for r in top),
                    worst_day_losses=max(Counter(iso(r["close"]) for r in top if r["payout"] < .5).values(), default=0)) if top else None
            target[name] = entry

    # Multiple comparisons on the 90-99c cell, and shrinkage. Sports families with at least 10 rows there.
    for label, key in (("all rows", "all"), ("timing-ok rows", "timing_ok")):
        cells = {k: v["bands"]["90-99"] for k, v in out["families"].items() if v["bands"]["90-99"][key]["n"] >= 10}
        above = [k for k, v in cells.items() if v[key]["ci"][0] > 0]
        below = [k for k, v in cells.items() if v[key]["ci"][1] < 0]
        expected = sum(v["chance_interval_above_zero"] for v in cells.values())
        model = shrink({k: (v[key]["gap"], v[key]["se"]) for k, v in cells.items()})
        out[f"multiple_comparisons_{key}"] = dict(cells=len(cells), interval_above_zero=above, interval_below_zero=below,
                                                  expected_above_by_chance=round(expected, 2),
                                                  expected_below_by_chance=round(sum(v["chance_interval_below_zero"] for v in cells.values()), 2))
        # One-sided p-values (favourite wins more than its price), Benjamini-Hochberg at 10%, and the pooled test.
        ps = sorted((.5*math.erfc(v[key]["z"]/math.sqrt(2)), k) for k, v in cells.items())
        passed = max([i+1 for i, (pv, _) in enumerate(ps) if pv <= .1*(i+1)/len(ps)], default=0)
        n_all = sum(v[key]["n"] for v in cells.values())
        pooled_gap = sum(v[key]["gap"]*v[key]["n"] for v in cells.values())/n_all
        pooled_se = math.sqrt(sum((v[key]["se"]*v[key]["n"])**2 for v in cells.values()))/n_all
        out[f"multiple_comparisons_{key}"].update(
            p_below_5pct=[k for pv, k in ps if pv < .05], expected_p_below_5pct=round(.05*len(ps), 2),
            benjamini_hochberg_10pct=[k for _, k in ps[:passed]], smallest_p=[(k, round(pv, 4)) for pv, k in ps[:5]],
            pooled=dict(n=n_all, gap=round(pooled_gap, 2), se=round(pooled_se, 2), z=round(pooled_gap/pooled_se, 2)))
        print(f"  one-sided p<5% in {[k for pv, k in ps if pv < .05]} (chance: {.05*len(ps):.1f} cells); Benjamini-Hochberg 10% keeps {[k for _, k in ps[:passed]]}; "
              f"smallest p {[(k, round(pv, 4)) for pv, k in ps[:4]]}; pooled {n_all} favourites gap {pooled_gap:+.2f}c (se {pooled_se:.2f})")
        out[f"shrinkage_{key}"] = model
        print(f"\n90-99c FAVOURITES, {label}: {len(cells)} families with 10+ rows; interval above zero in {len(above)} {above}, "
              f"below zero in {len(below)} {below}; expected above by chance {expected:.2f}")
        print(f'  overall mean gap {model["overall_mean"]:+.2f}c (se {model["overall_se"]}), between-family sd {model["between_family_sd"]}c {model["between_family_sd_95"]}, Q {model["q"]} on {model["df"]} df')
        print(f'  {"family":26} {"n":>5} {"price":>6} {"win":>6} {"raw":>6} {"interval":>15} {"shrunk":>7} {"sd":>5} {"P>0":>5} {"half 1":>11} {"half 2":>11} {"net@ask":>8} {"timing":>6}')
        for k in sorted(cells, key=lambda k: -model["families"][k]["shrunk"]):
            c, f, v = cells[k][key], model["families"][k], cells[k]
            h1, h2 = (v["first_half"], v["second_half"]) if key == "all" else (v["timing_ok_first_half"], v["timing_ok_second_half"])
            print(f'  {k:26} {c["n"]:5} {c["price"]:6.1f} {c["win"]:6.1f} {c["gap"]:+6.1f} [{c["ci"][0]:+6.1f},{c["ci"][1]:+6.1f}] {f["shrunk"]:+7.2f} {f["sd"]:5.2f} {f["prob_positive"]:5.2f} '
                  f'{(str(h1["n"])+" "+format(h1["gap"], "+.1f")) if h1["n"] else "-":>11} {(str(h2["n"])+" "+format(h2["gap"], "+.1f")) if h2["n"] else "-":>11} {c["net_at_ask"]:+8.2f} {out["families"][k]["timing_ok_share"]:6.2f}')
    # Every band x family cell, for the count of "significant" cells against chance.
    tally = Counter()
    for v in out["families"].values():
        for band, parts in v["bands"].items():
            if band != "90-99" and parts["all"]["n"] >= 10:
                tally["cells"] += 1
                tally["above"] += parts["all"]["ci"][0] > 0
                tally["below"] += parts["all"]["ci"][1] < 0
                tally["expected_above"] += parts["chance_interval_above_zero"]
                tally["expected_below"] += parts["chance_interval_below_zero"]
    out["all_band_cells"] = {k: round(v, 2) for k, v in tally.items()}
    print("\nAll family x band cells (six bands):", out["all_band_cells"])
    print("\nCAPACITY, 90-99c favourites (timing-ok rows)")
    for k, v in out["families"].items():
        c = v.get("capacity_90_99")
        if c and c["events"] >= 10:
            print(f'  {k:26} {c["events"]:4} events, {c["per_month"]:6.1f}/month, spread median {c["median_spread"]:.1f}c ({c["share_spread_1c"]:.0%} at 1c), '
                  f'15-min warm-up contracts median {c["median_contracts_in_15_warmup_minutes"]} mean {c["mean_contracts_in_15_warmup_minutes"]}, '
                  f'before first 5c move {c.get("median_contracts_before_first_5c_move")}, market volume {c["median_market_volume"]}')
    save(OUT/"summary.json", out)


def iso(t):
    return datetime.fromtimestamp(t, timezone.utc).isoformat()[:10]


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("cmd", choices=("scan", "discover", "download", "closing", "report"))
    parser.add_argument("--families", nargs="+")
    parser.add_argument("--minutes", type=float, default=20, help="stop starting new requests after this long; a rerun resumes from the cache")
    parser.add_argument("--cap", type=int, default=400, help="most recent events per new family")
    args = parser.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    if args.cmd == "scan":
        return scan(args.families or list(ON_DISK))
    if args.cmd == "report":
        return report()
    deadline = time.time()+60*args.minutes
    with httpx.Client(timeout=40) as client:
        if args.cmd == "discover":
            return discover(client)
        spec = json.loads((NEW_FILE if args.cmd == "download" else CLOSE_FILE).read_text())
        spec = {k: v for k, v in spec.items() if not args.families or k in args.families}
        (download if args.cmd == "download" else closing)(client, spec, args.cap, deadline)


if __name__ == "__main__":
    main()
