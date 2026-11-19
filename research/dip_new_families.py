"""The favourite dip rule in families where it was never run on a full sample.

The rule (research/favorite_dip.py): a competitor is a heavy favourite before
the event (90c+); rest a bid below that price once the event starts; a fill
counts only when a trade prints strictly below the bid in a later one-minute
bar; hold every fill to settlement. research/calibration_map.py found families
whose 90c+ favourites looked underpriced before the event (NBA, soccer
internationals, T20 cricket, basketball outside the NBA, boxing) but the dip
rule itself had been run only on parts of them. This script runs it on the
largest sample it can reach inside a download budget.

Definitions

* One canonical market per two-outcome event (the lowest ticker); the
  favourite is the side whose pre-event midpoint is above 50c. Three-way
  events (soccer) use the YES side of the team market with the highest
  pre-event midpoint; the tie market is never the favourite.
* Scheduled start: the time in the event ticker where it carries one
  (cricket, most basketball outside the US), otherwise Kalshi's
  `expected_expiration_time` minus three hours (checked against a play-by-play
  archive for the NBA: exact for 98% of games). A game whose settlement came
  less than 45 minutes or more than 8 hours after a start estimated that way is
  dropped as "odd schedule" (postponed games, settlements days late). Boxing,
  UFC and other MMA have no usable schedule: their rows are reported
  separately and flagged, with the order resting from the end of the warm-up.
* Window: the six hours before settlement; opened earlier (one hour before the
  scheduled start, at most 48 hours before settlement) when six hours would
  not reach back that far, as in research/dip_other_sports.py.
* Pre-event price: median midpoint of the first 15 one-minute bars of the
  window, each two-sided with a spread of at most 4c, all within 3c, and the
  15th bar must end before the scheduled start.
* The order rests from the scheduled start (or the end of the warm-up if that
  is later) to settlement. Strict fill: a trade printed strictly below the bid
  in a bar that began at or after that moment. Touch: a print at the bid also
  counts. Prints below the bid between the warm-up and the start are counted
  separately ("before start") and are not fills.
* Bids: 85, 80, 75, 70c and 10c below the pre-event price (rounded down), each
  only when at least 5c below the pre-event price. Bands: 90c+ and, as a
  control, 85-90c. NBA also 70-80 and 80-90.
* The order pays its limit. Maker fee: nothing on "quadratic" series,
  0.0175*p*(1-p) (unconfirmed rate) on "quadratic_with_maker_fees" series; both
  the gross and the fee-adjusted figure are kept, tables show the latter.
* Intervals: 95% bootstrap resampling events. Because a cell with no losing
  fill has a degenerate bootstrap, a Wilson interval on the win rate (minus the
  price needed) is printed beside it, and the pass test (first half above
  zero, second half positive, neighbouring bids positive) uses the Wilson one. Halves: the family's orders in that band
  sorted by settlement time and cut in two.

Commands

    listing   settled-event listings (and fee types) for every family
    plan      what is cached and what each download stage would cost
    download  --stage recent   batch endpoint, all families pooled
              --stage archive  one read per market, families in priority order
    build     replay the cache into one row per event. No network.
    trades    public trade record around a sample of fills (capacity)
    report    tables, NBA reconciliation, checks, multiplicity. No network.

Network: public reads only, strictly one request at a time, at most one per
PACE seconds, long pauses after HTTP 429 (a live trader shares the limit).
Everything goes into the shared cache data/raw/cost-research-v1.

    .venv/bin/python research/dip_new_families.py listing --minutes 9
    .venv/bin/python research/dip_new_families.py download --stage recent --minutes 9
    .venv/bin/python research/dip_new_families.py download --stage archive --minutes 9
    .venv/bin/python research/dip_new_families.py build
    .venv/bin/python research/dip_new_families.py trades --minutes 9
    .venv/bin/python research/dip_new_families.py report
"""
import argparse
from collections import Counter, defaultdict
from datetime import date, datetime, timezone
import json
import math
from pathlib import Path
import sys
import time

import httpx
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import config
from research.api_cost import ROOT, midnight, save, timestamp
from research.calibration_map import cache_file, listing, scheduled, ts
from research.near_end import WINDOW, cents

OUT = ROOT / "data/processed/dip-new-families"
CAL = ROOT / "data/processed/calibration-map"
CUTOFF = midnight(date(2026, 9, 18))
WARMUP, PACE = 15, 1.5
LEVELS = (85, 80, 75, 70)
BANDS = {"90+": (90, 101), "85-90": (85, 90), "80-90": (80, 90), "70-80": (70, 80), "55-70": (55, 70)}
RNG = np.random.default_rng(7)
last_request = [0.0]
counts = Counter()

# timing: "start" = scheduled start known (ticker or expected-3h); "none" = no usable schedule.
# archive: also read events settled before 2026-09-18 (one request per market). cache_only: never download per-market archive bars.
FAMILIES = {
    "NBA": dict(series=["KXNBAGAME"], archive=True, screen=68),
    "Soccer internationals": dict(series=["KXUEFANLGAME", "KXINTLFRIENDLYGAME", "KXCONCACAFNLGAME", "KXAFCONGAME", "KXWCGAME",
                                          "KXFIFAGAME", "KXFIFAWGAME", "KXASEANGAME", "KXAFCACGAME"], archive=True),
    "Cricket T20": dict(series=["KXT20MATCH", "KXWT20MATCH", "KXIPLGAME", "KXCPLMATCH", "KXHUNDREDMATCH", "KXWHUNDREDMATCH",
                                "KXPSLGAME", "KXLPLMATCH", "KXCRICKETT20IMATCH"], archive=True),
    "Basketball Asia-Pacific": dict(series=["KXCBAGAME", "KXNBLGAME", "KXJBLEAGUEGAME", "KXKBLGAME", "KXPBAGAME", "KXNZNBLGAME"], archive=True),
    "Basketball Europe": dict(series=["KXEUROLEAGUEGAME", "KXEUROCUPGAME", "KXBBLGAME", "KXACBGAME", "KXBSLGAME", "KXLNBELITEGAME",
                                      "KXFIBACHAMPLEAGUEGAME", "KXISLGAME", "KXABAGAME", "KXVTBGAME", "KXBBSERIEAGAME", "KXGBLGAME",
                                      "KXFIBAECUPGAME"], archive=True),
    "Basketball Americas other": dict(series=["KXLNBPGAME", "KXNBBGAME", "KXBSNGAME", "KXARGLNBGAME", "KXFIBAGAME", "KXWFIBAGAME",
                                              "KXNBASUMMERGAME"], archive=True),
    "WNBA": dict(series=["KXWNBAGAME"], archive=True),
    "College basketball women": dict(series=["KXNCAAWBGAME"], archive=True),
    "Cricket ODI": dict(series=["KXODIMATCH", "KXWODIMATCH", "KXCRICKETODIMATCH"], archive=True),
    "Soccer Europe top": dict(series=["KXEPLGAME", "KXLALIGAGAME", "KXSERIEAGAME", "KXBUNDESLIGAGAME", "KXLIGUE1GAME", "KXUCLGAME",
                                      "KXUELGAME"], archive=True, cache_only=True),
    "Soccer Americas": dict(series=["KXMLSGAME", "KXLIGAMXGAME", "KXBRASILEIROGAME", "KXARGPREMDIVGAME", "KXDIMAYORGAME"]),
    "Soccer other leagues": dict(series=["KXSAUDIPLGAME", "KXECULPGAME", "KXUECLGAME", "KXEFLCHAMPIONSHIPGAME", "KXBRASILEIROBGAME",
                                         "KXLIGAPORTUGALGAME", "KXJLEAGUEGAME", "KXLALIGA2GAME", "KXSUPERLIGGAME", "KXUCLWGAME",
                                         "KXEREDIVISIEGAME", "KXCONMEBOLLIBGAME", "KXNWSLGAME", "KXSCOTTISHPREMGAME", "KXBELGIANPLGAME",
                                         "KXEFLCUPGAME"]),
    "NHL": dict(series=["KXNHLGAME"], archive=True, cache_only=True),
    "NFL": dict(series=["KXNFLGAME"], archive=True, cache_only=True),
    "MLB": dict(series=["KXMLBGAME"], archive=True, cache_only=True),
    "Boxing": dict(series=["KXBOXING"], archive=True, timing="none", cache_only=True),
    "UFC": dict(series=["KXUFCFIGHT"], archive=True, timing="none", cache_only=True),
    "MMA other": dict(series=["KXMMAFIGHT"], timing="none"),
}
ARCHIVE_ORDER = ["NBA", "Soccer internationals", "Cricket T20", "Basketball Asia-Pacific", "Basketball Europe", "WNBA",
                 "Basketball Americas other", "Cricket ODI", "College basketball women"]
MONTHS = {m: i+1 for i, m in enumerate("JAN FEB MAR APR MAY JUN JUL AUG SEP OCT NOV DEC".split())}


# ---------------------------------------------------------------- network

def read(client, path, params, compact=True):
    """Cached read, else (with a client) one paced public read. None when offline and not cached. Raises on HTTP 4xx."""
    file = cache_file(path, params)
    if file.exists():
        counts["cache"] += 1
        try:
            return json.loads(file.read_text())
        except ValueError:
            return None
    if client is None:
        return None
    for pause in (20, 45, 90, 180):
        wait = last_request[0]+PACE-time.time()
        if wait > 0:
            time.sleep(wait)
        last_request[0] = time.time()
        counts["network"] += 1
        try:
            resp = client.get(config.KALSHI_PROD_BASE+path, params=params)
        except httpx.RequestError:
            resp = None
        if resp is not None and resp.status_code == 200:
            data = resp.json()
            temp = file.with_suffix(".tmp")
            temp.write_text(json.dumps(data) if compact else json.dumps(data, indent=2))
            temp.replace(file)
            return data
        if resp is not None and resp.status_code != 429 and resp.status_code < 500:
            counts["http4xx"] += 1
            raise RuntimeError(f"{path} returned HTTP {resp.status_code}")
        counts["throttled"] += 1
        time.sleep(pause)
    raise RuntimeError(f"Read failed after long back-off: {path}")


def events_of(client, name):
    """Settled events of a family: (settle, series, event, archived, markets), oldest first. Cached listings only when client is None."""
    spec, out = FAMILIES[name], {}
    for series in spec["series"]:
        try:
            got = listing(client, series, archive=bool(spec.get("archive")))
        except RuntimeError as exc:
            print("  listing failed", series, exc, flush=True)
            continue
        for event, info in got.items():
            if len(info[4]) >= 2:
                out[event] = info
    return sorted(out.values(), key=lambda e: (e[0], e[2]))


def start_of(spec, info):
    """(scheduled start, kind) or (None, reason)."""
    if spec.get("timing") == "none":
        return None, "none"
    settle, _, event, _, ms = info
    sched = scheduled(event)
    if sched:
        return sched, "ticker"
    expected = ts(ms[0], "expected_expiration_time")
    if not expected:
        return None, "odd"
    est = expected-3*3600
    if not 2700 <= settle-est <= 8*3600:
        return None, "odd"
    return est, "expected"


def window(info, est):
    settle = info[0]
    open_ts = settle-WINDOW
    if est and est-3600 < open_ts:
        open_ts = max(est-3600, settle-48*3600)
    return open_ts, settle


def is_tie(m):
    return m["ticker"].rsplit("-", 1)[-1] in ("TIE", "DRAW") or str(m.get("yes_sub_title", "")).lower() in ("tie", "draw")


def candidates(info):
    """Markets that can hold the favourite."""
    ms = info[4]
    return [m for m in ms if not is_tie(m)] if len(ms) >= 3 else ms[:1]


def market_requests(series, archived, ticker, open_ts, settle):
    """Per-market reads that this or an earlier script may have cached, the wanted one first."""
    out = []
    for arch in (archived, not archived):
        path = f"/historical/markets/{ticker}/candlesticks" if arch else f"/series/{series}/markets/{ticker}/candlesticks"
        out.append((path, dict(start_ts=open_ts, end_ts=settle, period_interval=1)))
        if open_ts != settle-WINDOW:
            out.append((path, dict(start_ts=settle-WINDOW, end_ts=settle, period_interval=1)))
    return out


class Bars:
    """One-minute bars of a market from the cache: a per-market read, or one of this script's batch reads."""

    def __init__(self):
        path = OUT/"batch-index.json"
        self.index = json.loads(path.read_text()) if path.exists() else {}       # ticker -> batch params
        self.loaded = {}

    def note(self, params):
        for t in params["market_tickers"].split(","):
            self.index[t] = params

    def flush(self):
        save(OUT/"batch-index.json", self.index)

    def get(self, series, archived, ticker, open_ts, settle):
        reqs = market_requests(series, archived, ticker, open_ts, settle)
        for i, (path, params) in enumerate(reqs):
            file = cache_file(path, params)
            if file.exists():
                if params["start_ts"] > open_ts:         # a cached six-hour read where a longer window is wanted
                    continue
                try:
                    data = json.loads(file.read_text()).get("candlesticks") or []
                except ValueError:
                    continue
                return sorted((b for b in data if open_ts <= b["end_period_ts"] <= settle), key=lambda b: b["end_period_ts"])
        params = self.index.get(ticker)
        if params and params["start_ts"] <= open_ts and params["end_ts"] >= settle:
            key = json.dumps(params, sort_keys=True)
            if key not in self.loaded:
                if len(self.loaded) > 12:
                    self.loaded.clear()
                file = cache_file("/markets/candlesticks", params)
                data = json.loads(file.read_text()) if file.exists() else {}
                self.loaded[key] = {x["market_ticker"]: x.get("candlesticks") or [] for x in data.get("markets", [])}
            data = self.loaded[key].get(ticker, [])
            return sorted((b for b in data if open_ts <= b["end_period_ts"] <= settle), key=lambda b: b["end_period_ts"])
        return None


def screen():
    """Pre-event prices already measured by calibration_map.py, used only to skip archive reads of games with no favourite."""
    out = {}
    path = CAL/"rows-nba.json"
    if path.exists():
        out["NBA"] = {r["event"]: r["start"] for r in json.loads(path.read_text())}
    return out


def todo(name, bars, known):
    """Events of a family whose bars are not cached: (recent jobs, archive jobs). A job is (info, open, settle, tickers)."""
    spec = FAMILIES[name]
    recent, archive, have = [], [], 0
    for info in events_of(None, name):
        est, kind = start_of(spec, info)
        if kind == "odd":
            continue
        open_ts, settle = window(info, est)
        need = [m["ticker"] for m in candidates(info) if bars.get(info[1], info[3], m["ticker"], open_ts, settle) is None]
        if not need:
            have += 1
            continue
        if spec.get("screen") and known.get(name, {}).get(info[2], 100) < spec["screen"]:
            continue
        if info[0] >= CUTOFF and not info[3]:
            recent.append((info, open_ts, settle, need))
        elif not spec.get("cache_only"):
            archive.append((info, open_ts, settle, need))
    return recent, archive, have


def chunks(jobs):
    """Group jobs sorted by window open so that tickers x minutes of the covering range stays under the batch cap."""
    out, cur = [], []
    for job in sorted(jobs, key=lambda j: (j[1], j[0][2])):
        trial = cur+[job]
        n = sum(len(j[3]) for j in trial)
        minutes = (max(j[2] for j in trial)-trial[0][1])//60+2
        if cur and (n > 100 or n*minutes > 9500):
            out.append(cur)
            cur = [job]
        else:
            cur = trial
    return out+[cur] if cur else out


def batch_params(group):
    return dict(market_tickers=",".join(t for j in group for t in j[3]), start_ts=group[0][1], end_ts=max(j[2] for j in group), period_interval=1)


def download(client, stage, names, deadline):
    bars, known = Bars(), screen()
    if stage == "recent":
        jobs = [j for name in names for j in todo(name, bars, known)[0]]
        groups = chunks(jobs)
        print(f"recent: {len(jobs)} events in {len(groups)} batch requests", flush=True)
        for n, group in enumerate(groups):
            if time.time() > deadline:
                print(f"  out of time after {n} of {len(groups)}", flush=True)
                break
            params = batch_params(group)
            try:
                read(client, "/markets/candlesticks", params)
                bars.note(params)
            except RuntimeError as exc:
                print("  batch failed", exc, len(group), flush=True)
            if (n+1) % 40 == 0:
                bars.flush()
                print(f"  {n+1} of {len(groups)} {dict(counts)}", flush=True)
        bars.flush()
        return
    for name in names:
        jobs = sorted(todo(name, bars, known)[1], key=lambda j: -j[0][0])        # newest first
        done = 0
        for info, open_ts, settle, need in jobs:
            if time.time() > deadline:
                break
            for ticker in need:
                path, params = market_requests(info[1], True, ticker, open_ts, settle)[0]
                try:
                    data = read(client, path, params)
                except RuntimeError:
                    data = None
                if len(need) > 1 and data:       # three-way: skip the other team once this one is clearly not a 15c-or-less outsider
                    row = pre_event(sorted(data.get("candlesticks") or [], key=lambda b: b["end_period_ts"]), None)
                    if row[0] is not None and row[0] > 15:
                        break
            done += 1
        print(f"archive {name:26} read {done} of {len(jobs)} events {dict(counts)}", flush=True)
        if time.time() > deadline:
            print("  out of time", flush=True)
            break


# ---------------------------------------------------------------- simulation

def quote(bar):
    bid, ask = cents(bar, "yes_bid", "close"), cents(bar, "yes_ask", "close")
    if bid is None or ask is None or not 0 < bid <= ask < 100:
        return None
    return bid, ask


def pre_event(bars, est):
    """(median midpoint, mean spread) of the warm-up, or (None, reason)."""
    if len(bars) < WARMUP+1:
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
    if est and bars[WARMUP-1]["end_period_ts"] > est:
        return None, "warmup_after_start"
    return float(np.median(mids)), float(np.mean(spreads))


def simulate(name, info, m, bars, yes, pre, spread, est, kind, three_way):
    settle, series, event, archived, _ = info
    warm_end = bars[WARMUP-1]["end_period_ts"]
    t0 = max(warm_end, est) if est else warm_end

    def low(b):
        if yes:
            return cents(b, "price", "low")
        high = cents(b, "price", "high")
        return None if high is None else 100-high

    def mid(b):
        q = quote(b)
        return None if q is None else ((q[0]+q[1])/2 if yes else 100-(q[0]+q[1])/2)

    at_start = None
    for b in bars:
        if b["end_period_ts"] <= t0 and mid(b) is not None:
            at_start = mid(b)
    payout = m["_outcome"] if yes else 1-m["_outcome"]
    row = dict(family=name, series=series, event=event, ticker=m["ticker"], yes=yes, settle=settle, archived=archived, three_way=three_way,
               est=est, kind=kind, pre=pre, spread=spread, at_start=at_start, payout=payout, warm_end=warm_end, t0=t0,
               window_hours=round((settle-bars[0]["end_period_ts"])/3600, 2), volume=float(m.get("volume_fp") or m.get("volume") or 0),
               levels={})
    after = [(i, b, low(b)) for i, b in enumerate(bars) if i >= WARMUP]
    lows = [x for _, b, x in after if x is not None and b["end_period_ts"]-60 >= t0]
    row["min_after_start"] = min(lows) if lows else None
    labels = {str(v): v for v in LEVELS}
    labels.update({"pre-10": math.floor(pre)-10, "pre-15": math.floor(pre)-15})
    for label, limit in labels.items():
        if limit > pre-5:
            continue
        cell = dict(limit=limit, strict=False, touch=False, before_start=False)
        for i, b, x in after:
            if x is None:
                continue
            if b["end_period_ts"]-60 < t0:
                cell["before_start"] = cell["before_start"] or x < limit
                continue
            if x <= limit:
                cell["touch"] = True
            if x < limit:
                cell.update(strict=True, ts=b["end_period_ts"], through=limit-x, bar_volume=float(b.get("volume_fp") or b.get("volume") or 0))
                if i+1 < len(bars) and mid(bars[i+1]) is not None:
                    cell["mid_after"] = mid(bars[i+1])-limit
                break
        row["levels"][label] = cell
    return row


def nba_extra(row, info, slates):
    """Season phase by date, and whether the favourite is the home team (ticker suffix is AWAY+HOME)."""
    stamp = row["event"].split("-")[1]
    day = date(2000+int(stamp[:2]), MONTHS[stamp[2:5]], int(stamp[5:7]))
    if date(2025, 4, 15) <= day <= date(2025, 6, 30) or date(2026, 4, 14) <= day <= date(2026, 6, 30):
        phase = "playoffs"
    elif date(2025, 10, 1) <= day <= date(2025, 10, 19) or date(2026, 10, 1) <= day <= PRESEASON_2026_END:
        phase = "preseason"
    else:
        phase = "regular"
    codes = [m["ticker"].rsplit("-", 1)[1] for m in info[4]]
    mine = row["ticker"].rsplit("-", 1)[1]
    fav = mine if row["yes"] else next(c for c in codes if c != mine)
    row.update(phase=phase, day=str(day), fav=fav, home_fav=stamp[7:].endswith(fav))
    for g in slates.get(day.strftime("%Y%m%d"), []):
        if set(g["teams"].values()) == set(codes):
            row["tipoff"] = g["start"]


PRESEASON_2026_END = date(2026, 10, 18)      # checked against the listing in `plan`: no games on 19 Oct, opening night 20 Oct


def build(names):
    bars = Bars()
    fees = json.loads((OUT/"fees.json").read_text()) if (OUT/"fees.json").exists() else {}
    slates = {}
    for f in (ROOT/"data/raw/game-state/nba").glob("day-*.json"):
        slates[f.stem[4:]] = json.loads(f.read_text())
    coverage = json.loads((OUT/"coverage.json").read_text()) if (OUT/"coverage.json").exists() else {}
    for name in names:
        spec, rows, cover = FAMILIES[name], [], Counter()
        events = events_of(None, name)
        cover["listed"] = len(events)
        for info in events:
            est, kind = start_of(spec, info)
            if kind == "odd":
                cover["odd_schedule"] += 1
                continue
            open_ts, settle = window(info, est)
            best, why, missing = None, None, False
            cands = candidates(info)
            for m in cands:
                b = bars.get(info[1], info[3], m["ticker"], open_ts, settle)
                if b is None:
                    missing = True
                    continue
                if not b:
                    why = why or "no_bars"
                    continue
                price, spread = pre_event(b, est)
                if price is None:
                    why = spread
                    continue
                three = len(info[4]) >= 3
                start = price if three or price > 50 else 100-price
                if best is None or start > best[0]:
                    best = (start, three or price > 50, spread, m, b)
            if best is None or (len(cands) > 1 and missing and best[0] < 50):
                cover["not_cached" if missing else (why or "no_bars")] += 1
                continue
            cover["clean"] += 1
            if info[3]:
                cover["clean_archived"] += 1
            start, yes, spread, m, b = best
            row = simulate(name, info, m, b, yes, start, spread, est, kind, len(info[4]) >= 3)
            row["maker_fee"] = fees.get(info[1], {}).get("fee_type") == "quadratic_with_maker_fees"
            if name == "NBA":
                nba_extra(row, info, slates)
            rows.append(row)
        save(OUT/f"rows-{slug(name)}.json", rows)
        coverage[name] = dict(cover, first=iso(rows[0]["settle"]) if rows else None, last=iso(rows[-1]["settle"]) if rows else None)
        save(OUT/"coverage.json", coverage)
        fav = sum(r["pre"] >= 90 for r in rows)
        print(f"{name:26} {dict(cover)} 90c+ {fav} 85-90 {sum(85 <= r['pre'] < 90 for r in rows)}", flush=True)


def slug(name):
    return name.lower().replace(" ", "-")


def iso(t):
    return datetime.fromtimestamp(t, timezone.utc).isoformat()[:10]


# ---------------------------------------------------------------- trade record

def trade_sample(cap):
    """Strict fills of the 80c bid on 90c+ favourites, NBA first, then the other families in turn."""
    by = {}
    for name in FAMILIES:
        path = OUT/f"rows-{slug(name)}.json"
        if path.exists():
            by[name] = [r for r in json.loads(path.read_text()) if r["pre"] >= 90 and r["levels"].get("80", {}).get("strict")]
    out = list(by.pop("NBA", []))
    queues = [sorted(v, key=lambda r: -r["settle"]) for v in by.values() if v]
    while queues and len(out) < cap:
        for q in list(queues):
            if q and len(out) < cap:
                out.append(q.pop(0))
            if not q:
                queues.remove(q)
    return out


def tape(client, row, cutoff, deadline, pages=6):
    """Trades from the start of the fill bar for six minutes: (list, truncated) or None when out of time."""
    begin = row["levels"]["80"]["ts"]-60
    path = "/historical/trades" if begin < cutoff else "/markets/trades"
    base, out, cursor = dict(ticker=row["ticker"], min_ts=begin, max_ts=begin+360, limit=1000), [], ""
    for _ in range(pages):
        params = dict(base, cursor=cursor) if cursor else base
        if not cache_file(path, params).exists() and (client is None or time.time() > deadline):
            return None
        try:
            data = read(client, path, params)
        except RuntimeError:
            return None
        if data is None:
            return None
        out += data.get("trades") or []
        cursor = data.get("cursor")
        if not cursor or not data.get("trades"):
            return out, False
    return out, True


def measure(row, raw):
    """Contracts printed strictly below the bid within five minutes of the first such print, and who initiated them."""
    limit, rows = row["levels"]["80"]["limit"], []
    for t in raw:
        try:
            price = round(100*float(t["yes_price_dollars"])) if "yes_price_dollars" in t else round(float(t["yes_price"]))
            size = float(t.get("count_fp", t.get("count", 0)))
        except (KeyError, TypeError, ValueError):
            continue
        taker_yes = (t.get("taker_side") or t.get("taker_outcome_side")) == "yes"
        rows.append((timestamp(t["created_time"]), price if row["yes"] else 100-price, size, taker_yes != row["yes"]))   # last: taker sold the favourite
    rows.sort()
    below = [r for r in rows if r[1] < limit]
    if not below:
        return dict(found=False, trades=len(rows))
    first = below[0][0]
    span = [r for r in below if r[0] <= first+300]
    at = [r for r in rows if r[1] == limit and first-60 <= r[0] <= first+300]
    return dict(found=True, trades=len(rows), contracts_below=sum(r[2] for r in span), sold_by_taker=sum(r[2] for r in span if r[3]),
                contracts_at_bid=sum(r[2] for r in at), first_below_ts=first, first_below_price=below[0][1],
                seconds_into_bar=first-(row["levels"]["80"]["ts"]-60))


def trades(minutes, cap):
    deadline = time.time()+60*minutes
    sample = trade_sample(cap)
    out = []
    with httpx.Client(timeout=40) as client:
        try:
            cutoff = timestamp(read(client, "/historical/cutoff", {})["trades_created_ts"])
        except (RuntimeError, KeyError, TypeError):
            cutoff = CUTOFF
        for row in sample:
            got = tape(client, row, cutoff, deadline)
            if got is None:
                continue
            raw, truncated = got
            out.append(dict(measure(row, raw), family=row["family"], event=row["event"], truncated=truncated, settle=row["settle"]))
    save(OUT/"trades.json", out)
    print(f"trade record read for {len(out)} of {len(sample)} sampled fills; requests {dict(counts)}")


# ---------------------------------------------------------------- report

def wilson(wins, n, z=1.96):
    if not n:
        return None
    p = wins/n
    centre, half = (p+z*z/(2*n))/(1+z*z/n), z*math.sqrt(p*(1-p)/n+z*z/(4*n*n))/(1+z*z/n)
    return centre-half, centre+half


def boot(values, draws=4000):
    if len(values) < 5:
        return None
    v = np.asarray(values, float)
    means = v[RNG.integers(0, len(v), (draws, len(v)))].mean(1)
    return [float(x) for x in np.percentile(means, [2.5, 97.5])]


def net_of(r, level, fee=True):
    c = r["levels"][level]
    p = c["limit"]/100
    return 100*r["payout"]-c["limit"]-(1.75*p*(1-p) if fee and r.get("maker_fee") else 0)


def cell(rows, level):
    """The dip table for one set of events and one bid level."""
    orders = [r for r in rows if level in r["levels"]]
    fills = [r for r in orders if r["levels"][level]["strict"]]
    out = dict(orders=len(orders), fills=len(fills), touch=sum(r["levels"][level]["touch"] for r in orders),
               before_start=sum(r["levels"][level]["before_start"] for r in orders))
    if not fills:
        return out
    nets = [net_of(r, level) for r in fills]
    limit = float(np.mean([r["levels"][level]["limit"] for r in fills]))
    wins = sum(r["payout"] for r in fills)
    w = wilson(wins, len(fills))
    after = [r["levels"][level]["mid_after"] for r in fills if "mid_after" in r["levels"][level]]
    out.update(fill_rate=len(fills)/len(orders), win=wins/len(fills), needed=limit/100, net=float(np.mean(nets)),
               gross=float(np.mean([net_of(r, level, False) for r in fills])), ci=boot(nets), wilson=[100*w[0]-limit, 100*w[1]-limit],
               per_order=sum(nets)/len(orders), per_order_ci=boot([net_of(r, level) if r["levels"][level]["strict"] else 0.0 for r in orders]),
               ret=float(np.mean(nets))/limit, losses=sum(r["payout"] < .5 for r in fills),
               mid_after=float(np.mean(after)) if after else None)
    return out


def halves(rows):
    rows = sorted(rows, key=lambda r: (r["settle"], r["event"]))
    return rows[:len(rows)//2], rows[len(rows)//2:]


def fmt(c, short=False):
    if not c["fills"]:
        return f'orders {c["orders"]:4} fills    0' + (f' (touch {c["touch"]})' if not short else "")
    if short:
        return f'{c["fills"]:3}/{c["orders"]:<4} {100*c["win"]:5.1f}% {c["net"]:+6.1f}c'
    ci = "[  n<5      ]" if c["ci"] is None else f'[{c["ci"][0]:+5.1f},{c["ci"][1]:+5.1f}]'
    return (f'orders {c["orders"]:4} fills {c["fills"]:4} touch {c["touch"]:4} rate {c["fill_rate"]:4.0%} win {100*c["win"]:5.1f}% need {100*c["needed"]:4.1f}% '
            f'net {c["net"]:+6.1f}c boot {ci} wilson [{c["wilson"][0]:+5.1f},{c["wilson"][1]:+5.1f}] ret {c["ret"]:+6.1%} /order {c["per_order"]:+5.2f}c L {c["losses"]}')


def band(rows, key):
    lo, hi = BANDS[key]
    return [r for r in rows if lo <= r["pre"] < hi]


def gap(rows):
    if not rows:
        return None
    price, win = float(np.mean([r["pre"] for r in rows])), float(np.mean([r["payout"] for r in rows]))
    w = wilson(win*len(rows), len(rows))
    return dict(n=len(rows), price=price, win=100*win, gap=100*win-price, ci=[100*w[0]-price, 100*w[1]-price], losses=sum(r["payout"] < .5 for r in rows))


def passes(rows, level, levels):
    a, b = halves(rows)
    ca, cb = cell(a, level), cell(b, level)
    i = levels.index(level)
    near = [cell(rows, levels[j]) for j in (i-1, i+1) if 0 <= j < len(levels)]
    # The first-half test uses the Wilson interval: the bootstrap of a half with no losing fill is a single point.
    ok = bool(ca["fills"] >= 5 and ca["wilson"][0] > 0 and cb.get("net", -1) > 0 and all(n.get("net", -1) > 0 for n in near))
    return ok, ca, cb


def chance_of_pass(rows, level, sims=3000):
    """If every filled bid were worth exactly its price, how often would this cell pass the half tests (neighbours ignored)?"""
    a, b = halves(rows)
    fa = [r["levels"][level]["limit"]/100 for r in a if level in r["levels"] and r["levels"][level]["strict"]]
    fb = [r["levels"][level]["limit"]/100 for r in b if level in r["levels"] and r["levels"][level]["strict"]]
    if len(fa) < 5 or not fb:
        return 0.0
    pa, pb = np.array(fa), np.array(fb)
    wins = (RNG.random((sims, len(pa))) < pa).sum(1)
    low = np.array([wilson(w, len(pa))[0] for w in wins])
    second = 100*((RNG.random((sims, len(pb))) < pb)-pb).mean(1)
    return float(np.mean((low > pa.mean()) & (second > 0)))


def shrink(cells, prior=0.5):
    """edge_reality.py's model: logit(p_family) ~ N(mu, tau), wins ~ Binomial(fills, p); flat mu, half-normal tau. cells: {name: (wins, fills, limit)}."""
    names = [k for k, v in cells.items() if v[1] > 0]
    if len(names) < 2:
        return {}
    mus, taus, zs = np.linspace(-1, 5, 121), np.linspace(0.01, 2, 60), np.linspace(-3, 3, 41)
    wz = np.exp(-zs**2/2)
    wz /= wz.sum()
    logp = np.zeros((len(mus), len(taus)))
    post = {k: np.zeros((len(mus), len(taus))) for k in names}
    for i, mu in enumerate(mus):
        for j, tau in enumerate(taus):
            total = -.5*(tau/prior)**2
            for k in names:
                wins, n, _ = cells[k]
                p = 1/(1+np.exp(-(mu+tau*zs)))
                like = wz*np.exp(wins*np.log(p)+(n-wins)*np.log(1-p))
                total += math.log(like.sum())
                post[k][i, j] = (like*p).sum()/like.sum()
            logp[i, j] = total
    w = np.exp(logp-logp.max())
    w /= w.sum()
    return {k: dict(fills=cells[k][1], raw=100*cells[k][0]/cells[k][1]-cells[k][2], shrunk=float(100*(w*post[k]).sum()-cells[k][2])) for k in names}


def load(name):
    path = OUT/f"rows-{slug(name)}.json"
    return json.loads(path.read_text()) if path.exists() else []


def drop_best_month(rows, level):
    fills = [r for r in rows if level in r["levels"] and r["levels"][level]["strict"]]
    by = defaultdict(list)
    for r in fills:
        by[iso(r["settle"])[:7]].append(net_of(r, level))
    if len(by) < 2:
        return None
    best = max(by, key=lambda k: sum(by[k]))
    rest = [x for k, v in by.items() if k != best for x in v]
    return dict(dropped=best, dropped_fills=len(by[best]), fills=len(rest), net=float(np.mean(rest)) if rest else None)


def report():
    lines = []

    def say(text=""):
        print(text)
        lines.append(text)
    cover = json.loads((OUT/"coverage.json").read_text())
    fees = json.loads((OUT/"fees.json").read_text()) if (OUT/"fees.json").exists() else {}
    data = {name: load(name) for name in FAMILIES}
    nba = data["NBA"]
    groups = {"NBA regular season": [r for r in nba if r["phase"] == "regular"], "NBA playoffs": [r for r in nba if r["phase"] == "playoffs"],
              "NBA preseason": [r for r in nba if r["phase"] == "preseason"]}
    groups.update({k: v for k, v in data.items() if k != "NBA"})
    pooled = ("Basketball Asia-Pacific", "Basketball Europe", "Basketball Americas other")
    summary = dict(coverage=cover, families={})
    levels = ["85", "80", "75", "70"]

    say("COVERAGE (events listed, with a clean pre-event price, and why the rest were lost)")
    for name, c in cover.items():
        say(f"  {name:26} {c}")
    say("\nFEE TYPES: " + ", ".join(f'{k} {v.get("fee_type")}x{v.get("fee_multiplier")}' for k, v in sorted(fees.items())))

    say("\nPRE-EVENT GAP AND THE DIP RULE BY FAMILY (maker fee charged where the series has one)")
    tested = []
    for name, rows in groups.items():
        if not rows:
            continue
        entry = dict(events=len(rows), first=iso(min(r["settle"] for r in rows)), last=iso(max(r["settle"] for r in rows)), bands={})
        say(f'\n{name}: {len(rows)} events with a clean pre-event price, {entry["first"]}..{entry["last"]}'
            + ("  [NO USABLE SCHEDULE: pre-event price and fill timing unverified]" if FAMILIES.get(name, {}).get("timing") == "none" else ""))
        for key in ("90+", "85-90"):
            sub = band(rows, key)
            g = gap(sub)
            if not g:
                say(f"  {key}: no favourites")
                continue
            a, b = halves(sub)
            ga, gb = gap(a), gap(b)
            say(f'  {key}: n {g["n"]} price {g["price"]:.1f} won {g["win"]:.1f}% gap {g["gap"]:+.1f}c wilson [{g["ci"][0]:+.1f},{g["ci"][1]:+.1f}] losses {g["losses"]}'
                f' | halves {ga["n"] if ga else 0} {ga["gap"] if ga else 0:+.1f}c / {gb["n"]} {gb["gap"]:+.1f}c')
            entry["bands"][key] = dict(gap=g, levels={})
            for level in (levels if key == "90+" else levels[1:])+["pre-10"]:
                c = cell(sub, level)
                ok, ca, cb = passes(sub, level, levels) if level in levels else (False, cell(a, level), cell(b, level))
                entry["bands"][key]["levels"][level] = dict(all=c, first=ca, second=cb, passes=ok)
                say(f'    bid {level:6} {fmt(c)} | h1 {fmt(ca, True)} | h2 {fmt(cb, True)}{"  PASS" if ok else ""}')
                if key == "90+" and level in levels and FAMILIES.get(name, {}).get("timing") != "none":
                    tested.append((name, level, c, ok, chance_of_pass(sub, level)))
        summary["families"][name] = entry
    sub = [r for k in pooled for r in band(data[k], "90+")]
    say(f"\nPooled basketball outside the NBA and WNBA (three families above), 90c+: {gap(sub) and {k: round(v, 1) if isinstance(v, float) else v for k, v in gap(sub).items()}}")
    for level in levels+["pre-10"]:
        a, b = halves(sub)
        say(f"    bid {level:6} {fmt(cell(sub, level))} | h1 {fmt(cell(a, level), True)} | h2 {fmt(cell(b, level), True)}")

    say("\nVARIANT: favourite still 90c+ at the scheduled start (last quote before it), 80c bid")
    for name, rows in groups.items():
        sub = [r for r in band(rows, "90+") if r["at_start"] is not None and r["at_start"] >= 90]
        moved = [r for r in band(rows, "90+") if r["at_start"] is not None and r["at_start"] < 85]
        if sub:
            say(f"  {name:26} {fmt(cell(sub, '80'))} | fell under 85c before the start: {len(moved)}")

    # ---- NBA reconciliation
    say("\nNBA: THE DIP RULE BY PRE-GAME BAND, HOME/AWAY, PHASE AND MONTH")
    for key in ("55-70", "70-80", "80-90", "90+"):
        sub = band(nba, key)
        g = gap(sub)
        say(f'  pre-game {key}: n {g["n"]} price {g["price"]:.1f} won {g["win"]:.1f}% gap {g["gap"]:+.1f}c [{g["ci"][0]:+.1f},{g["ci"][1]:+.1f}]')
        for level in (["80", "75", "70"] if key == "90+" else [])+["pre-10", "pre-15"]:
            a, b = halves(sub)
            say(f"    bid {level:6} {fmt(cell(sub, level))} | h1 {fmt(cell(a, level), True)} | h2 {fmt(cell(b, level), True)}")
        for label, pick in (("home favourite", lambda r: r["home_fav"]), ("away favourite", lambda r: not r["home_fav"]),
                            ("regular", lambda r: r["phase"] == "regular"), ("playoffs", lambda r: r["phase"] == "playoffs"),
                            ("preseason", lambda r: r["phase"] == "preseason")):
            part = [r for r in sub if pick(r)]
            if part:
                g = gap(part)
                level = "80" if key == "90+" else "pre-10"
                say(f'      {label:15} n {g["n"]:4} gap {g["gap"]:+5.1f}c | bid {level}: {fmt(cell(part, level), True)}')
    say("  by month, pre-game 90c+ (n, gap, then the 80c bid fills/orders, win, net):")
    by = defaultdict(list)
    for r in band(nba, "90+"):
        by[r["day"][:7]].append(r)
    for month in sorted(by):
        g = gap(by[month])
        say(f'      {month} n {g["n"]:3} gap {g["gap"]:+5.1f}c losses {g["losses"]} | {fmt(cell(by[month], "80"), True)}')
    say("  by month, pre-game 80-90c (bid pre-10):")
    by = defaultdict(list)
    for r in band(nba, "80-90"):
        by[r["day"][:7]].append(r)
    for month in sorted(by):
        g = gap(by[month])
        say(f'      {month} n {g["n"]:3} gap {g["gap"]:+5.1f}c | {fmt(cell(by[month], "pre-10"), True)}')
    say("  by period, pre-game 90c+ and 80-90c (research/favorite_dip.py had already run January to mid-September 2026; the other two are new):")
    for label, lo, hi in (("Apr-Dec 2025 (new)", "2025-01", "2026-01"), ("Jan-17 Sep 2026 (seen before)", "2026-01", "2026-09-18"),
                          ("18 Sep-Nov 2026 (new)", "2026-09-18", "2027-01")):
        for key, level in (("90+", "80"), ("90+", "pre-10"), ("80-90", "pre-10")):
            part = [r for r in band(nba, key) if lo <= iso(r["settle"]) < hi]
            g = gap(part)
            if g:
                say(f'      {label:30} {key:5} n {g["n"]:3} gap {g["gap"]:+5.1f}c losses {g["losses"]:2} | bid {level:6} {fmt(cell(part, level), True)}')
    known = [r for r in nba if "tipoff" in r]
    if known:
        off = np.array([(r["est"]-r["tipoff"])/60 for r in known])
        say(f"  schedule check: estimated start against the play-by-play archive's tip-off for {len(known)} games: equal in {np.mean(off == 0):.1%}, "
            f"within 15 minutes in {np.mean(abs(off) <= 15):.1%}")
    moved = [(r["pre"], r["at_start"]) for r in nba if r["at_start"] is not None and r["pre"] >= 90]
    if moved:
        d = np.array([b-a for a, b in moved])
        say(f"  90c+ games: price at the start minus the pre-game price: mean {d.mean():+.1f}c, 10th/50th/90th pct {np.percentile(d, 10):+.1f}/{np.percentile(d, 50):+.1f}/{np.percentile(d, 90):+.1f}")

    # ---- sanity checks
    say("\nSANITY CHECKS (80c bid, 90c+ favourites)")
    trade_rows = json.loads((OUT/"trades.json").read_text()) if (OUT/"trades.json").exists() else []
    for name, rows in groups.items():
        sub = band(rows, "90+")
        fills = [r for r in sub if r["levels"]["80"]["strict"]]
        if not sub:
            continue
        frac = sum(r["payout"] not in (0, 1, 0.0, 1.0) for r in sub)
        late = sum(1 for r in sub if r["est"] and r["settle"]-r["est"] > 5*3600)
        text = f'  {name:26} orders {len(sub)} fills {len(fills)} | fractional settlements {frac}, settled 5h+ after start {late}, prints under 80c before the start {sum(r["levels"]["80"]["before_start"] for r in sub)}'
        if fills and fills[0]["est"]:
            mins = np.array([(r["levels"]["80"]["ts"]-60-r["est"])/60 for r in fills])
            left = np.array([(r["settle"]-r["levels"]["80"]["ts"])/60 for r in fills])
            text += (f' | fill minutes after start: min {mins.min():.0f} p10 {np.percentile(mins, 10):.0f} median {np.median(mins):.0f}, first 10 min: {int((mins < 10).sum())}'
                     f' | minutes before settlement median {np.median(left):.0f}')
        d = drop_best_month(sub, "80")
        if d and d["net"] is not None:
            text += f' | without best month {d["dropped"]} ({d["dropped_fills"]} fills): {d["fills"]} fills {d["net"]:+.1f}c'
        after = [r["levels"]["80"]["mid_after"] for r in fills if "mid_after" in r["levels"]["80"]]
        if after:
            text += f' | mid a minute later vs bid {np.mean(after):+.1f}c'
        if fills:
            vol = np.array([r["levels"]["80"]["bar_volume"] for r in fills])
            recent = [r for r in fills if r["settle"] >= CUTOFF]
            text += (f' | contracts traded in the fill minute: p10 {np.percentile(vol, 10):.0f} median {np.median(vol):.0f}, under 100: {int((vol < 100).sum())}'
                     f' | fills since 18 Sep: {len(recent)} ({len(recent)/2.0:.1f} a month)')
        say(text)
    if trade_rows:
        say("  trade record around sampled fills (contracts strictly below the bid within five minutes of the first such print):")
        by = defaultdict(list)
        for t in trade_rows:
            by["NBA" if t["family"] == "NBA" else "other families"].append(t)
        for k, v in by.items():
            found = [t for t in v if t["found"]]
            vol = np.array([t["contracts_below"] for t in found]) if found else np.array([0.0])
            sold = sum(t["sold_by_taker"] for t in found)/max(sum(t["contracts_below"] for t in found), 1)
            say(f'    {k:15} sampled {len(v)}, print below the bid confirmed in {len(found)}, truncated tapes {sum(t["truncated"] for t in v)}; contracts below bid: '
                f'p10 {np.percentile(vol, 10):.0f} median {np.median(vol):.0f} p90 {np.percentile(vol, 90):.0f}; share under 100 contracts {np.mean(vol < 100):.0%}; '
                f'seller-initiated {sold:.0%}; median contracts at the bid itself {np.median([t["contracts_at_bid"] for t in found]) if found else 0:.0f}')

    # ---- multiplicity
    say("\nMULTIPLICITY (90c+ favourites, fixed bids 85/80/75/70, families with a usable schedule)")
    with_fills = [t for t in tested if t[2]["fills"]]
    won = [t for t in tested if t[3]]
    say(f"  cells examined: {len(tested)} ({len({t[0] for t in tested})} families x 4 bids), {len(with_fills)} with at least one fill, "
        f"{sum(t[2]['fills'] >= 10 for t in tested)} with 10+ fills")
    say(f"  pass (first-half Wilson interval above zero on 5+ fills, second half positive, neighbouring bids positive): {len(won)} {[(t[0], t[1]) for t in won]}")
    say(f"  expected to pass the two half tests by chance if every fill were worth its price: {sum(t[4] for t in tested):.2f}")
    say(f'  cells positive: {sum(t[2].get("net", -1) > 0 for t in with_fills)} of {len(with_fills)}; with bootstrap interval above zero: '
        f'{sum(bool(t[2].get("ci")) and t[2]["ci"][0] > 0 for t in with_fills)}')
    for level in ("80", "pre-10"):
        cells = {}
        for name, rows in groups.items():
            if FAMILIES.get(name, {}).get("timing") == "none":
                continue
            fills = [r for r in band(rows, "90+") if level in r["levels"] and r["levels"][level]["strict"]]
            if fills:
                cells[name] = (sum(r["payout"] for r in fills), len(fills), float(np.mean([r["levels"][level]["limit"] for r in fills])))
        model = shrink(cells)
        say(f"  shrinkage across families, bid {level} (net cents per fill, raw -> shrunk):")
        for k, v in sorted(model.items(), key=lambda kv: -kv[1]["shrunk"]):
            say(f'      {k:26} fills {v["fills"]:3} raw {v["raw"]:+6.1f}c shrunk {v["shrunk"]:+5.1f}c')
        tot = (sum(v[0] for v in cells.values()), sum(v[1] for v in cells.values()))
        if tot[1]:
            lim = sum(v[2]*v[1] for v in cells.values())/tot[1]
            w = wilson(tot[0], tot[1])
            say(f"      all families pooled: {tot[1]} fills, won {100*tot[0]/tot[1]:.1f}% against {lim:.1f}% needed, net {100*tot[0]/tot[1]-lim:+.1f}c wilson [{100*w[0]-lim:+.1f},{100*w[1]-lim:+.1f}]")
        summary[f"shrink_{level}"] = model
    summary["tested"] = [dict(family=t[0], level=t[1], cell=t[2], passes=t[3], chance=t[4]) for t in tested]
    save(OUT/"summary.json", summary)
    (OUT/"report.txt").write_text("\n".join(lines)+"\n")


def plan(names):
    bars, known = Bars(), screen()
    total = Counter()
    for name in names:
        events = events_of(None, name)
        recent, archive, have = todo(name, bars, known)
        total["recent_batches"] += len(chunks(recent))
        total["archive_reads"] += sum(len(j[3]) for j in archive)
        print(f"{name:26} listed {len(events):5} archived {sum(e[3] for e in events):5} bars cached {have:5} recent to fetch {len(recent):5} "
              f"archive to fetch {len(archive):5} ({sum(len(j[3]) for j in archive)} reads)")
        if name == "NBA":
            days = Counter(e[2].split("-")[1][:7] for e in events if e[2].split("-")[1][:5] == "26OCT")
            print("   October 2026 games per day:", sorted(days.items()))
    print(dict(total), "all recent pooled:", len(chunks([j for n in names for j in todo(n, bars, known)[0]])), "batches")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("cmd", choices=("listing", "plan", "download", "build", "trades", "report"))
    parser.add_argument("--families", nargs="+", choices=tuple(FAMILIES))
    parser.add_argument("--stage", choices=("recent", "archive"), default="recent")
    parser.add_argument("--minutes", type=float, default=9, help="stop starting new requests after this long; a rerun resumes from the cache")
    parser.add_argument("--cap", type=int, default=110, help="fills sampled by `trades`")
    args = parser.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    names = args.families or (ARCHIVE_ORDER if args.cmd == "download" and args.stage == "archive" else list(FAMILIES))
    deadline = time.time()+60*args.minutes
    if args.cmd == "plan":
        return plan(names)
    if args.cmd == "build":
        return build(names)
    if args.cmd == "report":
        return report()
    if args.cmd == "trades":
        return trades(args.minutes, args.cap)
    with httpx.Client(timeout=60) as client:
        if args.cmd == "listing":
            import research.calibration_map as cm
            cm.PACE = PACE
            fees = json.loads((OUT/"fees.json").read_text()) if (OUT/"fees.json").exists() else {}
            for name in names:
                if time.time() > deadline:
                    print("out of time")
                    break
                events = events_of(client, name)
                for series in FAMILIES[name]["series"]:
                    if series not in fees and any(e[1] == series for e in events) and time.time() < deadline:
                        try:
                            s = read(client, f"/series/{series}", {}).get("series", {})
                            fees[series] = dict(fee_type=s.get("fee_type"), fee_multiplier=s.get("fee_multiplier"))
                        except RuntimeError:
                            pass
                save(OUT/"fees.json", fees)
                print(f"{name:26} events {len(events):5} archived {sum(e[3] for e in events):5} requests {dict(counts)} {dict(cm.counts)}", flush=True)
            return
        download(client, args.stage, names, deadline)
        print("requests", dict(counts))


if __name__ == "__main__":
    main()
