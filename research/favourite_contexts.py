"""Where do heavy pre-event favourites concentrate their underpricing? ("What is the first-round qualifying of other sports?")

Background: research/calibration_map.py found that favourites priced 90-99c
before the event win about 1.7c more often than their price, pooled over 19
sports families, and research/tennis_contexts.py found that in tennis nearly
all of it sits in first-round qualifying. This script looks for the same kind
of context, knowable before the event, in men's and women's college
basketball, soccer, esports, cricket and basketball outside the NBA.

Definitions (those of calibration_map.py and dip_other_sports.py)

* One row per event. Two-outcome events use the canonical (lowest) ticker and
  the favourite is the side with a pre-event midpoint above 50c; three-way
  soccer events use the YES side of the dearest outcome, so there is one row
  per event and the dependence between outcomes never enters.
* Pre-event price = median midpoint of the first 15 one-minute bars of the
  window (spread at most 4c on every bar, midpoints within 3c). The window
  opens six hours before settlement, or one hour before the scheduled start
  when that is earlier (long or late-settled events). The warm-up must end
  before the start: the start in the ticker where there is one, otherwise
  Kalshi's expected expiration minus three hours (soccer, and this season's
  college basketball), otherwise the close minus 130 minutes (last season's
  college basketball, whose listings carry no usable schedule). Events not settled at 0 or 1 (cancelled) are dropped.
* Gap = 100 x settlement - pre-event price, on the favourite. Prices of 99c
  and above are excluded.
* Buy at the ask: pays midpoint + half the warm-up spread + the taker fee
  0.07*p*(1-p). Returns are also given as a percentage of the price paid.
* Dip bid: favourite at 90c+, a bid rests at 80c (80-90c favourites, the
  control: 10c below the price). Filled only when a trade prints strictly
  below the bid in a minute after the start of play. Pays the limit, held to
  settlement; shown with no maker fee and with 0.0175*p*(1-p).
* Intervals: gap +/- 1.96 x the LARGEST of three standard errors: the sample
  one, the one the prices imply (p(1-p), so a cell with no losses is not
  certain) and one clustered by the favourite team. One row per event, so
  the event is the unit throughout.
* Halves: each family's 80c+ rows are ordered by settlement and split at the
  median. A context "passes" when its interval is above zero in the first
  half and its gap is positive in the second.
* Multiplicity: the number of context cells examined, the number that pass,
  and the number that pass in 1,000 simulations in which every favourite
  wins with probability price + its family's pooled gap (so a context adds
  nothing). Each context is also contrasted with the rest of its family, and
  the contrasts are shrunk with a normal-normal model over all cells, as
  tennis_contexts.py and edge_reality.py do.
* Walk-forward features (college basketball) use only events that settled
  before the row's window opened.
* "at tip" (college basketball only): the live rule (paper/rules.py) prices
  the favourite from its last 10-15 quotes before play, not from a quote
  three hours earlier. The families "cbb-men at tip" and "cbb-women at tip"
  repeat every table on that definition: the median midpoint of the last
  10-15 one-minute bars ending two minutes before the scheduled start (25
  minutes before the estimated start for last season, whose start is only
  known to about 20 minutes), same spread and range limits. They are the
  same games as "cbb-men" and "cbb-women", so their cells are not
  independent of those; both sets are counted in the multiplicity totals,
  and only the "at tip" versions enter the cross-family averages.
* This season's college start is Kalshi's expected expiration minus three
  hours (checked: the close then falls a median 130 minutes later).

Network (commands `listings`, `cbb`, `others`): public reads only, one request
at a time, at most one per PACE seconds, long pauses after HTTP 429 (a live
trader and a recorder share the limit). Recent events go through the batch
endpoint GET /markets/candlesticks: first hourly bars to find the events with
a favourite near 80c or more (100 tickers per request), then one-minute bars
of the favourite's market only. Archived college basketball games cost one
read each and are taken in a fixed pseudo-random order, so a run cut short by
the clock leaves a uniform sample of the season rather than its latest weeks.
Responses are cached under data/raw/cost-research-v1 and rows are saved as
they are built, so an interrupted run resumes. No orders, no authentication.

    .venv/bin/python research/favourite_contexts.py listings --minutes 6
    .venv/bin/python research/favourite_contexts.py cbb --minutes 9
    .venv/bin/python research/favourite_contexts.py others --minutes 9
    .venv/bin/python research/favourite_contexts.py rebuild      (offline; after changing study())
    .venv/bin/python research/favourite_contexts.py report
"""
import argparse
from collections import Counter, defaultdict
from datetime import date, datetime, timezone
import hashlib
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
from research.api_cost import ROOT, midnight, timestamp
from research.calibration_map import bar_requests, cache_file, listing, pregame, quote, scheduled, ts, volume
from research.near_end import WINDOW, cents
from research.thin_book import read

OUT = ROOT / "data/processed/favourite-contexts"
CALIB = ROOT / "data/processed/calibration-map"
PACE, WARMUP = 1.4, 15
EASTERN = ZoneInfo("America/New_York")
RNG = np.random.default_rng(23)
FRESH = dict(status="settled", limit=1000, min_close_ts=midnight(date(2026, 11, 15)))
used = Counter()
FEE_CHECK = {"KXEPLGAME", "KXUCLGAME", "KXINTLFRIENDLYGAME", "KXCS2GAME", "KXLOLGAME", "KXT20MATCH", "KXEUROLEAGUEGAME", "KXCBAGAME"}

SOCCER_TYPES = {
    "top league (big five)": "EPL LALIGA SERIEA BUNDESLIGA LIGUE1",
    "lower league": "EFLCHAMPIONSHIP BRASILEIROB LALIGA2 BUNDESLIGA2 SERIEB SERIEC BRASILEIROC J2LEAGUE TFF1LIG CZEFNL EERSTEDIV EFLL1 "
                    "EFLL2 ISRNL ETTAN SVK2L LIGUE2 ENGNL CHNL1 GER3L K2LEAGUE FINYL TWEEDEDIV LIGAPORTUGAL2 ARGNACB LIGAEXP USL USLL1",
    "domestic cup": "EFLCUP FACUP COPADELREY DFBPOKAL COPPAITALIA COPADOBRASIL COUPEDEFRANCE TACAPORT GRECUP KNVBCUP SCOCUP EFLTROPHY "
                    "SVKCUP USOPENCUP USLCUP SERIECCUP PERLEAGUECUP JLEAGUECUP ARGCUP ISRPLCUP ESPSUPERCUP FRASUPERCUP GERSC ENGCS ARGSUPERCUP",
    "continental club": "UCL UEL UECL CONMEBOLLIB CONMEBOLSUD AFCCL CONCACAFCCUP LEAGUESCUP CLUBWC INTERCONCUP UEFASC",
    "international friendly": "INTLFRIENDLY",
    "international competitive": "UEFANL CONCACAFNL AFCON FIFA WC ASEAN SAFF AFCAC",
    "club friendly": "CLUBF",
    "women": "UCLW NWSL EWSL LIGAF SERIEAW BUNDESLIGAW FRAPLW EREDIVISIEW LIGAMXW FIFAW",
}
SKIP_SERIES = {"KXWCGOALEVERYGAME", "KXWCTEAMSINGAME", "KXMLSASTGAME", "KXNBAGAME", "KXNCAAMBGAME", "KXNCAAWBGAME", "KXWNBAGAME",
               "KXNBASUMMERGAME", "KXNBAALLSTARGAME", "KXNBACELEBRITYGAME", "KXESOCCERGAME", "KXEBASKETBALLGAME", "KXEFOOTBALLGAME",
               "KXSOWBBALLGAME", "KXBILGAME", "KXTBTGAME", "KXUNRIVALEDGAME"}


def families():
    """{family: [series]}: the two college families, and every soccer / esports / cricket / other-basketball game series
    that `discover` (calibration_map.py) saw with at least 300,000 contracts of lifetime volume."""
    out = {"cbb-men": ["KXNCAAMBGAME"], "cbb-women": ["KXNCAAWBGAME"], "soccer": [], "esports": [], "cricket": [], "basketball-other": []}
    names = {"Soccer": "soccer", "Esports": "esports", "Cricket": "cricket", "Basketball": "basketball-other"}
    for s in json.loads((CALIB/"discovery.json").read_text())["Sports"]:
        tag = next((names[t] for t in s["tags"] if t in names), None)
        if tag and re.search(r"(GAME|MATCH)$", s["ticker"]) and s["ticker"] not in SKIP_SERIES and s["volume"] >= 3e5:
            out[tag].append(s["ticker"])
    return out


def store(fam):
    path = OUT/f"rows-{fam}.json"
    return json.loads(path.read_text()) if path.exists() else {}


def keep(fam, rows):
    OUT.mkdir(parents=True, exist_ok=True)
    temp = OUT/f"rows-{fam}.tmp"
    temp.write_text(json.dumps(rows))
    temp.replace(OUT/f"rows-{fam}.json")


def net(client, path, params):
    began = time.time()
    data, throttled = read(client, path, params, PACE)
    used["requests"] += 1
    used["throttled"] += throttled
    used["seconds"] += time.time()-began
    return data


def parse(markets, events, archived, series):
    group = defaultdict(list)
    for m in markets:
        try:
            m["_settle"], m["_outcome"] = int(timestamp(m["settlement_ts"])), float(m["settlement_value_dollars"])
        except (KeyError, TypeError, ValueError):
            continue
        group[m["event_ticker"]].append(m)
    for event, ms in group.items():
        ms.sort(key=lambda m: m["ticker"])
        if event not in events or len(ms) > len(events[event][4]):
            events[event] = (ms[0]["_settle"], series, event, archived, ms)


def events_of(series, client=None, archive=False):
    """Settled events of a series from the cached listings (plus, with a client, anything missing and a fresh page of the
    last three days for the college series)."""
    events = listing(client, series, archive=archive)
    path, params = "/markets", dict(FRESH, series_ticker=series)
    if client is not None and series.startswith("KXNCAA") and not cache_file(path, params).exists():
        net(client, path, params)
    if cache_file(path, params).exists():
        parse(json.loads(cache_file(path, params).read_text())["markets"], events, False, series)
    return events


def start_of(fam, e):
    """Scheduled start: from the ticker, else expected expiration - 3h (soccer), else close - 130 minutes (college basketball)."""
    sched = scheduled(e[2])
    if sched:
        return sched
    m = e[4][0]
    expected = ts(m, "expected_expiration_time")
    if fam.startswith("cbb") and (e[3] or not expected):      # last season's expected expiration is rounded and unreliable
        close = ts(m, "close_time")
        return min(close, e[0])-130*60 if close else e[0]-135*60
    return expected-3*3600 if expected else None


def anchor_of(fam, e):
    start = start_of(fam, e)
    if start and not fam.startswith("cbb") and start-3600 < e[0]-WINDOW:
        return start-3600
    return e[0]-WINDOW


def names_of(e, fav):
    """(favourite, opponent, favourite listed first?, text between "wins the" and "originally scheduled")."""
    others = [m for m in e[4] if m["ticker"] != fav["ticker"] and not m["ticker"].endswith("-TIE")]
    a, b = fav.get("yes_sub_title") or "", (others[0].get("yes_sub_title") or "") if others else ""
    text = (fav.get("rules_primary") or "").split(" originally scheduled")[0]
    text = text.split(" wins the ", 1)[1] if " wins the " in text else text
    body = text.split(": ", 1)[1] if re.search(r"^[^:]{3,90}: ", text) and " vs. " in text else text
    first = bool(re.match(re.escape(a)+r" (at|vs\.?) ", body)) if a else None
    return a, b, first, text


def study(fam, e, per, full):
    """One row from {ticker: one-minute bars from the window's opening}; a dict with `skip` when there is no clean price."""
    settle, series, event, archived, ms = e
    three = len(ms) >= 3
    best, why = None, "no_bars"
    for m in ms:
        bars = per.get(m["ticker"])
        if not bars:
            continue
        mid, spread = pregame(bars)
        if mid is None:
            why = spread
            continue
        yes = three or mid > 50
        price = mid if yes else 100-mid
        if best is None or price > best[0]:
            best = (price, yes, spread, m, bars)
    if best is None:
        return dict(event=event, skip=why)
    price, yes, spread, m, bars = best
    fav = m if yes else next((x for x in ms if x["ticker"] != m["ticker"]), m)
    start = start_of(fam, e)
    warm = bars[:WARMUP]
    row = base(fam, e, fav, price, spread, m["_outcome"] if yes else 1-m["_outcome"], warm[-1]["end_period_ts"],
               float(warm[-1].get("open_interest_fp") or warm[-1].get("open_interest") or 0), sum(volume(b) for b in warm))
    row.update(full=full, bar_ticker=m["ticker"], yes=yes)
    if full:
        limit = 80 if price >= 90 else math.floor(price)-10
        row.update(limit=limit, fill=False, fill_any=False, pre_volume=sum(volume(b) for b in bars[WARMUP:] if b["end_period_ts"] <= (start or 0)))
        # The price as the live rule sees it: the last 10-15 quotes ending two minutes before the scheduled start (25 minutes
        # before the estimated start for last season's college games, whose start is only known to about 20 minutes).
        margin = 25*60 if fam.startswith("cbb") and archived else 120
        late = [quote(b) for b in bars if b["end_period_ts"] <= (start or 0)-margin][-WARMUP:]
        if len(late) >= 10 and all(q and q[1]-q[0] <= 4 for q in late):
            mids = [(q[0]+q[1])/2 if yes else 100-(q[0]+q[1])/2 for q in late]
            if max(mids)-min(mids) <= 3:
                row.update(tip_price=float(np.median(mids)), tip_spread=float(np.mean([q[1]-q[0] for q in late])))
        lows = []
        for b in bars[WARMUP:]:
            high = cents(b, "price", "high")
            low = cents(b, "price", "low") if yes else (None if high is None else 100-high)
            if low is None:
                continue
            if low < limit:
                row["fill_any"] = True
            if start and b["end_period_ts"] > start+60:
                lows.append(low)
                if low < limit and not row["fill"]:
                    row.update(fill=True, fill_minutes_after_start=(b["end_period_ts"]-start)/60)
        row["low_in_play"] = min(lows) if lows else None
    return row


def base(fam, e, fav, price, spread, payout, warm_end, oi, warm_volume):
    a, b, first, text = names_of(e, fav)
    m = e[4][0]
    return dict(family=fam, series=e[1], event=e[2], settle=e[0], archived=e[3], three_way=len(e[4]) >= 3, start=price, spread=spread,
                payout=payout, warm_end=warm_end, open_interest=oi, warm_volume=warm_volume, begin=start_of(fam, e),
                fav=a, opp=b, fav_code=fav["ticker"].rsplit("-", 1)[-1], fav_first=first, text=text,
                market_volume=float(fav.get("volume_fp") or fav.get("volume") or 0), close=ts(m, "close_time"))


def offline(fam, e):
    """A full row from the per-market six-hour reads earlier scripts cached, or None."""
    sched, per = scheduled(e[2]), {}
    markets = e[4] if len(e[4]) >= 3 else e[4][:1]
    for m in markets:
        for path, params in bar_requests(e[1], e[3], m, sched):
            file = cache_file(path, params)
            if file.exists():
                per[m["ticker"]] = sorted(json.loads(file.read_text()).get("candlesticks") or [], key=lambda b: b["end_period_ts"])
                break
    if len(per) < len(markets):
        return None
    return study(fam, e, per, True)


def binary(e):
    return all(m["_outcome"] in (0.0, 1.0) for m in e[4]) and len(e[4]) >= 2


# ---------------------------------------------------------------- downloads

def chunk(jobs, span, period):
    """Group (anchor, tickers) jobs so that a request has at most 100 tickers and tickers x periods stays under 9,000."""
    out, cur, n = [], [], 0
    for anchor, tickers in sorted(jobs):
        periods = (anchor-cur[0][0]+span)//period+3 if cur else 0
        if cur and (n+len(tickers) > 100 or (n+len(tickers))*periods > 9000):
            out.append(cur)
            cur, n = [], 0
        cur.append((anchor, tickers))
        n += len(tickers)
    return out+[cur] if cur else out


def batch(client, group, span, period):
    params = dict(market_tickers=",".join(t for _, ts_ in group for t in ts_), start_ts=group[0][0], end_ts=group[-1][0]+span,
                  period_interval=period)
    try:
        data = net(client, "/markets/candlesticks", params)
    except RuntimeError as exc:
        print("   batch failed", exc, flush=True)
        used["failed"] += 1
        return None
    return {x["market_ticker"]: sorted(x.get("candlesticks") or [], key=lambda b: b["end_period_ts"]) for x in data.get("markets", [])}


def hourly_favourite(bars, three):
    best = 0
    for b in bars:
        bid, ask = cents(b, "yes_bid", "close"), cents(b, "yes_ask", "close")
        if bid is None or ask is None or not 0 < bid <= ask < 100:
            continue
        mid = (bid+ask)/2
        best = max(best, mid if three else max(mid, 100-mid))
    return best


def recent(client, fam, serieses, deadline, everything=False):
    """Recent-tier events: reuse cached reads, then hourly bars to find favourites near 80c+, then their one-minute bars.
    `everything` (college basketball) skips the hourly screen so that every game gets a row."""
    rows, cover = store(fam), Counter()
    events = {}
    for series in serieses:
        events.update({k: v for k, v in events_of(series).items() if not v[3]})
    todo = []
    for e in events.values():
        cover["listed"] += 1
        if not binary(e):
            cover["not settled at 0 or 1"] += 1
            continue
        old = rows.get(e[2])
        if old is not None and (old.get("skip") or old.get("full") or old["start"] < 80):
            continue
        if old is None:
            row = offline(fam, e)
            if row is not None:
                rows[e[2]] = row
                cover["from earlier caches"] += 1
                continue
        todo.append(e)
    keep(fam, rows)
    span = (8 if fam in ("esports", "cricket", "basketball-other") else 6)*3600+120
    screen = [] if everything else [e for e in todo if e[2] not in rows]
    jobs = [(anchor_of(fam, e)-3600, [m["ticker"] for m in (e[4] if len(e[4]) < 3 else e[4]) if not m["ticker"].endswith("-TIE")][:1 if len(e[4]) < 3 else 3])
            for e in screen]
    lookup = {j[1][0]: e for j, e in zip(jobs, screen)}
    dropped = set()
    for group in chunk(jobs, 4*3600, 3600):
        if time.time() > deadline:
            cover["hourly screen cut by the clock"] += len(group)
            dropped.update(lookup[t[0]][2] for _, t in group)
            continue
        got = batch(client, group, 4*3600, 60)
        if got is None:
            dropped.update(lookup[t[0]][2] for _, t in group)
            continue
        for anchor, tickers in group:
            e = lookup[tickers[0]]
            best = max((hourly_favourite([b for b in got.get(t, []) if anchor < b["end_period_ts"] <= anchor+4*3600], len(e[4]) >= 3), t)
                       for t in tickers)
            if best[0] < 75:
                rows[e[2]] = dict(event=e[2], skip="no_bars" if best[0] == 0 else "hourly screen: no favourite at 75c+")
                cover["screened out"] += 1
            else:
                e[4][0]["_candidate"] = best[1]
        keep(fam, rows)
    jobs, lookup = [], {}
    for e in todo:
        if e[2] in dropped or (e[2] in rows and rows[e[2]].get("skip")):
            continue
        if e[2] in rows:
            ticker = rows[e[2]]["bar_ticker"]
        else:
            ticker = e[4][0].get("_candidate") or e[4][0]["ticker"]
        jobs.append((anchor_of(fam, e), [ticker]))
        lookup[ticker] = e
    for n, group in enumerate(chunk(jobs, span, 60)):
        if time.time() > deadline:
            cover["one-minute bars cut by the clock"] += len(group)
            continue
        got = batch(client, group, span, 1)
        if got is None:
            continue
        for anchor, tickers in group:
            e = lookup[tickers[0]]
            bars = [b for b in got.get(tickers[0], []) if anchor < b["end_period_ts"] <= min(anchor+span, e[0])]
            complete = anchor+span >= min(e[0], (start_of(fam, e) or e[0])+5*3600)
            market = next(m for m in e[4] if m["ticker"] == tickers[0])
            rows[e[2]] = study(fam, (e[0], e[1], e[2], e[3], [market]+[m for m in e[4] if m is not market] if len(e[4]) < 3 else e[4]),
                               {tickers[0]: bars}, complete)
            cover["built"] += 1
        if n % 5 == 4:
            keep(fam, rows)
    keep(fam, rows)
    print(f"{fam:18} {dict(cover)} rows {sum(1 for r in rows.values() if not r.get('skip'))} {dict(used)}", flush=True)


def order_key(event):
    return hashlib.sha256(event.encode()).hexdigest()


def cbb_archive(client, deadline):
    """Last season's men's college basketball from /historical: one read of the canonical market's six-hour window per game."""
    fam, rows, cover = "cbb-men", store("cbb-men"), Counter()
    events = {k: v for k, v in events_of("KXNCAAMBGAME", archive=True).items() if v[3]}
    short = {r["event"]: r for r in json.loads((CALIB/"rows-college-basketball-men.json").read_text())}
    todo = []
    for e in events.values():
        cover["listed"] += 1
        if not binary(e):
            cover["not settled at 0 or 1"] += 1
            continue
        if e[2] in rows and (rows[e[2]].get("skip") or rows[e[2]].get("full") or rows[e[2]]["start"] < 80):
            continue
        row = offline(fam, e)
        if row is not None:
            rows[e[2]] = row
            cover["from earlier caches"] += 1
            continue
        old = short.get(e[2])
        if old is not None and e[2] not in rows:      # calibration_map.py priced it from a 150-minute read
            m = next(x for x in e[4] if x["ticker"] == old["ticker"])
            yes = old["payout"] == m["_outcome"]
            fav = m if yes else next(x for x in e[4] if x["ticker"] != m["ticker"])
            rows[e[2]] = dict(base(fam, e, fav, old["start"], old["spread"], old["payout"], old["warm_end"], old["open_interest"],
                                   old["warm_volume"]), full=False, bar_ticker=m["ticker"], yes=yes)
            cover["priced by calibration_map"] += 1
            if old["start"] < 80:
                continue
        todo.append(e)
    keep(fam, rows)
    todo.sort(key=lambda e: (e[2] not in rows, order_key(e[2])))      # favourites that only lack the in-play bars first
    print(f"cbb archive: {dict(cover)}; to read {len(todo)} ({sum(e[2] in rows for e in todo)} already priced at 80c+)", flush=True)
    for n, e in enumerate(todo):
        if time.time() > deadline:
            cover["cut by the clock"] = len(todo)-n
            break
        m = e[4][0]
        path, params = f"/historical/markets/{m['ticker']}/candlesticks", dict(start_ts=e[0]-WINDOW, end_ts=e[0], period_interval=1)
        try:
            bars = sorted(net(client, path, params).get("candlesticks") or [], key=lambda b: b["end_period_ts"])
        except RuntimeError as exc:
            cover["failed"] += 1
            print("   read failed", exc, flush=True)
            continue
        rows[e[2]] = study(fam, e, {m["ticker"]: bars}, True)
        cover["read"] += 1
        if n % 100 == 99:
            keep(fam, rows)
            print(f"   {n+1} of {len(todo)} {dict(used)}", flush=True)
    keep(fam, rows)
    print(f"cbb archive done: {dict(cover)} {dict(used)}", flush=True)


def rebuild():
    """Offline: recompute the college rows from the cached one-minute reads (after a change to study()). Rows built from a
    batch request (the newest games) are kept as they are."""
    for fam, series in (("cbb-men", "KXNCAAMBGAME"), ("cbb-women", "KXNCAAWBGAME")):
        rows, cover = store(fam), Counter()
        for e in events_of(series, archive=True).values():
            old = rows.get(e[2])
            if old is None or old.get("skip") or not old.get("full"):
                continue
            new = offline(fam, e)
            if new is None:
                cover["kept (built from a batch request)"] += 1
            else:
                rows[e[2]] = new
                cover["recomputed"] += 1
        keep(fam, rows)
        print(fam, dict(cover))


def fetch_listings(client, deadline):
    """Recent-tier listings of every series in families() that is not cached yet, and the series' fee types."""
    fees_path = OUT/"fees.json"
    fees = json.loads(fees_path.read_text()) if fees_path.exists() else {}
    for fam, serieses in families().items():
        for series in serieses:
            if time.time() > deadline:
                print("listings cut by the clock at", series)
                break
            try:
                before = used["requests"]
                events = events_of(series, client)
                if series not in fees and (fam.startswith("cbb") or series in FEE_CHECK):
                    fees[series] = net(client, f"/series/{series}", {}).get("series", {}).get("fee_type")
                if used["requests"] == before:
                    time.sleep(0)      # everything came from the cache
            except RuntimeError as exc:
                print("  listing failed", series, exc, flush=True)
                continue
            time.sleep(PACE)      # calibration_map.listing paces itself; this keeps the two pacers from overlapping
        print(f"{fam:18} series {len(serieses)}", flush=True)
    OUT.mkdir(parents=True, exist_ok=True)
    fees_path.write_text(json.dumps(fees, indent=1))


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("cmd", choices=("listings", "cbb", "others", "rebuild", "report"))
    parser.add_argument("--minutes", type=float, default=9)
    parser.add_argument("--families", nargs="+", default=["soccer", "esports", "cricket", "basketball-other"])
    parser.add_argument("--no-archive", action="store_true")
    args = parser.parse_args()
    if args.cmd == "report":
        return report()
    if args.cmd == "rebuild":
        return rebuild()
    deadline = time.time()+60*args.minutes
    with httpx.Client(timeout=60) as client:
        if args.cmd == "listings":
            fetch_listings(client, deadline)
        elif args.cmd == "cbb":
            events_of("KXNCAAMBGAME", client), events_of("KXNCAAWBGAME", client)
            recent(client, "cbb-men", ["KXNCAAMBGAME"], deadline, everything=True)
            recent(client, "cbb-women", ["KXNCAAWBGAME"], deadline, everything=True)
            if not args.no_archive:
                cbb_archive(client, deadline)
        else:
            fams = families()
            for fam in args.families:
                recent(client, fam, fams[fam], deadline)
    print("network use:", dict(used))


# ---------------------------------------------------------------- report

BANDS = (("90-99", 90, 99), ("80-90", 80, 90))
MAKER = 1.75


def fee(p, k=7.0):
    return k*p/100*(1-p/100)


def load():
    rows, drop = [], Counter()
    for fam in families():
        for r in store(fam).values():
            if r.get("skip"):
                drop[fam, r["skip"]] += 1
                continue
            if fam.startswith("cbb") and r.get("full") and r.get("begin") and 80 <= r.get("tip_price", 0) < 99:
                limit = 80 if r["tip_price"] >= 90 else math.floor(r["tip_price"])-10
                rows.append(dict(r, family=fam+" at tip", start=r["tip_price"], spread=r["tip_spread"], limit=limit,
                                 fill=r["low_in_play"] is not None and r["low_in_play"] < limit))
            if r["start"] >= 99 or r["start"] < 80:
                drop[fam, "price outside 80-99c"] += 1
            elif not r.get("begin") or r["warm_end"] > r["begin"]:
                drop[fam, "warm-up not before the start"] += 1
            else:
                rows.append(r)
    rows.sort(key=lambda r: (r["settle"], r["event"]))
    return rows, drop


def walk_forward(fam, series, rows):
    """Adds to each college row what was known before its window opened: how often each team had been listed, the team's
    mean open interest, and its record against its own prices (cents per game, favourite or not)."""
    events = sorted(events_of(series, archive=True).values(), key=lambda e: e[0])
    priced = {r["event"]: r for r in store(fam).values() if not r.get("skip")}
    listed, record, attention = Counter(), defaultdict(list), defaultdict(list)
    i, by_event, history = 0, {r["event"]: r for r in rows}, []
    for e in events:
        r = by_event.get(e[2])
        if r is None:
            continue
        while events[i][0] < e[0]-WINDOW:
            old = events[i]
            codes = [m["ticker"].rsplit("-", 1)[-1] for m in old[4]]
            for c in codes:
                listed[c] += 1
            p = priced.get(old[2])
            if p is not None:
                resid = 100*p["payout"]-p["start"]
                for c in codes:
                    record[c].append(resid if c == p["fav_code"] else -resid)
                    attention[c].append(p["open_interest"])
                history.append((p["open_interest"], listed[p["fav_code"]], float(np.mean(attention[p["fav_code"]]))))
            i += 1
        opp = next(m["ticker"].rsplit("-", 1)[-1] for m in e[4] if not m["ticker"].endswith("-"+r["fav_code"]))
        last = history[-300:]
        if len(last) >= 50:
            med = np.median(np.array(last), axis=0)
            r["oi_high"] = r["open_interest"] > med[0]
            r["listed_high"] = listed[r["fav_code"]] > med[1]
            if attention[r["fav_code"]]:
                r["team_oi_high"] = float(np.mean(attention[r["fav_code"]])) > med[2]
        for who, code in (("fav_record", r["fav_code"]), ("opp_record", opp)):
            if len(record[code]) >= 3:
                r[who] = float(np.mean(record[code]))


def eastern(r):
    return datetime.fromtimestamp(r["begin"], EASTERN)


def phase(r):
    d = eastern(r)
    if d.month == 11:
        return "November"
    if d.month == 12:
        return "December"
    if d.month in (1, 2) or (d.month == 3 and d.day <= 8):
        return "January to 8 March (conference play)"
    if d.month == 3 and d.day <= 15:
        return "9-15 March (conference tournaments)"
    return "16 March on (national tournaments)"


def soccer_type(r):
    code = r["series"][2:-4]
    for name, codes in SOCCER_TYPES.items():
        if code in codes.split():
            return name
    return "other top league"


TOP_TIER = re.compile(r"worlds|major|iem\b|esl pro league|blast|pgl|vct|the international|esports world cup|rlcs world|msi\b|"
                      r"dreamleague season|riyadh|\blck\b|\blpl\b|\blec\b|six invitational", re.I)
QUALIFIER = re.compile(r"qualif|play-in|last chance|promotion", re.I)
INTERNATIONAL = re.compile(r"tour of|t20i|odis|world cup|qualifier|tri-nation| vs |asian games|icc|continent cup|quadrangular|test series", re.I)


BASKETBALL = {"continental or regional cup": "EUROLEAGUE EUROCUP FIBACHAMPLEAGUE FIBAECUP FIBA WFIBA FIBACHAMP ABA BNXT",
              "Asia-Pacific league": "CBA NBL JBLEAGUE KBL NZNBL PBA VBA", "Americas league": "LNBP NBB BSN ARGLNB SPB LBP"}


def contexts(r):
    """Every (group, label) this row belongs to. Nothing here uses the result or anything known only after the start."""
    fam, out = r["family"], []
    add = lambda group, label: out.append((group, label)) if label is not None else None
    if fam.startswith("cbb"):
        d = eastern(r)
        add("site", None if r["fav_first"] is None else "favourite listed second (home)" if not r["fav_first"] else "favourite listed first (away)")
        add("phase", phase(r))
        add("season", "season 2025-26" if r["settle"] < midnight(date(2026, 7, 1)) else "season 2026-27")
        add("day", "Saturday or Sunday" if d.weekday() >= 5 else "Monday to Friday")
        add("tip", "tip before 5pm ET" if d.hour < 17 else "tip 5pm to 8:59pm ET" if d.hour < 21 else "tip 9pm ET or later")
        for key, hi, lo in (("oi_high", "open interest above the running median", "open interest below the running median"),
                            ("listed_high", "favourite listed more often than usual", "favourite listed less often than usual"),
                            ("team_oi_high", "favourite's past games drew above-median interest", "favourite's past games drew below-median interest")):
            if key in r:
                add(key, hi if r[key] else lo)
        if "fav_record" in r:
            add("fav_record", "favourite has beaten its prices so far" if r["fav_record"] > 0 else "favourite has fallen short of its prices so far")
        if "opp_record" in r:
            add("opp_record", "underdog has beaten its prices so far" if r["opp_record"] > 0 else "underdog has fallen short of its prices so far")
    elif fam == "soccer":
        add("type", soccer_type(r))
        add("site", None if r["fav_first"] is None else "favourite listed first (home)" if r["fav_first"] else "favourite listed second (away)")
        add("competition", r["series"][2:-4])
    elif fam == "esports":
        add("game", r["series"][2:-4])
        name = r["text"].split(": ")[0]
        add("tier", "top-tier event" if TOP_TIER.search(name) else "lower-tier event")
        add("stage", "qualifier, play-in or promotion" if QUALIFIER.search(name) else None)
        add("start", "scheduled start on the hour" if eastern(r).minute == 0 else "scheduled start off the hour")
    else:
        add("competition", r["series"][2:].replace("GAME", "").replace("MATCH", ""))
        if fam == "cricket":
            add("type", "women" if "women" in r["text"].lower() else "international or qualifier" if INTERNATIONAL.search(r["text"]) else "men's domestic")
        if fam == "basketball-other":
            code = r["series"][2:-4]
            add("region", next((name for name, codes in BASKETBALL.items() if code in codes.split()), "Europe, domestic league"))
            add("site", None if r["fav_first"] is None else "favourite listed first" if r["fav_first"] else "favourite listed second")
    add("price", "90-95c" if 90 <= r["start"] < 95 else "95-99c" if r["start"] >= 95 else None)
    add("three-way", None)
    return out


class Table:
    """Arrays for one family x band, with the cell statistics."""

    def __init__(self, rows, cut):
        self.rows, self.cut = rows, cut
        self.p = np.array([r["start"] for r in rows])
        self.y = np.array([r["payout"] for r in rows])
        self.spread = np.array([r["spread"] for r in rows])
        self.first = np.array([r["settle"] < cut for r in rows])
        teams = {}
        self.team = np.array([teams.setdefault(r["fav_code"], len(teams)) for r in rows])
        self.gap = float((100*self.y-self.p).mean()) if rows else 0.0
        self.fill = np.array([bool(r.get("full") and r.get("fill")) for r in rows])
        self.limit = np.array([float(r.get("limit", 0)) for r in rows])
        self.fill_win = float(self.y[self.fill].mean()) if self.fill.any() else 0.0

    def dip_test(self, mask, y=None):
        """(net per fill, standard error, passes?) for the dip bid: first-half interval above zero and second half positive.
        The standard error is never below the one implied by the family's own fill win rate."""
        y = self.y if y is None else y
        out = []
        for part in (np.ones(len(self.p), bool), self.first, ~self.first):
            m = mask & part & self.fill
            n = int(m.sum())
            if n < 8:
                out.append(None)
                continue
            net = 100*y[m]-self.limit[m]
            out.append((float(net.mean()), max(float(net.std(ddof=1)), 100*math.sqrt(self.fill_win*(1-self.fill_win)))/math.sqrt(n), n))
        ok = bool(out[1] and out[2] and out[1][0]-1.96*out[1][1] > 0 and out[2][0] > 0)
        return out[0], ok

    def dip_chance(self, masks, sims=1000):
        """Dip cells passing when every fill in the family wins with the family's pooled fill win rate."""
        counts = np.zeros(sims, int)
        for s in range(sims):
            y = (RNG.random(len(self.p)) < self.fill_win).astype(float)
            counts[s] = sum(self.dip_test(mask, y)[1] for mask in masks)
        return counts

    def stats(self, mask, y=None):
        y = self.y if y is None else y
        n = int(mask.sum())
        if n == 0:
            return None
        p, yy = self.p[mask], y[mask]
        resid = 100*yy-p
        gap = resid.mean()
        se = max(resid.std(ddof=1)/math.sqrt(n) if n > 1 else 0, 100*math.sqrt((p/100*(1-p/100)).sum())/n,
                 math.sqrt((np.bincount(self.team[mask], resid-gap)**2).sum())/n)
        return dict(n=n, price=float(p.mean()), win=float(100*yy.mean()), gap=float(gap), se=float(se), lo=float(gap-1.96*se), hi=float(gap+1.96*se))

    def cell(self, mask):
        out = self.stats(mask)
        out["h1"], out["h2"] = self.stats(mask & self.first), self.stats(mask & ~self.first)
        out["passes"] = bool(out["h1"] and out["h2"] and out["h1"]["n"] >= 10 and out["h2"]["n"] >= 10 and out["h1"]["lo"] > 0 and out["h2"]["gap"] > 0)
        cost = self.p[mask]+self.spread[mask]/2
        net = 100*self.y[mask]-cost-fee(cost)
        out.update(ask_net=float(net.mean()), ask_pct=float(100*net.mean()/(cost+fee(cost)).mean()), spread=float(self.spread[mask].mean()))
        rest = self.stats(~mask)
        out["contrast"] = {}
        for name, part in (("all", np.ones(len(self.p), bool)), ("h1", self.first), ("h2", ~self.first)):
            a, b = self.stats(mask & part), self.stats(~mask & part)
            if a and b and a["n"] >= 10 and b["n"] >= 10:
                se = math.hypot(a["se"], b["se"])
                out["contrast"][name] = dict(diff=a["gap"]-b["gap"], se=se, z=(a["gap"]-b["gap"])/se)
        c = out["contrast"]
        out["contrast_passes"] = bool("h1" in c and "h2" in c and c["h1"]["z"]*c["h2"]["z"] > 0 and min(abs(c["h1"]["z"]), abs(c["h2"]["z"])) > 1.96)
        out.update(dip(np.array(self.rows, dtype=object)[mask], self.cut), volume=capacity(np.array(self.rows, dtype=object)[mask]))
        out["dip_all"], out["dip_passes"] = self.dip_test(mask)
        return out

    def chance(self, masks, sims=1000):
        """Number of cells passing when every favourite wins with probability price + the family's pooled gap."""
        prob = np.clip((self.p+self.gap)/100, 0, 1)
        counts = np.zeros(sims, int)
        var = self.p/100*(1-self.p/100)
        for s in range(sims):
            y = (RNG.random(len(prob)) < prob).astype(float)
            resid = 100*y-self.p
            for mask in masks:
                ok = True
                for part, need_ci in ((self.first, True), (~self.first, False)):
                    m = mask & part
                    n = m.sum()
                    if n < 10:
                        ok = False
                        break
                    g = resid[m].mean()
                    if need_ci:
                        se = max(resid[m].std(ddof=1)/math.sqrt(n), 100*math.sqrt(var[m].sum())/n)
                        ok = g-1.96*se > 0
                    else:
                        ok = g > 0
                    if not ok:
                        break
                counts[s] += ok
        return counts


def boot(values, draws=2000):
    values = np.asarray(values, float)
    if len(values) < 10:
        return None
    means = np.sort(RNG.choice(values, (draws, len(values))).mean(axis=1))
    return [float(means[int(.025*draws)]), float(means[int(.975*draws)])]


def dip(rows, cut):
    orders = [r for r in rows if r.get("full") and "limit" in r]
    fills = [r for r in orders if r["fill"]]
    out = dict(dip_orders=len(orders), dip_fills=len(fills))
    if fills:
        nets = np.array([100*r["payout"]-r["limit"] for r in fills])
        limit = float(np.mean([r["limit"] for r in fills]))
        halves = [[100*r["payout"]-r["limit"] for r in fills if (r["settle"] < cut) == first] for first in (True, False)]
        out.update(dip_net=float(nets.mean()), dip_ci=boot(nets), dip_win=float(100*np.mean([r["payout"] for r in fills])), dip_limit=limit,
                   dip_pct=float(100*nets.mean()/limit), dip_net_maker_fee=float(nets.mean()-fee(limit, MAKER)),
                   dip_per_order=float(nets.sum()/len(orders)), dip_halves=[(len(h), float(np.mean(h)) if h else None) for h in halves])
    return out


def capacity(rows):
    recent = [r for r in rows if not r["archived"]]
    if not recent:
        return dict(per_month=0.0)
    days = max(1.0, (max(r["settle"] for r in recent)-min(r["settle"] for r in recent))/86400)
    pre = [r["pre_volume"] for r in recent if "pre_volume" in r]
    return dict(per_month=len(recent)/max(days, 14)*30, median_open_interest=float(np.median([r["open_interest"] for r in recent])),
                median_pre_event_volume=float(np.median(pre)) if pre else None)


def halves(c):
    return " / ".join(f'{n} fills {"%+.1f" % g if g is not None else "n/a"}' for n, g in c["dip_halves"])


def line(label, c):
    half = lambda h: f'{h["n"]:4} {h["gap"]:+5.1f} [{h["lo"]:+5.1f},{h["hi"]:+5.1f}]' if h else "   0                    "
    z = c["contrast"].get("all")
    return (f'  {label[:50]:50} n {c["n"]:4} price {c["price"]:4.1f} win {c["win"]:5.1f} gap {c["gap"]:+5.1f} [{c["lo"]:+5.1f},{c["hi"]:+5.1f}] | '
            f'{half(c["h1"])} | {half(c["h2"])} | ask {c["ask_net"]:+5.2f}c {c["ask_pct"]:+5.2f}% | vs rest {z["diff"] if z else float("nan"):+5.1f} z {z["z"] if z else float("nan"):+4.1f}'
            f'{"  PASS" if c["passes"] else ""}{"  CONTRAST" if c["contrast_passes"] else ""}')


def dip_line(label, c):
    if not c["dip_orders"]:
        return None
    if not c["dip_fills"]:
        return f'  {label[:50]:50} orders {c["dip_orders"]:4} fills    0'
    ci = "[%+5.1f,%+5.1f]" % tuple(c["dip_ci"]) if c["dip_ci"] else "[ too few fills]"
    h = " | ".join(f'{n:3} {"%+5.1f" % g if g is not None else "     "}' for n, g in c["dip_halves"])
    return (f'  {label[:50]:50} orders {c["dip_orders"]:4} fills {c["dip_fills"]:4} ({c["dip_fills"]/c["dip_orders"]:4.0%}) win {c["dip_win"]:5.1f} '
            f'net/fill {c["dip_net"]:+5.1f}c {ci} = {c["dip_pct"]:+5.1f}% on {c["dip_limit"]:.0f}c | maker fee {c["dip_net_maker_fee"]:+5.1f}c | '
            f'per order {c["dip_per_order"]:+5.2f}c | halves {h}')


def report():
    rows, drop = load()
    lines = []
    say = lambda text="": (lines.append(text), print(text))
    for fam, series in (("cbb-men", "KXNCAAMBGAME"), ("cbb-women", "KXNCAAWBGAME")):
        walk_forward(fam, series, [r for r in rows if r["family"] == fam])
        walk_forward(fam, series, [r for r in rows if r["family"] == fam+" at tip"])
    fees = json.loads((OUT/"fees.json").read_text()) if (OUT/"fees.json").exists() else {}
    say(f"fee types: {fees}")
    iso = lambda t: datetime.fromtimestamp(t, timezone.utc).isoformat()[:10]
    cells, summary, chance_total, observed_total = [], {}, np.zeros(1000, int), 0
    dip_chance_total, dip_observed, dip_cells = np.zeros(1000, int), 0, 0
    for fam in ["cbb-men", "cbb-men at tip", "cbb-women", "cbb-women at tip"]+list(families())[2:]:
        mine = [r for r in rows if r["family"] == fam]
        if not mine:
            continue
        cut = float(np.median([r["settle"] for r in mine]))
        say(f"\n{'='*20} {fam}: {len(mine)} favourites at 80-99c, {iso(mine[0]['settle'])}..{iso(mine[-1]['settle'])}, halves split at {iso(cut)}; "
            f"full in-play bars for {sum(bool(r.get('full')) for r in mine)}")
        say(f"  dropped: { {k[1]: v for k, v in drop.items() if k[0] == fam} }")
        for band, lo, hi in BANDS+((("80-99", 80, 99),) if fam == "soccer" else ()):
            sub = [r for r in mine if lo <= r["start"] < hi]
            if len(sub) < 20:
                continue
            t = Table(sub, cut)
            whole = t.cell(np.ones(len(sub), bool))
            say(f"\n  -- {fam} {band}c: pooled")
            say(line("ALL", whole))
            text = dip_line("ALL", whole)
            if text:
                say("  dip bid:")
                say(text)
            groups = defaultdict(lambda: np.zeros(len(sub), bool))
            for i, r in enumerate(sub):
                for key in contexts(r):
                    if band == "90-99" or key[0] != "price":
                        groups[key][i] = True
            kept = {k: m for k, m in groups.items() if m.sum() >= 30 and (~m).sum() >= 1}
            say(f"  -- contexts with 30+ events ({len(kept)}; {len(groups)-len(kept)} smaller ones not tested)")
            for key in sorted(kept):
                c = t.cell(kept[key])
                c.update(family=fam, band=band, group=key[0], label=key[1], family_gap=whole["gap"], family_se=whole["se"])
                cells.append(c)
                say(line(f"{key[0]}: {key[1]}", c))
            dips = [dip_line(f"{k[0]}: {k[1]}", c) for k, c in ((k, next(x for x in cells if x["family"] == fam and x["band"] == band and (x["group"], x["label"]) == k)) for k in sorted(kept))]
            if any(dips):
                say("  -- dip bid by context")
                for text in dips:
                    if text:
                        say(text)
            if kept:
                sim = t.chance(list(kept.values()))
                got = sum(c["passes"] for c in cells if c["family"] == fam and c["band"] == band)
                chance_total += sim
                observed_total += got
                say(f"  multiplicity {fam} {band}: {len(kept)} cells, {got} pass; under 'every favourite has the pooled gap {whole['gap']:+.1f}c' "
                    f"{sim.mean():.1f} pass on average (95% range {np.percentile(sim, 2.5):.0f}-{np.percentile(sim, 97.5):.0f}), "
                    f"P(at least {got}) = {(sim >= got).mean():.2f}")
            tested = [np.ones(len(sub), bool)]+[m for m in kept.values() if (m & t.fill).sum() >= 16]
            mine_cells = [whole]+[c for c in cells if c["family"] == fam and c["band"] == band and c["dip_fills"] >= 16]
            if whole["dip_fills"] >= 16:
                sim, got = t.dip_chance(tested), sum(c["dip_passes"] for c in mine_cells)
                dip_chance_total += sim
                dip_observed += got
                dip_cells += len(tested)
                say(f"  dip multiplicity {fam} {band}: {len(tested)} cells with 16+ fills (the pooled one included), {got} pass; if every fill wins "
                    f"at the family's rate ({100*t.fill_win:.1f}%, net {100*t.fill_win-t.limit[t.fill].mean():+.1f}c) {sim.mean():.1f} pass on average, "
                    f"P(at least {got}) = {(sim >= got).mean():.2f}")
            whole.update(family=fam, band=band, group="", label="ALL")
            summary[f"{fam} {band}"] = whole
    # ---- multiplicity over everything, and shrinkage
    say(f"\n{'='*20} MULTIPLICITY")
    n_con = sum(1 for c in cells if "h1" in c["contrast"] and "h2" in c["contrast"])
    say(f"cells examined {len(cells)}; pass (first-half interval above zero and second half positive): {observed_total}; expected if contexts add nothing "
        f"to their family's pooled gap: {chance_total.mean():.1f} (95% range {np.percentile(chance_total, 2.5):.0f}-{np.percentile(chance_total, 97.5):.0f}), "
        f"P(at least {observed_total}) = {(chance_total >= observed_total).mean():.2f}")
    say(f"contrasts with the rest of the family, same sign and |z|>1.96 in both halves: {sum(c['contrast_passes'] for c in cells)} of {n_con} testable; "
        f"expected by chance {2*.025**2*n_con:.2f}. |z|>1.96 pooled: {sum(abs(c['contrast']['all']['z']) > 1.96 for c in cells if 'all' in c['contrast'])} of "
        f"{sum('all' in c['contrast'] for c in cells)}; expected by chance {.05*sum('all' in c['contrast'] for c in cells):.1f}")
    say(f"dip-bid cells with 16+ fills: {dip_cells}; pass (first-half interval above zero, second half positive): {dip_observed}; expected if a context "
        f"adds nothing to its family's fill win rate: {dip_chance_total.mean():.1f} (95% range {np.percentile(dip_chance_total, 2.5):.0f}-"
        f"{np.percentile(dip_chance_total, 97.5):.0f}), P(at least {dip_observed}) = {(dip_chance_total >= dip_observed).mean():.2f}")
    # the dip bid across families (90-99c favourites, 80c bid): partial pooling of the family means
    fams = [v for k, v in summary.items() if k.endswith("90-99") and v["dip_all"] and v["family"] not in ("cbb-men", "cbb-women")]
    m, se = np.array([v["dip_all"][0] for v in fams]), np.array([v["dip_all"][1] for v in fams])
    w = 1/se**2
    centre = float((w*m).sum()/w.sum())
    between = max(0.0, float((w*((m-centre)**2-se**2)).sum()/w.sum()))
    say(f"dip bid at 80c on 90-99c favourites, family by family (college basketball on the price at the tip): precision-weighted mean {centre:+.1f}c (se {1/math.sqrt(w.sum()):.1f}), "
        f"between-family sd {math.sqrt(between):.1f}c")
    for v, mi, si in zip(fams, m, se):
        v["dip_shrunk"] = float((mi/si**2+centre/between)/(1/si**2+1/between)) if between else centre
        say(f"  {v['family']:18} orders {v['dip_orders']:4} fills {v['dip_fills']:4} ({v['dip_fills']/v['dip_orders']:4.0%}) win {v['dip_win']:5.1f} "
            f"net/fill {mi:+5.1f}c (se {si:.1f}) = {100*mi/80:+5.1f}% | halves {halves(v)} | shrunk {v['dip_shrunk']:+5.1f}c = {100*v['dip_shrunk']/80:+5.1f}%, "
            f"with maker fee {v['dip_shrunk']-fee(80, MAKER):+5.1f}c | per order {v['dip_shrunk']*v['dip_fills']/v['dip_orders']:+5.2f}c | "
            f"{v['volume'].get('per_month', 0):4.0f} orders/month, {v['volume'].get('per_month', 0)*v['dip_fills']/v['dip_orders']:4.0f} fills/month | median open interest "
            f"{v['volume'].get('median_open_interest') or 0:.0f}, pre-event volume {v['volume'].get('median_pre_event_volume') or 0:.0f} contracts")
    say("dip cells that pass:")
    for c in list(summary.values())+cells:
        if c["dip_passes"]:
            say(f"  {c['family']:16} {c['band']} {c['group']+': '+c['label']:55.55} fills {c['dip_fills']:4} net {c['dip_net']:+5.1f}c "
                f"[{c['dip_ci'][0]:+.1f},{c['dip_ci'][1]:+.1f}] halves {halves(c)}")
    use = [c for c in cells if "all" in c["contrast"]]
    d, se = np.array([c["contrast"]["all"]["diff"] for c in use]), np.array([c["contrast"]["all"]["se"] for c in use])
    w = 1/se**2
    tau2 = max(0.0, float((w*(d**2-se**2)).sum()/w.sum()))
    say(f"shrinkage: spread of true context effects (sd) estimated at {math.sqrt(tau2):.2f}c from {len(use)} contrasts")
    prior = {}
    for band in ("90-99", "80-90", "80-99"):      # family gaps are shrunk towards the mean of these families, between-family sd 0.81c (calibration_map report)
        fams = [v for k, v in summary.items() if k.endswith(band) and v["family"] not in ("cbb-men", "cbb-women")]
        if not fams:
            continue
        prior[band] = sum(v["gap"]/v["se"]**2 for v in fams)/sum(1/v["se"]**2 for v in fams)
    for c in use:
        fam_shrunk = (c["family_gap"]/c["family_se"]**2+prior[c["band"]]/.81**2)/(1/c["family_se"]**2+1/.81**2)
        share = c["n"]/summary[f'{c["family"]} {c["band"]}']["n"]
        c["shrunk_contrast"] = float(c["contrast"]["all"]["diff"]*tau2/(tau2+c["contrast"]["all"]["se"]**2))
        c["forward_gap"] = float(fam_shrunk+(1-share)*c["shrunk_contrast"])      # gap_in = family + (1-share) x (in - out)
        c["forward_ask_net"] = c["forward_gap"]-c["spread"]/2-fee(c["price"]+c["spread"]/2)
        c["family_shrunk"] = float(fam_shrunk)
    say(f"family gaps shrunk towards {prior} (precision-weighted mean of these families):")
    for k, v in summary.items():
        band = k.split()[-1]
        s = (v["gap"]/v["se"]**2+prior[band]/.81**2)/(1/v["se"]**2+1/.81**2)
        v["shrunk"] = s
        say(f"  {k:26} n {v['n']:5} raw {v['gap']:+5.2f} [{v['lo']:+5.2f},{v['hi']:+5.2f}] halves {v['h1']['gap'] if v['h1'] else float('nan'):+5.2f} / "
            f"{v['h2']['gap'] if v['h2'] else float('nan'):+5.2f} shrunk {s:+5.2f} ask net {v['ask_net']:+5.2f}c ({v['ask_pct']:+5.2f}%) "
            f"{v['volume'].get('per_month', 0):6.0f}/month")
    say(f"\n{'='*20} CELLS THAT PASS, by honest forward expectation (shrunk gap; buy-at-ask net = gap - half spread - taker fee)")
    for c in sorted((c for c in use if c["passes"]), key=lambda c: -c["forward_ask_net"]):
        v = c["volume"]
        say(f'  {c["family"]:16} {c["band"]} {c["group"]+": "+c["label"]:58.58} n {c["n"]:4} raw {c["gap"]:+5.1f} family {c["family_gap"]:+5.1f} -> forward gap '
            f'{c["forward_gap"]:+5.2f}c, at the ask {c["forward_ask_net"]:+5.2f}c ({100*c["forward_ask_net"]/c["price"]:+5.2f}%) | {v["per_month"]:5.0f}/month '
            f'open interest {v.get("median_open_interest") or 0:8.0f} pre-event volume {v.get("median_pre_event_volume") or 0:8.0f}'
            f'{"  CONTRAST" if c["contrast_passes"] else ""}')
    say(f"\n{'='*20} LARGEST CONTRASTS (|z| pooled), whether or not they pass")
    for c in sorted(use, key=lambda c: -abs(c["contrast"]["all"]["z"]))[:25]:
        k = c["contrast"]
        say(f'  {c["family"]:16} {c["band"]} {c["group"]+": "+c["label"]:58.58} n {c["n"]:4} vs rest {k["all"]["diff"]:+5.1f} z {k["all"]["z"]:+4.1f} | halves '
            f'{k["h1"]["diff"] if "h1" in k else float("nan"):+5.1f} (z {k["h1"]["z"] if "h1" in k else float("nan"):+4.1f}) / '
            f'{k["h2"]["diff"] if "h2" in k else float("nan"):+5.1f} (z {k["h2"]["z"] if "h2" in k else float("nan"):+4.1f}) shrunk {c["shrunk_contrast"]:+5.2f}')
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT/"report.txt").write_text("\n".join(lines)+"\n")
    strip = lambda c: {k: v for k, v in c.items()}
    (OUT/"summary.json").write_text(json.dumps(dict(families=summary, cells=[strip(c) for c in cells], tau=math.sqrt(tau2)), indent=1, default=float))


if __name__ == "__main__":
    main()
