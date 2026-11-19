"""Does the 80c dip bid on heavy favourites extend to other college and lower-tier sports (to fill spring and summer)?

The edge being extended (research/favourite_contexts.py, research/wcbb_backfill.py): a team priced 90-99c just before
the start in a low-attention, mismatch-heavy competition wins more often than its price, and a bid resting at 80c during
play, held to settlement, pays (men's and women's college basketball, first-round tennis qualifying). College
basketball stops in early April, so this script looks at college baseball and softball, Summer League, spring
training, college hockey / volleyball / lacrosse / FCS football, the AHL, and overseas baseball.

Commands

    inventory   listings only (one read per 1,000 markets): every candidate series, settled games by month, contracts
                per game, fee type, where the start time can come from. Listings carry no pre-game price (the last and
                previous prices are post-settlement), so the share of heavy favourites cannot be read from them.
    probe       a small uniform sample of games per family (sha256 order of the event ticker), one-minute bars of the
                canonical market over the six hours before settlement: the share of games with a usable two-sided
                pre-game quote and the share of 90c+ favourites. The same reads serve the backfill.
    download    more of one family, same order, so that a run cut short leaves a uniform sample of the season.
    build       cache -> rows.json (offline).
    peek        one line per family: games read, share with a clean pre-game price, heavy favourites, fills.
    trades      the public trade record around every 80c fill on a 90-99c favourite (one read per fill).
    report      per family: tradeability, start-time check, pre-game gap, buy at the ask, resting bids at 85, 80, 75,
                70 and price-10, halves, months, the pass test of wcbb_backfill.py, sanity checks; then every family
                pooled (decided before any result was seen) and shrunk with the families studied earlier.
    cbb         offline: men's college basketball pre-game calibration in 5c bands (the mild-underdog question).

Conventions (those of research/wcbb_backfill.py, whose functions are called directly)

* One market per game, the alphabetically first; the favourite is the side whose pre-game midpoint is above 50c.
  Games are kept at their settlement value, including the few settled away from 0 or 1 (postponed or cancelled
  games settle at 50c or at a last price); the tables on 0-or-1 games only are shown as a check.
* Pre-game price ("tip-only", what the live rule sees): the median midpoint of the last 10-15 one-minute bars ending
  MARGIN minutes before the start, each two-sided with a spread of at most 4c, all within 3c. Prices of 99c and
  above are excluded.
* Start: from the ticker when it carries a time, else Kalshi's expected expiration minus the family's allowance, else
  the close minus the family's typical length. The estimate is checked against the first visible price movement
  (wcbb_backfill.onset) and every dip table is repeated with the start moved.
* A resting bid fills when a one-minute bar ending more than a minute after the start has a trade strictly below it.
  Fixed bids must sit at least 5c below the price. All series here have fee type "quadratic" (no maker fee): net per
  fill is 100 x settlement - bid. Buying at the ask pays midpoint + half the spread + 0.07 p (1-p).
* Pass test: halves by settlement over the orders; first-half interval above zero and second half positive.

Network: public reads only, strictly one at a time, at least 2.6 seconds apart, 30 seconds or more after any HTTP 429
(wcbb_backfill.read). Responses are cached under data/raw/cost-research-v1. No orders, no authentication.

Result of the run of 18 November 2026 (about 1,030 requests in about 45 minutes of downloading, one HTTP 429).
Read: college baseball 161 games in sha256 order (2% of 7,280) and all 128 from 25 May (the postseason); all 12
softball games listed; all 94 Summer League; lacrosse 46 in sha256 order and all 48 from 27 April; all 218 AFL; all 76
World Baseball Classic; all 77 IIHF; all 16 FCS playoff games; volleyball 671 of 1,161 (everything settled since 18
September; the archived first half of September was not read). Not read: KBO, NPB, Mexican league, spring training,
AHL, college hockey, CFL, UFL (earlier rows on disk show no 90c favourite in professional baseball, hockey or these
football leagues), PLL, TBT, Big3, Unrivaled, CEBL, NRL (too few games).

* No spring or summer family passes. College baseball's regular season has no book (clean pre-game price in 4% of
  games; median spread 45c) and baseball does not make 90c favourites (0 of 79 Summer League, 2 of 67 baseball
  postseason, 0 of 3 softball priced games). The dip rule needs both a quoted book and frequent 90c+ favourites, and
  only AFL (March-September: 25 orders, 7 fills, 7 won) and college volleyball (September-December: 53 orders, 8
  fills, 8 won, about 10-25 contracts a fill) have them. Pooled over every family read: 101 favourites at 90-99c
  won 98.0% at 94.6c; 80c bid 23 fills, 91.3% won, +11.3c a fill, bootstrap -1.7 to +20.0, first half +7.5c (-11.2
  to +20.0), second +20.0c: the pass test fails for want of fills, not because of a loss.
* Men's college basketball: mild underdogs at 35-50c won 47.2% at 42.9c three hours out (1,322 games, +4.3 points,
  Wilson +1.6 to +7.0) and 47.3% at 42.5c at the tip (583 games, +4.8, +0.8 to +8.9), in both seasons, both halves,
  and equally in the months read completely and the months sampled at 30%, so it is not an artefact of which games
  had bars. Buying them at the ask after the fee returned +5.6% of cost (+2.5c, bootstrap -1.3 to +6.7): not
  distinguishable from zero. Women's games show nothing (-0.4 on 269).

    .venv/bin/python research/college_sports.py inventory --minutes 8
    .venv/bin/python research/college_sports.py probe --minutes 9
    .venv/bin/python research/college_sports.py download --family afl --cap 160 --minutes 9
    .venv/bin/python research/college_sports.py build
    .venv/bin/python research/college_sports.py peek
    .venv/bin/python research/college_sports.py trades --minutes 4
    .venv/bin/python research/college_sports.py report
    .venv/bin/python research/college_sports.py cbb
"""
import argparse
from collections import Counter, defaultdict
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys
import time

import httpx
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from research.api_cost import ROOT
from research.calibration_map import LIVE_B, cache_file, listing, quote, scheduled, ts, volume
from research.near_end import WINDOW, cents
from research import favourite_contexts as fc
from research import wcbb_backfill as wb
from research.dip_new_families import measure, shrink, wilson

OUT = ROOT / "data/processed/college-sports"
PACE = 2.6
CUTOFF = 1789689600      # 2026-09-18: earlier settlements are served from /historical

# family -> series, typical length in minutes (close minus start), allowance (expected expiration minus start, hours),
# months of the season, and what it is.
FAMILIES = {
    "college-baseball": dict(series=["KXNCAABBGAME"], length=190, season="Feb-Jun", prefix=True),
    "college-baseball-postseason": dict(series=["KXNCAABBGAME"], length=190, season="late May-Jun", since=1779681600),      # from 25 May 2026
    "college-softball": dict(series=["KXNCAASBGAME"], length=140, season="Feb-Jun"),
    "summer-league": dict(series=["KXNBASUMMERGAME"], length=125, season="Jul", expected=True),
    "spring-training": dict(series=["KXMLBSTGAME"], length=185, season="Feb-Mar"),
    # Lacrosse: no schedule in the listing. The close falls about 195 minutes after the start (a two-hour game and, as in
    # college baseball where the schedule is known, about an hour before the market is closed): with 135 the first price
    # movement came a median 51 minutes BEFORE the estimated start, against 2 to 15 minutes after it where the start is known.
    "college-lacrosse": dict(series=["KXNCAAMLAXGAME"], length=195, season="Feb-May", prefix=True),
    "college-lacrosse-postseason": dict(series=["KXNCAAMLAXGAME"], length=195, season="late Apr-May", since=1777262400),     # from 27 April 2026
    "college-hockey": dict(series=["KXNCAAHOCKEYGAME", "KXNCAAHGAME"], length=150, season="Oct-Apr"),
    "college-volleyball": dict(series=["KXNCAAWVMATCH"], length=120, season="Sep-Dec", span=7),
    "fcs-football": dict(series=["KXNCAAFCSGAME", "KXNCAAFD3GAME"], length=200, season="Sep-Dec", expected=True),
    "college-soccer": dict(series=["KXNCAAMSOCCERGAME"], length=125, season="Aug-Dec"),
    "ahl": dict(series=["KXAHLGAME"], length=155, season="Oct-Jun"),
    "ufl": dict(series=["KXUFLGAME"], length=190, season="Mar-Jun"),
    "cfl": dict(series=["KXCFLGAME"], length=185, season="Jun-Nov"),
    "lacrosse-pro": dict(series=["KXPLLGAME"], length=130, season="Jun-Sep"),
    "summer-basketball": dict(series=["KXTBTGAME", "KXBIG3GAME", "KXUNRIVALEDGAME", "KXCEBLGAME"], length=125, season="Jan-Mar, Jun-Aug"),
    "kbo": dict(series=["KXKBOGAME"], length=195, season="Mar-Oct"),
    "npb": dict(series=["KXNPBGAME"], length=200, season="Mar-Oct"),
    "mexican-baseball": dict(series=["KXLMBGAME", "KXLMPGAME"], length=195, season="Apr-Sep, Oct-Jan"),
    "wbc": dict(series=["KXWBCGAME"], length=195, season="Mar (every 3 years)"),
    "winter-baseball": dict(series=["KXLIDOMGAME", "KXLVBPGAME"], length=195, season="Oct-Jan"),
    "hockey-europe": dict(series=["KXKHLGAME", "KXSHLGAME", "KXLIIGAGAME"], length=150, season="Sep-May"),
    "iihf": dict(series=["KXIIHFGAME"], length=150, season="May (worlds), Dec-Jan (juniors)"),
    "afl": dict(series=["KXAFLGAME"], length=150, season="Mar-Sep"),
    "nrl": dict(series=["KXRUGBYNRLMATCH"], length=110, season="Mar-Oct"),
    "fiba-women": dict(series=["KXWFIBAGAME"], length=120, season="summer windows"),
}
MARGIN = 5*60      # the pre-game quote ends this long before the start


def iso(t):
    return datetime.fromtimestamp(t, timezone.utc).isoformat()[:10]


def month_of(t):
    return datetime.fromtimestamp(t, fc.EASTERN).strftime("%Y-%m")


def order_key(event):
    return hashlib.sha256(event.encode()).hexdigest()


# ---------------------------------------------------------------- listings

def fetch_pages(client, path, params):
    cursor, seen, n = "", set(), 0
    while True:
        data = wb.read(client, path, dict(params, cursor=cursor) if cursor else params)
        n += len(data.get("markets") or [])
        cursor = data.get("cursor")
        if not cursor or cursor in seen:
            return n
        seen.add(cursor)


def events_of(series):
    """Settled two-market events of a series from cached listings: (settle, series, event, archived, markets)."""
    return [e for e in listing(None, series).values() if len(e[4]) == 2]


def all_events(series):
    return list(listing(None, series).values())


def start_of(fam, e):
    """(start, source). Ticker time; else, in families where Kalshi's expected expiration proved a fixed three hours after the
    start (summer league, FCS), expected expiration minus three hours when that lands 30-360 minutes before the close; else
    the close minus the family's typical length."""
    sched = scheduled(e[2])
    if sched:
        return sched, "ticker"
    m = e[4][0]
    close = ts(m, "close_time")
    close = min(close, e[0]) if close else e[0]-300
    expected = ts(m, "expected_expiration_time")
    if FAMILIES[fam].get("expected") and expected and 1800 <= close-(expected-3*3600) <= 6*3600:
        return expected-3*3600, "expected"
    return close-60*FAMILIES[fam]["length"], "close"


def inventory(minutes):
    deadline = time.time()+60*minutes
    fees = {s["ticker"]: (s.get("fee_type"), s.get("fee_multiplier"), float(s.get("volume_fp") or 0))
            for s in json.loads(cache_file("/series", dict(category="Sports", include_volume="true")).read_text())["series"]}
    with httpx.Client(timeout=60) as client:
        for fam, spec in FAMILIES.items():
            for s in spec["series"]:
                if time.time() > deadline:
                    break
                try:
                    fetch_pages(client, "/historical/markets", dict(series_ticker=s, limit=1000))
                    fetch_pages(client, "/markets", dict(series_ticker=s, **LIVE_B))
                except RuntimeError as exc:
                    print("   listing failed", s, exc, flush=True)
    print(f"network: {dict(wb.used)}")
    out, lines = {}, []
    for fam, spec in FAMILIES.items():
        for s in spec["series"]:
            evs = all_events(s)
            two = [e for e in evs if len(e[4]) == 2]
            if not evs:
                lines.append(f"{fam:20} {s:20} no settled markets listed; lifetime contracts {fees.get(s, (0, 0, 0))[2]/1e3:.0f}k")
                out[s] = dict(family=fam, games=0)
                continue
            vol = np.array([sum(float(m.get("volume_fp") or 0) for m in e[4]) for e in two] or [0])
            months = Counter(month_of(e[0]) for e in two)
            frac = sum(not fc.binary(e) for e in two)
            src = Counter(start_of(fam, e)[1] for e in two)
            gaps = [(ts(e[4][0], "expected_expiration_time")-ts(e[4][0], "close_time"))/60 for e in two
                    if ts(e[4][0], "expected_expiration_time") and ts(e[4][0], "close_time")]
            late = sum(1 for e in two if ts(e[4][0], "close_time") and e[0]-ts(e[4][0], "close_time") > 3600)
            out[s] = dict(family=fam, games=len(two), events=len(evs), archived=sum(e[3] for e in two), months=dict(sorted(months.items())),
                          median_contracts=float(np.median(vol)), p25=float(np.percentile(vol, 25)), p75=float(np.percentile(vol, 75)),
                          fee=fees.get(s, (None, None))[:2], fractional=frac, start_source=dict(src),
                          expected_minus_close=[float(np.percentile(gaps, q)) for q in (10, 50, 90)] if gaps else None, late=late,
                          season=spec["season"])
            lines.append(f"{fam:20} {s:20} games {len(two):5} (archive {out[s]['archived']:5}) contracts/game median {np.median(vol):9.0f} "
                         f"[{np.percentile(vol, 25):.0f}-{np.percentile(vol, 75):.0f}] fee {fees.get(s, ('?', '?'))[0]} x{fees.get(s, ('?', '?'))[1]} "
                         f"fractional {frac} late-settled {late} start from {dict(src)} season {spec['season']}")
            lines.append(f"{'':41} by month: " + " ".join(f"{k[2:]}:{v}" for k, v in sorted(months.items())))
            if gaps:
                lines.append(f"{'':41} expected expiration minus close (minutes): 10th {out[s]['expected_minus_close'][0]:.0f}, median "
                             f"{out[s]['expected_minus_close'][1]:.0f}, 90th {out[s]['expected_minus_close'][2]:.0f}")
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT/"inventory.json").write_text(json.dumps(out, indent=1))
    (OUT/"inventory.txt").write_text("\n".join(lines)+"\n")
    print("\n".join(lines))



# ---------------------------------------------------------------- bars

# Batch window of a recent game: three hours before the scheduled start to nine hours after (FAMILIES[...]["span"] hours overrides).


def games(fam, archived=None):
    out = [e for s in FAMILIES[fam]["series"] for e in events_of(s) if archived is None or e[3] == archived]
    since = FAMILIES[fam].get("since")
    if since:
        out = [e for e in out if start_of(fam, e)[0] >= since]
    return sorted(out, key=lambda e: order_key(e[2]))


def span_of(fam):
    return 3600*FAMILIES[fam].get("span", 12)


def request(e):
    """The archive read of one game: three hours before the start in the ticker to nine hours after it (or settlement, when
    earlier); without a ticker time, the six hours before settlement (wcbb_backfill's window)."""
    m, sched = e[4][0], scheduled(e[2])
    path = f"/historical/markets/{m['ticker']}/candlesticks"
    if sched and sched+1800 < e[0]:
        return path, dict(start_ts=sched-3*3600, end_ts=min(e[0], sched+9*3600), period_interval=1)
    return path, dict(start_ts=e[0]-WINDOW, end_ts=e[0], period_interval=1)


def recent_requests(fam):
    """Batch requests (GET /markets/candlesticks) for games settled since 2026-09-18."""
    jobs = []
    for e in games(fam, False):
        sched = scheduled(e[2])
        jobs.append(((sched-3*3600) if sched and sched+1800 < e[0] else e[0]-WINDOW, [e[4][0]["ticker"]]))
    return [("/markets/candlesticks", dict(market_tickers=",".join(t for _, ts_ in g for t in ts_), start_ts=g[0][0], end_ts=g[-1][0]+span_of(fam),
                                           period_interval=1)) for g in fc.chunk(jobs, span_of(fam), 60)]


def download(minutes, family, cap):
    """Up to `cap` more archived games of one family (sha256 order), then its recent games through the batch endpoint."""
    deadline = time.time()+60*minutes
    todo = [e for e in games(family, True) if not cache_file(*request(e)).exists()]
    have = len(games(family, True))-len(todo)
    print(f"{family}: archived games {len(games(family, True))}, read {have}, taking up to {cap} more", flush=True)
    with httpx.Client(timeout=60) as client:
        for e in todo[:cap]:
            if time.time() > deadline:
                break
            try:
                wb.read(client, *request(e))
            except RuntimeError as exc:
                wb.used["failed"] += 1
                print("   read failed", e[2], exc, flush=True)
                if wb.used["failed"] >= 5:
                    break
        for r in sorted(recent_requests(family), key=lambda r: order_key(r[1]["market_tickers"])):      # a uniform subset when cut short
            if time.time() > deadline:
                break
            if not cache_file(*r).exists():
                try:
                    wb.read(client, *r)
                    time.sleep(2)
                except RuntimeError as exc:
                    print("   batch failed", exc, flush=True)
    print(f"done: {dict(wb.used)}; archived games read {sum(cache_file(*request(e)).exists() for e in games(family, True))}; "
          f"recent batches missing {sum(not cache_file(*r).exists() for r in recent_requests(family))}")


def probe(minutes, family, cap):
    deadline = time.time()+60*minutes
    for fam in ([family] if family else PROBE):
        download(max(0.1, (deadline-time.time())/60), fam, cap)


PROBE = ("college-baseball", "college-lacrosse", "afl", "summer-league")


def compact(bars):
    """[end of minute, yes bid, yes ask, low, high, contracts] per one-minute bar."""
    out = []
    for b in sorted(bars, key=lambda b: b["end_period_ts"]):
        q = quote(b)
        out.append([b["end_period_ts"], q[0] if q else None, q[1] if q else None, cents(b, "price", "low"), cents(b, "price", "high"), volume(b)])
    return out


def build():
    rows, cover = [], Counter()
    for fam in FAMILIES:
        batches = {}
        for r in recent_requests(fam):
            if cache_file(*r).exists():
                for x in json.loads(cache_file(*r).read_text()).get("markets", []):
                    batches[x["market_ticker"]] = x.get("candlesticks") or []
        for e in games(fam):
            cover[fam, "listed"] += 1
            m = e[4][0]
            if e[3]:
                file = cache_file(*request(e))
                if not file.exists():
                    if FAMILIES[fam].get("prefix"):
                        break      # the uniform sample is the leading run of the sha256 order; the postseason is read in full separately
                    continue
                bars = json.loads(file.read_text()).get("candlesticks") or []
            else:
                if m["ticker"] not in batches:
                    continue
                sched = scheduled(e[2])
                anchor = (sched-3*3600) if sched and sched+1800 < e[0] else e[0]-WINDOW
                bars = [b for b in batches[m["ticker"]] if anchor < b["end_period_ts"] <= min(anchor+span_of(fam), e[0])]
            cover[fam, "read"] += 1
            start, src = start_of(fam, e)
            close = ts(m, "close_time")
            rows.append(dict(family=fam, series=e[1], event=e[2], ticker=m["ticker"], settle=e[0], archived=e[3], start=start, src=src, close=close,
                             outcome=m["_outcome"], binary=fc.binary(e), late=bool(close and e[0]-close > 3600),
                             contracts=sum(float(x.get("volume_fp") or 0) for x in e[4]), codes=[x["ticker"].rsplit("-", 1)[-1] for x in e[4]],
                             bars=compact(bars)))
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT/"rows.json").write_text(json.dumps(rows))
    cov = defaultdict(dict)
    for (fam, k), v in cover.items():
        cov[fam][k] = v
    (OUT/"coverage.json").write_text(json.dumps(cov, indent=1))
    for fam, c in cov.items():
        if c.get("read"):
            print(fam, c)


# ---------------------------------------------------------------- views

def view(r, shift=0, least=10):
    """The game as the rule sees it with the start moved by `shift` minutes, or None when there is no clean pre-game quote:
    the last 10-15 bars ending two minutes before a scheduled start (25 before an estimated one), no older than four hours,
    each two-sided with a spread of at most 4c, midpoints within 3c."""
    start = r["start"]+60*shift
    margin = 25*60 if r["src"] == "close" else 120
    pre = [b for b in r["bars"] if start-4*3600 <= b[0] <= start-margin][-wb.WARMUP:]
    if len(pre) < least or any(b[1] is None or b[2]-b[1] > 4 for b in pre):
        return None
    mids = [(b[1]+b[2])/2 for b in pre]
    if max(mids)-min(mids) > 3:
        return None
    mid = float(np.median(mids))
    yes = mid > 50
    price = mid if yes else 100-mid
    if price >= 99:
        return None
    plays = []
    for b in r["bars"]:
        if b[3] is None or b[4] is None or b[0] <= start-margin:
            continue
        m = None if b[1] is None else ((b[1]+b[2])/2 if yes else 100-(b[1]+b[2])/2)
        plays.append([b[0], b[3] if yes else 100-b[4], b[4] if yes else 100-b[3], b[5], m])
    ask = pre[-1][2] if yes else 100-pre[-1][1]
    return dict(price=price, spread=float(np.mean([b[2]-b[1] for b in pre])), ask=ask, payout=r["outcome"] if yes else 1-r["outcome"], plays=plays, yes=yes,
                est_start=start, event=r["event"], ticker=r["ticker"], settle=r["settle"], season=r["family"], month=month_of(start),
                archived=r["archived"], binary=r["binary"], late=r["late"], total_volume=r["contracts"], fav_code=r["codes"][0 if yes else 1],
                quote_age=(start-pre[0][0])/60, src=r["src"],
                before=min((x[1] for x in plays if x[0] <= start+60), default=None))


def views(rows, fam, shift=0, least=10):
    out = [v for r in rows if r["family"] == fam for v in [view(r, shift, least)] if v]
    out.sort(key=lambda v: (v["settle"], v["event"]))
    return out


def fee(p):
    return 7*p/100*(1-p/100)


def ask_cell(vs):
    """Buying the favourite at its last pre-game ask, taker fee paid: (n, net cents, percent of cost, bootstrap interval)."""
    if not vs:
        return None
    cost = np.array([v["ask"]+fee(v["ask"]) for v in vs])
    net = np.array([100*v["payout"] for v in vs])-cost
    return dict(n=len(vs), cost=float(cost.mean()), net=float(net.mean()), pct=float(100*net.mean()/cost.mean()), boot=wb.boot(list(net)))


def load():
    return json.loads((OUT/"rows.json").read_text())


def peek():
    """Short summary per family: tradeability and the share of heavy favourites."""
    rows = load()
    for fam in FAMILIES:
        mine = [r for r in rows if r["family"] == fam]
        if not mine:
            continue
        vs, loose = views(rows, fam), views(rows, fam, least=3)
        n = len(mine)
        nb = np.array([len(r["bars"]) for r in mine])
        say = (f"{fam:18} read {n:4} bars/game median {np.median(nb):4.0f} | clean pre-game quote {len(vs):4} ({len(vs)/n:4.0%}), with 3+ bars {len(loose):4} ({len(loose)/n:4.0%}) | "
               f"90-99c {sum(90 <= v['price'] < 99 for v in vs):3} ({sum(90 <= v['price'] < 99 for v in vs)/max(1, len(vs)):4.0%}) 80-90c {sum(80 <= v['price'] < 90 for v in vs):3} "
               f"({sum(80 <= v['price'] < 90 for v in vs)/max(1, len(vs)):4.0%}) | 80c fills on 90c+: {sum(wb.fill_of(v, 80) is not None for v in vs if 90 <= v['price'] < 99)}")
        print(say)



# ---------------------------------------------------------------- men's college basketball calibration

def cal_line(rs, key, flip=False):
    """Team-price cell: n, mean price, win rate, gap, Wilson interval on the gap, z against the prices' own variance."""
    if not rs:
        return "n    0"
    p = np.array([100-r[key] if flip else r[key] for r in rs])
    y = np.array([1-r["payout"] if flip else r["payout"] for r in rs])
    n = len(rs)
    w = wilson(y.sum(), n)
    se = 100*np.sqrt((p/100*(1-p/100)).sum())/n
    return (f"n {n:4} price {p.mean():5.1f}c win {100*y.mean():5.1f}% gap {100*y.mean()-p.mean():+5.1f} [{100*w[0]-p.mean():+5.1f},{100*w[1]-p.mean():+5.1f}] "
            f"z {(100*y.mean()-p.mean())/se:+5.2f}")


def cbb():
    """Pre-game calibration of men's college basketball from the rows favourite_contexts.py left on disk (no network)."""
    lines = []

    def say(text=""):
        print(text)
        lines.append(text)
    rows = json.loads((fc.OUT/"rows-cbb-men.json").read_text())
    ev = fc.events_of("KXNCAAMBGAME", archive=True)
    mo = lambda t: datetime.fromtimestamp(t, fc.EASTERN).strftime("%Y-%m")
    ok = {k: r for k, r in rows.items() if not r.get("skip") and r["payout"] in (0.0, 1.0)}
    tip = [r for r in ok.values() if "tip_price" in r and r["tip_price"] < 99]
    early = [r for r in ok.values() if r["warm_end"] <= r["begin"] and r["start"] < 99]
    say("=== MEN'S COLLEGE BASKETBALL: COVERAGE OF THE ROWS ON DISK (listing: every settled game)")
    listed, have, tipped, priced = (Counter(mo(e[0]) for e in ev.values()), Counter(mo(ev[k][0]) for k in rows if k in ev),
                                    Counter(mo(r["settle"]) for r in tip), Counter(mo(r["settle"]) for r in early))
    for m in sorted(listed):
        say(f"  {m}: listed {listed[m]:5} bars read {have[m]:5} ({have[m]/listed[m]:4.0%}) priced three hours out {priced[m]:5} ({priced[m]/listed[m]:4.0%}) "
            f"priced at the tip {tipped[m]:5} ({tipped[m]/listed[m]:4.0%})")
    say(f"  all: listed {len(ev)}, bars read {len(rows)} ({len(rows)/len(ev):.0%}), priced three hours out {len(early)}, priced at the tip {len(tip)}")
    complete = {m for m in listed if have[m]/listed[m] > .95}
    say(f"  months read (almost) completely: {sorted(complete)}; the others were read in a uniform pseudo-random order (about 30%)")
    for key, name, rs in (("tip_price", "AT THE TIP (last quotes before play)", tip), ("start", "ABOUT THREE HOURS BEFORE THE TIP", early)):
        rs = sorted(rs, key=lambda r: r["settle"])
        say(f"\n=== CALIBRATION {name}: {len(rs)} games. Each game is one favourite (50c+) and one underdog (the mirror image),")
        say("    so the underdog rows repeat the favourite rows with the sign changed. Gap = win rate minus price; Wilson interval; z uses p(1-p).")
        for lo in range(5, 95, 5):
            hi = lo+5
            flip = lo < 50
            band = [r for r in rs if (100-hi < r[key] <= 100-lo if flip else lo <= r[key] < hi)]
            if lo == 5:
                band = [r for r in rs if 90 < r[key] <= 99]
            if lo == 90:
                band = [r for r in rs if 90 <= r[key] < 99]
            a, b = band[:len(band)//2], band[len(band)//2:]
            say(f"  team at {1 if lo == 5 else lo:2}-{99 if lo == 90 else hi:2}c: {cal_line(band, key, flip)} | 2025-26 {cal_line([r for r in band if r['archived']], key, flip)[:43]} | "
                f"Nov 2026 {cal_line([r for r in band if not r['archived']], key, flip)[:43]}")
            say(f"                  first half {cal_line(a, key, flip)[:43]} | second half {cal_line(b, key, flip)[:43]}")
        z = []
        for lo in range(50, 95, 5):
            band = [r for r in rs if lo <= r[key] < (99 if lo == 90 else lo+5)]
            p = np.array([r[key] for r in band])
            y = np.array([r["payout"] for r in band])
            z.append(float((100*y.mean()-p.mean())/(100*np.sqrt((p/100*(1-p/100)).sum())/len(band))))
        say(f"  z of the nine favourite bands: {' '.join(f'{x:+.1f}' for x in z)}; sum of squares {sum(x*x for x in z):.1f} (9 expected under perfect pricing; 16.9 is the 5% point)")
        mild = [r for r in rs if 50 < r[key] <= 65]
        say(f"\n  -- MILD UNDERDOGS 35-50c ({name.lower()})")
        parts = [("all", mild), ("2025-26", [r for r in mild if r["archived"]]), ("November 2026", [r for r in mild if not r["archived"]]),
                 ("first half", mild[:len(mild)//2]), ("second half", mild[len(mild)//2:]),
                 ("months read completely", [r for r in mild if mo(r["settle"]) in complete]),
                 ("months read about 30%", [r for r in mild if mo(r["settle"]) not in complete])]
        parts += [(m, [r for r in mild if mo(r["settle"]) == m]) for m in sorted({mo(r["settle"]) for r in mild})]
        for label, part in parts:
            say(f"     {label:26} {cal_line(part, key, True)}")
        # reweighted to the listing's month mix
        gaps = {m: np.mean([100*(1-r["payout"])-(100-r[key]) for r in mild if mo(r["settle"]) == m]) for m in listed if any(mo(r["settle"]) == m for r in mild)}
        tot = sum(listed[m] for m in gaps)
        say(f"     weighted by each month's share of all listed games: gap {sum(gaps[m]*listed[m] for m in gaps)/tot:+.1f}")
        if key == "tip_price":
            cost = np.array([100-r[key]+r["tip_spread"]/2 for r in mild])
            cost_fee = cost+np.array([fee(c) for c in cost])
            pay = np.array([100*(1-r["payout"]) for r in mild])
            for label, m in (("all", np.ones(len(mild), bool)), ("2025-26", np.array([r["archived"] for r in mild])), ("November 2026", np.array([not r["archived"] for r in mild])),
                             ("first half", np.arange(len(mild)) < len(mild)//2), ("second half", np.arange(len(mild)) >= len(mild)//2)):
                net = (pay-cost_fee)[m]
                b = wb.boot(list(net))
                say(f"     BUY THE UNDERDOG AT THE ASK, fee paid, {label:14}: n {int(m.sum()):4} cost {cost_fee[m].mean():5.1f}c (ask {cost[m].mean():.1f} + fee {(cost_fee-cost)[m].mean():.1f}) "
                    f"won {pay[m].mean():5.1f}% net {net.mean():+5.1f}c = {100*net.mean()/cost_fee[m].mean():+5.1f}% of cost, bootstrap [{b[0]:+.1f},{b[1]:+.1f}]c")
    # ---- do games with bars differ from games without?
    say("\n=== DO GAMES WITH BARS DIFFER FROM GAMES WITHOUT? (the listing holds no pre-game price: last and previous prices are post-settlement)")

    def home(e):
        """1 when the home team won, None when the title is not 'A at B'."""
        title = e[4][0].get("title") or ""
        if " at " not in title:
            return None
        name = title.split(" at ", 1)[1].replace(" Winner?", "").strip()
        m = next((x for x in e[4] if (x.get("yes_sub_title") or "").strip() == name), None)
        return None if m is None or m["_outcome"] not in (0.0, 1.0) else m["_outcome"]
    vol = lambda e: sum(float(m.get("volume_fp") or 0) for m in e[4])
    for label, months in (("months read about 30% (Nov 2025 - Jan 2026)", [m for m in listed if m not in complete]), ("all months", list(listed))):
        a = [e for k, e in ev.items() if k in rows and mo(e[0]) in months]
        b = [e for k, e in ev.items() if k not in rows and mo(e[0]) in months]
        for name, part in (("with bars", a), ("without bars", b)):
            h = [x for x in (home(e) for e in part) if x is not None]
            w = wilson(sum(h), len(h)) if h else (0, 0)
            say(f"  {label}: {name:12} games {len(part):5} median contracts {np.median([vol(e) for e in part]) if part else 0:9.0f}; home team won {100*np.mean(h) if h else 0:5.1f}% "
                f"[{100*w[0]:.1f},{100*w[1]:.1f}] of {len(h)} 'A at B' games")
    both = [r for r in early if 50 < r["start"] <= 65]
    say(f"  within the games read, mild underdogs three hours out: with a tip price too {cal_line([r for r in both if 'tip_price' in r], 'start', True)}")
    say(f"                                                         without a tip price  {cal_line([r for r in both if 'tip_price' not in r], 'start', True)}")
    lo_v = np.median([r["open_interest"] for r in both])
    say(f"  by open interest three hours out (known before the game): below the median {cal_line([r for r in both if r['open_interest'] < lo_v], 'start', True)}")
    say(f"                                                            above the median {cal_line([r for r in both if r['open_interest'] >= lo_v], 'start', True)}")
    lo_v = np.median([r["market_volume"] for r in both])
    say("  by contracts traded over the market's whole life (NOT known before the game: close games and upsets trade more, so this split looks ahead):")
    say(f"                                                            below the median {cal_line([r for r in both if r['market_volume'] < lo_v], 'start', True)}")
    say(f"                                                            above the median {cal_line([r for r in both if r['market_volume'] >= lo_v], 'start', True)}")
    # ---- the women's games as an outside check
    path = ROOT/"data/processed/wcbb-backfill/rows.json"
    if path.exists():
        w_rows = [r for r in json.loads(path.read_text()) if "alt_price" in r and r["binary"] and r["alt_price"] < 99]
        say("\n=== OUTSIDE CHECK: WOMEN'S COLLEGE BASKETBALL (wcbb_backfill rows, price at the tip), the same bands")
        for lo, hi in ((35, 50), (20, 35), (5, 20)):
            band = [dict(p=r["alt_price"], payout=r["alt_payout"]) for r in w_rows if 100-hi < r["alt_price"] <= 100-lo]
            say(f"  team at {lo}-{hi}c: {cal_line(band, 'p', True)}")
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT/"cbb-calibration.txt").write_text("\n".join(lines)+"\n")



# ---------------------------------------------------------------- report

BANDS = (("90-99c", 90, 99), ("80-90c (control)", 80, 90))


def onset_of(r):
    """Minutes from the start to the first minute at which the midpoint has travelled 4c within ten minutes (wcbb_backfill.onset on compact bars)."""
    path, last = [], None
    for b in r["bars"]:
        if b[1] is None or b[2]-b[1] > 6 or b[0] < r["start"]-7200:
            continue
        mid = (b[1]+b[2])/2
        path.append((b[0], abs(mid-last) if last is not None else 0.0))
        last = mid
        if sum(d for t, d in path if t > b[0]-600) >= 4:
            return (b[0]-r["start"])/60
    return None


def report_family(rows, fam, say):
    mine = [r for r in rows if r["family"] == fam]
    n = len(mine)
    listed = games(fam)
    vs = views(rows, fam)
    summary = dict(read=n, listed=len(listed), priced=len(vs))
    say(f"\n{'#'*110}\n# {fam.upper()}: {', '.join(FAMILIES[fam]['series'])}; season {FAMILIES[fam]['season']}\n{'#'*110}")
    # ---- coverage
    lm, rm = Counter(month_of(start_of(fam, e)[0]) for e in listed), Counter(month_of(r["start"]) for r in mine)
    say(f"COVERAGE: games listed {len(listed)}, read {n} ({n/len(listed):.0%}), a uniform sample in sha256 order of the event ticker")
    say("  " + "  ".join(f"{m}: {rm[m]}/{lm[m]} ({rm[m]/lm[m]:.0%})" for m in sorted(lm)))
    # ---- tradeability
    nb = np.array([len(r["bars"]) for r in mine])
    traded_pre = [sum(b[5] for b in r["bars"] if b[0] <= r["start"]) for r in mine]
    quoted_pre = [sum(1 for b in r["bars"] if b[0] <= r["start"] and b[1] is not None and b[2]-b[1] <= 4) for r in mine]
    loose = views(rows, fam, least=3)
    say(f"TRADEABLE AT ALL? bars per game median {np.median(nb):.0f}; games with no bar at all {int((nb == 0).sum())} ({(nb == 0).mean():.0%}); "
        f"no contract traded before the start {np.mean(np.array(traded_pre) == 0):.0%}; median contracts over the market's life {np.median([r['contracts'] for r in mine]):.0f} (both sides)")
    say(f"  two-sided quote of 4c or tighter in 10+ pre-game minutes {np.mean(np.array(quoted_pre) >= 10):.0%}; CLEAN PRE-GAME PRICE (the rule's definition) {len(vs)} ({len(vs)/n:.0%}); "
        f"with 3+ bars instead of 10+ {len(loose)} ({len(loose)/n:.0%})")
    say(f"  of the priced games: favourite 90-99c {sum(90 <= v['price'] < 99 for v in vs)} ({sum(90 <= v['price'] < 99 for v in vs)/max(1, len(vs)):.1%}), "
        f"80-90c {sum(80 <= v['price'] < 90 for v in vs)} ({sum(80 <= v['price'] < 90 for v in vs)/max(1, len(vs)):.1%}), "
        f"70-80c {sum(70 <= v['price'] < 80 for v in vs)}, 50-70c {sum(v['price'] < 70 for v in vs)}; age of the oldest quote used: median {np.median([v['quote_age'] for v in vs]) if vs else 0:.0f} minutes")
    wide = []
    for r in mine:      # a looser look at how common heavy favourites are: the last pre-game quote, any spread up to 10c
        pre = [b for b in r["bars"] if b[0] <= r["start"]-(25*60 if r["src"] == "close" else 120) and b[1] is not None]
        if pre and pre[-1][2]-pre[-1][1] <= 10:
            wide.append(max((pre[-1][1]+pre[-1][2])/2, 100-(pre[-1][1]+pre[-1][2])/2))
    last = [b[2]-b[1] for r in mine for b in [next((b for b in reversed(r["bars"]) if b[0] <= r["start"]-120 and b[1] is not None), None)] if b]
    say(f"  last quote before the start: none in {1-len(last)/n:.0%} of games; spread median {np.median(last) if last else 0:.0f}c, 25th pct {np.percentile(last, 25) if last else 0:.0f}c, "
        f"75th {np.percentile(last, 75) if last else 0:.0f}c. Loose price (that quote when its spread is 10c or less): {len(wide)} games ({len(wide)/n:.0%}), "
        f"favourite 90c+ in {sum(x >= 90 for x in wide)} ({sum(x >= 90 for x in wide)/max(1, len(wide)):.1%}), 80-90c in {sum(80 <= x < 90 for x in wide)} ({sum(80 <= x < 90 for x in wide)/max(1, len(wide)):.1%})")
    summary.update(loose_priced=len(wide), loose_90=sum(x >= 90 for x in wide), loose_80=sum(80 <= x < 90 for x in wide))
    # ---- settlement oddities
    frac = [r for r in mine if not r["binary"]]
    late = [r for r in mine if r["late"]]
    dur = np.array([(r["close"]-r["start"])/60 for r in mine if r["close"] and r["src"] == "ticker"])
    say(f"SETTLEMENT: fractional (not 0 or 1) {len(frac)} of {n} read ({Counter(round(max(r['outcome'], 1-r['outcome']), 2) for r in frac).most_common(6)}); "
        f"settled more than an hour after the close {len(late)}")
    if len(dur):
        say(f"  close minus the scheduled start (minutes): 5th pct {np.percentile(dur, 5):.0f}, median {np.median(dur):.0f}, 95th {np.percentile(dur, 95):.0f}; "
            f"closed before the scheduled start {int((dur < 0).sum())}; more than 10 hours after {int((dur > 600).sum())} (postponed or suspended)")
    # ---- start-time check
    on = np.array([x for x in (onset_of(r) for r in mine) if x is not None])
    src = Counter(r["src"] for r in mine)
    if len(on) >= 5:
        say(f"START TIME: source {dict(src)}. First visible price movement (4c within ten minutes), minutes after the start used: n {len(on)}, 10th pct {np.percentile(on, 10):+.0f}, "
            f"25th {np.percentile(on, 25):+.0f}, median {np.median(on):+.0f}, 75th {np.percentile(on, 75):+.0f}; before the start in {np.mean(on < 0):.0%}, more than 25 minutes before in {np.mean(on < -25):.0%}")
    summary["onset_median"] = float(np.median(on)) if len(on) else None
    # ---- bands
    for name, lo, hi in BANDS:
        sub = wb.band(vs, lo, hi)
        if not sub:
            say(f"\n-- favourites {name}: none")
            continue
        h1, h2 = wb.halves(sub)
        parts = [("all", sub)]
        if len(sub) >= 8:
            parts += [(f"first half (to {iso(h1[-1]['settle'])})", h1), (f"second half (from {iso(h2[0]['settle'])})", h2)]
        months = sorted({v["month"] for v in sub})
        if len(sub) >= 10:
            parts += [(m, [v for v in sub if v["month"] == m]) for m in months]
        say(f"\n-- favourites {name}: pre-game gap (win rate minus price; interval = largest of sample, price-implied and team-clustered errors) and Wilson interval")
        for label, part in parts:
            w = wilson(sum(v["payout"] for v in part), len(part))
            price = np.mean([v["price"] for v in part])
            say(f"     {label:34} {wb.gap_text(wb.gap(part)) if len(part) > 1 else 'n    1'} wilson [{100*w[0]-price:+5.1f},{100*w[1]-price:+5.1f}]")
        a = ask_cell(sub)
        say(f"   buy at the pre-game ask, taker fee paid: n {a['n']} cost {a['cost']:.1f}c net {a['net']:+.2f}c = {a['pct']:+.1f}% of cost"
            + (f", bootstrap [{a['boot'][0]:+.1f},{a['boot'][1]:+.1f}]c" if a["boot"] else "")
            + "".join(f"; {lab} {x['net']:+.1f}c ({x['pct']:+.1f}%)" for lab, x in (("first half", ask_cell(h1)), ("second half", ask_cell(h2))) if x and len(sub) >= 8))
        summary[f"{name} gap"] = wb.gap(sub) if len(sub) > 1 else None
        summary[f"{name} ask"] = a
        for level in wb.LEVELS:
            if not any(wb.limit_of(v, level) is not None for v in sub):
                continue
            say(f"   resting bid {level if level == 'price-10' else level+'c'}")
            for label, part in parts:
                say(f"     {label:34} {wb.fmt(wb.cell(part, level))}")
            c = wb.cell(sub, level)
            if c["fills"] >= 5:
                ca, cb, ok_b, ok_w = wb.pass_test(sub, level)
                d = wb.drop_best_month(sub, level)
                say(f"     PASS TEST (first-half interval above zero, second half positive): bootstrap {'PASS' if ok_b else 'FAIL'}, Wilson {'PASS' if ok_w else 'FAIL'}"
                    + (f" | best month {d['dropped']} dropped ({d['dropped_fills']} fills at {d['dropped_net']:+.1f}c): {d['fills']} fills, net {d['net']:+.1f}c" if d else ""))
                summary[f"{name} bid {level}"] = dict(all=c, first_half=ca, second_half=cb, pass_boot=ok_b, pass_wilson=ok_w)
            else:
                summary[f"{name} bid {level}"] = dict(all=c)
    # ---- sanity checks on the main cell
    main = wb.band(vs, 90, 99)
    fills = [(v, f) for v in main for f in [wb.fill_of(v, 80)] if f]
    say(f"\nSANITY CHECKS: 80c bid on 90-99c favourites ({len(main)} orders, {len(fills)} fills)")
    if fills:
        mins = np.array([f["minute"] for _, f in fills])
        vol = np.array([f["bar_volume"] for _, f in fills])
        b5, w5 = np.array([f["below5"] for _, f in fills]), np.array([f["wholly5"] for _, f in fills])
        say(f"  fill minutes after the start: min {mins.min():.0f}, median {np.median(mins):.0f}, max {mins.max():.0f}; in the first 10 minutes {int((mins <= 10).sum())}")
        say(f"  contracts in the fill minute (all prices): median {np.median(vol):.0f}, 25th pct {np.percentile(vol, 25):.0f}; under 10 in {np.mean(vol < 10):.0%}, under 50 in {np.mean(vol < 50):.0%}")
        say(f"  next five minutes: contracts in bars with a print under 80c: median {np.median(b5):.0f}; in bars wholly under 80c (a lower bound on what the bid could buy): "
            f"median {np.median(w5):.0f}, 25th pct {np.percentile(w5, 25):.0f}, zero in {np.mean(w5 == 0):.0%}, under 10 in {np.mean(w5 < 10):.0%}")
        after = [f["mid_close"]-80 for _, f in fills if f["mid_close"] is not None]
        if after:
            say(f"  favourite's midpoint minus the bid at the end of the fill minute: {np.mean(after):+.1f}c (n {len(after)})")
        say(f"  fills by size of the fill minute: under 10 contracts {wb.fmt(wb.cell([v for v, f in fills if f['bar_volume'] < 10], '80'))}")
        say(f"                                    10 or more        {wb.fmt(wb.cell([v for v, f in fills if f['bar_volume'] >= 10], '80'))}")
    for shift in (-30, -20, -10, 10, 20, 30):
        alt = wb.band(views(rows, fam, shift), 90, 99)
        say(f"  start moved {shift:+d} minutes: {wb.fmt(wb.cell(alt, '80'))}")
    alt = wb.band(loose, 90, 99)
    say(f"  price from 3+ pre-game bars instead of 10+: {wb.fmt(wb.cell(alt, '80'))}")
    early = [v for v in main if v["before"] is not None and v["before"] < 80]
    say(f"  a print under 80c between the pre-game quote and the start (not counted as a fill): {len(early)} of {len(main)} orders")
    fr = [v for v in main if not v["binary"]]
    say(f"  fractional settlements among the orders {len(fr)}, among the fills {sum(not v['binary'] for v, _ in fills)} (kept at their settlement value); "
        f"binary games only: {wb.fmt(wb.cell([v for v in main if v['binary']], '80'))}")
    say(f"  late settlements among the orders {sum(v['late'] for v in main)}, among the fills {sum(v['late'] for v, _ in fills)}")
    # ---- activity
    if main:
        share = n/len(listed)
        season_days = max(14, (max(start_of(fam, e)[0] for e in listed)-min(start_of(fam, e)[0] for e in listed))/86400)
        c = wb.cell(main, "80")
        say(f"ACTIVITY: {len(main)} orders and {c['fills']} fills on {share:.0%} of the games over a season of {season_days:.0f} days = "
            f"{len(main)/share/season_days*30:.0f} orders and {c['fills']/share/season_days*30:.1f} fills per 30 days on every game")
        summary["per_30_days"] = dict(orders=len(main)/share/season_days*30, fills=c["fills"]/share/season_days*30)
    return summary



# ---------------------------------------------------------------- trade record of the fills

def all_fills(rows):
    out = []
    for fam in FAMILIES:
        for v in wb.band(views(rows, fam), 90, 99):
            f = wb.fill_of(v, 80)
            if f:
                out.append((fam, v, f))
    return out


def trades(minutes):
    """The public trade record around every 80c fill on a 90-99c favourite (one read each)."""
    deadline, rows = time.time()+60*minutes, load()
    with httpx.Client(timeout=40) as client:
        for fam, v, f in all_fills(rows):
            if time.time() > deadline:
                break
            try:
                wb.read(client, *wb.tape_requests(v, f, CUTOFF))
            except RuntimeError as exc:
                print("   read failed", v["event"], exc, flush=True)
    print(f"done: {dict(wb.used)}")


def tapes(rows):
    out = []
    for fam, v, f in all_fills(rows):
        file = cache_file(*wb.tape_requests(v, f, CUTOFF))
        if file.exists():
            got = measure(dict(levels={"80": dict(limit=80, ts=f["ts"])}, yes=v["yes"]), json.loads(file.read_text()).get("trades") or [])
            out.append(dict(got, family=fam, event=v["event"], payout=v["payout"], bar_volume=f["bar_volume"]))
    return out


POOL = ("college-baseball-postseason", "college-softball", "college-lacrosse-postseason", "college-lacrosse", "college-volleyball", "fcs-football",
        "summer-league", "iihf", "afl", "wbc")


def pooled(rows, say):
    """Every family read here together (decided before any result was seen): the samples are too small one by one."""
    vs = sorted((dict(v, family=fam) for fam in POOL for v in views(rows, fam)), key=lambda v: (v["settle"], v["event"]))
    seen, uniq = set(), []
    for v in vs:      # a postseason game also sits in its family's uniform sample
        if v["event"] not in seen:
            seen.add(v["event"])
            uniq.append(v)
    vs = uniq
    out = {}
    say(f"\n{'#'*110}\n# ALL NEW FAMILIES POOLED ({', '.join(POOL)}): {len(vs)} priced games\n{'#'*110}")
    for name, lo, hi in BANDS:
        sub = wb.band(vs, lo, hi)
        if len(sub) < 2:
            continue
        say(f"\n-- favourites {name}: {len(sub)} games; by family " + ", ".join(f"{k} {n}" for k, n in Counter(v["family"] for v in sub).most_common()))
        h1, h2 = wb.halves(sub)
        w = wilson(sum(v["payout"] for v in sub), len(sub))
        price = np.mean([v["price"] for v in sub])
        say(f"   pre-game gap: {wb.gap_text(wb.gap(sub))} wilson [{100*w[0]-price:+5.1f},{100*w[1]-price:+5.1f}]; first half {wb.gap_text(wb.gap(h1))}; second half {wb.gap_text(wb.gap(h2))}")
        a = ask_cell(sub)
        say(f"   buy at the pre-game ask, fee paid: cost {a['cost']:.1f}c net {a['net']:+.2f}c = {a['pct']:+.1f}%" + (f" [{a['boot'][0]:+.1f},{a['boot'][1]:+.1f}]c" if a["boot"] else ""))
        for level in wb.LEVELS:
            if not any(wb.limit_of(v, level) is not None for v in sub):
                continue
            c = wb.cell(sub, level)
            say(f"   bid {level:8} all         {wb.fmt(c)}")
            if c["fills"] >= 8:
                ca, cb, ok_b, ok_w = wb.pass_test(sub, level)
                say(f"                first half  {wb.fmt(ca)}")
                say(f"                second half {wb.fmt(cb)}")
                say(f"                PASS TEST: bootstrap {'PASS' if ok_b else 'FAIL'}, Wilson {'PASS' if ok_w else 'FAIL'}")
                out[f"{name} bid {level}"] = dict(all=c, first_half=ca, second_half=cb, pass_boot=ok_b, pass_wilson=ok_w)
        if lo == 90:
            for fam in POOL:
                part = [v for v in sub if v["family"] == fam]
                if part:
                    say(f"   80c bid, {fam:28} {wb.fmt(wb.cell(part, '80'))}")
            liquid = [v for v in sub if v["family"] in ("summer-league", "wbc", "afl", "college-baseball-postseason", "college-softball", "fcs-football")]
            say(f"   80c bid, families with real volume only (summer league, WBC, AFL, baseball and softball postseason, FCS): {wb.fmt(wb.cell(liquid, '80'))}")
            cells = {"new families pooled (this study)": (round(wb.cell(sub, "80").get("win", 0)*wb.cell(sub, "80")["fills"]/100), wb.cell(sub, "80")["fills"], 80)}
            path = fc.OUT/"summary.json"
            if path.exists():
                fams = json.loads(path.read_text())["families"]
                for k in ("cbb-men at tip 90-99", "cbb-women at tip 90-99", "soccer 90-99", "esports 90-99", "basketball-other 90-99"):
                    if k in fams and fams[k].get("dip_fills"):
                        cells[k] = (round(fams[k]["dip_win"]*fams[k]["dip_fills"]/100), fams[k]["dip_fills"], 80)
            if cells["new families pooled (this study)"][1]:
                for k, x in shrink(cells).items():
                    say(f"   shrunk with the families already studied (dip_new_families.shrink): {k:36} fills {x['fills']:4} raw {x['raw']:+5.1f}c shrunk {x['shrunk']:+5.1f}c ({100*x['shrunk']/80:+.1f}%)")
                    out.setdefault("shrunk", {})[k] = x
    tp = tapes(rows)
    if tp:
        found = [t for t in tp if t["found"]]
        say(f"\n-- TRADE RECORD of the 80c fills: read {len(tp)}, a print under 80c found in {len(found)}")
        if found:
            c, s_ = np.array([t["contracts_below"] for t in found]), np.array([t["sold_by_taker"] for t in found])
            say(f"   contracts printed under 80c within five minutes of the first such print: median {np.median(c):.0f}, 25th pct {np.percentile(c, 25):.0f}, 10th {np.percentile(c, 10):.0f}; "
                f"sold by the taker (what a resting bid buys): median {np.median(s_):.0f}, 25th pct {np.percentile(s_, 25):.0f}; fewer than 10 in {np.mean(s_ < 10):.0%}, fewer than 100 in {np.mean(s_ < 100):.0%}")
            for fam in POOL:
                mine = [t for t in found if t["family"] == fam]
                if mine:
                    say(f"   {fam:28} fills {len(mine):3}: taker-sold contracts under 80c median {np.median([t['sold_by_taker'] for t in mine]):.0f}, "
                        f"min {min(t['sold_by_taker'] for t in mine):.0f}; won {100*np.mean([t['payout'] for t in mine]):.0f}%")
            big = [t for t in found if t["sold_by_taker"] >= 10]
            if big:
                say(f"   fills with at least 10 taker-sold contracts under 80c: {len(big)}, won {100*np.mean([t['payout'] for t in big]):.1f}%, net {100*np.mean([t['payout'] for t in big])-80:+.1f}c")
    return out

def report():
    rows, lines = load(), []

    def say(text=""):
        print(text)
        lines.append(text)
    summary = {}
    for fam in FAMILIES:
        if sum(r["family"] == fam for r in rows) >= 10:
            summary[fam] = report_family(rows, fam, say)
    summary["pooled"] = pooled(rows, say)
    (OUT/"report.txt").write_text("\n".join(lines)+"\n")
    (OUT/"summary.json").write_text(json.dumps(summary, indent=1, default=float))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("cmd", choices=("inventory", "probe", "download", "build", "peek", "trades", "report", "cbb"))
    parser.add_argument("--minutes", type=float, default=9)
    parser.add_argument("--family", default=None)
    parser.add_argument("--cap", type=int, default=40)
    args = parser.parse_args()
    wb.pace[0] = PACE
    globals()[args.cmd](**({"minutes": args.minutes} if args.cmd in ("inventory", "trades") else
                            {"minutes": args.minutes, "family": args.family, "cap": args.cap} if args.cmd in ("probe", "download") else {}))
