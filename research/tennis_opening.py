"""The opening line and the pre-match drift in Kalshi tennis match-winner markets.

Every earlier tennis study here looked only at the six hours before settlement.
This one follows each market from the moment it is listed (`open_time`, about
12-26 hours before the match) to the start of play, and asks whether the
opening line is soft, whether the price drifts predictably before the match,
when it is best to buy a favourite, and whether quotes are stale at listing.

Commands

    download  public reads only, one request at a time, about one a second,
              cached by research.api_cost.fetch (data/raw/cost-research-v1).
              Uses the batch endpoint GET /markets/candlesticks (100 tickers and
              10,000 bar periods per request; periods are counted whether or
              not a bar exists, so chunks are sized on tickers x time span).
                hourly   hourly bars, BOTH markets of every match, open_time to
                         settlement: ATP, WTA, ATP Challenger, WTA Challenger,
                         ITF men, ITF women, settled since 2026-09-18.
                opening  one-minute bars, both markets, the first three hours
                         after open_time (the four main tours).
                minute   one-minute bars from open_time to six hours before
                         settlement for every fifth main-tour match (a check
                         on the hourly bars in the middle of a market's life).
                trades   GET /markets/trades for the first six hours of both
                         markets of a sample of matches.
              The listing itself and the one-minute bars of the six hours
              before settlement (first market of each match) are already on
              disk from research/favorite_dip.py and are read from the cache.
    build     no network. One row per match: a stitched price path of the first
              market (minute bars after listing, hourly bars in the middle,
              minute bars for the last six hours), the detected start of play,
              quotes at fixed times, volumes, round and tournament.
    report    all tables (questions 2-5 of the brief), report.txt and
              summary.json under data/processed/tennis-opening/.

    holdout-download / holdout-report
              the one result of `report` that looked new (first-round
              qualifying favourites priced 90c+ at the open rose about 1.2c by
              the close) checked on matches settled January to mid-September
              2026: one hourly read per market from /historical (no batch
              endpoint there) for the heavy favourites in
              data/processed/favourite-gap/rows.json, then the same tables.
              It did not hold (+0.1c on 333 matches).

Coverage of the run of 2026-11-18: 1,037 network requests (about 20 minutes)
for the recent period and 836 (about 16 minutes) for the holdout check:
295 hourly batches (14,695 matches, both books), 368 opening-minute batches
(4,433 main-tour matches, both books), 172 full-minute batches (every fifth
main-tour match), trades for 202 markets of 101 matches. 10,160 matches have a
detected start (Challenger 2,160, WTA 577, ATP 598, WTA Challenger 636, ITF
6,189); a third of ITF matches fail the stable-quote test and are dropped.

Definitions

* One canonical market per match (the alphabetically first ticker), as in
  every earlier study. Prices are in cents. "Favourite" always means the side
  whose midpoint is above 50c AT THE TIME NAMED (so a table about the opening
  price selects favourites on the opening price and nothing later).
* Start of play: research/favourite_gap.py's detector on the cached one-minute
  bars of the last six hours - the first 15 bars must be two-sided, at most 4c
  wide and within 3c of each other (so the window opens before play); play
  starts at the first later bar whose midpoint has moved more than 2c in ten
  minutes or whose open interest has jumped. Matches that fail are dropped and
  counted. The CLOSING pre-match quote is the quote ten minutes before that
  bar (the reference the detector compares against). A sharp pre-match move
  would be read as the start; such moves are therefore missing from "pre-match".
* Opening quote (knowable at the time): the first moment the canonical book
  has been two-sided with a spread of at most 5c for ten consecutive minutes
  (one-minute bars; bars exist only for minutes in which something changed, so
  quotes are carried forward). ITF has hourly bars only: the first hourly close
  with a spread of at most 5c.
* Quotes at fixed times: 1h, 3h, 6h after open_time and 12h, 6h, 3h, 1h, 15min
  before the detected start, each the carried-forward quote at that instant
  (hourly resolution in the middle of a market's life). A quote counts as a
  price when it is two-sided and at most 10c wide.
* Taker: pays the ask plus 0.07*p*(1-p). Maker: a resting bid is filled only
  when a LATER bar prints strictly below it (hourly low in the middle of the
  life, minute lows at both ends), before the detected start; maker fee zero
  (Challenger series) with 0.0175*p*(1-p) shown as the assumed ATP/WTA case.
* Round: "first-round qualifying" / "later qualifying" / main draw from the
  rules text, read as paper/rules.py reads it. (favourite_gap.parse_rules takes
  "... Qualification Final" for the round "Final", so its "qualifying" misses
  the last qualifying round; it is used here only for the tournament name.)
* Intervals: 95% bootstrap over matches. Halves: by settlement time, the
  median of the matches in the table.

    .venv/bin/python research/tennis_opening.py download --minutes 25
    .venv/bin/python research/tennis_opening.py build
    .venv/bin/python research/tennis_opening.py report
    .venv/bin/python research/tennis_opening.py holdout-download --minutes 17
    .venv/bin/python research/tennis_opening.py holdout-report

No orders, no authentication.
"""
import argparse
from bisect import bisect_right
from collections import Counter, defaultdict
from datetime import date
import hashlib
import json
import math
from pathlib import Path
import pickle
import re
import sys
import time

import httpx
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from research import calibration_map as cm
from research.api_cost import ROOT, midnight, save, timestamp
from research.favourite_gap import detect_start, parse_rules, side_path
from research.near_end import WINDOW, cents

OUT = ROOT / "data/processed/tennis-opening"
SINCE = midnight(date(2026, 9, 18))
TOURS = {"KXATPCHALLENGERMATCH": "Challenger", "KXWTAMATCH": "WTA", "KXATPMATCH": "ATP",
         "KXWTACHALLENGERMATCH": "WTA Challenger", "KXITFMATCH": "ITF men", "KXITFWMATCH": "ITF women"}
MAIN = ("Challenger", "WTA", "ATP", "WTA Challenger")
NO_MAKER_FEE = ("Challenger", "WTA Challenger")
OPEN_SPAN, MAX_LIFE, WARMUP = 3*3600, 96*3600, 15
RNG = np.random.default_rng(11)
NAN = float("nan")


# ---------------------------------------------------------------- listing and download

def matches():
    """Settled two-market matches since SINCE from the cached listing: list of dicts, oldest first."""
    out = []
    for series, tour in TOURS.items():
        for settle, _, event, _, ms in cm.listing(None, series, archive=False).values():
            if settle < SINCE or len(ms) != 2:
                continue
            opened = cm.ts(ms[0], "open_time")
            if opened is None or settle-opened > MAX_LIFE or settle-opened < 3600:
                continue
            out.append(dict(series=series, tour=tour, event=event, settle=settle, open=opened, markets=ms))
    return sorted(out, key=lambda m: (m["open"], m["event"]))


def sampled(event, modulo):
    return int(hashlib.sha256(event.encode()).hexdigest(), 16) % modulo == 0


def plan(kind, all_matches):
    """Deterministic batch requests for one kind of download: list of (params, {ticker: (lo, hi)})."""
    jobs = []
    for m in all_matches:
        both = [x["ticker"] for x in m["markets"]]
        if kind == "hourly":
            jobs.append((m["open"]-3600, m["settle"], both))
        elif kind == "opening" and m["tour"] in MAIN:
            jobs.append((m["open"]-300, m["open"]+OPEN_SPAN, both))
        elif kind == "minute" and m["tour"] in MAIN and sampled(m["event"], 5) and m["settle"]-WINDOW > m["open"]+OPEN_SPAN:
            jobs.append((m["open"]+OPEN_SPAN, m["settle"]-WINDOW+60, both[:1]))
    unit = 3600 if kind == "hourly" else 60
    out, cur = [], []

    def close():
        if cur:
            params = dict(market_tickers=",".join(t for _, _, ts_ in cur for t in ts_), start_ts=min(j[0] for j in cur),
                          end_ts=max(j[1] for j in cur), period_interval=unit//60)
            out.append((params, {t: (lo, hi) for lo, hi, ts_ in cur for t in ts_}))
    for job in sorted(jobs):
        trial = cur+[job]
        n = sum(len(j[2]) for j in trial)
        span = (max(j[1] for j in trial)-min(j[0] for j in trial))//unit+3
        if cur and (n > 100 or n*span > 9500):
            close()
            cur = []
        cur.append(job)
    close()
    return out


def trade_sample(all_matches):
    return [m for m in all_matches if m["tour"] in MAIN and sampled(m["event"], 40)]


def trade_params(m, ticker):
    return dict(ticker=ticker, min_ts=m["open"]-60, max_ts=m["open"]+6*3600, limit=1000)


def trades_of(client, m, ticker, deadline):
    out, cursor = [], ""
    while True:
        params = dict(trade_params(m, ticker), cursor=cursor) if cursor else trade_params(m, ticker)
        if not cm.cache_file("/markets/trades", params).exists() and (client is None or time.time() > deadline):
            return None
        data = cm.get(client, "/markets/trades", params)
        out += data.get("trades") or []
        cursor = data.get("cursor")
        if not cursor or not data.get("trades"):
            return out


def download(minutes, kinds):
    all_matches = matches()
    deadline = time.time()+60*minutes
    print(f"{len(all_matches)} matches listed: {dict(Counter(m['tour'] for m in all_matches))}", flush=True)
    with httpx.Client(timeout=60) as client:
        for kind in kinds:
            if kind == "trades":
                done = 0
                for m in trade_sample(all_matches):
                    for x in m["markets"]:
                        try:
                            done += trades_of(client, m, x["ticker"], deadline) is not None
                        except RuntimeError as exc:
                            print("  trades failed", x["ticker"], exc, flush=True)
                print(f"trades: {done} of {2*len(trade_sample(all_matches))} markets on disk; requests {dict(cm.counts)}", flush=True)
                continue
            requests = plan(kind, all_matches)
            have = 0
            for i, (params, _) in enumerate(requests):
                if cm.cache_file("/markets/candlesticks", params).exists():
                    have += 1
                    continue
                if time.time() > deadline:
                    continue
                try:
                    cm.get(client, "/markets/candlesticks", params)
                    have += 1
                except RuntimeError as exc:
                    print(f"  {kind} request {i} failed: {exc} ({len(params['market_tickers'].split(','))} tickers, "
                          f"{(params['end_ts']-params['start_ts'])//60} minutes)", flush=True)
                if i % 50 == 0:
                    print(f"  {kind} {i}/{len(requests)} {dict(cm.counts)}", flush=True)
            print(f"{kind}: {have} of {len(requests)} requests on disk; requests {dict(cm.counts)}", flush=True)


def batches(kind, all_matches):
    """{ticker: bars} from whatever batch responses are on disk, and the share of requests that were."""
    out, have, requests = {}, 0, plan(kind, all_matches)
    for params, windows in requests:
        file = cm.cache_file("/markets/candlesticks", params)
        if not file.exists():
            continue
        have += 1
        got = {x["market_ticker"]: x.get("candlesticks") or [] for x in json.loads(file.read_text()).get("markets", [])}
        for ticker, (lo, hi) in windows.items():
            out[ticker] = sorted((b for b in got.get(ticker, []) if lo <= b["end_period_ts"] <= hi), key=lambda b: b["end_period_ts"])
    return out, have, len(requests)


# ---------------------------------------------------------------- build

def element(bar, unit):
    """(end, length, bid, ask, low print, high print, volume, open interest); bid 0 / ask 100 when that side is empty."""
    bid, ask = cents(bar, "yes_bid", "close"), cents(bar, "yes_ask", "close")
    low, high = cents(bar, "price", "low"), cents(bar, "price", "high")
    return (bar["end_period_ts"], unit, 0 if bid is None else bid, 100 if ask is None else ask,
            NAN if low is None else low, NAN if high is None else high, cm.volume(bar),
            float(bar.get("open_interest_fp") or bar.get("open_interest") or 0))


def stitch(m, opening, hourly, last, span=OPEN_SPAN):
    """One path for a market: minute bars to open+3h, hourly bars to the six-hour window, minute bars after."""
    h1, w0 = m["open"]+span, m["settle"]-WINDOW
    out = [element(b, 60) for b in opening or [] if b["end_period_ts"] <= h1]
    edge = h1 if opening else 0
    if edge < w0:
        for b in hourly or []:
            if b["end_period_ts"] > edge and b["end_period_ts"]-3600 < w0:
                out.append(element(b, 3600))
    edge = max(edge, out[-1][0] if out else 0)
    out += [element(b, 60) for b in last if b["end_period_ts"] > edge]
    return out


def qualifying(text):
    """"first" for a first-round qualifying match, "later" for later qualifying rounds, else None (as paper/rules.py).
    favourite_gap.parse_rules reads "... Qualification Final" as the round "Final", so it is not used for this."""
    found = re.search(r"\bQualif\w*( Final| Round (\d+))?", text or "")
    if not found:
        return None
    return "first" if found.group(2) == "1" else "later"


def build():
    all_matches = matches()
    hourly, *cover_h = batches("hourly", all_matches)
    opening, *cover_o = batches("opening", all_matches)
    middle, *cover_m = batches("minute", all_matches)
    print(f"hourly requests on disk {cover_h}, opening {cover_o}, full-minute {cover_m}", flush=True)
    rows, why = [], Counter()
    for m in all_matches:
        a, b = (x["ticker"] for x in m["markets"])
        file = cm.cache_file(f"/series/{m['series']}/markets/{a}/candlesticks", dict(start_ts=m["settle"]-WINDOW, end_ts=m["settle"], period_interval=1))
        if not file.exists():
            why[m["tour"], "no cached last six hours"] += 1
            continue
        last = sorted(json.loads(file.read_text()).get("candlesticks") or [], key=lambda x: x["end_period_ts"])
        if a not in hourly or not hourly[a]:
            why[m["tour"], "no hourly bars"] += 1
            continue
        if m["tour"] in MAIN and a not in opening:
            why[m["tour"], "no opening minute bars"] += 1
            continue
        mids = []
        for x in last[:WARMUP]:
            q = cm.quote(x)
            if q is None or q[1]-q[0] > 4:
                break
            mids.append((q[0]+q[1])/2)
        if len(mids) < WARMUP or max(mids)-min(mids) > 3:
            why[m["tour"], "last six hours do not open on a stable quote"] += 1
            continue
        path = side_path(last, True)
        began = detect_start(path)
        if began is None:
            why[m["tour"], "no start detected"] += 1
            continue
        start = path[began][0]
        tournament, stage = parse_rules(m["markets"][0].get("rules_primary"))
        row = dict(tour=m["tour"], series=m["series"], event=m["event"], ticker=a, sibling=b, open=m["open"], settle=m["settle"],
                   created=cm.ts(m["markets"][0], "created_time"), start=start, outcome=m["markets"][0]["_outcome"],
                   tournament=tournament, round=stage, qualifying=qualifying(m["markets"][0].get("rules_primary")),
                   match_volume=sum(float(x.get("volume_fp") or 0) for x in m["markets"]),
                   path=stitch(m, opening.get(a), hourly[a], last), hourly_path=stitch(m, None, hourly[a], last),
                   warm=float(np.median(mids)),
                   sibling_path=stitch(m, opening.get(b), hourly.get(b), []),
                   minute_path=None)
        if a in middle:      # the full one-minute record of the sampled matches, for checking the hourly middle
            row["minute_path"] = stitch(m, (opening.get(a) or [])+middle[a], None, last, m["settle"]-WINDOW-m["open"])
        why[m["tour"], "kept"] += 1
        rows.append(row)
    OUT.mkdir(parents=True, exist_ok=True)
    with open(OUT/"paths.pkl", "wb") as f:
        pickle.dump(rows, f)
    cover = defaultdict(dict)
    for (tour, reason), n in sorted(why.items()):
        cover[tour][reason] = n
    save(OUT/"coverage.json", dict(requests=dict(hourly=cover_h, opening=cover_o, minute=cover_m), matches=cover))
    for tour, c in cover.items():
        print(f"  {tour:15} {c}")
    return rows


# ---------------------------------------------------------------- holdout check (settled before 2026-09-18)

HOLDOUT_Q1, HOLDOUT_MAIN = 86, 88      # pre-match (warm-up) favourite price needed for a holdout match to be downloaded


def holdout_jobs():
    """Heavy favourites of the earlier period from research/favourite_gap.py's rows: every first-round qualifying match
    with a pre-match price of 86c+, and a third of the main-draw matches at 88c+ (a control). One hourly read each."""
    rows = json.loads((ROOT/"data/processed/favourite-gap/rows.json").read_text())
    listings, jobs = {}, []
    for r in rows:
        if r["period"] != "holdout" or r["tour"] not in MAIN:
            continue
        stage = qualifying(f'{r["tournament"]} {r["round"]}')
        if not ((stage == "first" and r["start"] >= HOLDOUT_Q1) or (stage is None and r["start"] >= HOLDOUT_MAIN and sampled(r["event"], 3))):
            continue
        if r["series"] not in listings:
            listings[r["series"]] = cm.listing(None, r["series"], archive=True, live=False)
        info = listings[r["series"]].get(r["event"])
        if info is None or info[4][0]["ticker"] != r["ticker"] or info[0] != r["settle"]:
            continue
        opened = cm.ts(info[4][0], "open_time")
        if opened is None or not 3600 < r["settle"]-opened <= MAX_LIFE or r["settle"]-WINDOW <= opened:
            continue
        jobs.append(dict(series=r["series"], tour=r["tour"], event=r["event"], ticker=r["ticker"], settle=r["settle"], open=opened,
                         outcome=info[4][0]["_outcome"], qualifying=stage, tournament=r["tournament"], round=r["round"],
                         request=(f"/historical/markets/{r['ticker']}/candlesticks",
                                  dict(start_ts=opened-3600, end_ts=r["settle"]-WINDOW+3600, period_interval=60))))
    return sorted(jobs, key=lambda j: (j["settle"], j["event"]))


def holdout_download(minutes):
    jobs, deadline, have = holdout_jobs(), time.time()+60*minutes, 0
    print(f"{len(jobs)} holdout matches: {dict(Counter((j['tour'], j['qualifying']) for j in jobs))}", flush=True)
    with httpx.Client(timeout=60) as client:
        for i, j in enumerate(jobs):
            if cm.cache_file(*j["request"]).exists():
                have += 1
                continue
            if time.time() > deadline:
                continue
            try:
                cm.get(client, *j["request"])
                have += 1
            except RuntimeError as exc:
                print("  failed", j["ticker"], exc, flush=True)
            if i % 100 == 0:
                print(f"  {i}/{len(jobs)} {dict(cm.counts)}", flush=True)
    print(f"holdout: {have} of {len(jobs)} on disk; requests {dict(cm.counts)}", flush=True)


def holdout_rows():
    """Rows in the same shape as build()'s, with an hourly path up to the six-hour window and minute bars after it."""
    rows, why = [], Counter()
    for j in holdout_jobs():
        file = cm.cache_file(*j["request"])
        last_file = cm.cache_file(f"/historical/markets/{j['ticker']}/candlesticks", dict(start_ts=j["settle"]-WINDOW, end_ts=j["settle"], period_interval=1))
        if not file.exists() or not last_file.exists():
            why["not on disk"] += 1
            continue
        hourly = sorted(json.loads(file.read_text()).get("candlesticks") or [], key=lambda x: x["end_period_ts"])
        last = sorted(json.loads(last_file.read_text()).get("candlesticks") or [], key=lambda x: x["end_period_ts"])
        mids = []
        for x in last[:WARMUP]:
            q = cm.quote(x)
            if q is None or q[1]-q[0] > 4:
                break
            mids.append((q[0]+q[1])/2)
        if len(mids) < WARMUP or max(mids)-min(mids) > 3 or not hourly:
            why["no stable warm-up or no hourly bars"] += 1
            continue
        path = side_path(last, True)
        began = detect_start(path)
        if began is None:
            why["no start detected"] += 1
            continue
        why["kept"] += 1
        rows.append(dict(j, start=path[began][0], warm=float(np.median(mids)), path=stitch(j, None, hourly, last), sibling_path=[], minute_path=None,
                         match_volume=0.0))
    return rows, dict(why)


def holdout_report():
    """Does the upward drift of first-round qualifying favourites priced 90c+ at the open exist before 2026-09-18?"""
    summary = {}
    say("\n=== HOLDOUT CHECK (settled 2026-01-01 to 2026-09-17, hourly bars from listing): heavy favourites from the open to the close ===")
    say(f"    Holdout matches were downloaded only if their pre-match (six hours before settlement) favourite price was {HOLDOUT_Q1}c+ (first-round qualifying)")
    say(f"    or {HOLDOUT_MAIN}c+ (main draw, one match in three). The recent sample is shown under the SAME filter and the same hourly definition of the opening quote")
    say("    (first hourly close at most 5c wide), so the two are comparable. The filter drops favourites that fell several cents after the open, which flatters")
    say("    the drift a little in both columns and both periods alike.")
    old, why = holdout_rows()
    say(f"    holdout rows: {why}")
    with open(OUT/"paths.pkl", "rb") as f:
        recent = pickle.load(f)

    def keep(r):
        fav = max(r["warm"], 100-r["warm"])
        return (r["qualifying"] == "first" and fav >= HOLDOUT_Q1) or (r["qualifying"] is None and fav >= HOLDOUT_MAIN)
    recent = [dict(r, path=r["hourly_path"]) for r in recent if r["tour"] in MAIN and keep(r)]
    sets = {"Challenger + WTA + WTA Challenger": ("Challenger", "WTA", "WTA Challenger"), "Challenger": ("Challenger",),
            "WTA + WTA Challenger": ("WTA", "WTA Challenger"), "ATP": ("ATP",)}
    for label, rows in (("HOLDOUT", [r for r in old if keep(r)]), ("RECENT, same definition and filter", recent)):
        rows.sort(key=lambda r: r["settle"])
        table = Table(rows, [derive(r) for r in rows])
        say(f"\n----- {label}: {table.n} matches -----")
        for name, tours in sets.items():
            t = table.sub(np.isin(table.tour, tours))
            yes, ok = t.fav("open")
            _, _, mid, pay = t.side("open", yes)
            _, _, close, _ = t.side("close", yes)
            for lab, m in (("first-round qualifying, open fav 90+", ok & t.qual1 & (mid >= 90)), ("main draw, open fav 90+", ok & ~t.qual & (mid >= 90))):
                if m.sum() >= 20:
                    summary[f"{label} | {name} | {lab} | drift"] = show(f"{name}: {lab}, close mid - open mid", cell((close-mid)[m], t.settle[m]), 86)
        execution(table, summary, sets, f"execution {label}", labels=("first-round qualifying, fav 90+", "main draw, fav 90+"))
    (OUT/"holdout-report.txt").write_text("\n".join(LINES)+"\n")
    save(OUT/"holdout-summary.json", json.loads(json.dumps(summary, default=float).replace("NaN", "null")))


# ---------------------------------------------------------------- report: shared helpers

CHECKS = ("open", "o+1h", "o+3h", "o+6h", "s-12h", "s-6h", "s-3h", "s-1h", "s-15m", "close")
BANDS = ((50, 60), (60, 70), (70, 80), (80, 90), (90, 95), (95, 99.5))
WIDE = ((50, 70), (70, 90), (90, 99.5))
GROUPS = {"Main tours": MAIN, "Challenger": ("Challenger",), "WTA Challenger": ("WTA Challenger",), "WTA": ("WTA",), "ATP": ("ATP",),
          "ITF": ("ITF men", "ITF women")}
LINES = []


def say(text=""):
    print(text, flush=True)
    LINES.append(text)


def taker_fee(p):
    return 7*p/100*(1-p/100)


def maker_fee(p):
    return 1.75*p/100*(1-p/100)


def ci(values, draws=2000):
    values = np.asarray(values, float)
    values = values[~np.isnan(values)]
    if len(values) < 10:
        return [NAN, NAN]
    means = np.array([values[RNG.integers(0, len(values), len(values))].mean() for _ in range(draws)]) if len(values) > 4000 else \
        values[RNG.integers(0, len(values), (draws, len(values)))].mean(axis=1)
    return [float(v) for v in np.percentile(means, [2.5, 97.5])]


def ratio_ci(num, den, draws=2000):
    """Bootstrap interval, resampling matches, for sum(num)/sum(den) (net per fill when den flags fills)."""
    num, den = np.asarray(num, float), np.asarray(den, float)
    if den.sum() < 10:
        return [NAN, NAN]
    pick = RNG.integers(0, len(num), (draws, len(num))) if len(num) <= 4000 else None
    out = []
    for k in range(draws):
        idx = pick[k] if pick is not None else RNG.integers(0, len(num), len(num))
        d = den[idx].sum()
        if d > 0:
            out.append(num[idx].sum()/d)
    return [float(v) for v in np.percentile(out, [2.5, 97.5])]


def band(c):
    return "      n/a      " if c is None or math.isnan(c[0]) else f"[{c[0]:+6.2f},{c[1]:+6.2f}]"


def cell(values, settle=None):
    """n, mean, interval and the two chronological halves of a per-match series."""
    values = np.asarray(values, float)
    keep = ~np.isnan(values)
    v = values[keep]
    out = dict(n=int(len(v)), mean=float(v.mean()) if len(v) else NAN, ci=ci(v))
    if settle is not None and len(v) >= 20:
        s = np.asarray(settle)[keep]
        cut = np.median(s)
        out["first"], out["second"] = float(v[s < cut].mean()), float(v[s >= cut].mean())
    return out


def show(name, c, width=34):
    halves = f' halves {c["first"]:+6.2f} / {c["second"]:+6.2f}' if "first" in c else ""
    say(f'  {name:{width}} n {c["n"]:5}  {c["mean"]:+6.2f}c {band(c["ci"])}{halves}')
    return c


def derive(row):
    """Arrays of the stitched path, the opening quote and the quotes at the fixed times."""
    p = np.array(row["path"], float).reshape(-1, 8)
    T, D, B, A = p[:, 0], p[:, 1], p[:, 2], p[:, 3]
    tc = row["start"]-600
    opened = None
    run = None
    for i in range(len(T)):
        if T[i] > tc:
            break
        ok = 0 < B[i] <= A[i] < 100 and A[i]-B[i] <= 5
        if D[i] == 3600:
            if ok:
                opened = T[i]
                break
            run = None
            continue
        if not ok:
            run = None
            continue
        if run is None:
            run = T[i]
        following = T[i+1] if i+1 < len(T) else math.inf
        if following >= run+600:
            opened = run+600 if run+600 <= tc else None
            break
    times = {"open": opened, "o+1h": row["open"]+3600, "o+3h": row["open"]+3*3600, "o+6h": row["open"]+6*3600,
             "s-12h": row["start"]-12*3600, "s-6h": row["start"]-6*3600, "s-3h": row["start"]-3*3600,
             "s-1h": row["start"]-3600, "s-15m": row["start"]-900, "close": tc}
    quotes = {}
    for name, t in times.items():
        quotes[name] = (NAN, NAN, NAN)
        if opened is None or t is None or t < opened or t > tc:
            continue
        i = int(np.searchsorted(T, t, "right"))-1
        if i >= 0 and 0 < B[i] <= A[i] < 100 and A[i]-B[i] <= 10:
            quotes[name] = (t, B[i], A[i])
    return dict(T=T, S=T-D, B=B, A=A, LO=p[:, 4], HI=p[:, 5], V=p[:, 6], OI=p[:, 7], tc=tc, quotes=quotes)


def first_fill(d, yes, level, placed, until):
    """A bid on one side at `level`, resting from `placed`: the first later bar (wholly after placement, ending by
    `until`) with a print strictly below it. Returns (bar end, bar length, bar volume) or None."""
    mask = (d["S"] >= placed) & (d["T"] <= until)
    hit = (d["LO"] < level) if yes else (d["HI"] > 100-level)
    idx = np.flatnonzero(mask & hit)
    if not len(idx):
        return None
    i = idx[0]
    return d["T"][i], d["T"][i]-d["S"][i], d["V"][i]


class Table:
    """Per-match arrays for the matches of one group."""

    def __init__(self, rows, derived):
        self.rows, self.d = rows, derived
        self.n = len(rows)
        self.settle = np.array([r["settle"] for r in rows], float)
        self.outcome = np.array([r["outcome"] for r in rows], float)
        self.tour = np.array([r["tour"] for r in rows])
        self.qual = np.array([r["qualifying"] is not None for r in rows], bool)
        self.qual1 = np.array([r["qualifying"] == "first" for r in rows], bool)
        self.bid = {c: np.array([d["quotes"][c][1] for d in derived]) for c in CHECKS}
        self.ask = {c: np.array([d["quotes"][c][2] for d in derived]) for c in CHECKS}
        self.time = {c: np.array([d["quotes"][c][0] for d in derived]) for c in CHECKS}
        self.mid = {c: (self.bid[c]+self.ask[c])/2 for c in CHECKS}

    def side(self, c, yes):
        """Bid, ask, midpoint and payout of the YES side (yes true) or the NO side, per match."""
        bid = np.where(yes, self.bid[c], 100-self.ask[c])
        ask = np.where(yes, self.ask[c], 100-self.bid[c])
        return bid, ask, (bid+ask)/2, np.where(yes, self.outcome, 1-self.outcome)

    def fav(self, c):
        """Favourite at checkpoint c: (is YES, valid) - midpoint strictly off 50."""
        return self.mid[c] > 50, ~np.isnan(self.mid[c]) & (self.mid[c] != 50)

    def sub(self, mask):
        idx = np.flatnonzero(mask)
        return Table([self.rows[i] for i in idx], [self.d[i] for i in idx])


def load_tables():
    with open(OUT/"paths.pkl", "rb") as f:
        rows = pickle.load(f)
    rows.sort(key=lambda r: r["settle"])
    derived = [derive(r) for r in rows]
    full = Table(rows, derived)
    return full, {name: full.sub(np.isin(full.tour, tours)) for name, tours in GROUPS.items()}


# ---------------------------------------------------------------- report sections

def describe(groups, summary):
    say("\n=== COVERAGE AND THE SHAPE OF A MARKET'S LIFE ===")
    out = {}
    for name, t in groups.items():
        if name == "Main tours":
            continue
        life = np.array([(r["start"]-r["open"])/3600 for r in t.rows])
        lag = np.array([(t.time["open"][i]-r["open"])/60 for i, r in enumerate(t.rows)])
        say(f"\n{name}: {t.n} matches; hours from listing to detected start: median {np.median(life):.1f} "
            f"(5%-95% {np.percentile(life, 5):.1f}-{np.percentile(life, 95):.1f}); opening quote found {np.mean(~np.isnan(lag)):.0%}, "
            f"median {np.nanmedian(lag):.0f} min after listing; qualifying {t.qual.mean():.0%}")
        played = np.array([(r["settle"]-r["start"])/60 for r in t.rows])
        say(f"   minutes from detected start to settlement: median {np.median(played):.0f} (5%-95% {np.percentile(played, 5):.0f}-{np.percentile(played, 95):.0f}); "
            f"over 300 (start probably detected too early, or a suspended match): {np.mean(played > 300):.1%}")
        say("   time     quoted  median spread  mean spread  share<=2c   median contracts traded since previous row")
        info = {}
        for k, c in enumerate(CHECKS):
            spread = t.ask[c]-t.bid[c]
            vols = []
            if k and c not in ("s-12h",):
                prev = CHECKS[k-1]
                for i, d in enumerate(t.d):
                    a, b = t.time[prev][i], t.time[c][i]
                    if not (math.isnan(a) or math.isnan(b)) and b > a:
                        vols.append(d["V"][(d["T"] > a) & (d["T"] <= b)].sum())
            ok = ~np.isnan(spread)
            info[c] = dict(quoted=float(ok.mean()), median_spread=float(np.nanmedian(spread)) if ok.any() else NAN,
                           mean_spread=float(np.nanmean(spread)) if ok.any() else NAN, median_volume=float(np.median(vols)) if vols else NAN)
            say(f"   {c:7} {ok.mean():6.0%}  {info[c]['median_spread']:8.1f}c  {info[c]['mean_spread']:10.2f}c  "
                f"{np.mean(spread[ok] <= 2) if ok.any() else NAN:8.0%}   {info[c]['median_volume']:10.0f}")
        total = np.array([d["V"][d["T"] <= d["tc"]].sum() for d in t.d])
        say(f"   contracts traded on the canonical book before play: median {np.median(total):.0f}, mean {total.mean():.0f}; "
            f"whole match, both books: median {np.median([r['match_volume'] for r in t.rows]):.0f}")
        early = [[sum(x[6] for x in r[key] if x[0] <= r["settle"]-WINDOW) for key in ("path", "sibling_path")] for r in t.rows]
        own, other = np.array(early).T
        say(f"   from listing to six hours before settlement (the only stretch with both books here): canonical book median {np.median(own):.0f} "
            f"mean {own.mean():.0f} contracts, other book median {np.median(other):.0f} mean {other.mean():.0f}")
        out[name] = dict(matches=t.n, checkpoints=info, prematch_volume_median=float(np.median(total)),
                         early_volume_canonical_median=float(np.median(own)), early_volume_other_median=float(np.median(other)))
    summary["describe"] = out


def calibration(groups, summary):
    say("\n=== Q2. IS THE OPENING LINE SOFT? calibration of the opening and closing pre-match prices (same matches) ===")
    out = {}
    for name in ("Main tours", "Challenger", "WTA Challenger", "WTA", "ATP", "ITF"):
        t = groups[name]
        both = ~np.isnan(t.mid["open"]) & ~np.isnan(t.mid["close"])
        say(f"\n{name}: {int(both.sum())} matches with both quotes. Gap = 100 x settlement - midpoint of the favourite AT THAT TIME.")
        out[name] = {}
        for c in ("open", "close"):
            yes, ok = t.fav(c)
            _, _, mid, pay = t.side(c, yes)
            for lo, hi in BANDS+((50, 99.5),):
                m = both & ok & (mid >= lo) & (mid < hi)
                if m.sum() < 20:
                    continue
                g = cell(100*pay[m]-mid[m], t.settle[m])
                out[name][f"{c} {lo}-{hi}"] = dict(g, price=float(mid[m].mean()), win=float(pay[m].mean()))
                show(f"{c:5} fav {lo:>2}-{hi:<4} price {mid[m].mean():5.1f} won {pay[m].mean():6.1%}", g, 44)
        # accuracy of each checkpoint against the close, canonical orientation, paired
        say("  accuracy (canonical side; Brier x100, lower is better), each time paired with the close on the same matches:")
        for c in CHECKS[:-1]:
            m = ~np.isnan(t.mid[c]) & ~np.isnan(t.mid["close"])
            if m.sum() < 50:
                continue
            b1 = 100*(t.mid[c][m]/100-t.outcome[m])**2
            b2 = 100*(t.mid["close"][m]/100-t.outcome[m])**2
            diff = cell(b1-b2, t.settle[m])
            move = t.mid["close"][m]-t.mid[c][m]
            out[name][f"brier {c}"] = dict(n=int(m.sum()), at=float(b1.mean()), close=float(b2.mean()), diff=diff,
                                           mean_abs_move=float(np.abs(move).mean()), sd_move=float(move.std()))
            say(f"   {c:6} n {int(m.sum()):5} Brier {b1.mean():6.3f} vs close {b2.mean():6.3f}  diff {diff['mean']:+.3f} {band(diff['ci'])} "
                f"halves {diff.get('first', NAN):+.3f}/{diff.get('second', NAN):+.3f}   |move to close| mean {np.abs(move).mean():.2f}c sd {move.std():.2f}c")
        # does the move carry information?  outcome - open = b * (close - open)
        m = both
        x = t.mid["close"][m]-t.mid["open"][m]
        y1 = 100*t.outcome[m]-t.mid["open"][m]
        y2 = 100*t.outcome[m]-t.mid["close"][m]
        for label, y in (("settlement minus OPEN on the open-to-close move", y1), ("settlement minus CLOSE on the open-to-close move", y2)):
            slopes = []
            for _ in range(1000):
                idx = RNG.integers(0, len(x), len(x))
                slopes.append((x[idx]*y[idx]).sum()/(x[idx]**2).sum())
            b = (x*y).sum()/(x**2).sum()
            lo_, hi_ = np.percentile(slopes, [2.5, 97.5])
            out[name][label] = dict(slope=float(b), ci=[float(lo_), float(hi_)])
            say(f"   slope of {label}: {b:+.2f} [{lo_:+.2f},{hi_:+.2f}]  (1 = the move is all information, 0 = none)")
    summary["calibration"] = out


def predictability(groups, summary):
    say("\n=== Q2b. HOW MUCH OF THE OPEN-TO-CLOSE MOVE IS PREDICTABLE AT THE OPEN? (favourite at the open; move in its price) ===")
    out = {}
    for name in ("Main tours", "ITF"):
        t = groups[name]
        yes, ok = t.fav("open")
        bid, ask, mid, pay = t.side("open", yes)
        _, _, close, _ = t.side("close", yes)
        m = ok & ~np.isnan(close)
        sib = np.full(t.n, NAN)      # favourite's price implied by the other book at the same moment, minus its own midpoint
        for i in np.flatnonzero(m):
            sp = np.array(t.rows[i]["sibling_path"], float).reshape(-1, 8)
            j = int(np.searchsorted(sp[:, 0], t.time["open"][i], "right"))-1
            if j >= 0 and 0 < sp[j, 2] <= sp[j, 3] < 100 and sp[j, 3]-sp[j, 2] <= 10:
                implied = 100-(sp[j, 2]+sp[j, 3])/2
                sib[i] = (implied if yes[i] else 100-implied)-mid[i]
        hours = (t.time["open"]-np.array([r["open"] for r in t.rows]))/3600
        feats = {"favourite price - 75 (per 10c)": (mid-75)/10, "qualifying round": t.qual.astype(float), "opening spread (c)": ask-bid,
                 "other book's implied price minus own (c)": np.nan_to_num(sib), "hours from listing to opening quote": hours}
        for tour in GROUPS[name][1:]:
            feats[f"tour {tour}"] = (t.tour == tour).astype(float)
        move = close-mid
        X = np.column_stack([np.ones(t.n)]+list(feats.values()))[m]
        y, s = move[m], t.settle[m]
        say(f"\n{name}: {int(m.sum())} matches. Move: mean {y.mean():+.2f}c, sd {y.std():.2f}c, mean |move| {np.abs(y).mean():.2f}c; "
            f"other book quoted for {np.mean(~np.isnan(sib[m])):.0%}")
        coef = np.linalg.lstsq(X, y, rcond=None)[0]
        boots = []
        for _ in range(600):
            idx = RNG.integers(0, len(y), len(y))
            boots.append(np.linalg.lstsq(X[idx], y[idx], rcond=None)[0])
        lo_, hi_ = np.percentile(boots, [2.5, 97.5], axis=0)
        for k, label in enumerate(["intercept"]+list(feats)):
            say(f"   {label:44} {coef[k]:+.3f} [{lo_[k]:+.3f},{hi_[k]:+.3f}]")
        cut = np.median(s)
        fit = np.linalg.lstsq(X[s < cut], y[s < cut], rcond=None)[0]
        pred = X[s >= cut]@fit
        test = y[s >= cut]
        r2 = 1-((test-pred)**2).sum()/(test**2).sum()
        corr = float(np.corrcoef(pred, test)[0, 1])
        top = pred >= np.percentile(pred, 80)
        bottom = pred <= np.percentile(pred, 20)
        say(f"   fitted on the earlier half, scored on the later half: out-of-sample R2 {r2:+.4f} (against a forecast of no move), "
            f"correlation {corr:+.3f}")
        say(f"   later half, top fifth of predicted moves: predicted {pred[top].mean():+.2f}c, actual {test[top].mean():+.2f}c {band(ci(test[top]))}; "
            f"bottom fifth: predicted {pred[bottom].mean():+.2f}c, actual {test[bottom].mean():+.2f}c {band(ci(test[bottom]))}")
        out[name] = dict(n=int(m.sum()), coefficients={label: [float(coef[k]), float(lo_[k]), float(hi_[k])] for k, label in enumerate(["intercept"]+list(feats))},
                         out_of_sample_r2=float(r2), correlation=corr, top_fifth_actual=float(test[top].mean()), bottom_fifth_actual=float(test[bottom].mean()))
    summary["predictability"] = out


def drift(groups, summary):
    say("\n=== Q3a. DRIFT: the favourite AT THE OPEN, change in its midpoint from the opening quote (cents) ===")
    out = {}
    for name in ("Main tours", "Challenger", "WTA Challenger", "WTA", "ATP", "ITF"):
        t = groups[name]
        yes, ok = t.fav("open")
        _, _, mid0, pay = t.side("open", yes)
        say(f"\n{name}")
        out[name] = {}
        for c in ("o+1h", "o+3h", "s-3h", "s-1h", "close"):
            _, _, mid, _ = t.side(c, yes)
            out[name][f"to {c}"] = show(f"open -> {c}", cell((mid-mid0)[ok], t.settle[ok]))
        _, _, close, _ = t.side("close", yes)
        for lo, hi in BANDS:
            m = ok & (mid0 >= lo) & (mid0 < hi)
            if m.sum() >= 30:
                out[name][f"band {lo}-{hi}"] = show(f"open -> close, open fav {lo}-{hi}", cell((close-mid0)[m], t.settle[m]))
        for label, m in (("qualifying", ok & t.qual), ("main draw", ok & ~t.qual), ("qualifying, fav 70+", ok & t.qual & (mid0 >= 70)),
                         ("first-round qualifying, fav 70+", ok & t.qual1 & (mid0 >= 70)), ("first-round qualifying, fav 90+", ok & t.qual1 & (mid0 >= 90)),
                         ("main draw, fav 90+", ok & ~t.qual & (mid0 >= 90)), ("main draw, fav 70+", ok & ~t.qual & (mid0 >= 70))):
            if m.sum() >= 30:
                out[name][label] = show(f"open -> close, {label}", cell((close-mid0)[m], t.settle[m]))
    say("\n=== Q3b. MOMENTUM OR REVERSAL (canonical side, regression through the origin; slope > 0 momentum, < 0 reversal) ===")
    for name in ("Main tours", "ITF"):
        t = groups[name]
        say(f"\n{name}")
        for a, b, c in (("open", "o+1h", "close"), ("open", "o+3h", "close"), ("open", "s-3h", "close"), ("open", "s-1h", "close"), ("s-3h", "s-1h", "close"),
                        ("open", "o+3h", "settle"), ("open", "s-3h", "settle"), ("open", "s-1h", "settle"), ("open", "close", "settle")):
            x = t.mid[b]-t.mid[a]
            y = (100*t.outcome if c == "settle" else t.mid[c])-t.mid[b]
            m = ~np.isnan(x) & ~np.isnan(y) & (t.time[b] > t.time[a])
            if m.sum() < 100:
                continue
            x, y, s = x[m], y[m], t.settle[m]
            slope = lambda i: (x[i]*y[i]).sum()/max((x[i]**2).sum(), 1e-9)
            every = np.arange(len(x))
            boots = [slope(RNG.integers(0, len(x), len(x))) for _ in range(1000)]
            lo_, hi_ = np.percentile(boots, [2.5, 97.5])
            cut = np.median(s)
            big = np.abs(x) >= 3
            follow = np.sign(x[big])*y[big]
            out[f"{name} momentum {a}->{b} then ->{c}"] = dict(n=int(m.sum()), slope=float(slope(every)), ci=[float(lo_), float(hi_)],
                                                               first=float(slope(every[s < cut])), second=float(slope(every[s >= cut])))
            say(f"   move {a:5}->{b:5} predicting {b:5}->{c:6}: n {int(m.sum()):5} sd of move {x.std():4.2f}c slope {slope(every):+.3f} [{lo_:+.3f},{hi_:+.3f}] "
                f"halves {slope(every[s < cut]):+.3f}/{slope(every[s >= cut]):+.3f}; after moves of 3c+ (n {int(big.sum())}): "
                f"{follow.mean() if big.any() else NAN:+.2f}c in the same direction {band(ci(follow))}")
    summary["drift"] = out


def strategy(name, net, settle, extra=""):
    c = cell(net, settle)
    halves = f' halves {c["first"]:+6.2f} / {c["second"]:+6.2f}' if "first" in c else ""
    say(f'  {name:50} n {c["n"]:5}  net {c["mean"]:+6.2f}c {band(c["ci"])}{halves}{extra}')
    return c


def takers(groups, summary):
    say("\n=== Q3c / Q4. TAKER TRADES: buy at the ask at time T plus the taker fee ===")
    out = {}
    for name in ("Main tours", "Challenger", "WTA Challenger", "WTA", "ATP", "ITF"):
        t = groups[name]
        say(f"\n{name}: buy the favourite of the moment, hold to settlement (all prices, then by band)")
        out[name] = {}
        for c in CHECKS:
            yes, ok = t.fav(c)
            bid, ask, mid, pay = t.side(c, yes)
            net = 100*pay-ask-taker_fee(ask)
            if ok.sum() < 50:
                continue
            out[name][f"fav hold {c}"] = strategy(f"favourite at {c} (ask {np.nanmean(ask[ok]):.1f}, gap at mid {np.nanmean((100*pay-mid)[ok]):+.2f})", net[ok], t.settle[ok])
            if True:
                for lo, hi in WIDE:
                    m = ok & (mid >= lo) & (mid < hi)
                    if m.sum() >= 50:
                        out[name][f"fav hold {c} {lo}-{hi}"] = strategy(f"     fav {lo}-{hi} at {c} (gap at mid {np.nanmean((100*pay-mid)[m]):+.2f})", net[m], t.settle[m])
        if name in ("Main tours", "ITF"):
            say(f"{name}: buy the UNDERDOG of the moment, hold to settlement")
            for c in ("open", "o+3h", "s-3h", "close"):
                yes, ok = t.fav(c)
                bid, ask, mid, pay = t.side(c, ~yes)
                out[name][f"dog hold {c}"] = strategy(f"underdog at {c}", (100*pay-ask-taker_fee(ask))[ok], t.settle[ok])
            say(f"{name}: buy at T, sell at the bid at the closing pre-match quote (two taker fees)")
            for c in ("open", "o+1h", "o+3h", "s-12h", "s-6h", "s-3h", "s-1h"):
                for label, flip in (("favourite", False), ("underdog", True)):
                    yes, ok = t.fav(c)
                    side = yes ^ flip
                    _, ask, _, _ = t.side(c, side)
                    exit_bid, _, _, _ = t.side("close", side)
                    net = exit_bid-ask-taker_fee(ask)-taker_fee(exit_bid)
                    m = ok & ~np.isnan(net)
                    if m.sum() >= 50:
                        out[name][f"{label} round trip {c}"] = strategy(f"{label} bought at {c}, sold at close", net[m], t.settle[m])
            say(f"{name}: follow (momentum) or fade (reversal) the move since the opening quote, at the ask, hold to settlement / sell at close")
            for c in ("o+1h", "o+3h", "s-6h", "s-3h", "s-1h", "close"):
                for size in (2, 4):
                    move = t.mid[c]-t.mid["open"]
                    m = ~np.isnan(move) & (np.abs(move) >= size) & (t.time[c] > t.time["open"])
                    if m.sum() < 40:
                        continue
                    for label, sign in (("follow", 1), ("fade", -1)):
                        side = (move*sign) > 0
                        _, ask, _, pay = t.side(c, side)
                        net = 100*pay-ask-taker_fee(ask)
                        out[name][f"{label} {c} {size}c hold"] = strategy(f"{label} a {size}c+ move by {c}, hold", net[m], t.settle[m])
                        if c != "close":
                            exit_bid, _, _, _ = t.side("close", side)
                            trip = exit_bid-ask-taker_fee(ask)-taker_fee(exit_bid)
                            out[name][f"{label} {c} {size}c exit"] = strategy(f"{label} a {size}c+ move by {c}, sell at close", trip[m], t.settle[m])
    summary["takers"] = out


def when_to_buy(groups, summary):
    say("\n=== Q4. WHEN TO BUY A FAVOURITE: the same side (favourite at the OPEN) bought at each time; matches quoted at all five times ===")
    out = {}
    times = ("open", "o+3h", "s-3h", "s-1h", "close")
    for name in ("Main tours", "Challenger", "WTA Challenger", "WTA", "ATP", "ITF"):
        t = groups[name]
        yes, ok = t.fav("open")
        _, _, mid0, pay = t.side("open", yes)
        every = ok.copy()
        for c in times:
            every &= ~np.isnan(t.mid[c])
        say(f"\n{name}")
        sets = [("all favourites", every), ("fav 50-70 at open", every & (mid0 < 70)), ("fav 70-90 at open", every & (mid0 >= 70) & (mid0 < 90)),
                ("fav 90+ at open", every & (mid0 >= 90)), ("qualifying, all", every & t.qual), ("qualifying, fav 70+", every & t.qual & (mid0 >= 70)),
                ("qualifying, fav 90+", every & t.qual & (mid0 >= 90)), ("first-round qualifying, fav 70+", every & t.qual1 & (mid0 >= 70)),
                ("first-round qualifying, fav 90+", every & t.qual1 & (mid0 >= 90)), ("main draw, all", every & ~t.qual), ("main draw, fav 70+", every & ~t.qual & (mid0 >= 70)),
                ("main draw, fav 90+", every & ~t.qual & (mid0 >= 90))]
        for label, m in sets:
            if m.sum() < 30:
                continue
            say(f"  {label}: n {int(m.sum())}, won {pay[m].mean():.1%}")
            out[f"{name} | {label}"] = {}
            for c in times:
                bid, ask, mid, _ = t.side(c, yes)
                gap = cell((100*pay-mid)[m], t.settle[m])
                net = cell((100*pay-ask-taker_fee(ask))[m], t.settle[m])
                out[f"{name} | {label}"][c] = dict(mid=float(mid[m].mean()), ask=float(ask[m].mean()), gap=gap, net=net)
                say(f"     at {c:5} mid {mid[m].mean():5.2f} ask {ask[m].mean():5.2f}  gap at mid {gap['mean']:+5.2f}c {band(gap['ci'])} "
                    f"halves {gap.get('first', NAN):+5.2f}/{gap.get('second', NAN):+5.2f}   net at ask after fee {net['mean']:+5.2f}c {band(net['ci'])}")
            _, _, close, _ = t.side("close", yes)
            diff = cell((close-mid0)[m], t.settle[m])
            say(f"     closing line value of buying at the open (close mid - open mid): {diff['mean']:+.2f}c {band(diff['ci'])} "
                f"halves {diff.get('first', NAN):+.2f}/{diff.get('second', NAN):+.2f}")
            out[f"{name} | {label}"]["clv"] = diff
    say("\n  Qualifying-round effect on favourites DEFINED AT EACH TIME (gap at the midpoint), favourite 70c+ at that time:")
    for name in ("Main tours", "Challenger", "WTA"):
        t = groups[name]
        for c in times:
            yes, ok = t.fav(c)
            _, _, mid, pay = t.side(c, yes)
            for label, m in (("first-round qualifying", ok & t.qual1 & (mid >= 70)), ("later qualifying", ok & t.qual & ~t.qual1 & (mid >= 70)),
                             ("main draw", ok & ~t.qual & (mid >= 70)), ("first-round qualifying 90+", ok & t.qual1 & (mid >= 90)),
                             ("main draw 90+", ok & ~t.qual & (mid >= 90))):
                if m.sum() >= 30:
                    out[f"{name} | {label} defined at {c}"] = show(f"{name}, {label}, at {c} (price {mid[m].mean():.1f})", cell((100*pay-mid)[m], t.settle[m]), 46)
    summary["when_to_buy"] = out


def execution(full, summary, sets=None, key="execution", labels=None):
    say("\n=== Q4b. EXECUTION OF THE LIVE KIND OF TRADE: the favourite at the OPENING quote, bought early, late, or with a resting bid ===")
    say("    'saving' is paired on the same matches: cost of buying at the closing pre-match ask (plus taker fee) minus the cost of the alternative, per order.")
    say("    Bid plan: rest a bid from the opening quote; if a later pre-match print is strictly below it (before 10 min ahead of play) it fills at the bid with no fee,")
    say("    otherwise buy at the closing ask. Every order ends up filled, so the saving is the whole difference between the plans.")
    out = {}
    sets = sets or {"Challenger + WTA + WTA Challenger": ("Challenger", "WTA", "WTA Challenger"), "Challenger": ("Challenger",),
                    "WTA + WTA Challenger": ("WTA", "WTA Challenger"), "ATP": ("ATP",), "ITF": ("ITF men", "ITF women")}
    for name, tours in sets.items():
        t = full.sub(np.isin(full.tour, tours))
        yes, ok = t.fav("open")
        bid, ask, mid, pay = t.side("open", yes)
        cbid, cask, cmid, _ = t.side("close", yes)
        ok = ok & ~np.isnan(cmid)
        cost_close = cask+taker_fee(cask)
        say(f"\n{name}")
        for label, m in (("first-round qualifying, fav 90+", ok & t.qual1 & (mid >= 90)), ("first-round qualifying, fav 80-90", ok & t.qual1 & (mid >= 80) & (mid < 90)),
                         ("first-round qualifying, fav 70-80", ok & t.qual1 & (mid >= 70) & (mid < 80)), ("later qualifying, fav 90+", ok & t.qual & ~t.qual1 & (mid >= 90)),
                         ("main draw, fav 90+", ok & ~t.qual & (mid >= 90)), ("main draw, fav 80-90", ok & ~t.qual & (mid >= 80) & (mid < 90)),
                         ("all favourites 90+", ok & (mid >= 90)), ("all favourites", ok)):
            idx = np.flatnonzero(m)
            if len(idx) < 30 or (labels and label not in labels):
                continue
            s_ = t.settle[idx]
            base = cell((100*pay-cost_close)[idx], s_)
            say(f"  {label}: n {len(idx)}, open mid {mid[idx].mean():.1f} ask {ask[idx].mean():.1f}; close mid {cmid[idx].mean():.1f} ask {cask[idx].mean():.1f}; "
                f"won {pay[idx].mean():.1%}; taker at the close nets {base['mean']:+.2f}c {band(base['ci'])}")
            out[f"{name} | {label}"] = dict(n=len(idx), taker_at_close=base)
            traded = np.array([[t.d[i]["V"][(t.d[i]["T"] > lo_) & (t.d[i]["T"] <= hi_)].sum() for lo_, hi_ in
                                ((t.time["open"][i], t.time["open"][i]+3600), (t.time["open"][i], t.rows[i]["start"]-3*3600),
                                 (t.rows[i]["start"]-3*3600, t.d[i]["tc"]))] for i in idx])
            say(f"    contracts traded on this book (both sides of the trade, not depth): hour after the opening quote median {np.median(traded[:, 0]):.0f} "
                f"(mean {traded[:, 0].mean():.0f}, none in {np.mean(traded[:, 0] == 0):.0%}); opening quote to 3h before play median {np.median(traded[:, 1]):.0f}; "
                f"last 3h before play median {np.median(traded[:, 2]):.0f}; matches per month {len(idx)/((t.settle.max()-t.settle.min())/86400/30.4):.0f}")
            out[f"{name} | {label}"]["volume_medians"] = [float(np.median(traded[:, k])) for k in range(3)]
            early = cell((cost_close-ask-taker_fee(ask))[idx], s_)
            out[f"{name} | {label}"]["taker at open"] = show("    saving: buy at the opening ask instead", early, 44)
            for c in ("o+3h", "s-6h", "s-3h"):
                _, a2, _, _ = t.side(c, yes)
                out[f"{name} | {label}"][f"taker at {c}"] = show(f"    saving: buy at the ask at {c} instead", cell((cost_close-a2-taker_fee(a2))[idx], s_), 44)
            for lname, level in (("at the opening bid", bid), ("1c under the opening mid", np.floor(mid-1)), ("2c under the opening mid", np.floor(mid-2))):
                save_, filled = np.zeros(len(idx)), np.zeros(len(idx))
                for k, i in enumerate(idx):
                    hit = first_fill(t.d[i], bool(yes[i]), level[i], t.time["open"][i], t.d[i]["tc"])
                    if hit is not None:
                        filled[k], save_[k] = 1, cost_close[i]-level[i]
                c_ = cell(save_, s_)
                c_["fill_rate"] = float(filled.mean())
                c_["saving_on_fills"] = float(save_.sum()/max(filled.sum(), 1))
                c_["after_maker_fee"] = float(c_["mean"]-(filled*maker_fee(level[idx])).mean())
                out[f"{name} | {label}"][f"bid {lname}"] = show(f"    saving: bid {lname}, else take", c_, 44)
                say(f"         filled {filled.mean():.0%}; saving on filled orders {c_['saving_on_fills']:+.2f}c; per order after an assumed maker fee {c_['after_maker_fee']:+.2f}c")
    summary[key] = out


def makers(groups, full, summary):
    say("\n=== Q3d. MAKER: a resting bid placed at the opening quote (or 1h after listing), filled only by a later print strictly below it ===")
    say("    held to settlement; maker fee zero (the 'fee' column applies 0.0175*p*(1-p)); orders rest until the detected start ('sched': cancelled 10 min before)")
    out = {}
    sets = {"Challenger + WTA Challenger": ("Challenger", "WTA Challenger"), "ATP + WTA": ("ATP", "WTA"), "ITF": ("ITF men", "ITF women")}
    for name, tours in sets.items():
        t = full.sub(np.isin(full.tour, tours))
        say(f"\n{name}")
        for placed in ("open", "o+1h"):
            yes, ok = t.fav(placed)
            for label, flip in (("favourite", False), ("underdog", True)):
                side = yes ^ flip
                bid, ask, mid, pay = t.side(placed, side)
                close_bid, close_ask, close_mid, _ = t.side("close", side)
                levels = {"at bid": bid, "mid-1": np.floor(mid-1), "mid-2": np.floor(mid-2), "mid-3": np.floor(mid-3)}
                for lname, level in levels.items():
                    for subset, m in (("all", ok), ("fav 50-70", ok & (np.maximum(mid, 100-mid) < 70)), ("fav 70-90", ok & (np.maximum(mid, 100-mid) >= 70) & (np.maximum(mid, 100-mid) < 90)),
                                      ("fav 90+", ok & (np.maximum(mid, 100-mid) >= 90)), ("qualifying", ok & t.qual), ("qual R1", ok & t.qual1), ("main draw", ok & ~t.qual),
                                      ("qual R1 90+", ok & t.qual1 & (np.maximum(mid, 100-mid) >= 90)), ("main 90+", ok & ~t.qual & (np.maximum(mid, 100-mid) >= 90))):
                        if subset != "all" and (placed != "open" or lname not in ("at bid", "mid-1")):
                            continue
                        idx = np.flatnonzero(m & (level >= 1))
                        if len(idx) < 40:
                            continue
                        filled, sched, net, mark, vol, wait, later = (np.zeros(len(idx)) for _ in range(7))
                        for k, i in enumerate(idx):
                            hit = first_fill(t.d[i], bool(side[i]), level[i], t.time[placed][i], t.rows[i]["start"])
                            if hit is None:
                                continue
                            filled[k], net[k] = 1, 100*pay[i]-level[i]
                            sched[k] = hit[0] <= t.d[i]["tc"]
                            mark[k] = close_mid[i]-level[i] if not math.isnan(close_mid[i]) else 0
                            vol[k], wait[k] = hit[2], (hit[0]-t.time[placed][i])/3600
                        if filled.sum() < 10:
                            continue
                        s = t.settle[idx]
                        cut = np.median(s)
                        f = filled.astype(bool)
                        per_fill, per_order = net.sum()/filled.sum(), net.sum()/len(idx)
                        fee = np.where(f, maker_fee(level[idx]), 0).sum()/filled.sum()
                        halves = [net[(s < cut)].sum()/max(filled[s < cut].sum(), 1), net[(s >= cut)].sum()/max(filled[s >= cut].sum(), 1)]
                        key = f"{name} | {placed} | {label} | {lname} | {subset}"
                        # the unfilled orders' result had they bought at the placement ask: shows whether fills are the bad half
                        unfilled = (100*pay[idx]-ask[idx])[~f]
                        out[key] = dict(orders=len(idx), fills=int(filled.sum()), fill_rate=float(filled.mean()), per_fill=float(per_fill),
                                        per_fill_ci=ratio_ci(net, filled), per_fill_halves=[float(h) for h in halves], per_order=float(per_order),
                                        per_order_ci=ci(net) if len(idx) <= 4000 else ratio_ci(net, np.ones(len(idx))), per_fill_after_maker_fee=float(per_fill-fee),
                                        markout_to_close=float(mark.sum()/filled.sum()), sched_fill_rate=float(sched.mean()),
                                        sched_per_fill=float((net*sched).sum()/max(sched.sum(), 1)), median_bar_volume=float(np.median(vol[f])),
                                        median_hours_to_fill=float(np.median(wait[f])), mean_level=float(level[idx][f].mean()), win=float(pay[idx][f].mean()),
                                        unfilled_gap_at_ask=float(unfilled.mean()) if len(unfilled) else NAN)
                        o = out[key]
                        say(f"  {placed:5} {label:9} {lname:6} {subset:10} orders {len(idx):5} fills {int(filled.sum()):5} ({filled.mean():4.0%}) level {o['mean_level']:5.1f} "
                            f"won {o['win']:5.1%} per fill {per_fill:+6.2f}c {band(o['per_fill_ci'])} halves {halves[0]:+6.2f}/{halves[1]:+6.2f} fee {o['per_fill_after_maker_fee']:+6.2f} "
                            f"per order {per_order:+5.2f}c {band(o['per_order_ci'])} | close mid - level {o['markout_to_close']:+5.2f}c | sched {sched.mean():4.0%} {o['sched_per_fill']:+6.2f}c "
                            f"| fill-bar volume median {o['median_bar_volume']:.0f}, {o['median_hours_to_fill']:.1f}h to fill")
    summary["makers"] = out


def minute_check(full, summary):
    say("\n=== CHECK: hourly bars in the middle of the life against the full one-minute record (sampled main-tour matches) ===")
    diffs, agree, total = defaultdict(list), Counter(), Counter()
    n = 0
    for i, r in enumerate(full.rows):
        if not r["minute_path"]:
            continue
        n += 1
        fine = derive(dict(r, path=r["minute_path"]))
        for c in ("s-12h", "s-6h", "o+6h"):
            a, b = full.d[i]["quotes"][c], fine["quotes"][c]
            if not math.isnan(a[1]) and not math.isnan(b[1]):
                diffs[c].append((b[1]+b[2])/2-(a[1]+a[2])/2)
        t0 = full.d[i]["quotes"]["open"]
        if math.isnan(t0[1]):
            continue
        for lname, level in (("at bid", t0[1]), ("mid-1", math.floor((t0[1]+t0[2])/2-1))):
            a = first_fill(full.d[i], True, level, t0[0], r["start"]) is not None
            b = first_fill(fine, True, level, t0[0], r["start"]) is not None
            total[lname] += 1
            agree[lname, a, b] += 1
    say(f"  {n} matches with the full minute record")
    out = dict(matches=n)
    for c, v in diffs.items():
        v = np.array(v)
        out[c] = dict(n=len(v), mean=float(v.mean()), mean_abs=float(np.abs(v).mean()), p95_abs=float(np.percentile(np.abs(v), 95)))
        say(f"  midpoint at {c}: minute record minus hourly carry-forward, n {len(v)}: mean {v.mean():+.2f}c, mean |diff| {np.abs(v).mean():.2f}c, "
            f"95th pct {np.percentile(np.abs(v), 95):.1f}c")
    for lname in total:
        say(f"  YES bid {lname} at the opening quote: filled in both {agree[lname, True, True]}, hourly only {agree[lname, True, False]}, "
            f"minute only {agree[lname, False, True]}, neither {agree[lname, False, False]} (of {total[lname]})")
        out[f"fills {lname}"] = {f"{a}/{b}": agree[lname, a, b] for a in (True, False) for b in (True, False)}
    summary["minute_check"] = out


def grid(path, start, minutes):
    """Carried-forward (bid, ask) at each whole minute after `start` from a minute path; (0, 100) before the first bar."""
    p = np.array(path, float).reshape(-1, 8)
    p = p[p[:, 1] == 60]
    at = start+60*np.arange(1, minutes+1)
    idx = np.searchsorted(p[:, 0], at, "right")-1
    bid = np.where(idx >= 0, p[np.maximum(idx, 0), 2], 0)
    ask = np.where(idx >= 0, p[np.maximum(idx, 0), 3], 100)
    return bid, ask, p


def stale(groups, full, summary):
    say("\n=== Q5. THE FIRST THREE HOURS AFTER LISTING (main tours, one-minute bars, both books) ===")
    out = {}
    t = groups["Main tours"]
    span = OPEN_SPAN//60
    G = []
    for i, r in enumerate(t.rows):
        if r["start"]-600 < r["open"]+OPEN_SPAN+3600:      # needs three quiet hours and one more for the reference
            continue
        own_b, own_a, own = grid(r["path"], r["open"], span)
        sib_b, sib_a, sib = grid(r["sibling_path"], r["open"], span)
        ref = t.d[i]["quotes"]["o+3h"]
        G.append(dict(i=i, own_b=own_b, own_a=own_a, sib_b=sib_b, sib_a=sib_a, own=own, sib=sib, ref=(ref[1]+ref[2])/2,
                      settle=r["settle"], outcome=r["outcome"], tour=r["tour"]))
    say(f"  {len(G)} matches listed at least four hours before play")
    OB, OA, SB, SA = (np.array([g[k] for g in G]) for k in ("own_b", "own_a", "sib_b", "sib_a"))
    two = (OB > 0) & (OA < 100)
    spread = np.where(two, OA-OB, np.nan)
    mid = np.where(two, (OA+OB)/2, np.nan)
    say("  minutes after listing: share of canonical books two-sided, median / mean spread when two-sided, share <=2c, share <=5c,"
        " mean |midpoint - midpoint 60 min later|, share moving 3c+")
    rowsout = {}
    for k in (1, 2, 5, 10, 15, 30, 60, 90, 120):
        j = k-1
        later = mid[:, j+60] if j+60 < span else np.full(len(G), np.nan)
        dm = np.abs(later-mid[:, j])
        rowsout[k] = dict(two_sided=float(two[:, j].mean()), median_spread=float(np.nanmedian(spread[:, j])), mean_spread=float(np.nanmean(spread[:, j])),
                          le2=float(np.nanmean(spread[:, j] <= 2)), le5=float(np.nanmean(spread[:, j] <= 5)), abs_move_60=float(np.nanmean(dm)),
                          move3=float(np.nanmean(dm[~np.isnan(dm)] >= 3)))
        o = rowsout[k]
        say(f"   +{k:3} min  two-sided {o['two_sided']:4.0%}  spread median {o['median_spread']:5.1f}c mean {o['mean_spread']:5.1f}c  <=2c {np.mean(spread[:, j] <= 2):4.0%} "
            f"<=5c {np.mean(spread[:, j] <= 5):4.0%}  |move in next hour| {o['abs_move_60']:.2f}c  3c+ {o['move3']:4.0%}")
    out["spreads"] = rowsout
    by_tour = {}
    for tour in MAIN:
        m = np.array([g["tour"] == tour for g in G])
        if m.sum():
            by_tour[tour] = {k: [float(two[m, k-1].mean()), float(np.nanmedian(spread[m, k-1]))] for k in (5, 30, 120)}
            say(f"   {tour:15} two-sided / median spread at +5 min {by_tour[tour][5][0]:.0%} / {by_tour[tour][5][1]:.0f}c, +30 min {by_tour[tour][30][0]:.0%} / "
                f"{by_tour[tour][30][1]:.0f}c, +120 min {by_tour[tour][120][0]:.0%} / {by_tour[tour][120][1]:.0f}c")
    out["spreads_by_tour"] = by_tour

    # stale quotes against the eventual consensus (the canonical midpoint three hours after listing)
    ref = np.array([g["ref"] for g in G])
    settle = np.array([g["settle"] for g in G])
    outcome = np.array([g["outcome"] for g in G])
    say("  quotes resting far from the consensus (canonical midpoint three hours after listing), either book, either side:")
    refs = {"own": ref, "sib": 100-ref}
    for k in (2, 5, 15, 30, 60):
        j = k-1
        cheap = np.concatenate([refs["own"]-OA[:, j], refs["sib"]-SA[:, j]])        # an offer below the consensus: a taker's gain
        rich = np.concatenate([OB[:, j]-refs["own"], SB[:, j]-refs["sib"]])        # a bid above it
        has_ask = np.concatenate([OA[:, j] < 100, SA[:, j] < 100])
        has_bid = np.concatenate([OB[:, j] > 0, SB[:, j] > 0])
        ok = ~np.isnan(np.concatenate([ref, ref]))
        say(f"   +{k:3} min: offers 3c+ below consensus {np.mean((cheap >= 3) & has_ask & ok):5.1%} of books (5c+ {np.mean((cheap >= 5) & has_ask & ok):5.1%}); "
            f"bids 3c+ above {np.mean((rich >= 3) & has_bid & ok):5.1%} (5c+ {np.mean((rich >= 5) & has_bid & ok):5.1%})")
        out[f"stale +{k}"] = dict(cheap_offer_3c=float(np.mean((cheap >= 3) & has_ask & ok)), rich_bid_3c=float(np.mean((rich >= 3) & has_bid & ok)))

    # crossed books: buy both players for less than 100 (or sell both for more), at the same minute close
    both_asks = OA+SA
    both_bids = OB+SB
    fee2 = taker_fee(OA)+taker_fee(SA)
    arb = (both_asks+fee2 < 100) & (OA < 100) & (SA < 100)
    arb_gross = (both_asks < 100) & (OA < 100) & (SA < 100)
    crossed = (both_bids > 100)
    say(f"  the two books against each other (minute closes, first three hours): both asks sum below 100 in {arb_gross.any(axis=1).mean():.1%} of matches "
        f"({arb_gross.mean():.2%} of minutes); below 100 after two taker fees in {arb.any(axis=1).mean():.1%} of matches ({arb.mean():.3%} of minutes, "
        f"mean profit when present {np.mean((100-both_asks-fee2)[arb]) if arb.any() else 0:.1f}c, median run {np.median([r for row in arb for r in runs(row)]) if arb.any() else 0:.0f} min); "
        f"both bids sum above 100 in {crossed.any(axis=1).mean():.1%} of matches")
    out["crossed"] = dict(gross_matches=float(arb_gross.any(axis=1).mean()), net_matches=float(arb.any(axis=1).mean()), net_minutes=float(arb.mean()),
                          bids_crossed_matches=float(crossed.any(axis=1).mean()))

    # a taker who buys an offer that sits below the OTHER book's midpoint (knowable at the time)
    say("  TAKER on a stale offer: at each minute close, buy the offer of a book when it is X cents below the price implied by the other book's midpoint")
    say("  (other book two-sided, at most 4c wide); first signal per match; hold to settlement, and marked at the consensus three hours after listing")
    for edge in (2, 3, 5):
        nets, marks, sett, when = [], [], [], []
        for n_, g in enumerate(G):
            found = None
            for own_a, sb, sa, flip in ((OA[n_], SB[n_], SA[n_], False), (SA[n_], OB[n_], OA[n_], True)):
                ok = (sb > 0) & (sa < 100) & (sa-sb <= 4) & (own_a < 100)
                implied = 100-(sb+sa)/2
                hit = np.flatnonzero(ok & (implied-own_a >= edge))
                if len(hit) and (found is None or hit[0] < found[0]):
                    found = (hit[0], own_a[hit[0]], flip)
            if found is None:
                continue
            j, price, flip = found
            pay = 1-g["outcome"] if flip else g["outcome"]
            consensus = 100-g["ref"] if flip else g["ref"]
            nets.append(100*pay-price-taker_fee(price))
            marks.append(consensus-price-taker_fee(price))
            sett.append(g["settle"])
            when.append(j+1)
        if len(nets) >= 20:
            c = strategy(f"offer {edge}c+ below the other book (median +{np.median(when):.0f} min)", nets, np.array(sett))
            mk = cell(marks, np.array(sett))
            say(f"       marked at the three-hour consensus after the fee: {mk['mean']:+.2f}c {band(mk['ci'])} halves {mk.get('first', NAN):+.2f}/{mk.get('second', NAN):+.2f}; "
                f"signals in {len(nets)/len(G):.0%} of matches")
            out[f"stale offer taker {edge}c"] = dict(hold=c, marked=mk, share=len(nets)/len(G))

    # an early maker quoting around a model price
    say("  MAKER at listing: five minutes after listing, rest a bid on each book at (model - k), k = 1, 2, 3, until three hours after listing;")
    say("  filled by a later minute print strictly below. Model A (knowable): the other book's midpoint if two-sided and at most 6c wide at +5 min, else no order.")
    say("  Model B (NOT knowable, an upper bound): the canonical midpoint three hours after listing.")
    for model in ("A other book", "B eventual consensus"):
        for k in (1, 2, 3):
            num, den, mark, sett, orders, vol = [], [], [], [], 0, []
            for n_, g in enumerate(G):
                tot_net = tot_fill = tot_mark = n_orders = 0
                for own, ob, oa, sb, sa, flip in ((g["own"], OB[n_], OA[n_], SB[n_], SA[n_], False), (g["sib"], SB[n_], SA[n_], OB[n_], OA[n_], True)):
                    consensus = 100-g["ref"] if flip else g["ref"]
                    if model.startswith("A"):
                        if not (sb[4] > 0 and sa[4] < 100 and sa[4]-sb[4] <= 6):
                            continue
                        fair = 100-(sb[4]+sa[4])/2
                    else:
                        fair = consensus
                    if math.isnan(fair) or math.isnan(consensus):
                        continue
                    level = math.floor(fair-k)
                    if level < 1 or level >= oa[4]:      # would cross the book's own offer: that is a taker trade, not this test
                        continue
                    n_orders += 1
                    t0 = t.rows[g["i"]]["open"]+300
                    later = own[(own[:, 0]-60 >= t0) & (own[:, 0] <= t.rows[g["i"]]["open"]+OPEN_SPAN) & (own[:, 4] < level)]
                    if len(later):
                        pay = 1-g["outcome"] if flip else g["outcome"]
                        tot_fill += 1
                        tot_net += 100*pay-level
                        tot_mark += consensus-level
                        vol.append(later[0, 6])
                if n_orders:
                    orders += n_orders
                    num.append(tot_net)
                    den.append(tot_fill)
                    mark.append(tot_mark)
                    sett.append(g["settle"])
            num, den, mark, sett = (np.array(v, float) for v in (num, den, mark, sett))
            if den.sum() < 10:
                say(f"   model {model}, k={k}: {orders} orders, {int(den.sum())} fills")
                continue
            cut = np.median(sett)
            key = f"listing maker {model} k={k}"
            out[key] = dict(orders=orders, fills=int(den.sum()), fill_rate=float(den.sum()/orders), per_fill=float(num.sum()/den.sum()), per_fill_ci=ratio_ci(num, den),
                            per_fill_halves=[float(num[sett < cut].sum()/max(den[sett < cut].sum(), 1)), float(num[sett >= cut].sum()/max(den[sett >= cut].sum(), 1))],
                            markout=float(mark.sum()/den.sum()), markout_ci=ratio_ci(mark, den), per_order=float(num.sum()/orders),
                            median_fill_bar_volume=float(np.median(vol)))
            o = out[key]
            say(f"   model {model:20} k={k}: orders {orders:5} fills {o['fills']:4} ({o['fill_rate']:4.0%})  to settlement {o['per_fill']:+6.2f}c per fill {band(o['per_fill_ci'])} "
                f"halves {o['per_fill_halves'][0]:+.2f}/{o['per_fill_halves'][1]:+.2f}  per order {o['per_order']:+.2f}c | consensus - level {o['markout']:+.2f}c {band(o['markout_ci'])} "
                f"| fill-minute volume median {o['median_fill_bar_volume']:.0f}")
    summary["listing"] = out


def runs(flags):
    out, n = [], 0
    for f in flags:
        if f:
            n += 1
        elif n:
            out.append(n)
            n = 0
    return out+[n] if n else out


def first_trades(full, summary):
    say("\n=== Q5b. THE FIRST TRADES AFTER LISTING (public trade record, first six hours, sampled main-tour matches, both books) ===")
    index = {r["event"]: i for i, r in enumerate(full.rows)}
    sample = [m for m in trade_sample(matches())]
    markets = 0
    first_delay, sizes, per_hour, sides, vs_quote, marks, every = [], [], defaultdict(list), [], Counter(), [], []
    notional = []
    for m in sample:
        i = index.get(m["event"])
        for which, x in enumerate(m["markets"]):
            raw = trades_of(None, m, x["ticker"], 0)
            if raw is None:
                continue
            markets += 1
            tr = sorted(((timestamp(tr_["created_time"]), round(100*float(tr_["yes_price_dollars"])), float(tr_.get("count_fp") or tr_.get("count") or 0),
                          1 if (tr_.get("taker_side") or tr_.get("taker_outcome_side")) == "yes" else -1) for tr_ in raw))
            for h in (1, 3, 6):
                sel = [t_ for t_ in tr if t_[0] <= m["open"]+3600*h]
                per_hour[h].append((len(sel), sum(t_[2] for t_ in sel)))
            if not tr:
                continue
            first_delay.append((tr[0][0]-m["open"])/60)
            for n_, (ts_, price, count, side) in enumerate(tr[:10]):
                sizes.append(count)
                sides.append(side)
                notional.append(count*price/100)
            if i is None:
                continue
            r, d = full.rows[i], full.d[i]
            path = np.array(r["path"] if which == 0 else r["sibling_path"], float).reshape(-1, 8)
            ref = d["quotes"]["o+3h"]
            close = d["quotes"]["close"]
            for n_, (ts_, price, count, side) in enumerate(tr[:10]):
                j = int(np.searchsorted(path[:, 0], ts_-60, "right"))-1      # the last minute close before the trade's minute
                if j >= 0 and path[j, 2] > 0 and path[j, 3] < 100:
                    vs_quote["at or above the prior ask" if price >= path[j, 3] else "at or below the prior bid" if price <= path[j, 2] else "inside the prior quote"] += 1
                for label, q in (("3h", ref), ("close", close)):
                    if math.isnan(q[1]) or ts_ > q[0]:
                        continue
                    fair = (q[1]+q[2])/2 if which == 0 else 100-(q[1]+q[2])/2
                    marks.append((label, side*(fair-price), count, m["event"], n_))
            if not math.isnan(close[1]):      # every trade of the first six hours, for the size-weighted view
                for ts_, price, count, side in tr:
                    if ts_ <= close[0] and ts_ <= r["start"]-3600:
                        fair = (close[1]+close[2])/2 if which == 0 else 100-(close[1]+close[2])/2
                        every.append((side*(fair-price), count, m["event"], (ts_-m["open"])/3600, price if side > 0 else 100-price, r["tour"]))
    say(f"  {markets} markets of {len(sample)} sampled matches on disk; {len(first_delay)} had a trade in the first six hours")
    if not first_delay:
        summary["first_trades"] = dict(markets=markets)
        return
    sizes, sides = np.array(sizes), np.array(sides)
    say(f"  first trade: median {np.median(first_delay):.0f} min after listing (25%-75% {np.percentile(first_delay, 25):.0f}-{np.percentile(first_delay, 75):.0f})")
    for h in (1, 3, 6):
        n_, v = np.array(per_hour[h]).T
        say(f"  first {h}h per market: trades median {np.median(n_):.0f} mean {n_.mean():.1f}; contracts median {np.median(v):.0f} mean {v.mean():.0f}; no trade at all {np.mean(n_ == 0):.0%}")
    say(f"  size of the first ten trades: median {np.median(sizes):.0f} contracts, mean {sizes.mean():.0f}, 90th pct {np.percentile(sizes, 90):.0f}; "
        f"median dollars {np.median(notional):.0f}; taker bought YES in {np.mean(sides > 0):.0%}")
    tot = sum(vs_quote.values())
    say("  against the quote one minute earlier: "+", ".join(f"{k} {v/tot:.0%}" for k, v in vs_quote.most_common()))
    out = dict(markets=markets, first_trade_minutes=float(np.median(first_delay)), median_size=float(np.median(sizes)), taker_yes=float(np.mean(sides > 0)),
               versus_quote={k: v/tot for k, v in vs_quote.items()})
    for label in ("3h", "close"):
        sel = [x for x in marks if x[0] == label]
        if len(sel) < 20:
            continue
        by_event = defaultdict(list)
        for _, gain, count, event, _ in sel:
            by_event[event].append(gain)
        per_event = np.array([np.mean(v) for v in by_event.values()])
        out[f"taker markout {label}"] = dict(trades=len(sel), mean=float(np.mean([x[1] for x in sel])), by_match_ci=ci(per_event))
        say(f"  the taker's gain in the first ten trades, marked at the {'canonical midpoint three hours after listing' if label == '3h' else 'closing pre-match midpoint'}: "
            f"{np.mean([x[1] for x in sel]):+.2f}c per trade (n {len(sel)}; by match {per_event.mean():+.2f}c {band(ci(per_event))}); before the taker fee")
    if every:
        say("  EVERY trade of the first six hours (at least an hour before play), taker's gain marked at the closing pre-match midpoint, before the 0.07*p*(1-p) fee:")
        gain, count, event, hours, paid, tour = (np.array(v) for v in zip(*every))
        events = sorted(set(event))
        where = {e: np.flatnonzero(event == e) for e in events}

        def weighted(mask, label):
            if mask.sum() < 20:
                return
            boots = []
            for _ in range(1000):
                pick = RNG.integers(0, len(events), len(events))
                idx = np.concatenate([where[events[k]][mask[where[events[k]]]] for k in pick])
                if len(idx):
                    boots.append((gain[idx]*count[idx]).sum()/count[idx].sum())
            lo_, hi_ = np.percentile(boots, [2.5, 97.5])
            w = (gain[mask]*count[mask]).sum()/count[mask].sum()
            fee = (taker_fee(paid[mask])*count[mask]).sum()/count[mask].sum()
            out[f"all trades {label}"] = dict(trades=int(mask.sum()), contracts=float(count[mask].sum()), per_trade=float(gain[mask].mean()), per_contract=float(w),
                                             ci=[float(lo_), float(hi_)], fee_per_contract=float(fee))
            say(f"   {label:34} trades {int(mask.sum()):5} contracts {count[mask].sum():9.0f}  per trade {gain[mask].mean():+.2f}c  per contract {w:+.2f}c [{lo_:+.2f},{hi_:+.2f}] "
                f"(matches resampled); taker fee {fee:.2f}c per contract")
        weighted(np.ones(len(gain), bool), "all")
        weighted(count < 10, "under 10 contracts")
        weighted((count >= 10) & (count < 100), "10-99 contracts")
        weighted((count >= 100) & (count < 1000), "100-999 contracts")
        weighted(count >= 1000, "1,000+ contracts")
        weighted(hours <= 1, "first hour after listing")
        weighted((hours > 1) & (hours <= 3), "hours 1-3")
        weighted(hours > 3, "hours 3-6")
        for name in MAIN:
            weighted(tour == name, name)
    summary["first_trades"] = out


def report():
    full, groups = load_tables()
    summary = dict(matches=full.n, by_tour=dict(Counter(full.tour.tolist())), window=[int(full.settle.min()), int(full.settle.max())])
    say(f"{full.n} matches with a detected start: {summary['by_tour']}")
    describe(groups, summary)
    minute_check(full, summary)
    calibration(groups, summary)
    predictability(groups, summary)
    drift(groups, summary)
    takers(groups, summary)
    when_to_buy(groups, summary)
    execution(full, summary)
    makers(groups, full, summary)
    stale(groups, full, summary)
    first_trades(full, summary)
    (OUT/"report.txt").write_text("\n".join(LINES)+"\n")
    save(OUT/"summary.json", json.loads(json.dumps(summary, default=float).replace("NaN", "null")))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("commands", nargs="+", choices=("download", "build", "report", "holdout-download", "holdout-report"))
    parser.add_argument("--minutes", type=float, default=20, help="download budget")
    parser.add_argument("--kinds", nargs="+", default=["hourly", "opening", "minute", "trades"])
    args = parser.parse_args()
    for command in args.commands:
        if command == "download":
            download(args.minutes, args.kinds)
        elif command == "build":
            build()
        elif command == "holdout-download":
            holdout_download(args.minutes)
        elif command == "holdout-report":
            holdout_report()
        else:
            report()
