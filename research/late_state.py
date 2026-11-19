"""Is buying the late-game leader in basketball profitable once the true game state is known?

Kalshi's history has prices and trades but no score or clock. This script joins
a public play-by-play archive (ESPN's unauthenticated JSON: clock, period, score
and a win-probability number for every play, most plays with a wall-clock stamp)
to Kalshi one-minute quotes for the NBA (KXNBAGAME), men's college basketball
(KXNCAAMBGAME) and women's college basketball (KXNCAAWBGAME), October 2025 to
now, and tests state-conditioned rules for buying the leader late.

Public, unauthenticated reads only. Nothing is traded. data/paper is not read.

Commands
    listing   refresh the last days of Kalshi's settled listings (a few reads)
    espn      scoreboards and play-by-play for every game linked to a Kalshi event (1 read a second)
    prior     the NBA's previous season (2024-25) play-by-play, used only to fit the win-probability table
    bars      Kalshi one-minute bars around the end of games that have none on disk (2.5 s apart, capped)
    build     the state table (no network)
    trades    Kalshi trade tape for close late games (staleness, capacity), capped
    live      poll one in-progress game: is the win probability served in real time, and how late are plays stamped?
    report    tests 4a-4e, capacity and the bottom line (no network)

    .venv/bin/python research/late_state.py listing
    .venv/bin/python research/late_state.py espn
    .venv/bin/python research/late_state.py bars --cap 2600
    .venv/bin/python research/late_state.py build
    .venv/bin/python research/late_state.py trades --cap 1200
    .venv/bin/python research/late_state.py report

Definitions
* Link: an NBA event is linked by date and both team codes; a college event by
  date, tip-off time and both team names (research/college_basketball.py
  `link`). The archive's winner must agree with the settlement.
* Score: taken ONLY from scoring plays (the archive's other rows can show a
  score that has not happened yet). Clock and period: the latest play.
* Wall clock: a play's own stamp; stamps are made non-decreasing in play order;
  a play without a stamp takes the next stamped play's (it is known no sooner).
* Row: one real-time minute. The decision is made at a bar's close T using
  plays stamped at or before T-45 s (a live poll found plays are served 25-41 s
  after their stamp; 15 s and 75 s are run as sensitivity checks). The
  price is the leader's ask at T: the next bar's opening quote when that bar
  exists, else the last quote carried forward (bars exist only for minutes
  with activity). The leader on the NO side of the canonical market pays
  100 - YES bid.
* Late: the final 10 game-minutes of regulation, and overtime. "left" is the
  seconds on the clock at the latest play (overtime: left in that period).
* Taker fee 7*p*(1-p) cents. Maker fee 1.75*p*(1-p) on the NBA and men's
  college (unconfirmed), none on women's.
* Maker fill: a bid 1c under the ask rests for three minutes and fills only if
  a trade prints strictly below it (the bar's low on the leader's side).
* Halves: each league's games in settlement order, split at the median game.
  Thresholds are chosen on the first half and scored on the second.
* Intervals: 2,000 bootstrap draws of whole games.

Outputs under data/processed/late-state/. Raw reads under data/raw/late-state/.
"""
import argparse
from bisect import bisect_right
from collections import Counter, defaultdict
from datetime import date, datetime, timedelta, timezone
import hashlib
import json
import math
from pathlib import Path
import random
import sys
import time

import httpx
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import config
from research.api_cost import ROOT, timestamp
from research.calibration_map import cache_file, ts as field_ts
from research.near_end import cents

RAW = ROOT / "data/raw/late-state"
OUT = ROOT / "data/processed/late-state"
OLD_ESPN = ROOT / "data/raw/game-state"
ESPN = "https://site.api.espn.com/apis/site/v2/sports/"
HEADERS = {"User-Agent": "Mozilla/5.0 (research script)"}
UTC = timezone.utc
MONTHS = {m: i+1 for i, m in enumerate("JAN FEB MAR APR MAY JUN JUL AUG SEP OCT NOV DEC".split())}
LEAGUES = {
    "nba": dict(series="KXNBAGAME", path="basketball/nba", reg=4, plen=720, old="nba", maker=1.75, family="nba",
                alias={"GS": "GSW", "NO": "NOP", "NY": "NYK", "SA": "SAS", "UTAH": "UTA", "WSH": "WAS"}),
    "mcbb": dict(series="KXNCAAMBGAME", path="basketball/mens-college-basketball", reg=2, plen=1200, old="ncaamb",
                 maker=1.75, family="cbb-men"),
    "wcbb": dict(series="KXNCAAWBGAME", path="basketball/womens-college-basketball", reg=4, plen=600, old=None,
                 maker=0.0, family="cbb-women"),
}
FIRST_DAY = date(2025, 10, 21)             # the NBA regular season opened that day
HISTORICAL_BEFORE = datetime(2026, 9, 18, tzinfo=UTC).timestamp()
OLD_SLATE_UNTIL = date(2026, 11, 13)      # older cached scoreboards were read after the games ended
ESPN_PACE, KALSHI_PACE = 1.0, 2.5
DELAYS = (45, 15, 75)                      # seconds between a play's stamp and using it; the first is the main case
LATE = 600
REST = 180
DRAWS = 2000
RNG = np.random.default_rng(7)
last = dict(espn=0.0, kalshi=0.0)
counts = Counter()


def save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix+".tmp")
    temp.write_text(json.dumps(value))
    temp.replace(path)


def load(path):
    try:
        return json.loads(path.read_text())
    except (OSError, ValueError):
        return None


def log(*a):
    print(datetime.now().strftime("%H:%M:%S"), *a, flush=True)


# ------------------------------------------------------------------ network

def espn_get(client, path, params):
    """One paced public read of the play-by-play archive; None on failure."""
    for attempt in range(4):
        wait = last["espn"]+ESPN_PACE-time.time()
        if wait > 0:
            time.sleep(wait)
        last["espn"] = time.time()
        counts["espn"] += 1
        try:
            r = client.get(ESPN+path, params=params)
        except httpx.RequestError:
            time.sleep(3*(attempt+1))
            continue
        if r.status_code == 200:
            try:
                return r.json()
            except ValueError:
                return None
        if r.status_code in (400, 404):
            return None
        time.sleep(10*(attempt+1))
    return None


def budget_file():
    return RAW/"kalshi-requests.json"


def kalshi_used():
    return (load(budget_file()) or {}).get("requests", 0)


def kalshi_get(client, path, params, cap):
    """Old cache, then this script's cache, then one paced public read. Returns (data, source); data None when over the cap."""
    old = cache_file(path, params)
    if old.exists():
        data = load(old)
        if data is not None:
            return data, "old"
    key = hashlib.sha256(json.dumps([path, params], sort_keys=True).encode()).hexdigest()
    mine = RAW/"kalshi"/key[:2]/(key+".json")
    if mine.exists():
        return load(mine), "mine"
    if client is None:
        return None, "offline"
    for attempt in range(6):
        used = kalshi_used()
        if used >= cap:
            return None, "cap"
        wait = last["kalshi"]+KALSHI_PACE-time.time()
        if wait > 0:
            time.sleep(wait)
        last["kalshi"] = time.time()
        save(budget_file(), dict(requests=used+1))
        counts["kalshi"] += 1
        try:
            r = client.get(config.KALSHI_PROD_BASE+path, params=params)
        except httpx.RequestError:
            time.sleep(5*(attempt+1))
            continue
        if r.status_code == 429 or r.status_code >= 500:
            counts["throttled"] += 1
            time.sleep(min(20*2**attempt, 240))
            continue
        if r.status_code != 200:
            counts[f"http{r.status_code}"] += 1
            save(mine, dict(error=r.status_code))
            return dict(error=r.status_code), "net"
        data = r.json()
        save(mine, data)
        return data, "net"
    return None, "failed"


# ------------------------------------------------------------------ Kalshi events

def day_of(event):
    stamp = event.split("-")[1]
    return date(2000+int(stamp[:2]), MONTHS[stamp[2:5]], int(stamp[5:7]))


def events_of(league):
    """Settled two-market events since FIRST_DAY: dict(event, ticker, yes, names, day, settle, end, outcome, archived)."""
    from research import basketball_reversion as br
    from research import favourite_contexts as fc
    cfg = LEAGUES[league]
    events = br.events_of(cfg["family"])
    fresh = load(RAW/"listing"/f"{league}.json")
    if fresh:
        fc.parse(fresh, events, False, cfg["series"])
    out = []
    for settle, _, event, archived, ms in events.values():
        try:
            day = day_of(event)
        except (KeyError, ValueError, IndexError):
            continue
        if len(ms) != 2 or day < FIRST_DAY or not all(m.get("_outcome") in (0.0, 1.0) for m in ms):
            continue
        m = ms[0]
        close = field_ts(m, "close_time")
        out.append(dict(league=league, event=event, ticker=m["ticker"], yes=m["ticker"].rsplit("-", 1)[1],
                        names={x["ticker"].rsplit("-", 1)[1]: x.get("yes_sub_title") or "" for x in ms}, day=day,
                        settle=settle, end=min(close, settle) if close else settle, outcome=m["_outcome"],
                        archived=settle < HISTORICAL_BEFORE))
    return sorted(out, key=lambda g: (g["settle"], g["event"]))


def listing(cap):
    since = int(datetime(2026, 11, 16, tzinfo=UTC).timestamp())
    with httpx.Client(timeout=40, headers=HEADERS) as client:
        for league, cfg in LEAGUES.items():
            rows, cursor = [], ""
            while True:
                params = dict(series_ticker=cfg["series"], status="settled", limit=1000, min_close_ts=since)
                if cursor:
                    params["cursor"] = cursor
                data, _ = kalshi_get(client, "/markets", params, cap)
                if not data or "markets" not in data:
                    break
                rows += data["markets"]
                cursor = data.get("cursor")
                if not cursor:
                    break
            save(RAW/"listing"/f"{league}.json", rows)
            log(league, "fresh settled markets", len(rows))


# ------------------------------------------------------------------ play-by-play archive

def trim_slate(j, cfg):
    out = []
    for e in j.get("events", []):
        teams = {}
        for c in e["competitions"][0]["competitors"]:
            t = c["team"]
            code = t.get("abbreviation") or ""
            teams[c["homeAway"]] = dict(code=cfg.get("alias", {}).get(code, code), names=sorted(
                {t.get(k) for k in ("location", "shortDisplayName", "displayName") if t.get(k)}))
        if len(teams) == 2:
            out.append(dict(id=e["id"], status=e["status"]["type"]["name"], start=int(timestamp(e["date"])), teams=teams))
    return out


def slate(client, league, day):
    """The archive's games on a US date. Cached; an older script's cached copy is reused when it was read after the fact."""
    cfg = LEAGUES[league]
    mine = RAW/"espn"/league/f"day-{day:%Y%m%d}.json"
    if mine.exists():
        return load(mine) or []
    old = OLD_ESPN/cfg["old"]/f"day-{day:%Y%m%d}.json" if cfg["old"] else None
    if old and old.exists() and day <= OLD_SLATE_UNTIL:
        rows = load(old) or []
        for r in rows:                            # the NBA copy holds bare codes
            r["teams"] = {k: v if isinstance(v, dict) else dict(code=v, names=[]) for k, v in r["teams"].items()}
        if rows:
            return rows
    if client is None:
        return []
    params = dict(dates=f"{day:%Y%m%d}", limit=100) if league == "nba" else dict(dates=f"{day:%Y%m%d}", groups=50, limit=400)
    j = espn_get(client, cfg["path"]+"/scoreboard", params)
    if j is None:
        return []
    rows = trim_slate(j, cfg)
    if day < date.today():                         # a slate still in play is not kept
        save(mine, rows)
    return rows


def clock_seconds(text):
    parts = str(text).split(":")
    try:
        return float(parts[0])*60+float(parts[1]) if len(parts) == 2 else float(parts[0])
    except ValueError:
        return None


def trim_summary(j, cfg):
    """Late plays as rows [index, wall, period, clock seconds, home, away, scoring, value, type, side, home win probability]."""
    comp = j["header"]["competitions"][0]
    teams = {c["homeAway"]: dict(id=str(c["team"]["id"]), code=c["team"].get("abbreviation"), score=c.get("score"),
                                 winner=c.get("winner")) for c in comp["competitors"]}
    side_of = {teams[k]["id"]: k[0] for k in teams}
    wp = {w.get("playId"): w.get("homeWinPercentage") for w in j.get("winprobability") or []}
    plays, rows = j.get("plays") or [], []
    walls = [int(timestamp(p["wallclock"])) if p.get("wallclock") else None for p in plays]
    for i, p in enumerate(plays):
        try:
            period = int(p["period"]["number"])
            if period < cfg["reg"]:
                continue
            rows.append([i, walls[i], period, clock_seconds(p["clock"]["displayValue"]), int(p["homeScore"]), int(p["awayScore"]),
                         int(bool(p.get("scoringPlay"))), int(p.get("scoreValue") or 0), (p.get("type") or {}).get("text") or "",
                         side_of.get(str((p.get("team") or {}).get("id"))), wp.get(p.get("id"))])
        except (KeyError, TypeError, ValueError):
            continue
    stamped = [w for w in walls if w]
    return dict(status=comp.get("status", {}).get("type", {}).get("name"), date=comp.get("date"), teams=teams,
                n_plays=len(plays), n_wall=len(stamped), n_wp=len(wp), first_wall=stamped[0] if stamped else None,
                last_wall=stamped[-1] if stamped else None, plays=rows)


def game_file(league, game):
    return RAW/"espn"/league/f"game-{game}.json"


def summary(client, league, game):
    file = game_file(league, game)
    if file.exists():
        return load(file)
    if client is None:
        return None
    j = espn_get(client, LEAGUES[league]["path"]+"/summary", dict(event=game))
    if j is None or "header" not in j:
        return None
    try:
        data = trim_summary(j, LEAGUES[league])
    except (KeyError, IndexError, TypeError):
        return None
    if data["status"] == "STATUS_FINAL":
        save(file, data)
    return data


def link_all(client, league):
    """[(kalshi game, archive game, yes_home)] plus the reasons the rest were not linked."""
    from research import college_basketball as cb
    games, linked, why = events_of(league), [], Counter()
    days = sorted({g["day"]+timedelta(days=s) for g in games for s in (-1, 0, 1)})
    slates = {d: slate(client, league, d) for d in days if d <= date.today()}
    for g in games:
        if league == "nba":
            teams = set(g["names"])
            found = [e for s in (0, -1, 1) for e in slates.get(g["day"]+timedelta(days=s), [])
                     if {t["code"] for t in e["teams"].values()} == teams and e["status"] == "STATUS_FINAL"
                     and 3600 <= g["settle"]-e["start"] <= 8*3600]
            if len({e["id"] for e in found}) != 1:
                why["no single archive game with both teams"] += 1
                continue
            linked.append((g, found[0], found[0]["teams"]["home"]["code"] == g["yes"]))
        else:
            hit, reason = cb.link(g, slates)
            why[reason] += 1
            if hit:
                linked.append((g, hit[1], hit[2][g["yes"]]))
    return games, linked, why


def shuffled(items, key):
    return sorted(items, key=lambda x: hashlib.sha256(str(key(x)).encode()).hexdigest())


def espn(leagues, limit):
    with httpx.Client(timeout=40, headers=HEADERS) as client:
        jobs = []
        for league in leagues:
            games, linked, why = link_all(client, league)
            save(OUT/f"link-{league}.json", dict(kalshi_events=len(games), linked=len(linked), reasons=dict(why),
                                                 pairs=[[g["event"], e["id"], yes_home] for g, e, yes_home in linked]))
            log(league, "events", len(games), "linked", len(linked), dict(why), "requests", counts["espn"])
            todo = [(league, e["id"]) for _, e, _ in linked if not game_file(league, e["id"]).exists()]
            jobs.append(shuffled(todo, lambda x: x[1]))
        # The NBA first, then the two college leagues interleaved in a fixed random order.
        order = jobs[0] if leagues[0] == "nba" else []
        rest = jobs[1:] if leagues[0] == "nba" else jobs
        for group in zip(*rest) if len(rest) > 1 and min(map(len, rest)) else []:
            order += group
        done = set(order)
        order += [x for group in rest for x in group if x not in done]
        log("summaries to read", len(order))
        for n, (league, game) in enumerate(order[:limit]):
            data = summary(client, league, game)
            counts["empty" if not data or not data["plays"] else "plays"] += 1
            if data is not None and data["status"] != "STATUS_FINAL":
                counts["not final"] += 1
            if n % 200 == 0:
                log("summary", n, dict(counts))
    log("done", dict(counts))


def prior(limit):
    """The NBA's 2024-25 regular season and playoffs: every final game's late plays, for fitting only."""
    league, day, ids = "nba", date(2024, 10, 22), []
    with httpx.Client(timeout=40, headers=HEADERS) as client:
        while day <= date(2025, 6, 22):
            file = RAW/"espn"/"nba-prior"/f"day-{day:%Y%m%d}.json"
            rows = load(file)
            if rows is None:
                j = espn_get(client, LEAGUES[league]["path"]+"/scoreboard", dict(dates=f"{day:%Y%m%d}", limit=100))
                rows = trim_slate(j, LEAGUES[league]) if j else []
                save(file, rows)
            ids += [e["id"] for e in rows if e["status"] == "STATUS_FINAL"]
            day += timedelta(days=1)
        ids = sorted(set(ids))
        log("prior-season games", len(ids))
        for n, game in enumerate(ids[:limit]):
            file = RAW/"espn"/"nba-prior"/f"game-{game}.json"
            if file.exists():
                continue
            j = espn_get(client, LEAGUES[league]["path"]+"/summary", dict(event=game))
            try:
                save(file, trim_summary(j, LEAGUES[league]))
            except (KeyError, IndexError, TypeError):
                continue
            if n % 200 == 0:
                log("prior", n, counts["espn"])


# ------------------------------------------------------------------ Kalshi bars

def bars_on_disk(store, g):
    """Every cached one-minute bar of the canonical market near the end of the game, by bar end."""
    cfg = LEAGUES[g["league"]]
    e = (g["settle"], cfg["series"], g["event"], g["archived"], [g["_market"]])
    got = {b["end_period_ts"]: b for b in store.get(cfg["series"], e)}
    path, params = bar_request(g)
    data, _ = kalshi_get(None, path, params, 0)
    for b in (data or {}).get("candlesticks") or []:
        got.setdefault(b["end_period_ts"], b)
    return [got[k] for k in sorted(got)]


def bar_request(g):
    t, series = g["ticker"], LEAGUES[g["league"]]["series"]
    path = f"/historical/markets/{t}/candlesticks" if g["archived"] else f"/series/{series}/markets/{t}/candlesticks"
    return path, dict(start_ts=g["end"]-3*3600, end_ts=g["end"]+1200, period_interval=1)


def with_markets(league):
    """events_of rows with the raw market attached (the bar store wants it)."""
    from research import basketball_reversion as br
    from research import favourite_contexts as fc
    cfg = LEAGUES[league]
    raw = br.events_of(cfg["family"])
    fresh = load(RAW/"listing"/f"{league}.json")
    if fresh:
        fc.parse(fresh, raw, False, cfg["series"])
    games = events_of(league)
    for g in games:
        g["_market"] = raw[g["event"]][4][0]
    return games


def covered(bars, g):
    """True when the cached bars reach the end of the game: one of the last three quotes is decided (95c+ or 5c-).

    Archived bars stop when the book goes quiet, which can be long before the listed close, so time alone cannot tell.
    """
    for b in bars[-3:]:
        bid, ask = cents(b, "yes_bid", "close"), cents(b, "yes_ask", "close")
        if bid is not None and ask is not None and (bid >= 95 or ask <= 5):
            return True
    return False


def bars(leagues, cap, limit):
    from research import basketball_reversion as br
    store, todo = br.Bars(), []
    for league in leagues:
        games = with_markets(league)
        missing = [g for g in games if not covered(bars_on_disk(store, g), g)]
        log(league, "events", len(games), "without late bars on disk", len(missing))
        todo.append(shuffled(missing, lambda g: g["event"]))
    order = [g for group in todo for g in group[:limit]]
    with httpx.Client(timeout=40, headers=HEADERS) as client:
        for n, g in enumerate(order):
            data, source = kalshi_get(client, *bar_request(g), cap)
            if source == "cap":
                log("request cap reached at", n)
                break
            counts[source] += 1
            if n % 100 == 0:
                log("bars", n, "of", len(order), dict(counts), "used", kalshi_used())
    log("done", dict(counts), "used", kalshi_used())


# ------------------------------------------------------------------ state table

def other(side):
    return {"h": "a", "a": "h"}.get(side)


def timeline(data):
    """Late plays in order: usable wall times, the score from scoring plays only, a possession guess.

    Returns (walls, rows, share stamped); a row is (period, clock, home, away, possession side, home win probability,
    type, side, scoring, value, stamped).
    """
    plays = data.get("plays") or []
    if len(plays) < 20:
        return None
    walls = [r[1] for r in plays]
    share = sum(w is not None for w in walls)/len(walls)
    later = data.get("last_wall")
    for i in range(len(plays)-1, -1, -1):
        if walls[i] is None:
            walls[i] = later
        else:
            later = walls[i]
    if any(w is None for w in walls):
        return None
    walls = [int(w) for w in np.maximum.accumulate(walls)]
    home, away, poss, wp, rows = plays[0][4], plays[0][5], None, None, []
    for r in plays:
        _, stamp, period, clock, h, a, scoring, value, kind, side, prob = r
        low = kind.lower()
        if scoring:
            home, away = h, a
        if prob is not None:
            wp = prob
        if "turnover" in low:
            poss = other(side)
        elif "steal" in low or "rebound" in low:
            poss = side or poss
        elif scoring and "free" in low:
            parts = low.replace("free throw -", "").split(" of ")
            last_one = len(parts) == 2 and parts[0].strip()[-1:] == parts[1].strip()[:1]
            poss = other(side) if last_one else None
        elif scoring:
            poss = other(side)
        elif "shot" in low or "layup" in low or "dunk" in low or "jump" in low:
            poss = None
        if clock is None:
            continue
        rows.append((period, clock, home, away, poss, wp, kind, side, scoring, value, stamp is not None))
    walls = [w for w, r in zip(walls, plays) if r[3] is not None]
    return walls, rows, share


def leader_price(bar, leader_yes, field):
    """(bid, ask) on the leader's side from a bar's opening or closing quote."""
    bid, ask = cents(bar, "yes_bid", field), cents(bar, "yes_ask", field)
    if bid is None or ask is None:
        return None, None
    return (bid, ask) if leader_yes else (100-ask, 100-bid)


def traded(bar, leader_yes):
    """(low, high, contracts) of the bar's trades on the leader's side; (None, None, 0) when nothing traded."""
    from research.calibration_map import volume
    low, high, size = cents(bar, "price", "low"), cents(bar, "price", "high"), volume(bar)
    if low is None or high is None or size <= 0:
        return None, None, 0.0
    return (low, high, size) if leader_yes else (100-high, 100-low, size)


def decided_at(bars, yes_won):
    """End of the first bar from which the winner's bid stays at 98c or more."""
    when = None
    for b in reversed(bars):
        bid, ask = cents(b, "yes_bid", "close"), cents(b, "yes_ask", "close")
        if bid is None or ask is None or (bid == 0 and ask == 100):
            continue
        if (bid >= 98) if yes_won else (ask <= 2):
            when = b["end_period_ts"]
        else:
            break
    return when


def build(leagues):
    from research import basketball_reversion as br
    store = br.Bars()
    for league in leagues:
        cfg = LEAGUES[league]
        market = {g["event"]: g["_market"] for g in with_markets(league)}
        games, linked, why = link_all(None, league)
        cover = Counter(kalshi_events=len(games), linked=len(linked))
        rows, info = {d: [] for d in DELAYS}, []
        for g, e, yes_home in linked:
            g["_market"] = market[g["event"]]
            data = load(game_file(league, e["id"]))
            if data is None:
                cover["play-by-play not read"] += 1
                continue
            line = timeline(data)
            if line is None:
                cover["no play-by-play in the archive"] += 1
                continue
            walls, plays, share = line
            if share < .9:
                cover["under 90% of plays stamped"] += 1
                continue
            final = plays[-1]
            yes_won = g["outcome"] == 1
            if final[2] == final[3] or (final[2] > final[3]) != (yes_won == yes_home):
                cover["archive winner disagrees with settlement"] += 1
                continue
            series = bars_on_disk(store, g)
            if not covered(series, g):
                cover["no late bars"] += 1
                continue
            cover["joined"] += 1
            by_end = {b["end_period_ts"]: b for b in series}
            ends = sorted(by_end)
            decided = decided_at(series, yes_won)
            info.append(dict(league=league, event=g["event"], ticker=g["ticker"], game=e["id"], settle=g["settle"], day=str(g["day"]),
                             yes_home=yes_home, yes_won=yes_won, share_stamped=share, last_wall=walls[-1], close=g["end"],
                             decided=decided, margin=abs(final[2]-final[3]), periods=final[0], archived=g["archived"]))
            start = next((w for w, r in zip(walls, plays) if r[0] > cfg["reg"] or r[1] <= 720), None)
            if start is None:
                continue
            t = int(math.ceil(start/60)*60)
            while t <= walls[-1]+60:
                for delay in DELAYS:
                    i = bisect_right(walls, t-delay)-1
                    if i < 0 or i >= len(plays)-1 or plays[i][6] == "End Game":
                        continue
                    period, clock, home, away, poss, wp, *_ = plays[i]
                    if home == away or (period == cfg["reg"] and clock > 720):
                        continue
                    leader_home = home > away
                    leader_yes = leader_home == yes_home
                    nxt = by_end.get(t+60)
                    if nxt is not None:
                        bid, ask, age = *leader_price(nxt, leader_yes, "open"), 0
                    else:
                        j = bisect_right(ends, t)-1
                        if j < 0 or t-ends[j] > 900:
                            cover[f"minute without a quote (delay {delay})"] += 1
                            continue
                        bid, ask, age = *leader_price(by_end[ends[j]], leader_yes, "close"), t-ends[j]
                    if ask is None or not 1 <= ask <= 99:
                        cover[f"minute with no offer under 100c (delay {delay})"] += 1
                        continue
                    low3, size_next, at_or_below, straddle = None, 0.0, 0.0, 0.0
                    for k in (1, 2, 3):
                        b = by_end.get(t+60*k)
                        if b is None:
                            continue
                        lo, hi, size = traded(b, leader_yes)
                        if lo is None:
                            continue
                        low3 = lo if low3 is None else min(low3, lo)
                        if k == 1:
                            size_next = size
                            if hi <= ask:
                                at_or_below = size
                            elif lo <= ask:
                                straddle = size
                    rows[delay].append(dict(
                        league=league, event=g["event"], settle=g["settle"], t=t, period=period, ot=period > cfg["reg"],
                        left=clock, lead=abs(home-away), poss=None if poss is None else int((poss == "h") == leader_home),
                        ask=ask, bid=bid, age=age, wp=None if wp is None else 100*(wp if leader_home else 1-wp),
                        won=int(leader_yes == yes_won), low3=low3, size_next=size_next, at_or_below=at_or_below, straddle=straddle,
                        leader_yes=leader_yes))
                t += 60
        OUT.mkdir(parents=True, exist_ok=True)
        for delay in DELAYS:
            pd.DataFrame(rows[delay]).to_pickle(OUT/f"states-{league}-d{delay}.pkl")
        pd.DataFrame(info).to_pickle(OUT/f"games-{league}.pkl")
        cover["minutes"] = len(rows[DELAYS[0]])
        save(OUT/f"coverage-{league}.json", dict(cover, link_reasons=dict(why)))
        log(league, dict(cover))
        fit_states(league)


def fit_states(league):
    """Game-clock states from every archive game read (linked to Kalshi or not): lead, seconds left, did the leader win."""
    cfg, out = LEAGUES[league], []
    folders = [(RAW/"espn"/league, "current")] + ([(RAW/"espn"/"nba-prior", "prior")] if league == "nba" else [])
    for folder, label in folders:
        for file in sorted(folder.glob("game-*.json")):
            data = load(file)
            line = timeline(data) if data else None
            if line is None or data.get("status") != "STATUS_FINAL":
                continue
            _, plays, _ = line
            final = plays[-1]
            if final[2] == final[3]:
                continue
            home_won, i, when = final[2] > final[3], 0, int(timestamp(data["date"])) if data.get("date") else 0
            for period in range(cfg["reg"], final[0]+1):
                size = 600 if period == cfg["reg"] else 300
                for mark in range(size, 0, -10):
                    while i+1 < len(plays) and (plays[i+1][0] < period or (plays[i+1][0] == period and plays[i+1][1] >= mark)):
                        i += 1
                    p = plays[i]
                    if p[0] > period or p[2] == p[3]:
                        continue
                    leader_home = p[2] > p[3]
                    out.append((label, file.stem[5:], when, period > cfg["reg"], mark, abs(p[2]-p[3]), int(leader_home == home_won),
                                -1 if p[4] is None else int((p[4] == "h") == leader_home)))
    frame = pd.DataFrame(out, columns=["season", "game", "when", "ot", "left", "lead", "won", "poss"])
    frame.to_pickle(OUT/f"fit-{league}.pkl")
    log(league, "fit states", len(frame), "games", frame.game.nunique() if len(frame) else 0)


# ------------------------------------------------------------------ live check

def live(minutes):
    """Poll one in-progress game. For each new play: when it was first served, its stamp, and whether a win probability came with it."""
    with httpx.Client(timeout=30, headers=HEADERS) as client:
        pick = None
        for league, cfg in LEAGUES.items():
            params = dict(limit=100) if league == "nba" else dict(groups=50, limit=400)
            j = espn_get(client, cfg["path"]+"/scoreboard", params) or {}
            for e in j.get("events", []):
                if e["status"]["type"]["name"] == "STATUS_IN_PROGRESS" and e["status"].get("period", 0) >= 1:
                    pick = (league, e["id"], e.get("name"))
                    break
            if pick:
                break
        if not pick:
            log("no game in progress")
            return
        league, game, name = pick
        log("polling", league, game, name)
        seen, first, until = {}, True, time.time()+60*minutes
        while time.time() < until:
            j = espn_get(client, LEAGUES[league]["path"]+"/summary", dict(event=game))
            now = time.time()
            if j:
                wp = {w.get("playId"): w.get("homeWinPercentage") for w in j.get("winprobability") or []}
                for p in j.get("plays") or []:
                    if p["id"] not in seen:
                        stamp = timestamp(p["wallclock"]) if p.get("wallclock") else None
                        seen[p["id"]] = dict(seen=None if first else now, stamp=stamp, wp=wp.get(p["id"]),
                                             period=p["period"]["number"], clock=p["clock"]["displayValue"])
                first = False
            time.sleep(4)
        save(RAW/"live"/f"{league}-{game}-{int(time.time())}.json", dict(league=league, game=game, name=name, plays=seen))
        fresh = [v for v in seen.values() if v["seen"] and v["stamp"]]
        lags = sorted(v["seen"]-v["stamp"] for v in fresh)
        if lags:
            log("new plays", len(fresh), "served after their stamp by (s): median", round(lags[len(lags)//2], 1),
                "p10", round(lags[len(lags)//10], 1), "p90", round(lags[9*len(lags)//10], 1),
                "| with a win probability at first sight", sum(v["wp"] is not None for v in fresh))


def livecheck():
    """After the polled games ended: were the win probabilities served live later revised?"""
    with httpx.Client(timeout=30, headers=HEADERS) as client:
        for file in sorted((RAW/"live").glob("*.json")):
            snap = load(file)
            if "final_wp" not in snap:
                j = espn_get(client, LEAGUES[snap["league"]]["path"]+"/summary", dict(event=snap["game"]))
                if not j or j["header"]["competitions"][0]["status"]["type"]["name"] != "STATUS_FINAL":
                    log(file.name, "not final yet")
                    continue
                snap["final_wp"] = {w.get("playId"): w.get("homeWinPercentage") for w in j.get("winprobability") or []}
                save(file, snap)
            pairs = [(v["wp"], snap["final_wp"].get(k)) for k, v in snap["plays"].items() if v["wp"] is not None and v["seen"]]
            both = [(a, b) for a, b in pairs if b is not None]
            log(file.name, "plays with a live win probability", len(pairs), "still present", len(both),
                "changed by more than 0.5 points", sum(abs(a-b) > .005 for a, b in both),
                "largest change (points)", round(100*max((abs(a-b) for a, b in both), default=0), 1))


# ------------------------------------------------------------------ trade tape

def close_games(league):
    """Games with a lead of 6 or less inside the last three minutes (or in overtime): the window for the staleness test."""
    states, games = pd.read_pickle(OUT/f"states-{league}-d{DELAYS[0]}.pkl"), pd.read_pickle(OUT/f"games-{league}.pkl")
    tight = states[((states.left <= 180) | states.ot) & (states.lead <= 6)]
    first = tight.groupby("event").t.min()
    games = games[games.event.isin(first.index)].copy()
    games["start"] = games.event.map(first)-180
    games["stop"] = games.last_wall+120
    return games


def rule_games(league):
    """Games with a paper-rule minute (lead 6+, ask 97-99c): three minutes of tape from the first one, for capacity."""
    states, games = pd.read_pickle(OUT/f"states-{league}-d{DELAYS[0]}.pkl"), pd.read_pickle(OUT/f"games-{league}.pkl")
    first = states[(states.lead >= 6) & (states.ask >= 97) & (states.ask <= 99)].groupby("event").t.min()
    games = games[games.event.isin(first.index)].copy()
    games["start"] = games.event.map(first)
    games["stop"] = games.start+REST
    return games


def tape_file(ticker, kind="close"):
    return RAW/"trades"/kind/f"{ticker}.json"


def tape(client, r, cutoff, cap, pages):
    """Up to `pages` pages of the public trade record for one window; None when the request cap is reached."""
    path = "/historical/trades" if r["start"] < cutoff else "/markets/trades"
    rows, cursor, truncated = [], "", False
    for page in range(pages):
        params = dict(ticker=r["ticker"], min_ts=int(r["start"]), max_ts=int(r["stop"]), limit=1000)
        if cursor:
            params["cursor"] = cursor
        data = None
        for attempt in range(5):
            used = kalshi_used()
            if used >= cap:
                return None
            wait = last["kalshi"]+KALSHI_PACE-time.time()
            if wait > 0:
                time.sleep(wait)
            last["kalshi"] = time.time()
            save(budget_file(), dict(requests=used+1))
            counts["kalshi"] += 1
            try:
                resp = client.get(config.KALSHI_PROD_BASE+path, params=params)
            except httpx.RequestError:
                time.sleep(5*(attempt+1))
                continue
            if resp.status_code == 429 or resp.status_code >= 500:
                counts["throttled"] += 1
                time.sleep(min(20*2**attempt, 240))
                continue
            data = resp.json() if resp.status_code == 200 else {}
            break
        if not data:
            counts["failed"] += 1
            return dict(start=r["start"], stop=r["stop"], truncated=False, failed=True, rows=[])
        for t in data.get("trades", []):
            if not t.get("is_block_trade"):
                rows.append([timestamp(t["created_time"]), t.get("taker_side"),
                             round(100*float(t.get("yes_price_dollars", t.get("yes_price", 0))), 2),
                             float(t.get("count_fp", t.get("count", 0)))])
        cursor = data.get("cursor")
        if not cursor:
            break
        truncated = page == pages-1
    return dict(start=r["start"], stop=r["stop"], truncated=truncated, rows=sorted(rows))


def trades(leagues, cap, limit, rule_sample=110):
    jobs = []
    for kind, pick, pages, n in (("rule", rule_games, 2, rule_sample), ("close", close_games, 3, limit)):
        todo = []
        for league in leagues:
            rows = [dict(r, kind=kind, pages=pages) for r in pick(league).to_dict("records")]
            log(league, kind, "games", len(rows), "already read", sum(tape_file(r["ticker"], kind).exists() for r in rows))
            todo.append([r for r in shuffled(rows, lambda r: r["event"])[:n] if not tape_file(r["ticker"], kind).exists()])
        longest = max(map(len, todo), default=0)
        jobs += [group[i] for i in range(longest) for group in todo if i < len(group)]
    with httpx.Client(timeout=40, headers=HEADERS) as client:
        cut, _ = kalshi_get(client, "/historical/cutoff", {}, cap)
        cutoff = timestamp(cut["trades_created_ts"]) if cut and cut.get("trades_created_ts") else HISTORICAL_BEFORE
        log("historical trades before", datetime.fromtimestamp(cutoff, UTC), "jobs", len(jobs))
        for n, r in enumerate(jobs):
            data = tape(client, r, cutoff, cap, r["pages"])
            if data is None:
                log("request cap reached at job", n)
                break
            if not data.get("failed"):
                save(tape_file(r["ticker"], r["kind"]), data)
            if n % 50 == 0:
                log("tapes", n, "of", len(jobs), dict(counts), "used", kalshi_used())
    log("done", dict(counts), "used", kalshi_used())


# ------------------------------------------------------------------ report

BANDS = (("90-94", 90, 94), ("95-96", 95, 96), ("97-98", 97, 98), ("99", 99, 99), ("97-99", 97, 99), ("95-99", 95, 99))
LEADS = (("1-3", 1, 3), ("4-5", 4, 5), ("6-8", 6, 8), ("9-12", 9, 12), ("13+", 13, 999))
TIMES = (("OT", None, None), ("0-60s", 0, 60), ("1-3m", 60, 180), ("3-6m", 180, 360), ("6-10m", 360, 600))
NAMES = dict(nba="NBA", mcbb="Men's college", wcbb="Women's college")


def fee(p, rate=7.0):
    return rate*p/100*(1-p/100)


def states_of(league, delay=DELAYS[0]):
    df = pd.read_pickle(OUT/f"states-{league}-d{delay}.pkl")
    order = df.groupby("event").settle.first().sort_values()
    rank = pd.Series(np.arange(len(order)), index=order.index)
    df["half"] = np.where(df.event.map(rank) < len(order)/2, 1, 2)
    df["net"] = 100*df.won-df.ask-fee(df.ask)
    df["week"] = pd.to_datetime(df.settle, unit="s").dt.strftime("%G-%V")
    df["late"] = df.ot | (df.left <= LATE)
    return df, float(order.iloc[len(order)//2]) if len(order) else 0.0


def boot(df, col="net"):
    """Mean per row with a game-clustered 95% interval."""
    if not len(df):
        return None
    g = df.groupby("event")[col].agg(["sum", "count"]).to_numpy(float)
    idx = RNG.integers(0, len(g), (DRAWS, len(g)))
    draws = g[idx, 0].sum(1)/g[idx, 1].sum(1)
    return g[:, 0].sum()/g[:, 1].sum(), float(np.percentile(draws, 2.5)), float(np.percentile(draws, 97.5))


def first_minute(df):
    return df.sort_values("t").drop_duplicates("event")


def cell(df, col="net", price="ask"):
    """Summary of a set of rows: every minute, the first qualifying minute per game, and the halves of the latter."""
    if not len(df):
        return None
    one = first_minute(df)
    every, once = boot(df, col), boot(one, col)
    out = dict(minutes=len(df), games=len(one), ask=float(one[price].mean()), win=float(one.won.mean()), losses=int((1-one.won).sum()),
               net_all=every, net=once, pct=100*once[0]/float(one[price].mean()), weeks=one.week.nunique())
    for h in (1, 2):
        part = one[one.half == h]
        out[f"h{h}"] = (len(part), float(part[col].mean()) if len(part) else float("nan"))
    return out


def fmt(c):
    if c is None:
        return "no rows"
    return (f"games {c['games']:5d} ask {c['ask']:5.1f} won {100*c['win']:5.1f}% losses {c['losses']:3d} | first minute net {c['net'][0]:+5.2f}c "
            f"[{c['net'][1]:+5.2f},{c['net'][2]:+5.2f}] = {c['pct']:+5.2f}% | halves {c['h1'][1]:+5.2f}c (n={c['h1'][0]}) / {c['h2'][1]:+5.2f}c (n={c['h2'][0]})"
            f" | every minute ({c['minutes']}) {c['net_all'][0]:+5.2f}c [{c['net_all'][1]:+5.2f},{c['net_all'][2]:+5.2f}]")


def in_time(df, name, lo, hi):
    return df[df.ot] if name == "OT" else df[~df.ot & (df.left > lo) & (df.left <= hi)] if lo else df[~df.ot & (df.left <= hi)]


def calibration(league, df, say):
    late = df[df.late]
    say(f"\n4a. {NAMES[league]}: calibration of the leader's ask, final 10 game-minutes and overtime (taker, net of fee)")
    say("  By ask band:")
    out = {}
    for name, lo, hi in BANDS:
        c = cell(late[(late.ask >= lo) & (late.ask <= hi)])
        out[name] = c
        say(f"    {name:6s} {fmt(c)}")
    say("  By lead x time left (ask 90-99c; each cell: games, mean ask, won, first-minute net c [95% interval], H1/H2 net):")
    grid, flagged = {}, []
    band = late[(late.ask >= 90) & (late.ask <= 99)]
    for tname, tlo, thi in TIMES:
        for lname, llo, lhi in LEADS:
            part = in_time(band, tname, tlo, thi)
            c = cell(part[(part.lead >= llo) & (part.lead <= lhi)])
            grid[f"{tname}|{lname}"] = c
            if c is None or c["games"] < 20:
                continue
            mark = " *" if c["net"][1] > 0 else ""
            if mark:
                flagged.append(f"{tname}|{lname}")
            say(f"    {tname:6s} lead {lname:5s} games {c['games']:4d} ask {c['ask']:5.1f} won {100*c['win']:5.1f}% net {c['net'][0]:+5.2f} "
                f"[{c['net'][1]:+5.2f},{c['net'][2]:+5.2f}] H1 {c['h1'][1]:+5.2f} (n={c['h1'][0]}) H2 {c['h2'][1]:+5.2f} (n={c['h2'][0]}){mark}")
    say(f"    cells whose interval excludes zero (*): {flagged or 'none'}; {sum(1 for c in grid.values() if c and c['games'] >= 20)} cells shown, "
        f"so about {0.025*sum(1 for c in grid.values() if c and c['games'] >= 20):.1f} would be starred by chance")
    say("  All asks, lead x time (is the leader mispriced at any price?):")
    for tname, tlo, thi in TIMES:
        line = []
        for lname, llo, lhi in LEADS:
            part = in_time(late, tname, tlo, thi)
            part = first_minute(part[(part.lead >= llo) & (part.lead <= lhi)])
            b = boot(part)
            line.append(f"{lname}: n={len(part)} ask {part.ask.mean():.1f} won {100*part.won.mean():.1f}% net {b[0]:+.2f}{'*' if b[1] > 0 or b[2] < 0 else ''}" if len(part) >= 20 else f"{lname}: n={len(part)}")
        say(f"    {tname:6s} " + " | ".join(line))
    rule = df[(df.lead >= 6) & (df.ask >= 97) & (df.ask <= 99)]
    c = cell(rule)
    out["paper rule"] = c
    say(f"  The paper rule (last period or overtime, lead 6+, ask 97-99c): {fmt(c)}")
    band = late[(late.ask >= 95) & (late.ask <= 99)]
    for label, part in (("leader has the ball", band[band.poss == 1]), ("trailer has the ball", band[band.poss == 0]), ("possession unknown", band[band.poss.isna()])):
        say(f"    ask 95-99c, {label} (a guess from the last play): {fmt(cell(part))}")
    eras = (("to 31 Dec 2025 (never examined before)", 0, datetime(2026, 1, 1, tzinfo=UTC).timestamp()),
            ("1 Jan - 17 Sep 2026", datetime(2026, 1, 1, tzinfo=UTC).timestamp(), HISTORICAL_BEFORE),
            ("18 Sep 2026 on (where the price rule was first seen)", HISTORICAL_BEFORE, 9e12))
    for label, lo, hi in eras:
        part = rule[(rule.settle >= lo) & (rule.settle < hi)]
        if len(part):
            say(f"    {label}: {fmt(cell(part))}")
            out[f"paper rule, {label}"] = cell(part)
    return dict(bands=out, grid=grid, flagged=flagged)


def search(league, df, say):
    """Pick the best of a fixed grid of rules on the first half; score it once on the second."""
    late, rules = df[df.late], []
    for lead in (1, 4, 6, 8, 10, 13):
        for left in (600, 360, 180, 60):
            for name, lo, hi in BANDS:
                part = late[(late.lead >= lead) & (late.ot | (late.left <= left)) & (late.ask >= lo) & (late.ask <= hi)]
                one = first_minute(part) if len(part) else part
                h1, h2 = (one[one.half == h] for h in (1, 2)) if len(one) else (one, one)
                if len(h1) < 60:
                    continue
                t = h1.net.mean()/(h1.net.std(ddof=1)/math.sqrt(len(h1))) if h1.net.std(ddof=1) > 0 else 0.0
                rules.append(dict(rule=f"lead>={lead}, left<={left}s or OT, ask {name}", n1=len(h1), net1=float(h1.net.mean()), t1=float(t),
                                  n2=len(h2), net2=float(h2.net.mean()) if len(h2) else float("nan"),
                                  ci2=boot(h2)[1:] if len(h2) else None, ask2=float(h2.ask.mean()) if len(h2) else float("nan"),
                                  losses2=int((1-h2.won).sum()) if len(h2) else 0, weeks2=h2.week.nunique() if len(h2) else 0))
    rules.sort(key=lambda r: -r["net1"])
    say(f"\n6a. {NAMES[league]}: rule search. {len(rules)} rules (lead floor x time ceiling x ask band) with 60+ first-half games, ranked on the first half by net cents, scored on the second (first minute per game)")
    for r in rules[:5]:
        ci = f"[{r['ci2'][0]:+.2f},{r['ci2'][1]:+.2f}]" if r["ci2"] else ""
        say(f"    {r['rule']:42s} H1 n={r['n1']:4d} {r['net1']:+5.2f}c (t={r['t1']:+.1f}) -> H2 n={r['n2']:4d} {r['net2']:+5.2f}c {ci} losses {r['losses2']}")
    if rules:
        top = rules[:10]
        say(f"    mean of the ten best: H1 {np.mean([r['net1'] for r in top]):+.2f}c -> H2 {np.nanmean([r['net2'] for r in top]):+.2f}c; "
            f"rules positive in H1: {sum(r['net1'] > 0 for r in rules)} of {len(rules)}, of those positive in H2: {sum(r['net1'] > 0 and r['net2'] > 0 for r in rules)}")
    return rules[:10]


def design(lead, left, ot):
    lead, left = np.asarray(lead, float), np.asarray(left, float)
    z = lead/np.sqrt(left+3)
    return np.column_stack([np.ones(len(lead)), z, lead, np.log(left+3), z*np.log(left+3), np.asarray(ot, float)])


def fit_logit(x, y):
    from scipy.optimize import minimize

    def loss(b):
        u = x@b
        return np.mean(np.logaddexp(0, u)-y*u)+1e-6*b@b

    def grad(b):
        u = x@b
        return x.T@(1/(1+np.exp(-u))-y)/len(y)+2e-6*b
    return minimize(loss, np.zeros(x.shape[1]), jac=grad, method="BFGS", options=dict(maxiter=2000)).x


def recalibrate(p, y, bins=60):
    """A monotone map from the model's probability to the fitted games' own win rate (equal-count bins, then interpolation)."""
    order = np.argsort(p)
    xs = np.array([p[c].mean() for c in np.array_split(order, bins)])
    ys = np.maximum.accumulate(np.array([y[c].mean() for c in np.array_split(order, bins)]))
    return lambda q: np.interp(q, xs, ys)


def table_cell(lead, left, ot):
    edges = np.array([30, 60, 120, 180, 300, 450, 600])
    return np.minimum(np.asarray(lead), 20)*100+np.searchsorted(edges, np.asarray(left), side="left")+50*np.asarray(ot, int)


def fair_values(league, df, split, say):
    """Two fair values from play-by-play alone, fitted on games before the test games: a smooth model and a plain table."""
    fit = pd.read_pickle(OUT/f"fit-{league}.pkl")
    prior = fit[fit.season == "prior"]
    if league == "nba" and prior.game.nunique() >= 800:
        train, label, test = prior, f"the 2024-25 season ({prior.game.nunique()} games)", fit[fit.season == "current"]
        oos = df.half > 0
    else:
        cur = fit[fit.season == "current"]
        train, test = cur[cur.when < split-4*3600], cur[cur.when >= split-4*3600]
        label, oos = f"games before the split ({train.game.nunique()} games)", df.half == 2
    beta = fit_logit(design(train.lead, train.left, train.ot), train.won.to_numpy(float))
    cells = train.assign(cell=table_cell(train.lead, train.left, train.ot)).groupby("cell").won.agg(["mean", "count"])
    raw = lambda d: 1/(1+np.exp(-design(d.lead, d.left, d.ot)@beta))
    fix = recalibrate(raw(train), train.won.to_numpy(float))
    df["fair_model"] = 100*fix(raw(df))
    key = pd.Series(table_cell(df.lead, df.left, df.ot), index=df.index)
    df["fair_table"] = np.where(key.map(cells["count"]).fillna(0) >= 300, 100*key.map(cells["mean"]), np.nan)
    say(f"\n4b. {NAMES[league]}: win probability from play-by-play alone (a logistic curve in lead, seconds left and overtime, then a monotone recalibration), fitted on {label}")
    if len(test):
        p = 100*fix(raw(test))
        say("  Check on later game-clock states (not Kalshi rows): model bucket -> states, games, model %, actual %")
        for lo, hi in ((50, 80), (80, 90), (90, 95), (95, 97), (97, 98), (98, 99), (99, 99.5), (99.5, 100.01)):
            m = (p >= lo) & (p < hi)
            if m.sum():
                say(f"    {lo:>5}-{hi:<6} states {int(m.sum()):7d} games {test[m].game.nunique():5d} model {p[m].mean():6.2f} actual {100*test[m].won.mean():6.2f}")
    return oos, label


def cheap(league, df, oos, col, title, say):
    """Buy the leader when the fair value exceeds ask + fee by x cents; out-of-sample rows only."""
    out = {}
    say(f"  {title}: buy when fair - ask - fee >= x (ask 80-99c, final 10 minutes or overtime, out-of-sample games only)")
    base = df[oos & df.late & (df.ask >= 80) & (df.ask <= 99) & df[col].notna()]
    weeks = max(base.week.nunique(), 1)
    for x in (0, 1, 2, 3, 5):
        part = base[base[col]-base.ask-fee(base.ask) >= x]
        c = cell(part)
        out[x] = c
        if c is None:
            say(f"    x>={x}: no rows")
            continue
        say(f"    x>={x}: {fmt(c)} | {c['games']/weeks:.1f} bets a week over {weeks} weeks | mean fair {first_minute(part)[col].mean():.1f}")
    hi = base[(base.ask >= 95)]
    for x in (0, 1):
        part = hi[hi[col]-hi.ask-fee(hi.ask) >= x]
        c = cell(part)
        out[f"95+,{x}"] = c
        say(f"    ask 95-99c only, x>={x}: {fmt(c)}" + (f" | {c['games']/weeks:.1f} bets a week" if c else ""))
    return out


def espn_wp(league, df, say):
    have = df[df.wp.notna() & df.late]
    say(f"\n4c. {NAMES[league]}: the archive's own win probability. Present on {len(have)} of {int(df.late.sum())} late minutes "
        f"({have.event.nunique()} of {df[df.late].event.nunique()} games)")
    if not len(have):
        return None, None
    say("  Its calibration on these minutes (bucket -> minutes, games, mean %, leader won %, mean Kalshi ask):")
    for lo, hi in ((50, 80), (80, 90), (90, 95), (95, 97), (97, 98), (98, 99), (99, 99.5), (99.5, 100.01)):
        m = have[(have.wp >= lo) & (have.wp < hi)]
        if len(m):
            b = boot(m, "won")
            say(f"    {lo:>5}-{hi:<6} minutes {len(m):6d} games {m.event.nunique():5d} said {m.wp.mean():6.2f} won {100*b[0]:6.2f} [{100*b[1]:.2f},{100*b[2]:.2f}] ask {m.ask.mean():5.1f}")
    return cheap(league, df, df.half > 0, "wp", "As fair value, whole sample (nothing is fitted)", say), cheap(league, df, df.half == 2, "wp", "As fair value, second half", say)


def maker(league, df, say):
    rate = LEAGUES[league]["maker"]
    say(f"\n4d. {NAMES[league]}: maker version. First qualifying minute per game; a bid 1c under the ask rests {REST//60} minutes; filled only by a print strictly below it. Maker fee rate {rate}")
    out = {}
    sets = [("paper rule (lead 6+, ask 97-99, last period)", df[(df.lead >= 6) & (df.ask >= 97) & (df.ask <= 99)])]
    late = df[df.late]
    sets += [(f"ask {name}, final 10 minutes", late[(late.ask >= lo) & (late.ask <= hi)]) for name, lo, hi in BANDS[:4]]
    for name, part in sets:
        if not len(part):
            continue
        one = first_minute(part).copy()
        one["limit"] = one.ask-1
        one["filled"] = one.low3.notna() & (one.low3 < one.limit)
        one["mnet"] = 100*one.won-one.limit-fee(one.limit, rate)
        f, u = one[one.filled], one[~one.filled]
        if not len(f):
            say(f"  {name}: signals {len(one)}, none filled")
            continue
        b, taker = boot(f, "mnet"), boot(one)
        per_signal = one.filled*one.mnet
        ps = boot(one.assign(ps=per_signal), "ps")
        halves = " / ".join(f"{f[f.half == h].mnet.mean():+.2f}c (n={int((f.half == h).sum())})" for h in (1, 2))
        say(f"  {name}: signals {len(one)}, filled {len(f)} ({100*len(f)/len(one):.0f}%), filled won {100*f.won.mean():.1f}% (unfilled {100*u.won.mean() if len(u) else float('nan'):.1f}%), "
            f"bid {f.limit.mean():.1f}, net per fill {b[0]:+.2f}c [{b[1]:+.2f},{b[2]:+.2f}] = {100*b[0]/f.limit.mean():+.2f}%, halves {halves}; "
            f"per signal {ps[0]:+.2f}c [{ps[1]:+.2f},{ps[2]:+.2f}] against taker {taker[0]:+.2f}c [{taker[1]:+.2f},{taker[2]:+.2f}]; fill losses {int((1-f.won).sum())}")
        out[name] = dict(signals=len(one), filled=len(f), win_filled=float(f.won.mean()), win_unfilled=float(u.won.mean()) if len(u) else None,
                         net_fill=b, per_signal=ps, taker=taker)
    return out


def capacity(league, df, say):
    say(f"\n5. {NAMES[league]}: capacity in the minute after the first qualifying minute. From bars: contracts in a bar whose high is at or below the ask (exact); "
        "a bar that straddles the ask is counted as zero in 'low' and in full in 'high'")
    out = {}
    sets = [("paper rule", df[(df.lead >= 6) & (df.ask >= 97) & (df.ask <= 99)])]
    late = df[df.late]
    sets += [(f"ask {name}", late[(late.ask >= lo) & (late.ask <= hi)]) for name, lo, hi in BANDS[:4]]
    for name, part in sets:
        if not len(part):
            continue
        one = first_minute(part)
        high = one.at_or_below+one.straddle
        weeks = max(one.week.nunique(), 1)
        say(f"  {name:11s} games {len(one):5d} ({len(one)/weeks:.1f} a week over {weeks} weeks): contracts low median {one.at_or_below.median():8.0f} mean {one.at_or_below.mean():8.0f}; "
            f"high median {high.median():8.0f} mean {high.mean():8.0f}; nothing traded {100*(one.size_next == 0).mean():3.0f}%; "
            f"dollars a week low {(one.at_or_below*one.ask/100).sum()/weeks:10.0f} high {(high*one.ask/100).sum()/weeks:10.0f}")
        out[name] = dict(games=len(one), games_week=len(one)/weeks, low_median=float(one.at_or_below.median()), high_median=float(high.median()),
                         low_week=float((one.at_or_below*one.ask/100).sum()/weeks), high_week=float((high*one.ask/100).sum()/weeks))
    # the trade record for a random sample of paper-rule games
    rule = first_minute(df[(df.lead >= 6) & (df.ask >= 97) & (df.ask <= 99)]) if len(df) else df
    games = pd.read_pickle(OUT/f"games-{league}.pkl").set_index("event")
    rows = []
    for r in rule.to_dict("records"):
        data = load(tape_file(games.ticker[r["event"]], "rule"))
        if not data or data.get("start") != r["t"]:
            continue
        mine = [(w, taker == ("yes" if r["leader_yes"] else "no"), y if r["leader_yes"] else 100-y, size) for w, taker, y, size in data["rows"]]
        minute = [x for x in mine if x[0] <= r["t"]+60]
        rows.append(dict(event=r["event"], ask=r["ask"], truncated=data["truncated"],
                         at_or_below=sum(x[3] for x in minute if x[2] <= r["ask"]),
                         takeable=sum(x[3] for x in minute if x[1] and x[2] <= r["ask"]),
                         takeable10=sum(x[3] for x in mine if x[1] and x[2] <= r["ask"] and x[0] <= r["t"]+10),
                         through=sum(x[3] for x in mine if x[2] < r["ask"]-1), at_bid=sum(x[3] for x in mine if x[2] == r["ask"]-1)))
    if rows:
        f = pd.DataFrame(rows)
        weeks = max(rule.week.nunique(), 1)
        say(f"  Trade record, random sample of {len(f)} paper-rule games ({int(f.truncated.sum())} cut off at 2,000 trades): in the first minute, contracts at or below the ask median {f.at_or_below.median():.0f} "
            f"mean {f.at_or_below.mean():.0f}; bought by takers at or below the ask median {f.takeable.median():.0f} mean {f.takeable.mean():.0f} "
            f"(100+ in {100*(f.takeable >= 100).mean():.0f}%, 1,000+ in {100*(f.takeable >= 1000).mean():.0f}%; within 10 s: 100+ in {100*(f.takeable10 >= 100).mean():.0f}%)")
        say(f"    dollars a taker could have matched: median ${(f.takeable*f.ask/100).median():,.0f} a game, about ${(f.takeable*f.ask/100).mean()*len(rule)/weeks:,.0f} a week at {len(rule)/weeks:.1f} games a week. "
            f"Printed strictly through a bid 1c under the ask within 3 minutes: median {f.through.median():.0f} contracts (any in {100*(f.through > 0).mean():.0f}%); printed at the bid: median {f.at_bid.median():.0f}")
        out["tape"] = dict(games=len(f), takeable_median=float(f.takeable.median()), takeable_mean=float(f.takeable.mean()),
                           dollars_week=float((f.takeable*f.ask/100).mean()*len(rule)/weeks), through_median=float(f.through.median()))
    return out


def alignment(league, say):
    games = pd.read_pickle(OUT/f"games-{league}.pkl")
    gap = (games.decided-games.last_wall).dropna()
    tight = (games[games.margin <= 3].decided-games[games.margin <= 3].last_wall).dropna()
    close = (games.close-games.last_wall)
    q = lambda v: "n/a" if not len(v) else "/".join(f"{x:.0f}" for x in np.percentile(v, [10, 50, 90]))
    say(f"\n2. {NAMES[league]}: clock alignment over {len(games)} joined games. Share of late plays stamped: median {100*games.share_stamped.median():.1f}%, "
        f"games fully stamped {100*(games.share_stamped >= .999).mean():.0f}%")
    say(f"   Kalshi close minus the last play's stamp (s, p10/p50/p90): {q(close)}; winner's bid at 98c+ for good minus the last play (all games): {q(gap)}; "
        f"games decided by 3 or fewer ({len(tight)}): {q(tight)}; of those within -60..+120 s: {100*((tight >= -60) & (tight <= 120)).mean() if len(tight) else float('nan'):.0f}%")


def staleness(leagues, say):
    """Trade prices around made baskets and turnovers in close late games, in the benefiting team's terms."""
    offsets = (-60, -30, -15, -5, 0, 5, 10, 20, 30, 45, 60, 120)
    waits = (0, 5, 15, 30, 45)
    say("\n4e. Staleness: last trade price around late scoring plays (lead 6 or less before the play, last 3 minutes or overtime), cents in the benefiting team's favour relative to 60 s before the play's stamp. The live poll found plays reach the public feed 25-41 s after their stamp")
    result = {}
    for league in leagues:
        cfg, rows = LEAGUES[league], []
        games = pd.read_pickle(OUT/f"games-{league}.pkl")
        for g in games.to_dict("records"):
            tape = load(tape_file(g["ticker"], "close"))
            data = load(game_file(league, g["game"]))
            if not tape or not tape["rows"] or not data:
                continue
            line = timeline(data)
            if line is None:
                continue
            walls, plays, _ = line
            times = np.array([r[0] for r in tape["rows"]])
            yes = np.array([r[2] for r in tape["rows"]])
            takers = [r[1] for r in tape["rows"]]
            lo_t = times[0] if tape["truncated"] else tape["start"]
            scoring_walls = [w for w, p in zip(walls, plays) if p[8]]
            for k in range(1, len(plays)):
                period, clock, home, away, _, _, kind, side, scoring, value, stamped = plays[k]
                before = plays[k-1]
                low = kind.lower()
                basket = scoring and "free" not in low
                turnover = "turnover" in low
                if not (basket or turnover) or side is None or not stamped:
                    continue
                if not (period > cfg["reg"] or clock <= 180) or abs(before[2]-before[3]) > 6:
                    continue
                w = walls[k]
                if w-60 < lo_t or w+60 > walls[-1] or w+120 > tape["stop"]:
                    continue
                gain = side if basket else other(side)
                sign_yes = (gain == "h") == g["yes_home"]
                path = []
                for off in offsets:
                    j = bisect_right(times, w+off)-1
                    path.append(None if j < 0 else (yes[j] if sign_yes else 100-yes[j]))
                n_near = int(((times >= w-60) & (times <= w+60)).sum())
                if path[0] is None or n_near < 20:
                    continue
                alone = sum(abs(x-w) <= 30 for x in scoring_walls) <= (1 if basket else 0)
                # the first taker purchase of the benefiting side after the stamp, and at stamp + 5 s
                buys = {}
                for d in waits:
                    j = bisect_right(times, w+d)
                    while j < len(times) and takers[j] != ("yes" if sign_yes else "no"):
                        j += 1
                    if j < len(times) and times[j] <= w+d+20:
                        paid = yes[j] if sign_yes else 100-yes[j]
                        k2 = bisect_right(times, times[j]+60)-1
                        later = yes[k2] if sign_yes else 100-yes[k2]
                        buys[d] = (paid, later-paid-fee(paid), 100*(g["yes_won"] == sign_yes)-paid-fee(paid))
                rows.append(dict(event=g["event"], kind="basket" if basket else "turnover", value=value, alone=alone,
                                 **{f"m{off}": path[i]-path[0] for i, off in enumerate(offsets)}, level=path[0],
                                 **{f"buy{d}": buys.get(d, (None,)*3)[1] for d in waits}, **{f"hold{d}": buys.get(d, (None,)*3)[2] for d in waits}))
        frame = pd.DataFrame(rows)
        result[league] = len(frame)
        if not len(frame):
            say(f"  {NAMES[league]}: no tapes")
            continue
        frame.to_pickle(OUT/f"staleness-{league}.pkl")
        say(f"  {NAMES[league]}: {frame.event.nunique()} games with a tape")
        for name, part in (("made baskets", frame[frame.kind == "basket"]), ("made baskets, no other score within 30 s", frame[(frame.kind == "basket") & frame.alone]),
                           ("made threes, alone", frame[(frame.kind == "basket") & frame.alone & (frame.value == 3)]),
                           ("turnovers, no score within 30 s", frame[(frame.kind == "turnover") & frame.alone])):
            if len(part) < 20:
                say(f"    {name}: {len(part)} plays")
                continue
            line = " ".join(f"{off:+d}s {boot(part, f'm{off}')[0]:+.2f}" for off in offsets[1:])
            total, at0 = boot(part, "m120"), boot(part, "m0")
            say(f"    {name} ({len(part)} plays, {part.event.nunique()} games): {line}")
            say(f"      move by +120 s {total[0]:+.2f}c [{total[1]:+.2f},{total[2]:+.2f}]; already in the price at the stamp {at0[0]:+.2f}c [{at0[1]:+.2f},{at0[2]:+.2f}] "
                f"({100*at0[0]/total[0] if total[0] else float('nan'):.0f}%), by +5 s {100*boot(part, 'm5')[0]/total[0] if total[0] else float('nan'):.0f}%, by +30 s {100*boot(part, 'm30')[0]/total[0] if total[0] else float('nan'):.0f}%")
            for d in waits:
                sub = part[part[f"buy{d}"].notna()]
                if len(sub) >= 20:
                    a, h = boot(sub, f"buy{d}"), boot(sub, f"hold{d}")
                    say(f"      copying the first taker purchase of the benefiting team at stamp+{d}s or later ({len(sub)}): last price 60 s later minus price paid minus entry fee {a[0]:+.2f}c [{a[1]:+.2f},{a[2]:+.2f}]; "
                        f"held to settlement net of fee {h[0]:+.2f}c [{h[1]:+.2f},{h[2]:+.2f}]")
    return result


def wp_filter(leagues, say):
    """Found after looking at 4c, so not a test: 95-99c leaders split by whether the archive's live win probability covers ask + fee."""
    say("\n6b. Post hoc: ask 95-99c in the final 10 minutes or overtime, split by the archive's live win probability (first qualifying minute per game)")
    pooled = {}
    for delay in (DELAYS[0], DELAYS[2]):
        for league in leagues:
            df = states_of(league, delay)[0]
            base = df[df.late & (df.ask >= 95) & (df.ask <= 99) & df.wp.notna()]
            edge = base.wp-base.ask-fee(base.ask)
            for label, part in (("covers ask + fee", base[edge >= 0]), ("does not", base[edge < 0])):
                c = cell(part)
                if c is None:
                    continue
                one = first_minute(part)
                weeks = max(one.week.nunique(), 1)
                say(f"  delay {delay}s {NAMES[league]:15s} {label:16s}: {fmt(c)} | {len(one)/weeks:.1f} a week; next-minute contracts at or below the ask "
                    f"median {one.at_or_below.median():.0f} (with straddling bars {(one.at_or_below+one.straddle).median():.0f})")
                pooled.setdefault((delay, label), []).append(one.assign(event=league+one.event))
        for label in ("covers ask + fee", "does not"):
            both = pd.concat(pooled[delay, label])
            b = boot(both)
            say(f"  delay {delay}s all three leagues  {label:16s}: games {len(both)} ask {both.ask.mean():.1f} won {100*both.won.mean():.2f}% net {b[0]:+.2f}c [{b[1]:+.2f},{b[2]:+.2f}] = {100*b[0]/both.ask.mean():+.2f}%; "
                f"halves {both[both.half == 1].net.mean():+.2f}c / {both[both.half == 2].net.mean():+.2f}c")


def live_summary(say):
    files = sorted((RAW/"live").glob("*.json")) if (RAW/"live").exists() else []
    say("\n4c (live check). Is the archive's win probability served while the game is on, and how late do plays appear?")
    if not files:
        say("  no live poll was made")
    for file in files:
        snap = load(file)
        fresh = [v for v in snap["plays"].values() if v["seen"] and v["stamp"]]
        if not fresh:
            continue
        lags = np.array([v["seen"]-v["stamp"] for v in fresh])
        line = (f"  {snap['league']} {snap['name']}: {len(fresh)} new plays; first served p10/p50/p90 {np.percentile(lags, 10):.0f}/{np.percentile(lags, 50):.0f}/{np.percentile(lags, 90):.0f} s after their stamp "
                f"(polled every ~5 s); a win probability came with {sum(v['wp'] is not None for v in fresh)} of them")
        if "final_wp" in snap:
            both = [(v["wp"], snap["final_wp"].get(k)) for k, v in snap["plays"].items() if v["wp"] is not None and v["seen"]]
            both = [(a, b) for a, b in both if b is not None]
            line += f"; after the game {len(both)} of those values were still listed and {sum(abs(a-b) > .005 for a, b in both)} had changed by more than half a point"
        say(line)


def report(leagues):
    lines, summary = [], {}

    def say(text=""):
        print(text, flush=True)
        lines.append(text)
    say(f"Late-game leaders: play-by-play joined to Kalshi quotes. Run {datetime.now(UTC):%Y-%m-%d %H:%M} UTC. Kalshi network reads used: {kalshi_used()}")
    say("\n1. Coverage")
    for league in leagues:
        cover = load(OUT/f"coverage-{league}.json")
        df, _ = states_of(league)
        say(f"  {NAMES[league]}: {json.dumps(cover)}")
        say(f"     joined games {df.event.nunique()} from {pd.to_datetime(df.settle.min(), unit='s'):%Y-%m-%d} to {pd.to_datetime(df.settle.max(), unit='s'):%Y-%m-%d}, "
            f"{df.week.nunique()} weeks with games, {len(df)} late minutes with an offer under 100c")
    for league in leagues:
        alignment(league, say)
    for league in leagues:
        df, split = states_of(league)
        res = dict(calibration=calibration(league, df, say))
        oos, label = fair_values(league, df, split, say)
        res["cheap_model"] = cheap(league, df, oos, "fair_model", "Smooth model", say)
        res["cheap_table"] = cheap(league, df, oos, "fair_table", "Plain table (cells with 300+ fitted states)", say)
        res["espn"] = espn_wp(league, df, say)
        res["maker"] = maker(league, df, say)
        res["capacity"] = capacity(league, df, say)
        res["search"] = search(league, df, say)
        for delay in DELAYS[1:]:
            slow = states_of(league, delay)[0]
            slow_rule = slow[(slow.lead >= 6) & (slow.ask >= 97) & (slow.ask <= 99)]
            say(f"\n   Sensitivity, play-by-play used {delay} s after its stamp instead of {DELAYS[0]} s. Paper rule: {fmt(cell(slow_rule))}")
            for name, lo, hi in BANDS[:4]:
                part = slow[slow.late & (slow.ask >= lo) & (slow.ask <= hi)]
                say(f"     ask {name}: {fmt(cell(part))}")
        rule = df[(df.lead >= 6) & (df.ask >= 97) & (df.ask <= 99)]
        if len(rule):
            one = first_minute(rule)
            a = one.ask.mean()
            say(f"\n   Loss arithmetic at the paper rule's mean ask {a:.1f}c: a win earns {100-a-fee(a):.2f}c, a loss costs {a+fee(a):.2f}c, so one loss erases {(a+fee(a))/(100-a-fee(a)):.0f} wins; "
                f"observed {int((1-one.won).sum())} losses in {len(one)} games (1 in {len(one)/max((1-one.won).sum(), 1):.0f}); break-even is 1 in {100/(100-a-fee(a)):.0f}")
        summary[league] = res
    wp_filter(leagues, say)
    summary["staleness"] = staleness(leagues, say)
    live_summary(say)
    (OUT/"report.txt").write_text("\n".join(lines)+"\n")
    (OUT/"summary.json").write_text(json.dumps(summary, default=lambda o: None if isinstance(o, float) and math.isnan(o) else str(o)))


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("command", choices=("listing", "espn", "prior", "bars", "build", "trades", "live", "livecheck", "report"))
    parser.add_argument("--leagues", nargs="+", choices=tuple(LEAGUES), default=list(LEAGUES))
    parser.add_argument("--cap", type=int, default=4000, help="total Kalshi network reads this script may ever make")
    parser.add_argument("--limit", type=int, default=10**9)
    parser.add_argument("--minutes", type=float, default=8)
    args = parser.parse_args()
    if args.command == "listing":
        listing(args.cap)
    elif args.command == "espn":
        espn(args.leagues, args.limit)
    elif args.command == "prior":
        prior(args.limit)
    elif args.command == "bars":
        bars(args.leagues, args.cap, args.limit)
    elif args.command == "build":
        build(args.leagues)
    elif args.command == "trades":
        trades(args.leagues, args.cap, args.limit)
    elif args.command == "report":
        report(args.leagues)
    elif args.command == "live":
        live(args.minutes)
    elif args.command == "livecheck":
        livecheck()


if __name__ == "__main__":
    main()
