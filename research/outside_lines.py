"""Does an OUTSIDE estimate of fair value improve the heavy-favourite edge in college basketball?

So far the favourite rules use only Kalshi's own prices. This script attaches
three outside numbers to each Kalshi college-basketball game (men, KXNCAAMBGAME,
and women, KXNCAAWBGAME) and asks whether they sort the winners.

Outside source: ESPN's public JSON (no key, no login)
  site.api.espn.com/apis/site/v2/sports/basketball/{league}/scoreboard?dates=YYYYMMDD&groups=50
      every Division I game of a day: ids, teams, start, final score
  sports.core.api.espn.com/v2/sports/basketball/leagues/{league}/events/{id}/competitions/{id}/odds
      the sportsbook line ESPN shows (ESPN BET to early December 2025, DraftKings after): point spread and
      moneyline at the OPEN and at the CLOSE. "Live Odds" providers are ignored.
  .../predictor
      ESPN's BPI pre-game win probability, with the time it was last computed.
Requests are one at a time, PACE seconds apart, with a generic User-Agent. The scoreboard is cached slimmed to the
fields used (a full day is about 1 MB); odds and predictor responses are cached whole. All under data/raw/outside-lines/.

Kalshi side (cached rows of earlier studies; nothing is read from Kalshi)
  men    data/processed/favourite-contexts/rows-cbb-men.json (pre-game "tip" price), dip orders/fills from
         data/processed/jit-bids/orders.pkl (group CBB; the 510 orders / 116 fills of research/dip_capacity.py)
  women  research/wcbb_backfill.py view "tip"; dip fills by its fill_of(v, 80) (667 orders / 176 fills)
  Price = the tip midpoint; ask = midpoint + half the spread. Only games with a midpoint of 80c to under 99c are fetched.

Matching: a Kalshi game is paired with the ESPN game of the same league that starts within WINDOW hours of Kalshi's
start and whose two team names both match (normalised names: "St." -> State/Saint, punctuation dropped, a short alias
list; the Kalshi team code against ESPN's abbreviation; otherwise the closest spelling, accepted only when clearly
better than the runner-up). Every pair is audited against the result: ESPN's winner must be Kalshi's settlement.

Outside probability of the Kalshi favourite
  spread  PRIMARY. Closing point spread s (favourite's points) -> Phi(s / sigma). sigma is the standard deviation of
          (favourite's final margin - closing spread) on the FIRST chronological half of matched games, per league.
          Used as primary because books post no moneyline on the heaviest favourites (spreads of about 30+).
  ml      closing moneylines, vig removed by scaling the two implied probabilities to sum to 1.
  open    opening spread, same sigma (known hours before the tip).
  bpi     ESPN BPI pre-game win probability (dropped when ESPN recomputed it after the scheduled start).
Each is tested two ways: as (outside - Kalshi ask), and as the RESIDUAL of the outside probability on a quadratic in
Kalshi's midpoint fitted on the first half of each league. The residual does not depend on how a spread is turned into
a probability at the 95%+ end, where sigma and the vig method move the number by more than the edge being measured.

Tests (taker fee 7*p*(1-p) cents; net = 100 x settlement - ask - fee; return = net / (ask + fee))
  3a basket  Kalshi favourites at 90-99c: net by tercile of (outside - ask) and as a slope; cuts and thresholds are
             chosen on the first chronological half and applied to the second.
  3b dip     the filled 80c bids: does the outside probability sort winners; a skip rule chosen on the first half.
  3c mirror  Kalshi favourites at 80-90c: the same as 3a.
Intervals resample games (one row per game), 4,000 draws. Four outside measures x three tests x two leagues are
looked at: single cells are not corrected for that and are flagged in the report.

    .venv/bin/python research/outside_lines.py download --minutes 9     (repeat until nothing is left)
    .venv/bin/python research/outside_lines.py report
    .venv/bin/python research/outside_lines.py forward                  (frozen out-of-sample test; rerun weekly, or daily before the tips)

`forward`: the one cell of 3a that looked good (men's 90-99c favourites whose OPENING-spread residual is at least
+2.65c: 63 of 63 won in the second half, one cell out of about 70 looked at) is tracked on games that started after
the data above ended. Everything the rule needs is frozen in FROZEN below and nothing is refitted. Each run lists
Kalshi's settled KXNCAAMBGAME markets closed since the freeze (public, unauthenticated, KALSHI_PACE seconds apart, long
pauses on HTTP 429), reads the one-minute bars of the 25 minutes before each new game's start, prices the favourite as
`report` does (median midpoint of the last 10-15 quotes ending two minutes before the start, spread at most 4c, range
at most 3c; start = expected expiration minus three hours), reads ESPN's opening spread, applies the rule and appends
to data/processed/outside-lines/forward.json. Games already in that file are not read again. One difference from
`report`: its rows also required a clean quote three hours before the game (research/favourite_contexts.py); `forward`
looks only at the quotes before the tip, as a live rule would.
Rule B (pre-registered 2026-11-19, logged beside the residual rule, history not rescored): the spread moved against the
favourite from open to close. The close is only known at the tip, so it is also scored on the last pre-tip scoreboard
line this script saved, where there is one. Opening spreads that have the favourite as underdog while the close has it
a heavy favourite are flagged as likely sign errors.
Look-ahead guard: ESPN's archive gives the opening spread without the time it was posted. So every run also saves the
opening spread ESPN's scoreboard shows for games that have NOT started yet (today and tomorrow), with the time seen; a
kept game is later marked "seen before the tip" only when such a snapshot exists. Run it before the tips for that.
"""
import argparse
from collections import Counter, defaultdict
from datetime import datetime, timedelta, timezone
from difflib import SequenceMatcher
import io
import json
import math
from pathlib import Path
import re
import sys
import time
from zoneinfo import ZoneInfo

import httpx
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
RAW = ROOT / "data/raw/outside-lines"
OUT = ROOT / "data/processed/outside-lines"
PROC = ROOT / "data/processed"
EASTERN = ZoneInfo("America/New_York")
LEAGUES = {"men": "mens-college-basketball", "women": "womens-college-basketball"}
SITE = "https://site.api.espn.com/apis/site/v2/sports/basketball"
CORE = "https://sports.core.api.espn.com/v2/sports/basketball/leagues"
HEADERS = {"User-Agent": "Mozilla/5.0 (compatible; research-script)", "Accept": "application/json"}
PACE, WINDOW, DRAWS = 1.05, 5.0, 4000
DEFAULT_SIGMA = {"men": 11.0, "women": 12.0}
MEASURES = (("spread", "closing spread"), ("ml", "closing moneyline, no vig"), ("open", "opening spread"), ("bpi", "ESPN BPI"))
RNG = np.random.default_rng(53)
ALIASES = {  # normalised Kalshi name -> normalised ESPN name, only where spelling rules do not get there
    "connecticut": "uconn", "mississippi": "ole miss", "massachusetts": "umass", "pittsburgh": "pitt",
    "appalachian state": "app state", "illinois chicago": "uic", "long island university": "liu", "long island": "liu",
    "fairleigh dickinson": "fdu", "southern illinois edwardsville": "siue", "siu edwardsville": "siue",
    "texas a and m corpus christi": "texas a and m cc", "ut martin": "ut martin", "tennessee martin": "ut martin",
    "louisiana monroe": "ul monroe", "ulm": "ul monroe", "hawaii": "hawai i", "san jose state": "san jose state",
    "nc state": "nc state", "north carolina state": "nc state", "southern california": "usc", "central florida": "ucf",
    "brigham young": "byu", "southern methodist": "smu", "texas christian": "tcu", "virginia commonwealth": "vcu",
    "nevada las vegas": "unlv", "louisiana state": "lsu", "american": "american university", "miami fl": "miami",
    "miami florida": "miami", "miami ohio": "miami oh", "loyola chicago": "loyola chicago", "loyola maryland": "loyola maryland",
    "loyola md": "loyola maryland", "loyola il": "loyola chicago", "saint francis pa": "saint francis", "queens": "queens university",
    "st thomas": "saint thomas minnesota", "saint thomas": "saint thomas minnesota", "seattle": "seattle u",
    "nebraska omaha": "omaha", "missouri kansas city": "kansas city", "umkc": "kansas city", "texas arlington": "ut arlington",
    "texas rio grande valley": "ut rio grande valley", "utrgv": "ut rio grande valley", "california baptist": "california baptist",
    "cal baptist": "california baptist", "sam houston state": "sam houston", "college of charleston": "charleston",
    "southeastern louisiana": "se louisiana", "southeast missouri state": "southeast missouri state", "albany": "ualbany", "university at albany": "ualbany",
    "pennsylvania": "penn", "mcneese state": "mcneese", "nicholls state": "nicholls", "grambling state": "grambling",
    "purdue fort wayne": "purdue fort wayne", "iu indianapolis": "iu indianapolis", "iupui": "iu indianapolis",
    "florida international": "fiu", "middle tennessee state": "middle tennessee", "western kentucky": "western kentucky",
    "east tennessee state": "east tennessee state", "etsu": "east tennessee state", "detroit": "detroit mercy",
    "st francis pa": "saint francis", "army": "army", "prairie view": "prairie view a and m", "southern university": "southern",
    "texas southern": "texas southern", "uncw": "unc wilmington", "uncg": "unc greensboro", "unca": "unc asheville",
    "north carolina wilmington": "unc wilmington", "north carolina greensboro": "unc greensboro", "north carolina asheville": "unc asheville",
    "south carolina upstate": "south carolina upstate", "usc upstate": "south carolina upstate", "maryland eastern shore": "maryland eastern shore",
    "md eastern shore": "maryland eastern shore", "southern mississippi": "southern miss", "louisiana lafayette": "louisiana",
    "arkansas little rock": "little rock", "tennessee chattanooga": "chattanooga", "wisconsin green bay": "green bay",
    "wisconsin milwaukee": "milwaukee", "st peters": "saint peters", "mount st marys": "mount saint marys", "csu northridge": "cal state northridge",
    "csu fullerton": "cal state fullerton", "csu bakersfield": "cal state bakersfield", "csun": "cal state northridge", "bowling green state": "bowling green",
}
last_request = [0.0]
buffer = io.StringIO()


def say(text=""):
    print(text, flush=True)
    buffer.write(text+"\n")


# ---------------------------------------------------------------- Kalshi side (cached rows only)

def universe():
    """One dict per Kalshi game priced 80c to under 99c at the tip: league, event, start, settle, mid, ask, payout,
    fav, opp, fav_code, opp_code, order (a 90-99c dip order of the earlier backtests), fill (its 80c bid filled)."""
    from research import jit_bids as jit
    from research import wcbb_backfill as w
    games = []
    dips = {r["event"]: r for r in jit.load() if r["group"] == "CBB"}
    for r in json.loads((PROC/"favourite-contexts/rows-cbb-men.json").read_text()).values():
        if r.get("skip") or not r.get("full") or r.get("tip_price") is None or not r.get("begin") or not 80 <= r["tip_price"] < 99:
            continue
        if r["payout"] not in (0.0, 1.0):
            continue
        d = dips.get(r["event"])
        games.append(dict(league="men", event=r["event"], start=r["begin"], settle=r["settle"], mid=r["tip_price"],
                          ask=r["tip_price"]+r["tip_spread"]/2, payout=r["payout"], fav=r["fav"], opp=r["opp"], fav_code=r["fav_code"],
                          order=d is not None, fill=bool(d and d["fill_ts"] is not None), archived=r["archived"]))
    names = {r["event"]: r for r in w.load()}
    for v in w.view(w.load(), "tip"):
        if not 80 <= v["price"] < 99:
            continue
        r = names[v["event"]]
        order = 90 <= v["price"] < 99
        games.append(dict(league="women", event=v["event"], start=v["est_start"], settle=v["settle"], mid=v["price"],
                          ask=v["price"]+v["spread"]/2, payout=v["payout"], fav=r["fav"], opp=r["opp"], fav_code=v["fav_code"],
                          order=order, fill=bool(order and w.fill_of(v, 80)), archived=v["archived"]))
    for g in games:
        pair = g["event"].split("-")[1][7:]
        code = g["fav_code"]
        g["opp_code"] = pair[len(code):] if pair.startswith(code) and not pair.endswith(code) else pair[:-len(code)] if pair.endswith(code) else ""
        g["day"] = (datetime.fromtimestamp(g["start"], EASTERN)-timedelta(hours=5)).strftime("%Y-%m-%d")
    return sorted(games, key=lambda g: (g["start"], g["event"]))


# ---------------------------------------------------------------- ESPN reads

def fetch(client, url, params=None):
    """A public GET, PACE seconds after the last one; None when the answer is not JSON 200 after three tries."""
    for pause in (0, 20, 60):
        time.sleep(pause)
        wait = last_request[0]+PACE-time.monotonic()
        if wait > 0:
            time.sleep(wait)
        try:
            resp = client.get(url, params=params)
        except httpx.RequestError:
            resp = None
        last_request[0] = time.monotonic()
        if resp is not None and resp.status_code == 200:
            return resp.json()
        if resp is not None and resp.status_code == 404:
            return {"missing": 404}
    return None


def board_file(league, day):
    return RAW/league/"scoreboard"/f"{day}.json"


def slim(data):
    out = []
    for e in data.get("events") or []:
        c = e["competitions"][0]
        teams = [dict(id=t["team"]["id"], abbr=t["team"].get("abbreviation", ""), location=t["team"].get("location", ""),
                      short=t["team"].get("shortDisplayName", ""), display=t["team"].get("displayName", ""), name=t["team"].get("name", ""),
                      home=t["homeAway"] == "home", winner=t.get("winner"), score=t.get("score")) for t in c["competitors"]]
        out.append(dict(id=e["id"], date=e["date"], status=e["status"]["type"]["name"], neutral=c.get("neutralSite"), teams=teams,
                        board_odds=bool(c.get("odds"))))
    return out


def board_days(games):
    """Every (league, YYYYMMDD) worth reading: the US Eastern date of each Kalshi start and the date in its ticker."""
    days = set()
    for g in games:
        days.add((g["league"], datetime.fromtimestamp(g["start"], EASTERN).strftime("%Y%m%d")))
        days.add((g["league"], datetime.strptime(g["event"].split("-")[1][:7], "%y%b%d").strftime("%Y%m%d")))
    return sorted(days)


def boards(games):
    out = defaultdict(dict)
    for league, day in board_days(games):
        f = board_file(league, day)
        if f.exists():
            for e in json.loads(f.read_text()):
                out[league][e["id"]] = e
    return out


def detail_file(league, kind, espn_id):
    return RAW/league/kind/f"{espn_id}.json"


def download(minutes):
    deadline = time.monotonic()+60*minutes
    games = universe()
    count = Counter()
    with httpx.Client(headers=HEADERS, timeout=30, follow_redirects=True) as client:
        for league, day in board_days(games):
            f = board_file(league, day)
            if f.exists():
                continue
            if time.monotonic() > deadline:
                return print(f"time is up during scoreboards: {dict(count)}")
            data = fetch(client, f"{SITE}/{LEAGUES[league]}/scoreboard", dict(dates=day, groups=50, limit=500))
            if data is None or "events" not in data:
                count["scoreboard failed"] += 1
                continue
            f.parent.mkdir(parents=True, exist_ok=True)
            f.write_text(json.dumps(slim(data)))
            count["scoreboards"] += 1
        pairs, _ = match(games, boards(games))
        todo = [(kind, g) for kind in ("odds", "predictor") for g in games if g["event"] in pairs
                and not detail_file(g["league"], kind, pairs[g["event"]]["id"]).exists()]
        print(f"matched {len(pairs)} of {len(games)}; detail reads to do {len(todo)}", flush=True)
        for kind, g in todo:
            if time.monotonic() > deadline:
                break
            espn_id = pairs[g["event"]]["id"]
            data = fetch(client, f"{CORE}/{LEAGUES[g['league']]}/events/{espn_id}/competitions/{espn_id}/{kind}")
            if data is None:
                count[kind+" failed"] += 1
                continue
            f = detail_file(g["league"], kind, espn_id)
            f.parent.mkdir(parents=True, exist_ok=True)
            f.write_text(json.dumps(data))
            count[kind] += 1
        left = sum(not detail_file(g["league"], kind, pairs[g["event"]]["id"]).exists() for kind, g in todo)
    print(f"done: {dict(count)}; detail reads left {left}")


# ---------------------------------------------------------------- matching

def norm(name):
    s = name.lower().replace("&", " and ").replace("-", " ").replace("ʻ", " ").replace("'", "").replace("’", "")
    s = re.sub(r"[.()]", " ", s)
    words = s.split()
    out = []
    for i, word in enumerate(words):
        if word in ("st", "saint"):
            out.append("saint" if i == 0 or (i == 1 and words[0] == "mount") else "state")
        elif word == "univ":
            out.append("university")
        elif word == "mt":
            out.append("mount")
        else:
            out.append(word)
    s = " ".join(out)
    return ALIASES.get(s, s)


def names_of(team):
    mascot = team["display"][:-len(team["name"])].strip() if team["name"] and team["display"].endswith(team["name"]) else team["display"]
    return {norm(x) for x in (team["location"], team["short"], mascot) if x}


def likeness(name, code, team):
    k = norm(name)
    names = names_of(team)
    if k in names:
        return 1.0
    best = max(SequenceMatcher(None, k, n).ratio() for n in names)
    if code and code == team["abbr"].upper():
        best = max(best, 0.93)
    return best


def match(games, by_league):
    """event -> the ESPN game (with 'flip': the Kalshi favourite is ESPN's second team, and 'how'), and the misses."""
    pairs, missed = {}, []
    for g in games:
        scored = []
        for e in by_league[g["league"]].values():
            if len(e["teams"]) != 2:
                continue
            hours = abs(datetime.fromisoformat(e["date"].replace("Z", "+00:00")).timestamp()-g["start"])/3600
            if hours > 30:
                continue
            a, b = e["teams"]
            straight = min(likeness(g["fav"], g["fav_code"], a), likeness(g["opp"], g["opp_code"], b))
            flipped = min(likeness(g["fav"], g["fav_code"], b), likeness(g["opp"], g["opp_code"], a))
            scored.append((max(straight, flipped), -hours, flipped > straight, e))
        near = sorted((s for s in scored if -s[1] <= WINDOW), key=lambda s: s[:2], reverse=True)
        far = sorted((s for s in scored if -s[1] > WINDOW and s[0] >= 0.93), key=lambda s: s[:2], reverse=True)
        pick = None
        if near and near[0][0] >= 0.72 and (len(near) == 1 or near[0][0]-near[1][0] >= 0.08 or near[0][0] == 1.0):
            pick = near[0]
        elif far:
            pick = far[0]
        if pick is None:
            missed.append(dict(event=g["event"], fav=g["fav"], opp=g["opp"], mid=g["mid"],
                               best=[(round(s[0], 2), round(-s[1], 1), s[3]["teams"][0]["location"], s[3]["teams"][1]["location"]) for s in near[:2]]))
            continue
        e = dict(pick[3])
        e["flip"], e["hours"] = pick[2], -pick[1]
        e["how"] = ("exact names" if pick[0] == 1.0 else "team code" if pick[0] == 0.93 else "closest spelling")+("" if -pick[1] <= WINDOW else ", start far off")
        pairs[g["event"]] = e
    return pairs, missed


# ---------------------------------------------------------------- outside probabilities

def american(node, key):
    """American odds or a point spread out of the core API's {'american': '-110'} nodes."""
    x = ((node or {}).get(key) or {}).get("american")
    if x in (None, "", "OFF"):
        return None
    try:
        return 0.0 if str(x).upper() in ("EVEN", "PK", "PICK") else float(str(x).replace("+", ""))
    except ValueError:
        return None


def implied(odds):
    return None if odds is None or odds == 0 else (100/(odds+100) if odds > 0 else -odds/(-odds+100))


def lines(g, e):
    """What the book and BPI said about the Kalshi favourite: provider, closing/opening spread (points the favourite
    is favoured by), no-vig closing moneyline probability, BPI probability, and the favourite's final margin."""
    out = dict(provider=None, close=None, open=None, ml=None, bpi=None, bpi_late=False, margin=None)
    fav, opp = (e["teams"][1], e["teams"][0]) if e["flip"] else (e["teams"][0], e["teams"][1])
    try:
        out["margin"] = float(fav["score"])-float(opp["score"])
    except (TypeError, ValueError):
        pass
    f = detail_file(g["league"], "odds", e["id"])
    items = [i for i in (json.loads(f.read_text()).get("items") or []) if "live" not in i.get("provider", {}).get("name", "").lower()] if f.exists() else []
    out["read"] = f.exists()
    for item in sorted(items, key=lambda i: i.get("provider", {}).get("priority", 9)):
        side, other = ("homeTeamOdds", "awayTeamOdds") if fav["home"] else ("awayTeamOdds", "homeTeamOdds")
        mine, theirs = item.get(side) or {}, item.get(other) or {}
        close = american(mine.get("close"), "pointSpread")
        if close is None and item.get("spread") is not None:
            close = item["spread"] if fav["home"] else -item["spread"]
        opened = american(mine.get("open"), "pointSpread")
        a, b = mine.get("moneyLine"), theirs.get("moneyLine")
        if a is None or b is None:
            a, b = american(mine.get("close"), "moneyLine"), american(theirs.get("close"), "moneyLine")
        if close is None and a is None:
            continue
        out["provider"] = item["provider"]["name"]
        out["close"] = None if close is None else -close
        out["open"] = None if opened is None else -opened
        if implied(a) and implied(b):
            out["ml"] = 100*implied(a)/(implied(a)+implied(b))
        break
    f = detail_file(g["league"], "predictor", e["id"])
    if f.exists():
        data = json.loads(f.read_text())
        node = data.get("homeTeam" if fav["home"] else "awayTeam") or {}
        stats = {s["name"]: s.get("value") for s in node.get("statistics") or []}
        value = stats.get("gameProjection", stats.get("teampredwinpct"))
        if value is not None:
            out["bpi"] = float(value)
            modified = data.get("lastModified")
            if modified:
                out["bpi_late"] = datetime.fromisoformat(modified.replace("Z", "+00:00")) > datetime.fromisoformat(e["date"].replace("Z", "+00:00"))
    return out


def phi(x):
    return 0.5*(1+math.erf(x/math.sqrt(2)))


# ---------------------------------------------------------------- statistics

def fee(ask):
    return 7*(ask/100)*(1-ask/100)


def boot(values, draws=DRAWS):
    a = np.asarray(values, float)
    if len(a) < 5:
        return (float("nan"), float("nan"))
    return tuple(float(x) for x in np.percentile(a[RNG.integers(0, len(a), (draws, len(a)))].mean(1), [2.5, 97.5]))


def slope(x, y, draws=DRAWS):
    """OLS slope of y on x and its interval from resampling games."""
    x, y = np.asarray(x, float), np.asarray(y, float)
    if len(x) < 10 or np.ptp(x) == 0:
        return float("nan"), (float("nan"), float("nan"))
    def fit(i):
        xs, ys = x[i], y[i]
        v = xs.var()
        return float("nan") if v == 0 else float(((xs-xs.mean())*(ys-ys.mean())).mean()/v)
    idx = RNG.integers(0, len(x), (draws, len(x)))
    sims = [fit(i) for i in idx]
    return fit(np.arange(len(x))), tuple(float(v) for v in np.nanpercentile(sims, [2.5, 97.5]))


def partial(x, control, y):
    """Slope of y on x with `control` also in the regression, with a game-resampled interval."""
    X = np.column_stack([np.ones(len(x)), np.asarray(x, float), np.asarray(control, float)])
    y = np.asarray(y, float)
    if len(y) < 15:
        return float("nan"), (float("nan"), float("nan"))
    def fit(i):
        return float(np.linalg.lstsq(X[i], y[i], rcond=None)[0][1])
    sims = [fit(i) for i in RNG.integers(0, len(y), (1000, len(y)))]
    return fit(np.arange(len(y))), tuple(float(v) for v in np.nanpercentile(sims, [2.5, 97.5]))


def cell(rows, cost="ask"):
    """n, mean cost, win rate, net cents with interval, % return. cost 'ask': taker at the ask with the fee; 'dip': an 80c maker fill, no fee."""
    if not rows:
        return dict(n=0)
    if cost == "ask":
        nets = [100*r["payout"]-r["ask"]-fee(r["ask"]) for r in rows]
        paid = float(np.mean([r["ask"]+fee(r["ask"]) for r in rows]))
    else:
        nets = [100*r["payout"]-80 for r in rows]
        paid = 80.0
    lo, hi = boot(nets)
    return dict(n=len(rows), price=float(np.mean([r["ask"] for r in rows])), win=float(np.mean([r["payout"] for r in rows])),
                net=float(np.mean(nets)), lo=lo, hi=hi, ret=100*float(np.mean(nets))/paid, ret_lo=100*lo/paid, ret_hi=100*hi/paid)


def text(c):
    if not c.get("n"):
        return "n    0"
    return (f"n {c['n']:4d} ask {c['price']:5.1f} won {100*c['win']:5.1f}% net {c['net']:+6.2f}c [{c['lo']:+6.2f},{c['hi']:+6.2f}]c"
            f" = {c['ret']:+6.2f}% [{c['ret_lo']:+6.2f},{c['ret_hi']:+6.2f}]%")


def halves(rows):
    rows = sorted(rows, key=lambda r: (r["start"], r["event"]))
    return rows[:len(rows)//2], rows[len(rows)//2:]


def span(rows):
    return f"{rows[0]['day']}..{rows[-1]['day']}" if rows else "-"


def split_test(rows, key, cost, label, out, min_keep=25):
    """Terciles (cuts from the first half), slope, and a keep-above threshold picked on the first half and scored on the second."""
    rows = [r for r in rows if r.get(key) is not None]
    result = dict(n=len(rows))
    if len(rows) < 40:
        say(f"    {label}: only {len(rows)} games with this number; not split")
        return result
    first, second = halves(rows)
    cuts = [float(c) for c in np.percentile([r[key] for r in first], [100/3, 200/3])]
    net = (lambda r: 100*r["payout"]-r["ask"]-fee(r["ask"])) if cost == "ask" else (lambda r: 100*r["payout"]-80)
    say(f"    {label}: {len(rows)} games, tercile cuts from the first half ({span(first)}) at {cuts[0]:+.1f} and {cuts[1]:+.1f}; all {text(cell(rows, cost))}")
    result["terciles"] = []
    for name, test in (("low", lambda v: v < cuts[0]), ("middle", lambda v: cuts[0] <= v < cuts[1]), ("high", lambda v: v >= cuts[1])):
        part = [[r for r in h if test(r[key])] for h in (rows, first, second)]
        cells = [cell(p, cost) for p in part]
        say(f"      {name:6s} all    {text(cells[0])}")
        say(f"             first  {text(cells[1])}")
        say(f"             second {text(cells[2])}")
        result["terciles"].append(dict(name=name, all=cells[0], first=cells[1], second=cells[2]))
    result["slopes"] = {}
    for name, part in (("all", rows), ("first half", first), ("second half", second)):
        b, ci = slope([r[key] for r in part], [net(r) for r in part])
        c, cci = partial([r[key] for r in part], [r["ask"] for r in part], [net(r) for r in part])
        say(f"      slope, {name:11s}: {b:+.3f}c of net per unit [{ci[0]:+.3f},{ci[1]:+.3f}]; with the Kalshi ask also in the regression {c:+.3f} [{cci[0]:+.3f},{cci[1]:+.3f}]")
        result["slopes"][name] = dict(slope=b, ci=ci, partial=c, partial_ci=cci)
    best = None
    for q in range(10, 81, 5):
        t = float(np.percentile([r[key] for r in first], q))
        kept = [r for r in first if r[key] >= t]
        if len(kept) >= min_keep:
            m = float(np.mean([net(r) for r in kept]))
            if best is None or m > best[0]:
                best = (m, t, q)
    if best:
        _, t, q = best
        kept1, kept2 = [r for r in first if r[key] >= t], [r for r in second if r[key] >= t]
        drop2 = [r for r in second if r[key] < t]
        say(f"      keep-above rule picked on the first half: {key} >= {t:+.1f} (its {q}th percentile)")
        say(f"        first half kept (in sample)  {text(cell(kept1, cost))}")
        say(f"        second half kept             {text(cell(kept2, cost))}")
        say(f"        second half dropped          {text(cell(drop2, cost))}")
        say(f"        second half, everything      {text(cell(second, cost))}")
        if kept2 and drop2:
            diff = [float(np.mean([net(r) for r in np.array(kept2, object)[RNG.integers(0, len(kept2), len(kept2))]]))
                    - float(np.mean([net(r) for r in np.array(drop2, object)[RNG.integers(0, len(drop2), len(drop2))]])) for _ in range(2000)]
            lo, hi = np.percentile(diff, [2.5, 97.5])
            say(f"        second half kept minus dropped: {np.mean([net(r) for r in kept2])-np.mean([net(r) for r in drop2]):+.2f}c [{lo:+.2f},{hi:+.2f}]")
            result["rule"] = dict(threshold=t, percentile=q, first=cell(kept1, cost), second=cell(kept2, cost), second_dropped=cell(drop2, cost),
                                  second_all=cell(second, cost), diff_ci=[float(lo), float(hi)], months=dict(Counter(r["day"][:7] for r in rows if r[key] >= t)))
            say(f"        games passing by month (all periods): {dict(sorted(Counter(r['day'][:7] for r in rows if r[key] >= t).items()))}")
    return result


# ---------------------------------------------------------------- report

def build():
    games = universe()
    pairs, missed = match(games, boards(games))
    rows = []
    for g in games:
        e = pairs.get(g["event"])
        if e is None:
            continue
        r = dict(g, espn=e["id"], how=e["how"], hours=e["hours"], espn_teams=[t["location"] for t in e["teams"]], status=e["status"])
        r.update(lines(g, e))
        fav = e["teams"][1] if e["flip"] else e["teams"][0]
        r["espn_won"] = fav.get("winner")
        rows.append(r)
    return games, rows, missed


def report():
    OUT.mkdir(parents=True, exist_ok=True)
    games, rows, missed = build()
    summary = {}
    say("OUTSIDE LINES AGAINST KALSHI'S HEAVY FAVOURITES, college basketball (research/outside_lines.py; ESPN public JSON; nothing traded)")
    say("\n=== 1. SOURCE AND COVERAGE (Kalshi games priced 80c to under 99c at the tip)")
    for league in LEAGUES:
        part = [g for g in games if g["league"] == league]
        got = [r for r in rows if r["league"] == league]
        hows = Counter(r["how"] for r in got)
        agree = [r for r in got if r["espn_won"] is not None]
        wrong = [r for r in agree if bool(r["espn_won"]) != (r["payout"] == 1.0)]
        say(f"  {league}: Kalshi games {len(part)} ({span(part)}); matched to an ESPN game {len(got)} = {100*len(got)/max(1, len(part)):.1f}%  how: {dict(hows)}")
        say(f"     result audit: ESPN's winner equals Kalshi's settlement in {len(agree)-len(wrong)} of {len(agree)} matched games with a final"
            f"{'; DISAGREE: '+', '.join(r['event'] for r in wrong[:8]) if wrong else ''}")
        say(f"     median |ESPN start - Kalshi start| {np.median([r['hours'] for r in got])*60:.0f} min; detail files read for {sum(r['read'] for r in got)} of {len(got)}")
        say(f"     {'band':8s} {'games':>6s} {'matched':>8s} {'spread':>7s} {'moneyline':>10s} {'open':>6s} {'BPI':>6s}   (share of Kalshi games with each number)")
        for name, lo, hi in (("90-99c", 90, 99), ("80-90c", 80, 90)):
            for period, test in (("2025-26", lambda g: g["day"] < "2026-07"), ("Nov 2026", lambda g: g["day"] >= "2026-07")):
                k = [g for g in part if lo <= g["mid"] < hi and test(g)]
                m = [r for r in got if lo <= r["mid"] < hi and test(r)]
                if k:
                    say(f"     {name:8s} {len(k):6d} {100*len(m)/len(k):7.1f}% {100*sum(r['close'] is not None for r in m)/len(k):6.1f}% "
                        f"{100*sum(r['ml'] is not None for r in m)/len(k):9.1f}% {100*sum(r['open'] is not None for r in m)/len(k):5.1f}% "
                        f"{100*sum(r['bpi'] is not None and not r['bpi_late'] for r in m)/len(k):5.1f}%   {period}")
        say(f"     providers: {dict(Counter(r['provider'] for r in got if r['provider']))}; BPI recomputed after the scheduled start (dropped): {sum(r['bpi_late'] for r in got)}")
        dip = [g for g in part if g["order"]]
        dipm = [r for r in got if r["order"]]
        say(f"     dip orders {len(dip)} / fills {sum(g['fill'] for g in dip)}; matched {len(dipm)} / {sum(r['fill'] for r in dipm)}; fills with a closing spread "
            f"{sum(r['fill'] and r['close'] is not None for r in dipm)}, with BPI {sum(r['fill'] and r['bpi'] is not None and not r['bpi_late'] for r in dipm)}")
        summary[league] = dict(kalshi=len(part), matched=len(got), how=dict(hows), result_disagreements=[r["event"] for r in wrong])
    if missed:
        say(f"  unmatched examples ({len(missed)} in all): "+"; ".join(f"{m['event'].split('-')[1]} {m['fav']} v {m['opp']} (closest {m['best'][:1]})" for m in missed[:12]))

    # sigma from the first chronological half, then the outside probabilities in cents
    sigma = {}
    say("\n=== 2. FROM LINES TO PROBABILITIES")
    for league in LEAGUES:
        part = [r for r in rows if r["league"] == league and r["close"] is not None and r["margin"] is not None]
        first, _ = halves(part)
        errors = [r["margin"]-r["close"] for r in first]
        sigma[league] = float(np.std(errors)) if len(errors) >= 50 else DEFAULT_SIGMA[league]
        say(f"  {league}: sigma = sd of (favourite's margin - closing spread) on the first half of matched games = {sigma[league]:.2f} points "
            f"(n {len(errors)}, mean error {np.mean(errors) if errors else float('nan'):+.2f}); all games {np.std([r['margin']-r['close'] for r in part]) if part else float('nan'):.2f}")
    for r in rows:
        s = sigma[r["league"]]
        r["p_spread"] = None if r["close"] is None else 100*phi(r["close"]/s)
        r["p_open"] = None if r["open"] is None else 100*phi(r["open"]/s)
        r["p_ml"] = r["ml"]
        r["p_bpi"] = None if r["bpi"] is None or r["bpi_late"] else r["bpi"]
        for key, _ in MEASURES:
            r["x_"+key] = None if r["p_"+key] is None else r["p_"+key]-r["ask"]
    for league in LEAGUES:      # the part of each outside number that Kalshi's own price does not already carry
        for key, _ in MEASURES:
            part = [r for r in rows if r["league"] == league and r["p_"+key] is not None]
            first, _half = halves(part)
            fit = np.polyfit([r["mid"] for r in first], [r["p_"+key] for r in first], 2) if len(first) >= 30 else None
            for r in part:
                r["r_"+key] = None if fit is None else float(r["p_"+key]-np.polyval(fit, r["mid"]))
    say("  calibration inside this sample (Kalshi 80-99c favourites; mean outside probability against the share that won, and Kalshi's own ask)")
    for key, name in MEASURES:
        for league in LEAGUES:
            part = [r for r in rows if r["league"] == league and r["p_"+key] is not None]
            if len(part) < 20:
                continue
            gap = [r["x_"+key] for r in part]
            corr = float(np.corrcoef([r["p_"+key] for r in part], [r["mid"] for r in part])[0, 1])
            say(f"    {name:26s} {league:6s} n {len(part):4d} outside {np.mean([r['p_'+key] for r in part]):5.1f} Kalshi ask {np.mean([r['ask'] for r in part]):5.1f} won {100*np.mean([r['payout'] for r in part]):5.1f}%"
                f" | outside - ask: mean {np.mean(gap):+5.2f} sd {np.std(gap):4.2f} p10 {np.percentile(gap, 10):+5.1f} p90 {np.percentile(gap, 90):+5.1f} | correlation with Kalshi mid {corr:.2f}")
    summary["sigma"] = sigma

    tests = (("3a", "BASKET: Kalshi favourites at 90-99c, bought at the ask (taker fee paid)", lambda r: 90 <= r["mid"] < 99, "ask"),
             ("3b", "DIP: filled 80c bids on 90-99c favourites (net = 100 x settlement - 80, no fee)", lambda r: r["order"] and r["fill"], "dip"),
             ("3c", "MIRROR: Kalshi favourites at 80-90c, bought at the ask (taker fee paid)", lambda r: 80 <= r["mid"] < 90, "ask"))
    for code, title, keep, cost in tests:
        say(f"\n=== {code}. {title}")
        summary[code] = {}
        base = [r for r in rows if keep(r)]
        for league in ("men", "women", "both"):
            part = [r for r in base if league == "both" or r["league"] == league]
            first, second = halves(part)
            say(f"  {league}: matched games {len(part)}; all {text(cell(part, cost))}")
            say(f"     halves: {span(first)} {text(cell(first, cost))}")
            say(f"             {span(second)} {text(cell(second, cost))}")
            for key, name in MEASURES:
                field = "p_"+key if cost == "dip" else "x_"+key
                label = f"[{name}] "+("outside probability of the favourite" if cost == "dip" else "outside probability minus Kalshi ask, cents")
                summary[code][f"{league} {key}"] = split_test(part, field, cost, label, summary)
                if True:
                    summary[code][f"{league} {key} residual"] = split_test(part, "r_"+key, cost, f"[{name}] outside probability RESIDUAL on the Kalshi midpoint (quadratic fitted on the first half)", summary)
        if code == "3b":
            say("  Does the outside number predict WHICH orders fill? (90-99c dip orders; fill rate and return per ORDER by tercile of the closing-spread probability)")
            for league in ("men", "women"):
                part = [r for r in rows if r["order"] and r["league"] == league and r["p_spread"] is not None]
                if len(part) < 30:
                    continue
                first, _ = halves(part)
                cuts = np.percentile([r["p_spread"] for r in first], [100/3, 200/3])
                for name, test in (("low", lambda v: v < cuts[0]), ("middle", lambda v: cuts[0] <= v < cuts[1]), ("high", lambda v: v >= cuts[1])):
                    p = [r for r in part if test(r["p_spread"])]
                    f = [r for r in p if r["fill"]]
                    per_order = [(100*r["payout"]-80) if r["fill"] else 0.0 for r in p]
                    lo, hi = boot(per_order)
                    say(f"    {league:6s} {name:6s} (cuts {cuts[0]:.1f}/{cuts[1]:.1f}) orders {len(p):4d} fills {len(f):3d} = {100*len(f)/max(1, len(p)):4.1f}% | fills won "
                        f"{100*np.mean([r['payout'] for r in f]) if f else float('nan'):5.1f}% net {np.mean([100*r['payout']-80 for r in f]) if f else float('nan'):+5.1f}c a fill | {np.mean(per_order):+5.2f}c per order [{lo:+5.2f},{hi:+5.2f}]")

    say("\n=== 4. FREQUENCY: matched Kalshi games by month and band (last season is a sample of its games: men about 57% read, women about 56%; November 2026 is complete to the 18th)")
    for league in LEAGUES:
        for name, lo, hi in (("90-99c", 90, 99), ("80-90c", 80, 90)):
            part = [r for r in rows if r["league"] == league and lo <= r["mid"] < hi]
            say(f"  {league:6s} {name}: {dict(sorted(Counter(r['day'][:7] for r in part).items()))}")
    (OUT/"report.txt").write_text(buffer.getvalue())
    (OUT/"summary.json").write_text(json.dumps(summary, indent=1, default=str))
    keep = ("league", "event", "day", "start", "mid", "ask", "payout", "fav", "opp", "order", "fill", "espn", "how", "hours", "espn_teams", "provider",
            "close", "open", "ml", "bpi", "bpi_late", "margin", "p_spread", "p_open", "p_ml", "p_bpi")
    (OUT/"matched.json").write_text(json.dumps([{k: r.get(k) for k in keep} for r in rows], indent=0))
    (OUT/"unmatched.json").write_text(json.dumps(missed, indent=0))


# ---------------------------------------------------------------- forward test of one frozen cell

# Frozen on 2026-11-19 from the `report` run of that day (data/processed/outside-lines/summary.json and matched.json).
# Nothing below is refitted by `forward`.
FROZEN = dict(
    sigma=11.238753316082297,             # men: sd of (favourite's margin - closing spread), first half of matched games (n 443)
    fit=(0.013672833023704846, -1.4293244733263775, 105.46718767747727),   # np.polyfit(mid, p_open, 2) on the first half of men's
                                          # 80-99c games with an opening spread (443 games, 2025-11-03..2026-02-28); highest power first
    threshold=2.6537621638760527,         # residual cut: 80th percentile of the first half of men's 90-99c games (n 257), the
                                          # keep-above rule `report` picked ("men open residual" in section 3a)
    freeze=1794974400,                    # latest men's game start in matched.json (2026-11-18 04:00 UTC); only later starts count
)
SERIES, KALSHI_PACE = "KXNCAAMBGAME", 2.5
SUSPECT_CLOSE = 10.0      # an opening spread with the favourite as underdog is flagged when the close has it favoured by this many points
FORWARD = OUT/"forward.json"
kalshi_last = [0.0]


def kalshi(client, path, params):
    """A public, unauthenticated Kalshi read, KALSHI_PACE seconds after the last one, long pauses after HTTP 429."""
    import config
    for pause in (30, 60, 120, 240):
        wait = kalshi_last[0]+KALSHI_PACE-time.monotonic()
        if wait > 0:
            time.sleep(wait)
        try:
            resp = client.get(config.KALSHI_PROD_BASE+path, params=params)
        except httpx.RequestError:
            resp = None
        kalshi_last[0] = time.monotonic()
        if resp is not None and resp.status_code == 200:
            return resp.json()
        if resp is not None and resp.status_code != 429 and resp.status_code < 500:
            raise RuntimeError(f"{path} returned HTTP {resp.status_code}")
        print(f"   Kalshi throttled ({resp.status_code if resp is not None else 'network'}): pausing {pause}s", flush=True)
        time.sleep(pause)
    raise RuntimeError(f"Kalshi read failed after long back-off: {path}")


def stamp(text):
    return None if not text else int(datetime.fromisoformat(text.replace("Z", "+00:00")).timestamp())


def settled_events(client):
    """Settled two-outcome games of the series closed since the freeze: event -> dict(start, settle, markets)."""
    group, cursor = defaultdict(list), ""
    while True:
        params = dict(series_ticker=SERIES, status="settled", min_close_ts=FROZEN["freeze"], limit=1000)
        if cursor:
            params["cursor"] = cursor
        data = kalshi(client, "/markets", params)
        for m in data.get("markets") or []:
            group[m["event_ticker"]].append(m)
        cursor = data.get("cursor")
        if not cursor or not data.get("markets"):
            break
    out = {}
    for event, ms in group.items():
        ms.sort(key=lambda m: m["ticker"])
        try:
            values = [float(m["settlement_value_dollars"]) for m in ms]
        except (KeyError, TypeError, ValueError):
            continue
        expected = stamp(ms[0].get("expected_expiration_time"))
        if len(ms) != 2 or any(v not in (0.0, 1.0) for v in values) or not expected or expected-3*3600 <= FROZEN["freeze"]:
            continue
        out[event] = dict(start=expected-3*3600, settle=stamp(ms[0].get("settlement_ts")), markets=ms, outcome=values[0])
    return out


def tip_quote(bars, start):
    """(midpoint of the YES side, mean spread) from the last 10-15 one-minute quotes ending two minutes before the start, as
    research/favourite_contexts.py `study` does; None when there is no clean quote."""
    from research.near_end import cents
    late = []
    for b in sorted(bars, key=lambda b: b["end_period_ts"]):
        if b["end_period_ts"] <= start-120:
            bid, ask = cents(b, "yes_bid", "close"), cents(b, "yes_ask", "close")
            late.append(None if bid is None or ask is None or not 0 < bid <= ask < 100 else (bid, ask))
    late = late[-15:]
    if len(late) < 10 or not all(q and q[1]-q[0] <= 4 for q in late):
        return None
    mids = [(q[0]+q[1])/2 for q in late]
    if max(mids)-min(mids) > 3:
        return None
    return float(np.median(mids)), float(np.mean([q[1]-q[0] for q in late]))


def board_final(events):
    return bool(events) and all(e["status"].startswith(("STATUS_FINAL", "STATUS_POSTPONED", "STATUS_CANCELED", "STATUS_FORFEIT")) for e in events)


def binomial_tail(n, wins, p):
    """P(at least `wins` wins in n) when each wins with probability p."""
    return float(sum(math.comb(n, k)*p**k*(1-p)**(n-k) for k in range(wins, n+1)))


def side_text(rows):
    if not rows:
        return "n   0"
    cost = float(np.mean([r["ask"]+fee(r["ask"]) for r in rows]))
    net = float(np.mean([100*r["payout"]-r["ask"]-fee(r["ask"]) for r in rows]))
    return (f"n {len(rows):3d} won {sum(r['payout'] == 1.0 for r in rows):3d} lost {sum(r['payout'] != 1.0 for r in rows):2d} "
            f"mean ask {np.mean([r['ask'] for r in rows]):5.2f}c return {100*net/cost:+6.2f}% ({net:+.2f}c a bet, taker fee paid)")


def forward(minutes):
    deadline = time.monotonic()+60*minutes
    OUT.mkdir(parents=True, exist_ok=True)
    state = json.loads(FORWARD.read_text()) if FORWARD.exists() else dict(frozen=json.loads(json.dumps(FROZEN)), games={}, snapshots={})
    if state.get("frozen") != json.loads(json.dumps(FROZEN)):
        raise SystemExit("forward.json was built with different frozen constants; move it aside rather than mixing rules")
    games, snaps = state["games"], state["snapshots"]
    now = datetime.now(timezone.utc)
    count = Counter()

    def save():
        temp = FORWARD.with_suffix(".tmp")
        temp.write_text(json.dumps(state, indent=0))
        temp.replace(FORWARD)

    with httpx.Client(headers=HEADERS, timeout=30, follow_redirects=True) as client:
        # 1. look-ahead guard: what ESPN shows NOW for games that have not started
        for offset in (0, 1):
            day = (now.astimezone(EASTERN)+timedelta(days=offset)).strftime("%Y%m%d")
            data = fetch(client, f"{SITE}/{LEAGUES['men']}/scoreboard", dict(dates=day, groups=50, limit=500))
            for e in (data or {}).get("events") or []:
                c = e["competitions"][0]
                odds = (c.get("odds") or [None])[0]
                if e["status"]["type"]["name"] != "STATUS_SCHEDULED" or stamp(e["date"]) <= now.timestamp() or not odds:
                    continue
                home = ((odds.get("pointSpread") or {}).get("home") or {})
                opened, current = (home.get("open") or {}).get("line"), (home.get("close") or {}).get("line")
                if opened is None and current is None:
                    continue
                if e["id"] in snaps:        # keep the first sighting; refresh the LAST pre-tip line (rule B's usable "close")
                    snaps[e["id"]].update(last_seen_at=now.isoformat(timespec="seconds"), home_last=current, tip=e["date"])
                    count["pre-tip snapshots refreshed"] += 1
                    continue
                snaps[e["id"]] = dict(seen_at=now.isoformat(timespec="seconds"), tip=e["date"], home_open=opened, home_current=current,
                                      provider=(odds.get("provider") or {}).get("name"))
                count["pre-tip snapshots saved"] += 1
        save()

        # 2. new settled Kalshi games since the freeze
        with httpx.Client(timeout=30, headers={"User-Agent": HEADERS["User-Agent"]}) as kc:
            events = settled_events(kc)
            new = {k: v for k, v in events.items() if k not in games}
            count["settled Kalshi games after the freeze"] = len(events)
            count["new this run"] = len(new)
            by_start = defaultdict(list)
            for event, e in new.items():
                by_start[e["start"]].append(event)
            for start, names in sorted(by_start.items()):
                for i in range(0, len(names), 100):
                    if time.monotonic() > deadline:
                        break
                    part = names[i:i+100]
                    tickers = {new[k]["markets"][0]["ticker"]: k for k in part}
                    data = kalshi(kc, "/markets/candlesticks", dict(market_tickers=",".join(tickers), start_ts=start-25*60, end_ts=start, period_interval=1))
                    got = {x["market_ticker"]: x.get("candlesticks") or [] for x in data.get("markets") or []}
                    for ticker, event in tickers.items():
                        e, bars = new[event], got.get(ticker, [])
                        raw = RAW/"kalshi-forward"/f"{event}.json"
                        raw.parent.mkdir(parents=True, exist_ok=True)
                        raw.write_text(json.dumps(dict(ticker=ticker, start=start, candlesticks=bars)))
                        a, b = e["markets"]
                        row = dict(event=event, league="men", start=start, settle=e["settle"],
                                   day=(datetime.fromtimestamp(start, EASTERN)-timedelta(hours=5)).strftime("%Y-%m-%d"), status="no clean quote before the tip")
                        q = tip_quote(bars, start)
                        if q:
                            yes = q[0] > 50
                            fav, other = (a, b) if yes else (b, a)
                            code = fav["ticker"].rsplit("-", 1)[-1]
                            row.update(mid=q[0] if yes else 100-q[0], spread=q[1], payout=e["outcome"] if yes else 1-e["outcome"],
                                       fav=fav.get("yes_sub_title") or "", opp=other.get("yes_sub_title") or "", fav_code=code,
                                       opp_code=other["ticker"].rsplit("-", 1)[-1], status="priced")
                            row["ask"] = row["mid"]+row["spread"]/2
                        games[event] = row
                    save()

        # 3. ESPN opening spread for the priced 90-99c favourites not yet judged
        todo = [g for g in games.values() if g["status"] == "priced" and 90 <= g["mid"] < 99 and "kept" not in g]
        boards_now = defaultdict(dict)
        for league, day in board_days(todo):
            f = board_file(league, day)
            events_of_day = json.loads(f.read_text()) if f.exists() else None
            if events_of_day is None or not board_final(events_of_day):
                data = fetch(client, f"{SITE}/{LEAGUES[league]}/scoreboard", dict(dates=day, groups=50, limit=500))
                if data is None or "events" not in data:
                    continue
                events_of_day = slim(data)
                if board_final(events_of_day):
                    f.parent.mkdir(parents=True, exist_ok=True)
                    f.write_text(json.dumps(events_of_day))
            for e in events_of_day:
                boards_now[league][e["id"]] = e
        pairs, _ = match(todo, boards_now)
        for g in todo:
            if time.monotonic() > deadline:
                break
            e = pairs.get(g["event"])
            if e is None or not e["status"].startswith("STATUS_FINAL"):
                g["note"] = "no ESPN game matched yet"
                continue
            f = detail_file("men", "odds", e["id"])
            if not f.exists():
                data = fetch(client, f"{CORE}/{LEAGUES['men']}/events/{e['id']}/competitions/{e['id']}/odds")
                if data is None:
                    continue
                f.parent.mkdir(parents=True, exist_ok=True)
                f.write_text(json.dumps(data))
            x = lines(g, e)
            fav = e["teams"][1] if e["flip"] else e["teams"][0]
            g.update(espn=e["id"], espn_tip=e["date"], espn_won=fav.get("winner"), provider=x["provider"], open=x["open"], close=x["close"],
                     fav_home=fav["home"], judged_at=now.isoformat(timespec="seconds"))
            g.pop("note", None)
            if x["open"] is None:
                g.update(kept=None, residual=None, p_open=None)
            else:
                g["p_open"] = 100*phi(x["open"]/FROZEN["sigma"])
                g["residual"] = float(g["p_open"]-np.polyval(FROZEN["fit"], g["mid"]))
                g["kept"] = bool(g["residual"] >= FROZEN["threshold"])
            snap = snaps.get(e["id"])
            if snap and snap["home_open"] is not None and stamp(snap["seen_at"]) < stamp(e["date"]):
                try:
                    line = float(snap["home_open"])
                except ValueError:      # "PK" / "EVEN"
                    line = 0.0
                seen = -line if fav["home"] else line
                g["open_seen_before_tip"] = dict(seen_at=snap["seen_at"], hours_before=round((stamp(e["date"])-stamp(snap["seen_at"]))/3600, 1),
                                                 open_then=seen, same_as_archive=x["open"] is not None and abs(seen-x["open"]) < 0.01)
            else:
                g["open_seen_before_tip"] = None
            save()
        save()

    # ---- output (everything recorded so far, not just this run)
    done = [g for g in games.values() if g["status"] == "priced" and 90 <= g["mid"] < 99]
    judged = sorted((g for g in done if "kept" in g), key=lambda g: (g["start"], g["event"]))
    kept, dropped, no_line = [g for g in judged if g["kept"]], [g for g in judged if g["kept"] is False], [g for g in judged if g["kept"] is None]
    print("FORWARD TEST, frozen rule: men's college basketball favourites at 90-99c before the tip whose opening-spread residual is at least the threshold")
    print(f"  frozen constants: sigma {FROZEN['sigma']}, quadratic {FROZEN['fit']}, residual threshold {FROZEN['threshold']}, "
          f"games starting after {datetime.fromtimestamp(FROZEN['freeze'], timezone.utc):%Y-%m-%d %H:%M} UTC")
    print(f"  residual = 100 x Phi(opening spread / sigma) - (a x mid^2 + b x mid + c); kept when residual >= threshold")
    print(f"  this run: {dict(count)}")
    print(f"  recorded so far: Kalshi games {len(games)} ({Counter(g['status'] for g in games.values()).most_common()}); 90-99c favourites {len(done)}; "
          f"judged {len(judged)}; waiting for an ESPN match or line read {len(done)-len(judged)}; pre-tip snapshots on file {len(snaps)}")
    if judged:
        print(f"  span of judged games: {judged[0]['day']} .. {judged[-1]['day']}")
    print(f"  KEPT     {side_text(kept)}")
    print(f"  DROPPED  {side_text(dropped)}")
    print(f"  NO OPENING SPREAD (not in the test) {side_text(no_line)}")
    history = [r for r in json.loads((OUT/'matched.json').read_text()) if r["league"] == "men" and 90 <= r["mid"] < 99 and r.get("p_open") is not None
               and r["p_open"]-np.polyval(FROZEN["fit"], r["mid"]) >= FROZEN["threshold"]] if (OUT/"matched.json").exists() else []

    def verdict(name, rows, past=None):
        """Break-even win rate, the run needed, and the one-sided binomial p-value of a kept set."""
        for label, part in ((f"forward {name} bets", rows), (f"{name}'s historical bets (price only; outcomes not used)", past)):
            if not part:
                continue
            even = float(np.mean([r["ask"]+fee(r["ask"]) for r in part]))/100
            need = math.ceil(math.log(0.05)/math.log(even))
            with_one = next(n for n in range(need, 2000) if binomial_tail(n, n-1, even) < 0.05)
            print(f"  break-even win rate at the mean ask plus fee of {label}: {100*even:.2f}% -> {need} kept bets in a row with no loss for a one-sided p < 0.05; with one loss, {with_one} bets")
        if rows:
            even = float(np.mean([r["ask"]+fee(r["ask"]) for r in rows]))/100
            wins = sum(r["payout"] == 1.0 for r in rows)
            print(f"  one-sided binomial p-value, {name}: P(at least {wins} wins in {len(rows)} | break-even {100*even:.2f}%) = {binomial_tail(len(rows), wins, even):.3f}")

    def number(text):
        try:
            return float(text)
        except (TypeError, ValueError):
            return 0.0 if str(text).upper() in ("PK", "EVEN", "PICK") else None

    # per-game fields worked out at print time from what is stored (forward games only; history is not rescored)
    for g in judged:
        o, c = g.get("open"), g.get("close")
        g["suspect"] = o is not None and c is not None and o < 0 and c >= SUSPECT_CLOSE
        g["b"] = None if o is None or c is None else bool(c < o)
        snap = snaps.get(g.get("espn"))
        g["b_pretip"], g["pretip"] = None, None
        if snap and stamp(snap.get("last_seen_at") or snap["seen_at"]) < stamp(g["espn_tip"]):
            first, last = number(snap.get("home_open")), number(snap.get("home_last", snap.get("home_current")))
            if first is not None and last is not None:
                sign = -1 if g["fav_home"] else 1
                seen = snap.get("last_seen_at") or snap["seen_at"]
                g["pretip"] = dict(open=sign*first, last=sign*last, seen_at=seen, hours_before=round((stamp(g["espn_tip"])-stamp(seen))/3600, 1))
                g["b_pretip"] = bool(sign*last < sign*first)

    verdict("rule A", kept, history)
    seen = sum(bool(g.get("open_seen_before_tip")) for g in kept)
    print(f"  look-ahead guard: {seen} of {len(kept)} kept games had their opening spread saved from ESPN's scoreboard BEFORE the tip. For the others it cannot be told:"
          " ESPN's archive gives the opening spread with no time stamp. Run this command before the day's tips to record it.")
    print(f"  rule A kept games ({len(kept)}):")
    print(f"    {'date':10s} {'favourite':24s} {'opponent':24s} {'ask':>5s} {'open':>6s} {'resid':>6s} {'result':6s} open seen before the tip?")
    for g in kept:
        x = g.get("open_seen_before_tip")
        guard = f"yes, {x['hours_before']}h before ({x['open_then']:+.1f}{'' if x['same_as_archive'] else ', DIFFERS from the archive'})" if x else "cannot tell (no pre-tip snapshot)"
        print(f"    {g['day']:10s} {g['fav'][:24]:24s} {g['opp'][:24]:24s} {g['ask']:5.1f} {-g['open']:+6.1f} {g['residual']:+6.2f} {'won' if g['payout'] == 1.0 else 'LOST':6s} {guard}"
              f"{'  SUSPECT OPEN SIGN' if g['suspect'] else ''}")
    lost = [g for g in dropped if g["payout"] != 1.0]
    if lost:
        print("  rule A dropped games that lost: "+"; ".join(f"{g['day']} {g['fav']} ask {g['ask']:.1f} open {-g['open']:+.1f} resid {g['residual']:+.2f}" for g in lost))

    print("\nRULE B, frozen 2026-11-19 (pre-registered; history is not rescored here): men's 90-99c favourites whose spread moved AGAINST the favourite from open to"
          " close (the favourite is favoured by fewer points at the close than at the open). Same games and prices as rule A.")
    print("  The closing spread is only known at the tip. 'archive' scores the rule on ESPN's archived close (not knowable in time); 'pre-tip' scores it on the LAST"
          " spread this script saved from ESPN's scoreboard before the tip, for the games where it saved one.")
    both = [g for g in judged if g["b"] is not None]
    b_kept, b_rest = [g for g in both if g["b"]], [g for g in both if not g["b"]]
    print(f"  archive  KEPT (moved against)  {side_text(b_kept)}")
    print(f"  archive  REST (flat or toward) {side_text(b_rest)}")
    print(f"  archive  no open or no close (not in the test) n {len(judged)-len(both)}")
    verdict("rule B (archive close)", b_kept)
    pre = [g for g in judged if g["b_pretip"] is not None]
    p_kept, p_rest = [g for g in pre if g["b_pretip"]], [g for g in pre if not g["b_pretip"]]
    print(f"  pre-tip  games with a snapshot before the tip: {len(pre)} of {len(judged)}"+("" if pre else " (none yet: snapshots exist only for games that had not started when this command ran)"))
    print(f"  pre-tip  KEPT (moved against)  {side_text(p_kept)}")
    print(f"  pre-tip  REST (flat or toward) {side_text(p_rest)}")
    verdict("rule B (pre-tip snapshot)", p_kept)
    if pre:
        print(f"  pre-tip and archive versions agree on {sum(g['b_pretip'] == g['b'] for g in pre if g['b'] is not None)} of {sum(g['b'] is not None for g in pre)} games")
    print(f"  rule B kept games, archive close ({len(b_kept)}):")
    print(f"    {'date':10s} {'favourite':24s} {'opponent':24s} {'ask':>5s} {'open':>6s} {'close':>6s} {'result':6s} {'rule A':7s} last pre-tip snapshot")
    for g in b_kept:
        x = g["pretip"]
        guard = f"open {-x['open']:+.1f} last {-x['last']:+.1f}, {x['hours_before']}h before -> {'kept' if g['b_pretip'] else 'NOT kept'}" if x else "none (cannot be scored pre-tip)"
        print(f"    {g['day']:10s} {g['fav'][:24]:24s} {g['opp'][:24]:24s} {g['ask']:5.1f} {-g['open']:+6.1f} {-g['close']:+6.1f} {'won' if g['payout'] == 1.0 else 'LOST':6s} "
              f"{'kept' if g['kept'] else 'dropped':7s} {guard}{'  SUSPECT OPEN SIGN' if g['suspect'] else ''}")
    lost = [g for g in b_rest if g["payout"] != 1.0]
    if lost:
        print("  rule B rest games that lost: "+"; ".join(f"{g['day']} {g['fav']} ask {g['ask']:.1f} open {-g['open']:+.1f} close {-g['close']:+.1f}" for g in lost))

    print("\nRULE A x RULE B (archive close), games with both an opening and a closing spread")
    for a_name, a_test in (("A kept   ", True), ("A dropped", False)):
        for b_name, b_test in (("B kept", True), ("B rest", False)):
            print(f"  {a_name} / {b_name}  {side_text([g for g in both if g['kept'] is a_test and g['b'] is b_test])}")

    suspects = [g for g in judged if g["suspect"]]
    print(f"\nSUSPECT OPENING SPREADS (ESPN's open has the Kalshi favourite as the UNDERDOG while its close has it favoured by {SUSPECT_CLOSE:g}+ points; likely a sign error): {len(suspects)}")
    for g in suspects:
        print(f"    {g['day']} {g['fav']} v {g['opp']}: ask {g['ask']:.1f} open {-g['open']:+.1f} close {-g['close']:+.1f} | rule A {'kept' if g['kept'] else 'dropped'}, "
              f"rule B {'kept' if g['b'] else 'rest'}, {'won' if g['payout'] == 1.0 else 'LOST'}")
    if suspects:
        clean = [g for g in judged if not g["suspect"]]
        print("  the same blocks WITHOUT the suspect games:")
        print(f"    rule A KEPT    {side_text([g for g in clean if g['kept']])}")
        print(f"    rule A DROPPED {side_text([g for g in clean if g['kept'] is False])}")
        print(f"    rule B KEPT    {side_text([g for g in clean if g['b']])}")
        print(f"    rule B REST    {side_text([g for g in clean if g['b'] is False])}")
    else:
        print("  none among the forward games so far; the rule blocks above use every game")
    for g in judged:      # print-time fields are not part of the stored record, except the rule B verdicts, kept for the log
        g.pop("pretip", None)
    save()
    print(f"\n  saved {FORWARD}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=("download", "report", "forward"))
    parser.add_argument("--minutes", type=float, default=9)
    args = parser.parse_args()
    {"download": lambda: download(args.minutes), "report": report, "forward": lambda: forward(args.minutes)}[args.command]()


if __name__ == "__main__":
    main()
