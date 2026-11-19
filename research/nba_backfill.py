"""Backfill the NBA favourite-dip rule with Polymarket's price history, after testing it as a proxy for Kalshi.

The rule (research/dip_new_families.py): an NBA team is a heavy pre-game
favourite (90c or more); a bid rests at 80c from the scheduled start and any
fill is held to settlement; a fill counts only when a trade prints strictly
below the bid. Kalshi's NBA history starts in April 2025 and holds 22 such
fills. Polymarket listed NBA game winners a season earlier. This script asks
whether Polymarket's record reproduces Kalshi's on the games both venues
listed, and only then reads the rule off the games Kalshi never had.

Public, unauthenticated reads only. Nothing is traded. No Kalshi request is
made: the Kalshi side is data/processed/dip-new-families/rows-nba.json (one
row per game, built from cached one-minute bars) plus the pre-game prices in
data/processed/calibration-map/rows-nba.json for games that script screened out.

Sources (each response cached, trimmed, under data/raw/nba-backfill/)
  gamma-api.polymarket.com/markets      closed NBA "moneyline" markets, chunked by end date
  clob.polymarket.com/prices-history    one-minute history of a token's MIDPOINT (not a trade, bid or ask)
  data-api.polymarket.com/v2/trades     trade tape, newest first, cursor paging, no time filter
  site.api.espn.com scoreboard          scheduled tip-off, season type and final score

Definitions
* Game: a Polymarket slug nba-<away>-<home>-<date> matched to ESPN's game of
  the same two teams on that US date (one day either side allowed). The
  scheduled start is ESPN's; the result is Polymarket's settlement, which must
  agree with ESPN's winner.
* Pre-game price: the median Polymarket midpoint between 195 and 180 minutes
  before the scheduled start (where Kalshi's warm-up sits: its median end is
  177 minutes before the start), at least 10 points, range at most 3c. The
  favourite is the side above 50c.
* Proxy fills of a bid at L, from the scheduled start to five hours later:
    mid0 / mid1 / mid2   a one-minute midpoint strictly below L, L-1, L-2
    tape                 a trade strictly below L in favourite terms (a trade
                         in the other token at q counts as 100-q)
    tape_fav             the same using trades in the favourite's token only
    tape100              at least 100 shares printed strictly below L
* A fill pays its limit; Kalshi's NBA maker fee 1.75*p*(1-p) cents is charged
  (0.28c at 80c) so that numbers compare with dip_new_families.py.
* Kalshi era: games from 15 April 2025 (Kalshi's first NBA listing). Earlier
  games are the only ones that add independent evidence.

Commands
    listing   Polymarket markets and ESPN slates
    history   one-minute midpoints for every matched game
    build     one row per game (no network)
    tape      trade tapes for games whose pre-game price is at least --min-pre
    liquidity dollars traded before and during the game, and trade-to-midpoint distance, on a sample
    validate  proxy against Kalshi on the overlap (no network)
    report    the rule on Polymarket by season (no network)

    .venv/bin/python research/nba_backfill.py listing
    .venv/bin/python research/nba_backfill.py history
    .venv/bin/python research/nba_backfill.py build
    .venv/bin/python research/nba_backfill.py tape --min-pre 79
    .venv/bin/python research/nba_backfill.py build
    .venv/bin/python research/nba_backfill.py validate
    .venv/bin/python research/nba_backfill.py report
"""
import argparse
import bisect
from collections import Counter, defaultdict
from datetime import date, datetime, timedelta, timezone
import hashlib
import json
import math
from pathlib import Path
from statistics import mean, median
import sys
import time

import httpx

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data/raw/nba-backfill"
OUT = ROOT / "data/processed/nba-backfill"
KALSHI_ROWS = ROOT / "data/processed/dip-new-families/rows-nba.json"
KALSHI_SCREENED = ROOT / "data/processed/calibration-map/rows-nba.json"
GAMMA, CLOB, DATA = "https://gamma-api.polymarket.com", "https://clob.polymarket.com", "https://data-api.polymarket.com"
ESPN = "https://site.api.espn.com/apis/site/v2/sports/basketball/nba/scoreboard"
HEADERS = {"User-Agent": "Mozilla/5.0 (research script)"}
UTC = timezone.utc
FIRST, TODAY = date(2024, 4, 1), date(2026, 11, 18)
KALSHI_FIRST = date(2025, 4, 15)
PACE = 0.35
LEVELS = (85, 80, 75, 70)
MONTHS = {m: i+1 for i, m in enumerate("JAN FEB MAR APR MAY JUN JUL AUG SEP OCT NOV DEC".split())}
# Both venues' and ESPN's team codes mapped to one spelling.
ALIAS = {"no": "nop", "gs": "gsw", "ny": "nyk", "sa": "sas", "utah": "uta", "wsh": "was", "pho": "phx", "bro": "bkn", "nor": "nop"}
last_request = [0.0]
counts = Counter()


def save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(".tmp")
    temp.write_text(json.dumps(value))
    temp.replace(path)


def read(client, base, path, params, trim, name=None, keep=True):
    """Cached read, else one paced public read. `trim` shrinks the response before it is stored."""
    key = name or hashlib.sha256(json.dumps([base, path, params], sort_keys=True).encode()).hexdigest()
    cache = RAW / (path.strip("/").replace("/", "_") or "espn") / (key + ".json")
    if cache.exists():
        counts["cache"] += 1
        return json.loads(cache.read_text())
    for attempt in range(6):
        wait = last_request[0]+PACE-time.time()
        if wait > 0:
            time.sleep(wait)
        last_request[0] = time.time()
        counts["network"] += 1
        try:
            resp = client.get(base+path, params=params)
        except httpx.RequestError:
            time.sleep(2**attempt)
            continue
        if resp.status_code == 429 or resp.status_code >= 500:
            counts["throttled"] += 1
            time.sleep(5*2**attempt)
            continue
        if resp.status_code != 200:
            raise RuntimeError(f"{path} returned HTTP {resp.status_code}: {resp.text[:200]}")
        data = trim(resp.json())
        if keep:
            save(cache, data)
        return data
    raise RuntimeError(f"Read failed after retries: {path}")


def ts(text):
    return int(datetime.fromisoformat(text.replace(" ", "T").replace("+00", "+00:00").replace("Z", "+00:00").replace("+00:00:00", "+00:00")).timestamp())


def code(c):
    c = c.lower()
    return ALIAS.get(c, c)


def season_of(day):
    return f"{day.year}-{str(day.year+1)[2:]}" if day.month >= 9 else f"{day.year-1}-{str(day.year)[2:]}"


def wilson(wins, n, z=1.96):
    if not n:
        return None
    p = wins/n
    centre, half = (p+z*z/(2*n))/(1+z*z/n), z*math.sqrt(p*(1-p)/n+z*z/(4*n*n))/(1+z*z/n)
    return centre-half, centre+half


# ---------------------------------------------------------------- downloads

KEEP = ("id", "slug", "question", "outcomes", "outcomePrices", "clobTokenIds", "volume", "gameStartTime", "closedTime", "endDate",
        "conditionId", "sportsMarketType", "createdAt")


def poly_markets(client):
    """Every closed NBA moneyline market by end date (which sits anywhere from the game day to a week after it)."""
    out, day = {}, FIRST
    while day <= TODAY+timedelta(days=9):
        nxt = day+timedelta(days=3)
        offset = 0
        while True:
            page = read(client, GAMMA, "/markets", dict(
                tag_id=745, sports_market_types="moneyline", closed="true", limit=100, offset=offset, order="endDate", ascending="true",
                end_date_min=f"{day}T00:00:00Z", end_date_max=f"{nxt}T00:00:00Z"),
                lambda j: [{k: m.get(k) for k in KEEP} for m in j], keep=nxt < TODAY-timedelta(days=9))
            for m in page:
                out[m["id"]] = m
            if len(page) < 100:
                break
            offset += 100
        day = nxt
    return list(out.values())


def espn_days(client):
    """ESPN's slate for every day of the four seasons: {(us date, team set): game}."""
    def trim(j):
        rows = []
        for e in j.get("events", []):
            comp = e["competitions"][0]
            teams = {c["homeAway"]: c["team"].get("abbreviation", "") for c in comp["competitors"]}
            rows.append(dict(id=e["id"], start=ts(e["date"]), status=e["status"]["type"]["name"], kind=e.get("season", {}).get("type"),
                             teams=teams, score={c["homeAway"]: c.get("score") for c in comp["competitors"]},
                             winner=next((c["homeAway"] for c in comp["competitors"] if c.get("winner")), None),
                             note=" ".join(n.get("headline", "") for n in comp.get("notes", []))))
        return rows
    games, day = {}, FIRST
    while day <= TODAY:
        if day.month in (7, 8, 9):
            day += timedelta(days=1)
            continue
        rows = read(client, ESPN, "", dict(dates=f"{day:%Y%m%d}", limit=100), trim, name=f"day-{day:%Y%m%d}", keep=day < TODAY-timedelta(days=2))
        for g in rows:
            local = datetime.fromtimestamp(g["start"]-5*3600, UTC).date()
            games[(local, frozenset(code(c) for c in g["teams"].values()))] = g
        day += timedelta(days=1)
    return games


def matched(client):
    """Polymarket game markets joined to ESPN: list of games, plus counts of what was dropped."""
    markets, slate, drop, games = poly_markets(client), espn_days(client), Counter(), []
    for m in markets:
        parts = (m.get("slug") or "").split("-")
        if parts[6:] == ["moneyline"]:                # a few weeks of March 2025 carry this suffix
            parts = parts[:6]
        if not (len(parts) == 6 and parts[0] == "nba" and parts[3].isdigit()):
            drop["not a team-team-date slug"] += 1
            continue
        try:
            day = date(int(parts[3]), int(parts[4]), int(parts[5]))
            outcomes, prices = json.loads(m["outcomes"]), [float(p) for p in json.loads(m["outcomePrices"])]
            tokens = json.loads(m["clobTokenIds"])
        except (TypeError, ValueError):
            drop["unreadable"] += 1
            continue
        pair = frozenset(code(c) for c in parts[1:3])
        hit = next((d for d in (day, day-timedelta(days=1), day+timedelta(days=1)) if (d, pair) in slate), None)
        g = slate.get((hit, pair))
        if g is None:
            drop["no ESPN game"] += 1
            continue
        if g["status"] != "STATUS_FINAL" or sorted(prices) != [0.0, 1.0]:
            drop["not final or not settled 0/1"] += 1
            continue
        home = code(g["teams"]["home"])
        side_home = [code(c) for c in parts[1:3]].index(home)          # index of the home team in the slug
        if g["winner"] and prices[side_home] != (1.0 if g["winner"] == "home" else 0.0):
            drop["settled against ESPN's winner"] += 1
            continue
        games.append(dict(slug=m["slug"], espn=g["id"], exact=hit == day, day=str(day), season=season_of(day), phase={1: "preseason", 2: "regular", 3: "playoffs", 5: "playoffs"}.get(g["kind"], "other"),
                          note=g["note"], start=g["start"], teams=[code(c) for c in parts[1:3]], outcomes=outcomes, won0=prices[0], token0=tokens[0],
                          token1=tokens[1], condition=m["conditionId"], volume=float(m["volume"] or 0),
                          poly_start=ts(m["gameStartTime"]) if m.get("gameStartTime") else None, closed=ts(m["closedTime"]) if m.get("closedTime") else None))
    best = {}
    for g in games:                                   # a relisted or misdated market: keep the one on ESPN's date, then the larger
        k = g["espn"]
        if k not in best or (g["exact"], g["volume"]) > (best[k]["exact"], best[k]["volume"]):
            best[k] = g
    drop["duplicate listing"] = len(games)-len(best)
    return sorted(best.values(), key=lambda g: (g["start"], g["slug"])), drop, markets


def history(client, g):
    """One-minute midpoints of the first outcome, in cents: [(t, mid)] from four hours before the start to five after."""
    data = read(client, CLOB, "/prices-history", dict(market=g["token0"], startTs=g["start"]-4*3600, endTs=g["start"]+5*3600, fidelity=1),
                lambda j: [[h["t"], round(100*h["p"], 2)] for h in j.get("history", [])], name=g["slug"])
    return sorted(map(tuple, data))


def tape(client, g, max_pages=80):
    """Trades from ten minutes before the start onward: [t, price of outcome 0's token or of outcome 1's, size, outcome index]."""
    cache = RAW / "tape" / (g["slug"]+".json")
    if cache.exists():
        return json.loads(cache.read_text())
    if client is None:
        return None
    rows, cursor, pages, complete = [], None, 0, False
    while pages < max_pages:
        page = read(client, DATA, "/v2/trades", dict(condition=g["condition"], limit=1000, **({"cursor": cursor} if cursor else {})),
                    lambda j: dict(data=[[t["timestamp"], t["price"], t["size"], t["outcome_index"], t["side"][0]] for t in j.get("data", [])],
                                   next=j.get("pagination", {}).get("next_cursor"), more=j.get("pagination", {}).get("has_more")), keep=False)
        pages += 1
        rows += page["data"]
        cursor = page["next"]
        if not page["data"] or not page["more"] or not cursor or page["data"][-1][0] < g["start"]-600:
            complete = True
            break
    out = dict(complete=complete, pages=pages, trades=sorted(r for r in rows if r[0] >= g["start"]-600))
    save(cache, out)
    return out


def liquidity(client, per_season=30):
    """Full tapes back to a day before the start for a sample of heavy favourites: dollars traded before and during the game,
    and how far trades in the last three hours before the start sat from the midpoint (half the spread, roughly)."""
    games = {g["slug"]: g for g in json.loads((OUT/"games.json").read_text())}
    rows = [r for r in json.loads((OUT/"rows.json").read_text()) if r.get("pre", 0) >= 88 and r["phase"] != "preseason"]
    by, out = defaultdict(list), {}
    for r in rows:
        by[r["season"]].append(r)
    for season, sub in sorted(by.items()):
        step = max(1, len(sub)//per_season)
        cells = []
        for r in sub[::step][:per_season]:
            g = games[r["slug"]]
            cache = RAW / "tape-full" / (g["slug"]+".json")
            if cache.exists():
                cells.append(json.loads(cache.read_text()))
                continue
            trades, cursor, pages, complete = [], None, 0, False
            while pages < 25:
                page = read(client, DATA, "/v2/trades", dict(condition=g["condition"], limit=1000, **({"cursor": cursor} if cursor else {})),
                            lambda j: dict(data=[[t["timestamp"], t["price"], t["size"], t["outcome_index"]] for t in j.get("data", [])],
                                           next=j.get("pagination", {}).get("next_cursor"), more=j.get("pagination", {}).get("has_more")), keep=False)
                pages += 1
                trades += page["data"]
                cursor = page["next"]
                if not page["data"] or not page["more"] or not cursor or page["data"][-1][0] < g["start"]-24*3600:
                    complete = True
                    break
            hist = sorted(map(tuple, json.loads((RAW/"prices-history"/(g["slug"]+".json")).read_text())))
            times, off = [h[0] for h in hist], []
            for t, price, size, idx in trades:
                if g["start"]-3*3600 <= t < g["start"]:
                    i = bisect.bisect_right(times, t)-1
                    if i >= 0 and t-times[i] <= 90:
                        off.append(abs((100*price if idx == 0 else 100-100*price)-hist[i][1]))
            cell = dict(slug=g["slug"], season=season, complete=complete,
                        usd_day_before=sum(p*s for t, p, s, _ in trades if g["start"]-24*3600 <= t < g["start"]),
                        usd_in_play=sum(p*s for t, p, s, _ in trades if g["start"] <= t < g["start"]+3*3600),
                        trades_day_before=sum(g["start"]-24*3600 <= t < g["start"] for t, *_ in trades),
                        trades_in_play=sum(g["start"] <= t < g["start"]+3*3600 for t, *_ in trades),
                        trade_to_mid=median(off) if len(off) >= 5 else None, trade_to_mid_p90=sorted(off)[int(.9*len(off))] if len(off) >= 5 else None)
            save(cache, cell)
            cells.append(cell)
        out[season] = cells
    save(OUT/"liquidity.json", out)
    for season, cells in out.items():
        def med(key):
            v = sorted(c[key] for c in cells if c[key] is not None)
            return v[len(v)//2] if v else float("nan")
        print(f"  {season}: sampled {len(cells)} favourites at 88c+ (tape cut short in {sum(not c['complete'] for c in cells)}); median dollars traded in the day before the start "
              f"${med('usd_day_before'):,.0f} ({med('trades_day_before')} trades), in play ${med('usd_in_play'):,.0f} ({med('trades_in_play')} trades); "
              f"median distance of a pre-game trade from the midpoint {med('trade_to_mid'):.2f}c (p90 within a game {med('trade_to_mid_p90'):.2f}c)")


# ---------------------------------------------------------------- rows

def first_below(points, level, t0):
    for t, p in points:
        if t >= t0 and p < level:
            return t
    return None


def build(args):
    games = json.loads((OUT/"games.json").read_text())
    rows, cover = [], Counter()
    for g in games:
        cache = RAW / "prices-history" / (g["slug"]+".json")
        if not cache.exists():
            cover["no history downloaded"] += 1
            continue
        raw = sorted(map(tuple, json.loads(cache.read_text())))
        cover["games"] += 1
        warm = [p for t, p in raw if g["start"]-195*60 <= t < g["start"]-180*60]
        late = [p for t, p in raw if g["start"]-20*60 <= t < g["start"]-5*60]
        play = [(t, p) for t, p in raw if t >= g["start"]]
        row = dict(slug=g["slug"], day=g["day"], season=g["season"], phase=g["phase"], start=g["start"], volume=g["volume"], teams=g["teams"],
                   kalshi_era=g["day"] >= str(KALSHI_FIRST), points_play=len(play), points_warm=len(warm),
                   max_gap=max((b[0]-a[0] for a, b in zip(play, play[1:])), default=None))
        if len(warm) < 10 or len(play) < 60:
            row["why"] = "thin history"
        elif max(warm)-min(warm) > 3:
            row["why"] = "unstable pre-game"
        if "why" in row:
            cover[row["why"]] += 1
            rows.append(row)
            continue
        fav0 = median(warm) > 50
        sign = (lambda p: p) if fav0 else (lambda p: 100-p)
        pts = [(t, round(sign(p), 2)) for t, p in play]
        pre = round(sign(median(warm)), 2)
        before = [sign(p) for t, p in raw if t < g["start"]]
        row.update(pre=pre, pre_late=round(sign(median(late)), 2) if len(late) >= 5 else None, at_start=round(before[-1], 2), fav=g["teams"][0 if fav0 else 1],
                   payout=g["won0"] if fav0 else 1-g["won0"], min_mid=min(p for _, p in pts), distinct=len({p for _, p in pts}),
                   # share of in-play minutes in which the midpoint did not change: a stale midpoint would show here
                   flat=round(mean(a[1] == b[1] for a, b in zip(pts, pts[1:])), 3), levels={})
        trades = None
        t = tape(None, g)
        if t is not None:
            fav_idx = 0 if fav0 else 1
            trades = [(r[0], round(100*r[1] if r[3] == fav_idx else 100-100*r[1], 2), r[2], r[3] == fav_idx) for r in t["trades"] if r[0] >= g["start"]]
            row.update(tape=True, tape_complete=t["complete"], trades_play=len(trades), min_trade=min((x[1] for x in trades), default=None))
        labels = {str(v): v for v in LEVELS}
        labels["pre-10"] = math.floor(pre)-10
        for label, limit in labels.items():
            if limit > pre-5:
                continue
            cell = dict(limit=limit, mid0=first_below(pts, limit, g["start"]), mid1=first_below(pts, limit-1, g["start"]),
                        mid2=first_below(pts, limit-2, g["start"]))
            if trades is not None:
                below = [x for x in trades if x[1] < limit]
                shares = sum(x[2] for x in below)
                cum, t100 = 0, None
                for x in below:
                    cum += x[2]
                    if cum >= 100:
                        t100 = x[0]
                        break
                cell.update(tape=below[0][0] if below else None, tape_fav=next((x[0] for x in below if x[3]), None), tape100=t100,
                            shares_below=round(shares), trades_below=len(below))
                if below:      # where the midpoint stood a minute after the first print below the bid
                    i = bisect.bisect_right([p[0] for p in pts], below[0][0]+60)
                    if i < len(pts):
                        cell["mid_after"] = round(pts[i][1]-limit, 2)
            row["levels"][label] = cell
        rows.append(row)
    save(OUT/"rows.json", rows)
    cover["with a clean pre-game price"] = sum("pre" in r for r in rows)
    cover["with a tape"] = sum(bool(r.get("tape")) for r in rows)
    print(dict(cover))


# ---------------------------------------------------------------- Kalshi side

def kalshi_games():
    """{(day, team set): row} from the dip study, plus pre-game prices of the games it screened out as having no favourite."""
    out = {}
    for r in json.loads(KALSHI_ROWS.read_text()):
        stamp = r["event"].split("-")[1]
        teams = frozenset(code(m) for m in (stamp[7:10], stamp[10:13]))
        out[(str(date(2000+int(stamp[:2]), MONTHS[stamp[2:5]], int(stamp[5:7]))), teams)] = r
    if KALSHI_SCREENED.exists():
        for c in json.loads(KALSHI_SCREENED.read_text()):
            stamp = c["event"].split("-")[1]
            k = (str(date(2000+int(stamp[:2]), MONTHS[stamp[2:5]], int(stamp[5:7]))), frozenset(code(m) for m in (stamp[7:10], stamp[10:13])))
            if k not in out and len(stamp) == 13:
                pre = c["start"] if c["start"] > 50 else 100-c["start"]
                out[k] = dict(event=c["event"], pre=pre, screened=True, levels={}, payout=None)
    return out


def load():
    rows = [r for r in json.loads((OUT/"rows.json").read_text()) if "pre" in r]
    kal = kalshi_games()
    for r in rows:
        k = kal.get((r["day"], frozenset(r["teams"])))
        if k:
            r["k"] = k
    return rows


def net(payout, limit):
    return 100*payout-limit-1.75*(limit/100)*(1-limit/100)


def filled(r, level, how):
    c = r["levels"].get(level)
    return bool(c) and c.get(how) is not None


def line(rows, level, how):
    """orders, fills, wins, net cents per fill, return on the stake, Wilson interval on the net."""
    orders = [r for r in rows if level in r["levels"] and (not how.startswith("tape") or r.get("tape"))]
    fills = [r for r in orders if filled(r, level, how)]
    if not fills:
        return dict(orders=len(orders), fills=0)
    limit = mean(r["levels"][level]["limit"] for r in fills)
    wins = sum(r["payout"] for r in fills)
    nets = [net(r["payout"], r["levels"][level]["limit"]) for r in fills]
    w = wilson(wins, len(fills))
    fee = 1.75*(limit/100)*(1-limit/100)
    return dict(orders=len(orders), fills=len(fills), wins=int(wins), win=100*wins/len(fills), needed=limit, net=mean(nets), ret=100*mean(nets)/limit,
                lo=100*w[0]-limit-fee, hi=100*w[1]-limit-fee)


def show(c):
    if not c["fills"]:
        return f'orders {c["orders"]:4} fills   0'
    return (f'orders {c["orders"]:4} fills {c["fills"]:3} ({c["fills"]/c["orders"]:4.0%}) won {c["wins"]:3} = {c["win"]:5.1f}% need {c["needed"]:4.1f}% '
            f'net {c["net"]:+6.1f}c [{c["lo"]:+6.1f},{c["hi"]:+5.1f}] return {c["ret"]:+6.1f}%')


def gap(rows, key="pre"):
    if not rows:
        return "n 0"
    price, wins = mean(r[key] for r in rows), sum(r["payout"] for r in rows)
    w = wilson(wins, len(rows))
    return (f'n {len(rows):4} price {price:5.1f}c won {100*wins/len(rows):5.1f}% gap {100*wins/len(rows)-price:+5.1f}c '
            f'[{100*w[0]-price:+5.1f},{100*w[1]-price:+5.1f}] losses {int(len(rows)-wins)}')


def halves(rows):
    rows = sorted(rows, key=lambda r: (r["start"], r["slug"]))
    return rows[:len(rows)//2], rows[len(rows)//2:]


class Say:
    def __init__(self, name):
        self.lines, self.name = [], name

    def __call__(self, text=""):
        print(text)
        self.lines.append(text)

    def close(self):
        (OUT/self.name).write_text("\n".join(self.lines)+"\n")


# ---------------------------------------------------------------- coverage

def coverage(games, drop, markets, say):
    say("POLYMARKET NBA GAME-WINNER MARKETS BY SEASON (closed 'moneyline' markets under the NBA tag)")
    say(f"  listed {len(markets)}; dropped {dict(drop)}")
    by = defaultdict(list)
    for g in games:
        by[(g["season"], g["phase"])].append(g)
    for (season, phase), gs in sorted(by.items()):
        vols = sorted(g["volume"] for g in gs)
        say(f"  {season} {phase:9} games {len(gs):5} first {gs[0]['day']} last {gs[-1]['day']} volume median ${vols[len(vols)//2]:>11,.0f} p10 ${vols[len(vols)//10]:>10,.0f}")
    say(f"  earliest team-team-date market: {games[0]['slug']}. Nothing under the NBA tag is a game market before the April 2024 play-in.")


# ---------------------------------------------------------------- validation

HOWS = ("mid0", "mid1", "mid2", "tape", "tape_fav", "tape100")


def confusion(pairs):
    """pairs of (kalshi filled, proxy filled) -> both, kalshi only, proxy only, neither."""
    c = Counter(pairs)
    return c[(True, True)], c[(True, False)], c[(False, True)], c[(False, False)]


def validate(args):
    say = Say("validation.txt")
    rows = load()
    both = [r for r in rows if "k" in r]
    full = [r for r in both if not r["k"].get("screened")]
    say("PROXY VALIDATION ON GAMES LISTED BY BOTH VENUES")
    say(f"  Polymarket games with a clean pre-game price: {len(rows)}; of those in the Kalshi era {sum(r['kalshi_era'] for r in rows)}; "
        f"matched to a Kalshi game {len(both)} ({len(full)} with Kalshi bars, {len(both)-len(full)} screened out there as below 68c)")
    say(f"  by phase: {dict(Counter(r['phase'] for r in full))}; first {min(r['day'] for r in full)} last {max(r['day'] for r in full)}")
    bad = [r for r in full if min(r["pre"], r["k"]["pre"]) >= 55 and (r["fav"].upper() != r["k"]["fav"] or r["payout"] != r["k"]["payout"])]
    say(f"  favourite or result disagrees between venues (favourites at 55c+ on both): {len(bad)} {[r['slug'] for r in bad][:6]}")
    full = [r for r in full if r not in bad]

    say("\n1. PRE-GAME PRICE (Polymarket midpoint 195-180 minutes before the start, against Kalshi's warm-up midpoint)")
    for label, lo, hi in (("all favourites", 50, 101), ("Kalshi 80c+", 80, 101), ("Kalshi 88c+", 88, 101)):
        d = [r["pre"]-r["k"]["pre"] for r in full if lo <= r["k"]["pre"] < hi]
        a = sorted(abs(x) for x in d)
        say(f"  {label:15} n {len(d):4} mean (Polymarket-Kalshi) {mean(d):+.2f}c median abs {a[len(a)//2]:.2f}c p90 abs {a[int(.9*len(a))]:.2f}c within 1c {mean(x <= 1 for x in a):.0%}")
    d = [(r["start"]-r["k"]["est"])/60 for r in full]
    say(f"  scheduled start: ESPN equals Kalshi's estimate in {mean(x == 0 for x in d):.1%} of games, within 15 minutes in {mean(abs(x) <= 15 for x in d):.1%}")
    reg = [r for r in full if r["phase"] == "regular"]
    for cut in (90, 80):
        k, p = {r["slug"] for r in both if r["k"]["pre"] >= cut and r["phase"] == "regular"}, {r["slug"] for r in both if r["pre"] >= cut and r["phase"] == "regular"}
        say(f"  regular-season favourites at {cut}c+: Kalshi {len(k)}, Polymarket {len(p)}, both {len(k & p)}, Kalshi only {len(k-p)}, Polymarket only {len(p-k)}")
        if cut == 90:
            for name, only, key in (("Kalshi only", k-p, None), ("Polymarket only", p-k, None)):
                say(f"    {name}: " + ", ".join(f'{r["slug"][4:]} K{r["k"]["pre"]:.1f}/P{r["pre"]:.1f}' for r in both if r["slug"] in only))
    near = Counter((r["k"]["pre"], r["pre"]) for r in reg if r["k"]["pre"] in (89.5, 90.5))
    say(f"  around the cut: Kalshi 90.5c is Polymarket {dict((p, n) for (k, p), n in sorted(near.items()) if k == 90.5)}; "
        f"Kalshi 89.5c is Polymarket {dict((p, n) for (k, p), n in sorted(near.items()) if k == 89.5)}")
    for cut in (89.5, 89):
        k, p = {r["slug"] for r in both if r["k"]["pre"] >= 90 and r["phase"] == "regular"}, {r["slug"] for r in both if r["pre"] >= cut and r["phase"] == "regular"}
        say(f"  if Polymarket's cut were {cut}c: both {len(k & p)}, Kalshi only {len(k-p)}, Polymarket only {len(p-k)}")
    k90 = [r for r in reg if r["k"]["pre"] >= 90]
    p90 = [r for r in rows if r["kalshi_era"] and r["phase"] == "regular" and r["pre"] >= 90]
    say(f"  pre-game gap on the same games, Kalshi's 90c+ set:      Kalshi price   {gap([dict(r, pre=r['k']['pre']) for r in k90])}")
    say(f"                                                         Polymarket price {gap(k90)}")
    say(f"  Polymarket's own 90c+ set in the Kalshi era:            {gap(p90)}")
    for lo, hi in ((80, 90), (70, 80)):
        kb = [r for r in reg if lo <= r["k"]["pre"] < hi]
        say(f"  {lo}-{hi}c on Kalshi's set: Kalshi price {gap([dict(r, pre=r['k']['pre']) for r in kb])}")
        say(f"  {lo}-{hi}c Polymarket's own set (Kalshi era): {gap([r for r in rows if r['kalshi_era'] and r['phase'] == 'regular' and lo <= r['pre'] < hi])}")

    say("\n2. FILLS: does the proxy fill when Kalshi's bid filled? (same games, same bid; Kalshi fill = a print strictly below the bid)")
    say("   columns: both filled / Kalshi only (proxy MISSED a real fill) / proxy only (FALSE fill) / neither")
    sets = {
        "A. 80c bid, Kalshi's regular-season 90c+ favourites (the rule itself)": [(r, "80") for r in k90],
        "B. bids 85/80/75/70, Kalshi's 90c+ favourites": [(r, lv) for r in k90 for lv in ("85", "80", "75", "70")],
        "C. bid 10c under the pre-game price, favourites 80-90c on both venues": [(r, "pre-10") for r in reg if 80 <= r["k"]["pre"] < 90 and 80 <= r["pre"] < 90
                                                                              and r["levels"].get("pre-10", {}).get("limit") == r["k"]["levels"].get("pre-10", {}).get("limit")],
    }
    for name, items in sets.items():
        items = [(r, lv) for r, lv in items if lv in r["levels"] and lv in r["k"]["levels"]]
        say(f"  {name}: {len(items)} orders, Kalshi filled {sum(r['k']['levels'][lv]['strict'] for r, lv in items)}; with a tape {sum(bool(r.get('tape')) for r, _ in items)}")
        for how in HOWS:
            use = [(r, lv) for r, lv in items if not how.startswith("tape") or r.get("tape")]
            if not use:
                continue
            b, ko, po, n = confusion([(bool(r["k"]["levels"][lv]["strict"]), filled(r, lv, how)) for r, lv in use])
            lag = sorted((r["levels"][lv][how]-(r["k"]["levels"][lv]["ts"]-60))/60 for r, lv in use if r["k"]["levels"][lv]["strict"] and filled(r, lv, how))
            text = f"    {how:8} {b:3} / {ko:3} / {po:3} / {n:3}   found {b}/{b+ko} of Kalshi's fills; {po} of {b+po} proxy fills false"
            if lag:
                text += f" | proxy time minus start of Kalshi's fill minute: median {lag[len(lag)//2]:+.1f} min, p10 {lag[len(lag)//10]:+.1f}, p90 {lag[int(.9*len(lag))]:+.1f}, within 2 min {mean(abs(x) <= 2 for x in lag):.0%}"
            say(text)
    say("  the misses and false fills at the 80c bid, set A (lowest Kalshi print after the start / lowest Polymarket midpoint / lowest Polymarket trade):")
    for r in k90:
        if "80" not in r["levels"] or "80" not in r["k"]["levels"]:
            say(f"    {r['slug'][4:]} no 80c order on Polymarket's price (pre {r['pre']})")
            continue
        kf = bool(r["k"]["levels"]["80"]["strict"])
        flags = {h: filled(r, "80", h) for h in HOWS if not h.startswith("tape") or r.get("tape")}
        if any(v != kf for v in flags.values()):
            say(f"    {r['slug'][4:]:22} Kalshi {'FILL' if kf else 'none'} low {r['k']['min_after_start']} | mid low {r['min_mid']} trade low {r.get('min_trade')} | "
                + " ".join(f"{h}={'Y' if v else 'n'}" for h, v in flags.items()))

    say("\n3. THE RULE END TO END (regular season, Kalshi era): Kalshi's own record against Polymarket alone")
    kc = [r["k"] for r in k90]
    kf = [k for k in kc if k["levels"].get("80", {}).get("strict")]
    say(f"  Kalshi, matched games:       orders {len(kc)} fills {len(kf)} won {int(sum(k['payout'] for k in kf))} net {mean(net(k['payout'], 80) for k in kf):+.1f}c")
    for how in HOWS:
        c = line(p90, "80", how)
        fills = {r["slug"] for r in p90 if filled(r, "80", how) and (not how.startswith("tape") or r.get("tape"))}
        kfs = {r["slug"] for r in k90 if r["k"]["levels"].get("80", {}).get("strict")}
        say(f"  Polymarket alone, {how:8}  {show(c)} | same games as Kalshi's fills {len(fills & kfs)}, not a Kalshi fill {len(fills-kfs)}, Kalshi fills absent {len(kfs-fills)}")
    for level in ("85", "75", "70"):
        kf = [k for k in kc if k["levels"].get(level, {}).get("strict")]
        text = f"  bid {level}: Kalshi fills {len(kf)} won {int(sum(k['payout'] for k in kf))}"
        for how in ("mid0", "tape"):
            c = line(p90, level, how)
            text += f" | Polymarket {how} fills {c['fills']} won {c.get('wins', 0)}"
        say(text)
    say.close()


# ---------------------------------------------------------------- report

def report(args):
    say = Say("report.txt")
    rows, how = load(), args.how
    coverage(json.loads((OUT/"games.json").read_text()), json.loads((OUT/"dropped.json").read_text()), json.loads((OUT/"markets.json").read_text()), say)
    alls = json.loads((OUT/"rows.json").read_text())
    say(f"\nHISTORY QUALITY: games with history {len(alls)}; unusable {dict(Counter(r['why'] for r in alls if 'why' in r))}")
    for season in sorted({r["season"] for r in rows}):
        sub = [r for r in rows if r["season"] == season]
        gaps, flat, pts = [r["max_gap"] for r in sub], [r["flat"] for r in sub], [r["points_play"] for r in sub]
        fav = [r for r in sub if r["pre"] >= 88 and r.get("tape")]
        tr = sorted(r["trades_play"] for r in fav)
        say(f"  {season}: games {len(sub):5} in-play points median {sorted(pts)[len(pts)//2]} largest gap between points: median {sorted(gaps)[len(gaps)//2]}s, over 5 min in {mean(g > 300 for g in gaps):.1%} of games; "
            f"minutes with an unchanged midpoint {mean(flat):.0%}; in-play trades per 88c+ game median {tr[len(tr)//2] if tr else None} p10 {tr[len(tr)//10] if tr else None}; "
            f"incomplete tapes {sum(not r['tape_complete'] for r in fav)}")

    groups = [("BEFORE KALSHI: 2024 playoffs", lambda r: not r["kalshi_era"] and r["season"] == "2023-24"),
              ("BEFORE KALSHI: 2024-25 regular season", lambda r: not r["kalshi_era"] and r["season"] == "2024-25" and r["phase"] == "regular"),
              ("BEFORE KALSHI: all games", lambda r: not r["kalshi_era"]),
              ("KALSHI ERA: 2025-26 regular season", lambda r: r["kalshi_era"] and r["season"] == "2025-26" and r["phase"] == "regular"),
              ("KALSHI ERA: 2026-27 regular season so far", lambda r: r["kalshi_era"] and r["season"] == "2026-27" and r["phase"] == "regular"),
              ("KALSHI ERA: regular season", lambda r: r["kalshi_era"] and r["phase"] == "regular"),
              ("KALSHI ERA: playoffs (2025 and 2026)", lambda r: r["kalshi_era"] and r["phase"] == "playoffs"),
              ("KALSHI ERA: games Kalshi has no row for", lambda r: r["kalshi_era"] and "k" not in r and r["phase"] != "preseason"),
              ("POOLED: every regular-season game", lambda r: r["phase"] == "regular"),
              ("POOLED: every playoff game", lambda r: r["phase"] == "playoffs"),
              ("POOLED: everything except preseason", lambda r: r["phase"] != "preseason")]
    say(f"\nTHE RULE ON POLYMARKET (fill definition: {how}; maker fee of Kalshi's NBA series charged)")
    for name, pick in groups:
        sub = [r for r in rows if pick(r)]
        if not sub:
            continue
        say(f"\n{name}: {len(sub)} games, {min(r['day'] for r in sub)}..{max(r['day'] for r in sub)}")
        for label, lo, hi in (("90c+", 90, 101), ("80-90c", 80, 90)):
            band = [r for r in sub if lo <= r["pre"] < hi]
            a, b = halves(band)
            say(f"  pre-game {label}: {gap(band)}")
            if len(band) >= 20:
                say(f"      first half {gap(a)}")
                say(f"      second half {gap(b)}")
            for level in (("85", "80", "75", "70") if lo == 90 else ("pre-10",)):
                say(f"    bid {level:6} {show(line(band, level, how))}")
                if level in ("80", "pre-10") and len(band) >= 20:
                    ca, cb = line(a, level, how), line(b, level, how)
                    say(f"        halves: {show(ca)}")
                    say(f"                {show(cb)}")
            if lo == 90:
                for alt in HOWS:
                    if alt != how:
                        say(f"    bid 80 by {alt:8} {show(line(band, '80', alt))}")
    pre = [r for r in rows if not r["kalshi_era"] and r["pre"] >= 90 and r["phase"] != "preseason"]
    say("\nEVERY 80c PROXY FILL BEFORE KALSHI (90c+ favourites)")
    for r in pre:
        c = r["levels"].get("80")
        if c and c.get(how) is not None:
            say(f"  {r['slug']:24} {r['phase']:8} pre {r['pre']:5.1f} start {r['at_start']:5.1f} fill +{(c[how]-r['start'])/60:5.0f} min low mid {r['min_mid']:5.1f} low trade {r.get('min_trade')} "
                f"shares under 80c {c.get('shares_below')} mid a minute later {c.get('mid_after')} {'WON' if r['payout'] else 'LOST'} volume ${r['volume']:,.0f}")
    say("\nSENSITIVITY: pre-game price taken 20-5 minutes before the start instead (90c+ on that price, 80c bid)")
    for name, pick in (("before Kalshi", lambda r: not r["kalshi_era"]), ("Kalshi era regular season", lambda r: r["kalshi_era"] and r["phase"] == "regular")):
        band = [r for r in rows if pick(r) and r["phase"] != "preseason" and (r["pre_late"] or 0) >= 90]
        say(f"  {name:26} {gap(band, 'pre_late')} | {show(line(band, '80', how))}")
    say("\nSENSITIVITY: Polymarket cut at 89.5c instead of 90c (Kalshi's 90.5c is often Polymarket's 89.5c; chosen after seeing the overlap), 80c bid")
    for name, pick in (("before Kalshi", lambda r: not r["kalshi_era"]), ("Kalshi era regular season", lambda r: r["kalshi_era"] and r["phase"] == "regular")):
        band = [r for r in rows if pick(r) and r["phase"] != "preseason" and r["pre"] >= 89.5]
        say(f"  {name:26} {gap(band)} | {show(line(band, '80', how))}")
    say("\nTHE 90c CUT: 80c bid by pre-game band, regular season. Kalshi's own record / Polymarket before Kalshi / Polymarket in the Kalshi era")
    kal = [r for r in json.loads(KALSHI_ROWS.read_text()) if r["phase"] == "regular"]

    def tally(pays):
        if not pays:
            return "fills  0"
        w = wilson(sum(pays), len(pays))
        return f"fills {len(pays):3} won {int(sum(pays)):3} = {100*mean(pays):5.1f}% [{100*w[0]:4.1f},{100*w[1]:5.1f}] net {100*mean(pays)-80.28:+6.1f}c"
    for lo, hi in ((85, 88), (88, 90), (90, 92), (92, 101), (88, 101), (89.5, 101), (90, 101)):
        k = [r["payout"] for r in kal if lo <= r["pre"] < hi and r["levels"].get("80", {}).get("strict")]
        a = [r["payout"] for r in rows if not r["kalshi_era"] and r["phase"] == "regular" and lo <= r["pre"] < hi and filled(r, "80", how)]
        b = [r["payout"] for r in rows if r["kalshi_era"] and r["phase"] == "regular" and lo <= r["pre"] < hi and filled(r, "80", how)]
        say(f"  {lo:>4}-{hi:<3} Kalshi {tally(k)} | before {tally(a)} | era {tally(b)}")
    k = [r["payout"] for r in kal if r["pre"] >= 90 and r["levels"].get("80", {}).get("strict")]
    a = [r["payout"] for r in rows if not r["kalshi_era"] and r["phase"] == "regular" and r["pre"] >= 90 and filled(r, "80", how)]
    say(f"  independent fills at 90c+ (Kalshi's own plus Polymarket's 2024-25 regular season): {tally(k+a)}")
    say("\nSENSITIVITY: favourites 88-90c and 90-92c and 92c+ (80c bid), all non-preseason games")
    for lo, hi in ((88, 90), (90, 92), (92, 101)):
        for name, pick in (("before Kalshi", lambda r: not r["kalshi_era"]), ("Kalshi era", lambda r: r["kalshi_era"])):
            band = [r for r in rows if pick(r) and r["phase"] != "preseason" and lo <= r["pre"] < hi]
            say(f"  {lo}-{hi} {name:14} {gap(band)} | {show(line(band, '80', how))}")
    say.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("cmd", choices=("listing", "history", "build", "tape", "liquidity", "validate", "report"))
    parser.add_argument("--min-pre", type=float, default=79, help="tape: only games whose Polymarket pre-game price is at least this")
    parser.add_argument("--minutes", type=float, default=9, help="stop starting new downloads after this long; a rerun resumes from the cache")
    parser.add_argument("--how", default="tape", choices=HOWS, help="report: the proxy fill definition")
    args = parser.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    deadline = time.time()+60*args.minutes
    if args.cmd in ("build", "validate", "report"):
        return dict(build=build, validate=validate, report=report)[args.cmd](args)
    with httpx.Client(timeout=40, headers=HEADERS) as client:
        if args.cmd == "listing":
            games, drop, markets = matched(client)
            save(OUT/"games.json", games)
            save(OUT/"dropped.json", dict(drop))
            save(OUT/"markets.json", [dict(slug=m["slug"], volume=m["volume"], endDate=m["endDate"]) for m in markets])
            coverage(games, drop, markets, print)
        elif args.cmd == "liquidity":
            liquidity(client)
        elif args.cmd == "history":
            games = json.loads((OUT/"games.json").read_text())
            for n, g in enumerate(games):
                if time.time() > deadline:
                    print(f"out of time at {n} of {len(games)}")
                    break
                history(client, g)
        else:
            games = {g["slug"]: g for g in json.loads((OUT/"games.json").read_text())}
            want = [r for r in json.loads((OUT/"rows.json").read_text()) if r.get("pre", 0) >= args.min_pre and not r.get("tape")]
            want.sort(key=lambda r: -r["pre"])
            for n, r in enumerate(want):
                if time.time() > deadline:
                    print(f"out of time at {n} of {len(want)}")
                    break
                tape(client, games[r["slug"]])
        print("requests", dict(counts))


if __name__ == "__main__":
    main()
