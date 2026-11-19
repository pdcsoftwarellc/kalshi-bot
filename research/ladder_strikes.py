"""Are the near-certain strikes of pre-game SPREAD and TOTAL ladders underpriced, the way heavy game-winner favourites are?

Background: on game-winner markets, pre-game favourites priced 90-99c win a
little more often than their price (research/calibration_map.py,
research/favourite_basket.py) and a resting 80c bid on them during play earns
more (research/dip_new_families.py). Every game also lists ladders (spread,
total, team total, first-half spread and total): many strikes per game, whose
far strikes are near-certain contracts on the YES side ("total over 120.5" at
95c) or on the NO side ("wins by over 22.5" at 5c, so NO at 95c). Earlier work
(research/spreads_totals.py) only tested an in-game rule on them.

Definitions

* Game = the part of the event ticker after the series ("26NOV11NCATECU"),
  shared by the game-winner and ladder series of a family.
* Scheduled start = the start in the ticker where it carries a time, else the
  game's most common `expected_expiration_time` minus three hours (as in
  research/favourite_contexts.py and research/dip_new_families.py). The report
  prints settle minus start per family and how far quotes moved in the last
  half hour, as a check that the reference is before play. A family in which
  more than 10% of strikes moved over 5c in that half hour is left out of
  every table (the soccer leagues: their scheduled start is not expected
  expiration minus three hours here, so the reference fell during play; and
  women's college basketball, whose ladder books are empty).
* Reference quote = the standing YES bid and ask REF minutes (5) before the
  scheduled start: the close of the last one-minute bar ending at or before
  that minute. Bars exist only for minutes in which something changed, and the
  batch endpoint is asked to prepend the latest earlier bar, so this is the
  quote that was resting then. A second reference 30 minutes before the start
  is reported as a check.
* A row is a strike whose quote is two-sided and at most 4c wide and whose
  dearer side (YES if the midpoint is above 50c, else NO) has a midpoint of
  80-99c. Bands 80-90, 90-95, 95-99 are on that midpoint. Markets not settled
  at 0 or 1 are dropped.
* Gap = 100 x payout of the dearer side - its midpoint. Tradeable = buy at the
  ask (NO: 100 - YES bid) and pay the taker fee 0.07 x multiplier x p(1-p)
  (rounded up per 100-contract order), in cents per contract and as a
  percentage of the amount paid.
* Intervals: bootstrap over GAMES (all strikes of a game move together: one
  blowout loses several at once) and, separately, over calendar DAYS (US
  Eastern date of the start). Every interval is at least as wide as the
  prices imply (1.96 x sqrt(sum p(1-p))/n), so a cell that happened to have
  no losses is not treated as certain. Halves split each family's games at the median
  start.
* Relation to the favourite: the game-winner market is priced at the same
  reference. Tables are repeated for games with and without a 90c+ favourite.
* Dip: strikes whose dearer side is 90c+ at the reference. A bid rests on that
  side at 80c (and at 10c below the reference midpoint) from the scheduled
  start; it fills only when a one-minute bar after the start has a trade
  strictly below it (NO side: a YES trade strictly above 100 - bid); held to
  settlement. Shown with no fee and with the maker fee 0.0175 x p(1-p) where
  the series charges one. "Through" = contracts traded in the minutes whose
  trade range went through the bid: an upper bound on what a bid could get.
* Multiplicity: a family x ladder x side x band cell passes when its game
  clustered interval is above zero in the first half, its gap is positive in
  the second half and every neighbouring band of the same family, ladder and
  side has a positive gap. Chance: outcomes are redrawn at the quoted
  midpoints with one uniform draw per game and ladder (a strike wins when the
  draw falls below its price, so a ladder's strikes lose together exactly as
  one final margin or total makes them), and the passes are counted again.
  Ladders of one game are drawn independently, which leaves each cell's pass
  rate exact and slightly understates the spread of the total count.
  Cells are then shrunk together with the normal-normal model of
  calibration_map.py / edge_reality.py.

Network: public reads only, ONE request at a time, at least PACE (2.5 s)
after the previous response, 30 s or more after HTTP 429 (a live trader, a
recorder and a paper program share the limit). Candles come from the batch
endpoint GET /markets/candlesticks (at most 100 tickers and under 10,000
ticker-periods a request), cached compactly under data/raw/cost-research-v1
with the same file naming as research.api_cost.fetch. Slim listings live in
data/raw/ladder-strikes. No orders, no authentication.

    .venv/bin/python research/ladder_strikes.py list --minutes 15
    .venv/bin/python research/ladder_strikes.py pregame --minutes 9 [--families NBA ...] [--skip team_total ...]
    .venv/bin/python research/ladder_strikes.py build        (offline: rows.json, needed before ingame)
    .venv/bin/python research/ladder_strikes.py ingame --minutes 9 [--families NBA ...]
    .venv/bin/python research/ladder_strikes.py report       (offline)

Coverage of the first run (2026-11-18, about 75 minutes and 1,500 requests, one HTTP 429): NBA (spread, total,
first-half spread and total; team totals listed only), men's college basketball (all four ladders) and NFL
(all five; team totals and halves for 121 of 134 games) with in-game bars for NBA, men's college and NFL
spreads and totals; NHL pre-game only. Women's college basketball ladders have no book. EPL and La Liga were
downloaded but their reference fell during play; MLB, Champions League, Bundesliga and MLS are listed only;
college football was not reached.
"""
import argparse
from collections import Counter, defaultdict
from datetime import date, datetime, timezone
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
import config
from research.api_cost import ROOT, midnight, timestamp
from research.calibration_map import cache_file, scheduled, shrink

RAW = ROOT / "data/raw/ladder-strikes"
OUT = ROOT / "data/processed/ladder-strikes"
CUTOFF = midnight(date(2026, 9, 18))
RECENT = dict(status="settled", limit=1000, min_close_ts=CUTOFF, max_close_ts=midnight(date(2026, 11, 19)))
PACE = 2.5
REF, REF2, BEFORE = 5, 30, 40          # minutes before the scheduled start: reference, check reference, window asked for
EASTERN = ZoneInfo("America/New_York")
BANDS = ((80, 90), (90, 95), (95, 99))
RNG = np.random.default_rng(41)
DRAWS = 2000
MOVED_LIMIT = .10      # share of strikes moving more than 5c in the last half hour above which a family is left out

# family -> winner series and ladder series by type, in download priority order
FAMILIES = {
    "NBA": dict(winner="KXNBAGAME", ladders=dict(spread="KXNBASPREAD", total="KXNBATOTAL", team_total="KXNBATEAMTOTAL",
                                                 half_spread="KXNBA1HSPREAD", half_total="KXNBA1HTOTAL")),
    "CBB men": dict(winner="KXNCAAMBGAME", ladders=dict(spread="KXNCAAMBSPREAD", total="KXNCAAMBTOTAL",
                                                        half_spread="KXNCAAMB1HSPREAD", half_total="KXNCAAMB1HTOTAL")),
    "CBB women": dict(winner="KXNCAAWBGAME", ladders=dict(spread="KXNCAAWBSPREAD", total="KXNCAAWBTOTAL")),
    "NFL": dict(winner="KXNFLGAME", ladders=dict(spread="KXNFLSPREAD", total="KXNFLTOTAL", team_total="KXNFLTEAMTOTAL",
                                                 half_spread="KXNFL1HSPREAD", half_total="KXNFL1HTOTAL")),
    "College football": dict(winner="KXNCAAFGAME", ladders=dict(spread="KXNCAAFSPREAD", total="KXNCAAFTOTAL",
                                                                team_total="KXNCAAFTEAMTOTAL", half_spread="KXNCAAF1HSPREAD",
                                                                half_total="KXNCAAF1HTOTAL")),
    "NHL": dict(winner="KXNHLGAME", ladders=dict(spread="KXNHLSPREAD", total="KXNHLTOTAL")),
    "MLB": dict(winner="KXMLBGAME", ladders=dict(spread="KXMLBSPREAD", total="KXMLBTOTAL")),
    "EPL": dict(winner="KXEPLGAME", ladders=dict(spread="KXEPLSPREAD", total="KXEPLTOTAL")),
    "La Liga": dict(winner="KXLALIGAGAME", ladders=dict(spread="KXLALIGASPREAD", total="KXLALIGATOTAL")),
    "Champions League": dict(winner="KXUCLGAME", ladders=dict(spread="KXUCLSPREAD", total="KXUCLTOTAL")),
    "Bundesliga": dict(winner="KXBUNDESLIGAGAME", ladders=dict(spread="KXBUNDESLIGASPREAD", total="KXBUNDESLIGATOTAL")),
    "MLS": dict(winner="KXMLSGAME", ladders=dict(spread="KXMLSSPREAD", total="KXMLSTOTAL")),
}
log = Counter()
last_request = [0.0]
deadline = [float("inf")]


# ---------------------------------------------------------------- network

class OutOfTime(Exception):
    pass


def read(client, path, params, cache=True):
    """One paced public read. Parsed JSON, or None on a 4xx other than 429. With cache, stored like api_cost.fetch (compact)."""
    file = cache_file(path, params)
    if cache and file.exists():
        log["cache"] += 1
        return json.loads(file.read_text())
    for pause in (30, 60, 120, 240, None):
        if time.time() > deadline[0]:
            raise OutOfTime
        wait = last_request[0]+PACE-time.time()
        if wait > 0:
            time.sleep(wait)
        log["requests"] += 1
        try:
            resp = client.get(config.KALSHI_PROD_BASE+path, params=params, headers={"User-Agent": "research-client"})
        except httpx.RequestError:
            resp = None
        last_request[0] = time.time()
        if resp is not None and resp.status_code == 200:
            data = resp.json()
            if cache:
                temp = file.with_suffix(".tmp")
                temp.write_text(json.dumps(data))
                temp.replace(file)
            return data
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
    rows.append(dict(step=step, began=datetime.fromtimestamp(began, timezone.utc).isoformat(), seconds=round(time.time()-began), **log))
    dump(OUT/"download-log.json", rows)
    print(step, dict(log), f"{(time.time()-began)/60:.1f} min", flush=True)


# ---------------------------------------------------------------- listings

def ts(value):
    try:
        return int(timestamp(value))
    except (TypeError, ValueError, AttributeError):
        return None


def slim(m):
    try:
        value = float(m["settlement_value_dollars"])
    except (KeyError, TypeError, ValueError):
        value = None
    return dict(ticker=m["ticker"], event=m["event_ticker"], floor=m.get("floor_strike"), kind=m.get("strike_type"),
                sub=m.get("yes_sub_title"), value=value, open=ts(m.get("open_time")), close=ts(m.get("close_time")),
                settle=ts(m.get("settlement_ts")), expected=ts(m.get("expected_expiration_time")),
                volume=float(m.get("volume_fp") or 0))


def series_of(names=None):
    for name, spec in FAMILIES.items():
        if names and name not in names:
            continue
        yield name, "winner", spec["winner"]
        for kind, series in spec["ladders"].items():
            yield name, kind, series


def cmd_list(client, names):
    for name, kind, series in series_of(names):
        path = RAW/f"markets-{series}.json"
        have = load(path, None)
        if have and have["done"]:
            continue
        have = dict(fee=None, markets={}, done=False)
        data = read(client, f"/series/{series}", {}, cache=False)
        s = (data or {}).get("series") or {}
        have["fee"] = dict(fee_type=s.get("fee_type"), fee_multiplier=s.get("fee_multiplier"), title=s.get("title"))
        cursor, seen = "", set()
        while True:
            data = read(client, "/markets", dict(RECENT, series_ticker=series, **({"cursor": cursor} if cursor else {})), cache=False)
            for m in (data or {}).get("markets", []):
                have["markets"][m["ticker"]] = slim(m)
            cursor = (data or {}).get("cursor")
            if not cursor or cursor in seen or not data["markets"]:
                break
            seen.add(cursor)
        have["done"] = True
        dump(path, have)
        ms = have["markets"].values()
        print(f"{name:18} {kind:12} {series:20} {len(ms):6} markets {len({m['event'] for m in ms}):5} events fee {have['fee']}", flush=True)


def games_of(name):
    """{game key: dict(start, settle, markets={type: [slim market]})} from the listings on disk."""
    games = {}
    for _, kind, series in series_of([name]):
        for m in load(RAW/f"markets-{series}.json", dict(markets={}))["markets"].values():
            if m["value"] not in (0.0, 1.0) or not m["settle"]:
                continue
            key = m["event"].split("-", 1)[1]
            g = games.setdefault(key, dict(key=key, markets=defaultdict(list)))
            g["markets"][kind].append(m)
    out = {}
    for key, g in games.items():
        every = [m for ms in g["markets"].values() for m in ms]
        sched = scheduled(every[0]["event"])
        expected = Counter(m["expected"] for m in every if m["expected"]).most_common(1)
        start = sched or (expected[0][0]-3*3600 if expected else None)
        full = [m["settle"] for kind, ms in g["markets"].items() if not kind.startswith("half") for m in ms]
        settle = max(full or [m["settle"] for m in every])
        if not start or not 1800 <= settle-start <= 10*3600:
            log[name+" odd timing"] += 1
            continue
        g.update(start=start, settle=settle, timing="ticker" if sched else "expected")
        out[key] = g
    return out


# ---------------------------------------------------------------- candles

def chunks(jobs, span):
    """jobs: (anchor ts, [tickers], extra minutes). Tickers are packed in anchor order, at most 100 a request and under
    9,500 ticker-periods, where a request spans from the first anchor - span to the last anchor + extra."""
    flat = sorted((anchor, t, extra) for anchor, tickers, extra in jobs for t in tickers)
    out, cur, tail = [], [], 0
    for anchor, ticker, extra in flat:
        far = max(tail, anchor+60*extra)
        minutes = ((far-cur[0][0])//60 if cur else extra)+span+2
        if cur and (len(cur) >= 100 or (len(cur)+1)*minutes > 9500):
            out.append((cur, tail))
            cur, far = [], anchor+60*extra
        cur.append((anchor, [ticker]))
        tail = far
    return out+[(cur, tail)] if cur else out


def batch_params(group, tail, before):
    return dict(market_tickers=",".join(t for _, part in group for t in part), start_ts=group[0][0]-60*before, end_ts=tail,
                period_interval=1, include_latest_before_start="true")


def compact(bars):
    """[end ts, yes bid close, yes ask close, trade low, trade high, volume] per bar, cents."""
    def c(group, field):
        v = (group or {}).get(field+"_dollars")
        return None if v is None else round(100*float(v), 2)
    return [[b["end_period_ts"], c(b.get("yes_bid"), "close"), c(b.get("yes_ask"), "close"), c(b.get("price"), "low"),
             c(b.get("price"), "high"), float(b.get("volume_fp") or 0)] for b in sorted(bars or [], key=lambda b: b["end_period_ts"])]


def run_plan(client, label, plan_file, groups, before):
    """Download every chunk not yet done; the list of request parameters is the index used to rebuild."""
    plan = load(plan_file, [])
    done = {p["market_tickers"] for p in plan}
    try:
        for n, (group, tail) in enumerate(groups):
            params = batch_params(group, tail, before)
            if params["market_tickers"] in done:
                continue
            data = read(client, "/markets/candlesticks", params)
            if data is None:
                log["batch refused"] += 1
                continue
            plan.append(params)
            if len(plan) % 20 == 0:
                dump(plan_file, plan)
                print(f"  {label}: chunk {n+1} of {len(groups)}  {dict(log)}", flush=True)
    except OutOfTime:
        print(f"  {label}: out of time", flush=True)
        raise
    finally:
        dump(plan_file, plan)


def bars_from(plan_file):
    out = {}
    for params in load(plan_file, []):
        file = cache_file("/markets/candlesticks", params)
        if not file.exists():
            continue
        for x in json.loads(file.read_text()).get("markets", []):
            out[x["market_ticker"]] = compact(x.get("candlesticks"))
    return out


def cmd_pregame(client, names, skip=()):
    for name in names or FAMILIES:
        games = games_of(name)
        jobs = []
        for g in games.values():
            tickers = sorted(m["ticker"] for kind, ms in g["markets"].items() for m in ms
                             if m["open"] and m["open"] < g["start"]-60*REF and kind not in skip)
            if tickers:
                jobs.append((g["start"], tickers, 0))
        groups = chunks(jobs, BEFORE)
        print(f"{name}: {len(games)} games, {sum(len(j[1]) for j in jobs)} markets, {len(groups)} requests", flush=True)
        run_plan(client, name+" pregame", RAW/f"plan-pre-{slug(name)}.json", groups, BEFORE)


def cmd_ingame(client, names):
    rows = load(OUT/"rows.json", [])
    for name in names or FAMILIES:
        games = games_of(name)
        jobs = defaultdict(list)
        for r in rows:
            if r["family"] == name and 90 <= r["price"] < 99 and r["spread"] <= 4 and r["ladder"] != "winner":
                jobs[r["game"]].append(r["ticker"])
        jobs = [(games[k]["start"], sorted(v), (games[k]["settle"]-games[k]["start"])//60+3) for k, v in jobs.items() if k in games]
        groups = chunks(jobs, 0)
        print(f"{name}: {len(jobs)} games, {sum(len(j[1]) for j in jobs)} markets at 90c+, {len(groups)} requests", flush=True)
        run_plan(client, name+" ingame", RAW/f"plan-in-{slug(name)}.json", groups, 0)


def slug(name):
    return re.sub(r"[^a-z0-9]+", "-", name.lower()).strip("-")


# ---------------------------------------------------------------- rows

def quote_at(bars, when):
    """(bid, ask) standing at `when`: the last bar ending at or before it."""
    last = None
    for b in bars:
        if b[0] > when:
            break
        last = b
    if last is None or last[1] is None or last[2] is None or not 0 < last[1] <= last[2] < 100:
        return None
    return last[1], last[2]


def team_of(m):
    tail = re.fullmatch(r"([A-Z]*?)(\d+)", m["ticker"].rsplit("-", 1)[1])
    return tail.group(1) if tail else ""


def fee_of(series):
    f = load(RAW/f"markets-{series}.json", dict(fee={}))["fee"] or {}
    return f.get("fee_type") or "?", float(f.get("fee_multiplier") or 1)


def build():
    rows, cover = [], {}
    for name, spec in FAMILIES.items():
        pre, inn = bars_from(RAW/f"plan-pre-{slug(name)}.json"), bars_from(RAW/f"plan-in-{slug(name)}.json")
        if not pre:
            continue
        games = games_of(name)
        c = Counter(games=len(games))
        fees = {kind: fee_of(series) for _, kind, series in series_of([name])}
        for g in sorted(games.values(), key=lambda g: (g["start"], g["key"])):
            ref, ref2 = g["start"]-60*REF, g["start"]-60*REF2
            fav = None
            for m in g["markets"].get("winner", []):
                q = quote_at(pre.get(m["ticker"], []), ref)
                if q and q[1]-q[0] <= 4 and not m["ticker"].endswith(("TIE", "DRAW")):
                    mid = (q[0]+q[1])/2
                    if fav is None or mid > fav[0]:
                        fav = (mid, m["value"], team_of(m) or m["ticker"].rsplit("-", 1)[1])
            seen = False
            for kind, ms in g["markets"].items():
                for m in ms:
                    bars = pre.get(m["ticker"])
                    if bars is None:
                        c["not downloaded"] += 1
                        continue
                    c["markets "+kind] += 1
                    q = quote_at(bars, ref)
                    if q is None:
                        c["no two-sided quote"] += 1
                        continue
                    bid, ask = q
                    mid, yes = (bid+ask)/2, (bid+ask)/2 > 50
                    row = dict(family=name, ladder=kind, game=g["key"], ticker=m["ticker"], start=g["start"], settle=g["settle"],
                               day=str(datetime.fromtimestamp(g["start"], EASTERN).date()), floor=m["floor"], team=team_of(m),
                               side="YES" if yes else "NO", price=mid if yes else 100-mid, spread=ask-bid,
                               cost=ask if yes else 100-bid, payout=m["value"] if yes else 1-m["value"], volume=m["volume"],
                               pre_volume=sum(b[5] for b in bars if g["start"]-60*BEFORE < b[0] <= g["start"]),
                               fee_type=fees[kind][0], fee_mult=fees[kind][1], fav=fav[0] if fav else None,
                               fav_won=fav[1] if fav else None, fav_team=fav[2] if fav else None)
                    q2 = quote_at(bars, ref2)
                    if q2:
                        m2 = (q2[0]+q2[1])/2
                        row["price30"], row["spread30"] = (m2 if yes else 100-m2), q2[1]-q2[0]
                    game_bars = inn.get(m["ticker"])
                    if game_bars is not None:
                        after = [b for b in game_bars if b[0] > g["start"]+60 and b[5] > 0 and b[3] is not None]
                        # the dearer side's trade prices: (low, volume) per minute, on that side
                        row["path"] = [[b[0]-g["start"], b[3] if yes else 100-b[4], b[5]] for b in after]
                    rows.append(row)
                    seen = True
            c["games with a quote"] += seen
        cover[name] = dict(c)
        print(name, dict(c), flush=True)
    dump(OUT/"rows.json", rows)
    dump(OUT/"coverage.json", cover)
    return rows


# ---------------------------------------------------------------- statistics

def fee(r, price=None, maker=False):
    """Fee in cents per contract on a 100-contract order (rounded up to a cent per order)."""
    p = (r["cost"] if price is None else price)/100
    rate = .0175 if maker else .07
    if maker and "maker" not in r["fee_type"]:
        return 0.0
    return math.ceil(100*rate*r["fee_mult"]*p*(1-p)*100-1e-9)/100


def net(r):
    return 100*r["payout"]-r["cost"]-fee(r)


def boot(rows, value, unit, weight=None, draws=DRAWS):
    """95% interval of the mean of value(row) (or sum(value)/sum(weight)), resampling whole units."""
    by = defaultdict(lambda: [0.0, 0.0])
    for r in rows:
        c = by[r[unit]]
        c[0] += value(r)
        c[1] += weight(r) if weight else 1
    a = np.array(list(by.values()))
    if len(a) < 2:
        return None
    pick = RNG.integers(0, len(a), (draws, len(a)))
    stat = a[pick, 0].sum(1)/np.maximum(a[pick, 1].sum(1), 1e-9)
    lo, hi = np.percentile(stat, [2.5, 97.5])
    return float(lo), float(hi)


def cluster_se(rows, value, unit):
    by = defaultdict(float)
    m = mean(value(r) for r in rows)
    for r in rows:
        by[r[unit]] += value(r)-m
    return math.sqrt(sum(v*v for v in by.values()))/len(rows)


def gap(r):
    return 100*r["payout"]-r["price"]


def gap_se(rows):
    """Standard error of the mean gap: the larger of the game-clustered one and the one the prices imply
    (sum of p(1-p)), so a cell that happened to have no losses is not treated as certain."""
    implied = 100*math.sqrt(sum(r["price"]/100*(1-r["price"]/100) for r in rows))/len(rows)
    return max(cluster_se(rows, gap, "gkey"), implied)


def widen(pair, centre, se):
    lo, hi = pair if pair else (centre, centre)
    return min(lo, centre-1.96*se), max(hi, centre+1.96*se)


def cell(rows):
    if not rows:
        return dict(n=0)
    n = len(rows)
    paid = mean(r["cost"]+fee(r) for r in rows)
    g, se = mean(gap(r) for r in rows), gap_se(rows)
    implied = 100*math.sqrt(sum(r["price"]/100*(1-r["price"]/100) for r in rows))/n
    out = dict(n=n, games=len({(r["family"], r["game"]) for r in rows}), days=len({r["day"] for r in rows}),
               price=mean(r["price"] for r in rows), win=100*mean(r["payout"] for r in rows), gap=g,
               ci_game=widen(boot(rows, gap, "gkey"), g, implied), ci_day=widen(boot(rows, gap, "day"), g, implied), se=se,
               spread=mean(r["spread"] for r in rows), net=mean(net(r) for r in rows),
               pct=100*mean(net(r) for r in rows)/paid, losses=sum(r["payout"] < .5 for r in rows))
    out["net_ci"] = widen(boot(rows, net, "gkey"), out["net"], implied)
    out["pct_ci"] = [100*v/paid for v in out["net_ci"]]
    return out


def mark_halves(rows):
    """first_half: the game starts before its family's median game start (over games with any quoted ladder strike)."""
    cut = {}
    for fam in {r["family"] for r in rows}:
        starts = sorted({(r["start"], r["game"]) for r in rows if r["family"] == fam and r["ladder"] != "winner"})
        cut[fam] = starts[len(starts)//2] if starts else (0, "")
    for r in rows:
        r["first_half"] = (r["start"], r["game"]) < cut[r["family"]]


def halves(rows):
    return [r for r in rows if r["first_half"]], [r for r in rows if not r["first_half"]]


def in_band(rows, lo, hi):
    return [r for r in rows if lo <= r["price"] < hi]


def f(v, digits=1, sign=True):
    return "    ." if v is None else (f"{v:+.{digits}f}" if sign else f"{v:.{digits}f}")


def ci(pair, digits=1):
    return "[      .      ]" if not pair else f"[{pair[0]:+.{digits}f} to {pair[1]:+.{digits}f}]"


def line(label, rows, width=34):
    c = cell(rows)
    if not c["n"]:
        return f"{label:{width}}      0"
    a, b = (cell(h) for h in halves(rows))
    return (f"{label:{width}} {c['n']:6} {c['games']:5} {c['price']:6.1f} {c['win']:6.1f} {f(c['gap'], 2):>6} {ci(c['ci_game']):>16} "
            f"{ci(c['ci_day']):>16} {f(a.get('gap'), 1):>5}/{f(b.get('gap'), 1):<5} {c['spread']:4.1f} {f(c['net'], 2):>6} "
            f"{f(c['pct'], 2):>6}% {ci(c.get('pct_ci'), 2):>17} {c['losses']:5}")


HEAD = (f"{'':34} {'n':>6} {'games':>5} {'price':>6} {'win%':>6} {'gap c':>6} {'95% by game':>16} {'95% by day':>16} "
        f"{'halves':>11} {'sprd':>4} {'net c':>6} {'net %':>7} {'net % 95% (game)':>17} {'lost':>5}")


# ---------------------------------------------------------------- dip

def dip(r, level=None):
    """(bid, filled?, contracts through, minutes to fill, lowest trade of the fill minute) for a bid resting on the dearer side."""
    bid = 80 if level is None else level
    through = [(t, v, low) for t, low, v in r["path"] if low < bid-1e-9]
    return bid, bool(through), sum(x[1] for x in through), through[0][0]/60 if through else None, through[0][2] if through else None


def dip_cell(rows, level_of, maker=True):
    orders = [r for r in rows if "path" in r]
    fills = []
    for r in orders:
        bid, hit, size, when, low = dip(r, level_of(r))
        if hit:
            fills.append(dict(r, bid=bid, through=size, when=when, overshoot=bid-low, dnet=100*r["payout"]-bid-(fee(r, bid, True) if maker else 0)))
    if not fills:
        return dict(orders=len(orders), fills=0)
    value = lambda x: x["dnet"]
    paid = mean(x["bid"] for x in fills)
    return dict(orders=len(orders), fills=len(fills), games=len({x["gkey"] for x in fills}), bid=paid,
                win=100*mean(x["payout"] for x in fills), net=mean(map(value, fills)), ci=boot(fills, value, "gkey"),
                ci_day=boot(fills, value, "day"), per_order=sum(map(value, fills))/len(orders), pct=100*mean(map(value, fills))/paid,
                through=median(x["through"] for x in fills), through_q1=float(np.percentile([x["through"] for x in fills], 25)),
                share_100=mean(x["through"] >= 100 for x in fills), share_1000=mean(x["through"] >= 1000 for x in fills),
                minutes=median(x["when"] for x in fills), overshoot=mean(x["overshoot"] for x in fills),
                overshoot_median=median(x["overshoot"] for x in fills), rows=fills)


def dip_line(label, rows, level_of, width=34):
    c = dip_cell(rows, level_of)
    if not c["fills"]:
        return f"{label:{width}} {c['orders']:6} orders, no fills"
    a, b = (dip_cell(h, level_of) for h in halves(rows))
    return (f"{label:{width}} {c['orders']:6} {c['fills']:5} {c['games']:5} {c['bid']:5.1f} {c['win']:6.1f} {f(c['net'], 2):>6} "
            f"{ci(c['ci']):>16} {ci(c['ci_day']):>16} {f(a.get('net'), 1):>5}/{f(b.get('net'), 1):<5} ({a['fills']}/{b['fills']}) "
            f"{f(c['pct'], 1):>6}% {f(c['per_order'], 2):>6} {c['through']:8.0f} {c['through_q1']:7.0f} {100*c['share_1000']:4.0f}% {c['minutes']:5.0f}")


DIP_HEAD = (f"{'':34} {'orders':>6} {'fills':>5} {'games':>5} {'bid':>5} {'win%':>6} {'net c':>6} {'95% by game':>16} {'95% by day':>16} "
            f"{'halves (fills)':>19} {'net %':>7} {'/order':>6} {'through':>8} {'25th':>7} {'1000+':>5} {'min':>5}")


# ---------------------------------------------------------------- multiplicity

def simulate_passes(cells, rows, sims=400):
    """Passes per simulation when every strike is fairly priced and a ladder's strikes share one draw per game."""
    unit = lambda r: (r["gkey"], r["ladder"], r["team"] if r["ladder"] == "team_total" else "")
    keys = sorted({unit(r) for r in rows})
    index = {k: i for i, k in enumerate(keys)}
    idx = np.array([index[unit(r)] for r in rows])
    p = np.array([r["price"]/100 for r in rows])
    # YES rows win when the draw is below their price; NO rows on the same quantity win when it is above 1 - price.
    # Spread strikes on the two teams are opposite ends of one margin: the listed-first team's YES is the low end.
    flip = np.array([(r["side"] == "NO") != (r["ladder"].endswith("spread") and r["team"] != r["first_team"]) for r in rows])
    members = {k: np.array(v) for k, v in cells.items()}
    half = {k: (np.array([rows[i]["first_half"] for i in v])) for k, v in members.items()}
    games = {g: i for i, g in enumerate(sorted({r["gkey"] for r in rows}))}
    game_ids = np.array([games[r["gkey"]] for r in rows])
    implied = {k: 100*math.sqrt((p[v][half[k]]*(1-p[v][half[k]])).sum())/max(half[k].sum(), 1) for k, v in members.items()}
    counts, per_cell = [], Counter()
    for _ in range(sims):
        u = RNG.random(len(keys))[idx]
        win = np.where(flip, u > 1-p, u < p)
        g = 100*win-100*p
        gaps = {k: (g[v][half[k]], g[v][~half[k]], g[v]) for k, v in members.items()}
        passed = 0
        for k, v in members.items():
            a, b, _ = gaps[k]
            if len(a) < 20 or len(b) < 20 or b.mean() <= 0 or a.mean() <= 0:
                continue
            if any(gaps[n][2].mean() <= 0 for n in neighbours(k, members)):
                continue
            ids = game_ids[v][half[k]]
            inverse = np.unique(ids, return_inverse=True)[1]
            se = max(math.sqrt((np.bincount(inverse, weights=a-a.mean())**2).sum())/len(a), implied[k])
            if a.mean()-1.96*se > 0:
                passed += 1
                per_cell[k] += 1
        counts.append(passed)
    return np.array(counts), per_cell


def neighbours(key, cells):
    fam, ladder, side, band = key
    i = BANDS.index(band)
    return [(fam, ladder, side, BANDS[j]) for j in (i-1, i+1) if 0 <= j < len(BANDS) and (fam, ladder, side, BANDS[j]) in cells]


# ---------------------------------------------------------------- report

def cmd_report():
    rows = build()
    lines = []

    def say(text=""):
        print(text, flush=True)
        lines.append(text)
    summary = {}
    for r in rows:
        r["gkey"] = r["family"]+" "+r["game"]
    teams = defaultdict(set)
    for r in rows:
        teams[r["gkey"], r["ladder"]].add(r["team"])
    for r in rows:
        r["first_team"] = min(teams[r["gkey"], r["ladder"]])
    every = rows
    moved = {}
    for name in FAMILIES:
        got = [r for r in rows if r["family"] == name and "price30" in r]
        if got:
            moved[name] = mean(abs(r["price"]-r["price30"]) > 5 for r in got)
    # A family whose quotes are still jumping in the half hour before the "start" is either already in play (the
    # scheduled start is wrong) or has no real book: it stays in the inventory and leaves every other table.
    dropped = [name for name, share in moved.items() if share > MOVED_LIMIT]
    rows = [r for r in rows if r["family"] not in dropped]
    winners = [r for r in rows if r["ladder"] == "winner"]
    ladder = [r for r in rows if r["ladder"] != "winner" and 80 <= r["price"] < 99 and r["spread"] <= 4]
    fams = [n for n in FAMILIES if any(r["family"] == n for r in rows)]
    mark_halves(rows)

    say("LADDER STRIKES: are near-certain spread and total strikes underpriced before the game?")
    say(f"Reference: standing quote {REF} minutes before the scheduled start; two-sided, at most 4c wide; dearer side 80-99c.")
    say()
    say("1. INVENTORY (settled since 2026-09-18; quotes at the reference)")
    say(f"{'family':17} {'ladder':12} {'series':20} {'fee type':26} {'games':>5} {'strikes/game':>12} {'quoted':>6} {'med vol':>9} "
        f"{'med sprd':>8} {'<=4c':>5} {'80-99c/game':>11} {'settle-start h':>14}")
    cover = load(OUT/"coverage.json", {})
    inventory = []
    for name, kind, series in series_of():
        listed = load(RAW/f"markets-{series}.json", None)
        if not listed:
            continue
        ms = [m for m in listed["markets"].values() if m["value"] in (0.0, 1.0)]
        events = len({m["event"] for m in ms})
        got = [r for r in every if r["family"] == name and r["ladder"] == kind]
        games = {r["game"] for r in got}
        near = [r for r in got if 80 <= r["price"] < 99 and r["spread"] <= 4]
        dur = [(r["settle"]-r["start"])/3600 for r in got]
        fee_type = (listed["fee"] or {}).get("fee_type")
        item = dict(family=name, ladder=kind, series=series, fee_type=fee_type, fee_multiplier=(listed["fee"] or {}).get("fee_multiplier"),
                    listed_markets=len(ms), listed_events=events, strikes_per_game=len(ms)/events if events else None,
                    games_quoted=len(games), quoted=len(got), median_volume=median(r["volume"] for r in got) if got else None,
                    median_spread=median(r["spread"] for r in got) if got else None,
                    share_within_4c=mean(r["spread"] <= 4 for r in got) if got else None,
                    near_certain_per_game=len(near)/len(games) if games else None, median_hours=median(dur) if dur else None)
        inventory.append(item)
        if not got:
            say(f"{name:17} {kind:12} {series:20} {str(fee_type):26} {events:5} {item['strikes_per_game'] or 0:12.1f}   (listed; not downloaded)")
            continue
        say(f"{name:17} {kind:12} {series:20} {str(fee_type):26} {len(games):5} {len(ms)/events:12.1f} {len(got):6} {item['median_volume']:9.0f} "
            f"{item['median_spread']:8.1f} {100*item['share_within_4c']:4.0f}% {item['near_certain_per_game']:11.1f} {item['median_hours']:14.2f}")
    summary["inventory"] = inventory
    summary["coverage"] = cover
    say()
    say("Is the reference before play? Share of quoted strikes whose midpoint moved more than 5c between 30 and 5 minutes before the start:")
    for name, share in moved.items():
        say(f"  {name:18} {100*share:5.1f}%" + (f"   EXCLUDED from every table below (limit {100*MOVED_LIMIT:.0f}%)" if name in dropped else ""))
    summary["excluded_families"] = dropped

    say("First trade 8c or more below the reference price on a 90c+ strike, minutes after the scheduled start (1st, 5th, 25th, 50th percentile):")
    for name in fams:
        when = [next((x[0]/60 for x in r["path"] if x[1] < r["price"]-8), None) for r in rows if r["family"] == name and "path" in r]
        when = [w for w in when if w is not None]
        if len(when) >= 20:
            q = np.percentile(when, [1, 5, 25, 50])
            say(f"  {name:18} {q[0]:5.0f} {q[1]:5.0f} {q[2]:5.0f} {q[3]:5.0f}   ({len(when)} strikes; {100*mean(w < 5 for w in when):.1f}% within five minutes)")
    say()
    say("2. CALIBRATION OF NEAR-CERTAIN STRIKES (dearer side 80-99c). gap = win% - price; net = buy at the ask after the taker fee")
    cells, table = {}, {}
    for name in fams:
        say()
        say(f"--- {name}")
        say(HEAD)
        fam_rows = [r for r in ladder if r["family"] == name]
        for kind in FAMILIES[name]["ladders"]:
            for side in ("YES", "NO"):
                for lo, hi in BANDS:
                    part = [r for r in in_band(fam_rows, lo, hi) if r["ladder"] == kind and r["side"] == side]
                    if len(part) < 40:
                        continue
                    cells[name, kind, side, (lo, hi)] = part
                    table[f"{name}|{kind}|{side}|{lo}-{hi}"] = {k: v for k, v in cell(part).items()}
                    say(line(f"{kind} {side} {lo}-{hi}", part))
        for lo, hi in BANDS+((90, 99),):
            say(line(f"ALL ladders {lo}-{hi}", in_band(fam_rows, lo, hi)))
        say(line("winner favourite 90-99 (same ref)", [r for r in winners if r["family"] == name and r["side"] == "YES" and 90 <= r["price"] < 99 and r["spread"] <= 4]))
    summary["cells"] = table
    say()
    say("--- pooled over all families")
    say(HEAD)
    pooled = {}
    for label, pick in (("spreads (full and half)", lambda r: r["ladder"].endswith("spread")), ("totals (full, half, team)", lambda r: r["ladder"].endswith("total")),
                        ("YES side", lambda r: r["side"] == "YES"), ("NO side", lambda r: r["side"] == "NO"), ("everything", lambda r: True)):
        for lo, hi in BANDS+((90, 99),):
            part = [r for r in in_band(ladder, lo, hi) if pick(r)]
            say(line(f"{label} {lo}-{hi}", part))
            pooled[f"{label}|{lo}-{hi}"] = cell(part)
    summary["pooled"] = pooled
    say()
    say(f"Check: same tables on the quote {REF2} minutes before the start (pooled, 90-99c)")
    alt = []
    for r in rows:
        if r["ladder"] != "winner" and "price30" in r and 80 <= r["price30"] < 99 and r["spread30"] <= 4:
            alt.append(dict(r, price=r["price30"], cost=r["price30"]+r["spread30"]/2))
    say(HEAD)
    for lo, hi in BANDS+((90, 99),):
        say(line(f"everything {lo}-{hi} at -{REF2} min", in_band(alt, lo, hi)))

    say()
    say("3. RELATION TO THE GAME-WINNER FAVOURITE")
    say(HEAD)
    relation = {}
    groups = (("game has a 90c+ favourite", lambda r: r["fav"] is not None and r["fav"] >= 90),
              ("favourite 80-90c", lambda r: r["fav"] is not None and 80 <= r["fav"] < 90),
              ("favourite under 80c", lambda r: r["fav"] is not None and r["fav"] < 80),
              ("no winner quote", lambda r: r["fav"] is None))
    for label, pick in groups:
        for what, kind_pick in (("spreads", lambda r: r["ladder"].endswith("spread")), ("totals", lambda r: r["ladder"].endswith("total"))):
            for lo, hi in ((80, 90), (90, 99)):
                part = [r for r in in_band(ladder, lo, hi) if pick(r) and kind_pick(r)]
                say(line(f"{label[:22]}: {what} {lo}-{hi}", part))
                relation[f"{label}|{what}|{lo}-{hi}"] = cell(part)
    summary["relation"] = relation
    say()
    say("Spread strikes in games with a 90c+ favourite, by direction (with = it wins if the favourite covers; against = it wins if the favourite's margin stays small)")
    say(HEAD)
    heavy = [r for r in ladder if r["fav"] is not None and r["fav"] >= 90]
    with_fav = lambda r: (r["team"] == r["fav_team"]) == (r["side"] == "YES")
    for label, pick in (("spread with favourite", lambda r: r["ladder"].endswith("spread") and with_fav(r)),
                        ("spread against favourite", lambda r: r["ladder"].endswith("spread") and not with_fav(r)),
                        ("total YES (over)", lambda r: r["ladder"].endswith("total") and r["side"] == "YES"),
                        ("total NO (under)", lambda r: r["ladder"].endswith("total") and r["side"] == "NO")):
        for lo, hi in ((80, 90), (90, 99)):
            part = [r for r in in_band(heavy, lo, hi) if pick(r)]
            say(line(f"{label} {lo}-{hi}", part))
            relation[f"heavy|{label}|{lo}-{hi}"] = cell(part)
    say()
    say("All ladder strikes at 90-99c in games with a 90c+ favourite, by family, beside the favourite itself in those games (split looked at after the pooled result)")
    say(HEAD)
    for name in fams+["ALL"]:
        part = [r for r in in_band(heavy, 90, 99) if name == "ALL" or r["family"] == name]
        keys = {r["gkey"] for r in part}
        favs = [r for r in winners if r["gkey"] in keys and r["side"] == "YES" and 90 <= r["price"] < 99 and r["spread"] <= 4]
        if len(part) >= 20:
            say(line(f"{name}: ladder strikes", part))
            say(line(f"{name}: the favourite", favs))
            relation[f"heavy all|{name}"] = dict(ladder=cell(part), favourite=cell(favs))
    say()
    say("Same bet as the favourite? Near-certain 90-99c strikes in games with a 90c+ favourite:")
    part = [r for r in in_band(ladder, 90, 99) if r["fav"] is not None and r["fav"] >= 90]
    for what, kind_pick in (("spreads", lambda r: r["ladder"].endswith("spread")), ("totals", lambda r: r["ladder"].endswith("total"))):
        sub = [r for r in part if kind_pick(r)]
        if sub:
            lost_fav = [r for r in sub if r["fav_won"] == 0]
            won_fav = [r for r in sub if r["fav_won"] == 1]
            both = len({r["payout"] for r in sub}) > 1 and bool(lost_fav) and bool(won_fav)
            corr = np.corrcoef([r["payout"] for r in sub], [r["fav_won"] for r in sub])[0, 1] if both else float("nan")
            say(f"  {what:8} {len(sub):6} strikes; win rate when the favourite won {100*mean(r['payout'] for r in won_fav) if won_fav else float('nan'):5.1f}% "
                f"({len(won_fav)}), when it lost {100*mean(r['payout'] for r in lost_fav) if lost_fav else float('nan'):5.1f}% ({len(lost_fav)}); "
                f"correlation with the favourite's result {corr:+.2f}")
    say()
    say("Dependence inside a game (90-99c strikes): intraclass correlation of the result minus the price, and losses per losing game")
    for what, kind_pick in (("spreads", lambda r: r["ladder"].endswith("spread")), ("totals", lambda r: r["ladder"].endswith("total")), ("all ladders", lambda r: True)):
        sub = [r for r in in_band(ladder, 90, 99) if kind_pick(r)]
        by = defaultdict(list)
        for r in sub:
            by[r["gkey"]].append(r["payout"]-r["price"]/100)
        multi = [v for v in by.values() if len(v) >= 2]
        if not multi:
            continue
        grand = mean(x for v in multi for x in v)
        between = sum(len(v)*(mean(v)-grand)**2 for v in multi)/(len(multi)-1)
        within = sum((x-mean(v))**2 for v in multi for x in v)/(sum(len(v) for v in multi)-len(multi))
        k = mean(len(v) for v in multi)
        icc = (between-within)/(between+(k-1)*within)
        losing = [sum(x < -.5 for x in v) for v in by.values() if any(x < -.5 for x in v)]
        design = (cluster_se(sub, gap, "gkey")/(np.std([gap(r) for r in sub])/math.sqrt(len(sub))))**2
        say(f"  {what:12} {len(sub):6} strikes in {len(by)} games ({k:.1f} per game); ICC {icc:+.2f}; design effect {design:.1f} "
            f"(effective n {len(sub)/design:.0f}); {len(losing)} games with a loss, {mean(losing):.1f} losses each, most {max(losing)}")
        summary.setdefault("dependence", {})[what] = dict(icc=icc, design_effect=design, losing_games=len(losing), mean_losses=mean(losing), most=max(losing))

    say()
    say("4. DIP VERSION: strike at 90c+ before the game, bid resting from the start, filled by a later trade strictly below it, held")
    orders = [r for r in rows if r["ladder"] != "winner" and 90 <= r["price"] < 99 and r["spread"] <= 4 and "path" in r]
    summary["dip"] = {}
    for label, level_of in (("bid 80c", lambda r: 80), ("bid 10c below the pre-game price", lambda r: math.floor(r["price"])-10), ("bid 70c", lambda r: 70)):
        say()
        say(f"{label} (net after the maker fee where the series has one)")
        say(DIP_HEAD)
        for name in fams:
            fam_orders = [r for r in orders if r["family"] == name]
            if not fam_orders:
                continue
            for what, kind_pick in (("spreads", lambda r: r["ladder"].endswith("spread")), ("totals", lambda r: r["ladder"].endswith("total"))):
                for side in ("YES", "NO"):
                    say(dip_line(f"{name} {what} {side}", [r for r in fam_orders if kind_pick(r) and r["side"] == side], level_of))
            say(dip_line(f"{name} ALL", fam_orders, level_of))
        say(dip_line("ALL families", orders, level_of))
        for tag, pick in (("spreads", lambda r: r["ladder"].endswith("spread")), ("totals", lambda r: r["ladder"].endswith("total")),
                          ("with a 90c+ favourite", lambda r: r["fav"] is not None and r["fav"] >= 90),
                          ("without a 90c+ favourite", lambda r: r["fav"] is None or r["fav"] < 90),
                          ("pre-game 90-95", lambda r: r["price"] < 95), ("pre-game 95-99", lambda r: r["price"] >= 95)):
            say(dip_line(f"ALL {tag}", [r for r in orders if pick(r)], level_of))
        c = dip_cell(orders, level_of)
        summary["dip"][label] = {k: v for k, v in c.items() if k != "rows"}
        if c["fills"]:
            one = {}
            for x in sorted(c["rows"], key=lambda x: x["when"]):
                one.setdefault(x["gkey"], x)
            v = [x["dnet"] for x in one.values()]
            se = np.std(v)/math.sqrt(len(v))
            say(f"  first fill per game only: {len(v)} fills, win {100*mean(x['payout'] for x in one.values()):.1f}%, net {mean(v):+.2f}c "
                f"[{mean(v)-1.96*se:+.1f} to {mean(v)+1.96*se:+.1f}]")
            nofee = dip_cell(orders, level_of, maker=False)
            say(f"  with no maker fee: net {nofee['net']:+.2f}c per fill; break-even win rate {c['bid']:.1f}%; won {c['win']:.1f}%")
            say(f"  the lowest trade of the minute that filled the bid was {c['overshoot']:.1f}c below it on average (median {c['overshoot_median']:.1f}c): "
                f"the price had usually jumped through the bid, not drifted to it")

    say()
    say("5. TAIL: a book holding one contract of every qualifying 90-99c strike, bought at the ask after fees (cents)")
    book = in_band(ladder, 90, 99)
    tail = {}
    for label, unit in (("game", "gkey"), ("day", "day")):
        by = defaultdict(list)
        for r in book:
            by[r[unit]].append(r)
        totals = sorted(((sum(net(r) for r in v), k, len(v), sum(r["payout"] < .5 for r in v)) for k, v in by.items()))
        pnl = [t[0] for t in totals]
        staked = sum(r["cost"] for r in book)/len(by)
        say(f"  by {label}: {len(by)} {label}s, mean {mean(pnl):+.1f}c, median {median(pnl):+.1f}c, share losing {100*mean(p < 0 for p in pnl):.0f}%, "
            f"mean stake {staked:.0f}c")
        for t in totals[:5]:
            say(f"     worst {label} {t[1]:34} {t[0]:+8.1f}c  {t[3]} of {t[2]} strikes lost")
        tail[label] = dict(units=len(by), mean=mean(pnl), median=median(pnl), share_losing=mean(p < 0 for p in pnl),
                           worst=[dict(key=t[1], pnl=t[0], strikes=t[2], lost=t[3]) for t in totals[:5]])
    losing = Counter()
    by = defaultdict(list)
    for r in book:
        by[r["gkey"]].append(r)
    for v in by.values():
        losing[sum(r["payout"] < .5 for r in v)] += 1
    say("  losses per game: "+", ".join(f"{k} lost: {n} games" for k, n in sorted(losing.items())))
    summary["tail"] = tail

    say()
    say("6. CAPACITY AND FREQUENCY (90-99c strikes; 80-90c in brackets)")
    capacity = {}
    for name in fams+["ALL"]:
        part = [r for r in book if name == "ALL" or r["family"] == name]
        wide = [r for r in in_band(ladder, 80, 90) if name == "ALL" or r["family"] == name]
        if not part:
            continue
        days = sorted({r["day"] for r in ladder if name == "ALL" or r["family"] == name})
        span = (date.fromisoformat(days[-1])-date.fromisoformat(days[0])).days+1
        per_day = len(part)/len(days)
        cost = mean(r["cost"] for r in part)/100
        n_day = mean(net(r) for r in part)*per_day/100
        say(f"  {name:18} {len(part):6} strikes on {len(days)} game days of {span} ({per_day:6.1f} a game day; [{len(wide)/len(days):.1f}]); "
            f"median spread {median(r['spread'] for r in part):.0f}c, median volume in last {BEFORE} min {median(r['pre_volume'] for r in part):,.0f}, "
            f"lifetime {median(r['volume'] for r in part):,.0f}; stake a day ${10*cost*per_day:,.0f} at 10, ${100*cost*per_day:,.0f} at 100; "
            f"net a day ${10*n_day:+,.2f} / ${100*n_day:+,.2f}")
        by_day = defaultdict(lambda: [0.0, 0.0])
        for r in part:
            by_day[r["day"]][0] += net(r)
            by_day[r["day"]][1] += r["cost"]+fee(r)
        returns = sorted(100*a/b for a, b in by_day.values())
        say(f"  {'':18} trades in the last {BEFORE} min: {100*mean(r['pre_volume'] > 0 for r in part):.0f}% of strikes, mean {mean(r['pre_volume'] for r in part):,.0f} contracts; "
            f"return on the day's stake: mean {mean(returns):+.2f}%, median {median(returns):+.2f}%, worst {returns[0]:+.1f}%, best {returns[-1]:+.1f}%, "
            f"losing days {100*mean(v < 0 for v in returns):.0f}%")
        capacity[name] = dict(day_return_mean=mean(returns), day_return_median=median(returns), day_return_worst=returns[0],
                              share_losing_days=mean(v < 0 for v in returns), share_traded_pre=mean(r["pre_volume"] > 0 for r in part),strikes=len(part), game_days=len(days), calendar_days=span, per_game_day=per_day, median_spread=median(r["spread"] for r in part),
                              median_pre_volume=median(r["pre_volume"] for r in part), median_volume=median(r["volume"] for r in part),
                              stake_10=10*cost*per_day, stake_100=100*cost*per_day, net_day_10=10*n_day, net_day_100=100*n_day)
    summary["capacity"] = capacity

    say()
    say("7. MULTIPLICITY")
    passed = []
    for key, part in cells.items():
        a, b = [r for r in part if r["first_half"]], [r for r in part if not r["first_half"]]
        if len(a) < 20 or len(b) < 20:
            continue
        ga, gb = mean(map(gap, a)), mean(map(gap, b))
        ok = ga-1.96*gap_se(a) > 0 and gb > 0 and all(mean(map(gap, cells[n])) > 0 for n in neighbours(key, cells))
        if ok:
            passed.append(key)
    index = {id(r): i for i, r in enumerate(ladder)}
    members = {k: [index[id(r)] for r in v] for k, v in cells.items()}
    sims, per_cell = simulate_passes(members, ladder)
    say(f"  cells with at least 40 strikes: {len(cells)}; passing (first-half interval above zero, second half positive, neighbours positive): {len(passed)}")
    for k in passed:
        c = cell(cells[k])
        say(f"     {k[0]} {k[1]} {k[2]} {k[3][0]}-{k[3][1]}: n {c['n']}, gap {c['gap']:+.2f}c {ci(c['ci_game'])}, net {c['net']:+.2f}c; passed in {100*per_cell[k]/len(sims):.1f}% of fair simulations")
    say(f"  under fair pricing with the same within-game dependence: mean {sims.mean():.2f} passes, 95th percentile {np.percentile(sims, 95):.0f}, "
        f"P(at least {len(passed)}) = {float((sims >= len(passed)).mean()):.2f}")
    above = [k for k, v in cells.items() if cell(v)["ci_game"] and cell(v)["ci_game"][0] > 0]
    below = [k for k, v in cells.items() if cell(v)["ci_game"] and cell(v)["ci_game"][1] < 0]
    say(f"  cells whose whole-sample game-clustered interval is above zero: {len(above)}; below zero: {len(below)}; "
        f"cells with a positive gap {sum(cell(v)['gap'] > 0 for v in cells.values())} of {len(cells)}")
    est = {" ".join((k[0], k[1], k[2], f"{k[3][0]}-{k[3][1]}")): (cell(v)["gap"], max(cell(v)["se"], .05)) for k, v in cells.items()}
    sh = shrink(est, prior_sd=3.0)
    say(f"  shrunk together: overall mean gap {sh['overall_mean']:+.2f}c (se {sh['overall_se']:.2f}), between-cell sd {sh['between_family_sd']:.2f}c "
        f"{sh['between_family_sd_95']}")
    ranked = sorted(sh["families"].items(), key=lambda kv: -kv[1]["shrunk"])
    for k, v in ranked[:8]+ranked[-5:]:
        say(f"     {k:44} raw {est[k][0]:+6.2f}c  shrunk {v['shrunk']:+5.2f}c (sd {v['sd']:.2f}), P(>0) {v['prob_positive']:.2f}")
    say("  whole families (all ladders, 90-99c), shrunk together the same way:")
    fam_est = {name: (cell(v)["gap"], cell(v)["se"]) for name in fams if len(v := [r for r in book if r["family"] == name]) >= 40}
    fam_sh = shrink(fam_est, prior_sd=3.0) if len(fam_est) >= 2 else None
    if fam_sh:
        for k, v in fam_sh["families"].items():
            c = cell([r for r in book if r["family"] == k])
            z = fam_est[k][0]/fam_est[k][1]
            say(f"     {k:18} raw {fam_est[k][0]:+5.2f}c (se {fam_est[k][1]:.2f}, two-sided p {math.erfc(abs(z)/math.sqrt(2)):.3f}, "
                f"x{len(fam_est)} families = {min(1, len(fam_est)*math.erfc(abs(z)/math.sqrt(2))):.2f})  shrunk {v['shrunk']:+5.2f}c (sd {v['sd']:.2f}), "
                f"P(>0) {v['prob_positive']:.2f};  at the ask after fees {c['net']:+.2f}c = {c['pct']:+.2f}% {ci(c['pct_ci'], 2)}; "
                f"shrunk gap less the raw cost {v['shrunk']-(c['gap']-c['net']):+.2f}c")
    summary["family_shrink"] = fam_sh
    summary["multiplicity"] = dict(cells=len(cells), passed=[list(map(str, k)) for k in passed], fair_mean=float(sims.mean()),
                                   fair_p95=float(np.percentile(sims, 95)), p_at_least=float((sims >= len(passed)).mean()),
                                   intervals_above=len(above), intervals_below=len(below), shrink=sh)
    dump(OUT/"summary.json", summary)
    (OUT/"report.txt").write_text("\n".join(lines)+"\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("command", choices=("list", "pregame", "ingame", "build", "report"))
    parser.add_argument("--minutes", type=float, default=10)
    parser.add_argument("--families", nargs="*")
    parser.add_argument("--skip", nargs="*", default=[], help="pregame: ladder types to leave out (for the request budget)")
    args = parser.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    RAW.mkdir(parents=True, exist_ok=True)
    if args.command == "report":
        return cmd_report()
    if args.command == "build":
        return build()
    began = time.time()
    deadline[0] = began+60*args.minutes
    try:
        with httpx.Client(timeout=60) as client:
            if args.command == "pregame":
                cmd_pregame(client, args.families, set(args.skip))
            else:
                dict(list=cmd_list, ingame=cmd_ingame)[args.command](client, args.families)
    except OutOfTime:
        pass
    finish_log(args.command, began)


if __name__ == "__main__":
    main()
