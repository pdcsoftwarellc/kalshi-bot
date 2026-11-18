"""Join play-by-play to Kalshi minute quotes for NBA, MLB, NHL and NFL winner markets.

For every settled game with Kalshi one-minute bars, rebuild the score and the
time (or outs) remaining at each minute from a public play-by-play archive in
which every play carries a wall-clock time. A decision at a bar's close may
use plays stamped at least 15 seconds earlier. Each output row is one
late-game minute: the leader, the lead, what remains, and the next minute's
opening ask on the leader, which is the price a simulated order would pay.

`rule_search.py` reads these rows. Nothing here selects or scores a rule.
"""
import argparse
from bisect import bisect_right
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from datetime import date
import json
from pathlib import Path
import sys
import time

import httpx

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from research.api_cost import ROOT, fetch, midnight, pages, save, timestamp
from research.near_end import WINDOW, cents

RAW = ROOT / "data/raw/game-state"
OUT = ROOT / "data/processed/game-state"
ARCHIVE = "https://site.api.espn.com/apis/site/v2/sports/"
FEED_DELAY = 15
MONTHS = dict(JAN=1, FEB=2, MAR=3, APR=4, MAY=5, JUN=6, JUL=7, AUG=8, SEP=9, OCT=10, NOV=11, DEC=12)


def clock(text):
    parts = str(text).split(":")
    return float(parts[0])*60+float(parts[1]) if len(parts) == 2 else float(parts[0])


def timed(regulation, length, overtime, counts_up=False):
    """Seconds left in regulation (or in the overtime period) for clock sports."""
    def left(p):
        number, shown = p["period"]["number"], clock(p["clock"]["displayValue"])
        size = length if number <= regulation else overtime
        remaining = max(size-shown, 0) if counts_up else shown
        return number, remaining+(regulation-number)*length if number <= regulation else remaining
    return left


def baseball(p):
    """Outs the trailing team still has in regulation; extra innings count as the ninth."""
    inning, half, outs = min(p["period"]["number"], 9), p["period"]["type"], int(p.get("outs") or 0)
    home, away = int(p["homeScore"]), int(p["awayScore"])
    if home == away:
        return p["period"]["number"], 0
    top = half == "Top"
    if away < home:                       # the trailing team bats in the top half
        left = (9-inning)*3+(3-outs if top else 0)
    else:
        left = (9-inning)*3+(3 if top else 3-outs)
    return p["period"]["number"], left


SPORTS = {
    "nba": dict(series="KXNBAGAME", path="basketball/nba", state=timed(4, 720, 300), late=4, unit="seconds",
                alias={"GS": "GSW", "NO": "NOP", "NY": "NYK", "SA": "SAS", "UTAH": "UTA", "WSH": "WAS"}),
    "nhl": dict(series="KXNHLGAME", path="hockey/nhl", state=timed(3, 1200, 300, counts_up=True), late=3, unit="seconds",
                alias={"UTAH": "UTA"}),
    "nfl": dict(series="KXNFLGAME", path="football/nfl", state=timed(4, 900, 600), late=4, unit="seconds",
                alias={"WSH": "WAS", "JAX": "JAC"}),
    "mlb": dict(series="KXMLBGAME", path="baseball/mlb", state=baseball, late=7, unit="outs",
                alias={"CHW": "CWS", "OAK": "ATH", "WSH": "WSH", "ARI": "AZ"}),
}


def cached(client, file, path, params, trim):
    if file.exists():
        return json.loads(file.read_text())
    for attempt in range(4):
        try:
            r = client.get(ARCHIVE+path, params=params)
        except httpx.RequestError:
            time.sleep(1+attempt)
            continue
        if r.status_code == 200:
            break
        time.sleep(1+attempt)
    else:
        return []                              # not cached, so a later run can try again
    data = trim(r.json())
    save(file, data)
    return data


def schedule(client, sport, day):
    cfg = SPORTS[sport]
    def trim(j):
        return [dict(id=e["id"], status=e["status"]["type"]["name"], start=int(timestamp(e["date"])),
                     teams={c["homeAway"]: cfg["alias"].get(c["team"]["abbreviation"], c["team"]["abbreviation"])
                            for c in e["competitions"][0]["competitors"]}) for e in j.get("events", [])]
    return cached(client, RAW/sport/f"day-{day:%Y%m%d}.json", cfg["path"]+"/scoreboard",
                  dict(dates=f"{day:%Y%m%d}", limit=100), trim)


def plays(client, sport, game):
    cfg = SPORTS[sport]
    def trim(j):
        raw = j.get("plays") or [p for d in (j.get("drives") or {}).get("previous", []) for p in d.get("plays", [])]
        rows = []
        for p in raw:
            try:
                period, left = cfg["state"](p)
                rows.append([int(timestamp(p["wallclock"])), period, left, int(p["homeScore"]), int(p["awayScore"])])
            except (KeyError, TypeError, ValueError):
                continue
        return sorted(rows)
    return cached(client, RAW/sport/f"game-{game}.json", cfg["path"]+"/summary", dict(event=game), trim)


def kalshi_games(client, series):
    """Canonical market, both team codes, settlement and listed date for every settled event."""
    events = defaultdict(list)
    listings = (pages(client, "/historical/markets", "markets", series_ticker=series, limit=1000),
                pages(client, "/markets", "markets", series_ticker=series, status="settled", limit=1000,
                      min_close_ts=midnight(date(2026, 9, 1)), max_close_ts=midnight(date(2026, 11, 23))))
    for archived, listing in zip((True, False), listings):
        for m in listing:
            events[m["event_ticker"]].append((archived, m))
    for event, group in events.items():
        archived, m = min(group, key=lambda g: g[1]["ticker"])
        stamp = event.split("-")[1]
        try:
            day = date(2000+int(stamp[:2]), MONTHS[stamp[2:5]], int(stamp[5:7]))
            outcome, settle = float(m["settlement_value_dollars"]), int(timestamp(m["settlement_ts"]))
        except (KeyError, ValueError, TypeError):
            continue
        teams = {g[1]["ticker"].rsplit("-", 1)[1] for g in group}
        if len(teams) == 2 and outcome in (0, 1) and day >= date(2026, 1, 1):
            yield dict(event=event, ticker=m["ticker"], yes=m["ticker"].rsplit("-", 1)[1], teams=teams,
                       day=day, settle=settle, outcome=outcome, archived=archived)


def build(client, sport):
    cfg = SPORTS[sport]
    (RAW/sport).mkdir(parents=True, exist_ok=True)
    games = list(kalshi_games(client, cfg["series"]))
    skip, rows, missing = Counter(), [], Counter()
    days = sorted({g["day"] for g in games})
    with ThreadPoolExecutor(4) as pool:
        slates = dict(zip(days, pool.map(lambda d: schedule(client, sport, d), days)))
        linked = []
        for g in games:
            match = [e for e in slates[g["day"]] if set(e["teams"].values()) == g["teams"]
                     and e["status"] == "STATUS_FINAL" and e["start"] < g["settle"]]
            if not match:
                skip["no_matching_final_game"] += 1
                missing.update(g["teams"]-{t for e in slates[g["day"]] for t in e["teams"].values()})
                continue
            linked.append((g, max(match, key=lambda e: e["start"])))       # a doubleheader: the game that just ended
        logs = pool.map(lambda pair: plays(client, sport, pair[1]["id"]), linked)

        def bars_for(g):
            path = ("/historical/markets/{t}/candlesticks" if g["archived"]
                    else "/series/"+cfg["series"]+"/markets/{t}/candlesticks").format(t=g["ticker"])
            try:
                return fetch(client, path, dict(start_ts=g["settle"]-WINDOW, end_ts=g["settle"], period_interval=1))["candlesticks"]
            except (RuntimeError, KeyError):
                return None
        all_bars = pool.map(bars_for, [g for g, _ in linked])
        for (g, e), log, bars in zip(linked, logs, all_bars):
            if not bars or len(log) < 50:
                skip["no_bars_or_plays"] += 1
                continue
            times = [p[0] for p in log]
            yes_home = e["teams"]["home"] == g["yes"]
            if (log[-1][3] > log[-1][4]) != ((g["outcome"] == 1) == yes_home):
                skip["archive_winner_disagrees_with_settlement"] += 1
                continue
            bars = sorted(bars, key=lambda b: b["end_period_ts"])
            for bar, nxt in zip(bars, bars[1:]):
                i = bisect_right(times, bar["end_period_ts"]-FEED_DELAY)-1
                if i < 0 or nxt["end_period_ts"]-bar["end_period_ts"] > 120 or bar["end_period_ts"] > times[-1]:
                    continue
                _, period, left, home, away = log[i]
                if period < cfg["late"] or home == away:
                    continue
                leader_yes = (home > away) == yes_home
                if leader_yes:
                    ask = cents(nxt, "yes_ask", "open")
                else:
                    bid = cents(nxt, "yes_bid", "open")
                    ask = None if bid is None else 100-bid
                if ask is None or not 0 < ask < 100:
                    continue
                won = (g["outcome"] == 1) == leader_yes
                rows.append(dict(event=g["event"], day=str(g["day"]), ts=nxt["end_period_ts"], period=period,
                                 left=left, lead=abs(home-away), ask=ask, won=won,
                                 net=100*(won-ask/100-.07*ask/100*(1-ask/100))))
    cover = dict(skip, kalshi_games=len(games), joined=len({r["event"] for r in rows}), minutes=len(rows))
    if missing:
        cover["team_codes_not_in_archive"] = dict(missing.most_common(12))
    return rows, cover


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sports", nargs="+", choices=tuple(SPORTS), default=list(SPORTS))
    args = parser.parse_args()
    with httpx.Client(timeout=40, headers={"User-Agent": "research-script"}) as client:
        for sport in args.sports:
            rows, cover = build(client, sport)
            save(OUT/f"{sport}-minutes.json", rows)
            save(OUT/f"{sport}-coverage.json", cover)
            print(sport, json.dumps(cover), flush=True)


if __name__ == "__main__":
    main()
