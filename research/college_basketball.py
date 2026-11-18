"""Join men's college basketball play-by-play to Kalshi minute quotes, then search late-game rules.

Kalshi series KXNCAAMBGAME. The play-by-play archive lists every Division I
game for a date (groups=50, limit=300). Team codes differ between the two
sources, so a Kalshi event is linked to an archive game by date, tip-off time
and BOTH team names (Kalshi's yes_sub_title against the archive's names). An
event with no unique link is skipped, never guessed. After linking, the
archive's winner must agree with the exchange settlement or the game is dropped.

Rows have the same shape as the other sports in game_state.py (plus the entry
minute's volume and spread, for judging thin books) and go to
data/processed/game-state/ncaamb-minutes.json. `--search` then runs
rule_search.py's selection on the earliest 60% of games and scores the latest 40%.
"""
import argparse
from bisect import bisect_right
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from datetime import date, timedelta
from difflib import SequenceMatcher
import json
from pathlib import Path
import re
import sys
import time
import unicodedata

import httpx

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from research import game_state, rule_search
from research.api_cost import fetch, midnight, pages, save, timestamp
from research.game_state import FEED_DELAY, MONTHS, OUT, RAW, cached, plays, timed
from research.near_end import WINDOW, cents

SPORT, SERIES = "ncaamb", "KXNCAAMBGAME"
PATH = "basketball/mens-college-basketball"
# Registered at run time so game_state.plays and rule_search.search work unchanged.
game_state.SPORTS[SPORT] = dict(series=SERIES, path=PATH, state=timed(2, 1200, 300), late=2, unit="seconds", alias={})
rule_search.GRID[SPORT] = dict(lead=(1, 4, 6, 8, 10, 12, 15), left_max=(2400, 1200, 600, 300, 180),
                               left_min=(0, 30, 60, 120), cushion=(0, 2, 3, 4, 5))
WORDS = {"st": "state", "univ": "university", "u": "university", "mt": "mount", "ft": "fort",
         "n": "north", "s": "south", "e": "east", "w": "west", "so": "southern", "no": "northern"}
# Schools the two sources name differently (Kalshi's name -> the archive's). Both teams must still match on the date.
SAME = {"liu": "long island university", "north carolina state": "nc state", "tennessee martin": "ut martin",
        "university at albany": "ualbany", "louisiana monroe": "ul monroe", "southeastern louisiana": "se louisiana",
        "appalachian state": "app state"}


def norm(name):
    text = unicodedata.normalize("NFKD", name or "").lower().replace("&", " and ")
    text = "".join(c for c in text if not unicodedata.combining(c)).replace("'", "").replace("’", "").replace("`", "")
    words = re.findall(r"[a-z0-9]+", text)
    if words and words[0] in ("st", "saint"):       # "St. John's" is a saint; "Ohio St." is a state
        words[0] = "saint"
    text = " ".join(WORDS.get(w, w) for w in words)
    return SAME.get(text, text)


def kalshi_games(client):
    """One canonical market per settled event, with both team codes and the names Kalshi shows."""
    events = defaultdict(list)
    listings = (pages(client, "/historical/markets", "markets", series_ticker=SERIES, limit=1000),
                pages(client, "/markets", "markets", series_ticker=SERIES, status="settled", limit=1000,
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
        names = {g[1]["ticker"].rsplit("-", 1)[1]: g[1].get("yes_sub_title") or "" for g in group}
        if len(names) == 2 and outcome in (0, 1):
            yield dict(event=event, ticker=m["ticker"], yes=m["ticker"].rsplit("-", 1)[1], names=names,
                       day=day, settle=settle, outcome=outcome, archived=archived)


def slate(client, day):
    def trim(j):
        out = []
        for e in j.get("events", []):
            teams = {}
            for c in e["competitions"][0]["competitors"]:
                t = c["team"]
                teams[c["homeAway"]] = dict(code=t.get("abbreviation") or "", names=sorted(
                    {t.get(k) for k in ("location", "shortDisplayName", "displayName") if t.get(k)}))
            if len(teams) == 2:
                out.append(dict(id=e["id"], status=e["status"]["type"]["name"], start=int(timestamp(e["date"])), teams=teams))
        return out
    return cached(client, RAW/SPORT/f"day-{day:%Y%m%d}.json", PATH+"/scoreboard",
                  dict(dates=f"{day:%Y%m%d}", groups=50, limit=300), trim)


def strength(code, name, team):
    """2: Kalshi's name equals one of the archive's names. 1: close (same code, or a near spelling). 0: no."""
    mine, theirs = norm(name), [norm(n) for n in team["names"]]
    if mine and mine in theirs:
        return 2
    if not mine:
        return 0
    a = set(mine.split())
    for other in theirs:
        b = set(other.split())
        if SequenceMatcher(None, mine, other).ratio() >= .8 or (len(a & b) >= 1 and (a <= b or b <= a) and code == team["code"]):
            return 1
    return 0


def link(g, slates):
    """The single archive game on the same date with both teams; None when absent or ambiguous."""
    (c1, n1), (c2, n2) = sorted(g["names"].items())
    found = []
    for shift in (0, -1, 1):
        for e in slates.get(g["day"]+timedelta(days=shift), []):
            # A game lasts about two hours; settlement follows the final whistle.
            if e["status"] != "STATUS_FINAL" or not 3600 <= g["settle"]-e["start"] <= 6*3600:
                continue
            home, away = e["teams"]["home"], e["teams"]["away"]
            for first, second in ((home, away), (away, home)):
                a, b = strength(c1, n1, first), strength(c2, n2, second)
                if a and b and a+b >= 3:                   # at least one exact name, the other exact or close
                    found.append((a+b, e, {c1: first is home, c2: second is home}))
    if not found:
        return None, "no_archive_game_with_both_teams"
    best = max(f[0] for f in found)
    top = [f for f in found if f[0] == best]
    if len({f[1]["id"] for f in top}) != 1 or len(top) != 1:
        return None, "ambiguous_archive_match"
    return top[0], "exact_names" if best == 4 else "one_exact_one_close"


def join(g, e, yes_home, log, bars):
    """Late-game minutes of one game. Returns rows or a reason for skipping."""
    if not bars or len(log) < 50:
        return "no_bars" if not bars else "no_timed_plays"
    times = [p[0] for p in log]
    if (log[-1][3] > log[-1][4]) != ((g["outcome"] == 1) == yes_home):
        return "archive_winner_disagrees_with_settlement"
    rows = []
    bars = sorted(bars, key=lambda b: b["end_period_ts"])
    for bar, nxt in zip(bars, bars[1:]):
        i = bisect_right(times, bar["end_period_ts"]-FEED_DELAY)-1
        if i < 0 or nxt["end_period_ts"]-bar["end_period_ts"] > 120 or bar["end_period_ts"] > times[-1]:
            continue
        _, period, left, home, away = log[i]
        if period < 2 or home == away:
            continue
        leader_yes = (home > away) == yes_home
        bid, offer = cents(nxt, "yes_bid", "open"), cents(nxt, "yes_ask", "open")
        ask = offer if leader_yes else (None if bid is None else 100-bid)
        if ask is None or not 0 < ask < 100:
            continue
        won = (g["outcome"] == 1) == leader_yes
        rows.append(dict(event=g["event"], day=str(g["day"]), ts=nxt["end_period_ts"], period=period,
                         left=left, lead=abs(home-away), ask=ask, won=won,
                         net=100*(won-ask/100-.07*ask/100*(1-ask/100)),
                         spread=None if bid is None or offer is None else offer-bid,
                         volume=float(nxt.get("volume_fp") or nxt.get("volume") or 0)))
    return rows


def build(client, since, until, budget):
    (RAW/SPORT).mkdir(parents=True, exist_ok=True)
    games = [g for g in kalshi_games(client) if since <= g["day"] <= until]
    # Current season first, then the most recent of the older games, so a short download budget covers what matters most.
    games.sort(key=lambda g: (g["day"] < date(2026, 11, 1), -g["day"].toordinal() if g["day"] < date(2026, 11, 1) else g["day"].toordinal()))
    cover, unmatched = Counter(kalshi_games=len(games)), []
    days = sorted({g["day"]+timedelta(days=s) for g in games for s in (-1, 0, 1)})
    with ThreadPoolExecutor(4) as archive, ThreadPoolExecutor(2) as kalshi:      # limits: 4 archive, 2 Kalshi requests at once
        slates = dict(zip(days, archive.map(lambda d: slate(client, d), days)))
        linked = []
        for g in games:
            hit, how = link(g, slates)
            cover[how] += 1
            if hit:
                linked.append((g, hit[1], hit[2][g["yes"]]))
            else:
                unmatched.append(dict(event=g["event"], names=g["names"], why=how))
        cover["matched"] = len(linked)
        logs = list(archive.map(lambda item: plays(client, SPORT, item[1]["id"]), linked))
        deadline = time.time()+budget*60

        def bars_for(item):
            g, log = item
            if len(log) < 50:
                return None                           # no timed plays: the bars would be unusable, so do not download them
            if time.time() > deadline:
                return "late"
            path = ("/historical/markets/{t}/candlesticks" if g["archived"]
                    else "/series/"+SERIES+"/markets/{t}/candlesticks").format(t=g["ticker"])
            try:
                return fetch(client, path, dict(start_ts=g["settle"]-WINDOW, end_ts=g["settle"], period_interval=1))["candlesticks"]
            except (RuntimeError, KeyError):
                return None
        rows, used = [], []
        all_bars = kalshi.map(bars_for, [(g, log) for (g, _, _), log in zip(linked, logs)])
        for n, ((g, e, yes_home), log, bars) in enumerate(zip(linked, logs, all_bars)):
            if n % 250 == 0:
                print(f"  bars {n}/{len(linked)}", flush=True)
            if bars == "late":
                cover["not_downloaded_within_budget"] += 1
                continue
            out = join(g, e, yes_home, log, bars)
            if isinstance(out, str):
                cover[out] += 1
            elif out:
                rows += out
                used.append(g["day"])
            else:
                cover["no_late_minute_with_a_quote"] += 1
    cover.update(joined=len(used), minutes=len(rows))
    cover = dict(cover, first_day=str(min(used)) if used else None, last_day=str(max(used)) if used else None,
                 by_month=dict(sorted(Counter(f"{d:%Y-%m}" for d in used).items())))
    return rows, cover, unmatched


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--since", type=date.fromisoformat, default=date(2025, 11, 1))
    parser.add_argument("--until", type=date.fromisoformat, default=date(2026, 11, 30))
    parser.add_argument("--budget", type=float, default=30, help="minutes allowed for downloading Kalshi bars")
    parser.add_argument("--search", action="store_true", help="only run the rule search on rows already joined")
    args = parser.parse_args()
    if not args.search:
        with httpx.Client(timeout=40, headers={"User-Agent": "research-script"}) as client:
            rows, cover, unmatched = build(client, args.since, args.until, args.budget)
        save(OUT/f"{SPORT}-minutes.json", rows)
        save(OUT/f"{SPORT}-coverage.json", cover)
        save(OUT/f"{SPORT}-unmatched.json", unmatched)
        print(json.dumps(cover), flush=True)
    report = rule_search.search(SPORT)
    save(OUT/f"{SPORT}-rules.json", report)
    rule_search.show(report)


if __name__ == "__main__":
    main()
