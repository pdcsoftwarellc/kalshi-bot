"""Are Kalshi sports markets that the score has already decided still offered below 99-100c?

A market side counts as decided only when the play-by-play makes reversal
impossible under the market's rules text:
  over     game total, team total, or a first-period/half total: the count has
           already passed the line (points and runs are never taken back by play).
  final    the game's last play is on the record (winner, spread, total, team total).
  period   the period the market covers is complete (first half, first five
           innings, first period, first quarter).
Play-by-play is joined as in game_state.py: a decision at a bar's close may only
use plays stamped at least 15 seconds earlier, and the simulated order buys at
the NEXT minute's opening ask and pays the taker fee 0.07*p*(1-p).

Every "decided" state is re-checked against the result Kalshi settled on: the
play-by-play final must equal Kalshi's expiration value and the running count
must never later fall back to the line, otherwise the row is marked unverified
and reported separately (a wrong archive score, a stat correction, or an
overturned basket all show up this way).

`download` fetches one-minute bars (two workers, paced, cached through
api_cost.fetch, same cache entries as spreads_totals.py). `analyze` reads the
cache only. Research only: public reads, no orders.
"""
import argparse
from bisect import bisect_right
from collections import Counter, defaultdict
import json
from pathlib import Path
import random
import re
from statistics import mean, median
import sys
import time

import httpx

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from research.api_cost import ROOT, fetch, midnight, pages, save, timestamp
from research.game_state import FEED_DELAY, SPORTS, plays
from research.near_end import WINDOW, cents
from research.spreads_totals import cache_file, download, link, request, requests_for
from datetime import date

OUT = ROOT / "data/processed/decided"
# family -> (series, scope). scope: "game", or the last period number the market covers.
FAMILIES = {
    "nba": {"total": ("KXNBATOTAL", "game"), "team_total": ("KXNBATEAMTOTAL", "game"), "spread": ("KXNBASPREAD", "game"),
            "h1_total": ("KXNBA1HTOTAL", 2), "h1_winner": ("KXNBA1H", 2), "h1_spread": ("KXNBA1HSPREAD", 2),
            "q1_winner": ("KXNBA1Q", 1), "q1_total": ("KXNBA1QTOTAL", 1)},
    "mlb": {"total": ("KXMLBTOTAL", "game"), "team_total": ("KXMLBTEAMTOTAL", "game"), "spread": ("KXMLBSPREAD", "game"),
            "f5_total": ("KXMLBF5TOTAL", 5), "f5_winner": ("KXMLBF5", 5), "f5_spread": ("KXMLBF5SPREAD", 5)},
    "nhl": {"total": ("KXNHLTOTAL", "game"), "team_total": ("KXNHLTEAMTOTAL", "game"), "spread": ("KXNHLSPREAD", "game"),
            "p1_total": ("KXNHL1PTOTAL", 1), "p1_winner": ("KXNHL1P", 1)},
    "nfl": {"total": ("KXNFLTOTAL", "game"), "team_total": ("KXNFLTEAMTOTAL", "game"), "spread": ("KXNFLSPREAD", "game"),
            "h1_total": ("KXNFL1HTOTAL", 2), "h1_winner": ("KXNFL1H", 2), "h1_spread": ("KXNFL1HSPREAD", 2)},
}
ORDER = [("nhl", "total"), ("nfl", "total"), ("nba", "team_total"), ("mlb", "team_total"), ("nba", "h1_total"),
         ("nba", "h1_winner"), ("mlb", "f5_total"), ("mlb", "f5_winner"), ("nhl", "team_total"), ("nfl", "team_total"),
         ("nfl", "h1_total"), ("nfl", "h1_winner"), ("nhl", "p1_total"), ("nhl", "p1_winner"), ("nhl", "spread"),
         ("nfl", "spread"), ("nba", "q1_winner"), ("nba", "h1_spread"), ("mlb", "f5_spread")]


def listing(client, series):
    """Every settled market of a series, archived and recent: ticker -> description."""
    found = {}
    lists = ((True, pages(client, "/historical/markets", "markets", series_ticker=series, limit=1000)),
             (False, pages(client, "/markets", "markets", series_ticker=series, status="settled", limit=1000,
                           min_close_ts=midnight(date(2026, 9, 1)), max_close_ts=midnight(date(2026, 11, 23)))))
    for archived, rows in lists:
        try:
            for m in rows:
                try:
                    outcome, settle = float(m["settlement_value_dollars"]), int(timestamp(m["settlement_ts"]))
                except (KeyError, TypeError, ValueError):
                    continue
                if outcome not in (0, 1):
                    continue
                tail = m["ticker"].rsplit("-", 1)[1]
                strike = m.get("floor_strike")
                found[m["ticker"]] = dict(
                    ticker=m["ticker"], key=m["event_ticker"].split("-", 1)[1], series=series, archived=archived,
                    settle=settle, close=int(timestamp(m["close_time"])) if m.get("close_time") else settle,
                    outcome=outcome, strike=None if strike is None else float(strike),
                    team=re.match(r"[A-Z]*", tail).group(0), value=m.get("expiration_value"),
                    title=m.get("title"), rules=m.get("rules_primary"))
        except RuntimeError:
            continue
    return found


def by_game(found):
    out = defaultdict(list)
    for m in found.values():
        if m["key"][:2] == "26":
            out[m["key"]].append(m)
    return out


def read_bars(group_of_game, plan=None):
    """ticker -> bars from the cache, or None when a read for the game is missing.
    plan: ticker -> the batch it was downloaded in, when that differs from this family's own batches."""
    out = {}
    groups = requests_for(group_of_game) if plan is None else list({id(plan[m["ticker"]]): plan[m["ticker"]] for m in group_of_game}.values())
    wanted = {m["ticker"]: m for m in group_of_game}
    for group in groups:
        file = cache_file(*request(group))
        if not file.exists():
            return None
        data = json.loads(file.read_text())
        if group[0]["archived"]:
            out[group[0]["ticker"]] = data.get("candlesticks") or []
        else:
            got = {x["market_ticker"]: x["candlesticks"] for x in data.get("markets", [])}
            for m in group:
                if m["ticker"] in wanted:
                    out[m["ticker"]] = [b for b in got.get(m["ticker"], []) if m["settle"]-WINDOW <= b["end_period_ts"] <= m["settle"]]
    return out


def get_download(args):
    deadline = time.time()+60*args.minutes
    with httpx.Client(timeout=40, headers={"User-Agent": "research-script"}) as client:
        jobs = []
        for sport, family in ORDER:
            if args.only and f"{sport}:{family}" not in args.only:
                continue
            games = by_game(listing(client, FAMILIES[sport][family][0]))
            recent = sorted(k for k, g in games.items() if not any(m["archived"] for m in g))
            random.Random(4).shuffle(recent)
            part = [requests_for(games[k]) for k in recent]
            print(sport, family, "games listed", len(games), "recent", len(recent), "requests", sum(map(len, part)), flush=True)
            jobs.append(part)
        # Interleave the families so a cut-off run still has every family over the whole date range.
        flat, i = [], 0
        while any(i < len(p) for p in jobs[:args.first]):
            flat += [p[i] for p in jobs[:args.first] if i < len(p)]
            i += 1
        for p in jobs[args.first:]:
            flat += p
        print("download", download(client, flat, deadline), flush=True)
        if args.archived:      # archived game totals: only the highest line the final score passed (one read a game)
            more = []
            for sport in ("nba", "mlb"):
                games = by_game(listing(client, FAMILIES[sport]["total"][0]))
                old = sorted(k for k, g in games.items() if all(m["archived"] for m in g))
                random.Random(5).shuffle(old)
                for k in old:
                    won = [m for m in games[k] if m["outcome"] == 1 and m["strike"] is not None]
                    if won:
                        more.append([[max(won, key=lambda m: m["strike"])]])
            random.Random(6).shuffle(more)
            print("archived top lines", len(more), download(client, more, deadline), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    d = sub.add_parser("download")
    d.add_argument("--minutes", type=float, default=38)
    d.add_argument("--first", type=int, default=8, help="how many leading families to interleave")
    d.add_argument("--only", nargs="*", help="sport:family pairs")
    d.add_argument("--archived", action="store_true")
    sub.add_parser("analyze")
    sub.add_parser("verify")
    sub.add_parser("tape")
    sub.add_parser("table")
    args = parser.parse_args()
    if args.command == "download":
        get_download(args)
    elif args.command == "table":
        from research.decided_report import table
        table()
    elif args.command == "tape":
        from research.decided_report import tape
        tape()
    elif args.command == "verify":
        from research.decided_report import verify_original
        verify_original()
    else:
        from research.decided_report import run
        run()


if __name__ == "__main__":
    main()
