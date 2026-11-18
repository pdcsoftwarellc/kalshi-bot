"""Does the late-game premium on NBA and MLB winner markets extend to spreads and totals?

Every strike market of a game is evaluated, but a rule takes at most ONE trade
per game per market type: the earliest qualifying minute (ties: lowest ask,
then ticker). Games are the unit for the 60/40 date split and the bootstrap.

Two tests:
  price  first minute a side's ask closes at or above a threshold (<= 99c) in the
         six hours before settlement, bought at the next minute's opening ask
         (near_end.entries). No game state.
  state  play-by-play joined as in game_state.py (a bar close may use plays
         stamped >= 15 s earlier). Entries only in the late period. The plain
         rule is "next-minute opening ask 97-99c"; a small grid of cushion rules
         is chosen on the earliest 60% of games and scored on the rest.

Download uses two workers and a pause after every network read. Recent markets
come through the batched candlestick endpoint; archived ones one at a time.
Research only: public reads, no orders.
"""
import argparse
from bisect import bisect_right
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from datetime import date
import hashlib
from itertools import product
import json
from pathlib import Path
import random
import re
from statistics import mean, median
import sys
import time

import httpx

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from research.api_cost import COLLECTION, ROOT, fetch, midnight, pages, save, timestamp
from research.game_state import FEED_DELAY, SPORTS, kalshi_games, plays, schedule
from research.near_end import WINDOW, cents, entries

OUT = ROOT / "data/processed/spreads-totals"
THRESHOLDS = (95, 97, 98)
PAUSE = .3                      # per worker after a network read: about 6 requests a second in total
# scope: which part of the game the market covers. late: periods in which entries are allowed.
TYPES = {
    "nba": {"spread": ("KXNBASPREAD", "game"), "total": ("KXNBATOTAL", "game"), "team_total": ("KXNBATEAMTOTAL", "game"),
            "half_spread": ("KXNBA1HSPREAD", "half"), "half_total": ("KXNBA1HTOTAL", "half")},
    "mlb": {"spread": ("KXMLBSPREAD", "game"), "total": ("KXMLBTOTAL", "game"), "team_total": ("KXMLBTEAMTOTAL", "game"),
            "f5_spread": ("KXMLBF5SPREAD", "half"), "f5_total": ("KXMLBF5TOTAL", "half")},
}
LATE = {("nba", "game"): lambda p: p >= 4, ("nba", "half"): lambda p: p == 2,
        ("mlb", "game"): lambda p: p >= 7, ("mlb", "half"): lambda p: p in (4, 5)}
RATE = {"total": 4.6, "team_total": 2.3, "half_total": 4.6}      # NBA points per minute, fixed in advance


def cache_file(path, params):
    return COLLECTION / (hashlib.sha256(json.dumps([path, params], sort_keys=True).encode()).hexdigest()+".json")


def markets(client, series):
    """Every settled market of a series, archived and recent, keyed by ticker."""
    found = {}
    lists = ((True, pages(client, "/historical/markets", "markets", series_ticker=series, limit=1000)),
             (False, pages(client, "/markets", "markets", series_ticker=series, status="settled", limit=1000,
                           min_close_ts=midnight(date(2026, 9, 1)), max_close_ts=midnight(date(2026, 11, 23)))))
    for archived, listing in lists:
        for m in listing:
            try:
                outcome, settle = float(m["settlement_value_dollars"]), int(timestamp(m["settlement_ts"]))
                strike = float(m["floor_strike"])
            except (KeyError, TypeError, ValueError):
                continue
            tail = re.fullmatch(r"([A-Z]*)(\d+)", m["ticker"].rsplit("-", 1)[1])
            if not tail:
                continue
            found[m["ticker"]] = dict(ticker=m["ticker"], key=m["event_ticker"].split("-", 1)[1], series=series,
                                      archived=archived, settle=settle, outcome=outcome, strike=strike, team=tail.group(1))
    return found


def batches(group):
    """Split one game's recent markets so a batched read stays under 10,000 bars."""
    group = sorted(group, key=lambda m: m["settle"])
    out, chunk = [], []
    for m in group:
        trial = chunk+[m]
        span = (trial[-1]["settle"]-trial[0]["settle"]+WINDOW)/60+2
        if len(trial) > 25 or len(trial)*span > 9500:
            out.append(chunk)
            chunk = [m]
        else:
            chunk = trial
    return out+[chunk] if chunk else out


def request(group):
    """(path, params) of the cached read that holds these markets' bars."""
    if group[0]["archived"]:
        m = group[0]
        return (f"/historical/markets/{m['ticker']}/candlesticks",
                dict(start_ts=m["settle"]-WINDOW, end_ts=m["settle"], period_interval=1))
    return ("/markets/candlesticks", dict(market_tickers=",".join(m["ticker"] for m in group),
            start_ts=group[0]["settle"]-WINDOW, end_ts=group[-1]["settle"], period_interval=1))


def requests_for(game_markets):
    live = [m for m in game_markets if not m["archived"]]
    return [[m] for m in game_markets if m["archived"]]+batches(live)


def download(client, jobs, deadline):
    """jobs: lists of market groups per game, in priority order. Stops starting games at the deadline."""
    done = Counter()

    def read(group):
        path, params = request(group)
        if cache_file(path, params).exists():
            return True
        began = time.time()
        try:
            fetch(client, path, params)       # backs off on 429 and 5xx
            ok = True
        except RuntimeError:
            ok = False
        time.sleep(max(PAUSE-(time.time()-began), 0)+(2 if not ok else 0))
        done["read" if ok else "failed"] += 1
        return ok
    with ThreadPoolExecutor(2) as pool:
        for n, groups in enumerate(jobs):
            if time.time() > deadline:
                done["games_not_reached"] = len(jobs)-n
                break
            list(pool.map(read, groups))
            if n % 50 == 0:
                print("  download", n, len(jobs), dict(done), flush=True)
    return dict(done)


def bars_of(game_markets, plan=None):
    """ticker -> bars from the cache, or None if any read for the game is missing."""
    out = {}
    groups = requests_for(game_markets) if plan is None else {id(plan[m["ticker"]]): plan[m["ticker"]] for m in game_markets}.values()
    wanted = {m["ticker"] for m in game_markets}
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
                    out[m["ticker"]] = [b for b in got.get(m["ticker"], [])
                                        if m["settle"]-WINDOW <= b["end_period_ts"] <= m["settle"]]
    return out


def link(client, sport):
    """Winner-market games joined to the play-by-play archive: key -> game with home, away and plays."""
    cfg, out = SPORTS[sport], {}
    games = list(kalshi_games(client, cfg["series"]))
    days = sorted({g["day"] for g in games})
    with ThreadPoolExecutor(4) as pool:                 # play-by-play archive, not Kalshi
        slates = dict(zip(days, pool.map(lambda d: schedule(client, sport, d), days)))
    for g in games:
        match = [e for e in slates[g["day"]] if set(e["teams"].values()) == g["teams"]
                 and e["status"] == "STATUS_FINAL" and e["start"] < g["settle"]]
        if match:
            e = max(match, key=lambda e: e["start"])
            out[g["event"].split("-", 1)[1]] = dict(g, espn=e["id"], home=e["teams"]["home"], away=e["teams"]["away"])
    return out


def value(kind, m, game, home, away):
    """The quantity the market settles on, from the current score."""
    if "spread" in kind:
        return (home-away) if m["team"] == game["home"] else (away-home)
    if kind == "team_total":
        return home if m["team"] == game["home"] else away
    return home+away


def candidates(sport, kind, scope, game, log, group, bars):
    """Late-period minutes where either side's next-minute opening ask is 95-99c."""
    rows, times = [], [p[0] for p in log]
    late = LATE[sport, scope]
    for m in group:
        if "spread" in kind or kind == "team_total":
            if m["team"] not in (game["home"], game["away"]):
                continue
        series = sorted(bars.get(m["ticker"]) or [], key=lambda b: b["end_period_ts"])
        for bar, nxt in zip(series, series[1:]):
            i = bisect_right(times, bar["end_period_ts"]-FEED_DELAY)-1
            if i < 0 or nxt["end_period_ts"]-bar["end_period_ts"] > 120 or bar["end_period_ts"] > times[-1]:
                continue
            _, period, left, home, away = log[i]
            if not late(period):
                continue
            bid, offer = cents(nxt, "yes_bid", "open"), cents(nxt, "yes_ask", "open")
            if bid is None or offer is None or not 0 < bid <= offer < 100:
                continue
            v = value(kind, m, game, home, away)
            if sport == "nba":
                seconds = left-1440 if scope == "half" else left      # regulation seconds, or seconds of the overtime
                minutes = max(seconds, 0)/60
            else:
                minutes = None
            for yes, ask in ((True, offer), (False, 100-bid)):
                if not 95 <= ask <= 99:
                    continue
                ahead = (v-m["strike"]) if yes else (m["strike"]-v)      # how far this side is in front of the line
                payout = m["outcome"] if yes else 1-m["outcome"]
                rows.append(dict(key=game["key"], ticker=m["ticker"], ts=nxt["end_period_ts"], side="yes" if yes else "no",
                                 ask=ask, width=offer-bid, period=period, left=left, minutes=minutes, ahead=ahead, tied=home == away,
                                 lead=abs(home-away), decided="spread" not in kind and yes and v > m["strike"],
                                 payout=payout, net=100*(payout-ask/100-.07*ask/100*(1-ask/100)),
                                 vol=float(nxt.get("volume_fp") or nxt.get("volume") or 0),
                                 seen_vol=float(bar.get("volume_fp") or bar.get("volume") or 0)))
    rows.sort(key=lambda r: (r["ts"], r["ask"], r["ticker"]))
    return rows


def grid(sport, kind):
    """Rules as (name, test). Small, fixed before looking at results."""
    rules = []
    bands = ((95, 99), (97, 99), (98, 99))
    if sport == "nba":
        windows = ((None, "any time"), (6, "<=6 min"), (3, "<=3 min"))
        def timed(r, w):
            return w is None or r["period"] > 4 or r["minutes"] <= w
        if "spread" in kind:
            for (lo, hi), (w, wname), z in product(bands, windows, (0, 2, 3, 4, 5)):
                rules.append((f"ask {lo}-{hi}, {wname}, ahead of line >= {z}*sqrt(min left)",
                              lambda r, lo=lo, hi=hi, w=w, z=z: lo <= r["ask"] <= hi and timed(r, w)
                              and r["ahead"] >= z*(r["minutes"]+.5)**.5))
        else:
            rate = RATE[kind]
            for (lo, hi), (w, wname), side, q in product(bands, windows, ("both", "over", "under"), (0, 1.5, 2, 3)):
                def test(r, lo=lo, hi=hi, w=w, side=side, q=q):
                    if not (lo <= r["ask"] <= hi and timed(r, w)):
                        return False
                    if r["side"] == "yes":          # over: points still needed are few for the time left
                        return side != "under" and (r["decided"] or q == 0 or rate*r["minutes"] >= q*(-r["ahead"]))
                    return side != "over" and (q == 0 or r["ahead"] >= q*(rate*r["minutes"]+1))
                rules.append((f"ask {lo}-{hi}, {wname}, {side}, room >= {q}x expected scoring", test))
    else:
        innings = (7, 8, 9) if kind in ("spread", "total", "team_total") else (4, 5)
        if "spread" in kind:
            for (lo, hi), inn, z in product(bands, innings, (0, .34, .5, 1)):
                rules.append((f"ask {lo}-{hi}, inning >= {inn}, ahead of line >= {z} per trailing out left",
                              lambda r, lo=lo, hi=hi, inn=inn, z=z: lo <= r["ask"] <= hi and r["period"] >= inn
                              and (z == 0 or (not r["tied"] and r["ahead"] >= z*max(r["left"], 1)))))
        else:
            last = 9 if kind in ("total", "team_total") else 5
            for (lo, hi), inn, side, q in product(bands, innings, ("both", "over", "under"), (0, 1, 2, 3)):
                def test(r, lo=lo, hi=hi, inn=inn, side=side, q=q):
                    if not (lo <= r["ask"] <= hi and r["period"] >= inn):
                        return False
                    if r["side"] == "yes":
                        return side != "under" and (q == 0 or r["decided"])
                    return side != "over" and (q == 0 or r["ahead"] >= q*(last+1-min(r["period"], last)))
                rules.append((f"ask {lo}-{hi}, inning >= {inn}, {side}, under room >= {q} runs per inning left "
                              f"(over: already decided)", test))
    return rules


def first(rows_by_game, keys, test):
    """Earliest qualifying row of each game."""
    out = []
    for k in keys:
        for r in rows_by_game.get(k, ()):
            if test(r):
                out.append(r)
                break
    return out


def score(hits, games, boot=False):
    if not hits:
        return dict(games=len(games), trades=0)
    price = mean(h["ask"] for h in hits)/100
    row = dict(games=len(games), trades=len(hits), losses=sum(h["payout"] < .5 for h in hits),
               win_rate=mean(h["payout"] for h in hits), needed=price+.07*price*(1-price), mean_ask=100*price,
               net_cents=mean(h["net"] for h in hits), per_game_cents=sum(h["net"] for h in hits)/len(games),
               median_minute_volume=median(h["vol"] for h in hits))
    if boot and len(hits) >= 10:
        by_game = defaultdict(list)
        for h in hits:
            by_game[h["key"]].append(h["net"])
        rng, nets, means = random.Random(7), [by_game.get(k, []) for k in games], []
        for _ in range(2000):
            pick = [n for g in rng.choices(nets, k=len(nets)) for n in g]
            if pick:
                means.append(mean(pick))
        means.sort()
        row["ci95"] = [means[int(.025*len(means))], means[int(.975*len(means))-1]]
    return row


def every(rows_by_game, keys, test):
    """Earliest qualifying row of each market side: several trades a game."""
    out = []
    for k in keys:
        seen = set()
        for r in rows_by_game.get(k, ()):
            if (r["ticker"], r["side"]) not in seen and test(r):
                seen.add((r["ticker"], r["side"]))
                out.append(r)
    return out


def two_parts(rows_by_game, order, test, pick=first):
    cut = int(len(order)*.6)
    early, late = order[:cut], order[cut:]
    a, b = pick(rows_by_game, early, test), pick(rows_by_game, late, test)
    return dict(early=score(a, early), later=score(b, late, boot=True)), a+b


def winner_trades(sport):
    """The established winner-market trade of each game: key -> won."""
    rows = json.loads((ROOT/f"data/processed/game-state/{sport}-minutes.json").read_text())
    out = {}
    for r in sorted(rows, key=lambda r: r["ts"]):
        key = r["event"].split("-", 1)[1]
        if key in out:
            continue
        if sport == "nba":
            ok = 97 <= r["ask"] <= 99
        else:
            ok = 90 <= r["ask"] <= 99 and r["lead"] >= 3 and r["lead"] >= max(r["left"], 1)
        if ok:
            out[key] = dict(won=bool(r["won"]), ts=r["ts"], net=r["net"])
    return out


def together(hits, winner):
    """Do the new trade and the winner-market trade of the same game win and lose together?"""
    both = [(h["payout"] >= .5, winner[h["key"]]["won"]) for h in hits if h["key"] in winner]
    c = Counter(both)
    n, a, b = len(both), sum(x for x, _ in both), sum(y for _, y in both)
    var = a*(n-a)*b*(n-b)
    phi = (n*c[True, True]-a*b)/var**.5 if var else None
    nets = [(h["net"], winner[h["key"]]["net"]) for h in hits if h["key"] in winner]
    return dict(new_trades=len(hits), games_with_both=n, both_win=c[True, True], new_loses_winner_wins=c[False, True],
                new_wins_winner_loses=c[True, False], both_lose=c[False, False], phi=phi,
                new_trades_in_games_without_winner_trade=len(hits)-n,
                net_in_games_without_winner_trade=(mean(h["net"] for h in hits if h["key"] not in winner)
                                                   if len(hits) > n else None))


def run(args):
    sport = args.sport
    report = dict(sport=sport, types={})
    with httpx.Client(timeout=40, headers={"User-Agent": "research-script"}) as client:
        linked = link(client, sport)
        wanted = {}
        for kind in args.types:
            series, scope = TYPES[sport][kind]
            by_game = defaultdict(list)
            for m in markets(client, series).values():
                if m["key"][:2] == "26":
                    by_game[m["key"]].append(m)
            wanted[kind] = by_game
            print(kind, series, "games listed", len(by_game), "markets", sum(map(len, by_game.values())), flush=True)
        # Games to cover: every recent game, and a seeded random sample of archived games (uniform over dates).
        keys = sorted({k for by_game in wanted.values() for k in by_game})
        recent = [k for k in keys if all(not m["archived"] for kind in wanted for m in wanted[kind].get(k, []))]
        archived = sorted(set(keys)-set(recent))
        random.Random(3).shuffle(archived)
        random.Random(4).shuffle(recent)                # a cut-off download still covers the whole date range
        archive_types = [k for k in args.types if k in args.archive_types]
        plan = None
        if args.bundle:           # one batched read per recent game across market types (fewer requests)
            plan = {}
            for k in keys:
                for g in requests_for([m for kind in args.types for m in wanted[kind].get(k, [])]):
                    for m in g:
                        plan[m["ticker"]] = g
        if args.minutes:
            deadline = time.time()+60*args.minutes
            if plan is None:
                jobs = [[g for kind in args.types for g in requests_for(wanted[kind].get(k, []))] for k in recent]
            else:
                jobs = [list({id(plan[m["ticker"]]): plan[m["ticker"]] for kind in args.types
                              for m in wanted[kind].get(k, [])}.values()) for k in recent]
            jobs += [[g for kind in archive_types for g in requests_for(wanted[kind].get(k, []))] for k in archived]
            print("download", download(client, jobs, deadline), flush=True)
        winner = winner_trades(sport)
        for kind in args.types:
            series, scope = TYPES[sport][kind]
            games, price_rows, state_rows, skip = {}, defaultdict(list), {}, Counter()
            for k, group in wanted[kind].items():
                bars = bars_of(group, plan)
                if bars is None:
                    skip["not_downloaded"] += 1
                    continue
                if not any(bars.values()):
                    skip["no_bars"] += 1
                    continue
                games[k] = min(m["settle"] for m in group)
                for m in group:
                    for e in entries(bars[m["ticker"]], m["settle"], m["outcome"]):
                        if e["threshold"] in THRESHOLDS:
                            price_rows[k].append(dict(key=k, ticker=m["ticker"], ts=e["ts"], ask=e["price"], side=e["side"],
                                                      threshold=e["threshold"], payout=e["payout"], net=e["net_cents"],
                                                      vol=e["minute_volume"]))
                game = linked.get(k)
                if not game:
                    skip["no_play_by_play_link"] += 1
                    continue
                log = plays(client, sport, game["espn"])
                if len(log) < 50:
                    skip["no_plays"] += 1
                    continue
                state_rows[k] = candidates(sport, kind, scope, dict(game, key=k), log, group, bars)
            for rows in price_rows.values():
                rows.sort(key=lambda r: (r["ts"], r["ask"], r["ticker"]))
            order = sorted(games, key=games.get)
            out = dict(series=series, markets_listed=sum(map(len, wanted[kind].values())), games_listed=len(wanted[kind]),
                       games_with_bars=len(order), markets_with_bars=sum(len(wanted[kind][k]) for k in order),
                       skipped=dict(skip), price={}, state={})
            if not order:
                report["types"][kind] = out
                continue
            day = lambda k: time.strftime("%Y-%m-%d", time.gmtime(games[k]))
            out["dates"] = [day(order[0]), day(order[-1])]
            out["split_date"] = day(order[int(len(order)*.6)])
            out["recent_games"] = sum(k in set(recent) for k in order)
            for t in THRESHOLDS:
                out["price"][f"ask >= {t}c"], _ = two_parts(price_rows, order, lambda r, t=t: r["threshold"] == t)
            sorder = [k for k in order if k in state_rows]
            out["state_games"] = len(sorder)
            if sorder:
                out["state_split_date"] = day(sorder[int(len(sorder)*.6)])
                plain = lambda r: 97 <= r["ask"] <= 99
                out["state"]["plain: ask 97-99c in the late period"], hits = two_parts(state_rows, sorder, plain)
                out["with_winner_trade"] = together(hits, winner)
                traded = lambda r: plain(r) and r["seen_vol"] > 0
                out["state"]["plain, and the market traded in the decision minute"], hits3 = two_parts(state_rows, sorder, traded)
                out["with_winner_trade_traded_minute"] = together(hits3, winner)
                out["state"]["plain, EVERY strike once (several trades a game, clustered by game)"], _ = two_parts(
                    state_rows, sorder, plain, every)
                out["state"]["plain + traded minute, EVERY strike once"], _ = two_parts(state_rows, sorder, traded, every)
                # A wide book makes "100 minus the bid" look like a near-certain price when it is only an absent bid.
                narrow = lambda r: plain(r) and r["width"] <= 2
                out["state"]["plain, and bid-ask spread <= 2c"], hits4 = two_parts(state_rows, sorder, narrow)
                out["with_winner_trade_narrow"] = together(hits4, winner)
                out["state"]["plain + spread <= 2c, EVERY strike once"], _ = two_parts(state_rows, sorder, narrow, every)
                out["state"]["plain + spread <= 2c + traded minute"], _ = two_parts(
                    state_rows, sorder, lambda r: narrow(r) and r["seen_vol"] > 0)
                if "spread" not in kind:
                    name = "already decided over (score past the line), ask 95-99c"
                    out["state"][name], _ = two_parts(state_rows, sorder, lambda r: r["decided"])
                    out["state"]["plain, excluding already-decided overs"], hits2 = two_parts(
                        state_rows, sorder, lambda r: plain(r) and not r["decided"])
                    out["with_winner_trade_excluding_decided"] = together(hits2, winner)
                rules = grid(sport, kind)
                early = sorder[:int(len(sorder)*.6)]
                ranked = []
                for name, test in rules:
                    hits = first(state_rows, early, test)
                    if len(hits) >= args.min_trades:
                        ranked.append((sum(h["net"] for h in hits), name, test))
                ranked.sort(key=lambda x: -x[0])
                out["rules_tried"] = len(rules)
                for rank, (_, name, test) in enumerate(ranked[:3]):
                    out["state"][f"selected #{rank+1}: {name}"], _ = two_parts(state_rows, sorder, test)
            report["types"][kind] = out
    save(OUT/f"{sport}-{args.name}.json", report)
    show(report)


def show(report):
    def line(name, parts):
        print(f"  {name}")
        for tag in ("early", "later"):
            s = parts[tag]
            if not s["trades"]:
                print(f"    {tag:5} games {s['games']:4} no trades")
                continue
            ci = s.get("ci95") and [round(v, 2) for v in s["ci95"]]
            print(f"    {tag:5} games {s['games']:4} trades {s['trades']:4} losses {s['losses']:3} win {s['win_rate']:.4f} "
                  f"need {s['needed']:.4f} ask {s['mean_ask']:.1f} net {s['net_cents']:+.2f}c ci {ci} "
                  f"per game {s['per_game_cents']:+.2f}c vol {s['median_minute_volume']:.0f}")
    for kind, out in report["types"].items():
        print(f"\n{report['sport'].upper()} {kind} ({out['series']}): games {out['games_with_bars']} of {out['games_listed']} listed, "
              f"markets {out['markets_with_bars']}, dates {out.get('dates')}, split {out.get('split_date')}, "
              f"recent-season games {out.get('recent_games')}, skipped {out['skipped']}")
        print(" PRICE ONLY (six hours before settlement, first crossing)")
        for name, parts in out["price"].items():
            line(name, parts)
        print(f" LATE PERIOD WITH GAME STATE: games {out.get('state_games')}, split {out.get('state_split_date')}, "
              f"rules tried {out.get('rules_tried')}")
        for name, parts in out["state"].items():
            line(name, parts)
        for k in ("with_winner_trade", "with_winner_trade_traded_minute", "with_winner_trade_narrow",
                  "with_winner_trade_excluding_decided"):
            if k in out:
                print(f"  {k}: {out[k]}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("sport", choices=tuple(TYPES))
    parser.add_argument("--types", nargs="+", default=["spread", "total"])
    parser.add_argument("--archive-types", nargs="+", default=["spread", "total"],
                        help="market types to download for archived games (others: recent games only)")
    parser.add_argument("--minutes", type=float, default=0, help="download budget; 0 analyses the cache only")
    parser.add_argument("--bundle", action="store_true", help="batch a game's market types into shared reads")
    parser.add_argument("--min-trades", type=int, default=30)
    parser.add_argument("--name", default="summary")
    run(parser.parse_args())


if __name__ == "__main__":
    main()
