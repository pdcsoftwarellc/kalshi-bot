"""Analysis half of decided.py: reads cached bars and play-by-play only (see decided.py for the rules)."""
from bisect import bisect_left, bisect_right
from collections import Counter, defaultdict
import json
from pathlib import Path
from statistics import mean, median
import sys
import time

import httpx

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from research.api_cost import fetch, save
from research.decided import FAMILIES, OUT, by_game, listing, read_bars
from research.game_state import FEED_DELAY, SPORTS, plays
from research.near_end import WINDOW, cents
from research.spreads_totals import link, requests_for


# (highest ask accepted, seconds the deciding play must be old at the bar close). 15 s is the house rule;
# 120 s asks whether the offer outlives the minute in which the play happened.
RULES = ((99, 15), (98, 15), (99, 120), (98, 120))


def number(text):
    try:
        return float(text)
    except (TypeError, ValueError):
        return None


def quantity(family, m, game, home, away):
    """The count a market settles on, from a (home, away) score. None if the team code is unknown."""
    if family.endswith("total") and "team" not in family:
        return home+away
    if m["team"] == game["home"]:
        mine, other = home, away
    elif m["team"] == game["away"]:
        mine, other = away, home
    elif m["team"] == "TIE":
        return 1 if home == away else 0
    else:
        return None
    if "team_total" in family:
        return mine
    return mine-other                      # spread and winner: this team's margin


def finished(sport, log):
    """The archive's last play is a real end of game: a leader, and no time or outs left."""
    _, period, left, home, away = log[-1]
    if home == away:
        return False
    if sport == "mlb":
        return period >= 9 or len(log) > 150           # rain-shortened games still end with a leader
    regulation = {"nba": 4, "nhl": 3, "nfl": 4}[sport]
    return period > regulation or (period == regulation and left <= 1)


def official_scores(found):
    """Game key -> what Kalshi settled on, from a listing's expiration values (a number, or 'Home: x, Away: y')."""
    out = {}
    for m in found.values():
        text = str(m["value"] or "")
        if text.startswith("Home:"):
            try:
                out[m["key"]] = tuple(float(x.split(":")[1]) for x in text.split(","))
            except (IndexError, ValueError):
                pass
        elif number(text) is not None:                  # overs that expire early carry the count at that moment
            out[m["key"]] = max(out.get(m["key"], 0), number(text))
    return out


def decision(family, scope, m, game, log):
    """(index of the deciding play, kind, side, flags) or None. Uses only plays up to that index,
    except that a period's last play is recognised by the next play belonging to a later period."""
    if scope == "game":
        last = len(log)-1
        first = 0
        ended = finished(game["sport"], log)
    else:
        inside = [i for i, p in enumerate(log) if p[1] <= scope]
        if not inside or inside[-1] == len(log)-1:
            return None                    # the period never completed on the record
        last = inside[-1]
        first = 0
        ended = True
    is_total = "total" in family
    values = [quantity(family, m, game, p[3], p[4]) for p in log[:last+1]]
    if values[-1] is None:
        return None
    if is_total:
        if m["strike"] is None:
            return None
        for i in range(first, last+1):
            if values[i] > m["strike"]:
                back = min(values[i:]) <= m["strike"]          # the archive later took the count back
                return i, "over", "yes", dict(value=values[i], end_value=values[-1], reverted=back)
        if not ended:
            return None
        return last, ("final" if scope == "game" else "period"), "no", dict(value=values[-1], end_value=values[-1], reverted=False)
    if not ended:
        return None
    if "winner" in family or family == "winner":
        won = values[-1] > 0 if m["team"] != "TIE" else values[-1] == 1
    else:
        if m["strike"] is None:
            return None
        won = values[-1] > m["strike"]
    return last, ("final" if scope == "game" else "period"), "yes" if won else "no", dict(value=values[-1], end_value=values[-1], reverted=False)


def scan(sport, family, scope, game, log, group, bars):
    """One record per market that had at least one quoted minute after its decision."""
    times, out = [p[0] for p in log], []
    for m in group:
        d = decision(family, scope, m, game, log)
        if not d:
            continue
        i, kind, side, flags = d
        when = log[i][0]
        series = sorted(bars.get(m["ticker"]) or [], key=lambda b: b["end_period_ts"])
        minutes = []                       # (bar close, next-open ask of the decided side, next-minute volume)
        for bar, nxt in zip(series, series[1:]):
            if bar["end_period_ts"]-FEED_DELAY < when or nxt["end_period_ts"]-bar["end_period_ts"] > 120:
                continue
            if bar["end_period_ts"] > m["close"]:
                continue
            bid, offer = cents(nxt, "yes_bid", "open"), cents(nxt, "yes_ask", "open")
            if bid is None or offer is None:
                continue
            ask = offer if side == "yes" else 100-bid
            minutes.append((bar["end_period_ts"], ask, float(nxt.get("volume_fp") or nxt.get("volume") or 0)))
        if not minutes:
            continue
        payout = m["outcome"] if side == "yes" else 1-m["outcome"]
        official = number(m["value"])
        rec = dict(sport=sport, family=family, kind=kind, key=game["key"], ticker=m["ticker"], side=side, strike=m["strike"],
                   decided_ts=when, period=log[i][1], left=log[i][2], value=flags["value"], end_value=flags["end_value"],
                   official=m["value"], reverted=flags["reverted"], payout=payout, settle=m["settle"], close=m["close"],
                   decided_minutes=len(minutes), open_after=(minutes[-1][0]-when)/60,
                   archive_matches_official=None if official is None or "total" not in family else official == flags["end_value"])
        rec["game_ok"] = game.get("ok")
        rec["verified"] = payout == 1 and not flags["reverted"] and game.get("ok") is not False
        if payout != 1 and family == "winner":
            rec["why_lost"] = "the raw archive's final leader is not the team Kalshi settled on (archive score error)"
        elif payout != 1:
            rec["why_lost"] = ("archive scoring plays do not add up to the official final" if game.get("ok") is False
                               else "archive count was later taken back" if flags["reverted"] else "unexplained")
        offered = [x for x in minutes if 0 < x[1] <= 99]
        rec["offered_minutes"] = len(offered)
        rec["offered_volume"] = sum(x[2] for x in offered)
        rec["min_ask"] = min((x[1] for x in offered), default=None)
        for cap, wait in RULES:
            hit = next((n for n, x in enumerate(minutes) if 0 < x[1] <= cap and x[0]-when >= wait), None)
            if hit is None:
                continue
            ts, ask, vol = minutes[hit]
            run = 1
            while hit+run < len(minutes) and 0 < minutes[hit+run][1] <= cap and minutes[hit+run][0]-minutes[hit+run-1][0] <= 60:
                run += 1
            rec[f"entry{cap}_{wait}"] = dict(ts=ts, ask=ask, vol=vol, lag=ts-when, persist=run,
                                             net=100*(payout-ask/100-.07*ask/100*(1-ask/100)))
        out.append(rec)
    return out


def winner_markets(client, sport, linked):
    """The game-winner market of each linked game, shaped like a listing row, with its cached bars."""
    series = SPORTS[sport]["series"]
    for key, g in linked.items():
        path = ("/historical/markets/{t}/candlesticks" if g["archived"] else "/series/"+series+"/markets/{t}/candlesticks").format(t=g["ticker"])
        params = dict(start_ts=g["settle"]-WINDOW, end_ts=g["settle"], period_interval=1)
        from research.spreads_totals import cache_file
        if not cache_file(path, params).exists():
            continue
        bars = fetch(client, path, params).get("candlesticks") or []
        m = dict(ticker=g["ticker"], key=key, settle=g["settle"], close=g["settle"], outcome=g["outcome"], strike=None,
                 team=g["yes"], value=None, archived=g["archived"])
        yield key, m, bars


def summarise(records, games, cap, wait):
    """games: key -> settle for every game examined in this family. One entry per market; also per game."""
    order = sorted(games, key=games.get)
    cut = int(len(order)*.6)
    out = {}
    for tag, part in (("early60", order[:cut]), ("late40", order[cut:]), ("all", order)):
        keys = set(part)
        recs = [r for r in records if r["key"] in keys]
        hits = [r for r in recs if f"entry{cap}_{wait}" in r]
        row = dict(games=len(part), decided_markets=len(recs), markets_offered=len(hits), trades=len(hits))
        if part:
            weeks = max((games[part[-1]]-games[part[0]])/604800, 1/7)
            row["weeks"] = weeks
            row["dates"] = [time.strftime("%Y-%m-%d", time.gmtime(games[part[0]])), time.strftime("%Y-%m-%d", time.gmtime(games[part[-1]]))]
        if hits:
            e = [r[f"entry{cap}_{wait}"] for r in hits]
            per_game = defaultdict(list)
            for r in hits:
                per_game[r["key"]].append(r)
            firsts = [min(v, key=lambda r: r[f"entry{cap}_{wait}"]["ts"])[f"entry{cap}_{wait}"] for v in per_game.values()]
            asks = Counter("<=95" if x["ask"] <= 95 else "96-97" if x["ask"] <= 97 else str(x["ask"]) for x in e)
            row.update(
                share_of_decided_markets_offered=len(hits)/len(recs), losses=sum(r["payout"] < .5 for r in hits),
                unverified=sum(not r["verified"] for r in hits), mean_ask=mean(x["ask"] for x in e), asks=dict(asks),
                net_cents=mean(x["net"] for x in e),
                net_cents_verified=(mean(r[f"entry{cap}_{wait}"]["net"] for r in hits if r["verified"]) if any(r["verified"] for r in hits) else None),
                trades_per_week=len(hits)/weeks, games_with_trade=len(per_game), games_with_trade_per_week=len(per_game)/weeks,
                per_game_first_net_cents=mean(x["net"] for x in firsts), per_game_losses=sum(x["net"] < 0 for x in firsts),
                median_entry_minute_volume=median(x["vol"] for x in e), mean_entry_minute_volume=mean(x["vol"] for x in e),
                share_entry_minute_traded=mean(x["vol"] > 0 for x in e),
                median_persist_minutes=median(x["persist"] for x in e), share_persist_3plus=mean(x["persist"] >= 3 for x in e),
                median_lag_seconds=median(x["lag"] for x in e),
                median_offered_minutes=median(r["offered_minutes"] for r in hits),
                median_offered_volume=median(r["offered_volume"] for r in hits),
                dollars_per_week_at_entry_minute_volume=sum(x["net"]/100*x["vol"] for x in e)/weeks)
        out[tag] = row
    return out


def run():
    from concurrent.futures import ThreadPoolExecutor
    report, trades, rules, cover = {}, [], {}, {}
    with httpx.Client(timeout=40, headers={"User-Agent": "research-script"}) as client:
        for sport in FAMILIES:
            linked = link(client, sport)
            loaded = {}
            plan = {}                       # spreads_totals.py --bundle shared one batched read between spreads and totals
            both = [by_game(listing(client, FAMILIES[sport][f][0])) for f in ("spread", "total")]
            for k in set(both[0]) | set(both[1]):
                for g in requests_for([m for part in both for m in part.get(k, [])]):
                    for m in g:
                        plan[m["ticker"]] = g
            for family, (series, scope) in FAMILIES[sport].items():       # cache only: which games have bars
                games_all, skip, have = by_game(listing(client, series)), Counter(), {}
                for key, group in games_all.items():
                    if not any(m["archived"] for m in group):
                        bars = read_bars(group)
                        if bars is None and family in ("spread", "total"):
                            bars = read_bars(group, plan)
                    else:                  # archived: whatever single markets were downloaded
                        bars = {}
                        for m in group:
                            if m["archived"]:
                                bars.update(read_bars([m]) or {})
                        bars = bars or None
                    if bars is None:
                        skip["not_downloaded"] += 1
                    elif not any(bars.values()):
                        skip["no_bars"] += 1
                    elif key not in linked:
                        skip["no_play_by_play_link"] += 1
                    else:
                        have[key] = (group, bars)
                loaded[family] = (series, scope, len(games_all), skip, have)
                rules.setdefault(f"{sport}:{family}", next((m["rules"] for g in games_all.values() for m in g if m.get("rules")), None))
            wanted = sorted({k for f in loaded.values() for k in f[4]})
            with ThreadPoolExecutor(6) as pool:        # play-by-play archive, not Kalshi
                data = dict(zip(wanted, pool.map(lambda k: scoring(client, sport, linked[k]["espn"]), wanted)))
            logs = {k: safe_log(sport, d) for k, d in data.items() if d and len(d["plays"]) >= 50}
            # A game whose last play is stamped far from the moment Kalshi settled the winner has unusable stamps
            # (the archive's clock is an hour off on the day daylight saving ends) or is the wrong game.
            gaps = {k: linked[k]["settle"]-v[0][-1][0] for k, v in logs.items()}
            def jumpy(key):                # 1 Nov 2026, the day US clocks went back: some games are stamped an hour early
                spans = {}                 # for part of the game, which shows as a period lasting far too long
                for row in data[key]["plays"]:
                    lo, hi = spans.get(row[1], (row[0], row[0]))
                    spans[row[1]] = (min(lo, row[0]), max(hi, row[0]))
                return key.startswith("26NOV01") or (sport != "mlb" and max(hi-lo for lo, hi in spans.values()) > 4800)
            off = [k for k, g in gaps.items() if not CLOCK[0] <= g <= CLOCK[1] or jumpy(k)]
            for k in off:
                del logs[k]
            old = {k: plays(client, sport, linked[k]["espn"]) for k in logs}
            differs = sum(1 for k in logs if len(old[k]) >= 50 and
                          old[k][-1][3]+old[k][-1][4] != data[k]["final"]["home"]+data[k]["final"]["away"])
            cover[f"{sport}:archive"] = dict(games=len(logs), dropped_for_clock=len(off), median_seconds_last_play_to_settlement=median(gaps.values()) if gaps else None,
                                             scoring_plays_do_not_add_up_to_official_final=sum(not v[1] for v in logs.values()),
                                             raw_running_score_ends_off_the_official_final=differs)
            print(sport, cover[f"{sport}:archive"], flush=True)
            for family, (series, scope, listed, skip, have) in loaded.items():
                games, records = {}, []
                for key, (group, bars) in have.items():
                    if key not in logs:
                        skip["no_plays"] += 1
                        continue
                    log, sound = logs[key]
                    games[key] = min(m["settle"] for m in group)
                    records += scan(sport, family, scope, dict(linked[key], key=key, sport=sport, ok=sound), log,
                                    [m for m in group if bars.get(m["ticker"])], bars)
                cover[f"{sport}:{family}"] = dict(series=series, games_listed=listed, games_used=len(games), skipped=dict(skip))
                for kind in sorted({r["kind"] for r in records}):
                    part = [r for r in records if r["kind"] == kind]
                    report[f"{sport}:{family}:{kind}"] = {f"ask<={cap} wait>={wait}s": summarise(part, games, cap, wait) for cap, wait in RULES}
                trades += [r for r in records if "entry99_15" in r]
                print(sport, family, cover[f"{sport}:{family}"], "decided markets", len(records), flush=True)
            # game winner after the last play: every game, on the raw archive (only its last stamp and leader are used)
            games, records = {}, []
            for key, m, bars in winner_markets(client, sport, linked):
                log = plays(client, sport, linked[key]["espn"])
                if len(log) < 50 or not bars or not CLOCK[0] <= m["settle"]-log[-1][0] <= CLOCK[1] or key.startswith("26NOV01"):
                    continue
                games[key] = m["settle"]
                records += scan(sport, "winner", "game", dict(linked[key], key=key, sport=sport, ok=None), log, [m], {m["ticker"]: bars})
            cover[f"{sport}:winner"] = dict(series=SPORTS[sport]["series"], games_used=len(games))
            report[f"{sport}:winner:final"] = {f"ask<={cap} wait>={wait}s": summarise(records, games, cap, wait) for cap, wait in RULES}
            trades += [r for r in records if "entry99_15" in r]
            print(sport, "winner", len(games), len(records), flush=True)
    save(OUT/"summary.json", dict(coverage=cover, rules=rules, families=report))
    save(OUT/"trades.json", trades)
    show(report)


CLOCK = (0, 1500)    # seconds from the archive's last play to Kalshi's settlement of the winner market
NFL_SHIFT = 90       # a touchdown row already carries the extra point, which is kicked up to a minute or so later
PERIODS = {"nba": 2, "nfl": 2, "mlb": 5, "nhl": 1}


def scoring(client, sport, game):
    """Official final and period scores and every play of a game, from the play-by-play archive (cached, trimmed).
    Rows keep the archive's own order: [wallclock, period, left, scoring play?, score value, home, away]."""
    from research.game_state import ARCHIVE, RAW
    from research.api_cost import timestamp
    file = RAW/"decided"/f"plays-{sport}-{game}.json"
    if file.exists():
        return json.loads(file.read_text())
    for attempt in range(4):
        try:
            r = client.get(ARCHIVE+SPORTS[sport]["path"]+"/summary", params=dict(event=game))
            if r.status_code == 200:
                break
        except httpx.RequestError:
            pass
        time.sleep(1+attempt)
    else:
        return None
    j = r.json()
    try:
        sides = {c["homeAway"]: c for c in j["header"]["competitions"][0]["competitors"]}
        final = {k: int(sides[k]["score"]) for k in ("home", "away")}
        lines = {k: [int(float(x.get("displayValue") or 0)) for x in sides[k].get("linescores") or []] for k in ("home", "away")}
    except (KeyError, TypeError, ValueError):
        return None
    raw = j.get("plays") or [p for d in (j.get("drives") or {}).get("previous", []) for p in d.get("plays", [])]
    rows = []
    for p in raw:
        try:
            period, left = SPORTS[sport]["state"](p)
            rows.append([int(timestamp(p["wallclock"])), period, left, int(bool(p.get("scoringPlay"))),
                         int(p.get("scoreValue") or 0), int(p["homeScore"]), int(p["awayScore"])])
        except (KeyError, TypeError, ValueError):
            continue
    data = dict(final=final, lines=lines, plays=rows)
    save(file, data)
    return data


def safe_log(sport, data):
    """Rows [ts, period, left, home, away] in which points exist only from the moment their own SCORING PLAY is
    stamped. The archive's running score cannot be used directly: rows stamped earlier (a substitution during
    free throws, the first pitch of an at-bat) already carry the score of a later play, and some games carry
    score changes that belong to no scoring play at all. Returns (rows, sound): sound means the credited
    points add up to the official final score."""
    events, ch, ca, ph, pa, clean = [], 0, 0, 0, 0, True
    for ts, period, left, flag, value, home, away in data["plays"]:
        if flag:
            row, credit = (home-ph, away-pa), (home-ch, away-ca)
            if value and sum(row) == value and min(row) >= 0:
                dh, da = row
            elif min(credit) >= 0 and (not value or sum(credit) in (0, value)):
                dh, da = credit            # (0, 0): a second row for a play already credited
            else:                          # the running score moved by something other than this play
                clean = False
                dh, da = max(credit[0], 0), max(credit[1], 0)
            if dh or da:
                events.append((ts+(NFL_SHIFT if sport == "nfl" else 0), dh, da))
                ch, ca = ch+dh, ca+da
        ph, pa = home, away
    sound = clean and (ch, ca) == (data["final"]["home"], data["final"]["away"])
    events.sort()
    stamps, h, a, cum = [e[0] for e in events], 0, 0, []
    for _, dh, da in events:
        h, a = h+dh, a+da
        cum.append((h, a))
    rows = []
    for ts, period, left, *_ in sorted(data["plays"]):
        i = bisect_right(stamps, ts)-1
        rows.append([ts, period, left, *(cum[i] if i >= 0 else (0, 0))])
    if sport == "nfl":                     # the shifted scoring stamps themselves
        base = [r[0] for r in rows]
        extra = []
        for ts, (h, a) in zip(stamps, cum):
            i = bisect_right(base, ts)-1
            if 0 <= i < len(rows)-1:
                extra.append([ts, rows[i][1], rows[i][2], h, a])
        rows = sorted(rows+extra, key=lambda r: (r[0], r[3]+r[4]))
    # Official scores replace the credited ones at the two moments everyone knows them: period end and game end.
    n = PERIODS[sport]
    if all(len(data["lines"][k]) >= n for k in ("home", "away")):
        inside = [i for i, r in enumerate(rows) if r[1] <= n]
        if inside and inside[-1] < len(rows)-1:
            rows[inside[-1]][3:] = [sum(data["lines"]["home"][:n]), sum(data["lines"]["away"][:n])]
    if rows:
        rows[-1][3:] = [data["final"]["home"], data["final"]["away"]]
    return rows, sound


def verify_original():
    """Re-check the 53 'over already past the line' trades of spreads_totals.py one by one.

    A trade is genuinely decided at entry only if the points credited by SCORING PLAYS stamped at least 15 s
    before the decision bar closed already exceed the line.
    """
    from research import spreads_totals as st
    out = []
    with httpx.Client(timeout=40, headers={"User-Agent": "research-script"}) as client:
        for sport, bundled in (("nba", False), ("mlb", True)):
            linked = st.link(client, sport)
            games = by_game(st.markets(client, st.TYPES[sport]["total"][0]))
            info = listing(client, st.TYPES[sport]["total"][0])
            plan = None
            if bundled:                    # the MLB run shared one batched read between spreads and totals
                both = [by_game(st.markets(client, st.TYPES[sport]["spread"][0])), games]
                plan = {}
                for k in set(both[0]) | set(both[1]):
                    for g in st.requests_for([m for part in both for m in part.get(k, [])]):
                        for m in g:
                            plan[m["ticker"]] = g
            settle, rows = {}, {}
            for k, group in games.items():
                bars = st.bars_of(group, plan)
                if bars is None or not any(bars.values()):
                    continue
                game = linked.get(k)
                log = st.plays(client, sport, game["espn"]) if game else []
                if len(log) < 50:
                    continue
                settle[k] = min(m["settle"] for m in group)
                rows[k] = st.candidates(sport, "total", "game", dict(game, key=k), log, group, bars)
            hits = st.first(rows, sorted(settle, key=settle.get), lambda r: r["decided"])
            for h in hits:
                game = linked[h["key"]]
                data = scoring(client, sport, game["espn"])
                log, sound = safe_log(sport, data)
                cutoff = h["ts"]-60-FEED_DELAY           # the state the original rule was allowed to see
                strike = float(info[h["ticker"]]["strike"])
                official = data["final"]["home"]+data["final"]["away"]
                i = bisect_right([r[0] for r in log[:-1]], cutoff)-1
                at_entry = log[i][3]+log[i][4] if i >= 0 else 0
                passed = next((r[0] for r in log if r[3]+r[4] > strike), None)
                phantom = sum(1 for x, y in zip(data["plays"], data["plays"][1:]) if not y[3] and (y[5], y[6]) != (x[5], x[6]))
                out.append(dict(sport=sport, ticker=h["ticker"], entry_ts=h["ts"], ask=h["ask"], strike=strike,
                                archive_running_total=h["ahead"]+strike, scoring_play_total_at_entry=at_entry,
                                seconds_until_line_really_passed=None if passed is None else max(passed-cutoff, 0),
                                official_final=official, kalshi_value=info[h["ticker"]]["value"],
                                scoring_plays_sum_to_official=sound, rows_whose_score_moved_without_a_scoring_play=phantom,
                                genuinely_decided=at_entry > strike, payout=h["payout"], net=h["net"], minute_volume=h["vol"]))
    save(OUT/"verify-53.json", out)
    for sport in ("nba", "mlb"):
        part = [r for r in out if r["sport"] == sport]
        real = [r for r in part if r["genuinely_decided"]]
        print(sport, "trades", len(part), "losses", sum(r["payout"] < .5 for r in part), "genuinely decided", len(real),
              "of which lost", sum(r["payout"] < .5 for r in real),
              "median minute volume of genuine", median([r["minute_volume"] for r in real] or [0]),
              "net of genuine", round(mean([r["net"] for r in real] or [0]), 2),
              "mean ask of genuine", round(mean([r["ask"] for r in real] or [0]), 2))
        for r in part:
            if not r["genuinely_decided"]:
                print("   NOT DECIDED", {k: r[k] for k in ("ticker", "ask", "strike", "archive_running_total", "scoring_play_total_at_entry",
                                                         "seconds_until_line_really_passed", "official_final", "payout")})


def tape(limit=400):
    """Trade tape after the deciding play, for the verified entries of recent (not archived) markets.

    Minute bars cannot say how long an offer stood. The tape can: every fill in which the TAKER bought the
    decided side at 99c or less, with its delay from the deciding play's stamp. Fills at least 15 s after the
    stamp are what a 15-second-late trader could at best have shared in.
    """
    from research.api_cost import timestamp
    rows = [r for r in json.loads((OUT/"trades.json").read_text()) if r["verified"] and r["family"] != "winner"]
    rows.sort(key=lambda r: (r["sport"], r["family"], r["kind"], r["ticker"]))
    out = []
    with httpx.Client(timeout=40, headers={"User-Agent": "research-script"}) as client:
        for r in rows[:limit]:
            try:
                data = fetch(client, "/markets/trades", dict(ticker=r["ticker"], min_ts=r["decided_ts"]-60,
                                                             max_ts=min(r["decided_ts"]+900, r["close"]+60), limit=1000))
            except RuntimeError:
                continue
            time.sleep(.4)
            fills = []
            for t in data.get("trades", []):
                if t.get("taker_side") != r["side"]:
                    continue
                price = round(100*float(t["yes_price_dollars" if r["side"] == "yes" else "no_price_dollars"]))
                if price <= 99:
                    fills.append((timestamp(t["created_time"])-r["decided_ts"], price, float(t["count_fp"])))
            late = [f for f in fills if f[0] >= FEED_DELAY]
            profit = lambda fs: sum((100-p-7*p/100*(1-p/100))*c for _, p, c in fs)/100
            out.append(dict(sport=r["sport"], family=r["family"], kind=r["kind"], ticker=r["ticker"], key=r["key"], settle=r["settle"],
                            contracts_after_15s=sum(f[2] for f in late), contracts_after_15s_at_98_or_less=sum(f[2] for f in late if f[1] <= 98),
                            contracts_after_60s=sum(f[2] for f in late if f[0] >= 60),
                            dollars_after_15s=profit(late), dollars_after_60s=profit([f for f in late if f[0] >= 60]),
                            last_fill_seconds=max((f[0] for f in late), default=None),
                            # Fills far below 95c so soon after a "decided" stamp mean the stamp ran ahead of the
                            # event (a review, a free throw still to come); only 95-99c fills are counted as clean.
                            clean_contracts_after_15s=sum(f[2] for f in late if f[1] >= 95),
                            clean_dollars_after_15s=profit([f for f in late if f[1] >= 95]),
                            clean_dollars_after_60s=profit([f for f in late if f[1] >= 95 and f[0] >= 60]),
                            clean_last_fill_seconds=max((f[0] for f in late if f[1] >= 95), default=None),
                            fills=[[round(f[0], 1), f[1], f[2]] for f in sorted(fills)][:40]))
    save(OUT/"tape.json", out)
    groups = defaultdict(list)
    for x in out:
        groups[x["sport"], x["family"], x["kind"]].append(x)
    families = json.loads((OUT/"summary.json").read_text())["families"]
    season = {"nba": 50, "mlb": 95, "nhl": 50, "nfl": 16}          # games in a regular-season week
    for name, part in sorted(groups.items()):
        games = families[":".join(name)]["ask<=99 wait>=15s"]["all"]["games"]
        hit = [x for x in part if x["clean_contracts_after_15s"] > 0]
        clean, slow, dirty = (sum(x[k] for x in part) for k in ("clean_dollars_after_15s", "clean_dollars_after_60s", "dollars_after_15s"))
        print(f"{':'.join(name):22} games {games:4} markets {len(part):3} with 95-99c taker fills >=15s after the play {len(hit):3} | "
              f"median contracts {median([x['clean_contracts_after_15s'] for x in hit] or [0]):5.0f} median last fill {median([x['clean_last_fill_seconds'] for x in hit] or [0]):4.0f}s | "
              f"all takers' profit ${clean:5.0f} (>=60s ${slow:4.0f}; incl. fills below 95c ${dirty:5.0f}) | "
              f"${clean/games*season[name[0]]:4.0f} per in-season week (>=60s ${slow/games*season[name[0]]:3.0f})")


def table():
    """Compact table from summary.json: every family and kind, earliest 60% / latest 40% of games and all."""
    families = json.loads((OUT/"summary.json").read_text())["families"]
    lines = ["family:kind | rule (deciding play at least this old) | part | games | decided markets | trades (share of decided markets) | "
             "losses/unverified | mean ask | share at 99c | net c | net c verified | trades per 100 games | games with a trade per 100 | "
             "trades per week of the sample span | median entry-minute volume | share with a trade in that minute | "
             "median minutes the offer lasted, share 3+ | median seconds from play to entry"]
    for name, rules in families.items():
        for rule in ("ask<=99 wait>=15s", "ask<=98 wait>=15s", "ask<=99 wait>=120s"):
            for tag in ("early60", "late40", "all"):
                r = rules[rule][tag]
                if not r.get("trades"):
                    lines.append(f"{name:22} {rule:18} {tag:7} g {r['games']:4} dm {r['decided_markets']:5} trades 0")
                    continue
                ver = r["net_cents_verified"]
                lines.append(
                    f"{name:22} {rule:18} {tag:7} g {r['games']:4} dm {r['decided_markets']:5} tr {r['trades']:3} "
                    f"({100*r['share_of_decided_markets_offered']:.1f}%) L{r['losses']}/U{r['unverified']} ask {r['mean_ask']:.1f} "
                    f"at99 {100*r['asks'].get('99', 0)/r['trades']:.0f}% net {r['net_cents']:+.2f} ver {'-' if ver is None else format(ver, '+.2f')} "
                    f"/100g {100*r['trades']/r['games']:.1f} gm/100g {100*r['games_with_trade']/r['games']:.1f} /wk {r['trades_per_week']:.1f} "
                    f"vol {r['median_entry_minute_volume']:.0f} trd {100*r['share_entry_minute_traded']:.0f}% "
                    f"pers {r['median_persist_minutes']:.0f}m,{100*r['share_persist_3plus']:.0f}% lag {r['median_lag_seconds']:.0f}s")
    (OUT/"table.txt").write_text("\n".join(lines)+"\n")
    print("\n".join(lines))


def show(report, tags=("early60", "late40", "all"), rules=("ask<=99 wait>=15s", "ask<=99 wait>=120s")):
    for name, caps in report.items():
        for cap, parts in caps.items():
            if cap not in rules:
                continue
            for tag in tags:
                s = parts[tag]
                if not s.get("trades"):
                    continue
                print(f"{name:24} {cap:20} {tag:7} games {s['games']:4} decided mkts {s['decided_markets']:5} "
                      f"trades {s['trades']:4} ({100*s['share_of_decided_markets_offered']:.1f}%) loss {s['losses']} unver {s['unverified']} "
                      f"ask {s['mean_ask']:.1f} net {s['net_cents']:+.2f}c /wk {s['trades_per_week']:.1f} games/wk {s['games_with_trade_per_week']:.1f} "
                      f"vol med {s['median_entry_minute_volume']:.0f} traded {100*s['share_entry_minute_traded']:.0f}% "
                      f"persist {s['median_persist_minutes']:.0f}m (3m+ {100*s['share_persist_3plus']:.0f}%) lag {s['median_lag_seconds']:.0f}s "
                      f"asks {s['asks']} net verified {s['net_cents_verified'] if s['net_cents_verified'] is None else round(s['net_cents_verified'], 2)} "
                      f"per 100 games {100*s['trades']/s['games']:.1f} $/wk {s['dollars_per_week_at_entry_minute_volume']:.0f}")


if __name__ == "__main__":
    run()
