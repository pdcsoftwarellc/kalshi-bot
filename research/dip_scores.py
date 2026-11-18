"""Which score situations do tennis favourite dips come from?

Reads the live_games.py recording (data/raw/live-games/*.jsonl) and, for each
tennis match, works out which competitor the recorded YES market is on, the
favourite and its reference price, a game-by-game score timeline, and the first
moment the favourite's ask fell to each dip level (80c, 70c, 65c, and 10c below
the reference price) with the score at that moment. Matches that finished inside
the recording get the winner, so each dip has an outcome.

A dip is "ask <= level": a bid resting at the level would have been at or
through the best offer. The recording is a ~20 second poll of a cached listing
quote, not trades, so a dip that came and went between polls is missed, the
fill is not proven, and the score feed lags the book by 5-10 seconds (the score
shown beside a dip can be one point old). `score_30s_later` is kept for that.

Kalshi's live data carries competitor ids but no names, so the YES side cannot
be matched by name. It is taken from the position of the market's code in the
event ticker (…-26NOV17RONBAR + …-BAR -> competitor 2) and checked two ways:
against the final result and against which way the price moved when each
competitor won a game. A match where those disagree is left out and reported.

Offline; reads files only.
"""
import argparse
from collections import defaultdict
from datetime import datetime
import json
from pathlib import Path
import re
from statistics import median
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from research.api_cost import ROOT, UTC, save

RAW = ROOT / "data/raw/live-games"
OUT = ROOT / "data/processed/dip-scores"
TENNIS = ("KXATPMATCH", "KXATPCHALLENGERMATCH", "KXWTAMATCH")
DEAD = {"closed", "ended", "complete", "completed", "cancelled", "canceled", "postponed", "finished", "final"}
IDLE = {"not_started", "scheduled", "created", "sch", "", "none"}
LEVELS = (80, 70, 65)
RELATIVE = 10
MAX_SPREAD = 4          # cents; a wider quote is not used as a reference price
GAP = 90                # seconds; a longer silence before a dip is flagged
MIN_DIP = 8             # cents; the tables only count a fixed level this far below the reference (as favorite_dip.py)
BUCKETS = ((90, 101, "90c+"), (80, 90, "80-90c"), (0, 80, "under 80c"))


# ----------------------------------------------------------------- reading

def load(folder=RAW):
    """Tennis rows per event in time order; a partial last line is skipped."""
    events = defaultdict(list)
    for path in sorted(Path(folder).glob("*.jsonl")):
        with path.open() as lines:
            for line in lines:
                try:
                    row = json.loads(line)
                    if row["event"].startswith(TENNIS) and isinstance(row.get("state"), dict):
                        events[row["event"]].append(row)
                except (ValueError, KeyError, TypeError):
                    continue
    for rows in events.values():
        rows.sort(key=lambda r: r["ts"])
    return events


def phase(state):
    status = str(state.get("status", "")).lower()
    return "pre" if status in IDLE else "done" if status in DEAD else "live"


def yes_quote(row):
    """(bid, ask) of the YES side in cents, or None when the quote is unusable."""
    try:
        bid, ask = 100*float(row["quote"]["yes_bid_dollars"]), 100*float(row["quote"]["yes_ask_dollars"])
    except (KeyError, TypeError, ValueError):
        return None
    return (bid, ask) if 0 <= bid <= ask <= 100 else None


def side_quote(row, side, yes):
    """(bid, ask) in cents for competitor `side` when YES is competitor `yes`."""
    q = yes_quote(row)
    if q is None:
        return None
    return q if side == yes else (100-q[1], 100-q[0])


# ----------------------------------------------------------------- score state

def sets(state):
    """[(games1, games2, winner or None)] per set, from the two round-score lists."""
    out = []
    for a, b in zip(state.get("competitor1_round_scores") or [], state.get("competitor2_round_scores") or []):
        won = 1 if a.get("outcome") == "winner" else 2 if b.get("outcome") == "winner" else None
        out.append((a.get("score") or 0, b.get("score") or 0, won))
    return out


def winner(state):
    return {state.get("competitor1_id"): 1, state.get("competitor2_id"): 2}.get(state.get("winner") or None)


def score_state(state, fav):
    """The score from the favourite's side (fav is competitor 1 or 2).

    break_diff is the favourite's breaks minus the opponent's in the current
    set: games won minus service games completed. It needs the current server
    when an odd number of games has been played, and is None without one.
    """
    dog = 3-fav
    rounds = sets(state)
    done = [r for r in rounds if r[2]]
    current = next((r for r in rounds if not r[2]), None)
    sets_fav, sets_dog = sum(r[2] == fav for r in done), sum(r[2] == dog for r in done)
    games_fav, games_dog = (current[fav-1], current[dog-1]) if current else (0, 0)
    played = games_fav+games_dog
    server = {state.get("competitor1_id"): 1, state.get("competitor2_id"): 2}.get(state.get("server") or None)
    tiebreak = bool(current) and games_fav == games_dog >= 6
    if not current:
        break_diff = None
    elif played % 2 == 0:
        break_diff = games_fav-played//2
    elif server:
        break_diff = games_fav-(played//2 if server == fav else played//2+1)
    else:
        break_diff = None
    points = (state.get(f"competitor{fav}_current_round_score"), state.get(f"competitor{dog}_current_round_score"))
    out = dict(sets_fav=sets_fav, sets_dog=sets_dog, set_number=len(done)+bool(current),
               games_fav=games_fav, games_dog=games_dog, games_into_set=played,
               points_fav=points[0], points_dog=points[1], tiebreak=tiebreak,
               fav_serving=None if not server else server == fav, break_diff=break_diff,
               lost_last_set=bool(done) and done[-1][2] == dog,
               set_scores=[(r[fav-1], r[dog-1]) for r in done], phase=phase(state),
               match_status=state.get("match_status"))
    # A break point: the returner is one point from the game (40 against less, or advantage).
    adv = {state.get("competitor1_id"): 1, state.get("competitor2_id"): 2}.get(state.get("advantage") or None)
    out["facing_break_point"] = out["has_break_point"] = False
    if server and current and not tiebreak and out["phase"] == "live":
        returner = 3-server
        behind, ahead = state.get(f"competitor{server}_current_round_score"), state.get(f"competitor{returner}_current_round_score")
        if adv == returner or ahead == 50 or (ahead == 40 and behind in (0, 15, 30)):
            out["facing_break_point" if server == fav else "has_break_point"] = True
    out["just_lost_set"] = out["lost_last_set"] and bool(current) and played <= 1
    out["text"] = " ".join(f"{a}-{b}" for a, b in out["set_scores"]+([(games_fav, games_dog)] if current else []))
    out["sets_situation"], out["break_situation"] = situation(out)
    return out


def situation(s):
    """Two coarse labels for the table: the set score, and the state of the current set."""
    if s["phase"] == "done":
        return "match over", "match over"
    if s["sets_dog"] > s["sets_fav"]:
        in_sets = "behind in sets, just lost the set" if s["just_lost_set"] else "behind in sets"
    elif s["sets_fav"] > s["sets_dog"]:
        in_sets = "ahead in sets"
    elif s["sets_fav"]:
        in_sets = "level in sets"
    else:
        in_sets = "first set"
    diff = s["break_diff"]
    in_set = "tiebreak" if s["tiebreak"] else "break unknown" if diff is None else \
        "down a break" if diff == -1 else "down two breaks or more" if diff < -1 else "on serve" if diff == 0 else "up a break"
    if s.get("facing_break_point"):
        in_set += ", facing break point"
    return in_sets, in_set


# ----------------------------------------------------------------- YES side

def code_side(event, ticker):
    """Competitor number from where the market's code sits in the event ticker, or None if unclear."""
    names = re.sub(r"^\d{2}[A-Z]{3}\d{2}", "", event.rsplit("-", 1)[-1])      # drop the date, e.g. 26NOV17
    code = ticker.rsplit("-", 1)[-1]
    first, last = names.startswith(code), names.endswith(code)
    if first == last:
        return None
    return 1 if first else 2


def result_side(rows):
    """Competitor YES must be on, judged from the settled price and the feed's winner."""
    for row in reversed(rows):
        won, q = winner(row["state"]), yes_quote(row)
        if phase(row["state"]) == "done" and won and q:
            if q[0] >= 90:
                return won
            if q[1] <= 10:
                return 3-won
    return None


def direction_side(rows):
    """Competitor YES must be on, judged from how the YES price moved across each game won.

    Returns (side or None, games agreeing with side 1, games agreeing with side 2).
    """
    marks = []
    for i, row in enumerate(rows):
        q = yes_quote(row)
        if phase(row["state"]) == "live" and q and q[1]-q[0] <= 2*MAX_SPREAD:
            games = [sum(r[k] for r in sets(row["state"])) for k in (0, 1)]
            marks.append((games, (q[0]+q[1])/2))
    one = two = 0
    for i in range(1, len(marks)):
        d1, d2 = marks[i][0][0]-marks[i-1][0][0], marks[i][0][1]-marks[i-1][0][1]
        if sorted((d1, d2)) != [0, 1]:
            continue
        before, after = marks[max(0, i-2)][1], marks[min(len(marks)-1, i+1)][1]
        if after == before:
            continue
        if (after > before) == (d1 == 1):
            one += 1
        else:
            two += 1
    side = None if max(one, two) < 5 or min(one, two)*2 > max(one, two) else 1 if one > two else 2
    return side, one, two


def yes_side(event, rows):
    """(side or None, how it was decided, the three votes)."""
    code, result = code_side(event, rows[0]["ticker"]), result_side(rows)
    direction, one, two = direction_side(rows)
    votes = dict(code=code, result=result, direction=direction, games_for_1=one, games_for_2=two)
    # The result outranks the noisy game-by-game direction; the code needs one of them to back it.
    firm = [v for v in (code, result) if v]
    if len(set(firm)) > 1 or (len(firm) == 1 and direction and direction != firm[0]):
        return None, "conflict", votes
    side = firm[0] if firm else direction
    if side is None:
        return None, "unknown", votes
    return side, "+".join(k for k in ("code", "result", "direction") if votes[k] == side), votes


# ----------------------------------------------------------------- matches and dips

def reference(rows, yes):
    """(favourite, its price, how good the reference is, index of the row it came from).

    pre_match: quotes recorded before play (not possible until the recorder keeps them).
    first_game: the first recorded row is live but still 0-0 in the first set.
    mid_match: the recording joined the match later; the price is just the earliest seen.
    """
    def clean(row):
        q = yes_quote(row)
        return q is not None and q[1]-q[0] <= MAX_SPREAD

    before = [i for i, r in enumerate(rows) if phase(r["state"]) == "pre"]
    first_live = next((i for i, r in enumerate(rows) if phase(r["state"]) != "pre"), len(rows))
    before = [i for i in before if i < first_live and clean(rows[i])][-15:]
    if before:
        mid, at, kind = median(sum(yes_quote(rows[i]))/2 for i in before), before[-1], "pre_match"
    else:
        at, kind = None, "mid_match"
        for i in range(first_live, len(rows)):
            rounds = sets(rows[i]["state"])
            fresh = len(rounds) <= 1 and sum(sum(r[:2]) for r in rounds) == 0 and phase(rows[i]["state"]) == "live"
            if i == first_live:
                kind = "first_game" if fresh else "mid_match"
            if kind == "first_game" and not fresh:
                break
            if clean(rows[i]):
                at = i
                break
        if at is None:
            at = next((i for i in range(first_live, len(rows)) if clean(rows[i])), None)
            if at is None:
                return None
            kind = "mid_match"
        mid = sum(yes_quote(rows[at]))/2
    if mid == 50:
        return None
    fav = yes if mid > 50 else 3-yes
    return fav, mid if mid > 50 else 100-mid, kind, at


def later_score(rows, i, fav, seconds=30):
    for row in rows[i+1:]:
        if row["ts"] >= rows[i]["ts"]+seconds:
            return score_state(row["state"], fav)["text"]
    return None


def dips(rows, yes, fav, price, start):
    """First row at or after `start` where the favourite's ask is at or below each level."""
    levels = {str(level): level for level in LEVELS if price > level}
    levels[f"pre-{RELATIVE}"] = price-RELATIVE
    out = []
    for name, level in levels.items():
        hit = strict = None
        for i in range(start+1, len(rows)):
            q = side_quote(rows[i], fav, yes)
            if q is None:
                continue
            if hit is None and q[1] <= level:
                hit = i
            if q[1] < level and phase(rows[i]["state"]) == "live":
                strict = True
                break
        if hit is None:
            continue
        bid, ask = side_quote(rows[hit], fav, yes)
        state = score_state(rows[hit]["state"], fav)
        out.append(dict(level_name=name, level=level, depth=price-level, ts=rows[hit]["ts"],
                        time=f"{datetime.fromtimestamp(rows[hit]['ts'], UTC):%Y-%m-%d %H:%M:%S}",
                        fav_bid=bid, fav_ask=ask, ask_below_level_while_live=bool(strict),
                        seconds_since_previous_row=rows[hit]["ts"]-rows[hit-1]["ts"],
                        after_gap=rows[hit]["ts"]-rows[hit-1]["ts"] > GAP,
                        in_play=state["phase"] == "live", score=state,
                        score_30s_later=later_score(rows, hit, fav)))
    return out


def timeline(rows, yes, fav):
    """One entry per change of the game score, with the favourite's quote at that row."""
    out, last = [], None
    for row in rows:
        s = score_state(row["state"], fav)
        key = (s["text"], s["phase"])
        if key == last:
            continue
        last, q = key, side_quote(row, fav, yes)
        out.append(dict(ts=row["ts"], score=s["text"], phase=s["phase"], fav_serving=s["fav_serving"],
                        break_diff=s["break_diff"], fav_bid=q and q[0], fav_ask=q and q[1]))
    return out


def match(event, rows):
    side, how, votes = yes_side(event, rows)
    base = dict(event=event, ticker=rows[0]["ticker"], yes_name=rows[0].get("yes"), rows=len(rows),
                first_ts=rows[0]["ts"], last_ts=rows[-1]["ts"], scheduled_start=rows[0].get("start"),
                yes_competitor=side, yes_mapping=how, mapping_votes=votes,
                longest_gap_seconds=max((b["ts"]-a["ts"] for a, b in zip(rows, rows[1:])
                                         if phase(b["state"]) == "live"), default=0))
    if side is None:
        return dict(base, usable=False, reason="YES side "+how)
    ref = reference(rows, side)
    if ref is None:
        return dict(base, usable=False, reason="no usable reference price")
    fav, price, kind, at = ref
    won = winner(rows[-1]["state"]) if phase(rows[-1]["state"]) == "done" else None
    first = score_state(rows[at]["state"], fav)
    asks = [q[1] for r in rows[at:] if phase(r["state"]) == "live" and (q := side_quote(r, fav, side))]
    return dict(base, usable=True, favourite_competitor=fav, favourite_is_yes=fav == side,
                favourite_name=rows[0].get("yes") if fav == side else None,
                reference_price=price, reference_kind=kind, reference_score=first["text"],
                reference_seconds_after_scheduled_start=seconds_after_start(rows[at]),
                finished=phase(rows[-1]["state"]) == "done", final_status=rows[-1]["state"].get("match_status"),
                winner_competitor=won, favourite_won=None if won is None else won == fav,
                final_score=score_state(rows[-1]["state"], fav)["text"], lowest_live_ask=min(asks, default=None),
                dips=dips(rows, side, fav, price, at), timeline=timeline(rows, side, fav))


def seconds_after_start(row):
    try:
        return row["ts"]-datetime.fromisoformat(row["start"].replace("Z", "+00:00")).timestamp()
    except (AttributeError, TypeError, ValueError):
        return None


# ----------------------------------------------------------------- report

def bucket(price):
    return next(name for lo, hi, name in BUCKETS if lo <= price < hi)


def tally(found):
    won = sum(d["favourite_won"] is True for d in found)
    lost = sum(d["favourite_won"] is False for d in found)
    nets = [100*d["favourite_won"]-d["level"] for d in found if d["favourite_won"] is not None]
    return dict(dips=len(found), favourite_won=won, favourite_lost=lost, unfinished=len(found)-won-lost,
                net_cents_per_fill=sum(nets)/len(nets) if nets else None)


def summarise(matches):
    usable = [m for m in matches if m["usable"]]
    flat = [dict(d, event=m["event"], reference_price=m["reference_price"], reference_kind=m["reference_kind"],
                 bucket=bucket(m["reference_price"]), favourite_won=m["favourite_won"])
            for m in usable for d in m["dips"]]
    at_start = [m for m in usable if m["reference_kind"] != "mid_match"]
    counts = dict(matches=len(matches), usable=len(usable), unmapped=len(matches)-len(usable),
                  finished=sum(m["finished"] for m in usable),
                  reference_pre_match=sum(m["reference_kind"] == "pre_match" for m in usable),
                  reference_first_game=sum(m["reference_kind"] == "first_game" for m in usable),
                  reference_mid_match=sum(m["reference_kind"] == "mid_match" for m in usable),
                  favourites_by_bucket={name: sum(bucket(m["reference_price"]) == name for m in at_start)
                                        for _, _, name in BUCKETS})
    tables = {}
    for d in flat:
        if d["reference_kind"] == "mid_match" or d["depth"] < MIN_DIP or not d["in_play"]:
            continue
        for label in (d["score"]["sets_situation"], "  "+d["score"]["break_situation"], "ALL"):
            tables.setdefault((d["bucket"], d["level_name"], label), []).append(d)
    table = [dict(bucket=b, level=lv, situation=label, **tally(found)) for (b, lv, label), found in sorted(tables.items())]
    counts["dips_in_table"] = sum(row["dips"] for row in table if row["situation"] == "ALL")
    counts["dips_recorded"] = len(flat)
    return dict(counts=counts, table=table,
                unmapped=[dict(event=m["event"], reason=m["reason"], votes=m["mapping_votes"]) for m in matches if not m["usable"]]), flat


def show(matches, summary, flat):
    c = summary["counts"]
    print(f'{c["matches"]} tennis matches recorded, {c["usable"]} usable ({c["unmapped"]} left out), {c["finished"]} finished in the recording')
    print(f'reference price: {c["reference_pre_match"]} before play, {c["reference_first_game"]} during the first game, '
          f'{c["reference_mid_match"]} joined mid-match (kept out of the tables)')
    print("favourites by reference price (matches caught from the start): "
          + ", ".join(f"{k} {v}" for k, v in c["favourites_by_bucket"].items()))
    for row in summary["unmapped"]:
        print("  left out:", row["event"], row["reason"], row["votes"])
    print("\nmatches")
    for m in sorted((m for m in matches if m["usable"]), key=lambda m: m["first_ts"]):
        print(f'  {m["event"].split("-", 1)[1]:16} fav {m["reference_price"]:5.1f}c ({m["reference_kind"]:10} at {m["reference_score"] or "-":9}) '
              f'low ask {m["lowest_live_ask"] if m["lowest_live_ask"] is not None else float("nan"):5.1f} '
              f'{"won " if m["favourite_won"] else "LOST" if m["favourite_won"] is False else "open"} {m["final_score"]:14} map {m["yes_mapping"]}')
    print("\ndips (first time the favourite's ask was at or below the level)")
    for d in sorted(flat, key=lambda d: (d["ts"], d["level"])):
        s = d["score"]
        print(f'  {d["time"][11:]} {d["event"].split("-", 1)[1]:16} ref {d["reference_price"]:5.1f} ({d["reference_kind"][:5]}) level {d["level_name"]:6} '
              f'ask {d["fav_ask"]:5.1f} score {s["text"]:12} pts {s["points_fav"]}-{s["points_dog"]} | {s["sets_situation"]}; {s["break_situation"]}'
              f'{" [after gap]" if d["after_gap"] else ""}{"" if d["in_play"] else " [not in play]"} -> '
              f'{"won" if d["favourite_won"] else "LOST" if d["favourite_won"] is False else "open"}')
    print(f"\noutcome by score situation: matches caught from the start, dip of at least {MIN_DIP}c, in play (indented rows split the same dips by the state of the current set)")
    if not summary["table"]:
        print("  no qualifying dips yet")
    for row in summary["table"]:
        net = "" if row["net_cents_per_fill"] is None else f' net {row["net_cents_per_fill"]:+.1f}c'
        print(f'  fav {row["bucket"]:9} level {row["level"]:6} {row["situation"]:36} dips {row["dips"]:2} won {row["favourite_won"]:2} '
              f'lost {row["favourite_lost"]:2} open {row["unfinished"]:2}{net}')


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--raw", type=Path, default=RAW)
    parser.add_argument("--out", type=Path, default=OUT)
    args = parser.parse_args()
    matches = [match(event, rows) for event, rows in sorted(load(args.raw).items())]
    summary, flat = summarise(matches)
    save(args.out/"matches.json", matches)
    save(args.out/"dips.json", flat)
    save(args.out/"summary.json", summary)
    show(matches, summary, flat)


if __name__ == "__main__":
    main()
