"""Paper-trade the 15-40 rule on synchronized score and order-book recordings.

Entry: the feed shows the server newly at 15-40 inside a game already being
watched. Two seconds later buy one contract on the receiver at the displayed
ask. Exit: two seconds after the feed first shows deuce or a finished game,
sell at the displayed bid. Both books must be two-sided, sized and at most
five seconds old. A tape that ends first leaves the position unresolved.
"""
import argparse
from bisect import bisect_right
from collections import Counter, defaultdict
from decimal import Decimal as D
import json
from pathlib import Path
from statistics import mean, median
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from research.api_cost import ROOT, fee, save
from research.native_replay import quote
from research.score_break import Books, SECOND

OUT = ROOT / "data/processed/fifteen-forty"
LATENCY, MAX_AGE = 2*SECOND, 5*SECOND
RANK = {"0": 0, "15": 1, "30": 2, "40": 3, "A": 4, "AD": 4}


def state(d):
    """(games tuple, server 1|2, server points, receiver points) or None."""
    serve = (d.get("native") or {}).get("event_serve")
    try:
        sets = tuple((int(float(s["score_first"])), int(float(s["score_second"]))) for s in d["set_scores"] or [])
        a, b = (RANK[x.strip().upper()] for x in d["game_score"].split("-"))
    except (KeyError, TypeError, ValueError, AttributeError):
        return None
    if serve not in ("First Player", "Second Player") or not str(d.get("status", "")).startswith("Set"):
        return None
    return (sets, 1, a, b) if serve == "First Player" else (sets, 2, b, a)


def read(tape):
    """Return matches, per-ticker quote history and per-fixture score receipts."""
    books, history, scores = Books(), defaultdict(lambda: ([], [])), defaultdict(list)
    matches, last = {}, 0
    with tape.open() as f:
        for line in f:
            try:
                r = json.loads(line)
            except ValueError:      # final line of a recording still being written
                break
            k, d, last = r["kind"], r["data"], r["mono_ns"]
            if k == "start":
                matches = d["matches"]
            elif k in ("orderbook_snapshot", "orderbook_delta"):
                books.apply(k, d, last)
                t = d["market_ticker"]
                q = quote(books.levels.get(t), t, 0)
                times, rows = history[t]
                times.append(last)
                rows.append((q.get("yes_bid"), q.get("yes_ask"), q["bid_size"], q["ask_size"]))
            elif k == "score":
                scores[d["fixture"]].append((last, state(d), d.get("status"), d.get("winner")))
    return matches, history, scores, last, books.gaps


def touch(history, ticker, yes, at, max_age=MAX_AGE):
    """Executable (bid, ask) in cents for the chosen side, or None."""
    times, rows = history[ticker]
    i = bisect_right(times, at)-1
    if i < 0 or at-times[i] > max_age:
        return None
    bid, ask, bid_size, ask_size = rows[i]
    if not bid_size or not ask_size or bid is None or ask is None or not 0 < bid <= ask < 100:
        return None
    return (bid, ask) if yes else (100-ask, 100-bid)


def trades(tape):
    matches, history, scores, end, gaps = read(tape)
    out = []
    for fixture, rows in scores.items():
        meta = matches[fixture]
        i = 1
        while i < len(rows):
            (_, prev, *_), (at, now, *_) = rows[i-1], rows[i]
            i += 1
            if not now or now[2:] != (1, 3):
                continue
            row = dict(tape=tape.name, fixture=fixture, match=f'{meta["player1"]} v {meta["player2"]}',
                       server=now[1], receipt_s=round((at-rows[0][0])/1e9, 1))
            out.append(row)
            if not prev or prev[:2] != now[:2] or prev[2:] == (1, 3) or prev[2:] not in ((1, 2), (0, 3)):
                # Covers a repeated receipt, a game joined part-way and a skipped point.
                if prev and prev == now:
                    out.pop()
                else:
                    row["status"] = "not_a_clean_transition"
                continue
            yes = meta["yes_player"] != now[1]          # receiver is the YES player
            def mid(offset, age=15*SECOND):
                q = touch(history, meta["ticker"], yes, at+offset, age)
                return mean(q) if q else None
            row["receiver_mid"] = {str(s): mid(s*SECOND) for s in (-60, -30, -15, -5, 0, 2)}
            entry = touch(history, meta["ticker"], yes, at+LATENCY)
            if not entry:
                row["status"] = "no_fresh_entry_book"
                continue
            # First later receipt that shows deuce or a different game.
            result = None
            for j in range(i, len(rows)):
                t, s, status, winner = rows[j]
                if status == "Finished" or (s and s[0] != now[0]):
                    games_now = [sum(x) for x in zip(*s[0])] if s else None
                    games_then = [sum(x) for x in zip(*now[0])]
                    if s:
                        result = "break" if games_now[now[1]-1] == games_then[now[1]-1] else "hold"
                    else:
                        result = "match_over"
                elif s and s[:2] == now[:2] and min(s[2:]) >= 3:
                    result = "deuce"
                if result:
                    i = j
                    break
            if not result:
                row["status"] = "unresolved_at_tape_end"
                continue
            leave = touch(history, meta["ticker"], yes, t+LATENCY)
            row.update(result=result, held_s=round((t-at)/1e9, 1), entry_ask=entry[1])
            if not leave:
                row["status"] = "no_fresh_exit_book"
                continue
            buy, sell = D(entry[1])/100, D(leave[0])/100
            costs = fee(buy)+fee(sell)
            row.update(status="closed", exit_bid=leave[0], gross_cents=leave[0]-entry[1],
                       fees_cents=float(100*costs), net_cents=float(leave[0]-entry[1]-100*costs),
                       mid_to_mid_cents=mean(leave)-mean(entry))
    return out, gaps


def summarize(rows):
    closed = [r for r in rows if r.get("status") == "closed"]
    out = dict(events=dict(Counter(r["status"] for r in rows)), closed=len(closed),
               distinct_matches=len({r["fixture"] for r in closed}))
    if closed:
        lead = [r["receiver_mid"]["0"]-r["receiver_mid"]["-30"] for r in closed
                if r["receiver_mid"]["0"] is not None and r["receiver_mid"]["-30"] is not None]
        out.update(results=dict(Counter(r["result"] for r in closed)),
                   net_cents_total=sum(r["net_cents"] for r in closed),
                   net_cents_mean=mean(r["net_cents"] for r in closed),
                   net_cents_median=median(r["net_cents"] for r in closed),
                   gross_cents_mean=mean(r["gross_cents"] for r in closed),
                   mid_to_mid_cents_mean=mean(r["mid_to_mid_cents"] for r in closed),
                   profitable=sum(r["net_cents"] > 0 for r in closed),
                   receiver_mid_move_in_30s_before_receipt_cents=mean(lead) if lead else None)
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("tapes", type=Path, nargs="+")
    args = parser.parse_args()
    rows, gaps = [], 0
    for tape in args.tapes:
        found, g = trades(tape)
        rows += found
        gaps += g
    report = dict(summary=summarize(rows), book_sequence_gaps=gaps, trades=rows)
    save(OUT/"tape.json", report)
    print(json.dumps(report["summary"], indent=2))
    for r in rows:
        if r.get("status") == "closed":
            print(f'{r["match"]:38} {r["result"]:6} buy {r["entry_ask"]:>3} sell {r["exit_bid"]:>3} '
                  f'net {r["net_cents"]:+.0f}c held {r["held_s"]}s mids {r["receiver_mid"]}')


if __name__ == "__main__":
    main()
