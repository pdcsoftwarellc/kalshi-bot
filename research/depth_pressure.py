"""Replay a frozen ATP depth-imbalance signal against local WebSocket receipt tapes."""
import argparse
from collections import Counter
from decimal import Decimal as D
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from research.api_cost import ROOT, fee, save
from research.score_break import Books, SECOND


OUT = ROOT / "data/processed/parallel-round2/depth_pressure"
PRIOR_TAPES = ("1794944180459469000", "1794944411099876000", "1794944775625226000")
OTHER = {"yes": "no", "no": "yes"}


def top(book):
    if not book or not all(book.get(side) for side in OTHER):
        return None
    bids = {side: max(book[side]) for side in OTHER}
    spread = 1 - bids["no"] - bids["yes"]
    if not 0 <= spread < 1 or not all(0 < p < 1 for p in bids.values()):
        return None
    return bids, spread


def pressure(book):
    quote = top(book)
    if quote is None or quote[1] > D(".02"):
        return None
    best = quote[0]
    qty = {side: sum((n for p, n in book[side].items() if p >= best[side] - D(".02")), D(0))
           for side in OTHER}
    total = sum(qty.values())
    for side, other in OTHER.items():
        if total > 0 and qty[side] >= D(".80") * total and book[other][best[other]] >= 1:
            return side
    return None


def reachable(book, side, buying):
    quote = top(book)
    if quote is None:
        return None
    book_side = OTHER[side] if buying else side
    best = quote[0][book_side]
    if book[book_side][best] < 1:
        return None
    return 1 - best if buying else best


def live_score(message):
    return (str(message.get("live")) == "1" and not message.get("winner")
            and str(message.get("status", "")).lower().startswith("set "))


def protect_missing_intervals(rows):
    """A skipped sequence makes the interval since the preceding book frame unobservable."""
    frames = list(rows)
    last = {}
    invalidate_after = set()
    for i, frame in enumerate(frames):
        if frame["kind"] in ("orderbook_snapshot", "orderbook_delta"):
            msg = frame["data"]
            previous = last.get(msg["sid"])
            if previous and msg["seq"] != previous[0] + 1:
                invalidate_after.add(previous[1])
            last[msg["sid"]] = (msg["seq"], i)
    for i, frame in enumerate(frames):
        yield frame
        if i in invalidate_after:
            yield dict(frame, kind="missing_sequence_interval", data={})


def replay(rows):
    """Process updates in receipt order, firing timers between observations."""
    books = Books()
    markets = {}
    active, ever_active = set(), set()
    streaks, attempts, timers = {}, {}, {}
    counts = Counter()
    first = last = last_utc = None

    def refresh(ticker, now):
        side = pressure(books.levels.get(ticker)) if ticker in active and ticker not in attempts else None
        if side is None:
            streaks.pop(ticker, None)
        elif ticker not in streaks or streaks[ticker][0] != side:
            streaks[ticker] = (side, now)

    def work_until(until):
        while True:
            due = [(start + 3 * SECOND, ticker, "signal") for ticker, (_, start) in streaks.items()]
            due.extend((at, ticker, operation) for ticker, (at, operation) in timers.items())
            if not due:
                break
            at, ticker, operation = min(due)
            if at > until:
                break
            utc = last_utc + at - last
            book = books.levels.get(ticker)
            if operation == "signal":
                side, start = streaks.pop(ticker)
                ask = reachable(book, side, True)
                if ask is None or side != pressure(book) or ticker not in active:
                    raise ValueError("Unobserved book state at signal")
                attempts[ticker] = dict(fixture=markets[ticker], ticker=ticker, side=side,
                                        streak_start_utc_ns=utc - (at-start), signal_utc_ns=utc,
                                        limit=str(ask), status="entry_pending")
                timers[ticker] = (at + 2 * SECOND, "entry")
            elif operation == "entry":
                row = attempts[ticker]
                del timers[ticker]
                ask = reachable(book, row["side"], True)
                if ticker not in active or ask is None or ask > D(row["limit"]):
                    row["status"] = "entry_unfilled"
                    continue
                row.update(status="open", entry_utc_ns=utc, buy=str(ask), entry_fee=str(fee(ask)))
                timers[ticker] = (at + 32 * SECOND, "exit")
            else:
                row = attempts[ticker]
                del timers[ticker]
                bid = reachable(book, row["side"], False)
                if bid is None:
                    row["status"] = "unpriced_exit"
                    continue
                buy = D(row["buy"])
                gross = bid - buy
                fees = fee(buy) + fee(bid)
                row.update(status="closed", exit_utc_ns=utc, sell=str(bid),
                           exit_fee=str(fee(bid)), gross=str(gross), fees=str(fees),
                           pnl=str(gross - fees))

    for row in protect_missing_intervals(rows):
        now, utc, kind, msg = row["mono_ns"], row["utc_ns"], row["kind"], row["data"]
        if first is None:
            first = utc
        if last is not None:
            if now < last:
                raise ValueError("Reordered capture")
            # At identical timestamps, process the full group of messages first.
            work_until(now - 1)
        counts[kind] += 1
        if kind == "start":
            markets = {m["ticker"]:key for key, m in msg["matches"].items()}
        elif kind in ("orderbook_snapshot", "orderbook_delta"):
            gaps = books.gaps
            books.apply(kind, msg, now)
            if books.gaps != gaps:
                streaks.clear()
            refresh(msg["market_ticker"], now)
        elif kind == "missing_sequence_interval":
            books.levels.clear()
            streaks.clear()
        elif kind == "score":
            for ticker, fixture in markets.items():
                if fixture == str(msg["fixture"]):
                    if live_score(msg):
                        active.add(ticker)
                        ever_active.add(fixture)
                    else:
                        active.discard(ticker)
                    refresh(ticker, now)
        last, last_utc = now, utc
    if last is not None:
        work_until(last)
    for row in attempts.values():
        if row["status"] == "open":
            row["status"] = "open_at_tape_end"
    return dict(start_utc_ns=first, end_utc_ns=last_utc, events=dict(counts),
                fixtures_with_observed_live_score=sorted(ever_active), gaps=books.gaps,
                attempts=list(attempts.values()))


def totals(attempts):
    closed = [r for r in attempts if r["status"] == "closed"]
    spent_open = sum((D(r["buy"])+D(r["entry_fee"]) for r in attempts if "buy" in r and "pnl" not in r), D(0))
    pnl = sum((D(r["pnl"]) for r in closed), D(0))
    return dict(attempts=len(attempts), entered=sum("buy" in r for r in attempts), closed=len(closed),
                wins=sum(D(r["pnl"]) > 0 for r in closed), unresolved=sum("buy" in r and "pnl" not in r for r in attempts),
                open_cost_dollars=str(spent_open), gross_closed=str(sum((D(r["gross"]) for r in closed), D(0))),
                fees_closed=str(sum((D(r["fees"]) for r in closed), D(0))), pnl_closed=str(pnl),
                one_cent_worse_each_leg=str(pnl - D(".02") * len(closed)))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--forward", type=Path, required=True, help="Separate forward recording")
    parser.add_argument("--output-dir", type=Path, default=OUT, help="Destination for this screen and forward comparison")
    args = parser.parse_args()
    tapes = [ROOT / f"data/raw/first-break/{stem}.jsonl" for stem in PRIOR_TAPES] + [args.forward]
    reports, pooled, fixtures_seen = [], [], set()
    for path in tapes:
        with path.open() as stream:
            report = replay(json.loads(line) for line in stream)
        if reports and path == args.forward and report["start_utc_ns"] <= max(r["end_utc_ns"] for r in reports):
            raise ValueError("Forward recording overlaps the historical screen")
        report["tape"] = str(path)
        report["period"] = "forward" if path == args.forward else "screen"
        report["totals"] = totals(report["attempts"])
        for row in report["attempts"]:
            if row["fixture"] not in fixtures_seen:
                fixtures_seen.add(row["fixture"])
                pooled.append(dict(row, period=report["period"], tape_id=path.stem))
        reports.append(report)
    active_by_period = {period: sorted({f for r in reports if r["period"] == period
                                           for f in r["fixtures_with_observed_live_score"]})
                        for period in ("screen", "forward")}
    unique = set(active_by_period["screen"]) | set(active_by_period["forward"])
    summary = dict(rule=dict(depth_band_dollars="0.02 inclusive below each best bid", fraction="0.80",
                    spread_max="0.02", persistence_s=3, latency_s=2, holding_s=30,
                    contract_count=1, fees="quadratic taker multiplier 1, rounded up to cents on both legs",
                    activity="Set in progress and live=1 receipt required before starting the 3s timer",
                    execution="Displayed touch, size >=1; cross after 2s, market exit at bid 32s after entry",
                    pooled="First attempted signal per fixture across all captures, including misses"),
                   active_fixtures_by_period=active_by_period, independent_matches=len(unique),
                   previously_unseen_forward_matches=len(set(active_by_period["forward"])-set(active_by_period["screen"])),
                   capture_level={period: totals([row for r in reports if r["period"] == period for row in r["attempts"]])
                                  for period in ("screen","forward")},
                   deduplicated=totals(pooled), ledger=pooled)
    save(args.output_dir / "captures.json", reports)
    save(args.output_dir / "summary.json", summary)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
