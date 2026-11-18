"""Screen other Kalshi sports series with the price-only late rule from near_end.py.

Rule: when a side's ask closes a minute at or above a threshold (at most 99c), buy
at the next minute's opening ask, pay the taker fee, hold to settlement. One entry
per event per threshold. Events are ordered by settlement time and the rule is
reported separately on the earliest 60% and the latest 40%; thresholds are fixed,
so nothing is selected.

Two-outcome events use one canonical market and both of its sides (as near_end
does). Events with three or more markets (soccer: home, away, tie) are evaluated
on the YES side of every outcome market, keeping the earliest entry per event;
"any" additionally allows the NO side of each outcome.

Public Kalshi reads only, at most two requests in flight, with a wall-clock budget.
"""
import argparse
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from datetime import date, datetime, timezone
import json
from pathlib import Path
import re
from statistics import mean, median
import sys
import time

import httpx

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from research.api_cost import ROOT, fetch, midnight, pages, save, timestamp
from research.near_end import WINDOW, entries, interval

OUT = ROOT / "data/processed/sports-survey"
CUTOFF = date(2026, 9, 18)      # markets settled earlier live on /historical
START, END = date(2026, 9, 1), date(2026, 11, 18)
THRESHOLDS = (95, 97, 98)
WORKERS = 2
COVERED = {"KXNBAGAME", "KXMLBGAME", "KXNFLGAME", "KXNHLGAME", "KXNCAAMBGAME"}
EXTRA = {"KXBOXING", "KXPGAH2H"}


def candidates(client, floor):
    resp = client.get(__import__("config").KALSHI_PROD_BASE+"/series", params=dict(category="Sports", include_volume="true"))
    resp.raise_for_status()
    out = []
    for s in resp.json()["series"]:
        tags = s.get("tags") or []
        named = re.search(r"(GAME|MATCH|FIGHT)$", s["ticker"]) and "EXACT" not in s["ticker"]
        niche = any(t in ("Darts", "Table Tennis", "Boxing", "Rugby", "Volleyball", "Squash", "Padel") for t in tags)
        if s["ticker"] in COVERED or "Tennis" in tags or not (named or niche or s["ticker"] in EXTRA):
            continue
        if float(s.get("volume_fp") or 0) >= floor:
            out.append(dict(ticker=s["ticker"], title=s["title"], tags=tags, all_time_volume=float(s["volume_fp"]),
                            fee_multiplier=s.get("fee_multiplier")))
    return sorted(out, key=lambda s: -s["all_time_volume"])


def listing(client, series, archived=False):
    """Settled markets grouped by event, with settlement inside [START, END)."""
    if archived:
        rows = pages(client, "/historical/markets", "markets", series_ticker=series, limit=1000)
    else:
        rows = pages(client, "/markets", "markets", series_ticker=series, status="settled", limit=1000,
                     min_close_ts=midnight(CUTOFF)-7*86400, max_close_ts=midnight(END)+7*86400)
    events = defaultdict(list)
    for m in rows:
        try:
            m["_settle"], m["_outcome"] = int(timestamp(m["settlement_ts"])), float(m["settlement_value_dollars"])
        except (KeyError, TypeError, ValueError):
            continue
        if midnight(START) <= m["_settle"] < midnight(END):
            m["_archived"] = archived
            events[m["event_ticker"]].append(m)
    return events


def discover(client, floor):
    report = []
    series = candidates(client, floor)

    def count(s):
        try:
            events = listing(client, s["ticker"])
        except RuntimeError as exc:
            return dict(s, error=str(exc))
        settles = [min(m["_settle"] for m in g) for g in events.values()]
        iso = lambda ts: datetime.fromtimestamp(ts, timezone.utc).isoformat()[:10]
        return dict(s, events=len(events), markets=sum(map(len, events.values())),
                    volume=sum(float(m.get("volume_fp") or m.get("volume") or 0) for g in events.values() for m in g),
                    first=iso(min(settles)) if settles else None, last=iso(max(settles)) if settles else None)
    with ThreadPoolExecutor(WORKERS) as pool:
        for row in pool.map(count, series):
            report.append(row)
            print(row["ticker"], row.get("events"), round(row.get("volume", 0)/1e6, 1), row.get("first"), row.get("last"), flush=True)
    report.sort(key=lambda r: -r.get("volume", 0))
    save(OUT/"discovery.json", report)


def bars(client, series, m):
    path = (f"/historical/markets/{m['ticker']}/candlesticks" if m["_archived"]
            else f"/series/{series}/markets/{m['ticker']}/candlesticks")
    params = dict(start_ts=m["_settle"]-WINDOW, end_ts=m["_settle"], period_interval=1)
    for pause in (20, None):
        try:
            return fetch(client, path, params)["candlesticks"]
        except KeyError:
            return None
        except RuntimeError:      # fetch already backed off five times on 429/5xx
            if pause is None:
                return None
            time.sleep(pause)


def collect(client, name, series_list, cap, deadline, archived, budget=None):
    """Entries for up to `cap` most recent events of a family; stops submitting at the deadline."""
    events = []
    for series in series_list:
        found = {} if archived == "only" else listing(client, series)
        if archived:
            try:
                for event, group in listing(client, series, archived=True).items():
                    found.setdefault(event, group)
            except RuntimeError as exc:
                print(name, series, "archive listing failed", exc, flush=True)
        events += [(min(m["_settle"] for m in g), series, e, sorted(g, key=lambda m: m["ticker"])) for e, g in found.items()]
    events.sort()
    events = events[-cap:]
    jobs = []
    for settle, series, event, group in reversed(events):      # newest first, so a cut-off run keeps a contiguous recent block
        for m in (group if len(group) >= 3 else group[:1]):
            jobs.append((settle, series, event, len(group), m))

    if budget:
        deadline = min(deadline, time.time()+60*budget)

    def work(job):
        if time.time() > deadline:
            return "late"
        return bars(client, job[1], job[4])
    rows, skip, done = [], Counter(), {}
    with ThreadPoolExecutor(WORKERS) as pool:
        for (settle, series, event, size, m), candles in zip(jobs, pool.map(work, jobs)):
            info = done.setdefault(event, dict(event=event, series=series, settle=settle, markets=size, ok=0, late=False))
            if candles == "late":
                info["late"] = True
                continue
            if not candles:
                skip["no_bars"] += 1
                continue
            info["ok"] += 1
            expected = m.get("expected_expiration_time") or m.get("occurrence_datetime")
            for e in entries(candles, m["_settle"], m["_outcome"], int(timestamp(expected)) if expected else None):
                if e["threshold"] in THRESHOLDS:
                    rows.append(dict(e, family=name, series=series, event=event, ticker=m["ticker"], markets=size,
                                     event_settle=settle, outcome_label=m.get("yes_sub_title")))
    kept = [v for v in done.values() if not v["late"] and v["ok"] == (v["markets"] if v["markets"] >= 3 else 1)]
    names = {v["event"] for v in kept}
    return [r for r in rows if r["event"] in names], kept, dict(skip, listed=len(events), complete=len(kept),
                                                                 cut_by_deadline=sum(v["late"] for v in done.values()))


def one_per_event(rows, threshold, any_side=False):
    """Earliest entry per event; three-way events use YES sides only unless any_side."""
    best = {}
    for r in rows:
        if r["threshold"] != threshold or (r["markets"] >= 3 and r["side"] == "no" and not any_side):
            continue
        key = (r["ts"], r["ticker"], r["side"])
        if r["event"] not in best or key < best[r["event"]][0]:
            best[r["event"]] = (key, r)
    return [r for _, r in best.values()]


def stats(trades, events):
    if not events:
        return None
    weeks = max((max(e["settle"] for e in events)-min(e["settle"] for e in events))/604800, 1/7)
    out = dict(events=len(events), trades=len(trades), weeks=weeks, trades_per_week=len(trades)/weeks)
    if trades:
        price = mean(r["price"] for r in trades)/100
        out.update(losses=sum(r["payout"] < .5 for r in trades), win_rate=mean(r["payout"] for r in trades),
                   breakeven_win_rate=price+.07*price*(1-price), mean_price_cents=100*price,
                   net_cents=mean(r["net_cents"] for r in trades),
                   ci95=interval({r["event"]: [r["net_cents"]] for r in trades}),
                   median_minute_volume=median(r["minute_volume"] for r in trades),
                   median_minutes_to_settlement=median(r["minutes_to_settlement"] for r in trades))
    return out


def verdict(early, late):
    if not early or not late or not early.get("trades") or not late.get("trades"):
        return "too thin"
    thin = late["trades"] < 30 or late["trades_per_week"] < 3 or late["median_minute_volume"] < 50
    if early["net_cents"] > 0 and late["net_cents"] > 0:
        if late["ci95"] and late["ci95"][0] > 0:
            word = "passes"
        else:
            word = "closer look"
    else:
        word = "fails"
    return word + (" (thin)" if thin else "")


def summarise(rows, events):
    events = sorted(events, key=lambda e: (e["settle"], e["event"]))
    cut = int(.6*len(events))
    halves = dict(early=events[:cut], late=events[cut:])
    out = dict(events=len(events), first=events[0]["settle"] if events else None, last=events[-1]["settle"] if events else None,
               three_way_events=sum(e["markets"] >= 3 for e in events), rules={})
    for variant in ("yes", "any"):
        if variant == "any" and not out["three_way_events"]:
            continue
        for t in THRESHOLDS:
            picked = one_per_event(rows, t, variant == "any")
            row = {}
            for half, group in halves.items():
                names = {e["event"] for e in group}
                row[half] = stats([r for r in picked if r["event"] in names], group)
            row["verdict"] = verdict(row["early"], row["late"])
            hours = Counter(datetime.fromtimestamp(r["ts"], timezone.utc).hour for r in picked)
            row["entry_hours_utc"] = {str(h): hours[h] for h in sorted(hours)}
            out["rules"][f"{variant}>={t}"] = row
    return out


def show(name, summary):
    iso = lambda ts: datetime.fromtimestamp(ts, timezone.utc).isoformat()[:10]
    if not summary["events"]:
        print(name, "no events")
        return
    print(f'{name}: {summary["events"]} events {iso(summary["first"])}..{iso(summary["last"])} three-way {summary["three_way_events"]}')
    for rule, row in summary["rules"].items():
        cells = []
        for half in ("early", "late"):
            s = row[half]
            if not s or not s.get("trades"):
                cells.append(f"{half}: ev {s['events'] if s else 0} no trades")
                continue
            ci = s["ci95"] and [round(v, 2) for v in s["ci95"]]
            cells.append(f'{half}: ev {s["events"]} n {s["trades"]} L {s["losses"]} win {s["win_rate"]:.4f} need {s["breakeven_win_rate"]:.4f} '
                         f'net {s["net_cents"]:+.2f} ci {ci} vol {s["median_minute_volume"]:.0f} /wk {s["trades_per_week"]:.1f}')
        print(f'  {rule:8} {" | ".join(cells)} => {row["verdict"]}', flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="cmd", required=True)
    d = sub.add_parser("discover")
    d.add_argument("--floor", type=float, default=8e6, help="minimum all-time series volume")
    r = sub.add_parser("run")
    r.add_argument("--families", required=True, help="JSON file: {family: [series, ...]} in priority order")
    r.add_argument("--cap", type=int, default=600)
    r.add_argument("--three-way-cap", type=int, default=300, help="event cap when a family has tie markets")
    r.add_argument("--minutes", type=float, default=35)
    r.add_argument("--archived", nargs="*", default=[], help="families that also read /historical listings")
    r.add_argument("--archive-only", action="store_true", help="read only /historical listings (holdout on older games)")
    r.add_argument("--start", type=date.fromisoformat, default=START)
    r.add_argument("--end", type=date.fromisoformat, default=END)
    r.add_argument("--name", default="survey")
    args = parser.parse_args()
    if args.cmd == "run":
        globals().update(START=args.start, END=args.end)
    with httpx.Client(timeout=40) as client:
        if args.cmd == "discover":
            return discover(client, args.floor)
        families = json.loads(Path(args.families).read_text())
        deadline, report, everything = time.time()+60*args.minutes, {}, []
        for name, spec in families.items():
            series_list, cap = (spec, args.cap) if isinstance(spec, list) else (spec["series"], spec.get("cap", args.cap))
            budget = None if isinstance(spec, list) else spec.get("minutes")
            began = time.time()
            rows, events, cover = collect(client, name, series_list, cap, deadline,
                                          "only" if args.archive_only else name in args.archived, budget)
            report[name] = dict(series=series_list, coverage=cover, seconds=round(time.time()-began), **summarise(rows, events))
            everything += rows
            save(OUT/f"{args.name}.json", report)
            save(OUT/f"{args.name}-trades.json", everything)
            print(cover, f"{time.time()-began:.0f}s", flush=True)
            show(name, report[name])


if __name__ == "__main__":
    main()
