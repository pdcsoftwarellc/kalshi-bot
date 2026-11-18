"""The tennis dip rule on team events, exhibitions and doubles (the December off-season question).

Same simulation as favorite_dip.py (its `orders` and `score` are reused): the
pre-match price is the median midpoint of the first 15 one-minute bars of the
six hours before settlement and must be stable; a bid fills only when a trade
prints strictly below it in a later minute; fills are held to settlement.
One order set per match. Bids tested for 90c+ favourites: 80, 70, 65 and
10c below the pre-match price (rounded down); for 80-90c favourites, 10c below.

Every Kalshi request goes through one sequential loop paced to about one per
second (cache hits are free), because a live trader shares the rate limit.

  python research/dip_team_events.py            # listings, bars, tables
  python research/dip_team_events.py --open     # also list open/upcoming events (not cached)
"""
import argparse
from collections import defaultdict
from datetime import date, datetime, timezone
import hashlib
import json
from pathlib import Path
import sys
import time

import httpx

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import config
from research import favorite_dip
from research.api_cost import COLLECTION, ROOT, fetch, midnight, save, timestamp
from research.near_end import WINDOW

OUT = ROOT / "data/processed/favorite-dip"
GROUPS = {
    "team": ("KXUNITEDCUPMATCH", "KXDAVISCUPMATCH", "KXLAVERCUPMATCH", "KXLAVERCUPDOUBLESMATCH"),
    "exhibition": ("KXSIXKINGSMATCH", "KXSIXKINGSSLAMMATCH", "KXSIXKINGSSEMI", "KXSIXKINGSQUARTER", "KXTENNISEXHIBITION",
                   "KXEXHIBITIONMEN", "KXEXHIBITIONWOMEN", "KXBATTLEOFSEXES", "KXATPNEXTGEN"),
    "doubles": ("KXATPDOUBLES", "KXWTADOUBLES", "KXATPCHALLENGERDOUBLES", "KXITFDOUBLES", "KXITFWDOUBLES", "KXMIXEDDOUBLESMATCH"),
}
CUTOFF = midnight(date(2026, 9, 18))        # Kalshi's historical cutoff
TODAY = midnight(date(2026, 11, 19))
FIXED, RELATIVE = (80, 70, 65), 10
favorite_dip.LIMITS = tuple(range(40, 92))   # every whole-cent level, so the relative bid can be picked out


def paced(client, path, params):
    """research.api_cost.fetch, one request at a time, about one per second, with a long back-off on failure."""
    key = hashlib.sha256(json.dumps([path, params], sort_keys=True).encode()).hexdigest()
    if (COLLECTION / (key + ".json")).exists():
        return fetch(client, path, params)
    for attempt in range(4):
        time.sleep(1.1)
        try:
            return fetch(client, path, params)
        except RuntimeError as error:
            if "HTTP 4" in str(error):          # 400/404: no such market or no bars
                raise
            time.sleep(30*(attempt+1))          # fetch already retried 429/5xx five times
    raise RuntimeError(f"Read failed: {path}")


def listing(client, path, **params):
    cursor = ""
    while True:
        data = paced(client, path, dict(params, cursor=cursor) if cursor else params)
        yield from data["markets"]
        cursor = data.get("cursor")
        if not cursor:
            return


def volume(m):
    for field in ("volume_fp", "volume"):
        try:
            return float(m[field])
        except (KeyError, TypeError, ValueError):
            pass
    return 0.0


def settled(client, series):
    """Settled matches of one series from the archive and the recent listing: (job, source, volume, title)."""
    events = defaultdict(list)
    for source, rows in (("archive", listing(client, "/historical/markets", series_ticker=series, limit=1000)),
                         ("recent", listing(client, "/markets", series_ticker=series, status="settled", limit=1000,
                                            max_close_ts=TODAY))):
        for m in rows:
            events[m["event_ticker"]].append((source, m))
    for event, group in sorted(events.items()):
        unique = {m["ticker"]: (source, m) for source, m in group}
        source, m = unique[min(unique)]
        try:
            outcome, settle = float(m["settlement_value_dollars"]), int(timestamp(m["settlement_ts"]))
        except (KeyError, TypeError, ValueError):
            continue
        yield (event, m["ticker"], settle, outcome), source, sum(volume(x) for _, x in unique.values()), m.get("title", "")


def bars(client, series, job, source):
    paths = ["/historical/markets/{t}/candlesticks", "/series/"+series+"/markets/{t}/candlesticks"]
    if source == "recent" and job[2] >= CUTOFF:
        paths.reverse()
    for path in paths:
        try:
            candles = paced(client, path.format(t=job[1]),
                            dict(start_ts=job[2]-WINDOW, end_ts=job[2], period_interval=1)).get("candlesticks")
        except RuntimeError:
            continue
        if candles:
            return sorted(candles, key=lambda b: b["end_period_ts"])
    return None


def pick(rows):
    """From one match's rows at every level, the bids under test, labelled by rule."""
    if not rows:
        return []
    start = rows[0]["start"]
    relative = int(start)-RELATIVE
    wanted = {}
    if start >= 90:
        wanted = {f"90+ bid {limit}": limit for limit in FIXED}
        wanted["90+ bid start-10"] = relative
    elif start >= 80:
        wanted = {"80-90 bid start-10": relative}
    by_limit = {r["limit"]: r for r in rows}
    return [dict(by_limit[limit], rule=rule) for rule, limit in wanted.items() if limit in by_limit]


def table(rows):
    out = {}
    for rule in ("90+ bid 80", "90+ bid 70", "90+ bid 65", "90+ bid start-10", "80-90 bid start-10"):
        sub = [r for r in rows if r["rule"] == rule]
        if not sub:
            continue
        filled = [r for r in sub if r["strict"]]
        cell = dict(orders=len(sub), fills=len(filled), touches=sum(r["touch"] for r in sub))
        if filled:
            # `score` assumes one limit; the relative bid varies, so net is computed per row and the bootstrap reused.
            nets = [100*r["payout"]-r["limit"] for r in filled]
            cell.update(win_rate=sum(r["payout"] for r in filled)/len(filled), needed=sum(r["limit"] for r in filled)/len(filled)/100,
                        net_cents=sum(nets)/len(nets), ci95=favorite_dip.interval(nets), losses=sum(r["payout"] < .5 for r in filled),
                        profit_per_order_cents=sum(nets)/len(sub))
            if "start" not in rule:
                check = favorite_dip.score(sub, int(rule[-2:]))
                assert abs(check["net_cents"]-cell["net_cents"]) < 1e-9 and check["fills"] == cell["fills"]
        out[rule] = cell
    return out


def show(name, cells):
    for rule, s in cells.items():
        line = f"  {name:26} {rule:19} orders {s['orders']:4} fills {s['fills']:3} touch {s['touches']:3}"
        if s["fills"]:
            ci = s["ci95"] and [round(v, 1) for v in s["ci95"]]
            line += (f" win {s['win_rate']:.3f} need {s['needed']:.2f} net {s['net_cents']:+.1f}c ci {ci}"
                     f" losses {s['losses']} per order {s['profit_per_order_cents']:+.2f}c")
        print(line)


def day(ts):
    return datetime.fromtimestamp(ts, timezone.utc).date().isoformat()


def open_events(client):
    """Open and unopened events per series, straight from the API (never cached: the answer changes daily)."""
    out = {}
    for series in sum(GROUPS.values(), ()):
        found = []
        for status in ("open", "unopened"):
            time.sleep(1.1)
            resp = client.get(config.KALSHI_PROD_BASE+"/events", params=dict(series_ticker=series, status=status, limit=200))
            if resp.status_code == 429:
                time.sleep(30)
                resp = client.get(config.KALSHI_PROD_BASE+"/events", params=dict(series_ticker=series, status=status, limit=200))
            if resp.status_code == 200:
                found += [dict(event=e["event_ticker"], title=e.get("title"), sub_title=e.get("sub_title"), status=status,
                               strike_date=e.get("strike_date")) for e in resp.json().get("events", [])]
        out[series] = found
        print(f"  {series:26} open/unopened events {len(found)}", [e["event"] for e in found[:6]])
    save(OUT/"team-events-open.json", out)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--open", action="store_true", help="also list open and unopened events per series")
    parser.add_argument("--groups", nargs="+", choices=tuple(GROUPS), default=list(GROUPS))
    args = parser.parse_args()
    report, every = {}, []
    with httpx.Client(timeout=40) as client:
        for group in args.groups:
            for series in GROUPS[group]:
                found = list(settled(client, series))
                matches, rows = [], []
                for n, (job, source, traded, title) in enumerate(found):
                    candles = bars(client, series, job, source)
                    picked = pick(favorite_dip.orders(job, candles)) if candles else []
                    all_levels = favorite_dip.orders(job, candles) if candles else []
                    matches.append(dict(event=job[0], date=day(job[2]), source=source, volume=traded, title=title,
                                        bars=len(candles or []), start=all_levels[0]["start"] if all_levels else None))
                    rows += [dict(r, series=series, group=group) for r in picked]
                    if n % 100 == 99:
                        print(f"    {series} {n+1}/{len(found)}", flush=True)
                every += rows
                volumes = sorted(m["volume"] for m in matches)
                info = dict(group=group, settled_matches=len(matches),
                            archive=sum(m["source"] == "archive" for m in matches), recent=sum(m["source"] == "recent" for m in matches),
                            first=min((m["date"] for m in matches), default=None), last=max((m["date"] for m in matches), default=None),
                            by_month=dict(sorted((k, sum(m["date"][:7] == k for m in matches)) for k in {m["date"][:7] for m in matches})),
                            median_volume=volumes[len(volumes)//2] if volumes else None,
                            mean_volume=sum(volumes)/len(volumes) if volumes else None,
                            with_bars=sum(m["bars"] > 0 for m in matches),
                            clean_pregame=sum(m["start"] is not None for m in matches),
                            favourite_90=sum((m["start"] or 0) >= 90 for m in matches),
                            favourite_80_90=sum(80 <= (m["start"] or 0) < 90 for m in matches),
                            table=table(rows), matches=matches)
                report[series] = info
                print(f"\n{series} ({group}): {info['settled_matches']} settled (archive {info['archive']}, recent {info['recent']}), "
                      f"{info['first']} to {info['last']}, median volume {info['median_volume']}, bars {info['with_bars']}, "
                      f"clean pre-match {info['clean_pregame']}, 90c+ {info['favourite_90']}, 80-90c {info['favourite_80_90']}", flush=True)
                print("   by month:", info["by_month"])
                show(series, info["table"])
        print("\nPOOLED")
        report["pooled"] = {}
        for group in args.groups:
            report["pooled"][group] = table([r for r in every if r["group"] == group])
            show("ALL "+group, report["pooled"][group])
        both = [r for r in every if r["group"] in ("team", "exhibition")]
        report["pooled"]["team+exhibition"] = table(both)
        show("ALL team+exhibition", report["pooled"]["team+exhibition"])
        report["pooled"]["everything"] = table(every)
        show("EVERYTHING", report["pooled"]["everything"])
        save(OUT/"team-events-orders.json", every)
        save(OUT/"team-events-summary.json", report)
        if args.open:
            print("\nOPEN AND UPCOMING")
            open_events(client)


if __name__ == "__main__":
    main()
