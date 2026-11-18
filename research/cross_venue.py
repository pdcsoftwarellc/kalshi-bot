"""Compare Kalshi with Polymarket on the same events: coverage, price gaps and lead-lag.

Public, unauthenticated reads only. Nothing is traded.

Kalshi side: the settled-market listings and one-minute quote bars that
research/near_end.py already cached (six hours before settlement, one market
per event). Bars that are not on disk are skipped unless --kalshi-network is
given, in which case they are read one at a time, about one per second.

Polymarket side (international venue, not the US-regulated one):
  gamma-api.polymarket.com/markets   closed game-winner ("moneyline") markets by tag
  clob.polymarket.com/prices-history one-minute history of a token's MIDPOINT
  clob.polymarket.com/book           live order book (the `live` command only)
Each response is cached under data/raw/cross-venue/.

Matching. NBA: Kalshi's event ticker carries the date and both team codes
(KXNBAGAME-26OCT29SACNOP), Polymarket's slug the same (nba-sac-nop-2026-10-29).
Tennis: both player names must match (first initial and surname) on dates at
most one day apart; doubles are excluded. The Kalshi YES side is mapped to the
Polymarket outcome of the same team or player. A pair is kept only when both
venues settled the same way and the two price series are positively correlated.

Alignment. A Kalshi bar ending at T closes with the quote at T. Polymarket
history is sampled about once a minute at arbitrary seconds. "stale" uses the
last Polymarket point at or before T (what a Kalshi trader could have seen);
"interp" interpolates to T (uses a later point; descriptive lead-lag only).

Rule. When Polymarket's stale midpoint is at least X cents above Kalshi's
closing ask (or X below the bid, buying NO), buy at the NEXT bar's opening ask,
pay 0.07*p*(1-p), and either hold to settlement or sell at the opening bid of
the bar after Polymarket's midpoint is no longer above Kalshi's. One entry per
event per rule; events are split 60/40 by date.

Minute bars cannot show a lead shorter than about a minute, displayed size, or
Polymarket's spread at the time (history is midpoint only).

  python research/cross_venue.py coverage
  python research/cross_venue.py build
  python research/cross_venue.py analyze
  python research/cross_venue.py live
  python research/cross_venue.py tape      # second-level check on a few games (slow Kalshi reads)
"""
import argparse
import bisect
from collections import Counter, defaultdict
from datetime import date, datetime, timedelta
import hashlib
import json
from pathlib import Path
import random
import re
from statistics import mean, median
import sys
import time

import httpx

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import config
from research.api_cost import COLLECTION, ROOT, UTC, fetch, midnight, pages, save, timestamp, tokens

RAW = ROOT / "data/raw/cross-venue"
OUT = ROOT / "data/processed/cross-venue"
GAMMA, CLOB = "https://gamma-api.polymarket.com", "https://clob.polymarket.com"
HEADERS = {"User-Agent": "Mozilla/5.0 (research script)"}
START, END = date(2026, 9, 1), date(2026, 11, 16)      # the window near_end.py cached
WINDOW = 6*3600
# Polymarket tag ids, from gamma /sports.
TAGS = {"nba": 745, "nfl": 450, "nhl": 899, "mlb": 100381, "college basketball": 28,
        "college football": 100351, "tennis": 864, "soccer": 100350, "ufc": 279}
KALSHI = {"nba": ("KXNBAGAME",), "nfl": ("KXNFLGAME",), "nhl": ("KXNHLGAME",), "mlb": ("KXMLBGAME",),
          "college basketball": ("KXNCAAMBGAME",), "college football": ("KXNCAAFGAME",),
          "tennis": ("KXATPMATCH", "KXWTAMATCH", "KXATPCHALLENGERMATCH", "KXWTACHALLENGERMATCH")}
MONTHS = {m: i+1 for i, m in enumerate("JAN FEB MAR APR MAY JUN JUL AUG SEP OCT NOV DEC".split())}
NBA_ALIAS = {}      # Kalshi code -> Polymarket code where they differ (none found)


class Offline:
    """Stands in for a client so that a Kalshi cache miss fails instead of using the network."""
    def get(self, *a, **k):
        raise RuntimeError("not cached")


def poly(client, base, path, params):
    key = hashlib.sha256(json.dumps([base, path, params], sort_keys=True).encode()).hexdigest()
    cache = RAW / path.strip("/").replace("/", "_") / (key + ".json")
    if cache.exists():
        return json.loads(cache.read_text())
    for attempt in range(6):
        try:
            resp = client.get(base + path, params=params)
        except httpx.RequestError:
            time.sleep(2 ** attempt)
            continue
        if resp.status_code == 429 or resp.status_code >= 500:
            time.sleep(2 ** attempt)
            continue
        if resp.status_code != 200:
            raise RuntimeError(f"{path} returned HTTP {resp.status_code}: {resp.text[:200]}")
        data = resp.json()
        save(cache, data)
        time.sleep(.3)
        return data
    raise RuntimeError(f"Read failed after retries: {path}")


KEEP = ("id", "slug", "question", "outcomes", "outcomePrices", "clobTokenIds", "volume", "gameStartTime",
        "closedTime", "endDate", "conditionId", "sportsMarketType")


def poly_markets(client, tag, lo, hi, shift=0, step=4):
    """Closed game-winner markets whose game start falls in [lo, hi). Tennis end dates sit a week after the game."""
    out, day = {}, lo
    while day < hi + timedelta(days=shift):
        nxt = day + timedelta(days=step)
        offset = 0
        while True:
            page = poly(client, GAMMA, "/markets", dict(
                tag_id=tag, sports_market_types="moneyline", closed="true", limit=100, offset=offset,
                order="endDate", ascending="true", end_date_min=f"{day}T00:00:00Z", end_date_max=f"{nxt}T00:00:00Z"))
            for m in page:
                out[m["id"]] = {k: m.get(k) for k in KEEP}
            if len(page) < 100:
                break
            offset += 100
        day = nxt
    keep = []
    for m in out.values():
        try:
            m["start_ts"] = int(timestamp(m["gameStartTime"].replace(" ", "T").replace("+00", "+00:00")))
            m["outcomes"], m["outcomePrices"] = json.loads(m["outcomes"]), [float(p) for p in json.loads(m["outcomePrices"])]
            m["clobTokenIds"] = json.loads(m["clobTokenIds"])
        except (AttributeError, KeyError, TypeError, ValueError):
            continue
        if midnight(lo) <= m["start_ts"] < midnight(hi) and len(m["outcomes"]) == 2:
            keep.append(m)
    return keep


def kalshi_events(series):
    """Settled events from the cached listing: (event, yes market, other market)."""
    groups = defaultdict(list)
    try:
        for m in pages(Offline(), "/markets", "markets", series_ticker=series, status="settled", limit=1000,
                       min_close_ts=midnight(START), max_close_ts=midnight(END+timedelta(days=7))):
            groups[m["event_ticker"]].append(m)
    except RuntimeError:
        return []
    out = []
    for event, group in groups.items():
        if len(group) != 2:
            continue
        yes, other = sorted(group, key=lambda m: m["ticker"])
        try:
            outcome, settle = float(yes["settlement_value_dollars"]), int(timestamp(yes["settlement_ts"]))
        except (KeyError, TypeError, ValueError):
            continue
        if outcome in (0, 1) and midnight(START) <= settle < midnight(END):
            out.append(dict(series=series, event=event, ticker=yes["ticker"], settle=settle, outcome=outcome,
                            yes_name=yes["yes_sub_title"], other_name=other["yes_sub_title"],
                            yes_code=yes["ticker"].rsplit("-", 1)[1], other_code=other["ticker"].rsplit("-", 1)[1],
                            volume=float(yes.get("volume_fp") or 0)+float(other.get("volume_fp") or 0)))
    return out


def ticker_date(event):
    m = re.search(r"-(\d\d)([A-Z]{3})(\d\d)", event)
    return date(2000+int(m.group(1)), MONTHS[m.group(2)], int(m.group(3)))


def same_person(a, b):
    a, b = tokens(a), tokens(b)
    if not a or not b:
        return False
    return (a[-1] == b[-1] and a[0][0] == b[0][0]) or len(set(a) & set(b)) >= 2


def match(events, markets, sport):
    """Pair Kalshi events with Polymarket markets; return (pairs, reasons unmatched)."""
    pairs, miss = [], Counter()
    if sport == "nba":
        index = {}
        for m in markets:
            parts = m["slug"].split("-")
            if len(parts) == 6 and parts[0] == "nba":
                index[(date(int(parts[3]), int(parts[4]), int(parts[5])), parts[1], parts[2])] = m
        for e in events:
            day = ticker_date(e["event"])
            codes = [NBA_ALIAS.get(c, c).lower() for c in (e["yes_code"], e["other_code"])]
            hit = index.get((day, codes[0], codes[1])) or index.get((day, codes[1], codes[0]))
            if not hit:
                miss["no polymarket market"] += 1
                continue
            side = hit["slug"].split("-")[1:3].index(codes[0])
            pairs.append((e, hit, side))
    else:
        by_day = defaultdict(list)
        for m in markets:
            if "doubles" not in m["slug"]:
                by_day[datetime.fromtimestamp(m["start_ts"], UTC).date()].append(m)
        for e in events:
            day, hits = ticker_date(e["event"]), []
            for d in (day-timedelta(days=1), day, day+timedelta(days=1)):
                for m in by_day[d]:
                    for side in (0, 1):
                        if same_person(e["yes_name"], m["outcomes"][side]) and same_person(e["other_name"], m["outcomes"][1-side]):
                            hits.append((m, side))
            hits = [h for h in hits if -3600 <= e["settle"]-h[0]["start_ts"] <= 12*3600] or hits
            if len(hits) != 1:
                miss["no polymarket market" if not hits else "ambiguous"] += 1
                continue
            pairs.append((e, hits[0][0], hits[0][1]))
    return pairs, miss


def cents(bar, side, field):
    try:
        return round(100*float(bar[side].get(field+"_dollars", bar[side].get(field))))
    except (KeyError, TypeError, ValueError, AttributeError):
        return None


def kalshi_bars(e, client):
    try:
        bars = fetch(client, f"/series/{e['series']}/markets/{e['ticker']}/candlesticks",
                     dict(start_ts=e["settle"]-WINDOW, end_ts=e["settle"], period_interval=1))["candlesticks"]
    except (RuntimeError, KeyError):
        return None
    rows = []
    for b in sorted(bars, key=lambda b: b["end_period_ts"]):
        q = [cents(b, s, f) for s in ("yes_bid", "yes_ask") for f in ("open", "close")]
        if None in q or not (0 < q[0] <= q[2] < 100 and 0 < q[1] <= q[3] < 100):
            continue
        rows.append([b["end_period_ts"], *q])       # ts, bid open, bid close, ask open, ask close
    return rows


def build(args):
    RAW.mkdir(parents=True, exist_ok=True)
    kalshi = Offline()
    report, sample = {}, []
    with httpx.Client(timeout=40, headers=HEADERS) as client:
        live = httpx.Client(timeout=40) if args.kalshi_network else None
        lo, hi = START, END + timedelta(days=1)
        listings = {"nba": poly_markets(client, TAGS["nba"], lo, hi),
                    "tennis": poly_markets(client, TAGS["tennis"], lo, hi, shift=9)}
        for sport, series in (("nba", "KXNBAGAME"), ("tennis", "KXATPMATCH"), ("tennis", "KXWTAMATCH")):
            events = kalshi_events(series)
            pairs, miss = match(events, listings[sport], sport)
            for e, m, side in sorted(pairs, key=lambda p: p[0]["settle"]):
                if m["outcomePrices"][side] not in (0, 1) or m["outcomePrices"][side] != e["outcome"]:
                    miss["settled differently or void"] += 1
                    continue
                bars = kalshi_bars(e, kalshi)
                if bars is None and live:
                    bars = kalshi_bars(e, live)
                    time.sleep(1)
                if not bars or len(bars) < 30:
                    miss["no kalshi bars"] += 1
                    continue
                hist = poly(client, CLOB, "/prices-history", dict(
                    market=m["clobTokenIds"][side], startTs=e["settle"]-WINDOW, endTs=e["settle"]+600, fidelity=1))
                points = sorted((h["t"], round(100*h["p"], 2)) for h in hist.get("history", []))
                if len(points) < 30:
                    miss["no polymarket history"] += 1
                    continue
                sample.append(dict(e, sport=sport, slug=m["slug"], side=side, start=m["start_ts"],
                                   poly_volume=float(m["volume"] or 0), bars=bars, poly=points))
            report[series] = dict(kalshi_events=len(events), paired=len(pairs), unmatched=dict(miss))
            print(series, report[series], flush=True)
    OUT.mkdir(parents=True, exist_ok=True)
    save(OUT / "matched.json", sample)
    save(OUT / "matching.json", report)


def align(e):
    """Per Kalshi bar: ts, bid/ask open/close, Polymarket stale and interpolated midpoints, phase."""
    ts, ps = [p[0] for p in e["poly"]], [p[1] for p in e["poly"]]
    rows = []
    for t, bo, bc, ao, ac in e["bars"]:
        i = bisect.bisect_right(ts, t) - 1
        stale = ps[i] if i >= 0 and t - ts[i] <= 90 else None
        interp = None
        if stale is not None and i+1 < len(ts) and ts[i+1] - ts[i] <= 180:
            w = (t - ts[i]) / (ts[i+1] - ts[i]) if ts[i+1] > ts[i] else 0
            interp = ps[i] + w*(ps[i+1] - ps[i])
        rows.append(dict(t=t, bo=bo, bc=bc, ao=ao, ac=ac, mid=(bc+ac)/2, stale=stale, interp=interp,
                         live=t >= e["start"]))
    return rows


def fee(price):
    return 7*(price/100)*(1-price/100)       # cents per contract


def quantiles(values, qs=(.05, .25, .5, .75, .95)):
    v = sorted(values)
    return {f"p{int(100*q)}": v[min(len(v)-1, int(q*len(v)))] for q in qs} if v else {}


def boot(groups, stat, draws=1000, seed=11):
    """95% interval resampling whole events."""
    if len(groups) < 10:
        return None
    rng = random.Random(seed)
    vals = sorted(stat(rng.choices(groups, k=len(groups))) for _ in range(draws))
    return [round(vals[int(.025*draws)], 3), round(vals[int(.975*draws)], 3)]


def gaps(sample):
    out = {}
    for sport in ("nba", "tennis"):
        for phase in ("pre-game", "in-play", "in-play mid 10-90"):
            diffs, edges, runs, spreads, per_event = [], [], [], [], []
            for e in sample:
                if e["sport"] != sport:
                    continue
                run, mine = 0, []
                for r in e["rows"]:
                    keep = r["stale"] is not None and r["live"] == (phase != "pre-game") and \
                        (phase != "in-play mid 10-90" or 10 <= r["mid"] <= 90)
                    if not keep:
                        if run:
                            runs.append(run)
                        run = 0
                        continue
                    diffs.append(r["mid"] - r["stale"])
                    mine.append(r["mid"] - r["stale"])
                    spreads.append(r["ac"] - r["bc"])
                    edge = max(r["stale"] - r["ac"] - fee(r["ac"]), r["bc"] - r["stale"] - fee(r["bc"]))
                    edges.append(edge)
                    if edge > 0:
                        run += 1
                    elif run:
                        runs.append(run)
                        run = 0
                if run:
                    runs.append(run)
                if mine:
                    per_event.append(mine)
            if not diffs:
                continue
            absd = [abs(d) for d in diffs]
            out[f"{sport} {phase}"] = dict(
                events=len(per_event), minutes=len(diffs), mean_gap=round(mean(diffs), 3),
                mean_gap_ci=boot([(sum(ev), len(ev)) for ev in per_event],
                                 lambda g: sum(x[0] for x in g)/sum(x[1] for x in g)),
                gap=quantiles(diffs), mean_abs_gap=round(mean(absd), 3), median_abs_gap=median(absd),
                kalshi_spread_mean=round(mean(spreads), 2), kalshi_spread_median=median(spreads),
                share_abs_gap_ge={c: round(mean(a >= c for a in absd), 4) for c in (1, 2, 3, 5, 10)},
                share_beyond_quote_plus_fee={c: round(mean(x > c for x in edges), 4) for c in (0, 1, 2, 3, 5)},
                runs_beyond_cost=dict(count=len(runs), median_minutes=median(runs) if runs else None,
                                      share_1_minute=round(mean(r == 1 for r in runs), 3) if runs else None,
                                      share_ge_3=round(mean(r >= 3 for r in runs), 3) if runs else None,
                                      share_ge_10=round(mean(r >= 10 for r in runs), 3) if runs else None))
    return out


def leadlag(sample, key):
    """Which venue closes the gap. Positive = that venue moved toward the other."""
    out = {}
    for sport in ("nba", "tennis"):
        for phase in ("pre-game", "in-play"):
            for h in (1, 5, 15):
                for floor in (0, 2, 5):
                    per_event = []
                    for e in sample:
                        if e["sport"] != sport:
                            continue
                        rows, by_t, obs = e["rows"], {r["t"]: r for r in e["rows"]}, []
                        for r in rows:
                            nxt = by_t.get(r["t"] + 60*h)
                            if (nxt is None or r[key] is None or nxt[key] is None or r["live"] != (phase == "in-play")
                                    or not 5 <= r["mid"] <= 95):
                                continue
                            g = r[key] - r["mid"]
                            if abs(g) < max(floor, 1e-9):
                                continue
                            s = 1 if g > 0 else -1
                            obs.append((g, nxt["mid"] - r["mid"], nxt[key] - r[key], s))
                        if obs:      # per-event sums keep the bootstrap cheap
                            per_event.append((sum(x[0]*x[0] for x in obs), sum(x[0]*x[1] for x in obs),
                                              sum(x[0]*x[2] for x in obs), sum(x[3]*x[1] for x in obs),
                                              sum(x[3]*x[2] for x in obs), len(obs), sum(abs(x[0]) for x in obs)))
                    if len(per_event) < 10:
                        continue

                    def stats(groups):
                        gg, gk, gp, sk, sp, n = (sum(ev[i] for ev in groups) for i in range(6))
                        return gk/gg, -gp/gg, sk/n, -sp/n
                    point = stats(per_event)
                    names = ("kalshi_share_of_gap_closed", "polymarket_share_of_gap_closed",
                             "kalshi_cents_toward", "polymarket_cents_toward")
                    minutes = sum(ev[5] for ev in per_event)
                    row = dict(events=len(per_event), minutes=minutes,
                               mean_abs_gap=round(sum(ev[6] for ev in per_event)/minutes, 2))
                    for i, name in enumerate(names):
                        row[name] = round(point[i], 3)
                        row[name+"_ci"] = boot(per_event, lambda g, i=i: stats(g)[i], draws=200)
                    out[f"{sport} {phase} h={h} |gap|>={floor}"] = row
    return out


def crosscorr(sample, key):
    out = {}
    for sport in ("nba", "tennis"):
        acc = defaultdict(lambda: [0, 0.0, 0.0, 0.0, 0.0, 0.0])
        for e in sample:
            if e["sport"] != sport:
                continue
            by_t = {r["t"]: r for r in e["rows"]}
            for r in e["rows"]:
                a, b, c = by_t.get(r["t"]-60), by_t.get(r["t"]+60), r
                if (a is None or b is None or not r["live"] or not 5 <= r["mid"] <= 95
                        or None in (a[key], b[key], c[key])):
                    continue
                dk0, dk1, dp0, dp1 = c["mid"]-a["mid"], b["mid"]-c["mid"], c[key]-a[key], b[key]-c[key]
                for name, x, y in (("same minute", dk0, dp0), ("polymarket then kalshi", dp0, dk1),
                                   ("kalshi then polymarket", dk0, dp1)):
                    s = acc[name]
                    s[0] += 1; s[1] += x; s[2] += y; s[3] += x*x; s[4] += y*y; s[5] += x*y
        for name, (n, sx, sy, sxx, syy, sxy) in acc.items():
            cov, vx, vy = sxy/n - sx*sy/n/n, sxx/n - (sx/n)**2, syy/n - (sy/n)**2
            out[f"{sport} {name}"] = dict(minutes=n, correlation=round(cov/(vx*vy)**.5, 4))
    return out


def trades(sample, x, exit_rule):
    found = []
    for e in sample:
        rows = e["rows"]
        for i in range(len(rows)-1):
            r, nxt = rows[i], rows[i+1]
            if r["stale"] is None or nxt["t"] - r["t"] > 120:
                continue
            if r["stale"] - r["ac"] >= x:
                yes = True
            elif r["bc"] - r["stale"] >= x:
                yes = False
            else:
                continue
            price = nxt["ao"] if yes else 100 - nxt["bo"]
            net = 100*(e["outcome"] if yes else 1-e["outcome"]) - price - fee(price)
            held, how = (e["settle"] - nxt["t"])/60, "settlement"
            if exit_rule == "gap closes":
                for j in range(i+1, len(rows)-1):
                    q, after = rows[j], rows[j+1]
                    if q["stale"] is None or after["t"] - q["t"] > 120:
                        continue
                    if (q["stale"] <= q["mid"]) if yes else (q["stale"] >= q["mid"]):
                        sell = after["bo"] if yes else 100 - after["ao"]
                        net, held, how = sell - fee(sell) - price - fee(price), (after["t"]-nxt["t"])/60, "gap closed"
                        break
            found.append(dict(event=e["event"], sport=e["sport"], settle=e["settle"], side="yes" if yes else "no",
                              signal_gap=round((r["stale"]-r["ac"]) if yes else (r["bc"]-r["stale"]), 2),
                              signal_ask=r["ac"] if yes else 100-r["bc"], price=price, live=r["live"],
                              minutes_from_start=(r["t"]-e["start"])/60, minutes_held=held, exit=how,
                              net_cents=round(net, 3)))
            break
    return found


def score(rows):
    if not rows:
        return dict(trades=0)
    nets = [r["net_cents"] for r in rows]
    return dict(trades=len(rows), mean_entry_price=round(mean(r["price"] for r in rows), 1),
                profitable=round(mean(n > 0 for n in nets), 3), net_cents=round(mean(nets), 2),
                ci95=boot([[n] for n in nets], lambda g: mean(x[0] for x in g)),
                slipped_at_entry=round(mean(r["price"] > r["signal_ask"] for r in rows), 3),
                median_minutes_held=round(median(r["minutes_held"] for r in rows), 1))


def analyze(args):
    sample = json.loads((OUT / "matched.json").read_text())
    kept, dropped = [], Counter()
    for e in sample:
        e["rows"] = align(e)
        both = [(r["mid"], r["stale"]) for r in e["rows"] if r["stale"] is not None]
        if len(both) < 30:
            dropped["under 30 aligned minutes"] += 1
            continue
        mk, mp = mean(b[0] for b in both), mean(b[1] for b in both)
        cov = sum((a-mk)*(b-mp) for a, b in both)
        mirrored = mean(abs(a-(100-b)) for a, b in both)
        direct = mean(abs(a-b) for a, b in both)
        if cov <= 0 or mirrored < direct:
            dropped["series disagree (side check)"] += 1
            continue
        kept.append(e)
    kept.sort(key=lambda e: e["settle"])
    report = dict(sample={s: dict(events=sum(e["sport"] == s for e in kept),
                                  first=str(datetime.fromtimestamp(min(e["settle"] for e in kept if e["sport"] == s), UTC).date()),
                                  last=str(datetime.fromtimestamp(max(e["settle"] for e in kept if e["sport"] == s), UTC).date()),
                                  median_kalshi_contracts=median(e["volume"] for e in kept if e["sport"] == s),
                                  median_polymarket_usd=median(e["poly_volume"] for e in kept if e["sport"] == s))
                          for s in ("nba", "tennis")}, dropped=dict(dropped))
    report["gaps_kalshi_mid_minus_polymarket"] = gaps(kept)
    report["leadlag_interp"] = leadlag(kept, "interp")
    report["leadlag_stale"] = leadlag(kept, "stale")
    report["one_minute_change_correlation"] = dict(interp=crosscorr(kept, "interp"), stale=crosscorr(kept, "stale"))
    cut = kept[int(.6*len(kept))]["settle"]
    rule, everything = {}, []
    for x in (2, 3, 5):
        for exit_rule in ("settlement", "gap closes"):
            rows = trades(kept, x, exit_rule)
            everything += [dict(r, x=x, exit_rule=exit_rule) for r in rows]
            cells = {"all": rows, "early 60%": [r for r in rows if r["settle"] < cut],
                     "late 40%": [r for r in rows if r["settle"] >= cut],
                     "late 40% nba": [r for r in rows if r["settle"] >= cut and r["sport"] == "nba"],
                     "late 40% tennis": [r for r in rows if r["settle"] >= cut and r["sport"] == "tennis"],
                     "pre-game": [r for r in rows if not r["live"]], "in-play": [r for r in rows if r["live"]],
                     "in-play entry 10-90c": [r for r in rows if r["live"] and 10 <= r["price"] <= 90],
                     "in-play entry >90c or <10c": [r for r in rows if r["live"] and not 10 <= r["price"] <= 90],
                     "within 10 min of start": [r for r in rows if abs(r["minutes_from_start"]) <= 10]}
            rule[f"X={x} exit={exit_rule}"] = {k: score(v) for k, v in cells.items()}
    report["rule"] = rule
    report["split_date"] = str(datetime.fromtimestamp(cut, UTC).date())
    save(OUT / "summary.json", report)
    save(OUT / "trades.json", everything)
    print(json.dumps(report, indent=1))


def coverage(args):
    lo, hi = date(2026, 9, 18), date(2026, 11, 18)
    report = {}
    with httpx.Client(timeout=40, headers=HEADERS) as client:
        for sport, tag in TAGS.items():
            rows = poly_markets(client, tag, lo, hi, shift=9 if sport == "tennis" else 0,
                                step=1 if sport in ("soccer", "college basketball") else 4)
            if sport == "college basketball":      # the basketball tag also holds the NBA and foreign leagues
                rows = [m for m in rows if m["slug"].startswith("cbb-")]
            kinds = Counter(m["slug"].split("-")[0] for m in rows)
            vols = [float(m["volume"] or 0) for m in rows]
            cell = dict(polymarket_markets=len(rows), polymarket_median_usd=round(median(vols)) if vols else None,
                        polymarket_total_usd=round(sum(vols)), slug_prefixes=dict(kinds.most_common(12)))
            ks = [e for s in KALSHI.get(sport, ()) for e in kalshi_events(s)]
            ks = [e for e in ks if e["settle"] >= midnight(lo)]
            if ks:
                cell.update(kalshi_events_to_nov16=len(ks), kalshi_median_contracts=round(median(e["volume"] for e in ks)),
                            kalshi_by_series=dict(Counter(e["series"] for e in ks)))
            report[sport] = cell
            print(sport, cell, flush=True)
    OUT.mkdir(parents=True, exist_ok=True)
    save(OUT / "coverage.json", dict(window=[str(lo), str(hi)], sports=report))


def live(args):
    """Snapshot of open events on both venues: quoted spread and top-of-book size. A few slow Kalshi reads."""
    rows = []
    with httpx.Client(timeout=40, headers=HEADERS) as client, httpx.Client(timeout=40) as kalshi:
        for sport, tag, series in (("nba", 745, ("KXNBAGAME",)), ("tennis", 864, ("KXATPMATCH", "KXWTAMATCH")),
                                   ("nhl", 899, ("KXNHLGAME",))):
            pm = client.get(GAMMA + "/markets", params=dict(tag_id=tag, sports_market_types="moneyline", closed="false",
                                                            limit=100, order="volume24hr", ascending="false")).json()
            time.sleep(.3)
            open_markets = []
            for s in series:
                cursor = ""
                while True:
                    resp = kalshi.get(config.KALSHI_PROD_BASE + "/markets",
                                      params=dict(series_ticker=s, status="open", limit=1000, **({"cursor": cursor} if cursor else {})))
                    time.sleep(1.1)
                    if resp.status_code == 429:
                        time.sleep(5)
                        continue
                    data = resp.json()
                    open_markets += data["markets"]
                    cursor = data.get("cursor")
                    if not cursor:
                        break
            groups = defaultdict(list)
            for m in open_markets:
                groups[m["event_ticker"]].append(m)
            for event, group in groups.items():
                if len(group) != 2:
                    continue
                yes, other = sorted(group, key=lambda m: m["ticker"])
                for m in pm:
                    names = json.loads(m["outcomes"])
                    if sport == "tennis":
                        if "doubles" in m["slug"]:
                            continue
                        side = next((i for i in (0, 1) if same_person(yes["yes_sub_title"], names[i])
                                     and same_person(other["yes_sub_title"], names[1-i])), None)
                    else:
                        parts, codes = m["slug"].split("-"), [yes["ticker"].rsplit("-", 1)[1].lower(), other["ticker"].rsplit("-", 1)[1].lower()]
                        side = None
                        if len(parts) == 6 and set(parts[1:3]) == set(codes) and \
                                date(int(parts[3]), int(parts[4]), int(parts[5])) == ticker_date(event):
                            side = parts[1:3].index(codes[0])
                    if side is None:
                        continue
                    book = client.get(CLOB + "/book", params=dict(token_id=json.loads(m["clobTokenIds"])[side])).json()
                    time.sleep(.3)
                    bids = sorted(((float(b["price"]), float(b["size"])) for b in book.get("bids", [])), reverse=True)
                    asks = sorted((float(a["price"]), float(a["size"])) for a in book.get("asks", []))
                    if not bids or not asks:
                        continue
                    kb, ka = 100*float(yes["yes_bid_dollars"]), 100*float(yes["yes_ask_dollars"])
                    rows.append(dict(sport=sport, event=event, slug=m["slug"], start=m.get("gameStartTime"),
                                     kalshi_bid=kb, kalshi_ask=ka, kalshi_bid_size=float(yes.get("yes_bid_size_fp") or 0),
                                     kalshi_ask_size=float(yes.get("yes_ask_size_fp") or 0),
                                     kalshi_volume=float(yes.get("volume_fp") or 0)+float(other.get("volume_fp") or 0),
                                     poly_bid=round(100*bids[0][0], 2), poly_ask=round(100*asks[0][0], 2),
                                     poly_bid_size=bids[0][1], poly_ask_size=asks[0][1], poly_volume=float(m.get("volume") or 0),
                                     mid_gap=round((kb+ka)/2 - 50*(bids[0][0]+asks[0][0]), 2)))
                    break
    OUT.mkdir(parents=True, exist_ok=True)
    save(OUT / f"live-{int(time.time())}.json", rows)
    for sport in sorted({r["sport"] for r in rows}):
        mine = [r for r in rows if r["sport"] == sport]
        print(sport, dict(pairs=len(mine), kalshi_spread_median=median(r["kalshi_ask"]-r["kalshi_bid"] for r in mine),
                          poly_spread_median=round(median(r["poly_ask"]-r["poly_bid"] for r in mine), 2),
                          kalshi_top_size_median=median(min(r["kalshi_bid_size"], r["kalshi_ask_size"]) for r in mine),
                          poly_top_size_median=median(min(r["poly_bid_size"], r["poly_ask_size"]) for r in mine),
                          kalshi_volume_median=median(r["kalshi_volume"] for r in mine),
                          poly_volume_median=round(median(r["poly_volume"] for r in mine)),
                          abs_mid_gap_median=median(abs(r["mid_gap"]) for r in mine),
                          abs_mid_gap_max=max(abs(r["mid_gap"]) for r in mine)))


class Paced:
    """Kalshi client for uncached reads: one at a time, a little over a second apart."""
    def __init__(self):
        self.client = httpx.Client(timeout=40)

    def get(self, *a, **k):
        time.sleep(1.2)
        return self.client.get(*a, **k)


def last_price(trades, grid):
    """Last traded price at each grid second (None before the first trade)."""
    out, i, last = [], 0, None
    for t in grid:
        while i < len(trades) and trades[i][0] <= t:
            last = trades[i][1]
            i += 1
        out.append(last)
    return out


def corr(xs, ys):
    pairs = [(x, y) for x, y in zip(xs, ys) if x is not None and y is not None]
    if len(pairs) < 50:
        return None
    mx, my = mean(p[0] for p in pairs), mean(p[1] for p in pairs)
    vx, vy = sum((p[0]-mx)**2 for p in pairs), sum((p[1]-my)**2 for p in pairs)
    return sum((p[0]-mx)*(p[1]-my) for p in pairs)/(vx*vy)**.5 if vx and vy else None


def tape(args):
    """Second-level check on a few games using both venues' public trade tapes.

    Asks (1) at what offset Polymarket's one-minute history best matches its own trades,
    i.e. how old a history point really is, and (2) at what lag Kalshi's and Polymarket's
    30-second price changes correlate best. Polymarket trade times are block times, a few
    seconds after the match; Kalshi's are matching-engine times.
    """
    sample = [e for e in json.loads((OUT / "matched.json").read_text()) if e["sport"] == "nba"]
    sample = sorted(sample, key=lambda e: e["settle"])[-args.games:]
    kalshi, step, span = Paced(), 5, 30
    lags = list(range(-120, 125, 5))
    rows, by_lag, hist_lag, bar_lag = [], defaultdict(list), defaultdict(list), defaultdict(list)
    tape_gaps, gap_runs = [], []
    with httpx.Client(timeout=40, headers=HEADERS) as client:
        listing = {m["slug"]: m for m in poly_markets(client, TAGS["nba"], START, END + timedelta(days=1))}
        for e in sample:
            m = listing[e["slug"]]
            pm, cursor = [], None
            while True:
                page = poly(client, "https://data-api.polymarket.com", "/v2/trades",
                            dict(condition=m["conditionId"], limit=1000, **({"cursor": cursor} if cursor else {})))
                for t in page["data"]:
                    price = 100*t["price"] if t["token_id"] == m["clobTokenIds"][e["side"]] else 100-100*t["price"]
                    pm.append((t["timestamp"], price))
                cursor = page["pagination"].get("next_cursor")
                if not cursor or (page["data"] and page["data"][-1]["timestamp"] < e["start"]):
                    break
            ks = [(timestamp(t["created_time"]), 100*float(t["yes_price_dollars"]))
                  for t in pages(kalshi, "/markets/trades", "trades", ticker=e["ticker"], min_ts=e["start"],
                                 max_ts=e["settle"], limit=1000)]
            pm, ks = sorted(pm), sorted(ks)
            grid = list(range(e["start"] + 900, e["settle"] - 300, step))
            k, p = last_price(ks, grid), last_price(pm, grid)
            n, hop = len(grid), span // step
            live = [i for i in range(n - hop) if k[i] is not None and p[i] is not None and 5 <= k[i] <= 95]
            dk = {i: k[i+hop] - k[i] for i in live}
            dp = {i: p[i+hop] - p[i] for i in live}
            best = None
            for lag in lags:      # positive lag: Polymarket's change comes later than Kalshi's
                c = corr([dk[i] for i in live if i + lag//step in dp], [dp[i + lag//step] for i in live if i + lag//step in dp])
                if c is not None:
                    by_lag[lag].append(c)
                    best = max(best or (c, lag), (c, lag))
            # Gap between the venues' last trades, Polymarket read one grid step later to undo its block-time delay.
            run = 0
            for i in live:
                if i + 1 < n and p[i+1] is not None:
                    d = abs(k[i] - p[i+1])
                    tape_gaps.append(d)
                    if d >= 3:
                        run += step
                        continue
                if run:
                    gap_runs.append(run)
                run = 0
            # How old is a history point? Compare it with Polymarket's own last trade `age` seconds earlier.
            ages = {}
            for age in range(-30, 125, 5):
                diffs = []
                for t, price in e["poly"]:
                    if e["start"] + 900 <= t < e["settle"] - 300 and 5 <= price <= 95:
                        j = bisect.bisect_right(pm, (t - age, 1e9)) - 1
                        if j >= 0:
                            diffs.append(abs(price - pm[j][1]))
                if diffs:
                    ages[age] = mean(diffs)
                    hist_lag[age].append(mean(diffs))
            # Same question for Kalshi's bars: does the close of the bar stamped T match Kalshi trades at T?
            for age in range(-90, 95, 5):
                diffs = []
                for t, bo, bc, ao, ac in e["bars"]:
                    if e["start"] + 900 <= t < e["settle"] - 300 and 5 <= bc and ac <= 95:
                        j = bisect.bisect_right(ks, (t - age, 1e9)) - 1
                        if j >= 0:
                            diffs.append(abs((bc+ac)/2 - ks[j][1]))
                if diffs:
                    bar_lag[age].append(mean(diffs))
            rows.append(dict(event=e["event"], kalshi_trades=len(ks), polymarket_trades=len(pm),
                             best_lag_seconds=best[1] if best else None, best_correlation=round(best[0], 3) if best else None,
                             history_best_age_seconds=min(ages, key=ages.get) if ages else None))
            print(rows[-1], flush=True)
    report = dict(games=rows,
                  last_trade_gap=dict(seconds=len(tape_gaps)*step, mean_abs=round(mean(tape_gaps), 2), **quantiles(tape_gaps, (.5, .9, .99)),
                                      share_ge={c: round(mean(g >= c for g in tape_gaps), 4) for c in (2, 3, 5)},
                                      runs_ge_3c=dict(count=len(gap_runs), median_seconds=median(gap_runs) if gap_runs else None,
                                                      share_ge_30s=round(mean(r >= 30 for r in gap_runs), 3) if gap_runs else None,
                                                      share_ge_60s=round(mean(r >= 60 for r in gap_runs), 3) if gap_runs else None)),
                  mean_correlation_by_lag={lag: round(mean(v), 3) for lag, v in sorted(by_lag.items())},
                  history_mean_abs_error_by_age={a: round(mean(v), 3) for a, v in sorted(hist_lag.items())},
                  kalshi_bar_close_mean_abs_error_by_age={a: round(mean(v), 3) for a, v in sorted(bar_lag.items())})
    save(OUT / "tape.json", report)
    print(json.dumps(report, indent=1))


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("command", choices=("coverage", "build", "analyze", "live", "tape"))
    parser.add_argument("--kalshi-network", action="store_true", help="read uncached Kalshi bars, one per second")
    parser.add_argument("--games", type=int, default=6, help="games for the tape command")
    args = parser.parse_args()
    dict(coverage=coverage, build=build, analyze=analyze, live=live, tape=tape)[args.command](args)


if __name__ == "__main__":
    main()
