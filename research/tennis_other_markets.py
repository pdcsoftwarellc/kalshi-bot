"""Tennis markets other than the match winner: inventory, calibration, consistency, settlement rules.

Everything so far in this repository studied the match-winner series. Kalshi also lists, per ATP / WTA main-tour
match, an exact set score (KXATPEXACTMATCH: 2-0, 2-1 for each player; best-of-five 3-0, 3-1, 3-2), a winner of
set 1 and of set 2 (KX{ATP,WTA}SETWINNER), three total-games strikes (KX{ATP,WTA}GTOTAL) and three game-spread
strikes (KXATPGSPREAD), plus tournament-winner outrights (KXATP, KXWTA and one-off slam series). Challenger and
ITF matches have no side markets.

Commands

    inventory  one GET /series/{ticker} and the settled listings (recent tier since 2026-09-01 and the /historical
               archive) for every tennis-tagged series that is not a plain match winner and has traded at least
               --floor contracts. Writes inventory.json (structure, fee type, events, volume, rules text).
    download   recent tier only (settled after 2026-09-18), batch endpoint GET /markets/candlesticks. Per match:
               all exact-score markets, the first market of the set-1 and set-2 events, all game-total and
               game-spread strikes, from 60 minutes before the reference time to settlement (one-minute bars).
               The match-winner bars come from the cache written by earlier scripts. The request list is saved
               (requests-{tour}.json) so `report` re-reads exactly those cached responses.
    archive    /historical per-market reads (no batch endpoint there): exact-score markets of the most recent
               --cap archived matches, 90 minutes up to the reference time only (pre-match calibration holdout).
    outrights  tournament-winner events of KXATP and KXWTA: hourly bars from open to the first elimination.
    report     calibration tables, consistency with the match-winner book, retirement settlement. No network.

Definitions

* Reference time T ("pre-match"): the start of play is detected on the match-winner book as in
  favourite_gap.detect_start (first bar after a 15-bar warm-up where the midpoint moved more than 2c in ten
  minutes, or open interest jumped); T is 12 minutes before that bar. A quote at T is the close of the last
  one-minute bar at or before T (bars exist only for minutes in which the book or tape changed, so the last
  bar is the standing quote); it must be two-sided (0 < bid <= ask < 100) and at most MAX_SPREAD wide.
* Calibration row = one market. Gap = 100 x settlement value - midpoint (fractional settlements count as
  fractions). Intervals: bootstrap over matches (outcomes inside a match are dependent). Tradeable versions:
  "buy YES" pays the ask plus 0.07*p*(1-p); "buy NO" (sell the outcome) pays 100 - bid plus the fee on that.
* Outrights have no start time: reference times are 24h and 3h before the FIRST market of the event closed
  (a player is closed out when eliminated, so the first close is the end of the first match of the event).
* Consistency: with the forward-filled one-minute quotes of the four exact-score markets and the match-winner
  market, taker baskets that pay at least 100c in a completed match (buy all four YES; buy all four NO;
  exact pair of one player + the other player's match contract; and the NO-side mirror images), net of the
  fee on every leg. A retirement breaks the cross-book baskets (see `retirements` in the report).

Network use: public reads only, one request at a time through calibration_map.get (paced 1.1 s, long pause on
HTTP 429), all cached under data/raw/cost-research-v1. No orders, no authentication.

    .venv/bin/python research/tennis_other_markets.py inventory
    .venv/bin/python research/tennis_other_markets.py download --minutes 25
    .venv/bin/python research/tennis_other_markets.py archive --minutes 20 --cap 300
    .venv/bin/python research/tennis_other_markets.py outrights
    .venv/bin/python research/tennis_other_markets.py report
"""
import argparse
from bisect import bisect_right
from collections import Counter, defaultdict
from datetime import date, datetime, timezone
import json
import math
from pathlib import Path
import re
from statistics import mean, median
import sys
import time

import httpx
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from research.api_cost import ROOT, midnight, save, timestamp
from research import calibration_map as cm
from research.calibration_map import cache_file, get, pages

OUT = ROOT / "data/processed/tennis-other-markets"
LIVE = dict(status="settled", limit=1000, min_close_ts=midnight(date(2026, 9, 1)), max_close_ts=midnight(date(2026, 11, 23)))
CUTOFF = midnight(date(2026, 9, 18))
MATCH_WINNER = {"KXATPMATCH", "KXATPCHALLENGERMATCH", "KXWTAMATCH", "KXITFWMATCH", "KXITFMATCH", "KXWTACHALLENGERMATCH",
                "KXITFDOUBLES", "KXITFWDOUBLES", "KXWTADOUBLES", "KXATPDOUBLES", "KXATPCHALLENGERDOUBLES", "KXDAVISCUPMATCH",
                "KXUNITEDCUPMATCH", "KXLAVERCUPMATCH", "KXMIXEDDOUBLESMATCH", "KXBATTLEOFSEXES", "KXLAVERCUPDOUBLESMATCH",
                "KXTENNISEXHIBITION", "KXSIXKINGSSLAMMATCH", "KXEXHIBITIONMEN"}
TOURS = {"ATP": dict(match="KXATPMATCH", exact="KXATPEXACTMATCH", sets="KXATPSETWINNER", total="KXATPGTOTAL", spread="KXATPGSPREAD"),
         "WTA": dict(match="KXWTAMATCH", exact="KXWTAEXACTMATCH", sets="KXWTASETWINNER", total="KXWTAGTOTAL", spread=None)}
WINDOW, WARMUP, MAX_SPREAD, LEAD = 6*3600, 15, 6, 720
RNG = np.random.default_rng(11)


def fee(p):
    """Taker fee in cents for a price in cents."""
    return 7*(p/100)*(1-p/100)


def val(m):
    try:
        return float(m["settlement_value_dollars"])
    except (KeyError, TypeError, ValueError):
        return None


def vol(m):
    return float(m.get("volume_fp") or m.get("volume") or 0)


def tennis_series():
    rows = json.loads((ROOT/"data/processed/calibration-map/discovery.json").read_text())["Sports"]
    return [s for s in rows if "Tennis" in s["tags"] and s["ticker"] not in MATCH_WINNER]


def inventory(client, floor):
    out = []
    for s in tennis_series():
        if s["volume"] < floor:
            continue
        t = s["ticker"]
        meta = (get(client, f"/series/{t}", {}) or {}).get("series", {})
        live = pages(client, "/markets", series_ticker=t, **LIVE) or []
        hist = pages(client, "/historical/markets", series_ticker=t, limit=1000) or []
        vols = [vol(m) for m in live] or [vol(m) for m in hist]
        per_event = Counter(m["event_ticker"] for m in live or hist)
        sample = (live or hist or [{}])[0]
        row = dict(ticker=t, title=s["title"], series_volume=s["volume"], fee_type=meta.get("fee_type"), fee_multiplier=meta.get("fee_multiplier"),
                   live_markets=len(live), live_events=len({m["event_ticker"] for m in live}),
                   hist_markets=len(hist), hist_events=len({m["event_ticker"] for m in hist}),
                   markets_per_event=median(per_event.values()) if per_event else None,
                   median_market_volume=median(vols) if vols else None, mean_market_volume=mean(vols) if vols else None,
                   share_zero_volume=mean(v == 0 for v in vols) if vols else None,
                   sample_ticker=sample.get("ticker"), sample_title=sample.get("title"),
                   rules_primary=sample.get("rules_primary"), rules_secondary=sample.get("rules_secondary"),
                   fractional=sum(1 for m in live+hist if val(m) is not None and 0.001 < val(m) < 0.999))
        out.append(row)
        print(f'{t:22} fee {str(row["fee_type"]):26} live {row["live_events"]:5}ev/{row["live_markets"]:6}m hist {row["hist_events"]:5}ev/{row["hist_markets"]:6}m '
              f'vol med {row["median_market_volume"]} frac {row["fractional"]} {dict(cm.counts)}', flush=True)
        save(OUT/"inventory.json", out)


# ---------- listings and bars ----------

def key_of(event):
    """KXATPEXACTMATCH-26NOV17ALCMED -> 26NOV17ALCMED; KXATPSETWINNER-26NOV17ALCMED-2 -> (26NOV17ALCMED, 2)."""
    return event.split("-", 1)[1]


def grouped(rows):
    out = defaultdict(list)
    for m in rows or []:
        if val(m) is None or not m.get("settlement_ts"):
            continue
        m["_settle"], m["_outcome"] = int(timestamp(m["settlement_ts"])), val(m)
        out[m["event_ticker"]].append(m)
    for ms in out.values():
        ms.sort(key=lambda m: m["ticker"])
    return out


def listing(series, archived=False):
    if not series:
        return {}
    if archived:
        return grouped(pages(None, "/historical/markets", series_ticker=series, limit=1000))
    rows = pages(None, "/markets", series_ticker=series, **LIVE)
    if rows is None:
        rows = []
        for v in (cm.LIVE_A, cm.LIVE_B):
            rows += pages(None, "/markets", series_ticker=series, **v) or []
    return grouped(rows)


def simple(bars):
    """[(ts, bid, ask, volume, low print, high print, open interest)] in cents, sorted."""
    out = []
    for b in sorted(bars, key=lambda b: b["end_period_ts"]):
        def c(side, field):      # archived bars drop the "_dollars" and "_fp" suffixes
            v = (b.get(side) or {}).get(field+"_dollars", (b.get(side) or {}).get(field))
            return None if v is None else round(100*float(v), 2)
        out.append((b["end_period_ts"], c("yes_bid", "close"), c("yes_ask", "close"), float(b.get("volume_fp") or b.get("volume") or 0),
                    c("price", "low"), c("price", "high"), float(b.get("open_interest_fp") or b.get("open_interest") or 0)))
    return out


def match_bars(series, m, archived):
    """Cached full-window bars of a match-winner market, or None."""
    for path, params in cm.bar_requests(series, archived, m, None):
        file = cache_file(path, params)
        if file.exists():
            return simple(json.loads(file.read_text()).get("candlesticks") or [])
    return None


def mid_of(p):
    return None if p[1] is None or p[2] is None or not 0 < p[1] <= p[2] < 100 else (p[1]+p[2])/2


def detect_start(path):
    """favourite_gap.detect_start on sparse bars: index of the first bar after the warm-up where the midpoint moved
    more than 2c in ten minutes or open interest jumped."""
    for i in range(WARMUP, len(path)):
        j = next((j for j in range(i, -1, -1) if path[j][0] <= path[i][0]-600), None)
        if j is None or mid_of(path[i]) is None or mid_of(path[j]) is None:
            continue
        if abs(mid_of(path[i])-mid_of(path[j])) > 2 or (path[i][6]-path[j][6] >= 2000 and path[i][6] >= 1.5*path[j][6]):
            return i
    return None


def quote_at(path, t, max_spread=MAX_SPREAD):
    """Standing quote at time t: (bid, ask) from the last bar at or before t, or None."""
    i = bisect_right([p[0] for p in path], t)-1
    if i < 0:
        return None
    p = path[i]
    if p[1] is None or p[2] is None or not 0 < p[1] <= p[2] < 100 or p[2]-p[1] > max_spread:
        return None
    return p[1], p[2]


def matches(tour, archived=False):
    """One record per main-tour match that has side markets: the match-winner market, T, and the side tickers."""
    spec = TOURS[tour]
    mw = {key_of(e): ms for e, ms in listing(spec["match"], archived).items()}
    side = {k: listing(spec[k], archived) for k in ("exact", "total", "spread")}
    sets = defaultdict(dict)
    for e, ms in listing(spec["sets"], archived).items():
        k, n = key_of(e).rsplit("-", 1)
        sets[k][n] = ms
    keys = set(sets) | {key_of(e) for d in side.values() for e in d}
    out, cover = [], Counter()
    for k in keys:
        cover["listed"] += 1
        if k not in mw or len(mw[k]) != 2:
            cover["no_match_winner_market"] += 1
            continue
        m = mw[k][0]
        path = match_bars(spec["match"], m, archived)
        if not path:
            cover["match_bars_not_cached"] += 1
            continue
        i = detect_start(path)
        if i is None:
            cover["no_start_detected"] += 1
            continue
        t = path[i][0]-LEAD
        if t < path[0][0]:
            cover["start_too_early_in_window"] += 1
            continue
        cover["usable"] += 1
        out.append(dict(key=k, tour=tour, settle=m["_settle"], t=t, start=path[i][0], match=mw[k], path=path, archived=archived,
                        exact=side["exact"].get(f'{spec["exact"]}-{k}', []), total=side["total"].get(f'{spec["total"]}-{k}', []),
                        spread=side["spread"].get(f'{spec["spread"]}-{k}', []) if spec["spread"] else [],
                        sets={n: ms for n, ms in sets.get(k, {}).items()}))
    out.sort(key=lambda r: (-r["settle"], r["key"]))
    return out, cover


def side_tickers(r):
    return ([m["ticker"] for m in r["exact"]] + [ms[0]["ticker"] for n, ms in sorted(r["sets"].items()) if n in ("1", "2")]
            + [m["ticker"] for m in r["total"]] + [m["ticker"] for m in r["spread"]])


def pack(jobs):
    """Group (start, end, tickers) jobs so each batch request stays under 100 tickers and 9,500 ticker-minutes."""
    out, cur = [], []
    for job in sorted(jobs, key=lambda j: j[0]):
        trial = cur+[job]
        n = sum(len(j[2]) for j in trial)
        minutes = (max(j[1] for j in trial)-min(j[0] for j in trial))//60+2
        if cur and (n > 100 or n*minutes > 9500):
            out.append(cur)
            cur = [job]
        else:
            cur = trial
    return out+[cur] if cur else out


def request_of(group, interval=1):
    n, start = sum(len(j[2]) for j in group), min(j[0] for j in group)
    end = min(max(j[1] for j in group), start+(9500//n-1)*60)      # a lone match whose side markets settled days late: truncate
    return dict(market_tickers=",".join(t for j in group for t in j[2]), start_ts=start, end_ts=end, period_interval=interval)


def download(client, cap, deadline):
    for tour in TOURS:
        rows, cover = matches(tour)
        rows = rows[:cap]
        jobs = [(r["t"]-3600, max(r["settle"], max((m["_settle"] for m in r["exact"]), default=0))+60, side_tickers(r)) for r in rows]
        reqs, done = [request_of(g) for g in pack([j for j in jobs if j[2]])], 0
        save(OUT/f"requests-{tour}.json", reqs)
        for params in reqs:
            if time.time() > deadline and not cache_file("/markets/candlesticks", params).exists():
                continue
            try:
                get(client, "/markets/candlesticks", params)
                done += 1
            except RuntimeError as exc:
                print("  batch failed", exc, flush=True)
        print(f"{tour}: {dict(cover)} taken {len(rows)} requests {done}/{len(reqs)} {dict(cm.counts)}", flush=True)


def side_bars(tour):
    """{ticker: path} from the cached batch responses listed in requests-{tour}.json."""
    out, missing = {}, 0
    file = OUT/f"requests-{tour}.json"
    for params in (json.loads(file.read_text()) if file.exists() else []):
        f = cache_file("/markets/candlesticks", params)
        if not f.exists():
            missing += 1
            continue
        for x in json.loads(f.read_text()).get("markets", []):
            out[x["market_ticker"]] = simple(x.get("candlesticks") or [])
    return out, missing


def archive_request(m, t):
    return f"/historical/markets/{m['ticker']}/candlesticks", dict(start_ts=t-5400, end_ts=t+60, period_interval=1)


def archive(client, cap, deadline):
    for tour in TOURS:
        rows, cover = matches(tour, archived=True)
        rows = [r for r in rows if r["exact"]][:cap]
        n = 0
        for r in rows:
            for m in r["exact"]:
                req = archive_request(m, r["t"])
                if time.time() > deadline and not cache_file(*req).exists():
                    continue
                try:
                    get(client, *req)
                    n += 1
                except RuntimeError:
                    pass
        print(f"{tour} archive: {dict(cover)} taken {len(rows)} market reads {n} {dict(cm.counts)}", flush=True)


def outright_request(series, ms):
    first = min(int(timestamp(m["close_time"])) for m in ms)
    opened = min(int(timestamp(m["open_time"])) for m in ms)
    return dict(market_tickers=",".join(m["ticker"] for m in ms[:100]), start_ts=max(opened, first-12*86400), end_ts=first, period_interval=60), first


def outrights(client):
    for series in ("KXATP", "KXWTA"):
        for e, ms in listing(series).items():
            if ms[0]["_settle"] < CUTOFF or len(ms) < 8:
                continue
            params, _ = outright_request(series, ms)
            try:
                get(client, "/markets/candlesticks", params)
            except RuntimeError as exc:
                print("  failed", e, exc, flush=True)
        print(series, dict(cm.counts), flush=True)


# ---------- statistics ----------

def boot(rows, value, draws=2000):
    """Mean of value(row) with a 95% bootstrap interval over matches."""
    by = defaultdict(lambda: [0.0, 0])
    for r in rows:
        v = value(r)
        if v is None:
            continue
        by[r["key"]][0] += v
        by[r["key"]][1] += 1
    c = np.array(list(by.values()))
    if len(c) == 0:
        return None
    est = c[:, 0].sum()/c[:, 1].sum()
    if len(c) < 8:
        return dict(mean=round(float(est), 2), ci=None, n=int(c[:, 1].sum()), matches=len(c))
    pick = RNG.integers(0, len(c), (draws, len(c)))
    b = c[pick, 0].sum(1)/c[pick, 1].sum(1)
    lo, hi = np.percentile(b, [2.5, 97.5])
    return dict(mean=round(float(est), 2), ci=[round(float(lo), 2), round(float(hi), 2)], n=int(c[:, 1].sum()), matches=len(c))


def gap(r):
    return 100*r["payout"]-r["mid"]


def net_yes(r):
    return 100*r["payout"]-r["ask"]-fee(r["ask"])


def net_no(r):
    return 100*(1-r["payout"])-(100-r["bid"])-fee(100-r["bid"])


def halves(rows):
    order = sorted({(r["settle"], r["key"]) for r in rows})
    first = {k for _, k in order[:len(order)//2]}
    return [r for r in rows if r["key"] in first], [r for r in rows if r["key"] not in first]


def show(b):
    if not b:
        return f'{"-":>24}'
    ci = f'[{b["ci"][0]:+5.1f},{b["ci"][1]:+5.1f}]' if b["ci"] else "[  n/a  ]"
    return f'{b["mean"]:+6.2f} {ci:>15}'


def table(title, rows, bands, label=None):
    """Price-band calibration: gap with a match-clustered interval, both tradeable versions, chronological halves."""
    out = {}
    print(f"\n{title}: {len(rows)} markets in {len({r['key'] for r in rows})} matches")
    print(f'  {"band":>9} {"n":>5} {"match":>5} {"price":>6} {"real":>6} {"gap":>6} {"95% interval":>15} {"buyYES":>6} {"interval":>15} {"buyNO":>6} {"interval":>15} {"sprd":>4} | {"gap h1":>7} {"gap h2":>7} | {"NO h1":>6} {"NO h2":>6} {"YES h1":>6} {"YES h2":>6}')
    a, b = halves(rows)
    for lo, hi in bands:
        pick = lambda rs: [r for r in rs if lo <= (r["mid"] if label is None else r[label]) < hi]
        sub = pick(rows)
        if len(sub) < 8:
            continue
        g, y, n = boot(sub, gap), boot(sub, net_yes), boot(sub, net_no)
        h = [boot(pick(part), f) for f in (gap, net_no, net_yes) for part in (a, b)]
        hm = [f'{x["mean"]:+6.1f}' if x else "     -" for x in h]
        out[f"{lo}-{hi}"] = dict(n=len(sub), matches=g["matches"], price=round(mean(r["mid"] for r in sub), 2), realised=round(100*mean(r["payout"] for r in sub), 2),
                                 gap=g, buy_yes=y, buy_no=n, spread=round(mean(r["ask"]-r["bid"] for r in sub), 2),
                                 gap_halves=[h[0], h[1]], buy_no_halves=[h[2], h[3]], buy_yes_halves=[h[4], h[5]],
                                 median_volume=median(r["volume"] for r in sub),
                                 median_volume_hour_before_T=median(r.get("pre_volume", 0) for r in sub),
                                 median_open_interest_at_T=median(r.get("pre_oi", 0) for r in sub),
                                 share_spread_1c=round(mean(r["ask"]-r["bid"] <= 1.01 for r in sub), 2))
        print(f'  {lo:>4}-{hi:<4} {len(sub):5} {g["matches"]:5} {out[f"{lo}-{hi}"]["price"]:6.1f} {out[f"{lo}-{hi}"]["realised"]:6.1f} {show(g)} {show(y)} {show(n)} '
              f'{out[f"{lo}-{hi}"]["spread"]:4.1f} | {hm[0]:>7} {hm[1]:>7} | {hm[2]} {hm[3]} {hm[4]} {hm[5]}')
    return out


# ---------- report ----------

def exact_label(m, fav_code):
    """KX...-ALC20 -> ('fav', '2-0') given which player is the pre-match favourite."""
    code = m["ticker"].rsplit("-", 1)[1]
    hit = re.match(r"(.+?)(\d)(\d)$", code)
    if not hit:
        return None
    return ("fav" if hit[1] == fav_code else "dog"), f"{hit[2]}-{hit[3]}", hit[1]


def side_rows(tour, archived=False):
    """Pre-match rows for every side market of every usable match, plus the per-match record for the consistency study."""
    rows, cover = matches(tour, archived)
    bars, missing = ({}, 0) if archived else side_bars(tour)
    out = defaultdict(list)
    records = []
    for r in rows:
        mq = quote_at(r["path"], r["t"])
        if mq is None:
            cover["match_quote_missing_or_wide"] += 1
            continue
        code0 = r["match"][0]["ticker"].rsplit("-", 1)[1]
        code1 = r["match"][1]["ticker"].rsplit("-", 1)[1]
        pm = (mq[0]+mq[1])/2
        fav = code0 if pm >= 50 else code1
        base = dict(key=r["key"], tour=tour, settle=r["settle"], t=r["t"], match_mid=pm, fav_price=max(pm, 100-pm), archived=archived,
                    match_frac=0.001 < r["match"][0]["_outcome"] < 0.999)
        rec = dict(base, codes=(code0, code1), match_bid=mq[0], match_ask=mq[1], match_outcome=r["match"][0]["_outcome"], exact={}, start=r["start"],
                   path=r["path"], paths={})

        def q(m):
            if archived:
                f = cache_file(*archive_request(m, r["t"]))
                if not f.exists():
                    return "not_downloaded"
                path = simple(json.loads(f.read_text()).get("candlesticks") or [])
            else:
                path = bars.get(m["ticker"])
                if path is None:
                    return "not_downloaded"
            rec["paths"][m["ticker"]] = path
            got = quote_at(path, r["t"])
            return got or "no_quote_or_wide"

        for kind, ms in (("exact", r["exact"]), ("total", r["total"]), ("spread", r["spread"]),
                         ("set1", r["sets"].get("1", [])[:1]), ("set2", r["sets"].get("2", [])[:1])):
            for m in ms:
                got = q(m)
                if isinstance(got, str):
                    cover[f"{kind}_{got}"] += 1
                    continue
                row = dict(base, kind=kind, ticker=m["ticker"], bid=got[0], ask=got[1], mid=(got[0]+got[1])/2, payout=m["_outcome"],
                           volume=vol(m), frac=0.001 < m["_outcome"] < 0.999,
                           pre_volume=sum(p[3] for p in rec["paths"][m["ticker"]] if p[0] <= r["t"]),
                           pre_oi=next((p[6] for p in reversed(rec["paths"][m["ticker"]]) if p[0] <= r["t"]), 0))
                if kind == "exact":
                    lab = exact_label(m, fav)
                    if lab is None:
                        continue
                    row.update(side=lab[0], score=lab[1], outcome=f"{lab[0]} {lab[1]}", code=lab[2])
                    rec["exact"][lab[2], lab[1]] = row
                if kind in ("set1", "set2"):      # express as the pre-match favourite's side
                    if m["ticker"].rsplit("-", 1)[1] != fav:
                        row.update(bid=100-got[1], ask=100-got[0], mid=100-row["mid"], payout=1-m["_outcome"])
                out[kind].append(row)
                cover[f"{kind}_rows"] += 1
        records.append(rec)
    return out, records, cover, missing


def consistency(records):
    """Taker baskets across the exact-score and match-winner books, minute by minute from T-30min to settlement."""
    res = dict(pre=Counter(), play=Counter())
    gaps = dict(pre=[], play=[])
    episodes, lead = [], []
    soft = []
    for rec in records:
        ex = rec["exact"]
        a, b = rec["codes"]
        scores = sorted({s for _, s in ex})
        if len(ex) != 2*len(scores) or len(scores) not in (2, 3) or any((c, s) not in ex for c in (a, b) for s in scores):
            continue
        paths = {k: rec["paths"][v["ticker"]] for k, v in ex.items()}
        times = {k: [p[0] for p in v] for k, v in paths.items()}
        mt = [p[0] for p in rec["path"]]
        end = min(min(v[-1][0] for v in paths.values()), rec["path"][-1][0])-120
        run, prev = defaultdict(int), None
        for t in range(rec["t"]-1800, end, 60):
            phase = "pre" if t <= rec["t"] else "play"
            i = bisect_right(mt, t)-1
            if i < 0:
                continue
            mb, ma = rec["path"][i][1], rec["path"][i][2]
            if mb is None or ma is None or not 0 < mb <= ma < 100:
                continue
            q = {}
            for k in paths:
                j = bisect_right(times[k], t)-1
                if j < 0:
                    break
                bid, ask = paths[k][j][1], paths[k][j][2]
                if bid is None or ask is None or ask < bid:
                    break
                q[k] = (bid, ask)
            else:
                res[phase]["minutes"] += 1
                asks_ok = all(v[1] < 100 for v in q.values())
                bids_ok = all(v[0] > 0 for v in q.values())
                A = [q[a, s] for s in scores]
                B = [q[b, s] for s in scores]
                baskets = {}
                if asks_ok:
                    baskets["buy all exact YES"] = 100-sum(v[1]+fee(v[1]) for v in q.values())
                    baskets["exact YES pair A + match NO A"] = 100-sum(v[1]+fee(v[1]) for v in A)-(100-mb)-fee(100-mb)
                    baskets["exact YES pair B + match YES A"] = 100-sum(v[1]+fee(v[1]) for v in B)-ma-fee(ma)
                if bids_ok:
                    n = len(q)
                    baskets["buy all exact NO"] = 100*(n-1)-sum(100-v[0]+fee(100-v[0]) for v in q.values())
                    baskets["exact NO pair A + match YES A"] = 100*len(A)-sum(100-v[0]+fee(100-v[0]) for v in A)-ma-fee(ma)
                    baskets["exact NO pair B + match NO A"] = 100*len(B)-sum(100-v[0]+fee(100-v[0]) for v in B)-(100-mb)-fee(100-mb)
                if asks_ok and bids_ok and all(v[1]-v[0] <= MAX_SPREAD for v in q.values()) and ma-mb <= MAX_SPREAD:
                    res[phase]["tight_minutes"] += 1
                    mids = {k: (v[0]+v[1])/2 for k, v in q.items()}
                    total = sum(mids.values())
                    implied = 100*sum(mids[a, s] for s in scores)/total
                    pm = (mb+ma)/2
                    gaps[phase].append((implied-pm, total-100, sum(mids[a, s] for s in scores)-pm, sum(v[1]-v[0] for v in q.values()), ma-mb))
                    if prev is not None and prev[0] == t-60:
                        lead.append((prev[1]-prev[2], pm-prev[2], implied-prev[1], phase))
                    prev = (t, implied, pm)
                for name, profit in baskets.items():
                    res[phase][f"priced {name}"] += 1
                    if profit > 0:
                        res[phase][f"profitable {name}"] += 1
                        run[name] += 1
                    elif run[name]:
                        episodes.append((name, phase, run[name], rec["key"]))
                        run[name] = 0
                    if profit > 0 and phase == "play":
                        res[phase]["cents "+name] += profit
        # the soft question, at T
        try:
            mids = {k: v["mid"] for k, v in ex.items()}
            total = sum(mids.values())
            implied = 100*sum(mids[a, s] for s in scores)/total
            if not rec["match_frac"] and not any(v["frac"] for v in ex.values()):
                soft.append(dict(key=rec["key"], settle=rec["settle"], p=rec["match_mid"], q=implied, y=rec["match_outcome"], total=total,
                                 raw=sum(mids[a, s] for s in scores), bid=rec["match_bid"], ask=rec["match_ask"]))
        except KeyError:
            pass
    return res, gaps, episodes, lead, soft


def set_link(rows, name):
    """Set-1 price against the set probability implied by the match price under independent, equal sets:
    match = s^2 (3 - 2s). Does the difference predict who wins the set?"""
    d, e = [], []
    for r in rows:
        if r["frac"] or r["match_frac"]:
            continue
        lo, hi = 0.5, 1.0
        for _ in range(40):
            s = (lo+hi)/2
            lo, hi = (s, hi) if s*s*(3-2*s) < r["fav_price"]/100 else (lo, s)
        d.append(100*s-r["mid"])
        e.append(100*r["payout"]-r["mid"])
    if len(d) < 30:
        return None
    d, e = np.array(d), np.array(e)
    c = d-d.mean()
    slope = (c*e).sum()/(c**2).sum()
    se = math.sqrt(((c*(e-e.mean()-slope*c))**2).sum())/(c**2).sum()
    print(f"  {name}: {len(d)} matches; model-implied set price minus quoted set price: mean {d.mean():+.2f}c, sd {d.std():.2f}c; "
          f"slope of (set outcome - set price) on it {slope:+.2f} (se {se:.2f}); 0 = the set book is right")
    return dict(n=len(d), mean=float(d.mean()), sd=float(d.std()), slope=float(slope), se=float(se))


def retirements(tour):
    """Exact-score events with a fractional settlement, split by what the match-winner market did."""
    out = Counter()
    detail = []
    spec = TOURS[tour]
    for archived in (True, False):
        mw = {key_of(e): ms for e, ms in listing(spec["match"], archived).items()}
        for e, ms in listing(spec["exact"], archived).items():
            if archived and ms[0]["_settle"] >= CUTOFF:
                continue
            k = key_of(e)
            out["events"] += 1
            if not any(0.001 < m["_outcome"] < 0.999 for m in ms):
                continue
            out["fractional"] += 1
            if k not in mw:
                out["fractional, no match-winner market found"] += 1
                continue
            m0 = mw[k]
            if any(0.001 < m["_outcome"] < 0.999 for m in m0):
                out["walkover or void (match winner fractional too)"] += 1
                continue
            out["retirement (match winner paid 0 or 100)"] += 1
            winner = next(m["ticker"].rsplit("-", 1)[1] for m in m0 if m["_outcome"] > .5)
            win_pair = sum(100*m["_outcome"] for m in ms if exact_label(m, winner)[0] == "fav")
            lose_pair = sum(100*m["_outcome"] for m in ms if exact_label(m, winner)[0] == "dog")
            detail.append(dict(event=e, winner_pair=round(win_pair, 1), loser_pair=round(lose_pair, 1),
                               values={m["ticker"].rsplit("-", 1)[1]: m["_outcome"] for m in ms}))
    return out, detail


def report():
    summary = {}
    bands_long = ((1, 5), (5, 10), (10, 15), (15, 20), (20, 30), (30, 40), (40, 50), (50, 65), (65, 99))
    bands_two = ((1, 10), (10, 20), (20, 35), (35, 50), (50, 65), (65, 80), (80, 90), (90, 99))
    fav_bands = ((50, 60), (60, 70), (70, 80), (80, 90), (90, 99))
    pooled = {}
    for tour in TOURS:
        rows, records, cover, missing = side_rows(tour)
        print(f"\n{'='*30} {tour} recent tier {'='*30}\ncoverage {dict(cover)}; batch responses missing {missing}")
        s = summary[tour] = dict(coverage=dict(cover))
        ex = [r for r in rows["exact"] if r["score"] in ("2-0", "2-1")]
        if ex:
            s["exact_bands"] = table(f"{tour} exact score (best of three), all outcomes by YES midpoint", ex, bands_long)
            s["exact_outcomes"] = {}
            for name in ("fav 2-0", "fav 2-1", "dog 2-1", "dog 2-0"):
                sub = [r for r in ex if r["outcome"] == name]
                s["exact_outcomes"][name] = table(f"{tour} exact score, outcome '{name}' by match favourite's price", sub, fav_bands+((50, 99),), "fav_price")
            ex5 = [r for r in rows["exact"] if r["score"] not in ("2-0", "2-1")]
            if ex5:
                s["exact_bo5"] = table(f"{tour} exact score (best of five)", ex5, bands_long)
        for kind, bands in (("set1", fav_bands), ("set2", fav_bands), ("total", bands_two), ("spread", bands_two)):
            if rows[kind]:
                what = "match favourite's side, by its set price" if kind.startswith("set") else "by YES midpoint"
                s[kind] = table(f"{tour} {kind}: {what}", rows[kind], bands if not kind.startswith("set") else ((1, 50),)+bands)
        print(f"\n{tour} set winner against the match winner")
        s["set_link"] = {k: set_link(rows[k], k) for k in ("set1", "set2")}
        if tour == "ATP":
            res, gaps, episodes, lead, soft = consistency(records)
            s["consistency"] = cons_report(res, gaps, episodes, lead, soft)
        ret, detail = retirements(tour)
        s["retirements"] = dict(ret)
        print(f"\n{tour} exact-score settlement on incomplete matches: {dict(ret)}")
        if detail:
            w = [d["winner_pair"] for d in detail]
            print(f"  retirements: the advancing player's exact-score pair was paid {mean(w):.1f}c on average (median {median(w):.1f}c, min {min(w):.1f}c) "
                  f"against 100c on the match-winner contract; the retiring player's pair was paid {mean(d['loser_pair'] for d in detail):.1f}c against 0c")
            s["retirement_pairs"] = dict(n=len(detail), winner_pair_mean=mean(w), winner_pair_median=median(w), loser_pair_mean=mean(d["loser_pair"] for d in detail))
            save(OUT/f"retirements-{tour}.json", detail)
        hold, hrecords, hcover, _ = side_rows(tour, archived=True)
        ex = [r for r in hold["exact"]]
        if ex:
            print(f"\n{'-'*20} {tour} archive holdout (exact score, pre-match quote only) coverage {dict(hcover)}")
            s["archive_exact_bo3"] = table(f"{tour} ARCHIVE exact score best of three", [r for r in ex if r["score"] in ("2-0", "2-1")], bands_long)
            s["archive_exact_bo5"] = table(f"{tour} ARCHIVE exact score best of five", [r for r in ex if r["score"] not in ("2-0", "2-1")], bands_long)
            for name in ("fav 2-0", "fav 2-1", "dog 2-1", "dog 2-0"):
                table(f"{tour} ARCHIVE '{name}' by favourite price", [r for r in ex if r["outcome"] == name], ((50, 70), (70, 99), (50, 99)), "fav_price")
            s["archive_consistency"] = cons_report(*consistency(hrecords), title=f"{tour} ARCHIVE holdout, quotes up to the reference time only")
        both = [r for r in rows["total"]]
        pooled.setdefault("total", []).extend(both)
    summary["pooled_total"] = table("ATP + WTA total games pooled, by YES midpoint", pooled["total"], bands_two+((90, 95), (95, 99), (80, 99)))
    summary["outrights"] = outright_report()
    save(OUT/"summary.json", summary)


def cons_report(res, gaps, episodes, lead, soft, title="ATP, recent tier"):
    out = {}
    print(f"\nCONSISTENCY of the exact-score book with the match-winner book ({title})")
    for phase in ("pre", "play"):
        g = np.array(gaps[phase])
        r = res[phase]
        print(f'  {phase}: {r["minutes"]} match-minutes with all books quoted, {r["tight_minutes"]} with every spread <= {MAX_SPREAD}c')
        if len(g):
            d = g[:, 0]
            print(f'    exact-implied match price minus match midpoint: mean {d.mean():+.2f}c, sd {d.std():.2f}c, |gap| median {np.median(abs(d)):.2f}c, '
                  f'90th pct {np.percentile(abs(d), 90):.2f}c, 99th {np.percentile(abs(d), 99):.2f}c; share over 3c {np.mean(abs(d) > 3):.3f}')
            print(f'    sum of the exact midpoints minus 100: mean {g[:, 1].mean():+.2f}c; raw pair minus match mid: mean {g[:, 2].mean():+.2f}c; '
                  f'sum of exact spreads {g[:, 3].mean():.1f}c, match spread {g[:, 4].mean():.2f}c')
            out[phase] = dict(minutes=r["minutes"], tight=r["tight_minutes"], gap_mean=float(d.mean()), gap_sd=float(d.std()), abs_median=float(np.median(abs(d))),
                              abs_p90=float(np.percentile(abs(d), 90)), abs_p99=float(np.percentile(abs(d), 99)), overround=float(g[:, 1].mean()),
                              exact_spreads=float(g[:, 3].mean()), match_spread=float(g[:, 4].mean()))
        for k in sorted(k for k in r if k.startswith("priced ")):
            name = k[7:]
            print(f'    {name:34} priced {r[k]:7} minutes, profitable after fees {r["profitable "+name]:5} ({r["profitable "+name]/max(r[k], 1):.4%})'
                  + (f', mean {r["cents "+name]/max(r["profitable "+name], 1):.2f}c' if phase == "play" else ""))
            out.setdefault("baskets", {})[f"{phase} {name}"] = dict(priced=r[k], profitable=r["profitable "+name])
    by = defaultdict(list)
    for name, phase, n, key in episodes:
        by[name, phase].append(n)
    for (name, phase), v in sorted(by.items()):
        print(f'    episodes {phase:4} {name:34} {len(v):4} episodes in {len({k for n2, p2, _, k in episodes if (n2, p2) == (name, phase)})} matches, '
              f'length median {median(v)} min, share of one minute {mean(x == 1 for x in v):.2f}, longest {max(v)}')
        out.setdefault("episodes", {})[f"{phase} {name}"] = dict(n=len(v), median=median(v), one_minute=mean(x == 1 for x in v), longest=max(v))
    # which book closes the gap in the next minute
    for phase in ("pre", "play"):
        L = np.array([(g, dm, dq) for g, dm, dq, p in lead if p == phase and abs(g) >= 2])
        if len(L) > 30:
            bm = (L[:, 0]*L[:, 1]).sum()/(L[:, 0]**2).sum()
            bq = -(L[:, 0]*L[:, 2]).sum()/(L[:, 0]**2).sum()
            print(f'    {phase}: when the books differ by 2c or more ({len(L)} minutes), the next minute the match book closes {bm:.1%} of the gap '
                  f'and the exact book closes {bq:.1%}')
            out[f"lead_{phase}"] = dict(n=len(L), match_closes=float(bm), exact_closes=float(bq))
    # the soft edge: at T, does the exact book know something the match book does not?
    if soft:
        p, q, y = (np.array([s[k] for s in soft]) for k in ("p", "q", "y"))
        d = q-p
        slope = (d*(100*y-p)).sum()/(d**2).sum()
        se = math.sqrt(((d*(100*y-p-slope*d))**2).sum())/(d**2).sum()
        print(f'\n  SOFT EDGE at the pre-match reference time ({len(soft)} completed matches): exact-implied minus match mid has sd {d.std():.2f}c; '
              f'regressing (outcome - match price) on it gives slope {slope:+.2f} (se {se:.2f}); 0 = match book right, 1 = exact book right')
        print(f'    Brier: match book {np.mean((y-p/100)**2):.5f}, exact-implied {np.mean((y-q/100)**2):.5f}')
        out["soft"] = dict(n=len(soft), sd=float(d.std()), slope=float(slope), se=float(se), brier_match=float(np.mean((y-p/100)**2)), brier_exact=float(np.mean((y-q/100)**2)))
        for cut in (1.5, 2.5):
            for sign, name in ((1, "exact book higher"), (-1, "exact book lower")):
                sub = [s for s in soft if sign*(s["q"]-s["p"]) >= cut]
                if len(sub) >= 8:
                    # trade the match book in the direction the exact book says, at the match ask, taker fee
                    def net(s):
                        price = s["ask"] if sign > 0 else 100-s["bid"]
                        return 100*(s["y"] if sign > 0 else 1-s["y"])-price-fee(price)
                    g = boot(sub, lambda s: sign*(100*s["y"]-s["p"]))
                    n = boot(sub, net)
                    print(f'    {name} by {cut}c+: {len(sub):4} matches, outcome minus match price in that direction {show(g)}, net of following it on the match book {show(n)}')
                    out[f"soft {name} {cut}"] = dict(n=len(sub), gap=g, net=n)
    return out


def outright_report():
    out = {}
    for series in ("KXATP", "KXWTA"):
        rows = []
        events = 0
        for e, ms in listing(series).items():
            if ms[0]["_settle"] < CUTOFF or len(ms) < 8:
                continue
            params, first = outright_request(series, ms)
            f = cache_file("/markets/candlesticks", params)
            if not f.exists():
                continue
            events += 1
            data = {x["market_ticker"]: simple(x.get("candlesticks") or []) for x in json.loads(f.read_text()).get("markets", [])}
            for hours in (24, 3):
                for m in ms:
                    path = data.get(m["ticker"])
                    if not path:
                        continue
                    i = bisect_right([p[0] for p in path], first-hours*3600)-1
                    if i < 0:
                        continue
                    bid, ask = path[i][1], path[i][2]
                    if bid is None or ask is None or not 0 <= bid <= ask <= 100 or ask >= 100:
                        continue
                    rows.append(dict(key=e, settle=ms[0]["_settle"], hours=hours, bid=bid, ask=ask, mid=(bid+ask)/2, payout=m["_outcome"], volume=vol(m),
                                     has_bid=bid > 0, wide=ask-bid))
        print(f"\n{'='*20} {series} tournament winner: {events} events (recent tier)")
        for hours in (24, 3):
            sub = [r for r in rows if r["hours"] == hours]
            if not sub:
                continue
            by = defaultdict(list)
            for r in sub:
                by[r["key"]].append(r)
            print(f"  {hours}h before the first elimination: {len(sub)} quoted markets; sum of asks per event median {median(sum(r['ask'] for r in v) for v in by.values()):.0f}c, "
                  f"sum of bids {median(sum(r['bid'] for r in v) for v in by.values()):.0f}c, markets with a bid {mean(r['has_bid'] for r in sub):.0%}, median spread {median(r['wide'] for r in sub):.0f}c")
            tradeable = [r for r in sub if r["has_bid"] and r["wide"] <= 10]
            out[f"{series} {hours}h"] = table(f"{series} {hours}h, markets with a bid and spread <= 10c", tradeable, ((0, 3), (3, 10), (10, 25), (25, 50), (50, 99)))
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("cmd", choices=("inventory", "download", "archive", "outrights", "report"))
    parser.add_argument("--floor", type=float, default=1e5)
    parser.add_argument("--minutes", type=float, default=15)
    parser.add_argument("--cap", type=int, default=600, help="most recent matches per tour")
    args = parser.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    if args.cmd == "report":
        return report()
    deadline = time.time()+60*args.minutes
    with httpx.Client(timeout=40) as client:
        if args.cmd == "inventory":
            inventory(client, args.floor)
        elif args.cmd == "download":
            download(client, args.cap, deadline)
        elif args.cmd == "archive":
            archive(client, args.cap, deadline)
        else:
            outrights(client)


if __name__ == "__main__":
    main()
