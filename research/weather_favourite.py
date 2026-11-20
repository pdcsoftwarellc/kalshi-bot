"""Weather favourite: a resting NO bid one cent inside the spread on the favourite of a weather ladder, 24h before close.

Research only. Public Kalshi reads, no orders, no authentication. The rule, the samples and the verdict test are frozen in
data/processed/weather-favourite/prereg.txt (written before any holdout bar was downloaded). This file reuses the universe, the bar
format and the quote/fill logic of research/overround.py without changing them.

Commands (all resumable; every network command is bounded by --minutes and by a 5 hour budget over all runs)

    fees     GET /series once: fee_type and multiplier of the weather series, live
    rbars    hourly bars of the recent weather events the original study did not fetch (batch)
    abars    hourly bars of archive weather events, sha256 order, one outcome per request
    trades   trade record after the bid for a sha256-ordered sample of filled favourites (recent tier)
    xbars    DIAGNOSTIC, outside the frozen universe: hourly bars of the recent ladder events that FAILED the 5,000-contract filter (batch)
    report   offline -> data/processed/weather-favourite/report.txt and summary.json
    status   what is on disk and how much of the budget is used
"""
import argparse
from collections import Counter, defaultdict
from datetime import datetime, timezone
import json
from pathlib import Path
import re
import sys
import time

import httpx
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import config
from research import overround as ov
from research.api_cost import ROOT, timestamp

RAW = ROOT / "data/raw/weather-favourite"
OUT = ROOT / "data/processed/weather-favourite"
PACE = 5.0
BUDGET = 5*3600
BACKOFF = (60, 120, 240, 300, None)
GROUP = "Weather ranges"
OLD_HIGH = {"KXHIGHAUS": "AUS", "KXHIGHCHI": "CHI", "KXHIGHDEN": "DEN", "KXHIGHHOU": "HOU", "KXHIGHLAX": "LAX", "KXHIGHMIA": "MIA", "KXHIGHNY": "NYC",
            "KXHIGHPHIL": "PHIL"}
DAY = re.compile(r"-(\d{2}[A-Z]{3}\d{2})$")
STRIKE = re.compile(r"-[TB](-?[\d.]+)$")

last_request = [0.0]
deadline = [float("inf")]
log = Counter()


class OutOfTime(Exception):
    pass


def read(client, path, params):
    """One paced public read (at most one request per PACE seconds). Parsed JSON, or None on a 4xx other than 429."""
    for pause in BACKOFF:
        if time.time() > deadline[0]:
            raise OutOfTime
        wait = last_request[0]+PACE-time.time()
        if wait > 0:
            time.sleep(wait)
        log["requests"] += 1
        try:
            resp = client.get(config.KALSHI_PROD_BASE+path, params=params, headers={"User-Agent": "research-client"})
        except httpx.RequestError:
            resp = None
        last_request[0] = time.time()
        if resp is not None and resp.status_code == 200:
            return resp.json()
        if resp is not None and resp.status_code != 429 and resp.status_code < 500:
            log[f"http_{resp.status_code}"] += 1
            return None
        log["http_429" if resp is not None and resp.status_code == 429 else "server_or_network_error"] += 1
        if pause is None:
            raise RuntimeError("read failed after long back-off: "+path)
        print(f"  back-off {pause}s ({'429' if resp is not None else 'network'})", flush=True)
        time.sleep(pause)


def used_seconds():
    return sum(r["seconds"] for r in ov.load(RAW/"download-log.json", []))


# ---------------------------------------------------------------- samples

def weather():
    """Eligible weather events with the outcomes to fetch ('picked', as overround.sample()) and the sample tag."""
    orb, oab = ov.load(ov.RAW/"recent-bars.json", {}), ov.load(ov.RAW/"archive-bars.json", {})
    out = []
    for e in ov.universe().values():
        if e["group"] != GROUP:
            continue
        e["picked"] = e["markets"] if e["tier"] == "recent" else sorted(e["markets"], key=lambda m: (-m["volume"], m["ticker"]))[:ov.ARCHIVE_OUTCOMES]
        old = orb if e["tier"] == "recent" else oab
        e["sample"] = "original" if all(m["ticker"] in old for m in e["picked"]) else "holdout"
        out.append(e)
    return out, orb, oab


def archive_order(events):
    return sorted((e for e in events if e["tier"] == "archive" and e["sample"] == "holdout"), key=lambda e: ov.pick_hash("wf-archive", e["event"]))


# ---------------------------------------------------------------- downloads

def fees_step(client):
    data = read(client, "/series", dict(include_volume="true")) or {}
    rows = {s["ticker"]: dict(fee_type=s.get("fee_type"), fee_multiplier=s.get("fee_multiplier"), category=s.get("category"))
            for s in data.get("series") or [] if s.get("category") == "Climate and Weather"}
    ov.dump(RAW/"series-fees.json", dict(read_at=int(time.time()), series=rows))
    print(f"{len(rows)} weather series; fee types {Counter(r['fee_type'] for r in rows.values())}", flush=True)


def rbars_step(client):
    have = ov.load(RAW/"recent-bars.json", {})
    events, _, _ = weather()
    jobs = sorted(((ov.span(e, m), m["ticker"]) for e in events if e["tier"] == "recent" and e["sample"] == "holdout" for m in e["picked"]
                   if m["ticker"] not in have), key=lambda j: (j[0][1], j[0][0]))
    chunks, cur = [], []
    for j in jobs:
        trial = cur+[j]
        hours = (max(x[0][1] for x in trial)-min(x[0][0] for x in trial))//3600+2
        if cur and (len(trial) > 100 or len(trial)*hours > 9000):
            chunks.append(cur)
            cur = []
        cur.append(j)
    chunks += [cur] if cur else []
    print(f"recent bars: {len(have)} on disk, {len(jobs)} markets in {len(chunks)} requests", flush=True)
    try:
        for n, group in enumerate(chunks):
            lo, hi = min(x[0][0] for x in group), max(x[0][1] for x in group)
            data = read(client, "/markets/candlesticks", dict(market_tickers=",".join(x[1] for x in group), start_ts=lo, end_ts=hi, period_interval=60))
            if data is None:
                log["batch_refused"] += 1
                continue
            got = {x["market_ticker"]: ov.compact(x.get("candlesticks")) for x in data.get("markets", [])}
            for (a, b), t in group:
                have[t] = [bar for bar in got.get(t, []) if a <= bar[0] <= b]
            if n % 10 == 9:
                ov.dump(RAW/"recent-bars.json", have)
                print(f"  {n+1}/{len(chunks)}", flush=True)
    finally:
        ov.dump(RAW/"recent-bars.json", have)
    print(f"recent bars for {len(have)} markets", flush=True)


def lowvol():
    """Recent settled ladder events that fail only the 5,000-contract filter (diagnostic of that filter's look-ahead; not part of the rule)."""
    meta = {s["ticker"]: s for s in ov.load(ov.RAW/"series.json", [])}
    out = []
    for series, v in ov.load(ov.RAW/"events.json", {}).items():
        if kind(series) == "other" or meta[series]["category"] != "Climate and Weather":
            continue
        for ev in v["events"]:
            ms = ev["markets"]
            if not ev["me"] or len(ms) < ov.MIN_OUTCOMES or any(m["result"] not in ("yes", "no") or not m["close"] or not m["open"] for m in ms):
                continue
            volume = sum(m["volume"] for m in ms)
            closes = Counter(m["close"]//600 for m in ms)
            close = 600*min(closes, key=lambda k: (-closes[k], k))
            if volume >= ov.MIN_EVENT_VOLUME or not (ov.HOLD_FROM <= close < ov.NOW):
                continue
            out.append(dict(event=ev["event"], series=series, tier="recent", sample="lowvol", close=close, n=len(ms), yes=sum(m["result"] == "yes" for m in ms),
                            volume=volume, markets=ms, picked=ms, maker_fee=meta[series]["fee_type"] == "quadratic_with_maker_fees",
                            mult=float(meta[series]["fee_multiplier"] if meta[series]["fee_multiplier"] is not None else 1)))
    return out


def batch_bars(client, path, jobs):
    have = ov.load(path, {})
    jobs = sorted((j for j in jobs if j[1] not in have), key=lambda j: (j[0][1], j[0][0]))
    chunks, cur = [], []
    for j in jobs:
        trial = cur+[j]
        hours = (max(x[0][1] for x in trial)-min(x[0][0] for x in trial))//3600+2
        if cur and (len(trial) > 100 or len(trial)*hours > 9000):
            chunks.append(cur)
            cur = []
        cur.append(j)
    chunks += [cur] if cur else []
    print(f"{path.name}: {len(have)} on disk, {len(jobs)} markets in {len(chunks)} requests", flush=True)
    try:
        for group in chunks:
            lo, hi = min(x[0][0] for x in group), max(x[0][1] for x in group)
            data = read(client, "/markets/candlesticks", dict(market_tickers=",".join(x[1] for x in group), start_ts=lo, end_ts=hi, period_interval=60))
            if data is None:
                log["batch_refused"] += 1
                continue
            got = {x["market_ticker"]: ov.compact(x.get("candlesticks")) for x in data.get("markets", [])}
            for (a, b), t in group:
                have[t] = [bar for bar in got.get(t, []) if a <= bar[0] <= b]
    finally:
        ov.dump(path, have)
    print(f"{path.name}: {len(have)} markets", flush=True)


def xbars_step(client):
    batch_bars(client, RAW/"lowvol-bars.json", [(ov.span(e, m), m["ticker"]) for e in lowvol() for m in e["picked"]])


def abars_step(client):
    have = ov.load(RAW/"archive-bars.json", {})
    events, _, _ = weather()
    jobs = [(ov.span(e, m), m["ticker"]) for e in archive_order(events) for m in e["picked"] if m["ticker"] not in have]
    print(f"archive bars: {len(have)} markets on disk, {len(jobs)} still to fetch", flush=True)
    try:
        for n, ((a, b), t) in enumerate(jobs):
            data = read(client, f"/historical/markets/{t}/candlesticks", dict(start_ts=a, end_ts=b, period_interval=60))
            if data is None:
                log["no_bars"] += 1
            have[t] = ov.compact((data or {}).get("candlesticks"))
            if n % 24 == 23:
                ov.dump(RAW/"archive-bars.json", have)
            if n % 120 == 119:
                print(f"  {n+1}/{len(jobs)}", flush=True)
    finally:
        ov.dump(RAW/"archive-bars.json", have)
    print(f"archive bars for {len(have)} markets", flush=True)


def trades_step(client):
    recs = [r for r in build(24)[0] if r["tier"] == "recent" and r["ok1"] and r["fill1"]]
    recs.sort(key=lambda r: ov.pick_hash("wf-trade", r["ticker"]))
    have = ov.load(RAW/"trades.json", {})
    todo = [r for r in recs if r["ticker"] not in have]
    print(f"trades: {len(have)} on disk, {len(todo)} filled recent favourites not yet read", flush=True)
    try:
        for r in todo:
            rows, cursor, pages = [], "", 0
            while pages < 6:
                data = read(client, "/markets/trades", dict(ticker=r["ticker"], limit=1000, min_ts=int(r["qts"]), **({"cursor": cursor} if cursor else {}))) or {}
                pages += 1
                for t in data.get("trades") or []:
                    rows.append([int(timestamp(t["created_time"])), round(100*float(t["yes_price_dollars"]), 2), float(t.get("count_fp", t.get("count", 0))),
                                 1 if (t.get("taker_side") or t.get("taker_outcome_side")) == "yes" else 0])
                cursor = data.get("cursor")
                if not cursor or not data.get("trades"):
                    break
            have[r["ticker"]] = dict(complete=not cursor, qts=int(r["qts"]), trades=rows)
            if len(have) % 20 == 0:
                ov.dump(RAW/"trades.json", have)
    finally:
        ov.dump(RAW/"trades.json", have)
    print(f"trades for {len(have)} markets", flush=True)


# ---------------------------------------------------------------- the rule

def city(series):
    if series in OLD_HIGH:
        return OLD_HIGH[series]
    for prefix in ("KXHIGHT", "KXLOWT"):
        if series.startswith(prefix):
            return series[len(prefix):]
    return series


def kind(series):
    return "high" if series.startswith("KXHIGH") else "low" if series.startswith("KXLOWT") else "other"


def score(e, bars, hours):
    """The favourite of one event at `hours` before its close and what a NO bid on it would have done. None when nothing is quoted."""
    t = e["close"]-hours*3600
    if not all(m["ticker"] in bars for m in e["picked"]) or t < min(m["open"] for m in e["picked"])+3600:
        return None
    rows = []
    for m in e["picked"]:
        b = bars[m["ticker"]]
        if m["open"] > t or m["close"] <= t:
            continue
        i = ov.quote_at(b, t) if b else None
        if i is None:
            continue
        bid, ask = b[i][1], b[i][2]
        rows.append(dict(m=m, i=i, rb=bid, ra=ask, bid=bid if bid is not None else 0.0, ask=ask if ask is not None else 100.0))
    if not rows:
        return None
    for r in rows:
        r["mid"] = (r["bid"]+r["ask"])/2 if r["ask"] <= 99 else r["bid"]
    ranked = sorted(rows, key=lambda r: -r["mid"])
    f = ranked[0]
    m, b = f["m"], bars[f["m"]["ticker"]]
    later = [x for x in b[f["i"]+1:] if x[0] <= max(m["close"], e["close"])+3600]
    two = f["rb"] is not None and f["ra"] is not None and 1 <= f["rb"] < f["ra"] <= 99
    day = DAY.search(e["event"])
    rec = dict(event=e["event"], series=e["series"], tier=e["tier"], sample=e["sample"], close=e["close"], ticker=m["ticker"], hours=hours, t=t,
               qts=b[f["i"]][0], day=day.group(1) if day else datetime.fromtimestamp(e["close"], timezone.utc).strftime("%y%b%d").upper(),
               month=datetime.fromtimestamp(t, timezone.utc).strftime("%Y-%m"), bid=f["bid"], ask=f["ask"], mid=f["mid"], y=m["result"] == "yes",
               tie=len(ranked) > 1 and ranked[1]["mid"] == f["mid"], maker_fee=e["maker_fee"], mult=e["mult"], quoted=len(rows), n=e["n"],
               sum_mids=sum(r["mid"] for r in rows))
    for k in (0, 1, 2):
        ok = bool(two and f["ra"]-f["rb"] >= max(k+1, 2 if k == 1 else 1))
        offer = f["ask"]-k
        hit = [x for x in later if x[5] > 0 and x[4] is not None and x[4] > offer] if ok else []
        price = 100-offer
        rec.update({f"ok{k}": ok, f"fill{k}": bool(hit), f"price{k}": price, f"fts{k}": hit[0][0] if hit else None, f"vol{k}": sum(x[5] for x in hit),
                    f"net{k}": 100*(not rec["y"])-price-(ov.maker_fee(price, e["mult"]) if e["maker_fee"] else 0.0)})
    yhit = [x for x in later if x[5] > 0 and x[3] is not None and x[3] < f["bid"]+1] if rec["ok1"] else []
    rec.update(yfill=bool(yhit), yprice=f["bid"]+1, ynet=100*rec["y"]-(f["bid"]+1)-(ov.maker_fee(f["bid"]+1, e["mult"]) if e["maker_fee"] else 0.0))
    # ladder position (daily ladders with every outcome quoted and a parsable strike)
    rec["offset"], rec["priced"] = None, None
    strikes = [STRIKE.search(r["m"]["ticker"]) for r in rows]
    if kind(e["series"]) != "other" and len(rows) == e["n"] == 6 and all(strikes) and e["yes"] == 1:
        ladder = sorted(zip((float(s.group(1)) for s in strikes), rows), key=lambda p: p[0])
        fi = next(j for j, (_, r) in enumerate(ladder) if r is f)
        wi = next(j for j, (_, r) in enumerate(ladder) if r["m"]["result"] == "yes")
        total = sum(r["mid"] for _, r in ladder)
        rec["offset"] = wi-fi
        rec["priced"] = {j-fi: r["mid"]/total for j, (_, r) in enumerate(ladder)} if total > 0 else None
        rec["fav_index"] = fi
    return rec


def build(hours):
    events, orb, oab = weather()
    bars = {}
    for part in (ov.load(RAW/"recent-bars.json", {}), ov.load(RAW/"archive-bars.json", {}), orb, oab):
        bars.update(part)
    recs = [r for r in (score(e, bars, hours) for e in events) if r is not None]
    return recs, events, bars


# ---------------------------------------------------------------- evidence

def cluster_boot(num, den, codes, draws=10000, seed=7):
    """sum(num)/sum(den) in % with a percentile interval from a bootstrap over the clusters in `codes`."""
    num, den = np.asarray(num, float), np.asarray(den, float)
    if len(num) == 0 or den.sum() == 0:
        return None
    _, inv = np.unique(np.asarray(codes), return_inverse=True)
    k = inv.max()+1
    a, b = np.bincount(inv, num, k), np.bincount(inv, den, k)
    rng = np.random.default_rng(seed)
    est = []
    for start in range(0, draws, 1000):
        idx = rng.integers(0, k, (min(1000, draws-start), k))
        est.append(a[idx].sum(1)/np.maximum(b[idx].sum(1), 1e-9))
    est = np.concatenate(est)
    lo, hi = np.percentile(est, [2.5, 97.5])
    return dict(ret=100*a.sum()/b.sum(), lo=100*lo, hi=100*hi, clusters=int(k))


def split_halves(fills):
    order = sorted({(r["close"], r["event"]) for r in fills})
    first = {ev for _, ev in order[:len(order)//2]}
    return [r for r in fills if r["event"] in first], [r for r in fills if r["event"] not in first]


def week(r):
    return (r["t"]-4*86400)//(7*86400)


def evidence(recs, k=1, draws=10000, side="no"):
    """Headline numbers for the filled orders among `recs`."""
    if side == "no":
        fills = [r for r in recs if r[f"ok{k}"] and r[f"fill{k}"]]
        num, den = [r[f"net{k}"] for r in fills], [r[f"price{k}"] for r in fills]
    else:
        fills = [r for r in recs if r["ok1"] and r["yfill"]]
        num, den = [r["ynet"] for r in fills], [r["yprice"] for r in fills]
    orders = sum(r[f"ok{k}"] for r in recs)
    out = dict(events=len(recs), orders=int(orders), fills=len(fills), days=len({r["day"] for r in fills}))
    if len(fills) < 5:
        return out
    by_day = cluster_boot(num, den, [r["day"] for r in fills], draws)
    by_event = cluster_boot(num, den, [r["event"] for r in fills], draws)
    by_week = cluster_boot(num, den, [week(r) for r in fills], draws)
    out.update(ret=by_day["ret"], lo=by_day["lo"], hi=by_day["hi"], ev_lo=by_event["lo"], ev_hi=by_event["hi"], wk_lo=by_week["lo"], wk_hi=by_week["hi"],
               weeks=by_week["clusters"], win=100*np.mean([n > 0 for n in num]), price=float(np.mean(den)), cents=float(np.mean(num)))
    if side == "no":
        for name, part in zip(("h1", "h2"), split_halves(fills)):
            b = cluster_boot([r[f"net{k}"] for r in part], [r[f"price{k}"] for r in part], [r["day"] for r in part], draws)
            out[name] = dict(fills=len(part), ret=b["ret"], lo=b["lo"], hi=b["hi"], days=b["clusters"]) if b else None
    return out


def fmt(x):
    if "ret" not in x:
        return f"fills {x['fills']:5d}  (too few)"
    text = (f"fills {x['fills']:5d} of {x['orders']:5d} orders, {x['events']:5d} events, {x['days']:3d} days | {x['ret']:+6.2f}% [{x['lo']:+6.2f}, {x['hi']:+6.2f}] by day"
            f" | by event [{x['ev_lo']:+6.2f}, {x['ev_hi']:+6.2f}] | NO wins {x['win']:4.1f}% at {x['price']:4.1f}c")
    if x.get("h1") and x.get("h2"):
        text += f" | H1 {x['h1']['ret']:+6.2f}% [{x['h1']['lo']:+6.2f}, {x['h1']['hi']:+6.2f}]  H2 {x['h2']['ret']:+6.2f}% [{x['h2']['lo']:+6.2f}, {x['h2']['hi']:+6.2f}]"
    return text


def verdict(x):
    if "ret" not in x:
        return "UNDECIDED (no fills)", None
    half = (x["hi"]-x["lo"])/2
    target = x["ret"] if x["ret"] > 0 else 3.0
    need = int(np.ceil(x["fills"]*(half/target)**2))
    if x["lo"] > 0 and x["h1"]["ret"] > 0 and x["h2"]["ret"] > 0:
        return "CONFIRMED", need
    if x["hi"] < 3.0:
        return "KILLED", need
    return "UNDECIDED", need


# ---------------------------------------------------------------- planted edge

def icc(fills, q):
    """Intraclass correlation by day of the standardised NO-win residuals."""
    y = np.array([not r["y"] for r in fills], float)
    z = (y-q)/np.sqrt(np.maximum(q*(1-q), 1e-6))
    _, inv = np.unique([r["day"] for r in fills], return_inverse=True)
    n = np.bincount(inv)
    s = np.bincount(inv, z)
    pairs = (n*(n-1)).sum()
    if pairs == 0:
        return 0.0
    return float(((s**2).sum()-(z**2).sum())/pairs/max(z.var(), 1e-9))


def norm_cdf(x):
    from math import erf
    return 0.5*(1+np.vectorize(erf)(x/np.sqrt(2)))


def redraw(fills, q, rho, rng):
    _, inv = np.unique([r["day"] for r in fills], return_inverse=True)
    latent = np.sqrt(rho)*rng.standard_normal(inv.max()+1)[inv]+np.sqrt(1-rho)*rng.standard_normal(len(fills))
    return norm_cdf(latent) < q                                       # True = the NO bid wins


def planted(fills, edge, rho, redraws=300, draws=1000, seed=3):
    """Share of outcome redraws (true return = edge on these fills) in which the pre-registered CONFIRMED test fires."""
    rng = np.random.default_rng(seed)
    price = np.array([r["price1"] for r in fills])
    q = np.clip(price*(1+edge/100)/100, 0, 1)
    day = np.array([r["day"] for r in fills])
    order = sorted({(r["close"], r["event"]) for r in fills})
    first = {ev for _, ev in order[:len(order)//2]}
    h1 = np.array([r["event"] in first for r in fills])
    est, hw, hit, pos, both = [], [], 0, 0, 0
    for n in range(redraws):
        net = 100*redraw(fills, q, rho, rng)-price
        b = cluster_boot(net, price, day, draws, seed=n)
        r1, r2 = net[h1].sum()/price[h1].sum(), net[~h1].sum()/price[~h1].sum()
        est.append(b["ret"])
        hw.append((b["hi"]-b["lo"])/2)
        pos += b["lo"] > 0
        both += r1 > 0 and r2 > 0
        hit += b["lo"] > 0 and r1 > 0 and r2 > 0
    return dict(edge=edge, rho=rho, fills=len(fills), true=100*(100*q-price).sum()/price.sum(), mean=float(np.mean(est)), sd=float(np.std(est)),
                half=float(np.mean(hw)), interval=100*pos/redraws, halves=100*both/redraws, confirmed=100*hit/redraws)


def match_rho(fills, target, seed=5):
    """Latent daily correlation whose redraws reproduce the observed intraclass correlation of the residuals."""
    if target <= 0:
        return 0.0, 0.0
    rng = np.random.default_rng(seed)
    price = np.array([r["price1"] for r in fills])
    q = np.clip(price/100, 0.01, 0.99)
    best = (0.0, 0.0)
    for rho in (0.0, 0.02, 0.05, 0.1, 0.15, 0.2, 0.3, 0.4, 0.5, 0.6):
        sims = []
        for _ in range(40):
            win = redraw(fills, q, rho, rng)
            sims.append(icc([dict(r, y=not w) for r, w in zip(fills, win)], q))
        got = float(np.mean(sims))
        if abs(got-target) < abs(best[1]-target) or rho == 0.0:
            best = (rho, got)
    return best


# ---------------------------------------------------------------- report

def report():
    OUT.mkdir(parents=True, exist_ok=True)
    lines, summary = [], {}

    def say(text=""):
        print(text, flush=True)
        lines.append(text)
    recs, events, bars = build(24)
    fees = ov.load(RAW/"series-fees.json", {})
    dl = ov.load(RAW/"download-log.json", [])
    ladders = lambda rs: [r for r in rs if kind(r["series"]) != "other"]
    hold = [r for r in recs if r["sample"] == "holdout"]
    orig = [r for r in recs if r["sample"] == "original"]

    say("WEATHER FAVOURITE: resting NO bid one cent inside the spread on the favourite, 24h before close, held to settlement (rule: prereg.txt)")
    say("Returns are % of the money put up per fill (sum of net / sum of NO price). Intervals: 95%, bootstrap over DAYS unless stated.")
    say()
    say("(0) COVERAGE")
    elig = Counter((e["tier"], e["sample"]) for e in events)
    got = Counter((r["tier"], r["sample"]) for r in recs)
    full = Counter((e["tier"], e["sample"]) for e in events if all(m["ticker"] in bars for m in e["picked"]))
    for key in sorted(elig):
        say(f"    {key[0]:8} {key[1]:9} eligible events {elig[key]:5d} | bars for every outcome {full[key]:5d} | with a quote 24h before close {got[key]:5d}")
    say(f"    download: {sum(r['requests'] for r in dl)} requests in {sum(r['seconds'] for r in dl)/3600:.2f} h over {len(dl)} runs,"
        f" HTTP 429 {sum(r.get('http_429', 0) for r in dl)}, network/server errors {sum(r.get('server_or_network_error', 0) for r in dl)},"
        f" outcomes without bars {sum(r.get('no_bars', 0) for r in dl)}")
    if fees:
        mine = {s: fees["series"].get(s) for s in sorted({e["series"] for e in events})}
        types = Counter((v or {}).get("fee_type") for v in mine.values())
        say(f"    fee_type, live GET /series at {datetime.fromtimestamp(fees['read_at'], timezone.utc):%Y-%m-%d %H:%M}Z: {dict(types)}"
            f" (quadratic = no maker fee); multipliers {dict(Counter(str((v or {}).get('fee_multiplier')) for v in mine.values()))}")
        summary["fee_types"] = {str(k): v for k, v in types.items()}
    days = lambda rs: f"{min(r['t'] for r in rs):%Y-%m-%d}".replace("%", "") if False else (
        f"{datetime.fromtimestamp(min(r['t'] for r in rs), timezone.utc):%Y-%m-%d} .. {datetime.fromtimestamp(max(r['t'] for r in rs), timezone.utc):%Y-%m-%d}")
    if hold:
        say(f"    holdout spans {days(hold)}; original spans {days(orig)}")
    say(f"    favourite tied for the highest mid: {sum(r['tie'] for r in recs)} of {len(recs)} events; favourite two-sided and 2c+ wide (an order): "
        f"{sum(r['ok1'] for r in recs)}")

    say()
    say("(1) THE FROZEN RULE")
    cells = [("ORIGINAL sample (reproduction; lead was +11.53% on 728 fills)", orig), ("HOLDOUT (the verdict)", hold), ("EVERYTHING", recs),
             ("holdout, recent tier (close >= 2026-09-19)", [r for r in hold if r["tier"] == "recent"]),
             ("holdout, archive tier", [r for r in hold if r["tier"] == "archive"]),
             ("holdout, daily temperature ladders only", ladders(hold)), ("everything, daily temperature ladders only", ladders(recs)),
             ("holdout, tied favourites dropped", [r for r in hold if not r["tie"]])]
    for label, rs in cells:
        x = evidence(rs)
        summary[label] = x
        say(f"    {label}")
        say(f"        {fmt(x)}")
        if "ret" in x:
            say(f"        by week [{x['wk_lo']:+.2f}, {x['wk_hi']:+.2f}] ({x['weeks']} weeks); mean net {x['cents']:+.2f}c per fill")
    hx = summary["HOLDOUT (the verdict)"]
    word, need = verdict(hx)
    summary["verdict"] = dict(word=word, fills_needed=need)
    say()
    say(f"    VERDICT ON THE HOLDOUT: {word}." + (f" Fills at which the day-clustered half-width would equal {'the observed point' if hx.get('ret', 0) > 0 else '3.0%'}:"
                                                f" about {need} (have {hx['fills']})." if need else ""))
    ax = summary["EVERYTHING"]
    say(f"    Same test on everything (not a holdout; includes the sample the rule was found on): {verdict(ax)[0]}.")
    mx = evidence(hold, side="yes")
    summary["holdout mirror"] = mx
    if "ret" in mx:
        say(f"    Mirror control on the holdout (YES bid at bid+1 on the favourite): fills {mx['fills']}, {mx['ret']:+.2f}% [{mx['lo']:+.2f}, {mx['hi']:+.2f}]")

    say()
    say("(2) BREAKDOWNS (fills; return; day-clustered interval). H = holdout, A = everything")
    def table(title, key, min_fills=1):
        say(f"    {title}")
        summary[title] = {}
        for g in sorted({key(r) for r in recs}):
            parts = []
            for tag, rs in (("H", hold), ("A", recs)):
                x = evidence([r for r in rs if key(r) == g], draws=2000)
                summary[title][f"{g}|{tag}"] = x
                parts.append(f"{tag}: {x['fills']:4d} fills " + (f"{x['ret']:+7.2f}% [{x['lo']:+7.2f}, {x['hi']:+7.2f}]" if "ret" in x else " "*27))
            say(f"        {str(g):18} " + "   ".join(parts))
    table("by month of the bid", lambda r: r["month"])
    table("by ladder", lambda r: kind(r["series"]))
    table("by city (high and low ladders together)", lambda r: city(r["series"]))
    table("by series", lambda r: r["series"])
    table("by price of the favourite (YES mid at the bid)", lambda r: "mid <30" if r["mid"] < 30 else "mid 30-40" if r["mid"] < 40 else "mid 40-50" if r["mid"] < 50
          else "mid 50-65" if r["mid"] < 65 else "mid 65+")
    pos = Counter()
    for tag, rs in (("H", hold), ("A", recs)):
        for g in sorted({city(r["series"]) for r in rs}):
            x = evidence([r for r in rs if city(r["series"]) == g], draws=200)
            if "ret" in x:
                pos[tag, x["ret"] > 0] += 1
    say(f"    cities with a positive point return: holdout {pos['H', True]} of {pos['H', True]+pos['H', False]}; everything {pos['A', True]} of {pos['A', True]+pos['A', False]}")

    say()
    say("(3) MECHANISM: is the favourite over-priced, or is it who gets filled?")
    say("    YES price of the favourite vs how often it settled YES (orders only). 'offer' = the YES price our NO bid sells at (ask - 1).")
    for label, rs in (("holdout", hold), ("everything", recs)):
        o = [r for r in rs if r["ok1"]]
        if not o:
            continue
        for name, part in (("all orders", o), ("filled", [r for r in o if r["fill1"]]), ("never filled", [r for r in o if not r["fill1"]])):
            if not part:
                continue
            won = np.array([r["y"] for r in part], float)
            b = cluster_boot(100*won, np.ones(len(part)), [r["day"] for r in part], 4000)
            every = cluster_boot([r["net1"] for r in part], [r["price1"] for r in part], [r["day"] for r in part], 4000)
            say(f"    {label:10} {name:12} n {len(part):5d} | mean bid {np.mean([r['bid'] for r in part]):5.1f} mid {np.mean([r['mid'] for r in part]):5.1f}"
                f" offer {np.mean([100-r['price1'] for r in part]):5.1f} | favourite won {b['ret']/100:5.1f}% [{b['lo']/100:4.1f}, {b['hi']/100:4.1f}]"
                f" | NO at the offer, if filled: {every['ret']:+6.2f}% [{every['lo']:+6.2f}, {every['hi']:+6.2f}]")
            summary[f"mechanism|{label}|{name}"] = dict(n=len(part), mid=float(np.mean([r["mid"] for r in part])), offer=float(np.mean([100-r["price1"] for r in part])),
                                                        won=b["ret"]/100, won_lo=b["lo"]/100, won_hi=b["hi"]/100, ret=every["ret"], lo=every["lo"], hi=every["hi"])
        f = [r for r in o if r["fill1"]]
        if f:
            wait = np.array([(r["fts1"]-r["qts"])/3600 for r in f])
            say(f"    {label:10} hours from the quote to the bar of the fill: median {np.median(wait):.0f}, quartiles {np.percentile(wait, 25):.0f}-{np.percentile(wait, 75):.0f};"
                f" fills in the first 6h {100*np.mean(wait <= 6):.0f}%: "
                + ", ".join(f"{name} {evidence([r for r in f if lo < (r['fts1']-r['qts'])/3600 <= hi], draws=2000).get('ret', float('nan')):+.1f}%"
                            f" (n {sum(lo < w <= hi for w in wait)})" for name, lo, hi in (("fill within 6h", -1, 6), ("6-14h", 6, 14), ("after 14h", 14, 999))))
    say("    Ladder position: where the settled bucket lies relative to the favourite at the bid (daily ladders, all six outcomes quoted).")
    say("    'priced' = mean normalised YES mid of the buckets at that offset; 'actual' = share of events settling there. Offsets in ladder order (+ = warmer).")
    for label, rs in (("holdout", hold), ("everything", recs)):
        for sub, part in (("all", rs), ("high ladders", [r for r in rs if kind(r["series"]) == "high"]), ("low ladders", [r for r in rs if kind(r["series"]) == "low"]),
                          ("filled orders", [r for r in rs if r["ok1"] and r["fill1"]]), ("unfilled orders", [r for r in rs if r["ok1"] and not r["fill1"]])):
            part = [r for r in part if r["priced"] is not None]
            if len(part) < 20:
                continue
            bins = (("<=-2", lambda d: d <= -2), ("-1", lambda d: d == -1), ("fav", lambda d: d == 0), ("+1", lambda d: d == 1), (">=+2", lambda d: d >= 2))
            cells_ = []
            for name, test in bins:
                priced = np.mean([sum(p for d, p in r["priced"].items() if test(d)) for r in part])
                actual = np.mean([test(r["offset"]) for r in part])
                cells_.append(f"{name} priced {100*priced:4.1f} actual {100*actual:4.1f}")
                summary[f"ladder|{label}|{sub}|{name}"] = dict(n=len(part), priced=100*priced, actual=100*actual)
            say(f"    {label:10} {sub:15} n {len(part):5d} | " + " | ".join(cells_))

    say()
    say("(4) ROBUSTNESS ONLY (not for re-picking). Clock x bid; holdout and everything. Fill always = a later print strictly above the YES offer.")
    for hours in (12, 24, 36, 48):
        rs = recs if hours == 24 else build(hours)[0]
        for k, name in ((0, "at the best NO bid"), (1, "+1c (headline)"), (2, "+2c")):
            for tag, part in (("H", [r for r in rs if r["sample"] == "holdout"]), ("A", rs)):
                x = evidence(part, k, draws=4000)
                summary[f"robust|{hours}h|{k}|{tag}"] = x
                say(f"    {hours:2d}h {name:19} {tag}: " + (fmt(x) if "ret" in x else f"events {x['events']}, orders {x['orders']}, fills {x['fills']} (too few)"))

    say()
    say("(5) CAPACITY")
    rec_f = [r for r in recs if r["tier"] == "recent" and r["ok1"] and r["fill1"]]
    rec_all = [r for r in recs if r["tier"] == "recent"]
    months = (ov.NOW-ov.CUTOFF)/86400/30.44
    cov = sum(1 for e in events if e["tier"] == "recent" and all(m["ticker"] in bars for m in e["picked"]))/max(1, sum(e["tier"] == "recent" for e in events))
    say(f"    recent tier (2026-09-19 .. 2026-11-18, {months:.2f} months, bars for {100*cov:.0f}% of the eligible events): {len(rec_all)} events with a quote,"
        f" {sum(r['ok1'] for r in rec_all)} orders, {len(rec_f)} fills = {len(rec_f)/months:.0f} fills a month ({len(rec_f)/months/30.44:.1f} a day)")
    summary["capacity"] = dict(months=months, fills=len(rec_f), fills_per_month=len(rec_f)/months)
    trades = ov.load(RAW/"trades.json", {})
    exact = {}
    for r in rec_f:
        tr = trades.get(r["ticker"])
        if tr is None:
            continue
        offer = 100-r["price1"]
        end = r["close"]+3600
        exact[r["ticker"]] = (sum(c for ts, p, c, _ in tr["trades"] if r["qts"] < ts <= end and p > offer), tr["complete"])
    if rec_f:
        bound = np.array([r["vol1"] for r in rec_f])
        won = np.array([not r["y"] for r in rec_f])
        say(f"    contracts in the hourly bars that printed through the bid (UPPER bound, whole bars), per fill: median {np.median(bound):.0f}, quartiles"
            f" {np.percentile(bound, 25):.0f}-{np.percentile(bound, 75):.0f}, mean {bound.mean():.0f}; winning fills median {np.median(bound[won]):.0f},"
            f" losing fills median {np.median(bound[~won]):.0f}")
    samp = [r for r in rec_f if r["ticker"] in exact]
    if samp:
        ex = np.array([exact[r["ticker"]][0] for r in samp])
        bd = np.array([r["vol1"] for r in samp])
        won = np.array([not r["y"] for r in samp])
        net = np.array([r["net1"] for r in samp])
        price = np.array([r["price1"] for r in samp])
        say(f"    EXACT, from the trade record of {len(samp)} filled recent favourites (sha256 order; {sum(not exact[r['ticker']][1] for r in samp)} truncated at 6,000 trades):")
        say(f"      a trade strictly through the bid after the quote exists for {100*np.mean(ex > 0):.1f}% of the bar-rule fills")
        say(f"      contracts strictly through per fill: median {np.median(ex):.0f}, quartiles {np.percentile(ex, 25):.0f}-{np.percentile(ex, 75):.0f}, mean {ex.mean():.0f};"
            f" exact / bar bound, median {np.median(ex[bd > 0]/bd[bd > 0]):.2f}")
        say(f"      winning fills (favourite lost, n {won.sum()}): median {np.median(ex[won]):.0f}, mean {ex[won].mean():.0f}; losing fills (n {(~won).sum()}):"
            f" median {np.median(ex[~won]):.0f}, mean {ex[~won].mean():.0f}")
        one = cluster_boot(net[ex > 0], price[ex > 0], [r["day"] for r, e_ in zip(samp, ex) if e_ > 0], 4000)
        say(f"      this sample at one contract: {one['ret']:+.2f}% [{one['lo']:+.2f}, {one['hi']:+.2f}]")
        say("      size per bid | filled share of size | return on money filled        | $ per fill | $ per month (x fills a month)")
        for size in (10, 100, 1000):
            got_ = np.minimum(size, ex)
            b = cluster_boot(got_*net, got_*price, [r["day"] for r in samp], 4000)
            per_fill = (got_*net).sum()/100/len(samp)
            lo_, hi_ = b["lo"]/b["ret"]*per_fill if b["ret"] else 0, b["hi"]/b["ret"]*per_fill if b["ret"] else 0
            lo_, hi_ = min(lo_, hi_), max(lo_, hi_)
            dollars = (got_*price).sum()/100/len(samp)
            say(f"      {size:5d}        | {100*got_.sum()/(size*len(samp)):5.1f}%               | {b['ret']:+6.2f}% [{b['lo']:+6.2f}, {b['hi']:+6.2f}]     | {per_fill:+8.2f}   |"
                f" {per_fill*len(rec_f)/months:+9.0f}  [{lo_*len(rec_f)/months:+.0f}, {hi_*len(rec_f)/months:+.0f}]   (money at work per fill ${dollars:,.0f})")
            summary[f"capacity|{size}"] = dict(ret=b["ret"], lo=b["lo"], hi=b["hi"], per_fill=per_fill, per_month=per_fill*len(rec_f)/months,
                                               per_month_lo=lo_*len(rec_f)/months, per_month_hi=hi_*len(rec_f)/months, filled_share=float(got_.sum()/(size*len(samp))))
    else:
        say("    no trade record on disk: only the bar upper bound is available")
    if rec_f:
        say("    Same with the bar UPPER bound on every recent fill (too generous: counts the whole hour's volume once any print is through):")
        bound = np.array([r["vol1"] for r in rec_f])
        net = np.array([r["net1"] for r in rec_f])
        price = np.array([r["price1"] for r in rec_f])
        for size in (10, 100, 1000):
            got_ = np.minimum(size, bound)
            b = cluster_boot(got_*net, got_*price, [r["day"] for r in rec_f], 4000)
            say(f"      {size:5d}: return {b['ret']:+6.2f}% [{b['lo']:+6.2f}, {b['hi']:+6.2f}], ${(got_*net).sum()/100/months:+,.0f} a month")
            summary[f"capacity_bound|{size}"] = dict(ret=b["ret"], lo=b["lo"], hi=b["hi"], per_month=(got_*net).sum()/100/months)

    say()
    say("(6) PLANTED EDGE: favourites' outcomes redrawn so the filled orders truly earn the stated return; fills, prices, days untouched.")
    say("    'confirmed' = the pre-registered test (day-clustered interval above zero and both halves positive). 300 redraws, 1,000 bootstrap draws each.")
    for label, rs in (("holdout", hold), ("everything", recs)):
        fills = [r for r in rs if r["ok1"] and r["fill1"]]
        if len(fills) < 50:
            continue
        price = np.array([r["price1"] for r in fills])
        obs = icc(fills, np.clip(price*(1+evidence(rs, draws=200)["ret"]/100)/100, 0.01, 0.99))
        rho, got_icc = match_rho(fills, obs)
        say(f"    {label}: {len(fills)} fills, {len({r['day'] for r in fills})} days; same-day correlation of the residuals {obs:+.3f} -> latent daily shock rho {rho:.2f}"
            f" (reproduces {got_icc:+.3f})")
        for r_ in sorted({0.0, rho}):
            for edge in (0.0, 3.0, 7.0):
                p = planted(fills, edge, r_)
                summary[f"planted|{label}|rho{r_}|{edge}"] = p
                say(f"      rho {r_:.2f} planted {edge:+.1f}% (true on these fills {p['true']:+.2f}%): estimate mean {p['mean']:+.2f}% sd {p['sd']:.2f}, mean half-width {p['half']:.2f}"
                    f" | interval above zero {p['interval']:5.1f}% | both halves positive {p['halves']:5.1f}% | confirmed {p['confirmed']:5.1f}%")

    say()
    say("(7) DIAGNOSTIC, OUTSIDE THE FROZEN UNIVERSE: does the 5,000-contract filter (the event's FINAL volume, unknowable at the bid) make the edge?")
    say("    Recent tier only, where every eligible event now has bars. Series grouped by how many of their recent events passed the filter (68 possible).")
    lb = ov.load(RAW/"lowvol-bars.json", {})
    low = [r for r in (score(e, lb, 24) for e in lowvol()) if r is not None]
    passed = Counter(e["series"] for e in events if e["tier"] == "recent")
    groups = (("every event passes (60+)", lambda n: n >= 60), ("31-59 pass", lambda n: 31 <= n < 60), ("30 or fewer pass", lambda n: n <= 30))
    say(f"    events that failed the filter, with bars and a quote 24h before close: {len(low)} of {len(lowvol())}")
    for name, test in groups:
        for tag, part in (("passed the filter (in the rule)", [r for r in rec_all if test(passed[r["series"]])]),
                          ("failed the filter (<5,000)", [r for r in low if test(passed[r["series"]])]),
                          ("no filter (both)", [r for r in rec_all+low if test(passed[r["series"]])])):
            x = evidence(part, draws=4000)
            summary[f"filter|{name}|{tag}"] = x
            won = f"favourite won {100*np.mean([r['y'] for r in part]):4.1f}% at mid {np.mean([r['mid'] for r in part]):4.1f}" if part else ""
            say(f"    {name:25} {tag:32} " + (f"fills {x['fills']:4d} of {x['orders']:4d} orders, {x['events']:4d} events | {x['ret']:+7.2f}% [{x['lo']:+7.2f}, {x['hi']:+7.2f}]"
                                                if "ret" in x else f"fills {x['fills']:4d} of {x['orders']:4d} orders, {x['events']:4d} events |") + f" | {won}")
    for tag, part in (("ALL recent ladders, passed the filter", ladders(rec_all)), ("ALL recent ladders, failed the filter", low), ("ALL recent ladders, no filter", ladders(rec_all)+low)):
        x = evidence(part)
        summary[f"filter|{tag}"] = x
        say(f"    {tag}")
        say(f"        {fmt(x) if 'ret' in x else x}")

    (OUT/"report.txt").write_text("\n".join(lines)+"\n")
    ov.dump(OUT/"summary.json", summary)
    ov.dump(OUT/"orders.json", [{k: v for k, v in r.items() if k != "priced"} for r in recs])


def status():
    events, _, _ = weather()
    rb, ab = ov.load(RAW/"recent-bars.json", {}), ov.load(RAW/"archive-bars.json", {})
    hr = [e for e in events if e["tier"] == "recent" and e["sample"] == "holdout"]
    ha = archive_order(events)
    print(f"budget used {used_seconds()/3600:.2f} of {BUDGET/3600:.0f} h")
    print(f"recent holdout events {len(hr)}, complete {sum(all(m['ticker'] in rb for m in e['picked']) for e in hr)}")
    print(f"archive holdout events {len(ha)}, complete {sum(all(m['ticker'] in ab for m in e['picked']) for e in ha)}; markets on disk {len(ab)}")
    print(f"trades on disk {len(ov.load(RAW/'trades.json', {}))}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("step", choices=["fees", "rbars", "abars", "trades", "xbars", "report", "status"])
    parser.add_argument("--minutes", type=float, default=50)
    args = parser.parse_args()
    if args.step == "report":
        return report()
    if args.step == "status":
        return status()
    if args.step == "xbars":                                          # its time was held back in the log; the real run replaces the reservation
        ov.dump(RAW/"download-log.json", [r for r in ov.load(RAW/"download-log.json", []) if r["step"] != "reserved"])
    left = BUDGET-used_seconds()
    if left <= 0:
        print("download budget of 5 hours is used up", flush=True)
        return
    RAW.mkdir(parents=True, exist_ok=True)
    start = time.time()
    deadline[0] = start+min(60*args.minutes, left)
    try:
        with httpx.Client(timeout=30) as client:
            dict(fees=fees_step, rbars=rbars_step, abars=abars_step, trades=trades_step, xbars=xbars_step)[args.step](client)
    except OutOfTime:
        print("out of time for this run", flush=True)
    finally:
        runs = ov.load(RAW/"download-log.json", [])
        runs.append(dict(step=args.step, started=int(start), seconds=int(time.time()-start), **log))
        ov.dump(RAW/"download-log.json", runs)
        print(dict(log), f"budget used {used_seconds()/3600:.2f} h", flush=True)


if __name__ == "__main__":
    main()
