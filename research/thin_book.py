"""Market making on the quieter of a tennis match's two books before play: does it earn anything?

Every Kalshi tennis match has two markets, "A wins" and "B wins"; YES on one is
NO on the other. The idea tested here: take the busier ("thick") book's
midpoint as fair value and rest a bid and an offer on the quieter ("thin") book
around it, before the match only, cancelling everything when play starts.

Commands

    download  one-minute bars of BOTH markets of recent settled matches (since
              2026-09-18) through the batch endpoint GET /markets/candlesticks,
              the six hours before settlement. All of ATP Challenger, WTA, ATP
              and WTA Challenger; the newest ITF_CAP matches of ITF men and
              women. One request at a time, cached in the shared cache.
    build     replay the cached batches into one minute grid per match
              (data/processed/thin-book/grid.pkl). No network.
    describe  the two books before play, by tour and favourite price band.
    simulate  the quoting strategies, fills, markouts and the ways of closing
              inventory, with bootstrap intervals over matches.
    trades    download the public trade record of both books for a sample of
              220 matches, check the simulated fills against it, and re-run the
              "join" strategy with an assumed number of contracts ahead of us.
    depth     a live snapshot (not cached) of displayed size at the best bid
              and offer of open matches that have not started.
    capacity  fills per day, dollars per day, order writes per day (needs the
              simulate and trades outputs).

Definitions

* Prices are in cents, in the THIN book's terms (YES of the thin book's
  player); the thick book is converted by 100 minus.
* Thick book: the one with more open interest at the end of the warm-up (known
  at the time). It turns out to be a weak label: see `describe`. `simulate`
  repeats the headline with the underdog's market as the thick book and with
  the book that turned out busier (which uses the future).
* Warm-up, as in research/favorite_dip.py: the first 15 bars of the thick book
  must be two-sided with a spread of at most 4c and midpoints within 3c, so the
  window really opens before play. The favourite's price is the median of
  those midpoints (the favourite's side).
* Start of play, as in research/favourite_gap.py (detect_start): the first bar
  after the warm-up where the midpoint has moved more than 2c in ten minutes or
  open interest jumped; the EARLIER of the two books' detections. Matches with
  no detected start are dropped. The pre-match period runs from the minute after
  both books' warm-ups to the detection bar.
* Bars exist only for minutes with activity; quotes are carried forward on a
  one-minute grid (checked: after a missing minute a bar always opens at the
  previous bar's closing quote).
* Quotes. At the close of minute t, F = thick book's midpoint at that close.
  The quote rests during minute t+1 only (so F is always the previous bar's).
  "k": bid floor(F-k), offer ceil(F+k), each placed only if it joins or
  improves the thin book's own quote at the close of t and does not cross it.
  "k=0.5" mirrors the thick book's own best quotes when its spread is 1c or 2c.
  "join": the thin book's own best bid and offer. "join edge 1": the same, but
  only on the side where that quote is at least 1c from F.
* Fills. "strict" (the rule of evidence used in this repository): a bid fills
  when a trade prints STRICTLY below it in minute t+1 (an offer: strictly
  above). "touch": a print at the price also fills, which assumes we are always
  first in the queue. "mixed": touch where our quote improved the thin book,
  strict where it joined. One contract per quote per minute, inventory
  unbounded (its range is reported). The truth lies between strict and touch;
  the `trades` step places it by assuming a queue ahead of us.
* Quotes placed at the close of the last pre-detection bar rest through the
  detection bar, because the start is known only once that bar has closed.
  The buffer variants stop quoting 10 or 30 minutes before the detection bar;
  that uses the future (the detection bar is defined by a later price move), so
  they are an upper bound for a trader who knows exactly when play starts.
* Closing inventory: (a) "settle": hold to settlement; (b) "exit": close the net
  position at the thick book's bid or ask at the close of the detection bar,
  paying the taker fee 0.07*p*(1-p); (c) round trips only (buys and sells
  matched first-in first-out, the rest ignored). "mid" marks the net position
  at the thick midpoint at the close of the detection bar without a fee: the
  expected value of holding to settlement with far less noise.
* Markout: thick-book midpoint h minutes after the fill bar, relative to the
  fill price, signed so that negative is adverse selection.
* Maker fee: zero where the series fee type is "quadratic" (ATP Challenger,
  WTA Challenger, ITF); ATP and WTA are "quadratic_with_maker_fees", rate
  assumed 0.0175*p*(1-p) (GET /series/{ticker} gives the type and a
  fee_multiplier of 1, not the rate).
* Intervals: 95% bootstrap over matches. Split: earliest 60% / latest 40% of
  matches by settlement time.

    .venv/bin/python research/thin_book.py download --minutes 25
    .venv/bin/python research/thin_book.py build describe simulate
    .venv/bin/python research/thin_book.py trades --minutes 11
    .venv/bin/python research/thin_book.py depth capacity

Public market data only; no orders, no authentication.
"""
import argparse
from collections import Counter, defaultdict
from datetime import date
import json
import math
from pathlib import Path
import pickle
import sys
import time

import httpx
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import config
from research.api_cost import ROOT, midnight, save, timestamp
from research.calibration_map import batch_bars, cache_file, chunks, listing
from research.favourite_gap import detect_start, side_path
from research.near_end import WINDOW

OUT = ROOT / "data/processed/thin-book"
CUTOFF = midnight(date(2026, 9, 18))
TOURS = {"KXATPCHALLENGERMATCH": "Challenger", "KXWTAMATCH": "WTA", "KXATPMATCH": "ATP",
         "KXWTACHALLENGERMATCH": "WTA Challenger", "KXITFMATCH": "ITF men", "KXITFWMATCH": "ITF women"}
MAKER_FEE = ("WTA", "ATP")                               # fee_type "quadratic_with_maker_fees"
ITF_CAP = 800
WARMUP, MINUTES = 15, WINDOW//60
BANDS = ((50, 60), (60, 70), (70, 80), (80, 90), (90, 101))
RNG = np.random.default_rng(11)
last_request = [0.0]


def events():
    """Recent settled two-market matches from the cached listings: (settle, series, event, archived, markets)."""
    out = []
    for series in TOURS:
        rows = [e for e in listing(None, series, archive=False).values() if e[0] >= CUTOFF and len(e[4]) == 2]
        rows.sort(key=lambda e: (-e[0], e[2]))
        out += rows[:ITF_CAP] if series.startswith("KXITF") else rows
    return out


def jobs_of(rows):
    return [(e[0]-WINDOW, [m["ticker"] for m in e[4]]) for e in rows]


def read(client, path, params, pace):
    """One paced public read into the shared cache (same file layout as research.api_cost.fetch). Returns (data, throttled)."""
    file = cache_file(path, params)
    if file.exists():
        return json.loads(file.read_text()), 0
    throttled = 0
    for pause in (20, 45, 90, 180):
        wait = last_request[0]+pace-time.time()
        if wait > 0:
            time.sleep(wait)
        try:
            resp = client.get(config.KALSHI_PROD_BASE+path, params=params)
        except httpx.RequestError:
            resp = None
        last_request[0] = time.time()
        if resp is not None and resp.status_code == 200:
            data = resp.json()
            temp = file.with_suffix(".tmp")               # compact: api_cost.save indents, which takes seconds on a 20 MB batch
            temp.write_text(json.dumps(data))
            temp.replace(file)
            return data, throttled
        if resp is not None and resp.status_code not in (429,) and resp.status_code < 500:
            raise RuntimeError(f"{path} returned HTTP {resp.status_code}")
        throttled += 1
        time.sleep(pause)                                 # HTTP 429 or a server error: a live trader shares the limit, so wait long
    raise RuntimeError(f"Read failed after long back-off: {path}")


def download(minutes, pace):
    deadline = time.time()+60*minutes
    rows = events()
    groups = chunks(jobs_of(rows), MINUTES)
    todo = [g for g in groups if not cache_file("/markets/candlesticks", params_of(g)).exists()]
    print(f"{len(rows)} matches in {len(groups)} batch requests, {len(todo)} not yet cached", flush=True)
    throttled = 0
    with httpx.Client(timeout=60) as client:
        for n, group in enumerate(todo):
            if time.time() > deadline:
                print(f"  out of time after {n} of {len(todo)}")
                break
            throttled += read(client, "/markets/candlesticks", params_of(group), pace)[1]
            if (n+1) % 50 == 0:
                print(f"  {n+1} of {len(todo)}, throttled {throttled} times", flush=True)
    print(f"done; throttled {throttled} times")


def params_of(group):
    return dict(market_tickers=",".join(t for _, ts_ in group for t in ts_), start_ts=group[0][0], end_ts=group[-1][0]+WINDOW,
                period_interval=1)


# ---------------------------------------------------------------- build

def warm(path):
    """Median midpoint of the first 15 bars if they are two-sided, at most 4c wide and within 3c; else None."""
    mids = []
    for p in path[:WARMUP]:
        if p[1] is None or p[2] is None or not 0 < p[1] <= p[2] < 100 or p[2]-p[1] > 4:
            return None
        mids.append((p[1]+p[2])/2)
    if len(mids) < WARMUP or max(mids)-min(mids) > 3:
        return None
    return float(np.median(mids))


def book(path, g0):
    """One market on the minute grid: quotes carried forward, prints and volume only where a bar has them."""
    n = MINUTES+1
    a = dict(bid=np.full(n, np.nan), ask=np.full(n, np.nan), lo=np.full(n, np.nan), hi=np.full(n, np.nan),
             vol=np.zeros(n), oi=np.full(n, np.nan), bar=np.zeros(n, bool))
    for ts_, bid, ask, low, high, vol, oi in path:
        i = (ts_-g0)//60
        if not 0 <= i < n:
            continue
        a["bar"][i] = True
        if bid is not None and ask is not None and 0 <= bid <= ask <= 100:
            a["bid"][i], a["ask"][i] = bid, ask
        if low is not None and vol > 0:
            a["lo"][i], a["hi"][i] = low, high
        a["vol"][i], a["oi"][i] = vol, oi
    for key in ("bid", "ask", "oi"):
        v = a[key]
        idx = np.where(~np.isnan(v), np.arange(n), 0)
        np.maximum.accumulate(idx, out=idx)
        a[key] = np.where(np.isnan(v[idx]), np.nan, v[idx])
    return a


def build():
    rows = events()
    lookup = {(e[0]-WINDOW, e[4][0]["ticker"]): e for e in rows}
    out, cover = [], Counter()
    for group in chunks(jobs_of(rows), MINUTES):
        got = batch_bars(None, group, 0, WINDOW, 0)
        if got is None:
            cover["not downloaded"] += len(group)
            continue
        for anchor, tickers in group:
            e = lookup[anchor, tickers[0]]
            tour = TOURS[e[1]]
            cover[tour+" listed"] += 1
            paths = [side_path([b for b in got.get(t, []) if anchor < b["end_period_ts"] <= e[0]], True) for t in tickers]
            if min(len(p) for p in paths) < WARMUP+5:
                cover[tour+" few bars"] += 1
                continue
            g0 = (anchor//60+1)*60
            began = [detect_start(p) for p in paths]
            out.append(dict(event=e[2], tour=tour, settle=e[0], tickers=tickers, payout=[m["_outcome"] for m in e[4]], g0=g0,
                            volume=[float(m.get("volume_fp") or 0) for m in e[4]],
                            warm=[warm(p) for p in paths], warm_end=[(p[WARMUP-1][0]-g0)//60 for p in paths],
                            warm_oi=[p[WARMUP-1][6] for p in paths], warm_vol=[sum(x[5] for x in p[:WARMUP]) for p in paths],
                            began=[None if b is None else (p[b][0]-g0)//60 for b, p in zip(began, paths)],
                            books=[book(p, g0) for p in paths]))
            cover[tour+" built"] += 1
    print(dict(sorted(cover.items())))
    with open(OUT/"grid.pkl", "wb") as f:
        pickle.dump(out, f)
    return out


def load():
    with open(OUT/"grid.pkl", "rb") as f:
        return pickle.load(f)



# ---------------------------------------------------------------- shared

def taker_fee(price):
    return 7*price/100*(1-price/100)


def maker_fee(price):
    return 1.75*price/100*(1-price/100)


def view(m, thick_rule="oi"):
    """One match in the thin book's terms, or (None, reason). Lower-case arrays are the thin book, upper-case the thick."""
    if thick_rule == "oi":
        k = 0 if (m["warm_oi"][0], m["warm_vol"][0]) >= (m["warm_oi"][1], m["warm_vol"][1]) else 1
    elif thick_rule == "expost":                            # the book that turned out to trade more before play (uses the future)
        began = [b for b in m["began"] if b is not None]
        s = slice(max(m["warm_end"])+1, min(began) if began else None)
        k = 0 if m["books"][0]["vol"][s].sum() >= m["books"][1]["vol"][s].sum() else 1
    else:                                                   # "underdog": the market of the player priced below 50c
        w = m["warm"][0] if m["warm"][0] is not None else (None if m["warm"][1] is None else 100-m["warm"][1])
        if w is None:
            return None, "warm-up not clean"
        k = 0 if w < 50 else 1
    if m["warm"][k] is None:
        return None, "warm-up not clean"
    began = [b for b in m["began"] if b is not None]
    if not began:
        return None, "no start detected"
    first, start = max(m["warm_end"])+1, min(began)
    if start-first < 10:
        return None, "pre-match under 10 minutes"
    thin, thick = m["books"][1-k], m["books"][k]
    v = dict(event=m["event"], tour=m["tour"], settle=m["settle"], g0=m["g0"], thin=m["tickers"][1-k], thick=m["tickers"][k],
             first=first, start=start, payout=100*m["payout"][1-k], fav=max(m["warm"][k], 100-m["warm"][k]),
             thin_is_favourite=m["warm"][k] < 50, b=thin["bid"], a=thin["ask"], lo=thin["lo"], hi=thin["hi"], vol=thin["vol"],
             B=100-thick["ask"], A=100-thick["bid"], LO=100-thick["hi"], HI=100-thick["lo"], VOL=thick["vol"],
             both_detected=len(began) == 2, market_volume=(m["volume"][1-k], m["volume"][k]))
    v["ok"] = (v["b"] > 0) & (v["b"] <= v["a"]) & (v["a"] < 100) & (v["B"] > 0) & (v["B"] <= v["A"]) & (v["A"] < 100)
    v["mid"], v["F"] = (v["b"]+v["a"])/2, (v["B"]+v["A"])/2
    return v, None


def views(rows, thick_rule="oi"):
    out, why = [], Counter()
    for m in rows:
        v, reason = view(m, thick_rule)
        why[m["tour"]+" | "+(reason or "used")] += 1
        if v:
            out.append(v)
    return out, why


def band(v):
    return next(f"{lo}-{min(hi, 100)}" for lo, hi in BANDS if lo <= v["fav"] < hi)


def cells(vs):
    """(label, matches) for every tour, every tour x band, and all tours by band."""
    out = []
    for tour in list(TOURS.values())+["all tours"]:
        part = [v for v in vs if tour == "all tours" or v["tour"] == tour]
        out.append((f"{tour} | all", part))
        for lo, hi in BANDS:
            out.append((f"{tour} | {lo}-{min(hi, 100)}", [v for v in part if lo <= v["fav"] < hi]))
    return [(k, p) for k, p in out if p]


def ratio_ci(num, den, draws=2000):
    """Bootstrap interval for sum(num)/sum(den), resampling matches."""
    num, den = np.asarray(num, float), np.asarray(den, float)
    if len(num) < 10 or den.sum() == 0:
        return None
    pick = RNG.integers(0, len(num), (draws, len(num)))
    d = den[pick].sum(1)
    r = num[pick].sum(1)[d > 0]/d[d > 0]
    return [round(float(x), 2) for x in np.percentile(r, [2.5, 97.5])]


def fmt(ci):
    return "      n/a      " if ci is None else f"[{ci[0]:+6.2f},{ci[1]:+6.2f}]"


# ---------------------------------------------------------------- describe

def describe_one(v):
    s = slice(v["first"], v["start"])
    ok = v["ok"][s]
    gap = (v["mid"]-v["F"])[s][ok]
    out = dict(minutes=v["start"]-v["first"], quoted=int(ok.sum()), vol_thin=float(v["vol"][s].sum()), vol_thick=float(v["VOL"][s].sum()),
               trade_minutes_thin=int((v["vol"][s] > 0).sum()), trade_minutes_thick=int((v["VOL"][s] > 0).sum()),
               thin_is_favourite=bool(v["thin_is_favourite"]))
    half = (v["first"]+v["start"])//2
    early, late = [(v["vol"][x].sum(), v["VOL"][x].sum()) for x in (slice(v["first"], half), slice(half, v["start"]))]
    if sum(early) and sum(late):                            # does the busier book of the first half stay the busier one?
        out["leader_persists"] = bool((early[0] > early[1]) == (late[0] > late[1]))
    if not ok.sum():
        return out
    b, a, B, A = (v[k][s][ok] for k in "baBA")
    out.update(spread_thin=float((a-b).mean()), spread_thick=float((A-B).mean()),
               gap1=float((abs(gap) >= 1).mean()), gap2=float((abs(gap) >= 2).mean()), gap3=float((abs(gap) >= 3).mean()),
               thin_wider=float(((a-b) > (A-B)).mean()), thin_inside=float(((b > B) | (a < A)).mean()),
               crossed=float(((a < B) | (b > A)).mean()), locked=float(((a == B) | (b == A)).mean()),
               crossed_after_fee=float(((B-a > taker_fee(a)+taker_fee(B)) | (b-A > taker_fee(b)+taker_fee(A))).mean()),
               crossed_minutes=int(((a < B) | (b > A)).sum()),
               crossed_after_fee_minutes=int(((B-a > taker_fee(a)+taker_fee(B)) | (b-A > taker_fee(b)+taker_fee(A))).sum()))
    # Episodes: runs of consecutive quoted minutes with the midpoints at least 1c apart on the same side.
    idx = np.arange(v["first"], v["start"])[ok]
    sign = np.where(abs(gap) >= 1, np.sign(gap), 0)
    runs, i = [], 0
    while i < len(sign):
        if sign[i] == 0:
            i += 1
            continue
        j = i
        while j+1 < len(sign) and sign[j+1] == sign[i] and idx[j+1] == idx[j]+1:
            j += 1
        closed = j+1 < len(sign) and idx[j+1] == idx[j]+1
        run = dict(length=int(j-i+1), closed=bool(closed), size=float(abs(gap[i])))
        if closed:      # which book moved between the first gap minute and the first minute without one
            run["thin_move"] = float(-sign[i]*(v["mid"][idx[j+1]]-v["mid"][idx[i]]))
            run["thick_move"] = float(sign[i]*(v["F"][idx[j+1]]-v["F"][idx[i]]))
        runs.append(run)
        i = j+1
    out["runs"] = runs
    # Lead-lag at one minute: next-minute change of each midpoint against this minute's gap and the other book's change.
    t = np.arange(v["first"]+1, v["start"]-1)
    t = t[v["ok"][t-1] & v["ok"][t] & v["ok"][t+1]]
    if len(t):
        g = (v["mid"]-v["F"])[t]
        d_thin, d_thick = v["mid"][t+1]-v["mid"][t], v["F"][t+1]-v["F"][t]
        p_thin, p_thick = v["mid"][t]-v["mid"][t-1], v["F"][t]-v["F"][t-1]
        out["sums"] = [float(x) for x in (len(t), (g*g).sum(), (g*d_thin).sum(), (g*d_thick).sum(), (p_thick*d_thin).sum(), (p_thick**2).sum(),
                                          (p_thin*d_thick).sum(), (p_thin**2).sum(), (d_thin != 0).sum(), (d_thick != 0).sum())]
    return out


def describe(rows):
    vs, why = views(rows)
    print("\nSAMPLE (thick book = more open interest at the end of the warm-up)")
    for k, n in sorted(why.items()):
        print(f"  {k:45} {n}")
    stats = [(v, describe_one(v)) for v in vs]
    report = dict(sample=dict(why))
    und = [d["vol_thick"]/(d["vol_thin"]+d["vol_thick"]) for v, d in stats if d["vol_thin"]+d["vol_thick"] > 0]
    agree = np.mean([v["thin_is_favourite"] for v in vs])
    expost = np.mean([d["vol_thick"] >= d["vol_thin"] for v, d in stats if d["vol_thin"]+d["vol_thick"] > 0])
    report["thick_rule"] = dict(thick_is_underdog_market=float(agree), thick_has_more_prematch_volume=float(expost))
    print(f"  the thick book is the underdog's market in {100*agree:.0f}% of matches and has the larger pre-match volume in {100*expost:.0f}%")
    print("\n(2) THE TWO BOOKS BEFORE PLAY. minutes = pre-match minutes in the window; contracts = median per match; spreads in cents;"
          "\n    gap = thin midpoint minus thick midpoint (thin terms), share of quoted minutes; crossed = one book's bid above the other's ask")
    print(f"  {'cell':24} {'n':>5} {'min':>4} {'thin':>7} {'thick':>7} {'thin%':>5} {'no thin':>7} {'spr thin':>8} {'spr thick':>9} {'gap1+':>6} {'gap2+':>6} {'gap3+':>6} "
          f"{'run min':>7} {'closed by thin':>14} {'crossed':>7} {'after fee':>9}")
    for label, part in cells(vs):
        ds = [d for v, d in stats if v in part_set(part)]
        have = [d for d in ds if "gap1" in d]
        tot_thin, tot_thick = sum(d["vol_thin"] for d in ds), sum(d["vol_thick"] for d in ds)
        runs = [r for d in have for r in d["runs"]]
        closed = [r for r in runs if r["closed"]]
        thin_part = sum(r["thin_move"] for r in closed)
        both = thin_part+sum(r["thick_move"] for r in closed)
        sums = np.sum([d["sums"] for d in have if "sums" in d], axis=0) if any("sums" in d for d in have) else None
        q = sum(d["quoted"] for d in have)
        r = dict(matches=len(ds), median_minutes=float(np.median([d["minutes"] for d in ds])),
                 minutes_quartiles=[float(x) for x in np.percentile([d["minutes"] for d in ds], [25, 75])],
                 median_contracts_thin=float(np.median([d["vol_thin"] for d in ds])), median_contracts_thick=float(np.median([d["vol_thick"] for d in ds])),
                 mean_contracts_thin=tot_thin/len(ds), mean_contracts_thick=tot_thick/len(ds),
                 thin_share_of_volume=tot_thin/(tot_thin+tot_thick) if tot_thin+tot_thick else None,
                 median_thin_share=float(np.median([d["vol_thin"]/(d["vol_thin"]+d["vol_thick"]) for d in ds if d["vol_thin"]+d["vol_thick"] > 0])),
                 share_no_thin_trade=float(np.mean([d["vol_thin"] == 0 for d in ds])),
                 favourite_market_share_of_volume=sum(d["vol_thin"] if d["thin_is_favourite"] else d["vol_thick"] for d in ds)/(tot_thin+tot_thick) if tot_thin+tot_thick else None,
                 median_busier_book_share=float(np.median([max(d["vol_thin"], d["vol_thick"])/(d["vol_thin"]+d["vol_thick"]) for d in ds if d["vol_thin"]+d["vol_thick"] > 0])),
                 busier_book_is_favourites=float(np.mean([(d["vol_thin"] > d["vol_thick"]) == d["thin_is_favourite"] for d in ds if d["vol_thin"] != d["vol_thick"]])),
                 busier_book_persists_half_to_half=float(np.mean([d["leader_persists"] for d in ds if "leader_persists" in d])),
                 thick_rule_picks_busier_book=float(np.mean([d["vol_thick"] > d["vol_thin"] for d in ds if d["vol_thin"] != d["vol_thick"]])),
                 median_trade_minutes_thin=float(np.median([d["trade_minutes_thin"] for d in ds])),
                 median_trade_minutes_thick=float(np.median([d["trade_minutes_thick"] for d in ds])),
                 spread_thin=float(np.mean([d["spread_thin"] for d in have])), spread_thick=float(np.mean([d["spread_thick"] for d in have])),
                 thin_wider_share=float(np.mean([d["thin_wider"] for d in have])), thin_quote_inside_thick=float(np.mean([d["thin_inside"] for d in have])),
                 gap1=sum(d["gap1"]*d["quoted"] for d in have)/q, gap2=sum(d["gap2"]*d["quoted"] for d in have)/q,
                 gap3=sum(d["gap3"]*d["quoted"] for d in have)/q, matches_with_any_gap1=float(np.mean([d["gap1"] > 0 for d in have])),
                 gap_runs=len(runs), run_median_minutes=float(np.median([r["length"] for r in runs])) if runs else None,
                 run_mean_minutes=float(np.mean([r["length"] for r in runs])) if runs else None,
                 run_share_over_10_minutes=float(np.mean([r["length"] > 10 for r in runs])) if runs else None,
                 share_of_closing_done_by_thin=thin_part/both if both else None,
                 crossed=sum(d["crossed"]*d["quoted"] for d in have)/q, locked=sum(d["locked"]*d["quoted"] for d in have)/q,
                 crossed_after_fee=sum(d["crossed_after_fee"]*d["quoted"] for d in have)/q,
                 crossed_after_fee_minutes=sum(d["crossed_after_fee_minutes"] for d in have),
                 matches_ever_crossed_after_fee=sum(d["crossed_after_fee_minutes"] > 0 for d in have), quoted_minutes=q)
        if sums is not None:
            r["lead_lag"] = dict(minutes=int(sums[0]), thin_next_change_per_cent_of_gap=sums[2]/sums[1], thick_next_change_per_cent_of_gap=sums[3]/sums[1],
                                 thin_follows_thick=sums[4]/sums[5] if sums[5] else None, thick_follows_thin=sums[6]/sums[7] if sums[7] else None,
                                 thin_mid_changes_per_hour=60*sums[8]/sums[0], thick_mid_changes_per_hour=60*sums[9]/sums[0])
        report[label] = r
        print(f"  {label:24} {r['matches']:5} {r['median_minutes']:4.0f} {r['median_contracts_thin']:7.0f} {r['median_contracts_thick']:7.0f} "
              f"{100*r['thin_share_of_volume']:5.0f} {100*r['share_no_thin_trade']:6.0f}% {r['spread_thin']:8.2f} {r['spread_thick']:9.2f} "
              f"{100*r['gap1']:5.1f}% {100*r['gap2']:5.1f}% {100*r['gap3']:5.1f}% {r['run_median_minutes'] or 0:7.0f} "
              f"{100*(r['share_of_closing_done_by_thin'] or 0):13.0f}% {100*r['crossed']:6.2f}% {100*r['crossed_after_fee']:8.3f}%")
    print("\n  which book is the busy one before play (contracts traded between the warm-up and the detected start)")
    print(f"  {'cell':24} {'favourite mkt share':>19} {'busier is favourite mkt':>23} {'busier book share (median)':>26} {'same leader both halves':>23} {'open-interest rule right':>24}")
    for label, r in report.items():
        if "favourite_market_share_of_volume" in r and (label.endswith("all") or label.startswith("all tours")):
            print(f"  {label:24} {100*r['favourite_market_share_of_volume']:18.0f}% {100*r['busier_book_is_favourites']:22.0f}% {100*r['median_busier_book_share']:25.0f}% "
                  f"{100*r['busier_book_persists_half_to_half']:22.0f}% {100*r['thick_rule_picks_busier_book']:23.0f}%")
    print("\n  lead-lag at one minute (pooled minutes): next-minute change of each midpoint per cent of gap, and per cent of the other book's last change")
    for label in [f"{t} | all" for t in list(TOURS.values())+["all tours"]]:
        if label in report and "lead_lag" in report[label]:
            x = report[label]["lead_lag"]
            print(f"  {label:24} thin closes {-100*x['thin_next_change_per_cent_of_gap']:5.1f}% of a gap in the next minute, thick {100*x['thick_next_change_per_cent_of_gap']:5.1f}%; "
                  f"thin follows thick {x['thin_follows_thick']:+.3f}, thick follows thin {x['thick_follows_thin']:+.3f}; "
                  f"midpoint changes per hour thin {x['thin_mid_changes_per_hour']:.1f} thick {x['thick_mid_changes_per_hour']:.1f}")
    save(OUT/"describe.json", report)


def part_set(part):
    return _Ids(part)


class _Ids:
    """Membership by identity (views hold arrays, so they cannot be hashed or compared)."""
    def __init__(self, part):
        self.ids = {id(v) for v in part}

    def __contains__(self, v):
        return id(v) in self.ids


# ---------------------------------------------------------------- simulate

VARIANTS = ("join", "join edge 1", "k=0.5", "k=1", "k=2", "k=3", "offer both")
HORIZONS = (0, 1, 5, 15)


def quotes(v, variant, buffer=0):
    """Quote times t (close of minute t; the quote rests during t+1) with the bid and offer prices (nan = not placed)."""
    t = np.arange(v["first"]-1, v["start"]-buffer)
    t = t[t+1 < len(v["b"])]
    b, a, F, ok = v["b"][t], v["a"][t], v["F"][t], v["ok"][t]
    if variant == "offer both":                             # join the YES offer of each book; the thick one is a bid in thin terms
        none = np.zeros(len(t), bool)
        return t, np.where(ok, v["B"][t], np.nan), np.where(ok, a, np.nan), none, none
    if variant.startswith("join"):
        qb, qa = b.copy(), a.copy()
        if variant == "join edge 1":                        # only where the thin book's own quote is at least 1c from fair value
            qb[~(qb <= F-1)] = np.nan
            qa[~(qa >= F+1)] = np.nan
    else:
        k = float(variant[2:])
        qb, qa = np.floor(F-k), np.ceil(F+k)
    with np.errstate(invalid="ignore"):
        qb = np.where(ok & (qb >= b) & (qb < a) & (qb >= 1), qb, np.nan)
        qa = np.where(ok & (qa <= a) & (qa > b) & (qa <= 99), qa, np.nan)
        return t, qb, qa, qb > b, qa < a


def simulate_one(v, variant, buffer=0, mode="strict", fee=True):
    """mode: "strict" (a print strictly through the quote), "touch" (a print at the quote also fills: front of the queue),
    "mixed" (touch where our quote improved the thin book and so stood alone at its price, strict where it joined a queue)."""
    t, qb, qa, better_b, better_a = quotes(v, variant, buffer)
    lo, hi = (v["LO"] if variant == "offer both" else v["lo"])[t+1], v["hi"][t+1]
    with np.errstate(invalid="ignore"):
        buy, sell = lo < qb, hi > qa
        touch = int((lo <= qb).sum()+(hi >= qa).sum())
        if mode == "touch":
            buy, sell = lo <= qb, hi >= qa
        elif mode == "mixed":
            buy, sell = buy | (better_b & (lo <= qb)), sell | (better_a & (hi >= qa))
    fills = sorted([(int(i+1), 1, float(p)) for i, p in zip(t[buy], qb[buy])]+[(int(i+1), -1, float(p)) for i, p in zip(t[sell], qa[sell])])
    return dict(account(v, fills, fee), bid_minutes=int((qb == qb).sum()), offer_minutes=int((qa == qa).sum()),
                improving_minutes=int(better_b.sum()+better_a.sum()), writes=writes(qb)+writes(qa), touch=touch)


def account(v, fills, fee=True):
    """Profit and markouts of one match's fills [(grid minute, +1 buy / -1 sell, price)], one contract each."""
    n = len(v["F"])
    rate = maker_fee if fee and v["tour"] in MAKER_FEE else (lambda p: 0.0)
    fees = sum(rate(p) for _, _, p in fills)
    cash = -sum(s*p for _, s, p in fills)-fees
    q = sum(s for _, s, _ in fills)
    # (b) exit the net position at the thick book's quote at the close of the detection bar, paying the taker fee
    e = v["start"]
    while e > 0 and not (v["B"][e] > 0 and v["A"][e] < 100):
        e -= 1
    px = v["B"][e] if q > 0 else v["A"][e]
    exit_b = q*px-abs(q)*taker_fee(px) if q else 0.0
    e2 = v["start"]-1
    px2 = v["B"][e2] if q > 0 else v["A"][e2]
    exit_before = q*px2-abs(q)*taker_fee(px2) if q and px2 == px2 else exit_b
    buys, sells = [p for _, s, p in fills if s > 0], [p for _, s, p in fills if s < 0]
    trips = min(len(buys), len(sells))
    inv = np.cumsum([s for _, s, _ in fills]) if fills else np.array([0])
    out = dict(event=v["event"], settle=v["settle"], fills=len(fills), buys=len(buys), sells=len(sells), fees=fees,
               settle_pnl=cash+q*v["payout"], exit_pnl=cash+exit_b, exit_before_pnl=cash+exit_before, mid_pnl=cash+q*v["F"][e] if q else cash,
               mid_before_pnl=cash+q*v["F"][e2] if q and v["F"][e2] == v["F"][e2] else cash+(q*v["F"][e] if q else 0),
               trips=trips, trip_pnl=sum(sells[:trips])-sum(buys[:trips])-(fees*2*trips/len(fills) if fills else 0),
               max_inventory=int(abs(inv).max()), end_inventory=int(q),
               detection_bar_fills=sum(i == v["start"] for i, _, _ in fills),
               edge=sum(s*(v["F"][i-1]-p) for i, s, p in fills), fill_volume=[float(v["vol"][i]) for i, _, _ in fills])
    for h in HORIZONS:
        marks = [s*(v["F"][min(i+h, n-1)]-p) for i, s, p in fills]
        marks = [x for x in marks if x == x]
        out[f"mark{h}"], out[f"mark{h}_n"] = sum(marks), len(marks)
    out["fill_list"] = fills
    return out


def writes(q):
    """Order writes for one side: a placement or a cancel each time the resting price changes (nan = nothing resting)."""
    prev, n = np.nan, 0
    for x in q:
        if not (x == prev or (x != x and prev != prev)):
            n += (x == x)+(prev == prev)                   # cancel the old one, place the new one
        prev = x
    return n+(prev == prev)                                # final cancel at the start of play


def summarise(res):
    """One cell: res is a list of per-match results."""
    fills = [r["fills"] for r in res]
    total = sum(fills)
    out = dict(matches=len(res), quote_minutes=sum(r.get("bid_minutes", 0)+r.get("offer_minutes", 0) for r in res),
               improving_share=sum(r.get("improving_minutes", 0) for r in res)/max(1, sum(r.get("bid_minutes", 0)+r.get("offer_minutes", 0) for r in res)),
               fills=total, touch=sum(r.get("touch", 0) for r in res),
               fills_per_match=total/len(res), matches_with_a_fill=float(np.mean([f > 0 for f in fills])),
               round_trips=sum(r["trips"] for r in res), writes_per_match=float(np.mean([r.get("writes", 0) for r in res])),
               max_inventory_p95=float(np.percentile([r["max_inventory"] for r in res], 95)), max_inventory=max(r["max_inventory"] for r in res),
               detection_bar_fills=sum(r["detection_bar_fills"] for r in res))
    if not total:
        return out
    trips = [r["trips"] for r in res]
    out["spread_per_round_trip"] = sum(r["trip_pnl"] for r in res)/sum(trips) if sum(trips) else None
    out["spread_per_round_trip_ci"] = ratio_ci([r["trip_pnl"] for r in res], trips)
    out["edge_at_quote"] = sum(r["edge"] for r in res)/total
    for h in HORIZONS:
        out[f"markout_{h}m"] = sum(r[f"mark{h}"] for r in res)/max(1, sum(r[f"mark{h}_n"] for r in res))
        out[f"markout_{h}m_ci"] = ratio_ci([r[f"mark{h}"] for r in res], [r[f"mark{h}_n"] for r in res])
    for key in ("settle_pnl", "exit_pnl", "exit_before_pnl", "mid_pnl", "mid_before_pnl"):
        x = [r[key] for r in res]
        out[key+"_per_fill"], out[key+"_per_fill_ci"] = sum(x)/total, ratio_ci(x, fills)
        out[key+"_per_match"], out[key+"_per_match_ci"] = sum(x)/len(res), ratio_ci(x, np.ones(len(res)))
    order = sorted(res, key=lambda r: (r["settle"], r["event"]))
    cut = int(.6*len(order))
    for name, part in (("early60", order[:cut]), ("late40", order[cut:])):
        n = sum(r["fills"] for r in part)
        out[name] = dict(matches=len(part), fills=n, exit_per_fill=sum(r["exit_pnl"] for r in part)/n if n else None,
                         settle_per_fill=sum(r["settle_pnl"] for r in part)/n if n else None,
                         markout_5m=sum(r["mark5"] for r in part)/max(1, sum(r["mark5_n"] for r in part)),
                         exit_per_fill_ci=ratio_ci([r["exit_pnl"] for r in part], [r["fills"] for r in part]))
    return out


def simulate(rows, buffers=(0, 10, 30)):
    vs, _ = views(rows)
    report = {}
    print("\n(3) SIMULATION. cents per contract; ATP and WTA include the assumed maker fee; intervals resample matches."
          "\n    q-min = quote-minutes placed (bid and offer counted separately); RT = round trips; c/RT = cents captured per round trip"
          "\n    edge = fair value minus fill price when quoted; mark = thick midpoint h minutes after the fill bar minus fill price (negative: adverse)"
          "\n    exit = net position closed at the thick book's quote at the close of the detection bar with the taker fee;"
          "\n    mid = net position marked at the thick midpoint there, no fee; settle = held to settlement; 60%/40% = exit per fill, earlier and later matches")
    head = (f"  {'cell':24} {'n':>5} {'q-min':>7} {'fills':>6} {'/match':>6} {'RT':>5} {'c/RT':>6} {'edge':>5} {'mark0':>6} {'mark1':>6} {'mark5':>6} "
            f"{'mark15':>6} {'mark5 interval':>15} {'exit/fill':>9} {'interval':>15} {'mid/fill':>8} {'interval':>15} {'settle/fill':>11} {'interval':>15} "
            f"{'exit/match':>10} {'interval':>15} {'60%':>6} {'40%':>6}")
    for mode, buffer in [("strict", b) for b in buffers]+[("touch", 0), ("mixed", 0), ("touch", 30)]:
        for variant in VARIANTS:
            res = {id(v): simulate_one(v, variant, buffer, mode) for v in vs}
            if buffer == 0 and mode != "mixed":
                save(OUT/f"fills-{mode}-{variant.replace(' ', '-').replace('=', '')}.json",
                     [dict(event=v["event"], tour=v["tour"], band=band(v), thin=v["thin"], thick=v["thick"], g0=v["g0"], first=v["first"], start=v["start"],
                           fills=res[id(v)]["fill_list"]) for v in vs])
            print(f"\n  {mode} fills, variant {variant}, stop quoting {buffer} min before the detection bar\n{head}")
            for label, part in cells(vs):
                s = summarise([res[id(v)] for v in part])
                report[f"{mode} | {variant} | buffer {buffer} | {label}"] = s
                if not s["fills"] or ((buffer or mode != "strict") and not (label.endswith("all") or label.startswith("all tours"))):
                    continue
                nan = float("nan")
                print(f"  {label:24} {s['matches']:5} {s['quote_minutes']:7} {s['fills']:6} {s['fills_per_match']:6.2f} {s['round_trips']:5} "
                      f"{s['spread_per_round_trip'] if s['spread_per_round_trip'] is not None else nan:6.2f} {s['edge_at_quote']:5.2f} {s['markout_0m']:+6.2f} "
                      f"{s['markout_1m']:+6.2f} {s['markout_5m']:+6.2f} {s['markout_15m']:+6.2f} {fmt(s['markout_5m_ci'])} {s['exit_pnl_per_fill']:+9.2f} {fmt(s['exit_pnl_per_fill_ci'])} "
                      f"{s['mid_pnl_per_fill']:+8.2f} {fmt(s['mid_pnl_per_fill_ci'])} {s['settle_pnl_per_fill']:+11.2f} {fmt(s['settle_pnl_per_fill_ci'])} "
                      f"{s['exit_pnl_per_match']:+10.2f} {fmt(s['exit_pnl_per_match_ci'])} "
                      f"{s['early60']['exit_per_fill'] if s['early60']['exit_per_fill'] is not None else nan:+6.2f} "
                      f"{s['late40']['exit_per_fill'] if s['late40']['exit_per_fill'] is not None else nan:+6.2f}")
    # Does the answer depend on how the thick book is chosen? Ex-post volume leader (uses the future) and the underdog's market.
    print("\n  robustness to the choice of thick book (strict and touch fills, join and k=0.5, all tours, no buffer): exit / mid / mark5 per fill")
    for rule in ("oi", "underdog", "expost"):
        alt, _ = views(rows, rule)
        for mode in ("strict", "touch"):
            for variant in ("join", "k=0.5"):
                s = summarise([simulate_one(v, variant, 0, mode) for v in alt])
                report[f"thick rule {rule} | {mode} | {variant}"] = s
                print(f"  thick = {rule:8} {mode:6} {variant:6} matches {s['matches']:5} fills {s['fills']:6} exit {s['exit_pnl_per_fill']:+6.2f} {fmt(s['exit_pnl_per_fill_ci'])} "
                      f"mid {s['mid_pnl_per_fill']:+6.2f} {fmt(s['mid_pnl_per_fill_ci'])} mark5 {s['markout_5m']:+6.2f} {fmt(s['markout_5m_ci'])}")
    save(OUT/"simulate.json", report)


# ---------------------------------------------------------------- trades

def trade_sample(vs, per_tour=(("Challenger", 120), ("WTA", 35), ("ATP", 35), ("WTA Challenger", 30))):
    rng, out = np.random.default_rng(4), []
    for tour, n in per_tour:
        part = sorted((v for v in vs if v["tour"] == tour), key=lambda v: v["event"])
        out += [part[i] for i in sorted(rng.choice(len(part), min(n, len(part)), replace=False))]
    return out


def trade_params(v, ticker):
    return dict(ticker=ticker, min_ts=v["g0"]+60*(v["first"]-2), max_ts=v["g0"]+60*(v["start"]+2), limit=1000)


def trades_of(client, v, ticker, deadline, pace):
    """Public trades of one market over the pre-match period, or None when not cached and out of time."""
    out, cursor = [], ""
    while True:
        params = dict(trade_params(v, ticker), cursor=cursor) if cursor else trade_params(v, ticker)
        if not cache_file("/markets/trades", params).exists() and (client is None or time.time() > deadline):
            return None
        data = read(client, "/markets/trades", params, pace)[0]
        out += data.get("trades") or []
        cursor = data.get("cursor")
        if not cursor or not data.get("trades"):
            return out


def parse_trades(raw, flip):
    """(time, price in the thin book's terms, contracts, +1 if the taker bought the thin player / -1 if sold)."""
    out = []
    for t in raw:
        price = round(100*float(t["yes_price_dollars"])) if "yes_price_dollars" in t else round(float(t["yes_price"]))
        side = 1 if (t.get("taker_side") or t.get("taker_outcome_side")) == "yes" else -1
        out.append((timestamp(t["created_time"]), 100-price if flip else price, float(t.get("count_fp", t.get("count", 0))), -side if flip else side))
    return sorted(out)


def trades(rows, minutes, pace):
    vs, _ = views(rows)
    sample = trade_sample(vs)
    deadline, have = time.time()+60*minutes, []
    with httpx.Client(timeout=40) as client:
        for n, v in enumerate(sample):
            got = [trades_of(client, v, v[k], deadline, pace) for k in ("thin", "thick")]
            if None not in got:
                have.append((v, parse_trades(got[0], False), parse_trades(got[1], True)))
            if (n+1) % 50 == 0:
                print(f"  {n+1} of {len(sample)} matches read", flush=True)
    print(f"\n(4) FILLS AGAINST THE TRADE RECORD. {len(have)} of {len(sample)} sampled matches have both books' trades")
    report = dict(matches=len(have))
    # Sanity: does the trade record reproduce the bars' volume and the taker sides make sense (buyers pay more than sellers)?
    vol_bar = sum(v["vol"][v["first"]:v["start"]+1].sum() for v, _, _ in have)
    vol_tr = sum(c for v, thin, _ in have for ts_, _, c, _ in thin if v["g0"]+60*(v["first"]-1) < ts_ <= v["g0"]+60*v["start"])
    report["check"] = dict(thin_contracts_in_bars=float(vol_bar), thin_contracts_in_trades=float(vol_tr))
    print(f"  thin-book contracts in the pre-match bars {vol_bar:.0f}, in the trade record over the same minutes {vol_tr:.0f}")
    flow = Counter()
    for v, thin, thick in have:
        for name, tr in (("thin", thin), ("thick", thick)):
            for ts_, p, c, s in tr:
                i = int(math.ceil((ts_-v["g0"])/60))
                if v["first"] <= i < v["start"] and v["ok"][i-1]:
                    b, a = (v["b"][i-1], v["a"][i-1]) if name == "thin" else (v["B"][i-1], v["A"][i-1])
                    where = "at or above the ask" if p >= a else "at or below the bid" if p <= b else "inside"
                    flow[name, "taker buys" if s > 0 else "taker sells", where] += c
                    flow[name, "contracts"] += c
                    flow[name, "trades"] += 1
        flow["hours"] += (v["start"]-v["first"])/60
    for name in ("thin", "thick"):
        tot = flow[name, "contracts"]
        report[f"flow {name}"] = dict(contracts_per_match_hour=tot/flow["hours"], trades_per_match_hour=flow[name, "trades"]/flow["hours"],
                                      **{f"{a} {w}": flow[name, a, w]/tot for a in ("taker buys", "taker sells") for w in ("at or above the ask", "at or below the bid", "inside")})
        print(f"  {name:5} book: {tot/flow['hours']:.0f} contracts and {flow[name, 'trades']/flow['hours']:.1f} trades per pre-match hour; taker buys {100*sum(flow[name, 'taker buys', w] for w in ('at or above the ask', 'at or below the bid', 'inside'))/tot:.0f}% "
              f"(of all volume: lifting the previous minute's ask or higher {100*flow[name, 'taker buys', 'at or above the ask']/tot:.0f}%, taker sells at the previous bid or lower {100*flow[name, 'taker sells', 'at or below the bid']/tot:.0f}%)")
    print(f"  {'fills':30} {'n':>5} {'in trades':>9} {'median through':>14} {'mean':>7} {'10+':>5} {'100+':>5} {'taker against us':>16} | thick book printed at or through our price: "
          f"{'before':>6} {'same sec':>8} {'<=60s after':>11} {'never':>6} | {'thick mid moved against in fill bar':>34} | {'median contracts at our price':>29}")
    for mode, variant in (("strict", "join"), ("strict", "k=0.5"), ("strict", "k=1"), ("touch", "join"), ("touch", "k=0.5"), ("touch", "k=1")):
        rows_ = []
        for v, thin, thick in have:
            for i, side, price in simulate_one(v, variant, 0, mode)["fill_list"]:
                lo_ts, hi_ts = v["g0"]+60*(i-1), v["g0"]+60*i
                bar = [t for t in thin if lo_ts < t[0] <= hi_ts]
                hit = [t for t in bar if (side*(price-t[1]) > 0 if mode == "strict" else side*(price-t[1]) >= 0)]
                row = dict(found=bool(hit), moved=bool(side*(v["F"][i]-v["F"][i-1]) < 0), at_price=sum(t[2] for t in bar if t[1] == price))
                if hit:
                    t0 = hit[0][0]
                    through = [t[0] for t in thick if lo_ts-60 < t[0] <= t0+60 and side*(price-t[1]) >= 0]
                    row.update(contracts=sum(t[2] for t in hit), against=sum(t[2] for t in hit if t[3] == -side)/sum(t[2] for t in hit),
                               seconds_into_bar=t0-lo_ts,
                               thick="before" if any(x < t0-1 for x in through) else "same second" if any(abs(x-t0) <= 1 for x in through)
                               else "after" if through else "never",
                               lead=t0-min(through) if through and min(through) < t0-1 else None)
                rows_.append(row)
        found = [r for r in rows_ if r["found"]]
        if not found:
            continue
        c = np.array([r["contracts"] for r in found])
        cls = Counter(r["thick"] for r in found)
        leads = [r["lead"] for r in found if r["lead"] is not None]
        report[f"{mode} | {variant}"] = dict(
            fills=len(rows_), found_in_trades=len(found), median_contracts=float(np.median(c)), mean_contracts=float(c.mean()),
            share_10_plus=float((c >= 10).mean()), share_100_plus=float((c >= 100).mean()), quartiles=[float(x) for x in np.percentile(c, [25, 75])],
            taker_against_us=float(np.mean([r["against"] for r in found])), thick_printed={k: cls[k]/len(found) for k in ("before", "same second", "after", "never")},
            median_seconds_thick_led=float(np.median(leads)) if leads else None,
            thick_mid_moved_against_in_fill_bar=float(np.mean([r["moved"] for r in rows_])),
            median_contracts_at_our_price=float(np.median([r["at_price"] for r in rows_])),
            mean_min_10=float(np.minimum(c, 10).mean()), mean_min_100=float(np.minimum(c, 100).mean()))
        r = report[f"{mode} | {variant}"]
        print(f"  {mode+' '+variant:30} {len(rows_):5} {100*len(found)/len(rows_):8.0f}% {r['median_contracts']:14.0f} {r['mean_contracts']:7.0f} {100*r['share_10_plus']:4.0f}% {100*r['share_100_plus']:4.0f}% "
              f"{100*r['taker_against_us']:15.0f}% | {'':44} {100*r['thick_printed']['before']:5.0f}% {100*r['thick_printed']['same second']:7.0f}% {100*r['thick_printed']['after']:10.0f}% "
              f"{100*r['thick_printed']['never']:5.0f}% | {100*r['thick_mid_moved_against_in_fill_bar']:33.0f}% | {r['median_contracts_at_our_price']:29.0f}")
    # Joining a queue: the same "join" quotes, filled only once the contracts assumed to be ahead of us have traded at our price
    # (takers hitting that price), or at once on a print through it. Back of the queue again after each fill or price change.
    print("\n  joining the thin book's best bid and offer with N contracts ahead in the queue (trade record, sampled matches; cents per fill)")
    print(f"  {'ahead':>6} {'stop before start':>17} {'fills':>6} {'/match':>6} {'edge':>5} {'mark1':>6} {'mark5':>6} {'mark15':>6} {'mid/fill':>8} {'interval':>15} {'exit/fill':>9} {'settle/fill':>11} {'interval':>15}")
    for name, part, both in (("all sampled tours", have, False), ("Challenger only", [h for h in have if h[0]["tour"] == "Challenger"], False),
                             ("all sampled tours, YES offer in BOTH books", have, True),
                             ("Challenger only, YES offer in BOTH books", [h for h in have if h[0]["tour"] == "Challenger"], True)):
        print(f"  {name}: {len(part)} matches")
        for buffer in (0, 30):
            for ahead in (0, 50, 200, 1000, 5000):
                s = summarise([account(v, queue_fills(v, thin, ahead, buffer, thick if both else None)) for v, thin, thick in part])
                report[f"queue {ahead} | buffer {buffer} | {name}"] = s
                if s["fills"]:
                    rt = s["spread_per_round_trip"]
                    print(f"  {ahead:6} {buffer:17} {s['fills']:6} {s['fills_per_match']:6.2f} {s['edge_at_quote']:5.2f} {s['markout_1m']:+6.2f} {s['markout_5m']:+6.2f} {s['markout_15m']:+6.2f} "
                          f"{s['mid_pnl_per_fill']:+8.2f} {fmt(s['mid_pnl_per_fill_ci'])} {s['exit_pnl_per_fill']:+9.2f} {s['settle_pnl_per_fill']:+11.2f} {fmt(s['settle_pnl_per_fill_ci'])} "
                          f"round trips {s['round_trips']:5} at {rt if rt is not None else float('nan'):5.2f}c, mid per match {s['mid_pnl_per_match']:+6.2f} {fmt(s['mid_pnl_per_match_ci'])}, "
                          f"inventory p95 {s['max_inventory_p95']:.0f}")
    save(OUT/"trades.json", report)


def queue_fills(v, thin, ahead, buffer=0, thick=None):
    """Fills of the "join" quotes with `ahead` contracts in front of us. With `thick` trades given, the buying side is
    instead a YES offer joined on the thick book (a bid at the thick book's bid in thin terms): "offer both"."""
    def minutes(trades_):
        out = defaultdict(list)
        for ts_, price, count, side in trades_:
            out[int(math.ceil((ts_-v["g0"])/60))].append((price, count, side))
        return out
    fills = []
    for sign, key, by in ((1, "B" if thick is not None else "b", minutes(thick if thick is not None else thin)), (-1, "a", minutes(thin))):
        resting, left = None, ahead
        for t in range(v["first"]-1, v["start"]-buffer):
            if not v["ok"][t]:
                resting = None
                continue
            if v[key][t] != resting:
                resting, left = v[key][t], ahead
            for price, count, side in by.get(t+1, ()):
                if sign*(resting-price) > 0 or (price == resting and side == -sign and left-count < 0):
                    fills.append((t+1, sign, float(resting)))
                    left = ahead
                    break
                if price == resting and side == -sign:
                    left -= count
    return sorted(fills)


# ---------------------------------------------------------------- depth (live snapshot) and capacity

def depth(pace, most=24):
    """Displayed size at the best bid and offer of both books for open tennis matches that look pre-match. Not cached."""
    now, out = time.time(), []
    with httpx.Client(timeout=40) as client:
        def live(path, **params):
            wait = last_request[0]+pace-time.time()
            if wait > 0:
                time.sleep(wait)
            resp = client.get(config.KALSHI_PROD_BASE+path, params=params)
            last_request[0] = time.time()
            if resp.status_code != 200:
                time.sleep(20)
                return {}
            return resp.json()
        for series in ("KXATPCHALLENGERMATCH", "KXWTAMATCH", "KXATPMATCH", "KXWTACHALLENGERMATCH"):
            by = defaultdict(list)
            for m in live("/markets", series_ticker=series, status="open", limit=200).get("markets") or []:
                by[m["event_ticker"]].append(m)
            pairs = [g for g in by.values() if len(g) == 2 and all(m.get("expected_expiration_time") and timestamp(m["expected_expiration_time"]) > now+3*3600 for m in g)]
            for g in pairs[:most//4]:
                row = dict(event=g[0]["event_ticker"], tour=TOURS[series], books=[])
                for m in sorted(g, key=lambda m: -float(m.get("volume_fp") or 0)):
                    ob = live(f"/markets/{m['ticker']}/orderbook").get("orderbook_fp") or {}
                    yes, no = (sorted(((round(100*float(p)), float(q)) for p, q in ob.get(k) or []), reverse=True) for k in ("yes_dollars", "no_dollars"))
                    if yes and no:
                        row["books"].append(dict(ticker=m["ticker"], volume=float(m.get("volume_fp") or 0), bid=yes[0][0], bid_size=yes[0][1],
                                                 ask=100-no[0][0], ask_size=no[0][1], bid_size_3c=sum(q for p, q in yes[:3]), ask_size_3c=sum(q for p, q in no[:3])))
                if len(row["books"]) == 2:
                    out.append(row)
    print(f"\n  LIVE DEPTH SNAPSHOT: {len(out)} open matches with an expected end more than three hours away (busier book first)")
    for r in out:
        print(f"  {r['tour']:15} {r['event'][-14:]:14} "+" | ".join(
            f"vol {b['volume']:7.0f} {b['bid']:2.0f}/{b['ask']:2.0f} size {b['bid_size']:6.0f} x {b['ask_size']:6.0f}" for b in r["books"]))
    sizes = [b[k] for r in out for b in r["books"] for k in ("bid_size", "ask_size")]
    if sizes:
        print(f"  displayed size at the best price: median {np.median(sizes):.0f}, quartiles {np.percentile(sizes, 25):.0f}-{np.percentile(sizes, 75):.0f}; "
              f"quieter book only: median {np.median([r['books'][1][k] for r in out for k in ('bid_size', 'ask_size')]):.0f}")
    save(OUT/"depth-snapshot.json", dict(taken=now, rows=out))


def capacity(rows):
    vs, why = views(rows)
    sim, tr = json.loads((OUT/"simulate.json").read_text()), json.loads((OUT/"trades.json").read_text())
    print("\n(5) CAPACITY. matches a day from the listing since 2026-09-18; a quoted match is one with a clean warm-up and a detected start")
    print(f"  {'tour':15} {'variant':20} {'matches/day':>11} {'quoted/day':>10} {'fills/day':>9} {'cents/fill exit':>15} {'mid':>6} {'$ /day @10 exit':>15} {'@100 exit':>10} {'@10 mid':>8} {'@100 mid':>9} {'writes/day':>10}")
    report = {}
    for series, tour in TOURS.items():
        ev = [e for e in listing(None, series, archive=False).values() if e[0] >= CUTOFF and len(e[4]) == 2]
        if not ev or not any(v["tour"] == tour for v in vs):
            continue
        per_day = len(ev)/((max(e[0] for e in ev)-min(e[0] for e in ev))/86400)
        used = why[tour+" | used"]/sum(n for k, n in why.items() if k.startswith(tour+" | "))
        for mode, variant in (("strict", "join"), ("strict", "k=0.5"), ("strict", "k=1"), ("touch", "join"), ("touch", "k=0.5")):
            s, t = sim[f"{mode} | {variant} | buffer 0 | {tour} | all"], tr.get(f"{mode} | {variant}")
            if not s["fills"] or not t:
                continue
            fills = per_day*used*s["fills_per_match"]
            r = dict(matches_per_day=per_day, quoted_per_day=per_day*used, fills_per_day=fills, writes_per_day=per_day*used*s["writes_per_match"],
                     **{f"dollars_per_day_{n}_{k}": fills*s[f"{k}_pnl_per_fill"]/100*t[f"mean_min_{n}"] for n in (10, 100) for k in ("exit", "mid")})
            report[f"{tour} | {mode} | {variant}"] = r
            print(f"  {tour:15} {mode+' '+variant:20} {per_day:11.1f} {per_day*used:10.1f} {fills:9.1f} {s['exit_pnl_per_fill']:+15.2f} {s['mid_pnl_per_fill']:+6.2f} "
                  f"{r['dollars_per_day_10_exit']:+15.2f} {r['dollars_per_day_100_exit']:+10.2f} {r['dollars_per_day_10_mid']:+8.2f} {r['dollars_per_day_100_mid']:+9.2f} {r['writes_per_day']:10.0f}")
    save(OUT/"capacity.json", report)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("steps", nargs="+", choices=("download", "build", "describe", "simulate", "trades", "depth", "capacity"))
    parser.add_argument("--minutes", type=float, default=25, help="download budget")
    parser.add_argument("--pace", type=float, default=2.0, help="seconds between requests")
    args = parser.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    for step in args.steps:
        if step == "download":
            download(args.minutes, args.pace)
        elif step == "build":
            build()
        elif step == "describe":
            describe(load())
        elif step == "simulate":
            simulate(load())
        elif step == "trades":
            trades(load(), args.minutes, args.pace)
        elif step == "depth":
            depth(args.pace)
        elif step == "capacity":
            capacity(load())
