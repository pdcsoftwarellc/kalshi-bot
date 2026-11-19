"""AUDIT: resting bids under 80-90c tennis favourites, every round and tour. Where does the edge come from?

Follow-up to dip_surface_blind.py. research/favorite_dip.py shows a 70c bid under 80-90c favourites at +3.9% on 1,135
fills (ATP + ATP Challenger + WTA, no maker fee); the surface audit shows "other rounds" at zero and first-round
qualifying (FRQ) positive. This script reconciles them.

Data (no network for `report`): data/processed/dip-scalp/matches.pkl + bars.npz (one-minute bars from the detected start
of play), data/processed/tennis-contexts/rows.json (the last quote before play), data/processed/favorite-dip/*-orders.json
(favorite_dip's own orders, for the reproduction).

Rules of evidence
  favourite / price  the side above 50c on the last quote before the detected start (spread <= 4c); band 80 <= P < 90.
  bids               fixed 80, 75, 70, 65, 60c (only when at least 5c below P) and floor(P) - 10 / 15 / 20.
  fill               a one-minute bar after the detection bar whose low print on the favourite's side is STRICTLY below
                     the bid. One fill per match per cell, held to settlement.
  fee                maker fee 1.75 x p x (1-p) cents on ATP and WTA main-tour series (fee_type quadratic_with_maker_fees),
                     zero on Challenger, WTA Challenger and ITF (fee_type quadratic; public /series endpoint, 2026-11-19).
  return             (100 x payout - bid - fee) / (bid + fee), % of stake per fill; 95% bootstrap over matches (events),
                     4,000 draws; halves = the cell's fills split at the median start time.
  rounds             FRQ = "Qualification Round 1" / "Qualification" / "Qualifying"; later qualifying = other rounds whose
                     name starts with "Qualif"; main draw = everything else.
  MDE                minimum detectable edge at 80% power for "interval above zero" = 2.8 x standard error.
  search             fair-market replay of dip_surface.replay's model on the real matches (win with probability P; a
                     winner's low M has P(M < b) = b(100-P)/(P(100-b)); a loser fills every bid): how often does the best
                     of the searched cells (tour group x round x bid, 100+ fills) reach the observed best z?
  capacity           contracts in in-play minutes wholly below the bid (lower bound of what printed through it) and in
                     minutes with any print below it (upper bound), per fill, winners against losers.

    .venv/bin/python research/audit/tennis_8090_deep.py report
    .venv/bin/python research/audit/tennis_8090_deep.py tape --minutes 9     (public trade record for 150 random fills)
"""
import argparse
from collections import Counter, defaultdict
from datetime import datetime, timezone
import json
from pathlib import Path
import sys
import time

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
OUT = ROOT/"data/processed/audit/tennis_8090"
MAKER = {"ATP": 1.75, "WTA": 1.75, "Challenger": 0.0, "WTA Challenger": 0.0, "ITF men": 0.0, "ITF women": 0.0}
BIDS = ("80", "75", "70", "65", "60", "P-10", "P-15", "P-20")
TOURS = ("ATP", "WTA", "Challenger", "WTA Challenger", "ITF men", "ITF women")
RNG = np.random.default_rng(8090)
CUTOFF = datetime(2026, 9, 18, tzinfo=timezone.utc).timestamp()
lines = []


def say(t=""):
    print(t, flush=True)
    lines.append(t)


def round_of(name):
    if name in ("Qualification Round 1", "Qualification", "Qualifying"):
        return "FRQ"
    return "later-qual" if str(name).startswith("Qualif") else "main"


def load(lo=80, hi=90, price="before"):
    """Matches whose favourite is priced lo <= P < hi, with the favourite-side in-play bars (time, low, high, contracts)."""
    m = pd.read_pickle(ROOT/"data/processed/dip-scalp/matches.pkl")
    z = np.load(ROOT/"data/processed/dip-scalp/bars.npz")
    data, off = z["data"], z["offsets"]
    ctx = {r["event"]: r for r in json.loads((ROOT/"data/processed/tennis-contexts/rows.json").read_text()) if r.get("detected")}
    out = []
    for i, r in enumerate(m.itertuples()):
        if r.outcome not in (0.0, 1.0):
            continue
        if price == "before":
            c = ctx.get(r.event)
            if c is None or c.get("price_before_play") is None or c["ticker"] != r.ticker or abs(c["start_ts"]-r.start_ts) > 1:
                continue
            if c["ask_before_play"]-c["bid_before_play"] > 4:
                continue
            p_yes = c["price_before_play"] if c["yes"] else 100-c["price_before_play"]
        else:
            p_yes = r.p0
        yes = p_yes > 50
        p = p_yes if yes else 100-p_yes
        if not lo <= p < hi:
            continue
        d = data[:, off[i]:off[i+1]].astype(float)[:, 1:]
        keep = ~np.isnan(d[3]) & ~np.isnan(d[4])
        t, low, high, vol = d[0][keep]+r.start_ts, d[3][keep], d[4][keep], np.nan_to_num(d[5][keep])
        if not yes:
            low, high = 100-high, 100-low
        out.append(dict(event=r.event, ticker=r.ticker, tour=r.tour, rnd=round_of(r.round), start=float(r.start_ts), settle=int(r.settle),
                        p=float(p), yes=bool(yes), payout=float(r.outcome if yes else 1-r.outcome), fee=MAKER[r.tour],
                        t=t, lowbar=low, highbar=high, vol=vol, low=float(low.min()) if len(low) else 999.0,
                        month=datetime.fromtimestamp(r.start_ts, timezone.utc).strftime("%Y-%m")))
    return out


def bid_of(g, name):
    if name.startswith("P-"):
        return int(np.floor(g["p"]))-int(name[2:])
    b = int(name)
    return b if b <= g["p"]-5 else None


def fills_of(games, name, fee=True):
    """[(game, bid, return %)] for the games filled at this bid."""
    out = []
    for g in games:
        b = bid_of(g, name)
        if b is None or g["low"] >= b:
            continue
        f = g["fee"]*b/100*(1-b/100) if fee else 0.0
        out.append((g, b, 100*(100*g["payout"]-b-f)/(b+f)))
    return out


def boot(v):
    v = np.asarray(v, float)
    if len(v) < 10:
        return float("nan"), float("nan")
    return tuple(np.percentile(v[RNG.integers(0, len(v), (4000, len(v)))].mean(axis=1), [2.5, 97.5]))


def cell(games, name, fee=True):
    placed = sum(bid_of(g, name) is not None for g in games)
    hit = sorted(fills_of(games, name, fee), key=lambda x: x[0]["start"])
    if not hit:
        return dict(placed=placed, n=0)
    r = np.array([x[2] for x in hit])
    h = len(r)//2
    lo, hi = boot(r)
    se = r.std(ddof=1)/len(r)**.5 if len(r) > 1 else float("nan")
    return dict(placed=placed, n=len(r), ret=r.mean(), lo=lo, hi=hi, h1=r[:h].mean() if h else float("nan"), h2=r[h:].mean(),
                win=100*np.mean([x[0]["payout"] for x in hit]), need=np.mean([x[1] for x in hit]), se=se, mde=2.8*se,
                z=r.mean()/se if se and se > 0 else float("nan"))


def fmt(c):
    if not c["n"]:
        return f"placed {c['placed']:5} fills    0"
    return (f"placed {c['placed']:5} fills {c['n']:4} win {c['win']:5.1f}% (need {c['need']:4.1f}) ret {c['ret']:+6.1f}% [{c['lo']:+6.1f},{c['hi']:+6.1f}] "
            f"halves {c['h1']:+6.1f}% / {c['h2']:+6.1f}%  MDE {c['mde']:4.1f}%  z {c['z']:+.2f}")


def through(g, b):
    """(lower, upper) bound of contracts printed strictly below bid b in play."""
    return float(g["vol"][g["highbar"] < b].sum()), float(g["vol"][g["lowbar"] < b].sum())


# ---------------------------------------------------------------- report

def reproduce():
    say("=== 1. REPRODUCTION of favorite_dip.py (its own orders: ATP + ATP Challenger + WTA, warm-up price, no fee), 80-90c favourites, 70c bid")
    rows = {}
    for tag, name in (("archive Jan-17 Sep", "tennis-archived-orders.json"), ("autumn Sep-Nov", "tennis-orders.json")):
        r = json.loads((ROOT/"data/processed/favorite-dip"/name).read_text())
        rows[tag] = [x for x in r if 80 <= x["start"] < 90 and x["limit"] == 70 and x["strict"]]
    both = {x["event"]: x for part in rows.values() for x in part}
    rows["both (events once)"] = list(both.values())
    rows["both (as summed: 927 + 208)"] = rows["archive Jan-17 Sep"]+rows["autumn Sep-Nov"]
    for tag, part in rows.items():
        part = sorted(part, key=lambda x: x["settle"])
        r = np.array([100*(100*x["payout"]-70)/70 for x in part])
        lo, hi = boot(r)
        say(f"  {tag:28} fills {len(r):5} win {100*np.mean([x['payout'] for x in part]):5.1f}% ret {r.mean():+5.2f}% [{lo:+5.1f},{hi:+5.1f}] halves {r[:len(r)//2].mean():+5.1f}% / {r[len(r)//2:].mean():+5.1f}%")
    for series, label in (("KXWTAMATCH", "WTA"), ("KXATPMATCH", "ATP"), ("KXATPCHALLENGERMATCH", "Challenger")):
        part = [x for x in rows["both (events once)"] if x["event"].startswith(series+"-")]
        r = np.array([100*(100*x["payout"]-70)/70 for x in part])
        lo, hi = boot(r)
        say(f"    {label:12} fills {len(r):5} ret {r.mean():+5.2f}% [{lo:+5.1f},{hi:+5.1f}]")
    return {x["event"] for x in rows["both (events once)"]}


def report():
    OUT.mkdir(parents=True, exist_ok=True)
    theirs = reproduce()
    games = load()
    say(f"\n=== 2. INDEPENDENT REBUILD (last quote before play, strict fills from the detected start, maker fee on ATP and WTA): {len(games)} matches at 80-90c")
    say("  matches by tour and round: "+", ".join(f"{k[0]}/{k[1]} {v}" for k, v in sorted(Counter((g["tour"], g["rnd"]) for g in games).items())))
    same = [g for g in games if g["tour"] in ("ATP", "WTA", "Challenger")]
    say("  favorite_dip's universe (ATP + Challenger + WTA), 70c bid:")
    say(f"    with maker fee     {fmt(cell(same, '70'))}")
    say(f"    without fee        {fmt(cell(same, '70', fee=False))}")
    mine = {x[0]["event"] for x in fills_of(same, "70")}
    say(f"    fills in both versions {len(mine & theirs)}, only favorite_dip {len(theirs-mine)}, only the rebuild {len(mine-theirs)}")
    p0 = [g for g in load(price="p0") if g["tour"] in ("ATP", "WTA", "Challenger")]
    say(f"    warm-up price, fee {fmt(cell(p0, '70'))}")

    say("\n=== 3. BY ROUND x BID, all six tours pooled (fee charged)")
    for rnd in ("FRQ", "later-qual", "main", "not FRQ", "all"):
        part = [g for g in games if rnd == "all" or (rnd == "not FRQ" and g["rnd"] != "FRQ") or g["rnd"] == rnd]
        say(f"  --- {rnd}: {len(part)} matches")
        for name in BIDS:
            say(f"    {name:5} {fmt(cell(part, name))}")

    say("\n=== 4. BY TOUR x ROUND (FRQ / not FRQ), bids 80, 70, 60, P-10, P-20")
    for tour in TOURS+("ATP+WTA", "Challenger+WTA Challenger", "ITF"):
        members = {"ATP+WTA": ("ATP", "WTA"), "Challenger+WTA Challenger": ("Challenger", "WTA Challenger"), "ITF": ("ITF men", "ITF women")}.get(tour, (tour,))
        for rnd in ("FRQ", "not FRQ"):
            part = [g for g in games if g["tour"] in members and (g["rnd"] == "FRQ") == (rnd == "FRQ")]
            if len(part) < 30:
                continue
            say(f"  --- {tour} / {rnd}: {len(part)} matches")
            for name in ("80", "70", "60", "P-10", "P-20"):
                say(f"    {name:5} {fmt(cell(part, name))}")

    say("\n=== 5. THE 70c BID: is it qualifying plus noise? Same tours as favorite_dip (ATP + Challenger + WTA), then all tours")
    for label, pool in (("ATP + Challenger + WTA", same), ("all six tours", games)):
        for rnd in ("FRQ", "later-qual", "main"):
            part = [g for g in pool if g["rnd"] == rnd]
            say(f"  {label:24} {rnd:10} {fmt(cell(part, '70'))}")
    say("  main draw only, deeper bid = better? (all tours)")
    main = [g for g in games if g["rnd"] == "main"]
    for name in ("80", "75", "70", "65", "60"):
        say(f"    {name:5} {fmt(cell(main, name))}")
    say("  main draw, WTA only (the cell quoted at +8.5%)")
    for name in BIDS:
        say(f"    {name:5} {fmt(cell([g for g in main if g['tour'] == 'WTA'], name))}")
    say("  WTA, all rounds")
    for name in ("80", "70", "60", "P-10", "P-20"):
        say(f"    {name:5} {fmt(cell([g for g in games if g['tour'] == 'WTA'], name))}")

    say("\n=== 6. BY MONTH, 70c bid: FRQ | not FRQ (all tours) | not FRQ in ATP + Challenger + WTA: fills, return")
    months = sorted({g["month"] for g in games})
    for mth in months:
        text = f"  {mth}"
        for part in ([g for g in games if g["rnd"] == "FRQ"], [g for g in games if g["rnd"] != "FRQ"], [g for g in same if g["rnd"] != "FRQ"]):
            hit = fills_of([g for g in part if g["month"] == mth], "70")
            text += f" | {len(hit):4} {np.mean([x[2] for x in hit]):+6.1f}%" if hit else " |    0    n/a "
        say(text)

    # ---- search
    say("\n=== 7. HOW OFTEN DOES A SEARCH THIS SIZE FIND A CELL THIS GOOD? Fair-market replay, 2,000 seasons")
    for label, pool in (("not FRQ", [g for g in games if g["rnd"] != "FRQ"]), ("all rounds incl. FRQ as its own split", games)):
        groups = {"all tours": TOURS, "ATP+WTA": ("ATP", "WTA"), "Challenger+WTA Challenger": ("Challenger", "WTA Challenger"), "ITF": ("ITF men", "ITF women"),
                  **{t: (t,) for t in TOURS}}
        rounds = ("later-qual", "main", "any") if label == "not FRQ" else ("FRQ", "later-qual", "main", "any")
        cells, observed = [], []
        price = np.array([g["p"] for g in pool])
        for gname, members in groups.items():
            for rnd in rounds:
                idx = np.array([i for i, g in enumerate(pool) if g["tour"] in members and (rnd == "any" or g["rnd"] == rnd)], int)
                if len(idx) < 100:
                    continue
                for name in BIDS:
                    bids = np.array([bid_of(pool[i], name) or -1 for i in idx], float)
                    ok = bids > 0
                    if ok.sum() < 100:
                        continue
                    fee = np.array([pool[i]["fee"] for i in idx])[ok]*bids[ok]/100*(1-bids[ok]/100)
                    cells.append((idx[ok], bids[ok], fee))
                    c = cell([pool[i] for i in idx], name)
                    if c["n"] >= 100:
                        observed.append((c["z"], gname, rnd, name, c))
        observed.sort(key=lambda x: -x[0])
        best = np.zeros(2000)
        count = np.zeros(2000)
        for s in range(2000):
            win = RNG.random(len(pool)) < price/100
            u = RNG.random(len(pool))
            low = np.where(win, 100*u*price/((100-price)+u*price), 0.0)
            top = -9.0
            for idx, bids, fee in cells:
                hit = low[idx] < bids
                if hit.sum() < 100:
                    continue
                r = (100*win[idx][hit]-bids[hit]-fee[hit])/(bids[hit]+fee[hit])
                sd = r.std(ddof=1)
                if sd > 0:
                    zz = r.mean()/sd*len(r)**.5
                    top = max(top, zz)
                    count[s] += zz > 1.96
            best[s] = top
        say(f"  {label}: {len(cells)} cells searched (tour group x round x bid, 100+ fills); observed cells with z > 1.96: {sum(o[0] > 1.96 for o in observed)} "
            f"(fair market: mean {count.mean():.1f}, 95th pct {np.percentile(count, 95):.0f})")
        for zz, gname, rnd, name, c in observed[:6]:
            say(f"    z {zz:+.2f}  {gname} / {rnd} / {name}: {fmt(c)}  -> a fair market's best cell reaches this z in {np.mean(best >= zz):.1%} of seasons")

    # ---- capacity
    say("\n=== 8. CAPACITY (bars): fills per 30 days (May-October 2026), contracts printed through the bid per fill [lower, upper bound], winners v losers")
    for label, part in (("FRQ, all tours", [g for g in games if g["rnd"] == "FRQ"]), ("not FRQ, all tours", [g for g in games if g["rnd"] != "FRQ"]),
                        ("not FRQ, WTA", [g for g in games if g["rnd"] != "FRQ" and g["tour"] == "WTA"]),
                        ("not FRQ, ATP", [g for g in games if g["rnd"] != "FRQ" and g["tour"] == "ATP"]),
                        ("not FRQ, Challenger", [g for g in games if g["rnd"] != "FRQ" and g["tour"] == "Challenger"]),
                        ("not FRQ, ITF", [g for g in games if g["rnd"] != "FRQ" and g["tour"].startswith("ITF")])):
        for name in ("70", "P-10", "P-20"):
            hit = fills_of(part, name)
            if len(hit) < 20:
                continue
            per30 = sum(x[0]["month"] in ("2026-05", "2026-06", "2026-07", "2026-08", "2026-09", "2026-10") for x in hit)*30/184
            cap = np.array([through(x[0], x[1]) for x in hit])
            won = np.array([x[0]["payout"] == 1 for x in hit])
            share = cap[~won, 1].sum()/max(cap[:, 1].sum(), 1)
            say(f"  {label:22} {name:5} fills/30d {per30:6.1f} | median contracts through [{np.median(cap[:, 0]):9,.0f}, {np.median(cap[:, 1]):9,.0f}] | winners median upper {np.median(cap[won, 1]):9,.0f} "
                f"losers {np.median(cap[~won, 1]):9,.0f} | losers are {100*np.mean(~won):4.1f}% of fills and {100*share:4.1f}% of the contracts through")
            # size-weighted return: a fixed share of everything printed through the bid, against one contract per fill
            w = cap[:, 1]
            r = np.array([x[2] for x in hit])
            capped = np.minimum(w, 1000)
            say(f"  {'':22}       return per fill {r.mean():+5.1f}% | weighted by contracts through {np.sum(r*w)/max(w.sum(), 1):+5.1f}% | weighted, capped at 1,000 contracts a fill {np.sum(r*capped)/max(capped.sum(), 1):+5.1f}%")
    (OUT/"report.txt").write_text("\n".join(lines))


# ---------------------------------------------------------------- trade tape

def tape(minutes):
    """The public trade record around the first bar-fill of 150 random fills (70c bid, all tours, 80-90c favourites)."""
    import httpx
    import config
    OUT.mkdir(parents=True, exist_ok=True)
    store = OUT/"tape"
    store.mkdir(exist_ok=True)
    games = load()
    hit = sorted(fills_of(games, "70"), key=lambda x: x[0]["event"])
    pick = np.random.default_rng(150).choice(len(hit), 150, replace=False)
    deadline, last = time.monotonic()+60*minutes, [0.0]
    done = 0
    with httpx.Client(timeout=30, headers={"User-Agent": "research-client/1.0"}) as client:
        for k in sorted(pick):
            g, b, _ = hit[k]
            file = store/(g["ticker"]+".json")
            if file.exists():
                done += 1
                continue
            if time.monotonic() > deadline:
                break
            i = int(np.argmax(g["lowbar"] < b))
            end = int(g["t"][i])
            lo, hi = end-60, end
            paths = ["/historical/trades", "/markets/trades"] if g["settle"] < CUTOFF else ["/markets/trades", "/historical/trades"]
            raw, status = [], None
            for path in paths:
                cursor, pages = "", 0
                while pages < 5:
                    params = dict(ticker=g["ticker"], min_ts=lo, max_ts=hi+1, limit=1000)
                    if cursor:
                        params["cursor"] = cursor
                    data = None
                    for pause in (60, 120, 300):
                        wait = last[0]+3.0-time.monotonic()
                        if wait > 0:
                            time.sleep(wait)
                        try:
                            resp = client.get(config.KALSHI_PROD_BASE+path, params=params)
                        except httpx.RequestError:
                            resp = None
                        last[0] = time.monotonic()
                        status = resp.status_code if resp is not None else None
                        if status == 200:
                            data = resp.json()
                            break
                        if status is not None and status != 429 and status < 500:
                            data = {}
                            break
                        print(f"   throttled ({status}): pausing {pause}s", flush=True)
                        time.sleep(pause)
                        last[0] = time.monotonic()
                    if data is None:
                        raise RuntimeError("read failed after long back-off; stopping")
                    pages += 1
                    raw += data.get("trades") or []
                    cursor = data.get("cursor")
                    if not cursor or not data.get("trades"):
                        break
                if raw:
                    break
            trades = []
            for t in raw:
                price = round(100*float(t["yes_price_dollars"]))
                when = datetime.fromisoformat(t["created_time"].replace("Z", "+00:00")).timestamp()
                trades.append([when, price if g["yes"] else 100-price, float(t["count_fp"])])
            file.write_text(json.dumps(dict(event=g["event"], ticker=g["ticker"], bid=b, bar_end=end, bar_low=float(g["lowbar"][i]), start=g["start"],
                                            payout=g["payout"], status=status, trades=trades)))
            done += 1
            print(f"  {done}/150 {g['ticker']} trades {len(trades)}", flush=True)
    print(f"{done} of 150 on disk")


def tape_report():
    rows = [json.loads(p.read_text()) for p in sorted((OUT/"tape").glob("*.json"))]
    out = [f"TRADE-TAPE CHECK of {len(rows)} random bar-fills (70c bid under 80-90c tennis favourites, all tours and rounds)"]
    ok = touch = none = empty = 0
    sizes, lows = [], []
    for r in rows:
        inside = [t for t in r["trades"] if r["bar_end"]-60 < t[0] <= r["bar_end"] and t[0] > r["start"]]
        if not r["trades"]:
            empty += 1
            continue
        below = [t for t in inside if t[1] < r["bid"]]
        if below:
            ok += 1
            sizes.append(sum(t[2] for t in below))
            lows.append(min(t[1] for t in below) == r["bar_low"])
        elif any(t[1] == r["bid"] for t in inside):
            touch += 1
        else:
            none += 1
    out.append(f"  a trade strictly below the bid in the fill minute: {ok}; only a print AT the bid: {touch}; no print at or below: {none}; no trades returned: {empty}")
    if sizes:
        out.append(f"  contracts printed below the bid in that first minute: median {np.median(sizes):,.0f}, 10th pct {np.percentile(sizes, 10):,.0f}, minimum {min(sizes):,.0f}; "
                   f"under 10 contracts in {np.mean(np.array(sizes) < 10):.0%}, under 100 in {np.mean(np.array(sizes) < 100):.0%}; tape low equals the bar low in {np.mean(lows):.0%}")
    won = [r["payout"] for r in rows]
    out.append(f"  sample win rate {100*np.mean(won):.1f}% on {len(rows)}")
    text = "\n".join(out)
    print(text)
    (OUT/"tape-check.txt").write_text(text)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=("report", "tape", "tape-report"))
    parser.add_argument("--minutes", type=float, default=9)
    args = parser.parse_args()
    if args.command == "report":
        report()
    elif args.command == "tape":
        tape(args.minutes)
    else:
        tape_report()


if __name__ == "__main__":
    main()
