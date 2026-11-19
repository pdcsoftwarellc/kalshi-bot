"""Audit of research/consensus_history.py (bookmaker consensus against Kalshi's prematch quote).

Offline: reads data/processed/consensus-history/matches.json and the cached one-minute bars
under data/raw/cost-research-v1. Never touches the network.

  1. reproduce the reported cents-per-contract table with a copy of the original trade()/interval()
  2. restate as % per bet with a match-clustered bootstrap and chronological halves
  3. planted-edge test: redraw outcomes so the selected bets carry a true +X% per bet
  4. resting-bid version: join the bid on the side the bookmaker prefers at T-60, fill only when a
     later minute prints strictly through the bid before the scheduled start, hold to settlement
"""
import hashlib, json, random, sys
from pathlib import Path
from statistics import mean

ROOT = Path(__file__).resolve().parents[2]
ROWS = json.loads((ROOT/"data/processed/consensus-history/matches.json").read_text())
SUMMARY = json.loads((ROOT/"data/processed/consensus-history/summary.json").read_text())
CACHE = ROOT/"data/raw/cost-research-v1"
OUT = ROOT/"data/processed/audit/inventory/consensus-audit.json"
SERIES = {"atp": "KXATPMATCH", "challenger": "KXATPCHALLENGERMATCH", "wta": "KXWTAMATCH"}
MAKER_FEE = {"atp": .0175, "challenger": 0.0, "wta": .0175}   # quadratic_with_maker_fees on ATP/WTA only
HORIZONS, THRESHOLDS, SOURCES = (60, 10), (0, 1, 2, 3, 5), ("pinnacle", "median")


def interval(values, draws=2000, seed=7):                    # verbatim from the original
    if len(values) < 5:
        return None
    rng = random.Random(seed)
    means = sorted(mean(rng.choices(values, k=len(values))) for _ in range(draws))
    return [means[int(.025*draws)], means[int(.975*draws)]]


def trade(row, source, horizon):                             # original, plus the price paid
    q, p = row["quotes"][str(horizon)], row[source]
    if not q or p is None:
        return None
    bid, ask = q
    sides = [(p, ask, row["outcome"]), (1-p, 1-bid, 1-row["outcome"])]
    prob, price, payout = max(sides, key=lambda s: s[0]-s[1])
    fee = .07*price*(1-price)
    return dict(edge=prob-price-fee, pnl=payout-price-fee, cost=price+fee, price=price, prob=prob,
                start=row["start"], tour=row["tour"], payout=payout)


def stats(bets, key="roi"):
    """mean, match-bootstrap interval and chronological halves of a per-bet list."""
    if not bets:
        return None
    bets = sorted(bets, key=lambda b: b["start"])
    v = [100*b[key] for b in bets]
    h = len(v)//2
    out = dict(n=len(v), mean=mean(v), ci95=interval(v, 4000), first_half=mean(v[:h]) if h else None,
               second_half=mean(v[h:]), first_ci=interval(v[:h], 4000), second_ci=interval(v[h:], 4000))
    return out


def fmt(s):
    if not s:
        return "none"
    ci = "[%+.1f, %+.1f]" % tuple(s["ci95"]) if s["ci95"] else "[n<5]"
    return "n=%4d  %+6.2f %s  halves %+6.2f / %+6.2f" % (s["n"], s["mean"], ci, s["first_half"] or 0, s["second_half"])


def taker_bets(source, horizon, cents):
    out = []
    for r in ROWS:
        t = trade(r, source, horizon)
        if t and t["edge"] >= cents/100:
            t["roi"] = t["pnl"]/t["cost"]
            out.append(t)
    return out


def reproduce():
    print("1. REPRODUCTION against summary.json (cents per contract)")
    ok = True
    for horizon in HORIZONS:
        for source in SOURCES:
            for cents in THRESHOLDS:
                bets = taker_bets(source, horizon, cents)
                pnl = [100*b["pnl"] for b in bets]
                ref = SUMMARY["all"][f"{horizon} min before start"][source]["thresholds"].get(f">={cents}c")
                if not ref:
                    continue
                same = len(pnl) == ref["trades"] and abs(mean(pnl)-ref["net_cents_per_contract"]) < 1e-9 \
                    and (interval(pnl) == ref["ci95"])
                ok &= same
    print("   all 20 cells identical (n, mean, interval):", ok)
    return ok


def percent():
    print("\n2. TAKER RULE AS % PER BET (profit / (price+fee)); match bootstrap; chronological halves")
    res = {}
    for horizon in HORIZONS:
        for source in SOURCES:
            for cents in THRESHOLDS:
                bets = taker_bets(source, horizon, cents)
                s = stats(bets)
                s["avg_price_c"] = 100*mean(b["price"] for b in bets)
                s["claimed_edge_pct"] = 100*mean(b["edge"]/b["cost"] for b in bets)
                s["cents"] = mean(100*b["pnl"] for b in bets)
                res[f"{source}|{horizon}|{cents}"] = s
                print(f"   {source:8s} T-{horizon:2d} edge>={cents}c  {fmt(s)}   avg price {s['avg_price_c']:.0f}c"
                      f"  bookmaker-claimed edge {s['claimed_edge_pct']:+.1f}%  ({s['cents']:+.2f}c)")
    return res


def planted(source="pinnacle", horizon=10, cents=0, sims=2000):
    """Redraw each selected bet's payout as Bernoulli(cost*(1+x)) so the true ROI is exactly x."""
    print(f"\n3. PLANTED EDGE on the headline cell ({source}, T-{horizon}, edge>={cents}c)")
    bets = sorted(taker_bets(source, horizon, cents), key=lambda b: b["start"])
    rng = random.Random(11)
    res = {}
    for x in (0.0, .015, .03, .05, .10, .15, .20, .25, .30):
        hit_ci = hit_both = hit_all = 0
        for s in range(sims):
            pnl = []
            for b in bets:
                q = min(1.0, b["cost"]*(1+x))
                pnl.append(100*((rng.random() < q)-b["cost"]))            # cents per contract, as the study scores
            m = mean(pnl)
            se = (sum((v-m)**2 for v in pnl)/(len(pnl)-1))**.5/len(pnl)**.5   # normal approx to its bootstrap
            lo = m-1.96*se
            h = len(pnl)//2
            both = mean(pnl[:h]) > 0 and mean(pnl[h:]) > 0
            hit_ci += lo > 0
            hit_both += both
            hit_all += lo > 0 and both
        res[x] = dict(interval_excludes_zero=hit_ci/sims, both_halves_positive=hit_both/sims, both_rules=hit_all/sims)
        print("   true edge %+5.1f%% per bet: interval excludes zero %5.1f%%   both halves positive %5.1f%%   both %5.1f%%"
              % (100*x, 100*hit_ci/sims, 100*hit_both/sims, 100*hit_all/sims))
    roi = [b["pnl"]/b["cost"] for b in bets]
    m = mean(roi)
    sd = (sum((v-m)**2 for v in roi)/(len(roi)-1))**.5
    se = sd/len(roi)**.5
    print("   per-bet sd %.0f%%, n %d, standard error %.1f%% -> 80%% power needs a true edge of about %+.0f%% per bet"
          % (100*sd, len(roi), 100*se, 100*2.8*se))
    res["se_pct"], res["mde80_pct"] = 100*se, 280*se
    return res


_BARS = {}


def bars(row):
    if row["ticker"] not in _BARS:
        _BARS[row["ticker"]] = _bars(row)
    return _BARS[row["ticker"]]


def _bars(row):
    path = f"/series/{SERIES[row['tour']]}/markets/{row['ticker']}/candlesticks"
    params = dict(start_ts=row["start"]-60*60-300, end_ts=row["start"], period_interval=1)
    key = hashlib.sha256(json.dumps([path, params], sort_keys=True).encode()).hexdigest()
    f = CACHE/(key+".json")
    return json.loads(f.read_text())["candlesticks"] if f.exists() else None


def num(bar, group, field):
    try:
        return float(bar[group][field+"_dollars"])
    except (KeyError, TypeError, ValueError):
        return None


def resting(source, cents, improve=0, planted_edge=None, rng=None):
    """Join (or improve by `improve` cents inside the spread) the bid on the bookmaker's side at T-60.

    YES bid at b fills when a later minute's trade low is strictly below b; a NO bid at 1-a (joining the
    YES ask a) fills when a later minute's trade high is strictly above a. Cancelled at the scheduled start.
    """
    orders = []
    for r in ROWS:
        q, p = r["quotes"]["60"], r[source]
        if not q or p is None:
            continue
        bid, ask = q
        step = improve/100 if ask-bid > improve/100+1e-9 else 0.0     # never cross or lock the book
        sides = [("yes", p, bid+step, r["outcome"]), ("no", 1-p, 1-ask+step, 1-r["outcome"])]
        side, prob, price, payout = max(sides, key=lambda s: s[1]-s[2])
        fee = MAKER_FEE[r["tour"]]*price*(1-price)
        if prob-price-fee < cents/100 or not 0 < price < 1:
            continue
        cs = bars(r)
        if cs is None:
            continue
        at = r["start"]-3600
        valid = [b for b in cs if at-300 < b["end_period_ts"] <= at]
        t0 = max(b["end_period_ts"] for b in valid)
        filled = False
        for b in cs:
            if b["end_period_ts"] <= t0 or b["end_period_ts"] > r["start"]:
                continue
            lo, hi = num(b, "price", "low"), num(b, "price", "high")
            if lo is None:
                continue
            if (side == "yes" and lo < price-1e-9) or (side == "no" and hi > (1-price)+1e-9):
                filled = True
                break
        if planted_edge is not None:
            payout = float(rng.random() < min(1.0, (price+fee)*(1+planted_edge)))
        orders.append(dict(start=r["start"], tour=r["tour"], side=side, price=price, cost=price+fee, filled=filled,
                           pnl=payout-price-fee, roi=(payout-price-fee)/(price+fee), edge=prob-price-fee,
                           ticker=r["ticker"], payout=payout))
    return orders


def resting_report():
    print("\n4. RESTING-BID VERSION (T-60 quote, live until the scheduled start, strict through-print fills,"
          "\n   maker fee 1.75%*p*(1-p) on ATP/WTA and zero on Challenger; % per filled bet; match bootstrap)")
    res = {}
    for improve in (0, 1):
        for source in SOURCES:
            for cents in (0, 1, 2, 3):
                orders = resting(source, cents, improve)
                fills = [o for o in orders if o["filled"]]
                unf = [o for o in orders if not o["filled"]]
                s = stats(fills)
                key = f"{source}|improve{improve}|{cents}"
                res[key] = dict(orders=len(orders), fills=len(fills), stats=s,
                                unfilled_roi=100*mean(o["roi"] for o in unf) if unf else None,
                                avg_price_c=100*mean(o["price"] for o in fills) if fills else None,
                                by_tour={t: stats([o for o in fills if o["tour"] == t]) for t in SERIES})
                print(f"   {source:8s} {'join bid ' if not improve else 'bid+1c   '} edge>={cents}c  orders {len(orders):4d} "
                      f"fills {len(fills):4d}  {fmt(s)}  avg price {res[key]['avg_price_c'] or 0:.0f}c"
                      f"  unfilled would have made {res[key]['unfilled_roi'] or 0:+.1f}%")
    print("   by tour, pinnacle, join bid, edge>=0c:")
    for t, s in res["pinnacle|improve0|0"]["by_tour"].items():
        print(f"     {t:10s} {fmt(s)}")
    return res


def resting_planted(source="pinnacle", cents=0, sims=1000):
    print(f"\n5. PLANTED EDGE on the resting version ({source}, join bid, edge>={cents}c)")
    rng = random.Random(5)
    res = {}
    for x in (0.0, .015, .03, .10, .20, .30):
        hit = both = 0
        base = sorted((o for o in resting(source, cents, 0) if o["filled"]), key=lambda o: o["start"])
        for _ in range(sims):
            v = [((rng.random() < min(1.0, o["cost"]*(1+x)))-o["cost"])/o["cost"] for o in base]
            m = mean(v)
            se = (sum((a-m)**2 for a in v)/(len(v)-1))**.5/len(v)**.5
            h = len(v)//2
            ok = m-1.96*se > 0
            hit += ok
            both += ok and mean(v[:h]) > 0 and mean(v[h:]) > 0
        res[x] = dict(interval_excludes_zero=hit/sims, plus_both_halves=both/sims)
        print("   true edge %+5.1f%%: interval excludes zero %5.1f%%   and both halves positive %5.1f%%"
              % (100*x, 100*hit/sims, 100*both/sims))
    return res


def calibration():
    """Is the bookmaker right about the gap? outcome - mid = beta * (book - mid)."""
    print("\n6. WHO IS RIGHT ABOUT THE GAP? slope of (outcome - Kalshi mid) on (bookmaker - Kalshi mid); 1 = bookmaker, 0 = Kalshi")
    res = {}
    for source in SOURCES:
        for h in HORIZONS:
            pts = [(r[source]-sum(r["quotes"][str(h)])/2, r["outcome"]-sum(r["quotes"][str(h)])/2)
                   for r in ROWS if r["quotes"][str(h)] and r[source] is not None]
            def slope(p):
                sxx = sum(x*x for x, _ in p)
                return sum(x*y for x, y in p)/sxx
            rng = random.Random(3)
            b = slope(pts)
            boots = sorted(slope(rng.choices(pts, k=len(pts))) for _ in range(1000))
            res[f"{source}|{h}"] = dict(n=len(pts), slope=b, ci95=[boots[25], boots[975]])
            print(f"   {source:8s} T-{h:2d}: n {len(pts)}  slope {b:+.2f}  [{boots[25]:+.2f}, {boots[975]:+.2f}]")
    return res


if __name__ == "__main__":
    out = dict(reproduced=reproduce(), taker_percent=percent(), planted=planted(), resting=resting_report(),
               calibration=calibration())
    if "--full" in sys.argv:
        out["resting_planted"] = resting_planted()
    OUT.write_text(json.dumps(out, indent=1, default=str))
    print("\nsaved", OUT)
