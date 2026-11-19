"""Audit of research/near_end.py on tennis (buy a side once its ask is 97-99c, hold to settlement).

Offline: saved rows data/processed/near-end/main-trades.json plus the cached one-minute bars
(data/raw/cost-research-v1). entries() is imported unmodified from the original.

  1. rebuild the tennis entries from the cached bars and compare with the saved summary
  2. the "any spread" entry rule: how much of the reported loss is quotes nobody could trade
  3. planted edge on the taker rule
  4. resting-bid version: bid 1c under the entry ask, strict through-print fills, halves, event bootstrap
"""
import hashlib, json, random, sys
from collections import defaultdict
from pathlib import Path
from statistics import mean

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from research.near_end import entries, cents, table, WINDOW

CACHE = ROOT/"data/raw/cost-research-v1"
OUT = ROOT/"data/processed/audit/inventory/near-end-audit.json"
TENNIS = ("KXATPMATCH", "KXATPCHALLENGERMATCH", "KXWTAMATCH")
MAKER = {"KXATPMATCH": .0175, "KXWTAMATCH": .0175, "KXATPCHALLENGERMATCH": 0.0}
SAVED = json.loads((ROOT/"data/processed/near-end/main.json").read_text())["groups"]["tennis"]


def load_bars(series, ticker, settle):
    path = f"/series/{series}/markets/{ticker}/candlesticks"
    key = hashlib.sha256(json.dumps([path, dict(start_ts=settle-WINDOW, end_ts=settle, period_interval=1)],
                                    sort_keys=True).encode()).hexdigest()
    f = CACHE/(key+".json")
    return sorted(json.loads(f.read_text())["candlesticks"], key=lambda b: b["end_period_ts"]) if f.exists() else None


def boot(rows, key, draws=3000, seed=11):
    by = defaultdict(list)
    for r in rows:
        by[r["event"]].append(r[key])
    g = list(by.values())
    if len(g) < 10:
        return None
    rng = random.Random(seed)
    ms = sorted(mean(v for x in rng.choices(g, k=len(g)) for v in x) for _ in range(draws))
    return [round(ms[int(.025*draws)], 2), round(ms[int(.975*draws)], 2)]


def show(name, rows, key="roi"):
    if len(rows) < 10:
        print(f"   {name:44} n {len(rows)}")
        return None
    rows = sorted(rows, key=lambda r: r["ts"])
    h = len(rows)//2
    m = lambda rs: mean(r[key] for r in rs)
    out = dict(n=len(rows), events=len({r["event"] for r in rows}), losses=sum(r["payout"] < .5 for r in rows),
               mean=m(rows), ci=boot(rows, key), first=m(rows[:h]), second=m(rows[h:]),
               first_ci=boot(rows[:h], key), second_ci=boot(rows[h:], key))
    print(f"   {name:44} n {out['n']:5} losses {out['losses']:3}  {out['mean']:+.2f}% {out['ci']}"
          f"   halves {out['first']:+.2f}% {out['first_ci']} / {out['second']:+.2f}% {out['second_ci']}")
    return out


def main():
    saved_rows = [r for r in json.loads((ROOT/"data/processed/near-end/main-trades.json").read_text()) if r["series"] in TENNIS]
    markets = {}
    for r in saved_rows:
        markets[r["ticker"]] = (r["series"], r["event"], round(r["ts"]+60*r["minutes_to_settlement"]))
    rows, bars_of, missing = [], {}, 0
    outcome = {}
    for r in saved_rows:
        outcome[r["ticker"]] = r["payout"] if r["side"] == "yes" else 1-r["payout"]
    for ticker, (series, event, settle) in markets.items():
        bars = load_bars(series, ticker, settle)
        if bars is None:
            missing += 1
            continue
        bars_of[ticker] = bars
        for e in entries(bars, settle, outcome[ticker]):
            rows.append(dict(e, series=series, event=event, ticker=ticker))
    mine = table(rows)
    same = all(mine[k]["trades"] == SAVED[k]["trades"] and abs(mine[k]["net_cents_per_contract"]-SAVED[k]["net_cents_per_contract"]) < 1e-9
               and mine[k]["ci95"] == SAVED[k]["ci95"] for k in SAVED)
    print(f"1. REPRODUCTION: {len(markets)} tennis markets with an entry, bars missing for {missing}; all {len(SAVED)} table rows identical"
          f" (n, mean, interval): {same}")
    print("   (markets that never produced an entry are not in the saved rows, so they are not rebuilt; they add no trades)")
    out = dict(reproduced=same)

    # per-entry extras from the bars
    index = {t: {b["end_period_ts"]: i for i, b in enumerate(bs)} for t, bs in bars_of.items()}
    for r in rows:
        bs = bars_of[r["ticker"]]
        i = index[r["ticker"]][r["ts"]]
        b = bs[i]
        bid, ask = cents(b, "yes_bid", "open"), cents(b, "yes_ask", "open")
        r["spread"] = ask-bid
        cost = r["price"]+7*r["price"]/100*(1-r["price"]/100)
        r["roi"] = 100*r["net_cents"]/cost
        r["cost"] = cost
        # resting bid 1c under the entry ask, placed during the entry minute, live for `life` later minutes
        limit = r["price"]-1
        r["rest"] = {}
        for life in (3, 10, 30, 9999):
            filled = False
            for nb in bs[i+1:]:
                if nb["end_period_ts"] > r["ts"]+60*life:
                    break
                lo, hi = cents(nb, "price", "low"), cents(nb, "price", "high")
                if lo is None:
                    continue
                if (r["side"] == "yes" and lo < limit) or (r["side"] == "no" and 100-hi < limit):
                    filled = True
                    break
            r["rest"][life] = filled
        fee = 100*MAKER[r["series"]]*limit/100*(1-limit/100)
        r["maker_roi"] = 100*(100*r["payout"]-limit-fee)/(limit+fee)
        r["maker_roi_nofee"] = 100*(100*r["payout"]-limit)/limit

    print("\n2. TAKER RULE, % per bet (net / (ask+fee)), event bootstrap, chronological halves")
    out["taker"] = {}
    for t in (97, 98):
        base = [r for r in rows if r["threshold"] == t]
        cuts = {f">={t}c all (as reported)": base,
                f">={t}c risen (as reported)": [r for r in base if r["risen"]],
                f">={t}c all, spread<=2c": [r for r in base if r["spread"] <= 2],
                f">={t}c risen, spread<=2c": [r for r in base if r["risen"] and r["spread"] <= 2],
                f">={t}c risen, spread<=2c, minute traded": [r for r in base if r["risen"] and r["spread"] <= 2 and r["minute_volume"] > 0],
                f">={t}c NOT risen": [r for r in base if not r["risen"]],
                f">={t}c spread>2c": [r for r in base if r["spread"] > 2]}
        for k, v in cuts.items():
            out["taker"][k] = show(k, v)
        for s in TENNIS:
            k = f">={t}c risen, spread<=2c, {s}"
            out["taker"][k] = show(k, [r for r in base if r["risen"] and r["spread"] <= 2 and r["series"] == s])
    both = defaultdict(set)
    for r in rows:
        if r["threshold"] == 98:
            both[r["event"]].add(r["side"])
    print(f"   matches bought on BOTH sides at >=98c: {sum(len(v) == 2 for v in both.values())} of {len(both)}")

    print("\n3. PLANTED EDGE on '>=98c risen' (payout redrawn per market so the true return is x; the study's rule: event-bootstrap interval above zero; also both halves positive)")
    base = sorted([r for r in rows if r["threshold"] == 98 and r["risen"]], key=lambda r: r["ts"])
    cap = 100*(100/mean(r["cost"] for r in base)-1)
    print(f"   largest edge that can exist at this price: {cap:+.2f}% per bet (a +3% edge is impossible; +1.5% needs a 99.9% win rate)")
    rng = random.Random(4)
    out["planted"] = {}
    for x in (0.0, .0025, .005, .0075, .01, .015):
        hit = both_ok = 0
        sims = 300
        for _ in range(sims):
            sim = []
            for r in base:
                win = rng.random() < min(1.0, r["cost"]/100*(1+x))
                sim.append(dict(event=r["event"], v=100*win-r["cost"]))
            m = mean(s["v"] for s in sim)
            se = (sum((s["v"]-m)**2 for s in sim)/(len(sim)-1))**.5/len(sim)**.5
            ok = m-1.96*se > 0
            h = len(sim)//2
            hit += ok
            both_ok += ok and mean(s["v"] for s in sim[:h]) > 0 and mean(s["v"] for s in sim[h:]) > 0
        out["planted"][x] = dict(interval=hit/sims, both=both_ok/sims)
        print(f"   true edge {100*x:+.2f}%: interval above zero in {100*hit/sims:3.0f}% of runs; and both halves positive {100*both_ok/sims:3.0f}%")

    print("\n4. RESTING BID 1c under the entry ask (>=98c risen, spread<=2c at entry); fill = a later minute's trade strictly through the bid;"
          "\n   maker fee 1.75%*p*(1-p) on ATP and WTA, zero on Challenger; % per filled bet")
    out["resting"] = {}
    for t in (97, 98):
        base = [r for r in rows if r["threshold"] == t and r["risen"] and r["spread"] <= 2]
        for life in (3, 10, 30, 9999):
            f = [r for r in base if r["rest"][life]]
            u = [r for r in base if not r["rest"][life]]
            k = f">={t}c bid live {life if life < 9999 else 'to settlement'} min"
            s = show(k, f, "maker_roi")
            if s:
                s.update(orders=len(base), fill_rate=len(f)/len(base), per_order=s["mean"]*len(f)/len(base),
                         unfilled_taker_roi=mean(r["roi"] for r in u) if u else None, filled_taker_roi=mean(r["roi"] for r in f),
                         filled_losses=sum(r["payout"] < .5 for r in f), unfilled_losses=sum(r["payout"] < .5 for r in u))
                print(f"       fill rate {100*s['fill_rate']:.0f}% of {len(base)}; per order placed {s['per_order']:+.2f}%; losers among filled"
                      f" {s['filled_losses']}/{len(f)} vs unfilled {s['unfilled_losses']}/{len(u)}; the same filled bets paid at the ask: {s['filled_taker_roi']:+.2f}%")
            out["resting"][k] = s
        for sname in TENNIS:
            k = f">={t}c bid live 3 min, {sname}"
            out["resting"][k] = show(k, [r for r in base if r["rest"][3] and r["series"] == sname], "maker_roi")
    OUT.write_text(json.dumps(out, indent=1, default=str))
    print("\nsaved", OUT)


if __name__ == "__main__":
    main()
