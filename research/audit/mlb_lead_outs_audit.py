"""Audit of the withdrawn MLB lead-and-outs rule (research/game_state.py + rule_search.py).

Rule: late innings (7th on), leader ahead by 3+ runs and by at least as many runs as the trailing side has
outs left, leader's ask 90-99c; buy at the next minute's opening ask, hold to settlement.

Offline: a client that refuses every request is passed to the original helpers, so only caches are read.

  1. reproduce the saved result from data/processed/game-state/mlb-minutes.json
  2. rebuild from the raw caches at feed delays of 15-300 s, with intervals (the write-up gave none past 90 s)
  3. the 239 games that have a per-play log with scoring-play flags: lookahead-free state against the raw state
  4. planted edge
  5. resting bid 1c under the ask (strict through-print), MLB maker fee
"""
import json, random, sys
from bisect import bisect_right
from collections import Counter
from pathlib import Path
from statistics import mean, pstdev

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from research import game_state as gs
from research.api_cost import fetch
from research.near_end import WINDOW, cents
from research.rule_search import entries as rule_entries, cushion
from research.decided_report import safe_log

OUT = ROOT/"data/processed/audit/inventory/mlb-lead-outs-audit.json"
RULE = (90, 99, 3, 27, 0, 1)
TAKER, MAKER = .07, .0175


class Offline:
    def get(self, *a, **k):
        raise RuntimeError("offline audit: cache miss")


def load():
    """(game, espn event, raw log, safe log or None, bars) for every joinable game, as game_state.build does."""
    cfg, client = gs.SPORTS["mlb"], Offline()
    games = list(gs.kalshi_games(client, cfg["series"]))
    skip, out = Counter(), []
    for g in games:
        f = gs.RAW/"mlb"/f"day-{g['day']:%Y%m%d}.json"
        if not f.exists():
            skip["no_cached_slate"] += 1
            continue
        slate = json.loads(f.read_text())
        match = [e for e in slate if set(e["teams"].values()) == g["teams"] and e["status"] == "STATUS_FINAL" and e["start"] < g["settle"]]
        if not match:
            skip["no_matching_final_game"] += 1
            continue
        e = max(match, key=lambda e: e["start"])
        pf = gs.RAW/"mlb"/f"game-{e['id']}.json"
        if not pf.exists():
            skip["no_cached_plays"] += 1
            continue
        log = json.loads(pf.read_text())
        path = ("/historical/markets/{t}/candlesticks" if g["archived"] else "/series/"+cfg["series"]+"/markets/{t}/candlesticks").format(t=g["ticker"])
        try:
            bars = fetch(client, path, dict(start_ts=g["settle"]-WINDOW, end_ts=g["settle"], period_interval=1))["candlesticks"]
        except (RuntimeError, KeyError):
            bars = None
        if not bars or len(log) < 50:
            skip["no_bars_or_plays"] += 1
            continue
        yes_home = e["teams"]["home"] == g["yes"]
        if (log[-1][3] > log[-1][4]) != ((g["outcome"] == 1) == yes_home):
            skip["archive_winner_disagrees"] += 1
            continue
        sf = gs.RAW/"decided"/f"plays-mlb-{e['id']}.json"
        safe = None
        if sf.exists():
            data = json.loads(sf.read_text())
            if data:
                rows, sound = safe_log("mlb", data)
                safe = rows if sound else None
        out.append(dict(g=g, yes_home=yes_home, log=log, safe=safe, bars=sorted(bars, key=lambda b: b["end_period_ts"])))
    return out, dict(skip, kalshi_games=len(games), joined=len(out))


def minutes(game, log, delay, mult=1.0):
    """game_state.build's inner loop, with the delay and the fee multiplier as parameters."""
    g, bars, rows = game["g"], game["bars"], []
    times = [p[0] for p in log]
    for j, (bar, nxt) in enumerate(zip(bars, bars[1:])):
        i = bisect_right(times, bar["end_period_ts"]-delay)-1
        if i < 0 or nxt["end_period_ts"]-bar["end_period_ts"] > 120 or bar["end_period_ts"] > times[-1]:
            continue
        _, period, left, home, away = log[i][:5]
        if period < 7 or home == away:
            continue
        leader_yes = (home > away) == game["yes_home"]
        if leader_yes:
            ask = cents(nxt, "yes_ask", "open")
        else:
            bid = cents(nxt, "yes_bid", "open")
            ask = None if bid is None else 100-bid
        if ask is None or not 0 < ask < 100:
            continue
        won = (g["outcome"] == 1) == leader_yes
        r = dict(event=g["event"], day=str(g["day"]), ts=nxt["end_period_ts"], period=period, left=left, lead=abs(home-away),
                 ask=ask, won=won, net=100*(won-ask/100-TAKER*mult*ask/100*(1-ask/100)), j=j+1, leader_yes=leader_yes)
        r["cushion"] = cushion("mlb", r)
        rows.append(r)
    return rows


def hits_for(games, delay, mult=1.0, which="log"):
    per_game = []
    for game in games:
        log = game[which]
        if log is None:
            continue
        rows = minutes(game, log, delay, mult)
        if rows:
            per_game.append((game, rows))
    per_game.sort(key=lambda x: x[1][0]["ts"])
    cut = int(len(per_game)*.6)
    hits = []
    for k, (game, rows) in enumerate(per_game):
        h = rule_entries("mlb", [rows], RULE)
        if h:
            h[0]["later"] = k >= cut
            h[0]["game"] = game
            hits.append(h[0])
    return hits


def ci(v, draws=3000, seed=5):
    if len(v) < 10:
        return None
    rng = random.Random(seed)
    ms = sorted(mean(rng.choices(v, k=len(v))) for _ in range(draws))
    return [round(ms[int(.025*draws)], 2), round(ms[int(.975*draws)], 2)]


def pct(h, key="net"):
    return [100*x[key]/(x["ask"]+(x["ask"]-100*x["won"]+x[key])*-1) for x in h] if key == "net" else [x[key] for x in h]


def line(name, hits, key="net"):
    v = pct(hits, key)
    a, b = [x for x, h in zip(v, hits) if not h["later"]], [x for x, h in zip(v, hits) if h["later"]]
    f = lambda z: "%+.2f%% %s" % (mean(z), ci(z)) if z else "none"
    print(f"   {name:34} bets {len(v):4} losses {sum(not h['won'] for h in hits):2}  all {f(v)}   early60 {f(a)}   later40 {f(b)}")
    return dict(n=len(v), losses=sum(not h["won"] for h in hits), all=mean(v) if v else None, all_ci=ci(v),
                early=mean(a) if a else None, later=mean(b) if b else None, later_ci=ci(b), early_ci=ci(a))


def main():
    out = {}
    saved = json.loads((ROOT/"data/processed/game-state/mlb-minutes.json").read_text())
    by = {}
    for r in sorted(saved, key=lambda r: r["ts"]):
        r["cushion"] = cushion("mlb", r)
        by.setdefault(r["event"], []).append(r)
    gl = sorted(by.values(), key=lambda g: g[0]["ts"])
    cut = int(len(gl)*.6)
    e, l = rule_entries("mlb", gl[:cut], RULE), rule_entries("mlb", gl[cut:], RULE)
    print(f"1. REPRODUCTION from the saved minutes: early {len(e)} bets {mean(h['net'] for h in e):+.3f}c, later {len(l)} bets "
          f"{mean(h['net'] for h in l):+.3f}c  (write-up: 707 at +0.88c, 457 at +1.32c [+0.24, +2.19])")
    games, cover = load()
    print("   rebuilt from raw caches:", cover, " games with a lookahead-free log:", sum(g["safe"] is not None for g in games))
    h15 = hits_for(games, 15)
    same = len(h15) == len(e)+len(l) and abs(mean(h["net"] for h in h15)-mean(h["net"] for h in e+l)) < 1e-9
    print(f"   rebuild at 15 s gives the same {len(h15)} bets and mean: {same}")
    out["reproduced"] = same

    print("\n2. FEED DELAY (state as of d seconds before the bar close; price still the next minute's opening ask). % per bet; MLB fee multiplier 0.5 in the second block")
    out["delay"] = {}
    for mult in (1.0, .5):
        print(f"   taker fee {TAKER*mult:.3f}*p*(1-p)")
        for d in (15, 60, 90, 120, 180, 300):
            out["delay"][f"{mult}|{d}"] = line(f"delay {d:3d} s", hits_for(games, d, mult))

    print("\n3. LOOKAHEAD-FREE STATE on the games that have scoring-play flags (runs credited only at the scoring play's own stamp, +15 s)")
    sub = [g for g in games if g["safe"] is not None]
    raw, safe = hits_for(sub, 15, .5, "log"), hits_for(sub, 15, .5, "safe")
    out["safe"] = dict(raw=line("raw state, 15 s (these games)", raw), safe=line("lookahead-free state, 15 s", safe),
                       d90=line("raw state, 90 s", hits_for(sub, 90, .5)), d180=line("raw state, 180 s", hits_for(sub, 180, .5)))
    rb, sb = {h["event"]: h for h in raw}, {h["event"]: h for h in safe}
    common = [k for k in rb if k in sb]
    later = [sb[k]["ts"]-rb[k]["ts"] for k in common]
    dearer = [sb[k]["ask"]-rb[k]["ask"] for k in common]
    print(f"   same games entered both ways: {len(common)}; raw-only {len(rb)-len(common)}, lookahead-free-only {len(sb)-len(common)};"
          f" lookahead-free entry is later by a mean {mean(later)/60:.1f} min (same minute in {sum(x == 0 for x in later)}),"
          f" and pays a mean {mean(dearer):+.2f}c more")
    out["safe"]["paired"] = dict(common=len(common), mean_later_min=mean(later)/60, mean_dearer_c=mean(dearer))

    print("\n4. PLANTED EDGE (delay 180 s, fee x0.5; outcomes redrawn; rule: later-40% interval above zero, as the study used; and whole-sample interval)")
    base = hits_for(games, 180, .5)
    cost = [h["ask"]+100*TAKER*.5*h["ask"]/100*(1-h["ask"]/100) for h in base]
    print(f"   largest edge that can exist at this price: {100*(100/mean(cost)-1):+.2f}% per bet, so +3% is impossible; +1.5% needs a {mean(cost)*1.015:.1f}% win rate")
    rng = random.Random(9)
    out["planted"] = {}
    for x in (0.0, .0025, .005, .0075, .01, .015):
        a = b = 0
        sims = 400
        for _ in range(sims):
            v = [100*((rng.random() < min(1.0, c/100*(1+x)))*100-c)/c for c in cost]
            lat = [z for z, h in zip(v, base) if h["later"]]
            se = lambda z: pstdev(z)/len(z)**.5
            a += mean(lat)-1.96*se(lat) > 0
            b += mean(v)-1.96*se(v) > 0
        out["planted"][x] = dict(later_rule=a/sims, whole_sample=b/sims)
        print(f"   true edge {100*x:+.2f}%: later-40% interval above zero {100*a/sims:3.0f}%   whole-sample interval above zero {100*b/sims:3.0f}%")

    print("\n5. RESTING BID 1c under the ask for 3 or 10 minutes (fill = a later minute's trade strictly through the bid; maker fee 0.0175*0.5*p*(1-p)); % per filled bet")
    out["resting"] = {}
    for d in (15, 180):
        base = hits_for(games, d, .5)
        for life in (3, 10):
            f = []
            for h in base:
                bars, bid = h["game"]["bars"], h["ask"]-1
                for nb in bars[h["j"]+1:]:
                    if nb["end_period_ts"] > h["ts"]+60*life:
                        break
                    lo, hi = cents(nb, "price", "low"), cents(nb, "price", "high")
                    if lo is None:
                        continue
                    if (h["leader_yes"] and lo < bid) or (not h["leader_yes"] and 100-hi < bid):
                        fee = 100*MAKER*.5*bid/100*(1-bid/100)
                        f.append(dict(h, maker=100*(100*h["won"]-bid-fee)/(bid+fee)))
                        break
            s = line(f"delay {d} s, bid live {life} min", f, "maker")
            s["orders"], s["fill_rate"] = len(base), len(f)/len(base)
            print(f"       filled {len(f)} of {len(base)} ({100*len(f)/len(base):.0f}%); per order placed {s['all']*len(f)/len(base):+.2f}%;"
                  f" losers: filled {sum(not h['won'] for h in f)}, all orders {sum(not h['won'] for h in base)}")
            out["resting"][f"{d}|{life}"] = s
    OUT.write_text(json.dumps(out, indent=1, default=str))
    print("\nsaved", OUT)


if __name__ == "__main__":
    main()
