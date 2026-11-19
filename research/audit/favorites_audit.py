"""Audit of research/favorites.py + favorites_report.py (90-99c contracts outside sports).

Offline: imports the original functions unmodified, passes client=None (a cache miss would raise, not
download) and writes nothing to the original output folder.

  1. rebuild the decisions from the cached bars and compare the rule table with the saved summary.json
  2. % per bet, event bootstrap, both parts, for the pooled rule and each cell
  3. planted edge: redraw payouts so every bet has a true +X% return, apply the script's own pass rule
  4. resting bids (saved resting.json, repo fill rule): % per fill, planted edge on the same rule
"""
import json, random, sys
from collections import defaultdict
from pathlib import Path
from statistics import mean

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from research import favorites_report as fr

OUT = ROOT/"data/processed/audit/inventory/favorites-audit.json"
SAVED = json.loads((ROOT/"data/processed/favorites/summary.json").read_text())


def pct(rows, net="net_cents", cost=lambda r: r["price"]+r["fee_cents"]):
    return 100*sum(r[net] for r in rows)/sum(cost(r) for r in rows)


def boot(rows, net="net_cents", cost=lambda r: r["price"]+r["fee_cents"], draws=3000, seed=11):
    by = defaultdict(list)
    for r in rows:
        by[r["event"]].append((r[net], cost(r)))
    groups = [(sum(a for a, _ in g), sum(b for _, b in g)) for g in by.values()]
    if len(groups) < 10:
        return None
    rng = random.Random(seed)
    out = []
    for _ in range(draws):
        pick = rng.choices(groups, k=len(groups))
        out.append(100*sum(a for a, _ in pick)/sum(b for _, b in pick))
    out.sort()
    return [round(out[int(.025*draws)], 2), round(out[int(.975*draws)], 2)]


def line(name, rows, **kw):
    if not rows:
        return
    a, b = fr.split(rows)
    f = lambda rs: "%+6.2f%% %s" % (pct(rs, **kw), boot(rs, **kw)) if rs else "none"
    print(f"   {name:34} bets {len(rows):4} events {len({r['event'] for r in rows}):4}  all {f(rows)}   early60 {f(a)}   late40 {f(b)}")
    return dict(bets=len(rows), events=len({r["event"] for r in rows}), all=pct(rows, **kw), all_ci=boot(rows, **kw),
                early=pct(a, **kw) if a else None, late=pct(b, **kw) if b else None, late_ci=boot(b, **kw) if b else None)


def passes(rows):
    """The script's own rule: both parts positive and the late part's event-bootstrap interval above zero."""
    a, b = fr.split(rows)
    if not a or not b:
        return False
    if mean(r["net_cents"] for r in a) <= 0 or mean(r["net_cents"] for r in b) <= 0:
        return False
    ci = fr.interval(b, draws=400)
    return bool(ci and ci[0] > 0)


def plant(rows, x, rng, net="net_cents", cost=lambda r: r["price"]+r["fee_cents"]):
    """One shared uniform per market side, so the same contract wins or loses at every snapshot."""
    u = {}
    out = []
    for r in rows:
        k = (r["ticker"], r["side"])
        if k not in u:
            u[k] = rng.random()
        c = cost(r)
        win = u[k] < min(1.0, c/100*(1+x))
        out.append(dict(r, payout=float(win), **{net: 100*win-c}))
    return out


def power(name, rows, rule, sims=300, **kw):
    rng = random.Random(21)
    res = {}
    for x in (0.0, .015, .03, .045):
        hits = sum(rule(plant(rows, x, rng, **kw)) for _ in range(sims))
        res[x] = hits/sims
    print(f"   {name:34} detected at true edge 0%: {100*res[0.0]:3.0f}%   +1.5%: {100*res[.015]:3.0f}%   +3%: {100*res[.03]:3.0f}%"
          f"   +4.5%: {100*res[.045]:3.0f}%   (largest possible edge at this price: {100*(100/mean(kw.get('cost', lambda r: r['price']+r['fee_cents'])(r) for r in rows)-1):+.1f}%)")
    return res


def main():
    series, markets, picked, fetched, rows = fr.build(None)
    taken = fr.one_per_event([r for r in rows if r["net_cents"] is not None])
    table = fr.rule_table(taken)
    saved = SAVED["tables"]["plain"]["overall"]
    same = all(table[k]["late"] == saved[k]["late"] and table[k]["early"] == saved[k]["early"] for k in saved)
    print(f"1. REPRODUCTION: sampled markets {len(picked)}, with cached bars {len(fetched)}, decisions {len(rows)}, entries {len(taken)};"
          f" all {len(saved)} rule rows identical to summary.json: {same}")
    print("   all rules late:", table["all rules"]["late"]["net_cents"], table["all rules"]["late"]["ci95"],
          " early:", table["all rules"]["early"]["net_cents"], " cells the script itself marked PASS:",
          [k for k, v in table.items() if v["passes"]])
    out = dict(reproduced=same, cells={}, power={}, resting={})

    print("\n2. % PER BET (net / (price+fee)), event bootstrap")
    cells = {"all rules": taken}
    for b in ("90-94", "95-97", "98-99"):
        cells[f"band {b}"] = [r for r in taken if r["band"] == b]
    for d in fr.DAYS:
        cells[f"D{d}"] = [r for r in taken if r["days"] == d]
    cells["D1 90-94"] = [r for r in taken if r["days"] == 1 and r["band"] == "90-94"]
    cells["liquid (vol>=100, spread<=3)"] = fr.one_per_event([r for r in rows if r["net_cents"] is not None and r["volume_24h"] >= 100 and r["spread"] <= 3])
    one = {}
    for r in sorted(taken, key=lambda r: (r["days"], r["ticker"])):      # one bet per event: its nearest-to-close entry
        one.setdefault(r["event"], r)
    cells["one bet per event (nearest close)"] = list(one.values())
    for k, v in cells.items():
        out["cells"][k] = line(k, v)

    print("\n3. PLANTED EDGE, taker rule, script's own pass rule (both parts > 0 and late interval > 0)")
    for k in ("all rules", "band 90-94", "band 95-97", "band 98-99", "D1", "one bet per event (nearest close)"):
        out["power"][k] = power(k, cells[k], passes)
    pooled = lambda rs: (boot(rs, draws=400) or [-1])[0] > 0
    print("   alternative rule: whole-sample event-bootstrap interval above zero")
    for k in ("all rules", "band 90-94", "band 98-99"):
        out["power"][k+" | pooled"] = power(k, cells[k], pooled)

    print("\n4. RESTING BID 1c under the ask for 24h (saved resting.json; fill = a taker print strictly through the bid)")
    rest = json.loads((ROOT/"data/processed/favorites/resting.json").read_text())
    fills = [r for r in rest if r["filled"]]
    mk = dict(net="maker_net", cost=lambda r: r["bid"]+(r["bid"]-(100*r["payout"]-r["maker_net"]))*-1)   # bid + maker fee
    print(f"   orders {len(rest)}, fills {len(fills)} ({100*len(fills)/len(rest):.0f}%)")
    groups = {"all fills": fills}
    for b in ("90-94", "95-97", "98-99"):
        groups[f"fills band {b}"] = [r for r in fills if r["band"] == b]
    groups["fills spread>=2c"] = [r for r in fills if r["spread"] >= 2]
    groups["fills spread 1c"] = [r for r in fills if r["spread"] == 1]
    for k, v in groups.items():
        out["resting"][k] = line(k, v, **mk)
    rule = lambda rs: (boot(rs, draws=400, **mk) or [-1])[0] > 0 and all(pct(h, **mk) > 0 for h in fr.split(rs) if h)
    print("   planted edge on fills (rule: interval above zero and both parts positive)")
    for k in ("all fills", "fills band 90-94"):
        out["power"]["resting "+k] = power(k, groups[k], rule, **mk)
    OUT.write_text(json.dumps(out, indent=1, default=str))
    print("\nsaved", OUT)


if __name__ == "__main__":
    main()
