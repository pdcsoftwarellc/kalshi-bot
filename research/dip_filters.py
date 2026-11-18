"""Two follow-ups on the tennis dip rule (WTA and ATP Challenger favourites at 90c+ before play).

(A) Does anything known before play separate good dip fills from bad ones?
    The cached window is the six hours before settlement, and Kalshi gives no
    start time, so the start of play is inferred causally from the bars: the
    first bar after the warm-up where the favourite's midpoint has moved more
    than 2c in ten minutes, or where open interest has grown by half and by
    2,000 contracts in ten minutes. Features use only bars before that one,
    over fixed lookbacks (a lookback tied to the window start would encode the
    match length, which is known only afterwards). The 80c bid then rests from
    the bar after the detected start. Buckets are holdout terciles, applied
    unchanged to the recent period.

(B) Bids set relative to the pre-match price (round(start)-k for k in 10..35),
    alone and as two- and three-rung ladders, one contract per rung, against
    the fixed 80c bid and the 80c+65c ladder. Fills follow favorite_dip.py: the
    bid rests from the end of the 15-bar warm-up and fills when a trade prints
    strictly below it in a later bar. The ladder is picked on the holdout only
    (highest profit per match placed with a 95% lower bound above zero) and
    then scored on the recent period.

Matches listed in both periods (settled 12 to 17 September) are counted in the
holdout only. Per-fill intervals resample matches, because the rungs of one
ladder win or lose together.

Replays from the on-disk cache only: the HTTP client is a stub that answers
404, so nothing is fetched. Holdout is January to 17 September 2026 (archived),
recent is September to mid November 2026.
"""
from itertools import combinations
from statistics import median
import sys
from pathlib import Path

import httpx
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from research.api_cost import ROOT, save
from research.favorite_dip import WARMUP, matches
from research.near_end import cents

OUT = ROOT / "data/processed/favorite-dip"
TOURS = ("KXWTAMATCH", "KXATPCHALLENGERMATCH")
OFFSETS = (10, 15, 20, 25, 30, 35)
FEATURES = ("drift_60m", "drift_120m", "drift_since_warmup", "spread_60m", "range_60m", "volume_60m",
            "open_interest", "price_before_play", "warmup_slope", "warmup_spread", "warmup_volume")
RNG = np.random.default_rng(9)


def number(bar, key):
    value = bar.get(key, bar.get(key+"_fp"))      # recent bars carry a "_fp" suffix
    return float(value) if value is not None else 0.0


def study(job, bars):
    """One heavy favourite: favourite-side bars, detected start of play, pre-play features, lowest prints."""
    event, ticker, settle, outcome = job
    mids = []
    for b in bars[:WARMUP]:
        bid, ask = cents(b, "yes_bid", "close"), cents(b, "yes_ask", "close")
        if bid is None or ask is None or not 0 < bid <= ask < 100 or ask-bid > 4:
            return None
        mids.append((bid+ask)/2)
    if len(mids) < WARMUP or max(mids)-min(mids) > 3:
        return None
    yes = median(mids) > 50
    start = median(mids) if yes else 100-median(mids)
    if start < 90:
        return None
    path = []                                   # (time, mid, spread, low print, volume, open interest), favourite side
    for b in bars:
        bid, ask = cents(b, "yes_bid", "close"), cents(b, "yes_ask", "close")
        low = cents(b, "price", "low") if yes else (None if cents(b, "price", "high") is None else 100-cents(b, "price", "high"))
        quoted = bid is not None and ask is not None
        mid = None if not quoted else ((bid+ask)/2 if yes else 100-(bid+ask)/2)
        path.append((b["end_period_ts"], mid, ask-bid if quoted else None, low, number(b, "volume"), number(b, "open_interest")))

    def at(i, seconds):                         # index of the last bar at least `seconds` before bar i, or None
        j = next((j for j in range(i, -1, -1) if path[j][0] <= path[i][0]-seconds), None)
        return j

    began = None
    for i in range(WARMUP, len(path)):
        j = at(i, 600)
        if j is None or path[i][1] is None or path[j][1] is None:
            continue
        if abs(path[i][1]-path[j][1]) > 2 or (path[i][5]-path[j][5] >= 2000 and path[i][5] >= 1.5*path[j][5]):
            began = i
            break
    lows = [p[3] for p in path[WARMUP:] if p[3] is not None]
    row = dict(event=event, series=ticker.split("-")[0], settle=settle, start=start,
               payout=outcome if yes else 1-outcome, low=min(lows) if lows else None, low_in_play=None, features=None,
               open_interest_at_window_start=path[0][5])
    if began is None:
        return row
    after = [p[3] for p in path[began+1:] if p[3] is not None]
    row["low_in_play"] = min(after) if after else None
    last = began-1
    now = next((path[j][1] for j in range(last, -1, -1) if path[j][1] is not None), None)
    side = 1 if yes else -1
    warm = [m if yes else 100-m for m in mids]
    f = dict(drift_since_warmup=now-start, price_before_play=now,
             warmup_slope=float(np.mean(warm[-5:])-np.mean(warm[:5])),
             warmup_spread=float(np.mean([p[2] for p in path[:WARMUP]])),
             warmup_volume=sum(p[4] for p in path[:WARMUP]), open_interest=path[last][5],
             minutes_before_settlement=(settle-path[began][0])/60)
    for name, seconds in (("60m", 3600), ("120m", 7200)):
        j = at(last, seconds)
        f["drift_"+name] = None if j is None or path[j][1] is None else now-path[j][1]
        if name == "60m":
            recent = path[(j or 0)+1:last+1] if j is not None else None
            quoted = [p for p in recent or [] if p[1] is not None]
            f["spread_60m"] = float(np.mean([p[2] for p in quoted])) if quoted else (path[last][2] if recent is not None else None)
            f["range_60m"] = None if recent is None else (max(p[1] for p in quoted+[path[j]])-min(p[1] for p in quoted+[path[j]]) if path[j][1] is not None else None)
            f["volume_60m"] = None if recent is None else sum(p[4] for p in recent)
    row["features"] = f
    return row


def ci(values, draws=4000):
    values = np.asarray(values, float)
    if len(values) < 10:
        return None
    means = values[RNG.integers(0, len(values), (draws, len(values)))].mean(axis=1)
    return [round(float(v), 2) for v in np.percentile(means, [2.5, 97.5])]


def per_fill(nets, limits, wins, by_match):
    """Fill statistics; the interval resamples matches, since rungs of one ladder win or lose together."""
    if not nets:
        return dict(fills=0)
    got = np.array([m for m in by_match if m[1]], float)        # (profit, fills) per match with a fill
    band = None
    if len(got) >= 10:
        pick = got[RNG.integers(0, len(got), (4000, len(got)))]
        band = [round(float(v), 2) for v in np.percentile(pick[:, :, 0].sum(axis=1)/pick[:, :, 1].sum(axis=1), [2.5, 97.5])]
    return dict(fills=len(nets), win_rate=float(np.mean(wins)), needed=float(np.mean(limits))/100,
                net_cents=float(np.mean(nets)), ci95=band)


def rung(rows, limit, low="low"):
    """Per-match profit of one bid; `limit` is a fixed price or a function of the row."""
    out = []
    for r in rows:
        price = limit(r) if callable(limit) else limit
        out.append((price, 100*r["payout"]-price) if r[low] is not None and r[low] < price else None)
    return out


def ladder(rows, limits, low="low"):
    legs = [rung(rows, limit, low) for limit in limits]
    profit = [sum(leg[i][1] for leg in legs if leg[i]) for i in range(len(rows))]
    fills = [leg[i] for leg in legs for i in range(len(rows)) if leg[i]]
    wins = [rows[i]["payout"] for leg in legs for i in range(len(rows)) if leg[i]]
    cost = sum(f[0] for f in fills)
    count = [sum(1 for leg in legs if leg[i]) for i in range(len(rows))]
    return dict(matches=len(rows), per_fill=per_fill([f[1] for f in fills], [f[0] for f in fills], wins, list(zip(profit, count))),
                matches_filled=sum(any(leg[i] for leg in legs) for i in range(len(rows))),
                per_match_cents=float(np.mean(profit)), per_match_ci95=ci(profit), worst_match_cents=min(profit),
                return_on_cost=sum(profit)/cost if cost else None)


def show(name, s):
    f = s["per_fill"]
    if not f["fills"]:
        print(f"  {name:22} no fills")
        return
    print(f'  {name:22} fills {f["fills"]:4} win {f["win_rate"]:.3f} need {f["needed"]:.3f} net {f["net_cents"]:+6.2f}c ci {f["ci95"]} | '
          f'per match {s["per_match_cents"]:+5.2f}c ci {s["per_match_ci95"]} worst {s["worst_match_cents"]:+.0f}c roc {s["return_on_cost"]:+.3f}')


def part_a(data, report):
    print("\n(A) PRE-PLAY FEATURES AS A FILTER ON THE 80c BID (bid rests from the bar after the detected start of play)")
    usable = {p: [r for r in rows if r["features"]] for p, rows in data.items()}
    for p, rows in data.items():
        mins = [r["features"]["minutes_before_settlement"] for r in usable[p]]
        base = ladder(usable[p], (80,), "low_in_play")
        print(f' {p}: {len(rows)} favourites, start of play detected in {len(usable[p])}, median {np.median(mins):.0f} minutes before settlement')
        show("all, 80c bid", base)
        report[f"A | {p} | all"] = base
    for name in FEATURES:
        have = [r for r in usable["holdout"] if r["features"][name] is not None]
        lo, hi = np.percentile([r["features"][name] for r in have], [100/3, 200/3])
        print(f' {name}: holdout tercile cuts {lo:.2f} and {hi:.2f}')
        for p in data:
            rows = [r for r in usable[p] if r["features"][name] is not None]
            parts = {"low": [r for r in rows if r["features"][name] <= lo],
                     "mid": [r for r in rows if lo < r["features"][name] <= hi],
                     "high": [r for r in rows if r["features"][name] > hi]}
            for k, part in parts.items():
                if not part:
                    continue
                s = ladder(part, (80,), "low_in_play")
                s["fill_rate"] = s["per_fill"]["fills"]/len(part)
                report[f"A | {p} | {name} | {k}"] = s
                f = s["per_fill"]
                print(f'  {p:7} {k:4} matches {len(part):4} fills {f["fills"]:3} ({s["fill_rate"]:.0%}) '
                      + (f'win {f["win_rate"]:.3f} net {f["net_cents"]:+6.2f}c ci {f["ci95"]} ' if f["fills"] else "")
                      + f'| per match {s["per_match_cents"]:+5.2f}c ci {s["per_match_ci95"]}')
            # Rank correlation between the feature and the result of a fill, the least bucket-dependent summary.
            filled = [(r["features"][name], r["payout"]) for r in rows if r["low_in_play"] is not None and r["low_in_play"] < 80]
            if len(filled) >= 20:
                x = np.argsort(np.argsort([f[0] for f in filled])).astype(float)
                y = np.array([f[1] for f in filled])
                rho = float(np.corrcoef(x, y)[0, 1])
                report[f"A | {p} | {name} | rank correlation with winning a fill"] = dict(fills=len(filled), rho=rho, noise_band=2/len(filled)**.5)
                print(f'  {p:7} rank correlation with winning a fill {rho:+.3f} (noise band about +/-{2/len(filled)**.5:.2f}, {len(filled)} fills)')


def part_b(data, report):
    print("\n(B) RELATIVE BIDS AND LADDERS (bid rests from the end of the warm-up, as in favorite_dip.py)")
    def rel(k):
        return lambda r: round(r["start"])-k
    plans = {"fixed 80": (80,), "fixed 65": (65,), "fixed 80+65": (80, 65)}
    for n in (1, 2, 3):
        for combo in combinations(OFFSETS, n):
            plans["start-"+"/".join(map(str, combo))] = tuple(rel(k) for k in combo)
    scores = {p: {name: ladder(rows, limits) for name, limits in plans.items()} for p, rows in data.items()}
    for p in data:
        print(f' {p}: {len(data[p])} matches placed')
        for name, s in scores[p].items():
            report[f"B | {p} | {name}"] = s
            show(name, s)
    eligible = {n: s for n, s in scores["holdout"].items() if n.startswith("start-") and s["per_match_ci95"] and s["per_match_ci95"][0] > 0}
    pick = max(eligible, key=lambda n: eligible[n]["per_match_cents"]) if eligible else None
    print(f"\n holdout pick (highest profit per match placed with a lower bound above zero): {pick}")
    report["B | holdout pick"] = pick
    for size in (1, 2, 3):
        best = max((n for n in eligible if n.count("/") == size-1), key=lambda n: eligible[n]["per_match_cents"], default=None)
        report[f"B | best {size}-rung on holdout"] = best
        if best:
            print(f" best {size}-rung on holdout: {best}")
            for p in data:
                show(f"  {p}", scores[p][best])
    print(" by tour, and with the bids resting only from the detected start of play")
    for name in dict.fromkeys(("fixed 80", "fixed 80+65", pick, report["B | best 2-rung on holdout"], "start-10/25", "start-10/20/30")):
        if name is None:
            continue
        for p, rows in data.items():
            for tour in TOURS:
                s = ladder([r for r in rows if r["series"] == tour], plans[name])
                report[f"B | {p} | {name} | {tour}"] = s
                show(f"{name} {p[:4]} {tour[2:6]}", s)
            s = ladder([r for r in rows if r["features"]], plans[name], "low_in_play")
            report[f"B | {p} | {name} | from detected start"] = s
            show(f"{name} {p[:4]} in-play", s)


def main():
    data = {"holdout": [], "recent": []}
    with httpx.Client(transport=httpx.MockTransport(lambda request: httpx.Response(404))) as client:      # cache only
        for archived in (True, False):
            for series in TOURS:
                for job, bars in matches(client, series, archived):
                    row = study(job, bars)
                    if row:
                        data["holdout" if archived else "recent"].append(row)
    # The two listings overlap for a few days in mid September; keep those matches in the holdout only.
    held = {r["event"] for r in data["holdout"]}
    both = [r for r in data["recent"] if r["event"] in held]
    data["recent"] = [r for r in data["recent"] if r["event"] not in held]
    save(OUT/"filters-rows.json", data)
    report = {"check | matches listed in both periods (dropped from recent)": dict(matches=len(both), flat_80=ladder(both, (80,)))}
    print(f"{len(both)} matches appear in both periods and are counted in the holdout only")
    show("  those, 80c bid", report["check | matches listed in both periods (dropped from recent)"]["flat_80"])
    for p, rows in data.items():
        busy = [r for r in rows if r["open_interest_at_window_start"] > 20000]
        print(f'{p}: {len(rows)} favourites; {len(busy)} already had more than 20,000 contracts open six hours before settlement '
              f'(possibly in play or suspended when the window opens)')
        report[f"check | {p} | open interest above 20000 at window start"] = dict(
            matches=len(busy), flat_80=ladder(busy, (80,)), rest=ladder([r for r in rows if r not in busy], (80,)))
        show("  those, 80c bid", report[f"check | {p} | open interest above 20000 at window start"]["flat_80"])
        show("  the rest, 80c bid", report[f"check | {p} | open interest above 20000 at window start"]["rest"])
    part_a(data, report)
    part_b(data, report)
    save(OUT/"filters.json", report)


if __name__ == "__main__":
    main()
