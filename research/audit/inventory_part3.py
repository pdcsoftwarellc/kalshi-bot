"""Audit helper (inventory part 3): event-clustered bootstrap intervals and chronological halves
for dip-bid rows that the studies rejected on a point estimate or a small fill count.

Read-only: it only loads saved order/row files under data/processed/. No network.

    .venv/bin/python research/audit/inventory_part3.py favorite_dip
    .venv/bin/python research/audit/inventory_part3.py new_families

Per-bet return = (100*payout - limit - maker_fee) / limit, in percent of stake.
Bootstrap: resample independent events (the event ticker) with replacement, 4000 draws,
ratio-free mean of per-bet returns. Halves: fills ordered by settlement time, split at the median fill.
Maker fee, where shown, is 1.75*p*(1-p) cents (rate unconfirmed in the repo) on series whose
fee_type is quadratic_with_maker_fees, times the series multiplier.
"""
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
P = ROOT / "data/processed"
DRAWS = 4000
# fee_type per data/processed/dip-new-families/fees.json and favourite-contexts/fees.json; the tennis
# entries follow research/parallel-round4.md ("NBA, NHL, ATP and WTA charge fees on resting orders;
# ATP Challenger ... do not") and are NOT in a saved fees file.
MAKER_MULT = {"KXNBAGAME": 1, "KXNHLGAME": 1, "KXNFLGAME": 1, "KXMLBGAME": .5, "KXWNBAGAME": 1, "KXNCAAMBGAME": 1,
              "KXATPMATCH": 1, "KXWTAMATCH": 1, "KXEPLGAME": 1, "KXLALIGAGAME": 1, "KXSERIEAGAME": 1,
              "KXBUNDESLIGAGAME": 1, "KXLIGUE1GAME": 1, "KXUCLGAME": 1, "KXWCGAME": 1}


def series_of(event):
    return event.split("-")[0]


def stat(fills, label, fee=False, out=sys.stdout):
    """fills: dicts with event, settle, limit, payout. Prints one line and returns the numbers."""
    if not fills:
        print(f"{label:58} fills 0", file=out)
        return None
    fills = sorted(fills, key=lambda r: (r["settle"], r["event"]))
    lim = np.array([float(r["limit"]) for r in fills])
    pay = np.array([float(r["payout"]) for r in fills])
    mult = np.array([MAKER_MULT.get(r.get("series") or series_of(r["event"]), 0) for r in fills]) if fee else np.zeros(len(fills))
    cost = 1.75*mult*lim/100*(1-lim/100)
    net = 100*pay-lim-cost
    pct = 100*net/lim
    events = sorted({r["event"] for r in fills})
    index = {e: i for i, e in enumerate(events)}
    owner = np.array([index[r["event"]] for r in fills])
    sums = np.bincount(owner, weights=pct, minlength=len(events))
    counts = np.bincount(owner, minlength=len(events)).astype(float)
    rng = np.random.default_rng(9)
    if len(events) >= 5:
        pick = rng.integers(0, len(events), (DRAWS, len(events)))
        means = sums[pick].sum(axis=1)/counts[pick].sum(axis=1)
        lo, hi = np.percentile(means, [2.5, 97.5])
        ci = f"[{lo:+.1f}, {hi:+.1f}]"
    else:
        ci = "n/a(<5 events)"
    half = len(fills)//2
    h1 = pct[:half].mean() if half else float("nan")
    h2 = pct[half:].mean()
    print(f"{label:58} fills {len(fills):5} events {len(events):5} avg-limit {lim.mean():5.1f} win {pay.mean():.3f} "
          f"net {net.mean():+6.2f}c = {pct.mean():+6.2f}% ci {ci} halves {h1:+.1f}% (n={half}) / {h2:+.1f}% (n={len(fills)-half})"
          f"{' [maker fee]' if fee else ''}", file=out)
    return dict(fills=len(fills), events=len(events), pct=float(pct.mean()), ci=ci, h1=float(h1), h2=float(h2))


def load(name):
    return json.loads((P / name).read_text())


def band(rows, lo, hi, limit=None, prefix=None, level=None, fill="strict"):
    out = []
    for r in rows:
        if not lo <= r["start"] < hi:
            continue
        if limit is not None and r.get("limit") != limit:
            continue
        if level is not None and r.get("level") != level and r.get("kind") != level and r.get("rule") != level:
            continue
        if prefix and not r["event"].startswith(prefix):
            continue
        out.append(r)
    return out, [r for r in out if r.get(fill)]


def show(rows, label, lo, hi, limit=None, prefix=None, level=None, fee_too=False):
    orders, fills = band(rows, lo, hi, limit, prefix, level)
    events = len({r["event"] for r in orders})
    print(f"  -- {label}: orders {len(orders)} (events {events})")
    stat(fills, "     "+label)
    if fee_too:
        stat(fills, "     "+label, fee=True)


def favorite_dip():
    recent, arch = load("favorite-dip/tennis-orders.json"), load("favorite-dip/tennis-archived-orders.json")
    for name, rows in (("recent Sep18-Nov16", recent), ("archive Jan1-Sep17", arch), ("both", recent+arch)):
        print(f"\nTENNIS (ATP+Challenger+WTA) {name}")
        for prefix, tour in (("KXATPMATCH-", "ATP"), ("KXATPCHALLENGERMATCH-", "Challenger"), ("KXWTAMATCH-", "WTA")):
            for limit in (80, 70):
                show(rows, f"{tour} 90+ bid {limit}", 90, 101, limit, prefix, fee_too=tour != "Challenger")
        for lo, hi, limits in ((80, 90, (75, 70, 65, 60)), (70, 80, (65, 60, 50))):
            for limit in limits:
                show(rows, f"all tours {lo}-{hi} bid {limit}", lo, hi, limit)
            for prefix, tour in (("KXATPMATCH-", "ATP"), ("KXATPCHALLENGERMATCH-", "Challenger"), ("KXWTAMATCH-", "WTA")):
                show(rows, f"{tour} {lo}-{hi} bid {limits[1] if lo == 80 else limits[1]}", lo, hi, limits[1], prefix)
    for group in ("nba", "nfl", "nhl", "mlb"):
        parts = []
        for tag in ("", "-archived"):
            path = P / f"favorite-dip/{group}{tag}-orders.json"
            if path.exists():
                parts.append((tag or "-recent", json.loads(path.read_text())))
        if len(parts) > 1:
            parts.append(("-both", parts[0][1]+parts[1][1]))
        for tag, rows in parts:
            print(f"\n{group.upper()}{tag}")
            for lo, hi, limits in ((90, 101, (80, 75, 70)), (80, 90, (70, 65, 60)), (70, 80, (60, 50))):
                for limit in limits:
                    show(rows, f"{group} {lo}-{hi} bid {limit}", lo, hi, limit, fee_too=True)
    for fam in ("cs2", "lol", "valorant", "dota2", "boxing", "ufc", "ncaaf", "cricket-t20", "cricket-odi", "wnba",
                "euroleague", "soccer-europe"):
        rows = load(f"favorite-dip/other-sports-{fam}-orders.json")
        print(f"\nOTHER SPORTS {fam}: rows {len(rows)} events {len({r['event'] for r in rows})}")
        levels = sorted({str(r.get('level')) for r in rows})
        print("   levels:", levels)
        for lo, hi, lv in ((90, 101, ("80", "70", "60", "pre-10", "pre-15")), (80, 90, ("pre-10", "pre-15"))):
            for level in lv:
                show(rows, f"{fam} {lo}-{hi} level {level}", lo, hi, level=level, fee_too=fam in ("wnba", "soccer-europe"))
    esports = sum((load(f"favorite-dip/other-sports-{f}-orders.json") for f in ("cs2", "lol", "valorant", "dota2")), [])
    print("\nESPORTS pooled (cs2+lol+valorant+dota2)")
    for lo, hi, lv in ((90, 101, ("80", "70", "pre-10")), (80, 90, ("pre-10", "pre-15"))):
        for level in lv:
            show(esports, f"esports {lo}-{hi} level {level}", lo, hi, level=level)
    combat = load("favorite-dip/other-sports-boxing-orders.json")+load("favorite-dip/other-sports-ufc-orders.json")
    print("\nCOMBAT pooled (boxing+ufc)")
    for lo, hi, lv in ((90, 101, ("80", "70", "pre-10")), (80, 90, ("pre-10", "pre-15"))):
        for level in lv:
            show(combat, f"combat {lo}-{hi} level {level}", lo, hi, level=level)
    for tour in ("itf-women", "itf-men", "wta-challenger"):
        rows = load(f"favorite-dip/{tour}-archive-holdout-orders.json")
        print(f"\nARCHIVE HOLDOUT {tour}: kinds {sorted({str(r.get('kind')) for r in rows})}")
        for limit in (80, 75, 70, 65):
            orders = [r for r in rows if r["start"] >= 90 and r.get("kind") in (None, "fixed", f"fixed {limit}", str(limit)) and r["limit"] == limit]
            print(f"  -- {tour} 90+ bid {limit}: orders {len(orders)}")
            stat([r for r in orders if r["strict"]], f"     {tour} archive 90+ bid {limit}")
        rec = load(f"favorite-dip/{tour}-orders.json")
        for limit in (80, 70):
            show(rec, f"{tour} RECENT 90+ bid {limit}", 90, 101, limit)
        both = [r for r in rows if r["limit"] == 80 and r["start"] >= 90 and r.get("kind") in (None, "fixed", "fixed 80", "80")] + \
               [r for r in rec if r["limit"] == 80 and r["start"] >= 90]
        seen, uniq = set(), []
        for r in both:
            if r["event"] not in seen:
                seen.add(r["event"])
                uniq.append(r)
        print(f"  -- {tour} archive+recent 90+ bid 80: orders {len(uniq)}")
        stat([r for r in uniq if r["strict"]], f"     {tour} archive+recent 90+ bid 80")
    team = load("favorite-dip/team-events-orders.json")
    print("\nTEAM EVENTS / EXHIBITION / DOUBLES: rules", sorted({str(r.get("rule")) for r in team}))
    for group in ("team", "exhibition", "doubles", None):
        for rule in sorted({str(r.get("rule")) for r in team}):
            sub = [r for r in team if (group is None or r["group"] == group) and str(r.get("rule")) == rule]
            print(f"  -- {group or 'all'} {rule}: orders {len(sub)}")
            stat([r for r in sub if r["strict"]], f"     {group or 'all'} {rule}")


if __name__ == "__main__":
    {"favorite_dip": favorite_dip}[sys.argv[1]]()
