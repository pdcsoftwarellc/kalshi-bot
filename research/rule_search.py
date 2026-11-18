"""Search late-game entry rules per sport, chosen on early games and scored on later ones.

A rule buys the leading side once, at the first late-game minute where:
the ask is inside a band, the lead is at least a minimum, what remains
(seconds or outs) is inside a window, and the lead is large for the time left
("cushion": lead per square-root minute, or lead per out, remaining).

Games are ordered by date. The earliest 60% pick the rule by total profit per
contract; the latest 40% score that rule once. Because hundreds of rules are
tried, the selection sample always flatters the winner: only the later column
is evidence. The report also shows the plain price rule and the average later
result of the ten best selected rules, which is harder to get by luck.
"""
import argparse
from itertools import product
import json
from pathlib import Path
import random
from statistics import mean, pstdev
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from research.api_cost import ROOT, save

DATA = ROOT / "data/processed/game-state"
GRID = {
    "nba": dict(lead=(1, 4, 6, 8, 10, 12, 15), left_max=(2880, 720, 480, 300, 180), left_min=(0, 30, 60, 120), cushion=(0, 2, 3, 4, 5)),
    "nfl": dict(lead=(1, 4, 7, 9, 14, 17), left_max=(3600, 900, 600, 300, 120), left_min=(0, 30, 60, 120), cushion=(0, 2, 3, 4, 6)),
    "nhl": dict(lead=(1, 2, 3), left_max=(3600, 1200, 600, 300, 120), left_min=(0, 30, 60, 120), cushion=(0, .5, 1, 1.5)),
    "mlb": dict(lead=(1, 2, 3, 4, 5), left_max=(27, 9, 6, 3), left_min=(0, 1, 2), cushion=(0, .34, .5, 1)),
}
ASK = ((90, 99), (93, 99), (95, 99), (96, 99), (97, 99), (98, 99), (90, 96), (93, 97), (95, 98), (96, 98), (97, 98))
MIN_TRADES = 40


def cushion(sport, r):
    if sport == "mlb":
        return r["lead"]/max(r["left"], 1)
    return r["lead"]/((r["left"]/60+.5)**.5)


def entries(sport, games, rule):
    """First qualifying minute of each game."""
    lo, hi, lead, left_max, left_min, cush = rule
    out = []
    for rows in games:
        for r in rows:
            if (lo <= r["ask"] <= hi and r["lead"] >= lead and left_min <= r["left"] <= left_max
                    and (not cush or r["cushion"] >= cush)):
                out.append(r)
                break
    return out


def score(hits, games):
    if not hits:
        return dict(trades=0, total_cents=0.0)
    price = mean(h["ask"] for h in hits)/100
    return dict(trades=len(hits), losses=sum(not h["won"] for h in hits), win_rate=mean(h["won"] for h in hits),
                needed=price+.07*price*(1-price), mean_ask=100*price, net_cents=mean(h["net"] for h in hits),
                total_cents=sum(h["net"] for h in hits), per_game_cents=sum(h["net"] for h in hits)/games)


def interval(hits, draws=2000):
    if len(hits) < 10:
        return None
    rng, nets = random.Random(5), [h["net"] for h in hits]
    means = sorted(mean(rng.choices(nets, k=len(nets))) for _ in range(draws))
    return [means[int(.025*draws)], means[int(.975*draws)]]


def describe(sport, rule):
    lo, hi, lead, left_max, left_min, cush = rule
    unit = "outs" if sport == "mlb" else "s"
    text = f"ask {lo}-{hi}c, lead >= {lead}, {left_min}-{left_max} {unit} left"
    return text+(f", cushion >= {cush}" if cush else "")


def search(sport):
    rows = json.loads((DATA/f"{sport}-minutes.json").read_text())
    by_game = {}
    for r in sorted(rows, key=lambda r: r["ts"]):
        r["cushion"] = cushion(sport, r)
        by_game.setdefault(r["event"], []).append(r)
    games = sorted(by_game.values(), key=lambda g: g[0]["ts"])
    cut = int(len(games)*.6)
    early, late = games[:cut], games[cut:]
    g = GRID[sport]
    rules = [(lo, hi, *rest) for (lo, hi), *rest in product(ASK, g["lead"], g["left_max"], g["left_min"], g["cushion"])
             if rest[1] > rest[2]]
    scored = []
    for rule in rules:
        s = score(entries(sport, early, rule), len(early))
        if s["trades"] >= MIN_TRADES:
            scored.append((s["total_cents"], rule, s))
    scored.sort(key=lambda x: -x[0])
    plain = (97, 99, 1, g["left_max"][0], 0, 0)
    report = dict(sport=sport, games=dict(selection=len(early), later=len(late)),
                  split_date=late[0][0]["day"] if late else None, rules_tried=len(rules), top=[])

    def line(rule):
        a, hits = score(entries(sport, early, rule), len(early)), entries(sport, late, rule)
        b = score(hits, len(late))
        return dict(rule=describe(sport, rule), selection=a, later=b, later_ci95=interval(hits))
    report["plain_price_rule"] = line(plain)
    report["top"] = [line(rule) for _, rule, _ in scored[:10]]
    # A more cautious pick: stay at 96c and above, and rank by a lower bound on profit per contract
    # (mean minus two standard errors) so a rule cannot win on a handful of lucky trades.
    def bound(rule):
        nets = [h["net"] for h in entries(sport, early, rule)]
        return mean(nets)-2*pstdev(nets)/len(nets)**.5 if len(nets) >= MIN_TRADES else -99
    high = sorted((r for r in rules if r[0] >= 96 and r[1] == 99), key=bound, reverse=True)
    distinct, seen = [], set()
    for rule in high:
        key = tuple(h["ts"] for h in entries(sport, early, rule))
        if key not in seen:
            seen.add(key)
            distinct.append(rule)
        if len(distinct) == 10:
            break
    report["cautious"] = [line(rule) for rule in distinct]
    later = [t["later"] for t in report["top"] if t["later"]["trades"]]
    report["top10_later_average"] = dict(net_cents=mean(t["net_cents"] for t in later) if later else None,
                                         per_game_cents=mean(t["per_game_cents"] for t in later) if later else None)
    # The rule that would have been best on the later games, shown only to expose how far selection can mislead.
    best_later = max(rules, key=lambda rule: score(entries(sport, late, rule), len(late))["total_cents"])
    report["best_in_hindsight_on_later_games"] = line(best_later)
    return report


def show(report):
    print(f'\n{report["sport"].upper()}: {report["games"]["selection"]} games choose, {report["games"]["later"]} games '
          f'score (from {report["split_date"]}); {report["rules_tried"]} rules tried')
    def row(tag, t):
        a, b = t["selection"], t["later"]
        if not b["trades"] or not a["trades"]:
            print(f'  {tag:8} {t["rule"]}: no trades')
            return
        ci = t["later_ci95"] and [round(v, 2) for v in t["later_ci95"]]
        print(f'  {tag:8} {t["rule"]}\n           chosen on: n {a["trades"]:4} losses {a["losses"]:3} net {a["net_cents"]:+.2f}c '
              f'per game {a["per_game_cents"]:+.2f}c | later: n {b["trades"]:4} losses {b["losses"]:3} win {b["win_rate"]:.4f} '
              f'need {b["needed"]:.4f} net {b["net_cents"]:+.2f}c ci {ci} per game {b["per_game_cents"]:+.2f}c')
    row("plain", report["plain_price_rule"])
    for i, t in enumerate(report["top"][:5], 1):
        row(f"pick {i}", t)
    avg = report["top10_later_average"]
    if avg["net_cents"] is not None:
        print(f'  ten best picks, later games on average: net {avg["net_cents"]:+.2f}c per contract, {avg["per_game_cents"]:+.2f}c per game')
    for i, t in enumerate(report["cautious"][:4], 1):
        row(f"safe {i}", t)
    row("hindsight", report["best_in_hindsight_on_later_games"])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sports", nargs="+", choices=tuple(GRID), default=list(GRID))
    args = parser.parse_args()
    for sport in args.sports:
        if not (DATA/f"{sport}-minutes.json").exists():
            print(f"\n{sport.upper()}: no joined data yet")
            continue
        report = search(sport)
        save(DATA/f"{sport}-rules.json", report)
        show(report)


if __name__ == "__main__":
    main()
