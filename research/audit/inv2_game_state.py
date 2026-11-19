"""Plain 97-99c rule and the MLB lead-and-outs rule from saved game-state minutes (15 s feed delay, taker)."""
import json, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
from inv2_common import boot, halves, fmt
D = Path(__file__).resolve().parents[2]/"data/processed/game-state"
LEFT = dict(nba=2880, nfl=3600, nhl=3600, mlb=27, ncaamb=2400)


def first(games, ok):
    out = []
    for rows in games:
        for r in rows:
            if ok(r):
                out.append(r)
                break
    return out


def show(tag, hits):
    if not hits:
        print(f"  {tag}: no trades")
        return
    pct = lambda r: 100*r["net"]/r["ask"]
    ev = lambda r: r["event"]
    b = boot(hits, ev, pct)
    c = boot(hits, ev, lambda r: r["net"], draws=1000)
    h = halves(hits, ev, lambda r: r["ts"], pct)
    ask = sum(r["ask"] for r in hits)/len(hits)
    print(f'  {tag}: ask {ask:.1f} losses {sum(not r["won"] for r in hits)} cents {c["mean"]:+.2f} [{c["lo"]:+.2f},{c["hi"]:+.2f}] | {fmt(b, h)}')


for sport in LEFT:
    rows = json.loads((D/f"{sport}-minutes.json").read_text())
    by = {}
    for r in sorted(rows, key=lambda r: r["ts"]):
        by.setdefault(r["event"], []).append(r)
    games = sorted(by.values(), key=lambda g: g[0]["ts"])
    cut = int(len(games)*.6)
    print(f"== {sport}: {len(games)} games, split {cut}/{len(games)-cut}, {games[0][0]['day']} to {games[-1][0]['day']}")
    plain = lambda r: 97 <= r["ask"] <= 99 and r["lead"] >= 1 and 0 <= r["left"] <= LEFT[sport]
    show("plain 97-99 ALL games", first(games, plain))
    show("plain 97-99 earliest 60%", first(games[:cut], plain))
    show("plain 97-99 latest 40%", first(games[cut:], plain))
    if sport == "mlb":
        rule = lambda r: 90 <= r["ask"] <= 99 and r["lead"] >= 3 and r["left"] <= 27 and r["lead"]/max(r["left"], 1) >= 1
        show("lead>=3 & lead>=outs, 90-99 ALL", first(games, rule))
        show("lead>=3 & lead>=outs earliest 60%", first(games[:cut], rule))
        show("lead>=3 & lead>=outs latest 40%", first(games[cut:], rule))
        half = lambda g: [dict(r, net=100*(r["won"]-r["ask"]/100-.035*r["ask"]/100*(1-r["ask"]/100))) for r in g]
        show("same, ALL, fee multiplier 0.5", first([half(g) for g in games], rule))
        show("plain ALL, fee multiplier 0.5", first([half(g) for g in games], plain))
    if sport == "ncaamb":
        r2 = lambda r: 98 <= r["ask"] <= 99 and r["lead"] >= 12 and 60 <= r["left"] <= 300
        show("safe: 98-99 lead>=12 60-300s ALL", first(games, r2))
    if sport == "nhl":
        r3 = lambda r: 96 <= r["ask"] <= 98 and r["lead"] >= 3
        show("top: 96-98 lead>=3 ALL", first(games, r3))
        show("top: 96-98 lead>=3 latest 40%", first(games[cut:], r3))
