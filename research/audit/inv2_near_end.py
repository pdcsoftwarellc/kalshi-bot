"""Event-clustered intervals and halves for near_end saved trades (taker, next-minute open ask, held)."""
import json, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
from inv2_common import boot, halves, fmt
ROOT = Path(__file__).resolve().parents[2]/"data/processed/near-end"
GROUP = {"KXATPMATCH": "tennis", "KXATPCHALLENGERMATCH": "tennis", "KXWTAMATCH": "tennis", "KXNFLGAME": "nfl", "KXNBAGAME": "nba",
         "KXNHLGAME": "nhl", "KXMLBGAME": "mlb", "KXNCAAFGAME": "ncaaf", "KXNCAAMBGAME": "ncaamb"}
pct = lambda r: 100*r["net_cents"]/r["price"]
for name in ("main-trades", "holdout-trades", "trades"):
    rows = json.load(open(ROOT/f"{name}.json"))
    print("==", name, len(rows))
    keys = sorted({(GROUP.get(r["series"], r["series"]), r["series"]) for r in rows})
    for grp in sorted({g for g, _ in keys}) + sorted({s for g, s in keys if g == "tennis"}):
        for t in (95, 97, 98, 99):
            for lab in ("all", "risen"):
                hit = [r for r in rows if (GROUP.get(r["series"]) == grp or r["series"] == grp) and r["threshold"] == t and (lab == "all" or r["risen"])]
                if not hit:
                    continue
                b = boot(hit, lambda r: r["event"], pct)
                c = boot(hit, lambda r: r["event"], lambda r: r["net_cents"], draws=500)
                h = halves(hit, lambda r: r["event"], lambda r: r["ts"], pct) if "ts" in hit[0] else None
                price = sum(r["price"] for r in hit)/len(hit)
                loss = sum(r["payout"] < .5 for r in hit)
                print(f'{grp:22} >={t} {lab:5} price {price:.1f} losses {loss:3} cents {c["mean"]:+.2f} | {fmt(b, h)}')
