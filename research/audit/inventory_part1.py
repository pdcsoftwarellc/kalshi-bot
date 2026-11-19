"""Inventory audit, part 1: event-clustered bootstrap intervals from saved ledgers.

Read-only. Reads only files under data/processed/. No network, no reruns.
Stake = entry price (limit) per contract. "as-reported" keeps the ledger's
cent-rounded-up fee on one contract; "fee-at-size" replaces it with the exact
0.07*p*(1-p) per leg (what the per-contract fee tends to at size).
"""
import json
import random
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2] / "data/processed"
random.seed(20261119)
N = 4000


def load(p):
    return json.loads((ROOT / p).read_text())


def exact(p):
    return 0.07 * p * (1 - p)


def boot(rows):
    """rows: (cluster, time, pct_of_stake, stake). Pooled % per bet = sum(profit)/sum(stake)
    (equals mean cents per contract / average entry price), 95% bootstrap resampling clusters,
    and the two chronological halves. A 3-tuple row is treated as stake 1 (plain mean)."""
    if not rows:
        return "n=0"
    rows = [(r[0], r[1], r[2]*(r[3] if len(r) > 3 else 1)/100, (r[3] if len(r) > 3 else 1)) for r in rows]
    by = defaultdict(list)
    for c, t, pnl, st in rows:
        by[c].append((pnl, st))
    keys = list(by)
    ratio = lambda x: 100*sum(a for a, _ in x)/sum(b for _, b in x)
    mean = ratio([(p, s) for _, _, p, s in rows])
    if len(keys) > 1:
        sims = sorted(ratio([v for k in random.choices(keys, k=len(keys)) for v in by[k]]) for _ in range(N))
        ci = f"[{sims[int(.025*N)]:+.1f}, {sims[int(.975*N)-1]:+.1f}]"
    else:
        ci = "[n/a: 1 cluster]"
    srt = sorted(rows, key=lambda r: r[1])
    h = len(srt) // 2
    half = lambda x: f"{ratio([(p, s) for _, _, p, s in x]):+.1f}% (n={len(x)})" if x else "n/a"
    wins = sum(1 for r in rows if r[2] > 0)
    cents = 100*sum(r[2] for r in rows)/len(rows)
    return f"n={len(rows)} events={len(keys)} pooled={mean:+.2f}% ({cents:+.2f}c/bet) CI95={ci} halves: {half(srt[:h])} | {half(srt[h:])} wins={wins}"


def show(name, rows):
    print(f"{name:58s} {boot(rows)}")


F = float

print("== short_rest (hold to settlement, taker at ask)")
fills, orders = [], []
for p in ("2026-10-01", "2026-11-01"):
    j = load(f"parallel-research/short_rest/{p}-ledger.json")
    fills += j["fills"]
    orders += j["orders"]
show("fills as-reported", [(r["fixture"], r["ts"], 100*F(r["pnl"])/F(r["limit"]), F(r["limit"])) for r in fills])
show("fills fee-at-size", [(r["fixture"], r["ts"], 100*(F(r["payout"])-F(r["limit"])-exact(F(r["limit"])))/F(r["limit"]), F(r["limit"])) for r in fills])
show("all quoted orders, fee-at-size", [(r["fixture"], r["ts"], 100*(F(r["payout"])-F(r["limit"])-exact(F(r["limit"])))/F(r["limit"]), F(r["limit"])) for r in orders if r.get("payout") is not None])
print("  avg entry", sum(F(r["limit"]) for r in fills)/len(fills))

print("== favorite_rebound (round trip, taker both legs)")
rows = []
for p in ("2026-10-01", "2026-11-01"):
    rows += [r for r in load(f"parallel-research/favorite_rebound/{p}.json")["ledger"] if r["status"] == "sold"]
show("round trips as-reported", [(r["fixture"], r["ts"], 100*F(r["pnl"])/F(r["limit"]), F(r["limit"])) for r in rows])
show("round trips fee-at-size", [(r["fixture"], r["ts"], 100*(F(r["gross"])-exact(F(r["limit"]))-exact(F(r["exit_limit"])))/F(r["limit"]), F(r["limit"])) for r in rows])
show("round trips gross (no fee)", [(r["fixture"], r["ts"], 100*F(r["gross"])/F(r["limit"]), F(r["limit"])) for r in rows])
print("  avg entry", sum(F(r["limit"]) for r in rows)/len(rows), "sum pnl", sum(F(r["pnl"]) for r in rows), "sum fees", sum(F(r["fees"]) for r in rows),
      "exact fees", sum(exact(F(r["limit"]))+exact(F(r["exit_limit"])) for r in rows))

print("== wta_tail (hold to settlement)")
fills, orders = [], []
for p in ("2026-09-19", "2026-10-01", "2026-11-01"):
    j = load(f"parallel-round2/wta_tail/{p}-ledger.json")
    fills += j["fills"]
    orders += j["orders"]
show("fills as-reported", [(r["fixture"], r["ts"], 100*F(r["pnl"])/F(r["limit"]), F(r["limit"])) for r in fills])
show("fills fee-at-size", [(r["fixture"], r["ts"], 100*(F(r["payout"])-F(r["limit"])-exact(F(r["limit"])))/F(r["limit"]), F(r["limit"])) for r in fills])
show("all quoted orders, fee-at-size", [(r["fixture"], r["ts"], 100*(F(r["payout"])-F(r["limit"])-exact(F(r["limit"])))/F(r["limit"]), F(r["limit"])) for r in orders if r.get("payout") is not None])
show("all quoted orders at BID, zero fee (hypothetical maker)", [(r["fixture"], r["ts"], 100*(F(r["payout"])-F(r["bid"]))/F(r["bid"]), F(r["bid"])) for r in orders if r.get("payout") is not None])
print("  avg entry", sum(F(r["limit"]) for r in fills)/len(fills), "orders", len(orders), "with payout", sum(r.get("payout") is not None for r in orders))

print("== trade_burst (round trip, taker both legs)")
rows = []
for p in ("2026-09-19", "2026-10-01", "2026-11-01"):
    rows += [r for r in load(f"parallel-round2/trade_burst/{p}-ledger.json") if r["status"] in ("sold", "settled")]
def tb_exact(r):
    f = exact(F(r["limit"])) + (exact(F(r["exit_limit"])) if r["status"] == "sold" else 0)
    return 100*(F(r["gross"])-f)/F(r["limit"])
show("positions as-reported", [(r["fixture"], r["ts"], 100*F(r["pnl"])/F(r["limit"]), F(r["limit"])) for r in rows])
show("positions fee-at-size", [(r["fixture"], r["ts"], tb_exact(r), F(r["limit"])) for r in rows])
show("positions gross (no fee)", [(r["fixture"], r["ts"], 100*F(r["gross"])/F(r["limit"]), F(r["limit"])) for r in rows])
print("  avg entry", sum(F(r["limit"]) for r in rows)/len(rows), "sum pnl", sum(F(r["pnl"]) for r in rows), "sum fees", sum(F(r["fees"]) for r in rows))

print("== depth_pressure (5 round trips)")
rows = load("parallel-round2/depth_pressure/challenger/summary.json")["ledger"]
show("as-reported", [(r["fixture"], r["entry_utc_ns"], 100*F(r["pnl"])/F(r["buy"]), F(r["buy"])) for r in rows])
show("fee-at-size", [(r["fixture"], r["entry_utc_ns"], 100*(F(r["gross"])-exact(F(r["buy"]))-exact(F(r["sell"])))/F(r["buy"]), F(r["buy"])) for r in rows])
print("  avg entry", sum(F(r["buy"]) for r in rows)/len(rows))

print("== player_routing")
for arm in ("control", "routed"):
    rows = []
    for p in ("2026-09-22", "2026-10-01", "2026-11-01"):
        rows += [r for r in load(f"parallel-round3/player_routing/{p}-ledger.json")[arm] if r["decision"] == "paper_fill"]
    show(f"{arm} fills as-reported", [(r["fixture"], r["ts"], 100*F(r["pnl"])/F(r["limit"]), F(r["limit"])) for r in rows])
    show(f"{arm} fills fee-at-size", [(r["fixture"], r["ts"], 100*(F(r["payout"])-F(r["limit"])-exact(F(r["limit"])))/F(r["limit"]), F(r["limit"])) for r in rows])
    print("  avg entry", sum(F(r["limit"]) for r in rows)/len(rows))

print("== api_cost (Elo vs market, hold to settlement)")
fills, orders, preds = [], [], []
for p in ("2026-10-01", "2026-11-01"):
    j = load(f"api-cost-research/{p}-ledger.json")
    fills += j["fills"]
    orders += j["orders"]
    preds += j["predictions"]
show("fills as-reported", [(r["fixture"], r["ts"], 100*F(r["pnl"])/F(r["limit"]), F(r["limit"])) for r in fills])
show("all orders at limit, as-reported fee", [(r["fixture"], r["ts"], 100*(F(r["payout"])-F(r["limit"])-F(r["fee"]))/F(r["limit"]), F(r["limit"])) for r in orders if r.get("payout") is not None])
show("all orders at limit, fee-at-size", [(r["fixture"], r["ts"], 100*(F(r["payout"])-F(r["limit"])-exact(F(r["limit"])))/F(r["limit"]), F(r["limit"])) for r in orders if r.get("payout") is not None])
print("  orders", len(orders), "avg limit", sum(F(r["limit"]) for r in orders)/len(orders))

print("== set_frequency (native model at set boundaries)")
led = []
for p in ("2026-10-01", "2026-11-01"):
    led += load(f"native-atp/set-frequency/{p}.json")["ledger"]
fl = [r for r in led if r["decision"] == "paper_fill"]
sig = [r for r in led if r["decision"] in ("paper_fill", "unfilled")]
show("fills as-reported", [(r["fixture"], r["t"], 100*F(r["pnl"])/F(r["limit"]), F(r["limit"])) for r in fl])
show("fills fee-at-size", [(r["fixture"], r["t"], 100*(F(r["pnl"])+F(r["fee"])-exact(F(r["limit"])))/F(r["limit"]), F(r["limit"])) for r in fl])
def settle_pnl(r, fee):
    pay = F(r["settlement"]) if r["side"] == "yes" else 1-F(r["settlement"])
    return 100*(pay-F(r["limit"])-fee)/F(r["limit"])
show("all signals at limit, fee-at-size (YES-settlement inferred)", [(r["fixture"], r["t"], settle_pnl(r, exact(F(r["limit"]))), F(r["limit"])) for r in sig])
print("  check inferred payout on fills:", [(r["pnl"], round(settle_pnl(r, F(r["fee"]))*F(r["limit"])/100, 2)) for r in fl])
print("  avg entry fills", sum(F(r["limit"]) for r in fl)/len(fl))

print("== market_anchor (3 periods)")
for arm in ("native", "market_anchored"):
    led = []
    for p in ("2026-09-22", "2026-10-01", "2026-11-01"):
        led += load(f"native-atp/market-anchor/{p}-ledger.json")[arm]
    fl = [r for r in led if r["decision"] == "paper_fill"]
    sig = [r for r in led if r["decision"] in ("paper_fill", "unfilled")]
    show(f"{arm} fills as-reported", [(r["fixture"], r["t"], 100*F(r["pnl"])/F(r["limit"]), F(r["limit"])) for r in fl])
    show(f"{arm} all signals at limit, fee-at-size (inferred)", [(r["fixture"], r["t"], settle_pnl(r, exact(F(r["limit"]))), F(r["limit"])) for r in sig])
    print("  avg entry fills", sum(F(r["limit"]) for r in fl)/len(fl), "sum pnl", sum(F(r["pnl"]) for r in fl))

print("== score_feed_markout (30s round trip after point receipt)")
rows = [r for r in load("native-atp/score-feed-markout/point-windows.json") if r["status"] == "measured"]
show("windows as-reported", [(r["fixture"], r["ts_ms"], r["net_cents"]/r["entry"], r["entry"]) for r in rows])
show("windows fee-at-size", [(r["fixture"], r["ts_ms"], (r["gross_cents"]-100*exact(r["entry"])-100*exact(r["exit"]))/r["entry"], r["entry"]) for r in rows])
show("windows gross", [(r["fixture"], r["ts_ms"], r["gross_cents"]/r["entry"], r["entry"]) for r in rows])
print("  avg entry", sum(r["entry"] for r in rows)/len(rows))

print("== fifteen_forty_tape")
rows = [r for r in load("fifteen-forty/tape.json")["trades"] if r["status"] == "closed"]
key = lambda r: (r["tape"], r["receipt_s"])
show("trades as-reported", [(r["fixture"], key(r), 100*r["net_cents"]/r["entry_ask"], r["entry_ask"]/100) for r in rows])
show("trades fee-at-size", [(r["fixture"], key(r), 100*(r["gross_cents"]-100*exact(r["entry_ask"]/100)-100*exact(r["exit_bid"]/100))/r["entry_ask"], r["entry_ask"]/100) for r in rows])
show("trades gross", [(r["fixture"], key(r), 100*r["gross_cents"]/r["entry_ask"], r["entry_ask"]/100) for r in rows])
show("mid-to-mid (no spread, no fee)", [(r["fixture"], key(r), 100*r["mid_to_mid_cents"]/r["entry_ask"], r["entry_ask"]/100) for r in rows])
print("  avg entry", sum(r["entry_ask"] for r in rows)/len(rows), "tapes", sorted({r["tape"] for r in rows}))

print("== cross_venue (Polymarket-gap rule; exact fee already)")
tr = load("cross-venue/trades.json")
groups = defaultdict(list)
for r in tr:
    groups[(r["sport"], r["x"], r["exit_rule"])].append(r)
    groups[("all", r["x"], r["exit_rule"])].append(r)
for k in sorted(groups):
    g = groups[k]
    show(f"{k} avg entry {sum(r['price'] for r in g)/len(g):.1f}c", [(r["event"], r["settle"], 100*r["net_cents"]/r["price"], r["price"]/100) for r in g])

print("== model vs market Brier (bootstrap by match): model_brier - market_brier")
def brier(name, rows, mk, pk, ok, ck):
    by = [((r[mk]-r[ok])**2-(r[pk]-r[ok])**2) for r in rows]
    m = sum(by)/len(by)
    sims = sorted(sum(random.choices(by, k=len(by)))/len(by) for _ in range(N))
    print(f"{name:58s} n={len(by)} diff={m:+.4f} CI95=[{sims[int(.025*N)]:+.4f}, {sims[int(.975*N)-1]:+.4f}]")
brier("model_market_audit (wimbledon-smoke)", load("model-market-audit/forecasts.json"), "model_yes", "market_mid", "outcome", "fixture")
brier("native-atp market-audit", load("native-atp/market-audit/forecasts.json"), "model_yes", "market_mid", "outcome", "fixture")
brier("api_cost Elo", preds, "p", "mid", "outcome", "fixture")

print("== break_hold: favourite win rate minus prematch price, by first-break-point result")
g = load("parallel-round2/break_hold/historical-games.json")
for res in ("hold", "break"):
    rows = [r for r in g if r["result"] == res and r.get("won") is not None]
    # hypothetical: buy favourite at the PREMATCH mid, zero fee (not executable after the game)
    show(f"{res}: (won - prior)/prior, hypothetical", [(r["fixture"], r["prematch_quote_ts"], 100*((1 if r["won"] else 0)-F(r["prior"]))/F(r["prior"]), F(r["prior"])) for r in rows])

print("== cross_venue by pre-game / in-play at signal (all sports)")
groups = defaultdict(list)
for r in tr:
    groups[("live" if r["live"] else "pre", r["x"], r["exit_rule"])].append(r)
for k in sorted(groups):
    g = groups[k]
    show(f"{k} avg entry {sum(r['price'] for r in g)/len(g):.1f}c", [(r["event"], r["settle"], 100*r["net_cents"]/r["price"], r["price"]/100) for r in g])
