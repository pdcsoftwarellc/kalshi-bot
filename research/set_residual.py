"""Test incremental first-set point information relative to the observed match market."""
from collections import Counter
from datetime import date, timedelta
from decimal import Decimal as D
from heapq import heappop, heappush
import json
from pathlib import Path
import sys

import httpx
import numpy as np
from scipy.optimize import brentq
from scipy.special import expit, logit

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from research.api_cost import ROOT, fee, pages, save, timestamp
from research.set_frequency import OUT as CHECKPOINT_DIR, initial_state, print_hit
from tennis.live_score import observed_points

OUT = ROOT / "data/processed/parallel-round3/set_residual"
START = (date(2026, 10, 1), date(2026, 11, 1))
END = (date(2026, 11, 1), date(2026, 11, 16))
MIN_NET = D(".05")
RULE = {
    "observations": "First publicly confirmed set only. One per fixture. 7-5 and 7-6 constitute the predefined close-set subgroup.",
    "feature": "Difference between YES player's and opponent's first-set serve-win fractions, each with 20 pseudo-points at 0.63.",
    "fit": "Minimize October summed binary match log loss plus 0.5 * beta^2 with logit(market midpoint) offset, no intercept, beta >= 0. Train only on results published before November 1 UTC.",
    "trade": "One contract if either side offers at least $0.05 after cent-rounded entry fee. $25 starting cash each period. Same-side non-block print of >=1 at limit or better from t+2s through t+5s exclusive; hold to exchange settlement.",
}


def first_set_feature(fixture, yes_player, expected_winner):
    """Return a signed serve-strength difference using only the reported first-set prefix."""
    first = [g for g in fixture["pointbypoint"] if g.get("set_number") in {"Set 1", "Set 1 TieBreak"}]
    best_of = initial_state(fixture, 0).match_format.best_of
    points = observed_points(first, best_of)
    if not points or points[-1].sets[expected_winner] != 1 or sum(points[-1].sets) != 1 or points[-1].games != (0, 0):
        raise ValueError("Score prefix disagrees with the published set winner")
    won = [0, 0]
    played = [0, 0]
    for pt in points:
        played[pt.served_by] += 1
        won[pt.served_by] += int(pt.won)
    means = [(w + 20*.63)/(n + 20) for w, n in zip(won, played)]
    last_game = max((g for g in first if g.get("set_number") == "Set 1"),
                    key=lambda g: int(g["number_game"]))
    set_score = tuple(int(x.strip()) for x in last_game["score"].split("-"))
    close = tuple(sorted(set_score)) in {(5, 7), (6, 7)}
    return dict(x=means[yes_player]-means[1-yes_player], close=close, first_set_score=set_score,
                yes_serve_won=won[yes_player], yes_served=played[yes_player],
                opp_serve_won=won[1-yes_player], opp_served=played[1-yes_player])


def market_mid(row):
    return (row["ask"] + row["bid"])/200


def fit(rows):
    """Nonnegative log-odds slope with unit ridge and an exchange-implied offset."""
    if not rows:
        raise ValueError("Empty calibration set")
    x = np.array([r["x"] for r in rows])
    y = np.array([float(r["settlement"]) for r in rows])
    offset = logit([market_mid(r) for r in rows])
    def gradient(beta):
        return beta + x @ (expit(offset+beta*x)-y)
    if gradient(0.) >= 0:
        return 0.
    return float(brentq(gradient, 0., 1. + sum(abs(x))))


def assemble(start, end):
    upstream = json.loads((CHECKPOINT_DIR/f"{start}.json").read_text())
    fixtures = {}
    day = start - timedelta(days=2)
    while day <= end:
        for directory in ("native-atp", "set-frequency-atp"):
            path = ROOT / "data/raw" / directory / f"{day}.json"
            if path.is_file():
                fixtures.update({str(f["event_key"]):f for f in json.loads(path.read_text())})
                break
        day += timedelta(days=1)
    result, excluded = [], Counter()
    seen = set()
    for record in upstream["ledger"]:
        if record["set_number"] != 1:
            continue
        fid = record["fixture"]
        if fid in seen:
            raise ValueError("Overlapping first-set observations for one fixture")
        seen.add(fid)
        if not record["confirmed"] <= record["t"]:
            raise ValueError("Quote precedes confirmation")
        row = {key:record[key] for key in ("fixture", "ticker", "t", "confirmed", "yes_player",
                                          "first_set_winner", "bid", "ask", "settled", "settlement")}
        try:
            row.update(first_set_feature(fixtures[fid], row["yes_player"], row["first_set_winner"]))
        except (KeyError, TypeError, ValueError, IndexError):
            excluded["incomplete_or_mismatched_prefix"] += 1
            continue
        if row["settlement"] is None or not row["settled"] or timestamp(row["settled"]) <= row["t"]+5:
            excluded["missing_or_early_payout"] += 1
            continue
        result.append(row)
    return result, dict(first_set_quoted=len(seen), exclusions=dict(excluded))


def pick(yes_probability, bid, ask):
    """A unit taker buy, priced according to the chosen side and entry limit."""
    if not 0 < bid <= ask < 100:
        raise ValueError("Unusable whole-cent book")
    p = D(str(yes_probability))
    sides = [("yes", p, D(ask)/100), ("no", 1-p, 1-D(bid)/100)]
    side, prob, limit = max(sides, key=lambda s:s[1]-s[2]-fee(s[2]))
    net_edge = prob-limit-fee(limit)
    return (side, limit, net_edge) if net_edge >= MIN_NET else None


def metrics(rows):
    binary = [r for r in rows if D(r["settlement"]) in (0, 1)]
    if not binary:
        return {"n":0}
    y = np.array([float(r["settlement"]) for r in binary])
    # Positive directional score means a small tilt toward the higher service
    # fraction would locally improve average binary log loss.
    directional = np.array([r["x"] for r in binary]) * (y - np.array([market_mid(r) for r in binary]))
    scores = {"n":len(y), "directional_score_at_zero":float(directional.mean())}
    losses = []
    for label, p in (("market", np.array([market_mid(r) for r in binary])),
                     ("residual", np.array([r["model_yes"] for r in binary]))):
        losses.append((y-p)**2)
        scores[label+"_brier"] = float(np.mean(losses[-1]))
        scores[label+"_log_loss"] = float(np.mean(-(y*np.log(p)+(1-y)*np.log1p(-p))))
    delta = losses[1]-losses[0]
    rng = np.random.default_rng(42)
    bootstrap = rng.choice(delta, (4000,len(delta)), replace=True).mean(axis=1)
    scores["residual_minus_market_brier"] = float(delta.mean())
    scores["fixture_resampled_95pct_brier_difference"] = np.quantile(bootstrap,[.025,.975]).tolist()
    return scores


def evaluate(client, rows, beta, start, end):
    counts = Counter()
    cash, pnl, fees, all_fill, max_gain = D(25), D(0), D(0), D(0), D(0)
    released = []
    for r in sorted(rows,key=lambda v:(v["t"], v["ticker"])):
        while released and released[0][0] <= r["t"]:
            cash += heappop(released)[1]
        r["model_yes"] = market_mid(r) if beta == 0 else float(expit(logit(market_mid(r))+beta*r["x"]))
        trade = pick(r["model_yes"], r["bid"], r["ask"])
        if trade is None:
            continue
        counts["signals"] += 1
        side, limit, edge = trade
        cost = limit+fee(limit)
        payout = D(r["settlement"]) if side=="yes" else 1-D(r["settlement"])
        all_fill += payout-cost
        max_gain += 1-cost
        r.update(side=side, limit=str(limit), net_edge=str(edge))
        if cost > cash:
            counts["cash_blocked"] += 1
            continue
        cash -= cost
        trades = pages(client, "/markets/trades","trades", ticker=r["ticker"], limit=1000,
                       min_ts=r["t"]+2, max_ts=r["t"]+5)
        hit = print_hit(trades, side, limit, r["t"],5)
        if hit is None:
            heappush(released,(r["t"]+5,cost))
            continue
        heappush(released,(timestamp(r["settled"]),payout))
        counts["fills"] += 1
        counts["winning_fills"] += int(payout>cost)
        r.update(proof=hit,pnl=str(payout-cost))
        pnl += payout-cost
        fees += fee(limit)

    return dict(fixtures=len(rows), metrics=metrics(rows), close_set_metrics=metrics([r for r in rows if r["close"]]),
                signals=counts["signals"], fills=counts["fills"], winning_fills=counts["winning_fills"],
                cash_blocked=counts["cash_blocked"], fees=str(fees), pnl=str(pnl),
                all_fill_scenario_before_cap=str(all_fill), all_fill_all_win_ceiling=str(max_gain),
                close_set_signals=sum("side" in r for r in rows if r["close"]),
                close_set_fills=sum("pnl" in r for r in rows if r["close"]),
                close_set_pnl=str(sum((D(r["pnl"]) for r in rows if r["close"] and "pnl" in r), D(0))),
                allocated_api_dollars="80" if start.month==10 else "40",
                pnl_after_api=str(pnl - (D(80) if start.month==10 else D(40))))


def main():
    oct_rows, oct_coverage = assemble(*START)
    nov_rows, nov_coverage = assemble(*END)
    if {r["fixture"] for r in oct_rows} & {r["fixture"] for r in nov_rows}:
        raise ValueError("Match appears in both fit and forward evaluation")
    fit_rows = [r for r in oct_rows if D(r["settlement"]) in (0,1)
                and timestamp(r["settled"]) < timestamp("2026-11-01T00:00:00Z")]
    beta = fit(fit_rows)
    result = dict(rule=RULE, fit_matches=len(fit_rows), ridge_coefficient=beta, periods=[])
    with httpx.Client(timeout=40) as c:
        for rows, coverage, dates in ((oct_rows,oct_coverage,START), (nov_rows,nov_coverage,END)):
            summary = dict(start=str(dates[0]), end_exclusive=str(dates[1]),coverage=coverage,
                           **evaluate(c,rows,beta,*dates))
            result["periods"].append(summary)
            save(OUT/f"{dates[0]}-ledger.json",dict(summary=summary,rows=rows))
    save(OUT/"summary.json",result)
    print(json.dumps(result,indent=2))


if __name__=="__main__":
    main()
