"""Offline opening forecast benchmark on exactly the archived, price-qualified ATP cohort."""
import argparse
import hashlib
import json
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from research.api_cost import COLLECTION, ROOT, same_player, save, timestamp
from tennis.engine import PredictionEngine
from tennis.markov import game_win_prob_from_score
from tennis.types import MatchState, MatchFormat

ARTIFACTS = ROOT / "data/artifacts/wimbledon-smoke"
COHORT = ROOT / "data/processed/parallel-research/price_drift"
OUT = ROOT / "data/processed/model-market-audit"


def opening_forecast(engine, fixture):
    # Service order is unknown at this market observation. Use common random
    # draws in both scenarios to reduce Monte Carlo noise in their average.
    seed = int.from_bytes(hashlib.sha256(str(fixture["event_key"]).encode()).digest()[:8], "big")
    estimates = []
    for server in (0, 1):
        state = MatchState(player1_id=str(fixture["first_player_key"]),
                           player2_id=str(fixture["second_player_key"]),
                           match_date=fixture["event_date"], server=server,
                           match_format=MatchFormat.best_of_3_tiebreak(),
                           surface="Unknown", tournament_level="O")
        engine._rng = np.random.default_rng(seed)
        engine.new_match(state)
        estimates.append(engine.predict(state).win_prob_p1)
    return sum(estimates) / 2


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifacts-dir", type=Path, default=ARTIFACTS)
    parser.add_argument("--out-dir", type=Path, default=OUT)
    args = parser.parse_args()
    engine = PredictionEngine.load(str(args.artifacts_dir), n_mc_samples=200)
    markets = {}
    for path in COLLECTION.glob("*.json"):
        response = json.loads(path.read_text())
        if isinstance(response, dict):
            markets.update({m["ticker"]: m for m in response.get("markets", [])
                            if "ticker" in m and "yes_sub_title" in m and "settlement_ts" in m})

    fixtures = {}
    for path in (ROOT / "data/raw/short-rest-v1").glob("ATP-*.json"):
        fixtures.update({str(f["event_key"]): f for f in json.loads(path.read_text())})

    observations = []
    counts = {}
    for path in sorted(COHORT.glob("*-ledger.json")):
        eligible = json.loads(path.read_text())["eligible_observations"]
        counts[path.name[:10]] = len(eligible)
        for q in eligible:
            f = fixtures.get(str(q["fixture"]))
            m = markets.get(q["ticker"])
            if not m or not f or f["event_status"] != "Finished":
                continue
            orientation = [same_player(f[key], m["yes_sub_title"])
                           for key in ("event_first_player", "event_second_player")]
            if sum(orientation) != 1 or f["event_winner"] not in ("First Player", "Second Player"):
                continue
            yes_is_first = orientation[0]
            y = float((f["event_winner"] == "First Player") == yes_is_first)
            if m.get("settlement_value_dollars") is None or float(m["settlement_value_dollars"]) != y:
                continue
            if not q["late_quote_ts"] <= q["ts"] < min(q["provider_start"], q["kalshi_start"], timestamp(m["settlement_ts"])):
                raise ValueError("Non-prematch observation")
            p1 = opening_forecast(engine, f)
            observations.append(dict(period=path.name[:10], fixture=q["fixture"], ticker=q["ticker"],
                                     model_yes=p1 if yes_is_first else 1-p1,
                                     market_mid=(float(q["late_bid"])+float(q["late_ask"]))/2,
                                     outcome=y, decision_ts=q["ts"]))
    if not observations:
        raise ValueError("No matched forecasts; run the price-drift and short-rest data workflows first.")
    if len({r["fixture"] for r in observations}) != len(observations):
        raise ValueError("Duplicated match")

    def brier(rows, key):
        return sum((r[key] - r["outcome"])**2 for r in rows)/len(rows)

    groups = {"combined": observations}
    groups.update({k: [r for r in observations if r["period"] == k] for k in counts})
    # At advantage the server can either win the next point outright or lose
    # it and return to deuce. Record both sides of this consistency check.
    point_p = .65
    deuce = game_win_prob_from_score(point_p, 3, 3)
    advantage_check = dict(point_probability=point_p, from_deuce=deuce,
                           reported_from_server_advantage=game_win_prob_from_score(point_p, 4, 3),
                           expected_from_server_advantage=point_p + (1-point_p)*deuce)
    summary = dict(game_probability_check=advantage_check, cohort_counts=counts, model=args.artifacts_dir.name, monte_carlo_samples=200,
                   calibrated_player1_range=[engine._calibrator.calibrate(p) for p in (0.,1.)],
                   evaluation="Prematch forecast at cached two-sided quotes; no order or fill simulation.",
                   assumptions="Equal weight for each first server; best of three, unknown surface and other tournament level; frozen model and as-of player history.",
                   periods={k: dict(matches=len(v), model_brier=brier(v,"model_yes"),
                                    market_brier=brier(v,"market_mid")) for k,v in groups.items() if v})
    save(args.out_dir/"forecasts.json", observations)
    save(args.out_dir/"summary.json", summary)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
