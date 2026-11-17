#!/usr/bin/env python3
"""Tune BayesianUpdater parameters for optimal point-to-point trading.

Grid searches over virtual_sample_size and decay_rate, evaluating
match-level log-loss at multiple checkpoints during each match.

Evaluates predictive accuracy at pre-point score snapshots.

Usage:
    python scripts/tune_updater.py [--artifacts-dir PATH] [--val-data PATH]
"""
import argparse
import itertools
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import log_loss

sys.path.insert(0, str(Path(__file__).parent.parent))

from tennis.features import FEATURE_COLUMNS
from tennis.research import labeled_starts, serve_priors, score_state
from tennis.engine import PredictionEngine
from tennis.calibration import MatchWinCalibrator
from tennis.model import PointWinModel
from tennis.markov import match_win_prob
from tennis.updater import BayesianUpdater
from tennis.types import MatchFormat

PROCESSED_DIR = Path(__file__).parent.parent / "data" / "processed"
ARTIFACTS_DIR = Path(__file__).parent.parent / "data" / "artifacts"

CHECKPOINTS = [10, 20, 40, 60, 80, 100]  # evaluate after this many points
VSS_VALUES = [5, 10, 15, 20, 30, 50]
DECAY_VALUES = [0.99, 0.995, 0.999, 1.0]


def simulate_match(
    model: PointWinModel,
    match_df: pd.DataFrame,
    match_format: MatchFormat,
    vss: int,
    decay: float,
) -> dict:
    """Simulate the engine pipeline on a single match.

    Returns dict mapping checkpoint → match_win_prob at that point count.
    """
    match_df = match_df.sort_values("point_idx")
    if len(match_df) < 5:
        return {}

    required = {"server", "best_of", "points_p1", "points_p2"}
    if not required.issubset(match_df.columns):
        raise ValueError("Rebuild the feature parquet to obtain pre-point score snapshots.")
    p1, p2 = serve_priors(model, match_df.iloc[:1])
    engine = PredictionEngine(None, MatchWinCalibrator(), None, None, n_mc_samples=200, rng_seed=42)
    engine.start_with_priors(p1[0], p2[0], int(match_df.iloc[0].server),
                             virtual_sample_size=vss, decay_rate=decay)
    results = {}
    for row in match_df.itertuples(index=False):
        if row.point_idx in CHECKPOINTS:
            result = engine.predict(score_state(row))
            results[row.point_idx] = float(np.clip(result.raw_win_prob_p1, 1e-6, 1-1e-6))
        if row.point_idx >= CHECKPOINTS[-1]:
            break
        engine.record_point(bool(row.server_won), int(row.server))
    return results


def main():
    parser = argparse.ArgumentParser(description="Tune BayesianUpdater parameters")
    parser.add_argument("--artifacts-dir", default=str(ARTIFACTS_DIR))
    parser.add_argument("--val-data", default=str(PROCESSED_DIR / "features_val.parquet"))
    parser.add_argument("--max-matches", type=int, default=200, help="Limit matches for speed")
    args = parser.parse_args()

    print("=== Loading artifacts ===")
    model = PointWinModel.load(os.path.join(args.artifacts_dir, "gbdt_model.pkl"))
    # Score raw probabilities; the calibrator was fitted on part of this validation data.

    print("=== Loading validation data ===")
    val_df = pd.read_parquet(args.val_data)
    val_df["match_date"] = pd.to_datetime(val_df["match_date"])
    print(f"  {len(val_df):,} points")

    if "match_id" not in val_df.columns:
        print("ERROR: match_id required")
        sys.exit(1)

    # Group by match, limit for speed
    match_groups = {mid: g for mid, g in val_df.groupby("match_id")}
    match_ids = list(match_groups.keys())[:args.max_matches]
    print(f"  Evaluating {len(match_ids)} matches")

    starts = labeled_starts(val_df)
    match_outcomes = starts.set_index("match_id").match_winner.eq(1).astype(int).to_dict()

    match_ids = [mid for mid in match_ids if mid in match_outcomes]
    match_format = MatchFormat.best_of_3_tiebreak()

    # Grid search
    print(f"\n=== Grid Search: {len(VSS_VALUES)} x {len(DECAY_VALUES)} = {len(VSS_VALUES)*len(DECAY_VALUES)} combinations ===")
    results = []

    for vss, decay in itertools.product(VSS_VALUES, DECAY_VALUES):
        checkpoint_probs = {cp: [] for cp in CHECKPOINTS}
        checkpoint_outcomes = {cp: [] for cp in CHECKPOINTS}

        for mid in match_ids:
            g = match_groups[mid]
            cp_results = simulate_match(model, g, match_format, vss, decay)
            outcome = match_outcomes[mid]
            for cp, prob in cp_results.items():
                checkpoint_probs[cp].append(prob)
                checkpoint_outcomes[cp].append(outcome)

        row = {"vss": vss, "decay": decay}
        total_ll = 0.0
        n_checkpoints = 0
        for cp in CHECKPOINTS:
            if len(checkpoint_probs[cp]) >= 10:
                ll = log_loss(checkpoint_outcomes[cp], checkpoint_probs[cp], labels=[0, 1])
                row[f"logloss_at_{cp}"] = ll
                total_ll += ll
                n_checkpoints += 1

        if not n_checkpoints:
            raise ValueError("Need at least ten labeled matches reaching a checkpoint.")
        row["avg_logloss"] = total_ll / n_checkpoints
        results.append(row)
        print(f"  vss={vss:3d}  decay={decay:.3f}  avg_logloss={row['avg_logloss']:.4f}")

    # Find best
    results_df = pd.DataFrame(results)
    best = results_df.loc[results_df["avg_logloss"].idxmin()]
    print(f"\n=== Best Parameters ===")
    print(f"  virtual_sample_size = {int(best['vss'])}")
    print(f"  decay_rate = {best['decay']}")
    print(f"  avg log-loss = {best['avg_logloss']:.4f}")

    # Save
    csv_path = os.path.join(args.artifacts_dir, "updater_tuning.csv")
    results_df.to_csv(csv_path, index=False)
    print(f"\n  Results saved: {csv_path}")


if __name__ == "__main__":
    main()
