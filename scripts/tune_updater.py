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
from tennis.research import labeled_starts, opening_state
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

    # Get initial p_serve from GBDT (first point features)
    X_first = match_df[FEATURE_COLUMNS].values[:1].astype(np.float32)
    p_prior = float(np.clip(model.predict_proba(X_first)[0], 0.01, 0.99))

    updater = BayesianUpdater(p_serve_prior=p_prior, virtual_sample_size=vss, decay_rate=decay)

    required = {"server", "best_of", "points_p1", "points_p2"}
    if not required.issubset(match_df.columns):
        raise ValueError("Rebuild the feature parquet to obtain pre-point score snapshots.")
    results = {}
    for n_obs, (_, row) in enumerate(match_df.iterrows()):
        # Predict using the observed score and previous n_obs outcomes.
        if n_obs in CHECKPOINTS:
            state = opening_state(row, match_format)
            p1_serves = state.server == 0
            state.sets_p1 = int(row.sets_server if p1_serves else row.sets_returner)
            state.sets_p2 = int(row.sets_returner if p1_serves else row.sets_server)
            state.games_p1 = int(row.games_server if p1_serves else row.games_returner)
            state.games_p2 = int(row.games_returner if p1_serves else row.games_server)
            state.points_p1, state.points_p2 = int(row.points_p1), int(row.points_p2)
            state.is_tiebreak = bool(row.is_tiebreak)
            raw = match_win_prob(state, float(updater.posterior_mean))
            results[n_obs] = float(np.clip(raw, 1e-6, 1 - 1e-6))
        if n_obs >= CHECKPOINTS[-1]:
            break
        updater.update(bool(row.server_won))

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
