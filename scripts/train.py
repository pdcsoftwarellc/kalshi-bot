#!/usr/bin/env python3
"""Train the GBDT point-win model and calibrator.

Steps:
1. Load train/val parquet feature files
2. Train XGBoost (and optionally LightGBM) with early stopping on val log-loss
3. Compute match-level win probability predictions on val set using the
   deployed pipeline (two serve priors → Bayesian posteriors → match probability)
4. Fit symmetric odds calibration on later validation score snapshots
5. Save artifacts: gbdt_model.pkl, calibrator.pkl

Usage:
    python scripts/train.py [--backend xgboost|lightgbm]
"""
import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))

from tennis.features import FEATURE_COLUMNS
from tennis.model import PointWinModel
from tennis.calibration import MatchWinCalibrator
from tennis.research import opening_predictions, checkpoint_predictions
from tennis.types import MatchFormat

PROCESSED_DIR = Path(__file__).parent.parent / "data" / "processed"
ARTIFACTS_DIR = Path(__file__).parent.parent / "data" / "artifacts"


def load_features(path: str) -> tuple[np.ndarray, np.ndarray, pd.DataFrame]:
    """Load feature parquet and return (X, y, meta_df)."""
    df = pd.read_parquet(path)
    X = df[FEATURE_COLUMNS].values.astype(np.float32)
    y = df["server_won"].values.astype(np.float32)
    meta = df[["match_date", "match_id", "point_idx"]].copy() if "match_id" in df.columns else df[["match_date"]].copy()
    return X, y, meta


def build_match_win_probs_on_val(
    model: PointWinModel,
    val_df: pd.DataFrame,
    match_format: MatchFormat,
) -> tuple[np.ndarray, np.ndarray]:
    """Evaluate one opening snapshot per match with a known outcome."""
    starts, probs = opening_predictions(model, val_df, match_format)
    return probs, starts.match_winner.eq(1).to_numpy(dtype=int)


def main():
    parser = argparse.ArgumentParser(description="Train tennis point-win model")
    parser.add_argument("--backend", choices=["xgboost", "lightgbm"], default="xgboost")
    parser.add_argument("--compare", action="store_true", help="Also train LightGBM and compare")
    parser.add_argument("--processed-dir", default=str(PROCESSED_DIR))
    parser.add_argument("--artifacts-dir", default=str(ARTIFACTS_DIR))
    args = parser.parse_args()

    os.makedirs(args.artifacts_dir, exist_ok=True)

    print("=== Loading features ===")
    train_path = os.path.join(args.processed_dir, "features_train.parquet")
    val_path = os.path.join(args.processed_dir, "features_val.parquet")

    if not os.path.exists(train_path) or not os.path.exists(val_path):
        print("ERROR: Feature files not found. Run build_features.py first.")
        sys.exit(1)

    X_train, y_train, meta_train = load_features(train_path)
    val_df = pd.read_parquet(val_path)
    val_df["match_date"] = pd.to_datetime(val_df["match_date"])

    train_dates = pd.to_datetime(meta_train["match_date"])
    if train_dates.max() >= val_df["match_date"].min():
        raise ValueError("Training dates must strictly precede validation dates")
    if "match_id" in meta_train and set(meta_train.match_id) & set(val_df.match_id):
        raise ValueError("Train and validation contain overlapping matches")

    # Split validation set: H1 for early stopping, H2 for calibration
    # This prevents the calibrator from seeing the same data used for GBDT training
    # Split at the midpoint between the min and max dates
    val_dates = val_df["match_date"].sort_values().unique()
    if len(val_dates) < 2:
        raise ValueError("Validation requires at least two distinct match dates.")
    val_cutoff_mid = val_dates[len(val_dates) // 2]
    mask_es = val_df["match_date"] < val_cutoff_mid
    mask_cal = val_df["match_date"] >= val_cutoff_mid

    if set(val_df.loc[mask_es, "match_id"]) & set(val_df.loc[mask_cal, "match_id"]):
        raise ValueError("Early stopping and calibration require distinct matches")
    X_val_es = val_df[mask_es][FEATURE_COLUMNS].values.astype(np.float32)
    y_val_es = val_df[mask_es]["server_won"].values.astype(np.float32)
    val_df_cal = val_df[mask_cal].copy()

    print(f"  Train:    {len(X_train):,} points")
    print(f"  Val (ES): {len(X_val_es):,} points (early stopping)")
    print(f"  Val (Cal):{len(val_df_cal):,} points (calibration)")

    print(f"\n=== Training {args.backend} model ===")
    model = PointWinModel(backend=args.backend)
    metrics = model.train(
        X_train, y_train,
        X_val_es, y_val_es,
        feature_names=FEATURE_COLUMNS,
        early_stopping_rounds=50,
        verbose=True,
    )

    print(f"\nMetrics:")
    print(f"  Train log-loss: {metrics['train_logloss']:.4f}")
    print(f"  Val   log-loss: {metrics['val_logloss']:.4f}")
    print(f"  Train Brier:    {metrics['train_brier']:.4f}")
    print(f"  Val   Brier:    {metrics['val_brier']:.4f}")

    if args.compare:
        alt = "lightgbm" if args.backend == "xgboost" else "xgboost"
        print(f"\n=== Comparing with {alt} ===")
        model_alt = PointWinModel(backend=alt)
        metrics_alt = model_alt.train(X_train, y_train, X_val_es, y_val_es,
                                       feature_names=FEATURE_COLUMNS, verbose=True)
        print(f"  {alt} val log-loss: {metrics_alt['val_logloss']:.4f}")
        print(f"  {alt} val Brier:    {metrics_alt['val_brier']:.4f}")
        if metrics_alt["val_logloss"] < metrics["val_logloss"]:
            print(f"  {alt} is better — using it.")
            model = model_alt

    # Feature importance
    imp = model.feature_importance()
    if imp:
        top10 = sorted(imp.items(), key=lambda x: x[1], reverse=True)[:10]
        print(f"\nTop 10 features by gain:")
        for name, score in top10:
            print(f"  {name:40s} {score:.1f}")

    print("\n=== Calibrating on validation set (H2) ===")
    snapshots = checkpoint_predictions(model, val_df_cal)
    print(f"  {len(snapshots):,} snapshots, {snapshots.match_id.nunique()} calibration matches")
    calibrator = MatchWinCalibrator()
    calibrator.fit_inplay(snapshots.raw, snapshots.outcome, snapshots.weight)
    if "player_id_source" in val_df:
        sources = val_df.player_id_source.unique()
        if len(sources) != 1:
            raise ValueError("Mixed player ID namespaces")
        calibrator.player_id_source = sources[0]
    print(f"  In-play calibration slope: {calibrator.slope:.4f}")

    print("\n=== Saving artifacts ===")
    model_path = os.path.join(args.artifacts_dir, "gbdt_model.pkl")
    cal_path = os.path.join(args.artifacts_dir, "calibrator.pkl")
    model.save(model_path)
    calibrator.save(cal_path)
    metadata = {
        "train_points": len(y_train), "train_matches": meta_train.match_id.nunique(),
        "train_through": str(train_dates.max().date()),
        "validation_start": str(val_dates.min())[:10], "calibration_from": str(val_cutoff_mid)[:10],
        "calibration_through": str(val_df_cal.match_date.max().date()),
        "calibration_snapshots": len(snapshots), "calibration_matches": snapshots.match_id.nunique(),
        "stride": 10, "calibration_slope": calibrator.slope, "player_id_source": calibrator.player_id_source,
    }
    (Path(args.artifacts_dir)/"fit_metadata.json").write_text(json.dumps(metadata, indent=2))
    print(f"  Model saved:      {model_path}")
    print(f"  Calibrator saved: {cal_path}")
    print("\nDone. Next step: python scripts/backtest.py")


if __name__ == "__main__":
    main()
