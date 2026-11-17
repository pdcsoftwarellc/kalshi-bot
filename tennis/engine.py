"""PredictionEngine — the live integration seam.

This is the ONLY class that downstream integrations (live feed, Kalshi adapter)
need to import. Everything else in the tennis/ package is internal.

Usage pattern (live match):
    engine = PredictionEngine.load("data/artifacts/")

    # At match start:
    engine.new_match(initial_state)
    result = engine.predict(initial_state)
    print(result.win_prob_p1, result.confidence)

    # After each point:
    engine.record_point(server_won=True)
    new_state = <update state from live feed>
    result = engine.predict(new_state)
"""
from __future__ import annotations

import os
from typing import Optional

import numpy as np
import pandas as pd

from tennis.types import MatchState, PredictionResult
from tennis.markov import match_win_prob_batch
from tennis.updater import BayesianUpdater
from tennis.features import (
    build_pre_match_features,
    build_inplay_features,
    combine_features,
    TOUR_AVERAGES,
)

# Lazy imports to avoid hard-dep at module load time
_model_cls = None
_calibrator_cls = None
_elo_cls = None


def _get_model_cls():
    global _model_cls
    if _model_cls is None:
        from tennis.model import PointWinModel
        _model_cls = PointWinModel
    return _model_cls


def _get_calibrator_cls():
    global _calibrator_cls
    if _calibrator_cls is None:
        from tennis.calibration import MatchWinCalibrator
        _calibrator_cls = MatchWinCalibrator
    return _calibrator_cls


def _get_elo_cls():
    global _elo_cls
    if _elo_cls is None:
        from tennis.elo import EloTracker
        _elo_cls = EloTracker
    return _elo_cls


class PredictionEngine:
    """Real-time tennis win probability engine.

    The engine maintains per-match state (BayesianUpdater) and holds the
    pre-trained GBDT and calibrator as read-only artifacts. It is stateful
    with respect to the current match but stateless otherwise.

    Attributes:
        n_mc_samples: Number of Monte Carlo samples for CI estimation.
                      500 is the default; increase for tighter CIs at the cost
                      of ~2x latency (still fast: ~4ms for 1000 samples).
    """

    def __init__(
        self,
        model,
        calibrator,
        elo_tracker,
        player_stats_df: Optional[pd.DataFrame],
        h2h_df: Optional[pd.DataFrame] = None,
        n_mc_samples: int = 500,
        rng_seed: Optional[int] = None,
    ):
        self._model = model
        self._calibrator = calibrator
        self._elo = elo_tracker
        self._player_stats = player_stats_df
        self._h2h = h2h_df
        self.n_mc_samples = n_mc_samples
        self._rng = np.random.default_rng(rng_seed)

        # Per-match state (reset by new_match())
        self._updaters: tuple[BayesianUpdater, BayesianUpdater] | None = None
        self._current_server: Optional[int] = None  # server at match start

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def new_match(self, state: MatchState) -> None:
        """Initialise the engine for a new match.

        Must be called once at the start of each match before any predict()
        or record_point() calls.

        Args:
            state: Initial MatchState (typically sets/games/points all at 0).
        """
        priors = []
        for srv, rtn in ((state.player1_id, state.player2_id), (state.player2_id, state.player1_id)):
            pre = build_pre_match_features(srv, rtn, state.surface, state.tournament_level,
                                           state.match_date, self._player_stats, self._elo, self._h2h)
            priors.append(self._serve_prior(pre))
        self.start_with_priors(*priors, server=state.server)

    def _serve_prior(self, pre):
        if self._model is None:
            return TOUR_AVERAGES["srv_total_won_pct"]
        blank = build_inplay_features(MatchState())
        return float(np.clip(self._model.predict_single(combine_features(pre, blank)), 0.01, 0.99))

    def start_with_priors(self, p1: float, p2: float, server: int = 0, *,
                          virtual_sample_size=None, decay_rate=1.0):
        """Reset a match with the two pre-point predictions for its players."""
        self._updaters = (BayesianUpdater(p1, virtual_sample_size, decay_rate),
                          BayesianUpdater(p2, virtual_sample_size, decay_rate))
        self._current_server = server

    def record_point(self, server_won: bool, server: int | None = None) -> None:
        """Attribute the outcome to the player who served the observed point.

        For sequential callers, omitting server uses the server in the last
        prediction. Feed and replay callers supply it explicitly at game changes.
        """
        if self._updaters is None:
            raise RuntimeError("Call new_match() before record_point().")
        serving = self._current_server if server is None else server
        if serving not in (0, 1):
            raise ValueError("Server must be player 0 or 1")
        self._updaters[serving].update(server_won)

    def predict(self, state: MatchState) -> PredictionResult:
        """Return the current match win probability and confidence estimate.

        This is the primary method called by the live integration layer.

        Args:
            state: Current MatchState. Should reflect the score AFTER the most
                   recent point (and after calling record_point()).

        Returns:
            PredictionResult with win_prob_p1, confidence, CI bounds, and
            the current p_serve estimate.
        """
        if self._updaters is None:
            # Auto-initialise if new_match was not called (convenience for testing)
            self.new_match(state)

        self._current_server = state.server
        current = self._updaters[state.server]
        other = self._updaters[1-state.server]
        p_serve_posterior = current.posterior_mean
        win_probs = match_win_prob_batch(
            state, current.sample(n=self.n_mc_samples, rng=self._rng),
            other.sample(n=self.n_mc_samples, rng=self._rng),
        )

        # 4. Summarise MC distribution
        win_prob_raw = float(np.mean(win_probs))
        ci_lower = float(np.percentile(win_probs, 2.5))
        ci_upper = float(np.percentile(win_probs, 97.5))

        # 5. Calibrate the mean win probability
        win_prob_cal = self._calibrator.calibrate(win_prob_raw)
        # Recalibrate CI bounds consistently
        ci_lower_cal = self._calibrator.calibrate(ci_lower)
        ci_upper_cal = self._calibrator.calibrate(ci_upper)

        win_prob_cal = float(np.clip(win_prob_cal, 1e-6, 1 - 1e-6))
        ci_lower_cal = float(np.clip(ci_lower_cal, 0.0, 1.0))
        ci_upper_cal = float(np.clip(ci_upper_cal, 0.0, 1.0))

        return PredictionResult(
            win_prob_p1=win_prob_cal,
            win_prob_p2=1.0 - win_prob_cal,
            confidence=float(1.0 - (ci_upper_cal - ci_lower_cal)),
            ci_lower=ci_lower_cal,
            ci_upper=ci_upper_cal,
            p_serve_estimate=p_serve_posterior,
            points_observed=current.n_observed + other.n_observed,
            raw_win_prob_p1=win_prob_raw,
        )

    # ------------------------------------------------------------------
    # Loading from artifacts directory
    # ------------------------------------------------------------------

    @classmethod
    def load(cls, artifacts_dir: str, n_mc_samples: int = 500, rng_seed: int | None = None) -> PredictionEngine:
        """Load a trained PredictionEngine from saved artifacts.

        Expected files in artifacts_dir:
            gbdt_model.pkl         — trained PointWinModel
            calibrator.pkl         — trained MatchWinCalibrator
            player_elo.parquet     — EloTracker history
            player_stats.parquet   — rolling player stats (optional)
            h2h.parquet            — head-to-head records (optional)

        Args:
            artifacts_dir: Path to directory containing model artifacts.
            n_mc_samples: Number of Monte Carlo samples for CI (default 500).

        Returns:
            Loaded PredictionEngine ready for inference.
        """
        ModelCls = _get_model_cls()
        CalCls = _get_calibrator_cls()
        EloCls = _get_elo_cls()

        model_path = os.path.join(artifacts_dir, "gbdt_model.pkl")
        cal_path = os.path.join(artifacts_dir, "calibrator.pkl")
        elo_path = os.path.join(artifacts_dir, "player_elo.parquet")
        stats_path = os.path.join(artifacts_dir, "player_stats.parquet")
        h2h_path = os.path.join(artifacts_dir, "h2h.parquet")

        if not os.path.exists(model_path):
            raise FileNotFoundError(f"GBDT model not found: {model_path}")
        if not os.path.exists(cal_path):
            raise FileNotFoundError(f"Calibrator not found: {cal_path}")

        model = ModelCls.load(model_path)
        calibrator = CalCls.load(cal_path)

        elo_tracker = EloCls.load(elo_path) if os.path.exists(elo_path) else EloCls()

        player_stats = (
            pd.read_parquet(stats_path) if os.path.exists(stats_path) else None
        )
        h2h = pd.read_parquet(h2h_path) if os.path.exists(h2h_path) else None

        return cls(
            model=model,
            calibrator=calibrator,
            elo_tracker=elo_tracker,
            player_stats_df=player_stats,
            h2h_df=h2h,
            n_mc_samples=n_mc_samples, rng_seed=rng_seed,
        )

    @classmethod
    def from_components(
        cls,
        model,
        calibrator,
        elo_tracker=None,
        player_stats_df=None,
        h2h_df=None,
        n_mc_samples: int = 500, rng_seed: int | None = None,
    ) -> PredictionEngine:
        """Construct engine from already-loaded components (useful for testing)."""
        if elo_tracker is None:
            EloCls = _get_elo_cls()
            elo_tracker = EloCls()
        return cls(
            model=model,
            calibrator=calibrator,
            elo_tracker=elo_tracker,
            player_stats_df=player_stats_df,
            h2h_df=h2h_df,
            n_mc_samples=n_mc_samples, rng_seed=rng_seed,
        )
