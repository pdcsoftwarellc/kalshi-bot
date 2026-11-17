"""Probability calibration for the tennis win probability engine.

New models fit a symmetric log-odds slope on held-out in-play snapshots.
The isotonic fit and loader remain available for earlier artifacts.
"""
from __future__ import annotations

import os
import pickle
from typing import Optional

import numpy as np
from scipy.special import expit, logit
from scipy.optimize import minimize_scalar
from sklearn.isotonic import IsotonicRegression


class MatchWinCalibrator:
    """Calibrate match probabilities using held-out labeled observations.

    Fits on (raw_prob, match_outcome) pairs from the validation set.
    At inference time, maps raw Markov chain probabilities to calibrated ones.
    """

    def __init__(self):
        self._calibrator: Optional[IsotonicRegression] = None
        self._fitted = False
        self.slope = None
        self.player_id_source = None

    # ------------------------------------------------------------------
    # Fitting
    # ------------------------------------------------------------------

    def fit(self, probs: np.ndarray, outcomes: np.ndarray) -> None:
        """Fit calibration model on held-out (raw_prob, outcome) pairs.

        Args:
            probs: Array of raw match win probabilities ∈ [0, 1].
                   These should come from the validation set only.
            outcomes: Binary match outcomes (1 if player 1 won, 0 if player 2).
                      Must have the same length as probs.
        """
        probs = np.asarray(probs, dtype=np.float64)
        outcomes = np.asarray(outcomes, dtype=np.float64)

        if len(probs) != len(outcomes):
            raise ValueError(f"probs ({len(probs)}) and outcomes ({len(outcomes)}) must have equal length.")

        # Isotonic regression: y_min=0, y_max=1, increasing=True
        # out_of_bounds="clip" ensures extrapolation stays in [0, 1]
        self.slope = None
        self._calibrator = IsotonicRegression(
            y_min=0.0, y_max=1.0, increasing=True, out_of_bounds="clip"
        )
        self._calibrator.fit(probs, outcomes)
        self._fitted = True

    # ------------------------------------------------------------------
    def fit_inplay(self, probs, outcomes, weights):
        """Fit a monotone odds slope. Equal total weight per match, no player-order bias.

        A weak unit penalty on log(slope) favors identity for small samples.
        A score of 0.5 stays 0.5 and exchanging players complements the result.
        """
        probs, outcomes, weights = (np.asarray(v, dtype=float) for v in (probs, outcomes, weights))
        if (not probs.size or probs.ndim != 1 or not (probs.shape == outcomes.shape == weights.shape)
                or not np.all((probs >= 0) & (probs <= 1))
                or not np.all(np.isin(outcomes, (0, 1)))
                or not np.all(np.isfinite(weights) & (weights > 0))):
            raise ValueError("Expected aligned probabilities, binary outcomes and positive match weights")
        logits = logit(np.clip(probs, 1e-7, 1-1e-7))
        def loss(log_slope):
            z = np.exp(log_slope) * logits
            return float(weights @ (np.logaddexp(0, z) - outcomes*z)) + .5*log_slope**2
        fit = minimize_scalar(loss, bounds=(-3., 3.), method="bounded")
        if not fit.success:
            raise ValueError("In-play calibration did not converge")
        self.slope = float(np.exp(fit.x))
        self._calibrator = None
        self._fitted = True

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------

    def calibrate(self, prob: float) -> float:
        """Map a raw probability to a calibrated probability.

        Args:
            prob: Raw match win probability from the Markov chain.

        Returns:
            Calibrated probability ∈ [0, 1].
        """
        if not self._fitted:
            return prob  # No-op if not fitted yet
        if self.slope is not None:
            return float(expit(self.slope * logit(prob)))
        return float(self._calibrator.predict([prob])[0])

    def calibrate_batch(self, probs: np.ndarray) -> np.ndarray:
        """Batch calibration for arrays of probabilities."""
        if not self._fitted:
            return np.asarray(probs, dtype=np.float64)
        if self.slope is not None:
            return expit(self.slope * logit(probs))
        return self._calibrator.predict(probs).astype(np.float64)

    # ------------------------------------------------------------------
    # Diagnostics
    # ------------------------------------------------------------------

    def expected_calibration_error(
        self,
        probs: np.ndarray,
        outcomes: np.ndarray,
        n_bins: int = 20,
    ) -> float:
        """Expected Calibration Error (ECE) using equal-frequency bins.

        Lower is better. A perfectly calibrated model has ECE = 0.

        Args:
            probs: Predicted probabilities (calibrated or uncalibrated).
            outcomes: True binary outcomes.
            n_bins: Number of bins.

        Returns:
            ECE ∈ [0, 1].
        """
        probs = np.asarray(probs, dtype=np.float64)
        outcomes = np.asarray(outcomes, dtype=np.float64)
        n = len(probs)

        # Equal-frequency binning
        sorted_idx = np.argsort(probs)
        bin_size = n // n_bins

        ece = 0.0
        for i in range(n_bins):
            start = i * bin_size
            end = (i + 1) * bin_size if i < n_bins - 1 else n
            if end <= start:
                continue
            idx = sorted_idx[start:end]
            bin_prob = probs[idx].mean()
            bin_outcome = outcomes[idx].mean()
            ece += (end - start) / n * abs(bin_prob - bin_outcome)

        return float(ece)

    def calibration_curve(
        self,
        probs: np.ndarray,
        outcomes: np.ndarray,
        n_bins: int = 20,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Compute calibration curve (mean predicted prob vs actual frequency).

        Returns:
            (mean_predicted, fraction_of_positives) arrays of length n_bins.
        """
        from sklearn.calibration import calibration_curve as sk_cal_curve

        fraction_of_positives, mean_predicted = sk_cal_curve(
            outcomes, probs, n_bins=n_bins, strategy="quantile"
        )
        return mean_predicted, fraction_of_positives

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self, path: str) -> None:
        """Save calibrator to a pickle file."""
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump({"calibrator": self._calibrator, "fitted": self._fitted, "slope": self.slope,
                         "player_id_source": self.player_id_source}, f)

    @classmethod
    def load(cls, path: str) -> MatchWinCalibrator:
        """Load calibrator from pickle file."""
        obj = cls()
        with open(path, "rb") as f:
            data = pickle.load(f)
        obj._calibrator = data["calibrator"]
        obj._fitted = data["fitted"]
        obj.slope = data.get("slope")
        obj.player_id_source = data.get("player_id_source")
        return obj
