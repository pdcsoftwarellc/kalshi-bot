"""Local startup checks; no network requests or trading side effects."""
import math
from pathlib import Path

import config


REQUIRED_ARTIFACTS = ("gbdt_model.pkl", "calibrator.pkl")
SUPPORTING_ARTIFACTS = ("player_elo.parquet", "player_stats.parquet", "h2h.parquet")


def startup_errors(*, paper: bool = True, demo: bool = True) -> list[str]:
    errors = []
    if not config.API_TENNIS_KEY.strip():
        errors.append("Set API_TENNIS_KEY in .env.")

    missing = [name for name in REQUIRED_ARTIFACTS
               if not (Path(config.ARTIFACTS_DIR) / name).is_file()]
    if missing:
        errors.append("Missing trained artifacts: " + ", ".join(missing)
                      + ". Restore data, then run build_features.py and train.py.")

    if not paper:
        from integration.kalshi_client import KalshiClient

        if demo and config.HYBRID_PRICING and config.PROD_KALSHI_API_KEY:
            try:
                if not config.PROD_KALSHI_PRIVATE_KEY_PATH:
                    raise ValueError("Missing hybrid price key")
                KalshiClient(config.PROD_KALSHI_API_KEY, config.PROD_KALSHI_PRIVATE_KEY_PATH, demo=False)
            except (OSError, ValueError, TypeError):
                errors.append("PROD_KALSHI_PRIVATE_KEY_PATH must point to a readable RSA or Ed25519 PEM private key.")
        if demo:
            api_key = config.DEMO_KALSHI_API_KEY
            key_path = config.DEMO_KALSHI_PRIVATE_KEY_PATH
            label = "DEMO_KALSHI"
        else:
            api_key = config.KALSHI_API_KEY
            key_path = config.KALSHI_PRIVATE_KEY_PATH
            label = "KALSHI"
        if not api_key.strip() or not key_path:
            errors.append(f"Set {label}_API_KEY and {label}_PRIVATE_KEY_PATH for this environment.")
        else:
            try:
                KalshiClient(api_key, key_path, demo=demo)
            except (OSError, ValueError, TypeError):
                errors.append(f"{label}_PRIVATE_KEY_PATH must point to a readable RSA or Ed25519 PEM private key.")

    for name in ("MIN_EDGE_PCT", "MIN_CONFIDENCE"):
        value = getattr(config, name)
        if not math.isfinite(value) or not 0 <= value < 1:
            errors.append(f"{name} must be finite and in [0, 1).")
    for name in ("MAX_CONTRACTS_PER_SIGNAL", "MAX_CONTRACTS_PER_MATCH",
                 "MAX_TOTAL_EXPOSURE", "REST_POLL_SECS", "MARKET_REFRESH_SECS",
                 "MATCH_STOP_LOSS_DOLLARS", "PORTFOLIO_STOP_LOSS_DOLLARS",
                 "PROFIT_TARGET_DOLLARS"):
        value = getattr(config, name)
        if not math.isfinite(value) or value <= 0:
            errors.append(f"{name} must be finite and greater than zero.")
    if config.MIN_POINTS_OBSERVED < 0:
        errors.append("MIN_POINTS_OBSERVED must be nonnegative.")
    if not math.isfinite(config.NET_EDGE_MIN_DOLLARS) or config.NET_EDGE_MIN_DOLLARS < 0:
        errors.append("NET_EDGE_MIN_DOLLARS must be finite and nonnegative.")
    if not 1 <= config.WEB_PORT <= 65535:
        errors.append("WEB_PORT must be between 1 and 65535.")
    return errors
