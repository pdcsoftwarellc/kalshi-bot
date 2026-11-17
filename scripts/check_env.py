#!/usr/bin/env python3
"""Check startup requirements and optionally make read-only API requests.

This checks configuration, not profitability or live execution correctness.
No orders are sent. Credentials and response bodies are never printed.
"""
import argparse
import asyncio
import importlib
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


async def check_connections(cfg, *, kalshi_env: str | None) -> bool:
    import httpx
    from integration.kalshi_client import KalshiClient

    passed = True
    async with KalshiClient("", "", demo=False) as client:
        try:
            markets = await client.get_markets(cfg.KALSHI_SERIES_TICKER, limit=1)
            print(f"[OK] Public Kalshi production API reachable ({len(markets)} market(s) in sample).")
        except Exception as exc:
            print(f"[FAIL] Public Kalshi API request failed ({type(exc).__name__}).")
            passed = False

    if cfg.API_TENNIS_KEY.strip():
        try:
            async with httpx.AsyncClient(timeout=10) as client:
                response = await client.get(cfg.API_TENNIS_REST, params={
                    "method": "get_livescore", "APIkey": cfg.API_TENNIS_KEY,
                })
                response.raise_for_status()
                payload = response.json()
                if not isinstance(payload, dict) or str(payload.get("success")) != "1":
                    raise ValueError("Feed rejected the request")
                print("[OK] API-Tennis credentials accepted for live-score REST access.")
        except Exception as exc:
            # HTTP errors can contain the API key in the query string.
            print(f"[FAIL] API-Tennis live-score request failed ({type(exc).__name__}); check credentials and subscription.")
            passed = False
    else:
        print("[SKIP] API-Tennis authentication: API_TENNIS_KEY is missing.")

    if kalshi_env:
        demo = kalshi_env == "demo"
        prefix = "DEMO_KALSHI" if demo else "KALSHI"
        api_key = getattr(cfg, prefix + "_API_KEY")
        key_path = getattr(cfg, prefix + "_PRIVATE_KEY_PATH")
        if not api_key.strip() or not key_path:
            print(f"[FAIL] Set {prefix}_API_KEY and {prefix}_PRIVATE_KEY_PATH.")
            passed = False
        else:
            try:
                async with KalshiClient(api_key, key_path, demo=demo) as client:
                    await client.get_balance()
                print(f"[OK] Kalshi {kalshi_env} authenticated balance request accepted (balance not printed).")
            except Exception as exc:
                print(f"[FAIL] Kalshi {kalshi_env} authentication failed ({type(exc).__name__}).")
                passed = False
    return passed


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--online", action="store_true", help="Check public prices and API-Tennis REST access")
    parser.add_argument("--kalshi-env", choices=("demo", "prod"),
                        help="With --online, also check credentials using a GET balance request")
    args = parser.parse_args()
    if args.kalshi_env and not args.online:
        parser.error("--kalshi-env requires --online")

    print("Read-only environment check; no orders are sent.")
    passed = sys.version_info >= (3, 12)
    print(f"[{'OK' if passed else 'FAIL'}] Python {sys.version.split()[0]} (requires 3.12+).")
    dependencies_ok = True
    for module in ("dotenv", "numpy", "pandas", "pyarrow", "scipy", "sklearn",
                   "xgboost", "lightgbm", "httpx", "cryptography", "websockets",
                   "fastapi", "uvicorn", "aiosqlite"):
        try:
            importlib.import_module(module)
        except Exception as exc:
            print(f"[FAIL] Cannot import {module} ({type(exc).__name__}).")
            if module in ("xgboost", "lightgbm") and sys.platform == "darwin":
                print("       On macOS, install the native runtime: brew install libomp")
            dependencies_ok = False
    if not dependencies_ok:
        return 1
    print("[OK] Python dependencies and native model libraries import successfully.")

    try:
        import config as cfg
        from integration.preflight import REQUIRED_ARTIFACTS, SUPPORTING_ARTIFACTS, startup_errors
    except (ValueError, TypeError, OSError):
        print("[FAIL] Cannot load .env configuration; check numeric values and file access.")
        return 1
    errors = startup_errors()
    for error in errors:
        print(f"[FAIL] {error}")
    passed = passed and not errors
    if cfg.API_TENNIS_KEY.strip():
        print("[OK] API_TENNIS_KEY is configured (value hidden).")
    for prefix in ("KALSHI", "DEMO_KALSHI"):
        api_key, key_path = (getattr(cfg, prefix + suffix)
                             for suffix in ("_API_KEY", "_PRIVATE_KEY_PATH"))
        if not api_key.strip() and not key_path:
            print(f"[SKIP] {prefix} credentials: optional for paper trading.")
        else:
            from integration.kalshi_client import KalshiClient
            try:
                if not api_key.strip() or not key_path:
                    raise ValueError("Incomplete credentials")
                KalshiClient(api_key, key_path)
                print(f"[OK] {prefix} key pair is configured and PEM is readable (values hidden).")
            except (OSError, ValueError, TypeError):
                print(f"[FAIL] {prefix} credentials are incomplete or the PEM is invalid.")
                passed = False

    uses_provider_ids = False
    artifacts = Path(cfg.ARTIFACTS_DIR)
    if all((artifacts / name).is_file() for name in REQUIRED_ARTIFACTS):
        try:
            from tennis.engine import PredictionEngine
            engine = PredictionEngine.load(str(artifacts))
            uses_provider_ids = engine._calibrator.player_id_source == "api-tennis"
            print("[OK] Trained model artifacts load successfully.")
        except Exception as exc:
            print(f"[FAIL] Model artifacts cannot be loaded ({type(exc).__name__}).")
            passed = False
    for name in SUPPORTING_ARTIFACTS:
        if not (artifacts / name).is_file():
            print(f"[WARN] Missing {name}; engine would use incomplete player history.")
    if not uses_provider_ids and not any(Path(cfg.ATP_RAW_DIR).glob("atp_matches_*.csv")):
        print("[WARN] Raw ATP match data is missing; player resolution is limited to aliases.")

    if args.online:
        online_ok = asyncio.run(check_connections(cfg, kalshi_env=args.kalshi_env))
        passed = passed and online_ok
    print("Startup prerequisites passed." if passed else "Setup is incomplete; resolve the failures above.")
    print("These checks do not validate strategy profitability, fills, or live risk controls.")
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
