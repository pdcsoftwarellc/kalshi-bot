#!/usr/bin/env python3
"""Download JeffSackmann tennis datasets to data/raw/.

Clones (or updates) the following repositories:
    tennis_atp            — ATP match-level data (rankings, stats)
    tennis_wta            — WTA match-level data
    tennis_pointbypoint   — Point-by-point sequences (ATP/WTA)
    tennis_slam_pointbypoint — Grand Slam point-by-point (2011+)

Usage:
    python scripts/download_data.py
    python scripts/download_data.py --update   # git pull existing repos
"""
import argparse
import json
import tempfile
from datetime import date, timedelta
from pathlib import Path
import os
import subprocess
import sys

REPOS = {
    "tennis_atp": "https://github.com/JeffSackmann/tennis_atp.git",
    "tennis_wta": "https://github.com/JeffSackmann/tennis_wta.git",
    "tennis_pointbypoint": "https://github.com/JeffSackmann/tennis_pointbypoint.git",
    "tennis_slam_pointbypoint": "https://github.com/JeffSackmann/tennis_slam_pointbypoint.git",
}

DATA_RAW_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "raw")


def clone_or_update(name: str, url: str, update: bool) -> bool:
    dest = os.path.join(DATA_RAW_DIR, name)
    if os.path.exists(dest):
        if update:
            print(f"  Updating {name}...")
            result = subprocess.run(["git", "-C", dest, "pull", "--ff-only"], capture_output=True, text=True)
            if result.returncode != 0:
                print(f"  WARNING: git pull failed for {name}: {result.stderr.strip()}")
                return False
            else:
                print(f"  {name}: {result.stdout.strip() or 'already up to date'}")
        else:
            print(f"  {name}: already exists (use --update to refresh)")
    else:
        print(f"  Cloning {name} from {url}...")
        result = subprocess.run(
            ["git", "clone", "--depth", "1", url, dest],
            capture_output=True, text=True,
        )
        if result.returncode != 0:
            print(f"  ERROR cloning {name}: {result.stderr.strip()}", file=sys.stderr)
            return False
        else:
            print(f"  {name}: cloned OK")
    return True


def download_fixtures(start: date, end: date, update: bool) -> None:
    """Cache one complete API response (ATP singles only) per UTC day."""
    import httpx
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    import config

    if not config.API_TENNIS_KEY:
        raise SystemExit("Set API_TENNIS_KEY before downloading fixtures.")
    destination = Path(DATA_RAW_DIR) / "api_tennis"
    destination.mkdir(parents=True, exist_ok=True)
    day = start
    with httpx.Client(timeout=45) as client:
        while day <= end:
            path = destination / f"{day}.json"
            if update or not path.exists():
                try:
                    response = client.get(config.API_TENNIS_REST, params={
                        "APIkey": config.API_TENNIS_KEY, "method": "get_fixtures",
                        "date_start": str(day), "date_stop": str(day), "timezone": "UTC",
                    })
                    response.raise_for_status()
                    payload = response.json()
                    if str(payload.get("success")) != "1" or not isinstance(payload.get("result"), list):
                        raise ValueError("Invalid fixtures response")
                    fixtures = [m for m in payload["result"] if m["event_type_type"] == "Atp Singles"]
                except (httpx.HTTPError, ValueError, KeyError, TypeError, AttributeError) as exc:
                    raise SystemExit(f"{day}: fixture request failed ({type(exc).__name__}). Retry to resume.") from None
                with tempfile.NamedTemporaryFile(mode="w", dir=destination, delete=False) as tmp:
                    json.dump(fixtures, tmp)
                os.replace(tmp.name, path)
                print(f"{day}: cached {len(fixtures)} ATP singles fixtures")
            day += timedelta(days=1)
    print(f"Fixture cache: {destination}")


def main():
    parser = argparse.ArgumentParser(description="Download JeffSackmann tennis datasets")
    parser.add_argument("--update", action="store_true", help="Update existing repos with git pull")
    parser.add_argument("--repos", nargs="+", choices=list(REPOS), default=list(REPOS),
                        help="Which repos to download (default: all)")
    parser.add_argument("--source", choices=["legacy", "api-tennis"], default="legacy")
    parser.add_argument("--start", type=date.fromisoformat)
    parser.add_argument("--end", type=date.fromisoformat)
    args = parser.parse_args()
    if args.source == "api-tennis":
        if not args.start or not args.end or args.start > args.end:
            parser.error("For api-tennis supply --start and --end in chronological order (YYYY-MM-DD).")
        download_fixtures(args.start, args.end, args.update)
        return
    if args.start or args.end:
        parser.error("Date ranges apply to --source api-tennis.")

    os.makedirs(DATA_RAW_DIR, exist_ok=True)
    print(f"Data directory: {DATA_RAW_DIR}")

    failed = []
    for name in args.repos:
        if not clone_or_update(name, REPOS[name], update=args.update):
            failed.append(name)

    if failed:
        print(f"\nDownload failed: {', '.join(failed)}. Restore local datasets or update the source URLs.", file=sys.stderr)
        sys.exit(1)

    print("\nDone. Next step: python scripts/build_features.py")


if __name__ == "__main__":
    main()
