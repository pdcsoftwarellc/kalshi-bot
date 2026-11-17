#!/usr/bin/env python3
"""Build training datasets from raw JeffSackmann data.

Steps:
1. Load ATP match-level CSVs (tennis_atp/atp_matches_YYYY.csv)
2. Build ELO tracker from match history (temporal order)
3. Compute rolling player stats (serve%, return%, ranking)
4. Load point-by-point data from two sources:
   a) tennis_slam_pointbypoint — rich point-level data (Grand Slams 2011+)
   b) tennis_pointbypoint — PBP string data (ATP main tour)
5. For each point in each match, emit a row with:
      - Pre-match features (ELO, serve%, ranking, surface, etc.)
      - In-play features at that point in time
      - Target: server_won (0/1)
      - match_date (for temporal split)
6. Save as train/val/test parquet files

Usage:
    python scripts/build_features.py [--atp-dir DATA_DIR] [--out-dir OUT_DIR]
"""
import argparse
import glob
import os
import sys
from pathlib import Path

import pandas as pd

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from tennis.elo import EloTracker
from tennis.features import (
    build_player_stats,
    build_pre_match_features,
    combine_features,
    FEATURE_COLUMNS,
    build_inplay_features,
)

from tennis.research import RecordedGame, historical_points, parse_pbp_games

TRAIN_CUTOFF = "2022-01-01"
VAL_CUTOFF = "2023-01-01"

DATA_DIR = Path(__file__).parent.parent / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
ARTIFACTS_DIR = DATA_DIR / "artifacts"

SLAM_SURFACE = {
    "ausopen": "Hard",
    "frenchopen": "Clay",
    "wimbledon": "Grass",
    "usopen": "Hard",
}


def load_atp_matches(atp_dir: Path, start_year: int = 2003) -> pd.DataFrame:
    """Load all atp_matches_YYYY.csv files and concatenate."""
    pattern = str(atp_dir / "atp_matches_*.csv")
    files = sorted(glob.glob(pattern))
    if not files:
        raise FileNotFoundError(f"No ATP match files found in {atp_dir}")

    dfs = []
    for f in files:
        stem = os.path.basename(f).replace("atp_matches_", "").replace(".csv", "")
        try:
            year = int(stem)
        except ValueError:
            continue  # skip non-year files like atp_matches_amateur.csv
        if year < start_year:
            continue
        try:
            df = pd.read_csv(f, low_memory=False)
            dfs.append(df)
        except Exception as e:
            print(f"  WARNING: Could not read {f}: {e}")

    print(f"  Loaded {len(dfs)} ATP match files ({sum(len(d) for d in dfs):,} matches)")
    return pd.concat(dfs, ignore_index=True)


def build_name_to_id(matches_df: pd.DataFrame) -> dict:
    """Build player name → ATP numeric ID mapping from match data."""
    mapping = {}
    for col_id, col_name in [("winner_id", "winner_name"), ("loser_id", "loser_name")]:
        for _, row in matches_df[[col_id, col_name]].drop_duplicates().iterrows():
            name = str(row[col_name]).strip()
            pid = str(int(row[col_id])) if pd.notna(row[col_id]) else ""
            if name and pid and name != "nan":
                mapping[name] = pid
    print(f"  Name→ID mapping: {len(mapping)} players")
    return mapping


# Slam PBP loading (tennis_slam_pointbypoint)
# ---------------------------------------------------------------------------

def load_slam_pbp(slam_dir: Path, name_to_id: dict, atp_matches: pd.DataFrame) -> list:
    """Load Grand Slam point-by-point data and return list of match dicts.

    Each match dict contains:
        match_id, player1_id, player2_id, surface, tourney_level,
        match_date, winner (1 or 2), best_of and a sequence of RecordedGame values.
    """
    # Exclude doubles and mixed files (they have -doubles.csv or -mixed.csv suffixes)
    matches_files = sorted(
        f for f in glob.glob(str(slam_dir / "*-matches.csv"))
        if not f.endswith(("-doubles.csv", "-mixed.csv"))
    )
    points_files = sorted(
        f for f in glob.glob(str(slam_dir / "*-points.csv"))
        if not f.endswith(("-doubles.csv", "-mixed.csv"))
    )

    if not matches_files:
        print("  No slam match files found")
        return []

    # Load all matches
    match_dfs = []
    for f in matches_files:
        try:
            df = pd.read_csv(f, low_memory=False)
            # Extract slam name from filename (e.g., "2011-ausopen-matches.csv")
            fname = os.path.basename(f)
            parts = fname.replace("-matches.csv", "").split("-", 1)
            if len(parts) == 2:
                df["_year"] = parts[0]
                df["_slam"] = parts[1]
            match_dfs.append(df)
        except Exception as e:
            print(f"  WARNING: {f}: {e}")

    matches = pd.concat(match_dfs, ignore_index=True) if match_dfs else pd.DataFrame()

    # Load all points
    point_dfs = []
    for f in points_files:
        try:
            df = pd.read_csv(f, low_memory=False)
            point_dfs.append(df)
        except Exception as e:
            print(f"  WARNING: {f}: {e}")

    points = pd.concat(point_dfs, ignore_index=True) if point_dfs else pd.DataFrame()

    if matches.empty or points.empty:
        print("  No slam data loaded")
        return []

    # Anchor every observation to its ATP tournament start. ATP outcomes
    # carry that same date, so a later guessed date could leak the result.
    slam_names = {"ausopen": "Australian Open", "frenchopen": "Roland Garros",
                  "wimbledon": "Wimbledon", "usopen": "US Open"}
    atp_lookup = {}
    for row in atp_matches.loc[atp_matches.tourney_level.eq("G")].itertuples():
        date = str(row.tourney_date)
        pair = frozenset((str(int(row.winner_id)), str(int(row.loser_id))))
        atp_lookup[(date[:4], row.tourney_name, pair)] = row

    print(f"  Slam matches: {len(matches):,}, slam points: {len(points):,}")

    # Group points by match_id
    points_grouped = points.groupby("match_id")

    result = []
    skipped = 0
    for _, mrow in matches.iterrows():
        mid = mrow.get("match_id", "")
        if mid not in points_grouped.groups:
            skipped += 1
            continue

        match_points = points_grouped.get_group(mid)
        # Source files contain non-point rows (e.g. clock updates).
        match_points = match_points.loc[match_points.PointServer.isin([1, 2]) & match_points.PointWinner.isin([1, 2])]
        if len(match_points) < 10:
            skipped += 1
            continue

        # Map player names to ATP IDs
        p1_name = str(mrow.get("player1", "")).strip()
        p2_name = str(mrow.get("player2", "")).strip()
        p1_id = name_to_id.get(p1_name, p1_name)
        p2_id = name_to_id.get(p2_name, p2_name)

        slam = str(mrow.get("_slam", mrow.get("slam", "")))
        surface = SLAM_SURFACE.get(slam, "Hard")
        year = str(mrow.get("_year", mrow.get("year", "")))
        row = atp_lookup.get((year, slam_names.get(slam), frozenset((p1_id, p2_id))))
        if row is None:
            skipped += 1
            continue
        games = []
        for (set_no, _), group in match_points.groupby(["SetNo", "GameNo"], sort=False):
            points_in_game = [(int(ps), int(ps == pw)) for ps, pw in
                              zip(group.PointServer, group.PointWinner)]
            games.append(RecordedGame(int(set_no), points_in_game, group.PointServer.nunique() > 1))
        result.append({
            "match_id": f"{row.tourney_id}-{row.match_num}",
            "player1_id": p1_id, "player2_id": p2_id,
            "surface": surface, "tourney_level": "G",
            "match_date": pd.to_datetime(str(row.tourney_date), format="%Y%m%d").date().isoformat(),
            "winner": 1 if str(int(row.winner_id)) == p1_id else 2,
            "best_of": int(row.best_of), "games": games,
        })

    print(f"  Slam matches loaded: {len(result):,} ({skipped} skipped)")
    return result


# ---------------------------------------------------------------------------
# Main tour PBP loading (tennis_pointbypoint)
# ---------------------------------------------------------------------------

def load_main_pbp(pbp_dir: Path, matches_df: pd.DataFrame, name_to_id: dict) -> list:
    """Load main-tour PBP data (S/R string format) and return list of match dicts.

    Joins with ATP match data on player names + approximate date to get
    surface, tourney_level, and player IDs.
    """
    patterns = [
        str(pbp_dir / "pbp_matches_atp_main_*.csv"),
        str(pbp_dir / "pbp_matches_atp_qual_*.csv"),
    ]
    dfs = []
    for pat in patterns:
        for f in sorted(glob.glob(pat)):
            try:
                df = pd.read_csv(f, low_memory=False)
                dfs.append(df)
                print(f"  Loaded {os.path.basename(f)}: {len(df):,} matches")
            except Exception as e:
                print(f"  WARNING: {f}: {e}")

    if not dfs:
        return []

    pbp_df = pd.concat(dfs, ignore_index=True)

    # Parse PBP date to a standard format
    # Format is "DD Mon YY" (e.g., "01 Jan 17")
    pbp_df["_parsed_date"] = pd.to_datetime(pbp_df["date"], format="mixed", dayfirst=True, errors="coerce")

    # Build match lookup from ATP matches: (winner_name, loser_name, approx_date) → match row
    # We'll use a name-pair → list of matches lookup for fuzzy date matching
    matches_df = matches_df.copy()
    matches_df["_date"] = pd.to_datetime(matches_df["tourney_date"].astype(str), format="%Y%m%d", errors="coerce")
    matches_df["_wname"] = matches_df["winner_name"].str.strip()
    matches_df["_lname"] = matches_df["loser_name"].str.strip()

    # Build lookup: frozenset(name1, name2) → list of match rows
    match_lookup = {}
    for idx, row in matches_df.iterrows():
        wn = str(row["_wname"])
        ln = str(row["_lname"])
        key = frozenset([wn, ln])
        if key not in match_lookup:
            match_lookup[key] = []
        match_lookup[key].append(row)

    result = []
    matched = 0
    unmatched = 0
    skipped = 0

    for _, pbp_row in pbp_df.iterrows():
        pbp_str = str(pbp_row.get("pbp", ""))
        if not pbp_str or pbp_str == "nan":
            skipped += 1
            continue

        try:
            games = parse_pbp_games(pbp_str)
        except ValueError:
            skipped += 1
            continue
        if sum(len(game.points) for game in games) < 10:
            skipped += 1
            continue

        s1 = str(pbp_row.get("server1", "")).strip()
        s2 = str(pbp_row.get("server2", "")).strip()
        pbp_date = pbp_row.get("_parsed_date")

        if pd.isna(pbp_date):
            skipped += 1
            continue

        # Try to find matching ATP match
        key = frozenset([s1, s2])
        candidates = match_lookup.get(key, [])

        best_match = None
        best_delta = pd.Timedelta(days=999)
        for c in candidates:
            if pd.notna(c["_date"]):
                delta = abs(pbp_date - c["_date"])
                if delta < best_delta:
                    best_delta = delta
                    best_match = c

        if best_match is not None and best_delta <= pd.Timedelta(days=7):
            matched += 1
            winner_id = str(int(best_match["winner_id"]))
            loser_id = str(int(best_match["loser_id"]))
            surface = str(best_match.get("surface", "Hard"))
            tourney_level = str(best_match.get("tourney_level", "A"))
            match_date_raw = str(best_match.get("tourney_date", ""))
            if len(match_date_raw) == 8:
                match_date = f"{match_date_raw[:4]}-{match_date_raw[4:6]}-{match_date_raw[6:8]}"
            else:
                match_date = str(pbp_date.date())

            # Determine who is server1 (= first server)
            winner_name = str(best_match["_wname"])
            if s1 == winner_name:
                p1_id, p2_id = winner_id, loser_id
            else:
                p1_id, p2_id = loser_id, winner_id

            result.append({
                "match_id": f"{best_match['tourney_id']}-{best_match['match_num']}",
                "player1_id": p1_id,
                "player2_id": p2_id,
                "surface": surface,
                "tourney_level": tourney_level,
                "match_date": match_date,
                "winner": 1 if s1 == winner_name else 2,
                "games": games, "best_of": int(best_match["best_of"]),
            })
        else:
            unmatched += 1

    print(f"  Main PBP: {matched:,} matched, {unmatched:,} unmatched, {skipped:,} skipped")
    return result


# ---------------------------------------------------------------------------
# Feature row generation
# ---------------------------------------------------------------------------

def build_training_rows_from_matches(
    match_list: list,
    player_stats_df: pd.DataFrame,
    elo_tracker: EloTracker,
    h2h_df: pd.DataFrame,
) -> pd.DataFrame:
    """Generate one training row per point from processed match dicts."""
    rows = []
    for match in match_list:
        p1, p2 = match["player1_id"], match["player2_id"]
        pre = [build_pre_match_features(srv, rtn, match["surface"], match["tourney_level"],
                                        match["match_date"], player_stats_df, elo_tracker, h2h_df)
               for srv, rtn in ((p1, p2), (p2, p1))]
        for idx, (state, won) in enumerate(historical_points(match["games"], match["best_of"])):
            features = combine_features(pre[state.server], build_inplay_features(state))
            rows.append({
                **dict(zip(FEATURE_COLUMNS, features)),
                "server_won": won, "match_date": match["match_date"],
                "player1_id": p1, "player2_id": p2, "surface": match["surface"],
                "tournament_level": match["tourney_level"],
                "match_id": match["match_id"], "match_winner": match.get("winner", 0),
                "point_idx": idx, "server": state.server, "best_of": match["best_of"],
                "points_p1": state.points_p1, "points_p2": state.points_p2,
            })
    print(f"  Generated {len(rows):,} training points")
    return pd.DataFrame(rows)


def build_h2h(matches_df: pd.DataFrame) -> pd.DataFrame:
    """Compute head-to-head win rates for each player pair."""
    df = matches_df.copy()
    df["tourney_date"] = pd.to_datetime(df["tourney_date"].astype(str), format="%Y%m%d")
    df = df.sort_values("tourney_date").reset_index(drop=True)

    records = []
    pair_stats: dict = {}

    for _, row in df.iterrows():
        w = str(row["winner_id"])
        l = str(row["loser_id"])
        p1, p2 = min(w, l), max(w, l)
        key = (p1, p2)

        n, p1_wins = pair_stats.get(key, (0, 0))
        n += 1
        p1_wins += int(w == p1)
        pair_stats[key] = (n, p1_wins)
        for server, returner, wins in ((p1, p2, p1_wins), (p2, p1, n - p1_wins)):
            records.append({"player1_id": server, "player2_id": returner,
                            "date": row["tourney_date"], "n_matches": n,
                            "srv_win_rate": wins / n})

    return pd.DataFrame(records)


def main():
    parser = argparse.ArgumentParser(description="Build training features from raw data")
    parser.add_argument("--atp-dir", default=str(RAW_DIR / "tennis_atp"))
    parser.add_argument("--pbp-dir", default=str(RAW_DIR / "tennis_pointbypoint"))
    parser.add_argument("--slam-dir", default=str(RAW_DIR / "tennis_slam_pointbypoint"))
    parser.add_argument("--out-dir", default=str(PROCESSED_DIR))
    parser.add_argument("--artifacts-dir", default=str(ARTIFACTS_DIR))
    parser.add_argument("--start-year", type=int, default=2003)
    parser.add_argument("--fixtures-dir", type=Path, help="Daily ATP JSON from download_data.py --source api-tennis")
    parser.add_argument("--train-before", default=TRAIN_CUTOFF, help="First validation date")
    parser.add_argument("--test-from", default=VAL_CUTOFF, help="First test date")
    args = parser.parse_args()
    train_end, val_end = pd.Timestamp(args.train_before), pd.Timestamp(args.test_from)
    if train_end >= val_end:
        parser.error("--train-before must precede --test-from")

    os.makedirs(args.out_dir, exist_ok=True)
    os.makedirs(args.artifacts_dir, exist_ok=True)

    print("=== Step 1: Loading ATP match data ===")
    if args.fixtures_dir:
        from tennis.fixture_history import load_fixtures
        matches_df, all_matches = load_fixtures(args.fixtures_dir)
    else:
        matches_df = load_atp_matches(Path(args.atp_dir), start_year=args.start_year)

    print("\n=== Step 2: Building ELO tracker ===")
    elo = EloTracker()
    elo.process_matches(matches_df)
    elo.save(os.path.join(args.artifacts_dir, "player_elo.parquet"))
    print("  ELO tracker saved.")

    print("\n=== Step 3: Building player stats ===")
    player_stats = build_player_stats(matches_df)
    player_stats.to_parquet(
        os.path.join(args.artifacts_dir, "player_stats.parquet"), index=False
    )
    print(f"  Player stats: {len(player_stats):,} rows")

    print("\n=== Step 4: Building H2H records ===")
    h2h = build_h2h(matches_df)
    h2h.to_parquet(os.path.join(args.artifacts_dir, "h2h.parquet"), index=False)
    print(f"  H2H records: {len(h2h):,} rows")

    if not args.fixtures_dir:
        print("\n=== Step 5: Building name→ID mapping ===")
        name_to_id = build_name_to_id(matches_df)

        print("\n=== Step 6: Loading slam PBP data ===")
        slam_matches = load_slam_pbp(Path(args.slam_dir), name_to_id, matches_df)

        print("\n=== Step 7: Loading main tour PBP data ===")
        main_matches = load_main_pbp(Path(args.pbp_dir), matches_df, name_to_id)

        # Both feeds cover some of the same matches. Prefer the Slam point log.
        by_id = {m["match_id"]: m for m in main_matches + slam_matches}
        all_matches = list(by_id.values())
    print(f"\n  Total matches with PBP: {len(all_matches):,}")

    if not all_matches:
        print("  ERROR: No matches with point-by-point data found.")
        sys.exit(1)

    print("\n=== Step 8: Building feature dataset ===")
    features_df = build_training_rows_from_matches(
        all_matches, player_stats, elo, h2h
    )

    if features_df.empty:
        print("  ERROR: No training rows generated.")
        sys.exit(1)

    features_df["player_id_source"] = "api-tennis" if args.fixtures_dir else "atp"
    features_df["match_date"] = pd.to_datetime(features_df["match_date"])

    print("\n=== Step 9: Time-based split ===")
    train = features_df[features_df["match_date"] < train_end]
    val = features_df[
        (features_df["match_date"] >= train_end)
        & (features_df["match_date"] < val_end)
    ]
    test = features_df[features_df["match_date"] >= val_end]

    print(f"  Train: {len(train):,} points")
    if not train.empty:
        print(f"         ({train['match_date'].min()} – {train['match_date'].max()})")
    print(f"  Val:   {len(val):,} points")
    if not val.empty:
        print(f"         ({val['match_date'].min()} – {val['match_date'].max()})")
    print(f"  Test:  {len(test):,} points")
    if not test.empty:
        print(f"         ({test['match_date'].min()} – {test['match_date'].max()})")

    train.to_parquet(os.path.join(args.out_dir, "features_train.parquet"), index=False)
    val.to_parquet(os.path.join(args.out_dir, "features_val.parquet"), index=False)
    test.to_parquet(os.path.join(args.out_dir, "features_test.parquet"), index=False)
    print(f"\n  Saved to {args.out_dir}/")
    print("\nDone. Next step: python scripts/train.py")


if __name__ == "__main__":
    main()
