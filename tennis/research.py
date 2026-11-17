"""Source game boundaries and pre-point snapshots for historical singles matches."""
from dataclasses import dataclass

import numpy as np
import pandas as pd

from tennis.features import FEATURE_COLUMNS, build_inplay_features, TOUR_AVERAGES
from tennis.engine import PredictionEngine
from tennis.calibration import MatchWinCalibrator
from tennis.markov import match_win_prob

from tennis.types import MatchFormat, MatchState


@dataclass
class RecordedGame:
    set_number: int
    points: list[tuple[int, int]]  # one-based server ID and server-won flag
    tiebreak: bool


def parse_pbp_games(pbp: str) -> list[RecordedGame]:
    """Decode Sackmann set (.), game (;) and tiebreak service (/) boundaries."""
    games = []
    server = 1
    outcomes = {"S": 1, "A": 1, "R": 0, "D": 0}
    for set_number, set_text in enumerate(pbp.strip().rstrip(".").split("."), 1):
        for game in set_text.strip(";").split(";"):
            first_server = server
            points = []
            for run in game.split("/"):
                if not run or any(c not in outcomes for c in run):
                    raise ValueError("Incomplete or unknown PBP service run")
                points.extend((server, outcomes[c]) for c in run)
                server = 3 - server
            games.append(RecordedGame(set_number, points, "/" in game))
            server = 3 - first_server
    return games


def historical_points(games: list[RecordedGame], best_of: int):
    """Yield the state immediately before each target. Consume each state before advancing."""
    state = MatchState(match_format=MatchFormat(best_of=best_of))
    previous_set = 1
    for game in games:
        if game.set_number != previous_set:
            if game.set_number != previous_set + 1 or state.games_p1 == state.games_p2:
                raise ValueError("Inconsistent recorded set boundary")
            state.sets_p1 += int(state.games_p1 > state.games_p2)
            state.sets_p2 += int(state.games_p2 > state.games_p1)
            state.games_p1 = state.games_p2 = 0
            previous_set = game.set_number
        state.is_tiebreak = game.tiebreak
        state.points_p1 = state.points_p2 = 0
        for server, won in game.points:
            state.server = server - 1
            yield state, won
            p1_won = bool(won) == (server == 1)
            state.points_p1 += int(p1_won)
            state.points_p2 += int(not p1_won)
            if server == 1:
                state.serve_pts_played_p1 += 1
                state.serve_pts_won_p1 += won
            else:
                state.serve_pts_played_p2 += 1
                state.serve_pts_won_p2 += won
            state.recent_points.append(won)
            del state.recent_points[:-10]
        # The source marks the game boundary. This also handles advantage sets
        # and different tiebreak targets without guessing the tournament rules.
        state.games_p1 += int(p1_won)
        state.games_p2 += int(not p1_won)


def labeled_starts(frame: pd.DataFrame) -> pd.DataFrame:
    """One opening observation per match with a recorded winner."""
    required = {"match_id", "point_idx", "match_winner"}
    if not required.issubset(frame):
        raise ValueError("Match evaluation requires match_id, point_idx and match_winner. Rebuild features.")
    starts = frame.loc[frame.point_idx.eq(0)]
    if starts.match_id.duplicated().any():
        raise ValueError("Duplicate opening point for the same match")
    return starts.loc[starts.match_winner.isin([1, 2])]


def opening_state(row, default_format):
    # Legacy parquets were produced assuming the first server was player 1.
    return MatchState(server=int(row.get("server", 0)), match_format=MatchFormat(
        best_of=int(row.get("best_of", default_format.best_of)),
        final_set_tiebreak=default_format.final_set_tiebreak,
        final_set_advantage=default_format.final_set_advantage,
    ))


def serve_priors(model, starts):
    """Use a match's opening features twice, once in each server orientation."""
    first = starts[FEATURE_COLUMNS].copy()
    for name, value in build_inplay_features(MatchState()).items():
        first[name] = value
    reverse = first.copy()
    for name in first.columns:
        if name.endswith("_server"):
            partner = name[:-7] + "_returner"
            reverse[name], reverse[partner] = first[partner], first[name]
        elif name.endswith("_diff"):
            reverse[name] = -first[name]
    reverse["h2h_srv_win_rate"] = 1-first.h2h_srv_win_rate
    direct, swapped = (np.clip(model.predict_proba(x.to_numpy(dtype=np.float32)), .01, .99)
                       for x in (first, reverse))
    server = starts.server.to_numpy() if "server" in starts else np.zeros(len(first))
    return np.where(server == 0, direct, swapped), np.where(server == 0, swapped, direct)


def opening_predictions(model, frame, default_format):
    starts = labeled_starts(frame)
    if starts.empty:
        return starts, np.array([])
    engine = PredictionEngine(None, MatchWinCalibrator(), None, None, n_mc_samples=200, rng_seed=42)
    raw = []
    for (_, row), p1, p2 in zip(starts.iterrows(), *serve_priors(model, starts), strict=True):
        state = opening_state(row, default_format)
        engine.start_with_priors(p1, p2, state.server)
        raw.append(engine.predict(state).raw_win_prob_p1)
    return starts, np.asarray(raw)


def score_state(row):
    """Restore a pre-point score from the server-oriented training columns."""
    state = MatchState(server=int(row.server), match_format=MatchFormat(best_of=int(row.best_of)),
                       points_p1=int(row.points_p1), points_p2=int(row.points_p2),
                       is_tiebreak=bool(row.is_tiebreak))
    sets = int(row.sets_server), int(row.sets_returner)
    games = int(row.games_server), int(row.games_returner)
    state.sets_p1, state.sets_p2 = sets if state.server == 0 else sets[::-1]
    state.games_p1, state.games_p2 = games if state.server == 0 else games[::-1]
    return state


def checkpoint_predictions(model, frame, calibrator=None, stride=10, *, score_baseline=False):
    """Predict before observing the target point, at fixed observation counts.

    No winner, future point outcome or match duration enters the engine.
    Equal total weight per match prevents long matches dominating calibration.
    """
    required = {"server", "best_of", "points_p1", "points_p2", "server_won"}
    if not required.issubset(frame):
        raise ValueError("Rebuild feature parquets for in-play calibration: missing pre-point score metadata.")
    starts = labeled_starts(frame)
    if starts.empty:
        raise ValueError("No opening snapshots with a recorded match winner")
    priors = dict(zip(starts.match_id, zip(*serve_priors(model, starts))))
    engine = PredictionEngine(None, calibrator or MatchWinCalibrator(), None, None,
                              n_mc_samples=200, rng_seed=42)
    results = []
    for mid, group in frame.groupby("match_id", sort=True):
        if mid not in priors:
            continue
        group = group.sort_values("point_idx")
        if not np.array_equal(group.point_idx.to_numpy(), np.arange(len(group))):
            raise ValueError(f"Noncontiguous point sequence for {mid}")
        engine.start_with_priors(*priors[mid], server=int(group.server.iloc[0]))
        for row in group.itertuples(index=False):
            if row.point_idx % stride == 0:
                state = score_state(row)
                pred = engine.predict(state)
                results.append((mid,row.point_idx,pred.raw_win_prob_p1,pred.win_prob_p1,
                                int(row.match_winner==1)))
                if score_baseline:
                    results[-1] += (match_win_prob(state, TOUR_AVERAGES["srv_total_won_pct"]),)
            engine.record_point(bool(row.server_won), int(row.server))
    columns = ["match_id", "points_observed", "raw", "prob", "outcome"]
    snapshots = pd.DataFrame(results, columns=columns + (["score_only"] if score_baseline else []))
    snapshots["weight"] = 1 / snapshots.groupby("match_id").match_id.transform("size")
    return snapshots
