"""Source game boundaries and pre-point snapshots for historical singles matches."""
from dataclasses import dataclass

import numpy as np
import pandas as pd

from tennis.features import FEATURE_COLUMNS
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


def opening_predictions(model, frame, default_format):
    """Return opening rows and raw P(player 1 wins), aligned by position."""
    starts = labeled_starts(frame)
    if starts.empty:
        return starts, np.array([])
    priors = model.predict_proba(starts[FEATURE_COLUMNS].to_numpy(dtype=np.float32))
    raw = [match_win_prob(opening_state(row, default_format), float(prior))
           for (_, row), prior in zip(starts.iterrows(), priors, strict=True)]
    return starts, np.array(raw)
