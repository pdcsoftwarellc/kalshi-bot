"""Decode the observed API-Tennis point history, in receipt (past-only) context."""
from dataclasses import dataclass
from tennis.fixture_history import PLAYERS, _pair, _point_winners


@dataclass(frozen=True)
class ScoredPoint:
    served_by: int
    won: bool
    sets: tuple[int, int]
    games: tuple[int, int]
    points: tuple[int, int]
    next_server: int
    tiebreak: bool


def _numbered_tiebreak(rows):
    points = []
    previous = (0, 0)
    for n, row in enumerate(sorted(rows,key=lambda r:int(r["number_game"])), 1):
        score = _pair(row["score"])
        change = tuple(x-y for x,y in zip(score,previous))
        if int(row["number_game"]) != n or change not in ((0,1),(1,0)):
            raise ValueError("Incomplete tiebreak history")
        server = PLAYERS[row["player_served"]]
        points.append((server, int(server == 1+change[1])))
        previous = score
    return points


def observed_points(history, best_of):
    """Return post-point scores; exclude the winning point until a game is confirmed."""
    groups = {}
    tie_rows = {}
    for row in history:
        words = str(row["set_number"]).lower().split()
        if len(words) not in (2,3) or words[0] != "set":
            raise ValueError("Unknown set format")
        sn = int(words[1])
        if len(words) == 3:
            if words[2] != "tiebreak":
                raise ValueError("Unknown tiebreak format")
            tie_rows.setdefault(sn,[]).append(row)
        else:
            key = sn, int(row["number_game"])
            if key in groups:
                raise ValueError("Duplicate game")
            groups[key] = row
    # A live tiebreak may have point-level entries before its ordinary game
    # summary appears. It follows twelve completed regular games.
    for sn, rows in tie_rows.items():
        groups.setdefault((sn,13), dict(player_served=rows[0]["player_served"],points=[]))
    sets, games = [0,0], [0,0]
    result = []
    set_number = 1
    for (sn, gn), game in sorted(groups.items()):
        if sn != set_number or gn != sum(games)+1:
            raise ValueError("Incomplete game history")
        server = PLAYERS[game["player_served"]]
        winner = PLAYERS.get(game.get("serve_winner"))
        tie = games == [6,6]
        target = (10 if best_of == 5 and sum(sets) == 4 else 7) if tie else 4
        if tie and sn in tie_rows:
            pts = _numbered_tiebreak(tie_rows[sn])
        else:
            pts = _point_winners(dict(points=game.get("points") or []), server, winner, tie, target)

        if winner:
            end_games = games.copy()
            end_games[winner-1] += 1
            if tuple(end_games) != _pair(game.get("score")):
                raise ValueError("Game totals disagree")
        tally = [0,0]
        for i,(s,won) in enumerate(pts, 1):
            who = s-1 if won else 2-s
            tally[who] += 1
            done = max(tally) >= target and abs(tally[0]-tally[1]) >= 2
            if done != bool(winner and i == len(pts)) or (done and who != winner-1):
                raise ValueError("Unconfirmed game boundary")
            next_server = s-1
            if tie and i%2:
                next_server = 1-next_server
            display = tally.copy()
            if not tie and min(display)>=3:
                offset = min(display)-3
                display = [v-offset for v in display]
            if done:
                games[who] += 1
                next_server = 1-(pts[0][0]-1)
                if tie or (max(games)>=6 and abs(games[0]-games[1])>=2):
                    sets[who] += 1
                    games = [0,0]
                    set_number += 1
                display = [0,0]
            result.append(ScoredPoint(s-1,bool(won),tuple(sets),tuple(games),tuple(display),
                                      next_server, games == [6,6]))
    return result
