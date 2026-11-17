"""Convert complete API-Tennis ATP singles fixtures into historical training rows."""
from collections import Counter, defaultdict
import json
from pathlib import Path

import pandas as pd

from tennis.research import RecordedGame


MAJOR_SURFACES = {"wimbledon": "Grass", "roland garros": "Clay", "french open": "Clay",
                  "australian open": "Hard", "us open": "Hard"}
PLAYERS = {"First Player": 1, "Second Player": 2}


def _pair(score):
    if not isinstance(score, str):
        raise ValueError("missing score")
    left, right = score.split("-")
    return int(left), int(right)


def _point_winners(game, server, game_winner, tiebreak, target):
    score = [0, 0]
    points = []

    def completed(next_score):
        return max(next_score) >= target and abs(next_score[0] - next_score[1]) >= 2

    def display(next_score):
        if tiebreak:
            return tuple(map(str, next_score))
        if min(next_score) >= 3:
            if next_score[0] == next_score[1]:
                return ("40", "40")
            return ("A", "40") if next_score[0] > next_score[1] else ("40", "A")
        return tuple(("0", "15", "30", "40")[n] for n in next_score)

    def add(winner):
        serving = server
        if tiebreak and ((len(points) + 1) // 2) % 2:
            serving = 3 - server
        points.append((serving, int(serving == winner)))
        score[winner - 1] += 1

    for idx, point in enumerate(game["points"], 1):
        if int(point["number_point"]) != idx or completed(score):
            raise ValueError("nonsequential point log")
        shown = tuple(v.strip().upper().replace("AD", "A") for v in str(point["score"]).split("-"))
        for i in (0, 1):
            candidate = score.copy()
            candidate[i] += 1
            if (tiebreak or not completed(candidate)) and display(candidate) == shown:
                add(i + 1)
                break
        else:
            raise ValueError("missing or ambiguous point score")
    if game_winner is None:
        return points
    # Standard game logs omit the winning point. Infer exactly one only
    # when the last observed score was game point for the recorded winner.
    if not completed(score):
        add(game_winner)
    if not completed(score) or score[game_winner - 1] <= score[2 - game_winner]:
        raise ValueError("incomplete game")
    return points


def parse_fixture(m):
    winner = PLAYERS[m["event_winner"]]
    name = m["tournament_name"].strip().lower()
    surface = MAJOR_SURFACES.get(name, m.get("surface") or "Unknown")
    major = name in MAJOR_SURFACES
    qualifying = str(m.get("event_qualification")).lower() in {"1", "true"}
    best_of = 5 if major and not qualifying else 3
    scores = {int(s["score_set"]): tuple(int(str(s[key]).split(".")[0]) for key in ("score_first", "score_second"))
              for s in m["scores"]}
    sets = [0, 0]
    games = []
    previous = (0, 0)
    set_number = 1
    # Tiebreak points are top-level game-like records; the ordinary 13th
    # game may appear interleaved with them and contain a blank score.
    tb_rows = defaultdict(list)
    regular_games = []
    for entry in m["pointbypoint"]:
        parts = entry["set_number"].split()
        if len(parts) == 3 and parts[2].lower() == "tiebreak":
            tb_rows[int(parts[1])].append(entry)
        else:
            regular_games.append(entry)
    for game in sorted(regular_games, key=lambda g: (int(g["set_number"].removeprefix("Set ")), int(g["number_game"]))):
        sn = int(game["set_number"].removeprefix("Set "))
        if sn != set_number:
            if sn != set_number + 1 or previous != scores[set_number] or previous[0] == previous[1]:
                raise ValueError("incomplete set")
            sets[int(previous[1] > previous[0])] += 1
            previous = (0, 0)
            set_number = sn
        end = _pair(game["score"])
        delta = (end[0] - previous[0], end[1] - previous[1])
        if delta not in ((1, 0), (0, 1)) or int(game["number_game"]) != sum(previous) + 1:
            raise ValueError("incomplete game log")
        # API fixture importer supports current ATP format. Skip ambiguous
        # historical deciding sets at 6-all for majors prior to harmonization.
        deciding = sum(sets) == best_of - 1
        tb = previous == (6, 6)
        if tb and deciding and major and m["event_date"] < "2022-05-01":
            raise ValueError("historical deciding set rules")
        target = (10 if major and deciding else 7) if tb else 4
        if tb and tb_rows[sn]:
            points = []
            running = (0, 0)
            for idx, point in enumerate(sorted(tb_rows[sn], key=lambda g: int(g["number_game"])), 1):
                current = _pair(point["score"])
                change = (current[0] - running[0], current[1] - running[1])
                if int(point["number_game"]) != idx or change not in ((1, 0), (0, 1)):
                    raise ValueError("incomplete tiebreak")
                serving = PLAYERS[point["player_served"]]
                points.append((serving, int(serving == 1 + change[1])))
                running = current
            if max(running) < target or abs(running[0] - running[1]) < 2 or int(running[1] > running[0]) != delta[1]:
                raise ValueError("incomplete tiebreak")
        else:
            points = _point_winners(game, PLAYERS[game["player_served"]], 1 + delta[1], tb, target)
        games.append(RecordedGame(sn, points, tb))
        previous = end
    if previous != scores[set_number] or previous[0] == previous[1]:
        raise ValueError("incomplete last set")
    sets[int(previous[1] > previous[0])] += 1
    if tuple(sets) != _pair(m["event_final_result"]) or sets[winner - 1] != best_of // 2 + 1:
        raise ValueError("match score disagrees with point log")

    p1, p2 = str(m["first_player_key"]), str(m["second_player_key"])
    match = dict(match_id=str(m["event_key"]), player1_id=p1, player2_id=p2,
                 winner=winner, match_date=m["event_date"], best_of=best_of,
                 games=games, surface=surface, tourney_level="G" if major else "O")
    ids = (p1, p2) if winner == 1 else (p2, p1)
    row = dict(winner_id=ids[0], loser_id=ids[1], tourney_date=m["event_date"].replace("-", ""),
               surface=surface, tourney_level=match["tourney_level"])
    for prefix, pid in zip(("w", "l"), ids):
        stats = {s["stat_name"].lower(): s for s in m.get("statistics") or []
                 if str(s["player_key"]) == pid and s["stat_period"] == "match"}
        first = stats.get("1st serve points won", {})
        second = stats.get("2nd serve points won", {})
        values = [entry.get(key) for entry, key in ((first, "stat_total"), (first, "stat_won"),
                                                    (second, "stat_total"), (second, "stat_won"))]
        if all(v is not None for v in values):
            f_total, f_won, s_total, s_won = map(int, values)
            row.update({prefix + "_svpt": f_total + s_total, prefix + "_1stIn": f_total,
                        prefix + "_1stWon": f_won, prefix + "_2ndWon": s_won})
    return row, match


def load_fixtures(directory: Path):
    history, matches, excluded = [], {}, Counter()
    for path in sorted(directory.glob("????-??-??.json")):
        for m in json.loads(path.read_text()):
            if m.get("event_status") != "Finished" or m.get("event_type_type") != "Atp Singles":
                excluded["outside finished ATP singles"] += 1
                continue
            if str(m["event_key"]) in matches:
                continue
            try:
                row, match = parse_fixture(m)
            except (ValueError, TypeError, KeyError, IndexError) as exc:
                excluded[str(exc)] += 1
                continue
            history.append(row)
            matches[match["match_id"]] = match
    print(f"Accepted {len(matches)} complete ATP singles matches. Exclusions: {dict(excluded)}")
    if not matches:
        raise ValueError("No complete ATP singles point histories in " + str(directory))
    return pd.DataFrame(history), list(matches.values())
