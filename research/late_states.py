"""How often does the leader win from each late-match score? ATP point history only.

A "clinching set" is one the leader would win the match by taking. For every
game boundary in such a set where one player is ahead in games, record the
score, who serves next, and whether that player went on to win the match. The
table is the fair price a late-match quote has to beat after fees.
"""
from collections import defaultdict
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from research.api_cost import ROOT, save
from research.fifteen_forty import fixtures, wilson
from tennis.fixture_history import MAJOR_SURFACES, PLAYERS

OUT = ROOT / "data/processed/late-states"


def states(f):
    """Yield ((leader games, trailer games, leader serves next), leader won match)."""
    winner = PLAYERS.get(f.get("event_winner"))
    if not winner:
        return
    major = f["tournament_name"].strip().lower() in MAJOR_SURFACES
    needed = 3 if major and str(f.get("event_qualification")).lower() not in ("1", "true") else 2
    games = defaultdict(list)
    for g in f["pointbypoint"]:
        parts = g["set_number"].split()
        if len(parts) == 2:
            games[int(parts[1])].append(g)
    sets = [0, 0]
    for number in sorted(games):
        end = (0, 0)
        for g in sorted(games[number], key=lambda g: int(g["number_game"])):
            try:
                end = tuple(int(x) for x in g["score"].split("-"))
                server = PLAYERS[g["player_served"]]
            except (KeyError, ValueError, AttributeError):
                return
            if end == (6, 6) or max(end) > 6 or end[0] == end[1]:
                continue
            lead = 1 if end[0] > end[1] else 2
            if sets[lead-1] != needed-1 or (max(end) >= 6 and abs(end[0]-end[1]) >= 2) or max(end) == 7:
                continue                      # not a clinching set, or the set is already over
            yield (end[lead-1], end[2-lead], server != lead), winner == lead
        if end[0] == end[1]:
            return
        sets[0 if end[0] > end[1] else 1] += 1
        if max(sets) >= needed:
            return


def main():
    table = defaultdict(lambda: [0, 0])
    matches = 0
    for f in fixtures():
        matches += 1
        for key, won in states(f):
            table[key][0] += won
            table[key][1] += 1
    rows = []
    for (lead, trail, serving), (won, n) in sorted(table.items()):
        if lead < 4 or n < 30:
            continue
        low, high = wilson(won, n)
        rows.append(dict(leader_games=lead, trailer_games=trail, leader_serves_next=serving, matches=n,
                         leader_won=won/n, ci95=[low, high],
                         # Highest ask that still breaks even at the lower confidence bound, after the fee.
                         max_ask_cents=max((c for c in range(50, 100) if c/100+.07*c/100*(1-c/100) <= low), default=None)))
    save(OUT/"table.json", dict(matches=matches, rows=rows))
    print(f"{matches} matches")
    for r in rows:
        print(f'{r["leader_games"]}-{r["trailer_games"]} {"serving " if r["leader_serves_next"] else "receiving"} '
              f'n {r["matches"]:4} leader won {r["leader_won"]:.3f} ci [{r["ci95"][0]:.3f}, {r["ci95"][1]:.3f}] '
              f'buy only at or below {r["max_ask_cents"]}c')


if __name__ == "__main__":
    main()
