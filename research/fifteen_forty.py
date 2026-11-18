"""Test the 15-40 rule: back the receiver at 15-40, close at deuce or the next game.

Historical point logs have no receipt times, so this file measures (a) what
servers actually do from 15-40 and (b) what the round trip is worth when both
legs trade at model-fair prices and pay Kalshi taker costs. Executable prices
come from `fifteen_forty_tape.py` on synchronized recordings.
"""
from collections import defaultdict
import json
import math
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from research.api_cost import ROOT, save
from tennis.fixture_history import PLAYERS, _point_winners
from tennis.markov import match_win_prob
from tennis.types import MatchFormat, MatchState

OUT = ROOT / "data/processed/fifteen-forty"
SOURCES = ("api_tennis", "native-atp", "set-frequency-atp")


def fixtures():
    seen = set()
    for source in SOURCES:
        for path in sorted((ROOT / "data/raw" / source).glob("*.json")):
            for f in json.loads(path.read_text()):
                key = f.get("event_key")
                if (key in seen or f.get("event_status") != "Finished" or not f.get("pointbypoint")
                        or f.get("event_type_type") != "Atp Singles"):
                    continue
                seen.add(key)
                yield f


def games(f):
    """Yield (server, [server_won, ...]) for each cleanly decoded ordinary game."""
    for g in f["pointbypoint"]:
        if len(g["set_number"].split()) != 2:
            continue
        try:
            server, winner = PLAYERS[g["player_served"]], PLAYERS[g["serve_winner"]]
            points = _point_winners(g, server, winner, False, 4)
        except (KeyError, TypeError, ValueError):
            continue
        yield (f["event_key"], server), [won for _, won in points]


def wilson(k, n, z=1.96):
    if not n:
        return None
    p, d = k/n, 1+z*z/n
    half = z*math.sqrt(p*(1-p)/n+z*z/(4*n*n))/d
    return [(p+z*z/(2*n))/d-half, (p+z*z/(2*n))/d+half]


def historical():
    totals = defaultdict(lambda: [0, 0])       # (fixture, server) -> [won, played]
    hits = []
    n_games = n_fixtures = 0
    for f in fixtures():
        n_fixtures += 1
        for key, wins in games(f):
            n_games += 1
            totals[key][0] += sum(wins)
            totals[key][1] += len(wins)
            s = r = 0
            for i, won in enumerate(wins):
                if (s, r) == (1, 3):
                    rest = wins[i:]
                    direct = not (rest[0] and rest[1])       # receiver took one of the next two
                    hits.append(dict(key=key, game_won=sum(wins), game_n=len(wins), next=rest[:2],
                                     direct_break=direct, held=sum(wins) > len(wins)-sum(wins)))
                    break
                s, r = s+won, r+(not won)
    n = len(hits)
    direct = sum(h["direct_break"] for h in hits)
    held = sum(h["held"] for h in hits)
    # Compare the points played at 15-40 / 30-40 with the same server's rate
    # in the rest of that match (the pressure game itself is left out).
    won = played = 0
    baseline = []
    for h in hits:
        w, t = totals[h["key"]]
        w, t = w-h["game_won"], t-h["game_n"]
        pts = h["next"][:1] if not h["next"][0] else h["next"]
        if t < 20:
            continue
        won += sum(pts)
        played += len(pts)
        baseline += [w/t]*len(pts)
    p = sum(w for w, _ in totals.values())/sum(t for _, t in totals.values())
    return dict(fixtures=n_fixtures, ordinary_games=n_games, games_reaching_15_40=n,
                share_of_service_games=n/n_games, tour_serve_point_rate=p,
                receiver_breaks_before_deuce=dict(rate=direct/n, ci95=wilson(direct, n)),
                reaches_deuce=dict(rate=1-direct/n, iid_at_tour_rate=p*p),
                server_eventually_holds=dict(rate=held/n, ci95=wilson(held, n)),
                pressure_points=dict(points=played, server_won=won/played, ci95=wilson(won, played),
                                     same_servers_rest_of_match=sum(baseline)/len(baseline)))


def fee(price, contracts):
    """Kalshi general taker fee per contract: 7% * p * (1-p), order total rounded up to a cent."""
    return math.ceil(round(7*contracts*price*(1-price), 6))/100/contracts


def fair_trade(p_server, p_receiver, games, sets=(0, 0), half_spread=.01, contracts=1):
    """One round trip on the receiver at model prices; player 1 serves."""
    def value(points, games_now, server=0):
        state = MatchState(sets_p1=sets[0], sets_p2=sets[1], games_p1=games_now[0], games_p2=games_now[1],
                           points_p1=points[0], points_p2=points[1], server=server,
                           match_format=MatchFormat.best_of_3_tiebreak())
        return 1-match_win_prob(state, p_server if server == 0 else p_receiver,
                                p_receiver if server == 0 else p_server)
    entry = value((1, 3), games)
    deuce = value((3, 3), games)
    broken = value((0, 0), (games[0], games[1]+1), server=1)
    q = 1-p_server
    p_break = q+p_server*q
    buy, win, lose = entry+half_spread, broken-half_spread, deuce-half_spread
    costs = fee(buy, contracts)+p_break*fee(win, contracts)+(1-p_break)*fee(lose, contracts)
    net = p_break*win+(1-p_break)*lose-buy-costs
    # How far the market's break chance at 15-40 must sit below the truth to pay those costs.
    slope = broken-deuce
    return dict(receiver_price_at_15_40=entry, after_break=broken, at_deuce=deuce,
                gain_if_break=broken-entry, loss_if_deuce=deuce-entry, p_break_before_deuce=p_break,
                gross_at_fair_mid=p_break*broken+(1-p_break)*deuce-entry,
                net_per_contract=net, required_break_mispricing=-net/slope)


def main():
    report = dict(historical=historical(), fair_value={})
    cases = {"even match, 2-2 first set": (.64, .64, (2, 2), (0, 0)),
             "even match, 4-4 first set": (.64, .64, (4, 4), (0, 0)),
             "even match, 4-4 deciding set": (.64, .64, (4, 4), (1, 1)),
             "weak server (underdog), 2-2": (.58, .68, (2, 2), (0, 0)),
             "strong server (favorite), 2-2": (.68, .58, (2, 2), (0, 0))}
    for name, (ps, pr, g, s) in cases.items():
        report["fair_value"][name] = {
            "one contract, 1c half-spread": fair_trade(ps, pr, g, s),
            "100 contracts, 1c half-spread": fair_trade(ps, pr, g, s, contracts=100)["net_per_contract"],
            "100 contracts, no spread": fair_trade(ps, pr, g, s, half_spread=0, contracts=100)["net_per_contract"]}
    save(OUT/"historical.json", report)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
