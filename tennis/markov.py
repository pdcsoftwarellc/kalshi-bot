"""Point, game, set and match probabilities under constant player-specific serve rates.

Each recurrence works on scalars or on a whole vector of posterior draws. Scores
are counts in player order, and state.server identifies the NEXT point's server.
"""
from functools import lru_cache
import numpy as np
from tennis.types import MatchFormat, MatchState


def _prob(x):
    return np.clip(x, 1e-9, 1 - 1e-9)


def _deuce(a, b):
    # From a tie, win the next pair, lose it, or return to an equivalent tie.
    win = a * b
    return win / (win + (1-a)*(1-b))


def _race(s, r, target, next_prob, tied):
    """Race to target with a two-point margin, with an exact deuce tail."""
    @lru_cache(None)
    def win(i, j):
        if max(i, j) >= target and abs(i-j) >= 2:
            return float(i > j)
        if min(i, j) >= target-1:
            if i == j:
                return tied
            p = next_prob(i+j)
            return p + (1-p)*tied if i > j else p*tied
        p = next_prob(i+j)
        return p*win(i+1, j) + (1-p)*win(i, j+1)
    result = win(s, r)
    win.cache_clear()
    return result


def _game(p):
    q = 1-p
    return p**4 * (1 + 4*q + 10*q*q) + 20*p**3*q**3*_deuce(p,p)


def game_win_prob(p: float) -> float:
    return float(_game(_prob(p)))


def game_win_prob_from_score(p: float, pts_server: int, pts_returner: int) -> float:
    p = _prob(p)
    return float(_race(pts_server, pts_returner, 4, lambda _: p, _deuce(p,p)))


def _tiebreak(serve, receive, s, r, first_server, target):
    # Point indices 0,3,4,7,8,... are served by the first server of the tiebreak.
    def next_prob(played):
        p1_serves = ((played % 4 in (0, 3)) == (first_server == 0))
        return serve if p1_serves else receive
    return _race(s, r, target, next_prob, _deuce(serve, receive))


def tiebreak_win_prob_from_score(p_odd: float, pts_server: int, pts_returner: int,
                                 p_even: float | None = None) -> float:
    p = _prob(p_odd)
    return float(_tiebreak(p, 1-p if p_even is None else _prob(p_even),
                           pts_server, pts_returner, 0, 7))


def tiebreak_win_prob(p_odd: float, p_even: float | None = None) -> float:
    return tiebreak_win_prob_from_score(p_odd, 0, 0, p_even)


def _set(s, r, server, hold, brk, tb, current_game=None):
    """P(P1 wins set); tb is P(P1 wins tiebreak), or None for an advantage set."""
    @lru_cache(None)
    def win(i, j):
        if max(i,j) >= 6 and abs(i-j) >= 2:
            return float(i > j)
        who = server ^ ((i+j-s-r) % 2)
        p = hold if who == 0 else brk
        if i == s and j == r and current_game is not None:
            p = current_game
        if min(i,j) >= 6:
            if tb is not None:
                return tb if i == j else float(i > j)
            tied = _deuce(hold, brk)
            if i == j:
                # The current partially played game may have a different P.
                q = brk if who == 0 else hold
                return p*(q+(1-q)*tied)+(1-p)*q*tied
            return p+(1-p)*tied if i>j else p*tied
        return p*win(i+1,j) + (1-p)*win(i,j+1)
    result = win(s,r)
    win.cache_clear()
    return result


def _invert_game_win_prob(g):
    lo, hi = 0., 1.
    for _ in range(40):
        mid = (lo+hi)/2
        if _game(mid) < g:
            lo = mid
        else:
            hi = mid
    return (lo+hi)/2


def _target(fmt, deciding):
    if deciding and (fmt.final_set_advantage or not fmt.final_set_tiebreak):
        return None
    return (10 if fmt.best_of == 5 else 7) if deciding else 7


def set_win_prob_from_score(g: float, games_server: int, games_returner: int,
                            match_format: MatchFormat, is_final_set: bool,
                            g_opp: float | None = None) -> float:
    hold = float(_prob(g))
    brk = 1-hold if g_opp is None else float(_prob(g_opp))
    target = _target(match_format, is_final_set)
    tb = None if target is None else _tiebreak(_invert_game_win_prob(hold),
                                               _invert_game_win_prob(brk), 0, 0, 0, target)
    return float(_set(games_server, games_returner, (games_server+games_returner)%2, hold, brk, tb))


def set_win_prob(g: float, g_opp: float | None = None, match_format: MatchFormat | None = None,
                 is_final_set: bool = False) -> float:
    return set_win_prob_from_score(g, 0, 0, match_format or MatchFormat(), is_final_set, g_opp)


def match_win_prob_batch(state: MatchState, p_serves: np.ndarray,
                         opponent_serves: np.ndarray | None = None) -> np.ndarray:
    """P(P1 wins), with paired draws for the current server and the other player.

    Omitting opponent_serves retains the single-rate, equally skilled baseline.
    """
    p = _prob(np.asarray(p_serves, dtype=float))
    other = p if opponent_serves is None else _prob(np.asarray(opponent_serves, dtype=float))
    serve, receive = (p, 1-other) if state.server == 0 else (other, 1-p)
    hold, brk = _game(serve), _game(receive)
    needed = state.sets_needed()
    if max(state.sets_p1, state.sets_p2) >= needed:
        return np.full_like(p,float(state.sets_p1 >= needed))

    @lru_cache(None)
    def tb_start(target):
        if target is None:
            return None
        return _tiebreak(serve,receive,0,0,0,target)

    target = _target(state.match_format, state.is_final_set())
    s, r = state.points_p1, state.points_p2
    if target and (state.is_tiebreak or state.games_p1 == state.games_p2 == 6):
        first = state.server ^ int((s+r)%4 in (1,2))
        current_set = _tiebreak(serve,receive,s,r,first,target)
    else:
        point = serve if state.server == 0 else receive
        current_game = _race(s,r,4,lambda _:point,_deuce(point,point))
        current_set = _set(state.games_p1,state.games_p2,state.server,hold,brk,tb_start(target),current_game)

    @lru_cache(None)
    def whole_set(target):
        # Starting service order does not change P(winning a *fresh* set):
        # the first ten games contain five service games for each player;
        # after 5-5 each two-game block contains one of each.
        return _set(0,0,0,hold,brk,tb_start(target))

    @lru_cache(None)
    def future(a,b):
        if a >= needed or b >= needed:
            return float(a >= needed)
        ps = whole_set(_target(state.match_format,a+b == 2*(needed-1)))
        return ps*future(a+1,b)+(1-ps)*future(a,b+1)

    a,b = state.sets_p1,state.sets_p2
    result = current_set*future(a+1,b)+(1-current_set)*future(a,b+1)
    return np.broadcast_to(np.clip(result, 0, 1), p.shape)


def match_win_prob(state: MatchState, p_serve: float, opponent_serve: float | None = None) -> float:
    return float(match_win_prob_batch(state, np.asarray(p_serve), opponent_serve))
