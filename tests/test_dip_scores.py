from research import dip_scores as study

A, B = "id-a", "id-b"


def state(one, two, points=(0, 0), server=A, status="live", winner="", advantage=""):
    """one/two: per-set (games, outcome) for competitor 1 and 2."""
    return dict(status=status, competitor1_id=A, competitor2_id=B, server=server, winner=winner, advantage=advantage,
                competitor1_round_scores=[dict(score=g, outcome=o) for g, o in one],
                competitor2_round_scores=[dict(score=g, outcome=o) for g, o in two],
                competitor1_current_round_score=points[0], competitor2_current_round_score=points[1])


def row(ts, bid, ask, st, event="KXATPMATCH-26NOV17RONBAR", ticker="KXATPMATCH-26NOV17RONBAR-BAR"):
    return dict(ts=ts, event=event, ticker=ticker, yes="B", start=None, state=st,
                quote=dict(yes_bid_dollars=f"{bid/100:.4f}", yes_ask_dollars=f"{ask/100:.4f}"))


def test_lost_first_set_and_down_a_break_from_the_favourites_side():
    # Competitor 2 is the favourite: lost the first set 4-6, trails 1-2 and is about to serve.
    s = study.score_state(state([(6, "winner"), (2, "ongoing")], [(4, "loser"), (1, "ongoing")], server=B), fav=2)
    assert (s["sets_fav"], s["sets_dog"], s["games_fav"], s["games_dog"], s["set_number"]) == (0, 1, 1, 2, 2)
    assert s["lost_last_set"] and not s["just_lost_set"] and s["break_diff"] == 0 and s["text"] == "4-6 1-2"
    # Same score with the opponent serving: the favourite has served twice and won one game.
    s = study.score_state(state([(6, "winner"), (2, "ongoing")], [(4, "loser"), (1, "ongoing")], server=A), fav=2)
    assert s["break_diff"] == -1 and (s["sets_situation"], s["break_situation"]) == ("behind in sets", "down a break")
    # First game of the next set: "just lost the set".
    s = study.score_state(state([(6, "winner"), (0, "ongoing")], [(4, "loser"), (0, "ongoing")], server=B), fav=2)
    assert s["just_lost_set"] and s["sets_situation"] == "behind in sets, just lost the set"


def test_break_point_tiebreak_and_finished_match():
    s = study.score_state(state([(3, "ongoing")], [(3, "ongoing")], points=(30, 40), server=A), fav=1)
    assert s["facing_break_point"] and s["break_situation"] == "on serve, facing break point"
    assert study.score_state(state([(3, "ongoing")], [(3, "ongoing")], points=(30, 40), server=A), fav=2)["has_break_point"]
    assert not study.score_state(state([(3, "ongoing")], [(3, "ongoing")], points=(40, 40), server=A), fav=1)["facing_break_point"]
    s = study.score_state(state([(6, "ongoing")], [(6, "ongoing")], points=(2, 4), server=""), fav=1)
    assert s["tiebreak"] and s["break_situation"] == "tiebreak" and s["fav_serving"] is None
    done = state([(6, "winner"), (6, "winner")], [(3, "loser"), (4, "loser")], server="", status="closed", winner=A)
    assert study.winner(done) == 1 and study.score_state(done, 1)["sets_situation"] == "match over"


def test_yes_side_reference_price_and_dip():
    assert study.code_side("KXATPMATCH-26NOV17RONBAR", "KXATPMATCH-26NOV17RONBAR-BAR") == 2
    assert study.code_side("KXATPMATCH-26NOV17RONBAR", "KXATPMATCH-26NOV17RONBAR-RON") == 1
    assert study.code_side("KXATPMATCH-26NOV17BARBAR", "KXATPMATCH-26NOV17BARBAR-BAR") is None
    # YES is competitor 2 at 8/9c, so competitor 1 is a 91.5c favourite; it loses the set and dips, then wins.
    rows = [row(0, 8, 9, state([(0, "ongoing")], [(0, "ongoing")])),
            row(20, 9, 10, state([(1, "ongoing")], [(0, "ongoing")], server=B)),
            row(900, 21, 22, state([(4, "loser"), (0, "ongoing")], [(6, "winner"), (0, "ongoing")])),
            row(3000, 0, 1, state([(4, "loser"), (6, "winner"), (6, "winner")], [(6, "winner"), (2, "loser"), (1, "loser")],
                                  server="", status="closed", winner=A))]
    m = study.match(rows[0]["event"], rows)
    assert m["usable"] and m["yes_competitor"] == 2 and m["favourite_competitor"] == 1 and not m["favourite_is_yes"]
    assert m["reference_price"] == 91.5 and m["reference_kind"] == "first_game" and m["favourite_won"] is True
    dip = next(d for d in m["dips"] if d["level_name"] == "80")
    assert dip["ts"] == 900 and dip["fav_ask"] == 79 and dip["after_gap"] and dip["score"]["just_lost_set"]
    # A settled price that contradicts the code position is a conflict, not a guess.
    rows[-1] = row(3000, 99, 100, rows[-1]["state"])
    assert study.match(rows[0]["event"], rows)["usable"] is False
