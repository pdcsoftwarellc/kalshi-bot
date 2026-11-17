"""Fit an October first-break adjustment and evaluate it on later ATP matches."""
import argparse
from collections import Counter
from datetime import date, timedelta
import json
import math
from pathlib import Path
import sys

import httpx
import numpy as np
from scipy.optimize import minimize
from scipy.special import expit, logit

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from research.api_cost import ROOT, COLLECTION, OUTPUT, same_player, save, timestamp, quotes_at
from research.forward_odds import tennis

OUT = ROOT / "data/processed/first-break"
SIDES = {"First Player": 1, "Second Player": 2}


def first_break(games):
    """Return the first confirmed break. Only use a contiguous prefix of set one."""
    previous = (0, 0)
    for i, game in enumerate(games[:6], 1):
        if str(game.get("number_game")) != str(i) or game.get("serve_winner") not in SIDES:
            return None
        server = SIDES.get(game.get("player_served"))
        if server is None:
            return None
        parts = str(game.get("score") or "").split("-")
        if len(parts) != 2 or not all(x.strip().isdigit() for x in parts):
            return None
        score = tuple(int(x.strip()) for x in parts)
        winner = SIDES[game["serve_winner"]]
        expected = (previous[0]+int(winner==1), previous[1]+int(winner==2))
        if score != expected:
            return None
        if server != winner:
            return dict(game=i, broken=server, games_down=score[winner-1]-score[server-1],
                        next_server=winner)
        previous = score
    return None


def regular_games(row):
    return sorted((g for g in (row.get("pointbypoint") or []) if g.get("set_number")=="Set 1"),
                  key=lambda g:int(g["number_game"]))


def fixtures(client):
    directory = ROOT/"data/raw/first-break-history"
    day = date(2026, 9, 29)
    end = date(2026, 11, 16)
    rows = {}
    while day <= end:
        last = min(day+timedelta(days=6),end)
        cache = directory/f"{day}-{last}.json"
        if not cache.exists():
            batch = tennis(client, "get_fixtures", date_start=str(day),date_stop=str(last),
                           timezone="UTC",event_type_key=265)
            values = []
            for f in batch:
                if f.get("event_type_type") != "Atp Singles":
                    continue
                games = regular_games(f)
                values.append(dict(id=str(f["event_key"]), date=f["event_date"], status=f["event_status"],
                    players=[f["event_first_player"],f["event_second_player"]], winner=SIDES.get(f.get("event_winner")),
                    first_break=first_break(games), first_set_games=len(games)))
            save(cache,values)
        rows.update({v["id"]:v for v in json.loads(cache.read_text())})
        day = last+timedelta(days=1)
    return rows


def features(event):
    return [1., (event["game"]-3.5)/2.5, event["games_down"]-1.5]


def probability(prior,event,coefficients):
    return float(expit(logit(prior)+np.dot(features(event),coefficients)))


def fit(rows):
    X = np.array([features(r) for r in rows])
    offset = logit([r["prior"] for r in rows])
    y = np.array([r["won"] for r in rows])
    def loss(b):
        z = offset + X@b
        return (np.logaddexp(0,z)-y*z).sum() + np.dot(b,b)/2, X.T@(expit(z)-y)+b
    result = minimize(loss,np.zeros(3),jac=True,method="BFGS")
    if not result.success:
        raise RuntimeError(f"Fit failed: {result.message}")
    return result.x.tolist()


def score(rows,coef):
    if not rows:
        return dict(n=0)
    y = np.array([r["won"] for r in rows])
    p = np.array([probability(r["prior"],r,coef) for r in rows])
    return dict(n=len(y), wins=int(y.sum()), mean_model=float(p.mean()),
                brier=float(np.mean((y-p)**2)), prematch_brier=float(np.mean((y-[r["prior"] for r in rows])**2)))


def run(client):
    history = fixtures(client)
    markets = {}
    scheduled = {}
    for path in COLLECTION.glob("*.json"):
        data = json.loads(path.read_text())
        if isinstance(data,list):
            scheduled.update({str(m["event_key"]):m for m in data if isinstance(m,dict) and m.get("event_time") and "event_key" in m})
        if isinstance(data,dict):
            markets.update({m["ticker"]:m for m in data.get("markets",[])})
    splits = {"train":[], "test":[]}
    late_splits = {"train":[], "test":[]}
    skipped = Counter()
    seen = set()
    for path in sorted(OUTPUT.glob("*-ledger.json")):
        for row in json.loads(path.read_text())["predictions"]:
            f = history.get(str(row["fixture"]))
            m = markets[row["ticker"]]
            if not f or f["status"]!="Finished" or not f["first_break"] or not f["winner"]:
                skipped["no_eligible_break_or_outcome"]+=1
                continue
            orientation = [i for i,name in enumerate(f["players"],1) if same_player(name,m["yes_sub_title"])]
            if len(orientation)!=1 or row["outcome"] != int(f["winner"]==orientation[0]):
                skipped["ambiguous_orientation_or_settlement"]+=1
                continue
            if f["id"] in seen or not 0 < row["mid"] < 1:
                skipped["duplicate_or_invalid_prior"]+=1
                continue
            event = f["first_break"]
            prior = row["mid"] if event["broken"]==orientation[0] else 1-row["mid"]
            split = "train" if f["date"] < "2026-11-01" else "test"
            if (split=="train" and timestamp(m["settlement_ts"])>=timestamp("2026-11-01T00:00:00Z")) or f["date"]>"2026-11-15":
                skipped["outside_split"]+=1
                continue
            seen.add(f["id"])
            splits[split].append(dict(**event, fixture=f["id"], date=f["date"], prior=prior,
                                     won=int(event["broken"]==f["winner"])))
            clock = scheduled.get(f["id"])
            if clock and m.get("occurrence_datetime"):
                anchor = int(min(timestamp(clock["event_date"]+"T"+clock["event_time"]+"+00:00"),
                                 timestamp(m["occurrence_datetime"]))//60)*60-1800
                late = quotes_at(client,m["ticker"],anchor) if anchor >= timestamp(m["open_time"]) else None
                if late:
                    mid = float((late[0]+late[1])/2)
                    late_splits[split].append(dict(splits[split][-1], prior=mid if event["broken"]==orientation[0] else 1-mid))
    if not splits["train"]:
        raise RuntimeError("No matched October training rows")
    coef = fit(splits["train"])
    report = dict(coefficients=coef, skipped=skipped, strong_favorite_min=.60, max_game=6,
                  fee_adjusted_min_edge=.05, delay_seconds=2,
                  **{k:{"all":score(v,coef), "favorites":score([r for r in v if r["prior"]>=.60],coef)} for k,v in splits.items()})
    if late_splits["train"]:
        late_coef = fit(late_splits["train"])
        report["before_start"] = dict(coefficients=late_coef,
            **{k:{"all":score(v,late_coef),"favorites":score([r for r in v if r["prior"]>=.60],late_coef)}
               for k,v in late_splits.items()})
        save(OUT/"historical-breaks-before-start.json",late_splits)
    save(OUT/"fit.json",report)
    save(OUT/"historical-breaks.json",splits)
    print(json.dumps(report,indent=2))


if __name__ == "__main__":
    argparse.ArgumentParser(description=__doc__).parse_args()
    with httpx.Client(timeout=45) as client:
        run(client)
