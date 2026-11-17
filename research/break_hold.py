"""Test the first break-point service game of a prematch favorite.

Historical fixture logs label the game and winner. Only matched receipt tapes can
establish the ordering of a completed service game and an executable rebound.
"""
import argparse
from collections import Counter, defaultdict
from datetime import date, timedelta
from decimal import Decimal as D
import json
from pathlib import Path
import sys

import httpx

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from research.api_cost import ROOT, COLLECTION, PERIODS, fee, same_player, save, timestamp
from research.forward_odds import tennis
from research.score_break import Books, SECOND, at_game_boundary
from research.price_drift import read_quote
from research.favorite_rebound import choose_favorite, signal
from tennis.fixture_history import PLAYERS, _point_winners

OUT = ROOT / "data/processed/parallel-round2/break_hold"
CACHE = ROOT / "data/raw/break-hold-fixtures"
PRIOR = ROOT / "data/processed/parallel-research"
RULE = {
    "favorite_floor": ".70",
    "game": "first ordinary first-set service game of the favorite to contain a break point",
    "primary": "one contract after a defended game, compared separately to a conceded game",
    "entry": "first prior-rebound-rule minute signal at/after fresh game confirmation; within ten minutes and in the same set",
    "fill": "2s after signal at original limit or better with one contract displayed; require book age <=5s, spread <=4c",
    "exit": "one contract at bid after 8c gain, 8c loss, or 30 min; recheck depth after 2s; open exposure is unresolved",
    "cost": "general taker fee, multiplier one, rounded up to a cent on entry and exit",
}


def break_points(points, server):
    """Count break point opportunities, using the score *before* each point."""
    a = b = faced = 0
    for serving, won in points:
        if serving != server:
            raise ValueError("ordinary game changed server")
        if b >= 3 and b > a:
            faced += 1
        if won:
            a += 1
        else:
            b += 1
    return faced


def fixture_labels(f):
    """Stop at the first incomplete game so an earlier break point cannot be skipped."""
    ordinary = [g for g in f.get("pointbypoint") or [] if g.get("set_number") == "Set 1"]
    labels, score = {}, (0, 0)
    status = "complete_first_set"
    try:
        for index, g in enumerate(sorted(ordinary, key=lambda g:int(g["number_game"])), 1):
            if score == (6, 6):
                break
            if index != int(g["number_game"]):
                raise ValueError("game sequence")
            server = PLAYERS[g["player_served"]]
            winner = PLAYERS[g["serve_winner"]]
            end = tuple(map(int,g["score"].split("-")))
            expected = (score[0]+(winner==1), score[1]+(winner==2))
            if end != expected:
                raise ValueError("game score")
            if server not in labels:
                points = _point_winners(g, server, winner, False, 4)
                count = break_points(points, server)
                if count:
                    labels[server] = dict(game=index, break_points=count, result="hold" if server == winner else "break")
            score = end
        if score != (6,6) and not (max(score)>=6 and abs(score[0]-score[1])>=2):
            status = "incomplete_game_history"
    except (TypeError, ValueError, KeyError, IndexError):
        status = "incomplete_game_history"
    return {str(k):v for k,v in labels.items()}, status


def load_history(client, start, end, tour):
    rows = {}
    day = start
    while day < end:
        last = min(day+timedelta(days=6), end-timedelta(days=1))
        file = CACHE / f"{tour}-{day}-{last}.json"
        if not file.exists():
            batch = tennis(client, "get_fixtures", date_start=str(day),date_stop=str(last),timezone="UTC",
                           event_type_key={"atp":265, "challenger":281}[tour])
            if not isinstance(batch, list) and batch:
                raise ValueError("Unexpected fixture envelope")
            small = []
            for f in batch:
                labels,status = fixture_labels(f)
                small.append(dict(fixture=str(f["event_key"]), players=[f["event_first_player"],f["event_second_player"]],
                                  match_status=f.get("event_status"), winner=PLAYERS.get(f.get("event_winner")),
                                  date=f["event_date"], labels=labels, score_quality=status))
            save(file,small)
        rows.update({m["fixture"]:m for m in json.loads(file.read_text())})
        day = last + timedelta(days=1)
    return rows


def market_names():
    names = {}
    for path in COLLECTION.glob("*.json"):
        data = json.loads(path.read_text())
        if isinstance(data, dict):
            names.update({m["ticker"]:m["yes_sub_title"] for m in data.get("markets",[]) if "yes_sub_title" in m and "ticker" in m})
    return names


def prior_rows(start, tour):
    if tour == "atp":
        data = json.loads((PRIOR/"price_drift"/f"{start}-ledger.json").read_text())["eligible_observations"]
        for r in data:
            mid = (D(r["late_bid"])+D(r["late_ask"]))/2
            yield str(r["fixture"]), r["ticker"], mid, r["late_quote_ts"]
    else:
        data = json.loads((PRIOR/"favorite_rebound"/f"{start}.json").read_text())["ledger"]
        for r in data:
            yield str(r["fixture"]), r.get("ticker"), D(r["prior"]) if "prior" in r else None, r.get("prior_quote_ts")


def historical(client):
    names = market_names()
    allrows, summaries = [], []
    for start,end in PERIODS:
        by_period = []
        for tour in ("atp","challenger"):
            history = load_history(client,start-timedelta(days=1),end+timedelta(days=1),tour)
            counts, seen = Counter(),set()
            rows = []
            for fixture,ticker,mid,quote_ts in prior_rows(start,tour):
                counts["prior_cohort"]+=1
                if fixture in seen:
                    raise ValueError("Duplicate fixture")
                seen.add(fixture)
                if mid is None or max(mid, 1-mid)<D(".70"):
                    continue
                counts["strong_favorites"]+=1
                f = history.get(fixture)
                if not f or not f["winner"] or f["match_status"]!="Finished" or ticker not in names:
                    counts["unresolved_match"]+=1
                    continue
                orientation = [i for i,n in enumerate(f["players"],1) if same_player(n,names[ticker])]
                if len(orientation)!=1:
                    counts["unknown_side"]+=1
                    continue
                favored = orientation[0] if mid>=D(".70") else 3-orientation[0]
                label = f["labels"].get(str(favored))
                if not label:
                    counts[f["score_quality"]+"_without_verified_break_point"]+=1
                    continue
                if not str(start)<=f["date"]<str(end):
                    counts["outside_period"]+=1
                    continue
                counts["matched_first_break_point_games"]+=1
                rows.append(dict(period=str(start),tour=tour,fixture=fixture,ticker=ticker,
                                 favorite=f["players"][favored-1],prior=str(max(mid,1-mid)),prematch_quote_ts=quote_ts,
                                 game_confirmation_ts=None, won=f["winner"]==favored,**label))
            by_period.extend(rows)
            allrows.extend(rows)
            summaries.append(dict(start=str(start),tour=tour,coverage=dict(counts),groups=group_summary(rows)))
        summaries.append(dict(start=str(start),tour="combined",groups=group_summary(by_period)))
    save(OUT/"historical-games.json",allrows)
    return summaries


def group_summary(rows):
    result={}
    for name in ("hold","break"):
        group = [r for r in rows if r["result"]==name]
        wins=sum(r["won"] for r in group)
        result[name] = dict(games=len(group),wins=wins,
                            win_rate=wins/len(group) if group else None,
                            avg_prior=sum(float(r["prior"]) for r in group)/len(group) if group else None)
    return result


def is_pressure(score, favorite):
    parts = str(score).upper().replace("AD","A").replace(" ","").split("-")
    if len(parts)!=2:
        return False
    serving,receiving = (parts if favorite==1 else parts[::-1])
    return ((receiving=="40" and serving in ("0","15","30")) or (receiving=="A" and serving=="40"))


class ServiceGameWatch:
    """Require contemporaneous break-point evidence for a hold, and a known prior game history."""
    def __init__(self, favorite):
        self.favorite = favorite
        self.observed = defaultdict(dict)
        self.pressure = set()
        self.last_open = {}
        self.done = False

    def process(self,msg,now):
        if self.done or msg.get("status")!="Set 1" or msg.get("winner") or str(msg.get("live"))!="1":
            return None
        games = sorted(msg.get("games") or [],key=lambda g:int(g["number_game"]))
        for i,g in enumerate(games[:12],1):
            if int(g["number_game"])!=i or g.get("player_served") not in PLAYERS:
                return None
            if PLAYERS[g["player_served"]]!=self.favorite:
                continue
            n = int(g.get("n_points",0))
            if "points" in g:
                points=g["points"]
                if not isinstance(points,list) or len(points)!=n or any(
                        str(p.get("number_point"))!=str(j) for j,p in enumerate(points,1)):
                    self.done=True
                    return "prior_game_incomplete"
                for j,p in enumerate(points,1):
                    self.observed[i][j]=p.get("score")
            if g.get("last_point") and n:
                self.observed[i][n]=g["last_point"]
                if is_pressure(g["last_point"],self.favorite):
                    self.pressure.add(i)
            if g.get("serve_winner") not in PLAYERS:
                if i==len(games) and is_pressure(msg.get("game_score"),self.favorite):
                    self.pressure.add(i)
                if any(is_pressure(p,self.favorite) for p in self.observed[i].values()):
                    self.pressure.add(i)
                self.last_open[i]=now
                return None
            held = PLAYERS[g["serve_winner"]]==self.favorite
            if any(is_pressure(p,self.favorite) for p in self.observed[i].values()):
                self.pressure.add(i)
            if not held or i in self.pressure:
                self.done=True
                if i not in self.last_open or now-self.last_open[i]>90*SECOND or not at_game_boundary(msg,games,{"game":i}):
                    return "left_censored_game"
                return "hold" if held else "break"
            # Recover a fully observed score sequence before declaring that
            # this earlier favorite game contained zero break points.
            if n<1 or set(self.observed[i])!=set(range(1,n+1)):
                self.done=True
                return "prior_game_incomplete"
            observed_game={"points":[dict(number_point=k,score=self.observed[i][k]) for k in range(1,n+1)]}
            try:
                points=_point_winners(observed_game,self.favorite,self.favorite,False,4)
            except (KeyError,IndexError,TypeError,ValueError):
                self.done=True
                return "prior_game_incomplete"
            if break_points(points,self.favorite):
                self.done=True
                return "left_censored_game"
        return None


def at_reference(client,matches,started):
    priors = {}
    for fixture,m in matches.items():
        if m.get("tour") == "challenger" or m.get("series") == "KXATPCHALLENGERMATCH":
            scheduled = int(timestamp(m["fixture_start"]))
            if scheduled - 600 >= started or not m.get("market_pair"):
                continue
            market,mid,quote_ts=choose_favorite(client,m["market_pair"],scheduled)
            if market:
                priors[fixture]=dict(prior=str(mid),side="yes" if m["ticker"]==market["ticker"] else "no",
                                     quote_ts=quote_ts,anchor=scheduled-600,
                                     market_ticker=market["ticker"],series="KXATPCHALLENGERMATCH")
            continue
        if not m.get("exchange_start") or not m.get("fixture_start"):
            continue
        anchor=int(min(timestamp(m["fixture_start"]),timestamp(m["exchange_start"]))//60)*60-1800
        if not m["open_ts"]<=anchor<started:
            continue
        q=read_quote(client,m["ticker"],anchor)
        if not q:
            continue
        bid,ask,ts=q
        mid=(bid+ask)/2
        priors[fixture]=dict(prior=str(max(mid,1-mid)),side="yes" if mid>=D(".5") else "no",
                             quote_ts=ts, anchor=anchor)
    return priors


def tape_replay(stream,priors):
    start=json.loads(next(stream))
    matches=start["data"]["matches"]
    books=Books()
    watches,actives,accepted,history,orders={},{},{},defaultdict(list),{}
    counts=Counter()
    transitions=[]
    observed=set()
    observed_status=defaultdict(set)
    started= start["utc_ns"]/SECOND
    observation_only={k for k,m in matches.items() if m.get("already_in_play") or (m.get("fixture_start") and timestamp(m["fixture_start"])<=started)}
    # The host's monotonic clock fixes the minute timer schedule.
    opened=start["mono_ns"]
    minute=60*SECOND
    next_bar=opened+minute-(start["utc_ns"]%minute)
    last=opened
    messages=Counter()
    final_kind=None
    candidates={k for k,p in priors.items() if D(p["prior"])>=D(".70")}

    def quote(key,now):
        m=matches[key]; side=priors[key]["side"]
        ask=books.ask(m["ticker"],side,now)
        other=books.ask(m["ticker"],"no" if side=="yes" else "yes",now)
        if ask is None or other is None:
            return None
        return 1-other,ask

    def bar(now):
        for key in candidates & observed:
            order=orders.get(key)
            q=quote(key,now)
            if not q:
                continue
            bid,ask=q
            if key in observation_only:
                continue
            if order:
                if order["status"]=="open":
                    if bid>=D(order["entry"])+D(".08"):
                        why="target"
                    elif bid<=D(order["entry"])-D(".08"):
                        why="stop"
                    elif now-order["entry_ts"]>=1800*SECOND:
                        why="time"
                    else:
                        continue
                    order.update(status="sell_pending",exit_reason=order.get("exit_reason",why),
                                 due=now+2*SECOND,limit=str(bid))
                continue
            live=actives.get(key)
            if not live or live[1]!="Set 1" or now-live[0]>90*SECOND:
                continue
            t=now//SECOND
            seq=history[key]
            seq.append((t,bid,ask))
            while seq and seq[0][0]<t-1800:
                seq.pop(0)
            # Earlier signals in this rolling window were already observed.
            # Evaluate the rule on the suffix after each earlier signal.
            window=seq
            entry=signal(window)
            while entry and entry["ts"]<t:
                window=[b for b in window if b[0]>entry["ts"]]
                entry=signal(window)
            confirmed=accepted.get(key)
            if not entry or not confirmed or not 0<=now-confirmed["ts"]<=600*SECOND:
                continue
            orders[key]=dict(fixture=key,group=confirmed["result"],signal_ts=now,
                             due=now+2*SECOND,limit=str(ask),status="buy_pending")

    def flush(until):
        nonlocal next_bar
        while True:
            due=[o["due"] for o in orders.values() if o["status"] in ("buy_pending","sell_pending")]
            tick=min([next_bar,*due])
            if tick>until:
                return
            if tick==next_bar:
                bar(tick)
                next_bar+=minute
            for key,o in orders.items():
                if o["status"] not in ("buy_pending","sell_pending") or o["due"]!=tick:
                    continue
                q=quote(key,tick)
                if o["status"]=="buy_pending":
                    still_first=actives[key][1]=="Set 1"
                    if still_first and q and q[1]<=D(o["limit"]) and q[1]-q[0]<=D(".04"):
                        o.update(status="open",entry=o["limit"],entry_ts=tick)
                    else:
                        o["status"]="buy_unfilled"
                elif q and q[0]>=D(o["limit"]):
                    entry=D(o["entry"]); price=D(o["limit"])
                    o.update(status="sold",exit=str(price),exit_ts=tick,
                             pnl=str(price-entry-fee(price)-fee(entry)))
                else:
                    o["status"]="open"

    for line in stream:
        row=json.loads(line)
        now,kind,m=row["mono_ns"],row["kind"],row["data"]
        final_kind=kind
        messages[kind]+=1
        if last>now:
            raise ValueError("nonchronological tape")
        flush(now-1)
        last=now
        if kind in ("orderbook_snapshot","orderbook_delta"):
            books.apply(kind,m,now)
        elif kind=="score":
            key=m["fixture"]
            observed.add(key)
            observed_status[key].add(m.get("status") or "unspecified")
            if not m.get("winner") and str(m.get("live"))=="1":
                actives[key]=(now,m.get("status"))
            else:
                actives[key]=(now,None)
            if key in candidates:
                side=priors[key]["side"]
                fav=matches[key]["yes_player"] if side=="yes" else 3-matches[key]["yes_player"]
                watch_id=(key,m.get("source"))
                if watch_id not in watches:
                    watches[watch_id]=ServiceGameWatch(fav)
                watching=watches[watch_id]
                result=watching.process(m,now)
                if result:
                    counts[result]+=1
                    transitions.append(dict(fixture=key,result=result,ts=now,utc_ns=row["utc_ns"]))
                    if result in ("hold","break") and key not in accepted:
                        accepted[key]=transitions[-1]
        flush(now)
    if final_kind!="end":
        raise ValueError("Recording is open or its closing marker is missing")
    return dict(tour=start["data"].get("tour","atp"),
                series=start["data"].get("series","KXATPMATCH"),
                tracked=len(matches),duration_seconds=(last-opened)/SECOND, receipts=dict(messages),
                statuses={k:sorted(v) for k,v in observed_status.items()},
                observation_only_scored=sorted(observed & observation_only),
                strong_favorites_with_new_eligible_start=sorted((candidates & observed)-observation_only),
                fixtures_with_scores=sorted(observed),qualifying_favorites_with_scores=sorted(candidates & observed),
                screened_quotes=priors,transitions=transitions,diagnostics=dict(counts),
                active_qualifying_games=len(accepted),orders=list(orders.values()),book_gaps=books.gaps,
                measured_pnl=str(sum((D(o["pnl"]) for o in orders.values() if o.get("pnl")),D(0))),
                unresolved=sum(o["status"] in ("open","sell_pending") for o in orders.values()))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tapes",nargs="*",type=Path,help="Defaults to completed synchronized local recordings")
    args=parser.parse_args()
    with httpx.Client(timeout=45) as client:
        hist=historical(client)
        save(OUT/"historical.json",dict(rule=RULE,periods=hist,pnl=None))
        tapes=[]
        existing=sorted((ROOT/"data/raw/first-break").glob("*.jsonl"))
        existing+=sorted((ROOT/"data/raw/first-break-challenger").glob("*.jsonl"))
        for path in args.tapes or existing:
            with path.open() as f:
                start=json.loads(next(f))
                priors=at_reference(client,start["data"]["matches"],start["utc_ns"]/SECOND)
                f.seek(0)
                report=tape_replay(f,priors)
            report["tape"]=path.name
            tapes.append(report)
        save(OUT/"forward.json",tapes)
        summary=dict(rule=RULE,periods=hist, historical_trade_pnl=None,
                     historical_limit="Game and point histories lack break-point confirmation receipt timestamps.",
                     historical_cohorts="ATP spread-qualified prematch price-drift study and deterministic one-eighth Challenger rebound study.",
                     live_tapes=len(tapes),independent_scored_fixtures=len({k for t in tapes for k in t["fixtures_with_scores"]}),
                     independent_strong_favorites=len({k for t in tapes for k in t["qualifying_favorites_with_scores"]}),
                     independent_entry_eligible_starts=len({k for t in tapes for k in t["strong_favorites_with_new_eligible_start"]}),
                     forward_exposure_count=sum(t["unresolved"] for t in tapes),
                     observed_orders=sum(len(t["orders"]) for t in tapes),live_signals=[o for t in tapes for o in t["orders"]])
        save(OUT/"summary.json",summary)
        print(json.dumps(summary))


if __name__ == "__main__":
    main()
