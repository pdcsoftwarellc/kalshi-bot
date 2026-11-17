"""Prospective, read-only Challenger live-odds revisions against a local Kalshi book."""
import argparse
import asyncio
from collections import Counter
from decimal import Decimal as D, InvalidOperation
import json
from pathlib import Path
import sys
import time
from urllib.parse import urlsplit

import httpx
import websockets

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from integration.kalshi_client import KalshiClient
from research.api_cost import ROOT, config, fee, save
from research.record_break import TOURS, discover
from research.score_break import Books, SECOND

OUT = ROOT / "data/processed/parallel-round3/live_odds_revision"
RULE = {"revision_min":".03","net_entry_min":".05","max_adjacent_poll_seconds":10,
        "unchanged_score_fields":["event_status","event_game_result","event_serve","event_final_result"],
        "entry_delay_seconds":2,"hold_seconds_after_fill":30,"book_max_age_seconds":5,
        "size":1,"maker_or_taker":"taker both legs; cent-rounded general fee multiplier one",
        "poll_seconds":5, "pair":"Exactly one unsuspended Home and Away To Win price. Both vendor update strings present, at least one advances and neither moves backwards.",
        "exposure":"one open position per fixture, one contract per entry",
        "prices":"limit at signal ask, charge signal ask if at least one contract remains offered at or below it after 2s; cross sized bid at exit"}


POINT_RULE = dict(RULE, unchanged_score_fields=["event_status","event_serve","event_final_result"],
                  transition="Exactly one legal, within-game point with no change of set, server or sets score. Revision must move toward its winner.")


def ordinary_point(before, after):
    """Identify the winner of one legal point strictly inside an advantage game."""
    if before[0]!=after[0] or before[2:]!=after[2:]:
        return None
    def parse(value):
        words = str(value).upper().replace("AD","A").replace(" ","").split("-")
        scale = {"0":0, "15":1, "30":2, "40":3, "A":4}
        if len(words)!=2 or any(w not in scale for w in words):
            return None
        score = tuple(scale[w] for w in words)
        if max(score)==4 and min(score)!=3:
            return None
        return score
    start, finish = parse(before[1]), parse(after[1])
    if start is None or finish is None or start==finish or finish==(0,0):
        return None
    for side in (0,1):
        next_score=list(start)
        next_score[side]+=1
        if max(next_score)>=4 and abs(next_score[0]-next_score[1])>=2:
            continue
        if min(next_score)>=3:
            shift=min(next_score)-3
            next_score=[v-shift for v in next_score]
        if tuple(next_score)==finish:
            return side+1
    return None


def verified_set(row, score):
    """Cross-check fixture ID mapping and the point in recent live score and live odds responses."""
    if not score or abs(score["receipt_utc_ns"]-row["odds_receipt_utc_ns"])>2*SECOND:
        return None
    if str(score.get("event_live"))!="1" or score.get("event_winner"):
        return None
    for key in ("event_status","event_game_result","event_serve","first_player_key","second_player_key"):
        if row.get(key) is None or str(score.get(key))!=str(row[key]):
            return None
    try:
        sn=int(row["event_status"].removeprefix("Set "))
        a,b=(int(x.strip()) for x in score["event_final_result"].split("-"))
        if a+b!=sn-1: return None
        current=[s for s in score["scores"] if int(s["score_set"])==sn]
        if len(current)!=1: return None
        g,h=(int(current[0][k]) for k in ("score_first","score_second"))
        if min(g,h)<0 or min(g,h)>=6: return None
        return (a,b,g,h)
    except (ValueError, TypeError, KeyError, AttributeError):
        return None


async def capture(path, matches, tour, seconds, point_mode=False):
    with path.open("x", buffering=1) as stream:
        def emit(kind,data):
            stream.write(json.dumps({"kind":kind,"utc_ns":time.time_ns(),"mono_ns":time.monotonic_ns(),
                                     "data":data},separators=(",",":"))+"\n")

        emit("start",{"matches":matches,"tour":tour,"seconds":seconds,"point_mode":point_mode,"rule":POINT_RULE if point_mode else RULE})
        async def odds():
            async with httpx.AsyncClient(timeout=8) as client:
                while True:
                    started = time.monotonic()
                    try:
                        async def read(method):
                            resp=await client.get(config.API_TENNIS_REST,params=dict(method=method,
                                APIkey=config.API_TENNIS_KEY,event_type_key=TOURS[tour][1],timezone="UTC"))
                            received=time.time_ns()
                            resp.raise_for_status()
                            payload=resp.json()
                            if str(payload.get("success"))!="1":
                                raise ValueError("Unsuccessful provider response")
                            return received,payload.get("result") or {}
                        scores={}
                        if point_mode:
                            (odds_at,response),(scores_at,score_rows)=await asyncio.gather(
                                read("get_live_odds"),read("get_livescore"))
                            for sr in score_rows if isinstance(score_rows,list) else score_rows.values():
                                ident=str(sr.get("event_key"))
                                if ident in matches:
                                    scores[ident]=dict(receipt_utc_ns=scores_at,**{k:sr.get(k) for k in
                                        ("event_status","event_live","event_winner","event_game_result",
                                         "event_serve","event_final_result","scores","first_player_key","second_player_key")})
                        else:
                            odds_at,response=await read("get_live_odds")
                        items = response.values() if isinstance(response,dict) else response
                        sample={}
                        for m in items:
                            key=str(m["event_key"])
                            if key in matches:
                                sample[key]={name:m.get(name) for name in
                                             ("event_status","event_winner","event_game_result","event_serve",
                                              "event_final_result","first_player_key","second_player_key")}
                                sample[key]["odds_receipt_utc_ns"]=odds_at
                                if point_mode:
                                    sample[key]["score_verification"]=scores.get(key)
                                sample[key]["lines"]=[{k:r.get(k) for k in
                                    ("odd_name","type","value","suspended","upd","handicap")}
                                    for r in (m.get("live_odds") or []) if r.get("odd_name")=="To Win"]
                        emit("odds",sample)
                    except (httpx.HTTPError,ValueError,TypeError,KeyError):
                        emit("odds_error",{})
                    await asyncio.sleep(max(0,5-(time.monotonic()-started)))

        async def books():
            signer=KalshiClient(config.KALSHI_API_KEY,config.KALSHI_PRIVATE_KEY_PATH,demo=False)
            headers=signer._auth_headers("GET",urlsplit(config.KALSHI_WS_PROD).path)
            async with websockets.connect(config.KALSHI_WS_PROD,additional_headers=headers) as ws:
                await ws.send(json.dumps(dict(id=1,cmd="subscribe",params=dict(
                    channels=["orderbook_delta"],market_tickers=[m["ticker"] for m in matches.values()],
                    use_yes_price=False))))
                async for raw in ws:
                    frame=json.loads(raw)
                    kind=frame["type"]
                    if kind=="error":
                        raise RuntimeError("Book subscription rejected")
                    if kind in ("orderbook_snapshot","orderbook_delta"):
                        safe=("market_ticker","side","price_dollars","delta_fp","yes_dollars_fp","no_dollars_fp","clock","ts_ms")
                        emit(kind,dict(seq=frame["seq"],sid=frame["sid"],**{k:v for k,v in frame["msg"].items() if k in safe}))

        tasks=[asyncio.create_task(fn()) for fn in (odds,books)]
        normal=False
        try:
            done,_=await asyncio.wait(tasks,timeout=seconds,return_when=asyncio.FIRST_COMPLETED)
            for task in done:
                task.result()
                raise RuntimeError("Feed ended early")
            normal=True
        finally:
            for task in tasks: task.cancel()
            await asyncio.gather(*tasks,return_exceptions=True)
            emit("end",dict(normal=normal))


def usable(row, yes_player, point_mode=False):
    if not row or row.get("event_winner") or not str(row.get("event_status")).lower().startswith("set "):
        return None
    lines = [r for r in row["lines"] if r.get("type") in ("Home","Away") and r.get("handicap") in (None,"")]
    if len(lines)!=2 or {r["type"] for r in lines}!={"Home","Away"}:
        return None
    if any(r.get("suspended")!="No" or not r.get("upd") for r in lines):
        return None
    ordered = sorted(lines,key=lambda r:r["type"]!="Home")
    try:
        home,away = (D(str(r["value"])) for r in ordered)
    except InvalidOperation:
        return None
    if not all(p.is_finite() and p>1 for p in (home,away)):
        return None
    context = tuple(row.get(k) for k in RULE["unchanged_score_fields"] + ["first_player_key","second_player_key"])
    if not all(context[:3]) or not all(context[-2:]):
        return None
    if point_mode:
        verified=verified_set(row,row.get("score_verification"))
        if verified is None: return None
        context+=verified
    return dict(p=(away if yes_player==1 else home)/(away+home), odds=(home,away),
                updates=tuple(r["upd"] for r in ordered),context=context)


def revision(previous, current, elapsed, *, point_mode=False, yes_player=1):
    if previous is None or current is None or not 0<elapsed<=10*SECOND:
        return None
    if previous["odds"]==current["odds"]:
        return None
    if point_mode:
        winner=ordinary_point(previous["context"],current["context"])
        if winner is None or (winner==yes_player)!=(current["p"]>previous["p"]):
            return None
    elif previous["context"]!=current["context"]:
        return None
    if previous["updates"]==current["updates"] or any(a>b for a,b in zip(previous["updates"],current["updates"])):
        return None
    change=current["p"]-previous["p"]
    return change if abs(change)>=D(".03") else None


def sized_bid(books,ticker,side,now):
    book=books.levels.get(ticker)
    if not book or not 0<=now-books.updated[ticker]<=5*SECOND or not all(book.values()):
        return None
    best=max(book[side]); opposite=max(book["no" if side=="yes" else "yes"])
    if not 0<best<1 or not 0<opposite<1 or best+opposite>1 or book[side][best]<1:
        return None
    return best


def replay(frames):
    books=Books()
    counts=Counter()
    by_fixture={}
    quoted,live,revision_fixtures=set(),set(),set()
    previous={}
    open_by_fixture={}
    ledger,revisions=[],[]
    max_change=D(0)
    max_same_change=D(0)
    first=finish=None
    last_mono=None
    normal=False
    point_mode=False
    cash=D(25)
    def work(until):
        nonlocal cash
        for key,row in list(open_by_fixture.items()):
            if row["state"]=="closed" or row["due_mono"]>until:
                continue
            at=row["due_mono"]
            if row["state"]=="awaiting_entry":
                ask=books.ask(row["ticker"],row["side"],at)
                if ask is None or ask>D(row["limit"]):
                    row["state"]="entry_missed"
                    del open_by_fixture[key]
                    continue
                price=D(row["limit"])
                row.update(state="holding",buy=str(price),entry_fee=str(fee(price)),
                           entry_utc_ns=row["signal_utc_ns"]+2*SECOND,due_mono=at+30*SECOND)
            if row["state"]=="holding" and row["due_mono"]<=until:
                bid=sized_bid(books,row["ticker"],row["side"],row["due_mono"])
                if bid is None:
                    row["state"]="unpriced_exit"
                else:
                    total_fee=D(row["entry_fee"])+fee(bid)
                    net=bid-D(row["buy"])-total_fee
                    row.update(state="closed",sell=str(bid),fees=str(total_fee),net=str(net),
                               exit_utc_ns=row["entry_utc_ns"]+30*SECOND)
                    cash += net
                    del open_by_fixture[key]

    for frame in frames:
        kind,now,utc,m=frame["kind"],frame["mono_ns"],frame["utc_ns"],frame["data"]
        if last_mono is not None and now<last_mono:
            raise ValueError("Nonchronological tape")
        last_mono=now
        work(now-1)
        if kind=="start":
            first=utc
            by_fixture=m["matches"]
            point_mode=m.get("point_mode",False)
        elif kind in ("orderbook_snapshot","orderbook_delta"):
            books.apply(kind,m,now)
            counts[kind]+=1
        elif kind in ("odds","odds_error"):
            counts[kind]+=1
            for key,meta in by_fixture.items():
                raw=m.get(key)
                if raw and str(raw.get("event_status")).lower().startswith("set "): live.add(key)
                if raw and point_mode and usable(raw,meta["yes_player"]) is not None:
                    counts["odds_usable_before_score_verification"]+=1
                item=usable(raw,meta["yes_player"],point_mode)
                prev=previous.pop(key,None)
                position=open_by_fixture.get(key)
                if position and position["state"]=="awaiting_entry":
                    limit=D(position["limit"])
                    fair = ((item["p"] if position["side"]=="yes" else 1-item["p"]) if item else None)
                    if not item or item["context"]!=position["context"] or fair-limit-fee(limit)<D(".05"):
                        position["state"]="intervening_odds_unusable"
                        del open_by_fixture[key]
                if not item: continue
                counts["usable_odds"]+=1; quoted.add(key)
                previous[key]=(now,item)
                if prev is None or not 0<now-prev[0]<=10*SECOND:continue
                counts["adjacent_pairs"]+=1
                if prev[1]["context"]==item["context"]:
                    counts["same_context_pairs"]+=1
                if prev[1]["odds"]!=item["odds"]:
                    counts["changed_prices"]+=1
                    move=abs(item["p"]-prev[1]["p"])
                    max_change=max(max_change,move)
                    counts["any_context_changes_at_least_3pp"]+=int(move>=D(".03"))
                    if prev[1]["context"]==item["context"]:
                        counts["same_context_changed_prices"]+=1
                        max_same_change=max(max_same_change,move)
                if point_mode:
                    winner = ordinary_point(prev[1]["context"],item["context"])
                    if winner is not None:
                        counts["legal_point_transitions"]+=1
                        change = item["p"]-prev[1]["p"]
                        if abs(change)>=D(".03"):
                            counts["legal_point_large_moves"]+=1
                            counts["legal_point_large_toward_winner"]+=int((winner==meta["yes_player"])==(change>0))
                delta=revision(prev[1],item,now-prev[0],point_mode=point_mode,yes_player=meta["yes_player"])
                if delta is None:continue
                side="yes" if delta>0 else "no"
                ticker=meta["ticker"]; fair=item["p"] if side=="yes" else 1-item["p"]
                ask=books.ask(ticker,side,now)
                item_row=dict(fixture=key,ticker=ticker,side=side,signal_utc_ns=utc,change_yes=str(delta),
                              fair=str(fair),ask=None if ask is None else str(ask))
                revisions.append(item_row); revision_fixtures.add(key)
                if ask is None:
                    item_row["decision"]="missing_fresh_depth"
                    continue
                edge=fair-ask-fee(ask); item_row["net_edge"]=str(edge)
                if edge<D(".05"):
                    item_row["decision"]="below_net_edge"
                    continue
                if key in open_by_fixture:
                    item_row["decision"]="existing_attempt"
                    continue
                reserved=sum(D(r["limit"])+fee(D(r["limit"])) for r in open_by_fixture.values())
                if min(cash-reserved,D(25)-reserved)<ask+fee(ask):
                    item_row["decision"]="cash_blocked"
                    continue
                item_row["decision"]="signal"
                order=dict(item_row,state="awaiting_entry",due_mono=now+2*SECOND,
                           limit=str(ask),context=item["context"])
                open_by_fixture[key]=order; ledger.append(order)
        elif kind=="end":
            finish=utc; normal=m["normal"]
        work(now)
    if not normal or books.gaps:
        raise ValueError("Incomplete or discontinuous observation; cannot report fill evidence")
    for order in open_by_fixture.values():
        if order["state"]=="holding":order["state"]="open_at_end"
        if order["state"]=="awaiting_entry":order["state"]="incomplete_entry_window"
    closed=[r for r in ledger if r["state"]=="closed"]
    unresolved=[r for r in ledger if r["state"] in ("open_at_end","unpriced_exit")]
    fees=sum((D(r["fees"]) for r in closed),D(0))
    net=sum((D(r["net"]) for r in closed),D(0))
    return dict(rule=POINT_RULE if point_mode else RULE, started_utc_ns=first,ended_utc_ns=finish, linked=len(by_fixture),
                live_matches=len(live),priced_matches=len(quoted),revision_matches=len(revision_fixtures),
                matches={"live":sorted(live),"priced":sorted(quoted),"revision":sorted(revision_fixtures)},
                coverage=dict(counts),max_any_price_change=str(max_change),max_same_context_price_change=str(max_same_change),
                revisions=revisions,revision_count=len(revisions),qualified=len(ledger),
                entered=sum("buy" in r for r in ledger),closed=len(closed),unresolved=len(unresolved),
                unresolved_cost=str(sum((D(r["buy"])+D(r["entry_fee"]) for r in unresolved),D(0))),
                net_closed_dollars=str(net),fees_closed_dollars=str(fees),ledger=ledger,
                limit="Single vendor, source update timezone and retirement equivalence unverified. Suspensions between polls are unobserved. Book depth is a fill approximation.")


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tour",choices=TOURS,default="challenger")
    parser.add_argument("--seconds",type=int,default=240)
    parser.add_argument("--replay",type=Path)
    parser.add_argument("--point-mode",action="store_true",help="Use the separate single-point odds revision rule")
    args=parser.parse_args()
    if args.replay:
        path=args.replay
    else:
        if not 1<=args.seconds<=600: parser.error("Duration must be 1 to 600 seconds.")
        with httpx.Client(timeout=15) as client: matches=discover(client,args.tour)
        if not matches: raise SystemExit("No uniquely linked open events.")
        directory=OUT.with_name("live_point_revision") if args.point_mode else OUT
        path=directory/f"{time.time_ns()}.jsonl"
        path.parent.mkdir(parents=True,exist_ok=True)
        print(f"Recording {len(matches)} linked listings in {path} for {args.seconds}s",flush=True)
        asyncio.run(capture(path,matches,args.tour,args.seconds,args.point_mode))
    with path.open() as stream:
        report=replay(json.loads(line) for line in stream)
    save(path.with_suffix(".json"),report)
    print(json.dumps({k:v for k,v in report.items() if k not in ("ledger","revisions","matches")},indent=2))


if __name__=="__main__":
    main()
