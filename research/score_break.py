"""Replay a synchronized capture and reconcile one-contract paper entries."""
import argparse
from collections import Counter, defaultdict
from decimal import Decimal
import json
import math
from pathlib import Path
import sys

import httpx

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from research.api_cost import fee, quotes_at, save, timestamp
from research.break_study import OUT, first_break, probability
from research.forward_odds import exchange

D = Decimal
SECOND = 10**9


class Books:
    def __init__(self):
        self.levels = {}
        self.updated = {}
        self.sequence = {}
        self.gaps = 0

    def apply(self, kind, msg, now):
        sid, seq, ticker = msg["sid"], msg["seq"], msg["market_ticker"]
        if sid in self.sequence and seq != self.sequence[sid]+1:
            self.gaps += 1
            self.levels.clear()
            self.updated.clear()
        self.sequence[sid] = seq
        if kind == "orderbook_snapshot":
            self.levels[ticker] = {side:{D(p):D(n) for p,n in msg.get(side+"_dollars_fp",[])}
                                   for side in ("yes","no")}
        elif ticker in self.levels:
            side = self.levels[ticker][msg["side"]]
            price = D(msg["price_dollars"])
            qty = side.get(price,D(0)) + D(msg["delta_fp"])
            if qty < 0:
                del self.levels[ticker]
                return
            if qty == 0:
                side.pop(price,None)
            else:
                side[price] = qty
        self.updated[ticker] = now

    def ask(self,ticker,side,now):
        book = self.levels.get(ticker)
        if not book or not 0 <= now-self.updated[ticker] <= 5*SECOND:
            return None
        opposite = book["no" if side=="yes" else "yes"]
        if not opposite or not book[side]:
            return None
        bid = max(opposite)
        ask = 1-bid
        if not 0 < ask < 1 or max(book[side]) > ask or opposite[bid] < 1:
            return None
        return ask


def prematch(client, matches, capture_ts):
    priors = {}
    for key, m in matches.items():
        if not m.get("exchange_start"):
            continue
        t = math.ceil(m["open_ts"]/60)*60+3600
        earliest = min(timestamp(m["exchange_start"]),timestamp(m["fixture_start"]))
        late = int(earliest//60)*60-1800
        for label,anchor in (("early_listing",t),("before_start",late)):
            if not m["open_ts"] <= anchor < min(capture_ts,earliest-300):
                continue
            quote = quotes_at(client,m["ticker"],anchor)
            if quote:
                bid,ask,seen = quote
                priors[key] = dict(mid=str((ask+bid)/2), quote_ts=seen,decision_ts=anchor,anchor=label)
                break
    return priors


def at_game_boundary(msg, games, event):
    """Confirm the live set totals agree and the next point has not been played."""
    if any(g["n_points"] for g in games[event["game"]:]):
        return False
    live_score = str(msg.get("game_score") or "").replace(" ", "")
    if live_score not in ("0-0", "-"):
        return False
    first = next((s for s in (msg.get("set_scores") or []) if str(s.get("score_set"))=="1"),None)
    if not first:
        return False
    after = games[event["game"]-1]["score"].replace(" ", "")
    return f"{first.get('score_first')}-{first.get('score_second')}" == after


def replay(rows, priors, models):
    books = Books()
    previous = {}
    seen = set()
    pending = {}
    checks = []
    counts = Counter()
    sources = Counter()
    reached = {}
    matches = None
    last = -1

    def price_due(now):
        for key, check in list(pending.items()):
            due = check["due_mono_ns"]
            if due > now:
                continue
            del pending[key]
            ask = books.ask(check["ticker"],check["side"],due)
            if ask is None:
                check["decision"] = "no_fresh_sized_ask"
                continue
            net = D(str(check["estimated_prob"]))-ask-fee(ask)
            check.update(ask=str(ask), fee=str(fee(ask)), net_edge=str(net),
                         decision="paper_fill" if net >= D(".05") else "edge_below_5c")

    for r in rows:
        kind,now,m = r["kind"],r["mono_ns"],r["data"]
        if now < last:
            raise ValueError("Nonchronological receipt tape")
        last = now
        price_due(now-1)
        counts[kind] += 1
        if kind=="start":
            matches = m["matches"]
        elif kind in ("orderbook_snapshot","orderbook_delta"):
            books.apply(kind,m,now)
        elif kind=="score":
            key = m["fixture"]
            games = sorted(m["games"], key=lambda g:int(g["number_game"]))
            sources[m.get("source","websocket")] += 1
            reached[key] = max(reached.get(key,0), len(games))
            if key in pending:
                check = pending[key]
                if (m["status"]!="Set 1" or m["winner"] or first_break(games)!=check["event"]
                    or not at_game_boundary(m,games,check["event"])):
                    check["decision"] = "score_changed_during_delay"
                    del pending[key]
            prev = previous.get((key,m.get("source","websocket")))
            previous[key,m.get("source","websocket")] = now,games
            event = first_break(games)
            if event and key not in seen:
                seen.add(key)
                info = matches[key]
                check = dict(fixture=key,ticker=info["ticker"],event=event, receipt_ns=r["utc_ns"],
                             source=m.get("source","websocket"),decision="catchup_or_late_break")
                checks.append(check)
                n = event["game"]
                # Require a recent observation of this game in progress from
                # the same feed. A reconnect's history cannot create an entry.
                if (m["status"]=="Set 1" and not m["winner"] and prev and now-prev[0] <= 90*SECOND
                    and len(prev[1])==n and prev[1][-1]["serve_winner"] is None
                    and at_game_boundary(m,games,event)):
                    prior = priors.get(key)
                    if prior is None:
                        check["decision"] = "no_prematch_prior"
                    else:
                        yes = info["yes_player"]==event["broken"]
                        p = D(prior["mid"])
                        p = p if yes else 1-p
                        check.update(prior=str(p),anchor=prior["anchor"],side="yes" if yes else "no")
                        if p < D(".6"):
                            check["decision"] = "broken_player_below_60pct"
                        elif prior["anchor"] not in models:
                            check["decision"] = "no_model_for_anchor"
                        else:
                            check.update(estimated_prob=probability(float(p),event,models[prior["anchor"]]),
                                         due_mono_ns=now+2*SECOND,decision="capture_ended_in_delay")
                            pending[key] = check
        price_due(now)
    return dict(messages=dict(counts),score_sources=dict(sources),max_first_set_game=reached,
                sequence_gaps=books.gaps,checks=checks,priors=priors)


def settle(client, checks):
    for c in checks:
        if c["decision"] != "paper_fill":
            continue
        m = exchange(client, "/markets/"+c["ticker"])["market"]
        payout = m.get("settlement_value_dollars")
        if m.get("status") not in ("finalized","settled") or payout is None:
            c["pnl_dollars"] = None
            continue
        val = D(payout)
        if not 0<=val<=1:
            raise ValueError("Unexpected settlement payout")
        proceeds = val if c["side"]=="yes" else 1-val
        c["pnl_dollars"] = str(proceeds-D(c["ask"])-D(c["fee"]))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("tape",type=Path)
    args=parser.parse_args()
    fit = json.loads((OUT/"fit.json").read_text())
    with args.tape.open() as f, httpx.Client(timeout=20) as client:
        start = json.loads(next(f))
        priors = prematch(client,start["data"]["matches"],start["utc_ns"]/SECOND)
        from itertools import chain
        models = {"early_listing":fit["coefficients"]}
        if "before_start" in fit:
            models["before_start"] = fit["before_start"]["coefficients"]
        report = replay(chain([start],(json.loads(line) for line in f)),priors,models)
        settle(client,report["checks"])
    path = OUT/(args.tape.stem+"-paper.json")
    save(path,report)
    print(json.dumps(dict(report=str(path), **{k:v for k,v in report.items() if k!="priors"}),indent=2))


if __name__ == "__main__":
    main()
