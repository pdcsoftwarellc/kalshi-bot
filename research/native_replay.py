"""Replay timestamped native feed predictions, top-of-book depth and published settlements."""
import argparse
import asyncio
from collections import Counter
from decimal import Decimal, ROUND_CEILING, ROUND_FLOOR
import json
from pathlib import Path
import sys

import httpx

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import config
from integration.match_tracker import MatchTracker
from integration.tennis_feed import TennisFeed
from research.api_cost import ROOT, save, timestamp
from research.score_break import Books
from scripts.replay_trades import replay, entry_fee
from integration.edge import compute_edge
from tennis.engine import PredictionEngine


def quote(book, ticker, ms):
    prices = dict(kind="quote", ts_ms=ms, ticker=ticker, bid_size=0, ask_size=0)
    if not book:
        return prices
    for side, prefix in (("yes", "bid"), ("no", "ask")):
        if not book[side]:
            continue
        bid = max(book[side])
        price = 100 * (bid if side=="yes" else 1-bid)
        cents = int(price.to_integral_value(rounding=ROUND_FLOOR if side=="yes" else ROUND_CEILING))
        if 0 < cents < 100:
            prices["yes_"+prefix] = cents
            prices[prefix+"_size"] = int(book[side][bid])
    if prices["ask_size"] and prices["bid_size"] and prices["yes_bid"] > prices["yes_ask"]:
        prices["ask_size"] = prices["bid_size"] = 0
    return prices


def recorded_fixture(msg, cached):
    if "native" in msg:
        return msg["native"]
    # Older enriched Challenger recordings include just set one. All player
    # IDs come from pre-existing fixture metadata in the provider namespace.
    games = msg.get("games") or []
    if (msg.get("status")!="Set 1" or not cached or not games or not msg.get("set_scores")
            or any("points" not in g or int(g["number_game"])>=13 for g in games)):
        return None
    last = games[-1]
    server = last.get("player_served")
    if last.get("serve_winner"):
        server = {"First Player":"Second Player","Second Player":"First Player"}.get(server)
    return dict(cached, event_status="Set 1",event_type_type="Challenger Men Singles",
                event_serve=server,event_game_result=msg.get("game_score"),scores=msg["set_scores"],
                pointbypoint=[dict(g,set_number="Set 1") for g in games])


async def convert(tape, model, fixtures):
    tracker = MatchTracker(model._model, model._calibrator, model._elo, model._player_stats,
                           model._h2h, resolver=None, n_mc_samples=200, rng_seed=42)
    books = Books()
    events, forecasts = [], []
    counts = Counter()
    decoded_in_receipt = 0
    async def observe(point):
        nonlocal decoded_in_receipt
        decoded_in_receipt += 1
        result = tracker.process_point(point)
        if result is None:
            return
        meta = matches[point.match_id]
        prob = result.win_prob_p1 if meta["yes_player"]==1 else result.win_prob_p2
        pred = dict(kind="prediction", ticker=meta["ticker"], ts_ms=ms, model_prob=prob,
                    confidence=result.confidence, points_observed=result.points_observed)
        events.append(pred)
        top = quote(books.levels.get(meta["ticker"]), meta["ticker"], ms)
        point_winner = point.point_server if point.server_won_last_point else 1 - point.point_server
        forecasts.append(dict(pred, fixture=point.match_id, bid=top.get("yes_bid"), ask=top.get("yes_ask"),
                              point_winner_yes=(point_winner == meta["yes_player"]-1),
                              decoded_in_receipt=decoded_in_receipt, score_source=m.get("source","websocket"),
                              bid_size=top["bid_size"], ask_size=top["ask_size"],
                              book_age_ms=(mono-books.updated[meta["ticker"]])/1e6 if meta["ticker"] in books.updated else None,
                              games=[point.games_p1,point.games_p2],
                              points=[point.points_p1,point.points_p2]))
    feed = TennisFeed("",observe)
    with tape.open() as f:
        start=json.loads(next(f))
        matches=start["data"]["matches"]
        for line in f:
            r=json.loads(line)
            ms=(start["utc_ns"]+r["mono_ns"]-start["mono_ns"])//1_000_000
            mono=r["mono_ns"]
            k,m=r["kind"],r["data"]
            if k in ("orderbook_snapshot","orderbook_delta"):
                books.apply(k,m,r["mono_ns"])
                events.append(quote(books.levels.get(m["market_ticker"]),m["market_ticker"],ms))
            elif k=="score":
                counts["score_receipts"]+=1
                fixture=m["fixture"]
                data=recorded_fixture(m,fixtures.get(fixture))
                if data:
                    decoded_in_receipt = 0
                    counts["usable_receipts"]+=1
                    await feed._process_match(data)
    if books.gaps:
        raise ValueError("Book sequence gap: exclude this tape from economic replay")
    return events, forecasts, counts


def market_scorecard(forecasts, settlements, settings):
    """Score the first fresh two-sided forecast per fixture, selected before looking at outcomes."""
    first = {}
    screens = Counter()
    seen = set()
    for row in sorted(forecasts, key=lambda r: r["ts_ms"]):
        key = row["fixture"], row["ticker"], row["ts_ms"], row["points_observed"]
        if key in seen:
            continue
        seen.add(key)
        bid, ask, age = row["bid"], row["ask"], row["book_age_ms"]
        if (age is None or not 0 <= age <= settings["max_age_ms"]
                or not row["bid_size"] or not row["ask_size"]
                or bid is None or ask is None or not 0 < bid <= ask < 100):
            screens["missing_fresh_two_sided_depth"] += 1
            continue
        if row["points_observed"] < settings["min_points"]:
            screens["too_few_points"] += 1
            continue
        p = row["model_prob"]
        first.setdefault(row["fixture"], dict(fixture=row["fixture"], ticker=row["ticker"],
            ts_ms=row["ts_ms"], points_observed=row["points_observed"],
            model_yes=p, market_mid=(bid+ask)/200))
        side, edge, cents = compute_edge(p, ask, bid)
        fee = float(entry_fee(cents, 1, 0, Decimal(str(settings["taker_multiplier"])),
                              Decimal(settings["fee_precision"]))) if side else 0
        if row["confidence"] < settings["min_confidence"]:
            screens["low_confidence"] += 1
        elif side is None or edge < settings["min_edge"]:
            screens["no_or_insufficient_gross_edge"] += 1
        elif edge - fee < settings["net_edge_min"]:
            screens["insufficient_net_edge"] += 1
        else:
            screens["passed_forecast_screen_before_sizing"] += 1

    scored = []
    for row in first.values():
        settlement = settlements.get(row["ticker"])
        if settlement is None:
            row["status"] = "awaiting_settlement_check_or_payout"
        elif settlement["ts_ms"] <= row["ts_ms"] or settlement["yes_cents"] not in (0,100):
            row["status"] = "ineligible_settlement"
        else:
            row.update(status="scored", outcome=settlement["yes_cents"] / 100)
            scored.append(row)
    def brier(key):
        return sum((r[key]-r["outcome"])**2 for r in scored) / len(scored) if scored else None
    return dict(rule="First forecast per fixture with sufficient observed points and a fresh two-sided whole-contract book.",
                forecasts=len(seen), eligible_matches=len(first), scored_matches=len(scored),
                model_brier=brier("model_yes"), market_brier=brier("market_mid"),
                forecast_gates=dict(screens), observations=list(first.values()))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("tapes",nargs="+",type=Path)
    parser.add_argument("--artifacts-dir",type=Path,default=ROOT/"data/artifacts/native-atp")
    parser.add_argument("--output-dir",type=Path,default=ROOT/"data/processed/native-atp/tape-replay")
    parser.add_argument("--settle",action="store_true",help="Read exchange settlements for forecasted tickers")
    args=parser.parse_args()
    model=PredictionEngine.load(str(args.artifacts_dir))
    if model._calibrator.player_id_source!="api-tennis":
        parser.error("These captures use API-Tennis player IDs. Supply artifacts trained on that source.")
    fixtures={}
    for p in (ROOT/"data/raw/challenger-rebound").glob("fixtures-*.json"):
        fixtures.update({str(m["event_key"]):m for m in json.loads(p.read_text())})
    reports=[]
    all_forecasts, settlements = [], {}
    for tape in args.tapes:
        events,forecasts,counts=asyncio.run(convert(tape,model,fixtures))
        # Published payouts enter only after predictions and paper decisions.
        # Read uncached settlement status, since active markets can settle later.
        if args.settle and forecasts:
            with httpx.Client(timeout=12) as client:
                for ticker in sorted({p["ticker"] for p in forecasts}):
                    resp=client.get(config.KALSHI_PROD_BASE+"/markets/"+ticker)
                    resp.raise_for_status()
                    m=resp.json()["market"]
                    payout=m.get("settlement_value_dollars")
                    if m.get("settlement_ts") and payout is not None:
                        cents=Decimal(payout)*100
                        if cents != int(cents):
                            raise ValueError("Subcent settlement requires a higher-precision ledger")
                        events.append(dict(kind="settlement",ticker=ticker,
                            ts_ms=int(timestamp(m["settlement_ts"])*1000),yes_cents=int(cents)))
        all_forecasts.extend(forecasts)
        settlements.update({e["ticker"]: e for e in events if e["kind"] == "settlement"})
        events.sort(key=lambda row:row["ts_ms"])
        settings=dict(bankroll=config.MAX_TOTAL_EXPOSURE,per_signal=config.MAX_CONTRACTS_PER_SIGNAL,
                      per_market=config.MAX_CONTRACTS_PER_MATCH,min_points=config.MIN_POINTS_OBSERVED,
                      min_confidence=config.MIN_CONFIDENCE,min_edge=config.MIN_EDGE_PCT,
                      net_edge_min=config.NET_EDGE_MIN_DOLLARS,latency_ms=2000,max_age_ms=5000,
                      taker_multiplier=1,fee_precision="0.01")
        summary=replay(events,**settings)
        report=dict(tape=tape.name,counts=counts,forecasted_matches=len({p["fixture"] for p in forecasts}),
                    thresholds=settings,summary=summary,forecasts=forecasts)
        save(args.output_dir/(tape.stem+".json"),report)
        reports.append({k:v for k,v in report.items() if k!="forecasts"})
    save(args.output_dir/"market-scorecard.json", market_scorecard(all_forecasts, settlements, settings))
    save(args.output_dir/"summary.json",reports)
    print(json.dumps(reports,indent=2))


if __name__=="__main__":
    main()
