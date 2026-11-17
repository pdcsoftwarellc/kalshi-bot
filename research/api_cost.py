"""Historical ATP Elo experiment. Only public Kalshi reads and API-Tennis fixtures."""
import argparse
from collections import Counter, defaultdict
from datetime import date, datetime, timedelta, timezone
from decimal import Decimal, ROUND_CEILING
import hashlib
import heapq
import json
import math
from pathlib import Path
import re
import statistics
import sys
import tempfile
import time
import unicodedata

import httpx

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import config

D = Decimal
UTC = timezone.utc
SERIES = "KXATPMATCH"
COLLECTION = ROOT / "data/raw/cost-research-v1"
OUTPUT = ROOT / "data/processed/api-cost-research"
WARMUP = date(2026, 7, 1)
PERIODS = [(date(2026, 10, 1), date(2026, 11, 1)), (date(2026, 11, 1), date(2026, 11, 16))]
STARTING_CASH = D("25")
MIN_EDGE = D(".10")


def timestamp(value):
    return datetime.fromisoformat(value.replace("Z", "+00:00")).timestamp()


def midnight(day):
    return int(datetime.combine(day, datetime.min.time(), tzinfo=UTC).timestamp())


def save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(mode="w", dir=path.parent, delete=False) as temp:
        json.dump(value, temp, indent=2, allow_nan=False)
    Path(temp.name).replace(path)


def fetch(client, path, params, tennis=False):
    # The cache key, errors and output must never contain the API key.
    key = hashlib.sha256(json.dumps([path, params], sort_keys=True).encode()).hexdigest()
    cache = COLLECTION / (key + ".json")
    if cache.exists():
        return json.loads(cache.read_text())
    url = config.API_TENNIS_REST if tennis else config.KALSHI_PROD_BASE + path
    query = dict(params, APIkey=config.API_TENNIS_KEY) if tennis else params
    for attempt in range(5):
        try:
            resp = client.get(url, params=query)
            if resp.status_code == 429 or resp.status_code >= 500:
                time.sleep(2 ** attempt)
                continue
            if resp.status_code != 200:
                raise RuntimeError(f"{path} returned HTTP {resp.status_code}")
            data = resp.json()
            if tennis:
                if str(data.get("success")) != "1" or not isinstance(data.get("result"), list):
                    raise RuntimeError("Historical fixtures unavailable for this window.")
                fields = ("event_key", "event_date", "event_time", "event_first_player", "event_second_player",
                          "first_player_key", "second_player_key", "event_status", "event_winner")
                data = [{k: m.get(k) for k in fields} for m in data["result"]
                        if m.get("event_type_type") == "Atp Singles"]
            save(cache, data)
            return data
        except httpx.RequestError:
            time.sleep(2 ** attempt)
    raise RuntimeError(f"Read failed after retries: {path}")


def history(client, end):
    rows = {}
    day = WARMUP
    while day < end:
        last = min(day + timedelta(days=6), end - timedelta(days=1))
        group = fetch(client, "fixtures", dict(method="get_fixtures", date_start=str(day),
                      date_stop=str(last), timezone="UTC", event_type_key=265), tennis=True)
        rows.update({str(m["event_key"]): m for m in group})
        day = last + timedelta(days=1)
    return sorted(rows.values(), key=lambda m: (m["event_date"], str(m["event_key"])))


def pages(client, path, key, **params):
    cursor = ""
    seen = set()
    while True:
        data = fetch(client, path, dict(params, cursor=cursor) if cursor else params)
        yield from data[key]
        cursor = data.get("cursor")
        if not cursor:
            return
        if cursor in seen:
            raise RuntimeError("Repeated pagination cursor")
        seen.add(cursor)


def tokens(name):
    norm = unicodedata.normalize("NFKD", name or "").lower()
    norm = "".join(c for c in norm if not unicodedata.combining(c))
    return re.findall(r"[a-z]+", norm)


def same_player(short, full):
    a, b = tokens(short), tokens(full)
    if min(len(a), len(b)) < 2 or a[0][0] != b[0][0]:
        return False
    # An abbreviated given name can contain multiple initials. Require the
    # remaining surname tokens as an exact suffix, plus the first initial.
    surname = [w for w in a[1:] if len(w) > 1]
    return bool(surname) and b[-len(surname):] == surname


def link_market(pair, fixtures):
    p, q = (m["yes_sub_title"] for m in pair)
    matches = []
    for f in fixtures:
        a, b = f["event_first_player"], f["event_second_player"]
        if same_player(a, p) and same_player(b, q):
            matches.append((f, str(f["first_player_key"]), str(f["second_player_key"])))
        if same_player(a, q) and same_player(b, p):
            matches.append((f, str(f["second_player_key"]), str(f["first_player_key"])))
    return matches[0] if len(matches) == 1 else None


def elo_probability(diff):
    return 1 / (1 + 10 ** (-diff / 400))


def rating_at(fixtures, day):
    """Use only finished matches whose fixture date was at least seven days earlier."""
    ratings, counts = defaultdict(lambda: 1500.), Counter()
    cutoff = str(day - timedelta(days=7))
    for m in fixtures:
        if m["event_date"] > cutoff:
            break
        if m["event_status"] != "Finished" or m["event_winner"] not in ("First Player", "Second Player"):
            continue
        a, b = str(m["first_player_key"]), str(m["second_player_key"])
        if a == b or not m["first_player_key"] or not m["second_player_key"]:
            continue
        change = 32 * (int(m["event_winner"] == "First Player") - elo_probability(ratings[a] - ratings[b]))
        ratings[a] += change
        ratings[b] -= change
        counts.update([a, b])
    return ratings, counts


def fee(price):
    # Single-contract general quadratic taker fee, rounded up to a whole cent.
    return (D(".07") * price * (1 - price)).quantize(D(".01"), rounding=ROUND_CEILING)


def quotes_at(client, ticker, decision):
    bars = fetch(client, f"/series/{SERIES}/markets/{ticker}/candlesticks",
                 dict(start_ts=decision - 300, end_ts=decision, period_interval=1))["candlesticks"]
    bars = [b for b in bars if decision - 300 < b["end_period_ts"] <= decision]
    if not bars:
        return None
    last = max(bars, key=lambda b: b["end_period_ts"])
    bid = last.get("yes_bid", {}).get("close_dollars")
    ask = last.get("yes_ask", {}).get("close_dollars")
    if not bid or not ask or not 0 < D(bid) <= D(ask) < 1:
        return None
    return D(bid), D(ask), last["end_period_ts"]


def qualifying_print(trades, side, limit, decision):
    for trade in sorted(trades, key=lambda t: timestamp(t["created_time"])):
        ts = timestamp(trade["created_time"])
        if not decision + 2 <= ts < decision + 300 or trade.get("is_block_trade"):
            continue
        if (trade.get("taker_outcome_side") or trade.get("taker_side")) != side:
            continue
        if D(trade["count_fp"]) >= 1 and D(trade[f"{side}_price_dollars"]) <= limit:
            return ts, trade["trade_id"]
    return None


def candidates(client, fixtures, start, end):
    by_day = defaultdict(list)
    for m in fixtures:
        by_day[date.fromisoformat(m["event_date"])].append(m)
    pairs = defaultdict(dict)
    # Buffer close times for matches that end outside the decision window.
    for m in pages(client, "/markets", "markets", series_ticker=SERIES, status="settled", limit=1000,
                   min_close_ts=midnight(start), max_close_ts=midnight(end + timedelta(days=7))):
        pairs[m["event_ticker"]][m["ticker"]] = m

    diagnostics, predictions, orders = Counter(), [], []
    daily_ratings = {}
    for pair_map in pairs.values():
        pair = sorted(pair_map.values(), key=lambda m: m["ticker"])
        if len(pair) != 2:
            diagnostics["incomplete_pairs_in_buffer"] += 1
            continue
        m = pair[0]
        # One canonical ticker. Set decision time using listings alone.
        decision = int(math.ceil(max(timestamp(v["open_time"]) for v in pair) / 60) * 60 + 3600)
        if not midnight(start) <= decision < midnight(end):
            continue
        diagnostics["paired_events"] += 1
        occur = m.get("occurrence_datetime")
        if not occur or decision + 300 >= timestamp(occur):
            diagnostics["outside_prematch_window"] += 1
            continue
        day = datetime.fromtimestamp(timestamp(occur), UTC).date()
        nearby = [f for shift in (-1, 0, 1) for f in by_day[day + timedelta(days=shift)]]
        link = link_market(pair, nearby)
        if not link:
            diagnostics["no_unique_fixture"] += 1
            continue
        f, yes_id, other_id = link
        if not f.get("event_time") or decision + 300 >= timestamp(f["event_date"]+"T"+f["event_time"]+"+00:00"):
            diagnostics["outside_fixture_window"] += 1
            continue
        decision_day = datetime.fromtimestamp(decision, UTC).date()
        if decision_day not in daily_ratings:
            daily_ratings[decision_day] = rating_at(fixtures, decision_day)
        rating, count = daily_ratings[decision_day]
        if min(count[yes_id], count[other_id]) < 5:
            diagnostics["short_player_history"] += 1
            continue
        price = quotes_at(client, m["ticker"], decision)
        if price is None:
            diagnostics["no_recent_two_sided_quote"] += 1
            continue
        bid, ask, quote_ts = price
        p = D(str(elo_probability(rating[yes_id] - rating[other_id])))
        settled = m.get("settlement_value_dollars")
        settled_ts = m.get("settlement_ts")
        if settled is None or not settled_ts or timestamp(settled_ts) <= decision + 300:
            diagnostics["missing_or_early_settlement"] += 1
            continue
        outcome = D(settled)
        if not 0 <= outcome <= 1:
            raise ValueError("Invalid settlement")
        predictions.append(dict(ticker=m["ticker"], fixture=f["event_key"], p=float(p),
                                mid=float((bid+ask)/2), outcome=float(outcome)))
        yes_edge, no_edge = p-ask-fee(ask), (1-p)-(1-bid)-fee(1-bid)
        if max(yes_edge, no_edge) < MIN_EDGE:
            continue
        side, limit = ("yes", ask) if yes_edge >= no_edge else ("no", 1-bid)
        trades = pages(client, "/markets/trades", "trades", ticker=m["ticker"], min_ts=decision+2,
                       max_ts=decision+300, limit=1000)
        hit = qualifying_print(trades, side, limit, decision)
        orders.append(dict(ticker=m["ticker"], fixture=f["event_key"], ts=decision, quote_ts=quote_ts,
                           side=side, limit=str(limit), fee=str(fee(limit)), probability=float(p),
                           settlement_ts=timestamp(settled_ts), payout=str(outcome if side=="yes" else 1-outcome),
                           print_ts=hit[0] if hit else None, trade_id=hit[1] if hit else None))
    return diagnostics, predictions, orders


def ledger(orders, capital):
    cash = capital
    releases = []
    pnl = D(0)
    fees = D(0)
    filled = []
    blocked = 0
    for row in sorted(orders, key=lambda r:(r["ts"],r["ticker"])):
        while releases and releases[0][0] <= row["ts"]:
            cash += heapq.heappop(releases)[1]
        cost = D(row["limit"]) + D(row["fee"])
        if cash < cost:
            blocked += 1
            continue
        cash -= cost
        if row["print_ts"] is None:
            heapq.heappush(releases, (row["ts"] + 300, cost))
            continue
        proceeds = D(row["payout"])
        heapq.heappush(releases, (row["settlement_ts"], proceeds))
        pnl += proceeds-cost
        fees += D(row["fee"])
        filled.append(dict(row, pnl=str(proceeds-cost)))
    return filled, dict(simulated_fills=len(filled), cash_blocked=blocked,
                        fees_dollars=float(fees), trading_pnl_dollars=float(pnl))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--monthly-api-cost", type=Decimal, default=D("80"))
    args = parser.parse_args()
    if not args.monthly_api_cost.is_finite() or args.monthly_api_cost < 0:
        parser.error("Use a finite, nonnegative monthly subscription cost.")
    reports = []
    with httpx.Client(timeout=45) as client:
        cutoff = fetch(client, "/historical/cutoff", {})
        # The experiment uses recent-tier endpoints. Cached requests can be
        # reused after archival; an uncached run must choose the proper tier.
        if timestamp(cutoff["market_settled_ts"]) >= midnight(PERIODS[0][0]) or timestamp(cutoff["trades_created_ts"]) >= midnight(PERIODS[0][0]):
            raise RuntimeError("Dates have moved to the historical tier. Use the saved inputs or implement archive routing.")
        fixtures = history(client, PERIODS[-1][1]+timedelta(days=4))
        print(f"Cached {len(fixtures)} unique ATP fixtures", flush=True)
        for start, end in PERIODS:
            counts, predictions, orders = candidates(client, fixtures, start, end)
            filled, totals = ledger(orders, STARTING_CASH)
            # Allocate this month's bill according to the number of calendar days observed.
            next_month = (start.replace(day=28)+timedelta(days=4)).replace(day=1)
            billed = args.monthly_api_cost * (end-start).days / (next_month-start).days
            report = dict(start=str(start), end_exclusive=str(end), fixture_coverage=dict(counts),
                          priced_predictions=len(predictions), orders=len(orders), **totals,
                          allocated_api_cost=float(billed), pnl_after_api=round(totals["trading_pnl_dollars"]-float(billed), 4))
            report["all_fill_all_win_ceiling_dollars"] = float(sum(
                (1-D(row["limit"])-D(row["fee"]) for row in orders), D(0)))
            report["all_fill_actual_settlement_scenario_dollars"] = float(sum(
                (D(row["payout"])-D(row["limit"])-D(row["fee"]) for row in orders), D(0)))
            binary = [r for r in predictions if r["outcome"] in (0,1)]
            if binary:
                for name in ("p","mid"):
                    report[name+"_brier"] = statistics.mean((r[name]-r["outcome"])**2 for r in binary)
            save(OUTPUT/f"{start}-ledger.json", dict(predictions=predictions, orders=orders, fills=filled))
            reports.append(report)
            print(json.dumps(report), flush=True)
            save(OUTPUT/"summary.json", reports)


if __name__ == "__main__":
    main()
