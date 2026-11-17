#!/usr/bin/env python3
"""Replay timestamped research predictions against sized, whole-cent top-of-book snapshots.

Input is JSON Lines in receipt order. See the research section in README.md.
The replay holds purchased contracts to an explicit exchange settlement.
"""
import argparse
from collections import Counter, defaultdict
from dataclasses import dataclass
from decimal import Decimal, ROUND_CEILING
import json
import math
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from integration.edge import compute_edge, evaluate_signal


def integer(row, name, lower, upper=None):
    value = row[name]
    if type(value) is not int or value < lower or (upper is not None and value > upper):
        raise ValueError(f"Invalid {name}")
    return value


def probability(row, key):
    value = row[key]
    if type(value) not in (float, int) or not 0 <= value <= 1:
        raise ValueError(f"Invalid {key}")
    return value


@dataclass
class Holding:
    yes: int = 0
    no: int = 0
    cost: Decimal = Decimal(0)
    fees: Decimal = Decimal(0)


def entry_fee(price_cents, quantity, fixed_fee, multiplier, precision):
    if multiplier is None:
        return fixed_fee * quantity
    price = Decimal(price_cents) / 100
    raw = Decimal(".07") * multiplier * quantity * price * (1 - price)
    return raw.quantize(precision, rounding=ROUND_CEILING)


def replay(events, *, bankroll=500., fee=None, taker_multiplier=None, fee_precision="0.0001", latency_ms=250, max_age_ms=2000,
           per_signal=20, per_market=50, min_edge=.05, net_edge_min=.10,
           min_confidence=.60, min_points=10):
    if (fee is None) == (taker_multiplier is None):
        raise ValueError("Choose exactly one fixed fee or taker multiplier.")
    if fee is not None and not 0 <= fee < 1:
        raise ValueError("Invalid fee")
    if taker_multiplier is not None and (not math.isfinite(taker_multiplier) or taker_multiplier < 0):
        raise ValueError("Invalid taker multiplier")
    if fee_precision not in ("0.01", "0.0001"):
        raise ValueError("Invalid fee precision")
    precision = Decimal(fee_precision)
    multiplier = None if taker_multiplier is None else Decimal(str(taker_multiplier))
    if not math.isfinite(bankroll) or bankroll <= 0:
        raise ValueError("Invalid bankroll or fee")
    if latency_ms < 0 or max_age_ms <= latency_ms or min(per_signal, per_market) < 1:
        raise ValueError("Invalid timing or position limits")
    if not 0 <= min_edge < 1 or not 0 <= min_confidence <= 1 or min_points < 0:
        raise ValueError("Invalid signal thresholds")
    if not math.isfinite(net_edge_min) or net_edge_min < 0:
        raise ValueError("Invalid net edge minimum")
    cash = Decimal(str(bankroll))
    fee = Decimal(str(fee or 0))
    settled_pnl = settled_cost = settled_fees = Decimal(0)
    quotes, pending = {}, {}
    holdings = defaultdict(Holding)
    settled = set()
    realized_peak = realized_drawdown = Decimal(0)
    stats = Counter()
    last_ms = -1

    for i, row in enumerate(events, 1):
        try:
            now = integer(row, "ts_ms", 0)
            if now < last_ms:
                raise ValueError("Events must be ordered by receipt time")
            last_ms = now
            ticker = row["ticker"]
            if not isinstance(ticker, str) or not ticker.strip() or ticker in settled:
                raise ValueError("Invalid ticker or event after settlement")
            kind = row["kind"]
            if ticker in pending and now - pending[ticker][0] > max_age_ms:
                del pending[ticker]
                stats["expired_orders"] += 1

            if kind == "settlement":
                payout = Decimal(integer(row, "yes_cents", 0, 100)) / 100
                held = holdings.pop(ticker, Holding())
                proceeds = payout * held.yes + (1 - payout) * held.no
                cash += proceeds
                settled_pnl += proceeds - held.cost
                realized_peak = max(realized_peak, settled_pnl)
                realized_drawdown = max(realized_drawdown, realized_peak - settled_pnl)
                settled_cost += held.cost
                settled_fees += held.fees
                stats["settled_positions"] += int(held.yes + held.no > 0)
                if pending.pop(ticker, None):
                    stats["unfilled_orders"] += 1
                settled.add(ticker)
                quotes.pop(ticker, None)
            elif kind == "quote":
                # Size 0 represents an unavailable side. YES bid depth is the
                # depth available to a NO buyer at the complementary price.
                ask_size = integer(row, "ask_size", 0)
                bid_size = integer(row, "bid_size", 0)
                ask = integer(row, "yes_ask", 1, 99) if ask_size else None
                bid = integer(row, "yes_bid", 1, 99) if bid_size else None
                if ask is not None and bid is not None and ask < bid:
                    raise ValueError("Crossed quote")
                quotes[ticker] = (now, ask, bid)
                if ticker not in pending:
                    continue
                decision_ms, signal = pending[ticker]
                if now == decision_ms or now - decision_ms < latency_ms:
                    continue
                del pending[ticker]
                yes = signal.side == "yes"
                price = ask if yes else (100 - bid if bid is not None else None)
                depth = ask_size if yes else bid_size
                if price is None or price > signal.price_cents:
                    stats["unfilled_orders"] += 1
                    continue
                unit_price = Decimal(price) / 100
                qty = min(depth, signal.contracts, int(cash / unit_price))
                # Binary search the affordable quantity with actual fill-level rounding.
                low, high = 0, qty
                while low < high:
                    mid = (low + high + 1) // 2
                    if mid * unit_price + entry_fee(price, mid, fee, multiplier, precision) <= cash:
                        low = mid
                    else:
                        high = mid - 1
                qty = low
                if not qty:
                    stats["unfilled_orders"] += 1
                    continue
                stats["filled_orders"] += 1
                stats["filled_contracts"] += qty
                stats["partial_fills"] += int(qty < signal.contracts)
                held = holdings[ticker]
                if yes:
                    held.yes += qty
                else:
                    held.no += qty
                fees_paid = entry_fee(price, qty, fee, multiplier, precision)
                cost = qty * unit_price + fees_paid
                held.cost += cost
                held.fees += fees_paid
                cash -= cost
            elif kind == "prediction":
                p = probability(row, "model_prob")
                confidence = probability(row, "confidence")
                observed = integer(row, "points_observed", 0)
                stats["predictions"] += 1
                if ticker not in quotes or now - quotes[ticker][0] > max_age_ms:
                    stats["missing_or_stale_quotes"] += 1
                    continue
                if ticker in pending:
                    continue
                _, ask, bid = quotes[ticker]
                held = holdings.get(ticker, Holding())
                side, _, decision_price = compute_edge(p, ask, bid)
                if side is None:
                    continue
                estimated_fee = entry_fee(decision_price, 1, fee, multiplier, precision)
                signal = evaluate_signal(
                    ticker, p, confidence, observed, ask, bid, float(cash),
                    min_edge=min_edge, min_confidence=min_confidence, min_points=min_points,
                    net_edge_min=net_edge_min, max_contracts=min(per_signal, per_market - held.yes - held.no),
                    fee_per_contract=float(estimated_fee),
                )
                if signal:
                    pending[ticker] = now, signal
                    stats["orders"] += 1
            else:
                raise ValueError("Unknown event kind")
        except (ValueError, TypeError, KeyError) as exc:
            raise ValueError(f"Event {i}: {exc}") from exc
    open_cost = sum((h.cost for h in holdings.values()), Decimal(0))
    return dict(stats, pending_orders=len(pending), unsettled_positions=len(holdings),
                unsettled_cost_dollars=float(open_cost), cash_dollars=float(cash),
                settled_cost_dollars=float(settled_cost), settled_fee_dollars=float(settled_fees),
                settled_pnl_dollars=float(settled_pnl),
                settlement_drawdown_dollars=float(realized_drawdown),
                settled_roi=float(settled_pnl / settled_cost) if settled_cost else None)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("events", type=Path)
    parser.add_argument("--bankroll", type=float, default=500.)
    fees = parser.add_mutually_exclusive_group(required=True)
    fees.add_argument("--fee", type=float, help="Fixed entry fee in dollars per contract")
    fees.add_argument("--taker-multiplier", type=float, help="Multiplier in the general quadratic fee schedule")
    parser.add_argument("--fee-precision", choices=["0.01", "0.0001"], default="0.0001",
                        help="Quadratic fee rounding for this account and historical period")
    parser.add_argument("--latency-ms", type=int, default=250)
    parser.add_argument("--max-age-ms", type=int, default=2000)
    parser.add_argument("--per-signal", type=int, default=20)
    parser.add_argument("--per-market", type=int, default=50)
    parser.add_argument("--min-edge", type=float, default=.05)
    parser.add_argument("--net-edge-min", type=float, default=.10)
    parser.add_argument("--min-confidence", type=float, default=.60)
    parser.add_argument("--min-points", type=int, default=10)
    options = vars(parser.parse_args())
    file = options.pop("events")
    try:
        with file.open() as stream:
            summary = replay((json.loads(line) for line in stream if line.strip()), **options)
    except (ValueError, OSError) as exc:
        parser.exit(1, f"Replay failed: {exc}\n")
    print(json.dumps({"assumptions": options, "results": summary}, indent=2))


if __name__ == "__main__":
    main()
