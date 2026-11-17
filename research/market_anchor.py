"""Paired ATP replay. Use a prematch market prior plus the native model's in-play odds change."""
from collections import Counter
from datetime import date, timedelta
from decimal import Decimal
import hashlib
import json
from pathlib import Path
from statistics import median
import sys

import httpx
import numpy as np
from scipy.special import expit, logit

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from research.api_cost import ROOT, PERIODS, save
from research.price_drift import read_quote
from research.set_frequency import MODEL, collect_forecasts, evaluate_records, initial_state
from tennis.engine import PredictionEngine

OUT = ROOT / "data/processed/native-atp/market-anchor"


def anchored_probability(prematch_market, prematch_model, current_model):
    """Use model odds change as a multiplier on the original market odds."""
    if any(not 0 < p < 1 for p in (prematch_market, prematch_model, current_model)):
        raise ValueError("Expected interior probabilities in the same player orientation")
    return float(expit(logit(prematch_market) + logit(current_model) - logit(prematch_model)))


def opening_yes(engine, fixture, yes_player):
    seed = int.from_bytes(hashlib.sha256(str(fixture["event_key"]).encode()).digest()[:8], "big")
    forecasts = []
    for server in (0, 1):
        state = initial_state(fixture, server)
        engine._rng = np.random.default_rng(seed)
        engine.new_match(state)
        forecast = engine.predict(state)
        forecasts.append(forecast.win_prob_p1 if yes_player == 0 else forecast.win_prob_p2)
    return sum(forecasts) / 2


def prematch_reference(client, row):
    cutoff = row["prematch_cutoff"]
    if cutoff is None or not row["opened"] <= cutoff < row["confirmed"]:
        return None
    q = read_quote(client, row["ticker"], cutoff)
    if q is None:
        return None
    bid, ask, ts = q
    if not row["opened"] <= ts <= cutoff or cutoff-ts > 300 or ask-bid > Decimal(".04"):
        return None
    return float((bid+ask)/2), ts


def fixture_metadata(start, end):
    fixtures = {}
    day = start-timedelta(days=2)
    while day <= end:
        for folder in ("native-atp", "set-frequency-atp"):
            path = ROOT / "data/raw" / folder / f"{day}.json"
            if path.exists():
                fixtures.update({str(row["event_key"]):row for row in json.loads(path.read_text())})
                break
        day += timedelta(days=1)
    return fixtures


def run_period(client, engine, start, end, settings):
    fixtures = fixture_metadata(start, end)
    rows, coverage = collect_forecasts(client, engine, start, end, settings)
    priors = {}
    native, anchored = [], []
    for row in rows:
        key = row["fixture"], row["ticker"], row["prematch_cutoff"]
        if key not in priors:
            quote = prematch_reference(client, row)
            if quote is None:
                priors[key] = None
            else:
                opening = opening_yes(engine, fixtures[row["fixture"]], row["yes_player"])
                priors[key] = quote[0], quote[1], opening
        if priors[key] is None:
            continue
        mid, ts, opening = priors[key]
        candidate = anchored_probability(mid, opening, row["model"])
        native.append(dict(row))
        anchored.append(dict(row, model=candidate, original_model=row["model"], opening_model=opening,
                             prematch_mid=mid, prematch_quote_ts=ts))

    stats = dict(start=str(start), end_exclusive=str(end), priced_checkpoints=len(rows),
                 excluded_no_anchor=len(rows)-len(native), full_coverage=dict(coverage))
    if native:
        stats["median_seconds_from_set_book_close_to_decision"] = median(r["t"]-r["set_book_close"] for r in native)
    for key, arm in (("native", native), ("market_anchored", anchored)):
        stats[key] = evaluate_records(client, arm, start, end, settings, Counter())
    save(OUT/f"{start}-ledger.json", dict(summary=stats, native=native, market_anchored=anchored))
    print(start, {key:(stats[key]["paired_forecasts"],stats[key]["qualified"],stats[key]["fills"],
                       stats[key]["net_trading_dollars"],stats[key].get("model_brier"))
                  for key in ("native","market_anchored")}, flush=True)
    return stats


def main():
    model = PredictionEngine.load(str(MODEL), n_mc_samples=200)
    settings = json.loads((ROOT/"data/processed/native-atp/set-frequency/summary.json").read_text())["settings"]
    if (settings["per_signal"],settings["latency_ms"],settings["taker_multiplier"],settings["fee_precision"]) != (1,2000,1,"0.01"):
        raise ValueError("Use the original one-contract set-frequency execution assumptions")
    periods = [*PERIODS,(date(2026,9,22),date(2026,10,1))]
    metadata = json.loads((MODEL/"fit_metadata.json").read_text())
    if metadata["calibration_through"] >= min(str(s) for s,e in periods):
        raise ValueError("Model fitting overlaps an evaluation period")
    with httpx.Client(timeout=45) as client:
        reports = [run_period(client,model,start,end,settings) for start,end in periods]
    save(OUT/"summary.json",dict(settings=settings, rule="Prematch midpoint odds times native odds change; native confidence gate.",periods=reports))


if __name__ == "__main__":
    main()
