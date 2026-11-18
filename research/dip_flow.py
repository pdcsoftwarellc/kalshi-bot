"""Who sells into the tennis dip, how much trades below the bid, and why the rule differs by tour.

The rule (research/favorite_dip.py): a player quoted at 90c or more before play
gets a resting 80c bid; a fill (a trade printing strictly below 80c in a later
one-minute bar) is held to settlement. This script takes every such fill on ATP,
ATP Challenger, WTA, WTA Challenger and ITF men and reads Kalshi's public trade
record from 11 minutes before the fill bar ends to 31 minutes after it.

Per fill, in the favourite's price (100 minus the yes price when the favourite
is the "no" side of the market), with t0 the first print below 80c:
  - who initiated: a taker buying the other side is a seller of the favourite;
  - trade sizes of the seller-initiated flow in the dip window (t0-2min to
    t0+5min); prints of one taker order at one instant are merged into a "sweep";
  - contracts traded strictly below 80c within 5 and 30 minutes (all, and
    seller-initiated only, which is what a resting bid can actually receive);
  - depth (lowest print) and recovery (first print above 80c, at 83c or more,
    and the last price 5 and 30 minutes after t0).
The recovery measures are known only after the fill: they can inform an exit,
never the entry.

From the cached one-minute bars (no network): spread, volume and open interest
per tour for every 90c+ favourite, filled or not.

Holdout is settled before 2026-09-18, recent is after. Splits use holdout
medians, applied unchanged to the recent period.

Downloads are strictly one request at a time, about one per second, with a long
pause on HTTP 429, because a live trader shares the rate limit. Responses are
cached by research.api_cost.fetch, so a rerun resumes. `--minutes` caps the time
spent downloading; `--no-download` analyses what is cached. Tours download in
the order Challenger, WTA, WTA Challenger, ATP, ITF men (`--itf` caps the ITF
sample, newest first). Public market data only; no orders, no authentication.
"""
import argparse
from collections import defaultdict
from datetime import date
import json
from pathlib import Path
from statistics import median
import sys
import time

import httpx
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from research.api_cost import ROOT, fetch, midnight, save, timestamp
from research.dip_archive_tours import cached, paced
from research.favorite_dip import WARMUP, matches, orders
from research.near_end import cents

OUT = ROOT / "data/processed/favorite-dip"
CUTOFF = midnight(date(2026, 9, 18))
TOURS = {"KXATPCHALLENGERMATCH": "Challenger", "KXWTAMATCH": "WTA", "KXWTACHALLENGERMATCH": "WTA Challenger",
         "KXATPMATCH": "ATP", "KXITFMATCH": "ITF men"}
SOURCES = [("KXATPCHALLENGERMATCH", True), ("KXWTAMATCH", True), ("KXATPMATCH", True), ("KXATPCHALLENGERMATCH", False),
           ("KXWTAMATCH", False), ("KXATPMATCH", False), ("KXWTACHALLENGERMATCH", False), ("KXITFMATCH", False)]
LIMIT, BEFORE, AFTER = 80, 660, 1860
RNG = np.random.default_rng(9)


def number(bar, key):
    value = bar.get(key, bar.get(key+"_fp"))      # recent bars carry a "_fp" suffix
    return float(value) if value is not None else 0.0


def bar_study(job, bars):
    """One 90c+ favourite from its cached bars: spreads, volume, and the 80c fill bar if there is one."""
    event, ticker, settle, outcome = job
    rows = [r for r in orders(job, bars) if r["limit"] == LIMIT and r["start"] >= 90]
    if not rows:
        return None
    r = rows[0]
    yes = (r["payout"] == outcome) if outcome in (0.0, 1.0) else None
    if yes is None:
        mids = [(cents(b, "yes_bid", "close")+cents(b, "yes_ask", "close"))/2 for b in bars[:WARMUP]]
        yes = median(mids) > 50
    spreads = [cents(b, "yes_ask", "close")-cents(b, "yes_bid", "close") for b in bars
               if cents(b, "yes_bid", "close") is not None and cents(b, "yes_ask", "close") is not None]
    volume = [number(b, "volume") for b in bars]
    out = dict(event=event, ticker=ticker, series=ticker.split("-")[0], settle=settle, start=r["start"], payout=r["payout"],
               yes=yes, filled=r["strict"], warmup_spread=float(np.mean(spreads[:WARMUP])),
               median_spread=float(np.median(spreads[WARMUP:])) if spreads[WARMUP:] else None,
               window_volume=sum(volume), warmup_volume=sum(volume[:WARMUP]),
               open_interest_max=max(number(b, "open_interest") for b in bars),
               minutes_with_trades=sum(v > 0 for v in volume))
    if r["strict"]:
        end = settle-round(60*r["minutes_before_settlement"])
        i = next(i for i, b in enumerate(bars) if b["end_period_ts"] == end)
        near = [j for j in range(max(WARMUP, i-5), min(len(bars), i+6))]
        before = [cents(bars[j], "yes_ask", "close")-cents(bars[j], "yes_bid", "close") for j in range(max(0, i-5), i)
                  if cents(bars[j], "yes_bid", "close") is not None and cents(bars[j], "yes_ask", "close") is not None]
        out.update(fill_bar_end=end, minutes_before_settlement=r["minutes_before_settlement"],
                   volume_fill_bar=volume[i], volume_near_dip=sum(volume[j] for j in near),
                   spread_before_dip=float(np.mean(before)) if before else None)
    return out


def load_bars():
    """Replay the cached bars only: the client answers 404, so nothing is fetched here."""
    seen, rows = set(), []
    with httpx.Client(transport=httpx.MockTransport(lambda request: httpx.Response(404))) as client:
        for series, archived in SOURCES:      # archived first, so a match listed twice keeps its archive copy
            try:
                for job, bars in matches(client, series, archived):
                    if job[0] in seen:
                        continue
                    row = bar_study(job, bars)
                    if row:
                        seen.add(job[0])
                        row["period"] = "holdout" if row["settle"] < CUTOFF else "recent"
                        rows.append(row)
            except RuntimeError as error:
                print(f"  {series} archived={archived}: listing not fully cached ({error}); using what was read")
    return rows


def trade_pages(client, row, download, deadline):
    """All trades in the window, newest first. None if not cached and downloading is off or out of time."""
    lo, hi = row["fill_bar_end"]-BEFORE, row["fill_bar_end"]+AFTER
    paths = ["/historical/trades", "/markets/trades"] if row["settle"] < CUTOFF else ["/markets/trades", "/historical/trades"]
    for path in paths:
        trades, cursor = [], ""
        while True:
            params = dict(ticker=row["ticker"], min_ts=lo, max_ts=hi, limit=1000)
            if cursor:
                params["cursor"] = cursor
            if not cached(path, params) and (not download or time.monotonic() > deadline):
                return None
            try:
                data = paced(client, path, params)
            except RuntimeError as error:
                if "returned HTTP" not in str(error):
                    raise
                data = dict(trades=[])
            trades += data.get("trades") or []
            cursor = data.get("cursor")
            if not cursor or not data.get("trades"):
                break
        if trades:
            return trades
    return []


def flow(row, raw):
    """Order-flow measures for one fill. Prices are the favourite's, in cents."""
    trades = []
    for t in raw:
        price = float(t.get("yes_price_dollars", t.get("yes_price")))
        price = price*100 if "yes_price_dollars" in t else price
        side = t.get("taker_side") or t.get("taker_outcome_side")
        size = float(t.get("count_fp", t.get("count")))
        fav = round(price) if row["yes"] else 100-round(price)
        sells = (side == "no") if row["yes"] else (side == "yes")
        trades.append((timestamp(t["created_time"]), fav, size, sells))
    trades.sort()
    end = row["fill_bar_end"]
    t0 = next((t[0] for t in trades if t[0] > end-75 and t[1] < LIMIT), None)
    out = dict(trades_in_window=len(trades), found=t0 is not None and t0 <= end+15)
    if not out["found"]:
        return out
    earlier = [t for t in trades if t[0] < t0]
    below_before = sum(t[2] for t in earlier if t[1] < LIMIT)
    five = [t for t in trades if t0 <= t[0] <= t0+300]
    thirty = [t for t in trades if t0 <= t[0] <= t0+1800]
    dip = [t for t in trades if t0-120 <= t[0] <= t0+300]
    sweeps = defaultdict(float)                 # prints of one seller order at one instant
    for t in dip:
        if t[3]:
            sweeps[t[0]] += t[2]
    sizes = sorted(sweeps.values())
    low5 = [t for t in five if t[1] < LIMIT]
    low30 = [t for t in thirty if t[1] < LIMIT]

    def last_price(seconds):
        seen = [t[1] for t in trades if t0 <= t[0] <= t0+seconds]
        return seen[-1]

    def first_at(level):
        return next((t[0]-t0 for t in thirty if t[1] >= level), None)

    volume = sum(t[2] for t in dip)
    gaps = []                                   # check on the taker side: sellers should print below buyers
    for minute in {int(t[0]//60) for t in dip}:
        sold = [t for t in dip if int(t[0]//60) == minute and t[3]]
        bought = [t for t in dip if int(t[0]//60) == minute and not t[3]]
        if sold and bought:
            gaps.append(sum(t[1]*t[2] for t in sold)/sum(t[2] for t in sold)-sum(t[1]*t[2] for t in bought)/sum(t[2] for t in bought))
    out.update(
        seller_minus_buyer_price=float(np.mean(gaps)) if gaps else None,
        t0=t0, seconds_into_bar=t0-(end-60), volume_below_before_t0=below_before,
        price_10m_before=earlier[0][1] if earlier and earlier[0][0] <= t0-420 else None,
        high_2m_before=max((t[1] for t in earlier if t[0] >= t0-120), default=None),
        dip_trades=len(dip), dip_volume=volume,
        seller_share_dip=sum(t[2] for t in dip if t[3])/volume,
        seller_share_below=sum(t[2] for t in low5 if t[3])/sum(t[2] for t in low5),
        seller_share_before=(sum(t[2] for t in earlier if t[3])/sum(t[2] for t in earlier)) if earlier else None,
        seller_sweeps=len(sizes), seller_sweep_median=float(np.median(sizes)) if sizes else None,
        seller_sweep_max=sizes[-1] if sizes else None,
        seller_top_share=sizes[-1]/sum(sizes) if sizes else None, seller_volume_dip=sum(sizes),
        first_print_size=sum(t[2] for t in trades if t[0] == t0 and t[1] < LIMIT),
        below_5m=sum(t[2] for t in low5), below_5m_sold=sum(t[2] for t in low5 if t[3]),
        at_or_below_5m=sum(t[2] for t in five if t[1] <= LIMIT),
        below_30m=sum(t[2] for t in low30), below_30m_sold=sum(t[2] for t in low30 if t[3]),
        prints_below_5m=len(low5), low_5m=min(t[1] for t in five), low_30m=min(t[1] for t in thirty),
        vwap_below_5m=sum(t[1]*t[2] for t in low5)/sum(t[2] for t in low5),
        seconds_to_above_80=first_at(81), seconds_to_83=first_at(83),
        price_after_5m=last_price(300), price_after_30m=last_price(1800),
        seconds_below_last=max((t[0]-t0 for t in low30), default=0.0))
    return out


def ci(values, draws=4000):
    values = np.asarray(values, float)
    if len(values) < 10:
        return None
    means = values[RNG.integers(0, len(values), (draws, len(values)))].mean(axis=1)
    return [round(float(v), 1) for v in np.percentile(means, [2.5, 97.5])]


def result(rows):
    if not rows:
        return dict(fills=0)
    nets = [100*r["payout"]-LIMIT for r in rows]
    return dict(fills=len(rows), win_rate=float(np.mean([r["payout"] for r in rows])), net_cents=float(np.mean(nets)), ci95=ci(nets))


def text(s):
    if not s["fills"]:
        return "fills   0"
    return f'fills {s["fills"]:3} win {s["win_rate"]:.3f} net {s["net_cents"]:+6.1f}c ci {s["ci95"]}'


def quart(values, digits=0):
    values = [v for v in values if v is not None]
    if not values:
        return None
    return [round(float(v), digits) for v in np.percentile(values, [25, 50, 75])]


SPLITS = (("seller_sweeps", "number of seller sweeps in the dip window"),
          ("seller_sweep_median", "median seller sweep size"),
          ("seller_sweep_max", "largest seller sweep"),
          ("seller_top_share", "largest sweep as a share of seller volume"),
          ("seller_share_dip", "seller-initiated share of dip-window volume"),
          ("seller_share_below", "seller-initiated share of volume below 80c"),
          ("dip_volume", "contracts traded in the dip window"),
          ("below_5m", "contracts below 80c within 5 minutes"),
          ("low_5m", "lowest print within 5 minutes"),
          ("high_2m_before", "highest print in the 2 minutes before t0 (speed of the drop)"),
          ("price_after_5m", "last price 5 minutes after t0 (known only afterwards)"),
          ("seconds_to_83", "seconds to a print at 83c+ (known only afterwards)"))


def analyse(fills, favourites):
    report = {}
    groups = {"all tours": list(TOURS.values()), "Challenger+WTA": ["Challenger", "WTA"], "ATP+ITF men": ["ATP", "ITF men"]}
    groups.update({t: [t] for t in TOURS.values()})
    ok = [f for f in fills if f.get("found")]

    print("\nBASELINE (fills with a trade record)")
    for name, tours in groups.items():
        for period in ("holdout", "recent"):
            part = [f for f in ok if f["tour"] in tours and f["period"] == period]
            report[f"baseline | {name} | {period}"] = result(part)
            print(f"  {name:15} {period:7} {text(result(part))}")

    print("\n(A) FLOW PER FILL, quartiles [25%, median, 75%]")
    keys = ("dip_trades", "dip_volume", "seller_share_dip", "seller_share_below", "seller_sweeps", "seller_sweep_median",
            "seller_sweep_max", "seller_top_share", "first_print_size", "below_5m", "below_5m_sold", "below_30m", "below_30m_sold",
            "low_5m", "low_30m", "seconds_to_above_80", "seconds_to_83", "price_after_5m", "price_after_30m", "high_2m_before")
    for name, tours in groups.items():
        for period in ("holdout", "recent"):
            part = [f for f in ok if f["tour"] in tours and f["period"] == period]
            if not part:
                continue
            cell = {k: quart([f[k] for f in part], 2 if "share" in k else 0) for k in keys}
            cell["fills"] = len(part)
            cell["share_back_above_80_within_60s"] = float(np.mean([f["seconds_to_above_80"] is not None and f["seconds_to_above_80"] <= 60 for f in part]))
            cell["share_83_within_5m"] = float(np.mean([f["seconds_to_83"] is not None and f["seconds_to_83"] <= 300 for f in part]))
            cell["share_never_83_in_30m"] = float(np.mean([f["seconds_to_83"] is None for f in part]))
            cell["mean_seller_share_dip"] = float(np.mean([f["seller_share_dip"] for f in part]))
            cell["volume_weighted_seller_share_dip"] = sum(f["seller_share_dip"]*f["dip_volume"] for f in part)/sum(f["dip_volume"] for f in part)
            report[f"A | flow | {name} | {period}"] = cell
            print(f'  {name:15} {period:7} n {len(part):3} trades {cell["dip_trades"]} volume {cell["dip_volume"]} seller share {cell["seller_share_dip"]} '
                  f'(below 80: {cell["seller_share_below"]})')
            print(f'  {"":23} seller sweeps {cell["seller_sweeps"]} median size {cell["seller_sweep_median"]} largest {cell["seller_sweep_max"]} top share {cell["seller_top_share"]}')
            print(f'  {"":23} below 80c in 5m {cell["below_5m"]} (seller-initiated {cell["below_5m_sold"]}) in 30m {cell["below_30m"]} low 5m {cell["low_5m"]} low 30m {cell["low_30m"]}')
            print(f'  {"":23} seconds to >80 {cell["seconds_to_above_80"]} to 83+ {cell["seconds_to_83"]} | back above 80 in 60s {cell["share_back_above_80_within_60s"]:.0%} '
                  f'83+ in 5m {cell["share_83_within_5m"]:.0%} never 83+ in 30m {cell["share_never_83_in_30m"]:.0%} | price +5m {cell["price_after_5m"]} +30m {cell["price_after_30m"]}')

    print("\n(A) OUTCOME BY FLOW, split at the holdout median of the same group")
    for name in ("all tours", "Challenger+WTA", "ATP+ITF men"):
        tours = groups[name]
        for key, label in SPLITS:
            hold = [f[key] for f in ok if f["tour"] in tours and f["period"] == "holdout" and f[key] is not None]
            if len(hold) < 20:
                continue
            cut = float(np.median(hold))
            print(f"  {name} | {label}: cut {cut:.2f}")
            for period in ("holdout", "recent"):
                part = [f for f in ok if f["tour"] in tours and f["period"] == period]
                low = [f for f in part if f[key] is not None and f[key] <= cut]
                high = [f for f in part if f[key] is not None and f[key] > cut]
                none = [f for f in part if f[key] is None]
                cell = dict(cut=cut, low=result(low), high=result(high))
                line = f"    {period:7} <=cut {text(cell['low'])} | >cut {text(cell['high'])}"
                if none:
                    cell["never"] = result(none)
                    line += f" | never {text(cell['never'])}"
                have = [(f[key], f["payout"]) for f in part if f[key] is not None]
                if len(have) >= 20 and len({h[1] for h in have}) > 1:
                    x = np.argsort(np.argsort([h[0] for h in have])).astype(float)
                    cell["rank_correlation"] = float(np.corrcoef(x, [h[1] for h in have])[0, 1])
                    cell["noise_band"] = 2/len(have)**.5
                    line += f' | rho {cell["rank_correlation"]:+.2f} (+/-{cell["noise_band"]:.2f})'
                report[f"A | outcome | {name} | {key} | {period}"] = cell
                print(line)

    print("\n(A) RECOVERY AND OUTCOME (known only after the fill)")
    bands = (("83c+ within 1 minute", lambda f: f["seconds_to_83"] is not None and f["seconds_to_83"] <= 60),
             ("83c+ in 1 to 5 minutes", lambda f: f["seconds_to_83"] is not None and 60 < f["seconds_to_83"] <= 300),
             ("83c+ in 5 to 30 minutes", lambda f: f["seconds_to_83"] is not None and f["seconds_to_83"] > 300),
             ("never 83c+ in 30 minutes", lambda f: f["seconds_to_83"] is None),
             ("price 5 minutes later >= 80c", lambda f: f["price_after_5m"] >= 80),
             ("price 5 minutes later < 80c", lambda f: f["price_after_5m"] < 80),
             ("price 5 minutes later < 70c", lambda f: f["price_after_5m"] < 70))
    for name in ("all tours", "Challenger+WTA", "ATP+ITF men"):
        for label, test in bands:
            line = f"  {name:15} {label:29}"
            for period in ("holdout", "recent"):
                part = [f for f in ok if f["tour"] in groups[name] and f["period"] == period and test(f)]
                report[f"A | recovery | {name} | {label} | {period}"] = result(part)
                line += f" | {period} {text(result(part))}"
            print(line)

    print("\n(B) BARS PER TOUR, every 90c+ favourite (filled or not), quartiles [25%, median, 75%]")
    for tour in TOURS.values():
        for period in ("holdout", "recent"):
            part = [r for r in favourites if r["tour"] == tour and r["period"] == period]
            hit = [r for r in part if r["filled"]]
            if not part:
                continue
            cell = dict(favourites=len(part), fills=len(hit), fill_rate=len(hit)/len(part),
                        warmup_spread=quart([r["warmup_spread"] for r in part], 1), in_window_spread=quart([r["median_spread"] for r in part], 1),
                        window_volume=quart([r["window_volume"] for r in part]), open_interest_max=quart([r["open_interest_max"] for r in part]),
                        minutes_with_trades=quart([r["minutes_with_trades"] for r in part]),
                        volume_near_dip=quart([r["volume_near_dip"] for r in hit]), volume_fill_bar=quart([r["volume_fill_bar"] for r in hit]),
                        spread_before_dip=quart([r["spread_before_dip"] for r in hit], 1),
                        fill_minutes_before_settlement=quart([r["minutes_before_settlement"] for r in hit]),
                        win_rate_all_favourites=float(np.mean([r["payout"] for r in part])), mean_start=float(np.mean([r["start"] for r in part])))
            report[f"B | bars | {tour} | {period}"] = cell
            print(f'  {tour:15} {period:7} favourites {len(part):4} fills {len(hit):3} ({cell["fill_rate"]:.0%}) favourite wins {cell["win_rate_all_favourites"]:.3f} '
                  f'pre-match {cell["mean_start"]:.1f}c | spread pre-match {cell["warmup_spread"]} in window {cell["in_window_spread"]} before dip {cell["spread_before_dip"]}')
            print(f'  {"":23} window volume {cell["window_volume"]} peak open interest {cell["open_interest_max"]} minutes with trades {cell["minutes_with_trades"]} '
                  f'| volume +/-5 bars of dip {cell["volume_near_dip"]} fill bar {cell["volume_fill_bar"]} | fill minutes before settlement {cell["fill_minutes_before_settlement"]}')

    print("\n(C) CAPACITY: contracts strictly below 80c within 5 minutes of the first print below 80c")
    for tour in TOURS.values():
        for period in ("holdout", "recent", "both"):
            part = [f for f in ok if f["tour"] == tour and period in (f["period"], "both")]
            if not part:
                continue
            cell = dict(fills=len(part), below_5m=quart([f["below_5m"] for f in part]), below_5m_seller_initiated=quart([f["below_5m_sold"] for f in part]),
                        first_print=quart([f["first_print_size"] for f in part]), at_or_below_5m=quart([f["at_or_below_5m"] for f in part]),
                        share_under_10_contracts=float(np.mean([f["below_5m_sold"] < 10 for f in part])),
                        share_under_100_contracts=float(np.mean([f["below_5m_sold"] < 100 for f in part])))
            report[f"C | {tour} | {period}"] = cell
            print(f'  {tour:15} {period:7} n {len(part):3} all prints {cell["below_5m"]} seller-initiated {cell["below_5m_seller_initiated"]} '
                  f'first print {cell["first_print"]} at or below 80c {cell["at_or_below_5m"]} | seller-initiated under 10: {cell["share_under_10_contracts"]:.0%}, under 100: {cell["share_under_100_contracts"]:.0%}')

    print("\n(C) RETURN PER CONTRACT FOR A BID OF Q CONTRACTS (filled for min(Q, seller-initiated volume below 80c in 30 minutes))")
    for name in ("Challenger+WTA", "Challenger", "WTA", "WTA Challenger", "ATP+ITF men"):
        for period in ("holdout", "recent"):
            part = [f for f in ok if f["tour"] in groups[name] and f["period"] == period]
            if not part:
                continue
            line, cell = f"  {name:15} {period:7} n {len(part):3}", {}
            for q in (1, 10, 100, 1000, 10000):
                got = np.array([min(q, f["below_30m_sold"]) for f in part])
                net = np.array([100*f["payout"]-LIMIT for f in part])
                pick = RNG.integers(0, len(part), (4000, len(part)))
                band = np.percentile((got[pick]*net[pick]).sum(axis=1)/np.maximum(got[pick].sum(axis=1), 1e-9), [2.5, 97.5])
                cell[q] = dict(mean_contracts=float(got.mean()), share_fully_filled=float((got >= q).mean()),
                               net_cents_per_contract=float((got*net).sum()/got.sum()), ci95=[round(float(v), 1) for v in band])
                line += f' | Q={q}: {cell[q]["net_cents_per_contract"]:+.1f}c {cell[q]["ci95"]} full {cell[q]["share_fully_filled"]:.0%}'
            report[f"C | sized | {name} | {period}"] = cell
            print(line)

    print("\nCHECK: seller-initiated prints minus buyer-initiated prints in the same minute (should be negative)")
    gaps = [f["seller_minus_buyer_price"] for f in ok if f["seller_minus_buyer_price"] is not None]
    report["check | seller minus buyer price"] = dict(fills=len(gaps), mean=float(np.mean(gaps)), share_negative=float(np.mean([g < 0 for g in gaps])))
    print(f'  {len(gaps)} fills, mean {np.mean(gaps):+.2f}c, negative in {np.mean([g < 0 for g in gaps]):.0%}')
    return report


def calibration():
    """Does the favourite win more often than its pre-match price, by tour and price band? Cached order rows only."""
    print("\n(B) PRE-MATCH CALIBRATION: favourite's win rate against its pre-match midpoint (before any dip)")
    seen, rows = set(), []
    for name in ("tennis-archived-orders.json", "tennis-orders.json", "wta-challenger-orders.json", "itf-men-orders.json"):
        for r in json.loads((OUT/name).read_text()):
            if r["event"] not in seen and r["event"].split("-")[0] in TOURS:
                seen.add(r["event"])
                rows.append(r)
    report = {}
    for tour in TOURS.values():
        for lo, hi in ((70, 80), (80, 90), (90, 101)):
            line = f"  {tour:15} {lo}-{min(hi, 100)}c"
            for period in ("holdout", "recent", "both"):
                part = [r for r in rows if TOURS[r["event"].split("-")[0]] == tour and lo <= r["start"] < hi
                        and period in ("holdout" if r["settle"] < CUTOFF else "recent", "both")]
                if len(part) < 10:
                    continue
                edge = [100*r["payout"]-r["start"] for r in part]
                cell = dict(matches=len(part), mean_price=float(np.mean([r["start"] for r in part])),
                            win_rate=float(np.mean([r["payout"] for r in part])), edge_cents=float(np.mean(edge)), ci95=ci(edge))
                report[f"B | calibration | {tour} | {lo}-{hi} | {period}"] = cell
                line += f' | {period} n {len(part):4} price {cell["mean_price"]:.1f} wins {cell["win_rate"]:.3f} edge {cell["edge_cents"]:+.1f}c {cell["ci95"]}'
            print(line)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--minutes", type=float, default=9, help="stop new downloads after this long")
    parser.add_argument("--no-download", action="store_true")
    parser.add_argument("--itf", type=int, default=150, help="ITF men fills to download, newest first (0 for all)")
    args = parser.parse_args()
    favourites = load_bars()
    for r in favourites:
        r["tour"] = TOURS[r["series"]]
    order = list(TOURS.values())
    todo = sorted((r for r in favourites if r["filled"]), key=lambda r: (order.index(r["tour"]), -r["settle"]))
    itf = [r for r in todo if r["tour"] == "ITF men"]
    if args.itf:
        skip = {r["event"] for r in itf[args.itf:]}
        todo = [r for r in todo if r["event"] not in skip]
    print(f"{len(favourites)} favourites at 90c+, {sum(r['filled'] for r in favourites)} filled at 80c, {len(todo)} to read trades for")
    deadline, fills, missing = time.monotonic()+60*args.minutes, [], defaultdict(int)
    with httpx.Client(timeout=40, headers={"User-Agent": "research-script"}) as client:
        for n, row in enumerate(todo):
            raw = trade_pages(client, row, not args.no_download, deadline)
            if raw is None:
                missing[row["tour"]] += 1
                continue
            fills.append(dict(row, **flow(row, raw)))
            if n % 50 == 0:
                print(f"  {n}/{len(todo)} {row['tour']}", flush=True)
    print("not downloaded yet:", dict(missing) or "none")
    lost = [f for f in fills if not f["found"]]
    print(f"{len(fills)} fills with a trade response; {len(lost)} have no print below 80c in the fill minute "
          f"({sum(f['trades_in_window'] == 0 for f in lost)} with no trades at all)")
    save(OUT/"flow-rows.json", fills)
    report = analyse(fills, favourites)
    report.update(calibration())
    report["coverage"] = dict(fills_wanted=len(todo), fills_read=len(fills), not_downloaded=dict(missing),
                              no_print_below_80_in_fill_minute=[f["event"] for f in lost])
    save(OUT/"flow.json", report)


if __name__ == "__main__":
    main()
