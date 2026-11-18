"""Analysis for research/favorites.py: part A (taker entries) and part B (resting bids)."""
from bisect import bisect_right
from collections import Counter, defaultdict
import json
import random
from statistics import mean, median

from research.api_cost import save, timestamp
from research.favorites import (BANDS, CUTOFF, DAY, DAYS, HOUR, OUT, RAW, START, END, bars_for, get, hindsight, load_universe,
                                references, sample, series_list, slim_trades, usable)

MAKER_RATE = .0175       # Kalshi fee schedule: maker fee on series flagged quadratic_with_maker_fees
TAKER_RATE = .07


def quote_at(bars, ends, t):
    """Yes bid and ask (cents) in force at t: the close of the last bar ending at or before t."""
    i = bisect_right(ends, t) - 1
    if i < 0:
        return None
    bid, ask = bars[i][3], bars[i][4]
    if bid is None or ask is None or not 1 <= bid < ask <= 99:      # two-sided quote required
        return None
    return bid, ask


def decisions(m, bars, series):
    """Every snapshot at which one side of this market is quoted 90-99c with a two-sided book."""
    bars = sorted(bars)
    ends = [b[0] for b in bars]
    value = float(m["settlement_value_dollars"])
    settle = timestamp(m["settlement_ts"])
    mult = series["fee_multiplier"] if series["fee_multiplier"] is not None else 1
    out = []
    for d in DAYS:
        t = m["ref_ts"] - d * DAY
        if t - HOUR < m["open_ts"] or t + HOUR >= m["close_ts"]:
            continue
        q0, q1 = quote_at(bars, ends, t), quote_at(bars, ends, t + HOUR)
        if not q0:
            continue
        for side in ("yes", "no"):
            ask0 = q0[1] if side == "yes" else 100 - q0[0]
            if not 90 <= ask0 <= 99:
                continue
            band = next(b for b in BANDS if b[0] <= ask0 <= b[1])
            payout = value if side == "yes" else 1 - value
            row = dict(ticker=m["ticker"], event=m["event_ticker"], series=m["series"], category=m["category"],
                       side=side, days=d, band=f"{band[0]}-{band[1]}", t=t, ref_ts=m["ref_ts"], early=m["early"],
                       ask0=ask0, spread=q0[1] - q0[0], payout=payout, fee_multiplier=mult,
                       fee_type=series["fee_type"], close_ts=m["close_ts"],
                       volume_24h=sum(b[5] for b in bars if t - DAY < b[0] <= t),
                       days_held=(settle - t - HOUR) / DAY, price=None, net_cents=None)
            if q1:
                ask1 = q1[1] if side == "yes" else 100 - q1[0]
                if ask1 <= band[1]:              # a limit at the top of the band; a lower ask still fills
                    p = ask1 / 100
                    row.update(price=ask1, fee_cents=100 * TAKER_RATE * mult * p * (1 - p))
                    row["net_cents"] = 100 * payout - ask1 - row["fee_cents"]
            out.append(row)
    return out


def one_per_event(rows):
    """One entry per event and rule: the candidate with the most volume in the prior 24 hours."""
    best = {}
    for r in rows:
        key = (r["event"], r["days"], r["band"])
        if key not in best or (r["volume_24h"], r["ticker"]) > (best[key]["volume_24h"], best[key]["ticker"]):
            best[key] = r
    return list(best.values())


def interval(rows, field="net_cents", draws=2000):
    by = defaultdict(list)
    for r in rows:
        by[r["event"]].append(r[field])
    groups = list(by.values())
    if len(groups) < 10:
        return None
    rng, means = random.Random(11), []
    for _ in range(draws):
        means.append(mean(v for g in rng.choices(groups, k=len(groups)) for v in g))
    means.sort()
    return [round(means[int(.025 * draws)], 2), round(means[int(.975 * draws)], 2)]


def stats(rows, ci=False):
    if not rows:
        return None
    cost = [(r["price"] + r["fee_cents"]) / 100 for r in rows]
    locked = sum(c * max(r["days_held"], 1 / 24) for c, r in zip(cost, rows))
    out = dict(events=len({r["event"] for r in rows}), trades=len(rows),
               full_losses=sum(r["payout"] < .5 for r in rows),
               fractional=sum(0 < r["payout"] < 1 for r in rows),
               win_rate=round(mean(r["payout"] for r in rows), 4),
               breakeven=round(mean(cost), 4), mean_price=round(mean(r["price"] for r in rows), 2),
               net_cents=round(mean(r["net_cents"] for r in rows), 3),
               median_days_held=round(median(r["days_held"] for r in rows), 2),
               annualised_return=round(sum(r["net_cents"] for r in rows) / 100 / locked * 365, 4),
               median_volume_24h=round(median(r["volume_24h"] for r in rows)),
               median_spread=median(r["spread"] for r in rows),
               top_series_share=round(Counter(r["series"] for r in rows).most_common(1)[0][1] / len(rows), 2),
               series=len({r["series"] for r in rows}))
    if ci:
        out["ci95"] = interval(rows)
    return out


def split(rows):
    """Earliest 60% and latest 40% of events by scheduled close."""
    when = {}
    for r in rows:
        when[r["event"]] = min(r["ref_ts"], when.get(r["event"], float("inf")))
    order = sorted(when, key=lambda e: (when[e], e))
    early = set(order[:int(round(.6 * len(order)))])
    return [r for r in rows if r["event"] in early], [r for r in rows if r["event"] not in early]


def rule_table(rows):
    out = {}
    groups = [(f"D{d} {b[0]}-{b[1]}", lambda r, d=d, b=b: r["days"] == d and r["band"] == f"{b[0]}-{b[1]}")
              for d in DAYS for b in BANDS]
    groups += [(f"all D {b[0]}-{b[1]}", lambda r, b=b: r["band"] == f"{b[0]}-{b[1]}") for b in BANDS]
    groups += [(f"D{d} all bands", lambda r, d=d: r["days"] == d) for d in DAYS]
    groups += [("all rules", lambda r: True)]
    for name, keep in groups:
        hits = [r for r in rows if keep(r)]
        if not hits:
            continue
        first, later = split(hits)
        a, b = stats(first), stats(later, ci=True)
        passes = bool(a and b and a["net_cents"] > 0 and b["net_cents"] > 0 and b.get("ci95") and b["ci95"][0] > 0)
        out[name] = dict(early=a, late=b, passes=passes)
    return out


def show(name, table):
    print(f"\n== {name}")
    print(f"{'rule':16}{'part':6}{'ev':>5}{'tr':>5}{'loss':>5}{'win':>8}{'need':>8}{'net c':>8}  {'ci95 (late)':16}"
          f"{'days':>6}{'ann':>8}{'vol24':>8}{'top':>5}")
    for rule, row in table.items():
        for part in ("early", "late"):
            s = row[part]
            if not s:
                continue
            print(f"{rule:16}{part:6}{s['events']:5}{s['trades']:5}{s['full_losses']:5}{s['win_rate']:8.4f}"
                  f"{s['breakeven']:8.4f}{s['net_cents']:+8.2f}  {str(s.get('ci95', '')):16}{s['median_days_held']:6.1f}"
                  f"{100 * s['annualised_return']:+7.1f}%{s['median_volume_24h']:8}{s['top_series_share']:5.2f}"
                  f"{'  PASS' if part == 'late' and row['passes'] else ''}")


def build(client, download=False):
    series = {s["ticker"]: s for s in series_list(client)}
    markets = references(load_universe())
    picked = sample(markets)
    rows, fetched = [], []
    for m in picked:
        bars = bars_for(client, m, download)
        if not bars:
            continue
        fetched.append((m, bars))
        rows += decisions(m, bars, series[m["series"]])
    return series, markets, picked, fetched, rows


def analyze(client):
    series, markets, picked, fetched, rows = build(client)
    report = dict(window=[str(START), str(END)], universe={}, fees={}, tables={})

    # universe counts, with the sampled share scaled up to the listed series
    cats = sorted({m["category"] for m in markets})
    for cat in cats:
        ms = [m for m in markets if m["category"] == cat]
        good = [m for m in ms if usable(m)]
        got = [(m, b) for m, b in fetched if m["category"] == cat]
        traded = sum(any(b[5] > 0 and b[6] is not None and (1 <= b[6] <= 10 or 90 <= (b[7] or 0) <= 99) for b in bars)
                     for m, bars in got)
        quoted = len({r["ticker"] for r in rows if r["category"] == cat})
        report["universe"][cat] = dict(
            series=len({m["series"] for m in ms}), settled_markets=len(ms),
            events=len({m["event_ticker"] for m in ms}),
            early_closed=sum(m["early"] for m in ms),
            volume_1000_plus=sum(float(m["volume_fp"] or 0) >= 1000 for m in ms),
            eligible_markets=len(good), eligible_events=len({m["event_ticker"] for m in good}),
            sampled_with_bars=len(got), sampled_events=len({m["event_ticker"] for m, _ in got}),
            sampled_traded_90_99=traded, sampled_quoted_90_99_at_a_snapshot=quoted,
            estimated_eligible_traded_90_99=round(len(good) * traded / len(got)) if got else 0,
            first_close=min(m["close_time"] for m in ms)[:10], last_close=max(m["close_time"] for m in ms)[:10])
    for cat in cats:
        tick = {m["series"] for m, _ in fetched if m["category"] == cat}
        report["fees"][cat] = dict(Counter(f'{series[t]["fee_type"]} x{series[t]["fee_multiplier"]}' for t in tick))
        report["fees"][cat + " exceptions"] = sorted(
            t for t in tick if series[t]["fee_type"] != "quadratic" or series[t]["fee_multiplier"] != 1)

    taken = one_per_event([r for r in rows if r["net_cents"] is not None])
    variants = {"plain": taken,
                "liquid (24h volume>=100, spread<=3c)": one_per_event(
                    [r for r in rows if r["net_cents"] is not None and r["volume_24h"] >= 100 and r["spread"] <= 3]),
                "scheduled close only": one_per_event(
                    [r for r in rows if r["net_cents"] is not None and not r["early"]])}
    for vname, entries in variants.items():
        report["tables"][vname] = {"overall": rule_table(entries)}
        for cat in cats:
            hits = [r for r in entries if r["category"] == cat]
            if hits:
                report["tables"][vname][cat] = rule_table(hits)

    # how the plain entries spread over series, months and capital
    months = (END - START).days / 30.44
    scale = {}
    for s in {m["series"] for m in picked}:
        allev = {m["event_ticker"] for m in markets if m["series"] == s and usable(m)}
        gotev = {m["event_ticker"] for m, _ in fetched if m["series"] == s}
        scale[s] = len(allev) / len(gotev) if gotev else 0
    report["activity"] = {}
    for cat in cats + ["overall"]:
        hits = [r for r in taken if cat == "overall" or r["category"] == cat]
        events = {r["event"]: r for r in hits}
        if not events:
            continue
        losses = [r for r in hits if r["payout"] < .5]
        report["activity"][cat] = dict(
            sampled_events_with_entry=len(events),
            estimated_events_with_entry_per_month=round(sum(scale[r["series"]] for r in events.values()) / months, 1),
            series_with_entries=len({r["series"] for r in hits}),
            top_series=Counter(r["series"] for r in events.values()).most_common(4),
            months_with_entries=len({int((r["ref_ts"] - 0) // (30.44 * DAY)) for r in hits}),
            median_days_held=round(median(r["days_held"] for r in hits), 1),
            losing_events=len({r["event"] for r in losses}),
            losing_series=Counter(r["series"] for r in {r["event"]: r for r in losses}.values()).most_common(5))
    # supplement: elections timed from the actual close (hindsight; see favorites.hindsight)
    extra = []
    for m in hindsight(markets):
        bars = bars_for(client, m, False)
        if bars:
            extra += decisions(m, bars, series[m["series"]])
    extra_taken = one_per_event([r for r in extra if r["net_cents"] is not None])
    report["elections_actual_close"] = rule_table(extra_taken)
    report["elections_actual_close_series"] = Counter(r["series"] for r in {r["event"]: r for r in extra_taken}.values()).most_common(12)
    save(OUT / "elections-entries.json", extra_taken)
    save(OUT / "summary.json", report)
    save(OUT / "decisions.json", rows)
    save(OUT / "entries.json", taken)

    print("UNIVERSE")
    for cat, u in report["universe"].items():
        print(f"  {cat:24} {u}")
    print("FEES")
    for cat, f in report["fees"].items():
        print(f"  {cat:36} {f}")
    for vname in variants:
        show(vname + " / overall", report["tables"][vname]["overall"])
    for cat in cats:
        if cat in report["tables"]["plain"]:
            show("plain / " + cat, report["tables"]["plain"][cat])
    show("SUPPLEMENT elections, D days before the ACTUAL close (hindsight timing)", report["elections_actual_close"])
    print(report["elections_actual_close_series"])
    print("\nACTIVITY")
    for cat, a in report["activity"].items():
        print(f"  {cat:24} {a}")


def trades_between(client, ticker, start, end):
    """Public prints in (start, end], from whichever tier holds them. Capped at 8 pages per tier."""
    out, truncated = [], False
    tiers = []
    if start < CUTOFF:
        tiers.append("/historical/trades")
    if end >= CUTOFF:
        tiers.append("/markets/trades")
    for path in tiers:
        cursor = ""
        for _ in range(8):
            params = dict(ticker=ticker, min_ts=int(start), max_ts=int(end), limit=1000)
            if cursor:
                params["cursor"] = cursor
            data = get(client, path, params, slim_trades, "trades")
            if not data:
                break
            out += data["trades"]
            cursor = data.get("cursor")
            if not cursor:
                break
        else:
            truncated = True
    return [t for t in out if start < t[0] <= end and t[1] is not None], truncated


def resting(row, trades):
    """A bid one cent under the ask on the favourite side, resting for up to 24 hours."""
    bid = row["ask0"] - 1
    through = touched = False
    for ts, yes_price, count, taker in trades:
        if count <= 0:
            continue
        if row["side"] == "yes" and taker == "no":           # a taker sold yes into the bids
            through |= yes_price < bid
            touched |= yes_price <= bid
        if row["side"] == "no" and taker == "yes":           # a taker bought yes from the no bids
            through |= 100 - yes_price < bid
            touched |= 100 - yes_price <= bid
    p = bid / 100
    maker = MAKER_RATE * row["fee_multiplier"] if row["fee_type"] != "quadratic" else 0
    ask = row["ask0"] / 100
    return dict(bid=bid, filled=through, touched=touched, prints=len(trades),
                maker_net=100 * row["payout"] - bid - 100 * maker * p * (1 - p),
                maker_net_taker_fee=100 * row["payout"] - bid - 100 * TAKER_RATE * row["fee_multiplier"] * p * (1 - p),
                take_now_net=100 * row["payout"] - row["ask0"] - 100 * TAKER_RATE * row["fee_multiplier"] * ask * (1 - ask))


def fill_stats(rows):
    if not rows:
        return None
    hit = [r for r in rows if r["filled"]]
    miss = [r for r in rows if not r["filled"]]
    touch = [r for r in rows if r["touched"]]
    def avg(group, field):
        return round(mean(r[field] for r in group), 3) if group else None
    return dict(orders=len(rows), events=len({r["event"] for r in rows}),
                fill_rate=round(len(hit) / len(rows), 3), touch_rate=round(len(touch) / len(rows), 3),
                no_prints_at_all=round(mean(r["prints"] == 0 for r in rows), 3),
                filled_net_maker_fee=avg(hit, "maker_net"), filled_net_ci95=interval(hit, "maker_net"),
                filled_net_taker_fee_stress=avg(hit, "maker_net_taker_fee"),
                touched_net_maker_fee=avg(touch, "maker_net"),
                filled_full_loss_rate=avg([dict(x=float(r["payout"] < .5)) for r in hit], "x"),
                unfilled_full_loss_rate=avg([dict(x=float(r["payout"] < .5)) for r in miss], "x"),
                filled_losses=sum(r["payout"] < .5 for r in hit), unfilled_losses=sum(r["payout"] < .5 for r in miss),
                filled_if_taken_at_ask=avg(hit, "take_now_net"), unfilled_if_taken_at_ask=avg(miss, "take_now_net"),
                all_taken_at_ask=avg(rows, "take_now_net"),
                net_per_order_placed=round(sum(r["maker_net"] for r in hit) / len(rows), 3),
                median_volume_24h=round(median(r["volume_24h"] for r in rows)))


def fills(client):
    rows = json.loads((OUT / "decisions.json").read_text())
    orders = one_per_event(rows)                 # same event and rule choice as part A, before the one-hour delay
    random.Random(7).shuffle(orders)             # an interrupted download is still a fair sample
    done = []
    from concurrent.futures import ThreadPoolExecutor

    def one(row):
        try:
            trades, truncated = trades_between(client, row["ticker"], row["t"], min(row["t"] + DAY, row["close_ts"]))
        except TimeoutError:
            return None
        return dict(row, truncated=truncated, **resting(row, trades))
    with ThreadPoolExecutor(2) as pool:
        for i, result in enumerate(pool.map(one, orders)):
            if result:
                done.append(result)
            if i % 500 == 0:
                print(i, len(done), flush=True)
    save(OUT / "resting.json", done)
    report = dict(orders_evaluated=len(done), orders_available=len(orders),
                  maker_fee_assumption=f"0 on fee_type quadratic; {MAKER_RATE}*p*(1-p) on series flagged with maker fees",
                  groups={})
    cats = sorted({r["category"] for r in done})
    groups = [("overall", lambda r: True)] + [(c, lambda r, c=c: r["category"] == c) for c in cats]
    groups += [(f"band {b[0]}-{b[1]}", lambda r, b=b: r["band"] == f"{b[0]}-{b[1]}") for b in BANDS]
    groups += [(f"D{d}", lambda r, d=d: r["days"] == d) for d in DAYS]
    groups += [("24h volume >= 1000", lambda r: r["volume_24h"] >= 1000),
               ("spread >= 2c (bid improves the book)", lambda r: r["spread"] >= 2),
               ("spread 1c (joins the queue)", lambda r: r["spread"] == 1),
               ("maker-fee series", lambda r: r["fee_type"] != "quadratic")]
    for name, keep in groups:
        hits = [r for r in done if keep(r)]
        first, later = split(hits)
        report["groups"][name] = dict(all=fill_stats(hits), early=fill_stats(first), late=fill_stats(later))
    save(OUT / "resting-summary.json", report)
    for name, g in report["groups"].items():
        for part in ("all", "early", "late"):
            s = g[part]
            if s:
                print(f"{name:38}{part:6}n {s['orders']:5} fill {s['fill_rate']:.2f} touch {s['touch_rate']:.2f} "
                      f"filled net {s['filled_net_maker_fee']} ci {s['filled_net_ci95']} stress {s['filled_net_taker_fee_stress']} "
                      f"loss f/u {s['filled_losses']}/{s['unfilled_losses']} ({s['filled_full_loss_rate']}/{s['unfilled_full_loss_rate']}) "
                      f"take: filled {s['filled_if_taken_at_ask']} unfilled {s['unfilled_if_taken_at_ask']} all {s['all_taken_at_ask']} "
                      f"per order {s['net_per_order_placed']}")
